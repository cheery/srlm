import jax
import jax.numpy as jnp
import haiku as hk


class AttnResidual(hk.Module):
    """Inter-block attention residual from ref/attn_residuals.py.

    Replaces the plain `y = layer_out + y` residual in S5Stack.  Instead of a
    simple sum, each token position attends over all completed block states plus
    the current partial sum to produce the input for the next sub-layer.

    The attention uses a single learned query vector (shared across positions),
    so it is O(N_blocks * B * L * D) — much cheaper than full MHA.

    Args:
        d_model: feature dimension D
        use_adaln: if True, normalize blocks with AdaLN conditioned on t instead
                   of plain RMSNorm.  Requires t to be passed to __call__.
    """
    def __init__(self, d_model, use_adaln=False, name=None):
        super().__init__(name=name)
        self.d_model = d_model
        self.use_adaln = use_adaln
        if use_adaln:
            self.norm = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False, name="norm")
            self.adaln_proj = hk.Linear(2 * d_model, name="adaln_proj")

    def __call__(self, blocks, partial_block, t=None):
        # blocks: list of (B, L, D) completed block states
        # partial_block: (B, L, D) current intra-block accumulation
        # t: (B, D_t) timestep embedding — required when use_adaln=True
        # Returns: (B, L, D) attended combination to use as next layer input

        if len(blocks) == 0:
            # No history yet — nothing to attend over
            return partial_block

        V = jnp.stack(blocks + [partial_block], axis=0)  # (N+1, B, L, D)

        if self.use_adaln:
            # AdaLN: condition the normalization on the noise level t.
            # Apply per-block (AdaLN expects (B, L, D)); stack result for scoring.
            scale, shift = jnp.split(self.adaln_proj(t), 2, axis=-1)  # (B, D) each
            def adaln(x):  # x: (B, L, D)
                return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]
            K = jnp.stack([adaln(b) for b in blocks + [partial_block]], axis=0)
        else:
            # RMSNorm along feature axis, with learned scale
            rms = jnp.sqrt(jnp.mean(V ** 2, axis=-1, keepdims=True) + 1e-6)
            norm_scale = hk.get_parameter(
                "norm_scale", shape=(self.d_model,), init=hk.initializers.Constant(1.0)
            )
            K = V / rms * norm_scale  # (N+1, B, L, D)

        # Learned query vector: dot with each block rep → scalar score per (block, b, pos)
        proj = hk.get_parameter(
            "proj", shape=(self.d_model,), init=hk.initializers.TruncatedNormal(stddev=0.02)
        )
        logits = jnp.einsum("d, nbld -> nbl", proj, K)   # (N+1, B, L)
        weights = jax.nn.softmax(logits, axis=0)           # softmax over blocks
        return jnp.einsum("nbl, nbld -> bld", weights, V)  # (B, L, D)

class Attention(hk.Module):
    """Drop-in replacement for jax.vmap(S5Dual) using full (bidirectional) multi-head
    attention with learned positional encoding capped at context_length.

    Receives (B, L, H) directly (vmap is applied to S5Dual in __init__, not here).
    Internally transposes to hk.MultiHeadAttention's [T, B, C] convention.
    Attention is computed in float32 regardless of mixed-precision policy.
    """
    def __init__(self, d_model, context_length, n_heads, name=None):
        super().__init__(name=name)
        assert d_model % n_heads == 0, (
            f"n_heads ({n_heads}) must divide d_model ({d_model})"
        )
        self.d_model = d_model
        self.context_length = context_length
        self.n_heads = n_heads

    def __call__(self, x):
        # x: (B, L, H) — called directly (vmap is applied to S5Dual in __init__, not here)
        L = x.shape[1]

        pos_emb = hk.get_parameter(
            "pos_emb",
            shape=(self.context_length, self.d_model),
            init=hk.initializers.TruncatedNormal(stddev=0.02),
        )
        x = x + pos_emb[:L]  # (B, L, H) + (L, H) broadcasts correctly

        # hk.MultiHeadAttention expects [T, B, C]; upcast to float32 for softmax precision.
        dtype = x.dtype
        x_tbf = x.transpose(1, 0, 2).astype(jnp.float32)  # (L, B, H)

        attn = hk.MultiHeadAttention(
            num_heads=self.n_heads,
            key_size=self.d_model // self.n_heads,
            model_size=self.d_model,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_avg", "uniform"),
        )

        out = attn(x_tbf, x_tbf, x_tbf)  # (L, B, H)

        return jax.nn.gelu(out.transpose(1, 0, 2).astype(dtype))  # (B, L, H)
