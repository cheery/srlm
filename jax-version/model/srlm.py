from dataclasses import dataclass
import jax
import jax.numpy as jnp
import haiku as hk
import math
from .s5 import S5Dual
from .attn import Attention, AttnResidual

@dataclass
class SRLMConfig:
    vocab_size : int
    d_model : int
    d_state : int
    n_priors : int
    n_posteriors : int
    d_frequency_embedding : int = 256
    N : int = 2
    T : int = 4
    dropout : float = 0.5
    # Attention-based alternative to S5Dual.
    # When True, S5Dual is replaced by multi-head attention with learned positional
    # encoding capped at context_length.  n_heads must divide d_model.
    use_attention : bool = False
    context_length : int = 512
    n_heads : int = 8
    # Replace plain residual addition in S5Stack with inter-block attention residuals.
    # Each layer input becomes a learned weighted blend of all prior block states.
    use_attn_residual : bool = False
    use_adaln_in_residual : bool = False

class SRLM(hk.Module):
    def __init__(self, cfg, name="srlm"):
        super().__init__(name=name)
        self.cfg = cfg

        self.input = InputLayer(cfg, name="input")
        self.prior = S5Stack(cfg, cfg.n_priors, use_attention=False, name="prior")
        self.main = HRM(cfg, name="main")
        self.posterior = S5Stack(cfg, cfg.n_posteriors, use_attention=False, name="posterior")
        self.norm = AdaLN(cfg.d_model)
        self.output = OutputLayer(cfg, name="output")
        if cfg.use_attn_residual:
            self.stage_res = AttnResidual(cfg.d_model, cfg.use_adaln_in_residual, name="stage_res")

    def __call__(self, z, x, sigma, is_training):
        cfg = self.cfg
        q, t = self.input(x, sigma, is_training)
        y = self.prior(q, t, is_training)
        z, y = self.main(z, y, t, is_training)
        y = self.posterior(y, t, is_training)
        if cfg.use_attn_residual:
            y = self.stage_res([q], y, t)
        else:
            y = y + q
        y = self.norm(y, t)
        return z, self.output(x, y, sigma, is_training)

class S5Stack(hk.Module):
    def __init__(self, cfg, n_layers=1, use_attention=False, name=None):
        super().__init__(name=name)
        self.n_layers = n_layers
        # use_attention is passed explicitly so SRLM can force S5 for prior/posterior
        # even when cfg.use_attention=True (for the HRM layers).
        self.layers = [S5Layer(cfg, use_attention=use_attention, name=f"layer_{i}")
                       for i in range(n_layers)]
        self.dropout = cfg.dropout
        self.use_attn_residual = cfg.use_attn_residual
        if cfg.use_attn_residual:
            self.attn_res = [AttnResidual(cfg.d_model, cfg.use_adaln_in_residual, name=f"attn_res_{i}")
                             for i in range(n_layers)]

    def __call__(self, y, t, is_training):
        if self.use_attn_residual:
            blocks = []
            partial = y
            for i, layer in enumerate(self.layers):
                h = self.attn_res[i](blocks, partial, t)
                out = layer(h, t, is_training)
                # out = h + learned_deltas; subtract h to accumulate only the deltas
                delta = out - h
                if is_training:
                    delta = hk.dropout(hk.next_rng_key(), self.dropout, delta)
                partial = partial + delta
                blocks.append(partial)
            return partial
        else:
            # S5Layer has internal residuals (DiT-style); no external +y needed.
            for layer in self.layers:
                out = layer(y, t, is_training)
                if is_training:
                    out = hk.dropout(hk.next_rng_key(), self.dropout, out)
                y = out
            return y

class S5Layer(hk.Module):
    def __init__(self, cfg, use_attention=False, name=None):
        super().__init__(name=name)
        self.m = hk.Linear(6 * cfg.d_model, w_init=jnp.zeros)
        if use_attention:
            self.s5d = Attention(cfg.d_model, cfg.context_length, cfg.n_heads, name="attn")
        else:
            self.s5d = jax.vmap(S5Dual(cfg.d_model, cfg.d_state, name="s5d"))
        self.norm2 = AdaLN(cfg.d_model)
        self.ff = FeedForward(cfg.d_model)

    def __call__(self, y, t, is_training):
        @hk.remat
        def _fwd(y):
            shift_a, scale_a, gate_a, shift_ff, scale_ff, gate_ff = jnp.split(self.m(t), 6, axis=-1)

            y_norm = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(y)
            y_modulated = modulate(y_norm, shift_a, scale_a)
            a_y = self.s5d(y_modulated)
            y = y + (gate_a[:, None] * a_y)

            y_norm = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(y)
            y_modulated = modulate(y_norm, shift_ff, scale_ff)
            ff_y = self.ff(y_modulated)
            y = y + (gate_ff[:, None] * ff_y)
            return y
        #@hk.remat
        #def _fwd(y):
        #    y_norm = self.norm1(y, t)
        #    s = jax.vmap(self.s5d)(y_norm)
        #    s_norm = self.norm2(s, t)
        #    return self.ff(s_norm)
        return _fwd(y)

def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]

class HRM(hk.Module):
    def __init__(self, cfg, name=None):
        super().__init__(name=name)
        self.N = cfg.N
        self.T = cfg.T
        self.fast = FastLayer(cfg, name=f"fast")
        self.slow = SlowLayer(cfg, name=f"slow")

    def __call__(self, z, x, t, is_training):
        zH, zL = z
        zL = jax.lax.stop_gradient(zL)
        zH = jax.lax.stop_gradient(zH)
        for i in range(self.N * self.T - 1):
            zL = self.fast(zH, zL, x, t, is_training)
            if (i + 1) % self.T == 0:
                zH = self.slow(zH, zL, t, is_training)
            zL = jax.lax.stop_gradient(zL)
            zH = jax.lax.stop_gradient(zH)

        zL = self.fast(zH, zL, x, t, is_training)
        zH = self.slow(zH, zL, t, is_training)
        return (zH, zL), zH

class FastLayer(hk.Module):
    def __init__(self, cfg, name=None):
        super().__init__(name=name)
        self.normL = AdaLN(cfg.d_model)
        self.normH = AdaLN(cfg.d_model)
        self.inj = hk.Linear(cfg.d_model, name=f"inj")
        if cfg.use_attention:
            self.s5d = Attention(cfg.d_model, cfg.context_length, cfg.n_heads, name="attn")
        else:
            self.s5d = jax.vmap(S5Dual(cfg.d_model, cfg.d_state // 4, name=f"s5d"))
        #self.proj = hk.Linear(cfg.d_model)
        self.dropout = cfg.dropout

    def __call__(self, zH, zL, x, t, is_training):
        zH = self.normH(zH, t)
        zL = self.normL(zL, t)
        x = jnp.concatenate([zH, zL, x], axis=-1)
        x = self.inj(x)
        x = self.s5d(x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        #x = self.proj(x)
        return x

class SlowLayer(hk.Module):
    def __init__(self, cfg, name=None):
        super().__init__(name=name)
        self.normL = AdaLN(cfg.d_model)
        self.normH = AdaLN(cfg.d_model)
        self.has_inj = cfg.use_attention
        if cfg.use_attention:
            # d_model*2 is the feature dim here; n_heads divides it since it divides d_model
            self.s5d = Attention(cfg.d_model * 2, cfg.context_length, cfg.n_heads, name="attn")
        else:
            self.s5d = jax.vmap(S5Dual(cfg.d_model*2, cfg.d_state, name="s5d"))
        self.proj = hk.Linear(cfg.d_model, name=f"proj")
        self.dropout = cfg.dropout

    def __call__(self, zH, zL, t, is_training):
        zH = self.normH(zH, t)
        zL = self.normL(zL, t)
        x = jnp.concatenate([zH, zL], axis=-1)
        if self.has_inj:
            x = hk.Linear(x.shape[-1], name=f"inj")(x)
        x = self.s5d(x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout, x)
        x = self.proj(x)
        return x

class InputLayer(hk.Module):
    def __init__(self, cfg, name=None):
        super().__init__(name=name)
        self.input_emb = hk.Embed(cfg.vocab_size, cfg.d_model, name=f"input_emb")
        self.timestep_emb = TimestepEmbedder(cfg.d_model, cfg.d_frequency_embedding, name=f"timestep_emb")
        self.dropout = cfg.dropout

    def __call__(self, x, sigma, is_training):
        y = self.input_emb(x)
        if is_training:
            y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        sigma_emb = jax.nn.silu(self.timestep_emb(sigma))
        if is_training:
            sigma_emb = hk.dropout(hk.next_rng_key(), self.dropout, sigma_emb)
        return y, sigma_emb

class OutputLayer(hk.Module):
    def __init__(self, cfg, name=None):
        super().__init__(name=name)
        self.ff = FeedForward(cfg.d_model, name=f"ff")
        self.proj = hk.Linear(cfg.vocab_size, with_bias=False, name=f"proj")
        self.dropout = cfg.dropout

    def __call__(self, x, y, sigma, is_training):
        y = self.ff(y)
        if is_training:
            y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        y = self.proj(y)
        return scatter(x, y, sigma)

class FeedForward(hk.Module):
    def __init__(self, dim, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.pre = hk.Linear(4 * dim, name=f"pre")
        self.pos = hk.Linear(dim, w_init=jnp.zeros, b_init=jnp.zeros, name=f"pos")

    def __call__(self, x):
        x = self.pre(x)
        x = jax.nn.gelu(x) # or relu
        x = self.pos(x)
        return x

class AdaLN(hk.Module):
    def __init__(self, d_model, name=None):
        super().__init__(name=name)
        self.norm = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)
        self.proj = hk.Linear(2 * d_model)

    def __call__(self, x, cond):
        # cond: (B, d_cond) — e.g. your sigma_emb (or t)
        scale, shift = jnp.split(self.proj(cond), 2, axis=-1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]

class TimestepEmbedder(hk.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, name=None):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

    def __call__(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        x = hk.Linear(self.hidden_size, name="pre")(t_freq)
        x = jax.nn.silu(x)
        x = hk.Linear(self.hidden_size, name="pos")(x)
        return x

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
    )
    args = t[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

def mk_z(b, l, d_model):
    zH = jnp.zeros((b, l, d_model), dtype=jnp.float32)
    zL = jnp.zeros((b, l, d_model), dtype=jnp.float32)
    return zH, zL

def scatter(indices, x, sigma):
    eps = 1e-3
    esigm1 = jnp.where(sigma < 0.5, jnp.expm1(sigma), jnp.exp(sigma) - 1)
    esigm1 = jnp.clip(esigm1, a_min=1e-6, a_max=None)
    esigm1_log = jnp.log(esigm1).astype(x.dtype)[:, None, None]
    x = x - esigm1_log - jnp.log(x.shape[-1] - 1)
    
    B, L = indices.shape
    x = x.at[jnp.arange(B)[:, None], jnp.arange(L)[None, :], indices].set(0.0)
    return x
