"""
MDLM + Memory Sparse Attention (MSA)
=====================================
MDLM denoiser augmented with MSA-style sparse cross-attention
to a large external memory bank.

Architecture:
  - Bidirectional self-attention over the sequence being denoised
  - Sparse cross-attention to a pre-encoded memory bank via
    learned routing (top-k document selection)
  - adaLN-zero time conditioning for diffusion timestep
  - Document-wise RoPE for memory, standard RoPE for query

The memory bank is encoded offline (one-time), then at each
denoising step the model retrieves the most relevant documents
via cosine-similarity routing and attends to their compressed
KV representations.

Compatible with EDLM: train this denoiser with mdlm_loss /
conditional_mdlm_loss, then add an energy model on top.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from edlm import MDLMLoss, NCELoss, Sampler, as_text

# ============================================================
# RoPE
# ============================================================

def _rope_freqs(head_dim, base=10000.0, device=None):
    return 1.0 / (base ** (torch.arange(0, head_dim, 2,
                   device=device).float() / head_dim))


def apply_rope(x, positions, base=10000.0):
    """Apply RoPE.  x: (..., seq_len, head_dim), positions: (seq_len,)."""
    hd = x.shape[-1]
    freqs = _rope_freqs(hd, base, device=x.device)
    angles = positions[:, None].float() * freqs[None, :]
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack([x1 * cos_a - x2 * sin_a,
                       x1 * sin_a + x2 * cos_a], dim=-1)
    return out.reshape(x.shape)


def apply_rope_heads(x, offset=0, base=10000.0):
    """Apply RoPE to (B, H, L, D) or (H, L, D)."""
    L = x.shape[-2]
    positions = torch.arange(L, device=x.device) + offset
    return apply_rope(x, positions, base)


# ============================================================
# Chunk-wise mean pooling  (ϕ in the paper)
# ============================================================

def chunk_mean_pool(x, chunk_size):
    """(..., seq_len, dim) → (..., ceil(seq_len/P), dim)."""
    seq_len, dim = x.shape[-2], x.shape[-1]
    pad_len = (-seq_len) % chunk_size
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, pad_len))
    leading = x.shape[:-2]
    n_chunks = x.shape[-2] // chunk_size
    x = x.reshape(*leading, n_chunks, chunk_size, dim)
    if pad_len > 0:
        # Mask out padding in the last chunk
        mask = torch.ones(n_chunks, chunk_size, 1, device=x.device)
        mask[-1, chunk_size - pad_len:] = 0
        return (x * mask).sum(-2) / mask.sum(-2).clamp(min=1e-8)
    return x.mean(-2)


# ============================================================
# Routing score computation  (Eq. 2)
# ============================================================
def compute_routing_scores(q_router, kr_pooled_tensor):
    """
    Args:
        q_router:         (B, H, Q, D) routing query
        kr_pooled_tensor: (N, H, C, D) padded/stacked routing keys
    Returns:
        (B, N) document scores
    """
    # Normalize inputs
    qn = F.normalize(q_router, dim=-1)           # (B, H, Q, D)
    kn = F.normalize(kr_pooled_tensor, dim=-1)   # (N, H, C, D)

    # Broadcast and multiply: (B, 1, H, Q, D) x (N, H, C, D) -> (B, N, H, Q, C)
    # Using einsum for clarity and speed:
    sim = torch.einsum("bhqd,nhcd->bnhqc", qn, kn)

    # Reductions:
    # 1. mean over heads (dim 2) -> (B, N, Q, C)
    # 2. max over chunks (dim -1) -> (B, N, Q)
    # 3. max over query tokens (dim -1) -> (B, N)
    scores = sim.mean(dim=2).max(dim=-1).values.max(dim=-1).values

    return scores


# ============================================================
# Auxiliary contrastive routing loss  (Eq. 5)
# ============================================================

def auxiliary_routing_loss(scores, positive_mask, temperature=0.05):
    """
    Supervised contrastive loss on router scores.

    Args:
        scores:        (B, N) routing scores
        positive_mask: (B, N) binary — 1 for positive documents
        temperature:   scalar
    Returns:
        scalar loss
    """
    s = scores / temperature
    pos_s = s * positive_mask
    neg_exp_sum = (torch.exp(s) * (1 - positive_mask)).sum(-1, keepdim=True)
    pos_exp = torch.exp(pos_s) * positive_mask
    log_probs = torch.log(pos_exp / (pos_exp + neg_exp_sum + 1e-8)
                          + 1e-8) * positive_mask
    n_pos = positive_mask.sum(-1).clamp(min=1)
    return -(log_probs.sum(-1) / n_pos).mean()


# ============================================================
# Transformer blocks
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                                + 1e-6).to(x.dtype) * self.scale


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SelfAttention(nn.Module):
    """Bidirectional multi-head self-attention with RoPE."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, rope_base=10000.0):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)         # 3 × (B, H, L, D)
        q = apply_rope_heads(q, base=rope_base)
        k = apply_rope_heads(k, base=rope_base)
        out = F.scaled_dot_product_attention(q, k, v)  # bidirectional
        return self.out(out.transpose(1, 2).reshape(B, L, -1))


class CrossAttention(nn.Module):
    """Cross-attention from query to memory KV."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, q_hidden, mem_k, mem_v):
        """
        Args:
            q_hidden: (B, L, dim)
            mem_k:    (B, H, M, D) pre-projected, pre-RoPE'd keys
            mem_v:    (B, H, M, D) pre-projected values
        Returns:
            (B, L, dim)
        """
        B, L, _ = q_hidden.shape
        q = self.wq(q_hidden).reshape(B, L, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)                     # (B, H, L, D)
        # No RoPE on cross-attention query — memory has doc-wise RoPE already
        out = F.scaled_dot_product_attention(q, mem_k, mem_v)
        return self.out(out.transpose(1, 2).reshape(B, L, -1))

class RouterProjector(nn.Module):
    """Learned routing projectors Q^R and K^R."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wqr = nn.Linear(dim, dim, bias=False)
        self.wkr_mean = nn.Linear(dim, dim)
        self.wkr_var  = nn.Linear(dim, dim)
        self.wkr_inv = nn.Linear(dim, dim)

    def project_query(self, x):
        """(B, L, dim) → (B, H, L, D)"""
        B, L, _ = x.shape
        return self.wqr(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def project_key(self, x):
        return self.project_key_vae(x)[0]

    def project_key_vae(self, x):
        """(B, L, dim) → (B, H, L, D)"""
        B, L, _ = x.shape
        mean = self.wkr_mean(x)
        var  = self.wkr_var(x)
        z = mean + var*torch.rand_like(var)
        return z.reshape(B, L, self.num_heads,
                               self.head_dim).permute(0, 2, 1, 3), mean, var

    def unproject_key(self, kr):
        """(B, H, L, D) → (B, L, dim)"""
        B, _, L, _ = kr.shape
        return self.wkr_inv(kr.permute(0, 2, 1, 3).reshape(B, L, self.num_heads*self.head_dim))


class KVProjector(nn.Module):
    """Standard K/V projectors (shared between encoding and attention)."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """(B, L, dim) → k: (B, H, L, D), v: (B, H, L, D)"""
        B, L, _ = x.shape
        k = self.wk(x).reshape(B, L, self.num_heads,
                                self.head_dim).permute(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.num_heads,
                                self.head_dim).permute(0, 2, 1, 3)
        return k, v


# ============================================================
# adaLN-zero time conditioning
# ============================================================

class AdaLNModulation(nn.Module):
    """Produces 6 modulation params from time embedding."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, c):
        """c: (B, dim) → 6 × (B, 1, dim)"""
        return self.proj(c).unsqueeze(1).chunk(6, dim=-1)


# ============================================================
# MSA-Denoiser Block
# ============================================================

class MSADenoiserBlock(nn.Module):
    """
    Transformer block with:
      1. adaLN-zero conditioned bidirectional self-attention
      2. Optional MSA sparse cross-attention to memory
      3. adaLN-zero conditioned FFN
    """
    def __init__(self, dim, num_heads, chunk_size, mlp_ratio=4, has_memory=False):
        super().__init__()
        self.has_memory = has_memory
        self.chunk_size = chunk_size

        # Self-attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = SelfAttention(dim, num_heads)
        self.adaln1 = AdaLNModulation(dim)

        # Cross-attention to memory (only for memory-enabled layers)
        if has_memory:
            self.norm_mem = nn.LayerNorm(dim, elementwise_affine=False)
            self.cross_attn = CrossAttention(dim, num_heads)
            # Router
            self.router = RouterProjector(dim, num_heads)
            # KV projector for memory encoding
            self.mem_kv = KVProjector(dim, num_heads)

        # FFN
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = SwiGLUFFN(dim, dim * mlp_ratio)
        self.adaln2 = AdaLNModulation(dim)

    def forward(self, x_in, c, mem_k=None, mem_v=None):
        """
        Args:
            x: (B, L, dim) hidden states
            c: (B, dim) time conditioning
            mem_k: (B, H, M, D) selected memory keys (pre-RoPE'd)
            mem_v: (B, H, M, D) selected memory values
        """
        # Self-attention with adaLN
        g1, b1, a1, _, _, _ = self.adaln1(c)
        h = self.norm1(x_in) * (1 + g1) + b1
        x = x_in + a1 * self.self_attn(h)

        # Cross-attention to memory
        if self.has_memory and mem_k is not None:
            h_mem = self.norm_mem(x_in)
            mem_out = self.cross_attn(h_mem, mem_k, mem_v)
            x = x + mem_out

        # FFN with adaLN
        _, _, _, g2, b2, a2 = self.adaln2(c)
        h = self.norm2(x) * (1 + g2) + b2
        x = x + a2 * self.ffn(h)

        return x

    def encode_memory_kv(self, doc_hidden):
        """
        Encode a document's hidden states into compressed KV + routing keys.

        Args:
            doc_hidden: (B, L_doc, dim)
        Returns:
            dict with k_pooled, v_pooled, kr_pooled — each (B, H, C, D)
        """
        assert self.has_memory
        norm_h = self.norm_mem(doc_hidden)
        k, v = self.mem_kv(norm_h)
        # Document-wise RoPE (positions starting from 0 per doc)
        k = apply_rope_heads(k, offset=0)
        # Router key
        kr = self.router.project_key(norm_h)
        return {
            "k_pooled": k.squeeze(0).cpu(),
            "v_pooled": v.squeeze(0).cpu(),
            "kr_pooled": chunk_mean_pool(kr, self.chunk_size).squeeze(0),
        }


# ============================================================
# Memory Bank
# ============================================================

class MemoryBank:
    """
    Stores pre-encoded document KV caches for sparse retrieval.

    Each document is a dict mapping layer_idx → {k_pooled, v_pooled, kr_pooled}.
    """
    def __init__(self, denoiser, device, chunk_size):
        self.denoiser = denoiser
        self.device = device
        self.chunk_size = chunk_size
        self.documents = {i:[] for i in range(denoiser.msa_start,
                                            denoiser.num_layers)}
        self.routing = {}
        self.size = 0

    def add_document(self, doc_tokens):
        """Add a pre-encoded document. doc_cache: dict[layer_idx → kv_dict]."""
        cache = self.denoiser.encode_document(doc_tokens.to(self.device))
        for layer_idx, entry in cache.items():
            self.documents[layer_idx].append(entry)
        self.size += 1

    def finalize(self):
        self.routing.clear()
        for layer_idx, entries in self.documents.items():
            krs = torch.stack([entry["kr_pooled"] for entry in entries])
            self.routing[layer_idx] = krs

    def __len__(self):
        return self.size

    def get_routing_keys(self, layer_idx):
        return self.routing[layer_idx]

    def get_kv(self, layer_idx, doc_indices):
        """
        Gather and concatenate KVs for selected documents.

        Args:
            layer_idx: which layer
            doc_indices: (B, top_k) tensor of indices to fetch
        Returns:
            k: (B, 1, top_k, L, D)
            v: (B, 1, top_k, L, D)
        """
        doc = self.documents[layer_idx]
        ak, av = [], []
        for indices in doc_indices:
            ks, vs = [], []
            for idx in indices.tolist():
                ks.append(doc[idx]["k_pooled"]) # (H,L,D//H)
                vs.append(doc[idx]["v_pooled"])
            ak.append( torch.stack(ks) ) # (top_k,H,L,D//H)
            av.append( torch.stack(vs) ) # (top_k,H,L,D//H)
        k = torch.stack(ak, dim=1).to(self.device)
        v = torch.stack(av, dim=1).to(self.device)
        return k, v # (top_k, B,H,L,D//H)

# ============================================================
# MSA-augmented MDLM Denoiser
# ============================================================

class MSADenoiser(nn.Module):
    """
    MDLM denoiser with MSA-style sparse memory cross-attention.

    Lower layers: standard bidirectional self-attention.
    Upper layers: self-attention + sparse cross-attention to memory.

    The model predicts x₀ from x_t (with MASK tokens), conditioned
    on diffusion time t and optionally on a memory bank of documents.
    """
    def __init__(self, n_vocab, max_len, dim, num_heads,
                 num_layers, msa_start, top_k,
                 chunk_size, mlp_ratio):
        super().__init__()
        self.n_vocab = n_vocab
        self.mask_id = n_vocab
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.msa_start = msa_start

        self.tok_embed = nn.Embedding(n_vocab + 1, dim)   # +1 for MASK
        self.out_proj = nn.Linear(dim, n_vocab)

        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            has_mem = (i >= msa_start)
            self.blocks.append(
                MSADenoiserBlock(dim, num_heads, chunk_size, mlp_ratio, has_memory=has_mem))

        self.final_norm = nn.LayerNorm(dim)

    def _time_embed(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(emb)

    def get_hidden(self, x, t, memory_bank, compute_aux=False):
        """Final hidden states (B, L, dim) — no memory."""
        B, L = x.shape
        h = self.tok_embed(x)
        c = self._time_embed(t)
        aux_loss = 0.0
        for i, block in enumerate(self.blocks):
            if (block.has_memory and memory_bank is not None
                    and len(memory_bank) > 0):

                # --- Routing: select top-k documents ---
                h_norm = block.norm_mem(h)
                q_router = block.router.project_query(h_norm)  # (B,H,L,D)

                kr_list = memory_bank.get_routing_keys(i)
                scores = compute_routing_scores(
                    q_router, kr_list)             # (B, N)
                actual_k = min(self.top_k, scores.shape[-1])
                _, topk_idx = scores.topk(actual_k, dim=-1)

                # Gather KVs from selected documents
                mem_k, mem_v = memory_bank.get_kv(i, topk_idx)

                h = block(h, c, mem_k=mem_k[0], mem_v=mem_v[0])
            elif block.has_memory:
                h_norm = block.norm_mem(h)
                if compute_aux:
                    # Compute VAE loss for router
                    kr, kr_mean, kr_var = block.router.project_key_vae(h_norm)
                    restored_h_norm = block.router.unproject_key(kr)
                    reproduction_loss = F.mse_loss(restored_h_norm, h_norm)
                    #reproduction_loss = nn.functional.binary_cross_entropy(restored_h_norm, h_norm, reduction='sum')
                    KLD = - 0.5 * torch.sum(1 + kr_var - kr_mean.pow(2) - kr_var.exp())
                    # Compute cosine similarity along
                    q = block.router.project_query(h_norm * torch.rand_like(h_norm))
                    cos = nn.CosineSimilarity(dim=-1)(kr, q).mean()
                    #print("DEBUG", reproduction_loss.item(), KLD.item(), (1 - cos).item())
                    aux_loss += reproduction_loss + KLD + 1 - cos
                h_norm = block.norm_mem(h)
                mem_k, mem_v = block.mem_kv(h_norm)
                h = block(h, c, mem_k=mem_k, mem_v=mem_v)
            else:
                h = block(h, c)
        if compute_aux:
            return self.final_norm(h), aux_loss
        else:
            return self.final_norm(h)

    @torch.no_grad()
    def encode_document(self, doc_tokens):
        """
        Stage 1 (offline): encode a document into compressed KV cache.

        Runs the document through all layers, extracting compressed
        KV + routing keys at memory-enabled layers.

        Args:
            doc_tokens: (L_doc) token ids
        Returns:
            dict[layer_idx → {k, v, kr_pooled}]
        """
        assert len(doc_tokens.shape) == 1
        h = self.tok_embed(doc_tokens.unsqueeze(0))
        # Use t=0 for encoding (documents are clean text)
        t_zero = torch.zeros(h.shape[0], device=h.device)
        c = self._time_embed(t_zero)
        cache = {}
        for i, block in enumerate(self.blocks):
            if block.has_memory:
                cache[i] = block.encode_memory_kv(h)
            h = block(h, c)
        return cache

    def forward(self, xt, t, memory_bank=None, compute_aux=False):
        """
        Forward pass with optional memory retrieval.

        Args:
            xt:          (B, L) noised tokens
            t:           (B,) diffusion timestep
            memory_bank: MemoryBank or None
        Returns:
            logits: (B, L, n_vocab)
        """
        if compute_aux:
            h, aux_loss = self.get_hidden(xt, t, memory_bank, compute_aux=True)
            return self.out_proj(h), aux_loss
        else:
            h = self.get_hidden(xt, t, memory_bank)
            return self.out_proj(h)

    #def forward(self, xt, t, memory_bank=None):
    #    """
    #    Forward pass with differentiable routing during training,
    #    and discrete top-k routing during evaluation.
    #    """
    #    B, L = xt.shape
    #    h = self.tok_embed(xt)
    #    c = self._time_embed(t)

    #    for i, block in enumerate(self.blocks):
    #        if (block.has_memory and memory_bank is not None
    #                and len(memory_bank) > 0):

    #            h_norm = block.norm_mem(h)
    #            q_router = block.router.project_query(h_norm)  # (B,H,L,D)
    #            kr_list = memory_bank.get_routing_keys(i)
    #            scores = compute_routing_scores(q_router, kr_list)  # (B, N)

    #            if self.training:
    #                # --- TRAINING: Differentiable Straight-Through Routing ---
    #                # Returns a one-hot vector (forward) with softmax gradients (backward)
    #                weights = F.gumbel_softmax(scores, tau=1.0, hard=True)

    #                all_k, all_v = memory_bank.get_all_kv(i)

    #                # Weighted sum over the N documents
    #                mem_k = (weights[:, :, None, None, None] * all_k[None, ...]).sum(1)
    #                mem_v = (weights[:, :, None, None, None] * all_v[None, ...]).sum(1)

    #                # Wrap in lists so CrossAttention's zip(mem_k, mem_v) iterates exactly once
    #                h = block(h, c, mem_k=[mem_k], mem_v=[mem_v])

    #            else:
    #                # --- EVALUATION: Sparse Top-K Retrieval ---
    #                actual_k = min(self.top_k, scores.shape[-1])
    #                _, topk_idx = scores.topk(actual_k, dim=-1)

    #                # Gather KVs from selected documents
    #                mem_k, mem_v = memory_bank.get_kv(i, topk_idx)

    #                h = block(h, c, mem_k=mem_k, mem_v=mem_v)
    #        else:
    #            h = block(h, c)

    #    h = self.final_norm(h)
    #    return self.out_proj(h), None


# ============================================================
# MDLM loss with memory
# ============================================================

def mdlm_msa_loss(denoiser, x0, schedule, memory_bank=None,
                  answer_mask=None, t_min=1e-4):
    """
    MDLM cross-entropy loss with optional memory bank.

    Args:
        x0:          (B, L) clean tokens
        schedule:    noise schedule with .alpha(t)
        memory_bank: MemoryBank or None
        answer_mask: (B, L) bool, True for answer positions (conditional)
        t_min:       minimum time value
    """
    mdlm_loss = MDLMLoss(schedule, denoiser.mask_id)
    xt, t, is_masked = mdlm_loss.perturb(x0, answer_mask=None)
    if memory_bank is None:
        logits, aux = denoiser(xt, t, compute_aux=True)
    else:
        aux = 0
        logits = denoiser(xt, t, memory_bank)
    loss = mdlm_loss(logits, x0, is_masked)
    return loss + aux * 0.01

def nce_loss(energy_model, denoiser, x0, schedule, memory_bank,
             answer_mask=None, t_min=1e-4):
    nce_loss = NCELoss(schedule, denoiser.mask_id, denoiser.n_vocab)
    xt, t, is_masked = nce_loss.perturb(x0, answer_mask=answer_mask)
    with torch.no_grad():
        logits = denoiser(xt, t, memory_bank)
    x1 = nce_loss.sample_neg(x0, logits, is_masked)
    loss = nce_loss(
        e_pos = energy_model(x0, xt, t, memory_bank),                    # (B,)
        e_neg = energy_model(x1, xt, t, memory_bank))                    # (B,)
    return loss

# ============================================================
# Sampling with memory
# ============================================================

@torch.no_grad()
def sample_with_memory(
    denoiser, schedule, memory_bank,
    batch_size, seq_len, num_steps=128,
    energy_model=None, k=8, window_w=0.2,
    device=torch.device("cpu"),
):
    """
    MDLM sampling with memory-augmented denoiser.

    Fully masked → progressively unmask, retrieving from
    memory bank at each denoising step.
    """
    mask_id = denoiser.mask_id
    n = denoiser.n_vocab
    sampler = Sampler(schedule, denoiser.mask_id, denoiser.n_vocab)
    xt, stepper = sampler(batch_size, seq_len, device, num_steps,
            energy_model=energy_model,
            k=k,
            window_w=window_w)
    for s in stepper:
        logits = denoiser(xt, s.t, memory_bank)
        xt = s.sample(xt, logits)
    return xt


class EnergyModel(nn.Module):
    """
    Residual energy function  E_φ(x₀, x_t, t)  — Eq (7).

    Architecture (Section 5.1 / Appendix C.1):
      1. Bidirectional transformer backbone (initialised from the
         pretrained MDLM denoiser).
      2. Mean-pool the final-layer token representations.
      3. Project to a single scalar energy.

    Low energy  → coherent / realistic x₀ proposal.
    High energy → incoherent / unlikely x₀ proposal.
    """
    def __init__(self, n_vocab, seq_len, dim,
                 num_heads, num_layers,
                 msa_start, top_k,
                 chunk_size, mlp_ratio
                 ):
        super().__init__()
        self.backbone = MSADenoiser(
            n_vocab, seq_len, dim,
            num_heads, num_layers,
            msa_start, top_k,
            chunk_size, mlp_ratio=2)
        self.energy_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1))
        # Start near zero so initial energies are small
        nn.init.zeros_(self.energy_head[-1].weight)
        nn.init.zeros_(self.energy_head[-1].bias)

    def init_from_denoiser(self, denoiser):
        """Copy pretrained MDLM weights into the backbone."""
        self.backbone.load_state_dict(denoiser.state_dict())

    def forward(self, x0, xt, t, memory_bank=None):
        """
        E_φ(x₀, x_t, t) → (B,) scalar energies.

        The candidate x₀ is fed through the backbone (conditioned on t);
        since carry-over means x₀ = x_t at unmasked positions, the model
        implicitly has access to x_t through x₀ itself.
        """
        h = self.backbone.get_hidden(x0, t, memory_bank) # (B, L, dim)
        pooled = h.mean(dim=1)                        # (B, dim)
        return self.energy_head(pooled).squeeze(-1)   # (B,)
    

# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- config ---
    n_vocab = 256
    batch_size = 32
    seq_len = 64
    dim = 128
    num_heads = 4
    num_layers = 4
    msa_start = 2
    top_k = 1 # TODO: consider what to do with multiple mem_k,mem_v
    chunk_size = 16
    # ---- data ----
    from edlm import load_kalevala, create_dataloader
    text = load_kalevala()
    dataloader = create_dataloader(
        text, batch_size=batch_size, length=seq_len, stride=64)
    def infinite_loader(dl):
        while True:
            yield from dl
    loader = infinite_loader(dataloader)
    # --- model ---
    denoiser = MSADenoiser(
            n_vocab, seq_len, dim,
            num_heads, num_layers,
            msa_start, top_k,
            chunk_size, mlp_ratio=2).to(device)
    params = sum(p.numel() for p in denoiser.parameters())
    print(f"MSADenoiser params: {params:,}")
    memory_bank = MemoryBank(denoiser, device, chunk_size)
    print("Testing memory bank")
    for _ in range(10):
        for doc_tokens in next(loader):
            memory_bank.add_document(doc_tokens)
    memory_bank.finalize()
    print("Testing denoiser with memory bank")
    xt = torch.randint(0, n_vocab+1, (batch_size, seq_len), device=device)
    t = torch.rand(batch_size, device=device)
    print("Testing without memory")
    logits = denoiser(xt, t)
    print(logits.shape)
    print("Testing with memory")
    logits = denoiser(xt, t, memory_bank)
    print(logits.shape)
    
    # --- loss ---
    from edlm import LogLinearSchedule
    schedule = LogLinearSchedule()

    mdlm_steps = 10000     # phase 1: train denoiser
    mdlm_lr    = 3e-4

    nce_steps  = 10000       # phase 2: train energy (NCE)
    nce_lr     = 1e-4

    sample_steps  = 128     # sampling
    importance_k  = 8
    importance_w  = 0.2

    from ema import EMA

    # ==========================================================
    # Phase 1: Train MDLM Denoiser
    # ==========================================================
    print("=" * 60)
    print("Phase 1: Training MDLM Denoiser")
    print("=" * 60)

    print(f"Denoiser params: "
          f"{sum(p.numel() for p in denoiser.parameters()):,}")

    ema_den = EMA(denoiser, mu=0.999)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=mdlm_lr)

    ema_lv = None
    for step in range(1, mdlm_steps + 1):
        x0 = next(loader).to(device)
        denoiser.train()
        opt.zero_grad()

        loss = mdlm_msa_loss(denoiser, x0, schedule)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        opt.step()
        ema_den.update(denoiser)

        lv = loss.item()
        ema_lv = lv if ema_lv is None else 0.99 * ema_lv + 0.01 * lv

        if step % 100 == 0:
            print(f"  step {step:5d}  loss {lv:.4f}  ema {ema_lv:.4f}")

        if step % 1000 == 0:
            denoiser.eval()
            ema_den.apply(denoiser)
            s = sample_with_memory(denoiser, schedule, None, batch_size=2,
                       seq_len=seq_len, num_steps=64, device=device)
            for i in range(2):
                print(f"  MDLM sample {i+1}: "
                      f"{repr(as_text(s[i][:80]))}")
            ema_den.restore(denoiser)
    ema_den.apply(denoiser)
    # ==========================================================
    # Phase 1.5: Train memory system
    # ==========================================================
    #for p in denoiser.parameters():
    #    p.requires_grad = False
    #for block in denoiser.blocks:
    #    if block.has_memory:
    #        for p in block.norm_mem.parameters():
    #            p.requires_grad = True
    #        for p in block.cross_attn.parameters():
    #            p.requires_grad = True
    #        for p in block.router.parameters():
    #            p.requires_grad = True
    #        for p in block.mem_kv.parameters():
    #            p.requires_grad = True

    # --- encode documents ---
    print("\nEncoding documents...")
    denoiser.eval()
    with torch.no_grad():
        memory_bank = MemoryBank(denoiser, device, chunk_size)
        i = 0
        for x0 in dataloader:
            x0 = x0.to(device)
            for x in x0:
                memory_bank.add_document(x)
            if i % 100 == 0:
                print(f"  doc {i}: bank len: {len(memory_bank)} ")
            i += 1
        memory_bank.finalize()

    # ==========================================================
    # Phase 2: Train MDLM Denoiser with memory
    # ==========================================================
    for p in denoiser.parameters():
        p.requires_grad = True
    print("=" * 60)
    print("Phase 2: Training MDLM Denoiser with memory")
    print("=" * 60)

    print(f"Denoiser params: "
          f"{sum(p.numel() for p in denoiser.parameters()):,}")

    ema_den = EMA(denoiser, mu=0.999)
    opt = torch.optim.AdamW(denoiser.parameters(), lr=mdlm_lr)

    ema_lv = None
    for step in range(1, mdlm_steps + 1):
        x0 = next(loader).to(device)
        denoiser.train()
        opt.zero_grad()

        loss = mdlm_msa_loss(denoiser, x0, schedule, memory_bank)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        opt.step()
        ema_den.update(denoiser)

        lv = loss.item()
        ema_lv = lv if ema_lv is None else 0.99 * ema_lv + 0.01 * lv

        if step % 100 == 0:
            print(f"  step {step:5d}  loss {lv:.4f}  ema {ema_lv:.4f}")

        if step % 1000 == 0:
            denoiser.eval()
            ema_den.apply(denoiser)
            s = sample_with_memory(denoiser, schedule, memory_bank, batch_size=2,
                       seq_len=seq_len, num_steps=64, device=device)
            for i in range(2):
                print(f"  MDLM sample {i+1}: "
                      f"{repr(as_text(s[i][:80]))}")
            ema_den.restore(denoiser)

    # switch to EMA weights for denoiser
    ema_den.apply(denoiser)
    denoiser.eval()
    for p in denoiser.parameters():
        p.requires_grad = False


    # ==========================================================
    # Phase 2: Train Energy Model (NCE)
    # ==========================================================
    print("\n" + "=" * 60)
    print("Phase 3: Training Energy Model (NCE)")
    print("=" * 60)

    energy = EnergyModel(
        n_vocab, seq_len=seq_len, dim=dim,
        num_heads=num_heads, num_layers=num_layers,
        msa_start=msa_start, top_k=top_k,
        chunk_size=chunk_size, mlp_ratio=2).to(device)
    energy.init_from_denoiser(denoiser)
    print(f"Energy model params: "
          f"{sum(p.numel() for p in energy.parameters()):,}")

    ema_eng = EMA(energy, mu=0.999)
    opt_e = torch.optim.AdamW(energy.parameters(), lr=nce_lr)

    loader = infinite_loader(dataloader)
    ema_lv = None
    for step in range(1, nce_steps + 1):
        x0 = next(loader).to(device)
        energy.train()
        opt_e.zero_grad()

        loss = nce_loss(energy, denoiser, x0, schedule, memory_bank)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(energy.parameters(), 1.0)
        opt_e.step()
        ema_eng.update(energy)

        lv = loss.item()
        ema_lv = lv if ema_lv is None else 0.99 * ema_lv + 0.01 * lv

        if step % 100 == 0:
            print(f"  step {step:5d}  nce_loss {lv:.4f}  "
                  f"ema {ema_lv:.4f}")

    ema_eng.apply(energy)
    energy.eval()

    # ==========================================================
    # Phase 3: Sampling comparison
    # ==========================================================
    print("\n" + "=" * 60)
    print("Phase 3: Sampling")
    print("=" * 60)

    print("\n--- MDLM samples (no energy correction) ---")
    for _ in range(10):
        s = sample_with_memory(denoiser, schedule, None,
                               batch_size=4, seq_len=seq_len,
                               num_steps=sample_steps, device=device)
        for i in range(4):
            print(f"  {i+1}: {repr(as_text(s[i][:100]))}")

    print(f"\n--- EDLM samples (k={importance_k}, w={importance_w}) ---")
    for _ in range(100):
        s = sample_with_memory(denoiser, schedule, memory_bank, batch_size=4, seq_len=seq_len,
                   num_steps=sample_steps, energy_model=energy,
                   k=importance_k, window_w=importance_w, device=device)
        for i in range(4):
            print(f"  {i+1}: {repr(as_text(s[i][:100]))}")
