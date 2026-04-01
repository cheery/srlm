"""
SRLM -- Straightforward reasoning language model
        (Or something like that...)

SRLM is an energy-based diffusion language model
that contains a G-Mem and CMM as an optional layer.

The memory carries context across text segments during
training and sampling, helping the model to keep coherence.

CMM is there in hopes of improving reasoning capabilities
in the model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable, Any, Tuple
from .attnres import BlockDivider, BlockAttnResOp
from .edlm import (
        MDLMLoss, NCELoss, mask_tokens, Sampler, SamplingStep,
        LogLinearSchedule, EnergyModelBase,
)
from .gmem import LatentMemoryBank, MemoryLoss
from .cmm import (
    PonderBlock, equilibrium_loss, routh_hurwitz_stable_loss,
    routh_hurwitz_unstable_loss, repulsion_loss,
)

@dataclass
class GMemConfig:
    memory_dim:            int = 256
    num_slots:             int = 1024

@dataclass
class PonderConfig:
    N_H:                   int = 3
    N_L:                   int = 6
    noise_sigma:           float = 0.01
    noise_type:            str = "additive"
    use_stablemax:         str = "3"
    use_attention:         bool = True

@dataclass
class SRLMConfig:
    gmem:                  GMemConfig
    ponder:                PonderConfig
    vocab_size:            int = 256
    max_context_length:    int = 128
    hidden_dim:            int = 256
    num_heads:             int = 8
    mlp_ratio:             int = 4
    front_layers:          int = 2
    back_layers:           int = 2
    dropout:               float = 0.2

class SRLMEnergyModel(EnergyModelBase):
    def __init__(self, cfg):
        super().__init__(cfg.hidden_dim)
        self.denoiser = SRLMDenoiser(cfg)

    def init_from_denoiser(self, denoiser):
        self.denoiser.load_state_dict(denoiser.state_dict())

    def forward(self, x0, t, memory=None):
        h, memory, importance_scores = self.denoiser.get_hidden(x0, t, memory)
        return self.outlet(h), memory, importance_scores

class SRLMDenoiser(nn.Module):
    def __init__(self, cfg: SRLMConfig):
        super().__init__()
        self.cfg = cfg
        self.input = InputLayer(cfg.vocab_size,
                                cfg.hidden_dim,
                                cfg.num_heads,
                                cfg.max_context_length)
        self.front_layers = nn.ModuleList([
            init_layer(cfg)
            for _ in range(cfg.front_layers)
        ])
        self.latent_memory = LatentMemoryBank(
                    cfg.hidden_dim,
                    cfg.gmem.memory_dim,
                    cfg.gmem.num_slots,
                    cfg.num_heads)
        self.ponder = Ponder(cfg)
        self.back_layers = nn.ModuleList([
            init_layer(cfg)
            for _ in range(cfg.back_layers)
        ])
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, -6.0)

    def decide(self, xt, t, memory=None):
        h, c, cos, sin, memory, importance_scores = self.get_front(xt, t, memory)
        q = self.ponder.decide(h)
        return q, h, c, cos, sin, memory, importance_scores

    def get_front(self, xt, t, memory=None):
        h, c, cos, sin = self.input(xt, t)
        for layer in self.front_layers:
            h = layer(h, c, cos, sin)
        h, memory, importance_scores = self.latent_memory(h, memory)
        return h, c, cos, sin, memory, importance_scores
        # TODO: fill importance scores with zeroes
        #       if latent memory not present

    def get_back(self, h, c, cos, sin):
        for layer in self.back_layers:
            h = layer(h, c, cos, sin)
        return h

    def get_behind(self, h, c, cos, sin):
        h = self.get_back(h, c, cos, sin)
        return self.out_proj(h)

    def get_hidden(self, xt, t, memory=None):
        h, c, cos, sin, memory, importance_scores = self.get_front(xt, t, memory)
        h = self.get_back(h, c, cos, sin)
        return h, memory, importance_scores

    def forward(self, xt, t, memory=None):
        h, memory, importance_scores = self.get_hidden(xt, t, memory)
        return self.out_proj(h), memory, importance_scores

class Ponder(nn.Module):
    """
    Gated pondering system that connects to the denoiser through memory.
    """
    def __init__(self, cfg: SRLMConfig):
        super().__init__()
        dim = cfg.hidden_dim
        mem_dim = cfg.gmem.memory_dim
        num_slots = cfg.gmem.num_slots

        # Difficulty gate: hidden state → scalar
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 2),
        )

        # PonderBlock operates in memory_dim space on memory slots
        self.ponder = PonderBlock(
            mem_dim, num_slots,
            num_heads=cfg.num_heads,
            N_H=cfg.ponder.N_H,
            N_L=cfg.ponder.N_L,
            noise_sigma=cfg.ponder.noise_sigma,
            noise_type=cfg.ponder.noise_type,
            use_attention=cfg.ponder.use_attention,
            use_stablemax=cfg.ponder.use_stablemax,
        )

    def decide(self, h):
        return torch.sigmoid(self.gate(h.mean(dim=1)))  # (B, 1)

    def forward(self, memory, z_H=None, z_L=None):
        """
        Args:
            memory: memory bank slots (B, num_slots, mem_dim)
            z_H:    carried ponder state (B, num_slots, mem_dim) or None
            z_L:    carried ponder state (B, num_slots, mem_dim) or None

        Returns:
            memory:     gated-updated memory
            z_H, z_L:   ponder states (for carry-over)
            q_values:   halt/continue Q-values (B, 2)
            gate_value: difficulty activation (B, 1)
        """
        # Ponder over memory slots
        z_H, z_L, q_values = self.ponder(memory, z_H, z_L)
        return z_H, z_L, q_values

def init_layer(cfg):
    return Block(cfg.hidden_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)

class InputLayer(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, max_context_length):
        super().__init__()
        self.dim = dim
        self.tok_embed = nn.Embedding(vocab_size + 1, dim)  # +1 for MASK
        self.time_mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim))

        # Rotary position embeddings
        head_dim = dim // num_heads
        half = head_dim // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(half).float() / half))
        pos = torch.arange(max_context_length).float()
        angles = pos[:, None] * freqs[None, :]
        self.register_buffer("rot_cos", torch.cos(angles), persistent=False)
        self.register_buffer("rot_sin", torch.sin(angles), persistent=False)

    def _time_embed(self, t):
        """Sinusoidal time embedding → MLP."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(emb)

    def forward(self, x, t):
        B, L = x.shape
        h = self.tok_embed(x)
        c = self._time_embed(t)
        cos = self.rot_cos[:L][None, None, :, :]
        sin = self.rot_sin[:L][None, None, :, :]
        return h, c, cos, sin

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim))
        # 6 modulation params: (γ₁, β₁, α₁, γ₂, β₂, α₂)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

        self.drop_attn = nn.Dropout(dropout)
        self.drop_mlp  = nn.Dropout(dropout)

    def forward(self, y, c, cos, sin):
        g1, b1, a1, g2, b2, a2 = self.adaln(c).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(y) * (1 + g1) + b1
        y = y + a1 * self.drop_attn(self.attn(h, cos, sin))
        h = self.norm2(y) * (1 + g2) + b2
        y = y + a2 * self.drop_mlp(self.mlp(h))
        return y

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, h, cos, sin):
        B, L, _ = h.shape
        qkv = self.qkv(h).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(out)

def _apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin,
                      x2 * cos + x1 * sin], dim=-1)


# ============================================================
# Training losses
# ============================================================

def mdlm_loss(denoiser, x0, schedule, memory=None,
              answer_mask=None, t_min=1e-4):
    """
    MDLM denoiser loss: cross-entropy on masked positions.

    Returns:
        loss, memory, importance_scores
    """
    mask_id = denoiser.cfg.vocab_size
    loss_fn = MDLMLoss(schedule, mask_id, t_min=t_min)
    xt, t, is_masked = loss_fn.perturb(x0, answer_mask=answer_mask)
    logits, memory, importance_scores = denoiser(xt, t, memory)
    loss = loss_fn(logits, x0, is_masked)
    return loss, memory, importance_scores


def nce_loss(energy_model, denoiser, x0, schedule,
             memory=None, answer_mask=None, t_min=1e-4):
    """
    NCE loss for the energy model.
    Denoiser is frozen (no_grad); only energy_model gets gradients.

    Returns:
        loss, memory, importance_scores
    """
    n_vocab = denoiser.cfg.vocab_size
    mask_id = n_vocab
    loss_fn = NCELoss(schedule, mask_id, n_vocab, t_min=t_min)

    xt, t, is_masked = loss_fn.perturb(x0, answer_mask=answer_mask)

    # Negative sample from frozen denoiser
    with torch.no_grad():
        logits, memory, importance_scores = denoiser(xt, t, memory)
    x_neg = loss_fn.sample_neg(x0, logits, is_masked)

    # Energies — positive (true x0) and negative (denoiser sample)
    e_pos, _, _ = energy_model(x0, t, memory)
    e_neg, _, _ = energy_model(x_neg, t, memory)

    loss = loss_fn(e_pos, e_neg)
    return loss, memory, importance_scores


# ============================================================
# Sampling
# ============================================================

@torch.no_grad()
def sample(denoiser, schedule, batch_size, seq_len, num_steps=256,
           energy_model=None, k=8, window_w=0.2,
           memory=None,
           device=torch.device("cpu")):
    """
    MDLM / EDLM sampling with memory

    Args:
        denoiser:      SRLMDenoiser
        schedule:      LogLinearSchedule
        energy_model:  SRLMEnergyModel or None (pure MDLM if None)
        k:             importance sampling candidates (only with energy_model)
        window_w:      importance sampling window [1-w, 1]
        memory:        initial G-Mem state or None

    Returns:
        xt:      generated tokens (batch_size, seq_len)
        memory:  final memory state
    """
    n_vocab = denoiser.cfg.vocab_size
    mask_id = n_vocab
    sampler = Sampler(schedule, mask_id, n_vocab)
    xt, stepper = sampler(batch_size, seq_len, device, num_steps)

    for s in stepper:
        logits, memory, _ = denoiser(xt, s.t, memory)
        if energy_model is not None and s.tau_n >= 1.0 - window_w:
            candidates = s.propose_x0_k(xt, logits, k)       # (k, B, L)
            c_flat = candidates.reshape(k * batch_size, seq_len)
            t_flat = s.t.unsqueeze(0).expand(k, -1).reshape(k * batch_size)
            if memory is not None:
                mem_flat = memory.unsqueeze(0).expand(k, -1, -1, -1) \
                                 .reshape(k * batch_size, *memory.shape[1:])
            else:
                mem_flat = None
            energies, _, _ = energy_model(c_flat, t_flat, mem_flat)
            energies = energies.reshape(k, batch_size)
            x0 = s.select_by_energy(candidates, energies)
        else:
            x0 = s.propose_x0(xt, logits)

        xt = s.reverse_step(xt, x0)

    return xt, memory


# ============================================================
# Ponder-enhanced forward (no deep supervision, just better logits)
# ============================================================

def ponder_forward(denoiser, xt, t, memory=None, n_ponder=3):
    """
    Forward pass with pondering for reasoning tasks.

    Runs front layers → ponder over memory → gated write-back →
    re-read memory → back layers → logits.

    Unlike PonderTrainer (deep supervision), this is a single forward
    pass suitable for use inside GRPO or plain training on hard tasks.

    Returns:
        logits, memory, importance_scores
    """
    h_front, c, cos, sin, memory, importance_scores = denoiser.get_front(xt, t, memory)

    z_H, z_L = None, None
    for _ in range(n_ponder):
        z_H, z_L, q_values = denoiser.ponder(memory, z_H, z_L)
        h_for_gate, _, _ = denoiser.latent_memory(h_front, memory)
        gate = denoiser.ponder.decide(h_for_gate)[:, 0:1].unsqueeze(-1)
        memory = memory + gate * (z_H - memory)

    # Re-read memory so ponder's updates flow into logits
    h, _, _ = denoiser.latent_memory(h_front, memory)
    logits = denoiser.get_behind(h, c, cos, sin)
    return logits, memory, importance_scores


# ============================================================
# Ponder Training (deep supervision through SRLM)
# ============================================================

@dataclass
class PonderTrainer:
    """
    Deep supervision trainer for the PonderBlock inside SRLM.

    Per segment:
      1. Read memory via G-Mem cross-attention → h
      2. Ponder over memory slots → z_H, z_L
      3. Gated write-back: memory += gate * (z_H - memory)
      4. Re-read memory → h_enriched  (so ponder gets gradient signal)
      5. get_behind(h_enriched) → logits → F.cross_entropy

    Front layers (input + transformer blocks) run once.
    G-Mem re-reads each segment so ponder's memory updates
    flow into the logits and produce gradients.
    """
    denoiser: SRLMDenoiser
    schedule: Any
    N_super: int = 16
    lambda_LM: float = 1.0
    lambda_BCE: float = 0.5
    lambda_mem: float = 0.01
    lambda_rep: float = 1e3
    lambda_equil: float = 1.0
    lambda_RH_stable: float = 1e4

    def train_step(
        self,
        x0: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        memory: Optional[torch.Tensor] = None,
        answer_mask: Optional[torch.Tensor] = None,
        t_min: float = 1e-4,
    ) -> tuple[dict[str, float], torch.Tensor]:
        """
        Returns:
            losses:  dict of loss values for logging
            memory:  updated memory state (detached)
        """
        denoiser = self.denoiser
        denoiser.train()
        mask_id = denoiser.cfg.vocab_size
        device = x0.device

        # Perturb x0 once — same noising for all segments
        loss_fn = MDLMLoss(self.schedule, mask_id, t_min=t_min)
        xt, t, is_masked = loss_fn.perturb(x0, answer_mask=answer_mask)

        # Front layers once (shared across segments)
        h_front, c, cos, sin = denoiser.input(xt, t)
        for layer in denoiser.front_layers:
            h_front = layer(h_front, c, cos, sin)

        # Initial memory read
        _, memory, importance_scores = denoiser.latent_memory(h_front, memory)

        z_H = None
        z_L = None

        all_losses = {k: 0.0 for k in [
            "LM", "BCE", "mem", "rep", "equil", "RH_stable", "total"
        ]}

        for seg in range(self.N_super):
            # Ponder over memory slots
            z_H_new, z_L_new, q_values = denoiser.ponder(memory, z_H, z_L)

            # Gated memory write-back
            # decide() uses hidden state to assess difficulty
            h_for_gate, _, _ = denoiser.latent_memory(h_front, memory)
            q_gate = denoiser.ponder.decide(h_for_gate)  # (B, 2)
            gate = q_gate[:, 0:1].unsqueeze(-1)           # (B, 1, 1)
            memory = memory + gate * (z_H_new - memory)

            # Re-read memory so ponder's updates flow into logits
            h, _, _ = denoiser.latent_memory(h_front, memory)

            # Observe via back layers
            logits = denoiser.get_behind(h, c, cos, sin)

            # --- Losses ---
            losses = {}
            losses["LM"] = loss_fn(logits, x0, is_masked)

            # ACT halt loss
            with torch.no_grad():
                correct = (logits.argmax(-1) == x0).all(dim=-1).float()
            target = torch.stack([correct, 1.0 - correct], dim=-1)
            losses["BCE"] = F.binary_cross_entropy(q_values, target)

            # Memory regularization
            if importance_scores is not None:
                mem_loss_fn = MemoryLoss()
                losses["mem"] = mem_loss_fn(importance_scores)
            else:
                losses["mem"] = torch.tensor(0.0, device=device)

            losses["rep"] = repulsion_loss(z_H_new)

            # Expensive losses only on last segment
            is_last = (seg == self.N_super - 1)
            if is_last:
                block = denoiser.ponder.ponder.block
                losses["equil"] = equilibrium_loss(z_H_new, z_L_new, block)
                losses["RH_stable"] = routh_hurwitz_stable_loss(
                    z_H_new + z_L_new, block)
            else:
                losses["equil"] = torch.tensor(0.0, device=device)
                losses["RH_stable"] = torch.tensor(0.0, device=device)

            total = (
                self.lambda_LM * losses["LM"]
                + self.lambda_BCE * losses["BCE"]
                + self.lambda_mem * losses["mem"]
                + self.lambda_rep * losses["rep"]
                + self.lambda_equil * losses["equil"]
                + self.lambda_RH_stable * losses["RH_stable"]
            ) / self.N_super

            total.backward(retain_graph=(seg < self.N_super - 1))

            # Detach for next segment
            z_H = z_H_new.detach()
            z_L = z_L_new.detach()
            memory = memory.detach()

            for k, v in losses.items():
                all_losses[k] += v.item()
            all_losses["total"] += total.item() * self.N_super

            # ACT early stopping
            with torch.no_grad():
                q_mean = q_values.mean(0)
                if q_mean[0] > q_mean[1] and seg >= 1:
                    break

        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        n_run = seg + 1
        return {k: v / n_run for k, v in all_losses.items()}, memory
