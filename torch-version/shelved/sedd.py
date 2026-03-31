"""
Score Entropy Discrete Diffusion (SEDD)
========================================
PyTorch implementation of the algorithms from:
"Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution"
(Lou, Meng, Ermon — ICML 2024)

Implements:
  - Algorithm 1: Score Entropy Training (DWDSE loss)
  - Algorithm 2: Unconditional Sampling (Euler & Tweedie)
  - Algorithm 3: Conditional Sampling (infilling / prompting)

The score network s_θ : X × R → R^{d×n} learns the concrete score,
i.e. the ratios p_t(y)/p_t(x) for Hamming-distance-1 neighbours.
Two transition matrices are supported: Q^uniform and Q^absorb (Eqs 15–16).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Callable, Any, Tuple
from dataclasses import dataclass

ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

from pathlib import Path
from torch.utils.data import Dataset, DataLoader

directory = Path(__file__).parent

def load_kalevala():
    filename = (directory / "../../../data/kalevala.plain.txt").resolve()
    with filename.open("r", encoding="utf-8") as fd:
        text = fd.read().replace("\n", " ")
    return text

def create_dataloader(text,
                      encoder=None,
                      batch_size=4,
                      length=256,
                      stride=128,
                      shuffle=True,
                      drop_last=False,
                      num_workers=0):
    dataset = KalevalaDataset(text, encoder or text_to_tensor, length, stride)
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle = shuffle,
            drop_last = drop_last,
            num_workers = num_workers)
    return dataloader

class KalevalaDataset(Dataset):
    def __init__(self, text, encoder, length, stride):
        self.inputs = []
        self.targets = []

        data = encoder(text)
        for i in range(0, len(data) - length - 1, stride):
            input_chunk = data[i:i + length]
            target_chunk = data[i+1:i+1+length]
            self.inputs.append(input_chunk)
            self.targets.append(target_chunk)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

def text_to_tensor(text):
    data = text.encode("utf-8")
    raw = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return raw.type(torch.long)

def as_text(p: torch.Tensor) -> str:
    return p.cpu().to(torch.uint8).numpy().tobytes().decode("utf-8", errors="replace")

# ============================================================
# Noise Schedules  (Appendix C.1)
# ============================================================
# σ̄(t) = cumulative noise = ∫₀ᵗ σ(s)ds
# σ(t) = instantaneous rate = dσ̄/dt
# t ∈ [0, 1].  At t=0 little noise; at t=1 base distribution.

class GeometricSchedule:
    r"""
    σ̄(t) = σ_min^{1-t} · σ_max^t          (Appendix C.1)
    σ(t) = σ̄(t) · ln(σ_max / σ_min)
    """
    def __init__(self, sigma_min: float = 1e-5, sigma_max: float = 20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_ratio = math.log(sigma_max / sigma_min)

    def sigma_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min ** (1.0 - t) * self.sigma_max ** t

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_bar(t) * self.log_ratio


class LogLinearSchedule:
    r"""
    σ̄(t) = -log(1 - (1-ε)t)               (Appendix C.1)
    σ(t) = (1-ε) / (1 - (1-ε)t)
    """
    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def sigma_bar(self, t: torch.Tensor) -> torch.Tensor:
        return -torch.log1p(-(1.0 - self.eps) * t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.eps) / (1.0 - (1.0 - self.eps) * t)


# ============================================================
# Forward Transition  p_{t|0}(·|x₀)  (Section 3.3, Algorithm 1)
# ============================================================
# Each token is perturbed independently via
#     x_t^i  ~  p_{t|0}^{tok}(·|x₀^i) = exp(σ̄(t) Q^{tok})_{x₀^i}   (Eq 14)
# Closed forms for the two Q matrices:

def _forward_probs_absorb(sigma_bar: torch.Tensor, x0: torch.Tensor, n: int):
    """
    Absorbing diffusion (Eq 16).  MASK = token index n-1.

    p_{t|0}(y | x₀) = e^{-σ̄}·δ(y, x₀)  +  (1 - e^{-σ̄})·δ(y, MASK)

    Args:
        sigma_bar: (B,)
        x0:        (B, d)  tokens in {0, …, n-1}
        n:         vocab size (last token = MASK)
    Returns:
        (B, d, n)  transition probabilities
    """
    B, d = x0.shape
    sb = sigma_bar[:, None, None]                     # (B,1,1)
    stay = torch.exp(-sb)                             # prob of no change
    probs = torch.zeros(B, d, n, device=x0.device, dtype=sb.dtype)
    probs.scatter_(2, x0.unsqueeze(-1), stay.expand(B, d, 1))
    probs[:, :, n - 1] += (1.0 - stay).squeeze(-1)   # go to MASK
    return probs


def _forward_probs_uniform(sigma_bar: torch.Tensor, x0: torch.Tensor, n: int):
    """
    Uniform diffusion (Eq 15).

    p_{t|0}(y | x₀) = (1 - e^{-σ̄})/n   ∀y
                     + e^{-σ̄}           additionally when y = x₀

    (Algorithm 1, "else if Q is Uniform" branch.)
    """
    B, d = x0.shape
    sb = sigma_bar[:, None, None]
    unif = (1.0 - torch.exp(-sb)) / n                # (B,1,1)
    probs = unif.expand(B, d, n).clone()
    probs.scatter_add_(2, x0.unsqueeze(-1),
                       torch.exp(-sb).expand(B, d, 1))
    return probs

def forward_transition(sigma_bar, x0, n, mode):
    if mode == "absorb":
        return _forward_probs_absorb(sigma_bar, x0, n)
    return _forward_probs_uniform(sigma_bar, x0, n)

def sample_xt(x0, sigma_bar, n, mode):
    """Sample x_t ~ p_{t|0}(·|x₀) per token (Eq 13)."""
    probs = forward_transition(sigma_bar, x0, n, mode)
    flat = probs.reshape(-1, n)
    xt = torch.multinomial(flat, 1).reshape(x0.shape)
    return xt, probs


# ============================================================
# Algorithm 1 — Score Entropy Training Loss  (L̂_DWDSE)
# ============================================================
# Full score entropy (Eq 5) including K(r) = r(log r − 1):
#   L = Σ_{y≠x} [ s_y − r_y · log(s_y) + K(r_y) ]
#     = Σ_{y≠x, r>0} r_y · (exp(log_s − log_r) − 1 − (log_s − log_r))
#     + Σ_{y≠x, r=0} exp(log_s_y)
#
# This is always ≥ 0 (Bregman divergence of exp), which makes
# the loss interpretable and the EMA meaningful.

def _log_expm1(x):
    """Compute log(exp(x) - 1) stably."""
    return torch.where(x > 20.0, x, torch.log(torch.expm1(x.clamp(max=20.0))))


def score_entropy_loss(
    score_net: ScoreFn,
    x0: torch.Tensor,
    vocab_size: int,
    schedule:Any,
    mode: Literal["absorb", "uniform"] = "absorb",
    t: Optional[torch.Tensor] = None,
):
    """
    One training step of Algorithm 1, computed in log-space.

    The score network must return **log-scores** (not scores).
    Loss includes K(r) so it is always non-negative.

    Args:
        score_net: (x_t, σ̄) → (B, d, n) log-scores
        x0:        (B, d)  clean data
        n:         vocab size
        schedule:  noise schedule object
        mode:      "absorb" | "uniform"
        t:         (B,) optional pre-sampled times; else U[0,1]
    Returns:
        scalar loss (mean over batch)
    """
    B, d = x0.shape
    device = x0.device

    if t is None:
        t = torch.rand(B, device=device)

    sb = schedule.sigma_bar(t)                         # σ̄(t)   (B,)
    st = schedule.sigma(t)                             # σ(t)    (B,)

    # --- construct x_t (forward noising) ---
    xt, probs = sample_xt(x0, sb, vocab_size, mode)            # xt (B,d), probs (B,d,n)

    # --- log-scores from network ---
    log_scores = score_net(xt, sb)                     # (B, d, n)

    # --- target ratios  r_y = p_{t|0}(y|x₀) / p_{t|0}(x_t|x₀) ---
    p_xt = probs.gather(2, xt.unsqueeze(-1))           # (B, d, 1)
    ratios = probs / p_xt.clamp(min=1e-30)             # (B, d, n)

    # --- mask diagonal  y ≠ x_t ---
    mask = torch.ones_like(log_scores)
    mask.scatter_(2, xt.unsqueeze(-1), 0.0)

    # --- loss in log-space, always ≥ 0 ---
    # For r > 0:  r · (exp(log_s − log_r) − 1 − (log_s − log_r))
    # For r = 0:  exp(log_s)
    has_r = (probs > 1e-30) & (mask > 0.5)
    log_r = torch.log(ratios.clamp(min=1e-30))
    u = log_scores - log_r                             # log(s/r)
    u_clamped = u.clamp(max=50.0)                      # prevent exp overflow
    loss_pos = ratios * (torch.expm1(u_clamped) - u)   # ≥ 0

    loss_zero = torch.exp(log_scores.clamp(max=50.0))  # ≥ 0

    loss_entries = torch.where(has_r, loss_pos, loss_zero) * mask

    # --- weight by σ(t) and average ---
    loss = loss_entries.sum(dim=(-1, -2)) * st          # (B,)
    return loss.mean()


# ============================================================
# Conditional Training Loss  (SFT on Q/A pairs)
# ============================================================
# For supervised fine-tuning with prompt/answer pairs:
#   - Question tokens stay clean (matching Algorithm 3 at inference)
#   - Only answer tokens are noised
#   - Loss computed only on answer positions
#
# By Eq 22, conditional and unconditional scores coincide at
# unfilled positions, so this trains exactly the scores needed
# for conditional sampling.

def conditional_score_entropy_loss(
    score_net: ScoreFn,
    x0: torch.Tensor,
    n: int,
    schedule,
    answer_mask: torch.Tensor,
    mode: Literal["absorb", "uniform"] = "absorb",
    t: Optional[torch.Tensor] = None,
):
    """
    Score entropy loss for Q/A fine-tuning.

    Only answer positions (where answer_mask is True) are noised
    and contribute to the loss.  Question positions stay clean,
    matching Algorithm 3 at inference.

    Args:
        score_net:   (x_t, σ̄) → (B, d, n) log-scores
        x0:          (B, d)  full sequence [question | answer]
        n:           vocab size
        schedule:    noise schedule object
        answer_mask: (B, d)  bool — True for answer positions
        mode:        "absorb" | "uniform"
        t:           (B,) optional pre-sampled times; else U[0,1]
    Returns:
        scalar loss (mean over batch)
    """
    B, d = x0.shape
    device = x0.device

    if t is None:
        t = torch.rand(B, device=device)

    sb = schedule.sigma_bar(t)
    st = schedule.sigma(t)

    # --- forward transition probs for all positions ---
    probs = forward_transition(sb, x0, n, mode)          # (B, d, n)

    # --- noise only answer positions, keep questions clean ---
    xt = x0.clone()
    noised = torch.multinomial(probs.reshape(-1, n), 1).reshape(B, d)
    xt[answer_mask] = noised[answer_mask]

    # --- log-scores from network ---
    log_scores = score_net(xt, sb)

    # --- target ratios (only meaningful at answer positions) ---
    p_xt = probs.gather(2, xt.unsqueeze(-1))
    ratios = probs / p_xt.clamp(min=1e-30)

    # --- mask: y ≠ x_t  AND  position is answer ---
    mask = torch.ones_like(log_scores)
    mask.scatter_(2, xt.unsqueeze(-1), 0.0)
    mask = mask * answer_mask.unsqueeze(-1).float()

    # --- log-space loss (same as score_entropy_loss) ---
    has_r = (probs > 1e-30) & (mask > 0.5)
    log_r = torch.log(ratios.clamp(min=1e-30))
    u = log_scores - log_r
    u_clamped = u.clamp(max=50.0)
    loss_pos = ratios * (torch.expm1(u_clamped) - u)
    loss_zero = torch.exp(log_scores.clamp(max=50.0))

    loss_entries = torch.where(has_r, loss_pos, loss_zero) * mask

    loss = loss_entries.sum(dim=(-1, -2)) * st
    return loss.mean()


# ============================================================
# ELBO evaluation  (Theorem 3.6, Appendix C.6)
# ============================================================
# −log p₀^θ(x₀) ≤ L_DWDSE(x₀) + D_KL(p_{T|0}(·|x₀) ‖ π)
# Monte-Carlo over 1000 random timesteps as in the paper.

@torch.no_grad()
def estimate_elbo(
    score_net: ScoreFn,
    x0: torch.Tensor,
    n: int,
    schedule,
    mode: Literal["absorb", "uniform"] = "absorb",
    num_t: int = 1000,
):
    """
    Estimate −ELBO (upper bound on NLL) for a batch of data.

    Score network must return log-scores.  Returns per-sample
    values in nats.
    """
    B, d = x0.shape
    device = x0.device

    ts = torch.rand(num_t, device=device)
    total = torch.zeros(B, device=device)

    for ti in ts:
        t_batch = ti.expand(B)
        sb = schedule.sigma_bar(t_batch)
        st = schedule.sigma(t_batch)

        xt, probs = sample_xt(x0, sb, n, mode)
        log_scores = score_net(xt, sb)

        p_xt = probs.gather(2, xt.unsqueeze(-1))
        ratios = probs / p_xt.clamp(min=1e-30)

        mask = torch.ones_like(log_scores)
        mask.scatter_(2, xt.unsqueeze(-1), 0.0)

        has_r = (probs > 1e-30) & (mask > 0.5)
        log_r = torch.log(ratios.clamp(min=1e-30))
        u = (log_scores - log_r).clamp(max=50.0)
        loss_pos = ratios * (torch.expm1(u) - (log_scores - log_r))
        loss_zero = torch.exp(log_scores.clamp(max=50.0))
        entry = torch.where(has_r, loss_pos, loss_zero) * mask
        total += entry.sum(dim=(-1, -2)) * st

    integral = total / num_t

    # --- prior KL ---
    sb_T = schedule.sigma_bar(torch.ones(B, device=device))
    probs_T = forward_transition(sb_T, x0, n, mode)

    if mode == "absorb":
        log_pi = torch.full((n,), -float('inf'), device=device)
        log_pi[n - 1] = 0.0
    else:
        log_pi = torch.full((n,), -math.log(n), device=device)

    kl = (probs_T * (torch.log(probs_T.clamp(min=1e-30)) - log_pi)).sum(-1).sum(-1)
    return integral + kl


# ============================================================
# Algorithm 2 — Unconditional Sampling
# ============================================================
# Two strategies: Euler (Eq 17) and Tweedie τ-leaping (Eq 19).
# Both reverse from t=1 (base) to t=0 (data).

def _sample_base(B, d, n, mode, device):
    """x_T ∼ p_base.  Absorb → all MASK;  Uniform → random."""
    if mode == "absorb":
        return torch.full((B, d), n - 1, dtype=torch.long, device=device)
    return torch.randint(0, n, (B, d), device=device)


def _euler_transition(xt, scores, sigma_t, dt, n, mode):
    """
    Euler reverse step (Eq 17).

    p^i(y | x_t^i) = δ(y, x_t^i)
        + Δt · σ(t) · Q^{tok}(x_t^i, y) · s_θ(x_t, t)_{i,y}

    Q(x,y) is row-x col-y of the forward rate matrix, i.e. the
    forward rate of probability flowing *from state y to state x*.
    The reverse rate from x to y is  s_θ_y · Q(x,y) · σ(t).
    But here y is the destination in reverse, so the off-diagonal
    reverse-transition prob is  Δt · σ(t) · Q(x_t, y) · s_θ_y.

    Uniform:  Q(x,y)=1 for x≠y  →  rate = σ·s_y  for every y≠x.
    Absorb:   Q(MASK,y)=1 for y<MASK; all other off-diag = 0
              → only MASK tokens unmask; non-MASK tokens stay.
    """
    B, d, _ = scores.shape
    probs = F.one_hot(xt, n).float()                       # (B,d,n)
    rate = sigma_t[:, None, None] * scores * dt            # (B,d,n)

    if mode == "uniform":
        rate.scatter_(2, xt.unsqueeze(-1), 0.0)            # zero diagonal
        probs = probs + rate

    elif mode == "absorb":
        # Only MASK tokens get reverse transitions
        is_mask = (xt == n - 1).unsqueeze(-1).float()      # (B,d,1)
        rate[:, :, n - 1] = 0.0                            # no MASK→MASK
        probs = probs + rate * is_mask

    return probs


def _tweedie_transition(xt, scores, alpha, n, mode):
    """
    Tweedie τ-leaping step (Eq 19, Theorem 4.2).

    p^i(y | x_t^i) =
        [ exp(-α Q) · s_θ ]_y  ×  exp(α Q)_{x_t^i, y}

    where α = σ_t^{Δt} = σ̄(t) − σ̄(t−Δt) > 0.

    Closed forms for exp(±α Q):
      Uniform  eigenvalues 0, −1 (after absorbing n into σ̄):
        exp(αQ)_{x,y}  = e^{-α}δ_{xy} + (1−e^{-α})/n
        exp(−αQ)_{x,y} = e^{α}δ_{xy}  + (1−e^{α})/n

      Absorb  (MASK = n−1):
        exp(αQ): col y<M → row y: e^{-α}, row M: 1−e^{-α}
                 col M   → row M: 1
        exp(−αQ): same structure with −α.
    """
    B, d, _ = scores.shape
    a = alpha[:, None, None]                               # (B,1,1)
    ea  = torch.exp(a)                                     # e^α
    ema = torch.exp(-a)                                    # e^{-α}

    if mode == "uniform":
        # --- left factor:  [exp(−αQ) · s]_y ---
        s_sum = scores.sum(dim=-1, keepdim=True)           # (B,d,1)
        left = ea * scores + (1.0 - ea) / n * s_sum       # (B,d,n)

        # --- right factor:  exp(αQ)_{x_t, y} ---
        right = ((1.0 - ema) / n).expand(B, d, n).clone()
        right.scatter_add_(2, xt.unsqueeze(-1),
                           ema.expand(B, d, 1))
        probs = left * right

    elif mode == "absorb":
        # Non-MASK tokens always stay (derivation in text).
        # MASK tokens:
        #   y < n-1:  e^α · s_y · (1 − e^{-α})
        #   y = MASK: (1−e^α)·Σ_{z<M} s_z  +  s_{MASK}

        trans_mask = torch.zeros_like(scores)
        trans_mask[:, :, :n - 1] = (
            ea * scores[:, :, :n - 1] * (1.0 - ema)
        )
        s_nonmask_sum = scores[:, :, :n - 1].sum(-1, keepdim=True)
        trans_mask[:, :, n - 1:] = (
            (1.0 - ea) * s_nonmask_sum + scores[:, :, n - 1:]
        )

        trans_stay = F.one_hot(xt, n).float()
        is_mask = (xt == n - 1).unsqueeze(-1).float()
        probs = is_mask * trans_mask + (1.0 - is_mask) * trans_stay

    return probs

def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)

def sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
    """Sample indices from a categorical distribution via Gumbel-argmax."""
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)

def _clamp_and_sample(probs):
    """Clamp negatives, normalise, sample (Algorithm 2 post-processing)."""
    probs = probs.clamp(min=0.0)
    probs = probs / (probs.sum(-1, keepdim=True) + 1e-20)
    return sample_categorical(probs)
    #return torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1
    #                         ).reshape(probs.shape[:-1])


@torch.no_grad()
def sample(
    score_net: ScoreFn,
    batch: int,
    d: int,
    n: int,
    schedule,
    num_steps: int = 256,
    method: Literal["euler", "tweedie"] = "tweedie",
    mode: Literal["absorb", "uniform"] = "absorb",
    device: torch.device = torch.device("cpu"),
):
    """
    Algorithm 2: unconditional sampling.

    Reverses from t=1 → t=0 in `num_steps` uniform steps.
    """
    dt = 1.0 / num_steps
    xt = _sample_base(batch, d, n, mode, device)

    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t_vec = torch.full((batch,), t_val, device=device)
        sb = schedule.sigma_bar(t_vec)

        scores = torch.exp(score_net(xt, sb).clamp(max=50.0))  # log→scores

        if method == "euler":
            st = schedule.sigma(t_vec)
            probs = _euler_transition(xt, scores, st, dt, n, mode)
        else:
            t_prev = torch.full((batch,), max(t_val - dt, 0.0), device=device)
            alpha = sb - schedule.sigma_bar(t_prev)          # σ̄(t)−σ̄(t−Δt)
            probs = _tweedie_transition(xt, scores, alpha, n, mode)

        xt = _clamp_and_sample(probs)

    return xt


# ============================================================
# Algorithm 3 — Conditional Sampling  (infilling / prompting)
# ============================================================
# By Bayes' rule (Eq 22) the conditional and unconditional scores
# coincide when we only modify tokens at unfilled positions Ω.
# So we run normal reverse diffusion but freeze prompt positions.

@torch.no_grad()
def sample_conditional(
    score_net: ScoreFn,
    projector: Callable[[torch.Tensor], torch.Tensor],
    batch: int,
    seq_len: int,
    vocab_size: int,
    schedule,
    num_steps: int = 256,
    method: Literal["euler", "tweedie"] = "tweedie",
    mode: Literal["absorb", "uniform"] = "absorb",
    device: torch.device = torch.device("cpu"),
):
    """
    Algorithm 3: conditional sampling.

    Positions listed in `prompt_indices` are clamped to `prompt_tokens`.
    All other positions are generated via the reverse process.

    Args:
        project: Projector, this function should write in tokens that are given.
    Returns:
        (B, d) generated sequences with prompts inserted
    """
    dt = 1.0 / num_steps
    xt = _sample_base(batch, seq_len, vocab_size, mode, device)

    # Ω = prompt positions (fixed);  Ω̄ = free positions
    xt = projector(xt)

    for step in range(num_steps):
        t_val = 1.0 - step * dt
        t_vec = torch.full((batch,), t_val, device=device)
        sb = schedule.sigma_bar(t_vec)

        scores = torch.exp(score_net(xt, sb).clamp(max=50.0))  # log→scores

        if method == "euler":
            st = schedule.sigma(t_vec)
            probs = _euler_transition(xt, scores, st, dt, vocab_size, mode)
        else:
            t_prev = torch.full((batch,), max(t_val - dt, 0.0), device=device)
            alpha = sb - schedule.sigma_bar(t_prev)
            probs = _tweedie_transition(xt, scores, alpha, vocab_size, mode)

        xt = projector(_clamp_and_sample(probs))

    return xt

# ============================================================
# API
# ============================================================
Dimensions = Tuple[int, int]

@dataclass
class Outlet:
    mode:       Literal["absorb", "uniform"] = "absorb"
    def __call__(self, log_scores, sigma_bar):
        if self.mode == "absorb":
            log_scores = log_scores + _log_expm1(sigma_bar)[:, None, None]
        return log_scores

@dataclass
class Sampler:
    vocab_size: int
    schedule:   Any = LogLinearSchedule()
    method:     Literal["euler", "tweedie"] = "tweedie"
    mode:       Literal["absorb", "uniform"] = "absorb"

    def base(self, dimensions : Dimensions, device:torch.device=torch.device("cpu")):
        B, L = dimensions
        return _sample_base(B, L, self.vocab_size, self.mode, device)

    def transition(self, score_fn: ScoreFn, xt, step, num_steps, add_noise=False):
        batch = xt.shape[0]
        dt = 1.0 / num_steps
        t_val = 1.0 - step * dt
        t_vec = torch.full((batch,), t_val, device=xt.device)
        sb = self.schedule.sigma_bar(t_vec)

        if add_noise:
            probs = forward_transition(sb, xt, self.vocab_size, self.mode)
            xt    = _clamp_and_sample(probs)

        scores = torch.exp(score_fn(xt, sb).clamp(max=50.0))  # log→scores

        if self.method == "euler":
            st = self.schedule.sigma(t_vec)
            probs = _euler_transition(xt, scores, st, dt, self.vocab_size, self.mode)
        else:
            t_prev = torch.full((batch,), max(t_val - dt, 0.0), device=xt.device)
            alpha = sb - self.schedule.sigma_bar(t_prev)
            probs = _tweedie_transition(xt, scores, alpha, self.vocab_size, self.mode)

        return _clamp_and_sample(probs)

    def score_entropy_loss(self, score_fn, x0, t=None, mask=None):
        B, d = x0.shape
        device = x0.device
        if t is None:
            t = torch.rand(B, device=device)
        sb = self.schedule.sigma_bar(t)                         # σ̄(t)   (B,)
        st = self.schedule.sigma(t)                             # σ(t)    (B,)
        xt, probs = sample_xt(x0, sb, self.vocab_size, self.mode)
        if mask is not None:
            xt = torch.where(mask, xt, x0)
        # --- log-scores from network ---
        log_scores = score_fn(xt, sb)
        # --- target ratios (only meaningful at answer positions) ---
        p_xt = probs.gather(2, xt.unsqueeze(-1))
        ratios = probs / p_xt.clamp(min=1e-30)
        # --- mask: y ≠ x_t  AND  position is answer ---
        eq_mask = torch.ones_like(log_scores)
        eq_mask.scatter_(2, xt.unsqueeze(-1), 0.0)
        if mask is not None:
            eq_mask = eq_mask * mask.unsqueeze(-1).float()
        # --- loss in log-space, always ≥ 0 ---
        # For r > 0:  r · (exp(log_s − log_r) − 1 − (log_s − log_r))
        # For r = 0:  exp(log_s)
        has_r = (probs > 1e-30) & (eq_mask > 0.5)
        log_r = torch.log(ratios.clamp(min=1e-30))
        u = log_scores - log_r                             # log(s/r)
        u_clamped = u.clamp(max=50.0)                      # prevent exp overflow
        loss_pos = ratios * (torch.expm1(u_clamped) - u)   # ≥ 0
        loss_zero = torch.exp(log_scores.clamp(max=50.0))  # ≥ 0
        loss_entries = torch.where(has_r, loss_pos, loss_zero) * eq_mask
        # --- weight by σ(t) and average ---
        loss = loss_entries.sum(dim=(-1, -2)) * st          # (B,)
        return loss.mean()

def example(score_fn, dimensions, num_steps):
    sampler = Sampler(257)

    xt = sampler.base(dimensions)
    # project here if you like
    for step in range(num_steps):
        xt = sampler.transition(score_fn, xt, step, num_steps)
        # project here if you like

    # Add an answer mask if you're doing supervised finetuning.
    loss = sampler.score_entropy_loss(score_fn, x0, mask=None)

# ============================================================
# This is left here for completeness.
# ============================================================
# Score Transformer  (Section 5.1, Appendix C.2)
# ============================================================
# DiT-style encoder-only transformer (Peebles & Xie, 2023):
#   - adaLN-zero time conditioning on σ̄(t)  (not t itself)
#   - rotary positional embeddings  (Su et al., 2021)
#   - separate input embedding and output projection matrices
#   - output exponentiated for positivity; scaled by (e^σ̄ − 1)
#     for absorb  (Appendix C.2)

def _apply_rotary(x, cos, sin):
    """Apply rotary embedding to x: (B, H, L, D)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class _SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)          # 3 × (B,H,L,D)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)   # (B,H,L,D)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(out)


class _AdaLNBlock(nn.Module):
    """Transformer block with adaLN-zero conditioning (DiT)."""
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = _SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim))
        # 6 modulation parameters: (γ1, β1, α1, γ2, β2, α2)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x, c, cos, sin):
        g1, b1, a1, g2, b2, a2 = self.adaln(c).unsqueeze(1).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + g1) + b1
        x = x + a1 * self.attn(h, cos, sin)
        h = self.norm2(x) * (1 + g2) + b2
        x = x + a2 * self.mlp(h)
        return x


class ScoreTransformer(nn.Module):
    """
    Small DiT-style score network for SEDD (Section 5.1, Appendix C.2).

    The network is conditioned on σ̄(t) (not t) via sinusoidal
    embeddings fed through adaLN-zero.  Outputs are exponentiated
    for positivity and optionally scaled by (e^σ̄ − 1) for absorb.
    """
    def __init__(self, n: int, max_len: int, dim: int = 256,
                 num_heads: int = 4, num_layers: int = 4,
                 mode: Literal["absorb", "uniform"] = "absorb"):
        super().__init__()
        self.n = n
        self.dim = dim
        self.mode = mode

        self.tok_embed = nn.Embedding(n, dim)
        self.out_proj = nn.Linear(dim, n)
        # Initialise output bias negative so exp(logit) starts small
        nn.init.zeros_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, -6.0)

        self.time_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(
            [_AdaLNBlock(dim, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)

        # Precompute rotary frequencies
        head_dim = dim // num_heads
        half = head_dim // 2
        freqs = 1.0 / (10000.0 ** (torch.arange(half).float() / half))
        pos = torch.arange(max_len).float()
        angles = pos[:, None] * freqs[None, :]
        self.register_buffer("rot_cos", torch.cos(angles), persistent=False)
        self.register_buffer("rot_sin", torch.sin(angles), persistent=False)

    def _time_embed(self, sigma_bar):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=sigma_bar.device) / half)
        args = sigma_bar[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(emb)                                 # (B, dim)

    def forward(self, xt: torch.Tensor, sigma_bar: torch.Tensor):
        """
        Returns **log-scores** (not scores).

        log s_θ = logits + log(e^σ̄ − 1)   for absorb
        log s_θ = logits                    for uniform
        """
        B, L = xt.shape
        h = self.tok_embed(xt)                                    # (B,L,D)
        c = self._time_embed(sigma_bar)                           # (B,D)

        cos = self.rot_cos[:L][None, None, :, :]                  # (1,1,L,half)
        sin = self.rot_sin[:L][None, None, :, :]

        for block in self.blocks:
            h = block(h, c, cos, sin)
        h = self.final_norm(h)
        log_scores = self.out_proj(h)                             # (B,L,n)

        # Absorb scaling in log-space: log(e^σ̄ − 1), stable (Appendix C.2)
        if self.mode == "absorb":
            log_scores = log_scores + _log_expm1(sigma_bar)[:, None, None]
        return log_scores

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    n, d, B = 257, 128, 32
    mode = "absorb"

    dataloader = create_dataloader(
            load_kalevala(),
            text_to_tensor,
            batch_size=B,
            length=d,
            drop_last=True)

    from .ema import EMA

    #from .srlm import SRLMConfig, SRLM, ScoreTransformer, make_z

    #net = SRLM(SRLMConfig(
    #        vocab_size = n,
    #        context_length = 128,
    #        d_model=256,
    #        n_priors=3,
    #        n_posteriors=2,
    #        n_heads=4)).to(device)

    #@dataclass
    #class Sideways:
    #    model : SRLM
    #    z : tuple[torch.Tensor, torch.Tensor]
    #    def __call__(self, x, sigma_bar):
    #        self.z, log_score = self.model.sideways(self.z, x, sigma_bar)
    #        return log_score

    net = ScoreTransformer(
        n=n, max_len=d, dim=256, num_heads=4, num_layers=4, mode=mode
    ).to(device)
    ema = EMA(net)
    param_count = sum(p.numel() for p in net.parameters())
    print(f"model params: {param_count:,}")

    sampler = Sampler(n)
    
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.1)
    ema_loss = None
    step = 0

    for epoch in range(200):
        for k, (x0, _) in enumerate(dataloader):
            net.train()
            x0 = x0.to(device)
            optimizer.zero_grad()
            loss = sampler.score_entropy_loss(net, x0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            ema.update(net)
            #side = Sideways(net, make_z(B,d,net.cfg.d_model, device=device))
            #num_steps = 4
            #supervision_loss = 0
            #for step_i in range(num_steps):
                #t = 1.0 - step_i * (1.0 / num_steps)
                #t *= torch.ones(B, device=device)
            #    optimizer.zero_grad()
            #    loss = sampler.score_entropy_loss(side, x0)
            #    loss.backward()
            #    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            #    optimizer.step()
            #    supervision_loss += loss.detach()

            lv = loss.item() # / num_steps
            ema_loss = lv if ema_loss is None else 0.95 * ema_loss + 0.05 * lv
            step += 1
            if step % 20 == 0:
                print(f"step {step:5d}  loss {lv:10.2f}  ema {ema_loss:10.2f}")

            if step % 200 == 0:
                net.eval()
                ema.apply(net)
                #side = Sideways(net, make_z(2,d,net.cfg.d_model, device=device))
                xt = sampler.base((2,d), device)
                for s in range(128):
                    xt = sampler.transition(net, xt, s, 128, add_noise=True)
                print(f"  sample 1: {repr(as_text(xt[0]))}")
                print(f"  sample 2: {repr(as_text(xt[1]))}")
                xt = sampler.base((2,d), device)
                for s in range(128):
                    xt = sampler.transition(net, xt, s, 128, add_noise=False)
                print(f"  sample X: {repr(as_text(xt[0]))}")
                print(f"  sample Y: {repr(as_text(xt[1]))}")
                ema.restore(net)
