import torch
import torch.nn.functional as F
from .sedd import score_entropy_loss, conditional_score_entropy_loss, LogLinearSchedule, sample_xt, forward_transition


def deep_supervision_step(model, optimizer, schedule, z, batch,
                          n_supervision=4, perturbed_batch=None, memories=None,
                          grad_clip=0.1, ema=None):
    """TRM-style deep supervision for SEDD.

    Calls model.front() + model.step() n_supervision times, with an
    optimizer update after each step. Q_head learns to predict whether
    the output is correct (halting signal for adaptive computation).
    EMA shadow weights are updated after each optimizer step for stability.

    Args:
        model: SRLM with front()/step() API
        optimizer: optimizer to step
        schedule: noise schedule (e.g. LogLinearSchedule)
        z: HRM state tuple
        batch: (B, L) clean token IDs
        n_supervision: number of deep supervision steps
        perturbed_batch: optional pre-masked batch
        memories: optional memory tensors
        grad_clip: gradient clipping norm
        ema: optional EMA instance -- updated after each optimizer step
    Returns:
        avg_loss: average SEDD loss across supervision steps
        z: final HRM state (detached)
    """
    device = batch.device
    B, d = batch.shape
    vocab_size = model.cfg.vocab_size
    MASK_TOKEN = vocab_size - 1

    # Sample noise once -- same input across all supervision steps.
    t = torch.rand(B, device=device)
    sb = schedule.sigma_bar(t)
    st = schedule.sigma(t)

    # Forward noising
    xt, probs = sample_xt(batch, sb, vocab_size, "absorb")

    if perturbed_batch is not None:
        # Pre-masked batch (arithmetic, sudoku): protect visible (non-mask) positions
        visible = (perturbed_batch != MASK_TOKEN)
        xt = torch.where(visible, perturbed_batch, xt)
        # Recompute probs for the actual xt we'll use
        probs = forward_transition(sb, batch, vocab_size, "absorb")

    # Precompute loss ingredients that don't change across supervision steps
    p_xt = probs.gather(2, xt.unsqueeze(-1))
    ratios = probs / p_xt.clamp(min=1e-30)
    eq_mask = torch.ones(B, d, vocab_size, device=device)
    eq_mask.scatter_(2, xt.unsqueeze(-1), 0.0)
    if perturbed_batch is not None:
        answer_mask = (perturbed_batch == MASK_TOKEN).float()
        eq_mask = eq_mask * answer_mask.unsqueeze(-1)
    has_r = (probs > 1e-30) & (eq_mask > 0.5)
    log_r = torch.log(ratios.clamp(min=1e-30))

    total_loss = 0.0
    n_steps = 0

    for step_i in range(n_supervision):
        optimizer.zero_grad()

        # Recompute front() each step -- model weights change after optimizer.step()
        ix = model.front(xt, sb, memories=memories)
        z, log_scores, q, aux_loss = model.step(z, ix)

        # Compute SEDD loss using precomputed ratios
        u = log_scores - log_r
        u_clamped = u.clamp(max=50.0)
        loss_pos = ratios * (torch.expm1(u_clamped) - u)
        loss_zero = torch.exp(log_scores.clamp(max=50.0))
        loss_entries = torch.where(has_r, loss_pos, loss_zero) * eq_mask
        sedd_loss = (loss_entries.sum(dim=(-1, -2)) * st).mean()

        # Q_head BCE loss: predict whether output matches ground truth
        with torch.no_grad():
            preds = log_scores.argmax(dim=-1)                   # (B, L)
            accuracy = (preds == batch).float().mean(dim=-1)   # (B,)
            q_target = (accuracy > 0.5).float()
        q_loss = F.binary_cross_entropy_with_logits(q.squeeze(-1), q_target)

        loss = sedd_loss + q_loss*20 + aux_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # EMA: smooth shadow weights toward current weights
        if ema is not None:
            ema.update(model)

        total_loss += sedd_loss.item()
        n_steps += 1

        # Early stopping: Q > 0 means "output is good enough, stop recursing"
        with torch.no_grad():
            if n_supervision > 1 and (q.squeeze(-1) > 0).all():
                break

    return total_loss / n_steps, tuple(zi.detach() for zi in z)
