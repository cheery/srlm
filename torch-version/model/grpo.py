"""
DGPO-style GRPO for discrete diffusion (SEDD).

Based on "Reinforcing Diffusion Models by Direct Group Preference Optimization"
(Luo et al., 2025), adapted for absorbing-state discrete diffusion.

Flow:
1. Generate K candidates per prompt via reverse diffusion
2. Score each with reward_fn
3. Compute group-relative z-score advantages
4. DGPO loss: sigmoid-weighted advantage * score_entropy
   - Reference model = frozen weights before optimization
   - Sigmoid weight adapts based on model drift from reference
"""

import torch
import torch.nn.functional as F
from .sedd import (
    sample_conditional, sample_xt, forward_transition,
    LogLinearSchedule, _clamp_and_sample,
)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def arithmetic_reward(tokens):
    """Check if an arithmetic expression N+M=Y is correct.

    Args:
        tokens: 1-D int tensor of token IDs (byte values)
    Returns:
        float reward: 1.0 if correct, 0.0 if wrong/unparseable
    """
    try:
        text = bytes(tokens.clamp(0, 255).cpu().tolist()).decode("utf-8", errors="replace").strip()
        # Find the pattern: N+M=Y
        if '=' not in text or '+' not in text:
            return 0.0
        lhs, rhs = text.split('=', 1)
        # rhs might have trailing spaces/garbage -- take leading digits
        rhs = rhs.strip()
        rhs_digits = ''
        for ch in rhs:
            if ch.isdigit():
                rhs_digits += ch
            else:
                break
        if not rhs_digits:
            return 0.0
        parts = lhs.split('+', 1)
        if len(parts) != 2:
            return 0.0
        n = int(parts[0].strip())
        m = int(parts[1].strip())
        y = int(rhs_digits)
        correct = n + m
        if y == correct:
            return 1.0
        # Partial reward: 0.1 for producing any number after =,
        # plus up to 0.9 for closeness
        error = abs(y - correct)
        closeness = max(0.0, 1.0 - error / max(correct, 1))
        return 0.1 + 0.9 * closeness
    except (ValueError, OverflowError):
        return 0.0


def sudoku_reward(generated_tokens, puzzle_tokens=None):
    """Score a sudoku solution by constraint satisfaction.

    Checks row, column, and box uniqueness (27 constraints).
    Returns fraction of constraints satisfied (0.0 to 1.0).

    Args:
        generated_tokens: 1-D int tensor (89 bytes: 9 rows + 8 newlines)
        puzzle_tokens:    unused (kept for API compatibility)
    Returns:
        float reward: 0.0 to 1.0
    """
    try:
        text = bytes(generated_tokens[:89].clamp(0, 255).cpu().tolist()).decode("utf-8", errors="replace")
        lines = text.split('\n')
        if len(lines) != 9:
            return 0.0
        grid = []
        for line in lines:
            row = [int(ch) for ch in line[:9] if ch.isdigit()]
            if len(row) != 9:
                return 0.0
            grid.append(row)

        score = 0.0
        checks = 0
        # Row uniqueness
        for r in range(9):
            digits = [d for d in grid[r] if 1 <= d <= 9]
            checks += 1
            if len(digits) == len(set(digits)) == 9:
                score += 1.0
        # Column uniqueness
        for c in range(9):
            digits = [grid[r][c] for r in range(9) if 1 <= grid[r][c] <= 9]
            checks += 1
            if len(digits) == len(set(digits)) == 9:
                score += 1.0
        # Box uniqueness
        for br in range(3):
            for bc in range(3):
                digits = []
                for r in range(br*3, br*3+3):
                    for c in range(bc*3, bc*3+3):
                        if 1 <= grid[r][c] <= 9:
                            digits.append(grid[r][c])
                checks += 1
                if len(digits) == len(set(digits)) == 9:
                    score += 1.0
        return score / checks  # 0.0 to 1.0
    except (ValueError, IndexError):
        return 0.0


# ---------------------------------------------------------------------------
# DGPO-style GRPO step
# ---------------------------------------------------------------------------

def grpo_step(
    model,
    optimizer,
    schedule,
    prompt_batch,        # (B, L) int tensor -- prompts with masks
    clean_batch,         # (B, L) int tensor -- ground truth tokens
    reward_fn,           # callable(generated_tokens) -> float
    z,                   # HRM state
    device,
    verbose=False,       # print prompts, samples, rewards
    memories=None,       # optional memory tensors for model
    K=4,                 # candidates per prompt
    sampling_steps=50,   # diffusion steps for generation
    grad_clip=1.0,
    epochs=5,
    beta_dgpo=1.0,       # DGPO sigmoid temperature
    fused=False,          # fused backward: faster but more VRAM
    max_act_steps=8,     # adaptive computation steps per diffusion step
):
    """DGPO-style GRPO update for discrete diffusion.

    1. Generate K candidates per prompt via reverse diffusion
    2. Score each with reward_fn
    3. Compute group-relative z-score advantages
    4. Cache reference score_entropy (model before optimization)
    5. Optimize with DGPO loss: sigma(group_drift) * advantage * score_entropy
    """
    B, L = prompt_batch.shape
    d_model = z[0].shape[-1]
    vocab_size = model.cfg.vocab_size
    MASK_TOKEN = vocab_size - 1

    prompt_batch = prompt_batch.to(device)
    clean_batch = clean_batch.to(device)
    visible = (prompt_batch != MASK_TOKEN)

    model.eval()

    # --- 1. Generate K candidates per prompt via diffusion ---
    prompt_expanded = prompt_batch.repeat_interleave(K, dim=0)
    visible_mask = (prompt_expanded != MASK_TOKEN)
    visible_values = prompt_expanded.clone()

    def projector(x):
        return torch.where(visible_mask, visible_values, x)

    memories_expanded = None
    if memories is not None:
        memories_expanded = [m.repeat_interleave(K, dim=0) for m in memories]

    def sample_fn(z_gen, ix, max_act_steps):
        z_gen, y, q = model.recur(z_gen, ix)
        for i in range(1, max_act_steps):
            if i % 4 == 0 and (q.squeeze(-1) > 0).all():
                break
            z_gen, y, q = model.recur(z_gen, ix)
        log_score, _ = model.head(y, ix)
        return log_score

    score_fn_gen = lambda xt, sb: sample_fn(
        make_z_for_grpo(xt.shape[0], L, d_model, device),
        model.front(xt, sb, memories=memories_expanded),
        max_act_steps,
    )

    with torch.no_grad():
        generated = sample_conditional(
            score_fn_gen, projector,
            batch=B * K, seq_len=L, vocab_size=vocab_size,
            schedule=schedule, num_steps=sampling_steps,
            device=device,
        )

    generated = projector(generated)

    # --- 2. Score candidates with reward_fn ---
    rewards = torch.zeros(B * K, device=device)
    for i in range(B * K):
        rewards[i] = reward_fn(generated[i])

    if verbose:
        def _tok2str(t):
            return bytes(t.clamp(0, 255).cpu().tolist()).decode("utf-8", errors="replace").rstrip()
        print("    GRPO candidates:")
        for b in range(min(B, 4)):
            prompt_str = _tok2str(prompt_batch[b])
            clean_str = _tok2str(clean_batch[b])
            print(f"      prompt: {repr(prompt_str):30s}  target: {repr(clean_str)}")
            for k in range(K):
                idx = b * K + k
                gen_str = _tok2str(generated[idx])
                r = rewards[idx].item()
                print(f"        k={k}: {repr(gen_str):40s} reward={r:.3f}")
        print()

    # --- 3. Group-relative advantages (z-score per group) ---
    rewards_grouped = rewards.view(B, K)
    mean_r = rewards_grouped.mean(dim=1, keepdim=True)
    std_r = rewards_grouped.std(dim=1, keepdim=True)
    advantages = (rewards_grouped - mean_r) / (std_r + 1e-4)
    advantages = advantages.view(B * K)

    # --- 4. Pre-cache reference losses and perturbations ---
    sampling_eps = 1e-3
    ref_cache = []

    def _compute_loss(log_scores, xt, x0, sb, st, answer_mask=None):
        """Compute SEDD score entropy loss from log_scores and precomputed noise."""
        probs = forward_transition(sb, x0, vocab_size, "absorb")
        p_xt = probs.gather(2, xt.unsqueeze(-1))
        ratios = probs / p_xt.clamp(min=1e-30)
        eq_mask = torch.ones_like(log_scores)
        eq_mask.scatter_(2, xt.unsqueeze(-1), 0.0)
        if answer_mask is not None:
            eq_mask = eq_mask * answer_mask.unsqueeze(-1).float()
        has_r = (probs > 1e-30) & (eq_mask > 0.5)
        log_r = torch.log(ratios.clamp(min=1e-30))
        u = log_scores - log_r
        u_clamped = u.clamp(max=50.0)
        loss_pos = ratios * (torch.expm1(u_clamped) - u)
        loss_zero = torch.exp(log_scores.clamp(max=50.0))
        loss_entries = torch.where(has_r, loss_pos, loss_zero) * eq_mask
        return (loss_entries.sum(dim=(-1, -2)) * st)

    with torch.no_grad():
        for ep in range(epochs):
            ep_data = []
            for k in range(K):
                idx = torch.arange(B, device=device) * K + k
                candidate_k = generated[idx.cpu()].to(device)

                t = torch.rand(B, device=device)
                sb = schedule.sigma_bar(t)
                st = schedule.sigma(t)
                xt, _ = sample_xt(candidate_k, sb, vocab_size, "absorb")
                xt = torch.where(visible, prompt_batch, xt)

                z_k = tuple(zi[:B].detach() for zi in z) if z[0].shape[0] >= B else z
                log_score = model(xt, sb)
                ref_loss_ps = _compute_loss(log_score, xt, candidate_k, sb, st)

                ep_data.append({
                    'sb': sb,
                    'st': st,
                    'perturbed': xt.clone(),
                    'candidate': candidate_k,
                    'ref_loss': ref_loss_ps,
                })
            ref_cache.append(ep_data)

    # --- 5. DGPO optimization with Q_head ---
    model.train()
    total_loss = 0.0

    for ep in range(epochs):
        optimizer.zero_grad()
        n_tokens = 0

        if fused:
            losses_k = []
            q_loss_total = torch.tensor(0.0, device=device)
            aux_total = torch.tensor(0.0, device=device)
            for k in range(K):
                c = ref_cache[ep][k]
                z_k = tuple(zi[:B].detach() for zi in z) if z[0].shape[0] >= B else z
                ix = model.front(c['perturbed'], c['sb'], memories=memories)
                z_k, log_score, q, aux_loss = model.step(z_k, ix)
                loss_ps = _compute_loss(log_score, c['perturbed'], c['candidate'], c['sb'], c['st'])
                losses_k.append(loss_ps)
                aux_total = aux_total + aux_loss

                idx = torch.arange(B, device=device) * K + k
                q_target = (rewards[idx] > 0.5).float()
                q_loss_total = q_loss_total + F.binary_cross_entropy_with_logits(
                    q.squeeze(-1), q_target)

            group_terms = torch.zeros(B, device=device)
            for k in range(K):
                idx = torch.arange(B, device=device) * K + k
                delta = losses_k[k].detach() - ref_cache[ep][k]['ref_loss']
                group_terms += advantages[idx] * beta_dgpo * delta / K
            group_weights = torch.sigmoid(group_terms)

            weighted_sum = torch.tensor(0.0, device=device)
            for k in range(K):
                idx = torch.arange(B, device=device) * K + k
                weighted_sum = weighted_sum + (group_weights.detach() * advantages[idx] * losses_k[k]).sum()
                n_masked = (~visible).sum().item()
                n_tokens += max(n_masked, 1)

            (weighted_sum + q_loss_total + aux_total).backward()
            total_loss += weighted_sum.item()

        else:
            cur_losses_detached = []
            with torch.no_grad():
                for k in range(K):
                    c = ref_cache[ep][k]
                    log_score = model(c['perturbed'], c['sb'])
                    loss_ps = _compute_loss(log_score, c['perturbed'], c['candidate'], c['sb'], c['st'])
                    cur_losses_detached.append(loss_ps)

            group_terms = torch.zeros(B, device=device)
            for k in range(K):
                idx = torch.arange(B, device=device) * K + k
                delta = cur_losses_detached[k] - ref_cache[ep][k]['ref_loss']
                group_terms += advantages[idx] * beta_dgpo * delta / K
            group_weights = torch.sigmoid(group_terms)

            for k in range(K):
                c = ref_cache[ep][k]
                idx = torch.arange(B, device=device) * K + k
                adv_k = advantages[idx]

                z_k = tuple(zi[:B].detach() for zi in z) if z[0].shape[0] >= B else z
                ix = model.front(c['perturbed'], c['sb'], memories=memories)
                z_k, log_score, q, aux_loss = model.step(z_k, ix)
                loss_ps = _compute_loss(log_score, c['perturbed'], c['candidate'], c['sb'], c['st'])

                q_target = (rewards[idx] > 0.5).float()
                q_loss = F.binary_cross_entropy_with_logits(q.squeeze(-1), q_target)

                weighted_loss = (group_weights.detach() * adv_k * loss_ps).sum()
                n_masked = (~visible).sum().item()
                n_tokens += max(n_masked, 1)

                (weighted_loss + q_loss + aux_loss).backward()
                total_loss += weighted_loss.item()

        if n_tokens > 0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(n_tokens)

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        z_out = tuple(zi.detach() for zi in z_k)

    metrics = {
        'mean_reward': rewards.mean().item(),
        'max_reward': rewards.max().item(),
        'min_reward': rewards.min().item(),
        'std_reward': rewards.std().item(),
        'frac_correct': (rewards > 0.5).float().mean().item(),
    }

    return total_loss / max(n_tokens, 1), z_out, metrics


def make_z_for_grpo(batch_size, seq_len, d_model, device):
    """Create fresh HRM state for GRPO sampling."""
    z = torch.zeros(batch_size, seq_len, d_model, device=device)
    return z, z.clone()
