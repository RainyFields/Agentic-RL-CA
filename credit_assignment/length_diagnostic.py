# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 3c — RQ1 length-weighting diagnostic (diagnose only, no new ablation).

Turn-advantage broadcasting gives a 300-token turn ~10x the policy-gradient mass of a
30-token turn under global token-averaging (token PPO's mass also scales with length,
differently). We LOG, per training step: the per-turn response-length distribution, the
per-turn |advantage| distribution, each turn's effective loss weight
(sum(|A|*mask)/batch token count), and the length-|A| correlation — to verify there is no
large method-dependent length shift between `gae` and `gae_turn` (reported alongside F7).
Pure torch; CPU-testable.
"""
import torch


def _quantiles(t: torch.Tensor, qs=(0.1, 0.5, 0.9)):
    if t.numel() == 0:
        return {f"p{int(q * 100)}": 0.0 for q in qs}
    qt = torch.quantile(t.float(), torch.tensor(qs, dtype=torch.float32))
    return {f"p{int(q * 100)}": float(v) for q, v in zip(qs, qt)}


def length_weight_metrics(advantages: torch.Tensor, response_mask: torch.Tensor, prefix="lengthdiag"):
    """advantages: (B, L) token-level advantages; response_mask: (B, L) 0/1.
    One row = one turn (verl-agent's per-turn batch layout)."""
    mask = response_mask.float()
    lengths = mask.sum(-1)                                     # (B,) tokens per turn
    abs_adv_sum = (advantages.abs() * mask).sum(-1)            # (B,)
    total_tokens = mask.sum().clamp(min=1.0)
    eff_weight = abs_adv_sum / total_tokens                    # per-turn share of gradient mass
    mean_abs_adv = abs_adv_sum / lengths.clamp(min=1.0)        # per-turn mean |A|

    valid = lengths > 0
    L, W, A = lengths[valid], eff_weight[valid], mean_abs_adv[valid]

    def corr(x, y):
        if x.numel() < 2:
            return 0.0
        xc, yc = x - x.mean(), y - y.mean()
        d = (xc.norm() * yc.norm()).clamp(min=1e-12)
        return float((xc * yc).sum() / d)

    m = {f"{prefix}/n_turns": int(valid.sum())}
    m.update({f"{prefix}/resp_len/{k}": v for k, v in _quantiles(L).items()})
    m.update({f"{prefix}/abs_adv/{k}": v for k, v in _quantiles(A).items()})
    m.update({f"{prefix}/eff_loss_weight/{k}": v for k, v in _quantiles(W).items()})
    m[f"{prefix}/corr_len_absadv"] = corr(L, A)
    m[f"{prefix}/corr_len_effweight"] = corr(L, W)
    # top-decile turns' share of total gradient mass (the "long turns dominate" number)
    if W.numel() >= 10:
        k = max(1, int(0.1 * W.numel()))
        m[f"{prefix}/top10pct_len_weight_share"] = float(
            W[torch.argsort(L, descending=True)[:k]].sum() / W.sum().clamp(min=1e-12)
        )
    return m
