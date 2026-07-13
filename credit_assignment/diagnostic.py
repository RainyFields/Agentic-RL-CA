# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 3b — credit-alignment diagnostic statistics (RQ2's quantitative teeth).

Offline, never a training signal. For fixed logged trajectories at fixed checkpoints, the
GPU runner estimates the outcome-only continuation value of each turn prefix
    V̂(s_t) = (1/K) Σ_k R(τ_k | s_t)
by prefix-resume (state_tools + env_manager.restore_batch) with continuations sampled from
THE SAME checkpoint that generated the trajectory (asserted by construction: checkpoint id
recorded per trajectory and per continuation batch).

This module holds the pure statistics (CPU): the post-hoc step contribution
    ΔV̂_t = V̂(s_{t+1}) − V̂(s_t)
is the martingale-increment quantity — an ideal process reward's increments equal exact
advantages — so these metrics read as "distance from ideal credit":
  - POOLED Spearman over turn-pairs across trajectories (per-trajectory rank over ≤3 pairs
    is nearly meaningless with binary EM and K=8: std(V̂)≈0.17 near p=0.5)
  - bootstrap CIs on the pooled correlation
  - sign agreement
  - pairwise turn-ranking accuracy (within-trajectory pairs)
"""
import numpy as np


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert len(x) == len(y) and len(x) >= 2
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    # average ranks for ties
    for arr, r in ((x, rx), (y, ry)):
        for v in np.unique(arr):
            m = arr == v
            if m.sum() > 1:
                r[m] = r[m].mean()
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def credit_alignment_stats(pairs, n_boot=2000, seed=0):
    """pairs: list of dicts with keys
         traj_uid, turn_index, assigned_advantage (A_t), delta_v (ΔV̂_t)
       Returns the pooled diagnostic metrics with bootstrap CIs (resampling TRAJECTORIES,
       not pairs — pairs within a trajectory are dependent)."""
    if not pairs:
        return {"n_pairs": 0}
    a = np.array([p["assigned_advantage"] for p in pairs], dtype=float)
    d = np.array([p["delta_v"] for p in pairs], dtype=float)
    tuids = np.array([p["traj_uid"] for p in pairs], dtype=object)

    pooled_rho = spearman(a, d)
    nonzero = (a != 0) | (d != 0)
    sign_agree = float(np.mean(np.sign(a[nonzero]) == np.sign(d[nonzero]))) if nonzero.any() else float("nan")

    # pairwise within-trajectory turn-ranking accuracy
    correct = total = 0
    by_traj = {}
    for i, t in enumerate(tuids):
        by_traj.setdefault(t, []).append(i)
    for t, idxs in by_traj.items():
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                da, dd = a[idxs[i]] - a[idxs[j]], d[idxs[i]] - d[idxs[j]]
                if dd == 0:
                    continue  # no ground-truth ordering
                total += 1
                correct += (da * dd > 0) or (da == 0 and False)
    ranking_acc = correct / total if total else float("nan")

    # bootstrap over trajectories
    rng = np.random.RandomState(seed)
    uniq = list(by_traj.keys())
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(len(uniq), size=len(uniq), replace=True)
        idxs = np.concatenate([by_traj[uniq[c]] for c in chosen])
        if len(idxs) >= 2:
            boots.append(spearman(a[idxs], d[idxs]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan")))

    return {
        "n_pairs": len(pairs),
        "n_trajectories": len(by_traj),
        "pooled_spearman": pooled_rho,
        "pooled_spearman_ci95": [float(lo), float(hi)],
        "sign_agreement": sign_agree,
        "pairwise_ranking_accuracy": ranking_acc,
        "n_ranking_pairs": total,
    }


def build_pairs(prefix_values, assigned_advantages):
    """prefix_values: {(traj_uid, t): V̂(s_t)} including t=0..T (V̂ after last turn = terminal
    outcome); assigned_advantages: {(traj_uid, t): A_t} per method.
    ΔV̂_t = V̂(s_{t+1}) − V̂(s_t), paired with A_t."""
    pairs = []
    for (tuid, t), a in assigned_advantages.items():
        v_now = prefix_values.get((tuid, t))
        v_next = prefix_values.get((tuid, t + 1))
        if v_now is None or v_next is None:
            continue
        pairs.append(
            {
                "traj_uid": tuid,
                "turn_index": t,
                "assigned_advantage": float(a),
                "delta_v": float(v_next - v_now),
            }
        )
    return pairs
