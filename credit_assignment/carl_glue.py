# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""CARL trainer glue (Phase 2b.4/2b.5) — runs in RayPPOTrainer.fit() BEFORE adjust_batch.

Builds the edge rows from the tree-loop's non_tensor fields (carl_src/carl_dst are sha1
node ids of the policy-visible prompt before/after each action), computes edge advantages
via credit_assignment.core_carl, attaches the per-row scalar as `carl_adv`, and — in the
paper's primary drop mode (Eq. 13) — physically removes non-critical edges from the update
set. Running before adjust_batch means divisibility padding happens on the final D_upd and
copy-duplicates are harmless (they carry their row's precomputed scalar).

compute_advantage's CARL branch then only broadcasts `carl_adv` over response tokens.
"""
import numpy as np

from credit_assignment.core_carl import build_carl_rows, compute_carl_edge_advantages


def apply_carl_to_batch(batch, algo_cfg):
    """batch: DataProto of per-turn rows (post gather_rollout_data, pre adjust_batch).
    algo_cfg: config.algorithm (needs .gamma, .carl.*). Returns (batch, metrics)."""
    carl_cfg = algo_cfg.get('carl', {}) or {}
    drop = bool(carl_cfg.get('drop_noncritical', True))
    norm_adv = bool(carl_cfg.get('norm_adv', False))
    assert float(algo_cfg.gamma) == 1.0, "CARL assumes undiscounted terminal reward (locked protocol gamma=1.0)"

    nt = batch.non_tensor_batch
    n = len(batch)
    tuid = nt['traj_uid']
    tind = nt['turn_index']
    src = nt['carl_src']
    dst = nt['carl_dst']
    env_done = nt['env_done']
    env_reward = nt['env_reward']

    rows = build_carl_rows(tuid, tind, src, dst, env_done, env_reward)
    advantages, kept, metrics = compute_carl_edge_advantages(rows, drop_noncritical=drop)

    adv_arr = np.array([advantages[(str(tuid[i]), int(tind[i]))] for i in range(n)],
                       dtype=np.float32)
    keep_arr = np.array([kept[(str(tuid[i]), int(tind[i]))] for i in range(n)], dtype=bool)

    if norm_adv:
        kept_vals = adv_arr[keep_arr]
        if kept_vals.size:
            adv_arr = adv_arr / (kept_vals.std() + 1e-6)

    if 'carl_resumed' in nt:
        resumed_by_traj = {}
        for i in range(n):
            resumed_by_traj[tuid[i]] = bool(nt['carl_resumed'][i])
        metrics['carl/resumed_traj_frac'] = float(np.mean(list(resumed_by_traj.values())))

    kept_adv = adv_arr[keep_arr]
    metrics['carl/adv_abs_mean_kept'] = float(np.abs(kept_adv).mean()) if kept_adv.size else 0.0
    metrics['carl/n_rows_dropped'] = int(n - keep_arr.sum()) if drop else 0

    batch.non_tensor_batch['carl_adv'] = adv_arr

    if drop:
        if not keep_arr.any():
            # Degenerate batch (no critical states anywhere): keep everything with zero
            # advantage — a no-op policy-gradient step instead of a crash.
            metrics['carl/empty_update_set'] = 1.0
            print("[CARL] WARNING: no critical edges in batch; emitting zero-advantage no-op step", flush=True)
        else:
            metrics['carl/empty_update_set'] = 0.0
            batch = batch.select_idxs(np.nonzero(keep_arr)[0])

    return batch, metrics
