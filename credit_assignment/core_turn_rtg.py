# Copyright 2026 — Agentic-RL-CA rung-4: turn-level group-normalized reward-to-go (turn_rtg).
#
# Value-free advantage estimator for the ScienceWorld ORM-vs-PRM experiment
# (docs/reports/2026-07-28_sciworld_orm_prm_design/design.md §Algorithm):
#
#     G_{i,t} = sum_{s >= t} gamma^{s-t} r_{i,s}          (reward-to-go, gamma=1 in the spec)
#     A_{i,t} = (G_{i,t} - mu_t) / sigma_t
#
# where mu_t / sigma_t are computed per turn index t across the trajectories of the SAME
# group (uid) that are ALIVE at t (i.e. have a turn-row with that turn_index). The ORM arm
# (terminal-only reward) runs this exact code path: its G_{i,t} is constant along each
# trajectory (= P_T), so while every group member is alive the estimator coincides with
# vanilla trajectory-GRPO; the arms diverge only where reward placement makes them diverge.
# No special-casing of the reward layout anywhere.
#
# Deliberate divergence from core_turn_ppo (gae_turn): NO final verl_F.masked_whiten over
# the batch. The per-turn within-group z-scoring IS the normalization; a second batch-level
# whitening would mix groups and turn depths and (worse) re-scale the ORM arm differently
# from the PRM arm, breaking the "identical estimator, different reward placement" invariant.
#
# Truncation handling: no bootstrap anywhere — there is no critic. A trajectory cut by the
# rollout turn cap contributes the discounted sum of its REALIZED rewards, exactly like an
# env-terminal one. With gamma=1 and the environment's nonnegative max-cumulative reward
# channels this is the spec's intended semantics (foregone-future-reward penalty only).
#
# Robustness (see stack map / adjust_batch + _balance_batch):
#   - duplicate rows (same traj_uid, turn_index) injected by adjust_batch: deduped for both
#     the RTG recursion and the group statistics; every row (dup included) still receives
#     the advantage of its (traj, turn).
#   - row order is never relied upon; everything is grouped by ids.
#   - alive-at-t sets shrink as shorter trajectories finish; a size-1 alive set yields
#     advantage 0 (GiGPO singleton convention: mean = itself, std = 1).
#   - optional active_masks: inactive rows are excluded from statistics and get A = G = 0.
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_turn_rtg_group_advantage(
    response_mask: torch.Tensor,   # (B, T_resp)
    rewards: np.ndarray,           # (B,) per-turn scalar env reward
    uid: np.ndarray,               # (B,) prompt-group id (size env.rollout.n groups)
    traj_uid: np.ndarray,          # (B,) trajectory id
    turn_index: np.ndarray,        # (B,) step index within trajectory
    gamma: float = 1.0,
    sigma_floor: float = 1e-6,
    active_masks: np.ndarray = None,  # (B,) optional bool; False rows -> A=G=0, excluded from stats
):
    """Turn-level group-normalized reward-to-go. Returns (advantages, returns), each
    (B, T_resp): per-row scalars broadcast over the row's response tokens. `returns` is the
    un-normalized G_{i,t} (empirical reward-to-go), analogous to gae_turn's unwhitened
    returns; there is no critic to consume it, but downstream metrics read it."""
    device, dtype = response_mask.device, response_mask.dtype
    B = response_mask.shape[0]
    r = np.asarray(rewards, dtype=np.float64)
    uid = np.asarray(uid)
    traj_uid = np.asarray(traj_uid)
    ti_all = np.asarray(turn_index).astype(np.int64)
    assert r.shape[0] == uid.shape[0] == traj_uid.shape[0] == ti_all.shape[0] == B, (
        r.shape, uid.shape, traj_uid.shape, ti_all.shape, B)
    active = (np.ones(B, dtype=bool) if active_masks is None
              else np.asarray(active_masks).astype(bool))

    # ---- pass 1: per-trajectory reward-to-go over deduped, turn-ordered rows ----
    rtg = {}          # (traj_uid, turn_index) -> G
    traj_group = {}   # traj_uid -> uid
    for t_id in np.unique(traj_uid):
        idx = np.where((traj_uid == t_id) & active)[0]
        if len(idx) == 0:
            continue
        g_ids = np.unique(uid[idx])
        assert len(g_ids) == 1, f"trajectory {t_id} spans groups {g_ids}"
        traj_group[t_id] = g_ids[0]
        ti = ti_all[idx]
        order = np.argsort(ti, kind="stable")
        uniq_ti, first = np.unique(ti[order], return_index=True)   # dedup adjust_batch copies
        rep = idx[order][first]                                    # one representative row/turn
        g_next = 0.0
        for k in range(len(uniq_ti) - 1, -1, -1):                  # reverse over turns
            g_next = float(r[rep[k]]) + gamma * g_next
            rtg[(t_id, int(uniq_ti[k]))] = g_next

    # ---- pass 2: per (group, turn) statistics over alive trajectories ----
    by_group_turn = {}  # (uid, turn_index) -> list of G (one per distinct alive trajectory)
    for (t_id, t), g in rtg.items():
        by_group_turn.setdefault((traj_group[t_id], t), []).append(g)
    stats = {}
    for key, gs in by_group_turn.items():
        if len(gs) == 1:
            stats[key] = (gs[0], 1.0)                              # singleton -> A = 0
        else:
            arr = np.asarray(gs, dtype=np.float64)
            mu = float(arr.mean())
            sigma = float(arr.std(ddof=1))                         # repo convention (torch.std)
            stats[key] = (mu, max(sigma, sigma_floor))

    # ---- pass 3: per-row scalars (duplicates get their (traj, turn) values) ----
    A_scalar = np.zeros(B, dtype=np.float32)
    G_scalar = np.zeros(B, dtype=np.float32)
    for i in range(B):
        if not active[i]:
            continue
        key = (traj_uid[i], int(ti_all[i]))
        g = rtg[key]
        mu, sigma = stats[(traj_group[traj_uid[i]], int(ti_all[i]))]
        A_scalar[i] = (g - mu) / sigma
        G_scalar[i] = g

    A = torch.tensor(A_scalar, device=device, dtype=dtype).unsqueeze(-1)
    G = torch.tensor(G_scalar, device=device, dtype=dtype).unsqueeze(-1)
    advantages = A * response_mask
    returns = G * response_mask
    return advantages, returns
