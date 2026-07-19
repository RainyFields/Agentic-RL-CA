# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 3b.3 — reconstruct a method's ASSIGNED per-turn advantage A_t offline, from the
per-turn fields of a grouped diagnostic rollout, so it can be correlated against the ideal
per-turn credit dV-hat_t (MC continuation-value delta) in credit_align.

Fidelity: we do NOT re-implement the estimator — we call the exact training code
(gigpo.core_gigpo.compute_gigpo_outcome_advantage) with tensors reconstructed from the rollout.
step_rewards is the per-turn discounted return-to-go (gamma), matching
core_gigpo.compute_step_discounted_returns as invoked in ray_trainer.py:1273. A response length
of 1 suffices: the estimators reduce token_level_rewards over the response axis and broadcast a
per-row SCALAR advantage over response_mask, which we read back at column 0.

GRPO is intentionally absent: it assigns ONE trajectory-level advantage broadcast to every turn,
so its per-turn signal (and thus its correlation with dV-hat_t) is identically zero by
construction — reported analytically, no rollout needed. HCAPO needs a hindsight forward pass
(hindsight_log_probs); deferred.

Rows are dicts: {uid (prompt group id), traj_uid, turn_index, anchor_obs (str), reward (float)}.
"""
import numpy as np


def step_discounted_returns(rows, gamma=1.0):
    """Per-turn discounted return-to-go within each trajectory (matches
    core_gigpo.compute_step_discounted_returns). Returns {(traj_uid, turn_index): G_t}."""
    by_traj = {}
    for r in rows:
        by_traj.setdefault(r["traj_uid"], []).append((int(r["turn_index"]), float(r["reward"])))
    out = {}
    for tuid, turns in by_traj.items():
        turns.sort(key=lambda x: x[0])
        running = 0.0
        for ti, rew in reversed(turns):
            running = rew + gamma * running
            out[(tuid, ti)] = running
    return out


def gigpo_per_turn_advantage(rows, gamma=1.0, step_advantage_w=1.0, mode="mean_std_norm",
                             enable_similarity=True, similarity_thresh=0.9):
    """Assigned per-turn scalar advantage A_t for GiGPO on a grouped rollout.
    Config defaults match configs/cond_gigpo.sh. Returns {(traj_uid, turn_index): A_t}."""
    import torch
    from gigpo import core_gigpo

    bs = len(rows)
    reward = np.array([float(r["reward"]) for r in rows], dtype=np.float32)
    sret = step_discounted_returns(rows, gamma)
    step_rewards = np.array([sret[(r["traj_uid"], int(r["turn_index"]))] for r in rows],
                            dtype=np.float32)

    token_level_rewards = torch.tensor(reward, dtype=torch.float32).unsqueeze(-1)  # (bs,1); sum=reward
    step_rewards_t = torch.tensor(step_rewards, dtype=torch.float32)               # (bs,)
    response_mask = torch.ones(bs, 1, dtype=torch.float32)
    anchor_obs = np.array([r["anchor_obs"] for r in rows], dtype=object)
    index = np.array([r["uid"] for r in rows], dtype=object)               # prompt group (episode)
    traj_index = np.array([r["traj_uid"] for r in rows], dtype=object)     # trajectory

    scores, _ = core_gigpo.compute_gigpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        step_rewards=step_rewards_t,
        response_mask=response_mask,
        anchor_obs=anchor_obs,
        index=index,
        traj_index=traj_index,
        step_advantage_w=step_advantage_w,
        mode=mode,
        enable_similarity=enable_similarity,
        similarity_thresh=similarity_thresh,
    )
    a = scores[:, 0].detach().cpu().numpy()
    return {(rows[i]["traj_uid"], int(rows[i]["turn_index"])): float(a[i]) for i in range(bs)}
