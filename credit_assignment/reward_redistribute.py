# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Random outcome redistribution (Wave 2, turn_ppo_redist arm).

For each trajectory that reached a true env terminal, the terminal outcome reward R is
redistributed across ALL of the trajectory's turns (terminal included) via per-trajectory
Dirichlet(1,..,1) weights: r_t = w_t * R, sum(r_t) = R exactly. The terminal turn keeps
only its Dirichlet share. Control question: does turn-PPO care WHERE dense reward mass
sits, given total mass and the critic?

NOT B1-based: this arm runs without env-side step_reward_mode, so every non-terminal
reward must be zero on entry (hard-asserted — guards accidental combination with B1).
Trajectories with no env_done row (rollout-truncated / early-stopped) are left untouched
and counted in redist/n_no_terminal.

Runs driver-side on the rollout DataProto BEFORE compute_advantage / adjust_batch, so
gae_turn consumes the redistributed per-turn rewards exactly as it consumes B1's.
Determinism: weights are seeded by (seed, traj_uid) — independent of batch order.
Duplicate rows of the same turn (padding copies) receive identical values.
"""
import zlib

import numpy as np


def redistribute_terminal_reward(batch, seed: int = 1234):
    """Mutates batch.non_tensor_batch['rewards'] in place; returns (batch, metrics)."""
    rewards = batch.non_tensor_batch["rewards"]
    traj_uid = batch.non_tensor_batch["traj_uid"]
    turn_index = np.array([int(x) for x in batch.non_tensor_batch["turn_index"]])
    env_done = np.array([bool(x) for x in batch.non_tensor_batch["env_done"]])

    r = np.array([float(x) for x in rewards])
    n = len(r)
    idx_of_traj = {}
    for i in range(n):
        idx_of_traj.setdefault(traj_uid[i], []).append(i)

    sum_before = float(r.sum())
    n_redistributed = 0
    n_no_terminal = 0
    n_zero_outcome = 0
    n_single_turn = 0
    terminal_shares = []
    max_mass_err = 0.0

    for tuid, idxs in idx_of_traj.items():
        idxs = np.asarray(idxs)
        term_rows = idxs[env_done[idxs]]
        if len(term_rows) == 0:
            n_no_terminal += 1
            continue
        term_ti = turn_index[term_rows[0]]
        assert (turn_index[term_rows] == term_ti).all(), (
            f"trajectory {tuid} has env_done rows at different turn_index values"
        )
        nonterm = idxs[~env_done[idxs]]
        nonterm_mass = float(np.abs(r[nonterm]).sum()) if len(nonterm) else 0.0
        assert nonterm_mass < 1e-8, (
            f"trajectory {tuid} carries nonzero non-terminal reward ({nonterm_mass}); "
            "reward_redistribute must not be combined with env-side step rewards (B1)"
        )

        R = float(r[term_rows[0]])
        uniq_ti = np.unique(turn_index[idxs])
        K = len(uniq_ti)

        rng = np.random.RandomState((zlib.crc32(str(tuid).encode()) ^ (seed & 0xFFFFFFFF)) & 0x7FFFFFFF)
        w = rng.dirichlet(np.ones(K))

        if R == 0.0:
            n_zero_outcome += 1
            continue
        if K == 1:
            n_single_turn += 1
            continue
        n_redistributed += 1
        for k, ti in enumerate(uniq_ti):
            for i in idxs[turn_index[idxs] == ti]:
                r[i] = w[k] * R
        terminal_shares.append(float(w[uniq_ti == term_ti][0]))
        # per-trajectory mass check on unique turns (duplicates carry copies)
        traj_sum = sum(w[k] * R for k in range(K))
        max_mass_err = max(max_mass_err, abs(traj_sum - R))

    batch.non_tensor_batch["rewards"] = np.array([r[i] for i in range(n)], dtype=object)

    sum_after = float(r.sum())
    metrics = {
        "redist/n_traj": len(idx_of_traj),
        "redist/n_redistributed": n_redistributed,
        "redist/n_no_terminal": n_no_terminal,
        "redist/n_zero_outcome": n_zero_outcome,
        "redist/n_single_turn": n_single_turn,
        "redist/terminal_share_mean": float(np.mean(terminal_shares)) if terminal_shares else 0.0,
        "redist/reward_sum_before": sum_before,
        "redist/reward_sum_after": sum_after,
        "redist/max_traj_mass_err": max_mass_err,
    }
    assert max_mass_err < 1e-6, "per-trajectory redistributed mass differs from outcome"
    return batch, metrics
