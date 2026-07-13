# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""B1-shuffle control (plan Phase 2.2) — THE key RQ3 experiment.

For each trajectory carrying a B1 step reward (first-hit-only => at most one nonzero
non-terminal reward per trajectory), reassign that reward uniformly at random to a
DIFFERENT eligible non-terminal search turn of the same trajectory, preserving count and
magnitude. The terminal answer turn NEVER receives shaping reward (that would silently
convert it into extra terminal reward). Trajectories with <2 eligible turns are unchanged
(identical to B1 by construction) — their fraction is tracked via `shuffle_active_frac`
(fraction of B1-positive trajectories with >=2 eligible turns); if it lands below ~30% at
Wave 1 the 8-turn stress test is PROMOTED to required (pre-registered plan B).

Runs driver-side on the rollout DataProto BEFORE compute_advantage / adjust_batch, so
gae_turn consumes the shuffled per-turn rewards exactly as it would consume B1's.
Determinism: destination choice is seeded by (seed, traj_uid) — independent of batch order.

Eligibility: non-terminal turn that actually issued a valid search
(env_done=False & tool_calling & is_action_valid).
"""
import zlib

import numpy as np


def _field(batch, name, default=None):
    nt = batch.non_tensor_batch
    if name in nt:
        return nt[name]
    return default


def shuffle_step_rewards(batch, seed: int = 1234):
    """Mutates batch.non_tensor_batch['rewards'] in place; returns (batch, metrics).

    metrics (raw, pre-whitening — plan requires recording them next to any B1-vs-shuffle
    claim): shuffle/active_frac, shuffle/n_b1_pos_traj, shuffle/n_moved,
    shuffle/step_reward_sum_before/after (must be equal), shuffle/terminal_reward_leak (=0).
    """
    rewards = batch.non_tensor_batch["rewards"]
    traj_uid = batch.non_tensor_batch["traj_uid"]
    env_done = np.array([bool(x) for x in batch.non_tensor_batch["env_done"]])
    tool_calling = _field(batch, "tool_calling")
    if tool_calling is None:
        tool_calling = ~env_done  # conservative fallback: any non-terminal turn
    tool_calling = np.array([bool(x) for x in tool_calling])
    is_valid = _field(batch, "is_action_valid")
    if is_valid is None:
        is_valid = np.ones(len(rewards), dtype=bool)
    is_valid = np.array([bool(x) for x in is_valid])

    r = np.array([float(x) for x in rewards])
    n = len(r)
    idx_of_traj = {}
    for i in range(n):
        idx_of_traj.setdefault(traj_uid[i], []).append(i)

    sum_before = float(r[~env_done].sum())
    n_b1_pos = 0
    n_active = 0
    n_moved = 0

    for tuid, idxs in idx_of_traj.items():
        src = [i for i in idxs if (not env_done[i]) and r[i] != 0.0]
        if not src:
            continue
        n_b1_pos += 1
        assert len(src) == 1, (
            f"first-hit-only violated: trajectory {tuid} has {len(src)} nonzero "
            f"non-terminal step rewards"
        )
        s = src[0]
        eligible = [i for i in idxs if (not env_done[i]) and tool_calling[i] and is_valid[i]]
        if s not in eligible:
            # the rewarded turn is by construction a search turn; be safe anyway
            eligible = sorted(set(eligible + [s]))
        if len(eligible) < 2:
            continue  # necessarily identical to B1
        n_active += 1
        rng = np.random.RandomState((zlib.crc32(str(tuid).encode()) ^ (seed & 0xFFFFFFFF)) & 0x7FFFFFFF)
        dests = [i for i in eligible if i != s]
        d = dests[rng.randint(len(dests))]
        r[d] = r[s]
        r[s] = 0.0
        n_moved += 1

    # write back preserving object dtype
    batch.non_tensor_batch["rewards"] = np.array([r[i] for i in range(n)], dtype=object)

    sum_after = float(r[~env_done].sum())
    terminal_leak = float(np.abs(r[env_done] - np.array([float(x) for x in rewards])[env_done]).sum())
    metrics = {
        "shuffle/n_traj": len(idx_of_traj),
        "shuffle/n_b1_pos_traj": n_b1_pos,
        "shuffle/active_frac": (n_active / n_b1_pos) if n_b1_pos else 0.0,
        "shuffle/n_moved": n_moved,
        "shuffle/step_reward_sum_before": sum_before,
        "shuffle/step_reward_sum_after": sum_after,
        "shuffle/terminal_reward_leak": terminal_leak,
    }
    assert abs(sum_before - sum_after) < 1e-6, "shuffle changed total step-reward mass"
    assert terminal_leak == 0.0, "shuffle touched a terminal-turn reward"
    return batch, metrics


def step_reward_stats(batch):
    """Raw (pre-whitening) B1 step-reward + retrieval-hit stats for W&B. Returns {} when the
    batch carries no B1 instrumentation (non-B1 arms)."""
    if "b1_hit" not in batch.non_tensor_batch:
        return {}
    rewards = np.array([float(x) for x in batch.non_tensor_batch["rewards"]])
    env_done = np.array([bool(x) for x in batch.non_tensor_batch["env_done"]])
    b1_hit = np.array([bool(x) for x in batch.non_tensor_batch["b1_hit"]])
    traj_uid = batch.non_tensor_batch["traj_uid"]

    trajs = {}
    for i, t in enumerate(traj_uid):
        trajs.setdefault(t, []).append(i)
    n_traj = len(trajs)
    traj_hit = sum(1 for idxs in trajs.values() if any(b1_hit[i] for i in idxs))
    traj_pos = sum(
        1 for idxs in trajs.values() if any((rewards[i] != 0) and not env_done[i] for i in idxs)
    )
    step_r = rewards[~env_done]
    return {
        "b1/retrieval_hit_turn_rate": float(b1_hit[~env_done].mean()) if (~env_done).any() else 0.0,
        "b1/retrieval_hit_traj_rate": traj_hit / max(n_traj, 1),
        "b1/step_reward_traj_rate": traj_pos / max(n_traj, 1),
        "b1/step_reward_mean_nonzero": float(step_r[step_r != 0].mean()) if (step_r != 0).any() else 0.0,
        "b1/step_reward_sum_per_traj": float(step_r.sum() / max(n_traj, 1)),
    }
