# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 2 unit checks (plan §Verification):
- B1: first-hit-only; hit detection via normalized substring match; no bonus without hit.
- B1-shuffle: per-trajectory count+magnitude preserved; terminal turn NEVER receives
  shaping; single-eligible trajectories unchanged; shuffle_active_frac correct; seeded
  determinism independent of row order.
Run: python -m pytest credit_assignment/tests/test_phase2_shaping.py -q  (CPU only)
"""
import numpy as np
import pytest

from credit_assignment.step_rewards import b1_hit, compute_b1_step_reward
from credit_assignment.b1_shuffle import shuffle_step_rewards, step_reward_stats

GOLD = {"target": np.array(["Wilhelm Conrad Röntgen"], dtype=object)}


class FakeBatch:
    def __init__(self, non_tensor):
        self.non_tensor_batch = {k: np.array(v, dtype=object) for k, v in non_tensor.items()}


# ---------------- B1 ----------------

def test_b1_hit_normalized_substring():
    obs = "<information>Doc 1: the prize went to wilhelm conrad rontgen in 1901.</information>"
    # normalize_answer strips punctuation/case but NOT accents — use exact-normalized text
    obs2 = "<information>Doc 1: Wilhelm Conrad Röntgen, physicist.</information>"
    assert b1_hit(obs2, GOLD)
    assert not b1_hit("<information>nothing relevant</information>", GOLD)
    assert not b1_hit(None, GOLD)
    assert not b1_hit("", GOLD)


def test_b1_first_hit_only():
    obs = "<information>Wilhelm Conrad Röntgen</information>"
    bonus1, hit1, given = compute_b1_step_reward(obs, GOLD, False, 0.2)
    assert bonus1 == pytest.approx(0.2) and hit1 and given
    bonus2, hit2, given = compute_b1_step_reward(obs, GOLD, given, 0.2)
    assert bonus2 == 0.0 and hit2 and given  # raw hit still reported, no second bonus


def test_b1_no_hit_no_latch():
    bonus, hit, given = compute_b1_step_reward("<information>x</information>", GOLD, False, 0.2)
    assert bonus == 0.0 and not hit and not given


# ---------------- shuffle ----------------

def make_batch(trajs):
    """trajs: list of (traj_uid, [(reward, done, tool_calling, valid), ...])."""
    nt = {"rewards": [], "traj_uid": [], "env_done": [], "tool_calling": [],
          "is_action_valid": [], "turn_index": [], "b1_hit": []}
    for tuid, turns in trajs:
        for k, (r, d, tc, v) in enumerate(turns):
            nt["rewards"].append(r)
            nt["traj_uid"].append(tuid)
            nt["env_done"].append(d)
            nt["tool_calling"].append(tc)
            nt["is_action_valid"].append(v)
            nt["turn_index"].append(k)
            nt["b1_hit"].append(r != 0 and not d)
    return FakeBatch(nt)


def rewards_of(batch):
    return [float(x) for x in batch.non_tensor_batch["rewards"]]


def test_shuffle_moves_and_preserves_mass():
    # 3 eligible search turns, B1 on turn 0, terminal EM=1 on turn 3
    b = make_batch([("t1", [(0.2, False, True, True), (0.0, False, True, True),
                            (0.0, False, True, True), (1.0, True, False, True)])])
    b, m = shuffle_step_rewards(b, seed=7)
    r = rewards_of(b)
    assert r[3] == 1.0                      # terminal untouched
    assert r[0] == 0.0                      # moved away from source ("another" turn)
    assert sorted(r[:3]) == [0.0, 0.0, pytest.approx(0.2)]
    assert m["shuffle/active_frac"] == 1.0 and m["shuffle/n_moved"] == 1
    assert m["shuffle/terminal_reward_leak"] == 0.0


def test_shuffle_single_eligible_unchanged_and_active_frac():
    # t1: only one eligible search turn -> unchanged; t2: two eligible -> moved
    b = make_batch([
        ("t1", [(0.2, False, True, True), (0.0, True, False, True)]),
        ("t2", [(0.2, False, True, True), (0.0, False, True, True), (1.0, True, False, True)]),
        ("t3", [(0.0, False, True, True), (0.0, True, False, True)]),  # no B1 reward
    ])
    before = rewards_of(b)
    b, m = shuffle_step_rewards(b, seed=0)
    after = rewards_of(b)
    assert after[0] == before[0] == 0.2       # single-eligible: identical to B1
    assert after[2] == 0.0 and after[3] == pytest.approx(0.2)  # t2 moved to the other turn
    assert m["shuffle/n_b1_pos_traj"] == 2
    assert m["shuffle/active_frac"] == pytest.approx(0.5)
    # per-trajectory count+magnitude preserved
    assert sum(before) == pytest.approx(sum(after))


def test_shuffle_never_hits_terminal_or_invalid():
    rng_hits_everything = [
        ("t%d" % i, [(0.2, False, True, True), (0.0, False, True, False),   # invalid turn
                     (0.0, False, True, True), (1.0, True, False, True)])
        for i in range(50)
    ]
    b = make_batch(rng_hits_everything)
    b, m = shuffle_step_rewards(b, seed=123)
    r = rewards_of(b)
    for i in range(50):
        base = 4 * i
        assert r[base + 1] == 0.0            # invalid turn never receives
        assert r[base + 3] == 1.0            # terminal never receives shaping
        assert r[base + 2] == pytest.approx(0.2) and r[base] == 0.0


def test_shuffle_deterministic_and_order_independent():
    trajs = [("t%d" % i, [(0.2, False, True, True), (0.0, False, True, True),
                          (0.0, False, True, True), (0.0, True, False, True)]) for i in range(20)]
    b1 = make_batch(trajs)
    b2 = make_batch(list(reversed(trajs)))
    b1, _ = shuffle_step_rewards(b1, seed=42)
    b2, _ = shuffle_step_rewards(b2, seed=42)
    m1 = {t: [] for t, _ in trajs}
    m2 = {t: [] for t, _ in trajs}
    for i, t in enumerate(b1.non_tensor_batch["traj_uid"]):
        m1[t].append(float(b1.non_tensor_batch["rewards"][i]))
    for i, t in enumerate(b2.non_tensor_batch["traj_uid"]):
        m2[t].append(float(b2.non_tensor_batch["rewards"][i]))
    assert m1 == m2                          # same destination per traj regardless of order
    b3 = make_batch(trajs)
    b3, _ = shuffle_step_rewards(b3, seed=43)
    assert any(
        [float(x) for x in b1.non_tensor_batch["rewards"]] != [float(x) for x in b3.non_tensor_batch["rewards"]]
        for _ in [0]
    )  # different seed can differ (20 trajs, 2 dests each -> astronomically unlikely equal)


def test_step_reward_stats():
    b = make_batch([
        ("t1", [(0.2, False, True, True), (1.0, True, False, True)]),
        ("t2", [(0.0, False, True, True), (0.0, True, False, True)]),
    ])
    s = step_reward_stats(b)
    assert s["b1/retrieval_hit_traj_rate"] == pytest.approx(0.5)
    assert s["b1/step_reward_traj_rate"] == pytest.approx(0.5)
    assert s["b1/step_reward_mean_nonzero"] == pytest.approx(0.2)
    # non-B1 batch -> {}
    nt = {k: v for k, v in b.non_tensor_batch.items() if k != "b1_hit"}
    assert step_reward_stats(FakeBatch(nt)) == {}
