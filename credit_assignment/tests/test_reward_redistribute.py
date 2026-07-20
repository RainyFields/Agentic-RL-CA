# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Wave-2 unit checks — random outcome redistribution (turn_ppo_redist arm):
mass conservation (per-trajectory sum == outcome), simplex validity, terminal keeps only
its share, zero-outcome/single-turn/no-env_done edge cases, hard-assert on nonzero
non-terminal rewards (no accidental B1 combination), seeded determinism independent of
row order, duplicate rows carry identical values.
Run: python -m pytest credit_assignment/tests/test_reward_redistribute.py -q  (CPU only)
"""
import numpy as np
import pytest

from credit_assignment.reward_redistribute import redistribute_terminal_reward


class FakeBatch:
    def __init__(self, non_tensor):
        self.non_tensor_batch = {k: np.array(v, dtype=object) for k, v in non_tensor.items()}


def make_batch(trajs):
    """trajs: list of (traj_uid, [(reward, done), ...]) — turn_index = position."""
    nt = {"rewards": [], "traj_uid": [], "env_done": [], "turn_index": []}
    for tuid, turns in trajs:
        for k, (r, d) in enumerate(turns):
            nt["rewards"].append(r)
            nt["traj_uid"].append(tuid)
            nt["env_done"].append(d)
            nt["turn_index"].append(k)
    return FakeBatch(nt)


def rewards_of(batch):
    return [float(x) for x in batch.non_tensor_batch["rewards"]]


def traj_rewards(batch):
    out = {}
    for i, t in enumerate(batch.non_tensor_batch["traj_uid"]):
        out.setdefault(t, []).append(float(batch.non_tensor_batch["rewards"][i]))
    return out


def test_mass_conservation_and_simplex():
    b = make_batch([("t%d" % i, [(0.0, False), (0.0, False), (0.0, False), (1.0, True)])
                    for i in range(20)])
    b, m = redistribute_terminal_reward(b, seed=7)
    for t, rs in traj_rewards(b).items():
        assert sum(rs) == pytest.approx(1.0, abs=1e-9)
        assert all(r >= 0.0 for r in rs)           # Dirichlet weights are a simplex
        assert rs[3] < 1.0                          # terminal keeps only its share
        assert max(rs) < 1.0 and len([r for r in rs if r > 0]) == 4  # all turns funded a.s.
    assert m["redist/n_redistributed"] == 20
    assert m["redist/reward_sum_before"] == pytest.approx(m["redist/reward_sum_after"], abs=1e-6)
    assert 0.0 < m["redist/terminal_share_mean"] < 1.0
    assert m["redist/max_traj_mass_err"] < 1e-9


def test_zero_outcome_untouched():
    b = make_batch([("t1", [(0.0, False), (0.0, False), (0.0, True)])])
    b, m = redistribute_terminal_reward(b, seed=0)
    assert rewards_of(b) == [0.0, 0.0, 0.0]
    assert m["redist/n_zero_outcome"] == 1 and m["redist/n_redistributed"] == 0


def test_single_turn_noop():
    b = make_batch([("t1", [(1.0, True)])])
    b, m = redistribute_terminal_reward(b, seed=0)
    assert rewards_of(b) == [1.0]
    assert m["redist/n_single_turn"] == 1 and m["redist/n_redistributed"] == 0


def test_no_terminal_skipped():
    # rollout-truncated trajectory: no env_done row anywhere
    b = make_batch([("t1", [(0.0, False), (0.0, False)]),
                    ("t2", [(0.0, False), (1.0, True)])])
    b, m = redistribute_terminal_reward(b, seed=3)
    r = traj_rewards(b)
    assert r["t1"] == [0.0, 0.0]                    # untouched
    assert sum(r["t2"]) == pytest.approx(1.0)
    assert m["redist/n_no_terminal"] == 1 and m["redist/n_redistributed"] == 1


def test_nonzero_nonterminal_raises():
    # B1-style step reward present -> must hard-fail (mutually exclusive by design)
    b = make_batch([("t1", [(0.2, False), (0.0, False), (1.0, True)])])
    with pytest.raises(AssertionError):
        redistribute_terminal_reward(b, seed=0)


def test_deterministic_and_order_independent():
    trajs = [("t%d" % i, [(0.0, False), (0.0, False), (0.0, False), (1.0, True)])
             for i in range(20)]
    b1 = make_batch(trajs)
    b2 = make_batch(list(reversed(trajs)))
    b1, _ = redistribute_terminal_reward(b1, seed=42)
    b2, _ = redistribute_terminal_reward(b2, seed=42)
    assert traj_rewards(b1) == traj_rewards(b2)     # same weights per traj regardless of order
    b3 = make_batch(trajs)
    b3, _ = redistribute_terminal_reward(b3, seed=43)
    assert traj_rewards(b1) != traj_rewards(b3)     # different seed differs (a.s.)


def test_duplicate_rows_same_value():
    # duplicate rows of the same turn (padding copies) must carry identical values
    nt = {"rewards": [0.0, 0.0, 1.0, 1.0], "traj_uid": ["t1"] * 4,
          "env_done": [False, False, True, True], "turn_index": [0, 0, 1, 1]}
    b = FakeBatch({k: v for k, v in nt.items()})
    b, m = redistribute_terminal_reward(b, seed=5)
    r = rewards_of(b)
    assert r[0] == r[1] and r[2] == r[3]
    assert r[0] + r[2] == pytest.approx(1.0)        # unique-turn mass sums to outcome
