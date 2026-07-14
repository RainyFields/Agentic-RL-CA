# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 2b.2/2b.4 unit checks (CPU-only, pure parts of the CARL tree loop):
- node identity: sha1 over the tokenized policy-visible prompt — equal iff the token
  sequences are identical (the plan's hard node-merge guarantee, 2b.3).
- carl_resume_schedule: deterministic round-robin over depth-sorted snapshots,
  include_root gating, root-only fallback.
- build_carl_rows: terminal-reward placement (env_done rows + truncated-trajectory
  last rows) and (traj_uid, turn_index) keying.
- end-to-end: rows -> compute_carl_edge_advantages on a hand-built two-prompt batch.
"""
import numpy as np

from credit_assignment.core_carl import (
    build_carl_rows,
    carl_resume_schedule,
    compute_carl_edge_advantages,
)
from credit_assignment.state_tools import node_id_from_prompt_ids


# ---------------------------------------------------------------- node identity
def test_node_id_equal_iff_tokens_equal():
    a = [151644, 872, 198, 3838, 374]
    assert node_id_from_prompt_ids(a) == node_id_from_prompt_ids(np.array(a, dtype=np.int64))
    assert node_id_from_prompt_ids(a) == node_id_from_prompt_ids(np.array(a, dtype=np.int32))
    # merged rows must have byte-identical tokenized prompts: any difference => new node
    assert node_id_from_prompt_ids(a) != node_id_from_prompt_ids(a + [0])
    assert node_id_from_prompt_ids(a) != node_id_from_prompt_ids(list(reversed(a)))
    assert node_id_from_prompt_ids([]) == node_id_from_prompt_ids(np.array([], dtype=np.int64))


# ---------------------------------------------------------------- resume schedule
def _snaps(depths):
    return [{"depth": d, "snap": f"s{d}_{i}", "node": f"n{d}_{i}"} for i, d in enumerate(depths)]


def test_schedule_round_robin_excludes_root_by_default():
    snaps = _snaps([0, 1, 2, 3])
    picks = carl_resume_schedule(snaps, n_resume=5, include_root=False)
    assert [p["depth"] for p in picks] == [1, 2, 3, 1, 2]


def test_schedule_include_root():
    snaps = _snaps([0, 1])
    picks = carl_resume_schedule(snaps, n_resume=4, include_root=True)
    assert [p["depth"] for p in picks] == [0, 1, 0, 1]


def test_schedule_root_only_fallback():
    # 1-turn trajectory: only the depth-0 snapshot exists; fall back to it rather
    # than dropping the group's resume budget.
    snaps = _snaps([0])
    picks = carl_resume_schedule(snaps, n_resume=3, include_root=False)
    assert [p["depth"] for p in picks] == [0, 0, 0]
    assert carl_resume_schedule([], n_resume=3) == []


def test_schedule_is_deterministic_and_depth_sorted():
    snaps = _snaps([3, 1, 2])  # insertion order not depth order
    picks = carl_resume_schedule(snaps, n_resume=6)
    assert [p["depth"] for p in picks] == [1, 2, 3, 1, 2, 3]
    assert picks == carl_resume_schedule(_snaps([3, 1, 2]), n_resume=6)


# ---------------------------------------------------------------- row building
def test_build_rows_terminal_placement():
    #  tA: 2 turns, env_done on the last  |  tB: truncated after 1 turn, no env_done
    tuid = np.array(["tA", "tA", "tB"], dtype=object)
    tind = np.array([0, 1, 0], dtype=np.int64)
    src = np.array(["r", "u1", "r"], dtype=object)
    dst = np.array(["u1", "leafA", "u2"], dtype=object)
    env_done = np.array([False, True, False], dtype=object)
    env_reward = np.array([0.0, 1.0, 0.0], dtype=object)
    rows = build_carl_rows(tuid, tind, src, dst, env_done, env_reward)
    assert rows[0]["terminal_reward"] is None            # tA mid-trajectory
    assert rows[1]["terminal_reward"] == 1.0             # tA env_done
    assert rows[2]["terminal_reward"] == 0.0             # tB truncated last row
    assert rows[1]["key"] == ("tA", 1) and rows[2]["key"] == ("tB", 0)


# ---------------------------------------------------------------- end to end
def test_rows_to_advantages_end_to_end():
    """Two trajectories from one root: root -> u (shared action-merge) then diverging
    -> win / lose. Root has 1 distinct child (u) => NOT critical => root edges dropped.
    u has 2 distinct children => critical => V(u)=0.5, adv(win)=+0.5, adv(lose)=-0.5."""
    tuid = np.array(["t1", "t1", "t2", "t2"], dtype=object)
    tind = np.array([0, 1, 0, 1], dtype=np.int64)
    src = np.array(["root", "u", "root", "u"], dtype=object)
    dst = np.array(["u", "win", "u", "lose"], dtype=object)
    env_done = np.array([False, True, False, True], dtype=object)
    env_reward = np.array([0.0, 1.0, 0.0, 0.0], dtype=object)

    rows = build_carl_rows(tuid, tind, src, dst, env_done, env_reward)
    adv, kept, metrics = compute_carl_edge_advantages(rows, drop_noncritical=True)

    assert not kept[("t1", 0)] and not kept[("t2", 0)]   # root edges: non-critical, dropped
    assert kept[("t1", 1)] and kept[("t2", 1)]
    assert abs(adv[("t1", 1)] - 0.5) < 1e-9
    assert abs(adv[("t2", 1)] + 0.5) < 1e-9
    assert metrics["carl/critical_state_frac"] == 0.5    # {root, u} sources, u critical
