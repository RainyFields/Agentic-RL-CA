# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 3b.2 unit checks (CPU-only): continuation-job planning, V-hat aggregation, and
wiring into diagnostic.build_pairs (terminal outcome appended at depth T)."""
from credit_assignment.diag_plan import (
    aggregate_vhat,
    plan_continuation_jobs,
    prefix_values_for_pairs,
)
from credit_assignment.diagnostic import build_pairs


def test_plan_jobs_deterministic_and_k_upgrade_per_trajectory():
    td = {"t0": [0, 1, 2], "t1": [0, 1], "t2": [0], "t3": [0, 1, 2, 3]}
    jobs = plan_continuation_jobs(td, k=8, k_check=16, k_check_frac=0.5, seed=1)
    assert jobs == plan_continuation_jobs(td, k=8, k_check=16, k_check_frac=0.5, seed=1)
    # exactly 2 of 4 trajectories upgraded, and every prefix of an upgraded trajectory
    # gets k=16 (the noise check needs whole trajectories)
    by_traj = {}
    for j in jobs:
        by_traj.setdefault(j["traj"], set()).add(j["k"])
    assert all(len(ks) == 1 for ks in by_traj.values())
    assert sum(ks == {16} for ks in by_traj.values()) == 2
    # all depths covered, sorted per trajectory
    t0 = [j["depth"] for j in jobs if j["traj"] == "t0"]
    assert t0 == [0, 1, 2]


def test_plan_jobs_no_check():
    jobs = plan_continuation_jobs({"a": [0, 1]}, k=4)
    assert [(j["traj"], j["depth"], j["k"]) for j in jobs] == [("a", 0, 4), ("a", 1, 4)]


def test_aggregate_and_pairs_end_to_end():
    job_results = [
        {"traj": "t0", "depth": 0, "rewards": [0, 0, 1, 1]},   # V(s0)=0.5
        {"traj": "t0", "depth": 1, "rewards": [1, 1, 1, 0]},   # V(s1)=0.75
    ]
    vhat = aggregate_vhat(job_results)
    assert vhat["t0"][0]["vhat"] == 0.5 and vhat["t0"][1]["k"] == 4
    # terminal outcome 1.0 appended at depth 2
    pv = prefix_values_for_pairs(vhat, terminal_rewards={"t0": 1.0})
    assert pv[("t0", 2)] == 1.0
    adv = {("t0", 0): 0.3, ("t0", 1): -0.1}
    pairs = build_pairs(pv, adv)
    assert len(pairs) == 2
    d = {(p["traj_uid"], p["turn_index"]): p["delta_v"] for p in pairs}
    assert abs(d[("t0", 0)] - 0.25) < 1e-9   # 0.75 - 0.5
    assert abs(d[("t0", 1)] - 0.25) < 1e-9   # 1.0 - 0.75
