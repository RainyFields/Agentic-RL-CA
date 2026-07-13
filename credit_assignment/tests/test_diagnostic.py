# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 3b unit checks: V̂/ΔV̂ pairing on a hand-built 2-turn case matches closed form;
pooled Spearman/sign/ranking behave correctly on known inputs."""
import numpy as np
import pytest

from credit_assignment.diagnostic import build_pairs, credit_alignment_stats, spearman


def test_spearman_known_values():
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert abs(spearman([1, 2, 3, 4], [2, 1, 4, 3])) < 1.0


def test_build_pairs_closed_form_two_turn():
    # Hand-built 2-turn trajectory: V̂(s_0)=0.25, V̂(s_1)=0.75, terminal outcome 1.0
    # => ΔV̂_0 = 0.5 (the searching turn found the answer), ΔV̂_1 = 0.25
    pv = {("T", 0): 0.25, ("T", 1): 0.75, ("T", 2): 1.0}
    adv = {("T", 0): 0.6, ("T", 1): 0.2}
    pairs = build_pairs(pv, adv)
    assert len(pairs) == 2
    d = {p["turn_index"]: p["delta_v"] for p in pairs}
    assert d[0] == pytest.approx(0.5) and d[1] == pytest.approx(0.25)
    # missing prefix value -> pair dropped
    pairs2 = build_pairs({("T", 0): 0.5}, adv)
    assert pairs2 == []


def test_stats_perfect_alignment():
    rng = np.random.RandomState(0)
    pairs = []
    for k in range(40):
        for t in range(3):
            dv = rng.uniform(-1, 1)
            pairs.append({"traj_uid": f"t{k}", "turn_index": t,
                          "assigned_advantage": 2.0 * dv, "delta_v": dv})
    s = credit_alignment_stats(pairs, n_boot=200, seed=1)
    assert s["pooled_spearman"] == pytest.approx(1.0)
    assert s["sign_agreement"] == pytest.approx(1.0)
    assert s["pairwise_ranking_accuracy"] == pytest.approx(1.0)
    assert s["pooled_spearman_ci95"][0] > 0.99


def test_stats_anti_alignment_and_uninformative():
    pairs = [{"traj_uid": f"t{k}", "turn_index": t,
              "assigned_advantage": -(t + 0.1 * k), "delta_v": (t + 0.1 * k)}
             for k in range(30) for t in range(2)]
    s = credit_alignment_stats(pairs, n_boot=100, seed=2)
    assert s["pooled_spearman"] == pytest.approx(-1.0)
    assert s["pairwise_ranking_accuracy"] == pytest.approx(0.0)
    # constant advantages (GRPO within a trajectory): ranking accuracy undefined-ish -> 0
    pairs_const = [{"traj_uid": "t", "turn_index": t, "assigned_advantage": 0.7,
                    "delta_v": [0.2, -0.1, 0.4][t]} for t in range(3)]
    s2 = credit_alignment_stats(pairs_const, n_boot=50, seed=3)
    assert s2["pairwise_ranking_accuracy"] == pytest.approx(0.0)  # never strictly correct
    assert s2["n_ranking_pairs"] == 3
