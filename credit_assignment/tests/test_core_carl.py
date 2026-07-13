# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""CARL unit checks (plan §Verification): hand-built toy tree backup/criticality/zeroing
vs drop; duplicated-row case; deep-chain recursion safety."""
import pytest

from credit_assignment.core_carl import compute_carl_edge_advantages, compute_node_values


def toy_tree_rows():
    r"""Hand-built tree:
        root ── a ── A ── c ── C(leaf r=1)
             │        └── d ── D(leaf r=0)
             └─ b ── B ── e ── E(leaf r=0)
    V(C)=1, V(D)=0 -> V(A)=0.5; V(E)=0 -> V(B)=0; V(root)=mean(0.5,0)=0.25.
    critical: root (2 children), A (2 children); B has 1 child -> non-critical.
    """
    return [
        {"key": ("t1", 0), "src": "root", "dst": "A", "terminal_reward": None},
        {"key": ("t1", 1), "src": "A", "dst": "C", "terminal_reward": 1.0},
        {"key": ("t2", 0), "src": "root", "dst": "A", "terminal_reward": None},
        {"key": ("t2", 1), "src": "A", "dst": "D", "terminal_reward": 0.0},
        {"key": ("t3", 0), "src": "root", "dst": "B", "terminal_reward": None},
        {"key": ("t3", 1), "src": "B", "dst": "E", "terminal_reward": 0.0},
    ]


def test_toy_tree_values_and_advantages():
    adv, kept, m = compute_carl_edge_advantages(toy_tree_rows())
    # A(root->A) = V(A)-V(root) = 0.5-0.25 = 0.25 ; A(root->B) = -0.25
    assert adv[("t1", 0)] == pytest.approx(0.25)
    assert adv[("t3", 0)] == pytest.approx(-0.25)
    # A(A->C) = 1-0.5 ; A(A->D) = -0.5
    assert adv[("t1", 1)] == pytest.approx(0.5)
    assert adv[("t2", 1)] == pytest.approx(-0.5)
    # B non-critical -> dropped (paper Eq. 13 behavior)
    assert kept[("t3", 1)] is False and adv[("t3", 1)] == 0.0
    assert kept[("t1", 0)] and kept[("t1", 1)] and kept[("t2", 1)]
    assert m["carl/critical_state_frac"] == pytest.approx(2 / 3)  # {root, A} of {root, A, B}
    assert m["carl/kept_row_frac"] == pytest.approx(5 / 6)
    assert m["carl/n_leaves"] == 3


def test_zeroing_ablation_keeps_rows():
    adv, kept, m = compute_carl_edge_advantages(toy_tree_rows(), drop_noncritical=False)
    assert kept[("t3", 1)] is True and adv[("t3", 1)] == 0.0
    assert m["carl/kept_row_frac"] == 1.0


def test_duplicated_rows_deduped():
    rows = toy_tree_rows()
    dup = rows + [dict(rows[0]), dict(rows[1])]  # adjust_batch copy-duplicates
    adv, kept, m = compute_carl_edge_advantages(dup)
    adv0, _, m0 = compute_carl_edge_advantages(rows)
    assert adv == adv0
    assert m["carl/n_rows_unique"] == 6 and m["carl/n_rows_input"] == 8
    assert m["carl/tree_nodes"] == m0["carl/tree_nodes"]


def test_shared_node_merging_across_trajectories():
    # two trajectories that reach the SAME dst node (byte-identical next prompt) then split
    rows = [
        {"key": ("t1", 0), "src": "root", "dst": "M", "terminal_reward": None},
        {"key": ("t2", 0), "src": "root", "dst": "M", "terminal_reward": None},
        {"key": ("t1", 1), "src": "M", "dst": "X", "terminal_reward": 1.0},
        {"key": ("t2", 1), "src": "M", "dst": "Y", "terminal_reward": 0.0},
    ]
    adv, kept, m = compute_carl_edge_advantages(rows)
    # root has ONE distinct child (M) despite two rows -> non-critical, dropped
    assert kept[("t1", 0)] is False and kept[("t2", 0)] is False
    assert adv[("t1", 1)] == pytest.approx(0.5) and adv[("t2", 1)] == pytest.approx(-0.5)


def test_deep_chain_no_recursion_error():
    rows = [{"key": ("t", i), "src": f"n{i}", "dst": f"n{i+1}",
             "terminal_reward": (1.0 if i == 4999 else None)} for i in range(5000)]
    adv, kept, m = compute_carl_edge_advantages(rows)
    assert all(v == 0.0 for v in adv.values())  # pure chain: nothing critical
    assert m["carl/kept_row_frac"] == 0.0


def test_leaf_without_reward_asserts():
    rows = [{"key": ("t", 0), "src": "r", "dst": "L", "terminal_reward": None}]
    with pytest.raises(AssertionError):
        compute_carl_edge_advantages(rows)
