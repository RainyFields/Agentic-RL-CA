# Standalone unit test for credit_assignment/core_turn_rtg.py (rung-4 turn_rtg estimator).
# Run: python3 tests/test_turn_rtg.py  (torch + numpy only, no pytest needed)
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from credit_assignment.core_turn_rtg import compute_turn_rtg_group_advantage  # noqa: E402

T_RESP = 5


def build(rows):
    """rows: list of (uid, traj, turn, reward, n_resp_tokens). Returns kwargs dict."""
    B = len(rows)
    mask = torch.zeros((B, T_RESP))
    for i, (_, _, _, _, n) in enumerate(rows):
        mask[i, :n] = 1.0
    return dict(
        response_mask=mask,
        rewards=np.array([r for (_, _, _, r, _) in rows], dtype=object),
        uid=np.array([u for (u, _, _, _, _) in rows], dtype=object),
        traj_uid=np.array([t for (_, t, _, _, _) in rows], dtype=object),
        turn_index=np.array([ti for (_, _, ti, _, _) in rows], dtype=object),
    )


def scalars(adv, mask):
    """Recover the per-row broadcast scalar (0 for fully-masked rows)."""
    out = []
    for i in range(adv.shape[0]):
        m = mask[i].bool()
        out.append(float(adv[i][m][0]) if m.any() else 0.0)
    return np.array(out)


def test_alive_at_t_hand_computed():
    # One group, 3 trajectories, lengths 3/2/1, PRM-style rewards.
    #   A: r = [1, 0, 2] -> G = [3, 2, 2]
    #   B: r = [0, 1]    -> G = [1, 1]
    #   C: r = [4]       -> G = [4]
    rows = [
        ("g", "A", 0, 1.0, 3), ("g", "A", 1, 0.0, 2), ("g", "A", 2, 2.0, 4),
        ("g", "B", 0, 0.0, 1), ("g", "B", 1, 1.0, 5),
        ("g", "C", 0, 4.0, 2),
    ]
    adv, ret = compute_turn_rtg_group_advantage(**build(rows))
    a = scalars(adv, build(rows)["response_mask"])
    g = scalars(ret, build(rows)["response_mask"])
    np.testing.assert_allclose(g, [3, 2, 2, 1, 1, 4], atol=1e-6)
    # turn 0 alive {A,B,C}: G = [3,1,4], mu = 8/3, sd(ddof=1) = 1.527525
    sd0 = np.std([3, 1, 4], ddof=1)
    # turn 1 alive {A,B}: G = [2,1], mu = 1.5, sd = 0.7071
    sd1 = np.std([2, 1], ddof=1)
    expect = [(3 - 8 / 3) / sd0, (2 - 1.5) / sd1, 0.0,  # A: turn2 singleton -> 0
              (1 - 8 / 3) / sd0, (1 - 1.5) / sd1,
              (4 - 8 / 3) / sd0]
    np.testing.assert_allclose(a, expect, atol=1e-5)
    print("ok: alive-at-t normalization matches hand computation (incl. lone-survivor turn -> 0)")


def make_two_group_batch(layout):
    """2 groups x 4 trajectories, unequal lengths; totals fixed per trajectory.
    layout='prm': rewards spread across turns; layout='orm': total on last turn only."""
    lengths = {"a": 4, "b": 3, "c": 2, "d": 4, "e": 1, "f": 3, "g": 2, "h": 4}
    groups = {"a": "g1", "b": "g1", "c": "g1", "d": "g1",
              "e": "g2", "f": "g2", "g": "g2", "h": "g2"}
    prm = {"a": [10, 0, 5, 5], "b": [0, 30, 0], "c": [20, 0], "d": [0, 0, 0, 40],
           "e": [70], "f": [10, 10, 10], "g": [0, 50], "h": [25, 25, 0, 0]}
    rows = []
    for t, L in lengths.items():
        for k in range(L):
            if layout == "prm":
                r = prm[t][k]
            else:
                r = sum(prm[t]) if k == L - 1 else 0.0
            rows.append((groups[t], t, k, float(r), 1 + (k % T_RESP)))
    return rows


def test_return_equivalence_at_t0():
    rows_p, rows_o = make_two_group_batch("prm"), make_two_group_batch("orm")
    kp, ko = build(rows_p), build(rows_o)
    ap = scalars(compute_turn_rtg_group_advantage(**kp)[0], kp["response_mask"])
    ao = scalars(compute_turn_rtg_group_advantage(**ko)[0], ko["response_mask"])
    t0 = [i for i, (_, _, ti, _, _) in enumerate(rows_p) if ti == 0]
    np.testing.assert_allclose(ap[t0], ao[t0], atol=1e-6)
    assert not np.allclose(ap, ao), "arms must diverge at later turns"
    print("ok: ORM/PRM advantages identical at t=0 (return equivalence), diverge later")


def test_duplicates():
    rows = make_two_group_batch("prm")
    kp = build(rows)
    base = scalars(compute_turn_rtg_group_advantage(**kp)[0], kp["response_mask"])
    dup_rows = rows + [rows[0], rows[7]]          # adjust_batch-style random duplicates
    kd = build(dup_rows)
    dup = scalars(compute_turn_rtg_group_advantage(**kd)[0], kd["response_mask"])
    np.testing.assert_allclose(dup[: len(rows)], base, atol=1e-6)   # stats unchanged
    np.testing.assert_allclose(dup[len(rows)], base[0], atol=1e-6)  # dup == original
    np.testing.assert_allclose(dup[len(rows) + 1], base[7], atol=1e-6)
    # row order must not matter (_balance_batch): shuffle and compare
    perm = np.random.RandomState(0).permutation(len(dup_rows))
    ks = build([dup_rows[i] for i in perm])
    shuf = scalars(compute_turn_rtg_group_advantage(**ks)[0], ks["response_mask"])
    np.testing.assert_allclose(shuf, dup[perm], atol=1e-6)
    print("ok: adjust_batch duplicates + row reordering handled")


def test_broadcast_shapes():
    rows = make_two_group_batch("prm")
    k = build(rows)
    adv, ret = compute_turn_rtg_group_advantage(**k)
    assert adv.shape == k["response_mask"].shape == ret.shape
    m = k["response_mask"]
    assert torch.all(adv[m == 0] == 0) and torch.all(ret[m == 0] == 0)
    for i in range(adv.shape[0]):  # constant scalar across the row's unmasked tokens
        v = adv[i][m[i].bool()]
        assert v.numel() == 0 or torch.allclose(v, v[0])
    print("ok: broadcast/shape/mask behavior")


def test_singleton_group():
    rows = [("solo", "z", 0, 3.0, 2), ("solo", "z", 1, 1.0, 2)]
    k = build(rows)
    adv, ret = compute_turn_rtg_group_advantage(**k)
    a = scalars(adv, k["response_mask"])
    np.testing.assert_allclose(a, [0.0, 0.0], atol=1e-6)  # no baseline -> no signal
    g = scalars(ret, k["response_mask"])
    np.testing.assert_allclose(g, [4.0, 1.0], atol=1e-6)
    print("ok: group-size-1 guard (advantage 0, returns intact)")


def test_zero_variance_group():
    rows = [("g", "x", 0, 1.0, 1), ("g", "y", 0, 1.0, 1)]  # identical rewards
    k = build(rows)
    adv, _ = compute_turn_rtg_group_advantage(**k)
    a = scalars(adv, k["response_mask"])
    np.testing.assert_allclose(a, [0.0, 0.0], atol=1e-6)   # sigma floor, numerator 0
    print("ok: zero-variance group -> 0 (variance floor)")


if __name__ == "__main__":
    test_alive_at_t_hand_computed()
    test_return_equivalence_at_t0()
    test_duplicates()
    test_broadcast_shapes()
    test_singleton_group()
    test_zero_variance_group()
    print("ALL PASS")
