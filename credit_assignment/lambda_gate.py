# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Lambda-sweep gate readout (docs/plan_lambda_sweep.md §2a) — pure CPU.

For each diag label: pairs the critic's one-step deltas dV_phi_t = V_phi(s_{t+1}) -
V_phi(s_t) (critic_values.json, from critic_score.py) with the MC continuation deltas
dVhat_t (prefix_values.json) on the SAME prefixes, and runs
diagnostic.credit_alignment_stats (pooled Spearman over turn-pairs, trajectory-level
bootstrap CIs). Pairs use only depths where BOTH d and d+1 were snapshotted (non-terminal
turn-pairs) so the two series are computed on identical support.

  python -m credit_assignment.lambda_gate --diag-root outputs/diag \
      --labels b0_s1_step100 b0_s1_step200 b0_s1_step500 [b0_s0_step500] \
      --gate-labels b0_s1_step100 b0_s1_step200 b0_s1_step500

Trigger (pre-registered, adapted set per decision_log 2026-07-16): pooled Spearman > 0.2
with 95% CI excluding 0 at >= 2 of the 3 --gate-labels checkpoints.
"""
import argparse
import json
import os

from credit_assignment.diagnostic import build_pairs, credit_alignment_stats


def load_label(root, label):
    pv = json.load(open(os.path.join(root, label, "prefix_values.json")))
    cv = json.load(open(os.path.join(root, label, "critic_values.json")))
    vhat = {(t, int(d)): per["vhat"] for t, depths in pv["vhat"].items()
            for d, per in depths.items()}
    critic = {(t, int(d)): v for t, depths in cv["values"].items()
              for d, v in depths.items()}
    # assigned_advantage := critic one-step delta on depths with both d and d+1 scored
    critic_delta = {}
    for (t, d), v in critic.items():
        nxt = critic.get((t, d + 1))
        if nxt is not None:
            critic_delta[(t, d)] = nxt - v
    return vhat, critic_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-root", default="outputs/diag")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--gate-labels", nargs="+", default=None,
                    help="subset the trigger is evaluated on (default: all --labels)")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report, passes = {}, {}
    for label in args.labels:
        vhat, critic_delta = load_label(args.diag_root, label)
        stats = credit_alignment_stats(build_pairs(vhat, critic_delta))
        report[label] = stats
        lo, hi = stats.get("pooled_spearman_ci95", [float("nan")] * 2)
        rho = stats.get("pooled_spearman", float("nan"))
        passes[label] = bool(rho > args.threshold and (lo > 0 or hi < 0))
        print(f"[gate] {label}: rho={rho:.3f} ci95=[{lo:.3f},{hi:.3f}] "
              f"n_pairs={stats.get('n_pairs')} -> {'PASS' if passes[label] else 'fail'}")

    gate_set = args.gate_labels or args.labels
    n_pass = sum(passes[g] for g in gate_set)
    unlocked = n_pass >= 2
    print(f"[gate] TRIGGER: {n_pass}/{len(gate_set)} gate checkpoints pass "
          f"-> lambda sweep {'UNLOCKED' if unlocked else 'DROPPED'}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"per_label": report, "passes": passes,
                       "gate_labels": gate_set, "unlocked": unlocked}, fh, indent=1)
        print(f"[gate] wrote {args.out}")


if __name__ == "__main__":
    main()
