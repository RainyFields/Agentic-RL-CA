# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 3b.3 — per-method credit-alignment readout (pure CPU).

Pairs a method's ASSIGNED per-turn advantage A_t (diag_runner_methods -> method_values.json)
with the ideal per-turn credit dV-hat_t = V-hat(s_{t+1}) - V-hat(s_t) (prefix_values.json, MC
continuations from the SAME checkpoint; terminal state appended with the observed outcome so the
last turn's dV-hat is R(tau) - V-hat(s_{T-1})), then runs diagnostic.credit_alignment_stats
(pooled Spearman over turn-pairs + trajectory-level bootstrap CIs). This is the RQ2 quantitative
teeth: does the method assign per-turn credit that tracks true continuation-value progress?

  python -m credit_assignment.method_align --diag-root outputs/diag_methods \
      --labels gigpo_1p7b_step500 [gigpo_1p7b_step100 ...] --out outputs/diag_methods/align.json
"""
import argparse
import json
import os

from credit_assignment.diag_plan import prefix_values_for_pairs
from credit_assignment.diagnostic import build_pairs, credit_alignment_stats


def load_label(root, label):
    pv = json.load(open(os.path.join(root, label, "prefix_values.json")))
    mv = json.load(open(os.path.join(root, label, "method_values.json")))
    vhat_by_traj = {t: {int(d): per for d, per in depths.items()}
                    for t, depths in pv["vhat"].items()}
    terminal = {t: float(b["final_reward"]) for t, b in pv["base"].items()}
    prefix_vals = prefix_values_for_pairs(vhat_by_traj, terminal_rewards=terminal)
    assigned = {(t, int(turn)): float(a)
                for t, turns in mv["values"].items() for turn, a in turns.items()}
    return prefix_vals, assigned, mv.get("method", "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-root", default="outputs/diag_methods")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {}
    for label in args.labels:
        prefix_vals, assigned, method = load_label(args.diag_root, label)
        stats = credit_alignment_stats(build_pairs(prefix_vals, assigned))
        report[label] = {"method": method, **stats}
        lo, hi = stats.get("pooled_spearman_ci95", [float("nan")] * 2)
        rho = stats.get("pooled_spearman", float("nan"))
        print(f"[align] {label} ({method}): rho={rho:+.3f} ci95=[{lo:+.3f},{hi:+.3f}] "
              f"sign_agree={stats.get('sign_agreement', float('nan')):.3f} "
              f"rank_acc={stats.get('pairwise_ranking_accuracy', float('nan')):.3f} "
              f"n_pairs={stats.get('n_pairs')}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"[align] wrote {args.out}")


if __name__ == "__main__":
    main()
