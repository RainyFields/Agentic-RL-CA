# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 3b.3 — critic credit-alignment on the MATCHED pairing (for a fair GiGPO-vs-critic
comparison). Unlike lambda_gate.py (which paired the critic's local TD residual
delta = V(s_{t+1})-V(s_t) against dV-hat_t on INTERIOR turn-pairs only), this uses the critic's
ACTUAL assigned per-turn advantage the way method_align does for GiGPO:

  A_t^critic = R_traj - V_phi(s_t)      (gae_turn with lambda=gamma=1, V(terminal)=0)

paired with dV-hat_t on the SAME terminal-appended pairing (prefix_values_for_pairs), so the
critic and GiGPO numbers are apples-to-apples (same quantity type = assigned advantage; same
turn-pair support incl. the terminal turn).

  python -m credit_assignment.critic_align_matched --diag-root outputs/diag \
      --labels b0_s0_step500 b0_s1_step500 --out outputs/diag/critic_align_matched.json
"""
import argparse
import json
import os

from credit_assignment.diag_plan import prefix_values_for_pairs
from credit_assignment.diagnostic import build_pairs, credit_alignment_stats


def load_label(root, label):
    pv = json.load(open(os.path.join(root, label, "prefix_values.json")))
    cv = json.load(open(os.path.join(root, label, "critic_values.json")))
    vhat_by_traj = {t: {int(d): per for d, per in depths.items()}
                    for t, depths in pv["vhat"].items()}
    terminal = {t: float(b["final_reward"]) for t, b in pv["base"].items()}
    prefix_vals = prefix_values_for_pairs(vhat_by_traj, terminal_rewards=terminal)
    # critic ASSIGNED advantage A_t = R_traj - V_phi(s_t)  (matches GiGPO's assigned-advantage form)
    assigned = {(t, int(d)): terminal[t] - float(v)
                for t, depths in cv["values"].items() for d, v in depths.items()
                if t in terminal}
    return prefix_vals, assigned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-root", default="outputs/diag")
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {}
    for label in args.labels:
        prefix_vals, assigned = load_label(args.diag_root, label)
        stats = credit_alignment_stats(build_pairs(prefix_vals, assigned))
        report[label] = {"method": "critic (A_t=R-V)", **stats}
        lo, hi = stats.get("pooled_spearman_ci95", [float("nan")] * 2)
        rho = stats.get("pooled_spearman", float("nan"))
        print(f"[critic-matched] {label}: rho={rho:+.3f} ci95=[{lo:+.3f},{hi:+.3f}] "
              f"rank_acc={stats.get('pairwise_ranking_accuracy', float('nan')):.3f} "
              f"n_pairs={stats.get('n_pairs')}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"[critic-matched] wrote {args.out}")


if __name__ == "__main__":
    main()
