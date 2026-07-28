#!/usr/bin/env python3
"""AlfWorld panels for the cross-environment report: training curve (Fig 1-left), turns barplot
(Fig 2), and accuracy-vs-turns-used (Fig 5-left). Standalone previews; the final Fig 1 / Fig 5
place these beside the SearchQA panels. Pure post-processing over the reward_models eval logs."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "figures"))
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import PALETTE, apply_publication_style, finalize_figure  # noqa: E402
import alfworld_common as A  # noqa: E402

apply_publication_style(font_size=13, axes_linewidth=1.8)
FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(RES, exist_ok=True)

MCOLOR = {"grpo": PALETTE["blue_main"], "gigpo": PALETTE.get("teal", "#2a9d8f"),
          "turn_ppo": PALETTE.get("red_strong", "#d1495b"),
          "token_ppo": PALETTE.get("violet", "#7b5cff")}
TURN_BINS = [0, 6, 8, 12, 20, 100]
TURN_LABELS = ["≤6", "7–8", "9–12", "13–20", ">20"]


def fig_training(tr):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for m in A.ALF_METHOD_ORDER:
        d = tr[m]
        ax.plot(d.step, d.success_rate, lw=2.2, color=MCOLOR[m], label=A.ALF_PRETTY[m], marker="o", ms=3)
    ax.set_xlabel("training step")
    ax.set_ylabel("seen success rate (128-game greedy val)")
    ax.set_title("AlfWorld — training accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=11)
    ax.annotate("PPO trained 150 steps\n(GRPO/GiGPO 300)", xy=(150, 0.5), fontsize=8, color="#666")
    finalize_figure(fig, os.path.join(FIGS, "fig1_left_alfworld_training.png"))


def fig_turns_bar(ep):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    xs = range(len(A.ALF_METHOD_ORDER))
    means = [ep[ep.method == m]["turns"].mean() for m in A.ALF_METHOD_ORDER]
    stds = [ep[ep.method == m]["turns"].std() for m in A.ALF_METHOD_ORDER]
    bars = ax.bar(xs, means, yerr=stds, capsize=5,
                  color=[MCOLOR[m] for m in A.ALF_METHOD_ORDER], alpha=0.9)
    for x, m in zip(xs, A.ALF_METHOD_ORDER):
        v = ep[ep.method == m]["turns"].values
        ax.scatter(np.random.default_rng(x + 1).normal(x, 0.06, len(v)), v, s=6, color="#333", alpha=0.15, zorder=3)
    for b, mu in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, mu + 0.4, f"{mu:.1f}", ha="center", fontsize=11)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([A.ALF_PRETTY[m] for m in A.ALF_METHOD_ORDER])
    ax.set_ylabel("turns used per episode (eval)")
    ax.set_title("AlfWorld — trajectory length")
    ax.grid(True, axis="y", alpha=0.3)
    finalize_figure(fig, os.path.join(FIGS, "fig2_alfworld_turns.png"))


def fig_acc_vs_turns(ep):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ep = ep.copy()
    ep["bin"] = pd.cut(ep["turns"], bins=TURN_BINS, labels=TURN_LABELS)
    rows = []
    for m in A.ALF_METHOD_ORDER:
        d = ep[ep.method == m]
        g = d.groupby("bin", observed=False)["success"].agg(["mean", "size"])
        xs = [i for i, lab in enumerate(TURN_LABELS) if g.loc[lab, "size"] >= 3]
        ys = [g.loc[TURN_LABELS[i], "mean"] for i in xs]
        ax.plot(xs, ys, marker="o", lw=2.2, ms=7, color=MCOLOR[m], label=A.ALF_PRETTY[m])
        for i in xs:
            rows.append({"method": m, "turn_bin": TURN_LABELS[i],
                         "success": float(g.loc[TURN_LABELS[i], "mean"]), "n": int(g.loc[TURN_LABELS[i], "size"])})
    ax.set_xticks(range(len(TURN_LABELS)))
    ax.set_xticklabels(TURN_LABELS)
    ax.set_xlabel("turns used by the model")
    ax.set_ylabel("accuracy (success rate)")
    ax.set_title("AlfWorld — accuracy vs turns used")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=11)
    finalize_figure(fig, os.path.join(FIGS, "fig5_left_alfworld_acc_vs_turns.png"))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "alfworld_acc_vs_turns.csv"), index=False)


def main():
    ep = A.load_alfworld_episodes()
    tr = A.load_alfworld_training()
    print("=== AlfWorld eval table (unseen 134) ===")
    for m in A.ALF_METHOD_ORDER:
        d = ep[ep.method == m]
        print(f"  {A.ALF_PRETTY[m]:9s} success={d.success.mean():.3f}  mean_turns={d.turns.mean():.1f}  "
              f"med_turns={d.turns.median():.0f}  (ckpt step {A.ALF_EVAL_STEP[m]})")
    ep.to_csv(os.path.join(RES, "alfworld_episodes.csv"), index=False)
    fig_training(tr)
    fig_turns_bar(ep)
    fig_acc_vs_turns(ep)
    print("\nfigs ->", os.path.abspath(FIGS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
