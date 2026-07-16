#!/usr/bin/env python3
"""Build the proposal's preliminary-result figures from csv/<run>.csv.

Rerun:  python3 extract_metrics.py && python3 build_figures.py
Outputs figs/*.{png,pdf}. House style via the scientific-figure-making skill's figstyle.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/.claude/skills/scientific-figure-making/assets"))
from figstyle import PALETTE, apply_publication_style, finalize_figure, make_lines  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "csv")
FIGS = os.path.join(HERE, "figs")
BASE_EM = 0.2246  # untrained Qwen3-1.7B on val_2048

apply_publication_style(font_size=13, axes_linewidth=1.8)


def load(run):
    rows = {}
    with open(os.path.join(CSV, f"{run}.csv")) as f:
        for r in csv.DictReader(f):
            rows[int(r["step"])] = r
    return rows


def col(rows, name, every=1):
    x, y = [], []
    for s in sorted(rows):
        v = rows[s].get(name, "")
        if v not in ("", None):
            x.append(s), y.append(float(v))
    return np.array(x[::every]), np.array(y[::every])


R = {r: load(r) for r in [
    "token_grpo_s0", "gigpo_s0", "token_ppo_s0",
    "turn_ppo_b0_s0", "turn_ppo_b0_s1",
    "b1_s0", "b1_s1", "b1_shuffle_s0", "b1_shuffle_s1"]}

C = {
    "token_grpo": PALETTE["blue_main"],
    "gigpo": PALETTE["teal"],
    "token_ppo": PALETTE["neutral"],
    "b0": PALETTE["violet"],
    "b1": PALETTE["red_strong"],
    "b1_shuffle": PALETTE["green_3"],
}

# ---------------- Fig 1: validation learning curves (2 panels) ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)
ax = axes[0]
s = []
for run, lab, key in [("token_grpo_s0", "token-GRPO", "token_grpo"),
                      ("gigpo_s0", "GiGPO", "gigpo"),
                      ("token_ppo_s0", "token-PPO", "token_ppo"),
                      ("turn_ppo_b0_s0", "turn-PPO (B0), 2 seeds", "b0"),
                      ("turn_ppo_b0_s1", None, "b0")]:
    x, y = col(R[run], "val_macro_em")
    s.append({"x": x, "y": y, "label": lab, "color": C[key]})
make_lines(ax, s, ylabel="val-2048 macro-EM", xlabel="training step", smooth=False, raw_alpha=0)
ax.axhline(BASE_EM, color="black", ls=":", lw=1.4)
ax.text(15, BASE_EM + 0.004, "untrained", ha="left", fontsize=10)
ax.set_title("(a) Credit granularity & outcome-only CA (RQ1/RQ2)", fontsize=12)
ax.legend(loc="lower right", fontsize=10)

ax = axes[1]
s = []
for run, lab, key in [("turn_ppo_b0_s0", "B0 (sparse)", "b0"), ("turn_ppo_b0_s1", None, "b0"),
                      ("b1_s0", "B1 (progress reward)", "b1"), ("b1_s1", None, "b1"),
                      ("b1_shuffle_s0", "B1-shuffle (control)", "b1_shuffle"),
                      ("b1_shuffle_s1", None, "b1_shuffle")]:
    x, y = col(R[run], "val_macro_em")
    s.append({"x": x, "y": y, "label": lab, "color": C[key]})
make_lines(ax, s, xlabel="training step", smooth=False, raw_alpha=0)
ax.axhline(BASE_EM, color="black", ls=":", lw=1.4)
ax.set_title("(b) Progress supervision triad (RQ3), 2 seeds/arm", fontsize=12)
ax.legend(loc="lower right", fontsize=10)
finalize_figure(fig, os.path.join(FIGS, "fig_learning_curves.png"))

# ---------------- Fig 2: training reward + average turns ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
ARMS = [("token_grpo_s0", "token-GRPO", "token_grpo"),
        ("gigpo_s0", "GiGPO", "gigpo"),
        ("turn_ppo_b0_s0", "turn-PPO (B0)", "b0"),
        ("b1_s0", "B1", "b1"),
        ("b1_shuffle_s0", "B1-shuffle", "b1_shuffle")]
ax = axes[0]
s = [{"x": x, "y": y, "label": lab, "color": C[k]}
     for run, lab, k in ARMS for x, y in [col(R[run], "reward_mean", every=2)]]
make_lines(ax, s, ylabel="mean episode reward (train)", xlabel="training step",
           smooth=True, raw_alpha=0.10, markers=False)
ax.set_title("(a) Training reward", fontsize=12)
ax.legend(loc="lower right", fontsize=10)

ax = axes[1]
s = [{"x": x, "y": y, "label": lab, "color": C[k]}
     for run, lab, k in ARMS for x, y in [col(R[run], "avg_turns", every=2)]]
make_lines(ax, s, ylabel="avg turns / trajectory (train)", xlabel="training step",
           smooth=True, raw_alpha=0.10, markers=False)
ax.set_title("(b) Average number of turns", fontsize=12)
ax.set_ylim(1.0, 4.05)
finalize_figure(fig, os.path.join(FIGS, "fig_reward_turns.png"))

# ---------------- Fig 3: truncation tripwire vs late-run val ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
TRIP = [("b1_s1", "B1 s1", C["b1"], "-"),
        ("b1_s0", "B1 s0", C["b1"], "--"),
        ("turn_ppo_b0_s1", "B0 s1", C["b0"], "-"),
        ("b1_shuffle_s1", "B1-shuffle s1", C["b1_shuffle"], "-"),
        ("b1_shuffle_s0", "B1-shuffle s0", C["b1_shuffle"], "--")]
ax = axes[0]
for run, lab, c, ls in TRIP:
    x, y = col(R[run], "clip_ratio", every=2)
    ax.plot(x, y, color=c, ls=ls, lw=2.0, label=lab)
ax.set_xlabel("training step"); ax.set_ylabel("response truncation clip ratio")
ax.set_title("(a) Truncation tripwire (must fall to $\\approx$0)", fontsize=12)
ax.legend(loc="upper left", fontsize=10)

ax = axes[1]
for run, lab, c, ls in [("b1_s1", "B1 s1", C["b1"], "-"),
                        ("b1_s0", "B1 s0", C["b1"], "--"),
                        ("b1_shuffle_s1", "B1-shuffle s1", C["b1_shuffle"], "-"),
                        ("b1_shuffle_s0", "B1-shuffle s0", C["b1_shuffle"], "--")]:
    x, y = col(R[run], "val_macro_em")
    m = x >= 250
    ax.plot(x[m], y[m], color=c, ls=ls, lw=2.2, marker="o", markersize=4, label=lab)
ax.set_xlabel("training step"); ax.set_ylabel("val-2048 macro-EM")
ax.set_title("(b) Late-run val: B1 drifts, shuffle rises", fontsize=12)
ax.legend(loc="lower left", fontsize=10)
finalize_figure(fig, os.path.join(FIGS, "fig_tripwire.png"))

# ---------------- Fig 4: final / current val EM per arm ----------------
fig, ax = plt.subplots(figsize=(7.6, 4.2))
bars = [
    ("GiGPO (475*)", 0.398, C["gigpo"]),
    ("token-GRPO", 0.391, C["token_grpo"]),
    ("B1-shuffle s1", 0.363, C["b1_shuffle"]),
    ("B1 s0", 0.352, C["b1"]),
    ("B0 s0", 0.343, C["b0"]),
    ("B0 s1", 0.330, C["b0"]),
    ("B1 s1", 0.318, C["b1"]),
    ("token-PPO (375*)", 0.315, C["token_ppo"]),
]
xs = np.arange(len(bars))
bb = ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars],
            edgecolor="black", linewidth=1.2, width=0.68)
for b in bb:
    ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                textcoords="offset points", xytext=(0, 3), ha="center", fontsize=10)
ax.set_xticks(xs)
ax.set_xticklabels([b[0] for b in bars], fontsize=10, rotation=22, ha="right")
ax.set_ylabel("val-2048 macro-EM (final ckpt)")
ax.axhline(BASE_EM, color="black", ls=":", lw=1.4)
ax.text(len(bars) - 0.5, BASE_EM + 0.005, "untrained 0.225", ha="right", fontsize=10)
ax.set_ylim(0, 0.44)
ax.set_title("Fixed-budget final val macro-EM (* = still training)", fontsize=12)
finalize_figure(fig, os.path.join(FIGS, "fig_final_bars.png"))

print("figures written to", FIGS)
