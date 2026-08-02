#!/usr/bin/env python3
"""8B ASearcher arm-comparison figures (house style). Data inlined from the val series
(both arms, val@0/25/50/75 on the same 512-question set). Run: python make_figures.py"""
import json, os, sys

sys.path.insert(0, os.path.expanduser("~/.claude/skills/scientific-figure-making/assets"))
from figstyle import apply_publication_style, PALETTE, finalize_figure
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")
os.makedirs(A, exist_ok=True)

DATA = {
    "GRPO": {"steps": [0, 25, 50, 75], "em": [0.195, 0.227, 0.520, 0.520],
             "turns": [8.166, 17.141, 14.621, 17.041], "searches": [7.559, 16.729, 13.682, 16.082]},
    "turnPPO": {"steps": [0, 25, 50, 75], "em": [0.195, 0.234, 0.479, 0.561],
                "turns": [8.166, 11.021, 7.881, 7.486], "searches": [7.559, 10.584, 6.957, 6.486]},
}
COLORS = {"GRPO": PALETTE["blue_main"], "turnPPO": PALETTE["red_strong"]}
apply_publication_style(font_size=15)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
for name, d in DATA.items():
    axes[0].plot(d["steps"], d["em"], marker="o", lw=2.8, color=COLORS[name], label=name)
    axes[1].plot(d["steps"], d["turns"], marker="o", lw=2.8, color=COLORS[name], label=name)
axes[0].set_ylabel("val EM (512 questions, greedy)")
axes[0].set_xlabel("training step")
axes[0].set_ylim(0, 0.65)
axes[0].legend()
axes[1].set_ylabel("val avg turns per episode")
axes[1].set_xlabel("training step")
axes[1].set_ylim(0, 19)
axes[1].legend()
for ax in axes:
    ax.set_xticks([0, 25, 50, 75])
finalize_figure(fig, os.path.join(A, "fig_arms_curves"))
json.dump(DATA, open(os.path.join(A, "arms_val_series.json"), "w"), indent=1)

# efficiency view: EM per search call
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for name, d in DATA.items():
    ax.plot(d["searches"], d["em"], marker="o", lw=2.6, color=COLORS[name], label=name)
    for s, e, st in zip(d["searches"], d["em"], d["steps"]):
        ax.annotate(f"{st}", (s, e), textcoords="offset points", xytext=(6, -3), fontsize=10,
                    color="#555555")
ax.set_xlabel("searches per question (val)")
ax.set_ylabel("val EM")
ax.legend()
finalize_figure(fig, os.path.join(A, "fig_em_vs_search"))
print("figures written to", A)
