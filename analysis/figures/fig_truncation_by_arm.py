"""Figure: final truncation clip ratio per arm (4-turn protocol only), ascending.

Bar = mean over seeds of response_length/clip_ratio at step 500 (fraction of turns whose
generation hit MAX_RESPONSE_LENGTH=2048); open circles = individual seeds.
Data source: W&B rainyfields/agentic-rl-ca, all non-toy qwen3-1.7b 4turn_think2k runs,
final logged step (=500) per run, fragments merged across crash-resume run-IDs.
Rerun: python fig_truncation_by_arm.py   (reads fig_truncation_by_arm.csv next to it;
regenerate the CSV with the W&B pull snippet in README.md if runs change).
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from figstyle import PALETTE, apply_publication_style, finalize_figure
from matplotlib import pyplot as plt

# arm -> (display name, family, per-seed finals @500)
ARMS = {
    "token_grpo": ("token-GRPO", "group-relative", [0.0002, 0.0000]),
    "gigpo": ("GiGPO", "group-relative", [0.0002, 0.0002]),
    "b1_shuffle": ("turn-PPO + shuffled step reward", "critic + step reward", [0.0025, 0.0016, 0.0072]),
    "b1": ("turn-PPO + step reward", "critic + step reward", [0.0551, 0.1878, 0.0191]),
    "turn_ppo_b0": ("turn-PPO", "critic", [0.0354, 0.0910, 0.1374]),
    "token_ppo": ("token-PPO", "critic", [0.1014]),
    "hcapo": ("HCAPO", "per-turn amplifier", [0.7543]),
}
FAMILY_COLORS = {
    "group-relative": PALETTE["green_3"],
    "critic + step reward": PALETTE["teal"],
    "critic": PALETTE["blue_main"],
    "per-turn amplifier": PALETTE["red_strong"],
}

rows = sorted(
    ((name, fam, np.mean(seeds), seeds) for name, fam, seeds in ARMS.values()),
    key=lambda r: r[2],
)

with open(os.path.join(HERE, "fig_truncation_by_arm.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "family", "mean_final_clip_ratio", "per_seed_final_clip_ratio"])
    for name, fam, mean, seeds in rows:
        w.writerow([name.replace("\n", " "), fam, f"{mean:.4f}", ";".join(f"{s:.4f}" for s in seeds)])

apply_publication_style(font_size=15)
fig, ax = plt.subplots(figsize=(9.5, 5.2))
x = np.arange(len(rows))
bars = ax.bar(
    x, [r[2] for r in rows], width=0.66,
    color=[FAMILY_COLORS[r[1]] for r in rows], edgecolor="black", linewidth=1.4,
)
for xi, (_, _, mean, seeds) in zip(x, rows):
    ax.scatter([xi] * len(seeds), seeds, facecolors="white", edgecolors="black",
               s=42, linewidths=1.3, zorder=3)
    ax.annotate(f"{mean:.3f}", (xi, max(mean, max(seeds))), textcoords="offset points",
                xytext=(0, 6), ha="center", va="bottom", fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in rows], fontsize=12.5, rotation=18, ha="right")
ax.set_ylabel("Truncation clip ratio @ step 500")
ax.set_ylim(0, 0.85)
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", linewidth=1.2)
           for c in FAMILY_COLORS.values()]
handles.append(plt.Line2D([], [], marker="o", linestyle="none", markerfacecolor="white",
                          markeredgecolor="black", markersize=7))
ax.legend(handles, list(FAMILY_COLORS) + ["individual seed"], loc="upper left", fontsize=12)

print(finalize_figure(fig, os.path.join(HERE, "fig_truncation_by_arm.png")))
