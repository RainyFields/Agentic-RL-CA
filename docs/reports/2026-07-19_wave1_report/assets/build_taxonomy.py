#!/usr/bin/env python3
"""F1 — method taxonomy schematic: place the Wave-1 arms on the survey's
granularity x methodology grid (arXiv 2604.09459 Fig 2 / CREDIT_ASSIGNMENT_CHECKLIST §4).

Doubles as the paper's Fig 1. Cool = critic-free group-relative (the winners); warm = critic;
amber = per-turn amplifier; grey chips = second-wave methods (not in Wave 1). The dashed arrow
is the survey's fine-reasoning -> coarse-agentic evolution trend.

Rerun:  python3 build_taxonomy.py
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/.claude/skills/scientific-figure-making/assets"))
from figstyle import PALETTE, apply_publication_style, finalize_figure  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
apply_publication_style(font_size=13, axes_linewidth=1.8)

# axes (checklist §4). methodology = columns (x), granularity = rows (y, top = finest)
COLS = ["Monte-Carlo\n(group-relative)", "Temporal-Diff.\n(learned critic)",
        "Model-based /\nLLM-as-critic", "Game-\ntheoretic", "Information-\ntheoretic"]
ROWS = ["Token", "Segment", "Step / Turn", "Multi-agent"]

# chip: (label, col, row, colorkey, wave)
CF = PALETTE["teal"]; CFB = PALETTE["blue_main"]; AMP = PALETTE["highlight"]
CR = PALETTE["violet"]; STEP = PALETTE["red_strong"]; GREY = "#c7c7c7"
CHIPS = [
    ("token-GRPO", 0, 0, CFB, 1),
    ("token-PPO", 1, 0, PALETTE["neutral"], 1),
    ("GiGPO", 0, 2, CF, 1),
    ("HCAPO", 0.0, 2.30, AMP, 1),         # MC col, per-turn amplifier (slight offset)
    ("turn-PPO (B0)", 1, 2, CR, 1),
    ("B1 / B1-shuffle\n(+ step reward)", 1.0, 2.30, STEP, 1),
    # second wave (roadmap, not run in Wave 1)
    ("B2-lite\n(step judge)", 2, 2, GREY, 2),
    ("CARL", 3, 2, GREY, 2),
    ("StepSearch", 4, 2, GREY, 2),
]

fig, ax = plt.subplots(figsize=(11.2, 5.4))
nC, nR = len(COLS), len(ROWS)
# grid
for i in range(nC + 1):
    ax.axvline(i, color="#e3e3e3", lw=1.0, zorder=0)
for j in range(nR + 1):
    ax.axhline(j, color="#e3e3e3", lw=1.0, zorder=0)

def chip(label, cx, cy, color, wave):
    y = (nR - 1) - cy  # invert so Token is on top
    w, h = 0.86, 0.30
    fc = color if wave == 1 else "white"
    ec = "black" if wave == 1 else "#999"
    box = FancyBboxPatch((cx + 0.5 - w / 2, y + 0.5 - h / 2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.06",
                         linewidth=1.4, edgecolor=ec, facecolor=fc,
                         linestyle="-" if wave == 1 else "--", zorder=3,
                         alpha=1.0 if wave == 1 else 0.85)
    ax.add_patch(box)
    txt = "black"
    if wave == 1 and color in (CFB, CR, STEP, CF):
        txt = "white"
    ax.text(cx + 0.5, y + 0.5, label, ha="center", va="center", fontsize=9.2,
            color=txt if wave == 1 else "#666", fontweight="bold" if wave == 1 else "normal",
            zorder=4)

for c in CHIPS:
    chip(*c)

# survey evolution-trend arrow (upper-left fine/reasoning -> lower-right coarse/agentic)
ax.add_patch(FancyArrowPatch((0.35, nR - 0.35), (nC - 0.5, 0.7),
             arrowstyle="-|>", mutation_scale=22, lw=2.2, color="#9a9a9a",
             linestyle=(0, (6, 4)), zorder=1))
ax.text(nC - 1.15, 0.42, "reasoning $\\rightarrow$ agentic\n(survey trend)", fontsize=9,
        color="#8a8a8a", ha="center", style="italic")

ax.set_xticks([i + 0.5 for i in range(nC)]); ax.set_xticklabels(COLS, fontsize=10)
ax.set_yticks([j + 0.5 for j in range(nR)]); ax.set_yticklabels(ROWS[::-1], fontsize=11)
ax.set_xlim(0, nC); ax.set_ylim(0, nR)
ax.set_xlabel("methodology  (how credit is computed)", fontsize=12)
ax.set_ylabel("granularity", fontsize=12)
ax.set_title("F1 — Wave-1 arms on the credit-assignment taxonomy grid "
             "(filled = run; dashed grey = second-wave roadmap)", fontsize=11.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
finalize_figure(fig, os.path.join(FIGS, "fig_taxonomy.png"))
print("taxonomy figure written")
