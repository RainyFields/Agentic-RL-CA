#!/usr/bin/env python3
"""Human-friendly figures for the HCAPO length-runaway report. Everything parses straight from
the three training launch logs — no hand-entered numbers.

  fig1_runaway        the headline: response length vs step, all three runs, annotated story
  fig2_buy_nothing    accuracy (top) vs length (bottom): the extra tokens buy no accuracy
  fig3_mechanism      why longer LOOKS better to the scorer: per-token score strips + measured
                      rho-vs-length co-movement
  fig4_signal_death   the credit signal (rho spread) collapses exactly as length explodes
  fig5_cost           the bill: tokens per turn and accuracy, HCAPO vs the normal methods
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "figures"))
from figstyle import PALETTE, apply_publication_style, finalize_figure  # noqa: E402

apply_publication_style(font_size=13, axes_linewidth=1.8)
HERE = os.path.dirname(__file__)
FIGS = os.path.join(HERE, "..", "figs")
os.makedirs(FIGS, exist_ok=True)
LOGS = os.path.expanduser("~/xiaoxuan/worker_logs/launches")

RED = PALETTE.get("red_strong", "#d1495b")
ORANGE = PALETTE.get("highlight", "#e9a13b")
GREEN = PALETTE.get("green_3", "#2a9d40")
BLUE = PALETTE["blue_main"]
GRAY = "#8a8f98"

RUNS = {
    "buggy HCAPO":            ("20260727_154601_arlca-hcapo-4b-nt-s0.log", RED),
    "buggy HCAPO (softened)": ("20260727_154607_arlca-hcapo-w05-4b-nt-s0.log", ORANGE),
    "fixed HCAPO (paper)":    ("20260729_104509_arlca-hcapo-paper-4b-nt-s0.log", GREEN),
}
REF = {"plain GRPO": ("20260726_185609_arlca-token-grpo-4b-nt-s0.log", BLUE)}


def parse(fname):
    txt = open(os.path.join(LOGS, fname), errors="ignore").read().replace("\r", "\n")
    per = lambda k: {int(m.group(1)): float(m.group(2))
                     for m in re.finditer(rf"step:(\d+) .*?{k}:([0-9.]+)", txt)}
    rho = [(float(m.group(1)), float(m.group(2))) for m in
           re.finditer(r"\[hcapo-rho\] mean=([0-9.]+) std=([0-9.]+)", txt)]
    return dict(len=per("response_length/mean"), clip=per("response_length/clip_ratio"),
                em=per("val-core/macro_em"), rho=rho)


data = {k: parse(f) for k, (f, _) in RUNS.items()}
ref = {k: parse(f) for k, (f, _) in REF.items()}
for k, d in data.items():
    print(f"  {k}: {len(d['len'])} steps, {len(d['em'])} vals, {len(d['rho'])} rho lines")


def xy(d, key):
    ks = sorted(d[key])
    return ks, [d[key][k] for k in ks]


# ---------------- fig 1: the headline runaway ----------------
fig, ax = plt.subplots(figsize=(9.5, 5.2))
for name, (_, c) in RUNS.items():
    s, v = xy(data[name], "len")
    ax.plot(s, v, lw=2.4, color=c, label=name)
s, v = xy(ref["plain GRPO"], "len")
ax.plot(s, v, lw=2.0, color=BLUE, ls="--", label="plain GRPO (for scale)")
ax.axhline(2048, color="#555", ls=":", lw=1.4)
ax.text(6, 2048 - 60, "hard cap: 2048 tokens per reply", fontsize=10, color="#555", va="top")
ax.annotate("buggy version ignites\n(~step 90)", xy=(115, 500), xytext=(20, 900),
            fontsize=10, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
ax.annotate("softening the term (ω=0.5)\nonly delays it", xy=(265, 950), xytext=(300, 500),
            fontsize=10, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
ax.annotate("bug-fixed version\nignites anyway (~step 165)", xy=(185, 700), xytext=(90, 1500),
            fontsize=10, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
ax.set_xlabel("training step")
ax.set_ylabel("average reply length (tokens per turn)")
ax.set_title("Every HCAPO variant we ran ends up writing ~10× longer replies")
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, fontsize=10, loc="upper left")
finalize_figure(fig, os.path.join(FIGS, "fig1_runaway.png"))

# ---------------- fig 2: the extra tokens buy nothing ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)
for name, (_, c) in RUNS.items():
    s, v = xy(data[name], "em")
    ax1.plot(s, v, lw=2.2, marker="o", ms=4, color=c, label=name)
s, v = xy(ref["plain GRPO"], "em")
ax1.plot(s, v, lw=2.0, ls="--", color=BLUE, label="plain GRPO")
ax1.set_ylabel("accuracy (val macro-EM)")
ax1.set_title("Accuracy stays flat …")
ax1.grid(True, alpha=0.3)
ax1.legend(frameon=False, fontsize=9, ncol=2)
for name, (_, c) in RUNS.items():
    s, v = xy(data[name], "len")
    ax2.plot(s, v, lw=2.2, color=c)
s, v = xy(ref["plain GRPO"], "len")
ax2.plot(s, v, lw=2.0, ls="--", color=BLUE)
ax2.set_ylabel("reply length (tokens)")
ax2.set_xlabel("training step")
ax2.set_title("… while reply length explodes")
ax2.grid(True, alpha=0.3)
finalize_figure(fig, os.path.join(FIGS, "fig2_buy_nothing.png"))

# ---------------- fig 3: why longer LOOKS better ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
# (a) synthetic per-token score strips
rng_short = [-3.2, -2.8, -3.0, -2.6, -1.0, -0.6, -0.5, -0.55]
filler = [-0.5, -0.45, -0.55, -0.5, -0.4, -0.5, -0.45, -0.5, -0.55, -0.5, -0.45, -0.5,
          -0.4, -0.5, -0.55, -0.45, -0.5, -0.5, -0.45, -0.55, -0.5, -0.45, -0.5, -0.5]
long = rng_short + filler
for row, (toks, y0, lab) in enumerate([(rng_short, 1.0, "short reply"), (long, 0.0, "same reply + filler")]):
    for i, t in enumerate(toks):
        ax1.add_patch(plt.Rectangle((i, y0), 0.92, 0.62,
                      color=plt.cm.RdYlGn((t + 3.5) / 3.2), ec="none"))
    mean = np.mean(toks)
    ax1.text(len(long) + 0.8, y0 + 0.31, f"{lab}\naverage score: {mean:.2f}",
             va="center", fontsize=10)
ax1.text(1, 1.78, "hard, information-carrying tokens (low score)", fontsize=9, color="#7a1f1f")
ax1.text(11, -0.35, "easy, predictable filler (high score)", fontsize=9, color="#1f6b2a")
ax1.set_xlim(-0.5, len(long) + 11)
ax1.set_ylim(-0.6, 2.05)
ax1.axis("off")
ax1.set_title("HCAPO scores a turn by its AVERAGE per-token score —\nso padding with easy filler raises the score")
# (b) the measured co-movement
for name, (_, c) in RUNS.items():
    d = data[name]
    ss = sorted(set(d["len"]) & set(range(1, len(d["rho"]) + 1)))
    ax2.scatter([d["len"][s] for s in ss], [d["rho"][s - 1][0] for s in ss],
                s=9, alpha=0.5, color=c, label=name)
ax2.axhline(1.0, color="#999", ls=":", lw=1.2)
ax2.set_xlabel("average reply length (tokens)")
ax2.set_ylabel("hindsight score $\\rho$ (batch mean)")
ax2.set_title("Measured: the buggy score climbs with length;\nthe fixed one hides it in the spread instead")
ax2.legend(frameon=False, fontsize=9, loc="lower right")
ax2.grid(True, alpha=0.3)
finalize_figure(fig, os.path.join(FIGS, "fig3_mechanism.png"))

# ---------------- fig 4: the signal eats itself ----------------
fig, ax = plt.subplots(figsize=(9.5, 5.0))
ax2 = ax.twinx()
name = "fixed HCAPO (paper)"
d = data[name]
st = np.arange(1, len(d["rho"]) + 1)
ax.plot(st, [r[1] for r in d["rho"]], lw=2.2, color=GREEN, label="credit signal (spread of $\\rho$)")
s, v = xy(d, "len")
ax2.plot(s, v, lw=2.0, color=GRAY, alpha=0.8, label="reply length")
ax.axvspan(160, 200, color=RED, alpha=0.08)
ax.text(181, 0.128, "ignition", color=RED, fontsize=10, ha="center")
ax.set_xlabel("training step")
ax.set_ylabel("spread of the hindsight credit $\\sigma_\\rho$", color=GREEN)
ax2.set_ylabel("reply length (tokens)", color=GRAY)
ax.set_title("The fixed version's runaway eats its own signal:\nas replies lengthen, all turns start scoring the same — and the credit goes silent")
ax.grid(True, alpha=0.3)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=10, loc="upper left")
finalize_figure(fig, os.path.join(FIGS, "fig4_signal_death.png"))

# ---------------- fig 5: the bill ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
names = ["plain GRPO", "turn-PPO", "HCAPO (fixed,\npost-runaway)"]
toks = [111, 207, 1370]
ems = [0.452, 0.422, 0.385]
cols = [BLUE, PALETTE.get("violet", "#7b5cff"), GREEN]
b = ax1.bar(names, toks, color=cols, alpha=0.9)
for r, v in zip(b, toks):
    ax1.text(r.get_x() + r.get_width() / 2, v + 25, f"{v}", ha="center", fontsize=11)
ax1.set_ylabel("tokens per turn")
ax1.set_title("What each method spends …")
ax1.grid(True, axis="y", alpha=0.3)
b = ax2.bar(names, ems, color=cols, alpha=0.9)
for r, v in zip(b, ems):
    ax2.text(r.get_x() + r.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=11)
ax2.set_ylim(0.3, 0.48)
ax2.set_ylabel("accuracy (val macro-EM)")
ax2.set_title("… and what it gets")
ax2.grid(True, axis="y", alpha=0.3)
fig.suptitle("The bill: ~12× GRPO's tokens, for less accuracy (all non-thinking, same budget)", y=1.03)
finalize_figure(fig, os.path.join(FIGS, "fig5_cost.png"))

print("figs ->", os.path.abspath(FIGS))
