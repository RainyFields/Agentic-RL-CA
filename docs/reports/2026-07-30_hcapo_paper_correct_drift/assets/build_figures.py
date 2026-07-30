#!/usr/bin/env python3
"""Figures for the focused report: the PAPER-CORRECT HCAPO only, and why it still drifts.

  fig1_drift      what happened: reply length (with ignition annotation) + accuracy, GRPO for scale
  fig2_ratchet    why: the relative-credit treadmill, as a 3-frame cartoon
  fig3_incentive  the padding incentive measured BEFORE any runaway (healthy-regime rollouts,
                  within-trajectory: hindsight-lift score vs turn length)
  fig4_signal     the drift consumes its own signal: rho spread collapses as length explodes,
                  while the rho MEAN stays pinned at 1 (the invariance that hid it)
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "figures"))
from figstyle import PALETTE, apply_publication_style, finalize_figure  # noqa: E402

apply_publication_style(font_size=13, axes_linewidth=1.8)
HERE = os.path.dirname(__file__)
FIGS = os.path.join(HERE, "..", "figs")
os.makedirs(FIGS, exist_ok=True)
LOGS = os.path.expanduser("~/xiaoxuan/worker_logs/launches")
GREEN = PALETTE.get("green_3", "#2a9d40")
BLUE = PALETTE["blue_main"]
RED = PALETTE.get("red_strong", "#d1495b")
GRAY = "#8a8f98"

def parse(fname):
    txt = open(os.path.join(LOGS, fname), errors="ignore").read().replace("\r", "\n")
    per = lambda k: {int(m.group(1)): float(m.group(2))
                     for m in re.finditer(rf"step:(\d+) .*?{k}:([0-9.]+)", txt)}
    rho = [(float(m.group(1)), float(m.group(2))) for m in
           re.finditer(r"\[hcapo-rho\] mean=([0-9.]+) std=([0-9.]+)", txt)]
    return dict(len=per("response_length/mean"), clip=per("response_length/clip_ratio"),
                em=per("val-core/macro_em"), rho=rho)

hc = parse("20260729_104509_arlca-hcapo-paper-4b-nt-s0.log")
gr = parse("20260726_185609_arlca-token-grpo-4b-nt-s0.log")
print(f"paper-correct: {len(hc['len'])} steps, {len(hc['rho'])} rho lines")

def xy(d, k):
    ks = sorted(d[k]); return ks, [d[k][x] for x in ks]

# ---------------- fig 1: what happened ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.6), sharex=True,
                               gridspec_kw={"height_ratios": [1.3, 1]})
s, v = xy(hc, "len"); ax1.plot(s, v, lw=2.6, color=GREEN, label="HCAPO (paper-correct)")
s, v = xy(gr, "len"); ax1.plot(s, v, lw=2.0, ls="--", color=BLUE, label="plain GRPO (for scale)")
ax1.axhline(2048, color="#555", ls=":", lw=1.4)
ax1.text(6, 2000, "hard cap: 2048 tokens per reply", fontsize=10, color="#555", va="top")
ax1.axvspan(160, 200, color=RED, alpha=0.08)
ax1.annotate("~150 clean steps:\nthe fix looked like it worked", xy=(110, 240), xytext=(15, 1050),
             fontsize=10, color="#333", arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
ax1.annotate("then the drift ignites\n(steps ~160–200: 250 → 1230 tokens)", xy=(186, 800),
             xytext=(215, 320), fontsize=10, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
ax1.set_ylabel("average reply length (tokens)")
ax1.set_title("The paper-correct HCAPO: verified faithful to the paper — and it still drifts")
ax1.grid(True, alpha=0.3); ax1.legend(frameon=False, fontsize=10, loc="upper left")
s, v = xy(hc, "em"); ax2.plot(s, v, lw=2.2, marker="o", ms=4, color=GREEN)
s, v = xy(gr, "em"); ax2.plot(s, v, lw=2.0, ls="--", color=BLUE)
ax2.axvspan(160, 200, color=RED, alpha=0.08)
ax2.set_ylabel("accuracy (val macro-EM)")
ax2.set_xlabel("training step")
ax2.set_title("…while accuracy gains nothing from the extra words")
ax2.grid(True, alpha=0.3)
finalize_figure(fig, os.path.join(FIGS, "fig1_drift.png"))

# ---------------- fig 2: the ratchet cartoon ----------------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
frames = [
    ([5, 8, 6],  "batch 1", "the slightly longer,\nmore padded turn\nwins the comparison"),
    ([8, 11, 9], "batch 2 — everyone longer", "…and the (new) longest\nturn wins again"),
    ([11, 14, 12], "batch 3 — longer still", "no turn ever gains\nnet credit; lengths\nratchet anyway"),
]
for ax, (lens, title, note) in zip(axes, frames):
    rhos = np.array(lens, float)
    rhos = np.exp((rhos - rhos.mean()) / 18)
    rhos = rhos / rhos.mean()
    win = int(np.argmax(lens))
    for i, (L, r) in enumerate(zip(lens, rhos)):
        col = RED if i == win else "#b8bec7"
        ax.barh(i, L, height=0.6, color=col, alpha=0.9)
        ax.text(L + 0.25, i, f"$\\rho$={r:.2f}" + ("  ← reinforced" if i == win else ""),
                va="center", fontsize=10, color=(RED if i == win else "#555"))
    ax.set_xlim(0, 22)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["turn A", "turn B", "turn C"])
    ax.set_xlabel("turn length")
    ax.set_title(title, fontsize=11)
    ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#333",
            bbox=dict(fc="white", ec="#ccc", alpha=0.85, boxstyle="round,pad=0.3"))
    ax.grid(True, axis="x", alpha=0.25)
fig.suptitle("Why the trajectory-average normalisation doesn't save you: credit is relative, "
             "but the policy update is permanent", y=1.06)
finalize_figure(fig, os.path.join(FIGS, "fig2_ratchet.png"))

# ---------------- fig 3: the incentive exists BEFORE the runaway ----------------
d = json.load(open("/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/diag_hcapo/estdiag_floor.json"))
X, Y = [], []
for rec in d["per_traj"]:
    if rec["m_gold"] is None and rec["m_no"] is None:
        continue
    dm = np.array(rec["m_hind"]) - np.array(rec["m_no"])
    L = np.array(rec["length"], float)
    if len(L) < 2:
        continue
    X.append(L - L.mean()); Y.append(dm - dm.mean())
X = np.concatenate(X); Y = np.concatenate(Y)
r = float(np.corrcoef(X, Y)[0, 1])
fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.scatter(X, Y, s=12, alpha=0.35, color=GREEN)
k = (X @ Y) / (X @ X)
xs = np.linspace(X.min(), X.max(), 10)
ax.plot(xs, k * xs, lw=2.4, color=RED, label=f"fit  (within-trajectory r = {r:+.2f})")
ax.axhline(0, color="#999", lw=1, ls=":"); ax.axvline(0, color="#999", lw=1, ls=":")
ax.set_xlabel("turn length, relative to the trajectory's own average (tokens)")
ax.set_ylabel("hindsight score lift, relative to trajectory average")
ax.set_title("Measured before any runaway (healthy-length rollouts):\n"
             "within a trajectory, the LONGER turn already gets the higher hindsight score")
ax.legend(frameon=False, fontsize=10)
ax.grid(True, alpha=0.3)
finalize_figure(fig, os.path.join(FIGS, "fig3_incentive.png"))

# ---------------- fig 4: mean blind, spread dies ----------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)
st = np.arange(1, len(hc["rho"]) + 1)
ax1.plot(st, [r[0] for r in hc["rho"]], lw=2.0, color=GREEN)
ax1.axhline(1.0, color="#999", ls=":", lw=1.2)
ax1.set_ylim(0.9, 1.1)
ax1.set_ylabel("mean of $\\rho$")
ax1.set_title("The average credit ratio is pinned at 1 by construction — the drift is invisible here")
ax1.grid(True, alpha=0.3)
ax1b = ax1.twinx()
s, v = xy(hc, "len"); ax1b.plot(s, v, lw=1.6, color=GRAY, alpha=0.7)
ax1b.set_ylabel("reply length", color=GRAY)
ax2.plot(st, [r[1] for r in hc["rho"]], lw=2.2, color=GREEN)
ax2.axvspan(160, 200, color=RED, alpha=0.08)
ax2.text(180, 0.125, "ignition", color=RED, fontsize=10, ha="center")
ax2.set_ylabel("spread of $\\rho$  ($\\sigma_\\rho$)")
ax2.set_xlabel("training step")
ax2.set_title("…but the SPREAD — the credit signal's entire information content — collapses as padding\n"
              "homogenises the turns (0.135 → 0.030): the race ends when everyone looks the same")
ax2.grid(True, alpha=0.3)
finalize_figure(fig, os.path.join(FIGS, "fig4_signal.png"))

print("figs ->", os.path.abspath(FIGS))
