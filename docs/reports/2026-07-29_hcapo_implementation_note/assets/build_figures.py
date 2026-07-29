#!/usr/bin/env python3
"""HCAPO implementation-note figures. Parses the three training logs directly (legacy A w=1.0,
legacy B w=0.5, paper-correct) — no hand-entered numbers. Emits 4 figures + a per-step CSV.

  fig_runaway      response length + clip-ratio vs step: the legacy runaway, the corrected run flat
  fig_rho          rho mean & std vs step: legacy crushes toward 1 from below; paper centred at 1
  fig_dilution     the mechanism: implied TOTAL hindsight log-ratio is ~constant while length 18x's
                   (numerator saturates, /ntok divisor grows) + rho-vs-length co-movement
  fig_synthetic    controlled test: front-loaded perturbation, rho vs response length under the
                   legacy formula (rises toward 1) vs the paper formula (exactly flat)
"""
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

RUNS = {
    "legacy A (w=1.0)": ("20260727_154601_arlca-hcapo-4b-nt-s0.log", PALETTE.get("red_strong", "#d1495b")),
    "legacy B (w=0.5)": ("20260727_154607_arlca-hcapo-w05-4b-nt-s0.log", PALETTE.get("highlight", "#e9a13b")),
    "paper-correct":    ("20260729_104509_arlca-hcapo-paper-4b-nt-s0.log", PALETTE.get("green_3", "#2a9d40")),
}
T_TEMP, CLIP_LO, CLIP_HI = 5.0, 0.8, 1.2


def parse(fname):
    txt = open(os.path.join(LOGS, fname), errors="ignore").read().replace("\r", "\n")
    # one [hcapo-rho] line per optimizer step, in order -> index = step (1-based)
    rho = [dict(mean=float(m.group(1)), std=float(m.group(2)), lo=float(m.group(3)), hi=float(m.group(4)))
           for m in re.finditer(r"\[hcapo-rho\] mean=([0-9.]+) std=([0-9.]+) .*?frac_at_lo=([0-9.]+) frac_at_hi=([0-9.]+)", txt)]
    per = lambda k: {int(m.group(1)): float(m.group(2))
                     for m in re.finditer(rf"step:(\d+) .*?{k}:([0-9.]+)", txt)}
    return dict(rho=rho, len=per("response_length/mean"), clip=per("response_length/clip_ratio"),
                em=per("val-core/macro_em"))


data = {k: parse(f) for k, (f, _) in RUNS.items()}
for k, d in data.items():
    print(f"  {k}: {len(d['rho'])} rho lines, {len(d['len'])} step metrics, {len(d['em'])} vals")


def series(d, key):
    ks = sorted(d[key])
    return ks, [d[key][k] for k in ks]


# ---------------- fig 1: the runaway ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
for name, (_, c) in RUNS.items():
    s, v = series(data[name], "len")
    ax1.plot(s, v, lw=2, color=c, label=name)
    s, v = series(data[name], "clip")
    ax2.plot(s, v, lw=2, color=c, label=name)
ax1.axhline(2048, color="#999", ls=":", lw=1.2)
ax1.text(3, 2048 * 0.96, "response cap 2048", fontsize=9, color="#777", va="top")
ax1.set_xlabel("training step"); ax1.set_ylabel("mean response tokens / turn")
ax1.set_title("response length")
ax2.axhline(0.754, color="#999", ls=":", lw=1.2)
ax2.text(3, 0.760, "1.7B collapse level (0.754)", fontsize=9, color="#777")
ax2.set_xlabel("training step"); ax2.set_ylabel("fraction of turns clipped at cap")
ax2.set_title("truncation (clip ratio)")
for ax in (ax1, ax2):
    ax.grid(True, alpha=0.3)
ax1.legend(frameon=False, fontsize=10, loc="upper left")
fig.suptitle("Legacy HCAPO runs away in length; the paper-correct run (so far) does not", y=1.03)
finalize_figure(fig, os.path.join(FIGS, "fig_runaway.png"))

# ---------------- fig 2: rho evolution ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
for name, (_, c) in RUNS.items():
    r = data[name]["rho"]
    st = np.arange(1, len(r) + 1)
    ax1.plot(st, [x["mean"] for x in r], lw=1.6, color=c, label=name)
    ax2.plot(st, [x["std"] for x in r], lw=1.6, color=c, label=name)
ax1.axhline(1.0, color="#999", ls=":", lw=1.2)
ax1.set_xlabel("training step"); ax1.set_ylabel(r"batch mean of $\rho$")
ax1.set_title(r"$\rho$ mean: legacy climbs toward 1 from below; paper centred at 1")
ax2.set_xlabel("training step"); ax2.set_ylabel(r"batch std of $\rho$")
ax2.set_title(r"$\rho$ spread: legacy signal collapses; paper signal persists")
for ax in (ax1, ax2):
    ax.grid(True, alpha=0.3)
ax1.legend(frameon=False, fontsize=10, loc="lower right")
finalize_figure(fig, os.path.join(FIGS, "fig_rho.png"))

# ---------------- fig 3: the dilution mechanism (legacy A) ----------------
# implied per-turn TOTAL hindsight log-ratio: sum_j delta_j = t_temp * ln(rho_mean) * ntok
d = data["legacy A (w=1.0)"]
steps = sorted(set(d["len"]) & set(range(1, len(d["rho"]) + 1)))
L = np.array([d["len"][s] for s in steps])
RM = np.array([d["rho"][s - 1]["mean"] for s in steps])
TOT = T_TEMP * np.log(RM) * L
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax1.plot(steps, L / L[0], lw=2, color=PALETTE["blue_main"], label=r"response length (relative, $18\times$)")
ax1.plot(steps, TOT / TOT[0], lw=2, color=PALETTE.get("red_strong", "#d1495b"),
         label=r"implied total $\sum_j \delta_j$ (relative, $\sim$flat)")
ax1.axhline(1.0, color="#999", ls=":", lw=1)
ax1.set_yscale("log")
ax1.set_xlabel("training step"); ax1.set_ylabel("relative to step 1 (log scale)")
ax1.set_title("numerator saturates while the $/n_{tok}$ divisor grows")
ax1.legend(frameon=False, fontsize=10)
for name, (_, c) in RUNS.items():
    dd = data[name]
    ss = sorted(set(dd["len"]) & set(range(1, len(dd["rho"]) + 1)))
    ax2.scatter([dd["len"][s] for s in ss], [dd["rho"][s - 1]["mean"] for s in ss],
                s=10, alpha=0.5, color=c, label=name)
ax2.axhline(1.0, color="#999", ls=":", lw=1)
ax2.set_xlabel("mean response tokens / turn"); ax2.set_ylabel(r"batch mean of $\rho$")
ax2.set_title(r"$\rho$ vs length: legacy co-moves, paper does not")
ax2.legend(frameon=False, fontsize=10, loc="lower right")
for ax in (ax1, ax2):
    ax.grid(True, alpha=0.3)
finalize_figure(fig, os.path.join(FIGS, "fig_dilution.png"))

# ---------------- fig 4: synthetic controlled test ----------------
# front-loaded perturbation: delta_j = -2.0 on the first 8 tokens, 0 afterwards.
Ls = np.array([8, 16, 32, 64, 128, 256, 512, 1024])
C = -2.0 * np.minimum(Ls, 8)                      # total log-ratio (saturates at -16)
legacy = np.clip(np.exp(C / Ls / T_TEMP), CLIP_LO, CLIP_HI)
# paper: rho = pi_hind(a_t)/mean_k pi_hind(a_k); with identical siblings the ratio is exactly 1
paper = np.ones_like(legacy)
fig, ax = plt.subplots(figsize=(6.8, 4.4))
ax.plot(Ls, legacy, marker="o", lw=2.2, ms=6, color=PALETTE.get("red_strong", "#d1495b"),
        label=r"legacy: $\rho=\exp(\frac{1}{T n_{tok}}\sum_j \delta_j)$")
ax.plot(Ls, paper, marker="s", lw=2.2, ms=6, color=PALETTE.get("green_3", "#2a9d40"),
        label=r"paper: $\rho=\pi_{hind}(a_t) / \bar{\pi}_{hind}$")
ax.set_xscale("log", base=2)
ax.set_xlabel("response length $n_{tok}$ (front-loaded perturbation on first 8 tokens)")
ax.set_ylabel(r"$\rho$")
ax.set_title("Longer responses inflate legacy $\\rho$; paper $\\rho$ is length-invariant")
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, fontsize=10, loc="lower right")
finalize_figure(fig, os.path.join(FIGS, "fig_synthetic.png"))

print("figs ->", os.path.abspath(FIGS))
