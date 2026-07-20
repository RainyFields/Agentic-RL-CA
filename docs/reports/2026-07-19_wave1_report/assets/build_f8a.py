#!/usr/bin/env python3
"""Build F8a — the credit-alignment diagnostic figure (RQ2's quantitative teeth / the
lambda-gate negative). Reuses the EXACT pairing the gate used (credit_assignment.lambda_gate
+ diagnostic.build_pairs) so the scatter and the reported Spearman/CI are the same numbers
that live in outputs/diag/lambda_gate.json.

  (a) scatter: critic one-step delta  A_t = V_phi(s_{t+1})-V_phi(s_t)  (assigned turn credit)
      vs MC continuation delta  dVhat_t = Vhat(s_{t+1})-Vhat(s_t)  (the martingale-increment
      "ideal" credit), pooled turn-pairs, best-populated checkpoint (b0_s0_step500).
  (b) pooled Spearman with bootstrap 95% CI across the 4 diag checkpoints vs the pre-registered
      unlock threshold (rho>0.2, CI excluding 0). All fail -> lambda sweep DROPPED.

Rerun:  python3 build_f8a.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.expanduser("~/.claude/skills/scientific-figure-making/assets"))

from credit_assignment.diagnostic import build_pairs  # noqa: E402
from credit_assignment.lambda_gate import load_label  # noqa: E402
from figstyle import PALETTE, apply_publication_style, finalize_figure  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

FIGS = os.path.join(HERE, "figs")
DIAG = os.path.join(REPO, "outputs", "diag")
GATE = json.load(open(os.path.join(DIAG, "lambda_gate.json")))
apply_publication_style(font_size=13, axes_linewidth=1.8)

# checkpoints in temporal/label order; the gate set is the 3 b0_s1 steps (b0_s0_step500 extra)
LABELS = ["b0_s1_step100", "b0_s1_step200", "b0_s1_step500", "b0_s0_step500"]
SCATTER = "b0_s0_step500"   # most turn-pairs (n=495)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

# ---------- (a) scatter for one checkpoint ----------
vhat, critic_delta = load_label(DIAG, SCATTER)
pairs = build_pairs(vhat, critic_delta)
a = np.array([p["assigned_advantage"] for p in pairs])   # critic delta (y)
d = np.array([p["delta_v"] for p in pairs])               # MC dVhat (x)
st = GATE["per_label"][SCATTER]
ax = axes[0]
# jitter x a touch (dVhat is discrete-ish: multiples of 1/K) for visibility
rng = np.random.RandomState(0)
xj = d + rng.uniform(-0.015, 0.015, size=len(d))
ax.scatter(xj, a, s=14, color=PALETTE["violet"], alpha=0.45, edgecolors="none")
# reference lines + LS fit
ax.axhline(0, color="#888", lw=1.0, ls="--"); ax.axvline(0, color="#888", lw=1.0, ls="--")
if len(d) > 2:
    b1, b0 = np.polyfit(d, a, 1)
    xr = np.linspace(d.min(), d.max(), 50)
    ax.plot(xr, b1 * xr + b0, color=PALETTE["red_strong"], lw=2.2)
ax.set_xlabel(r"MC continuation delta  $\Delta\hat{V}_t = \hat{V}(s_{t+1})-\hat{V}(s_t)$")
ax.set_ylabel(r"critic one-step delta  $A_t = V_\phi(s_{t+1})-V_\phi(s_t)$")
ax.set_title(f"(a) Assigned vs ideal turn credit — {SCATTER}\n"
             fr"pooled $\rho$={st['pooled_spearman']:.02f}  "
             fr"CI95=[{st['pooled_spearman_ci95'][0]:.02f},{st['pooled_spearman_ci95'][1]:.02f}]  "
             f"n={st['n_pairs']}", fontsize=11)

# ---------- (b) Spearman + CI across checkpoints ----------
ax = axes[1]
ys = np.arange(len(LABELS))[::-1]
for lab, y in zip(LABELS, ys):
    s = GATE["per_label"][lab]
    rho = s["pooled_spearman"]; lo, hi = s["pooled_spearman_ci95"]
    is_gate = lab in GATE["gate_labels"]
    c = PALETTE["neutral"] if not is_gate else PALETTE["blue_main"]
    ax.plot([lo, hi], [y, y], color=c, lw=3, solid_capstyle="round")
    ax.plot(rho, y, "o", color=c, markersize=9, markeredgecolor="black", markeredgewidth=1.0)
    ax.text(hi + 0.02, y, f"n={s['n_pairs']}", va="center", fontsize=9, color="#555")
ax.axvline(0, color="#888", lw=1.2, ls="--")
ax.axvline(0.2, color=PALETTE["red_strong"], lw=1.8, ls="-")
ax.text(0.235, 1.5, "unlock threshold\n$\\rho>0.2$ &\nCI excludes 0", color=PALETTE["red_strong"],
        fontsize=9, va="center", ha="left")
ax.set_yticks(ys)
ax.set_yticklabels([lab.replace("b0_", "").replace("_", " ") + ("  (gate)" if lab in GATE["gate_labels"] else "")
                    for lab in LABELS], fontsize=10)
ax.set_xlabel(r"pooled Spearman $\rho$(A$_t$, $\Delta\hat{V}_t$)  with bootstrap 95% CI")
ax.set_xlim(-0.25, 0.45)
verdict = "UNLOCKED" if GATE["unlocked"] else "DROPPED"
ax.set_title(f"(b) Credit alignment $\\approx$ 0 at every checkpoint\n"
             f"0/{len(GATE['gate_labels'])} gate ckpts pass $\\Rightarrow$ $\\lambda$-sweep {verdict}", fontsize=11)
finalize_figure(fig, os.path.join(FIGS, "fig_f8a_credit_alignment.png"))
print("F8a figure written; scatter n_pairs =", len(pairs), "verdict =", verdict)

# ============ Fig: matched assigned-advantage alignment (GiGPO vs critic) ============
# Apples-to-apples RQ2 readout: each method's ASSIGNED per-turn advantage A_t vs the ideal
# dV-hat_t, same terminal-appended pairing (method_align / critic_align_matched). Contrasted
# with the critic VALUE-DELTA (the lambda-gate quantity) to separate two questions:
#   value-delta ~0  -> the critic can't be bootstrapped through (lambda-sweep dropped);
#   advantage  ~0.2 -> but the assigned advantage is weakly outcome-aligned for BOTH methods,
#                      and GiGPO is NOT better than the critic -> GiGPO's EM win is variance/
#                      stability, not superior per-turn credit.
GA = json.load(open(os.path.join(REPO, "outputs", "diag_methods", "gigpo_align.json")))["gigpo_1p7b_s0_step500"]
CM = json.load(open(os.path.join(DIAG, "critic_align_matched.json")))
ROWS = [  # (label, colorkey, stats, is_delta)
    ("critic value-$\\Delta$  $V(s_{t+1}){-}V(s_t)$\n(s1@500, the $\\lambda$-gate quantity)", "neutral", GATE["per_label"]["b0_s1_step500"], True),
    ("GiGPO advantage $A_t$  (s0@500)", "teal", GA, False),
    ("critic advantage $R{-}V(s_t)$  (s0@500)", "blue_main", CM["b0_s0_step500"], False),
    ("critic advantage $R{-}V(s_t)$  (s1@500)", "blue_main", CM["b0_s1_step500"], False),
]
fig, ax = plt.subplots(figsize=(9.6, 3.9))
ys = list(range(len(ROWS)))[::-1]
ax.axvspan(0.5, 1.0, color="#e8f3e8", zorder=0)
ax.text(0.61, 1.5, "strong\nalignment", color="#4a7a4a", fontsize=8.5, va="center", ha="center")
for (lab, ck, s, is_delta), y in zip(ROWS, ys):
    rho = s["pooled_spearman"]; lo, hi = s["pooled_spearman_ci95"]
    c = PALETTE[ck]; mk = "s" if is_delta else "o"
    ax.plot([lo, hi], [y, y], color=c, lw=3, solid_capstyle="round", alpha=0.9)
    ax.plot(rho, y, mk, color=c, markersize=10, markeredgecolor="black", markeredgewidth=1.1)
    ax.text(hi + 0.012, y, f"$\\rho$={rho:+.2f}  rank={s['pairwise_ranking_accuracy']:.2f}",
            va="center", fontsize=9, color="#333")
ax.axvline(0, color="#888", lw=1.2, ls="--")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in ROWS], fontsize=9.3)
ax.set_xlim(-0.15, 0.72)
ax.set_xlabel(r"pooled Spearman $\rho$(assigned credit, $\Delta\hat{V}_t$)  with bootstrap 95% CI")
ax.set_title("Matched credit-alignment: the critic's VALUE can't track progress ($\\rho{\\approx}0$),\n"
             "yet both ASSIGNED advantages are only weakly outcome-aligned ($\\rho{\\approx}0.2$) — GiGPO not $>$ critic",
             fontsize=10.5)
finalize_figure(fig, os.path.join(FIGS, "fig_credit_align_matched.png"))
print(f"matched fig: GiGPO rho={GA['pooled_spearman']:+.3f}  critic-s0 rho={CM['b0_s0_step500']['pooled_spearman']:+.3f}")
