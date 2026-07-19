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
