#!/usr/bin/env python3
"""Run the ENTIRE analysis suite of the cross-environment report (2026-07-21_4b_horizon_eval)
over the HCAPO arms alongside GRPO / turn-PPO / floor — by reusing that report's own modules
(horizon_common, horizon_figures, alfworld_common, alfworld_figures, make_remaining_figs) with
extended method maps, and writing all outputs into THIS report's figs/ and results/.

SearchQA methods (all non-thinking@2048, protocol-matched; GRPO is the Δ-reference):
  floor, grpo(nt), turn_ppo(nt), hcapo_paper, hcapo_ans, hcapo_lift, hcapo_lift_ans
AlfWorld methods (1.7B, unseen 134):
  grpo, gigpo, turn_ppo, token_ppo, hcapo_buggy, hcapo_paper
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
PREV = os.path.abspath(os.path.join(HERE, "..", "..", "2026-07-21_4b_horizon_eval", "assets"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "analysis", "figures"))
sys.path.insert(0, PREV)
FIGS = os.path.abspath(os.path.join(HERE, "..", "figs"))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
os.makedirs(FIGS, exist_ok=True)
os.makedirs(RES, exist_ok=True)

from figstyle import PALETTE  # noqa: E402
import horizon_common as H  # noqa: E402
import horizon_figures as HF  # noqa: E402
import alfworld_common as A  # noqa: E402
import alfworld_figures as AF  # noqa: E402
import make_remaining_figs as MR  # noqa: E402

BLUE, TEAL = PALETTE["blue_main"], PALETTE.get("teal", "#2a9d8f")
RED, ORANGE = PALETTE.get("red_strong", "#d1495b"), PALETTE.get("highlight", "#e9a13b")
GREEN, VIOLET = PALETTE.get("green_3", "#2a9d40"), PALETTE.get("violet", "#7b5cff")
MAUVE, GRAY = "#b07aa1", "#8a8f98"

# ---------------- SearchQA method maps (non-thinking, protocol-matched) ----------------
SQA_LABELS = {
    "floor":          "eval4b_floor",
    "grpo":           "eval4b_nt_token_grpo_s0",
    "turn_ppo":       "eval4b_nt_turn_ppo_b0_s0",
    "hcapo_paper":    "eval4b_hcapo_paper_s0",
    "hcapo_ans":      "eval4b_hcapo_ans_s0",
    "hcapo_lift":     "eval4b_hcapo_lift_s0",
    "hcapo_lift_ans": "eval4b_hcapo_lift_ans_s0",
}
ORDER = ["floor", "grpo", "turn_ppo", "hcapo_paper", "hcapo_ans", "hcapo_lift", "hcapo_lift_ans"]
PRETTY = {"floor": "pre-RL floor", "grpo": "GRPO", "turn_ppo": "turn-PPO",
          "hcapo_paper": "HCAPO hind+obs", "hcapo_ans": "HCAPO hind+ans",
          "hcapo_lift": "HCAPO lift+obs", "hcapo_lift_ans": "HCAPO lift+ans"}
COLOR = {"floor": GRAY, "grpo": BLUE, "turn_ppo": VIOLET, "hcapo_paper": RED,
         "hcapo_ans": TEAL, "hcapo_lift": GREEN, "hcapo_lift_ans": MAUVE}

# patch horizon_common / horizon_figures
H.METHOD_ORDER = ORDER
H.METHOD_PRETTY = PRETTY
H.REF_METHOD = "grpo"
HF.MCOLOR = COLOR
HF.FIGS, HF.RES = FIGS, RES

data = H.load_all(SQA_LABELS)
print("SearchQA methods loaded:", list(data))

hl = HF.headline_table(data)
print(hl.round(4).to_string(index=False))
a1 = HF.a1_score_vs_hops(data)
a2 = HF.a2_horizon_gain(data)
if a2 is not None:
    print(a2.round(3).to_string(index=False))
HF.a3_spend_vs_need(data)
HF.a4_turn_violins(data)
HF.a5_secondary(data)
v = HF.verdicts(data, a1, a2, hl)
print(v)

# per-stratum cap-fail table (A3 companion)
import pandas as pd  # noqa: E402
rows = []
for m in ORDER:
    if m not in data:
        continue
    g = data[m].groupby("H_gold").agg(EM=("em", "mean"), med_turns=("turns", "median"),
                                      med_search=("n_search_calls", "median"),
                                      cap_fail=("cap_fail", "mean"))
    for hgold, r in g.iterrows():
        rows.append(dict(method=m, H_gold=int(hgold), **{k: float(r[k]) for k in g.columns}))
pd.DataFrame(rows).to_csv(os.path.join(RES, "a3_capfail_by_stratum.csv"), index=False)

# ---------------- AlfWorld: extend to both HCAPO legs ----------------
A.ALF.update({
    "hcapo_buggy": ("unseen_cmp_hcapo",       "history_HCAPO_51q4rbh3.csv"),
    "hcapo_paper": ("unseen_cmp_hcapo_paper", "history_HCAPOpaper_local.csv"),
})
A.ALF_EVAL.update({"hcapo_buggy": 0.791, "hcapo_paper": 0.784})
A.ALF_EVAL_STEP.update({"hcapo_buggy": 145, "hcapo_paper": 145})
A.ALF_METHOD_ORDER = ["grpo", "gigpo", "turn_ppo", "token_ppo", "hcapo_buggy", "hcapo_paper"]
A.ALF_PRETTY.update({"hcapo_buggy": "HCAPO (buggy)", "hcapo_paper": "HCAPO (paper)"})
AF.MCOLOR = {"grpo": BLUE, "gigpo": TEAL, "turn_ppo": VIOLET, "token_ppo": "#9b8bb4",
             "hcapo_buggy": ORANGE, "hcapo_paper": RED}
AF.FIGS, AF.RES = FIGS, RES

ep = A.load_alfworld_episodes()
tr = A.load_alfworld_training()


def _fig_turns_bar_rotated(ep):
    """6 methods overcrowd AF.fig_turns_bar's horizontal labels — rebuild with rotation."""
    from matplotlib import pyplot as plt
    from figstyle import finalize_figure
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ms = A.ALF_METHOD_ORDER
    means = [ep[ep.method == m].turns.mean() for m in ms]
    sds = [ep[ep.method == m].turns.std() for m in ms]
    ax.bar(range(len(ms)), means, yerr=sds, capsize=4,
           color=[AF.MCOLOR[m] for m in ms], alpha=0.88)
    for i, v in enumerate(means):
        ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([A.ALF_PRETTY[m] for m in ms], rotation=18, ha="right")
    ax.set_ylabel("turns used per episode (mean$\\pm$sd)")
    ax.set_title("AlfWorld — trajectory length (unseen 134)")
    ax.grid(True, axis="y", alpha=0.3)
    finalize_figure(fig, os.path.join(FIGS, "fig2_alfworld_turns.png"))
print("\nAlfWorld eval table (unseen 134):")
for m in A.ALF_METHOD_ORDER:
    d = ep[ep.method == m]
    print(f"  {A.ALF_PRETTY[m]:14s} success={d.success.mean():.3f} mean_turns={d.turns.mean():.1f}")
ep.to_csv(os.path.join(RES, "alfworld_episodes_extended.csv"), index=False)
AF.fig_training(tr)
_fig_turns_bar_rotated(ep)
AF.fig_acc_vs_turns(ep)

# per-task-type success table (the AlfWorld stratification)
ct = pd.crosstab(ep.task_type, ep.method, values=ep.success, aggfunc="mean").reindex(A.TASK_ORDER)
ct.to_csv(os.path.join(RES, "alfworld_success_by_tasktype.csv"))
print("\nAlfWorld success x task_type:")
print(ct.round(3).to_string())

# ---------------- SearchQA turn-behaviour figures (box / turns-vs-gold / acc-vs-turns) --------
MR.UCOLOR = dict(COLOR)
MR.UPRETTY = dict(PRETTY)
MR.UORDER = ORDER
MR.SQA_CANON = {k: k for k in SQA_LABELS}
MR.SQA_LABELS = SQA_LABELS
MR.FIGS, MR.RES = FIGS, RES
sqa = {m: d for m, d in data.items()}
MR.fig3_turns_box(sqa)
MR.fig4_turns_vs_gold(sqa)

# acc-vs-turns, both envs, extended methods (adapted from MR.fig5 to the patched AlfWorld set)
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.9))
bins, labels = [0, 6, 8, 12, 20, 100], ["<=6", "7-8", "9-12", "13-20", ">20"]
ep2 = ep.copy()
ep2["bin"] = pd.cut(ep2["turns"], bins=bins, labels=labels)
for m in A.ALF_METHOD_ORDER:
    d = ep2[ep2.method == m]
    g = d.groupby("bin", observed=False)["success"].agg(["mean", "size"])
    xs = [i for i, lab in enumerate(labels) if g.loc[lab, "size"] >= 3]
    axL.plot(xs, [g.loc[labels[i], "mean"] for i in xs], marker="o", lw=2.0, ms=6,
             color=AF.MCOLOR[m], label=A.ALF_PRETTY[m])
axL.set_xticks(range(len(labels))); axL.set_xticklabels(labels)
axL.set_xlabel("turns used"); axL.set_ylabel("success rate"); axL.set_title("AlfWorld (1.7B)")
axL.grid(True, alpha=0.3); axL.legend(frameon=False, fontsize=8.5)
for m in ORDER:
    if m not in sqa:
        continue
    d = sqa[m]
    g = d.groupby("turns")["em"].agg(["mean", "size"])
    xs = [t for t in [1, 2, 3, 4] if t in g.index and g.loc[t, "size"] >= 20]
    axR.plot(xs, [g.loc[t, "mean"] for t in xs], marker="o", lw=2.0, ms=6,
             color=COLOR[m], label=PRETTY[m], ls="--" if m == "floor" else "-")
axR.set_xticks([1, 2, 3, 4]); axR.set_xlabel("turns used"); axR.set_ylabel("EM")
axR.set_title("SearchQA (4B)"); axR.grid(True, alpha=0.3); axR.legend(frameon=False, fontsize=8.5)
fig.suptitle("Accuracy vs turns used by the model", y=1.02)
from figstyle import finalize_figure  # noqa: E402
finalize_figure(fig, os.path.join(FIGS, "fig_acc_vs_turns_bothenv.png"))

# ---------------- training curves, both envs ----------------
import re  # noqa: E402
LOGS = os.path.expanduser("~/xiaoxuan/worker_logs/launches")
SQA_TRAIN = {
    "grpo": "20260726_185609_arlca-token-grpo-4b-nt-s0.log",
    "turn_ppo": "20260726_185615_arlca-turn-ppo-b0-4b-nt-s0.log",
    "hcapo_paper": "20260729_104509_arlca-hcapo-paper-4b-nt-s0.log",
    "hcapo_ans": "20260730_154925_arlca-hcapo-ans-4b-nt-s0.log",
    "hcapo_lift": "20260731_113201_arlca-hcapo-lift-4b-nt-s0.log",
    "hcapo_lift_ans": "20260731_113207_arlca-hcapo-lift-ans-4b-nt-s0.log",
}
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.9))
for m in A.ALF_METHOD_ORDER:
    d = tr[m]
    axL.plot(d.step, d.success_rate, lw=1.9, ms=2.5, marker="o", color=AF.MCOLOR[m],
             label=A.ALF_PRETTY[m])
axL.set_xlabel("training step"); axL.set_ylabel("seen success rate")
axL.set_title("AlfWorld (1.7B) — training"); axL.grid(True, alpha=0.3)
axL.legend(frameon=False, fontsize=8.5)
for m, f in SQA_TRAIN.items():
    t = open(os.path.join(LOGS, f), errors="ignore").read().replace("\r", "\n")
    pairs = {int(a): float(b) for a, b in re.findall(r"step:(\d+) .*?val-core/macro_em:([0-9.]+)", t)}
    ks = sorted(pairs)
    axR.plot(ks, [pairs[k] for k in ks], lw=1.9, ms=2.5, marker="o", color=COLOR[m], label=PRETTY[m])
axR.set_xlabel("training step"); axR.set_ylabel("val-2048 macro-EM")
axR.set_title("SearchQA (4B, non-thinking) — training"); axR.grid(True, alpha=0.3)
axR.legend(frameon=False, fontsize=8.5)
fig.suptitle("Training accuracy per environment", y=1.02)
finalize_figure(fig, os.path.join(FIGS, "fig_training_curves.png"))

print("\nALL OUTPUTS ->", FIGS, "and", RES)
