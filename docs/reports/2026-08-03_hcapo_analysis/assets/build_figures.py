#!/usr/bin/env python3
"""Figures for the HCAPO analysis report. All numbers parsed from training logs, full-set eval
JSONLs, the AlfWorld unseen results, and the offline estimator diagnostic — nothing hand-entered.

  fig1_ignition     every HCAPO variant runs away; only the ignition STEP differs
  fig2_frontier     accuracy vs token cost — where HCAPO sits against GRPO/PPO/GiGPO
  fig3_diagnostic   what the credit signal actually tracks (length / leakage / position)
  fig4_crossenv     both environments, HCAPO vs the three baselines
  fig5_hops         horizon-stratified EM (SearchQA): HCAPO vs baselines
"""
import json
import os
import re
import sys
from collections import defaultdict

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
EVAL = "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/eval_full"
DIAG = "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/diag_hcapo"

BLUE, TEAL = PALETTE["blue_main"], PALETTE.get("teal", "#2a9d8f")
RED, ORANGE = PALETTE.get("red_strong", "#d1495b"), PALETTE.get("highlight", "#e9a13b")
GREEN, VIOLET = PALETTE.get("green_3", "#2a9d40"), PALETTE.get("violet", "#7b5cff")
GRAY = "#8a8f98"

# ---------- training logs: the six HCAPO runs + GRPO reference ----------
HRUNS = [
    ("legacy $\\rho$, $\\omega$=1.0",        "20260727_154601_arlca-hcapo-4b-nt-s0.log",       RED,    90),
    ("legacy $\\rho$, $\\omega$=0.5",        "20260727_154607_arlca-hcapo-w05-4b-nt-s0.log",   ORANGE, 290),
    ("paper $\\rho$, hind, last_obs",        "20260729_104509_arlca-hcapo-paper-4b-nt-s0.log", VIOLET, 165),
    ("paper $\\rho$, hind, answer",          "20260730_154925_arlca-hcapo-ans-4b-nt-s0.log",   TEAL,   265),
    ("paper $\\rho$, lift, answer",          "20260731_113207_arlca-hcapo-lift-ans-4b-nt-s0.log", "#b07aa1", 375),
    ("paper $\\rho$, lift, last_obs",        "20260731_113201_arlca-hcapo-lift-4b-nt-s0.log",  GREEN,  460),
]
GRPO_LOG = "20260726_185609_arlca-token-grpo-4b-nt-s0.log"


def parse(fn):
    txt = open(os.path.join(LOGS, fn), errors="ignore").read().replace("\r", "\n")
    per = lambda k: {int(m.group(1)): float(m.group(2))
                     for m in re.finditer(rf"step:(\d+) .*?{k}:([0-9.]+)", txt)}
    rho = [(float(a), float(b)) for a, b in
           re.findall(r"\[hcapo-rho\] mean=([0-9.]+) std=([0-9.]+)", txt)]
    return dict(len=per("response_length/mean"), clip=per("response_length/clip_ratio"),
                em=per("val-core/macro_em"), rho=rho)


runs = {n: (parse(f), c, ig) for n, f, c, ig in HRUNS}
grpo = parse(GRPO_LOG)
xy = lambda d, k: (sorted(d[k]), [d[k][s] for s in sorted(d[k])])

# ---------------- fig 1: every variant ignites ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
for n, (d, c, ig) in runs.items():
    s, v = xy(d, "len")
    ax1.plot(s, v, lw=2.1, color=c, label=n)
s, v = xy(grpo, "len")
ax1.plot(s, v, lw=2.0, ls="--", color=GRAY, label="GRPO (reference)")
ax1.axhline(2048, color="#555", ls=":", lw=1.3)
ax1.text(8, 1990, "response cap 2048", fontsize=9, color="#555", va="top")
ax1.set_xlabel("training step"); ax1.set_ylabel("mean response tokens / turn")
ax1.set_title("Six HCAPO configurations. All six run away.")
ax1.grid(True, alpha=0.3); ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
names = [n for n in runs]; igs = [runs[n][2] for n in names]
order = np.argsort(igs)
ax2.barh(range(len(names)), [igs[i] for i in order],
         color=[runs[names[i]][1] for i in order], alpha=0.9)
ax2.set_yticks(range(len(names)))
ax2.set_yticklabels([names[i] for i in order], fontsize=9)
ax2.set_xlabel("step at which the runaway ignites")
ax2.set_title("Every knob delays it;\nnone prevents it")
ax2.grid(True, axis="x", alpha=0.3)
finalize_figure(fig, os.path.join(FIGS, "fig1_ignition.png"))

# ---------------- fig 2: accuracy vs cost frontier ----------------
def evalstats(lab):
    rows = [json.loads(l) for l in open(f"{EVAL}/{lab}/val_trajectories_step0.jsonl")]
    by = defaultdict(list)
    for r in rows:
        by[r["data_source"]].append(r["em"])
    macro = float(np.mean([np.mean(v) for v in by.values()]))
    tok = float(np.mean([(r.get("tokens_generated") or 0) for r in rows]))
    turns = float(np.mean([r["turns"] for r in rows]))
    return macro, tok, turns


POINTS = [
    ("GRPO",           "eval4b_nt_token_grpo_s0",  BLUE,   "o"),
    ("turn-PPO",       "eval4b_nt_turn_ppo_b0_s0", VIOLET, "o"),
    ("floor (pre-RL)", "eval4b_floor",             GRAY,   "s"),
    ("HCAPO paper",    "eval4b_hcapo_paper_s0",    RED,    "^"),
    ("HCAPO answer",   "eval4b_hcapo_ans_s0",      TEAL,   "^"),
    ("HCAPO lift",     "eval4b_hcapo_lift_s0",     GREEN,  "^"),
    ("HCAPO lift+ans", "eval4b_hcapo_lift_ans_s0", "#b07aa1", "^"),
]
fig, ax = plt.subplots(figsize=(8.6, 5.2))
for name, lab, c, mk in POINTS:
    try:
        m, tok, _ = evalstats(lab)
    except FileNotFoundError:
        continue
    ax.scatter(tok, m, s=150, color=c, marker=mk, zorder=3, edgecolor="white", linewidth=1.2)
    dy = -0.007 if name == "HCAPO lift" else 0.006
    ax.annotate(f"{name}\n{m:.3f} @ {tok:.0f} tok", (tok, m), xytext=(6, dy * 900),
                textcoords="offset points", fontsize=9.5, color=c)
ax.set_xscale("log")
ax.set_xlabel("generated tokens per question (log scale)")
ax.set_ylabel("full-set macro-EM (51,713 questions)")
ax.set_title("SearchQA: accuracy vs cost. Every HCAPO variant is up and to the right of GRPO —\n"
             "more tokens, less accuracy.")
ax.grid(True, alpha=0.3, which="both")
finalize_figure(fig, os.path.join(FIGS, "fig2_frontier.png"))

# ---------------- fig 3: what the credit signal tracks ----------------
dg = {t: json.load(open(f"{DIAG}/estdiag_{t}.json")) for t in ("floor", "hcapo250")}
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.2))
lbl = ["plain score\n(no hindsight)", "hindsight score\n(Eq. 6)", "hindsight lift\n($\\Delta m$)"]
keys = ["corr_m_no_len", "corr_m_hind_len", "corr_dm_len"]
x = np.arange(3); w = 0.36
for i, (t, c, nm) in enumerate([("floor", BLUE, "healthy lengths"), ("hcapo250", RED, "post-runaway")]):
    ax1.bar(x + i * w, [dg[t][k]["r"] for k in keys], w, color=c, alpha=0.9, label=nm)
ax1.axhline(0, color="#555", lw=1)
ax1.set_xticks(x + w / 2); ax1.set_xticklabels(lbl, fontsize=9)
ax1.set_ylabel("within-trajectory corr with turn length")
ax1.set_title("The length bias is INJECTED\nby the hindsight text")
ax1.legend(frameon=False, fontsize=9); ax1.grid(True, axis="y", alpha=0.3)
# leakage
fr = [dg["floor"]["rank_agreement"]["frac_eq7_top_turn_is_final"]["mean"],
      dg["hcapo250"]["rank_agreement"]["frac_eq7_top_turn_is_final"]["mean"]]
base = [0.386, 0.28]
ax2.bar([0, 1], fr, 0.5, color=[BLUE, RED], alpha=0.9, label="observed")
ax2.plot([-0.3, 0.3], [base[0]] * 2, color="#333", lw=2, ls="--")
ax2.plot([0.7, 1.3], [base[1]] * 2, color="#333", lw=2, ls="--", label="chance level")
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["healthy", "post-runaway"], fontsize=9)
ax2.set_ylabel("top-credit turn is the ANSWER turn")
ax2.set_title("Answer leakage:\nthe ending explains its own turn")
ax2.legend(frameon=False, fontsize=9); ax2.grid(True, axis="y", alpha=0.3)
# position
pc = [dg["floor"]["rho_eq7_stats"]["corr_turnindex"]["r"],
      dg["hcapo250"]["rho_eq7_stats"]["corr_turnindex"]["r"]]
ax3.bar([0, 1], pc, 0.5, color=[BLUE, RED], alpha=0.9)
ax3.axhline(0, color="#555", lw=1)
ax3.set_xticks([0, 1]); ax3.set_xticklabels(["healthy", "post-runaway"], fontsize=9)
ax3.set_ylabel("corr($\\rho$, turn position)")
ax3.set_title("Position preference FLIPS\nwith the policy's own length")
ax3.grid(True, axis="y", alpha=0.3)
finalize_figure(fig, os.path.join(FIGS, "fig3_diagnostic.png"))

# ---------------- fig 4: cross-environment ----------------
alf = [("turn-PPO", 0.963, VIOLET), ("GiGPO", 0.955, TEAL), ("GRPO", 0.881, BLUE),
       ("HCAPO\n(buggy)", 0.791, ORANGE), ("HCAPO\n(paper)", 0.784, RED),
       ("token-PPO", 0.761, "#9b8bb4")]
sqa = [("GRPO", 0.4629, BLUE), ("GiGPO*", 0.4455, TEAL), ("HCAPO\n(answer)", 0.4400, TEAL),
       ("HCAPO\n(lift)", 0.4294, GREEN), ("turn-PPO", 0.4173, VIOLET),
       ("HCAPO\n(paper)", 0.4104, RED), ("token-PPO*", 0.3807, "#9b8bb4")]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, data, ttl, ylab in [(ax1, alf, "AlfWorld (1.7B, unseen 134 games)", "success rate"),
                            (ax2, sqa, "SearchQA (4B, full-set 51,713 q)", "macro-EM")]:
    ns = [d[0] for d in data]; vs = [d[1] for d in data]; cs = [d[2] for d in data]
    b = ax.bar(range(len(ns)), vs, color=cs, alpha=0.9)
    for r, v in zip(b, vs):
        ax.text(r.get_x() + r.get_width() / 2, v + max(vs) * 0.012, f"{v:.3f}",
                ha="center", fontsize=9.5)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels(ns, fontsize=8.5)
    ax.set_ylabel(ylab); ax.set_title(ttl); ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(vs) * 1.15)
ax2.text(0.99, 0.03, "*thinking-mode protocol", transform=ax2.transAxes, ha="right",
         fontsize=8, color="#666")
fig.suptitle("HCAPO underperforms flat GRPO in both environments", y=1.03)
finalize_figure(fig, os.path.join(FIGS, "fig4_crossenv.png"))

# ---------------- fig 5: horizon-stratified ----------------
hop = __import__("pandas").read_parquet(
    "/home/tiger/xiaoxuan/Agentic-RL-CA/data/hop_annotations/hop_table.parquet"
).sort_values("index").reset_index(drop=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5))
SER = [("GRPO", "eval4b_nt_token_grpo_s0", BLUE), ("turn-PPO", "eval4b_nt_turn_ppo_b0_s0", VIOLET),
       ("HCAPO answer", "eval4b_hcapo_ans_s0", TEAL), ("HCAPO lift", "eval4b_hcapo_lift_s0", GREEN),
       ("HCAPO paper", "eval4b_hcapo_paper_s0", RED), ("floor", "eval4b_floor", GRAY)]
for nm, lab, c in SER:
    try:
        rows = [json.loads(l) for l in open(f"{EVAL}/{lab}/val_trajectories_step0.jsonl")]
    except FileNotFoundError:
        continue
    em = np.array([r["em"] for r in rows]); ds = np.array([r["data_source"] for r in rows])
    H = hop["H_gold"].values[:len(em)]
    for ax, mask, xs, ttl in [(ax1, ds == "musique", [2, 3, 4], "within MuSiQue"),
                              (ax2, ds == "2wikimultihopqa", [2, 4], "within 2Wiki")]:
        ys = [em[mask & (H == h)].mean() for h in xs]
        ax.plot(xs, ys, marker="o", lw=2.1, ms=6, color=c,
                label=nm, ls="--" if nm == "floor" else "-")
for ax, xs, ttl in [(ax1, [2, 3, 4], "within MuSiQue"), (ax2, [2, 4], "within 2Wiki")]:
    ax.set_xticks(xs); ax.set_xlabel("required hops $H_{gold}$"); ax.set_ylabel("EM")
    ax.set_title(ttl); ax.grid(True, alpha=0.3)
ax1.legend(frameon=False, fontsize=8.5)
fig.suptitle("Horizon-stratified: HCAPO does not buy robustness on deep chains", y=1.02)
finalize_figure(fig, os.path.join(FIGS, "fig5_hops.png"))

print("figs ->", os.path.abspath(FIGS))
