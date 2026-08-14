#!/usr/bin/env python3
"""Per-environment headline figures: training curve (left) + final-eval table (right),
one figure per environment, house style (scientific-figure-making figstyle).

Reads results/*.csv written by collect_results.py; methods missing from the curves file
(still training) are simply absent from the panel, so this can be rerun incrementally.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~/.claude/skills/scientific-figure-making/assets"))
from figstyle import apply_publication_style, PALETTE, make_lines, finalize_figure

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIGS = os.path.join(HERE, "..", "figs")

ORDER = ["token_ppo", "turn_ppo", "grpo", "gigpo", "hcapo", "hcapo_ans"]
PRETTY = {
    "token_ppo": "token-PPO", "turn_ppo": "turn-PPO", "grpo": "GRPO",
    "gigpo": "GiGPO", "hcapo": "HCAPO", "hcapo_ans": "HCAPO (adapted)",
    "floor": "pre-RL init",
}
COLORS = {
    "token_ppo": PALETTE["blue_main"], "turn_ppo": PALETTE["teal"],
    "grpo": PALETTE["green_3"], "gigpo": PALETTE["red_strong"],
    "hcapo": PALETTE["violet"], "hcapo_ans": PALETTE["neutral"],
}


def _table(ax, rows, headers):
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(13)
    tab.scale(1.0, 1.7)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
    # bold the best non-floor value (column 1 holds the headline metric)
    vals = [float(x[1]) for x in rows if x[0] != PRETTY["floor"]]
    if vals:
        best = max(vals)
        for i, x in enumerate(rows):
            if x[0] != PRETTY["floor"] and float(x[1]) == best:
                tab[i + 1, 1].set_text_props(weight="bold")
    return tab


def build_env_figure(env, curve_csv, eval_csv, ycol, ylab, eval_col, eval_head, title, out):
    curves = pd.read_csv(os.path.join(RES, curve_csv))
    evals = pd.read_csv(os.path.join(RES, eval_csv))
    fig, (axc, axt) = plt.subplots(
        1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.45, 1.0]})
    series = []
    for m in ORDER:
        sub = curves[curves.method == m]
        if not len(sub):
            continue
        series.append({"x": sub.step.values, "y": sub[ycol].values,
                       "label": PRETTY[m], "color": COLORS[m]})
    make_lines(axc, series, ylabel=ylab, xlabel="training step", smooth=False)
    axc.legend(frameon=False, fontsize=12, ncol=2, loc="lower right")
    axc.set_title(f"{title} — training", fontsize=15)

    rows = []
    for m in ORDER + ["floor"]:
        sub = evals[evals.method == m]
        if not len(sub):
            continue
        r = sub.iloc[0]
        step = int(r["eval_step"]) if "eval_step" in r and not pd.isna(r.get("eval_step", np.nan)) else ""
        rows.append([PRETTY[m], f"{r[eval_col]:.3f}", step])
    _table(axt, rows, ["method", eval_head, "eval step"])
    axt.set_title(f"{title} — final eval", fontsize=15)
    finalize_figure(fig, os.path.join(FIGS, out))
    print("wrote", out, f"({len(series)} curves, {len(rows)} table rows)")


def searchqa_eval_table():
    """Full-set per-task table -> per-method macro EM rows for the headline figure."""
    p = os.path.join(RES, "searchqa_eval.csv")
    t = pd.read_csv(p)
    # paper_table.csv carries its own summary rows; use the canonical 'macro' row
    agg = (t[t.dataset == "macro"][["method", "em"]]
           .rename(columns={"em": "macro_em"}).reset_index(drop=True))
    agg["eval_step"] = 500
    agg.loc[agg.method == "floor", "eval_step"] = 0
    agg.to_csv(os.path.join(RES, "searchqa_eval_headline.csv"), index=False)
    return agg


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    apply_publication_style(font_size=14)
    searchqa_eval_table()
    build_env_figure(
        "searchqa", "searchqa_curves.csv", "searchqa_eval_headline.csv",
        "val_macro_em", "val macro EM (2,048 q, greedy)", "macro_em",
        "full-set macro EM", "SearchQA (Qwen3-4B, 4-turn, non-thinking)",
        "fig1_searchqa")
    build_env_figure(
        "alfworld", "alfworld_curves.csv", "alfworld_eval.csv",
        "val_success", "val success rate (seen split)", "unseen_success",
        "unseen success", "ALFWorld (Qwen3-1.7B, ReAct transcript)",
        "fig2_alfworld")
