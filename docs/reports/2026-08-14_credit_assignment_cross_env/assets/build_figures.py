#!/usr/bin/env python3
"""Per-environment headline artifacts: a curve-only training figure (figs/) and a
separate LaTeX eval table (tables/, best bold, second-best underlined), one pair per
environment, house style (scientific-figure-making figstyle).

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
TABLES = os.path.join(HERE, "..", "tables")

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


def write_tex_table(rows, headers, out):
    """rows: [pretty_method, value(float), step]. Best non-floor value bold, second
    best underlined (paper convention)."""
    ranked = sorted({v for m, v, _ in rows if m != PRETTY["floor"]}, reverse=True)

    def fmt(m, v):
        s = f"{v:.3f}"
        if m == PRETTY["floor"] or not ranked:
            return s
        if v == ranked[0]:
            return rf"\textbf{{{s}}}"
        if len(ranked) > 1 and v == ranked[1]:
            return rf"\underline{{{s}}}"
        return s

    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    for m, v, step in rows:
        lines.append(f"{m} & {fmt(m, v)} & {step} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path = os.path.join(TABLES, out)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out, f"({len(rows)} rows)")


def build_env_figure(env, curve_csv, eval_csv, ycol, ylab, eval_col, eval_head, title,
                     out, table_out, xmax=None):
    curves = pd.read_csv(os.path.join(RES, curve_csv))
    evals = pd.read_csv(os.path.join(RES, eval_csv))
    fig, axc = plt.subplots(figsize=(7.6, 5.0))
    series = []
    for m in ORDER:
        sub = curves[curves.method == m]
        if xmax is not None:
            sub = sub[sub.step <= xmax]
        if not len(sub):
            continue
        series.append({"x": sub.step.values, "y": sub[ycol].values,
                       "label": PRETTY[m], "color": COLORS[m]})
    make_lines(axc, series, ylabel=ylab, xlabel="training step", smooth=False)
    axc.legend(frameon=False, fontsize=12, ncol=2, loc="lower right")
    axc.set_title(title, fontsize=15)
    finalize_figure(fig, os.path.join(FIGS, out))

    rows = []
    for m in ORDER + ["floor"]:
        sub = evals[evals.method == m]
        if not len(sub):
            continue
        r = sub.iloc[0]
        step = int(r["eval_step"]) if "eval_step" in r and not pd.isna(r.get("eval_step", np.nan)) else ""
        rows.append([PRETTY[m], float(r[eval_col]), step])
    write_tex_table(rows, ["method", eval_head, "eval step"], table_out)
    print("wrote", out, f"({len(series)} curves)")


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
    os.makedirs(TABLES, exist_ok=True)
    apply_publication_style(font_size=14)
    searchqa_eval_table()
    build_env_figure(
        "searchqa", "searchqa_curves.csv", "searchqa_eval_headline.csv",
        "val_macro_em", "val macro EM (2,048 q, greedy)", "macro_em",
        "macro EM", "SearchQA (Qwen3-4B, 4-turn, non-thinking)",
        "fig1_searchqa", "tab_searchqa.tex")
    build_env_figure(
        "alfworld", "alfworld_curves.csv", "alfworld_eval.csv",
        "val_success", "val success rate (seen split)", "unseen_success",
        "unseen success", "ALFWorld (Qwen3-1.7B, ReAct transcript)",
        "fig2_alfworld", "tab_alfworld.tex", xmax=200)
