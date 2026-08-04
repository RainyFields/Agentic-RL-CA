"""Paper-style figures (house figstyle) for the ASearcher s75 eval report."""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/.claude/skills/scientific-figure-making/assets"))
import numpy as np  # noqa: E402
from figstyle import apply_publication_style, PALETTE, make_grouped_bar, finalize_figure  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")
os.makedirs(A, exist_ok=True)
SRC = "/home/tiger/xiaoxuan/arlca-8b/outputs/judge"
apply_publication_style(font_size=15)

ARMS = {"GRPO": "grpo_s75", "turn-PPO": "turnppo_s75"}
S = {a: json.load(open(f"{SRC}/{lab}.summary.json")) for a, lab in ARMS.items()}
ORDER = ["NQ_rand1000", "TriviaQA_rand1000", "PopQA_rand1000", "HotpotQA_rand1000",
         "2WikiMultihopQA_rand1000", "Musique_rand1000", "Bamboogle",
         "GAIA", "frames", "xbench-deepsearch"]
SHORT = ["NQ", "TriviaQA", "PopQA", "Hotpot", "2Wiki", "Musique", "Bamb.",
         "GAIA$^{*}$", "frames$^{*}$", "xbench$^{*}$"]
COLORS = [PALETTE.get("blue", "#4477AA"), PALETTE.get("red", "#EE6677")]
colors = list(PALETTE.values())
C2 = [colors[0], colors[1]]

# ---- fig 1: metric ladder (wiki-7 macro under three scorers) ----
fig, ax = plt.subplots(figsize=(7.5, 4.6))
metrics = ["strict EM", "sub-EM", "judge"]
series = []
for arm in ARMS:
    w = S[arm]["wiki_answerable"]
    series.append([w["macro_em"], w["macro_subem"], w["macro_judge"]])
make_grouped_bar(ax, metrics, series, list(ARMS), ylabel="wiki-answerable macro score",
                 colors=C2, annotate=True)
ax.set_ylim(0, 0.62)
ax.legend(frameon=False, loc="upper left")
finalize_figure(fig, os.path.join(A, "fig_metric_ladder.png"))

# ---- fig 2: judge score per benchmark ----
fig, ax = plt.subplots(figsize=(11.5, 4.6))
series = [[S[arm]["by_benchmark"][b]["judge"] for b in ORDER] for arm in ARMS]
make_grouped_bar(ax, SHORT, series, list(ARMS), ylabel="judge score", colors=C2)
ax.axvline(6.5, color="gray", linestyle="--", linewidth=1.5)
ax.text(8.0, 0.72, "live-web\n(corpus gap)", ha="center", fontsize=12, color="gray")
ax.set_ylim(0, 0.85)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.72, 1.0))
finalize_figure(fig, os.path.join(A, "fig_judge_by_benchmark.png"))

# ---- fig 3: avg turns per benchmark ----
fig, ax = plt.subplots(figsize=(11.5, 4.2))
series = [[S[arm]["by_benchmark"][b]["avg_turns"] for b in ORDER] for arm in ARMS]
make_grouped_bar(ax, SHORT, series, list(ARMS), ylabel="avg turns / question", colors=C2)
ax.axvline(6.5, color="gray", linestyle="--", linewidth=1.5)
ax.axhline(32, color="gray", linewidth=1.0, linestyle=":")
ax.set_ylim(0, 21)
ax.legend(frameon=False, loc="upper right")
finalize_figure(fig, os.path.join(A, "fig_turns.png"))

# ---- tables.tex ----
rows_bench = []
for b, sh in zip(ORDER, SHORT):
    g, t = S["GRPO"]["by_benchmark"][b], S["turn-PPO"]["by_benchmark"][b]
    name = sh.replace("$^{*}$", "\\,$^{*}$")
    def bold(a, b_):
        return (f"\\textbf{{{a:.3f}}}", f"{b_:.3f}") if a > b_ else \
               ((f"{a:.3f}", f"\\textbf{{{b_:.3f}}}") if b_ > a else (f"{a:.3f}", f"{b_:.3f}"))
    gj, tj = bold(g["judge"], t["judge"])
    ge, te = bold(g["em"], t["em"])
    rows_bench.append(
        f"{name} & {g['n_questions']} & {gj} & {tj} & {ge} & {te} & "
        f"{g['avg_turns']:.1f} & {t['avg_turns']:.1f}\\\\")

def slice_rows(key):
    g, t = S["GRPO"][key], S["turn-PPO"][key]
    out = []
    for label, k in (("judge", "macro_judge"), ("strict EM", "macro_em"), ("sub-EM", "macro_subem")):
        a, b_ = g[k], t[k]
        av, bv = (f"\\textbf{{{a:.3f}}}", f"{b_:.3f}") if a > b_ else (f"{a:.3f}", f"\\textbf{{{b_:.3f}}}")
        out.append(f"{label} & {av} & {bv}\\\\")
    return out

with open(os.path.join(A, "tables.tex"), "w") as f:
    f.write("\\newcommand{\\benchrows}{%\n" + "\n".join(rows_bench) + "}\n")
    f.write("\\newcommand{\\wikirows}{%\n" + "\n".join(slice_rows("wiki_answerable")) + "}\n")
    f.write("\\newcommand{\\liverows}{%\n" + "\n".join(slice_rows("live_web")) + "}\n")
    jg, jt = S["GRPO"]["judge_vs_em"], S["turn-PPO"]["judge_vs_em"]
    f.write("\\newcommand{\\calibrows}{%\n"
            f"GRPO & {jg['agreement']:.3f} & {jg['judge1_em0']} & {jg['judge0_em1']} & 1.000 & {S['GRPO']['no_answer_frac']:.3f}\\\\\n"
            f"turn-PPO & {jt['agreement']:.3f} & {jt['judge1_em0']} & {jt['judge0_em1']} & 1.000 & \\textbf{{{S['turn-PPO']['no_answer_frac']:.3f}}}\\\\}}\n")
print("figures + tables written")
