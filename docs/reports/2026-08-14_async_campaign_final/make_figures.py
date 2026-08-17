"""Figures for the async-campaign final report (house figstyle).

Inputs (produced by sibling scripts; rerun them first):
  campaign_data.json   <- mine_campaign_data.py (per-step series, all arms)
  items_analysis.json  <- analyze_items.py (turns / no-answer / conditional acc)
  ~/xiaoxuan/arlca-8b/outputs/judge/*_s75.summary.json (transfer metrics)

Engine-comparison figures encode: color = algorithm (GRPO blue, turn-PPO red),
linestyle/hatch = engine (async solid/plain, old sync dashed/hatched).
"""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/.claude/skills/scientific-figure-making/assets"))
import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from figstyle import apply_publication_style, PALETTE, make_grouped_bar, finalize_figure  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")
os.makedirs(A, exist_ok=True)
JUDGE = os.path.expanduser("~/xiaoxuan/arlca-8b/outputs/judge")

D = json.load(open(os.path.join(HERE, "campaign_data.json")))
IT = json.load(open(os.path.join(HERE, "items_analysis.json")))

apply_publication_style(font_size=15)

C_GRPO, C_TPPO = PALETTE["blue_main"], PALETTE["red_strong"]
ENGINE_ARMS = {  # label -> (data key, color, linestyle)
    "async GRPO": ("async_grpo", C_GRPO, "-"),
    "async turn-PPO": ("async_tppo", C_TPPO, "-"),
    "old GRPO": ("old_grpo", C_GRPO, "--"),
    "old turn-PPO": ("old_tppo", C_TPPO, "--"),
}
ITEM_KEY = {"async GRPO": "async GRPO", "async turn-PPO": "async tPPO",
            "old GRPO": "old GRPO", "old turn-PPO": "old tPPO"}
JUDGE_LAB = {"async GRPO": "agrpo_s75", "async turn-PPO": "atppo_s75",
             "old GRPO": "grpo_s75", "old turn-PPO": "turnppo_s75"}


def ema(ys, alpha=0.3):
    out, m = [], None
    for y in ys:
        m = y if m is None else alpha * y + (1 - alpha) * m
        out.append(m)
    return out


def series(key, field):
    st = D[key]["steps"]
    xs = [s for s in sorted(map(int, st)) if field in st[str(s)]]
    return xs, [st[str(s)][field] for s in xs]


# ---------------- fig 1: train curves + val (per step) ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for lab, (k, c, ls) in ENGINE_ARMS.items():
    xs, ys = series(k, "train_sr")
    xs, ys = xs[1:], ys[1:]  # step-0 row is val-only
    axes[0].plot(xs, ys, color=c, linestyle=ls, linewidth=1.0, alpha=0.25)
    axes[0].plot(xs, ema(ys), color=c, linestyle=ls, linewidth=2.4, label=lab)
    vx = sorted(map(int, D[k]["vals"]))
    axes[1].plot(vx, [D[k]["vals"][str(s)] for s in vx], color=c, linestyle=ls,
                 linewidth=2.2, marker="o", markersize=7, label=lab)
axes[0].set_xlabel("training step")
axes[0].set_ylabel("train success rate")
axes[0].set_ylim(0, None)
axes[0].legend(frameon=False, loc="lower right", fontsize=12)
axes[1].set_xlabel("training step")
axes[1].set_ylabel("greedy validation EM (512 q)")
axes[1].set_xticks([0, 25, 50, 75])
axes[1].set_ylim(0.15, 0.60)
finalize_figure(fig, os.path.join(A, "fig_train_val"), formats=["png", "pdf"])

# ---------------- fig 2: sample-matched learning curves ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for lab, (k, c, ls) in ENGINE_ARMS.items():
    cum = D[k]["cum_released"]
    xs = sorted(map(int, cum))
    axes[0].plot(xs, [cum[str(s)] / 1e3 for s in xs], color=c, linestyle=ls,
                 linewidth=2.2, label=lab)
    vx = sorted(map(int, D[k]["vals"]))
    axes[1].plot([cum[str(s)] / 1e3 for s in vx], [D[k]["vals"][str(s)] for s in vx],
                 color=c, linestyle=ls, linewidth=2.2, marker="o", markersize=7, label=lab)
axes[0].set_xlabel("training step")
axes[0].set_ylabel("cumulative released trajectories (k)")
axes[0].legend(frameon=False, loc="upper left", fontsize=12)
axes[1].set_xlabel("cumulative released trajectories (k)")
axes[1].set_ylabel("greedy validation EM (512 q)")
axes[1].set_ylim(0.15, 0.60)
finalize_figure(fig, os.path.join(A, "fig_sample_matched"), formats=["png", "pdf"])

# ---------------- fig 3: 4-way transfer (wiki macro ladder + live-web) ----------------
S = {lab: json.load(open(f"{JUDGE}/{jl}.summary.json")) for lab, jl in JUDGE_LAB.items()}
order = ["old GRPO", "async GRPO", "old turn-PPO", "async turn-PPO"]
bar_colors = [PALETTE["blue_secondary"], C_GRPO, PALETTE["red_2"], C_TPPO]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [3, 1.4]})
metrics = ["strict EM", "sub-EM", "judge"]
series_ = [[S[a]["wiki_answerable"][m] for m in ("macro_em", "macro_subem", "macro_judge")]
           for a in order]
make_grouped_bar(axes[0], metrics, series_, order, ylabel="wiki-answerable macro (7,152-q suite)",
                 colors=bar_colors, annotate=True)
axes[0].set_ylim(0, 0.66)
axes[0].legend(frameon=False, loc="upper left", fontsize=11, ncol=2)
lw = [[S[a]["live_web"]["macro_judge"]] for a in order]
make_grouped_bar(axes[1], ["live-web judge"], lw, order, ylabel="live-web macro judge",
                 colors=bar_colors, annotate=True)
axes[1].set_ylim(0, 0.66)
axes[1].legend().remove()
finalize_figure(fig, os.path.join(A, "fig_transfer_4way"), formats=["png", "pdf"])

# ---------------- fig 4: no-answer decomposition ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ja = [IT[ITEM_KEY[a]]["judge_all"] for a in order]
jc = [IT[ITEM_KEY[a]]["judge_answered"] for a in order]
na = [IT[ITEM_KEY[a]]["noans_frac"] for a in order]
x = np.arange(len(order))
w = 0.38
b1 = axes[0].bar(x - w / 2, ja, w, color=[c for c in bar_colors], label="judge, all questions")
b2 = axes[0].bar(x + w / 2, jc, w, color=[c for c in bar_colors], alpha=0.45,
                 hatch="//", label="judge, answered only")
for xi, (v1, v2, nv) in enumerate(zip(ja, jc, na)):
    axes[0].text(xi - w / 2, v1 + 0.012, f"{v1:.3f}", ha="center", fontsize=11)
    axes[0].text(xi + w / 2, v2 + 0.012, f"{v2:.3f}", ha="center", fontsize=11)
    axes[0].text(xi, -0.27, f"no-ans {nv:.1%}", ha="center", fontsize=10.5,
                 transform=axes[0].get_xaxis_transform(), color="0.25")
axes[0].set_xticks(x, ["old\nGRPO", "async\nGRPO", "old\nturn-PPO", "async\nturn-PPO"])
axes[0].set_ylabel("wiki-answerable judge score")
axes[0].set_ylim(0, 0.72)
axes[0].legend(frameon=False, loc="upper left", fontsize=11)
# panel b: no-answer rate by benchmark, old vs async turn-PPO
bench = list(IT["async tPPO"]["noans_by_bench"])
short = ["NQ", "TriviaQA", "PopQA", "Hotpot", "2Wiki", "Musique", "Bamb."]
bshort = dict(zip(sorted(["NQ_rand1000", "TriviaQA_rand1000", "PopQA_rand1000",
                          "HotpotQA_rand1000", "2WikiMultihopQA_rand1000",
                          "Musique_rand1000", "Bamboogle"]),
                  ["2Wiki", "Bamb.", "Hotpot", "Musique", "NQ", "PopQA", "TriviaQA"]))
series_b = [[IT["old tPPO"]["noans_by_bench"][b] for b in bench],
            [IT["async tPPO"]["noans_by_bench"][b] for b in bench]]
make_grouped_bar(axes[1], [bshort[b] for b in bench], series_b,
                 ["old turn-PPO", "async turn-PPO"], ylabel="no-answer fraction",
                 colors=[PALETTE["red_2"], C_TPPO])
axes[1].tick_params(axis="x", labelsize=11)
axes[1].legend(frameon=False, loc="upper left", fontsize=11)
finalize_figure(fig, os.path.join(A, "fig_noanswer"), formats=["png", "pdf"])

# ---------------- fig 5: turn distributions ----------------
fig, ax = plt.subplots(figsize=(9, 4.8))
bins = list(range(0, 40, 4))
labels_h = [f"{lo}-{lo+3}" for lo in bins[:-1]]
series_h = []
for a in order:
    h = IT[ITEM_KEY[a]]["turns_hist"]
    tot = sum(h.values())
    series_h.append([h.get(str(lo), 0) / tot for lo in bins[:-1]])
make_grouped_bar(ax, labels_h, series_h, order, ylabel="fraction of eval questions",
                 colors=bar_colors)
ax.set_xlabel("turns used (wiki-answerable questions)")
ax.legend(frameon=False, loc="upper right", fontsize=11)
finalize_figure(fig, os.path.join(A, "fig_turns_dist"), formats=["png", "pdf"])

# ---------------- fig 6: estimator sweep (5 arms, old engine) ----------------
EST = {
    "GRPO": ("old_grpo", PALETTE["blue_main"]),
    "turn-PPO": ("old_tppo", PALETTE["red_strong"]),
    "token-PPO": ("token_ppo", PALETTE["teal"]),
    "GiGPO": ("gigpo", PALETTE["violet"]),
    "HCAPO": ("hcapo_ans", PALETTE["green_3"]),
}
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for lab, (k, c) in EST.items():
    xs, ys = series(k, "train_sr")
    xs, ys = xs[1:], ys[1:]
    axes[0].plot(xs, ys, color=c, linewidth=1.0, alpha=0.22)
    axes[0].plot(xs, ema(ys), color=c, linewidth=2.4, label=lab)
    vx = sorted(map(int, D[k]["vals"]))
    axes[1].plot(vx, [D[k]["vals"][str(s)] for s in vx], color=c, linewidth=2.0,
                 linestyle="--", marker="o", markersize=8, label=lab)
axes[0].set_xlabel("training step")
axes[0].set_ylabel("train success rate")
axes[0].set_ylim(0, None)
axes[0].legend(frameon=False, loc="lower right", fontsize=12)
axes[1].set_xlabel("training step")
axes[1].set_ylabel("greedy validation EM (512 q)")
axes[1].set_xticks([0, 25, 50, 75])
axes[1].set_ylim(0, 0.60)
finalize_figure(fig, os.path.join(A, "fig_estimators"), formats=["png", "pdf"])

# ---------------- fig 7: per-turn response length over training ----------------
fig, ax = plt.subplots(figsize=(9.5, 5.2))
for lab, (k, c) in EST.items():
    xs, ys = series(k, "resp_len_p50")
    xs, ys = xs[1:], ys[1:]  # step-0 row is val-only
    ax.plot(xs, ys, color=c, linewidth=1.0, alpha=0.22)
    ax.plot(xs, ema(ys), color=c, linewidth=2.4, label=lab)
# HCAPO p10-p90 band shows the cap-pinned upper tail
hx, h10 = series("hcapo_ans", "resp_len_p10")
_, h90 = series("hcapo_ans", "resp_len_p90")
ax.fill_between(hx[1:], ema(h10[1:]), ema(h90[1:]), color=EST["HCAPO"][1], alpha=0.13,
                linewidth=0, label="HCAPO p10–p90")
ax.axhline(1024, color="gray", linewidth=1.2, linestyle=":")
ax.text(2, 1024, "response cap (1,024)", fontsize=11, color="gray", va="bottom")
ax.set_xlabel("training step")
ax.set_ylabel("tokens per turn (batch median)")
ax.set_ylim(0, 1120)
ax.legend(frameon=False, loc="center right", fontsize=12)
finalize_figure(fig, os.path.join(A, "fig_resp_len"), formats=["png", "pdf"])

print("figures written to", A)
