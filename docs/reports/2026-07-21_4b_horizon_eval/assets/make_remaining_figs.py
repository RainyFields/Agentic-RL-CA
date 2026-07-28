#!/usr/bin/env python3
"""Fig 3 (SearchQA turns box per dataset), Fig 4 (SearchQA actual-vs-gold turns, split
accurate/failed/all), Fig 5 (AlfWorld+SearchQA accuracy vs turns-used), and Table 1 (combined eval
accuracy). Handles partial data — uses whatever SearchQA arms are present. Pure post-processing."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "figures"))
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import PALETTE, apply_publication_style, finalize_figure  # noqa: E402
import horizon_common as H  # noqa: E402
import alfworld_common as A  # noqa: E402

apply_publication_style(font_size=13, axes_linewidth=1.8)
FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")
RES = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# canonical method -> colour/label (shared across environments)
UCOLOR = {"grpo": PALETTE["blue_main"], "gigpo": PALETTE.get("teal", "#2a9d8f"),
          "turn_ppo": PALETTE.get("red_strong", "#d1495b"), "token_ppo": PALETTE.get("violet", "#7b5cff"),
          "floor": PALETTE.get("neutral", "#9aa0a6")}
UPRETTY = {"grpo": "GRPO", "gigpo": "GiGPO", "turn_ppo": "turn-PPO", "token_ppo": "token-PPO", "floor": "pre-RL floor"}
UORDER = ["floor", "grpo", "gigpo", "turn_ppo", "token_ppo"]
# SearchQA horizon_common keys -> canonical
SQA_CANON = {"token_grpo": "grpo", "gigpo": "gigpo", "turn_ppo": "turn_ppo", "token_ppo": "token_ppo", "floor": "floor"}
SQA_LABELS = {"token_grpo": "eval4b_token_grpo_s0", "gigpo": "eval4b_gigpo_s0",
              "turn_ppo": "eval4b_turn_ppo_s0", "token_ppo": "eval4b_token_ppo_s0", "floor": "eval4b_floor"}


def load_searchqa():
    """canonical method -> per-episode DataFrame (whatever is present under outputs/eval_full)."""
    hop = H.load_hop_table()
    out = {}
    for key, lab in SQA_LABELS.items():
        if os.path.isdir(os.path.join(H.EVAL_ROOT, lab)):
            try:
                out[SQA_CANON[key]] = H.load_eval(lab, hop)
            except Exception as e:
                print(f"  skip {key}: {e}")
    return out


def _present(sqa, include_floor=True):
    return [m for m in UORDER if m in sqa and (include_floor or m != "floor")]


# ---------------- Fig 3: SearchQA turns box per dataset ----------------
def fig3_turns_box(sqa):
    methods = _present(sqa, include_floor=False)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    nD, nM = len(H.DS_ORDER), len(methods)
    width = 0.8 / max(1, nM)
    for j, m in enumerate(methods):
        d = sqa[m]
        data = [d[d.data_source == ds]["turns"].values for ds in H.DS_ORDER]
        pos = [i + (j - (nM - 1) / 2) * width for i in range(nD)]
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black"))
        for box in bp["boxes"]:
            box.set(facecolor=UCOLOR[m], alpha=0.7)
    ax.set_xticks(range(nD))
    ax.set_xticklabels([H.DS_PRETTY[ds] for ds in H.DS_ORDER])
    ax.set_ylabel("turns used per episode")
    ax.set_title("Fig 3 — SearchQA trajectory length per dataset (cap = 4)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=[Line2D([0], [0], color=UCOLOR[m], lw=8, alpha=0.7, label=UPRETTY[m]) for m in methods],
              frameon=False, fontsize=10, ncol=len(methods), loc="upper left")
    finalize_figure(fig, os.path.join(FIGS, "fig3_searchqa_turns_box.png"))


# ---------------- Fig 4: SearchQA actual vs gold turns (accurate/failed/all) ----------------
def fig4_turns_vs_gold(sqa):
    methods = _present(sqa, include_floor=False)
    panels_ds = [("musique", [2, 3, 4]), ("2wikimultihopqa", [2, 4])]
    outcomes = [("all", None), ("accurate", 1.0), ("failed", 0.0)]
    fig, axes = plt.subplots(len(panels_ds), len(outcomes), figsize=(13, 7.5), sharey=True)
    for r, (ds, xs) in enumerate(panels_ds):
        for c, (oname, ofilter) in enumerate(outcomes):
            ax = axes[r][c]
            for m in methods:
                d = sqa[m]
                d = d[d.data_source == ds]
                if ofilter is not None:
                    d = d[d.em == ofilter]
                g = d.groupby("H_gold")["turns"].agg(["mean", "std", "size"])
                xv = [h for h in xs if h in g.index and g.loc[h, "size"] >= 5]
                mu = [g.loc[h, "mean"] for h in xv]
                sd = [g.loc[h, "std"] if pd.notna(g.loc[h, "std"]) else 0 for h in xv]
                ax.plot(xv, mu, marker="o", lw=2, ms=6, color=UCOLOR[m], label=UPRETTY[m])
                ax.fill_between(xv, np.array(mu) - np.array(sd), np.array(mu) + np.array(sd),
                                color=UCOLOR[m], alpha=0.12)
            ax.plot(xs, xs, ls=":", color="#bbb", lw=1.2)   # y=x reference
            ax.set_xticks(xs)
            if r == 0:
                ax.set_title(oname)
            if c == 0:
                ax.set_ylabel(f"{H.DS_PRETTY[ds]}\nmean turns used")
            if r == len(panels_ds) - 1:
                ax.set_xlabel("gold turns $H_{gold}$")
            ax.grid(True, alpha=0.3)
    axes[0][0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Fig 4 — SearchQA: turns used vs required hops (dotted = y=x; band = ±1 SD)", y=1.01)
    finalize_figure(fig, os.path.join(FIGS, "fig4_searchqa_turns_vs_gold.png"))


# ---------------- Fig 5: accuracy vs turns used (AlfWorld + SearchQA) ----------------
def fig5_acc_vs_turns(sqa):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
    # AlfWorld (left)
    ep = A.load_alfworld_episodes()
    ep = ep.copy()
    bins, labels = [0, 6, 8, 12, 20, 100], ["≤6", "7–8", "9–12", "13–20", ">20"]
    ep["bin"] = pd.cut(ep["turns"], bins=bins, labels=labels)
    for m in A.ALF_METHOD_ORDER:
        d = ep[ep.method == m]
        g = d.groupby("bin", observed=False)["success"].agg(["mean", "size"])
        xs = [i for i, lab in enumerate(labels) if g.loc[lab, "size"] >= 3]
        axL.plot(xs, [g.loc[labels[i], "mean"] for i in xs], marker="o", lw=2.2, ms=7,
                 color=UCOLOR[m], label=UPRETTY[m])
    axL.set_xticks(range(len(labels)))
    axL.set_xticklabels(labels)
    axL.set_xlabel("turns used by the model")
    axL.set_ylabel("accuracy")
    axL.set_title("AlfWorld (1.7B)")
    axL.grid(True, alpha=0.3)
    axL.legend(frameon=False, fontsize=10)
    # SearchQA (right): turns 1..4
    methods = _present(sqa, include_floor=True)
    for m in methods:
        d = sqa[m]
        g = d.groupby("turns")["em"].agg(["mean", "size"])
        xs = [t for t in [1, 2, 3, 4] if t in g.index and g.loc[t, "size"] >= 20]
        axR.plot(xs, [g.loc[t, "mean"] for t in xs], marker="o", lw=2.2, ms=7,
                 color=UCOLOR[m], label=UPRETTY[m])
    axR.set_xticks([1, 2, 3, 4])
    axR.set_xlabel("turns used by the model")
    axR.set_ylabel("accuracy (EM)")
    axR.set_title("SearchQA (4B)")
    axR.grid(True, alpha=0.3)
    axR.legend(frameon=False, fontsize=9)
    fig.suptitle("Fig 5 — accuracy vs turns used by the model", y=1.02)
    finalize_figure(fig, os.path.join(FIGS, "fig5_acc_vs_turns.png"))


# ---------------- Table 1: combined eval accuracy ----------------
def table1(sqa):
    rows = []
    for m in UORDER:
        rec = {"method": UPRETTY[m]}
        # AlfWorld (grpo/gigpo/turn_ppo only; floor≈BC baseline 0.44)
        rec["alfworld"] = A.ALF_EVAL.get(m, 0.44 if m == "floor" else np.nan)
        # SearchQA
        if m in sqa:
            d = sqa[m]
            per = [d[d.data_source == ds]["em"].mean() for ds in H.DS_ORDER]
            rec["searchqa_macro"] = float(np.nanmean(per))
            for ds, e in zip(H.DS_ORDER, per):
                rec[f"sqa_{ds}"] = e
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "table1_eval_accuracy.csv"), index=False)
    return df


def main():
    sqa = load_searchqa()
    print("SearchQA methods present:", list(sqa.keys()))
    fig3_turns_box(sqa)
    fig4_turns_vs_gold(sqa)
    fig5_acc_vs_turns(sqa)
    t1 = table1(sqa)
    print("\n=== Table 1 (eval accuracy) ===")
    cols = ["method", "alfworld", "searchqa_macro"]
    print(t1[[c for c in cols if c in t1]].round(3).to_string(index=False))
    print("\nfigs ->", os.path.abspath(FIGS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
