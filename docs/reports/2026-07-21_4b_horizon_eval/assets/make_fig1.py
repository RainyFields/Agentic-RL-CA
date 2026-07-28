#!/usr/bin/env python3
"""Fig 1 — training accuracy over steps, one panel per environment, one line per model.
AlfWorld (1.7B): seen success-rate (reward_models). SearchQA (4B): val_2048 macro-EM parsed from
the training launch logs. token-PPO is SearchQA-only (per the report's model matrix)."""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "analysis", "figures"))
sys.path.insert(0, os.path.dirname(__file__))
from figstyle import PALETTE, apply_publication_style, finalize_figure  # noqa: E402
import alfworld_common as A  # noqa: E402

apply_publication_style(font_size=13, axes_linewidth=1.8)
FIGS = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(FIGS, exist_ok=True)
LOGS = "/home/tiger/xiaoxuan/worker_logs/launches"

# unified colours: GRPO blue, GiGPO teal, turn-PPO red, token-PPO violet (SearchQA-only)
COL = {"grpo": PALETTE["blue_main"], "gigpo": PALETTE.get("teal", "#2a9d8f"),
       "turn_ppo": PALETTE.get("red_strong", "#d1495b"), "token_ppo": PALETTE.get("violet", "#7b5cff")}
PRETTY = {"grpo": "GRPO", "gigpo": "GiGPO", "turn_ppo": "turn-PPO", "token_ppo": "token-PPO"}

SEARCHQA_LOGS = {
    "grpo":      f"{LOGS}/20260719_161007_arlca-token-grpo-4b-s0.log",
    "gigpo":     f"{LOGS}/20260719_161015_arlca-gigpo-4b-s0.log",
    "turn_ppo":  f"{LOGS}/20260719_160959_arlca-turn-ppo-b0-4b-s0.log",
    "token_ppo": f"{LOGS}/20260719_160951_arlca-token-ppo-4b-s0.log",
}
SEARCHQA_ORDER = ["grpo", "gigpo", "turn_ppo", "token_ppo"]


def searchqa_training():
    """model -> DataFrame(step, val) from the training logs (val_2048 macro-EM per step, last wins)."""
    out = {}
    for m, f in SEARCHQA_LOGS.items():
        pairs = {}
        with open(f, errors="ignore") as fh:
            txt = fh.read().replace("\r", "\n")
        for line in txt.split("\n"):
            mt = re.search(r"step:(\d+) .*val-core/macro_em:([0-9.]+)", line)
            if mt:
                pairs[int(mt.group(1))] = float(mt.group(2))
        out[m] = pd.DataFrame(sorted(pairs.items()), columns=["step", "val"])
    return out


def main():
    alf = A.load_alfworld_training()
    sqa = searchqa_training()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.8))
    # AlfWorld
    for m in A.ALF_METHOD_ORDER:
        d = alf[m]
        axL.plot(d.step, d.success_rate, lw=2.2, ms=3, marker="o", color=COL[m], label=PRETTY[m])
    axL.set_title("AlfWorld (1.7B) — seen success rate")
    axL.set_xlabel("training step")
    axL.set_ylabel("success rate")
    axL.grid(True, alpha=0.3)
    axL.legend(frameon=False, fontsize=10)
    axL.annotate("PPO 150 steps;\nGRPO/GiGPO 300", xy=(150, 0.55), fontsize=8, color="#777")
    # SearchQA
    for m in SEARCHQA_ORDER:
        d = sqa[m]
        if len(d):
            axR.plot(d.step, d.val, lw=2.2, ms=3, marker="o", color=COL[m], label=PRETTY[m])
    axR.set_title("SearchQA (4B) — val macro-EM")
    axR.set_xlabel("training step")
    axR.set_ylabel("val_2048 macro-EM")
    axR.grid(True, alpha=0.3)
    axR.legend(frameon=False, fontsize=10)
    fig.suptitle("Fig 1 — training accuracy per environment", y=1.02)
    finalize_figure(fig, os.path.join(FIGS, "fig1_training_curves.png"))

    print("=== SearchQA training final vals ===")
    for m in SEARCHQA_ORDER:
        d = sqa[m]
        print(f"  {PRETTY[m]:10s} {len(d):2d} pts, step {d.step.min()}..{d.step.max()}, final {d.val.iloc[-1]:.3f}")
    print("=== AlfWorld training final vals ===")
    for m in A.ALF_METHOD_ORDER:
        d = alf[m]
        print(f"  {PRETTY[m]:10s} {len(d):2d} pts, final {d.success_rate.iloc[-1]:.3f}")
    print("\nfig ->", os.path.abspath(os.path.join(FIGS, "fig1_training_curves.png")))


if __name__ == "__main__":
    main()
