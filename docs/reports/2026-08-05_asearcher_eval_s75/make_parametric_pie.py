"""Parametric-recall confound pie: how the early-training 8B (temp 1, ~base policy)
solves training queries. From the fast-profiling GRPO training dump (steps 1-4,
944 unique queries x 5 attempts).

Slices (query level):
  unsolved                      - no winning attempt out of 5
  solved via search only        - every win used >=1 search
  solved mixed                  - wins both with and without search
  solved memory-only            - every win used ZERO searches (pure recall)
"""
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import zstandard as zstd

sys.path.insert(0, "/home/tiger/.claude/skills/scientific-figure-making/assets")
from figstyle import apply_publication_style, PALETTE, finalize_figure  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
DUMP = "/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/p8b_fast/rollout_grpo_fast.jsonl.zst"

groups = defaultdict(lambda: defaultdict(lambda: {"s": 0, "w": False}))
with open(DUMP, "rb") as fh:
    for line in io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh),
                                 encoding="utf-8", errors="replace"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = groups[r["uid"]][r["traj_uid"]]
        if "<search>" in r.get("raw_model_response", ""):
            t["s"] += 1
        if r.get("env_won"):
            t["w"] = True

unsolved = search_only = mixed = memory_only = 0
for trs in groups.values():
    wins = [t for t in trs.values() if t["w"]]
    if not wins:
        unsolved += 1
        continue
    zs = sum(1 for t in wins if t["s"] == 0)
    if zs == 0:
        search_only += 1
    elif zs == len(wins):
        memory_only += 1
    else:
        mixed += 1
n = len(groups)
print(f"queries={n} unsolved={unsolved} search_only={search_only} "
      f"mixed={mixed} memory_only={memory_only}")

apply_publication_style(font_size=15)
colors = list(PALETTE.values())
sizes = [unsolved, search_only, mixed, memory_only]
labels = [
    f"unsolved\n{unsolved} ({unsolved / n:.0%})",
    f"solved via search only\n{search_only} ({search_only / n:.0%})",
    f"solved: memory or search\n{mixed} ({mixed / n:.0%})",
    f"solved by memory only\n{memory_only} ({memory_only / n:.0%})",
]
cols = ["#B8C4CC", colors[0], colors[1], PALETTE.get("red_strong", "#C0392B")]
explode = [0, 0, 0.03, 0.08]

fig, ax = plt.subplots(figsize=(8.5, 6))
wedges, _ = ax.pie(sizes, colors=cols, explode=explode, startangle=90,
                   counterclock=False,
                   wedgeprops=dict(linewidth=2, edgecolor="white"))
for w, lab in zip(wedges, labels):
    ang = (w.theta1 + w.theta2) / 2
    import numpy as np
    x, y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    ax.annotate(lab, xy=(x, y), xytext=(1.22 * x, 1.18 * y),
                ha="center" if abs(x) < 0.35 else ("left" if x > 0 else "right"),
                va="center", fontsize=14)
ax.set(aspect="equal")

finalize_figure(fig, HERE / "assets" / "fig_parametric_pie")
print("wrote assets/fig_parametric_pie.{png,pdf}")
