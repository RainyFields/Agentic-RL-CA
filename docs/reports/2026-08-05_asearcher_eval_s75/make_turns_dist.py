"""Turn-count distribution, GRPO vs turn-PPO (step-75), ASearcher eval suite.

One panel, overlaid discrete histograms (fraction of trajectories per turn count),
house style + the report's arm colors (C2). Reads the judge items JSONL (n_turns per
trajectory, 7,152 each).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/tiger/.claude/skills/scientific-figure-making/assets")
from figstyle import apply_publication_style, PALETTE, finalize_figure  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
ITEMS = {
    "GRPO": "/home/tiger/xiaoxuan/arlca-8b/outputs/judge/grpo_s75.items.jsonl",
    "turn-PPO": "/home/tiger/xiaoxuan/arlca-8b/outputs/judge/turnppo_s75.items.jsonl",
}
colors = list(PALETTE.values())
C2 = {"GRPO": colors[0], "turn-PPO": colors[1]}

apply_publication_style()

turns = {}
for arm, path in ITEMS.items():
    turns[arm] = np.array([json.loads(l)["n_turns"] for l in open(path)])

max_t = max(int(t.max()) for t in turns.values())
bins = np.arange(0.5, max_t + 1.5, 1.0)

fig, ax = plt.subplots(figsize=(9, 5.5))
for arm, t in turns.items():
    w = np.ones_like(t, dtype=float) / len(t)
    ax.hist(t, bins=bins, weights=w, histtype="stepfilled", alpha=0.35,
            color=C2[arm], linewidth=0)
    ax.hist(t, bins=bins, weights=w, histtype="step",
            color=C2[arm], linewidth=2.5, label=f"{arm}  (median {int(np.median(t))})")
    ax.axvline(np.median(t), color=C2[arm], linestyle="--", linewidth=1.5, alpha=0.8)

ax.set_xlabel("turns per question")
ax.set_ylabel("fraction of trajectories")
ax.set_xlim(0.5, max_t + 0.5)
ax.legend(frameon=False, loc="upper center")
for arm, t in turns.items():
    print(f"{arm}: n={len(t)} mean={t.mean():.2f} median={np.median(t):.0f} "
          f"p10={np.percentile(t, 10):.0f} p90={np.percentile(t, 90):.0f} "
          f"at-cap(>=32)={np.mean(t >= 32) * 100:.1f}%")

finalize_figure(fig, HERE / "assets" / "fig_turns_dist")
print("wrote assets/fig_turns_dist.{png,pdf}")
