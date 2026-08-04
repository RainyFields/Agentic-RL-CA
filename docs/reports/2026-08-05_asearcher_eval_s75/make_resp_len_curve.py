"""Per-turn response length vs training step, GRPO vs turn-PPO (8B fast75 arms).

Mined from the arms' console logs: every `step:N - ...` metrics line carries
response_length/mean (mean generated tokens per TURN across that step's released
trajectories). One panel, both arms, house style.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/tiger/.claude/skills/scientific-figure-making/assets")
from figstyle import apply_publication_style, PALETTE, finalize_figure  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
LOGS = {
    "GRPO": "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/p8b_arm_grpo.log",
    "turn-PPO": "/home/tiger/xiaoxuan/Agentic-RL-CA/outputs/p8b_arm_ppo.log",
}
colors = list(PALETTE.values())
C2 = {"GRPO": colors[0], "turn-PPO": colors[1]}

apply_publication_style()

series = {}
for arm, path in LOGS.items():
    pts = {}
    for line in open(path, errors="replace"):
        m = re.search(r"step:(\d+) - ", line)
        if not m:
            continue
        r = re.search(r"response_length/mean:([\d.]+)", line)
        if r:
            pts[int(m.group(1))] = float(r.group(1))  # dedup restarts: last wins
    steps = sorted(pts)
    series[arm] = (steps, [pts[s] for s in steps])
    print(f"{arm}: {len(steps)} steps, first={pts[steps[0]]:.0f} last={pts[steps[-1]]:.0f} "
          f"min={min(pts.values()):.0f} max={max(pts.values()):.0f}")

fig, ax = plt.subplots(figsize=(9, 5.5))
for arm, (xs, ys) in series.items():
    ax.plot(xs, ys, color=C2[arm], linewidth=2.5, label=arm)
ax.axhline(1024, color="gray", linewidth=1.0, linestyle=":")
ax.text(2, 1024, "1024 cap", va="bottom", fontsize=12, color="gray")
ax.set_xlabel("training step")
ax.set_ylabel("response tokens / turn (mean)")
ax.legend(frameon=False)

finalize_figure(fig, HERE / "assets" / "fig_resp_len_steps")
print("wrote assets/fig_resp_len_steps.{png,pdf}")
