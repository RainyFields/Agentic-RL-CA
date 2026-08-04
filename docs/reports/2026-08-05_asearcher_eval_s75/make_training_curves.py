"""Training curves, GRPO vs turn-PPO (8B fast75 arms): per-step train success rate
(solid, EMA-smoothed) + greedy validation EM at steps 0/25/50/75 (markers, dashed).

Mined from the arms' console logs. Val values are embedded in step metric lines
(grep val-core/macro_em), except step-0 val which prints standalone.
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


def ema(ys, alpha=0.3):
    out, m = [], None
    for y in ys:
        m = y if m is None else alpha * y + (1 - alpha) * m
        out.append(m)
    return out


train, val = {}, {}
for arm, path in LOGS.items():
    tr, vl = {}, {}
    for line in open(path, errors="replace"):
        m = re.search(r"step:(\d+) - ", line)
        if m:
            s = int(m.group(1))
            r = re.search(r"episode/success_rate:([\d.]+)", line)
            if r:
                tr[s] = float(r.group(1))
            v = re.search(r"val-core/macro_em[\"']?[:=]\s*([\d.]+)", line)
            if v:
                vl[s] = float(v.group(1))
        elif "val-core/macro_em" in line:  # step-0 standalone val print
            v = re.search(r"val-core/macro_em[\"']?[:=]\s*([\d.]+)", line)
            if v and 0 not in vl:
                vl[0] = float(v.group(1))
    train[arm] = (sorted(tr), [tr[s] for s in sorted(tr)])
    val[arm] = (sorted(vl), [vl[s] for s in sorted(vl)])
    print(f"{arm}: train steps={len(tr)}, val points={{{', '.join(f'{s}:{vl[s]:.3f}' for s in sorted(vl))}}}")

fig, ax = plt.subplots(figsize=(9, 5.5))
for arm in LOGS:
    xs, ys = train[arm]
    ax.plot(xs, ys, color=C2[arm], linewidth=1.2, alpha=0.3)
    ax.plot(xs, ema(ys), color=C2[arm], linewidth=2.5, label=arm)
ax.set_xlabel("training step")
ax.set_ylabel("train success rate")
ax.set_ylim(0, None)
ax.legend(frameon=False, loc="lower right")

finalize_figure(fig, HERE / "assets" / "fig_training_curves")
print("wrote assets/fig_training_curves.{png,pdf}")
