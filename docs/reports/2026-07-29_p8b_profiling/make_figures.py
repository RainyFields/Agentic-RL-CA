#!/usr/bin/env python3
"""P8B profiling figures (house style). Inputs: assets/{timing,rollouts}.json.
Outputs: assets/fig_*.{png,pdf} + per-figure data JSON alongside.
Run: python make_figures.py
"""
import json, os, sys

sys.path.insert(0, os.path.expanduser("~/.claude/skills/scientific-figure-making/assets"))
from figstyle import (apply_publication_style, PALETTE, make_grouped_bar,
                      make_single_bar, make_lines, finalize_figure)
import numpy as np
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")
timing = json.load(open(os.path.join(A, "timing.json")))
rollouts = json.load(open(os.path.join(A, "rollouts.json")))
apply_publication_style(font_size=15)

CFGS = ["grpo_vanilla", "grpo_partial", "ppo_vanilla", "ppo_partial"]
NICE = {"grpo_vanilla": "GRPO\nvanilla", "grpo_partial": "GRPO\npartial",
        "ppo_vanilla": "turnPPO\nvanilla", "ppo_partial": "turnPPO\npartial"}


def released(cfg, rec):
    return rec.get("partial/released_traj", 1280.0)  # vanilla releases the full batch


# ---- fig 1: step-time breakdown (stacked gen / update / other), mean over steps ----
comp = {}
for cfg in CFGS:
    recs = [r for r in timing.get(cfg, []) if "timing_s/step" in r]
    if not recs:
        continue
    gen = np.mean([r["timing_s/gen"] for r in recs])
    upd = np.mean([r.get("timing_s/update_actor", 0) + r.get("timing_s/update_critic", 0)
                   for r in recs])
    tot = np.mean([r["timing_s/step"] for r in recs])
    comp[cfg] = (gen, upd, max(0.0, tot - gen - upd), tot)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
cats = [c for c in CFGS if c in comp]
x = np.arange(len(cats))
bottom = np.zeros(len(cats))
for j, (name, color) in enumerate([("generation", PALETTE["blue_main"]),
                                   ("actor/critic update", PALETTE["red_strong"]),
                                   ("logprob/ref/other", PALETTE["neutral"])]):
    vals = np.array([comp[c][j] for c in cats]) / 60.0
    ax.bar(x, vals, 0.62, bottom=bottom, color=color, label=name)
    bottom += vals
ax.set_xticks(x); ax.set_xticklabels([NICE[c] for c in cats])
ax.set_ylabel("mean wall-clock per cycle (min)")
ax.legend(loc="upper right")
finalize_figure(fig, os.path.join(A, "fig_step_breakdown"))
json.dump({c: comp[c] for c in cats}, open(os.path.join(A, "fig_step_breakdown.json"), "w"), indent=1)

# ---- fig 2: throughput — step-seconds per completed trajectory + projected days ----
fig, ax = plt.subplots(figsize=(6.6, 4.4))
sptraj = {}
for cfg in cats:
    recs = [r for r in timing[cfg] if "timing_s/step" in r]
    tot_t = sum(r["timing_s/step"] for r in recs)
    tot_rel = sum(released(cfg, r) for r in recs)
    sptraj[cfg] = tot_t / max(1.0, tot_rel)
vals = [sptraj[c] for c in cats]
bars = make_single_bar(ax, [NICE[c] for c in cats], vals,
                       colors=[PALETTE["blue_main"], PALETTE["blue_secondary"],
                               PALETTE["red_strong"], PALETTE["red_2"]],
                       ylabel="step-seconds / completed trajectory", fmt="{:.1f}")
for c, v, xi in zip(cats, vals, range(len(cats))):
    days = v * 1280 * 150 / 86400.0
    ax.text(xi, v / 2, f"{days:.1f} d\n/arm", ha="center", va="center", fontsize=12,
            color="white", fontweight="bold")
ax.set_ylim(0, max(vals) * 1.25)
finalize_figure(fig, os.path.join(A, "fig_throughput"))
json.dump(sptraj, open(os.path.join(A, "fig_throughput.json"), "w"), indent=1)

# ---- fig 3: turns histogram (per config) ----
fig, ax = plt.subplots(figsize=(7.2, 4.4))
series = []
colors = [PALETTE["blue_main"], PALETTE["blue_secondary"], PALETTE["red_strong"], PALETTE["red_2"]]
for i, cfg in enumerate(CFGS):
    if cfg not in rollouts:
        continue
    h = rollouts[cfg]["turns_histogram"]
    n = rollouts[cfg]["n_trajs"]
    xs = sorted(int(k) for k in h)
    series.append({"x": xs, "y": [h[str(k)] / n for k in xs],
                   "label": NICE[cfg].replace("\n", " "), "color": colors[i]})
make_lines(ax, series, ylabel="fraction of trajectories", xlabel="turns per trajectory",
           smooth=False, raw_alpha=0.0)
ax.legend()
finalize_figure(fig, os.path.join(A, "fig_turns_hist"))

# ---- fig 4: partial-rollout buffer dynamics per cycle ----
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
for ax, cfg, title in zip(axes, ["grpo_partial", "ppo_partial"], ["GRPO partial", "turnPPO partial"]):
    recs = [r for r in timing[cfg] if "partial/released_traj" in r]
    steps = [r["step"] for r in recs]
    for key, label, color in [("partial/released_traj", "released (trained)", PALETTE["blue_main"]),
                              ("partial/pending_traj", "pending (carried)", PALETTE["red_strong"]),
                              ("partial/held_complete_traj", "held (group-atomic)", PALETTE["neutral"])]:
        ax.plot(steps, [r[key] for r in recs], marker="o", lw=2.4, color=color, label=label)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("cycle")
    ax.set_xticks(steps)
axes[0].set_ylabel("trajectories")
axes[0].legend(fontsize=11)
finalize_figure(fig, os.path.join(A, "fig_partial_dynamics"))
print("figures written to", A)
