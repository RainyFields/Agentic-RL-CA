#!/usr/bin/env python3
"""Build the Wave-1 report figures from csv/<run>.csv (all 7 arms, every seed).

Rerun:  python3 extract_metrics.py && python3 build_figures.py && python3 build_f8a.py
Outputs figs/*.{png,pdf} + figs/finals.csv (per-seed step-500 finals, the bar-chart source).
House style via the scientific-figure-making skill's figstyle.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/.claude/skills/scientific-figure-making/assets"))
from figstyle import PALETTE, apply_publication_style, finalize_figure  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "csv")
FIGS = os.path.join(HERE, "figs")
BASE_EM = 0.2246  # untrained Qwen3-1.7B on val_2048 (macro-EM)
EVAL_STEPS = list(range(0, 501, 25))

apply_publication_style(font_size=13, axes_linewidth=1.8)

C = {
    "token_grpo": PALETTE["blue_main"],
    "gigpo": PALETTE["teal"],
    "hcapo": PALETTE["highlight"],   # amber — flags the truncation collapse
    "token_ppo": PALETTE["neutral"],
    "b0": PALETTE["violet"],
    "b1": PALETTE["red_strong"],
    "b1_shuffle": PALETTE["green_3"],
}

# arm -> (display label, colour key, [seed runs], credit family)
ARMS = {
    "gigpo":      ("GiGPO",         "gigpo",      ["gigpo_s0", "gigpo_s1"],                              "critic-free group-relative (turn)"),
    "token_grpo": ("token-GRPO",    "token_grpo", ["token_grpo_s0", "token_grpo_s1"],                   "critic-free group-relative (traj)"),
    "hcapo":      ("HCAPO",         "hcapo",      ["hcapo_s0"],                                          "critic-free per-turn amplifier"),
    "b1_shuffle": ("B1-shuffle",    "b1_shuffle", ["b1_shuffle_s0", "b1_shuffle_s1", "b1_shuffle_s2"],  "critic + shuffled step reward"),
    "b1":         ("B1",            "b1",         ["b1_s0", "b1_s1", "b1_s2"],                           "critic + privileged step reward"),
    "b0":         ("turn-PPO (B0)", "b0",         ["turn_ppo_b0_s0", "turn_ppo_b0_s1", "turn_ppo_b0_s2"], "critic, per-turn"),
    "token_ppo":  ("token-PPO",     "token_ppo",  ["token_ppo_s0"],                                      "critic, per-token"),
}
HORIZON = {
    "b0_8t": ("turn-PPO (B0), 8-turn", "b0", ["turn_ppo_b0_8t_s0"]),
    "b1_8t": ("B1, 8-turn",            "b1", ["b1_8t_s0"]),
}


def load(run):
    rows = {}
    with open(os.path.join(CSV, f"{run}.csv")) as f:
        for r in csv.DictReader(f):
            rows[int(r["step"])] = r
    return rows


R = {r: load(r) for arm in list(ARMS.values()) + list(HORIZON.values()) for r in arm[2]}


def col(run, name, every=1):
    x, y = [], []
    for s in sorted(R[run]):
        v = R[run][s].get(name, "")
        if v not in ("", None):
            x.append(s), y.append(float(v))
    return np.array(x[::every]), np.array(y[::every])


def final(run, name="val_macro_em"):
    return float(R[run][500][name])


def band(runs, name="val_macro_em"):
    """Mean / min / max across seeds on the shared eval grid (val evals align on EVAL_STEPS)."""
    steps, mean, lo, hi = [], [], [], []
    for s in EVAL_STEPS:
        vs = [float(R[run][s][name]) for run in runs if s in R[run] and R[run][s].get(name) not in ("", None)]
        if not vs:
            continue
        steps.append(s); mean.append(np.mean(vs)); lo.append(min(vs)); hi.append(max(vs))
    return np.array(steps), np.array(mean), np.array(lo), np.array(hi)


def plot_band(ax, runs, ckey, label, ls="-", marker="o", fill=True):
    x, m, lo, hi = band(runs)
    c = C[ckey]
    ax.plot(x, m, color=c, ls=ls, lw=2.4, marker=marker, markersize=4, label=label)
    if fill and len(runs) > 1:
        ax.fill_between(x, lo, hi, color=c, alpha=0.16, linewidth=0)


# ============================ finals.csv (bar-chart source) ============================
with open(os.path.join(FIGS, "finals.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["arm", "label", "family", "horizon", "seed_run",
                "final_val2048_macro_em", "final_clip_ratio", "final_avg_turns"])
    for arm, (lab, ck, seeds, fam) in ARMS.items():
        for run in seeds:
            w.writerow([arm, lab, fam, 4, run, f"{final(run):.4f}",
                        f"{final(run, 'clip_ratio'):.4f}", f"{final(run, 'avg_turns'):.3f}"])
    for arm, (lab, ck, seeds) in HORIZON.items():
        for run in seeds:
            w.writerow([arm, lab, "8-turn", 8, run, f"{final(run):.4f}",
                        f"{final(run, 'clip_ratio'):.4f}", f"{final(run, 'avg_turns'):.3f}"])


def arm_final(seeds):
    v = [final(r) for r in seeds]
    return float(np.mean(v)), min(v), max(v)


# ============================ Fig 1: learning curves (3 panels) ============================
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), sharey=True)

ax = axes[0]  # RQ1 / RQ2 — granularity & outcome-only
for arm in ["gigpo", "token_grpo", "hcapo", "b0", "token_ppo"]:
    lab, ck, seeds, _ = ARMS[arm]
    plot_band(ax, seeds, ck, lab + ("" if len(seeds) == 1 else f" ({len(seeds)}s)"))
ax.axhline(BASE_EM, color="black", ls=":", lw=1.4)
ax.text(8, BASE_EM + 0.006, "untrained 0.225", fontsize=9.5)
ax.set_title("(a) Granularity & outcome-only CA (RQ1/RQ2)", fontsize=12)
ax.set_ylabel("val-2048 macro-EM"); ax.set_xlabel("training step")
ax.legend(loc="lower right", fontsize=9.5)

ax = axes[1]  # RQ3 triad
for arm in ["b1_shuffle", "b1", "b0"]:
    lab, ck, seeds, _ = ARMS[arm]
    plot_band(ax, seeds, ck, {"b0": "B0 (sparse)", "b1": "B1 (privileged)",
                              "b1_shuffle": "B1-shuffle (control)"}[arm])
ax.axhline(BASE_EM, color="black", ls=":", lw=1.4)
ax.set_title("(b) Progress-supervision triad (RQ3), 3 seeds/arm", fontsize=12)
ax.set_xlabel("training step"); ax.legend(loc="lower right", fontsize=9.5)

ax = axes[2]  # RQ4 horizon
plot_band(ax, ARMS["b0"][2], "b0", "B0 4-turn (3s)", ls="--", fill=False)
plot_band(ax, HORIZON["b0_8t"][2], "b0", "B0 8-turn", ls="-")
plot_band(ax, ARMS["b1"][2], "b1", "B1 4-turn (3s)", ls="--", fill=False)
plot_band(ax, HORIZON["b1_8t"][2], "b1", "B1 8-turn", ls="-")
ax.axhline(BASE_EM, color="black", ls=":", lw=1.4)
ax.set_title("(c) Horizon stress: 4-turn vs 8-turn (RQ4)", fontsize=12)
ax.set_xlabel("training step"); ax.legend(loc="lower right", fontsize=9.5)
finalize_figure(fig, os.path.join(FIGS, "fig_learning_curves.png"))

# ============================ Fig 2: final val-EM bars (headline) ============================
fig, ax = plt.subplots(figsize=(9.2, 4.6))
order = sorted(ARMS, key=lambda a: -arm_final(ARMS[a][2])[0])
xs = np.arange(len(order))
means = [arm_final(ARMS[a][2])[0] for a in order]
los = [arm_final(ARMS[a][2])[1] for a in order]
his = [arm_final(ARMS[a][2])[2] for a in order]
cols = [C[ARMS[a][1]] for a in order]
yerr = np.array([[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]])
bars = ax.bar(xs, means, color=cols, edgecolor="black", linewidth=1.2, width=0.66,
              yerr=yerr, capsize=4, error_kw=dict(lw=1.4, ecolor="#333"))
# per-seed dots
for i, a in enumerate(order):
    for run in ARMS[a][2]:
        ax.plot(xs[i], final(run), "o", color="white", markeredgecolor="#222",
                markeredgewidth=1.0, markersize=4.5, zorder=5)
for i, m in enumerate(means):
    ax.annotate(f"{m:.3f}", (xs[i], his[i]), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=10)
# flag HCAPO collapse (well above its value label, clear of the legend which sits over x>=4)
hc = order.index("hcapo")
ax.text(xs[hc], his[hc] + 0.035, "clip→0.75\ncollapse", ha="center", va="bottom",
        fontsize=8.5, color=C["hcapo"], fontweight="bold")
ax.set_xticks(xs)
ax.set_xticklabels([ARMS[a][0] for a in order], fontsize=10.5, rotation=18, ha="right")
ax.set_ylabel("val-2048 macro-EM (FINAL ckpt @500)")
ax.axhline(BASE_EM, color="black", ls=":", lw=1.4)
ax.text(len(order) - 0.5, BASE_EM + 0.006, "untrained 0.225", ha="right", fontsize=9.5)
ax.set_ylim(0, 0.50)
ax.set_title("Fixed-budget final val macro-EM — 4-turn arms (dots = seeds, whiskers = min–max)",
             fontsize=11.5)
# family legend
fam_handles = [
    Line2D([0], [0], color=C["gigpo"], lw=8, label="critic-free group-relative"),
    Line2D([0], [0], color=C["hcapo"], lw=8, label="critic-free per-turn amplifier"),
    Line2D([0], [0], color=C["b1"], lw=8, label="critic + step reward"),
    Line2D([0], [0], color=C["b0"], lw=8, label="critic (PPO)"),
]
ax.legend(handles=fam_handles, loc="upper right", fontsize=8.5, framealpha=0.92,
          borderpad=0.5, labelspacing=0.35)
finalize_figure(fig, os.path.join(FIGS, "fig_final_bars.png"))

# ============================ Fig 3: RQ3 density-vs-content ============================
fig, ax = plt.subplots(figsize=(6.6, 4.5))
triad = ["b0", "b1", "b1_shuffle"]
labs = ["B0\n(sparse outcome)", "B1\n(privileged content)", "B1-shuffle\n(placebo: shuffled timing)"]
xs = np.arange(3)
means = [arm_final(ARMS[a][2])[0] for a in triad]
los = [arm_final(ARMS[a][2])[1] for a in triad]
his = [arm_final(ARMS[a][2])[2] for a in triad]
cols = [C[ARMS[a][1]] for a in triad]
yerr = np.array([[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]])
ax.bar(xs, means, color=cols, edgecolor="black", linewidth=1.2, width=0.6,
       yerr=yerr, capsize=5, error_kw=dict(lw=1.5, ecolor="#333"))
for i, a in enumerate(triad):
    for run in ARMS[a][2]:
        ax.plot(xs[i], final(run), "o", color="white", markeredgecolor="#222",
                markeredgewidth=1.1, markersize=6, zorder=5)
    ax.annotate(f"{means[i]:.3f}", (xs[i], his[i]), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=10)
ax.set_ylabel("val-2048 macro-EM (FINAL @500)")
ax.axhline(BASE_EM, color="black", ls=":", lw=1.4)
ax.text(2.4, BASE_EM + 0.005, "untrained", ha="right", fontsize=9.5)
ax.set_ylim(0, 0.42)
ax.set_title("RQ3: shuffle $\\geq$ B1 $>$ B0 $\\Rightarrow$ density, not content\n(3 seeds/arm)", fontsize=11.5)
# annotate the interpretation arrow
ax.annotate("", xy=(2, means[2]), xytext=(1, means[1]),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.3))
finalize_figure(fig, os.path.join(FIGS, "fig_rq3_density.png"))

# ============================ Fig 4: RQ4 horizon ============================
fig, ax = plt.subplots(figsize=(6.8, 4.5))
groups = [("turn-PPO (B0)", "b0", ARMS["b0"][2], HORIZON["b0_8t"][2]),
          ("B1", "b1", ARMS["b1"][2], HORIZON["b1_8t"][2])]
xs = np.arange(2); w = 0.36
for off, htxt, seeds_getter in [(-w / 2, "4-turn", lambda g: g[2]), (w / 2, "8-turn", lambda g: g[3])]:
    vals = [arm_final(seeds_getter(g))[0] for g in groups]
    cols = [C[g[1]] for g in groups]
    alpha = 0.55 if htxt == "4-turn" else 1.0
    bb = ax.bar(xs + off, vals, width=w, color=cols, edgecolor="black", linewidth=1.2,
                alpha=alpha, label=htxt, hatch="" if htxt == "8-turn" else "//")
    for b, v in zip(bb, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=10)
ax.set_xticks(xs); ax.set_xticklabels(["turn-PPO (B0)", "B1 (privileged content)"], fontsize=10.5)
ax.set_ylabel("val-2048 macro-EM (FINAL @500)")
ax.axhline(BASE_EM, color="black", ls=":", lw=1.4)
ax.set_ylim(0, 0.44)
ax.set_title("RQ4: horizon lifts B0 (0.33$\\to$0.37); content still no help\n(8-turn B1 0.343 $<$ 8-turn B0 0.368)", fontsize=11)
ax.legend(loc="upper left", fontsize=10, title="horizon")
finalize_figure(fig, os.path.join(FIGS, "fig_rq4_horizon.png"))

# ============================ Fig 5: truncation tripwire ============================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
ax = axes[0]
TRIP = [("hcapo_s0", "HCAPO", "hcapo", "-"),
        ("token_ppo_s0", "token-PPO", "token_ppo", "-"),
        ("b1_s1", "B1 (s1)", "b1", "-"),
        ("turn_ppo_b0_s2", "turn-PPO/B0 (s2)", "b0", "-"),
        ("b1_shuffle_s1", "B1-shuffle (s1)", "b1_shuffle", "-"),
        ("gigpo_s0", "GiGPO", "gigpo", "-"),
        ("token_grpo_s1", "token-GRPO", "token_grpo", "-")]
for run, lab, ck, ls in TRIP:
    x, y = col(run, "clip_ratio", every=2)
    ax.plot(x, y, color=C[ck], ls=ls, lw=2.1, label=lab)
ax.axhline(0.001, color="black", ls=":", lw=1.2)
ax.text(500, 0.012, "gate <0.1%", ha="right", fontsize=9)
ax.set_xlabel("training step"); ax.set_ylabel("response-length clip ratio")
ax.set_title("(a) Truncation tripwire: aggressive per-turn credit blows up", fontsize=11.5)
ax.legend(loc="upper left", fontsize=9, ncol=1)

ax = axes[1]  # HCAPO: clip runs away while EM plateaus
x1, y1 = col("hcapo_s0", "clip_ratio", every=1)
x2, y2 = col("hcapo_s0", "val_macro_em")
ln1, = ax.plot(x1, y1, color=C["hcapo"], lw=2.4, label="clip ratio")
ax.set_xlabel("training step"); ax.set_ylabel("response-length clip ratio", color=C["hcapo"])
ax.tick_params(axis="y", labelcolor=C["hcapo"]); ax.set_ylim(0, 0.85)
ax2 = ax.twinx()
ln2, = ax2.plot(x2, y2, color="#333", lw=2.2, marker="o", markersize=4, label="val macro-EM")
ax2.set_ylabel("val-2048 macro-EM"); ax2.set_ylim(0.2, 0.42)
ax2.axhline(BASE_EM, color="black", ls=":", lw=1.2)
ax.set_title("(b) HCAPO: clip$\\to$0.75 length runaway, EM plateaus $\\approx$0.35", fontsize=11.5)
ax.legend(handles=[ln1, ln2], loc="center right", fontsize=9.5)
finalize_figure(fig, os.path.join(FIGS, "fig_tripwire.png"))

# ============================ Fig 6: training reward + avg turns ============================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
REW = ["gigpo", "token_grpo", "hcapo", "b0", "b1", "b1_shuffle", "token_ppo"]
ax = axes[0]
for arm in REW:
    lab, ck, seeds, _ = ARMS[arm]
    x, y = col(seeds[0], "reward_mean", every=2)
    ax.plot(x, y, color=C[ck], lw=2.0, alpha=0.9, label=lab)
ax.set_xlabel("training step"); ax.set_ylabel("mean episode reward (train, s0)")
ax.set_title("(a) Training reward", fontsize=12)
ax.legend(loc="lower right", fontsize=8.8, ncol=2)

ax = axes[1]
for arm in REW:
    lab, ck, seeds, _ = ARMS[arm]
    x, y = col(seeds[0], "avg_turns", every=2)
    ax.plot(x, y, color=C[ck], lw=2.0, alpha=0.9, label=lab)
# 8-turn arms dashed
for arm in ["b0_8t", "b1_8t"]:
    lab, ck, seeds = HORIZON[arm]
    x, y = col(seeds[0], "avg_turns", every=2)
    ax.plot(x, y, color=C[ck], lw=2.0, ls="--", alpha=0.9, label=lab)
ax.set_xlabel("training step"); ax.set_ylabel("avg turns / trajectory (train)")
ax.set_title("(b) Average turns (dashed = 8-turn horizon)", fontsize=12)
ax.legend(loc="upper right", fontsize=8.5, ncol=2)
finalize_figure(fig, os.path.join(FIGS, "fig_reward_turns.png"))

print("figures + finals.csv written to", FIGS)
# quick console summary of the finals that drive the bars
print("\nFINAL @500 val-2048 macro-EM (mean [min,max]):")
for a in sorted(ARMS, key=lambda a: -arm_final(ARMS[a][2])[0]):
    m, lo, hi = arm_final(ARMS[a][2])
    print(f"  {ARMS[a][0]:16s} {m:.3f}  [{lo:.3f},{hi:.3f}]  n={len(ARMS[a][2])}")
for a in HORIZON:
    m, lo, hi = arm_final(HORIZON[a][2])
    print(f"  {HORIZON[a][0]:16s} {m:.3f}")
