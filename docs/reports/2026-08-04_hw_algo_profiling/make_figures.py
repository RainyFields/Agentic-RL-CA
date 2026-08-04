#!/usr/bin/env python3
"""Hardware/algorithm profiling figures + LaTeX table fragments (house style).

Reads assets/timings.json (produced by mine_timings.py) and emits:
  assets/fig_phase_breakdown.{png,pdf}   stacked phase composition, steps 1-N matched
  assets/fig_h100_fullrun.{png,pdf}      75-step H100 phase evolution, both algorithms
  assets/fig_speedup.{png,pdf}           per-phase H100->B200 speedup bars
  assets/tables.tex                      all numeric tables, \input by the report
  assets/computed_tables.json            the same numbers, machine-readable

Rerun after re-mining: python make_figures.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/.claude/skills/scientific-figure-making/assets"))
import numpy as np
from figstyle import apply_publication_style, PALETTE, finalize_figure
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")
T = json.load(open(os.path.join(A, "timings.json")))

# Phase display order and colors (composition figures)
PHASES = ["gen", "old_log_prob", "ref", "values", "update_critic", "update_actor", "other"]
PHASE_LABEL = {"gen": "rollout gen", "old_log_prob": "old logprob", "ref": "ref (KL)",
               "values": "critic fwd", "update_critic": "critic update",
               "update_actor": "actor update", "other": "reward/adv/other"}
PHASE_COLOR = {"gen": PALETTE["blue_main"], "old_log_prob": PALETTE["blue_secondary"],
               "ref": PALETTE["teal"], "values": PALETTE["green_2"],
               "update_critic": PALETTE["green_3"], "update_actor": PALETTE["red_strong"],
               "other": PALETTE["neutral"]}
TIMED = ["gen", "reward", "old_log_prob", "ref", "values", "adv",
         "update_critic", "update_actor", "save_checkpoint", "testing"]


def phase_means(steps_dict, step_ids):
    """Mean seconds per phase over the given steps; 'other' = step - itemized phases."""
    n = len(step_ids)
    out = {}
    for k in TIMED + ["step"]:
        out[k] = sum(steps_dict[s].get(k, 0.0) for s in step_ids) / n
    itemized = ["gen", "old_log_prob", "ref", "values", "update_critic", "update_actor"]
    out["other"] = out["step"] - sum(out[k] for k in itemized)
    out["tokens"] = sum(steps_dict[s].get("total_num_tokens", 0.0) for s in step_ids) / n
    return out


def matched_ids(name, n):
    have = sorted(T[name], key=int)
    return have[:n]


# How many matched steps we can use per algorithm = what B200 produced
N = {alg: len(T.get(f"{alg}_b200", {})) for alg in ("turn_ppo", "grpo")}
present = {alg: n for alg, n in N.items() if n > 0}

apply_publication_style(font_size=14)
computed = {}

# ---------------------------------------------------------------- fig 1: composition
if present:
    fig, axes = plt.subplots(1, len(present), figsize=(6.0 * len(present), 4.8), squeeze=False)
    for ai, (alg, n) in enumerate(present.items()):
        ax = axes[0][ai]
        cols, means = [], []
        for hw in ("h100", "b200"):
            ids = matched_ids(f"{alg}_{hw}", n)
            m = phase_means(T[f"{alg}_{hw}"], ids)
            cols.append(f"{hw.upper()}\n(s1-{n})")
            means.append(m)
        bottoms = np.zeros(2)
        for ph in PHASES:
            vals = np.array([m.get(ph, 0.0) for m in means])
            if vals.sum() <= 0:
                continue
            ax.bar([0, 1], vals, 0.55, bottom=bottoms, label=PHASE_LABEL[ph],
                   color=PHASE_COLOR[ph], edgecolor="black", linewidth=1.0)
            bottoms += vals
        for xi, m in enumerate(means):
            ax.annotate(f"{m['step']:.0f}s", (xi, bottoms[xi]), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(cols)
        ax.set_ylabel("seconds / step" if ai == 0 else "")
        ax.set_title({"turn_ppo": "turn-PPO", "grpo": "GRPO"}[alg], fontsize=15)
        computed[f"composition_{alg}"] = {c.replace("\n", " "): m for c, m in zip(cols, means)}
    axes[0][-1].legend(loc="upper right", fontsize=10.5)
    finalize_figure(fig, os.path.join(A, "fig_phase_breakdown.png"))

# ---------------------------------------------------------------- fig 2: H100 full run
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
for ai, alg in enumerate(("grpo", "turn_ppo")):
    ax = axes[ai]
    d = T[f"{alg}_h100"]
    xs = sorted(int(s) for s in d)
    for key, color, lab in (("step", "black", "total step"),
                            ("gen", PHASE_COLOR["gen"], "rollout gen"),
                            ("update_actor", PHASE_COLOR["update_actor"], "actor update"),
                            ("update_critic", PHASE_COLOR["update_critic"], "critic update")):
        ys = [d[str(s)].get(key, 0.0) for s in xs]
        if max(ys) <= 0:
            continue
        ax.plot(xs, ys, lw=2.4 if key != "step" else 3.0, color=color, label=lab,
                ls="-" if key != "step" else "--")
    ax.set_xlabel("training step")
    ax.set_ylabel("seconds" if ai == 0 else "")
    ax.set_title({"grpo": "GRPO (75 steps, 1×H100 node)",
                  "turn_ppo": "turn-PPO (75 steps, 1×H100 node)"}[alg], fontsize=14)
    ax.legend(fontsize=10.5)
finalize_figure(fig, os.path.join(A, "fig_h100_fullrun.png"))

# ---------------------------------------------------------------- fig 3: speedups
if present:
    fig, axes = plt.subplots(1, len(present), figsize=(6.2 * len(present), 4.6), squeeze=False)
    for ai, (alg, n) in enumerate(present.items()):
        ax = axes[0][ai]
        ids_h = matched_ids(f"{alg}_h100", n)
        ids_b = matched_ids(f"{alg}_b200", n)
        mh = phase_means(T[f"{alg}_h100"], ids_h)
        mb = phase_means(T[f"{alg}_b200"], ids_b)
        cats, ratios, ratios_tok = [], [], []
        for ph in ("gen", "old_log_prob", "values", "update_critic", "update_actor", "step"):
            if mh.get(ph, 0) <= 0 or mb.get(ph, 0) <= 0:
                continue
            cats.append("TOTAL" if ph == "step" else PHASE_LABEL.get(ph, ph))
            ratios.append(mh[ph] / mb[ph])
            ratios_tok.append((mh[ph] / mh["tokens"]) / (mb[ph] / mb["tokens"]))
        x = np.arange(len(cats))
        b1 = ax.bar(x - 0.19, ratios, 0.38, color=PALETTE["blue_main"],
                    edgecolor="black", linewidth=1.1, label="wall-clock ratio")
        b2 = ax.bar(x + 0.19, ratios_tok, 0.38, color=PALETTE["green_3"],
                    edgecolor="black", linewidth=1.1, label="per-token ratio")
        for bars in (b1, b2):
            for b in bars:
                ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
        ax.axhline(1.0, color="gray", lw=1.2, ls=":")
        ax.set_xticks(x); ax.set_xticklabels(cats, rotation=18, ha="right", fontsize=11)
        ax.set_ylabel("H100 time / B200 time" if ai == 0 else "")
        ax.set_title({"turn_ppo": "turn-PPO", "grpo": "GRPO"}[alg], fontsize=15)
        computed[f"speedup_{alg}"] = dict(zip(cats, zip(ratios, ratios_tok)))
    axes[0][0].legend(fontsize=11)
    finalize_figure(fig, os.path.join(A, "fig_speedup.png"))

# ---------------------------------------------------------------- tables.tex
def tex_escape(s):
    return s.replace("_", r"\_")


L = []
# full-run H100 table
L.append("% ---- auto-generated by make_figures.py ----")
for alg, disp in (("grpo", "GRPO"), ("turn_ppo", "turn-PPO")):
    d = T[f"{alg}_h100"]
    ids = sorted(d, key=int)
    tot = {k: sum(d[s].get(k, 0.0) for s in ids) for k in TIMED + ["step"]}
    toks = sum(d[s].get("total_num_tokens", 0) for s in ids)
    other = tot["step"] - sum(tot[k] for k in ["gen", "old_log_prob", "ref", "values",
                                               "update_critic", "update_actor"])
    L.append(f"\\newcommand{{\\fullrun{alg.replace('_','')}}}{{%")
    rows = [("rollout generation", tot["gen"]), ("actor update", tot["update_actor"])]
    if tot["update_critic"] > 0:
        rows += [("critic update", tot["update_critic"]), ("critic fwd (values)", tot["values"])]
    rows += [("old logprob", tot["old_log_prob"])]
    if tot["ref"] > 0:
        rows += [("ref policy (KL)", tot["ref"])]
    rows += [("val + ckpt + reward/adv/other",
              tot["testing"] + tot["save_checkpoint"] + other + tot["reward"] + tot["adv"])]
    for name, v in rows:
        L.append(f"{name} & {v/3600:.1f}\\,h & {100*v/tot['step']:.1f}\\% \\\\")
    L.append(f"\\midrule TOTAL (in-step) & {tot['step']/3600:.1f}\\,h & 100\\% \\\\")
    L.append("}")
    computed[f"fullrun_{alg}"] = {"rows": rows, "total_s": tot["step"], "tokens": toks}

# matched comparison table (per algorithm) + full-run extrapolation
for alg in ("turn_ppo", "grpo"):
    macro = alg.replace("_", "")
    if alg not in present:
        L.append(f"\\newcommand{{\\matched{macro}}}{{\\multicolumn{{7}}{{c}}{{(B200 run in progress)}} \\\\}}")
        L.append(f"\\newcommand{{\\extrap{macro}}}{{(pending)}}")
        continue
    n = present[alg]
    mh = phase_means(T[f"{alg}_h100"], matched_ids(f"{alg}_h100", n))
    mb = phase_means(T[f"{alg}_b200"], matched_ids(f"{alg}_b200", n))
    L.append(f"\\newcommand{{\\matched{macro}}}{{%")
    tok_ratio = {}
    for ph in ("gen", "old_log_prob", "ref", "values", "update_critic", "update_actor", "step"):
        if mh.get(ph, 0) <= 0:
            continue
        name = "TOTAL / step" if ph == "step" else PHASE_LABEL.get(ph, ph)
        rt = mh[ph] / mb[ph] if mb.get(ph, 0) > 0 else float("nan")
        rtok = (mh[ph] / mh["tokens"]) / (mb[ph] / mb["tokens"]) if mb.get(ph, 0) > 0 else float("nan")
        tok_ratio[ph] = rtok
        ph_h = 100 * mh[ph] / mh["step"]
        ph_b = 100 * mb[ph] / mb["step"]
        pct_h = "100\\%" if ph == "step" else f"{ph_h:.1f}\\%"
        pct_b = "100\\%" if ph == "step" else f"{ph_b:.1f}\\%"
        L.append(f"{name} & {mh[ph]:.0f} & {pct_h} & {mb[ph]:.0f} & {pct_b} & "
                 f"{rt:.2f}$\\times$ & {rtok:.2f}$\\times$ \\\\")
    L.append(f"\\midrule tokens/step & {mh['tokens']/1e6:.1f}M & & {mb['tokens']/1e6:.1f}M & & & \\\\")
    L.append("}")
    # 75-step extrapolation: divide each H100 full-run phase total by its measured
    # per-token speedup (identical token trajectory assumed); unmeasured phases (ref on
    # GRPO shares old_log_prob's ratio; other/val/ckpt kept at 1x, conservative).
    d = T[f"{alg}_h100"]
    ids = sorted(d, key=int)
    tot = {k: sum(d[s].get(k, 0.0) for s in ids) for k in TIMED + ["step"]}
    est = 0.0
    for ph in ("gen", "old_log_prob", "ref", "values", "update_critic", "update_actor"):
        if tot.get(ph, 0) <= 0:
            continue
        r = tok_ratio.get(ph) or tok_ratio.get("old_log_prob") or 1.0
        est += tot[ph] / r
    rest = tot["step"] - sum(tot.get(p, 0) for p in
                             ("gen", "old_log_prob", "ref", "values", "update_critic", "update_actor"))
    est += rest
    L.append(f"\\newcommand{{\\extrap{macro}}}{{{est/3600:.1f}\\,h "
             f"({tot['step']/3600:.1f}\\,h on H100, {tot['step']/est:.2f}$\\times$)}}")
    computed[f"extrap75_{alg}"] = {"h100_h": tot["step"] / 3600, "b200_est_h": est / 3600,
                                   "speedup": tot["step"] / est}

open(os.path.join(A, "tables.tex"), "w").write("\n".join(L) + "\n")
json.dump(computed, open(os.path.join(A, "computed_tables.json"), "w"), indent=2, default=list)
print("figures + tables written; matched steps:", present)
