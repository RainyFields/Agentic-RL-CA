# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Wave-1 RQ3 interim readout: B0 (turn_ppo_b0) vs B1 vs B1-shuffle from launch logs.

Pre-declared comparison form (docs/plan.md §Pre-registered decision rules): per-seed
deltas with seed ranges — NO pooling across seeds into a single mean, no p-values.
This interim readout uses val-core/macro_em at matched val steps plus val-AUC-so-far
(trapezoid over the common step grid); the HEADLINE comparison remains the fixed-budget
FINAL checkpoint + full-budget val-AUC (this script is re-run then). shuffle_active_frac
is printed next to every B1-vs-shuffle number (review round 3, point 8).

Usage: python scripts/rq3_readout.py --out outputs/readouts/<label> \
           [--runs name=logpath ...]   (defaults = the wave-1 fleet)
"""
import argparse
import json
import os
import re

DEFAULT_LOGS = {
    "b0_s0": "20260714_092728_arlca-turn-ppo-b0-s0.log",
    "b0_s1": "20260714_083039_arlca-turn-ppo-b0-s1.log",
    "b1_s0": "20260714_083044_arlca-b1-s0.log",
    "b1_s1": "20260714_083049_arlca-b1-s1.log",
    "b1sh_s0": "20260714_083054_arlca-b1-shuffle-s0.log",
    "b1sh_s1": "20260714_083059_arlca-b1-shuffle-s1.log",
}
LAUNCH_DIR = os.path.expanduser("~/xiaoxuan/worker_logs/launches")

STEP_RE = re.compile(r"step:(\d+) - ")
VAL_RE = re.compile(r"val-core/macro_em:([0-9.]+)")
SHUF_RE = re.compile(r"shuffle/active_frac:([0-9.]+)")


def parse_log(path):
    """Return (val_series {step: em}, shuffle_series [(step, frac)]). Later occurrences
    of a step win (crash-resume replays a checkpoint window; the rerun is authoritative)."""
    vals, shuf = {}, {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m_step = STEP_RE.search(line)
            if not m_step:
                continue
            step = int(m_step.group(1))
            m_val = VAL_RE.search(line)
            if m_val:
                vals[step] = float(m_val.group(1))
            m_sh = SHUF_RE.search(line)
            if m_sh:
                shuf[step] = float(m_sh.group(1))
    return vals, sorted(shuf.items())


def auc(vals, grid):
    """Trapezoid val-AUC over the matched step grid, normalized by span."""
    xs = [s for s in grid if s in vals]
    if len(xs) < 2:
        return None
    area = sum((vals[xs[i + 1]] + vals[xs[i]]) / 2 * (xs[i + 1] - xs[i])
               for i in range(len(xs) - 1))
    return area / (xs[-1] - xs[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="*", default=None, help="name=logpath overrides")
    args = ap.parse_args()

    runs = {n: os.path.join(LAUNCH_DIR, p) for n, p in DEFAULT_LOGS.items()}
    if args.runs:
        runs.update(dict(r.split("=", 1) for r in args.runs))

    data, missing = {}, []
    for name, path in runs.items():
        if not os.path.exists(path):
            missing.append(name)
            continue
        vals, shuf = parse_log(path)
        if not vals:
            missing.append(name)
            continue
        data[name] = {"vals": vals, "shuffle": shuf}

    # common val grid across present runs
    grids = [set(d["vals"]) for d in data.values()]
    common = sorted(set.intersection(*grids)) if grids else []
    top = common[-1] if common else None

    lines = ["# Wave-1 RQ3 interim readout", ""]
    if missing:
        lines.append(f"**PARTIAL — missing/queued runs: {', '.join(sorted(missing))}** "
                     "(per-seed rules still apply to present seeds; no pooling)")
        lines.append("")
    lines.append(f"Common val steps: {common}   (deltas at step {top})")
    lines.append("")
    lines.append("| run | " + " | ".join(f"s{s}" for s in common) + " | AUC-so-far |")
    lines.append("|---" * (len(common) + 2) + "|")
    for name in sorted(data):
        d = data[name]
        row = [f"{d['vals'].get(s, float('nan')):.3f}" for s in common]
        a = auc(d["vals"], common)
        auc_cell = f"{a:.3f}" if a is not None else "-"
        lines.append(f"| {name} | " + " | ".join(row) + f" | {auc_cell} |")
    lines.append("")

    # per-seed deltas at the top common step + AUC deltas
    deltas = {}
    for seed in ("s0", "s1"):
        b0, b1, sh = (data.get(f"b0_{seed}"), data.get(f"b1_{seed}"),
                      data.get(f"b1sh_{seed}"))
        if top is None:
            continue
        row = {}
        if b0 and b1:
            row["b1_minus_b0_em"] = b1["vals"][top] - b0["vals"][top]
            a1, a0 = auc(b1["vals"], common), auc(b0["vals"], common)
            row["b1_minus_b0_auc"] = (a1 - a0) if a1 is not None and a0 is not None else None
        if sh and b1:
            row["b1_minus_shuffle_em"] = b1["vals"][top] - sh["vals"][top]
        if sh and b0:
            row["shuffle_minus_b0_em"] = sh["vals"][top] - b0["vals"][top]
        if sh:
            row["shuffle_active_frac_latest"] = sh["shuffle"][-1][1] if sh["shuffle"] else None
        if row:
            deltas[seed] = row

    lines.append(f"## Per-seed deltas at step {top} (pre-declared form; seed range = the two rows)")
    lines.append("")
    for seed, row in deltas.items():
        lines.append(f"- **{seed}**: " + ", ".join(
            f"{k}={v:+.3f}" if isinstance(v, float) and k != "shuffle_active_frac_latest"
            else f"{k}={v}" for k, v in row.items()))
    lines.append("")
    lines.append("Interpretation rules (plan): B1 > shuffle ~= B0 => timing/content matters; "
                 "B1 ~= shuffle > B0 => density effect; all ~= => no help at this horizon. "
                 "shuffle_active_frac MUST accompany any B1-vs-shuffle claim.")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "readout.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "data.json"), "w") as fh:
        json.dump({"series": {n: d["vals"] for n, d in data.items()},
                   "shuffle": {n: d["shuffle"] for n, d in data.items()},
                   "common_steps": common, "deltas": deltas,
                   "missing": missing}, fh, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
