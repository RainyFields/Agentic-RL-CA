#!/usr/bin/env python3
"""p8bprof_grpo_fast analysis: parse the fast worker log, join with the 2026-07-29
profiling baselines, emit comparison JSON + LaTeX table fragments + figure.
Rerun: python analyze_fast.py [--fast-log ...] [--baseline ../2026-07-29_p8b_profiling/assets/timing.json]
Works with however many steps exist so far (preliminary mode).
"""
import argparse, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEP_RE = re.compile(r"step:(\d+) - ")
METRIC_RE = re.compile(r"([a-zA-Z0-9_/.]+):(-?[0-9.]+)")
ATTEMPT_RE = re.compile(r"P8B-FAST attempt EXP=(\S+) (?:DYNBSZ=(\S+) DYNBSZ_TOK=(\S*) )?")

KEEP = ("timing_s/gen", "timing_s/old_log_prob", "timing_s/ref", "timing_s/update_actor",
        "timing_s/step", "perf/max_memory_allocated_gb", "perf/max_memory_reserved_gb",
        "episode/success_rate", "episode/length/mean", "response_length/mean",
        "actor/entropy_loss", "lengthdiag/n_turns", "global_seqlen/mean")


def parse_fast_log(path):
    steps, attempts = [], []
    with open(path, errors="replace") as fh:
        for line in fh:
            am = ATTEMPT_RE.search(line)
            if am and "exit=" not in line:
                attempts.append(am.group(1) + (f" tok={am.group(3)}" if am.group(3) else ""))
            if "step:" not in line:
                continue
            sm = STEP_RE.search(line)
            if not sm:
                continue
            rec = {"step": int(sm.group(1))}
            for k, v in METRIC_RE.findall(line):
                if k in KEEP or k.startswith("partial/"):
                    rec[k] = float(v)
            if "timing_s/step" in rec:
                steps.append(rec)
    return steps, attempts


def agg(recs, key):
    vals = [r[key] for r in recs if key in r]
    return sum(vals) / len(vals) if vals else float("nan")


def per_traj(recs, key):
    t = sum(r.get(key, 0.0) for r in recs)
    rel = sum(r.get("partial/released_traj", 1280.0) for r in recs)
    return t / max(1.0, rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast-log", default=os.path.expanduser(
        "~/xiaoxuan/Agentic-RL-CA/outputs/p8b_fast.log"))
    ap.add_argument("--baseline", default=os.path.join(
        HERE, "..", "2026-07-29_p8b_profiling", "assets", "timing.json"))
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "assets"), exist_ok=True)

    fast, attempts = parse_fast_log(args.fast_log)
    base = json.load(open(args.baseline))
    gp = [r for r in base["grpo_partial"] if "timing_s/step" in r]
    gv = [r for r in base["grpo_vanilla"] if "timing_s/step" in r]

    def phase_rows(recs):
        return {
            "n_steps": len(recs),
            "gen_s": agg(recs, "timing_s/gen"),
            "old_log_prob_s": agg(recs, "timing_s/old_log_prob"),
            "ref_s": agg(recs, "timing_s/ref"),
            "update_s": agg(recs, "timing_s/update_actor"),
            "step_s": agg(recs, "timing_s/step"),
            "peak_mem_gb": max((r.get("perf/max_memory_allocated_gb", 0) for r in recs), default=0),
            "step_s_per_traj": per_traj(recs, "timing_s/step"),
            "gen_s_per_traj": per_traj(recs, "timing_s/gen"),
            "success": agg(recs, "episode/success_rate"),
            "ep_len": agg(recs, "episode/length/mean"),
            "resp_len": agg(recs, "response_length/mean"),
            "entropy": agg(recs, "actor/entropy_loss"),
        }

    out = {"attempts": attempts,
           "fast": phase_rows(fast),
           "grpo_partial": phase_rows(gp),
           "grpo_vanilla": phase_rows(gv),
           "fast_steps_raw": fast}
    # projections: 150 steps x 1280 trajectories
    for k in ("fast", "grpo_partial", "grpo_vanilla"):
        s = out[k]["step_s_per_traj"]
        out[k]["days_per_arm"] = s * 1280 * 150 / 86400.0
    json.dump(out, open(os.path.join(HERE, "assets", "fast_comparison.json"), "w"), indent=1)

    # LaTeX table fragment
    def row(label, key, fmt="{:.0f}"):
        f, p, v = out["fast"], out["grpo_partial"], out["grpo_vanilla"]
        return (label + " & " + " & ".join(
            (fmt.format(d[key]) if d[key] == d[key] else "--") for d in (v, p, f)) + r"\\")
    lines = [
        row("mean gen / cycle (s)", "gen_s"),
        row("mean old\\_log\\_prob (s)", "old_log_prob_s"),
        row("mean ref (s)", "ref_s"),
        row("mean actor update (s)", "update_s"),
        row("mean total step (s)", "step_s"),
        r"\midrule",
        row("step-s / completed traj", "step_s_per_traj", "{:.2f}"),
        row("gen-s / completed traj", "gen_s_per_traj", "{:.2f}"),
        row("projected days / 150-step arm", "days_per_arm", "{:.1f}"),
        r"\midrule",
        row("peak mem alloc (GB)", "peak_mem_gb", "{:.1f}"),
        row("train success (temp 1)", "success", "{:.3f}"),
        row("episode len (turns)", "ep_len", "{:.2f}"),
        row("response len (tok)", "resp_len", "{:.0f}"),
        row("entropy", "entropy", "{:.2f}"),
    ]
    open(os.path.join(HERE, "assets", "table_fragment.tex"), "w").write("\n".join(lines) + "\n")
    print(json.dumps({k: {kk: round(vv, 2) if isinstance(vv, float) else vv
                          for kk, vv in out[k].items() if kk != "n_steps"}
                      for k in ("fast", "grpo_partial", "grpo_vanilla")}, indent=1)[:1500])
    print("attempts:", attempts)


if __name__ == "__main__":
    main()
