#!/usr/bin/env python3
"""P8B profiling analysis (2026-07-29) — rerunnable.

Inputs:
  --driver-log  outputs/p8b_profile.log            (step metric lines, all configs)
  --dump-dir    HDFS logs/p8b_profile/             (rollout_<cfg>.jsonl.zst per-turn dumps)
Outputs (to --out):
  timing.json      per-config step timings + partial/* counters (from driver log)
  rollouts.json    per-config trajectory stats (turns histogram, grammar compliance,
                   info-cap hit rate, termination breakdown) from the dumps
Run: python analyze_profiling.py --driver-log ... --dump-dir ... --out assets/
"""
import argparse, collections, io, json, os, re, subprocess, sys

try:
    import zstandard as zstd
except ImportError:
    zstd = None  # falls back to the zstd CLI


STEP_RE = re.compile(r"step:(\d+) - ")
CFG_RE = re.compile(r"==== P8B-PROFILE cfg=(\w+) (start|exit)")
METRIC_RE = re.compile(r"([a-zA-Z0-9_/.]+):(-?[0-9.]+)")

KEEP_METRICS = (
    "timing_s/gen", "timing_s/update_actor", "timing_s/update_critic", "timing_s/old_log_prob",
    "timing_s/ref", "timing_s/values", "timing_s/step", "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb", "perf/throughput", "global_seqlen/mean",
    "episode/length/mean", "episode/length/max", "episode/success_rate",
    "episode/valid_action_ratio", "response_length/mean", "response_length/clip_ratio",
    "prompt_length/mean", "prompt_length/max", "prompt_length/clip_ratio",
    "lengthdiag/n_turns", "actor/entropy_loss", "training/rollout_probs_diff_max",
)


def parse_driver_log(path):
    out = collections.defaultdict(list)
    cur = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = CFG_RE.search(line)
            if m:
                cur = m.group(1) if m.group(2) == "start" else cur
                continue
            if cur is None or "step:" not in line:
                continue
            sm = STEP_RE.search(line)
            if not sm:
                continue
            rec = {"step": int(sm.group(1))}
            for k, v in METRIC_RE.findall(line):
                if k in KEEP_METRICS or k.startswith("partial/") or k.startswith("val"):
                    rec[k] = float(v)
            if len(rec) > 1:
                out[cur].append(rec)
    return dict(out)


def stream_dump(path):
    if zstd is not None:
        dctx = zstd.ZstdDecompressor()
        ctx = open(path, "rb")
        reader = dctx.stream_reader(ctx)
        text = io.TextIOWrapper(reader, errors="replace")
    else:
        proc = subprocess.Popen(["zstd", "-dcq", path], stdout=subprocess.PIPE)
        text = io.TextIOWrapper(proc.stdout, errors="replace")
    for line in text:
        line = line.strip()
        if line:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def analyze_dump(path):
    trajs = {}
    n_rows = 0
    parse_counter = collections.Counter()
    info_cap_hits = 0
    info_nonempty = 0
    resp_cap_hits = 0
    for rec in stream_dump(path):
        n_rows += 1
        t = rec["traj_uid"]
        cur = trajs.setdefault(t, {"max_turn": -1, "done": False, "early": False,
                                   "won": False, "turns": 0})
        cur["turns"] += 1
        ti = rec.get("turn_index", cur["turns"] - 1)
        if ti >= cur["max_turn"]:
            cur["max_turn"] = ti
            cur["done"] = bool(rec.get("env_done"))
            cur["early"] = bool(rec.get("early_stopped"))
            cur["won"] = bool(rec.get("env_won"))
        parse_counter[rec.get("parse_status", "unknown")] += 1
        info = rec.get("information") or ""
        if info:
            info_nonempty += 1
            if len(info) >= 4980:  # 5k cap minus tag overhead tolerance
                info_cap_hits += 1
        if rec.get("response_token_count", 0) >= 1024:
            resp_cap_hits += 1

    hist = collections.Counter(v["turns"] for v in trajs.values())
    term = {"answered_or_env_done": 0, "early_stop": 0, "cycle_or_ctx_cut": 0}
    for v in trajs.values():
        if v["early"]:
            term["early_stop"] += 1
        elif v["done"]:
            term["answered_or_env_done"] += 1
        else:
            term["cycle_or_ctx_cut"] += 1
    return {
        "n_turn_rows": n_rows,
        "n_trajs": len(trajs),
        "success_rate": sum(v["won"] for v in trajs.values()) / max(1, len(trajs)),
        "turns_histogram": dict(sorted(hist.items())),
        "parse_status": dict(parse_counter),
        "info_cap_hit_rate": info_cap_hits / max(1, info_nonempty),
        "response_cap_hit_rate": resp_cap_hits / max(1, n_rows),
        "termination": term,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver-log", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    timing = parse_driver_log(args.driver_log)
    with open(os.path.join(args.out, "timing.json"), "w") as fh:
        json.dump(timing, fh, indent=1)
    print("timing.json:", {k: len(v) for k, v in timing.items()})

    rollouts = {}
    for cfg in ("grpo_vanilla", "ppo_vanilla", "grpo_partial", "ppo_partial"):
        p = os.path.join(args.dump_dir, f"rollout_{cfg}.jsonl.zst")
        if os.path.exists(p):
            print("analyzing", p, "...", flush=True)
            rollouts[cfg] = analyze_dump(p)
    with open(os.path.join(args.out, "rollouts.json"), "w") as fh:
        json.dump(rollouts, fh, indent=1)
    print("rollouts.json written")


if __name__ == "__main__":
    main()
