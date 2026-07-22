#!/usr/bin/env python3
"""ScienceWorld structure audit (Wave-0 M5).

Enumerates all 30 tasks: variation splits, task descriptions, gold-path
replay on sampled train variations with per-step score/reward traces, so we
can fix (a) oracle subgoal-reward semantics, (b) per-task-family turn caps,
(c) horizon distributions, before any training touches this rung.

Outputs (under --out):
  task_map.csv            one row per task: splits, gold-path stats
  gold_traces.jsonl       one row per replayed (task, variation) with the
                          full per-step trace (action, reward, score, done)
  audit_summary.md        human-readable summary tables
Run: python audit_structure.py --out <dir> [--vars-per-task 3] [--step-limit 200]
"""

import argparse
import json
import random
import time
from pathlib import Path

from scienceworld import ScienceWorldEnv


def replay_gold(env, task, var_idx, step_limit, simplification):
    env.load(task, var_idx, simplification, generateGoldPath=True)
    gold = env.get_gold_action_sequence()
    desc = env.get_task_description()
    obs, info = env.reset()
    trace = []
    score_prev = info.get("score", 0)
    for i, action in enumerate(gold):
        obs, reward, done, info = env.step(action)
        score = info.get("score", 0)
        trace.append({
            "t": i,
            "action": action,
            "reward": reward,
            "score": score,
            "score_delta": score - score_prev,
            "done": bool(done),
            "obs_chars": len(obs),
            "obs_head": obs[:200],
        })
        score_prev = score
        if done:
            break
    return {
        "task": task,
        "variation": var_idx,
        "task_description": desc,
        "gold_len": len(gold),
        "steps_replayed": len(trace),
        "final_score": score_prev,
        "done": bool(trace and trace[-1]["done"]),
        "n_subgoal_firings": sum(1 for s in trace if s["score_delta"] > 0),
        "firing_steps": [s["t"] for s in trace if s["score_delta"] > 0],
        "firing_magnitudes": [s["score_delta"] for s in trace if s["score_delta"] > 0],
        "negative_deltas": [(s["t"], s["score_delta"]) for s in trace if s["score_delta"] < 0],
        "trace": trace,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vars-per-task", type=int, default=3)
    ap.add_argument("--step-limit", type=int, default=200)
    ap.add_argument("--simplification", default="easy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    env = ScienceWorldEnv("", envStepLimit=args.step_limit)
    tasks = env.get_task_names()

    rows = []
    traces_f = open(out / "gold_traces.jsonl", "w")
    for task in tasks:
        t0 = time.time()
        env.load(task, 0, args.simplification)
        train_v = env.get_variations_train()
        dev_v = env.get_variations_dev()
        test_v = env.get_variations_test()
        max_v = env.get_max_variations(task)

        sample = rng.sample(train_v, min(args.vars_per_task, len(train_v)))
        replays = []
        for v in sample:
            try:
                r = replay_gold(env, task, v, args.step_limit, args.simplification)
            except Exception as e:  # noqa: BLE001 - audit must survive bad variations
                r = {"task": task, "variation": v, "error": repr(e)}
            replays.append(r)
            traces_f.write(json.dumps(r) + "\n")
            traces_f.flush()

        ok = [r for r in replays if "error" not in r]
        rows.append({
            "task": task,
            "max_variations": max_v,
            "n_train": len(train_v),
            "n_dev": len(dev_v),
            "n_test": len(test_v),
            "gold_len_min": min((r["gold_len"] for r in ok), default=None),
            "gold_len_max": max((r["gold_len"] for r in ok), default=None),
            "gold_lens": [r["gold_len"] for r in ok],
            "final_scores": [r["final_score"] for r in ok],
            "done_all": all(r["done"] for r in ok) if ok else None,
            "subgoal_firings": [r["n_subgoal_firings"] for r in ok],
            "n_errors": len(replays) - len(ok),
            "secs": round(time.time() - t0, 1),
        })
        print(f"[{task}] train/dev/test={len(train_v)}/{len(dev_v)}/{len(test_v)} "
              f"gold_lens={[r.get('gold_len') for r in replays]} "
              f"final_scores={[r.get('final_score') for r in replays]} "
              f"({rows[-1]['secs']}s)", flush=True)

    traces_f.close()

    import csv
    with open(out / "task_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(out / "audit_summary.md", "w") as f:
        f.write("# ScienceWorld structure audit — raw pass\n\n")
        f.write(f"simplification={args.simplification} step_limit={args.step_limit} "
                f"vars/task={args.vars_per_task} seed={args.seed}\n\n")
        f.write("| task | max_var | train/dev/test | gold_len (sampled) | final_scores | subgoal firings | errors |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['task']} | {r['max_variations']} | "
                    f"{r['n_train']}/{r['n_dev']}/{r['n_test']} | {r['gold_lens']} | "
                    f"{r['final_scores']} | {r['subgoal_firings']} | {r['n_errors']} |\n")
    print(f"\nWrote {out}/task_map.csv, gold_traces.jsonl, audit_summary.md")


if __name__ == "__main__":
    main()
