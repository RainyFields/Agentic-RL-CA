#!/usr/bin/env python3
"""Rung-4 zero-shot gate analysis (design doc §gate, three pre-registered outcomes).

Reads the GATE=1 rollout dump (per-turn JSONL) and reports the metrics the gate
decision is keyed on. Run:
  python3 scripts/sciworld/gate_analysis.py outputs/sciworld_gate/prm_rtg/rollout_log.jsonl \
      --out outputs/sciworld_gate/report
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

MAX_RESPONSE = 256
GO_THRESH, NOGO_THRESH = 0.25, 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--out", default="outputs/sciworld_gate/report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = [json.loads(l) for l in open(args.dump)]
    trajs = defaultdict(list)
    for r in rows:
        trajs[r["traj_uid"]].append(r)
    for t in trajs.values():
        t.sort(key=lambda r: r["turn_index"])

    traj_stats, groups = {}, defaultdict(list)
    for tid, t in trajs.items():
        last = t[-1]
        s = {
            "uid": last["uid"], "task": last.get("sw_task", ""),
            "variation": last.get("sw_variation", -1),
            "P_T": last.get("sw_progress", 0.0), "won": last.get("sw_won", False),
            "focus_death": any(r.get("sw_focus_death") for r in t),
            "cap_hit": any(r.get("sw_cap_hit") for r in t),
            "env_crash": any(r.get("sw_env_crash") for r in t),
            "n_turns": len(t),
            "valid_frac": float(np.mean([r["is_action_valid"] for r in t])),
            "clip_frac": float(np.mean([r["response_token_count"] >= MAX_RESPONSE for r in t])),
        }
        traj_stats[tid] = s
        groups[last["uid"]].append(s)

    # gradient-bearing = within-group variance in P_T (the trained channel)
    gb = [1.0 if np.std([s["P_T"] for s in g]) > 1e-9 else 0.0 for g in groups.values()]
    gb_frac = float(np.mean(gb)) if gb else 0.0

    per_task = defaultdict(lambda: defaultdict(list))
    for s in traj_stats.values():
        pt = per_task[s["task"]]
        for k in ("P_T", "won", "focus_death", "cap_hit", "n_turns", "valid_frac", "clip_frac"):
            pt[k].append(float(s[k]))
    task_table = {
        task: {k: round(float(np.mean(v)), 4) for k, v in d.items()} | {"n": len(d["P_T"])}
        for task, d in sorted(per_task.items())
    }
    task_gb = {}
    for uid, g in groups.items():
        task_gb.setdefault(g[0]["task"], []).append(1.0 if np.std([s["P_T"] for s in g]) > 1e-9 else 0.0)
    for task, v in task_gb.items():
        task_table.setdefault(task, {})["gb_frac"] = round(float(np.mean(v)), 4)
    dead = [t for t, d in task_table.items() if d.get("gb_frac", 0) == 0.0 and d.get("P_T", 0) == 0.0]

    verdict = ("GO" if gb_frac >= GO_THRESH else
               "MIDDLE (raise G to 16 + oversample live families, re-measure)" if gb_frac >= NOGO_THRESH else
               "NO-GO (discuss SFT cold start)")

    all_s = list(traj_stats.values())
    report = {
        "n_trajectories": len(all_s), "n_groups": len(groups),
        "gradient_bearing_group_fraction": round(gb_frac, 4),
        "verdict": verdict,
        "mean_P_T": round(float(np.mean([s["P_T"] for s in all_s])), 4),
        "success_rate": round(float(np.mean([s["won"] for s in all_s])), 4),
        "focus_death_rate": round(float(np.mean([s["focus_death"] for s in all_s])), 4),
        "cap_hit_rate": round(float(np.mean([s["cap_hit"] for s in all_s])), 4),
        "env_crash_rate": round(float(np.mean([s["env_crash"] for s in all_s])), 4),
        "valid_action_rate": round(float(np.mean([s["valid_frac"] for s in all_s])), 4),
        "response_clip_rate": round(float(np.mean([s["clip_frac"] for s in all_s])), 4),
        "mean_turns": round(float(np.mean([s["n_turns"] for s in all_s])), 2),
        "dead_tasks": dead,
        "per_task": task_table,
    }
    with open(os.path.join(args.out, "gate_report.json"), "w") as f:
        json.dump(report, f, indent=1)

    # 3 inspection trajectories: a winner, a focus-death, the longest
    picks = {}
    for tid, s in traj_stats.items():
        if s["won"] and "winner" not in picks:
            picks["winner"] = tid
        if s["focus_death"] and "focus_death" not in picks:
            picks["focus_death"] = tid
    picks["longest"] = max(traj_stats, key=lambda t: traj_stats[t]["n_turns"])
    for name, tid in picks.items():
        with open(os.path.join(args.out, f"traj_{name}.txt"), "w") as f:
            s = traj_stats[tid]
            f.write(f"# {name}: task={s['task']} var={s['variation']} P_T={s['P_T']} "
                    f"won={s['won']} turns={s['n_turns']}\n\n")
            for r in trajs[tid]:
                f.write(f"===== turn {r['turn_index']} (valid={r['is_action_valid']}, "
                        f"parse={r['parse_status']}, r={r['env_reward']:.3f}) =====\n")
                f.write("--- PROMPT ---\n" + r["observation"] + "\n")
                f.write("--- MODEL ---\n" + r["raw_model_response"] + "\n\n")

    print(json.dumps({k: v for k, v in report.items() if k != "per_task"}, indent=1))
    print(f"report + 3 inspection trajectories -> {args.out}/")


if __name__ == "__main__":
    main()
