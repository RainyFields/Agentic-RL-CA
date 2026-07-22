#!/usr/bin/env python3
"""M5 addendum probes: rollout speed benchmark, sandbox surface, full-trajectory capture.

Outputs (under --out):
  speed_bench.json     reset/load latency + steps/sec (gold replay + random valid actions)
  sandbox_probe.json   rooms, action templates, valid-action counts, object counts
  full_traj_<task>_v<var>.json   complete gold trajectory with UNTRUNCATED observations
Run: python bench_and_probe.py --out <dir>
"""

import argparse
import json
import random
import time
from pathlib import Path

from scienceworld import ScienceWorldEnv


def bench_speed(env):
    out = {}
    # load latency (fresh task+variation), 5 samples
    t = []
    for i in range(5):
        t0 = time.time()
        env.load("boil", i, "easy")
        env.reset()
        t.append(time.time() - t0)
    out["load_plus_reset_sec"] = {"mean": sum(t) / len(t), "min": min(t), "max": max(t)}

    # gold-replay stepping speed
    env.load("change-the-state-of-matter-of", 0, "easy", generateGoldPath=True)
    gold = env.get_gold_action_sequence()
    env.reset()
    t0 = time.time()
    n = 0
    for a in gold:
        env.step(a)
        n += 1
    dt = time.time() - t0
    out["gold_steps_per_sec"] = n / dt
    out["gold_ms_per_step"] = 1000 * dt / n

    # random valid-action stepping speed (the RL-rollout-relevant number)
    env.load("boil", 0, "easy")
    env.reset()
    rng = random.Random(0)
    t0 = time.time()
    n = 0
    for _ in range(300):
        acts = env.get_valid_action_object_combinations()
        a = rng.choice(acts)
        _, _, done, _ = env.step(a)
        n += 1
        if done:
            env.reset()
    dt = time.time() - t0
    out["random_steps_per_sec_incl_validactions"] = n / dt
    out["random_ms_per_step_incl_validactions"] = 1000 * dt / n

    # random stepping without querying valid actions each step (query once)
    env.reset()
    acts = env.get_valid_action_object_combinations()
    t0 = time.time()
    n = 0
    for _ in range(300):
        _, _, done, _ = env.step(rng.choice(acts))
        n += 1
        if done:
            env.reset()
            acts = env.get_valid_action_object_combinations()
    dt = time.time() - t0
    out["random_steps_per_sec_step_only"] = n / dt
    out["random_ms_per_step_step_only"] = 1000 * dt / n
    return out


def probe_sandbox(env):
    env.load("boil", 0, "easy")
    obs, info = env.reset()
    d = {}
    d["action_templates"] = env.get_possible_actions()
    d["n_action_templates"] = len(d["action_templates"])
    d["n_objects_current_state"] = len(env.get_possible_objects())
    d["n_valid_action_object_combos_t0"] = len(env.get_valid_action_object_combinations())
    d["rooms"] = env.get_possible_locations() if hasattr(env, "get_possible_locations") else None
    d["simplifications"] = env.get_simplifications_used()
    d["initial_observation"] = obs
    d["inventory_cmd"] = env.step("inventory")[0]
    return d


def capture_full_traj(env, task, var):
    env.load(task, var, "easy", generateGoldPath=True)
    gold = env.get_gold_action_sequence()
    obs, info = env.reset()
    steps = [{"t": -1, "action": "(reset)", "obs": obs, "score": 0}]
    score_prev = 0
    for i, a in enumerate(gold):
        obs, reward, done, info = env.step(a)
        steps.append({"t": i, "action": a, "obs": obs, "reward": reward,
                      "score": info["score"], "score_delta": info["score"] - score_prev,
                      "done": bool(done)})
        score_prev = info["score"]
        if done:
            break
    return {"task": task, "variation": var, "task_description": env.get_task_description(),
            "gold_len": len(gold), "final_score": score_prev, "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--traj-task", default="find-animal")
    ap.add_argument("--traj-var", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = ScienceWorldEnv("", envStepLimit=300)

    speed = bench_speed(env)
    (out / "speed_bench.json").write_text(json.dumps(speed, indent=2))
    print("speed:", json.dumps(speed, indent=2))

    sandbox = probe_sandbox(env)
    (out / "sandbox_probe.json").write_text(json.dumps(sandbox, indent=2))
    print("sandbox: templates", sandbox["n_action_templates"],
          "objects", sandbox["n_objects_current_state"],
          "valid combos t0", sandbox["n_valid_action_object_combos_t0"])

    traj = capture_full_traj(env, args.traj_task, args.traj_var)
    (out / f"full_traj_{args.traj_task}_v{args.traj_var}.json").write_text(json.dumps(traj, indent=2))
    print(f"full traj: {traj['task']} v{traj['variation']} steps={len(traj['steps'])} "
          f"final={traj['final_score']}")


if __name__ == "__main__":
    main()
