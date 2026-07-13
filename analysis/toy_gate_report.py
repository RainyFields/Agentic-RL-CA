#!/usr/bin/env python3
"""Phase 1.3 toy-gate report — render trajectory dumps + pre-registered gate metrics.

Input: rollout_log.jsonl written by rollout_loop.py under DUMP_TRAIN_SAMPLE=1
(one record per active turn: traj_uid, uid, turn_index, observation (the full
policy-visible templated prompt text), raw_model_response, parse_status,
is_action_valid, env_reward, env_done, response_token_count, prompt_pretrunc_len).

Output:
  <out_dir>/trajectory_<algo>.md   annotated per-turn trajectory dump (user-reviewable)
  <out_dir>/gate_metrics_<algo>.json  the pre-registered Wave-0/base-model gate numbers

All horizon/length limits are ARGUMENTS (no literal 4s / 512s — plan §Horizon strategy).
Gate checks implemented (plan §Pre-registered decision rules; thresholds passed as args):
  valid-action rate, truncation-event rate (prompt-side pretrunc>max OR response at cap
  without terminal tag), search-invocation floor, answer-turn distribution, macro/micro EM,
  fraction of groups (uid) with nonzero within-group terminal-reward variance.
"""
import argparse
import json
import os
from collections import defaultdict


def load_records(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def group_trajectories(recs):
    trajs = defaultdict(list)
    for r in recs:
        trajs[r["traj_uid"]].append(r)
    for t in trajs.values():
        t.sort(key=lambda r: r["turn_index"])
    return dict(trajs)


def has_search(resp):
    return "<search>" in resp and "</search>" in resp


def has_answer(resp):
    return "<answer>" in resp and "</answer>" in resp


def compute_gate_metrics(trajs, max_prompt_length, max_response_length):
    n_turns = 0
    n_valid = 0
    prompt_trunc_events = 0
    resp_trunc_events = 0
    n_traj = len(trajs)
    traj_with_search = 0
    answer_turn_hist = defaultdict(int)
    no_answer_traj = 0
    terminal_reward = {}
    group_of_traj = {}
    ds_reward = defaultdict(list)

    for tuid, turns in trajs.items():
        any_search = False
        answered_at = None
        for r in turns:
            n_turns += 1
            n_valid += bool(r["is_action_valid"])
            if r.get("prompt_pretrunc_len", -1) > max_prompt_length:
                prompt_trunc_events += 1
            resp = r["raw_model_response"]
            if r["response_token_count"] >= max_response_length and not (
                has_answer(resp) or has_search(resp)
            ):
                resp_trunc_events += 1
            if has_search(resp):
                any_search = True
            if has_answer(resp) and answered_at is None:
                answered_at = r["turn_index"]
        traj_with_search += any_search
        if answered_at is None:
            no_answer_traj += 1
        else:
            answer_turn_hist[answered_at] += 1
        last = turns[-1]
        terminal_reward[tuid] = float(last["env_reward"])
        group_of_traj[tuid] = last.get("uid", tuid)
        ds_reward[last.get("data_source", "unknown")].append(float(last["env_reward"]))

    groups = defaultdict(list)
    for tuid, g in group_of_traj.items():
        groups[g].append(terminal_reward[tuid])
    multi_groups = {g: rs for g, rs in groups.items() if len(rs) >= 2}
    var_groups = sum(1 for rs in multi_groups.values() if max(rs) != min(rs))

    per_ds_em = {ds: sum(rs) / len(rs) for ds, rs in sorted(ds_reward.items())}
    micro_em = sum(terminal_reward.values()) / max(n_traj, 1)
    macro_em = sum(per_ds_em.values()) / max(len(per_ds_em), 1)

    return {
        "n_trajectories": n_traj,
        "n_turns": n_turns,
        "valid_action_rate": n_valid / max(n_turns, 1),
        "prompt_truncation_events": prompt_trunc_events,
        "response_truncation_events": resp_trunc_events,
        "truncation_event_rate": (prompt_trunc_events + resp_trunc_events) / max(n_turns, 1),
        "search_invocation_rate": traj_with_search / max(n_traj, 1),
        "answer_turn_distribution": dict(sorted(answer_turn_hist.items())),
        "trajectories_without_answer": no_answer_traj,
        "micro_em": micro_em,
        "macro_em": macro_em,
        "per_dataset_em": per_ds_em,
        "n_groups_with_ge2_traj": len(multi_groups),
        "frac_groups_nonzero_reward_variance": var_groups / max(len(multi_groups), 1),
        "avg_turns_per_traj": n_turns / max(n_traj, 1),
    }


def check_gates(m, args):
    return {
        f"macro_em >= {args.gate_min_em}": m["macro_em"] >= args.gate_min_em,
        f"valid_action_rate >= {args.gate_min_valid}": m["valid_action_rate"] >= args.gate_min_valid,
        f"truncation_event_rate < {args.gate_max_trunc}": m["truncation_event_rate"] < args.gate_max_trunc,
        f"search_invocation_rate >= {args.gate_min_search}": m["search_invocation_rate"] >= args.gate_min_search,
        f"frac_groups_nonzero_reward_variance >= {args.gate_min_group_var}": m[
            "frac_groups_nonzero_reward_variance"
        ]
        >= args.gate_min_group_var,
    }


def render_md(algo, trajs, metrics, gates, out_path, max_traj=None):
    lines = [f"# Toy-gate trajectory dump — `{algo}`", ""]
    lines += ["## Gate metrics", "", "```json", json.dumps(metrics, indent=2), "```", ""]
    lines += ["## Gate checks", ""]
    for k, ok in gates.items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} — {k}")
    lines.append("")
    shown = 0
    for tuid, turns in trajs.items():
        if max_traj is not None and shown >= max_traj:
            lines.append(f"\n*(… {len(trajs) - shown} more trajectories omitted)*")
            break
        shown += 1
        last = turns[-1]
        lines += [
            "---",
            f"## Trajectory `{tuid[:8]}` — data_source={last.get('data_source','?')} "
            f"terminal_reward={last['env_reward']} turns={len(turns)}",
        ]
        for r in turns:
            valid = "valid" if r["is_action_valid"] else "INVALID"
            lines += [
                "",
                f"### Turn {r['turn_index']} ({valid}, parse={r['parse_status']}, "
                f"reward={r['env_reward']}, done={r['env_done']}, "
                f"resp_tokens={r['response_token_count']}, prompt_tokens={r.get('prompt_pretrunc_len','?')})",
                "",
                "**Policy-visible prompt/observation:**",
                "",
                "````",
                r["observation"],
                "````",
                "",
                "**Raw model response (`<think>` + action):**",
                "",
                "````",
                r["raw_model_response"],
                "````",
            ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--algo", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_prompt_length", type=int, required=True)
    ap.add_argument("--max_response_length", type=int, required=True)
    ap.add_argument("--max_traj_in_md", type=int, default=32)
    ap.add_argument("--gate_min_em", type=float, default=0.03)
    ap.add_argument("--gate_min_valid", type=float, default=0.90)
    ap.add_argument("--gate_max_trunc", type=float, default=0.001)
    ap.add_argument("--gate_min_search", type=float, default=0.50)
    ap.add_argument("--gate_min_group_var", type=float, default=0.20)
    args = ap.parse_args()

    recs = load_records(args.jsonl)
    trajs = group_trajectories(recs)
    metrics = compute_gate_metrics(trajs, args.max_prompt_length, args.max_response_length)
    gates = check_gates(metrics, args)

    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, f"trajectory_{args.algo}.md")
    render_md(args.algo, trajs, metrics, gates, md_path, args.max_traj_in_md)
    json_path = os.path.join(args.out_dir, f"gate_metrics_{args.algo}.json")
    with open(json_path, "w") as f:
        json.dump({"metrics": metrics, "gates": gates}, f, indent=2)

    print(f"[toy-gate] {args.algo}: wrote {md_path} and {json_path}")
    for k, ok in gates.items():
        print(f"  {'PASS' if ok else 'FAIL'} — {k}")


if __name__ == "__main__":
    main()
