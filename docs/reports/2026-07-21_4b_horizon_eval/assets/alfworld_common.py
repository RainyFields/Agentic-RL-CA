#!/usr/bin/env python3
"""AlfWorld side of the cross-environment credit-assignment report (reward_models 2026-07-13
campaign, Qwen3-1.7B, single seed). Loads: per-episode eval (turns + success + task type) from the
unseen rollout logs, the training success-rate curves, and the unseen eval-accuracy table.

Environments/scales differ by design (AlfWorld 1.7B; SearchQA 4B) — see the report caveats.
Methods on AlfWorld: GRPO, GiGPO, turn-PPO (token-PPO was not run here — SearchQA-only).
"""
import json
import os
import re

import pandas as pd

SPA = "/mnt/hdfs/mlsys/users/xiaoxuan/alfworld_prm_gigpo/spa_data"
HIST = "/home/tiger/xiaoxuan/reward_models/docs/reports/2026-07-13_credit_assignment_comparison_assets"

# method -> (best-ckpt unseen rollout dir, training-history csv). Best ckpt per method (report table).
ALF = {
    "grpo":      ("unseen_cmp_grpo_265", "history_GRPO_ewoih80v.csv"),
    "gigpo":     ("unseen_cmp_gigpo",    "history_GiGPO_51al57ag.csv"),
    "turn_ppo":  ("unseen_cmp_ppo",      "history_PPO_mvn5d9np.csv"),
    # token-level PPO (adv_estimator=gae), trained 2026-07-24 to complete the turn-vs-token
    # contrast on the long-horizon environment. Same BC init, budget and eval step (150) as
    # turn-PPO — the two differ ONLY in GAE granularity.
    "token_ppo": ("unseen_cmp_ppo_token", "history_PPOtoken_z7gkaah4.csv"),
}
# unseen 134-game success at the eval ckpt (verified against the rollout logs + unseen_results.csv)
ALF_EVAL = {"grpo": 0.881, "gigpo": 0.955, "turn_ppo": 0.963, "token_ppo": 0.761}
ALF_EVAL_STEP = {"grpo": 265, "gigpo": 200, "turn_ppo": 150, "token_ppo": 150}
ALF_METHOD_ORDER = ["grpo", "gigpo", "turn_ppo", "token_ppo"]
ALF_PRETTY = {"grpo": "GRPO", "gigpo": "GiGPO", "turn_ppo": "turn-PPO", "token_ppo": "token-PPO"}
# ordered longest-horizon-last (approx expert-demo length: cool~7, pick_place~6, examine, clean~8, heat, pick_two~11)
TASK_ORDER = ["cool", "pick_place", "examine", "clean", "heat", "pick_two"]


def task_type(obs):
    """AlfWorld task type from the turn-0 observation. The transcript prompt embeds a one-shot
    EXAMPLE before the real goal, so take the LAST 'your task is to:' match, not the first."""
    m = list(re.finditer(r"your task is to:\s*(.+)", obs or "", re.I))
    if not m:
        return "?"
    g = m[-1].group(1).lower()
    if "clean" in g:
        return "clean"
    if "heat" in g:
        return "heat"
    if "cool" in g:
        return "cool"
    if "examine" in g or "look at" in g:
        return "examine"
    if "two" in g:
        return "pick_two"
    return "pick_place"


def load_alfworld_episodes(methods=None):
    """Per-episode eval DataFrame: method, traj_uid, turns, success, task_type.
    turns = #turn-rows for the trajectory; success = env_won on the terminal turn."""
    methods = methods or ALF_METHOD_ORDER
    rows = []
    for meth in methods:
        d = os.path.join(SPA, ALF[meth][0], "rollout_log.jsonl")
        recs = [json.loads(l) for l in open(d) if l.strip()]
        by = {}
        for r in recs:
            by.setdefault(r["traj_uid"], []).append(r)
        for uid, rs in by.items():
            rs = sorted(rs, key=lambda x: x["turn_index"])
            rows.append({
                "method": meth,
                "traj_uid": str(uid),
                "turns": len(rs),
                "success": bool(rs[-1].get("env_won", False)),
                "task_type": task_type(rs[0].get("observation", "")),
            })
    df = pd.DataFrame(rows)
    # sanity: per-method success should match ALF_EVAL within rounding
    for meth in methods:
        s = df[df.method == meth]["success"].mean()
        assert abs(s - ALF_EVAL[meth]) < 0.01, f"{meth} success {s:.4f} != table {ALF_EVAL[meth]}"
    return df


def load_alfworld_training(methods=None):
    """method -> DataFrame(step, success_rate) — the SEEN 128-game greedy val curve (every 5 steps)."""
    methods = methods or ALF_METHOD_ORDER
    out = {}
    for meth in methods:
        p = os.path.join(HIST, ALF[meth][1])
        d = pd.read_csv(p)
        v = d[["_step", "val/success_rate"]].dropna().rename(
            columns={"_step": "step", "val/success_rate": "success_rate"})
        out[meth] = v.reset_index(drop=True)
    return out


if __name__ == "__main__":
    ep = load_alfworld_episodes()
    print("=== per-episode summary ===")
    print(ep.groupby("method").agg(n=("success", "size"), success=("success", "mean"),
                                   med_turns=("turns", "median"),
                                   mean_turns=("turns", "mean")).round(3).to_string())
    print("\n=== success x task_type (mean success) ===")
    print(pd.crosstab(ep.task_type, ep.method, values=ep.success, aggfunc="mean").round(3)
          .reindex(TASK_ORDER).to_string())
    print("\n=== mean turns x task_type ===")
    print(pd.crosstab(ep.task_type, ep.method, values=ep.turns, aggfunc="mean").round(1)
          .reindex(TASK_ORDER).to_string())
    tr = load_alfworld_training()
    print("\n=== training curve rows ===")
    for m, d in tr.items():
        print(f"  {m}: {len(d)} val points, step {d.step.min()}..{d.step.max()}, "
              f"final success {d.success_rate.iloc[-1]:.3f}")
