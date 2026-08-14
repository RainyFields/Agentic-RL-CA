#!/usr/bin/env python3
"""Collect curves + eval tables for the cross-environment credit-assignment comparison
(token-PPO / turn-PPO / GRPO / GiGPO / HCAPO on SearchQA + ALFWorld).

SearchQA: Qwen3-4B, 4-turn, non-thinking (protocol_4turn_nothink2k), 500 steps, seed 0.
  Curves  <- wandb rainyfields/agentic-rl-ca (val-core/macro_em every 25 steps).
  Eval    <- outputs/eval_full/eval4b_*/paper_table.csv (51,713-q full set, macro EM).
ALFWorld: Qwen3-1.7B non-thinking, RL from shared replay-BC init, single seed
  (reward_models 2026-07-11..13 campaign + token-PPO leg 2026-07-24).
  Curves  <- reward_models 2026-07-13_credit_assignment_comparison_assets/history_*.csv.
  Eval    <- unseen_results.csv / unseen_cmp_results.txt (134-game OOD split, greedy).

Missing runs (e.g. SearchQA token-PPO / GiGPO while still training) are skipped with a
warning so the report can be rebuilt incrementally.

Outputs (results/): searchqa_curves.csv, searchqa_eval.csv, alfworld_curves.csv,
alfworld_eval.csv, provenance.json
"""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")

# ---------------------------------------------------------------- SearchQA ----
# method -> (wandb display_name, eval_full dir). HCAPO primary = paper-correct variant;
# the adapted variants are carried as extra table rows (see report §setup).
SQA_REPO = "/home/tiger/xiaoxuan/Agentic-RL-CA"
SQA_RUNS = {
    "token_ppo":  ("token_ppo_qwen3-4b_4turn_nothink2k_s0",   "eval4b_nt_token_ppo_s0"),
    "turn_ppo":   ("turn_ppo_b0_qwen3-4b_4turn_nothink2k_s0", "eval4b_nt_turn_ppo_b0_s0"),
    "grpo":       ("token_grpo_qwen3-4b_4turn_nothink2k_s0",  "eval4b_nt_token_grpo_s0"),
    "gigpo":      ("gigpo_qwen3-4b_4turn_nothink2k_s0",       "eval4b_nt_gigpo_s0"),
    "hcapo":      ("hcapo_paper_qwen3-4b_4turn_nothink2k_s0", "eval4b_hcapo_paper_s0"),
    "hcapo_ans":  ("hcapo_ans_qwen3-4b_4turn_nothink2k_s0",   "eval4b_hcapo_ans_s0"),
}
SQA_FLOOR_DIR = "eval4b_floor"  # pre-RL Qwen3-4B floor (same non-thinking eval protocol)


def collect_searchqa():
    import wandb
    api = wandb.Api()
    curves, evals, prov = [], [], {}
    for m, (run_name, eval_dir) in SQA_RUNS.items():
        runs = list(api.runs("rainyfields/agentic-rl-ca",
                             filters={"display_name": run_name}))
        if not runs:
            print(f"[searchqa] WARN: no wandb run for {m} ({run_name}) — skipped")
        rows = []
        for r in runs:
            for h in r.scan_history(keys=["_step", "val-core/macro_em"]):
                rows.append((h["_step"], h["val-core/macro_em"]))
        if rows:
            df = (pd.DataFrame(rows, columns=["step", "val_macro_em"])
                  .sort_values("step").drop_duplicates("step", keep="last"))
            df.insert(0, "method", m)
            curves.append(df)
            prov[f"searchqa_curve_{m}"] = [r.id for r in runs]
        pt = os.path.join(SQA_REPO, "outputs", "eval_full", eval_dir, "paper_table.csv")
        if os.path.exists(pt):
            t = pd.read_csv(pt)
            t.insert(0, "method", m)
            evals.append(t)
            prov[f"searchqa_eval_{m}"] = pt
        else:
            print(f"[searchqa] WARN: no full-set eval for {m} ({eval_dir}) — skipped")
    fl = os.path.join(SQA_REPO, "outputs", "eval_full", SQA_FLOOR_DIR, "paper_table.csv")
    if os.path.exists(fl):
        t = pd.read_csv(fl)
        t.insert(0, "method", "floor")
        evals.append(t)
        prov["searchqa_eval_floor"] = fl
    if curves:
        pd.concat(curves).to_csv(os.path.join(OUT, "searchqa_curves.csv"), index=False)
    if evals:
        pd.concat(evals).to_csv(os.path.join(OUT, "searchqa_eval.csv"), index=False)
    return prov


# ---------------------------------------------------------------- ALFWorld ----
ALF_HIST = ("/home/tiger/xiaoxuan/reward_models/docs/reports/"
            "2026-07-13_credit_assignment_comparison_assets")
ALF_LEDGER = "/mnt/hdfs/mlsys/users/xiaoxuan/alfworld_prm_gigpo/spa_data/unseen_cmp_results.txt"
ALF_RUNS = {  # method -> (history csv, unseen success, eval ckpt step)
    "token_ppo":   ("history_PPO-token_cmp150.csv",  0.761, 150),
    "turn_ppo":    ("history_PPO_mvn5d9np.csv",      0.963, 150),
    "grpo":        ("history_GRPO_ewoih80v.csv",     0.881, 265),
    "gigpo":       ("history_GiGPO_51al57ag.csv",    0.955, 200),
    "hcapo":       ("history_HCAPOpaper_local.csv",  0.784, 145),  # paper-correct v2
    "hcapo_ans":   ("history_HCAPO_51q4rbh3.csv",    0.791, 145),  # adapted v1
}
ALF_FLOOR = 0.440  # BC pi_base init (unseen_results.csv)


def collect_alfworld():
    curves, evals, prov = [], [], {}
    for m, (csv, unseen, step) in ALF_RUNS.items():
        p = os.path.join(ALF_HIST, csv)
        if not os.path.exists(p):
            print(f"[alfworld] WARN: missing {p} — skipped")
            continue
        df = pd.read_csv(p)
        keep = df[["_step", "val/success_rate"]].dropna()
        keep.columns = ["step", "val_success"]
        keep = keep.sort_values("step").drop_duplicates("step", keep="last")
        keep.insert(0, "method", m)
        curves.append(keep)
        evals.append({"method": m, "unseen_success": unseen, "eval_step": step})
        prov[f"alfworld_curve_{m}"] = p
    evals.append({"method": "floor", "unseen_success": ALF_FLOOR, "eval_step": 0})
    pd.concat(curves).to_csv(os.path.join(OUT, "alfworld_curves.csv"), index=False)
    pd.DataFrame(evals).to_csv(os.path.join(OUT, "alfworld_eval.csv"), index=False)
    # cross-check the hard-coded unseen numbers against the HDFS ledger when reachable
    if os.path.exists(ALF_LEDGER):
        prov["alfworld_ledger"] = ALF_LEDGER
    return prov


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    prov = {}
    prov.update(collect_alfworld())
    prov.update(collect_searchqa())
    with open(os.path.join(OUT, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=2, sort_keys=True)
    print("collected ->", os.path.abspath(OUT))
