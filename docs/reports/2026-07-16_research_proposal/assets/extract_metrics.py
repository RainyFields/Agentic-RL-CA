#!/usr/bin/env python3
"""Extract per-step training metrics from launch logs -> one CSV per run.

Last-occurrence-per-step parsing is authoritative (crash-resume replays steps);
pre/post-relaunch logs are merged in chronological order per rq3_readout convention.
Rerun:  python3 extract_metrics.py   (writes csv/<run>.csv next to this file)
"""
import csv
import os
import re

L = os.path.expanduser("~/xiaoxuan/worker_logs/launches")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(OUT, exist_ok=True)

RUNS = {
    "token_grpo_s0": ["20260713_214311_arlca-wave0-grpo-s0.log",
                      "20260713_232357_arlca-wave0-grpo-s0b.log"],
    "gigpo_s0": ["20260715_061654_arlca-gigpo-s0.log"],
    "token_ppo_s0": ["20260714_083104_arlca-token-ppo-s0.log",
                     "20260715_145258_arlca-token-ppo-s0-micro8.log"],
    "turn_ppo_b0_s0": ["20260714_082953_arlca-turn-ppo-b0-s0.log",
                       "20260714_092728_arlca-turn-ppo-b0-s0.log"],
    "turn_ppo_b0_s1": ["20260714_083039_arlca-turn-ppo-b0-s1.log"],
    "b1_s0": ["20260714_083044_arlca-b1-s0.log",
              "20260714_163146_arlca-b1-s0.log"],
    "b1_s1": ["20260714_083049_arlca-b1-s1.log",
              "20260714_182122_arlca-b1-s1-micro8.log"],
    "b1_shuffle_s0": ["20260714_083054_arlca-b1-shuffle-s0.log",
                      "20260715_000852_arlca-b1-shuffle-s0-micro8.log"],
    "b1_shuffle_s1": ["20260714_083059_arlca-b1-shuffle-s1.log",
                      "20260714_193722_arlca-b1-shuffle-s1-micro8.log"],
    "turn_ppo_b0_s2": ["20260716_013833_arlca-turn-ppo-b0-s2.log"],
    "b1_s2": ["20260716_014706_arlca-b1-s2.log"],
    "b1_shuffle_s2": ["20260716_072005_arlca-b1-shuffle-s2.log"],
}

METRICS = {
    "val-core/macro_em": "val_macro_em",
    "episode/reward/mean": "reward_mean",
    "episode/length/mean": "avg_turns",
    "response_length/clip_ratio": "clip_ratio",
    "episode/success_rate": "success_rate",
}

STEP_RE = re.compile(r"step:(\d+) - ")
KV_RES = {re.compile(re.escape(k) + r":(-?[0-9.]+)"): v for k, v in METRICS.items()}

for run, logs in RUNS.items():
    rows = {}  # step -> {col: val}
    for lg in logs:
        path = os.path.join(L, lg)
        if not os.path.exists(path):
            print(f"WARN missing {path}")
            continue
        with open(path, errors="replace") as f:
            for line in f:
                m = STEP_RE.search(line)
                if not m:
                    continue
                step = int(m.group(1))
                d = rows.setdefault(step, {})
                for kre, col in KV_RES.items():
                    km = kre.search(line)
                    if km:
                        d[col] = float(km.group(1))  # last occurrence wins
    cols = ["step"] + list(METRICS.values())
    with open(os.path.join(OUT, f"{run}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for step in sorted(rows):
            w.writerow([step] + [rows[step].get(c, "") for c in cols[1:]])
    n = len(rows)
    nval = sum(1 for r in rows.values() if "val_macro_em" in r)
    print(f"{run}: {n} steps, {nval} evals, max step {max(rows) if rows else '-'}")
