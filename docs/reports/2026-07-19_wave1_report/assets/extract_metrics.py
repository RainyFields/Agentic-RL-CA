#!/usr/bin/env python3
"""Extract per-step training metrics from launch logs -> one CSV per run (Wave-1 report).

Superset of the 2026-07-16 proposal extractor: ALL 7 completed Wave-1/2 arms + every seed,
including the post-hang relaunch logs (2026-07-17 fleet recovery). Last-occurrence-per-step
parsing is authoritative (crash-resume replays steps); pre/post-relaunch logs are listed in
CHRONOLOGICAL order per run so the relaunch value wins on any overlapping resume step.

Rerun:  python3 extract_metrics.py    (writes csv/<run>.csv next to this file)
"""
import csv
import os
import re

L = os.path.expanduser("~/xiaoxuan/worker_logs/launches")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv")
os.makedirs(OUT, exist_ok=True)

# Per run: launch logs in chronological order (pre-hang first, relaunch second).
RUNS = {
    # ---- RQ1 / RQ2 : granularity + outcome-only credit ----
    "token_grpo_s0": ["20260713_214311_arlca-wave0-grpo-s0.log",
                      "20260713_232357_arlca-wave0-grpo-s0b.log"],
    "token_grpo_s1": ["20260717_124054_arlca-token-grpo-s1.log"],  # launched 07-17, spared the hang
    "gigpo_s0": ["20260715_061654_arlca-gigpo-s0.log"],
    "gigpo_s1": ["20260716_181042_arlca-gigpo-s1.log",
                 "20260717_124530_3_arlca-gigpo-s1.log"],           # resumed @250 post-hang
    "hcapo_s0": ["20260716_175934_arlca-hcapo-s0.log",
                 "20260717_124530_5_arlca-hcapo-s0.log"],           # resumed @200 post-hang
    "token_ppo_s0": ["20260714_083104_arlca-token-ppo-s0.log",
                     "20260715_145258_arlca-token-ppo-s0-micro8.log"],
    # ---- RQ3 : progress-supervision triad, 3 seeds each ----
    "turn_ppo_b0_s0": ["20260714_082953_arlca-turn-ppo-b0-s0.log",
                       "20260714_092728_arlca-turn-ppo-b0-s0.log"],
    "turn_ppo_b0_s1": ["20260714_083039_arlca-turn-ppo-b0-s1.log"],
    "turn_ppo_b0_s2": ["20260716_013833_arlca-turn-ppo-b0-s2.log"],
    "b1_s0": ["20260714_083044_arlca-b1-s0.log",
              "20260714_163146_arlca-b1-s0.log"],
    "b1_s1": ["20260714_083049_arlca-b1-s1.log",
              "20260714_182122_arlca-b1-s1-micro8.log"],
    "b1_s2": ["20260716_014706_arlca-b1-s2.log",
              "20260717_124530_1_arlca-b1-s2.log"],                 # resumed @450 post-hang
    "b1_shuffle_s0": ["20260714_083054_arlca-b1-shuffle-s0.log",
                      "20260715_000852_arlca-b1-shuffle-s0-micro8.log"],
    "b1_shuffle_s1": ["20260714_083059_arlca-b1-shuffle-s1.log",
                      "20260714_193722_arlca-b1-shuffle-s1-micro8.log"],
    "b1_shuffle_s2": ["20260716_072005_arlca-b1-shuffle-s2.log",
                      "20260717_124530_2_arlca-b1sh-s2.log"],       # resumed @350 post-hang
    # ---- RQ4 : 8-turn horizon stress ----
    "turn_ppo_b0_8t_s0": ["20260716_104035_arlca-b0-8t-s0.log",
                          "20260717_124530_4_arlca-b0-8t-s0.log"],  # resumed @200 post-hang
    "b1_8t_s0": ["20260716_113437_arlca-b1-8t-s0.log",
                 "20260717_124530_6_arlca-b1-8t-s0.log"],           # resumed @150 post-hang
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


def main():
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
        last = max(rows) if rows else "-"
        fin = rows.get(last, {}).get("val_macro_em", "-") if rows else "-"
        print(f"{run:20s}: {n:4d} steps, {nval:3d} evals, max {last}, last-val {fin}")


if __name__ == "__main__":
    main()
