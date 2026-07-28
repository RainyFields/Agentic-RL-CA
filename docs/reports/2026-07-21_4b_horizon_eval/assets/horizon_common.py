#!/usr/bin/env python3
"""Phase 3.3 horizon analysis — shared loader + join (foundation for A1–A5).

Pure post-processing over the per-episode eval JSONL (§4 schema, one row per question) ⋈ the
H_gold annotation table. The eval dump carries `question` (+ data_source, em, turns,
n_search_calls, tokens_generated, truncated_at_cap, answered); we join it to hop_table by
(data_source, normalized question) — which recovers the global `index`, H_gold, and the
secondary-split fields. No rollouts are re-run by any analysis.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = "/home/tiger/xiaoxuan/Agentic-RL-CA"
TESTPARQUET = f"{REPO}/data/searchR1_processed_direct/test.parquet"
HOPTABLE = f"{REPO}/data/hop_annotations/hop_table.parquet"
EVAL_ROOT = f"{REPO}/outputs/eval_full"

# method -> eval-output label (the full-set run dir under outputs/eval_full/).
# PPO arms are added when their 4B evals finish; GRPO is the flat-credit reference for every Δ.
DEFAULT_METHOD_LABELS = {
    "floor": "eval4b_floor",
    "token_grpo": "eval4b_token_grpo_s0",
    "gigpo": "eval4b_gigpo_s0",
    # "token_ppo": "eval4b_token_ppo_s0",
    # "turn_ppo": "eval4b_turn_ppo_b0_s0",
}
METHOD_ORDER = ["floor", "token_ppo", "turn_ppo", "token_grpo", "gigpo"]
METHOD_PRETTY = {"floor": "pre-RL floor", "token_ppo": "token-PPO", "turn_ppo": "turn-PPO",
                 "token_grpo": "token-GRPO", "gigpo": "GiGPO"}
DS_ORDER = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
DS_PRETTY = {"nq": "NQ", "triviaqa": "TriviaQA", "popqa": "PopQA", "hotpotqa": "HotpotQA",
             "2wikimultihopqa": "2Wiki", "musique": "MuSiQue", "bamboogle": "Bamboogle"}
REF_METHOD = "token_grpo"   # GRPO = flat-credit reference


def load_hop_table():
    """hop_table sorted by global index (row i == the i-th global test item)."""
    return pd.read_parquet(HOPTABLE).sort_values("index").reset_index(drop=True)


def load_eval(label, hop_df=None, cap=4):
    """Load outputs/eval_full/<label>/val_trajectories_step*.jsonl -> tidy per-episode DataFrame,
    joined to H_gold by POSITION. The full-set dump is emitted in test.parquet order (validation is
    unshuffled and the DP gather preserves order — verified 100% data_source match on GRPO), so dump
    row i == the i-th test item == hop_table row i. NB the dumped `index` field is a per-batch local
    0..B-1 counter (rollout_loop.py:185 overwrites the global index with the batch position), so it
    is deliberately NOT used. The guard is the data_source consistency check: any mismatch means the
    dump order broke or this isn't the full 51,713-row test set (e.g. the val_2048 smoke)."""
    if hop_df is None:
        hop_df = load_hop_table()
    files = sorted(glob.glob(f"{EVAL_ROOT}/{label}/val_trajectories_step*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no eval JSONL for label={label} under {EVAL_ROOT}/{label}/")
    recs = []
    for f in files:                       # normally 1 file (val_before_train)
        with open(f) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    if len(recs) != len(hop_df):
        print(f"[load_eval] WARNING {label}: {len(recs)} dump records != {len(hop_df)} test items "
              f"— position join misaligned (is this the full test.parquet run?)")
    hop_recs = hop_df.to_dict("records")
    n = min(len(recs), len(hop_recs))
    rows, ds_mismatch = [], 0
    for i in range(n):
        r, h = recs[i], hop_recs[i]
        if h["data_source"] != r["data_source"]:
            ds_mismatch += 1
        rows.append({
            "data_source": r["data_source"],
            "index": int(h["index"]),
            "H_gold": int(h["H_gold"]),
            "em": float(r["em"]),
            "turns": int(r["turns"]),
            "n_search_calls": r.get("n_search_calls"),
            "tokens_generated": r.get("tokens_generated"),
            "hotpot_type": h["hotpot_type"],
            "wiki_type": h["wiki_type"],
            "popqa_pop_bucket": h["popqa_pop_bucket"],
            "popqa_s_pop": (float(h["popqa_s_pop"]) if pd.notna(h["popqa_s_pop"]) else np.nan),
        })
    df = pd.DataFrame(rows)
    # Truncation recomputed from the trustworthy fields (em, turns): the dumped parse_status-based
    # answered/truncated_at_cap disagree with EM (Qwen3 <think> blocks can score a correct answer
    # while parse_status != 'answer'). hit_cap = used the whole turn budget; cap_fail = the user's
    # "capacity-limited failure" (hit the cap AND still wrong).
    df["hit_cap"] = df["turns"] >= cap
    df["cap_fail"] = (df["em"] == 0.0) & df["hit_cap"]
    mm = ds_mismatch / max(1, len(df))
    print(f"[load_eval] {label}: {len(df)} episodes joined by position; task-consistency "
          f"mismatch {mm:.4f}")
    if mm > 0.01:
        print(f"[load_eval] WARNING {label}: {mm:.1%} position→task mismatch — dump order != "
              f"test.parquet order (or not the full 51,713-row set); H_gold UNRELIABLE.")
    df.attrs.update(label=label, ds_mismatch=mm)
    return df


def load_all(method_labels=None):
    method_labels = method_labels or DEFAULT_METHOD_LABELS
    hop_df = load_hop_table()
    out = {}
    for method, label in method_labels.items():
        try:
            d = load_eval(label, hop_df)
            d["method"] = method
            out[method] = d
        except FileNotFoundError as e:
            print(f"[load_all] skip {method}: {e}")
    return out


if __name__ == "__main__":
    import sys
    label = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_METHOD_LABELS["token_grpo"]
    df = load_eval(label)
    print("\n=== per-stratum EM (mini-A1) ===")
    print(df.groupby("H_gold").agg(n=("em", "size"), EM=("em", "mean"),
                                   med_turns=("turns", "median"),
                                   cap_fail=("cap_fail", "mean")).round(4).to_string())
    print("\n=== per-task EM ===")
    print(df.groupby("data_source").agg(n=("em", "size"), EM=("em", "mean")).round(4).to_string())
