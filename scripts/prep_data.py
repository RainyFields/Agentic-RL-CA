#!/usr/bin/env python3
"""Phase 1.1 data prep — build verl-agent-format train/test/val_2048 parquet.

Reuses the tr1 HDFS copy of PeterJinGo/nq_hotpotqa_train (Search-R1-processed format:
templated `prompt`, no `question` column) instead of re-downloading from HF. The raw
question is recovered from the fixed "Question: " marker of the Search-R1 template and
re-packed into the verl-agent schema (mirrors
examples/data_preprocess/preprocess_search_r1_dataset.py::process_single_row).

val_2048.parquet: fixed stratified subsample of the processed test set (seed 42),
proportional allocation with a per-dataset floor of 64 rows (bamboogle capped at its 125),
so val-core/macro_em (mean of per-dataset EMs) has usable per-dataset sample sizes.
"""
import argparse
import os

import numpy as np
import pandas as pd

SEED = 42
VAL_SIZE = 2048
VAL_FLOOR = 64
QUESTION_MARKER = "Question: "


def extract_question(prompt) -> str:
    content = prompt[0]["content"]
    assert QUESTION_MARKER in content, f"marker not found in prompt: {content[:200]!r}"
    return content.rsplit(QUESTION_MARKER, 1)[1].strip()


def process_row(row, split):
    question = extract_question(row["prompt"])
    reward_model_data = row["reward_model"]
    ground_truth = reward_model_data["ground_truth"]
    data_source = str(row["data_source"])
    tools_kwargs = {
        "search": {
            "create_kwargs": {
                "ground_truth": ground_truth,
                "question": question,
                "data_source": data_source,
            }
        }
    }
    extra_info = {
        "index": row.name,
        "need_tools_kwargs": True,
        "question": question,
        "split": split,
        "tools_kwargs": tools_kwargs,
    }
    return pd.Series(
        {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": "You are a helpful and harmless assistant."},
                {"role": "user", "content": question},
            ],
            "ability": row.get("ability"),
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "env_kwargs": {
                "ground_truth": ground_truth,
                "question": question,
                "data_source": data_source,
            },
        }
    )


def stratified_val(df_test: pd.DataFrame) -> pd.DataFrame:
    counts = df_test["data_source"].value_counts().sort_index()
    # floor allocation, capped by availability
    alloc = {ds: min(VAL_FLOOR, n) for ds, n in counts.items()}
    remaining = VAL_SIZE - sum(alloc.values())
    # distribute the rest proportionally over spare capacity
    spare = {ds: counts[ds] - alloc[ds] for ds in counts.index}
    total_spare = sum(spare.values())
    for ds in counts.index:
        alloc[ds] += int(round(remaining * spare[ds] / total_spare))
    # rounding drift -> fix on the largest dataset
    drift = VAL_SIZE - sum(alloc.values())
    alloc[counts.idxmax()] += drift

    rng = np.random.RandomState(SEED)
    parts = []
    for ds in counts.index:
        sub = df_test[df_test["data_source"] == ds]
        idx = rng.choice(len(sub), size=alloc[ds], replace=False)
        parts.append(sub.iloc[np.sort(idx)])
    val = pd.concat(parts).reset_index(drop=True)
    assert len(val) == VAL_SIZE, len(val)
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw_dir",
        default=os.environ.get(
            "SEARCHR1_DATA", "/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data"
        )
        + "/nq_hotpotqa_train",
    )
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for split in ["train", "test"]:
        raw = pd.read_parquet(os.path.join(args.raw_dir, f"{split}.parquet"))
        processed = raw.apply(lambda r: process_row(r, split), axis=1)
        out = os.path.join(args.out_dir, f"{split}.parquet")
        processed.to_parquet(out, index=False)
        print(f"{split}: {len(processed)} rows -> {out}")
        print(processed["data_source"].value_counts().to_dict())
        if split == "test":
            val = stratified_val(processed)
            val_out = os.path.join(args.out_dir, "val_2048.parquet")
            val.to_parquet(val_out, index=False)
            print(f"val_2048: {len(val)} rows -> {val_out}")
            print(val["data_source"].value_counts().to_dict())


if __name__ == "__main__":
    main()
