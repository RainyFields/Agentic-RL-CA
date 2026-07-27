#!/usr/bin/env python3
"""Convert ASearcher-Base-35k.jsonl -> verl/Search-R1 parquet (rung-3 feasibility).

- drops the compose_chain_qa_invalid slice (~13%, user decision 2026-07-27)
- EM target = answer + aug_answer aliases (deduped) so any alias counts correct
- emits schema identical to the existing nq_hotpotqa train.parquet
- carves a small held-out val split for short during-training validation

Run: python convert_base_to_parquet.py --src <base.jsonl> --out-dir <dir> [--val 512 --seed 0]
"""
import argparse, json, random
from pathlib import Path
import pandas as pd

SYS = "You are a helpful and harmless assistant."
DROP_SOURCES = {"compose_chain_qa_invalid"}


def targets(row):
    t = []
    a = row.get("answer")
    t += a if isinstance(a, list) else [a]
    aug = row.get("aug_answer") or []
    t += aug if isinstance(aug, list) else [aug]
    seen, out = set(), []
    for x in t:
        if x is None:
            continue
        x = str(x).strip()
        if x and x.lower() not in seen:
            seen.add(x.lower()); out.append(x)
    return out


def make_row(row, idx, split, data_source):
    q = row["question"]
    tgt = targets(row)
    gt = {"target": tgt}
    return {
        "data_source": data_source,
        "prompt": [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
        "ability": "fact-reasoning",
        "reward_model": {"ground_truth": gt, "style": "rule"},
        "extra_info": {"index": idx, "need_tools_kwargs": True, "question": q, "split": split,
                        "tools_kwargs": {"search": {"create_kwargs": {
                            "ground_truth": gt, "question": q, "data_source": data_source}}}},
        "env_kwargs": {"ground_truth": gt, "question": q, "data_source": data_source},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-source", default="asearcher_base")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.src, encoding="utf-8")]
    kept = [r for r in rows if r.get("source") not in DROP_SOURCES]
    dropped = len(rows) - len(kept)
    # drop empty-target items (would be all-fail, zero gradient)
    kept = [r for r in kept if targets(r)]
    print(f"total {len(rows)} | dropped invalid {dropped} | kept {len(kept)}")

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    val = kept[:args.val]
    train = kept[args.val:]

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, subset, split in [("train", train, "train"), ("val", val, "val")]:
        recs = [make_row(r, i, split, args.data_source) for i, r in enumerate(subset)]
        df = pd.DataFrame(recs)
        p = out / f"{name}.parquet"
        df.to_parquet(p, index=False)
        naliases = [len(r["reward_model"]["ground_truth"]["target"]) for r in recs]
        print(f"  {name}.parquet: {len(df)} rows -> {p} | aliases/q median "
              f"{sorted(naliases)[len(naliases)//2]} max {max(naliases)}")


if __name__ == "__main__":
    main()
