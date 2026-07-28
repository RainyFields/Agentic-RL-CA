#!/usr/bin/env python3
"""Phase 3.3 — build the H_gold (required-hops) annotation table for the 7-task SearchQA eval.

Our processed eval data (data/searchR1_processed_direct/test.parquet) was stripped to
question+gold+global-index by the Search-R1 packing (qa_search_test_merge.py's remove_columns).
We re-attach the required-hops axis from the ORIGINAL FlashRAG QA splits (data/hop_annotations/,
downloaded from RUC-NLPIR/FlashRAG_datasets). The join is POSITIONAL and exact: within each
data_source block our rows are the FlashRAG split in original order (proven 51,713/51,713 question
match against the packed source). local_pos = global_index - block_start[data_source].

H_gold (required reasoning hops, a property of the item — ordinal, NOT "expected #searches"):
  nq, triviaqa, popqa            -> 1   (single-hop by construction)
  hotpotqa, bamboogle            -> 2   (2-hop by construction)
  musique                        -> len(question_decomposition)  in {2,3,4}
  2wikimultihopqa                -> 2 if type in {compositional,comparison,inference}, 4 if bridge_comparison
Secondary splits kept for A5: hotpot_type (bridge/comparison), wiki_type (2wiki 4 types),
popqa s_pop/o_pop popularity (+ head/tail bucket at the median).

Output: data/hop_annotations/hop_table.parquet, keyed by the global `index`, columns:
  index, data_source, H_gold, hotpot_type, wiki_type, popqa_s_pop, popqa_o_pop, popqa_pop_bucket
Usage: python build_hop_annotations.py   (writes the table + prints strata sizes + join-validation)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = "/home/tiger/xiaoxuan/Agentic-RL-CA"
TESTPARQUET = f"{REPO}/data/searchR1_processed_direct/test.parquet"
ANNDIR = f"{REPO}/data/hop_annotations"
OUT = f"{ANNDIR}/hop_table.parquet"

# Verified block starts (global index of the first row of each data_source block).
BLOCK_START = {"nq": 0, "triviaqa": 3610, "popqa": 14923, "hotpotqa": 29190,
               "2wikimultihopqa": 36595, "musique": 49171, "bamboogle": 51588}
H_FIXED = {"nq": 1, "triviaqa": 1, "popqa": 1, "hotpotqa": 2, "bamboogle": 2}
WIKI_TYPE_HOPS = {"compositional": 2, "comparison": 2, "inference": 2, "bridge_comparison": 4}
FR_FILE = {"musique": "musique/dev.jsonl", "2wikimultihopqa": "2wikimultihopqa/dev.jsonl",
           "hotpotqa": "hotpotqa/dev.jsonl", "popqa": "popqa/test.jsonl"}


def _norm(q):
    """Normalize a question for join validation: our stored questions were .strip()'d with a
    forced trailing '?', so compare case/space-insensitively ignoring trailing punctuation."""
    return " ".join(str(q).lower().strip().rstrip("?.! ").split())


def load_flashrag(ds):
    return [json.loads(l) for l in open(f"{ANNDIR}/{FR_FILE[ds]}")]


def main():
    t = pq.read_table(TESTPARQUET, columns=["data_source", "extra_info"])
    ds_col = t["data_source"].to_pylist()
    ei = t["extra_info"].to_pylist()
    n = len(ds_col)
    idx = np.array([e["index"] for e in ei])
    ques = [e.get("question") for e in ei]
    assert (idx == np.arange(n)).all(), "global index is not the 0..N-1 row position"

    rows = []
    frcache = {}
    match_stats = {}
    for i in range(n):
        ds = ds_col[i]
        gi = int(idx[i])
        rec = {"index": gi, "data_source": ds, "H_gold": None, "hotpot_type": None,
               "wiki_type": None, "popqa_s_pop": None, "popqa_o_pop": None, "popqa_pop_bucket": None}
        if ds in H_FIXED and ds not in FR_FILE:      # nq/triviaqa/bamboogle: fixed, no source file
            rec["H_gold"] = H_FIXED[ds]
            rows.append(rec)
            continue
        if ds not in frcache:
            frcache[ds] = load_flashrag(ds)
            match_stats[ds] = [0, 0]
        src = frcache[ds]
        lp = gi - BLOCK_START[ds]
        s = src[lp]
        md = s.get("metadata", {})
        # join validation
        m0, m1 = match_stats[ds]
        match_stats[ds] = [m0 + 1, m1 + (1 if _norm(ques[i]) == _norm(s.get("question", "")) else 0)]
        if ds == "musique":
            rec["H_gold"] = len(md.get("question_decomposition", []))
        elif ds == "2wikimultihopqa":
            wt = md.get("type")
            rec["wiki_type"] = wt
            rec["H_gold"] = WIKI_TYPE_HOPS.get(wt)
        elif ds == "hotpotqa":
            rec["H_gold"] = 2
            rec["hotpot_type"] = md.get("type")
        elif ds == "popqa":
            rec["H_gold"] = 1
            rec["popqa_s_pop"] = md.get("s_pop")
            rec["popqa_o_pop"] = md.get("o_pop")
        rows.append(rec)

    df = pd.DataFrame(rows)
    # popqa head/tail bucket at the median s_pop (balanced split; raw s_pop kept for finer bins)
    pop = df["data_source"] == "popqa"
    med = float(df.loc[pop, "popqa_s_pop"].median())
    df.loc[pop, "popqa_pop_bucket"] = np.where(df.loc[pop, "popqa_s_pop"].astype(float) >= med, "head", "tail")

    # ---- validation ----
    print("=== positional-join question match (should be ~1.000) ===")
    ok = True
    for ds, (tot, mm) in sorted(match_stats.items()):
        r = mm / tot
        print(f"  {ds:18s} {mm}/{tot} = {r:.4f}")
        ok = ok and r > 0.98
    assert ok, "positional join failed for some data_source (<0.98 question match)"
    assert df["H_gold"].notna().all(), f"{df['H_gold'].isna().sum()} rows missing H_gold"

    print("\n=== H_gold strata (pooled) ===")
    print(df["H_gold"].value_counts().sort_index().to_string())
    print("\n=== per-task H_gold decomposition ===")
    print(pd.crosstab(df["data_source"], df["H_gold"]).to_string())
    print(f"\npopqa s_pop median (head/tail cut) = {med:.0f}")
    print("hotpot_type:", df.loc[df.data_source == 'hotpotqa', 'hotpot_type'].value_counts().to_dict())
    print("wiki_type:", df.loc[df.data_source == '2wikimultihopqa', 'wiki_type'].value_counts().to_dict())

    df.to_parquet(OUT, index=False)
    print(f"\nwrote {len(df)} rows -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
