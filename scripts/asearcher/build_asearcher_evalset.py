#!/usr/bin/env python3
"""Build the ASearcher-paper evaluation suite as a verl parquet.

Source: inclusionAI/ASearcher-test-data (public) — 10 benchmarks, 7,277 items:
  wiki-answerable   : NQ_rand1000, TriviaQA_rand1000, PopQA_rand1000, HotpotQA_rand1000,
                      2WikiMultihopQA_rand1000, Musique_rand1000, Bamboogle
  live-web-oriented : GAIA, frames, xbench-deepsearch
Each row is tagged data_source=<benchmark>, so verl's val path emits per-benchmark
EM/turn metrics with no extra plumbing.

IMPORTANT (report this alongside any numbers): GAIA / frames / xbench assume a LIVE WEB
agent. Our environment retrieves from the offline wiki-18 corpus, so those three are
largely unanswerable by construction and their scores measure the corpus gap, not the
credit-assignment method. Read them separately from the wiki-answerable seven.

Usage:
  python build_asearcher_evalset.py --out-dir <dir> [--repeat 1] [--limit-per-bench N]
`--repeat k` replicates every row k times, which is how we run Avg@k at temperature
(SearchMultiProcessEnv.reset rejects val_kwargs.n>1 beyond the env pool — data
replication is the supported path; see the 2026-07-28 pass-rate diagnostic).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "inclusionAI/ASearcher-test-data"
FILES = {
    "NQ_rand1000": "NQ_rand1000/test.jsonl",
    "TriviaQA_rand1000": "TriviaQA_rand1000/test.jsonl",
    "PopQA_rand1000": "PopQA_rand1000/test.jsonl",
    "HotpotQA_rand1000": "HotpotQA_rand1000/test.jsonl",
    "2WikiMultihopQA_rand1000": "2WikiMultihopQA_rand1000/test.jsonl",
    "Musique_rand1000": "Musique_rand1000/test.jsonl",
    "Bamboogle": "Bamboogle/test.jsonl",
    "GAIA": "GAIA/test.json",
    "frames": "frames/test.json",
    "xbench-deepsearch": "xbench-deepsearch/test.jsonl",
}
WIKI_ANSWERABLE = {"NQ_rand1000", "TriviaQA_rand1000", "PopQA_rand1000", "HotpotQA_rand1000",
                   "2WikiMultihopQA_rand1000", "Musique_rand1000", "Bamboogle"}
SYS = "You are a helpful and harmless assistant."

QUESTION_KEYS = ("question", "Question", "prompt", "query")
ANSWER_KEYS = ("answer", "Answer", "answers", "golden_answers", "final_answer",
               "true_answer", "label")


def load_rows(path):
    txt = Path(path).read_text(encoding="utf-8").strip()
    if not txt:
        return []
    if path.endswith(".json"):
        data = json.loads(txt)
        return data if isinstance(data, list) else data.get("data", [])
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def pick(row, keys):
    for k in keys:
        if k in row and row[k] not in (None, "", []):
            return row[k]
    return None


def targets(row):
    a = pick(row, ANSWER_KEYS)
    if a is None:
        return []
    vals = a if isinstance(a, list) else [a]
    seen, out = set(), []
    for x in vals:
        if isinstance(x, dict):
            x = x.get("answer") or x.get("value") or ""
        x = str(x).strip()
        if x and x.lower() not in seen:
            seen.add(x.lower())
            out.append(x)
    return out


def make_row(q, tgt, idx, bench):
    gt = {"target": tgt}
    return {
        "data_source": bench,
        "prompt": [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
        "ability": "fact-reasoning",
        "reward_model": {"ground_truth": gt, "style": "rule"},
        "extra_info": {"index": idx, "need_tools_kwargs": True, "question": q, "split": "test",
                       "tools_kwargs": {"search": {"create_kwargs": {
                           "ground_truth": gt, "question": q, "data_source": bench}}}},
        "env_kwargs": {"ground_truth": gt, "question": q, "data_source": bench},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repeat", type=int, default=1, help="replicate each row k times (Avg@k)")
    ap.add_argument("--limit-per-bench", type=int, default=0)
    args = ap.parse_args()

    recs, stats = [], {}
    idx = 0
    for bench, rel in FILES.items():
        p = hf_hub_download(REPO, rel, repo_type="dataset")
        rows = load_rows(p)
        kept = 0
        for r in rows:
            q = pick(r, QUESTION_KEYS)
            t = targets(r)
            if not q or not t:
                continue
            if args.limit_per_bench and kept >= args.limit_per_bench:
                break
            for _ in range(args.repeat):
                recs.append(make_row(str(q), t, idx, bench))
                idx += 1
            kept += 1
        stats[bench] = (len(rows), kept)
        print(f"{bench:26s} raw {len(rows):5d} -> usable {kept:5d}"
              f"{'' if bench in WIKI_ANSWERABLE else '   [live-web benchmark]'}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(recs)
    dest = out / f"asearcher_eval{'_x%d' % args.repeat if args.repeat > 1 else ''}.parquet"
    df.to_parquet(dest, index=False)
    wiki = sum(k for b, (_, k) in stats.items() if b in WIKI_ANSWERABLE)
    web = sum(k for b, (_, k) in stats.items() if b not in WIKI_ANSWERABLE)
    print(f"\nwrote {len(df)} rows -> {dest}")
    print(f"  wiki-answerable: {wiki} questions | live-web: {web} questions | repeat={args.repeat}")


if __name__ == "__main__":
    main()
