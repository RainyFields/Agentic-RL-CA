#!/usr/bin/env python3
"""ASearcher dataset inspection (Wave-0 M7, rung 3).

Downloads inclusionAI/ASearcher-train-data + ASearcher-test-data from the HF hub
(raw files; the HF dataset viewer is broken for these repos) and reports:
  - split names, exact row counts, field schemas
  - source-tag distribution, question-length (complexity proxy) stats
  - presence/absence of trajectory fields (the key question: no expert rollouts)
  - eval-benchmark inventory + reference-step annotations
  - leakage vectors vs our rung-1 eval benchmarks

Outputs under --out: train_stats.json, test_stats.json, sample_rows.json, summary.md
Run: python inspect_dataset.py --out <dir> [--cache <dl_dir>]
No HF token needed (public). Uses only stdlib + urllib.
"""
import argparse, json, os, statistics, urllib.request
from collections import Counter
from pathlib import Path

HF = "https://huggingface.co/datasets/inclusionAI"
TRAIN_FILES = ["ASearcher-Base-35k.jsonl", "ASearcher-LRM-35k.jsonl"]
TEST_SUBSETS = {  # subdir -> filename
    "GAIA": "test.json", "frames": "test.json", "xbench-deepsearch": "test.jsonl",
    "Bamboogle": "test.jsonl", "2WikiMultihopQA_rand1000": "test.jsonl",
    "HotpotQA_rand1000": "test.jsonl", "Musique_rand1000": "test.jsonl",
    "NQ_rand1000": "test.jsonl", "PopQA_rand1000": "test.jsonl",
    "TriviaQA_rand1000": "test.jsonl",
}
TRAJ_HINTS = ("traj", "turn", "step", "tool", "search", "thought", "rollout",
              "trace", "messages", "conversation", "action", "observation")


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    urllib.request.urlretrieve(url, dest)
    return dest


def load_any(path):
    txt = open(path, encoding="utf-8").read().strip()
    if not txt:
        return []
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError:
        return [json.loads(l) for l in txt.splitlines() if l.strip()]


def traj_fields(row):
    return [k for k in row if any(h in k.lower() for h in TRAJ_HINTS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache or (out / "_dl")); cache.mkdir(parents=True, exist_ok=True)

    # ---- train ----
    train_stats, samples = {}, {}
    for f in TRAIN_FILES:
        p = fetch(f"{HF}/ASearcher-train-data/resolve/main/{f}", str(cache / f))
        rows = load_any(p)
        qlens = [len(r["question"].split()) for r in rows]
        srcs = Counter(r.get("source", "(no source field)") for r in rows)
        schemas = Counter(tuple(sorted(r.keys())) for r in rows)
        tf = traj_fields(rows[0])
        st = {
            "rows": len(rows),
            "fields": list(rows[0].keys()),
            "schema_variants": {", ".join(k): v for k, v in schemas.items()},
            "source_distribution": dict(srcs.most_common()),
            "answer_type": Counter(type(r.get("answer")).__name__ for r in rows).most_common(),
            "question_words": {"min": min(qlens), "median": int(statistics.median(qlens)),
                                "mean": round(statistics.mean(qlens), 1),
                                "p95": sorted(qlens)[int(len(qlens) * 0.95)], "max": max(qlens)},
            "trajectory_fields": tf or "NONE",
        }
        if "aug_answer" in rows[0]:
            al = [len(r["aug_answer"]) for r in rows if isinstance(r.get("aug_answer"), list)]
            st["aug_answer_aliases"] = {"median": int(statistics.median(al)), "max": max(al)}
        train_stats[f] = st
        samples[f] = rows[:2]

    # ---- test ----
    test_stats = {}
    for sub, fn in TEST_SUBSETS.items():
        try:
            p = fetch(f"{HF}/ASearcher-test-data/resolve/main/{sub}/{fn}", str(cache / f"test_{sub}.json"))
            rows = load_any(p)
            test_stats[sub] = {"rows": len(rows), "fields": list(rows[0].keys()),
                                "trajectory_fields": traj_fields(rows[0]) or "NONE",
                                "has_reference_steps": any(k in rows[0] for k in
                                    ("reference_steps", "Annotator_Metadata", "Steps"))}
            samples[f"test/{sub}"] = rows[:1]
        except Exception as e:  # noqa: BLE001
            test_stats[sub] = {"error": repr(e)}

    (out / "train_stats.json").write_text(json.dumps(train_stats, indent=2, ensure_ascii=False))
    (out / "test_stats.json").write_text(json.dumps(test_stats, indent=2, ensure_ascii=False))
    (out / "sample_rows.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False))

    with open(out / "summary.md", "w") as fh:
        fh.write("# ASearcher dataset inspection — machine summary\n\n## Train (ASearcher-train-data)\n")
        for f, st in train_stats.items():
            fh.write(f"\n### {f} — {st['rows']} rows\n- fields: {st['fields']}\n"
                     f"- trajectory fields: {st['trajectory_fields']}\n"
                     f"- question words: {st['question_words']}\n"
                     f"- sources: {st['source_distribution']}\n")
        fh.write("\n## Test (ASearcher-test-data)\n\n| subset | rows | ref-steps | traj |\n|---|---|---|---|\n")
        for sub, st in test_stats.items():
            if "error" in st:
                fh.write(f"| {sub} | ERR | | |\n"); continue
            fh.write(f"| {sub} | {st['rows']} | {st['has_reference_steps']} | {st['trajectory_fields']} |\n")
    print("wrote", out / "summary.md")


if __name__ == "__main__":
    main()
