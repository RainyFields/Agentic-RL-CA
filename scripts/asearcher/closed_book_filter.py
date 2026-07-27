#!/usr/bin/env python3
"""Closed-book (stage-1) filter for ASearcher splits.

For each question, generate n tool-free answers with the base model (vLLM offline) and
sub-EM-check against answer + aug_answer aliases. Questions the model answers WITHOUT search
(parametric memory) are dropped; what remains genuinely requires retrieval.

Outputs under --out-dir/<split>/:
  scored.jsonl      per-question: pass_count (0..n), sample answers
  stats.json        pass-count histogram + drop fractions at thresholds
  keep_ge2.jsonl    KEEP if pass_count < 2 (drop items solved >=2/n closed-book) [primary]
  keep_ge1.jsonl    KEEP if pass_count == 0 (strict: drop anything ever solved closed-book)
Run: python closed_book_filter.py --input <jsonl> --split <name> --out-dir <dir> --model <path> [--n 4]
"""
import argparse, json, os, re, string
from pathlib import Path

# --- EM / sub-EM (matches the fork's ASearcher/utils/rewards.py) ---
def _norm(s):
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())

def subem(pred, golds):
    p = _norm(pred)
    return any(_norm(g) and _norm(g) in p for g in golds)

def targets(row):
    t = []
    a = row.get("answer"); t += a if isinstance(a, list) else [a]
    aug = row.get("aug_answer") or []; t += aug if isinstance(aug, list) else [aug]
    return [str(x) for x in t if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    rows = [r for r in rows if targets(r)]
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.out_dir) / args.split
    out.mkdir(parents=True, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    SYS = ("You are a knowledgeable assistant. Answer the question directly with a short, "
           "factual answer (a few words). If unsure, give your single best guess.")
    prompts = []
    for r in rows:
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": r["question"]}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    llm = LLM(model=args.model, tensor_parallel_size=int(os.environ.get("TP", "8")),
              gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=False)
    sp = SamplingParams(n=args.n, temperature=args.temp, top_p=0.95, max_tokens=args.max_tokens)
    outs = llm.generate(prompts, sp)

    scored, hist = [], {}
    for r, o in zip(rows, outs):
        gold = targets(r)
        answers = [c.text.strip() for c in o.outputs]
        passes = sum(1 for a in answers if subem(a, gold))
        hist[passes] = hist.get(passes, 0) + 1
        scored.append({"question": r["question"], "answer": r.get("answer"),
                        "aug_answer": r.get("aug_answer"), "source": r.get("source"),
                        "pass_count": passes, "n": args.n, "sample_answers": answers[:2]})

    with open(out / "scored.jsonl", "w") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    total = len(scored)
    drop_ge1 = sum(1 for s in scored if s["pass_count"] >= 1)
    drop_ge2 = sum(1 for s in scored if s["pass_count"] >= 2)
    stats = {"split": args.split, "total": total, "n": args.n,
             "pass_hist": {str(k): hist.get(k, 0) for k in range(args.n + 1)},
             "parametric_frac_ge1": round(drop_ge1 / total, 4),
             "parametric_frac_ge2": round(drop_ge2 / total, 4),
             "keep_ge2_count": total - drop_ge2, "keep_ge1_count": total - drop_ge1}
    json.dump(stats, open(out / "stats.json", "w"), indent=2)
    # keep-sets (write the ORIGINAL rows so the parquet converter can consume them)
    row_by_q = {r["question"]: r for r in rows}
    with open(out / "keep_ge2.jsonl", "w") as f2, open(out / "keep_ge1.jsonl", "w") as f1:
        for s in scored:
            orig = row_by_q[s["question"]]
            if s["pass_count"] < 2: f2.write(json.dumps(orig, ensure_ascii=False) + "\n")
            if s["pass_count"] == 0: f1.write(json.dumps(orig, ensure_ascii=False) + "\n")
    print(f"[{args.split}] total {total} | parametric >=1/{args.n}: {stats['parametric_frac_ge1']:.1%} "
          f"| >=2/{args.n}: {stats['parametric_frac_ge2']:.1%} | keep(<2) {stats['keep_ge2_count']} "
          f"| keep(0) {stats['keep_ge1_count']}")


if __name__ == "__main__":
    main()
