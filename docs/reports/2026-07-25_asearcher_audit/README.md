# ASearcher dataset inspection (Wave-0 M7, rung 3) — report package

**Deliverable:** `asearcher_inspection.pdf` — answers the three questions (turns / expert trajectories /
train-eval split) plus schemas, source distributions, leakage vectors, and rung-3 challenges. Source
`asearcher_inspection.tex` (compile with `tectonic asearcher_inspection.tex`).

## Headline findings
- **Turns:** none in the data (QA pairs only). Horizon is a training-time budget — paper uses 32 turns
  (7B/14B) / 128 turns (QwQ-32B); our rung-3 caps at 30.
- **Expert trajectories:** NONE. No rollouts, no SFT/cold-start dataset. ASearcher trains RL directly from
  base. (Only human *reference-solution outlines* exist, on GAIA/xbench eval sets — prose plans, eval-only.)
  ⇒ our rung-3 SFT π_base must be self-generated.
- **Train/eval split:** separate repos. `ASearcher-train-data` = 2 training splits (Base-35k=35,583 rows;
  LRM-35k=35,054 rows; ~70.6k QA pairs, no held-out test). `ASearcher-test-data` = 10 eval benchmarks
  (7,277 items: GAIA 103, frames 824, xbench 100, Bamboogle 125, +6×rand1000).
- **Leakage vectors:** LRM contains 2,572 `webwalker_hard` items (WebWalkerQA is a rung-3 eval); the
  eval repo *is* HotpotQA/2Wiki/Musique/NQ/TriviaQA/PopQA/Bamboogle (our rung-1 families) → fuzzy dedup
  required, use our own rung-1 splits.
- **Environment:** wiki-18 local-RAG (same corpus we host) vs live-web; strongest published numbers are
  live-web QwQ-32B. Our local-RAG/30-turn/4B config is harder → M7 solvability diagnostic is the go/no-go gate.

## Rerun
```
source ~/xiaoxuan/envs/sw_inspect/bin/activate
python scripts/asearcher/inspect_dataset.py --out <dir>/assets   # public HF, no token
tectonic asearcher_inspection.tex
```

## assets/
- `train_stats.json` — per-split rows, fields, source distribution, question-length stats, trajectory-field scan
- `test_stats.json` — per-benchmark row counts + reference-step presence
- `sample_rows.json` — sample rows from every split/subset
- `summary.md` — machine-generated summary
(raw `.jsonl` not committed — ~27 MB, public at huggingface.co/datasets/inclusionAI/ASearcher-train-data)

Paper: arXiv 2508.07976 "Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale
Asynchronous RL" (Qwen2.5-7B/14B + QwQ-32B).
