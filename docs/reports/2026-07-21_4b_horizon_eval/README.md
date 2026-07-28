# Credit assignment for multi-turn agents — AlfWorld + SearchQA (2026-07-22)

**`report.pdf`** — the final cross-environment report (6 pp). Compares four credit-assignment schemes
(**GRPO, GiGPO, turn-PPO, token-PPO**; + pre-RL floor) across two environments with very different
horizons: **AlfWorld** (embodied, ~30–50 turns, Qwen3-**1.7B**, from the `reward_models` project) and
**SearchQA** (agentic retrieval QA, 4-turn cap, Qwen3-**4B**, this project — full 51,713-question,
7-task test set, greedy, `4turn_think2k`).

**Headline:** the ranking of credit-assignment methods **reverses with horizon**. Long-horizon
AlfWorld → per-turn methods win (turn-PPO 0.963, GiGPO 0.955 > GRPO 0.881); short-horizon SearchQA →
flat group-relative wins (GRPO 0.441 ≈ GiGPO 0.446 > turn-PPO 0.389 > token-PPO 0.381). The only
positive SearchQA horizon-gain is the *critic* methods on MuSiQue's deep sequential chains (turn-PPO
+0.031). Scale is mixed by design (AlfWorld 1.7B / SearchQA 4B) — cross-env claims are about the
*pattern*, not levels; see report §Limitations.

**Status:** COMPLETE — all 5 SearchQA evals + the AlfWorld campaign done; figures + `report.pdf` built.

## Pipeline / rebuild

Two pythons: the project venv for the eval; a plotting python (matplotlib) for the analysis.

```bash
REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
TRAINPY=$REPO/../envs/agentic-rl-ca/bin/python        # torch/vllm/verl (runs the eval)
PLOTPY=$REPO/../envs/verl-agent/bin/python            # matplotlib+pandas+pyarrow (runs the analysis)
cd $REPO/docs/reports/2026-07-21_4b_horizon_eval/assets

# 1. re-attach the required-hops axis (downloads 4 small FlashRAG QA splits, ~120MB, once)
$TRAINPY build_hop_annotations.py                     # -> data/hop_annotations/hop_table.parquet

# 2. full-set eval of each checkpoint (on an 8xH100 worker; emits the §4 per-episode JSONL)
#    launched via scripts/wave_wrappers/.arlca-eval4b-*.sh -> scripts/worker_eval.sh, which calls:
#    scripts/eval_search_full.sh <ckpt/global_step_500> <label> <cond> 4turn_think2k
#    outputs -> outputs/eval_full/<label>/val_trajectories_step0.jsonl  (+ paper_table.{csv,json})

# 3. build every report figure + table (pure post-processing, no rollouts)
$PLOTPY make_fig1.py            # Fig 1  — training curves, both environments
$PLOTPY alfworld_figures.py     # Fig 2  — AlfWorld turn barplot (+ AlfWorld eval rows, episodes.csv)
$PLOTPY make_remaining_figs.py  # Fig 3 (box), Fig 4 (turns-vs-gold), Fig 5 (acc-vs-turns), Table 1
$PLOTPY horizon_figures.py      # supporting A1–A5 (EM-vs-hops, horizon-gain CIs) + verdicts.txt
$PLOTPY horizon_common.py <label>   # quick per-label sanity (coverage guard, per-stratum EM)

# 4. compile
cd $REPO/docs/reports/2026-07-21_4b_horizon_eval && tectonic report.tex   # -> report.pdf
```

The **AlfWorld** side (`alfworld_common.py`) reads the completed `reward_models` 2026-07-13 campaign:
per-episode turns+success from `…/alfworld_prm_gigpo/spa_data/unseen_cmp_{grpo_265,gigpo,ppo}/rollout_log.jsonl`,
training curves from `reward_models/docs/reports/2026-07-13_credit_assignment_comparison_assets/history_*.csv`,
eval accuracy from `unseen_results.csv`. token-PPO is SearchQA-only (not run on AlfWorld).

## What was built (design decisions)

- **§4 per-episode logging.** The eval harness already dumped a per-trajectory JSONL; extended
  `credit_assignment/eval_metrics.py` + `verl/trainer/ppo/ray_trainer.py::_validate` to add
  `n_search_calls` (tool-call count), `tokens_generated`, the global dataset `index`, and a
  recomputable truncation signal. All additions are **gated on `trainer.validation_data_dir`** (only
  the full-set eval sets it), so training-time validation is byte-identical — safe for the live
  Wave-2 workers on crash-resume.
- **Join key = global `index`, not question.** `env_kwargs` (which carries the question text) is
  consumed at `envs.reset()` and never reaches the dump; `data_source` survives because it is never
  consumed. So the global `extra_info.index` is threaded through the rollout the same way — giving an
  **exact integer join** (51,713 unique keys, no question-text ambiguity). `load_eval` cross-checks
  `index→annotation.data_source` against the dump's `data_source` and **warns on mismatch** (this is
  how a wrong index — e.g. val_2048's local 0..N-1 — is caught automatically).
- **H_gold recovery.** Search-R1 packing (`qa_search_test_merge.py`) stripped the datasets to
  question+gold+index. The required-hops axis is re-attached from the original FlashRAG dev/test
  splits (`RUC-NLPIR/FlashRAG_datasets`) by a **positional join proven exact** (51,713/51,713
  question match against the packed source). `build_hop_annotations.py` validates the join at build
  time (asserts >0.98 per-dataset question match).
  - `H_gold`: NQ/TriviaQA/PopQA = 1; HotpotQA/Bamboogle = 2; MuSiQue = `len(question_decomposition)`
    ∈ {2,3,4}; 2Wiki = `type` (bridge_comparison→4, else→2).
  - Secondary splits (A5): HotpotQA `type` (bridge/comparison), PopQA `s_pop` head/tail at the median.
- **Truncation recomputed from EM+turns.** The parse-status-based `answered` flag disagrees with EM
  (Qwen3 `<think>` blocks can score a correct answer while parse_status≠'answer'), so `load_eval`
  derives `hit_cap = turns≥cap` and `cap_fail = (em==0) & hit_cap` (the "capacity-limited failure"
  signal) rather than trusting the dumped flag.

## Strata (pooled; all ≫ 50-item minimum)

| H_gold | n | tasks |
|---|---|---|
| 1 | 29,190 | NQ, TriviaQA, PopQA |
| 2 | 18,607 | HotpotQA, Bamboogle, 2Wiki(compositional/comparison/inference), MuSiQue-2hop |
| 3 | 760 | MuSiQue-3hop |
| 4 | 3,156 | MuSiQue-4hop, 2Wiki(bridge_comparison) |

## Provenance

- **Checkpoints:** `…/agentic_rl_ca/checkpoints/{token_grpo,gigpo}_qwen3-4b_4turn_think2k_s0/global_step_500`
  (FINAL @500, the pre-registered primary); floor = base `Qwen3-4B` (untrained, same protocol).
- **Metric:** per-question EM with alias normalization (skyrl `search/utils.py`), greedy, 4-turn cap,
  top-3 wiki-18 retrieval — identical harness for all checkpoints and all 7 tasks.
- **Eval data:** `data/searchR1_processed_direct/test.parquet` (51,713 rows). Hop annotations:
  `data/hop_annotations/` (FlashRAG dev/test) + `hop_table.parquet`.

## Findings (see `report.pdf` for the full write-up)

1. **The method ranking reverses across environments** (Table 1). GRPO is the *worst* trained arm on
   AlfWorld (0.881) but *best-tied* on SearchQA (0.441); the critic-PPO arms flip the other way.
2. **RL on SearchQA mostly teaches turn usage.** Floor under-searches (2 turns/1 search, 0.313);
   RL ~doubles it for +0.13 macro. GiGPO then *over*-searches into the 4-turn cap (higher cap-fail
   than GRPO at every stratum) → ties GRPO on EM without a horizon-gain.
3. **Per-turn credit's value scales with genuine sequential depth.** A2 horizon-gain vs GRPO: the
   only positive signal is the *critic* methods on MuSiQue (turn-PPO **+0.031** [+0.003,+0.057]),
   negated by over-search losses on shallow 2Wiki comparison (−0.217). On AlfWorld the per-turn
   methods win outright by keeping trajectories short (GRPO 14.9 turns, incl. `look`-loops, vs 9.1).
4. **MuSiQue collapses with hops (0.27→0.05) while 2Wiki rises (0.43→0.50)** — because MuSiQue is
   genuinely sequential (cap-limited at 4 turns) and 2Wiki-4hop is a shallow, ~50%-base-rate
   comparison. The difficulty structure is present in the untrained floor too (report §6).

## Caveats (carried into the report limitations)

- **H_gold is ordinal, not "expected #searches"** — annotated reasoning chain length, not the
  searches a given model needs (parametric knowledge can shortcut; a weak retriever can inflate).
  A3 therefore measures each method's H_used-vs-H_gold slope rather than assuming turns should equal hops.
- **The 3-hop stratum is MuSiQue-only** — 2Wiki has no native 3-hop class (its 4 types are 2-hop or
  the 4-hop bridge_comparison), so within-2Wiki curves span {2,4}; within-MuSiQue span {2,3,4}.
- **Pooled cross-stratum comparisons are corroboration only** — pooled strata change task composition
  (H1 = NQ/TriviaQA/PopQA; H4 = MuSiQue/2Wiki), so headline horizon claims come from the within-task
  (MuSiQue, 2Wiki) curves where task identity is held fixed.
- **Metric = val-protocol EM on the full test set**, greedy; comparable to the Wave-1 val_2048 proxy
  and to tr1's Search-R1 numbers with the usual proxy caveat.
