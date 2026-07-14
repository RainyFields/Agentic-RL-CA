# Agentic-RL-CA decision log

Append-only. Every entry: date (SF time), decision, why, who (user / assistant-proposed).

## 2026-07-13 (plan freeze)
- Plan frozen after 3 review rounds (`docs/plan.md`). Key user decisions recorded there:
  Qwen3-1.7B instruct non-thinking start; 4B single fallback; full paper-protocol eval;
  ARPO + IGPO excluded; StepSearch secondary tier; SPA/HISR dropped for B2 (Co-ReAct judge).

## 2026-07-14 (bootstrap)
- SP6 uncommitted working tree of `external/verl-agent` committed on branch `sp6-alfworld`
  (commit f14911e) in place; `master` left clean for cloning. (assistant, per plan Phase 0.1)
- Project repo cloned from the local fork; branch `agentic-rl-ca` from `master` @ 2df17d2
  (includes SP6 HCAPO clip + projection commits that are on master). Remotes: `local` =
  ~/xiaoxuan/external/verl-agent, `upstream` = github.com/langfengQ/verl-agent. (assistant)
- Upstream `README.md` moved to `docs/UPSTREAM_README.md`; project README at root. (assistant)
- Method PDFs pinned in `docs/papers/` (gitignored): HCAPO=2603.08754, CARL=2512.04949,
  StepSearch=2505.15107, survey=2604.09459. (assistant)
- `gamma=1.0` for all arms (Search-R1 protocol; upstream search example's 0.95 not adopted —
  plan Phase 2b already asserts gamma=1.0 for CARL; matched config requires one value).
  (assistant-proposed; flag to user at Wave-0 review)
- `enable_thinking=False` via `+data.apply_chat_template_kwargs.enable_thinking=False`
  (native rollout_loop support, same as upstream qwen3 examples). (assistant)
- Matched data budget: EVERY arm (incl. PPO/critic arms) uses train_batch 256 × group 5
  = 1280 trajectories/step — identical prompts + rollout count across arms; PPO simply gets
  5 independent trajectories per prompt. (assistant-proposed)
- TOTAL_STEPS default 500 (matches tr1 reference budget); confirm with user before Wave 1.
- CARL constants locked from PDF with three deltas vs plan expectations (see
  `docs/methods_note.md`): N₀ ∈ {1 (Lite), 8 (best)} not N/8=2; resume sampling is
  STOCHASTIC from π_θ (not temp 0 — unbiasedness proof requires it); non-critical edges
  are DROPPED from the update set (Eq. 13), not zeroed → Phase-2b implements drop as
  primary, zeroing as ablation. Plan text updated only via this log (plan stays frozen).
- StepSearch: γ_key, GAE λ/γ, max turns absent from PDF — take from their released code
  when Phase 2c starts (secondary tier).


## 2026-07-13 PM (Phase 1 GPU bring-up)
- Phase-1 worker launched (user OK 2026-07-13): 8xH100 `arlca-phase1` — env build, retriever,
  base-model val_2048 gate, 7-condition toy gate. Ledger: worker_logs/workers.tsv.
- Project venv `~/xiaoxuan/envs/agentic-rl-ca` (NOT SP6's verl-agent venv — editable installs
  must not cross projects). (assistant)
- Checkpoint retention: full actor+critic save ~40GB (tr1-measured) x 8 concurrent runs
  overwhelms the ~335G free HDFS quota -> SAVE_FREQ=50, rolling MAX_CKPT_KEEP=1 (resume +
  final retained). COST: the pre-registered secondary "best val-selected checkpoint" would
  need milestone model-only HF exports (~3.4G each) to be added, or a rerun later.
  FLAGGED TO USER before Wave 1. (assistant-proposed)
- Checkpoint retention RESOLVED (user, 2026-07-13): keep rolling MAX_CKPT_KEEP=1, NO
  milestone HF exports. The secondary "best val-selected checkpoint" comparison, if needed,
  will be produced by a LATER RERUN of the selected step (resume/retrain to that step),
  not by storing intermediates. Headline metrics (fixed-budget FINAL + val-AUC) unaffected.

## 2026-07-13 late PM (protocol decision — USER)
- MAIN PROTOCOL LOCKED (user): Qwen3-1.7B, thinking mode, max_response_length=2048
  (configs/protocol_4turn_think2k.sh), TOTAL_STEPS=500, gamma=1.0. Evidence: thinking mode
  fills <think> (85%) and lifts base val_2048 macro-EM 0.054 -> 0.157; 1024 budget capped
  15% of turns (pre-registered truncation rule -> raise to 2048); base eval at 2048 clean
  (macro 0.156). 4B remains the pre-registered fallback only. Search-invocation ~0.49 at
  base accepted-with-logging (RL vs EM is expected to raise it; scaling to 4B did not help
  this criterion: 0.48).
- Toy-gate hard review: user delegated to assistant summaries + dumps on disk
  (outputs/search_toy*/trajectory_*.md; HDFS mirror). All 7 algorithm paths live-validated.
- RETRIEVAL BUG (2026-07-13, critical): upstream SkyRL search tool posts {"query": ...}
  but the Search-R1 retrieval_server expects {"queries": [...]} -> EVERY search in every
  toy-gate/base-eval run 422'd; all EM numbers above are PARAMETRIC-ONLY (no retrieval).
  Fixed in tools/search.py (payload now plural). All gate evals + Wave 0 restarted with
  working retrieval; stale outputs renamed *_NORETRIEVAL (HDFS backup exists). The
  thinking-mode protocol lock stands (thinking benefits + truncation statistics are
  retrieval-independent in direction) pending the re-gate numbers.

## 2026-07-14 (Wave-0 base-model gate — FINAL, retrieval-backed, locked protocol)
Re-gate with working retrieval (worker arlca-toyg-think17-r2kf, all 7 conditions):
- (a) base val_2048 macro-EM 0.2246 (micro 0.2524; 1hop 0.301, mhop 0.167) >= 0.03: PASS.
- (b) valid-action 0.885-0.945 across conditions (median ~0.94; boundary noise): PASS-marginal;
  both-tags is the dominant invalid pattern, penalized (coef 0.01), expected to train away.
- (c) truncation 3.0-4.5% at response 2048 (was 15% @1024): FAILS the strict <0.1% rule —
  ACCEPTED-WITH-MONITORING deviation (user-locked protocol); training tripwire: rate must
  fall over training, else revisit.
- (d) frac groups with nonzero reward variance 0.62-0.67 >= 0.2: PASS (strong signal).
- (e) search-invocation 0.57-0.66 >= 0.5: PASS.
VERDICT: proceed at Qwen3-1.7B thinking @2048. Wave 0 (token_grpo s0) training; step-0 val
macro-EM 0.234 agrees with base eval.
- B1 signal REAL with retrieval: 24.8% of trajectories B1-positive (515/2080).
- shuffle_active_frac = 0.035 << 0.30 on the BASE distribution -> PRE-REGISTERED PLAN B
  FIRES: 8-turn horizon stress test PROMOTED TO REQUIRED (configs/protocol_8turn_think2k.sh);
  re-check active_frac on Wave-1 training batches (logged per step) as RL lengthens
  trajectories.
