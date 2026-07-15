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

## 2026-07-14 (Phase 2b.2 CARL tree loop — implementation decisions)
Implemented per plan §2b.2-2b.5 with the methods_note-locked deltas (which supersede two
plan-text expectations; both were pre-flagged in methods_note 2026-07-14):
- Phase-2 resumed turns sample STOCHASTICALLY from pi_theta (plan text said "later turns
  greedy"; the paper's temp-0 appears only in the Eq. 5 preliminary study and the
  unbiasedness argument requires sampling). No per-turn do_sample switching needed.
- Non-critical edges are physically DROPPED from D_upd before adjust_batch (Eq. 13,
  primary mode); zero-advantage loss-exclusion retained as ablation flag
  (algorithm.carl.drop_noncritical=False). Degenerate all-non-critical batch => keep all
  rows at zero advantage (no-op step) rather than crash; logged carl/empty_update_set.
- Node identity: sha1 over the exact raw_prompt_ids pipeline (chat template +
  add_generation_prompt + protocol truncation). Runtime SOFT fidelity counters
  (chain: dst_t == src_{t+1}; resume: restored prompt == snapshot node; merge:
  same-node => byte-identical ids) — logged, not fatal; the hard guarantees live in
  credit_assignment/tests/ (30/30 pass incl. 7 new).
- Vanilla loop refactored into shared _turn_loop (plan-prescribed); behavior intended
  byte-identical — REQUIRES gigpo/grpo search toy smoke on next worker before any Wave-3
  launch relies on this tree (also validates restore fidelity on GPU: resume + same
  action => identical reward/obs).
- env.rollout.n for the carl condition sizes the env pool only (max(n0, n_total-n0));
  group repetition happens inside carl_multi_turn_loop.

## 2026-07-14 ~13:00 PDT (wave-1 tripwire observation + Phase 3b.2 complete)
- shuffle/active_frac on B1-SHUFFLE TRAINING batches has crossed the 0.30 rule during RL:
  b1_shuffle-s1 = 0.447 (step ~79), b1_shuffle-s0 = 0.248 and climbing (step ~61); was
  0.035 on the base distribution at the Wave-0 gate. Interpretation: RL increases search
  invocation/step-reward incidence, so the shuffle control becomes ACTIVE at 4 turns as
  training progresses. OBSERVATION ONLY — whether this un-promotes the REQUIRED 8-turn
  stress test is a user decision at the Wave-1 readout (the Wave-0 promotion was based on
  the base distribution; the pre-registered re-check contemplated exactly this).
- Phase 3b.2 diagnostic GPU runner implemented (diag_runner.py standalone vLLM +
  diag_plan.py pure helpers + run_diag.sh; 33/33 tests). Design decision: base
  trajectories are REGENERATED from the target checkpoint (not parsed from training
  dumps) — guarantees same-policy continuations by construction and gives exact live
  snapshots via the Phase-2b infra; training dumps lack ground_truth anyway. Pending
  GPU: first diagnostic run + CARL toy smoke + vanilla-path (gigpo) smoke after the
  _turn_loop refactor — all need a free worker slot.

## 2026-07-14 ~14:10 PDT (ops fix: allocator fragmentation OOMs on gae_turn arms)
Recurring actor-update CUDA OOMs on wave-1 critic arms as response lengths grow:
b1-s1 x2, b1_shuffle-s0 x2 (twice failed to reach its step-100 checkpoint => livelock
risk), turn_ppo_b0-s1 x1. Every OOM dump shows 16-27 GiB "reserved by PyTorch but
unallocated" (fragmentation) while a 9-11 GiB allocation fails. Fix: export
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in scripts/_common.sh (activate_env).
Allocator-only — no training-semantics change; numerics unaffected. Deploys per-run at
its next crash-resume (healthy processes untouched). Each crash-resume costs up to ~45
steps of redone work (save_freq 50); if OOMs persist AFTER this fix, next lever is
LOGPROB_MICRO/MICRO_BSZ reduction — that changes throughput only but will be raised for
user decision first.

## 2026-07-14 ~14:50 PDT (INCIDENT: expandable_segments revert)
The 14:10 allocator fix backfired: vLLM's CUDA-graph memory pool asserts
"Expandable segments are not compatible with memory pool" (pytorch#147851) at engine
init, so every poisoned crash-resume died at STARTUP instead of fixing the OOMs.
Reverted at 14:36 (428bcb5; _common.sh now explicitly unsets the var). Blast radius:
token_ppo-s0 burned attempts 3-5 (attempt 6 = last, runs clean, original OOM issue
unfixed), b1-s1 burned attempt 4 (attempts 5-6 clean). LESSON (memory updated): allocator
knobs interact with vLLM pools — never hot-deploy allocator changes fleet-wide without a
single-run test. Standing ops plan: if a worker exhausts its 6 attempts, RELAUNCH the
worker (verl resume_mode=auto continues from the last HDFS checkpoint — routine
continuation of the approved wave). Root-cause options for the fragmentation OOMs to
decide at the Wave-1 readout: (a) max_split_size_mb (pool-compatible) tested on ONE run
first, (b) MICRO_BSZ/LOGPROB_MICRO reduction (throughput-only), (c) MAX_ATTEMPTS raise
on future launches. NOTE: worker_train.sh must NEVER be edited while workers run (bash
reads the executing file by offset); _common.sh is safe (sourced fresh per attempt, ms
window).

## 2026-07-14 ~16:10 PDT (b1-s1 OOM livelock: single-run MICRO_BSZ=8 relaunch prepared)
b1-s1 has OOM'd on 5/6 attempts (organic fragmentation OOMs at steps ~62/~74/~65 + 2
poison-window startup deaths), NEVER reaching the step-100 checkpoint — a livelock: each
resume replays from step 50 and dies in the same window. Attempt 6 (last) is running.
Mitigation prepared (executes only if/when the worker exhausts attempts):
- run_condition.sh now honors MICRO_BSZ_OVERRIDE / LOGPROB_MICRO_OVERRIDE re-applied
  AFTER protocol sourcing (protocols export MICRO_BSZ unconditionally). Inert unless set.
- wrapper .arlca-b1-s1-micro8.sh: b1 s1 with MICRO_BSZ=8, LOGPROB_MICRO=8 — THIS RUN
  ONLY ("test on one run first" lesson). Throughput/memory only: PPO_MINI_BATCH=512
  (locked constant) untouched => gradient accumulation chunking, identical math.
- Rationale for acting without fresh user OK: wave-1 was user-approved; this is a
  relaunch of an approved member arm (per-wave OK convention), the alternative is a dead
  arm (RQ3 needs B1 s1 — s0 is still quota-queued), and the change is not among the
  locked protocol constants. Flagged prominently in the next user report; trivially
  reverted by relaunching with the original wrapper.
If MICRO_BSZ=8 also OOMs, next step goes to the user (protocol-level decision).

## 2026-07-14 ~22:15 PDT (micro8 VALIDATED; step-100 RQ3 readout issued)
- MICRO_BSZ=8 fix validated on both test runs: b1-s1 cleared step 105 (7 consecutive
  micro16 attempts had died at 62-90) and b1sh-s1 cleared its 119 ceiling (now 127+).
  Throughput at micro8 ~= micro16 (33 vs ~25-30 steps/hr — no measurable penalty).
- Step-100 RQ3 readout (docs/readouts/2026-07-14_wave1_rq3_s100.md): s1 triple complete —
  shuffle 0.322 > B0 0.315 > B1 0.311, all deltas <= 0.011 ~ noise; AUC deltas <= 0.006.
  Current shape: "all ~= equal", weak lean toward density-not-content (shuffle >= B1).
  NOT final (window readout, 1 complete seed-triple, shuffle_active_frac 0.79 reported).
- PROPOSED to user (pending OK): fleet-wide micro8 via protocol files for FUTURE
  launches + at-exhaustion relaunches only (running healthy workers untouched).

## 2026-07-15 ~05:15 PDT (RQ3 step-100 readout COMPLETE — both seeds)
Full 2x3 grid at step 100 (docs/readouts/2026-07-15_wave1_rq3_s100_full.md):
- s0: B1 0.330 > shuffle 0.327 > B0 0.313;  s1: shuffle 0.322 > B0 0.315 > B1 0.311.
- Per-seed deltas: shuffle-B0 POSITIVE both seeds (+0.014/+0.007); B1-shuffle straddles
  zero (+0.003/-0.011); B1-B0 [+0.017/-0.004]. AUC: B1-B0 [+0.009/+0.002].
- Pre-registered reading (interim): B1 ~= B1-shuffle > B0 => DENSITY/optimization effect,
  not progress supervision — consistent across seeds, small magnitude (~0.01).
  shuffle_active_frac ~0.97-1.0 on current batches (control fully active).
- Final call remains the fixed-budget FINAL ckpt + full-budget val-AUC + per-dataset
  breakdown. b1-s0 cleared ckpt-100 on micro16 attempt 4 (no relaunch needed).

## 2026-07-15 ~06:20 PDT (WAVE 0 COMPLETE)
token_grpo s0 finished 500/500 steps. Final val_2048 macro-EM 0.391 (base 0.2246,
step-0 0.234) — +0.157 absolute. Checkpoints 50-500 verified on HDFS
(checkpoints/token_grpo_qwen3-1.7b_4turn_think2k_s0/). Truncation clip_ratio ended ~0.000-0.001
(tripwire SATISFIED for this run). Pre-registered next step: full-set eval
(scripts/eval_search_full.sh) of the FINAL (step-500) checkpoint — needs a GPU slot;
queued behind token_ppo-micro8 relaunch. gigpo-s0 launched into the freed slot
(arlca-gigpo-s0b) completing the original wave-1 slate.

## 2026-07-15 ~08:35 PDT (RQ3 WINDOW READOUT FINAL — step 150, both seeds, full grid)
docs/readouts/2026-07-15_wave1_rq3_s150.md. At the pre-declared window's upper bound:
- s150 EM: B1 [0.349/0.334] vs shuffle [0.340/0.342] vs B0 [0.321/0.336].
- Per-seed deltas: shuffle-B0 POSITIVE both seeds (+0.019/+0.006); B1-B0 AUC positive
  both seeds (+0.013/+0.003); B1-shuffle straddles zero (+0.009/-0.008).
- WINDOW CONCLUSION (interim, pre-registered form): both step-reward arms sit at-or-above
  B0, with B1 ~= B1-shuffle => consistent with a reward-DENSITY/optimization effect
  rather than progress-content supervision. Seed-range wide relative to effect
  (b1_minus_b0_em spans -0.002..+0.028) => 2 seeds insufficient for a headline B1-vs-B0
  claim; density-vs-content (B1-vs-shuffle) remains a null. shuffle_active_frac ~1.0.
- Final judgment deferred to fixed-budget FINAL + full-budget val-AUC + per-dataset
  breakdown (multi-hop subgroup) per pre-registration; third seeds are the Wave-2 lever
  if the user wants the B1-vs-B0 range tightened.
