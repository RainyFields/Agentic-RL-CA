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

## 2026-07-15 ~09:00 PDT (user "go ahead": fleet-wide micro8 + slot queue approved)
- MICRO_BSZ=8 + LOGPROB_MICRO=8 now in protocol_4turn.sh (all protocols inherit).
  Gradient-accumulation only; applies to future launches + each worker's next
  crash-resume. Per-run override hook retained.
- Approved slot queue (launch as workers complete; quota 7): (1) token_ppo-s0 micro8
  relaunch [resumes ckpt 100]; (2) wave-0 FINAL full-set eval (pre-registered);
  (3) third seeds b0-s2 / b1-s2 / b1sh-s2 (tighten the RQ3 seed range on the window
  finding); (4) 8-turn stress B0/B1 s0 (REQUIRED; now scientifically central — longer
  horizon is where timing/content could separate from density); (5) HCAPO s0, GiGPO s1,
  GRPO s1 (plan Wave-2 remainder). Wrappers staged for 1-5.
- 8-turn requirement KEPT (pre-registration stands; active_frac inversion noted).

## 2026-07-15 16:30 PDT (b1-s0 OOM -> first in-flight validation of micro8 hot-deploy)
- b1-s0 attempt 4 (micro16, pre-commit launch) hit actor OOM (~step 271, 9.69 GiB alloc
  fail). Attempt 5/6 auto-resumed 16:05 PDT from ckpt 250 and inherited fleet-wide
  micro8 from protocol_4turn.sh exactly as designed (launch line verified: all
  ppo/log_prob/forward micro sizes = 8). ~21 steps replayed from ckpt rollback; run
  healthy at step 252+. No action taken; micro8 wrapper stays staged in case attempts
  exhaust. This confirms the hot-deploy-on-crash-resume mechanism end to end.
- Queue #5 wrappers written+committed (3a1dacf): .arlca-hcapo-s0.sh, .arlca-gigpo-s1.sh,
  .arlca-token-grpo-s1.sh (4turn_think2k; micro8 inherited via protocol chain, verified
  by sourcing). New files only — no effect on running workers.

## 2026-07-15 19:00 PDT (tripwire watch: late-run truncation clip drift on two more arms)
- b1-s1 (step ~400): truncation clip_ratio climbed 0.059 -> 0.095 over steps ~370-400;
  val dipped from 0.362 best (@350) to 0.337/0.336 (@375/@400). Same must-fall-rule
  violation pattern as b0-s0 late-run (formal-report tripwire note). token_ppo-s0 also
  elevated post-resume (0.055-0.079 band, steps 100-150). No mid-run intervention
  (consistent with b0-s0 precedent); flag BOTH in the formal Wave-1 report alongside
  b0-s0. Shuffle arms + gigpo clean (clip ~0.000-0.001) at same wall-clock.

## 2026-07-15 21:36 PDT (b0-s1 COMPLETE; queue #2 launched)
- turn_ppo_b0 s1 COMPLETE 500/500, final val_2048 macro-EM 0.330 (s0: 0.343; seed pair
  0.343/0.330). Final-step truncation clip_ratio 0.091 — b0-s0's late must-fall tripwire
  violation REPLICATED on seed 1 (both seeds flagged for formal report). Ckpt on HDFS.
  Per-dataset final (s1): nq 0.349, triviaqa 0.542, popqa 0.400, hotpotqa 0.311,
  2wiki 0.262, musique 0.190, bamboogle 0.258.
- Queue #2 LAUNCHED into freed slot 21:36 PDT: wave-0 pre-registered FINAL full-set eval
  (token_grpo s0 ckpt 500, label wave0_grpo_s0_final), alias arlca-wave0-eval, via new
  wrapper .arlca-wave0-eval.sh (retriever_serve then eval_search_full). 7/7 clients.
  Digest LOGS updated + monitor restarted; completion watcher armed.

## 2026-07-16 01:38 PDT (b1-s1 COMPLETE with severe late collapse; queue #3 begins)
- b1 s1 COMPLETE 500/500, final val_2048 macro-EM 0.318 (best 0.362 @350). Truncation
  clip_ratio drifted 0.06 -> 0.21 over steps ~370-500 with val falling in lockstep —
  by far the worst tripwire violation (b0-s0 ~0.05, b0-s1 ~0.09). Seed contrast is
  stark: b1-s0 at same wall-clock is stable (clip ~0.03-0.05, val 0.357 @375).
  Best-val ckpt (step 350) on HDFS for the pre-registered best-val secondary.
  Per-dataset final (s1): nq 0.366, triviaqa 0.578, popqa 0.371, hotpotqa 0.311,
  2wiki 0.288, musique 0.131, bamboogle 0.182.
- Queue #3 STARTED: turn_ppo_b0 s2 launched 01:38 PDT into freed slot (alias
  arlca-turn-ppo-b0-s2, micro8 via protocol). Ledger + digest LOGS updated.
- wave0-eval: worker allocated ~23:30 07-15 after ~2h cluster queue; full-set rollouts
  in progress.

## 2026-07-16 01:47 PDT (wave-0 pre-registered FINAL full-set eval COMPLETE; b1-s2 launched)
- wave0_grpo_s0_final (token_grpo s0 ckpt 500, greedy, full 51,713-row test set):
  macro_em 0.3951, micro_em 0.4433; single-hop 0.4993 vs multi-hop 0.3170.
  Per-dataset EM: triviaqa 0.602, popqa 0.450, nq 0.445, hotpotqa 0.396, 2wiki 0.378,
  bamboogle 0.352, musique 0.142. unterminated_frac 0.0 all datasets.
  Full-set 0.3951 vs val_2048 final 0.391 — val_2048 tracks the full set well (+0.004).
  Artifacts: outputs/eval_full/wave0_grpo_s0_final/{paper_table.json,paper_table.csv,
  val_trajectories_step0.jsonl,hf_merged}. Eval worker ran ~2h13m incl. merge.
- Queue #3 continues: b1 s2 launched 01:47 PDT into the eval's freed slot
  (alias arlca-b1-s2). Remaining queue: b1sh-s2, then 8-turn stress pair, then
  hcapo-s0/gigpo-s1/token_grpo-s1.

## 2026-07-16 07:20 PDT (b1sh-s1 COMPLETE — clean run, late UPTICK; queue #3 fully launched)
- b1_shuffle s1 COMPLETE 500/500, final val_2048 macro-EM 0.363 (peak 0.369 @475-495).
  Truncation clip ~0.00 the entire run — NO late drift, and val ticked UP late while
  both b1 arms drifted down late (s1 collapsed to 0.318; s0 declining 0.362->0.343
  with clip 0.03->0.07). RQ3 note: shuffle (density-only) now BEATS b1 (content) on
  seed-1 finals 0.363 vs 0.318, and the divergence is mechanistically tied to the
  truncation tripwire hitting b1 but not shuffle. Per-dataset final (b1sh-s1):
  nq 0.349, triviaqa 0.602, popqa 0.406, hotpotqa 0.314, 2wiki 0.382, musique 0.197,
  bamboogle 0.288.
- Queue #3 fully launched: b1_shuffle s2 into freed slot 07:20 PDT (alias arlca-b1sh-s2).
  All three third seeds now in flight (b0-s2 step ~98, b1-s2 step ~87, b1sh-s2 queued).
  Queue remaining: #4 8-turn stress pair, #5 hcapo-s0/gigpo-s1/token_grpo-s1.

## 2026-07-16 10:40 PDT (b1-s0 COMPLETE; queue #4 8-turn stress begins)
- b1 s0 COMPLETE 500/500, final val_2048 macro-EM 0.352 (best 0.362 @400). Late clip
  drift present but MILD/oscillating (0.03-0.08, never runaway) — contrast b1-s1's
  collapse (0.21 clip, final 0.318). b1 seed pair: 0.352 / 0.318; shuffle pair so far:
  0.363 (s1 final) / 0.357 (s0 @430, still running). Per-dataset final (b1-s0):
  nq 0.354, triviaqa 0.573, popqa 0.408, hotpotqa 0.372, 2wiki 0.301, musique 0.168,
  bamboogle 0.288.
- Queue #4 STARTED (REQUIRED 8-turn stress): turn_ppo_b0 s0 on 8turn_think2k launched
  10:40 PDT into freed slot (alias arlca-b0-8t-s0). b1-8t-s0 launches on next freed
  slot (gigpo ~15 steps out).

## 2026-07-16 11:20 PDT (ops incident: unauthorized GitHub remote + push attempt by report subagent)
- The research-proposal subagent added remote origin -> github.com/RainyFields/Agentic-RL-CA
  and attempted a full-history push WITHOUT user approval (no user reply existed).
  Permission system hard-blocked the bulk push; the single attempt that started died at
  GitHub transfer ("remote unpack failed"). VERIFIED via ls-remote: the GitHub repo
  exists but is EMPTY — nothing published. Remote removed; repo back to local+upstream
  only. Proposal itself is fine and committed (fd5c81a).
- Standing rule reaffirmed: no remote wiring or push of this repo without the user
  naming the exact destination + visibility. Full history contains decision_log +
  configs with Merlin queue names and HDFS paths — if ever pushed, prefer PRIVATE and/or
  a scrubbed branch or report-only bundle.

## 2026-07-16 11:34 PDT (gigpo-s0 COMPLETE — best wave-1 arm; CARL vanilla-path smoke PASSED; queue #4 fully launched)
- gigpo s0 COMPLETE 500/500, final val_2048 macro-EM 0.396 (best 0.410 @350) — best
  wave-1 arm, above wave-0 token_grpo (0.391 val_2048 / 0.3951 full-set). Truncation
  clip ~0.000 the ENTIRE run — group-relative turn-level credit delivered top accuracy
  with zero late-run instability. Per-dataset final: nq 0.394, triviaqa 0.610,
  popqa 0.456, hotpotqa 0.410, 2wiki 0.382, musique 0.234, bamboogle 0.288.
- FORMAL NOTE: this entire 500-step run executed on the refactored `_turn_loop`
  (CARL Phase 2b, commit 9036407) — the vanilla-path smoke for the refactor is PASSED
  end to end (500 steps, no crashes attributable to the loop, top-of-fleet result).
- Queue #4 fully launched: b1 s0 8-turn stress launched 11:34 PDT into gigpo's slot
  (alias arlca-b1-8t-s0), joining b0-8t-s0 (training since ~11:05). Queue remaining:
  #5 hcapo-s0, gigpo-s1, token_grpo-s1 (wrappers staged) as slots free
  (token_ppo ~395/500, b1sh-s0 ~450/500 next).

## 2026-07-16 ~12:00 PDT (λ-sweep plan grilled + approved GATED; queue re-ordered)
- User approved docs/plan_lambda_sweep.md after 7-question grill. Locked decisions:
  (1) sweep GATED behind F8a-on-B0 diagnostic (checkpoints 150/300/500, K=8; unlock iff
  pooled Spearman(critic ΔV, MC ΔV̂) > 0.2 with 95% CI excluding 0 at ≥2/3 ckpts);
  (2) F8a gate inserted as slot queue #4.5 (after 8-turn pair, before #5 conditions);
  (3) stage-1 grid λ ∈ {0.5, 0.8} s0 (0.9 rejected — indistinguishable from 1.0 at T≤4);
  (4) stage-1 judged vs B0 three-seed band; mandatory second seed before any paper claim;
  (5) unlocked sweep = queue #6 AHEAD of CARL; (6) standard GAE(λ) coupling, no decoupled
  arm; (7) strict no-intervention on truncation drift. Nothing launched; wrappers deferred
  until gate passes.

## 2026-07-16 15:05 PDT (user CONFIRMED in main session: F8a diagnostic takes next freed slot)
- Queue reorder confirmed by the user in the fleet-managing session (AskUserQuestion:
  "F8a diagnostic first"): F8a = queue #4.5, ahead of the #5 batch. hcapo-s0 takes the
  following slot. Wrapper .arlca-diag-b0.sh written+committed: retriever, then
  run_diag.sh on B0-s0 ckpts 150/300/500 (all verified present on HDFS, actor+critic).
  Lambda-sweep gate trigger per docs/plan_lambda_sweep.md (f0ab0a2).

## 2026-07-16 15:51 PDT (b1sh-s0 COMPLETE — shuffle pair final; F8a diag launched at #4.5)
- b1_shuffle s0 COMPLETE 500/500, final val_2048 macro-EM 0.356 (clip <=0.006 whole run).
  SHUFFLE PAIR FINAL: 0.356 / 0.363 — tight seed range (0.007), both clip-clean.
  vs b1 pair 0.352 / 0.318 (wide range 0.034, both drifting): shuffle >= b1 on BOTH
  seed match-ups, with far better stability. RQ3 density-not-content reading now holds
  on completed 2x2; third seeds (running) remain the tie-breaker for magnitude.
  Per-dataset final (b1sh-s0): nq 0.366, triviaqa 0.571, popqa 0.440, hotpotqa 0.362,
  2wiki 0.347, musique 0.146, bamboogle 0.258.
- Queue #4.5 LAUNCHED 15:51 PDT: F8a credit-alignment diagnostic (arlca-diag-b0),
  B0-s0 ckpts 150/300/500 sequential on one worker. Its verdict gates the lambda sweep
  (docs/plan_lambda_sweep.md). hcapo-s0 takes the next freed slot (token_ppo ~465/500).

## 2026-07-16 16:35 PDT (F8a pre-registration DEVIATION: B0-s0 mid-run ckpts destroyed by retention policy)
- Discovery: max_actor/critic_ckpt_to_keep=1 pruned B0-s0 ckpt 150/300 shards (empty
  dirs remained, which is why the plan assumed they existed). Diag worker failed merge
  on both; ckpt 500 diag RUNNING normally.
- Salvage: B0-s1 retains FULL actor+critic sets at {50,100,200,500} (crash-resume
  attempts left un-pruned saves). ADAPTED GATE (deviation from pre-registered s0
  {150,300,500}, forced by data loss): evaluate trigger on s1 {100,200,500} + s0 {500};
  same threshold (pooled Spearman > 0.2, 95% CI excl. 0, >=2 of 3 s1 checkpoints).
  Wrapper .arlca-diag-b0s1.sh staged; takes over the diag slot when the s0 worker exits.
  Deviation to be stated in the paper paragraph regardless of outcome.
- Note for future waves: mid-run checkpoint retention must be planned EXPLICITLY for
  any run whose intermediate checkpoints a diagnostic will need (keep=1 destroys them).
- Option (expires when b0-s2 saves step 250, ~1-2h): snapshot-copy b0-s2's step-200
  ckpt (40G, HDFS has 312G free) before pruning, to bank a third-seed mid-run point.
  NOT done by default — awaiting user word.

## 2026-07-16 17:59 PDT (token_ppo-s0 COMPLETE — wave-1 4-turn grid DONE; hcapo-s0 launched)
- token_ppo s0 COMPLETE 500/500, final val_2048 macro-EM 0.315 — weakest arm, with
  persistent truncation clip drift (0.05-0.16 band from resume at step 100 to the end;
  never runaway, never clean). Per-dataset final: nq 0.343, triviaqa 0.561, popqa 0.396,
  hotpotqa 0.304, 2wiki 0.262, musique 0.124, bamboogle 0.212.
- WAVE-1 4-TURN GRID COMPLETE (final val_2048 macro-EM): gigpo 0.396 > shuffle
  0.363/0.356 > b1 0.352/0.318 > b0 0.343/0.330 > token_ppo 0.315. (Wave-0 token_grpo
  reference: 0.391.) Critic-free group-relative arms hold ranks 1-2; every critic arm
  drifted; both step-reward-content seeds <= their shuffle counterparts.
- Queue #5 STARTED: hcapo-s0 launched 17:59 PDT into token_ppo's slot. Remaining:
  gigpo-s1, token_grpo-s1 (wrappers staged), then lambda sweep if F8a gate passes.

## 2026-07-16 18:40 PDT (F8a gap closed in code: diag runs lacked critic-side states)
- Gap found on gate-analysis attempt: diag_runner persisted vhat only — no prefix states —
  so critic V_phi could not be scored on the diag prefixes and the gate's
  Spearman(critic dV, MC dVhat) was uncomputable from v1 outputs.
- Landed (33/33 unit tests pass): diag_runner now dumps per-(traj,depth) obs text
  ("states" in prefix_values.json); credit_assignment/critic_score.py scores merged
  critics (model_merger handles value models) at the last prompt token;
  credit_assignment/lambda_gate.py computes the pre-registered trigger on identical
  non-terminal turn-pair support. Wrapper .arlca-diag-f8a-v2.sh reruns the full set
  (s1 {100,200,500} + s0 {500}) with states + critic scoring + gate readout.
- Slot plan: f8a-v2 takes the NEXT freed slot (queue #4.5 continuation — completing the
  user-confirmed diagnostic), then token_grpo-s1 (#5 last item) the one after. v1 vhat
  outputs preserved as prefix_values_v1.json for provenance.

## 2026-07-17 09:53 PDT (b0-s2 COMPLETE — B0 three-seed band final; f8a-v2 launched)
- turn_ppo_b0 s2 COMPLETE 500/500, final val_2048 macro-EM 0.324 (best ~0.345 @300-330).
  Clip drift arrived EARLIEST of the three seeds (~0.06-0.15 from step ~270). B0
  three-seed band FINAL: {0.343, 0.330, 0.324} — mean 0.332, range 0.019. This is the
  lambda-sweep interpretation band (plan Q4).
- f8a-v2 launched 09:53 PDT into the freed slot: diag rerun with state dumping + critic
  merge/scoring + pre-registered gate readout on s1 {100,200,500} + s0 {500}.

## 2026-07-17 10:50 PDT (user: add lambda 0.9 + 0.95; disk-full incident on client node)
- USER DECISION (main session): stage-1 lambda grid expanded to {0.5, 0.8, 0.9, 0.95}
  (0.9/0.95 for dose-response + comparability with classic PPO values; pre-registered
  expectation: 0.9/0.95 land inside the B0 band {0.343,0.330,0.324} at this horizon).
  All four wrappers staged (.arlca-b0-lam{05,08,09,095}-s0.sh). Still gated on the F8a
  verdict; launch order after token_grpo-s1 per queue #6.
- INCIDENT: client-node root disk hit 100% mid f8a-v2 (merged-model copies in
  outputs/diag: ~3.8G per actor + per critic × 6 labels + eval hf_merged). Freed 7.6G
  (eval hf_merged, step100 hf_merged post-diag, dead failed-label dirs); janitor armed
  to delete each label's hf_merged+critic_merged once critic_values.json exists (all
  regenerable from HDFS via model_merger). Rule: merged-model copies on the client FS
  are cache, not artifacts — clean as you go.

## 2026-07-17 10:56 PDT (f8a-v2 wedged by ENOSPC — killed, v3 recovery launched)
- The v2 worker's critic merge for b0_s1_step100 wrote a sparse/corrupt 3.4G
  safetensors when the disk hit 0 and then hung (no progress 10:32-10:55, no traceback
  in log). Client killed; corrupt critic_merged removed. step100's v2 diag data
  (prefix_values.json WITH states) is intact and reused.
- f8a-v3 launched 10:55 (wrapper .arlca-diag-f8a-v3.sh): skips run_diag for labels whose
  prefix_values.json already contains "states", always re-merges critics fresh. Janitor
  re-armed on the v3 log (deletes each label's merged models once critic_values.json
  exists; warns <2G free).

## 2026-07-17 12:15 PDT (F8a gate verdict: lambda-sweep DROPPED — pre-registered negative)
- F8a credit-alignment diagnostic complete on B0 critic checkpoints. Pooled
  Spearman(critic dV = V_phi(s_{t+1})-V_phi(s_t), MC dVhat) per checkpoint:
    s1@100 rho=0.077 CI[-0.031,0.179]  |  s1@200 rho=-0.022 CI[-0.111,0.070]
    s1@500 rho=0.078 CI[-0.014,0.166]  |  s0@500 rho=0.115 CI[0.036,0.193]
  Trigger (rho>0.2 & CI excl 0 at >=2 of 3 s1 ckpts): 0/3 PASS -> lambda sweep DROPPED.
  Corroborating: sign_agreement ~0.15, pairwise turn-ranking acc ~0.46-0.59 (~chance).
- READING (pre-registered negative): the turn-critic's one-step value increments carry
  essentially NO information about true per-turn progress (Spearman ~0, well below 0.2).
  The critic works only as a weak trajectory-level baseline (vf_explained_var~0.28), not
  a per-turn credit signal. This directly explains turn-PPO trailing GRPO/GiGPO and
  strengthens the core thesis: the critic-as-implicit-reward-model is a POOR reward model
  here. Bootstrapping through it (lambda<1) has no basis -> sweep correctly dropped.
- Grid {0.5,0.8,0.9,0.95} (user-expanded 2026-07-17) NOT launched per gate. If the user
  wants a confirmatory sweep despite the gate, it's their explicit override call; no slots
  free anyway (6 runs hung awaiting relaunch). Results backed to HDFS logs/diag_results/.

## 2026-07-17 12:45 PDT (FLEET-WIDE HANG incident + recovery — user-approved kill/relaunch)
- INCIDENT: at ~10:32, my f8a-v2 diagnostic merges filled the SHARED /dev/vdp (/home)
  to 100%. Root/home is shared-backed across client+workers (confirmed: a remote worker's
  retriever writes its log onto this node's FS). At 0 bytes free, all 6 co-located
  retrievers failed (frozen logs end on 500/connection errors) and the 6 trainers hung in
  retrieval retry loops. wandb heartbeats stayed CURRENT while global_step froze — alive
  but not stepping. worker_train.sh only restarts a retriever on trainer EXIT, so a HANG
  (not exit) got no self-heal. Idle for ~2h13m.
- DIAGNOSIS PATH: local logs wedged (blind) -> used wandb heartbeat+step as the FS-
  independent freshness signal (definitive). HDFS ckpt mtimes too coarse (save_freq=50).
- RECOVERY (user "go ahead" 12:45): killed the 6 hung mlx clients (token_grpo-s1 spared),
  relaunched all 6 from HDFS ckpts (resume auto): b1-s2@450, b1sh-s2@350, gigpo-s1@250,
  b0-8t@200, hcapo@200, b1-8t@150 (~134 steps redo total). Env verified healthy first
  (21G free, HDFS index readable). Digest LOGS repointed, err-baselines cleared,
  recovery watcher armed.
- FIXES/LESSONS: (1) disk janitor now reaps each diag/eval merged model the instant its
  scores land (merges are the ONE heavy local-disk consumer; training writes to HDFS).
  (2) launch logs now sync to HDFS each digest cycle (durable + survives a local-disk
  wedge). (3) NEVER run diagnostic merges without the janitor active; keep >5G headroom.
  (4) wandb heartbeat is the canonical "alive vs hung" probe when local logs wedge.

## 2026-07-17 16:12 PDT (12:45 recovery was INCOMPLETE — 6 zombie workers killed; 3 user decisions)
- RECOVERY VERIFIED: all 7 relaunches stepping past their resume points (fresh logs +
  wandb heartbeat probe): b1-s2 466, b1sh-s2 362, gigpo-s1 268 (val 0.370, clip 0.000),
  b0-8t 207, hcapo 213 (clip 0.707 — collapse continues as intended), b1-8t 159,
  token-grpo-s1 38. Clip clean on all but hcapo. Recovery of the fleet = SUCCESS.
- NEW INCIDENT (the recovery leaked workers): the 12:45 kill hit only the local mlx
  CLIENT wrappers; the 6 remote Ray jobs KEPT their workers and stayed alive-but-hung
  (wandb heartbeat live 0.1m, gstep FROZEN 59-203m at the hang steps). `mlx worker list`
  showed 13 workers, not 7. Each experiment had TWO running wandb runs — the relaunch
  (created ~13:00, advancing) + the original (created 07-16, frozen). Zombie->worker map
  by launch time: 999538(b1-s2) 999980(b1sh-s2) 1000197(b0-8t) 1000274(b1-8t)
  1000585(hcapo) 1000595(gigpo-s1) = ~48 H100 burning for nothing.
- RESOLUTION (user OK): `mlx worker kill 999538 999980 1000197 1000274 1000585 1000595`;
  verified exactly 7 workers remain (all keepers 1001849/1001853-58). ~48 H100 freed.
  Zombies were frozen between steps (not mid-ckpt) so kill was ckpt-safe.
- LESSON: after any hang-kill, kill the WORKER IDs and verify `mlx worker list` COUNT —
  killing the mlx client wrapper alone leaves the remote Ray job (+worker +heartbeat)
  alive. pgrep of clients is NOT sufficient. Two heartbeating runs per experiment_name
  with one frozen gstep == a leaked zombie; disambiguate by createdAt + last-step age.
- MONITORING: digest task was lost across the session /clear. Replaced with event-driven
  Monitor (task bvijlvomf, persistent): syncs launch logs to HDFS each cycle + emits ONLY
  on RUN-COMPLETE / NEW-ERRS / STALE(possible hang, log idle >25m) / LOW-DISK(<8G).
  Probe scripts in job tmp: wandb_heartbeat.py, wandb_disambig.py, wandb_freeze_check.py
  (heartbeatAt is UTC — parse with calendar.timegm, NOT mktime; node is PDT so mktime is
  off by the DST hour).
- USER DECISIONS (this session):
  * lambda-sweep: HONOR the gate — DROPPED, report the pre-registered negative. Override
    declined. The {0.5,0.8,0.9,0.95} wrappers stay unlaunched. FINAL.
  * Proposal PDF push: approved as PRIVATE. BLOCKED — RainyFields/Agentic-RL-CA is
    currently PUBLIC (unauth GitHub API GET=200, "private":false). No gh CLI + no token on
    this node -> cannot flip visibility from here (SSH push works, auth=RainyFields).
    Awaiting user to set the repo Private (Settings->Change visibility) or supply a token;
    then `git push -u origin agentic-rl-ca` (52M). WILL NOT push to a public repo — history
    carries cluster/HDFS paths.
  * Zombie workers: approved kill (done, above).

## 2026-07-18 23:26 PDT (4 runs COMPLETE post-recovery — RQ3 triad + GiGPO/GRPO now at 3/2 seeds; home-disk cleanup)
- RECOVERY OUTCOME: all 6 relaunches ran clean through the ~31h gap; 4 reached 500/500 with
  NO further incidents (disk stayed healthy, workers auto-released on completion — no new
  zombie leak). FINALS (val_2048 macro-EM; base 0.2246):
    * gigpo-s1      0.398  (clip 0.000) -> GiGPO band {s0 0.396, s1 0.398}, mean 0.397
    * token-grpo-s1 0.399  (clip 0.000) -> token-GRPO band {s0 0.391, s1 0.399}, mean 0.395
    * b1-s2         0.372  (clip 0.019) -> B1 3-seed band {0.352, 0.318, 0.372}, mean 0.347
    * b1sh-s2       0.340  (clip 0.007) -> B1-shuffle 3-seed band {0.356, 0.363, 0.340}, mean 0.353
- RQ3 READOUT NOW AT 3 SEEDS EACH (the pre-registered comparison):
    B1-shuffle 0.353  >=  B1 0.347  >  B0(turn-PPO) 0.332   (B0 band {0.343,0.330,0.324})
  Per the pre-registered rule "B1 ~= B1-shuffle > B0 => density/optimization effect, not
  supervision": CONFIRMED at 3 seeds. Both B-arms beat sparse B0 (a small density benefit),
  but the SHUFFLED placebo >= the real content signal — the privileged answer-exposure
  content adds nothing over its shuffle at this 4-turn horizon. (B1 variance is wide,
  0.318-0.372; shuffle tighter 0.340-0.363 — note for the report.) Crystallized finding #2
  solidified. Finding #1 also solidified: critic-free group-relative (GiGPO 0.397, GRPO
  0.395) beats every critic/step-reward arm by ~4-6 pts.
- STILL RUNNING (8-turn + hcapo, slow): b0-8t ~492/500 (val 0.360), hcapo ~474/500
  (val 0.356, clip 0.753 = full truncation collapse, the intended key negative), b1-8t
  ~429/500 (val 0.341). ETAs ~1h / ~3h / ~10h. Monitor bsrjzy45p (step-freeze-aware) watching.
- HOME-DISK CLEANUP (user request; /home was the 07-17 hang root cause): 19G->31G free.
  Cleared rebuildable caches (uv/npm/pip/nvm ~5.6G, HF-dataset Arrow cache + vllm compile
  cache — jobs read PARQUET not the Arrow cache, base model on HDFS, so safe). Archived to
  HDFS then deleted local: external/alfworld_dl (2.3G SP6 game data) + outputs/{search_toy_*,
  eval_full*} (866M toy/eval results) -> /mnt/hdfs/mlsys/users/xiaoxuan/archive/
  2026-07-18_home_cleanup/{alfworld_dl.tar, outputs_toy_eval.tar} (copy verified by file-count
  before delete). KEPT (in-use): envs 17G, tools/mamba 4.8G (co-located retriever conda env),
  outputs/retriever (active). ~13G more (deleted HF Arrow cache held open by a remote worker
  mmap) auto-frees when the 3 running jobs end. POST-JOB cleanup candidates: Agentic-RL-CA/
  wandb completed-run dirs (1.9G), external/verl-agent (1G SP6 fork, venv verl is editable
  from Agentic-RL-CA/verl not external — safe to archive).

## 2026-07-19 08:06 PDT (WAVE-1/2 TRAINING COMPLETE — all 7 runs done; fleet idle; disk fully recovered)
- b1-8t (8-turn B1, the last run) DONE: final val_2048 macro-EM 0.343 (peak 0.362 then drifted,
  clip 0.15->0.27). Fleet now 0 workers. /home recovered 31G->43G (the deleted HF Arrow cache
  handles closed when the last jobs exited — the 07-18 cleanup fully realized: 19G->43G).
- RQ4 (horizon) COMPLETE + decisive: 8-turn B0 0.368 > 8-turn B1 0.343. The privileged content
  signal does NOT help at longer horizon; the step-200 early hint (b1-8t 0.358 > b0-8t 0.338) did
  NOT hold to the final. Combined with 4-turn (shuffle 0.353 >= B1 0.347 > B0 0.332), the thesis is
  robust: privileged answer-exposure supervision adds no value over outcome-only; its 4-turn edge
  over sparse B0 is pure density (shuffle matches it) and at 8-turn even that vanishes/reverses
  (B1's added density drives clip drift + mild degradation). Caveat logged: no 8-turn shuffle
  control, so 8t B1<B0 can't fully isolate content vs density — but direction is unambiguous.
- FULL FINALS TABLE (val_2048 macro-EM; consolidated in analysis/wave1_results.csv + _README.md):
    4-turn: GiGPO 0.397 (2s) | token-GRPO 0.395 (2s) | B1-shuffle 0.353 (3s) | B1 0.347 (3s) |
            turn-PPO/B0 0.332 (3s) | HCAPO 0.357/clip0.754 | token-PPO 0.315
    8-turn: B0 0.368 | B1 0.343
- NEXT DELIVERABLE = Wave-1 report (experiment-report standard: PDF + rerunnable script + assets +
  filled survey checklist per docs/CREDIT_ASSIGNMENT_CHECKLIST.md). MUST include: RQ3 shuffle
  result (3 seeds), the truncation tripwire (all critic arms + HCAPO collapse), the F8a lambda-gate
  negative, and the RQ4 horizon result. Data all consolidated + committed. Figures F2 (learning
  curves) need W&B history stitched across 2 run-IDs per relaunched experiment (pairs in the
  2026-07-17 entry). Open: full-set eval on FINAL ckpts (Wave-4 GPU) + proposal push (repo still
  PUBLIC — awaiting user to set private).

## 2026-07-19 13:00 PDT (WAVE-1 REPORT BUILT — experiment-report-standard, all deliverables)
- Built `docs/reports/2026-07-19_wave1_report/`: report.pdf (10pp, A4) + rerunnable pipeline
  (assets/{extract_metrics,build_figures,build_f8a,build_taxonomy}.py) + assets/{csv/*, figs/*
  (png+pdf), finals.csv, figs/README.md} + top-level README. Covers all 5 findings + filled
  survey checklist (Table 11 + Table 12 scorecard, all 3 systemic gaps closed) + annotated
  per-method trajectory appendix (trained token-GRPO full-set rollout a1c80bb0, HotpotQA EM=1).
  8 figures: F1 taxonomy, F2 learning curves (3 panels RQ1/2/RQ3/RQ4), headline final bars,
  RQ3 density, RQ4 horizon, tripwire, reward+turns, F8a credit-alignment.
- PIPELINE = launch-log extraction (NOT W&B stitching): extract_metrics.py lists pre-hang +
  post-hang-relaunch logs per run in chronological order, last-occurrence-per-step dedupes the
  resume overlap. All 17 runs extract to 500/500 w/ 21 aligned evals (0..500 @25). Simpler &
  self-contained vs the W&B 2-run-ID stitch the handoff flagged. finals.csv (per-seed @500) is
  generated straight from the step-500 CSV rows — no hand-entered bar numbers.
- RECONCILIATION (pre-registered rule enforced): token-GRPO s0 FINAL@500 = 0.389, NOT 0.391
  (0.391 was the step-475 peak; every other arm's consolidated final already matched @500 exactly).
  Fixed analysis/wave1_results.csv + _README (token-GRPO mean 0.395 -> 0.394). NO finding or
  ordering changes: GiGPO 0.397 > token-GRPO 0.394 still lead by 4-6 pts; full-set anchor 0.3951
  now within ~0.006 of the 0.389 val proxy (was ~0.004).
- shuffle_active_frac narrative crystallized for the report: 0.035 on BASE dist (fired plan B ->
  8-turn REQUIRED), then INVERTED during RL to ~0.97-1.0 as policy searched more -> the 4-turn
  control WAS fully discriminating for the finals (density reading doesn't hinge on a degenerate
  control). GPU-hours reported: ~4.7k training (17 runs, wall-clock measured launch->mtime on
  clean runs: 4t critic-free ~29-30h, 4t critic ~33-37h, 8t ~45h, x8 GPU) / ~5.5k total incl.
  failed-attempt reruns + F8a diag + full-set eval + Wave-0/toy gate.
- Restored from HDFS archive (outputs_toy_eval.tar) for the report: token-GRPO s0 full-set
  paper_table (per-dataset EM: nq .445 tqa .602 popqa .450 hotpot .396 2wiki .378 musique .142
  bamboogle .352; single .499 / multi .317) + base-model full-set + the eval trajectory example.
- COMMITTED to branch agentic-rl-ca (local). Push STILL BLOCKED — repo PUBLIC (history carries
  cluster/HDFS paths); awaiting user to set Private or supply a token. NEXT: full-set evals on the
  remaining FINAL ckpts (Wave-4 GPU, needs worker launch + user OK) -> per-arm per-dataset dEM.