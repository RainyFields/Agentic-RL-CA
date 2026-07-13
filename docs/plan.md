# Agentic-RL-CA — Progress supervision & credit assignment for multi-turn LLM agents (Search-R1 QA, Qwen3-1.7B)

Revised 2026-07-13 after three rounds of user review (8-point critique, 5-point follow-up,
10-point follow-up — all incorporated; IGPO dropped per user in round 3).

## Research questions (the paper's spine)

```
RQ1  Does credit granularity matter?            token PPO vs turn PPO (controlled contrast)
RQ2  Can outcome-only CA recover useful         GRPO / GiGPO / HCAPO (+ CARL once core is
     intermediate credit?                        healthy) — benchmark + CREDIT-ALIGNMENT
                                                 diagnostic (assigned credit vs MC continuation
                                                 value; the quantitative answer, not just EM)
RQ3  Does explicit privileged progress          B0 vs B1 vs B1-shuffle
     supervision add value?
RQ4  Does the benefit grow with task/horizon    single-hop vs multi-hop (pre-registered subgroup);
     difficulty?                                 4-turn vs 8-turn stress test (promoted to
                                                 REQUIRED if shuffle_active_frac is low)
RQ5  Can a scalable evaluator approximate the   B2-lite (prompted step evaluator) → B2-full
     privileged signal?                          (trained; separate follow-up project)
```

Headline framing: **"Outcome-only credit assignment vs explicit progress supervision for
multi-turn LLM agents: when does intermediate reward actually help?"** — the CA benchmark (RQ2)
provides the context, not the headline. RQ1–RQ3 arms run in parallel under the worker budget,
with RQ3 front-loaded so the progress-supervision answer lands first.

**Minimum publishable unit (MPU, review round 3 point 10):** Wave 0 + Wave 1 (+ third seeds) +
full-set eval + the F8a credit-alignment diagnostic on the core arms = a complete,
self-sufficient paper. CARL, B2-lite, StepSearch, and the 8-turn stress test are each additive
and individually droppable. Every future scope decision is a one-line check against the MPU.

## Definition: what is a "turn" (used consistently in code + all reports)

A **turn** = one model generation segment — `<think>…</think>` followed by exactly one action,
either `<search>query</search>` or `<answer>…</answer>` — plus the environment's response to it
(the retrieved `<information>…</information>` block; empty for the answer turn). A **trajectory**
= up to `env.max_steps` turns (4 in the main protocol) ending in an answer (or truncation). In
the training batch, each turn is one row (`traj_uid` groups a trajectory's turns; `turn_index`
orders them).

**Token-level PPO vs turn-level PPO** (RQ1's controlled contrast): token-level PPO (`gae`) runs
GAE over every *token* of the flattened sequence with a token-level value head — credit diffuses
token-by-token across turn boundaries. Turn-level PPO (`gae_turn`) treats each *turn* as one
credit unit: one scalar V(s_t) per turn, GAE recursion over the turns, and each turn's scalar
advantage broadcast uniformly to that turn's response tokens. **Same rollout protocol and policy
objective; value estimation and advantage assignment differ in granularity.** (Independent runs
do not literally see the same realized rollouts.)

**Reporting commitment:** every report includes annotated example trajectories (per method: the
turn-by-turn text, the per-turn reward/advantage each algorithm actually assigned) so the credit
assignment is inspectable, not just curves.

## Context

The SP6 campaign compared credit-assignment methods (GRPO / GiGPO / turn-level PPO / HCAPO) on
ALFWorld with Qwen3-1.7B. The Search-R1 replication (tr1, avg EM 0.425 vs paper 0.431) established
the search-QA task, data, retriever, and paper-protocol eval on this cluster. This plan bridges
the two on the **Search-R1 dataset + eval setting** with **Qwen3-1.7B (instruct, non-thinking
start)**.

User decisions (2026-07-13):
- **Model: Qwen3-1.7B instruct, NON-thinking mode to start** (`enable_thinking=False`). The
  `<think>…</think>` block comes from the search prompt template as plain text — exactly the
  Search-R1-family format, so outputs stay format-compatible with the literature. Lengths
  grounded in prior work: Search-R1 uses `max_response_length=500`/turn; verl-agent's stock
  search example uses 512 → we use **512** with `max_prompt_length=4096`. Revisit thinking mode
  only if the toy gate shows truncation or degenerate reasoning.
- **Scale-up rule:** ONE baseline on Qwen3-1.7B first (Wave 0, pre-registered gate below); if it
  fails the gate, subsequent experiments scale to **Qwen3-4B** (single pre-registered fallback:
  Qwen3 has no 3B; 4B stays in-family and is CARL's reasoning backbone).
- **Eval: full paper protocol** — subsample val during training + post-hoc full-set greedy EM
  (51,713 rows, 7 datasets) on selected checkpoints (pre-registered selection rule below).
- **Excluded per user: ARPO (round 1), IGPO (round 3).** StepSearch included but **secondary
  tier** (see scope tiers).

### Horizon strategy (answers "why the 4096/4-turn cutoff?", review round 3 point 5)
The 4-turn / 4096-token setting is the **Search-R1 protocol** — it stays for the main table so
numbers are comparable to the literature; it is not a claim that 4 turns is enough. The strategy
for extending later: **the horizon is a config value with ZERO literal 4s in code** — B1 shaping,
shuffle eligibility, snapshot depth indexing, the eval aggregator, and the diagnostic's prefix
enumeration must all read `max_steps` / length limits from config (audited at implementation
time). An **8-turn config ships in `configs/` from day 1** (`max_steps=8`, `history_length=8`,
`max_prompt_length≈8192`), pre-registered as the labeled horizon stress test — extending is a
launch, not a refactor.

## Framework decision: verl-agent fork, NOT the old Search-R1 stack

Build everything on the `~/xiaoxuan/external/verl-agent` fork (verl 0.3.1.dev) + venv
`~/xiaoxuan/envs/verl-agent` (torch 2.8.0 / vllm 0.11.0 / transformers 4.57.3). Rationale:
- **Qwen3 cannot run on the old Search-R1 stack** (vllm 0.6.3 / transformers 4.47.1 — no qwen3
  support; upgrading would break the validated torch2.4/flash-attn/XFORMERS stack).
- The fork **already implements the 5 existing estimators**, validated on ALFWorld with
  Qwen3-1.7B: `gae` (`verl/trainer/ppo/core_algos.py:67`), `grpo` (`core_algos.py:113`),
  `gae_turn` (`verl/trainer/ppo/core_turn_ppo.py:19`), `gigpo` (`gigpo/core_gigpo.py:138`),
  `hcapo` (`gigpo/core_hcapo.py:74` + hindsight pass `ray_trainer.py:1370`). Dispatch:
  `AdvantageEstimator` enum `ray_trainer.py:96-100`, `compute_advantage()` `ray_trainer.py:248`.
- The fork **ships a native search env** (`SearchEnvironmentManager`, `env_manager.py:46`;
  `env_package/search/`; EM reward `search/utils.py::compute_score`; example
  `examples/gigpo_trainer/run_search.sh`).
- Estimators are env-agnostic (consume per-turn fields from `rollout_loop.py`: `traj_uid`,
  `turn_index`, `anchor_obs`, `rewards`, `env_done`).
- Prior plan docs endorse this route: `reward_models/docs/sp5_search_qa_plan.md`,
  `docs/searchr1_plan.md`.

**Net-new code: CARL, StepSearch (secondary), B1/B1-shuffle shaping, credit-alignment
diagnostic, eval port.**

## Project layout — the repo IS the fork (no nested git)

Project name (user): **Agentic-RL-CA**. `~/xiaoxuan/Agentic-RL-CA/` is itself the verl-agent
fork — cloned with full upstream history, project directories added at the repo root (one
publication artifact, diff-vs-upstream reviewable, no nested git):

```
Agentic-RL-CA/          # git clone of the verl-agent fork, branch `agentic-rl-ca`
├── verl/               # (upstream) trainer — dispatch/eval patches land here
├── agent_system/       # (upstream) envs + rollout — snapshot/restore, step rewards land here
├── gigpo/              # (upstream, UNTOUCHED) — GiGPO/HCAPO stay where upstream put them
├── credit_assignment/  # NEW top-level package: core_carl.py, B1/B1-shuffle helpers, the
│                       #   credit-alignment diagnostic. Net-new estimators do NOT go inside
│                       #   gigpo/ (a directory named after a competing method) — costs one
│                       #   import line in the dispatch branch, keeps upstream files untouched,
│                       #   and avoids implying CARL depends on GiGPO. (review round 3, point 3)
├── scripts/            # NEW: per-condition launchers, retriever bring-up, toy gate, full-set
│                       #   eval, resilient driver — adapted from searchr1/scripts/ +
│                       #   reward_models/scripts/ (run_cmp_* pattern)
├── configs/            # NEW: one file per condition (versioned), incl. the 8-turn stress config
├── docs/               # NEW: methods note, CREDIT_ASSIGNMENT_CHECKLIST.md, plan, decision log
├── analysis/           # NEW: figure/report builders (searchr1 build_report.py pattern)
├── outputs/, data/     # gitignored (toy dumps, eval CSVs; HDFS pointers + prep commands)
└── README.md           # paper-style: RQs, methods, repro instructions, results
```

Inheritance: searchr1/ contributes infra patterns (retriever standup, HDFS sync/prune, resilient
segmented driver, full-set eval protocol, report builder); reward_models/ contributes the
condition-script pattern + report standard. Both stay untouched for their own tracks. Publication
readiness from day 1: Apache-2.0 (inherited), seeds + variance per checklist, every figure backed
by CSV + provenance README, W&B project `agentic-rl-ca`, no cluster-internal paths in committed
scripts (env-var indirection).

## Reused assets (no re-download)

- **HDFS** `/mnt/hdfs/mlsys/users/xiaoxuan/searchr1/searchr1_data/`: e5_Flat.index 61G,
  wiki-18.jsonl 14G, test parquet (51,713 rows, 7 datasets with `data_source`).
- **Retriever recipe:** Search-R1 `retrieval_server.py` + micromamba env `searchr1-retr-conda`
  (conda-forge faiss-gpu — pip wheels lack sm_90); 61G index fp16-sharded over ≥2 GPUs;
  co-located on the training worker (tr1-validated). API needs `"return_scores": true`.
- **Train data:** `python -m examples.data_preprocess.preprocess_search_r1_dataset`
  (HF `PeterJinGo/nq_hotpotqa_train`).
- **Reference:** tr1 token-PPO EM curves + full-set table
  (`searchr1/docs/reports/2026-07-09_tr1_final_assets/`).

## Conditions, scope tiers & matched config

Common (all arms): Qwen3-1.7B, `enable_thinking=False`, `env.env_name=search`, `env.max_steps=4`,
`env.history_length=4`, `data.max_prompt_length=4096`, `data.max_response_length=512`,
`data.truncation=left`, matched group size / batch / total steps, W&B naming convention,
checkpoints direct-to-HDFS. All horizon/length values config-driven (see Horizon strategy).

Context/history: `history_length ≥ max_steps` ⇒ the per-turn prompt covers the whole episode —
no history dropped, matching Search-R1's full-history protocol (Search-R1 clips each retrieved
observation, not the history). `truncation=left` is a safety valve ONLY: **any nonzero truncation
rate is treated as a bug, not a tolerable rate** (review round 3, point 6) — left-truncation
deletes early history and silently changes the MDP mid-trajectory. Gate at <0.1%; the response to
any real rate is "raise max_prompt_length or clip retrieved passages harder," never "accept it."

**Core tier (the paper's main table):**

| RQ | condition | adv_estimator | critic | notes |
|---|---|---|---|---|
| 1 | token-level PPO | `gae` | yes | stock |
| 1 | turn-level PPO | `gae_turn` | yes | = B0; cross-turn GAE |
| 2 | token-level GRPO | `grpo` | no | `use_kl_loss=True` |
| 2 | GiGPO | `gigpo` | no | `enable_similarity=True, thresh=0.9, mean_std_norm, step_advantage_w=1.0` |
| 2 | HCAPO | `hcapo` | no | ω, T_temp, clip [0.8,1.2] — lock from PDF |
| 3 | B1 | `gae_turn` + step rewards | yes | privileged retrieval-hit signal (below) |
| 3 | B1-shuffle | `gae_turn` + shuffled step rewards | yes | **the key control** (below) |

**Second wave (added once core runs are healthy):** CARL (`carl`, NEW — largest engineering
risk, lands after core is stable).
**Secondary tier (appendix, not the matched main table):** StepSearch — its annotation pipeline
changes the data distribution (fallback is HotpotQA-only), so it cannot cleanly occupy a
matched-benchmark row; report as a secondary experiment. B2-lite/B2-full per RQ5 gating.
**Dropped per user:** ARPO; **IGPO** (round 3 — removed everywhere: no Phase 2b, no table row,
no wave slot, no hindsight-scoring consumer; freed compute goes to extra GiGPO/HCAPO seeds —
more seeds beats more methods).

Method rationales: **CARL** (arXiv 2512.04949, "Criticality-Aware Agentic RL" — NOT the
identically named constraint-aware planning paper 2607.04854) — rollout tree, criticality-gated
edge advantages. **StepSearch** (arXiv 2505.15107) — PPO with step-wise retrieval-IG +
redundancy-penalty rewards; needs golden sub-question annotations. Hyperparams to lock from PDFs
before coding: HCAPO (group 5, batch 512, β_KL=0.001), CARL (N₀=1/8, N=16, resume temp 0).

## Pre-registered decision rules (set before any curve is seen)

- **Wave 0 gate (1.7B feasibility — gates on the BASE MODEL, not on GRPO trainability):**
  proceed at 1.7B iff, measured on the untrained model + first rollout batches:
  (a) baseline fixed-val macro-EM ≥ 0.03 (nonzero floor), (b) valid-action rate ≥ 90%,
  (c) truncation-event rate < 0.1% (≈0; any real rate → fix lengths, don't accept),
  (d) ≥ 20% of GRPO groups have nonzero within-group reward variance (learnable-rollout signal),
  (e) **search-invocation floor: ≥ 50% of trajectories issue ≥1 search** (review round 3,
  point 7 — a valid-action rate can pass while the model never searches, which would make every
  retrieval-conditioned method vacuous; also log the answer-turn distribution).
  Any miss → scale to Qwen3-4B (single pre-registered fallback). The step-100 ΔEM of the Wave-0
  GRPO run is a **secondary warning signal only** — weak ΔEM with a passing base-model gate
  reads as "GRPO struggles at 1.7B", a finding, not a model-selection criterion.
- **Checkpoint selection:** primary performance comparison = **fixed-budget FINAL checkpoint**;
  primary sample-efficiency comparison = **validation-AUC over the fixed budget**; secondary =
  best val-selected checkpoint (reported, never the headline). Full test set evaluated ONLY for
  these pre-specified checkpoints.
- **RQ3 statistics at 2–3 seeds (review round 3, point 9 — no p-value theater):** the
  pre-declared comparison form is (i) per-seed final-EM and val-AUC deltas reported with seed
  ranges, and (ii) the paired per-dataset breakdown, with 7 datasets × seeds as the effective
  replication unit for the multi-hop subgroup. Interpretation rules fire on these statistics,
  not on whichever cut looks cleanest.
- **RQ4 subgroup:** multi-hop datasets (hotpotqa/2wiki/musique/bamboogle) pre-registered as the
  primary mechanism subgroup for B0-vs-B1 deltas; single-hop (nq/triviaqa/popqa) as contrast.
- **RQ3 interpretation rules:** B1 > B1-shuffle ≈ B0 ⇒ timing/information content of the
  progress signal matters; B1 ≈ B1-shuffle > B0 ⇒ density/optimization effect, not supervision;
  B1 ≈ B1-shuffle ≈ B0 ⇒ this privileged signal doesn't help at this horizon (NOT evidence that
  progress reward models are useless in general). `shuffle_active_frac` must be reported next to
  any B1-vs-B1-shuffle claim, **with a plan B (review round 3, point 8): if it lands below ~30%
  at Wave 1, the 8-turn horizon stress test is PROMOTED from optional to required** — that is
  the setting where the shuffle control has room to discriminate timing from density.

## Phases

### Phase 0 — Project bootstrap & hygiene
1. Commit the UNCOMMITTED SP6 working tree of `external/verl-agent` + `reward_models` on an SP6
   branch in place (tripwires, HCAPO clip, SPA hook) — SP6 recoverable, fork cleanly cloneable.
2. Clone the fork to `~/xiaoxuan/Agentic-RL-CA/` (repo root = fork, branch `agentic-rl-ca`),
   add `credit_assignment/` + scripts/configs/docs/analysis dirs, seed README/LICENSE/.gitignore;
   include the 8-turn stress config in `configs/` from day 1.
3. Pull HCAPO + CARL (+ StepSearch) PDFs; lock exact formulas/constants into
   `docs/methods_note.md`.
4. Land `docs/CREDIT_ASSIGNMENT_CHECKLIST.md` (survey Tables 11–12 verbatim) + commitments map.

### Phase 1 — Task bring-up (no new algorithms)
1. Data prep → train/test parquet + fixed stratified `val_2048.parquet` (fixed seed).
2. Retriever standup on a worker; curl sanity.
3. **Toy-first gate (HARD GATE, user reviews trajectory dumps):** tiny run per existing
   condition (TRAIN_BATCH=8, GROUP=4) dumping per turn: prompt/observation, raw `<think>`+action,
   parsed validity, retrieved `<information>`, per-turn reward, EM →
   `outputs/search_toy/trajectory_<algo>.md`. Checks: non-thinking output fills the template's
   `<think>` sensibly within 512 tokens; tags parse through `search_projection`; format matches
   Search-R1-family trajectories; truncation ≈0 (<0.1%); **search-invocation rate ≥50% and
   answer-turn distribution logged** (degenerate-behavior check).

### Phase 2 — B1 / B1-shuffle step-reward shaping (RQ3 — first new code, smallest)
Env-side in `SearchEnv._get_reward`/`step` (`env.py:49-56` — replace the hard-coded intermediate
`return 0`), env-var gated; helpers live in `credit_assignment/`; all turn limits read from
config (no literal 4s):
1. **B1 — privileged answer-exposure signal** (NOT a ground-truth progress measure — useful
   retrievals may lack the answer string; answer strings may appear in useless retrievals — it
   is a *privileged, verifiable* shaping signal): `normalize_answer(gold)` in this turn's
   retrieved `<information>` (reuse `subem_check`, `utils.py:22-62`) → **first-hit-only** bonus
   +w_step (anti-farming). w_step=0.2 main; 0.5 ablation (deprioritized below B1-shuffle).
2. **B1-shuffle control** (the key experiment; first-hit-only means ≤1 positive step reward per
   trajectory): for each trajectory containing a B1 reward, **reassign that reward uniformly to
   another eligible NON-TERMINAL search turn in the same trajectory**, preserving count and
   magnitude; **never assign shaping reward to the terminal answer turn** (that would convert it
   into extra terminal reward). Trajectories with only one eligible search turn are necessarily
   identical under B1 and B1-shuffle — log **`shuffle_active_frac`** (fraction of B1-positive
   trajectories with ≥2 eligible search turns); low value triggers the pre-registered plan B
   (8-turn stress test required). Implementation: compute B1 rewards env-side, reassign
   driver-side before `compute_advantage`, seeded. Isolates timing/information content from
   reward density, scale, and critic-conditioning effects — especially needed since `gae_turn`
   whitens advantages batch-wide (record raw pre-whitening reward stats).
3. Action legitimacy stays on the existing `is_action_valid` penalty path
   (`core_turn_ppo.py:43-44`); optional duplicate-query validity extension.
4. `gae_turn` needs NO estimator changes (GAE recursion already handles intermediate rewards,
   `core_turn_ppo.py:66-77`; per-turn signal = `non_tensor_batch['rewards']`,
   `rollout_loop.py:405`). Log retrieval-hit rate + step-reward stats to W&B.

### Phase 2b — CARL (after core runs are healthy; design code-verified)
Two-phase tree rollout, driver-side state restore (search env state is fully driver-local and
deterministic given the query — snapshots are exact). Estimator code in
`credit_assignment/core_carl.py` (NOT in `gigpo/` — upstream stays untouched):
1. `envs.py`: `SearchMultiProcessEnv.get_states()/set_states()`; `env_manager.py`:
   `snapshot(i, depth)`/`restore_batch(...)` (env chat_history/turns + SearchMemory prefix +
   task). **Built EARLY as shared infrastructure** — the Phase-3b credit-alignment diagnostic
   needs exactly this prefix-resume capability; only the heavy two-phase tree loop stays
   deferred until core runs are healthy.
2. `rollout_loop.py`: refactor vanilla loop into shared `_turn_loop(...)` (vanilla path verified
   unchanged via gigpo search smoke); `carl_multi_turn_loop`: phase 1 = n0 sampled rollouts with
   per-step snapshots; phase 2 = n_total−n0 resumes (round-robin depths; first resumed action
   sampled, later turns greedy via per-turn `meta_info do_sample`); `turn_index` = global depth.
3. **Node identity (hardened):** `carl_dst_node = sha1(full next policy-visible tokenized prompt
   at t+1)`, NOT sha1(parent + projected action). In this env the next prompt is rebuilt from
   template + memory (projected action + retrieved info; raw `<think>` does NOT persist), so
   this often coincides with action-level merging — but the prompt hash makes state equivalence
   definitional, and a **hard unit assertion** verifies: rows merged into one node ⇒ their
   tokenized t+1 prompts are byte-identical. `dedup_key` config retained for ablation only.
4. `credit_assignment/core_carl.py::compute_carl_advantage`: dedupe rows by (traj_uid,
   turn_index) (guards `adjust_batch` copy-duplicates); `E[R(u)]` = mean terminal EM over
   trajectories through u; critical iff >1 distinct child; advantage = `E[R(dst)]−E[R(src)]` if
   src critical else 0 (zeroed advantage = loss exclusion, Seq-MIS precedent; optional
   `drop_noncritical` strict mode). Unit test on a hand-built toy tree. Metrics:
   critical_state_frac, kept_row_frac, tree_nodes, resumed_frac.
5. Enum/dispatch/no-critic + `algorithm.carl` yaml (n0, n_total=16, resume_schedule,
   include_root, drop_noncritical, norm_adv); assert gamma=1.0 + filter_groups off.
6. Smoke (batch 8, n_total=4) with tree-field dump → toy gate. Budget: CARL-lite N=16 ≈ GRPO n=8
   generated turns (~34/prompt at T=4); also token-fair scale (n_total≈8-10) vs GRPO n=5;
   consider vLLM prefix caching (verify it survives weight re-sync). Report both scales.
Risk: state-restore fidelity (test: resume + same action ⇒ identical reward/obs).

### Phase 2c — StepSearch (secondary tier, appendix)
1. Data: run their released data-construction pipeline over OUR NQ+HotpotQA train set
   (annotation LLM = decision with user: strong local Qwen vs API); fallback = HotpotQA-only
   subset, reported as such (this is why it is NOT in the matched main table).
2. Rewards: per-step retrieval-IG + redundancy penalty (formulas from PDF), injected via the B1
   env-side path with per-episode seen-docs/queries state. Vehicle: `gae` per paper.
3. Toy gate with per-step IG/redundancy dumps; unit test with known golden keys.

### Phase 3 — Full-set eval port (paper protocol; `trainer.val_only`, 3 patches)
1. `_validate` patch: per-TRAJECTORY EM (dedupe by traj_uid — current `test_score` is
   length-biased) → `val/{ds}/em`, `val-core/macro_em`, `val-core/micro_em`, `val/{ds}/avg_turns`.
2. Per-trajectory JSONL dump (gated by `trainer.validation_data_dir`) + aggregator →
   paper-table CSV/JSON (incl. turns min/median/max).
3. `eval_search_full.sh`: val_only on all 51,713 rows, val_batch~1024, checkpoints via HF-merge
   route (`scripts/model_merger.py` — world-size independent).
During training: fixed `val_2048.parquet`, test_freq 25–50, greedy — `val-core/macro_em` tracks
the paper metric. Protocol sanity: untrained Qwen3-1.7B vs paper no-RL baselines; tr1
Qwen2.5-7B cross-check (expect approximate agreement only — template differs).

### Phase 3b — Credit-alignment diagnostic (RQ2's quantitative teeth)
Without this, RQ2 is only a performance benchmark: "GiGPO > GRPO in EM" shows a better-trained
policy, not better-recovered intermediate credit. Offline diagnostic (never a training signal),
implemented in `credit_assignment/`:
1. Fix a subset of logged training trajectories (256–512) at a few fixed checkpoints per method.
2. For each turn prefix s_t, estimate the outcome-only continuation value
   `V̂(s_t) = (1/K) Σ_k R(τ_k | s_t)` by **resuming from the prefix** (the Phase-2b
   snapshot/restore infra) and sampling K continuations. **Continuations MUST be sampled from
   the same checkpoint that generated the trajectory** (else ΔV̂ is a value under the wrong
   policy and the comparison is invalid — review round 3, point 4).
3. Post-hoc step contribution `ΔV̂_t = V̂(s_{t+1}) − V̂(s_t)`; compare each method's assigned
   turn credit A_t against ΔV̂_t. **Statistics hardened (review round 3, point 4):** with EM ∈
   {0,1} and K=8, V̂ has std ≈0.17 near p=0.5 and per-trajectory Spearman over ≤3 turn-pairs is
   nearly meaningless — pre-commit to (i) **pooling turn-pairs across trajectories**, (ii)
   **bootstrapped confidence intervals** on the correlation, (iii) a **K=16 noise check** on a
   subset. Metrics: pooled Spearman, sign agreement, pairwise turn-ranking accuracy.
4. **Principled interpretation:** ΔV̂_t is the martingale-increment quantity — an ideal
   process-reward's increments equal exact advantages — so F8a reads as "distance from ideal
   credit," not an ad-hoc correlation target.
Cost: ~512 traj × ≤3 prefixes × K continuations per checkpoint — one worker, hours; slots into
Wave-4 gaps. Output = **figure F8a**, a headline candidate.

### Phase 3c — RQ1 length-weighting diagnostic (diagnose only)
Turn-advantage broadcasting gives a 300-token turn ~10× the policy-gradient mass of a 30-token
turn under global token-averaging (token PPO also scales with length, differently). No new
ablation — log per turn: response length, |advantage|, and effective loss weight
(Σ|A|·mask / batch token count); verify no large method-dependent length shift between `gae` and
`gae_turn`, and report the distributions alongside F7.

### Phase 4 — Real runs: wave schedule (max 8 parallel workers)
Each run = one 8×H100 worker with co-located retriever; worker launches need user OK per wave;
count existing workers first (`mlx worker list`). Ops: crash-resume loop; HDFS checkpoints with
byte-verified prune (worker /tmp ~300G); ledger rows + GPU-hours in worker_logs/workers.tsv.

**Wave 0 — feasibility gate (1 worker):** token-GRPO on Qwen3-1.7B, judged by the pre-registered
BASE-MODEL gate (baseline macro-EM ≥0.03, valid-action ≥90%, truncation <0.1%, ≥20% groups with
reward variance, search-invocation ≥50%); step-100 ΔEM is a secondary warning only. Continues as
the condition run if healthy; else fleet scales to Qwen3-4B.

**Wave 1 — RQ3 triad at EQUAL seeds (review round 3, point 1) + RQ1/RQ2 start (8 workers):**
| workers | RQ | run |
|---|---|---|
| 2 | 3 | B0 = `gae_turn` sparse, seeds s0/s1 (also RQ1's turn-PPO arm) |
| 2 | 3 | B1 (privileged step reward, w_step=0.2), seeds s0/s1 |
| 2 | 3 | **B1-shuffle, seeds s0/s1** — the key control must not be the noisiest arm |
| 1 | 1 | token-PPO (`gae`), s0 |
| 1 | 2 | GiGPO, s0 |

(HCAPO s0 moves to Wave 2 to fund B1-shuffle s1.) First RQ3 readout = B0 vs B1 vs B1-shuffle at
2/2/2 seeds, judged by the pre-declared statistics. B1/B1-shuffle need only Phase-2 shaping code,
so Wave 1 launches while CARL is implemented (GPU-free work).

**Wave 2:** HCAPO s0, B0/B1/B1-shuffle s2 (third seeds), GiGPO s1, token-GRPO s1, w_step=0.5
ablation if slots allow (the ex-IGPO compute funds the extra GiGPO/HCAPO/GRPO seeds).
**Wave 3:** CARL (2 budget scales, after core healthy), HCAPO s1 + remaining headline third
seeds (picked with user from curves), CA-component ablations (GiGPO step_advantage_w=0,
HCAPO ω=0, CARL gating-off), B2-lite if RQ3 supports it (or user overrides), StepSearch
(secondary) after annotation, **8-turn stress test if promoted by the shuffle_active_frac rule**.
**Wave 4:** full-set evals of pre-specified checkpoints (1 worker each, ~hours) + the Phase-3b
credit-alignment diagnostic runs, slotted into free workers; eval-harness validation runs during
Wave 1 (independent of training).

### Phase 5 — Report
Full-set greedy EM per condition + per-dataset breakdown; report per experiment-report-standard
(PDF + rerunnable script + assets with per-figure data/README), filled checklist appendix,
annotated trajectory examples, push to `RainyFields/Agentic-RL-CA`.

## Thread B — Progress-supervision hypothesis (RQ3–RQ5)

**Hypothesis: when does explicit intermediate progress supervision add value beyond outcome-only
credit assignment?** (Not "do we need PRMs" — B1's signal cannot answer that in the negative;
see interpretation rules.)

**Signal taxonomy:** B0 = sparse outcome only. B1 = **privileged answer-exposure signal** —
verifiable and ground-truth-anchored, but NOT a true progress measure (a useful retrieval may
lack the answer string; the answer string may appear in a useless retrieval). B1-shuffle = same
rewards, shuffled timing (the control that separates information content from density/scale
effects). B2-lite = **prompted step evaluator** (NOT a "learned PRM"). B2-full = trained
evaluator (the only arm that truly tests a learned reward model).

Design cautions: short 4-turn horizon limits effect size — RQ4's multi-hop subgroup is
pre-registered as the mechanism readout, and the `max_steps=8` B0/B1 **horizon stress test**
(config ready from day 1; promoted to required if `shuffle_active_frac` is low) directly asks
whether the marginal value of progress reward grows with horizon. Reward-farming mitigations:
first-hit-only, small w_step, trajectory-dump inspection. B-arms share ALL config with B0; only
the per-turn reward vector differs.

### B2 — Co-ReAct-based step judge (RQ5; user decision, SPA/HISR dropped)
Per turn: rubric generator (question + prefix → weighted criteria) → verifier scores the action
→ `weighted_score` s_t ∈ [0,1] → `r_t = w_step·s_t`, injected via the same per-turn path.
Served as vLLM endpoints beside the retriever (retriever HTTP client is the template); 2 extra
LLM calls/turn/rollout — logged as CA-overhead.

Verified findings (2026-07-13): **no released checkpoint or dataset** (HF profile empty; README
placeholder; release-request issue unanswered) — only code + prompts + training recipe. The
verify prompt already makes the LLM emit `weighted_score` (code currently discards it; ~2-line
surface). Trained on ~30k DR-Tulu deep-research branching points (multi-judge Borda + GRPO
listwise-Spearman); evaluated only on deep-research benchmarks; their own ablation shows an
UNTRAINED generator performs at-or-below plain ReAct.

Two tiers:
- **B2-lite = prompted step evaluator:** off-the-shelf Qwen3-8B/14B-Instruct with Co-ReAct's
  prompts (adapted to tag-protocol short QA), `weighted_score` as step reward. **Mandatory
  offline validation gate:** judge scores on B0/B1 toy trajectories must correlate with
  retrieval-hit and eventual EM before any RL run; else dead on arrival.
- **B2-full = separate follow-up project:** reproduce their GRPO judge-training on our domain
  (branching-point collection, multi-judge labels, API budget) — scoped only with explicit user
  go-ahead, not part of this plan's deliverables.
- **Shuffled-judge control** accompanies any B2 run.

## Experimental standard — survey checklist (arXiv 2604.09459, Appendix C)

All experiments follow the survey's Table 11 reporting checklist (14 items; Table 12 = scorecard
format). The survey's three systemic gaps — GPU-hours (0/41 papers), compute-controlled baselines
(0/41), variance (2/41) — are exactly what F4, the ledger accounting, and the seed policy target.

### Report figures plan
Every figure ships with backing CSV + provenance README.
1. **F1** Method/taxonomy schematic: turn structure + where each method assigns credit, on the
   survey's granularity×methodology grid (doubles as paper Fig 1).
2. **F2** Learning curves: val-2048 macro-EM vs step, mean±std where ≥2 seeds; panels by RQ.
3. **F3** Final full-set EM per dataset + macro avg, grouped bars with seed error bars.
4. **F4** Compute-controlled comparison: EM vs GPU-hours and vs generated tokens (CARL's two
   scales + GRPO n=5/8 give the iso-compute contour) — the survey's #1 gap.
5. **F5** RQ3 answer: B0 vs B1 vs **B1-shuffle** (vs B2-lite) curves + per-dataset ΔEM split
   single-hop vs multi-hop (RQ4 subgroup), reported in the pre-declared per-seed-delta form.
6. **F6** CA-overhead per method (extra forwards / rollout turns / judge calls + wall-clock).
7. **F7** Turns-per-trajectory distributions (min/median/max), train + eval.
8. **F8** Diagnostics: advantage-collapse rate (GRPO-family), retrieval-hit + search-invocation
   rates over training, `shuffle_active_frac`, CARL critical-state fraction, gae_turn
   truncated-episode rate, turn-length/loss-weight distributions (Phase 3c).
8a. **F8a — Credit alignment (headline candidate):** assigned turn advantage vs Monte-Carlo
   continuation-value delta ΔV̂_t, per method (pooled Spearman with bootstrap CIs / sign
   agreement / pairwise ranking) — the quantitative RQ2 answer, interpretable as distance from
   ideal (martingale-increment) credit.
9. **Appendix**: annotated example trajectories per method; filled checklist scorecard.

Table-11 commitments (abridged): model + HF revision pinned; data source/size + versioned
preprocessing; granularity & taxonomy family per method in methods note; episode-level baselines
= token PPO/GRPO same model; compute accounting per run from the worker ledger + CARL budget
scales; ablations (GiGPO w=0, HCAPO ω=0, CARL gating-off, B1 w_step) at minimum one per headline
claim; exact splits + metric definitions documented; **≥3 seeds on headline arms** (B0, B1,
B1-shuffle, token-GRPO, GiGPO + others picked with user); GPU-hours from ledger; CA-overhead
logged per method (HCAPO hindsight pass, CARL tree turns, StepSearch annotation + reward
compute, B2 judge calls); trajectory-length stats in the eval aggregator.

Durable actions: land the verbatim checklist in `docs/`; memory entry pointing at it; every
report gets a filled scorecard appendix.

## Verification

- Toy gate: per-algorithm trajectory dumps reviewed by user (HARD GATE) + degenerate-behavior
  checks (search-invocation floor, answer-turn distribution, truncation <0.1%).
- Horizon-config audit: grep-level check that no shaping/snapshot/eval/diagnostic code contains
  a literal turn-count; the 8-turn config runs the toy gate successfully.
- CARL unit checks: toy-tree backup/criticality/zeroing; **node-merge assertion (merged rows ⇒
  byte-identical t+1 prompts)**; duplicated-row case.
- B1-shuffle unit checks: reassignment preserves per-trajectory count and magnitude of nonzero
  rewards; terminal answer turn NEVER receives shaping reward; `shuffle_active_frac` computed
  correctly on synthetic cases.
- Credit-alignment diagnostic checks: prefix-resume determinism (same prefix + greedy ⇒ identical
  continuation); V̂ on a hand-built 2-turn case matches closed form; same-checkpoint sampling
  enforced by construction (checkpoint id recorded per trajectory and asserted at resume).
- Eval port: tr1 Qwen2.5-7B cross-check (approximate agreement expected — template differs) +
  untrained-Qwen3 vs paper no-RL baselines.
- Training-val curves in W&B per condition vs tr1 reference shape.

## Risks

- Qwen3-1.7B below all published operating points — Wave-0 pre-registered gate; fallback
  Qwen3-4B.
- Non-thinking 1.7B may fill template `<think>` poorly — toy gate checks; thinking mode is the
  documented fallback (needs ≥1024 response length + template-clash check).
- `gae_turn` no-bootstrap on truncated episodes (V_next=0 at the turn cap) — monitor frequency
  (F8).
- CARL tree rollouts = largest engineering risk — deliberately sequenced after core runs;
  CARL-lite first.
- `shuffle_active_frac` may be structurally low at 4 turns — pre-registered plan B (8-turn
  stress test promoted to required).
- StepSearch annotation may shift data distribution — secondary tier by design.
- Retriever throughput at batch×group×turns — sanity-curl + latency watch; multiple
  `search_url`s supported if one server saturates.
- Scope risk: the MPU + pre-registered rules are the commitment; CARL/StepSearch/B2/8-turn are
  staged extensions, each behind its own gate.
