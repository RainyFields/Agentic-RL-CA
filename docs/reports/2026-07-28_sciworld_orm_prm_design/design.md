# ScienceWorld ORM-vs-PRM design — rung-4 experiment spec (round 1)

Date: 2026-07-28 (SF) · scienceworld 1.2.3 · Qwen3-4B-Instruct-2507, no SFT cold start
Status: SPEC FROZEN (grilling session 2026-07-28; decisions recorded as Amendments A12–A13)
Probe scripts (rerunnable): `assets/probe_optional.py`, `assets/seq_counts.py` (env: `~/xiaoxuan/envs/sw_inspect`)
Prior audit this builds on: `docs/reports/2026-07-23_scienceworld_audit/audit_report.md` (M5)

## Motivation

Given a process reward and an outcome reward engineered to be *return-equivalent on every
trajectory* — same total, different temporal placement — does localizing reward at the steps
where progress occurs improve credit assignment at ScienceWorld horizons? Two environment
versions, identical dynamics/termination, differing only in when the (identical-total) reward
is delivered; one estimator implementation consumes both, so the contrast is purely reward
placement.

## Environment & scope

- Simplification = **`easy`** (`noElectricalAction, openDoors, selfWateringFlowerPots,
  teleportAction`) — verified identical to the M5 audit setting, so all audit artifacts
  (gold traces, caps, inspection dumps) remain valid. NOTE: an earlier plan draft listed
  `openContainers` instead of `noElectricalAction`; that draft string is void.
- **Turn caps are per-task** (not per-family): cap = 2 × max sampled gold length (3 train
  variations/task, audit gold traces). Roster = tasks with cap ≤ ~190 (session enumeration
  below). Rationale: family caps are naming-group accidents (matter-state's 286 is driven by
  boil alone; its siblings are 54–128).
- Splits: env-provided **train** variations for training, **dev** for during-training val,
  **test** untouched until final reports.

### Task table (all 30; roster = 25 included, rule-primary)

N_seq / N_unord = sequential (required) vs unordered-optional subgoal counts at variation 0
(`assets/seq_counts.py`). Gold firings = aggregate score-delta events on the 3 audit gold
replays (ordered + unordered combined).

| task | gold lens | cap (2×max) | N_seq | N_unord | gold firings | in? | reason if out |
|---|---|---|---|---|---|---|---|
| boil | 81, 41, 143 | 286 | 3 | 15 | 8, 6, 10 | NO | cap 286 > 190 |
| change-the-state-of-matter-of | 26, 20, 27 | 54 | 3 | 18 | 4, 6, 4 | yes | |
| chemistry-mix | 24, 18, 22 | 48 | 1 | 6 | 5, 6, 7 | yes | |
| chemistry-mix-paint-secondary-color | 11, 15, 15 | 30 | 1 | 5 | 5, 5, 5 | yes | |
| chemistry-mix-paint-tertiary-color | 30, 24, 20 | 60 | 2 | 10 | 8, 8, 8 | yes | |
| find-animal | 16, 12, 14 | 32 | 2 | 4 | 5, 5, 5 | yes | |
| find-living-thing | 10, 12, 10 | 24 | 2 | 4 | 4, 5, 5 | yes | |
| find-non-living-thing | 7, 7, 5 | 14 | 2 | 4 | 4, 4, 3 | yes | |
| find-plant | 12, 8, 12 | 24 | 2 | 4 | 4, 5, 5 | yes | |
| freeze | 28, 64, 27 | 128 | 3 | 5 | 5, 4, 5 | yes | |
| grow-fruit | 85, 67, 81 | 170 | 1 | 28 | 20, 14, 18 | yes | |
| grow-plant | 68, 68, 66 | 136 | 4 | 17 | 10, 10, 10 | yes | |
| identify-life-stages-1 | 29, 27, 30 | 60 | 4 | 5 | 8, 6, 8 | yes | |
| identify-life-stages-2 | 12, 16, 10 | 32 | 4 | 5 | 5, 7, 4 | yes | |
| inclined-plane-determine-angle | 60, 99, 172 | 344 | 1 | 10 | 11, 11, 11 | NO | cap 344 > 190 |
| inclined-plane-friction-named-surfaces | 53, 71, 64 | 142 | 1 | 10 | 11, 11, 11 | YES | cap 142 qualifies under the ≤190 rule (family-level enumeration had wrongly excluded it; rule is primary — surfaced to user 2026-07-28, default accepted) |
| inclined-plane-friction-unnamed-surfaces | 86, 54, 202 | 404 | 1 | 10 | 11, 10, 10 | NO | cap 404 > 190 |
| lifespan-longest-lived | 8, 6, 6 | 16 | 1 | 2 | 3, 3, 3 | yes | |
| lifespan-longest-lived-then-shortest-lived | 9, 7, 7 | 18 | 2 | 2 | 4, 4, 4 | yes | |
| lifespan-shortest-lived | 8, 8, 6 | 16 | 1 | 2 | 3, 3, 3 | yes | |
| measure-melting-point-known-substance | 29, 21, 27 | 58 | 3 | 25 | 10, 11, 11 | yes | |
| measure-melting-point-unknown-substance | 94, 92, 94 | 188 | 3 | 25 | 7, 6, 11 | yes | |
| melt | 27, 35, 41 | 82 | 3 | 15 | 5, 6, 6 | yes | |
| mendelian-genetics-known-plant | 125, 123, 130 | 260 | 1 | 50 | 27, 19, 29 | NO | cap 260 > 190 |
| mendelian-genetics-unknown-plant | 126, 130, 130 | 260 | 1 | 50 | 25, 29, 28 | NO | cap 260 > 190 |
| power-component | 15, 9, 15 | 30 | 2 | 5 | 5, 4, 5 | yes | |
| power-component-renewable-vs-nonrenewable-energy | 15, 30, 26 | 60 | 3 | 5 | 6, 5, 5 | yes | |
| test-conductivity | 22, 45, 27 | 90 | 2 | 7 | 4, 6, 7 | yes | |
| test-conductivity-of-unknown-substances | 25, 39, 35 | 78 | 2 | 7 | 6, 7, 6 | yes | |
| use-thermometer | 22, 18, 24 | 48 | 3 | 8 | 10, 9, 10 | yes | |

Included-roster cap range: **14–188** (session shorthand said "18–188"; 14 is exact).
Caps are noisy (3 sampled variations) → runtime **cap-hit-rate counter per task**; high rates
trigger revisit, not silent acceptance. Long families (boil, genetics, inclined-plane) join a
later round once the truncation policy is validated in anger.

## Reward design

**Progress variable.** `P_t = max_{s≤t} score_s`, clipped ≥ 0, where `score_s` is the env's
native aggregate score (`info['score']`, 0–100). Monotone by construction, so both versions
below are well-defined on deaths, regressions, and timeouts.

- **Version-PRM (process):** per-step reward `r_t = ΔP_t = P_t − P_{t−1}` (≥ 0).
- **Version-ORM (outcome):** single terminal reward `r_T = P_T`, zero elsewhere.

**Return equivalence (proof sketch).** Undiscounted return of PRM = Σ_t ΔP_t = P_T (telescoping)
= return of ORM, on *every* trajectory — including focus-death (score → −100 → P frozen at its
max, clip irrelevant after first nonneg value), off-gold score regressions (max holds), and
timeouts. The two versions differ **only** in temporal placement of an identical total.

**Why the native aggregate, not ordered-only.** The original draft excluded unordered-optional
("canonical-path") credit. Probes killed that: **N_seq ∈ {1,2,3,4} across all 30 tasks; 9 tasks
have N_seq = 1**, and that single required state is the terminal answer action (e.g.
mendelian-genetics: "focus on the correct answer box"; grow-fruit: "focus on the grown fruit").
Ordered-only PRM would be *literally identical* to ORM on those tasks and 2–4-event-sparse on
the rest, while the audit's observed density (3–29 firings/episode) lives overwhelmingly in the
unordered channel. The unordered subgoals are state predicates crediting *alternative methods'*
waypoints (boil lists stove/oven/blast-furnace/hot-plate/fire-pit heater options; gold completes
5/15), i.e. closer to method-agnostic progress credit than gold-path imitation. Key native-score
property (verified 8/8 tasks probed): **completing the required sequence forces score = 100
regardless of unordered completion** — success stays cleanly defined, unordered credit only
differentiates partial trajectories.

**Ordered-progress is a logged metric, not a reward**: parse `getGoalProgressStr()` Sequential
flags per step; runtime assertion `score==100 ⟹ all sequential true` (any exception task
surfaces itself). Post-hoc ordered-only analyses (e.g. the 11-task N_seq≥3 subset) remain
recoverable from logs.

**Failure handling.** Env's wrong-`focus` insta-death (−100 + termination) maps to reward 0
via the clip; death is penalized in both versions only through foregone future reward. No
negative rewards anywhere. **Registered contingency (no preemptive change):** if the smoke
gate shows high focus-death with many all-zero groups (clip removes the death/timeout
distinction = within-group variance the gate measures), reintroduce failure = −ε (small, not
−100). Keyed to gate metric 3; decided at gate time, documented if invoked.

## Algorithm — one estimator, two reward placements

Turn-level group-normalized reward-to-go (critic-free). For rollout i in a group of G on the
same (task, variation), turn t:

    G_{i,t} = Σ_{s≥t} r_{i,s}   (γ = 1)
    A_{i,t} = (G_{i,t} − μ_t) / σ_t   broadcast to that turn's response tokens

μ_t, σ_t computed **across the rollouts still alive at turn t**, with a variance floor
(σ_t ← max(σ_t, ε_σ)). Under ORM, G_{i,t} = P_T for all t → the estimator **degenerates
exactly to vanilla trajectory-GRPO**; one implementation serves both arms. Under PRM,
G_{i,0} = P_T as well, so the arms share early-turn advantages and diverge only where
reward placement can matter — the designed contrast. Groups never mix variations. No SFT
cold start (plausibility prior: BEACON trains Qwen2.5-1.5B/7B-Instruct on ScienceWorld with
PPO/GRPO, no cold start; 1.5B-GRPO ≈ 32 score / 21% success — underspecified config, hence
the gate below, not a substitute for it).

## Context policy (part of the MDP; identical across arms, train and eval)

16k total: 256-token per-turn generation cap; ~15.7k prompt budget. **Three-tier deterministic
truncation, K = 20:**

1. Always keep: system + task description + one-shot example + last K=20 turns complete
   (Thought/Action/Obs).
2. Turns older than K: **Action line only** (thoughts/observations dropped — old actions are
   the state-defining trace).
3. Overflow fallback: drop oldest Action lines, with loud logging.

Arithmetic: 20×(256 gen + ~200 obs) ≈ 9.1k + preamble ~1.2k + worst-case action tail
168×~13 ≈ 2.2k → ~12.5k ≤ 15.7k, ~3k slack for fat observations. K=24 fits nominally (~14.4k)
but thins the slack against exactly the pathological cases the rule insures — K=20
pre-registered. With most roster caps ≤ 90, tier 2 rarely fires; it is insurance for the
long tail (freeze 128, grow-plant 136, grow-fruit 170, measure-melting-unknown 188).
Truncation-tier firing rates are logged training metrics.

## Gate — zero-shot smoke before any training launch (W0 audit-with-a-gate)

One worker, base Qwen3-4B-Instruct-2507, the **exact production wrapper/prompt/truncation**,
G=5 rollouts/group, ~200–400 episodes stratified across all 25 tasks (variations stratified
by horizon). Metrics: (1) gradient-bearing group fraction (nonzero within-group reward
variance, graded channel), (2) valid-action rate, (3) focus-death rate, (4) per-task score
distribution, (5) cap-hit + truncation rates; plus the 3-trajectory wrapper inspection
(prompt render, turn boundaries, no `info['score']` leakage) per the M5 checklist.

Pre-registered outcomes:
- **GO**: gradient-bearing fraction ≥ 25–30%.
- **Middle (10–25%)**: cheap levers, then re-measure — G→16 on this rung and/or oversample
  smoke-alive families. No SFT discussion yet.
- **NO-GO (<10% even at G=16 on easy families)**: stop; SFT cold-start discussion with smoke
  data in hand.

**Prompt freeze rule**: the smoke is the last prompt-tuning window; one documented prompt
revision + re-smoke allowed on formatting failure; prompt freezes at gate-pass.
**Dead tasks stay in the roster** (pruning would change the task distribution mid-benchmark
and break cross-arm comparability; per-task unlocking curves are a result). Instrument:
per-task gradient-bearing fraction over training; if the roster-wide fraction is below the
gate's GO bar after ~50 steps, that is a pre-registered *alarm for the analysis window*, not
an automatic change.

## Sampling & batch

- **Uniform-per-task, then uniform-variation-within-task** (resolves M5 open item 4; train
  distribution matches macro-averaged eval weighting). Train-variation counts range 4–692;
  per-variation sampling would overweight find-*/test-conductivity ~50:1.
- **G = 8** (per-turn group stats need more than the smoke's G=5), **64 groups/step → 512
  trajectories/step**. Parallel env processes = 512 JVMs, instantiated once and reused
  (`envs.py` pattern); ~256–512 GB on a 1.6 TB worker — smoke-test checkbox; documented
  fallback 32×8=256.
- **Straggler policy = accept + measure**: rollout is turn-synchronized, so step latency is
  set by the longest trajectory in the batch. Smoke logs per-round utilization; horizon
  stratification only if measured step time is unacceptable, decided *before* launch, never
  mid-run.

## Eval protocol

- Val on env **dev** variations: fixed pre-drawn set, 8/task × 25 tasks = 200 episodes
  (all-if-fewer), greedy decoding, same wrapper/caps/truncation as training.
- Every 25 steps; TOTAL_STEPS = 150; extend by documented resume if still climbing.
- Metrics (both arms, identical): macro mean score, success rate (score==100), per-task
  breakdowns; diagnostics: ordered-progress, focus-death rate, cap-hit rate, valid-action
  rate, per-task gradient-bearing fraction, truncation-tier rates.

## Bookkeeping

- W&B project **`ca-rung4-4b`** (kept out of pristine `ca-benchmark` until this graduates
  to a benchmark wave); runs `orm_grpo_s0`, `prm_rtg_s0`; single seed s0 (standing decision).
- HDFS: `/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/logs/rung4_sciworld_{orm_grpo,prm_rtg}/`.
- Inherited defaults, **flagged untuned**: LR 1e-6, KL 0.001, γ=1.0, MICRO_BSZ=1, bf16
  (from `configs/protocol_asearcher_8turn_4b.sh` lineage).

## Open items

1. ~~`inclined-plane-friction-named-surfaces` in/out~~ RESOLVED 2026-07-28: IN (rule-primary,
   roster = 25; user notified, no objection).
2. ε_σ variance floor value and the alive-at-t normalization edge cases (single survivor at
   turn t) — implementation decisions, to be recorded in the estimator code + decision log.
3. −ε failure contingency: value of ε if invoked (gate-time decision).

## Gate outcome (2026-07-28) + amendment A14

**GO.** 360 episodes / 72 groups, zero-shot Qwen3-4B-Instruct-2507 through the production
wrapper: gradient-bearing group fraction **0.736** (threshold 0.25); mean P_T 0.411; success
0.147; focus-death 0.256; cap-hit 0.597; valid-action 0.698; 256-clip 0.0005; no dead tasks;
no train-fleet env crashes. Prompt frozen as of gate-pass (no revision was needed). −ε
contingency not triggered. Full numbers: `gate/gate_report.json`; inspection trajectories on
HDFS `logs/rung4_gate/` (uniform sampling left find-living-thing and identify-life-stages-1
with 0 groups — accepted, siblings healthy).

**A14 (user-approved 2026-07-28), from gate measurements:**
1. `MAX_PROMPT_LENGTH` 16384 → **12288** (padding width only): measured real prompt tokens
   p50 2251 / p99 5072 / max 5667 over 17,409 turns — the 3-tier rule keeps prompts far
   below the ceiling; 16k padded rows overflowed the gate worker's 983G /tmp via Ray object
   spill during the update phase. Manager budget follows as 12288−768; empirically never binds.
2. **Horizon-stratified batches**: each step draws its whole batch from one stratum
   (short ≤32 / mid ≤90 / long ≤188 caps; rotation short,short,mid,mid,long ∝ sizes 10/10/5,
   per-task marginal stays uniform 1/25). Measured straggler cost motivated this: ~66 min
   rollout/step with every step paying the 188-round tail; expected rounds now ≈86.
3. `use_dynamic_bsz` + `ppo_max_token_len_per_gpu=32768` for update/logprob passes
   (compute packs real tokens; MICRO_BSZ retained as fallback).
4. JVM ops hardening (implementation, not protocol): `-XX:-UsePerfData` (hsperfdata race
   broke py4j port parse under mass spawn), `-Xmx2g`, auto-reboot of dead JVMs with the
   episode terminated as `env_crash` (return-equivalence preserved: ORM pays banked P_T).
