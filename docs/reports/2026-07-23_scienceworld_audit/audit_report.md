# ScienceWorld structure audit — Wave-0 M5 (round 1)

Date: 2026-07-23 · scienceworld 1.2.3 (pip) + OpenJDK 17 · simplification `easy`, step limit 300
Scripts (rerunnable): `scripts/scienceworld/audit_structure.py` (raw pass), `scripts/scienceworld/analyze_audit.py` (analysis + dumps)
Assets: `assets/` — task_map.csv, gold_traces.jsonl (90 traces), families.csv, turn_caps.csv, subgoal_semantics.md, horizon_hist.png, html/ (12 annotated trajectory dumps)

## What was audited

All 30 tasks enumerated; for each: train/dev/test variation splits, and gold-action-sequence
replay on 3 sampled train variations (90 traces total) recording per-step action, observation,
reward, cumulative score, and done. Plus 3 targeted probes: off-gold failure semantics,
observation-leakage scan, `task` command surface.

## Findings

### 1. Task-family map (10 families, 30 tasks)
Variation counts vary enormously (4 train variations for identify-life-stages-2 up to 692 for
inclined-plane-friction-named-surfaces; find-* and test-conductivity families have 150–450 train
variations). Full table: `assets/task_map.csv`, family mapping in `assets/families.csv`.
Split convention: env-provided train/dev/test variation indices (≈50/25/25) — our "unseen
variations" eval = env dev/test splits, no custom splitting needed.

### 2. Oracle reward semantics (fixes the granularity every future PRM arm must match)
- Score is **cumulative 0–100 per episode**; per-step `reward` = score delta. Subgoal firings are
  score increments at specific gold steps.
- **Magnitudes are heterogeneous and task-specific** (42 distinct values observed, 1 to 72): the
  oracle is NOT a uniform per-subgoal unit reward — any PRM-ladder or reward-channel arm must
  treat magnitudes as given (or renormalize explicitly and say so).
- Firing density on gold paths: median 0.27 firings/step (min 0.05, max 0.62) — dense relative to
  terminal-only but far from per-step.
- All 90 gold replays reach exactly 100 with done=True; no negative deltas on gold paths; every
  trace has ≥2 firings.
- **Catastrophic failure channel (off-gold probe): a wrong `focus on` action yields score −100 and
  immediate episode termination.** Reward-design consequences for our arms: (a) terminal-success
  arms must define success as score==100 (not score>0) and decide the mapping for −100 episodes
  (recommend: terminal reward 0, episode ends — mirroring ALFWorld failure, keeping terminal
  reward in {0,1}); (b) the dense channel (§8 B/C suite) contains a −100 event the summed-scalar
  arm will feel as a huge negative spike — exactly the kind of asymmetry the redistribution-vs-
  separation experiment is about; document, don't smooth away silently. The `focus` action is
  also an obvious reward-hacking / instant-death surface for RL exploration — a valid-action-rate
  and focus-death-rate counter belongs in training metrics.

### 3. Horizon distributions → proposed per-family turn caps (proposal §3 D3 rule: ~2× gold length)
| family | sampled gold-len range | proposed cap (2× max) |
|---|---|---|
| lifespan | 6–9 | 18 |
| find-thing | 5–16 | 32 |
| chemistry-mix | 11–30 | 60 |
| life-stages | 10–30 | 60 |
| electricity | 9–45 | 90 |
| plant-growth | 66–85 | 170 |
| thermal-measure | 18–94 | 188 |
| genetics | 123–130 | 260 |
| matter-state | 20–143 | 286 |
| inclined-plane | 53–202 | 404 |

The 20–100+ horizon claim of the proposal is confirmed and then some: within-ScienceWorld spread
is ~40× (5 → 202 gold steps), which gives RQ2 within-rung horizon strata for free.
**Flag for the harness (M1/M3):** caps of 260–404 turns × full-history transcripts will not fit any
reasonable context at 4B — the rung-4 context policy (oldest-observation truncation) is not
optional but load-bearing for 4 of 10 families, and cap finalization must wait for the strong-model
turns-to-success calibration (proposal: 95th percentile) rather than the 2×-gold heuristic alone;
gold paths contain filler (`look around`, repeated `wait`) an agent may not need — or may exceed.

### 4. Oracle isolation
- Observation text on gold paths contains no `score`/`reward`/`subgoal`/`points` strings (0 hits
  in 90 traces). The score channel exists only in the API `info` dict — the agent prompt must be
  built from `obs` only, which our D3 grammar already does.
- `task` command exposes the task description only (already in the prompt) — no oracle state.
- Remaining isolation work for the rubric arm (later round): audit our env *wrapper* once written,
  to guarantee `info['score']` never reaches prompt construction.

### 5. Inspection dumps
12 annotated HTML trajectories (green rows = subgoal firing with magnitude, red = negative delta)
covering the global shortest/median/longest traces plus each family's longest: `assets/html/`.
Checklist items coverable at this stage all pass (reward fires where expected; observations well-
formed; no leakage). Prompt-render/turn-boundary/parser checks apply to the future env wrapper,
not the raw env — deferred to the wrapper's inspection pass.

## Decisions this audit fixes
- Subgoal granularity = score-delta events with heterogeneous magnitudes (see §2) — recorded as
  the reference the §8B PRM ladder must match.
- Terminal-success definition for core-matrix arms: success ⇔ score==100; −100 focus-death maps to
  terminal 0 + episode end (proposed; confirm at W1 manifest).
- Eval splits = env-provided dev/test variation indices.
- Per-family turn caps: provisional table above; FINALIZE after strong-model rollout calibration
  (open item).

## Open items (next steps in M5 / M1)
1. Strong-model rollouts (gpt-oss-120b) on a variation sample → turns-to-success distribution →
   final per-family caps + context-policy stress test for the 4 long families.
2. Env wrapper for the trainer (agent_system) with the D3 ReAct grammar + loss-masked observations;
   then re-run the 3-sample inspection on wrapper-rendered transcripts (prompt/parser/turn-boundary
   checklist items).
3. Thought-line synthesis rule for gold→SFT transcripts (uniform, minimal; decide once, at M8).
4. Variation-count imbalance (4 → 692 train variations per task): decide train-time task sampling
   (uniform-per-task vs uniform-per-variation) before W1 — affects what "epoch" means on this rung.
5. Focus-death rate + valid-action rate counters wired into training metrics.
