# Structured Benchmark of Credit-Assignment Methods for LLM Agents — Project Overview

**Saved 2026-07-23** from Xiaoxuan's proposal v1 (2026-07-19), verbatim in §1–§14 below, followed by
the **Amendments** log (decisions taken after v1) and the **Submodule index** used to execute the
project as sequential, individually-monitored modules. This document is the single source of truth
for project scope; per-round execution plans and reports live in `docs/reports/`.

---

## Proposal v1 (2026-07-19) — verbatim

Date: 2026-07-19 · Target deadline: September 1, 2026 (set 2026-07-19; ~6.3-week runway. If Sept 1 is an internal buffer ahead of ICLR 2027's likely late-Sept CFP dates, the post-Sept-1 slack absorbs seed top-ups, the 14B check, and polish — see §13 #9.) Compute: 8 workers × 1 node each (multi-node untested; plan assumes single-node runs throughout) Status: v1 for review. A dedicated grilling/red-team session follows; §13 pre-lists the open questions it should attack. Preliminary results: a completed 5-method pilot campaign on ALFWorld (Qwen3-1.7B) motivates the roster, the recipe, and several design choices — summarized in §1.1 so this document is self-contained.

### 1. Research questions and claims

RQ1 (benchmark). A structured, controlled comparison of credit-assignment (CA) algorithms for multi-turn LLM agents: ≤6 methods × 4 environments × multi-seed, one identical training recipe and one identical eval harness, so that cross-cell differences reflect the CA mechanism, not the harness.

RQ2 (horizon/sparsity scaling). Does the advantage of structured CA over flat trajectory-level advantages (GRPO) grow with horizon length and reward sparsity? The four environments form a horizon ladder (2–4 → 8–14 → 10–30 → 20–100+ turns) and a sparsity spectrum (terminal EM → terminal success → oracle dense subgoals available).

RQ3 (robustness across reward regimes). Which CA methods are robust when the reward signal degrades — noisy LLM-judge/rubric rewards instead of clean EM (side experiment A), and across the process-supervision quality ladder from oracle dense rewards down to nothing (side experiment B)?

RQ4 (redistribution vs. separation — side experiment C). When an environment provides both intermediate and terminal rewards, do we need return-equivalent redistribution machinery at all, or is it sufficient (or better) to keep terminal and intermediate rewards as separate channels with separately estimated advantages?

Downstream goals (not in the ICLR scope, but shaping design choices). (a) Use the benchmark to pick the CA direction worth pursuing; (b) the filtering pipelines double as synthetic-dataset generation infrastructure; (c) if results identify where dense signal helps (which granularity, which horizon band), move toward query-specific rubric generation at key points + compaction for credit assignment. Every design choice below that has a "keep the door open" flavor (turn-level logging, granularity-matched PRM arms) serves (c).

#### 1.1 Preliminary results (completed ALFWorld pilot, July 2026)

Setup. ALFWorld text environment, ReAct full-transcript prompt (one-shot example, thinking disabled), Qwen3-1.7B, full-parameter RL from a shared behavior-cloning initialization (unseen success 0.440); one identical hardened RL recipe across all arms; greedy validation every 5 steps with checkpoint selection at the validation peak; "unseen" = a held-out 134-game out-of-distribution split evaluated under one shared protocol.

| Method | Unseen success | Steps → val ≥ 0.85 | Avg turns at end |
|---|---|---|---|
| PPO w/ turn-level GAE + learned critic | 0.963 | 30 | 9.7 |
| GiGPO (episode group + anchor-state step advantage) | 0.955 | 35 | 8.6 |
| GRPO (vanilla, token-level broadcast) | 0.881 | 70 | 10.4 |
| GRPO (seq-mean aggregation variant) | 0.866 | 85 | 9.6 |
| HCAPO (hindsight credit) | 0.791 | 135 | 13.8 |
| SPA (dense learned progress reward; best repaired variant) | 0.672 | never | ~20 |
| BC initialization (baseline) | 0.440 | — | 18–34 |

Findings that shape this plan. (1) Structured turn-level credit wins on both axes — final performance and speed (2–4.5× fewer steps to threshold than flat GRPO); RQ2 asks whether horizon amplifies exactly this gap. (2) Generalization is uniform: unseen ≈ seen-peak − 3–9 points for every method, ranking preserved — so unseen numbers are the honest currency. (3) Vanilla GRPO is stronger than its reputation once mini-batch sizing is fixed — it is a serious baseline, not a strawman. (4) Episode efficiency tracks method quality: average turns compress from 18–34 (BC) to 8.6–9.7 for the best methods, so turns-to-success is a leading diagnostic, not just a secondary metric. (5) Dense learned progress rewards are fragile: a faithful online transfer of SPA collapsed within ~15 steps by reward-hacking its grounding bonus (farming an always-valid action for per-turn income; completing the task ends the income stream), and even the repaired variant plateaued ≥0.2 below every outcome-reward method — the cautionary result motivating side experiment C's focus on constrained ways of adding dense signal. (6) Entropy inflation under GRPO is an exploration-phase artifact, not a pathology — nothing stops it cheaply, every arm self-normalizes; the recipe is train through it, checkpoint densely, select by validation peak. (7) All pilot numbers are single-seed — the multi-seed protocol in §5 exists to fix precisely this.

Hardened infrastructure carried forward. Large PPO mini-batch (1024; smaller provably collapses the trust region), eager-mode vLLM rollout (CUDA-graph capture caused deterministic crashes), NaN tripwires armed in every run, checkpoints every 5 steps with retention raised (a peak checkpoint was once lost to aggressive pruning), W&B heartbeat as liveness ground truth.

Theory grounding used by the side experiments. Reward redistribution means decomposing a terminal reward R into per-step rewards r̃_t under the return-equivalence constraint Σ_t r̃_t = R per trajectory; this provably leaves the objective J(π) unchanged for every policy — the optimal policy is preserved and only learning dynamics change — whereas an unconstrained dense reward (an LLM-judge PRM, SPA's progress bonus) can shift the optimal policy and invites hacking. The ideal redistribution pays the increment of estimated success probability, r̃_t = Ŵ_t − Ŵ_{t−1} (telescoping restores Σr̃_t = R); a perfect success-probability PRM is therefore exactly a critic, and its perfection is policy-relative — it goes stale unless continually updated alongside the policy. These three facts define the arms of side experiments B and C (§8).

### 2. Design overview

```
                     ┌─ Wave 0: harness + data + inspection (no training)            → G0: env audits — ASearcher go/no-go
                     │                                                                    + ScienceWorld structure/oracle audit
                     ├─ Wave 1: anchor pair (GRPO + turn-level PPO) × ALL 4 envs,
                     │          1 seed; candidate screening on SearchQA in parallel   → G1: first horizon read; roster freeze
Core matrix ─────────┼─ Wave 2: remaining roster (GiGPO + admitted candidates)
                     │          × all 4 envs, 1 seed                                  → G2: provisional headline matrix
                     ├─ Wave 3: seed expansion on the surviving matrix                → G3: seeded headline table
                     ├─ Side track: ScienceWorld oracle-dense arm runs INSIDE Wave 1 (3 of 4 ladder rungs by G1);
                     │             A (noise) + rubric / redistribution arms alongside W2–W3, gated at G1
                     └─ Wave 4: analysis lock, figures, writing
                     Every wave closes with its own analysis window + dated report before the next commits compute.
```

Everything trains with one RL recipe (the hardened recipe from the pilot, §1.1: ppo_mini_batch_size=1024, enforce_eager=True, KL-to-π_base, entropy 0.001, checkpoint+greedy-val every 5 steps, NaN tripwires, select-by-validation-peak) and evaluates through one harness (§6). Per-method knobs are only what defines the method (advantage estimator, γ, critic).

### 3. Base model, thinking mode, prompt structure (Decisions D1–D3)

D1 — Base model. ⚠️ Correction to the draft: Qwen3 has no 3B or 7B — the dense lineup is 0.6B / 1.7B / 4B / 8B / 14B / 32B (3B/7B are Qwen2.5 sizes, which is what Search-R1/StepSearch used). Recommendation:

| Role | Model | Why |
|---|---|---|
| Primary (full matrix) | Qwen3-4B (non-thinking) — decided 2026-07-19 | Big enough to get off the ground on rungs 3–4 (1.7B likely lands in the empty-band regime on ASearcher); small enough that the full matrix fits the budget (§12). |
| Extra-environment anchor | Qwen3-1.7B, ALFWorld only | Direct comparability with the pilot (§1.1); one seed per finalist method is enough. |
| Scale check, time permitting | Qwen3-14B, finalists × 1–2 envs | Chosen over 8B for a bigger scale contrast; ~3.5× the 4B run cost, so it comes only out of post-G3 slack (§12 spend lines). |

Run the full matrix at one size. A 2-size × full-matrix design doubles cost for a scaling claim that a targeted finalist-only scale check delivers more cheaply. (Qwen2.5-3B/7B-Instruct would maximize literature comparability with Search-R1-style baselines, but the 4B / 14B / 1.7B lineup above is the recorded decision as of 2026-07-19; the harness is model-agnostic if that ever reverses.)

D2 — Thinking mode: OFF. Four reasons. (1) Confound: thinking mode adds large, variable per-turn token counts, entangling the horizon variable (turns) with per-turn length — exactly the axis RQ2 needs clean. (2) CA granularity: turn-level methods (GiGPO anchors, gae_turn) assume the turn is the semantic unit; a 2k-token hidden reasoning block inside each turn changes what "turn-level credit" means per-arm in uncontrolled ways. (3) Continuity: the entire pilot campaign (§1.1) and its BC policy are non-thinking. (4) Cost: rollout length dominates wall-clock. The ReAct Thought: field already provides a visible, short reasoning slot. If reviewers ask, one probe arm (best method + GRPO on SearchQA, thinking on) is a cheap Wave-4 addendum — do not put it in the matrix. Checkpoint choice implied by D2: Qwen3 ships two lineages — the original (Apr 2025) hybrid checkpoints, where one model serves both modes via the enable_thinking template switch, and the 2507 refresh, which splits into Instruct (non-thinking only, no <think> blocks at all) and Thinking (always thinks) variants. Recommendation: Qwen3-4B-Instruct-2507 as the primary — purely non-thinking by construction (no template-switch leakage risk under RL) and a stronger checkpoint than the April 4B. The 1.7B continuity anchor stays on the original hybrid checkpoint with enable_thinking=False (no 1.7B exists in the 2507 line), matching the pilot campaign exactly.

D3 — Prompt structure: ReAct, frozen. Full-history transcript prompt, one-shot example, Thought:/Action: per turn — the format already validated in the ALFWorld campaign. QA rungs use the Search-R1-style turn grammar — the policy emits Thought: then either <search>query</search> or <answer>…</answer>; the environment then appends retrieved passages wrapped in <information>…</information> as the observation. Embodied rungs use the environment's text-command action space, with the env's textual feedback as the observation (no wrapper needed; ALFWorld/ScienceWorld transcripts already delimit it). Two rules that matter for training: (1) observation tokens (<information> blocks, env feedback) are loss-masked — they're state, not action; log-probs, KL, and advantages are computed only over policy-generated tokens, and the turn segmentation that CA methods operate on must draw the turn boundary so each turn = one policy emission + its observation; (2) if the policy emits <information> or other observation-mimicking tokens itself, that's format drift — the inspection checklist (§4) and a parser-side reject catch it. The system prompt, tool syntax, answer delimiter, turn cap, and decoding settings are fixed at training time and never varied at eval (§6 layer 2). Turn caps and context budgets are per-rung, not global — horizons differ by ~50× across rungs, so a single shared threshold would either truncate ScienceWorld episodes or leave SearchQA free to ramble. Set each rung's thresholds once at W0 (calibrated against the ~95th percentile of strong-model turns-to-success from the filter/diagnostic rollouts, so the cap doesn't clip the turn distribution the §9 figures analyze), then freeze them identically across training and every eval of that rung's checkpoint:

| Rung | Turn cap | Context policy |
|---|---|---|
| 1 SearchQA | ~8 (typical 2–4 + headroom) | full history; top-k retrieval with per-passage length cap — fits context easily |
| 2 ALFWorld | 50 (existing recipe) | full history, fits comfortably |
| 3 ASearcher | 30 (by design) | the binding case: per-doc length cap + truncate oldest <information> blocks first, keeping all Thought/Action lines; log every truncation event |
| 4 ScienceWorld | per task family, ~2× oracle subgoal-path length (finalize at W0) | full history; oldest-observation truncation if needed |

What must stay uniform is within a rung: every benchmark evaluated with a given checkpoint uses that rung's cap and context policy (a per-benchmark cap inside one eval suite silently truncates the turn distribution — §6). Cross-rung differences are fine — they're part of each environment's definition, and the horizon analysis treats them as such. This is also the future hook for the compaction research direction.

### 4. Environments and training data (four rungs)

Common properties: every rung has a verifiable or oracle reward; no external paid API needed for the reward; horizons vary by ~50× across rungs.

**Rung 1 — Multi-hop SearchQA (horizons 2–4).** Train source: PeterJinGo/nq_hotpotqa_train (HotpotQA + NQ), through the four-stage filter:
Stage 0 — subsample ~20–30k of 169k, stratified by source and hop label; dedup (exact + fuzzy) against the test splits of all seven eval benchmarks used anywhere in the project.
Stage 1 — closed-book probe: base model, no tools, n=4 at training temperature; drop items any/(≥2 of 4) sample answers correctly (parametric-memory items teach "don't search"). Expect NQ cut 30–50%, HotpotQA lightly.
Stage 2 — pass-rate rollouts: full training-time agent loop (same retriever, corpus, turn cap, temperature), n=8; drop pass=8/8 (zero GRPO advantage), hold 0/8 for stage 3, keep (0,1). Log turns-to-answer per item → free horizon-stratification metadata.
Stage 3 — label audit on the 0/8 tail with the strong model (gpt-oss-120b, generous budget, semantic-equivalence judging, second referee only on its uncertain calls): valid-hard → keep-able; mislabeled/unanswerable → drop; strong-model-also-fails → drop.
Assembly: ~3–5k items = (0,1) band core + ≤25% one-turn survivors ("search once and stop" anchor) + ~10% valid-hard 0/8 (learnable later; partially compensates the pro-GRPO bias of pass-rate filtering — say so in the paper). Stratify the draw by observed turns-to-answer, not hop labels. Freeze; all arms train on it identically. Report in/out counts per stage.
"Will the training set be too small? Should we keep the filter?" — Keep it; 3–5k is well-matched, not too small. At train_batch_size=16 tasks/step × ~200–300 steps, one run consumes ~3–5k task presentations — a 3–5k unique set means ~1 epoch of unique data, which is the healthy regime for on-policy RL (fresh rollouts each visit; repeat visits at different policy stages are informative, not wasted). The filter is precisely what buys gradient signal per rollout: unfiltered NQ would spend 30–50% of compute on zero-advantage or anti-search items. If stage 2 leaves <2.5k in the (0,1) band, loosen by raising n to 16 (widens what registers as non-zero pass rate) before loosening the closed-book criterion — never re-admit stage-1 drops.
Eval: HotpotQA / 2Wiki / Musique / Bamboogle standard test splits, unfiltered. Reward: EM with alias normalization, verifiable.

**Rung 2 — ALFWorld (horizons 8–14).** No data filter; default train split; eval = seen (128-ep val) + unseen (134 games). Reward: task success. Already done at 1.7B for 5 methods — §1.1 is this cell's pilot. New work here follows the wave structure (§7): anchor pair at 4B in Wave 1, remaining roster in Wave 2, seeds in Wave 3, reusing the environment configs, BC init recipe, and the harmonized unseen-eval protocol as-is. Known trap carried over: seen-val is eval_in_distribution; all conclusions rest on the dedicated unseen pass.

**Rung 3 — Long-horizon web search (horizons 10–30, capped).** Train source: ASearcher-train-data (inclusionAI/ASearcher; synthesized hard QA, validated multi-stage, trained up to 128 turns in the original). Our setup: ~30-turn cap, local-RAG-style corpus (budget), which changes answerability — hence:
Step 1 — environment alignment before any filtering. Strong-model answerability check on 200–300 items inside our exact setup. If the answerable fraction is low, fix the environment (bigger corpus / cached web snapshots) before touching training data.
Step 2 — fuzzy dedup against WebWalkerQA, GAIA validation, FRAMES.
Step 3 — closed-book probe (expect few drops; it's our guarantee, not theirs).
Step 4 — solvability diagnostic (the gate). n=8 pass-rate on ~500 items in the exact training setup. Regimes: band (0<pass<1) ≥25–30% → run the full protocol and proceed; 10–25% → select into the band aggressively + supplement (easier tier if difficulty metadata exists, n→16, or short SFT warm-start on a few hundred strong-model trajectories); <10% → do not burn budget — conclusion is "needs a curriculum bridge from rung 1" or "needs a bigger base model," and the rung is dropped from the ICLR matrix with the histogram reported as a finding.
Assembly: ~2–5k from the learnable band with deliberate difficulty spread (not mid-band only), frozen across arms.
Eval: ASearcher test + WebWalkerQA; text-GAIA / FRAMES as zero-shot transfer (no training implications; buys the generalization paragraph). Reward: EM, verifiable.

**Rung 4 — ScienceWorld (horizons 20–100+, variable).** No data filter; train/eval on unseen task variations. Reward: oracle dense subgoals + terminal success — the only rung with ground-truth intermediate reward, which is why the ScienceWorld suite (side experiments B + C) lives here. The W0 structure audit (§7 Wave 0) resolves this rung's open decisions: subgoal-reward granularity (what exactly the oracle emits per step — the granularity all PRM-ladder arms must match), per-task-family turn caps, and the oracle-isolation check for rubric/learned arms. Reward-channel convention: core-matrix arms train on terminal success only on this rung — that keeps the sparsity ladder clean (every rung's core comparison is outcome-reward CA) and makes the ScienceWorld GRPO/gae_turn cells double as the floor and implicit rungs of the §8B ladder. The oracle dense signal is reserved for the §8 suite and for eval-time diagnostics.

**Expert trajectories & SFT cold start (all rungs).** Every rung gets a small expert-trajectory set, re-rendered into the exact frozen agent format (the §3 D3 grammar: full-transcript ReAct prompt, Thought:/Action: structure, <search>/<information>/<answer> for QA rungs, observations loss-masked), used for a short SFT pass that produces that rung's shared initialization π_base. Rationale: the pilot showed initialization matters (its ALFWorld arms all start from BC); initializing some rungs from raw instruct and others from BC would confound every cross-rung comparison — so the previously open asymmetry question is resolved by harmonizing: SFT cold start on all four rungs. One SFT recipe (same lr, epochs, loss masking) everywhere; one shared π_base per rung used identically by every CA arm; each SFT checkpoint is evaluated and reported as its own baseline row — the floor every RL arm must beat.

| Rung | Expert source | Reformatting notes |
|---|---|---|
| SearchQA | Existing expert data from previous training (inventory + audit at W0); top up with strong-model (gpt-oss-120b) rollouts in our exact env only where gaps remain, EM-verified (~1–2k total) | Re-render existing data into the frozen grammar + 4B chat template and dedup against all eval splits (§4 stage 0); verify <information> blocks come from our retriever, not open web |
| ALFWorld | Pilot's replay-BC dataset (put↔move projection fix) | Already in format; re-render with the 4B chat template |
| ASearcher | Two candidate sources: (a) rl-research/dr-tulu-sft-data (DR Tulu's long-horizon research-agent SFT trajectories) filtered + remapped; (b) strong-model rollouts in our capped 30-turn setup, EM-verified (~300–500) | DR Tulu caveats: trajectories are open-web, multi-tool, and long-form-report-oriented — filter to short-answer-style items, remap tool calls to our <search> grammar, and either accept the observation-distribution mismatch (format/behavior prior only) or replay queries through our retriever and keep trajectories that still reach gold. Decide (a)-vs-(b) mix at W0 from a 20-trajectory audit; (b) doubles as the rung-3 step-4 warm-start option |
| ScienceWorld | Env-provided gold/oracle action sequences rendered as ReAct transcripts (+ strong-model rollouts for variety) | Thought lines: synthesize minimally and uniformly (decide once at W0); subgoal events annotated for the §8 suite but stripped from the SFT text (oracle isolation) |

Expert generation reuses the W0 harness (same adapters, parsers, caps), so format match is by construction; every expert set passes the same 3-sample manual inspection (below) before SFT. Cost: generation is inference-only and SFT passes are short — ~5 nd total across rungs, absorbed in the W0/W1 margin (§12).

**Manual trajectory inspection (all rungs).** After each preprocessing pipeline lands, sample 3 items per dataset stratified by horizon (short / median / long by observed turns-to-answer where available) and render full trajectories (prompt, per-turn thought/action/observation, reward events, final parse) as HTML dumps. Inspection checklist, checked off per sample in the W0 report: (1) prompt renders exactly as designed, one-shot example intact; (2) action parsing — no silent tool-call failures; (3) turn boundaries align with the trainer's turn segmentation (what GiGPO/gae_turn treat as a step); (4) reward fires where expected, gold answer/alias set is sane; (5) truncation policy behavior visible and correct; (6) no eval-set leakage artifacts (question text overlap). Re-run the same 3-sample inspection on the first RL checkpoint's rollouts in each wave — format drift under RL (thought degeneration, delimiter abuse) is a known failure mode and cheap to catch this way.

### 5. CA methods

Core roster (in the matrix from day one):

| Arm | Granularity | Mechanism | Pilot evidence (§1.1, unseen) |
|---|---|---|---|
| GRPO (token-level broadcast) | trajectory → tokens | episode-group advantage | 0.881 |
| GRPO (turn/seq-mean aggregation) | trajectory → turns | same, seq-mean-token-sum | 0.866 |
| PPO gae_turn | turn | learned critic + cross-turn GAE | 0.963, fastest to threshold |
| GiGPO | turn | episode group + anchor-state step advantage | 0.955, seen 1.000 |

(Token vs. turn GRPO counts as two arms of one method family; the matrix budget treats them as 1.5 — the turn variant can be dropped after Wave 1 if it stays within noise of token-level, as it did at 1.7B.)

Candidate list (screen on SearchQA before admission) — living doc:

| Candidate | Why it's on the list | Screening bar |
|---|---|---|
| ArCHer-style off-policy utterance critic (arXiv 2402.19446) | Same design point as gae_turn (turn-level critic + token-level PG) but with an off-policy TD critic over a replay buffer — a ~100× sample-efficiency claim worth testing as a drop-in critic swap | beats GRPO unseen at ≤½ the env samples, or beats gae_turn outright |
| SweetRL-style turn critic w/ training-time info (arXiv 2503.15478) | Asymmetric actor-critic: critic sees privileged info at training time — orthogonal axis to ours | beats GRPO on unseen, 1 seed |
| StepSearch / StePPO information-gain rewards (arXiv 2505.15107, EMNLP 2025) | Purpose-built for search agents; step-wise reward from retrieval information gain — natural rung-1/3 competitor | beats GRPO on unseen, 1 seed |
| HCAPO | Already tested: 0.791 unseen, slowest — presumption: excluded unless SearchQA reverses it | must beat token-GRPO to re-enter |
| VinePPO-style MC turn advantages | Critic-free turn-level advantages via rollouts-from-prefix — the direct MC estimator of the success-probability increments that ideal redistribution pays (§1.1 theory grounding); expensive | beats GiGPO at comparable compute |
| SPA | Tested; adapted-online plateaus at 0.672 — excluded, kept as a documented negative result | — |

Admission rule (Gate G1): a candidate enters Wave 2 only if it beats token-GRPO on the SearchQA unseen-eval suite (1 seed) by more than the pilot's observed run-to-run noise band — a confirmation seed joins in Wave 3 — and the total roster stays ≤5 methods (budget §12). Everything else is reported as a screening result. The living doc (List of Candidate CA algorithms) records every candidate + why kept/cut.

Multi-seed protocol: Seeds are staged by wave (§7): Waves 1–2 run seed 0 everywhere; Wave 3 adds seed 1 across the surviving matrix (priority: headline pairwise cells → top-3 methods + GRPO baseline) and seed 2 only where the first two disagree in sign. Seeds fixed a priori (0/1/2); seed controls data order, sampling, and init of any critic head. Screening runs: 1 seed. (The Sept-1 deadline is why the ceiling is 2–3 seeds rather than 3 everywhere — §12.) [AMENDED: see Amendments A4.]
Checkpoint/selection rule fixed a priori for every arm identically: greedy val every 5 steps on the seen/val split, select peak, report unseen at that checkpoint (the pilot protocol, now with checkpoint retention raised so the peak is never pruned again).
Report mean ± range across seeds; method-vs-method comparisons via bootstrap over eval episodes within seed + sign consistency across seeds (claim ordering only when all available seeds agree in sign or the pooled bootstrap CI excludes 0; sign disagreement triggers the 3rd seed).
Single-seed history is a known limitation of the pilot (§1.1 finding 7) — Wave 2 exists partly to fix that.

### 6. Format-consistency harness (build first — Wave 0)

One eval harness, per-benchmark adapters, two scorers (~few hundred lines). Built before any training run so filter-stage rollouts (rung 1 stage 2, rung 3 diagnostic), training, and eval share one format by construction. Five layers:

1. Task schema — normalize data, not the agent. One adapter per benchmark → {question_text, gold_answers[] (with aliases), answer_type, metadata (hops/difficulty)}. All benchmark quirks absorbed at data-load time; the agent never sees benchmark identity.
2. Agent interface — frozen, identical to training. System prompt, tool syntax, answer delimiter, turn cap, decoding = the training configuration, unchanged across every eval. Zero-shot, one prompt, everywhere. Only the question string varies (formatting instructions embedded in GAIA questions are part of the question — keep verbatim).
3. Environment — one execution setting per checkpoint, documented. Rung-3 checkpoints eval in the capped-turn, fixed-corpus setup everywhere. WebWalkerQA/GAIA/FRAMES were built for live open-web agents → estimate the answerable fraction in our environment with one strong-model pass and report scores alongside that ceiling. text-GAIA = our published list of kept item IDs (no canonical definition exists; our filter is the definition).
4. Answer extraction & normalization — one implementation. Single delimiter parser; single SQuAD-style normalizer; official alias sets via adapters; GAIA scored by its official quasi-EM type-specific rules.
5. Metrics — dual-track. Official track per benchmark (EM/F1 for HotpotQA/2Wiki/Musique/Bamboogle; GAIA quasi-EM; LLM-judge for WebWalkerQA/FRAMES per their protocols) → headline transfer table, comparable to literature. Uniform track: one pinned LLM-judge equivalence check (same prompt, same judge model+version, released) applied to every benchmark → all internal cross-benchmark and cross-arm comparisons, including the horizon/CA-slope analysis. Sanity check: the two tracks must agree closely on the EM benchmarks; a large gap is itself a finding.

Secondary metrics (turns-to-answer, search calls, tokens) logged by the same harness code everywhere, with each rung's turn cap applied uniformly across that rung's entire eval suite so the turn distribution isn't silently truncated per-benchmark (caps differ across rungs by design — see the per-rung threshold table in §3 D3).

### 7. Experiment waves, intermediate results, and gates

Each wave ends with a dated intermediate report written to the project, so results are consumable before the next wave commits compute.

**Wave 0 — Infrastructure, data, baselines (0 training node-days, heavy CPU/inference).** Build: harness + adapters + dual scorers; retrieval stack for rungs 1 & 3 (corpus, retriever, caching); rung-1 filter stages 0–3; rung-3 steps 1–4 (through the solvability diagnostic); ScienceWorld + ALFWorld env configs at 4B; expert-trajectory collection + reformat and the per-rung SFT cold-start checkpoints (§4); trajectory-inspection dumps (3/dataset + checklist, applied to expert sets too). Also: ScienceWorld structure audit — not just trajectory samples, but an explicit map of the environment: enumerate task families and their train/unseen variation splits; document the oracle's per-step subgoal reward semantics (when it fires, at what granularity, with what magnitudes, and how it composes with terminal success); measure horizon distribution per task family; and inspect full trajectories with subgoal-firing events annotated inline. This audit is load-bearing: it fixes the reward granularity every PRM-ladder arm must match (§8B), the per-task-family turn caps (§3 D3), and the isolation guarantee that rubric/learned arms never see oracle subgoal state. Also: SOTA reference table — current published numbers for HotpotQA/2Wiki/Musique/Bamboogle (Search-R1, StepSearch, ZeroSearch lineage), ASearcher test/WebWalkerQA/GAIA/FRAMES (ASearcher paper), ALFWorld (GiGPO paper lineage), ScienceWorld RL baselines — with base model + metric + eval setting columns so nothing is compared across mismatched settings. Plus base-model zero-shot ReAct baselines and post-SFT cold-start baselines (the true π_base rows) on all evals with our harness — the floors every RL number is judged against, and the closed-book-probe calibration. Gate G0: harness passes inspection checklist on all 4 rungs; rung-1 filtered set frozen; rung-3 diagnostic histogram in hand → decides whether ASearcher is in regime (i), (ii), or (iii); rung-4 structure audit complete → subgoal granularity + turn caps fixed. Deliverable: W0 report — data audit tables (in/out per filter stage), diagnostic histogram, ScienceWorld structure audit (task-family map, oracle reward semantics, horizon distributions, annotated subgoal trajectories), SOTA table, baseline table, 12 inspected trajectories.

**Wave 1 — Anchor pair on all four environments.** The two anchor arms — token-GRPO (flat baseline) and PPO gae_turn (best structured method from the pilot) — run on all four rungs, 1 seed each, every cell initializing from its rung's shared SFT cold-start checkpoint (§4): 8 runs ≈ ~31 node-days (ASearcher cells gated by G0). In parallel on free workers: the cheap screening track — up to 3 candidates × SearchQA × 1 seed (~6 nd) — and the ScienceWorld oracle-dense arm (ground-truth subgoal reward + terminal, ~5 nd; needs only the G0 structure audit, no judge infra). Since Wave 1's own ScienceWorld cells already instantiate the flat-GRPO floor and the gae_turn implicit rung, this one extra run means 3 of the 4 §8B PRM-ladder rungs exist by G1 — only the rubric arm waits. W1 total ≈ ~42 nd. Why anchors-first breadth: (a) it produces the earliest possible read on the core RQ2 claim — Δ(structured − flat) across the entire horizon ladder — before most of the budget is committed; (b) it debugs env infrastructure on every rung while there are only two configs to blame; (c) it calibrates per-run cost per rung for the §12 re-forecast. Analysis window → W1 report: anchor-pair Δ vs. horizon figure (v0 of the RQ2 money plot), preliminary 3-rung PRM ladder (oracle vs. implicit vs. none), per-rung wall-clock actuals + budget re-forecast, format-drift inspection on first RL checkpoints (§4), candidate screening table. Gate G1: freeze the Wave-2 roster (≤5 methods incl. anchors) + all per-method hyperparameters. Late additions banned — they'd fork the recipe.

**Wave 2 — Remaining roster on all four environments.** The rest of the roster — GiGPO, turn-GRPO aggregation variant, any admitted candidate — joins on all four rungs, 1 seed: ≤3 methods × 4 envs = ≤12 runs ≈ ~47 node-days, plus ALFWorld 1.7B continuity anchors (finalists × 1 seed, ~6 nd). Analysis window → W2 report: provisional full heatmap (1-seed), horizon-slope figure v1 with all methods, post-hoc figures F3–F5 v0. Gate G2: seed-expansion priorities for Wave 3 (which methods, which cells) + spend/cut decision.

**Wave 3 — Seed expansion.** 2nd/3rd seeds for the surviving matrix, in priority order: (1) the headline pairwise-comparison cells; (2) top-3 methods + GRPO baseline across all rungs; (3) 3rd seed [AMENDED: 3 seeds a priori on headline cells — Amendments A4]. Analysis window → W3 report: seeded headline table (mean ± range), sign-consistency calls, final figure set, side-track results.

**Side track — experiments A, B, C (phase 1 inside Wave 1; phase 2 alongside W2–W3; details §8).** Phase 1 (inside W1): the ScienceWorld oracle-dense arm (above) — with W1's GRPO and gae_turn ScienceWorld cells, 3 of the 4 §8B ladder rungs are done by G1. G1 gate for phase 2: if the oracle arm shows no headroom over terminal-only, rescope the remaining suite before spending. Phase 2 (alongside W2–W3): A — rubric-noise ablation on SearchQA (~4 runs, ~8 nd); rubric-PRM arm (B2) + redistribution/decoupled arms (C3, C4) ≈ 4–5 runs, ~20–25 nd. Results fold into the W3 report as their own section.

**Wave 4 — Analysis lock + writing.** Figures locked (§9), uniform-judge pass over all eval outputs, dual-track sanity check, limitations section, optional probes (thinking-mode addendum, 14B scale check) only if the gates freed budget.

### 8. Side experiments (side track, alongside Waves 2–3)

**A — Rubric-reward-noise ablation (SearchQA).** Question: does a CA method's advantage survive when the terminal reward becomes a noisy judge? Redistribution and turn-level GAE inherit whatever noise is in the terminal signal they redistribute; a method that shines under clean EM may amplify judge noise across turns. Design: stay on SearchQA where ground-truth EM exists; eval always with clean EM; corrupt only the training reward. Arms: token-GRPO (flat baseline) × best turn-level method from G1. Corruption settings: (i) realistic — an actual LLM-judge/rubric score replacing EM; (ii) controlled — EM with symmetric label flips p ∈ {0.1, 0.3}; (iii) optional asymmetric/calibration bias (false-positive-heavy) if (i) shows judge optimism. 1 seed, judge + flip p=0.3 first; p=0.1 only if budget frees. Clean-EM runs reused from W1/W2 as the 0-noise point. Readout: unseen-EM degradation vs. noise level, per method — the slope is the deliverable. Hypothesis to test: turn-level machinery degrades faster than flat GRPO (it redistributes noise), which would qualify every "structured credit wins" claim for the rubric-reward future. Cost: ~4 new runs ≈ 8 nd.

**B — PRM-quality ladder (ScienceWorld).** Question: how good does a process reward have to be before it helps, and what does a perfect one buy? Arms (same base model, data, budget, and — critically — the same reward granularity as the oracle): 1. Oracle PRM — ScienceWorld's ground-truth subgoal reward. Ceiling. Runs in Wave 1 (§7 side track phase 1); identical to experiment C's arm 2, so one training run serves both. 2. Rubric PRM — LLM judge scores each step against a rubric (RaR-style), provably blind to the oracle subgoal state (prompt receives only the transcript; audit the env wrapper to confirm no oracle leakage into observations). Phase 2 — deferred until judge infra + isolation audit are ready and G1 confirms the oracle arm has headroom worth chasing. 3. Implicit/learned credit — gae_turn critic and GiGPO anchors, terminal-reward-only (reused from Waves 1–2). Policy-tracking learned signal vs. oracle. 4. No intermediate feedback — flat GRPO on terminal reward (reused from Wave 1). Floor. Readout: final success + sample-efficiency curves as a function of process-supervision quality. Connects directly to the theory grounding in §1.1: arm 3 is the "policy-tracking critic = continually-updated PRM" point; arm 2 tests the frozen exogenous signal, where staleness/hacking is predicted.

**C — Redistribution vs. separated reward channels (ScienceWorld, shares arms with B).** Question (RQ4): given both intermediate and terminal rewards, is return-equivalent redistribution machinery necessary, or does simply keeping the channels separate do as well? Arms: 1. Terminal-only (= B4, reused). Floor. 2. Summed scalar — r_t = dense_t + terminal, standard GAE over the merged channel (the naive default everyone ships). = B1's oracle-PRM run (Wave 1) — shared, not duplicated. 3. Redistributed terminal — terminal reward redistributed to steps under the return-equivalence constraint Σr̃_t = R via estimated success-probability increments r̃_t = Ŵ_t − Ŵ_{t−1} (§1.1 theory grounding), added to the untouched dense channel. 4. Decoupled advantages — separate advantage estimates per channel (A_dense from the dense stream, A_term from group/GAE machinery on terminal only), combined as A = A_term + α·A_dense with α swept coarsely {0.3, 1.0}. Interpretation grid: if 4 ≈ 3 > 2, redistribution's benefit is really just "don't let the terminal signal drown/distort the dense one" — separation suffices and is simpler. If 3 > 4, the return-equivalence constraint itself (the policy-invariance guarantee, §1.1) earns its complexity. If 2 ≈ 3 ≈ 4, the whole question dissolves at this horizon — also worth knowing before building rubric-redistribution machinery. The SPA collapse (§1.1 finding 5) is the cautionary tale for arm 2-style unconstrained summing: document the connection. Cost for B+C combined: B1=C2 runs inside Wave 1 (~5 nd, counted there); phase-2 new runs = B2 + C3 + C4(×2 α values) ≈ 4–5 runs × 1 seed ≈ ~20–25 nd (B3/B4=C1 reused from the core matrix; 2nd seeds only if budget frees).

### 9. Evaluation outputs and figures

Headline: (F1) method × environment heatmap/table (unseen/OOD success or uniform-track score, mean ± range over seeds), with a normalized companion (Δ vs. token-GRPO per cell) — the cross-env relative-performance view. (F2) CA-advantage vs. horizon: x = environment ordered by median horizon (also within-dataset horizon strata), y = Δ over GRPO — the RQ2 money figure.
Post-hoc (within-dataset horizon stratification): (F3) violin plot — turns per episode by dataset × method (models differ in how long they take). (F4) EM/reward vs. number of gold turns/hops, one line per method — does performance decay with required depth, and whose decays slower? (F5) actual turns used vs. gold turns, one line per method — who over-searches vs. under-searches relative to required depth. Gold-turn metadata: hop labels for rung 1 + observed strong-model turns as a proxy elsewhere; stage-2 rollout logs supply the distributional metadata for free.
Transfer: dual-track table (official metric + uniform judge) with answerability ceilings for rung-3 benchmarks (§6 layers 3+5).
Side experiments: noise-degradation slopes (A); performance-vs-supervision-quality ladder with sample-efficiency curves (B); the 4-arm redistribution grid (C).
Secondary metrics on every figure's companion table: steps-to-threshold, avg turns, valid-action rate — the pilot (§1.1 findings 4–5) showed these diagnose method quality and reward hacking earlier than success curves.

### 10. SOTA reference table (shell — filled in Wave 0)

| Benchmark | Best published (method, base) | Metric | Setting caveats |
|---|---|---|---|
| HotpotQA / 2Wiki / Musique / Bamboogle | Search-R1 / StepSearch / ZeroSearch lineage — pull current numbers | EM | retriever + corpus differ per paper — record them |
| ASearcher test / WebWalkerQA | ASearcher paper | EM / judge | open-web vs. our capped local setup — report with our ceiling |
| text-GAIA / FRAMES | ASearcher + originals | quasi-EM / judge | our text-GAIA item list ≠ theirs |
| ALFWorld | GiGPO paper lineage + our pilot (0.963 unseen @1.7B, §1.1) | success | seen/unseen split conventions differ across papers |
| ScienceWorld | RL baselines from lit | score/success | task-variation splits differ |

Rule: never quote a SOTA number without its base model, retriever/corpus, and split in the same row; mismatched settings are the main way this table lies.

### 11. Papers to catch up (reading list with why)

SWEET-RL (arXiv 2503.15478, code) — turn-level credit via a critic with privileged training-time information; candidate arm + the "asymmetric information" axis our matrix doesn't cover.
ArCHer (arXiv 2402.19446) — hierarchical two-time-scale RL: off-policy TD critic at the utterance level + on-policy token-level PG inside each turn, with a ~100× sample-efficiency claim over token-level PPO; the critic swap is a live candidate arm (§5).
StepSearch (arXiv 2505.15107, EMNLP 2025) — step-wise PPO with information-gain intermediate rewards for search agents; candidate arm for rungs 1/3.
CompactionRL (arXiv 2607.05378) — RL with context compaction for long-horizon agents; directly relevant to the rung-3 context-policy decision and the downstream compaction-for-CA direction.
SAO — "Single-Rollout Asynchronous Optimization for Agentic RL" (arXiv 2607.07508) — identity confirmed 2026-07-19; reading-list only for now, not incorporated into the matrix or infrastructure plans.
Context for the CA landscape: From Reasoning to Agentic: Credit Assignment in RL for LLMs (2604.09459) — recent survey-style treatment worth mining for missed candidate methods before G1.

### 12. Compute budget

[AMENDED: deadlines removed — see Amendments A1. Node-day figures below retained as per-run cost estimates only.]
Per-run cost assumptions (Qwen3-4B, single node; calibrate against W1 actuals and re-forecast at G1): extrapolated from the pilot (~2–2.5 nd per 300-step ALFWorld run at 1.7B) × ~1.5 for 4B, adjusted for episode length and env latency: SearchQA ≈ 2 nd · ALFWorld ≈ 3.5 nd · ScienceWorld ≈ 5 nd · ASearcher ≈ 5 nd.

| Wave | Node-days |
|---|---|
| W1 anchor pair × 4 envs + screening + ScienceWorld oracle arm | ~42 |
| W2 remaining roster (≤3 methods) × 4 envs + 1.7B ALFWorld anchors | ~53 |
| W3 seed expansion (ceiling) | ~62 |
| Side track phase 2 (A: 8; B2 + C3–C4: 20–25) | ~28–33 |
| Total | ~185–190 |

Cut lines, in order: (1) ASearcher regime (iii) at G0 → drop all rung-3 cells across waves (–~45 nd) and report the diagnostic as a finding; (2) W2 roster to 4 methods (–~15 nd); (3) W3 2nd seeds to top-3 methods only (–~15 nd); (4) C4's α-sweep to a single α (–5 nd). Spend lines if under budget: 3rd seeds across the matrix → 2nd seeds for the side track → Qwen3-14B finalists on ALFWorld+SearchQA (~40 nd at ~3.5× the 4B run cost) → thinking-mode probe.

### 13. Pre-registered open questions (agenda for the grilling session)

1. Base model choice — decided 2026-07-19: Qwen3-4B primary, 14B time-permitting, 1.7B ALFWorld-only (D1). Still open: is 4B actually strong enough for rung 3, or does the G0 diagnostic force the curriculum-bridge conclusion?
2. Retrieval stack for rungs 1/3 — which corpus + retriever? [RESOLVED 2026-07-23: reuse the Search-R1 stack — e5_Flat.index 61 GB + wiki-18.jsonl on HDFS, retriever_serve.sh; ASearcher answerability under wiki-18 is exactly the G0 step-1 check. Amendments A6.]
3. Rung-3 context policy at 30 turns × retrieved docs (truncation vs. doc-caps) — and whether that choice already entangles with the compaction research direction.
4. Is 2 seeds enough for ASearcher cells given run-to-run stochasticity documented in the Mode-A history? [Superseded by Amendments A4: 3 seeds a priori on headline cells.]
5. Screening validity — does SearchQA (shortest horizon) screening systematically eliminate methods whose advantage only appears at long horizon? (Mitigation: any candidate with a long-horizon-specific mechanism can appeal to a single ScienceWorld screening run — budget permitting.)
6. Judge costs and pinning for the uniform track + rubric arms. [RESOLVED 2026-07-23: gpt-oss-120b everywhere, self-hosted, frozen prompts — Amendments A3.]
7. BC-initialization asymmetry — resolved 2026-07-19: SFT cold start on all four rungs from format-matched expert trajectories (§4). Residual: how much SFT is enough per rung. [RESOLVED 2026-07-23: earliest-90%-of-plateau + shared entropy floor rule — Amendments A8.]
8. The pass-rate filter's pro-GRPO bias — is the 10% valid-hard slice enough compensation for critic-based methods that extract signal from all-fail groups?
9. Deadline. [RESOLVED 2026-07-23: all deadlines dropped — Amendments A1.]
10. SAO paper identity — confirmed 2026-07-19: arXiv 2607.07508; reading-list only.

### 14. Milestone summary

[AMENDED: dates removed per A1 — milestones are now gate-ordered, not calendar-ordered.]
G0: harness + frozen rung-1 data + rung-3 diagnostic (ASearcher go/no-go) + ScienceWorld structure/oracle audit → G1: W1 report — anchor pair across the full horizon ladder (first RQ2 read); roster frozen (≤5) → G2: W2 report — provisional 1-seed headline matrix → G3: W3 report — seeded headline table + side-track results → analysis lock + writing.

---

## Amendments (2026-07-23 grilling session, recorded in decision_log)

| # | Decision |
|---|---|
| A1 | **All deadlines dropped** (user). Gates are results-driven; §12 dates and §14 calendar void; node-day figures kept as cost estimates only. |
| A2 | **8 concurrent workers confirmed** (quota raised from the old ~6-worker observation). |
| A3 | **Judge = gpt-oss-120b for every judging role** (stage-3 audit, uniform track, judge-protocol benchmarks, future rubric arms), self-hosted, one frozen prompt per role committed to the repo; W0 adds a judge-vs-EM calibration table. |
| A4 | **3 seeds (0/1/2) a priori on all headline cells**; 1 seed for screening/side-track. Staging unchanged (seed 0 first). Satisfies the CREDIT_ASSIGNMENT_CHECKLIST ≥3-seed requirement. |
| A5 | **Project home = this repo (Agentic-RL-CA)**; ALFWorld + ScienceWorld support ported in; reward_models frozen as pilot archive (GitHub `report-figures`). **One W&B project `ca-benchmark`** (entity rainyfields), run names `<method>_<rung>_<model>_s<seed>`. |
| A6 | **Retrieval stack = existing Search-R1 stack** (e5_Flat.index + wiki-18, HDFS `…/searchr1/searchr1_data/`, `scripts/retriever_serve.sh`, conda-forge faiss). Rung-3 answerability under this corpus is the G0 step-1 check. |
| A7 | **Rung-1 re-runs fresh under the unified benchmark protocol** (Qwen3-4B-Instruct-2507, non-thinking, ReAct grammar, SFT π_base, one recipe); the July 2026 1.7B-think SearchQA campaign becomes cited pilot evidence #2 (it partially inverts the ALFWorld ranking: critic-free won, turn-critic drifted, content step-reward ≈ placebo). The §4 filter re-runs with the 4B non-thinking agent. |
| A8 | **enforce_eager=False (CUDA graphs ON) for speed** (user decision, reversing the pilot recipe). Guardrails: W0 smoke gate at 4B incl. save→resume cycle; auto-fallback to eager only on resume-from-RL-checkpoint IMA; vLLM version recorded. |
| A9 | **SFT sizing rule**: ≤3 epochs, checkpoint every ½ epoch, select the earliest checkpoint ≥90% of SFT-val plateau subject to a shared rollout-entropy floor set once at W0 from all rungs jointly. One rule, four rungs; selection curves in the W0 report. |
| A10 | **Per-wave launch manifests**: one user OK per wave covers all listed worker launches + auto-resumes; off-manifest changes re-ask. Pushes to own RainyFields remotes pre-approved; the verl-agent fork NEVER pushes to upstream langfengQ. |
| A11 | **Round-1 scope (current)**: Wave 0 inspection with ScienceWorld first, then Wave 1 = PPO gae_turn + token-GRPO anchors only. Candidate screening (StepSearch/ArCHer/SweetRL/VinePPO) deferred to a later gate decision. |

**Known loss (2026-07-23):** the SP6 verl-agent fork branch `sp6-alfworld` (Seq-MIS, NaN tripwires, SPA reward hook) was never pushed to any writable remote and is not on this node; only its documentation survives (reward_models reports + memory). Re-porting into this repo's trainer is required before ALFWorld cells run (implementation spec: reward_models `docs/vllm_fsdp_mismatch.md` + 2026-07-07/08 reports).

## Submodule index (sequential; each gets its own STATUS + dated report)

| Module | Content | Round |
|---|---|---|
| M0 | Recover repos, save overview, decision-log append | 1 (done 2026-07-23) |
| M5 | ScienceWorld install + structure audit (task map, oracle semantics, horizons, turn caps, isolation) | 1 — IN PROGRESS |
| M1 | Eval harness + adapters + dual scorers + judge serving | 1 |
| M2 | Retrieval stack redeploy + health check | 1 |
| M3 | Qwen3-4B-Instruct-2507 + zero-shot baselines + graphs-on smoke gate + recipe freeze | 1 |
| M4 | Rung-1 data re-filter @4B (stages 0–3, freeze) | 1 |
| M6 | ALFWorld port @4B (env + recipe + tripwire re-port) | 1 |
| M7 | ASearcher alignment + solvability diagnostic (go/no-go) | 1 |
| M8 | Expert data + SFT π_base ×4 rungs (uniform rule) | 1 |
| M9 | W0 report → Gate G0 → Wave-1 launch manifest | 1 |
| M10 | Wave-1 anchor runs (GRPO + gae_turn × 4 envs, seed 0) | 1 |
| M11 | W1 analysis + report → G1 (roster/screening decision with user) | 1 |
| M12+ | Wave 2 roster, Wave 3 seeds, side experiments A/B/C, Wave 4 writing | later rounds (see §7–§8) |
