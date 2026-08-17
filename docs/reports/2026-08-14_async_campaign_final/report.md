# Async-rollout 8B campaign — final report

Date: 2026-08-14. Model: Qwen3-8B-Base, ASearcher Base-35k (unfiltered), wiki-18/e5
top-5 local retrieval, 32-turn / 16k-prompt / 1k-response protocol, 75 training steps.
Four campaign arms: {token GRPO, turn-PPO(b0)} × {old lockstep engine (`fast75`),
async collector (`async75kv60`)}. Plus a 5-way credit-assignment (estimator) comparison
on the old engine (§8). All transfer evals ran on the same frozen harness
(`~/xiaoxuan/arlca-8b`, old engine) over the 7,152-question 10-benchmark ASearcher
suite, judged by gpt-oss-120b with the frozen MBE prompt.

## 1. Executive summary

**The async engine delivers exactly what it was built for — ~2× wall-clock — and its
extra throughput buys proportionally more data, not proportionally more learning.**
Per *step*, async arms learn ~2× faster early (val 0.43 vs 0.23 at step 25). Per
*sample*, the sample-matched curves (fig_sample_matched) show the old-engine arms
matching or beating async at every matched trajectory count. On transfer, the
old-engine arms win everywhere: wiki-answerable macro judge 0.550 (old tPPO) / 0.535
(old GRPO) vs 0.506 (async GRPO) / 0.445 (async tPPO).

**Both async deficits are behavioral, not knowledge deficits.** Conditional on
producing an answer, async arms are as accurate or *more* accurate than their old
counterparts (judge-answered: async tPPO **0.623** — the best of all four arms; async
GRPO 0.573 vs old 0.571). What differs is *answer coverage*: async tPPO fails to emit
any `<answer>` on **30.2%** of wiki-answerable questions (old tPPO: 0.3%), and async
GRPO on 11.7% (old: 6.2%). The async engine trains policies that search longer and
commit less.

**Engine verdict**: keep the async collector for throughput (2.01× e2e, rollout phase
3–4×, zombie generation 49.6%→0), but the campaign identifies a data-distribution
interaction — most acute for the critic-based method — that must be fixed before the
async engine is the default for training runs (candidate fixes in §7).

## 2. The four campaign arms

| arm | engine | estimator | val@0 | val@25 | val@50 | val@75 |
|---|---|---|---|---|---|---|
| old GRPO | lockstep | token-level GRPO | 0.195 | 0.227 | 0.520 | 0.520 |
| old turn-PPO | lockstep | turn-level GAE, critic | 0.195 | 0.234 | 0.479 | **0.561** |
| async GRPO | async | token-level GRPO | 0.203 | 0.434 | 0.418 | 0.529 |
| async turn-PPO | async | turn-level GAE, critic | 0.199 | 0.322 | 0.486 | 0.404 (peak 0.486@50) |

Val = greedy strict EM on 512 held-out questions; reproducibility band ±0.02
(same-checkpoint re-vals: async GRPO@25 0.412→0.434 across pods; old GRPO@25
0.227→0.227 across its OOM resume). Both engines ran identical partial-rollout
config (cycle 8, max_age 4); async adds trajectory-level concurrency
(`async75kv60`: gpu_util 0.60, sticky routing, 17,408-token budget).

## 3. Engine mechanisms — what actually changed (corrected)

Measured on matched H100 profiles (same node class, config, seed, data; 3 steps + val):

| metric | sync | async | ratio |
|---|---|---|---|
| phase wall-clock (incl. val + ckpt) | 8,953 s | 4,454 s | **2.01×** |
| rollout gen, 3-step total | 4,890 s | 1,228 s | ~3–4× (see note) |
| zombie generated tokens | 49.6% | **0** | — |
| released trajectories (3 steps) | 2,300 | 2,695 | 1.17× |
| val@3 | 0.264 | 0.244 | Δ within ±0.02 band |

Steady state H100 @0.65 util: 6,473 gen tok/s, 174 traj/min, TTFT p50 15.5 s,
TPOT 36 ms. (B200: 0.88× H100 on rollout despite 2.87× KV — TRITON_ATTN decode
1.6× slower per token; H100 remains the rollout node.)

**⚠ Correction to earlier working notes: `max_age` stale-drops were ZERO in both
engines across all 75 steps of all four arms.** Stale-group censoring is NOT a
mechanism of any observed difference. The real data-side differences:

1. **Released trajectories per step**: async releases 1,200–1,700 traj/step vs the
   old engine's 690–1,150 (cumulative 92k/94k vs 55k/69k over 75 steps). The old
   engine's fixed batch geometry drops surplus completed groups; the async collector
   backfills freed slots continuously and releases everything.
2. **Completion mix**: no zombie generation and immediate slot backfill mean async
   batches contain more long/slow trajectories that the old engine would have
   truncated at cycle boundaries or never collected.
3. **Fresher resumes**: same partial-rollout snapshot format, but async resumes
   paused trajectories the moment a slot frees rather than at the next 8-turn cycle.
4. **Engine numerics**: vLLM V1 AsyncLLM token-in/token-out RPC vs the old batch
   path (byte-exact token ids verified; logprob/kernel numerics differ at fp16).

## 4. Per-step vs per-sample learning (the headline tension)

Fig `fig_train_val`: async arms reach val ~0.32–0.43 by step 25 while old arms sit
at ~0.23. Fig `fig_sample_matched` re-plots the same val points against cumulative
released trajectories: **the async advantage disappears** — old GRPO reaches 0.520
by ~38k trajectories; async GRPO needs ~92k to reach 0.529. Old turn-PPO reaches
0.561 at 69k; async turn-PPO peaks at 0.486 (61k) and *falls* to 0.404 at 94k.

Reading: the async engine's per-step advantage is a *data-rate* effect (more
trajectories per update at identical optimizer settings), not a sample-efficiency
improvement. Async GRPO consumed ~1.7× the trajectories of old GRPO to land at the
same train-val; per-sample efficiency is mildly *worse*, consistent with the larger
effective batch per update (fixed mini-batch count → fewer gradient updates per
trajectory) plus the behavioral drift in §6.

## 5. Transfer results (7,152-q ASearcher suite)

Wiki-answerable macro (7 benchmarks, 6,115 q):

| arm | judge | strict EM | sub-EM | live-web judge | no-answer |
|---|---|---|---|---|---|
| old turn-PPO | **0.550** | **0.405** | **0.461** | **0.126** | 0.3% |
| old GRPO | 0.535 | 0.395 | 0.446 | 0.097 | 6.2% |
| async GRPO | 0.506 | 0.365 | 0.416 | 0.108 | 11.7% |
| async turn-PPO | 0.445 | 0.326 | 0.371 | 0.065 | 30.2% |

The train-val ordering (async GRPO 0.529 > old GRPO 0.520) **reverses** on transfer
(0.506 < 0.535). Live-web benchmarks (GAIA/frames/xbench, 1,027 q) measure the
corpus gap, not the method — reported separately. Judge ≈ EM + 0.13 one-directional
(judge✓EM✗ ≈ 930–975 items/arm, converse ≤ 3); all three scorers agree on ranking.

## 6. Failure analysis: where the async deficits live

Decomposition (fig `fig_noanswer`, wiki-answerable):

| arm | judge all | judge answered-only | no-answer frac | mean turns | median turns |
|---|---|---|---|---|---|
| old GRPO | 0.535 | 0.571 | 6.2% | 17.5 | 18 |
| async GRPO | 0.506 | 0.573 | 11.7% | 18.7 | 19 |
| old turn-PPO | 0.551 | 0.552 | 0.3% | 7.8 | 3 |
| async turn-PPO | 0.435 | **0.623** | 30.2% | 8.5 | 3 |

- **Async turn-PPO's entire deficit is answer coverage.** When it answers, it is the
  most accurate arm in the campaign (0.623). Paired per-question vs old tPPO: 1,028
  questions lost, 318 gained — and **720 of the 1,028 losses are no-answer
  terminations**. No-answer trajectories average 17.2 turns yet **0% hit the 32-turn
  cap**: they terminate via `ctx_terminate` — the policy searches until the 16k
  context fills, never committing to an answer. No-answer concentrates on multihop
  (Musique 43%, 2Wiki 42%, Hotpot 31% vs TriviaQA/PopQA ~19%).
- **Async GRPO's paradox resolves the same way**: answered-only accuracy matches old
  GRPO exactly (0.573 vs 0.571); the transfer gap is 2× no-answer rate (11.7% vs
  6.2%) plus turn inflation (mean 18.7 vs 17.5; 20–23-turn bucket 35% vs 10%,
  fig `fig_turns_dist`). It beats old GRPO on train-val because the 512-q val set
  matches the training distribution where long searches pay off; the broader suite
  punishes the lost coverage.
- **The regression is training-time, not eval-time**: async tPPO's own val fell
  0.486→0.404 over steps 50–75 while train success kept rising — the policy drifted
  toward non-committal search during late training. The prime suspect remains the
  **critic × partial-rollout-resume interaction**: resumed trajectories carry rewards
  realized under older policies; a turn-level critic bootstraps through the resume
  seam, and with async's higher resume volume the value targets for "continue
  searching" are systematically stale-optimistic. GRPO (no critic) shows the same
  drift direction but 3× weaker. A behavior-IS/staleness correction (parked R2 item)
  and answer-commitment shaping are the candidate fixes.

## 7. Comparability caveats

1. Async-arm evals ran on A100 (old-arm evals on H100); cross-arch greedy drift is
   within ±0.02 — the deficits are 3–8× that.
2. Greedy same-checkpoint reproducibility ±0.02; judge ≈ EM + 0.13 (one-directional).
3. Async arms saved checkpoints every 5 steps (pod-reclaim replay bound), old arms
   every 25 — training data identical, replay differences are ≤5-step re-samples.
4. Train-SR dips at step ~26 in fig `fig_train_val` are pod-reclaim/OOM resume
   artifacts (replayed steps resolve to last occurrence), not learning events.

## 8. Credit-assignment sweep (old engine, 5 estimators)

Same protocol/engine (`fast75`, old lockstep + partial rollout), five estimators.
GRPO and turn-PPO completed 2026-08-02. The token-PPO / GiGPO / HCAPO arms started
2026-08-03 died with their pods at steps 42/44/46 (checkpoint grain 25), were
**resumed from step 25 on 2026-08-14** (same code, SAVE_FREQ 25→5) and **completed
2026-08-15/16**. Resume fidelity: each arm's restart re-val@25 reproduced its
original value exactly (0.312 / 0.064 / 0.297).

| estimator | mechanism | val@0 | val@25 | val@50 | val@75 | turns@75 |
|---|---|---|---|---|---|---|
| turn-PPO (b0) | turn-level GAE + critic | 0.195 | 0.234 | 0.479 | **0.561** | 7.5 |
| GiGPO | anchor-state step groups + episode groups | 0.195 | 0.064 ⚠ | 0.438 | 0.535 | 16.1 |
| HCAPO | hindsight answer-conditioned, γ=0.95 | 0.195 | 0.297 | 0.449 | 0.529 | 3.7 |
| GRPO | token-level group-relative | 0.195 | 0.227 | **0.520** | 0.520 | 17.0 |
| token-PPO | stock token GAE + critic | 0.195 | **0.312** | 0.447 | 0.426 | 8.4 |

Final readings (fig `fig_estimators`):
- **turn-PPO remains the winner (0.561)**; GiGPO (0.535) and HCAPO (0.529) edge past
  GRPO (0.520); token-PPO trails badly (0.426).
- **GiGPO's step-25 collapse (0.064) fully recovered** (0.438 → 0.535, second place).
  Its train success rate was normal (~0.45) throughout — the dip was a transient
  greedy-decode pathology (17.8 val turns at step 25, long unproductive searches),
  not estimator failure. It still searches long at 75 (16.1 turns, GRPO-like).
- **token-PPO regressed 0.447→0.426 over steps 50–75** — the only old-engine arm to
  regress late. Notably it is the token-level critic method: together with async
  turn-PPO's regression (§6), both late regressions in the campaign are critic-based
  arms, consistent with the critic-staleness suspicion (though token-PPO ran on the
  old engine, so partial-rollout resume alone — cycle 8/max_age 4 — is sufficient
  exposure).
- **HCAPO is the turn-efficiency standout**: 0.529 at just 3.7 turns/question —
  half of turn-PPO's 7.5 and ~4.5× leaner than GRPO/GiGPO — the best
  accuracy-per-turn in the sweep. Its γ=0.95 is a protocol deviation (paper value;
  all other arms γ=1.0).
- **⚠ HCAPO shows incipient per-turn length runaway.** It is the only arm that
  never compresses per-turn responses (median 270–400 tok/turn all run vs ~30 for
  GRPO/GiGPO, ~100–120 for the PPOs; p90 pinned at the 1024 cap throughout), and
  over steps 62–75 the median climbs ~280→390 with spikes to 716 (s72) and **1024 —
  the cap — at step 75**, while valid-action ratio sags 0.84→0.79 (cap truncation).
  Mechanism-consistent: γ=0.95 discounts per *turn*, so content migrates into
  fewer, longer turns; nothing in the hindsight objective penalizes within-turn
  tokens. Trajectory length is NOT running away (turns fell 4.8→3.4) and reward was
  unaffected through step 75 — but training longer would likely saturate the cap.
  Any HCAPO follow-up should add a per-token length penalty or per-turn cap margin.
  Fig `fig_resp_len` shows the divergence: four arms compress per-turn length within
  ~20 steps; HCAPO alone plateaus high and turns upward late (shaded band = HCAPO
  p10–p90; upper edge at the cap throughout).
- val@25 was weakly predictive of val@75 (rank correlation is poor: the step-25
  leader finished last; the step-25 collapse finished second).

## 9. Artifacts & reproduction

- Figures: `assets/fig_{train_val,sample_matched,transfer_4way,noanswer,turns_dist,estimators,resp_len}.{png,pdf}`
- Data: `campaign_data.json` (per-step series, all 7 runs), `items_analysis.json`
  (per-item turns/no-answer/conditional), judge summaries in
  `~/xiaoxuan/arlca-8b/outputs/judge/{agrpo,atppo,grpo,turnppo}_s75.summary.{md,json}`
- Rebuild: `python mine_campaign_data.py && python analyze_items.py && python make_figures.py`
  (venv: `/home/tiger/xiaoxuan/envs/agentic-rl-ca`)
- Checkpoints (HDFS `.../agentic_rl_ca/checkpoints/`): `{token_grpo,turn_ppo_b0}_qwen3-8b-base_32turn_{fast75,async75kv60}_s0/global_step_75`,
  `{token_ppo,gigpo,hcapo_ans}_..._fast75_s0/global_step_75`
- wandb `rainyfields/ca-rung3-8b-asearcher`: async zkdy7cw7/g2lwspq4; old
  uabse4uv+vgoq421c/odkf1zmj
- Eval dumps: HDFS `.../logs/p8b_eval_{agrpo,atppo,grpo,turnppo}_s75/rollout_eval_*.jsonl.zst`
