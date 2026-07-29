# Long-horizon RL config audit — verl/verl-agent stack (2026-07-30)

Scope: every configuration in our stack that materially affects long-horizon (many-turn)
RL wall-clock or training behavior; current value for the 8B ASearcher protocol
(`protocol_asearcher_32turn_8b.sh` + `run_condition.sh` + cond files + code defaults),
where it is set, expected impact at long horizon, and whether changing it alters the
**training distribution/gradients (VALIDITY)** or **only throughput (SPEED)**.
Numbers cited are from the P8B profiling report (`docs/reports/2026-07-29_p8b_profiling`).

## 1. Rollout-loop settings

| Setting | Current | Where | Long-horizon impact | Class |
|---|---|---|---|---|
| Trainer class | `RayPPOTrainer`, fully synchronous | `verl/trainer/ppo/ray_trainer.py` (fork 0.3.1.dev) | Step time = slowest trajectory; no async pipelining. `AsyncLLMServerManager` exists but is stubbed out (commented in `fit()`) | SPEED (arch) |
| Partial rollout | OFF by default; PoC via `+env.partial_rollout_enable=true` | `rollout_loop.py::partial_multi_turn_loop` | Caps sequential gen rounds per cycle; measured 1.5–1.6× overall, 2.2–2.5× gen | **VALIDITY** (staleness; see §4) |
| `partial_rollout_cycle_turns` | 8 | hydra `+env.…` | Lower = fuller gen batches + more staleness; must be ≥ typical turns (mean ~6) to keep release rate high | VALIDITY (mild) |
| `partial_rollout_max_age` | 4 cycles | hydra `+env.…` | Staleness bound; 0 drops observed at 8-turn cycles | VALIDITY |
| Max turns (`env.max_steps`) | 32 nominal | protocol `MAX_STEPS` | Never binds in practice (max observed 26); enforced env-side, survives snapshot/resume | VALIDITY |
| Per-turn response cap | 1024 tokens | `MAX_RESPONSE_LENGTH` | 24–37 % of turns hit it (base-model thinks); decode-time driver | VALIDITY |
| Prompt/context budget | 16,384 | `MAX_PROMPT_LENGTH` | The *actual* horizon limit: length wall at ~6–12 turns; prompts mean 4.6k | VALIDITY |
| History window | `HISTORY_LENGTH=32` (= everything) + ASearcher mode (full think+search responses kept) | protocol + `env.asearcher_history` | Context grows ~1.3–2.3k tokens/turn; drives prefill cost quadratically-ish with turn depth | VALIDITY |
| Overflow rule | clean terminate (`env.ctx_terminate`, margin 256) + `TRUNCATION=left` as never-fire safety valve | protocol / env manager | Episode ends before model ever sees a truncated prompt; residual 0.5 % clip from char-prefilter tail (tighten before arms) | VALIDITY |
| Finished-trajectory handling in gen batch | **NOT removed** — all 1280 slots re-generate every round until the whole batch finishes | `rollout_loop.py::_turn_loop` (no active-mask packing) | Vanilla pays ~26 rounds at full batch for mean-6-turn work; dead slots emit ~1-token junk but still cost prefill + engine swap | SPEED (pure waste) |
| vLLM prefix caching | ON (`enable_prefix_caching=True` hardcoded) **but defeated by the prompt template** — "you have already taken {N} step(s)" precedes the history, so the cacheable prefix is only the ~200-token header+question out of ~4.6k mean | `vllm_rollout_spmd.py:201`; template in `prompts/search.py` | Every turn re-prefills nearly the whole grown prompt. Template reorder (step-count after history, or dropped) → per-turn delta-prefill | SPEED via template change that is technically VALIDITY (prompt text changes) |
| Chunked prefill | **disabled** (`enable_chunked_prefill=False`), `max_num_batched_tokens` at default 8192 | `run_condition.sh:107` | With 16k prompts, long prefills serialize; enabling + raising the token budget is a standard win | SPEED |
| CUDA graphs | on (`enforce_eager=False`) | run_condition | good as-is | SPEED |
| GPU split rollout vs train | colocated, time-shared; vLLM KV fraction `GPU_MEMORY_UTIL` = 0.5 (8B; was 0.6) | protocol/run_condition | Every gen call crosses the FSDP↔vLLM sharding manager (weight sync + KV wake/sleep) — a per-round fixed cost, ×26 rounds vanilla; `free_cache_engine=False` | SPEED |
| Sampling | temp 1.0 / top-p 1 train; greedy val | yaml defaults; val_kwargs | — | VALIDITY |

## 2. Batching & padding

| Setting | Current | Where | Impact | Class |
|---|---|---|---|---|
| `use_dynamic_bsz` (+ `*_max_token_len_per_gpu`) | **False** (knob added: `DYNBSZ=1` → 24,576 tok/GPU for actor/logprob/ref/critic) | yaml defaults; protocol knob | Fixed micro=1 feeds ~5.2k real tokens per 8B forward (~1/5 utilization). Token-budget packing is the single biggest untested update-side win (rung-4 A14 precedent) | SPEED |
| Micro batch sizes | `MICRO_BSZ=1` (actor+critic), `LOGPROB_MICRO=2` | protocol | OOM-safe but underutilized; superseded by dynamic bsz if enabled | SPEED |
| `use_remove_padding` | True (actor+critic) | run_condition | Already strips pad tokens in training forwards (flash-attn varlen); without it the 17.4k pad width would be catastrophic | SPEED (keep) |
| `balance_batch` | True | yaml `trainer.balance_batch` | Token-balanced DP ranks (observed balanced_min≈max). Keep | SPEED (keep) |
| Train batch / group | 256 prompts × group 5 = 1280 traj/step | protocol | Row count/step ≈ 7.6k turn-rows at 8B. Halving batch halves step time but **halves the optimization budget** | VALIDITY |
| `ppo_mini_batch_size` | 512 turn-rows → ~15 optimizer minibatches per step | protocol | More minibatches per step at long horizon (rows/step grows with turns) — effective updates per collected token constant | VALIDITY |
| `ppo_epochs` | 1, `shuffle=False` | yaml default | — | VALIDITY |

## 3. Trainer/optimizer memory & sequence-length scaling

| Setting | Current | Where | Impact | Class |
|---|---|---|---|---|
| Gradient checkpointing | True (actor+critic) | run_condition | Mandatory at 17.4k seq; already on | SPEED (keep) |
| Ulysses sequence parallel | 1 (off) | yaml default | Option if context grows past ~32k (not needed at 16k+1k with rmpad+ckpt) | SPEED |
| Actor FSDP offload | param False / optim False | run_condition | Peak 85 GB allocated (expandable segments) — zero headroom at KV 0.6; OK at 0.5 | SPEED/memory |
| Ref policy | param_offload True | run_condition | fine | — |
| Critic (gae_turn only) | `CRITIC_PARAM_OFFLOAD=True`, `CRITIC_OPTIM_OFFLOAD=True` (8B); lr 1e-5; same micro/ckpt as actor | protocol knobs (post-OOM fix) | Without offload: OOM at vLLM `wake_up(kv_cache)` cycle 2. Offload costs update time (measured in ppo_partial: update 2,009s vs GRPO-partial 1,262s) | SPEED |
| `max_num_batched_tokens` | 8192 (default) | yaml | Pairs with chunked prefill above | SPEED |

## 4. Algorithm-level (horizon-sensitive)

| Setting | Current | Where | Impact | Class |
|---|---|---|---|---|
| Advantage estimator | `grpo` (episode reward, group-normalized over uid, broadcast to turns) vs `gae_turn` (turn-level GAE) per cond file | `cond_token_grpo.sh` / `cond_turn_ppo_b0.sh` | The experiment variable itself | VALIDITY (by design) |
| γ / λ | 1.0 / 1.0 | protocol | No discounting across turns: credit spreads undecayed over up to 26 turns | VALIDITY |
| Loss aggregation | `loss_agg_mode="token-mean"` | yaml default (not overridden) | Long turns get proportionally more gradient weight (tracked: `lengthdiag/corr_len_effweight≈0.68` at 4B). `seq-mean-token-mean` would equalize turns — different gradients | VALIDITY |
| PPO clipping | clip 0.2/0.2, dual-clip c=3.0 | yaml defaults | Only staleness guard currently active for resumed segments | VALIDITY |
| KL regularization | GRPO arm: `use_kl_loss=True`, coef 1e-3, `low_var_kl`. **turnPPO arm: no KL at all** (`use_kl_in_reward=False`, `use_kl_loss` default False) | cond files / run_condition | At long horizon + partial rollout, turnPPO relies purely on clipping for trust region — flag, but changing it mid-benchmark breaks comparability | VALIDITY |
| Entropy bonus | `entropy_coeff=0` | run_condition | — | VALIDITY |
| Invalid-action penalty | True, coef 0.01 | run_condition | Reward shaping per malformed turn; more turns ⇒ more shaping opportunities | VALIDITY |
| Off-policy correction for resumed trajectories | **None**: `old_log_probs` recomputed under current policy for ALL rows (ratio=1 for stale segments); per-row `policy_version` recorded; behavior logprobs available via `rollout.calculate_log_probs` (not enabled) | `ray_trainer.py` recompute block; design doc R2 | The partial-rollout validity caveat. Upgrade path: substitute stored vLLM behavior logprobs as `old_log_probs` for rows with `policy_version < current` (behavior-IS PPO), or AReaL-style decoupled loss | VALIDITY |
| Trajectory-atomicity for GRPO groups | uid-group-atomic release (PoC) | `partial_multi_turn_loop` | Prevents split-group baseline bias; adds holding staleness | VALIDITY (protective) |

## Top speedups that do NOT change the training distribution

1. **Dynamic token-budget micro-batching** (`DYNBSZ=1`, 24.5k tok/GPU): gradient math is
   identical (same minibatch partition, packing only changes forward grouping). Update +
   logprob + ref ≈ 45–60 % of step time; expect 2–4× on those phases. *The one to profile
   first.*
2. **Active-slot packing in `_turn_loop`**: stop generating for finished trajectories
   (they are already excluded from training — this is pure dead compute). Benefits vanilla
   most (up to ~3–4× on tail rounds) and partial somewhat. Moderate patch, byte-identical
   training data.
3. **Chunked prefill + `max_num_batched_tokens` ≥ 16k**: config-only vLLM throughput win
   at our prompt lengths.
4. **Prefix-cache-friendly template** (move "{step_count}" below the history or drop it):
   turns re-prefill into delta-prefill. *Caveat: changes the literal prompt → new MDP;
   do it only at a protocol boundary, not mid-comparison.*
5. **Partial rollout** (already built): 1.5–1.6× measured — but it is a semantics change
   (staleness), so it belongs in the VALIDITY column, not here; use per explicit decision.

Items 1–3 compose multiplicatively with partial rollout since they attack different
phases. Rough stack: GRPO-partial 8.9 d/arm → with (1)+(3) plausibly ~4–6 d/arm; vanilla
13.3 d → with (1)+(2)+(3) plausibly ~5–7 d (to be measured, not promised).
