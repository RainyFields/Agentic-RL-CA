# Lean async rollout — execution model, correctness, risks (2026-08-04)

Scope per the user's lean plan: remove **turn-level rollout barriers** only. The outer
PPO/GRPO training loop stays synchronous — one policy per collection cycle, engines
sleep during FSDP updates. No async trainer, no staleness/IS correction, no
AgentLoopManager port.

## Execution model

Old (sync, `_turn_loop`):

    [full-batch generate → wait for all → full-batch env step → wait for all] × turns
    + done trajectories keep occupying batch rows ("zombie generation")
    + freed slots refill only at the 8-turn cycle boundary

New (`agent_system/multi_turn_rollout/async_rollout.py`, `+env.async_rollout_enable=true`):

    one coroutine per trajectory:  generate → search → generate → … → done/pause
    - a trajectory's search starts the moment ITS generation finishes
    - its next turn is submitted the moment ITS search returns
    - a finished trajectory exits immediately and its slot backfills immediately
      from the work queue (resumed trajectories first, then fresh prompts)
    - zero zombie generation by construction

Serving: `rollout.mode=async` → verl's vendored `AsyncvLLMServer` Ray actors (vLLM V1
`AsyncLLM` over the same colocated WorkerDict GPUs, sleep/wake preserved). We added a
token-in/token-out RPC `generate_token_ids` — the collector sends the exact prompt ids
produced by the unchanged `preprocess_single_sample` path and receives exact sampled
ids + per-token logprobs. No chat templating, no HTTP, no detokenize/retokenize drift.
Server choice is sticky per slot (`slot % dp_size`) so a trajectory's growing prefix
stays on one engine (prefix-cache friendly).

## What is preserved (and how)

| invariant | mechanism |
|---|---|
| prompt construction/truncation | `preprocess_single_sample` reused verbatim |
| response tensors (pad/mask/pos ids) | mirrors `vllm_rollout_spmd.generate_sequences` one-row |
| uid / GRPO group identity | rows buffered per uid; **only complete uid groups released** (output-assembly property, never a scheduling barrier) |
| partial rollout | same `_pr_groups` state: snapshot format, max_age staleness drop, cross-cycle resume; sync↔async collections interoperate |
| policy_version | stamped per row from `meta_info['policy_version']` |
| turn_index continuity | `turn_off` carried through snapshots (tested across the resume seam) |
| early-stop counters (MDP) | replicated per trajectory (same thresholds/reasons) |
| rollout_records JSONL | same fields, same dump path |
| weight freshness | trainer wakes engines before each collection (sharding-manager FSDP→vLLM sync); post-init sleep so checkpoint resumes never generate with stale disk weights |

Collection boundary (drain): stop admitting new turns; each in-flight turn runs to its
end-of-turn checkpoint (generation bounded by `response_length`, search by its HTTP
timeout); a hard grace timeout (`env.async_rollout_hard_grace_s`, default 300s)
cancels + aborts the request and discards the partial turn — the env has not stepped,
so the snapshot at the previous turn boundary is exact.

## Test status (CPU harness, fake server/env driving the real driver + tokenizer)

`tests/test_async_rollout.py` — ALL PASS:
1. first-turn finisher generated exactly once, slot backfilled
2. mixed-length trajectories progress independently (wall ≪ serial)
3. timestamp proof: env step of one trajectory strictly inside another's generation
   interval; next generation submitted while another trajectory is mid-generation
4. cycle-boundary snapshot at exact end-of-turn state; resume releases whole groups
   with turn_index 0..T-1 and policy versions {v1,v2} across the seam
5. uid groups whole; response token ids byte-exact vs server log; rewards and
   episode_rewards correct; attention mask counts real tokens

## Measured results — matched H100 profile (2026-08-05)

Same node class, same config/seed/data/retriever, token_grpo, 3 steps + final val:
sync baseline (phase B, worker n124-104-025) vs async collector (phase C).

| metric | sync | async | ratio |
|---|---|---|---|
| phase wall-clock (setup→exit, incl. val + ckpt) | 8,953 s | 4,454 s | **2.01×** |
| rollout gen, step 1 / 2 / 3 (s) | 610 / 1,839 / 2,441 | 232 / 505 / 491 | 2.6× / 3.6× / 5.0× |
| rollout gen, 3-step total (s) | 4,890 | 1,228 | **3.98×** † |
| trained tokens (3 steps) | 51.8 M | 56.0 M | 1.08× more data |
| released trajectories | 2,300 | 2,695 | 1.17× |
| zombie generated tokens | 9.98 M / 20.1 M = **49.6%** | **0** | — |
| final-step throughput (tok/s) | 783 | 1,844 | 2.35× |
| effective trained-tokens/s (wall) | 5.8 k | 12.6 k | **2.17×** |
| val@3 (greedy, 512 q) | 0.264 / 6.41 turns | 0.244 / 5.52 turns | Δ0.020 ≈ noise band ‡ |
| overlap (timestamps) | n/a (lockstep) | 19,406/19,409 turns env-inside-gen, 1,280 concurrent gens | — |

† per-step gen counters exclude the sync engine's two no-release remainder
collections (unmetered in step metrics but included in wall-clock), so 3.98×
slightly overstates the pure collection ratio; the 2.01× wall figure is the
conservative, unimpeachable number. True rollout-phase speedup ≈ 3–4×.
‡ same-checkpoint greedy re-evals on this setup historically vary ±0.02; 3 steps
of divergent sampling also contribute. GRPO group semantics verified (atomic
release, groups whole in every batch).

Turn-PPO (phase D, TWO independent workers — full replication):

| run | gen s1 / s2 | critic upd s1 / s2 | val@2 |
|---|---|---|---|
| worker 1 | 238.6 / 483.0 | 50.5 / 381.3 | 0.205 |
| worker 2 | 239.5 / 517.5 | 42.9 / 430.0 | 0.207 |

Gen within 0.4% / 7% across runs, val within 0.002 — the async engine is
deterministic-stable at run granularity. Turn-PPO async gen ≈ the GRPO async legs
(232 / 505), vs sync turn-PPO's 2,057 s matched-step average.

## B200 async leg (2026-08-06, worker 1032626, instrumented run @ gpu_util 0.65)

Engine smoke on sm_100 (TRITON_ATTN + V1 AsyncLLM + sleep mode): **all 5 PASS** —
the async engine is validated on both chips. token_grpo, 4 steps + val:

| step | B200 gen (s) | B200 step (s) | H100 async gen (s) | H100 async step (s) |
|---|---|---|---|---|
| 1 | 270.5 | 332.8 | 231.7 | 326.9 |
| 2 | 439.7 | 793.7 | 505.0 | 1,068.0 |
| 3 | 455.5 | 1,492.0 | 491.2 | ~2,030 (ex-val) |
| 4 | 382.7 | 2,100.4 (incl. val) | — | — |

val@4 = 0.252 / 5.3 turns (same band as H100 async 0.244 and sync 0.264).

**Measured B200-over-H100 (async, GRPO): ~1.24–1.36× per step** (steps 2–3;
step 1 is a tie — decode-heavy early cycles hit the TRITON_ATTN penalty, gen
0.86×; mature prefill-heavy cycles run 1.08–1.15×). Matches the ~1.3× GRPO
estimate; turn-PPO (more update weight) projects ~1.4×. CONFOUND: the B200 run
used gpu_util 0.65 vs 0.5 on the H100 async leg (part of the parallel session's
KV-pressure ablation) — some of the B200 edge may be KV headroom, not silicon;
the parallel session's H100@0.65 leg is the exact comparator. B200 turn-PPO
async (phase D) not run — superseded by the instrumented-diagnosis design.

Async engine also released groups on EVERY collection (no no-release cycles),
vs sync's 2 empty cycles in 5 — immediate backfill keeps release pressure up.

**Launch criteria (user's five): all pass.** Smoke+tests ✓, GRPO semantics ✓
(val in noise band, groups atomic), no finished-trajectory regeneration ✓
(zombie = 0 by construction, verified), measured speedup ✓ (2.01× e2e, ~3–4×
rollout), no hangs/leaks/failed resumes across three clean runs (incl. a
cross-phase checkpoint auto-resume and 410 cross-cycle trajectory resumes) ✓.
**Recommendation: restart the campaign (GRPO + turn-PPO, H100) on the async
engine.** Residual risk: longest async soak so far is ~75 min — the 75-step arms
are the first long-duration run; mitigated by 3-attempt auto-resume + monitors.

## Risks / open items

- `mode: async` first exercised in this fork by our smoke test (vLLM 0.11.0 V1
  AsyncLLM + sleep mode + ExternalRayDistributedExecutor). Engine smoke gates the rest.
- Engine weights load from disk at init (`load_format=auto`) vs sync's dummy+sync;
  mitigated by post-init `sleep()` forcing a real weight sync on first wake.
- Per-request RPC overhead (Ray call per turn) vs sync's one batched call per turn:
  measured in the matched profile; expected small vs multi-second turn times.
- GiGPO: anchor-state step-grouping is uid-group-local and released groups are whole →
  compatible in principle; NOT validated (excluded from the restart campaign anyway).
- Failure handling is lean: a crashed trajectory restarts from scratch next
  collection (logged loudly); its partial rows are discarded.
