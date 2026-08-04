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

## Measured results

(to be filled from the matched H100 profile — phases B/C of
`scripts/asearcher/p8b_async_profile_worker.sh`)

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
