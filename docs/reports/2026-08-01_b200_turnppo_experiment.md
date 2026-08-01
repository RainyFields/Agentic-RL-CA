# B200 turn-PPO experiment — 4 cycles, ASearcher 8B fast config (2026-08-01)

**Result: B200 runs our full ASearcher training stack end-to-end, 1.53× faster per
completed trajectory than H100 (4.2 d vs 6.4 d per 75-step turn-PPO arm), with zero
retrieval failures.** The pilot's blocker (faiss on Blackwell) is solved by replacing
faiss with a torch matmul retrieval server. Run `p8bpilot_b200_ppo_fast_s0` (wandb offline
on the pod); worker `scripts/asearcher/p8b_b200_ppo_worker.sh`.

## Configuration
turn_ppo_b0, 4 cycles, fast config (partial rollout cycle 8 / max_age 4, DYNBSZ 24,576,
chunked prefill), Qwen3-8B-Base, Base-35k, 256×5, seed 0 — identical to the H100 baseline
`p8bprof_ppo_partial` **except** the critic runs **on-GPU** (B200's 192 GB removes the
offload that H100 needs) and rollout uses Triton attention (see fixes).

## Per-cycle timings
| cycle | released | gen (s) | update actor+critic (s) | total (s) | H100 total (s) |
|---|---|---|---|---|---|
| 1 | 365 | 496 | 64 | 579 | 863 |
| 2 | 750 | 1,687 | 536 | 2,353 | 3,885 |
| 3 | 1,035 | 2,296 | 1,128 | 3,700 | 7,373 |
| 4 | 1,130 | 2,143 | 1,000 | 5,695¹ | (n/a — H100 cfg timed out) |

¹cycle 4 includes end-of-run val. Peak memory 148/192 GB (H100 baseline: 77/80 GB).

## Aggregate
| metric | H100 | B200 | ratio |
|---|---|---|---|
| step-s / completed trajectory | 5.76 | **3.76** | **1.53×** |
| gen-s / completed trajectory | 2.26 | 2.02 | 1.12× |
| update phases | — | — | **3.0–3.7×** |
| projected 75-step turn-PPO arm | 6.4 d | **4.2 d** | 1.53× |

Generation gains little: it is bound by sequential turn rounds, per-round engine swaps and
retrieval, not by matmul throughput (and Triton attention is likely slower than H100's
vendored FA2). Updates gain 3×+, doubly so because the critic no longer offloads to CPU.
Compute microbench for reference: bf16 GEMM 1,672 TFLOPS ≈ 2.1× H100.

## The retrieval fix (the pilot's blocker)
No faiss build supports sm_100 — conda-forge tops out at 1.10 (sm_90) and PyPI
`faiss-gpu-cu12` 1.14.1 ships sm_70/sm_80 only, neither with PTX to JIT. CPU faiss was
repeatedly SIGKILLed under load on these pods (unidentified killer; survived neither
default nor `--cpu 200 --memory 1500`).

Replacement: `scripts/asearcher/torch_retrieval_server.py` — same `/retrieve` HTTP
contract, same e5 encoding (`query: ` prefix, mean pooling, L2 normalise, fp16), corpus
embeddings extracted once from the flat index (`extract_flat_index.py`) and sharded across
the 8 GPUs (~3.8 GB each); scoring = inner-product matmul + merged top-k, equivalent to
faiss `useFloat16` sharded FLAT-IP. Healthy in **90 s**, **zero watchdog failures across
all 4 cycles**, and it removes the 90 GB host-RAM process entirely. This server is
Blackwell-agnostic and would work on H100 too (untested there).

## B200 pod fixes (all baked into the worker)
1. `LD_PRELOAD` the real driver libcuda — images ship a stale 575 compat lib that shadows
   the 580.105 driver (CUDA error 803). **Resolve it dynamically**: some pods have only a
   stub at `/lib/x86_64-linux-gnu/libcuda.so.1` ("file too short").
2. `VLLM_ATTENTION_BACKEND=TRITON_ATTN` — vLLM's vendored flash-attn is sm_80/90-only and
   segfaults in CUDA-graph capture on sm_100. (The standalone flash-attn wheel used for
   *training* does carry sm_100 — no change needed there.)
3. `WANDB_MODE=offline` + `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` — useast1b pods have no
   egress to wandb.ai or huggingface.co; the e5 encoder loads from the shared `/home` cache.
4. Stage model + index to pod-local `/tmp` (cross-region HDFS: 25–360 MB/s, variable);
   full staging + index extraction ≈ 25 min per fresh pod.

## Recommendation
B200 is now a usable target for this workload: 1.5× end-to-end today, and the headroom
(148/192 GB used) allows a larger KV fraction and DYNBSZ budget, which would push
generation and update further. Costs: ~25 min pod warm-up, no wandb/HF egress (offline +
post-hoc sync), and the environment quirks above — all scripted. The running H100 arms
should finish where they are; **the next campaign is the right place to switch**, and if
generation matters more than updates, pair it with the active-slot packing patch.
