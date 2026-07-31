# B200 one-cycle pilot — summary (2026-07-31)

One fast-config GRPO cycle (identical knobs/seed/data to the H100 fast75 arms) on 8×B200
(useast1b/OCI). Run `p8bpilot_b200_grpo_fast_s0` (wandb offline; dir on the pod HLOG).

## Clean results (retriever-independent phases)
| phase | H100 c1 | B200 c1 | ratio |
|---|---|---|---|
| actor update | 51.4 s | 16.9 s | **3.0×** |
| old_log_prob | 14.6 s | 8.3 s | 1.8× |
| ref | 17.1 s | 8.7 s | 2.0× |
| peak mem | 77.9/80 GB | 131/192 GB | headroom for KV↑ + DYNBSZ↑ |
bf16 GEMM microbench (probe): 1,672 TFLOPS ≈ 2.1× H100 realizable.

Gen 1,342s and step 5,991s are NOT comparable: 15 retriever kills during the cycle
(HTTP-timeout retry storms account for the ~4.6 ks step-vs-phases gap). Trajectory stats
stayed in band (released 290 vs 325; success 0.186 vs 0.218).

## Integration ledger
Fixed (baked into scripts/asearcher/p8b_b200_pilot.sh + b200_probe2.sh):
1. `LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1` — image ships stale 575 compat libcuda
   over the 580.105 driver (CUDA error 803 otherwise).
2. `VLLM_ATTENTION_BACKEND=TRITON_ATTN` — vLLM's vendored flash-attn is sm_80/90-only
   (cuobjdump-verified) and segfaults in CUDA-graph capture on sm_100. Standalone
   flash-attn wheel HAS sm_100 (training side fine).
3. `WANDB_MODE=offline` — useast1b pods cannot reach wandb.ai (90s init timeout).
Open:
4. GPU faiss (conda 1.10 cuda12.9) lacks sm_100 → CPU fallback.
5. CPU-faiss retriever process repeatedly SIGKILLed under serving load even with
   `--cpu 200 --memory 1500` (training's 500GB RSS survives; killer unidentified — no
   dmesg access). BLOCKER for end-to-end training on these pods.
Also: cross-region HDFS ~25-360 MB/s (variable); model+index staging to /tmp ≈ 10 min.

## Recommendation
Compute case confirmed (2–3× on training phases, more available via memory headroom).
Do NOT schedule real runs on B200 until retrieval hosting is solved: (a) build faiss
with sm_100 cubins for GPU hosting, (b) diagnose the process killer with the queue
owners, or (c) host retrieval elsewhere reachable from useast1b. H100 arms unaffected.
