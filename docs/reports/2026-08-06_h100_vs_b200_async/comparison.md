# H100 vs B200 — async rollout, identical config (0.65 / sticky / 17,408 / 4 steps)

Same checkpoint/data/seed; warm-up (first collection) excluded; KV capacity: H100 250,256 vs B200 719,216 tokens/engine (2.87x).

| metric | H100 @0.65 | B200 @0.65 | B200/H100 |
|---|---|---|---|
| measured span (s) | 1609 | 1837 | 0.88x |
| gen tok/s | 6473 | 5716 | 0.88x |
| trajectories/min | 174.2 | 152.1 | 0.87x |
| TTFT p50 (s) | 15.48 | 0.55 | 28.12x |
| TTFT p90 (s) | 29.17 | 4.87 | 5.99x |
| TTFT p99 (s) | 34.11 | 7.14 | 4.78x |
| decode p50 (s) | 11.25 | 16.26 | 0.69x |
| decode p90 (s) | 37.65 | 66.95 | 0.56x |
| decode p99 (s) | 42.83 | 77.44 | 0.55x |
| TPOT p50 (ms) | 36.07 | 56.89 | 0.63x |
| TPOT p90 (ms) | 46.90 | 73.75 | 0.64x |
| TPOT p99 (ms) | 65.44 | 133.42 | 0.49x |
| tool/env p50 (ms) | 60.56 | 55.40 | 1.09x |
| tool/env p90 (ms) | 304.96 | 483.15 | 0.63x |
| tool/env p99 (ms) | 1499.97 | 2650.35 | 0.57x |

| share of turn time | H100 @0.65 | B200 @0.65 |
|---|---|---|
| TTFT | 47.4% | 5.2% |
| decode | 51.5% | 93.4% |
| tool/env | 0.5% | 1.1% |

## Verdict
B200 is 0.88x H100 on async rollout throughput at identical config, despite 2.87x
the KV capacity. The giant KV eliminates queueing entirely (TTFT p50 15.5s -> 0.55s;
queue share 47% -> 5%) but decode is 1.6x SLOWER per token (TPOT p50 36 -> 57ms) --
the TRITON_ATTN kernels vLLM must use on sm_100 (no FlashAttention support) do not
yet exploit Blackwell, and decode is 93% of B200 turn time. GPU util 78% / 663W
confirms underutilization. B200's measured 1.5-1.6x advantage in FSDP update phases
stands, but rollout dominates step time, so H100 remains the better node for this
workload until vLLM ships sm_100-optimized attention. Peak mem: H100 79.1/79.6GB
(tight), B200 142.3/192GB (comfortable).

