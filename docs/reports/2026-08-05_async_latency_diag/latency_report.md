# Async rollout latency analysis

events: 5737 requests total; 2677 excluded as warm-up (first 30s); collection span 261.5s

## Latency components (s)

| component | n | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| transport | 3060 | 0.002 | 0.002 | 0.003 | 0.003 | 0.004 | 0.370 |
| ttft_total | 3060 | 9.161 | 8.465 | 17.119 | 18.682 | 20.575 | 23.852 |
| ttft_service | 3060 | 9.158 | 8.463 | 17.116 | 18.680 | 20.573 | 23.850 |
| decode | 3060 | 15.622 | 14.601 | 29.581 | 31.055 | 33.109 | 34.929 |
| tpot | 3060 | 0.029 | 0.028 | 0.034 | 0.037 | 0.053 | 0.429 |
| e2e | 3060 | 24.797 | 23.919 | 41.160 | 43.438 | 46.635 | 49.171 |
| env_wait | 3060 | 0.035 | 0.009 | 0.086 | 0.105 | 0.158 | 1.060 |
| turn | 3060 | 24.832 | 23.955 | 41.185 | 43.474 | 46.712 | 49.249 |

generated tokens: 1763111 over 231.5s -> **7615 tok/s aggregate** (952 tok/s/engine)

## Trajectory completion latency (s)

| component | n | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| trajectory | 733 | 103.759 | 109.879 | 182.604 | 196.339 | 218.184 | 228.366 |

## Where trajectory time goes (sum over all turns)

- transport+queue+prefill (ttft_total): 28032s (36.9%)
- decode: 47803s (62.9%)
- tool/env wait: 108s (0.1%)

## By context length

| component | n | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 0-2k (e2e) | 466 | 22.769 | 20.887 | 38.539 | 39.390 | 42.229 | 43.884 |
| 0-2k (ttft_svc) | 466 | 5.699 | 5.744 | 10.290 | 12.022 | 13.535 | 18.394 |
| 2-4k (e2e) | 1265 | 24.903 | 23.770 | 41.353 | 43.067 | 45.925 | 49.110 |
| 2-4k (ttft_svc) | 1265 | 9.727 | 9.683 | 16.794 | 17.852 | 20.047 | 23.786 |
| 4-8k (e2e) | 1143 | 25.992 | 25.451 | 41.890 | 44.401 | 47.365 | 49.171 |
| 4-8k (ttft_svc) | 1143 | 10.069 | 9.930 | 18.197 | 19.661 | 21.856 | 23.850 |
| 8-12k (e2e) | 160 | 21.624 | 18.110 | 40.562 | 44.609 | 46.790 | 48.248 |
| 8-12k (ttft_svc) | 160 | 8.367 | 7.813 | 17.390 | 19.430 | 20.559 | 21.088 |
| 12k+ (e2e) | 26 | 22.932 | 20.275 | 43.318 | 44.706 | 45.454 | 45.605 |
| 12k+ (ttft_svc) | 26 | 8.299 | 9.333 | 17.443 | 20.095 | 20.589 | 20.619 |

## By generated length

| component | n | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 0-128 (e2e) | 637 | 10.904 | 10.334 | 18.432 | 20.216 | 23.081 | 25.353 |
| 0-128 (ttft_svc) | 637 | 9.329 | 8.735 | 17.273 | 18.822 | 20.356 | 23.580 |
| 128-256 (e2e) | 419 | 14.511 | 14.027 | 21.502 | 23.882 | 26.808 | 28.972 |
| 128-256 (ttft_svc) | 419 | 8.984 | 7.976 | 16.640 | 18.372 | 21.078 | 23.159 |
| 256-512 (e2e) | 436 | 19.663 | 19.446 | 26.909 | 29.314 | 32.138 | 33.952 |
| 256-512 (ttft_svc) | 436 | 9.043 | 8.205 | 17.054 | 18.722 | 20.359 | 22.538 |
| 512+ (e2e) | 1568 | 34.617 | 35.968 | 43.363 | 44.973 | 47.373 | 49.171 |
| 512+ (ttft_svc) | 1568 | 9.168 | 8.619 | 17.163 | 18.589 | 20.591 | 23.850 |

## By in-flight requests on engine at enqueue

| component | n | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 0-40 (e2e) | 364 | 18.253 | 17.653 | 32.287 | 33.571 | 34.808 | 36.966 |
| 0-40 (ttft_svc) | 364 | 3.714 | 2.752 | 8.808 | 9.402 | 11.504 | 12.304 |
| 40-80 (e2e) | 1503 | 27.401 | 28.896 | 42.542 | 44.726 | 47.287 | 49.171 |
| 40-80 (ttft_svc) | 1503 | 11.372 | 10.929 | 18.445 | 19.857 | 21.749 | 23.850 |
| 80-120 (e2e) | 1193 | 23.512 | 21.575 | 39.676 | 42.121 | 45.218 | 48.248 |
| 80-120 (ttft_svc) | 1193 | 8.031 | 6.796 | 14.103 | 17.147 | 18.780 | 19.971 |
| 120+ (e2e) | 0 | - | - | - | - | - | - |
| 120+ (ttft_svc) | 0 | - | - | - | - | - | - |

## By collection phase

| component | n | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| early (e2e) | 1960 | 23.978 | 22.251 | 39.840 | 41.812 | 45.198 | 48.933 |
| early (ttft_svc) | 1960 | 8.377 | 7.509 | 13.674 | 16.650 | 19.511 | 21.738 |
| middle (e2e) | 886 | 28.044 | 28.778 | 43.937 | 45.588 | 47.464 | 49.171 |
| middle (ttft_svc) | 886 | 11.836 | 11.662 | 19.315 | 20.132 | 22.771 | 23.850 |
| late (e2e) | 214 | 18.846 | 16.945 | 36.466 | 39.089 | 43.586 | 44.761 |
| late (ttft_svc) | 214 | 5.228 | 2.127 | 15.169 | 16.340 | 20.412 | 20.859 |

## Per-engine (e2e s / tok/s share)

| component | n | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| engine 0 | 344 | 20.294 | 17.718 | 36.465 | 37.041 | 38.101 | 38.357 |
| engine 1 | 369 | 23.280 | 20.626 | 38.641 | 40.811 | 42.263 | 42.826 |
| engine 2 | 335 | 20.363 | 18.816 | 34.427 | 35.888 | 37.590 | 38.542 |
| engine 3 | 422 | 25.611 | 23.533 | 43.024 | 44.593 | 46.571 | 47.741 |
| engine 4 | 418 | 29.059 | 29.129 | 44.506 | 45.960 | 48.412 | 49.171 |
| engine 5 | 463 | 28.623 | 27.754 | 44.083 | 45.327 | 47.373 | 48.248 |
| engine 6 | 364 | 24.584 | 23.746 | 39.831 | 40.723 | 41.640 | 42.807 |
| engine 7 | 345 | 24.141 | 22.441 | 39.049 | 39.629 | 41.850 | 42.549 |

runtime KV capacity lines: NOT FOUND (analytical ~160k/engine)

## Diagnosis (per the 5-way decision logic)

Primary bottleneck: **(a) vLLM queueing / KV-cache pressure**, with decode length as the
intrinsic secondary driver. Evidence:

1. Effective concurrent decode ≈ 27 sequences/engine (952 tok/s/engine × 0.028s TPOT),
   vs ~100+ requests submitted per engine — ~75% of submitted requests are queued
   inside the engine at any time. Matches the analytical KV budget: ~160k tokens/engine
   ÷ ~3–5k resident tokens/sequence ≈ 30–50 concurrently admittable sequences.
2. ttft_service (in-engine queue + prefill) p50 = 8.5s, p90 = 17.1s — but it is FLAT in
   context length (5.7s at 0–2k vs 8.4s at 8–12k) and strongly increasing in in-flight
   load (p50 2.8s at <40 in-flight → 10.9s at 40–80). Queueing, not prefill compute.
3. NOT tool/env underfeeding: env_wait p50 = 9ms, 0.1% of trajectory time.
4. NOT max_num_batched_tokens: per-iteration demand ≈ 27 decode tokens + one
   prefix-cached prefill tail (~1–2k) ≪ 17,408. Budget never approached
   (per-iteration scheduled-token counter not exposed; inference from demand, stated).
5. NOT max_num_seqs (1024 ≫ 160 assigned/engine).
6. Engine imbalance is real but secondary: mean e2e 20.3–29.1s across engines (1.43×),
   tracking request-count skew (335–463 reqs/engine) from sticky slot routing.
7. Median latency composition: decode 62.9% of turn time (mean gen len 495 tokens —
   intrinsic work), queue+prefill 36.9%, env 0.1%. Tail (p99 ≈ 47s) = long decode
   (512+ bucket) stacked on queue delay.

Implication (H100): raising gpu_memory_utilization (0.5 → ~0.65) is the first lever —
it directly raises admittable concurrency, cutting the 37% queue share; engine-load
rebalancing (least-loaded instead of sticky-only routing, keeping prefix locality as a
tiebreak) is second. No B200 extrapolation here per instruction.

Gaps: engine-side Running/Waiting/KV%/preemption counters unavailable (V1 stat logger
does not propagate from the Ray actor even with disable_log_stats=False); runtime KV
capacity line not emitted. Analytical capacity labeled as estimate throughout.
