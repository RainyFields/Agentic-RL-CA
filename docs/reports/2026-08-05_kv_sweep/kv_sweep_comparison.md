# KV-cache sweep — matched instrumented workload (token_grpo, 1 step)

resolved CLI overrides seen in log: gpu_memory_utilization = ['0.5', '0.60', '0.65']
runtime engine KV capacities: ['163,840', '221,440', '250,256'] tokens

| metric | 0.5 | 0.60 | 0.65 |
|---|---|---|---|
| requests (post-warm-up) | 6664 | 6869 | 6657 |
| completed trajectories | 1208 | 1224 | 1193 |
| collection span (s) | 959 | 861 | 859 |
| agg gen tok/s | 3874 | 4588 | 4210 |
| trajectories/min | 75.5 | 85.3 | 83.3 |
| eff. decode concurrency /engine | 10.9 | 13.8 | 12.7 |
| gen tokens/traj | 3076 | 3228 | 3031 |
| turns/traj | 5.52 | 5.61 | 5.58 |
| traj latency p50 (s) | 58.69 | 61.05 | 47.34 |
| traj latency p90 (s) | 193.03 | 203.75 | 178.75 |
| traj latency p99 (s) | 336.24 | 308.11 | 305.28 |
| traj latency mean (s) | 81.82 | 81.28 | 72.79 |
| TTFT p50 (s) | 0.44 | 0.26 | 0.19 |
| TTFT p90 (s) | 8.07 | 2.49 | 0.47 |
| TTFT p99 (s) | 13.35 | 5.78 | 0.78 |
| TTFT mean (s) | 2.55 | 0.72 | 0.23 |
| decode p50 (s) | 12.47 | 13.53 | 12.56 |
| decode p90 (s) | 26.92 | 32.52 | 30.54 |
| decode p99 (s) | 32.46 | 37.99 | 37.73 |
| decode mean (s) | 12.15 | 13.60 | 12.66 |
| req e2e p50 (s) | 14.17 | 14.10 | 12.84 |
| req e2e p90 (s) | 33.04 | 34.56 | 30.69 |
| req e2e p99 (s) | 38.98 | 38.80 | 37.83 |
| req e2e mean (s) | 14.72 | 14.35 | 12.92 |
| TPOT p50 (ms) | 20.7 | 20.8 | 21.4 |
| TPOT p90 (ms) | 32.0 | 36.4 | 36.4 |
| peak GPU mem (GB) | 73.9 | 73.1 | 79.1 |

## Speedups vs 0.5
| ratio | 0.60 | 0.65 |
|---|---|---|
| agg tok/s | 1.184x | 1.087x |
| traj/min | 1.129x | 1.103x |
| traj-latency p50 (lower=better) | 1.040x | 0.807x |
| traj-latency p90 | 1.056x | 0.926x |
| eff. decode concurrency | 1.267x | 1.160x |
