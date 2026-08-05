# Latency components by gpu_memory_utilization (matched workload, warm-up excluded)

| component | 0.5 | 0.60 | 0.65 |
|---|---|---|---|
| TTFT p50 (s) | 0.44 | 0.26 | 0.19 |
| TTFT p90 (s) | 8.07 | 2.49 | 0.47 |
| TTFT p99 (s) | 13.35 | 5.78 | 0.78 |
| TTFT mean (s) | 2.55 | 0.72 | 0.23 |
| decode p50 (s) | 12.47 | 13.53 | 12.56 |
| decode p90 (s) | 26.92 | 32.52 | 30.54 |
| decode p99 (s) | 32.46 | 37.99 | 37.73 |
| decode mean (s) | 12.15 | 13.60 | 12.66 |
| tool/env wait p50 (ms) | 60 | 60 | 61 |
| tool/env wait p90 (ms) | 142 | 149 | 154 |
| tool/env wait p99 (ms) | 487 | 1236 | 922 |
| tool/env wait mean (ms) | 76 | 95 | 89 |

## Share of total turn time

| | 0.5 | 0.60 | 0.65 |
|---|---|---|---|
| TTFT (in-engine queue + prefill) | 17.2% | 5.0% | 1.8% |
| decode | 82.1% | 94.1% | 97.3% |
| tool/env | 0.5% | 0.7% | 0.7% |

Reading: KV growth buys down the queue share (17.2 -> 5.0 -> 1.8% of turn time);
decode itself does not speed up (p50 ~12.5-13.5s at every setting; more sequences
share each decode step at higher admission). Tool/env stays at noise level
(~0.5-0.7%, median ~60ms) at every setting. Throughput saturates at 0.60 because
by then the engine is already >94% pure decode.
