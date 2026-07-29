# P8B profiling — Qwen3-8B-Base / ASearcher 32-turn / ±partial rollout (2026-07-29/30)

**Deliverable:** `p8b_profiling_report.pdf`. Headline: partial-rollout PoC functionally
correct (T1 byte-exact, invariants hold, zero stale drops), 1.5–1.6× overall throughput
(2.2–2.5× gen); **no config meets the ~3d/arm gate at the full 150-step budget**
(13.3d GRPO-vanilla / 8.9d GRPO-partial / 20.3d turnPPO-vanilla† / 12.8d turnPPO-partial)
→ arms NOT auto-launched; decision options in report §5.
† pre-memory-fix single-step estimate; retry under KV 0.5 + critic offload in flight.

Memory finding: actor-only fits at KV 0.6 (zero headroom, 85GB peak); +8B critic OOMs at
vLLM `wake_up(kv_cache)` on cycle 2 → fixed with `GPU_MEMORY_UTIL=0.5` +
`CRITIC_{PARAM,OPTIM}_OFFLOAD=True` (protocol defaults for 8B).

RL-from-base sanity: step-0 greedy em 0.199 (unfiltered val 512), 8.2 turns, clean
grammar zero-shot; length wall at ~6 turns mean / 26 max (16k binds before the 32-turn
budget, as designed); 5k info-cap hits 3–8%; 1024 response cap hits 24–37% of turns.

## Rerun
```
# worker (one 8xH100 slot; ONLY_CFG=<tag[,tag]> to subset the 4 configs):
mlx worker launch ... -- bash <repo>/scripts/asearcher/p8b_profile_worker.sh
# analysis + figures (dev node):
python analyze_profiling.py --driver-log outputs/p8b_profile.log \
  --dump-dir /mnt/hdfs/.../logs/p8b_profile --out assets/
python make_figures.py           # env needs matplotlib (sw_inspect works)
tectonic p8b_profiling_report.tex
```

## assets/
- `timing.json` — per-config step metric records (from driver log)
- `rollouts.json` — per-config trajectory stats from the per-turn dumps
- `fig_step_breakdown.{png,pdf,json}` — gen/update/other per cycle
- `fig_throughput.{png,pdf,json}` — step-seconds per completed trajectory + days/arm
- `fig_turns_hist.{png,pdf}` — turns-per-trajectory distribution
- `fig_partial_dynamics.{png,pdf}` — released/pending/held per cycle
(raw dumps not committed — 43–128 MB each, on HDFS `logs/p8b_profile/`)

Related: `docs/design/2026-07-29_sync_partial_rollout.md` (PoC design, risks R1–R8,
tests T1–T8), wandb `rainyfields/ca-rung3-8b-asearcher` (`p8bprof_*`).
