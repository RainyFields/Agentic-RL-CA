# Hardware / algorithm profiling — GRPO vs turn-PPO on 1×H100, 1×B200, (2×H100 verdict)

**Report:** `hw_algo_profiling.pdf` (built with tectonic).

## Data sources
- **H100 (both algorithms):** the finished 75-step arms' console logs
  (`outputs/p8b_arm_grpo.log`, `outputs/p8b_arm_ppo.log`) — per-phase `timing_s/*` every
  step. On replayed steps (GRPO's OOM resume) the last occurrence wins.
- **B200 (both algorithms):** `outputs/p8b_prof_b200.log` from
  `scripts/asearcher/p8b_b200_prof_worker.sh` — 8 fast-config steps per algorithm on one
  8×B200 node (alias `p8b-prof-b200`), DYNBSZ 24576, critic on-GPU, torch flat retrieval.
- **2×H100:** not runnable; evidence = `scripts/asearcher/nccl_probe.sh` output (2026-08-03)
  + the minipod finding from the platform's 资源队列使用手册 (no minipod queue in
  `mlsys_inference`, checked 2026-08-04).

## Rebuild
```bash
python mine_timings.py --out assets/timings.json \
  grpo_h100=<repo>/outputs/p8b_arm_grpo.log \
  turn_ppo_h100=<repo>/outputs/p8b_arm_ppo.log \
  b200=<repo>/outputs/p8b_prof_b200.log
python make_figures.py          # figures + assets/tables.tex + computed_tables.json
tectonic hw_algo_profiling.tex
```

## Assets (figure ↔ data pairing)
| file | content |
|---|---|
| `assets/timings.json` | raw mined per-step phase timings, all four runs |
| `assets/computed_tables.json` | every number appearing in tables/figures |
| `assets/fig_h100_fullrun.*` | 75-step phase evolution (from timings.json) |
| `assets/fig_phase_breakdown.*` | matched-step composition (from timings.json) |
| `assets/fig_speedup.*` | per-phase H100→B200 ratios (from timings.json) |
| `assets/tables.tex` | auto-generated LaTeX table bodies (`\input` by the report) |
