# 8B ASearcher arms — token-GRPO vs turn-PPO (75 steps, fast config), 2026-08-02

**Deliverable:** `p8b_arms_results.pdf`. Headline: **turn-PPO 0.561 vs GRPO 0.520** val EM
at step 75, turn-PPO using **6.5 vs 16.1 searches/question**. GRPO plateaued after step 50
and re-inflated turns (14.6→17.0); turn-PPO kept improving at ~7.5 turns. Both ~3× the
zero-shot 0.195. **Inverts the 4B filtered result** (GRPO 0.773 vs 0.496) — confounds:
unfiltered data (~40% of successes need no search), 32-turn vs 8-turn horizon, 8B vs 4B.

Single seed per arm; 512-question val (~1 SE ≈ 4 points); GRPO absorbed a clean
OOM-checkpoint restart (~10 cycles replayed at DYNBSZ 20480 instead of 24576).

wandb: `rainyfields/ca-rung3-8b-asearcher`, runs `*_qwen3-8b-base_32turn_fast75_s0`.

## Rerun
```
python make_figures.py     # needs matplotlib (envs/sw_inspect works); data inlined from val logs
tectonic p8b_arms_results.tex
```

## assets/
- `arms_val_series.json` — the val@0/25/50/75 series for both arms (EM, turns, searches)
- `fig_arms_curves.{png,pdf}` — EM and turns vs step
- `fig_em_vs_search.{png,pdf}` — EM vs search volume (efficiency view, points labelled by step)

Related: `docs/reports/2026-07-30_p8b_fast_profiling/` (config the arms ran),
`docs/design/2026-07-29_sync_partial_rollout.md`, `docs/reports/2026-08-01_b200_turnppo_experiment.md`.
