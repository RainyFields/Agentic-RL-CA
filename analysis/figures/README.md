# analysis/figures

## fig_truncation_by_arm.png / .pdf

Final truncation clip ratio per arm, **4-turn 1.7B runs only** (8-turn stress runs and the
in-flight Qwen3-4B scale-up excluded), sorted ascending. Bar = mean over seeds; open
circles = individual seeds; colors = arm family. Metric = `response_length/clip_ratio` at
the final logged step (=500): fraction of turn-level samples whose generation hit
`MAX_RESPONSE_LENGTH=2048` exactly (`verl/trainer/ppo/metric_utils.py:162`).

- Backing data: `fig_truncation_by_arm.csv` (regenerated on every script run).
- Source: W&B `rainyfields/agentic-rl-ca`, all non-toy `*qwen3-1.7b_4turn_think2k*` runs;
  crash-resume run-ID fragments merged per experiment name, value taken at max `_step`.
  Pulled 2026-07-20 (SF time).
- Rerun: `~/xiaoxuan/envs/verl-agent/bin/python fig_truncation_by_arm.py` (values are
  inlined in the script; re-pull from W&B only if runs change).
- Note: includes ALL seeds — b1 s1 (0.188, collapsed) and HCAPO s0 (0.754, collapsed) are
  shown, since truncation is the phenomenon plotted; the "healthy runs only" EM table in
  `analysis/wave1_results_README.md` is a different filter.
