# Cross-environment credit-assignment comparison (2026-08-14)

Five credit-assignment methods — token-PPO, turn-PPO, GRPO, GiGPO, HCAPO — compared on
SearchQA (Qwen3-4B, 4-turn non-thinking, 500 steps, full 51,713-q eval) and ALFWorld
(Qwen3-1.7B from replay-BC, unseen 134-game eval). One figure per environment: training
curves + final-eval table.

## Status

DRAFT. Two SearchQA cells (token-PPO, GiGPO) were launched 2026-08-14 as batched Merlin
jobs on the ark H100 queue (resubmit loop in `~/xiaoxuan/ca_cmp_jobs/`, checkpoints to
HDFS every 25 steps, resume across the queue's 4-hour pod reclamations). Rerun
`build_pdf.sh` after they finish (DONE markers
`.../agentic_rl_ca/logs/train_{token_ppo,gigpo}_s0_4turn_nothink2k/DONE`) and after their
full-set evals land in `outputs/eval_full/eval4b_nt_{token_ppo,gigpo}_s0/`.

## Rebuild

```
bash build_pdf.sh   # collect_results.py -> build_figures.py -> tectonic
```

## Data provenance (results/provenance.json for exact paths)

- SearchQA curves: wandb `rainyfields/agentic-rl-ca`, runs `*_qwen3-4b_4turn_nothink2k_s0`
  (`val-core/macro_em`). New arms log to the same project with pinned run ids
  `cacmp_{token_ppo,gigpo}_4bnt_s0`.
- SearchQA eval: `~/xiaoxuan/Agentic-RL-CA/outputs/eval_full/eval4b_*/paper_table.csv`
  (canonical `macro` row).
- ALFWorld curves: `~/xiaoxuan/reward_models/docs/reports/`
  `2026-07-13_credit_assignment_comparison_assets/history_*.csv` (+ HCAPO-paper local parse).
- ALFWorld eval: `unseen_results.csv` there; ledger
  `/mnt/hdfs/.../alfworld_prm_gigpo/spa_data/unseen_cmp_results.txt`.

## Method-name mapping

| report name | SearchQA run | ALFWorld run |
|---|---|---|
| token-PPO | `token_ppo_*` (gae) | `cmp_ppo_gae_token_bc` |
| turn-PPO | `turn_ppo_b0_*` (gae_turn) | `cmp_ppo_gae_turn_bc` |
| GRPO | `token_grpo_*` | `grpo_bc` (eval @265) |
| GiGPO | `gigpo_*` | `gigpo_bc` (eval @200) |
| HCAPO | `hcapo_paper_*` | `cmp_hcapo_paper_bc` (v2) |
| HCAPO (adapted) | `hcapo_ans_*` | `cmp_hcapo_bc` (v1) |
