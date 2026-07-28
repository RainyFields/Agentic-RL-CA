#!/bin/bash
# Full-set eval (51,713 q) of the NON-THINKING@2048 token-grpo 4B run, step 500 — gives Table 1 a
# full-set macro-EM for the thinking-mode ablation (val_2048 alone is the training-time proxy).
exec env \
  EVAL_SPECS="/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/token_grpo_qwen3-4b_4turn_nothink2k_s0/global_step_500|eval4b_nt_token_grpo_s0|token_grpo" \
  PROTOCOL=4turn_nothink2k JOB_TAG=eval4b_nt_token-grpo SMOKE=0 \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_eval.sh
