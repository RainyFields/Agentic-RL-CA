#!/bin/bash
# Full-set eval (51,713 q) of the NON-THINKING@2048 turn-ppo 4B run, step 500 — gives Table 1 a
# full-set macro-EM for the thinking-mode ablation (val_2048 alone is the training-time proxy).
exec env \
  EVAL_SPECS="/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/turn_ppo_b0_qwen3-4b_4turn_nothink2k_s0/global_step_500|eval4b_nt_turn_ppo_b0_s0|turn_ppo_b0" \
  PROTOCOL=4turn_nothink2k JOB_TAG=eval4b_nt_turn-ppo SMOKE=0 \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_eval.sh
