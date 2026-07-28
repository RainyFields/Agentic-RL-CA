#!/bin/bash
# Phase 3.3 horizon eval (user OK 2026-07-21): token-PPO (gae) Qwen3-4B s0 step500, full 51,713-row
# set, greedy, 4turn_think2k. SMOKE=0; eval forces a non-critic estimator (see eval_search_full.sh).
exec env \
  EVAL_SPECS="/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/token_ppo_qwen3-4b_4turn_think2k_s0/global_step_500|eval4b_token_ppo_s0|token_ppo" \
  PROTOCOL=4turn_think2k JOB_TAG=eval4b_token_ppo SMOKE=0 \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_eval.sh
