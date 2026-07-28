#!/bin/bash
# Phase 3.3 horizon eval (user OK 2026-07-21): token-GRPO Qwen3-4B s0 step500,
# full 51,713-row 7-task set, greedy, 4turn_think2k (the protocol it trained under). §4 per-episode dump.
exec env \
  EVAL_SPECS="/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/token_grpo_qwen3-4b_4turn_think2k_s0/global_step_500|eval4b_token_grpo_s0|token_grpo" \
  PROTOCOL=4turn_think2k JOB_TAG=eval4b_grpo \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_eval.sh
