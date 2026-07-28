#!/bin/bash
# Phase 3.3 horizon eval (user OK 2026-07-21): GiGPO Qwen3-4B s0 step500, then the pre-RL Qwen3-4B
# floor row. Full 51,713-row 7-task set, greedy, 4turn_think2k. §4 per-episode dump. Sequential on one worker.
exec env \
  EVAL_SPECS="/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/gigpo_qwen3-4b_4turn_think2k_s0/global_step_500|eval4b_gigpo_s0|gigpo;/mnt/hdfs/mlsys/models/Qwen3-4B|eval4b_floor|token_grpo" \
  PROTOCOL=4turn_think2k JOB_TAG=eval4b_gigpo_floor \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_eval.sh
