#!/bin/bash
# Phase 3.3 horizon eval (user OK 2026-07-21): turn-PPO (gae_turn, B0) Qwen3-4B s0 step500,
# full 51,713-row 7-task set, greedy, 4turn_think2k. Schema proven → SMOKE=0.
exec env \
  EVAL_SPECS="/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/turn_ppo_b0_qwen3-4b_4turn_think2k_s0/global_step_500|eval4b_turn_ppo_s0|turn_ppo_b0" \
  PROTOCOL=4turn_think2k JOB_TAG=eval4b_turn_ppo SMOKE=0 \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_eval.sh
