#!/bin/bash
# Eval the finished turn-PPO arm (val 0.561 @ step 75) over the ASearcher eval suite.
# COND stays turn_ppo_b0 only so the config validates; val_only never touches the critic.
export CKPT=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/turn_ppo_b0_qwen3-8b-base_32turn_fast75_s0/global_step_75
export LABEL=turnppo_s75
export COND=turn_ppo_b0
exec bash /home/tiger/xiaoxuan/arlca-8b/scripts/asearcher/p8b_eval_worker.sh
