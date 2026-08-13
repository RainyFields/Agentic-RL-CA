#!/bin/bash
# Collect temp-1.0 x4 val rollouts from the async75kv60 baseline at steps 25/50/75.
B=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/token_grpo_qwen3-8b-base_32turn_async75kv60_s0
export CKPTS="agrpo_s25=$B/global_step_25 agrpo_s50=$B/global_step_50 agrpo_s75=$B/global_step_75"
exec bash /home/tiger/xiaoxuan/arlca-8b-judge/scripts/asearcher/p8b_collect_worker.sh
