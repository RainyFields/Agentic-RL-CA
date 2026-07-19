#!/bin/bash
# Qwen3-4B scale-up (user OK 2026-07-19): turn-PPO/B0 s0 (RQ1, gae_turn critic). 4turn_think2k.
# Same 4B memory profile as token-PPO-4b. Direct scale test of the critic-quality diagnosis.
exec env MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B COND=turn_ppo_b0 SEED=0 PROTOCOL=4turn_think2k \
  MICRO_BSZ_OVERRIDE=4 LOGPROB_MICRO_OVERRIDE=4 GPU_MEM_UTIL=0.5 OPTIM_OFFLOAD=True \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
