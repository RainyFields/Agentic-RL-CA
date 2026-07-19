#!/bin/bash
# Qwen3-4B scale-up (user OK 2026-07-19): token-PPO s0 (RQ1, critic). 4turn_think2k, matched budget.
# 4B memory: MICRO_BSZ 8->4, vLLM util 0.6->0.5, actor+critic optimizer_offload -> CPU (avoid the
# 1.7B gae fragmentation-OOM livelocks at higher param count). Math unchanged (PPO_MINI_BATCH=512).
exec env MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B COND=token_ppo SEED=0 PROTOCOL=4turn_think2k \
  MICRO_BSZ_OVERRIDE=4 LOGPROB_MICRO_OVERRIDE=4 GPU_MEM_UTIL=0.5 OPTIM_OFFLOAD=True \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
