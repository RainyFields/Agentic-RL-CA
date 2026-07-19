#!/bin/bash
# Qwen3-4B scale-up (user OK 2026-07-19): GiGPO s0 (RQ2, critic-free group-relative turn). 4turn_think2k.
# Same lighter 4B profile as GRPO-4b.
exec env MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B COND=gigpo SEED=0 PROTOCOL=4turn_think2k \
  MICRO_BSZ_OVERRIDE=4 LOGPROB_MICRO_OVERRIDE=4 GPU_MEM_UTIL=0.5 \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
