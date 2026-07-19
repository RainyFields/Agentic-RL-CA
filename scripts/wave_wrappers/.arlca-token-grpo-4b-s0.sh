#!/bin/bash
# Qwen3-4B scale-up (user OK 2026-07-19): token-GRPO s0 (RQ2, critic-free). 4turn_think2k.
# No critic -> lighter; MICRO_BSZ 8->4 + vLLM util 0.6->0.5 for 4B headroom (no optimizer offload).
exec env MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B COND=token_grpo SEED=0 PROTOCOL=4turn_think2k \
  MICRO_BSZ_OVERRIDE=4 LOGPROB_MICRO_OVERRIDE=4 GPU_MEM_UTIL=0.5 \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
