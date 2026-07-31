#!/bin/bash
# Wave-4 2x2: lift-scored HCAPO, s_final=last_obs. Qwen3-4B non-thinking@2048, 500 steps, seed 0.
exec env MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B COND=hcapo_lift SEED=0 PROTOCOL=4turn_nothink2k \
  MICRO_BSZ_OVERRIDE=4 LOGPROB_MICRO_OVERRIDE=4 GPU_MEM_UTIL=0.5 HCAPO_LOG_RHO=1 \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
