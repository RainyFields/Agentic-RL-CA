#!/bin/bash
# HCAPO (paper-correct core) with s_final = agent's final answer. Qwen3-4B, non-thinking@2048,
# 500 steps, seed 0 — matched to the other non-thinking arms. Watch: does the drift ignite at all,
# and does rho spread stay alive past step 200 (last_obs variant died at sigma~0.03).
exec env MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B COND=hcapo_ans SEED=0 PROTOCOL=4turn_nothink2k \
  MICRO_BSZ_OVERRIDE=4 LOGPROB_MICRO_OVERRIDE=4 GPU_MEM_UTIL=0.5 HCAPO_LOG_RHO=1 \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
