#!/bin/bash
# Thinking ablation (user-requested 2026-07-26): gigpo Qwen3-4B, NON-thinking @2048, 500 steps.
# Matched 1:1 to .arlca-gigpo-4b-s0.sh (thinking) — identical model, budget, batch, memory config;
# ONLY enable_thinking differs. Pairs for the exact thinking-vs-non-thinking comparison.
exec env MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B COND=gigpo SEED=0 PROTOCOL=4turn_nothink2k \
  MICRO_BSZ_OVERRIDE=4 LOGPROB_MICRO_OVERRIDE=4 GPU_MEM_UTIL=0.5 \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
