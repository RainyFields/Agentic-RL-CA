#!/bin/bash
# Wave-3 HCAPO ablation (user-approved 2026-07-27): stock omega=1.0. Qwen3-4B, NON-thinking @2048, 500 steps.
# Matched to .arlca-{token-grpo,turn-ppo-b0}-4b-nt-s0.sh so GRPO/turn-PPO are direct comparators.
# Hypothesis: HCAPO's 1.7B SearchQA collapse (0.357 @ clip 0.754) was truncation-driven; at
# non-thinking@2048 projected mean resp ~250-460 << 2048 so clip should be ~0. rho instrumentation on.
exec env MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3-4B COND=hcapo SEED=0 PROTOCOL=4turn_nothink2k \
  MICRO_BSZ_OVERRIDE=4 LOGPROB_MICRO_OVERRIDE=4 GPU_MEM_UTIL=0.5 HCAPO_LOG_RHO=1 \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_train.sh
