#!/bin/bash
# Phase 3.3 horizon eval (user OK 2026-07-21): pre-RL Qwen3-4B FLOOR row, full 51,713-row 7-task set,
# greedy, 4turn_think2k. Dedicated worker so the floor runs in PARALLEL with the GiGPO full pass
# (schema already validated on GRPO/GiGPO smokes, so SMOKE=0 — straight to the full run).
exec env \
  EVAL_SPECS="/mnt/hdfs/mlsys/models/Qwen3-4B|eval4b_floor|token_grpo" \
  PROTOCOL=4turn_think2k JOB_TAG=eval4b_floor SMOKE=0 \
  bash /home/tiger/xiaoxuan/Agentic-RL-CA/scripts/worker_eval.sh
