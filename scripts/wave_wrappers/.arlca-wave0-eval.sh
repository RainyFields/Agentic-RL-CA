#!/bin/bash
# Queue #2 wrapper (pre-registered FINAL; user OK 2026-07-15): wave-0 token_grpo s0
# step-500 full-set eval (51,713 rows). Retriever first, then eval_search_full.sh.
set -uo pipefail
REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
bash "$REPO/scripts/retriever_serve.sh" || { echo "[eval-worker] exit: FAILED (retriever)"; exit 1; }
if bash "$REPO/scripts/eval_search_full.sh" \
    /mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/token_grpo_qwen3-1.7b_4turn_think2k_s0/global_step_500 \
    wave0_grpo_s0_final token_grpo 4turn_think2k; then
  echo "[eval-worker] exit: DONE (wave0_grpo_s0_final)"
else
  echo "[eval-worker] exit: FAILED (wave0_grpo_s0_final)"
  exit 1
fi
