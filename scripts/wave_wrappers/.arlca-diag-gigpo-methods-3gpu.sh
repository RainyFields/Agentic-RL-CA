#!/bin/bash
# GiGPO credit-alignment (Phase 3b.3) — 3-GPU variant (8-GPU pods couldn't schedule 2026-07-19;
# diag only needs ~1 GPU vLLM + retriever). Retriever on GPUs 1,2; diag_runner vLLM on GPU0.
set -uo pipefail
export RETR_GPUS=1,2
REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
CK=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/gigpo_qwen3-1.7b_4turn_think2k_s0
bash "$REPO/scripts/retriever_serve.sh" || { echo "[diagM-worker] exit: FAILED (retriever)"; exit 1; }
source "$REPO/scripts/_common.sh"; activate_env; cd "$REPO"
rc=0
bash "$REPO/scripts/run_diag_method.sh" gigpo "$CK/global_step_500" gigpo_1p7b_s0_step500 4turn_think2k \
  || { echo "[diagM-worker] gigpo step500 FAILED"; rc=1; }
if [ $rc -eq 0 ]; then
  python3 -m credit_assignment.method_align --diag-root "$REPO/outputs/diag_methods" \
    --labels gigpo_1p7b_s0_step500 --out "$REPO/outputs/diag_methods/gigpo_align.json" || rc=1
fi
[ $rc -eq 0 ] && echo "[diagM-worker] exit: DONE (gigpo credit-alignment step500)" \
  || { echo "[diagM-worker] exit: FAILED"; exit 1; }
