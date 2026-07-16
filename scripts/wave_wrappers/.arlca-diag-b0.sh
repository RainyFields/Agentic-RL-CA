#!/bin/bash
# Queue #4.5 wrapper (user OK in-session 2026-07-16): F8a credit-alignment diagnostic
# on B0-s0 checkpoints 150/300/500 (4turn_think2k, same-checkpoint rule). This is the
# pre-registered lambda-sweep gate — see docs/plan_lambda_sweep.md.
set -uo pipefail
REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
CKROOT=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/turn_ppo_b0_qwen3-1.7b_4turn_think2k_s0
bash "$REPO/scripts/retriever_serve.sh" || { echo "[diag-worker] exit: FAILED (retriever)"; exit 1; }
rc=0
for STEP in 150 300 500; do
  if bash "$REPO/scripts/run_diag.sh" "$CKROOT/global_step_$STEP" "b0_s0_step$STEP" 4turn_think2k; then
    echo "[diag-worker] ckpt$STEP DONE"
  else
    echo "[diag-worker] ckpt$STEP FAILED"
    rc=1
  fi
done
if [ $rc -eq 0 ]; then
  echo "[diag-worker] exit: DONE (f8a b0-s0 ckpts 150/300/500)"
else
  echo "[diag-worker] exit: FAILED (f8a b0-s0 — at least one ckpt failed)"
  exit 1
fi
