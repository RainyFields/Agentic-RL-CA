#!/bin/bash
# Queue #4.5 continuation (adapted after ckpt pruning discovery, 2026-07-16): F8a
# credit-alignment diagnostic on B0-s1 ckpts 100/200/500 — the surviving multi-checkpoint
# actor+critic set (B0-s0 150/300 were pruned to empty dirs by max_ckpt_to_keep=1).
# Pre-registration deviation logged in decision_log; gate evaluated on s1 {100,200,500}
# + s0 {500}. See docs/plan_lambda_sweep.md.
set -uo pipefail
REPO=/home/tiger/xiaoxuan/Agentic-RL-CA
CKROOT=/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/checkpoints/turn_ppo_b0_qwen3-1.7b_4turn_think2k_s1
bash "$REPO/scripts/retriever_serve.sh" || { echo "[diag-worker] exit: FAILED (retriever)"; exit 1; }
rc=0
for STEP in 100 200 500; do
  if bash "$REPO/scripts/run_diag.sh" "$CKROOT/global_step_$STEP" "b0_s1_step$STEP" 4turn_think2k; then
    echo "[diag-worker] ckpt$STEP DONE"
  else
    echo "[diag-worker] ckpt$STEP FAILED"
    rc=1
  fi
done
if [ $rc -eq 0 ]; then
  echo "[diag-worker] exit: DONE (f8a b0-s1 ckpts 100/200/500)"
else
  echo "[diag-worker] exit: FAILED (f8a b0-s1 — at least one ckpt failed)"
  exit 1
fi
