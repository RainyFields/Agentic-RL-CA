#!/bin/bash
# COLLECT: sampled rollouts (temp 1.0, training-matched) from several checkpoints over the
# repeated held-out val slice, for rubric derivation + p_t calibration (judge-reward-arms
# Phase 0.4/1). One p8b_eval_worker.sh pass per checkpoint, back to back on one worker.
#   CKPTS="s25=/mnt/hdfs/.../global_step_25 s50=... s75=..." bash p8b_collect_worker.sh
set -uo pipefail
CKPTS="${CKPTS:?set CKPTS=\"label=ckpt_dir [label=ckpt_dir ...]\"}"
REPO=${REPO:-/home/tiger/xiaoxuan/arlca-8b-judge}
EVAL_FILE="${EVAL_FILE:-/mnt/hdfs/mlsys/users/xiaoxuan/agentic_rl_ca/data_asearcher_base/val_512_x4.parquet}"
EVAL_TEMP="${EVAL_TEMP:-1.0}"

rc_all=0
for spec in $CKPTS; do
  LABEL="collect_${spec%%=*}"; CKPT="${spec#*=}"
  echo "==== COLLECT $LABEL from $CKPT $(date -u) ===="
  CKPT="$CKPT" LABEL="$LABEL" REPO="$REPO" EVAL_FILE="$EVAL_FILE" EVAL_TEMP="$EVAL_TEMP" \
    bash "$REPO/scripts/asearcher/p8b_eval_worker.sh"
  rc=$?; [ "$rc" = 0 ] || rc_all=1
  echo "==== COLLECT $LABEL exit=$rc $(date -u) ===="
  # eval passes stage per-ckpt merges under /tmp; drop them so 3 ckpts fit local disk
  rm -rf "/tmp/eval_merged_${LABEL}"
done
exit "$rc_all"
