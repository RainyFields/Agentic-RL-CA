#!/usr/bin/env bash
# Phase 3b.2 — credit-alignment diagnostic for ONE checkpoint (offline, single GPU).
#   scripts/run_diag.sh <ckpt_or_hf_dir> <run_label> [protocol]
#     ckpt_or_hf_dir : verl ckpt dir (…/global_step_N — HF-merged first) OR merged HF dir
#     run_label      : output tag, e.g. token_grpo_s0_step250
#     protocol       : configs/protocol_<protocol>.sh, default 4turn_think2k (MUST be the
#                      protocol the checkpoint was trained under — off-protocol prefixes
#                      are off-policy and invalidate dV-hat)
# Env knobs: DIAG_N_TRAJ (256), DIAG_K (8), DIAG_KCHECK (16), DIAG_KCHECK_FRAC (0.125),
#            DIAG_POOL (256), DIAG_DATA (val_2048.parquet), SEED (0).
# Run on a GPU worker with the retriever healthy (scripts/retriever_serve.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CKPT="${1:?usage: run_diag.sh <ckpt_or_hf_dir> <run_label> [protocol]}"
LABEL="${2:?need run_label}"
PROTOCOL="${3:-4turn_think2k}"

source "$REPO_DIR/configs/protocol_${PROTOCOL}.sh"
require_data
require_retriever
activate_env

OUT_DIR="$REPO_DIR/outputs/diag/$LABEL"
mkdir -p "$OUT_DIR"

# --- HF-merge route if given a verl checkpoint (same as eval_search_full.sh) ---
MODEL_DIR="$CKPT"
if [[ -d "$CKPT/actor" ]]; then
  MERGED="$OUT_DIR/hf_merged"
  if [[ ! -f "$MERGED/config.json" ]]; then
    echo "[diag] merging $CKPT/actor -> $MERGED"
    python3 "$REPO_DIR/scripts/model_merger.py" merge --backend fsdp \
        --local_dir "$CKPT/actor" --target_dir "$MERGED"
  fi
  MODEL_DIR="$MERGED"
fi

THINK_FLAG="--enable-thinking"
if [[ "${ENABLE_THINKING:-False}" != "True" ]]; then THINK_FLAG="--no-thinking"; fi

set -x
python3 -m credit_assignment.diag_runner \
    --model "$MODEL_DIR" \
    --data "${DIAG_DATA:-$DATA_DIR/val_2048.parquet}" \
    --out "$OUT_DIR" \
    --n-traj "${DIAG_N_TRAJ:-256}" \
    --k "${DIAG_K:-8}" \
    --k-check "${DIAG_KCHECK:-16}" \
    --k-check-frac "${DIAG_KCHECK_FRAC:-0.125}" \
    --pool "${DIAG_POOL:-256}" \
    --seed "${SEED:-0}" \
    --max-turns "$MAX_STEPS" \
    --history-length "$HISTORY_LENGTH" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --max-response-length "$MAX_RESPONSE_LENGTH" \
    --truncation "$TRUNCATION" \
    --search-url "$SEARCH_URL" \
    --topk "$TOPK" \
    $THINK_FLAG \
    2>&1 | tee "$OUT_DIR/diag.log"
