#!/usr/bin/env bash
# Phase 3b.3 — per-method credit-alignment diagnostic for ONE checkpoint (offline, single GPU).
#   scripts/run_diag_method.sh <method> <ckpt_or_hf_dir> <run_label> [protocol]
#     method         : gigpo (critic-free group-relative; the reconstructable case)
#     ckpt_or_hf_dir : verl ckpt dir (…/global_step_N — actor HF-merged first) OR merged HF dir
#     run_label      : output tag, e.g. gigpo_1p7b_s0_step500
#     protocol       : default 4turn_think2k (MUST match the checkpoint's training protocol)
# Env knobs: DIAG_N_PROMPTS (64), DIAG_GROUP_N (5, MUST match training GROUP_SIZE), DIAG_K (8),
#            DIAG_KCHECK (16), DIAG_KCHECK_FRAC (0.125), DIAG_POOL (320), DIAG_DATA, SEED (0).
# Run on a GPU worker with the retriever healthy (scripts/retriever_serve.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

METHOD="${1:?usage: run_diag_method.sh <method> <ckpt_or_hf_dir> <run_label> [protocol]}"
CKPT="${2:?need ckpt}"
LABEL="${3:?need run_label}"
PROTOCOL="${4:-4turn_think2k}"

source "$REPO_DIR/configs/protocol_${PROTOCOL}.sh"
require_data
require_retriever
activate_env

OUT_DIR="$REPO_DIR/outputs/diag_methods/$LABEL"
mkdir -p "$OUT_DIR"

# --- HF-merge the ACTOR if given a verl checkpoint (GiGPO is critic-free: no critic merge) ---
MODEL_DIR="$CKPT"
if [[ -d "$CKPT/actor" ]]; then
  MERGED="$OUT_DIR/hf_merged"
  if [[ ! -f "$MERGED/config.json" ]]; then
    echo "[diagM] merging $CKPT/actor -> $MERGED"
    python3 "$REPO_DIR/scripts/model_merger.py" merge --backend fsdp \
        --local_dir "$CKPT/actor" --target_dir "$MERGED"
  fi
  MODEL_DIR="$MERGED"
fi

THINK_FLAG="--enable-thinking"
if [[ "${ENABLE_THINKING:-False}" != "True" ]]; then THINK_FLAG="--no-thinking"; fi

set -x
python3 -m credit_assignment.diag_runner_methods \
    --method "$METHOD" \
    --model "$MODEL_DIR" \
    --data "${DIAG_DATA:-$DATA_DIR/val_2048.parquet}" \
    --out "$OUT_DIR" \
    --n-prompts "${DIAG_N_PROMPTS:-64}" \
    --group-n "${DIAG_GROUP_N:-5}" \
    --k "${DIAG_K:-8}" \
    --k-check "${DIAG_KCHECK:-16}" \
    --k-check-frac "${DIAG_KCHECK_FRAC:-0.125}" \
    --pool "${DIAG_POOL:-320}" \
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
