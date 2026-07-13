#!/usr/bin/env bash
# Phase 1.3 — toy-first gate (HARD GATE: user reviews the trajectory dumps).
# Runs a tiny training job per existing condition (TRAIN_BATCH=8, GROUP=4, 2 steps) with
# per-turn dumps, then builds trajectory_<algo>.md + gate_metrics_<algo>.json per condition.
# Run ON the GPU worker with the retriever already healthy.
#   scripts/run_toy_gate.sh [cond ...]     default: token_ppo turn_ppo_b0 token_grpo gigpo hcapo b1 b1_shuffle
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CONDS=("${@:-token_ppo turn_ppo_b0 token_grpo gigpo hcapo}")
[[ $# -eq 0 ]] && CONDS=(token_ppo turn_ppo_b0 token_grpo gigpo hcapo b1 b1_shuffle)

PROTOCOL="${PROTOCOL:-4turn}"
source "$REPO_DIR/configs/protocol_${PROTOCOL}.sh"

for cond in "${CONDS[@]}"; do
  echo "=========== toy gate: $cond ==========="
  TOY=1 "$REPO_DIR/scripts/run_condition.sh" "$cond" 0 "$PROTOCOL" || {
    echo "[toy-gate] $cond RUN FAILED"; exit 1; }
  python3 "$REPO_DIR/analysis/toy_gate_report.py" \
      --jsonl "$REPO_DIR/outputs/search_toy/${cond}/rollout_log.jsonl" \
      --algo "$cond" \
      --out_dir "$REPO_DIR/outputs/search_toy" \
      --max_prompt_length "$MAX_PROMPT_LENGTH" \
      --max_response_length "$MAX_RESPONSE_LENGTH"
done
echo "[toy-gate] all conditions done — dumps in outputs/search_toy/ (awaiting user review)"
