#!/usr/bin/env bash
# Phase 3.3 — full-set paper-protocol eval (51,713 rows, 7 datasets, greedy) of ONE checkpoint.
#   scripts/eval_search_full.sh <ckpt_or_hf_dir> <run_label> [condition] [protocol]
#     ckpt_or_hf_dir : verl ckpt dir (…/global_step_N — will be HF-merged, world-size
#                      independent) OR an already-merged HF model dir
#     run_label      : output tag, e.g. b1_s0_step500
# Full test set is evaluated ONLY for pre-registered checkpoints (fixed-budget FINAL +
# best-val secondary) — plan §Pre-registered decision rules.
# Run on a GPU worker with the retriever healthy.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CKPT="${1:?usage: eval_search_full.sh <ckpt_or_hf_dir> <run_label> [condition] [protocol]}"
LABEL="${2:?need run_label}"
COND_NAME="${3:-turn_ppo_b0}"     # condition only sets adv_estimator for config validity
PROTOCOL="${4:-4turn}"

source "$REPO_DIR/configs/protocol_${PROTOCOL}.sh"
source "$REPO_DIR/configs/cond_${COND_NAME}.sh"
require_data
require_retriever
activate_env

OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/eval_full/$LABEL}"   # env-overridable: eval worker routes to /tmp (300G) to keep 8GB HF-merges off the 26G-free shared volume
mkdir -p "$OUT_DIR"

# --- HF-merge route if given a verl checkpoint (contains actor/ with fsdp shards) ---
MODEL_DIR="$CKPT"
if [[ -d "$CKPT/actor" ]]; then
  MERGED="$OUT_DIR/hf_merged"
  if [[ ! -f "$MERGED/config.json" ]]; then
    echo "[eval] merging $CKPT/actor -> $MERGED"
    python3 "$REPO_DIR/scripts/model_merger.py" merge --backend fsdp \
        --local_dir "$CKPT/actor" --target_dir "$MERGED"
  fi
  MODEL_DIR="$MERGED"
fi

VAL_FILES="${VAL_FILES:-$DATA_DIR/test.parquet}"   # full 51,713-row set
EVAL_VAL_BATCH="${EVAL_VAL_BATCH:-1024}"

set -x
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="${EVAL_ADV_ESTIMATOR:-grpo}" \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size="$EVAL_VAL_BATCH" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation="$TRUNCATION" \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING:-False}" \
    actor_rollout_ref.model.path="$MODEL_DIR" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BSZ" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    env.env_name=search \
    env.seed=0 \
    env.max_steps="$MAX_STEPS" \
    env.rollout.n=1 \
    env.history_length="$HISTORY_LENGTH" \
    env.search.search_url="$SEARCH_URL" \
    env.search.topk="$TOPK" \
    trainer.logger="['console']" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="evalfull_${LABEL}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_only=True \
    trainer.val_before_train=True \
    +trainer.validation_data_dir="$OUT_DIR" \
    "${@:5}"
set +x

python3 "$REPO_DIR/analysis/aggregate_eval.py" \
    --jsonl "$OUT_DIR/val_trajectories_step*.jsonl" \
    --out_prefix "$OUT_DIR/paper_table" \
    --label "$LABEL"
echo "[eval] paper table at $OUT_DIR/paper_table.{csv,json}"
