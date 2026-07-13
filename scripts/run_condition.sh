#!/usr/bin/env bash
# Generic per-condition launcher.
#   scripts/run_condition.sh <condition> [seed] [protocol]
#     condition : configs/cond_<condition>.sh   (token_ppo | turn_ppo_b0 | token_grpo | gigpo | hcapo | b1 | b1_shuffle | ...)
#     seed      : integer, default 0
#     protocol  : configs/protocol_<protocol>.sh, default 4turn (8turn = horizon stress test)
# Env knobs: TOY=1 (toy-gate run: tiny batch + trajectory dumps), TOTAL_STEPS, W_STEP,
#            MODEL_PATH, SEARCH_URL, RESUME=auto|disable.
# Run on the GPU worker (inside the verl-agent venv), retriever already up.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

COND_NAME="${1:?usage: run_condition.sh <condition> [seed] [protocol]}"
SEED="${2:-0}"
PROTOCOL="${3:-4turn}"

source "$REPO_DIR/configs/protocol_${PROTOCOL}.sh"
source "$REPO_DIR/configs/cond_${COND_NAME}.sh"

require_model_path
require_data
require_retriever

# ---- Toy-gate mode (plan Phase 1.3): tiny run + per-turn trajectory dumps ----
TOY="${TOY:-0}"
if [[ "$TOY" == "1" ]]; then
  TRAIN_BATCH=8; GROUP_SIZE=4; PPO_MINI_BATCH=32; TOTAL_STEPS=2; VAL_FREQ=1000000; SAVE_FREQ=1000000
  EXTRA_OVERRIDES+=( +trainer.dump_trajectories_dir="$REPO_DIR/outputs/search_toy/${CONDITION}" )
fi

MODEL_TAG="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')"
EXP_NAME="${EXP_NAME:-${CONDITION}_${MODEL_TAG}_${PROTOCOL}_s${SEED}$([[ $TOY == 1 ]] && echo _toy)}"
CKPT_DIR="${CKPT_DIR:-$HDFS_PROJECT/checkpoints/$EXP_NAME}"

# Critic settings only for value-based arms (critic auto-on for gae / gae_turn).
CRITIC_OVERRIDES=()
if [[ "$ADV_ESTIMATOR" == "gae" || "$ADV_ESTIMATOR" == "gae_turn" ]]; then
  CRITIC_OVERRIDES=(
    critic.optim.lr="$CRITIC_LR"
    critic.model.path="$MODEL_PATH"
    critic.model.use_remove_padding=True
    critic.model.enable_gradient_checkpointing=True
    critic.ppo_micro_batch_size_per_gpu="$MICRO_BSZ"
    critic.model.fsdp_config.param_offload=False
    critic.model.fsdp_config.optimizer_offload=False
  )
fi

activate_env

set -x
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="$ADV_ESTIMATOR" \
    algorithm.gamma="$GAMMA" \
    algorithm.lam="$LAM" \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/val_2048.parquet" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size="$VAL_BATCH" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation="$TRUNCATION" \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BSZ" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    "${CRITIC_OVERRIDES[@]}" \
    env.env_name=search \
    env.seed="$SEED" \
    env.max_steps="$MAX_STEPS" \
    env.rollout.n="$GROUP_SIZE" \
    env.history_length="$HISTORY_LENGTH" \
    env.search.search_url="$SEARCH_URL" \
    env.search.topk="$TOPK" \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$VAL_FREQ" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.val_before_train=True \
    "${EXTRA_OVERRIDES[@]}" \
    "${@:4}"
