#!/usr/bin/env bash
# Rung-4 ScienceWorld launcher (fork of run_condition.sh: no retriever, dummy dataset,
# sciworld env block).
#   scripts/sciworld/run_sciworld.sh <condition> [seed] [protocol]
#     condition : configs/cond_<condition>.sh   (sw_orm_grpo | sw_prm_rtg)
#     protocol  : default sciworld_4b
# Env knobs: GATE=1 (zero-shot gate: 72 groups x G=5, 1 step, rollout dump, no val),
#            GATE_GROUP_SIZE, TOTAL_STEPS, MODEL_PATH, RESUME=auto|disable.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"

COND_NAME="${1:?usage: run_sciworld.sh <condition> [seed] [protocol]}"
SEED="${2:-0}"
PROTOCOL="${3:-sciworld_4b}"

source "$REPO_DIR/configs/protocol_${PROTOCOL}.sh"
source "$REPO_DIR/configs/cond_${COND_NAME}.sh"

require_model_path

MICRO_BSZ="${MICRO_BSZ_OVERRIDE:-$MICRO_BSZ}"
LOGPROB_MICRO="${LOGPROB_MICRO_OVERRIDE:-${LOGPROB_MICRO:-2}}"
VAL_BATCH="${VAL_BATCH_OVERRIDE:-$VAL_BATCH}"
# protocol files export TOTAL_STEPS/VAL_FREQ after the caller's env (lesson #3, handoff
# 2026-07-28) — the only way to override is after sourcing:
TOTAL_STEPS="${TOTAL_STEPS_OVERRIDE:-$TOTAL_STEPS}"
VAL_FREQ="${VAL_FREQ_OVERRIDE:-$VAL_FREQ}"

# ---- Zero-shot gate mode (design doc §gate): temp-1 rollouts of the BASE model through
# the exact production wrapper/prompt/truncation; dump every turn; die after 1 "step"
# (the dump happens during rollout, before the single param update ever matters).
GATE="${GATE:-0}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
if [[ "$GATE" == "1" ]]; then
  TRAIN_BATCH="${GATE_GROUPS:-72}"; GROUP_SIZE="${GATE_GROUP_SIZE:-5}"
  PPO_MINI_BATCH=512
  TOTAL_STEPS=1; VAL_FREQ=1000000; SAVE_FREQ=1000000
  VAL_BATCH=8            # val pool still gets built; keep its JVM count trivial
  VAL_BEFORE_TRAIN=False
  export DUMP_TRAIN_SAMPLE=1
  export DUMP_TRAIN_PATH="$REPO_DIR/outputs/sciworld_gate/${CONDITION}"
  mkdir -p "$DUMP_TRAIN_PATH"; rm -f "$DUMP_TRAIN_PATH/rollout_log.jsonl"
fi

# ---- dummy dataset (modality + batch-size carrier only, alfworld pattern) ----
DUMMY_DIR="${DUMMY_DIR:-/tmp/sciworld_dummy_${TRAIN_BATCH}_${VAL_BATCH}}"
if [[ ! -f "$DUMMY_DIR/text/train.parquet" ]]; then
  python3 -m examples.data_preprocess.prepare --mode text \
      --local_dir "$DUMMY_DIR" \
      --train_data_size "$TRAIN_BATCH" --val_data_size "$VAL_BATCH"
fi

MODEL_TAG="$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')"
GATE_SUFFIX=""; [[ "$GATE" == "1" ]] && GATE_SUFFIX="_gate"
EXP_NAME="${EXP_NAME:-${CONDITION}_s${SEED}${GATE_SUFFIX}}"
CKPT_DIR="${CKPT_DIR:-$HDFS_PROJECT/checkpoints/rung4_$EXP_NAME}"

activate_env

set -x
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator="$ADV_ESTIMATOR" \
    algorithm.gamma="$GAMMA" \
    algorithm.lam="$LAM" \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DUMMY_DIR/text/train.parquet" \
    data.val_files="$DUMMY_DIR/text/test.parquet" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.val_batch_size="$VAL_BATCH" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=False \
    data.truncation="$TRUNCATION" \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING:-False}" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BSZ" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    env.env_name=sciworld \
    env.seed="$SEED" \
    env.max_steps="$MAX_STEPS" \
    env.rollout.n="$GROUP_SIZE" \
    env.history_length="$HISTORY_LENGTH" \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.max_actor_ckpt_to_keep="${MAX_CKPT_KEEP:-1}" \
    trainer.max_critic_ckpt_to_keep="${MAX_CKPT_KEEP:-1}" \
    trainer.test_freq="$VAL_FREQ" \
    trainer.total_epochs=100000 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    "${EXTRA_OVERRIDES[@]}" \
    "${@:4}"
