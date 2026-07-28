#!/usr/bin/env bash
# 8B ASearcher-consistent protocol — Qwen3-8B-Base, ASearcher Base-35k UNFILTERED,
# single tool (search), wiki-18/e5 closed setting. Decisions grilled 2026-07-29:
#   - 32-turn nominal horizon; binding constraint is the 16k prompt with CLEAN
#     context-overflow termination (env.ctx_terminate — no silent left-truncation).
#   - ASearcher trajectory format: top-5 docs, one <information> block capped at 5k chars,
#     never discarded; history keeps the model's FULL response (<think> + <search>).
#   - RL from base, no cold start (ASearcher recipe). Chat template = Qwen3 ChatML
#     (ships with the base model); step-0 grammar compliance is a profiling gate.
#   - Optimization budget identical to the 4B arms (256x5, lr 1e-6, 150 steps).
#   - Partial rollout OFF here; enable per-run via +env.partial_rollout_enable=true.
# Profiling-tunable knobs use ${VAR:-default} so worker scripts can shorten runs.

export MAX_STEPS=32                # nominal ASearcher horizon (7B/14B budget in the paper)
export HISTORY_LENGTH=32           # >= MAX_STEPS => nothing dropped from history
export MAX_PROMPT_LENGTH=16384     # ASearcher-Web-7B setting; ctx_terminate ends episodes cleanly
export MAX_RESPONSE_LENGTH=1024    # ASearcher max_new_tokens
export TRUNCATION=left             # safety valve ONLY — ctx_terminate should keep this at zero
export SEARCH_DOC_MAX_WORDS=0      # per-doc word guard OFF (block-level 5k char cap governs instead)

export ENABLE_THINKING=False       # harmless for the base template; keeps parity with 4B configs

export TRAIN_BATCH=256             # matched optimization budget (identical to 4B arms)
export GROUP_SIZE=5                # -> 1280 trajectories/step
export PPO_MINI_BATCH=512
export MICRO_BSZ="${MICRO_BSZ:-1}" # 8B @ 17.4k ctx: conservative; profiling tunes upward
export LOGPROB_MICRO="${LOGPROB_MICRO:-2}"
export LR=1e-6
export CRITIC_LR=1e-5              # gae_turn arm only
export KL_LOSS_COEF=0.001
export GAMMA=1.0
export LAM=1.0
export TOTAL_STEPS="${TOTAL_STEPS:-150}"
export TOTAL_EPOCHS=2              # 35,583/256 = 139 steps/epoch; epochs=2 lets 150 bind (gotcha #6)
export VAL_FREQ="${VAL_FREQ:-25}"
export SAVE_FREQ="${SAVE_FREQ:-25}"
export VAL_BATCH="${VAL_BATCH:-256}"  # env-pool size for val (val file = val_2048.parquet)
export TOPK=5                      # ASearcher top-5 search docs

export WANDB_PROJECT=ca-rung3-8b-asearcher

# ASearcher trajectory-format env flags (new keys -> '+' prefix).
export PROTOCOL_OVERRIDES=(
  +env.search.info_char_cap=5000
  +env.asearcher_history=true
  +env.ctx_terminate=true
  +env.ctx_terminate_margin=256
)
