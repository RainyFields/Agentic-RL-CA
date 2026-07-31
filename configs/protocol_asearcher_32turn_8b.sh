#!/usr/bin/env bash
# Rung-3 8B protocol — Qwen3-8B-Base, ASearcher-Base UNFILTERED, single tool (search),
# ASearcher-consistent trajectory format (arXiv 2508.07976, ASearcher-Web-7B config):
#   32-turn nominal budget, 16k prompt, 1024 response/turn, top-5 docs, whole
#   <information> block capped at 5k chars and NEVER dropped from history, past model
#   responses (<think> + <search>) kept verbatim in the rebuilt prompt.
# Deviations from ASearcher (user-approved 2026-08-01 grilling session):
#   - synchronous verl rollouts (their partial rollout is an AReaL async feature);
#   - matched rung-3 optimization budget (256x5, lr 1e-6, 150 steps) instead of theirs;
#   - context overflow -> CLEAN episode termination (PROMPT_OVERFLOW_TERMINATE), because
#     32 turns of never-discarded top-5 blocks cannot fit 16k: length is the real bound
#     (~turn 6-8); left-truncation would silently eat the template header + question.

export MAX_STEPS=32                # ASearcher 7B/14B turn budget (nominal; length binds first)
export HISTORY_LENGTH=32           # >= MAX_STEPS => nothing dropped from history
export MAX_PROMPT_LENGTH=16384     # ASearcher-Web-7B max_prompt_length
export MAX_RESPONSE_LENGTH=1024    # ASearcher max_new_tokens per turn
export TRUNCATION=left             # safety valve only — overflow guard fires first for active rows
export TOPK=5                      # ASearcher: 5 top search docs per query
export SEARCH_DOC_MAX_WORDS=0      # per-doc cap OFF — ASearcher caps the whole block instead
export SEARCH_INFO_BLOCK_MAX_CHARS=5000   # ASearcher <search> observation cap (chars)
export SEARCH_HISTORY_KEEP_RESPONSE=1     # keep <think>+<search> verbatim in history
export PROMPT_OVERFLOW_TERMINATE=1        # clean terminate instead of left-truncate

# Qwen3-8B-Base ships a chat template; enable_thinking is inert for it (base model,
# reasoning happens in-response per the search grammar).
export ENABLE_THINKING=False

# Matched rung-3 optimization budget (identical to the 4B arms; only micro sizes retuned).
export TRAIN_BATCH=256             # prompts per step
export GROUP_SIZE=5                # -> 1280 trajectories/step
export PPO_MINI_BATCH=512
export MICRO_BSZ=1                 # 8B @ 16k+1k ctx (profiling 2026-08-01 sizes this)
export LOGPROB_MICRO=2
export LR=1e-6
export CRITIC_LR=1e-5              # PPO/gae_turn arms only
export KL_LOSS_COEF=0.001
export GAMMA=1.0
export LAM=1.0
export TOTAL_STEPS=150
export TOTAL_EPOCHS=2              # 30,318 rows / 256 = 118 steps/epoch — epochs=1 would cap
                                   # below 150 (verl mins total_training_steps with len(dataloader)*epochs)
export VAL_FREQ=25
export SAVE_FREQ=25
export VAL_BATCH=256               # short during-training validation (512-row val file)

export WANDB_PROJECT=ca-rung3-8b-asearcher
