#!/usr/bin/env bash
# Rung-3 feasibility protocol — Qwen3-4B-Instruct-2507, ASearcher-Base, single tool (search).
# no cold start (RL from base). 8-turn horizon, 16k prompt (fits ~30-turn full history on
# wiki-18 ~103-word chunks; 8 turns leaves large headroom). Thinking OFF (Instruct-2507 is
# non-thinking by construction). See docs/reports/2026-07-25_asearcher_audit + decision_log 2026-07-27.

export MAX_STEPS=8                 # 8-turn feasibility horizon (path to 30 once stable)
export HISTORY_LENGTH=8            # >= MAX_STEPS => full history, nothing dropped
export MAX_PROMPT_LENGTH=16384     # 16k: fits full 8-turn (and up to ~30-turn) history on wiki-18
export MAX_RESPONSE_LENGTH=512     # per-turn generation only (retrieved <information> is prompt, not response)
export TRUNCATION=left             # safety valve ONLY — any nonzero truncation rate is a bug/alarm
export SEARCH_DOC_MAX_WORDS=180    # per-doc cap guard (~240 tok); never fires on wiki-18 ~103-word chunks

# Non-thinking (Instruct-2507); if ever run on the hybrid Qwen3-4B, this flag disables think.
export ENABLE_THINKING=False

# Matched optimization budget (Search-QA column of the HCAPO/GiGPO papers; identical for all arms).
export TRAIN_BATCH=256             # prompts per step
export GROUP_SIZE=5                # -> 1280 trajectories/step
export PPO_MINI_BATCH=512
export MICRO_BSZ=2                 # 4B @ 16k ctx: smaller micro than 1.7B@4k (OOM guard; smoke will tune)
export LOGPROB_MICRO=2
export LR=1e-6
export CRITIC_LR=1e-5              # PPO/gae_turn arms only
export KL_LOSS_COEF=0.001
export GAMMA=1.0                   # protocol default (HCAPO/GiGPO papers use 0.95 — flagged for HCAPO)
export LAM=1.0
export TOTAL_STEPS=150             # feasibility budget (not the full 500); revisit after first read
export VAL_FREQ=25                 # short val cadence
export SAVE_FREQ=25
export VAL_BATCH=256               # SHORT validation (was 2048) to save time — user request 2026-07-27
export TOPK=3                      # retrieval top-k (single tool: search only)

# Separate W&B project to keep feasibility history clean (user request 2026-07-27).
export WANDB_PROJECT=ca-rung3-4b-feasibility
