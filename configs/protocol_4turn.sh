#!/usr/bin/env bash
# Matched 4-turn protocol (Search-R1 setting) — shared by ALL arms in the main table.
# Horizon values live HERE and only here (plan §Horizon strategy: no literal 4s in code).
# 8-turn stress test: source protocol_8turn.sh instead (same variable names).

export MAX_STEPS=4                 # env.max_steps  (a "turn" = one <think>+action + env response)
export HISTORY_LENGTH=4            # >= MAX_STEPS => full-history protocol, nothing dropped
export MAX_PROMPT_LENGTH=4096
export MAX_RESPONSE_LENGTH=512     # Search-R1 uses 500/turn; verl-agent stock search example 512
export TRUNCATION=left             # safety valve ONLY — any nonzero truncation rate is a bug

# Matched optimization budget (upstream run_search.sh + HCAPO paper Search-QA table:
# G=5, mini-batch 512, beta_KL=0.001; identical data budget for every arm incl. PPO arms).
export TRAIN_BATCH=256             # prompts per step
export GROUP_SIZE=5                # env.rollout.n  -> 1280 trajectories/step for all arms
export PPO_MINI_BATCH=512
export MICRO_BSZ=16                # per-GPU micro batch (actor); tune once at toy gate
export LR=1e-6
export CRITIC_LR=1e-5              # PPO arms only (critic auto-on for gae/gae_turn)
export KL_LOSS_COEF=0.001
export GAMMA=1.0                   # Search-R1 protocol (decision_log 2026-07-14; HCAPO/GiGPO papers use 0.95 — flag at Wave-0 review)
export LAM=1.0
export TOTAL_STEPS=500             # fixed budget, matches tr1 reference; confirm with user before Wave 1
export VAL_FREQ=25                 # test_freq on fixed val_2048 (greedy)
export SAVE_FREQ=25

export VAL_BATCH=2048              # fixed stratified val subsample (val_2048.parquet)
export TOPK=3                      # retrieval top-k (Search-R1 protocol)
