#!/usr/bin/env bash
# Rung-4 ScienceWorld protocol — Qwen3-4B-Instruct-2507, no cold start, 25-task roster,
# per-task caps 14-188 (task_roster.py), three-tier K=20 truncation inside 16k.
# Spec frozen in docs/reports/2026-07-28_sciworld_orm_prm_design/design.md (A12/A13).

export MAX_STEPS=188               # global turn-loop bound = max roster cap; per-task caps bind earlier
export HISTORY_LENGTH=999          # sciworld manager uses its own 3-tier builder; keep >= MAX_STEPS
export MAX_PROMPT_LENGTH=16384
export MAX_RESPONSE_LENGTH=256     # non-thinking Thought+Action; clip rate is a smoke metric
export TRUNCATION=left             # token-side safety valve ONLY — manager keeps prompts in budget
export ENABLE_THINKING=False

export TRAIN_BATCH=64              # groups per step
export GROUP_SIZE=8                # G -> 512 trajectories/step
export PPO_MINI_BATCH=512
export MICRO_BSZ=1                 # 4B @ 16k (P2 lineage: 53GB/GPU at 16k with MICRO_BSZ=1)
export LOGPROB_MICRO=2
export LR=1e-6
export CRITIC_LR=1e-5              # unused (both arms critic-free); kept for harness symmetry
export KL_LOSS_COEF=0.001
export GAMMA=1.0
export LAM=1.0
export TOTAL_STEPS=150
export VAL_FREQ=25
export SAVE_FREQ=25
export VAL_BATCH=200               # fixed pre-drawn val set: 8 dev variations x 25 tasks

export WANDB_PROJECT=ca-rung4-4b
