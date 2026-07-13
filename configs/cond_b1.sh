#!/usr/bin/env bash
# RQ3 arm: B1 = turn-level PPO + privileged answer-exposure step reward.
# First-hit-only bonus W_STEP when normalize_answer(gold) substring-matches this turn's
# retrieved <information> block (subem_check). NOT a true progress measure — a privileged,
# verifiable shaping signal (plan §Thread B taxonomy).
export CONDITION=b1
export ADV_ESTIMATOR=gae_turn
export W_STEP="${W_STEP:-0.2}"     # 0.5 = deprioritized ablation
export EXTRA_OVERRIDES=(
  +env.search.step_reward_mode=b1
  +env.search.step_reward_w="$W_STEP"
)
