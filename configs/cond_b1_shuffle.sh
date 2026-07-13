#!/usr/bin/env bash
# RQ3 arm: B1-shuffle — THE key control. Same B1 rewards computed env-side, then each
# trajectory's (single, first-hit-only) positive step reward is reassigned uniformly at
# random to another eligible NON-TERMINAL search turn of the same trajectory, driver-side,
# seeded, before compute_advantage. Never lands on the terminal answer turn.
# Logs shuffle_active_frac (B1-positive trajectories with >=2 eligible search turns).
export CONDITION=b1_shuffle
export ADV_ESTIMATOR=gae_turn
export W_STEP="${W_STEP:-0.2}"
export EXTRA_OVERRIDES=(
  +env.search.step_reward_mode=b1
  +env.search.step_reward_w="$W_STEP"
  +algorithm.step_reward_shuffle=True
  +algorithm.step_reward_shuffle_seed="${SHUFFLE_SEED:-1234}"
)
