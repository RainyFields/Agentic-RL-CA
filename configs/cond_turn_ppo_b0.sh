#!/usr/bin/env bash
# RQ1/RQ3 arm: turn-level PPO (`gae_turn`) with SPARSE outcome reward = B0.
# One scalar V per turn, GAE recursion over turns, advantage broadcast to the turn's tokens.
export CONDITION=turn_ppo_b0
export ADV_ESTIMATOR=gae_turn
export EXTRA_OVERRIDES=()
