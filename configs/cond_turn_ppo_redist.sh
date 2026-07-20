#!/usr/bin/env bash
# Wave-2 arm: step-reward PPO with RANDOM redistribution — the trajectory's terminal
# outcome R is redistributed across ALL its turns via per-trajectory Dirichlet(1,..,1)
# weights (r_t = w_t*R, sum = R; terminal keeps only its share). Driver-side, seeded,
# BEFORE compute_advantage; gae_turn consumes the redistributed per-turn rewards.
# NOT B1-based: no env-side step_reward_mode. Control: does turn-PPO care WHERE dense
# reward mass sits, given total mass and the critic?
export CONDITION=turn_ppo_redist
export ADV_ESTIMATOR=gae_turn
export EXTRA_OVERRIDES=(
  +algorithm.reward_redistribute=True
  +algorithm.reward_redistribute_seed="${REDIST_SEED:-1234}"
)
