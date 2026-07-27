#!/usr/bin/env bash
# Rung-4 arm: PRM (per-step dP_t of max-cumulative clipped native score) through
# the turn-RTG estimator. Return-equivalent to the ORM arm on every trajectory;
# the arms differ only in the temporal placement of reward.
export CONDITION=prm_rtg
export ADV_ESTIMATOR=turn_rtg
export EXTRA_OVERRIDES=(
  env.sciworld.reward_mode=prm
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
)
