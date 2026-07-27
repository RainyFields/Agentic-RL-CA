#!/usr/bin/env bash
# Rung-4 arm: ORM (single terminal reward P_T) through the turn-RTG estimator.
# With terminal-only reward the estimator degenerates to vanilla GRPO while all
# group members are alive — identical code path to the PRM arm; only reward
# placement differs (design doc §algorithm).
export CONDITION=orm_grpo
export ADV_ESTIMATOR=turn_rtg
export EXTRA_OVERRIDES=(
  env.sciworld.reward_mode=orm
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
)
