#!/usr/bin/env bash
# RQ2 arm: token-level GRPO (episode-level group baseline), critic-free, KL in loss.
export CONDITION=token_grpo
export ADV_ESTIMATOR=grpo
export EXTRA_OVERRIDES=(
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
)
