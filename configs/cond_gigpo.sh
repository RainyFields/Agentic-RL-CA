#!/usr/bin/env bash
# RQ2 arm: GiGPO (episode-level + anchor-state step-level group advantages), critic-free.
export CONDITION=gigpo
export ADV_ESTIMATOR=gigpo
export EXTRA_OVERRIDES=(
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  algorithm.gigpo.step_advantage_w=1.0
  algorithm.gigpo.mode=mean_std_norm
  algorithm.gigpo.enable_similarity=True
  algorithm.gigpo.similarity_thresh=0.9
)
