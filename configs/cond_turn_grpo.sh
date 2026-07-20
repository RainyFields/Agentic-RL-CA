#!/usr/bin/env bash
# Wave-2 arm: turn-GRPO — critic-free group-relative advantage at TURN granularity.
# = GiGPO's anchor-state step component ONLY (no episode term): per-turn reward-to-go
# normalized within anchor-state groups; singleton groups get advantage 0. Ablation
# ladder: token_grpo (episode term) / turn_grpo (step term) / gigpo (both). gigpo.*
# knobs mirror cond_gigpo.sh so the shared step term is bit-comparable.
export CONDITION=turn_grpo
export ADV_ESTIMATOR=turn_grpo
export EXTRA_OVERRIDES=(
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  algorithm.gigpo.mode=mean_std_norm
  algorithm.gigpo.enable_similarity=True
  algorithm.gigpo.similarity_thresh=0.9
)
