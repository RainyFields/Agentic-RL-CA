#!/usr/bin/env bash
# RQ2 arm: HCAPO (hindsight credit assignment, value-free). Constants locked from
# arXiv 2603.08754 (docs/methods_note.md): omega=1.0, T_temp=5.0, rho-clip [0.8,1.2],
# temporal smoothing alpha=0.5. Search-QA table: G=5, mini-batch 512, beta_KL=0.001
# (already the matched protocol values).
export CONDITION=hcapo
export ADV_ESTIMATOR=hcapo
export EXTRA_OVERRIDES=(
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  +algorithm.hcapo.omega=1.0
  +algorithm.hcapo.t_temp=5.0
  +algorithm.hcapo.clip_lo=0.8
  +algorithm.hcapo.clip_hi=1.2
  +algorithm.hcapo.temporal_alpha=0.5
)
