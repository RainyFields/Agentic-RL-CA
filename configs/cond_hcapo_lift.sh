#!/usr/bin/env bash
# Wave-4 2x2 (user 2026-07-31): rho scored on the per-token-mean LIFT (m_hind - m_no), Eq.7
# normalisation on top; s_final = last_obs. Completes the factorial with the two running Eq.6-score
# arms. Pre-registered from the offline diagnostic (rho_dm): the lift co-ranks with Eq.6 at
# Spearman 0.95-0.97 and carries the length channel (+0.43/+0.77), so prediction = same dynamics
# as the matching Eq.6 arm (last_obs: re-ignition ~step 160+-25; final_answer: damped plateau).
# A non-drift on last_obs would falsify the static->dynamic inference, which is the live question.
export CONDITION=hcapo_lift
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
  +algorithm.hcapo.rho_score=lift
  
)
