#!/usr/bin/env bash
# HCAPO paper-correct core (rho = pi_hind / intra-traj mean) with s_final = the AGENT'S FINAL
# ANSWER instead of the last retrieved-docs block (user-requested 2026-07-30).
# Pre-registered expectations: (a) the injection shrinks from ~hundreds of doc tokens to a
# few answer tokens, removing most of the off-distribution perturbation whose dilution created
# the length channel (diagnostic: corr(lift,len)=+0.43 pre-drift under last_obs) -> if the drift
# still ignites, the lever is not the injection size; (b) answer-turn leakage should INCREASE
# (s_final literally entails the final turn's text) -> watch frac(top-credit=final turn).
export CONDITION=hcapo_ans
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
  +algorithm.hcapo.s_final_source=final_answer
)
