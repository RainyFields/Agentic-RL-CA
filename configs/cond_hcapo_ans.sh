#!/usr/bin/env bash
# HCAPO with s_final = the trajectory's FINAL ANSWER (user spec 2026-08-03), the search-task
# reading of the paper's terminal state. Constants from arXiv 2603.08754 / docs/methods_note.md
# (Search-QA table): omega=1.0, T_temp=5.0, rho-clip [0.8,1.2], temporal alpha=0.5,
# G=5, mini-batch 512, beta_KL=0.001 — the last three already come from the matched protocol.
# gamma: the HCAPO paper's Search-QA setting uses 0.95 for the hindsight step return; our
# protocol default is 1.0. HCAPO_GAMMA below sets the value used INSIDE the Q^H discount only
# (algorithm.gamma stays at the protocol value so every other arm is untouched).
export CONDITION=hcapo_ans
export ADV_ESTIMATOR=hcapo
export EXTRA_OVERRIDES=(
  # Paper-consistent discount for the hindsight step return Q^H = rho * gamma^(T-1-k) * R.
  # Our protocol's gamma=1.0 would flatten that term to rho*R (turn position stops mattering),
  # so HCAPO alone runs at the paper's 0.95. HCAPO is value-free: gamma feeds ONLY
  # compute_hcapo_advantage here (no GAE), so nothing else in the run changes.
  algorithm.gamma=0.95
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  +algorithm.hcapo.omega=1.0
  +algorithm.hcapo.t_temp=5.0
  +algorithm.hcapo.clip_lo=0.8
  +algorithm.hcapo.clip_hi=1.2
  +algorithm.hcapo.temporal_alpha=0.5
  +algorithm.hcapo.hindsight_source=answer
)
