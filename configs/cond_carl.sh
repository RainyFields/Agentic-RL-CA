#!/usr/bin/env bash
# RQ2 arm (second wave): CARL — criticality-aware tree rollouts, value-free
# (arXiv 2512.04949; constants/deltas locked in docs/methods_note.md 2026-07-14):
# N_total=16; N0=1 (CARL-Lite default, CARL_N0=8 for best "CARL"); resumes sample
# stochastically from pi_theta (temp-0 was only the paper's Eq. 5 preliminary study);
# non-critical edges DROPPED from D_upd (Eq. 13); no critic / no adv-norm / no KL.
#
# env.rollout.n does NOT set the group size here — the tree loop repeats internally.
# It only sizes the env worker pool: >= max(N0, N_total-N0) slots per prompt.
#
# Compute budget (plan 2b.6): CARL-Lite N=16 ~ GRPO n=8 in generated turns; launch
# wrappers pick TRAIN_BATCH per the pre-registered budget scale (token-fair variant
# uses CARL_NTOTAL=8-10 vs GRPO n=5). Both scales reported.
export CONDITION=carl
export ADV_ESTIMATOR=carl

CARL_N0="${CARL_N0:-1}"
CARL_NTOTAL="${CARL_NTOTAL:-16}"
_pool=$(( CARL_NTOTAL - CARL_N0 > CARL_N0 ? CARL_NTOTAL - CARL_N0 : CARL_N0 ))
export GROUP_SIZE="$_pool"

export EXTRA_OVERRIDES=(
  actor_rollout_ref.actor.use_kl_loss=False
  algorithm.carl.n0="$CARL_N0"
  algorithm.carl.n_total="$CARL_NTOTAL"
  algorithm.carl.drop_noncritical="${CARL_DROP_NONCRITICAL:-True}"
  algorithm.carl.include_root="${CARL_INCLUDE_ROOT:-False}"
  algorithm.carl.norm_adv=False
  algorithm.filter_groups.enable=False
)
