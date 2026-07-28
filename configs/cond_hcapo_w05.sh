#!/usr/bin/env bash
# Wave-3 arm (b1): HCAPO with omega=0.5 — the ONLY change from cond_hcapo.sh.
# Tests the "HCAPO is too aggressive" hypothesis: the micro (hindsight) term is added at half
# weight to the macro GRPO term, shrinking the summed advantage magnitude. Evidence motivating it:
# at 1.7B HCAPO inflated response length ~2-3x and episode length ~40% vs every other method
# (AlfWorld), and blew through the response cap on SearchQA thinking@2048 (clip 0.754).
export CONDITION=hcapo_w05
export ADV_ESTIMATOR=hcapo
export EXTRA_OVERRIDES=(
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  +algorithm.hcapo.omega=0.5
  +algorithm.hcapo.t_temp=5.0
  +algorithm.hcapo.clip_lo=0.8
  +algorithm.hcapo.clip_hi=1.2
  +algorithm.hcapo.temporal_alpha=0.5
)
