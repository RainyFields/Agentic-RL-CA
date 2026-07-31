# HCAPO Wave-3 (SearchQA, Qwen3-4B, non-thinking@2048) — pre-registration + amendment

Runs: `hcapo_qwen3-4b_4turn_nothink2k_s0` (arm a, omega=1.0) and
`hcapo_w05_qwen3-4b_4turn_nothink2k_s0` (arm b1, omega=0.5). 500 steps, seed 0.
Comparators: token_grpo / turn_ppo_b0 at the same protocol. Floor 0.278.

## Registered BEFORE launch (2026-07-27)
Hypothesis: HCAPO's 1.7B SearchQA collapse (0.357 @ clip 0.754) was truncation-driven;
at non-thinking@2048 clipping should be ~0 and HCAPO should be rescued.

| outcome @500 (arm a) | reading |
|---|---|
| >= ~0.43 (near GRPO) | truncation was the whole story |
| ~0.38-0.42 (turn-PPO band) | partially rescued; truncation was *a* cause not *the* cause |
| <= ~0.36 with clip ~0 | truncation was NOT the cause; method intrinsically weak |
| any value with clip > 0.3 | premise broken -> inconclusive |

Abort: at step >= 250, stop iff val < 0.30 AND clip > 0.3.
Also registered: the rho readout is a result in its own right; (b1) is interpreted only
relative to (a) -- b1 > a supports the aggressiveness hypothesis, a ~ b1 says omega is not
the lever.

## AMENDMENT recorded 2026-07-28 at step ~150, BEFORE any final (step-500) value existed
Observed: both arms reach healthy EM (~0.39) WHILE clipping rises (a: 0.223, b1: 0.305).
This falls in a quadrant the table did not anticipate -- truncation present AND learning
well -- so the "clip > 0.3 -> inconclusive" row is mis-specified. That row assumed
truncation would co-occur with collapse (as it did at 1.7B).

Amended reading for the high-EM / high-clip quadrant:
  HCAPO buys its performance by SPENDING RESPONSE LENGTH, and hits caps as a side effect.
  The 1.7B collapse was the same mechanism meeting a tighter effective budget (thinking
  already consumed ~476 tokens/turn, so the runaway hit the cap far sooner).
  => clip is a MEDIATOR of HCAPO's behaviour, not a confound to be controlled away.

Supporting mechanism (measured, not assumed):
- rho is negative-side only (max 0.9986, frac_at_hi = 0.000): the hindsight prompt makes the
  taken action LESS likely, i.e. it perturbs the input off-distribution rather than informing.
- rho RISES toward 1.0 over training (0.909 -> 0.959/0.961) exactly as resp_len explodes
  (89 -> 487/736).
- mean_logratio is divided by ntok, so emitting more tokens dilutes the (negative) log-ratio
  toward 0, raising rho, raising Q^H, raising the advantage => a STRUCTURAL VERBOSITY
  INCENTIVE, independent of omega.
- omega is refuted as the lever: halving it did not reduce the runaway (b1 clipped MORE:
  0.305 vs 0.223) -- though at n=1 seed the a/b gap itself should not be over-read.

Open (deferred to wave b2, per user 2026-07-28): per-sample corr(rho, ntok) within a batch
is the decisive test of the causal claim; aggregate co-movement is consistent with it but
does not rule out the policy merely adapting to the off-distribution hindsight prompt.
b2 should therefore target the /ntok normalization (make rho length-invariant), NOT the
rho clip width as originally sketched.

## CORRECTION recorded 2026-07-28 at step 200 (still before any step-500 value)
The step-150 amendment above stated "omega is refuted as the lever". **That was premature and is
now contradicted by data.** Trajectories:

| step | A (omega=1.0) EM / clip / len | B (omega=0.5) EM / clip / len |
|---|---|---|
| 100 | 0.394 / 0.172 / 487  | 0.373 / 0.251 / 654 |
| 150 | 0.392 / 0.223 / 551  | 0.394 / 0.305 / 736 |
| 200 | 0.389 / 0.707 / 1504 | 0.412 / 0.331 / 742 |

At steps 100-150 b1 inflated MORE (hence the premature refutation); by step 200 it clearly inflates
LESS (742 vs 1504), clips less (0.331 vs 0.707) and scores HIGHER (0.412 vs 0.389). So omega DOES
restrain the runaway -- the arms simply needed ~200 steps to separate.

Also corrected: the step-150 note called the runaway "decelerating". It was not; arm A
re-accelerated 551 -> 1504 between steps 150 and 200, reaching clip 0.707 -- i.e. the 1.7B collapse
signature (clip 0.754) DOES reproduce at 4B non-thinking@2048, but WITHOUT the EM collapse
(0.389 here vs 0.357 at 1.7B).

Implication for b2: the length-invariance fix remains the primary variable, but omega is no longer
excluded as a contributing lever. Keep b2 at omega=1.0 (one variable at a time) so its effect is
attributable; if b2 fixes the runaway at omega=1.0, that is the stronger result.

## Prediction CONFIRMED, recorded 2026-07-28 at step 250 (before finals)
Predicted at step 200: "arm A at 1504/2048 is near saturation, so its length cannot grow much
further -- steps 200->500 will show whether EM degrades once verbosity stops paying."

Observed at step 250: arm A EM turned DOWN for the first time (0.394@100 -> 0.392@150 -> 0.389@200
-> 0.363@250) as resp_len saturated (1504 -> 1642 against the 2048 cap) and clip reached 0.784 --
which EXCEEDS the 1.7B collapse level (0.754). Arm B (omega=0.5) held at 0.406 / clip 0.366 /
len 895.

Two conclusions this supports, both before any step-500 value:
1. The length runaway is INTRINSIC to HCAPO, not an artifact of thinking mode: it reproduces at 4B
   non-thinking@2048 and reaches a HIGHER clip than the 1.7B thinking run did.
2. HCAPO's EM is BOUGHT with response length. While verbosity could still grow, EM held ~0.39;
   the moment it saturated against the cap, EM fell. This is the third row of the amended
   interpretation ("apparent competence is a length artifact") arriving as a within-run dynamic
   rather than a cross-arm comparison.
Abort rule evaluated at step 250 for both arms and correctly did NOT fire (EM > 0.30 in both).

## RETRACTION recorded 2026-07-29 at step 300 (before finals)
The step-250 entry above ("Prediction CONFIRMED ... HCAPO's EM is BOUGHT with response length")
is **RETRACTED**. It was based on a single val point.

arm A EM: 0.394@100, 0.392@150, 0.389@200, 0.363@250, 0.397@300.
The 0.363 was an isolated dip, not the onset of decline. EM is FLAT at ~0.39 (+/-0.02) across
steps 100-300.

Decisively: arm A's length has now PLATEAUED (1504@200 -> 1642@250 -> 1635@300; clip 0.707 ->
0.784 -> 0.773). So verbosity saturated against the 2048 cap and EM did NOT fall with it. That is
the opposite of the prediction. On current evidence HCAPO's ~0.39 is NOT purchased by ongoing
verbosity growth -- it survives saturation.

What still stands (measured, not inferred):
- the verbosity incentive itself (rho rises toward 1.0 as length grows; rho strictly < 1;
  mean_logratio divided by ntok while the perturbation is front-loaded);
- the runaway is intrinsic, reproducing at 4B non-thinking and reaching clip 0.784 > the 1.7B
  thinking run's 0.754;
- omega delays but does not prevent it (B is tracking A's path with a lag: 895@250 -> 1232@300).
What does NOT stand: that the runaway costs accuracy. So far it appears WASTEFUL (4-8x the tokens
of GRPO for lower EM) rather than DESTRUCTIVE.

Process note: this is the third single-point over-read in this experiment (step-150 "decelerating",
step-150 "omega refuted", step-250 "EM declining"). Adopting a rule for the remainder: do not call
a trend on fewer than THREE consecutive val points.

## MAJOR CORRECTION 2026-07-29 — OUR IMPLEMENTATION DOES NOT MATCH THE PAPER
Checked against the source: "Hindsight Credit Assignment for Long-Horizon LLM Agents",
Tan et al., arXiv:2603.08754 (Mar 2026).

PAPER Eq.(6)-(7):
  pi_hind(a_t) = exp( 1/(T_temp * |a_t|) * SUM_j log pi_theta(y_j | y_<j, s_t, s_final) )
  rho_t = clip( pi_hind(a_t) / PI_BAR_hind , C_min, C_max ),
          PI_BAR_hind = (1/T) SUM_k pi_hind(a_k)          <-- INTRA-TRAJECTORY MEAN over turns
  Paper's words: "the intra-trajectory normalization over pi_bar_hind provides a meaningful local
  reference, akin to group-normalization across actions within the same episode." rho is CENTERED
  AT 1.0 by construction: pivotal actions > 1, redundant actions < 1.
  Hyperparameters confirmed identical to ours: omega=1.0, T_temp=5.0, clip [0.8,1.2], alpha=0.5,
  gamma=0.95.

OUR gigpo/core_hcapo.py:
  mean_logratio = ((hindsight_log_probs - policy_log_probs)*m).sum(-1)/ntok
  rho = exp(mean_logratio/t_temp).clamp(0.8,1.2)
     == pi_hind(a_t) / PI_POLICY(a_t)                      <-- ON-POLICY prob of the same action

The NUMERATOR matches the paper. The DENOMINATOR does not.

This explains every anomaly we measured:
- rho strictly < 1 (max 0.9986, frac_at_hi=0.000): pi_hind/pi_policy measures the off-distribution
  penalty of injecting s_final into the prompt. The paper's ratio is between LIKE quantities
  (hindsight probs across turns), so that offset cancels.
- clip [0.8,1.2] never using its upper half, 10% pinned at the floor: correct for a ratio centered
  at 1.0 (the paper's), mis-specified for ours.
- the verbosity incentive: the paper's denominator is ALSO per-token normalized, so length effects
  largely cancel; ours has no counterpart, leaving the /ntok dilution exposed.
- HCAPO underperforming GRPO here, vs the paper's +13.8% over GRPO on ALFWorld / +7.7% WebShop.

CONSEQUENCES
1. RETRACT "HCAPO has a structural verbosity incentive". Correct statement: OUR PORT of HCAPO has an
   incorrect rho denominator, which creates a verbosity incentive. The measured runaway is real but
   is a property of the implementation, not the method.
2. The wave-3 A/B (omega=1.0 vs 0.5) characterises the buggy variant. Its EM numbers must NOT be
   reported as "HCAPO" in the cross-environment report.
3. The 1.7B SearchQA collapse (0.357 @ clip 0.754) and the AlfWorld 0.791 are ALSO from this
   implementation and are likewise not evidence about the published method.
4. b2 as designed (hindsight_window / length-invariant rho) is SUPERSEDED: it was a fix for a
   symptom of the bug. The correct fix is to implement Eq.(7) as written -- divide by the
   intra-trajectory mean of pi_hind.

Caveat: the paper was read via an automated fetch/summarisation of the arXiv HTML; Eq.(7) was
re-queried and quoted verbatim twice, but a human should confirm against the PDF before this
correction is published.

## FALSIFICATION recorded 2026-07-30 at step 200 of the PAPER-CORRECT run
The implementation note's structural claim ("the corrected form has no cross-rollout length
gradient and should not explode") is FALSIFIED behaviourally. The paper-correct run (Eq. 6-7,
intra-trajectory-mean denominator) ignited the same runaway on schedule:
  len: 147@100 -> 221@150 -> 249@160 -> 412@170 -> 611@180 -> 985@190 -> 1232@200
  clip: 0.022@100 -> 0.032@150 -> 0.505@200         EM: 0.401@100 -> 0.417@150 -> 0.408@200 (flat)

Mechanism (revised, from the rho instrumentation): rho std COLLAPSES as length explodes
(0.135@100 -> 0.068@150 -> 0.030@200, clip fractions -> 0). The common root of BOTH runaways is
the PER-TOKEN-MEAN scoring in Eq. 6: averaging log-probs over |a_t| means a turn's score is pulled
toward generic fluency as it lengthens. Under the paper's intra-trajectory normalisation this
does cancel UNIFORM inflation, but not the transient: a below-mean turn raises its rho by
lengthening (diluting its distinctive early tokens), so a length gradient exists until all turns
homogenise (rho ~ 1 everywhere, sigma -> 0), at which point the hindsight term is inert and HCAPO
degenerates to GRPO + a positional discount gamma^(T-1-k).

What stands: the legacy denominator was still a real bug (rho strictly <1, off-distribution
measurement, upper clip dead); the fix restored the paper's semantics (rho centred, two-sided,
discriminating for ~150 steps). What changes: the runaway is NOT specific to our bug -- on this
task (SearchQA, T<=4 turns, s_final = a long retrieved-docs block) the PUBLISHED formulation also
carries a length instability. Caveats: single seed; the paper's own tasks (ALFWorld/WebShop,
short action-style turns, 512-token caps, Qwen2.5-7B) may never enter the regime; and our
s_final injection (full <information> block) is a large prompt perturbation. The within-batch
corr(rho, |a_t|) probe -- now clearly the decisive instrument -- remains queued.
Run continues to 500 per the abort rule (EM 0.408 >> 0.30).

## WAVE-4 pre-registration, 2026-07-31 (before any lift-arm data): the 2x2
User-requested side-by-side of the two m_t formulations across injection sizes. New score option
rho_score="lift": m_t = per-token-mean of [log pi(y|..,s_final) - log pi(y|..)], Eq.7
normalisation unchanged. Grid (all Qwen3-4B non-thinking@2048, 500 steps, s0):
                      s_final=last_obs           s_final=final_answer
  score=hind (Eq.6)   hcapo_paper (running/378)  hcapo_ans (running/235)
  score=lift          hcapo_lift  (NEW)          hcapo_lift_ans (NEW)
Predictions (from the offline diagnostic's rho_dm: Spearman 0.95-0.97 vs Eq.6 score, corr(lift,len)
= +0.43 healthy / +0.77 post-ignition, leakage unchanged):
  P-L1: hcapo_lift re-ignites the drift at ~step 160+-25, matching hcapo_paper's trajectory.
  P-L2: hcapo_lift_ans tracks hcapo_ans (damped, plateau ~450-500 tokens, clip ~0.1).
  P-L3: EM differences between lift and hind arms at matched s_final stay within noise (~+-0.02).
  If P-L1 fails (no ignition on last_obs+lift), the static-ranking -> training-dynamics inference
  is wrong and the lift genuinely changes the feedback loop -- the most informative outcome.
