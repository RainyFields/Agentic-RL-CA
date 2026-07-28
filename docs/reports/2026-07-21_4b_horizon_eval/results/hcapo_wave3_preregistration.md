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
