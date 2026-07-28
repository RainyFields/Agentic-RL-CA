# HCAPO wave b2 — length-invariant rho  (designed 2026-07-28, user-approved: 2 runs, launch after wave-3 finals)

## Diagnosis this is built on (measured in wave-3)
- rho is strictly < 1 (max 0.9986, frac_at_hi = 0.000): the hindsight prompt makes the taken
  action LESS likely => it perturbs the input off-distribution rather than informing.
- rho RISES 0.909 -> 0.959/0.961 while resp_len explodes 89 -> 487/736; clip 0.000 -> 0.223/0.305.
- Mechanism: the hindsight prefix perturbs the PROMPT, so its effect on the response is
  FRONT-LOADED (large on early tokens, decaying as the response's own context dominates).
  The numerator is therefore ~constant while `/ntok` grows with length:
      mean_logratio ~ C^-/ntok -> 0^-   =>   rho -> 1^-   =>   Q^H up, advantage up.
  i.e. HCAPO structurally REWARDS VERBOSITY.
- omega is refuted as the lever (halving it did not reduce the runaway; b1 clipped MORE).

## The change (one variable)
Restrict the hindsight log-ratio to the first K response tokens, so the denominator is bounded
by K and rho loses its structural dependence on total length.

```python
# core_hcapo.compute_hcapo_advantage, replacing the mean_logratio line.
# hindsight_window=None keeps EXACTLY the current behaviour (default; no-op for existing runs).
if hindsight_window:
    m_k = m.clone()
    m_k[:, int(hindsight_window):] = 0.0
    n_k = m_k.sum(-1).clamp(min=1.0)
    mean_logratio = ((hindsight_log_probs - policy_log_probs) * m_k).sum(-1) / n_k
else:
    mean_logratio = ((hindsight_log_probs - policy_log_probs) * m).sum(-1) / ntok
```
K = 32 (responses start ~89 tokens, so K must sit well below that to give invariance from step 0).
Config: `+algorithm.hcapo.hindsight_window=32`.
Everything else identical to wave-3 arm (a): omega=1.0, t_temp=5.0, clip [0.8,1.2],
temporal_alpha=0.5, 4B, non-thinking@2048, 500 steps, seed 0.

## Runs (2)
1. `hcapo_k32`   — the fix, hindsight_window=32.
2. `hcapo_stock_instr` — stock HCAPO (window=None) re-run WITH the new diagnostics, so the
   corr(rho, ntok) contrast is measured on both sides. The wave-3 arms lack this logging.

## New instrumentation (both runs)
1. per-batch `corr(rho, ntok)` -- THE decisive causal test. Predict: >>0 stock, ~0 for k32.
2. per-position log-ratio profile (mean over token bins 0-7, 8-31, 32-127, 128+) -- verifies the
   front-loading premise that motivates K. If flat, the diagnosis is WRONG and K-windowing will
   not help; we want to know at step 25, not step 500.
3. existing rho stats + clip_ratio + response_length/mean.

## Pre-registered readings
| observation | reading |
|---|---|
| corr(rho,ntok) ~0 in k32, >>0 in stock; resp_len ~100-200; clip ~0 | mechanism confirmed AND fixed |
| EM >= wave-3 arm (a) (~0.39) with length controlled | HCAPO's performance is real; runaway was an unnecessary side effect (genuine improvement) |
| EM DROPS once length is controlled | HCAPO's EM was BOUGHT with verbosity -- its apparent competence is a length artifact |
| length still runs away | diagnosis wrong; look at temporal smoothing or the group-norm interaction |

## Sequencing
Do NOT edit core_hcapo.py while the wave-3 HCAPO runs are live -- they read it from the shared
checkout and a crash-resume would pick up the edit mid-run. Implement after they hit step 500.
K may be revised once the wave-3 final length distribution is known.
