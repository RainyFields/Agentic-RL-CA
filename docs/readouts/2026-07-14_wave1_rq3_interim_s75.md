# Wave-1 RQ3 interim readout

**PARTIAL — missing/queued runs: b1_s0** (per-seed rules still apply to present seeds; no pooling)

Common val steps: [0, 25, 50, 75]   (deltas at step 75)

| run | s0 | s25 | s50 | s75 | AUC-so-far |
|---|---|---|---|---|---|
| b0_s0 | 0.234 | 0.281 | 0.299 | 0.312 | 0.284 |
| b0_s1 | 0.234 | 0.274 | 0.303 | 0.312 | 0.283 |
| b1_s1 | 0.234 | 0.284 | 0.308 | 0.315 | 0.289 |
| b1sh_s0 | 0.234 | 0.294 | 0.316 | 0.304 | 0.293 |
| b1sh_s1 | 0.234 | 0.297 | 0.303 | 0.310 | 0.291 |

## Per-seed deltas at step 75 (pre-declared form; seed range = the two rows)

- **s0**: shuffle_minus_b0_em=-0.008, shuffle_active_frac_latest=0.639
- **s1**: b1_minus_b0_em=+0.003, b1_minus_b0_auc=+0.006, b1_minus_shuffle_em=+0.005, shuffle_minus_b0_em=-0.002, shuffle_active_frac_latest=0.792

Interpretation rules (plan): B1 > shuffle ~= B0 => timing/content matters; B1 ~= shuffle > B0 => density effect; all ~= => no help at this horizon. shuffle_active_frac MUST accompany any B1-vs-shuffle claim.
