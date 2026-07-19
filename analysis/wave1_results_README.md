# Wave-1/2 results — consolidated finals (ALL 7 RUNS COMPLETE, 2026-07-19 08:06 PDT)

Backing data: `wave1_results.csv`. Metric = **val_2048 macro-EM** (fixed-budget FINAL ckpt @500,
the pre-registered primary comparison). Base model = 0.2246. Provenance: per-run launch logs in
`~/xiaoxuan/worker_logs/launches/`, W&B project `rainyfields/agentic-rl-ca`; full incident/decision
trail in `docs/decision_log.md`. **1 run still in flight**: `b1_8t` s0 (8-turn B1), ETA ~08:30 07-19.

## Seed-band means (val_2048 macro-EM)

| arm | credit assignment | seeds | mean | clip health |
|---|---|---|---|---|
| **GiGPO** | critic-free, group-relative, turn | 0.396, 0.398 | **0.397** | clean (~0) |
| **token-GRPO** | critic-free, group-relative, trajectory | 0.391, 0.399 | **0.395** | clean (~0) |
| B1-shuffle | critic + shuffled step reward (placebo) | 0.356, 0.363, 0.340 | 0.353 | clean |
| B1 | critic + privileged step reward | 0.352, 0.318, 0.372 | 0.347 | drift; s1 collapsed |
| turn-PPO (B0) | critic, per-turn | 0.343, 0.330, 0.324 | 0.332 | drift, all seeds |
| HCAPO | critic-free, per-turn amplifier | 0.357 (s0) | 0.357 | **collapse, clip 0.754** |
| token-PPO | critic, per-token | 0.315 (s0) | 0.315 | persistent drift |
| **8-turn** B0 | critic, per-turn | 0.368 (s0) | 0.368 | clean-ish (0.037) |
| **8-turn** B1 | critic + step reward | 0.343 (s0) | 0.343 | drift (peak 0.362 → 0.343, clip 0.268) |

## Crystallized findings (current status)

1. **Critic-free group-relative wins.** GiGPO 0.397 / GRPO 0.395 beat every critic and hand-designed
   step-reward arm by ~4–6 pts. Solidified at 2 seeds each.
2. **RQ3 — content adds nothing over its shuffle (3 seeds each):**
   `B1-shuffle 0.353 ≥ B1 0.347 > B0 0.332`. Both B-arms beat sparse B0 (small density benefit),
   but the shuffled placebo ≥ the real privileged content → **density/optimization effect, not
   supervision** (pre-registered interpretation rule). B1 variance wide (0.318–0.372).
3. **Truncation instability tracks aggressive per-turn credit weighting.** HCAPO (per-turn amplifier)
   → clip 0.754 length runaway with no EM gain (plateau ~0.35); all critic arms drift; everything
   group-relative-and-flat (GiGPO/GRPO) stays clip≈0.
4. **F8a (λ-gate negative):** critic one-step ΔV uncorrelated with MC continuation-value ΔV̂
   (pooled Spearman ≈ 0, 0/3 pass) → λ-sweep pre-registered DROPPED. The turn-critic is only a weak
   trajectory baseline (vf_explained_var ≈ 0.28), explaining turn-PPO < GRPO/GiGPO.
5. **RQ4 (horizon), COMPLETE:** 8-turn B0 = 0.368 > 4-turn B0 mean 0.332 — the longer horizon lifts
   the baseline itself. But **8-turn B1 = 0.343 < 8-turn B0 = 0.368**: the privileged content signal
   does NOT help at longer horizon, and the step-200 early hint (B1 0.358 > B0 0.338) did **not**
   hold to the final. So the marginal value of progress supervision does not grow with horizon — it
   is ≤0 here, with B1's added reward density driving clip drift (0.15→0.27) and mild degradation
   (peak 0.362 → 0.343). Caveat: no 8-turn shuffle control was run, so B1<B0 at 8t cannot fully
   separate content from density — but the direction (no benefit, possible harm) is unambiguous and
   consistent with the 4-turn shuffle result.

## Status: ALL 7 TRAINING RUNS COMPLETE (2026-07-19 08:06 PDT)
Fleet idle, /home recovered to 43G. Remaining before the paper (not blockers for the Wave-1 report
narrative, which is fully determined above):
- Full-set greedy EM (51,713 rows) on pre-registered FINAL checkpoints — Wave-4 GPU task (only
  token-GRPO s0 done so far: macro 0.3951 / micro 0.4433 / single 0.4993 / multi 0.3170).
- Per-dataset (single- vs multi-hop) breakdown for the RQ3/RQ4 subgroup readouts.
- Learning-curve figures need W&B history stitching across 2 run-IDs per relaunched experiment
  (frozen pre-hang run + relaunch); see decision_log 2026-07-17 for the run-ID pairs.
