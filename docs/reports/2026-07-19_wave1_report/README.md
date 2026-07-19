# Wave-1 Results Report — When does intermediate reward actually help? (2026-07-19)

`report.pdf` — the full Wave-1/2 results report (all 7 credit-assignment arms, 17 seeded 500-step
runs complete). Companion to the 2026-07-16 research proposal (`../2026-07-16_research_proposal/`),
which framed the study; this report delivers the finals, the five findings, and the filled survey
checklist.

## The five findings (details + figures in the PDF)

1. **Outcome-only group-relative CA wins and is cheaper** — GiGPO 0.397 / token-GRPO 0.394 final
   val macro-EM beat every critic and step-reward arm by 4–6 pts, with no critic network (and
   ~15–20% less wall-clock than PPO).
2. **Privileged content adds nothing over its shuffle** (RQ3, 3 seeds/arm) — B1-shuffle 0.353 ≥
   B1 0.347 > B0 0.332 ⇒ a reward-density effect, not progress-content supervision.
3. **Truncation instability tracks aggressive per-turn credit** — HCAPO collapses (clip→0.75, no EM
   gain); all critic arms drift; group-relative-and-flat arms stay clean.
4. **The critic assigns credit uncorrelated with true progress** (F8a) — pooled Spearman ≈ 0
   (0/3 gate ckpts pass) ⇒ the λ-bootstrapping sweep is dropped as a pre-registered negative.
5. **Horizon lifts the baseline, content still does not help** (RQ4) — 8-turn B0 0.368 > 4-turn
   B0 0.332, but 8-turn B1 0.343 < 8-turn B0 0.368.

## Rebuild

```bash
cd assets
python3 extract_metrics.py     # launch logs -> csv/<run>.csv  (17 runs, all seeds)
python3 build_figures.py       # -> figs/fig_{learning_curves,final_bars,rq3_density,rq4_horizon,tripwire,reward_turns}.{png,pdf} + figs/finals.csv
python3 build_f8a.py           # -> figs/fig_f8a_credit_alignment.{png,pdf}  (needs outputs/diag/, repo on path)
python3 build_taxonomy.py      # -> figs/fig_taxonomy.{png,pdf}
cd .. && tectonic report.tex   # -> report.pdf
```

- `assets/extract_metrics.py` — parses per-step metrics from `~/xiaoxuan/worker_logs/launches/`
  (last-occurrence-per-step; pre-hang + post-hang-relaunch logs merged chronologically per run).
- `assets/build_figures.py` — the 6 data figures + `finals.csv` (the per-seed @500 bar source).
- `assets/build_f8a.py` — the credit-alignment figure; reuses `credit_assignment.lambda_gate` +
  `diagnostic.build_pairs` so the plotted Spearman/CI are the exact gate numbers
  (`outputs/diag/lambda_gate.json`).
- `assets/build_taxonomy.py` — the F1 taxonomy schematic.
- Per-figure data map: `assets/figs/README.md`.

## Provenance

- Metric: val_2048 macro-EM (fixed 2,048-q stratified subset, greedy) at the FINAL @500 checkpoint
  (the pre-registered primary comparison). Untrained baseline 0.2246.
- Consolidated finals also in `../../../analysis/wave1_results.csv` (+ `_README.md`); per-run launch
  logs + GPU-hours ledger in `~/xiaoxuan/worker_logs/`; incident/decision trail in
  `../../decision_log.md`; pre-registration in `../../plan.md`; survey checklist in
  `../../CREDIT_ASSIGNMENT_CHECKLIST.md`.
- Full-set eval (51,713 q) currently completed for token-GRPO s0 only
  (`eval_full/wave0_grpo_s0_final/paper_table.json`, archived to HDFS); the remaining arms'
  full-set evals are the next Wave-4 GPU task.
- The annotated appendix trajectory is a trained token-GRPO full-set-eval rollout
  (traj `a1c80bb0`, HotpotQA, EM=1).

Data snapshot: 2026-07-19 ~12:50 PDT (all 17 training runs complete; fleet idle).
