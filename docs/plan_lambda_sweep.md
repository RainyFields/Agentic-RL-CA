# APPROVED-GATED — λ-sweep for turn-level PPO (drafted + grilled 2026-07-16)

Status: **user-approved 2026-07-16 (7-question grill), GATED behind the F8a diagnostic.**
Queue placement (user): F8a-on-B0 diagnostic = slot queue **#4.5** (after 8-turn stress pair,
before hcapo/gigpo-s1/grpo-s1); if the unlock trigger (§2a) fires, λ stage 1 enters as queue
**#6, ahead of CARL**. No wrappers written until the gate passes. Per the MPU rule (plan.md)
the sweep is additive and individually droppable.

## 1. Motivation and framing

All PPO arms run GAE with **γ = λ = 1**, so advantages are pure Monte-Carlo returns minus a
learned baseline: A_t = G_t − V_φ(s_t). Preliminary Wave-1 results show both PPO arms trailing
the group-relative methods (B0 0.343/0.330, token-PPO ~0.315 vs GRPO 0.391, GiGPO ~0.40), with
`critic/vf_explained_var` ≈ 0.28 at run end — the critic is a weak baseline.

λ interpolates **how much intermediate credit is drawn from the learned value model vs the
observed outcome**:

- **λ = 1**: critic used *only* as a baseline; every turn sees the true final outcome
  (unbiased, high variance).
- **λ = 0**: A_t = r_t + V_φ(s_{t+1}) − V_φ(s_t) — credit flows entirely through one-step
  value differences. The critic's ΔV becomes an **implicit progress reward**, the learned
  counterpart of B1's explicit privileged signal.
- **0 < λ < 1**: geometric blend of the two.

This lands the sweep squarely in the project's reward-model story (RQ2/RQ3): the turn-level
critic IS an implicit step reward model, and λ is the knob that controls how much we trust it.
It also directly tests the leading hypothesis for the PPO–GRPO gap: *a learned per-state
baseline loses to a per-question Monte-Carlo baseline when returns are binary* — if that is
the whole story, no λ can close the gap; if variance is the binding constraint and the critic
is locally informative, some λ* < 1 should improve on λ = 1.

## 2. Pre-registered predictions (set before any curve is seen)

| observation | reading |
|---|---|
| some λ* < 1 beats λ=1 beyond the B0 seed range (|Δ| > 0.013) | variance was binding; critic is locally informative; PPO–GRPO gap partly closable without group sampling |
| monotone degradation as λ decreases | critic too weak to bootstrap through — baseline-quality explanation stands; "the implicit value model is the bottleneck" |
| flat within seed range | λ not a lever at this horizon (T ≤ 4 turns); gap must come from baseline/normalization, not credit propagation |
| λ<1 damps the late-run truncation drift | supports "MC advantage broadcast drives length instability"; λ<1 shortens the effective credit horizon on long turns |

Secondary mechanistic readout: the Phase-3b credit-alignment diagnostic (F8a) computes MC
continuation values V̂(s_t) by prefix resume; for the λ arms, compare the critic's own
one-step deltas (V_φ(s_{t+1}) − V_φ(s_t)) against ΔV̂_t. This measures directly whether the
implicit reward model is *correct*, independent of final EM.

## 2a. Unlock trigger (pre-registered gate, decided at grill Q1/Q2)

Run the Phase-3b credit-alignment diagnostic on the existing **B0-s0 checkpoints at steps
{150, 300, 500}** (K=8 continuations, pooled turn-pairs, bootstrap CIs — Phase 3b spec).
Compute the pooled Spearman correlation between the critic's own one-step deltas
V_φ(s_{t+1}) − V_φ(s_t) and the MC continuation deltas ΔV̂_t.
**Unlock the sweep iff Spearman > 0.2 with the 95% CI excluding 0 at ≥ 2 of the 3
checkpoints.** Otherwise the sweep is DROPPED and the paper reports the negative directly
("critic ΔV unaligned with true progress — bootstrapping through it has no basis").
Rationale: the diagnostic answers the observational half for ~5% of the sweep's cost; the
sweep is then the causal confirmation, not a fishing trip.

## 3. Design

- **Vehicle: turn-level PPO (B0), 4turn_think2k protocol** — the turn-level MDP has ≤ 4 credit
  steps, so a coarse grid is meaningful and each λ is interpretable. Everything else
  (data, seeds, batch, lengths, KL, penalty, eval cadence) identical to B0 — single-variable
  contrast, same as the rest of the benchmark.
- **Grid (staged; grill Q3/Q4 decisions):**
  - Stage 1 (4 workers; user expanded 2026-07-17): **λ = 0.5, 0.8, 0.9, 0.95**, seed 0. (0.9/0.95 added at user request for dose-response shape and comparability with the classic PPO regime — pre-registered expectation: both sit inside the B0 seed band at this ≤4-turn horizon, so they read as horizon-calibration points, not likely winners.) (0.9 rejected: with γ=1 and ≤4 turns
    the λ^l weights at 0.9 keep ≥0.73 of MC weight even at max distance — indistinguishable
    from λ=1 within the ±0.013 B0 seed band. 0.8 → final-step weight 0.51 = real
    bootstrapping; 0.5 → 0.13 = aggressive.) λ=1 s0/s1/s2 already exist (B0); λ=0 deferred.
  - **Interpretation unit: the B0 three-seed band, not B0-s0 alone.** A stage-1 λ point
    inside the band reads "no effect"; only outside-the-band results trigger stage 2.
  - Stage 2: second seed on the informative λ is **MANDATORY before any paper claim**
    (stage-1 single-seed results are directional only — the b1 seed pair differed by 0.034);
    optionally bracket a peak with λ=0 / λ=0.65.
- **Token-PPO λ-sweep: explicitly out of scope** (secondary tier at best). Token-level λ acts
  over thousands of tokens, so the interesting range is a different regime (λ ≈ 0.95–1) and
  the token-PPO arm is already the weakest; spend the slots on the interpretable turn-level
  contrast.
- **γ stays 1.0** (undiscounted finite-horizon objective matches EM; sweep exactly one thing).

## 4. Mechanics (launch, not refactor)

- `algorithm.lam` is already config-driven (all current launches pass `algorithm.lam=1.0`).
  Add `LAM_OVERRIDE` (default 1.0) to `scripts/run_condition.sh` in the same style as
  `MICRO_BSZ_OVERRIDE`, mapped to `algorithm.lam=$LAM_OVERRIDE`.
- **Standard GAE(λ) confirmed (grill Q6):** λ enters both the actor's advantages and the
  critic's regression targets; NO decoupled arm (actor-λ<1 / critic-λ=1) — that would be a
  nonstandard method claim. Interpretive ambiguity handled by logging vf_explained_var per λ
  and the F8a extension (critic ΔV vs ΔV̂ per λ). Document arms as "GAE(λ)".
- Wrappers `.arlca-b0-lam05-s0.sh`, `.arlca-b0-lam08-s0.sh` (COND=turn_ppo_b0, SEED=0,
  PROTOCOL=4turn_think2k, LAM_OVERRIDE=0.5/0.8); micro8 inherited via protocol.
- W&B names per convention: `turn_ppo_b0_lam05_qwen3-1.7b_4turn_think2k_s0`, etc.
- Unit check before launch: one-batch dry assertion that `algorithm.lam` propagates (grep the
  dumped hydra config in the log header), plus the existing 33-test suite.

## 5. Monitoring & tripwires (same fleet standard)

Standard digest metrics + specifically:
- `critic/vf_explained_var` (prediction: rises under λ<1 as targets get easier to fit);
- truncation clip_ratio tripwire (must-fall rule; watch damping prediction above);
- early-warning review at step 150 (RQ3-window style readout vs the three B0 seeds), no
  kill rule — fixed-budget FINAL stays the primary comparison per pre-registration.
- **STRICT no-intervention on truncation drift (grill Q7):** λ arms ride out any clip
  runaway to step 500, exactly like the B0/B1 precedent — the λ-vs-drift damping prediction
  (§2) requires untouched runs, and best-val is reported as secondary anyway.

## 6. Budget & scheduling

- Each run: 1 × 8×H100 worker, ~19 h wall-clock for 500 steps (B0 reference), retriever
  co-located. Stage 1 = 2 workers; stage 2 ≤ 2 more.
- Queue position (grill Q5, user-decided): **F8a diagnostic = queue #4.5** (after the 8-turn
  stress pair, before hcapo/gigpo-s1/grpo-s1); **unlocked λ stage 1 = queue #6, ahead of
  CARL** (launch-only vs CARL's engineering risk; keeps the F8a→sweep explanatory chain
  tight). Fires only on freed slots.
- GPU-hours logged in workers.tsv as usual (survey checklist F4/compute accounting).

## 7. Reporting

- Curve panel: val-2048 macro-EM for λ ∈ {0.5, 0.9, 1.0×3 seeds} + GRPO reference band
  (extends F2); bar of fixed-budget finals with seed ranges (extends F3).
- Mechanism panel: vf_explained_var over training per λ; clip_ratio per λ (tripwire).
- F8a extension: critic ΔV vs MC ΔV̂ correlation per λ arm.
- One paragraph in the paper: "how much should a multi-turn agent trust its implicit value
  model?" — with the interpretation grid of §2 filled in.

## 8. Risks

- λ<1 with a poor critic can *destabilize* early training (biased bootstrapping) — staged
  grid starts at 0.9/0.5 rather than 0; step-150 review catches pathology with ~70% of the
  budget still unspent (informational only, no kill rule).
- Critic-target coupling means λ changes two things at once (actor advantage + critic
  targets); acceptable — this is the standard GAE(λ) estimator, and the alternative
  (decoupled λ) would be a nonstandard method claim.
- Slot pressure: 2–4 workers competing with Wave-3 items (CARL, diagnostics); mitigated by
  queue position after MPU items and the droppability of stage 2.
