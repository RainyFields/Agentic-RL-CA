# PROPOSED — λ-sweep for turn-level PPO (drafted 2026-07-16, NOT yet approved)

Status: **draft for user review.** No wrappers written, no slots claimed, nothing launched.
If approved, this becomes a numbered item in the slot queue and gets its own decision_log
entry; per the MPU rule (plan.md) it is additive and individually droppable.

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

## 3. Design

- **Vehicle: turn-level PPO (B0), 4turn_think2k protocol** — the turn-level MDP has ≤ 4 credit
  steps, so a coarse grid is meaningful and each λ is interpretable. Everything else
  (data, seeds, batch, lengths, KL, penalty, eval cadence) identical to B0 — single-variable
  contrast, same as the rest of the benchmark.
- **Grid (staged):**
  - Stage 1 (2 workers): **λ = 0.5, λ = 0.9**, seed 0. λ=1 s0/s1/s2 already exist (B0);
    λ=0 deferred (highest collapse risk, only run if stage 1 shows a usable trend toward
    low λ).
  - Stage 2 (conditional, ≤ 2 workers): promote the better stage-1 λ to seeds 1 (+2 if it
    beats B0-λ1 beyond the seed range); or add λ=0 / λ=0.7 to bracket a peak.
- **Token-PPO λ-sweep: explicitly out of scope** (secondary tier at best). Token-level λ acts
  over thousands of tokens, so the interesting range is a different regime (λ ≈ 0.95–1) and
  the token-PPO arm is already the weakest; spend the slots on the interpretable turn-level
  contrast.
- **γ stays 1.0** (undiscounted finite-horizon objective matches EM; sweep exactly one thing).

## 4. Mechanics (launch, not refactor)

- `algorithm.lam` is already config-driven (all current launches pass `algorithm.lam=1.0`).
  Add `LAM_OVERRIDE` (default 1.0) to `scripts/run_condition.sh` in the same style as
  `MICRO_BSZ_OVERRIDE`, mapped to `algorithm.lam=$LAM_OVERRIDE`.
- Note: λ enters **both** the actor's advantages and the critic's regression targets
  (returns = advantages + values in GAE); this is standard and intended — document it in the
  methods note so the arms are described honestly as "GAE(λ)" not "MC baseline".
- Wrappers `.arlca-b0-lam05-s0.sh`, `.arlca-b0-lam09-s0.sh` (COND=turn_ppo_b0, SEED=0,
  PROTOCOL=4turn_think2k, LAM_OVERRIDE=0.5/0.9); micro8 inherited via protocol.
- W&B names per convention: `turn_ppo_b0_lam05_qwen3-1.7b_4turn_think2k_s0`, etc.
- Unit check before launch: one-batch dry assertion that `algorithm.lam` propagates (grep the
  dumped hydra config in the log header), plus the existing 33-test suite.

## 5. Monitoring & tripwires (same fleet standard)

Standard digest metrics + specifically:
- `critic/vf_explained_var` (prediction: rises under λ<1 as targets get easier to fit);
- truncation clip_ratio tripwire (must-fall rule; watch damping prediction above);
- early-warning review at step 150 (RQ3-window style readout vs the three B0 seeds), no
  kill rule — fixed-budget FINAL stays the primary comparison per pre-registration.

## 6. Budget & scheduling

- Each run: 1 × 8×H100 worker, ~19 h wall-clock for 500 steps (B0 reference), retriever
  co-located. Stage 1 = 2 workers; stage 2 ≤ 2 more.
- Queue position: **after** current queue #5 (hcapo-s0, gigpo-s1, token_grpo-s1) and the
  8-turn stress pair — those serve pre-registered headline claims (MPU); the λ-sweep is an
  explanatory ablation. Fires only on freed slots, needs explicit user OK to enter the queue.
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
