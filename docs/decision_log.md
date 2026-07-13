# Agentic-RL-CA decision log

Append-only. Every entry: date (SF time), decision, why, who (user / assistant-proposed).

## 2026-07-13 (plan freeze)
- Plan frozen after 3 review rounds (`docs/plan.md`). Key user decisions recorded there:
  Qwen3-1.7B instruct non-thinking start; 4B single fallback; full paper-protocol eval;
  ARPO + IGPO excluded; StepSearch secondary tier; SPA/HISR dropped for B2 (Co-ReAct judge).

## 2026-07-14 (bootstrap)
- SP6 uncommitted working tree of `external/verl-agent` committed on branch `sp6-alfworld`
  (commit f14911e) in place; `master` left clean for cloning. (assistant, per plan Phase 0.1)
- Project repo cloned from the local fork; branch `agentic-rl-ca` from `master` @ 2df17d2
  (includes SP6 HCAPO clip + projection commits that are on master). Remotes: `local` =
  ~/xiaoxuan/external/verl-agent, `upstream` = github.com/langfengQ/verl-agent. (assistant)
- Upstream `README.md` moved to `docs/UPSTREAM_README.md`; project README at root. (assistant)
- Method PDFs pinned in `docs/papers/` (gitignored): HCAPO=2603.08754, CARL=2512.04949,
  StepSearch=2505.15107, survey=2604.09459. (assistant)
- `gamma=1.0` for all arms (Search-R1 protocol; upstream search example's 0.95 not adopted —
  plan Phase 2b already asserts gamma=1.0 for CARL; matched config requires one value).
  (assistant-proposed; flag to user at Wave-0 review)
- `enable_thinking=False` via `+data.apply_chat_template_kwargs.enable_thinking=False`
  (native rollout_loop support, same as upstream qwen3 examples). (assistant)
- Matched data budget: EVERY arm (incl. PPO/critic arms) uses train_batch 256 × group 5
  = 1280 trajectories/step — identical prompts + rollout count across arms; PPO simply gets
  5 independent trajectories per prompt. (assistant-proposed)
- TOTAL_STEPS default 500 (matches tr1 reference budget); confirm with user before Wave 1.
- CARL constants locked from PDF with three deltas vs plan expectations (see
  `docs/methods_note.md`): N₀ ∈ {1 (Lite), 8 (best)} not N/8=2; resume sampling is
  STOCHASTIC from π_θ (not temp 0 — unbiasedness proof requires it); non-critical edges
  are DROPPED from the update set (Eq. 13), not zeroed → Phase-2b implements drop as
  primary, zeroing as ablation. Plan text updated only via this log (plan stays frozen).
- StepSearch: γ_key, GAE λ/γ, max turns absent from PDF — take from their released code
  when Phase 2c starts (secondary tier).

