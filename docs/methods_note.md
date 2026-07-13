# Methods note — locked formulas & constants (Phase 0.3)

Detailed per-paper references (equations numbered as in each PDF, algorithm boxes transcribed):
- `methods_hcapo.md` — HCAPO, arXiv 2603.08754
- `methods_carl.md` — CARL (Criticality-Aware Agentic RL), arXiv 2512.04949
- `methods_stepsearch.md` — StepSearch, arXiv 2505.15107
- `CREDIT_ASSIGNMENT_CHECKLIST.md` — survey arXiv 2604.09459 Appendix C (Tables 11–12 verbatim)

PDFs pinned in `docs/papers/` (gitignored). This file is the summary of record: the constants we
run with, and every delta between the plan's expectations and what the PDFs actually say.

## HCAPO (locked 2026-07-14)

| knob | value | source |
|---|---|---|
| ω (micro/macro mix) | 1.0 (ablation grid 0/0.2/0.5/1.0) | paper §5/App C |
| T_temp (hindsight temperature) | 5.0 | App C |
| ρ-clip [C_min, C_max] | [0.8, 1.2] (hindsight-ratio clip; PPO ε inherited from GiGPO, not given) | App C |
| temporal smoothing α | 0.5 (paper: 91.4 → 96.9 with smoothing) | App C |
| group size G | **5 for Search-QA** (8 is ALFWorld/WebShop) | App C |
| mini-batch | **512 for Search-QA** (256 ALFWorld / 64 WebShop) | App C |
| β_KL | **0.001 for Search-QA** (0.01 agentic envs) | App C |
| actor LR | 1e-6 | App C |
| γ (their runs) | 0.95 | App C |

Plan's expected values (group 5, batch 512, β_KL 0.001) are the Search-QA column — confirmed for
our domain. Ambiguity flagged: Eq. 8 says μ_H/σ_H "at time step t" but §5.2 + Algorithm 1 imply
cross-state (group-global) normalization → we implement **cross-state** (fork's SP4 code already
does; keep). Our matched protocol uses γ=1.0 (decision_log 2026-07-14) — HCAPO's own table used
0.95; flag at Wave-0 review.

## CARL (locked 2026-07-14) — three deltas vs plan expectations

| knob | value | source |
|---|---|---|
| N (total rollouts) | 16 | confirmed |
| N₀ (phase-1 rollouts) | **1 (CARL-Lite default) / 8 (best "CARL")** — paper never uses N/8=2 | App C; **plan said "N₀=1/8" meaning ratio — actual reading: the two paper variants ARE N₀=1 and N₀=8** |
| resume temperature | **NOT 0 — stochastic sampling from π_θ required** (unbiasedness proof needs it; temp-0 appears only in the Eq. 5 preliminary study) | §4/App |
| non-critical edges | **DROPPED from D_upd (Eq. 13), not zeroed** — plan's "zeroed advantage = loss exclusion" is equivalent only under strict drop mode → implement drop as primary, zeroing as ablation flag | Eq. 13 |
| advantage | A(e) = E[R(v)] − E[R(u)] (Eq. 11); values by unweighted child-mean Bellman recursion (Eq. 10) | Eq. 10–11 |
| critical state | source state with >1 distinct child | §4 |
| critic / adv-norm / KL | none / none / none; clip ε not given | App C |
| batch / LR | 128 & 5e-6 (Qwen2.5 3B/7B); 64 & 1e-5 (Qwen3-4B) | App C |
| reward | terminal-only format + F1 | App C |

Implementation ambiguities to resolve at Phase-2b coding (both flagged in `methods_carl.md`):
(a) "phase-1 actions not included in training" (App C.1) vs Eq. 15 counting that includes forked
phase-1 edges — empirical |D_upd| ≈ 2N = 32 supports **including forked phase-1 edges**;
(b) Eq. 14 says N leaves, but Algorithm 1 + |D_roll| imply **N₀ + N leaves** (N phase-2 forks).

## StepSearch (locked 2026-07-14; secondary tier)

| knob | value | source |
|---|---|---|
| overall reward | r = r_answer(F1) + γ_key·r_key at final token (Eq. 6) | Eq. 6 |
| step reward | r_step^t = G^t − P^t at last token of each search round (Eq. 7) | Eq. 7 |
| information gain G^t | mean_i max(c_i^t − m_i^{t−1}, 0), TF-IDF cosine vs golden docs (Eqs. 8–11) | Eqs. 8–11 |
| redundancy P^t | fraction of round-t docs already in seen-set H^{t−1} (Eqs. 12–13) | Eqs. 12–13 |
| per-episode state | memory vector m_i (max sim per golden doc) + seen-doc set H | §3 |
| PPO | critic + GAE; clip ε=0.2, β_KL=1e-3, actor lr 7e-7, critic lr 7e-6, batch 256/64/32, 500 steps, temp=top_p=1.0, info tokens loss-masked | App |
| data | GPT-4o-augmented MuSiQue: 19k questions, 60k filtered sub-question keywords (valid in ≥⌈M/2⌉ of M engines) | §3 |
| **NOT in paper** | γ_key value, GAE λ/γ, max turns, N/M pipeline counts — must come from released code | flagged |

## Matched-protocol constants (all arms; `configs/protocol_4turn.sh` is the source of truth)

Qwen3-1.7B instruct `enable_thinking=False`; max_steps=4, history=4, prompt 4096, response 512,
truncation=left (any nonzero rate = bug); train_batch 256 × group 5 = 1280 traj/step for every
arm (PPO arms included — identical data budget); mini-batch 512; lr 1e-6 (critic 1e-5);
β_KL=0.001 (KL-in-loss arms); γ=1.0, λ=1.0; 500 steps; val = fixed val_2048, greedy, every 25.
