# Agentic-RL-CA

**Progress supervision & credit assignment for multi-turn LLM agents** — Search-R1 QA setting,
Qwen3-1.7B. This repository is a fork of [verl-agent](https://github.com/langfengQ/verl-agent)
(verl 0.3.1.dev) with the project's net-new code added at the repo root; the diff vs upstream is
the publication artifact.

Headline question: **outcome-only credit assignment vs explicit progress supervision for
multi-turn LLM agents — when does intermediate reward actually help?**

## Research questions

| RQ | Question | Design |
|---|---|---|
| RQ1 | Does credit granularity matter? | token-level PPO (`gae`) vs turn-level PPO (`gae_turn`), controlled contrast |
| RQ2 | Can outcome-only CA recover useful intermediate credit? | GRPO / GiGPO / HCAPO (+ CARL) benchmark **+ credit-alignment diagnostic** (assigned credit vs MC continuation value) |
| RQ3 | Does explicit privileged progress supervision add value? | B0 (sparse) vs B1 (privileged answer-exposure step reward) vs **B1-shuffle** (timing control) |
| RQ4 | Does the benefit grow with task/horizon difficulty? | single-hop vs multi-hop subgroup (pre-registered); 4-turn vs 8-turn stress test |
| RQ5 | Can a scalable evaluator approximate the privileged signal? | B2-lite (prompted step evaluator) → B2-full (follow-up project) |

## Layout

- `credit_assignment/` — **NEW**: CARL estimator, B1/B1-shuffle shaping helpers, credit-alignment diagnostic
- `configs/` — **NEW**: one file per experimental condition (incl. the 8-turn horizon stress config)
- `scripts/` — upstream utilities + **NEW** per-condition launchers, retriever bring-up, toy gate, full-set eval
- `analysis/` — **NEW**: figure/report builders (every figure backed by CSV + provenance README)
- `docs/` — upstream docs + **NEW** `plan.md`, `methods_note.md`, `CREDIT_ASSIGNMENT_CHECKLIST.md`, `decision_log.md`
- `verl/`, `agent_system/`, `gigpo/` — upstream (dispatch/eval/rollout patches land in `verl/` and `agent_system/`; `gigpo/` untouched)
- `outputs/`, `data/` — gitignored (HDFS pointers + prep commands in docs)

## Setup / repro

- Model: Qwen3-1.7B instruct, `enable_thinking=False` (Search-R1-family template supplies the `<think>` block as plain text)
- Protocol: `env.max_steps=4`, `history_length=4`, `max_prompt_length=4096`, `max_response_length=512` — all horizon values config-driven (no literal turn counts in code); 8-turn stress config ships in `configs/`
- Data & retriever: Search-R1 assets (e5 Flat index + wiki-18 corpus + NQ/HotpotQA train, 51,713-row 7-dataset test set); see `docs/plan.md` §Reused assets
- Eval: full Search-R1 paper protocol — subsampled val during training + post-hoc full-set greedy EM on pre-registered checkpoints
- W&B project: `agentic-rl-ca`; experiment standard: arXiv 2604.09459 Appendix C checklist (`docs/CREDIT_ASSIGNMENT_CHECKLIST.md`)

Upstream README: `docs/UPSTREAM_README.md`. License: Apache-2.0 (inherited).
