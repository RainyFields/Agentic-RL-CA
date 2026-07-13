# Credit Assignment Reporting Checklist

**Provenance:** arXiv 2604.09459 — *survey on credit assignment in RL for LLMs (reasoning RL → agentic RL)*, Appendix C ("Reporting Checklist for Future Credit Assignment Papers"), Tables 11 and 12 (pp. 35, 46–47 of the PDF). Survey analysis snapshot: April 2026.

**Purpose:** Every Agentic-RL-CA experiment and report **must fill out this checklist** (Table 11 items) and may optionally include a Table-12-style scorecard. This directly targets the survey's identified systemic gaps: no reviewed paper reported GPU-hours, none included compute-controlled baselines, and variance estimates were rare (see §"Three Systemic Gaps" below).

From the survey's Appendix C:

> Based on the methodological gaps identified in this survey, we propose the following reporting checklist for future CA papers.

---

## 1. Reporting Checklist (Table 11, verbatim)

> **Table 11:** Recommended reporting checklist for credit assignment papers in LLM RL.

| # | Category | Priority | Item |
|---|----------|----------|------|
| 1 | Model & Data | Required | Base model name, size, and version |
| 2 | Model & Data | Required | Training data: source, size, and any filtering applied |
| 3 | CA Method | Required | Credit granularity (token / segment / step / turn / multi-agent) |
| 4 | CA Method | Required | Methodology family per our taxonomy |
| 5 | Baselines | Required | At least one episode-level baseline (GRPO or PPO) with identical base model |
| 6 | Baselines | Required | Baseline training recipe: same compute budget or explicit compute comparison |
| 7 | Baselines | Recommended | CA-component ablation isolating the contribution |
| 8 | Evaluation | Required | Benchmark names and specific splits |
| 9 | Evaluation | Required | Evaluation metric with exact definition |
| 10 | Evaluation | Recommended | Variance estimates (std across ≥3 seeds, or confidence intervals) |
| 11 | Compute | Required | Total GPU-hours for training |
| 12 | Compute | Recommended | CA-specific overhead: additional forward passes, environment resets, or LLM calls |
| 13 | Trajectory Info | Required | Average trajectory length (tokens for reasoning, turns for agentic) |
| 14 | Trajectory Info | Recommended | Trajectory length distribution (min, median, max) |

*(14 items: 9 Required, 5 Recommended. The `#` column is added for reference; Category/Priority/Item text is verbatim from the survey.)*

---

## 2. Scorecard Format (Table 12, verbatim)

The survey validates the checklist by scoring three representative papers — one per setting (reasoning / agentic / multi-agent): HICRA (reasoning), GiGPO (agentic), M-GRPO (multi-agent). Legend (from Appendix C): **✓ = reported, ∼ = partially reported, × = not reported.**

> **Table 12:** Checklist validation: three representative papers scored against the reporting checklist.

| Category | Item | HICRA (R) | GiGPO (A) | M-GRPO (M) |
|----------|------|-----------|-----------|------------|
| Model | Base model name/size | ✓ | ✓ | ✓ |
| Model | Training data source/size | ∼ | ∼ | × |
| CA Method | Credit granularity | ✓ | ✓ | ✓ |
| CA Method | Methodology family | ✓ | ✓ | ✓ |
| Baselines | Episode-level baseline (same model) | ✓ | ✓ | ✓ |
| Baselines | Compute-controlled baseline | × | × | × |
| Baselines | CA-component ablation | ✓ | ∼ | × |
| Evaluation | Benchmark names + splits | ✓ | ✓ | ∼ |
| Evaluation | Variance / confidence intervals | ∼ | × | × |
| Compute | Total GPU-hours | × | × | × |
| Compute | CA-specific overhead | ∼ | ∼ | × |
| Trajectory | Avg trajectory length | ∼ | ✓ | × |

*(12 scored rows. For Agentic-RL-CA reports: replace the three paper columns with our own runs/configs and score each against the same items.)*

---

## 3. The Three Systemic Gaps (exact quote)

From Appendix C ("Key gaps identified"), quoted verbatim:

> **Key gaps identified.** Three patterns emerge: (1) no paper reports total GPU-hours; (2) no paper provides compute-controlled baselines; (3) variance estimates are rare. A coarse manual audit across all 47 reviewed papers confirms this (note: this audit was informal and based on our reading, not a formal inter-rater coded review): of the 41 core CA methods, **0/41 report total GPU-hours**, **2/41 report variance or confidence intervals**, and **0/41 include a compute-controlled baseline**.

(Bold emphasis added; wording and numbers exact.)

---

## 4. Taxonomy Axes (for figure F1)

The survey's two-dimensional taxonomy grid (§2.4, Figure 2) organizes methods along **two orthogonal axes**:

**1. Granularity axis** — *At what level is credit assigned?*
- **Token-level**: Individual tokens within a generation
- **Segment-level**: Semantically meaningful spans (e.g., one reasoning step)
- **Step/Turn-level**: A complete LLM response or tool-call cycle
- **Multi-agent level**: Credit decomposition across collaborating agents

**2. Methodology axis** — *How is credit computed?*
- **Monte Carlo (MC)**: Rollouts from intermediate states
- **Temporal Difference (TD)**: Learned value functions with bootstrapping
- **Model-based / LLM-as-Critic**: LLMs evaluate intermediate states
- **Game-theoretic**: Shapley values, counterfactual baselines
- **Information-theoretic**: Information gain, entropy-based measures

Figure 2 of the survey plots the 4×5 grid (granularity vertical, methodology horizontal), color-coded by setting (blue = reasoning RL, red = agentic RL, purple = multi-agent), with a dashed "evolution trend" arrow from fine-grained reasoning methods (upper-left) toward coarser but environment-aware agentic methods (lower-right); the densest cluster sits at the Step/Turn level. Use these exact axis/category names in figure F1.
