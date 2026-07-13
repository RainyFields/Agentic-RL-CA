# HCAPO — Hindsight Credit Assignment Policy Optimization (locked reference fragment)

**Paper:** "Hindsight Credit Assignment for Long-Horizon LLM Agents", Tan, Yang, Chen et al. (Nanjing Univ. / Tencent FiT), arXiv:2603.08754v1, 7 Mar 2026 (preprint dated March 11, 2026).
**Source PDF:** `docs/papers/2603.08754.pdf` (16 pages). Equation numbers below are exactly as in the paper.

---

## 1. Full advantage / credit-assignment formulation

### 1.1 Setting (POMDP, sparse reward)

Trajectory $\tau = (s_1, a_1, \ldots, s_T, a_T)$, policy $\pi_\theta(a_t|s_t)$, sparse scalar terminal reward $R(\tau)$ at $t = T$ (intermediate rewards $r_{t<T} = 0$).

$$J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] \tag{1}$$

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} A_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right] \tag{2}$$

### 1.2 Macro (GRPO) advantage — group statistics $\mu_R, \sigma_R$

GRPO samples a group of $G$ trajectories $\{\tau_1, \ldots, \tau_G\}$ per input query with the current policy:

$$A^{\mathrm{GRPO}}_i = \frac{R(\tau_i) - \mu_R}{\sigma_R} \tag{3}$$

where $\mu_R$ and $\sigma_R$ are the **group statistics of outcome rewards** (mean and std of $R(\tau_i)$ over the $G$ trajectories of the group). This advantage is trajectory-level: constant over all steps $t$ of trajectory $i$.

### 1.3 Classical HCA background (Harutyunyan et al., 2019)

Hindsight distribution $h(a_t|s_t, s_k)$ = probability of taking $a_t$ at $s_t$ given the trajectory eventually visits future state $s_k$. Unbiased Q-value estimate:

$$Q(s_t, a) \approx \hat{r}(s_t, a) + \sum_{k=t+1}^{T-1} \gamma^{k-t} \frac{h(a|s_t, s_k)}{\pi(a|s_t)} R_k + \gamma^{T-t} \frac{h(a|s_t, s_T)}{\pi(a|s_t)} V(s_T) \tag{4}$$

with $\hat{r}$ an estimate of the immediate reward, $R_k$ the reward at step $k$, $\gamma$ the discount factor, $V$ the state-value function.

### 1.4 Refined Hindsight Q-value (sparse-reward simplification)

With $r_{t<T} = 0$, Eq. (4) collapses to:

$$Q^H_{i,t} = \rho_{i,t} \cdot G_{i,t}, \qquad \rho_{i,t} = \frac{h(a_t \mid s_t, s_{\mathrm{final}})}{\pi(a_t \mid s_t)} \tag{5}$$

where $G_{i,t} = \gamma^{T-t} R(\tau_i)$ is the discounted future return and $\rho_{i,t}$ is the **hindsight importance ratio** ("causal filter": $\rho > 1$ amplifies credit, $\rho < 1$ suppresses it).

### 1.5 Generative Verification — hindsight probability $\pi_{\mathrm{hind}}$

No separate model is trained; the successful outcome $s_{\mathrm{final}}$ is injected into the LLM's prompt and the LLM scores the existing action tokens in a single forward pass (prefix scoring, no generation). For action $a_t$ with tokens $(y_1, \ldots, y_{|a_t|})$:

$$\pi_{\mathrm{hind}}(a_t) = \exp\left(\frac{1}{T_{\mathrm{temp}} |a_t|} \sum_{j=1}^{|a_t|} \log \pi_\theta(y_j \mid y_{<j}, s_t, s_{\mathrm{final}})\right) \tag{6}$$

i.e. exponential of the **length-normalized mean token log-probability, divided by a sharpening temperature $T_{\mathrm{temp}}$**, conditioned on both $s_t$ and the hindsight information $s_{\mathrm{final}}$.

### 1.6 Self-normalized importance ratio estimator (with clip)

The intractable prior $\pi(a_t|s_t) = \mathbb{E}_{s_{\mathrm{final}}}[\pi(a_t \mid s_t, s_{\mathrm{final}})]$ (Law of Total Probability) is approximated by the **empirical mean of hindsight scores within the same trajectory** $\bar{\pi}_{\mathrm{hind}}$:

$$\rho_t = \mathrm{clip}\left(\frac{\pi_{\mathrm{hind}}(a_t)}{\bar{\pi}_{\mathrm{hind}}},\ C_{\min},\ C_{\max}\right), \qquad \bar{\pi}_{\mathrm{hind}} = \frac{1}{T} \sum_{k=1}^{T} \pi_{\mathrm{hind}}(a_k) \tag{7}$$

Note: normalization is **intra-trajectory** (mean over the $T$ actions of the same episode), described as "akin to group-normalization across actions within the same episode".

### 1.7 Multi-scale composite advantage — group statistics $\mu_H, \sigma_H$

For the $i$-th trajectory in a group of size $G$:

$$A^{\mathrm{HCAPO}}_{i,t} = \underbrace{\frac{R(\tau_i) - \mu_R}{\sigma_R}}_{\text{Macro (GRPO)}} + \ \omega \cdot \underbrace{\frac{Q^H_{i,t} - \mu_H}{\sigma_H}}_{\text{Micro (Hindsight)}} \tag{8}$$

- Eq. (8)'s in-text description: "$\mu_H$ and $\sigma_H$ are the group statistics of $Q^H$ **at time step $t$**."
- **However**, Section 5.2 ("Rationale for Cross-State Normalization") and Algorithm 1 (line 16, "cross-state normalization") describe $\mu_H$ as a **global group mean computed across heterogeneous states** — i.e. over *all* sampled state-action pairs across all trajectories in the group (see Eqs. 11–12 below). The theoretical analysis, the pseudocode, and the section title all support the global/cross-state reading; the "at time step t" phrasing next to Eq. (8) appears to be loose wording. **Re-implementation should use group-global statistics of $Q^H$ over all $(i,t)$ pairs in the group** (this is what "cross-state" means and what the bottleneck argument $V_{\mathrm{low}} < \mu_H < V_{\mathrm{high}}$ requires). Flagged as the one internal ambiguity of the paper.
- **"Do-no-harm" protective mask:** negative hindsight signals are zeroed out **in successful trials** (i.e. if trajectory $i$ succeeded and the micro term is negative, the micro term is masked to 0). Stated in §4.3; no equation number.

Restated in the theory section as:

$$A^{\mathrm{HCAPO}}_{i,t} = \underbrace{A^{\mathrm{GRPO}}_i}_{\text{Macro Signal}} + \ \omega \cdot \underbrace{\frac{Q^H_{i,t} - \mu_H}{\sigma_H}}_{\text{Micro Correction}} \tag{10}$$

### 1.8 Theoretical characterization of $\mu_H$ (for reference)

$$\mu_H \approx \mathbb{E}_{s \sim d^\pi, a \sim \pi}[Q^H(s, a)] \tag{11}$$

$$\mu_H \approx \mathbb{E}_{s \sim d^\pi}\left(\mathbb{E}_{a \sim \pi(a|s)}[Q^H(s,a) \mid s]\right) = \mathbb{E}_{s \sim d^\pi}[V^H(s)] \tag{12}$$

where $V^H(s)$ is the hindsight state-value function and $d^\pi(s)$ the state visitation distribution. At a bottleneck state $s^*$: $V_{\mathrm{low}} < \mu_H < V_{\mathrm{high}}$, so breakthrough actions get positive micro advantage ($V_{\mathrm{high}} - \mu_H > 0$), non-instrumental actions negative ($V_{\mathrm{low}} - \mu_H < 0$).

### 1.9 Temporal smoothing (optional; Appendix A — no equation number)

Applied to $Q^H_{i,t}$ before the micro normalization, to fix the "credit disconnection" problem (verifier scores final "CleanObject"-type actions high but preparatory "GoToPlace"/"OpenObject" actions low):

$$\tilde{Q}^H_{i,t} = \alpha\, Q^H_{i,t} + (1 - \alpha)\, Q^H_{i,t+1}$$

with $\alpha = 0.5$ — a one-step backward flow of the "breakthrough signal", treating a reasoning step and its immediate execution as one functional unit. Used ("HCAPO w Smooth") it lifts ALFWorld-7B from 91.4% to 96.9%.

### 1.10 PPO surrogate objective

$$J(\theta) = \mathbb{E}_{\{\tau_i\}_{i=1}^{K} \sim \pi_{\theta_{\mathrm{old}}}}\Bigg[\frac{1}{K}\sum_{i=1}^{K}\frac{1}{T_i}\sum_{t=1}^{T_i} \min\Big(r_{i,t}(\theta) A^{\mathrm{HCAPO}}_{i,t},\ \mathrm{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) A^{\mathrm{HCAPO}}_{i,t}\Big) - \beta_{\mathrm{KL}} D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})\Bigg] \tag{9}$$

where $\epsilon$ is the PPO clipping parameter (a **different** clip from $[C_{\min}, C_{\max}]$; its numeric value is **not given** in the paper — settings are stated to be "identical to those in GiGPO") and $\beta_{\mathrm{KL}}$ penalizes KL divergence against the reference policy $\pi_{\mathrm{ref}}$.

### 1.11 End-to-end per-turn advantage computation

1. Roll out $G$ trajectories per task with $\pi_{\theta_{\mathrm{old}}}$ (rollout temperature 1.0); collect terminal rewards $R(\tau_i)$.
2. Macro: compute $A^{\mathrm{GRPO}}_i = (R(\tau_i) - \mu_R)/\sigma_R$ from the group's outcome rewards (Eq. 3). Same value for every turn of trajectory $i$.
3. Hindsight scoring: for every turn $t$ of every trajectory, one forward pass with $s_{\mathrm{final}}$ injected in the prompt → $\pi_{\mathrm{hind}}(a_{i,t})$ via Eq. (6) with $T_{\mathrm{temp}} = 5.0$.
4. Ratio: $\rho_{i,t} = \mathrm{clip}(\pi_{\mathrm{hind}}(a_{i,t})/\bar{\pi}_{\mathrm{hind}},\ 0.8,\ 1.2)$, with $\bar{\pi}_{\mathrm{hind}}$ the per-trajectory mean (Eq. 7).
5. Hindsight Q: $Q^H_{i,t} = \rho_{i,t} \cdot \gamma^{T-t} R(\tau_i)$ with $\gamma = 0.95$ (Eq. 5 / Alg. 1 line 14).
6. (Optional) temporal smoothing $\tilde{Q}^H_{i,t} = 0.5\,Q^H_{i,t} + 0.5\,Q^H_{i,t+1}$.
7. Micro: $A^{\mathrm{Micro}}_{i,t} = (Q^H_{i,t} - \mu_H)/\sigma_H$ with group statistics of $Q^H$ (cross-state; see §1.7 flag).
8. Combine: $A^{\mathrm{HCAPO}}_{i,t} = A^{\mathrm{GRPO}}_i + \omega A^{\mathrm{Micro}}_{i,t}$, $\omega = 1.0$; apply do-no-harm mask (zero negative micro signals in successful trials).
9. Broadcast $A^{\mathrm{HCAPO}}_{i,t}$ to the tokens of turn $t$ and update with PPO-clip + KL objective (Eq. 9).

---

## 2. Hyperparameters (Appendix C.1, exhaustive)

### HCAPO-specific (identical across ALL benchmarks — stated explicitly)

| Symbol | Value | Notes |
|---|---|---|
| $T_{\mathrm{temp}}$ (sharpening temperature, Eq. 6) | **5.0** | Generative Verification |
| $[C_{\min}, C_{\max}]$ (ρ clip, Eq. 7) | **[0.8, 1.2]** | ✅ matches expected [0.8, 1.2] |
| $\omega$ (hindsight weighting, Eq. 8) | **1.0** | ablation: 0 / 0.2 / 0.5 / 1.0 → All SR 72.8 / 79.7 / 84.4 / 87.0 (1.5B ALFWorld); 1.0 is the default |
| $\alpha$ (temporal smoothing factor) | **0.5** | optional; used in "HCAPO w Smooth" |
| $\gamma$ (discount factor) | **0.95** | in $G_{i,t} = \gamma^{T-t}R(\tau_i)$ |

### Per-benchmark training hyperparameters

| Hyperparameter | ALFWorld | WebShop | Search-augmented QA |
|---|---|---|---|
| Max prompt length | 2048 | 4096 | 4096 |
| Max response length | 512 | 512 | 512 |
| Max env steps / turns | 50 steps/episode | 15 steps/episode | 4 turns |
| Actor learning rate | 1e-6 | 1e-6 | 1e-6 |
| Critic learning rate (PPO baseline only) | 1e-5 | 1e-5 | — (not listed) |
| Success reward | +10 (0 on failure) | +10 (0 on failure) | +1 (0 on failure) |
| Invalid-action penalty | −0.1 | −0.1 | −0.01 |
| Group size $G$ | **8** | **8** | **5** |
| Groups per rollout / batch | 16 groups × 8 = 128 envs | 16 groups × 8 = 128 envs | training data size 256 |
| Rollout temperature | 1.0 | 1.0 | 1.0 |
| Validation temperature | 0.4 | 0.4 | 0.0 |
| Mini-batch size | **256** | **64** | **512** |
| $\beta_{\mathrm{KL}}$ | **0.01** | **0.01** | **0.001** |
| Training iterations | 150 | 150 | 200 |
| Hardware | 1.5B: 4×H20; 7B: 8×H20 | 1.5B: 4×H20; 7B: 8×H20 | 3B: 8×H20; 7B: 8×H20 |
| History length in prompt | 2 | 2 | full history |

Base models: Qwen2.5-Instruct 1.5B / 3B / 7B. All settings stated to be identical to GiGPO (Feng et al., 2025) for fairness; baselines' numbers taken from the GiGPO and EMPG papers.

### ⚠️ Mismatches vs. expected values (as queried)

- **Clip range [0.8, 1.2]** — ✅ CONFIRMED (this is the ρ hindsight-ratio clip, *not* the PPO $\epsilon$ clip; PPO $\epsilon$ is never given numerically).
- **Group size "expected 5"** — ⚠️ ONLY for Search-augmented QA ($G=5$). For ALFWorld and WebShop $G = 8$ (16 groups × 8 = 128 environments).
- **Batch size "expected 512"** — ⚠️ 512 is the **mini-batch size for Search-QA only**. ALFWorld mini-batch = 256, WebShop mini-batch = 64. (Search-QA also has "training data size 256".) There is no global "batch size 512".
- **$\beta_{\mathrm{KL}}$ "expected 0.001"** — ⚠️ 0.001 ONLY for Search-augmented QA. ALFWorld and WebShop use $\beta_{\mathrm{KL}} = 0.01$.
- In short: the expected (G=5, batch 512, β_KL=0.001) triple is the **Search-QA config**; the agentic benchmarks (ALFWorld/WebShop) use (G=8, mini-batch 256/64, β_KL=0.01).

Other numbers appearing in the paper: "redundant actions" in the conciseness analysis are defined as actions with $\pi_{\mathrm{hind}} \le 0.9$ **when $T_{\mathrm{temp}} = 1$** (analysis-only threshold, not a training hyperparameter). Hindsight-prob computation ≈ 8.3% of per-iteration wall time (Gen 51.0% / 61.3s, Update_actor 27.1% / 32.5s, Hindsight_prob 8.3% / 10.0s, Old_log_prob 6.9% / 8.3s, Ref 6.7% / 8.0s).

---

## 3. Algorithm 1 (Appendix B) — verbatim transcription

```
Algorithm 1  Training LLM Agents with HCAPO
 1: Require: Initial policy πθ, task distribution p(X), weighting coefficient ω,
    batch size N, clipping bounds [Cmin, Cmax].
 2: for each training iteration do
 3:   Update old policy: θold ← θ
 4:   // 1. Multi-step Rollout Phase
 5:   Sample task x ~ p(X) and initialize N identical environments.
 6:   for t = 1 to T do
 7:     Sample actions a_{i,t} ~ πθold(· | s_{i,t}) for all i ∈ {1, . . . , N}.
 8:     Execute actions, observation {o_{i,t}}_{i=1}^N and then next states {s_{i,t+1}}_{i=1}^N.
 9:   end for
10:   // 2. Hindsight Credit Assignment Phase
11:   Compute Macro Advantage A^GRPO_{i,t} via trajectory-level relative rewards.
12:   Compute hindsight probabilities πhind(a_{i,t}) via Generative Verification.
13:   Estimate importance ratios ρ_{i,t} = clip(πhind(a_{i,t})/π̄hind, Cmin, Cmax).
14:   Derive refined Hindsight Q-values Q^H_{i,t} = ρ_{i,t} · γ^{T−t} R(τ_i).
15:   (Optional) Apply temporal smoothing: Q̃^H_{i,t} = αQ^H_{i,t} + (1 − α)Q^H_{i,t+1}.
16:   Compute Micro Advantage via cross-state normalization:
      A^Micro_{i,t} = (Q^H_{i,t} − µH) / σH.
17:   // 3. Policy Update Phase
18:   Combine multi-scale advantages: A^HCAPO_{i,t} = A^GRPO_i + ω A^Micro_{i,t}.
19:   Update policy θ by maximizing the PPO-clipped surrogate objective J^HCAPO(θ).
20: end for
```

(Transcription notes: line 11 writes the macro advantage with subscript "i,t" although Eq. 3/line 18 use the per-trajectory $A^{\mathrm{GRPO}}_i$ — the macro term is constant across $t$. Alg. 1 uses "batch size N" for the number of parallel environments; Eq. 9 uses $K$ for the group of sampled trajectories; §4.3/Eq. 8 use $G$ for group size.)

---

## 4. Implementation-relevant notes

1. **Value-free / critic-free.** No learned value network. The only "critic" is the policy LLM itself used post-hoc as a scorer (Generative Verification); critic LR 1e-5 in Appendix C applies solely to the PPO *baseline*.
2. **Hindsight scoring is teacher-forced prefix scoring, not generation.** One forward pass over existing action tokens with $s_{\mathrm{final}}$ injected into the prompt; extract token log-probs, average, divide by $T_{\mathrm{temp}}=5$, exponentiate (Eq. 6). Costs ≈8.3% of iteration time.
3. **Two distinct normalizations, at different scopes:**
   - $\bar{\pi}_{\mathrm{hind}}$: **intra-trajectory** mean of $\pi_{\mathrm{hind}}$ over that episode's $T$ actions (surrogate prior in Eq. 7).
   - $(\mu_R, \sigma_R)$: **per-group** mean/std of terminal rewards (macro, Eq. 3).
   - $(\mu_H, \sigma_H)$: **per-group, cross-state** mean/std of $Q^H$ (micro, Eq. 8); see the §1.7 flag — the paper's Eq.-8 caption says "at time step $t$" but §5.2 + Algorithm 1 establish cross-state (all $(i,t)$ in the group) normalization, which the bottleneck-threshold theory requires. Implement cross-state.
4. **Two distinct clips:** ρ-clip $[C_{\min}, C_{\max}] = [0.8, 1.2]$ on the hindsight ratio (Eq. 7), and standard PPO ratio clip $1 \pm \epsilon$ (Eq. 9, $\epsilon$ unspecified — inherit from the GiGPO codebase config since "all experimental settings are kept identical to those in GiGPO").
5. **Order of operations:** ρ-clip → multiply by $\gamma^{T-t}R(\tau_i)$ → (optional) temporal smoothing on $Q^H$ → group-normalize → scale by $\omega$ → add macro → do-no-harm mask. The smoothing (Alg. 1 line 15) happens **before** the micro normalization (line 16).
6. **Do-no-harm mask:** in *successful* trajectories, negative hindsight (micro) signals are zeroed so hindsight can only be neutral-or-amplifying on successes; suppression acts through relative normalization, not by pushing successful actions negative. (§4.3, prose only.)
7. **Advantage granularity:** $A^{\mathrm{HCAPO}}_{i,t}$ is per-turn (per action step), applied uniformly to that turn's response tokens; objective averages $\frac{1}{K}\sum_i \frac{1}{T_i}\sum_t$ (Eq. 9), i.e. per-trajectory length normalization.
8. **KL:** explicit KL-divergence loss term against $\pi_{\mathrm{ref}}$ with coefficient $\beta_{\mathrm{KL}}$ (0.01 agentic / 0.001 Search-QA), subtracted inside the objective (Eq. 9).
9. **Sparse-reward assumption is load-bearing:** Eq. 5 is only valid because $r_{t<T}=0$; with the invalid-action penalty (−0.1 / −0.01) the paper still treats the return as $\gamma^{T-t}R(\tau_i)$ from the terminal outcome reward.
10. **Discounting inside $Q^H$:** unlike vanilla GRPO (undiscounted terminal reward for all steps), HCAPO's micro term uses $\gamma^{T-t}R(\tau_i)$, $\gamma = 0.95$ — earlier steps get exponentially smaller raw hindsight Q before normalization.
11. **Prompts:** agent templates in Appendix C.2 use `<think></think>` + `<action></action>` (ALFWorld/WebShop, history length 2) and `<think>/<search>/<answer>/<information>` tags for Search-QA (full history). The hindsight-verification prompt conditions on state + hindsight info $s_{\mathrm{final}}$ (Figure 2a); its exact template text is **not** printed in the paper.
12. **Everything follows GiGPO's experimental configuration** (benchmarks, baselines, and unlisted knobs) — the GiGPO (verl-agent) codebase is the reference for any hyperparameter the paper omits (e.g. PPO $\epsilon$, optimizer details, warmup).
