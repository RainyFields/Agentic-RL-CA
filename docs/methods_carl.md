# CARL — Criticality-Aware Agentic Reinforcement Learning (arXiv:2512.04949v3, 11 May 2026)

Shen, Zhang, Ling, Zhao, Chua (NUS). Preprint dated May 12, 2026. 18 pages. Setting: multi-turn search agents (ASearcher-style), knowledge-intensive QA. Source: `/home/tiger/xiaoxuan/Agentic-RL-CA/docs/papers/2512.04949.pdf`.

Notation from the paper: MDP ⟨S, A, P, R⟩; R(s) is a **rule-based outcome reward, non-zero only at terminal states s_T**; policy π_θ(a|s) generates each action a = (a¹,…,a^|a|) autoregressively; trajectory τ = {(s_t, a_t)}_{t=1..T}; episode ends at an "answer" action or at the max step limit T_max. Action types in the search pipeline: **search, access, read, answer**.

---

## 1. Two-phase rollout-tree construction ("Entropy-Guided Progressive Rollout", §5 + Appendix C.1)

The rollout for each question builds a **tree**: **nodes represent states and directed edges represent actions** (Fig. 3). The paper calls the whole procedure "Uncertainty-Guided Progressive Forking" (Algorithm 1, Appendix C.1/p.16); the two phases are:

### Phase 1 — initial rollouts from scratch
- Generate **N₀ trajectories from the root state s₀** (full rollouts from scratch), "to establish a basic set of candidates" / "improve stability by ensuring a basic set of candidates to start with".
- For every (s, a) pair in these trajectories, compute the action entropy h = ENTROPY(a) and record the triple **(state, uncertainty, count) = (s, h, 1)** into a state buffer S.
- **Verbatim caveat (Appendix C.1): "Notably, actions in this phase are not included in training."** (See §5 of this fragment for the tension between this sentence and Eq. 13 / Eq. 15.)
- The paper never uses the word "snapshot"; resumability of intermediate states is implicit (fork = re-generate from a stored intermediate state/prefix).

### Phase 2 — progressive forking (resumes)
- **Resume-point selection is greedy by "action density", NOT a schedule and NOT round-robin.** Define action density (Eq. 12):

  d(s_t) = n(s_t) / H_{π_θ}(s_t)                (12)

  where n(s_t) = number of children already sampled from s_t, and H_{π_θ}(s_t) is the estimated action entropy of s_t (Eq. 6/7 below). At each expansion step CARL **greedily selects the node with the lowest action density** (equivalently, in Algorithm 1, "Select (ĥ, ŝ, n̂) with biggest ĥ/n̂ from S"), and rolls out one new trajectory from that state to termination.
- Each resumed rollout's new (s′, a′) pairs are also entropy-scored and inserted into the buffer S; if s′ equals the fork state ŝ, its entropy estimate is updated as a **running mean** ĥ ← (h′ + ĥ·n̂)/(n̂+1), n̂ ← n̂+1 (Algorithm 1 line 18). So the entropy estimate of a state is the Monte-Carlo average over all actions actually sampled from it (Eq. 7).
- The loop runs "for i = 1 to N" (N fork/resume iterations). See §5 "compute accounting" for the leaf-count ambiguity this creates.

### Sampling temperatures — MISMATCH WITH EXPECTATIONS
- **The paper specifies NO sampling temperature for either rollout phase and says nothing about greedy decoding after a resumed action during training.** The word "temperature" appears exactly once in the paper: in the *preliminary analysis* criticality definition (Eq. 5), where τ_{s_t,a′} means "the trajectory taking action a′ at state s_t **followed by greedy decoding thereafter (i.e., temperature=0)**". That temp-0 protocol is used only for the offline criticality-measurement study (70 tasks, 294 states), not stated for CARL training rollouts.
- Expected "resume temp 0 / greedy after first resumed action" — **NOT stated for the training algorithm**; Algorithm 1 just calls ROLLOUTFROM(ŝ; π_θ) with ordinary policy sampling (children must be "sampled independently from π_θ(·|u)" for the unbiasedness argument, which actually *requires* stochastic sampling, not greedy continuation, at forked states).

### Node/state identity
- There is **no state-merging mechanism**. The tree exists purely by prefix sharing: a node is an intermediate state of some generated trajectory (the token/interaction prefix up to that point), and forking re-samples a new action from that exact stored prefix. Identity in Algorithm 1 is the literal state object (test "if s′ = ŝ"); states from different prefixes are always distinct nodes ("insert new candidate"). Nodes with the same environment content reached via different paths are NOT merged.

---

## 2. Criticality definition and edge advantage

### Ground-truth criticality (preliminary study, §4)
Criticality of state s_t = degree to which the action choice at s_t influences the final outcome, quantified as the **std of outcome reward when stochasticity is isolated to the action sampled at that state** (Eq. 5):

  C_{π_θ}(s_t) = std_{a′∼π_θ(·|s_t)} [ R(τ_{s_t,a′}) ]        (5)

with τ_{s_t,a′} = trajectory taking a′ at s_t then greedy (temperature 0) thereafter. Empirics: >50% of states have near-zero reward std; ~10% have std > 0.4.

### Practical proxy: action entropy (Eqs. 6–8)
Since Eq. 5 is too expensive, CARL uses **action entropy** (sequence-level, not token-level) estimated by Monte Carlo:

  H_{π_θ}(s_t) = E_{Y∼π_θ(·|s_t)}[ −log π_θ(Y|s_t) ]          (6)
  ≈ (1/N) Σ_{i=1}^{N} [ −log π_θ(Y^(i)|s_t) ],  Y^(i) ∼ π_θ(·|s_t)   (7)

with **length-normalized** sequence log-probability (following Wu et al. 2016, GNMT) (Eq. 8):

  log π_θ(Y^(i)|s_t) = (1/|Y^(i)|) Σ_{j=1}^{|Y^(i)|} log π_θ(y_j^(i) | y_{<j}^(i), s_t)   (8)

Validation: high- vs low-criticality states differ in entropy with Brunner–Munzel test p = 0.002, Cliff's δ = 0.42.

### When is a state "critical" at training time?
Operationally, in the update rule a state counts as high-criticality **iff the rollout tree sampled more than one child from it** (|child(s_t)| > 1), i.e. iff the entropy-guided forking chose to branch there. There is **no explicit entropy threshold**; criticality is determined implicitly by where the density-greedy forking allocated branches.

### Expected reward of a state (Eqs. 9–10)

  E[R(u)] = E_{π_θ}[ R(τ) | s₀ = u ]                          (9)
  E[R(u)] = Σ_v p(v|u) E[R(v)] ≈ (1/N) Σ_{i=1}^{N} E[R(v_i)]   (10)

i.e. recursively computed bottom-up as the **plain (unweighted) mean of child-node values**; leaf value = environment outcome reward. (Appendix A.1 restates this as Eq. 16 and proves unbiasedness: children are i.i.d. samples from π_θ(·|u), so the sample mean is unbiased, propagating recursively; only requirement is independent sampling of children from π_θ — a full n-ary tree is NOT required. Contrasts with TreeRL's leaf-count-weighted average, Eqs. 25–26, shown biased in Eqs. 17–20, and ARPO's shared-token averaged group advantage, Eqs. 21–24, also biased.)

### Edge advantage (Eq. 11) — matches the expected E[R(dst)] − E[R(src)] form
For each action edge e = (u, v) (parent u, child v):

  A(e) = E[R(v)] − E[R(u)]                                    (11)

Sign convention: dst minus src, i.e. "how much taking action e improves the expected outcome compared to state u". No group mean subtraction and **no std normalization** anywhere in the advantage (unlike GRPO Eq. 4).

### Gating on criticality / fate of non-critical actions (Eq. 13)
Non-critical actions are **DROPPED from the update set entirely (excluded from gradient updates), not zeroed**:

  D_upd = { (s_t, a_t, A_t) | (s_t, a_t, A_t) ∈ D_roll, |child(s_t)| > 1 }    (13)

i.e. an action is trained on iff its source state has ≥2 sampled children (has siblings). Ablation Exp. #3 ("update All"): keeps low-criticality actions by letting them **inherit the advantage of their preceding action**; this is ~1 point worse than full CARL at comparable cost, confirming exclusion is better.

Per-token assignment: "we follow the setting of GRPO to assign the action-level reward to all tokens within the action" — A(e) is broadcast uniformly to every token of that action; update uses the PPO clipped loss (Eq. 2) with ratio r_i(θ) (Eq. 3).

---

## 3. Hyperparameters (everything the paper gives)

| Hyperparameter | Value | Where |
|---|---|---|
| Total rollouts N (Eq. 14 / Alg. 1) | **16** | §5 Efficiency Analysis ("N₀ and N are set to 1 and 16 respectively in our default setting") — matches expectation N=16 |
| Initial sample size N₀ (default / CARL-Lite) | **1** | §5 Efficiency Analysis; Table 2 CARL-Lite |
| Initial sample size N₀ (best / "CARL" rows) | **8** | §5 ("When we increase N₀ to 8 in our best performance setting"), §6.2 |
| Resume/rollout temperature | **NOT SPECIFIED** (temp 0 appears only in the Eq. 5 preliminary study) | — |
| Backbones | Qwen2.5-3B, Qwen2.5-7B (non-reasoning); Qwen3-4B (reasoning) | App. C.3 |
| Batch size (non-reasoning 3B/7B) | **128** | App. C.3 |
| Batch size (reasoning 4B) | **64** | App. C.3 |
| Learning rate (non-reasoning) | **5 × 10⁻⁶** | App. C.3 |
| Learning rate (reasoning) | **1 × 10⁻⁵** | App. C.3 |
| Optimization steps | **100** (all), except **50** for 4B-reasoning max-32-actions | App. C.3 |
| Max actions per episode | 32 (3B, 7B non-reasoning); **10** (4B reasoning, main analysis) and **32** (4B reasoning, extended) | Table 2 section headers |
| KL coefficient | **NOT MENTIONED anywhere** (no KL term in the loss, Eq. 2 has none) | — |
| Clip ϵ | appears symbolically in Eq. 2, **value never given** | — |
| Reward function | format reward + F1-based answer reward (combination weights not given) | App. C.3 |
| Training data | "valid subset of ASearcher-base with invalid questions excluded" | App. C.3 |
| Search environment | local retrieval server, 2018 Wikipedia dump (Karpukhin 2020), E5 retriever | §6.1 |
| Avg trajectory length statistic | E[T] = 5 (used in efficiency accounting) | §5 Efficiency Analysis |
| Hardware | 8×A100 | App. A.3 |
| LasJ judge | GPT-5-nano-2025-08-07, binary Correct/Incorrect (prompt in Fig. 6) | App. C.4 |
| OOD eval | GAIA, Frames, xBench-DeepSearch; online search + website access tools; 4 seeds, Avg@4 & Pass@4 | §6.2, Table 3 |

**Mismatches vs. stated expectations:**
1. **N₀ ≠ N/8.** Expected N₀ = 2 (=16/8); paper uses **N₀ = 1** (default / CARL-Lite) or **N₀ = 8** (best-performing "CARL"). N = 16 confirmed.
2. **Resume temperature 0: NOT confirmed.** No training-time temperature is given anywhere; greedy-after-resample (temp 0) is used only in the §4 preliminary criticality measurement. The unbiasedness proof (App. A.1) in fact requires stochastic i.i.d. child sampling from π_θ.
3. No KL coefficient, no clip value, no critic/GAE settings — none exist in the paper.

Key results for sanity-checking a re-implementation (Table 2, 4B reasoning): max-10: GRPO |D_roll|=80.9, |D_upd|=80.9, F1 57.0/LasJ 59.5; CARL-Lite (N₀=1) 39.8/30.5, 57.6/59.6; CARL (N₀=8) 81.8/**32.0**, 58.4/60.5. max-32: GRPO 115.2/115.2, 57.4/59.7; CARL 141.3/**32.0**, 59.2/61.9. GPU-hours (Table 6, 4B): GRPO 423.05, CARL-Lite 253.17, CARL 403.92. Ablations (Table 4): full CARL (#4) 58.4/60.5; outcome-reward variant (#1) 52.9/55.0; random-fork variant (#2) 56.2/58.1; update-all variant (#3) 57.5/59.4.

---

## 4. Algorithm boxes (transcribed verbatim, modulo notation typesetting)

### Algorithm 1 — Uncertainty-Guided Progressive Forking (Appendix C.1, p.16)

```
 1: Input: Policy π_θ, root state s₀, total rollouts N, initial sample size N₀
 2: Output: Sampled tree T
 3: Initialize T ← {s₀}, state buffer S ← ∅
 4: for i = 1 to N₀ do
 5:     τ ← RolloutFrom(s₀; π_θ)
 6:     for each (s, a) in τ do
 7:         h ← Entropy(a)
 8:         Add (s, h, 1) to S            // record (state, uncertainty, count)
 9:     end for
10: end for
11: for i = 1 to N do
12:     Select (ĥ, ŝ, n̂) with biggest ĥ/n̂ from S    // pick the state with the lowest action density
13:     τ ← RolloutFrom(ŝ; π_θ)
14:     T ← T + 1
15:     for each (s′, a′) in τ do
16:         h′ ← Entropy(a′)
17:         if s′ = ŝ then
18:             ĥ ← (h′ + ĥ·n̂)/(n̂ + 1),  n̂ ← n̂ + 1
19:             Update (ĥ, ŝ, n̂) in S     // Update uncertainty
20:         else
21:             Add (s′, h′, 1) to S       // insert new candidate
22:         end if
23:     end for
24: end for
```

(Transcription notes: line 12 as printed reads "Select (ĥ, ŝ, n̂) with biggest ĥ/n̂" — biggest entropy-per-sample = lowest action density d = n/H. Line 14 "T ← T + 1" is as printed (presumably "add τ to tree T"). The entropy of a *state* is maintained as the running mean of the entropies (length-normalized negative log-probs, Eq. 8) of the actions sampled from it — a 1-sample estimate until the state is forked. Note ENTROPY(a) is computed per sampled action a, i.e. h = −log π_θ(a|s) length-normalized; the Monte-Carlo average of Eq. 7 is realized incrementally via line 18.)

### Algorithm 2 — Tree-based Advantage Estimation (Appendix C.2, p.17)

```
 1: Input: Reasoning Tree T with root node u_root
 2: Output: Advantage training dataset D
 3: Initialize D ← ∅
    // Phase 1: Bottom-up Value Estimation (Post-Order DFS)
 4: Procedure EstimateValue(u)
 5: if u is a leaf node then
 6:     V(u) ← R(u)                        // Final reward from environment
 7: else
 8:     for each child v in u.children do
 9:         EstimateValue(v)
10:     end for
11:     V(u) ← (1/|u.children|) Σ_v V(v)   // Mean value aggregation
12: end if
13: End Procedure
    // Phase 2: Top-down Advantage Collection (Pre-Order DFS)
14: Procedure CollectAdvantage(u, V_parent)
15: if u ≠ u_root then
16:     A(u) ← V(u) − V_parent
17:     Add (seq_u, A(u)) to D             // Record sequence and local advantage
18: end if
19: for each child v in u.children do
20:     CollectAdvantage(v, V(u))
21: end for
22: End Procedure
    // Main Execution
23: EstimateValue(u_root)
24: CollectAdvantage(u_root, 0)
```

(Note: Algorithm 2 as printed collects an advantage for **every** non-root node; the |child(s)| > 1 filter of Eq. 13 is applied to form D_upd — the algorithm box itself does not show the filter.)

---

## 5. Implementation-relevant notes

- **No critic, no reward model.** Values are pure Monte-Carlo tree averages of the terminal rule-based reward (Alg. 2); the paper positions itself among "reward-model-free" methods. No GAE, no learned value head mentioned.
- **Loss over excluded rows: hard exclusion, not masking-to-zero.** D_upd (Eq. 13) simply omits actions whose source state has ≤1 child; the PPO clipped loss (Eq. 2, averaged 1/|D| over included samples) is computed only on D_upd. Ablation #3 shows the alternative (keep all, low-criticality actions inherit predecessor's advantage) is worse.
- **Phase-1 exclusion sentence.** Appendix C.1: "actions in this phase [the N₀ initial rollouts] are not included in training." Taken literally this excludes phase-1 actions even if their source state is later forked — but §5's Eq. 15 derivation says "forking from a state without siblings adds **two** new actions to the update set" (the pre-existing lone edge + the new edge), which counts the pre-existing (phase-1) edge as trained. These two statements are in tension; the Eq. 15 bound N+1 ≤ |D_upd| ≤ 2N and the observed |D_upd| ≈ 32 = 2N support the Eq.-15 reading (phase-1 edges DO enter D_upd once their source state acquires siblings). Most likely intended meaning: phase-1 actions are not *unconditionally* included — only those later gaining siblings are.
- **No advantage normalization.** A(e) is a raw expected-reward difference (rewards ∈ [0,1]-ish, F1 + format); no group std division (GRPO Eq. 4 is baseline-only). No mention of batch-level advantage whitening.
- **Entropy is sequence-level and length-normalized** (Eq. 8, GNMT-style). This matters: without length normalization, long "read" actions would dominate. Entropy per state is estimated from as few as 1 sample initially (running mean updated on revisits, Alg. 1 line 18).
- **Update-set size bound (Eq. 15):** N + 1 ≤ |D_upd| ≤ 2N. With N = 16: 17–32 samples per question; observed ≈ 32 (i.e., forking mostly hit fresh states). GRPO uses T·N action samples with E[T] = 5, hence "CARL uses 60% fewer actions for model update" and the headline "72% fewer updated actions" (Fig. 1, 4B max-32: 32.0 vs 115.2).
- **Rollout compute accounting (Eq. 14):** counted in **number of actions generated** (|D_roll| in tables = average actions performed during rollout per task). Assuming each trajectory has T actions and a resumed rollout costs T/2 on average (uniform fork position), CARL's rollout cost relative to GRPO's N·T is

  (T·N₀ + (T/2)·N) / (N·T) = N₀/N + 1/2.        (14)

  N₀=1, N=16 → 0.5625 ("saves 44% of resources"); N₀=8 → 1.0 ("same rollout resource consumption as GRPO"). Note prefix reuse is assumed free (no accounting for re-prefill of shared prefixes).
- **Leaf-count ambiguity.** Eq. 14's text says "the number of leaf nodes N remains unchanged", but Algorithm 1 performs N₀ from-scratch rollouts **plus** N forks → N₀ + N leaves. The empirical |D_roll| numbers (CARL-Lite 39.8 ≈ 1·5 + 16·(T/2); CARL 81.8 ≈ 8·5 + 16·(T/2), vs GRPO 80.9 ≈ 16·5) are consistent with the **phase-2 loop running the full N = 16 times on top of N₀**, i.e., N₀ + N = 17 or 24 leaves — despite baselines being matched on "an identical number of terminal states per group" (App. C.5). A re-implementation must choose; the numbers favor "N forks in phase 2, leaves = N₀ + N".
- **Reward:** rule-based, terminal-only; format reward + F1 answer reward (App. C.3). Non-terminal R(s) = 0 by MDP definition (§3).
- **Baseline adaptations (App. C.5)** if needed for comparison: TreeRPO — partial expansion by random node selection instead of full 8-ary depth-3 tree, matched leaf count; TreeRL — hybrid global+local advantage with re-weight factor |L(s_n)|^(−1/2) (Eqs. 25–26); ARPO — branch prob P_t = α + β·ΔH_t vs threshold τ, branch Z paths, GRPO leaf advantages averaged over shared tokens.
- **Efficiency side-effects:** CARL keeps policy entropy higher than GRPO throughout training and on test (Fig. 4) — claimed mechanism for OOD gains. Criticality correlates with action type (search/access > read/answer) and early step positions, but not deterministically (Fig. 5), motivating per-state identification.
- **Limitation (App. B):** entropy-as-criticality assumes the model is uncertain where it matters; confidently-wrong states are never explored. Weak models may need SFT cold-start first.
