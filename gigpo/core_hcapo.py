# SP4 — HCAPO (Hindsight Credit Assignment Policy Optimization), arXiv 2603.08754.
# Value-free turn-level credit assignment on group sampling (GRPO-like). Two parts:
#   macro = group-normalized trajectory outcome (the GRPO term)
#   micro = group-normalized hindsight-refined step return Q^H = rho * G, where rho is the
#           sharpened/clipped likelihood ratio of the action under a prompt augmented with the
#           trajectory's FINAL state (hindsight) vs the on-policy prompt.
# A = (R-muR)/sigmaR + omega * (Q^H-muH)/sigmaH , broadcast to response tokens. Optional temporal
# smoothing of Q^H across adjacent turns. No critic.
import os as _os
import numpy as np
import torch
from verl.utils.model import compute_position_id_with_mask


def build_hindsight_batch(data, tokenizer, max_prompt_length, max_response_length,
                          hindsight_prefix="\nThe episode's final observation was: ",
                          s_final_source="last_obs"):
    """DataProto whose prompts have s_final appended before the SAME response. Feed to
    actor_rollout_wg.compute_log_prob to get pi_hind.

    s_final_source (config: +algorithm.hcapo.s_final_source; default preserves prior behaviour):
      "last_obs"     -- the trajectory's final observation anchor (the last retrieved-docs block in
                        SearchQA). Wave-3 diagnostic showed this LONG injection creates the length
                        channel in the hindsight lift (corr(dm,len)=+0.43 pre-drift).
      "final_answer" -- the agent's own final answer string (last <answer>..</answer> of the final
                        turn; falls back to the final response text, then the anchor). A ~few-token
                        injection: tests whether shrinking the perturbation removes the drift lever.
                        Prefix becomes "The agent's final answer was: ". NB for successful
                        trajectories (the only ones where rho matters, since R=0 zeroes Q^H) the
                        agent's answer ~= the gold answer by EM.
    """
    from verl import DataProto
    prompts = data.batch['prompts']            # (B, Lp) left-padded
    responses = data.batch['responses']        # (B, Lr) right-padded
    attn = data.batch['attention_mask']        # (B, Lp+Lr)
    Lp, Lr = prompts.size(1), responses.size(1)
    B = prompts.size(0)
    device = prompts.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    anchor = np.asarray(data.non_tensor_batch['anchor_obs'])
    traj_uid = np.asarray(data.non_tensor_batch['traj_uid'])
    turn_index = np.asarray(data.non_tensor_batch['turn_index']).astype(np.int64)
    if s_final_source == "final_answer":
        import re as _re
        hindsight_prefix = "\nThe agent's final answer was: "
        _ans_re = _re.compile(r"<answer>(.*?)</answer>", _re.DOTALL)
    sfinal = {}
    for uid in np.unique(traj_uid):
        idx = np.where(traj_uid == uid)[0]
        last = idx[np.argmax(turn_index[idx])]
        if s_final_source == "final_answer":
            _resp_ids = responses[last][attn[last, -Lr:].bool()]
            _txt = tokenizer.decode(_resp_ids, skip_special_tokens=True)
            _m = _ans_re.findall(_txt)
            sfinal[uid] = (_m[-1].strip() if _m else _txt.strip()[-256:]) or str(anchor[last])
        else:
            sfinal[uid] = str(anchor[last])

    prompt_attn = attn[:, :Lp]
    resp_attn = attn[:, -Lr:]
    new_input, new_attn, new_resp = [], [], []
    for i in range(B):
        p_valid = prompts[i][prompt_attn[i].bool()]
        s_ids = tokenizer.encode(hindsight_prefix + sfinal[traj_uid[i]], add_special_tokens=False)
        s_ids = torch.tensor(s_ids[:max(0, max_prompt_length - 1)], dtype=prompts.dtype, device=device)
        new_p = torch.cat([p_valid, s_ids])[-max_prompt_length:]
        padp = max_prompt_length - new_p.size(0)
        p_part = torch.cat([torch.full((padp,), pad_id, dtype=prompts.dtype, device=device), new_p])
        p_mask = torch.cat([torch.zeros(padp, dtype=attn.dtype, device=device),
                            torch.ones(new_p.size(0), dtype=attn.dtype, device=device)])
        nr = int(resp_attn[i].sum())
        r_mask = torch.cat([torch.ones(nr, dtype=attn.dtype, device=device),
                            torch.zeros(Lr - nr, dtype=attn.dtype, device=device)])
        new_input.append(torch.cat([p_part, responses[i]]))
        new_attn.append(torch.cat([p_mask, r_mask]))
        new_resp.append(responses[i])

    input_ids = torch.stack(new_input)
    attention_mask = torch.stack(new_attn)
    position_ids = compute_position_id_with_mask(attention_mask)
    out = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids,
           'responses': torch.stack(new_resp), 'prompts': input_ids[:, :max_prompt_length]}
    hb = DataProto.from_single_dict(out)
    hb.meta_info = dict(data.meta_info)
    return hb


def _group_norm(x, groups, eps=1e-6):
    out = np.zeros_like(x, dtype=np.float32)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        v = x[idx]; mu = v.mean(); sd = v.std()
        out[idx] = (v - mu) / (sd + eps) if sd > eps else (v - mu)
    return out


def compute_hcapo_advantage(policy_log_probs, hindsight_log_probs, response_mask,
                            rewards, uid, traj_uid, turn_index,
                            gamma=0.95, omega=1.0, t_temp=5.0, clip_lo=0.8, clip_hi=1.2,
                            temporal_alpha=0.5, use_temporal=True, eps=1e-3, adv_clip=5.0,
                            rho_denominator="intra_traj_mean", rho_score="hind"):
    """Returns (advantages, returns) each (B, T_resp). Value-free (returns mirrors advantages).

    rho_denominator:
      "intra_traj_mean" (DEFAULT, paper-correct, arXiv:2603.08754 Eq. 6-7)
          pi_hind(a_t) = exp( (1/(T_temp*|a_t|)) * sum_j log pi(y_j | y_<j, s_t, s_final) )
          rho_t        = clip( pi_hind(a_t) / mean_k pi_hind(a_k), C_min, C_max )
          The denominator is the INTRA-TRAJECTORY MEAN over that trajectory's turns -- the paper's
          "group-normalization across actions within the same episode". rho is centred on 1.0 by
          construction (pivotal turns > 1, redundant turns < 1), which is what makes the symmetric
          [0.8, 1.2] clip well-specified.
      "on_policy" (LEGACY -- reproduces this repo's pre-2026-07-29 behaviour; DO NOT use for new
          science) divided by the on-policy prob of the same action, i.e. rho = pi_hind/pi_policy.
          That measures the off-distribution penalty of injecting s_final into the prompt, so it is
          systematically < 1 (the upper clip never binds) and, because only the numerator carries
          the 1/|a_t| normalisation, it makes rho rise as responses lengthen -- a structural
          verbosity incentive that produced a length runaway (clip -> 0.78) in wave-3.
    """
    device, dtype = policy_log_probs.device, policy_log_probs.dtype
    m = response_mask.float()
    ntok = m.sum(-1).clamp(min=1.0)

    r = np.asarray(rewards, dtype=np.float32)
    uid = np.asarray(uid); traj_uid = np.asarray(traj_uid)
    turn_index = np.asarray(turn_index).astype(np.int64)

    if rho_denominator == "on_policy":
        mean_logratio = ((hindsight_log_probs - policy_log_probs) * m).sum(-1) / ntok
        rho = torch.exp(mean_logratio / t_temp).clamp(clip_lo, clip_hi).detach().cpu().numpy()
    else:
        if rho_score == "lift":
            # Wave-4 (user 2026-07-31): score = the per-token-mean LIFT
            #   m_t = (1/|a_t|) sum_j [log pi(y_j|.., s_final) - log pi(y_j|..)]
            # then Eq. 7's intra-trajectory normalisation on top. Motivation: subtract the shared
            # per-token predictability profile. NB the offline diagnostic (estdiag_*.json, rho_dm)
            # measured this exact estimator: corr(lift, len) = +0.43/+0.77 (the injection-dilution
            # channel survives the subtraction) and Spearman 0.95-0.97 vs the Eq.6 score -- so the
            # pre-registered prediction is that it re-ignites the drift on last_obs and tracks the
            # damped trajectory on final_answer. This arm tests the DYNAMICS of that prediction.
            log_pi_hind = (((hindsight_log_probs - policy_log_probs) * m).sum(-1)
                           / ntok / t_temp).detach().cpu().numpy()
        else:
            # Eq. (6): per-token-normalised, temperature-sharpened hindsight log-prob. The POLICY
            # log-probs play no part in rho under the paper's formulation.
            log_pi_hind = ((hindsight_log_probs * m).sum(-1) / ntok / t_temp).detach().cpu().numpy()
        # Eq. (7): self-normalise by the mean over the TURNS of the same trajectory. Averaging is
        # over turns, so rows sharing a turn_index are deduped to one representative first.
        rho = np.ones_like(log_pi_hind, dtype=np.float32)
        for tu in np.unique(traj_uid):
            idx = np.where(traj_uid == tu)[0]
            ti = turn_index[idx]
            _, first = np.unique(ti, return_index=True)
            rep = idx[first]                                  # one row per distinct turn
            x = log_pi_hind[idx]
            xm = log_pi_hind[rep].max()                       # shift for numerical stability
            denom = np.exp(log_pi_hind[rep] - xm).mean()      # == mean_k pi_hind(a_k) / e^xm
            rho[idx] = np.exp(x - xm) / max(float(denom), eps)
        rho = np.clip(rho, clip_lo, clip_hi)

    B = policy_log_probs.shape[0]
    R_traj = np.zeros(B, dtype=np.float32)
    QH = np.zeros(B, dtype=np.float32)

    for tu in np.unique(traj_uid):
        idx = np.where(traj_uid == tu)[0]
        ti = turn_index[idx]
        order = np.argsort(ti, kind="stable")
        uniq, first = np.unique(ti[order], return_index=True)
        rep = idx[order][first]
        Rt = float(sum(r[x] for x in rep))
        Tn = len(uniq)
        qvals = np.array([rho[rep[k]] * (gamma ** (Tn - 1 - k)) * Rt for k in range(Tn)], dtype=np.float32)
        if use_temporal and Tn >= 2:
            sm = qvals.copy()
            for k in range(Tn):
                lo, hi = max(0, k - 1), min(Tn - 1, k + 1)
                neigh = np.mean([qvals[j] for j in (lo, hi) if j != k]) if hi != lo else qvals[k]
                sm[k] = (1 - temporal_alpha) * qvals[k] + temporal_alpha * neigh
            qvals = sm
        by_R = {int(uniq[k]): Rt for k in range(Tn)}
        by_Q = {int(uniq[k]): float(qvals[k]) for k in range(Tn)}
        for j in idx:
            R_traj[j] = by_R[int(turn_index[j])]
            QH[j] = by_Q[int(turn_index[j])]

    # Wave-3: keep the two terms separate so the instrumentation below can report their relative
    # magnitudes (macro == the GRPO term, micro == the hindsight term).
    A_macro = _group_norm(R_traj, uid, eps)
    A_micro = _group_norm(QH, uid, eps)
    A = A_macro + omega * A_micro
    # SP6: bound the advantage — a group with near-equal outcomes (tiny std) yields an extreme
    # normalized advantage that detonates the PPO update (entropy explodes, grounding->0). Clip it.
    if adv_clip and adv_clip > 0:
        A = np.clip(A, -adv_clip, adv_clip)
    # Wave-3 rho instrumentation (HCAPO_LOG_RHO=1 by default; additive, HCAPO code path only).
    # Answers whether the hindsight signal does real work or is squeezed toward a constant by
    # t_temp + the rho clip -- in which case Q^H degenerates to a positional discount
    # gamma^(T-1-k)*R and HCAPO is ~GRPO plus a reweighting, regardless of EM.
    if _os.environ.get("HCAPO_LOG_RHO", "1") == "1" and rho.size:
        _n = float(rho.size)
        print(
            f"[hcapo-rho] mean={rho.mean():.4f} std={rho.std():.4f} min={rho.min():.4f} "
            f"max={rho.max():.4f} frac_at_lo={(rho <= clip_lo + 1e-6).sum() / _n:.3f} "
            f"frac_at_hi={(rho >= clip_hi - 1e-6).sum() / _n:.3f} "
            f"abs_macro={np.abs(A_macro).mean():.4f} abs_micro={np.abs(A_micro).mean():.4f} "
            f"omega={omega} frac_adv_clipped={(np.abs(A) >= adv_clip - 1e-6).sum() / _n:.3f}",
            flush=True,
        )
    A_t = torch.tensor(A, device=device, dtype=dtype).unsqueeze(-1)
    advantages = A_t * response_mask
    return advantages, advantages
