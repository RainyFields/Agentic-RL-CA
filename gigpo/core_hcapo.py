# SP4 — HCAPO (Hindsight Credit Assignment Policy Optimization), arXiv 2603.08754.
# Value-free turn-level credit assignment on group sampling (GRPO-like). Two parts:
#   macro = group-normalized trajectory outcome (the GRPO term)
#   micro = group-normalized hindsight-refined step return Q^H = rho * G, where rho is the
#           sharpened/clipped likelihood ratio of the action under a prompt augmented with the
#           trajectory's FINAL state (hindsight) vs the on-policy prompt.
# A = (R-muR)/sigmaR + omega * (Q^H-muH)/sigmaH , broadcast to response tokens. Optional temporal
# smoothing of Q^H across adjacent turns. No critic.
import numpy as np
import torch
from verl.utils.model import compute_position_id_with_mask


def build_hindsight_batch(data, tokenizer, max_prompt_length, max_response_length,
                          hindsight_prefix="\nThe episode's final observation was: "):
    """DataProto whose prompts have s_final (the trajectory's last observation) appended before the
    SAME response. Feed to actor_rollout_wg.compute_log_prob to get pi_hind."""
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
    sfinal = {}
    for uid in np.unique(traj_uid):
        idx = np.where(traj_uid == uid)[0]
        last = idx[np.argmax(turn_index[idx])]
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
                            temporal_alpha=0.5, use_temporal=True, eps=1e-6):
    """Returns (advantages, returns) each (B, T_resp). Value-free (returns mirrors advantages)."""
    device, dtype = policy_log_probs.device, policy_log_probs.dtype
    m = response_mask.float()
    ntok = m.sum(-1).clamp(min=1.0)
    mean_logratio = ((hindsight_log_probs - policy_log_probs) * m).sum(-1) / ntok
    rho = torch.exp(mean_logratio / t_temp).clamp(clip_lo, clip_hi).detach().cpu().numpy()

    r = np.asarray(rewards, dtype=np.float32)
    uid = np.asarray(uid); traj_uid = np.asarray(traj_uid)
    turn_index = np.asarray(turn_index).astype(np.int64)
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

    A = _group_norm(R_traj, uid, eps) + omega * _group_norm(QH, uid, eps)
    A_t = torch.tensor(A, device=device, dtype=dtype).unsqueeze(-1)
    advantages = A_t * response_mask
    return advantages, advantages
