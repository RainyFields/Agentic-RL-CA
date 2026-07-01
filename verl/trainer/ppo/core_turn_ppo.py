# Copyright 2026 — SP3: true turn-level PPO for verl-agent ALFWorld.
#
# Cross-turn Generalized Advantage Estimation for the step-independent multi-turn rollout, where
# each turn is a separate training sample (obs->response). Uses a learned critic's per-turn scalar
# value V(s_t) (the value head at the last prompt token = response position 0) and the immediate
# per-turn env reward r_t to compute A_t = GAE(gamma, lambda) across the turns of each trajectory,
# then broadcasts the turn advantage/return to every response token of that turn.
#
# Robust to ray_trainer's adjust_batch duplication: trajectories are grouped by traj_uid and ordered
# by an explicit per-turn turn_index (not row order); the terminal turn is max(turn_index).
import logging
import numpy as np
import torch
import verl.utils.torch_functional as verl_F  # masked_whiten

logger = logging.getLogger(__name__)


def compute_gae_turn_advantage_return(
    values: torch.Tensor,            # (B, T_resp) per-token critic values (already mask-aligned)
    response_mask: torch.Tensor,     # (B, T_resp)
    rewards: np.ndarray,             # (B,) immediate per-turn env reward (object/float array)
    traj_uid: np.ndarray,            # (B,) trajectory id (shared across a trajectory's turns)
    turn_index: np.ndarray,          # (B,) step index within trajectory (=_step)
    is_action_valid: np.ndarray,     # (B,) bool
    gamma: float,
    lam: float,
    invalid_action_penalty_coef: float = 0.0,
    use_invalid_action_penalty: bool = False,
    max_turns: int = 50,
    env_done: np.ndarray = None,     # (B,) bool: True env-terminal at that turn (vs rollout truncation)
):
    """Turn-level GAE. Returns (advantages, returns), each (B, T_resp), matching the contract used by
    compute_advantage downstream (`data.batch['advantages'|'returns'] = ...`)."""
    # --- defensive assertions (toy-run guards) ---
    assert values.shape == response_mask.shape, (values.shape, response_mask.shape)
    assert (response_mask[:, 0] == 1).all(), "first response token must be present (values[:,0]=V(s_t))"
    device, dtype, B = values.device, values.dtype, values.shape[0]

    # scalar V(s_t): value head at the last prompt token == response position 0
    V = values[:, 0].float().detach().cpu().numpy()                  # (B,)
    r = np.asarray(rewards, dtype=np.float32).copy()                 # (B,)
    if use_invalid_action_penalty:
        r -= invalid_action_penalty_coef * (1.0 - np.asarray(is_action_valid, dtype=np.float32))

    turn_index = np.asarray(turn_index)
    env_done_arr = None if env_done is None else np.asarray(env_done).astype(bool)
    A_scalar = np.zeros(B, dtype=np.float32)
    ret_scalar = np.zeros(B, dtype=np.float32)
    n_truncated = 0

    for uid in np.unique(traj_uid):
        idx = np.where(traj_uid == uid)[0]                          # rows of this traj (may incl. dups)
        ti = turn_index[idx].astype(np.int64)
        order = np.argsort(ti, kind="stable")                       # step order
        uniq_ti, first = np.unique(ti[order], return_index=True)    # dedup, keep first occurrence
        rep = idx[order][first]                                     # representative row per unique turn
        assert 1 <= len(uniq_ti) <= max_turns, (str(uid), len(uniq_ti))

        # Terminal vs truncation: bootstrap V_next=0 only on a true env terminal. If the last turn is
        # a rollout truncation (max_turns / early-stop), correct GAE would bootstrap V(s_next); the
        # step-independent rollout has no turn-sample for that cut next-state, so bootstrap is NOT yet
        # supported -> warn + fall back to V_next=0 (TODO: add an extra critic forward on s_next).
        if env_done_arr is not None and not bool(env_done_arr[rep[-1]]):
            n_truncated += 1
        A_next, V_next = 0.0, 0.0                                   # terminal bootstrap = 0
        adv_by_turn, ret_by_turn = {}, {}
        for k in range(len(uniq_ti) - 1, -1, -1):                  # reverse over turns
            v_t = float(V[rep[k]])
            r_t = float(r[rep[k]])
            delta = r_t + gamma * V_next - v_t                     # V_next=0 at true terminal turn
            A_t = delta + gamma * lam * A_next
            ret_t = A_t + v_t
            assert abs(ret_t - (A_t + v_t)) < 1e-5
            adv_by_turn[int(uniq_ti[k])] = A_t
            ret_by_turn[int(uniq_ti[k])] = ret_t
            A_next, V_next = A_t, v_t

        for j in idx:                                              # rebroadcast to ALL rows (incl dups)
            A_scalar[j] = adv_by_turn[int(turn_index[j])]
            ret_scalar[j] = ret_by_turn[int(turn_index[j])]

    if n_truncated:
        n_traj = len(np.unique(traj_uid))
        logger.warning(
            "[gae_turn] %d/%d trajectories are rollout-truncated (max_turns/early-stop), NOT env "
            "terminal. Bootstrap on truncation is unsupported (no turn-sample for the cut next-state); "
            "treating as terminal V_next=0. TODO: add a critic forward on s_next to bootstrap.",
            n_truncated, n_traj,
        )

    A = torch.tensor(A_scalar, device=device, dtype=dtype).unsqueeze(-1)
    R = torch.tensor(ret_scalar, device=device, dtype=dtype).unsqueeze(-1)
    advantages = A * response_mask                                 # turn adv broadcast to its tokens
    returns = R * response_mask                                    # critic target (NOT whitened)
    advantages = verl_F.masked_whiten(advantages, response_mask)   # whiten AFTER broadcast (token-level; matches stock GAE)
    return advantages, returns
