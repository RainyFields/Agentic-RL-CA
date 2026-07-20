# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Wave-2 unit checks — turn-GRPO estimator (GiGPO's anchor-state step component only):
toy-batch advantage values under mean_std_norm, singleton anchor group -> advantage 0,
identity turn_grpo == gigpo - episode_norm_reward, mode routing, std=0 no-NaN.
Run: python -m pytest credit_assignment/tests/test_turn_grpo.py -q  (CPU only)
"""
import numpy as np
import pytest
import torch

from gigpo.core_gigpo import (
    compute_gigpo_outcome_advantage,
    compute_turn_grpo_outcome_advantage,
    episode_norm_reward,
)

T_RESP = 4


def make_inputs(rows):
    """rows: list of (uid, traj, anchor, step_reward). token_level_rewards puts the step
    reward on the last response token (only its sum matters to episode_norm_reward)."""
    bs = len(rows)
    step_rewards = torch.tensor([r[3] for r in rows], dtype=torch.float32)
    response_mask = torch.ones(bs, T_RESP)
    token_level_rewards = torch.zeros(bs, T_RESP)
    token_level_rewards[:, -1] = step_rewards
    anchor_obs = np.array([r[2] for r in rows], dtype=object)
    index = np.array([r[0] for r in rows], dtype=object)
    traj_index = np.array([r[1] for r in rows], dtype=object)
    return token_level_rewards, step_rewards, response_mask, anchor_obs, index, traj_index


def test_shared_anchor_mean_std_norm_values():
    # two turns share an anchor state within one uid group, rewards-to-go {1, 0}
    tlr, sr, mask, anchor, idx, tidx = make_inputs([
        ("g1", "t1", "obs_A", 1.0),
        ("g1", "t2", "obs_A", 0.0),
    ])
    adv, ret = compute_turn_grpo_outcome_advantage(sr, mask, anchor, idx, mode="mean_std_norm")
    # mean 0.5, torch.std (unbiased) = 0.7071...
    expect = 0.5 / (torch.std(torch.tensor([1.0, 0.0])) + 1e-6)
    assert adv.shape == (2, T_RESP)
    assert torch.allclose(adv[0], torch.full((T_RESP,), expect), atol=1e-4)
    assert torch.allclose(adv[1], -torch.full((T_RESP,), expect), atol=1e-4)
    assert torch.equal(adv, ret)


def test_singleton_anchor_group_zero():
    tlr, sr, mask, anchor, idx, tidx = make_inputs([
        ("g1", "t1", "obs_A", 1.0),
        ("g1", "t2", "obs_B", 0.7),   # unique anchor -> singleton group
    ])
    adv, _ = compute_turn_grpo_outcome_advantage(sr, mask, anchor, idx, mode="mean_std_norm")
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6)


def test_identity_gigpo_minus_episode_term():
    rows = [
        ("g1", "t1", "obs_A", 1.0), ("g1", "t2", "obs_A", 0.0),
        ("g1", "t1", "obs_B", 1.0), ("g1", "t2", "obs_B", 0.0),
        ("g2", "t3", "obs_C", 0.5), ("g2", "t4", "obs_C", 0.0),
    ]
    tlr, sr, mask, anchor, idx, tidx = make_inputs(rows)
    gigpo_adv, _ = compute_gigpo_outcome_advantage(
        tlr, sr, mask, anchor, idx, tidx, step_advantage_w=1.0, mode="mean_std_norm")
    episode_adv = episode_norm_reward(tlr, mask, idx, tidx, 1e-6, False)
    turn_adv, _ = compute_turn_grpo_outcome_advantage(sr, mask, anchor, idx, mode="mean_std_norm")
    assert torch.allclose(turn_adv, gigpo_adv - episode_adv, atol=1e-5)


def test_mode_routing():
    tlr, sr, mask, anchor, idx, tidx = make_inputs([
        ("g1", "t1", "obs_A", 1.0),
        ("g1", "t2", "obs_A", 0.0),
    ])
    adv_std, _ = compute_turn_grpo_outcome_advantage(sr, mask, anchor, idx, mode="mean_std_norm")
    adv_mean, _ = compute_turn_grpo_outcome_advantage(sr, mask, anchor, idx, mode="mean_norm")
    assert torch.allclose(adv_mean[0], torch.full((T_RESP,), 0.5), atol=1e-6)
    assert not torch.allclose(adv_std, adv_mean, atol=1e-4)
    with pytest.raises(ValueError):
        compute_turn_grpo_outcome_advantage(sr, mask, anchor, idx, mode="bogus")


def test_constant_group_no_nan():
    tlr, sr, mask, anchor, idx, tidx = make_inputs([
        ("g1", "t1", "obs_A", 0.3),
        ("g1", "t2", "obs_A", 0.3),
        ("g1", "t3", "obs_A", 0.3),
    ])
    adv, _ = compute_turn_grpo_outcome_advantage(sr, mask, anchor, idx, mode="mean_std_norm")
    assert torch.isfinite(adv).all()
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-5)


def test_respects_response_mask():
    tlr, sr, mask, anchor, idx, tidx = make_inputs([
        ("g1", "t1", "obs_A", 1.0),
        ("g1", "t2", "obs_A", 0.0),
    ])
    mask[0, 2:] = 0.0
    adv, _ = compute_turn_grpo_outcome_advantage(sr, mask, anchor, idx, mode="mean_std_norm")
    assert (adv[0, 2:] == 0.0).all() and (adv[0, :2] != 0.0).all()
