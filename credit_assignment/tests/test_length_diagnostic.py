# Copyright 2026 Agentic-RL-CA project. Apache-2.0.
"""Phase 3c unit checks: effective loss weight and correlations on hand-built cases."""
import pytest
import torch

from credit_assignment.length_diagnostic import length_weight_metrics


def test_uniform_broadcast_weights_scale_with_length():
    # gae_turn-style: uniform |A|=1 per turn; turn0 len 2, turn1 len 8 -> weights 0.2 / 0.8
    adv = torch.ones(2, 8)
    mask = torch.zeros(2, 8)
    mask[0, :2] = 1
    mask[1, :] = 1
    m = length_weight_metrics(adv, mask)
    assert m["lengthdiag/n_turns"] == 2
    assert m["lengthdiag/eff_loss_weight/p10"] == pytest.approx(0.2 * (2 / 2) / 10 * 10, rel=0.5)
    # exact check via median endpoints: weights are {2/10, 8/10}
    assert m["lengthdiag/eff_loss_weight/p90"] <= 0.8 + 1e-6
    assert m["lengthdiag/corr_len_effweight"] == pytest.approx(1.0, abs=1e-5)
    assert m["lengthdiag/corr_len_absadv"] == pytest.approx(0.0, abs=1e-5)  # |A| constant


def test_zero_mask_rows_excluded():
    adv = torch.randn(3, 4)
    mask = torch.zeros(3, 4)
    mask[0] = 1
    m = length_weight_metrics(adv, mask)
    assert m["lengthdiag/n_turns"] == 1
