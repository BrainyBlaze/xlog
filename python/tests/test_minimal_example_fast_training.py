import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyxlog")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../examples/neural/01_minimal"))


def test_addition_sum_distribution_one_hot_pairs():
    from train import addition_sum_distribution

    probs_a = torch.zeros(2, 10)
    probs_b = torch.zeros(2, 10)
    probs_a[0, 3] = 1.0
    probs_b[0, 4] = 1.0
    probs_a[1, 9] = 1.0
    probs_b[1, 0] = 1.0

    sums = addition_sum_distribution(probs_a, probs_b)
    assert sums.shape == (2, 19)
    assert sums[0, 7].item() == pytest.approx(1.0)
    assert sums[1, 9].item() == pytest.approx(1.0)
    assert sums[0].sum().item() == pytest.approx(1.0)
    assert sums[1].sum().item() == pytest.approx(1.0)


def test_addition_nll_loss_near_zero_for_perfect_sum_prediction():
    from train import addition_nll_loss

    probs_a = torch.zeros(1, 10)
    probs_b = torch.zeros(1, 10)
    probs_a[0, 2] = 1.0
    probs_b[0, 5] = 1.0
    targets = torch.tensor([7], dtype=torch.long)

    loss = addition_nll_loss(probs_a, probs_b, targets)
    assert loss.item() < 1e-6


def test_sample_addition_pairs_returns_requested_count_and_bounds():
    from train import sample_addition_pairs

    gen = torch.Generator().manual_seed(0)
    left, right = sample_addition_pairs(16, 64, generator=gen)
    assert left.shape == (64,)
    assert right.shape == (64,)
    assert int(left.min().item()) >= 0
    assert int(right.min().item()) >= 0
    assert int(left.max().item()) < 16
    assert int(right.max().item()) < 16

