import os
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../examples/neural/01_minimal"))


def test_compute_improvement_percent_single_epoch_returns_none():
    from train import compute_improvement_percent

    assert compute_improvement_percent([2.5]) is None


def test_compute_improvement_percent_two_epochs_returns_percent():
    from train import compute_improvement_percent

    assert compute_improvement_percent([2.0, 1.0]) == pytest.approx(50.0)

