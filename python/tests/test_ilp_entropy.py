# python/tests/test_ilp_entropy.py
"""Tests for entropy regularization."""
import pytest

torch = pytest.importorskip("torch")


def test_normalized_entropy_uniform():
    from pyxlog.ilp.entropy import normalized_entropy
    C = 10
    logits = torch.zeros(C)
    probs = torch.softmax(logits, dim=0)
    h = normalized_entropy(probs, C)
    assert abs(h.item() - 1.0) < 1e-5


def test_normalized_entropy_one_hot():
    from pyxlog.ilp.entropy import normalized_entropy
    C = 10
    logits = torch.full((C,), -100.0)
    logits[0] = 100.0
    probs = torch.softmax(logits, dim=0)
    h = normalized_entropy(probs, C)
    assert h.item() < 0.01


def test_normalized_entropy_gradient_flows():
    from pyxlog.ilp.entropy import normalized_entropy
    C = 10
    logits = torch.randn(C, requires_grad=True)
    probs = torch.softmax(logits, dim=0)
    h = normalized_entropy(probs, C)
    h.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_normalized_entropy_singleton_returns_zero():
    from pyxlog.ilp.entropy import normalized_entropy
    probs = torch.tensor([1.0])
    h = normalized_entropy(probs, C=1)
    assert h.item() == 0.0


def test_entropy_weight_decay():
    from pyxlog.ilp.entropy import entropy_weight_at_step
    w0 = entropy_weight_at_step(0, 100, start=0.1, end=0.0)
    w50 = entropy_weight_at_step(50, 100, start=0.1, end=0.0)
    w100 = entropy_weight_at_step(100, 100, start=0.1, end=0.0)
    assert abs(w0 - 0.1) < 1e-6
    assert abs(w50 - 0.05) < 1e-6
    assert abs(w100 - 0.0) < 1e-6
