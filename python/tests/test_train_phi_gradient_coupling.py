"""Unit tests for the phi-gradient coupling opt-in (`_phi_for_training`).

When ``NeuralBodySpec.train_phi_gradient`` is set, the per-binding entity feature
phi(x) is NOT detached on the training upload, so gradient flows into the
entity-feature producer (backbone coupling). Default off detaches phi (no gradient
past the neural head), byte-identical to the existing training path.

This is the engine-side (xlog) half of phi-gradient coupling; the producer-side
points live in the downstream backbone repository and land separately. The flag is
opt-in and the branch stays unmerged until a cross-context measurement confirms a
payoff. Pure torch (CPU); no engine/Rust/GPU.
"""

import torch

from pyxlog.ilp.neurosymbolic import NeuralBodySpec, _phi_for_training


def test_default_detaches_phi_no_gradient():
    # Default (flag off): phi is detached -> no gradient flows past the neural head.
    phi = torch.randn(3, 4, requires_grad=True)
    spec = NeuralBodySpec(features=phi)
    assert spec.train_phi_gradient is False
    out = _phi_for_training(spec)
    assert out.requires_grad is False
    assert torch.equal(out, phi.detach())


def test_flag_on_keeps_phi_gradient_flowing():
    # Flag on: phi stays attached -> backward reaches phi (the producer coupling).
    phi = torch.randn(3, 4, requires_grad=True)
    spec = NeuralBodySpec(features=phi, train_phi_gradient=True)
    out = _phi_for_training(spec)
    assert out.requires_grad is True
    out.sum().backward()
    assert phi.grad is not None
    assert float(phi.grad.abs().sum()) > 0.0


def test_flag_only_changes_autograd_linkage_not_values():
    # Un-detaching changes ONLY the autograd graph, never the uploaded values, so the
    # forward/training numerics are identical to the detached path.
    phi = torch.randn(5, 2, requires_grad=True)
    on = _phi_for_training(NeuralBodySpec(features=phi, train_phi_gradient=True))
    off = _phi_for_training(NeuralBodySpec(features=phi, train_phi_gradient=False))
    assert torch.equal(on.detach(), off)


if __name__ == "__main__":
    test_default_detaches_phi_no_gradient()
    test_flag_on_keeps_phi_gradient_flowing()
    test_flag_only_changes_autograd_linkage_not_values()
    print("3/3 passed")
