"""Unit tests for the within-set rank training gate (`_within_set_st_gate`).

A sibling straight-through gate whose FORWARD is the same hard derivation gate
as ``_st_neural_gate`` (a hard Boolean gate, not soft truth-mass: thinking
proposes, the engine disposes), but whose BACKWARD gradient flows through
``within_set_norm(mode="train")`` -- an offset-invariant within-set z-norm -- so
when phi carries gradient the head/backbone learn the transferable within-set
RANK rather than the non-transferable absolute level.

These validate: (1) forward byte-identical to the existing hard gate, (2)
backward offset-invariant (the discriminating property vs the absolute
per-entity gate), (3) gradient is actually carried, (4) degenerate set ->
neutral, no NaN, (5) the opt-in NeuralBodySpec flag defaults OFF.

Pure torch (CPU); no engine/Rust/GPU.

Run: `PYTHONPATH=crates/pyxlog/python python3 -m pytest python/tests/test_within_set_st_gate.py -q`
or directly: `PYTHONPATH=crates/pyxlog/python python3 python/tests/test_within_set_st_gate.py`
"""

import torch

from pyxlog.ilp.neurosymbolic import (
    NeuralBodySpec,
    _neural_gate_for,
    _st_neural_gate,
    _within_set_st_gate,
)


def test_forward_is_hard_threshold():
    # Forward value must be the hard Boolean sigmoid(logit) >= tau (a derivation
    # GATE, not soft truth-mass) -- thinking proposes, the engine disposes.
    logit = torch.tensor([-2.0, 0.5, 3.0, -0.1])
    grp = torch.zeros(4, dtype=torch.long)
    tau = 0.5
    gate = _within_set_st_gate(logit, tau, grp, 1.0)
    expected_hard = (torch.sigmoid(logit) >= tau).float()
    assert torch.allclose(gate.detach(), expected_hard), gate.detach()


def test_forward_matches_st_neural_gate():
    # the within-set gate is a drop-in on the forward: identical hard gate to the existing
    # _st_neural_gate across varied logits/thresholds (so eval is unaffected).
    torch.manual_seed(0)
    for _ in range(20):
        logit = torch.randn(8) * 5.0
        grp = torch.zeros(8, dtype=torch.long)
        tau = float(torch.rand(1).clamp(0.05, 0.95))
        a = _within_set_st_gate(logit, tau, grp, 1.0).detach()
        b = _st_neural_gate(logit, tau, 1.0, gumbel=False, training=False).detach()
        assert torch.allclose(a, b), (tau, a, b)


def test_backward_offset_invariant():
    # The discriminating property: gradient to the logit is invariant to a
    # set-uniform shift (z-norm subtracts the set mean), whereas the absolute
    # per-entity gate's gradient shifts. This is why this gate trains on RANK.
    grp = torch.zeros(5, dtype=torch.long)
    base = torch.tensor([1.0, 4.0, 2.0, 9.0, 3.0])
    c = 7.0

    def grad_of(gate_fn, shift):
        x = (base + shift).clone().requires_grad_(True)
        gate_fn(x).sum().backward()
        return x.grad.detach().clone()

    g0 = grad_of(lambda x: _within_set_st_gate(x, 0.5, grp, 1.0), 0.0)
    gc = grad_of(lambda x: _within_set_st_gate(x, 0.5, grp, 1.0), c)
    assert torch.allclose(g0, gc, atol=1e-5), (g0, gc)

    # Control: the existing absolute gate is NOT offset-invariant.
    s0 = grad_of(lambda x: _st_neural_gate(x, 0.5, 1.0, False, True), 0.0)
    sc = grad_of(lambda x: _st_neural_gate(x, 0.5, 1.0, False, True), c)
    assert not torch.allclose(s0, sc, atol=1e-5), (s0, sc)


def test_backward_carries_gradient():
    # theta must receive gradient through the gate (it is the training signal).
    grp = torch.zeros(4, dtype=torch.long)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    _within_set_st_gate(x, 0.5, grp, 1.0).sum().backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum()) > 0.0


def test_degenerate_set_neutral_no_nan():
    # A singleton comparison set carries no within-set RANK signal -> neutral
    # (0.5) backward (zero gradient); the forward stays the hard gate; composing
    # it in a loss (as the real loop does) must not crash or NaN.
    grp = torch.zeros(1, dtype=torch.long)
    x = torch.tensor([3.0], requires_grad=True)
    gate = _within_set_st_gate(x, 0.5, grp, 1.0)
    assert torch.isfinite(gate).all()
    assert torch.allclose(gate.detach(), (torch.sigmoid(x.detach()) >= 0.5).float())
    loss = gate.sum() + 0.0 * x.sum()  # grad path exists, as in the joint loss
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) == 0.0  # singleton -> no rank signal


def test_spec_flag_defaults_off():
    # Opt-in fence: the within-set training gate must be OFF by default so existing behavior
    # (the absolute _st_neural_gate train path) is byte-identical unless asked.
    spec = NeuralBodySpec(features=torch.zeros(3, 4))
    assert spec.train_within_set_norm is False


def _spec(flag):
    return NeuralBodySpec(features=torch.zeros(4, 3), train_within_set_norm=flag)


def test_gate_selection_flag_on_training_uses_within_set():
    # Flag on + training -> within-set gate (whole-set group), backward identical
    # to _within_set_st_gate over the whole candidate binding-set.
    logit = torch.tensor([1.0, 4.0, 2.0, 9.0])
    n, device = 4, logit.device
    got = _neural_gate_for(logit, _spec(True), n, device, training=True)
    grp = torch.zeros(n, dtype=torch.long, device=device)
    want = _within_set_st_gate(logit, 0.5, grp, 1.0)
    assert torch.allclose(got, want)


def test_gate_selection_flag_off_uses_st_neural_gate():
    # Flag off -> byte-identical to the existing absolute per-entity gate.
    logit = torch.tensor([1.0, 4.0, 2.0, 9.0])
    got = _neural_gate_for(logit, _spec(False), 4, logit.device, training=True)
    want = _st_neural_gate(logit, 0.5, 1.0, gumbel=False, training=True)
    assert torch.allclose(got, want)


def test_gate_selection_eval_uses_st_neural_gate_even_when_flag_on():
    # the within-set gate is a TRAIN-backward change only; at eval (training=False) the gate
    # is the hard forward via _st_neural_gate -> read path unchanged.
    logit = torch.tensor([1.0, 4.0, 2.0, 9.0])
    got = _neural_gate_for(logit, _spec(True), 4, logit.device, training=False)
    want = _st_neural_gate(logit, 0.5, 1.0, gumbel=False, training=False)
    assert torch.allclose(got, want)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
