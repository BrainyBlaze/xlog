"""Mixed trainable-rule bodies: a neural predicate joined with ordinary relations.

A ``trainable_rule`` body may join a neural predicate with ordinary world
relations (and builtins). The intended semantics:
  - fact atoms (ordinary relations) are HARD join conditions;
  - probability mass comes ONLY from nn-predicates x sigma(w);
  - gradients flow to the network and the rule weight, NEVER through fact atoms.

These tests pin that contract: an ordinary relation in a trainable body acts as
a membership filter on which groundings can fire, contributing no probability
and no gradient.
"""

import pytest

torch = pytest.importorskip("torch")

from pyxlog.ilp.neurosymbolic import (  # noqa: E402
    NeuroSymbolicTrainingConfig,
    _collect_examples,
    _desugar_source,
    _make_rule_weight_module,
    _TENSOR_SOURCE_NAME,
    train_neurosymbolic_program,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="xlog engine requires CUDA"
)

# A neural predicate joined with an ordinary relation `allowed` (a hard
# membership filter) inside one trainable rule.
MIXED_BODY_SOURCE = """
    allowed(0).
    allowed(2).
    pred allowed(i64).
    nn(root_net, [Case], Label, [negative, positive]) :: neural_root(Case, Label).
    trainable_rule(rule_mixed, weight=0.0) :: root_case(Case) :-
        neural_root(Case, positive), allowed(Case).
    train(root_case, binary_cross_entropy).
"""


def _root_net(initial=((0.0,), (0.05,))):
    network = torch.nn.Sequential(
        torch.nn.Linear(1, 2, bias=False),
        torch.nn.Softmax(dim=-1),
    )
    with torch.no_grad():
        network[0].weight.copy_(torch.tensor(initial, dtype=torch.float32))
    return network


def _examples():
    return [
        {
            "inputs": torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32),
            "targets": torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
        }
    ]


@requires_cuda
def test_mixed_body_compiles_and_trains() -> None:
    """A neural predicate joined with an ordinary relation must compile + train,
    with gradients reaching both the network and the rule weight."""
    network = _root_net()
    result = train_neurosymbolic_program(
        MIXED_BODY_SOURCE,
        networks={"root_net": network},
        examples=_examples(),
        config=NeuroSymbolicTrainingConfig(steps=4, learning_rate=0.2),
    )
    assert result.neural_parameter_grads["root_net"] > 0.0
    assert result.symbolic_weight_grads["rule_mixed"] > 0.0


@requires_cuda
def test_ordinary_relation_is_a_hard_join_not_a_probability_source() -> None:
    """`allowed` gates which cases can fire (hard filter); it contributes no
    probability mass. Cases 1 and 3 are NOT in `allowed`, so root_case is false
    for them regardless of the network — query probability ~0 there."""
    network = _root_net()
    result = train_neurosymbolic_program(
        MIXED_BODY_SOURCE,
        networks={"root_net": network},
        examples=_examples(),
        config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.2),
    )
    probs = result.query_probabilities
    assert probs[1] == pytest.approx(0.0, abs=1e-6)
    assert probs[3] == pytest.approx(0.0, abs=1e-6)


@requires_cuda
def test_derived_hard_condition_fails_loud_not_silent() -> None:
    """A hard condition checked against ground facts only must reject a DERIVED
    relation with a typed error, never silently filter every grounding to 0
    (which would corrupt training)."""
    source = """
        base(0).
        base(2).
        pred base(i64).
        pred contested(i64).
        contested(X) :- base(X).
        nn(root_net, [Case], Label, [negative, positive]) :: neural_root(Case, Label).
        trainable_rule(rule_mixed, weight=0.0) :: root_case(Case) :-
            neural_root(Case, positive), contested(Case).
        train(root_case, binary_cross_entropy).
    """
    with pytest.raises(Exception, match="(?i)derived"):
        train_neurosymbolic_program(
            source,
            networks={"root_net": _root_net()},
            examples=_examples(),
            config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.1),
        )


def test_optimizer_config_defaults_to_adam_and_is_selectable() -> None:
    """The training optimizer is configurable and defaults to Adam (SGD stalls on
    the multiplicative-loss plateau; see the separation test)."""
    from pyxlog.ilp.neurosymbolic import _make_optimizer

    assert NeuroSymbolicTrainingConfig().optimizer == "adam"
    params = [torch.zeros(1, requires_grad=True)]
    assert type(_make_optimizer("adam", params, 0.1)).__name__ == "Adam"
    assert type(_make_optimizer("sgd", params, 0.1)).__name__ == "SGD"
    with pytest.raises(ValueError, match="(?i)optimizer"):
        _make_optimizer("rmsprop", params, 0.1)


@requires_cuda
def test_default_adam_separates_linearly_separable_classes() -> None:
    """With the Adam default the training surface escapes the plateau and actually
    LEARNS: a cleanly linearly separable signal (positives low feature, negatives
    high) trains so every positive ranks well above every negative. Plain SGD
    stalled at the uniform-output floor on this exact problem (~1/10 inits); Adam
    separates (~8/10). This pins that the default optimizer makes the surface
    trainable, not just gradient-flowing."""
    torch.manual_seed(1)
    source = """
        nn(root_net, [Case], Label, [negative, positive]) :: neural_root(Case, Label).
        trainable_rule(rule_sep, weight=0.0) :: root_case(Case) :- neural_root(Case, positive).
        train(root_case, binary_cross_entropy).
    """
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    examples = [
        {
            "inputs": torch.tensor(
                [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]], dtype=torch.float32
            ),
            "targets": torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        }
    ]
    result = train_neurosymbolic_program(
        source,
        networks={"root_net": net},
        examples=examples,
        config=NeuroSymbolicTrainingConfig(steps=400, learning_rate=0.1),
    )
    probs = result.query_probabilities
    # Clean separation with a wide margin (positives ~1, negatives ~0).
    assert min(probs[:3]) > max(probs[3:]) + 0.5


def _build_mixed_program():
    """Compile MIXED_BODY_SOURCE and register its networks exactly as the
    wrapper does, returning (program, root_net, queries, expected). lr=0 so the
    program is a fixed point we can evaluate twice without weights drifting."""
    import pyxlog

    program_source, rules, train_head, _objective = _desugar_source(MIXED_BODY_SOURCE)
    inputs, targets = _collect_examples(_examples())

    program = pyxlog.Program.compile(program_source, device=0, memory_mb=4096)
    root_net = _root_net().cuda()
    program.register_network(
        "root_net", root_net, torch.optim.SGD(root_net.parameters(), lr=0.0)
    )
    for rule in rules:
        guard = _make_rule_weight_module(rule.initial_weight).cuda()
        program.register_network(
            rule.guard_network, guard, torch.optim.SGD(guard.parameters(), lr=0.0)
        )
    program.add_tensor_source(_TENSOR_SOURCE_NAME, inputs.cuda())

    queries = [f"{train_head}({i})" for i in range(len(targets))]
    return program, root_net, queries, list(targets)


def _abs_grad_sum(module) -> float:
    return float(
        sum(
            p.grad.detach().abs().sum().item()
            for p in module.parameters()
            if p.grad is not None
        )
    )


@requires_cuda
def test_grouped_matches_scalar_forward_backward() -> None:
    """forward_backward_grouped (the device-resident batched path) must be
    numerically identical to looping the scalar forward_backward over the same
    queries: same summed loss AND same accumulated gradients. This pins that the
    zero-host reroute changed performance, not training semantics."""
    # Batched/grouped path on a fresh program.
    prog_g, net_g, queries, expected = _build_mixed_program()
    prog_g.zero_grad()
    loss_grouped = prog_g.forward_backward_grouped(queries, expected)
    grad_grouped = _abs_grad_sum(net_g)

    # Scalar per-query loop on an identical fresh program (same fixed weights).
    prog_s, net_s, queries_s, expected_s = _build_mixed_program()
    prog_s.zero_grad()
    loss_scalar = sum(
        prog_s.forward_backward(q, t) for q, t in zip(queries_s, expected_s)
    )
    grad_scalar = _abs_grad_sum(net_s)

    assert loss_grouped == pytest.approx(loss_scalar, rel=1e-5, abs=1e-6)
    assert grad_grouped == pytest.approx(grad_scalar, rel=1e-4, abs=1e-6)
    # Gradient must be non-trivial (eligible cases drive it); a zero on both
    # sides would make the equality vacuous.
    assert grad_scalar > 0.0


@requires_cuda
def test_query_probabilities_grouped_matches_scalar() -> None:
    """The batched probability readout must match the per-query scalar readout
    (exp(-forward_backward(q, True))) for every query, including ineligible
    (hard-filtered) ones — so the readout is numerically transparent like the
    loss path while running O(templates) host syncs instead of O(N)."""
    import math

    prog_g, _net_g, queries, _expected = _build_mixed_program()
    prog_g.zero_grad()
    probs_grouped = prog_g.query_probabilities_grouped(queries)
    prog_g.zero_grad()

    prog_s, _net_s, queries_s, _expected_s = _build_mixed_program()
    prog_s.zero_grad()
    probs_scalar = [math.exp(-prog_s.forward_backward(q, True)) for q in queries_s]
    prog_s.zero_grad()

    assert len(probs_grouped) == len(probs_scalar)
    for pg, ps in zip(probs_grouped, probs_scalar):
        assert pg == pytest.approx(ps, rel=1e-5, abs=1e-9)
    # Ineligible (hard-filtered) cases 1 and 3 stay ~0 through the batched path.
    assert probs_grouped[1] == pytest.approx(0.0, abs=1e-6)
    assert probs_grouped[3] == pytest.approx(0.0, abs=1e-6)


@requires_cuda
def test_training_loop_has_no_tracked_host_transfers() -> None:
    """The warm training loop must perform NO tracked device<->host transfers in
    either direction. After the one-time cache warm-up, every step runs
    device-resident: no downloads (loss accumulates on device) and no uploads
    (query-var metadata is cached on device). dtoh AND htod, calls AND bytes,
    all stay at their reset baseline of 0 across the measured loop."""
    result = train_neurosymbolic_program(
        MIXED_BODY_SOURCE,
        networks={"root_net": _root_net()},
        examples=_examples(),
        config=NeuroSymbolicTrainingConfig(steps=3, learning_rate=0.2),
    )
    stats = result.training_host_transfer_stats
    assert stats is not None
    assert stats["dtoh_calls"] == 0
    assert stats["dtoh_bytes"] == 0
    assert stats["htod_calls"] == 0
    assert stats["htod_bytes"] == 0
