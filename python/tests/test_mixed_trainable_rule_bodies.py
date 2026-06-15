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
