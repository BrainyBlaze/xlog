"""Regression coverage for UCR-XLOG-001.

The contract under test: ``train_neurosymbolic_program`` must run the REAL
xlog engine — source parsed by the native parser, the symbolic part evaluated
through the compiled circuit on GPU, and gradients flowing through actual rule
inference. A scalar-bias surrogate must fail this suite:

* probabilities must match exact circuit conjunction semantics;
* the trainable rule BODY must determine the loss (mutation tests);
* invalid xlog must be rejected by the real parser;
* unsupported body shapes must fail closed with a typed engine error;
* proof-level credit assignment must come from DifferentiableProofTraceMap.
"""

import math

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from pyxlog.ilp.neurosymbolic import (
    NeuroSymbolicTrainingConfig,
    train_neurosymbolic_program,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="xlog engine requires CUDA"
)

CANONICAL_SOURCE = """
        nn(root_net, [Case], Label, [negative, positive]) :: neural_root(Case, Label).
        trainable_rule(rule_primary, weight=0.0) :: root_case(Case) :- neural_root(Case, positive).
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


def test_joint_training_trains_neural_and_symbolic_weights() -> None:
    network = _root_net()
    initial_weight = network[0].weight.detach().clone()

    result = train_neurosymbolic_program(
        CANONICAL_SOURCE,
        networks={"root_net": network},
        examples=_examples(),
        config=NeuroSymbolicTrainingConfig(steps=8, learning_rate=0.2),
    )

    assert result.engine == "xlog-exact-circuit"
    assert result.neural_parameter_grads["root_net"] > 0.0
    assert result.symbolic_weight_grads["rule_primary"] > 0.0
    assert not torch.equal(network[0].weight.detach().cpu(), initial_weight)
    # Learned rule weight is a probability, moved off its sigmoid(0)=0.5 init.
    learned = result.symbolic_rule_weights["rule_primary"]
    assert 0.0 < learned < 1.0
    assert learned != pytest.approx(0.5)
    assert len(result.losses) == 8
    assert result.losses[-1] < result.losses[0]
    assert len(result.query_probabilities) == 4

    inventory = result.learned_rule_inventory.to_dict()
    assert inventory["selected_clauses"][0]["id"] == "rule_primary"
    assert inventory["selected_clauses"][0]["neural_predicate"] == "neural_root"
    assert inventory["training_objective"] == "binary_cross_entropy"


def test_engine_probability_matches_circuit_semantics() -> None:
    """The decisive anti-surrogate test.

    With frozen parameters (lr=0), the reported query probability must equal
    the exact d-DNNF conjunction:  P(root_case(i)) = p_net(i)[positive] * sigmoid(w0).
    A logit-sum surrogate cannot reproduce this identity.
    """
    network = _root_net()
    w0 = 0.7

    source = CANONICAL_SOURCE.replace("weight=0.0", f"weight={w0}")
    result = train_neurosymbolic_program(
        source,
        networks={"root_net": network},
        examples=_examples(),
        config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.0),
    )

    inputs = _examples()[0]["inputs"]
    with torch.no_grad():
        p_positive = network(inputs.cuda().reshape(-1, 1))[:, 1].cpu()
    p_guard = 1.0 / (1.0 + math.exp(-w0))

    for i in range(4):
        expected = float(p_positive[i]) * p_guard
        assert result.query_probabilities[i] == pytest.approx(expected, abs=1e-5)


def test_rule_body_mutation_changes_training() -> None:
    """Changing ONLY the rule body must change the loss (body participates)."""
    losses = {}
    for label in ("positive", "negative"):
        torch.manual_seed(7)
        network = _root_net(initial=((0.3,), (-0.2,)))
        source = CANONICAL_SOURCE.replace(
            "neural_root(Case, positive)", f"neural_root(Case, {label})"
        )
        result = train_neurosymbolic_program(
            source,
            networks={"root_net": network},
            examples=_examples(),
            config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.1),
        )
        losses[label] = result.losses[0]

    assert losses["positive"] != pytest.approx(losses["negative"], abs=1e-9)


def test_multi_literal_trainable_rule_body() -> None:
    """Conjunction of two neural atoms: P = p1 * p2 * w, evaluated by the circuit."""
    network = _root_net()
    gate = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=False), torch.nn.Softmax(dim=-1))
    with torch.no_grad():
        gate[0].weight.copy_(torch.tensor([[0.1], [0.4]], dtype=torch.float32))

    source = """
        nn(root_net, [Case], Label, [negative, positive]) :: neural_root(Case, Label).
        nn(gate_net, [Case], Mode, [closed, open]) :: neural_gate(Case, Mode).
        trainable_rule(rule_joint, weight=0.0) :: root_case(Case) :-
            neural_root(Case, positive), neural_gate(Case, open).
        train(root_case, binary_cross_entropy).
    """
    result = train_neurosymbolic_program(
        source,
        networks={"root_net": network, "gate_net": gate},
        examples=_examples(),
        config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.0),
    )

    inputs = _examples()[0]["inputs"].cuda().reshape(-1, 1)
    with torch.no_grad():
        p1 = network(inputs)[:, 1].cpu()
        p2 = gate(inputs)[:, 1].cpu()

    for i in range(4):
        expected = float(p1[i]) * float(p2[i]) * 0.5  # sigmoid(0) = 0.5
        assert result.query_probabilities[i] == pytest.approx(expected, abs=1e-5)


def test_real_parser_rejects_invalid_xlog() -> None:
    """A malformed clause anywhere in the source must fail at real parse time."""
    source = CANONICAL_SOURCE + "\n        broken(X :- oops.\n"
    with pytest.raises(Exception) as excinfo:
        train_neurosymbolic_program(
            source,
            networks={"root_net": _root_net()},
            examples=_examples(),
            config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.1),
        )
    # The error must come from the engine's parser, not from a regex template.
    assert "trainable_rule" not in str(excinfo.value)


def test_existential_join_hard_condition_fails_closed() -> None:
    """A hard condition that joins an ordinary relation on an existential
    (non-head) variable is not yet supported: typed error, never a
    silently-wrong probability. Head-variable hard conditions ARE supported
    (see test_mixed_trainable_rule_bodies.py)."""
    source = """
        nn(root_net, [Case], Label, [negative, positive]) :: neural_root(Case, Label).
        link(0, 9).
        pred link(i64, i64).
        trainable_rule(rule_mixed, weight=0.0) :: root_case(Case) :-
            neural_root(Case, positive), link(Case, Other).
        train(root_case, binary_cross_entropy).
    """
    with pytest.raises(Exception, match="(?i)existential"):
        train_neurosymbolic_program(
            source,
            networks={"root_net": _root_net()},
            examples=_examples(),
            config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.1),
        )


def test_missing_network_validated_against_real_declarations() -> None:
    with pytest.raises(ValueError, match="missing network"):
        train_neurosymbolic_program(
            CANONICAL_SOURCE,
            networks={},
            examples=_examples(),
            config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.1),
        )


def test_proof_trace_credit_assignment() -> None:
    """Credit assignment must go through the native DifferentiableProofTraceMap."""
    result = train_neurosymbolic_program(
        CANONICAL_SOURCE,
        networks={"root_net": _root_net()},
        examples=_examples(),
        config=NeuroSymbolicTrainingConfig(steps=2, learning_rate=0.2),
    )

    traces = result.proof_trace_map.traces()
    # One trace per (example, trainable rule).
    assert len(traces) == 4
    clause_ids = {trace["clause_id"] for trace in traces}
    assert clause_ids == {"rule_primary"}
    answer_keys = {trace["answer_key"] for trace in traces}
    assert answer_keys == {f"root_case({i})" for i in range(4)}
    # Gradients were accumulated against the actual training targets.
    assert any(abs(trace["gradient"]) > 0.0 for trace in traces)


def test_rule_inventory_selection_is_weight_driven() -> None:
    """Inventory status must reflect the learned weight, not be hardcoded."""
    for w0, expected_status in ((5.0, "selected"), (-5.0, "rejected")):
        source = CANONICAL_SOURCE.replace("weight=0.0", f"weight={w0}")
        result = train_neurosymbolic_program(
            source,
            networks={"root_net": _root_net()},
            examples=_examples(),
            config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.0),
        )
        inventory = result.learned_rule_inventory.to_dict()
        all_clauses = inventory["selected_clauses"] + inventory.get("rejected_clauses", [])
        clause = next(c for c in all_clauses if c["id"] == "rule_primary")
        assert clause["status"] == expected_status


def test_objective_and_examples_validation() -> None:
    bad_objective = CANONICAL_SOURCE.replace("binary_cross_entropy", "hinge")
    with pytest.raises(ValueError, match="unsupported training objective"):
        train_neurosymbolic_program(
            bad_objective,
            networks={"root_net": _root_net()},
            examples=_examples(),
        )
    with pytest.raises(ValueError, match="examples"):
        train_neurosymbolic_program(
            CANONICAL_SOURCE,
            networks={"root_net": _root_net()},
            examples=[],
        )
