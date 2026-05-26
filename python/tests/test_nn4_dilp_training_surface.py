"""Regression coverage for UCR-XLOG-001."""

import pytest

torch = pytest.importorskip("torch")

from pyxlog.ilp.neurosymbolic import (
    NeuroSymbolicTrainingConfig,
    train_neurosymbolic_program,
)


def test_nn4_and_symbolic_rule_weight_train_in_one_program() -> None:
    network = torch.nn.Linear(1, 2, bias=False)
    with torch.no_grad():
        network.weight.copy_(torch.tensor([[0.0], [0.05]], dtype=torch.float32))
    initial_weight = network.weight.detach().clone()

    source = """
        nn(root_net, [Case], Label, [negative, positive]) :: neural_root(Case, Label).
        trainable_rule(rule_primary, weight=0.0) :: root_case(Case) :- neural_root(Case, positive).
        train(root_case, binary_cross_entropy).
    """
    examples = [
        {
            "inputs": torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32),
            "targets": torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
        }
    ]

    result = train_neurosymbolic_program(
        source,
        networks={"root_net": network},
        examples=examples,
        config=NeuroSymbolicTrainingConfig(steps=8, learning_rate=0.2),
    )

    assert result.neural_parameter_grads["root_net"] > 0.0
    assert result.symbolic_weight_grads["rule_primary"] > 0.0
    assert not torch.equal(network.weight.detach(), initial_weight)
    assert result.symbolic_rule_weights["rule_primary"] != pytest.approx(0.0)

    inventory = result.learned_rule_inventory.to_dict()
    assert inventory["selected_clauses"][0]["id"] == "rule_primary"
    assert inventory["selected_clauses"][0]["neural_predicate"] == "neural_root"
    assert inventory["training_objective"] == "binary_cross_entropy"
