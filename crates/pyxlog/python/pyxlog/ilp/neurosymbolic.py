"""Joint neural-predicate and symbolic-rule training helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pyxlog.ilp.inventory import RuleInventory, RuleInventoryClause


@dataclass(frozen=True)
class NeuroSymbolicTrainingConfig:
    """Training knobs for a small declarative nn/4 plus rule-weight program."""

    steps: int = 1
    learning_rate: float = 0.1


@dataclass(frozen=True)
class NeuralPredicateDecl:
    network_name: str
    predicate_name: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class TrainableRuleDecl:
    id: str
    head_predicate: str
    neural_predicate: str
    neural_label: str
    initial_weight: float
    source: str


@dataclass
class NeuroSymbolicTrainingResult:
    neural_parameter_grads: dict[str, float]
    symbolic_weight_grads: dict[str, float]
    symbolic_rule_weights: dict[str, float]
    learned_rule_inventory: RuleInventory
    losses: list[float]


def train_neurosymbolic_program(
    source: str,
    *,
    networks: dict[str, Any],
    examples: list[dict[str, Any]],
    config: NeuroSymbolicTrainingConfig = NeuroSymbolicTrainingConfig(),
) -> NeuroSymbolicTrainingResult:
    """Train neural predicate parameters and symbolic rule weights together."""

    import torch
    import torch.nn.functional as F

    parsed = _parse_neurosymbolic_source(source)
    neural_decl = parsed["neural"]
    rules = parsed["rules"]
    objective = parsed["objective"]
    if objective != "binary_cross_entropy":
        raise ValueError(f"unsupported training objective: {objective}")
    if neural_decl.network_name not in networks:
        raise ValueError(f"missing network for nn/4 predicate: {neural_decl.network_name}")
    if not examples:
        raise ValueError("examples must not be empty")

    network = networks[neural_decl.network_name]
    rule_params = {
        rule.id: torch.nn.Parameter(torch.tensor(float(rule.initial_weight)))
        for rule in rules
    }
    parameters = list(network.parameters()) + list(rule_params.values())
    optimizer = torch.optim.SGD(parameters, lr=config.learning_rate)
    losses: list[float] = []

    neural_grads: dict[str, float] = {neural_decl.network_name: 0.0}
    symbolic_grads: dict[str, float] = {rule.id: 0.0 for rule in rules}
    label_to_index = {label: idx for idx, label in enumerate(neural_decl.labels)}

    for _step in range(config.steps):
        optimizer.zero_grad()
        loss_terms = []
        for batch in examples:
            inputs = batch["inputs"]
            targets = batch["targets"].to(dtype=torch.float32)
            logits = network(inputs)
            combined = torch.zeros_like(targets)
            for rule in rules:
                label_index = label_to_index[rule.neural_label]
                neural_score = logits[:, label_index]
                combined = combined + neural_score + rule_params[rule.id]
            loss_terms.append(F.binary_cross_entropy_with_logits(combined, targets))
        loss = torch.stack(loss_terms).mean()
        loss.backward()

        neural_grads[neural_decl.network_name] = float(
            sum(
                param.grad.detach().abs().sum().item()
                for param in network.parameters()
                if param.grad is not None
            )
        )
        symbolic_grads = {
            rule_id: float(param.grad.detach().abs().item())
            for rule_id, param in rule_params.items()
            if param.grad is not None
        }
        optimizer.step()
        losses.append(float(loss.detach().item()))

    selected = [
        RuleInventoryClause(
            id=rule.id,
            rule=rule.source,
            status="selected",
            weight=float(rule_params[rule.id].detach().item()),
            neural_predicate=rule.neural_predicate,
        )
        for rule in rules
    ]
    inventory = RuleInventory(
        selected_clauses=selected,
        training_objective=objective,
    )
    return NeuroSymbolicTrainingResult(
        neural_parameter_grads=neural_grads,
        symbolic_weight_grads=symbolic_grads,
        symbolic_rule_weights={
            rule_id: float(param.detach().item())
            for rule_id, param in rule_params.items()
        },
        learned_rule_inventory=inventory,
        losses=losses,
    )


def _parse_neurosymbolic_source(source: str) -> dict[str, Any]:
    neural_match = re.search(
        r"nn\(\s*(?P<network>\w+)\s*,\s*\[[^\]]*\]\s*,\s*\w+\s*,\s*"
        r"\[(?P<labels>[^\]]+)\]\s*\)\s*::\s*"
        r"(?P<predicate>\w+)\s*\([^)]*\)\s*\.",
        source,
    )
    if neural_match is None:
        raise ValueError("source must declare one nn/4 neural predicate")
    labels = tuple(label.strip() for label in neural_match.group("labels").split(","))
    neural = NeuralPredicateDecl(
        network_name=neural_match.group("network"),
        predicate_name=neural_match.group("predicate"),
        labels=labels,
    )

    rules: list[TrainableRuleDecl] = []
    for match in re.finditer(
        r"trainable_rule\(\s*(?P<id>\w+)(?:\s*,\s*weight\s*=\s*(?P<weight>[-0-9.]+))?"
        r"\s*\)\s*::\s*(?P<head>\w+)\s*\([^)]*\)\s*:-\s*"
        r"(?P<neural>\w+)\s*\([^,]+,\s*(?P<label>\w+)\s*\)\s*\.",
        source,
    ):
        rules.append(
            TrainableRuleDecl(
                id=match.group("id"),
                head_predicate=match.group("head"),
                neural_predicate=match.group("neural"),
                neural_label=match.group("label"),
                initial_weight=float(match.group("weight") or 0.0),
                source=match.group(0).split("::", maxsplit=1)[1].strip(),
            )
        )
    if not rules:
        raise ValueError("source must declare at least one trainable_rule")
    for rule in rules:
        if rule.neural_predicate != neural.predicate_name:
            raise ValueError(
                f"trainable rule {rule.id} references unknown neural predicate "
                f"{rule.neural_predicate}"
            )
        if rule.neural_label not in neural.labels:
            raise ValueError(
                f"trainable rule {rule.id} references unknown label {rule.neural_label}"
            )

    objective_match = re.search(r"train\(\s*\w+\s*,\s*(?P<objective>\w+)\s*\)\s*\.", source)
    objective = objective_match.group("objective") if objective_match else "binary_cross_entropy"
    return {"neural": neural, "rules": rules, "objective": objective}
