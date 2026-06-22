"""UCR-XLOG-001 minimal reproducer — real-engine joint neuro-symbolic training.

Resolution is asserted BEHAVIORALLY, not by string containment:

1. circuit-semantics identity: with frozen parameters, the engine-reported
   query probability equals the exact d-DNNF conjunction
   ``p_net(case)[label] * sigmoid(rule_weight)``;
2. body participation: mutating ONLY the trainable rule body changes the loss;
3. joint gradients: both the nn/4 network and the symbolic rule weight receive
   non-zero gradients from the same circuit evaluation;
4. proof-level credit assignment is exported via DifferentiableProofTraceMap.

Requires the installed pyxlog wheel (native engine) and a CUDA device.
"""

from __future__ import annotations

import json
import math

import torch

from pyxlog.ilp.neurosymbolic import (
    NeuroSymbolicTrainingConfig,
    train_neurosymbolic_program,
)

SOURCE = """
    nn(production_root_net, [Case], Label, [distractor_root, primary_root]) :: neural_root_observation(Case, Label).
    trainable_rule(rule_primary_root, weight=0.0) :: root_case(Case) :- neural_root_observation(Case, primary_root).
    train(root_case, binary_cross_entropy).
"""


def make_network() -> torch.nn.Module:
    network = torch.nn.Sequential(
        torch.nn.Linear(1, 2, bias=False),
        torch.nn.Softmax(dim=-1),
    )
    with torch.no_grad():
        network[0].weight.copy_(torch.tensor([[0.0], [0.05]], dtype=torch.float32))
    return network


def make_examples() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "inputs": torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32),
            "targets": torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
        }
    ]


def main() -> None:
    # --- 1. Circuit-semantics identity (frozen parameters) -------------------
    frozen_net = make_network()
    frozen = train_neurosymbolic_program(
        SOURCE,
        networks={"production_root_net": frozen_net},
        examples=make_examples(),
        config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.0),
    )
    inputs = make_examples()[0]["inputs"].cuda()
    with torch.no_grad():
        p_primary = frozen_net(inputs)[:, 1].cpu()
    guard = 1.0 / (1.0 + math.exp(-0.0))
    semantics_abs_error = max(
        abs(frozen.query_probabilities[i] - float(p_primary[i]) * guard)
        for i in range(4)
    )
    assert semantics_abs_error < 1e-5, semantics_abs_error

    # --- 2. Body participation (mutation check) ------------------------------
    losses = {}
    for label in ("primary_root", "distractor_root"):
        mutated = SOURCE.replace(
            "neural_root_observation(Case, primary_root)",
            f"neural_root_observation(Case, {label})",
        )
        result = train_neurosymbolic_program(
            mutated,
            networks={"production_root_net": make_network()},
            examples=make_examples(),
            config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.1),
        )
        losses[label] = result.losses[0]
    assert abs(losses["primary_root"] - losses["distractor_root"]) > 1e-9

    # --- 3. Joint gradients through the circuit ------------------------------
    trained = train_neurosymbolic_program(
        SOURCE,
        networks={"production_root_net": make_network()},
        examples=make_examples(),
        config=NeuroSymbolicTrainingConfig(steps=4, learning_rate=0.1),
    )
    assert trained.engine == "xlog-exact-circuit"
    assert trained.neural_parameter_grads["production_root_net"] > 0.0
    assert trained.symbolic_weight_grads["rule_primary_root"] > 0.0

    # --- 4. Proof-level credit assignment ------------------------------------
    traces = trained.proof_trace_map.traces()
    assert len(traces) == 4
    assert any(abs(trace["gradient"]) > 0.0 for trace in traces)

    payload = {
        "finding": "UCR-XLOG-001",
        "resolved": True,
        "api": "pyxlog.ilp.neurosymbolic.train_neurosymbolic_program",
        "engine": trained.engine,
        "circuit_semantics_max_abs_error": semantics_abs_error,
        "body_mutation_loss_delta": abs(
            losses["primary_root"] - losses["distractor_root"]
        ),
        "neural_gradient_norm": trained.neural_parameter_grads["production_root_net"],
        "symbolic_rule_gradient_norm": trained.symbolic_weight_grads[
            "rule_primary_root"
        ],
        "learned_rule_weight": trained.symbolic_rule_weights["rule_primary_root"],
        "proof_trace_count": len(traces),
        "rule_inventory": trained.learned_rule_inventory.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
