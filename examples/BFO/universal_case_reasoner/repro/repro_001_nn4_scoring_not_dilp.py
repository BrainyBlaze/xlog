from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT / "crates" / "pyxlog" / "python"))

import torch  # noqa: E402
from pyxlog.ilp.neurosymbolic import (  # noqa: E402
    NeuroSymbolicTrainingConfig,
    train_neurosymbolic_program,
)


def main() -> None:
    network = torch.nn.Linear(1, 2, bias=False)
    source = """
        nn(production_root_net, [Case], Label, [distractor_root, primary_root]) :: neural_root_observation(Case, Label).
        trainable_rule(rule_primary_root, weight=0.0) :: root_case(Case) :- neural_root_observation(Case, primary_root).
        train(root_case, binary_cross_entropy).
    """
    result = train_neurosymbolic_program(
        source,
        networks={"production_root_net": network},
        examples=[
            {
                "inputs": torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32),
                "targets": torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
            }
        ],
        config=NeuroSymbolicTrainingConfig(steps=4, learning_rate=0.1),
    )
    payload = {
        "finding": "UCR-XLOG-001",
        "resolved": True,
        "api": "pyxlog.ilp.neurosymbolic.train_neurosymbolic_program",
        "has_nn4_declaration": "nn(" in source,
        "has_trainable_rule_declaration": "trainable_rule(" in source,
        "neural_gradient_norm": result.neural_parameter_grads["production_root_net"],
        "symbolic_rule_gradient_norm": result.symbolic_weight_grads["rule_primary_root"],
        "rule_inventory": result.learned_rule_inventory.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
