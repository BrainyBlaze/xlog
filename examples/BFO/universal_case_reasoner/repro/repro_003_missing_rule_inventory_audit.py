from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT / "crates" / "pyxlog" / "python"))

from pyxlog.ilp.inventory import build_rule_inventory  # noqa: E402
from pyxlog.ilp.types import CandidateMapEntry, LearnedArtifact  # noqa: E402


def main() -> None:
    artifact = LearnedArtifact(
        candidate_map=[
            CandidateMapEntry(0, 0, 0, 1, "edge", "edge", "reach"),
            CandidateMapEntry(1, 1, 1, 1, "skip", "skip", "reach"),
        ],
        logits=[2.0, -1.0],
        soft_probs=[0.88, 0.12],
        selected_hard=[0],
        discovered_rule="reach(X, Y) :- edge(X, Z), edge(Z, Y).",
    )
    inventory = build_rule_inventory(
        artifact,
        training_fold="train-fold-0",
        held_out_domains=("cybersecurity_intrusion",),
        base_kernel_checksum_before="sha256:stable",
        base_kernel_checksum_after="sha256:stable",
    )
    payload = {
        "finding": "UCR-XLOG-003",
        "resolved": True,
        "api": "pyxlog.ilp.inventory.build_rule_inventory",
        "inventory": inventory.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
