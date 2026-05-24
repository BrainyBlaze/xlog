from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT / "crates" / "pyxlog" / "python"))

import torch  # noqa: E402
from pyxlog.runtime_audit import CudaExecutionAudit, HostMaterializationError  # noqa: E402


def main() -> None:
    audit = CudaExecutionAudit(forbid_host_materialization=True)
    caught = False
    try:
        with audit:
            scores = torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32)
            audit.record_nn4_scores("production_root_net", scores, device_resident=True)
            scores.tolist()
    except HostMaterializationError:
        caught = True

    summary = audit.summary()
    payload = {
        "finding": "UCR-XLOG-004",
        "resolved": caught and not summary.passed,
        "api": "pyxlog.runtime_audit.CudaExecutionAudit",
        "summary": {
            "d2h_transfers": summary.d2h_transfers,
            "h2d_transfers": summary.h2d_transfers,
            "scalar_extractions": summary.scalar_extractions,
            "score_row_downloads": summary.score_row_downloads,
            "violations": [violation.operation for violation in summary.violations],
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
