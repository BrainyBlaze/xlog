from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT / "crates" / "pyxlog" / "python"))

from pyxlog.transfer_diagnostics import (  # noqa: E402
    PredictionRecord,
    compute_transfer_diagnostics,
)


def main() -> None:
    diagnostics = compute_transfer_diagnostics(
        [
            PredictionRecord("cyber", "clean", 1, 1, baseline_pred=0),
            PredictionRecord("cyber", "clean", 0, 0, baseline_pred=0),
            PredictionRecord("medical", "clean", 1, 1, baseline_pred=0),
            PredictionRecord("medical", "adversarial", 1, 0, baseline_pred=0),
        ],
        required_domains=("cyber", "medical"),
        required_variants=("clean", "adversarial"),
        bootstrap_samples=16,
        seed=7,
    )
    payload = {
        "finding": "UCR-XLOG-006",
        "resolved": diagnostics.passed,
        "api": "pyxlog.transfer_diagnostics.compute_transfer_diagnostics",
        "macro_f1": diagnostics.macro_f1,
        "minimum_domain_f1": diagnostics.minimum_domain_f1,
        "baseline_uplift": diagnostics.baseline_uplift,
        "confidence_interval": {
            "lower": diagnostics.bootstrap_ci.lower,
            "upper": diagnostics.bootstrap_ci.upper,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
