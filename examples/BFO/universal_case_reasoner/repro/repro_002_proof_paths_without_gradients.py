from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]


def main() -> None:
    proc = subprocess.run(
        [
            "cargo",
            "test",
            "-q",
            "-p",
            "xlog-logic",
            "--test",
            "differentiable_proof_trace",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    proof_trace_source = (
        REPO_ROOT / "crates" / "xlog-logic" / "src" / "proof_trace.rs"
    ).read_text(encoding="utf-8")
    payload = {
        "finding": "UCR-XLOG-002",
        "resolved": proc.returncode == 0,
        "api": "xlog_logic::DifferentiableProofTraceMap",
        "stable_proof_ids": "stable_proof_id" in proof_trace_source,
        "gradient_surface": "accumulate_binary_logistic_gradients" in proof_trace_source,
        "test_stdout": proc.stdout,
        "test_stderr": proc.stderr,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
