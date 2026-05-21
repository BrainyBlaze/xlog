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
            "module_boundary_diagnostics",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    diagnostic_source = (
        REPO_ROOT / "crates" / "xlog-logic" / "src" / "module_diagnostics.rs"
    ).read_text(encoding="utf-8")
    payload = {
        "finding": "UCR-XLOG-005",
        "resolved": proc.returncode == 0,
        "api": "xlog_logic::diagnose_module_boundaries",
        "has_frozen_kernel_diagnostic": "FrozenKernelMutation" in diagnostic_source,
        "has_held_out_leakage_diagnostic": "HeldOutLeakage" in diagnostic_source,
        "test_stdout": proc.stdout,
        "test_stderr": proc.stderr,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
