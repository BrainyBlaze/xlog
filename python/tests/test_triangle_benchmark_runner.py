from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT
    / "paper"
    / "artifacts"
    / "head-to-head"
    / "runners"
    / "triangle_counting_vs_souffle.py"
)


def test_runner_self_test_has_no_benchmark_runtime_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "-S", RUNNER, "--self-test"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "triangle benchmark runner self-test passed"
