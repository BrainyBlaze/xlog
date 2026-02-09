import subprocess
import sys
from pathlib import Path

import pytest

from neural_test_env import runtime_env

EXAMPLES = [
    (
        "examples/neural/02_coins/train.py",
        Path("examples/neural/02_coins/data/coins/train"),
    ),
    (
        "examples/neural/03_mnist_multidigit/train.py",
        Path("examples/neural/03_mnist_multidigit/data/svhn/train/digitStruct.mat"),
    ),
    (
        "examples/neural/04_hwf/train.py",
        Path("examples/neural/04_hwf/data/crohme"),
    ),
    (
        "examples/neural/05_poker/train.py",
        Path("examples/neural/05_poker/data/cards"),
    ),
    (
        "examples/neural/06_clutrr/train.py",
        Path("examples/neural/06_clutrr/data/clutrr/train.jsonl"),
    ),
]


RUNTIME_ENV = runtime_env()


def _run(script: str, threshold: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            script,
            "--mode",
            "ci",
            "--epochs",
            "1",
            "--min-accuracy",
            str(threshold),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=RUNTIME_ENV,
    )


@pytest.mark.parametrize("script,marker", EXAMPLES)
def test_examples_report_final_metric(script: str, marker: Path):
    if not marker.exists():
        pytest.skip(f"dataset missing for {script}")

    result = _run(script, threshold=0.0)
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, combined
    assert "FINAL_METRIC" in combined


@pytest.mark.parametrize("script,marker", EXAMPLES)
def test_examples_fail_accuracy_gate_when_threshold_too_high(script: str, marker: Path):
    if not marker.exists():
        pytest.skip(f"dataset missing for {script}")

    result = _run(script, threshold=1.1)
    combined = f"{result.stdout}\n{result.stderr}".lower()
    assert result.returncode != 0
    assert "accuracy gate failed" in combined
