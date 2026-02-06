import subprocess
import sys
from pathlib import Path


def test_clutrr_ci_mode():
    dataset_file = Path("examples/neural/06_clutrr/data/clutrr/train.jsonl")
    result = subprocess.run(
        [sys.executable, "examples/neural/06_clutrr/train.py", "--mode", "ci"],
        capture_output=True,
        text=True,
        check=False,
    )
    if dataset_file.exists():
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "CLUTRR dataset missing" in (result.stderr + result.stdout)
