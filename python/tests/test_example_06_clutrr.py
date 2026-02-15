import subprocess
import sys
from pathlib import Path
import pytest
torch = pytest.importorskip("torch")

from neural_test_env import runtime_env


RUNTIME_ENV = runtime_env()
TRAIN_SCRIPT = Path("examples/neural/06_clutrr/train.py")

if not TRAIN_SCRIPT.exists():
    pytest.skip(f"Missing example script: {TRAIN_SCRIPT}", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


def test_clutrr_ci_mode():
    dataset_file = Path("examples/neural/06_clutrr/data/clutrr/train.jsonl")
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--mode", "ci"],
        capture_output=True,
        text=True,
        check=False,
        env=RUNTIME_ENV,
    )
    if dataset_file.exists():
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "CLUTRR dataset missing" in (result.stderr + result.stdout)
