import subprocess
import sys
from pathlib import Path
import pytest
torch = pytest.importorskip("torch")

from neural_test_env import runtime_env


RUNTIME_ENV = runtime_env()
TRAIN_SCRIPT = Path("examples/neural/02_coins/train.py")

if not TRAIN_SCRIPT.exists():
    pytest.skip(f"Missing example script: {TRAIN_SCRIPT}", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


def test_coins_ci_mode():
    data_root = Path("examples/neural/02_coins/data/coins")
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--mode", "ci"],
        capture_output=True,
        text=True,
        check=False,
        env=RUNTIME_ENV,
    )
    if data_root.exists():
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "Dataset missing" in (result.stderr + result.stdout)
