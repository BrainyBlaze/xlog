import subprocess
import sys
from pathlib import Path
import pytest

from neural_test_env import runtime_env


RUNTIME_ENV = runtime_env()
TRAIN_SCRIPT = Path("examples/neural/03_mnist_multidigit/train.py")

if not TRAIN_SCRIPT.exists():
    pytest.skip(f"Missing example script: {TRAIN_SCRIPT}", allow_module_level=True)


def test_mnist_multidigit_ci_mode():
    marker = Path("examples/neural/03_mnist_multidigit/data/svhn/train/digitStruct.mat")
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--mode", "ci"],
        capture_output=True,
        text=True,
        check=False,
        env=RUNTIME_ENV,
    )
    if marker.exists():
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "SVHN data missing" in (result.stderr + result.stdout)
