import subprocess
import sys
from pathlib import Path

from neural_test_env import runtime_env


RUNTIME_ENV = runtime_env()


def test_mnist_multidigit_ci_mode():
    marker = Path("examples/neural/03_mnist_multidigit/data/svhn/train/digitStruct.mat")
    result = subprocess.run(
        [sys.executable, "examples/neural/03_mnist_multidigit/train.py", "--mode", "ci"],
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
