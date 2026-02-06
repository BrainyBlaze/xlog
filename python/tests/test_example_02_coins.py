import subprocess
import sys
from pathlib import Path


def test_coins_ci_mode():
    data_root = Path("examples/neural/02_coins/data/coins")
    result = subprocess.run(
        [sys.executable, "examples/neural/02_coins/train.py", "--mode", "ci"],
        capture_output=True,
        text=True,
        check=False,
    )
    if data_root.exists():
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "Dataset missing" in (result.stderr + result.stdout)
