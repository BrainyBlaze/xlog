import subprocess
import sys
from pathlib import Path

from neural_test_env import runtime_env


RUNTIME_ENV = runtime_env()


def test_poker_ci_mode():
    data_root = Path("examples/neural/05_poker/data/cards")
    result = subprocess.run(
        [sys.executable, "examples/neural/05_poker/train.py", "--mode", "ci"],
        capture_output=True,
        text=True,
        check=False,
        env=RUNTIME_ENV,
    )
    if data_root.exists():
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "Card dataset missing" in (result.stderr + result.stdout)
