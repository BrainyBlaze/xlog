import subprocess
import sys
from pathlib import Path


def test_poker_ci_mode():
    data_root = Path("examples/neural/05_poker/data/cards")
    result = subprocess.run(
        [sys.executable, "examples/neural/05_poker/train.py", "--mode", "ci"],
        capture_output=True,
        text=True,
        check=False,
    )
    if data_root.exists():
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "Card dataset missing" in (result.stderr + result.stdout)
