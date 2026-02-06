import subprocess
import sys
from pathlib import Path


def test_hwf_ci_mode():
    data_root = Path("examples/neural/04_hwf/data/crohme")
    result = subprocess.run(
        [sys.executable, "examples/neural/04_hwf/train.py", "--mode", "ci"],
        capture_output=True,
        text=True,
        check=False,
    )
    if data_root.exists():
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert "CROHME dataset missing" in (result.stderr + result.stdout)
