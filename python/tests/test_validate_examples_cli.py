import subprocess
import sys


def test_validate_examples_help():
    result = subprocess.run(
        [sys.executable, "scripts/validate_examples.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--mode" in result.stdout
