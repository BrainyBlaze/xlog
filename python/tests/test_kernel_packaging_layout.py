from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_repo_does_not_require_tracked_ptx_files() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--", "kernels/*.ptx"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    assert tracked == [], f"tracked PTX files should be removed: {tracked}"


def test_stage_kernels_help_works() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/stage_kernels.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "usage:" in result.stdout.lower()
    assert "--from-out-dir" in result.stdout
    assert "--to" in result.stdout
