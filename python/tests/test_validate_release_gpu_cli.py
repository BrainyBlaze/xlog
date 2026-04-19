from __future__ import annotations

import subprocess


def test_validate_release_gpu_help() -> None:
    proc = subprocess.run(
        ["bash", "scripts/validate_release_gpu.sh", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "usage: scripts/validate_release_gpu.sh" in proc.stdout
    assert "--mode smoke|release" in proc.stdout


def test_validate_release_gpu_dry_run() -> None:
    proc = subprocess.run(
        ["bash", "scripts/validate_release_gpu.sh", "--mode", "release", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "scripts/xlog_doctor.py --workflow release" in proc.stdout
    assert "cargo test -p xlog-cuda-tests --test certification_suite" in proc.stdout
    assert "Dry run complete." in proc.stdout
