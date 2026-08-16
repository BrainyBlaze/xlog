from __future__ import annotations

from pathlib import Path
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


def test_cuda_ci_pins_cuda_12_8_torch_wheel() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workflow = (repo_root / ".github/workflows/cuda-ci.yml").read_text()
    torch_installs = [
        line.strip()
        for line in workflow.splitlines()
        if line.strip().startswith("python -m pip install") and "torch" in line
    ]

    assert "python -m pip install --upgrade pip maturin pytest pytest-timeout\n" in workflow
    assert torch_installs == [
        "python -m pip install torch==2.11.0 "
        "--index-url https://download.pytorch.org/whl/cu128"
    ]
