from __future__ import annotations

import subprocess


def test_preflight_release_publish_help() -> None:
    proc = subprocess.run(
        ["bash", "scripts/preflight_release_publish.sh", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "usage: scripts/preflight_release_publish.sh" in proc.stdout
    assert "--dry-run" in proc.stdout


def test_preflight_release_publish_dry_run() -> None:
    proc = subprocess.run(
        ["bash", "scripts/preflight_release_publish.sh", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "python3 scripts/validate_package_metadata.py" in proc.stdout
    assert (
        "cargo package --locked --allow-dirty --no-verify --list -p xlog-cuda"
        in proc.stdout
    )
    assert (
        "cargo package --locked --allow-dirty --no-verify --list -p xlog-cli"
        in proc.stdout
    )
    assert "Dry run complete." in proc.stdout
