import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import sync_readme_release_version as syncer


def _readme(version: str) -> str:
    return f"""
# xlog

![version](https://img.shields.io/badge/version-v{version}-blue.svg)

Release status: `v{version}`
"""


def test_sync_readme_version_updates_badge_and_release_status() -> None:
    updated = syncer.sync_readme_version(_readme("0.5.0"), "0.5.1")

    assert "version-v0.5.1-blue.svg" in updated
    assert "Release status: `v0.5.1`" in updated
    assert "version-v0.5.0-blue.svg" not in updated


def test_sync_readme_version_updates_bold_release_status_label() -> None:
    updated = syncer.sync_readme_version(
        """
# xlog

![version](https://img.shields.io/badge/version-v0.5.0-blue.svg)

> **Release status:** `v0.5.0` — details
""",
        "0.5.1",
    )

    assert "**Release status:** `v0.5.1`" in updated


def test_sync_readme_version_supports_prerelease_and_build_metadata() -> None:
    updated = syncer.sync_readme_version(
        _readme("0.5.0-alpha.1"),
        "0.5.1-alpha.1+cu13",
    )

    assert "version-v0.5.1-alpha.1+cu13-blue.svg" in updated
    assert "Release status: `v0.5.1-alpha.1+cu13`" in updated


def test_sync_readme_version_requires_known_patterns() -> None:
    try:
        syncer.sync_readme_version("# xlog\n", "0.5.1")
    except ValueError as exc:
        assert "README version badge" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing README patterns")


def test_sync_readme_version_rejects_ambiguous_badge_matches() -> None:
    readme = f"{_readme('0.5.0')}\n![version](https://img.shields.io/badge/version-v0.5.0-blue.svg)\n"

    try:
        syncer.sync_readme_version(readme, "0.5.1")
    except ValueError as exc:
        assert "README version badge" in str(exc)
    else:
        raise AssertionError("expected ValueError for ambiguous README badge")


def test_sync_readme_version_rejects_ambiguous_release_status_matches() -> None:
    readme = f"{_readme('0.5.0')}\nRelease status: `v0.5.0`\n"

    try:
        syncer.sync_readme_version(readme, "0.5.1")
    except ValueError as exc:
        assert "README release status line" in str(exc)
    else:
        raise AssertionError(
            "expected ValueError for ambiguous README release status line"
        )


def test_sync_readme_release_version_script_runs_as_direct_entrypoint() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        readme = tmpdir / "README.md"
        cargo = tmpdir / "Cargo.toml"

        readme.write_text(_readme("0.5.0-alpha.1"), encoding="utf-8")
        cargo.write_text(
            '[workspace.package]\nversion = "0.5.1-alpha.1+cu13"\n',
            encoding="utf-8",
        )

        proc = subprocess.run(
            [
                sys.executable,
                "scripts/sync_readme_release_version.py",
                "--readme",
                str(readme),
                "--cargo",
                str(cargo),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        updated = readme.read_text(encoding="utf-8")

    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "version-v0.5.1-alpha.1+cu13-blue.svg" in updated
