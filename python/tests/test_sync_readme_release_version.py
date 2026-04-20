import sys
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


def test_sync_readme_version_requires_known_patterns() -> None:
    try:
        syncer.sync_readme_version("# xlog\n", "0.5.1")
    except ValueError as exc:
        assert "README version badge" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing README patterns")
