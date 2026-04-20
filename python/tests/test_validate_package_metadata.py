import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import validate_package_metadata as validator


def _readme(version: str) -> str:
    return f"""
# xlog

![version](https://img.shields.io/badge/version-v{version}-blue.svg)

Release status: `v{version}`

## Quickstart

python scripts/xlog_doctor.py
cargo build --release
cargo build --release -p xlog-cli --features host-io
./target/release/xlog
"""


def _metadata(bin_names: tuple[str, ...] = ("xlog",)) -> dict:
    return {
        "packages": [
            {
                "name": "xlog-cli",
                "source": None,
                "targets": [{"name": name, "kind": ["bin"]} for name in bin_names],
            }
        ]
    }


def test_release_plz_branch_requires_readme_version_to_match_workspace() -> None:
    errors = validator.validate_package_metadata(
        readme=_readme("0.5.0"),
        workspace_version="0.5.1",
        metadata=_metadata(),
        branch_name="release-plz-2026-04-20T00-23-02Z",
    )

    assert (
        "README version badge (0.5.0) does not match workspace version (0.5.1)."
        in errors
    )
    assert (
        "README release status (0.5.0) does not match workspace version (0.5.1)."
        in errors
    )


def test_main_branch_requires_readme_version_to_match_workspace() -> None:
    errors = validator.validate_package_metadata(
        readme=_readme("0.5.0"),
        workspace_version="0.5.1",
        metadata=_metadata(),
        branch_name="main",
    )

    assert (
        "README version badge (0.5.0) does not match workspace version (0.5.1)."
        in errors
    )
    assert (
        "README release status (0.5.0) does not match workspace version (0.5.1)."
        in errors
    )


def test_release_plz_branch_still_requires_quickstart_contract() -> None:
    errors = validator.validate_package_metadata(
        readme=_readme("0.5.0").replace("python scripts/xlog_doctor.py", "", 1),
        workspace_version="0.5.1",
        metadata=_metadata(),
        branch_name="release-plz-2026-04-20T00-23-02Z",
    )

    assert "README quickstart is missing required snippets:" in errors
    assert "  - python scripts/xlog_doctor.py" in errors
