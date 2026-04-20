import json
import subprocess
import sys
import tempfile
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
    )

    assert (
        "README version badge (0.5.0) does not match workspace version (0.5.1)."
        in errors
    )
    assert (
        "README release status (0.5.0) does not match workspace version (0.5.1)."
        in errors
    )


def test_prerelease_and_build_versions_match_readme_badge() -> None:
    errors = validator.validate_package_metadata(
        readme=_readme("0.5.1-alpha.1+cu13"),
        workspace_version="0.5.1-alpha.1+cu13",
        metadata=_metadata(),
    )

    assert errors == []


def test_release_plz_branch_still_requires_quickstart_contract() -> None:
    errors = validator.validate_package_metadata(
        readme=_readme("0.5.0").replace("python scripts/xlog_doctor.py", "", 1),
        workspace_version="0.5.1",
        metadata=_metadata(),
    )

    assert "README quickstart is missing required snippets:" in errors
    assert "  - python scripts/xlog_doctor.py" in errors


def test_validate_package_metadata_script_runs_as_direct_entrypoint() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        readme = tmpdir / "README.md"
        cargo = tmpdir / "Cargo.toml"
        metadata = tmpdir / "cargo-metadata.json"

        version = "0.5.1-alpha.1+cu13"
        readme.write_text(_readme(version), encoding="utf-8")
        cargo.write_text(f'[workspace.package]\nversion = "{version}"\n', encoding="utf-8")
        metadata.write_text(json.dumps(_metadata()), encoding="utf-8")

        proc = subprocess.run(
            [
                sys.executable,
                "scripts/validate_package_metadata.py",
                "--readme",
                str(readme),
                "--cargo",
                str(cargo),
                "--metadata",
                str(metadata),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

    assert proc.returncode == 0, proc.stderr or proc.stdout
