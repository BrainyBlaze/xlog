import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import validate_package_metadata as validator


def _readme() -> str:
    return """
# xlog

## Quickstart

python scripts/xlog_doctor.py
cargo build --release
cargo build --release -p xlog-cli --features host-io
python scripts/install_pyxlog_for_python.py --python
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


def test_quickstart_snippets_pass_when_all_present() -> None:
    errors = validator.validate_package_metadata(
        readme=_readme(),
        metadata=_metadata(),
    )

    assert errors == []


def test_missing_quickstart_snippet_is_reported() -> None:
    errors = validator.validate_package_metadata(
        readme=_readme().replace("python scripts/xlog_doctor.py", "", 1),
        metadata=_metadata(),
    )

    assert "README quickstart is missing required snippets:" in errors
    assert "  - python scripts/xlog_doctor.py" in errors


def test_xlog_cli_binary_target_required() -> None:
    errors = validator.validate_package_metadata(
        readme=_readme(),
        metadata=_metadata(bin_names=("not-xlog",)),
    )

    assert any("xlog-cli binary targets do not include `xlog`" in e for e in errors)


def test_missing_xlog_cli_package_is_reported() -> None:
    errors = validator.validate_package_metadata(
        readme=_readme(),
        metadata={"packages": []},
    )

    assert "cargo metadata did not include the local xlog-cli package." in errors


def test_validate_package_metadata_script_runs_as_direct_entrypoint() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        readme = tmpdir / "README.md"
        cargo = tmpdir / "Cargo.toml"
        metadata = tmpdir / "cargo-metadata.json"

        readme.write_text(_readme(), encoding="utf-8")
        cargo.write_text('[workspace.package]\nversion = "0.9.2"\n', encoding="utf-8")
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


def test_load_metadata_generates_fresh_cargo_metadata_when_file_is_missing(
    monkeypatch,
) -> None:
    cargo_path = Path("/tmp/xlog/Cargo.toml")
    metadata_path = Path("/tmp/xlog/cargo-metadata.json")
    expected = _metadata()

    def fake_run(cmd, check, capture_output, text):
        assert cmd[:5] == [
            "cargo",
            "metadata",
            "--locked",
            "--no-deps",
            "--format-version=1",
        ]
        assert cmd[5] == "--manifest-path"
        assert cmd[6] == str(cargo_path.resolve())
        assert check is False
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(expected),
            stderr="",
        )

    monkeypatch.setattr(validator.subprocess, "run", fake_run)

    assert validator.load_metadata(metadata_path, cargo_path) == expected
