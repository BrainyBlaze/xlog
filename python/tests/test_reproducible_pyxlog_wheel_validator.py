from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import validate_reproducible_pyxlog_wheel as validator


SOURCE_DATE_EPOCH = "1730000000"
DIST_INFO = "pyxlog-0.12.0.dist-info"
SBOM_PATH = f"{DIST_INFO}/sboms/pyxlog.cyclonedx.json"
RECORD_PATH = f"{DIST_INFO}/RECORD"


def deterministic_sbom() -> dict[str, object]:
    timestamp = (
        datetime.fromtimestamp(int(SOURCE_DATE_EPOCH), timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        + ".000000000Z"
    )
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": [
                {
                    "vendor": "CycloneDX",
                    "name": "cargo-cyclonedx",
                    "version": "0.5.9",
                }
            ],
        },
        "components": [],
    }


def record_bytes(members: dict[str, bytes]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    for name, data in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        writer.writerow(
            [name, f"sha256={digest.rstrip(b'=').decode('ascii')}", str(len(data))]
        )
    writer.writerow([RECORD_PATH, "", ""])
    return output.getvalue().encode("utf-8")


def write_wheel(
    path: Path,
    *,
    sbom: dict[str, object] | None = None,
    native_members: tuple[str, ...] = (
        "pyxlog/_native.cpython-311-x86_64-linux-gnu.so",
    ),
    package_bytes: bytes = b"package",
    extra_members: dict[str, bytes] | None = None,
    archive_timestamp: tuple[int, int, int, int, int, int] = (2024, 1, 1, 0, 0, 0),
    corrupt_record: bool = False,
) -> None:
    members = {
        "pyxlog/__init__.py": package_bytes,
        SBOM_PATH: json.dumps(
            sbom if sbom is not None else deterministic_sbom(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    }
    members.update({name: b"native-library" for name in native_members})
    members.update(extra_members or {})
    record = record_bytes(members)
    if corrupt_record:
        members["pyxlog/__init__.py"] = b"changed-after-record-generation"
    members[RECORD_PATH] = record

    with zipfile.ZipFile(path, "w") as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=archive_timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)


def test_source_date_epoch_preserves_a_valid_caller_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1720000000")

    def unexpected_git_call(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError(
            f"git must not run when SOURCE_DATE_EPOCH is set: {args} {kwargs}"
        )

    monkeypatch.setattr(validator.subprocess, "run", unexpected_git_call)

    assert validator.resolve_source_date_epoch(Path("/unused")) == "1720000000"


def test_source_date_epoch_defaults_to_the_checked_out_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    repo_root = Path("/repository")
    calls: list[tuple[list[str], Path]] = []

    def completed_git_call(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check
        assert capture_output
        assert text
        calls.append((command, cwd))
        return subprocess.CompletedProcess(command, 0, stdout="1730000000\n")

    monkeypatch.setattr(validator.subprocess, "run", completed_git_call)

    assert validator.resolve_source_date_epoch(repo_root) == "1730000000"
    assert calls == [
        (
            [
                "git",
                "-c",
                f"safe.directory={repo_root}",
                "show",
                "-s",
                "--format=%ct",
                "HEAD",
            ],
            repo_root,
        )
    ]


@pytest.mark.parametrize("value", ["", "-1", "not-a-timestamp"])
def test_source_date_epoch_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", value)

    with pytest.raises(RuntimeError, match="SOURCE_DATE_EPOCH"):
        validator.resolve_source_date_epoch(Path("/unused"))


def test_required_maturin_version_is_read_from_one_exact_constraint(
    tmp_path: Path,
) -> None:
    constraint = tmp_path / "python" / "constraints-build.txt"
    constraint.parent.mkdir()
    constraint.write_text("# wheel builder\nmaturin==1.14.1\n", encoding="utf-8")

    assert validator.required_maturin_version(tmp_path) == "1.14.1"


def test_outdated_maturin_is_rejected_before_a_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    constraint = tmp_path / "python" / "constraints-build.txt"
    constraint.parent.mkdir()
    constraint.write_text("maturin==1.14.1\n", encoding="utf-8")

    def completed_version_call(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert command == ["maturin", "--version"]
        return subprocess.CompletedProcess(command, 0, stdout="maturin 1.12.1\n")

    monkeypatch.setattr(validator.subprocess, "run", completed_version_call)

    with pytest.raises(RuntimeError, match=r"maturin 1\.14\.1.*1\.12\.1"):
        validator.require_supported_maturin(tmp_path)


def test_build_wheel_resolves_caller_relative_output_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    caller = tmp_path / "caller"
    repo_root = tmp_path / "repository"
    caller.mkdir()
    repo_root.mkdir()
    monkeypatch.chdir(caller)

    def completed_build(
        command: list[str], *, cwd: Path, env: dict[str, str], check: bool
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == repo_root
        assert check
        output = Path(command[command.index("--out") + 1])
        assert output == caller / "relative-wheelhouse"
        assert output.is_absolute()
        assert Path(env["CARGO_TARGET_DIR"]).is_absolute()
        (output / "pyxlog-test.whl").write_bytes(b"wheel")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(validator.subprocess, "run", completed_build)

    wheel = validator.build_wheel(
        repo_root,
        Path("relative-target"),
        Path("relative-wheelhouse"),
        SOURCE_DATE_EPOCH,
    )

    assert wheel == caller / "relative-wheelhouse" / "pyxlog-test.whl"


def test_wheel_integrity_accepts_deterministic_sbom_and_record(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "valid.whl"
    write_wheel(wheel)

    validator.validate_wheel_integrity(wheel, SOURCE_DATE_EPOCH)


def test_wheel_integrity_rejects_random_cyclonedx_identity(tmp_path: Path) -> None:
    wheel = tmp_path / "random-sbom.whl"
    sbom = deterministic_sbom()
    sbom["serialNumber"] = "urn:uuid:38713dff-e5b1-4d00-a05c-2e5460aed47b"
    write_wheel(wheel, sbom=sbom)

    with pytest.raises(RuntimeError, match="serialNumber"):
        validator.validate_wheel_integrity(wheel, SOURCE_DATE_EPOCH)


def test_wheel_integrity_rejects_wall_clock_cyclonedx_timestamp(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "wall-clock-sbom.whl"
    sbom = deterministic_sbom()
    metadata = sbom["metadata"]
    assert isinstance(metadata, dict)
    metadata["timestamp"] = "2026-08-28T07:04:19.984949665Z"
    write_wheel(wheel, sbom=sbom)

    with pytest.raises(RuntimeError, match="SOURCE_DATE_EPOCH"):
        validator.validate_wheel_integrity(wheel, SOURCE_DATE_EPOCH)


def test_wheel_integrity_rejects_old_cargo_cyclonedx(tmp_path: Path) -> None:
    wheel = tmp_path / "old-cargo-cyclonedx.whl"
    sbom = deterministic_sbom()
    metadata = sbom["metadata"]
    assert isinstance(metadata, dict)
    tools = metadata["tools"]
    assert isinstance(tools, list)
    tool = tools[0]
    assert isinstance(tool, dict)
    tool["version"] = "0.5.7"
    write_wheel(wheel, sbom=sbom)

    with pytest.raises(RuntimeError, match=r"cargo-cyclonedx 0\.5\.9"):
        validator.validate_wheel_integrity(wheel, SOURCE_DATE_EPOCH)


def test_wheel_integrity_rejects_stale_record_hash(tmp_path: Path) -> None:
    wheel = tmp_path / "stale-record.whl"
    write_wheel(wheel, corrupt_record=True)

    with pytest.raises(RuntimeError, match="RECORD hash"):
        validator.validate_wheel_integrity(wheel, SOURCE_DATE_EPOCH)


@pytest.mark.parametrize(
    "bytecode_path",
    [
        "pyxlog/__pycache__/module.cpython-311.pyc",
        "pyxlog/module.pyc",
        "pyxlog/module.pyo",
    ],
)
def test_wheel_integrity_rejects_python_bytecode(
    tmp_path: Path, bytecode_path: str
) -> None:
    wheel = tmp_path / "python-bytecode.whl"
    write_wheel(wheel, extra_members={bytecode_path: b"generated-bytecode"})

    with pytest.raises(RuntimeError, match="Python bytecode"):
        validator.validate_wheel_integrity(wheel, SOURCE_DATE_EPOCH)


def test_reproducible_wheels_accept_identical_real_archives(tmp_path: Path) -> None:
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    write_wheel(first)
    write_wheel(second)

    validator.validate_reproducible_wheels(first, second, SOURCE_DATE_EPOCH)


def test_reproducible_wheels_reject_differing_member_content(tmp_path: Path) -> None:
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    write_wheel(first, package_bytes=b"first")
    write_wheel(second, package_bytes=b"second")

    with pytest.raises(RuntimeError, match="pyxlog/__init__.py"):
        validator.validate_reproducible_wheels(first, second, SOURCE_DATE_EPOCH)


def test_reproducible_wheels_reject_differing_archive_metadata(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    write_wheel(first, archive_timestamp=(2024, 1, 1, 0, 0, 0))
    write_wheel(second, archive_timestamp=(2024, 1, 2, 0, 0, 0))

    with pytest.raises(RuntimeError, match="archive bytes"):
        validator.validate_reproducible_wheels(first, second, SOURCE_DATE_EPOCH)


def test_wheel_manifest_rejects_duplicate_member_names(tmp_path: Path) -> None:
    wheel = tmp_path / "duplicate.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pyxlog/__init__.py", b"first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("pyxlog/__init__.py", b"second")

    with pytest.raises(RuntimeError, match="duplicate member names"):
        validator.wheel_content_manifest(wheel)


@pytest.mark.parametrize("native_count", [0, 2])
def test_reproducible_wheels_require_exactly_one_native_library(
    tmp_path: Path, native_count: int
) -> None:
    native_members = tuple(
        f"pyxlog/_native_{index}.so" for index in range(native_count)
    )
    first = tmp_path / "first.whl"
    second = tmp_path / "second.whl"
    write_wheel(first, native_members=native_members)
    write_wheel(second, native_members=native_members)

    with pytest.raises(RuntimeError, match="exactly one native pyxlog library"):
        validator.validate_reproducible_wheels(first, second, SOURCE_DATE_EPOCH)
