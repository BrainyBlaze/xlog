#!/usr/bin/env python3
"""Build pyxlog twice and verify byte-for-byte reproducible wheels."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import shlex
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


MATURIN_CONSTRAINT = Path("python/constraints-build.txt")
EXPECTED_CARGO_CYCLONEDX_VERSION = "0.5.9"


def required_maturin_version(repo_root: Path) -> str:
    constraint_path = repo_root / MATURIN_CONSTRAINT
    requirements = [
        line.strip()
        for line in constraint_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(requirements) != 1 or not requirements[0].startswith("maturin=="):
        raise RuntimeError(
            f"{constraint_path} must contain exactly one exact maturin requirement"
        )
    version = requirements[0].removeprefix("maturin==")
    if not version or any(character.isspace() for character in version):
        raise RuntimeError(f"invalid exact maturin requirement: {requirements[0]!r}")
    return version


def require_supported_maturin(repo_root: Path) -> str:
    required = required_maturin_version(repo_root)
    completed = subprocess.run(
        ["maturin", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = completed.stdout.strip().split()
    actual = fields[1] if len(fields) == 2 and fields[0] == "maturin" else "unknown"
    if actual != required:
        raise RuntimeError(
            f"reproducible pyxlog wheels require maturin {required}, found {actual}"
        )
    return actual


def resolve_source_date_epoch(repo_root: Path) -> str:
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch is None:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repo_root}",
                "show",
                "-s",
                "--format=%ct",
                "HEAD",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        source_date_epoch = completed.stdout.strip()

    if not source_date_epoch.isascii() or not source_date_epoch.isdigit():
        raise RuntimeError(
            "SOURCE_DATE_EPOCH must be a non-negative integer timestamp, "
            f"got {source_date_epoch!r}"
        )
    return source_date_epoch


def build_wheel(
    repo_root: Path,
    target_dir: Path,
    output_dir: Path,
    source_date_epoch: str,
    *,
    compatibility: str = "linux",
    python_executable: str | None = None,
) -> Path:
    target_dir = target_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    env["SOURCE_DATE_EPOCH"] = source_date_epoch
    command = [
        "maturin",
        "build",
        "-m",
        "crates/pyxlog/Cargo.toml",
        "--release",
        "--locked",
        "--compatibility",
        compatibility,
        "--out",
        str(output_dir),
    ]
    if python_executable is not None:
        command.extend(["-i", python_executable])
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=repo_root, env=env, check=True)

    wheels = sorted(output_dir.glob("pyxlog-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"expected exactly one pyxlog wheel in {output_dir}, found {len(wheels)}"
        )
    return wheels[0]


def wheel_content_manifest(wheel: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256(data).hexdigest()
        for name, data in sorted(_wheel_members(wheel).items())
    }


def _wheel_members(wheel: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError(f"wheel contains duplicate member names: {wheel}")
        return {name: archive.read(name) for name in names if not name.endswith("/")}


def _validate_cyclonedx_sbom(
    wheel: Path, members: dict[str, bytes], source_date_epoch: str
) -> None:
    sbom_paths = sorted(
        name
        for name in members
        if name.endswith(".dist-info/sboms/pyxlog.cyclonedx.json")
    )
    if len(sbom_paths) != 1:
        raise RuntimeError(
            f"expected exactly one pyxlog CycloneDX SBOM in {wheel}, "
            f"found {len(sbom_paths)}"
        )
    sbom_path = sbom_paths[0]
    try:
        sbom = json.loads(members[sbom_path])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid CycloneDX JSON in {wheel}: {error}") from error
    if not isinstance(sbom, dict):
        raise RuntimeError(f"CycloneDX SBOM root must be an object: {wheel}")
    if "serialNumber" in sbom:
        raise RuntimeError(
            f"CycloneDX serialNumber is nondeterministic and must be omitted: {wheel}"
        )

    metadata = sbom.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"CycloneDX metadata must be an object: {wheel}")
    timestamp = metadata.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
        raise RuntimeError(f"CycloneDX timestamp must be UTC: {wheel}")
    expected_timestamp = datetime.fromtimestamp(int(source_date_epoch), timezone.utc)
    expected_seconds = expected_timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    fractional = timestamp.removeprefix(expected_seconds).removesuffix("Z")
    if not timestamp.startswith(expected_seconds) or not (
        fractional == ""
        or (fractional.startswith(".") and set(fractional[1:]) == {"0"})
    ):
        raise RuntimeError(
            "CycloneDX timestamp does not match SOURCE_DATE_EPOCH: "
            f"{timestamp} != {expected_timestamp.isoformat()}"
        )

    tools = metadata.get("tools")
    if not isinstance(tools, list) or not any(
        isinstance(tool, dict)
        and tool.get("name") == "cargo-cyclonedx"
        and tool.get("version") == EXPECTED_CARGO_CYCLONEDX_VERSION
        for tool in tools
    ):
        raise RuntimeError(
            "CycloneDX SBOM was not generated by the supported cargo-cyclonedx "
            f"{EXPECTED_CARGO_CYCLONEDX_VERSION}: {wheel}"
        )


def _record_digest(data: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
    return "sha256=" + encoded.rstrip(b"=").decode("ascii")


def _validate_record(wheel: Path, members: dict[str, bytes]) -> None:
    record_paths = sorted(
        name for name in members if name.endswith(".dist-info/RECORD")
    )
    if len(record_paths) != 1:
        raise RuntimeError(
            f"expected exactly one wheel RECORD in {wheel}, found {len(record_paths)}"
        )
    record_path = record_paths[0]
    try:
        rows = list(
            csv.reader(io.StringIO(members[record_path].decode("utf-8"), newline=""))
        )
    except UnicodeDecodeError as error:
        raise RuntimeError(f"wheel RECORD is not UTF-8: {wheel}") from error

    records: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            raise RuntimeError(f"wheel RECORD row must have three fields: {row!r}")
        name, digest, size = row
        if name in records:
            raise RuntimeError(f"wheel RECORD contains duplicate path: {name}")
        records[name] = (digest, size)

    member_names = set(members)
    record_names = set(records)
    if member_names != record_names:
        missing = sorted(member_names - record_names)
        unexpected = sorted(record_names - member_names)
        raise RuntimeError(
            f"wheel RECORD paths do not match archive members; "
            f"missing={missing} unexpected={unexpected}"
        )

    for name, data in members.items():
        digest, size = records[name]
        if name == record_path:
            if digest or size:
                raise RuntimeError("wheel RECORD must not hash or size itself")
            continue
        expected_digest = _record_digest(data)
        if digest != expected_digest:
            raise RuntimeError(
                f"wheel RECORD hash mismatch for {name}: {digest} != {expected_digest}"
            )
        if size != str(len(data)):
            raise RuntimeError(
                f"wheel RECORD size mismatch for {name}: {size} != {len(data)}"
            )


def validate_wheel_integrity(wheel: Path, source_date_epoch: str) -> None:
    members = _wheel_members(wheel)
    _validate_cyclonedx_sbom(wheel, members, source_date_epoch)
    _validate_record(wheel, members)


def canonical_manifest_hash(manifest: dict[str, str]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_reproducible_wheels(
    first_wheel: Path, second_wheel: Path, source_date_epoch: str
) -> None:
    validate_wheel_integrity(first_wheel, source_date_epoch)
    validate_wheel_integrity(second_wheel, source_date_epoch)
    first_manifest = wheel_content_manifest(first_wheel)
    second_manifest = wheel_content_manifest(second_wheel)
    if first_manifest != second_manifest:
        differing_members = sorted(
            name
            for name in first_manifest.keys() | second_manifest.keys()
            if first_manifest.get(name) != second_manifest.get(name)
        )
        raise RuntimeError(
            "pyxlog wheel contents depend on the build location; differing members: "
            + ", ".join(differing_members)
        )

    native_members = [
        name
        for name in first_manifest
        if name.startswith("pyxlog/") and name.endswith((".so", ".dylib", ".pyd"))
    ]
    if len(native_members) != 1:
        raise RuntimeError(
            f"expected exactly one native pyxlog library, found {len(native_members)}"
        )

    first_archive_hash = hashlib.sha256(first_wheel.read_bytes()).hexdigest()
    second_archive_hash = hashlib.sha256(second_wheel.read_bytes()).hexdigest()
    if first_archive_hash != second_archive_hash:
        raise RuntimeError(
            "pyxlog wheel archive bytes differ despite identical member contents: "
            f"{first_archive_hash} != {second_archive_hash}"
        )

    manifest_hash = canonical_manifest_hash(first_manifest)
    native_member = native_members[0]
    print(f"canonical_wheel_content_sha256={manifest_hash}")
    print(f"wheel_archive_sha256={first_archive_hash}")
    print(f"native_library_sha256={first_manifest[native_member]}")
    print("reproducible_pyxlog_wheel=PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory that will retain the first verified wheel",
    )
    parser.add_argument(
        "--compatibility",
        default="linux",
        help="maturin compatibility tag (default: linux)",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        help="explicit Python interpreter passed to maturin",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.out_dir.expanduser().resolve()
    maturin_version = require_supported_maturin(repo_root)
    print(f"maturin_version={maturin_version}")
    source_date_epoch = resolve_source_date_epoch(repo_root)
    print(f"SOURCE_DATE_EPOCH={source_date_epoch}")
    retained_wheels = sorted(output_dir.glob("pyxlog-*.whl"))
    if retained_wheels:
        raise RuntimeError(
            f"output directory already contains a pyxlog wheel: {output_dir}"
        )

    with tempfile.TemporaryDirectory(prefix="xlog-pyxlog-reproducibility-") as temp:
        temp_root = Path(temp)
        first_target = temp_root / "first-build" / "target"
        second_target = temp_root / "different-location" / "second-build" / "target"
        second_output = temp_root / "second-wheel"

        first_wheel = build_wheel(
            repo_root,
            first_target,
            output_dir,
            source_date_epoch,
            compatibility=args.compatibility,
            python_executable=args.python_executable,
        )
        second_wheel = build_wheel(
            repo_root,
            second_target,
            second_output,
            source_date_epoch,
            compatibility=args.compatibility,
            python_executable=args.python_executable,
        )
        validate_reproducible_wheels(first_wheel, second_wheel, source_date_epoch)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
