#!/usr/bin/env python3
"""Build pyxlog twice and verify location-independent wheel contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import zipfile
from pathlib import Path


def build_wheel(repo_root: Path, target_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    command = [
        "maturin",
        "build",
        "-m",
        "crates/pyxlog/Cargo.toml",
        "--release",
        "--locked",
        "--compatibility",
        "linux",
        "--out",
        str(output_dir),
    ]
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=repo_root, env=env, check=True)

    wheels = sorted(output_dir.glob("pyxlog-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"expected exactly one pyxlog wheel in {output_dir}, found {len(wheels)}"
        )
    return wheels[0]


def wheel_content_manifest(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError(f"wheel contains duplicate member names: {wheel}")
        return {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(names)
        }


def canonical_manifest_hash(manifest: dict[str, str]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_reproducible_wheels(first_wheel: Path, second_wheel: Path) -> None:
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

    manifest_hash = canonical_manifest_hash(first_manifest)
    native_member = native_members[0]
    print(f"canonical_wheel_content_sha256={manifest_hash}")
    print(f"native_library_sha256={first_manifest[native_member]}")
    print("location_independent_pyxlog_wheel=PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory that will retain the first verified wheel",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    retained_wheels = sorted(args.out_dir.glob("pyxlog-*.whl"))
    if retained_wheels:
        raise RuntimeError(
            f"output directory already contains a pyxlog wheel: {args.out_dir}"
        )

    with tempfile.TemporaryDirectory(prefix="xlog-pyxlog-reproducibility-") as temp:
        temp_root = Path(temp)
        first_target = temp_root / "first-build" / "target"
        second_target = temp_root / "different-location" / "second-build" / "target"
        second_output = temp_root / "second-wheel"

        first_wheel = build_wheel(repo_root, first_target, args.out_dir)
        second_wheel = build_wheel(repo_root, second_target, second_output)
        validate_reproducible_wheels(first_wheel, second_wheel)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
