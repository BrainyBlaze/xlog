#!/usr/bin/env python3
"""Stage generated CUDA kernel artifacts into a kernels/ directory.

This copies the build output artifacts produced by crates/xlog-cuda/build.rs
into a destination directory suitable for packaging or local release layouts.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

KERNEL_ARTIFACT_SUFFIXES = (".cubin", ".portable.ptx")


def _discover_kernel_artifacts(out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        raise SystemExit(f"from-out-dir does not exist: {out_dir}")
    if not out_dir.is_dir():
        raise SystemExit(f"from-out-dir is not a directory: {out_dir}")

    artifacts = [
        path
        for path in out_dir.iterdir()
        if path.is_file()
        and path.name.endswith(KERNEL_ARTIFACT_SUFFIXES)
    ]
    return sorted(artifacts, key=lambda p: p.name)


def _prune_stale_kernel_artifacts(dest_dir: Path, expected_names: set[str]) -> None:
    for path in dest_dir.iterdir():
        if path.is_file() and path.name.endswith(KERNEL_ARTIFACT_SUFFIXES):
            if path.name not in expected_names:
                path.unlink()


def _copy_artifact(src: Path, dest_dir: Path) -> tuple[Path, str, int]:
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    return dest, digest, dest.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy generated CUDA kernel artifacts into a kernels/ directory."
    )
    parser.add_argument("--from-out-dir", required=True, type=Path)
    parser.add_argument("--to", required=True, type=Path)
    args = parser.parse_args(argv)

    artifacts = _discover_kernel_artifacts(args.from_out_dir)
    if not artifacts:
        raise SystemExit(f"no kernel artifacts found in {args.from_out_dir}")

    args.to.mkdir(parents=True, exist_ok=True)
    _prune_stale_kernel_artifacts(args.to, {artifact.name for artifact in artifacts})

    manifest: list[tuple[str, str, int]] = []
    for artifact in artifacts:
        _, digest, size = _copy_artifact(artifact, args.to)
        manifest.append((artifact.name, digest, size))

    staged_names = {
        path.name
        for path in args.to.iterdir()
        if path.is_file() and path.name.endswith(KERNEL_ARTIFACT_SUFFIXES)
    }
    expected_names = {artifact.name for artifact in artifacts}
    if staged_names != expected_names:
        missing = sorted(expected_names - staged_names)
        unexpected = sorted(staged_names - expected_names)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise SystemExit("staged kernel tree validation failed: " + ", ".join(details))

    for name, digest, size in manifest:
        print(f"{name}\t{digest}\t{size}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
