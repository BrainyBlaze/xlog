"""Validate actual package ownership for manual release preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def validate_package_metadata(
    *,
    metadata: dict,
) -> list[str]:
    errors: list[str] = []

    xlog_cli = next(
        (
            package
            for package in metadata["packages"]
            if package["name"] == "xlog-cli" and package.get("source") is None
        ),
        None,
    )
    if xlog_cli is None:
        errors.append("cargo metadata did not include the local xlog-cli package.")
        return errors

    bin_targets = {
        target["name"] for target in xlog_cli["targets"] if "bin" in target["kind"]
    }
    if "xlog" not in bin_targets:
        errors.append(
            f"xlog-cli binary targets do not include `xlog`: {sorted(bin_targets)}"
        )

    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cargo", default="Cargo.toml")
    parser.add_argument("--metadata", default="cargo-metadata.json")
    return parser.parse_args(argv)


def load_metadata(metadata_path: Path, cargo_path: Path) -> dict:
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    proc = subprocess.run(
        [
            "cargo",
            "metadata",
            "--locked",
            "--no-deps",
            "--format-version=1",
            "--manifest-path",
            str(cargo_path.resolve()),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "cargo metadata failed"
        raise RuntimeError(
            f"Could not load package metadata from {metadata_path}: {detail}"
        )

    return json.loads(proc.stdout)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    cargo_path = Path(args.cargo)
    metadata_path = Path(args.metadata)

    metadata = load_metadata(metadata_path, cargo_path)

    errors = validate_package_metadata(
        metadata=metadata,
    )
    if errors:
        print("\n".join(errors))
        return 1

    print("Workspace package metadata validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
