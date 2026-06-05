"""Validate README quickstart and package metadata invariants used by CI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REQUIRED_SNIPPETS = (
    "python scripts/xlog_doctor.py",
    "cargo build --release",
    "cargo build --release -p xlog-cli --features host-io",
    "python scripts/install_pyxlog_for_python.py --python",
    "./target/release/xlog",
)


def validate_package_metadata(
    *,
    readme: str,
    metadata: dict,
) -> list[str]:
    errors: list[str] = []

    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in readme]
    if missing:
        errors.append("README quickstart is missing required snippets:")
        errors.extend(f"  - {snippet}" for snippet in missing)

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
    parser.add_argument("--readme", default="README.md")
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

    readme_path = Path(args.readme)
    cargo_path = Path(args.cargo)
    metadata_path = Path(args.metadata)

    readme = readme_path.read_text(encoding="utf-8")
    metadata = load_metadata(metadata_path, cargo_path)

    errors = validate_package_metadata(
        readme=readme,
        metadata=metadata,
    )
    if errors:
        print("\n".join(errors))
        return 1

    print("README quickstart assumptions and workspace package metadata validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
