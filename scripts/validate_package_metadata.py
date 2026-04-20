"""Validate README quickstart and package metadata invariants used by CI."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REQUIRED_SNIPPETS = (
    "python scripts/xlog_doctor.py",
    "cargo build --release",
    "cargo build --release -p xlog-cli --features host-io",
    "./target/release/xlog",
)


def _is_release_plz_branch(branch_name: str | None) -> bool:
    return bool(branch_name) and branch_name.startswith("release-plz-")


def validate_package_metadata(
    *,
    readme: str,
    workspace_version: str,
    metadata: dict,
    branch_name: str | None = None,
) -> list[str]:
    errors: list[str] = []
    readme_plain = readme.replace("**", "")

    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in readme]
    if missing:
        errors.append("README quickstart is missing required snippets:")
        errors.extend(f"  - {snippet}" for snippet in missing)

    if not _is_release_plz_branch(branch_name):
        badge_match = re.search(r"version-v([0-9][^\-]*)-blue\.svg", readme)
        if not badge_match:
            errors.append("Could not find README version badge.")
        elif badge_match.group(1) != workspace_version:
            errors.append(
                f"README version badge ({badge_match.group(1)}) does not match workspace version ({workspace_version})."
            )

        status_match = re.search(r"Release status:\s*`v([^`]+)`", readme_plain)
        if not status_match:
            errors.append("Could not find README release status line.")
        elif status_match.group(1) != workspace_version:
            errors.append(
                f"README release status ({status_match.group(1)}) does not match workspace version ({workspace_version})."
            )

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
    parser.add_argument("--branch", default=None)
    return parser.parse_args(argv)


def _workspace_version(cargo_text: str) -> str:
    match = re.search(
        r"^\[workspace\.package\][^\[]*?^version\s*=\s*\"([^\"]+)\"",
        cargo_text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError("Could not determine workspace.package.version from Cargo.toml.")
    return match.group(1)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    branch_name = (
        args.branch or os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME")
    )

    readme = Path(args.readme).read_text(encoding="utf-8")
    cargo_text = Path(args.cargo).read_text(encoding="utf-8")
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    workspace_version = _workspace_version(cargo_text)

    errors = validate_package_metadata(
        readme=readme,
        workspace_version=workspace_version,
        metadata=metadata,
        branch_name=branch_name,
    )
    if errors:
        print("\n".join(errors))
        return 1

    print("README quickstart assumptions and workspace package metadata validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
