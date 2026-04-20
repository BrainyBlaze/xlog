"""Sync the root README release badge and status line to the workspace version."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def workspace_version(cargo_text: str) -> str:
    match = re.search(
        r"^\[workspace\.package\][^\[]*?^version\s*=\s*\"([^\"]+)\"",
        cargo_text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError("Could not determine workspace.package.version from Cargo.toml.")
    return match.group(1)


def sync_readme_version(readme: str, version: str) -> str:
    updated, badge_count = re.subn(
        r"version-v([0-9][^\-]*)-blue\.svg",
        f"version-v{version}-blue.svg",
        readme,
        count=1,
    )
    if badge_count != 1:
        raise ValueError("README version badge not found or ambiguous.")

    updated, status_count = re.subn(
        r"((?:\*\*)?Release status:(?:\*\*)?\s*`v)([^`]+)(`)",
        rf"\g<1>{version}\g<3>",
        updated,
        count=1,
    )
    if status_count != 1:
        raise ValueError("README release status line not found or ambiguous.")

    return updated


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--cargo", default="Cargo.toml")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    readme_path = Path(args.readme)
    cargo_path = Path(args.cargo)

    cargo_text = cargo_path.read_text(encoding="utf-8")
    version = workspace_version(cargo_text)
    readme = readme_path.read_text(encoding="utf-8")
    updated = sync_readme_version(readme, version)

    if updated != readme:
        readme_path.write_text(updated, encoding="utf-8")
        print(f"Updated {readme_path} to release version {version}.")
    else:
        print(f"{readme_path} already matches release version {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
