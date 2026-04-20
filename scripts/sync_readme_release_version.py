"""Sync the root README release badge and status line to the workspace version."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.release_version_support import README_BADGE_VERSION_RE, workspace_version


RELEASE_STATUS_VERSION_RE = re.compile(
    r"((?:\*\*)?Release status:(?:\*\*)?\s*`v)([^`]+)(`)"
)


def sync_readme_version(readme: str, version: str) -> str:
    badge_count = len(README_BADGE_VERSION_RE.findall(readme))
    if badge_count != 1:
        raise ValueError("README version badge not found or ambiguous.")
    updated = README_BADGE_VERSION_RE.sub(
        f"version-v{version}-blue.svg",
        readme,
        count=1,
    )

    status_count = len(RELEASE_STATUS_VERSION_RE.findall(updated))
    if status_count != 1:
        raise ValueError("README release status line not found or ambiguous.")
    updated = RELEASE_STATUS_VERSION_RE.sub(
        rf"\g<1>{version}\g<3>",
        updated,
        count=1,
    )

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
