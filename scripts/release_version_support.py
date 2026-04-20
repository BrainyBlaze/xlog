"""Shared release-version parsing helpers for workspace and README metadata."""

from __future__ import annotations

import re

SEMVER_VERSION_PATTERN = (
    r"(?:0|[1-9]\d*)"
    r"(?:\.(?:0|[1-9]\d*)){2}"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)

README_BADGE_VERSION_RE = re.compile(
    rf"version-v({SEMVER_VERSION_PATTERN})-blue\.svg"
)

WORKSPACE_VERSION_RE = re.compile(
    r"^\[workspace\.package\][^\[]*?^version\s*=\s*\"([^\"]+)\"",
    re.MULTILINE | re.DOTALL,
)


def workspace_version(cargo_text: str) -> str:
    match = WORKSPACE_VERSION_RE.search(cargo_text)
    if match is None:
        raise ValueError("Could not determine workspace.package.version from Cargo.toml.")
    return match.group(1)
