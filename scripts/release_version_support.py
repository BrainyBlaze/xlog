"""Shared release-version parsing helpers for README metadata."""

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
