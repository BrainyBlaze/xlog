#!/usr/bin/env python3
"""Decide whether a Git revision range contains a release-worthy commit."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


RELEASE_SUBJECT = re.compile(
    r"^(?:feat|fix|perf|refactor|docs|build|ci|test|revert)"
    r"(?:\([^)]+\))?!?:"
)
BREAKING_FOOTER = re.compile(r"^BREAKING CHANGE:", re.MULTILINE)


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _default_range(repository: Path) -> tuple[str, str]:
    described = subprocess.run(
        ["git", "describe", "--tags", "--match", "xlog-cli-v*", "--abbrev=0"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if described.returncode == 0:
        tag = described.stdout.strip()
        return f"{tag}..HEAD", tag
    return "HEAD", "repository start"


def release_worthy(repository: Path, revision_range: str) -> bool:
    subjects = _git(repository, "log", "--format=%s", revision_range)
    bodies = _git(repository, "log", "--format=%B", revision_range)
    return any(
        RELEASE_SUBJECT.match(subject) for subject in subjects.splitlines()
    ) or bool(BREAKING_FOOTER.search(bodies))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="Git repository to inspect (default: current directory).",
    )
    parser.add_argument(
        "--range",
        dest="revision_range",
        help="Explicit Git revision range; defaults to the latest xlog-cli tag through HEAD.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    revision_range, origin = (
        (args.revision_range, args.revision_range)
        if args.revision_range is not None
        else _default_range(args.repository)
    )
    try:
        worthy = release_worthy(args.repository, revision_range)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = (
            error.stderr.strip()
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        print(f"git log failed for range {revision_range}: {detail}", file=sys.stderr)
        return 2

    if worthy:
        print(f"Release-worthy commits found in {revision_range} (from {origin}).")
        return 0
    print(f"No release-worthy commits found in {revision_range} (from {origin}).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
