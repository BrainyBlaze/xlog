#!/usr/bin/env python3
"""Reject malformed Conventional Commit breaking markers in a Git range."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


MISPLACED_BREAKING_BANG = re.compile(r"^[A-Za-z][A-Za-z0-9-]*!\([^()]+\):")


@dataclass(frozen=True)
class SubjectIssue:
    sha: str
    subject: str


def find_malformed_subjects(
    commits: list[tuple[str, str]],
) -> list[SubjectIssue]:
    return [
        SubjectIssue(sha=sha, subject=subject)
        for sha, subject in commits
        if MISPLACED_BREAKING_BANG.match(subject)
    ]


def _run_git(repository: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "git failed"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout.strip()


def newly_introduced_subjects(
    repository: Path, base: str, head: str
) -> list[tuple[str, str]]:
    _run_git(repository, "rev-parse", "--verify", f"{base}^{{commit}}")
    _run_git(repository, "rev-parse", "--verify", f"{head}^{{commit}}")
    revision_output = _run_git(repository, "rev-list", "--reverse", head, f"^{base}")
    revisions = [line for line in revision_output.splitlines() if line]
    return [
        (revision, _run_git(repository, "show", "-s", "--format=%s", revision))
        for revision in revisions
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", help="base commit excluded from validation")
    parser.add_argument("head", help="head commit included in validation")
    parser.add_argument("--repository", type=Path, default=Path("."))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    commits = newly_introduced_subjects(args.repository.resolve(), args.base, args.head)
    issues = find_malformed_subjects(commits)
    if issues:
        print(
            "Malformed breaking Conventional Commit subjects place `!` before "
            "the scope; use `type(scope)!:` instead:",
            file=sys.stderr,
        )
        for issue in issues:
            print(f"  {issue.sha} {issue.subject}", file=sys.stderr)
        return 1

    noun = "subject" if len(commits) == 1 else "subjects"
    print(f"Validated {len(commits)} newly introduced commit {noun}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
