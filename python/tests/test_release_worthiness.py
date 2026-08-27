from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "release_worthiness.py"


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--initial-branch=main")
    _git(repository, "config", "user.name", "Release Test")
    _git(repository, "config", "user.email", "release-test@example.invalid")
    return repository


def _commit(repository: Path, subject: str, body: str | None = None) -> None:
    args = ["commit", "--allow-empty", "-m", subject]
    if body is not None:
        args.extend(["-m", body])
    _git(repository, *args)


def _run(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repository", str(repository), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_feat_commit_is_release_worthy(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _commit(repository, "feat: add production capability")

    completed = _run(repository)

    assert completed.returncode == 0, completed.stderr
    assert "release-worthy commits found" in completed.stdout.lower()


@pytest.mark.parametrize(
    "subject",
    [
        "fix(core): correct execution",
        "perf: reduce allocations",
        "refactor!: remove obsolete API",
        "docs(runtime): document behavior",
        "build: update compiler image",
        "ci: validate complete workspace",
        "test: cover regression",
        "revert: restore supported behavior",
    ],
)
def test_every_configured_release_type_is_worthy(tmp_path: Path, subject: str) -> None:
    repository = _repository(tmp_path)
    _commit(repository, subject)

    assert _run(repository).returncode == 0


def test_breaking_change_footer_is_release_worthy(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _commit(
        repository,
        "chore: reorganize workspace",
        "BREAKING CHANGE: remove the obsolete configuration surface",
    )

    assert _run(repository).returncode == 0


@pytest.mark.parametrize(
    "subject",
    [
        "chore: maintain repository",
        "style: format comments",
        "Merge pull request #123 from contributor/topic",
        "Fix: capitalized non-conventional subject",
    ],
)
def test_non_release_commits_are_not_worthy(tmp_path: Path, subject: str) -> None:
    repository = _repository(tmp_path)
    _commit(repository, subject)

    completed = _run(repository)

    assert completed.returncode == 1
    assert "no release-worthy commits" in completed.stdout.lower()


def test_only_commits_after_latest_cli_tag_are_considered(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _commit(repository, "feat: already released")
    _git(repository, "tag", "xlog-cli-v0.11.0")
    _commit(repository, "chore: post-release maintenance")

    completed = _run(repository)

    assert completed.returncode == 1
    assert "xlog-cli-v0.11.0" in completed.stdout


def test_explicit_range_overrides_tag_discovery(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _commit(repository, "chore: baseline")
    baseline = _git(repository, "rev-parse", "HEAD").stdout.strip()
    _commit(repository, "fix: included correction")

    completed = _run(repository, "--range", f"{baseline}..HEAD")

    assert completed.returncode == 0
    assert f"{baseline}..HEAD" in completed.stdout


def test_invalid_range_is_an_execution_error(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _commit(repository, "feat: valid commit")

    completed = _run(repository, "--range", "missing-ref..HEAD")

    assert completed.returncode == 2
    assert "git log failed" in completed.stderr.lower()
