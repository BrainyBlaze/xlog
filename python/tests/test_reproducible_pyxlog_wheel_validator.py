from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import validate_reproducible_pyxlog_wheel as validator


def test_source_date_epoch_preserves_a_valid_caller_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1720000000")

    def unexpected_git_call(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError(
            f"git must not run when SOURCE_DATE_EPOCH is set: {args} {kwargs}"
        )

    monkeypatch.setattr(validator.subprocess, "run", unexpected_git_call)

    assert validator.resolve_source_date_epoch(Path("/unused")) == "1720000000"


def test_source_date_epoch_defaults_to_the_checked_out_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    repo_root = Path("/repository")
    calls: list[tuple[list[str], Path]] = []

    def completed_git_call(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check
        assert capture_output
        assert text
        calls.append((command, cwd))
        return subprocess.CompletedProcess(command, 0, stdout="1730000000\n")

    monkeypatch.setattr(validator.subprocess, "run", completed_git_call)

    assert validator.resolve_source_date_epoch(repo_root) == "1730000000"
    assert calls == [(["git", "show", "-s", "--format=%ct", "HEAD"], repo_root)]


@pytest.mark.parametrize("value", ["", "-1", "not-a-timestamp"])
def test_source_date_epoch_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", value)

    with pytest.raises(RuntimeError, match="SOURCE_DATE_EPOCH"):
        validator.resolve_source_date_epoch(Path("/unused"))
