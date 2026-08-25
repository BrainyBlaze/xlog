from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_release_workflow_uses_tested_worthiness_program() -> None:
    workflow = (ROOT / ".github/workflows/release-plz.yml").read_text(encoding="utf-8")

    assert "python3 scripts/release_worthiness.py" in workflow
    assert 'case "$release_worthiness_status" in' in workflow
    assert "git log --format=%s" not in workflow
    assert "grep -qE" not in workflow


def test_changelog_guard_uses_base_workflow_without_checkout() -> None:
    workflow = (ROOT / ".github/workflows/changelog-ownership.yml").read_text(
        encoding="utf-8"
    )

    assert "pull_request_target:" in workflow
    assert "changelog-ownership:" in workflow
    assert "actions/checkout" not in workflow
    assert 'index("release")' in workflow
    assert 'filename == "CHANGELOG.md"' in workflow


def test_claude_pointer_is_tracked_by_repository_rules() -> None:
    pointer = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "AGENTS.md" in pointer
    assert "ENGINEERING.md" in pointer
    assert "/CLAUDE.md" not in ignore
