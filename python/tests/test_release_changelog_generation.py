from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 local fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]

EPISTEMIC_RELEASE_BEHAVIORS = (
    "Generate-Propagate-Test for acyclic programs",
    "founded least fixpoints for positive FAEEL cycles",
    "greatest compatible exact-tuple fixpoints for supported positive Gelfond-1991 possibility cycles",
    "GPU-backed WFS for supported cycles through negation",
    "GPU upper bound and reevaluate against frozen relation snapshots until concrete tuples converge",
    "disjoint tuple domains cannot manufacture support",
    "empty founded extension for unseeded FAEEL cycles",
)


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / name
    assert path.exists(), f"missing release helper: {path}"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _legacy_release_section(
    package: str,
    version: str,
    *,
    previous: str = "0.11.0",
    date: str = "2026-08-16",
    entry: str = "- generated entry",
) -> str:
    return (
        f"## [{version}](https://example.invalid/org/repo/compare/"
        f"{package}-v{previous}...{package}-v{version}) - {date}\n\n"
        f"### Fixed\n\n{entry}\n\n"
    )


def _package_release_section(
    package: str,
    version: str,
    *,
    linked: bool = False,
    previous: str = "0.11.0",
    date: str = "2026-08-16",
    entry: str = "- generated entry",
) -> str:
    link = ""
    if linked:
        link = (
            "(https://example.invalid/org/repo/compare/"
            f"{package}-v{previous}...{package}-v{version})"
        )
    return f"## {package} [{version}]{link} - {date}\n\n### Fixed\n\n{entry}\n\n"


def test_unpublished_cleanup_accepts_legacy_and_package_labelled_sections() -> None:
    helper = _load_script("prepare_release_changelog.py")
    current = "0.12.0"
    changelog = (
        "# Changelog\n\n## [Unreleased]\n\n"
        + _legacy_release_section("xlog-core", current, entry="- legacy core")
        + _package_release_section("xlog-cli", current, linked=True, entry="- new cli")
        + _package_release_section("xlog-core", current, entry="- new core")
        + _legacy_release_section("xlog-cli", "0.11.0", previous="0.10.0")
    )

    updated, removed = helper.remove_unpublished_release_sections(
        changelog,
        version=current,
        packages={"xlog-cli", "xlog-core"},
        release_tag_exists=False,
    )

    assert removed == 3
    assert "legacy core" not in updated
    assert "new cli" not in updated
    assert "new core" not in updated
    assert "xlog-cli-v0.10.0...xlog-cli-v0.11.0" in updated

    second, removed_again = helper.remove_unpublished_release_sections(
        updated,
        version=current,
        packages={"xlog-cli", "xlog-core"},
        release_tag_exists=False,
    )
    assert second == updated
    assert removed_again == 0


def test_package_labelled_parser_rejects_cross_package_compare_links() -> None:
    helper = _load_script("prepare_release_changelog.py")
    mismatched = (
        "## xlog-core [0.12.0](https://example.invalid/org/repo/compare/"
        "xlog-cli-v0.11.0...xlog-cli-v0.12.0) - 2026-08-16\n"
    )

    assert helper.release_sections(mismatched) == []


def test_published_release_cleanup_is_a_noop() -> None:
    helper = _load_script("prepare_release_changelog.py")
    changelog = _package_release_section("xlog-cli", "0.12.0", linked=True)

    updated, removed = helper.remove_unpublished_release_sections(
        changelog,
        version="0.12.0",
        packages={"xlog-cli"},
        release_tag_exists=True,
    )

    assert updated == changelog
    assert removed == 0


def test_release_context_uses_one_authoritative_tag() -> None:
    helper = _load_script("prepare_release_changelog.py")
    context = helper.load_release_context(
        ROOT / "Cargo.toml", ROOT / "release-plz.toml"
    )

    assert context.version == "0.12.0"
    assert context.authoritative_package == "xlog-cli"
    assert context.release_tag == "xlog-cli-v0.12.0"
    assert len(context.changelog_packages) == 10


def test_changelog_template_labels_packages_and_links_only_authoritative_tag() -> None:
    config = tomllib.loads((ROOT / "release-plz.toml").read_text(encoding="utf-8"))
    body = config["changelog"]["body"]

    assert "## {{ package }} [{{ version }}]" in body
    assert 'package == "xlog-cli"' in body
    assert "({{ release_link }})" in body
    assert body.count("{{ release_link }}") == 1


def test_release_pr_template_uses_workspace_targets_for_git_only_packages() -> None:
    config = tomllib.loads((ROOT / "release-plz.toml").read_text(encoding="utf-8"))
    body = config["workspace"]["pr_body"]

    assert "## Workspace release" in body
    assert "{% if version" not in body
    assert "workspace target" in body
    assert 'release.package == "pyxlog"' in body
    assert 'release.package != "pyxlog"' in body
    assert "crates.io baseline" in body
    assert "registry baseline" not in body
    assert "published separately to PyPI" in body
    assert "initial release" not in body.lower()


def test_historical_subject_normalization_uses_behavioral_release_notes() -> None:
    config = tomllib.loads((ROOT / "release-plz.toml").read_text(encoding="utf-8"))
    preprocessors = config["changelog"]["commit_preprocessors"]
    replacements = {entry["replace"] for entry in preprocessors}

    assert (
        "feat(runtime): expose epistemic uncertainty over accepted world views"
        in replacements
    )
    assert "format accepted-world-view epistemic exports" in replacements


def test_generator_restores_recursive_epistemic_behavior_after_cleanup() -> None:
    helper = _load_script("prepare_release_changelog.py")
    historical_notes = "\n".join(
        f"- {behavior}" for behavior in EPISTEMIC_RELEASE_BEHAVIORS
    )
    changelog = (
        "# Changelog\n\n## [Unreleased]\n\n"
        + _legacy_release_section("xlog-logic", "0.12.0", entry=historical_notes)
        + _legacy_release_section("xlog-cli", "0.11.0", previous="0.10.0")
    )

    cleaned, removed = helper.remove_unpublished_release_sections(
        changelog,
        version="0.12.0",
        packages={"xlog-logic"},
        release_tag_exists=False,
    )

    assert removed == 1
    assert all(behavior not in cleaned for behavior in EPISTEMIC_RELEASE_BEHAVIORS)

    config = tomllib.loads((ROOT / "release-plz.toml").read_text(encoding="utf-8"))
    exact_subject = (
        "fix(logic): preserve predicate unions and recursive epistemic semantics (#195)"
    )
    preprocessors = config["changelog"]["commit_preprocessors"]

    def normalize(message: str) -> str:
        for preprocessor in preprocessors:
            message = re.sub(preprocessor["pattern"], preprocessor["replace"], message)
        return message

    normalized_subject = normalize(exact_subject)
    assert normalized_subject.startswith("fix(logic): ")
    assert normalized_subject.endswith(" (#195)")
    assert all(
        behavior in normalized_subject for behavior in EPISTEMIC_RELEASE_BEHAVIORS
    )

    commit_with_body = exact_subject + "\n\n* preserve the squash commit body"
    normalized_commit = normalize(commit_with_body)
    assert normalized_commit.splitlines()[0] == normalized_subject
    assert normalized_commit.endswith("* preserve the squash commit body")

    near_matches = (
        "fix(runtime): preserve predicate unions and recursive epistemic semantics (#195)",
        "docs(logic): preserve predicate unions and recursive epistemic semantics (#195)",
        "fix(logic): preserve predicate unions and recursive epistemic semantics (#194)",
        exact_subject + " follow-up",
        "revert: " + exact_subject,
        "docs: quote a historical subject\n\n" + exact_subject,
    )
    assert all(normalize(candidate) == candidate for candidate in near_matches)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def test_commit_subject_validator_rejects_only_misplaced_breaking_bang() -> None:
    validator = _load_script("validate_release_commit_subjects.py")
    commits = [
        ("a" * 40, "feat(runtime)!: valid breaking subject"),
        ("b" * 40, "fix: ordinary subject"),
        ("c" * 40, "feat!(runtime): malformed breaking subject"),
        ("d" * 40, "custom-type!(scope): malformed custom type"),
    ]

    issues = validator.find_malformed_subjects(commits)

    assert [(issue.sha, issue.subject) for issue in issues] == commits[2:]


def test_commit_subject_cli_checks_exactly_base_exclusive_range(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "validate_release_commit_subjects.py"
    assert script.exists()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Release Test")
    _git(repo, "config", "user.email", "release-test@example.invalid")
    _git(repo, "commit", "--allow-empty", "-m", "feat!(runtime): historical malformed")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "commit", "--allow-empty", "-m", "feat(runtime)!: valid breaking")
    valid_head = _git(repo, "rev-parse", "HEAD")

    valid = subprocess.run(
        [sys.executable, str(script), base, valid_head],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert valid.returncode == 0, valid.stderr or valid.stdout
    assert "1 newly introduced commit subject" in valid.stdout

    _git(repo, "commit", "--allow-empty", "-m", "fix!(parser): malformed new commit")
    invalid_head = _git(repo, "rev-parse", "HEAD")
    invalid = subprocess.run(
        [sys.executable, str(script), valid_head, invalid_head],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 1
    assert invalid_head in invalid.stderr
    assert "fix!(parser): malformed new commit" in invalid.stderr
    assert "feat!(runtime): historical malformed" not in invalid.stderr


def test_workflows_prepare_generation_and_lint_exact_event_range() -> None:
    release_workflow = (ROOT / ".github/workflows/release-plz.yml").read_text(
        encoding="utf-8"
    )
    prepare_command = "python3 scripts/prepare_release_changelog.py"
    generator_command = "release-plz release-pr"
    tomli_install = "python3 -m pip install --no-cache-dir tomli==2.2.1"
    assert tomli_install in release_workflow
    assert release_workflow.index(tomli_install) < release_workflow.index(
        prepare_command
    )
    assert prepare_command in release_workflow
    assert release_workflow.index(prepare_command) < release_workflow.index(
        generator_command
    )
    assert (
        "--allow-dirty" in release_workflow[release_workflow.index(generator_command) :]
    )
    assert "git status --porcelain=v1 --untracked-files=all" in release_workflow
    assert '[[ "$entry" == " M CHANGELOG.md" ]]' in release_workflow

    ci_workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    git_hygiene_job = ci_workflow.split("  git-hygiene:\n", 1)[1].split(
        "  python-contract:\n", 1
    )[0]
    assert "fetch-depth: 0" in git_hygiene_job
    assert (
        "BASE_SHA: ${{ github.event.pull_request.base.sha || github.event.before }}"
        in git_hygiene_job
    )
    assert (
        "HEAD_SHA: ${{ github.event.pull_request.head.sha || github.sha }}"
        in git_hygiene_job
    )
    assert (
        'python3 scripts/validate_release_commit_subjects.py "$BASE_SHA" "$HEAD_SHA"'
        in git_hygiene_job
    )

    python_contract_job = ci_workflow.split("  python-contract:\n", 1)[1].split(
        "  caviar-examples:\n", 1
    )[0]
    dependency_install = "python -m pip install --upgrade pip pytest PyYAML"
    pytest_command = "python -m pytest -q"
    assert dependency_install in python_contract_job
    assert pytest_command in python_contract_job
    assert python_contract_job.index(dependency_install) < python_contract_job.index(
        pytest_command
    )
    pytest_invocation = python_contract_job[python_contract_job.index(pytest_command) :]
    for test_path in (
        "python/tests/test_relation_provenance_contract.py",
        "python/tests/test_docs_contract.py",
        "python/tests/test_release_changelog_generation.py",
    ):
        assert test_path in pytest_invocation


def test_current_changelog_uses_package_sections_and_one_cli_link() -> None:
    helper = _load_script("prepare_release_changelog.py")
    context = helper.load_release_context(
        ROOT / "Cargo.toml", ROOT / "release-plz.toml"
    )
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    current = [
        section
        for section in helper.release_sections(changelog)
        if section.version == context.version
    ]

    assert {section.package for section in current} == context.changelog_packages
    assert len(current) == len(context.changelog_packages)
    current_text = "".join(
        changelog[section.start : section.end] for section in current
    )
    compare_targets = re.findall(r"/compare/([^)]*)", current_text)
    assert compare_targets == [f"xlog-cli-v0.11.0...{context.release_tag}"]
    assert "feat!(runtime):" not in current_text
    assert (
        "- *(runtime)* [**breaking**] mark the output stats structs non_exhaustive"
        in current_text
    )
    assert re.search(r"\b[A-Z]+-[0-9]+\b", current_text) is None
    assert "expose epistemic uncertainty over accepted world views" in current_text
    assert "format accepted-world-view epistemic exports" in current_text
    assert all(behavior in current_text for behavior in EPISTEMIC_RELEASE_BEHAVIORS)
