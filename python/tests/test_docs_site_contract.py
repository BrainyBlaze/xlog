from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_mkdocs_config_defines_curated_docs_and_reference_nav() -> None:
    config = yaml.safe_load(read("mkdocs.yml"))
    assert config["site_name"] == "XLOG"
    assert config["site_url"] == "https://xlog.md/"
    nav_text = repr(config["nav"])
    for expected in [
        "Language Reference",
        "Architecture",
        "Python API",
        "Rust API",
        "CUDA API",
    ]:
        assert expected in nav_text
    assert "docs/evidence" not in nav_text
    assert "docs/plans" not in nav_text


def test_custom_domain_file_is_present_for_pages_artifact() -> None:
    assert read("docs/CNAME").strip() == "xlog.md"


def test_docs_build_script_generates_reference_outputs() -> None:
    script = read("scripts/docs/build_docs.sh")
    for expected in [
        "gen_pyxlog_api.py",
        "cargo doc --workspace --no-deps",
        "cargo metadata --locked --no-deps --format-version=1",
        "target_directory",
        "doxygen Doxyfile.docs",
        "mkdocs build",
    ]:
        assert expected in script
    assert "target/doc" not in script


def test_rust_api_page_links_to_generated_crate_roots() -> None:
    rust_page = read("docs/api/rust.md")
    assert "generated/rust/index.html" in rust_page
    assert "generated/rust/pyxlog/index.html" in rust_page
    assert "docs/api/generated/rust/index.html" in read("scripts/docs/build_docs.sh")


def test_home_page_omits_local_generated_html_notice() -> None:
    home = read("docs/index.md")
    assert "Generated HTML is not committed" not in home
    assert "make docs when Rust and Doxygen dependencies are available" not in home


def test_internal_agent_workspace_paths_are_local_only() -> None:
    ignored = read(".gitignore")
    agents = read("AGENTS.md")
    claude = read("CLAUDE.md")
    for path in [
        "docs/evidence",
        "docs/plans",
        "docs/reports",
        "docs/superpowers",
    ]:
        assert f"{path}/" in ignored
        assert path in agents
        assert path in claude
    for guidance in [agents, claude]:
        assert "local-only agent workspaces" in guidance
        assert "must not be staged, committed, or pushed" in guidance


def test_github_docs_workflow_deploys_pages_only_from_main_docs_changes() -> None:
    workflow = read(".github/workflows/docs.yml")
    for expected in [
        "docs/**",
        "mkdocs.yml",
        "actions/deploy-pages",
        "permissions:",
        "id-token: write",
        "pages: write",
        "if: github.ref == 'refs/heads/main'",
    ]:
        assert expected in workflow


def test_pyxlog_stub_generator_extracts_classes_and_methods() -> None:
    sample = '''
class LogicProgram:
    """Factory."""

    @staticmethod
    def compile(source: str, device: int = 0) -> CompiledLogicProgram: ...
'''
    result = subprocess.run(
        [sys.executable, "scripts/docs/gen_pyxlog_api.py", "--stdin"],
        cwd=ROOT,
        input=sample,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "## LogicProgram" in result.stdout
    assert "compile(source: str, device: int = 0)" in result.stdout
