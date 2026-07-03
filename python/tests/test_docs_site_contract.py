from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_mintlify_config_defines_curated_docs_and_reference_nav() -> None:
    config = json.loads(read("docs-site/docs.json"))
    assert config["name"] == "XLOG"
    tab_text = repr(config["navigation"]["tabs"])
    for expected in [
        "Documentation",
        "Reference",
        "Architecture",
        "reference/python",
        "reference/rust",
        "reference/cuda",
    ]:
        assert expected in tab_text
    assert "docs/evidence" not in tab_text
    assert "docs/plans" not in tab_text


def test_custom_domain_is_present_for_app_platform_artifact() -> None:
    app = yaml.safe_load(read(".do/docs-app.yaml"))
    domains = {entry["domain"]: entry for entry in app["domains"]}
    assert domains["xlog.md"]["type"] == "PRIMARY"
    assert domains["www.xlog.md"]["type"] == "ALIAS"
    site = app["static_sites"][0]
    assert site["github"]["branch"] == "docs-dist"
    assert site["github"]["deploy_on_push"] is True


def test_docs_workflow_generates_reference_outputs_and_exports() -> None:
    workflow = read(".github/workflows/docs-site.yml")
    for expected in [
        "docs-site/**",
        "node-version: \"22\"",
        "mint@4.2.666",
        "scripts/docs/build_rust_api.sh",
        "mint validate",
        "mint broken-links",
        "mint export",
        "docs-dist",
    ]:
        assert expected in workflow

    script = read("scripts/docs/build_rust_api.sh")
    assert "cargo doc --workspace --no-deps --locked" in script
    assert "docs-site/generated/rust" in script


def test_rust_api_page_links_to_generated_crate_roots() -> None:
    rust_page = read("docs-site/reference/rust.mdx")
    assert "generated/rust/index.html" in rust_page
    assert "generated/rust/pyxlog/index.html" in rust_page
    assert "docs-site/generated/rust" in read("scripts/docs/build_rust_api.sh")


def test_home_page_omits_local_generated_html_notice() -> None:
    home = read("docs-site/index.mdx")
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


def test_github_docs_workflow_deploys_docs_dist_only_from_main_docs_site_changes() -> None:
    workflow = read(".github/workflows/docs-site.yml")
    for expected in [
        "docs-site/**",
        "Publish to docs-dist branch",
        "permissions:",
        "contents: write",
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
