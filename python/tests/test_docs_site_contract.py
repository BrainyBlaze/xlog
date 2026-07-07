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
    assert [tab["tab"] for tab in config["navigation"]["tabs"]] == [
        "Documentation",
        "Architecture",
        "Reference",
    ]
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


def test_reference_nav_includes_configuration_pages() -> None:
    config = json.loads(read("docs-site/docs.json"))
    reference_tab = next(
        tab for tab in config["navigation"]["tabs"] if tab["tab"] == "Reference"
    )
    groups = {group["group"]: group["pages"] for group in reference_tab["groups"]}
    assert groups["Configuration"] == [
        "reference/environment-variables",
        "reference/errors",
    ]


def test_mintlify_config_enables_copy_page_markdown_action() -> None:
    config = json.loads(read("docs-site/docs.json"))
    assert config["contextual"]["options"] == ["copy"]
    assert config["contextual"]["display"] == "header"


def test_custom_domain_is_present_for_app_platform_artifact() -> None:
    app = yaml.safe_load(read(".do/docs-app.yaml"))
    domains = {entry["domain"]: entry for entry in app["domains"]}
    assert domains["xlog.md"]["type"] == "PRIMARY"
    assert domains["www.xlog.md"]["type"] == "ALIAS"
    site = app["static_sites"][0]
    assert site["github"]["branch"] == "docs-dist"
    assert site["github"]["deploy_on_push"] is True
    assert site["error_document"] == "404.html"


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
    assert "XLOG_RUSTDOC_NO_CUDA=1" in script
    assert "XLOG_RUSTDOC_OUTPUT_DIR" in script
    assert ".site-rustdoc/generated/rust" in script
    assert "docs-site/generated/rust" not in script

    cuda_build = read("crates/xlog-cuda/build.rs")
    assert "XLOG_RUSTDOC_NO_CUDA" in cuda_build
    assert "DOCS_RS" in cuda_build
    assert "write_empty_embedded_kernel_data" in cuda_build


def test_rust_api_page_links_to_generated_crate_roots() -> None:
    rust_page = read("docs-site/reference/rust.mdx")
    assert "generated/rust/index.html" in rust_page
    assert "generated/rust/pyxlog/index.html" in rust_page
    assert "generated Rustdoc is attached after Mintlify export" in rust_page
    assert "docs-site/generated/rust" not in rust_page


def test_docs_workflow_attaches_rustdoc_after_mintlify_export() -> None:
    workflow = read(".github/workflows/docs-site.yml")
    assert "XLOG_RUSTDOC_OUTPUT_DIR" in workflow
    assert ".site-rustdoc/generated/rust" in workflow
    assert "Attach generated Rust API docs" in workflow
    assert "cp -R .site-rustdoc/generated/rust .site-dist/generated/rust" in workflow
    assert workflow.index("mint export") < workflow.index("Attach generated Rust API docs")


def test_docs_workflow_builds_self_hosted_search_before_rustdoc_graft() -> None:
    workflow = read(".github/workflows/docs-site.yml")
    assert "Build self-hosted search index (Pagefind)" in workflow
    assert "scripts/docs/build_markdown_exports.py docs-site .site-dist" in workflow
    assert "scripts/docs/inject_search_shim.py .site-dist" in workflow
    assert "pagefind@1.5.2 --site .site-dist" in workflow
    assert "test -f .site-dist/index.md" in workflow
    assert "test -f .site-dist/architecture/gpu-execution.md" in workflow
    assert "test -f .site-dist/pagefind/pagefind-ui.js" in workflow
    assert "test -f .site-dist/pagefind/pagefind-ui.css" in workflow
    assert '"scripts/docs/build_markdown_exports.py"' in workflow
    assert '"scripts/docs/inject_search_shim.py"' in workflow
    assert '"scripts/docs/copy-page-shim.js"' in workflow
    assert '"scripts/docs/search-shim.js"' in workflow
    assert '"scripts/docs/search-shim.css"' in workflow
    assert workflow.index("mint export") < workflow.index("Build self-hosted search index")
    assert workflow.index("Build self-hosted search index") < workflow.index("Attach generated Rust API docs")

    injector = read("scripts/docs/inject_search_shim.py")
    assert "data-pagefind-body" in injector
    assert "/copy-page-shim.js" in injector
    assert "/search-shim.js" in injector
    assert "/search-shim.css" in injector


def test_search_shim_uses_pagefind_assets_and_suppresses_mintlify_search() -> None:
    shim = read("scripts/docs/search-shim.js")
    assert "/pagefind/pagefind-ui.js" in shim
    assert "/pagefind/pagefind-ui.css" in shim
    assert "stopImmediatePropagation" in shim
    assert "#search-bar-entry" in shim
    assert 'key === "k"' in shim


def test_copy_page_shim_uses_markdown_export_and_suppresses_mintlify_copy() -> None:
    shim = read("scripts/docs/copy-page-shim.js")
    assert 'button[aria-label="Copy page"]' in shim
    assert 'link[rel="alternate"][type="text/markdown"]' in shim
    assert "navigator.clipboard.writeText" in shim
    assert "stopImmediatePropagation" in shim
    assert "xlog-copy-source" in shim


def test_static_markdown_export_generator_writes_route_markdown(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/docs/build_markdown_exports.py",
            "docs-site",
            str(tmp_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout

    index = tmp_path / "index.md"
    gpu = tmp_path / "architecture/gpu-execution.md"
    assert index.exists()
    assert gpu.exists()
    assert index.read_text(encoding="utf-8").startswith("# XLOG Documentation\n\n")
    gpu_text = gpu.read_text(encoding="utf-8")
    assert gpu_text.startswith("# GPU Execution\n\n")
    assert "XLOG's deterministic runtime" in gpu_text
    assert "title:" not in gpu_text.splitlines()[:5]


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
        ".do/docs-app.yaml",
        "Publish to docs-dist branch",
        "permissions:",
        "contents: write",
        "if: github.ref == 'refs/heads/main'",
    ]:
        assert expected in workflow


def test_docs_workflow_materializes_redirect_stubs_after_search_indexing() -> None:
    workflow = read(".github/workflows/docs-site.yml")
    assert '"scripts/docs/build_redirect_stubs.py"' in workflow
    assert "Add legacy redirects and 404 page" in workflow
    assert "scripts/docs/build_redirect_stubs.py docs-site .site-dist" in workflow
    assert "test -f .site-dist/404.html" in workflow
    assert "test -f .site-dist/architecture/xlog-prob/index.html" in workflow
    assert workflow.index("Build self-hosted search index") < workflow.index(
        "Add legacy redirects and 404 page"
    )
    assert workflow.index("Add legacy redirects and 404 page") < workflow.index(
        "Attach generated Rust API docs"
    )

    script = read("scripts/docs/build_redirect_stubs.py")
    assert "Never clobber a real exported page" in script
    assert "404.html written" in script


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
