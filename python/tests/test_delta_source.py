from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = ROOT / "docs-internal/evidence"

# EVIDENCE-GATED: only the certified-runtime-evidence test below needs this
# tree; the other tests in this module are pure source-contract checks and must
# keep running on a clean clone, so the gate is per-test, not a module-level
# `pytestmark`.
_EVIDENCE_SKIP_REASON = (
    "no local evidence workspace at docs-internal/evidence -- that tree is a "
    "local-only agent workspace excluded by .gitignore (the '/docs-internal/' "
    "entry, labelled 'Local-only agent workspaces'), so it is deliberately not "
    "shipped and is absent on a clean clone. When it IS present this test "
    "checks that some 'docs-internal/evidence/*-delta' directory holds a "
    "README.md naming RelationDelta next to a runtime_probe.json, and that the "
    "README records the API coverage, equivalence, delete-correctness, "
    "monotone-insert-path and delta-fixture results. Re-run the delta evidence "
    "job locally to recreate the package and exercise it."
)


def test_delta_api_is_exposed_in_stubs_and_python_docs() -> None:
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    init_stub = (ROOT / "crates/pyxlog/python/pyxlog/__init__.pyi").read_text()
    docs = (ROOT / "python/tests/contract_docs/python-bindings.md").read_text()

    for needle in [
        "def insert_relation(",
        "def delete_relation(",
        "def apply_relation_delta(",
        "def delta_stats(",
    ]:
        assert needle in native_stub

    for needle in [
        "insert_relation",
        "delete_relation",
        "apply_relation_delta",
        "delta_stats",
    ]:
        assert needle in init_stub or needle in docs


def test_delta_routes_through_runtime_relation_delta() -> None:
    logic_rs = (ROOT / "crates/pyxlog/src/logic.rs").read_text()
    gpu_logic_rs = (ROOT / "crates/xlog-gpu/src/logic.rs").read_text()
    rewrite_rs = (ROOT / "crates/xlog-runtime/src/executor/rewrite.rs").read_text()
    executor_rs = (ROOT / "crates/xlog-runtime/src/executor/mod.rs").read_text()

    assert "RelationDelta::new" in logic_rs
    assert "apply_relation_deltas" in gpu_logic_rs
    assert "apply_deltas_and_recompute" in gpu_logic_rs
    assert "DeltaRecomputeStats" in executor_rs
    assert "incremental_sccs" in rewrite_rs


@pytest.mark.skipif(not EVIDENCE_ROOT.is_dir(), reason=_EVIDENCE_SKIP_REASON)
def test_delta_has_certified_runtime_evidence() -> None:
    evidence = next(
        (
            path / "README.md"
            for path in sorted(EVIDENCE_ROOT.glob("*-delta"))
            if (path / "README.md").exists()
            and "RelationDelta" in (path / "README.md").read_text()
        ),
        None,
    )
    assert evidence is not None, (
        "docs-internal/evidence exists but holds no '*-delta' directory with a "
        "README.md mentioning RelationDelta -- the delta evidence package is "
        "missing or incomplete in this local workspace"
    )
    probe = evidence.parent / "runtime_probe.json"

    assert evidence.exists()
    assert probe.exists()

    text = evidence.read_text()
    for needle in [
        "API coverage",
        "equivalence",
        "delete correctness",
        "monotone insert path",
        "delta fixture",
        "RelationDelta",
    ]:
        assert needle in text
