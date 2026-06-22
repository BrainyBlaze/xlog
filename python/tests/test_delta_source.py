from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_delta_api_is_exposed_in_stubs_and_python_docs() -> None:
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    init_stub = (ROOT / "crates/pyxlog/python/pyxlog/__init__.pyi").read_text()
    docs = (ROOT / "docs/architecture/python-bindings.md").read_text()

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


def test_delta_has_certified_runtime_evidence() -> None:
    evidence_root = ROOT / "docs/evidence"
    evidence = next(
        path / "README.md"
        for path in sorted(evidence_root.glob("*-delta"))
        if (path / "README.md").exists()
        and "RelationDelta" in (path / "README.md").read_text()
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
