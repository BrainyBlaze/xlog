from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v080_delta_api_is_exposed_in_stubs_and_python_docs() -> None:
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


def test_v080_delta_routes_through_runtime_relation_delta() -> None:
    logic_rs = (ROOT / "crates/pyxlog/src/logic.rs").read_text()
    gpu_logic_rs = (ROOT / "crates/xlog-gpu/src/logic.rs").read_text()
    rewrite_rs = (ROOT / "crates/xlog-runtime/src/executor/rewrite.rs").read_text()
    executor_rs = (ROOT / "crates/xlog-runtime/src/executor/mod.rs").read_text()

    assert "RelationDelta::new" in logic_rs
    assert "apply_relation_deltas" in gpu_logic_rs
    assert "apply_deltas_and_recompute" in gpu_logic_rs
    assert "DeltaRecomputeStats" in executor_rs
    assert "incremental_sccs" in rewrite_rs


def test_v080_delta_has_certified_stage4_evidence() -> None:
    evidence = ROOT / "docs/evidence/2026-05-18-v080-delta/README.md"
    probe = ROOT / "docs/evidence/2026-05-18-v080-delta/runtime_probe.json"

    assert evidence.exists()
    assert probe.exists()

    text = evidence.read_text()
    for needle in [
        "M080_DELTA.1",
        "M080_DELTA.2",
        "M080_DELTA.3",
        "M080_DELTA.4",
        "M080_DELTA.5",
        "wmir_committed",
    ]:
        assert needle in text
