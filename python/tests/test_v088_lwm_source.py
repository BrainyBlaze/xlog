from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v088_rule_and_proof_introspection_is_exposed() -> None:
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    docs = (ROOT / "docs/architecture/python-bindings.md").read_text()
    logic_rs = (ROOT / "crates/pyxlog/src/logic.rs").read_text()
    main_rs = (ROOT / "crates/xlog-cli/src/main.rs").read_text()

    for needle in [
        "def rule_provenance(",
        "def proof_traces(",
    ]:
        assert needle in native_stub
        assert needle in docs

    for needle in [
        "pub fn rule_provenance(",
        "pub fn proof_traces(",
        "rule_provenance",
        "proof_traces",
    ]:
        assert needle in logic_rs or needle in main_rs


def test_v088_delta_debug_reports_relations_equivalence_and_trace() -> None:
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    docs = (ROOT / "docs/architecture/python-bindings.md").read_text()
    logic_rs = (ROOT / "crates/pyxlog/src/logic.rs").read_text()

    for needle in [
        "def apply_relation_delta_debug(",
        "changed_relation_names",
        "equivalent_to_full_recompute",
        "debug_trace",
    ]:
        assert needle in native_stub
        assert needle in docs
    assert "pub fn apply_relation_delta_debug(" in logic_rs
    for needle in [
        "changed_relation_names",
        "equivalent_to_full_recompute",
        "debug_trace",
    ]:
        assert needle in logic_rs


def test_v088_temporal_provenance_helpers_are_exposed() -> None:
    init_py = (ROOT / "crates/pyxlog/python/pyxlog/__init__.py").read_text()
    init_stub = (ROOT / "crates/pyxlog/python/pyxlog/__init__.pyi").read_text()
    docs = (ROOT / "docs/architecture/python-bindings.md").read_text()

    for needle in [
        "put_temporal_relation",
        "temporal_provenance",
        "timestamp_column",
        "dataset_id",
        "row_hashes",
        "field_hashes",
        "uncertainty",
        "stream_id",
    ]:
        assert needle in init_py
        assert needle in init_stub
        assert needle in docs


def test_v088_neural_hot_loop_diagnostics_are_unified() -> None:
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    docs = (ROOT / "docs/architecture/python-bindings.md").read_text()
    program_rs = (ROOT / "crates/pyxlog/src/program.rs").read_text()

    for needle in [
        "def neural_hot_loop_diagnostics(",
        "post_load_dtoh_bytes",
        "post_load_htod_bytes",
        "control_plane_bytes_per_iteration",
        "scalar_sync_checks",
        "cuda_graph",
        "circuit_cache",
    ]:
        assert needle in native_stub
        assert needle in docs
    assert "pub fn neural_hot_loop_diagnostics(" in program_rs
    for needle in [
        "post_load_dtoh_bytes",
        "post_load_htod_bytes",
        "control_plane_bytes_per_iteration",
        "scalar_sync_checks",
        "cuda_graph",
        "circuit_cache",
    ]:
        assert needle in program_rs
