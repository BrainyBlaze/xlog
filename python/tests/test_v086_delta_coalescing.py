from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v086_delta_batch_api_is_exposed_in_stubs_docs_and_rust() -> None:
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    docs = (ROOT / "docs/architecture/python-bindings.md").read_text()
    logic_rs = (ROOT / "crates/pyxlog/src/logic.rs").read_text()
    gpu_logic_rs = (ROOT / "crates/xlog-gpu/src/logic.rs").read_text()

    assert "def apply_relation_delta_batch(" in native_stub
    assert "apply_relation_delta_batch" in docs
    assert "apply_relation_delta_batch" in logic_rs
    assert "coalesce_relation_delta_batch" in gpu_logic_rs


def test_v086_delta_batch_stats_are_source_auditable() -> None:
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    logic_rs = (ROOT / "crates/pyxlog/src/logic.rs").read_text()
    gpu_logic_rs = (ROOT / "crates/xlog-gpu/src/logic.rs").read_text()

    for needle in [
        "input_delta_count",
        "coalesced_insert_rows",
        "coalesced_delete_rows",
        "canceled_rows",
    ]:
        assert needle in native_stub
        assert needle in logic_rs
        assert needle in gpu_logic_rs


def test_v086_delta_batch_coalescing_uses_gpu_set_ops_without_host_row_materialization() -> None:
    gpu_logic_rs = (ROOT / "crates/xlog-gpu/src/logic.rs").read_text()
    start = gpu_logic_rs.index("fn coalesce_relation_delta_batch")
    end = gpu_logic_rs.index("fn ensure_schema_type_compatible", start)
    coalesce_impl = gpu_logic_rs[start:end]

    for needle in [
        "dedup_full_row",
        "diff_full_row",
        "union_gpu",
    ]:
        assert needle in coalesce_impl

    for forbidden in [
        "download_column",
        "dtoh_sync_copy",
        "dtoh_sync_copy_into_tracked",
        "Vec::<u32>",
        "to_vec()",
    ]:
        assert forbidden not in coalesce_impl


def test_v086_delta_batch_has_certified_evidence() -> None:
    evidence = ROOT / "docs/evidence/2026-05-19-v086-delta-coalesce/README.md"
    measurements = ROOT / "docs/evidence/2026-05-19-v086-delta-coalesce/measurements.json"

    assert evidence.exists()
    assert measurements.exists()

    text = evidence.read_text()
    for needle in [
        "M086_DELTA.1",
        "M086_DELTA.2",
        "M086_DELTA.3",
        "M086_DELTA.4",
        "M086_DELTA.5",
        "M086_DELTA.6",
        "wmir_committed",
    ]:
        assert needle in text
