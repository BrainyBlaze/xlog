import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def delta_coalescing_evidence_dir() -> Path:
    matches = []
    for path in sorted((ROOT / "docs" / "evidence").iterdir()):
        measurements_path = path / "measurements.json"
        if not measurements_path.exists() or not (path / "README.md").exists():
            continue
        measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
        if (
            measurements.get("recompute_call_reduction_ratio") == 3.0
            and measurements.get("hot_path_dtoh_calls") == 0
            and measurements.get("final_output_transfer_excluded") is True
        ):
            matches.append(path)
    assert len(matches) == 1
    return matches[0]


def test_delta_batch_api_is_exposed_in_stubs_docs_and_rust() -> None:
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    docs = (ROOT / "docs/architecture/python-bindings.md").read_text()
    logic_rs = (ROOT / "crates/pyxlog/src/logic.rs").read_text()
    gpu_logic_rs = (ROOT / "crates/xlog-gpu/src/logic.rs").read_text()

    assert "def apply_relation_delta_batch(" in native_stub
    assert "apply_relation_delta_batch" in docs
    assert "apply_relation_delta_batch" in logic_rs
    assert "coalesce_relation_delta_batch" in gpu_logic_rs


def test_delta_batch_stats_are_source_auditable() -> None:
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


def test_delta_batch_coalescing_uses_gpu_set_ops_without_host_row_materialization() -> None:
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


def test_delta_batch_has_certified_evidence() -> None:
    evidence_dir = delta_coalescing_evidence_dir()
    evidence = evidence_dir / "README.md"
    measurements = evidence_dir / "measurements.json"

    assert evidence.exists()
    assert measurements.exists()

    text = evidence.read_text()
    for needle in [
        "Device-Resident Batch Relation Delta Coalescing",
        "insert/delete cancellation",
        "recompute calls reduce",
        "delta stats report",
    ]:
        assert needle in text
