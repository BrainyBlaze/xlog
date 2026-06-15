import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def relation_callback_evidence_dir() -> Path:
    matches = []
    for path in sorted((ROOT / "docs" / "evidence").iterdir()):
        measurements_path = path / "measurements.json"
        if not measurements_path.exists() or not (path / "README.md").exists():
            continue
        measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
        if (
            measurements.get("runtime_tests_passed") == 3
            and measurements.get("determinism_replays") == 100
            and measurements.get("deterministic_callback_sequence") is True
            and measurements.get("failed_delta_callback_count_unchanged") is True
            and measurements.get("unregister_removed_callback") is True
            and measurements.get("payload_metadata_only") is True
            and measurements.get("callback_disabled_dtoh_bytes") == 0
            and measurements.get("callback_disabled_dtoh_calls") == 0
        ):
            matches.append(path)
    assert len(matches) == 1
    return matches[0]


EVIDENCE_DIR = relation_callback_evidence_dir()


def test_relation_callback_api_is_exposed_in_stubs_docs_and_rust() -> None:
    native_stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    docs = (ROOT / "docs/architecture/python-bindings.md").read_text()
    logic_rs = (ROOT / "crates/pyxlog/src/logic.rs").read_text()
    lib_rs = (ROOT / "crates/pyxlog/src/lib.rs").read_text()

    for needle in [
        "def register_relation_callback(",
        "def unregister_relation_callback(",
    ]:
        assert needle in native_stub
        assert needle in docs

    for needle in [
        "pub fn register_relation_callback(",
        "pub fn unregister_relation_callback(",
    ]:
        assert needle in logic_rs

    assert "relation_callbacks" in lib_rs
    assert "relation_generations" in lib_rs


def test_relation_callback_payload_is_metadata_only_and_post_commit() -> None:
    logic_rs = (ROOT / "crates/pyxlog/src/logic.rs").read_text()
    callback_start = logic_rs.index("fn fire_relation_callbacks")
    callback_end = logic_rs.index("fn optional_delta_columns", callback_start)
    callback_impl = logic_rs[callback_start:callback_end]

    for needle in [
        '"relation"',
        '"generation"',
        '"input_delta_count"',
        '"insert_rows"',
        '"delete_rows"',
        '"coalesced_insert_rows"',
        '"coalesced_delete_rows"',
        '"canceled_rows"',
        '"affected_sccs"',
        '"recomputed_sccs"',
        '"incremental_sccs"',
        '"telemetry"',
    ]:
        assert needle in callback_impl

    for forbidden in [
        "export_relation",
        "download_column",
        "to_dlpack_table",
        "dlpack_capsule_from_tensor",
        "tensors",
        "columns",
    ]:
        assert forbidden not in callback_impl

    apply_delta = logic_rs[
        logic_rs.index("fn apply_single_relation_delta") : logic_rs.index(
            "fn fire_relation_callbacks"
        )
    ]
    assert ".apply_relation_deltas_with_session_runtime(" in apply_delta
    assert "self.fire_relation_callbacks(" in apply_delta
    assert apply_delta.index(".apply_relation_deltas_with_session_runtime(") < apply_delta.index(
        "self.fire_relation_callbacks("
    )


def test_relation_callback_ordering_and_unregister_are_source_auditable() -> None:
    logic_rs = (ROOT / "crates/pyxlog/src/logic.rs").read_text()
    docs = (ROOT / "docs/architecture/python-bindings.md").read_text()

    assert "next_relation_callback_id" in logic_rs
    assert "retain(|registered| registered.id != callback_id)" in logic_rs
    assert "for registered in &self.relation_callbacks" in logic_rs
    assert "100 replays" in docs
    assert "GIL" in docs


def test_relation_callbacks_have_certified_evidence() -> None:
    evidence = EVIDENCE_DIR / "README.md"
    measurements = EVIDENCE_DIR / "measurements.json"

    assert evidence.exists()
    assert measurements.exists()

    text = evidence.read_text()
    for needle in [
        "Opt-In Relation-Change Callbacks",
        "callbacks fire only after successful delta commits",
        "payloads contain metadata summaries",
        "nested",
        "deterministic across 100 runtime replays",
        "zero relation data-plane D2H transfers",
        "callback-disabled overhead within 2 percent",
    ]:
        assert needle in text
