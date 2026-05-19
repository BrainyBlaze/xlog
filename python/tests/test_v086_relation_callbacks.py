from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_v086_relation_callback_api_is_exposed_in_stubs_docs_and_rust() -> None:
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


def test_v086_relation_callback_payload_is_metadata_only_and_post_commit() -> None:
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
    assert ".apply_relation_deltas(" in apply_delta
    assert "self.fire_relation_callbacks(" in apply_delta
    assert apply_delta.index(".apply_relation_deltas(") < apply_delta.index(
        "self.fire_relation_callbacks("
    )


def test_v086_relation_callback_ordering_and_unregister_are_source_auditable() -> None:
    logic_rs = (ROOT / "crates/pyxlog/src/logic.rs").read_text()
    docs = (ROOT / "docs/architecture/python-bindings.md").read_text()

    assert "next_relation_callback_id" in logic_rs
    assert "retain(|registered| registered.id != callback_id)" in logic_rs
    assert "for registered in &self.relation_callbacks" in logic_rs
    assert "100 replays" in docs
    assert "GIL" in docs


def test_v086_relation_callbacks_have_certified_evidence() -> None:
    evidence = ROOT / "docs/evidence/2026-05-19-v086-notify/README.md"
    measurements = ROOT / "docs/evidence/2026-05-19-v086-notify/measurements.json"

    assert evidence.exists()
    assert measurements.exists()

    text = evidence.read_text()
    for needle in [
        "M086_NOTIFY.1",
        "M086_NOTIFY.2",
        "M086_NOTIFY.3",
        "M086_NOTIFY.4",
        "M086_NOTIFY.5",
        "M086_NOTIFY.6",
        "wmir_committed",
    ]:
        assert needle in text
