import inspect
import os

import pytest
import pyxlog
import pyxlog._native as native
import torch


def test_package_reexports_native_relation_metadata_objects_without_sidecar_hooks():
    assert pyxlog.RelationEvidence is native.RelationEvidence
    assert pyxlog.RelationMetadataError is native.RelationMetadataError
    assert "RelationEvidence" in pyxlog.__all__
    assert "RelationMetadataError" in pyxlog.__all__

    for method_name in (
        "put_relation_with_provenance",
        "put_relation_from_manifest",
        "relation",
        "evidence",
        "export_relation_with_provenance",
    ):
        descriptor = vars(native.LogicRelationSession)[method_name]
        assert inspect.ismethoddescriptor(descriptor)
        assert descriptor.__objclass__ is native.LogicRelationSession

    package_state = vars(pyxlog)
    assert "_RELATION_EVIDENCE" not in package_state
    assert "_record_relation_evidence" not in package_state
    assert "_hash_path" not in package_state
    assert "_stable_hash" not in package_state


def test_temporal_metadata_remains_separate_from_native_relation_evidence():
    if not torch.cuda.is_available():
        if os.environ.get("XLOG_REQUIRE_CUDA") == "1":
            raise RuntimeError("XLOG_REQUIRE_CUDA=1 but PyTorch cannot access CUDA")
        pytest.skip("CUDA is unavailable")

    program = pyxlog.LogicProgram.compile(
        "pred event(entity: u32, event_time: i64).",
        device=0,
        memory_mb=256,
    )
    session = program.session()
    columns = [
        torch.tensor([7], device="cuda", dtype=torch.int32),
        torch.tensor([1_700_000_000], device="cuda", dtype=torch.int64),
    ]

    temporal = session.put_temporal_relation(
        "event",
        columns,
        timestamp_column="event_time",
        dataset_id="live-events",
        row_hashes=["sha256:row"],
        source="device-stream",
    )
    assert session.temporal_provenance("event") == temporal

    plain_snapshot = session.relation("event").provenance()
    assert plain_snapshot == {
        "relation": "event",
        "metadata_present": False,
        "row_count": 1,
        "roles": [],
        "facts": [],
    }

    native_snapshot = session.put_relation_with_provenance(
        "event",
        columns,
        roles=[{"name": "entity"}, {"name": "event_time"}],
        facts=[
            {
                "tuple": [7, 1_700_000_000],
                "provenance": [{"source": "native-evidence"}],
            }
        ],
    )
    assert native_snapshot["metadata_present"] is True
    assert native_snapshot["facts"][0]["provenance"][0]["source"] == "native-evidence"
    assert session.temporal_provenance("event") == temporal
