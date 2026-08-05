import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


SOURCE = """
pred callback_input(i32).
pred out(i32).
out(X) :- callback_input(X).
?- out(X).
"""


def _session():
    program = pyxlog.LogicProgram.compile(SOURCE, device=0, memory_mb=256)
    session = program.session()
    if not hasattr(session, "register_relation_callback"):
        pytest.skip("installed pyxlog does not expose relation callbacks")
    return session


def _col(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def test_relation_callbacks_fire_after_success_skip_failed_delta_and_unregister():
    session = _session()
    events = []
    callback_id = session.register_relation_callback(events.append)

    stats = session.apply_relation_delta(
        "callback_input", insert_columns=[_col([1, 2])]
    )

    assert stats["insert_rows"] == 2
    assert len(events) == 1
    payload = events[0]
    assert payload["relation"] == "callback_input"
    assert payload["generation"] == 1
    assert payload["insert_rows"] == 2
    assert payload["delete_rows"] == 0
    assert payload["coalesced_insert_rows"] == 2
    assert payload["canceled_rows"] == 0
    assert "telemetry" in payload
    assert "tensors" not in payload
    assert "columns" not in payload

    with pytest.raises(ValueError, match="Unknown relation"):
        session.apply_relation_delta("missing_relation", insert_columns=[_col([3])])
    assert len(events) == 1

    assert session.unregister_relation_callback(callback_id) is True
    assert session.unregister_relation_callback(callback_id) is False
    session.apply_relation_delta("callback_input", insert_columns=[_col([4])])
    assert len(events) == 1


def test_relation_callback_ordering_is_deterministic_across_100_replays():
    expected = None
    for _ in range(100):
        session = _session()
        events = []
        session.register_relation_callback(
            lambda payload: events.append(
                (
                    payload["relation"],
                    payload["generation"],
                    payload["insert_rows"],
                    payload["delete_rows"],
                    payload["coalesced_insert_rows"],
                    payload["canceled_rows"],
                )
            )
        )

        session.apply_relation_delta("callback_input", insert_columns=[_col([1, 2])])
        session.apply_relation_delta("callback_input", delete_columns=[_col([2])])
        session.apply_relation_delta_batch(
            [
                {"name": "callback_input", "insert_columns": [_col([3, 4])]},
                {"name": "callback_input", "delete_columns": [_col([3])]},
            ]
        )

        if expected is None:
            expected = events
        assert events == expected


def test_callback_disabled_path_has_zero_callback_transfer_stats():
    session = _session()
    callback_id = session.register_relation_callback(lambda payload: None)
    assert session.unregister_relation_callback(callback_id) is True

    session.reset_host_transfer_stats()
    session.apply_relation_delta_batch(
        [
            {"name": "callback_input", "insert_columns": [_col([1, 2])]},
            {"name": "callback_input", "delete_columns": [_col([2])]},
        ]
    )
    transfer_stats = session.host_transfer_stats()
    assert transfer_stats["dtoh_bytes"] == 0
    assert transfer_stats["dtoh_calls"] == 0


def test_mixed_batch_notifies_only_effective_relations_in_first_occurrence_order():
    program = pyxlog.LogicProgram.compile(
        """
        pred first(i32).
        pred canceled(i32).
        pred second(i32).
        """,
        device=0,
        memory_mb=256,
    )
    session = program.session()
    events = []
    session.register_relation_callback(events.append)

    stats = session.apply_relation_delta_batch(
        [
            {"name": "second", "insert_columns": [_col([2])]},
            {"name": "canceled", "insert_columns": [_col([7])]},
            {"name": "first", "insert_columns": [_col([1])]},
            {"name": "canceled", "delete_columns": [_col([7])]},
            {"name": "second", "insert_columns": [_col([3])]},
        ]
    )

    assert stats["changed_relations"] == 2
    assert [event["relation"] for event in events] == ["second", "first"]
    assert [event["generation"] for event in events] == [1, 1]


def test_callback_exception_happens_after_consistent_commit_and_generation_publish():
    session = _session()

    def reject_event(_payload):
        raise LookupError("observer failed")

    callback_id = session.register_relation_callback(reject_event)
    with pytest.raises(LookupError, match="observer failed"):
        session.apply_relation_delta("callback_input", insert_columns=[_col([1])])

    assert session.delta_stats()["insert_rows"] == 1
    exported = session.export_relation("callback_input")
    assert torch.from_dlpack(exported[0]).cpu().tolist() == [1]

    assert session.unregister_relation_callback(callback_id) is True
    events = []
    session.register_relation_callback(events.append)
    session.apply_relation_delta("callback_input", insert_columns=[_col([2])])
    assert [event["generation"] for event in events] == [2]
