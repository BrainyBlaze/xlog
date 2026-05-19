import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


SOURCE = """
pred wmir_committed(i32).
pred out(i32).
out(X) :- wmir_committed(X).
?- out(X).
"""


def _session():
    program = pyxlog.LogicProgram.compile(SOURCE, device=0, memory_mb=256)
    session = program.session()
    if not hasattr(session, "register_relation_callback"):
        pytest.skip("installed pyxlog does not expose v0.8.6 relation callbacks")
    return session


def _col(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def test_relation_callbacks_fire_after_success_skip_failed_delta_and_unregister():
    session = _session()
    events = []
    callback_id = session.register_relation_callback(events.append)

    stats = session.apply_relation_delta("wmir_committed", insert_columns=[_col([1, 2])])

    assert stats["insert_rows"] == 2
    assert len(events) == 1
    payload = events[0]
    assert payload["relation"] == "wmir_committed"
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
    session.apply_relation_delta("wmir_committed", insert_columns=[_col([4])])
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

        session.apply_relation_delta("wmir_committed", insert_columns=[_col([1, 2])])
        session.apply_relation_delta("wmir_committed", delete_columns=[_col([2])])
        session.apply_relation_delta_batch(
            [
                {"name": "wmir_committed", "insert_columns": [_col([3, 4])]},
                {"name": "wmir_committed", "delete_columns": [_col([3])]},
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
            {"name": "wmir_committed", "insert_columns": [_col([1, 2])]},
            {"name": "wmir_committed", "delete_columns": [_col([2])]},
        ]
    )
    transfer_stats = session.host_transfer_stats()
    assert transfer_stats["dtoh_bytes"] == 0
    assert transfer_stats["dtoh_calls"] == 0
