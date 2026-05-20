import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


SOURCE = """
pred left(i32).
pred right(i32).
pred out(i32).
out(X) :- left(X), right(X).
?- out(X).
"""


def _col(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def test_pyxlog_session_reuses_persistent_hash_index_after_relation_delta():
    program = pyxlog.LogicProgram.compile(SOURCE, device=0, memory_mb=512)
    session = program.session()
    assert hasattr(session, "join_index_cache_stats")

    values = torch.arange(2500, device="cuda", dtype=torch.int32)
    session.put_relation("left", [values])
    session.put_relation("right", [values])
    session.reset_host_transfer_stats()

    result = session.evaluate()
    assert result.queries[0].num_rows == 2500

    for value in range(10_000, 10_005):
        stats = session.apply_relation_delta("left", insert_columns=[_col([value])])
        assert stats["changed_relations"] == 1
        assert stats["affected_sccs"] >= 1

    cache_stats = session.join_index_cache_stats()
    transfer_stats = session.host_transfer_stats()

    assert cache_stats["builds"] >= 1
    assert cache_stats["hits"] >= 1
    assert cache_stats["entries"] >= 1
    assert cache_stats["stale_rejections"] == 0
    assert transfer_stats["dtoh_calls"] == 0
    assert transfer_stats["htod_calls"] == 0
