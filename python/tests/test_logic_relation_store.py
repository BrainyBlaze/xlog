import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


SOURCE = """
pred edge(i32, i32).
pred reach(i32, i32).
reach(X, Y) :- edge(X, Y).
?- reach(X, Y).
"""


def test_logic_session_persists_named_dlpack_relation():
    program = pyxlog.LogicProgram.compile(SOURCE, device=0, memory_mb=256)
    session = program.session()

    left = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    right = torch.tensor([2, 3, 4], device="cuda", dtype=torch.int32)

    session.put_relation("edge", [left, right])

    result = session.evaluate()
    assert len(result.queries) == 1

    query = result.queries[0]
    out_left = torch.from_dlpack(query.tensors[0]).cpu()
    out_right = torch.from_dlpack(query.tensors[1]).cpu()
    assert out_left.tolist() == [1, 2, 3]
    assert out_right.tolist() == [2, 3, 4]

    exported = session.export_relation("edge")
    assert len(exported) == 2

    exported_left = torch.from_dlpack(exported[0])
    exported_right = torch.from_dlpack(exported[1])
    assert exported_left.device.type == "cuda"
    assert exported_right.device.type == "cuda"
    assert exported_left.data_ptr() == left.data_ptr()
    assert exported_right.data_ptr() == right.data_ptr()


def test_logic_session_replaces_named_relation():
    program = pyxlog.LogicProgram.compile(SOURCE, device=0, memory_mb=256)
    session = program.session()

    session.put_relation(
        "edge",
        [
            torch.tensor([1, 2], device="cuda", dtype=torch.int32),
            torch.tensor([2, 3], device="cuda", dtype=torch.int32),
        ],
    )
    first = session.evaluate().queries[0]
    assert torch.from_dlpack(first.tensors[0]).cpu().tolist() == [1, 2]
    assert torch.from_dlpack(first.tensors[1]).cpu().tolist() == [2, 3]

    session.put_relation(
        "edge",
        [
            torch.tensor([7], device="cuda", dtype=torch.int32),
            torch.tensor([8], device="cuda", dtype=torch.int32),
        ],
    )
    second = session.evaluate().queries[0]
    assert torch.from_dlpack(second.tensors[0]).cpu().tolist() == [7]
    assert torch.from_dlpack(second.tensors[1]).cpu().tolist() == [8]


def test_logic_session_remove_and_clear_relations():
    program = pyxlog.LogicProgram.compile(SOURCE, device=0, memory_mb=256)
    session = program.session()

    session.put_relation(
        "edge",
        [
            torch.tensor([1], device="cuda", dtype=torch.int32),
            torch.tensor([2], device="cuda", dtype=torch.int32),
        ],
    )

    assert session.remove_relation("edge") is True
    with pytest.raises(ValueError, match="not found"):
        session.export_relation("edge")

    session.put_relation(
        "edge",
        [
            torch.tensor([3], device="cuda", dtype=torch.int32),
            torch.tensor([4], device="cuda", dtype=torch.int32),
        ],
    )
    session.clear_relations()

    query = session.evaluate().queries[0]
    assert query.num_rows == 0
    assert torch.from_dlpack(query.tensors[0]).numel() == 0
    assert torch.from_dlpack(query.tensors[1]).numel() == 0


def test_logic_session_put_relation_rejects_schema_mismatch():
    program = pyxlog.LogicProgram.compile(SOURCE, device=0, memory_mb=256)
    session = program.session()

    with pytest.raises(RuntimeError, match="dtype"):
        session.put_relation(
            "edge",
            [
                torch.tensor([1.0], device="cuda", dtype=torch.float32),
                torch.tensor([2], device="cuda", dtype=torch.int32),
            ],
        )

    with pytest.raises(RuntimeError, match="arity"):
        session.put_relation(
            "edge",
            [torch.tensor([1], device="cuda", dtype=torch.int32)],
        )
