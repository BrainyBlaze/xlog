"""The join extension must come FROM THE ENGINE. If this test passed while the map
were supplied from Python, it would prove nothing -- so it asserts against the
engine's own answer for facts the engine compiled.

API NOTE (correction found on the GPU box, 2026-07-13): the object the trainer's
mixture holds is `pyxlog.Program.compile(...)` -> `CompiledProgram`, whose only
read surface is `evaluate*` -- it does not expose facts at all (`EvalResult.atoms`
comes back empty for ground facts, and it has no `batch_fact_membership`).
`pyxlog.IlpProgramFactory.compile(...)` -> `CompiledIlpProgram` does expose facts,
via `relation_facts(name) -> list[list[int]]`, a direct enumeration of the
relation's extension. `read_join_extension` is built on that call.
"""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from pyxlog.ilp.join_bodies import JoinBody, read_join_extension

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="xlog engine requires CUDA"
)

_SOURCE = """
    pre_before_post(0, 0). pre_before_post(1, 0).
    pre_before_post(2, 1).
    pre_before_post(4, 2).
    pred pre_before_post(i64, i64).
"""

_JB = JoinBody(
    neural_predicate="saliency", network="sal_net", join_var="Ev",
    relation="pre_before_post", event_arg=0, head_arg=1,
)


def test_extension_is_read_from_the_engine() -> None:
    prog = pyxlog.IlpProgramFactory.compile(_SOURCE, device=0, memory_mb=64)
    prog.evaluate()
    ext = read_join_extension(prog, _JB, num_bindings=4)
    assert ext == [[0, 1], [2], [4], []]     # edge 3 joins nothing -> empty


def test_an_edge_with_no_joined_events_gets_an_empty_extension() -> None:
    prog = pyxlog.IlpProgramFactory.compile(_SOURCE, device=0, memory_mb=64)
    prog.evaluate()
    ext = read_join_extension(prog, _JB, num_bindings=4)
    assert ext[3] == []
