"""G086_EXACT_TYPES runtime parity for native exact induction."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()

from pyxlog.ilp import induce_exact  # noqa: E402


def _target_source(scalar_type: str) -> str:
    return f"""
        pred p_A({scalar_type}, {scalar_type}).
        pred p_B({scalar_type}, {scalar_type}).
        pred p_C({scalar_type}, {scalar_type}).
        pred p_D({scalar_type}, {scalar_type}).
        pred p_E({scalar_type}, {scalar_type}).
        learnable(W_chain_p_A)  :: p_A(X, Y) :- bL(X, Z), bR(Z, Y).
        learnable(W_star_p_A)   :: p_A(X, Y) :- bL(X, Y), bR(X, Y).
        learnable(W_fanout_p_A) :: p_A(X, Y) :- bL(X, Z), bR(X, Y).
        learnable(W_fanin_p_A)  :: p_A(X, Y) :- bL(X, Y), bR(Z, Y).
    """


def _tensor(values: list[int], dtype: "torch.dtype") -> "torch.Tensor":
    return torch.tensor(values, dtype=dtype, device=torch.device("cuda"))


def _build_typed_request(scalar_type: str, dtype: "torch.dtype", n_candidates: int = 3):
    prog = pyxlog.IlpProgramFactory.compile(
        _target_source(scalar_type),
        device=0,
        memory_mb=64,
    )

    prog.put_relation("p_B", [_tensor([1, 2], dtype), _tensor([2, 3], dtype)])
    prog.put_relation("p_C", [_tensor([2, 3, 4], dtype), _tensor([4, 5, 6], dtype)])
    prog.put_relation("p_D", [_tensor([1, 2], dtype), _tensor([4, 5], dtype)])
    prog.put_relation("p_E", [_tensor([7], dtype), _tensor([8], dtype)])

    kwargs = dict(
        head_relation="p_A",
        candidate_relations=[f"p_{chr(ord('B') + i)}" for i in range(n_candidates)],
        positive_arg0=_tensor([1, 2], dtype),
        positive_arg1=_tensor([4, 5], dtype),
        negative_arg0=_tensor([7], dtype),
        negative_arg1=_tensor([8], dtype),
        k_per_topology=2,
        deterministic=True,
    )
    return prog, kwargs


def _assert_same_candidates(native_result, py_result) -> None:
    assert native_result.total_scored == py_result.total_scored
    assert native_result.candidate_count == py_result.candidate_count
    assert native_result.positive_count == py_result.positive_count
    assert native_result.negative_count == py_result.negative_count
    assert len(native_result.candidates) == len(py_result.candidates)
    for native, reference in zip(native_result.candidates, py_result.candidates):
        assert native.topology == reference.topology
        assert native.head_relation == reference.head_relation
        assert native.left_relation == reference.left_relation
        assert native.right_relation == reference.right_relation
        assert native.positives_covered == reference.positives_covered
        assert native.negatives_covered == reference.negatives_covered
        assert native.local_rank == reference.local_rank


@pytest.mark.parametrize(
    ("scalar_type", "dtype"),
    [
        ("u32", torch.uint32),
        ("symbol", torch.uint32),
    ],
)
def test_induce_exact_native_matches_python_reference_for_32_bit_pair_types(
    monkeypatch,
    scalar_type: str,
    dtype: "torch.dtype",
) -> None:
    prog, kwargs = _build_typed_request(scalar_type, dtype)
    annotations = dict(prog.relation_type_annotations())
    assert annotations["p_A"] == [scalar_type, scalar_type]
    assert annotations["p_B"] == [scalar_type, scalar_type]
    monkeypatch.setenv("XLOG_ALLOW_PYTHON_ILP_REFERENCE", "1")

    py_result = induce_exact(
        prog,
        backend="python",
        strict_per_topology=True,
        **kwargs,
    )
    prog.reset_d2h_transfer_count()
    native_result = induce_exact(prog, backend="native", **kwargs)

    _assert_same_candidates(native_result, py_result)
    assert prog.d2h_transfer_count() == 1


def test_induce_exact_native_rejects_mixed_logical_pair_types() -> None:
    source = """
        pred p_A(u32, u32).
        pred p_B(symbol, symbol).
        learnable(W_chain_p_A) :: p_A(X, Y) :- bL(X, Z), bR(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=64)
    prog.put_relation("p_B", [_tensor([1], torch.uint32), _tensor([2], torch.uint32)])

    with pytest.raises(Exception, match="type mismatch.*U32.*Symbol"):
        induce_exact(
            prog,
            backend="native",
            head_relation="p_A",
            candidate_relations=["p_B"],
            positive_arg0=_tensor([1], torch.uint32),
            positive_arg1=_tensor([2], torch.uint32),
            negative_arg0=_tensor([7], torch.uint32),
            negative_arg1=_tensor([8], torch.uint32),
            k_per_topology=1,
            deterministic=True,
        )
