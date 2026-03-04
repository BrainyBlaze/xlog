"""Tests for typed batch upload in batch_fact_membership / batch_tagged_credit."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()


def _compile_typed(source: str) -> "pyxlog.IlpProgramFactory":
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=64)
    prog.evaluate()
    return prog


def test_membership_u32_columns():
    """U32 columns (default) still work correctly."""
    prog = _compile_typed("""
        pred edge(u32, u32).
        edge(1, 2). edge(2, 3). edge(3, 4).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    mask = prog.batch_fact_membership("edge", [[1, 2], [99, 99], [3, 4]])
    assert mask == [True, False, True]


def test_membership_i32_columns():
    """I32 columns with negative values must round-trip correctly."""
    prog = _compile_typed("""
        pred data(i32, i32).
        data(-10, 20). data(30, -40). data(0, 0).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    mask = prog.batch_fact_membership("data", [[-10, 20], [30, -40], [1, 1]])
    assert mask == [True, True, False]


def test_membership_i64_columns():
    """I64 columns with large values must round-trip correctly."""
    big = 2**40  # exceeds u32 range
    prog = _compile_typed(f"""
        pred data(i64, i64).
        data({big}, 1). data(2, {big}).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    mask = prog.batch_fact_membership("data", [[big, 1], [2, big], [0, 0]])
    assert mask == [True, True, False]


def test_membership_bool_columns():
    """Bool columns accept 0/1 only."""
    prog = _compile_typed("""
        pred flag(u32, bool).
        flag(1, true). flag(2, false).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    mask = prog.batch_fact_membership("flag", [[1, 1], [2, 0], [1, 0]])
    assert mask == [True, True, False]


# --- Positive: batch_tagged_credit with typed columns ---


def test_tagged_credit_i32_columns():
    """batch_tagged_credit works with I32 typed columns (no crash, correct length)."""
    prog = _compile_typed("""
        pred data(i32, i32).
        data(-10, 20). data(30, -40). data(0, 0).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    credits = prog.batch_tagged_credit("data", [[-10, 20], [30, -40], [99, 99]])
    # Returns a list of per-fact credit lists; length must match input
    assert len(credits) == 3
    # Base facts are not derived by rules, so tagged credits are empty lists.
    # The key assertion is that i32 encoding did not raise an error.


# --- Negative tests: type validation errors ---


def test_membership_u32_overflow_error():
    """U32 column rejects values > u32::MAX."""
    prog = _compile_typed("""
        pred data(u32, u32).
        data(1, 2).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    with pytest.raises(ValueError, match=r"column 0 \(U32\).*out of range"):
        prog.batch_fact_membership("data", [[2**33, 1]])


def test_membership_u32_negative_error():
    """U32 column rejects negative values."""
    prog = _compile_typed("""
        pred data(u32, u32).
        data(1, 2).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    with pytest.raises(ValueError, match=r"column 0 \(U32\).*out of range"):
        prog.batch_fact_membership("data", [[-1, 1]])


def test_membership_u64_negative_error():
    """U64 column rejects negative values."""
    prog = _compile_typed("""
        pred data(u64, u64).
        data(1, 2).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    with pytest.raises(ValueError, match=r"column 0 \(U64\).*negative"):
        prog.batch_fact_membership("data", [[-5, 1]])


def test_membership_bool_invalid_error():
    """Bool column rejects values not in {0, 1}."""
    prog = _compile_typed("""
        pred data(u32, bool).
        data(1, true).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    with pytest.raises(ValueError, match=r"column 1 \(Bool\).*not in"):
        prog.batch_fact_membership("data", [[1, 2]])


def test_membership_f64_rejected():
    """F64 columns are explicitly rejected in batch APIs."""
    prog = _compile_typed("""
        pred data(f64, f64).
        data(1.5, 2.5).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """)
    with pytest.raises(ValueError, match=r"column 0 \(F64\).*not supported"):
        prog.batch_fact_membership("data", [[1, 2]])
