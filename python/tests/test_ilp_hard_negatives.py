# python/tests/test_ilp_hard_negatives.py
"""Tests for hard-negative mining via sample_false_positives."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4).
    learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""


def test_sample_false_positives_exists():
    """sample_false_positives is callable on CompiledIlpProgram."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W", False)
    n = prog.ilp_schema_size()

    # Set a mask that activates edge+edge -> reach
    M = torch.zeros(n * n * n, device="cuda")
    for c in candidates:
        if c["left_name"] == "edge" and c["right_name"] == "edge":
            flat = c["i"] * n * n + c["j"] * n + c["k"]
            M[flat] = 1.0
            break
    prog.set_rule_mask("W", M, M, n)
    prog.evaluate()

    exclude = [("reach", [1, 3]), ("reach", [2, 4])]
    fps = prog.sample_false_positives("reach", exclude, 10)
    assert isinstance(fps, list)
    for fp in fps:
        assert isinstance(fp, (list, tuple))


def test_sample_false_positives_excludes_positives():
    """Returned facts should not include excluded positives."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W", False)
    n = prog.ilp_schema_size()

    M = torch.zeros(n * n * n, device="cuda")
    for c in candidates:
        if c["left_name"] == "edge" and c["right_name"] == "edge":
            flat = c["i"] * n * n + c["j"] * n + c["k"]
            M[flat] = 1.0
            break
    prog.set_rule_mask("W", M, M, n)
    prog.evaluate()

    exclude = [("reach", [1, 3])]
    fps = prog.sample_false_positives("reach", exclude, 10)
    for fp in fps:
        assert tuple(fp) != (1, 3), "Should not return excluded positive"


def test_sample_false_positives_bounded():
    """Result is bounded by max_n."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W", False)
    n = prog.ilp_schema_size()

    M = torch.zeros(n * n * n, device="cuda")
    for c in candidates:
        if c["left_name"] == "edge" and c["right_name"] == "edge":
            flat = c["i"] * n * n + c["j"] * n + c["k"]
            M[flat] = 1.0
            break
    prog.set_rule_mask("W", M, M, n)
    prog.evaluate()

    fps = prog.sample_false_positives("reach", [], 2)
    assert len(fps) <= 2
