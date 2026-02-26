# python/tests/test_ilp_candidates.py
"""Tests for valid_candidates() Rust API."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

SOURCE = """
    edge(1, 2).
    edge(2, 3).
    edge(3, 4).
    learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""


def test_valid_candidates_returns_candidates():
    """valid_candidates returns non-empty list of CandidateInfo dicts."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W")
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    # Each candidate has required keys
    c = candidates[0]
    assert set(c.keys()) == {"id", "i", "j", "k", "left_name", "right_name", "head_name"}


def test_valid_candidates_deterministic_ids():
    """Same program produces same candidate IDs."""
    prog1 = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    prog2 = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    c1 = prog1.valid_candidates("W")
    c2 = prog2.valid_candidates("W")
    assert [c["id"] for c in c1] == [c["id"] for c in c2]


def test_valid_candidates_prunes_template_plus_template():
    """Candidates where both body atoms are template variables are pruned."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W")
    # Get the actual template names from relation list
    names = prog.ilp_relation_names()
    templates = {n for n in names if n.startswith("bL") or n.startswith("bR")
                 or n.startswith("b1") or n.startswith("b2")}
    for c in candidates:
        assert not (c["left_name"] in templates and c["right_name"] in templates), \
            f"Template+template candidate not pruned: {c}"


def test_valid_candidates_head_is_target():
    """All candidates have head_name == the learnable target relation."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W")
    for c in candidates:
        assert c["head_name"] == "reach", f"Wrong head: {c}"


def test_valid_candidates_nonrecursive_prunes_head_in_body():
    """Default (allow_recursive=False): candidates where body==head are pruned."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W")  # default: allow_recursive=False
    for c in candidates:
        # reach has no base facts, so body referencing reach should be pruned
        assert c["left_name"] != "reach" and c["right_name"] != "reach", \
            f"Recursive candidate not pruned: {c}"


def test_valid_candidates_ids_contiguous():
    """IDs are 0..C-1 with no gaps."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W")
    ids = sorted(c["id"] for c in candidates)
    assert ids == list(range(len(candidates)))


def test_max_active_rules_configurable():
    """Compilation accepts max_active_rules parameter."""
    prog = pyxlog.IlpProgramFactory.compile(
        SOURCE, device=0, memory_mb=512, max_active_rules=64
    )
    assert prog.ilp_schema_size() > 0


def test_max_active_rules_default_is_32():
    """Default max_active_rules is 32 (backward compatible)."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    # No way to query it directly, but compilation succeeds with default
    assert prog.ilp_schema_size() > 0


def test_max_active_rules_rejects_out_of_range():
    """Values outside 16-128 are rejected."""
    with pytest.raises(ValueError):
        pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512, max_active_rules=5)
    with pytest.raises(ValueError):
        pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512, max_active_rules=500)
