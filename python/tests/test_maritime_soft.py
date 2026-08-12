"""Unit tests for the pre-registered maritime soft-credit column
(`docs/experiments/maritime/PREREG_SOFT.md`): body enumeration + coverage
matrix, the noisy-OR weight trainer, the `sustained_240` relation, the
duration-vocabulary ceiling probe and the CV-runner columns. CPU only, no
real archives — synthetic fixtures throughout, following
`test_maritime_convert.py` / `test_maritime_cv.py`."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "maritime_woled"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "caviar_woled"))


# ---------------------------------------------------------------------------
# Task 2: enumerate_bodies + coverage_matrix (PREREG_SOFT.md section (b):
# the pool is every conjunction of 1..3 literals over the vocabulary)
# ---------------------------------------------------------------------------

BASELINE_VOCABULARY = [
    "proximity", "far", "both_lowspeed", "both_stopped_far",
    "both_low_or_stopped", "either_low_or_stopped", "any_near_ports",
    "both_open_sea", "became_far", "became_proximate", "any_slow_ended",
]


def test_enumerate_bodies_counts_11_choose_1_2_3():
    from enumerate_bodies import enumerate_bodies

    bodies = enumerate_bodies(BASELINE_VOCABULARY, max_literals=3)
    # C(11,1) + C(11,2) + C(11,3) = 11 + 55 + 165 = 231
    assert len(bodies) == 231
    assert sum(1 for b in bodies if len(b) == 1) == 11
    assert sum(1 for b in bodies if len(b) == 2) == 55
    assert sum(1 for b in bodies if len(b) == 3) == 165
    # no duplicates
    assert len(set(bodies)) == 231


def test_enumerate_bodies_combinations_not_permutations():
    from enumerate_bodies import enumerate_bodies

    bodies = enumerate_bodies(["b", "a"], max_literals=2)
    assert ("a", "b") in bodies
    assert ("b", "a") not in bodies
    # every body is a sorted tuple, no relation repeated inside a body
    for b in bodies:
        assert tuple(sorted(b)) == b
        assert len(set(b)) == len(b)


def test_coverage_matrix_rows_are_intersections():
    torch = pytest.importorskip("torch")
    from enumerate_bodies import coverage_matrix, enumerate_bodies

    # hand fixture: 3 relations x 6 pt
    relations = {"a": {0, 1, 2}, "b": {1, 2, 5}, "c": {2, 3}}
    bodies = enumerate_bodies(["a", "b", "c"], max_literals=3)
    assert bodies == [
        ("a",), ("b",), ("c",),
        ("a", "b"), ("a", "c"), ("b", "c"),
        ("a", "b", "c"),
    ]
    m = coverage_matrix(bodies, relations, n_pt=6)
    assert m.dtype == torch.bool
    assert m.shape == (7, 6)
    # hand-computed rows (True where pt is in EVERY relation of the body)
    assert m[0].tolist() == [True, True, True, False, False, False]     # a
    assert m[1].tolist() == [False, True, True, False, False, True]     # b
    assert m[2].tolist() == [False, False, True, True, False, False]    # c
    assert m[3].tolist() == [False, True, True, False, False, False]    # a&b
    assert m[4].tolist() == [False, False, True, False, False, False]   # a&c
    assert m[5].tolist() == [False, False, True, False, False, False]   # b&c
    assert m[6].tolist() == [False, False, True, False, False, False]   # a&b&c


def test_coverage_matrix_empty_intersection_is_a_zero_row():
    pytest.importorskip("torch")
    from enumerate_bodies import coverage_matrix

    # 'a' and 'b' never co-fire: the (a, b) row is all-False, legally
    m = coverage_matrix([("a", "b")], {"a": {0, 1}, "b": {2}}, n_pt=4)
    assert m.shape == (1, 4)
    assert m.sum().item() == 0
