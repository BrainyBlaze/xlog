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


# ---------------------------------------------------------------------------
# Task 3: soft_weights — the noisy-OR BCE trainer (PREREG_SOFT.md section
# (b): score(pt) = 1 - PROD_c (1 - sigmoid(w_c) * cover_c(pt)), BCE, Adam,
# steps=300, lr=0.05, seed=7, deterministic CPU, init w = -2.0)
# ---------------------------------------------------------------------------


def test_soft_scores_hand_computed_noisy_or():
    torch = pytest.importorskip("torch")
    import math

    from soft_weights import soft_scores

    cover = torch.tensor([[True, True, False], [False, True, False]])
    # weights are LOGITS: pick them so sigmoid(w) = [0.6, 0.4] exactly
    weights = torch.tensor([math.log(0.6 / 0.4), math.log(0.4 / 0.6)])
    scores = soft_scores(cover, weights)
    # pt0: 1 - (1 - 0.6)            = 0.6
    # pt1: 1 - (1 - 0.6)(1 - 0.4)   = 1 - 0.4*0.6 = 0.76
    # pt2: nothing covers it        = 0.0
    assert scores.tolist() == pytest.approx([0.6, 0.76, 0.0])


def _planted_cover_and_labels(torch, n_pt=40):
    """Body 0 fires exactly on the positives (t % 2 == 0); body 1 is noise
    (t % 5 == 0: covers some positives AND some negatives)."""
    cover = torch.zeros((2, n_pt), dtype=torch.bool)
    y = torch.zeros(n_pt, dtype=torch.bool)
    for t in range(n_pt):
        if t % 2 == 0:
            cover[0, t] = True
            y[t] = True
        if t % 5 == 0:
            cover[1, t] = True
    return cover, y


def test_train_soft_weights_recovers_planted_body_and_mutes_noise():
    torch = pytest.importorskip("torch")
    from soft_weights import soft_scores, train_soft_weights

    cover, y = _planted_cover_and_labels(torch)
    weights = train_soft_weights(cover, y)
    sig = torch.sigmoid(weights)
    assert sig[0].item() > 0.9, "the planted body must be turned on"
    assert sig[1].item() < 0.1, "the noise body must be turned off"
    # the trained scores separate the classes at the 0.5 threshold
    scores = soft_scores(cover, weights)
    assert ((scores > 0.5) == y).all()


def test_train_soft_weights_bce_falls_with_training():
    # The credit_nll parity pin, measured on the shared semantics: the BCE
    # of the noisy-OR scores against the labels falls as training proceeds
    # (checkpoints at steps 0 < 10 < 300; deterministic restarts make the
    # 10-step run a prefix of the 300-step run).
    torch = pytest.importorskip("torch")
    from soft_weights import soft_scores, train_soft_weights

    cover, y = _planted_cover_and_labels(torch)

    def bce(weights):
        scores = soft_scores(cover, weights).clamp(1e-7, 1 - 1e-7)
        return torch.nn.functional.binary_cross_entropy(scores, y.float()).item()

    bce_init = bce(torch.full((2,), -2.0))
    bce_10 = bce(train_soft_weights(cover, y, steps=10))
    bce_300 = bce(train_soft_weights(cover, y, steps=300))
    assert bce_300 < bce_10 < bce_init


def test_train_soft_weights_is_bitwise_deterministic():
    torch = pytest.importorskip("torch")
    from soft_weights import train_soft_weights

    cover, y = _planted_cover_and_labels(torch)
    w1 = train_soft_weights(cover, y, steps=50, seed=7)
    w2 = train_soft_weights(cover, y, steps=50, seed=7)
    assert torch.equal(w1, w2), "same seed must reproduce bitwise-equal weights"
