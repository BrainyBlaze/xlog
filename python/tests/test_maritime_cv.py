"""Unit tests for `examples/maritime_woled/run_maritime_cv.py` -- CPU, no
real archives (the real-data run is a separate, manual job; see
`docs/experiments/maritime/README.md`'s pre-registration). Follows the
style of `test_caviar_cv.py` (CLI parse tests, deterministic fold
assignment) and `test_maritime_convert.py` (synthetic tar/zip mini-archive
fixtures)."""

import io
import json
import os
import sys
import tarfile
import zipfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "maritime_woled"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "caviar_woled"))

import run_maritime_cv  # noqa: E402


# ---------------------------------------------------------------------------
# Fold assignment: pairs are atoms, LPT stratification, deterministic
# ---------------------------------------------------------------------------


def test_stratified_pair_folds_returns_one_fold_per_pair():
    # 6 positive pairs, 6 negative pairs, 3 folds.
    pos_counts = [100, 40, 30, 20, 10, 5]
    pt_counts = [200, 90, 70, 50, 30, 15, 400, 300, 200, 100, 50, 25]
    folds = run_maritime_cv.stratified_pair_folds(pos_counts, pt_counts, 6, 3)
    assert len(folds) == 12
    assert all(f in (0, 1, 2) for f in folds)


def test_stratified_pair_folds_every_fold_has_positive_and_negative_pairs():
    pos_counts = [100, 40, 30, 20, 10, 5]
    pt_counts = [200, 90, 70, 50, 30, 15, 400, 300, 200, 100, 50, 25]
    folds = run_maritime_cv.stratified_pair_folds(pos_counts, pt_counts, 6, 3)
    for fold in range(3):
        members = [i for i, f in enumerate(folds) if f == fold]
        assert any(i < 6 for i in members), f"fold {fold} has no positive pair"
        assert any(i >= 6 for i in members), f"fold {fold} has no negative pair"


def test_stratified_pair_folds_positive_balance_bounded_by_largest_pair():
    # Extreme concentration mirroring the real corpus: top pair carries ~33%.
    pos_counts = [1194, 370, 320, 271, 150, 120, 100, 90, 80, 70, 60, 50]
    pt_counts = list(pos_counts) + [500, 400, 300, 200, 100, 50]
    n_pos = len(pos_counts)
    folds = run_maritime_cv.stratified_pair_folds(pos_counts, pt_counts, n_pos, 5)
    sums = [0] * 5
    for i, f in enumerate(folds[:n_pos]):
        sums[f] += pos_counts[i]
    assert max(sums) - min(sums) <= max(pos_counts)


def test_stratified_pair_folds_negative_balance_bounded_by_largest_pair():
    pos_counts = [10, 9, 8, 7, 6, 5]
    neg_counts = [1652, 900, 400, 300, 200, 100, 50, 25, 10]
    pt_counts = [20, 18, 16, 14, 12, 10] + neg_counts
    folds = run_maritime_cv.stratified_pair_folds(pos_counts, pt_counts, 6, 3)
    sums = [0] * 3
    for j, f in enumerate(folds[6:]):
        sums[f] += neg_counts[j]
    assert max(sums) - min(sums) <= max(neg_counts)


def test_stratified_pair_folds_deterministic_and_exact_lpt():
    pos_counts = [100, 40, 30, 20, 10, 5]
    pt_counts = [0, 0, 0, 0, 0, 0, 400, 300, 200, 100, 50, 25]
    a = run_maritime_cv.stratified_pair_folds(pos_counts, pt_counts, 6, 3)
    b = run_maritime_cv.stratified_pair_folds(pos_counts, pt_counts, 6, 3)
    assert a == b
    # Hand-traced LPT: positives sorted desc 100,40,30,20,10,5 ->
    # folds 0,1,2, then loads (100,40,30): 20->fold2 (50), 10->fold1 (50),
    # 5->fold1? loads now (100,50,50) -> tie between folds 1,2 -> lowest
    # index wins: 5->fold1 (55).
    assert a[:6] == [0, 1, 2, 2, 1, 1]
    # Negatives sorted desc 400,300,200,100,50,25 -> folds 0,1,2, loads
    # (400,300,200): 100->fold2 (300), 50->fold1? loads (400,300,300) ->
    # tie 1,2 -> fold1 (350), 25->fold2 (325).
    assert a[6:] == [0, 1, 2, 2, 1, 2]


def test_stratified_pair_folds_rejects_too_few_pairs_or_folds():
    with pytest.raises(ValueError):
        run_maritime_cv.stratified_pair_folds([5, 5], [5, 5, 9, 9], 2, 1)
    with pytest.raises(ValueError):
        # only 2 positive pairs for 3 folds
        run_maritime_cv.stratified_pair_folds([5, 5], [5, 5, 9, 9, 9], 2, 3)
    with pytest.raises(ValueError):
        # only 2 negative pairs for 3 folds
        run_maritime_cv.stratified_pair_folds([5, 5, 5], [5, 5, 5, 9, 9], 3, 3)


def test_pair_counts_from_converted_corpus():
    converted = {
        "pt_pair_index": [0, 0, 0, 1, 1, 2, 2, 2, 2],
        "is_positive": [True, False, True, False, False, False, True, False, False],
    }
    pos_counts, pt_counts = run_maritime_cv.pair_counts(converted, n_pairs=3)
    assert pt_counts == [3, 2, 4]
    assert pos_counts == [2, 0, 1]
