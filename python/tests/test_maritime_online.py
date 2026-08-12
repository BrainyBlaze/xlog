"""Unit tests for the pre-registered maritime single-pass online weights
column (`docs/experiments/maritime/PREREG_ONLINE.md`): the chronological
window stream, the incremental `partial_fit` refactor of `soft_weights`
and the CV-runner online column. CPU only, no real archives — synthetic
fixtures throughout, following `test_maritime_soft.py`."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "maritime_woled"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "caviar_woled"))


# ---------------------------------------------------------------------------
# Task 2: online_stream.stream_windows — the chronological window iterator
# (PREREG_ONLINE.md section 2: one pass in ascending pt_time order, global
# time, mini-batch windows; tie on time broken by ascending row index;
# reverse = the chrono sequence reversed exactly)
# ---------------------------------------------------------------------------


def _shuffled_fixture():
    """Rows deliberately NOT in time order (pair-major, as the converter
    lays them out), with a time tie between rows 4 and 1."""
    rows_idx = [10, 11, 12, 20, 21, 22, 30]
    pt_times = [500, 100, 900, 100, 700, 300, 500]
    return rows_idx, pt_times


def test_stream_windows_chrono_covers_every_index_exactly_once():
    torch = pytest.importorskip("torch")
    from online_stream import stream_windows

    rows_idx, pt_times = _shuffled_fixture()
    windows = list(stream_windows(rows_idx, pt_times, window=3))
    assert all(w.dtype == torch.long for w in windows)
    flat = [i for w in windows for i in w.tolist()]
    assert sorted(flat) == sorted(rows_idx)
    assert len(flat) == len(rows_idx)


def test_stream_windows_chrono_order_is_globally_nondecreasing_with_stable_ties():
    pytest.importorskip("torch")
    from online_stream import stream_windows

    rows_idx, pt_times = _shuffled_fixture()
    time_of = dict(zip(rows_idx, pt_times))
    flat = [i for w in stream_windows(rows_idx, pt_times, window=3) for i in w.tolist()]
    times = [time_of[i] for i in flat]
    assert times == sorted(times), "stream must be globally non-decreasing in time"
    # the tie at t=100 (rows 11 and 20) and at t=500 (rows 10 and 30) breaks
    # by ascending row index — the full expected order is pinned by hand:
    assert flat == [11, 20, 22, 10, 30, 21, 12]


def test_stream_windows_window_sizes_are_window_then_remainder():
    pytest.importorskip("torch")
    from online_stream import stream_windows

    rows_idx, pt_times = _shuffled_fixture()
    windows = list(stream_windows(rows_idx, pt_times, window=3))
    assert [len(w) for w in windows] == [3, 3, 1]
    # window >= N: a single window carrying everything
    windows = list(stream_windows(rows_idx, pt_times, window=100))
    assert [len(w) for w in windows] == [7]


def test_stream_windows_reverse_is_the_exact_reversal_of_chrono():
    pytest.importorskip("torch")
    from online_stream import stream_windows

    rows_idx, pt_times = _shuffled_fixture()
    chrono = [i for w in stream_windows(rows_idx, pt_times, window=3) for i in w.tolist()]
    rev = [i for w in stream_windows(rows_idx, pt_times, window=3, order="reverse") for i in w.tolist()]
    assert rev == chrono[::-1]
    time_of = dict(zip(rows_idx, pt_times))
    times = [time_of[i] for i in rev]
    assert times == sorted(times, reverse=True), "reverse stream must be non-increasing in time"


def test_stream_windows_rejects_bad_arguments():
    pytest.importorskip("torch")
    from online_stream import stream_windows

    with pytest.raises(ValueError):
        list(stream_windows([1, 2], [10], window=2))          # length mismatch
    with pytest.raises(ValueError):
        list(stream_windows([1, 2], [10, 20], window=0))      # window < 1
    with pytest.raises(ValueError):
        list(stream_windows([1, 2], [10, 20], window=2, order="shuffled"))


def test_stream_windows_empty_input_yields_no_windows():
    pytest.importorskip("torch")
    from online_stream import stream_windows

    assert list(stream_windows([], [], window=5)) == []
