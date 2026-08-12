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


# ---------------------------------------------------------------------------
# Task 3: soft_weights.partial_fit — the incremental refactor. The Adam
# state (weights, m, v, t) moves into SoftState; train_soft_weights becomes
# init + one partial_fit per step over the full batch, and MUST stay
# byte-identical to the pre-refactor trainer (the batch column's pin).
# ---------------------------------------------------------------------------


def _planted_cover_and_labels(torch, n_pt=40):
    """Same fixture as test_maritime_soft: body 0 fires exactly on the
    positives (t % 2 == 0); body 1 is noise (t % 5 == 0)."""
    cover = torch.zeros((2, n_pt), dtype=torch.bool)
    y = torch.zeros(n_pt, dtype=torch.bool)
    for t in range(n_pt):
        if t % 2 == 0:
            cover[0, t] = True
            y[t] = True
        if t % 5 == 0:
            cover[1, t] = True
    return cover, y


def test_train_soft_weights_bitwise_matches_pre_refactor_snapshot():
    # The regression pin for the batch column: these values were computed
    # by the PRE-REFACTOR train_soft_weights (commit 2c9eeade) on the
    # planted fixture; the refactored wrapper must reproduce them bitwise
    # (float32 round-trips through these double reprs exactly).
    torch = pytest.importorskip("torch")
    from soft_weights import train_soft_weights

    cover, y = _planted_cover_and_labels(torch)
    w50 = train_soft_weights(cover, y, steps=50)
    assert torch.equal(
        w50, torch.tensor([0.30746927857398987, -1.1695338487625122])
    )
    w300 = train_soft_weights(cover, y)
    assert torch.equal(
        w300, torch.tensor([3.59657883644104, -4.932765960693359])
    )


def test_train_soft_weights_equals_partial_fit_composition():
    torch = pytest.importorskip("torch")
    from soft_weights import init_soft_state, partial_fit, train_soft_weights

    cover, y = _planted_cover_and_labels(torch)
    state = init_soft_state(cover.shape[0])
    for _ in range(50):
        state = partial_fit(state, cover, y)
    assert state.t == 50
    assert torch.equal(state.weights, train_soft_weights(cover, y, steps=50))


def test_partial_fit_state_after_n_windows_is_bitwise_deterministic():
    # The plan's determinism pin: incremental training over windows is not
    # required to equal the full-batch run (the gradient steps differ by
    # definition) — but the state after N windows must be a pure function
    # of the inputs: two identical passes agree bitwise.
    torch = pytest.importorskip("torch")
    from soft_weights import init_soft_state, partial_fit

    cover, y = _planted_cover_and_labels(torch)

    def run_pass():
        state = init_soft_state(cover.shape[0])
        for lo in range(0, cover.shape[1], 7):
            state = partial_fit(state, cover[:, lo:lo + 7], y[lo:lo + 7])
        return state

    a, b = run_pass(), run_pass()
    assert a.t == b.t == 6
    assert torch.equal(a.weights, b.weights)
    assert torch.equal(a.m, b.m)
    assert torch.equal(a.v, b.v)


def test_partial_fit_does_not_mutate_its_input_state():
    torch = pytest.importorskip("torch")
    from soft_weights import INIT_LOGIT, init_soft_state, partial_fit

    cover, y = _planted_cover_and_labels(torch)
    state0 = init_soft_state(cover.shape[0])
    partial_fit(state0, cover, y)
    assert state0.t == 0
    assert torch.equal(state0.weights, torch.full((2,), INIT_LOGIT))
    assert state0.m.abs().sum().item() == 0.0
    assert state0.v.abs().sum().item() == 0.0


def test_partial_fit_rejects_bad_shapes_and_empty_batches():
    torch = pytest.importorskip("torch")
    from soft_weights import init_soft_state, partial_fit

    cover, y = _planted_cover_and_labels(torch)
    state = init_soft_state(cover.shape[0])
    with pytest.raises(ValueError):
        partial_fit(state, cover, y[:-1])              # N mismatch
    with pytest.raises(ValueError):
        partial_fit(state, cover[:1], y)               # C mismatch vs state
    with pytest.raises(ValueError):
        partial_fit(state, cover[:, :0], y[:0])        # empty batch
