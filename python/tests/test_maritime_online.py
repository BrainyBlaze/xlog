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


def _pre_refactor_soft_scores(torch, cover, weights):
    """The reference `soft_scores` — verbatim from the pre-refactor module
    (`git show a79c16db:examples/maritime_woled/soft_weights.py`)."""
    active = torch.sigmoid(weights).unsqueeze(1) * cover.to(torch.get_default_dtype())
    return 1.0 - torch.prod(1.0 - active, dim=0)


def _pre_refactor_train_soft_weights(torch, cover, y, *, steps=300, lr=0.05, seed=7):
    """THE REFERENCE: the pre-refactor batch trainer, copied verbatim from
    `git show a79c16db:examples/maritime_woled/soft_weights.py`
    (constants inlined: INIT_LOGIT -2.0, clamp eps 1e-7, Adam defaults).
    Kept here on purpose so the refactored trainer is compared against
    the ORIGINAL ARITHMETIC ON THE SAME MACHINE — a platform-independent
    statement (a literal bit-pin taken on one machine is not: the CI
    Linux runner rounds the last float32 ulp differently, PR #270 review)."""
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)

    target = y.to(torch.get_default_dtype())
    weights = torch.full((cover.shape[0],), -2.0, requires_grad=True)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = torch.zeros_like(weights)
    v = torch.zeros_like(weights)
    for step in range(1, steps + 1):
        if weights.grad is not None:
            weights.grad = None
        scores = _pre_refactor_soft_scores(torch, cover, weights).clamp(1e-7, 1.0 - 1e-7)
        loss = torch.nn.functional.binary_cross_entropy(scores, target)
        loss.backward()
        g = weights.grad
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        with torch.no_grad():
            weights -= lr * m_hat / (v_hat.sqrt() + eps)
    return weights.detach()


def test_train_soft_weights_bitwise_matches_pre_refactor_reference():
    # The regression pin for the batch column (PREREG_SOFT section (b)):
    # the refactored trainer (init_soft_state + steps x partial_fit) must
    # reproduce the pre-refactor trainer BITWISE. Both run here, on the
    # same machine, so the comparison holds on every platform.
    torch = pytest.importorskip("torch")
    from soft_weights import train_soft_weights

    cover, y = _planted_cover_and_labels(torch)
    for steps in (50, 300):
        got = train_soft_weights(cover, y, steps=steps)
        want = _pre_refactor_train_soft_weights(torch, cover, y, steps=steps)
        assert torch.equal(got, want), (steps, got.tolist(), want.tolist())
    # and the default step count is the pre-registered 300
    assert torch.equal(
        train_soft_weights(cover, y),
        _pre_refactor_train_soft_weights(torch, cover, y, steps=300),
    )


def test_train_soft_weights_close_to_author_machine_values_smoke():
    # NOT a bitwise pin (that is the reference test above): these are the
    # values the pre-refactor trainer produced on the author's machine
    # (Windows x86-64, torch 2.9.1+cpu); other platforms round the last
    # float32 ulp differently (the CI Linux runner did, PR #270), so this
    # is only a gross-drift smoke against a wrong lr/steps/init/loss.
    torch = pytest.importorskip("torch")
    from soft_weights import train_soft_weights

    cover, y = _planted_cover_and_labels(torch)
    assert torch.allclose(
        train_soft_weights(cover, y, steps=50),
        torch.tensor([0.30746927857398987, -1.1695338487625122]),
        rtol=0.0, atol=1e-4,
    )
    assert torch.allclose(
        train_soft_weights(cover, y),
        torch.tensor([3.59657883644104, -4.932765960693359]),
        rtol=0.0, atol=1e-4,
    )


def test_partial_fit_composition_matches_pre_refactor_reference():
    # partial_fit on its own (not through the train_soft_weights wrapper):
    # init + N x partial_fit over the full batch is the pre-refactor
    # trainer's N-th iterate, bitwise, and the step counter advances.
    torch = pytest.importorskip("torch")
    from soft_weights import init_soft_state, partial_fit

    cover, y = _planted_cover_and_labels(torch)
    state = init_soft_state(cover.shape[0])
    for _ in range(50):
        state = partial_fit(state, cover, y)
    assert state.t == 50
    assert torch.equal(
        state.weights, _pre_refactor_train_soft_weights(torch, cover, y, steps=50)
    )


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


# ---------------------------------------------------------------------------
# Task 4: run_maritime_cv --column online — the O-online column
# (PREREG_ONLINE.md sections 1-3): the SAME pool + permutation-null gate as
# the soft column, then ONE chronological pass of partial_fit over the
# train fold's stream windows, the prequential curve (window error BEFORE
# the update) as a diagnostic, and the soft column's held-out scoring.
# ---------------------------------------------------------------------------

import io
import math
import tarfile
import zipfile


def _archives(tmp_path, hle_lines, lle_lines, stem="online"):
    tar_p = tmp_path / f"{stem}.tar.gz"
    data = "\n".join(hle_lines).encode()
    with tarfile.open(tar_p, "w:gz") as tf:
        info = tarfile.TarInfo("Maritime Composite Events/CEs/recognised_CEs.csv")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    zip_p = tmp_path / f"{stem}.zip"
    with zipfile.ZipFile(zip_p, "w") as z:
        z.writestr("brest_critical.csv", "\n".join(lle_lines))
    return str(tar_p), str(zip_p)


def _cv_archives(tmp_path, n_pos=6, n_neg=6):
    """The planted-rule mini-corpus of test_maritime_soft: per positive
    pair the pts are 900/1000/2000/2100/2200, gold rendezVous [1000, 2000)
    covers exactly {1000}, and `both_stopped_far` covers exactly {1000}
    too — a perfect 1-literal soft body. Negative pairs carry proximity
    only. Every pair shares the same clock, so the global chronological
    stream interleaves all pairs and exercises the time tie-break."""
    hle, lle = [], []
    for i in range(n_pos):
        a, b = f"P{i:02d}a", f"P{i:02d}b"
        hle += [
            f"rendezVous|{a}|{b}|true|1000|2000",
            f"lowSpeed|{a}| |true|900|2100",
            f"lowSpeed|{b}| |true|900|2100",
            f"stopped|{a}| |farFromPorts|1000|2000",
            f"stopped|{b}| |farFromPorts|1000|2000",
        ]
        lle.append(f"proximity|2200|900|2200|true|{a}|{b}")
    for j in range(n_neg):
        a, b = f"N{j:02d}a", f"N{j:02d}b"
        lle.append(f"proximity|2200|900|2200|true|{a}|{b}")
    return _archives(tmp_path, hle, lle, stem="cv")


CV_ARGS = ["--smoke", "--skip-verify", "--folds", "3"]


def _run_cv(tmp_path, extra_args, out_name="out.json"):
    import json

    import run_maritime_cv

    tar_p, zip_p = _cv_archives(tmp_path)
    out = tmp_path / out_name
    rc = run_maritime_cv.main(
        ["--tar", tar_p, "--zip", zip_p, "--out", str(out)] + CV_ARGS + extra_args
    )
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_cv_defaults_stay_byte_identical_with_the_online_column_present(tmp_path):
    pytest.importorskip("torch")
    result = _run_cv(tmp_path, [])
    # the same pre-change snapshot test_maritime_soft pins: adding the
    # online column must not move a single default-path byte.
    assert result["micro"]["point"] == {
        "tp": 6, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
    }
    assert [f["clauses"] for f in result["folds"]] == [
        [["both_low_or_stopped", "both_stopped_far"]],
        [["both_low_or_stopped", "both_stopped_far"]],
        [["both_low_or_stopped", "both_stopped_far"]],
    ]
    assert result["fold_of_pair"] == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    # the params stamp is additive: the stream knobs exist but are unset
    assert result["params"]["column"] == "hard"
    assert result["params"]["stream_order"] is None
    assert result["params"]["stream_window"] is None


def test_cv_online_column_recovers_planted_rule(tmp_path):
    pytest.importorskip("torch")
    result = _run_cv(tmp_path, ["--column", "online"], out_name="online.json")
    assert result["params"]["column"] == "online"
    assert result["params"]["stream_order"] == "chrono"
    assert result["params"]["stream_window"] == 1000
    assert result["micro"]["point"]["f1"] == pytest.approx(1.0)
    for record in result["folds"]:
        assert record["column"] == "online"
        assert record["stream_order"] == "chrono"
        # the pool and the gate are the soft column's, byte-the-same
        assert record["n_bodies_pool"] == 231
        assert record["n_bodies_gated"] >= 1
        assert record["min_fit"] == record["null_summary"]["threshold"]
        assert record["scoring"]["point"]["f1"] == pytest.approx(1.0)
        assert record["online_params"] == {
            "lr": 0.05, "threshold": 0.5, "init_logit": -2.0,
            "window": 1000, "order": "chrono", "passes": 1,
        }
        # one window on this tiny fold: the whole train side fits in 1000
        assert record["stream_windows"] == 1
        assert len(record["prequential_curve"]) == 1
        assert record["wall_s_pass"] >= 0.0


def test_cv_online_prequential_curve_has_one_entry_per_window(tmp_path):
    pytest.importorskip("torch")
    result = _run_cv(
        tmp_path, ["--column", "online", "--stream-window", "7"],
        out_name="win7.json",
    )
    assert result["params"]["stream_window"] == 7
    for record in result["folds"]:
        expected = math.ceil(record["n_train_pt"] / 7)
        assert record["stream_windows"] == expected
        assert len(record["prequential_curve"]) == expected
        for entry in record["prequential_curve"]:
            assert set(entry) == {"n_pt", "errors", "error_rate"}
            assert 0 <= entry["errors"] <= entry["n_pt"]
            assert entry["error_rate"] == pytest.approx(entry["errors"] / entry["n_pt"])
        # the windows partition the train rows exactly
        assert sum(e["n_pt"] for e in record["prequential_curve"]) == record["n_train_pt"]


def test_cv_online_reverse_pass_is_valid_and_differs_from_chrono(tmp_path):
    # H-O2's diagnostic. With the default 1000-row window the whole train
    # side of this fixture is ONE window and per-batch-mean BCE is
    # invariant to the row order, so reverse == chrono bitwise and a
    # runner that ignored --stream-order would pass a validity-only check
    # (PR #270 review, section 8). A 7-row window makes the pass genuinely
    # order-dependent: the reverse weights must differ from chrono's.
    pytest.importorskip("torch")
    window = ["--stream-window", "7"]
    chrono = _run_cv(tmp_path, ["--column", "online"] + window, out_name="chrono7.json")
    result = _run_cv(
        tmp_path, ["--column", "online", "--stream-order", "reverse"] + window,
        out_name="reverse7.json",
    )
    assert result["params"]["stream_order"] == "reverse"
    assert result["params"]["stream_window"] == 7
    for record, chrono_record in zip(result["folds"], chrono["folds"]):
        assert record["stream_order"] == "reverse"
        assert record["stream_windows"] == len(record["prequential_curve"])
        assert record["stream_windows"] == chrono_record["stream_windows"]
        point = record["scoring"]["point"]
        assert 0.0 <= point["f1"] <= 1.0
        # the same pool and gate as the chrono pass ...
        assert record["n_bodies_gated"] == chrono_record["n_bodies_gated"]
        assert record["null_summary"] == chrono_record["null_summary"]
        # ... but a different pass: the learned weights are not the chrono ones
        assert record["weights_top10"] != chrono_record["weights_top10"]


def test_cv_online_two_runs_identical(tmp_path):
    pytest.importorskip("torch")

    def strip_walls(result):
        result.pop("convert_wall_s", None)
        for record in result["folds"]:
            record.pop("wall_s", None)
            record.pop("wall_s_pass", None)
        return result

    a = strip_walls(_run_cv(tmp_path, ["--column", "online"], out_name="a.json"))
    b = strip_walls(_run_cv(tmp_path, ["--column", "online"], out_name="b.json"))
    assert a == b


def test_cv_stream_flags_refused_without_online_column(tmp_path):
    import run_maritime_cv

    tar_p, zip_p = _cv_archives(tmp_path)
    base = ["--tar", tar_p, "--zip", zip_p, "--out", str(tmp_path / "x.json")] + CV_ARGS
    with pytest.raises(SystemExit) as exc:
        run_maritime_cv.main(base + ["--stream-order", "reverse"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        run_maritime_cv.main(base + ["--column", "soft", "--stream-order", "chrono"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        run_maritime_cv.main(base + ["--stream-window", "500"])
    assert exc.value.code == 2


def test_cv_online_column_refuses_the_duration_vocabulary(tmp_path):
    # PREREG_ONLINE.md section 5: the online column runs ONLY on the base
    # vocabulary — `sustained_240` uses the FUTURE duration of its interval
    # (a leak that a stream must not see), so `--column online --vocab
    # duration` is a protocol violation and must be refused up front, like
    # the stream flags without the online column.
    import run_maritime_cv

    tar_p, zip_p = _cv_archives(tmp_path)
    base = ["--tar", tar_p, "--zip", zip_p, "--out", str(tmp_path / "x.json")] + CV_ARGS
    with pytest.raises(SystemExit) as exc:
        run_maritime_cv.main(base + ["--column", "online", "--vocab", "duration"])
    assert exc.value.code == 2
    # the batch soft column keeps the duration vocabulary (PREREG_SOFT (c))
    assert run_maritime_cv.parse_args(base + ["--column", "soft", "--vocab", "duration"]).vocab == "duration"
