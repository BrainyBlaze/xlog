"""Unit tests for `caviar_convert.derive_ec_targets` and `ec_scorer.py` --
CPU, no pkl, no engine. Every label sequence below is hand-built and every
expected `is_init`/`is_term`/holds value is hand-computable, following the
style of `test_caviar_convert.py`/`test_caviar_scorer.py`.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from caviar_convert import derive_ec_targets  # noqa: E402
from ec_scorer import frame_f1, reconstruct_holds  # noqa: E402


def _dp(complex_labels):
    """Minimal datapoint dict: `derive_ec_targets` (via `window_length` and
    its own label decoding) only ever reads `complex_labels`."""
    return {"complex_labels": list(complex_labels)}


# ---------------------------------------------------------------------------
# derive_ec_targets
# ---------------------------------------------------------------------------

# dp0 (T=4): target holds from the very first timestep, then terminates at
# the last -- covers "init at window start" and "termination clears".
DP0 = _dp([1, 1, 1, 0])
# dp1 (T=4): target starts mid-window -- covers "init mid-window".
DP1 = _dp([0, 0, 1, 1])
# dp2 (T=4): terminates then re-initiates then terminates again -- covers
# "re-initiation after termination".
DP2 = _dp([1, 0, 1, 0])


def test_init_at_window_start_and_termination_clears():
    out = derive_ec_targets([DP0], target_label_id=1)
    assert out["is_init"] == [True, False, False, False]
    assert out["is_term"] == [False, False, False, True]
    assert out["n_init"] == 1
    assert out["n_term"] == 1


def test_init_mid_window():
    out = derive_ec_targets([DP1], target_label_id=1)
    assert out["is_init"] == [False, False, True, False]
    assert out["is_term"] == [False, False, False, False]
    assert out["n_init"] == 1
    assert out["n_term"] == 0


def test_reinitiation_after_termination():
    out = derive_ec_targets([DP2], target_label_id=1)
    assert out["is_init"] == [True, False, True, False]
    assert out["is_term"] == [False, True, False, True]
    assert out["n_init"] == 2
    assert out["n_term"] == 2


def test_derive_ec_targets_aggregates_num_pt_and_t_across_datapoints():
    out = derive_ec_targets([DP0, DP1, DP2], target_label_id=1)
    assert out["T"] == 4
    assert out["num_pt"] == 12
    assert out["is_init"] == (
        [True, False, False, False]
        + [False, False, True, False]
        + [True, False, True, False]
    )
    assert out["is_term"] == (
        [False, False, False, True]
        + [False, False, False, False]
        + [False, True, False, True]
    )
    assert out["n_init"] == 1 + 1 + 2
    assert out["n_term"] == 1 + 0 + 2


def test_all_target_window_has_one_init_and_no_term():
    out = derive_ec_targets([_dp([1, 1, 1])], target_label_id=1)
    assert out["is_init"] == [True, False, False]
    assert out["is_term"] == [False, False, False]
    assert out["n_init"] == 1
    assert out["n_term"] == 0


def test_no_target_window_has_no_init_and_no_term():
    out = derive_ec_targets([_dp([0, 0, 0])], target_label_id=1)
    assert out["is_init"] == [False, False, False]
    assert out["is_term"] == [False, False, False]
    assert out["n_init"] == 0
    assert out["n_term"] == 0


def test_derive_ec_targets_respects_a_non_default_target_label_id():
    # Same raw labels as DP2, but target the OTHER value (0 instead of 1):
    # init/term swap relative to test_reinitiation_after_termination.
    out = derive_ec_targets([DP2], target_label_id=0)
    assert out["is_init"] == [False, True, False, True]
    assert out["is_term"] == [False, False, True, False]


def test_derive_ec_targets_rejects_empty_datapoints():
    with pytest.raises(ValueError):
        derive_ec_targets([])


def test_derive_ec_targets_rejects_disagreeing_window_length():
    short_dp = _dp([1, 0])
    with pytest.raises(ValueError, match="window length"):
        derive_ec_targets([DP0, short_dp])


# ---------------------------------------------------------------------------
# reconstruct_holds
# ---------------------------------------------------------------------------


def test_reconstruct_holds_matches_source_labels_for_a_single_termination():
    # Same shape as DP0's own is_init/is_term -- holds should reproduce the
    # original target-label sequence [True, True, True, False].
    init_pred = [True, False, False, False]
    term_pred = [False, False, False, True]
    assert reconstruct_holds(init_pred, term_pred, num_windows=1, T=4) == [
        True, True, True, False,
    ]


def test_reconstruct_holds_matches_source_labels_for_reinitiation():
    # Same shape as DP2's own is_init/is_term -- holds should reproduce
    # [True, False, True, False].
    init_pred = [True, False, True, False]
    term_pred = [False, True, False, True]
    assert reconstruct_holds(init_pred, term_pred, num_windows=1, T=4) == [
        True, False, True, False,
    ]


def test_reconstruct_holds_simultaneous_init_and_term_step_holds():
    # t1 has BOTH init and term true: the documented rule (term clears
    # first, then init re-sets in the same step) makes t1 HOLD.
    init_pred = [True, True, False]
    term_pred = [False, True, False]
    assert reconstruct_holds(init_pred, term_pred, num_windows=1, T=3) == [
        True, True, True,
    ]


def test_reconstruct_holds_resets_state_at_each_windows_own_start():
    # window0 ends HOLDING (init at t0, never terminated); window1 has no
    # init at all -- it must NOT inherit window0's holding state.
    init_pred = [True, True, False, False]
    term_pred = [False, False, False, False]
    assert reconstruct_holds(init_pred, term_pred, num_windows=2, T=2) == [
        True, True, False, False,
    ]


def test_reconstruct_holds_rejects_mismatched_init_length():
    with pytest.raises(ValueError):
        reconstruct_holds([True, False], [True, False, False], num_windows=1, T=3)


def test_reconstruct_holds_rejects_mismatched_term_length():
    with pytest.raises(ValueError):
        reconstruct_holds([True, False, False], [True, False], num_windows=1, T=3)


# ---------------------------------------------------------------------------
# frame_f1
# ---------------------------------------------------------------------------


def test_frame_f1_hand_computed_case():
    holds_pred = [True, False, True, False]
    holds_gold = [True, True, True, False]
    out = frame_f1(holds_pred, holds_gold)
    # tp: 0, 2 = 2; fn: 1 (pred False, gold True); fp: 0; tn: 3.
    assert out["tp"] == 2
    assert out["fp"] == 0
    assert out["fn"] == 1
    assert out["tn"] == 1
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(2 / 3)
    assert out["f1"] == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))
    assert out["degenerate"] is False


def test_frame_f1_length_mismatch_raises():
    with pytest.raises(ValueError):
        frame_f1([True, False], [True])
