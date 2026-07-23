"""Unit tests for `caviar_convert.convert_split` -- CPU, no pkl.

Builds two tiny synthetic datapoints (T=3, not 24: the converter must infer
the window length from `complex_labels`, never hardcode it) with hand-known
labels/atoms and asserts exact relation contents, positives, features, and
missing-coords handling.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "caviar_woled"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from caviar_convert import (  # noqa: E402
    SIMPLE_LABEL_NAMES,
    build_star_schema_source,
    convert_split,
    window_length,
)

ACTIVE = 0
INACTIVE = 1
RUNNING = 2
WALKING = 3
NO_INTERACTION = 0
MEETING = 1
MOVING = 2


def _atoms_for(coords_p1, coords_p2):
    """Build an atoms string with `coords(p1,X,Y,T)`/`coords(p2,X,Y,T)` for
    the given `{t: (x, y)}` maps, skipping a timestep entirely from one
    person's side to model a missing-coords row."""
    parts = []
    for t, (x, y) in coords_p1.items():
        parts.append(f"orientation(p1,0,{t}). coords(p1,{x},{y},{t}).  visible(p1,{t}).")
    for t, (x, y) in coords_p2.items():
        parts.append(f"orientation(p2,0,{t}). coords(p2,{x},{y},{t}).  visible(p2,{t}).")
    return "".join(parts)


def _dp(tag, p1_labels, p2_labels, complex_labels, coords_p1, coords_p2):
    T = len(complex_labels)
    return {
        "tag": tag,
        "p1_labels": torch.tensor(p1_labels, dtype=torch.int64).reshape(T, 1),
        "p2_labels": torch.tensor(p2_labels, dtype=torch.int64).reshape(T, 1),
        "complex_labels": torch.tensor(complex_labels, dtype=torch.int64).reshape(T, 1),
        "atoms": _atoms_for(coords_p1, coords_p2),
    }


# --- datapoint 0 (dp_index=0, T=3): a clean, fully-coordinated window -----
# t=0: both active,  close (dist=5),                    meeting
# t=1: both walking, far   (dist=100),                  no_interaction
# t=2: p1 active / p2 walking (mixed), close (dist=3),  moving
DP0 = _dp(
    tag=7,
    p1_labels=[ACTIVE, WALKING, ACTIVE],
    p2_labels=[ACTIVE, WALKING, WALKING],
    complex_labels=[MEETING, NO_INTERACTION, MOVING],
    coords_p1={0: (0, 0), 1: (0, 0), 2: (0, 0)},
    coords_p2={0: (3, 4), 1: (100, 0), 2: (0, 3)},
)

# --- datapoint 1 (dp_index=1, T=3): missing coords + both-inactive --------
# t=0: both inactive, p2 coords missing -> coords_missing, features (0,0)
# t=1: both running,  far (dist=50),                     no_interaction
# t=2: both inactive, close (dist=0),                    meeting
DP1 = _dp(
    tag=9,
    p1_labels=[INACTIVE, RUNNING, INACTIVE],
    p2_labels=[INACTIVE, RUNNING, INACTIVE],
    complex_labels=[NO_INTERACTION, NO_INTERACTION, MEETING],
    coords_p1={0: (10, 10), 1: (0, 0), 2: (5, 5)},
    coords_p2={1: (50, 0), 2: (5, 5)},  # t=0 missing on p2's side
)

DATAPOINTS = [DP0, DP1]


def test_window_length_is_inferred_not_hardcoded():
    assert window_length(DP0) == 3
    assert window_length(DP1) == 3


def test_num_pt_and_facts_cover_every_pair_time():
    out = convert_split(DATAPOINTS)
    assert out["num_pt"] == 6                       # 2 datapoints * T=3
    assert out["facts"] == [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]


def test_is_positive_matches_meeting_label_per_pair_time():
    out = convert_split(DATAPOINTS)
    # pt 0..2 = dp0 t0..2 (meeting, no_interaction, moving)
    # pt 3..5 = dp1 t0..2 (no_interaction, no_interaction, meeting)
    assert out["is_positive"] == [True, False, False, False, False, True]


def test_both_active_both_walking_mixed_relations_exact():
    out = convert_split(DATAPOINTS)
    rel = out["relations"]
    assert rel["both_active"] == [(0, 1)]                       # only dp0 t0
    assert rel["both_walking"] == [(1, 1)]                      # only dp0 t1
    assert rel["mixed_active_walking"] == [(2, 1)]               # dp0 t2 (active/walking)
    assert set(rel["both_inactive"]) == {(3, 1), (5, 1)}         # dp1 t0, t2


def test_close_far_partition_by_threshold_and_omit_missing_coords():
    out = convert_split(DATAPOINTS, close_threshold=25.0)
    rel = out["relations"]
    # dp0: t0 dist=5 (close), t1 dist=100 (far), t2 dist=3 (close)
    # dp1: t0 missing coords (omitted from both), t1 dist=50 (far), t2 dist=0 (close)
    assert set(rel["close"]) == {(0, 1), (2, 1), (5, 1)}
    assert set(rel["far"]) == {(1, 1), (4, 1)}
    assert (3, 1) not in rel["close"] and (3, 1) not in rel["far"]
    assert rel["coords_missing"] == [(3, 1)]
    assert out["n_coords_missing"] == 1


def test_features_are_scaled_signed_p1_minus_p2_deltas_and_zero_when_missing():
    out = convert_split(DATAPOINTS)
    feats = out["features"]
    assert feats.shape == (6, 2)
    assert feats.dtype == torch.float32
    # dp0 t0: p1=(0,0) p2=(3,4) -> dx=-3, dy=-4, /100
    assert feats[0].tolist() == pytest.approx([-0.03, -0.04])
    # dp0 t1: p1=(0,0) p2=(100,0) -> dx=-100, dy=0
    assert feats[1].tolist() == pytest.approx([-1.0, 0.0])
    # dp1 t0: coords missing -> (0, 0)
    assert feats[3].tolist() == pytest.approx([0.0, 0.0])
    # dp1 t2: p1=(5,5) p2=(5,5) -> (0, 0) but NOT coords_missing
    assert feats[5].tolist() == pytest.approx([0.0, 0.0])
    assert (5, 1) not in out["relations"]["coords_missing"]


def test_convert_split_rejects_datapoints_with_disagreeing_window_length():
    short_dp = _dp(
        tag=0, p1_labels=[ACTIVE, ACTIVE], p2_labels=[ACTIVE, ACTIVE],
        complex_labels=[MEETING, MEETING],
        coords_p1={0: (0, 0), 1: (0, 0)}, coords_p2={0: (0, 0), 1: (0, 0)},
    )
    with pytest.raises(ValueError, match="window length"):
        convert_split([DP0, short_dp])


def test_convert_split_rejects_empty_datapoints():
    with pytest.raises(ValueError):
        convert_split([])


def test_simple_label_names_match_the_pkl_encoder():
    # simple_label_encoder from the real caviar_folds.pkl: active=0,
    # inactive=1, running=2, walking=3 -- pinned here so a silent reordering
    # of SIMPLE_LABEL_NAMES is caught.
    assert SIMPLE_LABEL_NAMES == {0: "active", 1: "inactive", 2: "running", 3: "walking"}


def test_build_star_schema_source_has_one_seed_row_per_relation_and_the_rule():
    src = build_star_schema_source(["close", "both_active"])
    assert "close(0, 0)." in src
    assert "both_active(0, 0)." in src
    assert "learnable(W) :: init_meeting(X, Y) :- bL(X, Y), bR(X, Y)." in src
