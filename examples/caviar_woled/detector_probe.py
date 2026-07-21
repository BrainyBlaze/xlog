"""Pure-Python/torch detector-probe helpers for the CAVIAR neural-close
differentiator (task S4a).

This module is the DIFFERENTIATOR'S EVIDENCE: `run_caviar_neural.py` trains a
small MLP (`close_nn`) jointly through the star-topology engine credit, over
raw `(dx, dy)` pair-time coordinates, WITHOUT ever showing it the precomputed
`close`/`far` relations (those are excluded from the candidate vocabulary
entirely -- see `run_caviar_neural.py`'s module docstring). This module then
answers, on plain torch tensors with NO engine import at all: did the network
actually learn something that looks like a distance detector, compared
honestly against the ground-truth `close` relation it never trained on?

Every function here takes plain sequences/tensors (duck-typed: anything that
supports `float(x)` per element and `len()`) and returns plain dicts/lists --
no `torch` import at module level, matching `scorer.py`'s "CPU-testable, no
engine" contract; a caller may pass real `torch.Tensor` objects (as
`run_caviar_neural.py` does) or plain Python lists (as the unit tests do),
transparently.

Reuses `scorer.prf1` (already pure Python) for precision/recall/F1 rather
than re-deriving it -- the accuracy/prf1 arithmetic must match the rest of
the CAVIAR pipeline exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from scorer import prf1  # noqa: E402  -- pure Python, no torch/engine import

# Distance-bin edges (module constant, per the task brief): 10 finite 5-unit
# bins covering [0, 50) plus one open-ended "50+" bin for anything at or past
# the last edge -- 11 bins total. CAVIAR's `close_threshold` (25.0, see
# `caviar_convert.py`) falls at the boundary between bin index 4 ("20-25")
# and bin index 5 ("25-30"): a genuinely learned distance detector is
# expected to show its steepest drop ("knee") near that boundary, even
# though the network is never told the threshold or shown `close` at all.
DIST_BIN_EDGES: tuple[float, ...] = (
    0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0,
)


def bin_labels() -> list[str]:
    """Human-readable label per bin, e.g. ``"0-5"``, ..., ``"45-50"``,
    ``"50+"`` -- 11 labels, matching `DIST_BIN_EDGES`."""
    labels = [
        f"{DIST_BIN_EDGES[i]:.0f}-{DIST_BIN_EDGES[i + 1]:.0f}"
        for i in range(len(DIST_BIN_EDGES) - 1)
    ]
    labels.append(f"{DIST_BIN_EDGES[-1]:.0f}+")
    return labels


def assign_bin(dist: float) -> int:
    """Bin index for one non-negative distance: half-open ``[edges[i],
    edges[i+1])`` for the 10 finite bins, and the final ("50+") bin for
    anything ``>= edges[-1]``. A negative distance is a contract violation
    (a euclidean norm cannot be negative) and is refused rather than
    silently binned at index 0."""
    d = float(dist)
    if d < 0:
        raise ValueError(
            f"distance {d} is negative; a euclidean distance cannot be, "
            "refused rather than silently binned."
        )
    for i in range(len(DIST_BIN_EDGES) - 1):
        if DIST_BIN_EDGES[i] <= d < DIST_BIN_EDGES[i + 1]:
            return i
    return len(DIST_BIN_EDGES) - 1  # 50+


def monotone_decay_report(bins: list[dict]) -> dict:
    """Whether the per-bin mean scores (over bins that HAD at least one row,
    ``mean_score is not None``) are non-increasing as distance grows -- the
    signature of a genuinely learned distance detector -- and which adjacent
    populated-bin pair holds the single largest drop (the "knee"), named by
    label. Fewer than two populated bins makes monotonicity undefined
    (reported, never guessed at or crashed on)."""
    present = [
        (i, b["mean_score"]) for i, b in enumerate(bins) if b["mean_score"] is not None
    ]
    if len(present) < 2:
        return {
            "monotone_non_increasing": None,
            "knee_label": None,
            "knee_drop": 0.0,
            "reason": "fewer than two populated bins: monotonicity is undefined",
        }
    monotone = all(present[i][1] >= present[i + 1][1] for i in range(len(present) - 1))
    drops = [
        (present[i][1] - present[i + 1][1], i) for i in range(len(present) - 1)
    ]
    best_drop, best_i = max(drops, key=lambda t: t[0])
    knee_label = (
        f"{bins[present[best_i][0]]['label']}->{bins[present[best_i + 1][0]]['label']}"
    )
    return {
        "monotone_non_increasing": monotone,
        "knee_label": knee_label,
        "knee_drop": best_drop,
    }


def probe_detector(
    scores,
    close_rows: set[int],
    dists,
    threshold: float = 0.5,
    exclude_rows: set[int] | None = None,
) -> dict:
    """The differentiator's core evidence, over one split (train, test, or a
    control net's output).

    ``scores``: one real-valued score per pair-time row (e.g. the trained
    network's P(label=1) at that row -- the SAME quantity the star engine's
    cover-gated single-witness score reads, per `enumerate_specs`'
    star-mode docstring). ``close_rows``: the set of row indices the
    GROUND-TRUTH precomputed ``close`` relation marks true (never fed to
    training -- see the module docstring); an empty set is a legal,
    documented degenerate case (flagged via ``"no_close_rows"``, not a
    crash). ``dists``: the ground-truth euclidean distance per row (also
    never fed to training), used only to bucket rows for the mean-score-
    per-bin table -- see `DIST_BIN_EDGES`.

    ``exclude_rows`` (default ``None``, byte-identical to omitting it) drops
    rows from BOTH the accuracy/prf1 computation and the bin table -- e.g.
    CAVIAR's ``coords_missing`` rows, whose ``dist``/features are a
    fabricated ``(0, 0)`` (see `caviar_convert.py`), not a real distance;
    binning them would silently pollute the "0-5" bin with rows that have no
    real geometry at all.

    Returns a dict: ``"num_rows"`` (rows actually scored, after exclusion),
    ``"num_excluded"``, ``"no_close_rows"`` (bool), ``"accuracy"``,
    ``"prf1"`` (the full `scorer.prf1` dict, including its own
    ``"degenerate"`` flag), and ``"bins"`` (one entry per `bin_labels()`,
    each ``{"label", "count", "mean_score"}`` with ``mean_score`` ``None``
    for an empty bin -- never a divide-by-zero NaN)."""
    n_total = len(scores)
    if n_total == 0:
        raise ValueError(
            "probe_detector needs at least one row: an accuracy/bin table "
            "over zero rows is undefined."
        )
    if len(dists) != n_total:
        raise ValueError(
            f"scores has {n_total} rows but dists has {len(dists)}; they "
            "must be aligned one-to-one over the same pair-times."
        )
    # Intersect with the valid row range so num_excluded reports rows
    # actually removed, not the size of whatever set the caller passed.
    excluded = {i for i in (exclude_rows or set()) if 0 <= i < n_total}

    kept_idx = [i for i in range(n_total) if i not in excluded]
    if not kept_idx:
        raise ValueError(
            f"probe_detector: all {n_total} rows were excluded via "
            "`exclude_rows`; nothing remains to score."
        )

    scores_list = [float(scores[i]) for i in kept_idx]
    dists_list = [float(dists[i]) for i in kept_idx]
    pred = [s > threshold for s in scores_list]
    gold = [i in close_rows for i in kept_idx]

    metrics = prf1(pred, gold)
    n = len(pred)
    accuracy = sum(1 for p, g in zip(pred, gold) if p == g) / n

    labels = bin_labels()
    sums = [0.0] * len(labels)
    counts = [0] * len(labels)
    for s, d in zip(scores_list, dists_list):
        b = assign_bin(d)
        sums[b] += s
        counts[b] += 1
    bins = [
        {
            "label": labels[b],
            "count": counts[b],
            "mean_score": (sums[b] / counts[b]) if counts[b] else None,
        }
        for b in range(len(labels))
    ]

    return {
        "num_rows": n,
        "num_excluded": len(excluded),
        "no_close_rows": len(close_rows & set(kept_idx)) == 0,
        "accuracy": accuracy,
        "prf1": metrics,
        "bins": bins,
    }


# ---------------------------------------------------------------------------
# Anisotropy metrics (task S5a, deep-analysis proposal 3)
#
# The S4 deep analysis (`FINDINGS.md`, Finding 2) showed that `probe_detector`
# / `monotone_decay_report`'s per-DISTANCE-BIN aggregation, averaged over
# every angle that fell in a bin, completely hid a decision surface that was
# a diagonal band (not a disk): at EVERY radius 5-30 the raw score ranged
# 0.000-1.000 depending on angle alone (a polar-grid "spread" of 1.0), and
# the pair-order asymmetry (swapping which person is p1 vs p2, i.e. negating
# the (dx, dy) input) moved the score by 0.364 on average -- a third of the
# score's whole range, for a quantity ((dx, dy)'s sign) that is an artifact
# of person-numbering, not evidence. Bin-mean tables cannot show either
# effect; the two functions below make them visible directly in RESULT.json.
#
# Both take `score_fn`: a callable ``Tensor[N, 2] -> Tensor[N]`` (or anything
# a plain Python `for` loop and `float()` can walk -- these functions never
# call a torch-specific method on the RETURNED scores) that reads the
# network's own P(label=1) at each row, the exact quantity `probe_detector`
# already reads. `torch` itself is imported lazily inside each function (not
# at module level), keeping this module's own no-torch-import-at-module-level
# contract (see the module docstring) even though these two helpers need a
# real tensor to hand `score_fn` -- unlike `probe_detector`, which is
# duck-typed all the way through and works on plain floats already computed
# elsewhere, these two build the (dx, dy) grid THEMSELVES and so need torch
# to construct it.


def polar_spread(
    score_fn,
    radii: tuple[float, ...] = (10.0, 20.0, 25.0, 30.0),
    n_angles: int = 36,
    scale: float = 1.0 / 100.0,
) -> dict:
    """Per-radius angular spread of ``score_fn`` over a polar grid.

    For each ``r`` in ``radii``, samples ``n_angles`` points evenly spaced
    around the full circle of that RAW radius (``(r*cos(a), r*sin(a))`` for
    ``n_angles`` angles ``a`` in ``[0, 2*pi)``), scales them by ``scale``
    (default ``1/100`` -- matches `caviar_convert.FEATURE_SCALE`'s inverse,
    so a real trained ``close_nn`` net can be probed with its own native
    input convention with no unscaling on the caller's part), evaluates
    ``score_fn`` on the whole batch at once, and reports
    ``{"min", "max", "mean", "spread", "std"}`` (``spread = max - min``, the
    population ``std`` -- divide by ``n_angles``, not ``n_angles - 1``).

    A genuinely RADIAL score (a function of distance-from-origin alone) has
    every point on a fixed-radius circle at (up to floating-point trig
    rounding, ~1e-14) the exact same input norm, hence the exact same score
    -- ``spread`` collapses to ~0 for such a function, by construction, not
    by a statistical accident. A score that instead depends on angle (e.g.
    the S4 net's diagonal decision band) shows a spread up to 1.0 (a
    ``[0, 1]``-valued score's own full range) at every radius -- exactly the
    anisotropy `FINDINGS.md`'s polar grid found and the bin-mean tables
    could not show.

    ``radii``/``n_angles`` are refused (``ValueError``) if empty/non-positive
    -- a spread over zero points is undefined, not a silent ``0.0``.
    """
    import math

    import torch

    if not radii:
        raise ValueError("polar_spread needs at least one radius in `radii`.")
    if n_angles < 1:
        raise ValueError(f"polar_spread needs n_angles >= 1, got {n_angles}.")

    result: dict = {}
    for r in radii:
        angles = [2.0 * math.pi * i / n_angles for i in range(n_angles)]
        rows = [
            (r * math.cos(a) * scale, r * math.sin(a) * scale) for a in angles
        ]
        grid = torch.tensor(rows, dtype=torch.float32)
        scores = score_fn(grid)
        values = [float(s) for s in scores]

        mn = min(values)
        mx = max(values)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        result[r] = {
            "min": mn,
            "max": mx,
            "mean": mean,
            "spread": mx - mn,
            "std": variance ** 0.5,
        }
    return result


def pair_swap_asymmetry(score_fn, features) -> float:
    """``mean |score_fn(x) - score_fn(-x)|`` over every row of ``features``.

    The (dx, dy) pair-time input's SIGN is an artifact of which person in
    the pair is labeled p1 vs p2 -- an arbitrary numbering choice, not
    evidence -- so a detector that has actually learned a symmetric notion
    of "close" should score a pair the same regardless of which of the two
    (equivalent, mirrored) input rows it is handed. This is exactly the
    quantity `FINDINGS.md`'s Finding 2 measured as 0.364 on the S4 net (a
    third of the score's own range): the negated batch is built and scored
    HERE, in one extra `score_fn` call over the whole tensor, not per-row,
    so this stays cheap even for a large ``features``.

    ``features`` must support tensor negation (``-features``); a plain
    ``torch.Tensor`` is expected (this function does not itself construct
    the grid, unlike `polar_spread`, so it takes whatever the caller already
    has -- e.g. a real dataset's feature tensor). Refuses (``ValueError``) a
    zero-row ``features``: a mean over nothing is undefined, not ``0.0``.
    """
    import torch

    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features, dtype=torch.float32)
    if features.shape[0] == 0:
        raise ValueError(
            "pair_swap_asymmetry needs at least one row in `features`; a "
            "mean over zero rows is undefined."
        )

    s_pos = score_fn(features)
    s_neg = score_fn(-features)
    diffs = [abs(float(a) - float(b)) for a, b in zip(s_pos, s_neg)]
    return sum(diffs) / len(diffs)
