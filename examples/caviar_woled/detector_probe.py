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
