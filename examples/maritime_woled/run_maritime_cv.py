"""Pre-registered cross-validation over the converted Brest AIS rendezVous
corpus (`maritime_convert.convert`) -- the maritime counterpart of
`examples/caviar_woled/run_caviar_cv.py`, under the protocol fixed IN
ADVANCE by `docs/experiments/maritime/README.md`'s Pre-registration
section. This module implements EXACTLY that protocol, nothing else; every
constant below is named there.

FOLD ASSIGNMENT (`stratified_pair_folds`). Fold atoms are VESSEL PAIRS --
a pair's pt rows are never split across folds (the top-1 pair carries
33.4% of all positive pt, top-4 carry 60.2%; any finer split leaks the
dominant scene across the train/test boundary). Deterministic greedy LPT,
no RNG anywhere: positive pairs sorted by (positive-pt count DESC, pair
index ASC) are each dealt to the fold with the smallest running
positive-pt sum (ties: lowest fold index); negative pairs then sorted by
(pt-row count DESC, pair index ASC) are dealt the same way against the
folds' running NEGATIVE pt-row sums. Guaranteed property (tested): for
either load, ``max fold sum - min fold sum <= largest single pair's own
count`` -- when the eventual max fold received its LAST pair, it was the
least-loaded fold, so its pre-deal load was <= every other fold's FINAL
load (loads only grow); adding that one pair's count therefore lifts it
above the eventual minimum by at most that count.
"""

from __future__ import annotations


def stratified_pair_folds(
    positive_pt_counts: list[int],
    pt_counts: list[int],
    n_positive_pairs: int,
    n_folds: int,
) -> list[int]:
    """One fold index (0..n_folds-1) per pair, pairs ordered exactly as
    `maritime_convert.convert`'s own ``"pairs"`` list (positives first,
    then negatives -- ``n_positive_pairs`` marks the boundary).

    ``positive_pt_counts`` has one entry per POSITIVE pair (its own
    positive-pt row count); ``pt_counts`` has one entry per pair, positives
    AND negatives (its own total pt row count). See the module docstring
    for the LPT mechanism and its balance guarantee.

    Raises ``ValueError`` if ``n_folds < 2``, if either side has fewer
    pairs than folds (every fold must hold >= 1 positive and >= 1 negative
    pair -- a fold without positives cannot be scored on the positive
    class, a fold without negatives cannot expose false positives), if
    ``len(positive_pt_counts) != n_positive_pairs``, or if any count is
    negative."""
    n_pairs = len(pt_counts)
    n_negative_pairs = n_pairs - n_positive_pairs
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 (got {n_folds!r}).")
    if len(positive_pt_counts) != n_positive_pairs:
        raise ValueError(
            f"positive_pt_counts has {len(positive_pt_counts)} entries, "
            f"expected n_positive_pairs={n_positive_pairs}."
        )
    if n_positive_pairs < n_folds:
        raise ValueError(
            f"{n_positive_pairs} positive pairs is fewer than "
            f"n_folds={n_folds}: every fold needs >= 1 positive pair to "
            "score the positive class on."
        )
    if n_negative_pairs < n_folds:
        raise ValueError(
            f"{n_negative_pairs} negative pairs is fewer than "
            f"n_folds={n_folds}: every fold needs >= 1 negative pair to "
            "expose false positives."
        )
    if any(c < 0 for c in positive_pt_counts) or any(c < 0 for c in pt_counts):
        raise ValueError("pair counts must all be non-negative.")

    fold_of_pair = [0] * n_pairs

    def _deal(indices: list[int], counts_of: dict[int, int]) -> None:
        load = [0] * n_folds
        order = sorted(indices, key=lambda i: (-counts_of[i], i))
        for i in order:
            fold = min(range(n_folds), key=lambda f: (load[f], f))
            fold_of_pair[i] = fold
            load[fold] += counts_of[i]

    _deal(
        list(range(n_positive_pairs)),
        {i: positive_pt_counts[i] for i in range(n_positive_pairs)},
    )
    _deal(
        list(range(n_positive_pairs, n_pairs)),
        {i: pt_counts[i] for i in range(n_positive_pairs, n_pairs)},
    )
    return fold_of_pair


def pair_counts(converted: dict, n_pairs: int) -> tuple[list[int], list[int]]:
    """Per-pair stratification masses read off one converted corpus:
    ``(positive_pt_counts, pt_counts)``, each one entry per pair index
    0..n_pairs-1 -- ``positive_pt_counts[p]`` is pair ``p``'s own
    positive-pt row count (0 for negative pairs, by construction),
    ``pt_counts[p]`` its total pt row count. Reads only
    ``converted["pt_pair_index"]`` and ``converted["is_positive"]``."""
    positive_pt_counts = [0] * n_pairs
    pt_counts = [0] * n_pairs
    for pair_idx, pos in zip(converted["pt_pair_index"], converted["is_positive"]):
        pt_counts[pair_idx] += 1
        if pos:
            positive_pt_counts[pair_idx] += 1
    return positive_pt_counts, pt_counts
