"""Body pool + boolean coverage matrix for the pre-registered maritime
soft-credit column (`docs/experiments/maritime/PREREG_SOFT.md`, section
(b)): the pool is every conjunction of 1..3 DISTINCT literals over the
relation vocabulary — 1-literal bodies included, unlike the hard search's
`relational_search.enumerate_bodies` (2..3 literals, empty covers
skipped): a soft weight can turn a weak single relation down instead of
excluding it from the pool, and an empty cover is a legal all-zero row
here (the trainer just never receives gradient for it), so nothing is
dropped and no skip counter is needed.

This module owns ONLY the pool and its coverage geometry; it knows
nothing about folds, gates or training (those live in the CV runner and
`soft_weights`)."""

from __future__ import annotations

import itertools

Body = tuple[str, ...]


def enumerate_bodies(vocabulary: list[str], max_literals: int = 3) -> list[Body]:
    """Every sorted, unique combination of 1..`max_literals` DISTINCT
    relation names (combinations, not permutations: `("a", "b")` appears,
    `("b", "a")` never does), sizes in ascending order, each size in
    lexicographic order."""
    if max_literals < 1:
        raise ValueError(f"max_literals must be >= 1 (got {max_literals!r}).")
    names = sorted(vocabulary)
    if len(set(names)) != len(names):
        raise ValueError(f"vocabulary contains duplicate names: {vocabulary!r}")
    return [
        combo
        for size in range(1, max_literals + 1)
        for combo in itertools.combinations(names, size)
    ]


def coverage_matrix(bodies: list[Body], relations: dict[str, set[int]], n_pt: int):
    """Bool tensor `[n_bodies, n_pt]`: row b is True at pt iff pt is a
    member of EVERY relation named in body b (set intersection). An empty
    intersection is legal — an all-False row, never an error."""
    import torch

    sets = {name: set(members) for name, members in relations.items()}
    matrix = torch.zeros((len(bodies), n_pt), dtype=torch.bool)
    for row, body in enumerate(bodies):
        cover = set.intersection(*(sets[name] for name in body))
        for pt in cover:
            matrix[row, pt] = True
    return matrix
