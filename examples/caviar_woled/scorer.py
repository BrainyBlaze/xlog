"""Pure-Python/torch scoring for the CAVIAR star-search probe (task S3a).

CPU-only, no engine import: everything here works on the plain Python
``relations``/``is_positive`` structures `caviar_convert.convert_split`
already returns, so a selected rule (or a hand-picked baseline) can be
evaluated on the TEST split without a second engine pass. This is an exact
set-intersection reading of the star rule, not a re-derivation through the
engine -- honestly documented in `rule_predictions`' docstring below.

NAMING CONVENTION (read this before wiring a pod runner against it): a star
candidate is identified by a **tuple of two plain relation names**,
``(left, right)`` -- e.g. ``("both_active", "close")`` -- exactly as
`pyxlog.ilp.neural_credit.CandidateSpec.left`/``.right`` and
`HoldoutSelection.rule`/``.tied`` carry it. There is a *separate*,
purely-internal ``"left|right"`` STRING key that `neural_credit`'s
`_select_from_holdout` uses as a transient dict key for
`discovery.select_rule` (whose weights are keyed by a single string) -- it is
split back into a tuple before ever reaching a caller, so nothing in this
module or in `HoldoutSelection` ever hands you that pipe-joined form. Use the
tuple.
"""

from __future__ import annotations


def rule_predictions(
    rule_left: str, rule_right: str, relations: dict, num_pt: int
) -> list[bool]:
    """Prediction for every pair-time ``pt`` in ``range(num_pt)``: ``True``
    iff ``pt`` is a member of BOTH named relations.

    This is the STAR body read literally: ``head(X, Y) :- bL(X, Y), bR(X, Y)``
    with both atoms keyed by the same argument as the head (see
    `pyxlog.ilp.neural_credit.enumerate_specs`'s star-mode docstring) --
    the relational-relational cover there is exactly
    ``1.0 iff (x, y) in A and (x, y) in B``. `caviar_convert.convert_split`
    always emits relation rows as ``(pt, 1)`` (a fixed label column, per its
    own docstring), so membership is checked at that fixed label: a pair-time
    is covered iff ``(pt, 1)`` appears in both ``relations[rule_left]`` and
    ``relations[rule_right]``.

    This is an EXACT reimplementation of the engine's star cover in plain
    Python, not a second engine pass -- a star rule has no existential to
    join, so nothing is lost by scoring it this way on a split the engine
    never saw (the held-out TEST split), and doing so avoids compiling a
    second program just to read back a set intersection.

    Raises ``KeyError`` (with the offending name) if either relation is
    absent from ``relations`` -- a typo in a selected rule's name should
    surface immediately, not silently score every pair-time ``False``.
    """
    for name in (rule_left, rule_right):
        if name not in relations:
            raise KeyError(
                f"rule_predictions: relation {name!r} is not in `relations` "
                f"(have: {sorted(relations)}). A star rule's body names must "
                "come straight from the same `relations` dict, so a missing "
                "name is refused rather than silently scored empty."
            )
    left_set = set(relations[rule_left])
    right_set = set(relations[rule_right])
    return [(pt, 1) in left_set and (pt, 1) in right_set for pt in range(num_pt)]


def prf1(pred: list[bool], gold: list[bool]) -> dict:
    """Precision/recall/F1 plus the raw tp/fp/fn/tn counts.

    Every ratio that would otherwise divide by zero (no predicted positives,
    no actual positives, or both precision and recall are undefined/zero) is
    reported as ``0.0`` rather than raising or returning ``nan`` -- ``nan``
    would silently poison any downstream comparison (``nan == nan`` is
    ``False``, ``nan > x`` is always ``False``) -- and ``"degenerate"`` is
    set ``True`` whenever ANY such zero-division substitution happened, so a
    caller can tell "genuinely 0.0" apart from "undefined, reported as 0.0"
    without re-deriving it from tp/fp/fn.
    """
    if len(pred) != len(gold):
        raise ValueError(
            f"prf1: pred has {len(pred)} entries, gold has {len(gold)} -- "
            "they must be aligned one-to-one over the same pair-times."
        )
    tp = fp = fn = tn = 0
    for p, g in zip(pred, gold):
        p, g = bool(p), bool(g)
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif not p and g:
            fn += 1
        else:
            tn += 1

    degenerate = False
    if tp + fp == 0:
        precision = 0.0
        degenerate = True
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0.0
        degenerate = True
    else:
        recall = tp / (tp + fn)
    if precision + recall == 0:
        f1 = 0.0
        degenerate = True
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "degenerate": degenerate,
    }


DEFAULT_BASELINE_PAIRS: list[tuple[str, str]] = [
    ("both_active", "close"),
    ("both_inactive", "close"),
    ("mixed_active_walking", "close"),
]


def baseline_report(
    relations: dict, gold: list[bool], num_pt: int, pairs: list[tuple[str, str]] | None = None
) -> dict:
    """F1 table for a handful of hand-picked star bodies plus the trivial
    all-positive baseline, keyed by ``"{left}|{right}"`` (a plain display
    label here -- JSON object keys must be strings, so this is the
    RESULT.json-friendly form; it is NOT the engine's candidate identity,
    which stays the ``(left, right)`` tuple everywhere else -- see the
    module docstring).

    ``pairs`` defaults to `DEFAULT_BASELINE_PAIRS`; ``"all_positive"`` (every
    pair-time predicted ``True``) is always added on top of whatever
    ``pairs`` names, as the trivial reference point every real candidate
    should beat.
    """
    if pairs is None:
        pairs = DEFAULT_BASELINE_PAIRS
    report = {}
    for left, right in pairs:
        pred = rule_predictions(left, right, relations, num_pt)
        report[f"{left}|{right}"] = prf1(pred, gold)
    report["all_positive"] = prf1([True] * num_pt, gold)
    return report
