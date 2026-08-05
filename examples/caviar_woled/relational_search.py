"""Pure-Python, CPU-only, up-to-3-literal relational body search for the
Event-Calculus protocol -- the expressiveness upgrade `run_caviar_theory.py`'s
own `--protocol ec --mode relational` path was capped at (2-literal star
bodies: `(left, right)`), needed because the literature's CAVIAR
initiation/termination rules routinely need a THIRD literal (e.g.
`both_active & close & any_became_walking`) that a 2-literal pool can never
propose at all, no matter how the holdout arbiter is tuned.

KEY INSIGHT (this is the whole reason this module needs no `torch.cuda`, no
compiled `pyxlog` program, and no trained network at all): for a PURELY
RELATIONAL candidate pool, a body's "prediction" is a fixed SET-INTERSECTION
membership test -- `fact in (rel_a & rel_b & rel_c)` -- that does not depend
on any trained weight. `kfold_select`'s per-fold `train_engine_mode` call
exists to fit a MIXTURE WEIGHT (and, in neural mode, a detector network) over
the candidate pool; for an all-relational pool every candidate's held-out
score is the SAME whether or not that training happens, because the
prediction itself never reads a trained parameter (see
`run_caviar_theory.py`'s own `EMPTY_NEURAL_POOL_STEP_CAP` guard, which
already documents this for the EXISTING 2-literal relational path -- this
module is the 3-literal generalization of exactly that observation, not a
new claim). So a relational-only rule search needs only: (a) candidate
enumeration, (b) k-fold holdout scoring of a fixed cover's accuracy, (c) the
same tie/min_fit selection semantics `pyxlog.ilp.neural_credit`'s holdout
arbiter uses, and (d) the engine-agnostic sequential-covering loop
(`theory_loop.induce_theory`, reused here UNCHANGED). All four are plain
Python/`torch`-on-CPU; nothing here imports `pyxlog` or touches a CUDA
device.

WHY NOT REUSE `pyxlog.ilp.neural_credit._select_from_holdout` DIRECTLY. That
function's scores dict is keyed by a `(left, right)` 2-tuple and its internal
round-trip key format is `f"{left}|{right}"` (split back apart with
`l, r = rule.split("|")`) -- both hard-coded to arity 2, and its Occam
tie-break narrows toward "the relational candidate" specifically to choose
between a relational rule and a NEURAL one, a distinction that does not exist
in this module's pool (every candidate here is relational). `select_body`
below reuses the one piece of `_select_from_holdout` that IS shape-agnostic
-- `pyxlog.ilp.discovery.select_rule`, which only ever sees a
`dict[str, float]` of opaque string ids and returns an opaque string
winner/tie-list, no arity assumption anywhere -- and re-derives the
surrounding fit-gate/tie/abstain plumbing honestly for n-ary bodies (see
`select_body`'s own docstring for exactly how its Occam step differs from,
and generalizes, `_select_from_holdout`'s).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from pyxlog.ilp.discovery import select_rule
from pyxlog.ilp.neural_credit import holdout_fold_assignment
from scorer import prf1
from theory_loop import induce_theory

Body = tuple[str, ...]


def enumerate_bodies(
    relations: dict[str, list], max_literals: int = 3,
) -> tuple[list[Body], dict[int, int]]:
    """Every sorted, unique combination of 2 and (if `max_literals == 3`) 3
    DISTINCT relation names, in canonical (alphabetical) order -- so
    `("close", "both_active")` never appears, only `("both_active",
    "close")` does, and no relation appears twice in one body. A body whose
    literals' fact sets have an EMPTY intersection on this data (it can never
    fire) is skipped, not silently -- counted per body size and returned
    alongside the pool, so a caller can report exactly how many candidates
    were structurally vacuous rather than merely small in number.

    Returns `(bodies, skipped)`: `bodies` is the survivable pool (order:
    every 2-literal body before any 3-literal one, each size internally
    sorted); `skipped` maps body size (2, and 3 if requested) to the count of
    empty-cover combinations dropped at that size. Deviates from a bare
    `-> list[...]` return specifically so the skip counts are never lost --
    the task this module exists for is explicit that a silent cap is not
    acceptable here.
    """
    if max_literals not in (2, 3):
        raise ValueError(
            f"max_literals must be 2 or 3 (got {max_literals!r}) -- this "
            "search is scoped to the 2- and 3-literal EC bodies the "
            "literature's CAVIAR rules need; no other arity is wired up."
        )
    names = sorted(relations)
    sets = {n: set(relations[n]) for n in names}
    bodies: list[Body] = []
    skipped: dict[int, int] = {size: 0 for size in range(2, max_literals + 1)}
    for size in range(2, max_literals + 1):
        for combo in itertools.combinations(names, size):
            cover = set.intersection(*(sets[n] for n in combo))
            if not cover:
                skipped[size] += 1
                continue
            bodies.append(combo)
    return bodies, skipped


def body_cover(body: Body, relations: dict[str, list]) -> set:
    """The body's fixed cover: the SET INTERSECTION of every named relation's
    own fact set -- exactly `run_caviar_theory._predict_clause_relational`'s
    reading, generalized from exactly 2 names to any number."""
    return set.intersection(*(set(relations[name]) for name in body))


def make_predict_clause(relations: dict[str, list]):
    """A `predict_clause(rule, fact) -> bool` closure for a body of ANY
    arity: exact set-intersection membership, precomputing one `set` per
    relation name up front (not per call) -- the n-ary generalization of
    `run_caviar_theory._predict_clause_relational`'s 2-literal reading;
    behavior is identical to it when `rule` happens to have length 2."""
    sets = {name: set(rows) for name, rows in relations.items()}

    def predict(rule: Body, fact) -> bool:
        return all(fact in sets[name] for name in rule)

    return predict


def kfold_scores(
    bodies: list[Body],
    relations: dict[str, list],
    facts: list,
    labels: list[bool],
    folds: int,
    seed: int,
    covers: dict[Body, set] | None = None,
    score: str = "accuracy",
) -> dict[Body, float]:
    """Held-out score of each body's fixed cover-membership prediction,
    k-fold, using the EXACT SAME fold-assignment convention `pyxlog.ilp.
    neural_credit.kfold_select` uses -- both call the shared
    `pyxlog.ilp.neural_credit.holdout_fold_assignment(n_facts, folds, seed)`
    (see `test_relational_search.py`'s pinning test, which asserts this
    module's own usage matches that shared function's output).

    Because a body's prediction is a FIXED set-membership test (see this
    module's own KEY INSIGHT), no training happens per fold: each fold's
    score is computed directly from the fixed cover, and the reported score
    is the mean across folds -- the same "sum of per-fold score, divided by
    fold count" reading `kfold_select` computes for a trained candidate,
    specialized to the case where every fold's "training" step is a no-op.

    `score` (default `"accuracy"`, byte-identical to this function's
    pre-`score`-parameter behavior): the per-fold metric being averaged.

    * `"accuracy"`: a fold's score is the fraction of that fold's held-out
      facts whose cover membership matches their label. On a task with a
      rare positive class and a large held-out set (e.g. 10 EC-initiation
      positives against ~23,000 rows), this sits on the all-false base-rate
      plateau and can rank a body that predicts nothing at all ABOVE the
      best real detector -- the accuracy metric does not distinguish "always
      right by predicting nothing" from "right for the right reason". This
      is not a hypothetical: it is exactly the failure this module's own
      task (the "recall-aware" `score="f1"` option below) exists to fix.
    * `"f1"`: a fold's score is `scorer.prf1`'s F1 of that fold's cover-
      membership prediction against that fold's held-out labels -- the SAME
      F1 arithmetic `run_caviar_theory.py`'s own scoring uses, reused (not
      re-derived) so the two never drift apart. F1 is RECALL-AWARE: a body
      that predicts nothing scores F1 0.0 on any fold with a real positive,
      instead of accuracy's near-1.0, so a real detector's few true
      positives can outrank the empty predictor even against a heavily
      imbalanced holdout.

      A fold with ZERO held-out positives makes every body's recall (and
      therefore its F1) UNDEFINED, not merely 0 -- `prf1` substitutes 0.0
      for an undefined ratio and flags `degenerate=True`, but averaging that
      substituted 0.0 in here would penalize EVERY body identically for a
      fact about the fold split, not about the body. Such a fold is
      therefore SKIPPED from the mean entirely -- the reported score is the
      mean F1 over folds that can actually measure recall (folds with at
      least one held-out positive), not over all `folds` folds. Tradeoff,
      stated honestly: a body's reported "f1" score can therefore average
      over FEWER than `folds` folds (and different bodies always see the
      same skip set, since it depends only on the fold/label split, not on
      any body's cover), so it is not literally "k-fold F1" for k=`folds`
      when some folds are unmeasurable -- it is the honest mean over the
      folds where F1 means something. The alternative (counting a
      zero-positive fold's F1 as 0.0) would be a body-INDEPENDENT constant
      rescale of every mean -- the skip set depends only on the fold/label
      split -- so the choice cannot change any RANKING (relative order is
      preserved). It CAN change a selection, because `select_body` gates
      on the ABSOLUTE `min_fit` threshold, not on rank: the uniform
      downward rescale can drag every mean below `min_fit` (turning a
      commit into an abstention) even though no body's rank moved. Skip
      is preferred both for that reason and because the reported number
      then means "mean F1 where F1 is defined" rather than a value
      diluted by unmeasurable folds. In the degenerate case where NO fold has any held-out
      positive at all (recall is unmeasurable everywhere), every body's
      score is reported as `0.0` (empty-mean is otherwise undefined) --
      this can only happen if the residual holds zero positives overall,
      which `theory_loop.induce_theory` does not call `select_once` for
      (see its own stop condition).

    `covers` (default `None`, recomputed from `relations` here): callers that
    already hold a per-body cover dict (e.g. `induce_relational_theory`,
    across sequential-covering iterations where the cover never changes,
    only the residual facts do) should pass it, to avoid re-deriving the
    same set-intersection on every iteration.
    """
    if score not in ("accuracy", "f1"):
        raise ValueError(
            f"score must be 'accuracy' or 'f1' (got {score!r})."
        )
    if not 2 <= folds <= len(facts):
        raise ValueError(
            f"folds={folds} with {len(facts)} facts: every fold needs at "
            "least one held-out fact and training needs at least one fold's "
            "worth of facts left over -- mirrors kfold_select's identical "
            "guard."
        )
    if len(labels) != len(facts):
        raise ValueError(
            f"labels has {len(labels)} entries for {len(facts)} facts -- a "
            "mismatched pair would silently score every body against "
            "misaligned labels."
        )
    if covers is None:
        covers = {body: body_cover(body, relations) for body in bodies}

    fold_of = holdout_fold_assignment(len(facts), folds, seed)
    fold_held_ids = [
        [i for i in range(len(facts)) if fold_of[i] == fold]
        for fold in range(folds)
    ]

    if score == "accuracy":
        sums = {body: 0.0 for body in bodies}
        for held_ids in fold_held_ids:
            n_held = len(held_ids)
            for body in bodies:
                cover = covers[body]
                correct = sum(
                    1 for i in held_ids
                    if (facts[i] in cover) == bool(labels[i])
                )
                sums[body] += correct / n_held
        return {body: sums[body] / folds for body in bodies}

    # score == "f1": see this function's own docstring for why a
    # zero-positive fold is skipped from the mean rather than counted as 0.
    measurable_folds = [
        held_ids for held_ids in fold_held_ids
        if any(bool(labels[i]) for i in held_ids)
    ]
    if not measurable_folds:
        return {body: 0.0 for body in bodies}

    sums = {body: 0.0 for body in bodies}
    for held_ids in measurable_folds:
        gold = [bool(labels[i]) for i in held_ids]
        for body in bodies:
            cover = covers[body]
            pred = [facts[i] in cover for i in held_ids]
            sums[body] += prf1(pred, gold)["f1"]
    return {body: sums[body] / len(measurable_folds) for body in bodies}


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """The F1 arithmetic `scorer.prf1` computes from a `pred`/`gold` pair,
    read straight off the raw tp/fp/fn counts instead -- byte-identical
    formula (same zero-division-to-0.0 substitutions), used where a
    permutation's per-fold counts are already in hand as set sizes and
    materializing a `pred`/`gold` list per body per fold per permutation
    would be the whole cost `permutation_null_threshold`'s precomputation
    exists to avoid."""
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile (the same convention `numpy.quantile`/
    `torch.quantile` default to) over an ALREADY-SORTED list -- kept as a
    tiny pure-Python function rather than a `torch.quantile` call so a
    hand-built test can recompute the exact same value with a pocket
    calculator, not merely re-derive it from another tensor op."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = q * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def permutation_null_threshold(
    bodies: list[Body],
    relations: dict[str, list],
    facts: list,
    labels: list[bool],
    folds: int,
    seed: int,
    n_permutations: int = 1000,
    quantile: float = 0.95,
    perm_seed: int = 7,
    covers: dict[Body, set] | None = None,
) -> dict:
    """A pre-registered, data-derived `min_fit` threshold for `select_body`'s
    fit gate: the ``quantile`` (default 0.95) of the POOL MAXIMUM per-fold-
    mean F1 seen across ``n_permutations`` (default 1000) label permutations
    -- "how good can the BEST of these bodies look, by chance alone, if the
    label/fact pairing carries no real signal". A body's real holdout F1
    clearing this threshold is therefore evidence it beats the pool's own
    chance ceiling, not merely a hand-picked constant like `select_body`'s
    0.75 default.

    THE PERMUTATION. One `torch.Generator` seeded with ``perm_seed``
    (independent of ``seed``, which -- exactly as in `kfold_scores` -- only
    ever determines the FOLD split) draws ``n_permutations`` SEQUENTIAL
    `torch.randperm(len(facts))` draws; permutation ``p``'s labels are
    ``labels[perm[i]]`` for each position ``i`` -- the label VECTOR is
    reshuffled across the SAME fact list, exactly the null `select_body`'s
    fit gate is meant to be judged against (a real rule's cover is a fact
    about the geometry/activity vocabulary, not about which label happened
    to land where).

    THE FOLD SPLIT IS PERMUTATION-INVARIANT, computed ONCE: `holdout_fold_
    assignment(len(facts), folds, seed)` never changes across permutations
    (a permutation reshuffles WHICH FACT carries which label, not which fold
    a fact was assigned to), so it -- and each body's cover, and each body's
    per-fold PREDICTED-positive fact set -- are derived exactly once, before
    the permutation loop starts, not re-derived per permutation.

    THE EFFICIENT ARITHMETIC (why this needs no per-permutation, per-body
    O(facts) pass). A permutation only moves WHICH fact positions count as
    positive -- it never changes a body's cover, and it never changes how
    many facts a fold holds out. So, per permutation, only the (typically
    tiny) set of positive positions needs to be recomputed: the inverse of
    the drawn permutation, gathered at the ORIGINAL positive positions,
    gives exactly ``{i : labels[perm[i]] is True}`` (see this function's own
    derivation below for why the inverse-gather is algebraically identical
    to reshuffling the whole label array and reading off the True
    positions). From there, per fold, per body: ``tp = |held-out predicted-
    positive positions (precomputed) INTERSECT this permutation's held-out
    positive positions|`` -- a set intersection over the SMALLER of the two
    sides, never an ``O(facts)`` rescan.

    THE ZERO-POSITIVE-FOLD SKIP RULE APPLIES PER PERMUTATION, exactly as
    `kfold_scores(score="f1")` applies it to the real (unpermuted) labels: a
    fold with no held-out positive UNDER THIS PERMUTATION contributes to no
    body's mean for THIS permutation (a different permutation can, and
    generally will, make a different fold set measurable). If NO fold is
    measurable under a permutation (only possible when the real label
    vector itself holds zero positives, since permutation never changes
    the total positive COUNT), every body's mean is undefined for that
    permutation and its pool-max sample is reported as ``0.0`` -- the same
    substitution `kfold_scores` makes for the analogous real-run case.

    Returns ``{"threshold": float, "pool_max_samples_summary": {"min",
    "median", "p95", "max"}, "n_permutations": int, "quantile": float,
    "perm_seed": int}``. ``"threshold"`` is `_quantile` at ``quantile``;
    ``"pool_max_samples_summary"``'s own ``"p95"`` is ALWAYS the literal
    95th percentile (a fixed diagnostic), independent of whatever
    ``quantile`` the caller passed -- so a caller who deliberately chose a
    non-default ``quantile`` can still see where their chosen threshold
    sits relative to the conventional 95th-percentile reading.
    """
    if not 2 <= folds <= len(facts):
        raise ValueError(
            f"folds={folds} with {len(facts)} facts: every fold needs at "
            "least one held-out fact -- mirrors kfold_scores's identical "
            "guard."
        )
    if len(labels) != len(facts):
        raise ValueError(
            f"labels has {len(labels)} entries for {len(facts)} facts -- a "
            "mismatched pair would silently permute against misaligned "
            "facts."
        )
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1 (got {n_permutations!r}).")
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile must be in (0.0, 1.0] (got {quantile!r}).")

    import torch

    if covers is None:
        covers = {body: body_cover(body, relations) for body in bodies}

    n = len(facts)
    fold_of = holdout_fold_assignment(n, folds, seed)
    fold_held_ids = [
        [i for i in range(n) if fold_of[i] == fold] for fold in range(folds)
    ]
    held_ids_set = [set(ids) for ids in fold_held_ids]

    # Per body, per fold: the FIXED (permutation-invariant) held-out
    # predicted-positive position set and its size -- computed once here,
    # never inside the permutation loop.
    pred_pos_set: dict[Body, list[set]] = {}
    pred_pos_count: dict[Body, list[int]] = {}
    for body in bodies:
        cover = covers[body]
        pred_pos_set[body] = [
            {i for i in held_ids if facts[i] in cover} for held_ids in fold_held_ids
        ]
        pred_pos_count[body] = [len(s) for s in pred_pos_set[body]]

    original_positive_idx = torch.tensor(
        [i for i, y in enumerate(labels) if bool(y)], dtype=torch.long,
    )

    rng = torch.Generator().manual_seed(perm_seed)
    pool_max_samples: list[float] = []
    for _ in range(n_permutations):
        perm = torch.randperm(n, generator=rng)
        # Inverse-gather: perm_inv[perm[i]] = i, so perm_inv[original
        # positive index j] is the position i with perm[i] == j, i.e. the
        # position whose PERMUTED label reads labels[j] -- exactly the set
        # {i : labels[perm[i]] is True} a literal reshuffle-then-scan would
        # produce, without ever materializing the full permuted label array.
        perm_inv = torch.empty(n, dtype=torch.long)
        perm_inv[perm] = torch.arange(n)
        positive_positions = set(perm_inv[original_positive_idx].tolist())

        held_pos_by_fold = []
        for held_set in held_ids_set:
            if len(positive_positions) <= len(held_set):
                hp = {i for i in positive_positions if i in held_set}
            else:
                hp = {i for i in held_set if i in positive_positions}
            held_pos_by_fold.append(hp)
        measurable = [f for f, hp in enumerate(held_pos_by_fold) if hp]

        if not measurable:
            pool_max_samples.append(0.0)
            continue

        pool_max = 0.0
        for body in bodies:
            total = 0.0
            for f in measurable:
                hp = held_pos_by_fold[f]
                pset = pred_pos_set[body][f]
                if len(hp) <= len(pset):
                    tp = sum(1 for i in hp if i in pset)
                else:
                    tp = sum(1 for i in pset if i in hp)
                fp = pred_pos_count[body][f] - tp
                fn = len(hp) - tp
                total += _f1_from_counts(tp, fp, fn)
            body_score = total / len(measurable)
            if body_score > pool_max:
                pool_max = body_score
        pool_max_samples.append(pool_max)

    sorted_samples = sorted(pool_max_samples)
    return {
        "threshold": _quantile(sorted_samples, quantile),
        "pool_max_samples_summary": {
            "min": sorted_samples[0],
            "median": _quantile(sorted_samples, 0.5),
            "p95": _quantile(sorted_samples, 0.95),
            "max": sorted_samples[-1],
        },
        "n_permutations": n_permutations,
        "quantile": quantile,
        "perm_seed": perm_seed,
    }


@dataclass(frozen=True)
class BodySelection:
    """What the relational-body arbiter is entitled to claim -- mirrors
    `pyxlog.ilp.neural_credit.HoldoutSelection`'s shape (`rule`, `tied`,
    `margin`, `top_weight`, `reason`), generalized from a `(left, right)`
    2-tuple to an n-literal body tuple. `theory_loop.induce_theory` reads
    only `.rule` (required) and `.margin` (via `getattr`, optional) -- both
    present here, so this type is a drop-in `select_once` return value."""

    rule: Body | None
    tied: list[Body]
    margin: float
    top_weight: float
    reason: str

    @property
    def decided(self) -> bool:
        return self.rule is not None


def select_body(
    scores: dict[Body, float],
    covers: dict[Body, set],
    min_fit: float,
    tie_tolerance: float,
) -> BodySelection:
    """Selection over holdout scores for an ALL-RELATIONAL body pool --
    replicates `pyxlog.ilp.neural_credit._select_from_holdout`'s fit gate and
    unique-winner-or-abstain tie semantics faithfully (both call the same
    `pyxlog.ilp.discovery.select_rule`, with `min_weight=min_fit` equally
    vacuous here for the same reason: everything below the fit gate was
    already dropped), but its OCCAM STEP IS A HONEST GENERALIZATION, not a
    reuse, because `_select_from_holdout`'s tie-break ("prefer the relational
    candidate over the neural one") has no meaning in a pool that is
    entirely relational already:

    * A tie within `tie_tolerance` is first grouped by each tied body's own
      fixed COVER (the exact fact set it predicts True on). If every tied
      body shares the IDENTICAL cover, they are indistinguishable on the
      TRAINING data (their covers could still diverge on unseen facts --
      the identity holds for this dataset, not universally), so Occam
      licenses keeping the SHORTEST body (fewer literals is a simpler
      explanation of the same predictions), breaking any remaining tie
      lexicographically. This is the same "equal generalization, simpler
      explanation wins" principle `_select_from_holdout` applies, just
      measured on "identical cover" instead of "relational vs. neural".
    * If the tied bodies split into two or more DIFFERENT covers, Occam has
      nothing to say -- these bodies genuinely disagree on some fact, and
      picking one over another would be an arbitrary vocabulary-order
      choice, exactly the failure `select_rule`'s own docstring exists to
      refuse. This abstains, naming how many distinct covers the tie
      contains.

    `covers` must contain an entry for every body in `scores` (and,
    correspondingly, every body `select_rule` might return as tied) --
    raises a plain `KeyError` otherwise, the same way a caller-side
    programming error surfaces anywhere else in this module.
    """
    if not (isinstance(tie_tolerance, (int, float)) and tie_tolerance > 0.0):
        raise ValueError(
            f"tie_tolerance must be a positive number (got "
            f"{tie_tolerance!r}); a non-positive tolerance would treat "
            "holdout quantization noise as evidence."
        )
    for body in scores:
        for name in body:
            if "&" in name:
                raise ValueError(
                    f"relation name {name!r} (in body {body!r}) contains "
                    "'&', the internal key separator select_body's own "
                    "select_rule call round-trips through; scoring it would "
                    "corrupt the key split, refused."
                )

    fit = {b: v for b, v in scores.items() if v >= min_fit}
    if not fit:
        return BodySelection(
            rule=None, tied=sorted(scores), margin=0.0,
            top_weight=max(scores.values(), default=0.0),
            reason=f"no body passed the fit gate (min_fit={min_fit}): a "
                   "body that cannot fit held-out data is not a rule",
        )

    keyed = {"&".join(b): v for b, v in fit.items()}
    # min_weight=min_fit is deliberately vacuous, mirroring
    # _select_from_holdout: everything below min_fit was already dropped above.
    sel = select_rule(keyed, min_weight=min_fit, tie_tolerance=tie_tolerance)
    if sel.rule is not None:
        winner = tuple(sel.rule.split("&"))
        return BodySelection(
            rule=winner, tied=[winner], margin=sel.margin,
            top_weight=sel.top_weight, reason=sel.reason,
        )

    tied = [tuple(t.split("&")) for t in sel.tied]
    groups: dict[frozenset, list[Body]] = {}
    for b in tied:
        groups.setdefault(frozenset(covers[b]), []).append(b)

    if len(groups) == 1:
        group = next(iter(groups.values()))
        winner = sorted(group, key=lambda b: (len(b), b))[0]
        return BodySelection(
            rule=winner, tied=tied, margin=sel.margin, top_weight=fit[winner],
            reason=(
                "holdout tie broken by Occam: every tied body covers an "
                "IDENTICAL fact set (indistinguishable on this training "
                f"data), so the shortest/lexicographically-first body "
                f"({'&'.join(winner)}) is kept -- generalizes "
                "_select_from_holdout's relational-over-neural tie-break "
                "(equal generalization, simpler explanation) to 'simpler "
                "description, provably equal cover'"
            ),
        )
    return BodySelection(
        rule=None, tied=tied, margin=sel.margin, top_weight=sel.top_weight,
        reason=(
            f"holdout tie among {len(tied)} bodies clustering into "
            f"{len(groups)} genuinely different covers: Occam licenses "
            "narrowing an identical-cover tie to its simplest description, "
            "not choosing among bodies that actually disagree on some fact"
        ),
    )


def induce_relational_theory(
    relations: dict[str, list],
    facts: list,
    is_positive: list[bool],
    *,
    max_literals: int = 3,
    folds: int = 4,
    seed: int = 0,
    min_fit: float = 0.75,
    tie_tolerance: float | None = None,
    max_clauses: int = 4,
    min_new_covered: int = 10,
    holdout_score: str = "accuracy",
) -> dict:
    """The CPU-only, up-to-`max_literals`-literal relational counterpart to
    `run_caviar_theory.py`'s `kfold_select`-backed relational search: wires
    `enumerate_bodies`/`kfold_scores`/`select_body` through
    `theory_loop.induce_theory` (reused UNCHANGED -- it is opaque to a
    clause's own shape) exactly the way `run_caviar_theory._run_relational_
    theory`/`_run_relational_ec` wire `kfold_select` through it: one
    `select_once` call per iteration, re-scoring the CURRENT residual;
    `predict_clause` is the SAME body-cover predicate used both for the
    loop's own bookkeeping (which residual positives a committed clause
    newly covers) and, by the caller, for final held-out scoring.

    `min_fit` defaults to 0.75, matching `kfold_select`'s own default -- the
    two searches are meant to be compared on equal footing. It is ONE
    constant, applied unchanged at EVERY iteration -- including iterations
    >= 2, whose scores are computed on the RESIDUAL facts (the positives
    already covered are removed). A caller that DERIVES `min_fit` from the
    full training labels (e.g. `permutation_null_threshold`) should know
    this is exact only for iteration 1; for later iterations the same
    threshold is a deliberate approximation -- the null distribution is not
    re-derived per residual (see the derivation sites' own docstrings).

    `holdout_score` (default `"accuracy"`, byte-identical to this function's
    pre-`holdout_score`-parameter behavior): forwarded, unchanged, to every
    `kfold_scores` call this function makes -- see that function's own
    docstring for `"accuracy"` vs `"f1"`'s semantics (and, in particular, why
    `"f1"` exists: on a rare-positive-class holdout, accuracy can rank the
    empty predictor above the best real detector).

    `tie_tolerance=None` (default) resolves, EVERY iteration, to
    `kfold_select`'s identical derivation (`max(0.01, 1 / len(residual_
    facts))`) against THAT iteration's own residual size -- not computed
    once up front against the full fact list -- exactly mirroring
    `kfold_select`'s own per-call behavior (it, too, derives this fresh from
    whatever `facts` it was just handed).

    The candidate pool (`enumerate_bodies` over the FULL `relations`) and
    every body's cover are computed ONCE, up front, and reused across every
    `select_once` call -- a body's cover does not depend on the residual, so
    recomputing it per iteration would be wasted work, not a different
    result.

    Returns `theory_loop.induce_theory`'s own three keys (`"clauses"`,
    `"iterations"`, `"stop_reason"`) UNCHANGED, plus:

    * `"pool"`: `{"bodies_by_size": {2: n, 3: n, ...}, "skipped_empty_
      cover": {...}}` -- the FULL enumerated pool's size, so a caller can
      report "pool sizes (2-lit + 3-lit)" without re-deriving it.
    * `"scores_per_iteration"`: one `dict[Body, float]` PER `select_once`
      call, in call order -- so a caller whose search abstained can still
      report the top-N scores it came closest with, without a second search.
    * `"selection_reasons_per_iteration"`: the underlying selection's own
      `reason` string per `select_once` call (e.g. the fit-gate rejection
      message) -- the loop's iteration record alone cannot distinguish a
      fit-gate abstain from a tie abstain.
    """
    if max_literals not in (2, 3):
        raise ValueError(
            f"max_literals must be 2 or 3 (got {max_literals!r}) -- see "
            "enumerate_bodies."
        )

    all_bodies, skip_counts = enumerate_bodies(relations, max_literals=max_literals)
    covers = {body: body_cover(body, relations) for body in all_bodies}
    predict_clause = make_predict_clause(relations)

    scores_per_iteration: list[dict[Body, float]] = []
    selection_reasons_per_iteration: list[str] = []

    def select_once(residual_facts, residual_is_positive):
        resolved_tt = (
            tie_tolerance if tie_tolerance is not None
            else max(0.01, 1.0 / len(residual_facts))
        )
        scores = kfold_scores(
            all_bodies, relations, residual_facts, residual_is_positive,
            folds, seed, covers=covers, score=holdout_score,
        )
        scores_per_iteration.append(scores)
        sel = select_body(scores, covers, min_fit, resolved_tt)
        # The select-level reason (fit-gate rejection vs tie vs win) is the
        # diagnostic a result reader actually needs on an abstain; the theory
        # loop's own iteration record only says that select_once abstained.
        selection_reasons_per_iteration.append(sel.reason)
        return sel

    theory = induce_theory(
        select_once, predict_clause, facts, is_positive,
        max_clauses=max_clauses, min_new_covered=min_new_covered,
    )

    return {
        **theory,
        "pool": {
            "bodies_by_size": {
                size: sum(1 for b in all_bodies if len(b) == size)
                for size in range(2, max_literals + 1)
            },
            "skipped_empty_cover": skip_counts,
        },
        "scores_per_iteration": scores_per_iteration,
        "selection_reasons_per_iteration": selection_reasons_per_iteration,
    }
