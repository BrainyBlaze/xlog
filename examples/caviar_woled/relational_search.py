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

import torch

from pyxlog.ilp.discovery import select_rule
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
) -> dict[Body, float]:
    """Held-out accuracy of each body's fixed cover-membership prediction,
    k-fold, using the EXACT SAME fold-assignment convention `pyxlog.ilp.
    neural_credit.kfold_select` uses (`torch.Generator().manual_seed(seed)`,
    `torch.randperm(len(facts), generator=rng).tolist()`, `i % folds`) --
    reproduced here rather than imported, since `kfold_select`'s own fold
    assignment is inlined in its body, not a separate importable helper (see
    `test_relational_search.py`'s cross-check, which calls both derivations
    on the same tiny input and asserts they agree).

    Because a body's prediction is a FIXED set-membership test (see this
    module's own KEY INSIGHT), no training happens per fold: a fold's score
    is simply the fraction of that fold's held-out facts whose cover
    membership matches their label, and the reported score is the mean of
    that fraction across all `folds` folds -- the same "sum of per-fold
    accuracy, divided by fold count" reading `kfold_select` computes for a
    trained candidate, specialized to the case where every fold's "training"
    step is a no-op.

    `covers` (default `None`, recomputed from `relations` here): callers that
    already hold a per-body cover dict (e.g. `induce_relational_theory`,
    across sequential-covering iterations where the cover never changes,
    only the residual facts do) should pass it, to avoid re-deriving the
    same set-intersection on every iteration.
    """
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

    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(facts), generator=rng).tolist()
    fold_of = {f_idx: i % folds for i, f_idx in enumerate(order)}

    sums = {body: 0.0 for body in bodies}
    for fold in range(folds):
        held_ids = [i for i in range(len(facts)) if fold_of[i] == fold]
        n_held = len(held_ids)
        for body in bodies:
            cover = covers[body]
            correct = sum(
                1 for i in held_ids
                if (facts[i] in cover) == bool(labels[i])
            )
            sums[body] += correct / n_held

    return {body: sums[body] / folds for body in bodies}


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
      body shares the IDENTICAL cover, they are provably indistinguishable
      on this data -- no future fact could ever separate them -- so Occam
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
                "IDENTICAL fact set (provably indistinguishable on this "
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
    two searches are meant to be compared on equal footing.

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

    def select_once(residual_facts, residual_is_positive):
        resolved_tt = (
            tie_tolerance if tie_tolerance is not None
            else max(0.01, 1.0 / len(residual_facts))
        )
        scores = kfold_scores(
            all_bodies, relations, residual_facts, residual_is_positive,
            folds, seed, covers=covers,
        )
        scores_per_iteration.append(scores)
        return select_body(scores, covers, min_fit, resolved_tt)

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
    }
