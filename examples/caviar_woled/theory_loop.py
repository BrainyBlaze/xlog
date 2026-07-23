"""Multi-clause induction via sequential covering.

WHY. A SINGLE star clause is capped by composition alone: on CAVIAR fold1,
``both_inactive`` covers 77.2% of TRAIN
positives but only 21.5% of TEST positives, so any single-clause rule that
picks the "obviously best" train body is structurally recall-capped on test
(F1 <= 0.354 at P=1). A two-clause THEORY -- ``(both_inactive ^ close) v
(both_active ^ close)`` -- lifts test F1 to 0.921 by covering the OTHER
composition mode with a second clause. This module is the engine-agnostic
control logic for building such a theory by SEQUENTIAL COVERING: search,
commit a clause, remove what it already explains, search again on what is
left, until a stop condition fires.

This module is PURE Python: no `torch`, no `pyxlog`, no engine import at all.
Everything that touches the engine (`kfold_select`, a trained `close_nn` net,
`train_engine_mode`) is behind two caller-provided closures --
``select_once`` and ``predict_clause`` -- so this file is CPU-testable with
plain fakes (see `python/tests/test_theory_loop.py`) and reusable by BOTH the
relational-vocabulary mode and the neural-vocabulary mode of
`run_caviar_theory.py`.

SEQUENTIAL-COVERING SEMANTICS (the choice this module makes, spelled out):

* After a clause is COMMITTED, every fact that is (a) a residual POSITIVE
  AND (b) predicted True by that clause is REMOVED from the residual. This
  is the "covered, explained, move on" reading of sequential covering: the
  next clause's search never gets credit (or blame) for a positive the
  theory already explains.
* NEGATIVES ARE NEVER REMOVED, on purpose, for every clause, whether or not
  the clause covers them. Removing a negative that a clause "correctly"
  predicts False would only ever shrink the residual to positives, which
  turns every subsequent `select_once` call into a search for a rule that
  merely OVERLAPS the remaining positives, with no downside for firing on
  the unrelated negatives -- precision would stop being a real constraint
  for clause 2 onward. Keeping every negative in every fold of the loop
  means each new clause is still penalized (by whatever holdout arbiter
  `select_once` wraps) for false positives against the FULL negative set,
  exactly as clause 1 was.
* A negative a clause happens to predict True (a false positive) is NOT
  removed either, for the same reason: it stays available to penalize the
  NEXT clause too, if the next clause also fires on it.
"""

from __future__ import annotations


def induce_theory(
    select_once,
    predict_clause,
    facts,
    is_positive,
    *,
    max_clauses: int = 4,
    min_new_covered: int = 10,
) -> dict:
    """Build a multi-clause theory over ``facts``/``is_positive`` by
    sequential covering.

    ``select_once(residual_facts, residual_is_positive) -> selection``: a
    caller-provided closure wrapping one holdout-search call (e.g.
    `pyxlog.ilp.neural_credit.kfold_select`). ``selection.rule`` is the
    candidate the search landed on, or ``None`` for an honest abstain
    (`HoldoutSelection`'s own convention -- this module does not require
    anything beyond a ``.rule`` attribute, though real callers will also
    have ``.margin`` etc., read here only if present via `getattr`).

    ``predict_clause(rule, fact) -> bool``: a caller-provided closure
    reading one committed (or just-proposed) clause's prediction for one
    fact -- a plain-Python set-intersection for a relational rule, a
    trained network's gated score for a rule ending in a neural relation.

    ``facts``/``is_positive``: the FULL (train) fact list and its labels,
    aligned one-to-one, exactly as `kfold_select` takes them. Never mutated
    in place -- the residual is built from copies.

    STOPS (checked in this order, every iteration), each a distinct,
    honestly-named ``stop_reason``:

    1. ``max_clauses`` already committed -- checked BEFORE calling
       ``select_once`` again, so reaching the cap never spends one more
       search call than necessary.
    2. No positives remain in the residual -- also checked before calling
       ``select_once``: searching for a rule to explain zero remaining
       positives is not a real search.
    3. ``selection.rule is None`` -- ``select_once`` itself abstained (its
       own holdout arbiter found nothing that fit/generalized). Honored
       immediately, no second-guessing.
    4. The proposed clause's NEWLY covered residual positives (facts that
       are residual-positive AND ``predict_clause(rule, fact)`` is True)
       number FEWER than ``min_new_covered``. In that case this
       clause is REJECTED, not committed -- a clause that barely nudges
       coverage is more likely fold noise than a genuine second mode in the
       data, and letting it in would silently degrade the "each clause is a
       load-bearing coverage jump" reading a theory's clause list is
       supposed to have. The loop STOPS here rather than continuing to a
       5th, 6th, ... attempt: if the current residual can no longer produce
       a clause worth committing, later residuals (which only ever shrink
       further from here on an accepted commit) are not expected to do
       better either, and a bottomless retry loop with no `max_clauses`-like
       cap would have no principled endpoint.

    Returns ``{"clauses": [...], "iterations": [...], "stop_reason": str}``:

    * ``"clauses"``: the committed rules, in commit order (``select_once``'s
      own rule objects, untouched).
    * ``"iterations"``: one entry PER ``select_once`` CALL (not per stop
      check -- checks 1-2 above never call ``select_once`` and so never add
      an entry), each ``{"rule", "reason", "margin", "n_residual_pos_before",
      "n_newly_covered"}``. ``"reason"`` is one of ``"committed"``,
      ``"rejected: newly covered positives below min_new_covered"``, or
      ``"select_once abstained"``. ``"margin"`` is ``getattr(selection,
      "margin", None)`` -- present whenever the real `HoldoutSelection` is
      behind ``select_once``, ``None`` for a bare fake that has no such
      attribute. ``"n_newly_covered"`` is ``0`` on an abstain (no clause was
      ever proposed to cover anything).
    * ``"stop_reason"``: the human-readable reason the loop ended, one of
      ``"max_clauses reached"``, ``"no positives remain in the residual"``,
      ``"select_once abstained"``, ``"insufficient new coverage"``.
    """
    residual_facts = list(facts)
    residual_pos = list(is_positive)
    clauses: list = []
    iterations: list[dict] = []
    stop_reason: str

    while True:
        if len(clauses) >= max_clauses:
            stop_reason = "max_clauses reached"
            break

        n_residual_pos_before = sum(1 for p in residual_pos if p)
        if n_residual_pos_before == 0:
            stop_reason = "no positives remain in the residual"
            break

        selection = select_once(residual_facts, residual_pos)
        margin = getattr(selection, "margin", None)

        if selection.rule is None:
            iterations.append({
                "rule": None,
                "reason": "select_once abstained",
                "margin": margin,
                "n_residual_pos_before": n_residual_pos_before,
                "n_newly_covered": 0,
            })
            stop_reason = "select_once abstained"
            break

        rule = selection.rule
        covered = [predict_clause(rule, f) for f in residual_facts]
        n_newly_covered = sum(
            1 for c, p in zip(covered, residual_pos) if c and p
        )

        if n_newly_covered < min_new_covered:
            iterations.append({
                "rule": rule,
                "reason": "rejected: newly covered positives below min_new_covered",
                "margin": margin,
                "n_residual_pos_before": n_residual_pos_before,
                "n_newly_covered": n_newly_covered,
            })
            stop_reason = "insufficient new coverage"
            break

        clauses.append(rule)
        iterations.append({
            "rule": rule,
            "reason": "committed",
            "margin": margin,
            "n_residual_pos_before": n_residual_pos_before,
            "n_newly_covered": n_newly_covered,
        })

        # Sequential covering: drop covered POSITIVES only -- see the module
        # docstring for why negatives are never removed.
        new_facts = []
        new_pos = []
        for f, c, p in zip(residual_facts, covered, residual_pos):
            if c and p:
                continue
            new_facts.append(f)
            new_pos.append(p)
        residual_facts, residual_pos = new_facts, new_pos

    return {
        "clauses": clauses,
        "iterations": iterations,
        "stop_reason": stop_reason,
    }
