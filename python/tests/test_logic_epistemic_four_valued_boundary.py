"""Verification that the pyxlog PUBLIC surface (LogicProgram.compile
-> evaluate -> LogicQueryResult) exposes the epistemic four-valued disposition
OUTWARD as distinct know/possible/not-know/not-possible head relations, and that a
contested (multi-world) disposition cannot be laundered into a known head.

Boundary contract (verified by sweep + these runtime tests, not a new capability):
- epistemic operators are BODY literals (`h(X) :- ..., know p(X).`), NOT query atoms.
  The four-valued status is read from the DISTINCT wrapper-head relations the engine
  emits -- one LogicQueryResult per epistemic head predicate (membership read), plus
  `is_true` for nullary/ground heads.
- Belnap fence at the pyxlog `know`/`possible` boundary: the engine evaluates only
  STRATIFIED single-reconciled-world epistemic programs. A construct that would create
  a genuine multi-world / contested disposition (cross-component epistemic coupling)
  FAILS CLOSED with a typed error -- it is never silently collapsed into a known head.
  (The Belnap `both -> not_known_true` reconciliation itself lives one layer up, in the
  consumer's belnap_epistemic modal-support rules; this test covers the xlog surface
  role: distinct-head exposure + fail-closed on unsupported contested constructs.)

Epistemic e2e eval is GPU-only (no CPU fallback); CUDA-gated like the other GPU
logic-eval tests.
"""

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


def _heads(result):
    """Map each emitted query result to {relation_name: set(row tuples)}.

    Query order is component-ordered, not source-ordered, so index by name.
    """
    out = {}
    for q in result.queries:
        cols = [torch.from_dlpack(t).cpu().tolist() for t in q.tensors]
        rows = set(zip(*cols)) if cols and cols[0] else set()
        out[q.relation_name] = (q.num_rows, rows)
    return out


# --- Test A: all four modal channels exposed as DISTINCT head relations ------------

SOURCE_FOUR_CHANNELS = """
pred pair(u32, u32).
pred edge(u32, u32).
pred alt(u32, u32).
pred blocked(u32, u32).
pred seen(u32, u32).
pred known_edge(u32, u32).
pred possible_alt(u32, u32).
pred clear_pair(u32, u32).
pred unknown_pair(u32, u32).
pair(1, 2). pair(2, 3). pair(3, 4).
edge(1, 2). edge(3, 4).
alt(2, 3).
blocked(3, 4).
seen(1, 2).
known_edge(X, Y) :- pair(X, Y), know edge(X, Y).
possible_alt(X, Y) :- pair(X, Y), possible alt(X, Y).
clear_pair(X, Y) :- pair(X, Y), not possible blocked(X, Y).
unknown_pair(X, Y) :- pair(X, Y), not know seen(X, Y).
?- known_edge(X, Y).
?- possible_alt(X, Y).
?- clear_pair(X, Y).
?- unknown_pair(X, Y).
"""


def test_four_modal_channels_exposed_as_distinct_heads():
    """know / possible / not-possible / not-know each surface as their OWN
    LogicQueryResult with independent membership -- the four-valued disposition is
    observable outward, never collapsed into a single signal."""
    program = pyxlog.LogicProgram.compile(SOURCE_FOUR_CHANNELS, device=0, memory_mb=512)
    heads = _heads(program.evaluate())

    # Four distinct, separately-queryable epistemic head relations.
    assert set(heads) == {"known_edge", "possible_alt", "clear_pair", "unknown_pair"}

    # know edge: pairs whose edge holds in the reconciled world.
    assert heads["known_edge"] == (2, {(1, 2), (3, 4)})
    # possible alt: pairs whose alt is supported.
    assert heads["possible_alt"] == (1, {(2, 3)})
    # not possible blocked: pairs minus the blocked pair (3,4).
    assert heads["clear_pair"] == (2, {(1, 2), (2, 3)})
    # not know seen: pairs minus the known-seen pair (1,2).
    assert heads["unknown_pair"] == (2, {(2, 3), (3, 4)})


# --- Test A': nullary ground heads expose the boolean status via is_true -----------

SOURCE_NULLARY = """
pred p().
pred known_p().
pred poss_p().
p().
known_p() :- know p().
poss_p() :- possible p().
?- known_p().
?- poss_p().
"""


def test_nullary_ground_epistemic_heads_expose_is_true():
    """For a ground (nullary) wrapper head, the modal status is the head's `is_true`
    -- the per-claim boolean read a consumer uses for a specific atom."""
    program = pyxlog.LogicProgram.compile(SOURCE_NULLARY, device=0, memory_mb=512)
    by_name = {q.relation_name: q for q in program.evaluate().queries}

    assert by_name["known_p"].is_true is True  # p asserted -> known_true
    assert by_name["poss_p"].is_true is True   # p asserted -> possible_true


# --- Test B: a contested (multi-world) construct FAILS CLOSED, never laundered ------

SOURCE_CONTESTED_COUPLING = """
pred node(u32).
pred a(u32).
pred b(u32).
pred poss_a(u32).
pred known_a(u32).
node(1).
a(X) :- node(X), not know b(X).
b(X) :- node(X), not know a(X).
poss_a(X) :- node(X), possible a(X).
known_a(X) :- node(X), know a(X).
?- poss_a(X).
?- known_a(X).
"""


def test_contested_multiworld_construct_fails_closed():
    """An even epistemic negation loop (a :- not know b; b :- not know a) is the
    autoepistemic construct that would make `a` possible-but-not-known (contested /
    Belnap both). The engine REJECTS it fail-closed with a typed error rather than
    silently producing a laundered `known_a` row -- the Belnap fence at the pyxlog
    boundary holds by construction: a contested disposition is never admitted as known."""
    with pytest.raises(RuntimeError, match="Unsupported epistemic construct"):
        program = pyxlog.LogicProgram.compile(
            SOURCE_CONTESTED_COUPLING, device=0, memory_mb=512
        )
        program.evaluate()
