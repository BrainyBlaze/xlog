"""Neural JOIN bodies: a neural predicate on an EXISTENTIAL join variable.

    plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E).
                  \\____ neural on Ev ____/  \\__ ordinary relation __/

``Ev`` is not in the head, so it ranges over a domain; the head binding's truth is
the OR over that domain. This module owns three things and nothing else:

  1. reading that shape OUT OF THE RULE (the rule is the single source of truth --
     the join relation is never supplied by the caller). The shape is EXACTLY
     ``{one neural atom, one join relation}``; a body with any further conjunct is
     NOT this shape and is refused, never silently reduced to it;
  2. reading the join extension FROM THE ENGINE (:func:`read_join_extension`), by
     enumerating the relation's tuples (never from a Python side-channel: if the
     caller handed us the edge->events map, the OR would be Python's, not the
     logic's, and the whole claim would be hollow), and restating that extension --
     which comes back in RAW domain constants -- in ``domain_inputs`` ROW indices
     (:func:`translate_extension_to_rows`), against the caller's explicit
     ``domain_ids``. The STRUCTURE is still entirely the engine's; ``domain_ids``
     only says which row of the feature tensor holds which constant;
  3. the OR itself, in log space (a naive product underflows on a large domain),
     over a PRECOMPUTED index of that static extension so the training hot loop
     does no per-step host->device copies (:class:`JoinExtensionIndex`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_ATOM = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)")


@dataclass(frozen=True)
class JoinBody:
    neural_predicate: str
    network: str
    join_var: str
    relation: str
    event_arg: int   # position of the join var in the relation
    head_arg: int    # position of the head var in the relation


def _atoms(text: str) -> list[tuple[str, list[str]]]:
    """Every parenthesized atom MENTIONED anywhere in ``text`` (so it also sees the
    atom inside ``not p(X)`` or ``know p(X)``). Used for the crude routing question
    "does this body touch a neural predicate at all", NOT for shape recognition."""
    return [
        (m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()])
        for m in _ATOM.finditer(text)
    ]


def _bare_positive_atom(literal: str) -> tuple[str, list[str]] | None:
    """``p(A, B)`` -> ``("p", ["A", "B"])``; anything else -> ``None``.

    A body LITERAL is not necessarily an atom: xlog's ``BodyLiteral`` is also
    ``not p(X)``, ``X < 3``, ``Z is X + Y``, ``know p(X)``. Only a literal that is
    ENTIRELY one positive atom -- nothing before it, nothing after it -- is part of
    the join shape; ``fullmatch`` is what enforces that, and it is the whole reason
    the contract is checked against literals rather than against a count of
    parenthesized atoms (a count cannot see the ``not``, and dropping a ``not``
    trains the INVERSE rule).
    """
    m = _ATOM.fullmatch(literal.strip())
    if m is None:
        return None
    return m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()]


def _is_var(term: str) -> bool:
    """A variable, INCLUDING the anonymous wildcard.

    The grammar (grammar.pest: ``var_or_anon``) admits exactly two variable forms — an
    uppercase-first name and the bare ``_``. The wildcard must count as a variable
    HERE, in the routing predicate's alphabet: a body that puts the neural predicate
    on ``_`` is still an existential neural join as far as routing is concerned, and a
    router blind to ``_`` would wave it through to the plain relational path — trained
    as an always-true candidate, no gradient to the detector, no error. Whether the
    shape is then ACCEPTED is :func:`parse_join_body`'s decision, and it refuses ``_``
    (each ``_`` is a distinct variable, so no join exists to aggregate over).
    """
    return term[:1].isupper() or term == "_"


def mentions_neural_on_nonhead_var(
    body_literals: list[str], neural_predicates: dict[str, str], head_var: str
) -> bool:
    """Does this body put a declared nn/4 predicate on a variable OTHER than the head?

    The ROUTING question, deliberately cruder than :func:`parse_join_body`: it looks
    inside every literal (negated, compared, modal -- anything), so a body that is
    *about* an existential neural join but is not the supported SHAPE still answers
    True. That is the point. Such a candidate's relational truth is an OR over a join
    extension; treating it as a plain relational candidate -- e.g. because the engine
    happened to hand back a hard-filters-only eligibility mask for it -- would train an
    always-true rule with no gradient to the detector, silently. So the caller must
    either take it through the join path or reject it, never neither.
    """
    for literal in body_literals:
        for pred, args in _atoms(literal):
            if pred not in neural_predicates:
                continue
            if any(_is_var(a) and a != head_var for a in args):
                return True
    return False


def parse_join_body(
    body_literals: list[str], neural_predicates: dict[str, str], head_var: str
) -> JoinBody | None:
    """Recognize the Stage-B shape. Returns None for any other body -- this function
    never guesses.

    The shape is EXACTLY two BODY LITERALS, each a bare positive atom: the one neural
    atom and the one relation that joins the existential variable to the head, the
    relation carrying EXACTLY those two arguments and nothing else. The
    contract is checked against the literals the desugarer parsed, not against a
    regex count of parenthesized atoms -- an atom count cannot tell
    ``pre_before_post(Ev, E)`` from ``not pre_before_post(Ev, E)``, and cannot see a
    comparison literal (``Ev < 3``) at all, so it would have accepted both and trained
    the wrong rule (for the negation: the exact inverse of the written one).

    A longer body (an extra ``high_degree(E)`` conjunct), a negated literal, a
    comparison, an ``is`` expression or a modal literal is a DIFFERENT rule, and this
    module has no mask for it -- returning the two-literal mask anyway would silently
    drop the difference and train a rule nobody wrote. Out of scope must mean
    rejected, so anything but the exact shape returns None and the caller's typed
    rejection stands.
    """
    if len(body_literals) != 2:
        return None
    parsed = [_bare_positive_atom(lit) for lit in body_literals]
    if any(atom is None for atom in parsed):
        return None                          # a negation / comparison / is / modal
    atoms: list[tuple[str, list[str]]] = [atom for atom in parsed if atom is not None]

    neural = [(p, args) for p, args in atoms if p in neural_predicates]
    if len(neural) != 1:
        return None
    pred, nargs = neural[0]
    if len(nargs) != 2:
        # The join shape's neural atom is EXACTLY (join_var, label). A third argument
        # is a second existential this mask cannot express — `saliency(Ev, X, str)`
        # is a DIFFERENT rule, and dropping X would train the two-argument rule the
        # body merely resembles, silently. Same contract as the relation atom below.
        return None
    join_var = nargs[0]                      # nn(net, [Input], Label, ...) -> arg 0
    if join_var == head_var:
        return None                          # head-bound gate, not an existential join
    if not _is_var(join_var):
        return None                          # a constant, not a variable
    if join_var == "_":
        # Anonymous wildcards are DISTINCT variables per occurrence: `saliency(_, l),
        # rel(_, E)` shares nothing, so there is no join to OR over. Textual matching
        # below would wrongly see one; refuse instead.
        return None
    if _is_var(nargs[-1]):
        # The label slot must be a CONSTANT. `saliency(Ev, Lbl)` has no single output
        # column to train against; deferring to the engine's label_to_index would
        # surface the failure far from the rule that caused it.
        return None

    # exactly two atoms, exactly one of them neural -> exactly one relation
    p, args = next((q, a) for q, a in atoms if q not in neural_predicates)
    if len(args) != 2:
        # The join relation carries EXACTLY the join variable and the head variable.
        # `rel(Ev, E, W)` also "contains both", but its third argument is a second
        # existential: the relation's extension then holds one (event, edge) pair per
        # W, and `read_join_extension` — which buckets by the head argument and keeps
        # every tuple — would bucket the same event once per W and compute
        # `1 - (1 - p)^W` instead of `1 - (1 - p)`. There is no mask here for that
        # rule, so it is refused, not silently reduced to the two-argument one it
        # resembles. (A Rust check in the circuit path rejects the same shape; this
        # module states its own contract rather than borrowing that one's.)
        return None
    if join_var not in args or head_var not in args:
        return None
    return JoinBody(
        neural_predicate=pred,
        network=neural_predicates[pred],
        join_var=join_var,
        relation=p,
        event_arg=args.index(join_var),
        head_arg=args.index(head_var),
    )


def read_join_extension(
    ilp_program: Any, jb: JoinBody, num_bindings: int
) -> list[list[int]]:
    """head binding -> [event ids], read FROM THE ENGINE.

    The engine owns the relation; we only ask it to enumerate the tuples that
    hold, via ``CompiledIlpProgram.relation_facts(name) -> list[list[int]]``. This
    is deliberately NOT solved by taking an edge->events map from Python: that
    would move the aggregation structure out of the logic, and the claim that
    "the logic performs the OR" would be false -- it would be Python's OR over a
    caller-supplied hint, not the engine's own relation.

    This is O(|extension|) -- it enumerates only the tuples that actually hold,
    not O(num_bindings * domain_size) membership probes. A tuple whose head binding
    falls outside ``0 .. num_bindings - 1`` is a REFUSAL, not a skip: it is the same
    class of caller/world disagreement as a joined constant missing from
    ``domain_ids`` (loud there too), and silently dropping it would shrink the
    candidate's extension -- and therefore its OR -- without a trace. Bindings with
    no joined events get ``[]``. Events are sorted ascending within each bucket, and
    the traversal itself is a plain forward scan of ``relation_facts``, so the
    result is deterministic for a given compiled program.
    """
    buckets: list[list[int]] = [[] for _ in range(num_bindings)]
    for t in ilp_program.relation_facts(jb.relation):
        h = t[jb.head_arg]
        if not 0 <= h < num_bindings:
            raise ValueError(
                f"join relation '{jb.relation}' holds a tuple whose head binding is "
                f"{h}, outside the query range 0..{num_bindings - 1}. Head bindings "
                "must be the dense query indices; dropping the tuple would silently "
                "shrink this candidate's OR. Renumber the world or widen the query."
            )
        buckets[h].append(t[jb.event_arg])
    for bucket in buckets:
        bucket.sort()
    return buckets


def domain_row_index(domain_ids: Sequence[int], network: str = "") -> dict[int, int]:
    """``constant -> row`` for one network's ``domain_ids``.

    This map is the SINGLE source of truth for the correspondence, on both sides: the
    ids are also handed to the engine (``register_domain_tensor_source``), and the
    exact d-DNNF circuit resolves the constant it grounded a leaf at through the same
    list. Neither path infers the row from an ordering any more, so no id set — sparse,
    superset of the join domain, or dense — can put the two engines into disagreement.

    The ids may be given in ANY ORDER — a row is FOUND by the constant it holds, never
    counted off an ordering. Ascending order was once required, and that requirement was
    load-bearing for exactly one reason: it was what made the circuit's row-counting
    coincide with this map. The circuit no longer counts, so the reason is gone, and the
    rule goes with it rather than standing around to be re-justified by the next reader
    as "the circuit needs sorted ids" — which is the false belief that produced the bug.

    The ids must be DISTINCT. That is the part the ordering rule was really carrying:
    two rows claiming one constant leaves the row of that constant undefined, so it is
    refused by name rather than resolved by a tie-break the caller never asked for.
    """
    row_of: dict[int, int] = {}
    for row, c in enumerate(domain_ids):
        c = int(c)
        if c in row_of:
            where = f"domain_ids['{network}']" if network else "domain_ids"
            raise ValueError(
                f"{where} must not repeat a constant: {c} is claimed by row "
                f"{row_of[c]} and row {row}, so the row holding its features "
                f"would be ambiguous"
            )
        row_of[c] = row
    return row_of


def translate_extension_to_rows(
    extension: list[list[int]],
    domain_ids: Sequence[int],
    network: str = "",
    rule_id: str = "",
) -> list[list[int]]:
    """The join extension, restated in ``domain_inputs`` ROW indices.

    :func:`read_join_extension` returns the extension in RAW domain constants (they
    are what the engine's relation holds). The network's per-constant probabilities,
    however, are a tensor whose rows the CALLER laid out, and ``domain_ids`` is the
    caller's statement of which row holds which constant. This is the ONE place the
    two are reconciled; everything downstream (:func:`prepare_extension`, the OR)
    then speaks rows only, so no other code has to know the convention.

    A constant the engine's relation joins but ``domain_ids`` never mentions has no
    feature vector at all: it is named in a typed error rather than silently read off
    some other constant's row (or off the end of the tensor, which on CUDA is a
    device-side assert that poisons the whole process).
    """
    row_of = domain_row_index(domain_ids, network)
    out: list[list[int]] = []
    for events in extension:
        rows: list[int] = []
        for e in events:
            if e not in row_of:
                who = f"trainable_rule '{rule_id}': " if rule_id else ""
                where = f"domain_ids['{network}']" if network else "domain_ids"
                raise ValueError(
                    f"{who}the join extension contains domain constant {e}, which is "
                    f"not in {where} — so no row of domain_inputs"
                    f"{f'[{network!r}]' if network else ''} holds its feature vector. "
                    f"{where} lists {sorted(row_of)[:8]}"
                    f"{'...' if len(row_of) > 8 else ''}"
                )
            rows.append(row_of[e])
        out.append(rows)
    return out


@dataclass(frozen=True)
class JoinExtensionIndex:
    """The STATIC join extension, flattened once into device tensors.

    ``event_ids`` is every ``(binding, event)`` pair's event id, concatenated across
    bindings; ``binding_ids`` is the same length and says which head binding each
    entry belongs to (a segment id). Both live on the device from construction, so a
    training step's OR is one gather + one segmented sum -- no host->device copy, no
    per-binding Python loop, no stack of n scalars. The extension never changes
    across steps (that is the whole premise of reading it from the engine once), so
    this is built ONCE, outside the hot loop.
    """

    event_ids: Any       # [total] long, on device
    binding_ids: Any     # [total] long, on device (segment id per entry)
    num_bindings: int


def prepare_extension(
    extension: list[list[int]], device: Any, num_rows: int | None = None
) -> JoinExtensionIndex:
    """Flatten ``head binding -> [event ids]`` into the device-resident index.

    ``num_rows``, when given, is the row count of the tensor the index will gather
    from, and every entry is checked against it HERE, once, on the host. The two
    failure directions are asymmetric and both real: an id >= num_rows dies later as
    a CUDA device-side assert (which poisons the context), while an UNTRANSLATED
    extension whose raw constants happen to fall < num_rows gathers the wrong rows
    and computes a silently wrong OR. A one-time bounds check at build time closes
    both without touching the hot loop.
    """
    import torch

    event_ids: list[int] = []
    binding_ids: list[int] = []
    for h, events in enumerate(extension):
        event_ids.extend(events)
        binding_ids.extend([h] * len(events))
    if num_rows is not None and event_ids:
        lo, hi = min(event_ids), max(event_ids)
        if lo < 0 or hi >= num_rows:
            raise ValueError(
                f"join extension holds row index {lo if lo < 0 else hi}, outside the "
                f"feature tensor's 0..{num_rows - 1}. The extension must be "
                "TRANSLATED to rows (translate_extension_to_rows) before it is "
                "prepared; raw domain constants only coincide with rows on the "
                "dense layout."
            )
    return JoinExtensionIndex(
        event_ids=torch.as_tensor(event_ids, device=device, dtype=torch.long),
        binding_ids=torch.as_tensor(binding_ids, device=device, dtype=torch.long),
        num_bindings=len(extension),
    )


def noisy_or_from_index(p: Any, index: JoinExtensionIndex) -> Any:
    """1 - PROD_{e in ext(h)} (1 - p_e), in log space, over a prepared index.

    The ONE implementation of the math. Vectorized: gather log(1 - p) at every
    (binding, event) entry, segmented-sum it into a [num_bindings] accumulator, and
    exponentiate. A binding with an empty extension contributes no entry, so its
    accumulator stays 0 and it yields exactly ``1 - exp(0) = 0`` -- an OR over
    nothing is false, which is also what the provenance circuit does for an edge with
    no joined events.
    """
    import torch

    eps = 1e-7
    logq = torch.log1p(-p.clamp(eps, 1.0 - eps))     # log(1 - p_e)
    acc = torch.zeros(index.num_bindings, device=logq.device, dtype=logq.dtype)
    acc = acc.index_add(0, index.binding_ids, logq[index.event_ids])
    return 1.0 - torch.exp(acc)


def noisy_or_over_extension(p: Any, extension: list[list[int]], device: Any) -> Any:
    """1 - PROD_{e in ext(h)} (1 - p_e), for a raw (unprepared) extension.

    NOT ON THE HOT PATH, and deliberately: it has NO production callers. The trainer
    builds the index ONCE with :func:`prepare_extension` (outside the step loop) and
    calls :func:`noisy_or_from_index` per step. This wrapper exists for the CPU tests,
    which want to state the OR over a plain ``list[list[int]]``; it builds the index
    and delegates, so there is still exactly one implementation of the math.
    """
    return noisy_or_from_index(p, prepare_extension(extension, device))
