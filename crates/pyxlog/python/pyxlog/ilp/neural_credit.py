"""Engine-mode training: a real-valued credit over the ENGINE's own candidate space.

The dILP enumerator (`valid_candidates`) already holds a neural-bodied existential
candidate and derives with it (proven by the bridge spike); what the engine's credit
kernel cannot do is carry a per-event probability -- its CSR is binary. This module
re-implements the credit NLL as a torch graph so the neural column enters as a
noisy-OR over the ENGINE-read join extension, and -- separately -- selects the rule
by K-FOLD HOLDOUT with a fit gate, never by the training weight: training credit
cannot distinguish a crisp-but-coincidental rule from a soft-but-correct one even in
principle; generalization can.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pyxlog.ilp.join_bodies import (
    JoinExtensionIndex,
    noisy_or_from_index,
    prepare_extension,
)


@dataclass(frozen=True)
class NeuralRelationSpec:
    """Registration metadata for one neural relation (typed registry).

    ``num_rows`` is the feature-tensor row count -- the witness domain size the
    dense-identity law checks against. ``arity`` is the DECLARED relation arity:
    the engine-mode credit scores the chain template's binary ``R(Z, Y)`` only,
    so anything but 2 is refused at enumeration -- the pool is validated against
    this registry, never trusted by name convention. ``arg_sorts`` optionally
    names the argument domains (length must equal ``arity``); IN THIS PHASE
    nothing here reads the sorts -- like ``artifact_hash`` they are carried for
    consumer-side validation only, and slot sort-matching at enumeration is
    future surface, not a check you can rely on today. ``artifact_hash`` is an
    opaque content hash of the detector artifact, carried for consumer-side
    validation and never interpreted here.

    A plain ``int`` in ``neural_relations`` remains valid shorthand for
    ``NeuralRelationSpec(num_rows=that_int)``.
    """

    num_rows: int
    arity: int = 2
    arg_sorts: tuple | None = None
    artifact_hash: str | None = None

    def __post_init__(self) -> None:
        if self.num_rows <= 0:
            raise ValueError(
                f"num_rows={self.num_rows}: a neural relation needs at least "
                "one feature row."
            )
        if self.arg_sorts is not None and len(self.arg_sorts) != self.arity:
            raise ValueError(
                f"arg_sorts has {len(self.arg_sorts)} entries for declared "
                f"arity {self.arity}; the sorts name the arguments, so the "
                "lengths must match."
            )


def _registry(neural_relations):
    """Normalize ``{name: int | NeuralRelationSpec}`` to specs and validate.

    The int shorthand stays byte-compatible with the pre-registry surface.
    A declared arity other than 2 is refused HERE, for every entry -- a typed
    registry the candidate pool is checked against, not a name convention."""
    reg = {}
    for name, value in neural_relations.items():
        if isinstance(value, bool) or not isinstance(
            value, (int, NeuralRelationSpec)
        ):
            raise ValueError(
                f"neural relation '{name}': registry value must be an int "
                "(the num_rows shorthand) or a NeuralRelationSpec, got "
                f"{type(value).__name__}. A bool is refused explicitly -- "
                "isinstance(True, int) holds in Python, and num_rows=True "
                "silently means one feature row."
            )
        spec = (NeuralRelationSpec(num_rows=value)
                if isinstance(value, int) else value)
        if spec.arity != 2:
            raise ValueError(
                f"neural relation '{name}' declares arity {spec.arity}; the "
                "engine-mode credit scores the chain template's binary R(Z, Y) "
                "only, so a non-binary declaration is refused at the registry."
            )
        reg[name] = spec
    return reg


@dataclass(frozen=True)
class CandidateSpec:
    """One engine-enumerated candidate, ready for the torch credit.

    Exactly one of ``witness_index`` / ``binary_cover`` is set: a neural candidate
    scores each fact as a noisy-OR over its witnesses, a relational candidate is a
    fixed {0,1} cover. Anything else is a construction bug, checked at init.
    """

    cid: int
    left: str
    right: str
    is_neural: bool
    witness_index: JoinExtensionIndex | None
    binary_cover: Any | None
    masked_any: Any | None = None

    def __post_init__(self) -> None:
        if self.is_neural != (self.witness_index is not None) or (
            self.is_neural == (self.binary_cover is not None)
        ):
            raise ValueError(
                f"candidate {self.cid} ({self.left},{self.right}): a neural candidate "
                "carries witness_index and no binary_cover; a relational one the "
                "opposite. Mixed or missing is a construction bug, refused."
            )


def credit_nll(cand_probs, specs, p_event, is_positive, gamma: float = 1.0):
    """``credit[f] = sum_c p_c * s_c(f)``, NLL over facts. A torch graph end to end,
    so the gradient reaches BOTH the candidate logits and the network behind
    ``p_event``. ``p_event`` is whatever vector the witness indices point into --
    the trainer passes the network output flattened per (event, label) row, so a
    witness for fact (h, y) reads the probability AT THE FACT'S OWN LABEL.
    ``gamma`` sharpens only the neural score (calibration against gradient
    starvation next to crisp {0,1} covers; it never decides truth -- holdout
    does)."""
    import torch

    if not specs:
        raise ValueError(
            "credit over no candidates is undefined: the spec list is empty. "
            "enumerate_specs refuses an empty pool with per-filter counts; if you "
            "built the specs yourself, at least one is required."
        )
    credit = None
    for spec in specs:
        if spec.is_neural:
            s = noisy_or_from_index(p_event, spec.witness_index)
            if gamma != 1.0:
                s = s.clamp(1e-7, 1.0) ** gamma
        else:
            s = spec.binary_cover
        term = cand_probs[spec.cid] * s
        credit = term if credit is None else credit + term
    credit = credit.clamp(1e-8, 1 - 1e-8)
    pos = is_positive.to(credit.dtype)
    loss = -(pos * torch.log(credit) + (1 - pos) * torch.log(1 - credit))
    return loss.mean()


def enumerate_specs(prog, mask_name, facts, neural_relations, device, n_labels,
                    witness_mask=None):
    """One CandidateSpec per engine triple over the program's binary EDB relations.

    Witnesses come from the ENGINE (`relation_facts`), never from the caller: for a
    fact (h, y) and candidate (L, R) the witness set is {z : L(h, z)} scored by the
    network AT THE FACT'S OWN LABEL y for a neural R -- each witness is stored as
    the flat (event, label) row ``z * n_labels + y``, so the credit gathers from
    the network output flattened row-major and no positive column is ever guessed
    (a y outside ``0..n_labels-1`` is refused here, typed) -- and the binary cover
    is [exists z: L(h,z) and R(z,y)] for a relational R. A neural relation in the
    LEFT slot has no witness semantics in this credit and is SKIPPED — filtering an
    auto-enumerated pool is not the same as silently altering a user-declared rule;
    the engine's cross-product enumeration always contains such triples. The same
    pool always also contains the dILP TEMPLATE's own learnable placeholders (e.g.
    `bL`/`bR`) and any other tuple-less name `valid_candidates` cross-products in:
    these have no ground extension to read at all (`relation_facts` raises
    `ValueError` for them), so they are pool-filtered for the same reason as the
    `__xlog_` meta relations — this is a targeted skip of a known-unreadable slot,
    not a blanket swallow of engine errors. The same cross product also contains
    relations of every ARITY; only binary rows have (h, z)/(z, y) semantics here,
    so non-binary relations are pool-filtered too (counted, never a bare
    IndexError or a silent first-two-columns projection). A pool these filters
    empty out is refused with per-filter counts: silent caps are exactly what
    this module promises not to have.

    Raw engine constants index the caller's feature rows, so the mixture path's
    ``domain_ids`` law applies verbatim: with no explicit constant->row map, the
    identity is only unambiguous when each neural relation's witness domain (the
    union of its left partners' joined constants) is EXACTLY ``0..num_rows-1``.
    Anything else could gather other events' probabilities while staying in
    bounds — silently — and is refused, not guessed at (mirrors
    ``neurosymbolic._resolve_domain_ids``).

    ``witness_mask`` is the witness-level mask/abstain channel (contract
    #155): a bool tensor ``[num_rows, n_labels]`` (True = MASKED) or ``None``
    (= everything ACTIVE, byte-identical to omitting the argument). A MASKED
    (event, label) row is REMOVED from the witness index at build time -- it
    contributes exactly zero credit and zero gradient, never a coerced
    ``False`` cover -- and each fact with at least one masked witness is
    flagged in the neural spec's ``masked_any``. The mask is validated
    against EVERY registered neural relation's ``(num_rows, n_labels)`` up
    front, not lazily per-candidate."""
    import torch

    neural_relations = _registry(neural_relations)
    if witness_mask is not None:
        rows_by_relation = {rn: spec.num_rows for rn, spec in neural_relations.items()}
        distinct_rows = set(rows_by_relation.values())
        if len(distinct_rows) > 1:
            detail = ", ".join(f"'{rn}'={rows}" for rn, rows in sorted(rows_by_relation.items()))
            raise ValueError(
                f"witness_mask was supplied, but the registered neural "
                f"relations disagree on num_rows ({detail}): a single "
                "witness_mask tensor cannot be interpreted against two "
                "different row spaces at once, refused."
            )
        for rn, spec in neural_relations.items():
            expected_shape = (spec.num_rows, n_labels)
            if tuple(witness_mask.shape) != expected_shape:
                raise ValueError(
                    f"witness_mask has shape {tuple(witness_mask.shape)}, but "
                    f"neural relation '{rn}' expects shape {expected_shape} "
                    "((num_rows, n_labels)) -- refused, not guessed at."
                )
    for h, y in facts:
        if not (0 <= y < n_labels):
            raise ValueError(
                f"fact ({h}, {y}) carries label {y}, but the network has "
                f"{n_labels} output column(s) (0..{n_labels - 1}). The neural "
                "score reads the network at the fact's own label; a label with "
                "no column is a contract violation, refused."
            )

    facts_cache: dict[str, Any] = {}

    def _readable(name):
        """True iff `name` has a ground extension the engine can read, caching the
        rows on success (so `_left`/`_pairs` never call the engine twice) and
        `False` on `ValueError` (ONLY `ValueError` — anything else is a real bug
        and propagates)."""
        if name not in facts_cache:
            try:
                facts_cache[name] = prog.relation_facts(name)
            except ValueError:
                facts_cache[name] = False
        return facts_cache[name] is not False

    binary_cache: dict[str, bool] = {}

    def _binary(name):
        """True iff every row of `name` has exactly two columns. The engine's
        cross product carries relations of every arity; only binary rows have
        (h, z)/(z, y) semantics in this credit, so anything else is
        pool-filtered -- never indexed blind (a unary row would be a bare
        IndexError, an arity>=3 row a silently wrong first-two-columns cover).
        Assumes `_readable(name)` was already True."""
        if name not in binary_cache:
            binary_cache[name] = all(len(r) == 2 for r in facts_cache[name])
        return binary_cache[name]

    left_ext: dict[str, dict[int, list[int]]] = {}
    right_pairs: dict[str, set[tuple[int, int]]] = {}

    def _left(name):
        if name not in left_ext:
            buckets: dict[int, list[int]] = {}
            for row in facts_cache[name]:
                buckets.setdefault(int(row[0]), []).append(int(row[1]))
            left_ext[name] = buckets
        return left_ext[name]

    def _pairs(name):
        if name not in right_pairs:
            right_pairs[name] = {
                (int(r[0]), int(r[1])) for r in facts_cache[name]
            }
        return right_pairs[name]

    def _mask_masks(z, y):
        """True iff witness_mask marks witness (z, y) MASKED. An out-of-range z
        (>= the mask's own row count) is NOT excluded here -- bounds-checking raw
        engine constants is the downstream typed checks' job (prepare_extension's
        bounds check and the dense-identity law below), and they must fire
        exactly as they do on the no-mask path, not be preempted by a bare
        IndexError. A NEGATIVE z is likewise left alone rather than silently
        aliased to the mask's last row via Python's negative indexing."""
        return (witness_mask is not None
                and 0 <= z < witness_mask.shape[0]
                and bool(witness_mask[z, y]))

    specs: list[CandidateSpec] = []
    domain_union: dict[str, set[int]] = {}
    n_total = n_meta = n_neural_left = n_unreadable = n_non_binary = 0
    for cand in prog.valid_candidates(mask_name):
        n_total += 1
        ln, rn = cand["left_name"], cand["right_name"]
        if ln.startswith("__xlog_") or rn.startswith("__xlog_"):
            n_meta += 1
            continue                        # meta relations: arity-incompatible, skip
        if ln in neural_relations:
            n_neural_left += 1
            continue                        # neural-in-left: no witness semantics, skip
        if not _readable(ln):
            n_unreadable += 1
            continue                        # left slot has no ground extension (e.g. template placeholder), skip
        if not _binary(ln):
            n_non_binary += 1
            continue                        # non-binary rows have no (h, z) reading, skip
        if rn not in neural_relations:
            if not _readable(rn):
                n_unreadable += 1
                continue                    # same, right slot -- a neural right needs no facts
            if not _binary(rn):
                n_non_binary += 1
                continue
        if rn in neural_relations:
            witnesses = [
                [z * n_labels + y for z in _left(ln).get(h, [])
                 if not _mask_masks(z, y)]
                for h, y in facts
            ]
            masked_any = (None if witness_mask is None else torch.tensor(
                [any(_mask_masks(z, y) for z in _left(ln).get(h, []))
                 for h, y in facts], device=device))
            idx = prepare_extension(
                witnesses, device,
                num_rows=neural_relations[rn].num_rows * n_labels
            )
            specs.append(CandidateSpec(cand["id"], ln, rn, True, idx, None,
                                       masked_any=masked_any))
            # The FULL left extension (not the fact-restricted witnesses, which
            # vary per fold) is what can ever index this relation's feature rows.
            domain_union.setdefault(rn, set()).update(
                z for zs in _left(ln).values() for z in zs
            )
        else:
            pairs = _pairs(rn)
            lext = _left(ln)
            cover = torch.tensor(
                [1.0 if any((z, y) in pairs for z in lext.get(h, [])) else 0.0
                 for h, y in facts],
                device=device,
            )
            specs.append(CandidateSpec(cand["id"], ln, rn, False, None, cover))
    if not specs:
        raise ValueError(
            f"pool filtering left zero scoreable candidates out of {n_total} "
            f"enumerated: {n_meta} skipped as __xlog_ meta, {n_neural_left} as "
            f"neural-in-left (no witness semantics), {n_unreadable} with an "
            f"unreadable slot (no ground extension), {n_non_binary} with "
            "non-binary rows. Nothing remains to train or select over -- the "
            "filters above are the reason, not a silent cap."
        )
    for rn, joined in domain_union.items():
        rows = neural_relations[rn].num_rows
        if joined != set(range(rows)):
            missing = sorted(set(range(rows)) - joined)[:5]
            extra = sorted(joined - set(range(rows)))[:5]
            raise ValueError(
                f"neural relation '{rn}': raw engine constants index the "
                f"caller's {rows} feature rows, and with no explicit "
                "constant->row map that dense identity is only unambiguous "
                f"when the witness domain is exactly 0..{rows - 1}. Here it is "
                f"not (unjoined rows e.g. {missing}, out-of-range constants "
                f"e.g. {extra}) -- an in-range misalignment would gather other "
                "events' probabilities silently, so this is refused, not "
                "guessed at. Renumber the event constants to the dense range, "
                "or use the mixture path, whose domain_ids= states the map "
                "explicitly."
            )
    return specs


def _validated_output(network, features):
    """The network's output under no_grad, refused unless 2-D: the engine-mode
    credit reads the witness score at the fact's own label column, so the
    output must be [num_events, num_labels]."""
    import torch

    with torch.no_grad():
        out = network(features)
    if out.ndim != 2:
        raise ValueError(
            f"network(features) returned a {out.ndim}-D tensor of shape "
            f"{tuple(out.shape)}; the engine-mode credit reads the witness score "
            "at the fact's own label column, so the output must be 2-D "
            "[num_events, num_labels]."
        )
    return out


@dataclass(frozen=True)
class EngineModeResult:
    """The trained candidate mixture, the specs it was trained over, the (mutated
    in place) network, and the per-step loss trace -- the last is what determinism
    is checked against.

    ``masked_facts`` maps ``(left, right)`` -- for NEURAL specs only -- to the
    number of facts with at least one masked witness (``spec.masked_any.sum()``);
    entries whose count is zero are omitted, never held at 0. Empty when no
    ``witness_mask`` was supplied to ``train_engine_mode``, byte-identical to the
    pre-mask surface."""

    cand_probs: dict
    specs: list
    network: Any
    losses: list
    masked_facts: dict = field(default_factory=dict)


def train_engine_mode(prog, mask_name, facts, is_positive, network, features,
                      neural_relations, steps=400, lr=0.05, gamma=1.0,
                      entropy_start=0.0, entropy_end=0.1, seed=0,
                      witness_mask=None):
    """Train candidate logits + the network against the real-valued credit.

    Deterministic, mirroring the dILP trainer: the OR accumulates with index_add,
    whose default CUDA path is atomic float addition. NOTE two scope caveats of
    that contract: ``torch.use_deterministic_algorithms(True)`` is a PROCESS-GLOBAL
    switch this call never restores (co-resident code that needs nondeterministic
    kernels will start failing), and ``seed`` covers the TRAINING only -- the
    network arrives already constructed, so its init came from the caller's RNG
    (``kfold_select`` seeds each per-fold construction itself; a direct caller
    owns that seeding). Entropy is the Occam pressure the linear credit lacks
    (one-hot preference; weight annealed by entropy_weight_at_step). Selection is
    NOT here -- see kfold_select.

    ``witness_mask`` (contract #155) is threaded straight into
    ``enumerate_specs``: a MASKED witness is excluded from the index at build
    time, so it contributes exactly zero credit and zero gradient to the
    training loss below -- not a smaller one, an absent one. The training NLL
    therefore is a lower bound over the OR's ACTIVE (unmasked) witnesses only;
    the full masked-interval treatment ``frozen_select``/``kfold_select`` apply
    to held-out scoring is a SELECTION-time concern, documented there, not a
    training-loss one -- this bound is the Python-phase boundary (the interval
    itself is future kernel surface). ``EngineModeResult.masked_facts`` reports
    the per-candidate masked-fact accounting from the same specs."""
    import os

    import torch

    from pyxlog.ilp.entropy import entropy_weight_at_step, normalized_entropy

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

    device = features.device
    out = _validated_output(network, features)
    n_labels = out.shape[1]
    specs = enumerate_specs(
        prog, mask_name, facts, neural_relations, device, n_labels,
        witness_mask=witness_mask,
    )
    C = max(s.cid for s in specs) + 1
    # Skipped candidates must not hold probability mass — the mixture is over
    # scoreable candidates only.
    neg_inf_mask = torch.full((C,), float("-inf"), device=device)
    spec_cids = torch.tensor(sorted({s.cid for s in specs}), device=device)
    neg_inf_mask[spec_cids] = 0.0
    W = torch.zeros(C, requires_grad=True, device=device)
    opt = torch.optim.Adam([W] + list(network.parameters()), lr=lr)
    is_pos_t = torch.as_tensor(is_positive, device=device)

    losses: list[float] = []
    for step in range(steps):
        opt.zero_grad()
        p = torch.softmax(W + neg_inf_mask, dim=0)
        # Flattened row-major: witness (z, y) gathers row z * n_labels + y --
        # the fact's own label column, never a guessed positive column.
        p_event = network(features).reshape(-1)
        loss = credit_nll(p, specs, p_event, is_pos_t, gamma=gamma)
        w_ent = entropy_weight_at_step(step, steps, entropy_start, entropy_end)
        active = torch.stack([p[s.cid] for s in specs])
        loss = loss + w_ent * normalized_entropy(active, len(specs))
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    with torch.no_grad():
        p = torch.softmax(W + neg_inf_mask, dim=0)
    masked_facts = {
        (s.left, s.right): int(s.masked_any.sum())
        for s in specs
        if s.is_neural and s.masked_any is not None and bool(s.masked_any.any())
    }
    return EngineModeResult(
        cand_probs={(s.left, s.right): float(p[s.cid]) for s in specs},
        specs=specs, network=network, losses=losses,
        masked_facts=masked_facts,
    )


@dataclass(frozen=True)
class HoldoutSelection:
    """What the holdout arbiter is entitled to claim. Mirrors discovery.Selection's five
    fields, but ``rule``/``tied`` hold (left, right) TUPLES -- our candidates are keyed
    by a relation pair, not by a single string id, so reusing discovery.Selection here
    would be type-sloppy (it constructs fine since dataclasses carry no runtime type
    checks, but its annotations claim strings). A dedicated type keeps that honest.

    ``coverage`` maps each SURVIVING-OR-DROPPED candidate key to the fraction of
    its facts that were CERTAIN (predicted-true, or unmasked) rather than
    uncertain (masked and not yet provably true) -- see ``frozen_select`` /
    ``kfold_select`` for how it is computed. Empty when no ``witness_mask`` was
    supplied: the coverage channel simply did not run, not "everything was
    fully covered"."""

    rule: tuple[str, str] | None
    tied: list[tuple[str, str]]
    margin: float
    top_weight: float
    reason: str
    coverage: dict = field(default_factory=dict)

    @property
    def decided(self) -> bool:
        return self.rule is not None


def _select_from_holdout(scores, neural_rights, min_fit, tie_tolerance=0.01,
                         coverage=None, min_coverage=0.5):
    """Selection over HOLDOUT scores. The fit gate kills candidates that cannot fit
    even their best folds (the confident-wrong class the mixture's select_rule cannot
    see). A MIXED tie within ``tie_tolerance`` breaks toward the RELATIONAL candidate
    (Occam: at equal generalization, prefer the explanation without a network) -- but
    only when that narrowing yields a UNIQUE relational candidate: Occam licenses
    preferring relational over neural, it licenses nothing among relational
    duplicates, so a residual relational tie is an abstention, never a vocabulary-
    order pick. ``tie_tolerance`` lives on the HOLDOUT-accuracy axis; kfold_select
    derives it from the score quantum rather than reusing the weight-axis default.

    ``coverage`` (candidate key -> fraction of CERTAIN facts) gates BEFORE the fit
    gate, when supplied and non-empty: a candidate whose masked witnesses leave it
    with too little certain evidence (< ``min_coverage``) is dropped first --
    masked is not false, so a candidate should never be killed by facts it was
    never allowed to see. If that leaves nothing to score, the reason names
    coverage outright; if some candidates survive but still fail the fit gate,
    the fit-gate reason names the coverage-dropped candidates too, so the
    abstention stays traceable to masking rather than reading as a bare
    "nothing fit". Empty/``None`` coverage (no witness_mask in play) leaves this
    gate a no-op, byte-identical to the pre-mask behavior."""
    from pyxlog.ilp.discovery import select_rule

    coverage = coverage or {}
    for l, r in scores:
        if "|" in l or "|" in r:
            raise ValueError(
                f"relation name in candidate ({l!r}, {r!r}) contains '|', the "
                "internal key separator select_rule keys round-trip through; "
                "scoring it would corrupt the key split, refused."
            )
    dropped: dict = {}
    if coverage:
        low = {k for k, c in coverage.items() if c < min_coverage}
        dropped = {k: scores[k] for k in low if k in scores}
        scores = {k: v for k, v in scores.items() if k not in low}
        if not scores:
            return HoldoutSelection(
                rule=None, tied=sorted(dropped), margin=0.0,
                top_weight=0.0, coverage=coverage,
                reason=f"every candidate lost certain coverage below "
                       f"min_coverage={min_coverage} to masked witnesses: nothing "
                       "is scoreable — masked is not false",
            )
    fit = {k: v for k, v in scores.items() if v >= min_fit}
    if not fit:
        # If the coverage gate already dropped candidates above, the abstention
        # is still traceable to masking, not merely "nothing fit": name it, so a
        # reader is not left to guess whether the discarded candidates would
        # have passed the fit gate had their masked witnesses been readable.
        coverage_note = (
            f" ({len(dropped)} candidate(s) already dropped below "
            f"min_coverage={min_coverage} to masked witnesses before the fit "
            f"gate ran: {', '.join('|'.join(k) for k in sorted(dropped))})"
            if dropped else ""
        )
        return HoldoutSelection(
            rule=None, tied=sorted(scores), margin=0.0,
            top_weight=max(scores.values(), default=0.0), coverage=coverage,
            reason=f"no candidate passed the fit gate (min_fit={min_fit}): a rule "
                   f"that cannot fit held-out data is not a rule{coverage_note}",
        )
    keyed = {f"{l}|{r}": v for (l, r), v in fit.items()}
    # min_weight=min_fit is deliberately vacuous here -- everything below min_fit
    # was already dropped by the gate above; it is stated so select_rule's believed
    # threshold and our fit gate can never disagree.
    sel = select_rule(keyed, min_weight=min_fit, tie_tolerance=tie_tolerance)
    if sel.rule is not None:
        l, r = sel.rule.split("|")
        return HoldoutSelection(rule=(l, r), tied=[(l, r)], margin=sel.margin,
                                top_weight=sel.top_weight, reason=sel.reason,
                                coverage=coverage)
    tied = [tuple(t.split("|")) for t in sel.tied]
    relational = [t for t in tied if t[1] not in neural_rights]
    if relational and len(relational) < len(tied):
        best_fit = max(fit[t] for t in relational)
        rel_tied = sorted(t for t in relational
                          if best_fit - fit[t] <= tie_tolerance)
        if len(rel_tied) == 1:
            best = rel_tied[0]
            return HoldoutSelection(rule=best, tied=tied, margin=sel.margin,
                                    top_weight=fit[best], coverage=coverage,
                                    reason="holdout tie broken toward the relational candidate "
                                           "(Occam: equal generalization, simpler explanation)")
        return HoldoutSelection(
            rule=None, tied=tied, margin=sel.margin, top_weight=best_fit,
            coverage=coverage,
            reason=f"Occam narrowed the tie to {len(rel_tied)} relational "
                   f"candidates ({', '.join('|'.join(t) for t in rel_tied)}) the "
                   "data cannot distinguish: preferring relational over neural is "
                   "licensed, picking among relational duplicates is not",
        )
    return HoldoutSelection(rule=None, tied=tied, margin=sel.margin,
                            top_weight=sel.top_weight, reason=sel.reason,
                            coverage=coverage)


def frozen_select(prog, mask_name, facts, is_positive, network, features,
                  neural_relations, min_fit=0.75, witness_mask=None,
                  min_coverage=0.5):
    """Score every engine-enumerated candidate against a FROZEN detector.

    Engine mode as an ACCEPTANCE INSTRUMENT rather than a trainer: the detector
    is externally trained and never touched here -- no gradient, no optimizer
    (a parameterless module works), no folds (nothing trains, so every fact is
    already held out with respect to this system). Each candidate is scored by
    its own witness/cover semantics over ALL facts and the selection gates are
    exactly the holdout arbiter's: fit gate, select_rule tie semantics with the
    score-quantum tolerance, Occam narrowing to a unique relational candidate,
    abstention as a first-class outcome. This answers "is this rule right GIVEN
    this detector" -- ``kfold_select`` answers "can a detector be trained for
    this rule".

    The detector must be in EVAL mode (``network.eval()``): in train mode a
    BatchNorm mutates its running statistics even under no_grad and dropout
    makes two identical scoring calls disagree -- both would break the
    bit-identical-artifact guarantee, so a train-mode module is refused, not
    silently switched (refusal teaches the contract; mode-switching would hide
    caller bugs).

    ``witness_mask`` (contract #155, threaded into ``enumerate_specs``) turns a
    masked fact's truth into an INTERVAL, not a coerced ``False``: a neural
    fact is CERTAIN iff it is predicted-true (``OR_active >= 0.5`` -- the OR is
    monotone in its witness probabilities, so this stays true under ANY
    completion of the masked witnesses) or it never had a masked witness at all
    (``masked_any`` is ``False``). Facts that are neither -- masked AND not yet
    predicted-true -- are UNCERTAIN: excluded from the candidate's accuracy
    (they neither help nor hurt) and counted against its ``coverage`` (the
    fraction of facts that were certain). Relational candidates are never
    masked, so every one of their facts is certain. A candidate with zero
    certain facts gets accuracy 0.0 and coverage 0.0 (never NaN, which would
    corrupt ``select_rule``'s comparisons) -- the coverage gate in
    ``_select_from_holdout`` drops it by name before the fit gate ever runs."""
    import torch

    if not facts:
        raise ValueError(
            "frozen_select needs at least one fact: candidate accuracies are "
            "means over facts, and a mean over nothing is not a score."
        )
    if len(is_positive) != len(facts):
        raise ValueError(
            f"is_positive has {len(is_positive)} entries for {len(facts)} "
            "facts. A shorter tensor would BROADCAST silently and score every "
            "candidate against the wrong labels, so the mismatch is refused."
        )
    if getattr(network, "training", False):
        raise ValueError(
            "a frozen detector is scored in eval mode -- call network.eval() "
            "first. In train mode BatchNorm mutates its running statistics "
            "even under no_grad and dropout de-determinizes the scores, both "
            "violating the frozen guarantee this entry point exists to "
            "provide."
        )
    out = _validated_output(network, features)
    specs = enumerate_specs(
        prog, mask_name, facts, neural_relations, features.device, out.shape[1],
        witness_mask=witness_mask,
    )
    with torch.no_grad():
        p_event = out.reshape(-1)
        y = torch.tensor([bool(v) for v in is_positive],
                         device=features.device, dtype=torch.float32)
        scores = {}
        coverage = {}
        for spec in specs:
            key = (spec.left, spec.right)
            s = (noisy_or_from_index(p_event, spec.witness_index)
                 if spec.is_neural else spec.binary_cover)
            pred = (s >= 0.5).float()
            if spec.masked_any is not None:
                # OR>=0.5 is a CERTAIN true regardless of the masked witnesses
                # (monotonicity); the remaining facts are uncertain, not false.
                certain = pred.bool() | (~spec.masked_any)
            else:
                certain = torch.ones_like(pred, dtype=torch.bool)
            n_certain = int(certain.sum())
            n_total = certain.numel()
            scores[key] = (
                float((pred[certain] == y[certain]).float().mean())
                if n_certain else 0.0
            )
            if witness_mask is not None:
                coverage[key] = (n_certain / n_total) if n_total else 0.0
    # Same derivation as kfold_select: one fact is the smallest evidence step.
    tie_tolerance = max(0.01, 1.0 / len(facts))
    return _select_from_holdout(scores, set(neural_relations), min_fit,
                                tie_tolerance=tie_tolerance,
                                coverage=coverage, min_coverage=min_coverage)


def kfold_select(prog_factory, mask_name, facts, is_positive, make_network,
                 features, neural_relations, folds=4, min_fit=0.75, seed=0,
                 witness_mask=None, min_coverage=0.5, **train_kw):
    """Select a rule by K-FOLD HOLDOUT, not by training weight: per fold, train on
    the rest and score every engine-enumerated candidate on the held-out facts by
    its own witness/cover semantics (``s_c(f) >= 0.5``); average across folds, apply
    the fit gate, then hand the holdout scores to ``_select_from_holdout``.

    ``seed`` determines the WHOLE run, network inits included: each fold's
    ``make_network()`` is called right after ``torch.manual_seed`` with a seed
    derived from (seed, fold), so two calls with the same arguments are identical
    regardless of ambient RNG state -- the declared contract, not a fixture
    obligation.

    ``witness_mask`` is threaded into BOTH the per-fold ``train_engine_mode``
    call and the HELD-OUT scoring's ``enumerate_specs`` call: the mask is a
    GLOBAL (event, label) row space, the same tensor for every fold -- only the
    fact subsets differ per fold, already handled by the per-fold train/held-out
    split -- so passing the identical mask to each fold's training is correct,
    not a per-fold recomputation. Per fold, a fact is CERTAIN by the same
    interval rule as ``frozen_select`` (predicted-true, or never masked);
    uncertain facts are excluded from that fold's accuracy and contribute to
    ``n_certain``/``n_total``, which are summed ACROSS folds (each fact is held
    out exactly once) into a single pooled ``coverage`` fraction per candidate,
    gated the same way."""
    import torch

    if not 2 <= folds <= len(facts):
        raise ValueError(
            f"folds={folds} with {len(facts)} facts: every fold needs at least "
            "one held-out fact (an empty fold's mean accuracy is NaN and would "
            "poison every candidate's score) and training needs at least one "
            "fold's worth of facts left over."
        )
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(facts), generator=rng).tolist()
    fold_of = {f_idx: i % folds for i, f_idx in enumerate(order)}
    sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    certain_sums: dict[tuple[str, str], int] = {}
    total_sums: dict[tuple[str, str], int] = {}
    neural_rights = set(neural_relations)

    # ONE compilation serves every fold: the program is only ever READ here
    # (valid_candidates, relation_facts), and the scale probe measured per-fold
    # recompilation at ~75% of the kfold wall time at 10^4 events. A factory
    # with per-call side effects would observe one call instead of `folds`.
    prog = prog_factory()

    for fold in range(folds):
        train_ids = [i for i in range(len(facts)) if fold_of[i] != fold]
        held_ids = [i for i in range(len(facts)) if fold_of[i] == fold]
        # Derived (seed, fold) seeding right before construction: the network
        # init must come from OUR seed, not whatever ambient RNG state the
        # caller happens to be in (finding B, review of PR #154).
        torch.manual_seed(seed * 100_003 + fold)
        res = train_engine_mode(
            prog, mask_name,
            [facts[i] for i in train_ids],
            [is_positive[i] for i in train_ids],
            make_network(), features, neural_relations, seed=seed,
            witness_mask=witness_mask, **train_kw)
        with torch.no_grad():
            out = res.network(features)
            held_specs = enumerate_specs(
                prog, mask_name, [facts[i] for i in held_ids],
                neural_relations, features.device, out.shape[1],
                witness_mask=witness_mask)
            p_event = out.reshape(-1)
            y = torch.tensor([is_positive[i] for i in held_ids],
                             device=features.device, dtype=torch.float32)
            for spec in held_specs:
                s = (noisy_or_from_index(p_event, spec.witness_index)
                     if spec.is_neural else spec.binary_cover)
                pred = (s >= 0.5).float()
                if spec.masked_any is not None:
                    certain = pred.bool() | (~spec.masked_any)
                else:
                    certain = torch.ones_like(pred, dtype=torch.bool)
                n_certain = int(certain.sum())
                n_total = certain.numel()
                acc = (
                    float((pred[certain] == y[certain]).float().mean())
                    if n_certain else 0.0
                )
                key = (spec.left, spec.right)
                sums[key] = sums.get(key, 0.0) + acc
                counts[key] = counts.get(key, 0) + 1
                if witness_mask is not None:
                    certain_sums[key] = certain_sums.get(key, 0) + n_certain
                    total_sums[key] = total_sums.get(key, 0) + n_total

    scores = {k: sums[k] / counts[k] for k in sums}
    coverage = (
        {k: (certain_sums[k] / total_sums[k]) if total_sums[k] else 0.0
         for k in sums}
        if witness_mask is not None else {}
    )
    # The tie tolerance lives on the HOLDOUT-accuracy axis, not the guard-weight
    # axis select_rule's 0.01 default was calibrated for. Each fact is held out
    # exactly once, so flipping one fact moves the fold-mean score by roughly
    # 1/len(facts) -- differences below one fact are quantization noise, not
    # evidence, and must count as ties.
    tie_tolerance = max(0.01, 1.0 / len(facts))
    return _select_from_holdout(scores, neural_rights, min_fit,
                                tie_tolerance=tie_tolerance,
                                coverage=coverage, min_coverage=min_coverage)
