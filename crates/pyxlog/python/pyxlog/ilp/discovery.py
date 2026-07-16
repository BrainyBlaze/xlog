"""Rule DISCOVERY for neural join bodies: the world, and the candidate SWEEP.

The flagship claim is that nobody writes the rule. A relation VOCABULARY goes in;
one Stage-B candidate per relation comes out::

    trainable_rule(cand_<r>, weight=0.0) ::
        plastic(E) :- saliency(Ev, strengthen), <r>(Ev, E).

They all compete in the same joint mixture, and the system has to pick the relation
whose join extension actually carries the planted signal -- while learning, from
scratch, a per-EVENT detector of that signal. Two responsibilities live here and
nothing else: generating that world, and generating those candidates.

WHY THE DISTRACTORS ARE EQUAL-CARDINALITY (this is load-bearing, and it was learned
the hard way on GPU). A distractor relation that gives each edge FEWER events than the
correct relation has a SHARPER noisy-OR -- ``1-(1-p)^k`` is easier to push to 0 with
small ``k`` -- so it wins for purely STRUCTURAL reasons, with no reference to the
label. Worse, a distractor whose events ANTI-correlate with the label has its own
exact zero-loss solution with an INVERTED detector, which makes the whole test a coin
flip that no correct implementation can pass. Neither failure is a property of the
mechanism; both are defects of the world. So every relation here hands each edge the
SAME number of events as ``pre_before_post``, drawn from OTHER edges' events: same
cardinality, zero label information. Do not "simplify" this.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SALIENT_THRESHOLD = 0.5
HEAD = "plastic"
NETWORK = "sal_net"
NEURAL_PREDICATE = "saliency"
POSITIVE_LABEL = "strengthen"
NEGATIVE_LABEL = "low"
CORRECT_RELATION = "pre_before_post"


@dataclass(frozen=True)
class World:
    """A generated world. ``labels[i]`` is the planted truth of ``edges[i]``.

    Head bindings are the dense range ``0..n_edges-1`` and events are the dense range
    ``0..n_edges*events_per_edge-1``, so ``domain_inputs`` row ``e`` holds event ``e``
    and ``domain_ids`` may be left to its default.
    """

    event_features: list[float]
    pre_before_post: list[tuple[int, int]]
    post_before_pre: list[tuple[int, int]]
    co_occurs: list[tuple[int, int]]
    edges: list[int]
    labels: list[bool]

    def facts(self) -> str:
        """The EDB, as xlog source. The relations are all the engine ever gets; the
        edge->events map is never handed to the trainer (if it were, the OR would be
        Python's aggregation over a caller-supplied hint, not the logic's)."""
        lines: list[str] = []
        for relation, tuples in (
            ("pre_before_post", self.pre_before_post),
            ("post_before_pre", self.post_before_pre),
            ("co_occurs", self.co_occurs),
        ):
            lines += [f"{relation}({ev}, {edge})." for ev, edge in tuples]
        return "\n".join(lines)


def _fair_distractor(
    rng: random.Random,
    n_edges: int,
    events_per_edge: int,
    own: dict[int, list[int]],
    salient_events: set[int],
    all_events: list[int],
) -> list[tuple[int, int]]:
    """``events_per_edge`` events per edge, sampled from OTHER edges' events, with a
    CLASS-INDEPENDENT salient composition.

    Same cardinality as the correct relation (so its OR is exactly as sharp -- no
    structural advantage) and no information about the edge's own label. The second
    property needs more than "sample from other edges": a positive edge's pool holds
    S-1 salient events while a negative edge's holds S, so uniform sampling leaks
    O(1/(n_edges-1)) of ANTI-correlated label signal into the distractor -- exactly
    the inverted-detector zero-loss direction the module docstring calls a coin flip
    no correct implementation can pass, and material at small n_edges. So the
    COMPOSITION is drawn first, from one distribution shared by both classes
    (Binomial(k, (S-1)/(N-k)) -- the positive-pool rate, used for everyone), and only
    then are event identities sampled per stratum. The count a converged detector's
    OR sees is thereby exactly label-independent by construction.
    """
    n_total = len(all_events)
    n_salient = len(salient_events)
    pool_rate = max(0.0, (n_salient - 1) / max(1, n_total - events_per_edge))

    tuples: list[tuple[int, int]] = []
    for edge in range(n_edges):
        mine = set(own[edge])
        pool_sal = [ev for ev in all_events if ev in salient_events and ev not in mine]
        pool_quiet = [
            ev for ev in all_events if ev not in salient_events and ev not in mine
        ]
        n_sal = sum(1 for _ in range(events_per_edge) if rng.random() < pool_rate)
        n_sal = min(n_sal, len(pool_sal))
        n_quiet = min(events_per_edge - n_sal, len(pool_quiet))
        n_sal = events_per_edge - n_quiet     # backfill if the quiet pool ran short
        for ev in rng.sample(pool_sal, n_sal) + rng.sample(pool_quiet, n_quiet):
            tuples.append((ev, edge))
    return tuples


def make_world(n_edges: int, events_per_edge: int, seed: int) -> World:
    """Planted: an edge is plastic iff SOME of its pre->post events is salient
    (feature > 0.5). A positive edge gets EXACTLY ONE salient event; the rest are
    quiet -- which is precisely the shape a head-bound gate over a pooled edge
    feature cannot recover (a lone salient event dilutes), and which a per-event
    detector under an OR recovers exactly.
    """
    if events_per_edge < 1:
        raise ValueError("events_per_edge must be >= 1")
    if n_edges < 2:
        raise ValueError("n_edges must be >= 2 (a fair distractor needs other edges)")

    rng = random.Random(seed)
    features: list[float] = []
    own: dict[int, list[int]] = {}
    pre: list[tuple[int, int]] = []

    event_id = 0
    for edge in range(n_edges):
        positive = rng.random() < 0.5
        events: list[int] = []
        for slot in range(events_per_edge):
            salient = positive and slot == 0
            value = rng.uniform(0.6, 0.99) if salient else rng.uniform(0.01, 0.4)
            features.append(round(value, 3))
            events.append(event_id)
            pre.append((event_id, edge))
            event_id += 1
        own[edge] = events

    all_events = list(range(event_id))
    salient_events = {
        ev for ev in all_events if features[ev] > SALIENT_THRESHOLD
    }
    post = _fair_distractor(
        rng, n_edges, events_per_edge, own, salient_events, all_events
    )
    co = _fair_distractor(
        rng, n_edges, events_per_edge, own, salient_events, all_events
    )

    edges = list(range(n_edges))
    # The label is READ BACK off the generated relation, not off `positive`: it is a
    # statement about the world that exists, so a generator bug shows up as a label
    # that disagrees with the facts rather than as a silently unlearnable target.
    labels = [
        any(features[ev] > SALIENT_THRESHOLD for ev, e in pre if e == edge)
        for edge in edges
    ]
    return World(
        event_features=features,
        pre_before_post=pre,
        post_before_pre=post,
        co_occurs=co,
        edges=edges,
        labels=labels,
    )


def build_join_candidates(
    vocabulary: list[str], head: str = HEAD
) -> tuple[str, list[str]]:
    """Sweep a relation VOCABULARY into ONE Stage-B candidate per relation.

    The caller supplies the relation names; this fills the single free slot of a fixed
    template, once per name, and the mixture weighs the results. That is the whole of
    the search: |R| candidates, linear, one body shape. It is candidate SELECTION, not
    open rule induction -- see ``select_rule`` for how to read the answer honestly, and
    the module docstring of ``join_bodies`` for what the body may contain.

    Returns ``(source, candidate_ids)``. The source carries the ``nn/4`` declaration,
    the ``pred`` declarations, the candidates and the ``train`` statement; the caller
    prepends the world's facts.
    """
    lines: list[str] = [
        f"nn({NETWORK}, [Event], Label, [{NEGATIVE_LABEL}, {POSITIVE_LABEL}]) :: "
        f"{NEURAL_PREDICATE}(Event, Label).",
    ]
    lines += [f"pred {relation}(i64, i64)." for relation in vocabulary]
    lines.append(f"pred {head}(i64).")

    ids: list[str] = []
    for relation in vocabulary:
        rule_id = f"cand_{relation}"
        lines.append(
            f"trainable_rule({rule_id}, weight=0.0) :: {head}(E) :- "
            f"{NEURAL_PREDICATE}(Ev, {POSITIVE_LABEL}), {relation}(Ev, E)."
        )
        ids.append(rule_id)

    lines.append(f"train({head}, binary_cross_entropy).")
    return "\n".join(lines), ids


# Two candidates whose weights land this close are NOT ranked -- they are TIED. The
# value is not a taste: it is read off the measured collapse. On a nested-superset
# distractor (the correct relation's events plus a few foreign quiet ones) the winning
# margin fell from 3333x to 1.003x -- 0.99551 vs 0.99250, a gap of 0.003 -- and on an
# exact extensional duplicate the two weights agreed to TWELVE decimal places. Anything
# inside this band is the optimizer's coin, not the data's answer.
TIE_TOLERANCE = 0.01

# A candidate the mixture does not actually believe is not a rule. When a trivially-true
# relation is in the vocabulary the run can land in a degenerate minimum where EVERY
# weight is near zero and the argmax is meaningless (measured: 1 of 2 seeds, the winner
# was a relation with no signal at all, accuracy 0.625 -- below the head-gate baseline).
MIN_WEIGHT = 0.5

# A candidate can ALSO be believed and alone at the top while not actually fitting the
# data: a trivially-true relation's mask is ~1 everywhere, so its "fit" to the label is
# just the label's own base rate, not evidence of a relationship. MIN_FIT is the default
# floor on `fits[rule_id]` (== mean((mask >= 0.5) == targets) on the TRAIN set, from
# `neurosymbolic.train_neurosymbolic_program`'s `candidate_train_fit`) below which a
# candidate is dropped BEFORE ranking, whatever its guard weight. Measured: the
# trivially-true world's derailing seed selects `co_occurs` at weight 0.955 but fit
# 0.500 (a coin flip) -- comfortably caught by this gate.
MIN_FIT = 0.75


@dataclass(frozen=True)
class Selection:
    """What the mixture is entitled to claim. ``rule`` is None unless ONE candidate won.

    ``tied`` lists every candidate within ``TIE_TOLERANCE`` of the top weight, sorted by
    id, so the verdict never depends on the order the caller listed the vocabulary in.
    """

    rule: str | None
    tied: list[str]
    margin: float
    top_weight: float
    reason: str

    @property
    def decided(self) -> bool:
        return self.rule is not None


def select_rule(
    weights: dict[str, float],
    *,
    tie_tolerance: float = TIE_TOLERANCE,
    min_weight: float = MIN_WEIGHT,
    fits: dict[str, float] | None = None,
    min_fit: float = MIN_FIT,
) -> Selection:
    """Read a winner off the candidate weights -- or REFUSE to.

    ``max(weights, key=weights.get)`` is not a discovery signal. Python's ``max``
    returns the FIRST key holding the maximum, so on two extensionally identical
    candidates -- weights equal to twelve decimals -- it hands back whichever relation
    the caller happened to type first. Reversing the vocabulary reverses the "discovered
    rule", and the accuracy is 1.000 either way, so nothing downstream notices. That is
    a confident wrong answer, and this function exists to refuse to give one.

    A rule is claimed only when a single candidate is BOTH believed (weight >=
    ``min_weight``) AND alone at the top (the runner-up is more than ``tie_tolerance``
    behind). Otherwise ``rule`` is None and ``reason`` says which gate failed.

    ``fits``, when given, is the per-candidate TRAIN-set fit (e.g.
    ``NeuroSymbolicTrainingResult.candidate_train_fit``): ``mean((mask >= 0.5) ==
    targets)``. Any candidate whose fit is below ``min_fit`` is dropped BEFORE
    ranking -- whatever its guard weight -- because a believed, top-ranked candidate
    can still fail to actually fit the data (a trivially-true relation's mask is ~1
    everywhere, so it "wins" the weight ranking on a degenerate minimum while its fit
    is a coin flip). A candidate ``fits`` does not mention is treated as fit 0.0 (it
    cannot pass the gate). If NO candidate survives the gate, this returns an
    abstention naming the fit gate and the best fit value seen, and nothing below is
    reached. Omitting ``fits`` (the default, ``None``) skips this gate entirely --
    behavior is then byte-identical to before this parameter existed.

    When the gate runs, ``tied``, ``margin`` and ``top_weight`` are POST-GATE
    quantities, computed over the surviving candidates only: a gated-out rival
    with a near-top weight is not a viable alternative, so it no longer blocks
    selection through the tie refusal. This is intended (the seed-1
    trivially-true world depends on it), not an accident of ordering.
    """
    if not weights:
        return Selection(None, [], 0.0, 0.0, "no candidates")

    candidates = weights
    if fits is not None:
        candidate_fits = {rule_id: fits.get(rule_id, 0.0) for rule_id in weights}
        gated = {
            rule_id: w
            for rule_id, w in weights.items()
            if candidate_fits[rule_id] >= min_fit
        }
        if not gated:
            best_id, best_fit = max(
                candidate_fits.items(), key=lambda kv: (kv[1], kv[0])
            )
            return Selection(
                None, [], 0.0, 0.0,
                f"fit gate: no candidate reaches min_fit={min_fit} "
                f"(best fit: {best_id} at {best_fit:.5f})",
            )
        candidates = gated

    ranked = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0]))
    top_id, top = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top - runner_up
    tied = sorted(rid for rid, w in ranked if top - w <= tie_tolerance)

    if top < min_weight:
        return Selection(
            None, tied, margin, top,
            f"no candidate is believed: the top weight is {top:.5f} < {min_weight}",
        )
    if len(tied) > 1:
        return Selection(
            None, tied, margin, top,
            f"{len(tied)} candidates tie within {tie_tolerance} "
            f"({', '.join(tied)}): the relations are indistinguishable on this data",
        )
    return Selection(top_id, tied, margin, top, f"won by {margin:.5f}")
