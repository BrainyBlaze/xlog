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
    all_events: list[int],
) -> list[tuple[int, int]]:
    """``events_per_edge`` events per edge, sampled from OTHER edges' events.

    Same cardinality as the correct relation (so its OR is exactly as sharp -- no
    structural advantage) and no information about the edge's own label (so it cannot
    be fit, in either polarity, by any detector).
    """
    tuples: list[tuple[int, int]] = []
    for edge in range(n_edges):
        mine = set(own[edge])
        pool = [ev for ev in all_events if ev not in mine]
        for ev in rng.sample(pool, events_per_edge):
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
    post = _fair_distractor(rng, n_edges, events_per_edge, own, all_events)
    co = _fair_distractor(rng, n_edges, events_per_edge, own, all_events)

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
    """Sweep a relation vocabulary into ONE Stage-B candidate per relation.

    Nothing here is hand-written: give it ``["pre_before_post", "post_before_pre",
    "co_occurs"]`` and it emits three same-head candidates that differ ONLY in which
    relation joins the existential event variable to the head. Which of them is the
    rule is the mixture's answer, not the caller's.

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
