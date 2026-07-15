"""IDENTIFIABILITY. When can the mixture legitimately name a rule -- and when must it refuse?

The flagship world is the FRIENDLY case: its distractors are built to carry zero label
information, so exactly one candidate is fittable at all, and the winner comes back at
0.99975 against 0.0003. That margin measures the WORLD, not the method. In any real
event log the relations overlap (`co_occurs` and `pre_before_post` share the triggering
event), and this file pins what happens then. All numbers below are MEASURED (A40).

The mechanism at issue: the inter-candidate noisy-OR is MONOTONE, and there is no
sparsity term anywhere -- no L1, no `weight_decay`, no simplex over the candidates. So
two candidates with the same extension are exactly degenerate: the head probability
1-(1-w1*m)(1-w2*m) is reachable with the mass SPLIT, and the loss is flat between them.

    1. partial overlap        -> the correct relation still wins, and by a lot (971x)
    2. exact duplicate        -> a TIE. Weights agreed to twelve decimals; `argmax` would
                                 have handed back whichever relation was typed first.
    3. trivially-true relation-> a degenerate minimum `select_rule` does NOT catch:
                                 the WRONG candidate comes back believed and alone at
                                 the top. Re-measured after the class-independent
                                 distractor fix: seed 0 RECOVERED (correct rule at
                                 0.9996, accuracy 1.000); seed 1 still derails
                                 (co_occurs at 0.955, accuracy 0.500). XFAIL(strict) on
                                 the seed that still fails. Degeneracy is not ambiguity
                                 -- the gates below detect a mixture that believes
                                 nothing, not one that confidently believes the wrong
                                 thing; the root cause (no Occam term, no fit gate) is
                                 unchanged.
"""
import random

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from pyxlog.ilp.discovery import (
    CORRECT_RELATION,
    NEGATIVE_LABEL,
    NETWORK,
    NEURAL_PREDICATE,
    POSITIVE_LABEL,
    make_world,
    select_rule,
)
from pyxlog.ilp.neurosymbolic import (
    NeuroSymbolicTrainingConfig,
    train_neurosymbolic_program,
)

N_EDGES = 40
K = 6
QUIET_PRIOR_BIAS = -2.0
WINNER = f"cand_{CORRECT_RELATION}"


# ---------------------------------------------------------------------------
# The verdict is a pure function. These need no GPU, and they are the ones that
# would have caught the bug: `argmax` is not a discovery signal.
# ---------------------------------------------------------------------------
def test_a_clean_win_is_reported_as_a_win():
    s = select_rule({"cand_a": 0.99975, "cand_b": 0.00030, "cand_c": 0.00028})
    assert s.decided and s.rule == "cand_a"
    assert s.tied == ["cand_a"]
    assert s.margin == pytest.approx(0.99945, abs=1e-5)


def test_two_indistinguishable_candidates_are_a_tie_not_a_winner():
    # The measured exact-duplicate weights, to the twelve decimals they came back with.
    s = select_rule({"cand_pre_before_post": 0.994825720787,
                     "cand_co_occurs": 0.994825720787,
                     "cand_post_before_pre": 0.00034})
    assert not s.decided
    assert s.rule is None
    assert s.tied == ["cand_co_occurs", "cand_pre_before_post"]
    assert "indistinguishable" in s.reason


def test_the_verdict_does_not_depend_on_the_order_the_vocabulary_was_typed_in():
    """THE BUG THIS FILE EXISTS FOR.

    `max(dict, key=dict.get)` returns the FIRST key holding the maximum. On the measured
    exact-duplicate weights it therefore said `pre_before_post` when the caller listed
    that relation first, and `co_occurs` when they listed it first -- stamping "RULE
    DISCOVERED" on either, at accuracy 1.000, with nothing downstream to notice.
    """
    a, b = 0.994825720787, 0.994825720787
    forward = {"cand_pre_before_post": a, "cand_co_occurs": b}
    reverse = {"cand_co_occurs": b, "cand_pre_before_post": a}

    assert max(forward, key=forward.get) != max(reverse, key=reverse.get)  # the old way flips
    assert select_rule(forward) == select_rule(reverse)                    # the new way cannot
    assert select_rule(forward).rule is None


def test_a_near_duplicate_inside_the_tolerance_is_also_a_tie():
    # Measured on the nested superset: the margin collapsed from 3333x to 1.003x.
    s = select_rule({"cand_pre_before_post": 0.99551, "cand_co_occurs": 0.99250})
    assert not s.decided and s.margin < 0.01


def test_a_candidate_nobody_believes_is_not_a_rule():
    """A run where the mixture believes nothing still has an argmax. It means nothing.

    NOTE the limit of this gate, which the trivially-true world below makes painfully
    concrete: it catches a mixture that believes NOTHING, not a mixture that confidently
    believes the WRONG thing. Degeneracy is not ambiguity, and `select_rule` only sees
    the weights.
    """
    s = select_rule({"cand_post_before_pre": 0.0113, "cand_pre_before_post": 0.0024})
    assert not s.decided
    assert "no candidate is believed" in s.reason


# ---------------------------------------------------------------------------
# The worlds. These train, so they need CUDA.
# ---------------------------------------------------------------------------
cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="xlog engine requires CUDA"
)


def _source(world, co_occurs_tuples, extra_relation=None):
    """The demo's program, with `co_occurs` REPLACED by a chosen extension."""
    vocab = ["pre_before_post", "post_before_pre", "co_occurs"]
    lines = []
    for rel, tuples in (
        ("pre_before_post", world.pre_before_post),
        ("post_before_pre", world.post_before_pre),
        ("co_occurs", co_occurs_tuples),
    ):
        lines += [f"{rel}({ev}, {edge})." for ev, edge in tuples]
    if extra_relation is not None:
        name, tuples = extra_relation
        vocab.append(name)
        lines += [f"{name}({ev}, {edge})." for ev, edge in tuples]

    src = [
        f"nn({NETWORK}, [Event], Label, [{NEGATIVE_LABEL}, {POSITIVE_LABEL}]) :: "
        f"{NEURAL_PREDICATE}(Event, Label)."
    ]
    src += [f"pred {r}(i64, i64)." for r in vocab]
    src.append("pred plastic(i64).")
    for r in vocab:
        src.append(
            f"trainable_rule(cand_{r}, weight=0.0) :: plastic(E) :- "
            f"{NEURAL_PREDICATE}(Ev, {POSITIVE_LABEL}), {r}(Ev, E)."
        )
    src.append("train(plastic, binary_cross_entropy).")
    return "\n".join(lines) + "\n" + "\n".join(src)


def _own_events(world):
    own = {}
    for ev, edge in world.pre_before_post:
        own.setdefault(edge, []).append(ev)
    return own


def _train(source, world, seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    with torch.no_grad():
        net[0].bias[1] += QUIET_PRIOR_BIAS
    feats = torch.tensor([[f] for f in world.event_features], dtype=torch.float32)
    tgts = torch.tensor([1.0 if y else 0.0 for y in world.labels], dtype=torch.float32)
    result = train_neurosymbolic_program(
        source,
        networks={NETWORK: net},
        domain_inputs={NETWORK: feats},
        domain_ids={NETWORK: list(range(len(world.event_features)))},
        examples=[{"targets": tgts}],
        config=NeuroSymbolicTrainingConfig(steps=1500, learning_rate=0.05),
    )
    accuracy = sum(
        (p >= 0.5) == y for p, y in zip(result.query_probabilities, world.labels)
    ) / len(world.labels)
    return select_rule(result.symbolic_rule_weights), accuracy


@cuda
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_distractor_holding_five_of_six_own_events_still_loses(seed):
    """The GOOD news, and it is stronger than the theory predicted.

    `co_occurs` inherits five of the edge's own events -- INCLUDING the salient one --
    and one foreign event. The single foreign event is enough: it fires on some negative
    edges, and BCE's -log(1-p) hammers the candidate's gate into the floor. Measured
    margin: 971x (0.99974 vs 0.00103). Partial overlap does NOT break selection.
    """
    world = make_world(n_edges=N_EDGES, events_per_edge=K, seed=seed)
    rng = random.Random(1000 + seed)
    own = _own_events(world)
    all_events = list(range(len(world.event_features)))

    co = []
    for edge in range(N_EDGES):
        mine = own[edge]
        foreign = rng.sample([e for e in all_events if e not in set(mine)], K - 5)
        co += [(ev, edge) for ev in mine[:5] + foreign]

    selection, accuracy = _train(_source(world, co), world, seed)
    assert selection.rule == WINNER, selection.reason
    assert selection.margin > 0.5
    assert accuracy > 0.95


@cuda
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_an_exact_duplicate_relation_is_a_tie_and_the_run_says_so(seed):
    """`co_occurs` IS `pre_before_post` under another name.

    The mass does not go to one of them -- it is DUPLICATED onto both (measured: equal
    to twelve decimal places), because the noisy-OR needs only one of them to fire and
    nothing penalizes carrying two. There is no right answer here, and the honest output
    is a refusal. Accuracy stays 1.000 throughout, which is exactly why accuracy must
    never be read as evidence that the relation was identified.
    """
    world = make_world(n_edges=N_EDGES, events_per_edge=K, seed=seed)
    own = _own_events(world)
    co = [(ev, edge) for edge in range(N_EDGES) for ev in own[edge]]

    selection, accuracy = _train(_source(world, co), world, seed)
    assert not selection.decided, f"claimed {selection.rule} on indistinguishable relations"
    assert selection.tied == ["cand_co_occurs", f"cand_{CORRECT_RELATION}"]
    assert accuracy > 0.95, "and the labels are still fit perfectly -- that is the trap"


@cuda
@pytest.mark.parametrize(
    "seed",
    [
        # Seed 0 RECOVERED when the distractor became exactly class-independent
        # (review finding 7): re-measured, it now selects the correct rule at 0.9996,
        # accuracy 1.000. The strict xfail caught the recovery as an XPASS -- which is
        # the whole point of strict.
        0,
        pytest.param(
            1,
            marks=pytest.mark.xfail(
                reason="MEASURED, NOT FIXED, and `select_rule` does not save us here. "
                       "With a trivially-true relation in the vocabulary, seed 1 still "
                       "lands in a degenerate minimum: it selects `co_occurs` at weight "
                       "0.955 (accuracy 0.500 -- coin-flip, far below the 0.847 "
                       "head-gate baseline). The wrong candidate is BELIEVED and ALONE "
                       "at the top, so the tie/min-weight gates pass it through: a "
                       "confident wrong answer, not an abstention. The fair-distractor "
                       "fix recovered seed 0; the ROOT CAUSE -- no Occam/sparsity term, "
                       "no fit gate -- is unchanged, and this seed still exhibits it.",
                strict=True,
            ),
        ),
    ],
)
def test_a_trivially_true_relation_does_not_derail_the_selection(seed):
    """`anything(Ev, E)` holds for EVERY (event, edge): its mask is ~1 everywhere.

    This is the honest red test. It is not a corner case invented to embarrass the
    branch: a relation that holds of everything is the most ordinary thing a real
    vocabulary can contain, and it takes the mechanism apart.
    """
    world = make_world(n_edges=N_EDGES, events_per_edge=K, seed=seed)
    own = _own_events(world)
    rng = random.Random(2000 + seed)
    all_events = list(range(len(world.event_features)))

    co = []
    for edge in range(N_EDGES):
        mine = set(own[edge])
        co += [(ev, edge) for ev in rng.sample([e for e in all_events if e not in mine], K)]
    anything = [(ev, edge) for ev in all_events for edge in range(N_EDGES)]

    selection, accuracy = _train(
        _source(world, co, extra_relation=("anything", anything)), world, seed
    )
    assert selection.rule == WINNER, selection.reason
    assert accuracy > 0.95
