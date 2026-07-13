"""The join extension must come FROM THE ENGINE. If this test passed while the map
were supplied from Python, it would prove nothing -- so it asserts against the
engine's own answer for facts the engine compiled.

API NOTE (correction found on the GPU box, 2026-07-13): the object the trainer's
mixture holds is `pyxlog.Program.compile(...)` -> `CompiledProgram`, whose only
read surface is `evaluate*` -- it does not expose facts at all (`EvalResult.atoms`
comes back empty for ground facts, and it has no `batch_fact_membership`).
`pyxlog.IlpProgramFactory.compile(...)` -> `CompiledIlpProgram` does expose facts,
via `relation_facts(name) -> list[list[int]]`, a direct enumeration of the
relation's extension. `read_join_extension` is built on that call.
"""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from pyxlog.ilp.join_bodies import JoinBody, read_join_extension

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="xlog engine requires CUDA"
)

_SOURCE = """
    pre_before_post(0, 0). pre_before_post(1, 0).
    pre_before_post(2, 1).
    pre_before_post(4, 2).
    pred pre_before_post(i64, i64).
"""

_JB = JoinBody(
    neural_predicate="saliency", network="sal_net", join_var="Ev",
    relation="pre_before_post", event_arg=0, head_arg=1,
)


def test_extension_is_read_from_the_engine() -> None:
    prog = pyxlog.IlpProgramFactory.compile(_SOURCE, device=0, memory_mb=64)
    prog.evaluate()
    ext = read_join_extension(prog, _JB, num_bindings=4)
    assert ext == [[0, 1], [2], [4], []]     # edge 3 joins nothing -> empty


def test_an_edge_with_no_joined_events_gets_an_empty_extension() -> None:
    prog = pyxlog.IlpProgramFactory.compile(_SOURCE, device=0, memory_mb=64)
    prog.evaluate()
    ext = read_join_extension(prog, _JB, num_bindings=4)
    assert ext[3] == []


def test_a_head_binding_outside_the_range_is_ignored() -> None:
    """The engine's relation may carry tuples whose head binding lies outside the
    supervised range 0..num_bindings-1 (here: edge 2, with num_bindings=2). Those
    tuples must be dropped, not crash and not shift the buckets."""
    prog = pyxlog.IlpProgramFactory.compile(_SOURCE, device=0, memory_mb=64)
    prog.evaluate()
    ext = read_join_extension(prog, _JB, num_bindings=2)
    assert ext == [[0, 1], [2]]     # the (4, 2) tuple is out of range -> ignored


# ---------------------------------------------------------------------------
# Task 3: a neural-JOIN candidate competes inside the multi-candidate mixture
# ---------------------------------------------------------------------------

from pyxlog.ilp.neurosymbolic import (
    NeuroSymbolicTrainingConfig, train_neurosymbolic_program,
)

_EVENT_FEATURES = [0.9, 0.1, 0.2, 0.15, 0.85, 0.1]
_PRE = {0: [0, 1], 1: [2, 3], 2: [4], 3: [5]}
# The distractor is EQUAL-CARDINALITY with pre_before_post (2, 2, 1, 1 events per
# edge) and takes each edge's events from OTHER edges. That is the plan's honest-
# distractor contract, and it is load-bearing, not cosmetic: with the unbalanced
# 1-event-per-edge distractor {0: [5], 1: [4], 2: [1], 3: [0]} the world has TWO
# exact global optima -- pre_before_post with a correct detector, and
# post_before_pre with an INVERTED one (that distractor is a bijection edge->event
# whose features anti-correlate perfectly with the labels, so p(0.1)=1, p(0.85)=0,
# p(0.9)=0 fits it to loss 0.0003). Which optimum SGD lands in is then decided by
# the init, not by the mechanism; measured on this branch, the inverted one won.
# With the balanced distractor no function of the event feature fits it (events 1
# and 5 share the feature 0.1 but would need opposite probabilities), so the
# correct relation + the correct per-event detector is the UNIQUE optimum.
_POST = {0: [2, 3], 1: [0, 1], 2: [5], 3: [4]}


def _world_source() -> str:
    pre = "\n".join(f"    pre_before_post({e}, {k})." for k in sorted(_PRE) for e in _PRE[k])
    post = "\n".join(f"    post_before_pre({e}, {k})." for k in sorted(_POST) for e in _POST[k])
    return f"""
        nn(sal_net, [Event], Label, [low, strengthen]) :: saliency(Event, Label).
{pre}
{post}
        pred pre_before_post(i64, i64).
        pred post_before_pre(i64, i64).
        pred plastic(i64).
        trainable_rule(cand_pre, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E).
        trainable_rule(cand_post, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), post_before_pre(Ev, E).
        train(plastic, binary_cross_entropy).
    """


def _targets() -> list[float]:
    return [1.0 if any(_EVENT_FEATURES[e] > 0.5 for e in _PRE[k]) else 0.0 for k in sorted(_PRE)]


def test_a_stage_b_candidate_trains_in_the_multi_candidate_mixture() -> None:
    """Before this task it died with KeyError 'cand_pre' at neurosymbolic.py:530."""
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[f] for f in _EVENT_FEATURES], dtype=torch.float32)

    result = train_neurosymbolic_program(
        _world_source(),
        networks={"sal_net": net},
        domain_inputs={"sal_net": feats},
        examples=[{"targets": torch.tensor(_targets(), dtype=torch.float32)}],
        config=NeuroSymbolicTrainingConfig(steps=300, learning_rate=0.1),
    )

    # it trained, and gradient reached the PER-EVENT detector
    assert result.losses[-1] < result.losses[0]
    assert result.neural_parameter_grads["sal_net"] > 0.0
    # the correct timing relation won
    w = result.symbolic_rule_weights
    assert w["cand_pre"] > 0.7 and w["cand_post"] < 0.3, w
