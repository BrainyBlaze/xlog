"""Flagship: the in-circuit existential-join rule recovers a planted
saliency-driven plasticity rule end-to-end through the differentiable circuit.

Planted world. Each event carries a 1-D *saliency feature*; an event is
"salient" iff its feature exceeds 0.5. An edge *strengthens* iff at least one of
its pre->post events is salient. The plasticity rule

    plastic(Edge) :- saliency(Event, strengthen), pre_before_post(Event, Edge).

joins the neural predicate ``saliency`` to the ordinary ``pre_before_post``
relation on the EXISTENTIAL variable ``Event``: the engine grounds ``saliency``
over the real event domain inside the circuit and OR-aggregates the per-event
contributions at each edge (Stage B). Training recovers the planted rule — and,
crucially, ``saliency`` is learned as a generalizable FUNCTION of the per-event
feature (not an id lookup), so it classifies unseen events too.
"""

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from pyxlog.ilp.neurosymbolic import (
    NeuroSymbolicTrainingConfig,
    train_neurosymbolic_program,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="xlog engine requires CUDA"
)

# events 0..5; feature > 0.5 == salient. Events are edge-disjoint and the world
# is kept compact (6 events): the exact d-DNNF compiler builds one circuit over
# ALL edge queries and its fixed buffer caps around ~6-7 events, so a larger
# planted graph overflows it ("CNF too large to compile"). Edges 0 and 1 each
# join TWO events (exercising the noisy-OR aggregation); edges 2 and 3 join one.
# The saliency net never sees event ids — it learns a generalizable
# feature->saliency function, which the held-out check below confirms.
_EVENT_FEATURES = [0.9, 0.1, 0.2, 0.15, 0.85, 0.1]
# edge -> events joined to it by pre_before_post (edges 0..3, row-aligned with targets)
_EDGES = {0: [0, 1], 1: [2, 3], 2: [4], 3: [5]}


def _salient(ev: int) -> bool:
    return _EVENT_FEATURES[ev] > 0.5


def _edge_targets() -> list[float]:
    return [
        1.0 if any(_salient(e) for e in _EDGES[edge]) else 0.0
        for edge in sorted(_EDGES)
    ]


def _sal_net():
    # Bias is required: the planted saliency is a THRESHOLD on the feature
    # (salient iff feature > 0.5). Without a bias, P(strengthen)=sigmoid(w*feature)
    # exceeds 0.5 for every positive feature and cannot represent the threshold.
    return torch.nn.Sequential(
        torch.nn.Linear(1, 2, bias=True),
        torch.nn.Softmax(dim=-1),
    )


def _source() -> str:
    facts = "\n".join(
        f"        pre_before_post({ev}, {edge})."
        for edge in sorted(_EDGES)
        for ev in _EDGES[edge]
    )
    return f"""
        nn(sal_net, [Event], Label, [low, strengthen]) :: saliency(Event, Label).
{facts}
        pred pre_before_post(i64, i64).
        trainable_rule(rule_plastic, weight=0.0) :: plastic(Edge) :-
            saliency(Event, strengthen), pre_before_post(Event, Edge).
        train(plastic, binary_cross_entropy).
    """


def test_incircuit_plasticity_recovers_planted_rule() -> None:
    torch.manual_seed(0)
    net = _sal_net()
    feats = torch.tensor([[f] for f in _EVENT_FEATURES], dtype=torch.float32)
    targets = _edge_targets()

    result = train_neurosymbolic_program(
        _source(),
        networks={"sal_net": net},
        domain_inputs={"sal_net": feats},
        examples=[{"targets": torch.tensor(targets, dtype=torch.float32)}],
        config=NeuroSymbolicTrainingConfig(steps=120, learning_rate=0.15),
    )

    # 1. The rule trains end-to-end: loss falls and gradient reaches the net.
    assert result.losses[-1] < result.losses[0]
    assert result.neural_parameter_grads["sal_net"] > 0.0
    assert result.symbolic_weight_grads["rule_plastic"] > 0.0

    # 2. The planted rule is recovered: every edge is classified correctly by the
    #    in-circuit OR-aggregated query probability.
    preds = [1.0 if p >= 0.5 else 0.0 for p in result.query_probabilities]
    assert preds == targets, f"preds={preds} targets={targets} probs={result.query_probabilities}"

    # 3. saliency is a GENERALIZABLE function of the feature, not an id lookup:
    #    unseen events with novel feature values are classified correctly.
    with torch.no_grad():
        device = next(net.parameters()).device
        held = torch.tensor([[0.7], [0.05]], dtype=torch.float32).to(device)  # unseen feature values
        sal_strengthen = net(held)[:, 1].cpu()
    assert sal_strengthen[0] > 0.5  # salient held-out event -> strengthen
    assert sal_strengthen[1] < 0.5  # non-salient held-out event -> low
