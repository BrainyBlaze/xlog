#!/usr/bin/env python
"""In-circuit saliency-driven plasticity rule induction (xlog Stage B).

Recovers a planted STDP-style plasticity rule end-to-end through the xlog
differentiable circuit, using an EXISTENTIAL join between a neural predicate and
an ordinary relation:

    plastic(Edge) :- saliency(Event, strengthen), pre_before_post(Event, Edge).

``saliency`` is a neural predicate over a per-event feature; ``pre_before_post``
is the (deterministic) pre->post timing graph. The join variable ``Event`` is
existential — it does not appear in the head — so the engine grounds ``saliency``
over the REAL event domain inside the circuit and OR-aggregates the per-event
contributions at each edge. Unlike a torch-side per-edge scalar, ``saliency`` is
learned as a generalizable FUNCTION of the event feature, so it also classifies
events it never saw during training.

Planted world: an event is "salient" iff its feature exceeds 0.5; an edge
strengthens iff at least one of its pre->post events is salient.

Run (needs a CUDA build of pyxlog):  python run_demo.py
"""

import torch

from pyxlog.ilp.neurosymbolic import (
    NeuroSymbolicTrainingConfig,
    train_neurosymbolic_program,
)

# Compact planted graph (6 events): the exact d-DNNF compiler builds one circuit
# over all edge queries and its fixed buffer caps around ~6-7 events. Edges 0 and
# 1 each join TWO events (exercising the noisy-OR); edges 2 and 3 join one. The
# saliency net learns a generalizable feature->saliency function (never sees ids).
EVENT_FEATURES = [0.9, 0.1, 0.2, 0.15, 0.85, 0.1]
EDGES = {0: [0, 1], 1: [2, 3], 2: [4], 3: [5]}


def salient(ev: int) -> bool:
    return EVENT_FEATURES[ev] > 0.5


def edge_targets() -> list[float]:
    return [1.0 if any(salient(e) for e in EDGES[edge]) else 0.0 for edge in sorted(EDGES)]


def source() -> str:
    facts = "\n".join(
        f"        pre_before_post({ev}, {edge})."
        for edge in sorted(EDGES)
        for ev in EDGES[edge]
    )
    return f"""
        nn(sal_net, [Event], Label, [low, strengthen]) :: saliency(Event, Label).
{facts}
        pred pre_before_post(i64, i64).
        trainable_rule(rule_plastic, weight=0.0) :: plastic(Edge) :-
            saliency(Event, strengthen), pre_before_post(Event, Edge).
        train(plastic, binary_cross_entropy).
    """


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This demo requires a CUDA build of pyxlog.")

    torch.manual_seed(0)
    # Bias is required so the net can represent the planted threshold (salient iff
    # feature > 0.5); without it, sigmoid(w*feature) > 0.5 for every positive feature.
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    feats = torch.tensor([[f] for f in EVENT_FEATURES], dtype=torch.float32)
    targets = edge_targets()

    print("Planted rule: edge strengthens iff it has a salient (feature>0.5) pre->post event.")
    print(f"Event features : {EVENT_FEATURES}")
    print(f"Edge -> events : {EDGES}")
    print(f"Edge targets   : {[int(t) for t in targets]}\n")

    result = train_neurosymbolic_program(
        source(),
        networks={"sal_net": net},
        domain_inputs={"sal_net": feats},
        examples=[{"targets": torch.tensor(targets, dtype=torch.float32)}],
        config=NeuroSymbolicTrainingConfig(steps=120, learning_rate=0.15),
    )

    with torch.no_grad():
        device = next(net.parameters()).device
        sal = net(feats.to(device))[:, 1].cpu().tolist()

    print(f"loss {result.losses[0]:.4f} -> {result.losses[-1]:.4f}  "
          f"(guard weight learned to {result.symbolic_rule_weights['rule_plastic']:.3f})\n")

    print("Learned per-event P(saliency=strengthen):")
    for ev, f in enumerate(EVENT_FEATURES):
        mark = "salient" if salient(ev) else "       "
        print(f"  event {ev}  feature={f:.2f}  {mark}  P(strengthen)={sal[ev]:.3f}")

    print("\nRecovered rule on training edges (in-circuit OR-aggregated query prob):")
    preds = [1.0 if p >= 0.5 else 0.0 for p in result.query_probabilities]
    for edge in sorted(EDGES):
        ok = "OK " if preds[edge] == targets[edge] else "XX "
        print(f"  {ok} plastic({edge})  P={result.query_probabilities[edge]:.3f}  "
              f"pred={int(preds[edge])}  true={int(targets[edge])}  events={EDGES[edge]}")

    # Generalization: classify events with feature values never seen in training.
    with torch.no_grad():
        held = torch.tensor([[0.7], [0.05]], dtype=torch.float32).to(device)
        held_sal = net(held)[:, 1].cpu().tolist()
    print("\nGeneralization to unseen feature values (saliency is a learned function, not an id lookup):")
    print(f"  unseen salient feature 0.70 -> P(strengthen)={held_sal[0]:.3f}  (expect > 0.5)")
    print(f"  unseen quiet   feature 0.05 -> P(strengthen)={held_sal[1]:.3f}  (expect < 0.5)")

    recovered = preds == targets and held_sal[0] > 0.5 and held_sal[1] < 0.5
    print("\n" + ("RULE RECOVERED ✓" if recovered else "rule NOT recovered ✗"))


if __name__ == "__main__":
    main()
