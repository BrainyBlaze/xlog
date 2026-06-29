# In-circuit saliency-driven plasticity rule induction (Stage B)

This flagship recovers a planted STDP-style **plasticity rule** end-to-end through
the xlog differentiable circuit, using an **existential join** between a neural
predicate and an ordinary relation:

```
plastic(Edge) :- saliency(Event, strengthen), pre_before_post(Event, Edge).
```

`saliency` is a neural predicate over a per-event feature; `pre_before_post` is
the deterministic pre→post timing graph. The join variable `Event` is
**existential** — it does not appear in the head — so the engine grounds
`saliency` over the **real event domain inside the circuit** and OR-aggregates
the per-event contributions at each edge. This is the capability Stage B adds:
before it, an existential-join trainable body failed closed.

Why it matters: `saliency` is learned as a **generalizable function of the event
feature**, not a per-edge scalar or an id lookup. The same learned predicate
classifies events the trainer never saw — exactly what a per-edge torch-side gate
cannot do.

## Planted world

An event is *salient* iff its feature exceeds `0.5`; an edge *strengthens* iff at
least one of its pre→post events is salient. The graph is kept compact (6 events):
the exact d-DNNF compiler builds a single circuit over all edge queries and its
fixed buffer caps around ~6–7 events, so a larger planted graph overflows it.
Edges 0 and 1 each join **two** events (exercising the noisy-OR); edges 2 and 3
join one.

## Run

Requires a CUDA build of `pyxlog`.

```
python run_demo.py
```

## Expected output (verified, RTX 3090)

```
Planted rule: edge strengthens iff it has a salient (feature>0.5) pre->post event.
Event features : [0.9, 0.1, 0.2, 0.15, 0.85, 0.1]
Edge -> events : {0: [0, 1], 1: [2, 3], 2: [4], 3: [5]}
Edge targets   : [1, 0, 1, 0]

loss 0.7104 -> 0.0088  (guard weight learned to 0.997)

Learned per-event P(saliency=strengthen):
  event 0  feature=0.90  salient  P(strengthen)=0.996
  event 1  feature=0.10           P(strengthen)=0.002
  event 2  feature=0.20           P(strengthen)=0.010
  event 3  feature=0.15           P(strengthen)=0.005
  event 4  feature=0.85  salient  P(strengthen)=0.992
  event 5  feature=0.10           P(strengthen)=0.002

Recovered rule on training edges (in-circuit OR-aggregated query prob):
  OK  plastic(0)  P=0.993  pred=1  true=1  events=[0, 1]
  OK  plastic(1)  P=0.014  pred=0  true=0  events=[2, 3]
  OK  plastic(2)  P=0.989  pred=1  true=1  events=[4]
  OK  plastic(3)  P=0.002  pred=0  true=0  events=[5]

Generalization to unseen feature values (saliency is a learned function, not an id lookup):
  unseen salient feature 0.70 -> P(strengthen)=0.933  (expect > 0.5)
  unseen quiet   feature 0.05 -> P(strengthen)=0.001  (expect < 0.5)

RULE RECOVERED ✓
```

The regression/recovery contract is pinned by
`python/tests/test_plasticity_incircuit.py` (CUDA-gated).

## Notes

- The saliency net **must have a bias**: the planted saliency is a threshold on
  the feature, and `sigmoid(w·feature)` without a bias exceeds 0.5 for every
  positive feature, so it cannot represent the threshold.
- Events are edge-disjoint here; sharing an event across edges entangles all
  queries into one larger d-DNNF and shrinks the compilable graph size further.
- Structure (`pre_before_post`) is deterministic and receives no gradient; only
  the `saliency` network and the rule guard are trained.
