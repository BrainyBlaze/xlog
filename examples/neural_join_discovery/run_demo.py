#!/usr/bin/env python
"""Neural join bodies in the joint mixture: the rule is DISCOVERED, not written.

A relation VOCABULARY goes in. One same-head Stage-B candidate per relation comes
out, and they all compete in one joint mixture:

    plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E).
    plastic(E) :- saliency(Ev, strengthen), post_before_pre(Ev, E).
    plastic(E) :- saliency(Ev, strengthen), co_occurs(Ev, E).

Nobody writes any of them by hand: `build_join_candidates` sweeps them out of the
vocabulary. Each candidate's mask is the OR, over the join extension READ FROM THE
ENGINE, of the network's PER-EVENT probability:

    mask_k[edge] = 1 - PROD_{e : r_k(e, edge)} (1 - p_saliency(e))

The system has to put its mass on the relation whose join extension actually carries
the planted signal, while learning -- from scratch -- a per-EVENT detector of that
signal that generalizes to feature values the world never contained.

Planted world: an edge is plastic iff SOME of its pre->post events is salient
(feature > 0.5). A positive edge carries EXACTLY ONE salient event out of six, which
is precisely the shape a head-bound gate on a pooled per-edge feature cannot recover
(the lone salient event dilutes; that baseline provably caps at 0.847 accuracy here).

Run (needs a CUDA build of pyxlog):  python run_demo.py
"""

import torch

from pyxlog.ilp.discovery import (
    CORRECT_RELATION,
    NETWORK,
    POSITIVE_LABEL,
    build_join_candidates,
    make_world,
)
from pyxlog.ilp.neurosymbolic import (
    NeuroSymbolicTrainingConfig,
    train_neurosymbolic_program,
)

VOCABULARY = ["pre_before_post", "post_before_pre", "co_occurs"]
N_EDGES = 40
EVENTS_PER_EDGE = 6
SEED = 0

# `nn(sal_net, [Event], Label, [low, strengthen])` -- the positive label is column 1.
POSITIVE_COLUMN = 1

# The head-bound-gate baseline (ONE learned gate on a POOLED per-edge feature) provably
# caps at this accuracy on this world shape: a positive edge carries exactly one salient
# event among six, and any pooled per-edge feature dilutes it. The per-event detector
# under the join's OR has no such ceiling. Every accuracy below is read against it.
HEAD_GATE_CEILING = 0.847

# The detector's initial logit for `strengthen`, shifted down. THE PRIOR THAT EVENTS ARE
# MOSTLY QUIET -- true of this world by construction. It is load-bearing and it is an
# INITIALIZATION, not an assertion: without it the noisy-OR is already saturated at init
# (six events at p~0.5 start every binding at 0.984), the gradient is flat, and the
# optimizer lands in a degenerate inverted minimum that more steps do not escape. See
# the README's HONEST SCOPE section for the measured saturation table.
QUIET_PRIOR_BIAS = -2.0


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This demo requires a CUDA build of pyxlog.")

    world = make_world(n_edges=N_EDGES, events_per_edge=EVENTS_PER_EDGE, seed=SEED)
    candidates, candidate_ids = build_join_candidates(VOCABULARY)
    source = world.facts() + "\n" + candidates

    print("Planted rule : plastic(E) :- saliency(Ev, strengthen), "
          f"{CORRECT_RELATION}(Ev, E).")
    print(f"               (an event is salient iff its feature > 0.5; a positive edge "
          f"has exactly ONE\n                salient event out of {EVENTS_PER_EDGE})")
    print(f"Vocabulary   : {VOCABULARY}")
    print(f"World        : {N_EDGES} edges, {len(world.event_features)} events, "
          f"{sum(world.labels)} of {N_EDGES} edges plastic")
    print("\ncandidates: NONE hand-written — generated from the vocabulary")
    for rule_id in candidate_ids:
        relation = rule_id[len("cand_"):]
        print(f"  {rule_id:<24} plastic(E) :- saliency(Ev, strengthen), "
              f"{relation}(Ev, E).")
    print("\nThe distractors are equal-cardinality (the same number of events per edge as")
    print("the correct relation, drawn from OTHER edges): their OR is exactly as sharp and")
    print("they carry no label information. There is nothing to win on but the truth.\n")

    torch.manual_seed(SEED)
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    with torch.no_grad():
        net[0].bias[POSITIVE_COLUMN] += QUIET_PRIOR_BIAS

    features = torch.tensor([[f] for f in world.event_features], dtype=torch.float32)
    targets = torch.tensor([1.0 if y else 0.0 for y in world.labels], dtype=torch.float32)

    result = train_neurosymbolic_program(
        source,
        networks={NETWORK: net},
        domain_inputs={NETWORK: features},
        # Which CONSTANT each row holds. Events are the dense range here, so this is the
        # default -- it is stated anyway, because the row<->constant map is the caller's
        # to declare, never something the trainer should infer from an ordering.
        domain_ids={NETWORK: list(range(len(world.event_features)))},
        examples=[{"targets": targets}],
        config=NeuroSymbolicTrainingConfig(steps=1500, learning_rate=0.05),
    )

    weights = result.symbolic_rule_weights
    discovered = max(weights, key=weights.get)
    print(f"loss {result.losses[0]:.4f} -> {result.losses[-1]:.4f}\n")

    print("Candidate weights in the joint mixture:")
    for rule_id in candidate_ids:
        mark = "<-- DISCOVERED" if rule_id == discovered else ""
        print(f"  {rule_id:<24} weight={weights[rule_id]:.5f}  {mark}")
    relation = discovered[len("cand_"):]
    print(f"\nDiscovered rule: plastic(E) :- saliency(Ev, strengthen), {relation}(Ev, E).")
    print(f"  planted rule joined on '{CORRECT_RELATION}' -> "
          f"{'CORRECT' if relation == CORRECT_RELATION else 'WRONG'}\n")

    device = next(net.parameters()).device
    with torch.no_grad():
        saliency = net(features.to(device))[:, POSITIVE_COLUMN].cpu().tolist()

    salient = [p for f, p in zip(world.event_features, saliency) if f > 0.5]
    quiet = [p for f, p in zip(world.event_features, saliency) if f <= 0.5]
    print("Learned per-event P(saliency=strengthen) — one detector, "
          f"{len(world.event_features)} events:")
    print(f"  planted-salient events (feature > 0.5), n={len(salient):<3} "
          f"mean P={sum(salient) / len(salient):.4f}  min P={min(salient):.4f}")
    print(f"  planted-quiet   events (feature <= 0.5), n={len(quiet):<3} "
          f"mean P={sum(quiet) / len(quiet):.4f}  max P={max(quiet):.4f}")
    print("  first eight events:")
    for event in range(8):
        feature = world.event_features[event]
        mark = "salient" if feature > 0.5 else "       "
        print(f"    event {event:<3} feature={feature:.3f}  {mark}  "
              f"P(strengthen)={saliency[event]:.4f}")

    lo, hi = min(world.event_features), max(world.event_features)
    probes = [0.95, 0.02, 0.005]
    with torch.no_grad():
        unseen = net(torch.tensor([[v] for v in probes], dtype=torch.float32).to(device))
        unseen = unseen[:, POSITIVE_COLUMN].cpu().tolist()
    print("\nGeneralization — the detector is a FUNCTION of the feature, not an id lookup.")
    print(f"  the world's features span [{lo:.3f}, {hi:.3f}]; 0.95 and 0.005 lie strictly")
    print("  OUTSIDE that support, so classifying them is extrapolation, not interpolation:")
    for value, p in zip(probes, unseen):
        expect = "expect > 0.5" if value > 0.5 else "expect < 0.5"
        print(f"    feature={value:<6} P(strengthen)={p:.4f}  ({expect})")

    correct = sum(
        (p >= 0.5) == y for p, y in zip(result.query_probabilities, world.labels)
    )
    accuracy = correct / len(world.labels)
    print(f"\nAccuracy vs the planted labels: {accuracy:.3f} "
          f"({correct}/{len(world.labels)} edges)")
    print(f"  head-gate baseline ceiling on this world shape: {HEAD_GATE_CEILING}")
    print(f"  (ONE gate on a pooled per-edge feature cannot beat {HEAD_GATE_CEILING} here; "
          "a per-event\n   detector under the join's OR has no such ceiling)")

    recovered = relation == CORRECT_RELATION and accuracy > 0.95 and unseen[0] > 0.5
    print("\n" + ("RULE DISCOVERED ✓" if recovered else "rule NOT discovered ✗"))


if __name__ == "__main__":
    main()
