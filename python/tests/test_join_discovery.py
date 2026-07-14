"""FLAGSHIP. No candidate rule is hand-written: the candidate set is SWEPT from a
relation vocabulary (``build_join_candidates``), so the system is handed three
same-head rules that differ ONLY in which relation joins the existential event
variable to the head, and it has to pick the one that is true -- while learning, from
scratch, a per-EVENT detector that generalizes to feature values it never saw.

THE ONE THING THAT IS TUNED, STATED UP FRONT. The detector's initial logit for
``strengthen`` is shifted DOWN by :data:`QUIET_PRIOR_BIAS` (see there for why, and for
the measured numbers with and without it). That is an INITIALIZATION, not an
assertion, and it is not a free parameter fished for: it is the prior that events are
mostly quiet, which is true of this world by construction (a positive edge has one
salient event out of k). Everything else -- world, candidates, steps, lr -- is as the
plan specifies. The bare mechanism's numbers are measured and printed by the
saturation test below; nothing here is silently tuned.
"""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from pyxlog.ilp.discovery import (
    CORRECT_RELATION,
    NETWORK,
    NEURAL_PREDICATE,
    POSITIVE_LABEL,
    build_join_candidates,
    make_world,
    select_rule,
)
from pyxlog.ilp.neurosymbolic import (
    NeuroSymbolicTrainingConfig,
    _read_only_source,
    train_neurosymbolic_program,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="xlog engine requires CUDA"
)

VOCAB = ["pre_before_post", "post_before_pre", "co_occurs"]
WINNER = f"cand_{CORRECT_RELATION}"

# The head-bound-gate variant (variant A: ONE learned gate on a POOLED per-edge
# feature) provably caps at 0.847 accuracy on this world shape -- a positive edge
# carries exactly ONE salient event among `events_per_edge`, and any pooled per-edge
# feature dilutes it. A per-event detector under the join's OR has no such ceiling.
# That number is the baseline every accuracy in this file is measured against.
HEAD_GATE_CEILING = 0.847

# THE MITIGATION, and it is a real one -- reported, not hidden.
#
# The noisy-OR SATURATES. A binding with k joined events at per-event probability p has
# mask 1-(1-p)^k; at the default init p ~ 0.5, so at k=6 EVERY binding starts at 0.984
# and, once the three candidates are or-ed together, the head starts at 0.87 against a
# base rate of 0.375. Every binding already says "true", the gradient through
# 1-exp(sum log(1-p)) is vanishing, and the cheapest descent is to kill the detector
# outright -- p -> 0 for salient events too. That is a genuine DEGENERATE MINIMUM, not
# slow convergence: measured at seed 0, k=6, it sits at loss 0.640 (the entropy of the
# base rate) at 1500, 3000, 6000 AND 12000 steps, and hardens the WRONG candidate to
# weight 1.0. Raising `steps` does not rescue it.
#
# Shifting the detector's initial logit for `strengthen` down by 2.0 -- the prior that
# events are mostly quiet, which this world satisfies by construction -- starts the OR
# unsaturated and removes the basin. Measured over 5 seeds at n_edges=40:
#
#     events/edge     1     2     4     6     8    16
#     bare      :   5/5   5/5   4/5   3/5   3/5   4/5   seeds discovering the rule
#     with prior:   5/5   5/5   5/5   5/5   4/5   5/5
#
# It is an initialization, not an assertion, and it encodes a fact about the world
# rather than a fact about the answer.
QUIET_PRIOR_BIAS = -2.0


def _positive_index(source: str, config: NeuroSymbolicTrainingConfig) -> int:
    """Which output column is ``strengthen`` is the ENGINE's answer, not a hardcoded 1."""
    reader = pyxlog.Program.compile(
        _read_only_source(source), device=config.device, memory_mb=config.gpu_memory_mb
    )
    return int(reader.label_to_index(NEURAL_PREDICATE, POSITIVE_LABEL))


def _run(
    n_edges: int,
    events_per_edge: int,
    seed: int,
    steps: int = 1500,
    learning_rate: float = 0.05,
    output_bias: float = QUIET_PRIOR_BIAS,
):
    """Sweep the candidates, generate the world, train. ``output_bias=0.0`` is the BARE
    mechanism (no quiet-event prior); the default is the mitigated one."""
    world = make_world(n_edges=n_edges, events_per_edge=events_per_edge, seed=seed)
    candidates, _ids = build_join_candidates(VOCAB)
    source = world.facts() + "\n" + candidates
    config = NeuroSymbolicTrainingConfig(steps=steps, learning_rate=learning_rate)
    positive = _positive_index(source, config)

    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    if output_bias:
        with torch.no_grad():
            net[0].bias[positive] += output_bias

    features = torch.tensor([[f] for f in world.event_features], dtype=torch.float32)
    targets = torch.tensor(
        [1.0 if y else 0.0 for y in world.labels], dtype=torch.float32
    )
    result = train_neurosymbolic_program(
        source,
        networks={NETWORK: net},
        # Events are the dense range 0..D-1, so row e holds event e. It is stated
        # anyway: the row<->constant correspondence is the caller's to declare, never
        # something the trainer should have to infer from an ordering.
        domain_inputs={NETWORK: features},
        domain_ids={NETWORK: list(range(len(world.event_features)))},
        examples=[{"targets": targets}],
        config=config,
    )
    return world, net, result, positive


def _accuracy(result, world) -> float:
    return sum(
        (p >= 0.5) == y for p, y in zip(result.query_probabilities, world.labels)
    ) / len(world.labels)


def _winner(result) -> str | None:
    """The SELECTED rule, or None when the mixture is not entitled to name one.

    Deliberately not ``max(weights, key=weights.get)``: that returns the first key
    holding the maximum, so on two indistinguishable relations it silently reports
    whichever the caller typed first. ``select_rule`` refuses instead, and every
    assertion in this file is therefore an assertion that a rule was actually DECIDED --
    see test_join_identifiability for the worlds where it is not.
    """
    return select_rule(result.symbolic_rule_weights).rule


# ---------------------------------------------------------------------------
# TASK 6 -- the rule is DISCOVERED, not prompted
# ---------------------------------------------------------------------------


def test_discovers_the_correct_join_relation_unaided() -> None:
    """Three candidates, one per relation of the vocabulary, NONE written by hand.

    The mixture must put its mass on the relation whose join extension actually
    carries the planted signal. The two distractors are equal-cardinality (the same
    number of events per edge as the correct relation, drawn from OTHER edges), so
    they have exactly as sharp an OR and carry no label information: there is nothing
    to win on but the truth.
    """
    _world, _net, result, _pos = _run(n_edges=40, events_per_edge=6, seed=0)
    weights = result.symbolic_rule_weights
    print(f"\n[discovery] candidate weights = {weights}")

    assert _winner(result) == WINNER, weights
    assert weights["cand_pre_before_post"] > 0.7, weights
    assert weights["cand_post_before_pre"] < 0.3, weights
    assert weights["cand_co_occurs"] < 0.3, weights


def test_the_detector_is_learned_per_event_and_generalizes() -> None:
    """The head-bound-gate baseline (variant A) provably caps at 0.847 accuracy on this
    world shape: a positive edge carries exactly ONE salient event out of six, which any
    pooled per-edge feature dilutes. The per-event detector under the join's OR has no
    such ceiling.

    The second half is what proves the detector is a FUNCTION OF THE FEATURE and not a
    lookup table over event ids: it must classify feature values the world never
    contained. Two of the three probes (0.95 and 0.005) are asserted to lie strictly
    OUTSIDE the world's observed range, so they are not merely absent from the training
    set -- they are outside its support, and getting them right is extrapolation, not
    interpolation.
    """
    world, net, result, positive = _run(n_edges=40, events_per_edge=6, seed=0)

    accuracy = _accuracy(result, world)
    print(
        f"\n[detector] accuracy = {accuracy:.3f} "
        f"(head-gate ceiling on this shape is {HEAD_GATE_CEILING})"
    )
    assert accuracy > 0.95, (
        f"accuracy {accuracy:.3f} (head-gate ceiling on this shape is "
        f"{HEAD_GATE_CEILING})"
    )

    lo, hi = min(world.event_features), max(world.event_features)
    assert 0.95 > hi and 0.005 < lo, (lo, hi)      # strictly out of support
    probes = [0.95, 0.02, 0.005]
    with torch.no_grad():
        device = next(net.parameters()).device
        unseen = net(
            torch.tensor([[v] for v in probes], dtype=torch.float32).to(device)
        )[:, positive].cpu()
    for value, p in zip(probes, unseen.tolist()):
        print(f"[detector] P(strengthen | feature={value}) = {p:.4f}")
    assert float(unseen[0]) > 0.5, unseen.tolist()      # 0.95: above every value seen
    assert float(unseen[1]) < 0.5, unseen.tolist()      # 0.02: quiet
    assert float(unseen[2]) < 0.5, unseen.tolist()      # 0.005: below every value seen


def test_discovery_is_stable_across_seeds() -> None:
    """One lucky seed is an anecdote. (Measured: 5/5 with the quiet prior, 3/5 without
    -- see QUIET_PRIOR_BIAS and the saturation test.)"""
    wins = 0
    for seed in range(5):
        world, _net, result, _pos = _run(n_edges=40, events_per_edge=6, seed=seed)
        won = _winner(result) == WINNER
        wins += won
        weights = {k: round(v, 3) for k, v in result.symbolic_rule_weights.items()}
        print(
            f"\n[seed {seed}] winner={_winner(result)} "
            f"accuracy={_accuracy(result, world):.3f} weights={weights}"
        )
    print(f"\n[stability] {wins}/5 seeds discovered {WINNER}")
    assert wins >= 4, f"only {wins}/5 seeds discovered the correct relation"


# ---------------------------------------------------------------------------
# TASK 7 -- where the mechanism BREAKS. A measurement, not a pass/fail test.
# ---------------------------------------------------------------------------

# k=6 is in the sweep because it is in the PUBLISHED table (README and
# docs/architecture/dilp-training.md both carry a k=6 row) and because it is what the
# demo and the flagship run at. A published number no test reproduces is a number
# nobody can check.
SATURATION_K = (1, 2, 4, 6, 8, 16)
SATURATION_SEEDS = (0, 1, 2)


def _sweep(output_bias: float) -> dict[int, tuple[int, float]]:
    """k -> (how many seeds discovered the rule, mean accuracy).

    A SINGLE seed per k would be worthless here and would have hidden the finding: the
    failure is a basin of attraction, so at any fixed k some inits fall into it and
    others do not. Seed 0 alone reports 1.000/True at every k in this sweep even for
    the bare mechanism -- and the bare mechanism is nevertheless broken at k>=4. The
    quantity that degrades is the RATE.
    """
    table: dict[int, tuple[int, float]] = {}
    for k in SATURATION_K:
        accuracies: list[float] = []
        wins = 0
        for seed in SATURATION_SEEDS:
            world, _net, result, _pos = _run(
                n_edges=40, events_per_edge=k, seed=seed, output_bias=output_bias
            )
            accuracy = _accuracy(result, world)
            correct = _winner(result) == WINNER
            accuracies.append(accuracy)
            wins += correct
            print(
                f"  events/edge={k:>2} seed={seed}  accuracy={accuracy:.3f}  "
                f"correct_rule={correct}  "
                f"w({WINNER})={result.symbolic_rule_weights[WINNER]:.3f}"
            )
        table[k] = (wins, sum(accuracies) / len(accuracies))
    return table


def test_report_the_noisy_or_saturation_limit() -> None:
    """As events-per-edge grows the OR saturates: with k events at probability p the
    mask is 1-(1-p)^k, so at the p~0.5 default init EVERY binding starts at ~1 and the
    gradient to the detector vanishes. The optimizer's cheapest descent is then to kill
    the detector, and it lands in a degenerate minimum that MORE STEPS DO NOT ESCAPE
    (seed 0, k=6, bare: loss stuck at 0.640 -- the base-rate entropy -- at 1500, 3000,
    6000 and 12000 steps, with the WRONG candidate hardened to 1.0).

    This test finds where that breaks and REPORTS it. Both configurations are run and
    both are printed: the BARE mechanism, and the mechanism with the quiet-event prior
    (QUIET_PRIOR_BIAS). Nothing is silently tuned.

    THE FINDING (5 seeds, n_edges=40; this test re-measures 3 of them):

        events/edge      1     2     4     6     8    16
        bare       :   5/5   5/5   4/5   3/5   3/5   4/5
        with prior :   5/5   5/5   5/5   5/5   4/5   5/5

    The bare mechanism starts losing seeds at k=4 and is a coin-flip by k=6. The prior
    buys back the whole k<=6 range and most of k=8..16, but does not make the failure
    go away: at k=8 one seed in five still inverts, and at k=16 one seed discovers the
    correct relation yet never converges its detector (accuracy 0.60). The residual is
    reported, not asserted away.

    Only the regime we can stand behind is asserted (k = 1, 2, 4, WITH the prior).
    """
    print("\n=== noisy-OR saturation, BARE mechanism (output_bias=0.0) ===")
    bare = _sweep(output_bias=0.0)

    print(
        f"\n=== noisy-OR saturation, WITH the quiet-event prior "
        f"(output_bias={QUIET_PRIOR_BIAS}) ==="
    )
    mitigated = _sweep(output_bias=QUIET_PRIOR_BIAS)

    n = len(SATURATION_SEEDS)
    print(f"\n{'k':>3} | {'bare: found':>12} {'mean acc':>9} | {'prior: found':>13} {'mean acc':>9}")
    for k in SATURATION_K:
        bw, ba = bare[k]
        mw, ma = mitigated[k]
        print(f"{k:>3} | {f'{bw}/{n}':>12} {ba:>9.3f} | {f'{mw}/{n}':>13} {ma:>9.3f}")
    print(f"\nSATURATION REPORT (for the docs): bare={bare} with_prior={mitigated}")

    # k=1..4 must hold WITH the prior; beyond that we RECORD rather than demand.
    for k in (1, 2, 4):
        wins, accuracy = mitigated[k]
        assert wins == n and accuracy > 0.95, f"regression at k={k}: {mitigated[k]}"
