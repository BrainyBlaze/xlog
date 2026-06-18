"""Mixed trainable-rule bodies: a neural predicate joined with ordinary relations.

A ``trainable_rule`` body may join a neural predicate with ordinary world
relations (and builtins). The intended semantics:
  - fact atoms (ordinary relations) are HARD join conditions;
  - probability mass comes ONLY from nn-predicates x sigma(w);
  - gradients flow to the network and the rule weight, NEVER through fact atoms.

These tests pin that contract: an ordinary relation in a trainable body acts as
a membership filter on which groundings can fire, contributing no probability
and no gradient.
"""

import pytest

torch = pytest.importorskip("torch")

from pyxlog.ilp.neurosymbolic import (  # noqa: E402
    NeuralBodySpec,
    NeuroSymbolicTrainingConfig,
    _collect_examples,
    _desugar_source,
    _make_rule_weight_module,
    _TENSOR_SOURCE_NAME,
    evaluate_joint_mixture,
    train_neurosymbolic_program,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="xlog engine requires CUDA"
)

# A neural predicate joined with an ordinary relation `allowed` (a hard
# membership filter) inside one trainable rule.
MIXED_BODY_SOURCE = """
    allowed(0).
    allowed(2).
    pred allowed(i64).
    nn(root_net, [Case], Label, [negative, positive]) :: neural_root(Case, Label).
    trainable_rule(rule_mixed, weight=0.0) :: root_case(Case) :-
        neural_root(Case, positive), allowed(Case).
    train(root_case, binary_cross_entropy).
"""


def _root_net(initial=((0.0,), (0.05,))):
    network = torch.nn.Sequential(
        torch.nn.Linear(1, 2, bias=False),
        torch.nn.Softmax(dim=-1),
    )
    with torch.no_grad():
        network[0].weight.copy_(torch.tensor(initial, dtype=torch.float32))
    return network


def _examples():
    return [
        {
            "inputs": torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32),
            "targets": torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32),
        }
    ]


@requires_cuda
def test_mixed_body_compiles_and_trains() -> None:
    """A neural predicate joined with an ordinary relation must compile + train,
    with gradients reaching both the network and the rule weight."""
    network = _root_net()
    result = train_neurosymbolic_program(
        MIXED_BODY_SOURCE,
        networks={"root_net": network},
        examples=_examples(),
        config=NeuroSymbolicTrainingConfig(steps=4, learning_rate=0.2),
    )
    assert result.neural_parameter_grads["root_net"] > 0.0
    assert result.symbolic_weight_grads["rule_mixed"] > 0.0


@requires_cuda
def test_ordinary_relation_is_a_hard_join_not_a_probability_source() -> None:
    """`allowed` gates which cases can fire (hard filter); it contributes no
    probability mass. Cases 1 and 3 are NOT in `allowed`, so root_case is false
    for them regardless of the network — query probability ~0 there."""
    network = _root_net()
    result = train_neurosymbolic_program(
        MIXED_BODY_SOURCE,
        networks={"root_net": network},
        examples=_examples(),
        config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.2),
    )
    probs = result.query_probabilities
    assert probs[1] == pytest.approx(0.0, abs=1e-6)
    assert probs[3] == pytest.approx(0.0, abs=1e-6)


@requires_cuda
def test_derived_hard_condition_fails_loud_not_silent() -> None:
    """A hard condition checked against ground facts only must reject a DERIVED
    relation with a typed error, never silently filter every grounding to 0
    (which would corrupt training)."""
    source = """
        base(0).
        base(2).
        pred base(i64).
        pred contested(i64).
        contested(X) :- base(X).
        nn(root_net, [Case], Label, [negative, positive]) :: neural_root(Case, Label).
        trainable_rule(rule_mixed, weight=0.0) :: root_case(Case) :-
            neural_root(Case, positive), contested(Case).
        train(root_case, binary_cross_entropy).
    """
    with pytest.raises(Exception, match="(?i)derived"):
        train_neurosymbolic_program(
            source,
            networks={"root_net": _root_net()},
            examples=_examples(),
            config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.1),
        )


def test_optimizer_config_defaults_to_adam_and_is_selectable() -> None:
    """The training optimizer is configurable and defaults to Adam (SGD stalls on
    the multiplicative-loss plateau; see the separation test)."""
    from pyxlog.ilp.neurosymbolic import _make_optimizer

    assert NeuroSymbolicTrainingConfig().optimizer == "adam"
    params = [torch.zeros(1, requires_grad=True)]
    assert type(_make_optimizer("adam", params, 0.1)).__name__ == "Adam"
    assert type(_make_optimizer("sgd", params, 0.1)).__name__ == "SGD"
    with pytest.raises(ValueError, match="(?i)optimizer"):
        _make_optimizer("rmsprop", params, 0.1)


@requires_cuda
def test_default_adam_separates_linearly_separable_classes() -> None:
    """With the Adam default the training surface escapes the plateau and actually
    LEARNS: a cleanly linearly separable signal (positives low feature, negatives
    high) trains so every positive ranks well above every negative. Plain SGD
    stalled at the uniform-output floor on this exact problem (~1/10 inits); Adam
    separates (~8/10). This pins that the default optimizer makes the surface
    trainable, not just gradient-flowing."""
    torch.manual_seed(1)
    source = """
        nn(root_net, [Case], Label, [negative, positive]) :: neural_root(Case, Label).
        trainable_rule(rule_sep, weight=0.0) :: root_case(Case) :- neural_root(Case, positive).
        train(root_case, binary_cross_entropy).
    """
    net = torch.nn.Sequential(torch.nn.Linear(1, 2, bias=True), torch.nn.Softmax(dim=-1))
    examples = [
        {
            "inputs": torch.tensor(
                [[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]], dtype=torch.float32
            ),
            "targets": torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        }
    ]
    result = train_neurosymbolic_program(
        source,
        networks={"root_net": net},
        examples=examples,
        config=NeuroSymbolicTrainingConfig(steps=400, learning_rate=0.1),
    )
    probs = result.query_probabilities
    # Clean separation with a wide margin (positives ~1, negatives ~0).
    assert min(probs[:3]) > max(probs[3:]) + 0.5


def _build_mixed_program():
    """Compile MIXED_BODY_SOURCE and register its networks exactly as the
    wrapper does, returning (program, root_net, queries, expected). lr=0 so the
    program is a fixed point we can evaluate twice without weights drifting."""
    import pyxlog

    program_source, rules, train_head, _objective = _desugar_source(MIXED_BODY_SOURCE)
    inputs, targets = _collect_examples(_examples())

    program = pyxlog.Program.compile(program_source, device=0, memory_mb=4096)
    root_net = _root_net().cuda()
    program.register_network(
        "root_net", root_net, torch.optim.SGD(root_net.parameters(), lr=0.0)
    )
    for rule in rules:
        guard = _make_rule_weight_module(rule.initial_weight).cuda()
        program.register_network(
            rule.guard_network, guard, torch.optim.SGD(guard.parameters(), lr=0.0)
        )
    program.add_tensor_source(_TENSOR_SOURCE_NAME, inputs.cuda())

    queries = [f"{train_head}({i})" for i in range(len(targets))]
    return program, root_net, queries, list(targets)


def _abs_grad_sum(module) -> float:
    return float(
        sum(
            p.grad.detach().abs().sum().item()
            for p in module.parameters()
            if p.grad is not None
        )
    )


@requires_cuda
def test_grouped_matches_scalar_forward_backward() -> None:
    """forward_backward_grouped (the device-resident batched path) must be
    numerically identical to looping the scalar forward_backward over the same
    queries: same summed loss AND same accumulated gradients. This pins that the
    zero-host reroute changed performance, not training semantics."""
    # Batched/grouped path on a fresh program.
    prog_g, net_g, queries, expected = _build_mixed_program()
    prog_g.zero_grad()
    loss_grouped = prog_g.forward_backward_grouped(queries, expected)
    grad_grouped = _abs_grad_sum(net_g)

    # Scalar per-query loop on an identical fresh program (same fixed weights).
    prog_s, net_s, queries_s, expected_s = _build_mixed_program()
    prog_s.zero_grad()
    loss_scalar = sum(
        prog_s.forward_backward(q, t) for q, t in zip(queries_s, expected_s)
    )
    grad_scalar = _abs_grad_sum(net_s)

    assert loss_grouped == pytest.approx(loss_scalar, rel=1e-5, abs=1e-6)
    assert grad_grouped == pytest.approx(grad_scalar, rel=1e-4, abs=1e-6)
    # Gradient must be non-trivial (eligible cases drive it); a zero on both
    # sides would make the equality vacuous.
    assert grad_scalar > 0.0


@requires_cuda
def test_query_probabilities_grouped_matches_scalar() -> None:
    """The batched probability readout must match the per-query scalar readout
    (exp(-forward_backward(q, True))) for every query, including ineligible
    (hard-filtered) ones — so the readout is numerically transparent like the
    loss path while running O(templates) host syncs instead of O(N)."""
    import math

    prog_g, _net_g, queries, _expected = _build_mixed_program()
    prog_g.zero_grad()
    probs_grouped = prog_g.query_probabilities_grouped(queries)
    prog_g.zero_grad()

    prog_s, _net_s, queries_s, _expected_s = _build_mixed_program()
    prog_s.zero_grad()
    probs_scalar = [math.exp(-prog_s.forward_backward(q, True)) for q in queries_s]
    prog_s.zero_grad()

    assert len(probs_grouped) == len(probs_scalar)
    for pg, ps in zip(probs_grouped, probs_scalar):
        assert pg == pytest.approx(ps, rel=1e-5, abs=1e-9)
    # Ineligible (hard-filtered) cases 1 and 3 stay ~0 through the batched path.
    assert probs_grouped[1] == pytest.approx(0.0, abs=1e-6)
    assert probs_grouped[3] == pytest.approx(0.0, abs=1e-6)


@requires_cuda
def test_training_loop_has_no_tracked_host_transfers() -> None:
    """The warm training loop must perform NO tracked device<->host transfers in
    either direction. After the one-time cache warm-up, every step runs
    device-resident: no downloads (loss accumulates on device) and no uploads
    (query-var metadata is cached on device). dtoh AND htod, calls AND bytes,
    all stay at their reset baseline of 0 across the measured loop."""
    result = train_neurosymbolic_program(
        MIXED_BODY_SOURCE,
        networks={"root_net": _root_net()},
        examples=_examples(),
        config=NeuroSymbolicTrainingConfig(steps=3, learning_rate=0.2),
    )
    stats = result.training_host_transfer_stats
    assert stats is not None
    assert stats["dtoh_calls"] == 0
    assert stats["dtoh_bytes"] == 0
    assert stats["htod_calls"] == 0
    assert stats["htod_bytes"] == 0


# A single head derived by THREE trainable candidate rules (multi-rule same-head)
# — the ST-TRC Phase-1b joint soft-mixture topology. Only the correct candidate
# (supp ∩ refut) fires exactly on the positives {0,2}; the distractors fire on
# wrong rows.
JOINT_MIXTURE_SOURCE = """
    supp(0). supp(2). supp(1). refut(0). refut(2). refut(3).
    only_a(0). only_a(1). only_a(3). only_b(1). only_b(2). only_b(3).
    pred supp(i64). pred refut(i64). pred only_a(i64). pred only_b(i64).
    trainable_rule(cand_correct, weight=0.0) :: target(C) :- supp(C), refut(C).
    trainable_rule(cand_a, weight=0.0) :: target(C) :- only_a(C).
    trainable_rule(cand_b, weight=0.0) :: target(C) :- only_b(C).
    train(target, binary_cross_entropy).
"""


@requires_cuda
def test_joint_multi_rule_same_head_mixture_selects_correct_candidate() -> None:
    """ST-TRC Phase-1b acceptance gate: when N trainable rules derive ONE head
    (multi-rule same-head — previously rejected with 'expected exactly 1 matching
    rule'), the joint noisy-OR competition must drive the CORRECT candidate's
    guard high and ALL distractor guards low (the hard mask selects the correct
    (i,j)). Guard-only candidates; the training loop stays zero-host."""
    examples = [
        {
            "inputs": torch.zeros((4, 1), dtype=torch.float32),
            "targets": torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32),
        }
    ]
    result = train_neurosymbolic_program(
        JOINT_MIXTURE_SOURCE,
        networks={},
        examples=examples,
        config=NeuroSymbolicTrainingConfig(steps=400, learning_rate=0.1),
    )
    w = result.symbolic_rule_weights
    # (1) correct guard high, ALL distractor guards low — the competition selects.
    assert w["cand_correct"] > 0.7
    assert w["cand_a"] < 0.3
    assert w["cand_b"] < 0.3
    # The joint head probability separates positives from negatives.
    probs = result.query_probabilities
    assert min(probs[0], probs[2]) > 0.7
    assert max(probs[1], probs[3]) < 0.3
    # The joint loop is pure torch over guard params + static engine masks:
    # no tracked device<->host transfers.
    stats = result.training_host_transfer_stats
    assert stats["dtoh_calls"] == 0 and stats["htod_calls"] == 0


@requires_cuda
def test_evaluate_joint_mixture_matches_training_head_prob() -> None:
    """Faithfulness pin: the held-out read evaluated over the SAME facts as the
    train split must reproduce the training-time noisy-OR (query_probabilities)
    to within float tolerance. This guarantees evaluate_joint_mixture is the
    identical mixture as the trained forward — the property that makes the
    held-out generalization read honest, not a re-derivation that could drift."""
    examples = [
        {
            "inputs": torch.zeros((4, 1), dtype=torch.float32),
            "targets": torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32),
        }
    ]
    result = train_neurosymbolic_program(
        JOINT_MIXTURE_SOURCE,
        networks={},
        examples=examples,
        config=NeuroSymbolicTrainingConfig(steps=400, learning_rate=0.1),
    )
    probs = evaluate_joint_mixture(
        JOINT_MIXTURE_SOURCE,
        rule_weights=result.symbolic_rule_weights,
        num_queries=4,
    )
    assert len(probs) == 4
    for got, ref in zip(probs, result.query_probabilities):
        assert abs(got - ref) < 1e-5


@requires_cuda
def test_evaluate_joint_mixture_generalizes_on_held_out_split() -> None:
    """The held-out read is the anti-spurious generalization gate: the TRAINED
    guard mixture, evaluated on the engine's relational eligibility for held-out
    bindings the training never saw, must (a) stay high where the correct
    candidate's join fires (it GENERALIZES) and (b) collapse to ~0 where no
    candidate's facts are present (the materialization caveat: absent held-out
    facts read as non-coverage, not a real spurious signal)."""
    examples = [
        {
            "inputs": torch.zeros((4, 1), dtype=torch.float32),
            "targets": torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32),
        }
    ]
    result = train_neurosymbolic_program(
        JOINT_MIXTURE_SOURCE,
        networks={},
        examples=examples,
        config=NeuroSymbolicTrainingConfig(steps=400, learning_rate=0.1),
    )
    # Held-out bindings the training never saw: id 0 is covered by the correct
    # join (supp INT refut); id 1 has NO supporting facts at all.
    held_out_source = """
        supp(0). refut(0).
        pred supp(i64). pred refut(i64). pred only_a(i64). pred only_b(i64).
        trainable_rule(cand_correct, weight=0.0) :: target(C) :- supp(C), refut(C).
        trainable_rule(cand_a, weight=0.0) :: target(C) :- only_a(C).
        trainable_rule(cand_b, weight=0.0) :: target(C) :- only_b(C).
        train(target, binary_cross_entropy).
    """
    probs = evaluate_joint_mixture(
        held_out_source,
        rule_weights=result.symbolic_rule_weights,
        num_queries=2,
    )
    assert len(probs) == 2
    # id 0: trained correct candidate fires on the held-out binding -> generalizes.
    assert probs[0] > 0.7
    # id 1: no facts -> all candidates ineligible -> p_or ~ 0 (caveat case).
    assert probs[1] < 1e-3


@requires_cuda
def test_evaluate_joint_mixture_per_candidate_read_discriminates_train_tie() -> None:
    """Encodes the Phase-2 train-tie finding: when two candidates fit the train
    trigger equally their guards are EQUAL (the competition cannot separate them),
    so admission must read the SELECTED candidate's held-out probability — not the
    pool. With equal guards, the per-candidate (single-weight) held-out read
    discriminates: the true join covers held-out binding 0 (not 1), the spurious
    join covers binding 1 (not 0). The pool-wide read is inflated on BOTH bindings
    — which is exactly why the admission gate passes only the winner's weight."""
    held_out_source = """
        rel_a(0). rel_b(0). rel_a(1). rel_c(1).
        pred rel_a(i64). pred rel_b(i64). pred rel_c(i64).
        trainable_rule(cand_true, weight=0.0) :: target(C) :- rel_a(C), rel_b(C).
        trainable_rule(cand_spur, weight=0.0) :: target(C) :- rel_a(C), rel_c(C).
        train(target, binary_cross_entropy).
    """
    # Simulate the train-tie: both guards trained to the same high value.
    tie = 0.95
    true_read = evaluate_joint_mixture(
        held_out_source, rule_weights={"cand_true": tie}, num_queries=2
    )
    spur_read = evaluate_joint_mixture(
        held_out_source, rule_weights={"cand_spur": tie}, num_queries=2
    )
    pool_read = evaluate_joint_mixture(
        held_out_source,
        rule_weights={"cand_true": tie, "cand_spur": tie},
        num_queries=2,
    )
    # Per-candidate reads discriminate despite identical guards:
    assert true_read[0] > 0.9 and true_read[1] < 1e-3  # true covers 0, not 1
    assert spur_read[0] < 1e-3 and spur_read[1] > 0.9  # spurious covers 1, not 0
    # Pool-wide is inflated wherever EITHER fires -> not a single-candidate gate.
    assert pool_read[0] > 0.9 and pool_read[1] > 0.9


# ST-TRC slice-1: neural-bodied candidate. The "fragility-on-drop" anchor —
# vase/bulb break when dropped, the steel ball does not. ALL are dropped, so the
# relational candidate (dropped) fires on every instance and CANNOT separate +/-;
# only a learned predicate over entity features phi(x) (fragility) separates them.
_NEURAL_BODY_SOURCE = """
    dropped(0). dropped(1). dropped(2).
    pred dropped(i64). pred breaks(i64).
    trainable_rule(cand_rel, weight=0.0) :: breaks(C) :- dropped(C).
    trainable_rule(cand_neural, weight=0.0) :: breaks(C) :- dropped(C).
    train(breaks, binary_cross_entropy).
"""


def _train_fragility(steps: int = 500):
    # phi: vase[1,0], bulb[1,0] fragile; ball[0,1] sturdy.
    phi = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    examples = [
        {
            "inputs": torch.zeros((3, 1), dtype=torch.float32),
            "targets": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        }
    ]
    return train_neurosymbolic_program(
        _NEURAL_BODY_SOURCE,
        networks={},
        examples=examples,
        config=NeuroSymbolicTrainingConfig(steps=steps, learning_rate=0.1),
        neural_bodies={"cand_neural": NeuralBodySpec(features=phi)},
    )


@requires_cuda
def test_neural_body_separates_where_relational_cannot() -> None:
    """Necessity + sufficiency: the relational candidate (dropped) fires on every
    dropped object and cannot separate breaking from non-breaking; the neural
    candidate learns g_theta(phi) >= tau over fragility features and DOES separate.
    After joint training the mixture predicts break for vase/bulb, not the ball,
    the neural candidate is selected, and gradient reached theta."""
    result = _train_fragility()
    probs = result.query_probabilities
    assert min(probs[0], probs[1]) > 0.6  # fragile -> break
    assert probs[2] < 0.4  # sturdy ball -> no break (relational layer can't do this)
    # the neural candidate carries the separating rule and was selected
    assert result.symbolic_rule_weights["cand_neural"] > 0.5
    # ST gate routed gradient to the neural head params
    assert result.neural_parameter_grads["cand_neural"] > 0.0
    # trained head serialized for the driver's parametric HardenedClause
    assert result.neural_body_state is not None
    state = result.neural_body_state["cand_neural"]
    assert state.width == 2 and state.threshold == 0.5


@requires_cuda
def test_neural_body_held_out_generalizes_and_keeps_vigilance() -> None:
    """The trained g_theta gate generalizes to held-out entities the training never
    saw, AND keeps the held-out vigilance net: a held-out FRAGILE entity fires the
    gate (generalizes), a held-out STURDY entity does NOT (the neural analog of the
    spurious-correlate rejection — an overfit gate fails held-out exactly as a
    spurious relational join does)."""
    result = _train_fragility()
    held_out_source = """
        dropped(0). dropped(1).
        pred dropped(i64). pred breaks(i64).
        trainable_rule(cand_rel, weight=0.0) :: breaks(C) :- dropped(C).
        trainable_rule(cand_neural, weight=0.0) :: breaks(C) :- dropped(C).
        train(breaks, binary_cross_entropy).
    """
    # held-out 0 = a NEW fragile object, held-out 1 = a NEW sturdy object.
    held_phi = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    probs = evaluate_joint_mixture(
        held_out_source,
        rule_weights={"cand_neural": result.symbolic_rule_weights["cand_neural"]},
        num_queries=2,
        neural_heldout={
            "cand_neural": (result.neural_body_state["cand_neural"], held_phi)
        },
    )
    assert probs[0] > 0.6  # held-out fragile: trained gate fires -> generalizes
    assert probs[1] < 0.4  # held-out sturdy: gate correctly does NOT fire (vigilance)


@requires_cuda
def test_neural_body_training_has_no_tracked_host_transfers() -> None:
    """The neural-bodied joint loop stays zero-host: phi is uploaded once and the
    g_theta forward + ST gate + noisy-OR are torch over device tensors, so the
    engine performs no tracked device<->host transfers during training."""
    result = _train_fragility(steps=50)
    stats = result.training_host_transfer_stats
    assert stats["dtoh_calls"] == 0 and stats["htod_calls"] == 0


@requires_cuda
def test_neural_body_graded_admission_read_emits_decomposed_evidence() -> None:
    """Surface-1 Axis-I SAFE_GRADED graded read: ``mode='graded'`` returns the
    decomposed per-query evidence over the held-out split, and the hard default is
    byte-unchanged. The held-out fragile (label True) carries more graded mass than
    the sturdy (label False), so the Axis-I retention margin is positive — while
    production firing stays the hard default (the graded read carries NO firing
    certification; ``production_firing_mass`` is exposed only for the rubric's
    Axis-II read)."""
    result = _train_fragility()
    held_out_source = """
        dropped(0). dropped(1).
        pred dropped(i64). pred breaks(i64).
        trainable_rule(cand_rel, weight=0.0) :: breaks(C) :- dropped(C).
        trainable_rule(cand_neural, weight=0.0) :: breaks(C) :- dropped(C).
        train(breaks, binary_cross_entropy).
    """
    held_phi = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)  # fragile, sturdy
    weights = {"cand_neural": result.symbolic_rule_weights["cand_neural"]}
    neural_heldout = {"cand_neural": (result.neural_body_state["cand_neural"], held_phi)}

    hard = evaluate_joint_mixture(
        held_out_source, rule_weights=weights, num_queries=2, neural_heldout=neural_heldout
    )
    graded = evaluate_joint_mixture(
        held_out_source,
        rule_weights=weights,
        num_queries=2,
        neural_heldout=neural_heldout,
        mode="graded",
        heldout_labels=[True, False],
    )

    # hard default stays the unchanged list[float]
    assert isinstance(hard, list) and len(hard) == 2
    # graded mode is the decomposed evidence packet the rubric checker consumes
    assert graded["mode"] == "graded"
    pq = graded["per_query"]
    assert len(pq) == 2
    for r in pq:
        assert r["selected_rule_id"] == "cand_neural"
        assert set(r) >= {
            "query_index", "selected_rule_id", "relational_mask", "hard_gate",
            "graded_gate", "g_theta", "tau_logit", "hard_head_prob", "graded_mass",
            "production_firing_mass", "label",
        }
        assert 0.0 <= r["graded_mass"] <= 1.0
        assert 0.0 <= r["production_firing_mass"] <= 1.0
    # AUDIT INVARIANT end-to-end: mode='hard' equals each query's hard_head_prob —
    # both run through the SAME rel_mask * gate * sigma(guard) structure.
    assert hard[0] == pytest.approx(pq[0]["hard_head_prob"], abs=1e-5)
    assert hard[1] == pytest.approx(pq[1]["hard_head_prob"], abs=1e-5)
    # Axis-I retention: the held-out fragile out-ranks the sturdy by graded mass.
    assert pq[0]["graded_mass"] > pq[1]["graded_mass"]
    assert graded["axis1_margin"] == pytest.approx(
        pq[0]["graded_mass"] - pq[1]["graded_mass"], abs=1e-6
    )
    assert graded["axis1_margin"] > 0.0
