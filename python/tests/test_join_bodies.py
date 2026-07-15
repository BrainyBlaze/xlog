import pytest

torch = pytest.importorskip("torch")

from pyxlog.ilp.join_bodies import (
    JoinBody,
    domain_row_index,
    mentions_neural_on_nonhead_var,
    noisy_or_from_index,
    noisy_or_over_extension,
    parse_join_body,
    prepare_extension,
    translate_extension_to_rows,
)

NEURAL = {"saliency": "sal_net"}


def test_parses_the_stage_b_body() -> None:
    jb = parse_join_body(
        ["saliency(Ev, strengthen)", "pre_before_post(Ev, E)"], NEURAL, head_var="E"
    )
    assert jb == JoinBody(
        neural_predicate="saliency", network="sal_net", join_var="Ev",
        relation="pre_before_post", event_arg=0, head_arg=1,
    )


def test_arg_order_is_read_from_the_body_not_assumed() -> None:
    """The relation may put the head first; the parse must not hardcode positions."""
    jb = parse_join_body(
        ["saliency(Ev, strengthen)", "edge_of(E, Ev)"], NEURAL, head_var="E"
    )
    assert jb.relation == "edge_of" and jb.event_arg == 1 and jb.head_arg == 0


def test_a_plain_relational_body_is_not_a_join_body() -> None:
    assert parse_join_body(["edge_pre_post(E)"], NEURAL, head_var="E") is None


def test_a_body_whose_neural_var_is_the_head_is_not_a_join_body() -> None:
    """That is the head-bound gate shape (variant A), not an existential join."""
    assert parse_join_body(
        ["saliency(E, strengthen)", "rel(E, E)"], NEURAL, head_var="E"
    ) is None


def test_an_extra_conjunct_is_not_silently_dropped() -> None:
    """`high_degree(E)` is a real conjunct this module cannot mask. Returning the
    two-literal JoinBody anyway would TRAIN a rule nobody wrote. Out of scope must
    mean rejected, so the parse refuses the body and the caller's typed error stands."""
    assert (
        parse_join_body(
            ["saliency(Ev, strengthen)", "pre_before_post(Ev, E)", "high_degree(E)"],
            NEURAL,
            head_var="E",
        )
        is None
    )


def test_a_join_relation_with_a_third_argument_is_not_a_join_body() -> None:
    """`pre_before_post(Ev, E, W)` mentions both the join var and the head var, so a
    check that only asks "are both present?" accepts it. Its third argument is a SECOND
    existential, though: the relation holds one (event, edge) tuple per W, and
    `read_join_extension` — which buckets every tuple by its head argument — would bucket
    the same event W times and compute `1 - (1 - p)^W` where the rule says
    `1 - (1 - p)`. This module says its shape is EXACTLY one neural atom plus one join
    relation and that it never guesses; that has to include the relation's arity."""
    assert (
        parse_join_body(
            ["saliency(Ev, strengthen)", "pre_before_post(Ev, E, W)"],
            NEURAL,
            head_var="E",
        )
        is None
    )


def test_two_relations_on_the_join_var_are_not_a_join_body() -> None:
    assert (
        parse_join_body(
            ["saliency(Ev, strengthen)", "pre_before_post(Ev, E)", "other(Ev, E)"],
            NEURAL,
            head_var="E",
        )
        is None
    )


# --- the contract is over BODY LITERALS, not over parenthesized atoms -------------
# A literal need not be an atom (xlog's BodyLiteral is also Negated / Comparison /
# IsExpr / Univ / Epistemic). Counting atoms cannot see any of these, so each of the
# bodies below counts TWO atoms and would have parsed as the join shape -- silently
# dropping the third literal, or (worse) the `not`, training the INVERSE rule.


def test_a_comparison_literal_is_not_silently_dropped() -> None:
    assert (
        parse_join_body(
            ["saliency(Ev, strengthen)", "pre_before_post(Ev, E)", "Ev < 3"],
            NEURAL,
            head_var="E",
        )
        is None
    )


def test_a_negated_join_relation_is_rejected_not_inverted() -> None:
    """`not pre_before_post(Ev, E)` is the COMPLEMENT of the join relation. Masking it
    with the extension of `pre_before_post` would train the exact inverse of the rule
    that was written, and report it under the written rule's name."""
    assert (
        parse_join_body(
            ["saliency(Ev, strengthen)", "not pre_before_post(Ev, E)"],
            NEURAL,
            head_var="E",
        )
        is None
    )


def test_a_negated_neural_atom_is_rejected() -> None:
    assert (
        parse_join_body(
            ["not saliency(Ev, strengthen)", "pre_before_post(Ev, E)"],
            NEURAL,
            head_var="E",
        )
        is None
    )


def test_an_is_expression_literal_is_rejected() -> None:
    assert (
        parse_join_body(
            ["saliency(Ev, strengthen)", "pre_before_post(Ev, E)", "Z is Ev + 1"],
            NEURAL,
            head_var="E",
        )
        is None
    )


def test_a_modal_literal_is_rejected() -> None:
    assert (
        parse_join_body(
            ["saliency(Ev, strengthen)", "know pre_before_post(Ev, E)"],
            NEURAL,
            head_var="E",
        )
        is None
    )


# --- the routing predicate --------------------------------------------------------
# It answers "is this a join candidate at all", and must say YES for the out-of-shape
# bodies above too: those must be REJECTED by the caller, never trained as plain
# relational candidates on whatever mask the engine happened to hand back.


def test_the_routing_predicate_sees_a_neural_predicate_on_an_existential_var() -> None:
    assert mentions_neural_on_nonhead_var(
        ["saliency(Ev, strengthen)", "pre_before_post(Ev, E)"], NEURAL, "E"
    )


def test_the_routing_predicate_sees_through_a_negation_and_a_comparison() -> None:
    assert mentions_neural_on_nonhead_var(
        ["not saliency(Ev, strengthen)", "pre_before_post(Ev, E)", "Ev < 3"], NEURAL, "E"
    )


def test_the_routing_predicate_ignores_a_head_bound_neural_atom() -> None:
    """Variant A (the head-bound gate) is NOT a join candidate: its neural atom is on
    the head variable, so the engine's own eligibility mask is the whole relational
    truth and the join path must stay out of it."""
    assert not mentions_neural_on_nonhead_var(
        ["saliency(E, strengthen)", "allowed(E)"], NEURAL, "E"
    )


def test_the_routing_predicate_ignores_a_purely_relational_body() -> None:
    assert not mentions_neural_on_nonhead_var(
        ["pre_before_post(Ev, E)", "high_degree(E)"], NEURAL, "E"
    )


def test_noisy_or_matches_the_naive_product() -> None:
    p = torch.tensor([0.9, 0.1, 0.2, 0.85])
    ext = [[0, 1], [2], [], [0, 1, 2, 3]]
    got = noisy_or_over_extension(p, ext, "cpu")
    want = torch.tensor([
        1 - (1 - 0.9) * (1 - 0.1),
        0.2,
        0.0,                                  # empty extension -> the OR is FALSE
        1 - (1 - 0.9) * (1 - 0.1) * (1 - 0.2) * (1 - 0.85),
    ])
    assert torch.allclose(got, want, atol=1e-6)


def test_noisy_or_is_stable_for_a_large_extension() -> None:
    """The naive product underflows; log-space must not."""
    p = torch.full((500,), 0.02)
    got = noisy_or_over_extension(p, [list(range(500))], "cpu")
    assert torch.isfinite(got).all()
    assert 0.99 < float(got[0]) < 1.0     # 1 - 0.98^500 ~ 0.99996


def test_noisy_or_is_differentiable() -> None:
    p = torch.tensor([0.3, 0.7], requires_grad=True)
    noisy_or_over_extension(p, [[0, 1]], "cpu").sum().backward()
    assert p.grad is not None and float(p.grad.abs().sum()) > 0


def test_the_prepared_index_computes_the_hand_written_or() -> None:
    """The HOT LOOP calls `noisy_or_from_index` (never `noisy_or_over_extension`), so
    it is pinned against an independently hand-computed expectation, not against the
    other entry point -- comparing the two would only restate that one delegates to
    the other."""
    p = torch.tensor([0.9, 0.1, 0.2, 0.85])
    got = noisy_or_from_index(p, prepare_extension([[0, 1], [2], [], [0, 1, 2, 3]], "cpu"))
    want = torch.tensor([
        1 - (1 - 0.9) * (1 - 0.1),
        0.2,
        0.0,
        1 - (1 - 0.9) * (1 - 0.1) * (1 - 0.2) * (1 - 0.85),
    ])
    assert torch.allclose(got, want, atol=1e-6)


def test_prepare_extension_flattens_the_extension_into_device_tensors() -> None:
    """The per-step work must be a single vectorized op over device-resident tensors,
    so the static extension is flattened into (event_ids, segment_ids) tensors. (That
    this flattening happens ONCE per candidate rather than once per step is pinned by
    test_join_bodies_engine.py, which counts the calls across a real training run.)"""
    handle = prepare_extension([[0, 1], [2], [], [0, 1, 2, 3]], "cpu")
    assert handle.num_bindings == 4
    assert handle.event_ids.tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert handle.binding_ids.tolist() == [0, 0, 1, 3, 3, 3, 3]
    assert handle.event_ids.dtype == torch.long


def test_an_all_empty_extension_yields_exactly_zero() -> None:
    p = torch.tensor([0.9, 0.1])
    got = noisy_or_from_index(p, prepare_extension([[], []], "cpu"))
    assert got.tolist() == [0.0, 0.0]


def test_the_prepared_index_is_differentiable() -> None:
    p = torch.tensor([0.3, 0.7], requires_grad=True)
    noisy_or_from_index(p, prepare_extension([[0, 1], []], "cpu")).sum().backward()
    assert p.grad is not None and float(p.grad.abs().sum()) > 0


# --- constant -> row: the explicit-id translation (R3) -----------------------------
# The engine's extension speaks RAW domain constants; domain_inputs is a bare tensor
# whose rows the caller laid out. `domain_ids` is the caller's statement of which row
# holds which constant, and this translation is the ONE place the two meet.


def test_a_sparse_domain_maps_its_constants_onto_rows() -> None:
    """The constants need not be their own row numbers: {0,2,4,6,8,10} over 6 rows."""
    rows = translate_extension_to_rows(
        [[0, 2], [4, 6], [8], [10], []], [0, 2, 4, 6, 8, 10], network="sal_net"
    )
    assert rows == [[0, 1], [2, 3], [4], [5], []]


def test_the_dense_default_is_the_identity() -> None:
    """The default ids (0..D-1) must translate to exactly themselves -- that is what
    keeps every caller written before `domain_ids` existed behaving identically."""
    ext = [[0, 1], [2], [], [0, 1, 2, 3]]
    assert translate_extension_to_rows(ext, list(range(4))) == ext


def test_domain_row_index_maps_constant_to_row() -> None:
    assert domain_row_index([0, 2, 4, 6, 8, 10]) == {0: 0, 2: 1, 4: 2, 6: 3, 8: 4, 10: 5}


def test_a_constant_with_no_row_is_named_not_silently_mis_read() -> None:
    """Event 7 is joined by the engine's relation but has no feature row. Reading some
    other constant's row (or off the end of the tensor -- a CUDA device-side assert that
    poisons the process) is exactly the failure R3 exists to close, so it is a typed
    error that NAMES the constant."""
    with pytest.raises(ValueError, match="7"):
        translate_extension_to_rows([[0, 2], [7]], [0, 2, 4, 6], network="sal_net")


def test_ids_in_any_order_are_honoured_not_refused() -> None:
    """The row layout of the feature tensor is the CALLER's. Both engines FIND a row by
    the constant it holds -- the circuit looks the constant up in this same id list -- so
    there is no ordering left for either side to count in, and an unsorted list is simply
    the caller's honest statement of where things are. Ascending order used to be required
    because it was what made the circuit's row-counting coincide with this map; the
    counting is gone, so the requirement goes with it."""
    assert domain_row_index([6, 0, 4, 2], "sal_net") == {6: 0, 0: 1, 4: 2, 2: 3}
    assert translate_extension_to_rows(
        [[0, 6], [2]], [6, 0, 4, 2], network="sal_net"
    ) == [[1, 0], [3]]


def test_duplicate_ids_are_refused() -> None:
    """Two rows claiming one constant leaves that constant's row undefined -- refused by
    name, rather than resolved by a tie-break the caller never asked for. This is the part
    the old ascending-order rule was really carrying."""
    with pytest.raises(ValueError, match="must not repeat"):
        domain_row_index([0, 2, 2, 6])


# ---------------------------------------------------------------------------
# Review wave 2 (PR #152 feedback): three more spots where "out of scope must
# mean refused" had holes, each reproduced by execution before it was fixed.
# ---------------------------------------------------------------------------


def test_a_neural_atom_with_extra_arguments_is_rejected() -> None:
    """`saliency(Ev, X, strengthen)` is a DIFFERENT rule: X is a second existential
    this mask cannot express. Before the fix it parsed -- the middle argument simply
    vanished, and because the label lookup takes the last argument, nothing
    downstream errored: the mixture trained the two-argument rule the body merely
    resembled."""
    assert parse_join_body(
        ["saliency(Ev, X, strengthen)", "pre_before_post(Ev, E)"], NEURAL, head_var="E"
    ) is None


def test_a_variable_in_the_label_slot_is_rejected() -> None:
    """`saliency(Ev, Lbl)` has no single output column to train against. Refusing it
    here names the rule; deferring to the engine's label_to_index surfaces the same
    failure far from its cause."""
    assert parse_join_body(
        ["saliency(Ev, Lbl)", "pre_before_post(Ev, E)"], NEURAL, head_var="E"
    ) is None


def test_the_wildcard_is_a_variable_to_the_router_and_refused_by_the_parse() -> None:
    """The grammar admits `_` as an anonymous variable, and each `_` is a DISTINCT
    variable -- `saliency(_, l), rel(_, E)` shares nothing, so no join exists.

    Two properties, both load-bearing: the ROUTER must see the neural predicate on
    `_` (a router blind to it waved the candidate through to the plain relational
    path -- trained as an always-true rule, no gradient to the detector, no error),
    and the PARSE must refuse it (textual matching would wrongly see a shared
    variable where the semantics has two)."""
    body = ["saliency(_, strengthen)", "pre_before_post(_, E)"]
    assert mentions_neural_on_nonhead_var(body, NEURAL, "E")
    assert parse_join_body(body, NEURAL, head_var="E") is None


class _FakeIlpProgram:
    def __init__(self, facts: list[list[int]]) -> None:
        self._facts = facts

    def relation_facts(self, rel_name: str) -> list[list[int]]:
        return self._facts


def test_an_out_of_range_head_binding_is_a_loud_error_not_a_silent_drop() -> None:
    """A tuple whose head binding falls outside 0..num_bindings-1 is the same class
    of caller/world disagreement as a joined constant missing from domain_ids, which
    is loud. Silently dropping it shrank the candidate's extension -- and its OR --
    without a trace."""
    from pyxlog.ilp.join_bodies import read_join_extension

    jb = JoinBody(
        neural_predicate="saliency", network="sal_net", join_var="Ev",
        relation="pbp", event_arg=0, head_arg=1,
    )
    prog = _FakeIlpProgram([[0, 0], [1, 7]])   # binding 7 with num_bindings=2
    with pytest.raises(ValueError, match=r"outside the query range 0\.\.1"):
        read_join_extension(prog, jb, num_bindings=2)


def test_an_untranslated_extension_is_caught_at_prepare_time() -> None:
    """Both overrun directions, closed once at build time: an id >= num_rows dies
    later as a context-poisoning CUDA device assert, and an id that happens to fall
    inside num_rows gathers the WRONG rows and computes a silently wrong OR. The
    host-side bounds check catches the loud direction; the silent one is exactly why
    the check names translate_extension_to_rows."""
    with pytest.raises(ValueError, match="TRANSLATED to rows"):
        prepare_extension([[0], [12]], device="cpu", num_rows=4)
    # and a translated extension passes untouched
    idx = prepare_extension([[0], [3]], device="cpu", num_rows=4)
    assert idx.num_bindings == 2


def test_a_negated_only_neural_mention_still_registers_its_network() -> None:
    """`literal.split("(")` yields "not saliency" for a negated literal, so a rule
    whose ONLY mention of a neural predicate was negated never registered the
    network -- and the routing question downstream, which explicitly sees through
    negation, was asked against an empty map. Name derivation now uses the same
    _ATOM regex the join module parses with."""
    from pyxlog.ilp.neurosymbolic import TrainableRuleDecl, _neural_predicate_networks

    class _FakeProgram:
        def neural_predicate_info(self, name: str) -> dict:
            if name == "saliency":
                return {"network": "sal_net"}
            raise ValueError(f"{name} is not a neural predicate")

    rule = TrainableRuleDecl(
        id="r1", head="plastic(E)",
        body_literals=("not saliency(Ev, strengthen)", "pre_before_post(Ev, E)"),
        initial_weight=0.0,
        source="trainable_rule(r1, weight=0.0) :: plastic(E) :- "
               "not saliency(Ev, strengthen), pre_before_post(Ev, E).",
        guard_predicate="__xlog_rule_guard_r1", guard_network="__xlog_rule_gnet_r1",
        query_variable="E",
    )
    assert _neural_predicate_networks(_FakeProgram(), [rule]) == {"saliency": "sal_net"}


def test_the_distractor_salient_composition_is_class_independent() -> None:
    """Review (engine half), finding 7: uniform sampling from "other edges' events"
    leaks anti-correlated label signal -- a positive edge's pool holds S-1 salient
    events, a negative edge's holds S -- and the bias is MATERIAL at small n_edges
    (analytically about -0.20 salient-per-bucket at n_edges=6, k=6). The composition
    is now drawn from ONE distribution shared by both classes, so the salient count a
    converged detector's OR sees carries no label information by construction.

    Pinned where it discriminates: small world, 40 fixed seeds, SIGNED per-class gap
    averaged across seeds and both distractor relations. Measured: -0.026 for the
    shared-composition sampler vs -0.20 for the old one; the 0.1 bound sits between.
    Fixed seeds -- cannot flake."""
    from pyxlog.ilp.discovery import SALIENT_THRESHOLD, make_world

    gaps: list[float] = []
    for seed in range(40):
        world = make_world(n_edges=6, events_per_edge=6, seed=seed)
        salient = {
            ev for ev, f in enumerate(world.event_features) if f > SALIENT_THRESHOLD
        }
        for relation in (world.post_before_pre, world.co_occurs):
            per_edge: dict[int, int] = {}
            for ev, edge in relation:
                per_edge[edge] = per_edge.get(edge, 0) + (1 if ev in salient else 0)
            pos = [per_edge.get(e, 0) for e in world.edges if world.labels[e]]
            neg = [per_edge.get(e, 0) for e in world.edges if not world.labels[e]]
            if pos and neg:
                gaps.append(sum(pos) / len(pos) - sum(neg) / len(neg))
    mean_gap = sum(gaps) / len(gaps)
    assert abs(mean_gap) < 0.1, mean_gap
