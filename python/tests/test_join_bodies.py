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


def test_unsorted_ids_are_refused() -> None:
    """Load-bearing, not stylistic: the exact circuit reads row j as the j-th constant
    in SORTED order. Only ascending ids make "row j holds domain_ids[j]" true under BOTH
    conventions; unsorted ids would put the two engines back into silent disagreement."""
    with pytest.raises(ValueError, match="strictly increasing"):
        domain_row_index([0, 4, 2, 6], "sal_net")
    with pytest.raises(ValueError, match="strictly increasing"):
        translate_extension_to_rows([[0]], [0, 4, 2, 6], network="sal_net")


def test_duplicate_ids_are_refused() -> None:
    """Two rows claiming one constant has no meaning under either convention."""
    with pytest.raises(ValueError, match="strictly increasing"):
        domain_row_index([0, 2, 2, 6])
