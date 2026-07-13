import pytest

torch = pytest.importorskip("torch")

from pyxlog.ilp.join_bodies import (
    JoinBody,
    noisy_or_from_index,
    noisy_or_over_extension,
    parse_join_body,
    prepare_extension,
)

NEURAL = {"saliency": "sal_net"}


def test_parses_the_stage_b_body() -> None:
    jb = parse_join_body(
        "saliency(Ev, strengthen), pre_before_post(Ev, E)", NEURAL, head_var="E"
    )
    assert jb == JoinBody(
        neural_predicate="saliency", network="sal_net", join_var="Ev",
        relation="pre_before_post", event_arg=0, head_arg=1,
    )


def test_arg_order_is_read_from_the_body_not_assumed() -> None:
    """The relation may put the head first; the parse must not hardcode positions."""
    jb = parse_join_body(
        "saliency(Ev, strengthen), edge_of(E, Ev)", NEURAL, head_var="E"
    )
    assert jb.relation == "edge_of" and jb.event_arg == 1 and jb.head_arg == 0


def test_a_plain_relational_body_is_not_a_join_body() -> None:
    assert parse_join_body("edge_pre_post(E)", NEURAL, head_var="E") is None


def test_a_body_whose_neural_var_is_the_head_is_not_a_join_body() -> None:
    """That is the head-bound gate shape (variant A), not an existential join."""
    assert parse_join_body("saliency(E, strengthen), rel(E, E)", NEURAL, head_var="E") is None


def test_an_extra_conjunct_is_not_silently_dropped() -> None:
    """`high_degree(E)` is a real conjunct this module cannot mask. Returning the
    two-literal JoinBody anyway would TRAIN a rule nobody wrote. Out of scope must
    mean rejected, so the parse refuses the body and the caller's typed error stands."""
    assert (
        parse_join_body(
            "saliency(Ev, strengthen), pre_before_post(Ev, E), high_degree(E)",
            NEURAL,
            head_var="E",
        )
        is None
    )


def test_two_relations_on_the_join_var_are_not_a_join_body() -> None:
    assert (
        parse_join_body(
            "saliency(Ev, strengthen), pre_before_post(Ev, E), other(Ev, E)",
            NEURAL,
            head_var="E",
        )
        is None
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


def test_the_prepared_index_is_the_same_math_as_the_raw_extension() -> None:
    """The hot loop uses the PREPARED index; the raw entry point delegates to it. If
    they could disagree there would be two implementations of the OR."""
    p = torch.tensor([0.9, 0.1, 0.2, 0.85])
    ext = [[0, 1], [2], [], [0, 1, 2, 3]]
    handle = prepare_extension(ext, "cpu")
    assert torch.allclose(
        noisy_or_from_index(p, handle), noisy_or_over_extension(p, ext, "cpu")
    )


def test_the_prepared_index_is_built_once_and_holds_no_python_lists() -> None:
    """The per-step work must be a single vectorized op over device-resident tensors:
    the extension is static, so its indexing is precomputed, not rebuilt per step."""
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
