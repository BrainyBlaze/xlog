import pytest

torch = pytest.importorskip("torch")

from pyxlog.ilp.join_bodies import JoinBody, noisy_or_over_extension, parse_join_body

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
