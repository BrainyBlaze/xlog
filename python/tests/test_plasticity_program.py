from pyxlog.demos.plasticity.generator import make_fixed_split
from pyxlog.demos.plasticity.program import (
    CAND_PREPOST_NEURAL,
    CAND_PREPOST_REL,
    CAND_POSTPRE_NEURAL,
    TRAIN_HEAD,
    build_neural_bodies,
    build_source,
)


def test_source_declares_facts_and_three_candidates() -> None:
    split = make_fixed_split("e_tr")
    source = build_source(split)
    # head-bound projected relations as ground facts at binding indices
    assert "edge_pre_post(0)." in source
    assert "edge_pre_post(2)." in source  # weak pre-post still a fact (gate must reject it)
    assert "edge_post_pre(4)." in source
    # three same-head trainable candidates
    for cand in (CAND_PREPOST_REL, CAND_PREPOST_NEURAL, CAND_POSTPRE_NEURAL):
        assert f"trainable_rule({cand}" in source
    assert f"train({TRAIN_HEAD}, binary_cross_entropy)." in source
    # the relational-only and neural pre-post candidates share the SAME body
    assert source.count("edge_pre_post(E)") >= 2


def test_neural_bodies_cover_the_two_neural_candidates() -> None:
    split = make_fixed_split("e_tr")
    bodies = build_neural_bodies(split)
    assert set(bodies) == {CAND_PREPOST_NEURAL, CAND_POSTPRE_NEURAL}
    assert bodies[CAND_PREPOST_NEURAL].features.shape == (split.num_queries(), 2)
    assert bodies[CAND_PREPOST_NEURAL].threshold == 0.5
