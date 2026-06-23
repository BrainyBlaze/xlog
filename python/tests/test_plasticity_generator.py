import pytest
import torch

from pyxlog.demos.plasticity import (
    make_demo_data,
    make_fixed_split,
    make_weakens_demo_data,
    strengthens,
    weakens,
)
from pyxlog.demos.plasticity.generator import EdgeSample, SALIENCY_THRESHOLD


def test_demo_package_imports() -> None:
    import pyxlog.demos.plasticity as plasticity

    assert hasattr(plasticity, "make_demo_data")


def test_ground_truth_rule_is_prepost_and_high_saliency() -> None:
    assert strengthens(EdgeSample("e", pre_post=True, post_pre=False, saliency=0.9, distractor=0.0))
    assert not strengthens(EdgeSample("e", pre_post=True, post_pre=False, saliency=0.2, distractor=9.0))
    assert not strengthens(EdgeSample("e", pre_post=False, post_pre=True, saliency=0.9, distractor=0.0))
    assert SALIENCY_THRESHOLD == 0.5


def test_fixed_train_split_has_discriminating_cases() -> None:
    split = make_fixed_split("e_tr")
    labels = split.labels()
    # exactly the two strong pre-post edges are positive
    assert [i for i, t in enumerate(labels) if t] == [0, 1]
    # weak pre-post edges exist (relational-only candidate must over-fire on them)
    weak_prepost = [i for i, s in enumerate(split.samples) if s.pre_post and s.saliency < 0.5]
    assert weak_prepost, "need weak pre-post negatives so relational-only fails"
    # phi shape is [N, 2]; column 0 is saliency
    phi = split.phi()
    assert phi.shape == (len(split.samples), 2)
    assert torch.allclose(phi[:, 0], torch.tensor([s.saliency for s in split.samples]))


def test_relational_projections_match_samples() -> None:
    split = make_fixed_split("e_tr")
    assert split.relational_pre_post_ids() == [i for i, s in enumerate(split.samples) if s.pre_post]
    assert split.relational_post_pre_ids() == [i for i, s in enumerate(split.samples) if s.post_pre]


def test_demo_data_splits_are_entity_disjoint() -> None:
    train, held_out = make_demo_data()
    assert train.entity_ids().isdisjoint(held_out.entity_ids())
    # held-out carries a strong pre-post positive (generalize) and a weak pre-post negative (vigilance)
    assert any(s.pre_post and s.saliency >= 0.5 for s in held_out.samples)
    assert any(s.pre_post and s.saliency < 0.5 for s in held_out.samples)


def test_weakens_ground_truth_and_split() -> None:
    assert weakens(EdgeSample("e", pre_post=False, post_pre=True, saliency=0.9, distractor=0.0))
    assert not weakens(EdgeSample("e", pre_post=False, post_pre=True, saliency=0.2, distractor=0.0))
    assert not weakens(EdgeSample("e", pre_post=True, post_pre=False, saliency=0.9, distractor=0.0))
    train, held_out = make_weakens_demo_data()
    # exactly the two strong post-pre edges are positive for the weakens outcome
    assert [i for i, t in enumerate(train.labels_for("weakens")) if t] == [0, 1]
    assert train.entity_ids().isdisjoint(held_out.entity_ids())


def test_labels_for_rejects_unknown_outcome() -> None:
    with pytest.raises(ValueError, match="(?i)outcome"):
        make_fixed_split("e").labels_for("nope")
