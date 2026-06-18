"""Unit tests for the within-comparison-set reduction (`within_set_norm`).

These validate the algorithmic claims the operator rests on (the within-set
normalization circuit substrate): a [0,1] WMC-leaf mass, monotone-in-g within a comparison set,
train/eval rank-identity, offset-cancellation (z-norm makes a set-wide additive
offset vanish AND the bias receive no gradient), degenerate groups -> neutral
0.5, and independent per-group segmentation. Pure torch (CPU); no engine/Rust.

Run: `PYTHONPATH=crates/pyxlog/python python3 -m pytest python/tests/test_within_set_norm.py -q`
or directly: `PYTHONPATH=crates/pyxlog/python python3 python/tests/test_within_set_norm.py`
"""

import torch

from pyxlog.ilp.neurosymbolic import within_set_norm


def test_returns_unit_interval_both_modes():
    g = torch.tensor([70.0, 120.0, 200.0, 95.0])  # saturating-regime logits
    grp = torch.tensor([0, 0, 0, 0])
    for mode in ("train", "eval"):
        m = within_set_norm(g, grp, mode=mode)
        assert m.shape == g.shape
        assert torch.all(m >= 0.0) and torch.all(m <= 1.0), mode


def test_desaturates_where_per_entity_sigmoid_saturates():
    # Per-entity sigmoid(g - tau) collapses to ~1.0 for all (g in [70,200]);
    # within-set normalization must spread the mass and keep the g-rank.
    g = torch.tensor([70.0, 120.0, 200.0, 95.0])
    grp = torch.tensor([0, 0, 0, 0])
    sat = torch.sigmoid(g - 0.0)  # the saturating baseline
    assert float(sat.max() - sat.min()) < 1e-3  # confirms saturation
    for mode in ("train", "eval"):
        m = within_set_norm(g, grp, mode=mode)
        assert float(m.max() - m.min()) > 0.2, mode  # de-saturated spread


def test_monotonic_in_g_within_group():
    g = torch.tensor([1.0, 5.0, 2.0, 9.0, 3.0])
    grp = torch.tensor([0, 0, 0, 0, 0])
    order = torch.argsort(g)
    for mode in ("train", "eval"):
        m = within_set_norm(g, grp, mode=mode)
        ranked = m[order]
        assert torch.all(ranked[1:] >= ranked[:-1] - 1e-6), mode


def test_rank_identity_train_eval():
    # Both realizations are monotone in g within a group -> rank-identical.
    g = torch.tensor([3.1, -2.0, 8.7, 0.5, 4.4, -7.0])
    grp = torch.tensor([0, 0, 0, 0, 0, 0])
    mt = within_set_norm(g, grp, mode="train")
    me = within_set_norm(g, grp, mode="eval")
    assert torch.equal(torch.argsort(mt), torch.argsort(me))


def test_offset_invariance_train():
    # z-norm cancels a set-wide additive offset exactly (the non-transferring
    # component) -> mass unchanged under g -> g + c.
    g = torch.tensor([10.0, 40.0, 90.0, 25.0])
    grp = torch.tensor([0, 0, 0, 0])
    m1 = within_set_norm(g, grp, mode="train")
    m2 = within_set_norm(g + 75.0, grp, mode="train")
    assert torch.allclose(m1, m2, atol=1e-5)


def test_eval_invariant_under_monotone_shift():
    g = torch.tensor([10.0, 40.0, 90.0, 25.0])
    grp = torch.tensor([0, 0, 0, 0])
    m1 = within_set_norm(g, grp, mode="eval")
    m2 = within_set_norm(g + 75.0, grp, mode="eval")  # monotone (additive) shift
    assert torch.allclose(m1, m2, atol=1e-6)


def test_bias_receives_no_gradient_train():
    # The proof: d(z-norm)/d(set-wide bias) = 0. A uniform shift added to every
    # member of a group must produce ~zero gradient on that shift.
    base = torch.tensor([10.0, 40.0, 90.0, 25.0])
    grp = torch.tensor([0, 0, 0, 0])
    b = torch.zeros(1, requires_grad=True)
    g = base + b  # broadcast uniform offset over the single group
    m = within_set_norm(g, grp, mode="train")
    m.sum().backward()
    assert b.grad is not None
    assert abs(float(b.grad)) < 1e-4


def test_train_is_differentiable_in_g():
    g = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    grp = torch.tensor([0, 0, 0, 0])
    m = within_set_norm(g, grp, mode="train")
    m.sum().backward()
    assert g.grad is not None
    assert float(g.grad.abs().sum()) > 0.0  # within-group variation -> real grad


def test_degenerate_groups_return_neutral_half():
    # singleton group and zero-spread group both carry no within-set signal.
    g = torch.tensor([5.0, 99.0, 99.0, 99.0])  # entity 0 alone; {1,2,3} all-equal
    grp = torch.tensor([0, 1, 1, 1])
    for mode in ("train", "eval"):
        m = within_set_norm(g, grp, mode=mode)
        assert abs(float(m[0]) - 0.5) < 1e-6, mode  # singleton -> 0.5
        if mode == "train":
            # all-equal group has zero spread -> neutral
            assert torch.allclose(m[1:], torch.full((3,), 0.5), atol=1e-6)


def test_segmentation_is_per_group():
    # Same raw value normalizes differently depending on its group; an entity's
    # mass depends only on its own comparison set.
    g = torch.tensor([0.0, 100.0, 50.0, 50.0, 100.0, 0.0])
    grp = torch.tensor([0, 0, 0, 1, 1, 1])
    m = within_set_norm(g, grp, mode="eval")
    # group 0 = {0,100,50}: the 50 is the MIDDLE -> ~0.5
    assert abs(float(m[2]) - 0.5) < 1e-6
    # group 1 = {50,100,0}: the 50 is also the MIDDLE -> ~0.5
    assert abs(float(m[3]) - 0.5) < 1e-6
    # the 100 in group 0 is the max -> > 0.5; the 0 in group 1 is the min -> < 0.5
    assert float(m[1]) > 0.5 and float(m[5]) < 0.5


def test_invalid_mode_raises():
    g = torch.tensor([1.0, 2.0])
    grp = torch.tensor([0, 0])
    try:
        within_set_norm(g, grp, mode="bogus")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for bad mode")


def test_length_mismatch_raises():
    try:
        within_set_norm(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0, 0]), mode="train")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for length mismatch")


if __name__ == "__main__":  # allow running without pytest
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} within_set_norm tests passed.")
