"""H_ctx read-side: set-relative de-saturated graded admission mass (Step 2).

The per-entity graded gate ``sigmoid(g_theta - logit(tau))`` saturates to a
near-constant when the head is saturated (real cells: g_theta ~70-200), erasing
the rank numerically. The H_ctx within-set operator replaces it with a
SET-RELATIVE normalization of g_theta (rank-pct over the comparison set, eval
realization), which de-saturates AND preserves the rank.

This test pins the read-side seam: ``_graded_admission_evidence`` accepts the
within-set-norm function (dependency-injected — the helper is authored separately,
the read consumes it); when provided, the neural candidate's graded mass is the
set-relative normalization, so graded_mass becomes rank-faithful even on a
saturated head. When None, the per-entity behavior is unchanged.
"""

import pytest

torch = pytest.importorskip("torch")

from pyxlog.ilp.neurosymbolic import (  # noqa: E402
    NeuralBodyState,
    _GUARD_PREDICATE_PREFIX,
    _graded_admission_evidence,
    _graded_firing_evidence,
    _make_neural_body_head,
)


def _within_set_rankpct_ref(g_theta, group_id, *, mode):
    """Contract-conformant reference for the within_set_norm helper (eval):
    rank-percentile of g_theta within each group, returned as mass in [0, 1]."""
    out = torch.zeros_like(g_theta)
    for gid in group_id.unique():
        idx = (group_id == gid).nonzero().reshape(-1)
        vals = g_theta[idx]
        n = int(idx.numel())
        order = torch.argsort(vals)
        ranks = torch.empty(n, dtype=g_theta.dtype)
        for r, o in enumerate(order.tolist()):
            ranks[o] = (r / (n - 1)) if n > 1 else 0.5
        out[idx] = ranks
    return out


def _saturated_state(logits):
    """A width-1 depth-1 head whose logit = phi[:,0], so features carry the
    (saturated) logits directly; threshold 0.5 -> tau_logit 0."""
    head = _make_neural_body_head(1, 1, 16)
    with torch.no_grad():
        head[0].weight.copy_(torch.tensor([[1.0]]))
        head[0].bias.copy_(torch.tensor([0.0]))
    state = NeuralBodyState(
        state_dict={k: v.detach().cpu() for k, v in head.state_dict().items()},
        width=1, threshold=0.5, head_depth=1, hidden_dim=16,
    )
    features = torch.tensor([[x] for x in logits], dtype=torch.float32)
    return state, features


def _auc(vals, labels):
    pos = [v for v, lab in zip(vals, labels) if lab]
    neg = [v for v, lab in zip(vals, labels) if not lab]
    return sum((1 if p > n else 0.5 if p == n else 0) for p in pos for n in neg) / (
        len(pos) * len(neg)
    )


def test_set_relative_norm_desaturates_graded_mass_on_saturated_head():
    # Saturated, tight logits (like real H_ctx cells): per-entity sigmoid floors
    # to a constant; set-relative rank-pct preserves the g_theta order.
    logits = [200.0, 198.0, 197.0, 195.0]
    labels = [True, True, False, False]  # positives carry the higher logits
    state, features = _saturated_state(logits)
    eligibility = [(_GUARD_PREDICATE_PREFIX + "r", [True, True, True, True])]
    rw = {"r": 0.9}
    nh = {"r": (state, features)}

    saturated = _graded_admission_evidence(eligibility, rw, 4, nh, labels)
    setrel = _graded_admission_evidence(
        eligibility, rw, 4, nh, labels, within_set_norm_fn=_within_set_rankpct_ref
    )

    gm_sat = [r["graded_mass"] for r in saturated["per_query"]]
    gm_rel = [r["graded_mass"] for r in setrel["per_query"]]

    # per-entity path saturates -> constant graded_mass -> rank destroyed (AUC 0.5)
    assert max(gm_sat) - min(gm_sat) < 1e-6
    assert _auc(gm_sat, labels) == pytest.approx(0.5)

    # set-relative path de-saturates -> distinct, rank-faithful (positives outrank)
    assert max(gm_rel) - min(gm_rel) > 0.1
    assert _auc(gm_rel, labels) == pytest.approx(1.0)
    # and the operator output is surfaced per-query for the Axis-III emit
    for r in setrel["per_query"]:
        assert r["within_set_norm"] is not None and 0.0 <= r["within_set_norm"] <= 1.0


def test_none_norm_fn_preserves_surface1_behavior():
    # With no within-set fn the read is byte-identical to the per-entity gate (hard default
    # unchanged): graded_gate stays the per-entity sigmoid, within_set_norm absent.
    logits = [2.0, -1.0]
    state, features = _saturated_state(logits)
    eligibility = [(_GUARD_PREDICATE_PREFIX + "r", [True, True])]
    out = _graded_admission_evidence(
        eligibility, {"r": 0.9}, 2, {"r": (state, features)}, [True, False]
    )
    g = [1.0 / (1.0 + pow(2.718281828, -(x))) for x in logits]  # sigmoid(logit-0)
    for i, r in enumerate(out["per_query"]):
        assert r["graded_gate"] == pytest.approx(g[i], abs=1e-4)
        assert r["within_set_norm"] is None


def test_firing_evidence_groups_by_context_and_fires_top_k_pct():
    # H_ctx firing emit: comparison_set grouped by context_id; within each context
    # the top-k% by within-context rank fires (eligibility-fenced); the emit carries
    # raw g_theta (recompute-from-raw) and production_firing_mass = within-set mass.
    g = [100.0, 102.0, 98.0, 200.0, 196.0, 198.0]
    ctx = ["A", "A", "A", "B", "B", "B"]
    labels = [True, True, False, True, False, True]
    state, features = _saturated_state(g)
    eligibility = [(_GUARD_PREDICATE_PREFIX + "r", [True] * 6)]

    out = _graded_firing_evidence(
        eligibility,
        {"r": 0.9},
        {"r": (state, features)},
        context_ids=ctx,
        firing_rule={"kind": "top_k_pct", "k": 0.5},
        split="heldout",
        labels=labels,
        within_set_norm_fn=_within_set_rankpct_ref,
    )

    assert set(out.keys()) == {"A", "B"}
    for cid in ("A", "B"):
        ctx_ev = out[cid]
        assert ctx_ev["split"] == "heldout"
        assert ctx_ev["firing_rule"]["k"] == 0.5
        cs = ctx_ev["comparison_set"]
        assert len(cs) == 3
        for e in cs:
            assert e["axis1_admissible"] is True  # all g_theta > tau_logit 0
            assert e["g_theta"] is not None  # raw rank carrier for recompute-from-raw
            assert e["production_firing_mass"] == pytest.approx(e["within_set_norm"])
        # top ceil(0.5*3)=2 by within-context rank fire; lowest g_theta does not
        assert sum(1 for e in cs if e["fired"]) == 2
        by_g = sorted(cs, key=lambda e: e["g_theta"])
        assert by_g[0]["fired"] is False
        assert by_g[1]["fired"] is True and by_g[2]["fired"] is True


def test_firing_eligibility_fence_non_admissible_never_fires():
    # An entity below tau (hard ST gate False) is NOT axis1-admissible and must
    # never fire, regardless of its within-context rank (guard 1).
    g = [5.0, 4.0, -3.0]  # third is below tau_logit 0 -> not admissible
    ctx = ["C", "C", "C"]
    state, features = _saturated_state(g)
    eligibility = [(_GUARD_PREDICATE_PREFIX + "r", [True, True, True])]
    out = _graded_firing_evidence(
        eligibility, {"r": 0.9}, {"r": (state, features)},
        context_ids=ctx, firing_rule={"kind": "top_k_pct", "k": 1.0},
        split="heldout", labels=[True, True, False],
        within_set_norm_fn=_within_set_rankpct_ref,
    )
    cs = {e["x"]: e for e in out["C"]["comparison_set"]}
    assert cs[2]["axis1_admissible"] is False
    assert cs[2]["fired"] is False  # never fires even at k=1.0
    assert cs[0]["fired"] is True and cs[1]["fired"] is True  # admissible fire
