"""Graded admission read (set-relative admission-rank evidence, SAFE_GRADED).

The graded read swaps the hard ST gate for the GRADED gate
``g_tilde = sigmoid((g_theta - logit(tau)) / temp)`` and emits the decomposed,
auditable per-query evidence the locked two-axis rubric consumes. Production
firing stays the hard default; graded is opt-in and carries NO production-firing
certification — it is rank-preserving graded SUPPORT, not calibrated truth.

These tests pin the graded computation as a pure function of (eligibility,
neural state, features, guard weight, labels) — no engine, no CUDA — so the
audit invariant (hard_head_prob and graded_mass run through the SAME
rel_mask*.*guard structure, only the gate kind differs) is checked directly.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from pyxlog.ilp.neurosymbolic import (  # noqa: E402
    NeuralBodyState,
    _GUARD_PREDICATE_PREFIX,
    _graded_admission_evidence,
    _make_neural_body_head,
)


def _linear_state(weight_row, bias):
    """A width-2 depth-1 head whose logit = weight_row . phi + bias, serialized."""
    head = _make_neural_body_head(2, 1, 16)
    with torch.no_grad():
        head[0].weight.copy_(torch.tensor([weight_row], dtype=torch.float32))
        head[0].bias.copy_(torch.tensor([bias], dtype=torch.float32))
    return NeuralBodyState(
        state_dict={k: v.detach().cpu() for k, v in head.state_dict().items()},
        width=2,
        threshold=0.5,
        head_depth=1,
        hidden_dim=16,
    )


def test_graded_admission_evidence_decomposes_and_matches_hand_computation():
    # logit = phi[:,0]; tau=0.5 -> tau_logit=0; rel mask [1,1,0]; guard sigmoid 0.9.
    state = _linear_state([1.0, 0.0], 0.0)
    features = torch.tensor([[2.0, 0.0], [-1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
    eligibility = [(_GUARD_PREDICATE_PREFIX + "rule_x", [True, True, False])]
    rule_weights = {"rule_x": 0.9}
    neural_heldout = {"rule_x": (state, features)}
    labels = [True, True, False]

    out = _graded_admission_evidence(
        eligibility, rule_weights, 3, neural_heldout, labels
    )

    assert out["mode"] == "graded"
    pq = out["per_query"]
    assert [r["query_index"] for r in pq] == [0, 1, 2]

    g = [1.0 / (1.0 + math.exp(-x)) for x in (2.0, -1.0, 3.0)]  # graded gate = sigmoid(logit)
    rel = [1.0, 1.0, 0.0]
    for i, r in enumerate(pq):
        assert r["selected_rule_id"] == "rule_x"
        assert r["relational_mask"] == pytest.approx(rel[i])
        assert r["g_theta"] == pytest.approx([2.0, -1.0, 3.0][i], abs=1e-5)
        assert r["tau_logit"] == pytest.approx(0.0, abs=1e-6)
        # graded gate in (0,1); hard gate in {0,1}
        assert 0.0 < r["graded_gate"] < 1.0
        assert r["hard_gate"] in (0.0, 1.0)
        assert r["graded_gate"] == pytest.approx(g[i], abs=1e-5)
        assert r["hard_gate"] == pytest.approx(1.0 if [2.0, -1.0, 3.0][i] >= 0 else 0.0)
        # single-candidate noisy-OR collapses to rel*gate*guard
        assert r["graded_mass"] == pytest.approx(rel[i] * g[i] * 0.9, abs=1e-5)
        assert r["hard_head_prob"] == pytest.approx(
            rel[i] * (1.0 if [2.0, -1.0, 3.0][i] >= 0 else 0.0) * 0.9, abs=1e-5
        )
        # production firing mass = the selected winner's per-entity graded gate
        assert r["production_firing_mass"] == pytest.approx(g[i], abs=1e-5)
        assert r["label"] is labels[i]

    # axis1_margin is the LOGIT-space margin (matches the locked rubric, whose
    # LOW_MARGIN annotation is "< 1.0 logit"): min g_theta(pos) - max g_theta(neg).
    pos_logits = [2.0, -1.0]  # labels[0], labels[1] are True
    neg_logits = [3.0]  # labels[2] is False
    assert out["axis1_margin"] == pytest.approx(
        min(pos_logits) - max(neg_logits), abs=1e-5
    )


def test_graded_audit_invariant_same_structure_zero_mask_zeros_both():
    # Where the relational mask is 0, BOTH hard and graded head-prob must be 0 —
    # the gate kind cannot resurrect a relationally-absent grounding.
    state = _linear_state([1.0, 0.0], 0.0)
    features = torch.tensor([[5.0, 0.0], [5.0, 0.0]], dtype=torch.float32)
    eligibility = [(_GUARD_PREDICATE_PREFIX + "rule_x", [True, False])]
    out = _graded_admission_evidence(
        eligibility, {"rule_x": 0.95}, 2, {"rule_x": (state, features)}, None
    )
    pq = out["per_query"]
    assert pq[1]["relational_mask"] == 0.0
    assert pq[1]["graded_mass"] == 0.0
    assert pq[1]["hard_head_prob"] == 0.0
    # labels absent -> axis1_margin is None (no retention frame)
    assert out["axis1_margin"] is None
    assert pq[0]["label"] is None


def test_evaluate_joint_mixture_rejects_unknown_mode_before_engine():
    # An unknown mode must fail fast with a typed ValueError BEFORE any engine
    # compile — so callers learn the contract without a confusing engine error.
    from pyxlog.ilp.neurosymbolic import evaluate_joint_mixture

    with pytest.raises(ValueError, match="(?i)mode"):
        evaluate_joint_mixture(
            "p(0). pred p(i64). trainable_rule(r, weight=0.0) :: q(C) :- p(C). "
            "train(q, binary_cross_entropy).",
            rule_weights={},
            num_queries=1,
            mode="bogus",
        )
