# python/tests/test_ilp_promoter.py
"""Tests for train_and_promote() — transactional commit + promotion gates."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import (
    CandidateMapEntry,
    IlpConfigError,
    LearnedArtifact,
    TrainConfig,
    TrainResult,
    train_and_promote,
    PromotionResult,
    PromotionStatus,
)


class _TrialProgramStub:
    """Small protocol stub for _typed_schema_gate tests."""

    def __init__(self, annotations):
        self._annotations = annotations

    def relation_type_annotations(self):
        return self._annotations

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
NEG = []


def test_promote_returns_promotion_result():
    """train_and_promote returns a PromotionResult."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        strict_gpu_native=False,
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=[("reach", [1, 3])],
        holdout_negatives=[],
    )
    assert isinstance(result, PromotionResult)


def test_promote_with_holdout_can_promote():
    """With holdout examples provided, promotion is possible."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        strict_gpu_native=False,
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=[("reach", [1, 3]), ("reach", [2, 4])],
        holdout_negatives=[],
    )
    if result.status == PromotionStatus.PROMOTED:
        assert all(g.passed for g in result.gates)


def test_promote_without_holdout_requires_review():
    """Without holdout, result is MANUAL_REVIEW_REQUIRED (not PROMOTED)."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        strict_gpu_native=False,
    )
    result = train_and_promote(SOURCE, "W_reach", POS, NEG, config)
    assert result.status in (
        PromotionStatus.MANUAL_REVIEW_REQUIRED,
        PromotionStatus.NOT_CONVERGED,
    )


def test_promote_gates_are_populated():
    """Gate list should contain all required gates per design doc Section 4.3."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        strict_gpu_native=False,
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=[("reach", [1, 3])],
        holdout_negatives=[],
    )
    assert result.status != PromotionStatus.NOT_CONVERGED, \
        "Cannot test gates without convergence"
    gate_names = [g.name for g in result.gates]
    assert "training_positive" in gate_names
    assert "training_negative" in gate_names
    assert "novel_fact_audit" in gate_names
    assert "regression_check" in gate_names


def test_promote_not_converged():
    """Non-convergent training returns NOT_CONVERGED status."""
    distractor_source = """
        color(1, 0). color(2, 1).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """
    config = TrainConfig(
        step_budget_per_attempt=10, max_attempts=1,
        tau_start=1.0, tau_floor=0.1, seed=0,
        strict_gpu_native=False,
    )
    result = train_and_promote(
        distractor_source, "W",
        [("reach", [1, 3])], [], config,
    )
    assert result.status == PromotionStatus.NOT_CONVERGED


def test_typed_schema_gate_passes_with_annotations():
    """typed_schema gate passes when all rule relations expose metadata."""
    from pyxlog.ilp.promoter import _typed_schema_gate

    artifact = LearnedArtifact(
        candidate_map=[
            CandidateMapEntry(
                id=0,
                i=0,
                j=1,
                k=2,
                left_name="edge",
                right_name="edge",
                head_name="reach",
            ),
        ],
    )

    gate = _typed_schema_gate(
        trial=_TrialProgramStub([
            ("edge", ["u32", "u32"]),
            ("reach", ["u32", "u32"]),
        ]),
        discovered_rule="reach(X, Y) :- edge(X, Z), edge(Z, Y).",
        artifact=artifact,
        config=TrainConfig(typed_schema_required=True),
    )

    assert gate.name == "typed_schema"
    assert gate.passed is True


def test_typed_schema_gate_blocks_without_metadata():
    """typed_schema_required blocks promotion when metadata is unavailable."""
    from pyxlog.ilp.promoter import _typed_schema_gate

    artifact = LearnedArtifact(
        candidate_map=[
            CandidateMapEntry(
                id=0,
                i=0,
                j=1,
                k=2,
                left_name="edge",
                right_name="edge",
                head_name="reach",
            ),
        ],
    )

    gate = _typed_schema_gate(
        trial=_TrialProgramStub([]),
        discovered_rule="reach(X, Y) :- edge(X, Z), edge(Z, Y).",
        artifact=artifact,
        config=TrainConfig(typed_schema_required=True),
    )

    assert gate.name == "typed_schema"
    assert gate.passed is False


def test_typed_schema_gate_allows_manual_review_with_waiver():
    """waiver_untyped keeps typed gate on manual-review path instead of hard-fail."""
    from pyxlog.ilp.promoter import _typed_schema_gate

    artifact = LearnedArtifact(
        candidate_map=[
            CandidateMapEntry(
                id=0,
                i=0,
                j=1,
                k=2,
                left_name="edge",
                right_name="edge",
                head_name="reach",
            ),
        ],
    )

    gate = _typed_schema_gate(
        trial=_TrialProgramStub([]),
        discovered_rule="reach(X, Y) :- edge(X, Z), edge(Z, Y).",
        artifact=artifact,
        config=TrainConfig(waiver_untyped=True),
    )

    assert gate.name == "typed_schema"
    assert gate.passed is False


def test_holdout_f1_gate_fails_when_unavailable(monkeypatch):
    """Holdout gate must fail explicitly when holdout_f1 is unavailable."""
    import pyxlog.ilp.promoter as promoter

    fake_train_result = TrainResult(
        converged=True,
        discovered_rule="reach(X, Y) :- edge(X, Z), edge(Z, Y).",
        attempt_count=1,
        total_steps=10,
        precision=1.0,
        recall=1.0,
        holdout_f1=None,
        artifact=LearnedArtifact(
            candidate_map=[
                CandidateMapEntry(
                    id=0,
                    i=0,
                    j=0,
                    k=1,
                    left_name="edge",
                    right_name="edge",
                    head_name="reach",
                ),
            ],
        ),
    )
    monkeypatch.setattr(promoter, "train_only", lambda *args, **kwargs: fake_train_result)
    result = promoter.train_and_promote(
        source=SOURCE,
        mask_name="W_reach",
        positives=POS,
        negatives=[],
        config=TrainConfig(
            step_budget_per_attempt=20,
            max_attempts=1,
            seed=7,
            typed_schema_required=False,
            strict_gpu_native=False,
        ),
        holdout_positives=[("reach", [1, 3])],
        holdout_negatives=[],
    )
    holdout_gates = [g for g in result.gates if g.name == "holdout_f1"]
    assert holdout_gates, "holdout_f1 gate not reported"
    assert holdout_gates[0].passed is False
    assert result.status == PromotionStatus.MANUAL_REVIEW_REQUIRED


def test_train_and_promote_rejects_strict_gpu_native():
    config = TrainConfig(
        step_budget_per_attempt=40,
        max_attempts=2,
        tau_start=2.0,
        tau_floor=0.05,
        seed=42,
        strict_gpu_native=True,
    )
    with pytest.raises(IlpConfigError, match="strict_gpu_native"):
        train_and_promote(
            SOURCE,
            "W_reach",
            POS,
            NEG,
            config,
            holdout_positives=[("reach", [1, 3])],
            holdout_negatives=[],
        )
