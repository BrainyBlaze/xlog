# python/tests/test_ilp_types.py
"""Tests for dILP type definitions and exceptions."""

import pyxlog.ilp as ilp

from pyxlog.ilp import (
    TrainConfig,
    TrainResult,
    PromotionStatus,
    IlpConfigError,
    IlpCandidateError,
    IlpTrainingError,
)


def test_train_config_frozen():
    cfg = TrainConfig()
    try:
        cfg.tau_start = 99.0  # type: ignore
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.global_step_limit == 1000
    assert cfg.max_attempts == 5
    assert cfg.max_mined_negatives == 0
    assert cfg.tau_floor == 0.05
    assert cfg.max_active_rules == 32
    assert cfg.allow_recursive_candidates is False
    assert cfg.strict_gpu_native is True
    assert cfg.max_numeric_failures == 3
    assert cfg.device == 0
    assert cfg.memory_mb == 512


def test_train_config_custom():
    cfg = TrainConfig(tau_start=3.0, max_attempts=10, seed=42)
    assert cfg.tau_start == 3.0
    assert cfg.max_attempts == 10
    assert cfg.seed == 42


def test_promotion_status_values():
    assert PromotionStatus.PROMOTED.value == "promoted"
    assert PromotionStatus.MANUAL_REVIEW_REQUIRED.value == "manual_review_required"
    assert PromotionStatus.COMMIT_FAILED.value == "commit_failed"
    assert PromotionStatus.NOT_CONVERGED.value == "not_converged"
    assert PromotionStatus.GATE_FAILED.value == "gate_failed"


def test_train_result_defaults():
    result = TrainResult()
    assert result.converged is False
    assert result.discovered_rule is None
    assert result.attempt_count == 0


def test_strict_result_and_artifact_types_are_exported():
    assert getattr(ilp, "StrictTrainResult", None) is not None
    assert getattr(ilp, "StrictLearnedArtifact", None) is not None


def test_ilp_config_error_is_value_error():
    assert issubclass(IlpConfigError, ValueError)


def test_ilp_candidate_error_is_value_error():
    assert issubclass(IlpCandidateError, ValueError)


def test_ilp_training_error_has_context():
    err = IlpTrainingError("CUDA OOM", {"attempt": 2, "step": 50, "C": 100})
    assert err.context["attempt"] == 2
    assert err.context["step"] == 50
    assert "CUDA OOM" in str(err)


def test_ilp_training_error_default_context():
    err = IlpTrainingError("generic failure")
    assert err.context == {}
