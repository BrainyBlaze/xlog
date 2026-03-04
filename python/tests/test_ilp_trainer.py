# python/tests/test_ilp_trainer.py
"""Integration tests for train_only()."""
import math

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig, TrainResult, IlpConfigError

REACH_SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
REACH_POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
REACH_NEG = []


def test_train_only_converges_on_reach():
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    assert isinstance(result, TrainResult)
    assert result.converged
    assert "edge" in result.discovered_rule
    assert "reach" in result.discovered_rule
    assert result.attempt_count >= 1
    assert result.total_steps > 0


def test_train_only_returns_telemetry_level_1():
    config = TrainConfig(
        step_budget_per_attempt=50, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
        telemetry_level=1,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    assert len(result.artifact.telemetry.steps) > 0
    step0 = result.artifact.telemetry.steps[0]
    assert step0.step == 0
    assert isinstance(step0.loss, float)
    assert isinstance(step0.temperature, float)


def test_train_only_forward_p95_telemetry_populated():
    """Telemetry must include forward-pass timing in microseconds."""
    config = TrainConfig(
        step_budget_per_attempt=20, max_attempts=2,
        tau_start=2.0, tau_floor=0.05, seed=13,
        telemetry_level=1,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    steps = result.artifact.telemetry.steps
    assert len(steps) > 0
    step_forward = [s.forward_p95_us for s in steps]
    assert all(v >= 0.0 for v in step_forward)
    assert max(step_forward) >= 0.0

    timing = result.artifact.telemetry.step_timings
    assert isinstance(timing, dict)
    assert "forward_p95_us" in timing
    sorted_forward = sorted(step_forward)
    p95_idx = min(len(sorted_forward) - 1, max(0, math.ceil(0.95 * len(sorted_forward)) - 1))
    assert timing["forward_p95_us"] >= sorted_forward[p95_idx] - 1e-6


def test_train_only_telemetry_step_timings_population():
    """Telemetry step summary should include forward and memory stats."""
    config = TrainConfig(
        step_budget_per_attempt=40, max_attempts=2,
        tau_start=2.0, tau_floor=0.05, seed=13,
        telemetry_level=1,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    timing = result.artifact.telemetry.step_timings
    assert isinstance(timing, dict)
    assert "forward_p95_us" in timing
    assert timing["forward_p95_us"] >= 0.0
    assert "allocated_bytes_p95" in timing
    assert "allocated_bytes_max" in timing


def test_train_only_empty_positives_raises():
    config = TrainConfig(max_attempts=1)
    with pytest.raises(IlpConfigError, match="positives"):
        train_only(
            source=REACH_SOURCE, mask_name="W_reach",
            positives=[], negatives=[], config=config,
        )


def test_train_only_contradictory_examples_raises():
    config = TrainConfig(max_attempts=1)
    pos = [("reach", [1, 3])]
    neg = [("reach", [1, 3])]
    with pytest.raises(IlpConfigError, match="contradict"):
        train_only(
            source=REACH_SOURCE, mask_name="W_reach",
            positives=pos, negatives=neg, config=config,
        )


def test_train_only_precision_recall():
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    if result.converged:
        assert result.precision > 0.0
        assert result.recall > 0.0


def test_train_only_confidence_metrics():
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    if result.converged:
        assert 0.0 <= result.confidence_margin <= 1.0
        assert 0.0 <= result.top_k_concentration <= 1.0 + 1e-6
        assert 0.0 <= result.rule_frequency <= 1.0


def test_train_only_global_step_limit():
    config = TrainConfig(
        global_step_limit=50, step_budget_per_attempt=30,
        max_attempts=10, tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    assert result.total_steps <= 50


def test_train_only_nonconverged_has_partial_recall():
    """Non-converged result should report partial recall, not 0.0."""
    # Mix reachable + impossible positive → forces non-convergence
    pos_mixed = [("reach", [1, 3]), ("reach", [99, 99])]
    config = TrainConfig(
        step_budget_per_attempt=30, max_attempts=1,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=pos_mixed, negatives=REACH_NEG, config=config,
    )
    assert not result.converged
    # reach(1,3) is derivable from edge joins, so recall should be > 0
    assert result.recall > 0.0, "Non-converged recall should reflect partial witness coverage"


def test_train_only_reproducibility_selected_hard_and_probs():
    """Deterministic mode stabilizes learned artifacts across runs."""
    config = TrainConfig(
        step_budget_per_attempt=120,
        max_attempts=4,
        tau_start=2.0,
        tau_floor=0.05,
        seed=123,
        deterministic=True,
        debug_dense_mask=False,
    )
    a = train_only(
        source=REACH_SOURCE,
        mask_name="W_reach",
        positives=REACH_POS,
        negatives=REACH_NEG,
        config=config,
    )
    b = train_only(
        source=REACH_SOURCE,
        mask_name="W_reach",
        positives=REACH_POS,
        negatives=REACH_NEG,
        config=config,
    )

    assert a.discovered_rule == b.discovered_rule
    assert a.artifact.selected_hard == b.artifact.selected_hard
    assert len(a.artifact.soft_probs) == len(b.artifact.soft_probs)
    assert a.converged is True
    assert b.converged is True
    for p, q in zip(a.artifact.soft_probs, b.artifact.soft_probs):
        assert abs(p - q) <= max(1e-6, abs(q) * 1e-5 + 1e-6)


def test_train_only_telemetry_entropy_not_zero():
    """Telemetry entropy should reflect actual candidate distribution."""
    config = TrainConfig(
        step_budget_per_attempt=20, max_attempts=1,
        tau_start=2.0, tau_floor=0.05, seed=42,
        telemetry_level=1,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    steps = result.artifact.telemetry.steps
    assert len(steps) > 0
    # Early steps with high tau should have non-zero entropy
    assert any(s.entropy > 0.0 for s in steps), \
        "At least some telemetry steps should have non-zero entropy"


def test_train_only_phase_timing_keys():
    """Phase timing breakdown must include all expected keys."""
    config = TrainConfig(
        step_budget_per_attempt=20, max_attempts=2,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    timing = result.artifact.telemetry.step_timings
    expected_p95 = [
        "apply_mask_p95_us", "loss_credit_p95_us", "loss_reduce_p95_us",
        "backward_step_p95_us", "membership_p95_us", "convergence_p95_us",
    ]
    expected_total = [
        "apply_mask_total_ms", "loss_credit_total_ms", "loss_reduce_total_ms",
        "backward_step_total_ms", "membership_total_ms", "convergence_total_ms",
    ]
    for key in expected_p95 + expected_total:
        assert key in timing, f"Missing timing key: {key}"
        assert timing[key] >= 0.0, f"Negative timing: {key}={timing[key]}"


def test_train_only_rule_frequency_multi_attempt():
    """rule_frequency reflects how many attempts found the winning rule."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    if result.converged:
        # At least the winning attempt found this rule
        assert result.rule_frequency >= 1.0 / result.attempt_count
