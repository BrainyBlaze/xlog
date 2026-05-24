"""Regression coverage for UCR-XLOG-006."""

import pytest

from pyxlog.transfer_diagnostics import (
    PredictionRecord,
    compute_transfer_diagnostics,
)


def test_grouped_transfer_diagnostics_recompute_lodo_metrics() -> None:
    records = [
        PredictionRecord("cyber", "clean", 1, 1, baseline_pred=0),
        PredictionRecord("cyber", "clean", 0, 0, baseline_pred=0),
        PredictionRecord("cyber", "adversarial", 1, 1, baseline_pred=0),
        PredictionRecord("cyber", "adversarial", 0, 1, baseline_pred=0),
        PredictionRecord("medical", "clean", 1, 1, baseline_pred=0),
        PredictionRecord("medical", "clean", 0, 0, baseline_pred=0),
        PredictionRecord("medical", "adversarial", 1, 0, baseline_pred=0),
        PredictionRecord("medical", "adversarial", 0, 0, baseline_pred=0),
    ]

    diagnostics = compute_transfer_diagnostics(
        records,
        required_domains=("cyber", "medical"),
        required_variants=("clean", "adversarial"),
        bootstrap_samples=64,
        seed=7,
    )

    assert diagnostics.passed is True
    assert diagnostics.per_domain["cyber"].f1 == pytest.approx(0.8)
    assert diagnostics.per_domain["medical"].f1 == pytest.approx(2 / 3)
    assert diagnostics.macro_f1 == pytest.approx((0.8 + (2 / 3)) / 2)
    assert diagnostics.minimum_domain_f1 == pytest.approx(2 / 3)
    assert diagnostics.baseline_uplift > 0.0
    assert diagnostics.bootstrap_ci.lower <= diagnostics.macro_f1 <= diagnostics.bootstrap_ci.upper
    assert diagnostics.paired_sign_test is not None


def test_grouped_transfer_diagnostics_fail_missing_required_domain() -> None:
    diagnostics = compute_transfer_diagnostics(
        [PredictionRecord("cyber", "clean", 1, 1, baseline_pred=0)],
        required_domains=("cyber", "finance"),
        required_variants=("clean",),
        bootstrap_samples=0,
    )

    assert diagnostics.passed is False
    assert diagnostics.missing_domains == ["finance"]
