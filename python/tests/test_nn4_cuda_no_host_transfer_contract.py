"""Regression coverage for UCR-XLOG-004."""

import pytest

torch = pytest.importorskip("torch")

from pyxlog.runtime_audit import CudaExecutionAudit, HostMaterializationError


def test_cuda_execution_audit_reports_zero_for_device_resident_scores() -> None:
    audit = CudaExecutionAudit(forbid_host_materialization=True)

    with audit:
        scores = torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32)
        audit.record_nn4_scores("ranker", scores, device_resident=True)
        _ = scores + 1.0

    summary = audit.summary()
    assert summary.passed is True
    assert summary.d2h_transfers == 0
    assert summary.h2d_transfers == 0
    assert summary.scalar_extractions == 0
    assert summary.score_row_downloads == 0


def test_cuda_execution_audit_fails_on_score_row_download() -> None:
    audit = CudaExecutionAudit(forbid_host_materialization=True)

    with pytest.raises(HostMaterializationError, match="tolist"):
        with audit:
            scores = torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32)
            audit.record_nn4_scores("ranker", scores, device_resident=True)
            scores.tolist()

    summary = audit.summary()
    assert summary.passed is False
    assert summary.d2h_transfers == 1
    assert summary.score_row_downloads == 1
    assert summary.violations[0].operation == "tolist"


def test_cuda_execution_audit_fails_on_scalar_extraction() -> None:
    audit = CudaExecutionAudit(forbid_host_materialization=True)

    with pytest.raises(HostMaterializationError, match="item"):
        with audit:
            score = torch.tensor(0.9, dtype=torch.float32)
            score.item()

    summary = audit.summary()
    assert summary.passed is False
    assert summary.scalar_extractions == 1
    assert summary.violations[0].operation == "item"
