# python/tests/test_ilp_beta_gate.py
"""Beta reliability gate: 20/20 on sparse backend.

Extends alpha gate with sparse mask backend (debug_dense_mask=False).
"""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig

# Re-use stage definitions from reliability
from test_ilp_reliability import STAGES


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "stage_name,source,positives,negatives,mask_name",
    STAGES,
    ids=["reach", "grandparent", "colleague", "plus2"],
)
def test_beta_reliability_sparse(
    stage_name, source, positives, negatives, mask_name, seed,
):
    """Each stage converges with sparse backend (beta gate: 5/5 per stage)."""
    config = TrainConfig(
        step_budget_per_attempt=150,
        max_attempts=7,
        tau_start=2.0,
        tau_floor=0.05,
        seed=seed,
        device=0,
        memory_mb=512,
        debug_dense_mask=False,  # sparse backend
    )
    result = train_only(
        source=source,
        mask_name=mask_name,
        positives=positives,
        negatives=negatives,
        config=config,
    )
    assert result.converged, (
        f"Beta gate: {stage_name} failed with seed={seed}, "
        f"attempts={result.attempt_count}, steps={result.total_steps}"
    )
