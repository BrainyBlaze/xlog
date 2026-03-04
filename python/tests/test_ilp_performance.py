"""ILP performance and transfer-accounting smoke tests."""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
from pyxlog.ilp import TrainConfig, train_only

skip_unless_pyxlog_cuda()


def test_forward_and_memory_telemetry_smoke():
    """Telemetry must always include forward and memory summaries."""
    config = TrainConfig(
        step_budget_per_attempt=80,
        max_attempts=3,
        tau_start=2.0,
        tau_floor=0.05,
        seed=13,
        deterministic=True,
    )
    result = train_only(
        source="""
            edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
            learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
        """,
        mask_name="W_reach",
        positives=[("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5])],
        negatives=[],
        config=config,
        _compute_holdout=False,
    )

    timings = result.artifact.telemetry.step_timings
    assert "forward_p95_us" in timings
    assert "allocated_bytes_p95" in timings
    assert "allocated_bytes_max" in timings
    assert timings["forward_p95_us"] >= 0.0
    assert timings["allocated_bytes_p95"] >= 0.0
    assert timings["allocated_bytes_max"] >= timings["allocated_bytes_p95"]


def test_host_transfer_stats_tracks_sparse_mask_download():
    """Sparse mask setup should record dtoh bytes/calls for transfer accounting."""
    prog = pyxlog.IlpProgramFactory.compile(
        """
        edge(1,2). edge(2,3). edge(3,4).
        learnable(W) :: reach(X,Y) :- bL(X,Z), bR(Z,Y).
        """,
        device=0,
        memory_mb=64,
    )
    candidates = prog.valid_candidates("W", False)
    soft = torch.tensor([0.5] * len(candidates), device="cuda", dtype=torch.float64)

    prog.reset_host_transfer_stats()
    prog.set_rule_mask_sparse("W", list(range(len(candidates))), soft, 32)
    after = prog.host_transfer_stats()
    assert after["dtoh_calls"] > 0
    assert after["dtoh_bytes"] > 0


def _build_reach_chain_source(length: int) -> str:
    edges = "\n".join([f"edge({i}, {i+1})." for i in range(1, length)])
    return f"""
    {edges}
    learnable(W_reach_chain) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """


def _chain_positives(length: int):
    return [("reach", [i, i + 2]) for i in range(1, length - 1)]


@pytest.mark.slow
def test_forward_performance_slo_n50_smoke():
    """Optional slow benchmark verifies forward timing telemetry on a larger schema."""
    length = 50
    source = _build_reach_chain_source(length)
    config = TrainConfig(
        step_budget_per_attempt=150,
        max_attempts=4,
        tau_start=2.0,
        tau_floor=0.05,
        seed=7,
        deterministic=True,
        max_active_rules=16,
        telemetry_level=1,
    )
    result = train_only(
        source=source,
        mask_name="W_reach_chain",
        positives=_chain_positives(length),
        negatives=[],
        config=config,
        _compute_holdout=False,
    )

    forward_p95_us = result.artifact.telemetry.step_timings.get("forward_p95_us", 0.0)
    alloc_max = result.artifact.telemetry.step_timings.get("allocated_bytes_max", 0.0)
    assert forward_p95_us > 0.0
    assert alloc_max >= 0.0

    # Optional strict GA-performance check can be enabled in CI stage.
    if os.getenv("ILP_PERF_ENFORCE_SLO", "0") == "1":
        assert forward_p95_us <= 500_000.0, (
            f"forward p95 {forward_p95_us}us exceeds N<=150 GA SLO target 500000us"
        )
        assert alloc_max <= 100 * 1024 * 1024, (
            f"alloc max {alloc_max} exceeds N<=150 GA memory target 100MB"
        )
