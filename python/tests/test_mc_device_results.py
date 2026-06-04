import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")


def test_mc_device_results_returns_dlpack_counts():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    source = """
1.0::base().
query(base()).
"""

    program = pyxlog.Program.compile(source, prob_engine="mc")
    result = program.evaluate_device(samples=4096, seed=0)

    from torch.utils.dlpack import from_dlpack

    query_counts = from_dlpack(result.query_counts).cpu()
    evidence_count = from_dlpack(result.evidence_count).cpu()

    assert query_counts.numel() == 1
    assert evidence_count.numel() == 1
    assert query_counts.item() == result.total_samples
    assert evidence_count.item() == result.total_samples
    assert result.seed == 0
    assert result.resident_no_host_certified is True
    assert result.resident_no_host_policy_result == "passed"
    assert result.resident_no_host_tracked_htod_calls == 0
    assert result.resident_no_host_tracked_dtoh_calls == 0
    assert result.resident_no_host_host_loop_iterations == 0
    assert result.resident_no_host_host_fixpoint_iterations == 0
    assert result.resident_no_host_per_operator_host_allocations == 0
    assert result.resident_no_host_per_sample_host_launches == 0
