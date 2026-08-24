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
    assert result.resident_no_host_policy_result == "certified"
    assert result.resident_no_host_tracked_dtoh_calls == 0
    assert result.resident_no_host_tracked_htod_calls == 0
    assert result.resident_no_host_host_loop_iterations == 0
    assert result.resident_no_host_per_sample_host_launches == 0
    assert result.resident_no_host_untracked_metadata_reads == 0
    assert result.resident_no_host_host_fixpoint_iterations == 0
    assert result.resident_no_host_per_operator_host_allocations == 0
    assert result.resident_no_host_engine_launches >= 1


def test_python_mc_evaluation_inherits_directives_and_explicit_args_override():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    source = """
#pragma prob_engine = mc
#pragma prob_samples = 64
#pragma prob_seed = 17
#pragma prob_confidence = 0.8
#pragma prob_method = evidence_clamping
#pragma prob_max_nonmonotone_iterations = 7

1.0::base().
query(base()).
"""

    program = pyxlog.Program.compile(source)

    host_result = program.evaluate()
    assert host_result.samples == 64
    assert host_result.seed == 17
    assert host_result.confidence == pytest.approx(0.8)
    assert host_result.sampling_method == "evidence_clamping"

    device_result = program.evaluate_device()
    assert device_result.total_samples == 64
    assert device_result.seed == 17
    assert device_result.confidence == pytest.approx(0.8)
    assert device_result.sampling_method == "evidence_clamping"

    host_override = program.evaluate(
        samples=96,
        seed=29,
        confidence=0.9,
        max_nonmonotone_iterations=11,
        sampling_method="rejection",
    )
    assert host_override.samples == 96
    assert host_override.seed == 29
    assert host_override.confidence == pytest.approx(0.9)
    assert host_override.sampling_method == "rejection"

    device_override = program.evaluate_device(
        samples=96,
        seed=29,
        confidence=0.9,
        max_nonmonotone_iterations=11,
        sampling_method="rejection",
    )
    assert device_override.total_samples == 96
    assert device_override.seed == 29
    assert device_override.confidence == pytest.approx(0.9)
    assert device_override.sampling_method == "rejection"


def test_exact_evaluation_rejects_explicit_mc_only_options():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    program = pyxlog.Program.compile(
        """
#pragma prob_engine = exact_ddnnf
0.5::rain().
query(rain()).
"""
    )

    with pytest.raises(ValueError, match="only supported for prob_engine='mc'"):
        program.evaluate(confidence=0.9)
