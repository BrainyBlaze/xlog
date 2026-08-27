from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

EXAMPLES = [
    "01_async_streaming_reachability",
    "02_relation_deltas",
    "03_neural_bridge_topk_belnap",
    "04_native_exact_induction",
    "05_probabilistic_async_diagnostics",
]


def test_external_consumer_examples_layout_is_committed() -> None:
    suite = ROOT / "examples/external-consumer-python"

    assert (suite / "README.md").exists()
    for name in EXAMPLES:
        assert (suite / name / "program.xlog").exists()
        assert (suite / name / "run.py").exists()


def test_external_consumer_examples_validator_contract_is_committed() -> None:
    validator = ROOT / "scripts/validate_external_consumer_examples.py"

    assert validator.exists()

    source = validator.read_text()
    for needle in [
        "example_count",
        "per_example",
        "cuda_tensor_checks",
        "host_transfer_diagnostics",
        "relation_delta_equivalence",
        "deterministic_topk_selected_labels",
        "exact_induction_parity",
        "async_completion",
        "streaming_chunk_counts",
    ]:
        assert needle in source
