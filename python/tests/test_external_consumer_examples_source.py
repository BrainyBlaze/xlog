import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

EXAMPLES = [
    "01_async_streaming_reachability",
    "02_relation_deltas",
    "03_neural_bridge_topk_belnap",
    "04_native_exact_induction",
    "05_probabilistic_async_diagnostics",
]


def external_consumer_examples_evidence_dir() -> Path:
    matches = []
    for path in sorted((ROOT / "docs-internal" / "evidence").glob("*-examples")):
        summary_path = path / "validation_summary.json"
        if not summary_path.exists() or not (path / "README.md").exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("example_count") == len(EXAMPLES)
            and "exact_induction_parity" in summary
            and "relation_delta_equivalence" in summary
        ):
            matches.append(path)
    assert len(matches) == 1
    return matches[0]


def test_external_consumer_examples_layout_is_committed() -> None:
    suite = ROOT / "examples/external-consumer-python"

    assert (suite / "README.md").exists()
    for name in EXAMPLES:
        assert (suite / name / "program.xlog").exists()
        assert (suite / name / "run.py").exists()


def test_external_consumer_examples_validator_and_evidence_contract_is_committed() -> None:
    validator = ROOT / "scripts/validate_external_consumer_examples.py"
    evidence = external_consumer_examples_evidence_dir()

    assert validator.exists()
    assert (evidence / "README.md").exists()
    assert (evidence / "validation_summary.json").exists()

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
