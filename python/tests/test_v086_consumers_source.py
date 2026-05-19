import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "examples/v086-runtime"
EVIDENCE = ROOT / "docs/evidence/2026-05-19-v086-consumers"

EXAMPLES = [
    "01_dts_delta_optimizer",
    "02_neutral_material_flow",
    "03_neutral_signal_diagnostics",
    "04_v090_substrate_primitives",
    "05_pyxlog_session_compatibility",
]

REQUIRED_CONSUMERS = {
    "dts-dlm",
    "mistaber-neutral",
    "v090-substrate",
    "pyxlog-compatibility",
}

REQUIRED_FEATURES = [
    "delta",
    "exact_induction",
    "chain_shared_memory",
    "common_subexpression_elimination",
    "adaptive_reoptimization",
    "persistent_hash_index",
    "v090_substrate",
    "pyxlog_compatibility",
    "production_path_reuse",
]

PROJECT_TERMS = ["mistaber"]


def _load_expected(name: str) -> dict:
    return json.loads((SUITE / name / "expected.json").read_text(encoding="utf-8"))


def test_v086_consumer_examples_layout_is_committed() -> None:
    assert (SUITE / "README.md").exists()

    for name in EXAMPLES:
        example = SUITE / name
        assert (example / "program.xlog").exists(), name
        assert (example / "expected.json").exists(), name
        assert (example / "README.md").exists(), name


def test_v086_consumer_examples_cover_named_consumers_and_features() -> None:
    observed_consumers = set()
    observed_features = set()

    for name in EXAMPLES:
        expected = _load_expected(name)
        observed_consumers.add(expected["consumer"])
        observed_features.update(expected.get("features", []))
        assert any(key in expected["checks"] for key in ["run", "explain_json"]), name

    assert REQUIRED_CONSUMERS <= observed_consumers
    for feature in REQUIRED_FEATURES:
        assert feature in observed_features


def test_v086_neutral_mistaber_examples_do_not_leak_project_terminology() -> None:
    neutral_count = 0

    for name in EXAMPLES:
        expected = _load_expected(name)
        if expected["consumer"] != "mistaber-neutral":
            continue
        neutral_count += 1
        source = (SUITE / name / "program.xlog").read_text(encoding="utf-8").lower()
        for term in PROJECT_TERMS:
            assert term not in source, name

    assert neutral_count >= 2


def test_v086_consumer_validator_reuses_existing_compatibility_gates() -> None:
    validator = ROOT / "scripts/validate_v086_examples.py"
    assert validator.exists()

    source = validator.read_text(encoding="utf-8")
    for needle in [
        "G086_CONSUMERS",
        "validate_v080_examples.py",
        "validate_v085_examples.py",
        "feature_coverage",
        "raw_measurements",
        "compatibility_gates",
        "production_path_reuse",
        "reuse_audit",
    ]:
        assert needle in source


def test_v086_consumer_evidence_records_feature_coverage_and_reuse_audit() -> None:
    summary_path = EVIDENCE / "validation_summary.json"
    assert (EVIDENCE / "README.md").exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["suite"] == "G086_CONSUMERS"
    assert summary["status"] == "PASS"
    assert summary["example_count"] == len(EXAMPLES)

    for feature in REQUIRED_FEATURES:
        assert summary["feature_coverage"][feature]

    assert summary["compatibility_gates"]["v080_examples"]["status"] == "PASS"
    assert summary["compatibility_gates"]["v085_examples"]["status"] == "PASS"
    assert summary["production_path_reuse"]["status"] == "PASS"
    assert summary["reuse_audit"]["status"] == "PASS"
