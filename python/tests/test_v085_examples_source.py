import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

EXAMPLES = [
    "01_list_typed_relation",
    "02_findall_aggregate",
    "03_maplist_static_predref",
    "04_magic_reach_explain",
    "05_prob_aggregate_exact",
    "06_prob_aggregate_mc",
    "07_aggregate_lifting",
    "08_approx_confidence",
    "09_repl_watch_explain",
    "10_scientific_incremental",
]

REQUIRED_FEATURES = [
    "types",
    "lists",
    "findall",
    "aggregate_query",
    "maplist",
    "naf",
    "magic_sets",
    "prob_aggregate_exact",
    "prob_aggregate_mc",
    "aggregate_lifting",
    "approx_inference",
    "incremental_parse",
    "cli_repl",
    "cli_watch",
    "cli_explain",
]


def test_v085_examples_layout_is_committed() -> None:
    suite = ROOT / "examples/v085-language/showcase"

    assert (suite / "README.md").exists()
    for name in EXAMPLES:
        example = suite / name
        assert (example / "program.xlog").exists(), name
        assert (example / "expected.json").exists(), name
        assert (example / "README.md").exists(), name


def test_v085_examples_expected_contracts_include_semantic_execution() -> None:
    suite = ROOT / "examples/v085-language/showcase"

    for name in EXAMPLES:
        expected = json.loads((suite / name / "expected.json").read_text())
        checks = expected["checks"]
        assert any(key in checks for key in ["run", "prob_json"]), name


def test_v085_showcase_run_checks_do_not_accept_raw_kernel_schema_errors() -> None:
    suite = ROOT / "examples/v085-language/showcase"
    raw_schema_error = " ".join(["Union requires compatible", "schemas"])

    for name in EXAMPLES:
        expected = json.loads((suite / name / "expected.json").read_text())
        run_check = expected["checks"].get("run")
        if not run_check:
            continue

        combined_needles = run_check.get("combined_contains", [])
        assert raw_schema_error not in combined_needles, name


def test_v085_examples_validator_and_evidence_contract_is_committed() -> None:
    validator = ROOT / "scripts/validate_v085_examples.py"
    evidence = ROOT / "docs/evidence/2026-05-19-v085-examples"

    assert validator.exists()
    assert (evidence / "README.md").exists()
    assert (evidence / "validation_summary.json").exists()

    source = validator.read_text(encoding="utf-8")
    for needle in [
        "G085_EXAMPLES",
        "feature_coverage",
        "interaction_count",
        "raw_outputs",
        "explain_json",
        "prob_json",
        "repl",
        "run",
        "watch",
    ]:
        assert needle in source
    for feature in REQUIRED_FEATURES:
        assert feature in source
