import json
import sys
from importlib import util
from pathlib import Path
from types import SimpleNamespace


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


def _load_validator_module():
    script = ROOT / "scripts/validate_v086_examples.py"
    spec = util.spec_from_file_location("validate_v086_examples_under_test", script)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        "child.name.startswith(\"_native\")",
        "pyxlog_persistent_index_session_reuse",
    ]:
        assert needle in source


def test_v086_consumer_evidence_records_feature_coverage_and_reuse_audit() -> None:
    summary_path = EVIDENCE / "validation_summary.json"
    assert (EVIDENCE / "README.md").exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["suite"] == "G086_CONSUMERS"
    assert summary["status"] == "PASS"
    assert summary["example_execution_status"] == "PASS"
    assert summary["consumer_certification_status"] == "BLOCKED"
    assert summary["feature_coverage_source"] == "expected_json_declarations"
    assert summary["consumer_proof_gaps"]
    assert summary["example_count"] == len(EXAMPLES)

    for feature in REQUIRED_FEATURES:
        assert summary["feature_coverage"][feature]

    assert summary["compatibility_gates"]["v080_examples"]["status"] == "PASS"
    assert summary["compatibility_gates"]["v085_examples"]["status"] == "PASS"
    assert (
        summary["compatibility_gates"]["pyxlog_persistent_index_session_reuse"]["status"]
        == "PASS"
    )
    assert summary["production_path_reuse"]["status"] == "PASS"
    assert summary["reuse_audit"]["status"] == "PASS"
    assert all(
        gap["id"] != "pyxlog-persistent-index-session-reuse"
        for gap in summary["consumer_proof_gaps"]
    )


def test_v086_validator_accepts_absolute_and_relative_output_paths(monkeypatch, tmp_path) -> None:
    module = _load_validator_module()
    args = SimpleNamespace(python=sys.executable, compat_timeout=1)

    def fake_run_command(*_args, **_kwargs):
        return {"returncode": 0, "duration_sec": 0.01, "stdout": "", "stderr": ""}

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    absolute_output = tmp_path / "v080.json"
    absolute_output.write_text('{"status":"PASS","example_count":5}', encoding="utf-8")
    absolute = module._run_existing_validator(
        args,
        "scripts/validate_v080_examples.py",
        absolute_output,
        {},
    )
    assert absolute["output"] == str(absolute_output)

    relative_output = Path("target/v086-relative-output-test.json")
    (ROOT / relative_output).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / relative_output).write_text('{"status":"PASS","example_count":5}', encoding="utf-8")
    try:
        relative = module._run_existing_validator(
            args,
            "scripts/validate_v080_examples.py",
            relative_output,
            {},
        )
        assert relative["output"] == "target/v086-relative-output-test.json"
    finally:
        (ROOT / relative_output).unlink(missing_ok=True)


def test_v086_validator_separates_example_execution_from_consumer_certification() -> None:
    module = _load_validator_module()
    fake_results = [
        {
            "name": "example",
            "status": "PASS",
            "consumer": "dts-dlm",
            "features": ["exact_induction", "production_path_reuse"],
            "checks": ["run"],
            "raw_measurements": {"run_duration_sec": 0.01, "explain_duration_sec": None},
            "raw_outputs": {},
        },
        *[
            {
                "name": f"{consumer}-example",
                "status": "PASS",
                "consumer": consumer,
                "features": [feature],
                "checks": ["run"],
                "raw_measurements": {"run_duration_sec": 0.01, "explain_duration_sec": None},
                "raw_outputs": {},
            }
            for consumer, feature in [
                ("mistaber-neutral", "delta"),
                ("v090-substrate", "chain_shared_memory"),
                ("pyxlog-compatibility", "pyxlog_compatibility"),
            ]
        ],
        {
            "name": "optimizer-example",
            "status": "PASS",
            "consumer": "mistaber-neutral",
            "features": [
                "common_subexpression_elimination",
                "adaptive_reoptimization",
                "persistent_hash_index",
                "v090_substrate",
            ],
            "checks": ["run"],
            "raw_measurements": {"run_duration_sec": 0.01, "explain_duration_sec": None},
            "raw_outputs": {},
        },
    ]
    feature_measurements = {feature: {"path": f"{feature}.json", "raw": {}} for feature in REQUIRED_FEATURES[:6]}
    feature_measurements["persistent_hash_index"] = {
        "path": "persistent_hash_index.json",
        "raw": {
            "performance_fixture": {
                "speedup_ratio": 3.206,
                "transfer_budget": {
                    "cached_tracked_dtoh_calls": 0,
                    "cached_tracked_htod_calls": 0,
                },
            }
        },
    }
    summary = module._aggregate(
        fake_results,
        feature_measurements,
        {
            "v080_examples": {"status": "PASS"},
            "v085_examples": {"status": "PASS"},
            "v080_v085_source_guards": {"status": "PASS"},
        },
    )

    assert summary["status"] == "PASS"
    assert summary["example_execution_status"] == "PASS"
    assert summary["consumer_certification_status"] == "BLOCKED"
    assert summary["feature_coverage_source"] == "expected_json_declarations"
    assert summary["feature_node_behavior_proofs"]["persistent_hash_index"]["status"] == "PASS"
    assert summary["feature_node_behavior_proofs"]["persistent_hash_index"]["speedup_ratio"] == 3.206
    assert any("label-derived" in gap["reason"] for gap in summary["consumer_proof_gaps"])
