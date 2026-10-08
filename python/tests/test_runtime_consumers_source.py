import sys
from importlib import util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def _load_validator_module():
    script = ROOT / "scripts/validate_runtime_consumers.py"
    spec = util.spec_from_file_location("validate_runtime_consumers_under_test", script)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_validator_accepts_absolute_and_relative_output_paths(monkeypatch, tmp_path) -> None:
    module = _load_validator_module()
    args = SimpleNamespace(python=sys.executable, compat_timeout=1)

    def fake_run_command(*_args, **_kwargs):
        return {"returncode": 0, "duration_sec": 0.01, "stdout": "", "stderr": ""}

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    absolute_output = tmp_path / "external-consumer-examples.json"
    absolute_output.write_text('{"status":"PASS","example_count":5}', encoding="utf-8")
    absolute = module._run_existing_validator(
        args,
        "scripts/validate_external_consumer_examples.py",
        absolute_output,
        {},
    )
    assert absolute["output"] == str(absolute_output)

    relative_output = Path("target/runtime-consumers-relative-output-test.json")
    (ROOT / relative_output).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / relative_output).write_text('{"status":"PASS","example_count":5}', encoding="utf-8")
    try:
        relative = module._run_existing_validator(
            args,
            "scripts/validate_external_consumer_examples.py",
            relative_output,
            {},
        )
        assert relative["output"] == "target/runtime-consumers-relative-output-test.json"
    finally:
        (ROOT / relative_output).unlink(missing_ok=True)


def test_runtime_validator_separates_example_execution_from_consumer_certification() -> None:
    module = _load_validator_module()
    fake_results = [
        {
            "name": "example",
            "status": "PASS",
            "consumer": "external-delta-consumer",
            "features": ["exact_induction"],
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
                ("neutral-external-consumer", "delta"),
                ("runtime-substrate-primitives", "chain_shared_memory"),
                ("pyxlog-compatibility", "pyxlog_compatibility"),
            ]
        ],
        {
            "name": "optimizer-example",
            "status": "PASS",
            "consumer": "neutral-external-consumer",
            "features": [
                "common_subexpression_elimination",
                "adaptive_reoptimization",
                "persistent_hash_index",
                "runtime_substrate_primitives",
            ],
            "checks": ["run"],
            "raw_measurements": {"run_duration_sec": 0.01, "explain_duration_sec": None},
            "raw_outputs": {},
        },
    ]
    feature_measurements = {
        "delta": {
            "path": "delta.json",
            "raw": {
                "recompute_call_reduction_ratio": 3.0,
                "hot_path_dtoh_calls": 0,
                "final_output_transfer_excluded": True,
            },
        },
        "exact_induction": {
            "path": "exact_induction.json",
            "raw": {
                "provider_typed_tests_passed": 7,
                "core_dlpack_compatibility_tests_passed": 1,
                "u32": {"3": {"parity": True}},
                "symbol": {"3": {"parity": True}},
            },
        },
        "chain_shared_memory": {
            "path": "chain_shared_memory.json",
            "raw": {
                "chain_hot": {"parity": True, "speedup_ratio": 5.5},
                "transfer_budget": {"added_dtoh_calls": 0},
            },
        },
        "common_subexpression_elimination": {
            "path": "common_subexpression_elimination.json",
            "raw": {
                "deterministic_fixture": {
                    "output_parity": True,
                    "duplicate_subplan_reduction_percent": 50.0,
                    "added_dtoh_calls": 0,
                },
                "unsafe_rejections": {
                    "aggregate_boundary": True,
                    "negation_or_difference_boundary": True,
                    "provenance_or_tensor_boundary": True,
                    "specialized_dispatch_boundary": True,
                },
            },
        },
        "adaptive_reoptimization": {
            "path": "adaptive_reoptimization.json",
            "raw": {
                "deterministic_fixture": {
                    "adopted": 1,
                    "data_plane_dtoh_calls": 0,
                    "decision_replays": 100,
                },
                "rollback_fixture": {"rolled_back": 1},
            },
        },
        "persistent_hash_index": {
            "path": "persistent_hash_index.json",
            "raw": {
                "performance_fixture": {
                    "speedup_ratio": 3.206,
                    "transfer_budget": {
                        "cached_tracked_dtoh_calls": 0,
                        "cached_tracked_htod_calls": 0,
                    },
                },
                "repeated_session_fixture": {"builds": 1, "hits": 1, "tracked_dtoh_calls": 0},
            },
        },
    }
    summary = module._aggregate(
        fake_results,
        feature_measurements,
        {
            "external_consumer_examples": {"status": "PASS"},
            "language_examples": {"status": "PASS"},
            "pyxlog_persistent_index_session_reuse": {"status": "PASS"},
        },
    )

    assert summary["status"] == "PASS"
    assert summary["example_execution_status"] == "PASS"
    assert summary["consumer_certification_status"] == "PASS"
    assert summary["feature_coverage_source"] == "behavior_probes"
    assert summary["feature_node_behavior_proofs"]["persistent_hash_index"]["status"] == "PASS"
    assert summary["feature_node_behavior_proofs"]["persistent_hash_index"]["speedup_ratio"] == 3.206
    assert summary["consumer_proof_gaps"] == []
    assert summary["behavior_probes"]["persistent_hash_index"]["status"] == "PASS"
