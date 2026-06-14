import json
import os
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
    "external-delta-consumer",
    "neutral-external-consumer",
    "runtime-substrate-primitives",
    "pyxlog-compatibility",
}

REQUIRED_FEATURES = [
    "delta",
    "exact_induction",
    "chain_shared_memory",
    "common_subexpression_elimination",
    "adaptive_reoptimization",
    "persistent_hash_index",
    "runtime_substrate_primitives",
    "pyxlog_compatibility",
    "production_path_reuse",
]

PROJECT_TERM_PARTS = [("mista", "ber")]


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


def test_v086_neutral_external_examples_do_not_leak_project_terminology() -> None:
    neutral_count = 0

    for name in EXAMPLES:
        expected = _load_expected(name)
        if expected["consumer"] != "neutral-external-consumer":
            continue
        neutral_count += 1
        source = (SUITE / name / "program.xlog").read_text(encoding="utf-8").lower()
        for term_parts in PROJECT_TERM_PARTS:
            assert "".join(term_parts) not in source, name

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
        "behavior_probes",
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
    assert summary["consumer_certification_status"] == "PASS"
    assert summary["feature_coverage_source"] == "behavior_probes"
    assert summary["consumer_proof_gaps"] == []
    assert summary["example_count"] == len(EXAMPLES)

    for feature in REQUIRED_FEATURES:
        assert summary["feature_coverage"][feature]

    assert summary["behavior_probes"]
    assert all(probe["status"] == "PASS" for probe in summary["behavior_probes"].values())
    assert summary["compatibility_gates"]["v080_examples"]["status"] == "PASS"
    assert summary["compatibility_gates"]["v085_examples"]["status"] == "PASS"
    assert (
        summary["compatibility_gates"]["pyxlog_persistent_index_session_reuse"]["status"]
        == "PASS"
    )
    assert summary["production_path_reuse"]["status"] == "PASS"
    assert summary["reuse_audit"]["status"] == "PASS"


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


def test_v086_validator_stages_fresh_debug_kernels_over_package_local_stale(tmp_path) -> None:
    module = _load_validator_module()

    target_dir = tmp_path / "target" / "debug"
    stale_out = target_dir / "build" / "xlog-cuda-stale" / "out"
    fresh_out = target_dir / "build" / "xlog-cuda-fresh" / "out"
    stale_out.mkdir(parents=True)
    fresh_out.mkdir(parents=True)

    (stale_out / "weights.sm_120.cubin").write_text("stale cubin", encoding="utf-8")
    (stale_out / "weights.portable.ptx").write_text("stale ptx", encoding="utf-8")
    (fresh_out / "weights.sm_120.cubin").write_text(
        "fresh cubin with weights_count_lift_exact",
        encoding="utf-8",
    )
    (fresh_out / "weights.portable.ptx").write_text("fresh ptx", encoding="utf-8")

    deps_dir = target_dir / "deps"
    deps_dir.mkdir()
    (deps_dir / "xlog_cuda-current.d").write_text(
        f"libxlog_cuda.rlib:\n# env-dep:OUT_DIR={fresh_out}\n",
        encoding="utf-8",
    )

    os.utime(stale_out, (3, 3))
    os.utime(fresh_out, (2, 2))

    staged_pkg = target_dir / "pyxlog"
    package_kernels = staged_pkg / "kernels"
    package_kernels.mkdir(parents=True)
    (package_kernels / "weights.sm_120.cubin").write_text(
        "ignored package-local stale cubin",
        encoding="utf-8",
    )
    (package_kernels / "obsolete.sm_120.cubin").write_text("obsolete", encoding="utf-8")

    staged_kernels = module._stage_debug_pyxlog_kernels(target_dir, staged_pkg)

    assert staged_kernels == package_kernels
    assert (staged_kernels / "weights.sm_120.cubin").read_text(encoding="utf-8") == (
        "fresh cubin with weights_count_lift_exact"
    )
    assert (staged_kernels / "weights.portable.ptx").read_text(encoding="utf-8") == "fresh ptx"
    assert not (staged_kernels / "obsolete.sm_120.cubin").exists()


def test_v086_validator_separates_example_execution_from_consumer_certification() -> None:
    module = _load_validator_module()
    fake_results = [
        {
            "name": "example",
            "status": "PASS",
            "consumer": "external-delta-consumer",
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
            "v080_examples": {"status": "PASS"},
            "v085_examples": {"status": "PASS"},
            "v080_v085_source_guards": {"status": "PASS"},
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
