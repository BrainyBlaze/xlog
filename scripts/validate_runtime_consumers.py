#!/usr/bin/env python3
"""Validate runtime consumer certification examples."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.install_pyxlog_for_python import VERIFY_IMPORT_SNIPPET

EXAMPLE_ROOT = ROOT / "examples" / "runtime-consumers"
DEFAULT_OUTPUT = ROOT / "docs-internal" / "evidence" / "runtime-consumers" / "validation_summary.json"
DEFAULT_EVIDENCE = ROOT / "docs-internal" / "evidence" / "runtime-consumers"
EXAMPLES = [
    "01_external_delta_optimizer",
    "02_neutral_material_flow",
    "03_neutral_signal_diagnostics",
    "04_runtime_substrate_primitives",
    "05_pyxlog_session_compatibility",
]
REQUIRED_FEATURES = [
    "delta",
    "exact_induction",
    "chain_shared_memory",
    "common_subexpression_elimination",
    "adaptive_reoptimization",
    "persistent_hash_index",
    "runtime_substrate_primitives",
    "pyxlog_compatibility",
]
REQUIRED_CONSUMERS = {
    "external-delta-consumer",
    "neutral-external-consumer",
    "runtime-substrate-primitives",
    "pyxlog-compatibility",
}
MEASURED_FEATURES = (
    "delta",
    "exact_induction",
    "chain_shared_memory",
    "common_subexpression_elimination",
    "adaptive_reoptimization",
    "persistent_hash_index",
)


def _find_measurement_file(feature: str) -> Path:
    def matches_feature(measurements: dict[str, Any]) -> bool:
        if feature == "delta":
            return "recompute_call_reduction_ratio" in measurements
        if feature == "exact_induction":
            return "provider_typed_tests_passed" in measurements and "symbol" in measurements
        if feature == "chain_shared_memory":
            return measurements.get("chain_hot", {}).get("parity") is True
        if feature == "common_subexpression_elimination":
            return "unsafe_rejections" in measurements and "deterministic_fixture" in measurements
        if feature == "adaptive_reoptimization":
            return "rollback_fixture" in measurements and "deterministic_fixture" in measurements
        if feature == "persistent_hash_index":
            return "performance_fixture" in measurements and "repeated_session_fixture" in measurements
        return False

    evidence_root = ROOT / "docs-internal" / "evidence"
    if not evidence_root.is_dir():
        raise RuntimeError(
            f"Runtime feature evidence directory does not exist: {evidence_root}"
        )

    matches = []
    for path in sorted(evidence_root.iterdir()):
        measurements_path = path / "measurements.json"
        if not measurements_path.exists():
            continue
        measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
        if matches_feature(measurements):
            matches.append(measurements_path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one measurements file for {feature}, found {matches}")
    return matches[0]


def _probe(
    *,
    status: bool,
    features: list[str],
    consumers: list[str],
    proof: str,
    evidence: list[str],
    raw: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "PASS" if status else "BLOCKED",
        "features": features,
        "consumers": consumers,
        "proof": proof,
        "evidence": evidence,
        "raw": raw,
    }


def _feature_node_behavior_proofs(feature_measurements: dict[str, Any]) -> dict[str, Any]:
    proofs: dict[str, Any] = {}
    persistent = feature_measurements.get("persistent_hash_index", {}).get("raw", {})
    performance = persistent.get("performance_fixture")
    if performance:
        transfer_budget = performance.get("transfer_budget", {})
        proofs["persistent_hash_index"] = {
            "status": "PASS"
            if performance.get("speedup_ratio", 0.0) >= 1.5
            and transfer_budget.get("cached_tracked_dtoh_calls") == 0
            and transfer_budget.get("cached_tracked_htod_calls") == 0
            else "BLOCKED",
            "proof": "feature-node runtime performance fixture",
            "speedup_ratio": performance.get("speedup_ratio"),
            "target_speedup_ratio": performance.get("target_speedup_ratio", 1.5),
            "cached_median_seconds": performance.get("cached", {}).get("median_seconds"),
            "uncached_median_seconds": performance.get("uncached", {}).get("median_seconds"),
            "cached_tracked_dtoh_calls": transfer_budget.get("cached_tracked_dtoh_calls"),
            "cached_tracked_htod_calls": transfer_budget.get("cached_tracked_htod_calls"),
        }
    return proofs


def _all_exact_parity(raw: dict[str, Any], key: str) -> bool:
    fixtures = raw.get(key, {})
    return bool(fixtures) and all(item.get("parity") is True for item in fixtures.values())


def _consumer_behavior_probes(
    results: list[dict[str, Any]],
    feature_measurements: dict[str, Any],
    compatibility_gates: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    examples_by_consumer = {
        consumer: [result["name"] for result in results if result["consumer"] == consumer]
        for consumer in REQUIRED_CONSUMERS
    }
    feature_raw = {
        feature: payload.get("raw", {}) for feature, payload in feature_measurements.items()
    }

    delta = feature_raw.get("delta", {})
    exact = feature_raw.get("exact_induction", {})
    chain = feature_raw.get("chain_shared_memory", {})
    cse = feature_raw.get("common_subexpression_elimination", {})
    adaptive = feature_raw.get("adaptive_reoptimization", {})
    persistent = feature_raw.get("persistent_hash_index", {})

    persistent_perf = persistent.get("performance_fixture", {})
    persistent_transfer = persistent_perf.get("transfer_budget", {})
    persistent_repeated = persistent.get("repeated_session_fixture", {})
    chain_hot = chain.get("chain_hot", {})
    cse_deterministic = cse.get("deterministic_fixture", {})
    adaptive_deterministic = adaptive.get("deterministic_fixture", {})

    return {
        "delta": _probe(
            status=delta.get("recompute_call_reduction_ratio", 0.0) >= 1.0
            and delta.get("hot_path_dtoh_calls") == 0
            and delta.get("final_output_transfer_excluded") is True,
            features=["delta"],
            consumers=["external-delta-consumer", "pyxlog-compatibility"],
            proof="device-side relation delta coalescing fixture records reduced recomputes and zero hot-path DTOH",
            evidence=[feature_measurements["delta"]["path"]],
            raw={
                "recompute_call_reduction_ratio": delta.get("recompute_call_reduction_ratio"),
                "hot_path_dtoh_calls": delta.get("hot_path_dtoh_calls"),
                "examples": examples_by_consumer["external-delta-consumer"]
                + examples_by_consumer["pyxlog-compatibility"],
            },
        ),
        "exact_induction": _probe(
            status=exact.get("provider_typed_tests_passed", 0) >= 7
            and exact.get("core_dlpack_compatibility_tests_passed", 0) >= 1
            and _all_exact_parity(exact, "u32")
            and _all_exact_parity(exact, "symbol"),
            features=["exact_induction"],
            consumers=["external-delta-consumer", "runtime-substrate-primitives"],
            proof="native exact-induction typed provider tests pass for U32 and Symbol pair buffers with parity",
            evidence=[feature_measurements["exact_induction"]["path"]],
            raw={
                "provider_typed_tests_passed": exact.get("provider_typed_tests_passed"),
                "core_dlpack_compatibility_tests_passed": exact.get(
                    "core_dlpack_compatibility_tests_passed"
                ),
                "examples": examples_by_consumer["external-delta-consumer"]
                + examples_by_consumer["runtime-substrate-primitives"],
            },
        ),
        "chain_shared_memory": _probe(
            status=chain_hot.get("parity") is True
            and chain_hot.get("speedup_ratio", 0.0) > 1.0
            and chain.get("transfer_budget", {}).get("added_dtoh_calls") == 0,
            features=["chain_shared_memory"],
            consumers=["runtime-substrate-primitives"],
            proof="chain-topology shared-memory scorer records parity, speedup, and no added DTOH",
            evidence=[feature_measurements["chain_shared_memory"]["path"]],
            raw={
                "speedup_ratio": chain_hot.get("speedup_ratio"),
                "parity": chain_hot.get("parity"),
                "added_dtoh_calls": chain.get("transfer_budget", {}).get("added_dtoh_calls"),
                "examples": examples_by_consumer["runtime-substrate-primitives"],
            },
        ),
        "common_subexpression_elimination": _probe(
            status=cse_deterministic.get("output_parity") is True
            and cse_deterministic.get("duplicate_subplan_reduction_percent", 0.0) > 0.0
            and cse_deterministic.get("added_dtoh_calls") == 0
            and all(cse.get("unsafe_rejections", {}).values()),
            features=["common_subexpression_elimination"],
            consumers=[
                "external-delta-consumer",
                "neutral-external-consumer",
                "runtime-substrate-primitives",
            ],
            proof="runtime CSE duplicate-subplan fixture records parity, duplicate-work reduction, unsafe-boundary rejection, and zero added DTOH",
            evidence=[feature_measurements["common_subexpression_elimination"]["path"]],
            raw={
                "duplicate_subplan_reduction_percent": cse_deterministic.get(
                    "duplicate_subplan_reduction_percent"
                ),
                "output_parity": cse_deterministic.get("output_parity"),
                "added_dtoh_calls": cse_deterministic.get("added_dtoh_calls"),
                "examples": examples_by_consumer["external-delta-consumer"]
                + examples_by_consumer["neutral-external-consumer"]
                + examples_by_consumer["runtime-substrate-primitives"],
            },
        ),
        "adaptive_reoptimization": _probe(
            status=adaptive_deterministic.get("adopted", 0) >= 1
            and adaptive_deterministic.get("data_plane_dtoh_calls") == 0
            and adaptive_deterministic.get("decision_replays", 0) >= 100
            and adaptive.get("rollback_fixture", {}).get("rolled_back", 0) >= 1,
            features=["adaptive_reoptimization"],
            consumers=[
                "external-delta-consumer",
                "neutral-external-consumer",
                "runtime-substrate-primitives",
            ],
            proof="adaptive reoptimization fixture records adoption, deterministic replay, rollback, and zero data-plane DTOH",
            evidence=[feature_measurements["adaptive_reoptimization"]["path"]],
            raw={
                "adopted": adaptive_deterministic.get("adopted"),
                "decision_replays": adaptive_deterministic.get("decision_replays"),
                "data_plane_dtoh_calls": adaptive_deterministic.get("data_plane_dtoh_calls"),
                "rolled_back": adaptive.get("rollback_fixture", {}).get("rolled_back"),
                "examples": examples_by_consumer["external-delta-consumer"]
                + examples_by_consumer["neutral-external-consumer"]
                + examples_by_consumer["runtime-substrate-primitives"],
            },
        ),
        "persistent_hash_index": _probe(
            status=persistent_perf.get("speedup_ratio", 0.0) >= 1.5
            and persistent_transfer.get("cached_tracked_dtoh_calls") == 0
            and persistent_transfer.get("cached_tracked_htod_calls") == 0
            and persistent_repeated.get("builds", 0) >= 1
            and persistent_repeated.get("hits", 0) >= 1,
            features=["persistent_hash_index"],
            consumers=[
                "external-delta-consumer",
                "neutral-external-consumer",
                "runtime-substrate-primitives",
                "pyxlog-compatibility",
            ],
            proof="persistent hash-index runtime fixture records >=1.5x speedup, repeated-session build/hit, and zero tracked transfers",
            evidence=[
                feature_measurements["persistent_hash_index"]["path"],
                "python/tests/test_pyxlog_persistent_index_runtime.py",
            ],
            raw={
                "speedup_ratio": persistent_perf.get("speedup_ratio"),
                "builds": persistent_repeated.get("builds"),
                "hits": persistent_repeated.get("hits"),
                "tracked_dtoh_calls": persistent_repeated.get("tracked_dtoh_calls"),
                "pyxlog_session_reuse_status": compatibility_gates.get(
                    "pyxlog_persistent_index_session_reuse", {}
                ).get("status"),
                "examples": examples_by_consumer["external-delta-consumer"]
                + examples_by_consumer["neutral-external-consumer"]
                + examples_by_consumer["runtime-substrate-primitives"]
                + examples_by_consumer["pyxlog-compatibility"],
            },
        ),
        "runtime_substrate_primitives": _probe(
            status=bool(examples_by_consumer["runtime-substrate-primitives"]),
            features=["runtime_substrate_primitives"],
            consumers=["runtime-substrate-primitives"],
            proof="runtime substrate primitive example executes through xlog-cli and maps to exact/index/CSE/adaptive primitive behavior probes",
            evidence=["runtime substrate primitive example program"],
            raw={"examples": examples_by_consumer["runtime-substrate-primitives"]},
        ),
        "pyxlog_compatibility": _probe(
            status=compatibility_gates.get("external_consumer_examples", {}).get("status") == "PASS"
            and compatibility_gates.get("language_examples", {}).get("status") == "PASS"
            and compatibility_gates.get("pyxlog_persistent_index_session_reuse", {}).get(
                "status"
            )
            == "PASS",
            features=["pyxlog_compatibility"],
            consumers=["pyxlog-compatibility"],
            proof="public pyxlog compatibility validators and session persistent-index probe pass against the installed package",
            evidence=[
                "scripts/validate_external_consumer_examples.py",
                "scripts/validate_language_examples.py",
                "python/tests/test_pyxlog_persistent_index_runtime.py",
            ],
            raw={"examples": examples_by_consumer["pyxlog-compatibility"]},
        ),
    }


def _behavior_feature_coverage(
    behavior_probes: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    return {
        feature: [
            probe_id
            for probe_id, probe in behavior_probes.items()
            if probe["status"] == "PASS" and feature in probe["features"]
        ]
        for feature in REQUIRED_FEATURES
    }


def _behavior_probe_gaps(
    behavior_probes: dict[str, dict[str, Any]],
    feature_coverage: dict[str, list[str]],
) -> list[dict[str, Any]]:
    gaps = [
        {
            "id": f"behavior-probe-{probe_id}",
            "status": "BLOCKED",
            "reason": probe["proof"],
        }
        for probe_id, probe in behavior_probes.items()
        if probe["status"] != "PASS"
    ]
    gaps.extend(
        {
            "id": f"missing-behavior-feature-{feature}",
            "status": "BLOCKED",
            "reason": f"{feature} has no passing behavior probe",
        }
        for feature, probes in feature_coverage.items()
        if not probes
    )
    return gaps


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _resolve_output_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (ROOT / expanded).resolve()


def _display_path(path: Path) -> str:
    resolved = _resolve_output_path(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _base_xlog_command(args: argparse.Namespace) -> list[str]:
    if args.xlog_bin:
        return [str(args.xlog_bin)]
    return ["cargo", "run", "-q", "-p", "xlog-cli", "--"]


def _run_command(
    cmd: list[str],
    *,
    timeout: int,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=env,
    )
    duration = time.perf_counter() - start
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "duration_sec": round(duration, 6),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _record_raw(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "cmd": raw["cmd"],
        "returncode": raw["returncode"],
        "duration_sec": raw["duration_sec"],
        "stdout_preview": raw["stdout"][-2000:],
        "stderr_preview": raw["stderr"][-2000:],
    }


def _load_expected(example_dir: Path) -> dict[str, Any]:
    expected_path = example_dir / "expected.json"
    _require(expected_path.exists(), f"Missing expected.json for {example_dir.name}")
    return json.loads(expected_path.read_text(encoding="utf-8"))


def _check_required_substrings(haystack: str, needles: list[str], context: str) -> None:
    for needle in needles:
        _require(needle in haystack, f"{context} missing expected substring: {needle}")


def _check_run(example: str, raw: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    expected_returncode = expected.get("returncode", 0)
    _require(
        raw["returncode"] == expected_returncode,
        f"{example} run returncode mismatch: expected {expected_returncode}, got {raw['returncode']}\n"
        f"STDOUT:\n{raw['stdout']}\nSTDERR:\n{raw['stderr']}",
    )
    _check_required_substrings(
        raw["stdout"],
        expected.get("stdout_contains", []),
        f"{example} run stdout",
    )
    return {"status": "PASS", "returncode": raw["returncode"], "duration_sec": raw["duration_sec"]}


def _check_explain_json(
    example: str,
    raw: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    _require(
        raw["returncode"] == 0,
        f"{example} explain failed with exit {raw['returncode']}\nSTDOUT:\n{raw['stdout']}\nSTDERR:\n{raw['stderr']}",
    )
    try:
        report = json.loads(raw["stdout"])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{example} explain did not emit JSON: {exc}\n{raw['stdout']}") from exc
    _require(
        report["parse"]["statements"] >= expected.get("min_statements", 1),
        f"{example} parse statements below expected floor: {report['parse']['statements']}",
    )
    if "stratification_status" in expected:
        _require(
            report["stratification"]["status"] == expected["stratification_status"],
            f"{example} stratification status mismatch: {report['stratification']}",
        )
    if "rir_status" in expected:
        _require(
            report["rir"]["status"] == expected["rir_status"],
            f"{example} RIR status mismatch: {report['rir']}",
        )
    return {
        "status": "PASS",
        "statements": report["parse"]["statements"],
        "duration_sec": raw["duration_sec"],
    }


def _load_example_result(example: str, args: argparse.Namespace) -> dict[str, Any]:
    example_dir = EXAMPLE_ROOT / example
    program = example_dir / "program.xlog"
    _require(program.exists(), f"{example} missing program.xlog")
    expected = _load_expected(example_dir)

    program_arg = str(program.relative_to(ROOT))
    raw_outputs: dict[str, Any] = {}
    check_results: dict[str, Any] = {}
    checks = expected.get("checks", {})
    base = _base_xlog_command(args)

    if "run" in checks:
        raw = _run_command(base + ["run", program_arg], timeout=args.timeout)
        raw_outputs["run"] = _record_raw(raw)
        check_results["run"] = _check_run(example, raw, checks["run"])

    if "explain_json" in checks:
        raw = _run_command(base + ["explain", "--format", "json", program_arg], timeout=args.timeout)
        raw_outputs["explain_json"] = _record_raw(raw)
        check_results["explain_json"] = _check_explain_json(example, raw, checks["explain_json"])

    features = expected.get("features", [])
    return {
        "name": example,
        "status": "PASS",
        "consumer": expected["consumer"],
        "program": program_arg,
        "features": features,
        "checks": sorted(checks),
        "check_results": check_results,
        "raw_outputs": raw_outputs,
        "raw_measurements": {
            "run_duration_sec": check_results.get("run", {}).get("duration_sec"),
            "explain_duration_sec": check_results.get("explain_json", {}).get("duration_sec"),
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_measurements() -> dict[str, Any]:
    measurements: dict[str, Any] = {}
    for feature in MEASURED_FEATURES:
        path = _find_measurement_file(feature)
        payload = _read_json(path)
        measurements[feature] = {
            "path": str(path.relative_to(ROOT)),
            "raw": payload,
        }
    return measurements


def _run_existing_validator(
    args: argparse.Namespace,
    script: str,
    output: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    output_path = _resolve_output_path(output)
    raw = _run_command(
        [args.python, script, "--output", str(output_path), "--timeout", str(args.compat_timeout)],
        timeout=args.compat_timeout * 20,
        env=env,
    )
    if raw["returncode"] != 0:
        raise SystemExit(
            f"{script} failed with exit {raw['returncode']}\nSTDOUT:\n{raw['stdout']}\nSTDERR:\n{raw['stderr']}"
        )
    payload = _read_json(output_path)
    return {
        "status": payload.get("status", "UNKNOWN"),
        "script": script,
        "output": _display_path(output_path),
        "duration_sec": raw["duration_sec"],
        "returncode": raw["returncode"],
        "example_count": payload.get("example_count"),
    }


def _run_pyxlog_persistent_index_probe(
    args: argparse.Namespace,
    env: dict[str, str],
) -> dict[str, Any]:
    raw = _run_command(
        [
            args.python,
            "-m",
            "pytest",
            "-q",
            "python/tests/test_pyxlog_persistent_index_runtime.py",
        ],
        timeout=args.compat_timeout,
        env=env,
    )
    if raw["returncode"] != 0:
        raise SystemExit(
            "pyxlog persistent-index session probe failed with exit "
            f"{raw['returncode']}\nSTDOUT:\n{raw['stdout']}\nSTDERR:\n{raw['stderr']}"
        )
    return {
        "status": "PASS",
        "cmd": raw["cmd"],
        "duration_sec": raw["duration_sec"],
        "returncode": raw["returncode"],
        "stdout_preview": raw["stdout"][-1000:],
    }


def _compatibility_gates(args: argparse.Namespace, evidence_dir: Path) -> dict[str, Any]:
    pyxlog_env = os.environ.copy()
    pyxlog_env["XLOG_REQUIRE_CUDA"] = "1"
    verify = _run_command(
        [args.python, "-c", VERIFY_IMPORT_SNIPPET],
        timeout=args.timeout,
        env=pyxlog_env,
    )
    _require(
        verify["returncode"] == 0,
        "An installed pyxlog package with kernels is required. "
        "Use scripts/install_pyxlog_for_python.py for an authorized manual LOCAL build/install.\n"
        f"STDOUT:\n{verify['stdout']}\nSTDERR:\n{verify['stderr']}",
    )
    external_consumer = _run_existing_validator(
        args,
        "scripts/validate_external_consumer_examples.py",
        evidence_dir / "compat_external_consumer_examples_validation_summary.json",
        pyxlog_env,
    )
    language = _run_existing_validator(
        args,
        "scripts/validate_language_examples.py",
        evidence_dir / "compat_language_examples_validation_summary.json",
        pyxlog_env,
    )
    pyxlog_persistent = _run_pyxlog_persistent_index_probe(args, pyxlog_env)
    _require(
        external_consumer["status"] == "PASS",
        f"external consumer validator did not pass: {external_consumer}",
    )
    _require(language["status"] == "PASS", f"language validator did not pass: {language}")
    return {
        "external_consumer_examples": external_consumer,
        "language_examples": language,
        "pyxlog_persistent_index_session_reuse": pyxlog_persistent,
    }


def _aggregate(
    results: list[dict[str, Any]],
    feature_measurements: dict[str, Any],
    compatibility_gates: dict[str, Any],
) -> dict[str, Any]:
    declared_feature_coverage = {
        feature: [result["name"] for result in results if feature in result["features"]]
        for feature in REQUIRED_FEATURES
    }
    missing_declared_features = [
        feature for feature, examples in declared_feature_coverage.items() if not examples
    ]
    _require(
        not missing_declared_features,
        f"Missing required declared feature coverage: {missing_declared_features}",
    )
    observed_consumers = {result["consumer"] for result in results}
    _require(
        REQUIRED_CONSUMERS <= observed_consumers,
        f"Missing required consumers: {sorted(REQUIRED_CONSUMERS - observed_consumers)}",
    )

    behavior_probes = _consumer_behavior_probes(
        results,
        feature_measurements,
        compatibility_gates,
    )
    feature_coverage = _behavior_feature_coverage(behavior_probes)
    consumer_proof_gaps = _behavior_probe_gaps(behavior_probes, feature_coverage)
    certification_status = "PASS" if not consumer_proof_gaps else "BLOCKED"

    return {
        "suite": "runtime_consumers",
        "status": "PASS" if certification_status == "PASS" else "BLOCKED",
        "example_execution_status": "PASS",
        "consumer_certification_status": certification_status,
        "feature_coverage_source": "behavior_probes",
        "feature_node_behavior_proofs": _feature_node_behavior_proofs(feature_measurements),
        "consumer_proof_gaps": consumer_proof_gaps,
        "example_count": len(results),
        "required_example_count": len(EXAMPLES),
        "consumer_coverage": {
            consumer: [result["name"] for result in results if result["consumer"] == consumer]
            for consumer in sorted(REQUIRED_CONSUMERS)
        },
        "feature_coverage": feature_coverage,
        "declared_feature_coverage": declared_feature_coverage,
        "behavior_probes": behavior_probes,
        "feature_proof_model": {
            "example_execution": "xlog-cli run/explain over committed .xlog programs",
            "declared_feature_coverage": "expected.json declarations retained for traceability only",
            "feature_coverage": "validator-owned behavior probes over committed feature evidence and public pyxlog compatibility gates",
            "certification_limit": None,
        },
        "per_example": [
            {
                "name": result["name"],
                "status": result["status"],
                "consumer": result["consumer"],
                "features": result["features"],
                "checks": result["checks"],
                "raw_measurements": result["raw_measurements"],
            }
            for result in results
        ],
        "raw_outputs": {result["name"]: result["raw_outputs"] for result in results},
        "raw_measurements": {
            "feature_evidence": feature_measurements,
            "example_timings": {
                result["name"]: result["raw_measurements"] for result in results
            },
        },
        "compatibility_gates": compatibility_gates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Summary JSON output path.")
    parser.add_argument("--timeout", type=int, default=120, help="Per xlog command timeout in seconds.")
    parser.add_argument("--compat-timeout", type=int, default=180, help="Compatibility command timeout in seconds.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used to run validators.")
    parser.add_argument("--xlog-bin", type=Path, help="Use an existing xlog binary instead of cargo run.")
    args = parser.parse_args(argv)

    _require(EXAMPLE_ROOT.exists(), f"Missing example root: {EXAMPLE_ROOT}")
    args.output = _resolve_output_path(args.output)
    evidence_dir = args.output.parent
    evidence_dir.mkdir(parents=True, exist_ok=True)

    results = [_load_example_result(example, args) for example in EXAMPLES]
    feature_measurements = _feature_measurements()
    compatibility_gates = _compatibility_gates(args, evidence_dir)
    summary = _aggregate(results, feature_measurements, compatibility_gates)

    _require(summary["example_count"] == len(EXAMPLES), f"Unexpected example count: {summary['example_count']}")
    _require(
        all(item["status"] == "PASS" for item in summary["per_example"]),
        f"Expected all per-example statuses PASS: {summary['per_example']}",
    )

    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
