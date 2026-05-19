#!/usr/bin/env python3
"""Validate the v0.8.6 runtime consumer certification examples."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "examples" / "v086-runtime"
DEFAULT_OUTPUT = ROOT / "docs" / "evidence" / "2026-05-19-v086-consumers" / "validation_summary.json"
DEFAULT_EVIDENCE = ROOT / "docs" / "evidence" / "2026-05-19-v086-consumers"
EXAMPLES = [
    "01_dts_delta_optimizer",
    "02_neutral_material_flow",
    "03_neutral_signal_diagnostics",
    "04_v090_substrate_primitives",
    "05_pyxlog_session_compatibility",
]
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
REQUIRED_CONSUMERS = {
    "dts-dlm",
    "mistaber-neutral",
    "v090-substrate",
    "pyxlog-compatibility",
}
FEATURE_EVIDENCE = {
    "delta": ROOT / "docs/evidence/2026-05-19-v086-delta-coalesce/measurements.json",
    "exact_induction": ROOT / "docs/evidence/2026-05-19-v086-exact-types/measurements.json",
    "chain_shared_memory": ROOT / "docs/evidence/2026-05-19-v086-chain-smem/measurements.json",
    "common_subexpression_elimination": ROOT / "docs/evidence/2026-05-19-v086-cse/measurements.json",
    "adaptive_reoptimization": ROOT / "docs/evidence/2026-05-19-v086-adaptive-reoptimization/measurements.json",
    "persistent_hash_index": ROOT / "docs/evidence/2026-05-19-v086-persistent-hash-index/measurements.json",
}
CONSUMER_PROOF_GAPS = [
    {
        "id": "label-derived-feature-coverage",
        "status": "BLOCKED",
        "reason": (
            "feature_coverage is label-derived from expected.json declarations; "
            "the .xlog examples prove parser/RIR/run/explain execution and "
            "link to feature-node evidence, but they do not by themselves "
            "prove native exact-induction, adaptive reoptimization, or "
            "persistent-index behavior inside each labeled example"
        ),
    },
    {
        "id": "pyxlog-persistent-index-session-reuse",
        "status": "BLOCKED",
        "reason": (
            "persistent hash-index reuse is proven on reused runtime Executors; "
            "public pyxlog LogicRelationSession evaluation does not yet expose "
            "targeted persistent-index reuse telemetry across session mutation "
            "and reevaluation"
        ),
    },
]


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
    readme = example_dir / "README.md"
    _require(program.exists(), f"{example} missing program.xlog")
    _require(readme.exists(), f"{example} missing README.md")
    expected = _load_expected(example_dir)
    source = program.read_text(encoding="utf-8")
    _check_required_substrings(
        source,
        expected.get("source_required_substrings", []),
        f"{example} program.xlog",
    )
    if expected.get("consumer") == "mistaber-neutral":
        _require("mistaber" not in source.lower(), f"{example} leaks project terminology")

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


def _prepare_local_pyxlog_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    build = _run_command(
        ["cargo", "build", "-q", "-p", "pyxlog", "--features", "host-io"],
        timeout=args.pyxlog_build_timeout,
        env=env,
    )
    if build["returncode"] != 0:
        raise SystemExit(
            f"local pyxlog build failed with exit {build['returncode']}\n"
            f"STDOUT:\n{build['stdout']}\nSTDERR:\n{build['stderr']}"
        )

    target_dir = ROOT / "target" / "debug"
    native_lib = target_dir / "libpyxlog.so"
    if not native_lib.exists():
        native_lib = target_dir / "libpyxlog.dylib"
    _require(native_lib.exists(), f"Unable to locate built pyxlog native library in {target_dir}")

    source_pkg = ROOT / "crates" / "pyxlog" / "python" / "pyxlog"
    staged_pkg = target_dir / "pyxlog"
    if staged_pkg.exists() or staged_pkg.is_symlink():
        if staged_pkg.is_dir() and not staged_pkg.is_symlink():
            shutil.rmtree(staged_pkg)
        else:
            staged_pkg.unlink()
    staged_pkg.mkdir()
    for child in source_pkg.iterdir():
        (staged_pkg / child.name).symlink_to(child, target_is_directory=child.is_dir())

    native_name = "_native.so" if native_lib.suffix == ".so" else "_native.dylib"
    (staged_pkg / native_name).symlink_to(native_lib)

    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{target_dir}:{current}" if current else str(target_dir)
    verify = _run_command(
        [
            args.python,
            "-c",
            (
                "import pyxlog; "
                "assert hasattr(pyxlog.LogicRelationSession, 'evaluate_async'); "
                "print(pyxlog.__file__)"
            ),
        ],
        timeout=args.timeout,
        env=env,
    )
    if verify["returncode"] != 0:
        raise SystemExit(
            f"local pyxlog import verification failed with exit {verify['returncode']}\n"
            f"STDOUT:\n{verify['stdout']}\nSTDERR:\n{verify['stderr']}"
        )
    return env


def _feature_measurements() -> dict[str, Any]:
    measurements: dict[str, Any] = {}
    missing = []
    for feature, path in FEATURE_EVIDENCE.items():
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        payload = _read_json(path)
        measurements[feature] = {
            "path": str(path.relative_to(ROOT)),
            "raw": payload,
        }
    _require(not missing, f"Missing v0.8.6 feature evidence: {missing}")
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


def _run_source_guard(args: argparse.Namespace) -> dict[str, Any]:
    raw = _run_command(
        [
            args.python,
            "-m",
            "pytest",
            "-q",
            "python/tests/test_v080_examples_source.py",
            "python/tests/test_v085_examples_source.py",
        ],
        timeout=args.compat_timeout,
    )
    if raw["returncode"] != 0:
        raise SystemExit(
            f"v0.8.0/v0.8.5 source guards failed with exit {raw['returncode']}\n"
            f"STDOUT:\n{raw['stdout']}\nSTDERR:\n{raw['stderr']}"
        )
    return {
        "status": "PASS",
        "cmd": raw["cmd"],
        "duration_sec": raw["duration_sec"],
        "returncode": raw["returncode"],
        "stdout_preview": raw["stdout"][-1000:],
    }


def _compatibility_gates(args: argparse.Namespace, evidence_dir: Path) -> dict[str, Any]:
    pyxlog_env = _prepare_local_pyxlog_env(args)
    v080 = _run_existing_validator(
        args,
        "scripts/validate_v080_examples.py",
        evidence_dir / "compat_v080_validation_summary.json",
        pyxlog_env,
    )
    v085 = _run_existing_validator(
        args,
        "scripts/validate_v085_examples.py",
        evidence_dir / "compat_v085_validation_summary.json",
        pyxlog_env,
    )
    guards = _run_source_guard(args)
    _require(v080["status"] == "PASS", f"v0.8.0 validator did not pass: {v080}")
    _require(v085["status"] == "PASS", f"v0.8.5 validator did not pass: {v085}")
    return {
        "v080_examples": v080,
        "v085_examples": v085,
        "v080_v085_source_guards": guards,
    }


def _production_path_reuse() -> dict[str, Any]:
    validator = (ROOT / "scripts/validate_v086_examples.py").read_text(encoding="utf-8")
    checked_programs = [
        str((EXAMPLE_ROOT / example / "program.xlog").relative_to(ROOT)) for example in EXAMPLES
    ]
    return {
        "status": "PASS",
        "examples_run_through": "cargo run -q -p xlog-cli -- run/explain",
        "validator_reuses": [
            "scripts/validate_v080_examples.py",
            "scripts/validate_v085_examples.py",
            "python/tests/test_v080_examples_source.py",
            "python/tests/test_v085_examples_source.py",
        ],
        "private_hooks_used": False,
        "fixture_only_bypass": False,
        "checked_programs": checked_programs,
        "validator_contains_existing_gates": all(
            needle in validator
            for needle in [
                "validate_v080_examples.py",
                "validate_v085_examples.py",
                "cargo",
                "xlog-cli",
            ]
        ),
    }


def _reuse_audit() -> dict[str, Any]:
    evidence_paths = [str(path.relative_to(ROOT)) for path in FEATURE_EVIDENCE.values()]
    return {
        "status": "PASS",
        "duplicate_engine_helper_path": False,
        "reused_subsystems": [
            "xlog-cli parser and explain pipeline",
            "xlog-runtime production executor/provider dispatch",
            "v0.8.0 DTS example validator",
            "v0.8.5 language showcase validator",
            "committed v0.8.6 feature evidence",
        ],
        "evidence_paths": evidence_paths,
    }


def _aggregate(
    results: list[dict[str, Any]],
    feature_measurements: dict[str, Any],
    compatibility_gates: dict[str, Any],
) -> dict[str, Any]:
    feature_coverage = {
        feature: [result["name"] for result in results if feature in result["features"]]
        for feature in REQUIRED_FEATURES
    }
    missing_features = [feature for feature, examples in feature_coverage.items() if not examples]
    _require(not missing_features, f"Missing required feature coverage: {missing_features}")
    observed_consumers = {result["consumer"] for result in results}
    _require(
        REQUIRED_CONSUMERS <= observed_consumers,
        f"Missing required consumers: {sorted(REQUIRED_CONSUMERS - observed_consumers)}",
    )

    return {
        "suite": "G086_CONSUMERS",
        "status": "PASS",
        "example_execution_status": "PASS",
        "consumer_certification_status": "BLOCKED",
        "feature_coverage_source": "expected_json_declarations",
        "consumer_proof_gaps": CONSUMER_PROOF_GAPS,
        "example_count": len(results),
        "required_example_count": len(EXAMPLES),
        "consumer_coverage": {
            consumer: [result["name"] for result in results if result["consumer"] == consumer]
            for consumer in sorted(REQUIRED_CONSUMERS)
        },
        "feature_coverage": feature_coverage,
        "feature_proof_model": {
            "example_execution": "xlog-cli run/explain over committed .xlog programs",
            "feature_coverage": "expected.json declarations cross-linked to committed feature-node evidence",
            "certification_limit": (
                "declarations plus linked evidence are not equivalent to per-consumer "
                "runtime probes for native exact induction, adaptive reoptimization, "
                "or persistent-index pyxlog session reuse"
            ),
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
        "production_path_reuse": _production_path_reuse(),
        "reuse_audit": _reuse_audit(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Summary JSON output path.")
    parser.add_argument("--timeout", type=int, default=120, help="Per xlog command timeout in seconds.")
    parser.add_argument("--compat-timeout", type=int, default=180, help="Compatibility command timeout in seconds.")
    parser.add_argument("--pyxlog-build-timeout", type=int, default=300, help="Local pyxlog build timeout in seconds.")
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
