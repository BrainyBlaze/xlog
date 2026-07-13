#!/usr/bin/env python3
"""Validate the language completeness showcase examples."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = ROOT / "examples" / "language-completeness" / "showcase"
DEFAULT_OUTPUT = ROOT / "docs-internal" / "evidence" / "language-examples" / "validation_summary.json"
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


def _base_xlog_command(args: argparse.Namespace, *, host_io: bool = False) -> list[str]:
    if host_io and args.xlog_host_io_bin:
        return [str(args.xlog_host_io_bin)]
    if args.xlog_bin:
        return [str(args.xlog_bin)]
    cmd = ["cargo", "run", "-q", "-p", "xlog-cli"]
    if host_io:
        cmd.extend(["--features", "host-io"])
    cmd.append("--")
    return cmd


def _run_xlog(
    args: argparse.Namespace,
    xlog_args: list[str],
    *,
    input_text: str | None = None,
    host_io: bool = False,
) -> dict[str, Any]:
    cmd = _base_xlog_command(args, host_io=host_io) + xlog_args
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        input=input_text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=args.timeout,
    )
    elapsed = time.perf_counter() - start
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "duration_sec": round(elapsed, 6),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


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
    combined = f"{raw['stdout']}\n{raw['stderr']}"
    _check_required_substrings(
        combined,
        expected.get("combined_contains", []),
        f"{example} run output",
    )
    return {"status": "PASS", "returncode": raw["returncode"]}


def _json_payload_from_stdout(example: str, stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    _require(start >= 0, f"{example} prob_json emitted no JSON object:\n{stdout}")
    try:
        return json.loads(stdout[start:])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{example} prob_json did not emit parseable JSON: {exc}\n{stdout}") from exc


def _check_prob_json(
    example: str,
    raw: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    _require(
        raw["returncode"] == 0,
        f"{example} prob_json failed with exit {raw['returncode']}\nSTDOUT:\n{raw['stdout']}\nSTDERR:\n{raw['stderr']}",
    )
    report = _json_payload_from_stdout(example, raw["stdout"])
    if "engine" in expected:
        _require(
            report.get("engine") == expected["engine"],
            f"{example} prob_json engine mismatch: {report}",
        )
    if "mc_engine" in expected:
        _require(
            report.get("mc_engine") == expected["mc_engine"],
            f"{example} prob_json mc_engine mismatch: {report}",
        )
    for field in ["total_samples", "seed"]:
        if field in expected:
            _require(
                report.get(field) == expected[field],
                f"{example} prob_json {field} mismatch: {report}",
            )
    if "confidence" in expected:
        _require(
            abs(float(report.get("confidence")) - float(expected["confidence"])) < 1e-12,
            f"{example} prob_json confidence mismatch: {report}",
        )

    queries = report.get("queries", [])
    by_atom = {query.get("atom"): query for query in queries}
    for atom in expected.get("query_atoms", []):
        _require(atom in by_atom, f"{example} prob_json missing query atom {atom}: {report}")
    for atom, probability in expected.get("probabilities", {}).items():
        _require(atom in by_atom, f"{example} prob_json missing probability atom {atom}: {report}")
        actual = float(by_atom[atom]["prob"])
        _require(
            abs(actual - float(probability)) <= 1e-9,
            f"{example} prob_json probability mismatch for {atom}: expected {probability}, got {actual}",
        )
    for atom, bounds in expected.get("probability_ranges", {}).items():
        _require(atom in by_atom, f"{example} prob_json missing range atom {atom}: {report}")
        actual = float(by_atom[atom]["prob"])
        lower, upper = float(bounds[0]), float(bounds[1])
        _require(
            lower <= actual <= upper,
            f"{example} prob_json probability for {atom} outside [{lower}, {upper}]: {actual}",
        )
    return report


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
    if "magic_status" in expected:
        _require(
            report["magic_sets"]["status"] == expected["magic_status"],
            f"{example} magic status mismatch: {report['magic_sets']}",
        )
    for generated in expected.get("generated_predicates", []):
        _require(
            generated in report["magic_sets"]["generated_predicates"],
            f"{example} missing generated predicate {generated}: {report['magic_sets']}",
        )
    if "prob_engine" in expected:
        _require(
            report["probability"]["engine"] == expected["prob_engine"],
            f"{example} probability engine mismatch: {report['probability']}",
        )
    if "aggregate_lifting_count_min" in expected:
        _require(
            report["probability"]["aggregate_lifting_count"]
            >= expected["aggregate_lifting_count_min"],
            f"{example} aggregate lifting count mismatch: {report['probability']}",
        )
    statuses = {entry["status"] for entry in report.get("aggregate_lifting", [])}
    for status in expected.get("aggregate_lifting_statuses", []):
        _require(
            status in statuses,
            f"{example} missing aggregate lifting status {status}: {statuses}",
        )
    if "min_dynamic_programming_states" in expected:
        max_states = max(
            (entry.get("dynamic_programming_states", 0) for entry in report.get("aggregate_lifting", [])),
            default=0,
        )
        _require(
            max_states >= expected["min_dynamic_programming_states"],
            f"{example} DP states below expected floor: {max_states}",
        )
    return report


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
    program_arg = str(program.relative_to(ROOT))

    raw_outputs: dict[str, Any] = {}
    check_results: dict[str, Any] = {}
    checks = expected.get("checks", {})

    if "run" in checks:
        raw = _run_xlog(args, ["run", program_arg])
        raw_outputs["run"] = raw
        check_results["run"] = _check_run(example, raw, checks["run"])

    if "prob_json" in checks:
        # `extra_args` lets MC examples in the resident-rejected fragment
        # (aggregates, negation) opt into the labeled CPU oracle explicitly;
        # such examples must also pin `mc_engine` to "cpu-oracle".
        extra_args = list(checks["prob_json"].get("extra_args", []))
        raw = _run_xlog(
            args,
            ["prob", program_arg, "--output", "json", *extra_args],
            host_io=True,
        )
        raw_outputs["prob_json"] = raw
        check_results["prob_json"] = _check_prob_json(example, raw, checks["prob_json"])

    if "explain_json" in checks:
        raw = _run_xlog(args, ["explain", "--format", "json", program_arg])
        raw_outputs["explain_json"] = raw
        check_results["explain_json"] = _check_explain_json(
            example,
            raw,
            checks["explain_json"],
        )

    if "repl" in checks:
        raw = _run_xlog(args, ["repl"], input_text=source)
        raw_outputs["repl"] = raw
        _require(
            raw["returncode"] == 0,
            f"{example} repl failed with exit {raw['returncode']}\nSTDOUT:\n{raw['stdout']}\nSTDERR:\n{raw['stderr']}",
        )
        _check_required_substrings(
            raw["stdout"],
            checks["repl"].get("required_substrings", []),
            f"{example} repl stdout",
        )
        check_results["repl"] = {"status": "PASS"}

    if "watch" in checks:
        raw = _run_xlog(args, ["watch", "--once", "--explain", program_arg])
        raw_outputs["watch"] = raw
        _require(
            raw["returncode"] == 0,
            f"{example} watch failed with exit {raw['returncode']}\nSTDOUT:\n{raw['stdout']}\nSTDERR:\n{raw['stderr']}",
        )
        _check_required_substrings(
            raw["stdout"],
            checks["watch"].get("required_substrings", []),
            f"{example} watch stdout",
        )
        check_results["watch"] = {"status": "PASS"}

    features = expected.get("features", [])
    return {
        "name": example,
        "status": "PASS",
        "program": program_arg,
        "features": features,
        "interaction": len(features) >= 2,
        "checks": sorted(checks),
        "check_results": check_results,
        "raw_outputs": raw_outputs,
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    feature_coverage = {
        feature: [result["name"] for result in results if feature in result["features"]]
        for feature in REQUIRED_FEATURES
    }
    missing = [feature for feature, examples in feature_coverage.items() if not examples]
    _require(not missing, f"Missing required feature coverage: {missing}")
    interaction_count = sum(1 for result in results if result["interaction"])
    _require(interaction_count >= 5, f"Expected at least 5 interactions, got {interaction_count}")

    return {
        "suite": "language_examples",
        "status": "PASS",
        "example_count": len(results),
        "required_example_count": 10,
        "interaction_count": interaction_count,
        "required_interaction_count": 5,
        "feature_coverage": feature_coverage,
        "per_example": [
            {
                "name": result["name"],
                "status": result["status"],
                "features": result["features"],
                "interaction": result["interaction"],
                "checks": result["checks"],
            }
            for result in results
        ],
        "raw_outputs": {result["name"]: result["raw_outputs"] for result in results},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Summary JSON output path.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-command timeout in seconds.")
    parser.add_argument("--xlog-bin", type=Path, help="Use an existing xlog binary instead of cargo run.")
    parser.add_argument(
        "--xlog-host-io-bin",
        type=Path,
        help="Use an existing host-io-enabled xlog binary for probabilistic semantic checks.",
    )
    args = parser.parse_args(argv)

    _require(EXAMPLE_ROOT.exists(), f"Missing example root: {EXAMPLE_ROOT}")
    results = [_load_example_result(example, args) for example in EXAMPLES]
    summary = _aggregate(results)
    _require(summary["example_count"] >= 10, f"Expected at least 10 examples: {summary['example_count']}")
    _require(
        all(item["status"] == "PASS" for item in summary["per_example"]),
        f"Expected all per-example statuses PASS: {summary['per_example']}",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
