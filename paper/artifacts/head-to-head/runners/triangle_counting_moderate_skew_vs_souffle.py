#!/usr/bin/env python3
"""Re-measure the moderate-skew triangle comparison (`triangle_wcoj_bulk_edb`).

Three arms per case over one identical edge relation: XLOG with the WCOJ
subsystem engaged (``--wcoj``), the same XLOG binary with every WCOJ-family
dispatch pinned off so binary joins run, and a standalone Souffle executable
compiled once per case. Edges are bulk-loaded as an EDB through ``--input``
Arrow IPC, never as inline facts. The query enumerates whole triangles, so the
XLOG row count and the Souffle tuple count are the same quantity and are
compared as a gate, relation hash included.

The published artifact records no generator parameters, only edge counts. The
moderate-skew parameterization here is therefore a *choice*, written into the
artifact in full (``protocol.graph_generator`` and ``results[*].generator``)
together with an explicit list of what could not be recovered
(``protocol.unrecoverable_from_published_artifact``). A run of this script is a
new measurement of the same class, not a byte-faithful re-shoot.

Nothing about the host is asserted: GPU, driver, CPU and core counts are
recorded. A command that cannot run is recorded as ``UNAVAILABLE`` with its
stderr. A failed arm is recorded as a failure and is never replaced by another
execution path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


# Case names and edge counts are the four published ones. ``hubs`` is not
# published; the values below were picked offline so that the resulting
# triangle totals land within a factor of ~1.3 of the published totals, which
# keeps the workload in the same class. node_count = hubs * --nodes-per-hub.
DEFAULT_CASES = (
    ("tri_40000", 40_000, 27),
    ("tri_100000", 100_000, 40),
    ("tri_200000", 200_000, 50),
    ("tri_400000", 400_000, 62),
)

# The heavy-skew companion runner puts a hub endpoint on 80% of its proposals.
# "Moderate" here means a materially lower share; the self-test refuses a
# realized share at or above the ceiling, so the two artifacts cannot silently
# converge on the same graph family.
HEAVY_SKEW_HUB_EDGE_FRACTION = 0.8
MODERATE_SKEW_CEILING = 0.5
DEFAULT_HUB_EDGE_FRACTION = 0.25
DEFAULT_NODES_PER_HUB = 101
HUB_FRACTION_TOLERANCE = 0.02

TIME_BIN = Path("/usr/bin/time")

# Whole triangles, not per-root counts: `rows` on the XLOG side and `count` on
# the Souffle side are then the same number. The body shape
# `e(X, Y), e(Y, Z), e(X, Z)` in head-position order with no comparison filters
# is the one the WCOJ triangle dispatcher recognizes (see
# crates/xlog-integration/src/wcoj_dispatch.rs); adding `A < B`-style guards
# would silently take the rule off that path.
XLOG_SOURCE = """\
pred edge(u32, u32).
pred triangle(u32, u32, u32).

triangle(A, B, C) :- edge(A, B), edge(B, C), edge(A, C).

?- triangle(A, B, C).
"""
SOUFFLE_SOURCE = """\
.decl edge(a:number, b:number)
.input edge
.decl triangle(a:number, b:number, c:number)
.output triangle

triangle(a, b, c) :- edge(a, b), edge(b, c), edge(a, c).
"""

# `stats.format_json` (crates/xlog-runtime/src/profiler.rs) publishes these five
# dispatch counters under "wcoj"; the CLI sums exactly this set when it reports
# whether a WCOJ kernel fired.
WCOJ_DISPATCH_COUNTERS = (
    "triangle_dispatch",
    "four_cycle_dispatch",
    "groupby_fusion_dispatch",
    "free_join_dispatch",
    "factorized_delta_dispatch",
)

# `--wcoj` sets these two inside the CLI process; the runner only clears them so
# an inherited value cannot decide the arm.
WCOJ_FORCE_ENV = ("XLOG_USE_WCOJ_TRIANGLE_U32", "XLOG_USE_WCOJ_4CYCLE")

# Adaptive triangle dispatch is default-ON (crates/xlog-integration/tests/
# test_wcoj_adaptive_default_on.rs), so merely omitting `--wcoj` does NOT
# guarantee a binary join on skewed input. The binary arm therefore sets the
# kill switches, which beat force, adaptive and the default.
WCOJ_KILL_ENV = (
    "XLOG_DISABLE_WCOJ_TRIANGLE",
    "XLOG_DISABLE_WCOJ_GROUPBY_FUSION",
    "XLOG_DISABLE_FREE_JOIN",
    "XLOG_DISABLE_FACTORIZED_DELTA",
)
WCOJ_CHAIN_ENV = "XLOG_WCOJ_CHAIN_ENABLE"

ARM_WCOJ = "xlog_wcoj"
ARM_BINARY = "xlog_binary"


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    wall_s: float
    max_rss_kb: int
    stdout: str
    stderr: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_text(argv: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stdout or completed.stderr).strip()


def probe_command(argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    """Record a descriptive command, or record why it could not be run.

    Never silent: a missing binary or a non-zero exit is written into the
    artifact as ``UNAVAILABLE`` together with its stderr.
    """
    record: dict[str, Any] = {"command": shlex.join(str(item) for item in argv)}
    try:
        completed = subprocess.run(
            [str(item) for item in argv],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        record["status"] = "UNAVAILABLE"
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    if completed.returncode != 0:
        record["status"] = "UNAVAILABLE"
        record["returncode"] = completed.returncode
        record["stderr"] = completed.stderr.strip()
        return record
    record["status"] = "OK"
    record["output"] = (completed.stdout or completed.stderr).strip()
    return record


def parse_time_metrics(lines: Iterable[str], observed_wall_s: float) -> tuple[float, int]:
    metrics: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.strip().partition("=")
        if separator and key in {"xlog_wall_s", "xlog_max_rss_kb"}:
            metrics[key] = value
    wall_s = float(metrics.get("xlog_wall_s", observed_wall_s))
    max_rss_kb = int(metrics.get("xlog_max_rss_kb", "0"))
    return wall_s, max_rss_kb


def run_timed(
    argv: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    timeout_s: int,
    metrics_path: Path,
) -> CommandResult:
    if not TIME_BIN.is_file():
        raise RuntimeError(f"GNU time is required at {TIME_BIN}")
    timed_argv = (
        str(TIME_BIN),
        "-f",
        "xlog_wall_s=%e\nxlog_max_rss_kb=%M\nxlog_exit_status=%x",
        "-o",
        str(metrics_path),
        *argv,
    )
    started = time.perf_counter()
    process = subprocess.Popen(
        timed_argv,
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        returncode = 124
        stderr += f"\nbenchmark command timed out after {timeout_s} seconds"
    observed_wall_s = time.perf_counter() - started
    metric_lines = (
        metrics_path.read_text(encoding="utf-8").splitlines() if metrics_path.exists() else []
    )
    wall_s, max_rss_kb = parse_time_metrics(metric_lines, observed_wall_s)
    return CommandResult(tuple(argv), returncode, wall_s, max_rss_kb, stdout, stderr)


def generate_moderate_skew_edges(
    hubs: int,
    node_count: int,
    hub_edge_fraction: float,
    edge_count: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Generate exactly ``edge_count`` unique directed edges with moderate hub skew.

    Node ids ``[0, hubs)`` are hubs. A proposal is hub-incident with probability
    ``hub_edge_fraction`` (source-hub and target-hub equally likely); otherwise
    both endpoints are drawn uniformly from the non-hub nodes, so the realized
    hub-incident share tracks the requested one instead of drifting upward with
    the hub count. Self loops are rejected, duplicates are absorbed by the set,
    and the result is sorted, which makes the output a pure function of the
    arguments.
    """
    if hubs <= 0:
        raise ValueError("hubs must be positive")
    if node_count <= hubs + 1:
        raise ValueError("node_count must leave at least two non-hub nodes")
    if not 0.0 <= hub_edge_fraction <= 1.0:
        raise ValueError("hub_edge_fraction must lie in [0, 1]")
    if edge_count <= 0:
        raise ValueError("edge_count must be positive")
    non_hub_count = node_count - hubs
    hub_capacity = 2 * hubs * (node_count - 1)
    background_capacity = non_hub_count * (non_hub_count - 1)
    if hub_capacity + background_capacity < edge_count:
        raise ValueError("case does not have enough distinct non-self directed edges")
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < edge_count:
        if rng.random() < hub_edge_fraction:
            if rng.random() < 0.5:
                source = rng.randrange(hubs)
                target = rng.randrange(node_count)
            else:
                source = rng.randrange(node_count)
                target = rng.randrange(hubs)
        else:
            source = hubs + rng.randrange(non_hub_count)
            target = hubs + rng.randrange(non_hub_count)
        if source != target:
            edges.add((source, target))
    return sorted(edges)


def hub_incident_fraction(edges: Sequence[tuple[int, int]], hubs: int) -> float:
    if not edges:
        raise ValueError("cannot measure hub incidence of an empty relation")
    incident = sum(1 for source, target in edges if source < hubs or target < hubs)
    return incident / len(edges)


def write_inputs(case_dir: Path, edges: Sequence[tuple[int, int]]) -> tuple[Path, Path]:
    """Write one relation twice: Arrow IPC for XLOG's EDB, TSV facts for Souffle."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    arrow_path = case_dir / "edge.arrow"
    facts_path = case_dir / "edge.facts"
    table = pa.table(
        {
            "a": pa.array([edge[0] for edge in edges], type=pa.uint32()),
            "b": pa.array([edge[1] for edge in edges], type=pa.uint32()),
        }
    )
    with arrow_path.open("wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
    with facts_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerows(edges)
    return arrow_path, facts_path


def read_arrow_pairs(path: Path) -> set[tuple[int, int]]:
    import pyarrow.ipc as ipc

    with path.open("rb") as source:
        table = ipc.open_stream(source).read_all()
    if table.num_columns != 2:
        raise RuntimeError(f"expected two edge columns, got {table.num_columns}")
    left = table.column(0).combine_chunks().to_pylist()
    right = table.column(1).combine_chunks().to_pylist()
    if any(value is None for value in left) or any(value is None for value in right):
        raise RuntimeError("Arrow edge relation contains a null field")
    return {(int(a), int(b)) for a, b in zip(left, right)}


def read_facts_pairs(path: Path) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 2:
                raise RuntimeError(f"expected two fact columns, got {len(row)}")
            pairs.add((int(row[0]), int(row[1])))
    return pairs


def parse_stats(stderr: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for line in stderr.splitlines():
        line = line.strip()
        if line.startswith("{") and '"total_ms"' in line:
            value = json.loads(line)
            if isinstance(value, dict):
                candidates.append(value)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one XLOG stats object, found {len(candidates)}")
    return candidates[0]


def relation_sha256(rows: Iterable[tuple[int, ...]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows):
        digest.update(("\t".join(str(value) for value in row) + "\n").encode())
    return digest.hexdigest()


def triangle_relation_summary(rows: Iterable[tuple[int, int, int]]) -> dict[str, Any]:
    """Validate and summarize an enumerated triangle relation.

    These are protocol checks on what an engine returned, not assertions about
    the host: a duplicate tuple or a degenerate triangle means the two sides are
    not counting the same thing and the comparison must not be reported as sound.
    """
    canonical_rows = sorted(rows)
    if len(canonical_rows) != len(set(canonical_rows)):
        raise RuntimeError("triangle relation contains duplicate tuples")
    for left, middle, right in canonical_rows:
        if left == middle or middle == right or left == right:
            raise RuntimeError(f"degenerate triangle in relation: {(left, middle, right)}")
    return {
        "rows": len(canonical_rows),
        "relation_sha256": relation_sha256(canonical_rows),
    }


def read_xlog_triangles(path: Path) -> dict[str, Any]:
    import pyarrow.ipc as ipc

    with path.open("rb") as source:
        table = ipc.open_stream(source).read_all()
    if table.num_columns != 3:
        raise RuntimeError(f"expected three XLOG output columns, got {table.num_columns}")
    columns = [table.column(index).combine_chunks().to_pylist() for index in range(3)]
    if any(value is None for column in columns for value in column):
        raise RuntimeError("XLOG emitted a null triangle field")
    return triangle_relation_summary(
        (int(a), int(b), int(c)) for a, b, c in zip(*columns)
    )


def read_souffle_triangles(path: Path) -> dict[str, Any]:
    rows: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 3:
                raise RuntimeError(f"expected three Souffle output columns, got {len(row)}")
            rows.append((int(row[0]), int(row[1]), int(row[2])))
    return triangle_relation_summary(rows)


def relations_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        left.get("complete")
        and right.get("complete")
        and left.get("rows") == right.get("rows")
        and left.get("relation_sha256") == right.get("relation_sha256")
    )


def normalized_command(argv: Iterable[str], repo: Path, work_dir: Path) -> str:
    replacements = ((str(repo), "{repo}"), (str(work_dir), "{workdir}"))
    normalized = []
    for argument in argv:
        value = str(argument)
        for prefix, replacement in replacements:
            value = value.replace(prefix, replacement)
        normalized.append(value)
    return shlex.join(normalized)


def error_record(result: CommandResult) -> dict[str, Any]:
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    diagnostic = next((line for line in reversed(lines) if "Error:" in line), None)
    diagnostic = diagnostic or (lines[-1] if lines else "command failed without stderr")
    match = re.search(r"(?:Error:\s*)?([A-Za-z][A-Za-z0-9]+)\s*\{", diagnostic)
    return {
        "kind": match.group(1) if match else "process_failure",
        "diagnostic": diagnostic,
        "stdout_sha256": sha256_bytes(result.stdout.encode()),
        "stderr_sha256": sha256_bytes(result.stderr.encode()),
        "stderr_tail": "\n".join(lines[-20:]),
    }


def wcoj_dispatch_total(wcoj: Mapping[str, Any]) -> int:
    return sum(int(wcoj.get(counter, 0)) for counter in WCOJ_DISPATCH_COUNTERS)


def dispatch_matches_arm(wcoj: Mapping[str, Any], arm: str) -> bool:
    """Decide from the dispatch counters whether the run took the claimed route.

    `rc == 0` proves nothing about the route: WCOJ declines fall back to binary
    joins silently, and adaptive triangle dispatch is default-on, so a run
    without `--wcoj` can still fire a WCOJ kernel unless the kill switch is set.
    """
    if arm not in {ARM_WCOJ, ARM_BINARY}:
        raise ValueError(f"unknown arm: {arm}")
    total = wcoj_dispatch_total(wcoj)
    return total > 0 if arm == ARM_WCOJ else total == 0


def arm_environment(arm: str) -> dict[str, str]:
    """Environment overrides that pin the arm, applied on top of os.environ."""
    if arm == ARM_WCOJ:
        return {name: "" for name in (*WCOJ_FORCE_ENV, *WCOJ_KILL_ENV, WCOJ_CHAIN_ENV)}
    if arm == ARM_BINARY:
        overrides: dict[str, str] = {name: "" for name in WCOJ_FORCE_ENV}
        overrides.update({name: "1" for name in WCOJ_KILL_ENV})
        overrides[WCOJ_CHAIN_ENV] = "0"
        return overrides
    raise ValueError(f"unknown arm: {arm}")


def describe_environment(arm: str) -> dict[str, str]:
    """Readable form of the overrides for the artifact: "unset" instead of ""."""
    return {
        name: (value if value else "unset") for name, value in arm_environment(arm).items()
    }


def build_environment(arm: str) -> dict[str, str]:
    env = os.environ.copy()
    for name, value in arm_environment(arm).items():
        if value == "":
            env.pop(name, None)
        else:
            env[name] = value
    return env


def xlog_run(
    xlog_bin: Path,
    source: Path,
    arrow_input: Path,
    output_dir: Path,
    memory_mb: int,
    arm: str,
    repetition: int,
    repo: Path,
    work_dir: Path,
    timeout_s: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True)
    argv: tuple[str, ...] = (
        str(xlog_bin),
        "run",
        str(source),
        "--input",
        f"edge={arrow_input}",
        *(("--wcoj",) if arm == ARM_WCOJ else ()),
        "--memory-mb",
        str(memory_mb),
        "--output",
        "arrow",
        "--output-dir",
        str(output_dir),
        "--stats",
        "--stats-format",
        "json",
    )
    result = run_timed(
        argv,
        repo,
        build_environment(arm),
        timeout_s,
        output_dir / f"time-{repetition}.txt",
    )
    record: dict[str, Any] = {
        "repetition": repetition,
        "arm": arm,
        "command": normalized_command(argv, repo, work_dir),
        "env_overrides": describe_environment(arm),
        "returncode": result.returncode,
        "wall_s": round(result.wall_s, 6),
        "process_max_rss_kb": result.max_rss_kb,
    }
    if result.returncode != 0:
        record["error"] = error_record(result)
        return record
    try:
        stats = parse_stats(result.stderr)
        wcoj = stats.get("wcoj", {})
        record["wcoj_dispatch"] = wcoj
        record["wcoj_dispatch_total"] = wcoj_dispatch_total(wcoj)
        if not dispatch_matches_arm(wcoj, arm):
            raise RuntimeError(f"arm {arm} did not take its route; wcoj counters: {wcoj}")
        triangles = read_xlog_triangles(output_dir / "query_0.arrow")
        record.update(
            {
                "engine_total_ms": stats["total_ms"],
                "engine_peak_mb": stats["peak_memory_mb"],
                "engine_budget_mb": stats["budget_memory_mb"],
                "engine_output_rows": stats.get("output_rows"),
                "stats_output_rows_equals_query_rows": (
                    stats.get("output_rows") == triangles["rows"]
                ),
                **triangles,
            }
        )
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        record["returncode"] = 125
        record["error"] = {
            "kind": "protocol_violation",
            "diagnostic": str(error),
            "stdout_sha256": sha256_bytes(result.stdout.encode()),
            "stderr_sha256": sha256_bytes(result.stderr.encode()),
        }
    return record


def souffle_compile_command(
    souffle_bin: Path, source: Path, executable: Path, jobs: int
) -> tuple[str, ...]:
    return (
        str(souffle_bin),
        f"--jobs={jobs}",
        f"--dl-program={executable}",
        str(source),
    )


def souffle_execution_command(
    executable: Path, case_dir: Path, output_dir: Path, jobs: int
) -> tuple[str, ...]:
    return (
        str(executable),
        f"--jobs={jobs}",
        "-F",
        str(case_dir),
        "-D",
        str(output_dir),
    )


def compile_souffle(
    souffle_bin: Path,
    source: Path,
    case_dir: Path,
    jobs: int,
    repo: Path,
    work_dir: Path,
    timeout_s: int,
) -> tuple[dict[str, Any], Path | None]:
    executable = case_dir / "triangle_souffle"
    argv = souffle_compile_command(souffle_bin, source, executable, jobs)
    result = run_timed(argv, repo, os.environ, timeout_s, case_dir / "souffle-compile-time.txt")
    record: dict[str, Any] = {
        "command": normalized_command(argv, repo, work_dir),
        "returncode": result.returncode,
        "wall_s": round(result.wall_s, 6),
        "process_max_rss_kb": result.max_rss_kb,
    }
    if result.returncode != 0:
        record["error"] = error_record(result)
        return record, None
    if not executable.is_file() or not os.access(executable, os.X_OK):
        record["returncode"] = 125
        record["error"] = {
            "kind": "protocol_violation",
            "diagnostic": f"Souffle did not create executable: {executable}",
        }
        return record, None
    record["executable_sha256"] = sha256_file(executable)
    return record, executable


def souffle_run(
    executable: Path,
    case_dir: Path,
    repetition: int,
    jobs: int,
    repo: Path,
    work_dir: Path,
    timeout_s: int,
) -> dict[str, Any]:
    output_dir = case_dir / f"souffle-{repetition}"
    output_dir.mkdir(parents=True)
    argv = souffle_execution_command(executable, case_dir, output_dir, jobs)
    result = run_timed(argv, repo, os.environ, timeout_s, output_dir / "time.txt")
    record: dict[str, Any] = {
        "repetition": repetition,
        "arm": "souffle",
        "command": normalized_command(argv, repo, work_dir),
        "returncode": result.returncode,
        "wall_s": round(result.wall_s, 6),
        "process_max_rss_kb": result.max_rss_kb,
    }
    if result.returncode != 0:
        record["error"] = error_record(result)
        return record
    try:
        record.update(read_souffle_triangles(output_dir / "triangle.csv"))
    except (OSError, RuntimeError, ValueError) as error:
        record["returncode"] = 125
        record["error"] = {"kind": "protocol_violation", "diagnostic": str(error)}
    return record


def summarize_runs(runs: Sequence[dict[str, Any]], include_engine: bool) -> dict[str, Any]:
    successful = [run for run in runs if run["returncode"] == 0]
    summary: dict[str, Any] = {
        "complete": len(successful) == len(runs),
        "successful_repetitions": len(successful),
        "requested_repetitions": len(runs),
        "runs": list(runs),
    }
    if not successful:
        return summary
    rows = {run["rows"] for run in successful}
    relation_hashes = {run["relation_sha256"] for run in successful}
    if len(rows) != 1 or len(relation_hashes) != 1:
        summary["complete"] = False
        summary["consistency_error"] = "successful repetitions produced different relations"
        return summary
    summary.update(
        {
            "wall_s": round(median(run["wall_s"] for run in successful), 6),
            "process_max_rss_kb": max(run["process_max_rss_kb"] for run in successful),
            "rows": rows.pop(),
            "relation_sha256": relation_hashes.pop(),
        }
    )
    if include_engine:
        summary["compute_ms"] = median(run["engine_total_ms"] for run in successful)
        summary["peak_mb"] = max(run["engine_peak_mb"] for run in successful)
        summary["budget_mb"] = successful[0]["engine_budget_mb"]
        summary["wcoj_dispatch_total"] = max(
            run.get("wcoj_dispatch_total", 0) for run in successful
        )
    return summary


def first_failure(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return next((run for run in runs if run["returncode"] != 0), None)


def published_xlog_view(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The five keys the published artifact carries per XLOG arm, plus failures."""
    failure = first_failure(summary["runs"])
    if summary["complete"] and failure is None:
        return {
            "compute_ms": summary["compute_ms"],
            "rows": summary["rows"],
            "wall_s": round(summary["wall_s"], 2),
            "rc": 0,
            "err": None,
        }
    view: dict[str, Any] = {
        "compute_ms": None,
        "rows": summary.get("rows"),
        "wall_s": round(summary["wall_s"], 2) if "wall_s" in summary else None,
        "rc": failure["returncode"] if failure else 125,
        "err": (
            failure.get("error")
            if failure
            else {
                "kind": "protocol_violation",
                "diagnostic": summary.get("consistency_error", "arm did not complete"),
            }
        ),
    }
    return view


def published_souffle_view(summary: Mapping[str, Any]) -> dict[str, Any]:
    failure = first_failure(summary["runs"])
    if summary["complete"] and failure is None:
        return {
            "compute_s": round(summary["wall_s"], 3),
            "count": summary["rows"],
            "rc": 0,
            "err": None,
        }
    return {
        "compute_s": None,
        "count": summary.get("rows"),
        "rc": failure["returncode"] if failure else 125,
        "err": (
            failure.get("error")
            if failure
            else {
                "kind": "protocol_violation",
                "diagnostic": summary.get("consistency_error", "arm did not complete"),
            }
        ),
    }


def ratio_or_note(numerator_s: float | None, denominator_s: float | None) -> Any:
    """Ratio of two second-valued measurements, or a string saying why not.

    XLOG's engine time is published in whole milliseconds, so a fast case can
    report 0 ms; that is a resolution limit and must be visible as one rather
    than as a crash or an invented number.
    """
    if numerator_s is None or denominator_s is None:
        return "UNAVAILABLE: an arm did not complete"
    if denominator_s <= 0:
        return "UNAVAILABLE: denominator below the 1 ms reporting resolution"
    return round(numerator_s / denominator_s, 2)


def resolve_executable(label: str, requested: Path) -> Path:
    resolved = Path(shutil.which(str(requested)) or requested).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"{label} binary is not executable: {resolved}")
    return resolved


def repository_state(repo: Path, allow_dirty: bool) -> dict[str, Any]:
    commit = run_text(("git", "rev-parse", "HEAD"), repo)
    dirty_lines = run_text(("git", "status", "--porcelain"), repo).splitlines()
    if dirty_lines and not allow_dirty:
        raise RuntimeError("official benchmark requires a clean checkout; commit changes first")
    return {
        "commit": commit,
        "dirty": bool(dirty_lines),
        "remote": run_text(("git", "remote", "get-url", "origin"), repo),
    }


def cpu_model_name(cpuinfo: str) -> str | None:
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"model name", "Hardware", "Processor"}:
            model = value.strip()
            if model:
                return model
    return None


def parse_cpu_quota_cores(quota: str, period: str) -> float | None:
    if quota in {"max", "-1"}:
        return None
    quota_value = int(quota)
    period_value = int(period)
    if quota_value <= 0 or period_value <= 0:
        raise RuntimeError(f"invalid cgroup CPU quota: quota={quota_value} period={period_value}")
    return round(quota_value / period_value, 6)


def cgroup_cpu_quota_cores() -> dict[str, Any]:
    """Report the cgroup v2 and v1 CPU quotas separately; neither is required."""
    report: dict[str, Any] = {"v2": None, "v1": None}
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        parts = cpu_max.read_text(encoding="utf-8").split()
        if len(parts) != 2:
            raise RuntimeError(f"invalid cgroup v2 cpu.max: {parts}")
        report["v2"] = parse_cpu_quota_cores(parts[0], parts[1])
    else:
        report["v2"] = "UNAVAILABLE: /sys/fs/cgroup/cpu.max is absent"
    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.is_file() and period_path.is_file():
        report["v1"] = parse_cpu_quota_cores(
            quota_path.read_text(encoding="utf-8").strip(),
            period_path.read_text(encoding="utf-8").strip(),
        )
    else:
        report["v1"] = "UNAVAILABLE: /sys/fs/cgroup/cpu/cpu.cfs_{quota,period}_us are absent"
    return report


def hardware_state(repo: Path) -> dict[str, Any]:
    gpu = probe_command(
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        repo,
    )
    if gpu.get("status") == "OK":
        gpu["gpus"] = gpu.pop("output").splitlines()
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.is_file():
        cpu = cpu_model_name(cpuinfo_path.read_text(encoding="utf-8"))
        cpu_record: Any = cpu or "UNAVAILABLE: /proc/cpuinfo does not identify the CPU model"
    else:
        cpu_record = "UNAVAILABLE: /proc/cpuinfo is absent"
    state: dict[str, Any] = {
        "gpu": gpu,
        "cpu": cpu_record,
        "logical_cpu_count": os.cpu_count(),
        "cpu_quota_cores": cgroup_cpu_quota_cores(),
        "platform": platform.platform(),
    }
    try:
        state["host_memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError) as error:
        state["host_memory_bytes"] = f"UNAVAILABLE: {type(error).__name__}: {error}"
    return state


def default_repository(script: Path) -> Path:
    """Prefer an enclosing checkout, then the working directory, then the layout."""
    for candidate in (*script.resolve().parents, Path.cwd().resolve()):
        if (candidate / ".git").exists():
            return candidate
    parents = script.resolve().parents
    return parents[4] if len(parents) > 4 else Path.cwd().resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_repo = default_repository(script)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=default_repo)
    parser.add_argument("--xlog-bin", type=Path, default=default_repo / "target/release/xlog")
    parser.add_argument("--souffle-bin", type=Path, default=Path("souffle"))
    parser.add_argument("--nvcc-bin", type=Path, default=Path("nvcc"))
    parser.add_argument("--souffle-jobs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            default_repo
            / "paper/artifacts/head-to-head/triangle_counting_moderate_skew_vs_souffle.json"
        ),
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--memory-mb", type=int, default=18 * 1024)
    parser.add_argument("--seed", type=int, default=20_260_824)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument(
        "--case",
        action="append",
        choices=[case[0] for case in DEFAULT_CASES],
        help="run only the selected case; repeat for multiple cases",
    )
    parser.add_argument(
        "--hub-edge-fraction",
        type=float,
        default=DEFAULT_HUB_EDGE_FRACTION,
        help=(
            "probability that a proposed edge is hub-incident; the heavy-skew "
            f"companion uses {HEAVY_SKEW_HUB_EDGE_FRACTION}, this run refuses "
            f"anything at or above {MODERATE_SKEW_CEILING}"
        ),
    )
    parser.add_argument(
        "--nodes-per-hub",
        type=int,
        default=DEFAULT_NODES_PER_HUB,
        help="node_count = hubs * this; larger values thin the graph out",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def self_test() -> None:
    """Exercise the pure parts: no GPU, no XLOG, no Souffle, no nvidia-smi."""
    hubs, node_count, fraction = 8, 8 * 101, DEFAULT_HUB_EDGE_FRACTION
    first = generate_moderate_skew_edges(hubs, node_count, fraction, 4000, 7)
    second = generate_moderate_skew_edges(hubs, node_count, fraction, 4000, 7)
    assert first == second, "generator is not reproducible for a fixed seed"
    assert relation_sha256(first) == relation_sha256(second)
    assert len(first) == len(set(first)) == 4000, "generator did not emit 4000 unique edges"
    assert all(left != right for left, right in first), "generator emitted a self loop"
    assert all(
        0 <= value < node_count for edge in first for value in edge
    ), "generator emitted an out-of-range node id"
    assert generate_moderate_skew_edges(hubs, node_count, fraction, 4000, 8) != first

    realized = hub_incident_fraction(first, hubs)
    assert abs(realized - fraction) <= HUB_FRACTION_TOLERANCE, (
        f"realized hub-incident share {realized:.4f} is outside "
        f"{fraction} +- {HUB_FRACTION_TOLERANCE}"
    )
    assert realized < MODERATE_SKEW_CEILING, "the 'moderate' graph is not moderate"
    heavy = generate_moderate_skew_edges(
        hubs, node_count, HEAVY_SKEW_HUB_EDGE_FRACTION, 4000, 7
    )
    assert hub_incident_fraction(heavy, hubs) - realized > 0.4, (
        "moderate and heavy parameterizations produce indistinguishable skew"
    )
    assert hub_incident_fraction([(0, 500), (500, 600)], 8) == 0.5
    for bad_arguments in (
        (0, 808, 0.25, 10, 1),
        (8, 9, 0.25, 10, 1),
        (8, 808, 1.5, 10, 1),
        (8, 808, 0.25, 0, 1),
        (4, 10, 0.25, 10_000, 1),
    ):
        try:
            generate_moderate_skew_edges(*bad_arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid generator arguments accepted: {bad_arguments}")

    # The Arrow half of this check needs pyarrow, which is a benchmark runtime
    # dependency: python/tests/test_triangle_benchmark_runner.py runs the sibling
    # runner's self-test under `python -S` precisely to prove the self-test does
    # not need one. Skipping is announced rather than silent - a check that
    # quietly did not run reads exactly like a check that passed.
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print(
            "self-test: pyarrow absent, Arrow<->facts equivalence NOT checked",
            file=sys.stderr,
        )
        arrow_available = False
    else:
        arrow_available = True

    if arrow_available:
        with tempfile.TemporaryDirectory(prefix="xlog-moderate-selftest-") as raw_directory:
            directory = Path(raw_directory)
            arrow_path, facts_path = write_inputs(directory, first)
            arrow_pairs = read_arrow_pairs(arrow_path)
            facts_pairs = read_facts_pairs(facts_path)
            assert arrow_pairs == set(first), "Arrow IPC round-trip lost or changed edges"
            assert facts_pairs == set(first), "Souffle facts round-trip lost or changed edges"
            assert arrow_pairs == facts_pairs, "the two encodings hold different relations"
            second_directory = directory / "again"
            second_directory.mkdir()
            repeat_arrow, repeat_facts = write_inputs(second_directory, first)
            assert sha256_file(repeat_facts) == sha256_file(facts_path)
            assert sha256_file(repeat_arrow) == sha256_file(arrow_path)

    triangles = triangle_relation_summary([(2, 3, 4), (1, 5, 9)])
    assert triangles == triangle_relation_summary([(1, 5, 9), (2, 3, 4)])
    assert triangles["rows"] == 2
    assert relations_match({"complete": True, **triangles}, {"complete": True, **triangles})
    assert not relations_match(
        {"complete": True, **triangles},
        {"complete": True, **triangle_relation_summary([(1, 5, 9)])},
    )
    assert not relations_match({"complete": False, **triangles}, {"complete": True, **triangles})
    for invalid_rows in ([(1, 2, 3), (1, 2, 3)], [(1, 1, 3)], [(1, 2, 2)], [(1, 2, 1)]):
        try:
            triangle_relation_summary(invalid_rows)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid triangle relation accepted: {invalid_rows}")

    wcoj_fired = {"triangle_dispatch": 3, "four_cycle_dispatch": 0}
    fusion_only = {"groupby_fusion_dispatch": 2}
    silent = {counter: 0 for counter in WCOJ_DISPATCH_COUNTERS}
    silent_with_fallback = {**silent, "fallback": {"total": 7}, "error_decline": 1}
    assert wcoj_dispatch_total(wcoj_fired) == 3
    assert wcoj_dispatch_total(silent_with_fallback) == 0
    assert dispatch_matches_arm(wcoj_fired, ARM_WCOJ)
    assert dispatch_matches_arm(fusion_only, ARM_WCOJ)
    assert not dispatch_matches_arm(silent, ARM_WCOJ)
    assert not dispatch_matches_arm(silent_with_fallback, ARM_WCOJ)
    assert dispatch_matches_arm(silent, ARM_BINARY)
    assert dispatch_matches_arm(silent_with_fallback, ARM_BINARY)
    assert not dispatch_matches_arm(wcoj_fired, ARM_BINARY)
    assert not dispatch_matches_arm(fusion_only, ARM_BINARY)
    try:
        dispatch_matches_arm(silent, "souffle")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown arm accepted by the dispatch check")

    binary_env = arm_environment(ARM_BINARY)
    assert all(binary_env[name] == "1" for name in WCOJ_KILL_ENV)
    assert all(binary_env[name] == "" for name in WCOJ_FORCE_ENV)
    assert binary_env[WCOJ_CHAIN_ENV] == "0"
    wcoj_env = arm_environment(ARM_WCOJ)
    assert all(value == "" for value in wcoj_env.values())
    assert set(wcoj_env) == set(binary_env)
    described_binary = describe_environment(ARM_BINARY)
    assert all(described_binary[name] == "1" for name in WCOJ_KILL_ENV)
    assert all(described_binary[name] == "unset" for name in WCOJ_FORCE_ENV)
    assert described_binary[WCOJ_CHAIN_ENV] == "0"
    assert set(describe_environment(ARM_WCOJ).values()) == {"unset"}

    summary = summarize_runs(
        [
            {
                "returncode": 0,
                "wall_s": 1.0,
                "process_max_rss_kb": 2,
                "rows": 42,
                "relation_sha256": "stable",
                "engine_total_ms": 7,
                "engine_peak_mb": 6,
                "engine_budget_mb": 18432,
                "wcoj_dispatch_total": 1,
            },
            {
                "returncode": 0,
                "wall_s": 3.0,
                "process_max_rss_kb": 8,
                "rows": 42,
                "relation_sha256": "stable",
                "engine_total_ms": 9,
                "engine_peak_mb": 6,
                "engine_budget_mb": 18432,
                "wcoj_dispatch_total": 1,
            },
            {
                "returncode": 0,
                "wall_s": 2.0,
                "process_max_rss_kb": 4,
                "rows": 42,
                "relation_sha256": "stable",
                "engine_total_ms": 11,
                "engine_peak_mb": 6,
                "engine_budget_mb": 18432,
                "wcoj_dispatch_total": 1,
            },
        ],
        include_engine=True,
    )
    assert summary["complete"] and summary["wall_s"] == 2.0 and summary["compute_ms"] == 9
    assert published_xlog_view(summary) == {
        "compute_ms": 9,
        "rows": 42,
        "wall_s": 2.0,
        "rc": 0,
        "err": None,
    }
    inconsistent = summarize_runs(
        [
            {
                "returncode": 0,
                "wall_s": 1.0,
                "process_max_rss_kb": 2,
                "rows": 42,
                "relation_sha256": "a",
                "engine_total_ms": 7,
                "engine_peak_mb": 6,
                "engine_budget_mb": 1,
            },
            {
                "returncode": 0,
                "wall_s": 1.0,
                "process_max_rss_kb": 2,
                "rows": 42,
                "relation_sha256": "b",
                "engine_total_ms": 7,
                "engine_peak_mb": 6,
                "engine_budget_mb": 1,
            },
        ],
        include_engine=True,
    )
    assert not inconsistent["complete"] and "consistency_error" in inconsistent
    failed = summarize_runs(
        [{"returncode": 101, "wall_s": 0.2, "error": {"kind": "CapacityExceeded"}}],
        include_engine=True,
    )
    assert not failed["complete"]
    failed_view = published_xlog_view(failed)
    assert failed_view["rc"] == 101 and failed_view["compute_ms"] is None
    assert failed_view["err"] == {"kind": "CapacityExceeded"}
    souffle_view = published_souffle_view(
        summarize_runs(
            [
                {
                    "returncode": 0,
                    "wall_s": 0.0554,
                    "process_max_rss_kb": 2,
                    "rows": 42,
                    "relation_sha256": "stable",
                }
            ],
            include_engine=False,
        )
    )
    assert souffle_view == {"compute_s": 0.055, "count": 42, "rc": 0, "err": None}

    assert ratio_or_note(0.055, 0.007) == 7.86
    assert ratio_or_note(0.009, 0.007) == 1.29
    assert isinstance(ratio_or_note(0.055, 0.0), str)
    assert isinstance(ratio_or_note(None, 0.007), str)

    assert parse_time_metrics(
        [
            "Command exited with non-zero status 1",
            "xlog_wall_s=1.25",
            "xlog_max_rss_kb=2048",
            "xlog_exit_status=1",
        ],
        9.0,
    ) == (1.25, 2048)
    assert parse_time_metrics([], 9.0) == (9.0, 0)
    assert parse_cpu_quota_cores("765000", "100000") == 7.65
    assert parse_cpu_quota_cores("max", "100000") is None
    assert parse_cpu_quota_cores("-1", "100000") is None
    assert cpu_model_name("processor: 0\nmodel name: Example CPU\n") == "Example CPU"
    assert cpu_model_name("processor: 0\n") is None

    # Path rendering is platform-dependent, so assert the contract rather than a
    # literal string: both roots are replaced by their placeholders, neither
    # leaks, and the same arguments always normalize to the same text.
    repo = Path("/repo").resolve()
    work = Path("/work").resolve()
    argv = (repo / "target/release/xlog", "run", "--input", f"edge={work / 'c/edge.arrow'}")
    normalized = normalized_command(argv, repo, work)
    assert "{repo}" in normalized and "{workdir}" in normalized
    assert str(repo) not in normalized and str(work) not in normalized
    assert normalized == normalized_command(argv, repo, work)
    assert normalized_command(("a b", "c"), repo, work) == "'a b' c"
    souffle_path = Path("/souffle").resolve()
    dl_source = Path("/c/t.dl").resolve()
    dl_program = Path("/c/t").resolve()
    case_directory = Path("/c").resolve()
    output_directory = Path("/c/out").resolve()
    assert souffle_compile_command(souffle_path, dl_source, dl_program, 8) == (
        str(souffle_path),
        "--jobs=8",
        f"--dl-program={dl_program}",
        str(dl_source),
    )
    assert souffle_execution_command(dl_program, case_directory, output_directory, 8) == (
        str(dl_program),
        "--jobs=8",
        "-F",
        str(case_directory),
        "-D",
        str(output_directory),
    )

    assert sha256_bytes(b"") == sha256_bytes(b"")
    assert sha256_bytes(XLOG_SOURCE.encode()) != sha256_bytes(SOUFFLE_SOURCE.encode())
    for kind in ("CapacityExceeded", "ResourceExhausted"):
        parsed_error = error_record(
            CommandResult(
                argv=("xlog",),
                returncode=1,
                wall_s=0.1,
                max_rss_kb=1,
                stdout="",
                stderr=f"Error: {kind} {{ diagnostic: test }}",
            )
        )
        assert parsed_error["kind"] == kind
    assert (
        error_record(
            CommandResult(("xlog",), 1, 0.1, 1, "", "")
        )["diagnostic"]
        == "command failed without stderr"
    )

    missing = probe_command(("xlog-self-test-missing-command",), Path.cwd())
    assert missing["status"] == "UNAVAILABLE" and "error" in missing
    present = probe_command((sys.executable, "-c", "print('ok')"), Path.cwd())
    assert present["status"] == "OK" and present["output"] == "ok"
    failing = probe_command((sys.executable, "-c", "raise SystemExit(3)"), Path.cwd())
    assert failing["status"] == "UNAVAILABLE" and failing["returncode"] == 3

    assert resolve_executable("Python", Path(sys.executable)) == Path(sys.executable).resolve()
    try:
        resolve_executable("missing test", Path("/xlog-self-test-missing-executable"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing executable was accepted")

    for name, edges, hubs_count in DEFAULT_CASES:
        node_total = hubs_count * DEFAULT_NODES_PER_HUB
        assert edges > 0 and hubs_count > 0 and node_total > hubs_count + 1, name

    sys.stderr.write("moderate-skew triangle runner self-test passed\n")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    import pyarrow as pa

    if args.repetitions <= 0 or args.memory_mb <= 0 or args.timeout_s <= 0:
        raise ValueError("repetitions, memory-mb and timeout-s must be positive")
    if args.souffle_jobs <= 0 or args.nodes_per_hub <= 1:
        raise ValueError("souffle-jobs must be positive and nodes-per-hub must exceed 1")
    if not 0.0 <= args.hub_edge_fraction < MODERATE_SKEW_CEILING:
        raise ValueError(
            f"hub-edge-fraction must lie in [0, {MODERATE_SKEW_CEILING}); this artifact is "
            "the moderate-skew companion and must not silently become the heavy-skew one"
        )
    repo = args.repo.resolve()
    resolve_executable("GNU time", TIME_BIN)
    xlog_bin = resolve_executable("XLOG", args.xlog_bin)
    souffle_bin = resolve_executable("Souffle", args.souffle_bin)

    selected = set(args.case or (case[0] for case in DEFAULT_CASES))
    cases = [case for case in DEFAULT_CASES if case[0] in selected]
    owned_work_dir = args.work_dir is None
    work_dir = Path(
        tempfile.mkdtemp(
            prefix="xlog-triangle-moderate-",
            dir=str(args.work_dir.resolve()) if args.work_dir else None,
        )
    )
    try:
        repo_info = repository_state(repo, args.allow_dirty)
        runner_path = Path(__file__).resolve()
        try:
            runner_relative: str = str(runner_path.relative_to(repo))
            runner_inside_repo = True
        except ValueError:
            runner_relative = str(runner_path)
            runner_inside_repo = False
        hardware_info = hardware_state(repo)
        software_info = {
            "python": platform.python_version(),
            "pyarrow": pa.__version__,
            "xlog": {
                "path": normalized_command((str(xlog_bin),), repo, work_dir),
                "sha256": sha256_file(xlog_bin),
                "version": probe_command((str(xlog_bin), "--version"), repo),
            },
            "souffle": {
                "path": str(souffle_bin),
                "sha256": sha256_file(souffle_bin),
                "version": probe_command((str(souffle_bin), "--version"), repo),
            },
            "nvcc": probe_command((str(args.nvcc_bin), "--version"), repo),
            "host_cxx": probe_command(("c++", "--version"), repo),
        }
        results: list[dict[str, Any]] = []
        for case_name, edge_count, hubs in cases:
            sys.stderr.write(f"BEGIN {case_name}\n")
            sys.stderr.flush()
            case_dir = work_dir / case_name
            case_dir.mkdir()
            source = case_dir / "triangle.xlog"
            souffle_source = case_dir / "triangle.dl"
            source.write_text(XLOG_SOURCE, encoding="utf-8")
            souffle_source.write_text(SOUFFLE_SOURCE, encoding="utf-8")
            node_count = hubs * args.nodes_per_hub
            case_seed = args.seed + hubs * 1_000_003 + edge_count
            edges = generate_moderate_skew_edges(
                hubs, node_count, args.hub_edge_fraction, edge_count, case_seed
            )
            realized_fraction = hub_incident_fraction(edges, hubs)
            arrow_input, facts_input = write_inputs(case_dir, edges)
            souffle_compile, souffle_executable = compile_souffle(
                souffle_bin,
                souffle_source,
                case_dir,
                args.souffle_jobs,
                repo,
                work_dir,
                args.timeout_s,
            )
            sys.stderr.write(
                f"COMPILED {case_name} success={souffle_executable is not None}\n"
            )
            sys.stderr.flush()

            wcoj_runs: list[dict[str, Any]] = []
            binary_runs: list[dict[str, Any]] = []
            souffle_runs: list[dict[str, Any]] = []
            for repetition in range(1, args.repetitions + 1):
                for arm, sink in ((ARM_WCOJ, wcoj_runs), (ARM_BINARY, binary_runs)):
                    sink.append(
                        xlog_run(
                            xlog_bin,
                            source,
                            arrow_input,
                            case_dir / f"{arm}-{repetition}",
                            args.memory_mb,
                            arm,
                            repetition,
                            repo,
                            work_dir,
                            args.timeout_s,
                        )
                    )
                if souffle_executable is None:
                    souffle_runs.append(
                        {
                            "repetition": repetition,
                            "arm": "souffle",
                            "returncode": 126,
                            "error": {
                                "kind": "prerequisite_failure",
                                "diagnostic": "Souffle compilation failed; see souffle_compile",
                            },
                        }
                    )
                else:
                    souffle_runs.append(
                        souffle_run(
                            souffle_executable,
                            case_dir,
                            repetition,
                            args.souffle_jobs,
                            repo,
                            work_dir,
                            args.timeout_s,
                        )
                    )

            wcoj = summarize_runs(wcoj_runs, include_engine=True)
            binary = summarize_runs(binary_runs, include_engine=True)
            souffle = summarize_runs(souffle_runs, include_engine=False)
            wcoj_view = published_xlog_view(wcoj)
            binary_view = published_xlog_view(binary)
            souffle_view = published_souffle_view(souffle)
            wcoj_match = relations_match(wcoj, souffle)
            binary_match = relations_match(binary, souffle)
            wcoj_compute_s = (
                wcoj_view["compute_ms"] / 1000 if wcoj_view["compute_ms"] is not None else None
            )
            binary_compute_s = (
                binary_view["compute_ms"] / 1000
                if binary_view["compute_ms"] is not None
                else None
            )
            case_result = {
                "case": case_name,
                "edges": edge_count,
                "generator": {
                    "hubs": hubs,
                    "nodes": node_count,
                    "nodes_per_hub": args.nodes_per_hub,
                    "requested_hub_edge_fraction": args.hub_edge_fraction,
                    "realized_hub_incident_fraction": round(realized_fraction, 6),
                    "seed": case_seed,
                    "unique_directed_edges": len(edges),
                },
                "input": {
                    "arrow_sha256": sha256_file(arrow_input),
                    "souffle_facts_sha256": sha256_file(facts_input),
                    "edge_relation_sha256": relation_sha256(edges),
                },
                ARM_WCOJ: wcoj_view,
                ARM_BINARY: binary_view,
                "souffle": souffle_view,
                "counts_match": bool(
                    wcoj_match and (binary_match or not binary["complete"])
                ),
                "wcoj_vs_souffle_counts_match": bool(wcoj_match),
                "binary_vs_souffle_counts_match": bool(binary_match),
                "wcoj_vs_binary": ratio_or_note(binary_compute_s, wcoj_compute_s),
                "souffle_over_wcoj": ratio_or_note(souffle_view["compute_s"], wcoj_compute_s),
                "comparison_acceptable": bool(wcoj_match and binary_match),
                "detail": {
                    ARM_WCOJ: wcoj,
                    ARM_BINARY: binary,
                    "souffle": souffle,
                    "souffle_compile": souffle_compile,
                },
            }
            results.append(case_result)
            sys.stderr.write(
                f"END {case_name} wcoj_match={bool(wcoj_match)} "
                f"binary_match={bool(binary_match)}\n"
            )
            sys.stderr.flush()

        artifact = {
            "schema_version": 4,
            "benchmark": "triangle_wcoj_bulk_edb",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": (
                "edges bulk-loaded via --input Arrow IPC (EDB), NOT inline facts. "
                "xlog compute=total_ms; souffle compute=exe run."
            ),
            "repository": repo_info,
            "runner": {
                "path": runner_relative,
                "inside_repository": runner_inside_repo,
                "sha256": sha256_file(runner_path),
                "argv": sys.argv,
            },
            "hardware": hardware_info,
            "software": software_info,
            "protocol": {
                "query": (
                    "whole-triangle enumeration on both sides: xlog `rows` and Souffle "
                    "`count` are the same quantity and are gated on an identical "
                    "relation sha256, not only on equal cardinality"
                ),
                "graph_generator": (
                    "seeded unique directed edges; a proposal is hub-incident with "
                    "probability --hub-edge-fraction (source-hub and target-hub equally "
                    "likely), otherwise both endpoints are uniform over the non-hub nodes; "
                    "self loops rejected, duplicates absorbed, result sorted"
                ),
                "node_count": "hubs * --nodes-per-hub",
                "hub_edge_fraction": args.hub_edge_fraction,
                "nodes_per_hub": args.nodes_per_hub,
                "heavy_skew_companion_hub_edge_fraction": HEAVY_SKEW_HUB_EDGE_FRACTION,
                "edb_loading": "--input edge=<Arrow IPC stream file>; no inline facts",
                "repetitions": args.repetitions,
                "reported_time": (
                    "compute_ms is the XLOG engine total_ms from --stats-format json "
                    "(median over repetitions); compute_s is the Souffle executable's "
                    "full-process wall time from GNU time (median); wall_s is the "
                    "full-process wall time of the XLOG invocation"
                ),
                "ratios": (
                    "wcoj_vs_binary = xlog_binary.compute_ms / xlog_wcoj.compute_ms; "
                    "souffle_over_wcoj = souffle.compute_s / (xlog_wcoj.compute_ms / 1000)"
                ),
                "arm_routing": (
                    "route is taken from the --stats wcoj dispatch counters, never from "
                    "rc=0: xlog_wcoj requires a nonzero sum over "
                    f"{list(WCOJ_DISPATCH_COUNTERS)}, xlog_binary requires zero"
                ),
                "wcoj_env": describe_environment(ARM_WCOJ),
                "binary_env": describe_environment(ARM_BINARY),
                "wcoj_cli_flag": "--wcoj (xlog_wcoj arm only)",
                "souffle_mode": (
                    "standalone C++ executable generated once per case with --dl-program; "
                    "compilation is recorded separately and excluded from compute_s"
                ),
                "souffle_jobs": args.souffle_jobs,
                "process_memory": "maximum resident set size from GNU time",
                "xlog_memory": "provider allocation high-water from --stats-format json",
                "xlog_memory_budget_mb": args.memory_mb,
                "timeout_s": args.timeout_s,
                "seed": args.seed,
                "xlog_source_sha256": sha256_bytes(XLOG_SOURCE.encode()),
                "souffle_source_sha256": sha256_bytes(SOUFFLE_SOURCE.encode()),
                "unrecoverable_from_published_artifact": [
                    "generator parameters: the published file records only edge counts, "
                    "so hubs, nodes_per_hub and hub_edge_fraction here are a choice of "
                    "this runner, not a recovered protocol",
                    "the exact XLOG and Souffle rule text behind the published rows",
                    "how the published xlog_binary arm was pinned to binary joins; this "
                    "runner uses the documented kill switches because adaptive triangle "
                    "dispatch is default-on and omitting --wcoj is not sufficient",
                    "whether the published run enforced a per-repetition relation-hash "
                    "gate or only compared cardinalities",
                    "the published --memory-mb budget, souffle --jobs, repetition count "
                    "and seed",
                ],
            },
            "comparison_acceptable": all(
                result["comparison_acceptable"] for result in results
            ),
            "counts_comparison_acceptable": all(result["counts_match"] for result in results),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary_output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_output, args.output)
        sys.stderr.write(f"WROTE {args.output}\n")
        sys.stderr.flush()
        return 0 if artifact["comparison_acceptable"] else 1
    finally:
        if owned_work_dir and not args.keep_work_dir:
            shutil.rmtree(work_dir)
        else:
            sys.stderr.write(f"WORK_DIR {work_dir}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"benchmark runner failed: {error}\n")
        raise SystemExit(2) from error
