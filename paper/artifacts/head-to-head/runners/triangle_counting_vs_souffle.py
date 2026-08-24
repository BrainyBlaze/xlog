#!/usr/bin/env python3
"""Reproduce the skewed triangle-counting comparison against Souffle.

The runner generates each graph deterministically, writes the same edge relation
to Arrow IPC and Souffle facts, executes fused and enumerate-then-count XLOG
paths plus Souffle, validates dispatch and result counts, and writes one
self-contained JSON artifact. Command failures are recorded; they are never
replaced by another execution path.
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

import pyarrow as pa
import pyarrow.ipc as ipc


DEFAULT_CASES = (
    ("h30_e150000", 30, 150_000),
    ("h50_e300000", 50, 300_000),
    ("h80_e500000", 80, 500_000),
)
TIME_BIN = Path("/usr/bin/time")
XLOG_SOURCE = """\
pred edge(u32, u32).
pred triangle_count(u32, u64).

triangle_count(A, count(C)) :- edge(A, B), edge(B, C), edge(A, C).

?- triangle_count(A, N).
"""
SOUFFLE_SOURCE = """\
.decl edge(a:number, b:number)
.input edge
.decl triangle_count(a:number, n:number)
.output triangle_count

triangle_count(a, n) :- edge(a, _), n = count : {
    edge(a, b), edge(b, c), edge(a, c)
}, n > 0.
"""


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


def parse_time_metrics(
    lines: Iterable[str], observed_wall_s: float
) -> tuple[float, int]:
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
        metrics_path.read_text(encoding="utf-8").splitlines()
        if metrics_path.exists()
        else []
    )
    wall_s, max_rss_kb = parse_time_metrics(metric_lines, observed_wall_s)
    return CommandResult(tuple(argv), returncode, wall_s, max_rss_kb, stdout, stderr)


def generate_hub_skewed_edges(
    hubs: int, edge_count: int, seed: int
) -> list[tuple[int, int]]:
    """Generate exactly ``edge_count`` unique directed edges with reproducible hub skew."""
    node_count = hubs * 101
    if hubs <= 0 or node_count * (node_count - 1) < edge_count:
        raise ValueError("case does not have enough distinct non-self directed edges")
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < edge_count:
        selector = rng.randrange(10)
        if selector < 4:
            source = rng.randrange(node_count)
            target = rng.randrange(hubs)
        elif selector < 8:
            source = rng.randrange(hubs)
            target = rng.randrange(node_count)
        else:
            source = rng.randrange(node_count)
            target = rng.randrange(node_count)
        if source != target:
            edges.add((source, target))
    return sorted(edges)


def write_inputs(case_dir: Path, edges: Sequence[tuple[int, int]]) -> tuple[Path, Path]:
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


def relation_sha256(rows: Iterable[tuple[int, int]]) -> str:
    digest = hashlib.sha256()
    for left, right in sorted(rows):
        digest.update(f"{left}\t{right}\n".encode())
    return digest.hexdigest()


def triangle_count_summary(rows: Iterable[tuple[int, int]]) -> dict[str, Any]:
    canonical_rows = sorted(rows)
    roots = [root for root, _ in canonical_rows]
    if len(roots) != len(set(roots)):
        raise RuntimeError("triangle-count relation contains duplicate roots")
    if any(count <= 0 for _, count in canonical_rows):
        raise RuntimeError("triangle-count relation contains a non-positive count")
    return {
        "roots": len(canonical_rows),
        "total_triangles": sum(count for _, count in canonical_rows),
        "relation_sha256": relation_sha256(canonical_rows),
    }


def read_xlog_counts(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        table = ipc.open_stream(source).read_all()
    if table.num_columns != 2:
        raise RuntimeError(f"expected two XLOG output columns, got {table.num_columns}")
    keys = table.column(0).combine_chunks().to_pylist()
    counts = table.column(1).combine_chunks().to_pylist()
    if any(value is None for value in keys) or any(value is None for value in counts):
        raise RuntimeError("XLOG emitted a null triangle-count field")
    return triangle_count_summary(
        (int(key), int(count)) for key, count in zip(keys, counts)
    )


def read_souffle_counts(path: Path) -> dict[str, Any]:
    rows: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 2:
                raise RuntimeError(
                    f"expected two Souffle output columns, got {len(row)}"
                )
            rows.append((int(row[0]), int(row[1])))
    return triangle_count_summary(rows)


def relations_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        left.get("complete")
        and right.get("complete")
        and left.get("roots") == right.get("roots")
        and left.get("total_triangles") == right.get("total_triangles")
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


def xlog_run(
    xlog_bin: Path,
    source: Path,
    arrow_input: Path,
    output_dir: Path,
    memory_mb: int,
    enumerate_then_count: bool,
    repetition: int,
    repo: Path,
    work_dir: Path,
    timeout_s: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True)
    argv = (
        str(xlog_bin),
        "run",
        str(source),
        "--input",
        f"edge={arrow_input}",
        "--wcoj",
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
    env = os.environ.copy()
    if enumerate_then_count:
        env["XLOG_DISABLE_WCOJ_GROUPBY_FUSION"] = "1"
    else:
        env.pop("XLOG_DISABLE_WCOJ_GROUPBY_FUSION", None)
    result = run_timed(
        argv,
        repo,
        env,
        timeout_s,
        output_dir / f"time-{repetition}.txt",
    )
    record: dict[str, Any] = {
        "repetition": repetition,
        "command": normalized_command(argv, repo, work_dir),
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
        if enumerate_then_count:
            dispatch_valid = (
                wcoj.get("groupby_fusion_dispatch") == 0
                and wcoj.get("triangle_dispatch", 0) > 0
            )
        else:
            dispatch_valid = wcoj.get("groupby_fusion_dispatch", 0) > 0
        if not dispatch_valid:
            raise RuntimeError(f"unexpected WCOJ dispatch counters: {wcoj}")
        record.update(
            {
                "engine_total_ms": stats["total_ms"],
                "engine_peak_mb": stats["peak_memory_mb"],
                "engine_budget_mb": stats["budget_memory_mb"],
                "wcoj_dispatch": wcoj,
                **read_xlog_counts(output_dir / "query_0.arrow"),
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


def souffle_run(
    souffle_bin: Path,
    source: Path,
    case_dir: Path,
    repetition: int,
    repo: Path,
    work_dir: Path,
    timeout_s: int,
) -> dict[str, Any]:
    output_dir = case_dir / f"souffle-{repetition}"
    output_dir.mkdir(parents=True)
    argv = (
        str(souffle_bin),
        "-F",
        str(case_dir),
        "-D",
        str(output_dir),
        str(source),
    )
    result = run_timed(
        argv,
        repo,
        os.environ,
        timeout_s,
        output_dir / "time.txt",
    )
    record: dict[str, Any] = {
        "repetition": repetition,
        "command": normalized_command(argv, repo, work_dir),
        "returncode": result.returncode,
        "wall_s": round(result.wall_s, 6),
        "process_max_rss_kb": result.max_rss_kb,
    }
    if result.returncode != 0:
        record["error"] = error_record(result)
        return record
    try:
        record.update(read_souffle_counts(output_dir / "triangle_count.csv"))
    except (OSError, RuntimeError, ValueError) as error:
        record["returncode"] = 125
        record["error"] = {"kind": "protocol_violation", "diagnostic": str(error)}
    return record


def summarize_runs(
    runs: Sequence[dict[str, Any]], include_engine: bool
) -> dict[str, Any]:
    successful = [run for run in runs if run["returncode"] == 0]
    summary: dict[str, Any] = {
        "complete": len(successful) == len(runs),
        "successful_repetitions": len(successful),
        "requested_repetitions": len(runs),
        "runs": list(runs),
    }
    if not successful:
        return summary
    totals = {run["total_triangles"] for run in successful}
    roots = {run["roots"] for run in successful}
    relation_hashes = {run["relation_sha256"] for run in successful}
    if len(totals) != 1 or len(roots) != 1 or len(relation_hashes) != 1:
        summary["complete"] = False
        summary["consistency_error"] = (
            "successful repetitions produced different counts"
        )
        return summary
    summary.update(
        {
            "wall_s": round(median(run["wall_s"] for run in successful), 6),
            "process_max_rss_kb": max(run["process_max_rss_kb"] for run in successful),
            "roots": roots.pop(),
            "total_triangles": totals.pop(),
            "relation_sha256": relation_hashes.pop(),
        }
    )
    if include_engine:
        summary["compute_ms"] = median(run["engine_total_ms"] for run in successful)
        summary["peak_mb"] = max(run["engine_peak_mb"] for run in successful)
        summary["budget_mb"] = successful[0]["engine_budget_mb"]
    return summary


def command_version(argv: Sequence[str], cwd: Path) -> dict[str, Any]:
    try:
        return {"command": shlex.join(argv), "output": run_text(argv, cwd)}
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"version command failed: {shlex.join(argv)}: {error}"
        ) from error


def repository_state(repo: Path, allow_dirty: bool) -> dict[str, Any]:
    commit = run_text(("git", "rev-parse", "HEAD"), repo)
    dirty_lines = run_text(("git", "status", "--porcelain"), repo).splitlines()
    if dirty_lines and not allow_dirty:
        raise RuntimeError(
            "official benchmark requires a clean checkout; commit changes first"
        )
    return {
        "commit": commit,
        "dirty": bool(dirty_lines),
        "remote": run_text(("git", "remote", "get-url", "origin"), repo),
    }


def hardware_state(repo: Path) -> dict[str, Any]:
    gpu = run_text(
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        repo,
    ).splitlines()
    page_size = os.sysconf("SC_PAGE_SIZE")
    physical_pages = os.sysconf("SC_PHYS_PAGES")
    return {
        "gpu": gpu,
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "host_memory_bytes": page_size * physical_pages,
        "platform": platform.platform(),
    }


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_repo = script.parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=default_repo)
    parser.add_argument(
        "--xlog-bin", type=Path, default=default_repo / "target/release/xlog"
    )
    parser.add_argument("--souffle-bin", type=Path, default=Path("souffle"))
    parser.add_argument(
        "--output",
        type=Path,
        default=script.parent.parent / "triangle_counting_vs_souffle.json",
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
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    first = generate_hub_skewed_edges(3, 500, 7)
    second = generate_hub_skewed_edges(3, 500, 7)
    assert first == second
    assert len(first) == len(set(first)) == 500
    assert all(left != right for left, right in first)
    assert summarize_runs(
        [
            {
                "returncode": 0,
                "wall_s": 1.0,
                "process_max_rss_kb": 2,
                "roots": 3,
                "total_triangles": 4,
                "relation_sha256": "stable",
                "engine_total_ms": 5,
                "engine_peak_mb": 6,
                "engine_budget_mb": 7,
            }
        ],
        include_engine=True,
    )["complete"]
    expected = triangle_count_summary([(2, 3), (1, 4)])
    assert expected == triangle_count_summary([(1, 4), (2, 3)])
    assert relations_match(
        {"complete": True, **expected}, {"complete": True, **expected}
    )
    assert not relations_match(
        {"complete": True, **expected},
        {"complete": True, **triangle_count_summary([(1, 3), (2, 4)])},
    )
    assert parse_time_metrics(
        [
            "Command exited with non-zero status 1",
            "xlog_wall_s=1.25",
            "xlog_max_rss_kb=2048",
            "xlog_exit_status=1",
        ],
        9.0,
    ) == (1.25, 2048)
    for invalid_rows in ([(1, 2), (1, 3)], [(1, 0)]):
        try:
            triangle_count_summary(invalid_rows)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid relation was accepted: {invalid_rows}")
    print("triangle benchmark runner self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.repetitions <= 0 or args.memory_mb <= 0 or args.timeout_s <= 0:
        raise ValueError("repetitions, memory-mb, and timeout-s must be positive")
    repo = args.repo.resolve()
    xlog_bin = args.xlog_bin.resolve()
    souffle_bin = Path(
        shutil.which(str(args.souffle_bin)) or args.souffle_bin
    ).resolve()
    if not xlog_bin.is_file() or not os.access(xlog_bin, os.X_OK):
        raise RuntimeError(f"XLOG binary is not executable: {xlog_bin}")
    if not souffle_bin.is_file() or not os.access(souffle_bin, os.X_OK):
        raise RuntimeError(f"Souffle binary is not executable: {souffle_bin}")

    selected = set(args.case or (case[0] for case in DEFAULT_CASES))
    cases = [case for case in DEFAULT_CASES if case[0] in selected]
    owned_work_dir = args.work_dir is None
    work_dir = Path(
        tempfile.mkdtemp(
            prefix="xlog-triangle-souffle-",
            dir=str(args.work_dir.resolve()) if args.work_dir else None,
        )
    )
    try:
        repo_info = repository_state(repo, args.allow_dirty)
        runner_path = Path(__file__).resolve()
        results = []
        for case_name, hubs, edge_count in cases:
            print(f"BEGIN {case_name}", flush=True)
            case_dir = work_dir / case_name
            case_dir.mkdir()
            source = case_dir / "triangle_count.xlog"
            souffle_source = case_dir / "triangle_count.dl"
            source.write_text(XLOG_SOURCE, encoding="utf-8")
            souffle_source.write_text(SOUFFLE_SOURCE, encoding="utf-8")
            case_seed = args.seed + hubs * 1_000_003 + edge_count
            edges = generate_hub_skewed_edges(hubs, edge_count, case_seed)
            arrow_input, facts_input = write_inputs(case_dir, edges)

            fused_runs = []
            enum_runs = []
            souffle_runs = []
            for repetition in range(1, args.repetitions + 1):
                fused_runs.append(
                    xlog_run(
                        xlog_bin,
                        source,
                        arrow_input,
                        case_dir / f"xlog-fused-{repetition}",
                        args.memory_mb,
                        False,
                        repetition,
                        repo,
                        work_dir,
                        args.timeout_s,
                    )
                )
                enum_runs.append(
                    xlog_run(
                        xlog_bin,
                        source,
                        arrow_input,
                        case_dir / f"xlog-enum-{repetition}",
                        args.memory_mb,
                        True,
                        repetition,
                        repo,
                        work_dir,
                        args.timeout_s,
                    )
                )
                souffle_runs.append(
                    souffle_run(
                        souffle_bin,
                        souffle_source,
                        case_dir,
                        repetition,
                        repo,
                        work_dir,
                        args.timeout_s,
                    )
                )

            fused = summarize_runs(fused_runs, include_engine=True)
            enumerated = summarize_runs(enum_runs, include_engine=True)
            souffle = summarize_runs(souffle_runs, include_engine=False)
            fused_match = relations_match(fused, souffle)
            enum_match = relations_match(enumerated, souffle)
            case_result = {
                "case": case_name,
                "hubs": hubs,
                "nodes": hubs * 101,
                "edges": edge_count,
                "seed": case_seed,
                "input": {
                    "arrow_sha256": sha256_file(arrow_input),
                    "souffle_facts_sha256": sha256_file(facts_input),
                    "edge_relation_sha256": relation_sha256(edges),
                },
                "fused_wcoj": fused,
                "enum_then_count": enumerated,
                "souffle": souffle,
                "fused_vs_souffle_counts_match": bool(fused_match),
                "enum_vs_souffle_counts_match": bool(enum_match),
                "comparison_acceptable": bool(fused_match and enum_match),
            }
            if fused_match and fused.get("wall_s", 0) > 0:
                case_result["souffle_over_fused"] = round(
                    souffle["wall_s"] / fused["wall_s"], 2
                )
            if enum_match and fused.get("wall_s", 0) > 0:
                case_result["enum_over_fused"] = round(
                    enumerated["wall_s"] / fused["wall_s"], 2
                )
            results.append(case_result)
            print(
                f"END {case_name} fused_match={bool(fused_match)} "
                f"enum_match={bool(enum_match)}",
                flush=True,
            )

        artifact = {
            "schema_version": 2,
            "benchmark": "per_root_triangle_counting",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository": repo_info,
            "runner": {
                "path": str(runner_path.relative_to(repo)),
                "sha256": sha256_file(runner_path),
                "argv": sys.argv,
            },
            "hardware": hardware_state(repo),
            "software": {
                "python": platform.python_version(),
                "pyarrow": pa.__version__,
                "xlog": {
                    "path": normalized_command((str(xlog_bin),), repo, work_dir),
                    "sha256": sha256_file(xlog_bin),
                    "version": command_version((str(xlog_bin), "--version"), repo),
                },
                "souffle": {
                    "path": str(souffle_bin),
                    "sha256": sha256_file(souffle_bin),
                    "version": command_version((str(souffle_bin), "--version"), repo),
                },
                "nvcc": command_version(("nvcc", "--version"), repo),
            },
            "protocol": {
                "graph_generator": (
                    "seeded unique directed edges; each proposal has at least one hub "
                    "endpoint with 80 percent probability"
                ),
                "node_count": "101 * hubs",
                "repetitions": args.repetitions,
                "reported_time": "median full-process wall time from GNU time",
                "process_memory": "maximum resident set size from GNU time",
                "xlog_memory": "provider allocation high-water from --stats-format json",
                "xlog_memory_budget_mb": args.memory_mb,
                "timeout_s": args.timeout_s,
                "fused_env": {"XLOG_DISABLE_WCOJ_GROUPBY_FUSION": "unset"},
                "enumerate_env": {"XLOG_DISABLE_WCOJ_GROUPBY_FUSION": "1"},
                "xlog_source_sha256": sha256_bytes(XLOG_SOURCE.encode()),
                "souffle_source_sha256": sha256_bytes(SOUFFLE_SOURCE.encode()),
            },
            "core_comparison_acceptable": all(
                result["fused_vs_souffle_counts_match"] for result in results
            ),
            "comparison_acceptable": all(
                result["comparison_acceptable"] for result in results
            ),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary_output.write_text(
            json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_output, args.output)
        print(f"WROTE {args.output}", flush=True)
        return 0 if artifact["comparison_acceptable"] else 1
    finally:
        if owned_work_dir and not args.keep_work_dir:
            shutil.rmtree(work_dir)
        else:
            print(f"WORK_DIR {work_dir}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"benchmark runner failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
