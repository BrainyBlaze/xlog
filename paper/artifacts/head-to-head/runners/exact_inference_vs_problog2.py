#!/usr/bin/env python3
"""Reproduce the exact-probabilistic-inference comparison against ProbLog2.

Five programs (a conditioned wet/sprinkler net and ``reach_chain_{5,10,15,20}``)
are run through ``pyxlog.Program.compile`` + ``evaluate`` end-to-end and through
ProbLog2 on the matched programs. Every repetition is kept; the reported number
is the median. Query probabilities are gated against the analytic answer. A
failed arm is recorded as a failure and is never replaced by another execution
path; ``comparison_acceptable`` then becomes ``false``.

The artifact carries its own provenance: commit and tree cleanliness, the sha256
of this runner, the GPU/CPU/cgroup description of the host, and the versions of
every component that took part.

Imports beyond the standard library, and why each one is here:

* ``pyxlog`` -- the engine under measurement (probabilistic arm).
* ``torch`` -- ``EvalResult.prob`` is a DLPack capsule; ``torch.utils.dlpack``
  is how the in-repo example ``examples/python/02_prob_wet_conditioning_torch.py``
  reads it, so the same path is used here.
* ``problog`` -- the reference engine, when ``--problog-mode module`` (default).

None of the three is imported by ``--self-test``: that mode exercises only the
pure functions (generators, parsers, formatters, the median) and runs on a
machine without CUDA, without ProbLog and without pyxlog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
BENCHMARK = "probabilistic_exact_xlog_vs_problog"
NOTE = (
    "xlog GPU-native verified KC vs ProbLog2 CPU (D4/c2d). "
    "Timing=full inference (compile+evaluate), median of --repetitions. "
    "Correctness gate: probs match analytic expected within --tolerance."
)

# Byte-for-byte the committed example, modulo line endings; --self-test proves it.
WET_EXAMPLE_RELATIVE = Path("examples/prob/01-wet-conditioning.xlog")
WET_XLOG_SOURCE = """0.7::rain().
0.2::sprinkler().

wet() :- rain().
wet() :- sprinkler().

evidence(wet(), true).

query(rain()).
query(sprinkler()).
"""
WET_PROBLOG_SOURCE = """0.7::rain.
0.2::sprinkler.

wet :- rain.
wet :- sprinkler.

evidence(wet,true).

query(rain).
query(sprinkler).
"""
CHAIN_SIZES = (5, 10, 15, 20)
PROB_DECIMALS = 6
TIME_DECIMALS = 4

FACT_PATTERN = re.compile(
    r"^\s*(?P<prob>[0-9][0-9eE.+-]*)\s*::\s*"
    r"(?P<name>[a-z_][A-Za-z0-9_]*)\s*(?P<args>\([^()]*\))?\s*\.\s*$"
)
QUERY_PATTERN = re.compile(r"^\s*query\s*\(\s*(?P<atom>.+?)\s*\)\s*\.\s*$")
PROBLOG_OUTPUT_PATTERN = re.compile(
    r"^\s*(?P<term>\S.*?)\s*:\s*(?P<prob>[-+0-9][0-9eE.+-]*)\s*$"
)


@dataclass(frozen=True)
class Query:
    """One query of a case: its xlog atom, its ProbLog term, its analytic value."""

    xlog_atom: str
    problog_term: str
    expected: float


@dataclass(frozen=True)
class Case:
    name: str
    xlog_source: str
    problog_source: str
    queries: tuple[Query, ...]


# ---------------------------------------------------------------------------
# Program generation
# ---------------------------------------------------------------------------


def chain_xlog_source(n: int) -> str:
    """xlog text for ``reach_chain_n``: nodes 1..n, one 0.5 edge between neighbours."""
    lines = [f"0.5::edge({i},{i + 1})." for i in range(1, n)]
    lines.append("")
    lines.append("reach(X,Y) :- edge(X,Y).")
    lines.append("reach(X,Y) :- edge(X,Z), reach(Z,Y).")
    lines.append("")
    lines.append(f"query(reach(1,{n})).")
    return "\n".join(lines) + "\n"


def chain_problog_source(n: int) -> str:
    """ProbLog text for the same chain, written out independently of the xlog one."""
    lines = [f"0.5::edge({i},{i + 1})." for i in range(1, n)]
    lines.append("")
    lines.append("reach(X,Y) :- edge(X,Y).")
    lines.append("reach(X,Y) :- edge(X,Z), reach(Z,Y).")
    lines.append("")
    lines.append(f"query(reach(1,{n})).")
    return "\n".join(lines) + "\n"


def chain_expected(n: int) -> float:
    """P(reach(1,n)) over a chain of n-1 independent edges, each present with p=0.5."""
    return 0.5 ** (n - 1)


def chain_case(n: int) -> Case:
    return Case(
        name=f"reach_chain_{n}",
        xlog_source=chain_xlog_source(n),
        problog_source=chain_problog_source(n),
        queries=(Query(f"reach(1,{n})", f"reach(1,{n})", chain_expected(n)),),
    )


def wet_case() -> Case:
    evidence = 1.0 - 0.3 * 0.8
    return Case(
        name="wet_conditioning",
        xlog_source=WET_XLOG_SOURCE,
        problog_source=WET_PROBLOG_SOURCE,
        queries=(
            Query("rain()", "rain", 0.7 / evidence),
            Query("sprinkler()", "sprinkler", 0.2 / evidence),
        ),
    )


def all_cases() -> tuple[Case, ...]:
    return (wet_case(), *(chain_case(n) for n in CHAIN_SIZES))


CASE_NAMES = ("wet_conditioning", *(f"reach_chain_{n}" for n in CHAIN_SIZES))


# ---------------------------------------------------------------------------
# Pure helpers: parsing, normalization, statistics
# ---------------------------------------------------------------------------


def normalize_newlines(text: str) -> str:
    """CRLF/CR -> LF and no trailing blank lines; nothing else is touched."""
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def normalize_atom(atom: str) -> str:
    """Compare atoms without whitespace and without empty argument lists.

    ``reach(1, 5)`` (xlog output) and ``reach(1,5)`` (ProbLog output) are the same
    atom, and so are ``rain()`` (xlog) and ``rain`` (ProbLog).
    """
    compact = "".join(atom.split())
    while compact.endswith("()"):
        compact = compact[:-2]
    return compact


def parse_probabilistic_facts(text: str) -> list[tuple[float, str, tuple[str, ...]]]:
    """Extract ``p::name(args).`` facts from either an xlog or a ProbLog program."""
    facts: list[tuple[float, str, tuple[str, ...]]] = []
    for line in text.splitlines():
        match = FACT_PATTERN.match(line)
        if match is None:
            continue
        raw_args = match.group("args")
        inner = "" if raw_args is None else raw_args[1:-1].strip()
        args = tuple(part.strip() for part in inner.split(",")) if inner else ()
        facts.append((float(match.group("prob")), match.group("name"), args))
    return facts


def parse_query_atoms(text: str) -> list[str]:
    """Extract the normalized atom of every ``query(...)`` line of a program."""
    atoms: list[str] = []
    for line in text.splitlines():
        match = QUERY_PATTERN.match(line)
        if match is not None:
            atoms.append(normalize_atom(match.group("atom")))
    return atoms


def parse_problog_output(stdout: str) -> dict[str, float]:
    """Parse the ``term : probability`` block the ProbLog CLI writes to stdout."""
    probabilities: dict[str, float] = {}
    for line in stdout.splitlines():
        match = PROBLOG_OUTPUT_PATTERN.match(line)
        if match is None:
            continue
        term = match.group("term").strip()
        probabilities[term] = float(match.group("prob"))
    if not probabilities:
        raise RuntimeError("ProbLog produced no 'term : probability' lines")
    return probabilities


def median_seconds(times: Sequence[float]) -> float:
    """Median (not mean) of the per-repetition wall times, rounded for the artifact."""
    return round(median(times), TIME_DECIMALS)


def round_probability(value: float) -> float:
    return round(value, PROB_DECIMALS)


def max_abs_error(
    probabilities: Mapping[str, float], queries: Sequence[Query], problog: bool
) -> float:
    """Largest |p - analytic| over the queries; raises if a query atom is missing."""
    normalized = {normalize_atom(key): value for key, value in probabilities.items()}
    if len(normalized) != len(probabilities):
        raise RuntimeError(
            f"query atoms collide after normalization: {list(probabilities)}"
        )
    expected_keys = {
        normalize_atom(
            query.problog_term if problog else query.xlog_atom
        ): query.expected
        for query in queries
    }
    missing = sorted(set(expected_keys) - set(normalized))
    unexpected = sorted(set(normalized) - set(expected_keys))
    if missing or unexpected:
        raise RuntimeError(
            "query atoms do not match the program: "
            f"missing={missing} unexpected={unexpected}"
        )
    return max(abs(normalized[key] - value) for key, value in expected_keys.items())


def probabilities_are_stable(runs: Sequence[Mapping[str, float]]) -> bool:
    """True when every repetition returned the same probabilities (to 1e-12)."""
    if not runs:
        return False
    first = {normalize_atom(key): value for key, value in runs[0].items()}
    for run in runs[1:]:
        current = {normalize_atom(key): value for key, value in run.items()}
        if set(current) != set(first):
            return False
        if any(abs(current[key] - first[key]) > 1e-12 for key in first):
            return False
    return True


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def failure_record(kind: str, diagnostic: str, **extra: Any) -> dict[str, Any]:
    """The single shape every recorded failure takes. Nothing is ever swallowed."""
    record: dict[str, Any] = {"kind": kind, "diagnostic": diagnostic}
    record.update(extra)
    return record


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def probe_command(
    argv: Sequence[str], cwd: Path, timeout_s: int = 120
) -> dict[str, Any]:
    """Run a provenance command. An unavailable command is recorded, never skipped."""
    record: dict[str, Any] = {"command": shlex.join(str(item) for item in argv)}
    try:
        completed = subprocess.run(
            [str(item) for item in argv],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as error:
        record["status"] = "UNAVAILABLE"
        record["diagnostic"] = f"{type(error).__name__}: {error}"
        return record
    if completed.returncode != 0:
        record["status"] = "UNAVAILABLE"
        record["returncode"] = completed.returncode
        record["stderr"] = (completed.stderr or "").strip()
        record["stdout"] = (completed.stdout or "").strip()
        return record
    record["status"] = "ok"
    record["output"] = (completed.stdout or completed.stderr).strip()
    return record


def probe_file(path: Path) -> dict[str, Any]:
    """Read a provenance file. Absence and failure are two different recorded facts."""
    record: dict[str, Any] = {"path": str(path)}
    try:
        record["raw"] = path.read_text(encoding="utf-8").strip()
        record["status"] = "ok"
    except FileNotFoundError:
        record["status"] = "absent"
    except OSError as error:
        record["status"] = "UNAVAILABLE"
        record["diagnostic"] = f"{type(error).__name__}: {error}"
    return record


def parse_cpu_quota_cores(quota: str, period: str) -> float | None:
    """cgroup quota in cores; ``None`` when the group is unlimited."""
    if quota in {"max", "-1"}:
        return None
    quota_value = int(quota)
    period_value = int(period)
    if quota_value <= 0 or period_value <= 0:
        raise RuntimeError(
            f"invalid cgroup CPU quota: quota={quota_value} period={period_value}"
        )
    return round(quota_value / period_value, 6)


def cgroup_cpu_quota() -> dict[str, Any]:
    """Both cgroup generations, each reported on its own, plus the effective value."""
    v2 = probe_file(Path("/sys/fs/cgroup/cpu.max"))
    if v2["status"] == "ok":
        parts = v2["raw"].split()
        if len(parts) == 2:
            try:
                v2["cores"] = parse_cpu_quota_cores(parts[0], parts[1])
            except (RuntimeError, ValueError) as error:
                v2["status"] = "UNAVAILABLE"
                v2["diagnostic"] = str(error)
        else:
            v2["status"] = "UNAVAILABLE"
            v2["diagnostic"] = f"invalid cgroup v2 cpu.max: {parts}"

    quota = probe_file(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period = probe_file(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    v1: dict[str, Any] = {"quota": quota, "period": period}
    if quota["status"] == "ok" and period["status"] == "ok":
        try:
            v1["cores"] = parse_cpu_quota_cores(quota["raw"], period["raw"])
            v1["status"] = "ok"
        except (RuntimeError, ValueError) as error:
            v1["status"] = "UNAVAILABLE"
            v1["diagnostic"] = str(error)
    else:
        v1["status"] = quota["status"] if quota["status"] != "ok" else period["status"]

    effective: float | None = None
    if v2.get("status") == "ok":
        effective = v2.get("cores")
    elif v1.get("status") == "ok":
        effective = v1.get("cores")
    return {"cgroup_v2": v2, "cgroup_v1": v1, "cores": effective}


def cpu_model_name(cpuinfo: str) -> str | None:
    """The CPU model line, or ``None`` -- never a guess."""
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"model name", "Hardware", "Processor"}:
            model = value.strip()
            if model:
                return model
    return None


def host_memory_bytes() -> dict[str, Any]:
    if not hasattr(os, "sysconf"):
        return {"status": "UNAVAILABLE", "diagnostic": "os.sysconf is not available"}
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError) as error:
        return {
            "status": "UNAVAILABLE",
            "diagnostic": f"{type(error).__name__}: {error}",
        }
    return {"status": "ok", "bytes": page_size * physical_pages}


def hardware_state(cwd: Path) -> dict[str, Any]:
    """Everything host-dependent is recorded here. Nothing here is ever asserted on."""
    cpuinfo = probe_file(Path("/proc/cpuinfo"))
    cpu: dict[str, Any] = {"source": cpuinfo["path"], "status": cpuinfo["status"]}
    if cpuinfo["status"] == "ok":
        model = cpu_model_name(cpuinfo["raw"])
        cpu["model"] = model
        if model is None:
            cpu["status"] = "UNAVAILABLE"
            cpu["diagnostic"] = "/proc/cpuinfo does not identify the CPU model"
    else:
        cpu["diagnostic"] = cpuinfo.get("diagnostic", "not present on this platform")
        cpu["model"] = platform.processor() or None
    quota = cgroup_cpu_quota()
    return {
        "gpu": probe_command(
            (
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ),
            cwd,
        ),
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "cpu_quota_cores": quota["cores"],
        "cpu_quota_sources": {
            "cgroup_v2": quota["cgroup_v2"],
            "cgroup_v1": quota["cgroup_v1"],
        },
        "host_memory": host_memory_bytes(),
        "platform": platform.platform(),
    }


def repository_state(repo: Path, allow_dirty: bool) -> dict[str, Any]:
    """Refuse a dirty checkout unless --allow-dirty; record what git said either way."""
    commit = probe_command(("git", "rev-parse", "HEAD"), repo)
    status = probe_command(("git", "status", "--porcelain"), repo)
    remote = probe_command(("git", "remote", "get-url", "origin"), repo)
    if status["status"] != "ok":
        if not allow_dirty:
            diagnostic = status.get("diagnostic") or status.get("stderr")
            raise RuntimeError(f"cannot verify a clean checkout: {diagnostic}")
        dirty: bool | None = None
        dirty_entries: list[str] = []
    else:
        dirty_entries = [line for line in status["output"].splitlines() if line.strip()]
        dirty = bool(dirty_entries)
        if dirty and not allow_dirty:
            raise RuntimeError(
                "official benchmark requires a clean checkout; commit changes first "
                "or pass --allow-dirty"
            )
    return {
        "path": str(repo),
        "commit": commit.get("output") if commit["status"] == "ok" else None,
        "commit_probe": commit,
        "dirty": dirty,
        "dirty_entries": dirty_entries,
        "allow_dirty": allow_dirty,
        "remote": remote.get("output") if remote["status"] == "ok" else None,
        "remote_probe": remote,
    }


def runner_state(repo: Path) -> dict[str, Any]:
    runner_path = Path(__file__).resolve()
    try:
        relative: str | None = str(runner_path.relative_to(repo))
    except ValueError:
        relative = None
    return {
        "path": relative or str(runner_path),
        "absolute_path": str(runner_path),
        "sha256": sha256_file(runner_path),
        "argv": list(sys.argv),
    }


# ---------------------------------------------------------------------------
# Measurement arms
# ---------------------------------------------------------------------------


def evaluate_xlog_once(
    source: str, device: int, memory_mb: int
) -> tuple[float, dict[str, float]]:
    """One end-to-end compile+evaluate -> (wall seconds, probabilities by atom)."""
    import pyxlog
    import torch

    started = time.perf_counter()
    program = pyxlog.Program.compile(source, device=device, memory_mb=memory_mb)
    result = program.evaluate(return_grads=False)
    probabilities = [
        float(value)
        for value in torch.utils.dlpack.from_dlpack(result.prob)
        .cpu()
        .reshape(-1)
        .tolist()
    ]
    elapsed = time.perf_counter() - started
    atoms = list(result.atoms)
    if len(atoms) != len(probabilities):
        raise RuntimeError(
            f"xlog returned {len(atoms)} atoms and {len(probabilities)} probabilities"
        )
    return elapsed, dict(zip(atoms, probabilities, strict=True))


def run_xlog_arm(
    case: Case, repetitions: int, device: int, memory_mb: int
) -> dict[str, Any]:
    times: list[float] = []
    runs: list[dict[str, float]] = []
    record: dict[str, Any] = {"repetitions_requested": repetitions}
    for repetition in range(1, repetitions + 1):
        try:
            elapsed, probabilities = evaluate_xlog_once(
                case.xlog_source, device, memory_mb
            )
        except Exception as error:  # noqa: BLE001 - the engine may raise anything
            record["times"] = [round(value, TIME_DECIMALS) for value in times]
            record["error"] = failure_record(
                type(error).__name__,
                f"repetition {repetition} failed: {error}",
                repetition=repetition,
            )
            return record
        times.append(elapsed)
        runs.append(probabilities)
    record["times"] = [round(value, TIME_DECIMALS) for value in times]
    record["times_raw"] = times
    record["median_sec"] = median_seconds(times)
    if not probabilities_are_stable(runs):
        record["error"] = failure_record(
            "unstable_result", "repetitions returned different probabilities", runs=runs
        )
        return record
    atoms = list(runs[0])
    record["atoms"] = atoms
    record["probs"] = [round_probability(runs[0][atom]) for atom in atoms]
    record["probs_raw"] = [runs[0][atom] for atom in atoms]
    return record


def evaluate_problog_module_once(source: str) -> tuple[float, dict[str, float]]:
    """One in-process ProbLog2 inference (knowledge compilation + evaluation)."""
    from problog import get_evaluatable
    from problog.program import PrologString

    started = time.perf_counter()
    evaluated = get_evaluatable().create_from(PrologString(source)).evaluate()
    elapsed = time.perf_counter() - started
    return elapsed, {str(term): float(value) for term, value in evaluated.items()}


def evaluate_problog_binary_once(
    problog_bin: Path, source: str, work_dir: Path, repetition: int, timeout_s: int
) -> tuple[float, dict[str, float]]:
    """One ProbLog2 inference through the CLI. Includes interpreter startup."""
    program_path = work_dir / f"program-{repetition}.pl"
    program_path.write_text(source, encoding="utf-8", newline="\n")
    argv = (str(problog_bin), str(program_path))
    started = time.perf_counter()
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout_s, cwd=str(work_dir)
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"{shlex.join(argv)} exited with {completed.returncode}: "
            f"{(completed.stderr or completed.stdout).strip()[-2000:]}"
        )
    return elapsed, parse_problog_output(completed.stdout)


def run_problog_arm(
    case: Case,
    repetitions: int,
    mode: str,
    problog_bin: Path,
    work_dir: Path,
    timeout_s: int,
) -> dict[str, Any]:
    times: list[float] = []
    runs: list[dict[str, float]] = []
    record: dict[str, Any] = {"repetitions_requested": repetitions, "mode": mode}
    for repetition in range(1, repetitions + 1):
        try:
            if mode == "module":
                elapsed, probabilities = evaluate_problog_module_once(
                    case.problog_source
                )
            else:
                elapsed, probabilities = evaluate_problog_binary_once(
                    problog_bin, case.problog_source, work_dir, repetition, timeout_s
                )
        except Exception as error:  # noqa: BLE001 - ProbLog may raise anything
            record["times"] = [round(value, TIME_DECIMALS) for value in times]
            record["error"] = failure_record(
                type(error).__name__,
                f"repetition {repetition} failed: {error}",
                repetition=repetition,
            )
            return record
        times.append(elapsed)
        runs.append(probabilities)
    record["times"] = [round(value, TIME_DECIMALS) for value in times]
    record["times_raw"] = times
    record["median_sec"] = median_seconds(times)
    if not probabilities_are_stable(runs):
        record["error"] = failure_record(
            "unstable_result", "repetitions returned different probabilities", runs=runs
        )
        return record
    record["probs"] = {
        term: round_probability(value) for term, value in runs[0].items()
    }
    record["probs_raw"] = dict(runs[0])
    return record


def gate_arm(
    record: dict[str, Any], queries: Sequence[Query], problog: bool, tolerance: float
) -> tuple[dict[str, Any], float | None, bool]:
    """Compute max|p - analytic| from the raw probabilities and apply the gate."""
    if "error" in record:
        return record, None, False
    raw = record["probs_raw"]
    probabilities = (
        raw if isinstance(raw, dict) else dict(zip(record["atoms"], raw, strict=True))
    )
    try:
        error_value = max_abs_error(probabilities, queries, problog=problog)
    except RuntimeError as error:
        record["error"] = failure_record("protocol_violation", str(error))
        return record, None, False
    return record, round(error_value, 12), error_value <= tolerance


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test(repo: Path) -> None:
    """Check the pure functions. No GPU, no pyxlog, no ProbLog, no external binaries."""
    # 1. Chain generator: exactly n-1 edge facts and one query reach(1,n).
    for size in (2, 5, 10, 15, 20):
        xlog_text = chain_xlog_source(size)
        problog_text = chain_problog_source(size)
        xlog_facts = parse_probabilistic_facts(xlog_text)
        assert len(xlog_facts) == size - 1, (size, len(xlog_facts))
        assert [fact[1] for fact in xlog_facts] == ["edge"] * (size - 1)
        assert [fact[2] for fact in xlog_facts] == [
            (str(i), str(i + 1)) for i in range(1, size)
        ]
        assert all(fact[0] == 0.5 for fact in xlog_facts)
        assert parse_query_atoms(xlog_text) == [f"reach(1,{size})"]
        assert parse_query_atoms(problog_text) == [f"reach(1,{size})"]
        # 3. Both texts describe the same probabilities and the same edges.
        assert xlog_facts == parse_probabilistic_facts(problog_text), size
        # 2. Analytic value of the chain.
        assert chain_expected(size) == 0.5 ** (size - 1)
    assert chain_expected(5) == 0.0625
    assert chain_expected(10) == 0.001953125
    assert abs(chain_expected(15) - 0.000061035) < 1e-9
    assert abs(chain_expected(20) - 0.0000019073) < 1e-10
    assert round_probability(chain_expected(15)) == 6.1e-05
    assert round_probability(chain_expected(20)) == 2e-06

    # 3 (wet). The xlog and ProbLog wet programs carry the same probabilistic facts.
    wet = wet_case()
    xlog_wet_facts = parse_probabilistic_facts(wet.xlog_source)
    problog_wet_facts = parse_probabilistic_facts(wet.problog_source)
    assert xlog_wet_facts == [(0.7, "rain", ()), (0.2, "sprinkler", ())], xlog_wet_facts
    assert xlog_wet_facts == problog_wet_facts
    assert parse_query_atoms(wet.xlog_source) == ["rain", "sprinkler"]
    assert parse_query_atoms(wet.xlog_source) == parse_query_atoms(wet.problog_source)
    evidence = 1.0 - 0.3 * 0.8
    assert abs(evidence - 0.76) < 1e-12
    assert round_probability(0.7 / evidence) == 0.921053
    assert round_probability(0.2 / evidence) == 0.263158

    # 4. The embedded wet program is the committed example, modulo line endings.
    example = repo / WET_EXAMPLE_RELATIVE
    if not example.is_file():
        raise RuntimeError(
            f"cannot verify the wet program: {example} is missing; "
            "pass --repo <checkout>"
        )
    on_disk = normalize_newlines(example.read_text(encoding="utf-8"))
    embedded = normalize_newlines(WET_XLOG_SOURCE)
    if on_disk != embedded:
        raise AssertionError(
            "embedded wet program differs from "
            f"{WET_EXAMPLE_RELATIVE.as_posix()}:\n"
            f"--- on disk ---\n{on_disk}\n--- embedded ---\n{embedded}"
        )
    assert sha256_bytes(on_disk.encode()) == sha256_bytes(embedded.encode())

    # 5. The median is a median, not a mean.
    assert median_seconds([1.0, 2.0, 6.0]) == 2.0
    assert median_seconds([1.0, 2.0, 6.0]) != round(9.0 / 3.0, TIME_DECIMALS)
    assert median_seconds([0.5268, 0.2547, 0.2521]) == 0.2547
    assert median_seconds([2.0, 1.0]) == 1.5

    # Atom normalization across the two engines' output conventions.
    assert normalize_atom("reach(1, 5)") == "reach(1,5)"
    assert normalize_atom("rain()") == "rain"
    assert normalize_atom("  sprinkler ") == "sprinkler"

    # ProbLog CLI output parsing.
    parsed = parse_problog_output("     rain : 0.92105263\nsprinkler : 0.26315789\n")
    assert parsed == {"rain": 0.92105263, "sprinkler": 0.26315789}, parsed
    assert parse_problog_output("reach(1,20) : 1.9073486e-06") == {
        "reach(1,20)": 1.9073486e-06
    }
    try:
        parse_problog_output("no probabilities here\n")
    except RuntimeError:
        pass
    else:
        raise AssertionError("empty ProbLog output was accepted")

    # The correctness gate itself.
    chain = chain_case(5)
    assert max_abs_error({"reach(1, 5)": 0.0625}, chain.queries, problog=False) == 0.0
    assert (
        abs(max_abs_error({"reach(1,5)": 0.0626}, chain.queries, problog=True) - 1e-4)
        < 1e-12
    )
    assert (
        max_abs_error(
            {"rain()": 0.921053, "sprinkler()": 0.263158}, wet.queries, problog=False
        )
        < 1e-6
    )
    for broken in ({"reach(1,4)": 0.0625}, {"reach(1,5)": 0.0625, "reach(1,4)": 0.5}):
        try:
            max_abs_error(broken, chain.queries, problog=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"mismatched query set was accepted: {broken}")

    # Repetition stability.
    assert probabilities_are_stable([{"rain()": 0.5}, {"rain()": 0.5}])
    assert probabilities_are_stable([{"reach(1, 5)": 0.5}, {"reach(1,5)": 0.5}])
    assert not probabilities_are_stable([{"rain()": 0.5}, {"rain()": 0.6}])
    assert not probabilities_are_stable([{"rain()": 0.5}, {"sprinkler()": 0.5}])
    assert not probabilities_are_stable([])

    # cgroup quota parsing, both generations.
    assert parse_cpu_quota_cores("765000", "100000") == 7.65
    assert parse_cpu_quota_cores("max", "100000") is None
    assert parse_cpu_quota_cores("-1", "100000") is None
    try:
        parse_cpu_quota_cores("0", "100000")
    except RuntimeError:
        pass
    else:
        raise AssertionError("a zero CPU quota was accepted")

    # CPU model extraction returns None instead of guessing.
    assert cpu_model_name("processor: 0\nmodel name: Example CPU\n") == "Example CPU"
    assert cpu_model_name("processor: 0\n") is None

    # Case selection covers exactly the five published programs.
    assert [case.name for case in all_cases()] == list(CASE_NAMES)
    assert CASE_NAMES == (
        "wet_conditioning",
        "reach_chain_5",
        "reach_chain_10",
        "reach_chain_15",
        "reach_chain_20",
    )

    # An unavailable command is recorded, not skipped.
    probe = probe_command(("xlog-self-test-missing-binary",), repo)
    assert probe["status"] == "UNAVAILABLE", probe
    assert probe["diagnostic"]
    absent = probe_file(Path("/xlog-self-test-missing-file"))
    assert absent["status"] == "absent", absent

    sys.stdout.write("exact-inference benchmark runner self-test passed\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    # paper/artifacts/head-to-head/runners/<this file> -> repository root.
    default_repo = script.parents[4] if len(script.parents) > 4 else script.parent
    parser = argparse.ArgumentParser(description="xlog vs ProbLog2 exact inference")
    parser.add_argument("--repo", type=Path, default=default_repo)
    parser.add_argument(
        "--output",
        type=Path,
        default=script.parent.parent / "exact_inference_vs_problog2.json",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--problog-bin", type=Path, default=Path("problog"))
    parser.add_argument(
        "--problog-mode",
        choices=("module", "binary"),
        default="module",
        help=(
            "module: in-process problog API, timing = knowledge compilation + "
            "evaluation; binary: the problog CLI, timing includes interpreter startup"
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=list(CASE_NAMES),
        help="run only the selected case; repeat for multiple cases",
    )
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--memory-mb", type=int, default=32768)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def software_state(args: argparse.Namespace, repo: Path) -> dict[str, Any]:
    import pyxlog
    import torch

    software: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pyxlog": {
            "version": getattr(pyxlog, "__version__", None),
            "file": getattr(pyxlog, "__file__", None),
        },
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
        "nvidia_smi": probe_command(("nvidia-smi", "--version"), repo),
    }
    try:
        import problog
        from problog import get_evaluatable

        problog_module: dict[str, Any] = {
            "status": "ok",
            "version": getattr(getattr(problog, "version", None), "version", None)
            or getattr(problog, "__version__", None),
            "file": getattr(problog, "__file__", None),
            "evaluatable": type(get_evaluatable()).__name__,
        }
    except (
        Exception
    ) as error:  # noqa: BLE001 - a missing engine is recorded, not raised
        problog_module = {
            "status": "UNAVAILABLE",
            "diagnostic": f"{type(error).__name__}: {error}",
        }
    software["problog_module"] = problog_module
    resolved = shutil.which(str(args.problog_bin))
    software["problog_binary"] = {
        "requested": str(args.problog_bin),
        "resolved": resolved,
        "version": probe_command(
            (resolved or str(args.problog_bin), "--version"), repo
        ),
    }
    return software


def require_engines(args: argparse.Namespace) -> None:
    """Fail loudly and early when an engine the run needs is not installed."""
    try:
        import pyxlog  # noqa: F401
    except ImportError as error:
        raise RuntimeError(f"pyxlog is required for the xlog arm: {error}") from error
    try:
        import torch  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            f"torch is required to read pyxlog's DLPack probability tensor: {error}"
        ) from error
    if args.problog_mode == "module":
        try:
            import problog  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                f"ProbLog is required for --problog-mode module: {error}"
            ) from error
    elif shutil.which(str(args.problog_bin)) is None:
        raise RuntimeError(
            f"ProbLog binary not found: {args.problog_bin} "
            "(see --problog-bin / --problog-mode)"
        )


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    if args.self_test:
        self_test(repo)
        return 0
    if args.repetitions <= 0 or args.timeout_s <= 0 or args.memory_mb <= 0:
        raise ValueError("repetitions, timeout-s and memory-mb must be positive")
    if args.tolerance <= 0:
        raise ValueError("tolerance must be positive")

    require_engines(args)
    repository = repository_state(repo, args.allow_dirty)
    selected = set(args.case or CASE_NAMES)
    cases = [case for case in all_cases() if case.name in selected]

    owned_work_dir = args.work_dir is None
    work_dir = Path(
        tempfile.mkdtemp(
            prefix="xlog-exact-problog-",
            dir=str(args.work_dir.resolve()) if args.work_dir else None,
        )
    )
    try:
        hardware = hardware_state(repo)
        software = software_state(args, repo)
        runner = runner_state(repo)
        results: list[dict[str, Any]] = []
        for case in cases:
            sys.stderr.write(f"BEGIN {case.name}\n")
            sys.stderr.flush()
            xlog = run_xlog_arm(case, args.repetitions, args.device, args.memory_mb)
            xlog, xlog_error, xlog_ok = gate_arm(
                xlog, case.queries, problog=False, tolerance=args.tolerance
            )
            problog = run_problog_arm(
                case,
                args.repetitions,
                args.problog_mode,
                args.problog_bin,
                work_dir,
                args.timeout_s,
            )
            problog, problog_error, problog_ok = gate_arm(
                problog, case.queries, problog=True, tolerance=args.tolerance
            )
            results.append(
                {
                    "name": case.name,
                    "expected": sorted(
                        round_probability(query.expected) for query in case.queries
                    ),
                    "expected_by_query": [
                        {
                            "xlog_atom": query.xlog_atom,
                            "problog_term": query.problog_term,
                            "expected": query.expected,
                        }
                        for query in case.queries
                    ],
                    "xlog": xlog,
                    "xlog_max_abs_err": xlog_error,
                    "problog": problog,
                    "problog_max_abs_err": problog_error,
                    "case_acceptable": bool(xlog_ok and problog_ok),
                    "sources": {
                        "xlog_sha256": sha256_bytes(case.xlog_source.encode()),
                        "problog_sha256": sha256_bytes(case.problog_source.encode()),
                    },
                }
            )
            sys.stderr.write(
                f"END {case.name} xlog_ok={xlog_ok} problog_ok={problog_ok}\n"
            )
            sys.stderr.flush()

        timing_note = (
            "get_evaluatable().create_from(...).evaluate() in the same process"
            if args.problog_mode == "module"
            else "one CLI invocation, interpreter startup included"
        )
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": BENCHMARK,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository": repository,
            "runner": runner,
            "hardware": hardware,
            "software": software,
            "protocol": {
                "programs": list(CASE_NAMES),
                "cases_run": [case.name for case in cases],
                "repetitions": args.repetitions,
                "reported_time": (
                    "median of the per-repetition end-to-end wall time; xlog = "
                    "Program.compile + evaluate in one process, ProbLog = "
                    + timing_note
                ),
                "problog_mode": args.problog_mode,
                "xlog_device": args.device,
                "xlog_memory_mb": args.memory_mb,
                "correctness_gate": (
                    "max |p - analytic| over the queries of a program, computed on "
                    "unrounded probabilities"
                ),
                "tolerance": args.tolerance,
                "analytic_answers": {
                    "wet_conditioning": (
                        "P(rain|wet)=0.7/0.76, P(sprinkler|wet)=0.2/0.76"
                    ),
                    "reach_chain_n": "0.5^(n-1)",
                },
                "wet_program_source": WET_EXAMPLE_RELATIVE.as_posix(),
                "timeout_s": args.timeout_s,
                "circuit_cache_dir": os.environ.get("XLOG_CIRCUIT_CACHE_DIR"),
                "failed_arm_policy": (
                    "a failed arm is recorded as a failure and never replaced by "
                    "another execution path"
                ),
            },
            "comparison_acceptable": bool(
                results and all(result["case_acceptable"] for result in results)
            ),
            "results": results,
            "note": NOTE,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary_output.write_text(
            json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_output, args.output)
        sys.stderr.write(f"WROTE {args.output}\n")
        return 0 if artifact["comparison_acceptable"] else 1
    finally:
        if owned_work_dir and not args.keep_work_dir:
            shutil.rmtree(work_dir)
        else:
            sys.stderr.write(f"WORK_DIR {work_dir}\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"benchmark runner failed: {error}\n")
        raise SystemExit(2) from error
