#!/usr/bin/env python3
"""Reproduce the MNIST-addition head-to-head between xlog and Scallop.

The runner orchestrates the two harnesses that already live in the repository --
``examples/neural/01_minimal/train.py --engine xlog`` and
``examples/neural/baseline/scallop/mnist_addition.py`` -- once per (protocol,
seed, repetition), reads back each side's frozen-schema ``metrics.json``, parses
the accuracy line each side prints, aggregates over seeds exactly the way the
published artifact does, and writes one self-contained JSON artifact with
provenance. It never trains anything itself and never substitutes a different
execution path for a failed one: a failed seed is recorded and
``comparison_acceptable`` becomes false.

Imports are standard library only, on purpose: the runner starts subprocesses
and parses their output, so it must be startable by an interpreter that has
neither torch, nor pyxlog, nor scallopy installed (the two sides may live in
different interpreters -- see ``--scallop-python``). Anything that needs those
packages is probed in a subprocess of the interpreter that owns them and is
recorded as ``UNAVAILABLE`` with its stderr when it fails.

Published artifact reproduced by this runner:
``paper/artifacts/head-to-head/mnist_addition_vs_scallop.json``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
BENCHMARK = "mnist_addition_vs_scallop"
ABSENT = "<absent>"
UNAVAILABLE = "UNAVAILABLE"

# Paths of the two harnesses, relative to the repository root.
XLOG_TRAIN_REL = Path("examples/neural/01_minimal/train.py")
SCALLOP_TRAIN_REL = Path("examples/neural/baseline/scallop/mnist_addition.py")
# examples/neural/baseline/scallop/mnist_addition.py hardcodes this directory as
# its MNIST root (MNIST_DATA); it has no data-directory flag. --data-dir can
# therefore only move the xlog side, and a mismatch is recorded as a divergence.
SCALLOP_MNIST_REL = Path("examples/neural/01_minimal/data/mnist")
NET_CLASS = "MNISTNet"

# Shared configuration of the published comparison ("shared_config" plus the
# metric definition in the artifact, and the Scallop provenance named in
# "notes").
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TEST_IMAGES = 10_000
TEST_PAIRS = TEST_IMAGES // 2
SCALLOP_PROVENANCE = "difftopbottomkclauses"
SCALLOP_K = 3


@dataclass(frozen=True)
class Protocol:
    name: str
    train_images: int
    epochs: int
    seeds: tuple[int, ...]


# train_images/epochs and the seed COUNTS (5 and 3) come from the published
# artifact. The seed VALUES do not: the artifact records only "n". (7, 42, 123)
# are the repository's DEFAULT_SEEDS (scripts/track_a_runner.py) and are the
# seeds the committed Scallop baseline was run with
# (examples/neural/baseline/results/scallop_mnist/seed_{42,123,7}); 0 and 1
# extend that triple to five, 0 being the default --seed of both harnesses.
# Pass --seeds to override; the choice is recorded in the artifact under
# protocol.seed_values_are_reconstructed.
PROTOCOLS: tuple[Protocol, ...] = (
    Protocol("whitepaper_512x5", 512, 5, (7, 42, 123, 0, 1)),
    Protocol("strong_20000x5", 20_000, 5, (7, 42, 123)),
)
PROTOCOL_NAMES = tuple(protocol.name for protocol in PROTOCOLS)

METRIC_KEYS = (
    "heldout_addition_acc",
    "first_epoch_sec",
    "steady_epoch_sec",
    "total_train_sec",
)

# Copied verbatim out of paper/artifacts/head-to-head/mnist_addition_vs_scallop.json
# so that --self-test can check the aggregator against real published numbers
# instead of against numbers this file made up. Keys: (protocol, side, metric),
# values: (values, published mean, published std).
PUBLISHED_AGGREGATES: dict[tuple[str, str, str], tuple[list[float], float, float]] = {
    ("whitepaper_512x5", "xlog", "heldout_addition_acc"): (
        [0.1092, 0.1374, 0.1024, 0.1042, 0.128],
        0.1162,
        0.0139,
    ),
    ("whitepaper_512x5", "xlog", "first_epoch_sec"): (
        [85.758, 64.773, 65.048, 65.073, 65.195],
        69.1694,
        8.2954,
    ),
    ("whitepaper_512x5", "xlog", "steady_epoch_sec"): (
        [0.224, 0.224, 0.219, 0.216, 0.223],
        0.2212,
        0.0032,
    ),
    ("whitepaper_512x5", "xlog", "total_train_sec"): (
        [86.654, 65.67, 65.922, 65.938, 66.085],
        70.0538,
        8.3012,
    ),
    ("whitepaper_512x5", "scallop", "heldout_addition_acc"): (
        [0.1024, 0.1122, 0.1024, 0.0842, 0.0636],
        0.093,
        0.0172,
    ),
    ("whitepaper_512x5", "scallop", "first_epoch_sec"): (
        [0.797, 0.809, 0.82, 0.796, 0.8],
        0.8044,
        0.009,
    ),
    ("whitepaper_512x5", "scallop", "steady_epoch_sec"): (
        [0.641, 0.665, 0.605, 0.641, 0.663],
        0.643,
        0.0216,
    ),
    ("whitepaper_512x5", "scallop", "total_train_sec"): (
        [3.359, 3.467, 3.239, 3.359, 3.453],
        3.3754,
        0.0819,
    ),
    ("strong_20000x5", "xlog", "heldout_addition_acc"): (
        [0.955, 0.9504, 0.9584],
        0.9546,
        0.0033,
    ),
    ("strong_20000x5", "xlog", "first_epoch_sec"): (
        [73.38, 73.885, 73.611],
        73.6253,
        0.2064,
    ),
    ("strong_20000x5", "xlog", "steady_epoch_sec"): (
        [8.776, 8.772, 8.58],
        8.7093,
        0.0915,
    ),
    ("strong_20000x5", "xlog", "total_train_sec"): (
        [108.49, 108.983, 107.939],
        108.4707,
        0.4264,
    ),
    ("strong_20000x5", "scallop", "heldout_addition_acc"): (
        [0.9564, 0.9416, 0.952],
        0.95,
        0.0062,
    ),
    ("strong_20000x5", "scallop", "first_epoch_sec"): (
        [26.267, 24.753, 24.888],
        25.3027,
        0.6841,
    ),
    ("strong_20000x5", "scallop", "steady_epoch_sec"): (
        [24.731, 23.626, 24.689],
        24.3487,
        0.5113,
    ),
    ("strong_20000x5", "scallop", "total_train_sec"): (
        [125.189, 119.259, 123.642],
        122.6967,
        2.5115,
    ),
}

FINAL_METRIC_RE = re.compile(r"FINAL_METRIC:\s*(\S+)=([0-9.]+),\s*threshold=(\S+)")
# examples/neural/01_minimal/train.py: print(f"Held-out Correct/Total {c} {t}")
XLOG_PAIRS_RE = re.compile(r"^Held-out Correct/Total\s+(\d+)\s+(\d+)\s*$")
# baseline/scallop/mnist_addition.py: "Held-out addition accuracy: 0.1024 (512/5000)"
SCALLOP_PAIRS_RE = re.compile(
    r"^Held-out addition accuracy:\s*[0-9.]+\s*\((\d+)/(\d+)\)\s*$"
)
# train.py: "Generated 256 training queries"
XLOG_TRAIN_PAIRS_RE = re.compile(r"^Generated\s+(\d+)\s+training queries\s*$")
# mnist_addition.py: "Training 5 epochs, 256 pairs, batch_size=64"
SCALLOP_TRAIN_PAIRS_RE = re.compile(
    r"^Training\s+(\d+)\s+epochs,\s*(\d+)\s+pairs,\s*batch_size=(\d+)\s*$"
)
PROBE_PARAMETERS_RE = re.compile(r"^n_parameters=(\d+)$")


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    wall_s: float
    stdout: str
    stderr: str
    timed_out: bool


# --------------------------------------------------------------------------
# pure helpers (everything below is exercised by --self-test)
# --------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(values: Sequence[float]) -> dict[str, Any]:
    """Aggregate one metric across seeds the way the published artifact does.

    mean and the *population* standard deviation, both rounded to four decimals,
    the count, and every value. Verified against all sixteen published series in
    --self-test.
    """
    clean = [float(value) for value in values]
    if not clean:
        return {"mean": None, "std": None, "n": 0, "values": []}
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / len(clean)
    return {
        "mean": round(mean, 4),
        "std": round(math.sqrt(variance), 4),
        "n": len(clean),
        "values": clean,
    }


def adjacent_pairs(indices: Sequence[int]) -> list[tuple[int, int]]:
    """Pair a test split the way both harnesses do: (0,1), (2,3), ... .

    Both ``compute_addition_accuracy`` implementations take ``left = 2*i`` and
    ``right = 2*i + 1`` over ``n // 2`` pairs, so a trailing odd element is
    dropped. Every index is used at most once.
    """
    pair_count = len(indices) // 2
    return [(indices[2 * i], indices[2 * i + 1]) for i in range(pair_count)]


def pairing_is_disjoint(pairs: Sequence[tuple[int, int]]) -> bool:
    flat = [index for pair in pairs for index in pair]
    return len(flat) == len(set(flat))


def steady_epoch_sec(epoch_sec: Sequence[float]) -> float | None:
    """Median of every epoch except the first.

    The first epoch carries CUDA JIT plus circuit compilation on the xlog side
    and is reported separately; a single-epoch run has no steady state at all.
    """
    if len(epoch_sec) < 2:
        return None
    return float(median(epoch_sec[1:]))


def mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def epoch_timing_source(epoch_sec: Sequence[float]) -> str:
    """Tell measured per-epoch times from train.py's uniform fallback.

    examples/neural/01_minimal/train.py divides total_train_sec evenly across
    epochs when the training history carries no epoch_times. That fallback makes
    the first-epoch / steady-epoch split meaningless, so it must be visible.
    """
    if len(epoch_sec) < 2:
        return "single_epoch"
    if len(set(round(value, 6) for value in epoch_sec)) == 1:
        return "approximated_uniform"
    return "measured"


def parse_final_metric(text: str) -> tuple[str, float] | None:
    """Return the last ``FINAL_METRIC:`` line as (name, value)."""
    for line in reversed(text.splitlines()):
        match = FINAL_METRIC_RE.search(line)
        if match:
            return match.group(1), float(match.group(2))
    return None


def parse_heldout_pairs(text: str) -> int | None:
    """Number of held-out pairs each harness reports having evaluated."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        match = XLOG_PAIRS_RE.match(stripped) or SCALLOP_PAIRS_RE.match(stripped)
        if match:
            return int(match.group(2))
    return None


def parse_train_pairs(text: str) -> int | None:
    """Number of training pairs/queries each harness reports building."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        match = XLOG_TRAIN_PAIRS_RE.match(stripped)
        if match:
            return int(match.group(1))
        match = SCALLOP_TRAIN_PAIRS_RE.match(stripped)
        if match:
            return int(match.group(2))
    return None


def parse_probe_parameters(text: str) -> int | None:
    for line in text.splitlines():
        match = PROBE_PARAMETERS_RE.match(line.strip())
        if match:
            return int(match.group(1))
    return None


def parse_seed_spec(
    specs: Sequence[str],
    defaults: Mapping[str, tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    """Parse --seeds: ``7,42,123`` for every protocol or ``name=7,42``.

    Repeatable; a bare list applies to every protocol that no explicit
    ``name=`` entry names.
    """
    resolved = {name: tuple(seeds) for name, seeds in defaults.items()}
    explicit: set[str] = set()
    shared: tuple[int, ...] | None = None
    for spec in specs:
        name, separator, raw = spec.partition("=")
        if separator:
            if name not in resolved:
                raise ValueError(f"unknown protocol in --seeds: {name}")
            seeds = parse_seed_list(raw)
            resolved[name] = seeds
            explicit.add(name)
        else:
            shared = parse_seed_list(spec)
    if shared is not None:
        for name in resolved:
            if name not in explicit:
                resolved[name] = shared
    return resolved


def parse_seed_list(raw: str) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("--seeds needs at least one seed")
    seeds = tuple(int(part) for part in parts)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds contains a duplicate: {seeds}")
    return seeds


def config_divergence(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    left_name: str = "xlog",
    right_name: str = "scallop",
) -> list[dict[str, Any]]:
    """Every configuration field on which the two sides disagree."""
    divergence: list[dict[str, Any]] = []
    for field in sorted(set(left) | set(right)):
        left_value = left.get(field, ABSENT)
        right_value = right.get(field, ABSENT)
        if left_value != right_value:
            divergence.append(
                {"field": field, left_name: left_value, right_name: right_value}
            )
    return divergence


def merge_observed(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Collapse one side's per-run observations into one dict.

    A field that is not constant across the runs of one side is kept as an
    ``inconsistent`` marker so that it shows up as a divergence instead of
    silently taking the first value.
    """
    merged: dict[str, Any] = {}
    seen: dict[str, list[Any]] = {}
    for record in records:
        for field, value in record.items():
            if value is None:
                continue
            seen.setdefault(field, [])
            if value not in seen[field]:
                seen[field].append(value)
    for field, values in seen.items():
        merged[field] = values[0] if len(values) == 1 else {"inconsistent": values}
    return merged


def extract_class_source(source: str, class_name: str) -> str:
    """Return the source text of one class definition, newline-normalised.

    Used to prove from the artifact that both sides define the same network:
    the two files each carry their own copy of ``MNISTNet``.
    """
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    tree = ast.parse(normalized)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            segment = ast.get_source_segment(normalized, node)
            if segment is None:
                raise RuntimeError(f"cannot recover source of class {class_name}")
            lines = [line.rstrip() for line in segment.split("\n")]
            while lines and not lines[-1]:
                lines.pop()
            return "\n".join(lines) + "\n"
    raise RuntimeError(f"class {class_name} not found")


def normalized_class_source(source: str, class_name: str) -> str:
    """Structural form of a class definition: no comments, no docstrings.

    The two harnesses carry two copies of ``MNISTNet`` whose comments and
    docstrings differ while the layers do not. Hashing the raw text would call
    those copies different networks, so the hash that decides identity is taken
    from the AST with docstrings removed; the raw-text hash is recorded next to
    it.
    """
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    tree = ast.parse(normalized)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            strip_docstrings(node)
            return ast.unparse(node) + "\n"
    raise RuntimeError(f"class {class_name} not found")


def strip_docstrings(node: ast.AST) -> None:
    for child in ast.walk(node):
        if not isinstance(
            child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)
        ):
            continue
        body = getattr(child, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
            if not body:
                body.append(ast.Pass())


def build_probe_source(class_source: str, class_name: str) -> str:
    """A standalone module that instantiates the extracted network and counts it.

    Run with each side's own interpreter, so the parameter count is measured in
    the environment that side trains in.
    """
    return (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "\n\n"
        f"{class_source}"
        "\n"
        'if __name__ == "__main__":\n'
        f"    net = {class_name}()\n"
        '    print("n_parameters=%d" % sum(p.numel() for p in net.parameters()))\n'
    )


def parse_cpu_quota_cores(quota: str, period: str) -> float | None:
    if quota in {"max", "-1"}:
        return None
    quota_value = int(quota)
    period_value = int(period)
    if quota_value <= 0 or period_value <= 0:
        raise RuntimeError(
            f"invalid cgroup CPU quota: quota={quota_value} period={period_value}"
        )
    return round(quota_value / period_value, 6)


def cpu_model_name(cpuinfo: str) -> str | None:
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"model name", "Hardware", "Processor"}:
            model = value.strip()
            if model:
                return model
    return None


def normalized_command(argv: Iterable[str], repo: Path) -> str:
    return shlex.join(str(argument).replace(str(repo), "{repo}") for argument in argv)


def median_over_repetitions(values: Sequence[float]) -> float | None:
    """One seed's value when --repetitions > 1: the median, never the mean."""
    if not values:
        return None
    return float(median(values))


# --------------------------------------------------------------------------
# subprocess helpers
# --------------------------------------------------------------------------


def probe_command(
    argv: Sequence[str], cwd: Path, timeout_s: int = 120
) -> dict[str, Any]:
    """Run a short informational command; never raise, always record.

    A missing binary, a non-zero exit and a timeout are all recorded as
    ``status: UNAVAILABLE`` together with the diagnostic, so that an absent
    version string is visible in the artifact rather than missing from it.
    """
    record: dict[str, Any] = {"command": shlex.join(str(part) for part in argv)}
    try:
        completed = subprocess.run(
            [str(part) for part in argv],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as error:
        record["status"] = UNAVAILABLE
        record["stderr"] = f"{type(error).__name__}: {error}"
        return record
    record["returncode"] = completed.returncode
    record["stdout"] = (completed.stdout or "").strip()
    record["stderr"] = (completed.stderr or "").strip()
    record["status"] = "ok" if completed.returncode == 0 else UNAVAILABLE
    return record


def run_logged(
    argv: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    log_dir: Path,
    timeout_s: int,
) -> CommandResult:
    """Run one training job, streaming both streams to files as they arrive.

    Output written to files survives a timeout, which is why the harness logs
    are readable after a kill.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    popen_kwargs: dict[str, Any] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        with stderr_path.open("w", encoding="utf-8", newline="\n") as err_handle:
            try:
                process = subprocess.Popen(
                    [str(part) for part in argv],
                    cwd=str(cwd),
                    env=dict(env),
                    stdout=out_handle,
                    stderr=err_handle,
                    **popen_kwargs,
                )
            except OSError as error:
                # A missing interpreter is a recorded failure of that side, not
                # a crash of the whole benchmark.
                message = (
                    f"runner: cannot start command: {type(error).__name__}: {error}\n"
                )
                err_handle.write(message)
                return CommandResult(
                    tuple(str(part) for part in argv),
                    127,
                    time.monotonic() - started,
                    "",
                    message,
                    False,
                )
            try:
                returncode = process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                terminate_process_tree(process)
                returncode = 124
                timed_out = True
    wall_s = time.monotonic() - started
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    if timed_out:
        stderr += f"\nrunner: command timed out after {timeout_s} seconds\n"
        stderr_path.write_text(stderr, encoding="utf-8", newline="\n")
    return CommandResult(
        tuple(str(part) for part in argv), returncode, wall_s, stdout, stderr, timed_out
    )


def terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except OSError:
            process.terminate()
        try:
            process.wait(timeout=30)
            return
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except OSError:
                process.kill()
    else:
        process.terminate()
    process.wait()


def error_record(result: CommandResult) -> dict[str, Any]:
    lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
    diagnostic = lines[-1] if lines else "command failed without stderr"
    return {
        "kind": "timeout" if result.timed_out else "process_failure",
        "diagnostic": diagnostic,
        "stdout_sha256": sha256_bytes(result.stdout.encode()),
        "stderr_sha256": sha256_bytes(result.stderr.encode()),
        "stderr_tail": "\n".join(lines[-20:]),
    }


def run_text(argv: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        [str(part) for part in argv],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stdout or completed.stderr).strip()


def repository_state(repo: Path, allow_dirty: bool) -> dict[str, Any]:
    commit = run_text(("git", "rev-parse", "HEAD"), repo)
    dirty_lines = run_text(("git", "status", "--porcelain"), repo).splitlines()
    if dirty_lines and not allow_dirty:
        raise RuntimeError(
            "official benchmark requires a clean checkout; commit changes first "
            "or pass --allow-dirty"
        )
    return {
        "commit": commit,
        "dirty": bool(dirty_lines),
        "dirty_entries": dirty_lines if dirty_lines else [],
        "remote": run_text(("git", "remote", "get-url", "origin"), repo),
    }


def cgroup_cpu_quota() -> dict[str, Any]:
    """Both cgroup generations, each recorded even when the file is absent."""
    state: dict[str, Any] = {}
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if cpu_max.is_file():
        parts = cpu_max.read_text(encoding="utf-8").split()
        if len(parts) != 2:
            raise RuntimeError(f"invalid cgroup v2 cpu.max: {parts}")
        state["v2_cores"] = parse_cpu_quota_cores(parts[0], parts[1])
        state["v2_raw"] = " ".join(parts)
    else:
        state["v2_cores"] = UNAVAILABLE
        state["v2_raw"] = f"{cpu_max} is absent"
    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.is_file() and period_path.is_file():
        quota = quota_path.read_text(encoding="utf-8").strip()
        period = period_path.read_text(encoding="utf-8").strip()
        state["v1_cores"] = parse_cpu_quota_cores(quota, period)
        state["v1_raw"] = f"quota={quota} period={period}"
    else:
        state["v1_cores"] = UNAVAILABLE
        state["v1_raw"] = f"{quota_path} / {period_path} are absent"
    return state


def hardware_state(repo: Path) -> dict[str, Any]:
    gpu = probe_command(
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        repo,
    )
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.is_file():
        cpu = cpu_model_name(cpuinfo_path.read_text(encoding="utf-8")) or UNAVAILABLE
    else:
        cpu = UNAVAILABLE
    try:
        host_memory_bytes: Any = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
            "SC_PHYS_PAGES"
        )
    except (AttributeError, ValueError, OSError) as error:
        host_memory_bytes = f"{UNAVAILABLE}: {type(error).__name__}: {error}"
    return {
        "gpu": gpu,
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "cpu_quota_cores": cgroup_cpu_quota(),
        "host_memory_bytes": host_memory_bytes,
        "platform": platform.platform(),
    }


def module_version(python_bin: Path, module: str, repo: Path) -> dict[str, Any]:
    return probe_command(
        (
            str(python_bin),
            "-c",
            f"import {module}; print(getattr({module}, '__version__', 'unknown'))",
        ),
        repo,
    )


def software_state(
    repo: Path, xlog_python: Path, scallop_python: Path, skip_scallop: bool
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "runner_python": platform.python_version(),
        "xlog_python": {
            "path": str(xlog_python),
            "version": probe_command((str(xlog_python), "-VV"), repo),
            "torch": module_version(xlog_python, "torch", repo),
            "torchvision": module_version(xlog_python, "torchvision", repo),
            "pyxlog": module_version(xlog_python, "pyxlog", repo),
        },
        "nvcc": probe_command(("nvcc", "--version"), repo),
    }
    if skip_scallop:
        state["scallop_python"] = {
            "status": UNAVAILABLE,
            "stderr": "--skip-scallop was passed; the Scallop side was not run",
        }
    else:
        state["scallop_python"] = {
            "path": str(scallop_python),
            "version": probe_command((str(scallop_python), "-VV"), repo),
            "torch": module_version(scallop_python, "torch", repo),
            "torchvision": module_version(scallop_python, "torchvision", repo),
            "scallopy": module_version(scallop_python, "scallopy", repo),
        }
    return state


def network_identity(
    repo: Path,
    work_dir: Path,
    xlog_python: Path,
    scallop_python: Path,
    skip_scallop: bool,
) -> dict[str, Any]:
    """Definition hash (always) and parameter count (when torch is importable)."""
    identity: dict[str, Any] = {}
    sides = [("xlog", XLOG_TRAIN_REL, xlog_python)]
    if not skip_scallop:
        sides.append(("scallop", SCALLOP_TRAIN_REL, scallop_python))
    for side, relative, python_bin in sides:
        source_path = repo / relative
        class_source = extract_class_source(
            source_path.read_text(encoding="utf-8"), NET_CLASS
        )
        probe_path = work_dir / f"net_probe_{side}.py"
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        probe_path.write_text(
            build_probe_source(class_source, NET_CLASS), encoding="utf-8", newline="\n"
        )
        probe = probe_command((str(python_bin), str(probe_path)), work_dir)
        parameters: Any = UNAVAILABLE
        if probe.get("status") == "ok":
            parsed = parse_probe_parameters(probe.get("stdout", ""))
            parameters = parsed if parsed is not None else UNAVAILABLE
        identity[side] = {
            "source": str(relative),
            "source_sha256": sha256_file(source_path),
            "class": NET_CLASS,
            "definition_sha256": sha256_bytes(
                normalized_class_source(
                    source_path.read_text(encoding="utf-8"), NET_CLASS
                ).encode()
            ),
            "definition_text_sha256": sha256_bytes(class_source.encode()),
            "definition_hash_basis": (
                "sha256 of the class AST with comments and docstrings removed"
            ),
            "n_parameters": parameters,
            "probe": probe,
        }
    if len([side for side in ("xlog", "scallop") if side in identity]) == 2:
        identical = (
            identity["xlog"]["definition_sha256"]
            == identity["scallop"]["definition_sha256"]
        )
        counts = [identity[side]["n_parameters"] for side in ("xlog", "scallop")]
        if UNAVAILABLE in counts:
            parameters_identical: Any = UNAVAILABLE
        else:
            parameters_identical = counts[0] == counts[1]
        identity["definitions_identical"] = identical
        identity["parameter_counts_identical"] = parameters_identical
        identity["networks_match"] = (
            bool(identical) and parameters_identical is not False
        )
    else:
        identity["definitions_identical"] = None
        identity["parameter_counts_identical"] = None
        identity["networks_match"] = None
    return identity


# --------------------------------------------------------------------------
# one training run
# --------------------------------------------------------------------------


def xlog_command(
    python_bin: Path,
    repo: Path,
    protocol: Protocol,
    seed: int,
    data_dir: Path,
    run_dir: Path,
) -> tuple[str, ...]:
    return (
        str(python_bin),
        "-u",
        str(repo / XLOG_TRAIN_REL),
        "--engine",
        "xlog",
        "--epochs",
        str(protocol.epochs),
        "--batch-size",
        str(BATCH_SIZE),
        "--lr",
        repr(LEARNING_RATE),
        "--seed",
        str(seed),
        "--train-limit",
        str(protocol.train_images),
        "--data-path",
        str(data_dir),
        "--save-path",
        str(run_dir / "mnist_net.pt"),
        "--metrics-path",
        str(run_dir / "metrics.json"),
    )


def scallop_command(
    python_bin: Path, repo: Path, protocol: Protocol, seed: int, run_dir: Path
) -> tuple[str, ...]:
    return (
        str(python_bin),
        "-u",
        str(repo / SCALLOP_TRAIN_REL),
        "--epochs",
        str(protocol.epochs),
        "--batch-size",
        str(BATCH_SIZE),
        "--lr",
        repr(LEARNING_RATE),
        "--seed",
        str(seed),
        "--train-limit",
        str(protocol.train_images),
        "--provenance",
        SCALLOP_PROVENANCE,
        "--k",
        str(SCALLOP_K),
        "--metrics-path",
        str(run_dir / "metrics.json"),
    )


def execute_side(
    side: str,
    argv: Sequence[str],
    repo: Path,
    run_dir: Path,
    protocol: Protocol,
    seed: int,
    repetition: int,
    data_dir: Path,
    timeout_s: int,
) -> dict[str, Any]:
    """Run one side once and turn its output into one record.

    Never raises for a failing child: the failure is the result.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env["PYTHONUNBUFFERED"] = "1"
    result = run_logged(argv, repo, env, run_dir, timeout_s)
    record: dict[str, Any] = {
        "side": side,
        "protocol": protocol.name,
        "seed": seed,
        "repetition": repetition,
        "command": normalized_command(argv, repo),
        "returncode": result.returncode,
        "wall_s": round(result.wall_s, 6),
        "log_dir": str(run_dir),
        "stdout_sha256": sha256_bytes(result.stdout.encode()),
        "stderr_sha256": sha256_bytes(result.stderr.encode()),
        "warnings": [],
        "observed": {},
    }
    if result.returncode != 0:
        record["status"] = "failed"
        record["error"] = error_record(result)
        return record

    metrics_path = run_dir / "metrics.json"
    problems: list[str] = []
    metrics: dict[str, Any] = {}
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"metrics.json unreadable: {type(error).__name__}: {error}")
    else:
        problems.append(f"metrics.json was not written to {metrics_path}")
    record["metrics_json"] = metrics or None

    final_metric = parse_final_metric(result.stdout)
    if final_metric is None:
        problems.append("no FINAL_METRIC line on stdout")
    elif final_metric[0] != "heldout_addition_acc":
        problems.append(f"unexpected FINAL_METRIC name: {final_metric[0]}")

    epoch_sec = [float(value) for value in metrics.get("epoch_sec", [])]
    if len(epoch_sec) != protocol.epochs:
        problems.append(
            f"metrics.json reports {len(epoch_sec)} epochs, protocol asks for "
            f"{protocol.epochs}"
        )
    timing_source = epoch_timing_source(epoch_sec)
    if timing_source == "approximated_uniform":
        problems.append(
            "per-epoch times are the harness's uniform fallback, so the "
            "first-epoch / steady-epoch split does not measure warm-up"
        )

    record["epoch_sec"] = epoch_sec
    record["epoch_timing_source"] = timing_source
    record["heldout_addition_acc"] = final_metric[1] if final_metric else None
    record["first_epoch_sec"] = epoch_sec[0] if epoch_sec else None
    record["steady_epoch_sec"] = steady_epoch_sec(epoch_sec)
    record["steady_epoch_sec_mean_over_epochs"] = mean_or_none(epoch_sec[1:])
    record["total_train_sec"] = metrics.get("total_train_sec")
    record["compile_api_sec"] = metrics.get("compile_api_sec")
    record["observed"] = {
        "epochs": protocol.epochs,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "train_images": protocol.train_images,
        "train_pairs": parse_train_pairs(result.stdout),
        "test_images": TEST_IMAGES,
        "heldout_pairs": parse_heldout_pairs(result.stdout),
        "pairing": "adjacent (2i, 2i+1)",
        "mnist_data_dir": str(
            data_dir if side == "xlog" else (repo / SCALLOP_MNIST_REL).resolve()
        ),
    }
    if side == "scallop":
        record["observed"]["provenance"] = SCALLOP_PROVENANCE
        record["observed"]["k"] = SCALLOP_K
    missing = [
        key
        for key in ("heldout_addition_acc", "first_epoch_sec", "total_train_sec")
        if record.get(key) is None
    ]
    if missing:
        problems.append(f"missing measurements: {', '.join(missing)}")
    record["warnings"] = problems
    record["status"] = "ok" if not problems else "protocol_violation"
    return record


def skipped_record(
    side: str, protocol: Protocol, seed: int, repetition: int, reason: str
) -> dict[str, Any]:
    return {
        "side": side,
        "protocol": protocol.name,
        "seed": seed,
        "repetition": repetition,
        "status": "skipped",
        "returncode": None,
        "error": {"kind": "skipped", "diagnostic": reason},
        "warnings": [reason],
        "observed": {},
    }


def side_summary(
    runs: Sequence[Mapping[str, Any]], seeds: Sequence[int]
) -> dict[str, Any]:
    """Per-seed values (median over repetitions) and the across-seed aggregates."""
    per_seed: list[dict[str, Any]] = []
    collected: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    complete = True
    for seed in seeds:
        seed_runs = [
            run for run in runs if run["seed"] == seed and run["status"] == "ok"
        ]
        entry: dict[str, Any] = {
            "seed": seed,
            "successful_repetitions": len(seed_runs),
            "requested_repetitions": len([run for run in runs if run["seed"] == seed]),
        }
        if not seed_runs:
            entry["status"] = "failed"
            complete = False
            per_seed.append(entry)
            continue
        entry["status"] = "ok"
        for key in METRIC_KEYS:
            values = [float(run[key]) for run in seed_runs if run.get(key) is not None]
            value = median_over_repetitions(values)
            entry[key] = value
            if value is None:
                complete = False
            else:
                collected[key].append(value)
        per_seed.append(entry)
    summary: dict[str, Any] = {
        "complete": complete and len(per_seed) == len(seeds),
        "per_seed": per_seed,
    }
    for key in METRIC_KEYS:
        summary[key] = aggregate(collected[key])
    return summary


def run_protocol(
    protocol: Protocol,
    seeds: Sequence[int],
    args: argparse.Namespace,
    repo: Path,
    work_dir: Path,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        for repetition in range(1, args.repetitions + 1):
            xlog_dir = (
                work_dir / protocol.name / f"seed_{seed}" / f"rep_{repetition}" / "xlog"
            )
            log(f"BEGIN {protocol.name} xlog seed={seed} rep={repetition}")
            runs.append(
                execute_side(
                    "xlog",
                    xlog_command(
                        args.python, repo, protocol, seed, args.data_dir, xlog_dir
                    ),
                    repo,
                    xlog_dir,
                    protocol,
                    seed,
                    repetition,
                    args.data_dir,
                    args.timeout_s,
                )
            )
            log(f"END   {protocol.name} xlog seed={seed} status={runs[-1]['status']}")
            if args.skip_scallop:
                runs.append(
                    skipped_record(
                        "scallop",
                        protocol,
                        seed,
                        repetition,
                        "--skip-scallop was passed",
                    )
                )
                continue
            scallop_dir = (
                work_dir
                / protocol.name
                / f"seed_{seed}"
                / f"rep_{repetition}"
                / "scallop"
            )
            log(f"BEGIN {protocol.name} scallop seed={seed} rep={repetition}")
            runs.append(
                execute_side(
                    "scallop",
                    scallop_command(
                        args.scallop_python, repo, protocol, seed, scallop_dir
                    ),
                    repo,
                    scallop_dir,
                    protocol,
                    seed,
                    repetition,
                    args.data_dir,
                    args.timeout_s,
                )
            )
            log(
                f"END   {protocol.name} scallop seed={seed} status={runs[-1]['status']}"
            )

    xlog_runs = [run for run in runs if run["side"] == "xlog"]
    scallop_runs = [run for run in runs if run["side"] == "scallop"]
    xlog_summary = side_summary(xlog_runs, seeds)
    scallop_summary = side_summary(scallop_runs, seeds)

    xlog_observed = merge_observed(run.get("observed", {}) for run in xlog_runs)
    scallop_observed = merge_observed(run.get("observed", {}) for run in scallop_runs)
    # provenance/k exist only on the Scallop side by construction; comparing them
    # would report a divergence that is not one.
    comparable_scallop = {
        field: value
        for field, value in scallop_observed.items()
        if field not in {"provenance", "k"}
    }
    if args.skip_scallop:
        divergence = [
            {
                "field": "scallop_side",
                "xlog": "executed",
                "scallop": "not executed (--skip-scallop)",
            }
        ]
    elif not xlog_observed or not comparable_scallop:
        # One side produced no usable output at all; listing every field as
        # "absent" would drown the real failure recorded in runs[].
        divergence = [
            {
                "field": "observed_config",
                "xlog": "recorded" if xlog_observed else "no successful run",
                "scallop": "recorded" if comparable_scallop else "no successful run",
            }
        ]
    else:
        divergence = config_divergence(xlog_observed, comparable_scallop)
    acceptable = bool(
        xlog_summary["complete"]
        and scallop_summary["complete"]
        and not divergence
        and not args.skip_scallop
    )
    return {
        "protocol": protocol.name,
        "train_images": protocol.train_images,
        "epochs": protocol.epochs,
        "seeds": list(seeds),
        "repetitions": args.repetitions,
        "observed_config": {"xlog": xlog_observed, "scallop": scallop_observed},
        "divergence": divergence,
        "xlog": xlog_summary,
        "scallop": scallop_summary,
        "comparison_acceptable": acceptable,
        "runs": runs,
    }


def log(message: str) -> None:
    """Progress goes to stderr; stdout carries only the final artifact path."""
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------


def self_test() -> None:
    # 1. Aggregator against every published series, values taken from the artifact.
    for (protocol, side, metric), (values, mean, std) in PUBLISHED_AGGREGATES.items():
        computed = aggregate(values)
        assert computed["values"] == values, (protocol, side, metric)
        assert computed["n"] == len(values), (protocol, side, metric)
        assert computed["mean"] == mean, (
            protocol,
            side,
            metric,
            computed["mean"],
            mean,
        )
        assert computed["std"] == std, (protocol, side, metric, computed["std"], std)
    # A sample standard deviation would not reproduce the artifact; make sure the
    # test would notice.
    sample_std = math.sqrt(
        sum(
            (v - 0.1162) ** 2
            for v in PUBLISHED_AGGREGATES[
                ("whitepaper_512x5", "xlog", "heldout_addition_acc")
            ][0]
        )
        / 4
    )
    assert round(sample_std, 4) != 0.0139
    assert aggregate([]) == {"mean": None, "std": None, "n": 0, "values": []}

    # 2. Pairing of the held-out split.
    synthetic = list(range(10))
    assert adjacent_pairs(synthetic) == [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    assert pairing_is_disjoint(adjacent_pairs(synthetic))
    assert adjacent_pairs(list(range(9))) == adjacent_pairs(synthetic)[:4]
    full = adjacent_pairs(list(range(TEST_IMAGES)))
    assert len(full) == TEST_PAIRS == 5000
    assert full[0] == (0, 1) and full[-1] == (9998, 9999)
    assert pairing_is_disjoint(full)
    assert adjacent_pairs(list(range(TEST_IMAGES))) == full  # deterministic
    assert adjacent_pairs([]) == []

    # 3. Steady epoch time ignores the first epoch.
    # even count of steady epochs -> the median averages the two middle values
    assert steady_epoch_sec([85.758, 0.224, 0.224, 0.219, 0.216]) == 0.2215
    assert steady_epoch_sec([85.758, 0.224, 0.219, 0.216]) == 0.219
    assert steady_epoch_sec([100.0, 1.0, 2.0, 3.0]) == 2.0
    assert steady_epoch_sec([100.0, 1.0, 2.0, 3.0]) == steady_epoch_sec(
        [0.001, 1.0, 2.0, 3.0]
    )
    assert steady_epoch_sec([7.0]) is None
    assert steady_epoch_sec([]) is None
    assert mean_or_none([1.0, 2.0]) == 1.5
    assert mean_or_none([]) is None
    assert epoch_timing_source([0.98, 0.297, 0.272]) == "measured"
    assert epoch_timing_source([13.2, 13.2, 13.2, 13.2]) == "approximated_uniform"
    assert epoch_timing_source([1.0]) == "single_epoch"
    assert median_over_repetitions([3.0, 1.0, 2.0]) == 2.0
    assert median_over_repetitions([]) is None

    # 4. Divergence detection between the two sides' configurations.
    base = {"batch_size": 64, "lr": 0.001, "train_images": 512}
    assert config_divergence(base, dict(base)) == []
    differing = config_divergence(base, {**base, "batch_size": 32, "lr": 0.0005})
    assert [entry["field"] for entry in differing] == ["batch_size", "lr"]
    assert differing[0] == {"field": "batch_size", "xlog": 64, "scallop": 32}
    missing_field = config_divergence(base, {"batch_size": 64, "lr": 0.001})
    assert missing_field == [{"field": "train_images", "xlog": 512, "scallop": ABSENT}]
    assert merge_observed([{"a": 1}, {"a": 1}, {"b": None}]) == {"a": 1}
    assert merge_observed([{"a": 1}, {"a": 2}]) == {"a": {"inconsistent": [1, 2]}}

    # 5. Output parsers, on the exact lines the two harnesses print.
    xlog_stdout = (
        "Generated 256 training queries\n"
        "  Held-out digit accuracy: 0.1010\n"
        "Held-out Accuracy 0.1092\n"
        "Held-out Correct/Total 546 5000\n"
        "FINAL_METRIC: heldout_addition_acc=0.1092, threshold=none\n"
    )
    scallop_stdout = (
        "Training 5 epochs, 256 pairs, batch_size=64\n"
        "  epoch 1/5: loss=2.906441 (0.80s)\n"
        "\nHeld-out addition accuracy: 0.1024 (512/5000)\n"
        "FINAL_METRIC: heldout_addition_acc=0.1024, threshold=none\n"
    )
    assert parse_final_metric(xlog_stdout) == ("heldout_addition_acc", 0.1092)
    assert parse_final_metric(scallop_stdout) == ("heldout_addition_acc", 0.1024)
    assert parse_final_metric("nothing here") is None
    assert parse_heldout_pairs(xlog_stdout) == 5000
    assert parse_heldout_pairs(scallop_stdout) == 5000
    assert parse_heldout_pairs("") is None
    assert parse_train_pairs(xlog_stdout) == 256
    assert parse_train_pairs(scallop_stdout) == 256
    assert parse_train_pairs("Generated 10000 training queries") == 10_000
    assert parse_probe_parameters("n_parameters=44426\n") == 44426
    assert parse_probe_parameters("") is None

    # 6. --seeds parsing.
    defaults = {protocol.name: protocol.seeds for protocol in PROTOCOLS}
    assert parse_seed_spec([], defaults) == defaults
    assert parse_seed_spec(["7,42"], defaults) == {
        "whitepaper_512x5": (7, 42),
        "strong_20000x5": (7, 42),
    }
    assert parse_seed_spec(["strong_20000x5=1,2,3"], defaults)["strong_20000x5"] == (
        1,
        2,
        3,
    )
    assert parse_seed_spec(["strong_20000x5=1", "9,8"], defaults) == {
        "whitepaper_512x5": (9, 8),
        "strong_20000x5": (1,),
    }
    for bad in (["no_such=1"], ["1,1"], [""]):
        try:
            parse_seed_spec(bad, defaults)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad --seeds accepted: {bad}")

    # 7. Network-definition extraction and the parameter probe it generates.
    module_source = (
        "import torch\r\n"
        "class Other:\r\n"
        "    pass\r\n"
        "class MNISTNet(nn.Module):\r\n"
        "    def __init__(self):\r\n"
        "        super().__init__()\r\n"
        "        self.fc = nn.Linear(4, 2)   \r\n"
        "\r\n"
    )
    extracted = extract_class_source(module_source, "MNISTNet")
    assert "\r" not in extracted
    assert extracted.startswith("class MNISTNet(nn.Module):\n")
    assert extracted.endswith("self.fc = nn.Linear(4, 2)\n")
    assert extract_class_source(module_source.replace("\r\n", "\n"), "MNISTNet") == (
        extracted
    )
    assert sha256_bytes(extracted.encode()) == sha256_bytes(
        extract_class_source(
            module_source.replace("class Other:\r\n    pass\r\n", ""), "MNISTNet"
        ).encode()
    )
    other = extract_class_source(
        module_source.replace("nn.Linear(4, 2)", "nn.Linear(4, 3)"), "MNISTNet"
    )
    assert sha256_bytes(other.encode()) != sha256_bytes(extracted.encode())
    # The two harnesses document their copies of MNISTNet differently, so the
    # hash that decides "same network" must ignore comments and docstrings and
    # must still notice a changed layer.
    documented = (
        "class MNISTNet(nn.Module):\n"
        '    """Same architecture as 01_minimal/train.py."""\n'
        "\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        # LeNet-5 style\n"
        "        self.fc = nn.Linear(4, 2)\n"
    )
    plain = (
        "class MNISTNet(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 2)\n"
    )
    assert normalized_class_source(documented, "MNISTNet") == normalized_class_source(
        plain, "MNISTNet"
    )
    assert sha256_bytes(
        extract_class_source(documented, "MNISTNet").encode()
    ) != sha256_bytes(extract_class_source(plain, "MNISTNet").encode())
    assert normalized_class_source(
        plain.replace("nn.Linear(4, 2)", "nn.Linear(4, 3)"), "MNISTNet"
    ) != normalized_class_source(plain, "MNISTNet")
    assert '"""' not in normalized_class_source(documented, "MNISTNet")
    try:
        normalized_class_source("x = 1\n", "MNISTNet")
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing class was accepted by the normaliser")
    try:
        extract_class_source("x = 1\n", "MNISTNet")
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing class was accepted")
    probe_source = build_probe_source(extracted, "MNISTNet")
    assert "import torch" in probe_source
    assert "sum(p.numel() for p in net.parameters())" in probe_source
    assert probe_source == build_probe_source(extracted, "MNISTNet")
    compile(probe_source, "<probe>", "exec")

    # 8. cgroup quota parsing, both generations.
    assert parse_cpu_quota_cores("765000", "100000") == 7.65
    assert parse_cpu_quota_cores("max", "100000") is None
    assert parse_cpu_quota_cores("-1", "100000") is None
    try:
        parse_cpu_quota_cores("0", "100000")
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid quota was accepted")
    assert cpu_model_name("processor: 0\nmodel name: Example CPU\n") == "Example CPU"
    assert cpu_model_name("processor: 0\n") is None

    # 9. An unavailable command is recorded, never swallowed.
    missing = probe_command(
        ("xlog-self-test-missing-binary", "--version"), Path(__file__).resolve().parent
    )
    assert missing["status"] == UNAVAILABLE
    assert missing["stderr"]
    failing = probe_command(
        (sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"),
        Path(__file__).resolve().parent,
    )
    assert failing["status"] == UNAVAILABLE
    assert failing["returncode"] == 3
    assert failing["stderr"] == "boom"
    working = probe_command((sys.executable, "-c", "print('ok')"), Path.cwd())
    assert working["status"] == "ok" and working["stdout"] == "ok"

    # 10. A side that fails is recorded as a failure, with no substitution.
    summary = side_summary(
        [
            {
                "seed": 7,
                "status": "ok",
                "heldout_addition_acc": 0.5,
                "first_epoch_sec": 10.0,
                "steady_epoch_sec": 1.0,
                "total_train_sec": 14.0,
            },
            {"seed": 42, "status": "failed"},
        ],
        [7, 42],
    )
    assert summary["complete"] is False
    assert summary["heldout_addition_acc"] == {
        "mean": 0.5,
        "std": 0.0,
        "n": 1,
        "values": [0.5],
    }
    assert summary["per_seed"][1]["status"] == "failed"
    repeated = side_summary(
        [
            {
                "seed": 7,
                "status": "ok",
                "heldout_addition_acc": 0.4,
                "first_epoch_sec": 9.0,
                "steady_epoch_sec": 3.0,
                "total_train_sec": 12.0,
            },
            {
                "seed": 7,
                "status": "ok",
                "heldout_addition_acc": 0.6,
                "first_epoch_sec": 11.0,
                "steady_epoch_sec": 1.0,
                "total_train_sec": 20.0,
            },
            {
                "seed": 7,
                "status": "ok",
                "heldout_addition_acc": 0.5,
                "first_epoch_sec": 10.0,
                "steady_epoch_sec": 2.0,
                "total_train_sec": 14.0,
            },
        ],
        [7],
    )
    assert repeated["complete"] is True
    assert repeated["per_seed"][0]["steady_epoch_sec"] == 2.0  # median, not mean
    assert repeated["total_train_sec"]["values"] == [14.0]

    # 11. Command builders.
    built = xlog_command(
        Path("/usr/bin/python3"),
        Path("/repo"),
        PROTOCOLS[0],
        42,
        Path("/data/mnist"),
        Path("/work/run"),
    )
    assert "--engine" in built and built[built.index("--engine") + 1] == "xlog"
    assert built[built.index("--batch-size") + 1] == "64"
    assert built[built.index("--lr") + 1] == "0.001"
    assert built[built.index("--train-limit") + 1] == "512"
    scallop_built = scallop_command(
        Path("/usr/bin/python3"), Path("/repo"), PROTOCOLS[1], 7, Path("/work/run")
    )
    assert scallop_built[scallop_built.index("--provenance") + 1] == (
        "difftopbottomkclauses"
    )
    assert scallop_built[scallop_built.index("--k") + 1] == "3"
    assert scallop_built[scallop_built.index("--train-limit") + 1] == "20000"
    assert scallop_built[scallop_built.index("--batch-size") + 1] == "64"
    repo_stub = Path("/repo")
    normalized = normalized_command((str(repo_stub / "x.py"), "--a"), repo_stub)
    assert "{repo}" in normalized
    assert str(repo_stub) not in normalized
    assert normalized.endswith("--a")

    print("mnist addition head-to-head runner self-test passed")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="path of the JSON artifact")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--protocol",
        action="append",
        choices=list(PROTOCOL_NAMES),
        help="run only this protocol; repeatable (default: both)",
    )
    parser.add_argument(
        "--seeds",
        action="append",
        default=[],
        metavar="[PROTOCOL=]S1,S2",
        help="override seeds for all protocols or for one; repeatable",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help=(
            "MNIST root for the xlog side (default: the directory the Scallop "
            "side hardcodes, so both read the same data)"
        ),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="interpreter with torch + pyxlog for the xlog side",
    )
    parser.add_argument(
        "--scallop-python",
        type=Path,
        help="interpreter with scallopy (default: --python)",
    )
    parser.add_argument("--skip-scallop", action="store_true")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="training repeats per seed; per-seed value is their median",
    )
    parser.add_argument("--timeout-s", type=int, default=7200)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="where per-run logs and metrics land (default: next to --output)",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output is None:
        raise ValueError("--output is required unless --self-test is passed")
    if args.repetitions <= 0 or args.timeout_s <= 0:
        raise ValueError("--repetitions and --timeout-s must be positive")

    repo = args.repo.resolve()
    args.output = args.output.resolve()
    if args.data_dir is None:
        args.data_dir = (repo / SCALLOP_MNIST_REL).resolve()
    else:
        args.data_dir = args.data_dir.resolve()
    if args.scallop_python is None:
        args.scallop_python = args.python
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir
        else args.output.parent / f"{args.output.stem}_runs"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    for relative in (XLOG_TRAIN_REL, SCALLOP_TRAIN_REL):
        if not (repo / relative).is_file():
            raise RuntimeError(
                f"harness is missing from the checkout: {repo / relative}"
            )

    selected = set(args.protocol or PROTOCOL_NAMES)
    protocols = [protocol for protocol in PROTOCOLS if protocol.name in selected]
    seeds_by_protocol = parse_seed_spec(
        args.seeds, {protocol.name: protocol.seeds for protocol in PROTOCOLS}
    )

    repo_info = repository_state(repo, args.allow_dirty)
    runner_path = Path(__file__).resolve()
    try:
        runner_name = str(runner_path.relative_to(repo))
    except ValueError:
        runner_name = str(runner_path)

    results = [
        run_protocol(protocol, seeds_by_protocol[protocol.name], args, repo, work_dir)
        for protocol in protocols
    ]
    network = network_identity(
        repo, work_dir, args.python, args.scallop_python, args.skip_scallop
    )
    net_identical = network.get("networks_match")
    acceptable = bool(
        results
        and all(result["comparison_acceptable"] for result in results)
        and net_identical is True
    )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repo_info,
        "runner": {
            "path": runner_name,
            "sha256": sha256_file(runner_path),
            "argv": sys.argv,
            "work_dir": str(work_dir),
        },
        "hardware": hardware_state(repo),
        "software": software_state(
            repo, args.python, args.scallop_python, args.skip_scallop
        ),
        "protocol": {
            "task": "MNIST addition trained from sum supervision only",
            "metric": (
                "held-out addition accuracy on the 10k MNIST test split, "
                f"{TEST_PAIRS} adjacent pairs"
            ),
            "pairing": "adjacent (2i, 2i+1) over the test split, in file order",
            "train_pairing": "adjacent (2i, 2i+1) over the first train_images images",
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "optimizer": "Adam (both harnesses)",
            "test_images": TEST_IMAGES,
            "test_pairs": TEST_PAIRS,
            "network": network,
            "xlog_harness": str(XLOG_TRAIN_REL),
            "scallop_harness": str(SCALLOP_TRAIN_REL),
            "scallop_provenance": {"provenance": SCALLOP_PROVENANCE, "k": SCALLOP_K},
            "mnist_data_dir": str(args.data_dir),
            "repetitions": args.repetitions,
            "reported_time": (
                "first_epoch_sec is epoch 1; steady_epoch_sec is the MEDIAN of "
                "epochs 2..n; total_train_sec is the harness's own total. Across "
                "seeds: mean, population std, n, and every value"
            ),
            "steady_epoch_definition": "median of epochs 2..n",
            "timeout_s": args.timeout_s,
            "skip_scallop": args.skip_scallop,
            "seed_values_are_reconstructed": True,
            "reconstruction_notes": [
                "The published artifact records only the seed COUNT (5 and 3), "
                "not the seed values; these seeds are this runner's default and "
                "may differ from the published run.",
                "The published artifact does not say whether its steady_epoch_sec "
                "was a median or a mean over epochs 2..n; the committed harnesses "
                "write steady_epoch_sec_mean, this runner reports the median and "
                "keeps the full epoch_sec list plus the mean per run.",
                "Pair shuffling differs by harness: the Scallop baseline reshuffles "
                "training pairs every epoch, the xlog side hands a fixed query list "
                "to pyxlog.train_model_tensor. This is not configurable from "
                "either command line and is not recorded in the published artifact.",
                "The published artifact records no data augmentation; both "
                "harnesses use ToTensor + Normalize((0.1307,), (0.3081,)) only.",
            ],
        },
        "comparison_acceptable": acceptable,
        "networks_match": net_identical,
        "results": {result["protocol"]: result for result in results},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    os.replace(temporary_output, args.output)
    print(f"WROTE {args.output}", flush=True)
    return 0 if acceptable else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"benchmark runner failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
