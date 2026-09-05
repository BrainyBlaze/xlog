#!/usr/bin/env python3
"""Reproduce the profile-gated shared-memory chain-scorer ablation.

The runner drives pyxlog's native exact induction twice over the same fixture in
one process -- once with ``XLOG_ILP_EXACT_CHAIN_SMEM`` on and once with it off --
on a chain-hot fixture above the row gate and on a small control below it, then
writes one self-contained JSON artifact.

It replaces the A/B core of ``scripts/measure_chain_shared_memory_profile.py``
and re-fills the artifact
``paper/artifacts/runtime-optimization/chain_shared_memory_scorer.json``.

Differences from the profile script, all deliberate:

* no assertion about the device-to-host transfer count. The observed count is
  recorded per iteration for each arm; the checked statement is the product one
  -- the shared-memory arm adds no host transfers relative to baseline, i.e.
  ``added_dtoh_calls == 0``.
* ``total_scored`` is still checked, because it is a property of the fixture and
  not of the card: ``xlog-induce`` computes it as
  ``len(Topology::ALL) * candidate_count**2`` (crates/xlog-induce/src/lib.rs), so
  the two candidate relations of this fixture give 4 * 2**2 = 16. The expected
  value is derived from the fixture, never written as a constant.
* every timed iteration is emitted, so median, min, max and the full sample are
  all in the artifact. The published record explicitly named the absence of
  per-iteration times as its own weakness.
* a provenance block (repository, runner hash, hardware, software, gate state)
  is written. The published record had no device provenance at all.

A failing arm is recorded as a failure and turns ``comparison_acceptable`` false;
it is never replaced by another execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
BENCHMARK = "chain_shared_memory_scorer"

CHAIN_SMEM_ENV = "XLOG_ILP_EXACT_CHAIN_SMEM"
CHAIN_SMEM_MIN_ROWS_ENV = "XLOG_ILP_EXACT_CHAIN_SMEM_MIN_ROWS"
# crates/xlog-cuda/src/provider/ilp_exact.rs:
# DEFAULT_ILP_EXACT_CHAIN_SMEM_MIN_ROWS = 256, parsed with u32 parse().unwrap_or(default).
DEFAULT_CHAIN_SMEM_MIN_ROWS = 256
U32_MAX = 2**32 - 1

# crates/xlog-induce/src/types.rs: Topology has exactly four arms
# (Chain, Star, Fanout, Fanin) and lib.rs derives total_scored = 4 * |C|**2.
TOPOLOGY_COUNT = 4

# paper/artifacts/runtime-optimization/chain_shared_memory_scorer.json, "protocol":
# warmup_iterations = 3, timed_iterations = 12. The README table for the same file
# repeats "12 timed, 3 warm-up".
DEFAULT_WARMUP = 3
DEFAULT_ITERATIONS = 12

# Fixture sizes from the published artifact's "fixtures" block.
DEFAULT_HOT_ROWS = 768
DEFAULT_HOT_QUERIES = 32
DEFAULT_SMALL_ROWS = 32
DEFAULT_SMALL_QUERIES = 8

CANDIDATE_RELATIONS = ("p_B", "p_C")
HEAD_RELATION = "p_A"
K_PER_TOPOLOGY = 2
PROGRAM_DEVICE = 0
PROGRAM_MEMORY_MB = 256

# Verbatim from scripts/measure_chain_shared_memory_profile.py: changing this text
# would change the fixture and break comparability with the published numbers.
SOURCE = """
    pred p_A(u64, u64).
    pred p_B(u64, u64).
    pred p_C(u64, u64).
    learnable(W_chain_p_A)  :: p_A(X, Y) :- bL(X, Z), bR(Z, Y).
    learnable(W_star_p_A)   :: p_A(X, Y) :- bL(X, Y), bR(X, Y).
    learnable(W_fanout_p_A) :: p_A(X, Y) :- bL(X, Z), bR(X, Y).
    learnable(W_fanin_p_A)  :: p_A(X, Y) :- bL(X, Y), bR(Z, Y).
"""

# python/tests/contract_docs/bounded-exact-induction.md, "Profile-Gated Chain
# Shared Memory": recorded as observations, never asserted.
CONTRACT_MIN_SPEEDUP = 1.2
CONTRACT_MAX_SMALL_REGRESSION_PERCENT = 5.0


# --------------------------------------------------------------------------
# Pure helpers -- everything below imports without CUDA and is covered by
# --self-test.
# --------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_total_scored(candidate_count: int) -> int:
    """Patterns the engine scores for ``candidate_count`` candidate relations."""
    if candidate_count <= 0:
        raise ValueError(f"candidate_count must be positive, got {candidate_count}")
    return TOPOLOGY_COUNT * candidate_count**2


def effective_min_rows(raw: str | None) -> int:
    """Mirror ``ilp_exact_chain_smem_min_rows()``: bad or unset input means the default."""
    if raw is None:
        return DEFAULT_CHAIN_SMEM_MIN_ROWS
    text = raw.strip()
    if not text.isdigit():
        return DEFAULT_CHAIN_SMEM_MIN_ROWS
    value = int(text)
    if value > U32_MAX:
        return DEFAULT_CHAIN_SMEM_MIN_ROWS
    return value


def aggregate_samples(samples: Sequence[float]) -> dict[str, Any]:
    """Median, min, max and the full per-iteration sample for one arm."""
    if not samples:
        raise ValueError("no timed iterations were recorded")
    return {
        "iterations": len(samples),
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "seconds_per_iteration": list(samples),
    }


def speedup_ratio(baseline_median: float | None, smem_median: float | None) -> float | None:
    """Baseline median over shared-memory median; ``None`` when it is undefined."""
    if baseline_median is None or smem_median is None or smem_median <= 0.0:
        return None
    return baseline_median / smem_median


def regression_percent(baseline_median: float | None, smem_median: float | None) -> float | None:
    """``(smem - baseline) / baseline * 100``; negative means the smem arm is faster."""
    if baseline_median is None or smem_median is None or baseline_median <= 0.0:
        return None
    return (smem_median - baseline_median) / baseline_median * 100.0


def summarize_dtoh_calls(counts: Sequence[int]) -> dict[str, Any]:
    """Record the observed device-to-host counts; assert nothing about them."""
    distinct = sorted(set(counts))
    return {
        "per_iteration": list(counts),
        "distinct_observed": distinct,
        "constant_across_iterations": len(distinct) <= 1,
        "value": max(counts) if counts else None,
    }


def added_dtoh_calls(
    baseline: Mapping[str, Any] | None, smem: Mapping[str, Any] | None
) -> int | None:
    """Difference between the two arms' transfer counts; never raises."""
    if not isinstance(baseline, Mapping) or not isinstance(smem, Mapping):
        return None
    baseline_value = baseline.get("value")
    smem_value = smem.get("value")
    if not isinstance(baseline_value, int) or not isinstance(smem_value, int):
        return None
    return smem_value - baseline_value


def normalize_signature(signature: Any) -> Any:
    """JSON turns tuples into lists; compare arms on the same shape."""
    if isinstance(signature, (list, tuple)):
        return [normalize_signature(item) for item in signature]
    return signature


def signatures_match(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return normalize_signature(left) == normalize_signature(right)


def arm_median(arm: Mapping[str, Any] | None) -> float | None:
    if not isinstance(arm, Mapping):
        return None
    value = arm.get("median_seconds")
    return value if isinstance(value, (int, float)) else None


def build_fixture_block(
    role: str,
    rows: int,
    queries: int,
    baseline: Mapping[str, Any],
    smem: Mapping[str, Any],
) -> dict[str, Any]:
    """One fixture's A/B pair in the shape of the published artifact."""
    baseline_median = arm_median(baseline)
    smem_median = arm_median(smem)
    return {
        "role": role,
        "rows_per_candidate": rows,
        "query_pairs_positive": queries,
        "query_pairs_negative": queries,
        "topology": "chain",
        "baseline": dict(baseline),
        "chain_smem": dict(smem),
        "parity": signatures_match(
            baseline.get("result_signature"), smem.get("result_signature")
        ),
        "speedup_ratio": speedup_ratio(baseline_median, smem_median),
        "median_delta_percent": regression_percent(baseline_median, smem_median),
        "delta_sign_note": (
            "median_delta_percent is (smem_median - baseline_median) / baseline_median "
            "* 100 -- the same quantity the profile runner calls regression_percent. "
            "A negative value means the shared-memory arm is faster."
        ),
        "arms_complete": baseline.get("status") == "ok" and smem.get("status") == "ok",
    }


def build_claim_checks(
    chain_hot: Mapping[str, Any],
    small_control: Mapping[str, Any],
    added_calls: int | None,
) -> dict[str, Any]:
    """Contract thresholds recorded as observations, not as assertions."""
    hot_speedup = chain_hot.get("speedup_ratio")
    control_delta = small_control.get("median_delta_percent")
    return {
        "chain_hot_speedup_ratio": hot_speedup,
        "chain_hot_speedup_meets_contract_min": (
            None if hot_speedup is None else hot_speedup >= CONTRACT_MIN_SPEEDUP
        ),
        "contract_min_speedup": CONTRACT_MIN_SPEEDUP,
        "small_control_median_delta_percent": control_delta,
        "small_control_within_contract_guard": (
            None
            if control_delta is None
            else control_delta <= CONTRACT_MAX_SMALL_REGRESSION_PERCENT
        ),
        "contract_max_small_regression_percent": CONTRACT_MAX_SMALL_REGRESSION_PERCENT,
        "chain_hot_parity": chain_hot.get("parity"),
        "small_control_parity": small_control.get("parity"),
        "added_dtoh_calls": added_calls,
        "shared_memory_adds_no_host_transfers": (
            None if added_calls is None else added_calls == 0
        ),
        "source": "python/tests/contract_docs/bounded-exact-induction.md",
    }


def comparison_is_acceptable(
    chain_hot: Mapping[str, Any],
    small_control: Mapping[str, Any],
    added_calls: int | None,
) -> bool:
    """Measurement validity only; the speed claim itself is reported, not gated."""
    for fixture in (chain_hot, small_control):
        if not fixture.get("arms_complete"):
            return False
        if not fixture.get("parity"):
            return False
        for arm_name in ("baseline", "chain_smem"):
            arm = fixture.get(arm_name)
            if not isinstance(arm, Mapping):
                return False
            if not arm.get("signature_stable_across_iterations"):
                return False
            if not arm.get("total_scored_matches_fixture"):
                return False
    return added_calls == 0


def cpu_model_name(cpuinfo: str) -> str:
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"model name", "Hardware", "Processor"}:
            model = value.strip()
            if model:
                return model
    raise RuntimeError("/proc/cpuinfo does not identify the CPU model")


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


def find_repository_root(start: Path) -> Path | None:
    """Nearest ancestor holding ``.git``; ``None`` when the runner is outside a checkout."""
    for directory in (start, *start.parents):
        if (directory / ".git").exists():
            return directory
    return None


def unavailable(command: Sequence[str], diagnostic: str, **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "command": shlex.join(str(part) for part in command),
        "status": "UNAVAILABLE",
        "diagnostic": diagnostic,
    }
    record.update(extra)
    return record


def stderr_tail(text: str, lines: int = 20) -> str:
    stripped = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(stripped[-lines:])


# --------------------------------------------------------------------------
# Probes -- every one records UNAVAILABLE with its stderr instead of passing.
# --------------------------------------------------------------------------


def run_probe(
    argv: Sequence[str], cwd: Path | None = None, timeout_s: int = 60
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(part) for part in argv],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return unavailable(argv, f"{type(error).__name__}: {error}")
    if completed.returncode != 0:
        return unavailable(
            argv,
            f"command exited with status {completed.returncode}",
            returncode=completed.returncode,
            stderr=stderr_tail(completed.stderr or ""),
        )
    return {
        "command": shlex.join(str(part) for part in argv),
        "status": "ok",
        "output": (completed.stdout or "").strip().splitlines(),
    }


def probe_output_line(probe: Mapping[str, Any]) -> str | None:
    output = probe.get("output")
    if probe.get("status") == "ok" and isinstance(output, list) and output:
        return output[0]
    return None


def repository_state(repo: Path | None, allow_dirty: bool) -> dict[str, Any]:
    if repo is None:
        diagnostic = (
            "no .git directory above the runner; run it from inside the xlog checkout "
            "(its committed home is scripts/measure_chain_shared_memory.py)"
        )
        if not allow_dirty:
            raise RuntimeError(
                f"repository state is unavailable and --allow-dirty was not given: {diagnostic}"
            )
        return {"status": "UNAVAILABLE", "diagnostic": diagnostic}
    head = run_probe(("git", "-C", str(repo), "rev-parse", "HEAD"))
    status = run_probe(("git", "-C", str(repo), "status", "--porcelain"))
    state: dict[str, Any] = {
        "path": str(repo),
        "commit": probe_output_line(head),
        "commit_probe": head,
        "branch": probe_output_line(
            run_probe(("git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"))
        ),
        "remote": probe_output_line(
            run_probe(("git", "-C", str(repo), "remote", "get-url", "origin"))
        ),
        "commit_subject": probe_output_line(
            run_probe(("git", "-C", str(repo), "log", "-1", "--format=%s"))
        ),
    }
    if status.get("status") != "ok":
        state["dirty"] = None
        state["status_probe"] = status
        if not allow_dirty:
            raise RuntimeError(
                "git status is unavailable and --allow-dirty was not given: "
                f"{status.get('diagnostic')}"
            )
        return state
    dirty_lines = [line for line in status.get("output", []) if line.strip()]
    state["dirty"] = bool(dirty_lines)
    state["dirty_entries"] = dirty_lines
    if dirty_lines and not allow_dirty:
        raise RuntimeError(
            "official measurement requires a clean checkout; commit changes or pass --allow-dirty"
        )
    return state


def cgroup_cpu_quota_state() -> dict[str, Any]:
    """Both cgroup generations, each recorded even when absent."""
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    state: dict[str, Any] = {
        "cores": None,
        "source": "none",
        "cgroup_v2_present": cpu_max.is_file(),
        "cgroup_v1_present": quota_path.is_file() and period_path.is_file(),
    }
    try:
        if cpu_max.is_file():
            parts = cpu_max.read_text(encoding="utf-8").split()
            if len(parts) != 2:
                state["diagnostic"] = f"invalid cgroup v2 cpu.max: {parts}"
                return state
            state["cores"] = parse_cpu_quota_cores(parts[0], parts[1])
            state["source"] = "cgroup_v2"
            return state
        if quota_path.is_file() and period_path.is_file():
            state["cores"] = parse_cpu_quota_cores(
                quota_path.read_text(encoding="utf-8").strip(),
                period_path.read_text(encoding="utf-8").strip(),
            )
            state["source"] = "cgroup_v1"
            return state
    except (OSError, RuntimeError, ValueError) as error:
        state["diagnostic"] = f"{type(error).__name__}: {error}"
    return state


def host_memory_state() -> dict[str, Any]:
    if not hasattr(os, "sysconf"):
        return {
            "bytes": None,
            "status": "UNAVAILABLE",
            "diagnostic": "os.sysconf is absent on this platform",
        }
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError) as error:
        return {
            "bytes": None,
            "status": "UNAVAILABLE",
            "diagnostic": f"{type(error).__name__}: {error}",
        }
    return {"bytes": page_size * physical_pages, "status": "ok"}


def cpu_state() -> dict[str, Any]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return {
            "model": None,
            "status": "UNAVAILABLE",
            "diagnostic": "/proc/cpuinfo is absent on this platform",
            "platform_processor": platform.processor() or None,
        }
    try:
        return {"model": cpu_model_name(cpuinfo.read_text(encoding="utf-8")), "status": "ok"}
    except (OSError, RuntimeError) as error:
        return {
            "model": None,
            "status": "UNAVAILABLE",
            "diagnostic": f"{type(error).__name__}: {error}",
        }


def hardware_state() -> dict[str, Any]:
    return {
        "gpu": run_probe(
            (
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            )
        ),
        "gpu_compute_capability": run_probe(
            ("nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader")
        ),
        "cpu": cpu_state(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_quota_cores": cgroup_cpu_quota_state(),
        "host_memory": host_memory_state(),
        "platform": platform.platform(),
        "device_provenance": "recorded by this runner from nvidia-smi at measurement time",
    }


def gate_state(env: Mapping[str, str]) -> dict[str, Any]:
    """The two gate variables as inherited, before this runner drives the A/B one."""
    raw_min_rows = env.get(CHAIN_SMEM_MIN_ROWS_ENV)
    return {
        CHAIN_SMEM_ENV: {
            "inherited_raw": env.get(CHAIN_SMEM_ENV),
            "default_when_unset": True,
            "driven_by_runner": True,
            "note": "set to 1 for the chain_smem arm and 0 for the baseline arm, then restored",
        },
        CHAIN_SMEM_MIN_ROWS_ENV: {
            "inherited_raw": raw_min_rows,
            "effective_min_rows": effective_min_rows(raw_min_rows),
            "default_when_unset": DEFAULT_CHAIN_SMEM_MIN_ROWS,
            "driven_by_runner": False,
        },
    }


def import_for_provenance(name: str) -> tuple[Any, dict[str, Any] | None]:
    """Import a module for version recording; a failure is returned, not raised."""
    try:
        return importlib.import_module(name), None
    except Exception as error:  # noqa: BLE001 - recorded in the artifact, not hidden
        return None, {"kind": type(error).__name__, "diagnostic": str(error)}


def software_state(repo: Path | None) -> dict[str, Any]:
    """Version provenance. Each module is imported on its own, so one missing
    package never erases the other's version."""
    torch_module, torch_error = import_for_provenance("torch")
    pyxlog_module, pyxlog_error = import_for_provenance("pyxlog")
    state: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "nvcc": run_probe(("nvcc", "--version"), cwd=repo),
        "nvidia_smi": run_probe(("nvidia-smi", "--version"), cwd=repo),
    }
    if torch_module is None:
        state["torch"] = {"status": "UNAVAILABLE", **(torch_error or {})}
    else:
        cuda_available = bool(torch_module.cuda.is_available())
        state["torch"] = {
            "status": "ok",
            "version": getattr(torch_module, "__version__", None),
            "cuda_build_version": getattr(getattr(torch_module, "version", None), "cuda", None),
            "cuda_available": cuda_available,
            "device_name": (
                torch_module.cuda.get_device_name(PROGRAM_DEVICE) if cuda_available else None
            ),
            "device_capability": (
                list(torch_module.cuda.get_device_capability(PROGRAM_DEVICE))
                if cuda_available
                else None
            ),
        }
    if pyxlog_module is None:
        state["pyxlog"] = {"status": "UNAVAILABLE", **(pyxlog_error or {})}
    else:
        state["pyxlog"] = {
            "status": "ok",
            "version": getattr(pyxlog_module, "__version__", None),
            "path": getattr(pyxlog_module, "__file__", None),
        }
    return state


# --------------------------------------------------------------------------
# Measurement -- torch and pyxlog are imported here, never at module import,
# so --self-test runs on a machine without CUDA.
# --------------------------------------------------------------------------


def load_measurement_modules() -> tuple[Any, Any, Any]:
    import torch

    import pyxlog
    from pyxlog.ilp import induce_exact

    return torch, pyxlog, induce_exact


def build_request(
    torch_module: Any, pyxlog_module: Any, rows: int, queries: int
) -> tuple[Any, dict[str, Any]]:
    """Fixture from scripts/measure_chain_shared_memory_profile.py, unchanged.

    The first query argument appears in every left row while the second matches
    only the final right row, so the chain predicate scans the whole right
    relation before finding coverage for each positive query.
    """

    def tensor(values: list[int]) -> Any:
        return torch_module.tensor(values, dtype=torch_module.int64, device="cuda")

    prog = pyxlog_module.IlpProgramFactory.compile(
        SOURCE, device=PROGRAM_DEVICE, memory_mb=PROGRAM_MEMORY_MB
    )
    left_arg0 = [1] * rows
    left_arg1 = list(range(10_000, 10_000 + rows))
    right_arg0 = list(range(10_000, 10_000 + rows))
    right_arg1 = list(range(20_000, 20_000 + rows))
    prog.put_relation("p_B", [tensor(left_arg0), tensor(left_arg1)])
    prog.put_relation("p_C", [tensor(right_arg0), tensor(right_arg1)])
    kwargs: dict[str, Any] = dict(
        head_relation=HEAD_RELATION,
        candidate_relations=list(CANDIDATE_RELATIONS),
        positive_arg0=tensor([1] * queries),
        positive_arg1=tensor([20_000 + rows - 1] * queries),
        negative_arg0=tensor([2] * queries),
        negative_arg1=tensor([999_999] * queries),
        k_per_topology=K_PER_TOPOLOGY,
        deterministic=True,
    )
    return prog, kwargs


def candidate_signature(result: Any) -> list[list[Any]]:
    return [
        [
            candidate.topology,
            candidate.left_relation,
            candidate.right_relation,
            candidate.positives_covered,
            candidate.negatives_covered,
            candidate.local_rank,
        ]
        for candidate in result.candidates
    ]


def measure_arm(
    rows: int,
    queries: int,
    iterations: int,
    warmup: int,
    *,
    chain_smem_enabled: bool,
) -> dict[str, Any]:
    """One arm of one fixture. Failures are returned, not raised."""
    expected_scored = expected_total_scored(len(CANDIDATE_RELATIONS))
    arm: dict[str, Any] = {
        "chain_smem": chain_smem_enabled,
        "gate_env": {CHAIN_SMEM_ENV: "1" if chain_smem_enabled else "0"},
        "warmup": warmup,
        "requested_iterations": iterations,
        "total_scored_expected": expected_scored,
        "total_scored_expected_from": (
            f"{TOPOLOGY_COUNT} topologies * "
            f"{len(CANDIDATE_RELATIONS)} candidate relations squared"
        ),
    }
    previous = os.environ.get(CHAIN_SMEM_ENV)
    os.environ[CHAIN_SMEM_ENV] = "1" if chain_smem_enabled else "0"
    try:
        torch_module, pyxlog_module, induce_exact = load_measurement_modules()
        prog, kwargs = build_request(torch_module, pyxlog_module, rows, queries)
        for _ in range(warmup):
            induce_exact(prog, backend="native", **kwargs)
        torch_module.cuda.synchronize()

        samples: list[float] = []
        dtoh_counts: list[int] = []
        scored_counts: list[int] = []
        signature: list[list[Any]] | None = None
        signature_stable = True
        for _ in range(iterations):
            prog.reset_d2h_transfer_count()
            start = time.perf_counter()
            result = induce_exact(prog, backend="native", **kwargs)
            torch_module.cuda.synchronize()
            elapsed = time.perf_counter() - start
            dtoh_counts.append(int(prog.d2h_transfer_count()))
            scored_counts.append(int(result.total_scored))
            iteration_signature = candidate_signature(result)
            if signature is not None and iteration_signature != signature:
                signature_stable = False
            signature = iteration_signature
            samples.append(elapsed)
    except Exception as error:  # noqa: BLE001 - an arm failure is recorded, never hidden
        arm["status"] = "failed"
        arm["error"] = {"kind": type(error).__name__, "diagnostic": str(error)}
        arm["signature_stable_across_iterations"] = False
        arm["total_scored_matches_fixture"] = False
        return arm
    finally:
        if previous is None:
            os.environ.pop(CHAIN_SMEM_ENV, None)
        else:
            os.environ[CHAIN_SMEM_ENV] = previous

    arm.update(aggregate_samples(samples))
    dtoh = summarize_dtoh_calls(dtoh_counts)
    arm["status"] = "ok"
    arm["rows_per_candidate"] = rows
    arm["query_pairs_positive"] = queries
    arm["query_pairs_negative"] = queries
    arm["dtoh_calls"] = dtoh["value"]
    arm["dtoh_calls_detail"] = dtoh
    arm["total_scored_observed"] = sorted(set(scored_counts))
    arm["total_scored_matches_fixture"] = set(scored_counts) == {expected_scored}
    arm["result_signature"] = signature
    arm["signature_stable_across_iterations"] = signature_stable
    arm["candidates_returned"] = len(signature or [])
    return arm


def measure_fixture(
    role: str, rows: int, queries: int, iterations: int, warmup: int
) -> dict[str, Any]:
    baseline = measure_arm(rows, queries, iterations, warmup, chain_smem_enabled=False)
    smem = measure_arm(rows, queries, iterations, warmup, chain_smem_enabled=True)
    return build_fixture_block(role, rows, queries, baseline, smem)


# --------------------------------------------------------------------------
# Artifact assembly
# --------------------------------------------------------------------------


def build_artifact(
    *,
    chain_hot: Mapping[str, Any],
    small_control: Mapping[str, Any],
    repository: Mapping[str, Any],
    runner: Mapping[str, Any],
    hardware: Mapping[str, Any],
    software: Mapping[str, Any],
    protocol: Mapping[str, Any],
    generated_at_utc: str,
) -> dict[str, Any]:
    baseline_dtoh = chain_hot.get("baseline", {}).get("dtoh_calls_detail")
    smem_dtoh = chain_hot.get("chain_smem", {}).get("dtoh_calls_detail")
    added_calls = added_dtoh_calls(baseline_dtoh, smem_dtoh)
    claim_checks = build_claim_checks(chain_hot, small_control, added_calls)
    acceptable = comparison_is_acceptable(chain_hot, small_control, added_calls)
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "generated_at_utc": generated_at_utc,
        "claim": (
            "Profile-gated shared-memory chain scorer: measured speedup on a chain-hot "
            "fixture, with the small-input control measured below the gate threshold."
        ),
        "paper_section": "sections/10_evaluation.tex, Runtime Optimization (sec:runtime-eval)",
        "kind": "single-system ablation (xlog against its own baseline)",
        "repository": dict(repository),
        "runner": dict(runner),
        "hardware": dict(hardware),
        "software": dict(software),
        "code_version": {
            "release_line": "see repository.commit and software.pyxlog.version",
            "kernel": (
                "crates/xlog-cuda/kernels/ilp_exact.cu "
                "(ilp_exact_score_chain_smem, ilp_exact_score_chain_smem_u32)"
            ),
            "gate_source": "crates/xlog-cuda/src/provider/ilp_exact.rs",
            "published_reference": {
                "release_line": "v0.8.6",
                "measurement_commit": "ce78e32f9c740187fb893f169632e1d9e67f1c25",
                "date": "2026-05-19",
            },
        },
        "how_to_reproduce": {
            "runner": "scripts/measure_chain_shared_memory.py",
            "command": (
                "python scripts/measure_chain_shared_memory.py --output "
                "paper/artifacts/runtime-optimization/chain_shared_memory_scorer.json"
            ),
            "path": (
                'pyxlog induce_exact(backend="native") -> xlog_induce::induce_exact -> '
                "CudaKernelProvider::ilp_exact_score"
            ),
            "gate": (
                f"{CHAIN_SMEM_ENV} (default on) and {CHAIN_SMEM_MIN_ROWS_ENV} "
                f"(default {DEFAULT_CHAIN_SMEM_MIN_ROWS}), "
                "crates/xlog-cuda/src/provider/ilp_exact.rs"
            ),
            "ab_method": (
                "the same fixture is run twice in one process, once with the shared-memory "
                "path enabled and once disabled"
            ),
        },
        "fixtures": {
            "chain_hot": {
                "role": chain_hot.get("role"),
                "rows_per_candidate": chain_hot.get("rows_per_candidate"),
                "query_pairs_positive": chain_hot.get("query_pairs_positive"),
                "query_pairs_negative": chain_hot.get("query_pairs_negative"),
                "topology": "chain",
            },
            "small_control": {
                "role": small_control.get("role"),
                "rows_per_candidate": small_control.get("rows_per_candidate"),
                "query_pairs_positive": small_control.get("query_pairs_positive"),
                "query_pairs_negative": small_control.get("query_pairs_negative"),
                "topology": "chain",
            },
            "source_sha256": sha256_bytes(SOURCE.encode()),
        },
        "protocol": dict(protocol),
        "measurements": {
            "chain_hot": dict(chain_hot),
            "small_control": dict(small_control),
            "transfer_budget": {
                "baseline_dtoh_calls": (baseline_dtoh or {}).get("value"),
                "chain_smem_dtoh_calls": (smem_dtoh or {}).get("value"),
                "added_dtoh_calls": added_calls,
                "note": (
                    "absolute counts are observations of this device and driver; only the "
                    "difference between the arms is a product statement"
                ),
            },
            "fallback": {
                "non_chain_uses_baseline_logic": (
                    "not measured by this runner; stated by "
                    "python/tests/contract_docs/bounded-exact-induction.md"
                )
            },
        },
        "results": claim_checks,
        "comparison_acceptable": acceptable,
    }


def build_protocol(args: argparse.Namespace, env: Mapping[str, str]) -> dict[str, Any]:
    return {
        "warmup_iterations": args.warmup,
        "timed_iterations": args.iterations,
        "aggregation": "median over the timed iterations",
        "dispersion_recorded": (
            "every timed iteration is emitted per arm, alongside median, min and max"
        ),
        "timing": (
            "time.perf_counter around induce_exact with torch.cuda.synchronize after each "
            "call; warm-up iterations are untimed and followed by one synchronize"
        ),
        "execution_order": [
            "small_control/baseline",
            "small_control/chain_smem",
            "chain_hot/baseline",
            "chain_hot/chain_smem",
        ],
        "execution_order_note": (
            "kept from scripts/measure_chain_shared_memory_profile.py, which produced the "
            "published numbers; each arm compiles its own program"
        ),
        "ab_method": (
            "the same fixture is run twice in one process, once with the shared-memory path "
            "enabled and once disabled"
        ),
        "gate": gate_state(env),
        "hardware_assertions": (
            "none. Device-to-host transfer counts and total_scored are recorded; only "
            "added_dtoh_calls (a difference between arms) and total_scored (a property of "
            "the fixture) participate in comparison_acceptable"
        ),
        "published_protocol_reference": {
            "warmup_iterations": 3,
            "timed_iterations": 12,
            "source": "paper/artifacts/runtime-optimization/chain_shared_memory_scorer.json",
        },
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hot-rows", type=int, default=DEFAULT_HOT_ROWS)
    parser.add_argument("--hot-queries", type=int, default=DEFAULT_HOT_QUERIES)
    parser.add_argument("--small-rows", type=int, default=DEFAULT_SMALL_ROWS)
    parser.add_argument("--small-queries", type=int, default=DEFAULT_SMALL_QUERIES)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup must not be negative")
    for name in ("iterations", "hot_rows", "hot_queries", "small_rows", "small_queries"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def write_artifact(artifact: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def self_test() -> None:
    # 1. Aggregation keeps every iteration and reports the median, not the mean.
    aggregate = aggregate_samples([0.3, 0.1, 0.2, 0.4])
    assert aggregate["iterations"] == 4
    assert aggregate["median_seconds"] == 0.25
    assert aggregate["min_seconds"] == 0.1
    assert aggregate["max_seconds"] == 0.4
    assert aggregate["seconds_per_iteration"] == [0.3, 0.1, 0.2, 0.4]
    assert aggregate_samples([2.0, 1.0, 3.0])["median_seconds"] == 2.0
    try:
        aggregate_samples([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty sample list was accepted")

    # 2. The published numbers must come back out of the published medians.
    hot_baseline = 0.027507680497365072
    hot_smem = 0.004927040485199541
    control_baseline = 0.0011941255070269108
    control_smem = 0.0011783409863710403
    published_speedup = 5.58300273358745
    published_delta = -1.3218477088869953
    assert speedup_ratio(hot_baseline, hot_smem) == published_speedup
    assert round(speedup_ratio(hot_baseline, hot_smem), 2) == 5.58
    assert regression_percent(control_baseline, control_smem) == published_delta
    assert speedup_ratio(1.0, 0.0) is None
    assert speedup_ratio(None, 1.0) is None
    assert regression_percent(0.0, 1.0) is None
    assert regression_percent(1.0, None) is None

    # 3. Transfer counts: recorded, differenced, never fatal on their own.
    two_calls = summarize_dtoh_calls([2, 2, 2])
    one_call = summarize_dtoh_calls([1, 1, 1])
    mixed = summarize_dtoh_calls([1, 2, 1])
    assert two_calls["value"] == 2 and two_calls["constant_across_iterations"]
    assert mixed["distinct_observed"] == [1, 2]
    assert not mixed["constant_across_iterations"]
    assert added_dtoh_calls(two_calls, two_calls) == 0
    assert added_dtoh_calls(one_call, one_call) == 0
    assert added_dtoh_calls(two_calls, one_call) == -1
    assert added_dtoh_calls(one_call, two_calls) == 1
    assert added_dtoh_calls(summarize_dtoh_calls([]), two_calls) is None
    assert added_dtoh_calls(None, two_calls) is None
    assert added_dtoh_calls(two_calls, {"value": "two"}) is None

    # 4. Fixture-derived expectations, not constants.
    assert expected_total_scored(2) == 16
    assert expected_total_scored(3) == 36
    assert expected_total_scored(len(CANDIDATE_RELATIONS)) == 16
    try:
        expected_total_scored(0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive candidate count was accepted")

    # 5. Signature parity survives the tuple-to-list round trip through JSON.
    published_signature = [["chain", "p_B", "p_C", 32, 0, 0]]
    assert signatures_match(published_signature, [("chain", "p_B", "p_C", 32, 0, 0)])
    assert not signatures_match(published_signature, [["chain", "p_B", "p_C", 31, 0, 0]])
    assert not signatures_match(published_signature, None)
    assert not signatures_match(None, None)

    # 6. Gate parsing mirrors the Rust default handling.
    assert effective_min_rows(None) == DEFAULT_CHAIN_SMEM_MIN_ROWS
    assert effective_min_rows(" 512 ") == 512
    assert effective_min_rows("0") == 0
    assert effective_min_rows("-1") == DEFAULT_CHAIN_SMEM_MIN_ROWS
    assert effective_min_rows("many") == DEFAULT_CHAIN_SMEM_MIN_ROWS
    assert effective_min_rows(str(U32_MAX + 1)) == DEFAULT_CHAIN_SMEM_MIN_ROWS
    gate = gate_state({CHAIN_SMEM_MIN_ROWS_ENV: "512"})
    assert gate[CHAIN_SMEM_MIN_ROWS_ENV]["effective_min_rows"] == 512
    assert gate[CHAIN_SMEM_ENV]["inherited_raw"] is None

    # 7. Fixture assembly on the published measurements.
    def published_arm(enabled: bool, median: float, signature: list[list[Any]]) -> dict[str, Any]:
        return {
            "chain_smem": enabled,
            "status": "ok",
            "median_seconds": median,
            "dtoh_calls_detail": summarize_dtoh_calls([2, 2]),
            "result_signature": signature,
            "signature_stable_across_iterations": True,
            "total_scored_matches_fixture": True,
        }

    hot_block = build_fixture_block(
        "above the gate threshold",
        768,
        32,
        published_arm(False, hot_baseline, published_signature),
        published_arm(True, hot_smem, published_signature),
    )
    control_signature = [["chain", "p_B", "p_C", 8, 0, 0]]
    control_block = build_fixture_block(
        "below the gate threshold",
        32,
        8,
        published_arm(False, control_baseline, control_signature),
        published_arm(True, control_smem, control_signature),
    )
    assert hot_block["speedup_ratio"] == published_speedup
    assert hot_block["parity"] and control_block["parity"]
    assert control_block["median_delta_percent"] == published_delta
    assert hot_block["arms_complete"] and control_block["arms_complete"]
    assert comparison_is_acceptable(hot_block, control_block, 0)
    assert not comparison_is_acceptable(hot_block, control_block, 1)
    assert not comparison_is_acceptable(hot_block, control_block, None)

    checks = build_claim_checks(hot_block, control_block, 0)
    assert checks["chain_hot_speedup_meets_contract_min"] is True
    assert checks["small_control_within_contract_guard"] is True
    assert checks["shared_memory_adds_no_host_transfers"] is True

    # A failed arm poisons the comparison and is never substituted.
    failed_hot = build_fixture_block(
        "above the gate threshold",
        768,
        32,
        {
            "status": "failed",
            "error": {"kind": "RuntimeError", "diagnostic": "boom"},
            "signature_stable_across_iterations": False,
            "total_scored_matches_fixture": False,
        },
        published_arm(True, hot_smem, published_signature),
    )
    assert not failed_hot["arms_complete"]
    assert not comparison_is_acceptable(failed_hot, control_block, 0)
    assert failed_hot["speedup_ratio"] is None

    unstable_baseline = published_arm(False, hot_baseline, published_signature)
    unstable_baseline["signature_stable_across_iterations"] = False
    unstable = build_fixture_block(
        "above the gate threshold",
        768,
        32,
        unstable_baseline,
        published_arm(True, hot_smem, published_signature),
    )
    assert not comparison_is_acceptable(unstable, control_block, 0)

    # 8. The artifact skeleton carries every required key and survives JSON.
    artifact = build_artifact(
        chain_hot=hot_block,
        small_control=control_block,
        repository={"commit": "0" * 40, "dirty": False},
        runner={"path": "scripts/measure_chain_shared_memory.py", "sha256": "0" * 64},
        hardware={"gpu": {"status": "UNAVAILABLE", "diagnostic": "self-test"}},
        software={"python": platform.python_version()},
        protocol={"warmup_iterations": 3, "timed_iterations": 12},
        generated_at_utc="1970-01-01T00:00:00+00:00",
    )
    for key in (
        "schema_version",
        "benchmark",
        "generated_at_utc",
        "repository",
        "runner",
        "hardware",
        "software",
        "protocol",
        "claim",
        "paper_section",
        "kind",
        "code_version",
        "how_to_reproduce",
        "fixtures",
        "measurements",
        "results",
        "comparison_acceptable",
    ):
        assert key in artifact, f"artifact is missing {key}"
    assert artifact["comparison_acceptable"] is True
    assert artifact["measurements"]["transfer_budget"]["added_dtoh_calls"] == 0
    round_tripped = json.loads(json.dumps(artifact))
    assert round_tripped["measurements"]["chain_hot"]["speedup_ratio"] == published_speedup
    assert round_tripped["fixtures"]["chain_hot"]["rows_per_candidate"] == 768
    assert round_tripped["fixtures"]["small_control"]["rows_per_candidate"] == 32

    # 9. Host-probe parsers.
    assert cpu_model_name("processor: 0\nmodel name: Example CPU\n") == "Example CPU"
    try:
        cpu_model_name("processor: 0\n")
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing CPU model was accepted")
    assert parse_cpu_quota_cores("765000", "100000") == 7.65
    assert parse_cpu_quota_cores("max", "100000") is None
    assert parse_cpu_quota_cores("-1", "100000") is None
    quota_state = cgroup_cpu_quota_state()
    assert "cgroup_v2_present" in quota_state and "cgroup_v1_present" in quota_state
    assert "bytes" in host_memory_state()
    assert cpu_state().get("status") in {"ok", "UNAVAILABLE"}

    # 10. Missing binaries are recorded as UNAVAILABLE, never skipped.
    missing = run_probe(("xlog-self-test-missing-binary", "--version"))
    assert missing["status"] == "UNAVAILABLE"
    assert missing["diagnostic"]
    assert probe_output_line(missing) is None
    failing = run_probe(
        (sys.executable, "-c", "import sys; sys.stderr.write('nope\\n'); sys.exit(3)")
    )
    assert failing["status"] == "UNAVAILABLE"
    assert failing["returncode"] == 3
    assert failing["stderr"] == "nope"
    working = run_probe((sys.executable, "-c", "print('hello')"))
    assert probe_output_line(working) == "hello"

    # 10b. A missing package is recorded per module, and never raises.
    module, module_error = import_for_provenance("json")
    assert module is not None and module_error is None
    module, module_error = import_for_provenance("xlog_self_test_missing_module")
    assert module is None
    assert module_error is not None and module_error["kind"] == "ModuleNotFoundError"

    # 11. Repository discovery and the dirty-tree refusal.
    with tempfile.TemporaryDirectory() as raw_directory:
        directory = Path(raw_directory)
        nested = directory / "a" / "b"
        nested.mkdir(parents=True)
        assert find_repository_root(nested) is None
        (directory / ".git").mkdir()
        assert find_repository_root(nested) == directory
    try:
        repository_state(None, allow_dirty=False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing repository was accepted without --allow-dirty")
    assert repository_state(None, allow_dirty=True)["status"] == "UNAVAILABLE"

    # 12. CLI defaults follow the published protocol block.
    defaults = parse_args([])
    assert defaults.iterations == DEFAULT_ITERATIONS == 12
    assert defaults.warmup == DEFAULT_WARMUP == 3
    assert defaults.hot_rows == 768 and defaults.hot_queries == 32
    assert defaults.small_rows == 32 and defaults.small_queries == 8
    assert defaults.output is None and not defaults.allow_dirty
    validate_args(defaults)
    for bad in (["--iterations", "0"], ["--warmup", "-1"], ["--hot-rows", "-8"]):
        try:
            validate_args(parse_args(bad))
        except ValueError:
            continue
        raise AssertionError(f"invalid arguments were accepted: {bad}")

    print("chain shared-memory runner self-test passed")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        try:
            self_test()
        except AssertionError as error:
            print(f"self-test failed: {error}", file=sys.stderr)
            return 1
        return 0
    if args.output is None:
        raise ValueError("--output is required outside --self-test")
    validate_args(args)

    runner_path = Path(__file__).resolve()
    repo = find_repository_root(runner_path.parent)
    repository = repository_state(repo, args.allow_dirty)
    inherited_env = dict(os.environ)

    print("BEGIN small_control", flush=True)
    small_control = measure_fixture(
        "below the gate threshold (baseline scorer must run unchanged)",
        args.small_rows,
        args.small_queries,
        args.iterations,
        args.warmup,
    )
    print(f"END small_control arms_complete={small_control['arms_complete']}", flush=True)
    print("BEGIN chain_hot", flush=True)
    chain_hot = measure_fixture(
        "above the gate threshold",
        args.hot_rows,
        args.hot_queries,
        args.iterations,
        args.warmup,
    )
    print(f"END chain_hot arms_complete={chain_hot['arms_complete']}", flush=True)

    software = software_state(repo)

    try:
        relative_runner = str(runner_path.relative_to(repo)) if repo else str(runner_path)
    except ValueError:
        relative_runner = str(runner_path)

    artifact = build_artifact(
        chain_hot=chain_hot,
        small_control=small_control,
        repository=repository,
        runner={
            "path": relative_runner,
            "absolute_path": str(runner_path),
            "sha256": sha256_file(runner_path),
            "argv": list(sys.argv),
        },
        hardware=hardware_state(),
        software=software,
        protocol=build_protocol(args, inherited_env),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_artifact(artifact, args.output)
    print(f"WROTE {args.output}", flush=True)
    return 0 if artifact["comparison_acceptable"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"chain shared-memory runner failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
