"""Preflight checks for the public xlog release contract."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_UNSUPPORTED = 2

MIN_PYTHON = (3, 8)
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CheckResult:
    slug: str
    status: str
    message: str
    detail: str | None = None


def _build_runtime_env() -> tuple[dict[str, str], str | None]:
    env = os.environ.copy()

    try:
        ctypes.CDLL("libcuda.so")
        return env, None
    except OSError:
        pass

    wsl_cuda = Path("/usr/lib/wsl/lib/libcuda.so.1")
    if not wsl_cuda.exists():
        return env, None

    shim_dir = Path("/tmp/xlog-cuda-shim")
    shim_dir.mkdir(parents=True, exist_ok=True)
    for soname in ("libcuda.so", "libnvcuda.so"):
        link = shim_dir / soname
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(wsl_cuda)

    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{shim_dir}:{existing}" if existing else str(shim_dir)
    return env, f"WSL loader shim prepared at {shim_dir}"


@lru_cache(maxsize=1)
def _runtime_env() -> tuple[dict[str, str], str | None]:
    return _build_runtime_env()


def _ok(slug: str, message: str) -> CheckResult:
    return CheckResult(slug=slug, status="OK", message=message)


def _fail(slug: str, message: str, detail: str) -> CheckResult:
    return CheckResult(slug=slug, status="FAIL", message=message, detail=detail)


def _unsupported(slug: str, message: str, detail: str | None = None) -> CheckResult:
    return CheckResult(slug=slug, status="UNSUPPORTED", message=message, detail=detail)


def _run_version_command(command: list[str], slug: str, name: str) -> CheckResult:
    resolved = shutil.which(command[0])
    if resolved is None:
        return _fail(
            slug,
            f"{name} is missing",
            f"Install {name} and make sure `{command[0]}` is on PATH.",
        )

    proc = subprocess.run(
        [resolved, *command[1:]],
        cwd=ROOT,
        env=_runtime_env()[0],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip() or f"{name} exited with {proc.returncode}"
        return _fail(
            slug,
            f"{name} failed to run",
            stderr,
        )

    output = proc.stdout.strip() or proc.stderr.strip() or f"{name} is available"
    return _ok(slug, output.splitlines()[0])


def _check_platform() -> CheckResult:
    system = platform.system()
    machine = platform.machine()
    if system != "Linux" or machine != "x86_64":
        return _unsupported(
            "platform",
            f"xlog public release supports Linux x86_64 only (found {system} {machine}).",
            "Use a Linux x86_64 host for source builds, release binaries, and public examples.",
        )
    return _ok("platform", "Linux x86_64")


def _check_nvidia_smi() -> CheckResult:
    return _run_version_command(["nvidia-smi"], "nvidia-smi", "nvidia-smi")


def _check_nvcc() -> CheckResult:
    resolved = shutil.which("nvcc")
    if resolved is None:
        return _fail(
            "nvcc",
            "nvcc is missing",
            "Install CUDA Toolkit 12.x and ensure `nvcc` is on PATH.",
        )

    proc = subprocess.run(
        [resolved, "--version"],
        cwd=ROOT,
        env=_runtime_env()[0],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip() or "nvcc --version failed"
        return _fail(
            "nvcc",
            "nvcc --version failed",
            f"Check the CUDA Toolkit installation: {stderr}",
        )

    first_line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else "nvcc is available"
    return _ok("nvcc", first_line)


def _check_rust() -> CheckResult:
    rustc = shutil.which("rustc")
    cargo = shutil.which("cargo")
    if rustc is None or cargo is None:
        missing = [name for name, tool in (("rustc", rustc), ("cargo", cargo)) if tool is None]
        return _fail(
            "rust",
            f"{', '.join(missing)} is missing",
            "Install Rust with rustup so both `rustc` and `cargo` are available.",
        )

    rustc_version = subprocess.run(
        [rustc, "--version"],
        cwd=ROOT,
        env=_runtime_env()[0],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    cargo_version = subprocess.run(
        [cargo, "--version"],
        cwd=ROOT,
        env=_runtime_env()[0],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if rustc_version.returncode != 0 or cargo_version.returncode != 0:
        return _fail(
            "rust",
            "Rust toolchain failed to run",
            "Check the rustup installation and PATH configuration.",
        )

    rustc_line = rustc_version.stdout.strip().splitlines()[0]
    cargo_line = cargo_version.stdout.strip().splitlines()[0]
    return _ok("rust", f"{rustc_line}; {cargo_line}")


def _check_python() -> CheckResult:
    current = sys.version_info
    if current < MIN_PYTHON:
        return _fail(
            "python",
            f"Python {current.major}.{current.minor} is too old",
            f"Install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer.",
        )
    return _ok("python", f"Python {current.major}.{current.minor}.{current.micro}")


def _check_maturin() -> CheckResult:
    return _run_version_command(["maturin", "--version"], "maturin", "maturin")


def _check_cuda_loader() -> CheckResult:
    try:
        ctypes.CDLL("libcuda.so")
        _, cuda_shim_note = _runtime_env()
        if cuda_shim_note:
            return _ok("cuda-loader", f"libcuda.so resolves; {cuda_shim_note}")
        return _ok("cuda-loader", "libcuda.so resolves")
    except OSError:
        wsl_cuda = Path("/usr/lib/wsl/lib/libcuda.so.1")
        if wsl_cuda.exists():
            return _ok(
                "cuda-loader",
                f"WSL CUDA shim can be prepared from {wsl_cuda}",
            )
        return _fail(
            "cuda-loader",
            "libcuda.so is not resolvable",
            "Ensure the NVIDIA driver is installed and visible to the dynamic loader.",
        )


def _workflow_note(workflow: str) -> CheckResult:
    if workflow == "release":
        return _ok(
            "workflow",
            "release workflow requires `host-io` for xlog-cli and `maturin` for pyxlog packaging",
        )
    if workflow == "prob-cli":
        return _ok(
            "workflow",
            "prob-cli workflow requires `host-io` when building xlog-cli",
        )
    if workflow == "run-cli":
        return _ok("workflow", "run-cli workflow works with the default xlog-cli build")
    return _ok("workflow", "smoke workflow")


def evaluate(workflow: str) -> list[CheckResult]:
    results: list[CheckResult] = []

    platform_result = _check_platform()
    results.append(platform_result)
    if platform_result.status == "UNSUPPORTED":
        return results

    if workflow == "release":
        results.extend(
            [
                _check_rust(),
                _check_python(),
                _check_maturin(),
                _workflow_note(workflow),
            ]
        )
        return results

    results.extend(
        [
            _check_nvidia_smi(),
            _check_nvcc(),
            _check_rust(),
            _check_python(),
            _check_cuda_loader(),
            _workflow_note(workflow),
        ]
    )
    return results


def _overall_status(results: Iterable[CheckResult]) -> tuple[str, int]:
    statuses = [result.status for result in results]
    if any(status == "FAIL" for status in statuses):
        return "FAIL", EXIT_FAIL
    if any(status == "UNSUPPORTED" for status in statuses):
        return "UNSUPPORTED", EXIT_UNSUPPORTED
    return "SUPPORTED", EXIT_OK


def _print_human(results: list[CheckResult], workflow: str) -> None:
    overall, _ = _overall_status(results)
    print(f"{overall}: xlog public setup doctor ({workflow})")
    for result in results:
        line = f"[{result.status}] {result.slug}: {result.message}"
        print(line)
        if result.detail:
            print(f"  {result.detail}")
    if overall == "SUPPORTED":
        print("SUPPORTED: public setup contract satisfied.")


def _print_json(results: list[CheckResult], workflow: str) -> None:
    overall, exit_code = _overall_status(results)
    payload = {
        "overall_status": overall,
        "exit_code": exit_code,
        "workflow": workflow,
        "checks": [asdict(result) for result in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preflight doctor for the public xlog setup.")
    parser.add_argument(
        "--workflow",
        choices=("smoke", "run-cli", "prob-cli", "release"),
        default="smoke",
        help="Select the install or runtime workflow to validate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-readable output.",
    )
    args = parser.parse_args(argv)

    results = evaluate(args.workflow)
    if args.json:
        _print_json(results, args.workflow)
    else:
        _print_human(results, args.workflow)

    _, exit_code = _overall_status(results)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
