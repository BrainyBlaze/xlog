import argparse
import ctypes
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _build_runtime_env() -> dict[str, str]:
    env = os.environ.copy()

    # WSL often exposes only libcuda.so.1; cudarc expects libcuda.so/libnvcuda.so.
    try:
        ctypes.CDLL("libcuda.so")
        return env
    except OSError:
        pass

    wsl_cuda = Path("/usr/lib/wsl/lib/libcuda.so.1")
    if not wsl_cuda.exists():
        return env

    shim_dir = Path("/tmp/xlog-cuda-shim")
    shim_dir.mkdir(parents=True, exist_ok=True)
    for soname in ("libcuda.so", "libnvcuda.so"):
        link = shim_dir / soname
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(wsl_cuda)

    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = (
        f"{shim_dir}:{existing}" if existing else str(shim_dir)
    )
    print(f"INFO: Added CUDA loader shim at {shim_dir}")
    return env


RUNTIME_ENV = _build_runtime_env()


def run_cmd(cmd, cwd=None, timeout_sec=None):
    print("+", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=RUNTIME_ENV, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"Command timed out after {exc.timeout}s: {' '.join(cmd)}")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def run_neural_cmd(cmd, cwd: Path, timeout_sec: int = 300) -> None:
    print("+", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=RUNTIME_ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"Command timed out after {exc.timeout}s: {' '.join(cmd)}")
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    if proc.returncode == 0:
        return

    combined = f"{proc.stdout}\n{proc.stderr}".lower()
    if (
        "dataset missing" in combined
        or "data missing" in combined
        or "missing dependency" in combined
    ):
        print(f"SKIP: {cwd / 'train.py'} (dataset/dependency unavailable)")
        return

    raise SystemExit(proc.returncode)


def _can_import_pyxlog() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", "import pyxlog"],
        cwd=ROOT,
        env=RUNTIME_ENV,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return probe.returncode == 0


def _prepend_env_path(var: str, path: Path) -> None:
    current = RUNTIME_ENV.get(var, "")
    RUNTIME_ENV[var] = f"{path}:{current}" if current else str(path)


def ensure_pyxlog_available(mode: str) -> None:
    if mode != "release" and _can_import_pyxlog():
        return

    cargo_cmd = ["cargo", "build", "-q", "-p", "pyxlog", "--features", "host-io"]
    if mode == "release":
        cargo_cmd.append("--release")
    run_cmd(cargo_cmd, cwd=ROOT)

    target_dir = ROOT / "target" / ("release" if mode == "release" else "debug")
    linux_lib = target_dir / "libpyxlog.so"
    mac_lib = target_dir / "libpyxlog.dylib"
    native_lib = linux_lib if linux_lib.exists() else mac_lib
    if not native_lib.exists():
        raise SystemExit(f"Unable to locate built pyxlog native library in {target_dir}")

    # pyxlog is a package (`pyxlog/__init__.py`) whose native module is
    # `pyxlog._native`; stage that package shape under target/{profile} so
    # examples can import the just-built workspace artifact without installing
    # a wheel.
    stale_top_level = target_dir / "pyxlog.so"
    if stale_top_level.exists() or stale_top_level.is_symlink():
        stale_top_level.unlink()

    source_pkg = ROOT / "crates" / "pyxlog" / "python" / "pyxlog"
    staged_pkg = target_dir / "pyxlog"
    staged_pkg.mkdir(exist_ok=True)
    for child in source_pkg.iterdir():
        dest = staged_pkg / child.name
        if dest.exists() or dest.is_symlink():
            if dest.is_dir() and not dest.is_symlink():
                continue
            dest.unlink()
        if not dest.exists():
            dest.symlink_to(child, target_is_directory=child.is_dir())

    native_name = "_native.so" if linux_lib.exists() else "_native.dylib"
    native_dest = staged_pkg / native_name
    if native_dest.exists() or native_dest.is_symlink():
        native_dest.unlink()
    native_dest.symlink_to(native_lib)

    _prepend_env_path("PYTHONPATH", target_dir)
    print(f"INFO: Added pyxlog module path at {target_dir}")

    if not _can_import_pyxlog():
        raise SystemExit("Unable to import pyxlog after building extension module")


def prob_engine_args(xlog: Path) -> list[str]:
    source = xlog.read_text(encoding="utf-8")
    match = re.search(r"^\s*#pragma\s+prob_engine\s*=\s*([a-zA-Z_]+)\s*$", source, re.MULTILINE)
    if not match:
        return []
    engine = match.group(1).lower()
    if engine == "exact_ddnnf":
        return ["--prob-engine", engine]
    if engine == "mc":
        # Keep CI runtime bounded for stochastic examples.
        mc_samples = os.environ.get("XLOG_VALIDATE_MC_SAMPLES", "100")
        return ["--prob-engine", engine, "--samples", mc_samples]
    return []


def train_help_text(train_script: Path, cwd: Path) -> str:
    probe = subprocess.run(
        [sys.executable, str(train_script), "--help"],
        cwd=cwd,
        env=RUNTIME_ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return f"{probe.stdout}\n{probe.stderr}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "dev", "release"], required=True)
    args = parser.parse_args()

    # Deterministic/probabilistic XLOG examples
    if args.mode == "release":
        RUNTIME_ENV["XLOG_EXAMPLES_RELEASE"] = "1"
    run_cmd(["bash", "scripts/run_xlog_examples.sh"], cwd=ROOT, timeout_sec=1800)

    # Probabilistic .xlog examples
    for xlog in sorted((ROOT / "examples/prob").glob("*.xlog")):
        engine_args = prob_engine_args(xlog)
        run_cmd(
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "xlog-cli",
                "--features",
                "host-io",
                "--",
                "prob",
                str(xlog),
                *engine_args,
            ],
            timeout_sec=300,
        )

    # Python examples (DLPack, etc.)
    ensure_pyxlog_available(args.mode)
    if "XLOG_PY_EXAMPLE_MC_SAMPLES" not in os.environ:
        # Keep the all-examples gate bounded; callers that want a longer
        # statistical run can override XLOG_PY_EXAMPLE_MC_SAMPLES directly.
        RUNTIME_ENV["XLOG_PY_EXAMPLE_MC_SAMPLES"] = os.environ.get(
            "XLOG_VALIDATE_MC_SAMPLES", "100"
        )
    for py in sorted((ROOT / "examples/python").glob("*.py")):
        run_cmd([sys.executable, str(py)], timeout_sec=180)

    # Neural examples (defer to per-example train.py)
    neural_fixture_smoke = os.environ.get("XLOG_VALIDATE_NEURAL_FULL", "").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }
    if neural_fixture_smoke:
        RUNTIME_ENV["XLOG_NEURAL_FIXTURE_SMOKE"] = "1"
    if "XLOG_PY_EXAMPLE_MNIST_LIMIT" not in os.environ:
        RUNTIME_ENV["XLOG_PY_EXAMPLE_MNIST_LIMIT"] = "64" if args.mode == "ci" else "256"
    for example in sorted((ROOT / "examples/neural").iterdir()):
        train = example / "train.py"
        if train.exists():
            help_text = train_help_text(train, example)
            cmd = [sys.executable, str(train)]
            if "--mode" in help_text:
                cmd.extend(["--mode", args.mode])
                if neural_fixture_smoke:
                    if "--epochs" in help_text:
                        cmd.extend(["--epochs", "1"])
                    if "--batch-size" in help_text:
                        cmd.extend(["--batch-size", "16"])
            else:
                if args.mode == "ci":
                    print(f"SKIP: {train} (no --mode support in ci mode)")
                    continue
                # Keep non-mode scripts lightweight in validator runs.
                if "--epochs" in help_text:
                    cmd.extend(["--epochs", "1"])
                if "--batch-size" in help_text:
                    cmd.extend(["--batch-size", "16"])
                if neural_fixture_smoke:
                    if "--train-limit" in help_text:
                        cmd.extend(["--train-limit", "64" if args.mode == "ci" else "256"])
                    if "--eval-limit" in help_text:
                        cmd.extend(["--eval-limit", "64" if args.mode == "ci" else "256"])
                    if "--save-path" in help_text:
                        cmd.extend(["--save-path", "/tmp/xlog_validate_mnist_net.pt"])
            run_neural_cmd(cmd, cwd=example)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
