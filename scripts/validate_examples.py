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


def run_cmd(cmd, cwd=None):
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, env=RUNTIME_ENV)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def run_neural_cmd(cmd, cwd: Path) -> None:
    print("+", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=RUNTIME_ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
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


def ensure_pyxlog_available() -> None:
    if _can_import_pyxlog():
        return

    run_cmd(["cargo", "build", "-q", "-p", "pyxlog", "--features", "host-io"], cwd=ROOT)

    target_dir = ROOT / "target" / "debug"
    linux_lib = target_dir / "libpyxlog.so"
    mac_lib = target_dir / "libpyxlog.dylib"
    py_name = target_dir / "pyxlog.so"

    if linux_lib.exists():
        if py_name.exists() or py_name.is_symlink():
            py_name.unlink()
        py_name.symlink_to(linux_lib)
    elif mac_lib.exists():
        if py_name.exists() or py_name.is_symlink():
            py_name.unlink()
        py_name.symlink_to(mac_lib)

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
        mc_samples = os.environ.get("XLOG_VALIDATE_MC_SAMPLES", "1000")
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
    run_cmd(["bash", "scripts/run_xlog_examples.sh"], cwd=ROOT)

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
            ]
        )

    # Python examples (DLPack, etc.)
    ensure_pyxlog_available()
    if "XLOG_PY_EXAMPLE_MC_SAMPLES" not in os.environ:
        RUNTIME_ENV["XLOG_PY_EXAMPLE_MC_SAMPLES"] = "5000" if args.mode == "ci" else "20000"
    for py in sorted((ROOT / "examples/python").glob("*.py")):
        run_cmd([sys.executable, str(py)])

    # Neural examples (defer to per-example train.py)
    if "XLOG_PY_EXAMPLE_MNIST_LIMIT" not in os.environ:
        RUNTIME_ENV["XLOG_PY_EXAMPLE_MNIST_LIMIT"] = "64" if args.mode == "ci" else "4096"
    for example in sorted((ROOT / "examples/neural").iterdir()):
        train = example / "train.py"
        if train.exists():
            help_text = train_help_text(train, example)
            cmd = [sys.executable, str(train)]
            if "--mode" in help_text:
                cmd.extend(["--mode", args.mode])
            else:
                if args.mode == "ci":
                    print(f"SKIP: {train} (no --mode support in ci mode)")
                    continue
                # Keep non-mode scripts lightweight in validator runs.
                if "--epochs" in help_text:
                    cmd.extend(["--epochs", "1"])
                if "--batch-size" in help_text:
                    cmd.extend(["--batch-size", "16"])
            run_neural_cmd(cmd, cwd=example)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
