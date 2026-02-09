import ctypes
import os
from pathlib import Path


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()

    target_dir = Path("target/debug").resolve()
    if target_dir.exists():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{target_dir}:{existing}" if existing else str(target_dir)

    try:
        ctypes.CDLL("libcuda.so")
    except OSError:
        wsl_cuda = Path("/usr/lib/wsl/lib/libcuda.so.1")
        if wsl_cuda.exists():
            shim_dir = Path("/tmp/xlog-cuda-shim")
            shim_dir.mkdir(parents=True, exist_ok=True)
            for soname in ("libcuda.so", "libnvcuda.so"):
                link = shim_dir / soname
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(wsl_cuda)
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{shim_dir}:{existing}" if existing else str(shim_dir)

    return env
