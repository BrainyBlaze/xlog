from __future__ import annotations

import os
from pathlib import Path


KERNEL_ARTIFACT_SUFFIXES = (".cubin", ".portable.ptx")


def package_kernel_dir(package_root: Path | None = None) -> Path:
    root = package_root if package_root is not None else Path(__file__).absolute().parent
    return root / "kernels"


def find_packaged_kernel_dir(package_root: Path | None = None) -> Path | None:
    kernels_dir = package_kernel_dir(package_root)
    if not kernels_dir.is_dir():
        return None

    for child in kernels_dir.iterdir():
        if child.is_file() and child.name.endswith(KERNEL_ARTIFACT_SUFFIXES):
            return kernels_dir
    return None


def configure_kernel_search_path(package_root: Path | None = None) -> str | None:
    explicit = os.environ.get("XLOG_CUBIN_DIR")
    if explicit:
        return explicit

    packaged = find_packaged_kernel_dir(package_root)
    if packaged is None:
        return None

    resolved = str(packaged)
    os.environ["XLOG_CUBIN_DIR"] = resolved
    return resolved
