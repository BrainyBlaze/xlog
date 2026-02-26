"""Shared test fixtures for ILP tests."""
import pytest


def _cuda_available_for_pyxlog() -> bool:
    """Check if CUDA is available for both torch AND pyxlog (cudarc).

    torch.cuda.is_available() can return True even when cudarc cannot find
    libcuda.so (e.g. missing from LD_LIBRARY_PATH). This probe actually tries
    to compile a trivial program to verify end-to-end GPU access.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except ImportError:
        return False
    try:
        import pyxlog
        prog = pyxlog.IlpProgramFactory.compile(
            "edge(1,2). learnable(W) :: r(X,Y) :- b1(X,Z), b2(Z,Y).",
            device=0, memory_mb=64,
        )
        _ = prog.ilp_schema_size()
        return True
    except (ImportError, RuntimeError):
        return False


# Cache the result at import time so we only probe once per session
_PYXLOG_CUDA_OK = _cuda_available_for_pyxlog()


def skip_unless_pyxlog_cuda():
    """Call at module level to skip the entire file if CUDA is not usable."""
    if not _PYXLOG_CUDA_OK:
        pytest.skip("CUDA not available for pyxlog (cudarc)", allow_module_level=True)
