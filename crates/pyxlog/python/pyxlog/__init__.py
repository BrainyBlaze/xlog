from pyxlog._kernel_paths import configure_kernel_search_path

configure_kernel_search_path()

# Re-export everything from the native Rust module
import pyxlog._native as _native
from pyxlog._native import *  # noqa: F401,F403

__doc__ = _native.__doc__
if hasattr(_native, "__all__"):
    __all__ = _native.__all__
