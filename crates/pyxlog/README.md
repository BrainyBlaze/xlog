# pyxlog

`pyxlog` provides Python bindings for the XLOG GPU-accelerated probabilistic
logic programming runtime.

The package is built from the `BrainyBlaze/xlog` repository and exposes the
native extension module together with the staged CUDA kernel artifacts needed by
the runtime.

At import time, `pyxlog` prefers packaged kernel artifacts under
`pyxlog/kernels/` and exports that path to `XLOG_CUBIN_DIR` automatically when
the wheel includes them. For source-tree validation, ad-hoc probe scripts, or
artifact runners that execute without the packaged kernel directory, set
`XLOG_CUBIN_DIR` explicitly to a directory containing `.cubin` or
`.portable.ptx` files before importing `pyxlog`.

Project documentation, setup instructions, and release notes live in the
repository root:

- https://github.com/BrainyBlaze/xlog

Use the root project README for installation requirements, CUDA expectations,
and end-to-end examples.
