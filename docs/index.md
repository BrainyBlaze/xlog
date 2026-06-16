# XLOG documentation

XLOG is a GPU-native logic programming system for deterministic Datalog, probabilistic inference, solver-backed reasoning, and neural-symbolic training on a shared CUDA runtime.

## Start here

- [Language reference](language-reference.md) describes XLOG syntax, semantics, and execution modes.
- [Architecture](ARCHITECTURE.md) explains the crate layout, IR stack, GPU execution model, and subsystem boundaries.
- [Python bindings](architecture/python-bindings.md) documents the `pyxlog` user-facing package.
- [CUDA certification](architecture/cuda-certification.md) records the CUDA kernel validation contract.
- [API reference](api/README.md) links to generated Python, Rust, and CUDA references.

## Generated references

The published site is built from repository Markdown plus generated reference output:

- Python API signatures are generated from checked-in `.pyi` stubs without importing the native GPU extension.
- Rust API pages are generated with `cargo doc`.
- CUDA source reference pages are generated with Doxygen.

Generated HTML is not committed. Run `make docs-validate` for a fast local MkDocs build, or `make docs` when Rust and Doxygen dependencies are available.

