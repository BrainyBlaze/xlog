# XLOG Validation Report (Current HEAD)

**Date:** February 3, 2026  
**Scope:** Repository state on `main` (unreleased; ahead of `v0.3.2`)

This document is the single source of truth for what is *actually verified* in the repository at a given point in
time. It is intentionally conservative: if something is not listed here, it is not considered validated.

---

## Release Status

- **Latest tagged release:** `v0.3.2` (see `git tag -l`).
- **Current `main`:** ahead of `v0.3.2` (unreleased). The codebase contains major additional functionality (GPU-native
  exact path + verifier/caching, device-only MC, neural-symbolic training APIs), but it is **not** yet considered a
  tagged release milestone.

**v0.4.0-alpha gate (not yet achieved):**
- End-to-end validation of *all* examples in `examples/` (CLI + Python where applicable)
- Additional neural examples beyond `examples/neural/01_minimal`

---

## Verified Commands (Most Recent Run)

### Rust workspace (excluding PyO3 extension crate)

```bash
# On WSL, the CUDA driver libraries are typically under /usr/lib/wsl/lib
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

cargo test --workspace --all-targets --exclude pyxlog -- --nocapture
```

### Probabilistic tier (host-io feature enabled)

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

cargo test -p xlog-prob --features host-io --all-targets -- --nocapture
```

### CUDA/PTX certification suite (full)

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

cargo test -p xlog-cuda-tests --test certification_suite -- --nocapture
```

Most recent run (February 3, 2026): **206/206** passing (C01-C25 + G01-G08).

---

## Example Validation Status

Examples are a release gate for neural milestones, but they are **not yet** validated end-to-end as a suite.

What *is* covered today:
- The Rust test suite includes a basic CLI smoke test (`test_xlog_run_basic`) that exercises `xlog run` on a small
  program.
- GPU kernels are validated via the certification suite (see above).

What is *not* yet covered:
- A systematic “run all deterministic `.xlog` examples” harness.
- End-to-end validation for `examples/python/*` (requires a Python environment with CUDA + PyTorch).
- End-to-end validation for neural training scripts beyond the minimal example.

---

## Notes

- CUDA-dependent tests in Rust skip cleanly when CUDA is unavailable (developer ergonomics). For production validation,
  the certification suite should be run on real CUDA hardware.
