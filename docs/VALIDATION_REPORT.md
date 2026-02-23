# XLOG Validation Report (Current HEAD)

**Date:** February 23, 2026  
**Scope:** Repository state on `v0.4.0-alpha-integrated` (v0.4.0-alpha milestone achieved)  

This document is the single source of truth for what is *actually verified* in the repository at a given point in
time. It is intentionally conservative: if something is not listed here, it is not considered validated.

---

## Release Status

- **Latest tagged release:** `v0.3.2` (see `git tag -l`).
- **Current `main`:** ahead of `v0.3.2` (v0.4.0-alpha milestone achieved). The codebase contains major additional functionality (GPU-native
  exact path + verifier/caching, device-only MC, neural-symbolic training APIs), and has now met the v0.4.0-alpha milestone requirements.

**v0.4.0-alpha gate (**achieved**):**
- End-to-end validation of *all* examples in `examples/` (CLI + Python where applicable) using `scripts/validate_examples.py`
- Full-dataset (`--mode release`) validation for all neural examples on real data
- Additional neural examples beyond `examples/neural/01_minimal` (**implemented**)

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

### Example harness and neural example smoke checks

```bash
# Example harness (runs deterministic/prob/python + neural train.py scripts)
python scripts/validate_examples.py --mode ci

# Neural example smoke checks (fail-fast on missing real datasets)
pytest python/tests/test_example_02_coins.py \
       python/tests/test_example_03_mnist_multidigit.py \
       python/tests/test_example_04_hwf.py \
       python/tests/test_example_05_poker.py \
       python/tests/test_example_06_clutrr.py -v
```

---

## Example Validation Status

Examples are a release gate for neural milestones, but they are **not yet** validated end-to-end on full datasets.

What *is* covered today:
- The Rust test suite includes a basic CLI smoke test (`test_xlog_run_basic`) that exercises `xlog run` on a small
  program.
- GPU kernels are validated via the certification suite (see above).
- A repository-level example harness exists at `scripts/validate_examples.py` with `--mode {ci,dev,release}`.
- Verified `scripts/validate_examples.py --mode ci` and `--mode release` runs on a fully provisioned real-dataset machine (Track A & Track B metrics logged).
- Full-dataset metric-threshold validation reports for all neural examples (stored in `examples/neural/results/`).
- Required neural examples are implemented:
  - `examples/neural/02_coins/`
  - `examples/neural/03_mnist_multidigit/`
  - `examples/neural/04_hwf/`
  - `examples/neural/05_poker/`
  - `examples/neural/06_clutrr/`
- Neural example smoke tests verify fail-fast behavior when required real datasets are missing.

What is *not* yet covered:
- Multi-GPU distributed training metrics.

---

## Notes

- CUDA-dependent tests in Rust skip cleanly when CUDA is unavailable (developer ergonomics). For production validation,
  the certification suite should be run on real CUDA hardware.
