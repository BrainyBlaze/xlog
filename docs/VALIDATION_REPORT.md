# XLOG Validation Report (Historical Snapshot)

**Date:** February 23, 2026  
**Scope:** Validation snapshot captured on `v0.4.0-alpha-integrated` during the
v0.4.0-alpha neural-symbolic milestone work.

This document records a historical validation pass. It is intentionally conservative
about what was verified in that snapshot, but it is **not** the authoritative source
for the current release line.

For current release status, see `docs/ROADMAP.md` and `CHANGELOG.md`.

---

## Release Status

- **Historical snapshot context:** at the time of this report, validation was being tracked
  against the `v0.4.0-alpha` milestone branch/worktree.
- **Current tagged release:** `v0.5.0` (see `git tag -l`).
- **Current `main`:** ahead of `v0.5.0`; at the time of this update, `HEAD` is the
  April 17, 2026 M8 exact-induction change (`3d63d197`).

**Historical v0.4.0-alpha gate status:**
- End-to-end validation of *all* examples in `examples/` (CLI + Python where applicable) using `scripts/validate_examples.py`
- Full-dataset (`--mode release`) validation for all neural examples on real data
- Additional neural examples beyond `examples/neural/01_minimal` (**implemented**)

---

## Verified Commands (Most Recent Run)

All gates run in `--release` mode (matches CI and production validation).

### Rust workspace (excluding PyO3 extension crate)

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

cargo test --workspace --all-targets --exclude pyxlog --release
```

### CUDA/PTX certification suite (full)

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}

cargo test -p xlog-cuda-tests --test certification_suite --release
```

Most recent run: **206/206** passing (C01-C25 + G01-G08).

### PyO3 compile gate

```bash
cargo check -p pyxlog
```

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

This section describes what the historical v0.4.0-alpha validation snapshot covered.
It should not be read as a complete current-release verification matrix.

What *was* covered in this snapshot:
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

What was *not* covered in this snapshot:
- Multi-GPU distributed training metrics.

---

## Notes

- CUDA-dependent tests in Rust skip cleanly when CUDA is unavailable (developer ergonomics). For production validation,
  the certification suite should be run on real CUDA hardware.
