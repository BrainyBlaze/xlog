# G38 G_INT M_INT.10 CUDA Certification Suite

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Sub-goal:** G_INT
**Metric:** M_INT.10 CUDA cert suite
**Branch:** `feat/w3-bundle-integration`
**Status:** PASS.

## Command

```text
cargo test -p xlog-cuda-tests --test certification_suite --release
```

## Result

```text
EXIT 0
running 1 test
test run_full_certification ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 21.65s
```

This is the fresh rerun after adding the M_INT.11 VRAM snapshot test target and
the W39 bench `cudaMemGetInfo` fixture instrumentation.
