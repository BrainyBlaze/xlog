# XLOG Validation Report

**Date:** 2026-01-14  
**Branch:** `main`  
**Status:** `xlog-logic` + `xlog-prob` tiers validated (Linux x86_64 + CUDA-only)  

This document summarizes the current validated state of XLOG on `main`, including the deterministic engine and the Phase 4 probabilistic tier. Historical validation notes were archived to `docs/plans/2026-01-09-validation-report.md`.

---

## Executive Summary

| Item | Status | Evidence |
|------|--------|----------|
| Workspace tests | ✅ PASS (release) | `cargo test --workspace --all-targets --release` |
| CUDA kernel certification | ✅ PASS (140/140) | `docs/plans/2026-01-14-cuda-certification-results.md` |
| Supported logic fragment | ✅ Stratified Datalog + recursion | `crates/xlog-logic`, `crates/xlog-runtime` |
| GPU relational ops | ✅ join/sort/filter/dedup/groupby/set-ops | `crates/xlog-cuda`, `kernels/*.cu`, `kernels/*.ptx` |
| Probabilistic tier | ✅ PASS (exact `exact_ddnnf` + approximate `mc`) | `crates/xlog-prob`, `crates/xlog-prob/tests/mc.rs` |
| Python package | ✅ `xlog_gpu` (PyO3 + DLPack) | `crates/xlog-gpu-py`, `examples/python/` |

---

## Validated Capabilities (Deterministic tier)

### Language / semantics
- Facts + rules, multi-way joins, self-joins, constraints (`:-`) and queries (`?-`) via desugaring.
- Recursion via semi-naive fixpoint iteration.
- Stratified negation (cycles through `not` rejected at compile time).
- Comparisons (`= != < <= > >=`) and arithmetic binding via `is` (including `abs/min/max/pow/cast`).

### GPU execution
- Hash joins: inner / semi / anti / left-outer, with key verification (hash collision safety).
- Stable multi-column GPU sort with on-device permutation.
- Filter/compact at scale (multi-block scans).
- Dedup/distinct on-device.
- GroupBy aggregates: `count/sum/min/max/logsumexp`.
- Set operations: union/diff (multi-type, multi-column schemas).

### Interop
- Arrow RecordBatch/IPC helpers (copying via host).
- DLPack import/export for zero-copy exchange of GPU columns (framework-agnostic).

---

## Validated Capabilities (Phase 4 probabilistic tier)

### Probabilistic profile (`xlog-prob`)
- Probabilistic facts and annotated disjunctions (`exact_ddnnf` backend via D4 Decision-DNNF).
- Query conditioning via evidence and GPU circuit evaluation (XGCF).
- P3 Monte Carlo execution (`prob_engine=mc`) for non-monotone recursion and approximate inference, returning uncertainty metadata (samples/seed + stderr/CI).

### Python API (`xlog-gpu`)
- `xlog_gpu.Program.compile(..., prob_engine="exact_ddnnf"|"mc")` compiles and runs on GPU.
- Results returned as DLPack capsules; Torch is optional (any DLPack producer works).

---

## Primary Validation Artifacts

- CUDA certification suite results: `docs/plans/2026-01-14-cuda-certification-results.md`
- Architecture overview: `docs/ARCHITECTURE.md`
- xlog-prob architecture: `docs/architecture/xlog-prob.md`
- Roadmap + current status: `docs/ROADMAP.md`
- Example programs (end-to-end): `examples/README.md` and `examples/xlog/`

---

## How To Reproduce

### Full workspace tests (release)

```bash
cargo test --workspace --all-targets --release
```

### CUDA certification suite (release)

```bash
cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture
```

### Run a `.xlog` example

```bash
cargo run -p xlog-logic --example xlog_run -- examples/xlog/00-basics/01_tc_reachability.xlog
```

---

## Known Limitations / Notes

- `symbol` values are currently represented as a `u32` hash (not reversible).
- Arrow IPC paths involve device↔host copies; DLPack is the zero-copy path.
- `xlog-elp` and `xlog-solve` remain planned for later phases; `mc` inference does not currently support gradients.
- `mc` inference samples on GPU but evaluates the deterministic core on CPU in Phase 4 (see `docs/architecture/xlog-prob.md`).
