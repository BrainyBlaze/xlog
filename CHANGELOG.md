# Changelog

All notable changes to this project are documented in this file.

## v0.3.2 — 2026-01-18

Module system, user-defined functions, and reversible symbols for expressive, modular Datalog programs.

### Added

**Module System:**
- File-based modules with explicit imports
- `use module.` to import all public predicates
- `use module::{pred1, pred2}.` for selective imports
- `use path/to/module.` for nested modules
- `private` keyword for module-internal predicates and functions

**User-Defined Functions:**
- Reusable functions in rule bodies
- Arithmetic functions: `func double(X) = X * 2.`
- Conditional functions: `func abs(X) = if X < 0 then 0 - X else X.`
- Recursive functions with base-case validation
- Optional type annotations: `func add(X: f64, Y: f64) -> f64 = X + Y.`
- Predicate-based functions: `func get_parent(X) = P :- parent(X, P).`

**Reversible Symbols:**
- Bidirectional string-to-ID mapping
- Symbols display as original strings in query output
- Arrow dictionary encoding for efficient serialization
- `--stats` shows symbol registry metrics

**CLI Enhancements:**
- `--module-path` flag for specifying module search directories

### Changed

- Symbol storage changed from hash-based to sequential ID allocation
- Module resolution now validates imports before compilation

### Breaking Changes

- Serialized Arrow files from v0.3.1 with symbol columns may need re-export
- `hash_symbol_to_u32` function removed from public API

---

## v0.3.1 — 2026-01-18

Float predicates, performance benchmarks, query statistics, fuzz testing, and property-based tests.

### Added

**Float Predicate Support:**
- IEEE 754 total ordering for `f32`/`f64` filter comparisons: `NaN > Inf > positive > +0 > -0 > negative > -Inf`
- Filter kernels: `filter_compare_f32_*` and `filter_compare_f64_*` with proper edge case handling
- Comprehensive tests for NaN, Infinity, subnormals, and signed zeros

**Performance Benchmarks:**
- Criterion.rs benchmarks for `xlog-gpu` (transitive closure, hash join, aggregation)
- Criterion.rs benchmarks for `xlog-prob` (exact inference, Monte Carlo sampling)
- `docs/BENCHMARKS.md` with methodology and baseline metrics
- `.github/workflows/bench.yml` for CI regression detection

**Query Timing & Statistics:**
- `--stats` CLI flag for execution profiling
- Per-stratum timing with iteration counts for recursive strata
- Per-operation timing (join, sort, dedup, filter)
- Memory usage tracking (peak, budget)
- Human-readable and JSON output formats

**Fuzz Testing:**
- `fuzz/` directory with cargo-fuzz targets:
  - `fuzz_parser` — raw byte input fuzzing
  - `fuzz_compiler` — structured program generation
  - `fuzz_type_inference` — type system stress testing
- AddressSanitizer (ASAN) integration for crash detection
- `.github/workflows/fuzz.yml` for continuous fuzzing

**Property-Based Testing:**
- proptest integration in `xlog-cuda-tests`
- Sort stability property (data preservation, ascending order)
- Join correctness property (CPU reference comparison)
- Filter idempotence property (`filter(filter(x)) = filter(x)`)
- Dedup determinism property (consistent output across runs)
- Stress tests for large datasets (50K+ rows)

### Validation
- Workspace tests pass: `cargo test --workspace --all-targets --release`
- Property tests pass: `cargo test -p xlog-cuda-tests --test properties --release`
- Fuzz targets build and run with ASAN

---

## v0.2.0 — 2026-01-14

Phase 4 probabilistic logic programming (`xlog-prob`) merged into `main`; Python bindings are the integration surface for GPU I/O.

### Added
- `xlog-prob`: exact inference via Decision-DNNF (vendored D4) + GPU weighted model counting and gradients.
- `xlog-prob`: P3 Monte Carlo engine (`prob_engine=mc`) with GPU sampling, deterministic non-monotone SCC semantics, and uncertainty metadata.
- New CUDA kernels: `kernels/circuit.ptx` (XGCF forward/backward) and `kernels/mc_sample.ptx` (MC sampling).
- New examples: `examples/prob/` (probabilistic `.xlog`) and `examples/python/` (DLPack bindings).
- `xlog-gpu` + `xlog-gpu-py`: `xlog_gpu` Python module (PyO3) with DLPack-first I/O for deterministic and probabilistic evaluation.
- New/updated docs: `docs/architecture/xlog-prob.md`, `docs/VALIDATION_REPORT.md`.

### Validation
- Workspace tests pass in release (`cargo test --workspace --all-targets --release`).
- CUDA certification suite passes: **140/140** (see `docs/plans/2026-01-14-cuda-certification-results.md`).

## v0.1.0 — 2026-01-13

Initial release of the deterministic `xlog-logic` tier (Phase 3 complete).

### Added
- `.xlog` parser + compiler with stratified negation and semi-naive fixpoint recursion.
- GPU execution backend (`xlog-cuda`) with kernels for join/sort/filter/dedup/groupby/scan/pack/set-ops.
- Arithmetic (`is`) and builtin functions (`abs/min/max/pow/cast`) in rule bodies.
- Aggregations: `count/sum/min/max/logsumexp`.
- Arrow IPC import/export utilities and DLPack zero-copy column interop.
- Example suite under `examples/xlog/` and runner example `crates/xlog-logic/examples/xlog_run.rs`.

### Validation
- Workspace tests pass in release (`cargo test --workspace --all-targets --release`).
- CUDA certification suite passes: **133/133** (see `docs/plans/2026-01-12-cuda-certification-results.md`).

### Known limitations
- `symbol` values are currently represented as a `u32` hash (not reversible).
- Arrow IPC interop involves device↔host copies; DLPack is the zero-copy path.
