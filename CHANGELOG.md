# Changelog

All notable changes to this project are documented in this file.

## Unreleased — Negation Support in Exact Inference

Full negation support for the exact d-DNNF inference engine with gradient computation.

### Added

**Negation in Probabilistic Programs:**
- `not` keyword in rule bodies for exact inference (`wet :- not rain.`)
- Stratified negation with automatic layer detection
- Non-monotone (cyclic) negation via Well-Founded Semantics (WFS)
- Exact gradients flow through negated literals for neural-symbolic training

**PIR Extension:**
- `NegLit { leaf: LeafId }` node for negated probabilistic leaves
- NNF (Negation Normal Form) transformation pushes negation to leaves
- Weight semantics: `NegLit` uses complemented probability `(1-p, p)`

**Stratification Analysis:**
- `analyze_stratification()` function detects non-monotone SCCs
- Edge polarity tracking in dependency graph (positive/negative edges)
- Automatic classification: stratified SCCs use two-valued evaluation, non-monotone use WFS

**Well-Founded Semantics (WFS):**
- Three-valued logic: True, False, Undefined
- Alternating fixed-point algorithm (unfounded set + consequence derivation)
- Undefined atoms return probability 0 with zero gradient
- Full 1,461-line implementation in `wfs.rs`

**Gradient Computation:**
- Sign flip for negated leaves: `∂(1-p)/∂p = -1`
- Gradients propagate correctly through WFS-evaluated programs
- Finite difference verification in test suite

**Testing:**
- 17 new Python tests across 5 test classes
- Stratified negation tests (`test_simple_not`, `test_multi_layer_stratified`)
- Non-monotone WFS tests (`test_classic_wfs_cycle`, `test_wfs_partial_definition`)
- Gradient correctness tests (`test_negation_gradient_sign`, `test_finite_difference_negation`)
- MC comparison tests for probability validation

### Changed

- Stratification analysis now tracks edge polarity for non-monotone detection
- Provenance extraction routes non-monotone SCCs to WFS evaluation
- CNF encoding emits Tseitin clauses for `NegLit` with negated polarity

### Technical Details

| Component | Change |
|-----------|--------|
| `pir.rs` | Added `NegLit` variant, `neg_lit()` builder |
| `provenance.rs` | Removed negation blocker, added `negate_provenance()`, WFS integration |
| `cnf.rs` | Tseitin encoding for `NegLit` with `v <-> !leaf_var` |
| `stratify.rs` | `analyze_stratification()` with polarity tracking |
| `wfs.rs` | Full WFS implementation (1,461 lines) |
| `exact.rs` | Gradient sign flip for negated leaves |

---

## v0.3.2 — 2026-01-19

Module system, user-defined functions, reversible symbols, and comprehensive showcase examples for expressive, modular Datalog programs.

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

**Showcase Examples:**
- Enterprise Analytics: HR management, compensation, org hierarchy with recursive management chains
- Knowledge Graph: Ontology modeling, citation analysis, semantic inference with type inheritance
- Game Analytics: Player statistics, achievements, guilds, leaderboards with social network analysis
- Supply Chain: Bill of Materials explosion, inventory management, supplier analytics

### Fixed

- **GroupBy count aggregation type**: Count now outputs `u64` (was `u32`) to match predicate declarations and prevent type mismatch errors when comparing count results
- **Optimizer predicate pushdown**: Fixed column width estimation to use schema information for accurate filtering

### Changed

- Symbol storage changed from hash-based to sequential ID allocation
- Module resolution now validates imports before compilation

### Breaking Changes

- Serialized Arrow files from v0.3.1 with symbol columns may need re-export
- `hash_symbol_to_u32` function removed from public API
- Count aggregation results are now `u64` instead of `u32`

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
