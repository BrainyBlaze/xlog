# Changelog

All notable changes to this project are documented in this file.

## Unreleased — 2026-02-01

### Added

- **Arrow C Data Interface device export** for `CudaBuffer` record batches (`to_arrow_device_record_batch`) returning
  `ArrowDeviceArrayOwned` handles with CUDA device descriptors and zero host transfers (export-only; import remains
  DLPack).
- **Arrow device export support for Bool/Symbol**: on-device boolean bit-packing and symbol metadata keys
  (`xlog.symbol=true`, `xlog.symbol_encoding=u32`) for downstream consumers.
- **GPU CDCL verifier (complete SAT/UNSAT)** in `kernels/sat.cu` + `xlog-solve::GpuCdclSolver` with on-GPU SAT model
  checking and on-GPU UNSAT proof checking.
- **GPU PIR→CNF encoder** (`encode_cnf_gpu`) with device-resident CSR emission, deterministic var numbering, and GPU
  reachability (zero host reads in the production path), plus CNF kernels in `kernels/cnf.cu`.
- **GPU neural fast-path (AD chain)** in `kernels/neural.cu` + `xlog-prob` integration:
  - device-side AD conditional-chain weight fill (`neural_fill_ad_chain_f32`)
  - device-side probability-gradient scatter using both `grad_true` and `grad_false` (`neural_scatter_ad_chain_grads_f32`)
- **Zero-host-read verifier API**: expectation-based methods `solve_expect_sat` / `solve_expect_unsat` that never
  download SAT/UNSAT status to the CPU (fail-fast via GPU trap / CUDA error).
- **Device-resident CNF metadata** (`GpuCnf::{num_vars,num_clauses,num_lits}`) to support GPU-native CNF builders where
  capacity > exact size.
- **GPU-native equivalence verification** (`xlog-prob::compilation`) proving `φ ≡ C` via two UNSAT checks on GPU:
  `UNSAT(φ ∧ ¬C)` and `UNSAT(C ∧ ¬φ)`, with zero device→host reads.
- **GPU D4 compile+verify entrypoint** (`compile_gpu_d4_and_verify`) that compiles CNF to device-resident XGCF and
  validates equivalence via the GPU CDCL verifier.
- **Device-resident circuit cache + cache-aware evaluation** (`GpuCircuitCache`, `compile_gpu_d4_and_verify_cached`,
  `kernels/cache.cu`) enabling zero-recompile warm-cache inference.
- **GPU-native exact inference path**: `ExactDdnnfProgram` now uses GPU D4 + GPU CDCL + cache (no CPU D4, no CNF/DDNNF
  host materialization in production).
- **GPU weight/evidence builders** (`kernels/weights.cu` + `gpu_weights.rs`) for device-resident weight tables.
- **Regression guardrails** enforcing “no device→host reads” in the production verifier modules.
- **Cache DTOH guardrails + integration tests** (`no_dtoh_in_gpu_cache`, `gpu_exact_cache_integration`, `gpu_weights`).
- **Device-only logZ outputs** for GPU XGCF evaluation (`eval_log_wmc_device_*`) plus a guard test to prevent
  device→host reads inside device-only evaluation paths.
- **GPU-native loss output for neural fast-path**: `ExactDdnnfProgram::neural_backward_nll_buffers_with_device_loss`
  returns the scalar NLL loss as a device-resident value (no dtoh).
- **DLPack helper for typed allocations**: `TrackedCudaSlice::into_bytes()` enables wrapping typed device scalars into
  `CudaBuffer` columns without copies (used to export scalar loss to Torch).

### Changed

- `GpuCnf` literal storage field renamed to `literals` (DIMACS `i32`) to match the solver/kernel interface.
- CUDA-dependent tests now skip cleanly when the CUDA runtime is unavailable (developer ergonomics).
- Workspace testing avoids building the PyO3 `extension-module` target when running `cargo test --workspace`.
- CUDA transfer/caching certification tests are stable under parallel test execution.

### Fixed

- Monte Carlo GPU initialization avoids reliance on CUDA device-count queries that can fail in restricted environments.
- `pyxlog` DLPack interop: detach `requires_grad` tensors before exporting probabilities to DLPack.
- `pyxlog` GPU neural fast-path ordering: replaced `torch.cuda.synchronize()` with stream-to-stream waits.
- GPU CNF reachability worklist hardened to avoid consuming uninitialized queue entries under concurrent expansion.
- nvcc deprecation warnings for `sm_70` offline PTX builds are suppressed in `kernels/CMakeLists.txt`.
- Release-mode CUDA crash in the GPU CDCL verifier/equivalence path caused by passing temporary scalar kernel arguments
  via raw parameter vectors (now backed by stable locals before `cuLaunchKernel`).
- Release-mode CUDA launch failures in GPU D4 tests and smoothing due to temporary scalar kernel arguments (now backed
  by stable locals before `cuLaunchKernel`).
- GPU smoothing now seeds root support with all random vars and levelizes with the emitted node count, ensuring
  unconditional probabilistic facts/evidence are handled correctly and preventing under-launched levels.
- GPU cache meta loading moved out of `gpu_cache.rs` to preserve dtoh-free guardrails for the cache module.

### Validation

- `cargo test --workspace` passes, including GPU verifier smoke tests (skipped when CUDA is unavailable).

## v0.4.0-alpha — 2026-01-22 — Neural-Symbolic Integration

First alpha release of the neural-symbolic integration layer, enabling differentiable training where neural network outputs become probabilistic facts in logic programs.

### Added

**Neural Predicates (`nn/4` syntax):**
- `nn(network, [inputs], output, [labels]) :: predicate(args).` declaration syntax
- Network-backed probabilistic facts with automatic annotated disjunction generation
- Support for classification mode (with labels) and embedding mode (without)
- Multiple input variables, symbol labels, and empty input lists

**Network Registry:**
- `register_network(name, module, optimizer, scheduler)` Python API
- `NetworkConfig` with neural predicate options: batching, k (top-k), det (deterministic), cache
- `NetworkHandle` with train/eval mode switching
- Automatic validation against declared neural predicates

**Tensor Source Registry:**
- `add_tensor_source(name, tensor)` for external data (images, embeddings)
- `set_active_tensor_source(name)` for switching between train/test
- Index validation and bounds checking
- Metadata tracking (size, shape, dtype)

**Neural → Probability Bridge:**
- Softmax outputs converted to annotated disjunctions
- `NeuralBridge` for numerical stability (epsilon clamping, normalization)
- Log probability computation for gradient stability
- Circuit leaf generation for d-DNNF integration

**Training Infrastructure:**
- `forward_backward()` for single query training with gradient computation
- `train_epoch()` for batch processing with configurable batch size
- `train_model()` for multi-epoch training with shuffle and logging
- `zero_grad()`, `optimizer_step()`, `scheduler_step()` for training loop control
- `TrainingHistory` object with epoch losses and batch metrics

**NLL Loss Functions:**
- `nll_loss(prob)` — negative log-likelihood from probability
- `nll_loss_batch(probs)` — batch NLL computation
- `nll_loss_mean(probs)` — mean NLL over batch
- `nll_loss_tensor(prob)` — PyTorch tensor output for autograd
- Numerical stability via epsilon (1e-10) clamping

**Backward Pass to Networks:**
- `backprop_circuit_gradients()` propagates d-DNNF gradients through neural networks
- Weight slot mapping for position-based gradient routing
- PyTorch `.backward()` integration with gradient tensors
- Support for multiple networks per query

**Circuit Caching:**
- `CachedCircuit` stores compiled d-DNNF circuits for reuse
- `WeightSlot` maps circuit variables to network outputs by position
- `evaluate_gpu_with_grads_weights()` for weight-only circuit evaluation
- Cache key generation from query templates
- Eliminates D4 recompilation bottleneck (100x+ speedup for repeated queries)

**Minimal MNIST Addition Example:**
- `examples/neural/01_minimal/train.py` — complete working example
- CNN network classifying MNIST digits
- Training purely from addition supervision (no digit labels)
- Demonstrates neural-symbolic gradient flow

**Negation in Probabilistic Programs:**
- `not` keyword in rule bodies for exact inference (`wet :- not rain.`)
- Stratified negation with automatic layer detection
- Non-monotone (cyclic) negation via Well-Founded Semantics (WFS)
- Exact gradients flow through negated literals for neural-symbolic training

**GPU Certification Suite (G01-G06):**
- G01: Circuit Forward Kernel tests (8 tests) — `xgcf_forward_level` PTX validation
- G02: Circuit Backward Kernel tests (12 tests) — gradient computation verification
- G03: Weight Injection tests (6 tests) — GPU weight buffer management
- G04: Transfer Efficiency tests (8 tests) — 0% CPU bottleneck verification
- G05: Circuit Cache tests (6 tests) — GpuXgcf reuse, D4 elimination
- G06: PTX Robustness tests (10 tests) — large circuits, edge cases, numerical stability
- Total: 50 new GPU-focused tests validating neural-symbolic kernel correctness

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

### Changed

- **Python package renamed from `xlog-gpu` to `pyxlog`** — cleaner, more memorable name
  - All imports: `import pyxlog` (was `import xlog_gpu`)
  - Crate renamed: `crates/pyxlog` (was `crates/xlog-gpu-py`)
  - PyPI package: `pyxlog` (was `xlog-gpu`)
- Stratification analysis now tracks edge polarity for non-monotone detection
- Provenance extraction routes non-monotone SCCs to WFS evaluation
- CNF encoding emits Tseitin clauses for `NegLit` with negated polarity

### Technical Details

| Component | Files | Purpose |
|-----------|-------|---------|
| Grammar | `grammar.pest:93-102` | `nn/4` syntax parsing |
| AST | `ast.rs:323-358` | `NeuralPredDecl`, `NeuralLabel` |
| Parser | `parser.rs:573-645` | `build_neural_pred_decl()` |
| Registry | `xlog-neural/src/registry.rs` | `NetworkRegistry`, `NetworkConfig` |
| Handle | `xlog-neural/src/handle.rs` | `NetworkHandle` with PyO3 objects |
| Bridge | `xlog-neural/src/bridge.rs` | `NeuralBridge`, `NeuralOutput` |
| Tensor | `xlog-neural/src/tensor_source.rs` | `TensorSourceRegistry` |
| Python | `crates/pyxlog/src/lib.rs` | Full training API |
| PIR | `pir.rs` | `NegLit` variant |
| WFS | `wfs.rs` | Well-Founded Semantics (1,461 lines) |
| Exact | `exact.rs` | `random_var_indices()`, `evaluate_gpu_with_grads_weights()` |
| G01-G06 | `xlog-cuda-tests/src/categories/g0*.rs` | GPU certification tests (50 tests) |

### Validation

- **CUDA Certification Suite:** 200/200 tests passed (C01-C25 + G01-G06)
- **Python Tests:** 109/109 tests passed
- **Spec Alignment:** All 50 G01-G06 tests match specification
- **Code Quality:** No stubs, placeholders, or TODOs

### Example: MNIST Addition Training

```python
import pyxlog
import torch

# Define neural predicate program
program = pyxlog.Program.compile("""
    nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
    addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
""")

# Register PyTorch network
net = MNISTNet()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
program.register_network("mnist_net", net, optimizer)

# Add training data
program.add_tensor_source("train", train_images)

# Train on addition queries (no digit labels!)
queries = ["addition(0, 1, 7)", "addition(2, 3, 5)", ...]
history = pyxlog.train_model(program, queries, epochs=50, batch_size=32)
```

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
- `xlog-gpu` + `pyxlog`: `pyxlog` Python module (PyO3) with DLPack-first I/O for deterministic and probabilistic evaluation.
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
