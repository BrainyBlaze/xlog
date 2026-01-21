# Changelog

All notable changes to this project are documented in this file.

## v0.4.0-alpha — 2026-01-21 — Neural-Symbolic Integration

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
| Python | `pyxlog/src/lib.rs` | Full training API |
| PIR | `pir.rs` | `NegLit` variant |
| WFS | `wfs.rs` | Well-Founded Semantics (1,461 lines) |
| Exact | `exact.rs` | `random_var_indices()`, `evaluate_gpu_with_grads_weights()` |

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
