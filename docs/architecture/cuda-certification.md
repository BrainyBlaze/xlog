# CUDA Certification Suite

This document describes the XLOG CUDA/PTX kernel certification test suite, which validates correctness, safety, determinism, portability, and performance pathologies across all GPU kernels.

## Overview

The certification suite is implemented in the `xlog-cuda-tests` crate and provides comprehensive coverage of all CUDA kernel operations.

**As of:** May 2026 (`main` / current HEAD)

| Metric | Value |
|--------|-------|
| Total tests | 207 |
| Categories | 33 (core categories 1-25 plus GPU-tier categories 1-8) |
| PTX modules | 22 |
| Execution mode | GPU-only (requires CUDA hardware) |

## Crate Structure

```
crates/xlog-cuda-tests/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public test harness API
│   ├── harness/
│   │   ├── mod.rs                # Test infrastructure
│   │   ├── provider.rs           # CudaKernelProvider setup + teardown
│   │   ├── generators.rs         # Property-based data generators
│   │   ├── validators.rs         # CPU reference implementations
│   │   └── diagnostics.rs        # Failure analysis and reporting
│   └── categories/
│       ├── mod.rs                # Category registry
│       ├── c01_toolchain.rs      # Core category 1: Toolchain/PTX/SASS
│       ├── c02_launch_config.rs  # Core category 2: Launch configuration
│       ... (core categories 1-25 plus GPU-tier categories 1-8)
│       ├── c25_float_filter.rs   # Category 25: Float filter semantics
│       └── g08_device_counts.rs  # GPU category: device-count / row-count invariants
└── tests/
    ├── certification_suite.rs    # Full certification runner
    ├── quick_smoke.rs            # Fast subset for CI
    └── category_isolation.rs     # Run individual categories
```

## Test Categories

### Infrastructure (core categories 1-8)

| Category | Name | Focus |
|----------|------|-------|
| Core category 1 | Toolchain/PTX/SASS | PTX loading, JIT compilation, symbol resolution |
| Core category 2 | Launch Configuration | Grid/block dimensions, shared memory sizing |
| Core category 3 | Pointer/Indexing/Bounds | Overflow, off-by-one, stride calculations |
| Core category 4 | Address Space | Global/shared/local/const memory correctness |
| Core category 5 | Global Memory Hazards | OOB access, alignment, uninitialized reads |
| Core category 6 | Shared Memory | Bank conflicts, barriers, dynamic smem |
| Core category 7 | Local Memory/Stack | Register spilling, stack overflow |
| Core category 8 | Synchronization/Ordering | Atomics, fences, stream ordering |

### Execution Model (core categories 9-16)

| Category | Name | Focus |
|----------|------|-------|
| Core category 9 | Warp-Level | Divergence, shuffle, ballot, partial warps |
| Core category 10 | Block/Grid Coordination | Cross-block behavior, atomic contention |
| Core category 11 | Control Flow/Predication | Early return, predicated operations |
| Core category 12 | Atomics | Correctness, contention, overflow, CAS loops |
| Core category 13 | Floating-Point | NaN, Inf, subnormals, FMA, accumulation |
| Core category 14 | Integer Edge Cases | Overflow, shifts, division |
| Core category 15 | Determinism | Reproducibility, sort stability |
| Core category 16 | Async/Pipeline | cp.async, tensor ops (sm_80+) |

### Environment (core categories 17-21)

| Category | Name | Focus |
|----------|------|-------|
| Core category 17 | Caching/Coherence | L1/L2, volatile, cache lines |
| Core category 18 | Host-Device Integration | Lifetime, async transfers, errors, OOM |
| Core category 19 | Multi-Stream Concurrency | Parallel streams, events |
| Core category 20 | Multi-GPU | Device enumeration, P2P, context switching |
| Core category 21 | Hardware Reliability | Timeout, reset, error reporting |

### Comprehensive (core categories 22-25)

| Category | Name | Focus |
|----------|------|-------|
| Core category 22 | Algorithm-Specific | Reduction, sort, join, groupby edge cases |
| Core category 23 | Testing Blind Spots | Non-power-of-two, misaligned, stress tests |
| Core category 24 | Edge Case Matrix | Size x Distribution x Type cross-product |
| Core category 25 | Float Filter | Float predicate semantics and total ordering edge cases |

### GPU Tier (GPU-tier categories 1-8)

These categories cover probabilistic/neural/solver kernels that sit above the core relational operator suite.

| Category | Name | Focus |
|----------|------|-------|
| GPU-tier category 1 | Circuit Forward | XGCF forward evaluation correctness |
| GPU-tier category 2 | Circuit Backward | XGCF backward/gradient correctness |
| GPU-tier category 3 | Weight Injection | GPU weight/evidence buffer correctness |
| GPU-tier category 4 | Transfer Efficiency | Guardrails for host-device transfers in critical paths |
| GPU-tier category 5 | Circuit Cache | Cache hit/miss correctness and reuse properties |
| GPU-tier category 6 | PTX Robustness | Large-scale + numerical edge cases for circuit kernels |
| GPU-tier category 7 | SAT/CDCL | GPU CDCL SAT/UNSAT verifier correctness |
| GPU-tier category 8 | Device Counts | Device-resident row-count invariants and related helpers |

## PTX Module Coverage

The toolchain/PTX/SASS category enumerates every `.entry` in each PTX module and verifies resolution via `CudaKernelProvider`:

| Module | Kernels |
|--------|---------|
| `join.ptx` | `hash_join_bucket_count_v2`, `hash_join_scatter_v2`, `hash_join_probe_v2`, `hash_join_semi`, `hash_join_anti`, `compute_composite_hash` |
| `dedup.ptx` | `mark_unique_*`, `compact_rows` |
| `groupby.ptx` | `detect_group_boundaries`, `extract_group_keys`, `groupby_*`, `groupby_logsumexp_*` |
| `scan.ptx` | `exclusive_scan_mask`, `count_mask`, `multiblock_scan_*` |
| `filter.ptx` | `filter_compare_*`, `compact_*_by_mask`, `mask_{and,or,not}` |
| `arith.ptx` | `arith_*` |
| `pack.ptx` | `pack_keys`, `pack_and_hash_keys`, `hash_packed_keys`, `gather_packed_rows`, `compare_packed_keys`, `pack_bools_to_bitmap` |
| `sort.ptx` | `radix_histogram`, `radix_scatter_*`, `init_indices`, `apply_permutation_*`, `gather_keys_*` |
| `set_ops.ptx` | `concat_{u32,bytes}`, `sorted_diff_mark` |
| `circuit.ptx` | `xgcf_forward_level`, `xgcf_backward_level_*` |
| `cache.ptx` | `cache_cnf_hash`, `cache_lookup_or_insert`, `cache_store_*`, `cache_evict_lru` |
| `cnf.ptx` | `cnf_reachability_*`, `cnf_count_clauses`, `cnf_emit_clauses` |
| `pir.ptx` | `pir_*` |
| `d4.ptx` | `d4_frontier_*`, `d4_compile_*`, `d4_smooth_*` |
| `neural.ptx` | `neural_fill_ad_chain_f32`, `neural_scatter_ad_chain_grads_f32` |
| `mc_sample.ptx` | `mc_sample_bernoulli` |
| `mc_eval.ptx` | `mc_eval_*` |
| `sat.ptx` | `sat_*`, `cdcl_*` |
| `ilp.ptx` | `ilp_*` kernels for sparse mask and dILP operations |
| `ilp_credit.ptx` | `ilp_credit_*` kernels for GPU-resident credit/loss |
| `weights.ptx` | `weights_fill_*`, `weights_apply_evidence`, `weights_map_nodes_to_vars` |

### Build-time compiled ILP-family modules

Three kernel modules are compiled from `.cu` at build time by
`crates/xlog-cuda/build.rs` and do **not** have checked-in `.ptx`
artifacts. They are loaded at runtime via the `KERNEL_MODULES` manifest
(currently 22 entries) but are *not* auto-discovered by
the toolchain category's kernel-function-resolution test, which enumerates
committed `.ptx` files only.

| Module | `.cu` source | Purpose |
|--------|--------------|---------|
| `xlog_ilp` | `kernels/ilp.cu` | Selected-ID mask helpers, sparse mask COO fill, CSR histogram, f32/f64 block reductions |
| `xlog_ilp_credit` | `kernels/ilp_credit.cu` | Credit forward/backward for dILP loss gradient |
| `xlog_ilp_exact` | `kernels/ilp_exact.cu` | External-consumer bounded exact-induction scorer (`ilp_exact_score`). See [bounded-exact-induction.md](bounded-exact-induction.md). |

Each module is covered by crate-local CUDA-gated tests instead of the
central toolchain enumeration (see each crate's `provider/ilp*.rs` test
submodules). If a future need arises for central toolchain-style coverage of
these modules, committing their `.portable.ptx` build output into
`kernels/` is the minimal incremental change.

## Test Harness

### TestContext

```rust
pub struct TestContext {
    pub provider: Arc<CudaKernelProvider>,
    pub memory: Arc<GpuMemoryManager>,
}

impl TestContext {
    pub fn new() -> Result<Self>;
    pub fn with_budget(bytes: u64) -> Result<Self>;
    pub fn reset(&mut self);  // Clear state between tests
}
```

### Data Generators

| Generator | Purpose |
|-----------|---------|
| `SizeGen` | Edge case sizes (0, 1, 31, 32, 33, ..., near `i32::MAX`) |
| `Distribution` | AllEqual, AllUnique, Sorted, Reverse, Adversarial |
| `NumericEdges` | Type-specific edge values (NaN, Inf, MIN, MAX) |
| `AlignmentGen` | Aligned and misaligned offsets |

### Validators

CPU reference implementations for all operations:

- Float comparison with ULP tolerance
- Set comparison (order-independent)
- Permutation and stability validation
- Aggregation correctness

## Execution Modes

| Mode | Command | Use Case |
|------|---------|----------|
| Full certification | `cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture` | Release gating |
| Quick smoke | `cargo test -p xlog-cuda-tests --test quick_smoke --release -- --nocapture` | CI pipeline |
| Single category | `cargo test -p xlog-cuda-tests --test category_isolation c03 --release -- --nocapture` | Debugging |

## Test Distribution

| Category Group | Tests |
|----------------|-------|
| Core CUDA kernels (core categories 1-25) | 151 |
| Probabilistic/Neural/Solver kernels (GPU-tier categories 1-8) | 56 |
| **Total** | **207** |

## Key Correctness Tests

### Hash Join Collision Safety

Tests verify that hash-only comparison (without key verification) can produce false positives, and that key verification mode eliminates them.

### Aggregation Overflow/Truncation

Tests verify that `sum` aggregation uses `u64` output to prevent truncation when summing `u32` values.

### Multi-Block Prefix Sum

Tests verify that inputs larger than 256 elements work correctly with the 3-phase multi-block scan algorithm.

### Sort Stability

Tests verify that radix sort maintains stable ordering (equal keys preserve original order).

## Adding New Tests

1. Identify the appropriate category (core categories or GPU-tier categories)
2. Add test function in the matching `src/categories/` file
3. Use generators for edge case coverage
4. Implement CPU reference validator
5. Register in category module
6. Update test count in documentation

## See Also

- [CUDA Kernels](../ARCHITECTURE.md#cuda-kernels) — Kernel documentation
- [GPU Execution](gpu-execution.md) — Runtime execution model
