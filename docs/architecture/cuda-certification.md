# CUDA Certification Suite

This document describes the XLOG CUDA/PTX kernel certification test suite, which validates correctness, safety, determinism, portability, and performance pathologies across all GPU kernels.

## Overview

The certification suite is implemented in the `xlog-cuda-tests` crate and provides comprehensive coverage of all CUDA kernel operations.

| Metric | Value |
|--------|-------|
| Total tests | 140 |
| Categories | 24 |
| PTX modules | 18 |
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
│       ├── c01_toolchain.rs      # Category 1: Toolchain/PTX/SASS
│       ├── c02_launch_config.rs  # Category 2: Launch configuration
│       ... (all 24 categories)
│       └── c24_edge_matrix.rs    # Category 24: Edge case matrix
└── tests/
    ├── certification_suite.rs    # Full certification runner
    ├── quick_smoke.rs            # Fast subset for CI
    └── category_isolation.rs     # Run individual categories
```

## Test Categories

### Infrastructure (C01-C08)

| Category | Name | Focus |
|----------|------|-------|
| C01 | Toolchain/PTX/SASS | PTX loading, JIT compilation, symbol resolution |
| C02 | Launch Configuration | Grid/block dimensions, shared memory sizing |
| C03 | Pointer/Indexing/Bounds | Overflow, off-by-one, stride calculations |
| C04 | Address Space | Global/shared/local/const memory correctness |
| C05 | Global Memory Hazards | OOB access, alignment, uninitialized reads |
| C06 | Shared Memory | Bank conflicts, barriers, dynamic smem |
| C07 | Local Memory/Stack | Register spilling, stack overflow |
| C08 | Synchronization/Ordering | Atomics, fences, stream ordering |

### Execution Model (C09-C16)

| Category | Name | Focus |
|----------|------|-------|
| C09 | Warp-Level | Divergence, shuffle, ballot, partial warps |
| C10 | Block/Grid Coordination | Cross-block behavior, atomic contention |
| C11 | Control Flow/Predication | Early return, predicated operations |
| C12 | Atomics | Correctness, contention, overflow, CAS loops |
| C13 | Floating-Point | NaN, Inf, subnormals, FMA, accumulation |
| C14 | Integer Edge Cases | Overflow, shifts, division |
| C15 | Determinism | Reproducibility, sort stability |
| C16 | Async/Pipeline | cp.async, tensor ops (sm_80+) |

### Environment (C17-C21)

| Category | Name | Focus |
|----------|------|-------|
| C17 | Caching/Coherence | L1/L2, volatile, cache lines |
| C18 | Host-Device Integration | Lifetime, async transfers, errors, OOM |
| C19 | Multi-Stream Concurrency | Parallel streams, events |
| C20 | Multi-GPU | Device enumeration, P2P, context switching |
| C21 | Hardware Reliability | Timeout, reset, error reporting |

### Comprehensive (C22-C24)

| Category | Name | Focus |
|----------|------|-------|
| C22 | Algorithm-Specific | Reduction, sort, join, groupby edge cases |
| C23 | Testing Blind Spots | Non-power-of-two, misaligned, stress tests |
| C24 | Edge Case Matrix | Size x Distribution x Type cross-product |

## PTX Module Coverage

Category C01 enumerates every `.entry` in each PTX module and verifies resolution via `CudaKernelProvider`:

| Module | Kernels |
|--------|---------|
| `join.ptx` | `hash_join_bucket_count_v2`, `hash_join_scatter_v2`, `hash_join_probe_v2`, `hash_join_semi`, `hash_join_anti`, `compute_composite_hash` |
| `dedup.ptx` | `mark_unique_*`, `compact_rows` |
| `groupby.ptx` | `detect_group_boundaries`, `extract_group_keys`, `groupby_*`, `groupby_logsumexp_*` |
| `scan.ptx` | `exclusive_scan_mask`, `count_mask`, `multiblock_scan_*` |
| `filter.ptx` | `filter_compare_*`, `compact_*_by_mask`, `mask_{and,or,not}` |
| `pack.ptx` | `pack_keys`, `pack_and_hash_keys`, `hash_packed_keys`, `gather_packed_rows`, `compare_packed_keys`, `pack_bools_to_bitmap` |
| `sort.ptx` | `radix_histogram`, `radix_scatter_*`, `init_indices`, `apply_permutation_*`, `gather_keys_*` |
| `set_ops.ptx` | `concat_{u32,bytes}`, `sorted_diff_mark` |
| `circuit.ptx` | `xgcf_forward_level`, `xgcf_backward_level_*` |
| `cache.ptx` | `cache_cnf_hash`, `cache_lookup_or_insert`, `cache_store_*`, `cache_evict_lru` |
| `cnf.ptx` | `cnf_reachability_*`, `cnf_count_clauses`, `cnf_emit_clauses` |
| `d4.ptx` | `d4_frontier_*`, `d4_compile_*`, `d4_smooth_*` |
| `neural.ptx` | `neural_fill_ad_chain_f32`, `neural_scatter_ad_chain_grads_f32` |
| `mc_sample.ptx` | `mc_sample_bernoulli` |
| `sat.ptx` | `sat_*`, `cdcl_*` |
| `weights.ptx` | `weights_fill_*`, `weights_apply_evidence`, `weights_map_nodes_to_vars` |

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
| Infrastructure | 13 |
| Memory Hierarchy | 33 |
| Execution Model | 22 |
| Numeric Correctness | 24 |
| System Integration | 25 |
| Algorithms & Edge Cases | 23 |
| **Total** | **140** |

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

1. Identify the appropriate category (C01-C24)
2. Add test function in `src/categories/cXX_*.rs`
3. Use generators for edge case coverage
4. Implement CPU reference validator
5. Register in category module
6. Update test count in documentation

## See Also

- [CUDA Kernels](../ARCHITECTURE.md#cuda-kernels) — Kernel documentation
- [GPU Execution](gpu-execution.md) — Runtime execution model
