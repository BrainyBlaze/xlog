# CUDA/PTX Kernel Certification Test Suite Design

## Overview

Comprehensive 24-category certification test suite for xlog CUDA kernels covering correctness, safety, determinism, portability, and performance pathologies.

## Scope

- **Goal:** Full certification suite covering all 24 edge case categories
- **Structure:** Dedicated `xlog-cuda-tests` crate
- **Execution:** GPU-only strict (requires real CUDA hardware)
- **Coverage:** ~3,590 tests across 54 kernel functions in 8 modules

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
1. **Toolchain/PTX/SASS** - PTX loading, JIT, symbol resolution
2. **Launch Configuration** - Grid/block dimensions, shared memory
3. **Pointer/Indexing/Bounds** - Overflow, off-by-one, strides
4. **Address Space** - Global/shared/local/const correctness
5. **Global Memory Hazards** - OOB, alignment, uninitialized
6. **Shared Memory** - Bank conflicts, barriers, dynamic smem
7. **Local Memory/Stack** - Register spilling, stack overflow
8. **Synchronization/Ordering** - Atomics, fences, stream ordering

### Execution Model (C09-C16)
9. **Warp-Level** - Divergence, shuffle, ballot, partial warps
10. **Block/Grid Coordination** - Cross-block behavior, atomic contention
11. **Control Flow/Predication** - Early return, predicated ops
12. **Atomics** - Correctness, contention, overflow, CAS loops
13. **Floating-Point** - NaN, Inf, subnormals, FMA, accumulation
14. **Integer Edge Cases** - Overflow, shifts, division
15. **Determinism** - Reproducibility, sort stability
16. **Async/Pipeline** - cp.async, tensor ops (sm_80+)

### Environment (C17-C21)
17. **Caching/Coherence** - L1/L2, volatile, cache lines
18. **Host-Device Integration** - Lifetime, async, errors, OOM
19. **Multi-Stream Concurrency** - Parallel streams, events
20. **Multi-GPU** - Device enumeration, P2P, context switching
21. **Hardware Reliability** - Timeout, reset, error reporting

### Comprehensive (C22-C24)
22. **Algorithm-Specific** - Reduction, sort, join, groupby edge cases
23. **Testing Blind Spots** - Non-power-of-two, misaligned, stress
24. **Edge Case Matrix** - Size × Distribution × Type cross-product

## Test Harness

### TestContext
- CudaKernelProvider setup with configurable memory budget
- Device synchronization and error checking
- State reset between tests

### Generators
- **SizeGen:** Edge case sizes (0, 1, 31, 32, 33, ..., near i32::MAX)
- **Distribution:** AllEqual, AllUnique, Sorted, Reverse, Adversarial
- **NumericEdges:** Type-specific edge values (NaN, Inf, MIN, MAX)
- **AlignmentGen:** Aligned and misaligned offsets

### Validators
- CPU reference implementations for all operations
- Float comparison with ULP tolerance
- Set comparison (order-independent)
- Permutation and stability validation

## Execution Modes

| Mode | Command | Runtime | Use Case |
|------|---------|---------|----------|
| Full certification | `cargo test --test certification_suite` | 30-60 min | Release |
| Quick smoke | `cargo test --test quick_smoke` | 2-5 min | CI |
| Single category | `cargo test --test category_isolation c03` | 1-5 min | Debug |

## Test Count

| Category Group | Tests |
|----------------|-------|
| C01-C08 (Infrastructure) | ~45 |
| C09-C16 (Execution Model) | ~55 |
| C17-C21 (Environment) | ~35 |
| C22 (Algorithms) | ~40 |
| C23 (Blind Spots) | ~15 |
| C24 (Edge Matrix) | ~3,400 |
| **Total** | **~3,590** |
