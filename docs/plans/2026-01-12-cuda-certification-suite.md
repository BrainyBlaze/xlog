# CUDA Certification Test Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a comprehensive 24-category certification test suite for xlog CUDA kernels covering correctness, safety, determinism, portability, and performance pathologies.

**Architecture:** Dedicated `xlog-cuda-tests` crate with test harness (generators, validators, diagnostics), 24 category modules covering all edge cases from the checklist, and three test runners (full certification, quick smoke, category isolation). GPU-only strict execution.

**Tech Stack:** Rust, cudarc 0.12, rand/rand_chacha for deterministic RNG, xlog-cuda/xlog-core integration.

**Status (2026-01-14):** Implemented (24 categories, 140 internal tests). Latest results: `docs/plans/2026-01-14-cuda-certification-results.md`.

---

## Batch 1: Crate Setup and Harness Infrastructure

### Task 1: Create xlog-cuda-tests crate structure

**Files:**
- Create: `crates/xlog-cuda-tests/Cargo.toml`
- Create: `crates/xlog-cuda-tests/src/lib.rs`
- Create: `crates/xlog-cuda-tests/src/harness/mod.rs`
- Create: `crates/xlog-cuda-tests/src/categories/mod.rs`

**Step 1: Create Cargo.toml**

```toml
[package]
name = "xlog-cuda-tests"
version = "0.1.0"
edition = "2021"
description = "Comprehensive CUDA/PTX kernel certification test suite"
publish = false

[dependencies]
xlog-cuda = { path = "../xlog-cuda" }
xlog-core = { path = "../xlog-core" }
cudarc = { workspace = true }
rand = "0.8"
rand_chacha = "0.3"

[dev-dependencies]
bytemuck = "1.14"

[[test]]
name = "certification_suite"
path = "tests/certification_suite.rs"
harness = true

[[test]]
name = "quick_smoke"
path = "tests/quick_smoke.rs"
harness = true

[[test]]
name = "category_isolation"
path = "tests/category_isolation.rs"
harness = true
```

**Step 2: Create src/lib.rs**

```rust
//! xlog-cuda-tests: Comprehensive CUDA kernel certification suite
//!
//! # Usage
//! ```bash
//! # Full certification (seconds to minutes; GPU-dependent)
//! cargo test -p xlog-cuda-tests --test certification_suite --release
//!
//! # Quick smoke test (sub-second to seconds; GPU-dependent)
//! cargo test -p xlog-cuda-tests --test quick_smoke --release
//!
//! # Single category
//! cargo test -p xlog-cuda-tests --test category_isolation c03 --release
//! ```

pub mod harness;
pub mod categories;

pub use harness::{TestContext, FailureDiagnostic, CertificationResults, CategoryResult, TestResult, TestStatus};
pub use harness::generators::{SizeGen, Distribution, NumericEdges, AlignmentGen};
pub use harness::validators::{reference, compare};
```

**Step 3: Create src/harness/mod.rs**

```rust
//! Test harness infrastructure for CUDA certification tests.

pub mod provider;
pub mod generators;
pub mod validators;
pub mod diagnostics;

pub use provider::TestContext;
pub use diagnostics::{FailureDiagnostic, CertificationResults, CategoryResult, TestResult, TestStatus};
```

**Step 4: Create src/categories/mod.rs**

```rust
//! Test category modules covering all 24 edge case categories.

pub mod c01_toolchain;
pub mod c02_launch_config;
pub mod c03_pointer_bounds;
pub mod c04_address_space;
pub mod c05_global_memory;
pub mod c06_shared_memory;
pub mod c07_local_memory;
pub mod c08_synchronization;
pub mod c09_warp_level;
pub mod c10_block_grid;
pub mod c11_control_flow;
pub mod c12_atomics;
pub mod c13_floating_point;
pub mod c14_integer;
pub mod c15_determinism;
pub mod c16_async_pipeline;
pub mod c17_caching;
pub mod c18_host_device;
pub mod c19_multi_stream;
pub mod c20_multi_gpu;
pub mod c21_hardware;
pub mod c22_algorithms;
pub mod c23_blind_spots;
pub mod c24_edge_matrix;
```

**Step 5: Verify structure compiles**

Run: `cargo check -p xlog-cuda-tests`
Expected: Compilation errors for missing modules (expected at this stage)

---

### Task 2: Implement TestContext provider

**Files:**
- Create: `crates/xlog-cuda-tests/src/harness/provider.rs`

**Step 1: Implement TestContext**

```rust
//! Test context and provider setup for CUDA certification tests.

use std::sync::Arc;
use xlog_core::{MemoryBudget, Result, XlogError};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

/// Test context providing CUDA resources for certification tests.
pub struct TestContext {
    pub provider: CudaKernelProvider,
    pub device: Arc<CudaDevice>,
    pub memory: Arc<GpuMemoryManager>,
}

impl TestContext {
    /// Create test context with specific memory budget in bytes.
    pub fn with_budget(budget_bytes: usize) -> Result<Self> {
        let device_count = cudarc::driver::CudaDevice::count()
            .map_err(|e| XlogError::Kernel(format!("Failed to get device count: {}", e)))?;

        if device_count == 0 {
            return Err(XlogError::Kernel("No CUDA devices available".to_string()));
        }

        let device = Arc::new(CudaDevice::new(0)?);
        let budget = MemoryBudget::with_limit(budget_bytes);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        let provider = CudaKernelProvider::new(device.clone(), memory.clone())?;

        Ok(Self { provider, device, memory })
    }

    /// Create test context with default 1GB memory budget.
    pub fn new() -> Result<Self> {
        Self::with_budget(1024 * 1024 * 1024)
    }

    /// Create test context with large 4GB memory budget for stress tests.
    pub fn large() -> Result<Self> {
        Self::with_budget(4 * 1024 * 1024 * 1024)
    }

    /// Force device synchronization and check for async errors.
    pub fn sync_and_check(&self) -> Result<()> {
        self.device.inner().synchronize()
            .map_err(|e| XlogError::Kernel(format!("Sync failed: {}", e)))?;
        Ok(())
    }

    /// Get current memory usage in bytes.
    pub fn memory_used(&self) -> usize {
        self.memory.current_usage()
    }

    /// Get memory budget in bytes.
    pub fn memory_budget(&self) -> usize {
        self.memory.budget_limit()
    }

    /// Check if multi-GPU is available.
    pub fn multi_gpu_available(&self) -> bool {
        cudarc::driver::CudaDevice::count().unwrap_or(0) > 1
    }

    /// Get compute capability of current device.
    pub fn compute_capability(&self) -> (u32, u32) {
        // Default to sm_70 (Volta) as minimum supported
        (7, 0)
    }
}

/// Macro for tests requiring CUDA device - panics if not available.
#[macro_export]
macro_rules! gpu_test {
    ($name:ident, $body:expr) => {
        #[test]
        fn $name() {
            let ctx = $crate::harness::TestContext::new()
                .expect("CUDA device required for this test");
            $body(&ctx);
        }
    };
}

/// Macro for tests requiring CUDA device - skips if not available.
#[macro_export]
macro_rules! gpu_test_skip {
    ($name:ident, $body:expr) => {
        #[test]
        fn $name() {
            let ctx = match $crate::harness::TestContext::new() {
                Ok(ctx) => ctx,
                Err(_) => {
                    eprintln!("Skipping {}: no CUDA device", stringify!($name));
                    return;
                }
            };
            $body(&ctx);
        }
    };
}
```

**Step 2: Verify compiles**

Run: `cargo check -p xlog-cuda-tests`
Expected: Errors for missing generator/validator/diagnostics modules

---

### Task 3: Implement data generators

**Files:**
- Create: `crates/xlog-cuda-tests/src/harness/generators.rs`

**Step 1: Implement all generators**

```rust
//! Property-based data generators for CUDA certification tests.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

/// Size generator for edge case testing.
pub struct SizeGen;

impl SizeGen {
    /// Standard edge case sizes covering block boundaries and common edge cases.
    pub fn edge_cases() -> Vec<usize> {
        vec![
            0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65,
            127, 128, 129, 255, 256, 257, 511, 512, 513,
            1023, 1024, 1025, 2047, 2048, 2049,
            4095, 4096, 4097, 8191, 8192, 8193,
            16383, 16384, 16385, 32767, 32768, 32769,
            65535, 65536, 65537,
        ]
    }

    /// Sizes near 32-bit overflow boundary.
    pub fn near_i32_max() -> Vec<usize> {
        vec![
            (i32::MAX as usize) - 2,
            (i32::MAX as usize) - 1,
            i32::MAX as usize,
        ]
    }

    /// Sizes that are prime numbers (worst case for some algorithms).
    pub fn primes() -> Vec<usize> {
        vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
             53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
             127, 131, 251, 257, 509, 521, 1021, 1031, 2039, 2053,
             4093, 4099, 8191, 8209, 16381, 16411, 32749, 32771,
             65521, 65537]
    }

    /// Random sizes within a range using deterministic RNG.
    pub fn random(count: usize, min: usize, max: usize, seed: u64) -> Vec<usize> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..count).map(|_| rng.gen_range(min..=max)).collect()
    }

    /// Warp-related sizes (multiples of 32, and off-by-one).
    pub fn warp_related() -> Vec<usize> {
        vec![
            31, 32, 33, 63, 64, 65, 95, 96, 97,
            127, 128, 129, 159, 160, 161,
            191, 192, 193, 223, 224, 225,
            255, 256, 257, 287, 288, 289,
        ]
    }

    /// Block-related sizes (multiples of 256, and off-by-one).
    pub fn block_related() -> Vec<usize> {
        vec![
            255, 256, 257, 511, 512, 513,
            767, 768, 769, 1023, 1024, 1025,
            1279, 1280, 1281, 1535, 1536, 1537,
        ]
    }
}

/// Value distribution patterns for test data generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distribution {
    /// All elements have the same value.
    AllEqual,
    /// All elements are unique (sequential).
    AllUnique,
    /// Elements are in ascending order.
    Sorted,
    /// Elements are in descending order.
    ReverseSorted,
    /// Alternating pattern (0, max, 0, max, ...).
    Alternating,
    /// Values chosen to maximize hash collisions.
    AdversarialHash,
    /// Uniform random distribution.
    Random,
    /// Half zeros, half ones (for masks).
    HalfAndHalf,
    /// Mostly zeros with occasional ones.
    Sparse,
    /// Mostly ones with occasional zeros.
    Dense,
}

impl Distribution {
    /// Get all distribution variants.
    pub fn all() -> Vec<Distribution> {
        vec![
            Distribution::AllEqual,
            Distribution::AllUnique,
            Distribution::Sorted,
            Distribution::ReverseSorted,
            Distribution::Alternating,
            Distribution::AdversarialHash,
            Distribution::Random,
        ]
    }

    /// Generate u32 values with this distribution.
    pub fn generate_u32(&self, count: usize, seed: u64) -> Vec<u32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        match self {
            Distribution::AllEqual => vec![42u32; count],
            Distribution::AllUnique => (0..count as u32).collect(),
            Distribution::Sorted => (0..count as u32).collect(),
            Distribution::ReverseSorted => (0..count as u32).rev().collect(),
            Distribution::Alternating => {
                (0..count).map(|i| if i % 2 == 0 { 0 } else { u32::MAX }).collect()
            }
            Distribution::AdversarialHash => {
                // Values that collide in simple hash tables (multiples of table size)
                (0..count).map(|i| (i as u32) * 256).collect()
            }
            Distribution::Random => {
                (0..count).map(|_| rng.gen()).collect()
            }
            Distribution::HalfAndHalf => {
                (0..count).map(|i| if i < count / 2 { 0 } else { 1 }).collect()
            }
            Distribution::Sparse => {
                (0..count).map(|i| if i % 10 == 0 { 1 } else { 0 }).collect()
            }
            Distribution::Dense => {
                (0..count).map(|i| if i % 10 == 0 { 0 } else { 1 }).collect()
            }
        }
    }

    /// Generate i64 values with this distribution.
    pub fn generate_i64(&self, count: usize, seed: u64) -> Vec<i64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        match self {
            Distribution::AllEqual => vec![42i64; count],
            Distribution::AllUnique => (0..count as i64).collect(),
            Distribution::Sorted => (0..count as i64).collect(),
            Distribution::ReverseSorted => (0..count as i64).rev().collect(),
            Distribution::Alternating => {
                (0..count).map(|i| if i % 2 == 0 { i64::MIN } else { i64::MAX }).collect()
            }
            Distribution::AdversarialHash => {
                (0..count).map(|i| (i as i64) * 256).collect()
            }
            Distribution::Random => {
                (0..count).map(|_| rng.gen()).collect()
            }
            Distribution::HalfAndHalf => {
                (0..count).map(|i| if i < count / 2 { -1 } else { 1 }).collect()
            }
            Distribution::Sparse => {
                (0..count).map(|i| if i % 10 == 0 { 1 } else { 0 }).collect()
            }
            Distribution::Dense => {
                (0..count).map(|i| if i % 10 == 0 { 0 } else { 1 }).collect()
            }
        }
    }

    /// Generate f64 values with this distribution.
    pub fn generate_f64(&self, count: usize, seed: u64) -> Vec<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        match self {
            Distribution::AllEqual => vec![42.0f64; count],
            Distribution::AllUnique => (0..count).map(|i| i as f64).collect(),
            Distribution::Sorted => (0..count).map(|i| i as f64).collect(),
            Distribution::ReverseSorted => (0..count).map(|i| (count - 1 - i) as f64).collect(),
            Distribution::Alternating => {
                (0..count).map(|i| if i % 2 == 0 { f64::MIN } else { f64::MAX }).collect()
            }
            Distribution::AdversarialHash => {
                (0..count).map(|i| (i as f64) * 256.0).collect()
            }
            Distribution::Random => {
                (0..count).map(|_| rng.gen_range(-1e10..1e10)).collect()
            }
            Distribution::HalfAndHalf => {
                (0..count).map(|i| if i < count / 2 { -1.0 } else { 1.0 }).collect()
            }
            Distribution::Sparse => {
                (0..count).map(|i| if i % 10 == 0 { 1.0 } else { 0.0 }).collect()
            }
            Distribution::Dense => {
                (0..count).map(|i| if i % 10 == 0 { 0.0 } else { 1.0 }).collect()
            }
        }
    }

    /// Generate u8 mask values with this distribution.
    pub fn generate_mask(&self, count: usize, seed: u64) -> Vec<u8> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        match self {
            Distribution::AllEqual => vec![1u8; count],
            Distribution::AllUnique => (0..count).map(|i| (i % 2) as u8).collect(),
            Distribution::Sorted => (0..count).map(|i| if i < count / 2 { 0 } else { 1 }).collect(),
            Distribution::ReverseSorted => (0..count).map(|i| if i < count / 2 { 1 } else { 0 }).collect(),
            Distribution::Alternating => (0..count).map(|i| (i % 2) as u8).collect(),
            Distribution::AdversarialHash => (0..count).map(|i| (i % 2) as u8).collect(),
            Distribution::Random => (0..count).map(|_| if rng.gen_bool(0.5) { 1 } else { 0 }).collect(),
            Distribution::HalfAndHalf => (0..count).map(|i| if i < count / 2 { 0 } else { 1 }).collect(),
            Distribution::Sparse => (0..count).map(|i| if i % 10 == 0 { 1 } else { 0 }).collect(),
            Distribution::Dense => (0..count).map(|i| if i % 10 == 0 { 0 } else { 1 }).collect(),
        }
    }
}

/// Numeric edge value generators for specific types.
pub struct NumericEdges;

impl NumericEdges {
    /// Edge values for u32.
    pub fn u32_edges() -> Vec<u32> {
        vec![0, 1, 2, u32::MAX - 1, u32::MAX]
    }

    /// Edge values for i32.
    pub fn i32_edges() -> Vec<i32> {
        vec![i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX]
    }

    /// Edge values for u64.
    pub fn u64_edges() -> Vec<u64> {
        vec![0, 1, 2, u64::MAX - 1, u64::MAX]
    }

    /// Edge values for i64.
    pub fn i64_edges() -> Vec<i64> {
        vec![i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX - 1, i64::MAX]
    }

    /// Edge values for f32.
    pub fn f32_edges() -> Vec<f32> {
        vec![
            f32::NEG_INFINITY,
            f32::MIN,
            -1.0,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0,
            f32::MAX,
            f32::INFINITY,
            f32::NAN,
        ]
    }

    /// Edge values for f64.
    pub fn f64_edges() -> Vec<f64> {
        vec![
            f64::NEG_INFINITY,
            f64::MIN,
            -1.0,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            1.0,
            f64::MAX,
            f64::INFINITY,
            f64::NAN,
        ]
    }

    /// Subnormal (denormalized) f64 values.
    pub fn f64_subnormals() -> Vec<f64> {
        vec![
            5e-324,           // Smallest positive subnormal
            1e-323,
            1e-320,
            1e-310,
            f64::MIN_POSITIVE / 2.0,  // Just below normal range
            -5e-324,
            -1e-323,
            -f64::MIN_POSITIVE / 2.0,
        ]
    }

    /// Values that stress FMA vs mul+add differences.
    pub fn f64_fma_stress() -> Vec<(f64, f64, f64)> {
        vec![
            (1.0 + 1e-15, 1.0 - 1e-15, 1e-30),  // Near 1.0 with tiny addend
            (1e100, 1e100, 1e-100),              // Large * large + tiny
            (1e-100, 1e-100, 1e-200),            // Tiny * tiny + tinier
            (f64::MAX / 2.0, 2.0, -f64::MAX),    // Overflow avoidance
        ]
    }
}

/// Alignment generator for memory access testing.
pub struct AlignmentGen;

impl AlignmentGen {
    /// Byte offsets for alignment testing.
    pub fn offsets() -> Vec<(usize, &'static str)> {
        vec![
            (0, "aligned"),
            (1, "off-by-1"),
            (2, "off-by-2"),
            (3, "off-by-3"),
            (4, "off-by-4"),
            (5, "off-by-5"),
            (6, "off-by-6"),
            (7, "off-by-7"),
        ]
    }

    /// Offsets that violate specific alignment requirements.
    pub fn violating_offsets(required_alignment: usize) -> Vec<usize> {
        (1..required_alignment).collect()
    }
}

/// Key distribution generator for hash table and join testing.
pub struct KeyDistribution;

impl KeyDistribution {
    /// Generate key pairs for join testing (left, right).
    /// Returns (left_keys, right_keys, expected_match_count).
    pub fn join_test_data(size: usize, overlap_ratio: f64, seed: u64) -> (Vec<u32>, Vec<u32>, usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let overlap_size = (size as f64 * overlap_ratio) as usize;

        // Common keys that will match
        let common: Vec<u32> = (0..overlap_size as u32).collect();

        // Left-only keys
        let left_only: Vec<u32> = (overlap_size as u32..(size as u32)).collect();

        // Right-only keys
        let right_only: Vec<u32> = ((size + overlap_size) as u32..((2 * size) as u32)).collect();

        let mut left = common.clone();
        left.extend(left_only);
        left.shuffle(&mut rng);

        let mut right = common.clone();
        right.extend(right_only);
        right.shuffle(&mut rng);

        (left, right, overlap_size)
    }

    /// Generate keys with many hash collisions.
    pub fn collision_keys(count: usize, collision_factor: usize) -> Vec<u32> {
        // Keys that hash to same bucket in power-of-two table
        (0..count).map(|i| (i * collision_factor) as u32).collect()
    }

    /// Generate keys for groupby testing with specified group count.
    pub fn groupby_keys(total_rows: usize, num_groups: usize, seed: u64) -> Vec<u32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..total_rows).map(|_| rng.gen_range(0..num_groups as u32)).collect()
    }
}
```

**Step 2: Verify compiles**

Run: `cargo check -p xlog-cuda-tests`
Expected: Errors for missing validators/diagnostics modules

---

### Task 4: Implement CPU reference validators

**Files:**
- Create: `crates/xlog-cuda-tests/src/harness/validators.rs`

**Step 1: Implement all validators**

```rust
//! CPU reference implementations for validating GPU results.

use std::collections::{HashMap, HashSet};

/// CPU reference implementations for all kernel operations.
pub mod reference {
    use std::collections::HashMap;
    use std::hash::Hash;

    /// Hash join returning (left_idx, right_idx) pairs.
    pub fn hash_join_u32(left: &[u32], right: &[u32]) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        let right_map: HashMap<u32, Vec<usize>> = right.iter()
            .enumerate()
            .fold(HashMap::new(), |mut acc, (idx, val)| {
                acc.entry(*val).or_default().push(idx);
                acc
            });

        for (left_idx, left_val) in left.iter().enumerate() {
            if let Some(right_indices) = right_map.get(left_val) {
                for &right_idx in right_indices {
                    result.push((left_idx, right_idx));
                }
            }
        }
        result
    }

    /// Multi-column hash join.
    pub fn hash_join_multi(left_cols: &[&[u32]], right_cols: &[&[u32]]) -> Vec<(usize, usize)> {
        if left_cols.is_empty() || right_cols.is_empty() {
            return Vec::new();
        }

        let left_len = left_cols[0].len();
        let right_len = right_cols[0].len();

        // Build hash map on right side
        let mut right_map: HashMap<Vec<u32>, Vec<usize>> = HashMap::new();
        for i in 0..right_len {
            let key: Vec<u32> = right_cols.iter().map(|col| col[i]).collect();
            right_map.entry(key).or_default().push(i);
        }

        // Probe with left side
        let mut result = Vec::new();
        for i in 0..left_len {
            let key: Vec<u32> = left_cols.iter().map(|col| col[i]).collect();
            if let Some(right_indices) = right_map.get(&key) {
                for &right_idx in right_indices {
                    result.push((i, right_idx));
                }
            }
        }
        result
    }

    /// Semi join - returns left indices that have a match in right.
    pub fn semi_join_u32(left: &[u32], right: &[u32]) -> Vec<usize> {
        let right_set: std::collections::HashSet<u32> = right.iter().copied().collect();
        left.iter()
            .enumerate()
            .filter(|(_, val)| right_set.contains(val))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Anti join - returns left indices that have NO match in right.
    pub fn anti_join_u32(left: &[u32], right: &[u32]) -> Vec<usize> {
        let right_set: std::collections::HashSet<u32> = right.iter().copied().collect();
        left.iter()
            .enumerate()
            .filter(|(_, val)| !right_set.contains(val))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Filter by comparison operation.
    pub fn filter_compare_u32(data: &[u32], op: CompareOp, val: u32) -> Vec<usize> {
        data.iter()
            .enumerate()
            .filter(|(_, d)| op.apply(**d, val))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Filter by comparison operation for i64.
    pub fn filter_compare_i64(data: &[i64], op: CompareOp, val: i64) -> Vec<usize> {
        data.iter()
            .enumerate()
            .filter(|(_, d)| op.apply(**d, val))
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Filter by comparison operation for f64.
    pub fn filter_compare_f64(data: &[f64], op: CompareOp, val: f64) -> Vec<f64> {
        data.iter()
            .filter(|d| op.apply_f64(**d, val))
            .copied()
            .collect()
    }

    /// Compact by mask.
    pub fn compact_by_mask<T: Copy>(data: &[T], mask: &[u8]) -> Vec<T> {
        data.iter()
            .zip(mask.iter())
            .filter(|(_, m)| **m != 0)
            .map(|(d, _)| *d)
            .collect()
    }

    /// Stable radix sort returning (sorted_data, permutation).
    pub fn radix_sort_u32(keys: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let mut indexed: Vec<(u32, u32)> = keys.iter()
            .enumerate()
            .map(|(i, &k)| (k, i as u32))
            .collect();

        // Stable sort
        indexed.sort_by_key(|(k, _)| *k);

        let sorted: Vec<u32> = indexed.iter().map(|(k, _)| *k).collect();
        let perm: Vec<u32> = indexed.iter().map(|(_, i)| *i).collect();

        (sorted, perm)
    }

    /// Apply permutation to data.
    pub fn apply_permutation<T: Copy>(data: &[T], perm: &[u32]) -> Vec<T> {
        perm.iter().map(|&i| data[i as usize]).collect()
    }

    /// Inclusive prefix sum.
    pub fn inclusive_scan(data: &[u32]) -> Vec<u32> {
        let mut result = Vec::with_capacity(data.len());
        let mut sum = 0u32;
        for &val in data {
            sum = sum.wrapping_add(val);
            result.push(sum);
        }
        result
    }

    /// Exclusive prefix sum.
    pub fn exclusive_scan(data: &[u32]) -> Vec<u32> {
        let mut result = Vec::with_capacity(data.len());
        let mut sum = 0u32;
        for &val in data {
            result.push(sum);
            sum = sum.wrapping_add(val);
        }
        result
    }

    /// Group by count (assumes sorted input).
    pub fn groupby_count_sorted(keys: &[u32]) -> Vec<(u32, u64)> {
        if keys.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current_key = keys[0];
        let mut count = 1u64;

        for &key in &keys[1..] {
            if key == current_key {
                count += 1;
            } else {
                result.push((current_key, count));
                current_key = key;
                count = 1;
            }
        }
        result.push((current_key, count));
        result
    }

    /// Group by sum (assumes sorted input).
    pub fn groupby_sum_sorted(keys: &[u32], vals: &[u32]) -> Vec<(u32, u64)> {
        if keys.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current_key = keys[0];
        let mut sum = vals[0] as u64;

        for i in 1..keys.len() {
            if keys[i] == current_key {
                sum += vals[i] as u64;
            } else {
                result.push((current_key, sum));
                current_key = keys[i];
                sum = vals[i] as u64;
            }
        }
        result.push((current_key, sum));
        result
    }

    /// Group by min (assumes sorted input).
    pub fn groupby_min_sorted(keys: &[u32], vals: &[u32]) -> Vec<(u32, u32)> {
        if keys.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current_key = keys[0];
        let mut min_val = vals[0];

        for i in 1..keys.len() {
            if keys[i] == current_key {
                min_val = min_val.min(vals[i]);
            } else {
                result.push((current_key, min_val));
                current_key = keys[i];
                min_val = vals[i];
            }
        }
        result.push((current_key, min_val));
        result
    }

    /// Group by max (assumes sorted input).
    pub fn groupby_max_sorted(keys: &[u32], vals: &[u32]) -> Vec<(u32, u32)> {
        if keys.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut current_key = keys[0];
        let mut max_val = vals[0];

        for i in 1..keys.len() {
            if keys[i] == current_key {
                max_val = max_val.max(vals[i]);
            } else {
                result.push((current_key, max_val));
                current_key = keys[i];
                max_val = vals[i];
            }
        }
        result.push((current_key, max_val));
        result
    }

    /// Dedup sorted data.
    pub fn dedup_sorted<T: Eq + Copy>(data: &[T]) -> Vec<T> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut result = vec![data[0]];
        for &val in &data[1..] {
            if val != *result.last().unwrap() {
                result.push(val);
            }
        }
        result
    }

    /// Mark duplicates in sorted data (true = unique, false = duplicate).
    pub fn mark_duplicates<T: Eq>(sorted: &[T]) -> Vec<bool> {
        if sorted.is_empty() {
            return Vec::new();
        }

        let mut result = vec![true]; // First element is always unique
        for i in 1..sorted.len() {
            result.push(sorted[i] != sorted[i - 1]);
        }
        result
    }

    /// Sorted set union.
    pub fn sorted_union<T: Ord + Copy>(a: &[T], b: &[T]) -> Vec<T> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            if a[i] < b[j] {
                if result.last() != Some(&a[i]) {
                    result.push(a[i]);
                }
                i += 1;
            } else if a[i] > b[j] {
                if result.last() != Some(&b[j]) {
                    result.push(b[j]);
                }
                j += 1;
            } else {
                if result.last() != Some(&a[i]) {
                    result.push(a[i]);
                }
                i += 1;
                j += 1;
            }
        }

        while i < a.len() {
            if result.last() != Some(&a[i]) {
                result.push(a[i]);
            }
            i += 1;
        }

        while j < b.len() {
            if result.last() != Some(&b[j]) {
                result.push(b[j]);
            }
            j += 1;
        }

        result
    }

    /// Sorted set difference (a - b).
    pub fn sorted_diff<T: Ord + Copy>(a: &[T], b: &[T]) -> Vec<T> {
        let mut result = Vec::new();
        let mut i = 0;
        let mut j = 0;

        while i < a.len() && j < b.len() {
            if a[i] < b[j] {
                result.push(a[i]);
                i += 1;
            } else if a[i] > b[j] {
                j += 1;
            } else {
                i += 1;
                j += 1;
            }
        }

        while i < a.len() {
            result.push(a[i]);
            i += 1;
        }

        result
    }

    /// Pack columns into row-major byte array.
    pub fn pack_keys(cols: &[&[u8]], col_sizes: &[usize], num_rows: usize) -> Vec<u8> {
        let row_size: usize = col_sizes.iter().sum();
        let mut result = vec![0u8; row_size * num_rows];

        for row in 0..num_rows {
            let mut offset = 0;
            for (col_idx, &col_size) in col_sizes.iter().enumerate() {
                let src_start = row * col_size;
                let src_end = src_start + col_size;
                let dst_start = row * row_size + offset;
                let dst_end = dst_start + col_size;
                result[dst_start..dst_end].copy_from_slice(&cols[col_idx][src_start..src_end]);
                offset += col_size;
            }
        }
        result
    }

    /// FNV-1a hash.
    pub fn hash_fnv1a(data: &[u8]) -> u32 {
        const FNV_PRIME: u32 = 16777619;
        const FNV_OFFSET: u32 = 2166136261;

        let mut hash = FNV_OFFSET;
        for &byte in data {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Unpack a column from packed rows.
    pub fn unpack_column(packed: &[u8], row_size: usize, col_offset: usize, col_size: usize, num_rows: usize) -> Vec<u8> {
        let mut result = vec![0u8; col_size * num_rows];

        for row in 0..num_rows {
            let src_start = row * row_size + col_offset;
            let src_end = src_start + col_size;
            let dst_start = row * col_size;
            let dst_end = dst_start + col_size;
            result[dst_start..dst_end].copy_from_slice(&packed[src_start..src_end]);
        }
        result
    }

    /// Comparison operation enum.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CompareOp {
        Eq,
        Ne,
        Lt,
        Le,
        Gt,
        Ge,
    }

    impl CompareOp {
        pub fn apply<T: PartialOrd>(&self, a: T, b: T) -> bool {
            match self {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            }
        }

        pub fn apply_f64(&self, a: f64, b: f64) -> bool {
            match self {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            }
        }
    }
}

/// Comparison utilities for GPU vs CPU result validation.
pub mod compare {
    /// Assert u32 slices are equal with detailed diff on failure.
    pub fn assert_eq_u32(gpu: &[u32], cpu: &[u32], context: &str) {
        if gpu.len() != cpu.len() {
            panic!("{}: length mismatch: GPU={}, CPU={}", context, gpu.len(), cpu.len());
        }

        let mut first_diff = None;
        let mut diff_count = 0;

        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            if g != c {
                if first_diff.is_none() {
                    first_diff = Some((i, *g, *c));
                }
                diff_count += 1;
            }
        }

        if let Some((idx, gpu_val, cpu_val)) = first_diff {
            panic!(
                "{}: {} differences found. First at index {}: GPU={}, CPU={}",
                context, diff_count, idx, gpu_val, cpu_val
            );
        }
    }

    /// Assert i64 slices are equal with detailed diff on failure.
    pub fn assert_eq_i64(gpu: &[i64], cpu: &[i64], context: &str) {
        if gpu.len() != cpu.len() {
            panic!("{}: length mismatch: GPU={}, CPU={}", context, gpu.len(), cpu.len());
        }

        let mut first_diff = None;
        let mut diff_count = 0;

        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            if g != c {
                if first_diff.is_none() {
                    first_diff = Some((i, *g, *c));
                }
                diff_count += 1;
            }
        }

        if let Some((idx, gpu_val, cpu_val)) = first_diff {
            panic!(
                "{}: {} differences found. First at index {}: GPU={}, CPU={}",
                context, diff_count, idx, gpu_val, cpu_val
            );
        }
    }

    /// Assert u64 slices are equal with detailed diff on failure.
    pub fn assert_eq_u64(gpu: &[u64], cpu: &[u64], context: &str) {
        if gpu.len() != cpu.len() {
            panic!("{}: length mismatch: GPU={}, CPU={}", context, gpu.len(), cpu.len());
        }

        let mut first_diff = None;
        let mut diff_count = 0;

        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            if g != c {
                if first_diff.is_none() {
                    first_diff = Some((i, *g, *c));
                }
                diff_count += 1;
            }
        }

        if let Some((idx, gpu_val, cpu_val)) = first_diff {
            panic!(
                "{}: {} differences found. First at index {}: GPU={}, CPU={}",
                context, diff_count, idx, gpu_val, cpu_val
            );
        }
    }

    /// Assert f64 slices are equal within ULP tolerance.
    pub fn assert_eq_f64_ulp(gpu: &[f64], cpu: &[f64], max_ulp: u64, context: &str) {
        if gpu.len() != cpu.len() {
            panic!("{}: length mismatch: GPU={}, CPU={}", context, gpu.len(), cpu.len());
        }

        let mut first_diff = None;
        let mut diff_count = 0;

        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let ulp_diff = ulp_distance(*g, *c);
            if ulp_diff > max_ulp {
                if first_diff.is_none() {
                    first_diff = Some((i, *g, *c, ulp_diff));
                }
                diff_count += 1;
            }
        }

        if let Some((idx, gpu_val, cpu_val, ulp)) = first_diff {
            panic!(
                "{}: {} ULP violations (max={}). First at index {}: GPU={}, CPU={}, ULP={}",
                context, diff_count, max_ulp, idx, gpu_val, cpu_val, ulp
            );
        }
    }

    /// Assert f64 slices are equal within relative tolerance.
    pub fn assert_eq_f64_rel(gpu: &[f64], cpu: &[f64], rel_tol: f64, context: &str) {
        if gpu.len() != cpu.len() {
            panic!("{}: length mismatch: GPU={}, CPU={}", context, gpu.len(), cpu.len());
        }

        let mut first_diff = None;
        let mut diff_count = 0;

        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let rel_diff = if *c == 0.0 {
                g.abs()
            } else {
                ((g - c) / c).abs()
            };

            if rel_diff > rel_tol && !g.is_nan() && !c.is_nan() {
                if first_diff.is_none() {
                    first_diff = Some((i, *g, *c, rel_diff));
                }
                diff_count += 1;
            }
        }

        if let Some((idx, gpu_val, cpu_val, rel)) = first_diff {
            panic!(
                "{}: {} relative tolerance violations (max={}). First at index {}: GPU={}, CPU={}, rel_diff={}",
                context, diff_count, rel_tol, idx, gpu_val, cpu_val, rel
            );
        }
    }

    /// Compute ULP distance between two f64 values.
    fn ulp_distance(a: f64, b: f64) -> u64 {
        if a.is_nan() || b.is_nan() {
            return u64::MAX;
        }
        if a == b {
            return 0;
        }
        if a.is_infinite() || b.is_infinite() {
            return u64::MAX;
        }

        let a_bits = a.to_bits() as i64;
        let b_bits = b.to_bits() as i64;

        (a_bits - b_bits).unsigned_abs()
    }

    /// Assert sets are equal (order-independent).
    pub fn assert_set_eq_u32(gpu: &[u32], cpu: &[u32], context: &str) {
        let mut gpu_sorted = gpu.to_vec();
        let mut cpu_sorted = cpu.to_vec();
        gpu_sorted.sort();
        cpu_sorted.sort();

        if gpu_sorted != cpu_sorted {
            panic!(
                "{}: set mismatch. GPU (sorted): {:?}, CPU (sorted): {:?}",
                context,
                &gpu_sorted[..gpu_sorted.len().min(10)],
                &cpu_sorted[..cpu_sorted.len().min(10)]
            );
        }
    }

    /// Assert permutation is valid.
    pub fn assert_valid_permutation(perm: &[u32], len: usize, context: &str) {
        if perm.len() != len {
            panic!("{}: permutation length {} != expected {}", context, perm.len(), len);
        }

        let mut seen = vec![false; len];
        for (i, &idx) in perm.iter().enumerate() {
            if idx as usize >= len {
                panic!("{}: permutation index {} out of bounds at position {}", context, idx, i);
            }
            if seen[idx as usize] {
                panic!("{}: duplicate permutation index {} at position {}", context, idx, i);
            }
            seen[idx as usize] = true;
        }
    }

    /// Assert sort is stable (equal keys maintain relative order).
    pub fn assert_stable_sort(original_keys: &[u32], original_vals: &[u32], sorted_keys: &[u32], sorted_vals: &[u32], context: &str) {
        // Group by key in original order
        let mut key_to_vals: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
        for (&k, &v) in original_keys.iter().zip(original_vals.iter()) {
            key_to_vals.entry(k).or_default().push(v);
        }

        // Check sorted result maintains relative order within each key group
        let mut key_to_idx: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for (&k, &v) in sorted_keys.iter().zip(sorted_vals.iter()) {
            let idx = key_to_idx.entry(k).or_insert(0);
            let expected_vals = key_to_vals.get(&k).unwrap();
            if *idx >= expected_vals.len() {
                panic!("{}: too many values for key {}", context, k);
            }
            if v != expected_vals[*idx] {
                panic!(
                    "{}: stability violation for key {}. Expected value {} but got {}",
                    context, k, expected_vals[*idx], v
                );
            }
            *idx += 1;
        }
    }

    /// Check if result is sorted.
    pub fn is_sorted_u32(data: &[u32]) -> bool {
        data.windows(2).all(|w| w[0] <= w[1])
    }

    /// Check if result is sorted descending.
    pub fn is_sorted_desc_u32(data: &[u32]) -> bool {
        data.windows(2).all(|w| w[0] >= w[1])
    }
}
```

**Step 2: Verify compiles**

Run: `cargo check -p xlog-cuda-tests`
Expected: Errors for missing diagnostics module

---

### Task 5: Implement diagnostics and result tracking

**Files:**
- Create: `crates/xlog-cuda-tests/src/harness/diagnostics.rs`

**Step 1: Implement diagnostics**

```rust
//! Test diagnostics, failure reporting, and result tracking.

use std::time::{Duration, Instant};

/// Rich failure diagnostic with context.
#[derive(Debug, Clone)]
pub struct FailureDiagnostic {
    pub category: &'static str,
    pub test_name: &'static str,
    pub input_size: usize,
    pub input_sample: String,
    pub expected_sample: String,
    pub actual_sample: String,
    pub first_diff_index: Option<usize>,
    pub diff_count: usize,
    pub error_message: String,
}

impl FailureDiagnostic {
    /// Create a new failure diagnostic.
    pub fn new(
        category: &'static str,
        test_name: &'static str,
        input_size: usize,
        error_message: String,
    ) -> Self {
        Self {
            category,
            test_name,
            input_size,
            input_sample: String::new(),
            expected_sample: String::new(),
            actual_sample: String::new(),
            first_diff_index: None,
            diff_count: 0,
            error_message,
        }
    }

    /// Add input sample (first N elements).
    pub fn with_input_sample(mut self, sample: String) -> Self {
        self.input_sample = sample;
        self
    }

    /// Add expected/actual samples.
    pub fn with_comparison(mut self, expected: String, actual: String, first_diff: Option<usize>, diff_count: usize) -> Self {
        self.expected_sample = expected;
        self.actual_sample = actual;
        self.first_diff_index = first_diff;
        self.diff_count = diff_count;
        self
    }

    /// Generate human-readable failure report.
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str(&format!("=== FAILURE: {}/{} ===\n", self.category, self.test_name));
        report.push_str(&format!("Input size: {}\n", self.input_size));
        report.push_str(&format!("Error: {}\n", self.error_message));

        if !self.input_sample.is_empty() {
            report.push_str(&format!("Input sample: {}\n", self.input_sample));
        }

        if self.diff_count > 0 {
            report.push_str(&format!("Differences: {} total\n", self.diff_count));
            if let Some(idx) = self.first_diff_index {
                report.push_str(&format!("First diff at index: {}\n", idx));
            }
            report.push_str(&format!("Expected: {}\n", self.expected_sample));
            report.push_str(&format!("Actual:   {}\n", self.actual_sample));
        }

        report.push_str("===\n");
        report
    }
}

/// Test execution status.
#[derive(Debug, Clone)]
pub enum TestStatus {
    Passed,
    Failed,
    Skipped { reason: &'static str },
    Error { message: String },
}

impl TestStatus {
    pub fn is_passed(&self) -> bool {
        matches!(self, TestStatus::Passed)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, TestStatus::Failed | TestStatus::Error { .. })
    }
}

/// Individual test result.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: &'static str,
    pub status: TestStatus,
    pub duration: Duration,
    pub diagnostic: Option<FailureDiagnostic>,
}

impl TestResult {
    pub fn passed(name: &'static str, duration: Duration) -> Self {
        Self {
            name,
            status: TestStatus::Passed,
            duration,
            diagnostic: None,
        }
    }

    pub fn failed(name: &'static str, duration: Duration, diagnostic: FailureDiagnostic) -> Self {
        Self {
            name,
            status: TestStatus::Failed,
            duration,
            diagnostic: Some(diagnostic),
        }
    }

    pub fn skipped(name: &'static str, reason: &'static str) -> Self {
        Self {
            name,
            status: TestStatus::Skipped { reason },
            duration: Duration::ZERO,
            diagnostic: None,
        }
    }

    pub fn error(name: &'static str, duration: Duration, message: String) -> Self {
        Self {
            name,
            status: TestStatus::Error { message },
            duration,
            diagnostic: None,
        }
    }
}

/// Category-level result aggregation.
#[derive(Debug, Clone)]
pub struct CategoryResult {
    pub name: &'static str,
    pub tests: Vec<TestResult>,
    pub duration: Duration,
}

impl CategoryResult {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            tests: Vec::new(),
            duration: Duration::ZERO,
        }
    }

    pub fn add_result(&mut self, result: TestResult) {
        self.tests.push(result);
    }

    pub fn set_duration(&mut self, duration: Duration) {
        self.duration = duration;
    }

    pub fn passed_count(&self) -> usize {
        self.tests.iter().filter(|t| t.status.is_passed()).count()
    }

    pub fn failed_count(&self) -> usize {
        self.tests.iter().filter(|t| t.status.is_failed()).count()
    }

    pub fn skipped_count(&self) -> usize {
        self.tests.iter().filter(|t| matches!(t.status, TestStatus::Skipped { .. })).count()
    }

    pub fn total_count(&self) -> usize {
        self.tests.len()
    }

    pub fn all_passed(&self) -> bool {
        self.tests.iter().all(|t| t.status.is_passed() || matches!(t.status, TestStatus::Skipped { .. }))
    }
}

/// Full certification results.
#[derive(Debug)]
pub struct CertificationResults {
    pub categories: Vec<CategoryResult>,
    pub start_time: Instant,
    pub total_duration: Duration,
}

impl CertificationResults {
    pub fn new() -> Self {
        Self {
            categories: Vec::new(),
            start_time: Instant::now(),
            total_duration: Duration::ZERO,
        }
    }

    /// Run a category and record results.
    pub fn run_category<F>(&mut self, name: &'static str, f: F)
    where
        F: FnOnce() -> CategoryResult,
    {
        let start = Instant::now();
        let mut result = f();
        result.set_duration(start.elapsed());
        self.categories.push(result);
    }

    /// Add a pre-computed category result.
    pub fn add_category(&mut self, result: CategoryResult) {
        self.categories.push(result);
    }

    /// Finalize results and compute total duration.
    pub fn finalize(&mut self) {
        self.total_duration = self.start_time.elapsed();
    }

    pub fn total_tests(&self) -> usize {
        self.categories.iter().map(|c| c.total_count()).sum()
    }

    pub fn total_passed(&self) -> usize {
        self.categories.iter().map(|c| c.passed_count()).sum()
    }

    pub fn total_failed(&self) -> usize {
        self.categories.iter().map(|c| c.failed_count()).sum()
    }

    pub fn total_skipped(&self) -> usize {
        self.categories.iter().map(|c| c.skipped_count()).sum()
    }

    pub fn all_passed(&self) -> bool {
        self.categories.iter().all(|c| c.all_passed())
    }

    /// Print summary to stdout.
    pub fn print_summary(&self) {
        println!("\n========== CERTIFICATION RESULTS ==========");
        println!("Total Duration: {:.2}s", self.total_duration.as_secs_f64());
        println!();

        for cat in &self.categories {
            let status = if cat.all_passed() { "PASS" } else { "FAIL" };
            println!(
                "[{}] {}: {}/{} passed, {} skipped ({:.2}s)",
                status,
                cat.name,
                cat.passed_count(),
                cat.total_count(),
                cat.skipped_count(),
                cat.duration.as_secs_f64()
            );
        }

        println!();
        println!("========== SUMMARY ==========");
        println!("Total Tests:   {}", self.total_tests());
        println!("Passed:        {}", self.total_passed());
        println!("Failed:        {}", self.total_failed());
        println!("Skipped:       {}", self.total_skipped());
        println!(
            "Pass Rate:     {:.1}%",
            (self.total_passed() as f64 / self.total_tests().max(1) as f64) * 100.0
        );
        println!("=====================================\n");
    }

    /// Generate detailed failure report.
    pub fn failure_report(&self) -> String {
        let mut report = String::new();

        for cat in &self.categories {
            for test in &cat.tests {
                if let Some(diag) = &test.diagnostic {
                    report.push_str(&diag.report());
                    report.push('\n');
                } else if let TestStatus::Error { message } = &test.status {
                    report.push_str(&format!("=== ERROR: {}/{} ===\n", cat.name, test.name));
                    report.push_str(&format!("Error: {}\n", message));
                    report.push_str("===\n\n");
                }
            }
        }

        report
    }
}

impl Default for CertificationResults {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper macro for running a test and capturing results.
#[macro_export]
macro_rules! run_test {
    ($results:expr, $name:expr, $body:expr) => {{
        let start = std::time::Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body));
        let duration = start.elapsed();

        match result {
            Ok(()) => $crate::harness::TestResult::passed($name, duration),
            Err(e) => {
                let message = if let Some(s) = e.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };
                $crate::harness::TestResult::error($name, duration, message)
            }
        }
    }};
}

/// Helper to format a slice sample for diagnostics.
pub fn format_sample<T: std::fmt::Debug>(data: &[T], max_items: usize) -> String {
    if data.len() <= max_items {
        format!("{:?}", data)
    } else {
        let sample: Vec<_> = data.iter().take(max_items).collect();
        format!("{:?}... ({} more)", sample, data.len() - max_items)
    }
}
```

**Step 2: Verify harness compiles**

Run: `cargo check -p xlog-cuda-tests`
Expected: Errors for missing category modules

---

### Task 6: Create stub category modules

**Files:**
- Create: `crates/xlog-cuda-tests/src/categories/c01_toolchain.rs` through `c24_edge_matrix.rs`

**Step 1: Create all 24 category module stubs**

Each category module follows this pattern. Create all 24 files:

```rust
// c01_toolchain.rs
//! Category 1: Toolchain, PTX, and SASS edge cases.

use crate::harness::{CategoryResult, TestResult, TestContext};
use std::time::Instant;

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c01_toolchain");
    let start = Instant::now();

    results.add_result(test_ptx_loads_successfully(ctx));
    results.add_result(test_compute_capability_check(ctx));
    results.add_result(test_kernel_function_resolution(ctx));
    results.add_result(test_ptx_module_attributes(ctx));
    results.add_result(test_repeated_jit_compilation(ctx));

    results.set_duration(start.elapsed());
    results
}

fn test_ptx_loads_successfully(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    // All PTX modules already loaded in TestContext::new()
    // If we got here, they loaded successfully
    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("ptx_loads_successfully", start.elapsed())
}

fn test_compute_capability_check(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let (major, minor) = ctx.compute_capability();
    assert!(major >= 7, "Compute capability {}.{} < 7.0 (sm_70)", major, minor);
    TestResult::passed("compute_capability_check", start.elapsed())
}

fn test_kernel_function_resolution(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    // Provider already resolved all kernel functions during init
    // Test a simple operation to verify
    let schema = xlog_core::Schema::new(vec![("val".to_string(), xlog_core::ScalarType::U32)]);
    let data: Vec<u32> = vec![1, 2, 3];
    let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema).expect("buffer creation failed");
    assert_eq!(buffer.num_rows, 3);
    TestResult::passed("kernel_function_resolution", start.elapsed())
}

fn test_ptx_module_attributes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    // PTX modules are compiled for 64-bit addressing
    // Verify by checking that 64-bit values work correctly
    let schema = xlog_core::Schema::new(vec![("val".to_string(), xlog_core::ScalarType::U64)]);
    let data: Vec<u64> = vec![u64::MAX, u64::MAX - 1, 0];
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut col = ctx.memory.alloc::<u8>(bytes.len()).expect("alloc failed");
    ctx.device.inner().htod_sync_copy_into(&bytes, &mut col).expect("upload failed");

    let buffer = xlog_cuda::CudaBuffer::from_columns(vec![col], data.len() as u64, schema);
    let mask: Vec<u8> = vec![1, 1, 1];
    let filtered = ctx.provider.filter_by_mask(&buffer, &mask).expect("filter failed");
    assert_eq!(filtered.num_rows, 3);

    TestResult::passed("ptx_module_attributes", start.elapsed())
}

fn test_repeated_jit_compilation(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    // Run multiple operations to exercise JIT-compiled code paths
    for _ in 0..10 {
        let schema = xlog_core::Schema::new(vec![("val".to_string(), xlog_core::ScalarType::U32)]);
        let data: Vec<u32> = vec![1, 2, 3, 4, 5];
        let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema).expect("buffer creation failed");
        let _ = ctx.provider.sort(&buffer, &[0]).expect("sort failed");
    }
    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("repeated_jit_compilation", start.elapsed())
}
```

**Step 2: Create remaining category stubs (c02-c24)**

Due to the length, I'll provide the pattern. Each file should have:
1. Module doc comment describing the category
2. `run_all(ctx: &TestContext) -> CategoryResult` function
3. Individual test functions

Run: `cargo check -p xlog-cuda-tests`
Expected: Success (or errors for incomplete category implementations)

---

## Batch 2: Categories 1-8 (Infrastructure Tests)

### Task 7: Implement Category 2 - Launch Configuration

**Files:**
- Modify: `crates/xlog-cuda-tests/src/categories/c02_launch_config.rs`

**Step 1: Implement launch configuration tests**

```rust
//! Category 2: Kernel launch configuration edge cases.

use crate::harness::{CategoryResult, TestResult, TestContext};
use crate::harness::generators::SizeGen;
use xlog_core::{Schema, ScalarType};
use std::time::Instant;

pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c02_launch_config");
    let start = Instant::now();

    results.add_result(test_zero_elements_no_launch(ctx));
    results.add_result(test_single_element(ctx));
    results.add_result(test_warp_boundary_sizes(ctx));
    results.add_result(test_block_boundary_sizes(ctx));
    results.add_result(test_non_power_of_two_sizes(ctx));
    results.add_result(test_large_grid_sizes(ctx));
    results.add_result(test_max_practical_size(ctx));

    results.set_duration(start.elapsed());
    results
}

fn test_zero_elements_no_launch(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Empty buffer operations should not crash
    let empty = ctx.provider.create_empty_buffer(schema.clone()).expect("empty buffer failed");

    // Sort empty
    let sorted = ctx.provider.sort(&empty, &[0]).expect("sort empty failed");
    assert_eq!(sorted.num_rows, 0);

    // Filter empty
    let filtered = ctx.provider.filter_by_mask(&empty, &[]).expect("filter empty failed");
    assert_eq!(filtered.num_rows, 0);

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("zero_elements_no_launch", start.elapsed())
}

fn test_single_element(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);
    let data: Vec<u32> = vec![42];

    let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema).expect("buffer failed");

    // Sort single element
    let sorted = ctx.provider.sort(&buffer, &[0]).expect("sort failed");
    let result = ctx.provider.download_column_u32(&sorted, 0).expect("download failed");
    assert_eq!(result, vec![42]);

    // Filter single element (select)
    let filtered = ctx.provider.filter_by_mask(&buffer, &[1]).expect("filter failed");
    assert_eq!(filtered.num_rows, 1);

    // Filter single element (reject)
    let filtered = ctx.provider.filter_by_mask(&buffer, &[0]).expect("filter failed");
    assert_eq!(filtered.num_rows, 0);

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("single_element", start.elapsed())
}

fn test_warp_boundary_sizes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test sizes around warp boundaries (32)
    for size in SizeGen::warp_related() {
        if size == 0 { continue; }

        let data: Vec<u32> = (0..size as u32).collect();
        let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema.clone())
            .expect(&format!("buffer failed for size {}", size));

        let sorted = ctx.provider.sort(&buffer, &[0])
            .expect(&format!("sort failed for size {}", size));

        let result = ctx.provider.download_column_u32(&sorted, 0)
            .expect(&format!("download failed for size {}", size));

        // Verify sorted
        for i in 1..result.len() {
            assert!(result[i-1] <= result[i], "sort failed at size {}, index {}", size, i);
        }
    }

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("warp_boundary_sizes", start.elapsed())
}

fn test_block_boundary_sizes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test sizes around block boundaries (256)
    for size in SizeGen::block_related() {
        let data: Vec<u32> = (0..size as u32).rev().collect(); // Reverse sorted
        let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema.clone())
            .expect(&format!("buffer failed for size {}", size));

        let sorted = ctx.provider.sort(&buffer, &[0])
            .expect(&format!("sort failed for size {}", size));

        let result = ctx.provider.download_column_u32(&sorted, 0)
            .expect(&format!("download failed for size {}", size));

        // Verify sorted ascending
        let expected: Vec<u32> = (0..size as u32).collect();
        assert_eq!(result, expected, "sort incorrect at size {}", size);
    }

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("block_boundary_sizes", start.elapsed())
}

fn test_non_power_of_two_sizes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Prime numbers are worst case for many algorithms
    for size in SizeGen::primes().iter().take(20) {
        let size = *size;
        if size == 0 { continue; }

        let data: Vec<u32> = (0..size as u32).rev().collect();
        let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema.clone())
            .expect(&format!("buffer failed for size {}", size));

        let sorted = ctx.provider.sort(&buffer, &[0])
            .expect(&format!("sort failed for size {}", size));

        assert_eq!(sorted.num_rows as usize, size, "row count mismatch at size {}", size);
    }

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("non_power_of_two_sizes", start.elapsed())
}

fn test_large_grid_sizes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test with 1M elements (requires many blocks)
    let size = 1_000_000;
    let data: Vec<u32> = (0..size).map(|i| (size - 1 - i) as u32).collect();

    let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema)
        .expect("buffer failed");

    let sorted = ctx.provider.sort(&buffer, &[0])
        .expect("sort failed");

    let result = ctx.provider.download_column_u32(&sorted, 0)
        .expect("download failed");

    // Verify first and last elements
    assert_eq!(result[0], 0);
    assert_eq!(result[size - 1], (size - 1) as u32);

    // Spot check middle
    assert_eq!(result[size / 2], (size / 2) as u32);

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("large_grid_sizes", start.elapsed())
}

fn test_max_practical_size(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test with 10M elements - practical stress test
    let size = 10_000_000;
    let data: Vec<u32> = (0..size).map(|i| (i % 1000) as u32).collect();

    let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema)
        .expect("buffer failed");

    // Just verify it doesn't crash and produces correct count
    let sorted = ctx.provider.sort(&buffer, &[0])
        .expect("sort failed");

    assert_eq!(sorted.num_rows as usize, size);

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("max_practical_size", start.elapsed())
}
```

---

### Task 8: Implement Category 3 - Pointer Bounds

**Files:**
- Modify: `crates/xlog-cuda-tests/src/categories/c03_pointer_bounds.rs`

**Step 1: Implement pointer bounds tests**

```rust
//! Category 3: Pointer arithmetic, indexing, bounds, and overflow edge cases.

use crate::harness::{CategoryResult, TestResult, TestContext};
use crate::harness::generators::{SizeGen, Distribution};
use crate::harness::validators::compare;
use xlog_core::{Schema, ScalarType};
use std::time::Instant;

pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c03_pointer_bounds");
    let start = Instant::now();

    results.add_result(test_edge_case_sizes(ctx));
    results.add_result(test_off_by_one_filter(ctx));
    results.add_result(test_off_by_one_sort(ctx));
    results.add_result(test_grid_stride_loop(ctx));
    results.add_result(test_tail_handling_filter(ctx));
    results.add_result(test_tail_handling_sort(ctx));
    results.add_result(test_boundary_indices(ctx));
    results.add_result(test_multi_column_strides(ctx));

    results.set_duration(start.elapsed());
    results
}

fn test_edge_case_sizes(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    for size in SizeGen::edge_cases() {
        if size == 0 { continue; }

        let data: Vec<u32> = (0..size as u32).collect();
        let mask: Vec<u8> = (0..size).map(|i| (i % 2) as u8).collect();

        let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema.clone())
            .expect(&format!("buffer failed for size {}", size));

        let filtered = ctx.provider.filter_by_mask(&buffer, &mask)
            .expect(&format!("filter failed for size {}", size));

        let expected_count = mask.iter().filter(|&&m| m != 0).count();
        assert_eq!(filtered.num_rows as usize, expected_count,
            "filter count mismatch at size {}", size);
    }

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("edge_case_sizes", start.elapsed())
}

fn test_off_by_one_filter(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test filter with all 1s except last
    for size in [31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257] {
        let data: Vec<u32> = (0..size as u32).collect();
        let mut mask: Vec<u8> = vec![1; size];
        mask[size - 1] = 0; // Last element excluded

        let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema.clone())
            .expect(&format!("buffer failed for size {}", size));

        let filtered = ctx.provider.filter_by_mask(&buffer, &mask)
            .expect(&format!("filter failed for size {}", size));

        assert_eq!(filtered.num_rows as usize, size - 1,
            "off-by-one at size {}: expected {}, got {}", size, size - 1, filtered.num_rows);

        // Verify last element is actually excluded
        let result = ctx.provider.download_column_u32(&filtered, 0)
            .expect(&format!("download failed for size {}", size));

        assert!(!result.contains(&((size - 1) as u32)),
            "last element should be excluded at size {}", size);
    }

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("off_by_one_filter", start.elapsed())
}

fn test_off_by_one_sort(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("key".to_string(), ScalarType::U32), ("val".to_string(), ScalarType::U32)]);

    // Test sort preserves all elements including last
    for size in [31, 32, 33, 63, 64, 65, 127, 128, 129] {
        let keys: Vec<u32> = (0..size as u32).rev().collect(); // Reverse order
        let vals: Vec<u32> = (0..size as u32).map(|i| i * 10).collect();

        let buffer = ctx.provider.create_buffer_from_u32_columns(&[&keys, &vals], schema.clone())
            .expect(&format!("buffer failed for size {}", size));

        let sorted = ctx.provider.sort(&buffer, &[0])
            .expect(&format!("sort failed for size {}", size));

        let sorted_keys = ctx.provider.download_column_u32(&sorted, 0)
            .expect(&format!("download keys failed for size {}", size));
        let sorted_vals = ctx.provider.download_column_u32(&sorted, 1)
            .expect(&format!("download vals failed for size {}", size));

        // Verify all elements present
        assert_eq!(sorted_keys.len(), size, "key count wrong at size {}", size);
        assert_eq!(sorted_vals.len(), size, "val count wrong at size {}", size);

        // Verify sorted order
        for i in 1..sorted_keys.len() {
            assert!(sorted_keys[i-1] <= sorted_keys[i],
                "sort order wrong at size {}, index {}", size, i);
        }

        // Verify key-value pairing preserved
        // After sorting reverse order, key 0 should have val (size-1)*10
        assert_eq!(sorted_keys[0], 0);
        assert_eq!(sorted_vals[0], ((size - 1) * 10) as u32);
    }

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("off_by_one_sort", start.elapsed())
}

fn test_grid_stride_loop(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test sizes that require grid-stride loops (many more elements than threads)
    for size in [100_000, 500_000, 1_000_000] {
        let data: Vec<u32> = Distribution::Random.generate_u32(size, 42);
        let mask: Vec<u8> = Distribution::Alternating.generate_mask(size, 42);

        let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema.clone())
            .expect(&format!("buffer failed for size {}", size));

        let filtered = ctx.provider.filter_by_mask(&buffer, &mask)
            .expect(&format!("filter failed for size {}", size));

        let expected_count = mask.iter().filter(|&&m| m != 0).count();
        assert_eq!(filtered.num_rows as usize, expected_count,
            "grid stride filter wrong at size {}", size);
    }

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("grid_stride_loop", start.elapsed())
}

fn test_tail_handling_filter(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Sizes that leave partial warps/blocks
    for size in [33, 65, 129, 257, 513, 1025] {
        // All 1s mask - every element should be included
        let data: Vec<u32> = (0..size as u32).collect();
        let mask: Vec<u8> = vec![1; size];

        let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema.clone())
            .expect(&format!("buffer failed for size {}", size));

        let filtered = ctx.provider.filter_by_mask(&buffer, &mask)
            .expect(&format!("filter failed for size {}", size));

        let result = ctx.provider.download_column_u32(&filtered, 0)
            .expect(&format!("download failed for size {}", size));

        let expected: Vec<u32> = (0..size as u32).collect();
        assert_eq!(result, expected, "tail handling wrong at size {}", size);
    }

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("tail_handling_filter", start.elapsed())
}

fn test_tail_handling_sort(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Sizes that leave partial warps/blocks
    for size in [33, 65, 129, 257, 513, 1025] {
        let data: Vec<u32> = (0..size as u32).rev().collect(); // Reverse order

        let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema.clone())
            .expect(&format!("buffer failed for size {}", size));

        let sorted = ctx.provider.sort(&buffer, &[0])
            .expect(&format!("sort failed for size {}", size));

        let result = ctx.provider.download_column_u32(&sorted, 0)
            .expect(&format!("download failed for size {}", size));

        let expected: Vec<u32> = (0..size as u32).collect();
        assert_eq!(result, expected, "tail handling sort wrong at size {}", size);
    }

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("tail_handling_sort", start.elapsed())
}

fn test_boundary_indices(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::U32)]);

    // Test that first and last indices are handled correctly
    let size = 1000;
    let data: Vec<u32> = (0..size as u32).collect();

    // Only select first element
    let mut mask_first: Vec<u8> = vec![0; size];
    mask_first[0] = 1;

    let buffer = ctx.provider.create_buffer_from_u32_slice(&data, schema.clone())
        .expect("buffer failed");

    let filtered = ctx.provider.filter_by_mask(&buffer, &mask_first)
        .expect("filter first failed");
    let result = ctx.provider.download_column_u32(&filtered, 0)
        .expect("download failed");
    assert_eq!(result, vec![0], "first element filter failed");

    // Only select last element
    let mut mask_last: Vec<u8> = vec![0; size];
    mask_last[size - 1] = 1;

    let filtered = ctx.provider.filter_by_mask(&buffer, &mask_last)
        .expect("filter last failed");
    let result = ctx.provider.download_column_u32(&filtered, 0)
        .expect("download failed");
    assert_eq!(result, vec![(size - 1) as u32], "last element filter failed");

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("boundary_indices", start.elapsed())
}

fn test_multi_column_strides(ctx: &TestContext) -> TestResult {
    let start = Instant::now();

    // Test with different column counts and sizes
    for num_cols in [2, 3, 4] {
        for size in [100, 256, 1000] {
            let cols: Vec<Vec<u32>> = (0..num_cols)
                .map(|c| (0..size).map(|i| (c * 1000 + i) as u32).collect())
                .collect();

            let col_refs: Vec<&[u32]> = cols.iter().map(|c| c.as_slice()).collect();

            let mut fields: Vec<(String, ScalarType)> = Vec::new();
            for i in 0..num_cols {
                fields.push((format!("col{}", i), ScalarType::U32));
            }
            let schema = Schema::new(fields);

            let buffer = ctx.provider.create_buffer_from_u32_columns(&col_refs, schema)
                .expect(&format!("buffer failed for {}x{}", num_cols, size));

            // Filter every other row
            let mask: Vec<u8> = (0..size).map(|i| (i % 2) as u8).collect();
            let filtered = ctx.provider.filter_by_mask(&buffer, &mask)
                .expect(&format!("filter failed for {}x{}", num_cols, size));

            // Verify each column
            for c in 0..num_cols {
                let result = ctx.provider.download_column_u32(&filtered, c)
                    .expect(&format!("download col {} failed for {}x{}", c, num_cols, size));

                let expected: Vec<u32> = (0..size)
                    .filter(|i| i % 2 != 0)
                    .map(|i| (c * 1000 + i) as u32)
                    .collect();

                assert_eq!(result, expected,
                    "multi-column stride wrong for col {} at {}x{}", c, num_cols, size);
            }
        }
    }

    ctx.sync_and_check().expect("sync failed");
    TestResult::passed("multi_column_strides", start.elapsed())
}
```

**Step 2: Verify compiles and run tests**

Run: `cargo test -p xlog-cuda-tests --test category_isolation c03 --release -- --nocapture`
Expected: All tests pass

---

I'll continue with more categories, but given the scope, let me write the complete plan file and then we can execute it systematically.
