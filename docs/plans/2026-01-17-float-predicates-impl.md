# Float Predicates Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable correct f32/f64 filter predicates using IEEE 754 total ordering for relational operators.

**Architecture:** Add `float_to_ordered_f64` and `float_to_ordered_f32` helper functions to filter.cu. Update Lt/Le/Gt/Ge branches in 6 filter kernels to transform operands before comparison. Eq/Ne keep IEEE semantics unchanged.

**Tech Stack:** CUDA C++, Rust, xlog-cuda-tests certification harness

**Design Doc:** `docs/plans/2026-01-17-float-predicates-design.md`

---

## Task 1: Add Total Ordering Helpers to filter.cu

**Files:**
- Modify: `kernels/filter.cu:1-27` (add after `#define BLOCK_SIZE 256`)

**Step 1: Add the helper functions**

Insert after line 26 (`#define BLOCK_SIZE 256`):

```cuda
/**
 * Transform f64 to comparable i64 for total ordering.
 *
 * IEEE 754 total order: -NaN < -Inf < ... < -0.0 < +0.0 < ... < +Inf < +NaN
 *
 * Bit manipulation trick:
 * - Negative floats: flip all bits (makes them sort before positives)
 * - Positive floats: flip sign bit only (preserves order)
 */
__device__ __forceinline__ int64_t float_to_ordered_f64(double val) {
    int64_t bits = __double_as_longlong(val);
    int64_t mask = (bits >> 63) | 0x8000000000000000LL;
    return bits ^ mask;
}

/**
 * Transform f32 to comparable i32 for total ordering.
 */
__device__ __forceinline__ int32_t float_to_ordered_f32(float val) {
    int32_t bits = __float_as_int(val);
    int32_t mask = (bits >> 31) | 0x80000000;
    return bits ^ mask;
}
```

**Step 2: Verify build succeeds**

Run: `cd /home/dev/xlog/.worktrees/float-predicates && cargo build -p xlog-cuda --release 2>&1 | tail -5`
Expected: `Finished` with no errors

**Step 3: Commit**

```bash
git add kernels/filter.cu
git commit -m "feat(cuda): add float_to_ordered helpers for total ordering"
```

---

## Task 2: Update filter_compare_f64 Kernel

**Files:**
- Modify: `kernels/filter.cu:85-108` (filter_compare_f64 function)

**Step 1: Update the kernel to use total ordering for Lt/Le/Gt/Ge**

Replace the filter_compare_f64 function (lines 85-108) with:

```cuda
/** Compare f64 column against constant.
 *  Eq/Ne use IEEE semantics. Lt/Le/Gt/Ge use total ordering.
 */
extern "C" __global__ void filter_compare_f64(
    const double* __restrict__ column,
    double constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    double val = column[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (val == constant); break;
        case OP_NE: result = (val != constant); break;
        case OP_LT: result = (float_to_ordered_f64(val) < float_to_ordered_f64(constant)); break;
        case OP_LE: result = (float_to_ordered_f64(val) <= float_to_ordered_f64(constant)); break;
        case OP_GT: result = (float_to_ordered_f64(val) > float_to_ordered_f64(constant)); break;
        case OP_GE: result = (float_to_ordered_f64(val) >= float_to_ordered_f64(constant)); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}
```

**Step 2: Verify build succeeds**

Run: `cargo build -p xlog-cuda --release 2>&1 | tail -3`
Expected: `Finished`

**Step 3: Commit**

```bash
git add kernels/filter.cu
git commit -m "feat(cuda): use total ordering in filter_compare_f64"
```

---

## Task 3: Update filter_compare_f32 Kernel

**Files:**
- Modify: `kernels/filter.cu:160-183` (filter_compare_f32 function)

**Step 1: Update the kernel**

Replace the filter_compare_f32 function with:

```cuda
/** Compare f32 column against constant.
 *  Eq/Ne use IEEE semantics. Lt/Le/Gt/Ge use total ordering.
 */
extern "C" __global__ void filter_compare_f32(
    const float* __restrict__ column,
    float constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    float val = column[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (val == constant); break;
        case OP_NE: result = (val != constant); break;
        case OP_LT: result = (float_to_ordered_f32(val) < float_to_ordered_f32(constant)); break;
        case OP_LE: result = (float_to_ordered_f32(val) <= float_to_ordered_f32(constant)); break;
        case OP_GT: result = (float_to_ordered_f32(val) > float_to_ordered_f32(constant)); break;
        case OP_GE: result = (float_to_ordered_f32(val) >= float_to_ordered_f32(constant)); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}
```

**Step 2: Verify build succeeds**

Run: `cargo build -p xlog-cuda --release 2>&1 | tail -3`
Expected: `Finished`

**Step 3: Commit**

```bash
git add kernels/filter.cu
git commit -m "feat(cuda): use total ordering in filter_compare_f32"
```

---

## Task 4: Update filter_compare_f64_col Kernel

**Files:**
- Modify: `kernels/filter.cu:340-364` (filter_compare_f64_col function)

**Step 1: Update the kernel**

Replace the filter_compare_f64_col function with:

```cuda
/** Compare f64 column against column.
 *  Eq/Ne use IEEE semantics. Lt/Le/Gt/Ge use total ordering.
 */
extern "C" __global__ void filter_compare_f64_col(
    const double* __restrict__ left,
    const double* __restrict__ right,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    double lval = left[gid];
    double rval = right[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (lval == rval); break;
        case OP_NE: result = (lval != rval); break;
        case OP_LT: result = (float_to_ordered_f64(lval) < float_to_ordered_f64(rval)); break;
        case OP_LE: result = (float_to_ordered_f64(lval) <= float_to_ordered_f64(rval)); break;
        case OP_GT: result = (float_to_ordered_f64(lval) > float_to_ordered_f64(rval)); break;
        case OP_GE: result = (float_to_ordered_f64(lval) >= float_to_ordered_f64(rval)); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}
```

**Step 2: Verify build**

Run: `cargo build -p xlog-cuda --release 2>&1 | tail -3`

**Step 3: Commit**

```bash
git add kernels/filter.cu
git commit -m "feat(cuda): use total ordering in filter_compare_f64_col"
```

---

## Task 5: Update filter_compare_f32_col Kernel

**Files:**
- Modify: `kernels/filter.cu:314-338` (filter_compare_f32_col function)

**Step 1: Update the kernel**

Replace the filter_compare_f32_col function with:

```cuda
/** Compare f32 column against column.
 *  Eq/Ne use IEEE semantics. Lt/Le/Gt/Ge use total ordering.
 */
extern "C" __global__ void filter_compare_f32_col(
    const float* __restrict__ left,
    const float* __restrict__ right,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    float lval = left[gid];
    float rval = right[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (lval == rval); break;
        case OP_NE: result = (lval != rval); break;
        case OP_LT: result = (float_to_ordered_f32(lval) < float_to_ordered_f32(rval)); break;
        case OP_LE: result = (float_to_ordered_f32(lval) <= float_to_ordered_f32(rval)); break;
        case OP_GT: result = (float_to_ordered_f32(lval) > float_to_ordered_f32(rval)); break;
        case OP_GE: result = (float_to_ordered_f32(lval) >= float_to_ordered_f32(rval)); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}
```

**Step 2: Verify build**

Run: `cargo build -p xlog-cuda --release 2>&1 | tail -3`

**Step 3: Commit**

```bash
git add kernels/filter.cu
git commit -m "feat(cuda): use total ordering in filter_compare_f32_col"
```

---

## Task 6: Update filter_compare_f64_scan_phase1 Kernel

**Files:**
- Modify: `kernels/filter.cu:458-522` (filter_compare_f64_scan_phase1 function)

**Step 1: Update the fused kernel**

In filter_compare_f64_scan_phase1, replace the switch statement (around lines 483-490) with:

```cuda
        bool result;
        switch (op) {
            case OP_EQ: result = (col_val == constant); break;
            case OP_NE: result = (col_val != constant); break;
            case OP_LT: result = (float_to_ordered_f64(col_val) < float_to_ordered_f64(constant)); break;
            case OP_LE: result = (float_to_ordered_f64(col_val) <= float_to_ordered_f64(constant)); break;
            case OP_GT: result = (float_to_ordered_f64(col_val) > float_to_ordered_f64(constant)); break;
            case OP_GE: result = (float_to_ordered_f64(col_val) >= float_to_ordered_f64(constant)); break;
            default: result = false;
        }
```

**Step 2: Verify build**

Run: `cargo build -p xlog-cuda --release 2>&1 | tail -3`

**Step 3: Commit**

```bash
git add kernels/filter.cu
git commit -m "feat(cuda): use total ordering in filter_compare_f64_scan_phase1"
```

---

## Task 7: Add filter_compare_f32_scan_phase1 Kernel

**Files:**
- Modify: `kernels/filter.cu` (add new kernel after filter_compare_f64_scan_phase1)

**Step 1: Add the new fused f32 kernel**

Add after filter_compare_f64_scan_phase1 (around line 522):

```cuda
/**
 * Fused compare + scan phase1 for f32 filters.
 *
 * Produces:
 * - mask[gid] (0/1)
 * - prefix_sum[gid] (exclusive scan within the block)
 * - block_sums[blockIdx] (number of kept rows in the block)
 */
extern "C" __global__ void filter_compare_f32_scan_phase1(
    const float* __restrict__ column,
    float constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask,
    uint32_t* __restrict__ prefix_sum,
    uint32_t* __restrict__ block_sums
) {
    __shared__ uint32_t temp[BLOCK_SIZE];

    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t val = 0;
    if (gid < num_rows) {
        float col_val = column[gid];
        bool result;
        switch (op) {
            case OP_EQ: result = (col_val == constant); break;
            case OP_NE: result = (col_val != constant); break;
            case OP_LT: result = (float_to_ordered_f32(col_val) < float_to_ordered_f32(constant)); break;
            case OP_LE: result = (float_to_ordered_f32(col_val) <= float_to_ordered_f32(constant)); break;
            case OP_GT: result = (float_to_ordered_f32(col_val) > float_to_ordered_f32(constant)); break;
            case OP_GE: result = (float_to_ordered_f32(col_val) >= float_to_ordered_f32(constant)); break;
            default: result = false;
        }
        uint8_t out = result ? 1 : 0;
        mask[gid] = out;
        val = (uint32_t)out;
    }

    temp[tid] = val;
    __syncthreads();

    // Inclusive scan within block (Hillis-Steele style).
    for (uint32_t stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        uint32_t left_val = 0;
        if (tid >= stride) {
            left_val = temp[tid - stride];
        }
        __syncthreads();
        temp[tid] += left_val;
        __syncthreads();
    }

    uint32_t inclusive = temp[tid];
    uint32_t exclusive = (tid == 0) ? 0 : temp[tid - 1];

    if (gid < num_rows) {
        prefix_sum[gid] = exclusive;
    }

    if (tid == BLOCK_SIZE - 1) {
        block_sums[blockIdx.x] = inclusive;
    }
}
```

**Step 2: Add kernel constant to provider.rs**

In `crates/xlog-cuda/src/provider.rs`, find the kernel constants section (around line 186) and add:

```rust
    pub const FILTER_COMPARE_F32_SCAN_PHASE1: &str = "filter_compare_f32_scan_phase1";
```

**Step 3: Verify build**

Run: `cargo build -p xlog-cuda --release 2>&1 | tail -3`

**Step 4: Commit**

```bash
git add kernels/filter.cu crates/xlog-cuda/src/provider.rs
git commit -m "feat(cuda): add filter_compare_f32_scan_phase1 with total ordering"
```

---

## Task 8: Create c25_float_filter.rs Test Module

**Files:**
- Create: `crates/xlog-cuda-tests/src/categories/c25_float_filter.rs`

**Step 1: Create the test file with structure and helpers**

```rust
//! Category 25: Float filter predicate edge cases
//!
//! Tests filter comparisons on f32/f64 columns with special values:
//! NaN, Inf, -Inf, +0, -0, subnormals, and precision extremes.
//!
//! Verifies IEEE 754 total ordering for Lt/Le/Gt/Ge and IEEE semantics for Eq/Ne.

use crate::harness::{CategoryResult, TestContext, TestResult};
use std::cmp::Ordering;
use std::time::Instant;
use xlog_core::{Schema, ScalarType};

/// Comparison operators matching CUDA kernel definitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompareOp {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}

impl CompareOp {
    pub fn all() -> &'static [CompareOp] {
        &[
            CompareOp::Eq,
            CompareOp::Ne,
            CompareOp::Lt,
            CompareOp::Le,
            CompareOp::Gt,
            CompareOp::Ge,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            CompareOp::Eq => "Eq",
            CompareOp::Ne => "Ne",
            CompareOp::Lt => "Lt",
            CompareOp::Le => "Le",
            CompareOp::Gt => "Gt",
            CompareOp::Ge => "Ge",
        }
    }
}

/// Compute expected result for f64 comparison.
/// Eq/Ne use IEEE semantics, Lt/Le/Gt/Ge use total ordering.
fn expected_f64(left: f64, right: f64, op: CompareOp) -> bool {
    match op {
        CompareOp::Eq => left == right,
        CompareOp::Ne => left != right,
        CompareOp::Lt => left.total_cmp(&right) == Ordering::Less,
        CompareOp::Le => left.total_cmp(&right) != Ordering::Greater,
        CompareOp::Gt => left.total_cmp(&right) == Ordering::Greater,
        CompareOp::Ge => left.total_cmp(&right) != Ordering::Less,
    }
}

/// Compute expected result for f32 comparison.
fn expected_f32(left: f32, right: f32, op: CompareOp) -> bool {
    match op {
        CompareOp::Eq => left == right,
        CompareOp::Ne => left != right,
        CompareOp::Lt => left.total_cmp(&right) == Ordering::Less,
        CompareOp::Le => left.total_cmp(&right) != Ordering::Greater,
        CompareOp::Gt => left.total_cmp(&right) == Ordering::Greater,
        CompareOp::Ge => left.total_cmp(&right) != Ordering::Less,
    }
}

/// Test values for f64 covering all IEEE 754 special cases
fn f64_test_values() -> Vec<f64> {
    vec![
        -f64::NAN,
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

/// Test values for f32 covering all IEEE 754 special cases
fn f32_test_values() -> Vec<f32> {
    vec![
        -f32::NAN,
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

/// Run all tests in this category.
pub fn run_all(ctx: &TestContext) -> CategoryResult {
    let mut results = CategoryResult::new("c25_float_filter");
    let start = Instant::now();

    // Core semantic tests
    results.add_result(test_f64_nan_ordering(ctx));
    results.add_result(test_f64_infinity_ordering(ctx));
    results.add_result(test_f64_zero_distinction(ctx));
    results.add_result(test_f64_equality_ieee(ctx));
    results.add_result(test_f32_nan_ordering(ctx));
    results.add_result(test_f32_infinity_ordering(ctx));

    // Matrix tests
    results.add_result(test_f64_constant_matrix(ctx));
    results.add_result(test_f64_column_matrix(ctx));
    results.add_result(test_f32_constant_matrix(ctx));
    results.add_result(test_f32_column_matrix(ctx));

    results.set_duration(start.elapsed());
    results
}

/// Test 1: Verify NaN > everything in total order (f64)
fn test_f64_nan_ordering(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let test_name = "test_f64_nan_ordering";

    // Test: NaN > INFINITY should be true
    let data = vec![f64::NAN, f64::INFINITY, 1.0, f64::NEG_INFINITY];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    let buffer = match ctx.provider.create_buffer_from_f64_slice(&data, schema) {
        Ok(b) => b,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Buffer creation failed: {}", e)),
    };

    // Filter: val > INFINITY (should return only NaN)
    let filtered = match ctx.provider.filter_f64(&buffer, 0, 4, f64::INFINITY) {
        Ok(f) => f,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Filter failed: {}", e)),
    };

    let result = match ctx.provider.download_column_f64(&filtered, 0) {
        Ok(r) => r,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Download failed: {}", e)),
    };

    // Should contain exactly 1 NaN
    if result.len() != 1 {
        return TestResult::error(
            test_name,
            start.elapsed(),
            format!("Expected 1 result (NaN > Inf), got {}", result.len()),
        );
    }

    if !result[0].is_nan() {
        return TestResult::error(
            test_name,
            start.elapsed(),
            format!("Expected NaN, got {}", result[0]),
        );
    }

    TestResult::passed(test_name, start.elapsed())
}

/// Test 2: Verify -Inf < finite < +Inf < NaN (f64)
fn test_f64_infinity_ordering(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let test_name = "test_f64_infinity_ordering";

    let data = vec![f64::NEG_INFINITY, -1.0, 0.0, 1.0, f64::INFINITY, f64::NAN];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    let buffer = match ctx.provider.create_buffer_from_f64_slice(&data, schema) {
        Ok(b) => b,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Buffer creation failed: {}", e)),
    };

    // Filter: val < INFINITY (should return -Inf, -1, 0, 1 but NOT NaN)
    let filtered = match ctx.provider.filter_f64(&buffer, 0, 2, f64::INFINITY) {
        Ok(f) => f,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Filter failed: {}", e)),
    };

    let result = match ctx.provider.download_column_f64(&filtered, 0) {
        Ok(r) => r,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Download failed: {}", e)),
    };

    // Should contain 4 values (not Inf, not NaN)
    if result.len() != 4 {
        return TestResult::error(
            test_name,
            start.elapsed(),
            format!("Expected 4 results (val < Inf), got {}: {:?}", result.len(), result),
        );
    }

    // Verify no NaN in results
    if result.iter().any(|v| v.is_nan()) {
        return TestResult::error(
            test_name,
            start.elapsed(),
            "NaN should not be < Inf in total ordering".to_string(),
        );
    }

    TestResult::passed(test_name, start.elapsed())
}

/// Test 3: Verify -0.0 < +0.0 in total order (f64)
fn test_f64_zero_distinction(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let test_name = "test_f64_zero_distinction";

    let data = vec![-0.0_f64, 0.0_f64];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    let buffer = match ctx.provider.create_buffer_from_f64_slice(&data, schema) {
        Ok(b) => b,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Buffer creation failed: {}", e)),
    };

    // Filter: val < 0.0 (should return only -0.0)
    let filtered = match ctx.provider.filter_f64(&buffer, 0, 2, 0.0) {
        Ok(f) => f,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Filter failed: {}", e)),
    };

    let result = match ctx.provider.download_column_f64(&filtered, 0) {
        Ok(r) => r,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Download failed: {}", e)),
    };

    // Should contain exactly 1 value (-0.0)
    if result.len() != 1 {
        return TestResult::error(
            test_name,
            start.elapsed(),
            format!("Expected 1 result (-0.0 < +0.0), got {}", result.len()),
        );
    }

    // Verify it's -0.0 by checking bit representation
    if result[0].to_bits() != (-0.0_f64).to_bits() {
        return TestResult::error(
            test_name,
            start.elapsed(),
            format!("Expected -0.0, got {} (bits: {:016x})", result[0], result[0].to_bits()),
        );
    }

    TestResult::passed(test_name, start.elapsed())
}

/// Test 4: Verify NaN == NaN is false (IEEE semantics)
fn test_f64_equality_ieee(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let test_name = "test_f64_equality_ieee";

    let data = vec![f64::NAN, 1.0, f64::NAN];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);

    let buffer = match ctx.provider.create_buffer_from_f64_slice(&data, schema) {
        Ok(b) => b,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Buffer creation failed: {}", e)),
    };

    // Filter: val == NaN (should return nothing - IEEE semantics)
    let filtered = match ctx.provider.filter_f64(&buffer, 0, 0, f64::NAN) {
        Ok(f) => f,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Filter failed: {}", e)),
    };

    let result = match ctx.provider.download_column_f64(&filtered, 0) {
        Ok(r) => r,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Download failed: {}", e)),
    };

    // Should be empty (NaN != NaN in IEEE)
    if !result.is_empty() {
        return TestResult::error(
            test_name,
            start.elapsed(),
            format!("Expected 0 results (NaN == NaN is false), got {}", result.len()),
        );
    }

    TestResult::passed(test_name, start.elapsed())
}

/// Test 5: Verify NaN ordering for f32
fn test_f32_nan_ordering(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let test_name = "test_f32_nan_ordering";

    let data = vec![f32::NAN, f32::INFINITY, 1.0_f32, f32::NEG_INFINITY];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F32)]);

    let buffer = match ctx.provider.create_buffer_from_f32_slice(&data, schema) {
        Ok(b) => b,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Buffer creation failed: {}", e)),
    };

    // Filter: val > INFINITY (should return only NaN)
    let filtered = match ctx.provider.filter_f32(&buffer, 0, 4, f32::INFINITY) {
        Ok(f) => f,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Filter failed: {}", e)),
    };

    let result = match ctx.provider.download_column_f32(&filtered, 0) {
        Ok(r) => r,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Download failed: {}", e)),
    };

    if result.len() != 1 || !result[0].is_nan() {
        return TestResult::error(
            test_name,
            start.elapsed(),
            format!("Expected 1 NaN result, got {:?}", result),
        );
    }

    TestResult::passed(test_name, start.elapsed())
}

/// Test 6: Verify infinity ordering for f32
fn test_f32_infinity_ordering(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let test_name = "test_f32_infinity_ordering";

    let data = vec![f32::NEG_INFINITY, 0.0_f32, f32::INFINITY, f32::NAN];
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F32)]);

    let buffer = match ctx.provider.create_buffer_from_f32_slice(&data, schema) {
        Ok(b) => b,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Buffer creation failed: {}", e)),
    };

    // Filter: val < INFINITY (should return -Inf, 0 but NOT NaN)
    let filtered = match ctx.provider.filter_f32(&buffer, 0, 2, f32::INFINITY) {
        Ok(f) => f,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Filter failed: {}", e)),
    };

    let result = match ctx.provider.download_column_f32(&filtered, 0) {
        Ok(r) => r,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Download failed: {}", e)),
    };

    if result.len() != 2 {
        return TestResult::error(
            test_name,
            start.elapsed(),
            format!("Expected 2 results, got {}: {:?}", result.len(), result),
        );
    }

    TestResult::passed(test_name, start.elapsed())
}

/// Test 7: Full f64 constant comparison matrix
fn test_f64_constant_matrix(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let test_name = "test_f64_constant_matrix";

    let values = f64_test_values();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F64)]);
    let mut failures = Vec::new();

    for &constant in &values {
        let buffer = match ctx.provider.create_buffer_from_f64_slice(&values, schema.clone()) {
            Ok(b) => b,
            Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Buffer creation failed: {}", e)),
        };

        for &op in CompareOp::all() {
            let filtered = match ctx.provider.filter_f64(&buffer, 0, op as u8, constant) {
                Ok(f) => f,
                Err(e) => {
                    failures.push(format!("Filter failed for {:?} {} {:?}: {}", values, op.name(), constant, e));
                    continue;
                }
            };

            let result = match ctx.provider.download_column_f64(&filtered, 0) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!("Download failed: {}", e));
                    continue;
                }
            };

            // Compute expected count
            let expected_count = values.iter().filter(|&&v| expected_f64(v, constant, op)).count();

            if result.len() != expected_count {
                failures.push(format!(
                    "val {} {:?}: expected {} results, got {}",
                    op.name(), constant, expected_count, result.len()
                ));
            }
        }
    }

    if failures.is_empty() {
        TestResult::passed(test_name, start.elapsed())
    } else {
        TestResult::error(
            test_name,
            start.elapsed(),
            format!("{} failures:\n{}", failures.len(), failures.join("\n")),
        )
    }
}

/// Test 8: Full f64 column-column comparison matrix
fn test_f64_column_matrix(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let test_name = "test_f64_column_matrix";

    // Test a subset to keep test time reasonable
    let values = vec![f64::NAN, f64::INFINITY, 0.0, -0.0, 1.0, f64::NEG_INFINITY];
    let n = values.len();
    let mut failures = Vec::new();

    // Create left column (repeat each value n times)
    let left: Vec<f64> = values.iter().flat_map(|&v| std::iter::repeat(v).take(n)).collect();
    // Create right column (cycle through values)
    let right: Vec<f64> = (0..n).flat_map(|_| values.iter().copied()).collect();

    let schema = Schema::new(vec![
        ("left".to_string(), ScalarType::F64),
        ("right".to_string(), ScalarType::F64),
    ]);

    let buffer = match ctx.provider.create_two_column_f64_buffer(&left, &right, schema) {
        Ok(b) => b,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Buffer creation failed: {}", e)),
    };

    for &op in CompareOp::all() {
        let filtered = match ctx.provider.filter_f64_col(&buffer, 0, 1, op as u8) {
            Ok(f) => f,
            Err(e) => {
                failures.push(format!("Filter column failed for {}: {}", op.name(), e));
                continue;
            }
        };

        let result_left = match ctx.provider.download_column_f64(&filtered, 0) {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("Download failed: {}", e));
                continue;
            }
        };

        // Compute expected count
        let expected_count = left.iter().zip(right.iter())
            .filter(|(&l, &r)| expected_f64(l, r, op))
            .count();

        if result_left.len() != expected_count {
            failures.push(format!(
                "col {} col: expected {} results, got {}",
                op.name(), expected_count, result_left.len()
            ));
        }
    }

    if failures.is_empty() {
        TestResult::passed(test_name, start.elapsed())
    } else {
        TestResult::error(
            test_name,
            start.elapsed(),
            format!("{} failures:\n{}", failures.len(), failures.join("\n")),
        )
    }
}

/// Test 9: Full f32 constant comparison matrix
fn test_f32_constant_matrix(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let test_name = "test_f32_constant_matrix";

    let values = f32_test_values();
    let schema = Schema::new(vec![("val".to_string(), ScalarType::F32)]);
    let mut failures = Vec::new();

    for &constant in &values {
        let buffer = match ctx.provider.create_buffer_from_f32_slice(&values, schema.clone()) {
            Ok(b) => b,
            Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Buffer creation failed: {}", e)),
        };

        for &op in CompareOp::all() {
            let filtered = match ctx.provider.filter_f32(&buffer, 0, op as u8, constant) {
                Ok(f) => f,
                Err(e) => {
                    failures.push(format!("Filter failed for op {}: {}", op.name(), e));
                    continue;
                }
            };

            let result = match ctx.provider.download_column_f32(&filtered, 0) {
                Ok(r) => r,
                Err(e) => {
                    failures.push(format!("Download failed: {}", e));
                    continue;
                }
            };

            let expected_count = values.iter().filter(|&&v| expected_f32(v, constant, op)).count();

            if result.len() != expected_count {
                failures.push(format!(
                    "val {} {:?}: expected {} results, got {}",
                    op.name(), constant, expected_count, result.len()
                ));
            }
        }
    }

    if failures.is_empty() {
        TestResult::passed(test_name, start.elapsed())
    } else {
        TestResult::error(
            test_name,
            start.elapsed(),
            format!("{} failures:\n{}", failures.len(), failures.join("\n")),
        )
    }
}

/// Test 10: Full f32 column-column comparison matrix
fn test_f32_column_matrix(ctx: &TestContext) -> TestResult {
    let start = Instant::now();
    let test_name = "test_f32_column_matrix";

    let values = vec![f32::NAN, f32::INFINITY, 0.0_f32, -0.0_f32, 1.0_f32, f32::NEG_INFINITY];
    let n = values.len();
    let mut failures = Vec::new();

    let left: Vec<f32> = values.iter().flat_map(|&v| std::iter::repeat(v).take(n)).collect();
    let right: Vec<f32> = (0..n).flat_map(|_| values.iter().copied()).collect();

    let schema = Schema::new(vec![
        ("left".to_string(), ScalarType::F32),
        ("right".to_string(), ScalarType::F32),
    ]);

    let buffer = match ctx.provider.create_two_column_f32_buffer(&left, &right, schema) {
        Ok(b) => b,
        Err(e) => return TestResult::error(test_name, start.elapsed(), format!("Buffer creation failed: {}", e)),
    };

    for &op in CompareOp::all() {
        let filtered = match ctx.provider.filter_f32_col(&buffer, 0, 1, op as u8) {
            Ok(f) => f,
            Err(e) => {
                failures.push(format!("Filter column failed for {}: {}", op.name(), e));
                continue;
            }
        };

        let result_left = match ctx.provider.download_column_f32(&filtered, 0) {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("Download failed: {}", e));
                continue;
            }
        };

        let expected_count = left.iter().zip(right.iter())
            .filter(|(&l, &r)| expected_f32(l, r, op))
            .count();

        if result_left.len() != expected_count {
            failures.push(format!(
                "col {} col: expected {} results, got {}",
                op.name(), expected_count, result_left.len()
            ));
        }
    }

    if failures.is_empty() {
        TestResult::passed(test_name, start.elapsed())
    } else {
        TestResult::error(
            test_name,
            start.elapsed(),
            format!("{} failures:\n{}", failures.len(), failures.join("\n")),
        )
    }
}
```

**Step 2: Verify file created**

Run: `ls -la crates/xlog-cuda-tests/src/categories/c25_float_filter.rs`

**Step 3: Commit (don't run tests yet - provider methods may need adding)**

```bash
git add crates/xlog-cuda-tests/src/categories/c25_float_filter.rs
git commit -m "test(cuda): add c25_float_filter certification tests (WIP)"
```

---

## Task 9: Register c25 Module

**Files:**
- Modify: `crates/xlog-cuda-tests/src/categories/mod.rs`
- Modify: `crates/xlog-cuda-tests/tests/certification_suite.rs`

**Step 1: Add module to mod.rs**

Add to end of `crates/xlog-cuda-tests/src/categories/mod.rs`:

```rust
pub mod c25_float_filter;
```

**Step 2: Add to certification_suite.rs**

In `crates/xlog-cuda-tests/tests/certification_suite.rs`, add after line 112 (after c24):

```rust
    println!("Running C25: Float Filter...");
    results.add_category(categories::c25_float_filter::run_all(&ctx));
```

**Step 3: Commit**

```bash
git add crates/xlog-cuda-tests/src/categories/mod.rs crates/xlog-cuda-tests/tests/certification_suite.rs
git commit -m "test(cuda): register c25_float_filter in certification suite"
```

---

## Task 10: Add Missing Provider Methods

**Files:**
- Modify: `crates/xlog-cuda/src/provider.rs`
- Modify: `crates/xlog-cuda-tests/src/harness/provider.rs`

**Step 1: Check what methods are missing and add them**

The tests need these methods on the provider:
- `create_buffer_from_f32_slice`
- `create_buffer_from_f64_slice` (likely exists)
- `create_two_column_f64_buffer`
- `create_two_column_f32_buffer`
- `filter_f32`
- `filter_f64` (likely exists)
- `filter_f32_col`
- `filter_f64_col`
- `download_column_f32`
- `download_column_f64` (likely exists)

Check existing methods and add any missing ones. This task may require exploring the provider.rs file.

**Step 2: Build and fix any compile errors**

Run: `cargo build -p xlog-cuda-tests --release 2>&1 | head -50`

Fix any missing method errors by adding stubs or implementations.

**Step 3: Commit**

```bash
git add crates/xlog-cuda/src/provider.rs crates/xlog-cuda-tests/src/harness/provider.rs
git commit -m "feat(cuda): add provider methods for f32/f64 filter tests"
```

---

## Task 11: Run Tests and Fix Failures

**Files:**
- May need to fix: `kernels/filter.cu`, test file, or provider

**Step 1: Run the certification suite**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture 2>&1 | tail -100`

**Step 2: If tests fail, diagnose and fix**

Common issues:
- Kernel not found: check kernel name in provider constants
- Wrong results: verify the bit manipulation is correct
- Compile errors: check CUDA syntax

**Step 3: Iterate until all tests pass**

**Step 4: Commit fixes**

```bash
git add -A
git commit -m "fix(cuda): resolve float filter test failures"
```

---

## Task 12: Update Documentation

**Files:**
- Modify: `docs/architecture/gpu-execution.md`

**Step 1: Add float comparison semantics section**

Add to `docs/architecture/gpu-execution.md` in the Filter Evaluation section:

```markdown
### Float Comparison Semantics

Filter predicates on `f32`/`f64` columns use mixed semantics for correctness with special values:

| Operator | Semantics | Example |
|----------|-----------|---------|
| `Eq`, `Ne` | IEEE 754 | `NaN == NaN` → false |
| `Lt`, `Le`, `Gt`, `Ge` | Total Order | `NaN > Inf` → true |

**Total ordering (smallest to largest):**
```
-NaN < -Inf < -MAX < ... < -0.0 < +0.0 < ... < +MAX < +Inf < +NaN
```

This matches Rust's `f64::total_cmp` and ensures consistent, deterministic behavior when filtering data containing special float values.

**Implementation:** Relational operators transform float bits to comparable integers using a sign-flip trick, enabling branchless comparison that respects total ordering.
```

**Step 2: Commit**

```bash
git add docs/architecture/gpu-execution.md
git commit -m "docs: document float comparison semantics"
```

---

## Task 13: Final Verification

**Step 1: Run full certification suite**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release`

Expected: All tests pass including new c25 category

**Step 2: Run existing c13 tests to ensure no regression**

Run: `cargo test -p xlog-cuda-tests --test certification_suite --release -- --nocapture 2>&1 | grep -A5 "C13"`

Expected: All c13_floating_point tests still pass

**Step 3: Verify test count increased**

Expected: 140 → 150 tests (10 new tests in c25)

**Step 4: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "test(cuda): finalize float predicate certification"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add total ordering helpers | `kernels/filter.cu` |
| 2 | Update filter_compare_f64 | `kernels/filter.cu` |
| 3 | Update filter_compare_f32 | `kernels/filter.cu` |
| 4 | Update filter_compare_f64_col | `kernels/filter.cu` |
| 5 | Update filter_compare_f32_col | `kernels/filter.cu` |
| 6 | Update filter_compare_f64_scan_phase1 | `kernels/filter.cu` |
| 7 | Add filter_compare_f32_scan_phase1 | `kernels/filter.cu`, `provider.rs` |
| 8 | Create c25_float_filter.rs | `categories/c25_float_filter.rs` |
| 9 | Register c25 module | `mod.rs`, `certification_suite.rs` |
| 10 | Add missing provider methods | `provider.rs` |
| 11 | Run tests and fix failures | Various |
| 12 | Update documentation | `gpu-execution.md` |
| 13 | Final verification | N/A |

**Acceptance Criteria:**
- [ ] All certification tests pass (150+ tests)
- [ ] c13_floating_point tests unchanged
- [ ] `NaN > Inf` returns true
- [ ] `NaN == NaN` returns false
- [ ] `-0.0 < +0.0` returns true
