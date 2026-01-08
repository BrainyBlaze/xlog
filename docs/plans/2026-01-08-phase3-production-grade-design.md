# Phase 3: Production-Grade GPU Execution Engine

**Date:** 2026-01-08
**Status:** Approved
**Scope:** Full production-grade implementation resolving all MVP limitations

---

## Summary

Phase 3 transforms the MVP GPU Datalog engine into a production-grade system by implementing:

- GPU-native sorting (radix sort)
- Multi-column joins with full type support
- Proper Semi/Anti/LeftOuter join implementations
- GPU-based filtering (replacing CPU fallback)
- Complete aggregation support (multi-agg, LogSumExp)
- Device-to-device operations (eliminating host roundtrips)
- GPU prefix sum for stream compaction
- Type-generic operations for all scalar types

---

## 1. GPU Sorting Foundation

GPU sorting is the foundation that unblocks correct dedup/groupby. Without it, other fixes are incomplete.

### Kernel Design

```cuda
// kernels/sort.cu

// Radix sort histogram - count occurrences per radix digit
extern "C" __global__ void radix_sort_histogram(
    const uint32_t* keys,
    uint32_t num_rows,
    uint32_t* histograms,
    uint32_t shift
);

// Radix sort scatter - reorder by prefix sums
extern "C" __global__ void radix_sort_scatter(
    const uint32_t* keys_in,
    const uint32_t* indices_in,
    uint32_t* keys_out,
    uint32_t* indices_out,
    const uint32_t* prefix_sums,
    uint32_t num_rows,
    uint32_t shift
);

// Apply permutation to reorder columns
extern "C" __global__ void apply_permutation(
    const uint8_t* input,
    uint8_t* output,
    const uint32_t* permutation,
    uint32_t num_rows,
    uint32_t row_bytes
);
```

### Rust API

```rust
impl CudaKernelProvider {
    /// Sort buffer by key columns, returning sorted buffer
    pub fn sort(&self, input: &CudaBuffer, key_cols: &[usize]) -> Result<CudaBuffer>;

    /// Sort in-place (more efficient when input can be consumed)
    pub fn sort_inplace(&mut self, input: &mut CudaBuffer, key_cols: &[usize]) -> Result<()>;
}
```

### Integration

- `dedup()` calls `sort()` before detecting unique rows
- `groupby_agg()` calls `sort()` before detecting group boundaries

---

## 2. Multi-Column Joins with Full Type Support

### Problem

Current joins only support single U32 column. Production needs multi-column keys of any type.

### Kernel Design

```cuda
// kernels/join.cu - Enhanced

// Hash multiple columns into single u64 hash value
extern "C" __global__ void compute_composite_hash(
    const uint8_t* columns,      // Packed column data
    const uint32_t* col_offsets, // Offset to each column
    const uint32_t* col_sizes,   // Size of each key column in bytes
    uint32_t num_key_cols,
    uint32_t num_rows,
    uint64_t* hashes             // Output: one hash per row
);

// Build phase with u64 hashes
extern "C" __global__ void hash_join_build_v2(
    const uint64_t* hashes,
    uint32_t num_rows,
    uint64_t* hash_table,        // Stores hash + row_id pairs
    uint32_t* next_ptrs,
    uint32_t hash_table_size
);

// Probe phase with collision verification
extern "C" __global__ void hash_join_probe_v2(
    const uint64_t* probe_hashes,
    uint32_t num_probe,
    const uint64_t* hash_table,
    const uint64_t* build_hashes,
    const uint32_t* next_ptrs,
    uint32_t hash_table_size,
    uint32_t* output_left_idx,
    uint32_t* output_right_idx,
    uint32_t* output_count,
    uint32_t max_output          // Prevent buffer overflow
);
```

### Rust API

```rust
pub fn hash_join(
    &self,
    left: &CudaBuffer,
    right: &CudaBuffer,
    left_keys: &[usize],     // Multiple columns supported
    right_keys: &[usize],    // Any ScalarType supported
) -> Result<CudaBuffer>;
```

### Key Improvements

- Multi-column keys via composite hashing
- All scalar types (hash bytes directly)
- `max_output` parameter prevents buffer overflow
- Collision verification with actual key comparison

---

## 3. Proper Join Types (Semi, Anti, LeftOuter)

### Problem

Semi/Anti/LeftOuter joins currently fall back to incorrect inner join behavior.

### Kernel Design

```cuda
// kernels/join.cu - Additional kernels

// Semi-join: Return left rows that have ANY match (no duplicates)
extern "C" __global__ void hash_join_semi(
    const uint64_t* probe_hashes,
    uint32_t num_probe,
    const uint64_t* hash_table,
    const uint32_t* next_ptrs,
    uint32_t hash_table_size,
    uint8_t* has_match           // Output: 1 if row has match
);

// Anti-join: Return left rows that have NO match
extern "C" __global__ void hash_join_anti(
    const uint64_t* probe_hashes,
    uint32_t num_probe,
    const uint64_t* hash_table,
    const uint32_t* next_ptrs,
    uint32_t hash_table_size,
    uint8_t* no_match            // Output: 1 if no match
);

// Left outer: Mark which build rows matched
extern "C" __global__ void mark_matched_build_rows(
    const uint32_t* output_right_idx,
    uint32_t output_count,
    uint8_t* build_matched,
    uint32_t num_build
);
```

### Rust Implementation

```rust
fn execute_join(&self, join_type: JoinType, ...) -> Result<CudaBuffer> {
    match join_type {
        JoinType::Inner => self.hash_join_inner(...),
        JoinType::Semi => {
            let has_match = self.probe_existence(left, right, ...)?;
            self.filter_by_mask(left, &has_match)
        }
        JoinType::Anti => {
            let no_match = self.probe_non_existence(left, right, ...)?;
            self.filter_by_mask(left, &no_match)
        }
        JoinType::LeftOuter => {
            self.hash_join_left_outer(...)
        }
    }
}
```

---

## 4. GPU-Based Filtering

### Problem

Filter currently copies to host, evaluates row-by-row in Rust, copies back.

### Kernel Design

```cuda
// kernels/filter.cu

// Comparison predicates
extern "C" __global__ void filter_compare_i64(
    const int64_t* column,
    int64_t constant,
    uint32_t num_rows,
    uint8_t op,              // 0=Eq, 1=Ne, 2=Lt, 3=Le, 4=Gt, 5=Ge
    uint8_t* result_mask
);

extern "C" __global__ void filter_compare_f64(
    const double* column,
    double constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* result_mask
);

extern "C" __global__ void filter_compare_columns(
    const uint8_t* col_left,
    const uint8_t* col_right,
    uint32_t elem_size,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* result_mask
);

// Mask combinators
extern "C" __global__ void mask_and(
    const uint8_t* a, const uint8_t* b, uint8_t* out, uint32_t n
);
extern "C" __global__ void mask_or(
    const uint8_t* a, const uint8_t* b, uint8_t* out, uint32_t n
);
extern "C" __global__ void mask_not(
    const uint8_t* a, uint8_t* out, uint32_t n
);

// Stream compaction
extern "C" __global__ void compact_by_mask(
    const uint8_t* input_rows,
    const uint8_t* mask,
    const uint32_t* prefix_sum,
    uint32_t num_rows,
    uint32_t row_bytes,
    uint8_t* output_rows
);
```

### Rust API

```rust
impl CudaKernelProvider {
    /// Filter buffer by predicate expression (fully on GPU)
    pub fn filter(&self, input: &CudaBuffer, predicate: &Expr) -> Result<CudaBuffer>;

    /// Evaluate predicate to mask (for reuse)
    pub fn evaluate_predicate(&self, input: &CudaBuffer, pred: &Expr) -> Result<CudaSlice<u8>>;

    /// Filter by pre-computed mask
    pub fn filter_by_mask(&self, input: &CudaBuffer, mask: &CudaSlice<u8>) -> Result<CudaBuffer>;
}
```

---

## 5. Complete Aggregation Support

### Problem

Only first aggregation executed, LogSumExp missing, assumes sorted input.

### Kernel Design

```cuda
// kernels/groupby.cu - Enhanced

// Multi-aggregation in one pass
extern "C" __global__ void groupby_multi_agg(
    const uint8_t* values,
    const uint32_t* value_offsets,
    const uint32_t* group_ids,
    const uint8_t* agg_ops,      // count/sum/min/max per column
    uint32_t num_value_cols,
    uint32_t num_rows,
    uint32_t num_groups,
    uint8_t* output
);

// LogSumExp for numerical stability
extern "C" __global__ void groupby_logsumexp(
    const double* values,
    const uint32_t* group_ids,
    uint32_t num_rows,
    uint32_t num_groups,
    double* output
);

// Average (SUM / COUNT)
extern "C" __global__ void groupby_avg(
    const double* values,
    const uint32_t* group_ids,
    uint32_t num_rows,
    uint32_t num_groups,
    double* output
);
```

### Rust Implementation

```rust
pub fn groupby_agg(
    &self,
    input: &CudaBuffer,
    key_cols: &[usize],
    aggs: &[(usize, AggOp)],  // Handles ALL aggregations
) -> Result<CudaBuffer> {
    // 1. Sort by key columns
    let sorted = self.sort(input, key_cols)?;

    // 2. Detect group boundaries
    let boundaries = self.detect_boundaries(&sorted, key_cols)?;

    // 3. Compute group IDs from boundaries
    let group_ids = self.boundaries_to_group_ids(&boundaries)?;

    // 4. Execute ALL aggregations in parallel
    let agg_results = self.execute_aggregations(&sorted, &group_ids, aggs)?;

    // 5. Combine key columns + aggregation results
    self.build_groupby_result(&sorted, key_cols, &agg_results)
}
```

### Supported AggOps

- Count
- Sum
- Min
- Max
- Avg
- LogSumExp

---

## 6. Device-to-Device Operations

### Problem

`union` and `diff` copy data to host, perform operations, copy back.

### Kernel Design

```cuda
// kernels/set_ops.cu

extern "C" __global__ void gpu_union(
    const uint32_t* a, uint32_t a_rows,
    const uint32_t* b, uint32_t b_rows,
    uint32_t cols,
    uint32_t* output,
    uint32_t* output_count
);

extern "C" __global__ void gpu_diff(
    const uint32_t* a, uint32_t a_rows,
    const uint32_t* b, uint32_t b_rows,
    uint32_t cols,
    uint32_t* output,
    uint32_t* output_count
);
```

### Rust Implementation

```rust
impl CudaKernelProvider {
    pub fn union(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        // 1. Allocate output = a.num_rows + b.num_rows
        // 2. D2D copy a to output[0..a.num_rows]
        // 3. D2D copy b to output[a.num_rows..]
        // 4. Call dedup() on result
    }

    pub fn diff(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        // 1. Sort both inputs
        // 2. Launch gpu_diff kernel
        // 3. Stream compact results
    }
}
```

### Expected Improvement

10-100x speedup for large relations (eliminates PCIe bottleneck).

---

## 7. GPU Prefix Sum

### Purpose

Stream compaction (filtering, semi/anti joins) requires prefix sum. GPU implementation avoids host roundtrips.

### Kernel Design

```cuda
// kernels/scan.cu

// Block-level inclusive scan
extern "C" __global__ void block_scan(
    const uint32_t* input,
    uint32_t* output,
    uint32_t* block_sums,
    uint32_t n
);

// Add block offsets for global prefix sum
extern "C" __global__ void add_block_offsets(
    uint32_t* data,
    const uint32_t* block_offsets,
    uint32_t n
);

// Exclusive prefix sum for stream compaction
extern "C" __global__ void exclusive_scan(
    const uint8_t* mask,
    uint32_t* prefix_sum,
    uint32_t* total_count,
    uint32_t n
);
```

### Rust API

```rust
impl CudaKernelProvider {
    /// Compute exclusive prefix sum of u32 array
    pub fn prefix_sum(&self, input: &CudaSlice<u32>) -> Result<CudaSlice<u32>>;

    /// Compute prefix sum of u8 mask, returns (prefix_sum, total_count)
    pub fn prefix_sum_mask(&self, mask: &CudaSlice<u8>) -> Result<(CudaSlice<u32>, u32)>;
}
```

### Used By

- `filter_by_mask` - compact rows matching predicate
- `hash_join_semi/anti` - compact rows with/without matches
- `radix_sort_scatter` - reorder elements by histogram

---

## 8. Type-Generic Operations

### Problem

Most operations hardcoded for U32. Production needs all scalar types.

### Kernel Design

```cuda
// kernels/types.cu

#define INSTANTIATE_COMPARE(TYPE, SUFFIX) \
extern "C" __global__ void filter_compare_##SUFFIX( \
    const TYPE* column, TYPE constant, uint32_t n, \
    uint8_t op, uint8_t* mask);

INSTANTIATE_COMPARE(int32_t, i32)
INSTANTIATE_COMPARE(int64_t, i64)
INSTANTIATE_COMPARE(uint32_t, u32)
INSTANTIATE_COMPARE(uint64_t, u64)
INSTANTIATE_COMPARE(float, f32)
INSTANTIATE_COMPARE(double, f64)

// Type-aware hashing
__device__ uint64_t hash_value(const uint8_t* ptr, uint8_t type_tag, uint32_t size);
```

### Rust Dispatch

```rust
impl CudaKernelProvider {
    fn dispatch_comparison(&self, col: &CudaSlice<u8>, scalar_type: ScalarType,
                           constant: Value, op: CompareOp) -> Result<CudaSlice<u8>> {
        match scalar_type {
            ScalarType::I32 => self.filter_compare_i32(...),
            ScalarType::I64 => self.filter_compare_i64(...),
            ScalarType::F32 => self.filter_compare_f32(...),
            ScalarType::F64 => self.filter_compare_f64(...),
            ScalarType::U32 => self.filter_compare_u32(...),
            ScalarType::U64 => self.filter_compare_u64(...),
            ScalarType::Bool => self.filter_compare_bool(...),
        }
    }
}
```

---

## 9. Implementation Order

### Dependency Graph

```
scan.cu (prefix sum)
    ↓
sort.cu (radix sort) ←── uses prefix sum for scatter
    ↓
filter.cu (GPU filtering) ←── uses prefix sum for compaction
    ↓
join.cu v2 (multi-column) ←── uses composite hashing
    ↓
set_ops.cu (union/diff) ←── uses sort + dedup
    ↓
groupby.cu v2 (multi-agg) ←── uses sort + boundaries
    ↓
types.cu (type dispatch) ←── wraps all kernels
```

### Task List (13 tasks)

1. **scan.cu** - GPU prefix sum kernel
2. **sort.cu** - Radix sort with permutation
3. **CudaKernelProvider::sort()** - Rust wrapper for sort
4. **filter.cu** - Comparison + mask kernels
5. **CudaKernelProvider::filter()** - GPU filtering with compaction
6. **join.cu v2** - Composite hash + join types (Semi/Anti/LeftOuter)
7. **CudaKernelProvider::hash_join() v2** - Multi-column, all types
8. **set_ops.cu** - GPU union/diff kernels
9. **CudaKernelProvider::union/diff()** - D2D implementations
10. **groupby.cu v2** - Multi-agg + LogSumExp
11. **CudaKernelProvider::groupby_agg() v2** - Full aggregation
12. **types.cu** - Type-dispatched kernels
13. **Integration tests** - Enable ignored tests, add coverage

---

## 10. Success Criteria

Phase 3 is complete when:

1. All 3 currently-ignored E2E tests pass
2. New tests pass for:
   - Multi-column joins (2+ key columns)
   - All scalar types (I32, I64, U32, U64, F32, F64, Bool)
   - All join types (Inner, Semi, Anti, LeftOuter)
   - All aggregations (Count, Sum, Min, Max, Avg, LogSumExp)
   - GPU filtering with float comparisons
3. No host roundtrips in hot paths (union, diff, filter)
4. Performance benchmarks show expected speedups

---

## 11. Files Modified/Created

### New CUDA Kernels
- `kernels/scan.cu` - Prefix sum
- `kernels/sort.cu` - Radix sort
- `kernels/filter.cu` - Comparison and compaction
- `kernels/set_ops.cu` - Union and diff
- `kernels/types.cu` - Type instantiations

### Modified CUDA Kernels
- `kernels/join.cu` - v2 with composite hashing and join types
- `kernels/groupby.cu` - v2 with multi-agg and LogSumExp

### Rust Modifications
- `crates/xlog-cuda/src/provider.rs` - New kernel wrappers
- `crates/xlog-runtime/src/executor.rs` - Updated to use new APIs

### New Tests
- `crates/xlog-cuda/tests/sort_tests.rs`
- `crates/xlog-cuda/tests/filter_tests.rs`
- `crates/xlog-cuda/tests/join_types_tests.rs`
- `crates/xlog-cuda/tests/aggregation_tests.rs`
