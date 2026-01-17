# Float Predicate Support Design

> **Status:** Approved
> **Target:** v0.3.1
> **Author:** Claude
> **Date:** 2026-01-17

---

## Summary

Enable correct `f32`/`f64` comparisons in filter predicates using IEEE 754 total ordering for relational operators while preserving IEEE equality semantics.

---

## Problem

Current filter kernels use standard IEEE comparison where:
- `NaN < x` returns `false` for all `x`
- `NaN == NaN` returns `false`
- `NaN > x` returns `false` for all `x`

This breaks filter predicates on float columns containing NaN or Inf values. Users cannot reliably filter data with special float values.

---

## Solution

### Semantics

| Operator | Semantics | `NaN == NaN` | `NaN > Inf` |
|----------|-----------|--------------|-------------|
| Eq, Ne | IEEE 754 | `false` | N/A |
| Lt, Le, Gt, Ge | Total Order | N/A | `true` |

**Total ordering:**
```
-NaN < -Inf < -MAX < ... < -MIN_POSITIVE < -0.0 < +0.0 < ... < +MAX < +Inf < +NaN
```

This matches Rust's `f64::total_cmp` and SQL NULL ordering conventions.

### Kernel Implementation

Use bit-manipulation for branchless total ordering:

```cuda
// Transform f64 to comparable i64
__device__ __forceinline__ int64_t float_to_ordered_f64(double val) {
    int64_t bits = __double_as_longlong(val);
    // Negative: flip all bits. Positive: flip sign bit only.
    int64_t mask = (bits >> 63) | 0x8000000000000000LL;
    return bits ^ mask;
}

// Transform f32 to comparable i32
__device__ __forceinline__ int32_t float_to_ordered_f32(float val) {
    int32_t bits = __float_as_int(val);
    int32_t mask = (bits >> 31) | 0x80000000;
    return bits ^ mask;
}
```

**Kernel changes:**
- Eq/Ne: No change (keep IEEE semantics)
- Lt/Le/Gt/Ge: Transform both operands, compare as integers

---

## Kernels to Update

| Kernel | Type | Mode |
|--------|------|------|
| `filter_compare_f32` | f32 | Constant |
| `filter_compare_f64` | f64 | Constant |
| `filter_compare_f32_col` | f32 | Column-column |
| `filter_compare_f64_col` | f64 | Column-column |
| `filter_compare_f32_scan_phase1` | f32 | Fused (new) |
| `filter_compare_f64_scan_phase1` | f64 | Fused (existing) |

---

## Test Matrix

**Test values (12 categories):**
```
-NaN, -Inf, -MAX, -1.0, -MIN_POSITIVE, -0.0, +0.0, +MIN_POSITIVE, +1.0, +MAX, +Inf, +NaN
```

**Matrix dimensions:**
- Operators: 6 (Eq, Ne, Lt, Le, Gt, Ge)
- Left operand: 12 special values
- Right operand: 12 special values
- Modes: 2 (Constant, Column-column)

**Total: 1,728 test cases**

**Expected results computed by:**
```rust
fn expected_result(left: f64, right: f64, op: Op) -> bool {
    match op {
        Eq => left == right,  // IEEE
        Ne => left != right,  // IEEE
        Lt => left.total_cmp(&right) == Ordering::Less,
        Le => left.total_cmp(&right) != Ordering::Greater,
        Gt => left.total_cmp(&right) == Ordering::Greater,
        Ge => left.total_cmp(&right) != Ordering::Less,
    }
}
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `kernels/filter.cu` | Add helpers, update 6 kernels, add f32 fused variant |
| `crates/xlog-cuda-tests/src/categories/c25_float_filter.rs` | New certification tests |
| `crates/xlog-cuda-tests/src/categories/mod.rs` | Register c25 module |
| `crates/xlog-cuda-tests/tests/certification_suite.rs` | Include c25 category |
| `docs/architecture/gpu-execution.md` | Document semantics |

---

## Implementation Order

1. Add `float_to_ordered_f64` and `float_to_ordered_f32` helpers to `filter.cu`
2. Update constant-comparison kernels (f32, f64)
3. Update column-column kernels (f32, f64)
4. Add fused scan variant for f32
5. Update existing fused f64 scan kernel
6. Create `c25_float_filter.rs` certification tests
7. Run full test matrix, verify against Rust `total_cmp`
8. Update documentation

---

## Acceptance Criteria

- [ ] All 1,728 matrix tests pass
- [ ] Existing c13_floating_point tests still pass
- [ ] `cargo test -p xlog-cuda-tests --test certification_suite --release` passes
- [ ] No performance regression (branchless implementation)

---

## Non-Goals

- Changing sort kernel behavior (already uses total ordering)
- Supporting signaling NaN vs quiet NaN distinction in results
- Custom NaN payloads preservation

---

## References

- [IEEE 754-2008 totalOrder predicate](https://en.wikipedia.org/wiki/IEEE_754#Total_ordering_predicate)
- [Rust f64::total_cmp](https://doc.rust-lang.org/std/primitive.f64.html#method.total_cmp)
- Existing implementation: `crates/xlog-cuda-tests/src/categories/c13_floating_point.rs`
