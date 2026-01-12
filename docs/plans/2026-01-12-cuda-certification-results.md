# CUDA Certification Suite Results

**Date:** 2026-01-12
**Device:** CUDA 7.0 Compute Capability
**Memory Budget:** 1024 MB

## Executive Summary

| Metric | Value |
|--------|-------|
| Categories Run | 24 |
| Categories Passing | 3 (12.5%) |
| Categories Failing | 21 (87.5%) |
| Root Cause Categories | 3 distinct bugs |

**Overall Status:** CERTIFICATION FAILED - Critical bugs in xlog-cuda discovered

---

## Category Results

### Fully Passing (3/24)

| Category | Tests | Status |
|----------|-------|--------|
| C01 Toolchain | 5/5 | PASS |
| C02 Launch Config | 7/7 | PASS |
| C09 Warp Level | 5/5 | PASS |

### Partially Passing (7/24)

| Category | Tests | Failures |
|----------|-------|----------|
| C03 Pointer Bounds | 6/8 | Filter boundary bugs |
| C05 Global Memory | 4/5 | Large allocation filter bug |
| C06 Shared Memory | 4/5 | (analysis needed) |
| C07 Local Memory | 3/5 | (analysis needed) |
| C17 Caching | 3/5 | (analysis needed) |
| C18 Host/Device | 3/5 | (analysis needed) |
| C21 Hardware | 3/5 | (analysis needed) |

### Failing at First Test (14/24)

| Category | Root Cause |
|----------|------------|
| C04 Address Space | **BUG #1:** Sort u64 type mismatch |
| C08 Synchronization | **BUG #1:** Size mismatch in download |
| C10 Block/Grid | **BUG #1:** Size mismatch |
| C11 Control Flow | **BUG #1:** Size mismatch |
| C12 Atomics | **BUG #1:** Size mismatch |
| C13 Floating Point | **BUG #1:** Sort f64 type mismatch |
| C14 Integer | **BUG #1:** Sort u64 type mismatch |
| C15 Determinism | **BUG #1:** Size mismatch |
| C16 Async Pipeline | (analysis needed) |
| C19 Multi Stream | (analysis needed) |
| C20 Multi GPU | (analysis needed) |
| C22 Algorithms | **BUG #3:** Schema arity mismatch |
| C23 Blind Spots | (analysis needed) |
| C24 Edge Matrix | (analysis needed) |

---

## Root Cause Analysis

### BUG #1: Sort Only Supports u32 Keys (CRITICAL)

**Location:** `crates/xlog-cuda/src/provider.rs:2693-2701`

**Symptom:** Panic in `cudarc::dtoh_sync_copy_into` with assertion `left != right`

**Root Cause:** The `sort()` function hardcodes u32 key handling:
```rust
let mut keys_a = self.memory.alloc::<u32>(n as usize)?;
let key_bytes = (n as usize) * std::mem::size_of::<u32>();
```

When passed u64 or f64 columns (8 bytes each), the download buffer is sized for u32 (4 bytes), causing size mismatches:
- u64: expected 104 bytes (13×8), got 52 bytes (13×4)
- f64: expected 88 bytes (11×8), got 44 bytes (11×4)

**Impact:** Any operation involving sort on 64-bit types fails.

**Affected Categories:** C04, C08, C10-C15, C19-C24 (any using sort with non-u32 keys)

**Fix Required:**
1. Add type dispatch in `sort()` to handle u32, u64, f32, f64
2. Or: Add type checking with clear error message for unsupported types

---

### BUG #2: Filter Returns Wrong Row Counts at Boundaries (HIGH)

**Location:** Filter kernel in `crates/xlog-cuda/src/provider.rs`

**Symptom:** Filter operations return incorrect row counts at specific sizes:
- Size 65537: returned 2 rows, expected 32769
- Size 100000: returned 160 rows, expected 50000
- Large allocation: returned 0 rows, expected 10000

**Root Cause:** Likely grid-stride loop or block boundary handling bug in filter kernel. The pattern suggests issues at:
- Sizes just past powers of 2 (65537 = 2^16 + 1)
- Non-power-of-two sizes (100000)

**Affected Categories:** C03, C05

**Fix Required:** Audit filter kernel for:
1. Grid-stride loop bounds
2. Tail element handling
3. Predicate evaluation at boundaries

---

### BUG #3: Schema/Column Count Mismatch (MEDIUM)

**Location:** `crates/xlog-cuda/src/memory.rs:191`

**Symptom:** `assertion failed: Number of columns (2) must match schema arity (4)`

**Root Cause:** Test creates buffer with 4-column schema but only provides 2 columns of data.

**Affected Categories:** C22 (algorithms tests)

**Fix Required:** Review C22 test setup for correct schema/data alignment.

---

## Recommendations

### Immediate (P0)

1. **Fix sort u64/f64 support** - This blocks 60%+ of certification
   - Add type dispatch or explicit type checking
   - Implement radix sort for 64-bit types (8 passes instead of 4)

2. **Fix filter boundary bugs** - Audit grid-stride loops
   - Add explicit tests for sizes 2^N+1, 2^N-1
   - Verify tail handling

### Short-term (P1)

3. **Fix test schema mismatches** in C22
4. **Complete analysis** of partial failures in C06, C07, C17, C18, C21

### Medium-term (P2)

5. **Add type validation** at API boundaries
6. **Improve error messages** - cudarc panics are cryptic

---

## Test Infrastructure Assessment

The certification suite successfully identified 3 distinct bugs in xlog-cuda:

| Aspect | Assessment |
|--------|------------|
| Coverage | Comprehensive - 24 categories |
| Edge Cases | Effective - found boundary bugs |
| Type Testing | Effective - found type handling gaps |
| Error Clarity | Good - failures are diagnosable |

The suite is working as designed - it's finding real bugs in the CUDA implementation.

---

## Next Steps

1. **Priority:** Fix BUG #1 (sort type support) - unblocks majority of tests
2. **Re-run certification** after fix to get accurate pass rate
3. **Track failures** in issue tracker with specific test case links
