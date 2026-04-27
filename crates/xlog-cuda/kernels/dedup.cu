// XLOG GPU Deduplication Kernels
// Sort-based deduplication with prefix sum compaction

#include <cstdint>
#include <cstring>

#define BLOCK_SIZE 256

// ScalarType encoding (must match host-side mapping in provider.rs)
#define XLOG_TY_U32    0
#define XLOG_TY_U64    1
#define XLOG_TY_I32    2
#define XLOG_TY_I64    3
#define XLOG_TY_F32    4
#define XLOG_TY_F64    5
#define XLOG_TY_BOOL   6
#define XLOG_TY_SYMBOL 7

__device__ __forceinline__ bool xlog_f32_equal(uint32_t a_bits, uint32_t b_bits) {
    float a = __uint_as_float(a_bits);
    float b = __uint_as_float(b_bits);
    // Treat -0.0 and +0.0 as equal; treat NaN==NaN only when bit-identical.
    return (a == b) || (a_bits == b_bits);
}

__device__ __forceinline__ bool xlog_f64_equal(uint64_t a_bits, uint64_t b_bits) {
    double a = __longlong_as_double((long long)a_bits);
    double b = __longlong_as_double((long long)b_bits);
    return (a == b) || (a_bits == b_bits);
}

// Mark unique rows by comparing adjacent rows across key columns stored column-major.
//
// Each key column is provided as a device pointer (address) in `col_ptrs`.
// The kernel compares row i and row i-1 for each key column; if all keys are equal,
// unique_mask[i]=0, else unique_mask[i]=1. Row 0 is always unique.
extern "C" __global__ void mark_unique_columnar(
    const uint64_t* __restrict__ col_ptrs,
    const uint32_t* __restrict__ col_sizes,
    const uint8_t* __restrict__ col_types,
    uint32_t num_key_cols,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint8_t* __restrict__ unique_mask
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_cap) return;

    uint32_t num_rows = *num_rows_device;
    if (num_rows > row_cap) {
        num_rows = row_cap;
    }
    if (row >= num_rows) {
        unique_mask[row] = 0;
        return;
    }

    if (row == 0) {
        unique_mask[0] = 1;
        return;
    }

    bool is_dup = true;
    for (uint32_t c = 0; c < num_key_cols; c++) {
        uint8_t ty = col_types[c];
        uint64_t base = col_ptrs[c];

        if (ty == XLOG_TY_F64) {
            const uint64_t* col = (const uint64_t*)(uintptr_t)base;
            uint64_t a = col[row - 1];
            uint64_t b = col[row];
            if (!xlog_f64_equal(a, b)) {
                is_dup = false;
                break;
            }
        } else if (ty == XLOG_TY_F32) {
            const uint32_t* col = (const uint32_t*)(uintptr_t)base;
            uint32_t a = col[row - 1];
            uint32_t b = col[row];
            if (!xlog_f32_equal(a, b)) {
                is_dup = false;
                break;
            }
        } else if (ty == XLOG_TY_U64 || ty == XLOG_TY_I64) {
            const uint64_t* col = (const uint64_t*)(uintptr_t)base;
            if (col[row - 1] != col[row]) {
                is_dup = false;
                break;
            }
        } else if (ty == XLOG_TY_U32 || ty == XLOG_TY_I32 || ty == XLOG_TY_SYMBOL) {
            const uint32_t* col = (const uint32_t*)(uintptr_t)base;
            if (col[row - 1] != col[row]) {
                is_dup = false;
                break;
            }
        } else if (ty == XLOG_TY_BOOL) {
            const uint8_t* col = (const uint8_t*)(uintptr_t)base;
            if (col[row - 1] != col[row]) {
                is_dup = false;
                break;
            }
        } else {
            // Fallback: byte compare using col_sizes[c]
            const uint8_t* col = (const uint8_t*)(uintptr_t)base;
            uint32_t sz = col_sizes[c];
            const uint8_t* a = col + (uint64_t)(row - 1) * sz;
            const uint8_t* b = col + (uint64_t)row * sz;
            for (uint32_t i = 0; i < sz; i++) {
                if (a[i] != b[i]) {
                    is_dup = false;
                    break;
                }
            }
            if (!is_dup) break;
        }
    }

    unique_mask[row] = is_dup ? 0 : 1;
}

// Fused kernel: mark_unique_columnar + multiblock_scan_phase1 (exclusive) for the unique mask.
//
// Produces:
// - unique_mask[row] (u8, 0/1)
// - prefix_sum[row] (exclusive scan of unique_mask within the block)
// - block_sums[blockIdx] (total uniques in the block)
extern "C" __global__ void mark_unique_and_scan_columnar(
    const uint64_t* __restrict__ col_ptrs,
    const uint32_t* __restrict__ col_sizes,
    const uint8_t* __restrict__ col_types,
    uint32_t num_key_cols,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint8_t* __restrict__ unique_mask,
    uint32_t* __restrict__ prefix_sum,
    uint32_t* __restrict__ block_sums
) {
    __shared__ uint32_t temp[BLOCK_SIZE];

    uint32_t tid = threadIdx.x;
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t val = 0;
    if (row < row_cap) {
        uint32_t num_rows = *num_rows_device;
        if (num_rows > row_cap) {
            num_rows = row_cap;
        }
        if (row < num_rows) {
            uint8_t unique = 0;
            if (row == 0) {
                unique = 1;
            } else {
                bool is_dup = true;
                for (uint32_t c = 0; c < num_key_cols; c++) {
                    uint8_t ty = col_types[c];
                    uint64_t base = col_ptrs[c];

                    if (ty == XLOG_TY_F64) {
                        const uint64_t* col = (const uint64_t*)(uintptr_t)base;
                        uint64_t a = col[row - 1];
                        uint64_t b = col[row];
                        if (!xlog_f64_equal(a, b)) {
                            is_dup = false;
                            break;
                        }
                    } else if (ty == XLOG_TY_F32) {
                        const uint32_t* col = (const uint32_t*)(uintptr_t)base;
                        uint32_t a = col[row - 1];
                        uint32_t b = col[row];
                        if (!xlog_f32_equal(a, b)) {
                            is_dup = false;
                            break;
                        }
                    } else if (ty == XLOG_TY_U64 || ty == XLOG_TY_I64) {
                        const uint64_t* col = (const uint64_t*)(uintptr_t)base;
                        if (col[row - 1] != col[row]) {
                            is_dup = false;
                            break;
                        }
                    } else if (ty == XLOG_TY_U32 || ty == XLOG_TY_I32 || ty == XLOG_TY_SYMBOL) {
                        const uint32_t* col = (const uint32_t*)(uintptr_t)base;
                        if (col[row - 1] != col[row]) {
                            is_dup = false;
                            break;
                        }
                    } else if (ty == XLOG_TY_BOOL) {
                        const uint8_t* col = (const uint8_t*)(uintptr_t)base;
                        if (col[row - 1] != col[row]) {
                            is_dup = false;
                            break;
                        }
                    } else {
                        const uint8_t* col = (const uint8_t*)(uintptr_t)base;
                        uint32_t sz = col_sizes[c];
                        const uint8_t* a = col + (uint64_t)(row - 1) * sz;
                        const uint8_t* b = col + (uint64_t)row * sz;
                        for (uint32_t i = 0; i < sz; i++) {
                            if (a[i] != b[i]) {
                                is_dup = false;
                                break;
                            }
                        }
                        if (!is_dup) break;
                    }
                }
                unique = is_dup ? 0 : 1;
            }

            unique_mask[row] = unique;
            val = (uint32_t)unique;
        } else {
            unique_mask[row] = 0;
            val = 0;
        }
    }

    temp[tid] = val;
    __syncthreads();

    // Inclusive scan within block (Hillis-Steele).
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

    if (row < row_cap) {
        prefix_sum[row] = exclusive;
    }

    if (tid == BLOCK_SIZE - 1) {
        block_sums[blockIdx.x] = inclusive;
    }
}

// Mark duplicates in a sorted array
extern "C" __global__ void mark_duplicates(
    const uint32_t* __restrict__ sorted_keys,
    uint32_t num_rows,
    uint32_t num_key_cols,
    uint32_t row_stride,
    uint8_t* __restrict__ unique_mask
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    if (tid == 0) {
        unique_mask[0] = 1;
        return;
    }

    bool is_duplicate = true;
    for (uint32_t k = 0; k < num_key_cols && is_duplicate; k++) {
        uint32_t curr = sorted_keys[tid * row_stride + k];
        uint32_t prev = sorted_keys[(tid - 1) * row_stride + k];
        if (curr != prev) {
            is_duplicate = false;
        }
    }

    unique_mask[tid] = is_duplicate ? 0 : 1;
}

// Compact rows based on unique mask using prefix sum offsets
extern "C" __global__ void compact_rows(
    const uint32_t* __restrict__ input,
    const uint8_t* __restrict__ unique_mask,
    const uint32_t* __restrict__ prefix_sum,
    uint32_t num_rows,
    uint32_t row_stride,
    uint32_t* __restrict__ output
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    if (unique_mask[tid]) {
        uint32_t out_idx = prefix_sum[tid];
        for (uint32_t c = 0; c < row_stride; c++) {
            output[out_idx * row_stride + c] = input[tid * row_stride + c];
        }
    }
}

// Mark unique rows on a sorted multi-column buffer using BYTEWISE per-column
// equality. This matches the deterministic Datalog set semantics enforced by
// the host fallback `BTreeSet<Vec<u8>>`: two rows collapse iff their raw
// column bytes are identical. For floats, this matches IEEE-754 totalOrder
// equality under the project's f32_to_ordered_u32 / f64_to_ordered_u64
// normalization (see kernels/sort.cu) — distinct bit patterns map to
// distinct ordered keys, so bytewise equality on the post-sort buffer is
// equivalent to total-order equality. +0.0 vs -0.0 stay distinct under this
// rule; two NaNs collapse iff bit-identical.
//
// Caller is responsible for sorting the buffer with the multi-column sort
// (which uses total-order normalization for floats) BEFORE invoking this
// kernel. Row 0 is always unique. unique_mask[i] is u8 0/1.
extern "C" __global__ void mark_unique_full_row_bytewise(
    const uint64_t* __restrict__ col_ptrs,
    const uint32_t* __restrict__ col_sizes,
    uint32_t num_cols,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint8_t* __restrict__ unique_mask
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_cap) return;

    uint32_t num_rows = *num_rows_device;
    if (num_rows > row_cap) {
        num_rows = row_cap;
    }
    if (row >= num_rows) {
        unique_mask[row] = 0;
        return;
    }
    if (row == 0) {
        unique_mask[0] = 1;
        return;
    }

    bool is_dup = true;
    for (uint32_t c = 0; c < num_cols && is_dup; c++) {
        const uint8_t* col = (const uint8_t*)(uintptr_t)col_ptrs[c];
        uint32_t sz = col_sizes[c];
        const uint8_t* prev = col + (uint64_t)(row - 1) * sz;
        const uint8_t* curr = col + (uint64_t)row * sz;
        for (uint32_t i = 0; i < sz; i++) {
            if (prev[i] != curr[i]) {
                is_dup = false;
                break;
            }
        }
    }

    unique_mask[row] = is_dup ? 0 : 1;
}

// Per-column typed compare matching the order convention of the project's
// multi-column sort (kernels/sort.cu). For each column, returns -1 / 0 / 1
// based on TYPED order:
//
//   * U32, U64, BOOL, SYMBOL: unsigned compare on raw bits.
//   * I32, I64: sign-flipped unsigned compare (matches `^ 0x80...`).
//   * F32, F64: total-order key compare (`f{32,64}_to_ordered_u{32,64}`),
//     so binary search converges with the sort's ordering.
//
// Equality is the same as bytewise equality across all the typed branches,
// so the dedup mask kernel above and this comparator agree on equality.
__device__ __forceinline__ uint32_t xlog_f32_to_ordered_u32(uint32_t bits) {
    uint32_t sign = bits >> 31;
    uint32_t mask = sign ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

__device__ __forceinline__ uint64_t xlog_f64_to_ordered_u64(uint64_t bits) {
    uint64_t sign = bits >> 63;
    uint64_t mask = sign ? 0xFFFFFFFFFFFFFFFFull : 0x8000000000000000ull;
    return bits ^ mask;
}

__device__ __forceinline__ int32_t xlog_typed_cmp_cell(
    const uint8_t* a_cell,
    const uint8_t* b_cell,
    uint8_t ty
) {
    if (ty == XLOG_TY_F64) {
        uint64_t a, b;
        memcpy(&a, a_cell, 8);
        memcpy(&b, b_cell, 8);
        uint64_t ka = xlog_f64_to_ordered_u64(a);
        uint64_t kb = xlog_f64_to_ordered_u64(b);
        if (ka < kb) return -1;
        if (ka > kb) return 1;
        return 0;
    } else if (ty == XLOG_TY_F32) {
        uint32_t a, b;
        memcpy(&a, a_cell, 4);
        memcpy(&b, b_cell, 4);
        uint32_t ka = xlog_f32_to_ordered_u32(a);
        uint32_t kb = xlog_f32_to_ordered_u32(b);
        if (ka < kb) return -1;
        if (ka > kb) return 1;
        return 0;
    } else if (ty == XLOG_TY_U64) {
        uint64_t a, b;
        memcpy(&a, a_cell, 8);
        memcpy(&b, b_cell, 8);
        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
    } else if (ty == XLOG_TY_I64) {
        uint64_t a, b;
        memcpy(&a, a_cell, 8);
        memcpy(&b, b_cell, 8);
        a ^= 0x8000000000000000ull;
        b ^= 0x8000000000000000ull;
        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
    } else if (ty == XLOG_TY_U32 || ty == XLOG_TY_SYMBOL) {
        uint32_t a, b;
        memcpy(&a, a_cell, 4);
        memcpy(&b, b_cell, 4);
        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
    } else if (ty == XLOG_TY_I32) {
        uint32_t a, b;
        memcpy(&a, a_cell, 4);
        memcpy(&b, b_cell, 4);
        a ^= 0x80000000u;
        b ^= 0x80000000u;
        if (a < b) return -1;
        if (a > b) return 1;
        return 0;
    } else if (ty == XLOG_TY_BOOL) {
        if (a_cell[0] < b_cell[0]) return -1;
        if (a_cell[0] > b_cell[0]) return 1;
        return 0;
    }
    return 0;
}

// Mark rows of sorted, deduped buffer `a` that do NOT appear in sorted,
// deduped buffer `b`. Each thread handles one a-row and performs a binary
// search over b's row space using the typed per-column comparator above,
// which agrees with the multi-column sort's order convention. The probe
// converges because the buffers were sorted with the same convention; the
// equality result is the same as bytewise equality (totalOrder
// normalization is bijective, so distinct bit patterns map to distinct
// ordered keys).
//
// keep_mask[t] = 1 iff a[t] is NOT found in b.
extern "C" __global__ void mark_diff_full_row_typed_sorted(
    const uint64_t* __restrict__ a_col_ptrs,
    const uint64_t* __restrict__ b_col_ptrs,
    const uint32_t* __restrict__ col_sizes,
    const uint8_t* __restrict__ col_types,
    uint32_t num_cols,
    const uint32_t* __restrict__ num_a_device,
    uint32_t num_b,
    uint32_t a_cap,
    uint8_t* __restrict__ keep_mask
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= a_cap) return;

    uint32_t num_a = *num_a_device;
    if (num_a > a_cap) num_a = a_cap;
    if (row >= num_a) {
        keep_mask[row] = 0;
        return;
    }

    if (num_b == 0) {
        keep_mask[row] = 1;
        return;
    }

    int32_t lo = 0;
    int32_t hi = (int32_t)num_b;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        int32_t cmp = 0;
        for (uint32_t c = 0; c < num_cols; c++) {
            const uint8_t* a_col = (const uint8_t*)(uintptr_t)a_col_ptrs[c];
            const uint8_t* b_col = (const uint8_t*)(uintptr_t)b_col_ptrs[c];
            uint32_t sz = col_sizes[c];
            const uint8_t* a_cell = a_col + (uint64_t)row * sz;
            const uint8_t* b_cell = b_col + (uint64_t)mid * sz;
            cmp = xlog_typed_cmp_cell(a_cell, b_cell, col_types[c]);
            if (cmp != 0) break;
        }
        if (cmp < 0) {
            hi = mid;
        } else if (cmp > 0) {
            lo = mid + 1;
        } else {
            keep_mask[row] = 0;
            return;
        }
    }
    keep_mask[row] = 1;
}
