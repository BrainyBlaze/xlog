// XLOG GPU Deduplication Kernels
// Sort-based deduplication with prefix sum compaction

#include <cstdint>

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
    uint32_t num_rows,
    uint8_t* __restrict__ unique_mask
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

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
    uint32_t num_rows,
    uint8_t* __restrict__ unique_mask,
    uint32_t* __restrict__ prefix_sum,
    uint32_t* __restrict__ block_sums
) {
    __shared__ uint32_t temp[BLOCK_SIZE];

    uint32_t tid = threadIdx.x;
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t val = 0;
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

    if (row < num_rows) {
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
