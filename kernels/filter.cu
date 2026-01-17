// kernels/filter.cu
#include <cstdint>

/**
 * GPU Filter Kernels
 *
 * Provides comparison operations and stream compaction for filtering.
 * Works with prefix_sum from scan.cu for compaction indices.
 *
 * Comparison operations (OP values):
 * - 0 = Eq (equal)
 * - 1 = Ne (not equal)
 * - 2 = Lt (less than)
 * - 3 = Le (less than or equal)
 * - 4 = Gt (greater than)
 * - 5 = Ge (greater than or equal)
 */

#define OP_EQ 0
#define OP_NE 1
#define OP_LT 2
#define OP_LE 3
#define OP_GT 4
#define OP_GE 5

#define BLOCK_SIZE 256

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

/**
 * Compare i64 column against constant, output mask.
 * @param column Input column data
 * @param constant Value to compare against
 * @param num_rows Number of elements
 * @param op Comparison operator (0-5)
 * @param mask Output: 1 if comparison true, 0 otherwise
 */
extern "C" __global__ void filter_compare_i64(
    const int64_t* __restrict__ column,
    int64_t constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    int64_t val = column[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (val == constant); break;
        case OP_NE: result = (val != constant); break;
        case OP_LT: result = (val < constant); break;
        case OP_LE: result = (val <= constant); break;
        case OP_GT: result = (val > constant); break;
        case OP_GE: result = (val >= constant); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/** Compare u32 column against constant */
extern "C" __global__ void filter_compare_u32(
    const uint32_t* __restrict__ column,
    uint32_t constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint32_t val = column[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (val == constant); break;
        case OP_NE: result = (val != constant); break;
        case OP_LT: result = (val < constant); break;
        case OP_LE: result = (val <= constant); break;
        case OP_GT: result = (val > constant); break;
        case OP_GE: result = (val >= constant); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/** Compare f64 column against constant */
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
        case OP_LT: result = (val < constant); break;
        case OP_LE: result = (val <= constant); break;
        case OP_GT: result = (val > constant); break;
        case OP_GE: result = (val >= constant); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/** Compare i32 column against constant */
extern "C" __global__ void filter_compare_i32(
    const int32_t* __restrict__ column,
    int32_t constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    int32_t val = column[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (val == constant); break;
        case OP_NE: result = (val != constant); break;
        case OP_LT: result = (val < constant); break;
        case OP_LE: result = (val <= constant); break;
        case OP_GT: result = (val > constant); break;
        case OP_GE: result = (val >= constant); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/** Compare u64 column against constant */
extern "C" __global__ void filter_compare_u64(
    const uint64_t* __restrict__ column,
    uint64_t constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint64_t val = column[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (val == constant); break;
        case OP_NE: result = (val != constant); break;
        case OP_LT: result = (val < constant); break;
        case OP_LE: result = (val <= constant); break;
        case OP_GT: result = (val > constant); break;
        case OP_GE: result = (val >= constant); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/** Compare f32 column against constant */
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
        case OP_LT: result = (val < constant); break;
        case OP_LE: result = (val <= constant); break;
        case OP_GT: result = (val > constant); break;
        case OP_GE: result = (val >= constant); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/** Compare u8 column against constant */
extern "C" __global__ void filter_compare_u8(
    const uint8_t* __restrict__ column,
    uint8_t constant,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint8_t val = column[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (val == constant); break;
        case OP_NE: result = (val != constant); break;
        case OP_LT: result = (val < constant); break;
        case OP_LE: result = (val <= constant); break;
        case OP_GT: result = (val > constant); break;
        case OP_GE: result = (val >= constant); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/** Compare u32 column against column */
extern "C" __global__ void filter_compare_u32_col(
    const uint32_t* __restrict__ left,
    const uint32_t* __restrict__ right,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint32_t lval = left[gid];
    uint32_t rval = right[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (lval == rval); break;
        case OP_NE: result = (lval != rval); break;
        case OP_LT: result = (lval < rval); break;
        case OP_LE: result = (lval <= rval); break;
        case OP_GT: result = (lval > rval); break;
        case OP_GE: result = (lval >= rval); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/** Compare i32 column against column */
extern "C" __global__ void filter_compare_i32_col(
    const int32_t* __restrict__ left,
    const int32_t* __restrict__ right,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    int32_t lval = left[gid];
    int32_t rval = right[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (lval == rval); break;
        case OP_NE: result = (lval != rval); break;
        case OP_LT: result = (lval < rval); break;
        case OP_LE: result = (lval <= rval); break;
        case OP_GT: result = (lval > rval); break;
        case OP_GE: result = (lval >= rval); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/** Compare i64 column against column */
extern "C" __global__ void filter_compare_i64_col(
    const int64_t* __restrict__ left,
    const int64_t* __restrict__ right,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    int64_t lval = left[gid];
    int64_t rval = right[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (lval == rval); break;
        case OP_NE: result = (lval != rval); break;
        case OP_LT: result = (lval < rval); break;
        case OP_LE: result = (lval <= rval); break;
        case OP_GT: result = (lval > rval); break;
        case OP_GE: result = (lval >= rval); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/** Compare u64 column against column */
extern "C" __global__ void filter_compare_u64_col(
    const uint64_t* __restrict__ left,
    const uint64_t* __restrict__ right,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint64_t lval = left[gid];
    uint64_t rval = right[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (lval == rval); break;
        case OP_NE: result = (lval != rval); break;
        case OP_LT: result = (lval < rval); break;
        case OP_LE: result = (lval <= rval); break;
        case OP_GT: result = (lval > rval); break;
        case OP_GE: result = (lval >= rval); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/** Compare f32 column against column */
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
        case OP_LT: result = (lval < rval); break;
        case OP_LE: result = (lval <= rval); break;
        case OP_GT: result = (lval > rval); break;
        case OP_GE: result = (lval >= rval); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/** Compare f64 column against column */
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
        case OP_LT: result = (lval < rval); break;
        case OP_LE: result = (lval <= rval); break;
        case OP_GT: result = (lval > rval); break;
        case OP_GE: result = (lval >= rval); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/** Compare u8 column against column */
extern "C" __global__ void filter_compare_u8_col(
    const uint8_t* __restrict__ left,
    const uint8_t* __restrict__ right,
    uint32_t num_rows,
    uint8_t op,
    uint8_t* __restrict__ mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint8_t lval = left[gid];
    uint8_t rval = right[gid];
    bool result;
    switch (op) {
        case OP_EQ: result = (lval == rval); break;
        case OP_NE: result = (lval != rval); break;
        case OP_LT: result = (lval < rval); break;
        case OP_LE: result = (lval <= rval); break;
        case OP_GT: result = (lval > rval); break;
        case OP_GE: result = (lval >= rval); break;
        default: result = false;
    }
    mask[gid] = result ? 1 : 0;
}

/**
 * Fused compare + scan phase1 for u32 filters.
 *
 * Produces:
 * - mask[gid] (0/1)
 * - prefix_sum[gid] (exclusive scan within the block)
 * - block_sums[blockIdx] (number of kept rows in the block)
 */
extern "C" __global__ void filter_compare_u32_scan_phase1(
    const uint32_t* __restrict__ column,
    uint32_t constant,
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
        uint32_t col_val = column[gid];
        bool result;
        switch (op) {
            case OP_EQ: result = (col_val == constant); break;
            case OP_NE: result = (col_val != constant); break;
            case OP_LT: result = (col_val < constant); break;
            case OP_LE: result = (col_val <= constant); break;
            case OP_GT: result = (col_val > constant); break;
            case OP_GE: result = (col_val >= constant); break;
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

/**
 * Fused compare + scan phase1 for f64 filters.
 *
 * Produces:
 * - mask[gid] (0/1)
 * - prefix_sum[gid] (exclusive scan within the block)
 * - block_sums[blockIdx] (number of kept rows in the block)
 */
extern "C" __global__ void filter_compare_f64_scan_phase1(
    const double* __restrict__ column,
    double constant,
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
        double col_val = column[gid];
        bool result;
        switch (op) {
            case OP_EQ: result = (col_val == constant); break;
            case OP_NE: result = (col_val != constant); break;
            case OP_LT: result = (col_val < constant); break;
            case OP_LE: result = (col_val <= constant); break;
            case OP_GT: result = (col_val > constant); break;
            case OP_GE: result = (col_val >= constant); break;
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

/** Combine two masks with AND */
extern "C" __global__ void mask_and(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    uint8_t* __restrict__ out,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = (a[gid] && b[gid]) ? 1 : 0;
    }
}

/** Combine two masks with OR */
extern "C" __global__ void mask_or(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    uint8_t* __restrict__ out,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = (a[gid] || b[gid]) ? 1 : 0;
    }
}

/** Negate a mask */
extern "C" __global__ void mask_not(
    const uint8_t* __restrict__ a,
    uint8_t* __restrict__ out,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out[gid] = a[gid] ? 0 : 1;
    }
}

/**
 * Compact u32 column by mask using prefix sum indices.
 * @param input Input column
 * @param mask Filter mask (1 = keep, 0 = discard)
 * @param prefix_sum Exclusive prefix sum of mask (from scan.cu)
 * @param num_rows Input size
 * @param output Output column (compacted)
 */
extern "C" __global__ void compact_u32_by_mask(
    const uint32_t* __restrict__ input,
    const uint8_t* __restrict__ mask,
    const uint32_t* __restrict__ prefix_sum,
    uint32_t num_rows,
    uint32_t* __restrict__ output
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    if (mask[gid]) {
        uint32_t out_idx = prefix_sum[gid];
        output[out_idx] = input[gid];
    }
}

/**
 * Compact i64 column by mask using prefix sum indices.
 * @param input Input column (int64_t)
 * @param mask Filter mask (1 = keep, 0 = discard)
 * @param prefix_sum Exclusive prefix sum of mask
 * @param num_rows Input size
 * @param output Output column (compacted)
 */
extern "C" __global__ void compact_i64_by_mask(
    const int64_t* __restrict__ input,
    const uint8_t* __restrict__ mask,
    const uint32_t* __restrict__ prefix_sum,
    uint32_t num_rows,
    int64_t* __restrict__ output
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    if (mask[gid]) {
        uint32_t out_idx = prefix_sum[gid];
        output[out_idx] = input[gid];
    }
}

/**
 * Compact f64 column by mask using prefix sum indices.
 * @param input Input column (double)
 * @param mask Filter mask (1 = keep, 0 = discard)
 * @param prefix_sum Exclusive prefix sum of mask
 * @param num_rows Input size
 * @param output Output column (compacted)
 */
extern "C" __global__ void compact_f64_by_mask(
    const double* __restrict__ input,
    const uint8_t* __restrict__ mask,
    const uint32_t* __restrict__ prefix_sum,
    uint32_t num_rows,
    double* __restrict__ output
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    if (mask[gid]) {
        uint32_t out_idx = prefix_sum[gid];
        output[out_idx] = input[gid];
    }
}

/**
 * Compact bytes by mask (for any element size).
 * @param input Input data
 * @param mask Filter mask
 * @param prefix_sum Exclusive prefix sum of mask
 * @param num_rows Number of rows
 * @param elem_size Bytes per element
 * @param output Output data (compacted)
 */
extern "C" __global__ void compact_bytes_by_mask(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ mask,
    const uint32_t* __restrict__ prefix_sum,
    uint32_t num_rows,
    uint32_t elem_size,
    uint8_t* __restrict__ output
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    if (mask[gid]) {
        uint32_t out_idx = prefix_sum[gid];
        for (uint32_t b = 0; b < elem_size; b++) {
            output[out_idx * elem_size + b] = input[gid * elem_size + b];
        }
    }
}
