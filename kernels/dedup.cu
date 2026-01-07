// XLOG GPU Deduplication Kernels
// Sort-based deduplication with prefix sum compaction

#include <cstdint>

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
