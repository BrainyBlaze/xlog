// XLOG GPU GroupBy Kernels
// Sorted-input group aggregation

#include <cstdint>
#include <cfloat>

// Detect group boundaries in sorted data
extern "C" __global__ void detect_group_boundaries(
    const uint32_t* __restrict__ sorted_keys,
    uint32_t num_rows,
    uint32_t num_key_cols,
    uint32_t row_stride,
    uint8_t* __restrict__ is_boundary
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    if (tid == 0) {
        is_boundary[0] = 1;
        return;
    }

    bool boundary = false;
    for (uint32_t k = 0; k < num_key_cols; k++) {
        uint32_t curr = sorted_keys[tid * row_stride + k];
        uint32_t prev = sorted_keys[(tid - 1) * row_stride + k];
        if (curr != prev) {
            boundary = true;
            break;
        }
    }

    is_boundary[tid] = boundary ? 1 : 0;
}

// Count aggregation per group
extern "C" __global__ void groupby_count(
    const uint8_t* __restrict__ is_boundary,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    uint32_t* __restrict__ counts
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t group = group_ids[tid];
    atomicAdd(&counts[group], 1);
}

// Sum aggregation per group
extern "C" __global__ void groupby_sum(
    const uint32_t* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    uint64_t* __restrict__ sums
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t group = group_ids[tid];
    atomicAdd((unsigned long long*)&sums[group], (unsigned long long)values[tid]);
}

// Min aggregation per group
extern "C" __global__ void groupby_min(
    const uint32_t* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    uint32_t* __restrict__ mins
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t group = group_ids[tid];
    atomicMin(&mins[group], values[tid]);
}

// Max aggregation per group
extern "C" __global__ void groupby_max(
    const uint32_t* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    uint32_t* __restrict__ maxs
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t group = group_ids[tid];
    atomicMax(&maxs[group], values[tid]);
}
