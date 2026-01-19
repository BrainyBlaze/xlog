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

// Build per-row group ids from boundary flags and exclusive prefix sum.
extern "C" __global__ void group_ids_from_boundaries(
    const uint8_t* __restrict__ boundaries,
    const uint32_t* __restrict__ boundary_pos,
    uint32_t num_rows,
    uint32_t* __restrict__ group_ids
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint32_t prefix = boundary_pos[gid];
    uint32_t is_boundary = boundaries[gid] ? 1 : 0;
    group_ids[gid] = prefix + is_boundary - 1;
}

// Record the first row index for each group from boundary flags.
extern "C" __global__ void group_start_indices(
    const uint8_t* __restrict__ boundaries,
    const uint32_t* __restrict__ boundary_pos,
    uint32_t num_rows,
    uint32_t* __restrict__ group_first_idx
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;
    if (boundaries[gid]) {
        uint32_t group = boundary_pos[gid];
        group_first_idx[group] = gid;
    }
}

// Count aggregation per group (outputs u64 to match predicate declarations)
extern "C" __global__ void groupby_count(
    const uint8_t* __restrict__ is_boundary,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    unsigned long long* __restrict__ counts
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t group = group_ids[tid];
    atomicAdd(&counts[group], 1ULL);
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

// Detect group boundaries in sorted data (single-column version)
// More efficient for single-key groupby operations
extern "C" __global__ void detect_boundaries(
    const uint32_t* __restrict__ keys,
    uint32_t num_rows,
    uint8_t* __restrict__ boundaries  // 1 if keys[i] != keys[i-1]
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    if (gid == 0) {
        boundaries[gid] = 1;  // First element is always a boundary
    } else {
        boundaries[gid] = (keys[gid] != keys[gid - 1]) ? 1 : 0;
    }
}

// Extract unique group keys (first element of each group)
extern "C" __global__ void extract_group_keys(
    const uint32_t* __restrict__ keys,
    const uint8_t* __restrict__ boundaries,
    const uint32_t* __restrict__ boundary_positions,  // prefix sum of boundaries
    uint32_t num_rows,
    uint32_t* __restrict__ group_keys
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    if (boundaries[gid] == 1) {
        uint32_t group_idx = boundary_positions[gid];
        group_keys[group_idx] = keys[gid];
    }
}

// LogSumExp Pass 1: Find max value per group
extern "C" __global__ void groupby_logsumexp_max(
    const double* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    uint32_t num_rows,
    double* __restrict__ maxs  // Initialize to -INFINITY
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t group = group_ids[tid];
    double val = values[tid];

    // Atomic max for doubles using CAS
    unsigned long long* addr = (unsigned long long*)&maxs[group];
    unsigned long long old = *addr;
    unsigned long long assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        if (val <= old_val) break;
        old = atomicCAS(addr, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

// LogSumExp Pass 2: Compute sum of exp(x - max) per group
extern "C" __global__ void groupby_logsumexp_sumexp(
    const double* __restrict__ values,
    const uint32_t* __restrict__ group_ids,
    const double* __restrict__ maxs,
    uint32_t num_rows,
    double* __restrict__ sumexps  // Initialize to 0
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t group = group_ids[tid];
    double val = values[tid];
    double max_val = maxs[group];
    double exp_val = exp(val - max_val);

    // Atomic add for doubles using CAS
    unsigned long long* addr = (unsigned long long*)&sumexps[group];
    unsigned long long old = *addr;
    unsigned long long assumed;
    do {
        assumed = old;
        double sum = __longlong_as_double(assumed) + exp_val;
        old = atomicCAS(addr, assumed, __double_as_longlong(sum));
    } while (assumed != old);
}

// LogSumExp Pass 3: Compute final result = max + log(sumexp)
// Special case: If max is +INFINITY, result is +INFINITY regardless of sumexp
// (since exp(val - inf) would produce NaN for finite or +inf values)
extern "C" __global__ void groupby_logsumexp_final(
    const double* __restrict__ maxs,
    const double* __restrict__ sumexps,
    uint32_t num_groups,
    double* __restrict__ results
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_groups) return;

    double max_val = maxs[gid];
    // If max is +INFINITY, result is +INFINITY regardless of sumexp
    if (isinf(max_val) && max_val > 0) {
        results[gid] = max_val;
    } else {
        results[gid] = max_val + log(sumexps[gid]);
    }
}
