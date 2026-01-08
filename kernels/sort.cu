// kernels/sort.cu
#include <cstdint>

#define BLOCK_SIZE 256
#define RADIX_BITS 4
#define RADIX_SIZE (1 << RADIX_BITS)  // 16 buckets

// Compute histogram of radix digits for current pass
extern "C" __global__ void radix_histogram(
    const uint32_t* __restrict__ keys,
    uint32_t num_rows,
    uint32_t* __restrict__ histograms,  // [grid_size * RADIX_SIZE]
    uint32_t shift
) {
    __shared__ uint32_t local_hist[RADIX_SIZE];

    // Initialize local histogram
    if (threadIdx.x < RADIX_SIZE) {
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_rows) {
        uint32_t digit = (keys[gid] >> shift) & (RADIX_SIZE - 1);
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    // Write local histogram to global
    if (threadIdx.x < RADIX_SIZE) {
        histograms[blockIdx.x * RADIX_SIZE + threadIdx.x] = local_hist[threadIdx.x];
    }
}

// Scatter keys to sorted positions based on prefix sums
extern "C" __global__ void radix_scatter(
    const uint32_t* __restrict__ keys_in,
    const uint32_t* __restrict__ indices_in,
    uint32_t* __restrict__ keys_out,
    uint32_t* __restrict__ indices_out,
    const uint32_t* __restrict__ prefix_sums,  // [RADIX_SIZE] global prefix sums
    uint32_t* __restrict__ local_offsets,      // [grid_size * RADIX_SIZE] for local counting
    uint32_t num_rows,
    uint32_t shift
) {
    __shared__ uint32_t local_prefix[RADIX_SIZE];

    // Load global prefix sums
    if (threadIdx.x < RADIX_SIZE) {
        local_prefix[threadIdx.x] = prefix_sums[threadIdx.x];
        // Add local offset from previous blocks
        for (uint32_t b = 0; b < blockIdx.x; b++) {
            local_prefix[threadIdx.x] += local_offsets[b * RADIX_SIZE + threadIdx.x];
        }
    }
    __syncthreads();

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_rows) {
        uint32_t key = keys_in[gid];
        uint32_t digit = (key >> shift) & (RADIX_SIZE - 1);
        uint32_t pos = atomicAdd(&local_prefix[digit], 1);
        keys_out[pos] = key;
        indices_out[pos] = indices_in[gid];
    }
}

// Initialize indices to identity permutation
extern "C" __global__ void init_indices(
    uint32_t* __restrict__ indices,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        indices[gid] = gid;
    }
}

// Apply permutation to reorder a column
extern "C" __global__ void apply_permutation_u32(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ permutation,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        output[gid] = input[permutation[gid]];
    }
}

// Apply permutation to reorder bytes (for any column type)
extern "C" __global__ void apply_permutation_bytes(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ permutation,
    uint32_t num_rows,
    uint32_t elem_size
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < num_rows) {
        uint32_t src_idx = permutation[gid];
        for (uint32_t b = 0; b < elem_size; b++) {
            output[gid * elem_size + b] = input[src_idx * elem_size + b];
        }
    }
}
