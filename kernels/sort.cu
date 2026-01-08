// kernels/sort.cu
#include <cstdint>

/**
 * GPU Radix Sort Kernels
 *
 * Implements a 4-bit radix sort for uint32 keys with associated indices.
 *
 * Algorithm Overview:
 * - Uses 8 passes (32 bits / 4 bits per pass = 8 passes)
 * - Each pass: histogram -> prefix sum -> scatter
 *
 * Calling Sequence (per pass):
 * 1. radix_histogram: Compute per-block histograms of radix digits
 *    - Output: histograms[grid_size * 16] (16 buckets per block)
 *
 * 2. Host/GPU prefix sum: Compute global prefix sums from histograms
 *    - Sum histograms across blocks to get global counts
 *    - Compute exclusive prefix sum of global counts
 *
 * 3. radix_scatter: Scatter keys to sorted positions
 *    - Uses prefix sums and per-block offsets to compute output positions
 *
 * Buffer Requirements:
 * - keys_in/keys_out: uint32[num_rows] - double-buffered, swap each pass
 * - indices_in/indices_out: uint32[num_rows] - tracks original positions
 * - histograms: uint32[grid_size * 16]
 * - prefix_sums: uint32[16] - global prefix sums
 *
 * Note: The radix_scatter kernel iterates through previous blocks to compute
 * offsets. For large datasets (>100K rows), consider pre-computing per-block
 * prefix sums on the host for better performance.
 */

#define BLOCK_SIZE 256
#define RADIX_BITS 4
#define RADIX_SIZE (1 << RADIX_BITS)  // 16 buckets

/**
 * Compute histogram of radix digits for current pass.
 * Each block computes a local histogram using shared memory atomics,
 * then writes to global histograms array.
 * @param keys Input keys
 * @param num_rows Number of elements
 * @param histograms Output: [grid_size * RADIX_SIZE]
 * @param shift Bit shift for current pass (0, 4, 8, ..., 28)
 */
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

/**
 * Scatter keys to sorted positions based on prefix sums.
 * Uses atomic adds to shared memory to handle intra-block ordering.
 * @param keys_in Input keys
 * @param indices_in Input indices (tracks original positions)
 * @param keys_out Output keys (sorted for this pass)
 * @param indices_out Output indices
 * @param prefix_sums Global prefix sums [RADIX_SIZE]
 * @param local_offsets Same as histograms from histogram pass
 * @param num_rows Number of elements
 * @param shift Bit shift for current pass
 */
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

/** Initialize identity permutation: indices[i] = i */
extern "C" __global__ void init_indices(
    uint32_t* __restrict__ indices,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        indices[gid] = gid;
    }
}

/** Apply permutation to reorder a uint32 column: output[i] = input[perm[i]] */
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

/** Apply permutation to reorder byte arrays of any element size */
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
