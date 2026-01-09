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
 * IMPORTANT: This is a STABLE sort. Elements with the same digit retain
 * their relative input order. This is required for radix sort correctness.
 */

#define BLOCK_SIZE 256
#define RADIX_BITS 4
#define RADIX_SIZE (1 << RADIX_BITS)  // 16 buckets

/**
 * Compute histogram of radix digits for current pass.
 * Each block computes a local histogram using shared memory atomics,
 * then writes to global histograms array.
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
 * Compute per-element ranks within each digit group.
 * This ensures stable sorting by assigning each element its position
 * within elements of the same digit based on input order.
 *
 * @param keys Input keys
 * @param num_rows Number of elements
 * @param ranks Output: rank of each element within its digit group
 * @param shift Bit shift for current pass
 */
extern "C" __global__ void compute_ranks(
    const uint32_t* __restrict__ keys,
    uint32_t num_rows,
    uint32_t* __restrict__ ranks,
    uint32_t shift
) {
    __shared__ uint32_t block_digits[BLOCK_SIZE];

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t block_start = blockIdx.x * blockDim.x;
    uint32_t block_end = min(block_start + BLOCK_SIZE, num_rows);
    uint32_t block_count = block_end - block_start;

    // Load this thread's digit into shared memory
    uint32_t my_digit = 0;
    if (gid < num_rows) {
        my_digit = (keys[gid] >> shift) & (RADIX_SIZE - 1);
        block_digits[threadIdx.x] = my_digit;
    }
    __syncthreads();

    // Compute rank for each element by counting prior elements with same digit
    // Process elements in order to ensure stability
    if (threadIdx.x < block_count) {
        uint32_t my_rank = 0;
        uint32_t target_digit = block_digits[threadIdx.x];

        // Count elements before this one with the same digit
        for (uint32_t i = 0; i < threadIdx.x; i++) {
            if (block_digits[i] == target_digit) {
                my_rank++;
            }
        }

        ranks[gid] = my_rank;
    }
}

/**
 * Stable scatter using pre-computed ranks.
 * Each element's output position = global_prefix[digit] + block_offset[digit] + rank
 */
extern "C" __global__ void radix_scatter_stable(
    const uint32_t* __restrict__ keys_in,
    const uint32_t* __restrict__ indices_in,
    const uint32_t* __restrict__ ranks,
    uint32_t* __restrict__ keys_out,
    uint32_t* __restrict__ indices_out,
    const uint32_t* __restrict__ prefix_sums,     // [RADIX_SIZE] global prefix sums
    const uint32_t* __restrict__ block_offsets,   // [grid_size * RADIX_SIZE]
    uint32_t num_rows,
    uint32_t shift
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint32_t key = keys_in[gid];
    uint32_t digit = (key >> shift) & (RADIX_SIZE - 1);
    uint32_t rank = ranks[gid];

    // Compute block offset for this digit (sum of histograms from previous blocks)
    uint32_t block_offset = 0;
    for (uint32_t b = 0; b < blockIdx.x; b++) {
        block_offset += block_offsets[b * RADIX_SIZE + digit];
    }

    // Final position = global prefix + block offset + rank within block
    uint32_t pos = prefix_sums[digit] + block_offset + rank;

    keys_out[pos] = key;
    indices_out[pos] = indices_in[gid];
}

/**
 * Legacy scatter using atomics (unstable, kept for reference).
 * WARNING: This kernel is NOT stable and should not be used.
 */
extern "C" __global__ void radix_scatter(
    const uint32_t* __restrict__ keys_in,
    const uint32_t* __restrict__ indices_in,
    uint32_t* __restrict__ keys_out,
    uint32_t* __restrict__ indices_out,
    const uint32_t* __restrict__ prefix_sums,
    uint32_t* __restrict__ local_offsets,
    uint32_t num_rows,
    uint32_t shift
) {
    __shared__ uint32_t local_prefix[RADIX_SIZE];

    if (threadIdx.x < RADIX_SIZE) {
        local_prefix[threadIdx.x] = prefix_sums[threadIdx.x];
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
