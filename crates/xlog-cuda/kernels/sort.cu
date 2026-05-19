// kernels/sort.cu
#include <cstdint>
#include "totalorder.cuh"

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
 *
 * Output layout is digit-major for efficient per-digit prefix scans:
 *   histograms[digit * grid_size + block] = count
 */
extern "C" __global__ void radix_histogram(
    const uint32_t* __restrict__ keys,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t* __restrict__ histograms,  // [RADIX_SIZE * grid_size] (digit-major)
    uint32_t shift
) {
    __shared__ uint32_t local_hist[RADIX_SIZE];

    // Initialize local histogram
    if (threadIdx.x < RADIX_SIZE) {
        local_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < actual) {
        uint32_t digit = (keys[gid] >> shift) & (RADIX_SIZE - 1);
        atomicAdd(&local_hist[digit], 1);
    }
    __syncthreads();

    // Write local histogram to global
    if (threadIdx.x < RADIX_SIZE) {
        uint32_t grid_size = gridDim.x;
        uint32_t digit = threadIdx.x;
        histograms[digit * grid_size + blockIdx.x] = local_hist[digit];
    }
}

/**
 * Compute global digit prefix sums (exclusive) for the current pass.
 *
 * prefix_sums[d] = sum_{k<d} total_count[k]
 *
 * This kernel assumes `radix_histogram` has populated `histograms` in digit-major layout:
 *   histograms[d * grid_size + block]
 */
extern "C" __global__ void compute_digit_prefix_sums(
    const uint32_t* __restrict__ histograms,  // [RADIX_SIZE * grid_size] (digit-major)
    uint32_t grid_size,
    uint32_t* __restrict__ prefix_sums        // [RADIX_SIZE]
) {
    __shared__ uint32_t totals[RADIX_SIZE];

    uint32_t digit = threadIdx.x;
    if (digit < RADIX_SIZE) {
        uint32_t sum = 0;
        const uint32_t base = digit * grid_size;
        for (uint32_t b = 0; b < grid_size; b++) {
            sum += histograms[base + b];
        }
        totals[digit] = sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t running = 0;
        for (uint32_t d = 0; d < RADIX_SIZE; d++) {
            uint32_t t = totals[d];
            prefix_sums[d] = running;
            running += t;
        }
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
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t* __restrict__ ranks,
    uint32_t shift
) {
    __shared__ uint32_t block_digits[BLOCK_SIZE];

    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t block_start = blockIdx.x * blockDim.x;
    uint32_t block_end = min(block_start + BLOCK_SIZE, actual);
    uint32_t block_count = block_end - block_start;

    // Load this thread's digit into shared memory
    uint32_t my_digit = 0;
    if (gid < actual) {
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
    const uint32_t* __restrict__ block_offsets,   // [RADIX_SIZE * grid_size] digit-major, exclusive offsets
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t shift
) {
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= actual) return;

    uint32_t key = keys_in[gid];
    uint32_t digit = (key >> shift) & (RADIX_SIZE - 1);
    uint32_t rank = ranks[gid];

    // Precomputed exclusive offset of this block within the digit group.
    uint32_t grid_size = gridDim.x;
    uint32_t block_offset = block_offsets[digit * grid_size + blockIdx.x];

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
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
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

    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < actual) {
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
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    if (gid < actual) {
        indices[gid] = gid;
    }
}

/** Apply permutation to reorder a uint32 column: output[i] = input[perm[i]] */
extern "C" __global__ void apply_permutation_u32(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    const uint32_t* __restrict__ permutation,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    if (gid < actual) {
        output[gid] = input[permutation[gid]];
    }
}

/** Apply permutation to reorder byte arrays of any element size */
extern "C" __global__ void apply_permutation_bytes(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ permutation,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t elem_size
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    if (gid < actual) {
        uint32_t src_idx = permutation[gid];
        for (uint32_t b = 0; b < elem_size; b++) {
            output[gid * elem_size + b] = input[src_idx * elem_size + b];
        }
    }
}

// ============== Key Transform + Gather Kernels ==============
//
// IEEE totalOrder mapping for f32 / f64 lives in `totalorder.cuh`
// (`xlog_f32_to_ordered_u32`, `xlog_f64_to_ordered_u64`). The sort
// codepath and the dedup/diff codepaths share that single source of
// truth so the deterministic-Datalog set algebra cannot drift between
// "sort by one ordering, probe by another".

extern "C" __global__ void gather_keys_i32_ordered_u32(
    const uint32_t* __restrict__ i32_bits,     // raw bits of i32 values
    const uint32_t* __restrict__ permutation,  // output row -> input row
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t* __restrict__ out_keys
) {
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= actual) return;
    uint32_t src = permutation[gid];
    // Flip sign bit to make signed order match unsigned compare.
    out_keys[gid] = i32_bits[src] ^ 0x80000000u;
}

extern "C" __global__ void gather_keys_f32_ordered_u32(
    const uint32_t* __restrict__ f32_bits,     // raw bits of f32 values
    const uint32_t* __restrict__ permutation,  // output row -> input row
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t* __restrict__ out_keys
) {
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= actual) return;
    uint32_t src = permutation[gid];
    out_keys[gid] = xlog_f32_to_ordered_u32(f32_bits[src]);
}

extern "C" __global__ void gather_keys_bool_ordered_u32(
    const uint8_t* __restrict__ bools,         // 0/1 bytes
    const uint32_t* __restrict__ permutation,  // output row -> input row
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t* __restrict__ out_keys
) {
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= actual) return;
    uint32_t src = permutation[gid];
    out_keys[gid] = bools[src] ? 1u : 0u;
}

extern "C" __global__ void gather_keys_u64_lo_u32(
    const uint64_t* __restrict__ vals,
    const uint32_t* __restrict__ permutation,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t* __restrict__ out_keys
) {
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= actual) return;
    uint64_t v = vals[permutation[gid]];
    out_keys[gid] = (uint32_t)(v & 0xFFFFFFFFull);
}

extern "C" __global__ void gather_keys_u64_hi_u32(
    const uint64_t* __restrict__ vals,
    const uint32_t* __restrict__ permutation,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t* __restrict__ out_keys
) {
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= actual) return;
    uint64_t v = vals[permutation[gid]];
    out_keys[gid] = (uint32_t)(v >> 32);
}

extern "C" __global__ void gather_keys_i64_lo_u32(
    const uint64_t* __restrict__ i64_bits,
    const uint32_t* __restrict__ permutation,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t* __restrict__ out_keys
) {
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= actual) return;
    uint64_t bits = i64_bits[permutation[gid]] ^ 0x8000000000000000ull;
    out_keys[gid] = (uint32_t)(bits & 0xFFFFFFFFull);
}

extern "C" __global__ void gather_keys_i64_hi_u32(
    const uint64_t* __restrict__ i64_bits,
    const uint32_t* __restrict__ permutation,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t* __restrict__ out_keys
) {
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= actual) return;
    uint64_t bits = i64_bits[permutation[gid]] ^ 0x8000000000000000ull;
    out_keys[gid] = (uint32_t)(bits >> 32);
}

extern "C" __global__ void gather_keys_f64_lo_u32(
    const uint64_t* __restrict__ f64_bits,
    const uint32_t* __restrict__ permutation,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t* __restrict__ out_keys
) {
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= actual) return;
    uint64_t ord = xlog_f64_to_ordered_u64(f64_bits[permutation[gid]]);
    out_keys[gid] = (uint32_t)(ord & 0xFFFFFFFFull);
}

extern "C" __global__ void gather_keys_f64_hi_u32(
    const uint64_t* __restrict__ f64_bits,
    const uint32_t* __restrict__ permutation,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t* __restrict__ out_keys
) {
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) {
        actual = row_cap;
    }
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= actual) return;
    uint64_t ord = xlog_f64_to_ordered_u64(f64_bits[permutation[gid]]);
    out_keys[gid] = (uint32_t)(ord >> 32);
}

// ===============================================================
// W4.3 — Detect ascending-sorted U32 key column.
//
// Each thread checks one adjacent pair `(keys[tid], keys[tid+1])`.
// On `keys[tid] > keys[tid+1]` (a violation), atomically writes
// 0 to the single-element `result` flag. Caller initializes
// `result[0] = 1` before launch; reads the result post-launch.
//
// Properties:
//   * Write-rare on sorted inputs (no atomic contention).
//   * Idempotent on violations (multiple threads writing 0 to
//     the same flag don't conflict semantically).
//   * Returns "sorted ascending" semantics: keys[i] <= keys[i+1]
//     for all i in [0, num_rows-1). Duplicates are admitted —
//     sort-merge handles run-length matching downstream.
//   * Caller MUST short-circuit `num_rows < 2` BEFORE launch
//     (per W4.3 plan iter-4 D1 + F-W43-4): the kernel grid
//     `(num_rows + 255) / 256` is undefined for num_rows == 0,
//     and a single-row sequence is trivially sorted (no
//     adjacent pair to check). The provider fn
//     `is_sorted_ascending_u32` enforces this fast path.
// ===============================================================
extern "C" __global__ void check_ascending_sorted_u32(
    const uint32_t* __restrict__ keys,
    uint32_t num_rows,
    uint32_t* __restrict__ result
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread checks the pair (tid, tid+1). The last
    // thread (tid == num_rows - 1) has no successor, so skip.
    if (tid + 1 >= num_rows) return;
    if (keys[tid] > keys[tid + 1]) {
        atomicExch(result, 0u);
    }
}
