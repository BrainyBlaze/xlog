// kernels/scan.cu
#include <cstdint>

#define BLOCK_SIZE 256

// Shared memory block-level inclusive scan (Blelloch)
extern "C" __global__ void block_inclusive_scan(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t* __restrict__ block_sums,
    uint32_t n
) {
    __shared__ uint32_t temp[BLOCK_SIZE * 2];

    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    temp[tid] = (gid < n) ? input[gid] : 0;
    __syncthreads();

    // Up-sweep (reduce)
    for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
        uint32_t index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Store block sum before down-sweep
    if (tid == blockDim.x - 1) {
        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = temp[tid];
        }
    }

    // Down-sweep for exclusive scan, then shift for inclusive
    if (tid == blockDim.x - 1) {
        temp[tid] = 0;
    }
    __syncthreads();

    for (uint32_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        uint32_t index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            uint32_t t = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += t;
        }
        __syncthreads();
    }

    // Write output (shift by one for inclusive scan)
    if (gid < n) {
        output[gid] = temp[tid] + input[gid];
    }
}

// Add block offsets to convert block-local scans to global scan
extern "C" __global__ void add_block_offsets(
    uint32_t* __restrict__ data,
    const uint32_t* __restrict__ block_offsets,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && blockIdx.x > 0) {
        data[gid] += block_offsets[blockIdx.x - 1];
    }
}

// Exclusive prefix sum of u8 mask (for stream compaction)
extern "C" __global__ void exclusive_scan_mask(
    const uint8_t* __restrict__ mask,
    uint32_t* __restrict__ prefix_sum,
    uint32_t n
) {
    __shared__ uint32_t temp[BLOCK_SIZE];

    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load mask as u32
    temp[tid] = (gid < n) ? (uint32_t)mask[gid] : 0;
    __syncthreads();

    // Exclusive scan within block
    for (uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
        uint32_t val = 0;
        if (tid >= stride) {
            val = temp[tid - stride];
        }
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write exclusive scan (shift by 1)
    if (gid < n) {
        prefix_sum[gid] = (tid == 0) ? 0 : temp[tid - 1];
    }
}

// Count total 1s in mask
extern "C" __global__ void count_mask(
    const uint8_t* __restrict__ mask,
    uint32_t n,
    uint32_t* __restrict__ count
) {
    __shared__ uint32_t block_count;
    if (threadIdx.x == 0) block_count = 0;
    __syncthreads();

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n && mask[gid]) {
        atomicAdd(&block_count, 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, block_count);
    }
}

// ============== Multi-Block Scan Kernels ==============
// Three-phase algorithm for arbitrary-length prefix sums:
// Phase 1: Each block computes local exclusive scan, outputs block total
// Phase 2: Scan the block totals array (sequential on CPU for small arrays)
// Phase 3: Add block offsets to each element

// Phase 1: Block-level exclusive scan of mask, output block totals
// Each block processes BLOCK_SIZE elements, outputs its total sum to block_sums
extern "C" __global__ void multiblock_scan_phase1(
    const uint8_t* __restrict__ mask,
    uint32_t* __restrict__ prefix_sum,
    uint32_t* __restrict__ block_sums,
    uint32_t n
) {
    __shared__ uint32_t temp[BLOCK_SIZE];

    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load mask as u32 into shared memory
    uint32_t val = (gid < n) ? (uint32_t)mask[gid] : 0;
    temp[tid] = val;
    __syncthreads();

    // Inclusive scan within block (Hillis-Steele style)
    for (uint32_t stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        uint32_t left_val = 0;
        if (tid >= stride) {
            left_val = temp[tid - stride];
        }
        __syncthreads();
        temp[tid] += left_val;
        __syncthreads();
    }

    // Now temp[tid] contains inclusive scan
    // For exclusive scan, shift by 1
    uint32_t inclusive = temp[tid];
    uint32_t exclusive = (tid == 0) ? 0 : temp[tid - 1];

    // Write exclusive scan result
    if (gid < n) {
        prefix_sum[gid] = exclusive;
    }

    // Last thread of each block writes block total (inclusive scan of last element)
    if (tid == BLOCK_SIZE - 1) {
        block_sums[blockIdx.x] = inclusive;
    }
}

// Phase 2: Scan the block sums array (exclusive scan)
// This kernel is designed for small arrays (num_blocks typically < 1024)
// Uses a single block to process all block sums
extern "C" __global__ void multiblock_scan_phase2(
    uint32_t* __restrict__ block_sums,
    uint32_t num_blocks
) {
    __shared__ uint32_t temp[BLOCK_SIZE];

    uint32_t tid = threadIdx.x;

    // Load block sums (handle case where num_blocks > BLOCK_SIZE with loop)
    // For simplicity, assume num_blocks <= BLOCK_SIZE * BLOCK_SIZE
    // We'll process in chunks if needed

    // Single-block approach for up to BLOCK_SIZE blocks
    // For larger arrays, we'd need recursive scan, but 256 blocks * 256 elements = 65536
    // which handles up to 65536 elements with this simple approach

    if (tid < num_blocks) {
        temp[tid] = block_sums[tid];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    // Inclusive scan
    for (uint32_t stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        uint32_t left_val = 0;
        if (tid >= stride) {
            left_val = temp[tid - stride];
        }
        __syncthreads();
        temp[tid] += left_val;
        __syncthreads();
    }

    // Convert to exclusive scan and write back
    if (tid < num_blocks) {
        block_sums[tid] = (tid == 0) ? 0 : temp[tid - 1];
    }
}

// Phase 3: Add block offsets to get global exclusive prefix sum
// Skip block 0 since it has no offset
extern "C" __global__ void multiblock_scan_phase3(
    uint32_t* __restrict__ prefix_sum,
    const uint32_t* __restrict__ block_offsets,
    uint32_t n
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Block 0 doesn't need offset (offset is 0)
    if (gid < n && blockIdx.x > 0) {
        prefix_sum[gid] += block_offsets[blockIdx.x];
    }
}
