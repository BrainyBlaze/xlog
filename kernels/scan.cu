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
