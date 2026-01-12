// kernels/set_ops.cu
// GPU Set Operations: Union, Difference
// Uses sorted arrays with binary search for efficient set membership

#include <cstdint>

/**
 * Concatenate two u32 arrays (device-to-device)
 * Used as first step in union: concat + sort + dedup
 *
 * @param a First input array
 * @param a_len Length of first array
 * @param b Second input array
 * @param b_len Length of second array
 * @param output Output array (must be pre-allocated to a_len + b_len)
 */
extern "C" __global__ void concat_u32(
    const uint32_t* __restrict__ a,
    uint32_t a_len,
    const uint32_t* __restrict__ b,
    uint32_t b_len,
    uint32_t* __restrict__ output
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = a_len + b_len;
    if (gid >= total) return;

    if (gid < a_len) {
        output[gid] = a[gid];
    } else {
        output[gid] = b[gid - a_len];
    }
}

/**
 * Concatenate two byte arrays (device-to-device).
 *
 * This is used to concatenate arbitrary typed columns stored as raw bytes.
 *
 * @param a First input byte array
 * @param a_bytes Length of first input in bytes
 * @param b Second input byte array
 * @param b_bytes Length of second input in bytes
 * @param output Output byte array (must be pre-allocated to a_bytes + b_bytes)
 */
extern "C" __global__ void concat_bytes(
    const uint8_t* __restrict__ a,
    uint32_t a_bytes,
    const uint8_t* __restrict__ b,
    uint32_t b_bytes,
    uint8_t* __restrict__ output
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = a_bytes + b_bytes;
    if (gid >= total) return;

    if (gid < a_bytes) {
        output[gid] = a[gid];
    } else {
        output[gid] = b[gid - a_bytes];
    }
}

/**
 * Mark elements in sorted array A that are NOT in sorted array B
 * Uses binary search for O(log n) lookup per element
 *
 * @param a First sorted array (the one we're filtering)
 * @param a_len Length of array a
 * @param b Second sorted array (the set to check against)
 * @param b_len Length of array b
 * @param in_diff Output mask: 1 if a[i] not in b, 0 if a[i] in b
 */
extern "C" __global__ void sorted_diff_mark(
    const uint32_t* __restrict__ a,
    uint32_t a_len,
    const uint32_t* __restrict__ b,
    uint32_t b_len,
    uint8_t* __restrict__ in_diff  // 1 if a[i] not in b
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= a_len) return;

    uint32_t val = a[gid];

    // Binary search in b for val
    uint32_t lo = 0, hi = b_len;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (b[mid] < val) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // Check if found: lo points to first element >= val
    // If b[lo] == val, then val is in b, so NOT in diff
    in_diff[gid] = (lo < b_len && b[lo] == val) ? 0 : 1;
}
