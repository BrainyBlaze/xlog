#include <stdint.h>

/// Extract nonzero indices from a flat N*N*N binary mask.
/// For each element where mask_hard[idx] > 0.5, outputs (i, j, k) and
/// the soft-mask priority value. Caller sorts by priority and truncates.
extern "C" __global__ void extract_nonzero_indices(
    const float* mask_hard,
    const float* mask_soft,
    uint32_t N,
    uint32_t* out_i,
    uint32_t* out_j,
    uint32_t* out_k,
    float*    out_priority,
    uint32_t* active_count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * N * N;
    if (idx >= total) return;
    if (mask_hard[idx] > 0.5f) {
        uint32_t pos = atomicAdd(active_count, 1);
        out_i[pos] = idx / (N * N);
        out_j[pos] = (idx / N) % N;
        out_k[pos] = idx % N;
        out_priority[pos] = mask_soft[idx];
    }
}
