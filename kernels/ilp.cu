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

/// Fill COO arrays from device-side mask + prefix-sum.
/// Reads write offset from d_offsets[cidx] on device, avoiding mask D2H.
/// For each set bit in mask, writes the corresponding fact_index and cidx
/// into the COO arrays at the position determined by (offset + prefix_sum[tid]).
extern "C" __global__ void ilp_coo_fill_from_mask(
    const uint8_t* mask,
    const uint32_t* prefix_sum,
    const uint32_t* fact_indices,
    uint32_t cidx,
    uint32_t num_query,
    const uint32_t* d_offsets,
    uint32_t* coo_fact,
    uint32_t* coo_cand
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_query) return;
    if (mask[tid]) {
        uint32_t offset = d_offsets[cidx];
        uint32_t write_idx = offset + prefix_sum[tid];
        coo_fact[write_idx] = fact_indices[tid];
        coo_cand[write_idx] = cidx;
    }
}

/// Histogram of fact indices for CSR row_offsets construction.
/// Each thread atomically increments hist[sorted_facts[tid]].
/// Caller must zero hist before launch.
extern "C" __global__ void ilp_csr_histogram(
    const uint32_t* sorted_facts,
    uint32_t nnz,
    uint32_t num_facts,
    uint32_t* hist
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;
    uint32_t f = sorted_facts[tid];
    if (f < num_facts) {
        atomicAdd(&hist[f], 1);
    }
}
