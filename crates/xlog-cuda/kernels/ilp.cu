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
/// Reads write offset from d_offsets[offset_idx] on device, avoiding a
/// device-to-host mask copy.
/// For each set bit in mask, writes the corresponding fact_index and cand_value
/// into the COO arrays at the position determined by (offset + prefix_sum[tid]).
///
/// Parameters:
///   offset_idx  — index into d_offsets for the write base position
///   cand_value  — actual candidate index to write into coo_cand
extern "C" __global__ void ilp_coo_fill_from_mask(
    const uint8_t* mask,
    const uint32_t* prefix_sum,
    const uint32_t* fact_indices,
    uint32_t offset_idx,
    uint32_t cand_value,
    uint32_t num_query,
    const uint32_t* d_offsets,
    uint32_t* coo_fact,
    uint32_t* coo_cand
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_query) return;
    if (mask[tid]) {
        uint32_t offset = d_offsets[offset_idx];
        uint32_t write_idx = offset + prefix_sum[tid];
        coo_fact[write_idx] = fact_indices[tid];
        coo_cand[write_idx] = cand_value;
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

// ---------------------------------------------------------------------------
// Strict sparse mask helpers
// ---------------------------------------------------------------------------

extern "C" __global__ void ilp_mark_selected_ids_u32(
    const uint32_t* selected_ids,
    uint32_t selected_len,
    uint32_t candidate_count,
    uint32_t* active_flags
) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= selected_len) return;
    uint32_t cid = selected_ids[pos];
    if (cid >= candidate_count) {
        return;
    }
    atomicExch(&active_flags[cid], 1u);
}

__device__ inline void ilp_record_selected_id_error(
    uint32_t code,
    uint32_t pos,
    uint32_t* error_code,
    uint32_t* error_pos
) {
    if (atomicCAS(error_code, 0u, code) == 0u) {
        *error_pos = pos;
    }
}

extern "C" __global__ void ilp_validate_selected_ids_u32(
    const uint32_t* selected_ids,
    uint32_t selected_len,
    uint32_t candidate_count,
    uint32_t* seen_flags,
    uint32_t* error_code,
    uint32_t* error_pos
) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= selected_len) return;
    uint32_t cid = selected_ids[pos];
    if (cid >= candidate_count) {
        ilp_record_selected_id_error(1u, pos, error_code, error_pos);
        return;
    }
    if (atomicExch(&seen_flags[cid], 1u) != 0u) {
        ilp_record_selected_id_error(2u, pos, error_code, error_pos);
    }
}

extern "C" __global__ void ilp_mark_selected_ids_i32(
    const int32_t* selected_ids,
    uint32_t selected_len,
    uint32_t candidate_count,
    uint32_t* active_flags
) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= selected_len) return;
    int32_t cid = selected_ids[pos];
    if (cid < 0 || static_cast<uint32_t>(cid) >= candidate_count) {
        return;
    }
    atomicExch(&active_flags[static_cast<uint32_t>(cid)], 1u);
}

extern "C" __global__ void ilp_validate_selected_ids_i32(
    const int32_t* selected_ids,
    uint32_t selected_len,
    uint32_t candidate_count,
    uint32_t* seen_flags,
    uint32_t* error_code,
    uint32_t* error_pos
) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= selected_len) return;
    int32_t cid = selected_ids[pos];
    if (cid < 0 || static_cast<uint32_t>(cid) >= candidate_count) {
        ilp_record_selected_id_error(1u, pos, error_code, error_pos);
        return;
    }
    if (atomicExch(&seen_flags[static_cast<uint32_t>(cid)], 1u) != 0u) {
        ilp_record_selected_id_error(2u, pos, error_code, error_pos);
    }
}

extern "C" __global__ void ilp_mark_selected_ids_i64(
    const int64_t* selected_ids,
    uint32_t selected_len,
    uint32_t candidate_count,
    uint32_t* active_flags
) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= selected_len) return;
    int64_t cid = selected_ids[pos];
    if (cid < 0 || static_cast<uint64_t>(cid) >= static_cast<uint64_t>(candidate_count)) {
        return;
    }
    atomicExch(&active_flags[static_cast<uint32_t>(cid)], 1u);
}

extern "C" __global__ void ilp_validate_selected_ids_i64(
    const int64_t* selected_ids,
    uint32_t selected_len,
    uint32_t candidate_count,
    uint32_t* seen_flags,
    uint32_t* error_code,
    uint32_t* error_pos
) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= selected_len) return;
    int64_t cid = selected_ids[pos];
    if (cid < 0 || static_cast<uint64_t>(cid) >= static_cast<uint64_t>(candidate_count)) {
        ilp_record_selected_id_error(1u, pos, error_code, error_pos);
        return;
    }
    if (atomicExch(&seen_flags[static_cast<uint32_t>(cid)], 1u) != 0u) {
        ilp_record_selected_id_error(2u, pos, error_code, error_pos);
    }
}

extern "C" __global__ void ilp_mark_selected_ids_u64(
    const uint64_t* selected_ids,
    uint32_t selected_len,
    uint32_t candidate_count,
    uint32_t* active_flags
) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= selected_len) return;
    uint64_t cid = selected_ids[pos];
    if (cid >= static_cast<uint64_t>(candidate_count)) {
        return;
    }
    atomicExch(&active_flags[static_cast<uint32_t>(cid)], 1u);
}

extern "C" __global__ void ilp_validate_selected_ids_u64(
    const uint64_t* selected_ids,
    uint32_t selected_len,
    uint32_t candidate_count,
    uint32_t* seen_flags,
    uint32_t* error_code,
    uint32_t* error_pos
) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= selected_len) return;
    uint64_t cid = selected_ids[pos];
    if (cid >= static_cast<uint64_t>(candidate_count)) {
        ilp_record_selected_id_error(1u, pos, error_code, error_pos);
        return;
    }
    if (atomicExch(&seen_flags[static_cast<uint32_t>(cid)], 1u) != 0u) {
        ilp_record_selected_id_error(2u, pos, error_code, error_pos);
    }
}

extern "C" __global__ void ilp_broadcast_candidate_flag(
    const uint32_t* active_flags,
    uint32_t candidate_idx,
    uint32_t num_rows,
    uint8_t* row_mask
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;
    row_mask[tid] = active_flags[candidate_idx] == 0u ? 0u : 1u;
}

// ---------------------------------------------------------------------------
// Reduction kernels for device-side loss summation
// ---------------------------------------------------------------------------

#define REDUCE_BLOCK_SIZE 256

/// Block-level sum reduction (f32). Each block reduces its stripe via shared
/// memory, then atomicAdds the partial sum to block_sums[0].
extern "C" __global__ void ilp_reduce_sum_f32(
    const float* input,
    uint32_t n,
    float* block_sums
) {
    __shared__ float sdata[REDUCE_BLOCK_SIZE];
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&block_sums[0], sdata[0]);
    }
}

/// Block-level sum reduction (f64). Requires sm_60+ for double atomicAdd.
extern "C" __global__ void ilp_reduce_sum_f64(
    const double* input,
    uint32_t n,
    double* block_sums
) {
    __shared__ double sdata[REDUCE_BLOCK_SIZE];
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? input[gid] : 0.0;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&block_sums[0], sdata[0]);
    }
}
