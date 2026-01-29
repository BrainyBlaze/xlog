// Monte Carlo evaluation helper kernels.
#include <stdint.h>

extern "C" __global__ void mc_eval_mask_var(
    const uint8_t* __restrict__ sample_bits,
    const uint32_t* __restrict__ var_idx,
    uint32_t n,
    uint8_t* __restrict__ out_mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    uint32_t idx = var_idx[gid];
    out_mask[gid] = sample_bits[idx] ? 1u : 0u;
}

extern "C" __global__ void mc_eval_mask_ad_choice(
    const uint8_t* __restrict__ sample_bits,
    const uint32_t* __restrict__ decision_vars,
    const uint32_t* __restrict__ decision_offsets,
    const uint32_t* __restrict__ decision_lengths,
    const uint32_t* __restrict__ choice_positions,
    uint32_t n,
    uint8_t* __restrict__ out_mask
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }

    const uint32_t offset = decision_offsets[gid];
    const uint32_t len = decision_lengths[gid];
    const uint32_t pos = choice_positions[gid];

    bool selected = true;
    if (pos < len) {
        // Must have current decision set and all previous unset.
        for (uint32_t i = 0; i < pos; ++i) {
            const uint32_t var = decision_vars[offset + i];
            if (sample_bits[var] != 0) {
                selected = false;
                break;
            }
        }
        if (selected) {
            const uint32_t var = decision_vars[offset + pos];
            selected = sample_bits[var] != 0;
        }
    } else {
        // Last choice when there is no explicit "none": all decisions must be false.
        for (uint32_t i = 0; i < len; ++i) {
            const uint32_t var = decision_vars[offset + i];
            if (sample_bits[var] != 0) {
                selected = false;
                break;
            }
        }
    }

    out_mask[gid] = selected ? 1u : 0u;
}

extern "C" __global__ void mc_eval_query_evidence_truth(
    const uint64_t* __restrict__ query_counts_ptrs,
    uint32_t query_count,
    const uint64_t* __restrict__ evidence_counts_ptrs,
    const uint8_t* __restrict__ evidence_expected,
    uint32_t evidence_count,
    uint8_t* __restrict__ out_query_flags,
    uint8_t* __restrict__ out_evidence_ok
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < query_count) {
        const uint32_t* count_ptr =
            reinterpret_cast<const uint32_t*>(query_counts_ptrs[gid]);
        const uint32_t count = count_ptr ? *count_ptr : 0u;
        out_query_flags[gid] = count > 0 ? 1u : 0u;
    }

    if (gid == 0) {
        uint8_t ok = 1u;
        for (uint32_t i = 0; i < evidence_count; i++) {
            const uint32_t* count_ptr =
                reinterpret_cast<const uint32_t*>(evidence_counts_ptrs[i]);
            const uint32_t count = count_ptr ? *count_ptr : 0u;
            const uint8_t holds = count > 0 ? 1u : 0u;
            if (holds != evidence_expected[i]) {
                ok = 0u;
                break;
            }
        }
        out_evidence_ok[0] = ok;
    }
}

extern "C" __global__ void mc_accumulate_counts(
    const uint8_t* __restrict__ query_flags,
    uint32_t num_queries,
    const uint8_t* __restrict__ evidence_ok,
    uint32_t* __restrict__ query_counts,
    uint32_t* __restrict__ evidence_count
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (!evidence_ok[0]) {
            return;
        }
        atomicAdd(evidence_count, 1u);
        for (uint32_t i = 0; i < num_queries; i++) {
            if (query_flags[i]) {
                atomicAdd(&query_counts[i], 1u);
            }
        }
    }
}
