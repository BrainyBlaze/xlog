#include <stdint.h>

extern "C" __global__ void epistemic_generate_candidate_assumptions_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint8_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = literal_count * candidate_count;
    if (gid >= total) return;

    uint32_t candidate = gid / literal_count;
    uint32_t literal = gid - candidate * literal_count;
    out[gid] = static_cast<uint8_t>((candidate >> literal) & 1u);
}

extern "C" __global__ void epistemic_propagate_candidates_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t world_stride,
    const uint8_t* __restrict__ candidate_assumptions,
    uint8_t* __restrict__ world_views,
    uint32_t* __restrict__ rejection_reasons
) {
    uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) return;

    uint8_t observed = 0;
    uint32_t base = candidate * literal_count;
    for (uint32_t literal = 0; literal < literal_count; ++literal) {
        observed |= candidate_assumptions[base + literal];
    }

    world_views[candidate * world_stride] = (observed == 255u) ? 0u : 1u;
    rejection_reasons[candidate] = 0u;
}

extern "C" __global__ void epistemic_validate_candidate_bits_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t world_stride,
    const uint8_t* __restrict__ candidate_assumptions,
    const uint8_t* __restrict__ world_views,
    uint32_t* __restrict__ rejection_reasons
) {
    uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) return;

    uint32_t reason = 0u;
    if (world_views[candidate * world_stride] == 0u) {
        reason = 2u;
    }

    uint32_t base = candidate * literal_count;
    for (uint32_t literal = 0; literal < literal_count; ++literal) {
        uint8_t value = candidate_assumptions[base + literal];
        if (value > 1u) {
            reason = 3u;
        }
    }

    rejection_reasons[candidate] = reason;
}

extern "C" __global__ void epistemic_populate_model_membership_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t reduction_count,
    uint32_t models_per_reduction,
    uint32_t world_stride,
    const uint32_t* __restrict__ output_row_count,
    const uint8_t* __restrict__ candidate_assumptions,
    const uint8_t* __restrict__ world_views,
    uint8_t* __restrict__ model_membership,
    uint32_t* __restrict__ rejection_reasons
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t per_candidate = reduction_count * models_per_reduction * literal_count;
    uint32_t total = candidate_count * per_candidate;
    if (gid >= total) return;

    uint32_t candidate = gid / per_candidate;
    uint32_t inner = gid - candidate * per_candidate;
    uint32_t literal = inner % literal_count;
    uint32_t model_slot = inner / literal_count;

    uint8_t active_world = world_views[candidate * world_stride];
    uint8_t accepted_so_far = (rejection_reasons[candidate] == 0u) ? 1u : 0u;
    uint8_t has_reduced_output = (output_row_count[0] > 0u) ? 1u : 0u;
    uint8_t candidate_bit = candidate_assumptions[candidate * literal_count + literal];
    model_membership[gid] =
        (active_world != 0u && accepted_so_far != 0u && has_reduced_output != 0u && model_slot < reduction_count * models_per_reduction)
            ? candidate_bit
            : 0u;

    if (inner == 0u && active_world == 0u && rejection_reasons[candidate] == 0u) {
        rejection_reasons[candidate] = 4u;
    }
}

extern "C" __global__ void epistemic_validate_world_views_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t reduction_count,
    uint32_t models_per_reduction,
    uint32_t world_stride,
    const uint8_t* __restrict__ model_membership,
    const uint8_t* __restrict__ world_views,
    uint32_t* __restrict__ rejection_reasons
) {
    uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) return;

    if (rejection_reasons[candidate] != 0u) return;

    uint8_t active_world = world_views[candidate * world_stride];
    if (active_world == 0u) {
        rejection_reasons[candidate] = 5u;
        return;
    }

    uint32_t per_candidate = reduction_count * models_per_reduction * literal_count;
    uint32_t base = candidate * per_candidate;
    uint8_t observed_membership = 0u;
    for (uint32_t offset = 0; offset < per_candidate; ++offset) {
        observed_membership |= model_membership[base + offset];
    }

    if (observed_membership == 0u) {
        rejection_reasons[candidate] = 5u;
    }
}

extern "C" __global__ void epistemic_materialize_accepted_candidates_u8(
    uint32_t candidate_count,
    uint32_t world_stride,
    const uint32_t* __restrict__ rejection_reasons,
    uint8_t* __restrict__ world_views
) {
    uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) return;

    world_views[candidate * world_stride] =
        (rejection_reasons[candidate] == 0u) ? 1u : 0u;
}

extern "C" __global__ void epistemic_materialize_final_result_flags_u8(
    uint32_t candidate_count,
    uint32_t world_stride,
    const uint32_t* __restrict__ output_row_count,
    const uint32_t* __restrict__ rejection_reasons,
    uint8_t* __restrict__ world_views
) {
    uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) return;

    uint8_t has_output = (output_row_count[0] > 0u) ? 1u : 0u;
    world_views[candidate * world_stride] =
        (rejection_reasons[candidate] == 0u && has_output != 0u) ? 1u : 0u;
}
