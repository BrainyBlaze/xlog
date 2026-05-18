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
