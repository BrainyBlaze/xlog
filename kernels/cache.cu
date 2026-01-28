// GPU cache helpers for CNF hashing and circuit cache management.

#include <cstdint>

// FNV-1a 64-bit constants
#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME  0x100000001b3ULL

__device__ __forceinline__ uint64_t fnv_mix_u32(uint64_t hash, uint32_t value) {
    hash ^= (uint64_t)value;
    hash *= FNV_PRIME;
    return hash;
}

// Compute a deterministic hash for a device-resident CNF.
// Hash input order: num_vars, num_clauses, num_lits, clause_offsets[0..num_clauses], literals[0..num_lits-1].
extern "C" __global__ void cache_cnf_hash(
    const uint32_t* __restrict__ num_vars,
    const uint32_t* __restrict__ num_clauses,
    const uint32_t* __restrict__ num_lits,
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ literals,
    uint64_t* __restrict__ out_hash
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    uint32_t vars = num_vars[0];
    uint32_t clauses = num_clauses[0];
    uint32_t lits = num_lits[0];

    uint64_t h = FNV_OFFSET;
    h = fnv_mix_u32(h, vars);
    h = fnv_mix_u32(h, clauses);
    h = fnv_mix_u32(h, lits);

    uint32_t offsets_len = clauses + 1;
    for (uint32_t i = 0; i < offsets_len; i++) {
        h = fnv_mix_u32(h, clause_offsets[i]);
    }

    for (uint32_t i = 0; i < lits; i++) {
        uint32_t v = static_cast<uint32_t>(literals[i]);
        h = fnv_mix_u32(h, v);
    }

    out_hash[0] = h;
}
