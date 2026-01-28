// GPU cache helpers for CNF hashing and circuit cache management.

#include <cstdint>
#include <cstddef>

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

__device__ __forceinline__ uint32_t cache_find_free_slot(
    uint32_t num_slots,
    uint32_t* __restrict__ slot_states
) {
    for (uint32_t i = 0; i < num_slots; i++) {
        if (slot_states[i] == 0) {
            slot_states[i] = 1;
            return i;
        }
    }
    return 0xffffffffu;
}

__device__ __forceinline__ int cache_evict_lru_entry(
    uint32_t table_size,
    uint32_t num_slots,
    uint64_t* __restrict__ keys,
    uint32_t* __restrict__ slots,
    uint32_t* __restrict__ state,
    uint64_t* __restrict__ last_used,
    uint32_t* __restrict__ slot_states,
    uint32_t* __restrict__ out_slot
) {
    uint64_t best_time = 0xffffffffffffffffULL;
    int best_idx = -1;
    for (uint32_t i = 0; i < table_size; i++) {
        if (state[i] == 0) {
            continue;
        }
        uint64_t t = last_used[i];
        if (t < best_time) {
            best_time = t;
            best_idx = static_cast<int>(i);
        }
    }
    if (best_idx < 0) {
        *out_slot = 0xffffffffu;
        return -1;
    }

    uint32_t slot = slots[best_idx];
    state[best_idx] = 0;
    keys[best_idx] = 0;
    last_used[best_idx] = 0;
    if (slot < num_slots) {
        slot_states[slot] = 0;
    }
    *out_slot = slot;
    return best_idx;
}

// Lookup or insert a cache key. Outputs: selected slot id and compile_needed (0=hit, 1=miss+insert).
extern "C" __global__ void cache_lookup_or_insert(
    uint64_t key,
    uint32_t table_size,
    uint32_t num_slots,
    uint64_t* __restrict__ keys,
    uint32_t* __restrict__ slots,
    uint32_t* __restrict__ state,
    uint64_t* __restrict__ last_used,
    uint32_t* __restrict__ slot_states,
    uint64_t* __restrict__ clock,
    uint32_t* __restrict__ out_slot,
    uint32_t* __restrict__ out_compile_needed
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    if (table_size == 0 || num_slots == 0) {
        out_slot[0] = 0xffffffffu;
        out_compile_needed[0] = 0xffffffffu;
        return;
    }

    uint64_t now = atomicAdd(reinterpret_cast<unsigned long long*>(clock), 1ULL) + 1ULL;
    uint32_t start = static_cast<uint32_t>(key % table_size);

    for (uint32_t probe = 0; probe < table_size; probe++) {
        uint32_t idx = (start + probe);
        if (idx >= table_size) {
            idx -= table_size;
        }

        if (state[idx] == 0) {
            uint32_t slot = cache_find_free_slot(num_slots, slot_states);
            if (slot == 0xffffffffu) {
                uint32_t evict_slot = 0xffffffffu;
                int evict_idx = cache_evict_lru_entry(
                    table_size,
                    num_slots,
                    keys,
                    slots,
                    state,
                    last_used,
                    slot_states,
                    &evict_slot
                );
                if (evict_idx < 0 || evict_slot == 0xffffffffu) {
                    out_slot[0] = 0xffffffffu;
                    out_compile_needed[0] = 0xffffffffu;
                    return;
                }
                slot = evict_slot;
            }

            keys[idx] = key;
            slots[idx] = slot;
            state[idx] = 1;
            last_used[idx] = now;
            out_slot[0] = slot;
            out_compile_needed[0] = 1;
            return;
        }

        if (keys[idx] == key) {
            last_used[idx] = now;
            out_slot[0] = slots[idx];
            out_compile_needed[0] = 0;
            return;
        }
    }

    uint32_t evict_slot = 0xffffffffu;
    int evict_idx = cache_evict_lru_entry(
        table_size,
        num_slots,
        keys,
        slots,
        state,
        last_used,
        slot_states,
        &evict_slot
    );
    if (evict_idx < 0 || evict_slot == 0xffffffffu) {
        out_slot[0] = 0xffffffffu;
        out_compile_needed[0] = 0xffffffffu;
        return;
    }

    keys[evict_idx] = key;
    slots[evict_idx] = evict_slot;
    state[evict_idx] = 1;
    last_used[evict_idx] = now;
    out_slot[0] = evict_slot;
    out_compile_needed[0] = 1;
}

// Evict the least-recently-used entry and return its slot id.
extern "C" __global__ void cache_evict_lru(
    uint32_t table_size,
    uint32_t num_slots,
    uint64_t* __restrict__ keys,
    uint32_t* __restrict__ slots,
    uint32_t* __restrict__ state,
    uint64_t* __restrict__ last_used,
    uint32_t* __restrict__ slot_states,
    uint32_t* __restrict__ out_slot,
    uint32_t* __restrict__ out_evicted
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    uint32_t evict_slot = 0xffffffffu;
    int idx = cache_evict_lru_entry(
        table_size,
        num_slots,
        keys,
        slots,
        state,
        last_used,
        slot_states,
        &evict_slot
    );
    if (idx < 0) {
        out_slot[0] = 0xffffffffu;
        out_evicted[0] = 0;
        return;
    }
    out_slot[0] = evict_slot;
    out_evicted[0] = 1;
}

__device__ __forceinline__ size_t slot_offset(uint32_t slot, uint32_t stride) {
    return static_cast<size_t>(slot) * static_cast<size_t>(stride);
}

extern "C" __global__ void cache_store_u8(
    const uint32_t* __restrict__ active_slot,
    const uint32_t* __restrict__ compile_needed,
    uint32_t stride,
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    uint32_t count
) {
    if (compile_needed[0] == 0u) {
        return;
    }
    uint32_t slot = active_slot[0];
    size_t base = slot_offset(slot, stride);
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
        dst[base + idx] = src[idx];
    }
}

extern "C" __global__ void cache_store_u32(
    const uint32_t* __restrict__ active_slot,
    const uint32_t* __restrict__ compile_needed,
    uint32_t stride,
    const uint32_t* __restrict__ src,
    uint32_t* __restrict__ dst,
    uint32_t count
) {
    if (compile_needed[0] == 0u) {
        return;
    }
    uint32_t slot = active_slot[0];
    size_t base = slot_offset(slot, stride);
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
        dst[base + idx] = src[idx];
    }
}

extern "C" __global__ void cache_store_i32(
    const uint32_t* __restrict__ active_slot,
    const uint32_t* __restrict__ compile_needed,
    uint32_t stride,
    const int32_t* __restrict__ src,
    int32_t* __restrict__ dst,
    uint32_t count
) {
    if (compile_needed[0] == 0u) {
        return;
    }
    uint32_t slot = active_slot[0];
    size_t base = slot_offset(slot, stride);
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
        dst[base + idx] = src[idx];
    }
}

extern "C" __global__ void cache_store_f64(
    const uint32_t* __restrict__ active_slot,
    const uint32_t* __restrict__ compile_needed,
    uint32_t stride,
    const double* __restrict__ src,
    double* __restrict__ dst,
    uint32_t count
) {
    if (compile_needed[0] == 0u) {
        return;
    }
    uint32_t slot = active_slot[0];
    size_t base = slot_offset(slot, stride);
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
        dst[base + idx] = src[idx];
    }
}

extern "C" __global__ void cache_store_meta(
    const uint32_t* __restrict__ active_slot,
    const uint32_t* __restrict__ compile_needed,
    uint32_t num_slots,
    uint32_t num_nodes,
    uint32_t num_levels,
    uint32_t root,
    uint32_t max_var,
    uint32_t* __restrict__ meta_num_nodes,
    uint32_t* __restrict__ meta_num_levels,
    uint32_t* __restrict__ meta_root,
    uint32_t* __restrict__ meta_max_var
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (compile_needed[0] == 0u) {
        return;
    }
    uint32_t slot = active_slot[0];
    if (slot >= num_slots) {
        return;
    }
    meta_num_nodes[slot] = num_nodes;
    meta_num_levels[slot] = num_levels;
    meta_root[slot] = root;
    meta_max_var[slot] = max_var;
}
