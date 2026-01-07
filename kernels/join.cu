// XLOG GPU Join Kernels
// Hash join with linked-list collision handling

#include <cstdint>

// Hash function for join keys
__device__ __forceinline__ uint32_t hash_key(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

// Build phase: insert keys into hash table with linked lists
extern "C" __global__ void hash_join_build(
    const uint32_t* __restrict__ keys,
    const uint32_t* __restrict__ payloads,
    uint32_t num_rows,
    uint32_t* __restrict__ hash_table,
    uint32_t* __restrict__ next_ptrs,
    uint32_t hash_table_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;

    uint32_t key = keys[tid];
    uint32_t payload = payloads[tid];
    uint32_t hash = hash_key(key) % hash_table_size;

    // Atomic linked list insertion
    uint32_t old = atomicExch(&hash_table[hash * 3 + 2], tid);
    next_ptrs[tid] = old;
    hash_table[hash * 3] = key;
    hash_table[hash * 3 + 1] = payload;
}

// Probe phase: find matches and output join results
extern "C" __global__ void hash_join_probe(
    const uint32_t* __restrict__ probe_keys,
    const uint32_t* __restrict__ probe_payloads,
    uint32_t num_probe_rows,
    const uint32_t* __restrict__ hash_table,
    const uint32_t* __restrict__ build_keys,
    const uint32_t* __restrict__ build_payloads,
    const uint32_t* __restrict__ next_ptrs,
    uint32_t hash_table_size,
    uint32_t* __restrict__ output_left,
    uint32_t* __restrict__ output_right,
    uint32_t* __restrict__ output_count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_probe_rows) return;

    uint32_t key = probe_keys[tid];
    uint32_t hash = hash_key(key) % hash_table_size;

    // Walk the linked list
    uint32_t current = hash_table[hash * 3 + 2];
    while (current != 0xFFFFFFFF) {
        if (build_keys[current] == key) {
            uint32_t out_idx = atomicAdd(output_count, 1);
            output_left[out_idx] = probe_payloads[tid];
            output_right[out_idx] = build_payloads[current];
        }
        current = next_ptrs[current];
    }
}
