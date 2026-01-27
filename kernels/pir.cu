// XLOG GPU PIR interning kernels
// Provides key packing, hashing, and uniqueness marking for GPU PIR nodes.

#include <cstdint>

// FNV-1a 64-bit constants
#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME  0x100000001b3ULL

__device__ __forceinline__ uint64_t fnv_mix_u32(uint64_t hash, uint32_t value) {
    hash ^= (uint64_t)value;
    hash *= FNV_PRIME;
    return hash;
}

// Pack PIR node keys into sortable arrays.
//
// Inputs:
// - node_type: PIR tag per node (u8)
// - payload: per-node payload (u32):
//     * CONST: 0/1 for false/true
//     * LIT/NEG_LIT: leaf_id
//     * DECISION: choice var id
//     * AND/OR: 0
// - child_offsets: CSR offsets into children array (len = num_nodes + 1)
//
// Outputs:
// - key_tag: u32 copy of node_type
// - key_payload: u32 copy of payload
// - key_child_len: number of children for each node
extern "C" __global__ void pir_pack_keys(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ payload,
    const uint32_t* __restrict__ child_offsets,
    uint32_t num_nodes,
    uint32_t* __restrict__ key_tag,
    uint32_t* __restrict__ key_payload,
    uint32_t* __restrict__ key_child_len
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    uint32_t start = child_offsets[idx];
    uint32_t end = child_offsets[idx + 1];
    uint32_t len = end - start;

    key_tag[idx] = (uint32_t)node_type[idx];
    key_payload[idx] = payload[idx];
    key_child_len[idx] = len;
}

// Compute deterministic hash for PIR keys (tag + payload + children).
//
// Inputs: node_type, payload, child_offsets, children
// Output: hashes (u64)
extern "C" __global__ void pir_hash_keys(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ payload,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ children,
    uint32_t num_nodes,
    uint64_t* __restrict__ hashes
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    uint32_t start = child_offsets[idx];
    uint32_t end = child_offsets[idx + 1];
    uint32_t len = end - start;

    uint64_t hash = FNV_OFFSET;
    hash = fnv_mix_u32(hash, (uint32_t)node_type[idx]);
    hash = fnv_mix_u32(hash, payload[idx]);
    hash = fnv_mix_u32(hash, len);

    for (uint32_t i = start; i < end; i++) {
        hash = fnv_mix_u32(hash, children[i]);
    }

    hashes[idx] = hash;
}

// Mark unique PIR nodes after sorting by (hash, tag, payload, child_len, children).
//
// unique_mask[i] = 1 if node i is unique relative to i-1, else 0.
extern "C" __global__ void pir_mark_unique(
    const uint64_t* __restrict__ hashes,
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ payload,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ children,
    uint32_t num_nodes,
    uint8_t* __restrict__ unique_mask
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    if (idx == 0) {
        unique_mask[0] = 1;
        return;
    }

    if (hashes[idx] != hashes[idx - 1]) {
        unique_mask[idx] = 1;
        return;
    }

    uint8_t tag_a = node_type[idx - 1];
    uint8_t tag_b = node_type[idx];
    if (tag_a != tag_b) {
        unique_mask[idx] = 1;
        return;
    }

    uint32_t payload_a = payload[idx - 1];
    uint32_t payload_b = payload[idx];
    if (payload_a != payload_b) {
        unique_mask[idx] = 1;
        return;
    }

    uint32_t start_a = child_offsets[idx - 1];
    uint32_t end_a = child_offsets[idx];
    uint32_t start_b = child_offsets[idx];
    uint32_t end_b = child_offsets[idx + 1];

    uint32_t len_a = end_a - start_a;
    uint32_t len_b = end_b - start_b;
    if (len_a != len_b) {
        unique_mask[idx] = 1;
        return;
    }

    for (uint32_t i = 0; i < len_a; i++) {
        if (children[start_a + i] != children[start_b + i]) {
            unique_mask[idx] = 1;
            return;
        }
    }

    unique_mask[idx] = 0;
}
