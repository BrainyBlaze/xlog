// XLOG GPU PIR interning kernels
// Key packing, hashing, uniqueness, and interning helpers for GPU PIR nodes.

#include <cstdint>

#define PIR_CONST    0
#define PIR_LIT      1
#define PIR_NEG_LIT  2
#define PIR_AND      3
#define PIR_OR       4
#define PIR_DECISION 5

// FNV-1a 64-bit constants
#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME  0x100000001b3ULL

__device__ __forceinline__ void pir_trap() {
    asm volatile("trap;");
}

__device__ __forceinline__ uint64_t fnv_mix_u32(uint64_t hash, uint32_t value) {
    hash ^= (uint64_t)value;
    hash *= FNV_PRIME;
    return hash;
}

// Pack PIR node keys into sortable arrays.
extern "C" __global__ void pir_pack_keys(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ leaf_id,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ child_offsets,
    uint32_t num_nodes,
    uint32_t* __restrict__ key_tag,
    uint32_t* __restrict__ key_payload,
    uint32_t* __restrict__ key_child_len
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    uint8_t tag = node_type[idx];
    uint32_t payload = 0;
    uint32_t len = 0;

    if (tag == PIR_CONST) {
        payload = leaf_id[idx];
    } else if (tag == PIR_LIT || tag == PIR_NEG_LIT) {
        payload = leaf_id[idx];
    } else if (tag == PIR_DECISION) {
        payload = decision_var[idx];
        (void)decision_child_false;
        (void)decision_child_true;
        len = 2;
    } else if (tag == PIR_AND || tag == PIR_OR) {
        uint32_t start = child_offsets[idx];
        uint32_t end = child_offsets[idx + 1];
        len = end - start;
    } else {
        payload = 0;
        len = 0;
    }

    key_tag[idx] = (uint32_t)tag;
    key_payload[idx] = payload;
    key_child_len[idx] = len;
}

// Compute deterministic hash for PIR keys (tag + payload + children or decision children).
extern "C" __global__ void pir_hash_keys(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ leaf_id,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ children,
    uint32_t num_nodes,
    uint64_t* __restrict__ hashes
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    uint8_t tag = node_type[idx];
    uint32_t payload = 0;
    uint32_t len = 0;

    if (tag == PIR_CONST) {
        payload = leaf_id[idx];
    } else if (tag == PIR_LIT || tag == PIR_NEG_LIT) {
        payload = leaf_id[idx];
    } else if (tag == PIR_DECISION) {
        payload = decision_var[idx];
        len = 2;
    } else if (tag == PIR_AND || tag == PIR_OR) {
        uint32_t start = child_offsets[idx];
        uint32_t end = child_offsets[idx + 1];
        len = end - start;
    }

    uint64_t hash = FNV_OFFSET;
    hash = fnv_mix_u32(hash, (uint32_t)tag);
    hash = fnv_mix_u32(hash, payload);
    hash = fnv_mix_u32(hash, len);

    if (tag == PIR_DECISION) {
        hash = fnv_mix_u32(hash, decision_child_false[idx]);
        hash = fnv_mix_u32(hash, decision_child_true[idx]);
    } else if (tag == PIR_AND || tag == PIR_OR) {
        uint32_t start = child_offsets[idx];
        uint32_t end = child_offsets[idx + 1];
        for (uint32_t i = start; i < end; i++) {
            hash = fnv_mix_u32(hash, children[i]);
        }
    }

    hashes[idx] = hash;
}

// Mark unique PIR nodes after sorting by (hash, tag, payload, child_len).
extern "C" __global__ void pir_mark_unique(
    const uint64_t* __restrict__ hashes,
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ leaf_id,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
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

    if (tag_b == PIR_CONST) {
        if (leaf_id[idx] != leaf_id[idx - 1]) {
            unique_mask[idx] = 1;
            return;
        }
    } else if (tag_b == PIR_LIT || tag_b == PIR_NEG_LIT) {
        if (leaf_id[idx] != leaf_id[idx - 1]) {
            unique_mask[idx] = 1;
            return;
        }
    } else if (tag_b == PIR_DECISION) {
        if (decision_var[idx] != decision_var[idx - 1]) {
            unique_mask[idx] = 1;
            return;
        }
        if (decision_child_false[idx] != decision_child_false[idx - 1]) {
            unique_mask[idx] = 1;
            return;
        }
        if (decision_child_true[idx] != decision_child_true[idx - 1]) {
            unique_mask[idx] = 1;
            return;
        }
    } else if (tag_b == PIR_AND || tag_b == PIR_OR) {
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
    }

    unique_mask[idx] = 0;
}

// Fill parent ids for each child entry using CSR offsets.
extern "C" __global__ void pir_fill_child_parents(
    const uint32_t* __restrict__ child_offsets,
    uint32_t num_nodes,
    uint32_t* __restrict__ parent_ids
) {
    uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;
    uint32_t start = child_offsets[node];
    uint32_t end = child_offsets[node + 1];
    for (uint32_t i = start; i < end; i++) {
        parent_ids[i] = node;
    }
}

// Mark unique (parent, child) pairs after sorting.
extern "C" __global__ void pir_mark_unique_pairs(
    const uint32_t* __restrict__ parent,
    const uint32_t* __restrict__ child,
    uint32_t num_pairs,
    uint8_t* __restrict__ unique_mask
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    if (idx == 0) {
        unique_mask[0] = 1;
        return;
    }

    if (parent[idx] == parent[idx - 1] && child[idx] == child[idx - 1]) {
        unique_mask[idx] = 0;
    } else {
        unique_mask[idx] = 1;
    }
}

// Compact unique (parent, child) pairs.
extern "C" __global__ void pir_compact_pairs(
    const uint32_t* __restrict__ parent_in,
    const uint32_t* __restrict__ child_in,
    const uint8_t* __restrict__ unique_mask,
    const uint32_t* __restrict__ prefix_sum,
    uint32_t num_pairs,
    uint32_t* __restrict__ parent_out,
    uint32_t* __restrict__ child_out
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    if (unique_mask[idx]) {
        uint32_t out_idx = prefix_sum[idx];
        parent_out[out_idx] = parent_in[idx];
        child_out[out_idx] = child_in[idx];
    }
}

// Count children per parent using atomic increments.
extern "C" __global__ void pir_count_children(
    const uint32_t* __restrict__ parent,
    const uint32_t* __restrict__ num_pairs,
    uint32_t num_nodes,
    uint32_t* __restrict__ counts
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = num_pairs[0];
    if (idx >= total) return;

    uint32_t p = parent[idx];
    if (p >= num_nodes) {
        pir_trap();
    }
    atomicAdd(&counts[p], 1);
}

// Write child offsets from prefix counts and total children.
extern "C" __global__ void pir_write_child_offsets(
    const uint32_t* __restrict__ prefix_counts,
    uint32_t num_nodes,
    const uint32_t* __restrict__ total_children,
    uint32_t* __restrict__ child_offsets
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nodes) {
        child_offsets[idx] = prefix_counts[idx];
    }
    if (idx == 0) {
        child_offsets[num_nodes] = total_children[0];
    }
}

// Gather children lists into sorted order.
extern "C" __global__ void pir_gather_children(
    const uint32_t* __restrict__ indices,
    const uint32_t* __restrict__ child_offsets_in,
    const uint32_t* __restrict__ children_in,
    const uint32_t* __restrict__ child_offsets_out,
    uint32_t num_nodes,
    uint32_t* __restrict__ children_out
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    uint32_t orig = indices[idx];
    uint32_t start_in = child_offsets_in[orig];
    uint32_t end_in = child_offsets_in[orig + 1];
    uint32_t start_out = child_offsets_out[idx];
    uint32_t len = end_in - start_in;

    for (uint32_t i = 0; i < len; i++) {
        children_out[start_out + i] = children_in[start_in + i];
    }
}

// Build child counts for AND/OR nodes that will be inserted.
extern "C" __global__ void pir_build_graph_child_counts(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint8_t* __restrict__ new_mask,
    uint32_t num_nodes,
    uint32_t* __restrict__ counts
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    uint32_t count = 0;
    if (new_mask[idx]) {
        uint8_t tag = node_type[idx];
        if (tag == PIR_AND || tag == PIR_OR) {
            count = child_offsets[idx + 1] - child_offsets[idx];
        }
    }
    counts[idx] = count;
}

// Sum u32 counts into a single total using atomic adds.
extern "C" __global__ void pir_sum_counts(
    const uint32_t* __restrict__ counts,
    uint32_t num_nodes,
    uint32_t* __restrict__ total
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    uint32_t v = counts[idx];
    if (v != 0) {
        atomicAdd(total, v);
    }
}

// Find existing graph node ids for sorted batch nodes using a bucketed hash table.
extern "C" __global__ void pir_find_existing(
    const uint64_t* __restrict__ hashes,
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ leaf_id,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ children,
    uint32_t num_nodes,
    const uint8_t* __restrict__ graph_node_type,
    const uint32_t* __restrict__ graph_child_offsets,
    const uint32_t* __restrict__ graph_children,
    const uint32_t* __restrict__ graph_leaf_id,
    const uint32_t* __restrict__ graph_decision_var,
    const uint32_t* __restrict__ graph_decision_child_false,
    const uint32_t* __restrict__ graph_decision_child_true,
    const uint32_t* __restrict__ graph_num_nodes,
    const uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_counts,
    const uint32_t* __restrict__ bucket_entries,
    const uint64_t* __restrict__ bucket_entry_hashes,
    uint32_t bucket_mask,
    uint32_t* __restrict__ existing_id
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    uint64_t h = hashes[idx];
    uint32_t h32 = (uint32_t)(h ^ (h >> 32));
    uint32_t bucket = h32 & bucket_mask;
    uint32_t start = bucket_offsets[bucket];
    uint32_t count = bucket_counts[bucket];
    uint32_t limit = graph_num_nodes[0];

    uint8_t tag = node_type[idx];
    uint32_t child_start = child_offsets[idx];
    uint32_t child_end = child_offsets[idx + 1];
    uint32_t child_len = child_end - child_start;

    uint32_t found = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t entry = bucket_entries[start + i];
        if (entry >= limit) {
            continue;
        }
        if (bucket_entry_hashes[start + i] != h) {
            continue;
        }
        if (graph_node_type[entry] != tag) {
            continue;
        }

        if (tag == PIR_CONST || tag == PIR_LIT || tag == PIR_NEG_LIT) {
            if (graph_leaf_id[entry] == leaf_id[idx]) {
                found = entry;
                break;
            }
            continue;
        }

        if (tag == PIR_DECISION) {
            if (graph_decision_var[entry] != decision_var[idx]) {
                continue;
            }
            if (graph_decision_child_false[entry] != decision_child_false[idx]) {
                continue;
            }
            if (graph_decision_child_true[entry] != decision_child_true[idx]) {
                continue;
            }
            found = entry;
            break;
        }

        if (tag == PIR_AND || tag == PIR_OR) {
            uint32_t g_start = graph_child_offsets[entry];
            uint32_t g_end = graph_child_offsets[entry + 1];
            uint32_t g_len = g_end - g_start;
            if (g_len != child_len) {
                continue;
            }
            bool match = true;
            for (uint32_t j = 0; j < child_len; j++) {
                if (graph_children[g_start + j] != children[child_start + j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                found = entry;
                break;
            }
        }
    }

    existing_id[idx] = found;
}

// Mark which unique groups correspond to newly inserted nodes.
extern "C" __global__ void pir_mark_new_groups(
    const uint8_t* __restrict__ unique_mask,
    const uint32_t* __restrict__ existing_id,
    uint32_t num_nodes,
    uint8_t* __restrict__ new_mask
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    uint8_t uniq = unique_mask[idx];
    uint32_t exists = existing_id[idx];
    new_mask[idx] = (uniq && exists == 0xFFFFFFFFu) ? 1 : 0;
}

// Build group id mapping for sorted unique groups.
extern "C" __global__ void pir_build_group_ids(
    const uint8_t* __restrict__ unique_mask,
    const uint32_t* __restrict__ unique_prefix,
    const uint32_t* __restrict__ existing_id,
    const uint32_t* __restrict__ new_prefix,
    const uint32_t* __restrict__ base_nodes,
    uint32_t num_nodes,
    uint32_t* __restrict__ group_node_id
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    if (!unique_mask[idx]) return;

    uint32_t group_id = unique_prefix[idx] + (uint32_t)unique_mask[idx] - 1u;
    uint32_t existing = existing_id[idx];
    uint32_t node_id = (existing != 0xFFFFFFFFu) ? existing : (base_nodes[0] + new_prefix[idx]);
    group_node_id[group_id] = node_id;
}

// Emit node ids back to batch order and append unique nodes to graph arrays.
extern "C" __global__ void pir_emit_nodes_and_ids(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ leaf_id,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ children,
    const uint8_t* __restrict__ unique_mask,
    const uint32_t* __restrict__ unique_prefix,
    const uint32_t* __restrict__ group_node_id,
    const uint32_t* __restrict__ graph_child_prefix,
    const uint32_t* __restrict__ indices,
    uint32_t num_nodes,
    const uint32_t* __restrict__ base_nodes,
    const uint32_t* __restrict__ base_children,
    uint32_t node_cap,
    uint32_t child_cap,
    uint8_t* __restrict__ graph_node_type,
    uint32_t* __restrict__ graph_child_offsets,
    uint32_t* __restrict__ graph_children,
    uint32_t* __restrict__ graph_leaf_id,
    uint32_t* __restrict__ graph_decision_var,
    uint32_t* __restrict__ graph_decision_child_false,
    uint32_t* __restrict__ graph_decision_child_true,
    const uint8_t* __restrict__ new_mask,
    const uint64_t* __restrict__ sorted_hashes,
    uint64_t* __restrict__ graph_hashes,
    uint32_t* __restrict__ out_ids
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    uint32_t base = base_nodes[0];
    uint32_t base_child = base_children[0];

    uint32_t group_id = unique_prefix[idx] + (uint32_t)unique_mask[idx] - 1u;
    uint32_t node_id = group_node_id[group_id];

    out_ids[indices[idx]] = node_id;

    if (!new_mask[idx]) return;

    if (node_id >= node_cap) {
        pir_trap();
    }

    uint8_t tag = node_type[idx];
    graph_node_type[node_id] = tag;
    graph_leaf_id[node_id] = 0;
    graph_decision_var[node_id] = 0;
    graph_decision_child_false[node_id] = 0;
    graph_decision_child_true[node_id] = 0;
    graph_hashes[node_id] = sorted_hashes[idx];

    uint32_t child_start = base_child + graph_child_prefix[idx];

    if (tag == PIR_AND || tag == PIR_OR) {
        uint32_t start_in = child_offsets[idx];
        uint32_t end_in = child_offsets[idx + 1];
        uint32_t len = end_in - start_in;
        if (child_start + len > child_cap) {
            pir_trap();
        }
        for (uint32_t i = 0; i < len; i++) {
            uint32_t c = children[start_in + i];
            if (c >= base) {
                pir_trap();
            }
            graph_children[child_start + i] = c;
        }
        graph_child_offsets[node_id] = child_start;
        graph_child_offsets[node_id + 1] = child_start + len;
    } else {
        graph_child_offsets[node_id] = child_start;
        graph_child_offsets[node_id + 1] = child_start;
        if (tag == PIR_LIT || tag == PIR_NEG_LIT) {
            graph_leaf_id[node_id] = leaf_id[idx];
        } else if (tag == PIR_DECISION) {
            uint32_t f = decision_child_false[idx];
            uint32_t t = decision_child_true[idx];
            if (f >= base || t >= base) {
                pir_trap();
            }
            graph_decision_var[node_id] = decision_var[idx];
            graph_decision_child_false[node_id] = f;
            graph_decision_child_true[node_id] = t;
        } else if (tag == PIR_CONST) {
            pir_trap();
        }
    }
}

// Update global node/child counters with bounds checks.
extern "C" __global__ void pir_update_counts(
    const uint32_t* __restrict__ add_nodes,
    const uint32_t* __restrict__ add_children,
    uint32_t node_cap,
    uint32_t child_cap,
    uint32_t* __restrict__ num_nodes,
    uint32_t* __restrict__ num_children
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint32_t base_nodes = num_nodes[0];
    uint32_t base_children = num_children[0];
    uint32_t new_nodes = base_nodes + add_nodes[0];
    uint32_t new_children = base_children + add_children[0];

    if (new_nodes > node_cap || new_children > child_cap) {
        pir_trap();
    }

    num_nodes[0] = new_nodes;
    num_children[0] = new_children;
}
