// XLOG GPU Join Kernels
// Hash join with linked-list collision handling

#include <cstdint>

/**
 * Multi-Column Join Kernels (v2)
 *
 * Uses composite hashing for multi-column keys.
 * All scalar types supported by hashing bytes directly.
 */

// FNV-1a hash constants
#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME 0x100000001b3ULL

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

/**
 * Compute composite hash for multi-column keys.
 * Hashes all key columns together using FNV-1a.
 * @param data Row-major packed data
 * @param col_offsets Byte offset of each key column within a row
 * @param col_sizes Size in bytes of each key column
 * @param num_key_cols Number of key columns
 * @param num_rows Number of rows
 * @param row_stride Total bytes per row
 * @param hashes Output: one u64 hash per row
 */
extern "C" __global__ void compute_composite_hash(
    const uint8_t* __restrict__ data,
    const uint32_t* __restrict__ col_offsets,
    const uint32_t* __restrict__ col_sizes,
    uint32_t num_key_cols,
    uint32_t num_rows,
    uint32_t row_stride,
    uint64_t* __restrict__ hashes
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;

    uint64_t hash = FNV_OFFSET;
    const uint8_t* row = data + (uint64_t)gid * row_stride;

    for (uint32_t c = 0; c < num_key_cols; c++) {
        uint32_t offset = col_offsets[c];
        uint32_t size = col_sizes[c];

        for (uint32_t b = 0; b < size; b++) {
            hash ^= row[offset + b];
            hash *= FNV_PRIME;
        }
    }

    hashes[gid] = hash;
}

/**
 * Build hash table with u64 hashes (v2).
 *
 * This is a coalesced, cache-friendly bucket layout:
 * - `bucket_counts[b]` is the number of build rows in bucket b
 * - `bucket_offsets[b]` (computed on host via scan) is the start index of bucket b in `bucket_entries`
 * - `bucket_entries` stores build row indices contiguously per bucket
 *
 * This replaces the linked-list chaining used by earlier versions.
 */
extern "C" __global__ void hash_join_bucket_count_v2(
    const uint64_t* __restrict__ hashes,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t* __restrict__ bucket_counts,
    uint32_t bucket_mask
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= row_cap) return;
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) actual = row_cap;
    if (tid >= actual) return;

    uint64_t hash = hashes[tid];
    uint32_t h32 = (uint32_t)(hash ^ (hash >> 32));
    uint32_t bucket = h32 & bucket_mask;
    atomicAdd(&bucket_counts[bucket], 1);
}

extern "C" __global__ void hash_join_scatter_v2(
    const uint64_t* __restrict__ hashes,
    const uint32_t* __restrict__ num_rows_device,
    uint32_t row_cap,
    uint32_t* __restrict__ bucket_cursors,
    uint32_t bucket_mask,
    uint32_t* __restrict__ bucket_entries,
    uint64_t* __restrict__ bucket_entry_hashes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= row_cap) return;
    uint32_t actual = num_rows_device[0];
    if (actual > row_cap) actual = row_cap;
    if (tid >= actual) return;

    uint64_t hash = hashes[tid];
    uint32_t h32 = (uint32_t)(hash ^ (hash >> 32));
    uint32_t bucket = h32 & bucket_mask;

    uint32_t pos = atomicAdd(&bucket_cursors[bucket], 1);
    bucket_entries[pos] = tid;
    bucket_entry_hashes[pos] = hash;
}

/**
 * Compare key bytes for two rows.
 * @param probe_keys Packed probe key data (row-major)
 * @param build_keys Packed build key data (row-major)
 * @param probe_idx Probe row index
 * @param build_idx Build row index
 * @param key_bytes Total bytes per key (row stride)
 * @return true if keys are equal byte-for-byte
 */
__device__ __forceinline__ bool keys_equal(
    const uint8_t* __restrict__ probe_keys,
    const uint8_t* __restrict__ build_keys,
    uint32_t probe_idx,
    uint32_t build_idx,
    uint32_t key_bytes
) {
    const uint8_t* probe_row = probe_keys + (uint64_t)probe_idx * key_bytes;
    const uint8_t* build_row = build_keys + (uint64_t)build_idx * key_bytes;

    for (uint32_t i = 0; i < key_bytes; i++) {
        if (probe_row[i] != build_row[i]) {
            return false;
        }
    }
    return true;
}

/**
 * Probe hash table (v2) - outputs matching row index pairs.
 *
 * This kernel first compares hash values for fast filtering, then
 * verifies actual key bytes to ensure mathematical correctness.
 *
 * @param probe_hashes Hash values for probe side
 * @param num_probe Number of probe rows
 * @param bucket_offsets Start offsets per bucket
 * @param bucket_counts Build row counts per bucket
 * @param bucket_entries Build row indices, bucketed contiguously
 * @param bucket_entry_hashes Build hashes aligned with bucket_entries
 * @param probe_keys Packed probe key data (row-major, key columns only)
 * @param build_keys Packed build key data (row-major, key columns only)
 * @param key_bytes Total bytes per key (sum of key column sizes)
 * @param output_left Output: probe row indices
 * @param output_right Output: build row indices
 * @param output_count Atomic counter for output size
 * @param max_output Maximum output size to prevent overflow
 */
extern "C" __global__ void hash_join_probe_v2(
    const uint64_t* __restrict__ probe_hashes,
    uint32_t num_probe,
    const uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_counts,
    const uint32_t* __restrict__ bucket_entries,
    const uint64_t* __restrict__ bucket_entry_hashes,
    uint32_t bucket_mask,
    const uint8_t* __restrict__ probe_keys,
    const uint8_t* __restrict__ build_keys,
    uint32_t key_bytes,
    uint32_t* __restrict__ output_left,
    uint32_t* __restrict__ output_right,
    uint32_t* __restrict__ output_count,
    uint32_t max_output
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_probe) return;

    uint64_t hash = probe_hashes[tid];
    uint32_t h32 = (uint32_t)(hash ^ (hash >> 32));
    uint32_t bucket = h32 & bucket_mask;

    uint32_t start = bucket_offsets[bucket];
    uint32_t count = bucket_counts[bucket];

    for (uint32_t i = 0; i < count; i++) {
        uint32_t pos = start + i;
        if (bucket_entry_hashes[pos] == hash) {
            uint32_t build_idx = bucket_entries[pos];
            if (keys_equal(probe_keys, build_keys, tid, build_idx, key_bytes)) {
                uint32_t out_idx = atomicAdd(output_count, 1);
                if (out_idx < max_output) {
                    output_left[out_idx] = tid;
                    output_right[out_idx] = build_idx;
                }
            }
        }
    }
}

/**
 * Inner-join probe — count-only pass for the deterministic
 * count→prefix-scan→materialize flow. Each thread writes its own slot
 * in `out_per_probe_count[num_probe]`, so there are no global atomics.
 *
 * The follow-up pipeline scans the per-probe array exclusively to
 * produce per-probe write offsets, then `hash_join_probe_v2_materialize`
 * writes to the deterministic offsets without atomic-append.
 */
extern "C" __global__ void hash_join_probe_v2_count_per_row(
    const uint64_t* __restrict__ probe_hashes,
    const uint32_t* __restrict__ num_probe_device,
    uint32_t probe_cap,
    const uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_counts,
    const uint32_t* __restrict__ bucket_entries,
    const uint64_t* __restrict__ bucket_entry_hashes,
    uint32_t bucket_mask,
    const uint8_t* __restrict__ probe_keys,
    const uint8_t* __restrict__ build_keys,
    uint32_t key_bytes,
    uint32_t* __restrict__ out_per_probe_count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= probe_cap) return;

    uint32_t num_probe = *num_probe_device;
    if (num_probe > probe_cap) num_probe = probe_cap;
    if (tid >= num_probe) {
        out_per_probe_count[tid] = 0;
        return;
    }

    uint64_t hash = probe_hashes[tid];
    uint32_t h32 = (uint32_t)(hash ^ (hash >> 32));
    uint32_t bucket = h32 & bucket_mask;

    uint32_t start = bucket_offsets[bucket];
    uint32_t count = bucket_counts[bucket];

    uint32_t matches = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t pos = start + i;
        if (bucket_entry_hashes[pos] == hash) {
            uint32_t build_idx = bucket_entries[pos];
            if (keys_equal(probe_keys, build_keys, tid, build_idx, key_bytes)) {
                matches++;
            }
        }
    }
    out_per_probe_count[tid] = matches;
}

/**
 * Inner-join probe — materialize pass with deterministic per-probe
 * offsets supplied by the caller (typically the output of an exclusive
 * prefix scan over the per-probe count array).
 *
 * Each probe row writes its `local`-th match to
 * `output[per_probe_offsets[tid] + local]` directly. Output order is a
 * deterministic function of probe-row index and per-row match-discovery
 * order; no global atomic determines positions.
 *
 * If a probe row's offset + match would exceed `output_capacity`, the
 * write is suppressed and `d_overflow` is raised. The caller treats
 * the overflow flag as a deferred status (consumed via a sanctioned
 * status-read API) — a non-zero flag means the result is truncated to
 * the first `output_capacity` rows in deterministic order.
 */
extern "C" __global__ void hash_join_probe_v2_materialize(
    const uint64_t* __restrict__ probe_hashes,
    const uint32_t* __restrict__ num_probe_device,
    uint32_t probe_cap,
    const uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_counts,
    const uint32_t* __restrict__ bucket_entries,
    const uint64_t* __restrict__ bucket_entry_hashes,
    uint32_t bucket_mask,
    const uint8_t* __restrict__ probe_keys,
    const uint8_t* __restrict__ build_keys,
    uint32_t key_bytes,
    const uint32_t* __restrict__ per_probe_offsets,
    uint32_t output_capacity,
    uint32_t* __restrict__ output_left,
    uint32_t* __restrict__ output_right,
    uint8_t* __restrict__ d_overflow
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= probe_cap) return;

    uint32_t num_probe = *num_probe_device;
    if (num_probe > probe_cap) num_probe = probe_cap;
    if (tid >= num_probe) return;

    uint64_t hash = probe_hashes[tid];
    uint32_t h32 = (uint32_t)(hash ^ (hash >> 32));
    uint32_t bucket = h32 & bucket_mask;

    uint32_t start = bucket_offsets[bucket];
    uint32_t count = bucket_counts[bucket];

    uint32_t base = per_probe_offsets[tid];
    uint32_t local = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t pos = start + i;
        if (bucket_entry_hashes[pos] == hash) {
            uint32_t build_idx = bucket_entries[pos];
            if (keys_equal(probe_keys, build_keys, tid, build_idx, key_bytes)) {
                uint32_t out_idx = base + local;
                local++;
                if (out_idx < output_capacity) {
                    output_left[out_idx] = tid;
                    output_right[out_idx] = build_idx;
                } else {
                    *d_overflow = 1;
                }
            }
        }
    }
}

/**
 * Left-outer probe — count-per-probe-row pass.
 *
 * For each probe row, write `matches > 0 ? matches : 1` so the
 * subsequent exclusive scan and materialize allocate one output slot
 * per unmatched probe row (the slot will hold the "null right" tuple).
 */
extern "C" __global__ void hash_join_left_outer_count_per_row(
    const uint64_t* __restrict__ probe_hashes,
    const uint32_t* __restrict__ num_probe_device,
    uint32_t probe_cap,
    const uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_counts,
    const uint32_t* __restrict__ bucket_entries,
    const uint64_t* __restrict__ bucket_entry_hashes,
    uint32_t bucket_mask,
    const uint8_t* __restrict__ probe_keys,
    const uint8_t* __restrict__ build_keys,
    uint32_t key_bytes,
    uint32_t* __restrict__ out_per_probe_count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= probe_cap) return;

    uint32_t num_probe = *num_probe_device;
    if (num_probe > probe_cap) num_probe = probe_cap;
    if (tid >= num_probe) {
        out_per_probe_count[tid] = 0;
        return;
    }

    uint64_t hash = probe_hashes[tid];
    uint32_t h32 = (uint32_t)(hash ^ (hash >> 32));
    uint32_t bucket = h32 & bucket_mask;
    uint32_t start = bucket_offsets[bucket];
    uint32_t count = bucket_counts[bucket];

    uint32_t matches = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t pos = start + i;
        if (bucket_entry_hashes[pos] == hash) {
            uint32_t build_idx = bucket_entries[pos];
            if (keys_equal(probe_keys, build_keys, tid, build_idx, key_bytes)) {
                matches++;
            }
        }
    }
    out_per_probe_count[tid] = matches > 0 ? matches : 1;
}

/**
 * Left-outer materialize. For each probe row writes either its matched
 * pairs at the deterministic per-probe offset or a single
 * null-sentinel row when there were no matches.
 *
 * `output_right[i] == 0xFFFFFFFF` is the unmatched sentinel; the
 * right-side gather kernel
 * (`apply_permutation_bytes_left_outer_null_sentinel`) interprets it
 * as "write zero bytes for this row".
 */
extern "C" __global__ void hash_join_left_outer_materialize(
    const uint64_t* __restrict__ probe_hashes,
    const uint32_t* __restrict__ num_probe_device,
    uint32_t probe_cap,
    const uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_counts,
    const uint32_t* __restrict__ bucket_entries,
    const uint64_t* __restrict__ bucket_entry_hashes,
    uint32_t bucket_mask,
    const uint8_t* __restrict__ probe_keys,
    const uint8_t* __restrict__ build_keys,
    uint32_t key_bytes,
    const uint32_t* __restrict__ per_probe_offsets,
    uint32_t output_capacity,
    uint32_t* __restrict__ output_left,
    uint32_t* __restrict__ output_right,
    uint8_t* __restrict__ d_overflow
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= probe_cap) return;

    uint32_t num_probe = *num_probe_device;
    if (num_probe > probe_cap) num_probe = probe_cap;
    if (tid >= num_probe) return;

    uint64_t hash = probe_hashes[tid];
    uint32_t h32 = (uint32_t)(hash ^ (hash >> 32));
    uint32_t bucket = h32 & bucket_mask;
    uint32_t start = bucket_offsets[bucket];
    uint32_t count = bucket_counts[bucket];

    uint32_t base = per_probe_offsets[tid];
    uint32_t local = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t pos = start + i;
        if (bucket_entry_hashes[pos] == hash) {
            uint32_t build_idx = bucket_entries[pos];
            if (keys_equal(probe_keys, build_keys, tid, build_idx, key_bytes)) {
                uint32_t out_idx = base + local;
                local++;
                if (out_idx < output_capacity) {
                    output_left[out_idx] = tid;
                    output_right[out_idx] = build_idx;
                } else {
                    *d_overflow = 1;
                }
            }
        }
    }
    if (local == 0) {
        // Unmatched probe row: emit one row at `base` with the null sentinel.
        if (base < output_capacity) {
            output_left[base] = tid;
            output_right[base] = 0xFFFFFFFFu;
        } else {
            *d_overflow = 1;
        }
    }
}

/**
 * Compute the total of an exclusive scan on a per-probe count array,
 * clamp to `capacity`, and write to `d_logical_count`. If the unclamped
 * total exceeds `capacity`, set `d_overflow` to 1.
 *
 * Single-thread kernel; reads two scalars from the device buffers and
 * writes one scalar. Used to derive the join result's logical row
 * count without a host-side D2H of the post-scan total.
 */
extern "C" __global__ void hash_join_total_from_scan(
    const uint32_t* __restrict__ per_probe_offsets,
    const uint32_t* __restrict__ per_probe_count,
    const uint32_t* __restrict__ num_probe_device,
    uint32_t probe_cap,
    uint32_t capacity,
    uint32_t* __restrict__ d_logical_count,
    uint8_t* __restrict__ d_overflow
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t num_probe = *num_probe_device;
    if (num_probe > probe_cap) num_probe = probe_cap;
    if (num_probe == 0) {
        *d_logical_count = 0;
        return;
    }
    uint32_t tail_off = per_probe_offsets[num_probe - 1];
    uint32_t tail_cnt = per_probe_count[num_probe - 1];
    uint32_t total = tail_off + tail_cnt;
    if (total > capacity) {
        *d_overflow = 1;
        *d_logical_count = capacity;
    } else {
        *d_logical_count = total;
    }
}

/**
 * Semi-join: mark probe rows that have any match.
 * @param probe_hashes Hash values for probe side
 * @param num_probe Number of probe rows
 * @param bucket_offsets Start offsets per bucket
 * @param bucket_counts Build row counts per bucket
 * @param bucket_entries Build row indices, bucketed contiguously
 * @param bucket_entry_hashes Build hashes aligned with bucket_entries
 * @param probe_keys Packed probe key data (row-major, key columns only)
 * @param build_keys Packed build key data (row-major, key columns only)
 * @param key_bytes Total bytes per key (sum of key column sizes)
 * @param has_match Output: 1 if row has match, 0 otherwise
 */
extern "C" __global__ void hash_join_semi(
    const uint64_t* __restrict__ probe_hashes,
    uint32_t num_probe,
    const uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_counts,
    const uint32_t* __restrict__ bucket_entries,
    const uint64_t* __restrict__ bucket_entry_hashes,
    uint32_t bucket_mask,
    const uint8_t* __restrict__ probe_keys,
    const uint8_t* __restrict__ build_keys,
    uint32_t key_bytes,
    uint8_t* __restrict__ has_match
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_probe) return;

    uint64_t hash = probe_hashes[tid];
    uint32_t h32 = (uint32_t)(hash ^ (hash >> 32));
    uint32_t bucket = h32 & bucket_mask;

    uint32_t start = bucket_offsets[bucket];
    uint32_t count = bucket_counts[bucket];
    for (uint32_t i = 0; i < count; i++) {
        uint32_t pos = start + i;
        if (bucket_entry_hashes[pos] == hash) {
            uint32_t build_idx = bucket_entries[pos];
            if (keys_equal(probe_keys, build_keys, tid, build_idx, key_bytes)) {
                has_match[tid] = 1;
                return;
            }
        }
    }
    has_match[tid] = 0;
}

/**
 * Anti-join: mark probe rows that have NO match.
 * @param probe_hashes Hash values for probe side
 * @param num_probe Number of probe rows
 * @param bucket_offsets Start offsets per bucket
 * @param bucket_counts Build row counts per bucket
 * @param bucket_entries Build row indices, bucketed contiguously
 * @param bucket_entry_hashes Build hashes aligned with bucket_entries
 * @param probe_keys Packed probe key data (row-major, key columns only)
 * @param build_keys Packed build key data (row-major, key columns only)
 * @param key_bytes Total bytes per key (sum of key column sizes)
 * @param no_match Output: 1 if no match, 0 otherwise
 */
extern "C" __global__ void hash_join_anti(
    const uint64_t* __restrict__ probe_hashes,
    uint32_t num_probe,
    const uint32_t* __restrict__ bucket_offsets,
    const uint32_t* __restrict__ bucket_counts,
    const uint32_t* __restrict__ bucket_entries,
    const uint64_t* __restrict__ bucket_entry_hashes,
    uint32_t bucket_mask,
    const uint8_t* __restrict__ probe_keys,
    const uint8_t* __restrict__ build_keys,
    uint32_t key_bytes,
    uint8_t* __restrict__ no_match
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_probe) return;

    uint64_t hash = probe_hashes[tid];
    uint32_t h32 = (uint32_t)(hash ^ (hash >> 32));
    uint32_t bucket = h32 & bucket_mask;

    uint32_t start = bucket_offsets[bucket];
    uint32_t count = bucket_counts[bucket];
    for (uint32_t i = 0; i < count; i++) {
        uint32_t pos = start + i;
        if (bucket_entry_hashes[pos] == hash) {
            uint32_t build_idx = bucket_entries[pos];
            if (keys_equal(probe_keys, build_keys, tid, build_idx, key_bytes)) {
                no_match[tid] = 0;
                return;
            }
        }
    }
    no_match[tid] = 1;
}

/**
 * Initialize hash table buckets to empty (0xFFFFFFFF).
 */
extern "C" __global__ void init_hash_table(
    uint32_t* __restrict__ hash_table,
    uint32_t size
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        hash_table[gid] = 0xFFFFFFFF;
    }
}
