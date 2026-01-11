// XLOG GPU Key Packing Kernels
// GPU-side key packing for multi-column joins
// Eliminates host roundtrips by performing all key packing on device

#include <cstdint>

// FNV-1a hash constants (same as join.cu for consistency)
#define FNV_OFFSET 0xcbf29ce484222325ULL
#define FNV_PRIME  0x100000001b3ULL

/**
 * Pack multiple columns into row-major byte array.
 *
 * This kernel takes up to 4 separate column buffers and packs them into
 * a contiguous row-major output buffer. Each thread processes one row.
 * Column data is assumed to be stored contiguously (element N of column C
 * is at offset N * col_sizes[C]).
 *
 * @param col0 First column data buffer (or nullptr if num_cols < 1)
 * @param col1 Second column data buffer (or nullptr if num_cols < 2)
 * @param col2 Third column data buffer (or nullptr if num_cols < 3)
 * @param col3 Fourth column data buffer (or nullptr if num_cols < 4)
 * @param col_sizes Size in bytes of each column element (array of num_cols entries)
 * @param num_cols Number of columns to pack (1-4)
 * @param num_rows Number of rows to process
 * @param row_size Total size of packed row in bytes (sum of col_sizes)
 * @param packed_output Output buffer for packed rows (row_size * num_rows bytes)
 */
extern "C" __global__ void pack_keys(
    const uint8_t* __restrict__ col0,
    const uint8_t* __restrict__ col1,
    const uint8_t* __restrict__ col2,
    const uint8_t* __restrict__ col3,
    const uint32_t* __restrict__ col_sizes,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_size,
    uint8_t* __restrict__ packed_output
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    uint8_t* out_row = packed_output + (uint64_t)row * row_size;
    uint32_t offset = 0;

    // Unrolled for up to 4 key columns (common case for multi-column joins)
    // Loop unrolling provides better instruction-level parallelism
    if (num_cols >= 1 && col0 != nullptr) {
        uint32_t sz = col_sizes[0];
        const uint8_t* col_data = col0 + (uint64_t)row * sz;
        for (uint32_t i = 0; i < sz; i++) {
            out_row[offset + i] = col_data[i];
        }
        offset += sz;
    }
    if (num_cols >= 2 && col1 != nullptr) {
        uint32_t sz = col_sizes[1];
        const uint8_t* col_data = col1 + (uint64_t)row * sz;
        for (uint32_t i = 0; i < sz; i++) {
            out_row[offset + i] = col_data[i];
        }
        offset += sz;
    }
    if (num_cols >= 3 && col2 != nullptr) {
        uint32_t sz = col_sizes[2];
        const uint8_t* col_data = col2 + (uint64_t)row * sz;
        for (uint32_t i = 0; i < sz; i++) {
            out_row[offset + i] = col_data[i];
        }
        offset += sz;
    }
    if (num_cols >= 4 && col3 != nullptr) {
        uint32_t sz = col_sizes[3];
        const uint8_t* col_data = col3 + (uint64_t)row * sz;
        for (uint32_t i = 0; i < sz; i++) {
            out_row[offset + i] = col_data[i];
        }
        offset += sz;
    }
}

/**
 * Compute FNV-1a hash from packed keys.
 *
 * This kernel hashes already-packed key data. Each thread processes one row.
 * Useful when packed keys are reused (e.g., stored for later comparison).
 *
 * @param packed_keys Input packed key data (row_size * num_rows bytes)
 * @param row_size Size of each packed row in bytes
 * @param num_rows Number of rows to hash
 * @param hashes Output hash values (one uint64_t per row)
 */
extern "C" __global__ void hash_packed_keys(
    const uint8_t* __restrict__ packed_keys,
    uint32_t row_size,
    uint32_t num_rows,
    uint64_t* __restrict__ hashes
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    const uint8_t* key = packed_keys + (uint64_t)row * row_size;
    uint64_t hash = FNV_OFFSET;

    for (uint32_t i = 0; i < row_size; i++) {
        hash ^= key[i];
        hash *= FNV_PRIME;
    }

    hashes[row] = hash;
}

/**
 * Fused pack + hash in single pass (better cache utilization).
 *
 * This kernel combines packing and hashing into a single pass over the data.
 * Benefits:
 * - Reduced memory traffic (each byte is read once, not stored then re-read)
 * - Better L1/L2 cache utilization
 * - Single kernel launch instead of two
 *
 * Use this kernel when you need both packed output and hashes.
 * If you only need hashes without packed output, consider using
 * compute_composite_hash from join.cu instead.
 *
 * @param col0 First column data buffer (or nullptr if num_cols < 1)
 * @param col1 Second column data buffer (or nullptr if num_cols < 2)
 * @param col2 Third column data buffer (or nullptr if num_cols < 3)
 * @param col3 Fourth column data buffer (or nullptr if num_cols < 4)
 * @param col_sizes Size in bytes of each column element
 * @param num_cols Number of columns to pack (1-4)
 * @param num_rows Number of rows to process
 * @param row_size Total size of packed row in bytes
 * @param packed_output Output buffer for packed rows
 * @param hashes Output hash values (one uint64_t per row)
 */
extern "C" __global__ void pack_and_hash_keys(
    const uint8_t* __restrict__ col0,
    const uint8_t* __restrict__ col1,
    const uint8_t* __restrict__ col2,
    const uint8_t* __restrict__ col3,
    const uint32_t* __restrict__ col_sizes,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_size,
    uint8_t* __restrict__ packed_output,
    uint64_t* __restrict__ hashes
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    uint8_t* out_row = packed_output + (uint64_t)row * row_size;
    uint64_t hash = FNV_OFFSET;
    uint32_t offset = 0;

    // Pack and hash simultaneously - each byte is processed exactly once
    if (num_cols >= 1 && col0 != nullptr) {
        uint32_t sz = col_sizes[0];
        const uint8_t* col_data = col0 + (uint64_t)row * sz;
        for (uint32_t i = 0; i < sz; i++) {
            uint8_t b = col_data[i];
            out_row[offset + i] = b;
            hash ^= b;
            hash *= FNV_PRIME;
        }
        offset += sz;
    }
    if (num_cols >= 2 && col1 != nullptr) {
        uint32_t sz = col_sizes[1];
        const uint8_t* col_data = col1 + (uint64_t)row * sz;
        for (uint32_t i = 0; i < sz; i++) {
            uint8_t b = col_data[i];
            out_row[offset + i] = b;
            hash ^= b;
            hash *= FNV_PRIME;
        }
        offset += sz;
    }
    if (num_cols >= 3 && col2 != nullptr) {
        uint32_t sz = col_sizes[2];
        const uint8_t* col_data = col2 + (uint64_t)row * sz;
        for (uint32_t i = 0; i < sz; i++) {
            uint8_t b = col_data[i];
            out_row[offset + i] = b;
            hash ^= b;
            hash *= FNV_PRIME;
        }
        offset += sz;
    }
    if (num_cols >= 4 && col3 != nullptr) {
        uint32_t sz = col_sizes[3];
        const uint8_t* col_data = col3 + (uint64_t)row * sz;
        for (uint32_t i = 0; i < sz; i++) {
            uint8_t b = col_data[i];
            out_row[offset + i] = b;
            hash ^= b;
            hash *= FNV_PRIME;
        }
        offset += sz;
    }

    hashes[row] = hash;
}

/**
 * Vectorized pack for 8-byte aligned columns (optimized path).
 *
 * When all column sizes are multiples of 8 bytes and properly aligned,
 * this kernel uses 64-bit loads/stores for significantly higher throughput.
 * Falls back to byte-by-byte copy for any partial (non-8-byte) portions.
 *
 * @param col0 First column data buffer (8-byte aligned, or nullptr)
 * @param col1 Second column data buffer (8-byte aligned, or nullptr)
 * @param col2 Third column data buffer (8-byte aligned, or nullptr)
 * @param col3 Fourth column data buffer (8-byte aligned, or nullptr)
 * @param col_sizes Size in bytes of each column element (all should be multiple of 8)
 * @param num_cols Number of columns to pack (1-4)
 * @param num_rows Number of rows to process
 * @param row_size Total size of packed row in bytes (should be multiple of 8)
 * @param packed_output Output buffer for packed rows (8-byte aligned)
 */
extern "C" __global__ void pack_keys_aligned(
    const uint8_t* __restrict__ col0,
    const uint8_t* __restrict__ col1,
    const uint8_t* __restrict__ col2,
    const uint8_t* __restrict__ col3,
    const uint32_t* __restrict__ col_sizes,
    uint32_t num_cols,
    uint32_t num_rows,
    uint32_t row_size,
    uint8_t* __restrict__ packed_output
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    uint8_t* out_row = packed_output + (uint64_t)row * row_size;
    uint32_t offset = 0;

    // Process each column with vectorized loads where possible
    if (num_cols >= 1 && col0 != nullptr) {
        uint32_t sz = col_sizes[0];
        const uint8_t* col_data = col0 + (uint64_t)row * sz;

        // Copy 8 bytes at a time
        uint32_t i = 0;
        for (; i + 8 <= sz; i += 8) {
            *((uint64_t*)(out_row + offset + i)) = *((const uint64_t*)(col_data + i));
        }
        // Handle remainder
        for (; i < sz; i++) {
            out_row[offset + i] = col_data[i];
        }
        offset += sz;
    }
    if (num_cols >= 2 && col1 != nullptr) {
        uint32_t sz = col_sizes[1];
        const uint8_t* col_data = col1 + (uint64_t)row * sz;

        uint32_t i = 0;
        for (; i + 8 <= sz; i += 8) {
            *((uint64_t*)(out_row + offset + i)) = *((const uint64_t*)(col_data + i));
        }
        for (; i < sz; i++) {
            out_row[offset + i] = col_data[i];
        }
        offset += sz;
    }
    if (num_cols >= 3 && col2 != nullptr) {
        uint32_t sz = col_sizes[2];
        const uint8_t* col_data = col2 + (uint64_t)row * sz;

        uint32_t i = 0;
        for (; i + 8 <= sz; i += 8) {
            *((uint64_t*)(out_row + offset + i)) = *((const uint64_t*)(col_data + i));
        }
        for (; i < sz; i++) {
            out_row[offset + i] = col_data[i];
        }
        offset += sz;
    }
    if (num_cols >= 4 && col3 != nullptr) {
        uint32_t sz = col_sizes[3];
        const uint8_t* col_data = col3 + (uint64_t)row * sz;

        uint32_t i = 0;
        for (; i + 8 <= sz; i += 8) {
            *((uint64_t*)(out_row + offset + i)) = *((const uint64_t*)(col_data + i));
        }
        for (; i < sz; i++) {
            out_row[offset + i] = col_data[i];
        }
        offset += sz;
    }
}

/**
 * Unpack single column from packed row data.
 *
 * Extracts one column's data from row-major packed storage back into
 * column-major format. Useful for projection after joins.
 *
 * @param packed_input Input packed row data
 * @param row_size Total size of each packed row in bytes
 * @param col_offset Byte offset of target column within packed row
 * @param col_size Size of target column in bytes
 * @param num_rows Number of rows to process
 * @param col_output Output buffer for extracted column data
 */
extern "C" __global__ void unpack_column(
    const uint8_t* __restrict__ packed_input,
    uint32_t row_size,
    uint32_t col_offset,
    uint32_t col_size,
    uint32_t num_rows,
    uint8_t* __restrict__ col_output
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    const uint8_t* src = packed_input + (uint64_t)row * row_size + col_offset;
    uint8_t* dst = col_output + (uint64_t)row * col_size;

    for (uint32_t i = 0; i < col_size; i++) {
        dst[i] = src[i];
    }
}

/**
 * Gather rows from packed data based on index array.
 *
 * Given an array of row indices, copies those rows from source packed
 * data to destination. Useful for materializing join results.
 *
 * @param src_packed Source packed row data
 * @param row_size Size of each packed row in bytes
 * @param indices Array of row indices to gather
 * @param num_indices Number of indices (output row count)
 * @param dst_packed Output buffer for gathered rows
 */
extern "C" __global__ void gather_packed_rows(
    const uint8_t* __restrict__ src_packed,
    uint32_t row_size,
    const uint32_t* __restrict__ indices,
    uint32_t num_indices,
    uint8_t* __restrict__ dst_packed
) {
    uint32_t out_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_row >= num_indices) return;

    uint32_t src_row = indices[out_row];
    const uint8_t* src = src_packed + (uint64_t)src_row * row_size;
    uint8_t* dst = dst_packed + (uint64_t)out_row * row_size;

    // Copy row data - use vectorized loads where possible
    uint32_t i = 0;
    for (; i + 8 <= row_size; i += 8) {
        *((uint64_t*)(dst + i)) = *((const uint64_t*)(src + i));
    }
    for (; i < row_size; i++) {
        dst[i] = src[i];
    }
}

/**
 * Scatter write: distribute packed rows to non-contiguous output positions.
 *
 * Inverse of gather - writes source rows to specified output positions.
 * Useful for join output where left/right side rows interleave.
 *
 * @param src_packed Source packed row data
 * @param row_size Size of each packed row in bytes
 * @param output_indices Array specifying output position for each input row
 * @param num_rows Number of rows to scatter
 * @param dst_packed Output buffer (must be pre-allocated to hold max(output_indices)+1 rows)
 */
extern "C" __global__ void scatter_packed_rows(
    const uint8_t* __restrict__ src_packed,
    uint32_t row_size,
    const uint32_t* __restrict__ output_indices,
    uint32_t num_rows,
    uint8_t* __restrict__ dst_packed
) {
    uint32_t src_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (src_row >= num_rows) return;

    uint32_t dst_row = output_indices[src_row];
    const uint8_t* src = src_packed + (uint64_t)src_row * row_size;
    uint8_t* dst = dst_packed + (uint64_t)dst_row * row_size;

    // Copy row data - use vectorized loads where possible
    uint32_t i = 0;
    for (; i + 8 <= row_size; i += 8) {
        *((uint64_t*)(dst + i)) = *((const uint64_t*)(src + i));
    }
    for (; i < row_size; i++) {
        dst[i] = src[i];
    }
}

/**
 * Compare packed keys for equality (used in merge joins, sorting validation).
 *
 * Compares two packed key arrays element-by-element and outputs comparison result.
 * Result is -1 if a < b, 0 if a == b, 1 if a > b (lexicographic comparison).
 *
 * @param packed_a First packed key array
 * @param packed_b Second packed key array
 * @param row_size Size of each packed row in bytes
 * @param num_rows Number of rows to compare
 * @param results Output comparison results (-1, 0, or 1 per row)
 */
extern "C" __global__ void compare_packed_keys(
    const uint8_t* __restrict__ packed_a,
    const uint8_t* __restrict__ packed_b,
    uint32_t row_size,
    uint32_t num_rows,
    int8_t* __restrict__ results
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    const uint8_t* a = packed_a + (uint64_t)row * row_size;
    const uint8_t* b = packed_b + (uint64_t)row * row_size;

    for (uint32_t i = 0; i < row_size; i++) {
        if (a[i] < b[i]) {
            results[row] = -1;
            return;
        }
        if (a[i] > b[i]) {
            results[row] = 1;
            return;
        }
    }
    results[row] = 0;
}
