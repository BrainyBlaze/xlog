#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __device__ void cudaGraphSetConditional(
    unsigned long long handle,
    unsigned int value
);

namespace {

constexpr uint32_t kMaxArity = 17;
constexpr uint32_t kRunning = 0;
constexpr uint32_t kSuccess = 1;
constexpr uint32_t kIterationLimit = 2;
constexpr uint32_t kCapacityOverflow = 3;
constexpr uint32_t kResourceExhausted = 4;
constexpr uint32_t kClaiming = 0xfffffffeU;
constexpr uint32_t kInputRows = 4;
constexpr uint32_t kOutputRows = 5;
constexpr uint64_t kTagBit = 1ULL << 63;

struct __align__(8) ResidentTerminalStatus {
    uint32_t code;
    uint32_t op_id;
    uint32_t resource_code;
    uint32_t iterations;
    uint32_t limit;
    uint32_t reserved;
    uint64_t required;
    uint64_t capacity;
};

struct __align__(8) ResidentRelationView {
    uint64_t columns[kMaxArity];
    uint32_t widths[kMaxArity];
    uint32_t arity;
    uint32_t capacity;
    uint32_t reserved;
    uint64_t num_rows;
};

__device__ __forceinline__ uint32_t bounded_rows(
    ResidentRelationView relation,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    const uint32_t rows = *reinterpret_cast<const uint32_t *>(relation.num_rows);
    if (rows > relation.capacity &&
        atomicCAS(&status->code, kRunning, kClaiming) == kRunning) {
        status->op_id = op_id;
        status->resource_code = kInputRows;
        status->required = rows;
        status->capacity = relation.capacity;
        __threadfence_system();
        atomicExch(&status->code, kResourceExhausted);
    }
    return min(rows, relation.capacity);
}

__device__ __forceinline__ uint64_t value_at(
    ResidentRelationView relation,
    uint32_t column,
    uint32_t row
) {
    if (relation.widths[column] == 4) {
        return reinterpret_cast<const uint32_t *>(relation.columns[column])[row];
    }
    return reinterpret_cast<const uint64_t *>(relation.columns[column])[row];
}

__device__ __forceinline__ bool rows_equal(
    ResidentRelationView a,
    uint32_t a_row,
    ResidentRelationView b,
    uint32_t b_row
) {
    if (a.arity != b.arity) return false;
    for (uint32_t column = 0; column < a.arity; ++column) {
        if (a.widths[column] != b.widths[column] ||
            value_at(a, column, a_row) != value_at(b, column, b_row)) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ uint64_t row_hash(
    ResidentRelationView relation,
    uint32_t row
) {
    uint64_t hash = 1469598103934665603ULL;
    for (uint32_t column = 0; column < relation.arity; ++column) {
        uint64_t value = value_at(relation, column, row);
        hash ^= value;
        hash *= 1099511628211ULL;
        hash ^= value >> 32;
        hash *= 1099511628211ULL;
    }
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;
    return hash;
}

__device__ __forceinline__ void copy_row(
    ResidentRelationView source,
    uint32_t source_row,
    ResidentRelationView output,
    uint32_t output_row,
    uint32_t output_column_offset
) {
    for (uint32_t column = 0; column < source.arity; ++column) {
        const uint32_t output_column = output_column_offset + column;
        if (source.widths[column] == 4) {
            reinterpret_cast<uint32_t *>(output.columns[output_column])[output_row] =
                static_cast<uint32_t>(value_at(source, column, source_row));
        } else {
            reinterpret_cast<uint64_t *>(output.columns[output_column])[output_row] =
                value_at(source, column, source_row);
        }
    }
}

__device__ __forceinline__ void publish_terminal(
    ResidentTerminalStatus *status,
    uint32_t terminal_code,
    uint32_t op_id,
    uint32_t resource_code,
    uint32_t iterations,
    uint32_t limit,
    uint64_t required,
    uint64_t capacity
) {
    if (atomicCAS(&status->code, kRunning, kClaiming) != kRunning) return;
    status->op_id = op_id;
    status->resource_code = resource_code;
    status->iterations = iterations;
    status->limit = limit;
    status->reserved = 0;
    status->required = required;
    status->capacity = capacity;
    __threadfence_system();
    atomicExch(&status->code, terminal_code);
}

__device__ __forceinline__ bool key_equal(
    ResidentRelationView left,
    uint32_t left_row,
    uint32_t left_key,
    ResidentRelationView right,
    uint32_t right_row,
    uint32_t right_key
) {
    return value_at(left, left_key, left_row) == value_at(right, right_key, right_row);
}

} // namespace

extern "C" __global__ void resident_set_insert(
    ResidentRelationView candidate,
    ResidentRelationView relation_zero,
    ResidentRelationView relation_one,
    uint32_t source_tag,
    uint32_t emit_rows,
    uint32_t materialize,
    uint64_t *slots,
    uint32_t slot_mask,
    ResidentRelationView output,
    unsigned long long *required,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t rows = bounded_rows(candidate, status, op_id);
    if (row >= rows || status->code != kRunning) return;
    const uint64_t encoded = (source_tag ? kTagBit : 0) | (static_cast<uint64_t>(row) + 1);
    uint32_t slot = static_cast<uint32_t>(row_hash(candidate, row)) & slot_mask;
    for (uint32_t probe = 0; probe <= slot_mask; ++probe) {
        const uint64_t previous = atomicCAS(
            reinterpret_cast<unsigned long long *>(&slots[slot]),
            0ULL,
            encoded
        );
        if (previous == 0) {
            if (emit_rows) {
                const uint64_t output_row = atomicAdd(required, 1ULL);
                if (materialize && output_row < output.capacity) {
                    copy_row(candidate, row, output, static_cast<uint32_t>(output_row), 0);
                }
            }
            return;
        }
        const uint32_t existing_tag = (previous & kTagBit) != 0;
        const uint32_t existing_row = static_cast<uint32_t>((previous & ~kTagBit) - 1);
        const ResidentRelationView existing = existing_tag ? relation_one : relation_zero;
        if (rows_equal(candidate, row, existing, existing_row)) return;
        slot = (slot + 1) & slot_mask;
    }
}

extern "C" __global__ void resident_set_finalize(
    const unsigned long long *required,
    ResidentRelationView output,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const uint64_t count = *required;
    *reinterpret_cast<uint32_t *>(output.num_rows) =
        static_cast<uint32_t>(min(count, static_cast<uint64_t>(output.capacity)));
    if (count > output.capacity) {
        publish_terminal(status, kCapacityOverflow, op_id, kOutputRows, status->iterations,
                         status->limit, count, output.capacity);
    }
}

extern "C" __global__ void resident_join_build(
    ResidentRelationView right,
    uint32_t right_key,
    uint32_t *bucket_heads,
    uint32_t bucket_mask,
    uint32_t *next,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t rows = bounded_rows(right, status, op_id);
    if (row >= rows || status->code != kRunning) return;
    uint64_t key = value_at(right, right_key, row);
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    const uint32_t bucket = static_cast<uint32_t>(key) & bucket_mask;
    next[row] = atomicExch(&bucket_heads[bucket], row);
}

extern "C" __global__ void resident_join_probe_inner(
    ResidentRelationView left,
    uint32_t left_key,
    ResidentRelationView right,
    uint32_t right_key,
    const uint32_t *bucket_heads,
    uint32_t bucket_mask,
    const uint32_t *next,
    ResidentRelationView output,
    unsigned long long *required,
    uint32_t materialize,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    const uint32_t left_row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t left_rows = bounded_rows(left, status, op_id);
    if (left_row >= left_rows || status->code != kRunning) return;
    uint64_t key = value_at(left, left_key, left_row);
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    for (uint32_t right_row = bucket_heads[static_cast<uint32_t>(key) & bucket_mask];
         right_row != 0xffffffffU;
         right_row = next[right_row]) {
        if (!key_equal(left, left_row, left_key, right, right_row, right_key)) continue;
        const uint64_t output_row = atomicAdd(required, 1ULL);
        if (materialize && output_row < output.capacity) {
            copy_row(left, left_row, output, static_cast<uint32_t>(output_row), 0);
            copy_row(right, right_row, output, static_cast<uint32_t>(output_row), left.arity);
        }
    }
}

extern "C" __global__ void resident_join_probe_semi(
    ResidentRelationView left,
    uint32_t left_key,
    ResidentRelationView right,
    uint32_t right_key,
    const uint32_t *bucket_heads,
    uint32_t bucket_mask,
    const uint32_t *next,
    ResidentRelationView output,
    unsigned long long *required,
    uint32_t materialize,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    const uint32_t left_row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t left_rows = bounded_rows(left, status, op_id);
    if (left_row >= left_rows || status->code != kRunning) return;
    uint64_t key = value_at(left, left_key, left_row);
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    for (uint32_t right_row = bucket_heads[static_cast<uint32_t>(key) & bucket_mask];
         right_row != 0xffffffffU;
         right_row = next[right_row]) {
        if (!key_equal(left, left_row, left_key, right, right_row, right_key)) continue;
        const uint64_t output_row = atomicAdd(required, 1ULL);
        if (materialize && output_row < output.capacity) {
            copy_row(left, left_row, output, static_cast<uint32_t>(output_row), 0);
        }
        return;
    }
}

extern "C" __global__ void resident_join_finalize(
    const unsigned long long *required,
    ResidentRelationView output,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const uint64_t count = *required;
    *reinterpret_cast<uint32_t *>(output.num_rows) =
        static_cast<uint32_t>(min(count, static_cast<uint64_t>(output.capacity)));
    if (count > output.capacity) {
        publish_terminal(status, kCapacityOverflow, op_id, kOutputRows, status->iterations,
                         status->limit, count, output.capacity);
    }
}

extern "C" __global__ void resident_control_initialize(
    ResidentTerminalStatus *status,
    uint32_t *changed,
    uint32_t *loop_iterations
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        status->code = kRunning;
        status->op_id = 0;
        status->resource_code = 0;
        status->iterations = 0;
        status->limit = 0;
        status->reserved = 0;
        status->required = 0;
        status->capacity = 0;
        *changed = 0;
        *loop_iterations = 0;
    }
}

extern "C" __global__ void resident_scc_begin(
    uint32_t iteration_limit,
    uint32_t op_id,
    ResidentTerminalStatus *status,
    uint32_t *changed,
    uint32_t *loop_iterations
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || status->code != kRunning) return;
    *changed = 0;
    *loop_iterations = 0;
    if (iteration_limit == 0) {
        publish_terminal(status, kIterationLimit, op_id, 0, status->iterations,
                         0, 0, 0);
    }
}

extern "C" __global__ void resident_changed_reset(uint32_t *changed) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *changed = 0;
}

extern "C" __global__ void resident_changed_mark(
    const uint32_t *novel_count,
    uint32_t *changed
) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && *novel_count != 0) {
        atomicOr(changed, 1U);
    }
}

extern "C" __global__ void resident_convergence(
    unsigned long long conditional_handle,
    uint32_t iteration_limit,
    uint32_t op_id,
    ResidentTerminalStatus *status,
    uint32_t *changed,
    uint32_t *loop_iterations
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (status->code != kRunning) {
        cudaGraphSetConditional(conditional_handle, 0);
        return;
    }
    const uint32_t loop_iteration = *loop_iterations + 1;
    *loop_iterations = loop_iteration;
    const uint32_t total_iteration = status->iterations + 1;
    status->iterations = total_iteration;
    if (*changed == 0) {
        cudaGraphSetConditional(conditional_handle, 0);
        return;
    }
    if (loop_iteration >= iteration_limit) {
        publish_terminal(status, kIterationLimit, op_id, 0, total_iteration,
                         iteration_limit, 0, 0);
        cudaGraphSetConditional(conditional_handle, 0);
        return;
    }
    cudaGraphSetConditional(conditional_handle, 1);
}

extern "C" __global__ void resident_terminal_success(
    uint32_t op_id,
    ResidentTerminalStatus *status
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        publish_terminal(status, kSuccess, op_id, 0, status->iterations,
                         0, 0, 0);
    }
}

extern "C" __global__ void resident_test_status(
    uint32_t terminal_code,
    uint32_t op_id,
    uint32_t resource_code,
    uint32_t iterations,
    uint32_t limit,
    uint64_t required,
    uint64_t capacity,
    ResidentTerminalStatus *status
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        publish_terminal(status, terminal_code, op_id, resource_code, iterations,
                         limit, required, capacity);
    }
}

extern "C" __global__ void resident_trace_initialize(
    uint32_t *scan_invocations,
    uint32_t *filter_invocations,
    uint32_t *semantic_scan_invocations,
    uint32_t *semantic_filter_invocations
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *scan_invocations = 0;
        *filter_invocations = 0;
        *semantic_scan_invocations = 0;
        *semantic_filter_invocations = 0;
    }
}

extern "C" __global__ void resident_trace_increment(
    uint32_t *counter,
    uint32_t *semantic_counter
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        atomicAdd(counter, 1U);
        atomicAdd(semantic_counter, 1U);
    }
}

extern "C" __global__ void resident_schema_winners_initialize(
    const uint64_t *count_ptrs,
    uint32_t count_len,
    uint32_t *seen_nonempty
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count_len) {
        seen_nonempty[index] =
            *reinterpret_cast<const uint32_t *>(count_ptrs[index]) != 0 ? 1U : 0U;
    }
}

extern "C" __global__ void resident_schema_winner_mark(
    const uint32_t *contribution_count,
    uint32_t *seen_nonempty,
    uint32_t *winner_schema_ids,
    uint32_t head_index,
    uint32_t schema_id
) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && *contribution_count != 0
        && atomicCAS(&seen_nonempty[head_index], 0U, 1U) == 0U) {
        winner_schema_ids[head_index] = schema_id;
    }
}

extern "C" __global__ void resident_receipt_pack(
    const ResidentTerminalStatus *status,
    const uint32_t *changed,
    const uint64_t *count_ptrs,
    uint32_t count_len,
    uint8_t *receipt
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *reinterpret_cast<ResidentTerminalStatus *>(receipt) = *status;
    uint32_t *counts = reinterpret_cast<uint32_t *>(receipt + sizeof(ResidentTerminalStatus));
    counts[0] = *changed;
    for (uint32_t index = 0; index < count_len; ++index) {
        counts[index + 1] = *reinterpret_cast<const uint32_t *>(count_ptrs[index]);
    }
}
