#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kMaxArity = 17;
constexpr uint32_t kRunning = 0;
constexpr uint32_t kCapacityOverflow = 3;
constexpr uint32_t kResourceExhausted = 4;
constexpr uint32_t kClaiming = 0xfffffffeU;
constexpr uint32_t kInputRows = 4;
constexpr uint32_t kOutputRows = 5;

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

struct __align__(8) ResidentFilterComparisonDescriptor {
    uint32_t left_kind;
    uint32_t left_column;
    uint32_t right_kind;
    uint32_t right_column;
    uint32_t op;
    uint32_t width;
    uint32_t reserved_zero;
    uint32_t reserved_one;
    uint64_t left_constant;
    uint64_t right_constant;
};

struct __align__(8) ResidentProjectDescriptor {
    uint32_t kind;
    uint32_t column;
    uint32_t width;
    uint32_t reserved;
    uint64_t constant;
};

__device__ __forceinline__ void publish_terminal(
    ResidentTerminalStatus *status,
    uint32_t terminal_code,
    uint32_t op_id,
    uint32_t resource_code,
    uint64_t required,
    uint64_t capacity
) {
    if (atomicCAS(&status->code, kRunning, kClaiming) != kRunning) return;
    status->op_id = op_id;
    status->resource_code = resource_code;
    status->reserved = 0;
    status->required = required;
    status->capacity = capacity;
    __threadfence_system();
    atomicExch(&status->code, terminal_code);
}

__device__ __forceinline__ uint32_t bounded_rows(
    ResidentRelationView relation,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    const uint32_t rows = *reinterpret_cast<const uint32_t *>(relation.num_rows);
    if (rows > relation.capacity) {
        publish_terminal(
            status,
            kResourceExhausted,
            op_id,
            kInputRows,
            rows,
            relation.capacity
        );
    }
    return min(rows, relation.capacity);
}

__device__ __forceinline__ uint64_t column_value(
    ResidentRelationView relation,
    uint32_t column,
    uint32_t row
) {
    if (relation.widths[column] == 4) {
        return reinterpret_cast<const uint32_t *>(relation.columns[column])[row];
    }
    return reinterpret_cast<const uint64_t *>(relation.columns[column])[row];
}

__device__ __forceinline__ uint64_t operand_value(
    ResidentRelationView input,
    uint32_t kind,
    uint32_t column,
    uint64_t constant,
    uint32_t row
) {
    return kind == 0 ? column_value(input, column, row) : constant;
}

__device__ __forceinline__ bool compare_values(
    uint64_t left,
    uint32_t op,
    uint64_t right
) {
    switch (op) {
        case 0: return left == right;
        case 1: return left != right;
        case 2: return left < right;
        case 3: return left <= right;
        case 4: return left > right;
        case 5: return left >= right;
        default: return false;
    }
}

__device__ __forceinline__ bool matches_all(
    ResidentRelationView input,
    const ResidentFilterComparisonDescriptor *comparisons,
    uint32_t comparison_count,
    uint32_t row
) {
    for (uint32_t index = 0; index < comparison_count; ++index) {
        const ResidentFilterComparisonDescriptor comparison = comparisons[index];
        const uint64_t left = operand_value(
            input,
            comparison.left_kind,
            comparison.left_column,
            comparison.left_constant,
            row
        );
        const uint64_t right = operand_value(
            input,
            comparison.right_kind,
            comparison.right_column,
            comparison.right_constant,
            row
        );
        if (!compare_values(left, comparison.op, right)) return false;
    }
    return true;
}

__device__ __forceinline__ void copy_row(
    ResidentRelationView input,
    uint32_t input_row,
    ResidentRelationView output,
    uint32_t output_row
) {
    for (uint32_t column = 0; column < input.arity; ++column) {
        if (input.widths[column] == 4) {
            reinterpret_cast<uint32_t *>(output.columns[column])[output_row] =
                static_cast<uint32_t>(column_value(input, column, input_row));
        } else {
            reinterpret_cast<uint64_t *>(output.columns[column])[output_row] =
                column_value(input, column, input_row);
        }
    }
}

} // namespace

extern "C" __global__ void resident_filter_mask_scan(
    ResidentRelationView input,
    const ResidentFilterComparisonDescriptor *comparisons,
    uint32_t comparison_count,
    uint32_t *mask,
    uint32_t *prefix,
    uint32_t *block_sums,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    __shared__ uint32_t scan[kBlockSize];
    const uint32_t lane = threadIdx.x;
    const uint32_t row = blockIdx.x * blockDim.x + lane;
    const uint32_t rows = bounded_rows(input, status, op_id);
    const uint32_t keep = row < rows && status->code == kRunning &&
        matches_all(input, comparisons, comparison_count, row);
    scan[lane] = keep;
    __syncthreads();
    for (uint32_t offset = 1; offset < kBlockSize; offset <<= 1) {
        const uint32_t addend = lane >= offset ? scan[lane - offset] : 0;
        __syncthreads();
        scan[lane] += addend;
        __syncthreads();
    }
    if (row < input.capacity) {
        mask[row] = keep;
        prefix[row] = scan[lane] - keep;
    }
    if (lane == kBlockSize - 1) block_sums[blockIdx.x] = scan[lane];
}

extern "C" __global__ void resident_filter_scan_blocks(
    const uint32_t *block_sums,
    uint32_t *block_offsets,
    uint32_t block_count
) {
    __shared__ uint32_t scan[kBlockSize];
    const uint32_t lane = threadIdx.x;
    scan[lane] = lane < block_count ? block_sums[lane] : 0;
    __syncthreads();
    for (uint32_t offset = 1; offset < kBlockSize; offset <<= 1) {
        const uint32_t addend = lane >= offset ? scan[lane - offset] : 0;
        __syncthreads();
        scan[lane] += addend;
        __syncthreads();
    }
    if (lane < block_count) block_offsets[lane] = scan[lane] - block_sums[lane];
}

extern "C" __global__ void resident_filter_add_offsets(
    uint32_t *prefix,
    const uint32_t *block_offsets,
    uint32_t capacity
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < capacity) prefix[row] += block_offsets[blockIdx.x];
}

extern "C" __global__ void resident_filter_finalize(
    ResidentRelationView input,
    const uint32_t *mask,
    const uint32_t *prefix,
    ResidentRelationView output,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || status->code != kRunning) return;
    const uint32_t rows = bounded_rows(input, status, op_id);
    if (status->code != kRunning) return;
    const uint64_t required = rows == 0 ? 0 :
        static_cast<uint64_t>(prefix[rows - 1]) + mask[rows - 1];
    if (required > output.capacity) {
        publish_terminal(
            status,
            kCapacityOverflow,
            op_id,
            kOutputRows,
            required,
            output.capacity
        );
        return;
    }
    *reinterpret_cast<uint32_t *>(output.num_rows) = static_cast<uint32_t>(required);
}

extern "C" __global__ void resident_filter_compact(
    ResidentRelationView input,
    const uint32_t *mask,
    const uint32_t *prefix,
    ResidentRelationView output,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t rows = bounded_rows(input, status, op_id);
    if (status->code != kRunning || row >= rows || mask[row] == 0) return;
    const uint32_t destination = prefix[row];
    if (destination >= output.capacity) {
        publish_terminal(
            status,
            kCapacityOverflow,
            op_id,
            kOutputRows,
            static_cast<uint64_t>(destination) + 1,
            output.capacity
        );
        return;
    }
    copy_row(input, row, output, destination);
}

extern "C" __global__ void resident_project_finalize(
    ResidentRelationView input,
    ResidentRelationView output,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || status->code != kRunning) return;
    const uint32_t rows = bounded_rows(input, status, op_id);
    if (status->code != kRunning) return;
    if (rows > output.capacity) {
        publish_terminal(status, kCapacityOverflow, op_id, kOutputRows, rows, output.capacity);
        return;
    }
    *reinterpret_cast<uint32_t *>(output.num_rows) = rows;
}

extern "C" __global__ void resident_project_materialize(
    ResidentRelationView input,
    const ResidentProjectDescriptor *descriptors,
    uint32_t expression_count,
    ResidentRelationView output,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t rows = bounded_rows(input, status, op_id);
    if (status->code != kRunning || row >= rows || row >= output.capacity) return;
    for (uint32_t output_column = 0; output_column < expression_count; ++output_column) {
        const ResidentProjectDescriptor descriptor = descriptors[output_column];
        const uint64_t value = descriptor.kind == 0
            ? column_value(input, descriptor.column, row)
            : descriptor.constant;
        if (descriptor.width == 4) {
            reinterpret_cast<uint32_t *>(output.columns[output_column])[row] =
                static_cast<uint32_t>(value);
        } else {
            reinterpret_cast<uint64_t *>(output.columns[output_column])[row] = value;
        }
    }
}
