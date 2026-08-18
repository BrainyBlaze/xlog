#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

extern "C" __device__ void cudaGraphSetConditional(
    unsigned long long handle,
    unsigned int value
);

namespace cg = cooperative_groups;

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kMaxArity = 17;
constexpr uint32_t kAbiVersion = 3;
constexpr uint32_t kRunning = 0;
constexpr uint32_t kSuccess = 1;
constexpr uint32_t kIterationLimit = 2;
constexpr uint32_t kCapacityOverflow = 3;
constexpr uint32_t kResourceExhausted = 4;
constexpr uint32_t kClaiming = 0xfffffffeU;
constexpr uint32_t kInputRows = 4;
constexpr uint32_t kOutputRows = 5;
constexpr uint32_t kScheduleDescriptor = 6;
constexpr uint32_t kHashWorkspace = 7;
constexpr uint32_t kSourceSlot = 1;
constexpr uint32_t kPermanentSlot = 2;
constexpr uint32_t kDefinedSlot = 4;
constexpr uint32_t kOpUnit = 0;
constexpr uint32_t kOpScan = 1;
constexpr uint32_t kOpFilter = 2;
constexpr uint32_t kOpProject = 3;
constexpr uint32_t kOpJoinInner = 4;
constexpr uint32_t kOpJoinSemi = 5;
constexpr uint32_t kOpUnion = 6;
constexpr uint32_t kOpDiff = 7;
constexpr uint32_t kOpTestStatus = 8;
constexpr uint32_t kOpTraceDelta = 9;
constexpr uint32_t kOpTraceSemanticGuard = 1;
constexpr uint32_t kOpMarkNovelty = 1;
constexpr uint32_t kOpMarkSchemaWinner = 2;
constexpr uint32_t kRegionInitialize = 1;
constexpr uint32_t kRegionSccBegin = 2;
constexpr uint32_t kRegionRecursive = 4;
constexpr uint32_t kRegionFinalize = 8;
constexpr uint64_t kTagBit = 1ULL << 63;
constexpr uint32_t kSetReferenceTagBit = 1U << 31;
constexpr uint32_t kSetReferenceTileSize = 1024;

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

struct __align__(16) ResidentRelationSlot {
    ResidentRelationView relation;
    uint32_t generation;
    uint32_t flags;
    uint32_t initial_count;
    uint32_t schema_tag;
};

struct ResidentOpDescriptor {
    uint32_t kind;
    uint32_t flags;
    uint32_t op_id;
    uint32_t out;
    uint32_t in0;
    uint32_t in1;
    uint32_t in0_generation;
    uint32_t in1_generation;
    uint32_t out_generation;
    uint32_t aux_offset;
    uint32_t aux_count;
    uint32_t left_key;
    uint32_t right_key;
    uint32_t scan_delta;
    uint32_t filter_delta;
    uint32_t schema_winner_head;
    uint32_t schema_winner_id;
    uint32_t reserved;
};

struct ResidentWaveDescriptor {
    uint32_t first_op;
    uint32_t op_count;
    uint32_t flags;
    uint32_t reserved;
};

struct ResidentRegionDescriptor {
    uint32_t first_wave;
    uint32_t wave_count;
    uint32_t iteration_limit;
    uint32_t op_id;
    uint32_t flags;
    uint32_t first_slot;
    uint32_t slot_count;
    uint32_t generation_offset;
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

struct __align__(8) ResidentProjectExpressionDescriptor {
    uint32_t kind;
    uint32_t column;
    uint32_t width;
    uint32_t reserved;
    uint64_t constant;
};

struct __align__(16) ResidentScheduleHeader {
    uint64_t slots;
    uint64_t ops;
    uint64_t waves;
    uint64_t regions;
    uint64_t generation_metadata;
    uint64_t filter_comparisons;
    uint64_t project_expressions;
    uint64_t filter_mask;
    uint64_t filter_prefix;
    uint64_t filter_block_sums;
    uint64_t filter_block_offsets;
    uint64_t set_slots;
    uint64_t set_required;
    uint64_t join_buckets;
    uint64_t join_next;
    uint64_t join_required;
    uint64_t status;
    uint64_t changed;
    uint64_t iterations;
    uint64_t scan_trace;
    uint64_t filter_trace;
    uint64_t semantic_scan_trace;
    uint64_t semantic_filter_trace;
    uint64_t schema_seen_nonempty;
    uint64_t schema_winner_ids;
    uint64_t receipt_table;
    uint64_t receipt_bytes;
    uint32_t slot_count;
    uint32_t op_count;
    uint32_t wave_count;
    uint32_t region_count;
    uint32_t filter_comparison_count;
    uint32_t project_expression_count;
    uint32_t filter_capacity;
    uint32_t filter_block_count;
    uint32_t set_slot_mask;
    uint32_t set_candidate_capacity;
    uint32_t join_bucket_mask;
    uint32_t join_right_capacity;
    uint32_t schema_winner_count;
    uint32_t receipt_count;
    uint32_t receipt_byte_count;
    uint32_t generation_metadata_count;
    uint32_t abi_version;
    uint32_t reserved;
};

static_assert(sizeof(ResidentRelationView) == 224, "ResidentRelationView size");
static_assert(alignof(ResidentRelationView) == 8, "ResidentRelationView alignment");
static_assert(offsetof(ResidentRelationView, columns) == 0, "ResidentRelationView columns");
static_assert(offsetof(ResidentRelationView, widths) == 136, "ResidentRelationView widths");
static_assert(offsetof(ResidentRelationView, arity) == 204, "ResidentRelationView arity");
static_assert(offsetof(ResidentRelationView, capacity) == 208, "ResidentRelationView capacity");
static_assert(offsetof(ResidentRelationView, reserved) == 212, "ResidentRelationView reserved");
static_assert(offsetof(ResidentRelationView, num_rows) == 216, "ResidentRelationView num_rows");
static_assert(sizeof(ResidentRelationSlot) == 240, "ResidentRelationSlot size");
static_assert(alignof(ResidentRelationSlot) == 16, "ResidentRelationSlot alignment");
static_assert(offsetof(ResidentRelationSlot, relation) == 0, "ResidentRelationSlot relation");
static_assert(offsetof(ResidentRelationSlot, generation) == 224, "ResidentRelationSlot generation");
static_assert(offsetof(ResidentRelationSlot, flags) == 228, "ResidentRelationSlot flags");
static_assert(offsetof(ResidentRelationSlot, initial_count) == 232, "ResidentRelationSlot initial_count");
static_assert(offsetof(ResidentRelationSlot, schema_tag) == 236, "ResidentRelationSlot schema_tag");
static_assert(sizeof(ResidentOpDescriptor) == 72, "ResidentOpDescriptor size");
static_assert(alignof(ResidentOpDescriptor) == 4, "ResidentOpDescriptor alignment");
static_assert(offsetof(ResidentOpDescriptor, kind) == 0, "ResidentOpDescriptor kind");
static_assert(offsetof(ResidentOpDescriptor, flags) == 4, "ResidentOpDescriptor flags");
static_assert(offsetof(ResidentOpDescriptor, op_id) == 8, "ResidentOpDescriptor op_id");
static_assert(offsetof(ResidentOpDescriptor, out) == 12, "ResidentOpDescriptor out");
static_assert(offsetof(ResidentOpDescriptor, in0) == 16, "ResidentOpDescriptor in0");
static_assert(offsetof(ResidentOpDescriptor, in1) == 20, "ResidentOpDescriptor in1");
static_assert(offsetof(ResidentOpDescriptor, in0_generation) == 24, "ResidentOpDescriptor in0_generation");
static_assert(offsetof(ResidentOpDescriptor, in1_generation) == 28, "ResidentOpDescriptor in1_generation");
static_assert(offsetof(ResidentOpDescriptor, out_generation) == 32, "ResidentOpDescriptor out_generation");
static_assert(offsetof(ResidentOpDescriptor, aux_offset) == 36, "ResidentOpDescriptor aux_offset");
static_assert(offsetof(ResidentOpDescriptor, aux_count) == 40, "ResidentOpDescriptor aux_count");
static_assert(offsetof(ResidentOpDescriptor, left_key) == 44, "ResidentOpDescriptor left_key");
static_assert(offsetof(ResidentOpDescriptor, right_key) == 48, "ResidentOpDescriptor right_key");
static_assert(offsetof(ResidentOpDescriptor, scan_delta) == 52, "ResidentOpDescriptor scan_delta");
static_assert(offsetof(ResidentOpDescriptor, filter_delta) == 56, "ResidentOpDescriptor filter_delta");
static_assert(offsetof(ResidentOpDescriptor, schema_winner_head) == 60, "ResidentOpDescriptor schema_winner_head");
static_assert(offsetof(ResidentOpDescriptor, schema_winner_id) == 64, "ResidentOpDescriptor schema_winner_id");
static_assert(offsetof(ResidentOpDescriptor, reserved) == 68, "ResidentOpDescriptor reserved");
static_assert(sizeof(ResidentWaveDescriptor) == 16, "ResidentWaveDescriptor size");
static_assert(alignof(ResidentWaveDescriptor) == 4, "ResidentWaveDescriptor alignment");
static_assert(offsetof(ResidentWaveDescriptor, first_op) == 0, "ResidentWaveDescriptor first_op");
static_assert(offsetof(ResidentWaveDescriptor, op_count) == 4, "ResidentWaveDescriptor op_count");
static_assert(offsetof(ResidentWaveDescriptor, flags) == 8, "ResidentWaveDescriptor flags");
static_assert(offsetof(ResidentWaveDescriptor, reserved) == 12, "ResidentWaveDescriptor reserved");
static_assert(sizeof(ResidentRegionDescriptor) == 32, "ResidentRegionDescriptor size");
static_assert(alignof(ResidentRegionDescriptor) == 4, "ResidentRegionDescriptor alignment");
static_assert(offsetof(ResidentRegionDescriptor, first_wave) == 0, "ResidentRegionDescriptor first_wave");
static_assert(offsetof(ResidentRegionDescriptor, wave_count) == 4, "ResidentRegionDescriptor wave_count");
static_assert(offsetof(ResidentRegionDescriptor, iteration_limit) == 8, "ResidentRegionDescriptor iteration_limit");
static_assert(offsetof(ResidentRegionDescriptor, op_id) == 12, "ResidentRegionDescriptor op_id");
static_assert(offsetof(ResidentRegionDescriptor, flags) == 16, "ResidentRegionDescriptor flags");
static_assert(offsetof(ResidentRegionDescriptor, first_slot) == 20, "ResidentRegionDescriptor first_slot");
static_assert(offsetof(ResidentRegionDescriptor, slot_count) == 24, "ResidentRegionDescriptor slot_count");
static_assert(offsetof(ResidentRegionDescriptor, generation_offset) == 28, "ResidentRegionDescriptor generation_offset");
static_assert(sizeof(ResidentScheduleHeader) == 288, "ResidentScheduleHeader size");
static_assert(alignof(ResidentScheduleHeader) == 16, "ResidentScheduleHeader alignment");
static_assert(offsetof(ResidentScheduleHeader, slots) == 0, "ResidentScheduleHeader slots");
static_assert(offsetof(ResidentScheduleHeader, generation_metadata) == 32, "ResidentScheduleHeader generation_metadata");
static_assert(offsetof(ResidentScheduleHeader, semantic_scan_trace) == 168, "ResidentScheduleHeader semantic_scan_trace");
static_assert(offsetof(ResidentScheduleHeader, semantic_filter_trace) == 176, "ResidentScheduleHeader semantic_filter_trace");
static_assert(offsetof(ResidentScheduleHeader, schema_seen_nonempty) == 184, "ResidentScheduleHeader schema_seen_nonempty");
static_assert(offsetof(ResidentScheduleHeader, schema_winner_ids) == 192, "ResidentScheduleHeader schema_winner_ids");
static_assert(offsetof(ResidentScheduleHeader, receipt_bytes) == 208, "ResidentScheduleHeader receipt_bytes");
static_assert(offsetof(ResidentScheduleHeader, slot_count) == 216, "ResidentScheduleHeader slot_count");
static_assert(offsetof(ResidentScheduleHeader, schema_winner_count) == 264, "ResidentScheduleHeader schema_winner_count");
static_assert(offsetof(ResidentScheduleHeader, generation_metadata_count) == 276, "ResidentScheduleHeader generation_metadata_count");
static_assert(offsetof(ResidentScheduleHeader, abi_version) == 280, "ResidentScheduleHeader abi_version");
static_assert(offsetof(ResidentScheduleHeader, reserved) == 284, "ResidentScheduleHeader reserved");

static_assert(sizeof(ResidentFilterComparisonDescriptor) == 48,
              "ResidentFilterComparisonDescriptor size");
static_assert(offsetof(ResidentFilterComparisonDescriptor, left_constant) == 32,
              "ResidentFilterComparisonDescriptor left_constant");
static_assert(offsetof(ResidentFilterComparisonDescriptor, right_constant) == 40,
              "ResidentFilterComparisonDescriptor right_constant");
static_assert(sizeof(ResidentProjectExpressionDescriptor) == 24,
              "ResidentProjectExpressionDescriptor size");
static_assert(offsetof(ResidentProjectExpressionDescriptor, constant) == 16,
              "ResidentProjectExpressionDescriptor constant");

template <typename T>
__device__ __forceinline__ T *device_ptr(uint64_t address) {
    return reinterpret_cast<T *>(address);
}

__device__ __forceinline__ uint32_t global_rank() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ uint32_t global_stride() {
    return gridDim.x * blockDim.x;
}

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
    const uint32_t rows = *device_ptr<const uint32_t>(relation.num_rows);
    if (rows > relation.capacity) {
        publish_terminal(status, kResourceExhausted, op_id, kInputRows, rows, relation.capacity);
    }
    return rows < relation.capacity ? rows : relation.capacity;
}

__device__ __forceinline__ uint64_t value_at(
    ResidentRelationView relation,
    uint32_t column,
    uint32_t row
) {
    if (relation.widths[column] == 4) {
        return device_ptr<const uint32_t>(relation.columns[column])[row];
    }
    return device_ptr<const uint64_t>(relation.columns[column])[row];
}

__device__ __forceinline__ void write_value(
    ResidentRelationView relation,
    uint32_t column,
    uint32_t row,
    uint64_t value
) {
    if (relation.widths[column] == 4) {
        device_ptr<uint32_t>(relation.columns[column])[row] = static_cast<uint32_t>(value);
    } else {
        device_ptr<uint64_t>(relation.columns[column])[row] = value;
    }
}

__device__ __forceinline__ void copy_row(
    ResidentRelationView source,
    uint32_t source_row,
    ResidentRelationView output,
    uint32_t output_row,
    uint32_t output_column_offset
) {
    for (uint32_t column = 0; column < source.arity; ++column) {
        write_value(output, output_column_offset + column, output_row,
                    value_at(source, column, source_row));
    }
}

__device__ __forceinline__ bool rows_equal(
    ResidentRelationView left,
    uint32_t left_row,
    ResidentRelationView right,
    uint32_t right_row
) {
    if (left.arity != right.arity) return false;
    for (uint32_t column = 0; column < left.arity; ++column) {
        if (left.widths[column] != right.widths[column] ||
            value_at(left, column, left_row) != value_at(right, column, right_row)) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ int32_t compare_set_references(
    uint32_t left_encoded,
    uint32_t right_encoded,
    ResidentRelationView relation_zero,
    ResidentRelationView relation_one
) {
    if (left_encoded == right_encoded) return 0;
    if (left_encoded == 0) return 1;
    if (right_encoded == 0) return -1;

    const uint32_t left_tag = (left_encoded & kSetReferenceTagBit) != 0;
    const uint32_t right_tag = (right_encoded & kSetReferenceTagBit) != 0;
    const uint32_t left_row = (left_encoded & ~kSetReferenceTagBit) - 1;
    const uint32_t right_row = (right_encoded & ~kSetReferenceTagBit) - 1;
    const ResidentRelationView left = left_tag ? relation_one : relation_zero;
    const ResidentRelationView right = right_tag ? relation_one : relation_zero;
    for (uint32_t column = 0; column < left.arity; ++column) {
        const uint64_t left_value = value_at(left, column, left_row);
        const uint64_t right_value = value_at(right, column, right_row);
        if (left_value < right_value) return -1;
        if (left_value > right_value) return 1;
    }
    if (left_tag < right_tag) return -1;
    if (left_tag > right_tag) return 1;
    return left_row < right_row ? -1 : 1;
}

__device__ void compact_set_winners_by_tile(
    cg::grid_group grid,
    const ResidentScheduleHeader *header,
    bool is_union,
    uint64_t *hash_slots,
    uint32_t *references,
    unsigned long long *required
) {
    __shared__ uint32_t scan[kBlockSize];
    __shared__ uint32_t block_base;
    const uint32_t lane = threadIdx.x;
    const uint32_t rank = global_rank();
    const uint32_t stride = global_stride();
    const uint64_t hash_slot_count = static_cast<uint64_t>(header->set_slot_mask) + 1;

    for (uint64_t tile = 0; tile < hash_slot_count; tile += stride) {
        const uint64_t slot_index = tile + rank;
        const uint64_t encoded = slot_index < hash_slot_count ? hash_slots[slot_index] : 0;
        const uint32_t source_tag = (encoded & kTagBit) != 0;
        const uint32_t eligible = encoded != 0 && (is_union || source_tag == 0);
        const uint32_t packed = eligible
            ? (source_tag ? kSetReferenceTagBit : 0) |
                (static_cast<uint32_t>((encoded & ~kTagBit) - 1) + 1)
            : 0;

        // Every hash entry in this tile is register-resident before compacted
        // u32 references begin overwriting the lower half of the u64 table.
        grid.sync();
        scan[lane] = eligible;
        __syncthreads();
        for (uint32_t offset = 1; offset < kBlockSize; offset <<= 1) {
            const uint32_t addend = lane >= offset ? scan[lane - offset] : 0;
            __syncthreads();
            scan[lane] += addend;
            __syncthreads();
        }
        const uint32_t local_prefix = scan[lane] - eligible;
        const uint32_t block_total = scan[kBlockSize - 1];
        if (lane == 0) {
            block_base = static_cast<uint32_t>(atomicAdd(
                required, static_cast<unsigned long long>(block_total)));
        }
        __syncthreads();
        if (eligible) references[block_base + local_prefix] = packed;
        grid.sync();
    }
}

__device__ void sort_set_reference_tiles(
    uint32_t *references,
    uint32_t count,
    ResidentRelationView relation_zero,
    ResidentRelationView relation_one
) {
    __shared__ uint32_t tile[kSetReferenceTileSize];
    const uint32_t lane = threadIdx.x;
    const uint32_t tile_count =
        (count + kSetReferenceTileSize - 1) / kSetReferenceTileSize;
    for (uint32_t tile_index = blockIdx.x; tile_index < tile_count;
         tile_index += gridDim.x) {
        const uint32_t tile_start = tile_index * kSetReferenceTileSize;
        const uint32_t tile_rows = count - tile_start < kSetReferenceTileSize
            ? count - tile_start
            : kSetReferenceTileSize;
        for (uint32_t offset = lane; offset < kSetReferenceTileSize;
             offset += blockDim.x) {
            tile[offset] = offset < tile_rows ? references[tile_start + offset] : 0;
        }
        __syncthreads();

        for (uint32_t width = 2; width <= kSetReferenceTileSize; width <<= 1) {
            for (uint32_t stride = width >> 1; stride > 0; stride >>= 1) {
                for (uint32_t offset = lane; offset < kSetReferenceTileSize;
                     offset += blockDim.x) {
                    const uint32_t partner = offset ^ stride;
                    if (partner <= offset) continue;
                    const uint32_t left = tile[offset];
                    const uint32_t right = tile[partner];
                    const int32_t comparison = compare_set_references(
                        left, right, relation_zero, relation_one);
                    const bool ascending = (offset & width) == 0;
                    const bool swap = ascending ? comparison > 0 : comparison < 0;
                    if (swap) {
                        tile[offset] = right;
                        tile[partner] = left;
                    }
                }
                __syncthreads();
            }
        }

        for (uint32_t offset = lane; offset < tile_rows; offset += blockDim.x) {
            references[tile_start + offset] = tile[offset];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ uint32_t merged_set_reference_at(
    const uint32_t *source,
    uint32_t left_start,
    uint32_t left_count,
    uint32_t right_count,
    uint32_t diagonal,
    ResidentRelationView relation_zero,
    ResidentRelationView relation_one
) {
    const uint32_t right_start = left_start + left_count;
    uint32_t low = diagonal > right_count ? diagonal - right_count : 0;
    uint32_t high = diagonal < left_count ? diagonal : left_count;
    while (low < high) {
        const uint32_t left_index = low + ((high - low) >> 1);
        const uint32_t right_index = diagonal - left_index;
        if (left_index < left_count && right_index > 0 &&
            compare_set_references(
                source[right_start + right_index - 1],
                source[left_start + left_index],
                relation_zero,
                relation_one) > 0) {
            low = left_index + 1;
        } else {
            high = left_index;
        }
    }
    const uint32_t left_index = low;
    const uint32_t right_index = diagonal - left_index;
    if (left_index < left_count &&
        (right_index >= right_count ||
         compare_set_references(
             source[left_start + left_index],
             source[right_start + right_index],
             relation_zero,
             relation_one) <= 0)) {
        return source[left_start + left_index];
    }
    return source[right_start + right_index];
}

__device__ uint32_t *merge_set_reference_runs(
    cg::grid_group grid,
    uint32_t *references_a,
    uint32_t *references_b,
    uint32_t count,
    ResidentRelationView relation_zero,
    ResidentRelationView relation_one
) {
    uint32_t *source = references_a;
    uint32_t *destination = references_b;
    for (uint32_t run_width = kSetReferenceTileSize; run_width < count;
         run_width <<= 1) {
        const uint32_t pair_width = run_width << 1;
        for (uint32_t output_index = global_rank(); output_index < count;
             output_index += global_stride()) {
            const uint32_t pair_start = (output_index / pair_width) * pair_width;
            const uint32_t left_count = count - pair_start < run_width
                ? count - pair_start
                : run_width;
            const uint32_t after_left = pair_start + left_count;
            const uint32_t right_count = count - after_left < run_width
                ? count - after_left
                : run_width;
            destination[output_index] = merged_set_reference_at(
                source,
                pair_start,
                left_count,
                right_count,
                output_index - pair_start,
                relation_zero,
                relation_one);
        }
        grid.sync();
        uint32_t *previous_source = source;
        source = destination;
        destination = previous_source;
    }
    return source;
}

__device__ __forceinline__ uint64_t row_hash(
    ResidentRelationView relation,
    uint32_t row
) {
    uint64_t hash = 1469598103934665603ULL;
    for (uint32_t column = 0; column < relation.arity; ++column) {
        const uint64_t value = value_at(relation, column, row);
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

__device__ __forceinline__ uint64_t key_hash(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    return key;
}

__device__ __forceinline__ bool compare_values(uint64_t left, uint32_t op, uint64_t right) {
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
        const uint64_t left = comparison.left_kind == 0
            ? value_at(input, comparison.left_column, row)
            : comparison.left_constant;
        const uint64_t right = comparison.right_kind == 0
            ? value_at(input, comparison.right_column, row)
            : comparison.right_constant;
        if (!compare_values(left, comparison.op, right)) return false;
    }
    return true;
}

__device__ __forceinline__ bool same_layout(
    ResidentRelationView left,
    ResidentRelationView right
) {
    if (left.arity != right.arity) return false;
    for (uint32_t column = 0; column < left.arity; ++column) {
        if (left.widths[column] != right.widths[column]) return false;
    }
    return true;
}

__device__ void execute_filter(
    cg::grid_group grid,
    const ResidentScheduleHeader *header,
    const ResidentOpDescriptor &op,
    ResidentRelationView input,
    ResidentRelationView output,
    ResidentTerminalStatus *status
) {
    uint32_t *mask = device_ptr<uint32_t>(header->filter_mask);
    uint32_t *prefix = device_ptr<uint32_t>(header->filter_prefix);
    uint32_t *block_sums = device_ptr<uint32_t>(header->filter_block_sums);
    uint32_t *block_offsets = device_ptr<uint32_t>(header->filter_block_offsets);
    const ResidentFilterComparisonDescriptor *comparisons =
        device_ptr<const ResidentFilterComparisonDescriptor>(header->filter_comparisons) +
        op.aux_offset;
    const uint32_t block_count = (input.capacity + kBlockSize - 1) / kBlockSize;
    if (global_rank() == 0 && status->code == kRunning) {
        *device_ptr<uint32_t>(output.num_rows) = 0;
    }
    grid.sync();

    __shared__ uint32_t scan[kBlockSize];
    const uint32_t lane = threadIdx.x;
    for (uint32_t tile = blockIdx.x; tile < block_count; tile += gridDim.x) {
        const uint32_t row = tile * kBlockSize + lane;
        const uint32_t rows = status->code == kRunning
            ? bounded_rows(input, status, op.op_id)
            : 0;
        const uint32_t keep = row < rows && status->code == kRunning &&
            matches_all(input, comparisons, op.aux_count, row);
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
        if (lane == kBlockSize - 1) block_sums[tile] = scan[lane];
        __syncthreads();
    }
    grid.sync();

    if (global_rank() == 0 && status->code == kRunning) {
        uint32_t running = 0;
        for (uint32_t block = 0; block < block_count; ++block) {
            block_offsets[block] = running;
            running += block_sums[block];
        }
    }
    grid.sync();

    for (uint32_t row = global_rank(); row < input.capacity; row += global_stride()) {
        if (status->code == kRunning) {
            prefix[row] += block_offsets[row / kBlockSize];
        }
    }
    grid.sync();

    if (global_rank() == 0 && status->code == kRunning) {
        const uint32_t rows = bounded_rows(input, status, op.op_id);
        const uint64_t required = rows == 0 ? 0 :
            static_cast<uint64_t>(prefix[rows - 1]) + mask[rows - 1];
        if (required > output.capacity) {
            publish_terminal(status, kCapacityOverflow, op.op_id, kOutputRows,
                             required, output.capacity);
        } else {
            *device_ptr<uint32_t>(output.num_rows) = static_cast<uint32_t>(required);
        }
    }
    grid.sync();

    const uint32_t rows = status->code == kRunning
        ? bounded_rows(input, status, op.op_id)
        : 0;
    for (uint32_t row = global_rank(); row < rows; row += global_stride()) {
        if (status->code != kRunning || mask[row] == 0) continue;
        const uint32_t destination = prefix[row];
        if (destination < output.capacity) copy_row(input, row, output, destination, 0);
    }
    grid.sync();
}

__device__ void execute_project(
    cg::grid_group grid,
    const ResidentScheduleHeader *header,
    const ResidentOpDescriptor &op,
    ResidentRelationView input,
    ResidentRelationView output,
    ResidentTerminalStatus *status
) {
    if (global_rank() == 0 && status->code == kRunning) {
        *device_ptr<uint32_t>(output.num_rows) = 0;
    }
    grid.sync();
    if (global_rank() == 0 && status->code == kRunning) {
        const uint32_t rows = bounded_rows(input, status, op.op_id);
        if (rows > output.capacity) {
            publish_terminal(status, kCapacityOverflow, op.op_id, kOutputRows,
                             rows, output.capacity);
        } else {
            *device_ptr<uint32_t>(output.num_rows) = rows;
        }
    }
    grid.sync();
    const ResidentProjectExpressionDescriptor *expressions =
        device_ptr<const ResidentProjectExpressionDescriptor>(header->project_expressions) +
        op.aux_offset;
    const uint32_t rows = status->code == kRunning
        ? bounded_rows(input, status, op.op_id)
        : 0;
    for (uint32_t row = global_rank(); row < rows; row += global_stride()) {
        if (status->code != kRunning || row >= output.capacity) continue;
        for (uint32_t column = 0; column < op.aux_count; ++column) {
            const ResidentProjectExpressionDescriptor expression = expressions[column];
            const uint64_t value = expression.kind == 0
                ? value_at(input, expression.column, row)
                : expression.constant;
            write_value(output, column, row, value);
        }
    }
    grid.sync();
}

__device__ bool insert_set_row(
    ResidentRelationView candidate,
    uint32_t row,
    ResidentRelationView relation_zero,
    ResidentRelationView relation_one,
    uint32_t source_tag,
    bool emit,
    uint64_t *slots,
    uint32_t slot_mask,
    unsigned long long *required,
    ResidentTerminalStatus *status,
    uint32_t op_id
) {
    const uint64_t encoded = (source_tag ? kTagBit : 0) | (static_cast<uint64_t>(row) + 1);
    uint32_t slot = static_cast<uint32_t>(row_hash(candidate, row)) & slot_mask;
    for (uint32_t probe = 0; probe <= slot_mask; ++probe) {
        const uint64_t previous = atomicCAS(
            reinterpret_cast<unsigned long long *>(&slots[slot]), 0ULL, encoded);
        if (previous == 0) {
            if (emit) {
                atomicAdd(required, 1ULL);
            }
            return true;
        }
        const uint32_t existing_tag = (previous & kTagBit) != 0;
        const uint32_t existing_row = static_cast<uint32_t>((previous & ~kTagBit) - 1);
        const ResidentRelationView existing = existing_tag ? relation_one : relation_zero;
        if (rows_equal(candidate, row, existing, existing_row)) return true;
        slot = (slot + 1) & slot_mask;
    }
    publish_terminal(status, kResourceExhausted, op_id, kHashWorkspace,
                     static_cast<uint64_t>(slot_mask) + 2,
                     static_cast<uint64_t>(slot_mask) + 1);
    return false;
}

__device__ void execute_set(
    cg::grid_group grid,
    const ResidentScheduleHeader *header,
    const ResidentOpDescriptor &op,
    ResidentRelationView left,
    ResidentRelationView right,
    ResidentRelationView output,
    ResidentTerminalStatus *status,
    bool is_union
) {
    uint64_t *slots = device_ptr<uint64_t>(header->set_slots);
    unsigned long long *required = device_ptr<unsigned long long>(header->set_required);
    for (uint32_t slot = global_rank(); slot <= header->set_slot_mask; slot += global_stride()) {
        if (status->code == kRunning) slots[slot] = 0;
    }
    if (global_rank() == 0 && status->code == kRunning) {
        *required = 0;
        *device_ptr<uint32_t>(output.num_rows) = 0;
    }
    grid.sync();

    const ResidentRelationView first = is_union ? left : right;
    const ResidentRelationView second = is_union ? right : left;
    const uint32_t first_tag = is_union ? 0 : 1;
    const uint32_t second_tag = is_union ? 1 : 0;
    const bool first_emit = is_union;
    const bool second_emit = true;
    const uint32_t first_rows = status->code == kRunning
        ? bounded_rows(first, status, op.op_id)
        : 0;
    for (uint32_t row = global_rank(); row < first_rows; row += global_stride()) {
        if (status->code == kRunning) {
            insert_set_row(first, row, left, right, first_tag, first_emit, slots,
                           header->set_slot_mask, required, status, op.op_id);
        }
    }
    grid.sync();

    const uint32_t second_rows = status->code == kRunning
        ? bounded_rows(second, status, op.op_id)
        : 0;
    for (uint32_t row = global_rank(); row < second_rows; row += global_stride()) {
        if (status->code == kRunning) {
            insert_set_row(second, row, left, right, second_tag, second_emit, slots,
                           header->set_slot_mask, required, status, op.op_id);
        }
    }
    grid.sync();

    const uint64_t emitted_count_u64 = *required;
    if (global_rank() == 0 && status->code == kRunning) {
        if (emitted_count_u64 > output.capacity ||
            emitted_count_u64 > header->set_candidate_capacity) {
            publish_terminal(status, kCapacityOverflow, op.op_id, kOutputRows,
                             emitted_count_u64, output.capacity);
        } else {
            *device_ptr<uint32_t>(output.num_rows) =
                static_cast<uint32_t>(emitted_count_u64);
        }
    }
    grid.sync();

    if (status->code != kRunning || emitted_count_u64 == 0) {
        if (global_rank() == 0) *required = 0;
        grid.sync();
        return;
    }

    const uint32_t count = static_cast<uint32_t>(emitted_count_u64);
    uint32_t *references_a = reinterpret_cast<uint32_t *>(slots);
    uint32_t *references_b = references_a + header->set_candidate_capacity;
    if (global_rank() == 0) *required = 0;
    grid.sync();
    compact_set_winners_by_tile(
        grid, header, is_union, slots, references_a, required);
    if (global_rank() == 0 && *required != emitted_count_u64) {
        *device_ptr<uint32_t>(output.num_rows) = 0;
        publish_terminal(status, kResourceExhausted, op.op_id, kHashWorkspace,
                         emitted_count_u64, *required);
    }
    grid.sync();
    if (status->code != kRunning) {
        if (global_rank() == 0) *required = 0;
        grid.sync();
        return;
    }

    sort_set_reference_tiles(references_a, count, left, right);
    grid.sync();
    uint32_t *ordered_references = merge_set_reference_runs(
        grid, references_a, references_b, count, left, right);
    for (uint32_t destination = global_rank(); destination < count;
         destination += global_stride()) {
        const uint32_t encoded = ordered_references[destination];
        const uint32_t source_tag = (encoded & kSetReferenceTagBit) != 0;
        const uint32_t row = (encoded & ~kSetReferenceTagBit) - 1;
        const ResidentRelationView source = source_tag ? right : left;
        copy_row(source, row, output, destination, 0);
    }
    grid.sync();
    if (global_rank() == 0) *required = 0;
    grid.sync();
}

__device__ void execute_join(
    cg::grid_group grid,
    const ResidentScheduleHeader *header,
    const ResidentOpDescriptor &op,
    ResidentRelationView left,
    ResidentRelationView right,
    ResidentRelationView output,
    ResidentTerminalStatus *status,
    bool semi
) {
    uint32_t *buckets = device_ptr<uint32_t>(header->join_buckets);
    uint32_t *next = device_ptr<uint32_t>(header->join_next);
    unsigned long long *required = device_ptr<unsigned long long>(header->join_required);
    for (uint32_t bucket = global_rank(); bucket <= header->join_bucket_mask;
         bucket += global_stride()) {
        if (status->code == kRunning) buckets[bucket] = 0xffffffffU;
    }
    for (uint32_t row = global_rank(); row < header->join_right_capacity;
         row += global_stride()) {
        if (status->code == kRunning) next[row] = 0xffffffffU;
    }
    if (global_rank() == 0 && status->code == kRunning) {
        *required = 0;
        *device_ptr<uint32_t>(output.num_rows) = 0;
    }
    grid.sync();

    const uint32_t right_rows = status->code == kRunning
        ? bounded_rows(right, status, op.op_id)
        : 0;
    for (uint32_t row = global_rank(); row < right_rows; row += global_stride()) {
        if (status->code != kRunning) continue;
        const uint32_t bucket = static_cast<uint32_t>(key_hash(value_at(right, op.right_key, row))) &
            header->join_bucket_mask;
        next[row] = atomicExch(&buckets[bucket], row);
    }
    grid.sync();

    const uint32_t left_rows = status->code == kRunning
        ? bounded_rows(left, status, op.op_id)
        : 0;
    for (uint32_t left_row = global_rank(); left_row < left_rows;
         left_row += global_stride()) {
        if (status->code != kRunning) continue;
        const uint64_t key = value_at(left, op.left_key, left_row);
        const uint32_t bucket = static_cast<uint32_t>(key_hash(key)) & header->join_bucket_mask;
        for (uint32_t right_row = buckets[bucket]; right_row != 0xffffffffU;
             right_row = next[right_row]) {
            if (key != value_at(right, op.right_key, right_row)) continue;
            atomicAdd(required, 1ULL);
            if (semi) break;
        }
    }
    grid.sync();

    if (global_rank() == 0 && status->code == kRunning) {
        const uint64_t count = *required;
        if (count > output.capacity) {
            publish_terminal(status, kCapacityOverflow, op.op_id, kOutputRows,
                             count, output.capacity);
        } else {
            *device_ptr<uint32_t>(output.num_rows) = static_cast<uint32_t>(count);
            *required = 0;
        }
    }
    grid.sync();

    for (uint32_t left_row = global_rank(); left_row < left_rows;
         left_row += global_stride()) {
        if (status->code != kRunning) continue;
        const uint64_t key = value_at(left, op.left_key, left_row);
        const uint32_t bucket = static_cast<uint32_t>(key_hash(key)) & header->join_bucket_mask;
        for (uint32_t right_row = buckets[bucket]; right_row != 0xffffffffU;
             right_row = next[right_row]) {
            if (key != value_at(right, op.right_key, right_row)) continue;
            const uint64_t destination = atomicAdd(required, 1ULL);
            if (destination < output.capacity) {
                copy_row(left, left_row, output, static_cast<uint32_t>(destination), 0);
                if (!semi) {
                    copy_row(right, right_row, output, static_cast<uint32_t>(destination),
                             left.arity);
                }
            }
            if (semi) break;
        }
    }
    grid.sync();
}

__device__ bool op_uses_second_input(uint32_t kind) {
    return kind == kOpJoinInner || kind == kOpJoinSemi ||
           kind == kOpUnion || kind == kOpDiff;
}

__device__ bool input_is_ready(
    const ResidentRelationSlot &input,
    uint32_t expected_generation
) {
    return (input.flags & kDefinedSlot) != 0 &&
        input.generation == expected_generation;
}

__device__ bool output_generation_is_valid(
    const ResidentRelationSlot &output,
    uint32_t expected_generation
) {
    return (output.flags & kSourceSlot) == 0 &&
        (output.generation == expected_generation ||
         (output.generation != UINT32_MAX &&
          output.generation + 1 == expected_generation));
}

__device__ bool valid_trace_delta(const ResidentOpDescriptor &op) {
    const bool has_semantic_guard = (op.flags & kOpTraceSemanticGuard) != 0;
    return op.kind == kOpTraceDelta &&
        (op.flags & ~kOpTraceSemanticGuard) == 0 && op.op_id == 0 &&
        op.out == 0 && op.in1 == 0 &&
        (has_semantic_guard || (op.in0 == 0 && op.in0_generation == 0)) &&
        op.in1_generation == 0 &&
        op.out_generation == 0 && op.aux_offset == 0 && op.aux_count == 0 &&
        op.left_key == 0 && op.right_key == 0 &&
        op.schema_winner_head == 0 && op.schema_winner_id == 0 &&
        op.reserved == 0;
}

__device__ bool checked_receipt_head_count(
    const ResidentScheduleHeader *header,
    uint32_t *head_count
) {
    const bool receipt_shape_valid = header->receipt_count >= 4 &&
        ((header->receipt_count - 4) & 1U) == 0 &&
        header->receipt_byte_count == sizeof(ResidentTerminalStatus) +
            sizeof(uint32_t) * (1 + header->receipt_count);
    if (!receipt_shape_valid) return false;
    *head_count = (header->receipt_count - 4) / 2;
    if (header->schema_winner_count != *head_count ||
        header->generation_metadata_count < *head_count) return false;
    return true;
}

__device__ bool validate_operation(
    const ResidentScheduleHeader *header,
    const ResidentOpDescriptor &op,
    const ResidentRegionDescriptor &region,
    const ResidentRelationSlot *slots,
    ResidentTerminalStatus *status
) {
    if (op.kind == kOpTraceDelta) {
        if (!valid_trace_delta(op)) return false;
        if ((op.flags & kOpTraceSemanticGuard) == 0) return true;
        const uint64_t slot_end = static_cast<uint64_t>(region.first_slot) +
            region.slot_count;
        return op.in0 < header->slot_count && op.in0 >= region.first_slot &&
            static_cast<uint64_t>(op.in0) < slot_end &&
            input_is_ready(slots[op.in0], op.in0_generation);
    }
    if (op.kind == kOpTestStatus) {
        return op.flags == 0 && op.in1_generation == 0 && op.right_key == 0 &&
            op.scan_delta == 0 && op.filter_delta == 0 &&
            op.schema_winner_head == 0 && op.schema_winner_id == 0 &&
            op.reserved == 0;
    }
    const bool marks_novelty = (op.flags & kOpMarkNovelty) != 0;
    const bool marks_schema_winner = (op.flags & kOpMarkSchemaWinner) != 0;
    uint32_t head_count = 0;
    const bool winner_encoding_valid = marks_schema_winner
        ? checked_receipt_head_count(header, &head_count) &&
            op.schema_winner_head < head_count &&
            header->schema_seen_nonempty != 0 && header->schema_winner_ids != 0
        : op.schema_winner_head == 0 && op.schema_winner_id == 0;
    if (op.kind > kOpDiff || op.reserved != 0 ||
        op.scan_delta != 0 || op.filter_delta != 0 ||
        (op.flags & ~(kOpMarkNovelty | kOpMarkSchemaWinner)) != 0 ||
        (marks_novelty &&
         ((op.kind != kOpDiff && op.kind != kOpProject) ||
          region.flags != kRegionRecursive))) {
        publish_terminal(status, kResourceExhausted, op.op_id, kScheduleDescriptor,
                         op.flags, kOpMarkNovelty | kOpMarkSchemaWinner);
        return false;
    }
    if (!winner_encoding_valid ||
        (marks_schema_winner && op.schema_winner_head >= head_count)) {
        publish_terminal(status, kResourceExhausted, op.op_id, kScheduleDescriptor,
                         op.schema_winner_head, head_count);
        return false;
    }
    const uint32_t region_slot_end = region.first_slot + region.slot_count;
    if (op.kind == kOpUnit) {
        if (op.out >= header->slot_count || op.out < region.first_slot ||
            op.out >= region_slot_end) {
            publish_terminal(status, kResourceExhausted, op.op_id, kScheduleDescriptor,
                             header->slot_count + 1ULL, header->slot_count);
            return false;
        }
        const ResidentRelationSlot output = slots[op.out];
        if (!output_generation_is_valid(output, op.out_generation) ||
            output.relation.arity != 0 ||
            output.relation.capacity > 65536 ||
            op.in0 != 0 || op.in1 != 0 ||
            op.in0_generation != 0 || op.in1_generation != 0 ||
            op.aux_offset != 0 || op.aux_count != 0 ||
            op.left_key != 0 || op.right_key != 0) {
            publish_terminal(status, kResourceExhausted, op.op_id, kScheduleDescriptor,
                             1, 0);
            return false;
        }
        return true;
    }
    if (op.kind == kOpScan) {
        if (op.in0 >= header->slot_count || op.out != op.in0 ||
            op.out < region.first_slot || op.out >= region_slot_end) {
            publish_terminal(status, kResourceExhausted, op.op_id, kScheduleDescriptor,
                             header->slot_count + 1ULL, header->slot_count);
            return false;
        }
        const ResidentRelationSlot source = slots[op.in0];
        if (!input_is_ready(source, op.in0_generation) ||
            source.generation != op.out_generation ||
            source.relation.arity > kMaxArity || source.relation.capacity > 65536 ||
            op.in1 != 0 || op.in1_generation != 0 ||
            op.aux_offset != 0 || op.aux_count != 0 ||
            op.left_key != 0 || op.right_key != 0) {
            publish_terminal(status, kResourceExhausted, op.op_id, kScheduleDescriptor,
                             1, 0);
            return false;
        }
        return true;
    }
    if (op.out >= header->slot_count || op.in0 >= header->slot_count ||
        (op_uses_second_input(op.kind) && op.in1 >= header->slot_count) ||
        op.out < region.first_slot || op.out >= region_slot_end ||
        op.in0 < region.first_slot || op.in0 >= region_slot_end ||
        (op_uses_second_input(op.kind) &&
         (op.in1 < region.first_slot || op.in1 >= region_slot_end)) ||
        (op.kind != kOpScan && (op.out == op.in0 ||
         (op_uses_second_input(op.kind) && op.out == op.in1)))) {
        publish_terminal(status, kResourceExhausted, op.op_id, kScheduleDescriptor,
                         header->slot_count + 1ULL, header->slot_count);
        return false;
    }
    const ResidentRelationSlot output = slots[op.out];
    const ResidentRelationSlot input_zero = slots[op.in0];
    if (!output_generation_is_valid(output, op.out_generation) ||
        !input_is_ready(input_zero, op.in0_generation)) {
        publish_terminal(status, kResourceExhausted, op.op_id, kScheduleDescriptor,
                         1, 0);
        return false;
    }
    const ResidentRelationView out = output.relation;
    const ResidentRelationView in0 = input_zero.relation;
    if (out.arity > kMaxArity || in0.arity > kMaxArity ||
        out.capacity > 65536 || in0.capacity > 65536) {
        publish_terminal(status, kResourceExhausted, op.op_id, kScheduleDescriptor,
                         65537, 65536);
        return false;
    }
    if (op.kind == kOpFilter) {
        const bool filter_range_valid =
            op.aux_offset <= header->filter_comparison_count &&
            op.aux_count <= header->filter_comparison_count - op.aux_offset;
        if (op.in1 != 0 || op.in1_generation != 0 ||
            op.left_key != 0 || op.right_key != 0 ||
            input_zero.schema_tag != output.schema_tag ||
            !filter_range_valid ||
            in0.capacity > header->filter_capacity) {
            publish_terminal(status, kResourceExhausted, op.op_id, kScheduleDescriptor,
                             op.aux_count,
                             header->filter_comparison_count);
            return false;
        }
        const ResidentFilterComparisonDescriptor *comparisons =
            device_ptr<const ResidentFilterComparisonDescriptor>(
                header->filter_comparisons) + op.aux_offset;
        for (uint32_t index = 0; index < op.aux_count; ++index) {
            const ResidentFilterComparisonDescriptor comparison = comparisons[index];
            const bool left_column_invalid = comparison.left_kind == 0 &&
                (comparison.left_column >= in0.arity ||
                 comparison.width != in0.widths[comparison.left_column] ||
                 comparison.left_constant != 0);
            const bool right_column_invalid = comparison.right_kind == 0 &&
                (comparison.right_column >= in0.arity ||
                 comparison.width != in0.widths[comparison.right_column] ||
                 comparison.right_constant != 0);
            const bool constant_operand_invalid =
                (comparison.left_kind == 1 && comparison.left_column != 0) ||
                (comparison.right_kind == 1 && comparison.right_column != 0);
            if (comparison.left_kind > 1 || comparison.right_kind > 1 ||
                comparison.op > 5 ||
                (comparison.width != 4 && comparison.width != 8) ||
                comparison.reserved_zero != 0 || comparison.reserved_one != 0 ||
                left_column_invalid || right_column_invalid ||
                constant_operand_invalid) {
                publish_terminal(status, kResourceExhausted, op.op_id,
                                 kScheduleDescriptor, index + 1ULL, op.aux_count);
                return false;
            }
        }
    } else if (op.kind == kOpProject) {
        const bool project_range_valid =
            op.aux_offset <= header->project_expression_count &&
            op.aux_count <= header->project_expression_count - op.aux_offset;
        if (op.in1 != 0 || op.in1_generation != 0 ||
            op.left_key != 0 || op.right_key != 0 ||
            !project_range_valid ||
            op.aux_count != out.arity) {
            publish_terminal(status, kResourceExhausted, op.op_id, kScheduleDescriptor,
                             op.aux_count,
                             header->project_expression_count);
            return false;
        }
        const ResidentProjectExpressionDescriptor *expressions =
            device_ptr<const ResidentProjectExpressionDescriptor>(
                header->project_expressions) + op.aux_offset;
        for (uint32_t column = 0; column < op.aux_count; ++column) {
            const ResidentProjectExpressionDescriptor expression = expressions[column];
            const bool column_invalid = expression.kind == 0 &&
                (expression.column >= in0.arity ||
                 expression.width != in0.widths[expression.column] ||
                 expression.constant != 0);
            const bool constant_invalid =
                expression.kind == 1 && expression.column != 0;
            if (expression.kind > 1 || expression.reserved != 0 ||
                (expression.width != 4 && expression.width != 8) ||
                expression.width != out.widths[column] ||
                column_invalid || constant_invalid) {
                publish_terminal(status, kResourceExhausted, op.op_id,
                                 kScheduleDescriptor, column + 1ULL, op.aux_count);
                return false;
            }
        }
    }
    if (op_uses_second_input(op.kind)) {
        const ResidentRelationSlot input_one = slots[op.in1];
        if (!input_is_ready(input_one, op.in1_generation) ||
            input_one.relation.arity > kMaxArity || input_one.relation.capacity > 65536) {
            publish_terminal(status, kResourceExhausted, op.op_id, kScheduleDescriptor,
                             1, 0);
            return false;
        }
        const ResidentRelationView in1 = input_one.relation;
        if (op.kind == kOpUnion || op.kind == kOpDiff) {
            if (op.aux_offset != 0 || op.aux_count != 0 ||
                op.left_key != 0 || op.right_key != 0 ||
                static_cast<uint64_t>(in0.capacity) + in1.capacity >
                    header->set_candidate_capacity ||
                input_zero.schema_tag != input_one.schema_tag ||
                input_zero.schema_tag != output.schema_tag ||
                !same_layout(in0, in1) || !same_layout(in0, out)) {
                publish_terminal(status, kResourceExhausted, op.op_id,
                                 kScheduleDescriptor, 1, 0);
                return false;
            }
        } else {
            const uint32_t expected_arity = op.kind == kOpJoinSemi
                ? in0.arity
                : in0.arity + in1.arity;
            if (op.aux_offset != 0 || op.aux_count != 0 ||
                expected_arity > kMaxArity ||
                op.left_key >= in0.arity || op.right_key >= in1.arity ||
                in0.widths[op.left_key] != in1.widths[op.right_key] ||
                out.arity != expected_arity ||
                in1.capacity > header->join_right_capacity) {
                publish_terminal(status, kResourceExhausted, op.op_id,
                                 kScheduleDescriptor, expected_arity, out.arity);
                return false;
            }
            for (uint32_t column = 0; column < in0.arity; ++column) {
                if (out.widths[column] != in0.widths[column]) return false;
            }
            if (op.kind == kOpJoinInner) {
                for (uint32_t column = 0; column < in1.arity; ++column) {
                    if (out.widths[in0.arity + column] != in1.widths[column]) return false;
                }
            }
        }
    }
    return op.kind <= kOpDiff;
}

} // namespace

extern "C" __global__ void resident_schedule_execute(
    const ResidentScheduleHeader *header,
    uint32_t region_index,
    uint64_t conditional_handle
) {
    cg::grid_group grid = cg::this_grid();
    ResidentTerminalStatus *status = device_ptr<ResidentTerminalStatus>(header->status);
    ResidentRelationSlot *slots = device_ptr<ResidentRelationSlot>(header->slots);
    const ResidentOpDescriptor *ops = device_ptr<const ResidentOpDescriptor>(header->ops);
    const ResidentWaveDescriptor *waves = device_ptr<const ResidentWaveDescriptor>(header->waves);
    const ResidentRegionDescriptor *regions =
        device_ptr<const ResidentRegionDescriptor>(header->regions);
    ResidentRegionDescriptor region = {};
    if (region_index < header->region_count) region = regions[region_index];
    const uint32_t allowed_region_flags = kRegionInitialize | kRegionSccBegin |
        kRegionRecursive | kRegionFinalize;
    const bool initializes = (region.flags & kRegionInitialize) != 0;
    const bool begins_scc = (region.flags & kRegionSccBegin) != 0;
    const bool recursive = (region.flags & kRegionRecursive) != 0;
    const bool finalizes = (region.flags & kRegionFinalize) != 0;
    uint32_t head_count = 0;
    const bool winner_shape_valid = header->schema_winner_count == 0 ||
        (checked_receipt_head_count(header, &head_count) &&
         header->schema_seen_nonempty != 0 && header->schema_winner_ids != 0);
    const bool generation_metadata_shape_valid = winner_shape_valid &&
        header->generation_metadata_count >= head_count;
    const uint32_t generation_base_count = generation_metadata_shape_valid
        ? header->generation_metadata_count - head_count : 0;
    const bool region_ranges_valid = region_index < header->region_count &&
        region.first_wave <= header->wave_count &&
        region.wave_count <= header->wave_count - region.first_wave &&
        region.first_slot <= header->slot_count &&
        region.slot_count <= header->slot_count - region.first_slot;
    const bool generation_range_valid =
        region.generation_offset <= generation_base_count &&
        region.slot_count <= generation_base_count - region.generation_offset;
    const uint64_t set_slot_count =
        static_cast<uint64_t>(header->set_slot_mask) + 1ULL;
    const bool set_workspace_shape_valid =
        (set_slot_count & (set_slot_count - 1ULL)) == 0 &&
        set_slot_count >= 2ULL * header->set_candidate_capacity;

    if (global_rank() == 0) {
        if (initializes) {
            status->code = kRunning;
            status->op_id = 0;
            status->resource_code = 0;
            status->iterations = 0;
            status->limit = 0;
            status->reserved = 0;
            status->required = 0;
            status->capacity = 0;
            *device_ptr<uint32_t>(header->changed) = 0;
            *device_ptr<uint32_t>(header->iterations) = 0;
            *device_ptr<uint32_t>(header->scan_trace) = 0;
            *device_ptr<uint32_t>(header->filter_trace) = 0;
            *device_ptr<uint32_t>(header->semantic_scan_trace) = 0;
            *device_ptr<uint32_t>(header->semantic_filter_trace) = 0;
        }
        const bool control_valid =
            (region.flags & ~allowed_region_flags) == 0 &&
            (!recursive || region.flags == kRegionRecursive) &&
            (!begins_scc || (!recursive && !finalizes)) &&
            (initializes == (region_index == 0)) &&
            (!initializes || (region.first_slot == 0 &&
                              region.slot_count == header->slot_count)) &&
            (finalizes == (region_index + 1 == header->region_count)) &&
            (recursive == (conditional_handle != 0));
        if (!region_ranges_valid || !generation_metadata_shape_valid ||
            !generation_range_valid ||
            !set_workspace_shape_valid ||
            header->abi_version != kAbiVersion || header->reserved != 0 ||
            !control_valid) {
            publish_terminal(status, kResourceExhausted, region.op_id,
                             kScheduleDescriptor, region_index + 1ULL,
                             header->region_count);
        } else {
            const uint32_t *generation_metadata =
                device_ptr<const uint32_t>(header->generation_metadata);
            for (uint32_t index = 0; index < region.slot_count; ++index) {
                ResidentRelationSlot &slot = slots[region.first_slot + index];
                const bool slot_flags_valid =
                    (slot.flags & ~(kSourceSlot | kPermanentSlot | kDefinedSlot)) == 0 &&
                    !((slot.flags & kSourceSlot) != 0 &&
                      (slot.flags & kPermanentSlot) != 0);
                if (!slot_flags_valid) {
                    publish_terminal(status, kResourceExhausted, region.op_id,
                                     kScheduleDescriptor, index + 1ULL, slot.flags);
                    break;
                }
                slot.generation = generation_metadata[region.generation_offset + index];
                if ((slot.flags & (kSourceSlot | kPermanentSlot)) != 0) {
                    slot.flags |= kDefinedSlot;
                } else {
                    slot.flags &= ~kDefinedSlot;
                    *device_ptr<uint32_t>(slot.relation.num_rows) = 0;
                }
            }
            if (initializes && head_count != 0 && status->code == kRunning) {
                uint32_t *schema_seen_nonempty =
                    device_ptr<uint32_t>(header->schema_seen_nonempty);
                uint32_t *schema_winner_ids =
                    device_ptr<uint32_t>(header->schema_winner_ids);
                for (uint32_t head = 0; head < head_count; ++head) {
                    schema_seen_nonempty[head] = 0U;
                    schema_winner_ids[head] =
                        generation_metadata[generation_base_count + head];
                }
            }
        }
        if (status->code == kRunning && recursive) {
            *device_ptr<uint32_t>(header->changed) = 0;
        }
    }
    grid.sync();

    const uint32_t safe_wave_count = region_ranges_valid ? region.wave_count : 0;
    for (uint32_t wave_offset = 0; wave_offset < safe_wave_count; ++wave_offset) {
        const ResidentWaveDescriptor wave = waves[region.first_wave + wave_offset];
        const bool wave_range_valid = wave.first_op <= header->op_count &&
            wave.op_count <= header->op_count - wave.first_op;
        if (global_rank() == 0 && status->code == kRunning && !wave_range_valid) {
            publish_terminal(status, kResourceExhausted, region.op_id,
                             kScheduleDescriptor,
                             static_cast<uint64_t>(wave.first_op) + wave.op_count,
                             header->op_count);
        }
        grid.sync();
        const uint32_t safe_op_count = wave_range_valid ? wave.op_count : 0;
        for (uint32_t op_offset = 0; op_offset < safe_op_count; ++op_offset) {
            const ResidentOpDescriptor op = ops[wave.first_op + op_offset];
            if (global_rank() == 0 && status->code == kRunning) {
            if (!validate_operation(header, op, region, slots, status)) {
                    if (status->code == kRunning) {
                        publish_terminal(status, kResourceExhausted, op.op_id,
                                         kScheduleDescriptor, 1, 0);
                    }
                }
            }
            grid.sync();

            const ResidentRelationView input_zero = op.in0 < header->slot_count
                ? slots[op.in0].relation
                : ResidentRelationView{};
            const ResidentRelationView input_one = op.in1 < header->slot_count
                ? slots[op.in1].relation
                : ResidentRelationView{};
            const ResidentRelationView output = op.out < header->slot_count
                ? slots[op.out].relation
                : ResidentRelationView{};

            if (op.kind == kOpTraceDelta) {
                if (global_rank() == 0 && valid_trace_delta(op)) {
                    atomicAdd(device_ptr<uint32_t>(header->scan_trace), op.scan_delta);
                    atomicAdd(device_ptr<uint32_t>(header->filter_trace), op.filter_delta);
                    const bool semantic_active =
                        (op.flags & kOpTraceSemanticGuard) == 0 ||
                        *device_ptr<const uint32_t>(input_zero.num_rows) != 0;
                    if (semantic_active) {
                        atomicAdd(device_ptr<uint32_t>(header->semantic_scan_trace),
                                  op.scan_delta);
                        atomicAdd(device_ptr<uint32_t>(header->semantic_filter_trace),
                                  op.filter_delta);
                    }
                }
                grid.sync();
            } else if (op.kind == kOpTestStatus) {
                if (global_rank() == 0 && status->code == kRunning &&
                    atomicCAS(&status->code, kRunning, kClaiming) == kRunning) {
                    status->op_id = op.op_id;
                    status->resource_code = op.in0;
                    status->iterations = op.in1;
                    status->limit = op.in0_generation;
                    status->reserved = op.in1_generation;
                    status->required = static_cast<uint64_t>(op.out_generation) |
                        (static_cast<uint64_t>(op.aux_offset) << 32);
                    status->capacity = static_cast<uint64_t>(op.aux_count) |
                        (static_cast<uint64_t>(op.left_key) << 32);
                    __threadfence_system();
                    atomicExch(&status->code, op.out);
                }
                grid.sync();
            } else if (op.kind == kOpUnit) {
                if (global_rank() == 0 && status->code == kRunning) {
                    if (output.capacity == 0) {
                        publish_terminal(status, kCapacityOverflow, op.op_id,
                                         kOutputRows, 1, 0);
                    } else {
                        *device_ptr<uint32_t>(output.num_rows) = 1;
                    }
                }
                grid.sync();
            } else if (op.kind == kOpScan) {
                grid.sync();
            } else if (op.kind == kOpFilter) {
                execute_filter(grid, header, op, input_zero, output, status);
            } else if (op.kind == kOpProject) {
                execute_project(grid, header, op, input_zero, output, status);
            } else if (op.kind == kOpJoinInner || op.kind == kOpJoinSemi) {
                execute_join(grid, header, op, input_zero, input_one, output,
                             status, op.kind == kOpJoinSemi);
            } else if (op.kind == kOpUnion || op.kind == kOpDiff) {
                execute_set(grid, header, op, input_zero, input_one, output,
                            status, op.kind == kOpUnion);
            } else {
                grid.sync();
            }

            if (global_rank() == 0 && status->code == kRunning) {
                const bool writes_output = op.kind == kOpUnit ||
                    op.kind == kOpFilter || op.kind == kOpProject ||
                    op.kind == kOpJoinInner || op.kind == kOpJoinSemi ||
                    op.kind == kOpUnion || op.kind == kOpDiff;
                if (writes_output) {
                    slots[op.out].generation = op.out_generation;
                    slots[op.out].flags |= kDefinedSlot;
                }
                if (writes_output && recursive &&
                    (op.flags & kOpMarkNovelty) != 0 &&
                    *device_ptr<const uint32_t>(output.num_rows) != 0) {
                    atomicOr(device_ptr<uint32_t>(header->changed), 1U);
                }
                if ((op.flags & kOpMarkSchemaWinner) != 0 &&
                    *device_ptr<const uint32_t>(output.num_rows) != 0) {
                    uint32_t *schema_seen_nonempty =
                        device_ptr<uint32_t>(header->schema_seen_nonempty);
                    uint32_t *schema_winner_ids =
                        device_ptr<uint32_t>(header->schema_winner_ids);
                    if (atomicCAS(&schema_seen_nonempty[op.schema_winner_head], 0U, 1U) == 0U) {
                        schema_winner_ids[op.schema_winner_head] = op.schema_winner_id;
                    }
                }
            }
            grid.sync();
        }
        grid.sync();
    }

    if (global_rank() == 0) {
        if (status->code == kRunning && begins_scc) {
            *device_ptr<uint32_t>(header->changed) = 0;
            *device_ptr<uint32_t>(header->iterations) = 0;
            status->limit = region.iteration_limit;
            if (region.iteration_limit == 0) {
                publish_terminal(status, kIterationLimit, region.op_id, 0, 0, 0);
            }
        }
        if (recursive) {
            uint32_t predicate = 0;
            if (status->code == kRunning) {
                const uint32_t loop_iteration =
                    *device_ptr<uint32_t>(header->iterations) + 1;
                *device_ptr<uint32_t>(header->iterations) = loop_iteration;
                const uint32_t total_iteration = status->iterations + 1;
                status->iterations = total_iteration;
                if (*device_ptr<const uint32_t>(header->changed) != 0 &&
                    loop_iteration < region.iteration_limit) {
                    predicate = 1;
                } else if (*device_ptr<const uint32_t>(header->changed) != 0) {
                    status->limit = region.iteration_limit;
                    publish_terminal(status, kIterationLimit, region.op_id, 0, 0, 0);
                }
            }
            if (conditional_handle != 0) {
                cudaGraphSetConditional(conditional_handle, predicate);
            }
        }
        if (status->code == kRunning && finalizes) {
            status->limit = 0;
            publish_terminal(status, kSuccess, region.op_id, 0, 0, 0);
        }
    }
    grid.sync();

    if (finalizes && global_rank() == 0) {
        const uint32_t required_receipt_bytes = sizeof(ResidentTerminalStatus) +
            sizeof(uint32_t) * (1 + header->receipt_count);
        if (header->receipt_byte_count >= required_receipt_bytes) {
            uint8_t *receipt = device_ptr<uint8_t>(header->receipt_bytes);
            *reinterpret_cast<ResidentTerminalStatus *>(receipt) = *status;
            uint32_t *counts = reinterpret_cast<uint32_t *>(
                receipt + sizeof(ResidentTerminalStatus));
            counts[0] = *device_ptr<uint32_t>(header->changed);
            const uint64_t *count_ptrs = device_ptr<const uint64_t>(header->receipt_table);
            for (uint32_t index = 0; index < header->receipt_count; ++index) {
                counts[index + 1] = *device_ptr<const uint32_t>(count_ptrs[index]);
            }
        }
    }
}
