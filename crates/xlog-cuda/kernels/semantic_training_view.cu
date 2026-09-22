#include <stdint.h>

struct SemanticTrainingViewOriginRecord {
    uint64_t present;
    uint64_t transition;
    uint64_t lineage_instance[4];
    uint64_t predecessor_instance[4];
    uint64_t predecessor_word;
    uint64_t predecessor_logical[4];
    uint64_t predecessor_state[4];
    uint64_t successor_instance[4];
    uint64_t successor_word;
    uint64_t successor_logical[4];
    uint64_t successor_state[4];
    uint64_t model_generation;
    uint64_t stream_serial;
    uint64_t family_id;
    uint64_t proposal;
    uint64_t model_geometry_digest[4];
    uint64_t model_numerical_digest[4];
};

struct TrainingViewRowDescriptor {
    uint64_t ordinal;
    uint64_t basis;
    uint64_t raw_offset;
    uint64_t raw_bytes;
    uint64_t window;
    uint64_t source_length;
    uint64_t block_size;
    uint64_t prefix_extent;
    uint64_t answer_start;
    uint64_t identity[4];
    uint64_t source_identity[4];
    uint64_t content_identity[4];
    uint64_t task_query_identity[4];
    uint64_t task_theory_program_identity[4];
    uint64_t task_result_identity[4];
    SemanticTrainingViewOriginRecord origin;
};

struct SemanticTrainingViewSelection {
    uint64_t status;
    uint64_t row_count;
    uint64_t capacity;
    uint64_t ordinal;
    uint64_t basis;
    uint64_t window;
    uint64_t source_length;
    uint64_t block_size;
    uint64_t prefix_extent;
    uint64_t answer_start;
    uint64_t identity[4];
    uint64_t source_identity[4];
    uint64_t content_identity[4];
    SemanticTrainingViewOriginRecord origin;
    uint64_t origin_candidate;
    uint64_t training_rng[4];
};

struct SemanticTrainingRosterRow {
    uint64_t ordinal;
    uint64_t basis;
    uint64_t window;
    uint64_t source_length;
    uint64_t block_size;
    uint64_t prefix_extent;
    uint64_t answer_start;
    uint64_t origin_candidate;
    uint64_t identity[4];
    uint64_t source_identity[4];
    uint64_t content_identity[4];
    SemanticTrainingViewOriginRecord origin;
};

struct TrainingViewLaunch {
    uint64_t descriptors;
    uint64_t raw;
    uint64_t row_count;
    uint64_t selected_view;
    uint64_t selected_view_bytes;
    uint64_t cursor;
    uint64_t training_rng[4];
    uint64_t coordinates;
    uint64_t origin_candidates;
    uint64_t origin_candidate_count;
    uint64_t selection;
    uint64_t roster_rows;
    uint64_t capacity;
    uint64_t token_ids;
    uint64_t mask_labels;
    uint64_t mask_weights;
    uint64_t ar_labels;
    uint64_t retention_labels;
    uint64_t source_slots;
    uint64_t logical_positions;
    uint64_t kinds;
    uint64_t parents;
};

__device__ bool semantic_training_identity_equal(const uint64_t* left, const uint64_t* right) {
    for (uint32_t i = 0; i < 4; ++i) {
        if (left[i] != right[i]) return false;
    }
    return true;
}

__device__ bool semantic_training_identity_present(const uint64_t* identity) {
    return identity[0] || identity[1] || identity[2] || identity[3];
}

__device__ bool semantic_training_historical_origin_valid(
        const SemanticTrainingViewOriginRecord& origin) {
    if (origin.present != 1 || !semantic_training_identity_present(origin.lineage_instance) ||
        !semantic_training_identity_equal(origin.lineage_instance, origin.predecessor_instance) ||
        !semantic_training_identity_equal(origin.predecessor_instance, origin.successor_instance) ||
        (origin.predecessor_word >> 1) == (UINT64_MAX >> 1)) {
        return false;
    }
    const uint64_t successor_word = (((origin.predecessor_word >> 1) + 1) << 1) |
        ((origin.predecessor_word & 1) ^ 1);
    return origin.successor_word == successor_word;
}

__device__ bool semantic_training_fresh_origin_valid(
        const SemanticTrainingViewOriginRecord& origin) {
    return origin.present == 1 && semantic_training_identity_present(origin.lineage_instance) &&
        semantic_training_identity_equal(origin.predecessor_instance, origin.successor_instance) &&
        !semantic_training_identity_equal(origin.predecessor_instance, origin.lineage_instance) &&
        origin.predecessor_word == 0 && origin.successor_word == 3;
}

__device__ bool semantic_training_origin_matches(const SemanticTrainingViewOriginRecord& historical,
                                                 const SemanticTrainingViewOriginRecord& fresh) {
    if (!semantic_training_historical_origin_valid(historical) ||
        !semantic_training_fresh_origin_valid(fresh) || historical.transition != fresh.transition ||
        historical.model_generation != fresh.model_generation ||
        historical.stream_serial != fresh.stream_serial || historical.family_id != fresh.family_id ||
        historical.proposal != fresh.proposal ||
        !semantic_training_identity_equal(historical.lineage_instance, fresh.lineage_instance)) {
        return false;
    }
    for (uint32_t i = 0; i < 4; ++i) {
        if (historical.predecessor_logical[i] != fresh.predecessor_logical[i] ||
            historical.predecessor_state[i] != fresh.predecessor_state[i] ||
            historical.successor_logical[i] != fresh.successor_logical[i] ||
            historical.successor_state[i] != fresh.successor_state[i] ||
            historical.model_geometry_digest[i] != fresh.model_geometry_digest[i] ||
            historical.model_numerical_digest[i] != fresh.model_numerical_digest[i]) {
            return false;
        }
    }
    return true;
}

extern "C" __global__ void semantic_training_view_select(TrainingViewLaunch launch) {
    auto* selection = reinterpret_cast<SemanticTrainingViewSelection*>(launch.selection);
    auto* selection_words = reinterpret_cast<uint64_t*>(selection);
    for (uint64_t word = threadIdx.x;
         word < sizeof(SemanticTrainingViewSelection) / sizeof(uint64_t);
         word += blockDim.x) {
        selection_words[word] = 0;
    }
    __syncthreads();
    const auto* coordinates = launch.coordinates
        ? reinterpret_cast<const uint64_t*>(launch.coordinates)
        : &launch.cursor;
    const uint64_t cursor = coordinates[0];
    if (threadIdx.x == 0) {
        selection->status = cursor < launch.row_count ? 0 : 1;
        selection->row_count = launch.row_count;
        selection->capacity = launch.capacity;
        selection->origin_candidate = UINT64_MAX;
    }
    __syncthreads();
    if (selection->status != 0) {
        return;
    }
    const auto* descriptors = reinterpret_cast<const TrainingViewRowDescriptor*>(launch.descriptors);
    const auto descriptor = descriptors[cursor];
    const auto* raw = reinterpret_cast<const uint8_t*>(launch.raw) + descriptor.raw_offset;
    const auto* selected = reinterpret_cast<const uint8_t*>(launch.selected_view);
    if (threadIdx.x == 0 &&
        (descriptor.ordinal != cursor || descriptor.raw_bytes != launch.selected_view_bytes)) {
        selection->status = 2;
    }
    __syncthreads();
    if (selection->status == 0) {
        for (uint64_t offset = threadIdx.x; offset < descriptor.raw_bytes; offset += blockDim.x) {
            if (raw[offset] != selected[offset]) {
                atomicCAS(reinterpret_cast<unsigned long long*>(&selection->status), 0ULL, 2ULL);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && selection->status == 0) {
        auto* roster = reinterpret_cast<SemanticTrainingRosterRow*>(launch.roster_rows);
        const auto* candidates = reinterpret_cast<const SemanticTrainingViewOriginRecord*>(
            launch.origin_candidates);
        for (uint64_t row = 0; row < launch.row_count; ++row) {
            const auto item = descriptors[row];
            uint64_t origin_candidate = UINT64_MAX;
            if (item.basis == 1 && semantic_training_historical_origin_valid(item.origin)) {
                uint64_t matches = 0;
                for (uint64_t candidate = 0; candidate < launch.origin_candidate_count; ++candidate) {
                    if (semantic_training_origin_matches(item.origin, candidates[candidate])) {
                        origin_candidate = candidate;
                        ++matches;
                    }
                }
                if (matches > 1 || (row == cursor && matches != 1)) {
                    selection->status = 3;
                }
            } else if ((item.basis != 2 && item.basis != 3) || item.origin.present != 0) {
                selection->status = 3;
            }
            roster[row].ordinal = item.ordinal;
            roster[row].basis = item.basis;
            roster[row].window = item.window;
            roster[row].source_length = item.source_length;
            roster[row].block_size = item.block_size;
            roster[row].prefix_extent = item.prefix_extent;
            roster[row].answer_start = item.answer_start;
            roster[row].origin_candidate = origin_candidate;
            for (uint32_t i = 0; i < 4; ++i) {
                roster[row].identity[i] = item.identity[i];
                roster[row].source_identity[i] = item.source_identity[i];
                roster[row].content_identity[i] = item.content_identity[i];
            }
            roster[row].origin = item.origin;
            if (row == cursor) {
                selection->origin_candidate = origin_candidate;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        selection->ordinal = descriptor.ordinal;
        selection->basis = descriptor.basis;
        selection->window = descriptor.window;
        selection->source_length = descriptor.source_length;
        selection->block_size = descriptor.block_size;
        selection->prefix_extent = descriptor.prefix_extent;
        selection->answer_start = descriptor.answer_start;
        selection->origin = descriptor.origin;
        for (uint32_t i = 0; i < 4; ++i) {
            selection->identity[i] = descriptor.identity[i];
            selection->source_identity[i] = descriptor.source_identity[i];
            selection->content_identity[i] = descriptor.content_identity[i];
            selection->training_rng[i] =
                launch.coordinates ? coordinates[i + 1] : launch.training_rng[i];
        }
    }
}

extern "C" __global__ void semantic_training_view_gather(TrainingViewLaunch launch) {
    const auto* selection = reinterpret_cast<const SemanticTrainingViewSelection*>(launch.selection);
    auto* output_token_ids = reinterpret_cast<int64_t*>(launch.token_ids);
    auto* output_mask_labels = reinterpret_cast<int64_t*>(launch.mask_labels);
    auto* output_mask_weights = reinterpret_cast<float*>(launch.mask_weights);
    auto* output_ar_labels = reinterpret_cast<int64_t*>(launch.ar_labels);
    auto* output_retention_labels = reinterpret_cast<int64_t*>(launch.retention_labels);
    auto* output_source_slots = reinterpret_cast<int64_t*>(launch.source_slots);
    auto* output_logical_positions = reinterpret_cast<int64_t*>(launch.logical_positions);
    auto* output_kinds = reinterpret_cast<int64_t*>(launch.kinds);
    auto* output_parents = reinterpret_cast<int64_t*>(launch.parents);
    const uint64_t row = blockIdx.x;
    if (row >= launch.row_count) {
        return;
    }
    const uint64_t output_offset = row * launch.capacity;
    for (uint64_t index = threadIdx.x; index < launch.capacity; index += blockDim.x) {
        output_token_ids[output_offset + index] = 0;
        output_mask_labels[output_offset + index] = -100;
        output_mask_weights[output_offset + index] = 0.0f;
        output_ar_labels[output_offset + index] = -100;
        output_retention_labels[output_offset + index] = -100;
        output_source_slots[output_offset + index] = static_cast<int64_t>(index);
        output_logical_positions[output_offset + index] = -1;
        output_kinds[output_offset + index] = 0;
        output_parents[output_offset + index] = -2;
    }
    __syncthreads();
    if (selection->status != 0) {
        return;
    }
    const auto* descriptors = reinterpret_cast<const TrainingViewRowDescriptor*>(launch.descriptors);
    const auto descriptor = descriptors[row];
    const auto* raw = reinterpret_cast<const uint8_t*>(launch.raw) + descriptor.raw_offset;
    const uint64_t window = descriptor.window;
    const uint64_t padding = (window & 1ULL) * 4ULL;
    const auto* token_ids = reinterpret_cast<const int64_t*>(raw + 136);
    const auto* mask_labels = reinterpret_cast<const int64_t*>(raw + 136 + window * 8);
    const auto* mask_weights = reinterpret_cast<const float*>(raw + 136 + window * 16);
    const auto* ar_labels = reinterpret_cast<const int64_t*>(raw + 136 + window * 20 + padding);
    const auto* retention_labels = ar_labels + window;
    const auto* source_slots = retention_labels + window;
    const auto* logical_positions = source_slots + window;
    const auto* kinds = logical_positions + window;
    const auto* parents = kinds + window;
    for (uint64_t index = threadIdx.x; index < launch.capacity; index += blockDim.x) {
        if (index < window) {
            output_token_ids[output_offset + index] = token_ids[index];
            output_mask_labels[output_offset + index] = mask_labels[index];
            output_mask_weights[output_offset + index] = mask_weights[index];
            output_ar_labels[output_offset + index] = ar_labels[index];
            output_retention_labels[output_offset + index] = retention_labels[index];
            output_source_slots[output_offset + index] = source_slots[index];
            output_logical_positions[output_offset + index] = logical_positions[index];
            output_kinds[output_offset + index] = kinds[index];
            output_parents[output_offset + index] = parents[index];
        }
    }
}
