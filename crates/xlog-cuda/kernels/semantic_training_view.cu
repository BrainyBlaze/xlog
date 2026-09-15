#include <stdint.h>

struct SemanticTrainingViewOriginRecord {
    uint64_t present;
    uint64_t transition;
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
    SemanticTrainingViewOriginRecord origin;
};

struct SemanticTrainingViewSelection {
    uint64_t status;
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

struct TrainingViewLaunch {
    uint64_t descriptors;
    uint64_t raw;
    uint64_t row_count;
    uint64_t selected_view;
    uint64_t selected_view_bytes;
    uint64_t cursor;
    uint64_t training_rng[4];
    uint64_t origin_candidates;
    uint64_t origin_candidate_count;
    uint64_t selection;
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

__device__ bool semantic_training_origin_equal(const SemanticTrainingViewOriginRecord& left,
                                               const SemanticTrainingViewOriginRecord& right) {
    if (left.present != right.present || left.transition != right.transition ||
        left.predecessor_word != right.predecessor_word ||
        left.successor_word != right.successor_word ||
        left.model_generation != right.model_generation ||
        left.stream_serial != right.stream_serial || left.family_id != right.family_id ||
        left.proposal != right.proposal) {
        return false;
    }
    for (uint32_t i = 0; i < 4; ++i) {
        if (left.predecessor_instance[i] != right.predecessor_instance[i] ||
            left.predecessor_logical[i] != right.predecessor_logical[i] ||
            left.predecessor_state[i] != right.predecessor_state[i] ||
            left.successor_instance[i] != right.successor_instance[i] ||
            left.successor_logical[i] != right.successor_logical[i] ||
            left.successor_state[i] != right.successor_state[i] ||
            left.model_geometry_digest[i] != right.model_geometry_digest[i] ||
            left.model_numerical_digest[i] != right.model_numerical_digest[i]) {
            return false;
        }
    }
    return true;
}

extern "C" __global__ void semantic_training_view_select(TrainingViewLaunch launch) {
    auto* selection = reinterpret_cast<SemanticTrainingViewSelection*>(launch.selection);
    if (threadIdx.x == 0) {
        selection->status = launch.cursor < launch.row_count ? 0 : 1;
    }
    __syncthreads();
    if (selection->status != 0) {
        return;
    }
    const auto* descriptors = reinterpret_cast<const TrainingViewRowDescriptor*>(launch.descriptors);
    const auto descriptor = descriptors[launch.cursor];
    const auto* raw = reinterpret_cast<const uint8_t*>(launch.raw) + descriptor.raw_offset;
    const auto* selected = reinterpret_cast<const uint8_t*>(launch.selected_view);
    if (threadIdx.x == 0 &&
        (descriptor.ordinal != launch.cursor || descriptor.raw_bytes != launch.selected_view_bytes)) {
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
        selection->origin_candidate = UINT64_MAX;
        if (descriptor.basis == 1 && descriptor.origin.present == 1) {
            const auto* candidates = reinterpret_cast<const SemanticTrainingViewOriginRecord*>(
                launch.origin_candidates);
            uint64_t matches = 0;
            for (uint64_t candidate = 0; candidate < launch.origin_candidate_count; ++candidate) {
                if (semantic_training_origin_equal(descriptor.origin, candidates[candidate])) {
                    selection->origin_candidate = candidate;
                    ++matches;
                }
            }
            if (matches != 1) {
                selection->status = 3;
            }
        } else if (descriptor.basis != 2 || descriptor.origin.present != 0) {
            selection->status = 3;
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
            selection->training_rng[i] = launch.training_rng[i];
        }
    }
}

extern "C" __global__ void semantic_training_view_gather(TrainingViewLaunch launch) {
    const auto* selection = reinterpret_cast<const SemanticTrainingViewSelection*>(launch.selection);
    if (selection->status != 0) {
        return;
    }
    const auto* descriptors = reinterpret_cast<const TrainingViewRowDescriptor*>(launch.descriptors);
    const auto descriptor = descriptors[selection->ordinal];
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
    auto* output_token_ids = reinterpret_cast<int64_t*>(launch.token_ids);
    auto* output_mask_labels = reinterpret_cast<int64_t*>(launch.mask_labels);
    auto* output_mask_weights = reinterpret_cast<float*>(launch.mask_weights);
    auto* output_ar_labels = reinterpret_cast<int64_t*>(launch.ar_labels);
    auto* output_retention_labels = reinterpret_cast<int64_t*>(launch.retention_labels);
    auto* output_source_slots = reinterpret_cast<int64_t*>(launch.source_slots);
    auto* output_logical_positions = reinterpret_cast<int64_t*>(launch.logical_positions);
    auto* output_kinds = reinterpret_cast<int64_t*>(launch.kinds);
    auto* output_parents = reinterpret_cast<int64_t*>(launch.parents);
    for (uint64_t index = threadIdx.x; index < launch.capacity; index += blockDim.x) {
        const bool active = index < window;
        output_token_ids[index] = active ? token_ids[index] : 0;
        output_mask_labels[index] = active ? mask_labels[index] : -100;
        output_mask_weights[index] = active ? mask_weights[index] : 0.0f;
        output_ar_labels[index] = active ? ar_labels[index] : -100;
        output_retention_labels[index] = active ? retention_labels[index] : -100;
        output_source_slots[index] = active ? source_slots[index] : -1;
        output_logical_positions[index] = active ? logical_positions[index] : -1;
        output_kinds[index] = active ? kinds[index] : 0;
        output_parents[index] = active ? parents[index] : -1;
    }
}
