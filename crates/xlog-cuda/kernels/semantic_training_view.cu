#include <stdint.h>

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
    if (threadIdx.x == 0) {
        selection->ordinal = descriptor.ordinal;
        selection->basis = descriptor.basis;
        selection->window = descriptor.window;
        selection->source_length = descriptor.source_length;
        selection->block_size = descriptor.block_size;
        selection->prefix_extent = descriptor.prefix_extent;
        selection->answer_start = descriptor.answer_start;
        for (uint32_t i = 0; i < 4; ++i) {
            selection->identity[i] = descriptor.identity[i];
            selection->source_identity[i] = descriptor.source_identity[i];
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
