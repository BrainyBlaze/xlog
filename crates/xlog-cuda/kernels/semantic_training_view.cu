#include <stdint.h>
#ifdef __CUDACC__
#include <cuda/atomic>
#else
#include <atomic>
#endif
#define XLOG_SEMANTIC_GRAPH_DEVICE_ONLY
#include "semantic_hypergraph.cu"

static constexpr uint64_t COMMITTED_APPEND_ORIGIN=UINT64_MAX-1;

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

struct SemanticTrainingObjectiveRecord {
    uint64_t evaluator_abi,identity[4],task_identity[4],row_count,capacity;
    uint64_t group_count,group_member_count,canary_count,protected_member_count;
    uint64_t evaluator_min_bits,evaluator_max_bits,coefficient_bits[9],cost_unit[4],cost_cap,truth_tokens[4];
};

struct SemanticTrainingObjectiveGroupRecord {
    uint64_t kind,denominator,member_offset,member_count;
};

struct SemanticTrainingReplayAppendHeader {
    uint64_t abi,count,capacity,payload_used_bytes,payload_capacity_bytes,eligible_count,chain_head[4];
};

struct SemanticTrainingReplayAppendEntry {
    uint64_t row_ordinal,content_identity[4];
    SemanticTrainingViewOriginRecord original_origin;
    uint64_t stable_intent[4],result_receipt_digest[4],payload_offset_bytes,payload_length_bytes,disposition;
};

struct TrainingPublicationRange {
    uint64_t role,index,storage_slot,generation,offset_bytes,length_bytes,logical_begin,logical_end;
    uint64_t digest[4],backing_digest[4];
};

struct TrainingPublicationStorageEntry { uint64_t pointer,bytes,generation; };

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
    uint64_t branch_labels;
    uint64_t branch_ids;
    uint64_t source_slots;
    uint64_t logical_positions;
    uint64_t kinds;
    uint64_t parents;
    uint64_t cold_work;
    uint64_t initial_row_count;
    uint64_t objective_template;
    uint64_t groups_template;
    uint64_t group_members_template;
    uint64_t objective;
    uint64_t groups;
    uint64_t group_members;
    uint64_t group_member_capacity;
    uint64_t publication_word;
    uint64_t directories[2];
    uint64_t directory_count;
    uint64_t publication_storage;
    uint64_t publication_storage_count;
    uint64_t append_entries[2];
    uint64_t append_entries_bytes[2];
    uint64_t append_payloads[2];
    uint64_t append_payload_bytes[2];
};

static_assert(sizeof(TrainingPublicationRange)==128 && sizeof(TrainingPublicationStorageEntry)==24 &&
    sizeof(SemanticTrainingReplayAppendHeader)==80 && sizeof(SemanticTrainingReplayAppendEntry)==480,
    "training replay publication ABI");

__device__ bool semantic_training_identity_equal(const uint64_t* left, const uint64_t* right) {
    for (uint32_t i = 0; i < 4; ++i) {
        if (left[i] != right[i]) return false;
    }
    return true;
}

__device__ bool semantic_training_identity_present(const uint64_t* identity) {
    return identity[0] || identity[1] || identity[2] || identity[3];
}

__device__ bool semantic_training_append_range(const TrainingViewLaunch& launch,
        const TrainingPublicationRange& range,uint64_t expected,uint64_t capacity,
        const uint8_t** bytes) {
    if (range.storage_slot >= launch.publication_storage_count) return false;
    const auto& storage = reinterpret_cast<const TrainingPublicationStorageEntry*>(
        launch.publication_storage)[range.storage_slot];
    if (range.generation != storage.generation || range.offset_bytes > storage.bytes ||
        range.length_bytes > storage.bytes-range.offset_bytes ||
        storage.pointer > UINT64_MAX-range.offset_bytes) return false;
    const uint64_t pointer = storage.pointer+range.offset_bytes;
    if (pointer < expected || pointer-expected > capacity ||
        range.length_bytes > capacity-(pointer-expected) || (range.length_bytes && !pointer)) return false;
    *bytes = reinterpret_cast<const uint8_t*>(pointer);
    return true;
}

__device__ bool semantic_training_append_queue(const TrainingViewLaunch& launch,
        const SemanticTrainingReplayAppendHeader** header,
        const SemanticTrainingReplayAppendEntry** entries,
        semantic_graph::NativeWorkTally* work) {
    if (!launch.publication_word || !launch.publication_storage) return false;
    auto* word = reinterpret_cast<uint64_t*>(launch.publication_word);
#ifdef __CUDACC__
    const uint64_t bank = cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(*word)
        .load(cuda::memory_order_acquire)&1;
#else
    const uint64_t bank = std::atomic_ref<uint64_t>(*word).load(std::memory_order_acquire)&1;
#endif
    const auto* directory = reinterpret_cast<const TrainingPublicationRange*>(launch.directories[bank]);
    const TrainingPublicationRange* queue = nullptr;
    const TrainingPublicationRange* payload = nullptr;
    if (!directory) return false;
    for (uint64_t index=0;index<launch.directory_count;++index) {
        semantic_graph::charge_native_parallel(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if (directory[index].role==56) {
            if (queue || directory[index].index) return false;
            queue=&directory[index];
        }
        if (directory[index].role==57) {
            if (payload || directory[index].index) return false;
            payload=&directory[index];
        }
    }
    const uint8_t* queue_bytes=nullptr;
    const uint8_t* payload_bytes=nullptr;
    if (!queue || !payload || queue->length_bytes<sizeof(SemanticTrainingReplayAppendHeader) ||
        !semantic_training_append_range(launch,*queue,launch.append_entries[bank],
            launch.append_entries_bytes[bank],&queue_bytes) ||
        !semantic_training_append_range(launch,*payload,launch.append_payloads[bank],
            launch.append_payload_bytes[bank],&payload_bytes)) return false;
    *header=reinterpret_cast<const SemanticTrainingReplayAppendHeader*>(queue_bytes);
    const auto& value=**header;
    if (value.abi!=1 || value.capacity!=launch.row_count-launch.initial_row_count ||
        value.count>value.capacity || value.eligible_count>value.count ||
        value.payload_used_bytes!=payload->length_bytes ||
        value.payload_used_bytes>value.payload_capacity_bytes ||
        value.payload_capacity_bytes>launch.append_payload_bytes[bank] ||
        value.count>(UINT64_MAX-sizeof(value))/sizeof(SemanticTrainingReplayAppendEntry) ||
        queue->length_bytes!=sizeof(value)+value.count*sizeof(SemanticTrainingReplayAppendEntry) ||
        (value.count && !semantic_training_identity_present(value.chain_head))) return false;
    *entries=reinterpret_cast<const SemanticTrainingReplayAppendEntry*>(queue_bytes+sizeof(value));
    return true;
}

__device__ bool semantic_training_committed_origin(const TrainingViewRowDescriptor& descriptor,
        const SemanticTrainingReplayAppendEntry& entry,const SemanticTrainingReplayAppendHeader& header) {
    if (entry.disposition!=1 || descriptor.basis!=1 || descriptor.ordinal!=entry.row_ordinal ||
        !semantic_training_identity_equal(descriptor.content_identity,entry.content_identity) ||
        !semantic_training_identity_present(entry.stable_intent) ||
        !semantic_training_identity_present(entry.result_receipt_digest) ||
        !entry.payload_length_bytes || entry.payload_offset_bytes>header.payload_used_bytes ||
        entry.payload_length_bytes>header.payload_used_bytes-entry.payload_offset_bytes) return false;
    const auto* original=reinterpret_cast<const uint64_t*>(&entry.original_origin);
    const auto* selected=reinterpret_cast<const uint64_t*>(&descriptor.origin);
    for (uint64_t word=0;word<sizeof(entry.original_origin)/sizeof(uint64_t);++word)
        if (original[word]!=selected[word]) return false;
    return true;
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
    auto* work = reinterpret_cast<semantic_graph::NativeWorkTally*>(launch.cold_work);
    auto* selection = reinterpret_cast<SemanticTrainingViewSelection*>(launch.selection);
    auto* selection_words = reinterpret_cast<uint64_t*>(selection);
    for (uint64_t word = threadIdx.x;
         word < sizeof(SemanticTrainingViewSelection) / sizeof(uint64_t);
         word += blockDim.x) {
        selection_words[word] = 0;
        semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
            sizeof(selection_words[word]));
    }
    auto* roster_words = reinterpret_cast<uint64_t*>(launch.roster_rows);
    for (uint64_t word = threadIdx.x;
         word < launch.row_count * (sizeof(SemanticTrainingRosterRow) / sizeof(uint64_t));
         word += blockDim.x) {
        roster_words[word] = 0;
        semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
            sizeof(roster_words[word]));
    }
    auto* member_words = reinterpret_cast<uint64_t*>(launch.group_members);
    for (uint64_t member=threadIdx.x;member<launch.group_member_capacity;member+=blockDim.x) {
        member_words[member]=0;
        semantic_graph::charge_native_parallel(work,semantic_graph::NativeWorkEvent::CanonicalByte,
            sizeof(member_words[member]));
    }
    __syncthreads();
    const auto* coordinates = launch.coordinates
        ? reinterpret_cast<const uint64_t*>(launch.coordinates)
        : &launch.cursor;
    const uint64_t cursor = coordinates[0];
    __shared__ const SemanticTrainingReplayAppendHeader* append_header;
    __shared__ const SemanticTrainingReplayAppendEntry* append_entries;
    if (threadIdx.x == 0) {
        append_header=nullptr;
        append_entries=nullptr;
        bool publication_valid=launch.initial_row_count<=launch.row_count;
        uint64_t published_rows=launch.initial_row_count;
        publication_valid=publication_valid && semantic_training_append_queue(launch,
            &append_header,&append_entries,work);
        if (publication_valid) published_rows+=append_header->eligible_count;
        selection->status = !publication_valid ? 4 : (cursor < published_rows ? 0 : 1);
        selection->row_count = published_rows;
        selection->capacity = launch.capacity;
        selection->origin_candidate = UINT64_MAX;
        semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
            sizeof(selection->status)+sizeof(selection->row_count)+sizeof(selection->capacity)+
            sizeof(selection->origin_candidate));
        auto* objective = reinterpret_cast<SemanticTrainingObjectiveRecord*>(launch.objective);
        const auto& objective_template =
            *reinterpret_cast<const SemanticTrainingObjectiveRecord*>(launch.objective_template);
        *objective = objective_template;
        objective->row_count = selection->row_count;
        auto* groups = reinterpret_cast<SemanticTrainingObjectiveGroupRecord*>(launch.groups);
        const auto* group_templates =
            reinterpret_cast<const SemanticTrainingObjectiveGroupRecord*>(launch.groups_template);
        const auto* member_templates = reinterpret_cast<const uint64_t*>(launch.group_members_template);
        auto* members = reinterpret_cast<uint64_t*>(launch.group_members);
        uint64_t member_count = 0;
        for (uint64_t group = 0; group < objective->group_count; ++group) {
            const auto& source = group_templates[group];
            auto& target = groups[group];
            target.kind = source.kind;
            target.member_offset = member_count;
            for (uint64_t member = 0; member < source.member_count; ++member) {
                semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::TableSlot, 1);
                const uint64_t ordinal = member_templates[source.member_offset + member];
                if (ordinal < selection->row_count) {
                    members[member_count++] = ordinal;
                    semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
                        sizeof(ordinal));
                }
            }
            target.member_count = member_count - target.member_offset;
            target.denominator = target.member_count;
            semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
                sizeof(target));
        }
        objective->group_member_count = member_count;
        semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
            sizeof(*objective));
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
        semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
            sizeof(selection->status));
    }
    __syncthreads();
    if (selection->status == 0) {
        for (uint64_t offset = threadIdx.x; offset < descriptor.raw_bytes; offset += blockDim.x) {
            semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::TableSlot, 1);
            if (raw[offset] != selected[offset]) {
                if (atomicCAS(reinterpret_cast<unsigned long long*>(&selection->status), 0ULL, 2ULL) == 0ULL)
                    semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
                        sizeof(selection->status));
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && selection->status == 0) {
        auto* roster = reinterpret_cast<SemanticTrainingRosterRow*>(launch.roster_rows);
        const auto* candidates = reinterpret_cast<const SemanticTrainingViewOriginRecord*>(
            launch.origin_candidates);
        for (uint64_t row = 0; row < selection->row_count; ++row) {
            semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::TableSlot, 1);
            const auto item = descriptors[row];
            uint64_t origin_candidate = UINT64_MAX;
            if (row>=launch.initial_row_count) {
                const uint64_t slot=row-launch.initial_row_count;
                uint64_t matches=0;
                if (append_header && slot<append_header->eligible_count &&
                    semantic_training_historical_origin_valid(item.origin)) {
                    for (uint64_t outcome=0;outcome<append_header->count;++outcome) {
                        semantic_graph::charge_native_parallel(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                        if (semantic_training_committed_origin(item,append_entries[outcome],*append_header))
                            ++matches;
                    }
                }
                if (matches!=1) {
                    selection->status=3;
                    semantic_graph::charge_native_parallel(work,semantic_graph::NativeWorkEvent::CanonicalByte,
                        sizeof(selection->status));
                } else {
                    origin_candidate=COMMITTED_APPEND_ORIGIN;
                }
            } else if (item.basis == 1 && semantic_training_historical_origin_valid(item.origin)) {
                uint64_t matches = 0;
                for (uint64_t candidate = 0; candidate < launch.origin_candidate_count; ++candidate) {
                    semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::TableSlot, 1);
                    if (semantic_training_origin_matches(item.origin, candidates[candidate])) {
                        origin_candidate = candidate;
                        ++matches;
                    }
                }
                if (matches > 1 || (row == cursor && matches != 1)) {
                    selection->status = 3;
                    semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
                        sizeof(selection->status));
                }
            } else if ((item.basis != 2 && item.basis != 3) || item.origin.present != 0) {
                selection->status = 3;
                semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
                    sizeof(selection->status));
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
            semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
                sizeof(roster[row]));
            if (row == cursor) {
                selection->origin_candidate = origin_candidate;
                semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
                    sizeof(selection->origin_candidate));
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
        semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
            sizeof(selection->ordinal)+sizeof(selection->basis)+sizeof(selection->window)+
            sizeof(selection->source_length)+sizeof(selection->block_size)+sizeof(selection->prefix_extent)+
            sizeof(selection->answer_start)+sizeof(selection->origin)+sizeof(selection->identity)+
            sizeof(selection->source_identity)+sizeof(selection->content_identity)+sizeof(selection->training_rng));
    }
}

extern "C" __global__ void semantic_training_view_gather(TrainingViewLaunch launch) {
    auto* work = reinterpret_cast<semantic_graph::NativeWorkTally*>(launch.cold_work);
    const auto* selection = reinterpret_cast<const SemanticTrainingViewSelection*>(launch.selection);
    auto* output_token_ids = reinterpret_cast<int64_t*>(launch.token_ids);
    auto* output_mask_labels = reinterpret_cast<int64_t*>(launch.mask_labels);
    auto* output_mask_weights = reinterpret_cast<float*>(launch.mask_weights);
    auto* output_ar_labels = reinterpret_cast<int64_t*>(launch.ar_labels);
    auto* output_retention_labels = reinterpret_cast<int64_t*>(launch.retention_labels);
    auto* output_branch_labels = reinterpret_cast<int64_t*>(launch.branch_labels);
    auto* output_branch_ids = reinterpret_cast<int64_t*>(launch.branch_ids);
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
        output_branch_labels[output_offset + index] = -100;
        output_branch_ids[output_offset + index] = -1;
        output_source_slots[output_offset + index] = static_cast<int64_t>(index);
        output_logical_positions[output_offset + index] = -1;
        output_kinds[output_offset + index] = 0;
        output_parents[output_offset + index] = -2;
        semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
            sizeof(output_token_ids[0])+sizeof(output_mask_labels[0])+sizeof(output_mask_weights[0])+
            sizeof(output_ar_labels[0])+sizeof(output_retention_labels[0])+sizeof(output_branch_labels[0])+
            sizeof(output_branch_ids[0])+sizeof(output_source_slots[0])+sizeof(output_logical_positions[0])+
            sizeof(output_kinds[0])+sizeof(output_parents[0]));
    }
    __syncthreads();
    if (selection->status != 0) {
        return;
    }
    if (row >= selection->row_count) {
        return;
    }
    const auto* descriptors = reinterpret_cast<const TrainingViewRowDescriptor*>(launch.descriptors);
    const auto descriptor = descriptors[row];
    const auto* raw = reinterpret_cast<const uint8_t*>(launch.raw) + descriptor.raw_offset;
    const uint64_t window = descriptor.window;
    const uint64_t padding = (window & 1ULL) * 4ULL;
    const auto* token_ids = reinterpret_cast<const int64_t*>(raw + 264);
    const auto* mask_labels = reinterpret_cast<const int64_t*>(raw + 264 + window * 8);
    const auto* mask_weights = reinterpret_cast<const float*>(raw + 264 + window * 16);
    const auto* ar_labels = reinterpret_cast<const int64_t*>(raw + 264 + window * 20 + padding);
    const auto* retention_labels = ar_labels + window;
    const auto* branch_labels = retention_labels + window;
    const auto* branch_ids = branch_labels + window;
    const auto* source_slots = branch_ids + window;
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
            output_branch_labels[output_offset + index] = branch_labels[index];
            output_branch_ids[output_offset + index] = branch_ids[index];
            output_source_slots[output_offset + index] = source_slots[index];
            output_logical_positions[output_offset + index] = logical_positions[index];
            output_kinds[output_offset + index] = kinds[index];
            output_parents[output_offset + index] = parents[index];
            semantic_graph::charge_native_parallel(work, semantic_graph::NativeWorkEvent::CanonicalByte,
                sizeof(output_token_ids[0])+sizeof(output_mask_labels[0])+sizeof(output_mask_weights[0])+
                sizeof(output_ar_labels[0])+sizeof(output_retention_labels[0])+sizeof(output_branch_labels[0])+
                sizeof(output_branch_ids[0])+sizeof(output_source_slots[0])+sizeof(output_logical_positions[0])+
                sizeof(output_kinds[0])+sizeof(output_parents[0]));
        }
    }
}
