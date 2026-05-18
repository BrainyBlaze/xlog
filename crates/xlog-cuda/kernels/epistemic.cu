#include <stdint.h>

extern "C" __global__ void epistemic_generate_candidate_assumptions_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint8_t* __restrict__ out
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = literal_count * candidate_count;
    if (gid >= total) return;

    uint32_t candidate = gid / literal_count;
    uint32_t literal = gid - candidate * literal_count;
    out[gid] = static_cast<uint8_t>((candidate >> literal) & 1u);
}

extern "C" __global__ void epistemic_propagate_candidates_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t world_stride,
    const uint8_t* __restrict__ candidate_assumptions,
    uint8_t* __restrict__ world_views,
    uint32_t* __restrict__ rejection_reasons
) {
    uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) return;

    uint8_t observed = 0;
    uint32_t base = candidate * literal_count;
    for (uint32_t literal = 0; literal < literal_count; ++literal) {
        observed |= candidate_assumptions[base + literal];
    }

    world_views[candidate * world_stride] = (observed == 255u) ? 0u : 1u;
    rejection_reasons[candidate] = 0u;
}

extern "C" __global__ void epistemic_validate_candidate_bits_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t world_stride,
    const uint8_t* __restrict__ candidate_assumptions,
    const uint8_t* __restrict__ world_views,
    uint32_t* __restrict__ rejection_reasons
) {
    uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) return;

    uint32_t reason = 0u;
    if (world_views[candidate * world_stride] == 0u) {
        reason = 2u;
    }

    uint32_t base = candidate * literal_count;
    for (uint32_t literal = 0; literal < literal_count; ++literal) {
        uint8_t value = candidate_assumptions[base + literal];
        if (value > 1u) {
            reason = 3u;
        }
    }

    rejection_reasons[candidate] = reason;
}

extern "C" __global__ void epistemic_populate_model_membership_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t reduction_count,
    uint32_t models_per_reduction,
    uint32_t world_stride,
    const uint32_t* __restrict__ output_row_count,
    const uint8_t* __restrict__ candidate_assumptions,
    const uint8_t* __restrict__ world_views,
    uint8_t* __restrict__ model_membership,
    uint32_t* __restrict__ rejection_reasons
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t per_candidate = reduction_count * models_per_reduction * literal_count;
    uint32_t total = candidate_count * per_candidate;
    if (gid >= total) return;

    uint32_t candidate = gid / per_candidate;
    uint32_t inner = gid - candidate * per_candidate;
    uint32_t literal = inner % literal_count;
    uint32_t model_slot = inner / literal_count;

    uint8_t active_world = world_views[candidate * world_stride];
    uint8_t accepted_so_far = (rejection_reasons[candidate] == 0u) ? 1u : 0u;
    uint8_t has_reduced_output = (output_row_count[0] > 0u) ? 1u : 0u;
    uint8_t candidate_bit = candidate_assumptions[candidate * literal_count + literal];
    model_membership[gid] =
        (active_world != 0u && accepted_so_far != 0u && has_reduced_output != 0u && model_slot < reduction_count * models_per_reduction)
            ? candidate_bit
            : 0u;

    if (inner == 0u && active_world == 0u && rejection_reasons[candidate] == 0u) {
        rejection_reasons[candidate] = 4u;
    }
}

extern "C" __global__ void epistemic_populate_model_membership_from_tuple_source_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t reduction_count,
    uint32_t models_per_reduction,
    uint32_t world_stride,
    uint32_t literal_index,
    uint32_t reduction_index,
    const uint32_t* __restrict__ tuple_source_row_count,
    const uint8_t* __restrict__ candidate_assumptions,
    const uint8_t* __restrict__ world_views,
    uint8_t* __restrict__ model_membership,
    uint32_t* __restrict__ rejection_reasons
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = candidate_count * models_per_reduction;
    if (gid >= total) return;
    if (literal_index >= literal_count || reduction_index >= reduction_count) return;

    uint32_t candidate = gid / models_per_reduction;
    uint32_t model = gid - candidate * models_per_reduction;
    uint32_t membership_index =
        (((candidate * reduction_count + reduction_index) * models_per_reduction + model)
            * literal_count)
        + literal_index;

    uint8_t active_world = world_views[candidate * world_stride];
    uint8_t accepted_so_far = (rejection_reasons[candidate] == 0u) ? 1u : 0u;
    uint8_t has_tuple_source = (tuple_source_row_count[0] > 0u) ? 1u : 0u;
    uint8_t candidate_bit = candidate_assumptions[candidate * literal_count + literal_index];
    model_membership[membership_index] =
        (active_world != 0u && accepted_so_far != 0u && has_tuple_source != 0u)
            ? candidate_bit
            : 0u;

    if (model == 0u && active_world == 0u && rejection_reasons[candidate] == 0u) {
        rejection_reasons[candidate] = 4u;
    }
}

static __device__ uint8_t epistemic_tuple_key_cell_matches(
    const uint8_t* __restrict__ tuple_key_col,
    uint32_t tuple_key_col_width,
    uint32_t row,
    uint64_t expected_bits,
    uint8_t expected_type_code
) {
    if (tuple_key_col_width == 0u || tuple_key_col_width > 8u) return 0u;
    if (expected_type_code > 7u) return 0u;

    const volatile uint8_t* col = tuple_key_col;
    uint64_t actual_bits = 0u;
    uint32_t base = row * tuple_key_col_width;
    for (uint32_t byte = 0; byte < tuple_key_col_width; ++byte) {
        actual_bits |= static_cast<uint64_t>(col[base + byte]) << (byte * 8u);
    }

    uint64_t mask = (tuple_key_col_width >= 8u)
        ? 0xffffffffffffffffull
        : ((1ull << (tuple_key_col_width * 8u)) - 1ull);
    return ((actual_bits & mask) == (expected_bits & mask)) ? 1u : 0u;
}

static __device__ uint8_t epistemic_tuple_key_row_matches_arity1(
    const uint8_t* __restrict__ tuple_key_col0,
    uint32_t tuple_key_col0_width,
    uint64_t expected_key_col0_bits,
    uint8_t expected_key_col0_type_code,
    uint32_t row
) {
    return epistemic_tuple_key_cell_matches(
        tuple_key_col0,
        tuple_key_col0_width,
        row,
        expected_key_col0_bits,
        expected_key_col0_type_code
    );
}

extern "C" __global__ void epistemic_populate_model_membership_from_tuple_source_arity1_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t reduction_count,
    uint32_t models_per_reduction,
    uint32_t world_stride,
    uint32_t literal_index,
    uint32_t reduction_index,
    const uint32_t* __restrict__ tuple_source_row_count,
    const uint8_t* __restrict__ tuple_key_col0,
    uint32_t tuple_key_col0_width,
    uint64_t expected_key_col0_bits,
    uint8_t expected_key_col0_type_code,
    const uint8_t* __restrict__ candidate_assumptions,
    const uint8_t* __restrict__ world_views,
    uint8_t* __restrict__ model_membership,
    uint32_t* __restrict__ rejection_reasons
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = candidate_count * models_per_reduction;
    if (gid >= total) return;
    if (literal_index >= literal_count || reduction_index >= reduction_count) return;

    uint32_t candidate = gid / models_per_reduction;
    uint32_t model = gid - candidate * models_per_reduction;
    uint32_t membership_index =
        (((candidate * reduction_count + reduction_index) * models_per_reduction + model)
            * literal_count)
        + literal_index;

    uint32_t tuple_rows = tuple_source_row_count[0];
    uint8_t active_world = world_views[candidate * world_stride];
    uint8_t accepted_so_far = (rejection_reasons[candidate] == 0u) ? 1u : 0u;
    uint8_t has_tuple_source =
        (model < tuple_rows)
            ? epistemic_tuple_key_row_matches_arity1(
                  tuple_key_col0,
                  tuple_key_col0_width,
                  expected_key_col0_bits,
                  expected_key_col0_type_code,
                  model
              )
            : 0u;
    uint8_t candidate_bit = candidate_assumptions[candidate * literal_count + literal_index];
    model_membership[membership_index] =
        (active_world != 0u && accepted_so_far != 0u && has_tuple_source != 0u)
            ? candidate_bit
            : 0u;

    if (model == 0u && active_world == 0u && rejection_reasons[candidate] == 0u) {
        rejection_reasons[candidate] = 4u;
    }
}

static __device__ uint8_t epistemic_tuple_key_row_matches_arity2(
    const uint8_t* __restrict__ tuple_key_col0,
    uint32_t tuple_key_col0_width,
    uint64_t expected_key_col0_bits,
    uint8_t expected_key_col0_type_code,
    const uint8_t* __restrict__ tuple_key_col1,
    uint32_t tuple_key_col1_width,
    uint64_t expected_key_col1_bits,
    uint8_t expected_key_col1_type_code,
    uint32_t row
) {
    uint8_t col0_matches = epistemic_tuple_key_cell_matches(
        tuple_key_col0,
        tuple_key_col0_width,
        row,
        expected_key_col0_bits,
        expected_key_col0_type_code
    );
    uint8_t col1_matches = epistemic_tuple_key_cell_matches(
        tuple_key_col1,
        tuple_key_col1_width,
        row,
        expected_key_col1_bits,
        expected_key_col1_type_code
    );
    return (col0_matches != 0u && col1_matches != 0u) ? 1u : 0u;
}

extern "C" __global__ void epistemic_populate_model_membership_from_tuple_source_arity2_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t reduction_count,
    uint32_t models_per_reduction,
    uint32_t world_stride,
    uint32_t literal_index,
    uint32_t reduction_index,
    const uint32_t* __restrict__ tuple_source_row_count,
    const uint8_t* __restrict__ tuple_key_col0,
    uint32_t tuple_key_col0_width,
    uint64_t expected_key_col0_bits,
    uint8_t expected_key_col0_type_code,
    const uint8_t* __restrict__ tuple_key_col1,
    uint32_t tuple_key_col1_width,
    uint64_t expected_key_col1_bits,
    uint8_t expected_key_col1_type_code,
    const uint8_t* __restrict__ candidate_assumptions,
    const uint8_t* __restrict__ world_views,
    uint8_t* __restrict__ model_membership,
    uint32_t* __restrict__ rejection_reasons
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = candidate_count * models_per_reduction;
    if (gid >= total) return;
    if (literal_index >= literal_count || reduction_index >= reduction_count) return;

    uint32_t candidate = gid / models_per_reduction;
    uint32_t model = gid - candidate * models_per_reduction;
    uint32_t membership_index =
        (((candidate * reduction_count + reduction_index) * models_per_reduction + model)
            * literal_count)
        + literal_index;

    uint32_t tuple_rows = tuple_source_row_count[0];
    uint8_t active_world = world_views[candidate * world_stride];
    uint8_t accepted_so_far = (rejection_reasons[candidate] == 0u) ? 1u : 0u;
    uint8_t has_tuple_source =
        (model < tuple_rows)
            ? epistemic_tuple_key_row_matches_arity2(
                  tuple_key_col0,
                  tuple_key_col0_width,
                  expected_key_col0_bits,
                  expected_key_col0_type_code,
                  tuple_key_col1,
                  tuple_key_col1_width,
                  expected_key_col1_bits,
                  expected_key_col1_type_code,
                  model
              )
            : 0u;
    uint8_t candidate_bit = candidate_assumptions[candidate * literal_count + literal_index];
    model_membership[membership_index] =
        (active_world != 0u && accepted_so_far != 0u && has_tuple_source != 0u)
            ? candidate_bit
            : 0u;

    if (model == 0u && active_world == 0u && rejection_reasons[candidate] == 0u) {
        rejection_reasons[candidate] = 4u;
    }
}

static __device__ uint8_t epistemic_tuple_key_row_matches_arity3(
    const uint8_t* __restrict__ tuple_key_col0,
    uint32_t tuple_key_col0_width,
    uint64_t expected_key_col0_bits,
    uint8_t expected_key_col0_type_code,
    const uint8_t* __restrict__ tuple_key_col1,
    uint32_t tuple_key_col1_width,
    uint64_t expected_key_col1_bits,
    uint8_t expected_key_col1_type_code,
    const uint8_t* __restrict__ tuple_key_col2,
    uint32_t tuple_key_col2_width,
    uint64_t expected_key_col2_bits,
    uint8_t expected_key_col2_type_code,
    uint32_t row
) {
    uint8_t col0_matches = epistemic_tuple_key_cell_matches(
        tuple_key_col0,
        tuple_key_col0_width,
        row,
        expected_key_col0_bits,
        expected_key_col0_type_code
    );
    uint8_t col1_matches = epistemic_tuple_key_cell_matches(
        tuple_key_col1,
        tuple_key_col1_width,
        row,
        expected_key_col1_bits,
        expected_key_col1_type_code
    );
    uint8_t col2_matches = epistemic_tuple_key_cell_matches(
        tuple_key_col2,
        tuple_key_col2_width,
        row,
        expected_key_col2_bits,
        expected_key_col2_type_code
    );
    return (col0_matches != 0u && col1_matches != 0u && col2_matches != 0u) ? 1u : 0u;
}

extern "C" __global__ void epistemic_populate_model_membership_from_tuple_source_arity3_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t reduction_count,
    uint32_t models_per_reduction,
    uint32_t world_stride,
    uint32_t literal_index,
    uint32_t reduction_index,
    const uint32_t* __restrict__ tuple_source_row_count,
    const uint8_t* __restrict__ tuple_key_col0,
    uint32_t tuple_key_col0_width,
    uint64_t expected_key_col0_bits,
    uint8_t expected_key_col0_type_code,
    const uint8_t* __restrict__ tuple_key_col1,
    uint32_t tuple_key_col1_width,
    uint64_t expected_key_col1_bits,
    uint8_t expected_key_col1_type_code,
    const uint8_t* __restrict__ tuple_key_col2,
    uint32_t tuple_key_col2_width,
    uint64_t expected_key_col2_bits,
    uint8_t expected_key_col2_type_code,
    const uint8_t* __restrict__ candidate_assumptions,
    const uint8_t* __restrict__ world_views,
    uint8_t* __restrict__ model_membership,
    uint32_t* __restrict__ rejection_reasons
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = candidate_count * models_per_reduction;
    if (gid >= total) return;
    if (literal_index >= literal_count || reduction_index >= reduction_count) return;

    uint32_t candidate = gid / models_per_reduction;
    uint32_t model = gid - candidate * models_per_reduction;
    uint32_t membership_index =
        (((candidate * reduction_count + reduction_index) * models_per_reduction + model)
            * literal_count)
        + literal_index;

    uint32_t tuple_rows = tuple_source_row_count[0];
    uint8_t active_world = world_views[candidate * world_stride];
    uint8_t accepted_so_far = (rejection_reasons[candidate] == 0u) ? 1u : 0u;
    uint8_t has_tuple_source =
        (model < tuple_rows)
            ? epistemic_tuple_key_row_matches_arity3(
                  tuple_key_col0,
                  tuple_key_col0_width,
                  expected_key_col0_bits,
                  expected_key_col0_type_code,
                  tuple_key_col1,
                  tuple_key_col1_width,
                  expected_key_col1_bits,
                  expected_key_col1_type_code,
                  tuple_key_col2,
                  tuple_key_col2_width,
                  expected_key_col2_bits,
                  expected_key_col2_type_code,
                  model
              )
            : 0u;
    uint8_t candidate_bit = candidate_assumptions[candidate * literal_count + literal_index];
    model_membership[membership_index] =
        (active_world != 0u && accepted_so_far != 0u && has_tuple_source != 0u)
            ? candidate_bit
            : 0u;

    if (model == 0u && active_world == 0u && rejection_reasons[candidate] == 0u) {
        rejection_reasons[candidate] = 4u;
    }
}

static __device__ uint8_t epistemic_tuple_key_row_matches_arity_n(
    const uint64_t* __restrict__ tuple_key_col_ptrs,
    const uint32_t* __restrict__ tuple_key_col_widths,
    const uint64_t* __restrict__ expected_key_bits,
    const uint8_t* __restrict__ expected_key_type_codes,
    uint32_t key_col_count,
    uint32_t row
) {
    if (key_col_count == 0u) return 1u;

    for (uint32_t key = 0u; key < key_col_count; ++key) {
        const uint8_t* tuple_key_col =
            reinterpret_cast<const uint8_t*>(tuple_key_col_ptrs[key]);
        if (tuple_key_col == nullptr) return 0u;

        uint8_t matches = epistemic_tuple_key_cell_matches(
            tuple_key_col,
            tuple_key_col_widths[key],
            row,
            expected_key_bits[key],
            expected_key_type_codes[key]
        );
        if (matches == 0u) return 0u;
    }

    return 1u;
}

extern "C" __global__ void epistemic_populate_model_membership_from_tuple_source_arity_n_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t reduction_count,
    uint32_t models_per_reduction,
    uint32_t world_stride,
    uint32_t literal_index,
    uint32_t reduction_index,
    const uint32_t* __restrict__ tuple_source_row_count,
    const uint64_t* __restrict__ tuple_key_col_ptrs,
    const uint32_t* __restrict__ tuple_key_col_widths,
    const uint64_t* __restrict__ expected_key_bits,
    const uint8_t* __restrict__ expected_key_type_codes,
    uint32_t key_col_count,
    const uint8_t* __restrict__ candidate_assumptions,
    const uint8_t* __restrict__ world_views,
    uint8_t* __restrict__ model_membership,
    uint32_t* __restrict__ rejection_reasons
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = candidate_count * models_per_reduction;
    if (gid >= total) return;
    if (literal_index >= literal_count || reduction_index >= reduction_count) return;

    uint32_t candidate = gid / models_per_reduction;
    uint32_t model = gid - candidate * models_per_reduction;
    uint32_t membership_index =
        (((candidate * reduction_count + reduction_index) * models_per_reduction + model)
            * literal_count)
        + literal_index;

    uint32_t tuple_rows = tuple_source_row_count[0];
    uint8_t active_world = world_views[candidate * world_stride];
    uint8_t accepted_so_far = (rejection_reasons[candidate] == 0u) ? 1u : 0u;
    uint8_t has_tuple_source =
        (model < tuple_rows)
            ? epistemic_tuple_key_row_matches_arity_n(
                  tuple_key_col_ptrs,
                  tuple_key_col_widths,
                  expected_key_bits,
                  expected_key_type_codes,
                  key_col_count,
                  model
              )
            : 0u;
    uint8_t candidate_bit = candidate_assumptions[candidate * literal_count + literal_index];
    model_membership[membership_index] =
        (active_world != 0u && accepted_so_far != 0u && has_tuple_source != 0u)
            ? candidate_bit
            : 0u;

    if (model == 0u && active_world == 0u && rejection_reasons[candidate] == 0u) {
        rejection_reasons[candidate] = 4u;
    }
}

extern "C" __global__ void epistemic_validate_world_views_u8(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t reduction_count,
    uint32_t models_per_reduction,
    uint32_t world_stride,
    const uint8_t* __restrict__ model_membership,
    const uint8_t* __restrict__ world_views,
    uint32_t* __restrict__ rejection_reasons
) {
    uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) return;

    if (rejection_reasons[candidate] != 0u) return;

    uint8_t active_world = world_views[candidate * world_stride];
    if (active_world == 0u) {
        rejection_reasons[candidate] = 5u;
        return;
    }

    uint32_t per_candidate = reduction_count * models_per_reduction * literal_count;
    uint32_t base = candidate * per_candidate;
    uint8_t observed_membership = 0u;
    for (uint32_t offset = 0; offset < per_candidate; ++offset) {
        observed_membership |= model_membership[base + offset];
    }

    if (observed_membership == 0u) {
        rejection_reasons[candidate] = 5u;
    }
}

extern "C" __global__ void epistemic_materialize_accepted_candidates_u8(
    uint32_t candidate_count,
    uint32_t world_stride,
    const uint32_t* __restrict__ rejection_reasons,
    uint8_t* __restrict__ world_views
) {
    uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) return;

    world_views[candidate * world_stride] =
        (rejection_reasons[candidate] == 0u) ? 1u : 0u;
}

extern "C" __global__ void epistemic_materialize_final_result_flags_u8(
    uint32_t candidate_count,
    uint32_t world_stride,
    const uint32_t* __restrict__ output_row_count,
    const uint32_t* __restrict__ rejection_reasons,
    uint8_t* __restrict__ world_views
) {
    uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count) return;

    uint8_t has_output = (output_row_count[0] > 0u) ? 1u : 0u;
    world_views[candidate * world_stride] =
        (rejection_reasons[candidate] == 0u && has_output != 0u) ? 1u : 0u;
}

extern "C" __global__ void epistemic_materialize_final_tuple_column_u8(
    uint32_t column_byte_len,
    uint32_t candidate_count,
    const uint32_t* __restrict__ output_row_count,
    const uint32_t* __restrict__ rejection_reasons,
    const uint8_t* __restrict__ source_column,
    uint8_t* __restrict__ final_column,
    uint32_t* __restrict__ final_row_count
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    uint8_t accepted_candidate = 0u;
    for (uint32_t candidate = 0; candidate < candidate_count; ++candidate) {
        accepted_candidate |= (rejection_reasons[candidate] == 0u) ? 1u : 0u;
    }

    uint32_t rows = output_row_count[0];
    uint8_t materialize = (accepted_candidate != 0u && rows != 0u) ? 1u : 0u;
    if (gid == 0u) {
        final_row_count[0] = (materialize != 0u) ? rows : 0u;
    }

    if (gid >= column_byte_len) return;
    final_column[gid] = (materialize != 0u) ? source_column[gid] : 0u;
}
