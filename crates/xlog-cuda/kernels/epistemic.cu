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
    uint8_t negated,
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
    uint8_t literal_holds =
        (negated != 0u) ? ((has_tuple_source == 0u) ? 1u : 0u) : has_tuple_source;
    uint8_t candidate_bit = candidate_assumptions[candidate * literal_count + literal_index];
    model_membership[membership_index] =
        (active_world != 0u && accepted_so_far != 0u && literal_holds != 0u)
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

static __device__ uint8_t epistemic_tuple_key_bound_cell_matches(
    const uint8_t* __restrict__ tuple_key_col,
    uint32_t tuple_key_col_width,
    uint32_t tuple_row,
    const uint8_t* __restrict__ bound_value_col,
    uint32_t bound_value_col_width,
    uint32_t bound_row
) {
    if (tuple_key_col == nullptr || bound_value_col == nullptr) return 0u;
    if (tuple_key_col_width == 0u || tuple_key_col_width > 8u) return 0u;
    if (tuple_key_col_width != bound_value_col_width) return 0u;

    const volatile uint8_t* tuple_col = tuple_key_col;
    const volatile uint8_t* bound_col = bound_value_col;
    uint32_t tuple_base = tuple_row * tuple_key_col_width;
    uint32_t bound_base = bound_row * bound_value_col_width;
    for (uint32_t byte = 0; byte < tuple_key_col_width; ++byte) {
        if (tuple_col[tuple_base + byte] != bound_col[bound_base + byte]) return 0u;
    }

    return 1u;
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
    uint8_t negated,
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
    uint8_t tuple_matches = 0u;
    if (model == 0u) {
        for (uint32_t tuple_row = 0u; tuple_row < tuple_rows; ++tuple_row) {
            if (epistemic_tuple_key_row_matches_arity1(
                    tuple_key_col0,
                    tuple_key_col0_width,
                    expected_key_col0_bits,
                    expected_key_col0_type_code,
                    tuple_row
                ) != 0u) {
                tuple_matches = 1u;
                break;
            }
        }
    }
    uint8_t literal_holds =
        (model == 0u)
            ? ((negated != 0u) ? ((tuple_matches == 0u) ? 1u : 0u) : tuple_matches)
            : 0u;
    uint8_t candidate_bit = candidate_assumptions[candidate * literal_count + literal_index];
    model_membership[membership_index] =
        (active_world != 0u && accepted_so_far != 0u && literal_holds != 0u)
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
    uint8_t negated,
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
    uint8_t tuple_matches = 0u;
    if (model == 0u) {
        for (uint32_t tuple_row = 0u; tuple_row < tuple_rows; ++tuple_row) {
            if (epistemic_tuple_key_row_matches_arity2(
                    tuple_key_col0,
                    tuple_key_col0_width,
                    expected_key_col0_bits,
                    expected_key_col0_type_code,
                    tuple_key_col1,
                    tuple_key_col1_width,
                    expected_key_col1_bits,
                    expected_key_col1_type_code,
                    tuple_row
                ) != 0u) {
                tuple_matches = 1u;
                break;
            }
        }
    }
    uint8_t literal_holds =
        (model == 0u)
            ? ((negated != 0u) ? ((tuple_matches == 0u) ? 1u : 0u) : tuple_matches)
            : 0u;
    uint8_t candidate_bit = candidate_assumptions[candidate * literal_count + literal_index];
    model_membership[membership_index] =
        (active_world != 0u && accepted_so_far != 0u && literal_holds != 0u)
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
    uint8_t negated,
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
    uint8_t tuple_matches = 0u;
    if (model == 0u) {
        for (uint32_t tuple_row = 0u; tuple_row < tuple_rows; ++tuple_row) {
            if (epistemic_tuple_key_row_matches_arity3(
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
                    tuple_row
                ) != 0u) {
                tuple_matches = 1u;
                break;
            }
        }
    }
    uint8_t literal_holds =
        (model == 0u)
            ? ((negated != 0u) ? ((tuple_matches == 0u) ? 1u : 0u) : tuple_matches)
            : 0u;
    uint8_t candidate_bit = candidate_assumptions[candidate * literal_count + literal_index];
    model_membership[membership_index] =
        (active_world != 0u && accepted_so_far != 0u && literal_holds != 0u)
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
    const uint8_t* __restrict__ tuple_key_match_modes,
    const uint64_t* __restrict__ bound_value_col_ptrs,
    const uint32_t* __restrict__ bound_value_col_widths,
    uint32_t key_col_count,
    uint32_t tuple_row,
    uint32_t bound_row
) {
    if (key_col_count == 0u) return 1u;

    for (uint32_t key = 0u; key < key_col_count; ++key) {
        const uint8_t* tuple_key_col =
            reinterpret_cast<const uint8_t*>(tuple_key_col_ptrs[key]);
        if (tuple_key_col == nullptr) return 0u;

        uint8_t match_mode = tuple_key_match_modes[key];
        uint8_t matches = 0u;
        if (match_mode == 1u) {
            const uint8_t* bound_value_col =
                reinterpret_cast<const uint8_t*>(bound_value_col_ptrs[key]);
            matches = epistemic_tuple_key_bound_cell_matches(
                tuple_key_col,
                tuple_key_col_widths[key],
                tuple_row,
                bound_value_col,
                bound_value_col_widths[key],
                bound_row
            );
        } else if (match_mode == 0u) {
            matches = epistemic_tuple_key_cell_matches(
                tuple_key_col,
                tuple_key_col_widths[key],
                tuple_row,
                expected_key_bits[key],
                expected_key_type_codes[key]
            );
        }
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
    uint8_t negated,
    const uint32_t* __restrict__ tuple_source_row_count,
    const uint64_t* __restrict__ tuple_key_col_ptrs,
    const uint32_t* __restrict__ tuple_key_col_widths,
    const uint64_t* __restrict__ expected_key_bits,
    const uint8_t* __restrict__ expected_key_type_codes,
    const uint8_t* __restrict__ tuple_key_match_modes,
    const uint64_t* __restrict__ bound_value_col_ptrs,
    const uint32_t* __restrict__ bound_value_col_widths,
    const uint32_t* __restrict__ bound_value_row_count,
    uint32_t key_col_count,
    uint8_t has_bound_value_keys,
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
    uint32_t bound_rows =
        (has_bound_value_keys != 0u && bound_value_row_count != nullptr)
            ? bound_value_row_count[0]
            : 0u;
    uint8_t active_world = world_views[candidate * world_stride];
    uint8_t accepted_so_far = (rejection_reasons[candidate] == 0u) ? 1u : 0u;
    uint8_t row_in_scope =
        (has_bound_value_keys != 0u) ? ((model < bound_rows) ? 1u : 0u)
                                    : ((model == 0u) ? 1u : 0u);
    uint8_t tuple_matches = 0u;
    if (row_in_scope != 0u) {
        uint32_t bound_row = (has_bound_value_keys != 0u) ? model : 0u;
        for (uint32_t tuple_row = 0u; tuple_row < tuple_rows; ++tuple_row) {
            if (epistemic_tuple_key_row_matches_arity_n(
                    tuple_key_col_ptrs,
                    tuple_key_col_widths,
                    expected_key_bits,
                    expected_key_type_codes,
                    tuple_key_match_modes,
                    bound_value_col_ptrs,
                    bound_value_col_widths,
                    key_col_count,
                    tuple_row,
                    bound_row
                ) != 0u) {
                tuple_matches = 1u;
                break;
            }
        }
    }
    uint8_t literal_holds =
        (row_in_scope != 0u)
            ? ((negated != 0u) ? ((tuple_matches == 0u) ? 1u : 0u) : tuple_matches)
            : 0u;
    uint8_t candidate_bit = candidate_assumptions[candidate * literal_count + literal_index];
    model_membership[membership_index] =
        (active_world != 0u && accepted_so_far != 0u && literal_holds != 0u)
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
    const uint8_t* __restrict__ candidate_assumptions,
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
    uint32_t assumption_base = candidate * literal_count;
    uint8_t complete_membership = 1u;
    for (uint32_t literal = 0u; literal < literal_count; ++literal) {
        uint8_t candidate_bit = candidate_assumptions[assumption_base + literal];
        uint8_t literal_supported = 0u;
        if (candidate_bit != 0u) {
            for (uint32_t reduction = 0u; reduction < reduction_count; ++reduction) {
                for (uint32_t model = 0u; model < models_per_reduction; ++model) {
                    uint32_t offset =
                        ((reduction * models_per_reduction + model) * literal_count) + literal;
                    literal_supported |= model_membership[base + offset];
                }
            }
        }
        if (candidate_bit == 0u || literal_supported == 0u) {
            complete_membership = 0u;
        }
    }

    if (complete_membership == 0u) {
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

static __device__ uint8_t epistemic_final_tuple_has_accepted_membership(
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t reduction_count,
    uint32_t models_per_reduction,
    uint32_t world_stride,
    const uint32_t* __restrict__ rejection_reasons,
    const uint8_t* __restrict__ model_membership,
    const uint8_t* __restrict__ world_views
) {
    if (literal_count == 0u || reduction_count == 0u || models_per_reduction == 0u) return 0u;

    uint32_t per_candidate = reduction_count * models_per_reduction * literal_count;
    for (uint32_t candidate = 0u; candidate < candidate_count; ++candidate) {
        if (rejection_reasons[candidate] != 0u) continue;
        if (world_views[candidate * world_stride] == 0u) continue;

        uint32_t base = candidate * per_candidate;
        for (uint32_t offset = 0u; offset < per_candidate; ++offset) {
            if (model_membership[base + offset] != 0u) return 1u;
        }
    }

    return 0u;
}

extern "C" __global__ void epistemic_build_final_tuple_row_map_u8(
    uint32_t output_row_capacity,
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t reduction_count,
    uint32_t models_per_reduction,
    uint32_t world_stride,
    const uint32_t* __restrict__ output_row_count,
    const uint32_t* __restrict__ rejection_reasons,
    const uint8_t* __restrict__ model_membership,
    const uint8_t* __restrict__ world_views,
    const uint64_t* __restrict__ tuple_source_row_count_ptrs,
    const uint8_t* __restrict__ row_filter_negated,
    const uint32_t* __restrict__ row_filter_key_offsets,
    const uint32_t* __restrict__ row_filter_key_counts,
    const uint64_t* __restrict__ tuple_key_col_ptrs,
    const uint32_t* __restrict__ tuple_key_col_widths,
    const uint64_t* __restrict__ expected_key_bits,
    const uint8_t* __restrict__ expected_key_type_codes,
    const uint8_t* __restrict__ tuple_key_match_modes,
    const uint64_t* __restrict__ bound_value_col_ptrs,
    const uint32_t* __restrict__ bound_value_col_widths,
    uint32_t row_filter_count,
    uint32_t* __restrict__ row_map,
    uint32_t* __restrict__ final_row_count
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rows = output_row_count[0];
    if (row >= rows || row >= output_row_capacity) return;

    uint8_t accepted_membership = epistemic_final_tuple_has_accepted_membership(
        literal_count,
        candidate_count,
        reduction_count,
        models_per_reduction,
        world_stride,
        rejection_reasons,
        model_membership,
        world_views
    );
    if (accepted_membership == 0u) return;

    uint8_t row_matches = 1u;
    for (uint32_t filter = 0u; filter < row_filter_count; ++filter) {
        const uint32_t* tuple_source_row_count =
            reinterpret_cast<const uint32_t*>(tuple_source_row_count_ptrs[filter]);
        if (tuple_source_row_count == nullptr) {
            row_matches = 0u;
            break;
        }

        uint8_t filter_matches = 0u;
        uint32_t tuple_rows = tuple_source_row_count[0];
        uint32_t key_offset = row_filter_key_offsets[filter];
        uint32_t key_col_count = row_filter_key_counts[filter];
        for (uint32_t tuple_row = 0u; tuple_row < tuple_rows; ++tuple_row) {
            if (epistemic_tuple_key_row_matches_arity_n(
                    tuple_key_col_ptrs + key_offset,
                    tuple_key_col_widths + key_offset,
                    expected_key_bits + key_offset,
                    expected_key_type_codes + key_offset,
                    tuple_key_match_modes + key_offset,
                    bound_value_col_ptrs + key_offset,
                    bound_value_col_widths + key_offset,
                    key_col_count,
                    tuple_row,
                    row
                ) != 0u) {
                filter_matches = 1u;
                break;
            }
        }
        uint8_t filter_holds =
            (row_filter_negated[filter] != 0u)
                ? ((filter_matches == 0u) ? 1u : 0u)
                : filter_matches;
        if (filter_holds == 0u) {
            row_matches = 0u;
            break;
        }
    }

    if (row_matches != 0u) {
        uint32_t out_row = atomicAdd(final_row_count, 1u);
        if (out_row < output_row_capacity) {
            row_map[out_row] = row;
        }
    }
}

extern "C" __global__ void epistemic_materialize_final_tuple_column_u8(
    uint32_t column_byte_len,
    uint32_t literal_count,
    uint32_t candidate_count,
    uint32_t reduction_count,
    uint32_t models_per_reduction,
    uint32_t world_stride,
    const uint32_t* __restrict__ output_row_count,
    const uint32_t* __restrict__ rejection_reasons,
    const uint8_t* __restrict__ model_membership,
    const uint8_t* __restrict__ world_views,
    const uint32_t* __restrict__ row_map,
    const uint8_t* __restrict__ source_column,
    uint8_t* __restrict__ final_column,
    uint32_t* __restrict__ final_row_count
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t rows = output_row_count[0];
    uint32_t final_rows = final_row_count[0];
    if (rows == 0u || final_rows == 0u || column_byte_len == 0u) return;

    uint32_t row_width = column_byte_len / rows;
    if (row_width == 0u) return;
    uint32_t total = final_rows * row_width;
    if (gid >= total) return;

    uint32_t dst_row = gid / row_width;
    uint32_t byte = gid - dst_row * row_width;
    uint32_t src_row = row_map[dst_row];
    if (src_row >= rows) return;
    final_column[gid] = source_column[src_row * row_width + byte];
}
