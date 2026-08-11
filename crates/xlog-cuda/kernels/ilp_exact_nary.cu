#include <stdint.h>

// ilp_exact_nary — n-ary bounded exact-induction scoring kernel.
//
// Scores a batch of flattened n-ary rule patterns against positive and
// negative example tuples in one launch. Each block owns one pattern
// (blockIdx.x); its threads cooperatively stride the example tuples and
// evaluate coverage with an iterative backtracking walk over the
// pattern's body atoms.
//
// The algorithm is the DEVICE transcription of
// `xlog_induce::nary_layout::score_pattern_flat` — same state (row
// cursors, join values, per-depth bound masks), same visit order. That
// host walk is pinned against the recursive reference scorer
// (`nary_reference.rs`) on CPU; this kernel must reproduce the host walk
// bit-for-bit, which the GPU parity test witnesses on real hardware.
//
// Layout contract (must equal nary_layout.rs):
//   * binding code u32: bit 31 set => join variable, clear => head
//     position; low 8 bits carry the index.
//   * bounds: at most 8 body atoms, 8 join variables, atom arity <= 8.
//     The host flattener REFUSES anything larger, so fixed-size local
//     state here is safe by construction.
//   * candidate relations ride as one concatenated row-major u64 buffer
//     with per-relation element offsets, arities and row counts.
//   * example tuples are row-major u64 with stride = head arity, which
//     is uniform across one launch (one induce call = one target
//     relation).

#define ILP_EXACT_NARY_BLOCK_SIZE 256u

#define ILP_EXACT_NARY_MAX_BODY_ATOMS 8u
#define ILP_EXACT_NARY_MAX_JOIN_VARS 8u

#define ILP_EXACT_NARY_JOIN_FLAG 0x80000000u
#define ILP_EXACT_NARY_INDEX_MASK 0xFFu

// Does the pattern owned by this block cover one example tuple?
//
// Iterative backtracking: depth = body atom index. Each depth records the
// row it is trying and the bitmask of join variables that row newly
// bound; unwinding a depth clears exactly that mask and advances the row
// cursor. Join values live in one array and are meaningful only while
// their bit is set in `bound`.
__device__ inline uint32_t ilp_exact_nary_covers(
    uint32_t body_offset,
    uint32_t body_len,
    const uint32_t* atom_candidate_slot,
    const uint32_t* atom_arity,
    const uint32_t* atom_binding_offset,
    const uint32_t* binding_codes,
    const uint64_t* cand_values,
    const uint32_t* cand_value_offset,
    const uint32_t* cand_rows,
    const uint64_t* example
) {
    uint64_t joins[ILP_EXACT_NARY_MAX_JOIN_VARS];
    uint32_t bound = 0u;
    uint32_t row_cursor[ILP_EXACT_NARY_MAX_BODY_ATOMS];
    uint32_t depth_mask[ILP_EXACT_NARY_MAX_BODY_ATOMS];

    uint32_t depth = 0u;
    row_cursor[0] = 0u;

    for (;;) {
        if (depth == body_len) return 1u;

        uint32_t atom = body_offset + depth;
        uint32_t slot = atom_candidate_slot[atom];
        uint32_t arity = atom_arity[atom];
        uint32_t bindings = atom_binding_offset[atom];
        const uint64_t* rows = cand_values + cand_value_offset[slot];
        uint32_t row_count = cand_rows[slot];

        uint32_t descended = 0u;
        while (row_cursor[depth] < row_count) {
            uint32_t row = row_cursor[depth];
            const uint64_t* values = rows + (size_t)row * arity;
            uint32_t mask = 0u;
            uint32_t matched = 1u;
            for (uint32_t position = 0; position < arity; position++) {
                uint64_t value = values[position];
                uint32_t code = binding_codes[bindings + position];
                uint32_t index = code & ILP_EXACT_NARY_INDEX_MASK;
                if (code & ILP_EXACT_NARY_JOIN_FLAG) {
                    uint32_t bit = 1u << index;
                    if (bound & bit) {
                        if (joins[index] != value) { matched = 0u; break; }
                    } else {
                        joins[index] = value;
                        bound |= bit;
                        mask |= bit;
                    }
                } else {
                    if (example[index] != value) { matched = 0u; break; }
                }
            }
            if (matched) {
                depth_mask[depth] = mask;
                depth++;
                if (depth < body_len) row_cursor[depth] = 0u;
                descended = 1u;
                break;
            }
            bound &= ~mask;
            row_cursor[depth]++;
        }
        if (descended) continue;

        // Depth exhausted: unwind one level and retry its next row.
        if (depth == 0u) return 0u;
        depth--;
        bound &= ~depth_mask[depth];
        row_cursor[depth]++;
    }
}

// Score every pattern of the batch: block = pattern, threads stride the
// positive and negative example tuples, block-reduce the local counts.
//
// The pattern batch rides as ONE packed u32 buffer (the launch ABI caps
// the argument tuple, and six parallel arrays would blow it). Section
// layout, with P = params[0] patterns and A = params[1] atoms:
//   batch[0        .. P)        body_offset
//   batch[P        .. 2P)       body_len
//   batch[2P       .. 2P+A)     atom_candidate_slot
//   batch[2P+A     .. 2P+2A)    atom_arity
//   batch[2P+2A    .. 2P+3A)    atom_binding_offset
//   batch[2P+3A    .. ]         binding_codes
// params = [num_patterns, num_atoms, num_pos, num_neg, head_arity].
// The host launcher (provider/ilp_exact_nary.rs) packs exactly this.
extern "C" __global__ void ilp_exact_nary_score(
    const uint32_t* batch,
    const uint32_t* params,
    // Candidate relations (concatenated row-major u64 rows).
    const uint64_t* cand_values,
    const uint32_t* cand_value_offset,
    const uint32_t* cand_rows,
    // Example tuples (row-major, stride = head_arity).
    const uint64_t* pos_values,
    const uint64_t* neg_values,
    // Outputs, one slot per pattern.
    uint32_t* pos_covered,
    uint32_t* neg_covered
) {
    uint32_t num_patterns = params[0];
    uint32_t num_atoms = params[1];
    uint32_t num_pos = params[2];
    uint32_t num_neg = params[3];
    uint32_t head_arity = params[4];

    const uint32_t* body_offset = batch;
    const uint32_t* body_len = batch + num_patterns;
    const uint32_t* atom_candidate_slot = batch + 2u * num_patterns;
    const uint32_t* atom_arity = atom_candidate_slot + num_atoms;
    const uint32_t* atom_binding_offset = atom_arity + num_atoms;
    const uint32_t* binding_codes = atom_binding_offset + num_atoms;

    uint32_t pattern = blockIdx.x;
    if (pattern >= num_patterns) return;
    uint32_t tid = threadIdx.x;

    uint32_t p_body_offset = body_offset[pattern];
    uint32_t p_body_len = body_len[pattern];

    uint32_t local_pos = 0u;
    for (uint32_t q = tid; q < num_pos; q += blockDim.x) {
        local_pos += ilp_exact_nary_covers(
            p_body_offset, p_body_len,
            atom_candidate_slot, atom_arity, atom_binding_offset,
            binding_codes,
            cand_values, cand_value_offset, cand_rows,
            pos_values + (size_t)q * head_arity);
    }
    uint32_t local_neg = 0u;
    for (uint32_t q = tid; q < num_neg; q += blockDim.x) {
        local_neg += ilp_exact_nary_covers(
            p_body_offset, p_body_len,
            atom_candidate_slot, atom_arity, atom_binding_offset,
            binding_codes,
            cand_values, cand_value_offset, cand_rows,
            neg_values + (size_t)q * head_arity);
    }

    __shared__ uint32_t pos_scratch[ILP_EXACT_NARY_BLOCK_SIZE];
    __shared__ uint32_t neg_scratch[ILP_EXACT_NARY_BLOCK_SIZE];
    pos_scratch[tid] = local_pos;
    neg_scratch[tid] = local_neg;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            pos_scratch[tid] += pos_scratch[tid + s];
            neg_scratch[tid] += neg_scratch[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0u) {
        pos_covered[pattern] = pos_scratch[0];
        neg_covered[pattern] = neg_scratch[0];
    }
}
