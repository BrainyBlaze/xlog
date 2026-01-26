// kernels/d4.cu
//
// GPU-native D4 compiler support kernels (Phase 1).
//
// Design constraints:
// - GPU-native data-plane: no device->host copies for CNF/circuit data
// - Deterministic: stable behavior and failure modes
// - Memory-bounded: fixed-capacity buffers; overflow => hard trap
//
// Primary spec:
// docs/design/2026-01-22-gpu-native-compilation-design.md (Sections 2, 2.5, 5.2.4)

#include <cstdint>
#include <cstddef>

// Fail-fast trap (matches kernels/sat.cu semantics).
__device__ __forceinline__ void d4_trap() {
    asm volatile("trap;");
}

// ---------------------------------------------------------------------------
// Literal helpers (DIMACS: +/- var_id, 1-based var ids; 0 unused)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t d4_var(int32_t lit) {
    return (lit > 0) ? static_cast<uint32_t>(lit) : static_cast<uint32_t>(-lit);
}

// ---------------------------------------------------------------------------
// D4 frontier work items (must match Rust `#[repr(C)]` layout)
// ---------------------------------------------------------------------------

struct D4WorkItem {
    uint32_t subproblem_id;
    uint32_t parent_node;
    uint8_t branch;
    uint16_t depth;
    uint32_t assignment_offset;
};

// ---------------------------------------------------------------------------
// CNF invariant validation
// ---------------------------------------------------------------------------

// Validate basic CSR invariants for a device-resident CNF.
//
// This kernel is used to enforce "no silent success" semantics: invalid inputs
// must fail-fast on device without relying on host reads.
extern "C" __global__ void d4_validate_cnf(
    uint32_t var_cap,
    uint32_t clause_cap,
    uint32_t lit_cap,
    const uint32_t* __restrict__ num_vars,    // len=1
    const uint32_t* __restrict__ num_clauses, // len=1
    const uint32_t* __restrict__ num_lits,    // len=1
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ literals
) {
    // Single-threaded: this is a debug/invariant check, not a hot kernel.
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    uint32_t nv = num_vars[0];
    uint32_t nc = num_clauses[0];
    uint32_t nl = num_lits[0];

    if (nv > var_cap || nc > clause_cap || nl > lit_cap) {
        d4_trap();
    }

    // Offsets must start at 0 and terminate at num_lits for the active clause range.
    if (nc > 0u && clause_offsets[0] != 0u) {
        d4_trap();
    }
    if (clause_offsets[nc] != nl) {
        d4_trap();
    }

    // Validate monotone offsets and literal bounds for active clauses.
    for (uint32_t c = 0; c < nc; c++) {
        uint32_t s = clause_offsets[c];
        uint32_t e = clause_offsets[c + 1u];
        if (s > e || e > nl) {
            d4_trap();
        }

        for (uint32_t i = s; i < e; i++) {
            int32_t lit = literals[i];
            if (lit == 0) {
                d4_trap();
            }
            uint32_t v = d4_var(lit);
            if (v == 0u || v > nv) {
                d4_trap();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Levelization helpers
// ---------------------------------------------------------------------------

// Compute per-level node counts from a device-resident `node_level[]` table.
//
// The output `level_counts[level]` is intended to be exclusive-scanned on device
// to form `level_offsets` for XGCF evaluation.
extern "C" __global__ void d4_levelize_counts(
    const uint32_t* __restrict__ node_level,
    uint32_t num_nodes,
    uint32_t num_levels,
    uint32_t* __restrict__ level_counts
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t node = tid; node < num_nodes; node += blockDim.x * gridDim.x) {
        uint32_t lvl = node_level[node];
        if (lvl >= num_levels) {
            d4_trap();
        }
        atomicAdd(&level_counts[lvl], 1u);
    }
}

// Emit `level_nodes` given `level_offsets` and per-level cursors.
//
// Notes:
// - `level_offsets` must be an exclusive scan of `level_counts` with length `num_levels + 1`.
// - `level_cursors` must be initialized to 0 before launch (len = num_levels).
// - Emission order within a level is unspecified; correctness does not depend on it.
extern "C" __global__ void d4_levelize_emit(
    const uint32_t* __restrict__ node_level,
    uint32_t num_nodes,
    uint32_t num_levels,
    const uint32_t* __restrict__ level_offsets, // len = num_levels + 1
    uint32_t* __restrict__ level_cursors,       // len = num_levels (must start at 0)
    uint32_t* __restrict__ level_nodes          // len = num_nodes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t node = tid; node < num_nodes; node += blockDim.x * gridDim.x) {
        uint32_t lvl = node_level[node];
        if (lvl >= num_levels) {
            d4_trap();
        }
        uint32_t idx = atomicAdd(&level_cursors[lvl], 1u);
        uint32_t pos = level_offsets[lvl] + idx;
        if (pos >= level_offsets[lvl + 1u] || pos >= num_nodes) {
            d4_trap();
        }
        level_nodes[pos] = node;
    }
}

// ---------------------------------------------------------------------------
// Frontier expansion (Phase 1): unit propagation + deterministic branching
// ---------------------------------------------------------------------------

// Assignment encoding for the dense form: 0=unassigned, 1=true, 2=false.
static constexpr uint8_t D4_ASSIGN_UNASSIGNED = 0;
static constexpr uint8_t D4_ASSIGN_TRUE = 1;
static constexpr uint8_t D4_ASSIGN_FALSE = 2;

// Evaluate a bitset assignment for a DIMACS literal.
// Returns: +1 satisfied, 0 unassigned, -1 falsified.
__device__ __forceinline__ int8_t d4_eval_lit_bitset(
    int32_t lit,
    const uint32_t* __restrict__ true_bits,
    const uint32_t* __restrict__ false_bits,
    uint32_t base_word,
    uint32_t words_per_item
) {
    uint32_t v = d4_var(lit);
    uint32_t wi = v >> 5;
    if (wi >= words_per_item) {
        d4_trap();
    }
    uint32_t mask = 1u << (v & 31u);
    bool t = (true_bits[base_word + wi] & mask) != 0u;
    bool f = (false_bits[base_word + wi] & mask) != 0u;
    if (t && f) {
        // Assignment representation must never contain contradictions.
        d4_trap();
    }
    if (!t && !f) {
        return 0;
    }
    bool var_true = t;
    bool lit_pos = (lit > 0);
    bool sat = (lit_pos && var_true) || (!lit_pos && !var_true);
    return sat ? 1 : -1;
}

__device__ __forceinline__ bool d4_assign_var_bitset(
    int32_t lit,
    uint32_t* __restrict__ true_bits,
    uint32_t* __restrict__ false_bits,
    uint32_t base_word,
    uint32_t words_per_item
) {
    uint32_t v = d4_var(lit);
    uint32_t wi = v >> 5;
    if (wi >= words_per_item) {
        d4_trap();
    }
    uint32_t mask = 1u << (v & 31u);
    uint32_t* tword = &true_bits[base_word + wi];
    uint32_t* fword = &false_bits[base_word + wi];
    bool t = (*tword & mask) != 0u;
    bool f = (*fword & mask) != 0u;
    if (t && f) {
        d4_trap();
    }
    if (lit > 0) {
        if (f) return false;
        *tword |= mask;
        return true;
    } else {
        if (t) return false;
        *fword |= mask;
        return true;
    }
}

__device__ __forceinline__ int8_t d4_eval_lit_dense(
    int32_t lit,
    const uint8_t* __restrict__ assign,
    uint32_t base_byte
) {
    uint32_t v = d4_var(lit);
    uint8_t a = assign[base_byte + v];
    if (a == D4_ASSIGN_UNASSIGNED) {
        return 0;
    }
    if (a != D4_ASSIGN_TRUE && a != D4_ASSIGN_FALSE) {
        d4_trap();
    }
    bool var_true = (a == D4_ASSIGN_TRUE);
    bool lit_pos = (lit > 0);
    bool sat = (lit_pos && var_true) || (!lit_pos && !var_true);
    return sat ? 1 : -1;
}

__device__ __forceinline__ bool d4_assign_var_dense(
    int32_t lit,
    uint8_t* __restrict__ assign,
    uint32_t base_byte
) {
    uint32_t v = d4_var(lit);
    uint8_t want = (lit > 0) ? D4_ASSIGN_TRUE : D4_ASSIGN_FALSE;
    uint8_t cur = assign[base_byte + v];
    if (cur == D4_ASSIGN_UNASSIGNED) {
        assign[base_byte + v] = want;
        return true;
    }
    if (cur == want) {
        return true;
    }
    if (cur != D4_ASSIGN_TRUE && cur != D4_ASSIGN_FALSE) {
        d4_trap();
    }
    return false;
}

// Prepare one BFS level:
// - Run unit propagation for each work item.
// - Decide whether to expand (2 children) or keep (1) based on terminal status / depth cap.
// - Select a deterministic branch variable for expandable items.
//
// Output:
// - counts[tid] in {1,2} (or 0 for tid>=size).
// - pick_var[tid] = chosen var id for counts==2, else 0.
extern "C" __global__ void d4_frontier_prepare(
    uint32_t frontier_depth,
    uint32_t max_depth,
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ literals,
    const uint32_t* __restrict__ num_vars,    // len=1
    const uint32_t* __restrict__ num_clauses, // len=1
    const D4WorkItem* __restrict__ cur_items,
    const uint32_t* __restrict__ cur_size, // len=1
    uint32_t* __restrict__ true_bits,
    uint32_t* __restrict__ false_bits,
    uint32_t words_per_item,
    uint32_t* __restrict__ counts,
    uint32_t* __restrict__ pick_var
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = cur_size[0];
    if (tid >= n) {
        return;
    }

    D4WorkItem it = cur_items[tid];
    if (static_cast<uint32_t>(it.depth) > max_depth) {
        d4_trap();
    }

    // If we already hit the BFS expansion depth, keep this item as-is.
    if (static_cast<uint32_t>(it.depth) >= frontier_depth) {
        counts[tid] = 1u;
        pick_var[tid] = 0u;
        return;
    }

    const uint32_t nv = num_vars[0];
    const uint32_t nc = num_clauses[0];
    const uint32_t base = it.assignment_offset;

    bool conflict = false;
    bool satisfied = false;
    uint32_t branch_var = 0u;

    // Deterministic unit propagation (confluent) with deterministic branching heuristic:
    // pick a variable from the clause with the smallest number of unassigned literals,
    // tie-breaking by clause id, then by variable id.
    for (;;) {
        bool changed = false;
        bool sat_all = true;
        uint32_t best_un = 0xFFFFFFFFu;
        uint32_t best_clause = 0xFFFFFFFFu;
        uint32_t best_var = 0u;

        for (uint32_t c = 0; c < nc; c++) {
            uint32_t s = clause_offsets[c];
            uint32_t e = clause_offsets[c + 1u];

            bool clause_sat = false;
            uint32_t unassigned = 0u;
            int32_t unit_lit = 0;
            uint32_t min_un_var = 0u;

            for (uint32_t i = s; i < e; i++) {
                int32_t lit = literals[i];
                int8_t ev = d4_eval_lit_bitset(lit, true_bits, false_bits, base, words_per_item);
                if (ev == 1) {
                    clause_sat = true;
                    break;
                }
                if (ev == 0) {
                    unassigned++;
                    unit_lit = lit;
                    uint32_t v = d4_var(lit);
                    if (v == 0u || v > nv) {
                        d4_trap();
                    }
                    if (min_un_var == 0u || v < min_un_var) {
                        min_un_var = v;
                    }
                }
            }

            if (clause_sat) {
                continue;
            }
            sat_all = false;

            if (unassigned == 0u) {
                conflict = true;
                break;
            }

            if (unassigned == 1u) {
                if (!d4_assign_var_bitset(unit_lit, true_bits, false_bits, base, words_per_item)) {
                    conflict = true;
                    break;
                }
                changed = true;
            } else {
                // Only consider non-unit clauses for branching once we reach a propagation fixpoint.
                if (unassigned < best_un || (unassigned == best_un && c < best_clause)) {
                    best_un = unassigned;
                    best_clause = c;
                    best_var = min_un_var;
                }
            }
        }

        if (conflict) {
            satisfied = false;
            branch_var = 0u;
            break;
        }

        if (changed) {
            // Another pass to detect new units created by the assignments above.
            continue;
        }

        satisfied = sat_all;
        branch_var = best_var;
        break;
    }

    if (conflict || satisfied) {
        counts[tid] = 1u;
        pick_var[tid] = 0u;
        return;
    }

    // Defensive depth cap: we are about to create children at depth+1.
    if (static_cast<uint32_t>(it.depth) + 1u > max_depth) {
        d4_trap();
    }

    if (branch_var == 0u || branch_var > nv) {
        d4_trap();
    }

    // Ensure the chosen variable is currently unassigned.
    uint32_t wi = branch_var >> 5;
    if (wi >= words_per_item) {
        d4_trap();
    }
    uint32_t mask = 1u << (branch_var & 31u);
    if ((true_bits[base + wi] & mask) != 0u || (false_bits[base + wi] & mask) != 0u) {
        d4_trap();
    }

    counts[tid] = 2u;
    pick_var[tid] = branch_var;
}

extern "C" __global__ void d4_frontier_expand(
    const uint32_t* __restrict__ cur_size, // len=1
    const D4WorkItem* __restrict__ cur_items,
    const uint32_t* __restrict__ counts,         // per-item output counts (len >= cur_size)
    const uint32_t* __restrict__ prefix_offsets, // exclusive scan of counts (len >= cur_size)
    const uint32_t* __restrict__ pick_var,      // len >= cur_size
    const uint32_t* __restrict__ cur_true_bits,
    const uint32_t* __restrict__ cur_false_bits,
    uint32_t words_per_item,
    D4WorkItem* __restrict__ next_items,
    uint32_t* __restrict__ next_size, // len=1
    uint32_t* __restrict__ next_true_bits,
    uint32_t* __restrict__ next_false_bits,
    uint32_t max_frontier_items
) {
    // Compute next_size deterministically.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t n = cur_size[0];
        uint32_t out = (n == 0u) ? 0u : (prefix_offsets[n - 1u] + counts[n - 1u]);
        if (out > max_frontier_items) {
            d4_trap();
        }
        next_size[0] = out;
    }

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = cur_size[0];
    if (tid >= n) {
        return;
    }

    uint32_t base = prefix_offsets[tid];
    uint32_t out_count = counts[tid];
    if (out_count == 0u || out_count > 2u) {
        d4_trap();
    }
    uint32_t end = base + out_count;
    if (end > max_frontier_items) {
        d4_trap();
    }

    D4WorkItem it = cur_items[tid];
    uint32_t src_off = it.assignment_offset;

    if (out_count == 1u) {
        uint32_t out_idx = base;
        uint32_t dst_off = out_idx * words_per_item;

        for (uint32_t w = 0; w < words_per_item; w++) {
            next_true_bits[dst_off + w] = cur_true_bits[src_off + w];
            next_false_bits[dst_off + w] = cur_false_bits[src_off + w];
        }

        D4WorkItem o = it;
        o.subproblem_id = out_idx;
        o.assignment_offset = dst_off;
        next_items[out_idx] = o;
        return;
    }

    uint32_t v = pick_var[tid];
    if (v == 0u) {
        d4_trap();
    }
    uint32_t wi = v >> 5;
    if (wi >= words_per_item) {
        d4_trap();
    }
    uint32_t mask = 1u << (v & 31u);

    // Children are emitted in deterministic order: false then true.
    for (uint32_t j = 0; j < 2u; j++) {
        uint32_t out_idx = base + j;
        uint32_t dst_off = out_idx * words_per_item;

        for (uint32_t w = 0; w < words_per_item; w++) {
            next_true_bits[dst_off + w] = cur_true_bits[src_off + w];
            next_false_bits[dst_off + w] = cur_false_bits[src_off + w];
        }

        // The branch variable must be unassigned in the parent; enforce.
        if ((next_true_bits[dst_off + wi] & mask) != 0u || (next_false_bits[dst_off + wi] & mask) != 0u) {
            d4_trap();
        }

        if (j == 0u) {
            next_false_bits[dst_off + wi] |= mask;
        } else {
            next_true_bits[dst_off + wi] |= mask;
        }

        D4WorkItem o = it;
        o.subproblem_id = out_idx;
        o.assignment_offset = dst_off;
        o.depth = static_cast<uint16_t>(static_cast<uint32_t>(it.depth) + 1u);
        next_items[out_idx] = o;
    }
}

// Dense assignment variants (tri-state bytes). These are used for small/medium var counts where
// byte-per-var storage is acceptable and can be faster for random access.
extern "C" __global__ void d4_frontier_prepare_dense(
    uint32_t frontier_depth,
    uint32_t max_depth,
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ literals,
    const uint32_t* __restrict__ num_vars,    // len=1
    const uint32_t* __restrict__ num_clauses, // len=1
    const D4WorkItem* __restrict__ cur_items,
    const uint32_t* __restrict__ cur_size, // len=1
    uint8_t* __restrict__ assignments,      // dense pool
    uint32_t stride_bytes,                  // = var_cap + 1
    uint32_t* __restrict__ counts,
    uint32_t* __restrict__ pick_var
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = cur_size[0];
    if (tid >= n) {
        return;
    }

    D4WorkItem it = cur_items[tid];
    if (static_cast<uint32_t>(it.depth) > max_depth) {
        d4_trap();
    }

    if (static_cast<uint32_t>(it.depth) >= frontier_depth) {
        counts[tid] = 1u;
        pick_var[tid] = 0u;
        return;
    }

    const uint32_t nv = num_vars[0];
    const uint32_t nc = num_clauses[0];
    const uint32_t base = it.assignment_offset;

    bool conflict = false;
    bool satisfied = false;
    uint32_t branch_var = 0u;

    for (;;) {
        bool changed = false;
        bool sat_all = true;
        uint32_t best_un = 0xFFFFFFFFu;
        uint32_t best_clause = 0xFFFFFFFFu;
        uint32_t best_var = 0u;

        for (uint32_t c = 0; c < nc; c++) {
            uint32_t s = clause_offsets[c];
            uint32_t e = clause_offsets[c + 1u];

            bool clause_sat = false;
            uint32_t unassigned = 0u;
            int32_t unit_lit = 0;
            uint32_t min_un_var = 0u;

            for (uint32_t i = s; i < e; i++) {
                int32_t lit = literals[i];
                int8_t ev = d4_eval_lit_dense(lit, assignments, base);
                if (ev == 1) {
                    clause_sat = true;
                    break;
                }
                if (ev == 0) {
                    unassigned++;
                    unit_lit = lit;
                    uint32_t v = d4_var(lit);
                    if (v == 0u || v > nv) {
                        d4_trap();
                    }
                    if (min_un_var == 0u || v < min_un_var) {
                        min_un_var = v;
                    }
                }
            }

            if (clause_sat) {
                continue;
            }
            sat_all = false;

            if (unassigned == 0u) {
                conflict = true;
                break;
            }

            if (unassigned == 1u) {
                if (!d4_assign_var_dense(unit_lit, assignments, base)) {
                    conflict = true;
                    break;
                }
                changed = true;
            } else {
                if (unassigned < best_un || (unassigned == best_un && c < best_clause)) {
                    best_un = unassigned;
                    best_clause = c;
                    best_var = min_un_var;
                }
            }
        }

        if (conflict) {
            satisfied = false;
            branch_var = 0u;
            break;
        }
        if (changed) {
            continue;
        }
        satisfied = sat_all;
        branch_var = best_var;
        break;
    }

    if (conflict || satisfied) {
        counts[tid] = 1u;
        pick_var[tid] = 0u;
        return;
    }

    if (static_cast<uint32_t>(it.depth) + 1u > max_depth) {
        d4_trap();
    }
    if (branch_var == 0u || branch_var > nv) {
        d4_trap();
    }

    uint8_t cur = assignments[base + branch_var];
    if (cur != D4_ASSIGN_UNASSIGNED) {
        d4_trap();
    }

    counts[tid] = 2u;
    pick_var[tid] = branch_var;
    (void)stride_bytes; // stride is implicit in assignment_offset; retained for signature stability
}

extern "C" __global__ void d4_frontier_expand_dense(
    const uint32_t* __restrict__ cur_size, // len=1
    const D4WorkItem* __restrict__ cur_items,
    const uint32_t* __restrict__ counts,         // per-item output counts (len >= cur_size)
    const uint32_t* __restrict__ prefix_offsets, // exclusive scan of counts (len >= cur_size)
    const uint32_t* __restrict__ pick_var,      // len >= cur_size
    const uint8_t* __restrict__ cur_assign,
    uint32_t stride_bytes,
    D4WorkItem* __restrict__ next_items,
    uint32_t* __restrict__ next_size, // len=1
    uint8_t* __restrict__ next_assign,
    uint32_t max_frontier_items
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t n = cur_size[0];
        uint32_t out = (n == 0u) ? 0u : (prefix_offsets[n - 1u] + counts[n - 1u]);
        if (out > max_frontier_items) {
            d4_trap();
        }
        next_size[0] = out;
    }

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = cur_size[0];
    if (tid >= n) {
        return;
    }

    uint32_t base = prefix_offsets[tid];
    uint32_t out_count = counts[tid];
    if (out_count == 0u || out_count > 2u) {
        d4_trap();
    }
    uint32_t end = base + out_count;
    if (end > max_frontier_items) {
        d4_trap();
    }

    D4WorkItem it = cur_items[tid];
    uint32_t src_off = it.assignment_offset;

    if (out_count == 1u) {
        uint32_t out_idx = base;
        uint32_t dst_off = out_idx * stride_bytes;
        for (uint32_t b = 0; b < stride_bytes; b++) {
            next_assign[dst_off + b] = cur_assign[src_off + b];
        }
        D4WorkItem o = it;
        o.subproblem_id = out_idx;
        o.assignment_offset = dst_off;
        next_items[out_idx] = o;
        return;
    }

    uint32_t v = pick_var[tid];
    if (v == 0u) {
        d4_trap();
    }

    for (uint32_t j = 0; j < 2u; j++) {
        uint32_t out_idx = base + j;
        uint32_t dst_off = out_idx * stride_bytes;
        for (uint32_t b = 0; b < stride_bytes; b++) {
            next_assign[dst_off + b] = cur_assign[src_off + b];
        }
        uint8_t curv = next_assign[dst_off + v];
        if (curv != D4_ASSIGN_UNASSIGNED) {
            d4_trap();
        }
        next_assign[dst_off + v] = (j == 0u) ? D4_ASSIGN_FALSE : D4_ASSIGN_TRUE;

        D4WorkItem o = it;
        o.subproblem_id = out_idx;
        o.assignment_offset = dst_off;
        o.depth = static_cast<uint16_t>(static_cast<uint32_t>(it.depth) + 1u);
        next_items[out_idx] = o;
    }
}

// ---------------------------------------------------------------------------
// GPU-only assertion helpers (used by tests and invariant enforcement)
// ---------------------------------------------------------------------------

extern "C" __global__ void d4_assert_u32_eq(const uint32_t* __restrict__ value, uint32_t expected) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (value[0] != expected) {
        d4_trap();
    }
}

extern "C" __global__ void d4_assert_bitset_var(
    const uint32_t* __restrict__ true_bits,
    const uint32_t* __restrict__ false_bits,
    uint32_t words_per_item,
    uint32_t work_id,
    uint32_t var_id,
    uint32_t expected_state // 0=unassigned,1=true,2=false
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    uint32_t base = work_id * words_per_item;
    uint32_t wi = var_id >> 5;
    if (wi >= words_per_item) {
        d4_trap();
    }
    uint32_t mask = 1u << (var_id & 31u);
    bool t = (true_bits[base + wi] & mask) != 0u;
    bool f = (false_bits[base + wi] & mask) != 0u;
    if (t && f) {
        d4_trap();
    }
    uint32_t st = t ? 1u : (f ? 2u : 0u);
    if (st != expected_state) {
        d4_trap();
    }
}

extern "C" __global__ void d4_assert_dense_var(
    const uint8_t* __restrict__ assignments,
    uint32_t stride_bytes,
    uint32_t work_id,
    uint32_t var_id,
    uint32_t expected_state // 0=unassigned,1=true,2=false
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (expected_state > 2u) {
        d4_trap();
    }
    if (var_id >= stride_bytes) {
        d4_trap();
    }
    uint32_t base = work_id * stride_bytes;
    uint8_t cur = assignments[base + var_id];
    if (cur != static_cast<uint8_t>(expected_state)) {
        d4_trap();
    }
}
