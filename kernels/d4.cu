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

// XGCF node tags (must match kernels/circuit.cu and crates/xlog-prob/src/xgcf.rs).
static constexpr uint8_t XGCF_CONST0 = 0;
static constexpr uint8_t XGCF_CONST1 = 1;
static constexpr uint8_t XGCF_LIT = 2;
static constexpr uint8_t XGCF_AND = 3;
static constexpr uint8_t XGCF_OR = 4;
static constexpr uint8_t XGCF_DECISION = 5;

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
// Per-frontier DFS compilation (Phase 1): count + emit
// ---------------------------------------------------------------------------

// NOTE: Device recursion is not enabled in this build (no -rdc). Use an explicit stack.
static constexpr uint32_t D4_MAX_DEPTH_LIMIT = 512u;

__device__ __forceinline__ void d4_unassign_var_bitset(
    uint32_t var,
    uint32_t* __restrict__ true_bits,
    uint32_t* __restrict__ false_bits,
    uint32_t base_word,
    uint32_t words_per_item
) {
    uint32_t wi = var >> 5;
    if (wi >= words_per_item) {
        d4_trap();
    }
    uint32_t mask = 1u << (var & 31u);
    true_bits[base_word + wi] &= ~mask;
    false_bits[base_word + wi] &= ~mask;
}

__device__ __forceinline__ void d4_undo_trail_to(
    int32_t* __restrict__ trail,
    uint32_t* __restrict__ trail_len,
    uint32_t target_len,
    uint32_t* __restrict__ true_bits,
    uint32_t* __restrict__ false_bits,
    uint32_t base_word,
    uint32_t words_per_item
) {
    while (*trail_len > target_len) {
        uint32_t idx = (*trail_len) - 1u;
        int32_t lit = trail[idx];
        uint32_t var = d4_var(lit);
        d4_unassign_var_bitset(var, true_bits, false_bits, base_word, words_per_item);
        *trail_len = idx;
    }
}

__device__ __forceinline__ uint32_t d4_uf_find(uint32_t* __restrict__ parent, uint32_t v);
__device__ __forceinline__ void d4_uf_union(
    uint32_t* __restrict__ parent,
    uint32_t* __restrict__ size,
    uint32_t a,
    uint32_t b
);

__device__ __forceinline__ void d4_unit_propagate_pick_bitset(
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ literals,
    uint32_t nv,
    uint32_t nc,
    uint32_t* __restrict__ true_bits,
    uint32_t* __restrict__ false_bits,
    uint32_t base_word,
    uint32_t words_per_item,
    int32_t* __restrict__ trail,
    uint32_t* __restrict__ trail_len,
    uint32_t trail_cap,
    uint32_t component_root,
    uint32_t* __restrict__ uf_parent,
    bool* __restrict__ out_conflict,
    bool* __restrict__ out_satisfied,
    uint32_t* __restrict__ out_branch_var
) {
    bool conflict = false;
    bool satisfied = false;
    uint32_t branch_var = 0u;

    // Deterministic unit propagation + deterministic branch heuristic (same as frontier):
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
                int8_t ev = d4_eval_lit_bitset(lit, true_bits, false_bits, base_word, words_per_item);
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

            if (component_root != 0u && unassigned > 0u) {
                if (uf_parent == nullptr) {
                    d4_trap();
                }
                uint32_t root = d4_uf_find(uf_parent, min_un_var);
                if (root != component_root) {
                    continue;
                }
            }

            sat_all = false;

            if (unassigned == 0u) {
                conflict = true;
                break;
            }

            if (unassigned == 1u) {
                if (!d4_assign_var_bitset(unit_lit, true_bits, false_bits, base_word, words_per_item)) {
                    conflict = true;
                    break;
                }
                uint32_t tl = *trail_len;
                if (tl >= trail_cap) {
                    d4_trap();
                }
                trail[tl] = unit_lit;
                *trail_len = tl + 1u;
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

    *out_conflict = conflict;
    *out_satisfied = satisfied;
    *out_branch_var = branch_var;
}

struct D4Frame {
    uint32_t trail_start;
    uint32_t trail_keep;
    uint32_t branch_var;
    uint32_t child_false;
    uint32_t child_true;
    uint16_t depth;
    uint8_t stage; // 0=enter,1=after_false,2=after_true
};

// ---------------------------------------------------------------------------
// Component decomposition helpers (union-find over unassigned variables)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t d4_uf_find(uint32_t* __restrict__ parent, uint32_t v) {
    uint32_t root = v;
    while (parent[root] != root) {
        root = parent[root];
    }
    while (parent[v] != v) {
        uint32_t p = parent[v];
        parent[v] = root;
        v = p;
    }
    return root;
}

__device__ __forceinline__ void d4_uf_union(
    uint32_t* __restrict__ parent,
    uint32_t* __restrict__ size,
    uint32_t a,
    uint32_t b
) {
    uint32_t ra = d4_uf_find(parent, a);
    uint32_t rb = d4_uf_find(parent, b);
    if (ra == rb) {
        return;
    }
    uint32_t sa = size[ra];
    uint32_t sb = size[rb];
    if (sa < sb) {
        parent[ra] = rb;
        size[rb] = sa + sb;
    } else {
        parent[rb] = ra;
        size[ra] = sa + sb;
    }
}

struct D4CountWriter {
    uint32_t next_id; // simulated node ids (0/1 reserved for consts)
    uint32_t nodes;
    uint32_t edges;

    __device__ __forceinline__ uint32_t alloc_node() {
        nodes++;
        return next_id++;
    }
};

struct D4EmitWriter {
    uint32_t next_node;
    uint32_t next_edge;
    uint32_t node_end;
    uint32_t edge_end;

    uint8_t* __restrict__ node_type;
    uint32_t* __restrict__ child_offsets;
    uint32_t* __restrict__ child_indices;
    int32_t* __restrict__ lit;
    uint32_t* __restrict__ decision_var;
    uint32_t* __restrict__ decision_child_false;
    uint32_t* __restrict__ decision_child_true;
    uint32_t* __restrict__ node_level;

    __device__ __forceinline__ uint32_t alloc_node() {
        if (next_node >= node_end) {
            d4_trap();
        }
        return next_node++;
    }

    __device__ __forceinline__ void reserve_edges(uint32_t deg) {
        if (deg == 0u) {
            return;
        }
        if (next_edge > edge_end || deg > (edge_end - next_edge)) {
            d4_trap();
        }
    }
};

__device__ __forceinline__ uint32_t d4_level_of_node(
    const uint32_t* __restrict__ node_level,
    uint32_t node_id
) {
    // Const nodes are fixed at level 0.
    if (node_id <= 1u) {
        return 0u;
    }
    return node_level[node_id];
}

template <typename Writer>
__device__ __forceinline__ uint32_t d4_new_lit(Writer* w, int32_t lit) {
    (void)lit;
    return w->alloc_node();
}

template <>
__device__ __forceinline__ uint32_t d4_new_lit<D4EmitWriter>(D4EmitWriter* w, int32_t litval) {
    uint32_t id = w->alloc_node();
    w->node_type[id] = XGCF_LIT;
    w->lit[id] = litval;
    w->node_level[id] = 0u;
    w->child_offsets[id] = w->next_edge;
    return id;
}

template <typename Writer>
__device__ __forceinline__ uint32_t d4_new_decision(
    Writer* w,
    uint32_t var,
    uint32_t child_false,
    uint32_t child_true
) {
    (void)var;
    (void)child_false;
    (void)child_true;
    return w->alloc_node();
}

template <>
__device__ __forceinline__ uint32_t d4_new_decision<D4EmitWriter>(
    D4EmitWriter* w,
    uint32_t var,
    uint32_t child_false,
    uint32_t child_true
) {
    uint32_t id = w->alloc_node();
    w->node_type[id] = XGCF_DECISION;
    w->decision_var[id] = var;
    w->decision_child_false[id] = child_false;
    w->decision_child_true[id] = child_true;
    uint32_t lf = d4_level_of_node(w->node_level, child_false);
    uint32_t lt = d4_level_of_node(w->node_level, child_true);
    uint32_t lvl = (lf > lt ? lf : lt) + 1u;
    w->node_level[id] = lvl;
    w->child_offsets[id] = w->next_edge;
    return id;
}

__device__ __forceinline__ uint32_t d4_new_and(D4CountWriter* w, uint32_t num_children) {
    w->edges += num_children;
    return w->alloc_node();
}

template <typename Writer>
__device__ __forceinline__ uint32_t d4_new_and_children(
    Writer* w,
    const uint32_t* __restrict__ children,
    uint32_t num_children
) {
    (void)children;
    return d4_new_and(w, num_children);
}

template <>
__device__ __forceinline__ uint32_t d4_new_and_children<D4EmitWriter>(
    D4EmitWriter* w,
    const uint32_t* __restrict__ children,
    uint32_t num_children
) {
    if (num_children == 0u) {
        d4_trap();
    }
    uint32_t id = w->alloc_node();
    w->node_type[id] = XGCF_AND;
    w->lit[id] = 0;
    w->decision_var[id] = 0u;
    w->decision_child_false[id] = 0u;
    w->decision_child_true[id] = 0u;
    w->child_offsets[id] = w->next_edge;
    w->reserve_edges(num_children);

    uint32_t max_lvl = 0u;
    for (uint32_t i = 0; i < num_children; i++) {
        uint32_t child = children[i];
        w->child_indices[w->next_edge + i] = child;
        uint32_t lvl = d4_level_of_node(w->node_level, child);
        if (lvl > max_lvl) {
            max_lvl = lvl;
        }
    }
    w->next_edge += num_children;
    w->node_level[id] = max_lvl + 1u;
    return id;
}

// AND node with children = contiguous LIT id range [+ optional main child].
// This is the only AND pattern used in the Phase 1 D4 kernel (unit propagation + decision-DPLL).
__device__ __forceinline__ uint32_t d4_new_and_lit_range(
    D4EmitWriter* w,
    uint32_t first_lit_id,
    uint32_t num_lits,
    uint32_t main,
    bool include_main
) {
    uint32_t total = num_lits + (include_main ? 1u : 0u);
    if (total == 0u) {
        d4_trap();
    }
    uint32_t id = w->alloc_node();
    w->node_type[id] = XGCF_AND;
    w->lit[id] = 0;
    w->decision_var[id] = 0u;
    w->decision_child_false[id] = 0u;
    w->decision_child_true[id] = 0u;
    w->child_offsets[id] = w->next_edge;
    w->reserve_edges(total);

    for (uint32_t i = 0; i < num_lits; i++) {
        w->child_indices[w->next_edge + i] = first_lit_id + i;
    }
    if (include_main) {
        w->child_indices[w->next_edge + num_lits] = main;
    }
    w->next_edge += total;

    uint32_t max_lvl = include_main ? d4_level_of_node(w->node_level, main) : 0u;
    w->node_level[id] = max_lvl + 1u;
    return id;
}

// Conjoin implied literals (from trail[implied_start..implied_end)) with `main`.
__device__ __forceinline__ uint32_t d4_conjoin_implied(
    D4CountWriter* w,
    const int32_t* __restrict__ trail,
    uint32_t implied_start,
    uint32_t implied_end,
    uint32_t main
) {
    uint32_t implied_len = implied_end - implied_start;
    if (main == 0u) {
        return 0u;
    }
    if (implied_len == 0u) {
        return main;
    }

    uint32_t first_lit_id = 0u;
    for (uint32_t i = 0; i < implied_len; i++) {
        uint32_t id = d4_new_lit(w, trail[implied_start + i]);
        if (i == 0u) {
            first_lit_id = id;
        }
    }
    (void)first_lit_id;

    if (main == 1u) {
        if (implied_len == 1u) {
            return first_lit_id;
        }
        return d4_new_and(w, implied_len);
    }
    return d4_new_and(w, implied_len + 1u);
}

__device__ __forceinline__ uint32_t d4_conjoin_implied(
    D4EmitWriter* w,
    const int32_t* __restrict__ trail,
    uint32_t implied_start,
    uint32_t implied_end,
    uint32_t main
) {
    uint32_t implied_len = implied_end - implied_start;
    if (main == 0u) {
        return 0u;
    }
    if (implied_len == 0u) {
        return main;
    }

    uint32_t first_lit_id = 0u;
    for (uint32_t i = 0; i < implied_len; i++) {
        uint32_t id = d4_new_lit(w, trail[implied_start + i]);
        if (i == 0u) {
            first_lit_id = id;
        }
    }

    if (main == 1u) {
        if (implied_len == 1u) {
            return first_lit_id;
        }
        return d4_new_and_lit_range(w, first_lit_id, implied_len, 0u, false);
    }
    return d4_new_and_lit_range(w, first_lit_id, implied_len, main, true);
}

__device__ __forceinline__ uint32_t d4_conjoin_base_lits(
    D4CountWriter* w,
    uint32_t first_lit_id,
    uint32_t base_lit_count,
    uint32_t residual
) {
    (void)first_lit_id;
    if (residual == 1u) {
        if (base_lit_count == 1u) {
            return first_lit_id;
        }
        return d4_new_and(w, base_lit_count);
    }
    return d4_new_and(w, base_lit_count + 1u);
}

__device__ __forceinline__ uint32_t d4_conjoin_base_lits(
    D4EmitWriter* w,
    uint32_t first_lit_id,
    uint32_t base_lit_count,
    uint32_t residual
) {
    if (residual == 1u) {
        if (base_lit_count == 1u) {
            return first_lit_id;
        }
        return d4_new_and_lit_range(w, first_lit_id, base_lit_count, 0u, false);
    }
    return d4_new_and_lit_range(w, first_lit_id, base_lit_count, residual, true);
}

__device__ __forceinline__ uint32_t d4_build_components_bitset(
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ literals,
    uint32_t nv,
    uint32_t nc,
    uint32_t* __restrict__ true_bits,
    uint32_t* __restrict__ false_bits,
    uint32_t base_word,
    uint32_t words_per_item,
    uint32_t* __restrict__ uf_parent,
    uint32_t* __restrict__ uf_aux,
    uint32_t* __restrict__ comp_list,
    bool* __restrict__ out_conflict
) {
    // Initialize union-find.
    uf_parent[0] = 0u;
    uf_aux[0] = 0u;
    for (uint32_t v = 1u; v <= nv; v++) {
        uf_parent[v] = v;
        uf_aux[v] = 1u;
    }

    bool conflict = false;

    // Union unassigned variables that appear together in an unsatisfied clause.
    for (uint32_t c = 0; c < nc; c++) {
        uint32_t s = clause_offsets[c];
        uint32_t e = clause_offsets[c + 1u];

        bool clause_sat = false;
        uint32_t unassigned = 0u;
        uint32_t first_var = 0u;

        for (uint32_t i = s; i < e; i++) {
            int32_t lit = literals[i];
            int8_t ev = d4_eval_lit_bitset(lit, true_bits, false_bits, base_word, words_per_item);
            if (ev == 1) {
                clause_sat = true;
                break;
            }
            if (ev == 0) {
                uint32_t v = d4_var(lit);
                if (v == 0u || v > nv) {
                    d4_trap();
                }
                unassigned++;
                if (first_var == 0u) {
                    first_var = v;
                } else {
                    d4_uf_union(uf_parent, uf_aux, first_var, v);
                }
            }
        }

        if (clause_sat) {
            continue;
        }
        if (unassigned == 0u) {
            conflict = true;
            break;
        }
    }

    if (conflict) {
        *out_conflict = true;
        return 0u;
    }

    // Reuse uf_aux as a "seen" marker (0/1).
    for (uint32_t v = 0u; v <= nv; v++) {
        uf_aux[v] = 0u;
    }

    uint32_t comp_count = 0u;
    for (uint32_t c = 0; c < nc; c++) {
        uint32_t s = clause_offsets[c];
        uint32_t e = clause_offsets[c + 1u];

        bool clause_sat = false;
        uint32_t unassigned = 0u;
        uint32_t first_var = 0u;

        for (uint32_t i = s; i < e; i++) {
            int32_t lit = literals[i];
            int8_t ev = d4_eval_lit_bitset(lit, true_bits, false_bits, base_word, words_per_item);
            if (ev == 1) {
                clause_sat = true;
                break;
            }
            if (ev == 0) {
                uint32_t v = d4_var(lit);
                if (v == 0u || v > nv) {
                    d4_trap();
                }
                unassigned++;
                if (first_var == 0u) {
                    first_var = v;
                }
            }
        }

        if (clause_sat) {
            continue;
        }
        if (unassigned == 0u) {
            conflict = true;
            break;
        }

        uint32_t root = d4_uf_find(uf_parent, first_var);
        if (uf_aux[root] == 0u) {
            uf_aux[root] = 1u;
            comp_list[comp_count++] = root;
        }
    }

    if (conflict) {
        *out_conflict = true;
        return 0u;
    }

    *out_conflict = false;
    return comp_count;
}

template <typename Writer>
__device__ __forceinline__ uint32_t d4_compile_residual_dpll(
    Writer* w,
    uint32_t start_depth,
    uint32_t max_depth,
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ literals,
    uint32_t nv,
    uint32_t nc,
    uint32_t* __restrict__ true_bits,
    uint32_t* __restrict__ false_bits,
    uint32_t base_word,
    uint32_t words_per_item,
    int32_t* __restrict__ trail,
    uint32_t trail_cap,
    uint32_t trail_base_len,
    uint32_t component_root,
    uint32_t* __restrict__ uf_parent
) {
    if (max_depth > D4_MAX_DEPTH_LIMIT) {
        d4_trap();
    }
    if (trail_base_len > trail_cap) {
        d4_trap();
    }

    uint32_t trail_len = trail_base_len;
    uint32_t sp = 0u;
    D4Frame stack[D4_MAX_DEPTH_LIMIT + 1u];
    stack[0].stage = 0u;
    stack[0].depth = static_cast<uint16_t>(start_depth);

    uint32_t ret = 1u; // Const1 default (unused)

    for (;;) {
        D4Frame& f = stack[sp];

        if (f.stage == 0u) {
            f.trail_start = trail_len;

            bool conflict = false;
            bool sat = false;
            uint32_t pick = 0u;
            d4_unit_propagate_pick_bitset(
                clause_offsets,
                literals,
                nv,
                nc,
                true_bits,
                false_bits,
                base_word,
                words_per_item,
                trail,
                &trail_len,
                trail_cap,
                component_root,
                uf_parent,
                &conflict,
                &sat,
                &pick
            );

            if (conflict) {
                d4_undo_trail_to(trail, &trail_len, f.trail_start, true_bits, false_bits, base_word, words_per_item);
                ret = 0u;
                if (sp == 0u) {
                    return ret;
                }
                sp--;
                continue;
            }

            if (sat) {
                uint32_t implied_end = trail_len;
                ret = d4_conjoin_implied(w, trail, f.trail_start, implied_end, 1u);
                d4_undo_trail_to(trail, &trail_len, f.trail_start, true_bits, false_bits, base_word, words_per_item);
                if (sp == 0u) {
                    return ret;
                }
                sp--;
                continue;
            }

            if (static_cast<uint32_t>(f.depth) >= max_depth) {
                d4_trap();
            }
            if (pick == 0u || pick > nv) {
                d4_trap();
            }

            f.trail_keep = trail_len;
            f.branch_var = pick;
            f.stage = 1u;

            // Assign branch var = false (decision assignment; not emitted as LIT).
            int32_t branch_lit = -static_cast<int32_t>(pick);
            if (!d4_assign_var_bitset(branch_lit, true_bits, false_bits, base_word, words_per_item)) {
                d4_trap();
            }
            if (trail_len >= trail_cap) {
                d4_trap();
            }
            trail[trail_len++] = branch_lit;

            sp++;
            stack[sp].stage = 0u;
            stack[sp].depth = static_cast<uint16_t>(static_cast<uint32_t>(f.depth) + 1u);
            continue;
        }

        if (f.stage == 1u) {
            f.child_false = ret;
            d4_undo_trail_to(trail, &trail_len, f.trail_keep, true_bits, false_bits, base_word, words_per_item);

            // Assign branch var = true.
            int32_t branch_lit = static_cast<int32_t>(f.branch_var);
            if (!d4_assign_var_bitset(branch_lit, true_bits, false_bits, base_word, words_per_item)) {
                d4_trap();
            }
            if (trail_len >= trail_cap) {
                d4_trap();
            }
            trail[trail_len++] = branch_lit;

            f.stage = 2u;

            sp++;
            stack[sp].stage = 0u;
            stack[sp].depth = static_cast<uint16_t>(static_cast<uint32_t>(f.depth) + 1u);
            continue;
        }

        // f.stage == 2
        f.child_true = ret;
        d4_undo_trail_to(trail, &trail_len, f.trail_keep, true_bits, false_bits, base_word, words_per_item);

        uint32_t main;
        if (f.child_false == f.child_true) {
            main = f.child_true;
        } else {
            main = d4_new_decision(w, f.branch_var, f.child_false, f.child_true);
        }

        ret = d4_conjoin_implied(w, trail, f.trail_start, f.trail_keep, main);

        d4_undo_trail_to(trail, &trail_len, f.trail_start, true_bits, false_bits, base_word, words_per_item);

        if (sp == 0u) {
            return ret;
        }
        sp--;
    }
}

template <typename Writer>
__device__ __forceinline__ uint32_t d4_compile_leaf(
    Writer* w,
    uint32_t start_depth,
    uint32_t max_depth,
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ literals,
    uint32_t nv,
    uint32_t nc,
    uint32_t* __restrict__ true_bits,
    uint32_t* __restrict__ false_bits,
    uint32_t base_word,
    uint32_t words_per_item,
    int32_t* __restrict__ trail,
    uint32_t trail_cap,
    uint32_t* __restrict__ uf_parent,
    uint32_t* __restrict__ uf_aux,
    uint32_t* __restrict__ comp_list
) {
    // Base unit propagation before component decomposition.
    uint32_t trail_len = 0u;
    bool conflict = false;
    bool sat = false;
    uint32_t pick = 0u;
    d4_unit_propagate_pick_bitset(
        clause_offsets,
        literals,
        nv,
        nc,
        true_bits,
        false_bits,
        base_word,
        words_per_item,
        trail,
        &trail_len,
        trail_cap,
        0u,
        nullptr,
        &conflict,
        &sat,
        &pick
    );

    if (conflict) {
        d4_undo_trail_to(trail, &trail_len, 0u, true_bits, false_bits, base_word, words_per_item);
        return 0u;
    }

    uint32_t residual = 1u;
    if (!sat) {
        bool comp_conflict = false;
        uint32_t comp_count = d4_build_components_bitset(
            clause_offsets,
            literals,
            nv,
            nc,
            true_bits,
            false_bits,
            base_word,
            words_per_item,
            uf_parent,
            uf_aux,
            comp_list,
            &comp_conflict
        );

        if (comp_conflict) {
            d4_undo_trail_to(trail, &trail_len, 0u, true_bits, false_bits, base_word, words_per_item);
            return 0u;
        }

        if (comp_count == 0u) {
            residual = 1u;
        } else if (comp_count == 1u) {
            uint32_t root = comp_list[0];
            residual = d4_compile_residual_dpll(
                w,
                start_depth,
                max_depth,
                clause_offsets,
                literals,
                nv,
                nc,
                true_bits,
                false_bits,
                base_word,
                words_per_item,
                trail,
                trail_cap,
                trail_len,
                root,
                uf_parent
            );
        } else {
            for (uint32_t i = 0; i < comp_count; i++) {
                uint32_t root = comp_list[i];
                uint32_t child = d4_compile_residual_dpll(
                    w,
                    start_depth,
                    max_depth,
                    clause_offsets,
                    literals,
                    nv,
                    nc,
                    true_bits,
                    false_bits,
                    base_word,
                    words_per_item,
                    trail,
                    trail_cap,
                    trail_len,
                    root,
                    uf_parent
                );
                comp_list[i] = child;
            }
            residual = d4_new_and_children(w, comp_list, comp_count);
        }
    }

    residual = d4_conjoin_implied(w, trail, 0u, trail_len, residual);
    d4_undo_trail_to(trail, &trail_len, 0u, true_bits, false_bits, base_word, words_per_item);

    if (residual == 0u) {
        return 0u;
    }

    // Emit base assignment literals as LIT nodes in deterministic var order.
    // These are required to make BFS frontier leaves mutually exclusive in the top-level OR.
    uint32_t base_lit_count = 0u;
    for (uint32_t widx = 0; widx < words_per_item; widx++) {
        uint32_t t = true_bits[base_word + widx];
        uint32_t fbits = false_bits[base_word + widx];
        if ((t & fbits) != 0u) {
            d4_trap();
        }
        uint32_t bits = t | fbits;
        if (widx == 0u) {
            bits &= ~1u; // var 0 is unused
        }
        base_lit_count += __popc(bits);
    }

    if (base_lit_count == 0u) {
        return residual;
    }

    uint32_t first_lit_id = 0u;
    uint32_t emitted = 0u;
    for (uint32_t widx = 0; widx < words_per_item; widx++) {
        uint32_t t = true_bits[base_word + widx];
        uint32_t fbits = false_bits[base_word + widx];
        uint32_t bits = t | fbits;
        if (widx == 0u) {
            bits &= ~1u;
        }
        while (bits != 0u) {
            uint32_t b = __ffs(bits) - 1u;
            uint32_t var = (widx << 5) + b;
            if (var == 0u || var > nv) {
                d4_trap();
            }
            uint32_t mask = 1u << b;
            int32_t litv = (t & mask) ? static_cast<int32_t>(var) : -static_cast<int32_t>(var);
            uint32_t id = d4_new_lit(w, litv);
            if (emitted == 0u) {
                first_lit_id = id;
            }
            emitted++;
            bits &= (bits - 1u);
        }
    }
    if (emitted != base_lit_count) {
        d4_trap();
    }

    if (residual == 1u) {
        return d4_conjoin_base_lits(w, first_lit_id, base_lit_count, residual);
    }

    // AND of base lits + residual.
    return d4_conjoin_base_lits(w, first_lit_id, base_lit_count, residual);
}

extern "C" __global__ void d4_compile_count(
    uint32_t max_depth,
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ literals,
    const uint32_t* __restrict__ num_vars,    // len=1
    const uint32_t* __restrict__ num_clauses, // len=1
    const D4WorkItem* __restrict__ frontier_items,
    const uint32_t* __restrict__ frontier_size, // len=1
    uint32_t* __restrict__ true_bits,
    uint32_t* __restrict__ false_bits,
    uint32_t words_per_item,
    int32_t* __restrict__ scratch_trail,
    uint32_t trail_stride,
    uint32_t* __restrict__ scratch_uf_parent,
    uint32_t* __restrict__ scratch_uf_aux,
    uint32_t* __restrict__ scratch_comp,
    uint32_t uf_stride,
    uint32_t* __restrict__ out_node_counts,
    uint32_t* __restrict__ out_edge_counts
) {
    if (threadIdx.x != 0u) {
        return;
    }
    uint32_t id = blockIdx.x;
    uint32_t n = frontier_size[0];
    if (id >= n) {
        out_node_counts[id] = 0u;
        out_edge_counts[id] = 0u;
        return;
    }

    uint32_t nv = num_vars[0];
    uint32_t nc = num_clauses[0];

    D4WorkItem it = frontier_items[id];
    uint32_t base_word = it.assignment_offset;

    D4CountWriter w;
    w.next_id = 2u;
    w.nodes = 0u;
    w.edges = 0u;

    if (uf_stride < nv + 1u) {
        d4_trap();
    }

    int32_t* trail = scratch_trail + static_cast<uint64_t>(id) * static_cast<uint64_t>(trail_stride);
    uint32_t* uf_parent = scratch_uf_parent + static_cast<uint64_t>(id) * static_cast<uint64_t>(uf_stride);
    uint32_t* uf_aux = scratch_uf_aux + static_cast<uint64_t>(id) * static_cast<uint64_t>(uf_stride);
    uint32_t* comp_list = scratch_comp + static_cast<uint64_t>(id) * static_cast<uint64_t>(uf_stride);
    (void)d4_compile_leaf(
        &w,
        static_cast<uint32_t>(it.depth),
        max_depth,
        clause_offsets,
        literals,
        nv,
        nc,
        true_bits,
        false_bits,
        base_word,
        words_per_item,
        trail,
        trail_stride,
        uf_parent,
        uf_aux,
        comp_list
    );

    out_node_counts[id] = w.nodes;
    out_edge_counts[id] = w.edges;
}

extern "C" __global__ void d4_compile_emit(
    uint32_t max_depth,
    uint32_t max_frontier_items,
    uint32_t node_cap,
    uint32_t edge_cap,
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ literals,
    const uint32_t* __restrict__ num_vars,    // len=1
    const uint32_t* __restrict__ num_clauses, // len=1
    const D4WorkItem* __restrict__ frontier_items,
    const uint32_t* __restrict__ frontier_size, // len=1
    uint32_t* __restrict__ true_bits,
    uint32_t* __restrict__ false_bits,
    uint32_t words_per_item,
    int32_t* __restrict__ scratch_trail,
    uint32_t trail_stride,
    uint32_t* __restrict__ scratch_uf_parent,
    uint32_t* __restrict__ scratch_uf_aux,
    uint32_t* __restrict__ scratch_comp,
    uint32_t uf_stride,
    const uint32_t* __restrict__ node_counts,
    const uint32_t* __restrict__ edge_counts,
    const uint32_t* __restrict__ node_offsets,
    const uint32_t* __restrict__ edge_offsets,
    uint8_t* __restrict__ node_type,
    uint32_t* __restrict__ child_offsets,
    uint32_t* __restrict__ child_indices,
    int32_t* __restrict__ lit,
    uint32_t* __restrict__ decision_var,
    uint32_t* __restrict__ decision_child_false,
    uint32_t* __restrict__ decision_child_true,
    uint32_t* __restrict__ node_level
) {
    uint32_t n = frontier_size[0];
    if (n > max_frontier_items) {
        d4_trap();
    }

    // Initialize reserved nodes once.
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        node_type[0] = XGCF_CONST0;
        node_type[1] = XGCF_CONST1;
        node_type[2] = XGCF_OR;
        lit[0] = 0;
        lit[1] = 0;
        lit[2] = 0;
        node_level[0] = 0u;
        node_level[1] = 0u;
        node_level[2] = (max_depth * 2u) + 7u;
        child_offsets[0] = 0u;
        child_offsets[1] = 0u;
        child_offsets[2] = 0u;

        // Capacity checks + CSR sentinel.
        uint32_t last = max_frontier_items - 1u;
        uint32_t total_nodes = node_offsets[last] + node_counts[last];
        uint32_t total_edges = edge_offsets[last] + edge_counts[last];

        uint32_t reserved_nodes = 3u;
        uint32_t reserved_edges = n;

        if (reserved_nodes + total_nodes > node_cap) {
            d4_trap();
        }
        if (reserved_edges + total_edges > edge_cap) {
            d4_trap();
        }

        uint32_t sentinel_idx = reserved_nodes + total_nodes;
        child_offsets[sentinel_idx] = reserved_edges + total_edges;
        // Ensure the root OR degree is well-defined even when there are no leaf nodes.
        child_offsets[reserved_nodes] = reserved_edges;
    }

    if (threadIdx.x != 0u) {
        return;
    }

    uint32_t id = blockIdx.x;
    if (id >= n) {
        return;
    }

    uint32_t nv = num_vars[0];
    uint32_t nc = num_clauses[0];

    D4WorkItem it = frontier_items[id];
    uint32_t base_word = it.assignment_offset;

    uint32_t reserved_nodes = 3u;
    uint32_t reserved_edges = n;

    uint32_t node_base = reserved_nodes + node_offsets[id];
    uint32_t edge_base = reserved_edges + edge_offsets[id];
    uint32_t node_end = node_base + node_counts[id];
    uint32_t edge_end = edge_base + edge_counts[id];

    D4EmitWriter w;
    w.next_node = node_base;
    w.next_edge = edge_base;
    w.node_end = node_end;
    w.edge_end = edge_end;
    w.node_type = node_type;
    w.child_offsets = child_offsets;
    w.child_indices = child_indices;
    w.lit = lit;
    w.decision_var = decision_var;
    w.decision_child_false = decision_child_false;
    w.decision_child_true = decision_child_true;
    w.node_level = node_level;

    if (uf_stride < nv + 1u) {
        d4_trap();
    }

    int32_t* trail = scratch_trail + static_cast<uint64_t>(id) * static_cast<uint64_t>(trail_stride);
    uint32_t* uf_parent = scratch_uf_parent + static_cast<uint64_t>(id) * static_cast<uint64_t>(uf_stride);
    uint32_t* uf_aux = scratch_uf_aux + static_cast<uint64_t>(id) * static_cast<uint64_t>(uf_stride);
    uint32_t* comp_list = scratch_comp + static_cast<uint64_t>(id) * static_cast<uint64_t>(uf_stride);

    uint32_t leaf_root = d4_compile_leaf(
        &w,
        static_cast<uint32_t>(it.depth),
        max_depth,
        clause_offsets,
        literals,
        nv,
        nc,
        true_bits,
        false_bits,
        base_word,
        words_per_item,
        trail,
        trail_stride,
        uf_parent,
        uf_aux,
        comp_list
    );

    // Root OR children (fixed degree = max_frontier_items). Default is Const0; overwrite active items.
    child_indices[id] = leaf_root;

    // Validate emission consumed exactly the counted budget.
    if (w.next_node != node_end || w.next_edge != edge_end) {
        d4_trap();
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

extern "C" __global__ void d4_assert_leaf_root_and_degree(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ child_indices,
    uint32_t node_cap,
    uint32_t leaf_index,
    uint32_t expected_degree
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    // Root OR is node 2 with fixed children stored at child_indices[0..].
    uint32_t child = child_indices[leaf_index];
    if (child + 1u >= node_cap) {
        d4_trap();
    }
    if (node_type[child] != XGCF_AND) {
        d4_trap();
    }
    uint32_t deg = child_offsets[child + 1u] - child_offsets[child];
    if (deg != expected_degree) {
        d4_trap();
    }
}
