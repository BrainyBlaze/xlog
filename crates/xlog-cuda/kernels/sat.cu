// kernels/sat.cu
//
// GPU-native CDCL SAT solver + GPU-native result validation.
//
// Design constraints:
// - Complete (SAT/UNSAT; no "unknown")
// - Deterministic (tie-breaking and schedules are stable)
// - Memory-bounded (fixed-capacity arenas; overflow => hard error)
// - GPU-native data-plane (host is control-plane only)

// NOTE: This file is compiled to PTX and embedded in the repo.
// To keep regeneration possible in environments that only have NVRTC (no full host C++ headers),
// we avoid libstdc++ includes and define the fixed-width integer aliases we need.
#ifndef __cplusplus
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned int uint32_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef long long int64_t;
#else
#include <cstdint>
#endif

// ---------------------------------------------------------------------------
// Encoding / constants
// ---------------------------------------------------------------------------

// Assignment encoding per variable: -1=false, 0=unassigned, +1=true.
static constexpr int8_t SAT_VAL_FALSE = -1;
static constexpr int8_t SAT_VAL_UNASSIGNED = 0;
static constexpr int8_t SAT_VAL_TRUE = 1;

// Solver status codes.
static constexpr int32_t SAT_STATUS_UNSAT = 0;
static constexpr int32_t SAT_STATUS_SAT = 1;
static constexpr int32_t SAT_STATUS_ERROR = -1;

// Reason encoding.
static constexpr int32_t SAT_REASON_NONE = -1;      // decision
static constexpr int32_t SAT_REASON_UNASSIGNED = -2; // internal sentinel

// Error codes.
static constexpr int32_t SAT_ERR_OK = 0;
static constexpr int32_t SAT_ERR_LEARNED_CLAUSES_OVERFLOW = 1;
static constexpr int32_t SAT_ERR_LEARNED_LITS_OVERFLOW = 2;
static constexpr int32_t SAT_ERR_PROOF_OVERFLOW = 3;
static constexpr int32_t SAT_ERR_WATCH_OVERFLOW = 4;
static constexpr int32_t SAT_ERR_INVALID_PROOF = 5;
static constexpr int32_t SAT_ERR_INVALID_MODEL = 6;
static constexpr int32_t SAT_ERR_INVALID_INPUT = 7;
static constexpr int32_t SAT_ERR_DECISION_EXHAUSTED = 8;

// Fail-fast trap. Used to enforce "verifier must not silently succeed" without host reads.
__device__ __forceinline__ void sat_trap() {
    asm volatile("trap;");
}

// ---------------------------------------------------------------------------
// Literal helpers (DIMACS: +/- var_id, 1-based var ids; 0 unused)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t sat_var(int32_t lit) {
    return (lit > 0) ? static_cast<uint32_t>(lit) : static_cast<uint32_t>(-lit);
}

__device__ __forceinline__ int8_t sat_lit_sign(int32_t lit) {
    return (lit > 0) ? SAT_VAL_TRUE : SAT_VAL_FALSE;
}

// Map literal -> dense index for watch lists: +v -> 2*(v-1), -v -> 2*(v-1)+1.
__device__ __forceinline__ uint32_t sat_lit_index(int32_t lit) {
    uint32_t v = sat_var(lit);
    uint32_t base = (v - 1u) * 2u;
    return (lit > 0) ? base : (base ^ 1u);
}

__device__ __forceinline__ uint32_t sat_neg_index(uint32_t lit_idx) {
    return lit_idx ^ 1u;
}

__device__ __forceinline__ int32_t sat_index_to_lit(uint32_t lit_idx) {
    uint32_t v = (lit_idx / 2u) + 1u;
    return (lit_idx % 2u == 0u) ? static_cast<int32_t>(v) : -static_cast<int32_t>(v);
}

// Evaluate a literal under current assignment.
// Returns: +1 satisfied, 0 unassigned, -1 falsified.
__device__ __forceinline__ int8_t sat_eval_lit(int32_t lit, const int8_t* __restrict__ assign) {
    uint32_t v = sat_var(lit);
    int8_t a = assign[v];
    if (a == SAT_VAL_UNASSIGNED) {
        return 0;
    }
    // If lit is positive, literal is satisfied when var=true.
    // If lit is negative, literal is satisfied when var=false.
    int8_t sign = sat_lit_sign(lit);
    return (a == sign) ? SAT_VAL_TRUE : SAT_VAL_FALSE;
}

// ---------------------------------------------------------------------------
// Clause view helpers (base CNF + learned arena)
// ---------------------------------------------------------------------------

struct SatClauseView {
    const int32_t* lits;
    uint32_t len;
};

__device__ __forceinline__ SatClauseView sat_get_clause(
    int32_t clause_id,
    const uint32_t* __restrict__ base_offsets,
    const int32_t* __restrict__ base_lits,
    uint32_t num_base_clauses,
    const uint32_t* __restrict__ learned_offsets,
    const int32_t* __restrict__ learned_lits
) {
    if (clause_id < static_cast<int32_t>(num_base_clauses)) {
        uint32_t c = static_cast<uint32_t>(clause_id);
        uint32_t s = base_offsets[c];
        uint32_t e = base_offsets[c + 1u];
        SatClauseView v;
        v.lits = base_lits + s;
        v.len = e - s;
        return v;
    }
    uint32_t lid = static_cast<uint32_t>(clause_id) - num_base_clauses;
    uint32_t s = learned_offsets[lid];
    uint32_t e = learned_offsets[lid + 1u];
    SatClauseView v;
    v.lits = learned_lits + s;
    v.len = e - s;
    return v;
}

// ---------------------------------------------------------------------------
// Watched literals: doubly-linked watch lists
// Node id = clause_id*2 + slot (0/1). Each node belongs to exactly one literal list.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void sat_watch_remove(
    int32_t node,
    int32_t* __restrict__ watch_head,
    int32_t* __restrict__ watch_next,
    int32_t* __restrict__ watch_prev,
    uint32_t lit_idx
) {
    // If the node is not currently linked into this list, do nothing.
    if (watch_prev[node] < 0 && watch_head[lit_idx] != node && watch_next[node] < 0) {
        return;
    }
    int32_t prev = watch_prev[node];
    int32_t next = watch_next[node];
    if (prev >= 0) {
        watch_next[prev] = next;
    } else {
        // node is head
        watch_head[lit_idx] = next;
    }
    if (next >= 0) {
        watch_prev[next] = prev;
    }
    watch_prev[node] = -1;
    watch_next[node] = -1;
}

__device__ __forceinline__ void sat_watch_insert_head(
    int32_t node,
    int32_t* __restrict__ watch_head,
    int32_t* __restrict__ watch_next,
    int32_t* __restrict__ watch_prev,
    uint32_t lit_idx
) {
    int32_t head = watch_head[lit_idx];
    watch_prev[node] = -1;
    watch_next[node] = head;
    if (head >= 0) {
        watch_prev[head] = node;
    }
    watch_head[lit_idx] = node;
}

// ---------------------------------------------------------------------------
// Deterministic VSIDS-style activity (integer, monotone bump + rescale)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t sat_bump_var(uint32_t v, uint32_t* __restrict__ var_activity, uint32_t bump) {
    uint32_t a = var_activity[v];
    uint32_t na = a + bump;
    if (na < a) {
        // Saturate on overflow; rescaling will bring values back down.
        na = 0xFFFFFFFFu;
    }
    var_activity[v] = na;
    return na;
}

// ---------------------------------------------------------------------------
// Decision heap (unassigned vars only), deterministic max-heap by (activity desc, var id asc)
// ---------------------------------------------------------------------------

static constexpr uint32_t SAT_HEAP_NONE = 0xFFFFFFFFu;

__device__ __forceinline__ bool sat_heap_better(
    uint32_t a,
    uint32_t b,
    const uint32_t* __restrict__ var_activity
) {
    uint32_t aa = var_activity[a];
    uint32_t bb = var_activity[b];
    if (aa != bb) {
        return aa > bb;
    }
    return a < b;
}

__device__ __forceinline__ void sat_heap_swap(
    uint32_t i,
    uint32_t j,
    uint32_t* __restrict__ heap,
    uint32_t* __restrict__ heap_pos
) {
    uint32_t vi = heap[i];
    uint32_t vj = heap[j];
    heap[i] = vj;
    heap[j] = vi;
    heap_pos[vi] = j;
    heap_pos[vj] = i;
}

__device__ __forceinline__ void sat_heap_sift_up(
    uint32_t i,
    uint32_t* __restrict__ heap,
    uint32_t* __restrict__ heap_pos,
    const uint32_t* __restrict__ var_activity
) {
    while (i > 0) {
        uint32_t p = (i - 1u) >> 1;
        uint32_t vp = heap[p];
        uint32_t vi = heap[i];
        if (sat_heap_better(vp, vi, var_activity)) {
            break;
        }
        sat_heap_swap(p, i, heap, heap_pos);
        i = p;
    }
}

__device__ __forceinline__ void sat_heap_sift_down(
    uint32_t i,
    uint32_t heap_size,
    uint32_t* __restrict__ heap,
    uint32_t* __restrict__ heap_pos,
    const uint32_t* __restrict__ var_activity
) {
    for (;;) {
        uint32_t l = (i << 1) + 1u;
        if (l >= heap_size) {
            return;
        }
        uint32_t r = l + 1u;
        uint32_t best = l;
        if (r < heap_size) {
            if (sat_heap_better(heap[r], heap[l], var_activity)) {
                best = r;
            }
        }
        uint32_t vi = heap[i];
        uint32_t vb = heap[best];
        if (sat_heap_better(vi, vb, var_activity)) {
            return;
        }
        sat_heap_swap(i, best, heap, heap_pos);
        i = best;
    }
}

__device__ __forceinline__ void sat_heap_remove(
    uint32_t v,
    uint32_t* __restrict__ heap_size,
    uint32_t* __restrict__ heap,
    uint32_t* __restrict__ heap_pos,
    const uint32_t* __restrict__ var_activity
) {
    uint32_t sz = *heap_size;
    uint32_t i = heap_pos[v];
    // Defensive: if heap_pos is corrupted but v is the current root, treat it as a root removal.
    // This prevents infinite loops in stale-root cleanup paths.
    if ((i == SAT_HEAP_NONE || i >= sz) && sz > 0u && heap[0] == v) {
        i = 0u;
    }
    if (i == SAT_HEAP_NONE || i >= sz) {
        return;
    }
    sz -= 1u;
    *heap_size = sz;
    heap_pos[v] = SAT_HEAP_NONE;
    if (i == sz) {
        return;
    }
    uint32_t mv = heap[sz];
    heap[i] = mv;
    heap_pos[mv] = i;
    // Restore heap property.
    if (i > 0) {
        uint32_t p = (i - 1u) >> 1;
        if (!sat_heap_better(heap[p], heap[i], var_activity)) {
            sat_heap_sift_up(i, heap, heap_pos, var_activity);
            return;
        }
    }
    sat_heap_sift_down(i, sz, heap, heap_pos, var_activity);
}

__device__ __forceinline__ void sat_heap_insert(
    uint32_t v,
    uint32_t* __restrict__ heap_size,
    uint32_t* __restrict__ heap,
    uint32_t* __restrict__ heap_pos,
    const uint32_t* __restrict__ var_activity
) {
    uint32_t i = *heap_size;
    heap[i] = v;
    heap_pos[v] = i;
    *heap_size = i + 1u;
    sat_heap_sift_up(i, heap, heap_pos, var_activity);
}

// ---------------------------------------------------------------------------
// CDCL core helpers (enqueue, propagate, analyze, backtrack)
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool sat_is_decision_var(
    uint32_t v,
    uint32_t base_limit,
    uint32_t extra_base,
    uint32_t extra_end_excl
) {
    if (v == 0u) {
        return false;
    }
    if (v <= base_limit) {
        return true;
    }
    return (v >= extra_base && v < extra_end_excl);
}

__device__ __forceinline__ bool sat_enqueue(
    int32_t lit,
    int32_t reason_clause,
    uint32_t decision_level,
    int8_t* __restrict__ assign,
    uint32_t* __restrict__ level,
    int32_t* __restrict__ reason,
    int32_t* __restrict__ trail,
    uint32_t* __restrict__ trail_len,
    uint32_t* __restrict__ assigned_count,
    int8_t* __restrict__ phase,
    uint32_t decision_base_limit,
    uint32_t decision_extra_base,
    uint32_t decision_extra_end_excl,
    uint32_t* __restrict__ decision_heap,
    uint32_t* __restrict__ decision_heap_pos,
    uint32_t* __restrict__ decision_heap_size,
    const uint32_t* __restrict__ var_activity
) {
    uint32_t v = sat_var(lit);
    int8_t val = sat_lit_sign(lit);
    int8_t cur = assign[v];
    if (cur == SAT_VAL_UNASSIGNED) {
        assign[v] = val;
        level[v] = decision_level;
        reason[v] = reason_clause;
        phase[v] = val; // phase saving (deterministic)
        trail[*trail_len] = lit;
        *trail_len = *trail_len + 1u;
        *assigned_count = *assigned_count + 1u;
        if (sat_is_decision_var(v, decision_base_limit, decision_extra_base, decision_extra_end_excl)) {
            sat_heap_remove(v, decision_heap_size, decision_heap, decision_heap_pos, var_activity);
        }
        return true;
    }
    return cur == val;
}

__device__ __forceinline__ void sat_unassign_lit(
    int32_t lit,
    int8_t* __restrict__ assign,
    uint32_t* __restrict__ level,
    int32_t* __restrict__ reason,
    uint32_t decision_base_limit,
    uint32_t decision_extra_base,
    uint32_t decision_extra_end_excl,
    uint32_t* __restrict__ decision_heap_size,
    uint32_t* __restrict__ decision_heap,
    uint32_t* __restrict__ decision_heap_pos,
    const uint32_t* __restrict__ var_activity
) {
    uint32_t v = sat_var(lit);
    assign[v] = SAT_VAL_UNASSIGNED;
    level[v] = 0;
    reason[v] = SAT_REASON_UNASSIGNED;
    if (sat_is_decision_var(v, decision_base_limit, decision_extra_base, decision_extra_end_excl)) {
        sat_heap_insert(v, decision_heap_size, decision_heap, decision_heap_pos, var_activity);
    }
}

__device__ __forceinline__ void sat_backtrack(
    uint32_t back_level,
    uint32_t* __restrict__ decision_level,
    uint32_t num_levels_capacity,
    uint32_t* __restrict__ trail_lim,
    uint32_t* __restrict__ qhead,
    int32_t* __restrict__ trail,
    uint32_t* __restrict__ trail_len,
    uint32_t* __restrict__ assigned_count,
    int8_t* __restrict__ assign,
    uint32_t* __restrict__ level,
    int32_t* __restrict__ reason,
    uint32_t decision_base_limit,
    uint32_t decision_extra_base,
    uint32_t decision_extra_end_excl,
    uint32_t* __restrict__ decision_heap_size,
    uint32_t* __restrict__ decision_heap,
    uint32_t* __restrict__ decision_heap_pos,
    const uint32_t* __restrict__ var_activity
) {
    uint32_t cur_level = *decision_level;
    if (back_level >= cur_level) {
        return;
    }
    // trail_lim is indexed by decision level (1..), with trail_lim[0]=0.
    // To keep assignments up to back_level, truncate at trail_lim[back_level+1].
    uint32_t cut = 0;
    if (back_level + 1u < num_levels_capacity) {
        cut = trail_lim[back_level + 1u];
    }
    // Unassign in reverse order (deterministic).
    for (uint32_t i = *trail_len; i > cut; i--) {
        int32_t lit = trail[i - 1u];
        sat_unassign_lit(
            lit,
            assign,
            level,
            reason,
            decision_base_limit,
            decision_extra_base,
            decision_extra_end_excl,
            decision_heap_size,
            decision_heap,
            decision_heap_pos,
            var_activity);
        *assigned_count = *assigned_count - 1u;
    }
    *trail_len = cut;
    if (*qhead > cut) {
        *qhead = cut;
    }
    *decision_level = back_level;
    // Clear higher-level markers.
    for (uint32_t l = back_level + 1u; l <= cur_level; l++) {
        if (l < num_levels_capacity) {
            trail_lim[l] = 0;
        }
    }
}

// Propagation returns conflict clause id (>=0) or -1 if none.
//
// Block-parallel execution model:
// - Thread 0 owns all global solver mutations (trail, assignments, watch list relinking).
// - The whole block participates in clause inspection for watched-literal propagation.
// - Thread 0 commits updates deterministically.
__device__ __forceinline__ int32_t sat_propagate(
    const uint32_t* __restrict__ base_offsets,
    const int32_t* __restrict__ base_lits,
    uint32_t num_vars,
    uint32_t num_base_clauses,
    const uint32_t* __restrict__ learned_offsets,
    const int32_t* __restrict__ learned_lits,
    const uint8_t* __restrict__ learned_deleted,
    uint32_t* __restrict__ learned_activity,
    uint32_t* __restrict__ watch0_pos,
    uint32_t* __restrict__ watch1_pos,
    int32_t* __restrict__ watch_head,
    int32_t* __restrict__ watch_next,
    int32_t* __restrict__ watch_prev,
    uint32_t* __restrict__ qhead,
    uint32_t decision_level,
    int8_t* __restrict__ assign,
    uint32_t* __restrict__ level,
    int32_t* __restrict__ reason,
    int32_t* __restrict__ trail,
    uint32_t* __restrict__ trail_len,
    uint32_t* __restrict__ assigned_count,
    int8_t* __restrict__ phase,
    uint32_t decision_base_limit,
    uint32_t decision_extra_base,
    uint32_t decision_extra_end_excl,
    uint32_t* __restrict__ decision_heap_size,
    uint32_t* __restrict__ decision_heap,
    uint32_t* __restrict__ decision_heap_pos,
    const uint32_t* __restrict__ var_activity
) {
    static constexpr uint32_t SAT_PROP_CHUNK = 256u;

    static constexpr uint8_t SAT_ACT_SKIP = 0;
    static constexpr uint8_t SAT_ACT_MOVE = 1;
    static constexpr uint8_t SAT_ACT_UNIT = 2;
    static constexpr uint8_t SAT_ACT_CONFLICT = 3;
    static constexpr uint8_t SAT_ACT_DELETE = 4;

    __shared__ int32_t sh_result;
    __shared__ uint32_t sh_has_p;
    __shared__ int32_t sh_p;
    __shared__ int32_t sh_list_cursor;
    __shared__ uint32_t sh_chunk_n;
    __shared__ int32_t sh_nodes[SAT_PROP_CHUNK];
    __shared__ uint8_t sh_act[SAT_PROP_CHUNK];
    __shared__ uint32_t sh_move_pos[SAT_PROP_CHUNK];
    __shared__ uint32_t sh_move_idx[SAT_PROP_CHUNK];
    __shared__ int32_t sh_unit_lit[SAT_PROP_CHUNK];

    if (threadIdx.x == 0) {
        sh_result = -1;
    }
    __syncthreads();

    for (;;) {
        // Pop next literal to propagate (thread 0), broadcast to block.
        if (threadIdx.x == 0) {
            if (*qhead < *trail_len) {
                sh_p = trail[*qhead];
                *qhead = *qhead + 1u;
                sh_has_p = 1u;
            } else {
                sh_has_p = 0u;
            }
        }
        __syncthreads();

        if (sh_has_p == 0u) {
            break;
        }

        int32_t p = sh_p;
        uint32_t fals_idx = sat_lit_index(-p);

        // Walk the watch list for fals_idx in fixed-size chunks.
        if (threadIdx.x == 0) {
            sh_list_cursor = watch_head[fals_idx];
        }
        __syncthreads();

        for (;;) {
            if (sh_list_cursor < 0 || sh_result >= 0) {
                break;
            }

            // Stage up to chunk_cap nodes from the watch list without modifying it.
            if (threadIdx.x == 0) {
                uint32_t chunk_cap = static_cast<uint32_t>(blockDim.x);
                if (chunk_cap > SAT_PROP_CHUNK) {
                    chunk_cap = SAT_PROP_CHUNK;
                }

                uint32_t n = 0;
                int32_t node = sh_list_cursor;
                while (node >= 0 && n < chunk_cap) {
                    sh_nodes[n++] = node;
                    node = watch_next[node];
                }
                sh_chunk_n = n;
                sh_list_cursor = node;
            }
            __syncthreads();

            uint32_t n = sh_chunk_n;
            uint32_t t = static_cast<uint32_t>(threadIdx.x);
            if (t < n) {
                int32_t node = sh_nodes[t];
                int32_t clause_id = node / 2;
                uint32_t slot = static_cast<uint32_t>(node & 1);

                uint8_t act = SAT_ACT_SKIP;
                uint32_t move_pos = 0;
                uint32_t move_idx = 0;
                int32_t unit_lit = 0;

                // Skip deleted learned clauses (but remove stale watch nodes deterministically).
                if (clause_id >= static_cast<int32_t>(num_base_clauses)) {
                    uint32_t lid = static_cast<uint32_t>(clause_id) - num_base_clauses;
                    if (learned_deleted[lid] != 0) {
                        act = SAT_ACT_DELETE;
                    }
                }

                if (act != SAT_ACT_DELETE) {
                    SatClauseView cv = sat_get_clause(
                        clause_id, base_offsets, base_lits, num_base_clauses, learned_offsets, learned_lits);
                    uint32_t w0 = watch0_pos[static_cast<uint32_t>(clause_id)];
                    uint32_t w1 = watch1_pos[static_cast<uint32_t>(clause_id)];
                    uint32_t wpos_false = (slot == 0) ? w0 : w1;
                    uint32_t wpos_other = (slot == 0) ? w1 : w0;

                    if (cv.len == 0) {
                        act = SAT_ACT_CONFLICT;
                    } else {
                        // Unit clause case: both watches may be identical.
                        int32_t other_lit = (wpos_other < cv.len) ? cv.lits[wpos_other] : 0;
                        if (other_lit == 0) {
                            act = SAT_ACT_CONFLICT;
                        } else if (sat_eval_lit(other_lit, assign) == SAT_VAL_TRUE) {
                            act = SAT_ACT_SKIP;
                        } else {
                            // Try to find a new watch in this clause that is not falsified.
                            bool moved = false;
                            for (uint32_t k = 0; k < cv.len; k++) {
                                if (k == wpos_other || k == wpos_false) {
                                    continue;
                                }
                                int32_t lit = cv.lits[k];
                                if (sat_eval_lit(lit, assign) != SAT_VAL_FALSE) {
                                    moved = true;
                                    act = SAT_ACT_MOVE;
                                    move_pos = k;
                                    move_idx = sat_lit_index(lit);
                                    break;
                                }
                            }

                            if (!moved) {
                                // No new watch found: clause is unit or conflicting.
                                int8_t other_eval = sat_eval_lit(other_lit, assign);
                                if (other_eval == SAT_VAL_FALSE) {
                                    act = SAT_ACT_CONFLICT;
                                } else {
                                    act = SAT_ACT_UNIT;
                                    unit_lit = other_lit;
                                }
                            }
                        }
                    }
                }

                sh_act[t] = act;
                sh_move_pos[t] = move_pos;
                sh_move_idx[t] = move_idx;
                sh_unit_lit[t] = unit_lit;
            }
            __syncthreads();

            // Deterministic commit: apply actions on thread 0 only.
            if (threadIdx.x == 0) {
                int32_t conflict = -1;
                for (uint32_t i = 0; i < n; i++) {
                    int32_t node = sh_nodes[i];
                    int32_t clause_id = node / 2;
                    uint32_t slot = static_cast<uint32_t>(node & 1);
                    uint8_t act = sh_act[i];

                    switch (act) {
                    case SAT_ACT_SKIP:
                        break;
                    case SAT_ACT_DELETE:
                        sat_watch_remove(node, watch_head, watch_next, watch_prev, fals_idx);
                        break;
                    case SAT_ACT_MOVE: {
                        uint32_t new_idx = sh_move_idx[i];
                        sat_watch_remove(node, watch_head, watch_next, watch_prev, fals_idx);
                        sat_watch_insert_head(node, watch_head, watch_next, watch_prev, new_idx);
                        if (slot == 0) {
                            watch0_pos[static_cast<uint32_t>(clause_id)] = sh_move_pos[i];
                        } else {
                            watch1_pos[static_cast<uint32_t>(clause_id)] = sh_move_pos[i];
                        }
                        break;
                    }
                    case SAT_ACT_CONFLICT:
                        if (conflict < 0 || clause_id < conflict) {
                            conflict = clause_id;
                        }
                        break;
                    case SAT_ACT_UNIT: {
                        int32_t unit_lit = sh_unit_lit[i];
                        if (!sat_enqueue(unit_lit, clause_id, decision_level, assign, level, reason, trail, trail_len,
                                         assigned_count, phase,
                                         decision_base_limit,
                                         decision_extra_base,
                                         decision_extra_end_excl,
                                         decision_heap, decision_heap_pos, decision_heap_size, var_activity)) {
                            if (conflict < 0 || clause_id < conflict) {
                                conflict = clause_id;
                            }
                        } else if (clause_id >= static_cast<int32_t>(num_base_clauses)) {
                            uint32_t lid = static_cast<uint32_t>(clause_id) - num_base_clauses;
                            learned_activity[lid] += 1u;
                        }
                        break;
                    }
                    default:
                        // Defensive: unknown action.
                        sat_trap();
                        break;
                    }
                }

                if (conflict >= 0) {
                    sh_result = conflict;
                }
            }
            __syncthreads();

            if (sh_result >= 0) {
                break;
            }
        }

        __syncthreads();

        if (sh_result >= 0) {
            break;
        }
    }

    __syncthreads();
    return sh_result;
}

// Compute LBD (Literal Block Distance) for learned clause: number of distinct decision levels.
__device__ __forceinline__ uint32_t sat_compute_lbd(
    const int32_t* __restrict__ lits,
    uint32_t len,
    const uint32_t* __restrict__ level
) {
    // O(len^2) unique count is fine for typical learned clause sizes.
    uint32_t uniq = 0;
    uint32_t levels[64]; // common case fast path; fallback to coarse bucket if longer
    if (len <= 64u) {
        for (uint32_t i = 0; i < len; i++) {
            uint32_t v = sat_var(lits[i]);
            uint32_t lv = level[v];
            bool seen = false;
            for (uint32_t j = 0; j < uniq; j++) {
                if (levels[j] == lv) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                levels[uniq++] = lv;
            }
        }
        return uniq;
    }
    // Coarse fallback: cap LBD at 256 by bucketing levels (still deterministic).
    uint8_t bucket[256];
    for (uint32_t i = 0; i < 256u; i++) {
        bucket[i] = 0;
    }
    for (uint32_t i = 0; i < len; i++) {
        uint32_t v = sat_var(lits[i]);
        uint32_t lv = level[v];
        uint32_t b = lv & 255u;
        bucket[b] = 1;
    }
    uint32_t c = 0;
    for (uint32_t i = 0; i < 256u; i++) {
        c += static_cast<uint32_t>(bucket[i]);
    }
    return c;
}

// Conflict analysis for decision_level==0 UNSAT. Produces a resolution trace that derives the empty clause
// by replaying unit-propagation reasons in reverse trail order.
__device__ __forceinline__ bool sat_analyze_level0_unsat(
    int32_t conflict_clause,
    const uint32_t* __restrict__ base_offsets,
    const int32_t* __restrict__ base_lits,
    uint32_t num_base_clauses,
    const uint32_t* __restrict__ learned_offsets,
    const int32_t* __restrict__ learned_lits,
    const uint8_t* __restrict__ learned_deleted,
    uint32_t learned_count,
    uint32_t num_vars,
    const int32_t* __restrict__ reason,
    const int32_t* __restrict__ trail,
    uint32_t trail_len,
    uint8_t* __restrict__ clause_mark, // len = num_vars+1
    uint32_t* __restrict__ proof_vars_tmp,
    uint32_t* __restrict__ proof_reason_tmp,
    uint32_t* __restrict__ out_proof_steps
) {
    uint32_t max_clause_id = num_base_clauses + learned_count;
    if (conflict_clause < 0 || static_cast<uint32_t>(conflict_clause) >= max_clause_id) {
        return false;
    }
    if (conflict_clause >= static_cast<int32_t>(num_base_clauses)) {
        uint32_t lid = static_cast<uint32_t>(conflict_clause) - num_base_clauses;
        if (lid >= learned_count || learned_deleted[lid] != 0) {
            return false;
        }
    }

    // clause_mark[v] encodes clause membership: 0=absent, 1=+v, 2=-v.
    for (uint32_t v = 0; v <= num_vars; v++) {
        clause_mark[v] = 0;
    }

    SatClauseView c0 = sat_get_clause(conflict_clause,
                                      base_offsets, base_lits, num_base_clauses,
                                      learned_offsets, learned_lits);
    uint32_t cur_len = 0;
    for (uint32_t i = 0; i < c0.len; i++) {
        int32_t lit = c0.lits[i];
        uint32_t v = sat_var(lit);
        uint8_t s = (lit > 0) ? 1u : 2u;
        uint8_t prev = clause_mark[v];
        if (prev == 0) {
            clause_mark[v] = s;
            cur_len++;
        } else if (prev == s) {
            // duplicate
        } else {
            // tautology (should be unreachable for a conflicting clause)
            return false;
        }
    }

    uint32_t proof_steps = 0;
    if (cur_len == 0) {
        *out_proof_steps = 0;
        return true;
    }

    // Resolve in reverse trail order. Any literals introduced by reasons must be for earlier trail vars.
    for (uint32_t ti = trail_len; ti > 0 && cur_len > 0; ti--) {
        int32_t p = trail[ti - 1u]; // assigned literal (satisfied)
        uint32_t v = sat_var(p);
        // In a conflicting clause, the present literal for v is falsified, i.e., -p.
        uint8_t want = (p > 0) ? 2u : 1u;
        if (clause_mark[v] != want) {
            continue;
        }

        int32_t rcid = reason[v];
        if (rcid < 0 || static_cast<uint32_t>(rcid) >= max_clause_id) {
            return false;
        }
        if (rcid >= static_cast<int32_t>(num_base_clauses)) {
            uint32_t lid = static_cast<uint32_t>(rcid) - num_base_clauses;
            if (lid >= learned_count || learned_deleted[lid] != 0) {
                return false;
            }
        }

        SatClauseView rc = sat_get_clause(rcid,
                                          base_offsets, base_lits, num_base_clauses,
                                          learned_offsets, learned_lits);
        // The reason clause must contain the complementary literal p.
        bool has_p = false;
        for (uint32_t i = 0; i < rc.len; i++) {
            if (rc.lits[i] == p) {
                has_p = true;
                break;
            }
        }
        if (!has_p) {
            return false;
        }

        proof_vars_tmp[proof_steps] = v;
        proof_reason_tmp[proof_steps] = static_cast<uint32_t>(rcid);
        proof_steps++;

        // Remove v from the current clause.
        clause_mark[v] = 0;
        cur_len--;

        // Add all literals from the reason clause except v (resolution).
        for (uint32_t i = 0; i < rc.len; i++) {
            int32_t lit = rc.lits[i];
            if (sat_var(lit) == v) {
                continue;
            }
            uint32_t w = sat_var(lit);
            uint8_t s = (lit > 0) ? 1u : 2u;
            uint8_t prev = clause_mark[w];
            if (prev == 0) {
                clause_mark[w] = s;
                cur_len++;
            } else if (prev == s) {
                // duplicate
            } else {
                // tautology in the resolvent is disallowed in this proof model
                return false;
            }
        }
    }

    if (cur_len != 0) {
        return false;
    }

    *out_proof_steps = proof_steps;
    return true;
}

// Conflict analysis (1-UIP). Returns learned clause length in out_learnt_len and backtrack level in out_bt_level.
// Also records a resolution trace for proof checking.
__device__ __forceinline__ bool sat_analyze(
    int32_t conflict_clause,
    const uint32_t* __restrict__ base_offsets,
    const int32_t* __restrict__ base_lits,
    uint32_t num_base_clauses,
    const uint32_t* __restrict__ learned_offsets,
    const int32_t* __restrict__ learned_lits,
    const uint8_t* __restrict__ learned_deleted,
    uint32_t learned_count,
    uint32_t decision_level,
    const int8_t* __restrict__ assign,
    const uint32_t* __restrict__ level,
    const int32_t* __restrict__ reason,
    const int32_t* __restrict__ trail,
    uint32_t trail_len,
    uint8_t* __restrict__ seen,
    int32_t* __restrict__ learnt_out,
    uint32_t* __restrict__ out_learnt_len,
    uint32_t* __restrict__ out_bt_level,
    // proof temp
    uint32_t* __restrict__ proof_vars_tmp,
    uint32_t* __restrict__ proof_reason_tmp,
    uint32_t* __restrict__ out_proof_steps
) {
    uint32_t learnt_len = 0;
    uint32_t pathC = 0;
    int32_t p = 0;
    uint32_t idx = trail_len;

    uint32_t proof_steps = 0;

    int32_t c = conflict_clause;
    uint32_t max_clause_id = num_base_clauses + learned_count;
    if (c < 0 || static_cast<uint32_t>(c) >= max_clause_id) {
        return false;
    }
    for (;;) {
        // Incorporate clause c.
        if (c < 0 || static_cast<uint32_t>(c) >= max_clause_id) {
            return false;
        }
        if (c >= static_cast<int32_t>(num_base_clauses)) {
            uint32_t lid = static_cast<uint32_t>(c) - num_base_clauses;
            if (lid >= learned_count || learned_deleted[lid] != 0) {
                return false;
            }
        }
        SatClauseView cv = sat_get_clause(c, base_offsets, base_lits, num_base_clauses, learned_offsets, learned_lits);

        for (uint32_t i = 0; i < cv.len; i++) {
            int32_t q = cv.lits[i];
            // After the first iteration, clause c is a reason clause for p. We must skip p itself
            // (it's the resolved literal), regardless of its position.
            if (p != 0 && q == p) {
                continue;
            }
            uint32_t v = sat_var(q);
            uint32_t lv = level[v];
            if (seen[v]) {
                continue;
            }
            seen[v] = 1;
            if (lv == decision_level) {
                pathC++;
            } else {
                learnt_out[learnt_len++] = q;
            }
        }

        // Select next literal to resolve.
        bool found = false;
        while (idx > 0) {
            idx--;
            p = trail[idx];
            uint32_t v = sat_var(p);
            if (seen[v]) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
        if (pathC == 0) {
            return false;
        }

        uint32_t pv = sat_var(p);
        seen[pv] = 0;
        pathC--;
        if (pathC == 0) {
            break;
        }
        int32_t rc = reason[pv];
        if (rc < 0) {
            // Decision literal shouldn't be resolved (UIP should have terminated already).
            return false;
        }
        if (static_cast<uint32_t>(rc) >= max_clause_id) {
            return false;
        }
        if (rc >= static_cast<int32_t>(num_base_clauses)) {
            uint32_t lid = static_cast<uint32_t>(rc) - num_base_clauses;
            if (lid >= learned_count || learned_deleted[lid] != 0) {
                return false;
            }
        }
        // Record resolution step: resolve current resolvent on pv using reason[pv].
        proof_vars_tmp[proof_steps] = pv;
        proof_reason_tmp[proof_steps] = static_cast<uint32_t>(rc);
        proof_steps++;
        c = rc;
    }

    // First literal is the UIP (negation of p).
    learnt_out[learnt_len++] = -p;

    // Determine backtrack level: max level of literals excluding UIP.
    uint32_t bt = 0;
    for (uint32_t i = 0; i + 1u < learnt_len; i++) {
        uint32_t v = sat_var(learnt_out[i]);
        uint32_t lv = level[v];
        if (lv > bt) {
            bt = lv;
        }
    }

    // Place UIP at position 0 for deterministic assert/propagate behavior.
    // learnt_out currently has UIP at the end; move it to front.
    int32_t uip = learnt_out[learnt_len - 1u];
    for (uint32_t i = learnt_len - 1u; i > 0; i--) {
        learnt_out[i] = learnt_out[i - 1u];
    }
    learnt_out[0] = uip;

    *out_learnt_len = learnt_len;
    *out_bt_level = bt;
    *out_proof_steps = proof_steps;
    // Clear seen marks for all literals in the learned clause (non-decision-level vars remain marked otherwise).
    for (uint32_t i = 0; i < learnt_len; i++) {
        uint32_t v = sat_var(learnt_out[i]);
        seen[v] = 0;
    }
    return true;
}

// Luby sequence (1-based index).
__device__ __forceinline__ uint32_t sat_luby(uint32_t i) {
    uint32_t k = 1;
    while (((1u << k) - 1u) < i) {
        k++;
    }
    while (i != ((1u << k) - 1u)) {
        i = i - (1u << (k - 1u)) + 1u;
        k = 1;
        while (((1u << k) - 1u) < i) {
            k++;
        }
    }
    return 1u << (k - 1u);
}

// ---------------------------------------------------------------------------
// Kernel: sat_cdcl_solve
// ---------------------------------------------------------------------------

extern "C" __global__ void sat_cdcl_solve(
    const uint32_t* __restrict__ compile_needed,
    // Base CNF (CSR)
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ clause_lits,
    const uint32_t* __restrict__ num_vars,    // len=1
    const uint32_t* __restrict__ num_clauses, // len=1
    const uint32_t* __restrict__ decision_base_limit,  // len=1
    const uint32_t* __restrict__ decision_extra_base,  // len=1
    const uint32_t* __restrict__ decision_extra_count, // len=1
    uint32_t var_cap,
    uint32_t clause_cap,
    // Solver configuration
    uint32_t max_learned_clauses,
    uint32_t max_learned_lits,
    uint32_t max_proof_u32,
    uint32_t restart_base,        // e.g. 100
    uint32_t reduce_interval,     // e.g. 2000
    const uint32_t* __restrict__ learned_import_count, // len = 1
    // Variable state
    int8_t* __restrict__ assign,      // len = num_vars+1
    uint32_t* __restrict__ level,     // len = num_vars+1
    int32_t* __restrict__ reason,     // len = num_vars+1
    uint32_t* __restrict__ var_activity, // len = num_vars+1
    int8_t* __restrict__ var_phase,   // len = num_vars+1
    // Decision heap (unassigned vars only)
    uint32_t* __restrict__ decision_heap,     // len = var_cap+1
    uint32_t* __restrict__ decision_heap_pos, // len = var_cap+1
    // Trail / levels
    int32_t* __restrict__ trail,      // len = num_vars+1
    uint32_t* __restrict__ trail_lim, // len = num_vars+1 (trail_lim[0]=0)
    // Analysis scratch
    uint8_t* __restrict__ seen,           // len = num_vars+1
    int32_t* __restrict__ learnt_tmp,     // len = num_vars+1
    uint32_t* __restrict__ proof_vars_tmp,   // len = num_vars+1
    uint32_t* __restrict__ proof_reason_tmp, // len = num_vars+1
    // Watched literals
    uint32_t* __restrict__ watch0_pos, // len = num_clauses + max_learned_clauses
    uint32_t* __restrict__ watch1_pos, // len = num_clauses + max_learned_clauses
    int32_t* __restrict__ watch_head,  // len = 2*num_vars
    int32_t* __restrict__ watch_next,  // len = 2*(num_clauses + max_learned_clauses)
    int32_t* __restrict__ watch_prev,  // len = 2*(num_clauses + max_learned_clauses)
    // Learned clause arena
    uint32_t* __restrict__ learned_offsets, // len = max_learned_clauses+1
    int32_t* __restrict__ learned_lits,     // len = max_learned_lits
    uint8_t* __restrict__ learned_deleted,  // len = max_learned_clauses
    uint32_t* __restrict__ learned_lbd,      // len = max_learned_clauses
    uint32_t* __restrict__ learned_activity, // len = max_learned_clauses
    uint8_t* __restrict__ learned_locked,    // len = max_learned_clauses
    // Proof trace arena (resolution trace)
    uint32_t* __restrict__ proof_offsets,   // len = max_learned_clauses+1
    uint32_t* __restrict__ proof_data,      // len = max_proof_u32
    // Outputs
    int32_t* __restrict__ out_status, // len = 1
    int32_t* __restrict__ out_error,  // len = 1
    uint32_t* __restrict__ out_learned_count // len = 1
) {
    // Deterministic single-block-per-instance execution.
    if (blockIdx.x != 0) {
        return;
    }
    if (compile_needed[0] == 0u) {
        return;
    }

    __shared__ uint32_t sh_nv;
    __shared__ uint32_t sh_nc;
    __shared__ uint32_t sh_decision_base;
    __shared__ uint32_t sh_decision_extra_base;
    __shared__ uint32_t sh_decision_extra_end_excl;
    __shared__ uint32_t sh_imported_learned_count;
    __shared__ uint32_t sh_done;
    __shared__ int32_t sh_final_status;
    __shared__ int32_t sh_final_error;
    __shared__ uint32_t sh_final_learned_count;

    if (threadIdx.x == 0) {
        sh_nv = num_vars[0];
        sh_nc = num_clauses[0];
        sh_decision_base = decision_base_limit[0];
        uint32_t eb = decision_extra_base[0];
        uint32_t ec = decision_extra_count[0];
        uint32_t ee = eb + ec;
        sh_decision_extra_base = eb;
        sh_decision_extra_end_excl = ee;
        sh_imported_learned_count = learned_import_count[0];
        sh_done = 0u;
        sh_final_status = SAT_STATUS_ERROR;
        sh_final_error = SAT_ERR_OK;
        sh_final_learned_count = 0u;
    }
    __syncthreads();

    uint32_t nv = sh_nv;
    uint32_t nc = sh_nc;
    uint32_t decision_base = sh_decision_base;
    uint32_t decision_extra_base_v = sh_decision_extra_base;
    uint32_t decision_extra_end_excl = sh_decision_extra_end_excl;

    // Validate inputs once and coordinate a single exit path for the whole block.
    if (threadIdx.x == 0) {
        bool ok = true;
        if (nv == 0u || nv > var_cap || nc > clause_cap) {
            ok = false;
        }
        if (decision_base == 0u || decision_base > nv) {
            ok = false;
        }
        // If decision_extra_count overflowed, ee < eb.
        if (decision_extra_end_excl < decision_extra_base_v) {
            ok = false;
        }
        if (sh_imported_learned_count > max_learned_clauses) {
            ok = false;
        }
        if (sh_imported_learned_count > 0u) {
            if (learned_offsets[sh_imported_learned_count] > max_learned_lits ||
                proof_offsets[sh_imported_learned_count] > max_proof_u32) {
                ok = false;
            }
        }
        // If an extra decision range is configured, validate it is within 1..=nv.
        if (decision_extra_end_excl != decision_extra_base_v) {
            if (decision_extra_base_v == 0u) {
                ok = false;
            }
            // extra_end_excl may equal nv+1 when the range ends at nv.
            if (decision_extra_end_excl > nv + 1u) {
                ok = false;
            }
        }
        if (!ok) {
            sh_done = 1u;
            sh_final_status = SAT_STATUS_ERROR;
            sh_final_error = SAT_ERR_INVALID_INPUT;
        }
    }
    __syncthreads();
    if (sh_done) {
        if (threadIdx.x == 0) {
            out_status[0] = sh_final_status;
            out_error[0] = sh_final_error;
            out_learned_count[0] = sh_final_learned_count;
        }
        return;
    }

    // Initialize outputs (thread 0 only).
    if (threadIdx.x == 0) {
        out_status[0] = SAT_STATUS_ERROR;
        out_error[0] = SAT_ERR_OK;
        out_learned_count[0] = 0;
    }

    // Initialize variable arrays (block-parallel, deterministic).
    for (uint32_t v = static_cast<uint32_t>(threadIdx.x); v <= nv; v += static_cast<uint32_t>(blockDim.x)) {
        assign[v] = SAT_VAL_UNASSIGNED;
        level[v] = 0;
        reason[v] = SAT_REASON_UNASSIGNED;
        var_activity[v] = 0;
        var_phase[v] = SAT_VAL_FALSE; // deterministic default: false
        seen[v] = 0;
    }
    __syncthreads();

    // Seed initial variable activity from literal occurrence counts, and derive a deterministic
    // starting phase. We reuse `level[v]` as a temporary counter for negative occurrences, then
    // reset it to 0 before starting the solve.
    if (threadIdx.x == 0) {
        for (uint32_t c = 0; c < nc; c++) {
            uint32_t s = clause_offsets[c];
            uint32_t e = clause_offsets[c + 1u];
            for (uint32_t i = s; i < e; i++) {
                int32_t lit = clause_lits[i];
                uint32_t v = sat_var(lit);
                if (v == 0u || v > nv) {
                    continue;
                }
                uint32_t a = var_activity[v];
                if (a != 0xFFFFFFFFu) {
                    var_activity[v] = a + 1u;
                }
                if (lit < 0) {
                    uint32_t n = level[v];
                    if (n != 0xFFFFFFFFu) {
                        level[v] = n + 1u;
                    }
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t v = static_cast<uint32_t>(threadIdx.x); v <= nv; v += static_cast<uint32_t>(blockDim.x)) {
        if (v == 0u) {
            level[v] = 0;
            var_phase[v] = SAT_VAL_FALSE;
            continue;
        }
        uint32_t total = var_activity[v];
        uint32_t neg = level[v];
        // pos >= neg  <=>  (pos + neg) >= 2*neg  <=>  total >= 2*neg
        var_phase[v] = (total >= (neg << 1)) ? SAT_VAL_TRUE : SAT_VAL_FALSE;
        level[v] = 0;
    }
    __syncthreads();

    // Equivalence-verifier specific: encourage branching on the ¬phi selector vars first by
    // forcing their saved phase to TRUE and giving them maximal initial activity.
    // (If the extra range is empty, this is a no-op.)
    if (decision_extra_end_excl > decision_extra_base_v) {
        for (uint32_t v = decision_extra_base_v + static_cast<uint32_t>(threadIdx.x);
             v < decision_extra_end_excl;
             v += static_cast<uint32_t>(blockDim.x)) {
            if (v >= 1u && v <= nv) {
                var_phase[v] = SAT_VAL_TRUE;
                var_activity[v] = 0xFFFFFFFFu;
            }
        }
    }
    __syncthreads();

    // Initialize trail.
    uint32_t trail_len = 0;
    uint32_t qhead = 0;
    uint32_t decision_level = 0;
    uint32_t assigned_count = 0;

    // trail_lim is indexed by decision level (0..). Clear deterministically.
    for (uint32_t i = static_cast<uint32_t>(threadIdx.x); i <= nv; i += static_cast<uint32_t>(blockDim.x)) {
        trail_lim[i] = 0;
    }
    __syncthreads();

    // Initialize decision heap with branchable variables.
    // Heap ordering is deterministic: higher activity first, ties by smaller var id.
    uint32_t decision_heap_size = 0;
    if (threadIdx.x == 0) {
        decision_heap_pos[0] = SAT_HEAP_NONE;
        for (uint32_t v = 1; v <= nv; v++) {
            if (sat_is_decision_var(v, decision_base, decision_extra_base_v, decision_extra_end_excl)) {
                decision_heap[decision_heap_size] = v;
                decision_heap_pos[v] = decision_heap_size;
                decision_heap_size++;
            } else {
                decision_heap_pos[v] = SAT_HEAP_NONE;
            }
        }
        // Heapify bottom-up (deterministic).
        for (int32_t i = (static_cast<int32_t>(decision_heap_size) >> 1) - 1; i >= 0; i--) {
            sat_heap_sift_down(
                static_cast<uint32_t>(i),
                decision_heap_size,
                decision_heap,
                decision_heap_pos,
                var_activity);
        }
    }
    __syncthreads();

    // Initialize learned arena.
    uint32_t learned_count = sh_imported_learned_count;
    uint32_t learned_lit_count = 0;
    uint32_t proof_len = 0;

    if (threadIdx.x == 0) {
        if (learned_count == 0u) {
            learned_offsets[0] = 0;
            proof_offsets[0] = 0;
        } else {
            learned_lit_count = learned_offsets[learned_count];
            proof_len = proof_offsets[learned_count];
        }
    }
    __syncthreads();
    for (uint32_t i = sh_imported_learned_count + static_cast<uint32_t>(threadIdx.x); i < max_learned_clauses; i += static_cast<uint32_t>(blockDim.x)) {
        learned_deleted[i] = 0;
        learned_lbd[i] = 0;
        learned_activity[i] = 0;
        learned_locked[i] = 0;
    }
    __syncthreads();

    // Initialize watch lists.
    for (uint32_t i = static_cast<uint32_t>(threadIdx.x); i < 2u * nv; i += static_cast<uint32_t>(blockDim.x)) {
        watch_head[i] = -1;
    }
    uint32_t max_total_clauses = nc + max_learned_clauses;
    for (uint32_t i = static_cast<uint32_t>(threadIdx.x); i < 2u * max_total_clauses; i += static_cast<uint32_t>(blockDim.x)) {
        watch_next[i] = -1;
        watch_prev[i] = -1;
    }
    __syncthreads();

    // Initialize base clause watches deterministically in clause_id order.
    if (threadIdx.x == 0) {
        for (uint32_t c = 0; c < nc; c++) {
            if (sh_done) {
                break;
            }

            uint32_t s = clause_offsets[c];
            uint32_t e = clause_offsets[c + 1u];
            uint32_t len = e - s;
            uint32_t w0 = 0;
            uint32_t w1 = (len > 1u) ? 1u : 0u;
            watch0_pos[c] = w0;
            watch1_pos[c] = w1;

            // Insert both watch nodes.
            int32_t lit0 = (len > 0u) ? clause_lits[s + w0] : 0;
            int32_t lit1 = (len > 0u) ? clause_lits[s + w1] : 0;

            if (len == 0u) {
                // Empty clause => UNSAT at level 0. Emit an empty learned clause with a trivial proof trace.
                if (learned_count >= max_learned_clauses) {
                    sh_done = 1u;
                    sh_final_status = SAT_STATUS_ERROR;
                    sh_final_error = SAT_ERR_LEARNED_CLAUSES_OVERFLOW;
                    sh_final_learned_count = learned_count;
                    break;
                }
                if (proof_len + 2u > max_proof_u32) {
                    sh_done = 1u;
                    sh_final_status = SAT_STATUS_ERROR;
                    sh_final_error = SAT_ERR_PROOF_OVERFLOW;
                    sh_final_learned_count = learned_count;
                    break;
                }
                uint32_t lid = learned_count;
                learned_offsets[lid] = learned_lit_count;
                learned_offsets[lid + 1u] = learned_lit_count;
                learned_deleted[lid] = 0;
                learned_lbd[lid] = 0;
                learned_activity[lid] = 0;
                learned_locked[lid] = 0;
                learned_count++;

                proof_offsets[lid] = proof_len;
                proof_data[proof_len++] = c;
                proof_data[proof_len++] = 0; // steps
                proof_offsets[lid + 1u] = proof_len;

                sh_done = 1u;
                sh_final_status = SAT_STATUS_UNSAT;
                sh_final_error = SAT_ERR_OK;
                sh_final_learned_count = learned_count;
                break;
            }

            uint32_t idx0 = sat_lit_index(lit0);
            uint32_t idx1 = sat_lit_index(lit1);
            int32_t node0 = static_cast<int32_t>(c * 2u);
            int32_t node1 = static_cast<int32_t>(c * 2u + 1u);
            sat_watch_insert_head(node0, watch_head, watch_next, watch_prev, idx0);
            sat_watch_insert_head(node1, watch_head, watch_next, watch_prev, idx1);

            // If unit clause, enqueue immediately at level 0.
            if (len == 1u) {
                if (!sat_enqueue(lit0, static_cast<int32_t>(c), 0, assign, level, reason, trail, &trail_len,
                                 &assigned_count, var_phase,
                                 decision_base,
                                 decision_extra_base_v,
                                 decision_extra_end_excl,
                                 decision_heap, decision_heap_pos, &decision_heap_size, var_activity)) {
                    // Contradictory unit clauses at level 0. Emit an empty learned clause + resolution trace.
                    uint32_t proof_steps = 0;
                    bool ok = sat_analyze_level0_unsat(
                        static_cast<int32_t>(c),
                        clause_offsets, clause_lits, nc,
                        learned_offsets, learned_lits, learned_deleted,
                        learned_count,
                        nv,
                        reason,
                        trail, trail_len,
                        seen,
                        proof_vars_tmp, proof_reason_tmp, &proof_steps);
                    if (!ok) {
                        sh_done = 1u;
                        sh_final_status = SAT_STATUS_ERROR;
                        sh_final_error = SAT_ERR_INVALID_PROOF;
                        sh_final_learned_count = learned_count;
                        break;
                    }

                    if (learned_count >= max_learned_clauses) {
                        sh_done = 1u;
                        sh_final_status = SAT_STATUS_ERROR;
                        sh_final_error = SAT_ERR_LEARNED_CLAUSES_OVERFLOW;
                        sh_final_learned_count = learned_count;
                        break;
                    }
                    if (proof_len + 2u + 2u * proof_steps > max_proof_u32) {
                        sh_done = 1u;
                        sh_final_status = SAT_STATUS_ERROR;
                        sh_final_error = SAT_ERR_PROOF_OVERFLOW;
                        sh_final_learned_count = learned_count;
                        break;
                    }
                    uint32_t lid = learned_count;
                    learned_offsets[lid] = learned_lit_count;
                    learned_offsets[lid + 1u] = learned_lit_count;
                    learned_deleted[lid] = 0;
                    learned_lbd[lid] = 0;
                    learned_activity[lid] = 0;
                    learned_locked[lid] = 0;
                    learned_count++;

                    proof_offsets[lid] = proof_len;
                    proof_data[proof_len++] = c;
                    proof_data[proof_len++] = proof_steps;
                    for (uint32_t i = 0; i < proof_steps; i++) {
                        proof_data[proof_len++] = proof_vars_tmp[i];
                        proof_data[proof_len++] = proof_reason_tmp[i];
                    }
                    proof_offsets[lid + 1u] = proof_len;

                    sh_done = 1u;
                    sh_final_status = SAT_STATUS_UNSAT;
                    sh_final_error = SAT_ERR_OK;
                    sh_final_learned_count = learned_count;
                    break;
                }
            }
        }
    }
    if (!sh_done && threadIdx.x == 0 && sh_imported_learned_count > 0u) {
        for (uint32_t lid = 0; lid < sh_imported_learned_count; lid++) {
            if (learned_deleted[lid] != 0u) {
                continue;
            }
            uint32_t c = nc + lid;
            uint32_t s = learned_offsets[lid];
            uint32_t e = learned_offsets[lid + 1u];
            uint32_t len = e - s;
            uint32_t w0 = 0;
            uint32_t w1 = (len > 1u) ? 1u : 0u;
            watch0_pos[c] = w0;
            watch1_pos[c] = w1;

            if (len == 0u) {
                sh_done = 1u;
                sh_final_status = SAT_STATUS_UNSAT;
                sh_final_error = SAT_ERR_OK;
                sh_final_learned_count = lid + 1u;
                break;
            }

            int32_t lit0 = learned_lits[s + w0];
            int32_t lit1 = learned_lits[s + w1];
            uint32_t idx0 = sat_lit_index(lit0);
            uint32_t idx1 = sat_lit_index(lit1);
            int32_t node0 = static_cast<int32_t>(c * 2u);
            int32_t node1 = static_cast<int32_t>(c * 2u + 1u);
            sat_watch_insert_head(node0, watch_head, watch_next, watch_prev, idx0);
            sat_watch_insert_head(node1, watch_head, watch_next, watch_prev, idx1);

            if (len == 1u) {
                if (!sat_enqueue(lit0, static_cast<int32_t>(c), 0, assign, level, reason, trail, &trail_len,
                                 &assigned_count, var_phase,
                                 decision_base,
                                 decision_extra_base_v,
                                 decision_extra_end_excl,
                                 decision_heap, decision_heap_pos, &decision_heap_size, var_activity)) {
                    sh_done = 1u;
                    sh_final_status = SAT_STATUS_ERROR;
                    sh_final_error = SAT_ERR_INVALID_PROOF;
                    sh_final_learned_count = learned_count;
                    break;
                }
            }
        }
    }
    __syncthreads();
    if (sh_done) {
        if (threadIdx.x == 0) {
            out_status[0] = sh_final_status;
            out_error[0] = sh_final_error;
            out_learned_count[0] = sh_final_learned_count;
        }
        return;
    }
    __syncthreads();
    if (sh_done) {
        if (threadIdx.x == 0) {
            out_status[0] = sh_final_status;
            out_error[0] = sh_final_error;
            out_learned_count[0] = sh_final_learned_count;
        }
        return;
    }

    // Initial propagation.
    int32_t confl = sat_propagate(
        clause_offsets, clause_lits, nv, nc,
        learned_offsets, learned_lits, learned_deleted, learned_activity,
        watch0_pos, watch1_pos,
        watch_head, watch_next, watch_prev,
        &qhead, decision_level,
        assign, level, reason,
        trail, &trail_len, &assigned_count,
        var_phase,
        decision_base,
        decision_extra_base_v,
        decision_extra_end_excl,
        &decision_heap_size, decision_heap, decision_heap_pos, var_activity);
    if (threadIdx.x == 0) {
        if (confl >= 0) {
            // Conflict at level 0 => UNSAT. Emit an empty learned clause + resolution trace certificate.
            uint32_t proof_steps = 0;
            bool ok = sat_analyze_level0_unsat(
                confl,
                clause_offsets, clause_lits, nc,
                learned_offsets, learned_lits, learned_deleted,
                learned_count,
                nv,
                reason,
                trail, trail_len,
                seen,
                proof_vars_tmp, proof_reason_tmp, &proof_steps);
            if (!ok) {
                sh_done = 1u;
                sh_final_status = SAT_STATUS_ERROR;
                sh_final_error = SAT_ERR_INVALID_PROOF;
                sh_final_learned_count = learned_count;
            } else if (learned_count >= max_learned_clauses) {
                sh_done = 1u;
                sh_final_status = SAT_STATUS_ERROR;
                sh_final_error = SAT_ERR_LEARNED_CLAUSES_OVERFLOW;
                sh_final_learned_count = learned_count;
            } else if (proof_len + 2u + 2u * proof_steps > max_proof_u32) {
                sh_done = 1u;
                sh_final_status = SAT_STATUS_ERROR;
                sh_final_error = SAT_ERR_PROOF_OVERFLOW;
                sh_final_learned_count = learned_count;
            } else {
                uint32_t lid = learned_count;
                learned_offsets[lid] = learned_lit_count;
                learned_offsets[lid + 1u] = learned_lit_count; // empty clause
                learned_deleted[lid] = 0;
                learned_lbd[lid] = 0;
                learned_activity[lid] = 0;
                learned_locked[lid] = 0;
                learned_count++;

                proof_offsets[lid] = proof_len;
                proof_data[proof_len++] = static_cast<uint32_t>(confl);
                proof_data[proof_len++] = proof_steps;
                for (uint32_t i = 0; i < proof_steps; i++) {
                    proof_data[proof_len++] = proof_vars_tmp[i];
                    proof_data[proof_len++] = proof_reason_tmp[i];
                }
                proof_offsets[lid + 1u] = proof_len;

                sh_done = 1u;
                sh_final_status = SAT_STATUS_UNSAT;
                sh_final_error = SAT_ERR_OK;
                sh_final_learned_count = learned_count;
            }
        }
    }
    __syncthreads();
    if (sh_done) {
        if (threadIdx.x == 0) {
            out_status[0] = sh_final_status;
            out_error[0] = sh_final_error;
            out_learned_count[0] = sh_final_learned_count;
        }
        return;
    }

    // CDCL loop.
    uint32_t conflicts = 0;
    uint32_t next_reduce = reduce_interval;
    uint32_t restart_iter = 1;
    uint32_t next_restart = restart_base * sat_luby(restart_iter);
    // Deterministic variable activity bump. Keep the bump constant to avoid frequent global
    // rescale + heap rebuild costs on large instances; saturation is handled in sat_bump_var.
    uint32_t var_bump = 1u;

    for (;;) {
        // Propagate (block-parallel).
        confl = sat_propagate(
            clause_offsets, clause_lits, nv, nc,
            learned_offsets, learned_lits, learned_deleted, learned_activity,
            watch0_pos, watch1_pos,
            watch_head, watch_next, watch_prev,
            &qhead, decision_level,
            assign, level, reason,
            trail, &trail_len, &assigned_count,
            var_phase,
            decision_base,
            decision_extra_base_v,
            decision_extra_end_excl,
            &decision_heap_size, decision_heap, decision_heap_pos, var_activity);

        if (threadIdx.x == 0) {
            if (confl >= 0) {
                // Conflict.
                conflicts++;
                if (decision_level == 0) {
                    // UNSAT. Emit an empty learned clause + resolution trace certificate.
                    uint32_t proof_steps = 0;
                    bool ok = sat_analyze_level0_unsat(
                        confl,
                        clause_offsets, clause_lits, nc,
                        learned_offsets, learned_lits, learned_deleted,
                        learned_count,
                        nv,
                        reason,
                        trail, trail_len,
                        seen,
                        proof_vars_tmp, proof_reason_tmp, &proof_steps);
                    if (!ok) {
                        sh_done = 1u;
                        sh_final_status = SAT_STATUS_ERROR;
                        sh_final_error = SAT_ERR_INVALID_PROOF;
                        sh_final_learned_count = learned_count;
                        goto iter_done;
                    }

                    if (learned_count >= max_learned_clauses) {
                        sh_done = 1u;
                        sh_final_status = SAT_STATUS_ERROR;
                        sh_final_error = SAT_ERR_LEARNED_CLAUSES_OVERFLOW;
                        sh_final_learned_count = learned_count;
                        goto iter_done;
                    }
                    if (proof_len + 2u + 2u * proof_steps > max_proof_u32) {
                        sh_done = 1u;
                        sh_final_status = SAT_STATUS_ERROR;
                        sh_final_error = SAT_ERR_PROOF_OVERFLOW;
                        sh_final_learned_count = learned_count;
                        goto iter_done;
                    }
                    uint32_t lid = learned_count;
                    learned_offsets[lid] = learned_lit_count;
                    learned_offsets[lid + 1u] = learned_lit_count; // empty clause
                    learned_deleted[lid] = 0;
                    learned_lbd[lid] = 0;
                    learned_activity[lid] = 0;
                    learned_locked[lid] = 0;
                    learned_count++;

                    proof_offsets[lid] = proof_len;
                    proof_data[proof_len++] = static_cast<uint32_t>(confl);
                    proof_data[proof_len++] = proof_steps;
                    for (uint32_t i = 0; i < proof_steps; i++) {
                        proof_data[proof_len++] = proof_vars_tmp[i];
                        proof_data[proof_len++] = proof_reason_tmp[i];
                    }
                    proof_offsets[lid + 1u] = proof_len;

                    sh_done = 1u;
                    sh_final_status = SAT_STATUS_UNSAT;
                    sh_final_error = SAT_ERR_OK;
                    sh_final_learned_count = learned_count;
                    goto iter_done;
                }

                // Analyze.
                uint32_t learnt_len = 0;
                uint32_t bt = 0;
                uint32_t proof_steps = 0;
                bool ok = sat_analyze(
                    confl,
                    clause_offsets, clause_lits, nc,
                    learned_offsets, learned_lits, learned_deleted,
                    learned_count,
                    decision_level,
                    assign, level, reason,
                    trail, trail_len,
                    seen, learnt_tmp,
                    &learnt_len, &bt,
                    proof_vars_tmp, proof_reason_tmp, &proof_steps);
                if (!ok) {
                    sh_done = 1u;
                    sh_final_status = SAT_STATUS_ERROR;
                    sh_final_error = SAT_ERR_INVALID_PROOF;
                    sh_final_learned_count = learned_count;
                    goto iter_done;
                }

                // Bump learned clause activity for clauses that participate in this conflict analysis.
                if (confl >= static_cast<int32_t>(nc)) {
                    uint32_t lid = static_cast<uint32_t>(confl) - nc;
                    if (lid < learned_count) {
                        learned_activity[lid] += 1u;
                    }
                }
                for (uint32_t i = 0; i < proof_steps; i++) {
                    uint32_t rc = proof_reason_tmp[i];
                    if (rc >= nc) {
                        uint32_t lid = rc - nc;
                        if (lid < learned_count) {
                            learned_activity[lid] += 1u;
                        }
                    }
                }

                // Bump variables in learned clause (deterministic).
                for (uint32_t i = 0; i < learnt_len; i++) {
                    uint32_t v = sat_var(learnt_tmp[i]);
                    (void)sat_bump_var(v, var_activity, var_bump);
                    // If v is currently unassigned (present in the decision heap), restore heap order.
                    uint32_t pos = decision_heap_pos[v];
                    if (pos != SAT_HEAP_NONE && pos < decision_heap_size) {
                        sat_heap_sift_up(pos, decision_heap, decision_heap_pos, var_activity);
                    }
                }

                // Backjump.
                sat_backtrack(bt, &decision_level, nv + 1u, trail_lim, &qhead,
                              trail, &trail_len, &assigned_count, assign, level, reason,
                              decision_base,
                              decision_extra_base_v,
                              decision_extra_end_excl,
                              &decision_heap_size, decision_heap, decision_heap_pos, var_activity);

                // Learn clause (append to arena).
                if (learned_count >= max_learned_clauses) {
                    sh_done = 1u;
                    sh_final_status = SAT_STATUS_ERROR;
                    sh_final_error = SAT_ERR_LEARNED_CLAUSES_OVERFLOW;
                    sh_final_learned_count = learned_count;
                    goto iter_done;
                }
                uint32_t lid = learned_count;
                uint32_t clause_id = nc + lid;
                learned_offsets[lid] = learned_lit_count;
                if (learned_lit_count + learnt_len > max_learned_lits) {
                    sh_done = 1u;
                    sh_final_status = SAT_STATUS_ERROR;
                    sh_final_error = SAT_ERR_LEARNED_LITS_OVERFLOW;
                    sh_final_learned_count = learned_count;
                    goto iter_done;
                }
                for (uint32_t i = 0; i < learnt_len; i++) {
                    learned_lits[learned_lit_count + i] = learnt_tmp[i];
                }
                learned_lit_count += learnt_len;
                learned_offsets[lid + 1u] = learned_lit_count;
                learned_deleted[lid] = 0;
                learned_lbd[lid] = sat_compute_lbd(learnt_tmp, learnt_len, level);
                learned_activity[lid] = 0;
                learned_locked[lid] = 0;
                learned_count++;

                // Record proof trace for this learned clause.
                if (proof_len + 2u + 2u * proof_steps > max_proof_u32) {
                    sh_done = 1u;
                    sh_final_status = SAT_STATUS_ERROR;
                    sh_final_error = SAT_ERR_PROOF_OVERFLOW;
                    sh_final_learned_count = learned_count;
                    goto iter_done;
                }
                proof_offsets[lid] = proof_len;
                proof_data[proof_len++] = static_cast<uint32_t>(confl);
                proof_data[proof_len++] = proof_steps;
                for (uint32_t i = 0; i < proof_steps; i++) {
                    proof_data[proof_len++] = proof_vars_tmp[i];
                    proof_data[proof_len++] = proof_reason_tmp[i];
                }
                proof_offsets[lid + 1u] = proof_len;

                // Initialize watches for learned clause.
                uint32_t w0 = 0;
                uint32_t w1 = (learnt_len > 1u) ? 1u : 0u;
                watch0_pos[clause_id] = w0;
                watch1_pos[clause_id] = w1;
                // Insert watch nodes for learned clause.
                int32_t lit0 = learnt_tmp[w0];
                int32_t lit1 = learnt_tmp[w1];
                uint32_t idx0 = sat_lit_index(lit0);
                uint32_t idx1 = sat_lit_index(lit1);
                int32_t node0 = static_cast<int32_t>(clause_id * 2u);
                int32_t node1 = static_cast<int32_t>(clause_id * 2u + 1u);
                sat_watch_insert_head(node0, watch_head, watch_next, watch_prev, idx0);
                sat_watch_insert_head(node1, watch_head, watch_next, watch_prev, idx1);

                // Enqueue asserting literal with reason = clause_id.
                (void)sat_enqueue(learnt_tmp[0], static_cast<int32_t>(clause_id), decision_level, assign, level, reason,
                                 trail, &trail_len, &assigned_count, var_phase,
                                 decision_base,
                                 decision_extra_base_v,
                                 decision_extra_end_excl,
                                 decision_heap, decision_heap_pos, &decision_heap_size, var_activity);
            } else {
                // No conflict. Check SAT.
                if (assigned_count == nv) {
                    sh_done = 1u;
                    sh_final_status = SAT_STATUS_SAT;
                    sh_final_error = SAT_ERR_OK;
                    sh_final_learned_count = learned_count;
                    goto iter_done;
                }

                // Restart (deterministic schedule).
                if (conflicts >= next_restart && decision_level > 0) {
                    sat_backtrack(0, &decision_level, nv + 1u, trail_lim, &qhead,
                                  trail, &trail_len, &assigned_count, assign, level, reason,
                                  decision_base,
                                  decision_extra_base_v,
                                  decision_extra_end_excl,
                                  &decision_heap_size, decision_heap, decision_heap_pos, var_activity);
                    restart_iter++;
                    next_restart = restart_base * sat_luby(restart_iter);
                }

                // Periodic clause DB management (deterministic heuristic).
                if (reduce_interval != 0 && conflicts >= next_reduce && learned_count > 0) {
                    // Simple reduction: delete half of learned clauses with LBD > 2 and low activity, excluding locked.
                    // Compute activity threshold among deletable clauses.
                    uint64_t sum = 0;
                    uint32_t cnt = 0;
                    for (uint32_t i = 0; i < learned_count; i++) {
                        if (learned_deleted[i] != 0) {
                            continue;
                        }
                        if (learned_lbd[i] <= 2u) {
                            continue;
                        }
                        sum += static_cast<uint64_t>(learned_activity[i]);
                        cnt++;
                    }
                    uint32_t thr = (cnt > 0) ? static_cast<uint32_t>(sum / cnt) : 0;

                    // Mark locked learned clauses (used as a reason for any assignment).
                    for (uint32_t i = 0; i < learned_count; i++) {
                        learned_locked[i] = 0;
                    }
                    for (uint32_t v = 1; v <= nv; v++) {
                        int32_t r = reason[v];
                        if (r >= static_cast<int32_t>(nc)) {
                            uint32_t lid = static_cast<uint32_t>(r) - nc;
                            if (lid < learned_count) {
                                learned_locked[lid] = 1;
                            }
                        }
                    }

                    uint32_t deleted = 0;
                    for (uint32_t i = 0; i < learned_count; i++) {
                        if (learned_deleted[i] != 0) {
                            continue;
                        }
                        if (learned_lbd[i] <= 2u) {
                            continue;
                        }
                        if (learned_locked[i] != 0) {
                            continue;
                        }
                        if (learned_activity[i] < thr) {
                            learned_deleted[i] = 1;
                            // Remove watch nodes for this learned clause.
                            uint32_t clause_id = nc + i;
                            int32_t node0 = static_cast<int32_t>(clause_id * 2u);
                            int32_t node1 = static_cast<int32_t>(clause_id * 2u + 1u);
                            // Remove using current watch positions to find literal indices.
                            SatClauseView cv = sat_get_clause(static_cast<int32_t>(clause_id), clause_offsets, clause_lits, nc,
                                                              learned_offsets, learned_lits);
                            if (cv.len > 0) {
                                int32_t l0 = cv.lits[watch0_pos[clause_id]];
                                int32_t l1 = cv.lits[watch1_pos[clause_id]];
                                sat_watch_remove(node0, watch_head, watch_next, watch_prev, sat_lit_index(l0));
                                sat_watch_remove(node1, watch_head, watch_next, watch_prev, sat_lit_index(l1));
                            }
                            deleted++;
                        }
                    }
                    (void)deleted;
                    next_reduce += reduce_interval;
                }

                // Decide next branch from the decision heap (unassigned vars only).
                uint32_t next_v = 0;
                while (decision_heap_size > 0) {
                    uint32_t v = decision_heap[0];
                    if (v >= 1u && v <= nv && assign[v] == SAT_VAL_UNASSIGNED) {
                        next_v = v;
                        break;
                    }
                    // Defensive: drop stale heap entries deterministically.
                    sat_heap_remove(v, &decision_heap_size, decision_heap, decision_heap_pos, var_activity);
                }
                if (next_v == 0) {
                    // No eligible decision variables remain, but we have not proven SAT/UNSAT.
                    // In verifier mode this indicates a bug in the encoding or decision set.
                    sh_done = 1u;
                    sh_final_status = SAT_STATUS_ERROR;
                    sh_final_error = SAT_ERR_DECISION_EXHAUSTED;
                    sh_final_learned_count = learned_count;
                    goto iter_done;
                }
                decision_level++;
                trail_lim[decision_level] = trail_len;
                qhead = trail_len;

                int8_t ph = var_phase[next_v];
                int32_t decision_lit = (ph == SAT_VAL_TRUE) ? static_cast<int32_t>(next_v) : -static_cast<int32_t>(next_v);
                if (!sat_enqueue(decision_lit, SAT_REASON_NONE, decision_level, assign, level, reason, trail, &trail_len,
                                 &assigned_count, var_phase,
                                 decision_base,
                                 decision_extra_base_v,
                                 decision_extra_end_excl,
                                 decision_heap, decision_heap_pos, &decision_heap_size, var_activity)) {
                    // Should never conflict on decision enqueue; treat as error.
                    sh_done = 1u;
                    sh_final_status = SAT_STATUS_ERROR;
                    sh_final_error = SAT_ERR_INVALID_MODEL;
                    sh_final_learned_count = learned_count;
                    goto iter_done;
                }
            }

iter_done:
            ;
        }

        __syncthreads();
        if (sh_done) {
            break;
        }
    }

    if (threadIdx.x == 0) {
        out_status[0] = sh_final_status;
        out_error[0] = sh_final_error;
        out_learned_count[0] = sh_final_learned_count;
    }
    return;
}

// ---------------------------------------------------------------------------
// Kernel: sat_check_model (CNF satisfaction check)
// ---------------------------------------------------------------------------

extern "C" __global__ void sat_check_model(
    const uint32_t* __restrict__ compile_needed,
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ clause_lits,
    const uint32_t* __restrict__ num_clauses, // len=1
    const int8_t* __restrict__ assign,
    int32_t* __restrict__ out_ok // 1 ok, 0 fail
) {
    if (compile_needed[0] == 0u) {
        return;
    }
    uint32_t nc = num_clauses[0];
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        out_ok[0] = 1;
    }
    __syncthreads();

    for (uint32_t c = tid; c < nc; c += blockDim.x * gridDim.x) {
        uint32_t s = clause_offsets[c];
        uint32_t e = clause_offsets[c + 1u];
        bool sat = false;
        for (uint32_t i = s; i < e; i++) {
            int32_t lit = clause_lits[i];
            if (sat_eval_lit(lit, assign) == SAT_VAL_TRUE) {
                sat = true;
                break;
            }
        }
        if (!sat) {
            atomicExch(out_ok, 0);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: sat_proof_check (resolution-trace checker for learned clauses)
// ---------------------------------------------------------------------------

// Resolve two clauses on variable v. Writes output literals into out_buf, returns out_len, or sets *ok=0 on failure.
__device__ __forceinline__ uint32_t sat_resolve_on_var(
    uint32_t v,
    const int32_t* __restrict__ a_lits,
    uint32_t a_len,
    const int32_t* __restrict__ b_lits,
    uint32_t b_len,
    int32_t* __restrict__ out_buf,
    uint32_t out_cap,
    int32_t* __restrict__ ok
) {
    // Determine required complementary literals exist.
    bool a_pos = false, a_neg = false, b_pos = false, b_neg = false;
    for (uint32_t i = 0; i < a_len; i++) {
        uint32_t av = sat_var(a_lits[i]);
        if (av != v) continue;
        if (a_lits[i] > 0) a_pos = true;
        else a_neg = true;
    }
    for (uint32_t i = 0; i < b_len; i++) {
        uint32_t bv = sat_var(b_lits[i]);
        if (bv != v) continue;
        if (b_lits[i] > 0) b_pos = true;
        else b_neg = true;
    }
    if (!((a_pos && b_neg) || (a_neg && b_pos))) {
        *ok = 0;
        return 0;
    }

    uint32_t out_len = 0;

    // Copy A excluding v literals.
    for (uint32_t i = 0; i < a_len; i++) {
        int32_t lit = a_lits[i];
        if (sat_var(lit) == v) continue;
        if (out_len >= out_cap) {
            *ok = 0;
            return 0;
        }
        out_buf[out_len++] = lit;
    }

    // Merge B excluding v literals with duplicate/tautology checks.
    for (uint32_t i = 0; i < b_len; i++) {
        int32_t lit = b_lits[i];
        uint32_t lv = sat_var(lit);
        if (lv == v) continue;

        // Check for tautology (both x and -x).
        for (uint32_t j = 0; j < out_len; j++) {
            if (sat_var(out_buf[j]) == lv) {
                if (out_buf[j] == lit) {
                    // duplicate
                    lv = 0;
                    break;
                }
                // opposite sign => tautology, invalid in this proof model
                *ok = 0;
                return 0;
            }
        }
        if (lv == 0) {
            continue;
        }
        if (out_len >= out_cap) {
            *ok = 0;
            return 0;
        }
        out_buf[out_len++] = lit;
    }

    return out_len;
}

__device__ __forceinline__ bool sat_clause_equal_set(
    const int32_t* __restrict__ a,
    uint32_t a_len,
    const int32_t* __restrict__ b,
    uint32_t b_len
) {
    if (a_len != b_len) {
        return false;
    }
    // O(n^2) set equality.
    for (uint32_t i = 0; i < a_len; i++) {
        bool found = false;
        for (uint32_t j = 0; j < b_len; j++) {
            if (a[i] == b[j]) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

// Compute the dependency closure of the final (empty) learned clause proof.
//
// Marks `needed[i]=1` for learned clauses that appear in the final empty-clause derivation chain
// (as either a conflict clause or a reason clause), transitively. This is a single-pass reverse scan
// because each proof trace references only earlier clauses (`reason_clause < nc + i`).
extern "C" __global__ void sat_proof_mark_needed(
    const uint32_t* __restrict__ compile_needed,
    const uint32_t* __restrict__ num_clauses,   // len=1
    const uint32_t* __restrict__ learned_count, // len=1
    const uint32_t* __restrict__ proof_offsets,
    const uint32_t* __restrict__ proof_data,
    uint32_t needed_cap,
    uint8_t* __restrict__ needed // len >= needed_cap
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (compile_needed[0] == 0u) {
        return;
    }

    uint32_t nc = num_clauses[0];
    uint32_t lc = learned_count[0];
    if (lc == 0u) {
        return;
    }
    if (lc > needed_cap) {
        sat_trap();
    }

    // Seed with the final learned clause (the UNSAT certificate).
    needed[lc - 1u] = 1u;

    // Reverse scan: whenever clause i is needed, mark any referenced learned clauses as needed.
    for (int64_t ii = static_cast<int64_t>(lc) - 1; ii >= 0; ii--) {
        uint32_t i = static_cast<uint32_t>(ii);
        if (needed[i] == 0u) {
            continue;
        }

        uint32_t poff = proof_offsets[i];
        uint32_t pend = proof_offsets[i + 1u];
        if (pend < poff + 2u) {
            sat_trap();
        }
        uint32_t conflict_clause = proof_data[poff + 0u];
        uint32_t steps = proof_data[poff + 1u];
        uint32_t expect = 2u + 2u * steps;
        if (poff + expect != pend) {
            sat_trap();
        }

        // Mark learned dependencies (clause ids >= nc).
        auto mark_clause = [&](uint32_t clause_id) {
            if (clause_id < nc) {
                return;
            }
            uint32_t lid = clause_id - nc;
            if (lid >= lc) {
                sat_trap();
            }
            needed[lid] = 1u;
        };

        mark_clause(conflict_clause);
        for (uint32_t s = 0; s < steps; s++) {
            uint32_t reason_clause = proof_data[poff + 2u + 2u * s + 1u];
            if (reason_clause == 0xFFFFFFFFu) {
                sat_trap();
            }
            mark_clause(reason_clause);
        }
    }
}

__device__ __forceinline__ void sat_mark_clear_clause(
    const int32_t* __restrict__ lits,
    uint32_t len,
    uint8_t* __restrict__ mark
) {
    for (uint32_t i = 0; i < len; i++) {
        uint32_t v = sat_var(lits[i]);
        if (v != 0u) {
            mark[v] = 0u;
        }
    }
}

// Resolve two clauses on variable v, using a per-block mark array to avoid O(n^2) duplicate scans.
//
// Mark encoding:
// - 0: absent
// - 1: positive literal present
// - 2: negative literal present
__device__ __forceinline__ uint32_t sat_resolve_on_var_mark(
    uint32_t v,
    const int32_t* __restrict__ a_lits,
    uint32_t a_len,
    const int32_t* __restrict__ b_lits,
    uint32_t b_len,
    int32_t* __restrict__ out_buf,
    uint32_t out_cap,
    uint8_t* __restrict__ mark,
    int32_t* __restrict__ ok
) {
    bool a_pos = false, a_neg = false, b_pos = false, b_neg = false;
    uint32_t out_len = 0;

    for (uint32_t i = 0; i < a_len; i++) {
        int32_t lit = a_lits[i];
        uint32_t lv = sat_var(lit);
        if (lv == v) {
            if (lit > 0) a_pos = true;
            else a_neg = true;
            continue;
        }
        if (lv == 0u) {
            continue;
        }
        uint8_t sign = (lit > 0) ? 1u : 2u;
        uint8_t cur = mark[lv];
        if (cur == 0u) {
            if (out_len >= out_cap) {
                *ok = 0;
                return 0;
            }
            mark[lv] = sign;
            out_buf[out_len++] = lit;
        } else if (cur != sign) {
            // Tautology detected.
            *ok = 0;
            return 0;
        }
    }

    for (uint32_t i = 0; i < b_len; i++) {
        int32_t lit = b_lits[i];
        uint32_t lv = sat_var(lit);
        if (lv == v) {
            if (lit > 0) b_pos = true;
            else b_neg = true;
            continue;
        }
        if (lv == 0u) {
            continue;
        }
        uint8_t sign = (lit > 0) ? 1u : 2u;
        uint8_t cur = mark[lv];
        if (cur == 0u) {
            if (out_len >= out_cap) {
                *ok = 0;
                return 0;
            }
            mark[lv] = sign;
            out_buf[out_len++] = lit;
        } else if (cur != sign) {
            *ok = 0;
            return 0;
        }
    }

    // Determine required complementary literals exist.
    if (!((a_pos && b_neg) || (a_neg && b_pos))) {
        *ok = 0;
        return 0;
    }

    return out_len;
}

extern "C" __global__ void sat_proof_check(
    const uint32_t* __restrict__ compile_needed,
    const uint32_t* __restrict__ base_offsets,
    const int32_t* __restrict__ base_lits,
    const uint32_t* __restrict__ num_clauses, // len=1
    const uint32_t* __restrict__ learned_offsets,
    const int32_t* __restrict__ learned_lits,
    const uint32_t* __restrict__ learned_count, // len=1
    const uint32_t* __restrict__ proof_offsets,
    const uint32_t* __restrict__ proof_data,
    const uint8_t* __restrict__ needed, // len = needed_cap
    uint32_t needed_cap,
    // scratch buffers: two clause literal buffers + per-var stamp/sign map
    int32_t* __restrict__ scratch_a,
    int32_t* __restrict__ scratch_b,
    uint32_t* __restrict__ scratch_map,
    uint32_t scratch_cap,
    int32_t* __restrict__ out_ok // 1 ok, 0 fail
) {
    if (compile_needed[0] == 0u) {
        return;
    }

    uint32_t nc = num_clauses[0];

    uint32_t lc = learned_count[0];
    // Only thread 0 performs the global UNSAT-certificate sanity checks.
    if (threadIdx.x == 0) {
        if (lc == 0u || lc > needed_cap) {
            atomicExch(out_ok, 0);
        } else {
            // Require last learned clause to be empty (unsat certificate).
            uint32_t last_len = learned_offsets[lc] - learned_offsets[lc - 1u];
            if (last_len != 0u) {
                atomicExch(out_ok, 0);
            }
        }
    }
    __syncthreads();
    if (out_ok[0] == 0) {
        return;
    }

    // Each block uses a disjoint scratch region:
    //   scratch_{a,b,map}[blockIdx.x * scratch_cap .. (blockIdx.x+1) * scratch_cap)
    uint64_t scratch_off =
        static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(scratch_cap);
    int32_t* buf0 = scratch_a + scratch_off;
    int32_t* buf1 = scratch_b + scratch_off;
    uint32_t* map = scratch_map + scratch_off;

    // Shared state for a single learned clause proof replay.
    __shared__ uint32_t stamp;
    __shared__ uint32_t cur_len;
    __shared__ unsigned int out_len;
    __shared__ unsigned int pivot_flags;
    __shared__ int ping;      // 0 => buf0 is current, 1 => buf1 is current
    __shared__ int block_fail;
    __shared__ int stop;
    __shared__ uint32_t poff;
    __shared__ uint32_t pend;
    __shared__ uint32_t conflict_clause;
    __shared__ uint32_t steps;
    __shared__ uint32_t pivot_var;
    __shared__ const int32_t* clause_lits;
    __shared__ uint32_t clause_len;

    if (threadIdx.x == 0) {
        stamp = 1u;
    }
    __syncthreads();

    const uint32_t STAMP_MAX = 0x3FFFFFFFu; // 30-bit stamp (packed <<2)

    auto mark_fail = [&]() {
        atomicExch(&block_fail, 1);
    };

    auto map_insert_lit = [&](uint32_t v, uint32_t sign, int32_t lit, int32_t* out_buf) {
        // `map[v] = (stamp<<2) | sign`, with sign in {1=pos,2=neg}.
        // Absent iff (map[v]>>2) != stamp.
        uint32_t new_val = (stamp << 2) | sign;
        for (;;) {
            uint32_t old = map[v];
            if ((old >> 2) == stamp) {
                if ((old & 3u) != sign) {
                    mark_fail();
                }
                return;
            }
            uint32_t prev = atomicCAS(map + v, old, new_val);
            if (prev == old) {
                unsigned int idx = atomicAdd(&out_len, 1u);
                if (idx >= scratch_cap) {
                    mark_fail();
                    return;
                }
                out_buf[idx] = lit;
                return;
            }
        }
    };

    for (uint32_t i = blockIdx.x; i < lc; i += gridDim.x) {
        // Cooperative stop: once any verifier fails it will set out_ok=0.
        if (threadIdx.x == 0) {
            stop = (out_ok[0] == 0);
        }
        __syncthreads();
        if (stop) {
            return;
        }
        if (needed[i] == 0u) {
            continue;
        }

        // Parse proof header for learned clause i.
        if (threadIdx.x == 0) {
            block_fail = 0;
            poff = proof_offsets[i];
            pend = proof_offsets[i + 1u];
            if (pend < poff + 2u) {
                block_fail = 1;
            } else {
                conflict_clause = proof_data[poff + 0u];
                steps = proof_data[poff + 1u];
                uint32_t expect = 2u + 2u * steps;
                if (poff + expect != pend) {
                    block_fail = 1;
                } else if (conflict_clause >= nc + i) {
                    block_fail = 1;
                }
            }
            // Start each learned-clause replay with buf0 as the current buffer.
            ping = 0;
        }
        __syncthreads();
        if (block_fail) {
            if (threadIdx.x == 0) {
                atomicExch(out_ok, 0);
            }
            __syncthreads();
            return;
        }

        // Load and normalize the initial conflict clause into buf0 under a fresh stamp.
        if (threadIdx.x == 0) {
            if (stamp >= STAMP_MAX - 2u) {
                block_fail = 1;
            } else {
                stamp++;
                out_len = 0;
                SatClauseView c0 = sat_get_clause(static_cast<int32_t>(conflict_clause),
                                                  base_offsets, base_lits, nc,
                                                  learned_offsets, learned_lits);
                clause_lits = c0.lits;
                clause_len = c0.len;
                if (clause_len > scratch_cap) {
                    block_fail = 1;
                }
            }
        }
        __syncthreads();
        if (block_fail) {
            if (threadIdx.x == 0) {
                atomicExch(out_ok, 0);
            }
            __syncthreads();
            return;
        }

        for (uint32_t k = threadIdx.x; k < clause_len; k += blockDim.x) {
            int32_t lit = clause_lits[k];
            uint32_t v = sat_var(lit);
            if (v == 0u || v >= scratch_cap) {
                mark_fail();
                continue;
            }
            uint32_t sign = (lit > 0) ? 1u : 2u;
            map_insert_lit(v, sign, lit, buf0);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            cur_len = out_len;
            if (block_fail) {
                atomicExch(out_ok, 0);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            stop = (out_ok[0] == 0);
        }
        __syncthreads();
        if (stop) {
            return;
        }

        // Replay each resolution step, ping-ponging between buf0 and buf1.
        for (uint32_t s = 0; s < steps; s++) {
            if (threadIdx.x == 0) {
                pivot_flags = 0;
                out_len = 0;
                if (stamp >= STAMP_MAX - 2u) {
                    block_fail = 1;
                } else {
                    stamp++;
                    pivot_var = proof_data[poff + 2u + 2u * s];
                    uint32_t reason_clause = proof_data[poff + 2u + 2u * s + 1u];
                    if (reason_clause == 0xFFFFFFFFu) {
                        block_fail = 1;
                    } else if (reason_clause >= nc + i) {
                        block_fail = 1;
                    } else {
                        SatClauseView rc = sat_get_clause(static_cast<int32_t>(reason_clause),
                                                          base_offsets, base_lits, nc,
                                                          learned_offsets, learned_lits);
                        clause_lits = rc.lits;
                        clause_len = rc.len;
                        if (clause_len > scratch_cap) {
                            block_fail = 1;
                        }
                    }
                }
            }
            __syncthreads();
            if (block_fail) {
                if (threadIdx.x == 0) {
                    atomicExch(out_ok, 0);
                }
                __syncthreads();
                return;
            }

            int32_t* cur_buf = (ping == 0) ? buf0 : buf1;
            int32_t* next_buf = (ping == 0) ? buf1 : buf0;

            // Merge current clause excluding pivot literals.
            for (uint32_t t = threadIdx.x; t < cur_len; t += blockDim.x) {
                int32_t lit = cur_buf[t];
                uint32_t v = sat_var(lit);
                if (v == pivot_var) {
                    atomicOr(&pivot_flags, (lit > 0) ? 1u : 2u);
                    continue;
                }
                if (v == 0u || v >= scratch_cap) {
                    mark_fail();
                    continue;
                }
                uint32_t sign = (lit > 0) ? 1u : 2u;
                map_insert_lit(v, sign, lit, next_buf);
            }

            // Merge reason clause excluding pivot literals.
            for (uint32_t t = threadIdx.x; t < clause_len; t += blockDim.x) {
                int32_t lit = clause_lits[t];
                uint32_t v = sat_var(lit);
                if (v == pivot_var) {
                    atomicOr(&pivot_flags, (lit > 0) ? 4u : 8u);
                    continue;
                }
                if (v == 0u || v >= scratch_cap) {
                    mark_fail();
                    continue;
                }
                uint32_t sign = (lit > 0) ? 1u : 2u;
                map_insert_lit(v, sign, lit, next_buf);
            }

            __syncthreads();
            if (threadIdx.x == 0) {
                // Determine required complementary pivot literals exist.
                bool ok =
                    ((pivot_flags & 1u) != 0u && (pivot_flags & 8u) != 0u) ||
                    ((pivot_flags & 2u) != 0u && (pivot_flags & 4u) != 0u);
                if (!ok) {
                    block_fail = 1;
                }
                cur_len = out_len;
                ping ^= 1;
                if (block_fail) {
                    atomicExch(out_ok, 0);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                stop = (out_ok[0] == 0);
            }
            __syncthreads();
            if (stop) {
                return;
            }
        }

        // Compare with stored learned clause i (set equality via stamp map).
        if (threadIdx.x == 0) {
            clause_lits = learned_lits + learned_offsets[i];
            clause_len = learned_offsets[i + 1u] - learned_offsets[i];
            if (clause_len != cur_len) {
                atomicExch(out_ok, 0);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            stop = (out_ok[0] == 0);
        }
        __syncthreads();
        if (stop) {
            return;
        }
        for (uint32_t k = threadIdx.x; k < clause_len; k += blockDim.x) {
            int32_t lit = clause_lits[k];
            uint32_t v = sat_var(lit);
            if (v == 0u || v >= scratch_cap) {
                mark_fail();
                continue;
            }
            uint32_t expected = (lit > 0) ? 1u : 2u;
            uint32_t st = map[v];
            if ((st >> 2) != stamp || (st & 3u) != expected) {
                mark_fail();
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && block_fail) {
            atomicExch(out_ok, 0);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            stop = (out_ok[0] == 0);
        }
        __syncthreads();
        if (stop) {
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Verifier enforcement kernels (no host reads)
// ---------------------------------------------------------------------------

extern "C" __global__ void sat_assert_status(
    const uint32_t* __restrict__ compile_needed,
    const int32_t* __restrict__ out_status, // len=1
    const int32_t* __restrict__ out_error,  // len=1
    int32_t expected_status
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (compile_needed[0] == 0u) {
        return;
    }
    int32_t st = out_status[0];
    int32_t er = out_error[0];
    if (er != 0 || st != expected_status) {
        sat_trap();
    }
}

extern "C" __global__ void sat_assert_ok(
    const uint32_t* __restrict__ compile_needed,
    const int32_t* __restrict__ out_ok
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (compile_needed[0] == 0u) {
        return;
    }
    if (out_ok[0] != 1) {
        sat_trap();
    }
}

// ---------------------------------------------------------------------------
// GPU CNF construction helpers for equivalence checking (XGCF -> CNF, ¬CNF)
//
// These kernels support the "GPU CDCL verifier" by building:
// - CNF(C) from a device-resident XGCF circuit (Tseitin encoding for internal nodes)
// - CNF(¬phi) from a device-resident CNF phi (unsatisfied-clause OR encoding)
// - simple CSR offset shifting for concatenation
//
// All outputs are DIMACS signed i32 literals with 1-based var ids.
// ---------------------------------------------------------------------------

// XGCF node type tags (must match crates/xlog-prob/src/xgcf.rs).
static constexpr uint8_t XGCF_CONST0 = 0;
static constexpr uint8_t XGCF_CONST1 = 1;
static constexpr uint8_t XGCF_LIT = 2;
static constexpr uint8_t XGCF_AND = 3;
static constexpr uint8_t XGCF_OR = 4;
static constexpr uint8_t XGCF_DECISION = 5;

__device__ __forceinline__ int32_t xgcf_node_lit(
    uint32_t node,
    const uint8_t* __restrict__ node_type,
    const int32_t* __restrict__ lit,
    const uint32_t* __restrict__ internal_prefix,
    uint32_t base_num_vars
) {
    // Literals are stored directly as signed DIMACS.
    if (node_type[node] == XGCF_LIT) {
        int32_t dim = lit[node];
        uint32_t v = (dim > 0) ? static_cast<uint32_t>(dim) : static_cast<uint32_t>(-dim);
        // Fail-fast: circuit leaves must refer to CNF vars in 1..=base_num_vars.
        if (v == 0u || v > base_num_vars) {
            sat_trap();
        }
        return dim;
    }
    // Internal node truth is represented by a fresh Tseitin variable:
    // v(node) = base_num_vars + internal_index(node) + 1.
    uint32_t v = base_num_vars + internal_prefix[node] + 1u;
    return static_cast<int32_t>(v);
}

// Compute per-node counts for Tseitin CNF encoding of an XGCF circuit.
//
// Outputs arrays are length num_nodes and are intended to be exclusive-scanned on device:
// - internal_counts[i] = 1 if node i is internal, else 0.
// - clause_counts[i] = number of CNF clauses emitted for node i (0 for Lit nodes).
// - lit_counts[i] = total number of literals emitted across those clauses (0 for Lit nodes).
extern "C" __global__ void sat_xgcf_cnf_counts(
    const uint32_t* __restrict__ compile_needed,
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    uint32_t num_nodes,
    uint32_t* __restrict__ internal_counts,
    uint32_t* __restrict__ clause_counts,
    uint32_t* __restrict__ lit_counts
) {
    uint32_t needed = compile_needed[0];
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (needed == 0u) {
        // Cache hit / compile not needed: do not touch the circuit buffers. Still produce
        // deterministic zeroed count arrays so downstream scans and totals remain valid.
        for (uint32_t i = tid; i < num_nodes; i += blockDim.x * gridDim.x) {
            internal_counts[i] = 0u;
            clause_counts[i] = 0u;
            lit_counts[i] = 0u;
        }
        return;
    }
    for (uint32_t i = tid; i < num_nodes; i += blockDim.x * gridDim.x) {
        uint8_t ty = node_type[i];
        if (ty == XGCF_LIT) {
            internal_counts[i] = 0u;
            clause_counts[i] = 0u;
            lit_counts[i] = 0u;
            continue;
        }

        internal_counts[i] = 1u;

        uint32_t cc = 0u;
        uint32_t lc = 0u;
        if (ty == XGCF_CONST0 || ty == XGCF_CONST1) {
            cc = 1u;
            lc = 1u;
        } else if (ty == XGCF_AND || ty == XGCF_OR) {
            uint32_t deg = child_offsets[i + 1u] - child_offsets[i];
            if (deg == 0u) {
                // AND([])=true, OR([])=false.
                cc = 1u;
                lc = 1u;
            } else {
                // For AND/OR: deg binary clauses + 1 long clause.
                cc = deg + 1u;
                // Total lits: 2*deg (binary) + (deg+1) (long) = 3*deg + 1.
                lc = 3u * deg + 1u;
            }
        } else if (ty == XGCF_DECISION) {
            cc = 4u;
            lc = 12u;
        }

        clause_counts[i] = cc;
        lit_counts[i] = lc;
    }
}

// Capture the last elements of the per-node count arrays (before in-place scans overwrite them).
extern "C" __global__ void sat_xgcf_cnf_capture_last_counts(
    const uint32_t* __restrict__ internal_counts,
    const uint32_t* __restrict__ clause_counts,
    const uint32_t* __restrict__ lit_counts,
    uint32_t num_nodes,
    uint32_t* __restrict__ out_internal_last, // len=1
    uint32_t* __restrict__ out_clause_last,   // len=1
    uint32_t* __restrict__ out_lit_last       // len=1
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (num_nodes == 0u) {
        out_internal_last[0] = 0u;
        out_clause_last[0] = 0u;
        out_lit_last[0] = 0u;
        return;
    }
    uint32_t last = num_nodes - 1u;
    out_internal_last[0] = internal_counts[last];
    out_clause_last[0] = clause_counts[last];
    out_lit_last[0] = lit_counts[last];
}

// Compute total counts after in-place exclusive scans and write them to device-resident scalars.
extern "C" __global__ void sat_xgcf_cnf_compute_totals(
    const uint32_t* __restrict__ internal_prefix,
    const uint32_t* __restrict__ clause_base,
    const uint32_t* __restrict__ lit_base,
    const uint32_t* __restrict__ internal_last, // len=1
    const uint32_t* __restrict__ clause_last,   // len=1
    const uint32_t* __restrict__ lit_last,      // len=1
    uint32_t num_nodes,
    const uint32_t* __restrict__ base_num_vars, // len=1
    uint32_t clause_cap,
    uint32_t lit_cap,
    uint32_t* __restrict__ out_num_vars,    // len=1
    uint32_t* __restrict__ out_num_clauses, // len=1
    uint32_t* __restrict__ out_num_lits     // len=1
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (num_nodes == 0u) {
        out_num_vars[0] = base_num_vars[0];
        out_num_clauses[0] = 0u;
        out_num_lits[0] = 0u;
        return;
    }

    uint32_t last = num_nodes - 1u;

    uint32_t internal_total = internal_prefix[last] + internal_last[0];
    uint32_t clause_total = clause_base[last] + clause_last[0];
    uint32_t lit_total = lit_base[last] + lit_last[0];

    // Deterministic, fail-fast overflow semantics.
    if (internal_total > num_nodes || clause_total > clause_cap || lit_total > lit_cap) {
        sat_trap();
    }

    uint32_t base = base_num_vars[0];
    if (base == 0u) {
        sat_trap();
    }
    out_num_vars[0] = base + internal_total;
    out_num_clauses[0] = clause_total;
    out_num_lits[0] = lit_total;
}

// Write CSR terminator offsets[num_clauses] = num_lits using device-resident totals.
extern "C" __global__ void sat_cnf_write_terminator(
    uint32_t* __restrict__ out_offsets,
    const uint32_t* __restrict__ num_clauses, // len=1
    const uint32_t* __restrict__ num_lits     // len=1
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    out_offsets[num_clauses[0]] = num_lits[0];
}

// Emit Tseitin CNF clauses for internal XGCF nodes into CSR buffers.
//
// Preconditions:
// - internal_prefix, clause_base, lit_base are exclusive scans of the corresponding count arrays.
// - out_offsets/out_lits have sufficient capacity for the emitted clauses.
extern "C" __global__ void sat_xgcf_cnf_emit(
    const uint32_t* __restrict__ compile_needed,
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ child_indices,
    const int32_t* __restrict__ lit,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ internal_prefix,
    const uint32_t* __restrict__ clause_base,
    const uint32_t* __restrict__ lit_base,
    const uint32_t* __restrict__ base_num_vars, // len=1
    uint32_t num_nodes,
    uint32_t* __restrict__ out_offsets,
    int32_t* __restrict__ out_lits
) {
    if (compile_needed[0] == 0u) {
        // Cache hit / compile not needed: do not touch circuit buffers or write output CNF.
        return;
    }
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base = base_num_vars[0];
    if (base == 0u) {
        sat_trap();
    }

    for (uint32_t node = tid; node < num_nodes; node += blockDim.x * gridDim.x) {
        uint8_t ty = node_type[node];
        if (ty == XGCF_LIT) {
            continue;
        }

        int32_t v = static_cast<int32_t>(base + internal_prefix[node] + 1u);
        uint32_t c0 = clause_base[node];
        uint32_t l0 = lit_base[node];

        if (ty == XGCF_CONST0) {
            // v is forced false.
            out_offsets[c0] = l0;
            out_lits[l0] = -v;
            continue;
        }
        if (ty == XGCF_CONST1) {
            // v is forced true.
            out_offsets[c0] = l0;
            out_lits[l0] = v;
            continue;
        }

        if (ty == XGCF_AND || ty == XGCF_OR) {
            uint32_t s = child_offsets[node];
            uint32_t e = child_offsets[node + 1u];
            uint32_t deg = e - s;
            if (deg == 0u) {
                out_offsets[c0] = l0;
                out_lits[l0] = (ty == XGCF_AND) ? v : -v;
                continue;
            }

            // Emit binary implications.
            for (uint32_t i = 0; i < deg; i++) {
                uint32_t child = child_indices[s + i];
                int32_t c_lit = xgcf_node_lit(child, node_type, lit, internal_prefix, base);

                out_offsets[c0 + i] = l0 + 2u * i;
                if (ty == XGCF_AND) {
                    // v -> child : (¬v ∨ child)
                    out_lits[l0 + 2u * i + 0u] = -v;
                    out_lits[l0 + 2u * i + 1u] = c_lit;
                } else {
                    // child -> v : (¬child ∨ v)
                    out_lits[l0 + 2u * i + 0u] = -c_lit;
                    out_lits[l0 + 2u * i + 1u] = v;
                }
            }

            // Emit long clause.
            uint32_t long_clause_idx = c0 + deg;
            uint32_t long_lit_idx = l0 + 2u * deg;
            out_offsets[long_clause_idx] = long_lit_idx;
            if (ty == XGCF_AND) {
                // (c1 ∧ ... ∧ ck) -> v  becomes  (v ∨ ¬c1 ∨ ... ∨ ¬ck)
                out_lits[long_lit_idx + 0u] = v;
                for (uint32_t i = 0; i < deg; i++) {
                    uint32_t child = child_indices[s + i];
                    int32_t c_lit = xgcf_node_lit(child, node_type, lit, internal_prefix, base);
                    out_lits[long_lit_idx + 1u + i] = -c_lit;
                }
            } else {
                // v -> (c1 ∨ ... ∨ ck) becomes (¬v ∨ c1 ∨ ... ∨ ck)
                out_lits[long_lit_idx + 0u] = -v;
                for (uint32_t i = 0; i < deg; i++) {
                    uint32_t child = child_indices[s + i];
                    int32_t c_lit = xgcf_node_lit(child, node_type, lit, internal_prefix, base);
                    out_lits[long_lit_idx + 1u + i] = c_lit;
                }
            }
            continue;
        }

        if (ty == XGCF_DECISION) {
            uint32_t x = decision_var[node];
            uint32_t f = decision_child_false[node];
            uint32_t t = decision_child_true[node];
            // Fail-fast: decision variables are original CNF vars in 1..=base.
            if (x == 0u || x > base) {
                sat_trap();
            }
            int32_t f_lit = xgcf_node_lit(f, node_type, lit, internal_prefix, base);
            int32_t t_lit = xgcf_node_lit(t, node_type, lit, internal_prefix, base);

            // Clause order matches CPU encoder for determinism.
            // (-x ∨ -t ∨ v)
            out_offsets[c0 + 0u] = l0 + 0u;
            out_lits[l0 + 0u] = -static_cast<int32_t>(x);
            out_lits[l0 + 1u] = -t_lit;
            out_lits[l0 + 2u] = v;

            // (x ∨ -f ∨ v)
            out_offsets[c0 + 1u] = l0 + 3u;
            out_lits[l0 + 3u] = static_cast<int32_t>(x);
            out_lits[l0 + 4u] = -f_lit;
            out_lits[l0 + 5u] = v;

            // (-x ∨ t ∨ -v)
            out_offsets[c0 + 2u] = l0 + 6u;
            out_lits[l0 + 6u] = -static_cast<int32_t>(x);
            out_lits[l0 + 7u] = t_lit;
            out_lits[l0 + 8u] = -v;

            // (x ∨ f ∨ -v)
            out_offsets[c0 + 3u] = l0 + 9u;
            out_lits[l0 + 9u] = static_cast<int32_t>(x);
            out_lits[l0 + 10u] = f_lit;
            out_lits[l0 + 11u] = -v;
            continue;
        }
    }

}

// Copy a CNF (CSR) into a destination CNF buffer at the given clause/literal base.
//
// This is a GPU-native building block used by the verifier path to concatenate CNFs without
// reading exact sizes on the host. It relies on device-resident `num_clauses/num_lits`.
extern "C" __global__ void sat_cnf_copy_into(
    const uint32_t* __restrict__ src_offsets,
    const int32_t* __restrict__ src_lits,
    const uint32_t* __restrict__ src_num_clauses, // len=1
    const uint32_t* __restrict__ src_num_lits,    // len=1
    uint32_t src_clause_cap,
    uint32_t src_lit_cap,
    const uint32_t* __restrict__ dst_clause_base, // len=1
    const uint32_t* __restrict__ dst_lit_base,    // len=1
    uint32_t dst_clause_cap,
    uint32_t dst_lit_cap,
    uint32_t* __restrict__ dst_offsets,
    int32_t* __restrict__ dst_lits
) {
    uint32_t m = src_num_clauses[0];
    uint32_t L = src_num_lits[0];
    uint32_t c0 = dst_clause_base[0];
    uint32_t l0 = dst_lit_base[0];

    // Bounds checks to avoid silent OOB reads/writes on malformed inputs.
    if (m > src_clause_cap || L > src_lit_cap) {
        sat_trap();
    }
    uint64_t dst_offsets_needed = static_cast<uint64_t>(c0) + static_cast<uint64_t>(m);
    uint64_t dst_lits_needed = static_cast<uint64_t>(l0) + static_cast<uint64_t>(L);
    if (dst_offsets_needed > static_cast<uint64_t>(dst_clause_cap)
        || dst_lits_needed > static_cast<uint64_t>(dst_lit_cap)) {
        sat_trap();
    }

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t i = tid; i < L; i += blockDim.x * gridDim.x) {
        dst_lits[l0 + i] = src_lits[i];
    }
    uint32_t n = m + 1u;
    for (uint32_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        dst_offsets[c0 + i] = src_offsets[i] + l0;
    }
}

// Shift CSR offsets by an additive constant and write into a destination offsets array.
extern "C" __global__ void sat_shift_offsets(
    const uint32_t* __restrict__ src_offsets,
    const uint32_t* __restrict__ src_num_clauses, // len=1
    uint32_t add,
    uint32_t dst_base,
    uint32_t* __restrict__ dst_offsets
) {
    uint32_t n = src_num_clauses[0] + 1u;
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        dst_offsets[dst_base + i] = src_offsets[i] + add;
    }
}

// Write a unit clause asserting the XGCF root to true/false.
//
// This is used when constructing SAT instances for equivalence checking:
// - CNF(phi ∧ ¬C): force_true=0
// - CNF(C ∧ ¬phi): force_true=1
//
// Note: `clause_base/lit_base` and `extra_num_*` are device-resident scalars (len=1 each) so the
// host never needs to read exact sizes to place the unit clause.
extern "C" __global__ void sat_xgcf_write_root_unit_clause(
    const uint32_t* __restrict__ compile_needed,
    const uint8_t* __restrict__ node_type,
    const int32_t* __restrict__ lit,
    const uint32_t* __restrict__ internal_prefix,
    const uint32_t* __restrict__ base_num_vars, // len=1
    uint32_t root,
    int32_t force_true, // 1=true, 0=false
    const uint32_t* __restrict__ clause_base, // len=1
    const uint32_t* __restrict__ lit_base,    // len=1
    const uint32_t* __restrict__ c_num_vars,    // len=1
    const uint32_t* __restrict__ c_num_clauses, // len=1
    const uint32_t* __restrict__ c_num_lits,    // len=1
    const uint32_t* __restrict__ extra_num_vars,    // len=1
    const uint32_t* __restrict__ extra_num_clauses, // len=1
    const uint32_t* __restrict__ extra_num_lits,    // len=1
    uint32_t out_var_cap,
    uint32_t out_clause_cap,
    uint32_t out_lit_cap,
    uint32_t* __restrict__ out_num_vars,    // len=1
    uint32_t* __restrict__ out_num_clauses, // len=1
    uint32_t* __restrict__ out_num_lits,    // len=1
    uint32_t* __restrict__ out_unsat_var_base,    // len=1
    uint32_t* __restrict__ out_extra_clause_base, // len=1
    uint32_t* __restrict__ out_extra_lit_base,    // len=1
    uint32_t* __restrict__ out_offsets,
    int32_t* __restrict__ out_lits
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    uint32_t needed = compile_needed[0];

    uint32_t cnv = c_num_vars[0];
    uint32_t cnc = c_num_clauses[0];
    uint32_t cnl = c_num_lits[0];

    uint32_t cb = clause_base[0];
    uint32_t lb = lit_base[0];
    uint32_t ev = extra_num_vars[0];
    uint32_t ec = extra_num_clauses[0];
    uint32_t el = extra_num_lits[0];

    if (cnc == 0u && cnl != 0u) {
        sat_trap();
    }

    // Cache hit / compile not needed: do not reference the circuit and do not write a unit clause.
    // Still compute deterministic device-resident totals/bases so CSR invariants remain valid and
    // downstream kernels can be no-ops.
    if (needed == 0u) {
        uint64_t total_vars64 = static_cast<uint64_t>(cnv) + static_cast<uint64_t>(ev);
        uint64_t total_clauses64 =
            static_cast<uint64_t>(cb) + static_cast<uint64_t>(cnc) + static_cast<uint64_t>(ec);
        uint64_t total_lits64 =
            static_cast<uint64_t>(lb) + static_cast<uint64_t>(cnl) + static_cast<uint64_t>(el);

        if (total_vars64 > static_cast<uint64_t>(out_var_cap)
            || total_clauses64 > static_cast<uint64_t>(out_clause_cap)
            || total_lits64 > static_cast<uint64_t>(out_lit_cap)) {
            sat_trap();
        }

        uint32_t total_vars = static_cast<uint32_t>(total_vars64);
        uint32_t total_clauses = static_cast<uint32_t>(total_clauses64);
        uint32_t total_lits = static_cast<uint32_t>(total_lits64);

        uint32_t extra_clause_base = cb + cnc;
        uint32_t extra_lit_base = lb + cnl;

        out_unsat_var_base[0] = cnv + 1u;
        out_extra_clause_base[0] = extra_clause_base;
        out_extra_lit_base[0] = extra_lit_base;

        out_num_vars[0] = total_vars;
        out_num_clauses[0] = total_clauses;
        out_num_lits[0] = total_lits;

        out_offsets[total_clauses] = total_lits;
        return;
    }

    uint64_t total_vars64 = static_cast<uint64_t>(cnv) + static_cast<uint64_t>(ev);
    uint64_t total_clauses64 = static_cast<uint64_t>(cb) + static_cast<uint64_t>(cnc) + 1ull
        + static_cast<uint64_t>(ec);
    uint64_t total_lits64 = static_cast<uint64_t>(lb) + static_cast<uint64_t>(cnl) + 1ull
        + static_cast<uint64_t>(el);

    if (total_vars64 > static_cast<uint64_t>(out_var_cap)
        || total_clauses64 > static_cast<uint64_t>(out_clause_cap)
        || total_lits64 > static_cast<uint64_t>(out_lit_cap)) {
        sat_trap();
    }

    uint32_t total_vars = static_cast<uint32_t>(total_vars64);
    uint32_t total_clauses = static_cast<uint32_t>(total_clauses64);
    uint32_t total_lits = static_cast<uint32_t>(total_lits64);

    uint32_t unit_clause_idx = cb + cnc;
    uint32_t unit_lit_idx = lb + cnl;

    uint32_t base = base_num_vars[0];
    if (base == 0u) {
        sat_trap();
    }
    int32_t root_lit = xgcf_node_lit(root, node_type, lit, internal_prefix, base);
    int32_t unit_lit = force_true ? root_lit : -root_lit;

    out_offsets[unit_clause_idx] = unit_lit_idx;
    out_lits[unit_lit_idx] = unit_lit;
    out_offsets[unit_clause_idx + 1u] = unit_lit_idx + 1u;

    uint32_t extra_clause_base = unit_clause_idx + 1u;
    uint32_t extra_lit_base = unit_lit_idx + 1u;

    out_unsat_var_base[0] = cnv + 1u;
    out_extra_clause_base[0] = extra_clause_base;
    out_extra_lit_base[0] = extra_lit_base;

    out_num_vars[0] = total_vars;
    out_num_clauses[0] = total_clauses;
    out_num_lits[0] = total_lits;

    // CSR terminator for the whole query CNF.
    out_offsets[total_clauses] = total_lits;
}

// Compute the exact CNF size contributions for the ¬phi encoding, using device-resident phi counts.
//
// ¬phi encoding (witness vars w_j):
//   - w_j -> ¬l for each literal l in clause j  (binary clauses only)
//   - OR_j w_j
//
// This encoding is SAT iff at least one clause of phi is unsatisfied, but avoids introducing the
// reverse implication (all literals false -> w_j). The witness vars are used as explicit decision
// variables in q2 so the solver can pick a violated clause early.
//
// Let m = #clauses in phi, L = #literals in phi (sum_j len_j):
//   clauses_notphi = L + 1
//   lits_notphi    = 2L + m
extern "C" __global__ void sat_not_phi_counts(
    const uint32_t* __restrict__ compile_needed,
    const uint32_t* __restrict__ phi_num_clauses, // len=1
    const uint32_t* __restrict__ phi_num_lits,    // len=1
    uint32_t* __restrict__ out_extra_num_vars,    // len=1
    uint32_t* __restrict__ out_extra_num_clauses, // len=1
    uint32_t* __restrict__ out_extra_num_lits     // len=1
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (compile_needed[0] == 0u) {
        out_extra_num_vars[0] = 0u;
        out_extra_num_clauses[0] = 0u;
        out_extra_num_lits[0] = 0u;
        return;
    }
    uint32_t m = phi_num_clauses[0];
    uint32_t L = phi_num_lits[0];

    uint64_t clauses64 = static_cast<uint64_t>(L) + 1ull;
    uint64_t lits64 = 2ull * static_cast<uint64_t>(L) + static_cast<uint64_t>(m);

    if (clauses64 > 0xffffffffull || lits64 > 0xffffffffull) {
        sat_trap();
    }

    out_extra_num_vars[0] = m;
    out_extra_num_clauses[0] = static_cast<uint32_t>(clauses64);
    out_extra_num_lits[0] = static_cast<uint32_t>(lits64);
}

// Emit CNF encoding of ¬phi where phi is a CNF in CSR form.
//
// Uses one fresh variable per clause j: w_j is a witness that selects clause j to be unsatisfied.
// Enforces:
// - w_j -> ¬l for each literal l in clause j
// - OR_j w_j   (at least one clause is selected as a witness)
//
// This is SAT iff phi is falsifiable, but keeps the witness vars as a pure selection mechanism
// (no reverse implication), which is more solver-friendly in the q2 equivalence query.
extern "C" __global__ void sat_emit_not_phi(
    const uint32_t* __restrict__ compile_needed,
    const uint32_t* __restrict__ phi_offsets,
    const int32_t* __restrict__ phi_lits,
    const uint32_t* __restrict__ num_clauses,     // len=1
    const uint32_t* __restrict__ unsat_var_base,  // len=1, u_j = unsat_var_base + j
    const uint32_t* __restrict__ out_clause_base, // len=1, clause index base in out_offsets
    const uint32_t* __restrict__ out_lit_base,    // len=1, literal index base in out_lits
    uint32_t* __restrict__ out_offsets,
    int32_t* __restrict__ out_lits
) {
    if (compile_needed[0] == 0u) {
        return;
    }
    uint32_t m = num_clauses[0];
    uint32_t u0 = unsat_var_base[0];
    uint32_t c0 = out_clause_base[0];
    uint32_t l0 = out_lit_base[0];

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t j = tid; j < m; j += blockDim.x * gridDim.x) {
        uint32_t s = phi_offsets[j];
        uint32_t e = phi_offsets[j + 1u];
        uint32_t len = e - s;

        // Each literal contributes one binary clause and 2 literals.
        uint32_t local_clause_base = s;
        uint32_t local_lit_base = 2u * s;
        uint32_t clause_base = c0 + local_clause_base;
        uint32_t lit_base = l0 + local_lit_base;

        int32_t w = static_cast<int32_t>(u0 + j);

        // Binary clauses: (¬w ∨ ¬l_i)
        for (uint32_t i = 0; i < len; i++) {
            uint32_t clause_idx = clause_base + i;
            uint32_t lit_idx = lit_base + 2u * i;
            int32_t l = phi_lits[s + i];

            out_offsets[clause_idx] = lit_idx;
            out_lits[lit_idx + 0u] = -w;
            out_lits[lit_idx + 1u] = -l;
        }
    }

    // Emit final OR over all witness vars, and set CSR terminator.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t total_phi_lits = phi_offsets[m];
        uint32_t big_clause_idx = c0 + total_phi_lits;
        uint32_t big_lit_idx = l0 + (2u * total_phi_lits);

        out_offsets[big_clause_idx] = big_lit_idx;
        for (uint32_t j = 0; j < m; j++) {
            out_lits[big_lit_idx + j] = static_cast<int32_t>(u0 + j);
        }
        out_offsets[big_clause_idx + 1u] = big_lit_idx + m;
    }
}
