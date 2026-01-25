// kernels/sat.cu
//
// GPU-native CDCL SAT solver + GPU-native result validation.
//
// Design constraints:
// - Complete (SAT/UNSAT; no "unknown")
// - Deterministic (tie-breaking and schedules are stable)
// - Memory-bounded (fixed-capacity arenas; overflow => hard error)
// - GPU-native data-plane (host is control-plane only)

#include <cstdint>
#include <cstddef>

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

__device__ __forceinline__ void sat_bump_var(uint32_t v, uint32_t* __restrict__ var_activity, uint32_t bump) {
    uint32_t a = var_activity[v];
    uint32_t na = a + bump;
    var_activity[v] = na;
}

// Rescale activities when close to overflow.
__device__ __forceinline__ void sat_maybe_rescale_activities(
    uint32_t num_vars,
    uint32_t* __restrict__ var_activity
) {
    // Rescale if any activity is very large (scan deterministically).
    uint32_t maxv = 0;
    for (uint32_t v = 1; v <= num_vars; v++) {
        uint32_t a = var_activity[v];
        if (a > maxv) {
            maxv = a;
        }
    }
    if (maxv < 0xF0000000u) {
        return;
    }
    for (uint32_t v = 1; v <= num_vars; v++) {
        var_activity[v] >>= 8;
    }
}

__device__ __forceinline__ uint32_t sat_pick_branch_var(
    uint32_t num_vars,
    const int8_t* __restrict__ assign,
    const uint32_t* __restrict__ var_activity
) {
    uint32_t best_v = 0;
    uint32_t best_a = 0;
    for (uint32_t v = 1; v <= num_vars; v++) {
        if (assign[v] != SAT_VAL_UNASSIGNED) {
            continue;
        }
        uint32_t a = var_activity[v];
        if (best_v == 0 || a > best_a || (a == best_a && v < best_v)) {
            best_v = v;
            best_a = a;
        }
    }
    return best_v;
}

// ---------------------------------------------------------------------------
// CDCL core helpers (enqueue, propagate, analyze, backtrack)
// ---------------------------------------------------------------------------

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
    int8_t* __restrict__ phase
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
        return true;
    }
    return cur == val;
}

__device__ __forceinline__ void sat_unassign_lit(
    int32_t lit,
    int8_t* __restrict__ assign,
    uint32_t* __restrict__ level,
    int32_t* __restrict__ reason
) {
    uint32_t v = sat_var(lit);
    assign[v] = SAT_VAL_UNASSIGNED;
    level[v] = 0;
    reason[v] = SAT_REASON_UNASSIGNED;
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
    int32_t* __restrict__ reason
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
        sat_unassign_lit(lit, assign, level, reason);
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
    int8_t* __restrict__ phase
) {
    while (*qhead < *trail_len) {
        int32_t p = trail[*qhead];
        *qhead = *qhead + 1u;

        // Clauses watching the falsified literal (~p) must be processed.
        uint32_t fals_idx = sat_lit_index(-p);

        int32_t node = watch_head[fals_idx];
        while (node >= 0) {
            int32_t next_node = watch_next[node]; // save before possible move
            int32_t clause_id = node / 2;
            uint32_t slot = static_cast<uint32_t>(node & 1);

            // Skip deleted learned clauses.
            if (clause_id >= static_cast<int32_t>(num_base_clauses)) {
                uint32_t lid = static_cast<uint32_t>(clause_id) - num_base_clauses;
                if (learned_deleted[lid] != 0) {
                    // Remove stale watch nodes deterministically.
                    sat_watch_remove(node, watch_head, watch_next, watch_prev, fals_idx);
                    node = next_node;
                    continue;
                }
            }

            SatClauseView cv = sat_get_clause(
                clause_id, base_offsets, base_lits, num_base_clauses, learned_offsets, learned_lits);
            uint32_t w0 = watch0_pos[static_cast<uint32_t>(clause_id)];
            uint32_t w1 = watch1_pos[static_cast<uint32_t>(clause_id)];
            uint32_t wpos_false = (slot == 0) ? w0 : w1;
            uint32_t wpos_other = (slot == 0) ? w1 : w0;

            // Unit clause case: both watches may be identical.
            int32_t other_lit = (wpos_other < cv.len) ? cv.lits[wpos_other] : 0;
            if (cv.len == 0) {
                return clause_id; // empty clause => immediate conflict
            }
            if (other_lit == 0) {
                // Defensive: malformed clause.
                return clause_id;
            }

            // If other watch is satisfied, clause is satisfied.
            if (sat_eval_lit(other_lit, assign) == SAT_VAL_TRUE) {
                node = next_node;
                continue;
            }

            // Try to find a new watch in this clause that is not falsified.
            bool moved = false;
            for (uint32_t k = 0; k < cv.len; k++) {
                if (k == wpos_other || k == wpos_false) {
                    continue;
                }
                int32_t lit = cv.lits[k];
                if (sat_eval_lit(lit, assign) != SAT_VAL_FALSE) {
                    // Move the false watch to this literal.
                    uint32_t new_idx = sat_lit_index(lit);
                    // Remove from current falsified list and insert into new list.
                    sat_watch_remove(node, watch_head, watch_next, watch_prev, fals_idx);
                    sat_watch_insert_head(node, watch_head, watch_next, watch_prev, new_idx);
                    // Update watch position for this clause slot.
                    if (slot == 0) {
                        // slot 0 corresponds to watch0_pos
                        watch0_pos[static_cast<uint32_t>(clause_id)] = k;
                    } else {
                        watch1_pos[static_cast<uint32_t>(clause_id)] = k;
                    }
                    moved = true;
                    break;
                }
            }

            if (moved) {
                node = next_node;
                continue;
            }

            // No new watch found: clause is unit or conflicting.
            int8_t other_eval = sat_eval_lit(other_lit, assign);
            if (other_eval == SAT_VAL_FALSE) {
                // Conflict.
                return clause_id;
            }
            // Unit: enqueue other_lit.
            if (!sat_enqueue(other_lit, clause_id, decision_level, assign, level, reason, trail, trail_len,
                             assigned_count, phase)) {
                return clause_id; // conflict assigning unit
            }
            if (clause_id >= static_cast<int32_t>(num_base_clauses)) {
                uint32_t lid = static_cast<uint32_t>(clause_id) - num_base_clauses;
                learned_activity[lid] += 1u;
            }

            node = next_node;
        }
    }
    return -1;
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
            if (lv == 0) {
                continue;
            }
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
    // Base CNF (CSR)
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ clause_lits,
    uint32_t num_vars,
    uint32_t num_clauses,
    // Solver configuration
    uint32_t max_learned_clauses,
    uint32_t max_learned_lits,
    uint32_t max_proof_u32,
    uint32_t restart_base,        // e.g. 100
    uint32_t reduce_interval,     // e.g. 2000
    // Variable state
    int8_t* __restrict__ assign,      // len = num_vars+1
    uint32_t* __restrict__ level,     // len = num_vars+1
    int32_t* __restrict__ reason,     // len = num_vars+1
    uint32_t* __restrict__ var_activity, // len = num_vars+1
    int8_t* __restrict__ var_phase,   // len = num_vars+1
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
    if (threadIdx.x != 0) {
        return;
    }

    // Initialize outputs.
    out_status[0] = SAT_STATUS_ERROR;
    out_error[0] = SAT_ERR_OK;
    out_learned_count[0] = 0;

    // Initialize variable arrays.
    for (uint32_t v = 0; v <= num_vars; v++) {
        assign[v] = SAT_VAL_UNASSIGNED;
        level[v] = 0;
        reason[v] = SAT_REASON_UNASSIGNED;
        var_activity[v] = 0;
        var_phase[v] = SAT_VAL_FALSE; // deterministic default: false
        seen[v] = 0;
    }

    // Initialize trail.
    uint32_t trail_len = 0;
    uint32_t qhead = 0;
    uint32_t decision_level = 0;
    uint32_t assigned_count = 0;

    trail_lim[0] = 0;
    for (uint32_t i = 1; i <= num_vars; i++) {
        trail_lim[i] = 0;
    }

    // Initialize learned arena.
    uint32_t learned_count = 0;
    uint32_t learned_lit_count = 0;
    learned_offsets[0] = 0;
    proof_offsets[0] = 0;
    uint32_t proof_len = 0;

    for (uint32_t i = 0; i < max_learned_clauses; i++) {
        learned_deleted[i] = 0;
        learned_lbd[i] = 0;
        learned_activity[i] = 0;
        learned_locked[i] = 0;
    }

    // Initialize watch lists.
    for (uint32_t i = 0; i < 2u * num_vars; i++) {
        watch_head[i] = -1;
    }
    uint32_t max_total_clauses = num_clauses + max_learned_clauses;
    for (uint32_t i = 0; i < 2u * max_total_clauses; i++) {
        watch_next[i] = -1;
        watch_prev[i] = -1;
    }

    // Initialize base clause watches deterministically in clause_id order.
    for (uint32_t c = 0; c < num_clauses; c++) {
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
                out_status[0] = SAT_STATUS_ERROR;
                out_error[0] = SAT_ERR_LEARNED_CLAUSES_OVERFLOW;
                out_learned_count[0] = learned_count;
                return;
            }
            if (proof_len + 2u > max_proof_u32) {
                out_status[0] = SAT_STATUS_ERROR;
                out_error[0] = SAT_ERR_PROOF_OVERFLOW;
                out_learned_count[0] = learned_count;
                return;
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

            out_status[0] = SAT_STATUS_UNSAT;
            out_error[0] = SAT_ERR_OK;
            out_learned_count[0] = learned_count;
            return;
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
                             &assigned_count, var_phase)) {
                // Contradictory unit clauses at level 0. Emit an empty learned clause + resolution trace.
                uint32_t proof_steps = 0;
                bool ok = sat_analyze_level0_unsat(
                    static_cast<int32_t>(c),
                    clause_offsets, clause_lits, num_clauses,
                    learned_offsets, learned_lits, learned_deleted,
                    learned_count,
                    num_vars,
                    reason,
                    trail, trail_len,
                    seen,
                    proof_vars_tmp, proof_reason_tmp, &proof_steps);
                if (!ok) {
                    out_status[0] = SAT_STATUS_ERROR;
                    out_error[0] = SAT_ERR_INVALID_PROOF;
                    out_learned_count[0] = learned_count;
                    return;
                }

                if (learned_count >= max_learned_clauses) {
                    out_status[0] = SAT_STATUS_ERROR;
                    out_error[0] = SAT_ERR_LEARNED_CLAUSES_OVERFLOW;
                    out_learned_count[0] = learned_count;
                    return;
                }
                if (proof_len + 2u + 2u * proof_steps > max_proof_u32) {
                    out_status[0] = SAT_STATUS_ERROR;
                    out_error[0] = SAT_ERR_PROOF_OVERFLOW;
                    out_learned_count[0] = learned_count;
                    return;
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

                out_status[0] = SAT_STATUS_UNSAT;
                out_error[0] = SAT_ERR_OK;
                out_learned_count[0] = learned_count;
                return;
            }
        }
    }

    // Initial propagation.
    int32_t confl = sat_propagate(
        clause_offsets, clause_lits, num_vars, num_clauses,
        learned_offsets, learned_lits, learned_deleted, learned_activity,
        watch0_pos, watch1_pos,
        watch_head, watch_next, watch_prev,
        &qhead, decision_level,
        assign, level, reason,
        trail, &trail_len, &assigned_count,
        var_phase);
    if (confl >= 0) {
        // Conflict at level 0 => UNSAT. Emit an empty learned clause + resolution trace certificate.
        uint32_t proof_steps = 0;
        bool ok = sat_analyze_level0_unsat(
            confl,
            clause_offsets, clause_lits, num_clauses,
            learned_offsets, learned_lits, learned_deleted,
            learned_count,
            num_vars,
            reason,
            trail, trail_len,
            seen,
            proof_vars_tmp, proof_reason_tmp, &proof_steps);
        if (!ok) {
            out_status[0] = SAT_STATUS_ERROR;
            out_error[0] = SAT_ERR_INVALID_PROOF;
            out_learned_count[0] = learned_count;
            return;
        }

        if (learned_count >= max_learned_clauses) {
            out_status[0] = SAT_STATUS_ERROR;
            out_error[0] = SAT_ERR_LEARNED_CLAUSES_OVERFLOW;
            out_learned_count[0] = learned_count;
            return;
        }
        if (proof_len + 2u + 2u * proof_steps > max_proof_u32) {
            out_status[0] = SAT_STATUS_ERROR;
            out_error[0] = SAT_ERR_PROOF_OVERFLOW;
            out_learned_count[0] = learned_count;
            return;
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

        out_status[0] = SAT_STATUS_UNSAT;
        out_error[0] = SAT_ERR_OK;
        out_learned_count[0] = learned_count;
        return;
    }

    // CDCL loop.
    uint32_t conflicts = 0;
    uint32_t next_reduce = reduce_interval;
    uint32_t restart_iter = 1;
    uint32_t next_restart = restart_base * sat_luby(restart_iter);
    uint32_t var_bump = 1u << 24;

    for (;;) {
        // Propagate.
        confl = sat_propagate(
            clause_offsets, clause_lits, num_vars, num_clauses,
            learned_offsets, learned_lits, learned_deleted, learned_activity,
            watch0_pos, watch1_pos,
            watch_head, watch_next, watch_prev,
            &qhead, decision_level,
            assign, level, reason,
            trail, &trail_len, &assigned_count,
            var_phase);

        if (confl >= 0) {
            // Conflict.
            conflicts++;
            if (decision_level == 0) {
                // UNSAT. Emit an empty learned clause + resolution trace certificate.
                uint32_t proof_steps = 0;
                bool ok = sat_analyze_level0_unsat(
                    confl,
                    clause_offsets, clause_lits, num_clauses,
                    learned_offsets, learned_lits, learned_deleted,
                    learned_count,
                    num_vars,
                    reason,
                    trail, trail_len,
                    seen,
                    proof_vars_tmp, proof_reason_tmp, &proof_steps);
                if (!ok) {
                    out_status[0] = SAT_STATUS_ERROR;
                    out_error[0] = SAT_ERR_INVALID_PROOF;
                    out_learned_count[0] = learned_count;
                    return;
                }

                if (learned_count >= max_learned_clauses) {
                    out_status[0] = SAT_STATUS_ERROR;
                    out_error[0] = SAT_ERR_LEARNED_CLAUSES_OVERFLOW;
                    out_learned_count[0] = learned_count;
                    return;
                }
                if (proof_len + 2u + 2u * proof_steps > max_proof_u32) {
                    out_status[0] = SAT_STATUS_ERROR;
                    out_error[0] = SAT_ERR_PROOF_OVERFLOW;
                    out_learned_count[0] = learned_count;
                    return;
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

                out_status[0] = SAT_STATUS_UNSAT;
                out_error[0] = SAT_ERR_OK;
                out_learned_count[0] = learned_count;
                return;
            }

            // Analyze.
            uint32_t learnt_len = 0;
            uint32_t bt = 0;
            uint32_t proof_steps = 0;
            bool ok = sat_analyze(
                confl,
                clause_offsets, clause_lits, num_clauses,
                learned_offsets, learned_lits, learned_deleted,
                learned_count,
                decision_level,
                assign, level, reason,
                trail, trail_len,
                seen, learnt_tmp,
                &learnt_len, &bt,
                proof_vars_tmp, proof_reason_tmp, &proof_steps);
            if (!ok) {
                out_status[0] = SAT_STATUS_ERROR;
                out_error[0] = SAT_ERR_INVALID_PROOF;
                out_learned_count[0] = learned_count;
                return;
            }

            // Bump learned clause activity for clauses that participate in this conflict analysis.
            if (confl >= static_cast<int32_t>(num_clauses)) {
                uint32_t lid = static_cast<uint32_t>(confl) - num_clauses;
                if (lid < learned_count) {
                    learned_activity[lid] += 1u;
                }
            }
            for (uint32_t i = 0; i < proof_steps; i++) {
                uint32_t rc = proof_reason_tmp[i];
                if (rc >= num_clauses) {
                    uint32_t lid = rc - num_clauses;
                    if (lid < learned_count) {
                        learned_activity[lid] += 1u;
                    }
                }
            }

            // Bump variables in learned clause (deterministic).
            for (uint32_t i = 0; i < learnt_len; i++) {
                uint32_t v = sat_var(learnt_tmp[i]);
                sat_bump_var(v, var_activity, var_bump);
            }
            // Increase bump slowly (integer approximation).
            var_bump += var_bump >> 5; // +3.125%
            sat_maybe_rescale_activities(num_vars, var_activity);

            // Backjump.
            sat_backtrack(bt, &decision_level, num_vars + 1u, trail_lim, &qhead,
                          trail, &trail_len, &assigned_count, assign, level, reason);

            // Learn clause (append to arena).
            if (learned_count >= max_learned_clauses) {
                out_status[0] = SAT_STATUS_ERROR;
                out_error[0] = SAT_ERR_LEARNED_CLAUSES_OVERFLOW;
                out_learned_count[0] = learned_count;
                return;
            }
            uint32_t lid = learned_count;
            uint32_t clause_id = num_clauses + lid;
            learned_offsets[lid] = learned_lit_count;
            if (learned_lit_count + learnt_len > max_learned_lits) {
                out_status[0] = SAT_STATUS_ERROR;
                out_error[0] = SAT_ERR_LEARNED_LITS_OVERFLOW;
                out_learned_count[0] = learned_count;
                return;
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
                out_status[0] = SAT_STATUS_ERROR;
                out_error[0] = SAT_ERR_PROOF_OVERFLOW;
                out_learned_count[0] = learned_count;
                return;
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
            if (!sat_enqueue(learnt_tmp[0], static_cast<int32_t>(clause_id), decision_level, assign, level, reason,
                             trail, &trail_len, &assigned_count, var_phase)) {
                // Immediate contradiction after learning is still valid UNSAT only if at level 0,
                // otherwise treat as conflict at this level on next propagate.
            }
            continue;
        }

        // No conflict. Check SAT.
        if (assigned_count == num_vars) {
            out_status[0] = SAT_STATUS_SAT;
            out_error[0] = SAT_ERR_OK;
            out_learned_count[0] = learned_count;
            return;
        }

        // Restart (deterministic schedule).
        if (conflicts >= next_restart && decision_level > 0) {
            sat_backtrack(0, &decision_level, num_vars + 1u, trail_lim, &qhead,
                          trail, &trail_len, &assigned_count, assign, level, reason);
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
            for (uint32_t v = 1; v <= num_vars; v++) {
                int32_t r = reason[v];
                if (r >= static_cast<int32_t>(num_clauses)) {
                    uint32_t lid = static_cast<uint32_t>(r) - num_clauses;
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
                    uint32_t clause_id = num_clauses + i;
                    int32_t node0 = static_cast<int32_t>(clause_id * 2u);
                    int32_t node1 = static_cast<int32_t>(clause_id * 2u + 1u);
                    // Remove using current watch positions to find literal indices.
                    SatClauseView cv = sat_get_clause(static_cast<int32_t>(clause_id), clause_offsets, clause_lits, num_clauses,
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

        // Decide next branch.
        uint32_t next_v = sat_pick_branch_var(num_vars, assign, var_activity);
        if (next_v == 0) {
            // No unassigned vars but assigned_count != num_vars: inconsistent accounting.
            out_status[0] = SAT_STATUS_ERROR;
            out_error[0] = SAT_ERR_INVALID_MODEL;
            out_learned_count[0] = learned_count;
            return;
        }
        decision_level++;
        trail_lim[decision_level] = trail_len;
        qhead = trail_len;

        int8_t ph = var_phase[next_v];
        int32_t decision_lit = (ph == SAT_VAL_TRUE) ? static_cast<int32_t>(next_v) : -static_cast<int32_t>(next_v);
        if (!sat_enqueue(decision_lit, SAT_REASON_NONE, decision_level, assign, level, reason, trail, &trail_len,
                         &assigned_count, var_phase)) {
            // Should never conflict on decision enqueue; treat as error.
            out_status[0] = SAT_STATUS_ERROR;
            out_error[0] = SAT_ERR_INVALID_MODEL;
            out_learned_count[0] = learned_count;
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: sat_check_model (CNF satisfaction check)
// ---------------------------------------------------------------------------

extern "C" __global__ void sat_check_model(
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ clause_lits,
    uint32_t num_clauses,
    const int8_t* __restrict__ assign,
    int32_t* __restrict__ out_ok // 1 ok, 0 fail
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        out_ok[0] = 1;
    }
    __syncthreads();

    for (uint32_t c = tid; c < num_clauses; c += blockDim.x * gridDim.x) {
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

extern "C" __global__ void sat_proof_check(
    const uint32_t* __restrict__ base_offsets,
    const int32_t* __restrict__ base_lits,
    uint32_t num_clauses,
    const uint32_t* __restrict__ learned_offsets,
    const int32_t* __restrict__ learned_lits,
    uint32_t learned_count,
    const uint32_t* __restrict__ proof_offsets,
    const uint32_t* __restrict__ proof_data,
    // scratch buffers (two alternating clause buffers)
    int32_t* __restrict__ scratch_a,
    int32_t* __restrict__ scratch_b,
    uint32_t scratch_cap,
    int32_t* __restrict__ out_ok // 1 ok, 0 fail
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    out_ok[0] = 1;

    if (learned_count == 0) {
        out_ok[0] = 0;
        return;
    }

    // Require last learned clause to be empty (unsat certificate).
    uint32_t last_len = learned_offsets[learned_count] - learned_offsets[learned_count - 1u];
    if (last_len != 0u) {
        out_ok[0] = 0;
        return;
    }

    // Verify each learned clause's resolution trace matches the stored learned clause.
    for (uint32_t i = 0; i < learned_count; i++) {
        uint32_t poff = proof_offsets[i];
        uint32_t pend = proof_offsets[i + 1u];
        if (pend < poff + 2u) {
            out_ok[0] = 0;
            return;
        }
        uint32_t conflict_clause = proof_data[poff + 0u];
        uint32_t steps = proof_data[poff + 1u];
        uint32_t expect = 2u + 2u * steps;
        if (poff + expect != pend) {
            out_ok[0] = 0;
            return;
        }

        if (conflict_clause >= num_clauses + i) {
            out_ok[0] = 0;
            return;
        }

        // Load initial clause into scratch_a.
        SatClauseView c0 = sat_get_clause(static_cast<int32_t>(conflict_clause),
                                          base_offsets, base_lits, num_clauses,
                                          learned_offsets, learned_lits);
        if (c0.len > scratch_cap) {
            out_ok[0] = 0;
            return;
        }
        for (uint32_t k = 0; k < c0.len; k++) {
            scratch_a[k] = c0.lits[k];
        }
        uint32_t cur_len = c0.len;

        bool use_a = true;
        for (uint32_t s = 0; s < steps; s++) {
            uint32_t var = proof_data[poff + 2u + 2u * s];
            uint32_t reason_clause = proof_data[poff + 2u + 2u * s + 1u];
            if (reason_clause == 0xFFFFFFFFu) {
                // Decision (no reason) shouldn't appear in a valid resolution trace.
                out_ok[0] = 0;
                return;
            }
            if (reason_clause >= num_clauses + i) {
                out_ok[0] = 0;
                return;
            }

            SatClauseView rc = sat_get_clause(static_cast<int32_t>(reason_clause),
                                              base_offsets, base_lits, num_clauses,
                                              learned_offsets, learned_lits);

            int32_t ok = 1;
            if (use_a) {
                uint32_t out_len = sat_resolve_on_var(var, scratch_a, cur_len, rc.lits, rc.len,
                                                      scratch_b, scratch_cap, &ok);
                if (!ok) {
                    out_ok[0] = 0;
                    return;
                }
                cur_len = out_len;
            } else {
                uint32_t out_len = sat_resolve_on_var(var, scratch_b, cur_len, rc.lits, rc.len,
                                                      scratch_a, scratch_cap, &ok);
                if (!ok) {
                    out_ok[0] = 0;
                    return;
                }
                cur_len = out_len;
            }
            use_a = !use_a;
        }

        // Compare with stored learned clause i.
        SatClauseView lc;
        lc.lits = learned_lits + learned_offsets[i];
        lc.len = learned_offsets[i + 1u] - learned_offsets[i];

        const int32_t* final_clause = use_a ? scratch_b : scratch_a;
        if (!sat_clause_equal_set(final_clause, cur_len, lc.lits, lc.len)) {
            out_ok[0] = 0;
            return;
        }
    }
}
