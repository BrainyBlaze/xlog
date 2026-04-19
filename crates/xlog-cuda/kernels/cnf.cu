// GPU CNF encoding kernels for PIR -> Tseitin CNF.
// Implements reachability, variable assignment, clause counting, and clause emission.

// NOTE: This file is compiled to PTX and embedded in the repo. To keep regeneration possible in
// environments that only have NVRTC available (no host C++ standard library headers), avoid
// libstdc++ includes and define the fixed-width integer aliases we need.
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

#define PIR_CONST    0
#define PIR_LIT      1
#define PIR_NEG_LIT  2
#define PIR_AND      3
#define PIR_OR       4
#define PIR_DECISION 5

__device__ __forceinline__ void cnf_trap() {
    asm volatile("trap;");
}

__device__ __forceinline__ uint32_t cnf_atomic_load_u32(const uint32_t* ptr) {
    return atomicAdd(const_cast<uint32_t*>(ptr), 0u);
}

extern "C" __global__ void cnf_reachability_init(
    const uint32_t* __restrict__ roots,
    uint32_t num_roots,
    uint32_t num_nodes,
    uint32_t* __restrict__ reachable,
    uint32_t* __restrict__ queue,
    uint32_t* __restrict__ queue_ready,
    uint32_t* __restrict__ head,
    uint32_t* __restrict__ tail,
    uint32_t* __restrict__ in_flight
) {
    // Deterministic, single-thread initialization. Reachability is a correctness-critical
    // preprocessing step; keep it simple and race-free.
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    head[0] = 0u;
    tail[0] = 0u;
    in_flight[0] = 0u;

    for (uint32_t i = 0; i < num_roots; i++) {
        uint32_t r = roots[i];
        if (r >= num_nodes) {
            cnf_trap();
        }
        if (reachable[r] == 0u) {
            reachable[r] = 1u;
            uint32_t pos = tail[0];
            if (pos >= num_nodes) {
                cnf_trap();
            }
            queue[pos] = r;
            queue_ready[pos] = 1u;
            tail[0] = pos + 1u;
        }
    }
}

extern "C" __global__ void cnf_reachability_bfs(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ children,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    uint32_t num_nodes,
    uint32_t* __restrict__ reachable,
    uint32_t* __restrict__ queue,
    uint32_t* __restrict__ queue_ready,
    uint32_t* __restrict__ head,
    uint32_t* __restrict__ tail,
    uint32_t* __restrict__ in_flight
) {
    // Deterministic, single-thread reachability closure. The reachable set is order-independent,
    // but parallel work-stealing variants are extremely easy to get wrong; prefer correctness.
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    for (;;) {
        uint32_t h = head[0];
        uint32_t t = tail[0];
        if (h >= t) {
            break;
        }
        while (queue_ready[h] == 0u) {
        }
        uint32_t node = queue[h];
        head[0] = h + 1u;

        if (node >= num_nodes) {
            cnf_trap();
        }
        uint8_t tag = node_type[node];
        if (tag == PIR_AND || tag == PIR_OR) {
            uint32_t start = child_offsets[node];
            uint32_t end = child_offsets[node + 1u];
            for (uint32_t j = start; j < end; j++) {
                uint32_t child = children[j];
                if (child >= num_nodes) {
                    cnf_trap();
                }
                if (reachable[child] == 0u) {
                    reachable[child] = 1u;
                    uint32_t pos = tail[0];
                    if (pos >= num_nodes) {
                        cnf_trap();
                    }
                    queue[pos] = child;
                    queue_ready[pos] = 1u;
                    tail[0] = pos + 1u;
                }
            }
        } else if (tag == PIR_DECISION) {
            uint32_t f = decision_child_false[node];
            uint32_t tt = decision_child_true[node];
            if (f >= num_nodes || tt >= num_nodes) {
                cnf_trap();
            }
            if (reachable[f] == 0u) {
                reachable[f] = 1u;
                uint32_t pos = tail[0];
                if (pos >= num_nodes) {
                    cnf_trap();
                }
                queue[pos] = f;
                queue_ready[pos] = 1u;
                tail[0] = pos + 1u;
            }
            if (reachable[tt] == 0u) {
                reachable[tt] = 1u;
                uint32_t pos = tail[0];
                if (pos >= num_nodes) {
                    cnf_trap();
                }
                queue[pos] = tt;
                queue_ready[pos] = 1u;
                tail[0] = pos + 1u;
            }
        }
    }
}

extern "C" __global__ void cnf_mark_leaf_choice(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ leaf_id,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ reachable,
    uint32_t num_nodes,
    uint32_t leaf_cap,
    uint32_t choice_cap,
    uint32_t* __restrict__ leaf_used,
    uint32_t* __restrict__ choice_used
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = tid; i < num_nodes; i += blockDim.x * gridDim.x) {
        if (reachable[i] == 0u) {
            continue;
        }
        uint8_t tag = node_type[i];
        if (tag == PIR_LIT || tag == PIR_NEG_LIT) {
            uint32_t leaf = leaf_id[i];
            if (leaf >= leaf_cap) {
                cnf_trap();
            }
            atomicExch(reinterpret_cast<unsigned int*>(&leaf_used[leaf]), 1u);
        } else if (tag == PIR_DECISION) {
            uint32_t var = decision_var[i];
            if (var >= choice_cap) {
                cnf_trap();
            }
            atomicExch(reinterpret_cast<unsigned int*>(&choice_used[var]), 1u);
        }
    }
}

extern "C" __global__ void cnf_assign_leaf_var(
    const uint32_t* __restrict__ leaf_used,
    const uint32_t* __restrict__ leaf_prefix,
    uint32_t leaf_cap,
    uint32_t* __restrict__ leaf_var
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = tid; i < leaf_cap; i += blockDim.x * gridDim.x) {
        if (leaf_used[i] != 0u) {
            leaf_var[i] = leaf_prefix[i] + 1u;
        } else {
            leaf_var[i] = 0u;
        }
    }
}

extern "C" __global__ void cnf_assign_choice_var(
    const uint32_t* __restrict__ choice_used,
    const uint32_t* __restrict__ choice_prefix,
    uint32_t choice_cap,
    const uint32_t* __restrict__ base_choice,
    uint32_t* __restrict__ choice_var
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base = base_choice[0];
    for (uint32_t i = tid; i < choice_cap; i += blockDim.x * gridDim.x) {
        if (choice_used[i] != 0u) {
            choice_var[i] = base + choice_prefix[i];
        } else {
            choice_var[i] = 0u;
        }
    }
}

extern "C" __global__ void cnf_mark_node_vars(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ reachable,
    uint32_t num_nodes,
    uint32_t* __restrict__ node_needs_var
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = tid; i < num_nodes; i += blockDim.x * gridDim.x) {
        if (reachable[i] == 0u) {
            node_needs_var[i] = 0u;
            continue;
        }
        node_needs_var[i] = (node_type[i] == PIR_LIT) ? 0u : 1u;
    }
}

extern "C" __global__ void cnf_count_clauses(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ reachable,
    uint32_t num_nodes,
    uint32_t* __restrict__ clause_counts,
    uint32_t* __restrict__ lit_counts
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = tid; i < num_nodes; i += blockDim.x * gridDim.x) {
        if (reachable[i] == 0u) {
            clause_counts[i] = 0u;
            lit_counts[i] = 0u;
            continue;
        }
        uint8_t tag = node_type[i];
        if (tag == PIR_LIT) {
            clause_counts[i] = 0u;
            lit_counts[i] = 0u;
            continue;
        }

        uint32_t cc = 0u;
        uint32_t lc = 0u;
        if (tag == PIR_CONST) {
            cc = 1u;
            lc = 1u;
        } else if (tag == PIR_NEG_LIT) {
            cc = 2u;
            lc = 4u;
        } else if (tag == PIR_AND || tag == PIR_OR) {
            uint32_t deg = child_offsets[i + 1u] - child_offsets[i];
            if (deg == 0u) {
                cc = 1u;
                lc = 1u;
            } else {
                cc = deg + 1u;
                lc = 3u * deg + 1u;
            }
        } else if (tag == PIR_DECISION) {
            cc = 4u;
            lc = 12u;
        } else {
            cnf_trap();
        }

        clause_counts[i] = cc;
        lit_counts[i] = lc;
    }
}

extern "C" __global__ void cnf_capture_last_counts(
    const uint32_t* __restrict__ node_needs_var,
    const uint32_t* __restrict__ clause_counts,
    const uint32_t* __restrict__ lit_counts,
    uint32_t num_nodes,
    uint32_t* __restrict__ out_node_last,
    uint32_t* __restrict__ out_clause_last,
    uint32_t* __restrict__ out_lit_last
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    if (num_nodes == 0u) {
        out_node_last[0] = 0u;
        out_clause_last[0] = 0u;
        out_lit_last[0] = 0u;
        return;
    }
    uint32_t last = num_nodes - 1u;
    out_node_last[0] = node_needs_var[last];
    out_clause_last[0] = clause_counts[last];
    out_lit_last[0] = lit_counts[last];
}

extern "C" __global__ void cnf_compute_leaf_choice_totals(
    const uint32_t* __restrict__ leaf_prefix,
    const uint32_t* __restrict__ leaf_used,
    uint32_t leaf_cap,
    const uint32_t* __restrict__ choice_prefix,
    const uint32_t* __restrict__ choice_used,
    uint32_t choice_cap,
    uint32_t* __restrict__ out_num_leaf,
    uint32_t* __restrict__ out_num_choice,
    uint32_t* __restrict__ out_base_choice,
    uint32_t* __restrict__ out_base_node,
    uint32_t* __restrict__ out_decision_var_limit
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    uint32_t leaf_total = 0u;
    if (leaf_cap > 0u) {
        uint32_t last = leaf_cap - 1u;
        leaf_total = leaf_prefix[last] + leaf_used[last];
    }
    uint32_t choice_total = 0u;
    if (choice_cap > 0u) {
        uint32_t last = choice_cap - 1u;
        choice_total = choice_prefix[last] + choice_used[last];
    }

    uint64_t base_choice = static_cast<uint64_t>(leaf_total) + 1ull;
    uint64_t base_node = base_choice + static_cast<uint64_t>(choice_total);

    if (base_choice > 0xffffffffull || base_node > 0xffffffffull) {
        cnf_trap();
    }

    out_num_leaf[0] = leaf_total;
    out_num_choice[0] = choice_total;
    out_base_choice[0] = static_cast<uint32_t>(base_choice);
    out_base_node[0] = static_cast<uint32_t>(base_node);
    out_decision_var_limit[0] = static_cast<uint32_t>(base_node - 1ull);
}

extern "C" __global__ void cnf_compute_totals(
    const uint32_t* __restrict__ node_prefix,
    const uint32_t* __restrict__ clause_base,
    const uint32_t* __restrict__ lit_base,
    const uint32_t* __restrict__ node_last,
    const uint32_t* __restrict__ clause_last,
    const uint32_t* __restrict__ lit_last,
    uint32_t num_nodes,
    const uint32_t* __restrict__ base_node,
    uint32_t var_cap,
    uint32_t clause_cap,
    uint32_t lit_cap,
    uint32_t* __restrict__ out_num_vars,
    uint32_t* __restrict__ out_num_clauses,
    uint32_t* __restrict__ out_num_lits
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    uint32_t base = base_node[0];
    if (base == 0u) {
        cnf_trap();
    }
    if (num_nodes == 0u) {
        uint32_t base_vars = base - 1u;
        if (base_vars > var_cap) {
            cnf_trap();
        }
        out_num_vars[0] = base_vars;
        out_num_clauses[0] = 0u;
        out_num_lits[0] = 0u;
        return;
    }

    uint32_t last = num_nodes - 1u;
    uint32_t node_total = node_prefix[last] + node_last[0];
    uint32_t clause_total = clause_base[last] + clause_last[0];
    uint32_t lit_total = lit_base[last] + lit_last[0];

    if (node_total > num_nodes || clause_total > clause_cap || lit_total > lit_cap) {
        cnf_trap();
    }

    uint64_t num_vars64 = static_cast<uint64_t>(base - 1u) + static_cast<uint64_t>(node_total);
    if (num_vars64 > static_cast<uint64_t>(var_cap)) {
        cnf_trap();
    }

    out_num_vars[0] = static_cast<uint32_t>(num_vars64);
    out_num_clauses[0] = clause_total;
    out_num_lits[0] = lit_total;
}

extern "C" __global__ void cnf_assign_node_var(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ leaf_id,
    const uint32_t* __restrict__ reachable,
    const uint32_t* __restrict__ node_prefix,
    const uint32_t* __restrict__ base_node,
    uint32_t num_nodes,
    uint32_t leaf_cap,
    const uint32_t* __restrict__ leaf_var,
    uint32_t* __restrict__ node_var
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base = base_node[0];
    for (uint32_t i = tid; i < num_nodes; i += blockDim.x * gridDim.x) {
        if (reachable[i] == 0u) {
            node_var[i] = 0u;
            continue;
        }
        uint8_t tag = node_type[i];
        if (tag == PIR_LIT) {
            uint32_t leaf = leaf_id[i];
            if (leaf >= leaf_cap) {
                cnf_trap();
            }
            uint32_t v = leaf_var[leaf];
            if (v == 0u) {
                cnf_trap();
            }
            node_var[i] = v;
        } else {
            node_var[i] = base + node_prefix[i];
        }
    }
}

extern "C" __global__ void cnf_emit_clauses(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ children,
    const uint32_t* __restrict__ leaf_id,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ reachable,
    const uint32_t* __restrict__ node_var,
    const uint32_t* __restrict__ leaf_var,
    const uint32_t* __restrict__ choice_var,
    const uint32_t* __restrict__ clause_base,
    const uint32_t* __restrict__ lit_base,
    uint32_t num_nodes,
    uint32_t leaf_cap,
    uint32_t choice_cap,
    uint32_t* __restrict__ clause_offsets,
    int32_t* __restrict__ literals
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t i = tid; i < num_nodes; i += blockDim.x * gridDim.x) {
        if (reachable[i] == 0u) {
            continue;
        }
        uint8_t tag = node_type[i];
        if (tag == PIR_LIT) {
            continue;
        }

        uint32_t c0 = clause_base[i];
        uint32_t l0 = lit_base[i];
        int32_t v = static_cast<int32_t>(node_var[i]);

        if (tag == PIR_CONST) {
            clause_offsets[c0] = l0;
            uint32_t val = leaf_id[i];
            literals[l0] = (val != 0u) ? v : -v;
            continue;
        }

        if (tag == PIR_NEG_LIT) {
            uint32_t leaf = leaf_id[i];
            if (leaf >= leaf_cap) {
                cnf_trap();
            }
            int32_t leaf_v = static_cast<int32_t>(leaf_var[leaf]);
            if (leaf_v == 0) {
                cnf_trap();
            }
            clause_offsets[c0] = l0;
            clause_offsets[c0 + 1u] = l0 + 2u;
            // (v ∨ leaf_v)
            literals[l0] = v;
            literals[l0 + 1u] = leaf_v;
            // (¬v ∨ ¬leaf_v)
            literals[l0 + 2u] = -v;
            literals[l0 + 3u] = -leaf_v;
            continue;
        }

        if (tag == PIR_AND) {
            uint32_t start = child_offsets[i];
            uint32_t end = child_offsets[i + 1u];
            uint32_t deg = end - start;
            if (deg == 0u) {
                clause_offsets[c0] = l0;
                literals[l0] = v;
                continue;
            }
            for (uint32_t j = 0; j < deg; j++) {
                uint32_t child = children[start + j];
                if (child >= num_nodes) {
                    cnf_trap();
                }
                int32_t c = static_cast<int32_t>(node_var[child]);
                clause_offsets[c0 + j] = l0 + 2u * j;
                literals[l0 + 2u * j] = -v;
                literals[l0 + 2u * j + 1u] = c;
            }
            uint32_t long_base = l0 + 2u * deg;
            clause_offsets[c0 + deg] = long_base;
            for (uint32_t j = 0; j < deg; j++) {
                uint32_t child = children[start + j];
                if (child >= num_nodes) {
                    cnf_trap();
                }
                int32_t c = static_cast<int32_t>(node_var[child]);
                literals[long_base + j] = -c;
            }
            literals[long_base + deg] = v;
            continue;
        }

        if (tag == PIR_OR) {
            uint32_t start = child_offsets[i];
            uint32_t end = child_offsets[i + 1u];
            uint32_t deg = end - start;
            if (deg == 0u) {
                clause_offsets[c0] = l0;
                literals[l0] = -v;
                continue;
            }
            for (uint32_t j = 0; j < deg; j++) {
                uint32_t child = children[start + j];
                if (child >= num_nodes) {
                    cnf_trap();
                }
                int32_t c = static_cast<int32_t>(node_var[child]);
                clause_offsets[c0 + j] = l0 + 2u * j;
                literals[l0 + 2u * j] = -c;
                literals[l0 + 2u * j + 1u] = v;
            }
            uint32_t long_base = l0 + 2u * deg;
            clause_offsets[c0 + deg] = long_base;
            literals[long_base] = -v;
            for (uint32_t j = 0; j < deg; j++) {
                uint32_t child = children[start + j];
                if (child >= num_nodes) {
                    cnf_trap();
                }
                int32_t c = static_cast<int32_t>(node_var[child]);
                literals[long_base + 1u + j] = c;
            }
            continue;
        }

        if (tag == PIR_DECISION) {
            uint32_t var = decision_var[i];
            if (var >= choice_cap) {
                cnf_trap();
            }
            int32_t x = static_cast<int32_t>(choice_var[var]);
            if (x == 0) {
                cnf_trap();
            }
            uint32_t f_node = decision_child_false[i];
            uint32_t t_node = decision_child_true[i];
            if (f_node >= num_nodes || t_node >= num_nodes) {
                cnf_trap();
            }
            int32_t f = static_cast<int32_t>(node_var[f_node]);
            int32_t t = static_cast<int32_t>(node_var[t_node]);

            clause_offsets[c0] = l0;
            clause_offsets[c0 + 1u] = l0 + 3u;
            clause_offsets[c0 + 2u] = l0 + 6u;
            clause_offsets[c0 + 3u] = l0 + 9u;

            // (¬x ∨ ¬t ∨ v)
            literals[l0] = -x;
            literals[l0 + 1u] = -t;
            literals[l0 + 2u] = v;
            // (x ∨ ¬f ∨ v)
            literals[l0 + 3u] = x;
            literals[l0 + 4u] = -f;
            literals[l0 + 5u] = v;
            // (¬x ∨ t ∨ ¬v)
            literals[l0 + 6u] = -x;
            literals[l0 + 7u] = t;
            literals[l0 + 8u] = -v;
            // (x ∨ f ∨ ¬v)
            literals[l0 + 9u] = x;
            literals[l0 + 10u] = f;
            literals[l0 + 11u] = -v;
            continue;
        }

        cnf_trap();
    }
}

extern "C" __global__ void cnf_set_clause_end(
    uint32_t* __restrict__ clause_offsets,
    const uint32_t* __restrict__ num_clauses,
    const uint32_t* __restrict__ num_lits
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    clause_offsets[num_clauses[0]] = num_lits[0];
}
