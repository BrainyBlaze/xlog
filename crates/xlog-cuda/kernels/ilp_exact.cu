#include <stdint.h>

// ilp_exact — native bounded exact-induction scoring kernel.
//
// For a single `induce_exact` call, score every (topology, L, R) triple in
// one launch. Each block owns one triple (topology=blockIdx.z, L=blockIdx.x,
// R=blockIdx.y) and cooperatively evaluates the topology's membership
// predicate against every positive and negative query fact.
//
// Design note: docs/plans/2026-04-17-m8-ilp-exact-kernel-design.md.

#define ILP_EXACT_BLOCK_SIZE 256u

// Topology IDs — must match `xlog_induce::Topology::ALL` ordering
// (chain, star, fanout, fanin).
#define ILP_EXACT_TOPO_CHAIN  0u
#define ILP_EXACT_TOPO_STAR   1u
#define ILP_EXACT_TOPO_FANOUT 2u
#define ILP_EXACT_TOPO_FANIN  3u

// Coverage predicate for a single (topology, query_pair) against a
// left/right candidate relation. Returns 1u if the head rule covers the
// query pair, 0u otherwise.
//
// Topology templates (docs/plans/2026-04-17-m8-ilp-exact-kernel-design.md):
//   chain:  H(X,Y) :- L(X,Z), R(Z,Y)   ∃ z . (qx, z) ∈ L ∧ (z, qy) ∈ R
//   star:   H(X,Y) :- L(X,Y), R(X,Y)   (qx, qy) ∈ L ∧ (qx, qy) ∈ R
//   fanout: H(X,Y) :- L(X,Z), R(X,Y)   ∃ _ . (qx, _) ∈ L ∧ (qx, qy) ∈ R
//   fanin:  H(X,Y) :- L(X,Y), R(Z,Y)   (qx, qy) ∈ L ∧ ∃ _ . (_, qy) ∈ R
template <typename PairT>
__device__ inline uint32_t ilp_exact_matches_t(
    uint32_t topology,
    const PairT* L_arg0, const PairT* L_arg1, uint32_t L_rows,
    const PairT* R_arg0, const PairT* R_arg1, uint32_t R_rows,
    PairT qx, PairT qy
) {
    if (topology == ILP_EXACT_TOPO_CHAIN) {
        for (uint32_t i = 0; i < L_rows; i++) {
            if (L_arg0[i] != qx) continue;
            PairT z = L_arg1[i];
            for (uint32_t j = 0; j < R_rows; j++) {
                if (R_arg0[j] == z && R_arg1[j] == qy) return 1u;
            }
        }
        return 0u;
    }
    if (topology == ILP_EXACT_TOPO_STAR) {
        uint32_t in_L = 0u;
        for (uint32_t i = 0; i < L_rows; i++) {
            if (L_arg0[i] == qx && L_arg1[i] == qy) { in_L = 1u; break; }
        }
        if (!in_L) return 0u;
        for (uint32_t j = 0; j < R_rows; j++) {
            if (R_arg0[j] == qx && R_arg1[j] == qy) return 1u;
        }
        return 0u;
    }
    if (topology == ILP_EXACT_TOPO_FANOUT) {
        uint32_t qx_in_L_arg0 = 0u;
        for (uint32_t i = 0; i < L_rows; i++) {
            if (L_arg0[i] == qx) { qx_in_L_arg0 = 1u; break; }
        }
        if (!qx_in_L_arg0) return 0u;
        for (uint32_t j = 0; j < R_rows; j++) {
            if (R_arg0[j] == qx && R_arg1[j] == qy) return 1u;
        }
        return 0u;
    }
    // ILP_EXACT_TOPO_FANIN
    uint32_t qxqy_in_L = 0u;
    for (uint32_t i = 0; i < L_rows; i++) {
        if (L_arg0[i] == qx && L_arg1[i] == qy) { qxqy_in_L = 1u; break; }
    }
    if (!qxqy_in_L) return 0u;
    for (uint32_t j = 0; j < R_rows; j++) {
        if (R_arg1[j] == qy) return 1u;
    }
    return 0u;
}

template <typename PairT>
__device__ inline uint32_t ilp_exact_count_chain_smem(
    const PairT* L_arg0, const PairT* L_arg1, uint32_t L_rows,
    const PairT* R_arg0, const PairT* R_arg1, uint32_t R_rows,
    const PairT* query_arg0, const PairT* query_arg1, uint32_t num_queries,
    PairT* shared_L_arg0, PairT* shared_L_arg1, uint32_t* shared_match
) {
    uint32_t tid = threadIdx.x;
    uint32_t local_count = 0u;
    uint32_t tile_capacity = blockDim.x;

    for (uint32_t q = 0; q < num_queries; q++) {
        PairT qx = query_arg0[q];
        PairT qy = query_arg1[q];
        uint32_t matched = 0u;

        for (uint32_t tile = 0; tile < L_rows; tile += tile_capacity) {
            uint32_t tile_rows = L_rows - tile;
            if (tile_rows > tile_capacity) tile_rows = tile_capacity;

            for (uint32_t i = tid; i < tile_rows; i += blockDim.x) {
                shared_L_arg0[i] = L_arg0[tile + i];
                shared_L_arg1[i] = L_arg1[tile + i];
            }
            __syncthreads();

            uint32_t thread_match = 0u;
            if (!matched) {
                for (uint32_t i = tid; i < tile_rows && !thread_match; i += blockDim.x) {
                    if (shared_L_arg0[i] != qx) continue;
                    PairT z = shared_L_arg1[i];
                    for (uint32_t j = 0; j < R_rows; j++) {
                        if (R_arg0[j] == z && R_arg1[j] == qy) {
                            thread_match = 1u;
                            break;
                        }
                    }
                }
            }
            shared_match[tid] = thread_match;
            __syncthreads();

            for (uint32_t s = blockDim.x / 2u; s > 0u; s >>= 1u) {
                if (tid < s) {
                    shared_match[tid] |= shared_match[tid + s];
                }
                __syncthreads();
            }
            if (shared_match[0]) matched = 1u;
            __syncthreads();
        }

        if (tid == 0u) local_count += matched;
    }

    return local_count;
}

// Batched scoring kernel.
//
// Launch geometry:
//   grid  = (C, C, 4)          — (L, R, topology)
//   block = (ILP_EXACT_BLOCK_SIZE, 1, 1)
//
// Each block writes exactly one slot each in `pos_covered` and `neg_covered`:
//   slot = topology * (C * C) + L * C + R
//
// Host-side invariants (validated in xlog-induce):
//   cand_arg0 / cand_arg1 : concatenated u64 columns, total `cand_offsets[C]` rows.
//   cand_offsets          : exclusive prefix sum; cand_offsets[0] = 0, cand_offsets[C] = total_rows.
//   pos_arg0 / pos_arg1   : u64 columns, num_pos rows.
//   neg_arg0 / neg_arg1   : u64 columns, num_neg rows (may be 0).
//   pos_covered / neg_covered : u32 arrays of length 4 * C * C, caller-zeroed not required
//                              (kernel writes every slot exactly once).
extern "C" __global__ void ilp_exact_score(
    const uint64_t* cand_arg0,
    const uint64_t* cand_arg1,
    const uint32_t* cand_offsets,
    uint32_t C,
    const uint64_t* pos_arg0,
    const uint64_t* pos_arg1,
    uint32_t num_pos,
    const uint64_t* neg_arg0,
    const uint64_t* neg_arg1,
    uint32_t num_neg,
    uint32_t* pos_covered,
    uint32_t* neg_covered
) {
    uint32_t L = blockIdx.x;
    uint32_t R = blockIdx.y;
    uint32_t topology = blockIdx.z;
    uint32_t tid = threadIdx.x;

    // Relation row slices.
    uint32_t L_off = cand_offsets[L];
    uint32_t L_rows = cand_offsets[L + 1] - L_off;
    uint32_t R_off = cand_offsets[R];
    uint32_t R_rows = cand_offsets[R + 1] - R_off;

    const uint64_t* L_a0 = cand_arg0 + L_off;
    const uint64_t* L_a1 = cand_arg1 + L_off;
    const uint64_t* R_a0 = cand_arg0 + R_off;
    const uint64_t* R_a1 = cand_arg1 + R_off;

    // Per-thread stripe over queries (positives then negatives).
    uint32_t local_pos = 0u;
    for (uint32_t q = tid; q < num_pos; q += blockDim.x) {
        local_pos += ilp_exact_matches_t<uint64_t>(
            topology, L_a0, L_a1, L_rows, R_a0, R_a1, R_rows,
            pos_arg0[q], pos_arg1[q]);
    }
    uint32_t local_neg = 0u;
    for (uint32_t q = tid; q < num_neg; q += blockDim.x) {
        local_neg += ilp_exact_matches_t<uint64_t>(
            topology, L_a0, L_a1, L_rows, R_a0, R_a1, R_rows,
            neg_arg0[q], neg_arg1[q]);
    }

    // Block reduction (pair-halving, sequential — deterministic by
    // construction).
    __shared__ uint32_t pos_scratch[ILP_EXACT_BLOCK_SIZE];
    __shared__ uint32_t neg_scratch[ILP_EXACT_BLOCK_SIZE];
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
        uint32_t slot = topology * (C * C) + L * C + R;
        pos_covered[slot] = pos_scratch[0];
        neg_covered[slot] = neg_scratch[0];
    }
}

extern "C" __global__ void ilp_exact_score_u32(
    const uint32_t* cand_arg0,
    const uint32_t* cand_arg1,
    const uint32_t* cand_offsets,
    uint32_t C,
    const uint32_t* pos_arg0,
    const uint32_t* pos_arg1,
    uint32_t num_pos,
    const uint32_t* neg_arg0,
    const uint32_t* neg_arg1,
    uint32_t num_neg,
    uint32_t* pos_covered,
    uint32_t* neg_covered
) {
    uint32_t L = blockIdx.x;
    uint32_t R = blockIdx.y;
    uint32_t topology = blockIdx.z;
    uint32_t tid = threadIdx.x;

    uint32_t L_off = cand_offsets[L];
    uint32_t L_rows = cand_offsets[L + 1] - L_off;
    uint32_t R_off = cand_offsets[R];
    uint32_t R_rows = cand_offsets[R + 1] - R_off;

    const uint32_t* L_a0 = cand_arg0 + L_off;
    const uint32_t* L_a1 = cand_arg1 + L_off;
    const uint32_t* R_a0 = cand_arg0 + R_off;
    const uint32_t* R_a1 = cand_arg1 + R_off;

    uint32_t local_pos = 0u;
    for (uint32_t q = tid; q < num_pos; q += blockDim.x) {
        local_pos += ilp_exact_matches_t<uint32_t>(
            topology, L_a0, L_a1, L_rows, R_a0, R_a1, R_rows,
            pos_arg0[q], pos_arg1[q]);
    }
    uint32_t local_neg = 0u;
    for (uint32_t q = tid; q < num_neg; q += blockDim.x) {
        local_neg += ilp_exact_matches_t<uint32_t>(
            topology, L_a0, L_a1, L_rows, R_a0, R_a1, R_rows,
            neg_arg0[q], neg_arg1[q]);
    }

    __shared__ uint32_t pos_scratch[ILP_EXACT_BLOCK_SIZE];
    __shared__ uint32_t neg_scratch[ILP_EXACT_BLOCK_SIZE];
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
        uint32_t slot = topology * (C * C) + L * C + R;
        pos_covered[slot] = pos_scratch[0];
        neg_covered[slot] = neg_scratch[0];
    }
}

extern "C" __global__ void ilp_exact_score_chain_smem(
    const uint64_t* cand_arg0,
    const uint64_t* cand_arg1,
    const uint32_t* cand_offsets,
    uint32_t C,
    const uint64_t* pos_arg0,
    const uint64_t* pos_arg1,
    uint32_t num_pos,
    const uint64_t* neg_arg0,
    const uint64_t* neg_arg1,
    uint32_t num_neg,
    uint32_t* pos_covered,
    uint32_t* neg_covered
) {
    uint32_t L = blockIdx.x;
    uint32_t R = blockIdx.y;
    uint32_t topology = blockIdx.z;
    uint32_t tid = threadIdx.x;

    uint32_t L_off = cand_offsets[L];
    uint32_t L_rows = cand_offsets[L + 1] - L_off;
    uint32_t R_off = cand_offsets[R];
    uint32_t R_rows = cand_offsets[R + 1] - R_off;

    const uint64_t* L_a0 = cand_arg0 + L_off;
    const uint64_t* L_a1 = cand_arg1 + L_off;
    const uint64_t* R_a0 = cand_arg0 + R_off;
    const uint64_t* R_a1 = cand_arg1 + R_off;

    extern __shared__ unsigned char ilp_exact_chain_smem_bytes[];
    uint64_t* shared_L_arg0 = reinterpret_cast<uint64_t*>(ilp_exact_chain_smem_bytes);
    uint64_t* shared_L_arg1 = shared_L_arg0 + blockDim.x;
    uint32_t* shared_match = reinterpret_cast<uint32_t*>(shared_L_arg1 + blockDim.x);

    uint32_t local_pos = 0u;
    uint32_t local_neg = 0u;
    if (topology == ILP_EXACT_TOPO_CHAIN) {
        local_pos = ilp_exact_count_chain_smem<uint64_t>(
            L_a0, L_a1, L_rows, R_a0, R_a1, R_rows,
            pos_arg0, pos_arg1, num_pos, shared_L_arg0, shared_L_arg1, shared_match);
        local_neg = ilp_exact_count_chain_smem<uint64_t>(
            L_a0, L_a1, L_rows, R_a0, R_a1, R_rows,
            neg_arg0, neg_arg1, num_neg, shared_L_arg0, shared_L_arg1, shared_match);
    } else {
        for (uint32_t q = tid; q < num_pos; q += blockDim.x) {
            local_pos += ilp_exact_matches_t<uint64_t>(
                topology, L_a0, L_a1, L_rows, R_a0, R_a1, R_rows,
                pos_arg0[q], pos_arg1[q]);
        }
        for (uint32_t q = tid; q < num_neg; q += blockDim.x) {
            local_neg += ilp_exact_matches_t<uint64_t>(
                topology, L_a0, L_a1, L_rows, R_a0, R_a1, R_rows,
                neg_arg0[q], neg_arg1[q]);
        }
    }

    __shared__ uint32_t pos_scratch[ILP_EXACT_BLOCK_SIZE];
    __shared__ uint32_t neg_scratch[ILP_EXACT_BLOCK_SIZE];
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
        uint32_t slot = topology * (C * C) + L * C + R;
        pos_covered[slot] = pos_scratch[0];
        neg_covered[slot] = neg_scratch[0];
    }
}

extern "C" __global__ void ilp_exact_score_chain_smem_u32(
    const uint32_t* cand_arg0,
    const uint32_t* cand_arg1,
    const uint32_t* cand_offsets,
    uint32_t C,
    const uint32_t* pos_arg0,
    const uint32_t* pos_arg1,
    uint32_t num_pos,
    const uint32_t* neg_arg0,
    const uint32_t* neg_arg1,
    uint32_t num_neg,
    uint32_t* pos_covered,
    uint32_t* neg_covered
) {
    uint32_t L = blockIdx.x;
    uint32_t R = blockIdx.y;
    uint32_t topology = blockIdx.z;
    uint32_t tid = threadIdx.x;

    uint32_t L_off = cand_offsets[L];
    uint32_t L_rows = cand_offsets[L + 1] - L_off;
    uint32_t R_off = cand_offsets[R];
    uint32_t R_rows = cand_offsets[R + 1] - R_off;

    const uint32_t* L_a0 = cand_arg0 + L_off;
    const uint32_t* L_a1 = cand_arg1 + L_off;
    const uint32_t* R_a0 = cand_arg0 + R_off;
    const uint32_t* R_a1 = cand_arg1 + R_off;

    extern __shared__ unsigned char ilp_exact_chain_smem_bytes[];
    uint32_t* shared_L_arg0 = reinterpret_cast<uint32_t*>(ilp_exact_chain_smem_bytes);
    uint32_t* shared_L_arg1 = shared_L_arg0 + blockDim.x;
    uint32_t* shared_match = shared_L_arg1 + blockDim.x;

    uint32_t local_pos = 0u;
    uint32_t local_neg = 0u;
    if (topology == ILP_EXACT_TOPO_CHAIN) {
        local_pos = ilp_exact_count_chain_smem<uint32_t>(
            L_a0, L_a1, L_rows, R_a0, R_a1, R_rows,
            pos_arg0, pos_arg1, num_pos, shared_L_arg0, shared_L_arg1, shared_match);
        local_neg = ilp_exact_count_chain_smem<uint32_t>(
            L_a0, L_a1, L_rows, R_a0, R_a1, R_rows,
            neg_arg0, neg_arg1, num_neg, shared_L_arg0, shared_L_arg1, shared_match);
    } else {
        for (uint32_t q = tid; q < num_pos; q += blockDim.x) {
            local_pos += ilp_exact_matches_t<uint32_t>(
                topology, L_a0, L_a1, L_rows, R_a0, R_a1, R_rows,
                pos_arg0[q], pos_arg1[q]);
        }
        for (uint32_t q = tid; q < num_neg; q += blockDim.x) {
            local_neg += ilp_exact_matches_t<uint32_t>(
                topology, L_a0, L_a1, L_rows, R_a0, R_a1, R_rows,
                neg_arg0[q], neg_arg1[q]);
        }
    }

    __shared__ uint32_t pos_scratch[ILP_EXACT_BLOCK_SIZE];
    __shared__ uint32_t neg_scratch[ILP_EXACT_BLOCK_SIZE];
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
        uint32_t slot = topology * (C * C) + L * C + R;
        pos_covered[slot] = pos_scratch[0];
        neg_covered[slot] = neg_scratch[0];
    }
}

#define ILP_EXACT_TOPK_FIELDS 9u

struct IlpExactRankedCandidate {
    uint32_t valid;
    uint32_t left;
    uint32_t right;
    uint32_t pos;
    uint32_t neg;
};

__device__ inline uint32_t ilp_exact_key_before(
    uint32_t a_pos,
    uint32_t a_neg,
    uint32_t a_left,
    uint32_t a_right,
    uint32_t b_pos,
    uint32_t b_neg,
    uint32_t b_left,
    uint32_t b_right
) {
    if (a_pos != b_pos) return a_pos > b_pos;
    if (a_neg != b_neg) return a_neg < b_neg;
    if (a_left != b_left) return a_left < b_left;
    return a_right < b_right;
}

__device__ inline IlpExactRankedCandidate ilp_exact_find_ranked_candidate(
    const uint32_t* pos_covered,
    const uint32_t* neg_covered,
    uint32_t C,
    uint32_t topology,
    uint32_t target_rank
) {
    IlpExactRankedCandidate previous;
    previous.valid = 0u;
    previous.left = 0u;
    previous.right = 0u;
    previous.pos = 0u;
    previous.neg = 0u;

    IlpExactRankedCandidate best;
    best.valid = 0u;
    best.left = 0u;
    best.right = 0u;
    best.pos = 0u;
    best.neg = 0u;

    for (uint32_t rank = 0u; rank <= target_rank; rank++) {
        best.valid = 0u;
        for (uint32_t left = 0u; left < C; left++) {
            for (uint32_t right = 0u; right < C; right++) {
                uint32_t slot = topology * (C * C) + left * C + right;
                uint32_t pos = pos_covered[slot];
                uint32_t neg = neg_covered[slot];
                if (pos == 0u) continue;
                if (previous.valid &&
                    !ilp_exact_key_before(
                        previous.pos, previous.neg, previous.left, previous.right,
                        pos, neg, left, right)) {
                    continue;
                }
                if (!best.valid ||
                    ilp_exact_key_before(
                        pos, neg, left, right,
                        best.pos, best.neg, best.left, best.right)) {
                    best.valid = 1u;
                    best.left = left;
                    best.right = right;
                    best.pos = pos;
                    best.neg = neg;
                }
            }
        }
        if (!best.valid) return best;
        previous = best;
    }

    return best;
}

extern "C" __global__ void ilp_exact_select_topk(
    const uint32_t* pos_covered,
    const uint32_t* neg_covered,
    uint32_t C,
    uint32_t k_per_topology,
    uint32_t* selected
) {
    uint32_t topology = blockIdx.x;
    if (topology >= 4u) return;

    for (uint32_t rank = 0u; rank < k_per_topology; rank++) {
        uint32_t out = (topology * k_per_topology + rank) * ILP_EXACT_TOPK_FIELDS;
        for (uint32_t field = 0u; field < ILP_EXACT_TOPK_FIELDS; field++) {
            selected[out + field] = 0u;
        }

        IlpExactRankedCandidate current =
            ilp_exact_find_ranked_candidate(pos_covered, neg_covered, C, topology, rank);
        if (!current.valid) continue;

        IlpExactRankedCandidate next =
            ilp_exact_find_ranked_candidate(pos_covered, neg_covered, C, topology, rank + 1u);
        uint32_t tie_count = 0u;
        for (uint32_t left = 0u; left < C; left++) {
            for (uint32_t right = 0u; right < C; right++) {
                uint32_t slot = topology * (C * C) + left * C + right;
                if (pos_covered[slot] == current.pos && neg_covered[slot] == current.neg) {
                    tie_count++;
                }
            }
        }

        selected[out + 0u] = topology;
        selected[out + 1u] = current.left;
        selected[out + 2u] = current.right;
        selected[out + 3u] = current.pos;
        selected[out + 4u] = current.neg;
        selected[out + 5u] = rank;
        selected[out + 6u] = next.valid ? next.pos : 0u;
        selected[out + 7u] = next.valid ? next.neg : 0u;
        selected[out + 8u] = tie_count;
    }
}
