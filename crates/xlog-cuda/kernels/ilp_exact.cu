#include <stdint.h>

// ilp_exact — M8 Phase 1 bounded exact-induction scoring kernel.
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
__device__ inline uint32_t ilp_exact_matches(
    uint32_t topology,
    const uint64_t* L_arg0, const uint64_t* L_arg1, uint32_t L_rows,
    const uint64_t* R_arg0, const uint64_t* R_arg1, uint32_t R_rows,
    uint64_t qx, uint64_t qy
) {
    if (topology == ILP_EXACT_TOPO_CHAIN) {
        for (uint32_t i = 0; i < L_rows; i++) {
            if (L_arg0[i] != qx) continue;
            uint64_t z = L_arg1[i];
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
        local_pos += ilp_exact_matches(
            topology, L_a0, L_a1, L_rows, R_a0, R_a1, R_rows,
            pos_arg0[q], pos_arg1[q]);
    }
    uint32_t local_neg = 0u;
    for (uint32_t q = tid; q < num_neg; q += blockDim.x) {
        local_neg += ilp_exact_matches(
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
