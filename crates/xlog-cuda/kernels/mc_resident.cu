// GPU-resident Datalog/MC execution engine.
//
// One megakernel evaluates ALL Monte Carlo worlds (samples) in a single launch
// with zero host interaction inside the measured region: no host loop over
// samples, no per-sample host kernel sequencing, no host metadata reads.
//
// Representation: world-segmented sparse columnar relations plus a dense
// device-side membership index. The host compiler maps every ground atom of a
// bounded Herbrand universe to a slot id in [0, U). Each world owns TWO sparse
// arena segments (double buffer) with slot/arg0/arg1/arg2 columns and device row
// counts. The dense `rel` sidecar is only a per-world dedup/membership index.
//
// Fixpoint semantics: NAIVE bottom-up with double buffering. Each pass copies
// `cur -> next`, then derives heads reading `cur` and writing `next`. This makes
// the per-pass derivation a pure function of the previous state, so the
// iteration count is deterministic and equals the derivation depth + 1 (the
// final no-change confirmation pass) — required for the device-side fixpoint
// trace (K4). Convergence is detected with a shared change flag; no host read.
//
// Argument packing (to stay within launch-arg limits):
//   cfg[]  : packed scalars + offsets (see indices below)
//   meta[] : concatenation of edb_slots | pf_slot | pf_var | rule_data |
//            q_slot | ev_slot | ev_expected, addressed via cfg offsets
//   rel, samples, query_counts, evidence_count, iter_trace : device buffers

#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

// cfg[] indices
#define CFG_NUM_WORLDS 0
#define CFG_U 1
#define CFG_NUM_VARS 2
#define CFG_MAX_ITERS 3
#define CFG_EDB_OFF 4
#define CFG_EDB_CNT 5
#define CFG_PFSLOT_OFF 6
#define CFG_PFVAR_OFF 7
#define CFG_PF_CNT 8
#define CFG_RULES_OFF 9
#define CFG_NUM_RULES 10
#define CFG_Q_OFF 11
#define CFG_Q_CNT 12
#define CFG_EVSLOT_OFF 13
#define CFG_EVEXP_OFF 14
#define CFG_EV_CNT 15
#define CFG_AD_OFF 16
#define CFG_NUM_ADS 17
#define CFG_BLOCKS_PER_WORLD 18

#define MC_RES_CONST_FLAG 0x80000000u
#define ATOM_REC 6u
#define RULE_REC 27u   // 3 header + head + 3 body atoms
#define MC_RES_MAX_VARS 8u

__device__ __forceinline__ void mc_res_world_sync(uint32_t blocks_per_world)
{
    if (blocks_per_world > 1u) {
        __threadfence();
        cg::this_grid().sync();
        __threadfence();
    } else {
        __syncthreads();
    }
}

// Resolve one encoded atom record (6 u32) to a slot id under assignment `vars`.
//   rec = [base, arity, arg0_spec, arg1_spec, arg2_spec, stride0]
__device__ __forceinline__ uint32_t mc_res_atom_slot(
    const uint32_t* atom, const uint32_t* vars, uint32_t domain)
{
    uint32_t base = atom[0];
    uint32_t arity = atom[1];
    uint32_t stride0 = atom[5];
    uint32_t slot = base;
    if (arity >= 1u) {
        uint32_t a0 = atom[2];
        uint32_t v0 = (a0 & MC_RES_CONST_FLAG) ? (a0 & ~MC_RES_CONST_FLAG) : vars[a0];
        slot += v0 * (arity >= 2u ? stride0 : 1u);
    }
    if (arity >= 2u) {
        uint32_t a1 = atom[3];
        uint32_t v1 = (a1 & MC_RES_CONST_FLAG) ? (a1 & ~MC_RES_CONST_FLAG) : vars[a1];
        slot += v1 * (arity >= 3u ? domain : 1u);
    }
    if (arity >= 3u) {
        uint32_t a2 = atom[4];
        uint32_t v2 = (a2 & MC_RES_CONST_FLAG) ? (a2 & ~MC_RES_CONST_FLAG) : vars[a2];
        slot += v2;
    }
    return slot;
}

__device__ __forceinline__ uint32_t mc_res_atom_slot_count(uint32_t arity, uint32_t domain)
{
    if (arity == 0u) return 1u;
    if (arity == 1u) return domain;
    if (arity == 2u) return domain * domain;
    return domain * domain * domain;
}

__device__ __forceinline__ bool mc_res_slot_belongs_to_atom_relation(
    uint32_t slot, const uint32_t* atom, uint32_t domain)
{
    uint32_t base = atom[0];
    uint32_t arity = atom[1];
    uint32_t count = mc_res_atom_slot_count(arity, domain);
    return slot >= base && slot < base + count;
}

__device__ __forceinline__ bool mc_res_bind_atom_from_slot(
    const uint32_t* atom,
    uint32_t slot,
    uint32_t domain,
    uint32_t* vars,
    uint32_t* bound_mask)
{
    if (!mc_res_slot_belongs_to_atom_relation(slot, atom, domain)) return false;
    uint32_t arity = atom[1];
    uint32_t stride0 = atom[5];
    uint32_t rel = slot - atom[0];
    uint32_t vals[3] = {0u, 0u, 0u};
    if (arity == 1u) {
        vals[0] = rel;
    } else if (arity == 2u) {
        vals[0] = rel / stride0;
        vals[1] = rel % stride0;
    } else if (arity >= 3u) {
        vals[0] = rel / stride0;
        uint32_t rem = rel % stride0;
        vals[1] = rem / domain;
        vals[2] = rem % domain;
    }
    for (uint32_t i = 0; i < arity; i++) {
        uint32_t spec = atom[2 + i];
        uint32_t val = vals[i];
        if (spec & MC_RES_CONST_FLAG) {
            if (val != (spec & ~MC_RES_CONST_FLAG)) return false;
        } else {
            uint32_t bit = 1u << spec;
            if (*bound_mask & bit) {
                if (vars[spec] != val) return false;
            } else {
                vars[spec] = val;
                *bound_mask |= bit;
            }
        }
    }
    return true;
}

__device__ __forceinline__ void mc_res_copy_vars(uint32_t* dst, const uint32_t* src)
{
    #pragma unroll
    for (uint32_t i = 0; i < MC_RES_MAX_VARS; i++) dst[i] = src[i];
}

__device__ __forceinline__ void mc_res_decode_slot_for_atom(
    const uint32_t* atom,
    uint32_t slot,
    uint32_t domain,
    uint32_t* arg0,
    uint32_t* arg1,
    uint32_t* arg2)
{
    uint32_t arity = atom[1];
    uint32_t rel = slot - atom[0];
    *arg0 = 0u;
    *arg1 = 0u;
    *arg2 = 0u;
    if (arity == 1u) {
        *arg0 = rel;
    } else if (arity == 2u) {
        uint32_t stride0 = atom[5];
        *arg0 = rel / stride0;
        *arg1 = rel % stride0;
    } else if (arity >= 3u) {
        uint32_t stride0 = atom[5];
        *arg0 = rel / stride0;
        uint32_t rem = rel % stride0;
        *arg1 = rem / domain;
        *arg2 = rem % domain;
    }
}

__device__ __forceinline__ void mc_res_append_sparse(
    const uint32_t* atom,
    uint32_t slot,
    uint32_t* sparse_slots,
    uint32_t* sparse_arg0,
    uint32_t* sparse_arg1,
    uint32_t* sparse_arg2,
    uint32_t* sparse_count,
    uint32_t* sparse_overflow,
    uint32_t sparse_cap,
    uint32_t domain)
{
    uint32_t pos = atomicAdd(sparse_count, 1u);
    if (pos < sparse_cap) {
        sparse_slots[pos] = slot;
        mc_res_decode_slot_for_atom(atom, slot, domain, &sparse_arg0[pos],
                                    &sparse_arg1[pos], &sparse_arg2[pos]);
    } else {
        atomicExch(sparse_overflow, 1u);
    }
}

__device__ __forceinline__ void mc_res_append_slot_only(
    uint32_t slot,
    uint32_t* sparse_slots,
    uint32_t* sparse_arg0,
    uint32_t* sparse_arg1,
    uint32_t* sparse_arg2,
    uint32_t* sparse_count,
    uint32_t* sparse_overflow,
    uint32_t sparse_cap)
{
    uint32_t pos = atomicAdd(sparse_count, 1u);
    if (pos < sparse_cap) {
        sparse_slots[pos] = slot;
        sparse_arg0[pos] = 0u;
        sparse_arg1[pos] = 0u;
        sparse_arg2[pos] = 0u;
    } else {
        atomicExch(sparse_overflow, 1u);
    }
}

// Apply all rules once, reading bodies from `cur` and writing heads into `next`
// (which the caller has pre-copied from `cur`). Returns count of atoms newly set
// in `next` relative to `cur` (nonzero => not yet converged).
__device__ uint32_t mc_res_apply_rules_pass(
    const uint32_t* cur,
    uint32_t* next,
    const uint32_t* cur_sparse_slots,
    uint32_t cur_sparse_count,
    uint32_t* next_sparse_slots,
    uint32_t* next_sparse_arg0,
    uint32_t* next_sparse_arg1,
    uint32_t* next_sparse_arg2,
    uint32_t* next_sparse_count,
    uint32_t* sparse_overflow,
    uint32_t sparse_cap,
    uint32_t worker_tid,
    uint32_t worker_threads,
    const uint32_t* rules,
    uint32_t num_rules)
{
    uint32_t derived = 0;
    for (uint32_t r = 0; r < num_rules; r++) {
        const uint32_t* rule = rules + (size_t)r * RULE_REC;
        uint32_t n_body = rule[0];
        uint32_t n_vars = rule[1];
        uint32_t domain = rule[2];
        const uint32_t* head = rule + 3;
        const uint32_t* body0 = rule + 3 + ATOM_REC;
        const uint32_t* body1 = rule + 3 + 2u * ATOM_REC;
        const uint32_t* body2 = rule + 3 + 3u * ATOM_REC;

        for (uint32_t idx = worker_tid; idx < cur_sparse_count; idx += worker_threads) {
            uint32_t vars[MC_RES_MAX_VARS] = {0u};
            uint32_t bound = 0u;
            uint32_t slot0 = cur_sparse_slots[idx];
            if (!mc_res_bind_atom_from_slot(body0, slot0, domain, vars, &bound)) continue;

            if (n_body == 1u) {
                uint32_t hs = mc_res_atom_slot(head, vars, domain);
                if (atomicCAS(&next[hs], 0u, 1u) == 0u) {
                    mc_res_append_sparse(head, hs, next_sparse_slots, next_sparse_arg0,
                                         next_sparse_arg1, next_sparse_arg2,
                                         next_sparse_count, sparse_overflow,
                                         sparse_cap, domain);
                    derived++;
                }
                continue;
            }

            for (uint32_t j = 0; j < cur_sparse_count; j++) {
                uint32_t vars1[MC_RES_MAX_VARS] = {0u};
                mc_res_copy_vars(vars1, vars);
                uint32_t bound1 = bound;
                if (!mc_res_bind_atom_from_slot(body1, cur_sparse_slots[j], domain, vars1, &bound1)) continue;
                if (n_body == 2u) {
                    uint32_t hs = mc_res_atom_slot(head, vars1, domain);
                    if (atomicCAS(&next[hs], 0u, 1u) == 0u) {
                        mc_res_append_sparse(head, hs, next_sparse_slots, next_sparse_arg0,
                                             next_sparse_arg1, next_sparse_arg2,
                                             next_sparse_count, sparse_overflow,
                                             sparse_cap, domain);
                        derived++;
                    }
                    continue;
                }
                for (uint32_t k = 0; k < cur_sparse_count; k++) {
                    uint32_t vars2[MC_RES_MAX_VARS] = {0u};
                    mc_res_copy_vars(vars2, vars1);
                    uint32_t bound2 = bound1;
                    if (!mc_res_bind_atom_from_slot(body2, cur_sparse_slots[k], domain, vars2, &bound2)) continue;
                    uint32_t hs = mc_res_atom_slot(head, vars2, domain);
                    if (atomicCAS(&next[hs], 0u, 1u) == 0u) {
                        mc_res_append_sparse(head, hs, next_sparse_slots, next_sparse_arg0,
                                             next_sparse_arg1, next_sparse_arg2,
                                             next_sparse_count, sparse_overflow,
                                             sparse_cap, domain);
                        derived++;
                    }
                }
            }
        }
    }
    return derived;
}

extern "C" __global__ void mc_resident_engine(
    const uint32_t* __restrict__ cfg,
    const uint32_t* __restrict__ meta,
    uint32_t* __restrict__ rel,
    const uint8_t* __restrict__ samples,
    uint32_t* __restrict__ query_counts,
    uint32_t* __restrict__ evidence_count,
    uint32_t* __restrict__ iter_trace,
    uint32_t* __restrict__ sparse_columns,
    uint32_t* __restrict__ sparse_counts,
    uint32_t* __restrict__ sparse_final_counts,
    uint32_t* __restrict__ sparse_offsets,
    uint32_t* __restrict__ resident_status_flags,
    uint32_t sparse_cap)
{
    uint32_t num_worlds = cfg[CFG_NUM_WORLDS];
    uint32_t U = cfg[CFG_U];
    uint32_t num_vars = cfg[CFG_NUM_VARS];
    uint32_t max_iters = cfg[CFG_MAX_ITERS];
    const uint32_t* edb = meta + cfg[CFG_EDB_OFF];
    uint32_t edb_cnt = cfg[CFG_EDB_CNT];
    const uint32_t* pf_slot = meta + cfg[CFG_PFSLOT_OFF];
    const uint32_t* pf_var = meta + cfg[CFG_PFVAR_OFF];
    uint32_t pf_cnt = cfg[CFG_PF_CNT];
    const uint32_t* rules = meta + cfg[CFG_RULES_OFF];
    uint32_t num_rules = cfg[CFG_NUM_RULES];
    const uint32_t* q_slot = meta + cfg[CFG_Q_OFF];
    uint32_t q_cnt = cfg[CFG_Q_CNT];
    const uint32_t* ev_slot = meta + cfg[CFG_EVSLOT_OFF];
    const uint32_t* ev_exp = meta + cfg[CFG_EVEXP_OFF];
    uint32_t ev_cnt = cfg[CFG_EV_CNT];
    const uint32_t* ad_data = meta + cfg[CFG_AD_OFF];
    uint32_t num_ads = cfg[CFG_NUM_ADS];
    uint32_t blocks_per_world = cfg[CFG_BLOCKS_PER_WORLD];

    uint32_t global_block = blockIdx.x;
    uint32_t w = global_block / blocks_per_world;
    uint32_t block_in_world = global_block - w * blocks_per_world;
    uint32_t worker_tid = block_in_world * blockDim.x + threadIdx.x;
    uint32_t worker_threads = blocks_per_world * blockDim.x;

    uint32_t* A = rel + (size_t)w * 2u * U;
    uint32_t* B = A + U;
    uint32_t sparse_total = num_worlds * 2u * sparse_cap;
    uint32_t* sparse_slots = sparse_columns;
    uint32_t* sparse_arg0 = sparse_columns + sparse_total;
    uint32_t* sparse_arg1 = sparse_columns + 2u * sparse_total;
    uint32_t* sparse_arg2 = sparse_columns + 3u * sparse_total;
    uint32_t buf0 = w * 2u;
    uint32_t buf1 = buf0 + 1u;
    uint32_t* A_slots = sparse_slots + (size_t)buf0 * sparse_cap;
    uint32_t* A_arg0 = sparse_arg0 + (size_t)buf0 * sparse_cap;
    uint32_t* A_arg1 = sparse_arg1 + (size_t)buf0 * sparse_cap;
    uint32_t* A_arg2 = sparse_arg2 + (size_t)buf0 * sparse_cap;
    uint32_t* B_slots = sparse_slots + (size_t)buf1 * sparse_cap;
    uint32_t* B_arg0 = sparse_arg0 + (size_t)buf1 * sparse_cap;
    uint32_t* B_arg1 = sparse_arg1 + (size_t)buf1 * sparse_cap;
    uint32_t* B_arg2 = sparse_arg2 + (size_t)buf1 * sparse_cap;
    uint32_t* A_count = sparse_counts + buf0;
    uint32_t* B_count = sparse_counts + buf1;
    uint32_t* converged_flags = resident_status_flags;
    uint32_t* sparse_overflow_flags = resident_status_flags + num_worlds;
    uint32_t* block_participation = resident_status_flags + 2u * num_worlds;
    uint32_t* changed_flags = resident_status_flags + 3u * num_worlds;
    uint32_t* global_continue = resident_status_flags + 4u * num_worlds;
    uint32_t* world_overflow = sparse_overflow_flags + w;
    uint32_t* world_changed = changed_flags + w;
    bool world_leader = (block_in_world == 0u && threadIdx.x == 0u);

    // --- Init buffer A: EDB facts + per-world Bernoulli draws. (B is filled by
    //     the per-pass copy below; for the no-rules case A is the final state.) ---
    for (uint32_t s = worker_tid; s < U; s += worker_threads) A[s] = 0u;
    if (world_leader) {
        *A_count = 0u;
        *B_count = 0u;
        *world_overflow = 0u;
        converged_flags[w] = 0u;
        block_participation[w] = 0u;
        *world_changed = 0u;
        sparse_offsets[w] = w * sparse_cap;
        if (w == 0u) sparse_offsets[num_worlds] = num_worlds * sparse_cap;
    }
    mc_res_world_sync(blocks_per_world);
    if (threadIdx.x == 0) atomicAdd(&block_participation[w], 1u);
    mc_res_world_sync(blocks_per_world);
    for (uint32_t i = worker_tid; i < edb_cnt; i += worker_threads) {
        uint32_t slot = edb[i];
        if (atomicCAS(&A[slot], 0u, 1u) == 0u) {
            mc_res_append_slot_only(slot, A_slots, A_arg0, A_arg1, A_arg2, A_count,
                                    world_overflow, sparse_cap);
        }
    }
    for (uint32_t i = worker_tid; i < pf_cnt; i += worker_threads) {
        uint32_t slot = pf_slot[i];
        if (samples[(size_t)w * num_vars + pf_var[i]]) {
            if (atomicCAS(&A[slot], 0u, 1u) == 0u) {
                mc_res_append_slot_only(slot, A_slots, A_arg0, A_arg1, A_arg2, A_count,
                                        world_overflow, sparse_cap);
            }
        }
    }
    mc_res_world_sync(blocks_per_world);

    // --- Annotated-disjunction / exclusive-choice decode (per world). ---
    // Each AD record: [n_choices, n_dvars, dvar_0..., slot_0...]. Walk the chain
    // of conditional Bernoulli decisions: the first decision var that fires
    // selects its choice; if none fire, the residual outcome (index n_dvars) is
    // either the last choice (no "none" mass) or nothing (n_dvars == n_choices).
    if (world_leader) {
        uint32_t p = 0;
        for (uint32_t a = 0; a < num_ads; a++) {
            uint32_t n_choices = ad_data[p++];
            uint32_t n_dvars = ad_data[p++];
            const uint32_t* dvars = ad_data + p;
            p += n_dvars;
            const uint32_t* slots = ad_data + p;
            p += n_choices;
            uint32_t sel = n_dvars; // residual outcome index
            for (uint32_t i = 0; i < n_dvars; i++) {
                if (samples[(size_t)w * num_vars + dvars[i]]) { sel = i; break; }
            }
            if (sel < n_choices) {
                uint32_t slot = slots[sel];
                if (atomicCAS(&A[slot], 0u, 1u) == 0u) {
                    mc_res_append_slot_only(slot, A_slots, A_arg0, A_arg1, A_arg2, A_count,
                                            world_overflow, sparse_cap);
                }
            }
        }
    }
    mc_res_world_sync(blocks_per_world);

    // --- Device-side bounded NAIVE fixpoint with double buffering. ---
    __shared__ uint32_t s_changed;
    uint32_t* cur = A;
    uint32_t* nxt = B;
    uint32_t* cur_slots = A_slots;
    uint32_t* cur_arg0 = A_arg0;
    uint32_t* cur_arg1 = A_arg1;
    uint32_t* cur_arg2 = A_arg2;
    uint32_t* nxt_slots = B_slots;
    uint32_t* nxt_arg0 = B_arg0;
    uint32_t* nxt_arg1 = B_arg1;
    uint32_t* nxt_arg2 = B_arg2;
    uint32_t* cur_count = A_count;
    uint32_t* nxt_count = B_count;
    uint32_t iters = 0u;
    uint32_t converged = (num_rules == 0u) ? 1u : 0u;
    if (world_leader) converged_flags[w] = converged;
    if (num_rules > 0u) {
        for (uint32_t it = 0; it < max_iters; it++) {
            if (blocks_per_world > 1u && global_block == 0u && threadIdx.x == 0u) {
                *global_continue = 0u;
            }
            mc_res_world_sync(blocks_per_world);

            bool active = (converged_flags[w] == 0u);
            uint32_t local_cur_count = active ? *cur_count : 0u;
            if (active) {
                for (uint32_t s = worker_tid; s < U; s += worker_threads) nxt[s] = cur[s];
                for (uint32_t s = worker_tid; s < local_cur_count; s += worker_threads) {
                    nxt_slots[s] = cur_slots[s];
                    nxt_arg0[s] = cur_arg0[s];
                    nxt_arg1[s] = cur_arg1[s];
                    nxt_arg2[s] = cur_arg2[s];
                }
            }
            if (world_leader && active) {
                s_changed = 0u;
                *world_changed = 0u;
                *nxt_count = local_cur_count;
            }
            mc_res_world_sync(blocks_per_world);

            if (active) {
                uint32_t d = mc_res_apply_rules_pass(
                    cur, nxt,
                    cur_slots, local_cur_count,
                    nxt_slots, nxt_arg0, nxt_arg1, nxt_arg2, nxt_count, world_overflow, sparse_cap,
                    worker_tid, worker_threads,
                    rules, num_rules);
                if (d > 0u) {
                    if (blocks_per_world == 1u) {
                        atomicAdd(&s_changed, d);
                    } else {
                        atomicAdd(world_changed, d);
                    }
                }
            }
            mc_res_world_sync(blocks_per_world);

            uint32_t changed = active
                ? (blocks_per_world == 1u ? s_changed : atomicAdd(world_changed, 0u))
                : 0u;
            if (world_leader && active) {
                iters = it + 1u;
                iter_trace[w] = iters;
                if (changed == 0u) {
                    converged = 1u;
                    converged_flags[w] = 1u;
                } else if (blocks_per_world > 1u) {
                    atomicExch(global_continue, 1u);
                }
            }
            mc_res_world_sync(blocks_per_world);

            if (active && changed > 0u) {
                uint32_t* tmp = cur; cur = nxt; nxt = tmp;
                tmp = cur_slots; cur_slots = nxt_slots; nxt_slots = tmp;
                tmp = cur_arg0; cur_arg0 = nxt_arg0; nxt_arg0 = tmp;
                tmp = cur_arg1; cur_arg1 = nxt_arg1; nxt_arg1 = tmp;
                tmp = cur_arg2; cur_arg2 = nxt_arg2; nxt_arg2 = tmp;
                tmp = cur_count; cur_count = nxt_count; nxt_count = tmp;
            }
            mc_res_world_sync(blocks_per_world);

            if (blocks_per_world == 1u) {
                if (active && changed == 0u) break;
            } else if (atomicAdd(global_continue, 0u) == 0u) {
                break;
            }
        }
    }
    if (world_leader) {
        iter_trace[w] = iters;
        sparse_final_counts[w] = *cur_count;
        converged_flags[w] = converged;
    }
    mc_res_world_sync(blocks_per_world);

    // --- On-device query/evidence counting (reads the final `cur` state). ---
    if (world_leader) {
        uint8_t ok = 1u;
        for (uint32_t i = 0; i < ev_cnt; i++) {
            uint8_t holds = cur[ev_slot[i]] ? 1u : 0u;
            if (holds != (uint8_t)ev_exp[i]) { ok = 0u; break; }
        }
        if (ok) {
            atomicAdd(evidence_count, 1u);
            for (uint32_t i = 0; i < q_cnt; i++) {
                if (cur[q_slot[i]]) atomicAdd(&query_counts[i], 1u);
            }
        }
    }
}
