// GPU-resident Datalog/MC execution engine.
//
// One megakernel evaluates ALL Monte Carlo worlds (samples) in a single launch
// with zero host interaction inside the measured region: no host loop over
// samples, no per-sample host kernel sequencing, no host metadata reads.
//
// Representation: bounded dense boolean. The host compiler maps every ground
// atom of a bounded Herbrand universe to a slot id in [0, U). Each world owns
// TWO contiguous [U] byte rows in `rel` (double buffer for deterministic naive
// fixpoint): world w lives at `rel[w*2*U ..]`, buffer A = +0, buffer B = +U.
// `R[slot] != 0` iff that atom holds in world w. One CUDA block evaluates one
// world; threads stride over slots / variable assignments.
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

#include <cstdint>

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

#define MC_RES_CONST_FLAG 0x80000000u
#define ATOM_REC 5u
#define RULE_REC 18u   // 3 header + 3*ATOM_REC

// Resolve one encoded atom record (5 u32) to a slot id under assignment `vars`.
//   rec = [base, arity, arg0_spec, arg1_spec, stride0]
__device__ __forceinline__ uint32_t mc_res_atom_slot(
    const uint32_t* atom, const uint32_t* vars)
{
    uint32_t base = atom[0];
    uint32_t arity = atom[1];
    uint32_t stride0 = atom[4];
    uint32_t slot = base;
    if (arity >= 1u) {
        uint32_t a0 = atom[2];
        uint32_t v0 = (a0 & MC_RES_CONST_FLAG) ? (a0 & ~MC_RES_CONST_FLAG) : vars[a0];
        slot += v0 * (arity >= 2u ? stride0 : 1u);
    }
    if (arity >= 2u) {
        uint32_t a1 = atom[3];
        uint32_t v1 = (a1 & MC_RES_CONST_FLAG) ? (a1 & ~MC_RES_CONST_FLAG) : vars[a1];
        slot += v1;
    }
    return slot;
}

// Apply all rules once, reading bodies from `cur` and writing heads into `next`
// (which the caller has pre-copied from `cur`). Returns count of atoms newly set
// in `next` relative to `cur` (nonzero => not yet converged).
__device__ uint32_t mc_res_apply_rules_pass(
    const uint8_t* cur, uint8_t* next, const uint32_t* rules, uint32_t num_rules)
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

        uint32_t total = 1u;
        for (uint32_t i = 0; i < n_vars; i++) total *= domain;

        for (uint32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
            uint32_t vars[3] = {0u, 0u, 0u};
            uint32_t t = idx;
            for (uint32_t i = 0; i < n_vars; i++) { vars[i] = t % domain; t /= domain; }

            bool holds = true;
            if (n_body >= 1u) { if (!cur[mc_res_atom_slot(body0, vars)]) holds = false; }
            if (holds && n_body >= 2u) { if (!cur[mc_res_atom_slot(body1, vars)]) holds = false; }
            if (holds) {
                uint32_t hs = mc_res_atom_slot(head, vars);
                // `next` was copied from `cur`, so next[hs]==0 iff cur[hs]==0.
                // Concurrent identical writes are idempotent; over-counting
                // `derived` only matters as a nonzero "changed" signal.
                if (next[hs] == 0u) { next[hs] = 1u; derived++; }
            }
        }
    }
    return derived;
}

extern "C" __global__ void mc_resident_engine(
    const uint32_t* __restrict__ cfg,
    const uint32_t* __restrict__ meta,
    uint8_t* __restrict__ rel,
    const uint8_t* __restrict__ samples,
    uint32_t* __restrict__ query_counts,
    uint32_t* __restrict__ evidence_count,
    uint32_t* __restrict__ iter_trace)
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

    uint32_t w = blockIdx.x;
    if (w >= num_worlds) return;

    uint8_t* A = rel + (size_t)w * 2u * U;
    uint8_t* B = A + U;

    // --- Init buffer A: EDB facts + per-world Bernoulli draws. (B is filled by
    //     the per-pass copy below; for the no-rules case A is the final state.) ---
    for (uint32_t s = threadIdx.x; s < U; s += blockDim.x) A[s] = 0u;
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < edb_cnt; i += blockDim.x) A[edb[i]] = 1u;
    for (uint32_t i = threadIdx.x; i < pf_cnt; i += blockDim.x) {
        A[pf_slot[i]] = samples[(size_t)w * num_vars + pf_var[i]];
    }
    __syncthreads();

    // --- Annotated-disjunction / exclusive-choice decode (per world). ---
    // Each AD record: [n_choices, n_dvars, dvar_0..., slot_0...]. Walk the chain
    // of conditional Bernoulli decisions: the first decision var that fires
    // selects its choice; if none fire, the residual outcome (index n_dvars) is
    // either the last choice (no "none" mass) or nothing (n_dvars == n_choices).
    if (threadIdx.x == 0) {
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
            if (sel < n_choices) A[slots[sel]] = 1u;
        }
    }
    __syncthreads();

    // --- Device-side bounded NAIVE fixpoint with double buffering. ---
    __shared__ uint32_t s_changed;
    uint8_t* cur = A;
    uint8_t* nxt = B;
    uint32_t iters = 0u;
    if (num_rules > 0u) {
        for (uint32_t it = 0; it < max_iters; it++) {
            for (uint32_t s = threadIdx.x; s < U; s += blockDim.x) nxt[s] = cur[s];
            if (threadIdx.x == 0) s_changed = 0u;
            __syncthreads();
            uint32_t d = mc_res_apply_rules_pass(cur, nxt, rules, num_rules);
            if (d > 0u) atomicAdd(&s_changed, d);
            __syncthreads();
            iters = it + 1u;
            if (s_changed == 0u) break;
            uint8_t* tmp = cur; cur = nxt; nxt = tmp;
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) iter_trace[w] = iters;
    __syncthreads();

    // --- On-device query/evidence counting (reads the final `cur` state). ---
    if (threadIdx.x == 0) {
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
