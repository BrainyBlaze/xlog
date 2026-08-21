// kernels/joint_solve.cu
// Joint constraint solve, stage 1: existential relation-label
// feasibility over entity sort domains.
//
// Relation legality is existential over the whole domain product: a
// label stays feasible for a candidate pair while SOME sort in the
// head entity's domain intersects the label's head signature AND some
// sort in the tail entity's domain intersects its tail signature.
// Domains are u64 bitset lanes; signatures are catalog-bound constant
// masks uploaded cold-path. The abstention label is always feasible.

#include <cstdint>

__device__ __forceinline__ uint32_t joint_component_root(
    uint32_t* __restrict__ parents,
    uint32_t candidate
) {
    uint32_t root = candidate;
    while (parents[root] != root) root = parents[root];
    return root;
}

/** Initialize carrier-owned component scratch for one solve. */
extern "C" __global__ void joint_component_plan_init(
    uint32_t* __restrict__ parents,
    uint32_t num_candidates,
    uint32_t* __restrict__ entity_owners,
    uint32_t num_entities,
    uint32_t* __restrict__ component_count,
    unsigned long long* __restrict__ fuel_spent
) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_candidates) parents[index] = index;
    if (index < num_entities) entity_owners[index] = 0xFFFFFFFFu;
    if (index == 0u) {
        component_count[0] = 0u;
        fuel_spent[0] = 0ull;
    }
}

/** Pick one deterministic candidate owner for every referenced entity. */
extern "C" __global__ void joint_component_entity_owners(
    const uint32_t* __restrict__ arguments,
    const uint32_t* __restrict__ argument_arities,
    uint32_t num_candidates,
    uint32_t max_arity,
    uint32_t num_entities,
    uint32_t* __restrict__ entity_owners
) {
    uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= num_candidates) return;
    uint32_t arity = argument_arities[candidate];
    if (arity < 2u || arity > max_arity) return;
    for (uint32_t role = 0; role < arity; ++role) {
        uint32_t entity = arguments[(uint64_t)candidate * max_arity + role];
        if (entity < num_entities) atomicMin(entity_owners + entity, candidate);
    }
}

/** Union candidates sharing any active argument entity. */
extern "C" __global__ void joint_component_union(
    const uint32_t* __restrict__ arguments,
    const uint32_t* __restrict__ argument_arities,
    uint32_t num_candidates,
    uint32_t max_arity,
    uint32_t num_entities,
    const uint32_t* __restrict__ entity_owners,
    uint32_t* __restrict__ parents
) {
    uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= num_candidates) return;
    uint32_t arity = argument_arities[candidate];
    if (arity < 2u || arity > max_arity) return;
    for (uint32_t role = 0; role < arity; ++role) {
        uint32_t entity = arguments[(uint64_t)candidate * max_arity + role];
        if (entity >= num_entities) continue;
        uint32_t owner = entity_owners[entity];
        if (owner == 0xFFFFFFFFu) continue;
        while (true) {
            uint32_t candidate_root = joint_component_root(parents, candidate);
            uint32_t owner_root = joint_component_root(parents, owner);
            if (candidate_root == owner_root) break;
            uint32_t lower = candidate_root < owner_root ? candidate_root : owner_root;
            uint32_t upper = candidate_root < owner_root ? owner_root : candidate_root;
            if (atomicCAS(parents + upper, upper, lower) == upper) break;
        }
    }
}

/** Compress the completed union forest and count component roots. */
extern "C" __global__ void joint_component_compress(
    uint32_t* __restrict__ parents,
    uint32_t num_candidates,
    uint32_t* __restrict__ component_count
) {
    uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= num_candidates) return;
    uint32_t root = joint_component_root(parents, candidate);
    parents[candidate] = root;
    if (root == candidate) atomicAdd(component_count, 1u);
}

/**
 * Per-candidate exact top-two over FEASIBLE labels.
 *
 * Consumes the feasibility stage's output: for each candidate the
 * best feasible label, an ambiguity flag, the best score and the
 * margin (best minus runner-up) are written. A score tie for the
 * maximum sets the ambiguity flag — the winner index is reported for
 * diagnostics but a tied maximum must never emit as a unique MAP
 * label downstream. The result is the exact global max-marginal for
 * a single-candidate component; multi-candidate components need the
 * cross-candidate stage and must not consume these rows as final.
 *
 * One thread owns one candidate row: no atomics, bit-deterministic.
 *
 * An empty feasible row (only produced by a poisoned feasibility
 * row) yields a POISONED map row: [0xFFFFFFFF, 1, 0, 0]. Scores
 * must never be NaN — a producer-contract violation, not a case
 * this stage defines behavior for.
 *
 * @param scores         candidate label scores, candidates x labels f32
 * @param feasible_sets  per-candidate feasible bitmasks,
 *                       candidates x label_words u64
 * @param num_candidates candidate row count
 * @param num_labels     label universe width
 * @param map_results    candidates x 4 u32:
 *                       [best_label, ambiguous_flag,
 *                        best_score_bits, margin_bits]
 *                       (f32 values stored as raw bits)
 */
extern "C" __global__ void joint_label_top2(
    const float* __restrict__ scores,
    const uint64_t* __restrict__ feasible_sets,
    uint32_t num_candidates,
    uint32_t num_labels,
    uint32_t* __restrict__ map_results
) {
    uint32_t cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= num_candidates) return;

    uint32_t label_words = (num_labels + 63u) / 64u;
    const float* row = scores + (uint64_t)cand * num_labels;
    const uint64_t* feasible = feasible_sets + (uint64_t)cand * label_words;

    // The feasibility stage marks abstention unconditionally, so a
    // healthy row always has at least one feasible label; an empty
    // row means the feasibility stage poisoned this candidate.
    float best = -__int_as_float(0x7f800000);  // -inf
    float second = best;
    uint32_t best_label = 0;
    bool tie = false;
    bool any = false;
    for (uint32_t label = 0; label < num_labels; ++label) {
        if (((feasible[label >> 6] >> (label & 63u)) & 1ull) == 0ull) {
            continue;
        }
        any = true;
        float s = row[label];
        if (s > best) {
            second = best;
            best = s;
            best_label = label;
            tie = false;
        } else if (s == best) {
            second = best;
            tie = true;
        } else if (s > second) {
            second = s;
        }
    }

    uint32_t* out = map_results + (uint64_t)cand * 4u;
    if (!any) {
        out[0] = 0xFFFFFFFFu;
        out[1] = 1u;
        out[2] = 0u;
        out[3] = 0u;
        return;
    }
    out[0] = best_label;
    out[1] = tie ? 1u : 0u;
    out[2] = __float_as_uint(best);
    // A sole feasible label has no runner-up: its margin is +inf
    // (unbounded confidence), never a fabricated finite number.
    float margin = second == -__int_as_float(0x7f800000)
        ? __int_as_float(0x7f800000)
        : best - second;
    out[3] = __float_as_uint(margin);
}

/**
 * Per-candidate existential label feasibility.
 *
 * One thread owns one candidate row and writes its entire output row,
 * so the kernel needs no atomics and is bit-deterministic.
 *
 * A candidate whose entity indices fall outside `num_entities` is a
 * corrupt producer record: its row is POISONED (count = 0xFFFFFFFF,
 * feasible set zeroed) instead of reading out-of-bounds device
 * memory.
 *
 * @param domains         entity sort domains, entities x lanes u64
 * @param arguments       padded entity arguments, candidates x max_arity u32
 * @param argument_arities active argument count for each candidate
 * @param role_masks      role-major signatures, max_arity x labels x lanes u64
 * @param max_arity       padded argument width and role capacity
 * @param num_entities    entity capacity; active indices must be below
 * @param num_candidates  candidate row count
 * @param num_labels      label universe width (including abstention)
 * @param lanes           u64 lanes per domain / signature row
 * @param abstain_label   label index that is feasible unconditionally
 * @param feasible_counts per-candidate feasible label count,
 *                        candidates u32; 0xFFFFFFFF = poisoned row
 * @param feasible_sets   per-candidate feasible label bitmask,
 *                        candidates x ceil(labels/64) u64
 */
extern "C" __global__ void joint_label_feasibility(
    const uint64_t* __restrict__ domains,
    const uint32_t* __restrict__ arguments,
    const uint32_t* __restrict__ argument_arities,
    const uint64_t* __restrict__ role_masks,
    uint32_t max_arity,
    uint32_t num_entities,
    uint32_t num_candidates,
    uint32_t num_labels,
    uint32_t lanes,
    uint32_t abstain_label,
    uint32_t* __restrict__ feasible_counts,
    uint64_t* __restrict__ feasible_sets
) {
    uint32_t cand = blockIdx.x * blockDim.x + threadIdx.x;
    if (cand >= num_candidates) return;

    uint32_t label_words = (num_labels + 63u) / 64u;
    uint32_t arity = argument_arities[cand];
    bool corrupt = arity == 0u || arity > max_arity;
    for (uint32_t role = 0; role < arity && !corrupt; ++role) {
        corrupt = arguments[(uint64_t)cand * max_arity + role] >= num_entities;
    }
    if (corrupt) {
        // Corrupt producer record: poison, never read out of bounds.
        for (uint32_t word = 0; word < label_words; ++word) {
            feasible_sets[(uint64_t)cand * label_words + word] = 0ull;
        }
        feasible_counts[cand] = 0xFFFFFFFFu;
        return;
    }
    uint32_t count = 0;
    for (uint32_t word = 0; word < label_words; ++word) {
        uint64_t set = 0;
        uint32_t base = word * 64u;
        uint32_t limit = num_labels - base < 64u ? num_labels - base : 64u;
        for (uint32_t bit = 0; bit < limit; ++bit) {
            uint32_t label = base + bit;
            bool feasible;
            if (label == abstain_label) {
                feasible = true;
            } else {
                feasible = true;
                for (uint32_t role = 0; role < arity && feasible; ++role) {
                    uint32_t entity = arguments[(uint64_t)cand * max_arity + role];
                    const uint64_t* domain = domains + (uint64_t)entity * lanes;
                    const uint64_t* mask = role_masks
                        + ((uint64_t)role * num_labels + label) * lanes;
                    bool hit = false;
                    for (uint32_t lane = 0; lane < lanes && !hit; ++lane) {
                        hit = (domain[lane] & mask[lane]) != 0ull;
                    }
                    feasible = hit;
                }
            }
            if (feasible) {
                set |= 1ull << bit;
                ++count;
            }
        }
        feasible_sets[(uint64_t)cand * label_words + word] = set;
    }
    feasible_counts[cand] = count;
}

/**
 * Exact joint solve of ONE multi-candidate component by complete
 * enumeration of feasible label combinations.
 *
 * One block owns one component; thread 0 enumerates (components
 * eligible for this stage are fuel-bounded small, and sequential
 * enumeration keeps the running-intersection state trivially
 * bit-deterministic). A combination is CONSISTENT when every entity
 * of the component keeps a nonempty domain under the intersection of
 * its role masks across the chosen labels. Because enumeration is
 * COMPLETE, the per-candidate results are exact GLOBAL max-marginals:
 * for each candidate the best consistent total using it, and the
 * best consistent total using any OTHER label — their difference is
 * the edge margin. A zero margin (or a tied optimum) is typed
 * ambiguity on that edge; a component with no consistent combination
 * poisons its rows.
 *
 * @param scores          candidate label scores, candidates x labels f32
 * @param feasible_sets   per-candidate feasible bitmasks,
 *                        candidates x label_words u64
 * @param arguments       padded entity arguments, candidates x max_arity u32
 * @param argument_arities active role count for each candidate
 * @param domains         entity sort domains, entities x lanes u64
 * @param role_masks      role-major signatures, max_arity x labels x lanes u64
 * @param component_parents compressed candidate-to-root mapping
 * @param component_count carrier-owned number of component roots
 * @param num_candidates  candidate row count in this launch
 * @param num_labels      label universe width
 * @param lanes           u64 lanes per domain / signature row
 * @param fuel_authorized whole node-expansion budget; each root
 *                        receives an equal deterministic share;
 *                        a component whose enumeration would exceed
 *                        it is left untouched with status refused
 * @param map_results     candidates x 4 u32 (overwritten for solved
 *                        components with joint-exact values)
 * @param solve_status    candidates u32: 2 = component-exact,
 *                        6 = escalate to another exact strategy,
 *                        0xFFFFFFFF = poisoned
 * @param fuel_spent      global device counter of ACTUAL node
 *                        expansions (enumerated combinations); each
 *                        solved component adds its exact count
 */
extern "C" __global__ void joint_component_enumerate(
    const float* __restrict__ scores,
    const uint64_t* __restrict__ feasible_sets,
    const uint32_t* __restrict__ arguments,
    const uint32_t* __restrict__ argument_arities,
    const uint64_t* __restrict__ domains,
    const uint64_t* __restrict__ role_masks,
    const uint32_t* __restrict__ component_parents,
    const uint32_t* __restrict__ component_count,
    uint32_t num_candidates,
    uint32_t num_labels,
    uint32_t lanes,
    uint32_t max_arity,
    uint64_t fuel_authorized,
    uint32_t* __restrict__ map_results,
    uint32_t* __restrict__ solve_status,
    unsigned long long* __restrict__ fuel_spent
) {
    uint32_t root = blockIdx.x;
    if (root >= num_candidates || threadIdx.x != 0) return;
    if (component_parents[root] != root) return;

    const uint32_t MAX_COMP_CANDS = 8;
    const uint32_t MAX_LANES = 8;
    uint32_t label_words = (num_labels + 63u) / 64u;
    uint32_t cand_ids[MAX_COMP_CANDS];
    uint32_t n = 0u;
    for (uint32_t candidate = 0; candidate < num_candidates; ++candidate) {
        if (component_parents[candidate] != root) continue;
        if (n < MAX_COMP_CANDS) cand_ids[n] = candidate;
        ++n;
    }
    if (n == 0u) return;

    // Stage-eligibility guards for this specialization. Components
    // outside its local arrays or enumeration budget escalate to the
    // next exact strategy; they are never approximated.
    bool refuse = (n > MAX_COMP_CANDS) || (lanes > MAX_LANES);
    if (n > MAX_COMP_CANDS) {
        for (uint32_t candidate = 0; candidate < num_candidates; ++candidate) {
                if (component_parents[candidate] == root) solve_status[candidate] = 6u;
        }
        return;
    }
    uint32_t num_components = component_count[0];
    uint64_t fuel_per_component = num_components == 0u
        ? 0ull
        : fuel_authorized / (uint64_t)num_components;
    uint32_t feas_labels[MAX_COMP_CANDS];
    uint64_t combos = 1;
    for (uint32_t i = 0; i < n && !refuse; ++i) {
        uint32_t cand = cand_ids[i];
        uint32_t fcount = 0;
        for (uint32_t w = 0; w < label_words; ++w) {
            uint64_t bits = feasible_sets[(uint64_t)cand * label_words + w];
            fcount += (uint32_t)__popcll(bits);
        }
        // A poisoned feasibility row arrives here as a ZEROED set
        // (its 0xFFFFFFFF marker lives in the counts buffer, which
        // this stage does not read), so an empty popcount covers
        // both poison and genuine infeasibility.
        if (fcount == 0) {
            // Poisoned or empty row poisons the whole component:
            // no consistent joint assignment can exist through it.
            for (uint32_t j = 0; j < n; ++j) {
                uint32_t cj = cand_ids[j];
                uint32_t* out = map_results + (uint64_t)cj * 4u;
                out[0] = 0xFFFFFFFFu; out[1] = 1u; out[2] = 0u; out[3] = 0u;
                solve_status[cj] = 0xFFFFFFFFu;
            }
            return;
        }
        feas_labels[i] = fcount;
        if (combos > fuel_per_component / (uint64_t)fcount) {
            refuse = true;
        } else {
            combos *= fcount;
        }
    }
    if (refuse) {
        for (uint32_t j = 0; j < n; ++j) solve_status[cand_ids[j]] = 6u;
        return;
    }

    // Per-candidate bests: overall best consistent total, runner-up
    // total over ALTERNATIVE labels, and the label realizing the best.
    float best_total[MAX_COMP_CANDS];
    float alt_total[MAX_COMP_CANDS];
    uint32_t best_lab[MAX_COMP_CANDS];
    uint32_t best_tied[MAX_COMP_CANDS];
    const float NEG_INF = -__int_as_float(0x7f800000);
    for (uint32_t i = 0; i < n; ++i) {
        best_total[i] = NEG_INF;
        alt_total[i] = NEG_INF;
        best_lab[i] = 0xFFFFFFFFu;
        best_tied[i] = 0u;
    }

    // Enumerate label combinations via per-candidate feasible-label
    // ordinals (odometer over feasible sets only).
    uint32_t ordinal[MAX_COMP_CANDS];
    uint32_t label_of[MAX_COMP_CANDS];
    for (uint32_t i = 0; i < n; ++i) ordinal[i] = 0;
    bool done = false;
    while (!done) {
        // Resolve ordinals to labels and score the combination.
        float total = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t cand = cand_ids[i];
            uint32_t seen = 0;
            uint32_t lab = 0xFFFFFFFFu;
            for (uint32_t l = 0; l < num_labels; ++l) {
                if ((feasible_sets[(uint64_t)cand * label_words + (l >> 6)]
                     >> (l & 63u)) & 1ull) {
                    if (seen == ordinal[i]) { lab = l; break; }
                    ++seen;
                }
            }
            label_of[i] = lab;
            total += scores[(uint64_t)cand * num_labels + lab];
        }

        // Consistency: every entity of the component keeps a nonempty
        // domain under the intersection of its role masks.
        bool consistent = true;
        for (uint32_t i = 0; i < n && consistent; ++i) {
            uint32_t arity = argument_arities[cand_ids[i]];
            for (uint32_t role = 0; role < arity && consistent; ++role) {
                uint32_t entity = arguments[(uint64_t)cand_ids[i] * max_arity + role];
                bool nonempty = false;
                for (uint32_t lane = 0; lane < lanes && !nonempty; ++lane) {
                    uint64_t acc = domains[(uint64_t)entity * lanes + lane];
                    for (uint32_t j = 0; j < n; ++j) {
                        uint32_t other_arity = argument_arities[cand_ids[j]];
                        for (uint32_t other_role = 0; other_role < other_arity; ++other_role) {
                            if (arguments[(uint64_t)cand_ids[j] * max_arity + other_role]
                                != entity) continue;
                            const uint64_t* mask = role_masks
                                + ((uint64_t)other_role * num_labels + label_of[j]) * lanes;
                            // An all-zero mask row asserts nothing
                            // (abstention): it imposes no constraint.
                            // A real label with a zero mask can never
                            // be chosen — feasibility already
                            // excluded it — so this convention is
                            // reachable only by abstention.
                            bool unconstrained = true;
                            for (uint32_t ml = 0; ml < lanes && unconstrained;
                                 ++ml) {
                                unconstrained = mask[ml] == 0ull;
                            }
                            if (unconstrained) continue;
                            acc &= mask[lane];
                        }
                    }
                    nonempty = acc != 0ull;
                }
                consistent = nonempty;
            }
        }

        if (consistent) {
            for (uint32_t i = 0; i < n; ++i) {
                if (total > best_total[i]) {
                    if (label_of[i] != best_lab[i]) {
                        alt_total[i] = best_total[i] > alt_total[i]
                            ? best_total[i] : alt_total[i];
                    }
                    best_total[i] = total;
                    best_lab[i] = label_of[i];
                    best_tied[i] = 0u;
                } else if (total == best_total[i]) {
                    if (label_of[i] != best_lab[i]) {
                        best_tied[i] = 1u;
                        alt_total[i] = total;
                    }
                } else if (label_of[i] != best_lab[i] && total > alt_total[i]) {
                    alt_total[i] = total;
                }
            }
        }

        // Odometer step.
        done = true;
        for (uint32_t i = 0; i < n; ++i) {
            if (++ordinal[i] < feas_labels[i]) { done = false; break; }
            ordinal[i] = 0;
        }
    }

    // Exact accounting: this component enumerated exactly `combos`
    // node expansions.
    atomicAdd(fuel_spent, (unsigned long long)combos);

    // Emit joint-exact rows. No consistent combination at all poisons
    // the component (existence failure is typed, never silent).
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t cand = cand_ids[i];
        uint32_t* out = map_results + (uint64_t)cand * 4u;
        if (best_lab[i] == 0xFFFFFFFFu) {
            out[0] = 0xFFFFFFFFu; out[1] = 1u; out[2] = 0u; out[3] = 0u;
            solve_status[cand] = 0xFFFFFFFFu;
            continue;
        }
        float margin = alt_total[i] == NEG_INF
            ? __int_as_float(0x7f800000)
            : best_total[i] - alt_total[i];
        uint32_t ambiguous = (best_tied[i] != 0u || margin == 0.0f) ? 1u : 0u;
        out[0] = best_lab[i];
        out[1] = ambiguous;
        out[2] = __float_as_uint(best_total[i]);
        out[3] = __float_as_uint(margin);
        solve_status[cand] = 2u;
    }
}

// Device-discovered memoized chain DP: exact joint MAP + exact
// per-candidate GLOBAL max-marginals for PATH components whose
// candidates arrive in chain order (tail of i == head of i+1). It
// consumes only roots produced by the carrier-owned component planner
// and only rows refused by complete enumeration. Wider frontiers and
// multi-lane domains REFUSE typed (status 3), never approximated.
extern "C" __global__ void joint_component_chain_dp(
    const float* __restrict__ scores,
    const uint64_t* __restrict__ feasible_sets,
    const uint32_t* __restrict__ arguments,
    const uint32_t* __restrict__ argument_arities,
    const uint64_t* __restrict__ domains,
    const uint64_t* __restrict__ role_masks,
    const uint32_t* __restrict__ component_parents,
    const uint32_t* __restrict__ component_count,
    uint32_t num_candidates,
    uint32_t num_labels,
    uint32_t lanes,
    uint32_t max_arity,
    uint64_t fuel_authorized,
    uint32_t* __restrict__ map_results,
    uint32_t* __restrict__ solve_status,
    unsigned long long* __restrict__ fuel_spent
) {
    uint32_t root = blockIdx.x;
    if (root >= num_candidates || threadIdx.x != 0) return;
    if (component_parents[root] != root || solve_status[root] != 6u) return;

    const uint32_t MAX_CHAIN = 32;
    const uint32_t MAX_LABELS = 32;
    const uint32_t MAX_STATES = 33;
    uint32_t cand_ids[MAX_CHAIN];
    uint32_t n = 0u;
    for (uint32_t candidate = 0; candidate < num_candidates; ++candidate) {
        if (component_parents[candidate] != root) continue;
        if (n < MAX_CHAIN) cand_ids[n] = candidate;
        ++n;
    }
    if (n == 0u) return;

    uint32_t num_components = component_count[0];
    uint64_t fuel_per_component = num_components == 0u
        ? 0ull
        : fuel_authorized / (uint64_t)num_components;
    uint32_t label_words = (num_labels + 63u) / 64u;
    const float NEG_INF = -__int_as_float(0x7f800000);

    // Eligibility is proven from the device-resident component itself:
    // single-lane domains, bounded binary chain, and chain order.
    bool refuse = (n > MAX_CHAIN) || (lanes != 1u)
        || (num_labels > MAX_LABELS);
    for (uint32_t i = 0; i < n && !refuse; ++i) {
        refuse = argument_arities[cand_ids[i]] != 2u;
    }
    for (uint32_t i = 0; i + 1 < n && !refuse; ++i) {
        uint32_t t = arguments[(uint64_t)cand_ids[i] * max_arity + 1u];
        uint32_t h = arguments[(uint64_t)cand_ids[i + 1u] * max_arity];
        refuse = t != h;
    }
    if (refuse) {
        for (uint32_t candidate = 0; candidate < num_candidates; ++candidate) {
            if (component_parents[candidate] == root) solve_status[candidate] = 6u;
        }
        return;
    }

    // Effective per-label masks: an all-zero row asserts nothing
    // (abstention convention shared with the enumeration stage).
    uint64_t full = ~0ull;
    unsigned long long transitions = 0;

    // One restricted forward pass: candidate `pin` forced to label
    // `pin_label` (pin == 0xFFFFFFFF: unrestricted). Returns the best
    // full-assignment total, accumulated left-to-right.
    // States: reachable bitsets of the link entity after each step.
    bool fuel_exhausted = false;
    auto head_eff = [&](uint32_t lab) {
        uint64_t m = role_masks[lab];
        return m == 0ull ? full : m;
    };
    auto tail_eff = [&](uint32_t lab) {
        uint64_t m = role_masks[num_labels + lab];
        return m == 0ull ? full : m;
    };
    auto feasible = [&](uint32_t cand, uint32_t lab) {
        return ((feasible_sets[(uint64_t)cand * label_words + (lab >> 6)]
                 >> (lab & 63u)) & 1ull) != 0ull;
    };
    auto run_pass = [&](uint32_t pin, uint32_t pin_label) -> float {
        uint64_t state_bits[MAX_STATES];
        float state_best[MAX_STATES];
        uint32_t states = 1;
        state_bits[0] = domains[arguments[(uint64_t)cand_ids[0] * max_arity]];
        state_best[0] = 0.0f;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t cand = cand_ids[i];
            uint64_t tail_dom = domains[arguments[(uint64_t)cand * max_arity + 1u]];
            uint64_t next_bits[MAX_STATES];
            float next_best[MAX_STATES];
            uint32_t next_states = 0;
            for (uint32_t s = 0; s < states; ++s) {
                if (state_best[s] == NEG_INF) continue;
                for (uint32_t l = 0; l < num_labels; ++l) {
                    if (i == pin && l != pin_label) continue;
                    if (!feasible(cand, l)) continue;
                    if ((state_bits[s] & head_eff(l)) == 0ull) continue;
                    uint64_t out = tail_dom & tail_eff(l);
                    if (out == 0ull) continue;
                    if (transitions == fuel_per_component) {
                        fuel_exhausted = true;
                        return NEG_INF;
                    }
                    ++transitions;
                    float total = state_best[s]
                        + scores[(uint64_t)cand * num_labels + l];
                    uint32_t k = 0;
                    for (; k < next_states; ++k) {
                        if (next_bits[k] == out) break;
                    }
                    if (k == next_states) {
                        if (next_states == MAX_STATES) return NEG_INF;
                        next_bits[next_states] = out;
                        next_best[next_states] = total;
                        ++next_states;
                    } else if (total > next_best[k]) {
                        next_best[k] = total;
                    }
                }
            }
            states = next_states;
            for (uint32_t s = 0; s < states; ++s) {
                state_bits[s] = next_bits[s];
                state_best[s] = next_best[s];
            }
            if (states == 0) return NEG_INF;
        }
        float best = NEG_INF;
        for (uint32_t s = 0; s < states; ++s) {
            if (state_best[s] > best) best = state_best[s];
        }
        return best;
    };

    float joint_best = run_pass(0xFFFFFFFFu, 0u);
    if (fuel_exhausted) {
        for (uint32_t i = 0; i < n; ++i) solve_status[cand_ids[i]] = 3u;
        atomicAdd(fuel_spent, transitions);
        return;
    }
    if (joint_best == NEG_INF) {
        // No consistent assignment exists: typed existence failure.
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t cand = cand_ids[i];
            uint32_t* out = map_results + (uint64_t)cand * 4u;
            out[0] = 0xFFFFFFFFu; out[1] = 1u; out[2] = 0u; out[3] = 0u;
            solve_status[cand] = 0xFFFFFFFFu;
        }
        atomicAdd(fuel_spent, transitions);
        return;
    }

    // Compute every row into locals FIRST: a fuel refusal must not
    // leave partially emitted rows behind (per-candidate execution
    // witness law — refusal emits status only, like the enumeration
    // stage).
    uint32_t row_lab[MAX_CHAIN];
    uint32_t row_amb[MAX_CHAIN];
    float row_total[MAX_CHAIN];
    float row_margin[MAX_CHAIN];
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t cand = cand_ids[i];
        uint32_t best_lab = 0xFFFFFFFFu;
        float best_total = NEG_INF;
        float alt_total = NEG_INF;
        uint32_t tied = 0;
        for (uint32_t l = 0; l < num_labels; ++l) {
            if (!feasible(cand, l)) continue;
            float t = run_pass(i, l);
            if (fuel_exhausted) break;
            if (t == NEG_INF) continue;
            if (t > best_total) {
                if (best_lab != 0xFFFFFFFFu) {
                    alt_total = best_total > alt_total ? best_total : alt_total;
                }
                best_total = t;
                best_lab = l;
                tied = 0;
            } else if (t == best_total && l != best_lab) {
                tied = 1u;
                alt_total = t;
            } else if (t > alt_total) {
                alt_total = t;
            }
        }
        row_lab[i] = best_lab;
        row_total[i] = best_total;
        row_margin[i] = alt_total == NEG_INF
            ? __int_as_float(0x7f800000)
            : best_total - alt_total;
        row_amb[i] = (tied != 0u || row_margin[i] == 0.0f) ? 1u : 0u;
        if (fuel_exhausted) break;
    }
    if (fuel_exhausted) {
        for (uint32_t i = 0; i < n; ++i) solve_status[cand_ids[i]] = 3u;
        atomicAdd(fuel_spent, transitions);
        return;
    }
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t cand = cand_ids[i];
        uint32_t* out = map_results + (uint64_t)cand * 4u;
        out[0] = row_lab[i];
        out[1] = row_amb[i];
        out[2] = __float_as_uint(row_total[i]);
        out[3] = __float_as_uint(row_margin[i]);
        solve_status[cand] = 4u;
    }
    atomicAdd(fuel_spent, transitions);
}

// General exact fallback for every component topology and candidate
// arity. One device thread owns one component and performs a bounded
// depth-first branch-and-bound over feasible labels. Search state lives
// in carrier-owned global scratch, so component size is not limited by
// fixed local arrays and no host-side component plan is required.
extern "C" __global__ void joint_component_branch_and_bound(
    const float* __restrict__ scores,
    const uint64_t* __restrict__ feasible_sets,
    const uint32_t* __restrict__ arguments,
    const uint32_t* __restrict__ argument_arities,
    const uint64_t* __restrict__ domains,
    const uint64_t* __restrict__ role_masks,
    const uint32_t* __restrict__ component_parents,
    const uint32_t* __restrict__ component_count,
    uint32_t num_candidates,
    uint32_t num_labels,
    uint32_t lanes,
    uint32_t max_arity,
    uint64_t fuel_authorized,
    uint32_t* __restrict__ map_results,
    uint32_t* __restrict__ solve_status,
    unsigned long long* __restrict__ fuel_spent,
    uint32_t* __restrict__ assignment,
    uint32_t* __restrict__ best_label,
    float* __restrict__ best_total,
    float* __restrict__ alt_total
) {
    uint32_t root = blockIdx.x;
    if (root >= num_candidates || threadIdx.x != 0) return;
    if (component_parents[root] != root || solve_status[root] != 6u) return;

    const uint32_t UNASSIGNED = 0xFFFFFFFFu;
    const float NEG_INF = -__int_as_float(0x7f800000);
    uint32_t label_words = (num_labels + 63u) / 64u;
    uint32_t num_components = component_count[0];
    uint64_t fuel_per_component = num_components == 0u
        ? 0ull
        : fuel_authorized / (uint64_t)num_components;

    uint32_t first = UNASSIGNED;
    for (uint32_t candidate = 0; candidate < num_candidates; ++candidate) {
        if (component_parents[candidate] != root) continue;
        if (first == UNASSIGNED) first = candidate;
        assignment[candidate] = UNASSIGNED;
        best_label[candidate] = UNASSIGNED;
        best_total[candidate] = NEG_INF;
        alt_total[candidate] = NEG_INF;
    }
    if (first == UNASSIGNED) return;

    auto next_candidate = [&](uint32_t current) {
        for (uint32_t candidate = current + 1u; candidate < num_candidates; ++candidate) {
            if (component_parents[candidate] == root) return candidate;
        }
        return UNASSIGNED;
    };
    auto previous_candidate = [&](uint32_t current) {
        for (uint32_t candidate = current; candidate-- > 0u;) {
            if (component_parents[candidate] == root) return candidate;
        }
        return UNASSIGNED;
    };
    auto feasible = [&](uint32_t candidate, uint32_t label) {
        return ((feasible_sets[(uint64_t)candidate * label_words + (label >> 6)]
                 >> (label & 63u)) & 1ull) != 0ull;
    };
    auto mask_is_empty = [&](const uint64_t* mask) {
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            if (mask[lane] != 0ull) return false;
        }
        return true;
    };
    auto partial_consistent = [&](uint32_t changed_candidate) {
        uint32_t changed_arity = argument_arities[changed_candidate];
        for (uint32_t changed_role = 0; changed_role < changed_arity; ++changed_role) {
            uint32_t entity = arguments[
                (uint64_t)changed_candidate * max_arity + changed_role
            ];
            bool any_lane = false;
            for (uint32_t lane = 0; lane < lanes && !any_lane; ++lane) {
                uint64_t intersection = domains[(uint64_t)entity * lanes + lane];
                for (uint32_t candidate = 0;
                     candidate < num_candidates && intersection != 0ull;
                     ++candidate) {
                    if (component_parents[candidate] != root
                        || assignment[candidate] == UNASSIGNED) continue;
                    uint32_t arity = argument_arities[candidate];
                    for (uint32_t role = 0; role < arity; ++role) {
                        if (arguments[(uint64_t)candidate * max_arity + role] != entity) {
                            continue;
                        }
                        const uint64_t* mask = role_masks
                            + ((uint64_t)role * num_labels + assignment[candidate]) * lanes;
                        if (!mask_is_empty(mask)) intersection &= mask[lane];
                    }
                }
                any_lane = intersection != 0ull;
            }
            if (!any_lane) return false;
        }
        return true;
    };
    auto branch_cannot_improve = [&]() {
        float upper = 0.0f;
        for (uint32_t candidate = 0; candidate < num_candidates; ++candidate) {
            if (component_parents[candidate] != root) continue;
            if (assignment[candidate] != UNASSIGNED) {
                upper += scores[(uint64_t)candidate * num_labels + assignment[candidate]];
                continue;
            }
            float row_max = NEG_INF;
            for (uint32_t label = 0; label < num_labels; ++label) {
                if (feasible(candidate, label)) {
                    float score = scores[(uint64_t)candidate * num_labels + label];
                    if (score > row_max) row_max = score;
                }
            }
            upper += row_max;
        }
        for (uint32_t candidate = 0; candidate < num_candidates; ++candidate) {
            if (component_parents[candidate] != root || best_label[candidate] == UNASSIGNED) {
                return false;
            }
            if (assignment[candidate] == UNASSIGNED) {
                if (!(upper < best_total[candidate] && upper < alt_total[candidate])) {
                    return false;
                }
            } else if (assignment[candidate] == best_label[candidate]) {
                if (!(upper < best_total[candidate])) return false;
            } else if (!(upper < alt_total[candidate])) {
                return false;
            }
        }
        return true;
    };

    unsigned long long expansions = 0ull;
    bool fuel_exhausted = false;
    bool found = false;
    uint32_t current = first;
    while (true) {
        bool descended = false;
        uint32_t first_label = assignment[current] == UNASSIGNED
            ? 0u : assignment[current] + 1u;
        assignment[current] = UNASSIGNED;
        for (uint32_t label = first_label; label < num_labels; ++label) {
            if (!feasible(current, label)) continue;
            if (expansions >= fuel_per_component) {
                fuel_exhausted = true;
                break;
            }
            ++expansions;
            assignment[current] = label;
            if (!partial_consistent(current) || branch_cannot_improve()) {
                assignment[current] = UNASSIGNED;
                continue;
            }

            uint32_t next = next_candidate(current);
            if (next != UNASSIGNED) {
                assignment[next] = UNASSIGNED;
                current = next;
                descended = true;
                break;
            }

            found = true;
            float total = 0.0f;
            for (uint32_t candidate = 0; candidate < num_candidates; ++candidate) {
                if (component_parents[candidate] == root) {
                    total += scores[
                        (uint64_t)candidate * num_labels + assignment[candidate]
                    ];
                }
            }
            for (uint32_t candidate = 0; candidate < num_candidates; ++candidate) {
                if (component_parents[candidate] != root) continue;
                uint32_t label_here = assignment[candidate];
                if (total > best_total[candidate]) {
                    if (label_here != best_label[candidate]) {
                        alt_total[candidate] = best_total[candidate] > alt_total[candidate]
                            ? best_total[candidate] : alt_total[candidate];
                    }
                    best_total[candidate] = total;
                    best_label[candidate] = label_here;
                } else if (total == best_total[candidate]
                           && label_here != best_label[candidate]) {
                    alt_total[candidate] = total;
                } else if (label_here != best_label[candidate]
                           && total > alt_total[candidate]) {
                    alt_total[candidate] = total;
                }
            }
            assignment[current] = UNASSIGNED;
        }
        if (fuel_exhausted) break;
        if (descended) continue;
        assignment[current] = UNASSIGNED;
        if (current == first) break;
        current = previous_candidate(current);
    }

    atomicAdd(fuel_spent, expansions);
    if (fuel_exhausted) {
        for (uint32_t candidate = 0; candidate < num_candidates; ++candidate) {
            if (component_parents[candidate] == root) solve_status[candidate] = 3u;
        }
        return;
    }
    for (uint32_t candidate = 0; candidate < num_candidates; ++candidate) {
        if (component_parents[candidate] != root) continue;
        uint32_t* out = map_results + (uint64_t)candidate * 4u;
        if (!found || best_label[candidate] == UNASSIGNED) {
            out[0] = UNASSIGNED; out[1] = 1u; out[2] = 0u; out[3] = 0u;
            solve_status[candidate] = UNASSIGNED;
            continue;
        }
        float margin = alt_total[candidate] == NEG_INF
            ? __int_as_float(0x7f800000)
            : best_total[candidate] - alt_total[candidate];
        out[0] = best_label[candidate];
        out[1] = margin == 0.0f ? 1u : 0u;
        out[2] = __float_as_uint(best_total[candidate]);
        out[3] = __float_as_uint(margin);
        solve_status[candidate] = 5u;
    }
}
