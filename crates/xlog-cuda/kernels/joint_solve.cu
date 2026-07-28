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
 * @param pairs           candidate entity pairs, candidates x 2 u32
 *                        (head entity index, tail entity index)
 * @param head_masks      label head signatures, labels x lanes u64
 * @param tail_masks      label tail signatures, labels x lanes u64
 * @param num_entities    entity capacity; pair indices must be below
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
    const uint32_t* __restrict__ pairs,
    const uint64_t* __restrict__ head_masks,
    const uint64_t* __restrict__ tail_masks,
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
    uint32_t head_entity = pairs[cand * 2u];
    uint32_t tail_entity = pairs[cand * 2u + 1u];
    if (head_entity >= num_entities || tail_entity >= num_entities) {
        // Corrupt producer record: poison, never read out of bounds.
        for (uint32_t word = 0; word < label_words; ++word) {
            feasible_sets[(uint64_t)cand * label_words + word] = 0ull;
        }
        feasible_counts[cand] = 0xFFFFFFFFu;
        return;
    }
    const uint64_t* head_domain = domains + (uint64_t)head_entity * lanes;
    const uint64_t* tail_domain = domains + (uint64_t)tail_entity * lanes;

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
                const uint64_t* head_mask = head_masks + (uint64_t)label * lanes;
                const uint64_t* tail_mask = tail_masks + (uint64_t)label * lanes;
                bool head_hit = false;
                for (uint32_t lane = 0; lane < lanes && !head_hit; ++lane) {
                    head_hit = (head_domain[lane] & head_mask[lane]) != 0ull;
                }
                bool tail_hit = false;
                for (uint32_t lane = 0; lane < lanes && head_hit && !tail_hit;
                     ++lane) {
                    tail_hit = (tail_domain[lane] & tail_mask[lane]) != 0ull;
                }
                feasible = head_hit && tail_hit;
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
 * @param pairs           candidate entity pairs, candidates x 2 u32
 * @param domains         entity sort domains, entities x lanes u64
 * @param head_masks      label head signatures, labels x lanes u64
 * @param tail_masks      label tail signatures, labels x lanes u64
 * @param comp_cand_offsets per-component start offset into
 *                        comp_cand_indices, components+1 u32
 * @param comp_cand_indices candidate indices grouped by component
 * @param num_components  component count in this launch
 * @param num_labels      label universe width
 * @param lanes           u64 lanes per domain / signature row
 * @param fuel_per_component node-expansion budget per component;
 *                        a component whose enumeration would exceed
 *                        it is left untouched with status refused
 * @param map_results     candidates x 4 u32 (overwritten for solved
 *                        components with joint-exact values)
 * @param solve_status    candidates u32: 2 = component-exact,
 *                        3 = refused (fuel), 0xFFFFFFFF = poisoned
 * @param fuel_spent      global device counter of ACTUAL node
 *                        expansions (enumerated combinations); each
 *                        solved component adds its exact count
 */
extern "C" __global__ void joint_component_enumerate(
    const float* __restrict__ scores,
    const uint64_t* __restrict__ feasible_sets,
    const uint32_t* __restrict__ pairs,
    const uint64_t* __restrict__ domains,
    const uint64_t* __restrict__ head_masks,
    const uint64_t* __restrict__ tail_masks,
    const uint32_t* __restrict__ comp_cand_offsets,
    const uint32_t* __restrict__ comp_cand_indices,
    uint32_t num_components,
    uint32_t num_labels,
    uint32_t lanes,
    uint64_t fuel_per_component,
    uint32_t* __restrict__ map_results,
    uint32_t* __restrict__ solve_status,
    unsigned long long* __restrict__ fuel_spent
) {
    uint32_t comp = blockIdx.x;
    if (comp >= num_components || threadIdx.x != 0) return;

    const uint32_t MAX_COMP_CANDS = 8;
    const uint32_t MAX_LANES = 8;
    uint32_t begin = comp_cand_offsets[comp];
    uint32_t end = comp_cand_offsets[comp + 1];
    uint32_t n = end - begin;
    uint32_t label_words = (num_labels + 63u) / 64u;

    // Stage-eligibility guards: capacity of the local arrays and the
    // component fuel budget. Oversized components are REFUSED, never
    // approximated.
    bool refuse = (n > MAX_COMP_CANDS) || (lanes > MAX_LANES);
    uint32_t cand_ids[MAX_COMP_CANDS];
    uint32_t feas_labels[MAX_COMP_CANDS];
    uint64_t combos = 1;
    for (uint32_t i = 0; i < n && !refuse; ++i) {
        uint32_t cand = comp_cand_indices[begin + i];
        cand_ids[i] = cand;
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
                uint32_t cj = comp_cand_indices[begin + j];
                uint32_t* out = map_results + (uint64_t)cj * 4u;
                out[0] = 0xFFFFFFFFu; out[1] = 1u; out[2] = 0u; out[3] = 0u;
                solve_status[cj] = 0xFFFFFFFFu;
            }
            return;
        }
        feas_labels[i] = fcount;
        combos *= fcount;
        if (combos > fuel_per_component) refuse = true;
    }
    if (refuse) {
        for (uint32_t j = begin; j < end; ++j) {
            solve_status[comp_cand_indices[j]] = 3u;
        }
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
            for (uint32_t side = 0; side < 2 && consistent; ++side) {
                uint32_t entity = pairs[cand_ids[i] * 2u + side];
                bool nonempty = false;
                for (uint32_t lane = 0; lane < lanes && !nonempty; ++lane) {
                    uint64_t acc = domains[(uint64_t)entity * lanes + lane];
                    for (uint32_t j = 0; j < n; ++j) {
                        for (uint32_t sj = 0; sj < 2; ++sj) {
                            if (pairs[cand_ids[j] * 2u + sj] != entity) continue;
                            const uint64_t* mask =
                                (sj == 0 ? head_masks : tail_masks)
                                + (uint64_t)label_of[j] * lanes;
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
