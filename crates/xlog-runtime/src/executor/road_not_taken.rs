//! ST-TRC (A) — road-not-taken deliberation source: the DERIVED epistemic
//! `possible-not-known` atom set, computed from accepted world-view bitsets.
//!
//! # What this is
//!
//! For sound-(A), the road-not-taken premises are atoms the engine *considered*
//! (true in some world it explored) but did *not* commit (not true in every
//! world), and that are net-new versus the induction request's positives:
//!
//! ```text
//! road_not_taken = (⋃ accepted world-views) − (⋂ accepted world-views) − positives
//! ```
//!
//! `⋂(accepted)` = the atoms true in *every* accepted world-view = `Know` (∀);
//! `⋃(accepted)` = the atoms true in *some* accepted world-view = `Possible` (∃);
//! their difference is exactly `Possible ∧ ¬Know` = considered-but-not-committed.
//!
//! # Why the (ii)-circular exclusion is INTRINSIC (no explicit reject filter)
//!
//! A road-not-taken atom must be (i)-plausible (a live hypothesis), never
//! (ii)-circular (an atom the logic actively pruned — feeding it back would be
//! circular, worse than null). On the epistemic GPU/GPT path the *only* notion of
//! rejection is **world-view-level constraint violation**: the Generate-Propagate-
//! **Test** classifier accepts a world-view iff it is a consistent model. A
//! logic-pruned atom forces world-view inconsistency, so *every* world-view
//! containing it is rejected, and it therefore never appears in any **accepted**
//! world-view. Because this reduction unions over **accepted** world-views only,
//! a (ii)-pruned atom is excluded **by construction** — it is absent from every
//! input bitset. This is the (b) grounded-semantic invariant of (A) §1:
//!
//!   "accepted world-view ⇒ consistent model ⇒ contains no logic-pruned atom",
//!
//! grounded in the classifier definition (accepted ≡ constraint-satisfying) plus
//! the engine fact that GPU-path rejection is world-view-constraint-violation
//! (there is no separate atom-level rejected set on this path; the CPU
//! `EpistemicInterpretation.rejected` declared-config is a different
//! representation, absent from xlog-runtime). The (a) necessary half — that the
//! reduction unions accepted-only and so excludes rejected-only atoms — is the
//! HARD GREEN unit gate verified by the tests below.
//!
//! The literal-index → atom decode (build-precision #2) is the integration step
//! that maps each epistemic literal to its `{pred_id, arg0, arg1}` payload; this
//! module operates on a caller-supplied `literal_atoms` map so the reduction
//! logic is unit-testable in isolation.

// Pending integration: the next (A) step wires this reduction to the
// `DeviceSemanticSummary` accepted world_views + `accepted_candidate_indices`.
// The (a) §1 HARD-GREEN gate is the unit tests below (verified on this logic).
#![allow(dead_code)]

use std::collections::BTreeSet;

/// An atom key as the induction-premise consumer reads it: `(pred_id, arg0, arg1)`.
pub type AtomKey = (u32, u32, u32);

/// Whether epistemic literal `i` is set in a world-view bitset
/// (`(literal_count + 7) / 8` bytes, bit `i` = byte `i / 8`, mask `1 << (i % 8)`).
#[inline]
fn literal_set(bitset: &[u8], literal_index: usize) -> bool {
    let byte = literal_index / 8;
    let mask = 1u8 << (literal_index % 8);
    byte < bitset.len() && (bitset[byte] & mask) != 0
}

/// Compute the road-not-taken `possible-not-known` atom set
/// = `(⋃ accepted) − (⋂ accepted) − positives` over the accepted world-view
/// bitsets, decoding each surviving epistemic literal to its atom payload.
///
/// `accepted_world_views`: one bit-per-literal bitset per **accepted** world-view
/// (already selected via `accepted_candidate_indices` — rejected world-views are
/// NOT passed, which is what makes (ii)-exclusion intrinsic).
/// `literal_atoms[i]`: atom payload for epistemic literal `i` (len `literal_count`).
/// `positives`: the induction request's positive facts to subtract (net-additivity).
pub fn road_not_taken_possible_not_known(
    accepted_world_views: &[&[u8]],
    literal_count: usize,
    literal_atoms: &[AtomKey],
    positives: &BTreeSet<AtomKey>,
) -> Vec<AtomKey> {
    // With no accepted world-views there is no consistent model => empty source.
    if accepted_world_views.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 0..literal_count {
        let in_some = accepted_world_views.iter().any(|wv| literal_set(wv, i)); // Possible (∃)
        let in_all = accepted_world_views.iter().all(|wv| literal_set(wv, i)); // Know (∀)
        if in_some && !in_all {
            let atom = literal_atoms[i];
            if !positives.contains(&atom) {
                out.push(atom);
            }
        }
    }
    out
}

/// (2b) go/no-go DISCRIMINATOR gate — pairs with the reduction above.
///
/// A net-new road-not-taken result (`road_not_taken_possible_not_known(..) > 0`)
/// is GENUINE — not trivially-free — only if the **all-candidates-false**
/// world-view was REJECTED, i.e. a required goal (`:- not g`) forced a choice.
/// Returns true iff the empty (every-literal-false) world-view is NOT among the
/// ACCEPTED bitsets.
///
/// This gate is NECESSARY because net-new OUTPUT is identical for genuine vs
/// trivially-free contestation (both yield `⋃−⋂` over the candidates with
/// `⋂∩candidate = ∅`); the reduction alone cannot distinguish them. It
/// **supersedes** an `⋂∩candidate`-size detector (which is `∅` for genuine AND
/// trivial-free alike). Double role: (1) confirms the corpus is forced-target
/// (not redundant-free); (2) confirms the GPU `:- not g` constraint ACTUALLY
/// fired at runtime — a silent negation failure also leaves the all-false world
/// accepted. The (2b) go/no-go is therefore `net-new > 0 AND forced_target_confirmed`.
pub fn forced_target_confirmed(accepted_world_views: &[&[u8]], literal_count: usize) -> bool {
    if accepted_world_views.is_empty() {
        return false;
    }
    // The all-candidates-false world-view = every literal bit clear.
    !accepted_world_views
        .iter()
        .any(|wv| (0..literal_count).all(|i| !literal_set(wv, i)))
}

/// Grounded world-fraction marginal for epistemic literal `i`
/// = `|{accepted-wv : literal i set}| / |accepted-wv|` ∈ [0,1].
///
/// This is the sub-gap-1-SOUND uncertainty measure: the literal fraction of
/// accepted world-views in which the atom holds. It is NOT a Belnap→probability
/// heuristic — four-valued information-state (`none/false/true/both`) has no
/// canonical map to `[0,1]`, ruled principled-unsound — it is the genuine
/// fraction of the (consistent) models the engine explored.
#[inline]
fn world_fraction_marginal(accepted_world_views: &[&[u8]], literal_index: usize) -> f32 {
    if accepted_world_views.is_empty() {
        return 0.0;
    }
    let hits = accepted_world_views
        .iter()
        .filter(|wv| literal_set(wv, literal_index))
        .count();
    hits as f32 / accepted_world_views.len() as f32
}

/// TIER-2 grounded-uncertainty export over accepted world-views — BOTH coupling
/// surfaces from one inference: grounded marginals (conditions / Arm-B re-weight)
/// + the §1 road-not-taken set (proposes / producer). Atom-keyed for the locked
/// `(pred_id, arg0, arg1)` consume-surface.
#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicExportResult {
    /// Grounded world-fraction marginal per candidate literal, index-aligned to
    /// `literal_atoms` (== the binding's `candidate_index_to_atom`). ∈ [0,1].
    pub marginals: Vec<f32>,
    /// §1 road-not-taken `possible-not-known` set (`⋃−⋂−positives`), atom-keyed.
    pub possible_not_known: Vec<AtomKey>,
    /// LOUD no-substrate diagnostic (`None` = substrate present): set when there is
    /// no consistent model (no accepted world-views) or no contestation
    /// (`< 2` candidate literals — a single-world / no-alternative input). The
    /// whole-group prune counts are DTS-pre-invoke diagnostics assembled by the
    /// binding, not computed here (the export never prunes).
    pub no_substrate_reason: Option<String>,
}

/// Grounded-uncertainty export over the accepted world-view bitsets — the pure,
/// host-testable core of the `epistemic_export` binding. The pyxlog binding reads
/// `DeviceSemanticSummary.world_views` (selected via `accepted_candidate_indices`)
/// into these bitsets, builds `literal_atoms` from its `candidate_index_to_atom`
/// map, and calls this; the result is wrapped to Python (DLPack marginals + dict).
///
/// Returns BOTH coupling directions from ONE accepted-world-view set:
///  * `marginals[i]` = grounded world-fraction of candidate literal `i`
///    (the conditions / Arm-B re-weight surface — existing-uncertain ∈ (0,1));
///  * `possible_not_known` = grounded `⋃−⋂` over accepted worlds (the proposes /
///    producer surface).
///
/// Net-additivity is NOT applied here: the engine returns the grounded `⋃−⋂` and
/// DTS subtracts `request.positive_facts` consume-side (net-additivity is a
/// DTS-induction concept, not engine-modal-state — the agreed seam). The §1 helper
/// is therefore called with an empty positives set to get the raw `⋃−⋂`.
///
/// Soundness invariants (build-contract): marginals are the literal
/// fraction-of-accepted-worlds (grounded), never a qualitative→prob heuristic; the
/// `⋃−⋂` unions ACCEPTED world-views only, so (ii)-circular logic-pruned atoms are
/// excluded by construction.
pub fn export_from_accepted_world_views(
    accepted_world_views: &[&[u8]],
    literal_count: usize,
    literal_atoms: &[AtomKey],
) -> EpistemicExportResult {
    let no_substrate_reason = if accepted_world_views.is_empty() {
        Some("no accepted world-views (no consistent model)".to_string())
    } else if literal_count < 2 {
        Some(format!(
            "no contestation: {literal_count} candidate literal(s) (< 2 alternatives)"
        ))
    } else {
        None
    };

    let marginals = (0..literal_count)
        .map(|i| world_fraction_marginal(accepted_world_views, i))
        .collect();

    // Grounded `⋃−⋂` (no positives subtraction — DTS does net-additivity consume-side).
    let possible_not_known = road_not_taken_possible_not_known(
        accepted_world_views,
        literal_count,
        literal_atoms,
        &BTreeSet::new(),
    );

    EpistemicExportResult {
        marginals,
        possible_not_known,
        no_substrate_reason,
    }
}

/// Per-world-view bitset byte count for `literal_count` literals: `(literal_count + 7) / 8`
/// (bit `i` lives in byte `i / 8`). This is the literal-bitset width; the device
/// `world_views` row STRIDE may be larger (padded to `max_worlds`).
#[inline]
fn world_view_bitset_bytes(literal_count: usize) -> usize {
    (literal_count + 7) / 8
}

/// Decode the accepted world-view literal-bitsets out of the flat device
/// `world_views` buffer (already read back to host via untracked DtoH).
///
/// The buffer is `candidate_count × world_view_stride` bytes, where
/// `world_view_stride = max(max_worlds, bitset_bytes(literal_count))` — the per-row
/// stride is padded to `max_worlds` (allocation-conservative), so each candidate's
/// literal-bitset is the PREFIX (`bitset_bytes`) of its stride-row, NOT a contiguous
/// `idx * bitset_bytes` slice. Returns one bitset slice per accepted candidate index,
/// ready for `export_from_accepted_world_views`. Out-of-range indices are skipped
/// (defensive against a malformed buffer) rather than panicking.
pub fn decode_accepted_bitsets<'a>(
    world_views: &'a [u8],
    accepted_candidate_indices: &[usize],
    literal_count: usize,
    max_worlds: usize,
) -> Vec<&'a [u8]> {
    let bitset_bytes = world_view_bitset_bytes(literal_count);
    let stride = max_worlds.max(bitset_bytes);
    let mut out = Vec::with_capacity(accepted_candidate_indices.len());
    for &idx in accepted_candidate_indices {
        let start = idx * stride;
        let end = start + bitset_bytes;
        if end <= world_views.len() {
            out.push(&world_views[start..end]);
        }
    }
    out
}

/// Host-side `epistemic_export`: the pure pipeline the pyxlog `evaluate_epistemic`
/// binding calls AFTER reading the device `world_views` buffer to host via the
/// provider's untracked metadata DtoH (`dtoh_small_metadata_untracked`). The binding
/// owns the device readback (it holds the provider/workspace context); this fn is
/// the pure, host-TDD-able core — stride-aware decode → grounded export — so the
/// only GPU-coupled step is the bounded metadata buffer copy the binding performs.
///
/// `literal_atoms[i]` is the binding's `candidate_index → (pred_id, arg0, arg1)` map
/// (input-order); the result is keyed on those tuples for the DTS consume-surface.
pub fn epistemic_export_from_host_buffer(
    world_views_host: &[u8],
    accepted_candidate_indices: &[usize],
    literal_count: usize,
    max_worlds: usize,
    literal_atoms: &[AtomKey],
) -> EpistemicExportResult {
    let bitsets = decode_accepted_bitsets(
        world_views_host,
        accepted_candidate_indices,
        literal_count,
        max_worlds,
    );
    export_from_accepted_world_views(&bitsets, literal_count, literal_atoms)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Literal layout: 0 = X, 1 = Y, 2 = Z.
    const X: AtomKey = (100, 1, 2);
    const Y: AtomKey = (101, 3, 4);
    const Z: AtomKey = (102, 5, 6);

    fn lits() -> [AtomKey; 3] {
        [X, Y, Z]
    }

    /// (A) §1 (a) — the HARD GREEN gate: the reduction unions ACCEPTED
    /// world-views only, so a rejected-only atom (`Z`, present in no accepted
    /// world-view) is excluded BY CONSTRUCTION; `Know` (∀) atoms are excluded;
    /// `Possible ∧ ¬Know` atoms are the road-not-taken source.
    #[test]
    fn excludes_rejected_only_and_known_keeps_possible_not_known() {
        // Accepted world-views (bit per literal):
        //   wv_a = {X, Y} = 0b011
        //   wv_b = {X}    = 0b001
        // Z (literal 2) is in NEITHER accepted world-view: it survives only in a
        // (rejected) world-view, i.e. it is the (ii)-circular / logic-pruned case.
        let wv_a = [0b011u8];
        let wv_b = [0b001u8];
        let accepted: Vec<&[u8]> = vec![&wv_a, &wv_b];
        let positives = BTreeSet::new();

        let result = road_not_taken_possible_not_known(&accepted, 3, &lits(), &positives);

        assert!(
            result.contains(&Y),
            "Y is true in some-but-not-all accepted world-views (Possible ∧ ¬Know) \
             => road-not-taken, must be included; got {result:?}"
        );
        assert!(
            !result.contains(&X),
            "X is true in EVERY accepted world-view (Know/∀) => committed, must be excluded"
        );
        assert!(
            !result.contains(&Z),
            "Z appears in NO accepted world-view (rejected-only / (ii)-circular) => \
             excluded by construction: the reduction unions accepted world-views only"
        );
    }

    /// (A) §1 net-additivity: an atom already in the request positives is NOT
    /// re-emitted as road-not-taken (duplicate-saturation guard, the 2b lesson).
    #[test]
    fn subtracts_request_positives_for_net_additivity() {
        let wv_a = [0b011u8]; // {X, Y}
        let wv_b = [0b001u8]; // {X}
        let accepted: Vec<&[u8]> = vec![&wv_a, &wv_b];
        let mut positives = BTreeSet::new();
        positives.insert(Y); // Y is already an induction-input positive.

        let result = road_not_taken_possible_not_known(&accepted, 3, &lits(), &positives);

        assert!(
            !result.contains(&Y),
            "Y ∈ request.positive_facts => not net-new => excluded; got {result:?}"
        );
        assert!(
            result.is_empty(),
            "only Y was possible-not-known; got {result:?}"
        );
    }

    /// No accepted world-views (no consistent model) => empty road-not-taken set.
    #[test]
    fn empty_accepted_yields_empty_source() {
        let accepted: Vec<&[u8]> = Vec::new();
        let positives = BTreeSet::new();
        let result = road_not_taken_possible_not_known(&accepted, 3, &lits(), &positives);
        assert!(result.is_empty());
    }

    /// (2b) GENUINE contestation — the required-goal trigger. A forced target `g`
    /// (`:- not g`) derivable by TWO competing rule-supported paths (`g :- c1.` /
    /// `g :- c2.`) makes the all-candidates-false world INCONSISTENT (g would be
    /// false) => that world is REJECTED, so the accepted world-views are exactly
    /// the ones choosing >=1 path. c1/c2 are then NECESSARY alternatives (each
    /// possible, neither forced), so the runner-up c2 (top-1 c1 committed) is
    /// genuine road-not-taken. This is the (2b) contestation-structure the corpus
    /// must engineer (criterion 4 = forced/required target).
    #[test]
    fn required_goal_two_paths_yields_genuine_runner_up() {
        // Literal layout: 0 = c1, 1 = c2. (g is the forced target, not a candidate literal.)
        const C1: AtomKey = (200, 1, 0);
        const C2: AtomKey = (200, 2, 0);
        let lits = [C1, C2];
        // `:- not g` rejects the {} world (g false there); accepted = choose >=1 path.
        let wv_c1 = [0b01u8]; // {c1}
        let wv_c2 = [0b10u8]; // {c2}
        let wv_both = [0b11u8]; // {c1, c2}
        let accepted: Vec<&[u8]> = vec![&wv_c1, &wv_c2, &wv_both];
        let mut positives = BTreeSet::new();
        positives.insert(C1); // top-1 path committed.

        // Discriminator: the all-candidates-false world (bitset 0) is NOT accepted => GENUINE.
        assert!(
            !accepted.iter().any(|wv| wv[0] == 0),
            "required-goal must reject the all-false world for GENUINE contestation"
        );

        let result = road_not_taken_possible_not_known(&accepted, 2, &lits, &positives);
        assert_eq!(
            result,
            vec![C2],
            "c2 = runner-up necessary-alternative (possible-not-known, net-new vs committed c1); got {result:?}"
        );
    }

    /// (2b) DISCRIMINATOR — a FREE (unconstrained) runner-up also lands in ⋃−⋂,
    /// but TRIVIALLY: with no required-goal the all-candidates-false world is
    /// ACCEPTED, so `r` is possible-not-known for the same reason ANY free atom
    /// is (the false-positive #2/#3 shape), distinguished from genuine
    /// contestation only by the rule-bearing label. => rule-bearing is NECESSARY
    /// but NOT SUFFICIENT; the required-goal (criterion 4) is the trigger. The §1
    /// reduction itself cannot tell the two apart — the discriminator is whether
    /// the all-false world is in the accepted set (i.e. whether the target forces
    /// a choice). Resolves @xlog-claude's free-vs-forced nuance.
    #[test]
    fn free_runner_up_is_trivially_possible_not_genuine() {
        const R: AtomKey = (201, 1, 0);
        let lits = [R];
        let wv_empty = [0b0u8]; // {} accepted: nothing forces a choice.
        let wv_r = [0b1u8]; // {r}
        let accepted: Vec<&[u8]> = vec![&wv_empty, &wv_r];
        let positives = BTreeSet::new();

        let result = road_not_taken_possible_not_known(&accepted, 1, &lits, &positives);
        assert_eq!(
            result,
            vec![R],
            "free r is in ⋃−⋂ — the §1 reduction alone cannot tell it is trivial; got {result:?}"
        );
        // Discriminator: the all-false world IS accepted => TRIVIAL-free, NOT genuine.
        assert!(
            accepted.iter().any(|wv| wv[0] == 0),
            "no required-goal => all-false world accepted => r trivially-free, NOT genuine \
             road-not-taken: rule-bearing alone insufficient; the forced target is the trigger"
        );
    }

    /// (2b) discriminator GATE: `forced_target_confirmed` is true iff the
    /// all-candidates-false world is rejected. Genuine (required-goal) accepted-set
    /// excludes it; the trivially-free accepted-set includes it. This is the
    /// co-equal go/no-go gate with net-new>0, and also the runtime check that the
    /// GPU `:- not g` constraint actually fired.
    #[test]
    fn forced_target_confirmed_distinguishes_genuine_from_trivial_free() {
        // Genuine: accepted = {c1},{c2},{c1,c2}; all-false {} REJECTED by the required goal.
        let g_c1 = [0b01u8];
        let g_c2 = [0b10u8];
        let g_both = [0b11u8];
        let genuine: Vec<&[u8]> = vec![&g_c1, &g_c2, &g_both];
        assert!(
            forced_target_confirmed(&genuine, 2),
            "all-false world rejected => forced-target operative => genuine"
        );
        // Trivial-free: accepted includes {} (all-false) => target did not force a choice
        // (or GPU `:- not g` did not fire).
        let f_empty = [0b00u8];
        let f_r = [0b01u8];
        let trivial: Vec<&[u8]> = vec![&f_empty, &f_r];
        assert!(
            !forced_target_confirmed(&trivial, 2),
            "all-false world accepted => not forced / `:- not g` silent => trivial-free"
        );
        // No accepted world-views => not confirmed (no consistent model).
        assert!(!forced_target_confirmed(&[], 2));
    }

    // ---- TIER-2 grounded-uncertainty export (Phase-1 export-impl) ----

    /// TIER-2 export — the GROUNDED world-fraction marginal per candidate is the
    /// literal fraction of accepted world-views in which the atom holds. For the
    /// `{X,Y}`/`{X}` accepted set: X ∈ both = 1.0; Y ∈ one = 0.5; Z ∈ none = 0.0.
    /// This is the sub-gap-1-SOUND measure (grounded), NOT a Belnap→prob heuristic.
    #[test]
    fn export_returns_grounded_world_fraction_marginals() {
        let wv_a = [0b011u8]; // {X, Y}
        let wv_b = [0b001u8]; // {X}
        let accepted: Vec<&[u8]> = vec![&wv_a, &wv_b];

        let result = export_from_accepted_world_views(&accepted, 3, &lits());

        assert_eq!(
            result.marginals,
            vec![1.0, 0.5, 0.0],
            "grounded world-fraction: X=2/2, Y=1/2, Z=0/2; got {:?}",
            result.marginals
        );
        assert!(
            result.no_substrate_reason.is_none(),
            "2 accepted worlds, 3 candidates => substrate present"
        );
    }

    /// TIER-2 export — ONE accepted-world-view set yields BOTH coupling surfaces:
    /// `marginals` (conditions / Arm-B re-weight) AND `possible_not_known` (§1
    /// proposes / producer). Y is possible-not-known (0.5 marginal); X is committed
    /// (1.0 marginal, excluded from §1); both come from the single export call.
    #[test]
    fn export_yields_both_surfaces_from_one_accepted_set() {
        let wv_a = [0b011u8]; // {X, Y}
        let wv_b = [0b001u8]; // {X}
        let accepted: Vec<&[u8]> = vec![&wv_a, &wv_b];

        let result = export_from_accepted_world_views(&accepted, 3, &lits());

        // conditions-direction: Y carries genuine uncertainty (0.5), X is committed (1.0).
        assert_eq!(result.marginals[1], 0.5, "Y world-fraction");
        // proposes-direction: §1 road-not-taken = {Y} (X committed, Z absent).
        assert_eq!(
            result.possible_not_known,
            vec![Y],
            "§1 road-not-taken; got {:?}",
            result.possible_not_known
        );
    }

    /// TIER-2 export — the engine returns the grounded `⋃−⋂` with NO positives
    /// subtraction (the agreed seam): even an atom that would be a request-positive
    /// is emitted here; DTS applies net-additivity consume-side. Y stays in the
    /// `⋃−⋂` regardless, and its marginal is reported.
    #[test]
    fn export_returns_union_minus_intersection_without_positives_subtraction() {
        let wv_a = [0b011u8]; // {X, Y}
        let wv_b = [0b001u8]; // {X}
        let accepted: Vec<&[u8]> = vec![&wv_a, &wv_b];

        let result = export_from_accepted_world_views(&accepted, 3, &lits());

        assert_eq!(
            result.possible_not_known,
            vec![Y],
            "engine returns grounded ⋃−⋂ (Y possible-not-known); positives subtraction is DTS-side; got {:?}",
            result.possible_not_known
        );
        assert_eq!(result.marginals[1], 0.5, "Y marginal reported");
    }

    /// TIER-2 export — empty accepted set is a LOUD no-substrate diagnostic
    /// (no consistent model), NOT a silent all-zero result.
    #[test]
    fn export_empty_accepted_is_loud_no_substrate() {
        let accepted: Vec<&[u8]> = Vec::new();
        let result = export_from_accepted_world_views(&accepted, 3, &lits());
        assert!(
            result.no_substrate_reason.is_some(),
            "empty accepted => LOUD no-substrate"
        );
        assert!(result.possible_not_known.is_empty());
    }

    /// TIER-2 export — a single candidate literal (no alternative) is a LOUD
    /// no-substrate diagnostic: contestation requires >=2 competing alternatives
    /// (the sub-gap-2 lesson — single-world / no-alternative is not substrate).
    #[test]
    fn export_single_candidate_is_loud_no_substrate() {
        let wv = [0b1u8]; // {X}
        let accepted: Vec<&[u8]> = vec![&wv];
        let one_lit: [AtomKey; 1] = [X];
        let result = export_from_accepted_world_views(&accepted, 1, &one_lit);
        assert!(
            result.no_substrate_reason.is_some(),
            "single candidate => <2 alternatives => LOUD no-substrate (sub-gap-2)"
        );
    }

    /// TIER-2 readback decode — the STRIDE-GOTCHA gate. The device `world_views`
    /// buffer rows are `world_view_stride = max(max_worlds, bitset_bytes)` bytes,
    /// so each candidate's literal-bitset is the PREFIX of its stride-row. A naive
    /// contiguous slice (`i * bitset_bytes`) would read the wrong row once the
    /// stride is padded. literal_count=3 (bitset_bytes=1), max_worlds=4 (stride=4):
    /// candidate 2 must be read at offset `2*4=8`, NOT `2*1=2`.
    #[test]
    fn decode_respects_world_view_stride_padding() {
        let buf: Vec<u8> = vec![
            0b011, 0xFF, 0xFF, 0xFF, // candidate 0 bitset {X,Y} + padding
            0b001, 0xFF, 0xFF, 0xFF, // candidate 1 bitset {X}
            0b100, 0xFF, 0xFF, 0xFF, // candidate 2 bitset {Z}
        ];
        let accepted = [0usize, 2];
        let bitsets = decode_accepted_bitsets(&buf, &accepted, 3, 4);
        assert_eq!(bitsets.len(), 2);
        assert_eq!(
            bitsets[0],
            &[0b011u8],
            "candidate 0 bitset-prefix (padding skipped)"
        );
        assert_eq!(
            bitsets[1],
            &[0b100u8],
            "candidate 2 read at offset 2*stride=8, NOT 2*bitset_bytes=2"
        );
    }

    /// TIER-2 readback decode — when there is no stride padding
    /// (`max_worlds <= bitset_bytes`) the rows are contiguous bitsets.
    #[test]
    fn decode_contiguous_when_no_stride_padding() {
        let buf: Vec<u8> = vec![0b011, 0b001, 0b100];
        let accepted = [0usize, 1, 2];
        let bitsets = decode_accepted_bitsets(&buf, &accepted, 3, 1);
        assert_eq!(
            bitsets,
            vec![&[0b011u8][..], &[0b001u8][..], &[0b100u8][..]]
        );
    }

    /// TIER-2 readback → export roundtrip over a PADDED-stride buffer: the decoded
    /// bitsets feed the pure export, and the grounded marginals are correct
    /// (X=2/2=1.0, Y=1/2=0.5, Z=0/2=0.0) — proving the stride handling is sound
    /// end-to-end, not just in isolation.
    #[test]
    fn decode_then_export_roundtrip_over_strided_buffer() {
        let buf: Vec<u8> = vec![
            0b011, 0xFF, 0xFF, 0xFF, // {X, Y}
            0b001, 0xFF, 0xFF, 0xFF, // {X}
        ];
        let accepted = [0usize, 1];
        let bitsets = decode_accepted_bitsets(&buf, &accepted, 3, 4);
        let result = export_from_accepted_world_views(&bitsets, 3, &lits());
        assert_eq!(
            result.marginals,
            vec![1.0, 0.5, 0.0],
            "strided readback → grounded marginals; got {:?}",
            result.marginals
        );
    }

    /// TIER-2 host-side wrapper — the full pure pipeline the pyxlog binding calls
    /// AFTER its provider untracked-dtoh readback of `world_views`: host buffer →
    /// stride-aware decode → grounded export. The binding owns the device readback
    /// (provider context); this fn is the pure, host-TDD-able core, so the only
    /// GPU-coupled step left is the buffer copy the binding performs.
    #[test]
    fn host_buffer_wrapper_decodes_and_exports() {
        // stride=4 (literal_count=3, max_worlds=4); accepted 0={X,Y}, 1={X}.
        let buf: Vec<u8> = vec![0b011, 0xFF, 0xFF, 0xFF, 0b001, 0xFF, 0xFF, 0xFF];
        let accepted = [0usize, 1];
        let result = epistemic_export_from_host_buffer(&buf, &accepted, 3, 4, &lits());
        assert_eq!(
            result.marginals,
            vec![1.0, 0.5, 0.0],
            "grounded marginals; got {:?}",
            result.marginals
        );
        assert_eq!(
            result.possible_not_known,
            vec![Y],
            "§1 ⋃−⋂; got {:?}",
            result.possible_not_known
        );
        assert!(
            result.no_substrate_reason.is_none(),
            "2 accepted, 3 candidates => substrate present"
        );
    }

    /// TIER-2 compact-readback (scaling follow-up, decode-side) — the per-accepted-index
    /// reader (xlog-cuda, to lift the 4096B whole-buffer cap for the ≤20 live range)
    /// packs ONLY the accepted bitsets contiguously into a compact buffer
    /// (`accepted_count × bitset_bytes`), NOT the exponential `2^lit` candidate space.
    /// The SAME `decode_accepted_bitsets` handles it: `max_worlds = 0` => `stride =
    /// bitset_bytes` => contiguous rows over `0..accepted_count`. Marginal/§1 are over
    /// the literals (bits), not candidate-row identity, so the result is identical.
    /// => the export-lane needs NO new decode code for the scaling fix; the follow-up
    /// reduces to the cuda per-offset reader alone.
    #[test]
    fn decode_handles_compact_accepted_only_buffer() {
        // Compact buffer: accepted bitsets only, 1 byte each (literal_count=3), no padding.
        let compact: Vec<u8> = vec![0b011, 0b001]; // {X,Y}, {X}
        let indices = [0usize, 1];
        let bitsets = decode_accepted_bitsets(&compact, &indices, 3, 0); // max_worlds=0 => contiguous
        let result = export_from_accepted_world_views(&bitsets, 3, &lits());
        assert_eq!(
            result.marginals,
            vec![1.0, 0.5, 0.0],
            "compact-readback decode → same grounded marginals; got {:?}",
            result.marginals
        );
        assert_eq!(
            result.possible_not_known,
            vec![Y],
            "§1 ⋃−⋂ unchanged under compact readback"
        );
    }
}
