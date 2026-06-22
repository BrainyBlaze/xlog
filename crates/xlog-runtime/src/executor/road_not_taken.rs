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
        assert!(result.is_empty(), "only Y was possible-not-known; got {result:?}");
    }

    /// No accepted world-views (no consistent model) => empty road-not-taken set.
    #[test]
    fn empty_accepted_yields_empty_source() {
        let accepted: Vec<&[u8]> = Vec::new();
        let positives = BTreeSet::new();
        let result = road_not_taken_possible_not_known(&accepted, 3, &lits(), &positives);
        assert!(result.is_empty());
    }
}
