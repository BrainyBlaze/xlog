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
}
