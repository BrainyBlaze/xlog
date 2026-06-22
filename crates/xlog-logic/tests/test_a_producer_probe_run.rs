//! ST-TRC (A)-producer epistemic net-new probe — run-part (@xlog-claude-2 lane).
//!
//! Pairs with @xlog-claude's construct-part (`build_probe_program`: ProbeExport JSON ->
//! `(program, candidate_atoms, committed_atoms, positives)`). This file owns the RUN:
//! GENERATE 2^N truth-assignments over `candidate_atoms`, classify each against the
//! background `program` via the CPU `run_generate_propagate_test` oracle (no GPU/RunPod),
//! then compute road-not-taken net-new = `⋃(accepted) − ⋂(accepted) − positives` over the
//! candidate atoms (the atom-set adaptation of the §1 reduction `aa8b423c`).
//!
//! Decoupled from the serde/JSON construct on purpose: `run_probe` takes the raw pieces, so
//! the run logic + trivial-detector are unit-tested here with hand-built programs (via
//! `parse_program`), independent of @dts-dlm-main's sample export (which is blocked on a
//! data-availability decision). The real-sample bin wires `build_probe_program -> run_probe`.
//!
//! Validity (probe-spec 2f20c8c): a candidate is KNOWN (∀, excluded) only if the background
//! committed-context + rules DERIVE it; FALSE (excluded) only if a constraint forbids it;
//! the genuinely underivable-but-consistent candidates are the road-not-taken. Because all
//! 2^N subsets include the empty guess (vacuously consistent), a probe over a background that
//! neither derives nor forbids any candidate accepts every subset -> trivially "all possible"
//! (false-positive). `trivial_suspect = (accepted_count == generated)` is the vacuity guard.

use xlog_logic::ast::{Atom, Program, Term};
use xlog_logic::epistemic::{
    run_generate_propagate_test, EpistemicInterpretation, GeneratePropagateTestConfig,
};
use xlog_logic::parse_program;

/// A ground atom over integer args (the probe atom shape: `pred(arg0, arg1, ...)`).
fn atom(pred: &str, args: &[i64]) -> Atom {
    Atom {
        predicate: pred.to_string(),
        terms: args.iter().copied().map(Term::Integer).collect(),
    }
}

/// Comparable key for a ground integer atom (for positives membership / equality).
fn atom_key(a: &Atom) -> (String, Vec<i64>) {
    let args = a
        .terms
        .iter()
        .map(|t| match t {
            Term::Integer(v) => *v,
            // probe atoms are ground integers; a non-int term is out-of-contract -> sentinel.
            _ => i64::MIN,
        })
        .collect();
    (a.predicate.clone(), args)
}

#[derive(Debug)]
struct ProbeResult {
    /// |road-not-taken| = possible-not-known candidate atoms ∉ positives.
    net_new: usize,
    /// The road-not-taken atoms themselves.
    road_not_taken: Vec<Atom>,
    /// |candidate_atoms| = the GPT literal-count (tractability is 2^literal_count).
    literal_count: usize,
    /// 2^literal_count generated truth-assignments.
    generated: usize,
    /// Accepted (consistent) world-views.
    accepted_count: usize,
    /// |⋂(accepted) ∩ candidates| -- candidate atoms KNOWN (∀, derivation-forced). Info only.
    known_count: usize,
    /// |⋃(accepted) ∩ candidates| -- candidate atoms POSSIBLE (∃). Info only.
    possible_count: usize,
    /// Vacuity guard: no world-view pruned (no constraint touched candidates) -> result trivial.
    trivial_suspect: bool,
}

/// Run the (A)-producer net-new probe over a background `program` and `candidate_atoms`.
///
/// Generates all 2^N truth-assignments (candidate `i` = `committed ∪ {candidate_atoms[j] :
/// bit j of i}`); since we build them, candidate `i`'s subset mask IS `i`, so accepted indices
/// map directly back to subset masks without reading the (private) `EpistemicInterpretation`.
fn run_probe(
    program: &Program,
    candidate_atoms: &[Atom],
    committed_atoms: &[Atom],
    positives: &[Atom],
) -> ProbeResult {
    let n = candidate_atoms.len();
    assert!(n <= 31, "probe literal-count {n} exceeds the ≤31 hard bound (no chunking)");
    let total = 1usize << n;
    let all_bits = total - 1;

    let mut candidates = Vec::with_capacity(total);
    for mask in 0..total {
        let mut interp = EpistemicInterpretation::new();
        for a in committed_atoms {
            interp = interp
                .with_known_terms(a.predicate.clone(), a.terms.clone())
                .expect("committed atom key");
        }
        for (j, a) in candidate_atoms.iter().enumerate() {
            if mask & (1 << j) != 0 {
                interp = interp
                    .with_known_terms(a.predicate.clone(), a.terms.clone())
                    .expect("candidate atom key");
            }
        }
        candidates.push(interp);
    }

    let outcome = run_generate_propagate_test(
        program,
        candidates,
        GeneratePropagateTestConfig { max_candidates: total },
    )
    .expect("generate-propagate-test run");
    let accepted = &outcome.accepted_candidate_indices;

    // ⋃ / ⋂ over accepted subset masks, restricted to candidate-atom bits.
    let mut in_some = 0usize;
    let mut in_all = if accepted.is_empty() { 0 } else { all_bits };
    for &idx in accepted {
        let mask = idx & all_bits; // candidate idx was built with subset == idx
        in_some |= mask;
        in_all &= mask;
    }
    let possible_not_known = in_some & !in_all & all_bits;

    let pos_keys: std::collections::BTreeSet<(String, Vec<i64>)> =
        positives.iter().map(atom_key).collect();

    let mut road_not_taken = Vec::new();
    for j in 0..n {
        if possible_not_known & (1 << j) != 0 && !pos_keys.contains(&atom_key(&candidate_atoms[j]))
        {
            road_not_taken.push(candidate_atoms[j].clone());
        }
    }

    ProbeResult {
        net_new: road_not_taken.len(),
        road_not_taken,
        literal_count: n,
        generated: total,
        accepted_count: accepted.len(),
        known_count: (in_all & all_bits).count_ones() as usize,
        possible_count: (in_some & all_bits).count_ones() as usize,
        trivial_suspect: accepted.len() == total,
    }
}

fn has_atom(atoms: &[Atom], pred: &str, args: &[i64]) -> bool {
    let want = atom_key(&atom(pred, args));
    atoms.iter().any(|a| atom_key(a) == want)
}

/// DECISIVE oracle-semantics finding (regression record): an ORDINARY relational constraint
/// `:- a(1), b(1)` does NOT prune a GPT candidate world-view, even when the candidate holds
/// both a(1) and b(1) as epistemic-`known`. `evaluate_epistemic_candidate` (epistemic.rs:2482)
/// rejects ONLY on (i) candidate self-contradiction (atom known∩rejected) and (ii) EPISTEMIC
/// rule-body literals (`BodyLiteral::Epistemic`); it never reads `program.constraints` and
/// skips non-epistemic rule-body literals. So all 2^N worlds are accepted => the result is
/// the vacuous all-possible false-positive (`trivial_suspect`). This is why the probe's
/// pruning CANNOT come from ordinary ontology constraints on the CPU GPT path; it must come
/// from epistemic (know/possible) rule-body literals over the candidate predicates.
#[test]
fn ordinary_relational_constraint_does_not_prune_gpt_candidates() {
    let program = parse_program(":- a(1), b(1).").expect("parse");
    let candidates = [atom("a", &[1]), atom("b", &[1])];
    let result = run_probe(&program, &candidates, &[], &[]);

    assert_eq!(
        result.accepted_count, result.generated,
        "ordinary constraints are ignored by the GPT candidate-test => every world accepted"
    );
    assert!(
        result.trivial_suspect,
        "no pruning => vacuous all-possible => trivial false-positive (NOT a usable result)"
    );
    assert_eq!(
        result.known_count, 0,
        "nothing forces a candidate true in every world (no epistemic derivation)"
    );
}

/// Vacuity guard: with NO constraints over the candidates, every 2^N world is accepted, so
/// every candidate is trivially possible-not-known. net_new>0 here is a FALSE positive and
/// `trivial_suspect` must flag it (the finding-#2 dual-failure: empty subset accepted =>
/// ⋂∩candidate=∅ => all possible, but meaningless without constraining/deriving context).
#[test]
fn no_constraint_background_is_flagged_trivial() {
    // A background with a fact unrelated to the candidates: nothing derives/forbids them.
    let program = parse_program("unrelated(0).").expect("parse");
    let candidates = [atom("a", &[1]), atom("b", &[1])];
    let result = run_probe(&program, &candidates, &[], &[]);

    assert_eq!(
        result.accepted_count, result.generated,
        "no constraint touches the candidates => every world accepted"
    );
    assert!(
        result.trivial_suspect,
        "all-accepted => vacuous all-possible => trivial false-positive must be flagged"
    );
    assert_eq!(result.known_count, 0, "nothing forces a candidate true in all worlds");
}

/// Net-additivity: a candidate already in request positives is subtracted (the 2b
/// duplicate-saturation guard), even when it is epistemically possible-not-known.
#[test]
fn positives_are_subtracted_for_net_additivity() {
    let program = parse_program(":- a(1), b(1).").expect("parse");
    let candidates = [atom("a", &[1]), atom("b", &[1])];
    let positives = [atom("a", &[1])]; // a(1) is already an induction-input positive.
    let result = run_probe(&program, &candidates, &[], &positives);

    assert!(
        !has_atom(&result.road_not_taken, "a", &[1]),
        "a(1) ∈ positives => not net-new => excluded; got {:?}",
        result.road_not_taken
    );
    assert!(
        has_atom(&result.road_not_taken, "b", &[1]),
        "b(1) is still net-new road-not-taken; got {:?}",
        result.road_not_taken
    );
    assert_eq!(result.net_new, 1, "only b(1) remains after subtracting positive a(1)");
    assert_eq!(result.literal_count, 2, "literal-count = |candidate_atoms| (tractability 2^2)");
}
