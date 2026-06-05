//! EGB-07 / v0.9.2 ITEM B: FAEEL founded-model EXTENSION over the production GPU
//! lowering path.
//!
//! Under default FAEEL, an atom/tuple supported ONLY by circular modal self-support
//! (`possible p`/`know p` with no independent founded derivation) is NOT in the founded
//! model: it is simply ABSENT. This is the standard founded/equilibrium semantics, and
//! it is executed (the program plans and compiles, then materializes the founded
//! extension on the GPU runtime), NOT rejected with an unsupported-construct error.
//!
//! These pilots run through `plan_epistemic_gpu_execution` /
//! `compile_epistemic_gpu_execution` (the accepted-execution boundary) and inspect the
//! reduced ordinary base produced by `reduce_epistemic_program_to_ordinary`, which is
//! the structural foundedness DECISION boundary: an unfounded circular self-support
//! rule is DROPPED from the reduced base (so its head is absent from the founded model),
//! while a founded rule is KEPT. The founded EXTENSION itself (exact tuples / exact
//! `rows: 0` empty result) is asserted on device in
//! `crates/xlog-runtime/tests/test_epistemic_gpu_workspace.rs`. The FAEEL-vs-G91 mode
//! difference (empty vs accepted) is verified in `test_epistemic_g91.rs`.

use xlog_logic::ast::BodyLiteral;
use xlog_logic::epistemic::{
    compile_epistemic_gpu_execution, plan_epistemic_gpu_execution,
    reduce_epistemic_program_to_ordinary,
};
use xlog_logic::parse_program;

/// Plan + compile a FAEEL program (it must NOT be rejected) and return its reduced
/// ordinary base for structural foundedness inspection.
fn reduced_base(src: &str) -> xlog_logic::ast::Program {
    let program = parse_program(src).unwrap();
    // The unfounded circular self-support case is now a DEFINED FAEEL result (empty
    // founded model), so planning + compiling must succeed, not raise a foundedness
    // rejection.
    plan_epistemic_gpu_execution(&program)
        .unwrap_or_else(|e| panic!("FAEEL founded-extension program must plan, got: {e}\n{src}"));
    compile_epistemic_gpu_execution(&program).unwrap_or_else(|e| {
        panic!("FAEEL founded-extension program must compile, got: {e}\n{src}")
    });
    reduce_epistemic_program_to_ordinary(&program)
}

/// Count reduced ordinary rules that derive `predicate` from a NON-empty body (i.e.
/// founding rules; pure facts have an empty body).
fn founding_rule_count(program: &xlog_logic::ast::Program, predicate: &str) -> usize {
    program
        .rules
        .iter()
        .filter(|rule| rule.head.predicate == predicate && !rule.body.is_empty())
        .count()
}

/// Whether any reduced ordinary rule for `predicate` retains a residual epistemic
/// literal (it must not — modal literals are stripped/resolved/dropped by reduction).
fn has_residual_modal(program: &xlog_logic::ast::Program, predicate: &str) -> bool {
    program.rules.iter().any(|rule| {
        rule.head.predicate == predicate
            && rule
                .body
                .iter()
                .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
    })
}

fn accepts(src: &str) {
    let program = parse_program(src).unwrap();
    plan_epistemic_gpu_execution(&program)
        .unwrap_or_else(|e| panic!("expected FAEEL acceptance for:\n{src}\ngot: {e}"));
    compile_epistemic_gpu_execution(&program)
        .unwrap_or_else(|e| panic!("expected FAEEL compile acceptance for:\n{src}\ngot: {e}"));
}

// === K1: zero-arity foundedness ===

#[test]
fn k1_faeel_unfounded_zero_arity_self_support_is_dropped_from_founded_base() {
    // `p() :- possible p()` is supported only by circular self-support: under FAEEL it
    // is UNFOUNDED, so the founded model is EMPTY. The reduced base drops the rule
    // entirely (no founding rule remains for p), making `p` absent from the model.
    // Exact `rows: 0` is asserted on device in
    // `parsed_faeel_unfounded_zero_arity_self_support_materializes_empty_on_gpu`.
    let reduced = reduced_base("p() :- possible p().");
    assert_eq!(
        founding_rule_count(&reduced, "p"),
        0,
        "unfounded circular self-support rule must be dropped from the founded base, \
         leaving p with no founding rule (empty founded model): {:?}",
        reduced.rules
    );
    assert!(!has_residual_modal(&reduced, "p"));
}

#[test]
fn k1_faeel_accepts_zero_arity_self_support_with_independent_foundation() {
    // possible p() is self-supporting, but p() also has an independent, non-circular
    // foundation via base(). The founded model contains p, so the founding rule is
    // KEPT in the reduced base. Exact `rows: 1` is asserted on device.
    let src = "p() :- possible p().\np() :- base().\nbase().";
    accepts(src);
    let reduced = reduced_base(src);
    assert!(
        founding_rule_count(&reduced, "p") >= 1,
        "the founded `p :- base` rule must survive reduction: {:?}",
        reduced.rules
    );
}

// === K2: nonzero-arity foundedness per tuple key ===

#[test]
fn k2_faeel_accepts_nonzero_self_support_when_support_subsumes_modal_domain() {
    // Support rule body { dom(X) } subsumes the modal rule's tuple-key domain
    // { dom(X) }: every modal tuple key has an independent founded proof, so the
    // self-support rule is FOUNDED and KEPT. Exact founded tuples on device.
    let src = "p(X) :- dom(X), possible p(X).\np(X) :- dom(X).\ndom(1).";
    accepts(src);
    let reduced = reduced_base(src);
    assert!(
        founding_rule_count(&reduced, "p") >= 1,
        "founded self-support (support subsumes modal domain) must survive: {:?}",
        reduced.rules
    );
}

#[test]
fn k2_faeel_accepts_nonzero_self_support_with_multi_atom_subsuming_support() {
    accepts("p(X) :- dom(X), base(X), possible p(X).\np(X) :- dom(X), base(X).\ndom(1).\nbase(1).");
}

#[test]
fn k2_faeel_predicate_level_support_keeps_only_the_founded_tuple_rule() {
    // Support exists for p(c) only, but the modal rule self-supports p(X) for every X
    // in dom. Predicate-level support must NOT found the open tuple key: the unfounded
    // open self-support rule is DROPPED, and the founded `p(c) :- base(c)` rule is
    // KEPT. The founded model is therefore exactly {p(c)} (not every dom tuple).
    // Exact founded tuple ({9}) asserted on device.
    let src = "p(X) :- dom(X), possible p(X).\np(9) :- base(9).\ndom(1).\nbase(9).";
    let reduced = reduced_base(src);
    // The open self-support rule (body has dom(X)+modal) is dropped; only the founded
    // p(9) rule remains.
    let open_self_support_remains = reduced.rules.iter().any(|rule| {
        rule.head.predicate == "p"
            && rule
                .head
                .terms
                .iter()
                .any(|t| matches!(t, xlog_logic::ast::Term::Variable(_)))
            && rule
                .body
                .iter()
                .any(|lit| matches!(lit, BodyLiteral::Positive(a) if a.predicate == "dom"))
    });
    assert!(
        !open_self_support_remains,
        "unfounded open-tuple self-support rule must be dropped: {:?}",
        reduced.rules
    );
    assert!(
        reduced.rules.iter().any(|rule| rule.head.predicate == "p"
            && rule
                .body
                .iter()
                .any(|lit| matches!(lit, BodyLiteral::Positive(a) if a.predicate == "base"))),
        "founded p(9):-base(9) rule must survive: {:?}",
        reduced.rules
    );
    assert!(!has_residual_modal(&reduced, "p"));
}

#[test]
fn k2_faeel_support_more_restrictive_keeps_only_the_founded_rule() {
    // Support body { dom(X), base(X) } is strictly more restrictive than the modal body
    // { dom(X) }: the modal can conclude p(X) for dom(X) tuples that have no independent
    // foundation (base(X) false). The unfounded open self-support rule is DROPPED; the
    // founded `p(X):-dom(X),base(X)` rule is KEPT, so the founded model is exactly the
    // base-founded tuples. With dom(1),base(1) the founded model is {p(1)}.
    let src = "p(X) :- dom(X), possible p(X).\np(X) :- dom(X), base(X).\ndom(1).\nbase(1).";
    let reduced = reduced_base(src);
    assert!(
        reduced.rules.iter().any(|rule| rule.head.predicate == "p"
            && rule
                .body
                .iter()
                .any(|lit| matches!(lit, BodyLiteral::Positive(a) if a.predicate == "base"))),
        "founded p:-dom,base rule must survive reduction: {:?}",
        reduced.rules
    );
    assert!(!has_residual_modal(&reduced, "p"));
}

#[test]
fn k2_faeel_unfounded_nonzero_self_support_with_domain_is_dropped() {
    // No founding rule for p at all: pure modal self-support over a bound domain
    // (`p(X) :- dom(X), possible p(X)`, X bound by dom so safe). The founded model is
    // EMPTY: the unfounded rule is dropped, leaving p with no founding rule. Exact
    // `rows: 0` asserted on device.
    let reduced = reduced_base("p(X) :- dom(X), possible p(X).\ndom(1).");
    assert_eq!(
        founding_rule_count(&reduced, "p"),
        0,
        "pure domain-bound modal self-support must be dropped (empty founded model): {:?}",
        reduced.rules
    );
    assert!(!has_residual_modal(&reduced, "p"));
}
