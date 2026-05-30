//! EGB-07: FAEEL founded self-support over the production GPU lowering path.
//!
//! These pilots run through `plan_epistemic_gpu_execution` /
//! `compile_epistemic_gpu_execution` (the accepted-execution boundary), NOT the
//! bounded-interpretation `evaluate_faeel_candidate` surface. The GOAL's
//! self-support / per-tuple-key foundedness semantics live only in the EIR
//! foundedness guard on this path.

use xlog_logic::epistemic::{compile_epistemic_gpu_execution, plan_epistemic_gpu_execution};
use xlog_logic::parse_program;

fn plan_err(src: &str) -> String {
    let program = parse_program(src).unwrap();
    match plan_epistemic_gpu_execution(&program) {
        Ok(_) => panic!("expected FAEEL foundedness rejection for:\n{src}"),
        Err(e) => e.to_string(),
    }
}

fn assert_accepts(src: &str) {
    let program = parse_program(src).unwrap();
    plan_epistemic_gpu_execution(&program)
        .unwrap_or_else(|e| panic!("expected FAEEL acceptance for:\n{src}\ngot: {e}"));
    // Acceptance must also survive the full compile boundary (reduced runtime
    // plan), not just the plan-level guard.
    compile_epistemic_gpu_execution(&program)
        .unwrap_or_else(|e| panic!("expected FAEEL compile acceptance for:\n{src}\ngot: {e}"));
}

// === K1: zero-arity foundedness ===

#[test]
fn k1_faeel_rejects_unfounded_zero_arity_self_support() {
    let err = plan_err("p() :- possible p().");
    assert!(err.contains("FAEEL foundedness guard"), "got: {err}");
    assert!(err.contains("self-supported"), "got: {err}");
    assert!(err.contains("p/0"), "got: {err}");
}

#[test]
fn k1_faeel_accepts_zero_arity_self_support_with_independent_foundation() {
    // possible p() is self-supporting, but p() also has an independent,
    // non-circular foundation via base().
    assert_accepts("p() :- possible p().\np() :- base().\nbase().");
}

// === K2: nonzero-arity foundedness per tuple key ===

#[test]
fn k2_faeel_accepts_nonzero_self_support_when_support_subsumes_modal_domain() {
    // Support rule body { dom(X) } subsumes the modal rule's tuple-key domain
    // { dom(X) }: every modal tuple key has an independent founded proof.
    assert_accepts("p(X) :- dom(X), possible p(X).\np(X) :- dom(X).\ndom(1).");
}

#[test]
fn k2_faeel_accepts_nonzero_self_support_with_multi_atom_subsuming_support() {
    assert_accepts(
        "p(X) :- dom(X), base(X), possible p(X).\np(X) :- dom(X), base(X).\ndom(1).\nbase(1).",
    );
}

#[test]
fn k2_faeel_rejects_predicate_level_support_without_matching_tuple_key() {
    // Support exists for p(c) only, but the modal rule self-supports p(X) for
    // every X in dom. Predicate-level support must NOT found the open tuple key.
    let err = plan_err("p(X) :- dom(X), possible p(X).\np(c) :- base(c).\ndom(1).\nbase(c).");
    assert!(err.contains("FAEEL foundedness guard"), "got: {err}");
    assert!(err.contains("nonzero-arity self-supported"), "got: {err}");
    assert!(err.contains("p/1"), "got: {err}");
}

#[test]
fn k2_faeel_rejects_support_more_restrictive_than_modal_domain() {
    // Support body { dom(X), base(X) } is strictly more restrictive than the
    // modal body { dom(X) }: modal can conclude p(X) for dom(X) tuples that
    // have no independent foundation (base(X) false). Statically unprovable, so
    // fail closed by design.
    let err =
        plan_err("p(X) :- dom(X), possible p(X).\np(X) :- dom(X), base(X).\ndom(1).\nbase(1).");
    assert!(err.contains("FAEEL foundedness guard"), "got: {err}");
}

#[test]
fn k2_faeel_rejects_unfounded_nonzero_modal_cycle() {
    // No founding rule for p at all: pure modal self-support cycle.
    let err = plan_err("p(X) :- dom(X), possible p(X).\ndom(1).");
    assert!(err.contains("FAEEL foundedness guard"), "got: {err}");
}

// === K4: precise missing-foundation diagnostic ===

#[test]
fn k4_rejection_names_missing_foundation_tuple_key() {
    let err = plan_err("p(X) :- dom(X), possible p(X).\ndom(1).");
    assert!(
        err.contains("no independent founded support proves the tuple key p(X)"),
        "diagnostic must name the missing tuple-key foundation, got: {err}"
    );
}

#[test]
fn k4_zero_arity_rejection_names_missing_foundation() {
    let err = plan_err("p() :- possible p().");
    assert!(
        err.contains("no independent founded support proves p()"),
        "diagnostic must name the missing foundation, got: {err}"
    );
}
