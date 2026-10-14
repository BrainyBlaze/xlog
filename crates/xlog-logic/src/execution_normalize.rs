//! Shared source normalization used by execution and static extraction.

use crate::{Atom, BodyLiteral, EpistemicLiteral, EpistemicOp, Program, Rule, Term};
use std::collections::BTreeSet;
use xlog_core::{Result, XlogError};

/// Normalize a parsed program through the pre-compilation passes used by execution.
///
/// This helper does not resolve imports; import-aware callers merge them first. It
/// expands user-defined functions with the entry program's recursion limit, normalizes
/// meta and list builtins, and desugars shared-variable epistemic constraints.
pub fn normalize_program_for_execution(mut program: Program) -> Result<Program> {
    if program.authored_constraint_source_bound.is_some() {
        program.validate_prepared_authored_constraint_identity()?;
    } else {
        program.prepare_authored_constraint_identity_at_root()?;
    }
    let max_recursion = program.directives.max_recursion_depth_or_default();
    let expanded = crate::expand_program_functions_owned(program, max_recursion)
        .map_err(|e| XlogError::Compilation(e.to_string()))?;
    let normalized = crate::normalize_meta_builtins_owned(expanded)?;
    let listed = crate::normalize_list_builtins_owned(normalized)?;
    Ok(desugar_shared_variable_epistemic_constraints(listed))
}

/// Desugar a shared-variable epistemic constraint — a constraint with at least one
/// epistemic literal and a variable appearing in more than one term position across the body
/// (the join `:- know p(X), possible q(X).`, the diagonal `:- know p(X, X).`, or the
/// negated-difference `:- q(X), not know p(X).`) — into an ordinary extraction rule plus a
/// single-occurrence modal over it:
///
/// ```text
///   :- BodyLit1, BodyLit2, ..., BodyLitN.
///        ==> __epi_join_N(Vars) :- ord(BodyLit1), ..., ord(BodyLitN).
///            :- know __epi_join_N(Vars).
/// ```
///
/// where `ord` ordinary-izes each modal literal (`know/possible r(..)` -> `r(..)`,
/// `not know/possible r(..)` -> `not r(..)`) and keeps non-modal literals unchanged. For a
/// base/EDB or purely-ordinary-derived modal target `know r == possible r == r`, so the
/// ordinary join `__epi_join_N` is exactly the set of variable bindings the constraint
/// forbids; the single-occurrence `:- know __epi_join_N(Vars)` then routes through the
/// existing variable-keyed world-view constraint path, which prunes the world view to empty —
/// with NO new kernel. Applied at the normalization choke point so BOTH the reduced ordinary
/// materialization and the epistemic planner observe the helper relation (an EIR-only rewrite
/// is accepted at planning but the helper is never materialized).
///
/// Guarded to non-modal-derived targets (where the `know == possible == ordinary`
/// equivalence holds); a constraint with a modal-derived target is left unchanged and falls
/// through to the core compiler's existing shared-variable rejection. Single-occurrence
/// variable-keyed modal, distinct-variable multi-literal, and ground constraints have no
/// repeated variable and are likewise untouched.
fn desugar_shared_variable_epistemic_constraints(mut program: Program) -> Program {
    // A predicate defined by any rule carrying an epistemic body literal is "modal-derived":
    // for it `know p`/`possible p` is NOT equal to the ordinary `p`, so ordinary-izing it
    // would be UNSOUND. Restrict the desugaring to base/EDB or purely-ordinary-derived
    // targets (where `know p == possible p == p`), the case for base tuple-key targets.
    let modal_derived: BTreeSet<String> = program
        .rules
        .iter()
        .filter(|rule| {
            rule.body
                .iter()
                .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
        })
        .map(|rule| rule.head.predicate.clone())
        .collect();
    let mut extraction_rules: Vec<Rule> = Vec::new();
    let mut counter = 0usize;
    for constraint in &mut program.constraints {
        let has_epistemic = constraint
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)));
        if !has_epistemic || !constraint_has_shared_variable(&constraint.body) {
            continue;
        }
        // Sound only when EVERY modal target is non-modal-derived (know == possible == ord).
        let has_modal_derived_target = constraint.body.iter().any(|lit| {
            matches!(lit, BodyLiteral::Epistemic(e) if modal_derived.contains(&e.atom.predicate))
        });
        if has_modal_derived_target {
            continue;
        }
        let distinct = distinct_body_variables(&constraint.body);
        let helper = format!("__epi_join_{counter}");
        counter += 1;
        let helper_terms: Vec<Term> = distinct.iter().map(|v| Term::Variable(v.clone())).collect();
        let helper_body: Vec<BodyLiteral> = constraint
            .body
            .iter()
            .map(ordinaryize_modal_literal)
            .collect();
        extraction_rules.push(Rule {
            head: Atom {
                predicate: helper.clone(),
                terms: helper_terms.clone(),
            },
            body: helper_body,
        });
        // Replace the whole constraint with a single-occurrence modal over the join helper.
        constraint.body = vec![BodyLiteral::Epistemic(EpistemicLiteral {
            op: EpistemicOp::Know,
            negated: false,
            atom: Atom {
                predicate: helper,
                terms: helper_terms,
            },
        })];
    }
    program.rules.extend(extraction_rules);
    program
}

/// Replace a modal literal with its ordinary counterpart (`know/possible r` -> `r`,
/// `not know/possible r` -> `not r`); non-modal literals are returned unchanged. Sound for
/// the shared-variable constraint desugaring when the modal target is non-modal-derived,
/// where `know r == possible r == r`.
fn ordinaryize_modal_literal(lit: &BodyLiteral) -> BodyLiteral {
    match lit {
        BodyLiteral::Epistemic(e) if e.negated => BodyLiteral::Negated(e.atom.clone()),
        BodyLiteral::Epistemic(e) => BodyLiteral::Positive(e.atom.clone()),
        other => other.clone(),
    }
}

/// True if some variable occurs in more than one atom term position across the constraint
/// body — the signature of a join / diagonal / negated-difference the core compiler rejects.
fn constraint_has_shared_variable(body: &[BodyLiteral]) -> bool {
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for lit in body {
        if let Some(atom) = lit.atom() {
            for term in &atom.terms {
                if let Term::Variable(name) = term {
                    *counts.entry(name.clone()).or_insert(0) += 1;
                }
            }
        }
    }
    counts.values().any(|&count| count > 1)
}

/// Ordered DISTINCT variable names appearing in atom positions across the constraint body
/// (first-appearance order), used as the extracted helper relation's columns.
fn distinct_body_variables(body: &[BodyLiteral]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut order = Vec::new();
    for lit in body {
        if let Some(atom) = lit.atom() {
            for term in &atom.terms {
                if let Term::Variable(name) = term {
                    if seen.insert(name.clone()) {
                        order.push(name.clone());
                    }
                }
            }
        }
    }
    order
}
