//! Classification of `trainable_rule` body literals for the neural joint-training
//! template path (Stage-A gates vs Stage-B joins).
//!
//! The neural template grounds a circuit from the neural (`nn/4`) predicates in a
//! trainable rule's body and reuses it across examples by swapping neural leaf
//! weights. Ordinary relations in the body are only admissible when they
//! introduce no new variable — a *gate* that can be evaluated per example and
//! injected as a fixed, non-differentiable circuit leaf. A relation that
//! introduces a fresh variable requires real relational grounding (a join), which
//! the single-cached-circuit template does not yet support.
//!
//! This module is pure over the AST: the caller supplies which predicates are
//! neural and which variables are already bound (the rule head variables plus the
//! variables appearing in neural-input positions).

use std::collections::BTreeSet;

use crate::ast::BodyLiteral;

/// Classification of a single `trainable_rule` body literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainableBodyClass {
    /// An `nn/4` neural predicate — handled as a neural group.
    Neural,
    /// A builtin (comparison / is-expression / univ) grounded by the circuit
    /// compiler itself.
    Builtin,
    /// A negated literal — unsupported by the neural template path.
    Negated,
    /// An epistemic literal — unsupported by the neural template path.
    Epistemic,
    /// Stage-A gate: a positive non-neural relation whose every named variable is
    /// already bound. It becomes a fixed, non-differentiable circuit leaf whose
    /// per-example weight is the boolean truth of the relation.
    Gate,
    /// Stage-B join: a positive non-neural relation that introduces a named
    /// variable not bound by the head or any neural input. Not yet supported by
    /// the template path.
    UnboundJoin {
        /// The first unbound named variable introduced by the literal.
        var: String,
    },
}

/// Classify a `trainable_rule` body literal for the neural template path.
///
/// `bound_vars` must contain every variable already bound before this literal is
/// considered — the rule head variables together with the variables appearing in
/// neural-predicate input positions. `is_neural` reports whether a predicate name
/// is an `nn/4` neural predicate.
///
/// A positive non-neural relation is a [`TrainableBodyClass::Gate`] iff every
/// *named* variable it mentions is in `bound_vars`. Anonymous wildcards (`_`)
/// bind nothing downstream and are permitted in a gate (existential projection).
/// A named variable that is not bound — even one used nowhere else — yields
/// [`TrainableBodyClass::UnboundJoin`]; rename it to `_` to request an explicit
/// existence check.
pub fn classify_trainable_body_literal(
    literal: &BodyLiteral,
    bound_vars: &BTreeSet<String>,
    is_neural: impl Fn(&str) -> bool,
) -> TrainableBodyClass {
    match literal {
        BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {
            TrainableBodyClass::Builtin
        }
        BodyLiteral::Negated(_) => TrainableBodyClass::Negated,
        BodyLiteral::Epistemic(_) => TrainableBodyClass::Epistemic,
        BodyLiteral::Positive(atom) => {
            if is_neural(&atom.predicate) {
                return TrainableBodyClass::Neural;
            }
            for var in atom.variables() {
                if !bound_vars.contains(var) {
                    return TrainableBodyClass::UnboundJoin {
                        var: var.to_string(),
                    };
                }
            }
            TrainableBodyClass::Gate
        }
    }
}
