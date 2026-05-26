//! Multi-predicate SCC fixpoint evaluator.
//!
//! Extends the single-target [`super::evaluate_fixpoint`] to a
//! mutually-recursive SCC: rules grouped by their target predicate,
//! evaluated jointly so each predicate's body can reference any
//! other predicate in the SCC. Naive evaluation: one iteration runs
//! every rule for every predicate; convergence is the global
//! "no relation grew this iteration" check.
//!
//! Pure-Rust, deterministic, set-semantics. Built to be the
//! recursive-multi-predicate WCOJ correctness oracle for PR 5+
//! mixed-execution kernels. Not optimized — semi-naive
//! delta-driven SCC fixpoint is a separate concern.
//!
//! ## Determinism
//!
//! * Rules grouped under each predicate are evaluated in input
//!   order (slice order). Rule order does NOT affect the result
//!   (locked by test) — it's an efficiency knob, not a semantic
//!   one.
//! * Predicates are iterated in [`BTreeMap`]'s sorted-by-key
//!   order. Predicate order does NOT affect the result (locked
//!   by test).
//! * Output rows for each predicate are sorted lexicographically
//!   and deduplicated.
//!
//! ## Schema management
//!
//! Per-predicate schemas are frozen from the first iteration
//! that produces non-empty rows for that predicate. Subsequent
//! iterations validate every newly-produced row's variants against
//! the frozen schema; mismatches surface as
//! [`SccFixpointError::InconsistentHeadValueTypes`] before the
//! row is unioned. Row arity drift surfaces as
//! [`SccFixpointError::HeadArityMismatch`] at function entry —
//! checked once per predicate group across its rules.

use super::{
    evaluate_rule, FixpointConfig, RefEvalError, RefRelation, RefRelationStore, RefValue,
    VariableOrder,
};
use crate::ast::Rule;
use std::collections::BTreeMap;
use xlog_core::ScalarType;

/// Errors surfaced by [`evaluate_scc_fixpoint`].
#[derive(Debug, Clone, PartialEq)]
pub enum SccFixpointError {
    /// A rule grouped under predicate `key` heads a different
    /// predicate. The grouping invariant — every rule's head
    /// predicate equals its `BTreeMap` key — is checked at
    /// function entry.
    RuleHeadPredicateMismatch {
        /// `BTreeMap` key under which the rule was grouped.
        group_key: String,
        /// Index of the rule within that group.
        rule_index: usize,
        /// Head predicate observed on the rule.
        observed: String,
    },
    /// Two rules grouped under the same predicate disagree on
    /// head arity.
    HeadArityMismatch {
        /// Predicate name.
        predicate: String,
        /// Index of the offending rule within its group.
        rule_index: usize,
        /// Head arity observed on this rule.
        observed_arity: usize,
        /// Head arity established by the first non-empty-head
        /// rule in the same group.
        expected_arity: usize,
    },
    /// A predicate's rules produced rows whose [`RefValue`]
    /// variants disagree across iterations or across rules
    /// within an iteration. Detected by validating each newly
    /// produced row's variant tuple against the predicate's
    /// frozen schema before unioning.
    InconsistentHeadValueTypes {
        /// Predicate name.
        predicate: String,
        /// Column index where the mismatch was first observed.
        column: usize,
        /// Schema-frozen scalar type at that column.
        expected: ScalarType,
        /// String description of the offending value.
        got: String,
    },
    /// `target_predicate` was already present in `base_relations`.
    /// SCC predicates are constructed by the fixpoint; allowing
    /// `base_relations` to seed any of them would silently shadow
    /// the caller's seed.
    PredicateInBaseRelations {
        /// The SCC predicate name as supplied.
        name: String,
    },
    /// A rule failed evaluation. Wraps the per-rule error with
    /// (predicate, rule_index) so the caller can pinpoint which
    /// rule of which group failed.
    RuleEval {
        /// Predicate group of the offending rule.
        predicate: String,
        /// Index within that group.
        rule_index: usize,
        /// The wrapped per-rule error.
        source: RefEvalError,
    },
    /// The SCC fixpoint did not converge within
    /// [`FixpointConfig::max_iterations`].
    MaxIterationsExceeded {
        /// The configured cap.
        limit: usize,
        /// Number of predicates in the SCC.
        predicate_count: usize,
        /// Total derived rows summed across all SCC predicates
        /// at the cap.
        total_observed_rows: usize,
    },
    /// At least one predicate had no rules with a non-empty head,
    /// so its arity could not be inferred.
    SchemaIndeterminable {
        /// Predicate name whose arity could not be inferred.
        predicate: String,
    },
    /// `max_iterations` was zero. Must be ≥ 1.
    InvalidMaxIterations,
}

/// Evaluate a mutually-recursive SCC of predicates to a fixpoint.
///
/// `rules` maps each predicate name to the list of rules deriving
/// it. Every rule's head predicate must equal its group key
/// (validated at entry). `base_relations` carries non-SCC
/// predicates referenced in rule bodies (e.g. EDB facts). The
/// SCC predicates must NOT appear in `base_relations`.
///
/// Returns a [`RefRelationStore`] whose keys are exactly the
/// keys of `rules`, each mapped to the converged relation. Set
/// semantics: rows sorted lexicographically, deduplicated.
#[allow(clippy::result_large_err)]
pub fn evaluate_scc_fixpoint(
    rules: &BTreeMap<String, Vec<Rule>>,
    base_relations: &RefRelationStore,
    order: &dyn VariableOrder,
    config: &FixpointConfig,
) -> Result<RefRelationStore, SccFixpointError> {
    if config.max_iterations == 0 {
        return Err(SccFixpointError::InvalidMaxIterations);
    }

    // Entry validation: per-predicate group invariants.
    let mut arities: BTreeMap<String, usize> = BTreeMap::new();
    for (predicate, group) in rules.iter() {
        if base_relations.contains_key(predicate) {
            return Err(SccFixpointError::PredicateInBaseRelations {
                name: predicate.clone(),
            });
        }
        for (idx, rule) in group.iter().enumerate() {
            if rule.head.predicate != *predicate {
                return Err(SccFixpointError::RuleHeadPredicateMismatch {
                    group_key: predicate.clone(),
                    rule_index: idx,
                    observed: rule.head.predicate.clone(),
                });
            }
        }
        // Establish arity from the first non-empty-head rule.
        let arity = group
            .iter()
            .find(|r| !r.head.terms.is_empty())
            .map(|r| r.head.terms.len())
            .ok_or_else(|| SccFixpointError::SchemaIndeterminable {
                predicate: predicate.clone(),
            })?;
        for (idx, rule) in group.iter().enumerate() {
            if rule.head.terms.is_empty() {
                continue;
            }
            if rule.head.terms.len() != arity {
                return Err(SccFixpointError::HeadArityMismatch {
                    predicate: predicate.clone(),
                    rule_index: idx,
                    observed_arity: rule.head.terms.len(),
                    expected_arity: arity,
                });
            }
        }
        arities.insert(predicate.clone(), arity);
    }

    // Per-predicate frozen schema and derived rows. Schemas seed
    // as `[U32; arity]` (matches PR 3); first non-empty iter
    // freezes from row variants.
    let mut frozen_schemas: BTreeMap<String, Option<Vec<ScalarType>>> = BTreeMap::new();
    let mut derived: BTreeMap<String, Vec<Vec<RefValue>>> = BTreeMap::new();
    for predicate in rules.keys() {
        frozen_schemas.insert(predicate.clone(), None);
        derived.insert(predicate.clone(), Vec::new());
    }

    for _iter in 0..config.max_iterations {
        // Build the per-iter store: base ∪ {predicate → derived}.
        let mut store = base_relations.clone();
        for (predicate, rows) in derived.iter() {
            let schema = frozen_schemas
                .get(predicate)
                .and_then(|s| s.clone())
                .unwrap_or_else(|| vec![ScalarType::U32; arities[predicate]]);
            store.insert(
                predicate.clone(),
                RefRelation {
                    schema,
                    rows: rows.clone(),
                },
            );
        }

        // For every predicate in sorted-key order, run every rule
        // and union new tuples into a per-predicate scratch buffer.
        let mut next: BTreeMap<String, Vec<Vec<RefValue>>> = derived.clone();
        for (predicate, group) in rules.iter() {
            let mut produced: Vec<Vec<RefValue>> = Vec::new();
            for (rule_index, rule) in group.iter().enumerate() {
                let rows =
                    evaluate_rule(rule, &store, order).map_err(|e| SccFixpointError::RuleEval {
                        predicate: predicate.clone(),
                        rule_index,
                        source: e,
                    })?;
                produced.extend(rows);
            }
            // Freeze the schema from the first iteration that
            // produces non-empty rows for THIS predicate; once
            // frozen, validate every newly-produced row's
            // variant tuple.
            let frozen_entry = frozen_schemas.get_mut(predicate).expect("inserted above");
            if frozen_entry.is_none() {
                if let Some(first) = produced.first() {
                    *frozen_entry = Some(infer_schema(first));
                }
            }
            if let Some(schema) = frozen_entry.as_ref() {
                for row in &produced {
                    if let Some((column, expected, got)) = first_type_mismatch(row, schema) {
                        return Err(SccFixpointError::InconsistentHeadValueTypes {
                            predicate: predicate.clone(),
                            column,
                            expected,
                            got,
                        });
                    }
                }
            }
            let target = next.get_mut(predicate).expect("predicate present");
            target.extend(produced);
            target.sort();
            target.dedup();
        }

        if next == derived {
            // Converged. Build the final RefRelationStore.
            let mut out: RefRelationStore = BTreeMap::new();
            for (predicate, rows) in derived.into_iter() {
                let schema = frozen_schemas
                    .get(&predicate)
                    .and_then(|s| s.clone())
                    .unwrap_or_else(|| vec![ScalarType::U32; arities[&predicate]]);
                out.insert(predicate, RefRelation { schema, rows });
            }
            return Ok(out);
        }
        derived = next;
    }

    let total: usize = derived.values().map(|v| v.len()).sum();
    Err(SccFixpointError::MaxIterationsExceeded {
        limit: config.max_iterations,
        predicate_count: rules.len(),
        total_observed_rows: total,
    })
}

fn infer_schema(row: &[RefValue]) -> Vec<ScalarType> {
    row.iter()
        .map(|v| match v {
            RefValue::U32(_) => ScalarType::U32,
            RefValue::U64(_) => ScalarType::U64,
            RefValue::I32(_) => ScalarType::I32,
            RefValue::I64(_) => ScalarType::I64,
            RefValue::Bool(_) => ScalarType::Bool,
            RefValue::Symbol(_) => ScalarType::Symbol,
        })
        .collect()
}

/// Return the first column where `row`'s [`RefValue`] variant does
/// not match `schema`'s [`ScalarType`]. Mirrors PR 2's row-level
/// `ref_value_matches_scalar_type`, applied at row produced (not
/// row stored) time so caller-facing errors point at rule output,
/// not at downstream relation validation.
///
/// Row arity is enforced upstream by [`SccFixpointError::HeadArityMismatch`]
/// at function entry; if a row of mismatched length somehow escapes
/// that check, it surfaces downstream as
/// [`RefEvalError::RelationRowArityMismatch`] from PR 2's validation
/// — which is honest about what it found. We do NOT add a synthetic
/// arity-mismatch arm here with a placeholder `ScalarType` (that
/// pattern was the silent-skip bug class PR 2's validator made us
/// fix in `RefEvalError::ConstantTypeMismatch`).
fn first_type_mismatch(
    row: &[RefValue],
    schema: &[ScalarType],
) -> Option<(usize, ScalarType, String)> {
    for (i, (val, ty)) in row.iter().zip(schema.iter()).enumerate() {
        let ok = matches!(
            (val, ty),
            (RefValue::U32(_), ScalarType::U32)
                | (RefValue::U64(_), ScalarType::U64)
                | (RefValue::I32(_), ScalarType::I32)
                | (RefValue::I64(_), ScalarType::I64)
                | (RefValue::Bool(_), ScalarType::Bool)
                | (RefValue::Symbol(_), ScalarType::Symbol)
        );
        if !ok {
            return Some((i, *ty, format!("{val:?}")));
        }
    }
    None
}
