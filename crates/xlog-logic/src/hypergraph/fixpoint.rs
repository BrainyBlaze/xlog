//! Naive fixpoint evaluator for recursive hypergraph rules.
//!
//! Builds on PR 2's [`super::evaluate_rule`]: each iteration runs
//! every supplied rule once against the union of supplied base
//! relations and the target predicate's currently-derived rows,
//! unions new rows into the derived set, sorts and deduplicates,
//! and stops when an iteration produces zero new rows.
//!
//! Pure-Rust, deterministic, set-semantics. Built to be the
//! recursive-WCOJ correctness oracle for PR 4+ mixed-execution
//! kernels. Not optimized — a real engine would use semi-naive
//! delta-driven evaluation; this slice prefers simplicity over
//! speed.
//!
//! ## Algorithm
//!
//! 1. Validate that every supplied rule's head predicate equals
//!    the target. (The slice ships single-predicate fixpoint;
//!    multi-predicate SCCs are a later concern.)
//! 2. Compute the target relation's schema by inferring per-vertex
//!    types from the *first* iteration's evaluation: each rule
//!    head's term shape gives an arity; every cell's [`RefValue`]
//!    variant gives a [`ScalarType`]. The first iteration that
//!    produces non-empty rows freezes the schema.
//! 3. Loop up to `max_iterations` times:
//!    a. For each rule, run [`super::evaluate_rule`] against
//!    `base_relations ∪ {target → derived}`. Union new rows
//!    into a per-iteration scratch buffer.
//!    b. Merge scratch into `derived`, sort+dedupe.
//!    c. If `derived` is unchanged this iteration, return.
//!    d. Otherwise increment the iteration counter and continue.
//! 4. If the loop exits without convergence, return
//!    [`FixpointError::MaxIterationsExceeded`].
//!
//! ## Schema seeding
//!
//! On iteration 1, rules referencing the target predicate would
//! ordinarily fail with [`RefEvalError::MissingRelation`]. The
//! evaluator pre-seeds an empty `target` relation whose schema is
//! taken from the head arity of the first rule with a non-empty
//! head. Once a real iteration produces tuples the schema is
//! frozen against per-cell variant types — drift in later
//! iterations surfaces as a
//! [`RefEvalError::RelationValueTypeMismatch`] from PR 2's
//! validation.

use super::{evaluate_rule, RefEvalError, RefRelation, RefRelationStore, RefValue, VariableOrder};
use crate::ast::Rule;
use xlog_core::ScalarType;

/// Configuration for [`evaluate_fixpoint`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixpointConfig {
    /// Hard cap on iteration count. Returns
    /// [`FixpointError::MaxIterationsExceeded`] if convergence
    /// is not reached within this many iterations. Must be ≥ 1.
    pub max_iterations: usize,
}

impl Default for FixpointConfig {
    /// Default cap is 32. Generous enough for typical
    /// transitive-closure and Same-Generation tests; tight enough
    /// that infinite-loop bugs surface fast.
    fn default() -> Self {
        Self { max_iterations: 32 }
    }
}

/// Errors surfaced by [`evaluate_fixpoint`].
#[derive(Debug, Clone, PartialEq)]
pub enum FixpointError {
    /// A rule's head predicate did not equal `target_predicate`.
    /// The fixpoint evaluator only accepts rules that contribute
    /// to the named target.
    RuleNotForTarget {
        /// Index of the offending rule in the input slice.
        rule_index: usize,
        /// Head predicate observed.
        observed: String,
        /// Expected target predicate.
        expected: String,
    },
    /// Two rules disagreed on the target predicate's head arity.
    /// Surfaced separately from the per-row arity check so the
    /// caller sees the rule-level shape mismatch directly rather
    /// than as a downstream `RelationRowArityMismatch`.
    HeadArityMismatch {
        /// Index of the offending rule in the input slice.
        rule_index: usize,
        /// Head arity observed on this rule.
        observed_arity: usize,
        /// Head arity established by the first non-empty-head rule.
        expected_arity: usize,
    },
    /// `target_predicate` was already present in `base_relations`.
    /// The fixpoint constructs the target relation; allowing
    /// `base_relations` to seed it would silently shadow the
    /// caller's seed on the first iteration. If you want a seed,
    /// encode it as a base-case rule.
    TargetPredicateInBaseRelations {
        /// The target predicate name as supplied.
        name: String,
    },
    /// A rule failed evaluation. Wraps the per-rule error from
    /// [`evaluate_rule`] with the rule's index in the input slice
    /// for diagnostic precision.
    RuleEval {
        /// Index in the input slice.
        rule_index: usize,
        /// The wrapped per-rule error.
        source: RefEvalError,
    },
    /// The fixpoint did not converge within
    /// [`FixpointConfig::max_iterations`].
    MaxIterationsExceeded {
        /// The configured cap.
        limit: usize,
        /// Size of the derived target relation at the cap.
        observed_size: usize,
    },
    /// No supplied rule produced a head whose arity could be
    /// inferred — caller supplied an empty rules slice or
    /// every rule had an empty head.
    TargetSchemaIndeterminable,
    /// `max_iterations` was zero. Must be ≥ 1.
    InvalidMaxIterations,
}

/// Evaluate a recursive set of rules to a fixpoint over a single
/// target predicate.
///
/// Every supplied rule must have its head predicate equal to
/// `target_predicate`. `base_relations` carries any non-target
/// predicates referenced in rule bodies (e.g. `edge` for transitive
/// closure, `parent` for Same Generation). The target predicate
/// must NOT appear in `base_relations`; it is constructed by the
/// fixpoint and shadowing would be ambiguous.
///
/// Returns the converged target relation. Set semantics: rows
/// are sorted lexicographically and deduplicated. Same input →
/// same output. Rule order in the input slice does not affect
/// the result (locked by test).
pub fn evaluate_fixpoint(
    rules: &[Rule],
    base_relations: &RefRelationStore,
    target_predicate: &str,
    order: &dyn VariableOrder,
    config: &FixpointConfig,
) -> Result<RefRelation, FixpointError> {
    if config.max_iterations == 0 {
        return Err(FixpointError::InvalidMaxIterations);
    }

    if base_relations.contains_key(target_predicate) {
        return Err(FixpointError::TargetPredicateInBaseRelations {
            name: target_predicate.to_string(),
        });
    }

    // 1. Validate that every rule heads `target_predicate`.
    for (i, rule) in rules.iter().enumerate() {
        if rule.head.predicate != target_predicate {
            return Err(FixpointError::RuleNotForTarget {
                rule_index: i,
                observed: rule.head.predicate.clone(),
                expected: target_predicate.to_string(),
            });
        }
    }

    // 2. Establish the target's arity from the first non-empty
    // rule head. Per-position scalar types are filled in by the
    // first iteration that produces non-empty rows.
    let target_arity = rules
        .iter()
        .find(|r| !r.head.terms.is_empty())
        .map(|r| r.head.terms.len())
        .ok_or(FixpointError::TargetSchemaIndeterminable)?;

    // Every other non-empty-head rule must agree on that arity.
    // Surfacing this as a rule-level error prevents downstream
    // confusion: PR 2's per-row validation would catch the same
    // problem on iteration 1 as `RelationRowArityMismatch`, but
    // pointing at "row index N" instead of "rule index M" is two
    // layers of indirection from the actual fixture problem.
    for (i, rule) in rules.iter().enumerate() {
        if rule.head.terms.is_empty() {
            continue;
        }
        if rule.head.terms.len() != target_arity {
            return Err(FixpointError::HeadArityMismatch {
                rule_index: i,
                observed_arity: rule.head.terms.len(),
                expected_arity: target_arity,
            });
        }
    }

    // Seed schema: every column is U32 as a placeholder. As soon
    // as the first iteration produces a non-empty row we replace
    // the schema with the variant-derived types and re-validate
    // future iterations against it.
    let mut derived_schema: Option<Vec<ScalarType>> = None;
    let mut derived_rows: Vec<Vec<RefValue>> = Vec::new();

    for _iter in 0..config.max_iterations {
        // 3a. Build the per-iteration relation store: base ∪ target.
        let mut store = base_relations.clone();
        let placeholder_schema = derived_schema
            .clone()
            .unwrap_or_else(|| vec![ScalarType::U32; target_arity]);
        store.insert(
            target_predicate.to_string(),
            RefRelation {
                schema: placeholder_schema,
                rows: derived_rows.clone(),
            },
        );

        // 3b. Run every rule and union new tuples.
        let mut new_rows: Vec<Vec<RefValue>> = derived_rows.clone();
        for (rule_index, rule) in rules.iter().enumerate() {
            // On iter 0, rules that reference `target_predicate`
            // see an empty target — that's fine; they contribute
            // base tuples. On later iters, recursive rules see
            // the derived set.
            let rows = evaluate_rule(rule, &store, order).map_err(|e| FixpointError::RuleEval {
                rule_index,
                source: e,
            })?;
            new_rows.extend(rows);
        }
        new_rows.sort();
        new_rows.dedup();

        // 3c. On the first iteration that produced rows, freeze
        // the schema from the cell variants of the first row.
        if derived_schema.is_none() {
            if let Some(first) = new_rows.first() {
                derived_schema = Some(infer_schema(first));
            }
        }

        // 3d. Convergence check: identical-to-previous → done.
        if new_rows == derived_rows {
            let schema = derived_schema.unwrap_or_else(|| vec![ScalarType::U32; target_arity]);
            return Ok(RefRelation {
                schema,
                rows: derived_rows,
            });
        }

        derived_rows = new_rows;
    }

    Err(FixpointError::MaxIterationsExceeded {
        limit: config.max_iterations,
        observed_size: derived_rows.len(),
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
