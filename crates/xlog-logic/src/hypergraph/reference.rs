//! CPU reference evaluator for hypergraph rules.
//!
//! Pure-Rust, deterministic, deduplicated. Built to be the WCOJ
//! oracle that PR 3+ GPU kernels are validated against, NOT to be
//! fast. Single-rule semantics only — recursive fixpoint is out of
//! scope for PR 2 (planned for the next slice).
//!
//! ## Algorithm
//!
//! For an [`Eligible`](super::Eligibility::Eligible) rule:
//!   1. Build the [`HypergraphRule`] from the AST [`Rule`].
//!   2. For each vertex, compute its **domain**: the union of values
//!      observed at that vertex's positions across all hyperedges'
//!      relations. Vertices appearing in zero hyperedges (impossible
//!      for an eligible rule, but checked) get an empty domain.
//!   3. Recurse over vertices in the order produced by the supplied
//!      [`VariableOrder`]. At each step, bind the next variable to
//!      one of its domain values.
//!   4. At full assignment, verify every positive hyperedge has at
//!      least one matching row — accounting for any constants and
//!      anonymous wildcards in the source atom.
//!   5. Apply [`BodyLiteral::Comparison`] filters.
//!   6. Project the head's terms into a [`Vec<RefValue>`] result row.
//!   7. Sort + dedupe the final row collection (stable, set semantics).
//!
//! Ineligible rules return an evaluator error — callers must gate on
//! [`super::analyze`] / [`super::analyze_typed`] before evaluation.
//! See [`evaluate_rule`] for the full contract.

use super::ir::{Hyperedge, HypergraphRule, VertexId};
use super::var_order::VariableOrder;
use super::{analyze, Eligibility, ExecutorContext};
use crate::ast::{Atom, BodyLiteral, CompOp, Comparison, Rule, Term};
use std::collections::BTreeMap;
use xlog_core::ScalarType;

/// A typed scalar value produced or consumed by the reference evaluator.
///
/// Mirrors [`xlog_core::ScalarType`] except for floating-point types,
/// which are deliberately omitted: PR 2's WCOJ supported set is
/// `{U32, U64, Symbol}` (see
/// [`super::WCOJ_SUPPORTED_KEY_TYPES`]); the wider non-float types
/// (`I32`, `I64`, `Bool`) are included so the evaluator can hold
/// non-key column values from rules where projections include them.
///
/// `RefValue::Symbol` carries an owned `String` rather than the AST's
/// interned `u32` — the evaluator is decoupled from
/// [`xlog_core::symbol`]'s global interner so tests stay hermetic.
/// Integration into a pipeline that uses interned symbols (PR 4+) is
/// expected to translate at the boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RefValue {
    /// Unsigned 32-bit value.
    U32(u32),
    /// Unsigned 64-bit value.
    U64(u64),
    /// Signed 32-bit value.
    I32(i32),
    /// Signed 64-bit value.
    I64(i64),
    /// Boolean.
    Bool(bool),
    /// String-encoded symbol (decoupled from the symbol interner —
    /// see type-level docs).
    Symbol(String),
}

/// A relation: schema (column types) + rows (lists of typed values).
///
/// Tests build these directly. PR 4+ may add construction from
/// `xlog-cuda` buffers or Arrow batches.
#[derive(Debug, Clone, PartialEq)]
pub struct RefRelation {
    /// One [`ScalarType`] per column.
    pub schema: Vec<ScalarType>,
    /// Rows; each inner vec has length `schema.len()`. Values may
    /// be of any [`RefValue`] variant in PR 2 (the evaluator does
    /// not enforce schema/value type agreement at construction —
    /// only at match time).
    pub rows: Vec<Vec<RefValue>>,
}

/// Map from predicate name to relation. Sorted iteration is
/// inherent to [`BTreeMap`] and contributes to evaluator
/// determinism.
pub type RefRelationStore = BTreeMap<String, RefRelation>;

/// Errors surfaced by [`evaluate_rule`].
#[derive(Debug, Clone, PartialEq)]
pub enum RefEvalError {
    /// The rule was [`Eligibility::Ineligible`] — callers must gate
    /// on [`super::analyze`] / [`super::analyze_typed`] before
    /// evaluation. Carries the boundary list for diagnostic use.
    Ineligible(Vec<super::Boundary>),
    /// A predicate referenced by the rule body was not present in
    /// the supplied [`RefRelationStore`].
    MissingRelation(String),
    /// A body atom's arity does not match the supplied relation's
    /// schema arity. The evaluator is the WCOJ correctness oracle —
    /// arity drift between rule and fixture must surface as a
    /// loud failure, not a silent skip.
    RelationArityMismatch {
        /// Predicate name being looked up.
        predicate: String,
        /// Number of terms in the body atom.
        atom_arity: usize,
        /// Number of columns in the relation schema.
        relation_arity: usize,
    },
    /// A relation row's length does not equal the schema's
    /// column count. Rejected before evaluation begins so a
    /// malformed fixture fails on entry rather than producing
    /// misleading partial-row matches.
    RelationRowArityMismatch {
        /// Predicate name owning the malformed row.
        predicate: String,
        /// Index of the offending row within the relation.
        row_index: usize,
        /// Observed length of the row.
        row_len: usize,
        /// Expected length per the schema.
        schema_len: usize,
    },
    /// A relation row carries a [`RefValue`] whose variant does not
    /// match the schema's [`ScalarType`] at that column. Detected
    /// before evaluation so type drift in fixtures fails loudly
    /// rather than silently corrupting GPU oracle comparisons.
    RelationValueTypeMismatch {
        /// Predicate name owning the malformed row.
        predicate: String,
        /// Row index within the relation.
        row_index: usize,
        /// Column index within the row.
        column: usize,
        /// Schema-declared type at that column.
        expected: ScalarType,
        /// String description of the actual value.
        got: String,
    },
    /// A constant in a body atom could not be coerced to a value
    /// compatible with the relation's column type at that position.
    ConstantTypeMismatch {
        /// Predicate name where the mismatch occurred.
        predicate: String,
        /// Argument position (0-based) within the predicate.
        position: usize,
        /// Expected scalar type per the relation's schema.
        expected: ScalarType,
        /// String description of the constant for diagnostic use.
        got: String,
    },
    /// A [`Comparison`] body literal could not be evaluated because
    /// its operands were of incompatible types.
    ComparisonTypeMismatch {
        /// Left-hand operand description.
        left: String,
        /// Right-hand operand description.
        right: String,
        /// Operator that was being applied.
        op: CompOp,
    },
    /// The rule head referenced a variable that is not bound by any
    /// body atom (a free variable in the head).
    UnboundHeadVariable(String),
    /// The rule body contained an [`crate::ast::IsExpr`] — PR 2
    /// rejects rules containing computed bindings via the
    /// `BodyIsExpr` boundary, so this is a defensive arm; if it
    /// fires the analyzer let an `IsExpr` rule through erroneously.
    IsExprNotSupported,
    /// A variable appears in two body atoms whose relation schemas
    /// disagree on the column type at the variable's position.
    ///
    /// Detected by [`super::evaluate_rule_typed`] (and the typed
    /// fixpoint variants) at the typed-gate phase, before
    /// evaluation. Without this check, an evaluator would either
    /// silently coerce values across types or fail later with a
    /// less precise error pointing at a row, not at the rule.
    ///
    /// The two `(predicate, position, type)` triples are recorded
    /// in *first-encountered* order: walking body atoms in source
    /// order, the first atom that types this variable wins
    /// `first_*`; the second atom whose type differs gets
    /// `second_*`. Subsequent agreeing atoms are silent.
    ConflictingVariableType {
        /// Variable name as it appears in the source rule.
        var: String,
        /// Predicate of the first atom that typed `var`.
        first_predicate: String,
        /// 0-based argument position within `first_predicate`.
        first_position: usize,
        /// Schema type at `(first_predicate, first_position)`.
        first_type: ScalarType,
        /// Predicate of the conflicting atom.
        second_predicate: String,
        /// 0-based argument position within `second_predicate`.
        second_position: usize,
        /// Schema type at `(second_predicate, second_position)`.
        second_type: ScalarType,
    },
    /// Two rules contributing to the same head predicate disagree
    /// on the type of the same column under the PR 8 SCC type
    /// inference pass.
    ///
    /// Distinct from [`Self::ConflictingVariableType`], which is a
    /// within-rule body conflict (one rule, two body atoms typing
    /// the same variable differently). This variant is a
    /// cross-rule head-column conflict, detected by
    /// [`super::infer_scc_predicate_schemas`] when back-propagating
    /// from rule heads to head-predicate column types.
    ///
    /// Surfaced through the typed evaluators via
    /// [`super::FixpointError::RuleEval`] /
    /// [`super::SccFixpointError::RuleEval`]; their `rule_index`
    /// field carries `second_rule_index` (the rule whose typing
    /// caused the conflict to be detected; the first rule's typing
    /// was already accepted by then).
    InferenceConflict {
        /// Head predicate name where the conflict was detected.
        predicate: String,
        /// 0-based column index where types disagree.
        column: usize,
        /// Rule index (within the predicate's group) that first
        /// typed the column.
        first_rule_index: usize,
        /// Type derived from the first rule's body.
        first_type: ScalarType,
        /// Rule index (within the predicate's group) that
        /// disagrees.
        second_rule_index: usize,
        /// Type derived from the conflicting rule's body.
        second_type: ScalarType,
    },
}

/// Evaluate `rule` against `relations` using `order` for variable
/// binding. Returns the result rows (one inner [`Vec<RefValue>`]
/// per row, with one element per term in the rule head) sorted
/// lexicographically and deduplicated.
///
/// ## Eligibility gating
///
/// The rule is structurally analyzed via [`super::analyze`] before
/// any evaluation work; if it is [`Eligibility::Ineligible`] the
/// function returns [`RefEvalError::Ineligible`] carrying the
/// boundary list. Callers wanting type-aware gating should run
/// [`super::analyze_typed`] in addition and translate the verdict
/// before calling this function — the evaluator itself does not
/// take a type map (the supplied [`RefRelationStore`] is the
/// authoritative type source for the values it operates on).
///
/// ## Determinism contract
///
/// Same `(rule, relations, order)` → same output. Output rows are
/// sorted (lexicographic over [`RefValue`]'s [`Ord`] impl) and
/// deduplicated. The variable order affects work done internally
/// but NOT the result set (locked by a test).
pub fn evaluate_rule(
    rule: &Rule,
    relations: &RefRelationStore,
    order: &dyn VariableOrder,
) -> Result<Vec<Vec<RefValue>>, RefEvalError> {
    let hg = HypergraphRule::from_rule(rule);
    match analyze(&hg, ExecutorContext::HashFallback) {
        Eligibility::Eligible => {}
        Eligibility::Ineligible(bs) => return Err(RefEvalError::Ineligible(bs)),
    }

    // Defensive: BodyIsExpr is one of the analyze() boundaries, so
    // an Eligible rule cannot contain an IsExpr. If one slipped
    // through, fail loudly.
    if rule
        .body
        .iter()
        .any(|l| matches!(l, BodyLiteral::IsExpr(_)))
    {
        return Err(RefEvalError::IsExprNotSupported);
    }

    let positive_atoms: Vec<&Atom> = rule
        .body
        .iter()
        .filter_map(|l| match l {
            BodyLiteral::Positive(a) => Some(a),
            _ => None,
        })
        .collect();

    let comparisons: Vec<&Comparison> = rule
        .body
        .iter()
        .filter_map(|l| match l {
            BodyLiteral::Comparison(c) => Some(c),
            _ => None,
        })
        .collect();

    // Validate that every referenced relation exists and rebuild
    // each atom's column-aware metadata once for downstream use.
    // The evaluator is the WCOJ correctness oracle — fixture
    // problems (missing relations, malformed rows, schema/value
    // type drift) must surface here rather than silently corrupt
    // downstream kernel comparisons.
    let mut atom_specs: Vec<AtomSpec> = Vec::with_capacity(positive_atoms.len());
    for (atom, edge) in positive_atoms.iter().zip(hg.hyperedges.iter()) {
        let rel = relations
            .get(&atom.predicate)
            .ok_or_else(|| RefEvalError::MissingRelation(atom.predicate.clone()))?;
        validate_relation_rows(&atom.predicate, rel)?;
        let spec = AtomSpec::build(atom, edge, rel)?;
        atom_specs.push(spec);
    }

    // Build per-vertex domain. A vertex's domain is the set of
    // values it can possibly take, computed as the union over all
    // its occurrences in the hyperedges. PR 2 takes the trivial
    // simple-union (no intersection refinement); the recursive
    // search below filters down to consistent assignments.
    let domains: Vec<Vec<RefValue>> = build_domains(&hg, &atom_specs);

    // Resolve the variable order into a [VertexId] sequence.
    let var_order: Vec<VertexId> = order.order(&hg);
    debug_assert_eq!(var_order.len(), hg.vertex_count());

    // Recurse and project.
    let mut binding: BTreeMap<VertexId, RefValue> = BTreeMap::new();
    let mut results: Vec<Vec<RefValue>> = Vec::new();
    enumerate(
        &hg,
        &atom_specs,
        &comparisons,
        &domains,
        &var_order,
        0,
        &mut binding,
        &rule.head,
        &mut results,
    )?;

    results.sort();
    results.dedup();
    Ok(results)
}

/// Per-atom metadata derived once at evaluator entry: which arg
/// positions match against constants vs vertices vs anonymous
/// wildcards, plus the atom's relation reference.
struct AtomSpec<'a> {
    relation: &'a RefRelation,
    /// Position descriptors, one per atom argument.
    positions: Vec<PositionSpec>,
}

enum PositionSpec {
    /// Atom argument is a variable bound to this vertex.
    Vertex(VertexId),
    /// Atom argument is a constant — already coerced against the
    /// relation's column type at this position.
    Constant(RefValue),
    /// Atom argument is `Term::Anonymous` — matches anything.
    Wildcard,
}

impl<'a> AtomSpec<'a> {
    fn build(
        atom: &Atom,
        edge: &Hyperedge,
        relation: &'a RefRelation,
    ) -> Result<Self, RefEvalError> {
        if atom.terms.len() != relation.schema.len() {
            return Err(RefEvalError::RelationArityMismatch {
                predicate: atom.predicate.clone(),
                atom_arity: atom.terms.len(),
                relation_arity: relation.schema.len(),
            });
        }
        let mut positions = Vec::with_capacity(atom.terms.len());
        for (i, term) in atom.terms.iter().enumerate() {
            let p = match term {
                Term::Variable(_) => {
                    // Vertex assignment matches whatever the IR
                    // built for this position.
                    match edge.vertex_positions.get(i).and_then(|p| *p) {
                        Some(vid) => PositionSpec::Vertex(vid),
                        None => {
                            // IR builder didn't tag a vertex here —
                            // shouldn't happen for `Term::Variable`,
                            // but treat as wildcard defensively.
                            PositionSpec::Wildcard
                        }
                    }
                }
                Term::Anonymous => PositionSpec::Wildcard,
                Term::Integer(n) => {
                    let coerced =
                        coerce_integer_constant(*n, relation.schema[i], &atom.predicate, i)?;
                    PositionSpec::Constant(coerced)
                }
                Term::String(s) => match relation.schema[i] {
                    ScalarType::Symbol => PositionSpec::Constant(RefValue::Symbol(s.clone())),
                    other => {
                        return Err(RefEvalError::ConstantTypeMismatch {
                            predicate: atom.predicate.clone(),
                            position: i,
                            expected: other,
                            got: format!("string {s:?}"),
                        });
                    }
                },
                Term::Symbol(_id) => {
                    // Interned symbols are not used by PR 2 tests
                    // (they construct via `Term::String` against a
                    // Symbol column). Reject explicitly so a future
                    // user sees a clear failure.
                    return Err(RefEvalError::ConstantTypeMismatch {
                        predicate: atom.predicate.clone(),
                        position: i,
                        expected: relation.schema[i],
                        got: "interned Symbol(u32) — use Term::String against a Symbol column"
                            .to_string(),
                    });
                }
                Term::Float(f) => {
                    return Err(RefEvalError::ConstantTypeMismatch {
                        predicate: atom.predicate.clone(),
                        position: i,
                        expected: relation.schema[i],
                        got: format!("float {f}"),
                    });
                }
                Term::Aggregate(_) => {
                    // Aggregates appear only in heads; eligibility
                    // analyzer rejects head-aggregation rules. If we
                    // see one in a body atom, it's a parser-level
                    // surprise.
                    return Err(RefEvalError::ConstantTypeMismatch {
                        predicate: atom.predicate.clone(),
                        position: i,
                        expected: relation.schema[i],
                        got: "aggregate in body atom".to_string(),
                    });
                }
            };
            positions.push(p);
        }
        Ok(AtomSpec {
            relation,
            positions,
        })
    }
}

fn coerce_integer_constant(
    n: i64,
    target: ScalarType,
    predicate: &str,
    position: usize,
) -> Result<RefValue, RefEvalError> {
    let mismatch = |got: String| RefEvalError::ConstantTypeMismatch {
        predicate: predicate.to_string(),
        position,
        expected: target,
        got,
    };
    match target {
        ScalarType::U32 => {
            if n < 0 || n > u32::MAX as i64 {
                Err(mismatch(format!("integer {n} out of range for U32")))
            } else {
                Ok(RefValue::U32(n as u32))
            }
        }
        ScalarType::U64 => {
            if n < 0 {
                Err(mismatch(format!("negative integer {n} for U64")))
            } else {
                Ok(RefValue::U64(n as u64))
            }
        }
        ScalarType::I32 => {
            if !(i32::MIN as i64..=i32::MAX as i64).contains(&n) {
                Err(mismatch(format!("integer {n} out of range for I32")))
            } else {
                Ok(RefValue::I32(n as i32))
            }
        }
        ScalarType::I64 => Ok(RefValue::I64(n)),
        ScalarType::Bool | ScalarType::F32 | ScalarType::F64 | ScalarType::Symbol => {
            Err(mismatch(format!("integer {n} for {target:?}")))
        }
    }
}

/// Validate that every row in `relation` has length equal to its
/// schema's column count and that every cell's [`RefValue`] variant
/// matches the schema's [`ScalarType`] at that column.
///
/// Returns the first encountered mismatch as a structured error.
/// Called once per (predicate, atom) pair before evaluation; the
/// per-atom redundancy is cheap and means the failure mentions the
/// predicate the rule actually referenced.
fn validate_relation_rows(predicate: &str, relation: &RefRelation) -> Result<(), RefEvalError> {
    let schema_len = relation.schema.len();
    for (row_index, row) in relation.rows.iter().enumerate() {
        if row.len() != schema_len {
            return Err(RefEvalError::RelationRowArityMismatch {
                predicate: predicate.to_string(),
                row_index,
                row_len: row.len(),
                schema_len,
            });
        }
        for (column, value) in row.iter().enumerate() {
            let expected = relation.schema[column];
            if !ref_value_matches_scalar_type(value, expected) {
                return Err(RefEvalError::RelationValueTypeMismatch {
                    predicate: predicate.to_string(),
                    row_index,
                    column,
                    expected,
                    got: format!("{value:?}"),
                });
            }
        }
    }
    Ok(())
}

/// True when `value`'s variant lines up with `expected`'s
/// [`ScalarType`]. The `RefValue::Symbol` variant matches the
/// `Symbol` schema type; integer / boolean variants match their
/// same-name `ScalarType`s. `F32` / `F64` schemas have no
/// matching `RefValue` variant (PR 2 deliberately excludes
/// floating-point values from the evaluator), so any value
/// against an `F*` schema column is a type mismatch.
fn ref_value_matches_scalar_type(value: &RefValue, expected: ScalarType) -> bool {
    matches!(
        (value, expected),
        (RefValue::U32(_), ScalarType::U32)
            | (RefValue::U64(_), ScalarType::U64)
            | (RefValue::I32(_), ScalarType::I32)
            | (RefValue::I64(_), ScalarType::I64)
            | (RefValue::Bool(_), ScalarType::Bool)
            | (RefValue::Symbol(_), ScalarType::Symbol)
    )
}

fn build_domains(hg: &HypergraphRule, atom_specs: &[AtomSpec<'_>]) -> Vec<Vec<RefValue>> {
    let mut domains: Vec<Vec<RefValue>> = vec![Vec::new(); hg.vertex_count()];
    for spec in atom_specs {
        for (i, pos) in spec.positions.iter().enumerate() {
            if let PositionSpec::Vertex(vid) = pos {
                let VertexId(idx) = *vid;
                for row in &spec.relation.rows {
                    // Row length is guaranteed equal to schema_len
                    // by `validate_relation_rows` upstream — direct
                    // index is safe.
                    domains[idx].push(row[i].clone());
                }
            }
        }
    }
    for d in domains.iter_mut() {
        d.sort();
        d.dedup();
    }
    domains
}

#[allow(clippy::too_many_arguments)]
fn enumerate(
    hg: &HypergraphRule,
    atom_specs: &[AtomSpec<'_>],
    comparisons: &[&Comparison],
    domains: &[Vec<RefValue>],
    var_order: &[VertexId],
    depth: usize,
    binding: &mut BTreeMap<VertexId, RefValue>,
    head: &Atom,
    results: &mut Vec<Vec<RefValue>>,
) -> Result<(), RefEvalError> {
    if depth == var_order.len() {
        // Full assignment. Verify every positive hyperedge has at
        // least one matching row given the binding (and any
        // constants / wildcards from the source atom).
        for spec in atom_specs {
            if !atom_has_matching_row(spec, binding) {
                return Ok(());
            }
        }
        // Apply comparisons.
        for cmp in comparisons {
            if !evaluate_comparison(cmp, binding, hg)? {
                return Ok(());
            }
        }
        // Project head terms.
        let mut row = Vec::with_capacity(head.terms.len());
        for term in &head.terms {
            match term {
                Term::Variable(name) => {
                    let vid = match hg.vertices.iter().position(|v| &v.name == name) {
                        Some(i) => VertexId(i),
                        None => return Err(RefEvalError::UnboundHeadVariable(name.clone())),
                    };
                    let v = binding
                        .get(&vid)
                        .ok_or_else(|| RefEvalError::UnboundHeadVariable(name.clone()))?
                        .clone();
                    row.push(v);
                }
                Term::Integer(n) => row.push(RefValue::I64(*n)),
                Term::String(s) => row.push(RefValue::Symbol(s.clone())),
                Term::Anonymous | Term::Symbol(_) | Term::Float(_) | Term::Aggregate(_) => {
                    // Heads with anonymous wildcards / interned
                    // symbols / floats / aggregates are out of scope
                    // for PR 2 — eligibility analyzer covers
                    // aggregates; for the others a future PR widens
                    // support.
                    return Err(RefEvalError::UnboundHeadVariable(format!(
                        "unsupported head term: {term:?}"
                    )));
                }
            }
        }
        results.push(row);
        return Ok(());
    }
    let next_vid = var_order[depth];
    let VertexId(idx) = next_vid;
    for value in &domains[idx] {
        binding.insert(next_vid, value.clone());
        enumerate(
            hg,
            atom_specs,
            comparisons,
            domains,
            var_order,
            depth + 1,
            binding,
            head,
            results,
        )?;
        binding.remove(&next_vid);
    }
    Ok(())
}

fn atom_has_matching_row(spec: &AtomSpec<'_>, binding: &BTreeMap<VertexId, RefValue>) -> bool {
    'rows: for row in &spec.relation.rows {
        // Row length is guaranteed equal to spec.positions.len() by
        // `validate_relation_rows` upstream — direct indexing is safe.
        for (i, pos) in spec.positions.iter().enumerate() {
            let row_v = &row[i];
            match pos {
                PositionSpec::Vertex(vid) => {
                    let bound = match binding.get(vid) {
                        Some(b) => b,
                        None => return false, // unbound at full assignment is a bug
                    };
                    if bound != row_v {
                        continue 'rows;
                    }
                }
                PositionSpec::Constant(c) => {
                    if c != row_v {
                        continue 'rows;
                    }
                }
                PositionSpec::Wildcard => {
                    // Any value matches — no constraint.
                }
            }
        }
        return true;
    }
    false
}

fn evaluate_comparison(
    cmp: &Comparison,
    binding: &BTreeMap<VertexId, RefValue>,
    hg: &HypergraphRule,
) -> Result<bool, RefEvalError> {
    let lhs = resolve_term_for_comparison(&cmp.left, binding, hg)?;
    let rhs = resolve_term_for_comparison(&cmp.right, binding, hg)?;
    apply_comparison_op(&lhs, &rhs, cmp.op)
}

fn resolve_term_for_comparison(
    term: &Term,
    binding: &BTreeMap<VertexId, RefValue>,
    hg: &HypergraphRule,
) -> Result<RefValue, RefEvalError> {
    match term {
        Term::Variable(name) => {
            let vid = hg
                .vertices
                .iter()
                .position(|v| &v.name == name)
                .map(VertexId)
                .ok_or_else(|| RefEvalError::UnboundHeadVariable(name.clone()))?;
            binding
                .get(&vid)
                .cloned()
                .ok_or_else(|| RefEvalError::UnboundHeadVariable(name.clone()))
        }
        Term::Integer(n) => Ok(RefValue::I64(*n)),
        Term::String(s) => Ok(RefValue::Symbol(s.clone())),
        Term::Anonymous | Term::Symbol(_) | Term::Float(_) | Term::Aggregate(_) => {
            Err(RefEvalError::ComparisonTypeMismatch {
                left: format!("{term:?}"),
                right: "<n/a>".to_string(),
                op: CompOp::Eq,
            })
        }
    }
}

fn apply_comparison_op(lhs: &RefValue, rhs: &RefValue, op: CompOp) -> Result<bool, RefEvalError> {
    // Numeric promotion to i128 for cross-type comparisons among
    // the integer kinds. Same-kind comparisons fall through directly.
    let to_i128 = |v: &RefValue| -> Option<i128> {
        match v {
            RefValue::U32(n) => Some(*n as i128),
            RefValue::U64(n) => Some(*n as i128),
            RefValue::I32(n) => Some(*n as i128),
            RefValue::I64(n) => Some(*n as i128),
            _ => None,
        }
    };
    let cmp_result = match (lhs, rhs) {
        (RefValue::Symbol(a), RefValue::Symbol(b)) => match op {
            CompOp::Eq => Some(a == b),
            CompOp::Ne => Some(a != b),
            CompOp::Lt => Some(a < b),
            CompOp::Le => Some(a <= b),
            CompOp::Gt => Some(a > b),
            CompOp::Ge => Some(a >= b),
        },
        (RefValue::Bool(a), RefValue::Bool(b)) => match op {
            CompOp::Eq => Some(a == b),
            CompOp::Ne => Some(a != b),
            CompOp::Lt => Some(!a & b),
            CompOp::Le => Some(*a <= *b),
            CompOp::Gt => Some(*a & !b),
            CompOp::Ge => Some(*a >= *b),
        },
        _ => match (to_i128(lhs), to_i128(rhs)) {
            (Some(l), Some(r)) => Some(match op {
                CompOp::Eq => l == r,
                CompOp::Ne => l != r,
                CompOp::Lt => l < r,
                CompOp::Le => l <= r,
                CompOp::Gt => l > r,
                CompOp::Ge => l >= r,
            }),
            _ => None,
        },
    };
    cmp_result.ok_or_else(|| RefEvalError::ComparisonTypeMismatch {
        left: format!("{lhs:?}"),
        right: format!("{rhs:?}"),
        op,
    })
}
