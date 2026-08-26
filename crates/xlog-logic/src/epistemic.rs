//! Epistemic validation, reduction, and executable planning.

use std::collections::{BTreeMap, BTreeSet};

use xlog_core::{Result, XlogError};
use xlog_ir::{
    EirBodyLiteral, EirEpistemicLiteral, EirEpistemicMode, EirEpistemicOp, EirProgram, EirTerm,
    EpistemicConstraintPlan, EpistemicExecutablePlan, EpistemicGpuPlan, EpistemicReductionPlan,
    EpistemicSolverAssumptionBinding, EpistemicSolverServiceContract,
    EpistemicTupleMembershipBinding, EpistemicWcojReductionStatus,
};
use xlog_stats::StatsSnapshot;

use crate::ast::{
    Atom, BodyLiteral, CompOp, Comparison, Constraint, EpistemicLiteral, EpistemicMode,
    EpistemicOp, Program, Term,
};
use crate::build_eir;
use crate::compile::Compiler;
use crate::eir::convert_term;
use crate::lower::{positive_body_bound_variables, Lowerer};

/// Boolean truth value for bounded epistemic fixture evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruthValue {
    /// The literal is true.
    True,
    /// The literal is false.
    False,
}

impl TruthValue {
    fn from_bool(value: bool) -> Self {
        if value {
            TruthValue::True
        } else {
            TruthValue::False
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum EpistemicTermKey {
    Integer(i64),
    FloatBits(u64),
    String(String),
    Symbol(u32),
    List(Vec<EpistemicTermKey>),
    Cons {
        head: Box<EpistemicTermKey>,
        tail: Box<EpistemicTermKey>,
    },
    Compound {
        functor: String,
        args: Vec<EpistemicTermKey>,
    },
    PredRef(String),
}

impl EpistemicTermKey {
    fn from_term(term: &Term) -> Result<Self> {
        Ok(match term {
            Term::Integer(value) => Self::Integer(*value),
            Term::Float(value) => Self::FloatBits(value.to_bits()),
            Term::String(value) => Self::String(value.clone()),
            Term::Symbol(value) => Self::Symbol(*value),
            Term::List(items) => Self::List(
                items
                    .iter()
                    .map(Self::from_term)
                    .collect::<Result<Vec<_>>>()?,
            ),
            Term::Cons { head, tail } => Self::Cons {
                head: Box::new(Self::from_term(head)?),
                tail: Box::new(Self::from_term(tail)?),
            },
            Term::Compound { functor, args } => Self::Compound {
                functor: functor.clone(),
                args: args
                    .iter()
                    .map(Self::from_term)
                    .collect::<Result<Vec<_>>>()?,
            },
            Term::PredRef(value) => Self::PredRef(value.clone()),
            Term::Variable(_) | Term::Anonymous | Term::Aggregate(_) => {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic tuple key".to_string(),
                    context: "tuple-key epistemic facts require ground terms".to_string(),
                });
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum EpistemicAtomKey {
    Arity {
        predicate: String,
        arity: usize,
    },
    Ground {
        predicate: String,
        terms: Vec<EpistemicTermKey>,
    },
}

impl EpistemicAtomKey {
    fn from_arity(predicate: impl Into<String>, arity: usize) -> Self {
        Self::Arity {
            predicate: predicate.into(),
            arity,
        }
    }

    fn from_terms(predicate: impl Into<String>, terms: &[Term]) -> Result<Self> {
        Ok(Self::Ground {
            predicate: predicate.into(),
            terms: terms
                .iter()
                .map(EpistemicTermKey::from_term)
                .collect::<Result<Vec<_>>>()?,
        })
    }

    fn predicate(&self) -> &str {
        match self {
            Self::Arity { predicate, .. } | Self::Ground { predicate, .. } => predicate,
        }
    }

    fn arity(&self) -> usize {
        match self {
            Self::Arity { arity, .. } => *arity,
            Self::Ground { terms, .. } => terms.len(),
        }
    }

    fn matches_atom(&self, atom: &Atom) -> bool {
        if self.predicate() != atom.predicate || self.arity() != atom.arity() {
            return false;
        }
        match self {
            Self::Arity { .. } => true,
            Self::Ground { terms, .. } => atom
                .terms
                .iter()
                .map(EpistemicTermKey::from_term)
                .collect::<Result<Vec<_>>>()
                .is_ok_and(|atom_terms| atom_terms == *terms),
        }
    }

    fn overlaps(&self, other: &Self) -> bool {
        if self.predicate() != other.predicate() || self.arity() != other.arity() {
            return false;
        }
        matches!(self, Self::Arity { .. }) || matches!(other, Self::Arity { .. }) || self == other
    }
}

/// Minimal interpretation used by G91/FAEEL distinction fixtures.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EpistemicInterpretation {
    known: BTreeSet<EpistemicAtomKey>,
    possible: BTreeSet<EpistemicAtomKey>,
    rejected: BTreeSet<EpistemicAtomKey>,
}

impl EpistemicInterpretation {
    /// Create an empty interpretation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a predicate/arity pair as known.
    pub fn with_known(mut self, predicate: impl Into<String>, arity: usize) -> Self {
        self.known
            .insert(EpistemicAtomKey::from_arity(predicate, arity));
        self
    }

    /// Mark a concrete tuple key as known.
    pub fn with_known_terms(
        mut self,
        predicate: impl Into<String>,
        terms: Vec<Term>,
    ) -> Result<Self> {
        self.known
            .insert(EpistemicAtomKey::from_terms(predicate, &terms)?);
        Ok(self)
    }

    /// Mark a predicate/arity pair as possible under G91 compatibility semantics.
    pub fn with_possible(mut self, predicate: impl Into<String>, arity: usize) -> Self {
        self.possible
            .insert(EpistemicAtomKey::from_arity(predicate, arity));
        self
    }

    /// Mark a concrete tuple key as possible under G91 compatibility semantics.
    pub fn with_possible_terms(
        mut self,
        predicate: impl Into<String>,
        terms: Vec<Term>,
    ) -> Result<Self> {
        self.possible
            .insert(EpistemicAtomKey::from_terms(predicate, &terms)?);
        Ok(self)
    }

    /// Mark a predicate/arity pair as rejected by the candidate.
    pub fn with_rejected(mut self, predicate: impl Into<String>, arity: usize) -> Self {
        self.rejected
            .insert(EpistemicAtomKey::from_arity(predicate, arity));
        self
    }

    /// Mark a concrete tuple key as rejected by the candidate.
    pub fn with_rejected_terms(
        mut self,
        predicate: impl Into<String>,
        terms: Vec<Term>,
    ) -> Result<Self> {
        self.rejected
            .insert(EpistemicAtomKey::from_terms(predicate, &terms)?);
        Ok(self)
    }

    fn first_contradiction(&self) -> Option<(String, usize)> {
        self.known
            .iter()
            .find(|key| self.rejected.iter().any(|rejected| key.overlaps(rejected)))
            .map(|key| (key.predicate().to_string(), key.arity()))
    }

    fn contains_known(&self, atom: &Atom) -> bool {
        self.known.iter().any(|key| key.matches_atom(atom))
    }

    fn contains_possible(&self, atom: &Atom) -> bool {
        self.possible.iter().any(|key| key.matches_atom(atom))
    }

    fn contains_rejected(&self, atom: &Atom) -> bool {
        self.rejected.iter().any(|key| key.matches_atom(atom))
    }

    fn epistemic_guess_count(&self) -> usize {
        self.known.len() + self.possible.len() + self.rejected.len()
    }
}

/// One stable model in a bounded epistemic world-view fixture.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EpistemicWorld {
    facts: BTreeSet<EpistemicAtomKey>,
}

impl EpistemicWorld {
    /// Create an empty world.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a predicate/arity fact to this world.
    pub fn with_fact(mut self, predicate: impl Into<String>, arity: usize) -> Self {
        self.facts
            .insert(EpistemicAtomKey::from_arity(predicate, arity));
        self
    }

    /// Add a concrete tuple fact to this world.
    pub fn with_fact_terms(
        mut self,
        predicate: impl Into<String>,
        terms: Vec<Term>,
    ) -> Result<Self> {
        self.facts
            .insert(EpistemicAtomKey::from_terms(predicate, &terms)?);
        Ok(self)
    }

    fn contains(&self, atom: &Atom) -> bool {
        self.facts.iter().any(|fact| fact.matches_atom(atom))
    }
}

/// Non-empty set of accepted stable models used as the epistemic boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicWorldView {
    worlds: Vec<EpistemicWorld>,
}

impl EpistemicWorldView {
    /// Construct a non-empty world view.
    pub fn from_worlds(worlds: Vec<EpistemicWorld>) -> Result<Self> {
        if worlds.is_empty() {
            return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                construct: "world view boundary".to_string(),
                context: "world view requires at least one stable model".to_string(),
            });
        }
        Ok(Self { worlds })
    }

    /// Return the number of worlds in this view.
    pub fn world_count(&self) -> usize {
        self.worlds.len()
    }

    /// Evaluate an epistemic literal over this world view.
    pub fn evaluate(&self, lit: &EpistemicLiteral) -> TruthValue {
        let value = match lit.op {
            EpistemicOp::Know => self.worlds.iter().all(|world| world.contains(&lit.atom)),
            EpistemicOp::Possible => self.worlds.iter().any(|world| world.contains(&lit.atom)),
        };

        TruthValue::from_bool(if lit.negated { !value } else { value })
    }
}

/// Build the production-facing GPU execution contract for an epistemic program.
///
/// This does not launch kernels. It proves that the semantic boundary can be
/// represented as a GPU-native execution plan with explicit hot-path phases,
/// required device buffers, WCOJ planning obligations, and a typed policy that
/// rejects unsupported execution shapes instead of falling back.
pub fn plan_epistemic_gpu_execution(program: &Program) -> Result<EpistemicGpuPlan> {
    let mut prepared = program.clone();
    if prepared.authored_constraint_source_bound.is_some() {
        prepared.validate_prepared_authored_constraint_identity()?;
    } else {
        prepared.prepare_authored_constraint_identity_at_root()?;
    }
    plan_prepared_epistemic_gpu_execution(&prepared)
}

fn plan_prepared_epistemic_gpu_execution(program: &Program) -> Result<EpistemicGpuPlan> {
    program.validate_prepared_authored_constraint_identity()?;
    reject_recursive_epistemic_program(program)?;
    validate_epistemic_relation_shapes(program, &BTreeSet::new())?;
    let eir = build_eir(program)?;
    // Modal dependency cycles are intercepted by the recursive reduction before this
    // single-pass boundary. The remaining EIR has no co-evolving cycle, so one
    // candidate enumeration and world-view validation is sufficient.
    let mut epistemic_literals = Vec::new();
    let mut reductions = Vec::new();
    let mut tuple_membership_bindings = Vec::new();
    let mut solver_assumption_bindings = Vec::new();

    for (rule_index, rule) in eir.rules.iter().enumerate() {
        let mut rule_epistemic_literals = Vec::new();
        let mut positive_relational_atoms = Vec::new();
        let mut has_negated_relational_atom = false;

        for lit in &rule.body {
            match lit {
                EirBodyLiteral::Relational { negated, atom } => {
                    if *negated {
                        has_negated_relational_atom = true;
                    } else {
                        positive_relational_atoms.push(atom.clone());
                    }
                }
                EirBodyLiteral::Epistemic(lit) => {
                    rule_epistemic_literals.push(lit.clone());
                }
                EirBodyLiteral::Constraint | EirBodyLiteral::Binding => {}
            }
        }

        if rule_epistemic_literals.is_empty() {
            continue;
        }

        let reduction_index = reductions.len();
        for lit in rule_epistemic_literals {
            // Flatten any STRUCTURED finite+typed key term (`[a, b]`, `f(a, b)`)
            // element-wise into scalar GPU key columns so the existing device
            // tuple-key matcher binds/matches each element directly, and store the
            // FLATTENED literal so its atom arity/terms equal the modal relation's
            // (the plan validators and runtime read the same flattened shape).
            // Scalar keys pass through unchanged; unbounded/untyped structured
            // forms fail closed here with a precise finiteness diagnostic.
            let lit = flatten_epistemic_literal(&lit)?;
            let literal_index = epistemic_literals.len();
            let augmented_head_terms = augmented_eir_head_terms(rule);
            tuple_membership_bindings.push(EpistemicTupleMembershipBinding {
                literal_index,
                reduction_index,
                predicate: lit.atom.predicate.clone(),
                arity: lit.atom.arity,
                key_columns: (0..lit.atom.arity).collect(),
                bound_output_columns: bound_output_columns_for_terms(
                    &lit.atom.terms,
                    &augmented_head_terms,
                ),
                key_terms: lit.atom.terms.clone(),
                op: lit.op,
                negated: lit.negated,
            });
            solver_assumption_bindings.push(EpistemicSolverAssumptionBinding {
                literal_index,
                reduction_index,
                predicate: lit.atom.predicate.clone(),
                arity: lit.atom.arity,
                terms: lit.atom.terms.clone(),
                op: lit.op,
                negated: lit.negated,
            });
            epistemic_literals.push(lit);
        }
        reductions.push(EpistemicReductionPlan {
            rule_index,
            head_predicate: rule.head.predicate.clone(),
            public_head_arity: rule.head.terms.len(),
            relational_body_atoms: positive_relational_atoms.len(),
            wcoj_status: wcoj_status_for_reduction(
                &positive_relational_atoms,
                has_negated_relational_atom,
            ),
        });
    }

    if epistemic_literals.is_empty() {
        return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
            construct: "epistemic GPU execution plan".to_string(),
            context: "requires at least one epistemic literal".to_string(),
        });
    }

    // World-view integrity constraints constrain accepted candidate world views.
    // Each in-fragment constraint epistemic literal becomes a first-class
    // epistemic literal sharing an existing reduction's active-model context, so
    // its modal value is evaluated by the same GPU world-view validation path as
    // rule-body modal literals. Out-of-fragment constraint shapes fail closed.
    let constraints = lower_epistemic_constraints(
        &eir,
        &mut epistemic_literals,
        &reductions,
        &mut tuple_membership_bindings,
        &mut solver_assumption_bindings,
    )?;

    let final_output_columns = final_output_columns_for_eir(&eir);
    let gpu_plan = EpistemicGpuPlan::new(eir.mode, epistemic_literals, reductions)
        .with_tuple_membership_bindings(tuple_membership_bindings)
        .with_constraints(constraints)
        .with_final_output_columns(final_output_columns)
        .with_solver_contract(EpistemicSolverServiceContract::production_default(
            solver_assumption_bindings,
        ));
    gpu_plan.validate_tuple_membership_bindings()?;
    gpu_plan.validate_solver_contract()?;
    gpu_plan.validate_constraints()?;
    Ok(gpu_plan)
}

/// Lower in-fragment epistemic integrity constraints into first-class epistemic
/// literals and return the per-constraint world-view constraint plans.
///
/// Each constraint epistemic literal is appended to `epistemic_literals` and
/// given a tuple-membership binding plus solver assumption binding attached to
/// the final rule reduction's active-model context. The constraint body's
/// conjunction (over the appended literal indices) is what the device kernel
/// rejects when it holds in an accepted world view.
///
/// Fail-closed (typed, with source context) when:
/// - no rule reduction exists to host the constraint's modal evaluation;
/// - a constraint body mixes relational/comparison/binding literals with the
///   epistemic literals (only pure-modal constraint bodies are in fragment);
/// - a constraint epistemic atom carries a non-ground tuple key (headless
///   constraints have no reduced output column to bind variables against).
fn lower_epistemic_constraints(
    eir: &EirProgram,
    epistemic_literals: &mut Vec<EirEpistemicLiteral>,
    reductions: &[EpistemicReductionPlan],
    tuple_membership_bindings: &mut Vec<EpistemicTupleMembershipBinding>,
    solver_assumption_bindings: &mut Vec<EpistemicSolverAssumptionBinding>,
) -> Result<Vec<EpistemicConstraintPlan>> {
    let mut constraint_plans = Vec::new();
    for constraint in &eir.constraints {
        let constraint_index = constraint.authored_index.ok_or_else(|| {
            XlogError::Compilation(
                "prepared constraint compilation requires authored identities".to_string(),
            )
        })?;
        let has_epistemic = constraint
            .body
            .iter()
            .any(|lit| matches!(lit, EirBodyLiteral::Epistemic(_)));
        if !has_epistemic {
            // Purely relational constraints are handled by the reduced ordinary
            // runtime plan; they are not world-view constraints.
            continue;
        }

        if reductions.is_empty() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU world-view constraint".to_string(),
                context: format!(
                    "constraint[{constraint_index}] is an epistemic integrity constraint but the \
                     program has no epistemic rule to host its world-view evaluation; add an \
                     epistemic rule whose reduced model provides the accepted world view, or \
                     express the constraint over an existing epistemic rule"
                ),
            });
        }
        // Attach constraint modal evaluation to the final rule reduction's
        // active-model context. The reduction's reduced output drives the
        // `has_reduced_output` active-model gate used by world-view validation.
        let reduction_index = reductions.len() - 1;

        // First pass: flatten every epistemic literal (structured finite+typed
        // keys reduce element-wise to scalar GPU key columns) and reject any
        // non-epistemic body literal up front, so variable-multiplicity counting
        // below sees the final flattened key shape. A non-epistemic literal makes
        // the whole constraint out of fragment.
        let mut flattened_literals = Vec::new();
        for lit in &constraint.body {
            match lit {
                EirBodyLiteral::Epistemic(lit) => {
                    flattened_literals.push(flatten_epistemic_literal(lit)?);
                }
                EirBodyLiteral::Relational { .. }
                | EirBodyLiteral::Constraint
                | EirBodyLiteral::Binding => {
                    return Err(XlogError::UnsupportedEpistemicConstruct {
                        construct: "epistemic GPU world-view constraint".to_string(),
                        context: format!(
                            "constraint[{constraint_index}] mixes non-epistemic body literals with \
                             modal literals; world-view integrity constraints currently support \
                             pure know/possible conjunctions so the constraint can be evaluated \
                             against accepted world views without an ordinary-RIR rewrite"
                        ),
                    });
                }
            }
        }

        // Variable-keyed world-view constraints (`:- know p(X).`) range the key
        // variable EXISTENTIALLY over the modal relation's tuple-key domain: the
        // world view is pruned iff there EXISTS a binding for which the body
        // holds. A constraint-local variable that occurs EXACTLY ONCE across the
        // whole constraint body carries no join obligation, so it lowers to an
        // ANONYMOUS wildcard key column — the existing GPU wildcard tuple-key
        // matcher then ranges it over every accepted tuple, giving exact
        // existential semantics with no host scan and no reduced head column.
        //
        // A variable that occurs MORE THAN ONCE (shared across literals as a join
        // key `:- know p(X), possible q(X).`, or repeated within one literal as a
        // diagonal `:- know p(X, X).`) cannot collapse to independent wildcards
        // without weakening the constraint, so it fails closed here as unimplemented
        // scope. This is finite+typed, NOT a finiteness/resource bound: the
        // diagnostic stays a plain UnsupportedEpistemicConstruct, never a
        // ResourceExhausted, so it is not mistaken for an unbounded-domain wall.
        let mut variable_occurrences: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for lit in &flattened_literals {
            for term in &lit.atom.terms {
                if let EirTerm::Variable(name) = term {
                    *variable_occurrences.entry(name.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut literal_indices = Vec::new();
        for lit in flattened_literals {
            // Anonymize single-occurrence constraint-local variables into wildcard
            // key columns; reject shared/repeated variables (multiplicity > 1).
            let mut anonymized_terms = Vec::with_capacity(lit.atom.terms.len());
            for term in &lit.atom.terms {
                match term {
                    EirTerm::Integer(_) | EirTerm::Symbol(_) | EirTerm::Anonymous => {
                        anonymized_terms.push(term.clone());
                    }
                    EirTerm::Variable(name) => {
                        if variable_occurrences.get(name).copied().unwrap_or(0) > 1 {
                            return Err(XlogError::UnsupportedEpistemicConstruct {
                                construct: "epistemic GPU world-view constraint".to_string(),
                                context: format!(
                                    "constraint[{constraint_index}] reuses tuple-key variable \
                                     {name} across literals/positions; shared-variable epistemic \
                                     constraint joins (`:- know p(X), q(X).` / diagonal \
                                     `:- know p(X, X).`) are not yet implemented for GPU world-view \
                                     pruning. Single-occurrence variable keys (`:- know p(X).`) are \
                                     supported and range existentially over the modal relation"
                                ),
                            });
                        }
                        // A NEGATED variable-keyed literal cannot collapse to a
                        // wildcard: the wildcard computes `not (EXISTS X: know p(X))`
                        // = `forall X: not know p(X)`, but a constraint variable is
                        // EXISTENTIAL, so the body should fire on `EXISTS X: not
                        // know p(X)`. forall-not != exists-not, so the wildcard would
                        // mis-prune (it would prune iff p is EMPTY). Fail closed —
                        // finite+typed UNIMPLEMENTED scope, NOT a finiteness bound, so
                        // a plain UnsupportedEpistemicConstruct (never ResourceExhausted).
                        // Negated ALL-GROUND constraint literals are unaffected (they
                        // bind no variable, no quantifier flip — the path).
                        //
                        // Reaching here, `name` is SINGLE-occurrence (the multiplicity > 1
                        // arm above already returned) AND appears under negation — so it has
                        // NO positive binder and is NOT range-restricted. This is exactly the
                        // unsafe shape ordinary Datalog rejects (`:- not r(X).`), so emit the
                        // analogous NAF safety error rather than implying a missing feature.
                        // The meaningful negated form `:- q(X), not know p(X).` binds X with a
                        // positive literal (multiplicity > 1) and exits via the shared-variable
                        // path above, so it never reaches this branch.
                        if lit.negated {
                            return Err(XlogError::Compilation(format!(
                                "v0.8.5 naf error: unbound variable {name} in negated modal atom \
                                 {}/{} in constraint[{constraint_index}]; bind it before not with \
                                 a positive atom, or use '_' for existential positions",
                                lit.atom.predicate, lit.atom.arity
                            )));
                        }
                        // Single occurrence, POSITIVE: existential over the relation
                        // domain == wildcard. Drop the variable identity (no join, no
                        // head column to bind), routing this column through the GPU
                        // wildcard tuple-key matcher.
                        anonymized_terms.push(EirTerm::Anonymous);
                    }
                    other => {
                        return Err(XlogError::UnsupportedEpistemicConstruct {
                            construct: "epistemic GPU world-view constraint".to_string(),
                            context: format!(
                                "constraint[{constraint_index}] uses {} {}/{} with an unsupported \
                                 tuple-key term {other:?}; headless world-view constraints support \
                                 ground (integer/symbol) and single-occurrence variable/anonymous \
                                 modal atoms",
                                eir_epistemic_literal_label(&lit),
                                lit.atom.predicate,
                                lit.atom.arity
                            ),
                        });
                    }
                }
            }
            // Rebuild the literal with anonymized terms so the stored literal, its
            // tuple-membership binding key_terms, and its solver assumption binding
            // terms all carry the SAME shape (the plan validator requires
            // binding.key_terms == literal.atom.terms).
            let mut lit = lit;
            lit.atom.terms = anonymized_terms;

            let literal_index = epistemic_literals.len();
            let bound_output_columns = vec![None; lit.atom.arity];
            tuple_membership_bindings.push(EpistemicTupleMembershipBinding {
                literal_index,
                reduction_index,
                predicate: lit.atom.predicate.clone(),
                arity: lit.atom.arity,
                key_columns: (0..lit.atom.arity).collect(),
                key_terms: lit.atom.terms.clone(),
                bound_output_columns,
                op: lit.op,
                negated: lit.negated,
            });
            solver_assumption_bindings.push(EpistemicSolverAssumptionBinding {
                literal_index,
                reduction_index,
                predicate: lit.atom.predicate.clone(),
                arity: lit.atom.arity,
                terms: lit.atom.terms.clone(),
                op: lit.op,
                negated: lit.negated,
            });
            epistemic_literals.push(lit);
            literal_indices.push(literal_index);
        }

        constraint_plans.push(EpistemicConstraintPlan {
            constraint_index,
            literal_indices,
        });
    }
    Ok(constraint_plans)
}

/// Structural classification of an epistemic program with respect to ordinary
/// (non-modal) recursion.
///
/// Recursion through positive/negated body literals normally fails closed in an
/// epistemic program because the single-pass world-view executor cannot iterate a
/// fixpoint. The well-defined sub-fragment "Case A" — recursion lives in the
/// ordinary predicate while every modal atom in a recursion-participating rule is a
/// positive `know`/`possible` over an *invariant* relation (an EDB or a lower
/// non-recursive, non-epistemic stratum) — is admitted instead: the modal atom's
/// extension is fixed independent of the recursion, so it can be resolved to its
/// gated relation and the reduced ordinary program iterated by the existing
/// recursive/semi-naive engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecursiveEpistemicClass {
    /// The program has no ordinary or modal dependency cycle; the single-pass
    /// epistemic world-view executor handles it.
    NonRecursive,
    /// Case A: ordinary recursion with every recursion-participating modal atom over
    /// an invariant relation. Routed to the ordinary recursive engine after a
    /// Case-A reduction (see [`reduce_case_a_epistemic_program_to_ordinary`]).
    CaseA,
    /// Case B: ordinary recursion where at least one POSITIVE `know`/`possible` modal
    /// ranges over a NON-invariant relation that CO-EVOLVES with the recursion (the
    /// modal target sits in the recursive SCC, or transitively depends on it). The
    /// modal truth and the ordinary derivation are a single co-evolving founded least
    /// fixpoint: resolving each positive modal to its (now recursive) ordinary atom and
    /// iterating the existing semi-naive engine computes the FAEEL founded least
    /// fixpoint directly — unfounded self-support is excluded by construction (the
    /// least model of a positive program IS its founded model), so no separate
    /// foundedness drop is needed. Routed exactly like Case A through
    /// [`reduce_case_a_epistemic_program_to_ordinary`] and the ordinary recursive
    /// engine.
    ///
    /// ADMISSION IS POLARITY/MODE-SCOPED (proved in
    /// [`classify_recursive_epistemic_program`]): a NEGATED modal over a non-invariant
    /// target is admitted when the reduced ordinary program is stratified; a genuine
    /// negation cycle is delegated to the high-level GPU-backed WFS alternating-fixpoint
    /// executor. A `possible` modal over a co-evolving target is admitted under FAEEL
    /// as the founded least fixpoint. Under G91, exact head-tuple cycles are
    /// intercepted by [`try_prepare_g91_compatibility_reduction`] and evaluated by an
    /// explicit descending compatibility fixpoint.
    CaseB,
    /// Recursion arises entirely through modal dependencies rather than an ordinary
    /// body cycle. FAEEL resolves the modal edges into an ordinary founded least
    /// fixpoint. G91 exact head-tuple `possible` cycles use the explicit descending
    /// compatibility plan; other admitted modal edges resolve to ordinary atoms. This
    /// class cannot use the single-pass planner, which cannot distinguish a founded
    /// predecessor chain from an unfounded tuple cycle.
    ModalCycle,
}

/// Reject epistemic programs that contain an ordinary or modal dependency cycle before
/// the single-pass GPU world-view planner.
///
/// [`plan_epistemic_gpu_execution`] builds a single-pass plan that evaluates each
/// candidate world view exactly once; it cannot iterate a fixpoint. Admissible cycles
/// are intercepted by recursive source preparation and delegated to either the
/// ordinary recursive engine, GPU-backed WFS, or the explicit G91 compatibility
/// fixpoint. This guard remains defense-in-depth for direct callers of the single-pass
/// planner.
fn reject_recursive_epistemic_program(program: &Program) -> Result<()> {
    match classify_recursive_epistemic_program(program) {
        Ok(RecursiveEpistemicClass::NonRecursive) => Ok(()),
        Ok(
            RecursiveEpistemicClass::CaseA
            | RecursiveEpistemicClass::CaseB
            | RecursiveEpistemicClass::ModalCycle,
        ) => Err(recursive_epistemic_rejection(
            "an epistemic program contains an ordinary or modal dependency cycle; the \
                 single-pass epistemic GPU planner cannot iterate a fixpoint. Admissible \
                 recursive epistemic programs require recursive source preparation and an \
                 iterative execution plan, not this planner.",
        )),
        // Recursive shapes outside the admissible fragment already carry a specific
        // typed diagnostic.
        Err(err) => Err(err),
    }
}

/// Classify ordinary and modal dependency cycles in an epistemic program.
///
/// Returns a typed [`XlogError::UnsupportedEpistemicConstruct`] for a recursive shape
/// outside the supported ordinary, co-evolving, or modal-cycle fragments.
pub fn classify_recursive_epistemic_program(program: &Program) -> Result<RecursiveEpistemicClass> {
    let has_epistemic = program.rules.iter().any(|rule| {
        rule.body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
    });
    if !has_epistemic {
        // No epistemic literals: the ordinary recursive engine handles this program.
        return Ok(RecursiveEpistemicClass::NonRecursive);
    }

    // Keep the ordinary graph separately from the full co-evolution graph. Positive
    // and negated ordinary atoms participate in both. Modal atoms participate in the
    // co-evolution graph because a modal dependency cycle must be solved as a
    // fixpoint: treating it as single-pass either fabricates an unfounded tuple cycle
    // or drops a valid transition from a founded predecessor.
    let (ordinary_deps, deps) = epistemic_dependency_graphs(program);

    let ordinary_recursive_predicates: BTreeSet<&str> = ordinary_deps
        .keys()
        .copied()
        .filter(|pred| {
            predicate_dependency_reaches(pred, pred, &ordinary_deps, &mut BTreeSet::new())
        })
        .collect();

    // Collect every predicate in an ordinary-or-modal dependency cycle.
    let recursive_predicates: BTreeSet<&str> = deps
        .keys()
        .copied()
        .filter(|pred| predicate_dependency_reaches(pred, pred, &deps, &mut BTreeSet::new()))
        .collect();

    if recursive_predicates.is_empty() {
        return Ok(RecursiveEpistemicClass::NonRecursive);
    }
    let modal_only_recursion = ordinary_recursive_predicates.is_empty();

    // Recursion is present. Two admissible classes (anything else fails closed):
    //
    //   Case A — every modal atom is a POSITIVE `know`/`possible` over an INVARIANT
    //   relation (extension fixed independent of the recursion). The recursion joins
    //   against a fixed gated relation.
    //
    //   Case B — at least one POSITIVE `know`/`possible` modal ranges over a
    //   NON-invariant relation that CO-EVOLVES with the recursion (the modal target is
    //   itself recursive / epistemic / transitively depends on the recursion). Modal
    //   truth and the ordinary derivation are a single founded least fixpoint: resolving
    //   the positive modal to its (now recursive) ordinary atom and iterating the
    //   semi-naive engine computes the FAEEL founded least fixpoint directly. The least
    //   model of the resulting POSITIVE program IS its founded model, so unfounded
    //   self-support is excluded by construction (no separate foundedness drop needed),
    //   and a program with no founding simply yields the exact empty extension.
    //
    // FAEEL and non-compatibility G91 edges use the same positive-modal-to-positive-
    // atom reduction, so the structural difference between Case A and Case B is whether
    // the resolved relation is fixed or part of the SCC. Exact G91 head-tuple
    // `possible` cycles are intercepted first and use the upper-bound/frozen-snapshot
    // reduction. The whole program is scanned because either reduction rewrites every
    // remaining modal literal.
    //
    // SOUNDNESS FLOOR:
    //   * a NEGATED modal over a non-invariant target is admitted as Case B. If the
    //     reduced program is stratified, ordinary stratified negation is enough; if it
    //     contains a reduced cycle through negation, the high-level executor routes it
    //     to GPU-backed WFS rather than host WFS.
    //   * an exact head-tuple `possible` modal over a co-evolving target under G91 is
    //     admitted only through the explicit descending compatibility reduction. FAEEL
    //     `possible` remains the founded least fixpoint. A cycle carried only by modal
    //     dependencies is classified as `ModalCycle`; execution then selects the
    //     semantic reduction before it can reach the single-pass path.
    let invariant = InvariantRelations::analyze(program);
    let mut saw_case_b = false;
    // A NEGATED modal over a NON-invariant target is admissible after reduction. The
    // high-level executor chooses ordinary stratified execution or GPU-backed WFS based
    // on the reduced program's monotonicity.
    let mut saw_negated_non_invariant_modal = false;
    for rule in &program.rules {
        for lit in &rule.body {
            let BodyLiteral::Epistemic(modal) = lit else {
                continue;
            };
            if invariant.is_invariant(&modal.atom.predicate) {
                // Modal over an INVARIANT relation: admissible Case-A. A positive
                // `know`/`possible` resolves to a positive ordinary join over the gated
                // relation; a NEGATED `not know`/`not possible` over an invariant
                // relation equals ordinary `not R` (the world view agrees with R on an
                // invariant relation), an anti-join with NO modal gating.
                continue;
            }

            // NON-invariant modal target: the gated relation co-evolves with the
            // recursion.
            if modal.negated {
                // A NEGATED modal over a NON-invariant relation is the deferred case.
                // SOUNDNESS ARGUMENT (why stratification decides it): when the reduced
                // ordinary program (`not know R` -> `not R`, `know R` -> `R`) is
                // STRATIFIED, its perfect model is TOTAL and 2-valued. A total
                // 2-valued model makes every modal target R 2-valued, so under FAEEL
                // `know R == possible R == R` and `not know R == not possible R == not
                // R` (the modal op stops mattering once R is determined -- the same
                // equivalence established for DETERMINED targets, generalized
                // here to STRATIFIED targets). Replacing each modal by its ordinary
                // atom therefore preserves truth values, so the stratified perfect
                // model of the reduced program IS the FAEEL model. The 2-valued
                // (stratified) property is the linchpin.
                //
                // When the reduced program is NOT stratified (a cycle through the
                // negation), the sound semantics is the 3-valued WELL-FOUNDED model
                // (R partly UNDEFINED). Host-side WFS / stable-model solving remains
                // precluded by the no-host-solver lock, so the high-level executor
                // delegates that reduced program to the GPU-backed WFS path.
                saw_negated_non_invariant_modal = true;
                saw_case_b = true;
                continue;
            }

            // POSITIVE `know` (any mode), FAEEL `possible`, or G91 `possible` over a
            // co-evolving target: admissible Case B. FAEEL/know resolve to the
            // ordinary atom. Exact G91 head-tuple `possible` cycles are intercepted by
            // the compatibility reduction; remaining admitted modal edges resolve to
            // ordinary atoms.
            saw_case_b = true;
        }
    }

    // NEGATED-MODAL DISCRIMINATOR: a deferred negated-modal-over-non-invariant is accepted
    // as Case B. The high-level executor inspects the reduced ordinary program: no
    // negation cycle routes to ordinary stratified execution; a negation cycle routes
    // to the GPU-backed WFS alternating-fixpoint plan.
    if saw_negated_non_invariant_modal {
        // Stratified reduced programs continue through the ordinary semi-naive path.
        // Non-monotone reduced programs are handled by the high-level GPU compiler's
        // WFS plan; host WFS is not an accepted execution fallback.
        let _reduced = reduce_case_a_epistemic_program_to_ordinary(program);
    }

    // SOUNDNESS GUARD: a recursive epistemic program (Case A/B) routes through the PURE
    // ordinary semi-naive engine (`LogicExecutionPlan::Ordinary`), which never runs the
    // world-view integrity-constraint kernel; the Case-A/B reduction DROPS every
    // constraint that contains a modal literal. For a NON-recursive program the
    // single-pass world-view path evaluates those constraints, but on the recursive
    // route a co-occurring epistemic constraint (`:- know X` / `:- not know X`) would be
    // SILENTLY IGNORED, yielding a result that includes rows a valid world view forbids.
    // That is an UNSOUND admission (worse than a rejection), so fail closed when an
    // epistemic constraint co-occurs with recursion. (Non-recursive epistemic-constraint
    // Non-recursive epistemic-constraint programs never reach here; they run the
    // constraint kernel on the single-pass path.)
    let has_epistemic_constraint = program.constraints.iter().any(|constraint| {
        constraint
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
    });
    if has_epistemic_constraint {
        return Err(recursive_epistemic_rejection(
            "a recursive epistemic program carries an epistemic integrity constraint \
             (`:- know ...` / `:- not know ...`). Recursive reductions do not run the \
             single-pass world-view constraint kernel and would otherwise drop the \
             modal constraint, yielding a result that ignores it. To keep results sound \
             this fails closed rather than silently dropping the constraint. \
             Remove the recursion or express the integrity constraint over a \
             non-recursive (single-pass) epistemic relation.",
        ));
    }

    if modal_only_recursion {
        debug_assert!(
            saw_case_b,
            "a modal-only cycle must have a co-evolving target"
        );
        Ok(RecursiveEpistemicClass::ModalCycle)
    } else if saw_case_b {
        Ok(RecursiveEpistemicClass::CaseB)
    } else {
        Ok(RecursiveEpistemicClass::CaseA)
    }
}

type PredicateDependencyMap<'a> = BTreeMap<&'a str, BTreeSet<&'a str>>;

fn epistemic_dependency_graphs(
    program: &Program,
) -> (PredicateDependencyMap<'_>, PredicateDependencyMap<'_>) {
    let mut ordinary_dependencies = BTreeMap::new();
    let mut all_dependencies = BTreeMap::new();
    for rule in &program.rules {
        let head = rule.head.predicate.as_str();
        let all = all_dependencies.entry(head).or_insert_with(BTreeSet::new);
        let ordinary = ordinary_dependencies
            .entry(head)
            .or_insert_with(BTreeSet::new);
        for literal in &rule.body {
            match literal {
                BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                    all.insert(atom.predicate.as_str());
                    ordinary.insert(atom.predicate.as_str());
                }
                BodyLiteral::Epistemic(modal) => {
                    all.insert(modal.atom.predicate.as_str());
                }
                BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {}
            }
        }
    }
    (ordinary_dependencies, all_dependencies)
}

fn predicate_dependency_reaches<'a>(
    start: &'a str,
    target: &str,
    dependencies: &BTreeMap<&'a str, BTreeSet<&'a str>>,
    seen: &mut BTreeSet<&'a str>,
) -> bool {
    let Some(next) = dependencies.get(start) else {
        return false;
    };
    for &predicate in next {
        if predicate == target {
            return true;
        }
        if seen.insert(predicate)
            && predicate_dependency_reaches(predicate, target, dependencies, seen)
        {
            return true;
        }
    }
    false
}

/// Modal dependency edges whose head and target belong to the same recursive
/// component. The modal literal itself supplies the head-to-target edge; a return
/// path from target to head proves SCC membership.
fn recursive_modal_dependency_edges(program: &Program) -> BTreeSet<(String, String)> {
    let (_, dependencies) = epistemic_dependency_graphs(program);
    let mut edges = BTreeSet::new();
    for rule in &program.rules {
        for literal in &rule.body {
            let BodyLiteral::Epistemic(modal) = literal else {
                continue;
            };
            if modal.atom.predicate == rule.head.predicate
                || predicate_dependency_reaches(
                    modal.atom.predicate.as_str(),
                    rule.head.predicate.as_str(),
                    &dependencies,
                    &mut BTreeSet::new(),
                )
            {
                edges.insert((rule.head.predicate.clone(), modal.atom.predicate.clone()));
            }
        }
    }
    edges
}

fn recursive_epistemic_rejection(context: &str) -> XlogError {
    XlogError::UnsupportedEpistemicConstruct {
        construct: "recursive epistemic program".to_string(),
        context: context.to_string(),
    }
}

/// Predicates whose extension is fixed independent of any ordinary recursion or
/// epistemic literal in the program.
///
/// A predicate is invariant when it is EDB (defined only by ground facts) or its
/// entire transitive ordinary-definition closure is free of epistemic literals and of
/// ordinary recursion. Such a relation is computed once in a lower stratum, so a
/// positive `know`/`possible` over it has a fixed gated extension that a recursive
/// fixpoint can join against.
struct InvariantRelations<'a> {
    /// Ordinary (positive/negated) body-predicate edges per head predicate.
    ordinary_deps: BTreeMap<&'a str, BTreeSet<&'a str>>,
    /// Predicates whose definition (any defining non-fact rule) contains an epistemic
    /// body literal.
    epistemic_heads: BTreeSet<&'a str>,
    /// Predicates defined by at least one non-fact rule (i.e. not pure EDB).
    derived_heads: BTreeSet<&'a str>,
}

impl<'a> InvariantRelations<'a> {
    fn analyze(program: &'a Program) -> Self {
        let mut ordinary_deps: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        let mut epistemic_heads: BTreeSet<&str> = BTreeSet::new();
        let mut derived_heads: BTreeSet<&str> = BTreeSet::new();
        for rule in &program.rules {
            if rule.body.is_empty() {
                continue;
            }
            let head = rule.head.predicate.as_str();
            derived_heads.insert(head);
            let entry = ordinary_deps.entry(head).or_default();
            for lit in &rule.body {
                match lit {
                    BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                        entry.insert(atom.predicate.as_str());
                    }
                    BodyLiteral::Epistemic(_) => {
                        epistemic_heads.insert(head);
                    }
                    BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {}
                }
            }
        }
        Self {
            ordinary_deps,
            epistemic_heads,
            derived_heads,
        }
    }

    /// Whether `predicate`'s extension is fixed independent of the recursion.
    fn is_invariant(&self, predicate: &str) -> bool {
        let mut seen = BTreeSet::new();
        self.is_invariant_inner(predicate, &mut seen)
    }

    fn is_invariant_inner<'b>(&'b self, predicate: &'b str, seen: &mut BTreeSet<&'b str>) -> bool {
        if !seen.insert(predicate) {
            // A cycle reaching `predicate` means recursion: not invariant.
            return false;
        }
        let invariant = if !self.derived_heads.contains(predicate) {
            // Pure EDB relation: invariant by construction.
            true
        } else if self.epistemic_heads.contains(predicate) {
            // Definition itself uses a modal literal: not a fixed lower stratum.
            false
        } else {
            match self.ordinary_deps.get(predicate) {
                None => true,
                Some(deps) => deps.iter().all(|dep| self.is_invariant_inner(dep, seen)),
            }
        };
        // `seen` is the active recursion stack, not a global visited set. Leaving a
        // completed dependency in it would mistake a shared acyclic dependency in a
        // diamond for a back edge when a sibling branch reaches the same predicate.
        seen.remove(predicate);
        invariant
    }
}

fn eir_epistemic_literal_label(lit: &xlog_ir::EirEpistemicLiteral) -> &'static str {
    match (lit.negated, lit.op) {
        (false, EirEpistemicOp::Know) => "know",
        (false, EirEpistemicOp::Possible) => "possible",
        (true, EirEpistemicOp::Know) => "not know",
        (true, EirEpistemicOp::Possible) => "not possible",
    }
}

fn has_independent_founded_support(eir: &EirProgram, atom: &xlog_ir::EirAtom) -> bool {
    if atom.arity > 0 && !atom.terms.iter().all(eir_term_is_ground) {
        return false;
    }

    let mut support_stack = Vec::new();
    has_independent_founded_support_inner(eir, atom, &mut support_stack)
}

/// Whether a ground atom is unconditionally derived by the authored ordinary rules.
///
/// Unlike `has_independent_founded_support`, this proof does not treat an undeclared
/// runtime EDB tuple as present merely because the predicate has no defining rule.
/// It is therefore suitable for proving that a global modal output gate is a no-op:
/// every positive dependency must itself be derivable from an explicit fact or an
/// ordinary rule, and constraints/bindings are rejected because EIR intentionally
/// erases the expression needed to prove them at this boundary.
fn has_unconditional_ground_founded_support(eir: &EirProgram, atom: &xlog_ir::EirAtom) -> bool {
    if !atom.terms.iter().all(eir_term_is_ground) {
        return false;
    }

    let mut support_stack = Vec::new();
    has_unconditional_ground_founded_support_inner(eir, atom, &mut support_stack)
}

fn has_unconditional_ground_founded_support_inner(
    eir: &EirProgram,
    atom: &xlog_ir::EirAtom,
    support_stack: &mut Vec<(String, Vec<EirTerm>)>,
) -> bool {
    let key = (atom.predicate.clone(), atom.terms.clone());
    if support_stack.iter().any(|ancestor| ancestor == &key) {
        return false;
    }
    support_stack.push(key);

    let supported = eir.rules.iter().any(|rule| {
        let Some(substitution) = head_substitution_to_atom(&rule.head, atom) else {
            return false;
        };
        rule.body.iter().all(|literal| match literal {
            EirBodyLiteral::Relational {
                negated: false,
                atom,
            } => substitute_eir_atom(atom, &substitution).is_some_and(|atom| {
                atom.terms.iter().all(eir_term_is_ground)
                    && has_unconditional_ground_founded_support_inner(eir, &atom, support_stack)
            }),
            EirBodyLiteral::Epistemic(_)
            | EirBodyLiteral::Relational { negated: true, .. }
            | EirBodyLiteral::Constraint
            | EirBodyLiteral::Binding => false,
        })
    });

    support_stack.pop();
    supported
}

fn has_tuple_level_independent_founded_support(
    eir: &EirProgram,
    modal_rule: &xlog_ir::EirRule,
    atom: &xlog_ir::EirAtom,
) -> bool {
    if atom.arity == 0 {
        return false;
    }

    let modal_domain = positive_relational_body_atoms(modal_rule);
    eir.rules.iter().any(|support_rule| {
        if !support_rule_head_matches_modal_atom(support_rule, atom) {
            return false;
        }
        let mut support_stack = vec![(atom.predicate.clone(), atom.arity)];
        if !eir_rule_has_independent_founded_body(eir, support_rule, &mut support_stack) {
            return false;
        }
        let Some(substitution) = head_substitution_to_atom(&support_rule.head, atom) else {
            return false;
        };
        let support_domain = positive_relational_body_atoms(support_rule);
        if support_domain.is_empty() {
            return false;
        }
        let Some(substituted_support_domain) = support_domain
            .iter()
            .map(|atom| substitute_eir_atom(atom, &substitution))
            .collect::<Option<Vec<_>>>()
        else {
            return false;
        };
        substituted_support_domain.iter().all(|support_atom| {
            modal_domain
                .iter()
                .any(|modal_atom| modal_atom == support_atom)
        })
    })
}

fn positive_relational_body_atoms(rule: &xlog_ir::EirRule) -> Vec<xlog_ir::EirAtom> {
    rule.body
        .iter()
        .filter_map(|lit| match lit {
            EirBodyLiteral::Relational {
                negated: false,
                atom,
            } => Some(atom.clone()),
            _ => None,
        })
        .collect()
}

fn support_rule_head_matches_modal_atom(rule: &xlog_ir::EirRule, atom: &xlog_ir::EirAtom) -> bool {
    rule.head.predicate == atom.predicate
        && rule.head.arity == atom.arity
        && head_substitution_to_atom(&rule.head, atom).is_some()
}

fn head_substitution_to_atom(
    head: &xlog_ir::EirAtom,
    atom: &xlog_ir::EirAtom,
) -> Option<BTreeMap<String, EirTerm>> {
    if head.predicate != atom.predicate || head.arity != atom.arity {
        return None;
    }
    let mut substitution = BTreeMap::new();
    for (head_term, atom_term) in head.terms.iter().zip(&atom.terms) {
        match head_term {
            EirTerm::Variable(name) => match substitution.get(name) {
                Some(existing) if existing != atom_term => return None,
                Some(_) => {}
                None => {
                    substitution.insert(name.clone(), atom_term.clone());
                }
            },
            EirTerm::Anonymous => return None,
            other if other == atom_term => {}
            _ => return None,
        }
    }
    Some(substitution)
}

fn substitute_eir_atom(
    atom: &xlog_ir::EirAtom,
    substitution: &BTreeMap<String, EirTerm>,
) -> Option<xlog_ir::EirAtom> {
    let terms = atom
        .terms
        .iter()
        .map(|term| substitute_eir_term(term, substitution))
        .collect::<Option<Vec<_>>>()?;
    Some(xlog_ir::EirAtom {
        predicate: atom.predicate.clone(),
        arity: atom.arity,
        terms,
    })
}

fn substitute_eir_term(
    term: &EirTerm,
    substitution: &BTreeMap<String, EirTerm>,
) -> Option<EirTerm> {
    match term {
        EirTerm::Variable(name) => Some(
            substitution
                .get(name)
                .cloned()
                .unwrap_or_else(|| term.clone()),
        ),
        EirTerm::Anonymous => None,
        EirTerm::List(items) => items
            .iter()
            .map(|item| substitute_eir_term(item, substitution))
            .collect::<Option<Vec<_>>>()
            .map(EirTerm::List),
        EirTerm::Cons { head, tail } => Some(EirTerm::Cons {
            head: Box::new(substitute_eir_term(head, substitution)?),
            tail: Box::new(substitute_eir_term(tail, substitution)?),
        }),
        EirTerm::Compound { functor, args } => Some(EirTerm::Compound {
            functor: functor.clone(),
            args: args
                .iter()
                .map(|arg| substitute_eir_term(arg, substitution))
                .collect::<Option<Vec<_>>>()?,
        }),
        EirTerm::Aggregate { .. } => None,
        EirTerm::Integer(_)
        | EirTerm::FloatBits(_)
        | EirTerm::String(_)
        | EirTerm::Symbol(_)
        | EirTerm::PredRef(_) => Some(term.clone()),
    }
}

fn has_independent_founded_support_inner(
    eir: &EirProgram,
    atom: &xlog_ir::EirAtom,
    support_stack: &mut Vec<(String, usize)>,
) -> bool {
    if atom.arity > 0 && !atom.terms.iter().all(eir_term_is_ground) {
        return false;
    }

    let key = (atom.predicate.clone(), atom.arity);
    if support_stack.iter().any(|ancestor| ancestor == &key) {
        return false;
    }
    support_stack.push(key);

    let supported = eir.rules.iter().any(|rule| {
        let Some(substitution) = head_substitution_to_atom(&rule.head, atom) else {
            return false;
        };
        eir_rule_has_independent_founded_body_with_substitution(
            eir,
            rule,
            &substitution,
            support_stack,
        )
    });

    support_stack.pop();
    supported
}

fn eir_rule_has_independent_founded_body(
    eir: &EirProgram,
    rule: &xlog_ir::EirRule,
    support_stack: &mut Vec<(String, usize)>,
) -> bool {
    eir_rule_has_independent_founded_body_with_substitution(
        eir,
        rule,
        &BTreeMap::new(),
        support_stack,
    )
}

fn eir_rule_has_independent_founded_body_with_substitution(
    eir: &EirProgram,
    rule: &xlog_ir::EirRule,
    substitution: &BTreeMap<String, EirTerm>,
    support_stack: &mut Vec<(String, usize)>,
) -> bool {
    rule.body.iter().all(|lit| match lit {
        EirBodyLiteral::Epistemic(_) => false,
        EirBodyLiteral::Relational { negated: true, .. } => false,
        EirBodyLiteral::Relational {
            negated: false,
            atom,
        } => {
            let Some(atom) = substitute_eir_atom(atom, substitution) else {
                return false;
            };
            let dependency_key = (atom.predicate.clone(), atom.arity);
            if support_stack
                .iter()
                .any(|ancestor| ancestor == &dependency_key)
            {
                return false;
            }
            if !eir
                .rules
                .iter()
                .any(|rule| head_substitution_to_atom(&rule.head, &atom).is_some())
            {
                return true;
            }
            has_independent_founded_support_inner(eir, &atom, support_stack)
        }
        // EIR preserves only the presence of comparisons and bindings, not the
        // expression needed to prove that they hold for every tuple in the modal
        // rule's domain. Treating them as unconditional would let a restricted
        // support rule (for example `X = 1`) found unrelated tuples. A richer proof
        // may admit such rules later; this structural foundedness check must remain
        // conservative until then.
        EirBodyLiteral::Constraint | EirBodyLiteral::Binding => false,
    })
}

fn eir_term_is_ground(term: &EirTerm) -> bool {
    match term {
        EirTerm::Variable(_) | EirTerm::Anonymous | EirTerm::Aggregate { .. } => false,
        EirTerm::Integer(_) | EirTerm::FloatBits(_) | EirTerm::String(_) | EirTerm::Symbol(_) => {
            true
        }
        EirTerm::List(items) => items.iter().all(eir_term_is_ground),
        EirTerm::Cons { head, tail } => eir_term_is_ground(head) && eir_term_is_ground(tail),
        EirTerm::Compound { args, .. } => args.iter().all(eir_term_is_ground),
        EirTerm::PredRef(_) => true,
    }
}

/// Compile an epistemic program into its GPU contract and reduced runtime plan.
///
/// This is the first production-lowering boundary for epistemic execution. It
/// removes epistemic literals only after `plan_epistemic_gpu_execution` proves
/// the explicit EIR/GPU semantic contract, then sends the ordinary reduced
/// program through the same compiler, optimizer, helper-splitting, and WCOJ
/// promotion pipeline used by non-epistemic programs.
pub fn compile_epistemic_gpu_execution(program: &Program) -> Result<EpistemicExecutablePlan> {
    compile_epistemic_gpu_execution_with_stats_snapshot(program, None)
}

/// Compile an epistemic program with an optional production statistics snapshot.
///
/// This preserves the reduced ordinary-body planner contract: cardinality,
/// selectivity, access heat, prefix-degree, sorted-layout, and helper-splitting
/// decisions are owned by the existing production compiler pipeline rather than
/// by an epistemic side planner.
pub fn compile_epistemic_gpu_execution_with_stats_snapshot(
    program: &Program,
    stats_snapshot: Option<&StatsSnapshot>,
) -> Result<EpistemicExecutablePlan> {
    let mut prepared = program.clone();
    if prepared.authored_constraint_source_bound.is_some() {
        prepared.validate_prepared_authored_constraint_identity()?;
    } else {
        prepared.prepare_authored_constraint_identity_at_root()?;
    }
    compile_epistemic_gpu_execution_inner(&prepared, stats_snapshot, false)
}

/// Lower an epistemic program to its GPU contract and reduced runtime plan.
///
/// When `allow_multiple_output_heads` is false (the default monolithic and
/// single-head split path) the single-output-buffer contract
/// ([`require_single_epistemic_output_relation`]) is enforced. When true, the
/// caller has proven the component is a JOINT-SOLVABLE coalesced multi-head
/// component (see [`classify_cross_component_modal_coupling`]): one candidate
/// enumeration + world-view validation over the combined modal literals, with
/// each head materialized against the shared accepted world view at runtime.
fn compile_epistemic_gpu_execution_inner(
    program: &Program,
    stats_snapshot: Option<&StatsSnapshot>,
    allow_multiple_output_heads: bool,
) -> Result<EpistemicExecutablePlan> {
    program.validate_prepared_authored_constraint_identity()?;
    let gpu_plan = plan_prepared_epistemic_gpu_execution(program)?;
    if !allow_multiple_output_heads {
        require_single_epistemic_output_relation(&gpu_plan)?;
    }
    // JOINT-SOLVING multi-head materialization now projects each coupled head by ITS
    // OWN `public_head_arity` (see `final_output_columns_for_materialization`): each
    // head is materialized from its own reduced relation buffer with its own
    // reduction row-filter, reading only the store/world-view boundary. An augmented
    // multi-head component (a modal-literal variable absent from a head) therefore
    // projects every head's public tuple shape soundly, including coupled heads of
    // DIFFERING arity. The former blanket fail-closed guard on
    // `final_output_columns.is_some()` over multiple heads is no longer needed.
    let reduced_program = reduce_epistemic_program_to_ordinary(program)?;
    let mut compiler = Compiler::new();
    let reduced_runtime_plan =
        compiler.compile_prepared_program_with_stats_snapshot(&reduced_program, stats_snapshot)?;
    let relation_ids = compiler
        .rel_ids()
        .iter()
        .map(|(name, rel)| (name.clone(), *rel))
        .collect();

    Ok(EpistemicExecutablePlan {
        gpu_plan,
        relation_ids,
        reduced_runtime_plan,
    })
}

/// Authored epistemic source after static validation and exact FAEEL foundedness
/// filtering, ready for dependency classification and executable planning.
#[derive(Debug, Clone)]
pub struct PreparedEpistemicProgram {
    active_program: Program,
    removed_unfounded_rule_count: usize,
}

/// Modal-free programs and frozen-relation bindings for a Gelfond-1991
/// compatibility greatest fixpoint.
///
/// The upper-bound program removes only selected positive `possible` gates in a
/// recursive component. The refinement program replaces those same gates with
/// reads from frozen snapshots of the preceding iteration. Re-evaluating the
/// refinement from the original extensional inputs until the selected relations
/// stop changing computes compatibility per concrete tuple instead of assuming
/// that predicate-level strongly connected component membership is sufficient.
#[derive(Debug, Clone)]
pub struct G91CompatibilityReduction {
    upper_bound_program: Program,
    refinement_program: Program,
    snapshot_relations: BTreeMap<String, String>,
    convergence_predicates: Vec<String>,
}

impl G91CompatibilityReduction {
    /// Program whose selected compatibility gates are removed to establish the
    /// finite initial upper bound.
    pub fn upper_bound_program(&self) -> &Program {
        &self.upper_bound_program
    }

    /// Program whose selected compatibility gates read the preceding iteration's
    /// frozen relation snapshots.
    pub fn refinement_program(&self) -> &Program {
        &self.refinement_program
    }

    /// Source relation to collision-free frozen snapshot relation name.
    pub fn snapshot_relations(&self) -> &BTreeMap<String, String> {
        &self.snapshot_relations
    }

    /// Intensional relations compared for convergence after each refinement.
    pub fn convergence_predicates(&self) -> &[String] {
        &self.convergence_predicates
    }
}

impl PreparedEpistemicProgram {
    /// Program remaining after exact foundedness filtering.
    pub fn active_program(&self) -> &Program {
        &self.active_program
    }

    /// Whether preparation removed at least one unfounded rule.
    pub fn removed_unfounded_rules(&self) -> bool {
        self.removed_unfounded_rule_count != 0
    }
}

/// Validate authored contracts before semantics can remove a rule, then exclude only
/// positive exact-tuple FAEEL self-support with no independent founded support.
pub fn prepare_epistemic_program(program: &Program) -> Result<PreparedEpistemicProgram> {
    let prepared = prepare_root_authored_constraint_identity(program)?;
    validate_prepared_epistemic_source_program(&prepared)?;
    let removed_rule_indices = faeel_unfounded_exact_tuple_self_support_rule_indices(&prepared);
    Ok(PreparedEpistemicProgram {
        active_program: program_without_rule_indices(&prepared, &removed_rule_indices),
        removed_unfounded_rule_count: removed_rule_indices.len(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct G91CompatibilityLiteralLocation {
    rule_index: usize,
    literal_index: usize,
}

/// Build the explicit tuple-level Gelfond-1991 compatibility reduction, when the
/// prepared program contains a supported positive `possible` dependency cycle.
pub fn try_prepare_g91_compatibility_reduction(
    prepared: &PreparedEpistemicProgram,
) -> Result<Option<G91CompatibilityReduction>> {
    let program = prepared.active_program();
    if program.directives.epistemic_mode_or_default() != EpistemicMode::G91 {
        return Ok(None);
    }
    if classify_recursive_epistemic_program(program)? == RecursiveEpistemicClass::NonRecursive {
        return Ok(None);
    }

    validate_epistemic_derived_relation_identity(program, &BTreeSet::new())?;
    let recursive_modal_edges = recursive_modal_dependency_edges(program);
    let invariant = InvariantRelations::analyze(program);
    let mut locations = BTreeSet::new();
    let mut target_arities = BTreeMap::new();
    for (rule_index, rule) in program.rules.iter().enumerate() {
        for (literal_index, literal) in rule.body.iter().enumerate() {
            let BodyLiteral::Epistemic(modal) = literal else {
                continue;
            };
            if !is_g91_compatibility_literal(rule, modal, &invariant, &recursive_modal_edges) {
                continue;
            }
            locations.insert(G91CompatibilityLiteralLocation {
                rule_index,
                literal_index,
            });
            target_arities
                .entry(modal.atom.predicate.clone())
                .and_modify(|arity| {
                    debug_assert_eq!(*arity, modal.atom.arity());
                })
                .or_insert(modal.atom.arity());
        }
    }
    if locations.is_empty() {
        return Ok(None);
    }

    reject_nonmonotone_g91_compatibility_components(program, &locations)?;
    let snapshot_relations = g91_snapshot_relation_names(program, target_arities.keys());
    let upper_bound_program = transform_g91_compatibility_program(
        program,
        &locations,
        G91CompatibilityTransform::UpperBound,
    );
    let mut refinement_program = transform_g91_compatibility_program(
        program,
        &locations,
        G91CompatibilityTransform::Snapshot(&snapshot_relations),
    );
    add_declared_g91_snapshot_relations(
        &mut refinement_program,
        program,
        &snapshot_relations,
        &target_arities,
    );

    let convergence_predicates = program
        .proper_rules()
        .map(|rule| rule.head.predicate.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    Ok(Some(G91CompatibilityReduction {
        upper_bound_program,
        refinement_program,
        snapshot_relations,
        convergence_predicates,
    }))
}

fn is_g91_compatibility_literal(
    rule: &crate::ast::Rule,
    modal: &EpistemicLiteral,
    invariant: &InvariantRelations<'_>,
    recursive_modal_edges: &BTreeSet<(String, String)>,
) -> bool {
    modal.op == EpistemicOp::Possible
        && !modal.negated
        && !invariant.is_invariant(&modal.atom.predicate)
        && recursive_modal_edges
            .contains(&(rule.head.predicate.clone(), modal.atom.predicate.clone()))
        && modal.atom.terms == rule.head.terms
}

enum G91CompatibilityTransform<'a> {
    UpperBound,
    Snapshot(&'a BTreeMap<String, String>),
}

fn transform_g91_compatibility_program(
    program: &Program,
    locations: &BTreeSet<G91CompatibilityLiteralLocation>,
    transform: G91CompatibilityTransform<'_>,
) -> Program {
    let mut reduced = program.clone();
    for (rule_index, rule) in reduced.rules.iter_mut().enumerate() {
        for (literal_index, literal) in rule.body.iter_mut().enumerate() {
            let BodyLiteral::Epistemic(modal) = literal else {
                continue;
            };
            if locations.contains(&G91CompatibilityLiteralLocation {
                rule_index,
                literal_index,
            }) {
                *literal = match &transform {
                    G91CompatibilityTransform::UpperBound => BodyLiteral::Comparison(Comparison {
                        left: Term::Integer(1),
                        op: CompOp::Eq,
                        right: Term::Integer(1),
                    }),
                    G91CompatibilityTransform::Snapshot(snapshot_relations) => {
                        let mut atom = modal.atom.clone();
                        atom.predicate = snapshot_relations
                            .get(&atom.predicate)
                            .expect("selected compatibility target has a snapshot name")
                            .clone();
                        BodyLiteral::Positive(atom)
                    }
                };
            } else {
                *literal = if modal.negated {
                    BodyLiteral::Negated(modal.atom.clone())
                } else {
                    BodyLiteral::Positive(modal.atom.clone())
                };
            }
        }
    }
    reduced.constraints.retain(|constraint| {
        !constraint
            .body
            .iter()
            .any(|literal| matches!(literal, BodyLiteral::Epistemic(_)))
    });
    qualify_extensional_multi_arity_predicates(&mut reduced, program, &BTreeSet::new());
    reduced
}

fn g91_snapshot_relation_names<'a>(
    program: &Program,
    targets: impl Iterator<Item = &'a String>,
) -> BTreeMap<String, String> {
    let mut reserved = collect_epistemic_relation_identities(program, &BTreeSet::new())
        .0
        .into_keys()
        .collect::<BTreeSet<_>>();
    let mut names = BTreeMap::new();
    for target in targets {
        let stem = target
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() || character == '_' {
                    character
                } else {
                    '_'
                }
            })
            .collect::<String>();
        let base = format!("__xlog_g91_snapshot_{stem}");
        let mut candidate = base.clone();
        let mut suffix = 0usize;
        while reserved.contains(&candidate) {
            candidate = format!("{base}_{suffix}");
            suffix += 1;
        }
        reserved.insert(candidate.clone());
        names.insert(target.clone(), candidate);
    }
    names
}

fn add_declared_g91_snapshot_relations(
    refinement: &mut Program,
    source: &Program,
    snapshots: &BTreeMap<String, String>,
    target_arities: &BTreeMap<String, usize>,
) {
    for (target, snapshot) in snapshots {
        let expected_arity = target_arities
            .get(target)
            .expect("snapshot target has an authored arity");
        if let Some(declaration) = source.predicates.iter().find(|declaration| {
            declaration.name == *target && declaration.arity() == *expected_arity
        }) {
            let mut declaration = declaration.clone();
            declaration.name = snapshot.clone();
            declaration.is_private = false;
            refinement.predicates.push(declaration);
        }
    }
}

fn reject_nonmonotone_g91_compatibility_components(
    program: &Program,
    locations: &BTreeSet<G91CompatibilityLiteralLocation>,
) -> Result<()> {
    let (_, dependencies) = epistemic_dependency_graphs(program);
    let selected_heads = locations
        .iter()
        .map(|location| program.rules[location.rule_index].head.predicate.as_str())
        .collect::<BTreeSet<_>>();
    for rule in &program.rules {
        let in_selected_component = selected_heads.iter().any(|selected| {
            rule.head.predicate == **selected
                || (predicate_dependency_reaches(
                    selected,
                    &rule.head.predicate,
                    &dependencies,
                    &mut BTreeSet::new(),
                ) && predicate_dependency_reaches(
                    &rule.head.predicate,
                    selected,
                    &dependencies,
                    &mut BTreeSet::new(),
                ))
        });
        if !in_selected_component {
            continue;
        }
        if rule.has_aggregation() {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "Gelfond-1991 compatibility cycle through aggregation".to_string(),
                context: format!(
                    "aggregate predicate `{}` belongs to a positive `possible` compatibility \
                     component; the tuple-level greatest fixpoint requires every dependency in \
                     that component to be monotone",
                    rule.head.predicate
                ),
            });
        }
        if rule
            .body
            .iter()
            .filter_map(|literal| match literal {
                BodyLiteral::Negated(atom) => Some(atom),
                BodyLiteral::Epistemic(modal) if modal.negated => Some(&modal.atom),
                BodyLiteral::Positive(_)
                | BodyLiteral::Epistemic(_)
                | BodyLiteral::Comparison(_)
                | BodyLiteral::IsExpr(_)
                | BodyLiteral::Univ(_) => None,
            })
            .any(|atom| {
                predicate_dependency_reaches(
                    &atom.predicate,
                    &rule.head.predicate,
                    &dependencies,
                    &mut BTreeSet::new(),
                )
            })
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "Gelfond-1991 compatibility cycle through negation".to_string(),
                context: format!(
                    "predicate `{}` belongs to a positive `possible` compatibility component \
                     that also has a recursive negated dependency; the tuple-level greatest \
                     fixpoint requires a monotone component",
                    rule.head.predicate
                ),
            });
        }
    }
    Ok(())
}

/// Return the ordinary fixpoint reduction selected for a prepared epistemic program.
pub fn try_reduce_prepared_recursive_epistemic_program(
    prepared: &PreparedEpistemicProgram,
) -> Result<Option<Program>> {
    if try_prepare_g91_compatibility_reduction(prepared)?.is_some() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "Gelfond-1991 tuple compatibility ordinary reduction".to_string(),
            context: "positive `possible` compatibility cycles require the explicit upper-bound \
                      and frozen-snapshot greatest-fixpoint plan returned by \
                      `try_prepare_g91_compatibility_reduction`; they cannot be represented by \
                      one ordinary least-fixpoint program"
                .to_string(),
        });
    }
    let active_program = prepared.active_program();
    let recursive_class = classify_recursive_epistemic_program(active_program)?;
    if recursive_class == RecursiveEpistemicClass::NonRecursive
        && !prepared.removed_unfounded_rules()
    {
        return Ok(None);
    }

    validate_epistemic_derived_relation_identity(active_program, &BTreeSet::new())?;
    match recursive_class {
        RecursiveEpistemicClass::NonRecursive => Ok(Some(
            reduce_founded_epistemic_program_to_ordinary(active_program),
        )),
        // After explicit G91 compatibility cycles have been intercepted above, every
        // remaining admitted class shares the same reduction: each positive
        // `know`/`possible` modal resolves to its ordinary atom. That atom is either
        // invariant or co-evolves inside an ordinary-or-modal dependency cycle. The
        // semi-naive least fixpoint computes the founded co-evolving result.
        RecursiveEpistemicClass::CaseA
        | RecursiveEpistemicClass::CaseB
        | RecursiveEpistemicClass::ModalCycle => Ok(Some(
            reduce_case_a_epistemic_program_to_ordinary(active_program),
        )),
    }
}

/// Validate an admissible recursive epistemic program and return its ordinary
/// fixpoint reduction.
///
/// This is the recursive counterpart to [`compile_epistemic_gpu_execution`]. It first
/// validates the complete authored source, removes only exact tuple-level circular
/// FAEEL support, and classifies the remaining dependency graph. Predecessor and tuple-
/// permutation edges therefore remain part of recursive-path selection. Surviving
/// positive modal literals resolve to ordinary joins and execute through the existing
/// least-fixpoint engine. Exact G91 compatibility cycles return a typed error because
/// callers must execute the upper-bound/frozen-snapshot reduction returned by
/// [`try_prepare_g91_compatibility_reduction`].
///
/// Returns `Ok(Some(reduced))` for an admitted recursive class, `Ok(None)` when the
/// program has no dependency cycle (the caller should use the single-pass epistemic
/// path), and a typed error for a recursive shape outside the supported fragment.
pub fn try_reduce_case_a_recursive_epistemic_program(program: &Program) -> Result<Option<Program>> {
    let prepared = prepare_epistemic_program(program)?;
    try_reduce_prepared_recursive_epistemic_program(&prepared)
}

fn require_single_epistemic_output_relation(gpu_plan: &EpistemicGpuPlan) -> Result<()> {
    let output_relations: BTreeSet<&str> = gpu_plan
        .reductions
        .iter()
        .map(|reduction| reduction.head_predicate.as_str())
        .collect();
    if output_relations.len() > 1 {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "epistemic GPU final output relation".to_string(),
            context: format!(
                "single-plan GPU execution materializes one final output buffer, but reductions \
                 target multiple head predicates {:?}; use split GPU execution for independent \
                 epistemic outputs",
                output_relations
            ),
        });
    }
    Ok(())
}

fn reject_epistemic_constraints(program: &Program) -> Result<()> {
    reject_epistemic_constraints_for_boundary(program, "epistemic GPU constraint", "GPU lowering")
}

fn reject_gpt_epistemic_constraints(program: &Program) -> Result<()> {
    reject_epistemic_constraints_for_boundary(
        program,
        "epistemic GPT constraint",
        "GPT candidate testing",
    )
}

fn reject_epistemic_constraints_for_boundary(
    program: &Program,
    construct: &str,
    boundary: &str,
) -> Result<()> {
    for constraint in &program.constraints {
        let constraint_index = constraint.require_authored_index()?;
        for lit in &constraint.body {
            let BodyLiteral::Epistemic(lit) = lit else {
                continue;
            };
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: construct.to_string(),
                context: format!(
                    "constraint[{constraint_index}] contains unsupported {} {}/{}; epistemic integrity constraints must be represented explicitly before {boundary}",
                    epistemic_literal_label(lit),
                    lit.atom.predicate,
                    lit.atom.arity()
                ),
            });
        }
    }
    Ok(())
}

fn epistemic_literal_label(lit: &EpistemicLiteral) -> &'static str {
    match (lit.negated, lit.op) {
        (false, EpistemicOp::Know) => "know",
        (false, EpistemicOp::Possible) => "possible",
        (true, EpistemicOp::Know) => "not know",
        (true, EpistemicOp::Possible) => "not possible",
    }
}

/// Flatten a modal literal's structured key terms, returning a literal whose
/// atom carries the FLATTENED arity/terms.
///
/// This is the single normalization point for structured modal keys: the stored
/// epistemic literal, its tuple-membership binding, and its solver assumption
/// binding are all derived from the same flattened atom, so the plan validators
/// (which require `binding.arity == literal.atom.arity` and `binding.key_terms ==
/// literal.atom.terms`) stay consistent and the runtime matches the modal
/// relation's real column tuple. Scalar-only keys are returned unchanged.
fn flatten_epistemic_literal(lit: &EirEpistemicLiteral) -> Result<EirEpistemicLiteral> {
    let (arity, terms, _key_columns) =
        flatten_structured_key_terms(&lit.atom.predicate, &lit.atom.terms)?;
    Ok(EirEpistemicLiteral {
        op: lit.op,
        negated: lit.negated,
        atom: xlog_ir::EirAtom {
            predicate: lit.atom.predicate.clone(),
            arity,
            terms,
        },
    })
}

/// Whether a term encodes directly into one scalar/Symbol GPU key column.
///
/// These are the leaf forms the device tuple-key matcher already handles per
/// column: bound variables (BOUND_OUTPUT), anonymous wildcards (WILDCARD), and
/// ground integer/float/string/symbol literals (GROUND).
fn eir_term_is_scalar_key_element(term: &EirTerm) -> bool {
    matches!(
        term,
        EirTerm::Variable(_)
            | EirTerm::Anonymous
            | EirTerm::Integer(_)
            | EirTerm::FloatBits(_)
            | EirTerm::String(_)
            | EirTerm::Symbol(_)
    )
}

/// Flatten a modal atom's key terms ELEMENT-WISE into a flat list of scalar key
/// terms plus the matching `0..n` key-column indices.
///
/// A STRUCTURED finite+typed key term -- a fixed-arity list `[a, b]` or compound
/// `f(a, b)` whose elements are each scalar/Symbol-typed -- is expanded into its
/// elements, each of which becomes one GPU key column. The flattened arity must
/// equal the modal relation's arity (the runtime arity check enforces that
/// downstream). Scalar terms pass through unchanged.
///
/// Genuinely unbounded or untyped structured forms (a `cons` with a non-list
/// tail, a nested structure, a `predref`, or an `aggregate`) carry no fixed,
/// typed column set and are rejected as unsupported tuple-key constructs.
fn flatten_structured_key_terms(
    predicate: &str,
    terms: &[EirTerm],
) -> Result<(usize, Vec<EirTerm>, Vec<usize>)> {
    let mut flattened: Vec<EirTerm> = Vec::with_capacity(terms.len());
    for term in terms {
        match term {
            EirTerm::List(items) => {
                flatten_structured_elements(predicate, "list", items, &mut flattened)?;
            }
            EirTerm::Compound { functor, args } => {
                flatten_structured_elements(
                    predicate,
                    &format!("compound {functor}/{}", args.len()),
                    args,
                    &mut flattened,
                )?;
            }
            EirTerm::Cons { .. } => {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "modal tuple-key cons pattern".to_string(),
                    context: format!(
                        "modal tuple-key for {predicate} uses a `cons` pattern whose tail length \
                         is not statically fixed, so it has no finite, typed GPU key-column set; \
                         bind it to a fixed-arity list literal `[a, b, ...]` instead"
                    ),
                });
            }
            EirTerm::PredRef(name) => {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "modal tuple-key predicate reference".to_string(),
                    context: format!(
                        "modal tuple-key for {predicate} uses predref `{name}`, which has no \
                         finite, typed GPU key-column encoding"
                    ),
                });
            }
            EirTerm::Aggregate { op, variable } => {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "modal tuple-key aggregate".to_string(),
                    context: format!(
                        "modal tuple-key for {predicate} uses aggregate `{op}({variable})`, whose \
                         value is not a finite, typed GPU key-column tuple"
                    ),
                });
            }
            scalar => flattened.push(scalar.clone()),
        }
    }

    let arity = flattened.len();
    let key_columns = (0..arity).collect();
    Ok((arity, flattened, key_columns))
}

/// Splice the elements of a fixed-arity structured key term into `flattened`.
///
/// Each element must itself be a scalar/Symbol key element; a nested structure
/// would need a column to hold its own sub-tuple, which a flat relation schema
/// cannot express, so it is rejected with a precise unsupported-shape diagnostic.
fn flatten_structured_elements(
    predicate: &str,
    shape: &str,
    elements: &[EirTerm],
    flattened: &mut Vec<EirTerm>,
) -> Result<()> {
    for element in elements {
        if eir_term_is_scalar_key_element(element) {
            flattened.push(element.clone());
        } else {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "nested modal tuple-key structure".to_string(),
                context: format!(
                    "modal tuple-key for {predicate} nests a non-scalar element {element:?} inside \
                     a {shape}; only fixed-arity structures of scalar/Symbol-typed elements have a \
                     finite, typed GPU key-column encoding"
                ),
            });
        }
    }
    Ok(())
}

fn bound_output_columns_for_terms(
    key_terms: &[EirTerm],
    output_terms: &[EirTerm],
) -> Vec<Option<usize>> {
    key_terms
        .iter()
        .map(|term| match term {
            EirTerm::Variable(variable) => output_terms.iter().position(
                |head_term| matches!(head_term, EirTerm::Variable(name) if name == variable),
            ),
            _ => None,
        })
        .collect()
}

fn augmented_eir_head_terms(rule: &xlog_ir::EirRule) -> Vec<EirTerm> {
    let mut output_terms = rule.head.terms.clone();
    for lit in &rule.body {
        let EirBodyLiteral::Epistemic(lit) = lit else {
            continue;
        };
        // A modal key variable may be NESTED inside a structured key term
        // (`know p([X, Y])`), so flatten before collecting variables that need a
        // reduced output column to bind against. Flattening failures are surfaced
        // by the binding-construction path; here we fall back to the raw terms so
        // diagnostics remain anchored at that site.
        let key_terms = flatten_structured_key_terms(&lit.atom.predicate, &lit.atom.terms)
            .map(|(_, terms, _)| terms)
            .unwrap_or_else(|_| lit.atom.terms.clone());
        for term in &key_terms {
            let EirTerm::Variable(variable) = term else {
                continue;
            };
            if !output_terms
                .iter()
                .any(|head_term| matches!(head_term, EirTerm::Variable(name) if name == variable))
            {
                output_terms.push(EirTerm::Variable(variable.clone()));
            }
        }
    }
    output_terms
}

fn final_output_columns_for_eir(eir: &EirProgram) -> Option<Vec<usize>> {
    let mut final_columns = Vec::new();
    let mut needs_projection = false;
    for rule in &eir.rules {
        if !rule
            .body
            .iter()
            .any(|lit| matches!(lit, EirBodyLiteral::Epistemic(_)))
        {
            continue;
        }
        let augmented_len = augmented_eir_head_terms(rule).len();
        if augmented_len > rule.head.terms.len() {
            needs_projection = true;
        }
        if final_columns.is_empty() {
            final_columns = (0..rule.head.terms.len()).collect();
        }
    }
    if needs_projection {
        Some(final_columns)
    } else {
        None
    }
}

/// Indices (into `program.rules`) of exact tuple-level FAEEL rules that are unfounded
/// by circular modal self-support and must be excluded from the reduced founded-model
/// base without removing predecessor or tuple-permutation edges.
///
/// A rule qualifies when (a) the program is in FAEEL mode, (b) the rule body contains a
/// modal literal `possible p`/`know p` over the rule's head predicate and arity,
/// (c) that head has NO independent founded support
/// ([`has_independent_founded_support`]) and NO tuple-level founded support
/// ([`has_tuple_level_independent_founded_support`]), and (d) excluding the rule does
/// NOT silently elide a mode-independent safety failure — i.e. the head carries no
/// variable bound ONLY by the self-supporting modal. Condition (d) preserves the clean
/// `UnsafeVariable` honest-exit for pure nonzero self-support (`p(X) :- possible p(X)`)
/// in EVERY mode (G91 rejects it identically): dropping such a rule would replace a
/// precise safety diagnostic with a confusing materialization error.
/// Every schema census, shape check, stratified plan, and executable reduction uses
/// this same decision so no broader predicate-level approximation can erase live
/// support.
fn faeel_unfounded_exact_tuple_self_support_rule_indices(program: &Program) -> Vec<usize> {
    let Ok(eir) = build_eir(program) else {
        return Vec::new();
    };
    if eir.mode != EirEpistemicMode::Faeel {
        return Vec::new();
    }
    let mut indices = Vec::new();
    for (index, (rule, eir_rule)) in program.rules.iter().zip(&eir.rules).enumerate() {
        let modal_only_output_variables = modal_only_bound_output_variables(rule);
        let drop = eir_rule.body.iter().any(|lit| {
            let EirBodyLiteral::Epistemic(modal) = lit else {
                return false;
            };
            if modal.negated
                || modal.atom.predicate != eir_rule.head.predicate
                || modal.atom.arity != eir_rule.head.arity
                || modal.atom.terms != eir_rule.head.terms
            {
                return false;
            }
            // Founded by an independent (non-circular) derivation: keep the rule; the
            // founded support proves the head, so it stays in the model.
            if has_independent_founded_support(&eir, &modal.atom)
                || has_tuple_level_independent_founded_support(&eir, eir_rule, &modal.atom)
            {
                return false;
            }
            // A head variable bound ONLY by this self-supporting modal would be unbound
            // (`UnsafeVariable`) in every mode once the modal is stripped: do NOT drop,
            // let the existing safety path raise the precise diagnostic.
            if modal
                .atom
                .terms
                .iter()
                .any(|term| matches!(term, EirTerm::Variable(name) if modal_only_output_variables.contains(name)))
            {
                return false;
            }
            true
        });
        if drop {
            indices.push(index);
        }
    }
    indices
}

fn program_without_rule_indices(program: &Program, removed_rule_indices: &[usize]) -> Program {
    if removed_rule_indices.is_empty() {
        return program.clone();
    }

    let removed_rule_indices = removed_rule_indices
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut filtered = program.clone();
    filtered.rules = program
        .rules
        .iter()
        .enumerate()
        .filter(|(index, _)| !removed_rule_indices.contains(index))
        .map(|(_, rule)| rule.clone())
        .collect();
    filtered
}

/// Validate the complete authored epistemic program before foundedness can elide a rule.
///
/// Every authored predicate signature receives a temporary name-and-arity identity. This
/// keeps a semantically dead `p/1` clause from colliding with a live `p/2` relation while
/// preserving all declaration, clause, arithmetic, and modal type evidence within each
/// signature. Modal range restriction is checked against the executable contract first;
/// the validation clone then reaches the ordinary compiler's production preprocessing and
/// lowering checks without requiring ordinary stratification, because supported negated
/// modal cycles are dispatched to well-founded execution later.
pub fn validate_epistemic_source_program(program: &Program) -> Result<()> {
    let prepared = prepare_root_authored_constraint_identity(program)?;
    validate_prepared_epistemic_source_program(&prepared)
}

fn validate_prepared_epistemic_source_program(program: &Program) -> Result<()> {
    validate_authored_modal_key_shapes(program)?;
    let invariant = InvariantRelations::analyze(program);
    let determined = EpistemicallyDeterminedPredicates::analyze(program);
    for rule in &program.rules {
        validate_modal_variable_bindings(&rule.body, &invariant, &determined)?;
    }
    for constraint in &program.constraints {
        validate_modal_variable_bindings(&constraint.body, &invariant, &determined)?;
    }

    let mut validation = program.clone();
    for rule in &mut validation.rules {
        rewrite_modal_literals_for_source_validation(&mut rule.body, &invariant, &determined);
    }
    for constraint in &mut validation.constraints {
        rewrite_modal_literals_for_source_validation(&mut constraint.body, &invariant, &determined);
    }

    let multi_arity_predicates = all_multi_arity_predicates(program);
    qualify_predicate_signatures(&mut validation, &multi_arity_predicates);
    Compiler::new().validate_program_without_stratification(&validation)
}

/// Validate every authored modal tuple key against the relation signatures it can
/// address before foundedness or dependency reduction can remove the containing rule.
///
/// Structured keys are syntax for a flat tuple: `p([X, Y])` addresses `p/2`, not
/// `p/1`. The ordinary AST arity is therefore not the runtime key arity. Derive the
/// target signatures from declarations and non-modal relation occurrences, flatten
/// each EIR modal key through the production normalizer, and reject a mismatch at the
/// source boundary. This also preserves the precise finiteness diagnostic for an
/// unbounded structured key instead of allowing semantic elision to hide it.
fn validate_authored_modal_key_shapes(program: &Program) -> Result<()> {
    let target_arities = non_modal_relation_arities(program);
    let eir = build_eir(program)?;
    for modal in eir
        .rules
        .iter()
        .flat_map(|rule| &rule.body)
        .chain(
            eir.constraints
                .iter()
                .flat_map(|constraint| &constraint.body),
        )
        .filter_map(|literal| match literal {
            EirBodyLiteral::Epistemic(modal) => Some(modal),
            EirBodyLiteral::Relational { .. }
            | EirBodyLiteral::Constraint
            | EirBodyLiteral::Binding => None,
        })
    {
        let flattened = flatten_epistemic_literal(modal)?;
        let Some(expected) = target_arities.get(&flattened.atom.predicate) else {
            // An undeclared relation with no non-modal occurrence may be supplied as
            // an external relation. Its schema is established by the caller.
            continue;
        };
        if expected.contains(&flattened.atom.arity) {
            continue;
        }

        let expected_description = if expected.len() == 1 {
            format!(
                "target arity {}",
                expected.first().expect("one target arity")
            )
        } else {
            format!("target arities {expected:?}")
        };
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "epistemic modal tuple key".to_string(),
            context: format!(
                "modal target `{}` has {expected_description}, but its tuple key flattens to \
                 binding arity {}; use one scalar key term per target column",
                flattened.atom.predicate, flattened.atom.arity
            ),
        });
    }
    Ok(())
}

fn non_modal_relation_arities(program: &Program) -> BTreeMap<String, BTreeSet<usize>> {
    let mut arities = BTreeMap::new();
    for declaration in &program.predicates {
        arities
            .entry(declaration.name.clone())
            .or_insert_with(BTreeSet::new)
            .insert(declaration.arity());
    }
    for rule in &program.rules {
        record_predicate_signature(&mut arities, &rule.head);
        record_non_modal_body_signatures(&mut arities, &rule.body);
    }
    for constraint in &program.constraints {
        record_non_modal_body_signatures(&mut arities, &constraint.body);
    }
    for query in &program.queries {
        record_predicate_signature(&mut arities, &query.atom);
    }
    for fact in &program.prob_facts {
        record_predicate_signature(&mut arities, &fact.atom);
    }
    for disjunction in &program.annotated_disjunctions {
        for choice in &disjunction.choices {
            record_predicate_signature(&mut arities, &choice.atom);
        }
    }
    for evidence in &program.evidence {
        record_predicate_signature(&mut arities, &evidence.atom);
    }
    for query in &program.prob_queries {
        record_predicate_signature(&mut arities, &query.atom);
    }
    for declaration in &program.neural_predicates {
        record_predicate_signature(&mut arities, &declaration.predicate);
    }
    for rule in &program.learnable_rules {
        record_predicate_signature(&mut arities, &rule.head);
        record_non_modal_body_signatures(&mut arities, &rule.body);
    }
    arities
}

fn record_non_modal_body_signatures(
    signatures: &mut BTreeMap<String, BTreeSet<usize>>,
    body: &[BodyLiteral],
) {
    for literal in body {
        if let BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) = literal {
            record_predicate_signature(signatures, atom);
        }
    }
}

/// Replace modals for validation without changing the authored ordinary literal order.
/// Positive modals over invariant or acyclically determined targets remain binders.
/// Every other scalar-key modal is appended as an ordinary negated atom so it
/// contributes schema and type evidence while the compiler independently verifies that
/// all of its variables already have a finite source.
fn rewrite_modal_literals_for_source_validation(
    body: &mut Vec<BodyLiteral>,
    invariant: &InvariantRelations<'_>,
    determined: &EpistemicallyDeterminedPredicates,
) {
    let mut non_binding_modals = Vec::new();
    body.retain_mut(|literal| {
        let BodyLiteral::Epistemic(modal) = literal else {
            return true;
        };
        if modal.atom.terms.iter().any(|term| {
            !matches!(
                term,
                Term::Variable(_)
                    | Term::Anonymous
                    | Term::Integer(_)
                    | Term::Float(_)
                    | Term::String(_)
                    | Term::Symbol(_)
            )
        }) {
            // Structured modal keys have a dedicated finite-key normalization and
            // diagnostic path. Ordinary list lowering cannot represent an unbounded
            // `cons` key and would preempt that precise epistemic diagnostic.
            *literal = BodyLiteral::Comparison(Comparison {
                left: Term::Integer(1),
                op: CompOp::Eq,
                right: Term::Integer(1),
            });
            return true;
        }
        if !modal.negated
            && (invariant.is_invariant(&modal.atom.predicate)
                || determined.contains(&modal.atom.predicate))
        {
            *literal = BodyLiteral::Positive(modal.atom.clone());
            true
        } else {
            non_binding_modals.push(BodyLiteral::Negated(modal.atom.clone()));
            false
        }
    });
    body.extend(non_binding_modals);
}

/// Variables with an ordinary finite source, matching the Lowerer's binding order:
/// every positive atom is joined first, then deterministic `is` expressions are applied
/// once in source order. A reversed arithmetic dependency is therefore not accepted by a
/// fixed-point approximation.
fn non_epistemic_bound_variables(body: &[BodyLiteral]) -> BTreeSet<String> {
    let mut bound = positive_body_bound_variables(body);

    for literal in body {
        let BodyLiteral::IsExpr(binding) = literal else {
            continue;
        };
        if binding
            .expr
            .variables()
            .iter()
            .all(|name| bound.contains(*name))
        {
            bound.insert(binding.target.clone());
        }
    }

    bound
}

/// Enforce the modal binding contract before a reduction turns modal atoms into ordinary
/// joins. A co-evolving or negated modal may filter an already-bound tuple but may not
/// invent a finite domain; only a positive modal over an invariant or acyclically
/// determined relation can bind.
fn validate_modal_variable_bindings(
    body: &[BodyLiteral],
    invariant: &InvariantRelations<'_>,
    determined: &EpistemicallyDeterminedPredicates,
) -> Result<()> {
    let mut bound = non_epistemic_bound_variables(body);

    // A rule body is a conjunction, so positive finite modal sources bind
    // independently of their textual order. Collect every such binder before
    // checking co-evolving or negated modal filters; only deterministic `is`
    // expressions retain the Lowerer's source-order contract.
    for literal in body {
        let BodyLiteral::Epistemic(modal) = literal else {
            continue;
        };
        let may_bind = !modal.negated
            && (invariant.is_invariant(&modal.atom.predicate)
                || determined.contains(&modal.atom.predicate));
        if may_bind {
            bound.extend(
                modal
                    .atom
                    .variables()
                    .into_iter()
                    .filter(|name| *name != "_")
                    .map(str::to_string),
            );
        }
    }

    for literal in body {
        let BodyLiteral::Epistemic(modal) = literal else {
            continue;
        };
        let may_bind = !modal.negated
            && (invariant.is_invariant(&modal.atom.predicate)
                || determined.contains(&modal.atom.predicate));
        for variable in modal.atom.variables() {
            if variable == "_" {
                continue;
            }
            if !may_bind && !bound.contains(variable) {
                return Err(XlogError::UnsafeVariable(variable.to_string()));
            }
        }
    }
    Ok(())
}

/// Return the ordinary runtime program selected by epistemic dependency
/// classification.
///
/// Admissible ordinary or modal dependency cycles resolve their modal edges into an
/// ordinary fixpoint program. Acyclic programs use the single-pass reduction; callers
/// must validate its explicit epistemic GPU contract before execution.
///
/// The augmenting positive-modal resolve is gated on INVARIANT targets only (see the
/// body comment): for an invariant `R`, `know R`/`possible R` ranges exactly over
/// `R`'s extension, so resolving the modal into an ordinary join binds the augmented
/// output column WITHOUT leaking — and the GPU membership filter re-gates
/// post hoc. A determined-but-not-invariant target (an epistemic-derived head like a
/// multi-column `r`) is NOT resolved here, so its augmenting output variable stays
/// unbound and the reduced program fails closed at this strict (execution) entry
/// point. See [`reduce_epistemic_program_to_ordinary_for_stratified_schema`] for the
/// schema-only relaxation used by the stratified driver.
pub fn reduce_epistemic_program_to_ordinary(program: &Program) -> Result<Program> {
    let prepared = prepare_root_authored_constraint_identity(program)?;
    if let Some(reduced) = try_reduce_case_a_recursive_epistemic_program(&prepared)? {
        return Ok(reduced);
    }
    reduce_epistemic_program_to_ordinary_inner(&prepared, &BTreeSet::new(), &BTreeMap::new())
}

/// Schema-only reduction for the stratified epistemic driver.
///
/// Identical to [`reduce_epistemic_program_to_ordinary`] EXCEPT it also resolves an
/// augmenting positive modal whose target is epistemically DETERMINED (as classified
/// by the internal determined-predicate analysis) but not invariant — e.g. a
/// multi-column determined head `r` in `out(X) :- node(X), know r(X, Y)`, where the
/// modal binds the augmented output column `Y`. This is used SOLELY to compute the
/// plan-wide relation SCHEMAS (column types/arities) for an
/// [`EpistemicStratifiedPlan`]; the resolved positive atom over `r` supplies
/// `Y`'s declared column type so the schema compiler does not reject the augmented
/// `out(X, Y)` head as unsafe.
///
/// SOUNDNESS / NON-LEAK: a determined `r` IS gated into the store as a materialized
/// base relation by the LOWER stratum before the higher stratum runs (the stratified
/// executor's `materialize_epistemic_head_relation` at the STORE boundary), and the
/// higher stratum is compiled by `compile_stratum_plan` over a sub-program where
/// `r`'s defining rule is DROPPED — so there `r` is invariant and the EXISTING strict
/// resolve binds `Y` against the GATED `r` for execution. The determined-relaxed
/// resolve here therefore NEVER drives runtime data: it only types columns. It is not
/// used by the single/joint or Case-A EXECUTION reduce, so it cannot resolve a modal
/// into a join over an UN-gated candidate relation.
pub fn reduce_epistemic_program_to_ordinary_for_stratified_schema(
    program: &Program,
) -> Result<Program> {
    let prepared = prepare_root_authored_constraint_identity(program)?;
    let determined = EpistemicallyDeterminedPredicates::analyze(&prepared);
    let path_specific_rules = stratified_schema_reduction_overrides(&prepared)?;
    reduce_epistemic_program_to_ordinary_inner(
        &prepared,
        &determined.determined,
        &path_specific_rules,
    )
}

fn prepare_root_authored_constraint_identity(program: &Program) -> Result<Program> {
    let mut prepared = program.clone();
    if prepared.authored_constraint_source_bound.is_some() {
        prepared.validate_prepared_authored_constraint_identity()?;
    } else {
        prepared.prepare_authored_constraint_identity_at_root()?;
    }
    Ok(prepared)
}

/// Shared body of the epistemic-to-ordinary reduction.
///
/// `schema_only_determined_resolve` names predicates that are epistemically
/// DETERMINED and whose augmenting positive modal may additionally be resolved into a
/// positive ordinary atom for SCHEMA inference only (empty for the strict execution
/// reduce). The INVARIANT-target resolve is always active for both entry points.
fn reduce_epistemic_program_to_ordinary_inner(
    program: &Program,
    schema_only_determined_resolve: &BTreeSet<String>,
    path_specific_rules: &BTreeMap<usize, crate::ast::Rule>,
) -> Result<Program> {
    let path_specific_rule_indices = path_specific_rules.keys().copied().collect::<BTreeSet<_>>();
    validate_epistemic_relation_shapes(program, &path_specific_rule_indices)?;

    // FAEEL FOUNDED-MODEL EXTENSION: a rule whose head is supported ONLY by circular
    // modal self-support (`possible p`/`know p` over its own head, with no independent
    // founded derivation) contributes nothing to the FAEEL founded model. Excluding the
    // rule from the reduced ordinary base is precisely the founded/equilibrium
    // semantics: the unfounded head is absent from the model rather than fabricated by
    // the stripped-modal `1=1` filler (which would wrongly found it, the G91 answer).
    //
    // This is the structural foundedness DECISION (compile-time, reusing the exact
    // `has_independent_founded_support` / `has_tuple_level_independent_founded_support`
    // structural support predicates) driving the EXTENSION COMPUTATION on the
    // GPU/runtime path: the dropped rule simply removes the unfounded head's founding
    // base, and the existing GPU world-view validation then accepts the empty/founded
    // candidate. G91 keeps the filler (no drop), so `possible p` stays accepted —
    // this drop IS the FAEEL-vs-G91 mode difference.
    //
    // SCOPE: the drop fires only for FAEEL mode. A rule whose head carries a variable
    // bound ONLY by the self-supporting modal is NOT dropped here; with the modal
    // stripped that variable is genuinely unbound (`UnsafeVariable`) in EVERY mode
    // (G91 included), so it must fall through to the existing safety path rather than
    // be silently elided. Dropping it would mask a mode-independent safety failure.
    let removed_rule_indices = faeel_unfounded_exact_tuple_self_support_rule_indices(program);
    let removed_rule_index_set = removed_rule_indices
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let active_original_rule_indices = (0..program.rules.len())
        .filter(|index| !removed_rule_index_set.contains(index))
        .collect::<Vec<_>>();
    let mut reduced = program_without_rule_indices(program, &removed_rule_indices);

    // AUGMENTING positive modals over INVARIANT relations are resolved into positive
    // ordinary join atoms (instead of being stripped) so the augmented head columns
    // they introduce are range-restricted in the reduced ordinary candidate program.
    //
    // An AUGMENTING modal carries a variable that is appended to the head by
    // `append_body_local_tuple_key_variables_to_head` (a modal-local variable absent
    // from the user-visible head, e.g. `Y` in `one_hop(X) :- node(X), know edge(X,
    // Y)`). After the modal is stripped, that augmented `Y` column has no binding, so
    // the reduced rule would be unsafe (`UnsafeVariable`). Resolving the positive
    // modal over its (invariant) gated relation into a positive ordinary atom binds
    // the column. This mirrors the proven-sound Case-A invariant resolution
    // (`reduce_case_a_epistemic_program_to_ordinary`): for an INVARIANT relation `R`,
    // `know R`/`possible R` ranges exactly over `R`'s extension, so the reduced
    // candidate join over `R` enumerates the correct augmented tuples and the GPU
    // membership filter then re-gates them against the accepted world view.
    //
    // STRICTLY SCOPED to keep the prohibition on resolving over still-modal relations
    // machine-checked: only POSITIVE modals (negated `not know`/`not possible` is an
    // anti-join that does NOT range-restrict, so it is never resolved) over INVARIANT
    // targets (a still-modal / epistemic-derived target is NOT invariant, so it is
    // never resolved — its augmenting variable stays unbound and the reduced program
    // fails closed). Non-augmenting modals keep the existing single- and joint-solver
    // strip-and-gate path.
    let invariant = InvariantRelations::analyze(program);

    // Every rule whose internal head gains tuple-key columns must use a widened
    // declaration. The recursive epistemic path has its own non-augmenting reducer;
    // this single-pass reduction records the actual head transformation, including
    // columns that an ordinary atom already binds. Rule indices retain the exact
    // source signature because the head arity changes before reconciliation.
    let mut augmented_rule_original_arities = BTreeMap::new();

    for ((rule_index, rule), original_rule_index) in reduced
        .rules
        .iter_mut()
        .enumerate()
        .zip(active_original_rule_indices)
    {
        if let Some(path_specific_rule) = path_specific_rules.get(&original_rule_index) {
            *rule = path_specific_rule.clone();
            continue;
        }
        let original_head_arity = rule.head.arity();
        // Head variables that NO non-epistemic positive body literal binds. After the
        // modal is stripped, an output (head) variable bound ONLY by the modal would
        // be unsafe in the reduced ordinary program. Computed BEFORE the head is
        // mutated by augmentation. (`append_body_local_tuple_key_variables_to_head`
        // appends modal-local variables to the head, so both already-present head
        // variables like `Y` in `pair(X,Y) :- ..possible edge(X,Y)` AND augmented
        // variables like `Y` in `one_hop(X) :- ..know edge(X,Y)` are covered here.)
        let modal_only_output_variables = modal_only_bound_output_variables(rule);
        append_body_local_tuple_key_variables_to_head(rule);
        if rule.head.arity() > original_head_arity {
            augmented_rule_original_arities.insert(rule_index, original_head_arity);
        }
        let was_fact = rule.body.is_empty();
        let had_epistemic_body = rule
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)));
        // Resolve a POSITIVE modal over an INVARIANT relation into a positive ordinary
        // join atom WHEN it is the sole binder of some output variable (so that output
        // variable is range-restricted in the reduced candidate program); strip every
        // other modal. For an invariant relation `R`, `know R`/`possible R` ranges
        // exactly over `R`'s extension, so the reduced join enumerates the correct
        // candidate tuples and the GPU filter re-gates against the accepted
        // world view. A NEGATED modal (anti-join) never binds and is never resolved; a
        // still-modal / epistemic-derived target is NOT invariant and is never
        // resolved, so its unbound output variable correctly fails closed downstream.
        for lit in &mut rule.body {
            if let BodyLiteral::Epistemic(modal) = lit {
                // The target is resolvable when it is INVARIANT (always — proven-sound
                // for both schema and execution), OR — for SCHEMA inference only — when
                // it is epistemically DETERMINED. The determined relaxation is empty for
                // the strict execution reduce, so an execution-path reduce never
                // resolves a modal over a still-derived (un-gated) relation.
                if resolves_augmented_head_variable(
                    modal,
                    &modal_only_output_variables,
                    &invariant,
                    schema_only_determined_resolve,
                ) {
                    *lit = BodyLiteral::Positive(modal.atom.clone());
                }
            }
        }
        rule.body
            .retain(|lit| !matches!(lit, BodyLiteral::Epistemic(_)));
        if !was_fact && had_epistemic_body && rule.body.is_empty() {
            rule.body.push(BodyLiteral::Comparison(Comparison {
                left: Term::Integer(1),
                op: CompOp::Eq,
                right: Term::Integer(1),
            }));
        }
    }
    // Head augmentation appends modal-local columns to a genuinely-augmented rule head
    // (e.g. `one_hop(X)` becomes `one_hop(X, Y)`), so the reduced relation carries the
    // augmented columns needed for the GPU tuple-key membership gate. The predicate
    // DECLARATION must be widened to the augmented arity, or the runtime would union
    // the augmented rule output against the narrow declared (empty) stub and fail with
    // a schema mismatch. Infer each appended column's type from the positive body
    // atom that binds it; modal-only columns use the resolved invariant atom.
    qualify_extensional_multi_arity_predicates(&mut reduced, program, &removed_rule_index_set);

    let augmented_signatures =
        reconcile_augmented_head_declarations(&mut reduced, &augmented_rule_original_arities)?;

    // Drop reduced-program queries that reference an AUGMENTED head: the reduced
    // relation is now arity-bumped, so an original arity-N query over it would union
    // the arity-N query projection against the augmented relation and fail with a
    // schema mismatch. The user-visible query results for epistemic heads are
    // surfaced separately from the GPU gated buffers (`epistemic_result_to_query_
    // results`, projected to public arity), and the surfacing gate
    // (`queried_predicates`) reads the ORIGINAL program's queries, so dropping the
    // redundant reduced query here is inert for display and only removes the crash.
    // Non-augmented epistemic heads keep their arity-matched reduced queries untouched.
    if !augmented_signatures.is_empty() {
        reduced.queries.retain(|query| {
            !augmented_signatures.contains_key(&(query.atom.predicate.clone(), query.atom.arity()))
        });
    }

    // Constraints that contain epistemic literals are world-view integrity
    // constraints: they constrain accepted candidate world views and are
    // evaluated by the GPU world-view constraint kernel, NOT by the reduced
    // ordinary runtime. Stripping their epistemic literals would leave an
    // always-true ordinary constraint, so drop them from the reduced program
    // entirely. Purely relational constraints stay as ordinary constraints.
    reduced.constraints.retain(|constraint| {
        !constraint
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
    });

    Ok(reduced)
}

/// Reduce an admitted recursive epistemic program to an ordinary program for the
/// existing fixpoint engine.
///
/// Unlike [`reduce_epistemic_program_to_ordinary`] (which strips modal literals and
/// gates the single-pass result post hoc), this RESOLVES each positive `know`/
/// `possible` literal to its gated relation by rewriting it into an ordinary positive
/// body atom over the same predicate. An invariant modal target becomes a fixed join;
/// a co-evolving FAEEL target becomes a recursive join whose least fixpoint is its
/// founded extension. Gelfond-1991 compatibility cycles are intercepted by
/// [`try_prepare_g91_compatibility_reduction`] and require their explicit descending
/// tuple fixpoint; this ordinary reducer never deletes those gates. Modal variables
/// become ordinary join variables, so tuple transitions remain inside the fixpoint
/// instead of being approximated by a post-hoc single-pass gate.
///
/// Callers MUST first admit the program through
/// [`classify_recursive_epistemic_program`]; this function assumes that contract for
/// every supported recursive class.
pub fn reduce_case_a_epistemic_program_to_ordinary(program: &Program) -> Program {
    let mut reduced = program.clone();
    for rule in &mut reduced.rules {
        resolve_recursive_epistemic_rule_modals(rule);
    }
    // World-view integrity constraints have no place in this ordinary recursive
    // program: the recursion already joins against the resolved relations. Drop any constraint that
    // still references a modal literal (purely relational constraints are retained).
    reduced.constraints.retain(|constraint| {
        !constraint
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
    });
    qualify_extensional_multi_arity_predicates(&mut reduced, program, &BTreeSet::new());
    reduced
}

/// Reduce the surviving acyclic portion of a FAEEL program after exact unfounded
/// self-support has been removed.
///
/// The removed cycle established that this source entered the founded recursive
/// route, but the remaining rules no longer need iteration through a modal edge.
/// Resolving their modal literals to ordinary atoms preserves the now-determined
/// founded extension and, unlike the single-pass candidate reducer, keeps every
/// surviving gate load-bearing. Modal integrity constraints are resolved as ordinary
/// constraints because the surviving program has a single determined model.
fn reduce_founded_epistemic_program_to_ordinary(program: &Program) -> Program {
    let mut reduced = program.clone();
    for rule in &mut reduced.rules {
        resolve_recursive_epistemic_rule_modals(rule);
    }
    for constraint in &mut reduced.constraints {
        for literal in &mut constraint.body {
            let BodyLiteral::Epistemic(modal) = literal else {
                continue;
            };
            *literal = if modal.negated {
                BodyLiteral::Negated(modal.atom.clone())
            } else {
                BodyLiteral::Positive(modal.atom.clone())
            };
        }
    }
    qualify_extensional_multi_arity_predicates(&mut reduced, program, &BTreeSet::new());
    reduced
}

fn resolve_recursive_epistemic_rule_modals(rule: &mut crate::ast::Rule) {
    for literal in &mut rule.body {
        if let BodyLiteral::Epistemic(modal) = literal {
            *literal = if modal.negated {
                BodyLiteral::Negated(modal.atom.clone())
            } else {
                BodyLiteral::Positive(modal.atom.clone())
            };
        }
    }
}

/// Output (head) variables of `rule` that are bound ONLY by epistemic literals, i.e.
/// no positive non-epistemic body literal binds them.
///
/// Includes BOTH variables already in the user-visible head (e.g. `Y` in
/// `pair(X,Y) :- color(X), possible edge(X,Y)`) AND modal-local variables that
/// augmentation will append to the head (e.g. `Y` in
/// `one_hop(X) :- node(X), know edge(X,Y)`). After the modal is stripped, each such
/// variable would be an unsafe head column unless a positive-invariant modal carrying
/// it is resolved into a positive ordinary atom. Computed from the ORIGINAL rule,
/// before the head is mutated by augmentation.
fn modal_only_bound_output_variables(rule: &crate::ast::Rule) -> BTreeSet<String> {
    let positively_bound = non_epistemic_bound_variables(&rule.body);

    // Candidate output variables: every variable occurring in the user-visible head
    // plus every modal-local variable (which augmentation will append to the head).
    let mut modal_only = BTreeSet::new();
    let mut consider = |name: &str| {
        if name != "_" && !positively_bound.contains(name) {
            modal_only.insert(name.to_string());
        }
    };
    for term in &rule.head.terms {
        if let Term::Variable(name) = term {
            consider(name);
        }
    }
    for lit in &rule.body {
        if let BodyLiteral::Epistemic(lit) = lit {
            for term in &lit.atom.terms {
                if let Term::Variable(name) = term {
                    consider(name);
                }
            }
        }
    }
    modal_only
}

/// Whether `modal`'s atom carries at least one output variable that no positive
/// non-epistemic body literal binds (so resolving this positive-invariant modal into a
/// positive ordinary atom range-restricts an otherwise-unbound head column).
fn modal_atom_binds_output_variable(
    modal: &EpistemicLiteral,
    modal_only_output_variables: &BTreeSet<String>,
) -> bool {
    modal.atom.terms.iter().any(
        |term| matches!(term, Term::Variable(name) if modal_only_output_variables.contains(name)),
    )
}

fn resolves_augmented_head_variable(
    modal: &EpistemicLiteral,
    modal_only_output_variables: &BTreeSet<String>,
    invariant: &InvariantRelations,
    schema_only_determined_resolve: &BTreeSet<String>,
) -> bool {
    !modal.negated
        && (invariant.is_invariant(&modal.atom.predicate)
            || schema_only_determined_resolve.contains(&modal.atom.predicate))
        && modal_atom_binds_output_variable(modal, modal_only_output_variables)
}

fn record_predicate_signature(
    signatures: &mut BTreeMap<String, BTreeSet<usize>>,
    atom: &crate::ast::Atom,
) {
    signatures
        .entry(atom.predicate.clone())
        .or_default()
        .insert(atom.arity());
}

fn record_body_predicate_signatures(
    signatures: &mut BTreeMap<String, BTreeSet<usize>>,
    body: &[BodyLiteral],
) {
    for literal in body {
        match literal {
            BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                record_predicate_signature(signatures, atom);
            }
            BodyLiteral::Epistemic(modal) => {
                let eir_terms = modal
                    .atom
                    .terms
                    .iter()
                    .map(convert_term)
                    .collect::<Vec<_>>();
                // Structured-key validation owns the typed error for unsupported
                // shapes. Keep this identity census total for reducers that inspect
                // the source before lowering; valid keys use the exact same
                // production flattener as planning, while an invalid key retains its
                // source arity until validation rejects it.
                let arity = flatten_structured_key_terms(&modal.atom.predicate, &eir_terms)
                    .map(|(arity, _, _)| arity)
                    .unwrap_or_else(|_| modal.atom.arity());
                signatures
                    .entry(modal.atom.predicate.clone())
                    .or_default()
                    .insert(arity);
            }
            BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {}
        }
    }
}

fn collect_epistemic_relation_identities(
    program: &Program,
    removed_rules: &BTreeSet<usize>,
) -> (BTreeMap<String, BTreeSet<usize>>, BTreeSet<String>) {
    let mut source_arities: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for declaration in &program.predicates {
        source_arities
            .entry(declaration.name.clone())
            .or_default()
            .insert(declaration.arity());
    }

    let mut derived_predicates = BTreeSet::new();
    for (index, rule) in program.rules.iter().enumerate() {
        if removed_rules.contains(&index) {
            continue;
        }
        record_predicate_signature(&mut source_arities, &rule.head);
        record_body_predicate_signatures(&mut source_arities, &rule.body);
        if !rule.body.is_empty() {
            derived_predicates.insert(rule.head.predicate.clone());
        }
    }
    for constraint in &program.constraints {
        record_body_predicate_signatures(&mut source_arities, &constraint.body);
    }
    for query in &program.queries {
        record_predicate_signature(&mut source_arities, &query.atom);
    }
    for fact in &program.prob_facts {
        record_predicate_signature(&mut source_arities, &fact.atom);
    }
    for disjunction in &program.annotated_disjunctions {
        for choice in &disjunction.choices {
            record_predicate_signature(&mut source_arities, &choice.atom);
        }
    }
    for evidence in &program.evidence {
        record_predicate_signature(&mut source_arities, &evidence.atom);
    }
    for query in &program.prob_queries {
        record_predicate_signature(&mut source_arities, &query.atom);
    }
    for declaration in &program.neural_predicates {
        record_predicate_signature(&mut source_arities, &declaration.predicate);
    }
    for rule in &program.learnable_rules {
        record_predicate_signature(&mut source_arities, &rule.head);
        record_body_predicate_signatures(&mut source_arities, &rule.body);
        if !rule.body.is_empty() {
            derived_predicates.insert(rule.head.predicate.clone());
        }
    }

    (source_arities, derived_predicates)
}

fn all_multi_arity_predicates(program: &Program) -> BTreeSet<String> {
    collect_epistemic_relation_identities(program, &BTreeSet::new())
        .0
        .into_iter()
        .filter_map(|(predicate, arities)| (arities.len() > 1).then_some(predicate))
        .collect()
}

fn qualify_atom_for_extensional_multi_arity(
    atom: &mut crate::ast::Atom,
    predicates: &BTreeSet<String>,
) {
    if predicates.contains(&atom.predicate) {
        atom.predicate = format!("{}/{}", atom.predicate, atom.arity());
    }
}

fn qualify_body_for_extensional_multi_arity(
    body: &mut [BodyLiteral],
    predicates: &BTreeSet<String>,
) {
    for literal in body {
        match literal {
            BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                qualify_atom_for_extensional_multi_arity(atom, predicates);
            }
            BodyLiteral::Epistemic(modal) => {
                qualify_atom_for_extensional_multi_arity(&mut modal.atom, predicates);
            }
            BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {}
        }
    }
}

/// Return extensional predicate names whose runtime identity must include arity.
///
/// The census covers every source surface that can name a relation, including
/// constraints, probabilistic constructs, neural declarations, and learnable
/// rules. FAEEL rules removed as unfounded support do not make a predicate derived
/// or add a live signature. Reducers and fact upload must use this same set so a
/// source fact and its compiled scan always receive the same runtime name.
pub fn epistemic_extensional_multi_arity_predicates(program: &Program) -> BTreeSet<String> {
    let removed_rules = faeel_unfounded_exact_tuple_self_support_rule_indices(program)
        .into_iter()
        .collect::<BTreeSet<_>>();
    extensional_multi_arity_predicates(program, &removed_rules)
}

fn extensional_multi_arity_predicates(
    program: &Program,
    removed_rules: &BTreeSet<usize>,
) -> BTreeSet<String> {
    let (source_arities, derived_predicates) =
        collect_epistemic_relation_identities(program, removed_rules);
    source_arities
        .into_iter()
        .filter_map(|(predicate, arities)| {
            (arities.len() > 1 && !derived_predicates.contains(&predicate)).then_some(predicate)
        })
        .collect()
}

/// Give each extensional source signature the same arity-qualified runtime name
/// used by the GPU fact loader.
///
/// This transformation is limited to predicates with no active defining rule.
/// Derived predicates are validated separately because their public output and
/// recursive relation identity remain name-keyed.
fn qualify_extensional_multi_arity_predicates(
    reduced: &mut Program,
    source: &Program,
    removed_rules: &BTreeSet<usize>,
) {
    let predicates = extensional_multi_arity_predicates(source, removed_rules);
    qualify_predicate_signatures(reduced, &predicates);
}

/// Apply canonical name-and-arity identities to every AST surface naming one of
/// `predicates`. Runtime reduction calls this only for extensional multi-arity names;
/// source validation calls it for every multi-arity name so each authored signature
/// retains its own declarations and clauses while being checked.
fn qualify_predicate_signatures(reduced: &mut Program, predicates: &BTreeSet<String>) {
    if predicates.is_empty() {
        return;
    }

    for declaration in &mut reduced.predicates {
        if predicates.contains(&declaration.name) {
            declaration.name = format!("{}/{}", declaration.name, declaration.arity());
        }
    }
    for rule in &mut reduced.rules {
        qualify_atom_for_extensional_multi_arity(&mut rule.head, predicates);
        qualify_body_for_extensional_multi_arity(&mut rule.body, predicates);
    }
    for constraint in &mut reduced.constraints {
        qualify_body_for_extensional_multi_arity(&mut constraint.body, predicates);
    }
    for query in &mut reduced.queries {
        qualify_atom_for_extensional_multi_arity(&mut query.atom, predicates);
    }
    for fact in &mut reduced.prob_facts {
        qualify_atom_for_extensional_multi_arity(&mut fact.atom, predicates);
    }
    for disjunction in &mut reduced.annotated_disjunctions {
        for choice in &mut disjunction.choices {
            qualify_atom_for_extensional_multi_arity(&mut choice.atom, predicates);
        }
    }
    for evidence in &mut reduced.evidence {
        qualify_atom_for_extensional_multi_arity(&mut evidence.atom, predicates);
    }
    for query in &mut reduced.prob_queries {
        qualify_atom_for_extensional_multi_arity(&mut query.atom, predicates);
    }
    for declaration in &mut reduced.neural_predicates {
        qualify_atom_for_extensional_multi_arity(&mut declaration.predicate, predicates);
    }
    for rule in &mut reduced.learnable_rules {
        qualify_atom_for_extensional_multi_arity(&mut rule.head, predicates);
        qualify_body_for_extensional_multi_arity(&mut rule.body, predicates);
    }
}

/// Validate the name-keyed runtime identity used for derived epistemic
/// relations and return every authored predicate signature.
///
/// Pure extensional predicates may use the same name at multiple arities because
/// their reduced runtime identities are arity-qualified. Once a predicate is
/// derived, however, the ordinary compiler and output materializer assign one
/// relation identity to its name. Every occurrence is included here so a query,
/// body atom, declaration, or auxiliary probabilistic construct cannot alias a
/// derived relation at a different arity.
fn validate_epistemic_derived_relation_identity(
    program: &Program,
    removed_rules: &BTreeSet<usize>,
) -> Result<BTreeMap<String, BTreeSet<usize>>> {
    let (source_arities, derived_predicates) =
        collect_epistemic_relation_identities(program, removed_rules);
    for predicate in derived_predicates {
        let arities = source_arities
            .get(&predicate)
            .expect("derived predicate has a source signature");
        if arities.len() > 1 {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic derived predicate schema".to_string(),
                context: format!(
                    "derived predicate `{predicate}` uses multiple source arities {arities:?}; \
                     epistemic derived relations require one source signature per predicate name"
                ),
            });
        }
    }

    Ok(source_arities)
}

/// Ensure every clause for an augmented predicate signature lowers to one internal
/// relation arity.
///
/// A modal-local output variable adds hidden tuple-key columns to a rule head. If a
/// sibling clause for the same original signature produces a different number of
/// columns, the clauses cannot be unioned into one relation without inventing values
/// that the shorter clause does not bind. Reject that unsupported shape before any
/// reduced program reaches schema inference.
fn validate_epistemic_relation_shapes(
    program: &Program,
    non_augmenting_rule_indices: &BTreeSet<usize>,
) -> Result<()> {
    let removed_rules = faeel_unfounded_exact_tuple_self_support_rule_indices(program)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let active_rules = program
        .rules
        .iter()
        .enumerate()
        .filter(|(index, _)| {
            !removed_rules.contains(index) && !non_augmenting_rule_indices.contains(index)
        })
        .collect::<Vec<_>>();

    validate_epistemic_derived_relation_identity(program, &removed_rules)?;

    let mut reduced_rule_arities = Vec::with_capacity(active_rules.len());
    let mut augmented_targets: BTreeMap<(String, usize), usize> = BTreeMap::new();

    for (_, rule) in &active_rules {
        let original_signature = (rule.head.predicate.clone(), rule.head.arity());
        let mut reduced_rule = (*rule).clone();
        append_body_local_tuple_key_variables_to_head(&mut reduced_rule);
        let reduced_arity = reduced_rule.head.arity();

        if reduced_arity > original_signature.1 {
            augmented_targets
                .entry(original_signature.clone())
                .and_modify(|target| *target = (*target).max(reduced_arity))
                .or_insert(reduced_arity);
        }
        reduced_rule_arities.push((original_signature, reduced_arity));
    }

    for ((predicate, original_arity), target_arity) in augmented_targets {
        let arities = reduced_rule_arities
            .iter()
            .filter(|((candidate, arity), _)| candidate == &predicate && *arity == original_arity)
            .map(|(_, arity)| *arity)
            .collect::<BTreeSet<_>>();
        if arities.len() != 1 || !arities.contains(&target_arity) {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic augmented predicate schema".to_string(),
                context: format!(
                    "rules defining `{predicate}/{original_arity}` lower to incompatible \
                     internal arities {arities:?}; every clause for one predicate signature \
                     must bind the same augmented tuple shape"
                ),
            });
        }

        for query in program.queries.iter().filter(|query| {
            query.atom.predicate == predicate && query.atom.arity() == original_arity
        }) {
            let mut variables = BTreeSet::new();
            let unconstrained = query.atom.terms.iter().all(|term| match term {
                Term::Variable(name) => name != "_" && variables.insert(name.as_str()),
                Term::Anonymous
                | Term::Integer(_)
                | Term::Float(_)
                | Term::String(_)
                | Term::Symbol(_)
                | Term::List(_)
                | Term::Cons { .. }
                | Term::Compound { .. }
                | Term::PredRef(_)
                | Term::Aggregate(_) => false,
            });
            if !unconstrained {
                return Err(XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic augmented head query".to_string(),
                    context: format!(
                        "query `{predicate}/{original_arity}` is not a tuple of distinct named \
                         variables; an augmented epistemic head can currently surface only \
                         queries whose arguments are distinct named variables"
                    ),
                });
            }
        }
    }

    let eir = build_eir(program)?;
    let mut clauses_by_signature: BTreeMap<(String, usize), Vec<(usize, &crate::ast::Rule)>> =
        BTreeMap::new();
    for (rule_index, rule) in active_rules {
        clauses_by_signature
            .entry((rule.head.predicate.clone(), rule.head.arity()))
            .or_default()
            .push((rule_index, rule));
    }
    for ((predicate, arity), clauses) in clauses_by_signature {
        if clauses.len() > 1
            && clauses.iter().any(|(_, rule)| {
                rule.body
                    .iter()
                    .any(|literal| matches!(literal, BodyLiteral::Epistemic(_)))
            })
            && !epistemic_rule_union_gates_are_redundant(program, &eir, &clauses)
        {
            return Err(XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic rule-union materialization".to_string(),
                context: format!(
                    "predicate `{predicate}/{arity}` has multiple defining clauses and at least \
                     one epistemic clause; single-pass materialization cannot preserve \
                     per-clause modal provenance, so it cannot safely filter the clause union"
                ),
            });
        }
    }

    Ok(())
}

/// Prove that applying one clause's modal filters to an already-unioned relation
/// cannot remove rows contributed by its ordinary sibling clauses.
///
/// The single-pass materializer has no per-clause provenance. A multi-clause head is
/// therefore admitted only when either every clause has the same normalized modal
/// conjunction relative to its output columns, or there is exactly one epistemic
/// clause and every one of its modal gates is positive and provably true for every
/// candidate row:
///
/// - a ground atom has unconditional ordinary support from explicit facts/rules; or
/// - the modal atom is exactly a bijective all-variable clause head tuple and that
///   tuple has independent founded support under the clause's positive relational
///   domain; or
/// - G91 admits that same exact tuple under a positive `possible` self-support gate.
///
/// Equal conjunctions distribute over a union. The exact-head proofs are safe because
/// ordinary sibling clauses derive each head tuple directly, while the epistemic
/// clause's own rows are founded or explicitly self-supported under G91. Repeated
/// variables, constants, and wildcards are not bijective: they can map a sibling row
/// to a different modal key and therefore remain unsupported unless every clause has
/// the same normalized filter.
fn epistemic_rule_union_gates_are_redundant(
    program: &Program,
    eir: &EirProgram,
    clauses: &[(usize, &crate::ast::Rule)],
) -> bool {
    let invariant = InvariantRelations::analyze(program);
    let mut normalized_conjunctions = clauses.iter().map(|(rule_index, _)| {
        eir.rules
            .get(*rule_index)
            .and_then(|rule| normalized_rule_union_gates(rule, &invariant))
    });
    if let Some(Some(first)) = normalized_conjunctions.next() {
        if !first.is_empty()
            && normalized_conjunctions.all(|candidate| {
                candidate.is_some_and(|candidate| rule_union_gate_sets_equal(&first, &candidate))
            })
        {
            return true;
        }
    }

    let epistemic_clauses = clauses
        .iter()
        .filter(|(_, rule)| {
            rule.body
                .iter()
                .any(|literal| matches!(literal, BodyLiteral::Epistemic(_)))
        })
        .collect::<Vec<_>>();
    if epistemic_clauses.is_empty() {
        return true;
    }

    // Every modal filter is redundant when it is a positive ground atom with an
    // unconditional founded proof. This remains distributive even when several
    // sibling clauses use different ground modal atoms: every gate is true before
    // the clause outputs are unioned, so no per-clause provenance is needed later.
    let every_gate_is_unconditionally_true = epistemic_clauses.iter().all(|(rule_index, _)| {
        eir.rules.get(*rule_index).is_some_and(|eir_rule| {
            let modal_literals = eir_rule
                .body
                .iter()
                .filter_map(|literal| match literal {
                    EirBodyLiteral::Epistemic(modal) => Some(modal),
                    _ => None,
                })
                .collect::<Vec<_>>();
            !modal_literals.is_empty()
                && modal_literals.iter().all(|modal| {
                    !modal.negated && has_unconditional_ground_founded_support(eir, &modal.atom)
                })
        })
    });
    if every_gate_is_unconditionally_true {
        return true;
    }

    if epistemic_clauses.len() != 1 {
        return false;
    }

    let (rule_index, _) = epistemic_clauses[0];
    let Some(eir_rule) = eir.rules.get(*rule_index) else {
        return false;
    };
    let modal_literals = eir_rule
        .body
        .iter()
        .filter_map(|literal| match literal {
            EirBodyLiteral::Epistemic(modal) => Some(modal),
            _ => None,
        })
        .collect::<Vec<_>>();

    !modal_literals.is_empty()
        && modal_literals.iter().all(|modal| {
            !modal.negated
                && (has_unconditional_ground_founded_support(eir, &modal.atom)
                    || (modal.atom == eir_rule.head
                        && eir_head_is_bijective_variable_tuple(&eir_rule.head)
                        && ((eir.mode == EirEpistemicMode::G91
                            && modal.op == EirEpistemicOp::Possible)
                            || has_tuple_level_independent_founded_support(
                                eir,
                                eir_rule,
                                &modal.atom,
                            ))))
        })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RuleUnionGateTerm {
    OutputColumn(usize),
    Literal(EirTerm),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuleUnionGate {
    predicate: String,
    arity: usize,
    terms: Vec<RuleUnionGateTerm>,
    op: Option<EirEpistemicOp>,
    negated: bool,
}

/// Normalize a clause's modal conjunction to the exact tuple-key binding that the
/// materializer applies. Variable names are replaced by output-column positions, and
/// `know`/`possible` are identified over invariant relations because those relations
/// have one fixed extension in every accepted world.
fn normalized_rule_union_gates(
    rule: &xlog_ir::EirRule,
    invariant: &InvariantRelations<'_>,
) -> Option<Vec<RuleUnionGate>> {
    let output_terms = augmented_eir_head_terms(rule);
    let mut gates = Vec::new();
    for literal in &rule.body {
        let EirBodyLiteral::Epistemic(modal) = literal else {
            continue;
        };
        let bound_columns = bound_output_columns_for_terms(&modal.atom.terms, &output_terms);
        let terms = modal
            .atom
            .terms
            .iter()
            .zip(bound_columns)
            .map(|(term, output_column)| match (term, output_column) {
                (EirTerm::Variable(_), Some(column)) => {
                    Some(RuleUnionGateTerm::OutputColumn(column))
                }
                (
                    term @ (EirTerm::Anonymous
                    | EirTerm::Integer(_)
                    | EirTerm::FloatBits(_)
                    | EirTerm::String(_)
                    | EirTerm::Symbol(_)
                    | EirTerm::PredRef(_)),
                    None,
                ) => Some(RuleUnionGateTerm::Literal(term.clone())),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        let gate = RuleUnionGate {
            predicate: modal.atom.predicate.clone(),
            arity: modal.atom.arity,
            terms,
            op: (!invariant.is_invariant(&modal.atom.predicate)).then_some(modal.op),
            negated: modal.negated,
        };
        if !gates.contains(&gate) {
            gates.push(gate);
        }
    }
    Some(gates)
}

fn rule_union_gate_sets_equal(left: &[RuleUnionGate], right: &[RuleUnionGate]) -> bool {
    left.len() == right.len() && left.iter().all(|gate| right.contains(gate))
}

fn eir_head_is_bijective_variable_tuple(head: &xlog_ir::EirAtom) -> bool {
    let mut variables = BTreeSet::new();
    head.terms.iter().all(|term| match term {
        EirTerm::Variable(name) => variables.insert(name),
        _ => false,
    })
}

/// Widen each predicate's declaration to the maximum arity of its (now possibly
/// augmented) defining rule heads, inferring appended column types from the positive
/// body atoms that bind the augmented head variables.
///
/// Augmentation appends modal-local columns to a rule head; without widening the
/// matching `PredDecl`, the runtime would union the augmented rule output against the
/// narrow declared (empty) relation stub and fail with a schema mismatch.
///
/// Only rules in `augmented_rule_original_arities` are reconciled. Original arity is
/// retained separately because augmentation has already changed the rule head by the
/// time reconciliation runs.
///
/// Returns each original predicate signature that was augmented and its resulting
/// arity, whether or not that signature has an explicit declaration.
fn reconcile_augmented_head_declarations(
    reduced: &mut Program,
    augmented_rule_original_arities: &BTreeMap<usize, usize>,
) -> Result<BTreeMap<(String, usize), usize>> {
    use crate::ast::{PredColumn, TypeRef};

    // Per original head signature: the maximum augmented rule-head arity and, per
    // column position, an inferred type from a positive body atom (the resolved modal
    // or any binder).
    let mut augmented_signatures: BTreeMap<(String, usize), usize> = BTreeMap::new();
    let mut inferred_types: BTreeMap<(String, usize), Vec<Option<TypeRef>>> = BTreeMap::new();

    // Use the ordinary lowerer's fixed-point schema inference after extensional
    // multi-arity names have been qualified. This is the same source of truth used
    // by production compilation, so undeclared facts, transitive rule chains,
    // declarations, domains, and arithmetic bindings all contribute their real
    // scalar types instead of falling through to a guessed hidden-column type.
    let mut lowerer = Lowerer::new();
    lowerer.infer_schemas(reduced)?;
    let schemas = lowerer.schemas().clone();

    for (rule_index, rule) in reduced.rules.iter().enumerate() {
        if rule.body.is_empty() {
            continue;
        }
        // Only rules where the invariant-resolve genuinely fired are reconciled.
        let Some(&original_arity) = augmented_rule_original_arities.get(&rule_index) else {
            continue;
        };
        let arity = rule.head.terms.len();
        if arity <= original_arity {
            continue;
        }
        let signature = (rule.head.predicate.clone(), original_arity);
        let entry = augmented_signatures.entry(signature.clone()).or_insert(0);
        if arity > *entry {
            *entry = arity;
        }
        let types = inferred_types
            .entry(signature)
            .or_insert_with(|| vec![None; arity]);
        if types.len() < arity {
            types.resize(arity, None);
        }
        let variable_types = lowerer.infer_rule_variable_types(rule, |atom, index| {
            schemas
                .get(&atom.predicate)
                .and_then(|schema| schema.column_type(index))
        })?;

        // Infer each head variable's type from every binding form understood by
        // ordinary lowering, including body atoms and deterministic `is` results.
        for (col, term) in rule.head.terms.iter().enumerate() {
            if types[col].is_some() {
                continue;
            }
            let Term::Variable(head_var) = term else {
                continue;
            };
            if let Some((typ, _)) = variable_types.get(head_var) {
                types[col] = Some(TypeRef::Scalar(*typ));
            }
        }
    }

    for decl in &mut reduced.predicates {
        let signature = (decl.name.clone(), decl.arity());
        let Some(&target_arity) = augmented_signatures.get(&signature) else {
            continue;
        };
        let mut columns = decl.schema_columns();
        if target_arity <= columns.len() {
            continue;
        }
        let inferred = inferred_types.get(&signature);
        for col in columns.len()..target_arity {
            let typ = inferred
                .and_then(|types| types.get(col))
                .and_then(|t| t.clone())
                // Default appended columns to U32 (the modal relation key column type).
                .unwrap_or(TypeRef::Scalar(xlog_core::ScalarType::U32));
            columns.push(PredColumn { name: None, typ });
        }
        decl.types = columns.iter().map(|column| column.typ.clone()).collect();
        decl.columns = columns;
    }

    Ok(augmented_signatures)
}

fn append_body_local_tuple_key_variables_to_head(rule: &mut crate::ast::Rule) {
    let mut hidden_variables = Vec::new();
    for lit in &rule.body {
        let BodyLiteral::Epistemic(lit) = lit else {
            continue;
        };
        for term in &lit.atom.terms {
            let Term::Variable(variable) = term else {
                continue;
            };
            if variable == "_" {
                continue;
            }
            let already_in_head = rule
                .head
                .terms
                .iter()
                .any(|head_term| matches!(head_term, Term::Variable(name) if name == variable));
            if !already_in_head && !hidden_variables.iter().any(|name| name == variable) {
                hidden_variables.push(variable.clone());
            }
        }
    }
    for variable in hidden_variables {
        rule.head.terms.push(Term::Variable(variable));
    }
}

fn wcoj_status_for_reduction(
    positive_relational_atoms: &[xlog_ir::EirAtom],
    has_negated_relational_atom: bool,
) -> EpistemicWcojReductionStatus {
    if !has_negated_relational_atom
        && positive_relational_atoms_are_supported_wcoj_shape(positive_relational_atoms)
    {
        EpistemicWcojReductionStatus::RequiresPlannerEligibility
    } else {
        EpistemicWcojReductionStatus::NotWcojCandidate
    }
}

fn positive_relational_atoms_are_supported_wcoj_shape(atoms: &[xlog_ir::EirAtom]) -> bool {
    let mut edges: BTreeSet<(String, String)> = BTreeSet::new();
    let mut degrees: BTreeMap<String, usize> = BTreeMap::new();
    for atom in atoms {
        if atom.arity != 2 || atom.terms.len() != 2 {
            return false;
        }
        let Some(left) = eir_variable_name(&atom.terms[0]) else {
            return false;
        };
        let Some(right) = eir_variable_name(&atom.terms[1]) else {
            return false;
        };
        if left == right {
            return false;
        }
        let edge = if left < right {
            (left.to_string(), right.to_string())
        } else {
            (right.to_string(), left.to_string())
        };
        if !edges.insert(edge.clone()) {
            return false;
        }
        *degrees.entry(edge.0).or_insert(0) += 1;
        *degrees.entry(edge.1).or_insert(0) += 1;
    }

    match edges.len() {
        3 => degrees.len() == 3 && degrees.values().all(|degree| *degree == 2),
        4 => degrees.len() == 4 && degrees.values().all(|degree| *degree == 2),
        10 | 15 | 21 | 28 => {
            let variable_count = degrees.len();
            (5..=8).contains(&variable_count)
                && edges.len() == variable_count * (variable_count - 1) / 2
                && degrees.values().all(|degree| *degree == variable_count - 1)
        }
        _ => false,
    }
}

fn eir_variable_name(term: &EirTerm) -> Option<&str> {
    match term {
        EirTerm::Variable(name) => Some(name.as_str()),
        _ => None,
    }
}

/// Result of bounded FAEEL candidate evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaeelCandidateResult {
    /// Candidate satisfies the bounded FAEEL fixture semantics.
    Model,
    /// Candidate has no model for a typed reason.
    NoModel(FaeelNoModelReason),
}

/// Typed no-model reason for bounded FAEEL fixtures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaeelNoModelReason {
    /// Candidate uses possible-only support where FAEEL requires founded knowledge.
    UnfoundedPossible {
        /// Predicate name.
        predicate: String,
        /// Predicate arity.
        arity: usize,
    },
    /// Candidate marks the same atom known and rejected.
    Contradiction {
        /// Predicate name.
        predicate: String,
        /// Predicate arity.
        arity: usize,
    },
    /// An epistemic literal is unsatisfied by the candidate.
    UnsatisfiedLiteral {
        /// Predicate name.
        predicate: String,
        /// Predicate arity.
        arity: usize,
    },
}

/// Configuration for bounded Generate-Propagate-Test fixture execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneratePropagateTestConfig {
    /// Maximum candidate count accepted by the generate phase.
    pub max_candidates: usize,
}

/// Phase counters emitted by bounded Generate-Propagate-Test execution.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GeneratePropagateTestTrace {
    /// Number of generated candidates.
    pub generated: usize,
    /// Number of epistemic guesses generated.
    pub guesses: usize,
    /// Number of candidates that survived propagation.
    pub propagated: usize,
    /// Number of candidates pruned during propagation.
    pub pruned: usize,
    /// Number of reduced-program models inspected by the test phase.
    pub reduced_program_models: usize,
    /// Number of candidates tested.
    pub tested: usize,
    /// Number of accepted candidates.
    pub accepted: usize,
    /// Number of accepted world views.
    pub accepted_world_views: usize,
    /// Number of rejected candidates.
    pub rejected: usize,
    /// Rejection reasons observed during propagation and testing.
    pub rejection_reasons: Vec<FaeelNoModelReason>,
}

/// Result of bounded Generate-Propagate-Test fixture execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratePropagateTestOutcome {
    /// Phase counts.
    pub trace: GeneratePropagateTestTrace,
    /// Original indices of accepted candidates.
    pub accepted_candidate_indices: Vec<usize>,
    /// Original indices of rejected candidates in rejection-reason order.
    pub rejected_candidate_indices: Vec<usize>,
}

/// Reason that two source rules were coalesced into the same dependency component.
///
/// These reasons make the split planner's structural decisions explainable: a
/// caller can read, for every component, *why* its rules could not be solved
/// independently of one another through split-component diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum EpistemicComponentMergeReason {
    /// Two rules share the same head predicate, so they jointly define one
    /// derived relation and must be solved together.
    SharedHeadPredicate {
        /// Head predicate defined by both rules.
        predicate: String,
    },
    /// One rule's body consumes a predicate that another rule derives in its
    /// head (an ordinary/negated derived dependency).
    DerivedPredicate {
        /// Head predicate produced by the producer rule and consumed by the
        /// consumer rule body.
        predicate: String,
    },
    /// Two rules reference the same epistemic (modal) predicate, so their
    /// world-view acceptance is mutually dependent.
    SharedModalPredicate {
        /// Epistemic predicate referenced by both rules, with arity.
        predicate: String,
    },
    /// An integrity constraint mentions head predicates owned by both rules, so
    /// the constraint coalesces exactly those components.
    Constraint {
        /// Constraint-mentioned head predicates that forced the coalesce.
        predicates: Vec<String>,
    },
}

/// One deterministic dependency component for epistemic splitting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicDependencyComponent {
    /// Sorted predicate names in the component.
    pub predicates: Vec<String>,
    /// Source rule indices owned by the component.
    pub rule_indices: Vec<usize>,
    /// Sorted, deduplicated reasons the component's rules were coalesced.
    ///
    /// Empty when the component is a single independent rule that no
    /// dependency forced together (it was split out on its own).
    pub merge_reasons: Vec<EpistemicComponentMergeReason>,
}

/// Deterministic dependency graph used by bounded epistemic splitting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicDependencyGraph {
    /// Sorted components.
    pub components: Vec<EpistemicDependencyComponent>,
}

/// Split plan for independently solvable epistemic components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicSplitPlan {
    /// Components to solve independently.
    pub components: Vec<EpistemicDependencyComponent>,
}

impl EpistemicSplitPlan {
    /// Return the original rule order recovered from all components.
    pub fn recomposed_rule_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self
            .components
            .iter()
            .flat_map(|component| component.rule_indices.iter().copied())
            .collect();
        indices.sort_unstable();
        indices
    }
}

/// One split component lowered through the production epistemic GPU plan path.
#[derive(Debug, Clone)]
pub struct EpistemicSplitExecutableComponent {
    /// Source dependency component covered by this executable subplan.
    pub component: EpistemicDependencyComponent,
    /// GPU contract plus reduced runtime plan for this component.
    pub executable: EpistemicExecutablePlan,
}

/// Executable split plan whose components reuse the normal epistemic GPU lowering.
#[derive(Debug, Clone)]
pub struct EpistemicSplitExecutablePlan {
    /// Original bounded split plan.
    pub split_plan: EpistemicSplitPlan,
    /// Epistemic components compiled into GPU executable subplans.
    pub components: Vec<EpistemicSplitExecutableComponent>,
}

impl EpistemicSplitExecutablePlan {
    /// Return the source rule indices actually recomposed by GPU split execution.
    ///
    /// This reflects the rules the *executable* plan runs: epistemic-bearing
    /// components only. Pure-ordinary independent components carry no epistemic
    /// output buffer and are not part of the epistemic execution surface, so
    /// they are intentionally excluded here. The full dependency-graph view
    /// (including non-executed ordinary components) lives on
    /// [`EpistemicSplitPlan::recomposed_rule_indices`]; the two coincide exactly
    /// when every component is epistemic-bearing.
    pub fn recomposed_rule_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self
            .components
            .iter()
            .flat_map(|component| component.component.rule_indices.iter().copied())
            .collect();
        indices.sort_unstable();
        indices
    }

    /// Return the full dependency-graph recomposition view, including
    /// independent non-epistemic components that the executable plan does not run.
    pub fn planned_recomposed_rule_indices(&self) -> Vec<usize> {
        self.split_plan.recomposed_rule_indices()
    }

    /// Return executable components ordered by the first source rule they cover.
    pub fn recomposed_components(&self) -> Vec<&EpistemicSplitExecutableComponent> {
        let mut components: Vec<_> = self.components.iter().collect();
        components.sort_by_key(|component| {
            component
                .component
                .rule_indices
                .iter()
                .copied()
                .min()
                .unwrap_or(usize::MAX)
        });
        components
    }
}

/// Evaluate a single parsed epistemic literal against a bounded interpretation.
pub fn evaluate_epistemic_literal(
    mode: EpistemicMode,
    lit: &EpistemicLiteral,
    interpretation: &EpistemicInterpretation,
) -> TruthValue {
    let value = match lit.op {
        EpistemicOp::Know => interpretation.contains_known(&lit.atom),
        EpistemicOp::Possible => match mode {
            EpistemicMode::G91 => {
                interpretation.contains_known(&lit.atom)
                    || interpretation.contains_possible(&lit.atom)
            }
            EpistemicMode::Faeel => interpretation.contains_known(&lit.atom),
        },
    };

    TruthValue::from_bool(if lit.negated { !value } else { value })
}

/// Evaluate all epistemic literals in a program under bounded FAEEL fixture semantics.
pub fn evaluate_faeel_candidate(
    program: &Program,
    interpretation: &EpistemicInterpretation,
) -> Result<FaeelCandidateResult> {
    evaluate_epistemic_candidate(program, interpretation, EpistemicMode::Faeel)
}

/// Evaluate all epistemic literals in a program under a bounded fixture semantics mode.
pub fn evaluate_epistemic_candidate(
    program: &Program,
    interpretation: &EpistemicInterpretation,
    mode: EpistemicMode,
) -> Result<FaeelCandidateResult> {
    let prepared = prepare_root_authored_constraint_identity(program)?;
    evaluate_prepared_epistemic_candidate(&prepared, interpretation, mode)
}

fn evaluate_prepared_epistemic_candidate(
    program: &Program,
    interpretation: &EpistemicInterpretation,
    mode: EpistemicMode,
) -> Result<FaeelCandidateResult> {
    reject_gpt_epistemic_constraints(program)?;
    if let Some((predicate, arity)) = interpretation.first_contradiction() {
        return Ok(FaeelCandidateResult::NoModel(
            FaeelNoModelReason::Contradiction { predicate, arity },
        ));
    }

    for rule in &program.rules {
        for body_lit in &rule.body {
            let BodyLiteral::Epistemic(lit) = body_lit else {
                continue;
            };
            if interpretation.contains_known(&lit.atom)
                && interpretation.contains_rejected(&lit.atom)
            {
                return Ok(FaeelCandidateResult::NoModel(
                    FaeelNoModelReason::Contradiction {
                        predicate: lit.atom.predicate.clone(),
                        arity: lit.atom.arity(),
                    },
                ));
            }
            if mode == EpistemicMode::Faeel
                && lit.op == EpistemicOp::Possible
                && interpretation.contains_possible(&lit.atom)
                && !interpretation.contains_known(&lit.atom)
            {
                return Ok(FaeelCandidateResult::NoModel(
                    FaeelNoModelReason::UnfoundedPossible {
                        predicate: lit.atom.predicate.clone(),
                        arity: lit.atom.arity(),
                    },
                ));
            }
            if evaluate_epistemic_literal(mode, lit, interpretation) == TruthValue::False {
                return Ok(FaeelCandidateResult::NoModel(
                    FaeelNoModelReason::UnsatisfiedLiteral {
                        predicate: lit.atom.predicate.clone(),
                        arity: lit.atom.arity(),
                    },
                ));
            }
        }
    }

    Ok(FaeelCandidateResult::Model)
}

/// Run bounded Generate-Propagate-Test execution over explicit candidates.
pub fn run_generate_propagate_test(
    program: &Program,
    candidates: Vec<EpistemicInterpretation>,
    config: GeneratePropagateTestConfig,
) -> Result<GeneratePropagateTestOutcome> {
    run_generate_propagate_test_with_mode(
        program,
        candidates,
        config,
        program.directives.epistemic_mode_or_default(),
    )
}

/// Run bounded Generate-Propagate-Test execution over explicit candidates and semantics mode.
pub fn run_generate_propagate_test_with_mode(
    program: &Program,
    candidates: Vec<EpistemicInterpretation>,
    config: GeneratePropagateTestConfig,
    mode: EpistemicMode,
) -> Result<GeneratePropagateTestOutcome> {
    let prepared = prepare_root_authored_constraint_identity(program)?;
    reject_gpt_epistemic_constraints(&prepared)?;
    if candidates.len() > config.max_candidates {
        return Err(xlog_core::XlogError::CapacityExceeded {
            context: "epistemic GPT candidate guard".to_string(),
            required: candidates.len() as u64,
            limit: config.max_candidates as u64,
            unit: "candidates".to_string(),
        });
    }

    let generated = candidates.len();
    let guesses = candidates
        .iter()
        .map(EpistemicInterpretation::epistemic_guess_count)
        .sum();
    let mut propagated_candidates = Vec::new();
    let mut rejection_reasons = Vec::new();
    let mut rejected_candidate_indices = Vec::new();
    for (idx, candidate) in candidates.into_iter().enumerate() {
        if let Some((predicate, arity)) = candidate.first_contradiction() {
            rejection_reasons.push(FaeelNoModelReason::Contradiction { predicate, arity });
            rejected_candidate_indices.push(idx);
        } else {
            propagated_candidates.push((idx, candidate));
        }
    }

    let mut trace = GeneratePropagateTestTrace {
        generated,
        guesses,
        propagated: propagated_candidates.len(),
        pruned: generated.saturating_sub(propagated_candidates.len()),
        reduced_program_models: propagated_candidates.len(),
        rejection_reasons,
        ..GeneratePropagateTestTrace::default()
    };
    let mut accepted_candidate_indices = Vec::new();

    for (idx, candidate) in &propagated_candidates {
        trace.tested += 1;
        match evaluate_prepared_epistemic_candidate(&prepared, candidate, mode)? {
            FaeelCandidateResult::Model => {
                trace.accepted += 1;
                trace.accepted_world_views += 1;
                accepted_candidate_indices.push(*idx);
            }
            FaeelCandidateResult::NoModel(reason) => {
                trace.rejected += 1;
                trace.rejection_reasons.push(reason);
                rejected_candidate_indices.push(*idx);
            }
        }
    }

    Ok(GeneratePropagateTestOutcome {
        trace,
        accepted_candidate_indices,
        rejected_candidate_indices,
    })
}

/// Build a deterministic dependency graph for bounded epistemic splitting.
pub fn build_epistemic_dependency_graph(program: &Program) -> Result<EpistemicDependencyGraph> {
    if program.rules.is_empty() {
        return Ok(EpistemicDependencyGraph { components: vec![] });
    }

    let mut parents: Vec<usize> = (0..program.rules.len()).collect();
    let mut rule_predicates = Vec::with_capacity(program.rules.len());
    let mut head_owner: BTreeMap<String, usize> = BTreeMap::new();
    // Each merge records (one source rule index touched by the merge, reason).
    // After roots collapse, reasons are attributed to the surviving root so the
    // emitted component carries an explainable account of why it was coalesced.
    let mut merge_log: Vec<(usize, EpistemicComponentMergeReason)> = Vec::new();

    for (idx, rule) in program.rules.iter().enumerate() {
        if rule.body.is_empty() {
            continue;
        }
        if let Some(owner) = head_owner.get(&rule.head.predicate).copied() {
            union_components(&mut parents, owner, idx);
            merge_log.push((
                idx,
                EpistemicComponentMergeReason::SharedHeadPredicate {
                    predicate: rule.head.predicate.clone(),
                },
            ));
        } else {
            head_owner.insert(rule.head.predicate.clone(), idx);
        }
    }

    let mut modal_owner: BTreeMap<EpistemicAtomKey, usize> = BTreeMap::new();
    for (idx, rule) in program.rules.iter().enumerate() {
        let mut predicates = BTreeSet::new();
        predicates.insert(rule.head.predicate.clone());
        for lit in &rule.body {
            if let BodyLiteral::Epistemic(lit) = lit {
                let key =
                    EpistemicAtomKey::from_arity(lit.atom.predicate.clone(), lit.atom.arity());
                if let Some(owner) = modal_owner.get(&key).copied() {
                    union_components(&mut parents, owner, idx);
                    merge_log.push((
                        idx,
                        EpistemicComponentMergeReason::SharedModalPredicate {
                            predicate: format!("{}/{}", lit.atom.predicate, lit.atom.arity()),
                        },
                    ));
                } else {
                    modal_owner.insert(key, idx);
                }
            }
            if let Some(atom) = lit.atom() {
                if let Some(owner) = head_owner.get(&atom.predicate).copied() {
                    if owner != idx {
                        union_components(&mut parents, owner, idx);
                        merge_log.push((
                            idx,
                            EpistemicComponentMergeReason::DerivedPredicate {
                                predicate: atom.predicate.clone(),
                            },
                        ));
                    }
                }
                predicates.insert(atom.predicate.clone());
            }
        }

        rule_predicates.push(predicates);
    }

    let mut constraint_predicates = Vec::with_capacity(program.constraints.len());
    for constraint in &program.constraints {
        let predicates = constraint_predicate_set(constraint);
        let mut owners = predicates
            .iter()
            .filter_map(|predicate| head_owner.get(predicate).copied());
        if let Some(first_owner) = owners.next() {
            let mut coalesced_any = false;
            for owner in owners {
                if find_component(&mut parents, first_owner) != find_component(&mut parents, owner)
                {
                    coalesced_any = true;
                }
                union_components(&mut parents, first_owner, owner);
            }
            if coalesced_any {
                let constraint_heads: Vec<String> = predicates
                    .iter()
                    .filter(|predicate| head_owner.contains_key(*predicate))
                    .cloned()
                    .collect();
                merge_log.push((
                    first_owner,
                    EpistemicComponentMergeReason::Constraint {
                        predicates: constraint_heads,
                    },
                ));
            }
        }
        constraint_predicates.push(predicates);
    }

    let mut grouped: BTreeMap<usize, (BTreeSet<String>, Vec<usize>)> = BTreeMap::new();
    for (idx, predicates) in rule_predicates.into_iter().enumerate() {
        let root = find_component(&mut parents, idx);
        let entry = grouped
            .entry(root)
            .or_insert_with(|| (BTreeSet::new(), vec![]));
        entry.0.extend(predicates);
        entry.1.push(idx);
    }
    for predicates in constraint_predicates {
        let Some(root) = predicates
            .iter()
            .filter_map(|predicate| head_owner.get(predicate).copied())
            .map(|idx| find_component(&mut parents, idx))
            .next()
        else {
            continue;
        };
        grouped
            .entry(root)
            .or_insert_with(|| (BTreeSet::new(), vec![]))
            .0
            .extend(predicates);
    }

    // Attribute every recorded merge reason to its surviving component root.
    let mut reasons_by_root: BTreeMap<usize, BTreeSet<EpistemicComponentMergeReason>> =
        BTreeMap::new();
    for (touched_idx, reason) in merge_log {
        let root = find_component(&mut parents, touched_idx);
        reasons_by_root.entry(root).or_default().insert(reason);
    }

    let mut components: Vec<EpistemicDependencyComponent> = grouped
        .into_iter()
        .map(|(root, (predicates, mut rule_indices))| {
            rule_indices.sort_unstable();
            let merge_reasons = reasons_by_root
                .remove(&root)
                .map(|reasons| reasons.into_iter().collect())
                .unwrap_or_default();
            EpistemicDependencyComponent {
                predicates: predicates.into_iter().collect(),
                rule_indices,
                merge_reasons,
            }
        })
        .collect();
    components.sort_by(|a, b| a.predicates.cmp(&b.predicates));
    Ok(EpistemicDependencyGraph { components })
}

fn constraint_predicate_set(constraint: &Constraint) -> BTreeSet<String> {
    constraint
        .body
        .iter()
        .filter_map(|lit| lit.atom().map(|atom| atom.predicate.clone()))
        .collect()
}

fn find_component(parents: &mut [usize], idx: usize) -> usize {
    if parents[idx] != idx {
        let root = find_component(parents, parents[idx]);
        parents[idx] = root;
    }
    parents[idx]
}

fn union_components(parents: &mut [usize], left: usize, right: usize) {
    let left_root = find_component(parents, left);
    let right_root = find_component(parents, right);
    if left_root != right_root {
        parents[right_root] = left_root;
    }
}

/// Split an epistemic program into independently solvable bounded components.
/// One stratum of a stratified epistemic program: a self-contained sub-program
/// whose epistemic heads gate only over EDB/invariant relations OR over the
/// materialized (now-base) outputs of strictly-lower strata.
#[derive(Debug, Clone)]
pub struct EpistemicStratum {
    /// The epistemic output head predicate(s) this stratum materializes.
    pub head_predicates: Vec<String>,
    /// Source-rule indices owned by this stratum.
    pub rule_indices: Vec<usize>,
    /// The self-contained sub-program for this stratum (its own defining rules
    /// plus the facts/EDB it needs). Lower-stratum heads are NOT redefined here;
    /// at execution they are present in the store as materialized base relations.
    pub program: Program,
}

/// A stratified epistemic execution plan: an ordered sequence of strata.
///
/// Stratum `i`'s epistemic heads are materialized (gated) into the relation store
/// BEFORE stratum `i+1` runs, so a higher stratum's `know`/`possible` over a
/// lower stratum's head reads the GATED extension through the EXISTING
/// membership filter (no resolve-into-body, no double-gating).
#[derive(Debug, Clone)]
pub struct EpistemicStratifiedPlan {
    /// Strata in execution (topological) order.
    pub strata: Vec<EpistemicStratum>,
    /// One prepared ordinary closure and constraint stage executed after every
    /// stratum has materialized its gated heads.
    pub ordinary_post_program: Program,
}

/// Predicates whose epistemic extension is DETERMINED once lower strata are fixed.
///
/// A predicate is *epistemically determined* when every defining rule uses only
/// (a) positive `know`/`possible` modals and ordinary positive/negated literals,
/// (b) all ranging over predicates that are themselves invariant (EDB/lower
/// non-epistemic stratum) OR already epistemically determined, and (c) the
/// dependency is acyclic through BOTH modal and ordinary edges. Such a head's
/// materialized (gated) extension IS its truth, so it can be materialized into the
/// store as a base relation and a higher stratum can gate against it.
///
/// This is a STANDALONE analysis: it never feeds
/// [`reduce_case_a_epistemic_program_to_ordinary`] / `is_invariant`, so it cannot
/// trigger the resolve-into-body double-gating that the single-pass GPU filter
/// already performs.
struct EpistemicallyDeterminedPredicates {
    determined: BTreeSet<String>,
}

impl EpistemicallyDeterminedPredicates {
    fn analyze(program: &Program) -> Self {
        let invariant = InvariantRelations::analyze(program);

        // Heads defined by at least one rule.
        let mut derived_heads: BTreeSet<&str> = BTreeSet::new();
        for rule in &program.rules {
            if !rule.body.is_empty() {
                derived_heads.insert(rule.head.predicate.as_str());
            }
        }

        // Least-fixpoint closure over ALL derived heads (epistemic AND ordinary): a
        // predicate becomes determined when EVERY rule defining it ranges (modal +
        // ordinary) only over invariant or already-determined predicates, with no
        // self-reference (acyclic).
        //
        // An ORDINARY head is determined transitively when every defining rule ranges
        // only over determined/invariant relations (e.g. `r :- a` with `a` a
        // determined epistemic head). Such an `r` is determined-in-principle: its
        // extension is fixed once the determined heads it derives from are fixed, so a
        // higher modal `know r`/`possible r` can stratify against the materialized
        // base `r` via the existing membership filter. The acyclicity guard in
        // `head_is_determined` (self-reference returns false) plus the fixpoint's
        // monotonicity keep every recursive predicate OUT of `determined`, so a
        // circular `know reach` in a recursive SCC is never determined
        // and stays fail-closed.
        let mut determined: BTreeSet<String> = BTreeSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            for head in &derived_heads {
                if determined.contains(*head) {
                    continue;
                }
                if Self::head_is_determined(program, head, &invariant, &derived_heads, &determined)
                {
                    determined.insert((*head).to_string());
                    changed = true;
                }
            }
        }

        Self { determined }
    }

    /// Whether `head`'s every defining rule ranges only over invariant or
    /// already-determined predicates (acyclic — no reference to `head` itself).
    fn head_is_determined(
        program: &Program,
        head: &str,
        invariant: &InvariantRelations,
        derived_heads: &BTreeSet<&str>,
        determined: &BTreeSet<String>,
    ) -> bool {
        let mut defined = false;
        for rule in &program.rules {
            if rule.head.predicate != head || rule.body.is_empty() {
                continue;
            }
            defined = true;
            for lit in &rule.body {
                let referenced = match lit {
                    BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                        atom.predicate.as_str()
                    }
                    BodyLiteral::Epistemic(modal) => modal.atom.predicate.as_str(),
                    BodyLiteral::Comparison(_) | BodyLiteral::IsExpr(_) | BodyLiteral::Univ(_) => {
                        continue
                    }
                };
                if referenced == head {
                    // Self-reference: not acyclically determined (recursion /
                    // circular modality). Hand back to the recursive/FAEEL paths.
                    return false;
                }
                let ok = invariant.is_invariant(referenced)
                    || determined.contains(referenced)
                    // A pure-EDB predicate not seen by `derived_heads` is invariant.
                    || !derived_heads.contains(referenced);
                if !ok {
                    return false;
                }
            }
        }
        defined
    }

    fn contains(&self, predicate: &str) -> bool {
        self.determined.contains(predicate)
    }
}

/// Plan a STRATIFIED epistemic execution when the program contains a modal literal
/// over an epistemic-derived head that is itself epistemically DETERMINED.
///
/// This intercepts exactly the chained/nested-epistemic coupling that the joint
/// single-enumeration path fails closed on (`b :- know a` where `a :- know p`, `p`
/// invariant). It partitions the program's epistemic heads into strata by modal
/// dependency, where a head whose modal ranges over a lower DETERMINED head sits in
/// a strictly-higher stratum. Each stratum is a self-contained sub-program compiled
/// through the EXISTING single/joint epistemic path; at runtime the executor
/// materializes each stratum's GATED head into the store before the next stratum
/// runs, so the higher stratum gates against the materialized (now-base) relation
/// via the existing membership filter — never via resolve-into-body.
///
/// Returns:
/// - `Ok(Some(plan))` when the program genuinely needs (and admits) stratification:
///   at least one modal literal ranges over an epistemically-determined derived
///   head, and a sound stratification exists.
/// - `Ok(None)` when no modal ranges over a determined derived head (the existing
///   joint/split/single paths own the program — for example, a shared modal whose
///   target is extensional data rather than a determined derived head), OR
///   when the nested target is NOT determined (circular modality / recursion /
///   unfounded self-support is handed back to the recursive + FAEEL/G91 guards,
///   which keep ownership and fail closed there).
pub fn try_plan_stratified_epistemic_program(
    program: &Program,
) -> Result<Option<EpistemicStratifiedPlan>> {
    let prepared = prepare_root_authored_constraint_identity(program)?;
    let program = &prepared;
    let determined = EpistemicallyDeterminedPredicates::analyze(program);

    // A stratification is needed only when some modal literal ranges over a
    // DETERMINED epistemic-derived head. (A modal over a base/EDB predicate is the
    // ordinary single/joint path and must NOT be intercepted.)
    let mut needs_stratification = false;
    for rule in &program.rules {
        for lit in &rule.body {
            if let BodyLiteral::Epistemic(modal) = lit {
                if determined.contains(modal.atom.predicate.as_str())
                    && modal.atom.predicate != rule.head.predicate
                {
                    needs_stratification = true;
                }
            }
        }
    }
    if !needs_stratification {
        return Ok(None);
    }
    let removed_rules = faeel_unfounded_exact_tuple_self_support_rule_indices(program)
        .into_iter()
        .collect::<BTreeSet<_>>();
    validate_epistemic_derived_relation_identity(program, &removed_rules)?;

    // Assign each epistemic-derived head a stratum level = longest modal-dependency
    // chain to a determined head it gates over. Heads not determined cannot be
    // stratified soundly here; if any modal ranges over a non-determined derived
    // epistemic head, hand back to the joint path's fail-closed diagnostic.
    let stratum_level = assign_epistemic_strata(program, &determined)?;
    let Some(stratum_level) = stratum_level else {
        return Ok(None);
    };

    // Group epistemic-bearing rules by their head's stratum level.
    let mut levels: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (idx, rule) in program.rules.iter().enumerate() {
        let has_epistemic = rule
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)));
        if !has_epistemic {
            continue;
        }
        let Some(level) = stratum_level.get(rule.head.predicate.as_str()) else {
            // An epistemic head with no assigned level means the analysis could not
            // place it soundly; hand back.
            return Ok(None);
        };
        levels.entry(*level).or_default().push(idx);
    }

    if levels.len() < 2 {
        // Only one stratum: there is no lower stratum to materialize, so this is
        // not a genuine stratification (the existing paths own it).
        return Ok(None);
    }

    let mut strata = Vec::with_capacity(levels.len());
    for (_level, rule_indices) in levels {
        let head_predicates: Vec<String> = rule_indices
            .iter()
            .filter_map(|idx| program.rules.get(*idx))
            .map(|rule| rule.head.predicate.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let stratum_program =
            build_stratum_subprogram(program, &rule_indices, &head_predicates, &stratum_level)?;
        strata.push(EpistemicStratum {
            head_predicates,
            rule_indices,
            program: stratum_program,
        });
    }

    // Validate each stratum according to the reducer that will execute it. An
    // admissible recursive stratum resolves modal literals into ordinary rule bodies
    // and never appends hidden head columns; non-recursive strata use the single-pass
    // materializer and therefore require its union/shape checks.
    for stratum in &strata {
        if try_reduce_case_a_recursive_epistemic_program(&stratum.program)?.is_none() {
            validate_epistemic_relation_shapes(&stratum.program, &BTreeSet::new())?;
        }
    }

    let ordinary_post_program = build_post_stratification_ordinary_program(program);
    ordinary_post_program.validate_prepared_authored_constraint_identity()?;

    Ok(Some(EpistemicStratifiedPlan {
        strata,
        ordinary_post_program,
    }))
}

/// Build the single ordinary epilogue for a stratified epistemic execution.
///
/// Modal rules belong to their ordered strata. Every non-modal rule is replayed once
/// after all gated heads are materialized so deferred transitive closure (for example
/// `c :- b` where `b` is a top-stratum head) reaches its final extension before the
/// authored ordinary constraints are evaluated. Queries stay attached so the
/// high-level executor can surface relations derived only by this final stage.
fn build_post_stratification_ordinary_program(program: &Program) -> Program {
    let mut post = program.clone();
    post.rules.retain(|rule| {
        rule.body
            .iter()
            .all(|literal| !matches!(literal, BodyLiteral::Epistemic(_)))
    });
    post.constraints.retain(|constraint| {
        constraint
            .body
            .iter()
            .all(|literal| !matches!(literal, BodyLiteral::Epistemic(_)))
    });
    post
}

/// Build the rule rewrites used only for plan-wide schema inference in a
/// stratified program.
///
/// Each recursive stratum must contribute schemas from the same recursive epistemic
/// reducer selected for its executable plan. Applying the generic single-pass reducer
/// to those rules would append hidden tuple-key columns that their actual recursive
/// plan never produces.
fn stratified_schema_reduction_overrides(
    program: &Program,
) -> Result<BTreeMap<usize, crate::ast::Rule>> {
    let Some(plan) = try_plan_stratified_epistemic_program(program)? else {
        return Ok(BTreeMap::new());
    };

    let mut overrides = BTreeMap::new();
    for stratum in plan.strata {
        if try_reduce_case_a_recursive_epistemic_program(&stratum.program)?.is_none() {
            continue;
        }
        for rule_index in stratum.rule_indices {
            let mut rule = program.rules.get(rule_index).cloned().ok_or_else(|| {
                XlogError::Compilation(format!(
                    "stratified epistemic rule index {rule_index} is outside the source program"
                ))
            })?;
            resolve_recursive_epistemic_rule_modals(&mut rule);
            overrides.insert(rule_index, rule);
        }
    }
    Ok(overrides)
}

/// Assign each epistemic-derived head an integer stratum level.
///
/// Level 0 heads gate only over invariant/EDB relations. A head whose modal ranges
/// over a determined head at level `k` is at level `>= k + 1`. Returns `Ok(None)`
/// if any modal ranges over a derived-epistemic head that is NOT determined (those
/// genuinely-undefined / fail-closed shapes are owned by the joint/recursive
/// guards, which already produce typed diagnostics).
fn assign_epistemic_strata(
    program: &Program,
    determined: &EpistemicallyDeterminedPredicates,
) -> Result<Option<BTreeMap<String, usize>>> {
    // Epistemic-derived heads.
    let mut epistemic_heads: BTreeSet<&str> = BTreeSet::new();
    for rule in &program.rules {
        if rule
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
        {
            epistemic_heads.insert(rule.head.predicate.as_str());
        }
    }

    // Modal-over-derived-epistemic-head edges: head -> set of derived-epistemic
    // predicates its modals range over.
    //
    // A modal can target either a determined EPISTEMIC head directly (`b :- know a`),
    // or an ORDINARY predicate transitively derived from determined epistemic heads
    // (`b :- know r` with `r :- a`, `a` epistemic-determined). For the ordinary case,
    // the modal's head must sit strictly ABOVE the epistemic head(s) in the ordinary
    // target's transitive determined support, so those epistemic heads are materialized
    // (gated) into the store first and the ordinary `r :- a` is then computed over the
    // materialized base (making `r` locally invariant). We therefore route an edge from
    // the modal's head to EACH epistemic determined head in the target's support.
    let mut modal_edges: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for rule in &program.rules {
        let head = rule.head.predicate.as_str();
        for lit in &rule.body {
            if let BodyLiteral::Epistemic(modal) = lit {
                let target = modal.atom.predicate.as_str();
                if epistemic_heads.contains(target) {
                    if !determined.contains(target) {
                        // Modal over a non-determined epistemic head: not soundly
                        // stratifiable here. Hand back to the joint/recursive guard.
                        return Ok(None);
                    }
                    modal_edges.entry(head).or_default().insert(target);
                } else if determined.contains(target) {
                    // Modal over an ORDINARY determined predicate: route edges to the
                    // epistemic determined heads in its transitive support so the
                    // modal's head sits above them.
                    let support =
                        epistemic_support_of_determined_ordinary(program, target, &epistemic_heads);
                    if support.is_empty() {
                        // No epistemic head in the support means the target is fully
                        // invariant (pure-ordinary over EDB) — that is the ordinary
                        // single/joint path, not a stratification. Hand back.
                        return Ok(None);
                    }
                    let entry = modal_edges.entry(head).or_default();
                    for support_head in support {
                        entry.insert(support_head);
                    }
                }
            }
        }
    }

    // Longest-path level via memoized DFS over modal_edges (acyclicity guaranteed
    // by `EpistemicallyDeterminedPredicates`, which rejects self-reference).
    let mut level: BTreeMap<String, usize> = BTreeMap::new();
    fn visit<'a>(
        head: &'a str,
        modal_edges: &BTreeMap<&'a str, BTreeSet<&'a str>>,
        level: &mut BTreeMap<String, usize>,
        active: &mut BTreeSet<&'a str>,
    ) -> Result<usize> {
        if let Some(l) = level.get(head) {
            return Ok(*l);
        }
        if !active.insert(head) {
            // A cycle through modal edges should have been excluded upstream; be
            // defensive and refuse to stratify.
            return Err(recursive_epistemic_rejection(
                "stratified epistemic planning encountered a modal dependency cycle",
            ));
        }
        let mut l = 0;
        if let Some(targets) = modal_edges.get(head) {
            for target in targets {
                let tl = visit(target, modal_edges, level, active)?;
                l = l.max(tl + 1);
            }
        }
        active.remove(head);
        level.insert(head.to_string(), l);
        Ok(l)
    }

    for head in &epistemic_heads {
        visit(head, &modal_edges, &mut level, &mut BTreeSet::new())?;
    }

    Ok(Some(level))
}

/// The epistemic determined heads in the transitive ordinary support of a determined
/// ORDINARY predicate.
///
/// For `r :- a` with `a` an epistemic-determined head, `support_of("r") = {"a"}`. The
/// search follows positive/negated ordinary body atoms (the ordinary derivation), and
/// collects any referenced predicate that is itself an epistemic head. Bounded by the
/// (acyclic) determined-closure, so a simple visited-set DFS terminates.
fn epistemic_support_of_determined_ordinary<'a>(
    program: &'a Program,
    predicate: &'a str,
    epistemic_heads: &BTreeSet<&'a str>,
) -> BTreeSet<&'a str> {
    let mut support: BTreeSet<&'a str> = BTreeSet::new();
    let mut seen: BTreeSet<&'a str> = BTreeSet::new();
    let mut stack: Vec<&'a str> = vec![predicate];
    while let Some(current) = stack.pop() {
        if !seen.insert(current) {
            continue;
        }
        for rule in &program.rules {
            if rule.head.predicate != current || rule.body.is_empty() {
                continue;
            }
            for lit in &rule.body {
                let referenced = match lit {
                    BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) => {
                        atom.predicate.as_str()
                    }
                    // An epistemic literal in the support means `current` is itself an
                    // epistemic head; record it and do not descend through the modal.
                    BodyLiteral::Epistemic(_)
                    | BodyLiteral::Comparison(_)
                    | BodyLiteral::IsExpr(_)
                    | BodyLiteral::Univ(_) => continue,
                };
                if epistemic_heads.contains(referenced) {
                    support.insert(referenced);
                } else {
                    // Descend through ordinary derivations toward their epistemic roots.
                    stack.push(referenced);
                }
            }
        }
        // If `current` itself is an epistemic head, it is its own support root.
        if epistemic_heads.contains(current) && current != predicate {
            support.insert(current);
        }
    }
    support
}

/// Build a self-contained sub-program for one stratum.
///
/// Includes this stratum's epistemic-defining rules plus every fact and every
/// ordinary (non-epistemic) supporting rule whose head is NOT a lower-stratum
/// epistemic head. Lower-stratum epistemic heads are intentionally OMITTED: at
/// execution they are present in the store as materialized base relations, and
/// including their (modal-stripped, ungated) defining rules would overwrite the
/// gated extension. Their `pred` declarations are retained so the reduced compiler
/// sees a schema for the materialized base relation.
fn build_stratum_subprogram(
    program: &Program,
    rule_indices: &[usize],
    head_predicates: &[String],
    stratum_level: &BTreeMap<String, usize>,
) -> Result<Program> {
    let this_level = head_predicates
        .iter()
        .filter_map(|h| stratum_level.get(h))
        .copied()
        .max()
        .unwrap_or(0);

    // Lower-stratum epistemic heads: present as materialized base relations at
    // runtime; their defining rules must NOT appear in this sub-program.
    let lower_epistemic_heads: BTreeSet<&str> = stratum_level
        .iter()
        .filter(|(_, level)| **level < this_level)
        .map(|(head, _)| head.as_str())
        .collect();

    // All epistemic-derived heads (used to compute an ordinary rule's epistemic
    // support for deferral of determined-ordinary supporting rules).
    let all_epistemic_heads: BTreeSet<&str> = program
        .rules
        .iter()
        .filter(|rule| {
            rule.body
                .iter()
                .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
        })
        .map(|rule| rule.head.predicate.as_str())
        .collect();

    let own_rule_indices: BTreeSet<usize> = rule_indices.iter().copied().collect();

    let mut stratum = program.clone();
    stratum.rules = program
        .rules
        .iter()
        .enumerate()
        .filter_map(|(idx, rule)| {
            if own_rule_indices.contains(&idx) {
                return Some(rule.clone());
            }
            // Drop any rule that (re)defines a lower-stratum epistemic head.
            if lower_epistemic_heads.contains(rule.head.predicate.as_str()) {
                return None;
            }
            // Keep facts and ordinary supporting rules (EDB + non-epistemic
            // derivations the stratum's bodies may reference).
            let has_epistemic = rule
                .body
                .iter()
                .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)));
            if has_epistemic && !own_rule_indices.contains(&idx) {
                // Another stratum's epistemic rule: exclude.
                return None;
            }
            // An ORDINARY supporting rule whose transitive epistemic support includes a
            // head NOT yet materialized (gated) at this level must NOT run here — it
            // would compute over the UNGATED candidate extension of that head and leak
            // the wrong tuples into the store (which the higher stratum then gates
            // against). Defer it to the lowest stratum where ALL its epistemic support
            // is already a materialized gated base relation. E.g. `r :- a` (a an
            // epistemic-determined head) is dropped from `a`'s own stratum (level 0) and
            // kept only in the strictly-higher stratum where `a` is materialized base,
            // so `r` is computed once from the gated `a`. Pure-ordinary rules over EDB
            // (empty epistemic support) are never deferred.
            let support = epistemic_support_of_determined_ordinary(
                program,
                rule.head.predicate.as_str(),
                &all_epistemic_heads,
            );
            if support
                .iter()
                .any(|h| stratum_level.get(*h).copied().unwrap_or(0) >= this_level)
            {
                return None;
            }
            Some(rule.clone())
        })
        .collect();

    // Authored queries are compiled exactly once by the post-stratification stage.
    // Keeping compiler-local `__xlog_query_N` relations in modal strata would reuse
    // the same local names across strata and can collide when query projections have
    // different schemas.
    let head_set: BTreeSet<&str> = head_predicates.iter().map(String::as_str).collect();
    stratum.queries.clear();

    // Ordinary constraints are global postconditions and belong only to the single
    // post-stratification ordinary stage. Modal constraints retain predicate-local
    // ownership because they participate in candidate/world-view semantics here.
    stratum.constraints = program
        .constraints
        .iter()
        .filter(|constraint| {
            let is_ordinary = constraint
                .body
                .iter()
                .all(|literal| !matches!(literal, BodyLiteral::Epistemic(_)));
            if is_ordinary {
                return false;
            }
            constraint_predicate_set(constraint)
                .iter()
                .all(|p| head_set.contains(p.as_str()) || !is_program_head(program, p))
        })
        .cloned()
        .collect();

    Ok(stratum)
}

fn is_program_head(program: &Program, predicate: &str) -> bool {
    program
        .rules
        .iter()
        .any(|rule| !rule.body.is_empty() && rule.head.predicate == predicate)
}

/// Partition an epistemic program into independently-evaluable components.
///
/// Builds the epistemic dependency graph (coalescing rules that couple distinct
/// epistemic body predicates into one component) and returns an
/// [`EpistemicSplitPlan`] describing which output heads evaluate together versus
/// in isolation. This is the entry point for the safe-split / joint-solving and
/// stratified-execution routing decisions in the GPU driver.
pub fn split_epistemic_program(program: &Program) -> Result<EpistemicSplitPlan> {
    // rules that couple more than one distinct epistemic body predicate
    // are NOT rejected here. The dependency graph already unions every such rule
    // into a single component (each epistemic predicate occurrence routes through
    // `modal_owner` in `build_epistemic_dependency_graph`), and that component is
    // recompiled through the unsplit joint path
    // (`compile_epistemic_gpu_execution`), which enumerates the full candidate
    // lattice and validates the FULL modal conjunction jointly on device. Any
    // genuinely out-of-fragment coupling (unsafe variables, unsupported
    // tuple-key/nested-modal semantics) stays fail-closed via the downstream
    // joint-path guards (`build_eir` safety analysis,
    // `validate_tuple_membership_bindings`, `validate_solver_contract`) with their
    // own typed source-contextualized diagnostics, so no blanket coupling
    // rejection is needed at the split boundary.
    Ok(EpistemicSplitPlan {
        components: build_epistemic_dependency_graph(program)?.components,
    })
}

/// Compile valid epistemic split components through the production GPU executable path.
pub fn compile_epistemic_gpu_split_execution(
    program: &Program,
) -> Result<EpistemicSplitExecutablePlan> {
    compile_epistemic_gpu_split_execution_with_stats_snapshot(program, None)
}

/// Compile valid epistemic split components with an optional production stats snapshot.
///
/// Each component subprogram is lowered through
/// [`compile_epistemic_gpu_execution_with_stats_snapshot`], so split execution
/// reuses the same GPU contract, reduced compiler pipeline, WCOJ promotion, and
/// helper-splitting surfaces as unsplit epistemic execution.
pub fn compile_epistemic_gpu_split_execution_with_stats_snapshot(
    program: &Program,
    stats_snapshot: Option<&StatsSnapshot>,
) -> Result<EpistemicSplitExecutablePlan> {
    let mut prepared = program.clone();
    if prepared.authored_constraint_source_bound.is_some() {
        prepared.validate_prepared_authored_constraint_identity()?;
    } else {
        prepared.prepare_authored_constraint_identity_at_root()?;
    }
    let program = &prepared;
    validate_epistemic_relation_shapes(program, &BTreeSet::new())?;
    reject_epistemic_constraints(program)?;
    let split_plan = split_epistemic_program(program)?;
    let mut components = Vec::new();

    for component in &split_plan.components {
        if !component_has_epistemic_rule(program, component) {
            continue;
        }

        // Cross-component coupling carrying >1 epistemic output head is either
        // JOINT-SOLVED (a coalesced component whose modal literals all range over
        // base/invariant predicates -- a shared accepted world view materializes
        // every head) or fails closed with a precise typed diagnostic (a modal
        // literal ranges over an epistemic-derived head of the same component, so
        // the heads' world-view acceptance is genuinely interdependent and the
        // independent split would be unsound). A single epistemic head is always
        // the existing single-output joint path.
        let coupling = classify_cross_component_modal_coupling(program, component)?;

        let component_program = split_component_program(program, component)?;
        let executable = compile_epistemic_gpu_execution_inner(
            &component_program,
            stats_snapshot,
            coupling.allows_multiple_output_heads(),
        )?;
        components.push(EpistemicSplitExecutableComponent {
            component: component.clone(),
            executable,
        });
    }

    if components.is_empty() {
        return Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "epistemic GPU split execution".to_string(),
            context: "requires at least one epistemic split component".to_string(),
        });
    }

    Ok(EpistemicSplitExecutablePlan {
        split_plan,
        components,
    })
}

fn component_has_epistemic_rule(
    program: &Program,
    component: &EpistemicDependencyComponent,
) -> bool {
    component
        .rule_indices
        .iter()
        .filter_map(|idx| program.rules.get(*idx))
        .any(|rule| {
            rule.body
                .iter()
                .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
        })
}

/// Distinct head predicates of the component's epistemic-bearing rules, sorted.
///
/// Each such head is a final epistemic output relation the joint single-pass GPU
/// path would have to materialize. The single-output-buffer contract
/// ([`require_single_epistemic_output_relation`]) admits exactly one, so a count
/// above one means the component is genuinely *coupled* across what local
/// analysis would otherwise split — its epistemic outputs cannot be solved
/// independently AND cannot be jointly materialized into one buffer.
fn component_epistemic_output_heads(
    program: &Program,
    component: &EpistemicDependencyComponent,
) -> Vec<String> {
    let mut heads: BTreeSet<String> = BTreeSet::new();
    for idx in &component.rule_indices {
        let Some(rule) = program.rules.get(*idx) else {
            continue;
        };
        let has_epistemic_body = rule
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)));
        if has_epistemic_body {
            heads.insert(rule.head.predicate.clone());
        }
    }
    heads.into_iter().collect()
}

/// Render a coalesced component's merge reasons into a stable, human-readable list
/// for the cross-component coupling diagnostic.
///
/// These reasons (`DerivedPredicate`, `SharedModalPredicate`, `SharedHeadPredicate`,
/// `Constraint`) are exactly *why* the dependency graph could not split the
/// component's epistemic outputs, so naming them tells the caller which structural
/// coupling forced the fail-closed.
fn format_component_merge_reasons(component: &EpistemicDependencyComponent) -> String {
    if component.merge_reasons.is_empty() {
        return "no recorded coalesce reason".to_string();
    }
    component
        .merge_reasons
        .iter()
        .map(|reason| match reason {
            EpistemicComponentMergeReason::SharedHeadPredicate { predicate } => {
                format!("SharedHeadPredicate({predicate})")
            }
            EpistemicComponentMergeReason::DerivedPredicate { predicate } => {
                format!("DerivedPredicate({predicate})")
            }
            EpistemicComponentMergeReason::SharedModalPredicate { predicate } => {
                format!("SharedModalPredicate({predicate})")
            }
            EpistemicComponentMergeReason::Constraint { predicates } => {
                format!("Constraint({})", predicates.join(", "))
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Classification of a coalesced epistemic component's cross-component coupling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrossComponentCoupling {
    /// At most one epistemic output head, or a multi-head component whose modal
    /// literals all range over base/invariant predicates. The shared accepted
    /// world view materializes every head, so the component is JOINT-SOLVED.
    JointSolvable,
}

impl CrossComponentCoupling {
    /// True when the component's GPU plan is permitted to carry more than one
    /// epistemic output head (joint multi-head materialization).
    fn allows_multiple_output_heads(self) -> bool {
        match self {
            CrossComponentCoupling::JointSolvable => true,
        }
    }
}

/// Classify a coalesced component's cross-component modal coupling, JOINT-SOLVING
/// the canonical shared-base-modal case and failing closed (with a precise typed
/// diagnostic) on genuinely interdependent nested-epistemic coupling.
///
/// A coalesced component carrying more than one epistemic output head is either:
///
/// - **Joint-solvable** — every modal literal in the component ranges over a
///   predicate that is NOT an epistemic-derived head of the component (a
///   base/invariant relation or an ordinary-derived relation). The accepted
///   world-view set is then determined independently of which head is being
///   materialized, so one joint candidate enumeration + world-view validation
///   over the combined modal literals yields a single accepted world view, and
///   each head materialized against THAT world view equals its per-head
///   reduced-program evaluation. This is the canonical `SharedModalPredicate`
///   joint-solving target (`a(X):-know q(X). b(X):-possible q(X).` over base `q`).
///
/// - **Genuinely interdependent (fail closed)** — some modal literal ranges over
///   an EPISTEMIC-DERIVED head of the same component (`flagged():-know trusted()`
///   where `trusted` is itself `know`-derived). The modal truth of that predicate
///   depends on a DIFFERENT head's accepted world view, so the heads' acceptance
///   is mutually entangled (nested/stratified epistemic dependency). Solving it
///   would require stratified world-view nesting that the single joint enumeration
///   does not provide, so it stays FAIL-CLOSED with a typed diagnostic naming the
///   coupled heads, the modal predicate, and the merge reason -- never silently
///   mis-evaluated.
///
/// SAFE single-epistemic-head coupling (an ordinary body consuming an epistemic
/// head, `b():-a()` over `a():-know p()`) and EDB-only sharing are both
/// `JointSolvable` (one or zero coupled heads), so they stay accepted.
/// Compute the predicates whose extension depends, directly or transitively
/// through ordinary rules in the component, on an epistemic-derived head.
///
/// Seeded with the component's epistemic output heads (each is "tainted" because
/// its extension is gated by a modal literal), then closed under the rule
/// dependency relation: a head becomes tainted when ANY rule defining it (within
/// the component) references an already-tainted predicate in its body. A modal
/// literal over a tainted predicate is a nested/stratified epistemic dependency.
fn epistemic_tainted_predicates<'a>(
    program: &'a Program,
    component: &EpistemicDependencyComponent,
    epistemic_heads: &'a [String],
) -> BTreeSet<&'a str> {
    let mut tainted: BTreeSet<&str> = epistemic_heads.iter().map(String::as_str).collect();
    // Iterate the component's rules to a least fixpoint: a rule's head is tainted
    // if any body atom references a tainted predicate.
    let mut changed = true;
    while changed {
        changed = false;
        for idx in &component.rule_indices {
            let Some(rule) = program.rules.get(*idx) else {
                continue;
            };
            if tainted.contains(rule.head.predicate.as_str()) {
                continue;
            }
            // `BodyLiteral::atom()` covers relational AND epistemic literals
            // (the modal predicate), so this taints a head whether it depends on a
            // tainted predicate ordinarily or through a modal literal.
            let body_touches_tainted = rule.body.iter().any(|lit| {
                lit.atom()
                    .map(|atom| tainted.contains(atom.predicate.as_str()))
                    .unwrap_or(false)
            });
            if body_touches_tainted {
                tainted.insert(rule.head.predicate.as_str());
                changed = true;
            }
        }
    }
    tainted
}

fn classify_cross_component_modal_coupling(
    program: &Program,
    component: &EpistemicDependencyComponent,
) -> Result<CrossComponentCoupling> {
    let epistemic_heads = component_epistemic_output_heads(program, component);
    if epistemic_heads.len() <= 1 {
        return Ok(CrossComponentCoupling::JointSolvable);
    }

    // A modal literal ranging over a predicate whose extension DEPENDS (directly
    // OR TRANSITIVELY, through ordinary rules in this component) on an
    // epistemic-derived head is a nested/stratified epistemic dependency that the
    // single joint enumeration cannot solve soundly: that modal's truth would have
    // to be re-evaluated under EACH candidate world view chosen for the head it
    // depends on, which one shared world-view enumeration does not provide.
    //
    // "Epistemic-tainted" predicates = epistemic-derived heads, closed under the
    // ordinary rule dependency relation within the component (least fixpoint). A
    // modal over any tainted predicate fails closed. A modal over a purely
    // base/invariant or epistemic-INDEPENDENT predicate is joint-solvable.
    let tainted = epistemic_tainted_predicates(program, component, &epistemic_heads);

    let mut nested_modal_predicates: BTreeSet<String> = BTreeSet::new();
    for idx in &component.rule_indices {
        let Some(rule) = program.rules.get(*idx) else {
            continue;
        };
        for lit in &rule.body {
            if let BodyLiteral::Epistemic(modal) = lit {
                if tainted.contains(modal.atom.predicate.as_str()) {
                    nested_modal_predicates.insert(format!(
                        "{}/{}",
                        modal.atom.predicate,
                        modal.atom.arity()
                    ));
                }
            }
        }
    }

    if nested_modal_predicates.is_empty() {
        // Every modal literal ranges over a predicate that is independent of every
        // epistemic-derived head, so the accepted world view is determined solely
        // by base/invariant relations and the component is joint-solvable over one
        // shared accepted world view.
        return Ok(CrossComponentCoupling::JointSolvable);
    }

    Err(XlogError::UnsupportedEpistemicConstruct {
        construct: "cross-component epistemic coupling".to_string(),
        context: format!(
            "epistemic output heads {:?} are coupled into a single dependency \
             component (reasons: {}) through nested modal literals over \
             epistemic-derived predicates {:?}; the modal truth of an \
             epistemic-derived head depends on another head's accepted world view, \
             so a single joint world-view enumeration would mis-evaluate the \
             nested modality and an independent split would be unsound, so this \
             fails closed",
            epistemic_heads,
            format_component_merge_reasons(component),
            nested_modal_predicates.into_iter().collect::<Vec<_>>(),
        ),
    })
}

fn split_component_program(
    program: &Program,
    component: &EpistemicDependencyComponent,
) -> Result<Program> {
    let mut component_program = program.clone();
    let component_predicates: BTreeSet<&str> =
        component.predicates.iter().map(String::as_str).collect();
    let component_rule_indices: BTreeSet<usize> = component.rule_indices.iter().copied().collect();
    let head_predicates: BTreeSet<&str> = program
        .rules
        .iter()
        .map(|rule| rule.head.predicate.as_str())
        .collect();
    component_program.rules = program
        .rules
        .iter()
        .enumerate()
        .filter_map(|(idx, rule)| {
            (component_rule_indices.contains(&idx)
                || (rule.body.is_empty()
                    && component_predicates.contains(rule.head.predicate.as_str())))
            .then_some(rule.clone())
        })
        .collect();
    component_program.constraints = program
        .constraints
        .iter()
        .filter(|constraint| {
            let predicates = constraint_predicate_set(constraint);
            let has_component_owned_predicate = predicates
                .iter()
                .any(|predicate| head_predicates.contains(predicate.as_str()));
            !has_component_owned_predicate
                || predicates
                    .iter()
                    .all(|predicate| component_predicates.contains(predicate.as_str()))
        })
        .cloned()
        .collect();
    Ok(component_program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{PredColumn, PredDecl, TypeRef};
    use crate::parse_program;

    #[test]
    fn augmented_head_reconciliation_preserves_columns_only_declarations() {
        let symbol = TypeRef::Scalar(xlog_core::ScalarType::Symbol);
        let wide_integer = TypeRef::Scalar(xlog_core::ScalarType::I64);
        let mut program = Program::new();
        program.predicates = vec![
            PredDecl {
                name: "result".to_string(),
                types: Vec::new(),
                columns: vec![PredColumn {
                    name: Some("key".to_string()),
                    typ: symbol.clone(),
                }],
                is_private: false,
            },
            PredDecl {
                name: "source".to_string(),
                types: Vec::new(),
                columns: vec![
                    PredColumn {
                        name: Some("key".to_string()),
                        typ: symbol.clone(),
                    },
                    PredColumn {
                        name: Some("value".to_string()),
                        typ: wide_integer.clone(),
                    },
                ],
                is_private: false,
            },
        ];
        program.rules.push(crate::ast::Rule {
            head: Atom {
                predicate: "result".to_string(),
                terms: vec![
                    Term::Variable("Key".to_string()),
                    Term::Variable("Value".to_string()),
                ],
            },
            body: vec![BodyLiteral::Positive(Atom {
                predicate: "source".to_string(),
                terms: vec![
                    Term::Variable("Key".to_string()),
                    Term::Variable("Value".to_string()),
                ],
            })],
        });
        let resolved = BTreeMap::from([(0, 1)]);

        let widened = reconcile_augmented_head_declarations(&mut program, &resolved)
            .expect("reconcile augmented declaration");
        let declaration = program
            .predicates
            .iter()
            .find(|declaration| declaration.name == "result")
            .unwrap();

        assert_eq!(widened.get(&("result".to_string(), 1)), Some(&2));
        assert_eq!(declaration.arity(), 2);
        assert_eq!(
            declaration.types,
            vec![symbol.clone(), wide_integer.clone()]
        );
        assert_eq!(
            declaration
                .columns
                .iter()
                .map(|column| column.typ.clone())
                .collect::<Vec<_>>(),
            vec![symbol, wide_integer]
        );
    }

    #[test]
    fn augmented_head_reconciliation_uses_body_declarations_by_name_and_arity() {
        let program = parse_program(
            r#"
            #pragma epistemic_mode = faeel
            pred node(symbol).
            pred source(symbol, i64).
            pred source(u32).
            pred result(symbol).
            node(key).
            source(key, 5000000000).
            source(1).
            result(X) :- node(X), know source(X, Y).
            "#,
        )
        .expect("parse same-name multi-arity schema fixture");

        let reduced = reduce_epistemic_program_to_ordinary_for_stratified_schema(&program)
            .expect("same-name body declarations should reduce");
        let result = reduced
            .predicates
            .iter()
            .find(|declaration| declaration.name == "result")
            .expect("missing result declaration");
        let types = result
            .schema_columns()
            .into_iter()
            .map(|column| column.typ)
            .collect::<Vec<_>>();

        assert_eq!(
            types,
            vec![
                TypeRef::Scalar(xlog_core::ScalarType::Symbol),
                TypeRef::Scalar(xlog_core::ScalarType::I64),
            ]
        );
        assert!(reduced
            .predicates
            .iter()
            .any(|declaration| declaration.name == "source/2"));
        assert!(reduced
            .predicates
            .iter()
            .any(|declaration| declaration.name == "source/1"));
        crate::compile::Compiler::new()
            .compile_program(&reduced)
            .expect("arity-qualified reduced program should compile");
        compile_epistemic_gpu_execution(&program)
            .expect("production epistemic compilation should use the exact source signature");
    }

    #[test]
    fn augmented_head_reconciliation_uses_undeclared_fixed_point_schemas() {
        let sources = [
            r#"
                #pragma epistemic_mode = faeel
                pred node(symbol).
                pred result(symbol).
                node(key).
                edge(key, 5000000000).
                result(X) :- node(X), know edge(X, Y).
            "#,
            r#"
                #pragma epistemic_mode = faeel
                pred node(symbol).
                pred result(symbol).
                node(key).
                raw(key, 5000000000).
                edge(X, Y) :- raw(X, Y).
                result(X) :- node(X), know edge(X, Y).
            "#,
        ];

        for source in sources {
            let program = parse_program(source).expect("parse inferred-schema fixture");
            let reduced = reduce_epistemic_program_to_ordinary(&program)
                .expect("inferred modal source should reduce");
            let result = reduced
                .predicates
                .iter()
                .find(|declaration| declaration.name == "result")
                .expect("missing result declaration");
            assert_eq!(
                result
                    .schema_columns()
                    .into_iter()
                    .map(|column| column.typ)
                    .collect::<Vec<_>>(),
                vec![
                    TypeRef::Scalar(xlog_core::ScalarType::Symbol),
                    TypeRef::Scalar(xlog_core::ScalarType::I64),
                ]
            );
            crate::compile::Compiler::new()
                .compile_program(&reduced)
                .expect("fixed-point inferred hidden-column type should compile");
        }
    }

    #[test]
    fn augmented_head_reconciliation_uses_arithmetic_binding_type() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = faeel
                pred node(symbol).
                pred allowed(u64).
                pred result(symbol).
                node(key).
                allowed(1).
                result(X) :- node(X), Y is cast(1, u64), not know allowed(Y).
            "#,
        )
        .expect("parse arithmetic-binding fixture");

        let reduced = reduce_epistemic_program_to_ordinary(&program)
            .expect("arithmetic-bound hidden column should reduce");
        let result = reduced
            .predicates
            .iter()
            .find(|declaration| declaration.name == "result")
            .expect("missing result declaration");
        assert_eq!(
            result
                .schema_columns()
                .into_iter()
                .map(|column| column.typ)
                .collect::<Vec<_>>(),
            vec![
                TypeRef::Scalar(xlog_core::ScalarType::Symbol),
                TypeRef::Scalar(xlog_core::ScalarType::U64),
            ]
        );
        crate::compile::Compiler::new()
            .compile_program(&reduced)
            .expect("arithmetic-bound hidden-column type should compile");
    }

    #[test]
    fn augmented_head_reconciliation_widens_only_the_original_signature() {
        let mut reduced = parse_program(
            r#"
            pred edge(symbol, i64).
            pred triple(symbol, i64, u32).
            pred result(symbol, i64, u32).
            pred result(symbol).
            result(X, Y, Z) :- triple(X, Y, Z).
            result(X, Y) :- edge(X, Y).
            ?- result(A, B, C).
            ?- result(X).
            "#,
        )
        .expect("parse exact-signature reconciliation fixture");

        let augmented =
            reconcile_augmented_head_declarations(&mut reduced, &BTreeMap::from([(1, 1)]))
                .expect("reconcile exact augmented signature");
        let declaration_arities = reduced
            .predicates
            .iter()
            .filter(|declaration| declaration.name == "result")
            .map(PredDecl::arity)
            .collect::<Vec<_>>();
        let rule_arities = reduced
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "result")
            .map(|rule| rule.head.arity())
            .collect::<Vec<_>>();
        let query_arities = reduced
            .queries
            .iter()
            .filter(|query| query.atom.predicate == "result")
            .map(|query| query.atom.arity())
            .collect::<Vec<_>>();

        assert_eq!(augmented.get(&("result".to_string(), 1)), Some(&2));
        assert_eq!(declaration_arities, vec![3, 2]);
        assert_eq!(rule_arities, vec![3, 2]);
        assert_eq!(query_arities, vec![3, 1]);
    }

    #[test]
    fn augmented_undeclared_head_removes_its_original_query() {
        let program = parse_program(
            r#"
            #pragma epistemic_mode = faeel
            node(key).
            edge(key, 5000000000).
            result(X) :- node(X), know edge(X, Y).
            ?- result(X).
            "#,
        )
        .expect("parse undeclared augmented-head fixture");

        let reduced = reduce_epistemic_program_to_ordinary_for_stratified_schema(&program)
            .expect("undeclared augmented head should reduce");

        assert!(reduced
            .rules
            .iter()
            .any(|rule| rule.head.predicate == "result" && rule.head.arity() == 2));
        assert!(!reduced
            .queries
            .iter()
            .any(|query| query.atom.predicate == "result" && query.atom.arity() == 1));
    }

    #[test]
    fn divergent_augmented_head_arities_are_rejected_before_reduction() {
        let different_modal_widths = r#"
            #pragma epistemic_mode = faeel
            pred node(symbol).
            pred edge(symbol, i64).
            pred triple(symbol, i64, u32).
            pred result(symbol).
            node(key).
            edge(key, 5000000000).
            triple(key, 5000000000, 1).
            result(X) :- node(X), know edge(X, Y).
            result(X) :- node(X), know triple(X, Y, Z).
        "#;
        let ordinary_sibling = r#"
            #pragma epistemic_mode = faeel
            pred base(symbol).
            pred node(symbol).
            pred edge(symbol, i64).
            pred result(symbol).
            base(key).
            node(key).
            edge(key, 5000000000).
            result(X) :- base(X).
            result(X) :- node(X), know edge(X, Y).
        "#;

        for source in [different_modal_widths, ordinary_sibling] {
            let program = parse_program(source).expect("parse divergent augmented-head fixture");
            let errors = [
                plan_epistemic_gpu_execution(&program)
                    .expect_err("GPU planning must reject divergent reduced arities"),
                reduce_epistemic_program_to_ordinary(&program)
                    .expect_err("execution reduction must reject divergent arities"),
                reduce_epistemic_program_to_ordinary_for_stratified_schema(&program)
                    .expect_err("schema reduction must reject divergent arities"),
            ];

            for error in errors {
                let message = error.to_string();
                assert!(message.contains("epistemic augmented predicate schema"));
                assert!(message.contains("result/1"), "{message}");
                assert!(
                    message.contains("incompatible internal arities"),
                    "{message}"
                );
            }
        }
    }

    #[test]
    fn single_pass_epistemic_rule_unions_without_clause_provenance_are_rejected() {
        let fixtures = [
            r#"
                #pragma epistemic_mode = faeel
                pred p().
                pred q().
                pred result(symbol).
                q().
                result(a) :- know p().
                result(b) :- know q().
                ?- result(X).
            "#,
            r#"
                #pragma epistemic_mode = faeel
                pred q().
                pred result(symbol).
                result(a).
                result(b) :- know q().
                ?- result(X).
            "#,
        ];

        for source in fixtures {
            let program = parse_program(source).expect("parse epistemic rule-union fixture");
            let errors = [
                plan_epistemic_gpu_execution(&program)
                    .expect_err("GPU planning must reject a provenance-free rule union"),
                reduce_epistemic_program_to_ordinary(&program)
                    .expect_err("execution reduction must reject a provenance-free rule union"),
                reduce_epistemic_program_to_ordinary_for_stratified_schema(&program)
                    .expect_err("schema reduction must reject a provenance-free rule union"),
            ];
            for error in errors {
                let message = error.to_string();
                assert!(message.contains("epistemic rule-union materialization"));
                assert!(message.contains("result/1"), "{message}");
                assert!(message.contains("per-clause modal provenance"), "{message}");
            }
        }
    }

    #[test]
    fn equivalent_epistemic_rule_union_filters_are_distributive() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = faeel
                pred target(u32).
                pred left(u32).
                pred right(u32).
                pred result(u32).
                target(1).
                target(2).
                left(1).
                right(2).
                result(X) :- left(X), know target(X).
                result(Y) :- right(Y), possible target(Y).
                ?- result(Value).
            "#,
        )
        .expect("parse equivalent-filter rule union");

        plan_epistemic_gpu_execution(&program)
            .expect("equivalent invariant modal filters must distribute over the rule union");
        reduce_epistemic_program_to_ordinary(&program)
            .expect("execution reduction must preserve an equivalent-filter rule union");
        reduce_epistemic_program_to_ordinary_for_stratified_schema(&program)
            .expect("schema reduction must preserve an equivalent-filter rule union");
    }

    #[test]
    fn independently_founded_ground_gates_are_distributive_across_rule_union() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = g91
                pred left(u32).
                pred right(u32).
                pred result(u32).
                left(1).
                right(1).
                result(1) :- possible left(1).
                result(2) :- possible right(1).
                ?- result(Value).
            "#,
        )
        .expect("parse independently founded ground-gate rule union");

        plan_epistemic_gpu_execution(&program)
            .expect("independently founded ground gates must distribute over the rule union");
        reduce_epistemic_program_to_ordinary(&program)
            .expect("execution reduction must preserve independently founded ground gates");
    }

    #[test]
    fn g91_exact_head_possible_union_preserves_compatibility_self_support() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = g91
                pred seed(u32).
                pred node(u32).
                pred p(u32).
                seed(1).
                node(2).
                p(X) :- seed(X).
                p(X) :- node(X), possible p(X).
                ?- p(X).
            "#,
        )
        .expect("parse G91 exact-head possibility union");

        assert_eq!(
            classify_recursive_epistemic_program(&program).unwrap(),
            RecursiveEpistemicClass::ModalCycle
        );
        let prepared = prepare_epistemic_program(&program).expect("validate G91 source");
        let compatibility = try_prepare_g91_compatibility_reduction(&prepared)
            .expect("G91 modal cycle must be admitted")
            .expect("G91 modal cycle must use an explicit compatibility fixpoint");
        Compiler::new()
            .compile_program(compatibility.upper_bound_program())
            .expect("G91 upper-bound program must compile");
        Compiler::new()
            .compile_program(compatibility.refinement_program())
            .expect("declared G91 snapshot program must compile");
        assert_eq!(compatibility.snapshot_relations().len(), 1);
    }

    #[test]
    fn g91_compatibility_applies_to_exact_tuples_across_one_recursive_component() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = g91
                pred domain(u32).
                pred p(u32).
                pred q(u32).
                domain(1).
                p(X) :- domain(X), possible q(X).
                q(X) :- domain(X), possible p(X).
                ?- p(X).
                ?- q(X).
            "#,
        )
        .expect("parse mutual G91 modal component");

        let prepared = prepare_epistemic_program(&program).expect("validate mutual G91 source");
        let compatibility = try_prepare_g91_compatibility_reduction(&prepared)
            .expect("mutual G91 modal component must be admitted")
            .expect("mutual G91 modal component must use compatibility iteration");
        for rule in compatibility
            .upper_bound_program()
            .rules
            .iter()
            .filter(|rule| matches!(rule.head.predicate.as_str(), "p" | "q"))
        {
            assert!(
                rule.body
                    .iter()
                    .any(|literal| matches!(literal, BodyLiteral::Comparison(_))),
                "the exact tuple compatibility edge must become a tautological conjunct: {rule:?}"
            );
            assert!(
                !rule.body.iter().any(|literal| {
                    matches!(literal, BodyLiteral::Positive(atom) if atom.predicate == "p" || atom.predicate == "q")
                }),
                "a compatibility edge must not become an ordinary recursive join: {rule:?}"
            );
        }
        for rule in compatibility
            .refinement_program()
            .rules
            .iter()
            .filter(|rule| matches!(rule.head.predicate.as_str(), "p" | "q"))
        {
            assert!(rule.body.iter().any(|literal| {
                matches!(literal, BodyLiteral::Positive(atom) if atom.predicate.starts_with("__xlog_g91_snapshot_"))
            }));
            assert!(!rule
                .body
                .iter()
                .any(|literal| matches!(literal, BodyLiteral::Epistemic(_))));
        }
    }

    #[test]
    fn g91_snapshot_names_avoid_programmatic_relation_collisions() {
        let mut program = parse_program("pred p(u32).").expect("parse source relation");
        program.predicates.push(PredDecl {
            name: "__xlog_g91_snapshot_p".to_string(),
            types: vec![TypeRef::Scalar(xlog_core::ScalarType::U32)],
            columns: vec![PredColumn {
                name: None,
                typ: TypeRef::Scalar(xlog_core::ScalarType::U32),
            }],
            is_private: false,
        });
        let target = "p".to_string();
        let names = g91_snapshot_relation_names(&program, std::iter::once(&target));
        assert_eq!(
            names.get("p").map(String::as_str),
            Some("__xlog_g91_snapshot_p_0")
        );
    }

    #[test]
    fn g91_compatibility_rejects_recursive_aggregation_in_the_selected_component() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = g91
                pred seed(u32).
                pred p(u32).
                pred totals(u64).
                seed(1).
                p(X) :- seed(X), possible p(X).
                p(X) :- seed(X), totals(_).
                totals(count(X)) :- p(X).
                ?- p(X).
            "#,
        )
        .expect("parse recursive aggregate compatibility fixture");

        let prepared = prepare_epistemic_program(&program).expect("validate G91 source");
        let error = try_prepare_g91_compatibility_reduction(&prepared)
            .expect_err("recursive aggregation makes compatibility refinement non-monotone");
        let message = error.to_string();
        assert!(message.contains("Gelfond-1991 compatibility"), "{message}");
        assert!(message.contains("aggregate"), "{message}");
        assert!(message.contains("totals"), "{message}");
    }

    #[test]
    fn g91_compatibility_rejects_recursive_negation_in_the_selected_component() {
        for negated_dependency in [
            "not blocked(X)",
            "not possible blocked(X)",
            "not know blocked(X)",
        ] {
            let program = parse_program(&format!(
                r#"
                    #pragma epistemic_mode = g91
                    pred seed(u32).
                    pred p(u32).
                    pred blocked(u32).
                    seed(1).
                    p(X) :- seed(X), possible p(X).
                    p(X) :- seed(X), {negated_dependency}.
                    blocked(X) :- p(X).
                    ?- p(X).
                "#,
            ))
            .expect("parse recursive negation compatibility fixture");

            let prepared = prepare_epistemic_program(&program).expect("validate G91 source");
            let error = match try_prepare_g91_compatibility_reduction(&prepared) {
                Err(error) => error,
                Ok(_) => panic!(
                    "recursive dependency `{negated_dependency}` must make compatibility \
                     refinement non-monotone"
                ),
            };
            let message = error.to_string();
            assert!(message.contains("Gelfond-1991 compatibility"), "{message}");
            assert!(message.contains("negation"), "{message}");
            assert!(message.contains("p"), "{message}");
        }
    }

    #[test]
    fn source_validation_combines_modal_and_arithmetic_type_evidence_before_elision() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = faeel
                pred p(u32).
                p(X) :- X is cast(1, u64), possible p(X).
                ?- p(X).
            "#,
        )
        .expect("parse arithmetic and modal type-conflict fixture");

        let error = prepare_epistemic_program(&program)
            .expect_err("foundedness must not hide the authored type conflict");
        let message = error.to_string();
        assert!(message.contains("Type mismatch"), "{message}");
        assert!(
            message.contains("U32") && message.contains("U64"),
            "{message}"
        );
    }

    #[test]
    fn source_validation_uses_lowerer_arithmetic_order_before_elision() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = faeel
                pred p(i64).
                p(X) :- X is Y + 1, Y is 1, possible p(X).
            "#,
        )
        .expect("parse reversed arithmetic dependency fixture");

        let error = prepare_epistemic_program(&program)
            .expect_err("a later arithmetic binding cannot retroactively validate an earlier one");
        assert!(
            error.to_string().contains("variable X not bound"),
            "{error}"
        );
    }

    #[test]
    fn source_validation_rejects_structured_modal_arity_before_rule_elision() {
        for mode in ["faeel", "g91"] {
            let program = parse_program(&format!(
                r#"
                    #pragma epistemic_mode = {mode}
                    pred p(list<symbol>).
                    p([a, b]) :- possible p([a, b]).
                    ?- p(X).
                "#
            ))
            .expect("parse structured exact-self-support fixture");

            let error = prepare_epistemic_program(&program)
                .expect_err("a flattened modal key must match its authored target arity");
            let message = error.to_string();
            assert!(message.contains("epistemic modal tuple key"), "{message}");
            assert!(message.contains("target arity 1"), "{message}");
            assert!(message.contains("binding arity 2"), "{message}");
        }
    }

    #[test]
    fn source_validation_accepts_structured_key_matching_flat_target_arity() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = faeel
                pred host(u32, u32).
                pred watched(u32, u32).
                pred out(u32, u32).
                host(1, 2).
                watched(1, 2).
                out(X, Y) :- host(X, Y), know watched([X, Y]).
                ?- out(X, Y).
            "#,
        )
        .expect("parse matching structured modal key fixture");

        validate_epistemic_source_program(&program)
            .expect("a two-element structured key must address a binary target");

        let multi_arity = epistemic_extensional_multi_arity_predicates(&program);
        assert!(
            !multi_arity.contains("watched"),
            "a two-column structured modal key and watched/2 are one signature"
        );
    }

    #[test]
    fn invariant_analysis_treats_shared_acyclic_dependencies_as_a_diamond() {
        let program = parse_program(
            r#"
                pred base(u32).
                pred left(u32).
                pred right(u32).
                pred joined(u32).
                pred out(u32).
                base(1).
                left(X) :- base(X).
                right(X) :- base(X).
                joined(X) :- left(X), right(X).
                out(X) :- possible joined(X).
                ?- out(X).
            "#,
        )
        .expect("parse invariant diamond fixture");

        let invariant = InvariantRelations::analyze(&program);
        assert!(invariant.is_invariant("joined"));
        validate_epistemic_source_program(&program)
            .expect("the positive modal over an acyclic diamond must bind its output");
        reduce_epistemic_program_to_ordinary(&program)
            .expect("the invariant modal binder must reduce to an ordinary join");
    }

    #[test]
    fn positive_modal_binders_are_independent_of_conjunct_order() {
        for mode in ["faeel", "g91"] {
            for modal_body in [
                "possible p(X), possible base(X)",
                "possible base(X), possible p(X)",
            ] {
                let program = parse_program(&format!(
                    r#"
                        #pragma epistemic_mode = {mode}
                        pred base(u32).
                        pred p(u32).
                        base(1).
                        p(X) :- {modal_body}.
                        ?- p(X).
                    "#
                ))
                .expect("parse modal binder ordering fixture");

                validate_epistemic_source_program(&program).unwrap_or_else(|error| {
                    panic!("{mode} body `{modal_body}` must be range-restricted: {error}")
                });
            }
        }
    }

    #[test]
    fn negated_exact_modal_cycles_are_never_removed_as_foundedness_elision() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = faeel
                pred p().
                p() :- not possible p().
                ?- p().
            "#,
        )
        .expect("parse negated exact modal cycle");

        let prepared = prepare_epistemic_program(&program)
            .expect("negated modal cycle must survive source preparation");
        assert!(!prepared.removed_unfounded_rules());
        assert!(prepared.active_program().rules.iter().any(|rule| {
            rule.body
                .iter()
                .any(|literal| matches!(literal, BodyLiteral::Epistemic(modal) if modal.negated))
        }));
    }

    #[test]
    fn modal_fixpoint_reduction_preserves_non_bijective_sibling_unions() {
        let fixtures = [
            r#"
                #pragma epistemic_mode = faeel
                pred domain(symbol).
                pred other(symbol, symbol).
                pred p(symbol, symbol).
                domain(a).
                other(c, d).
                p(X, X) :- domain(X).
                p(A, B) :- other(A, B).
                p(X, X) :- domain(X), know p(X, X).
                ?- p(A, B).
            "#,
            r#"
                #pragma epistemic_mode = faeel
                pred left(symbol).
                pred right(symbol).
                pred p(symbol, symbol).
                left(x).
                right(y).
                p(a, X) :- left(X).
                p(b, Y) :- right(Y).
                p(a, X) :- left(X), know p(a, X).
                ?- p(A, B).
            "#,
        ];

        for source in fixtures {
            let program = parse_program(source).expect("parse non-bijective rule union");
            assert_eq!(
                classify_recursive_epistemic_program(&program).unwrap(),
                RecursiveEpistemicClass::ModalCycle
            );
            let reduced = try_reduce_case_a_recursive_epistemic_program(&program)
                .expect("modal-cycle sibling union must be admitted")
                .expect("modal-cycle sibling union must reduce to a fixpoint");
            Compiler::new()
                .compile_program(&reduced)
                .expect("ordinary fixpoint preserves per-clause sibling rows");
        }
    }

    #[test]
    fn constrained_support_does_not_found_a_wider_modal_domain() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = faeel
                pred seed(u32).
                pred p(u32).
                seed(1).
                seed(2).
                p(X) :- seed(X), X = 1.
                p(X) :- seed(X), possible p(X).
                ?- p(X).
            "#,
        )
        .expect("parse constrained foundedness program");

        let reduced = reduce_epistemic_program_to_ordinary(&program)
            .expect("the unfounded modal clause must be removed, not rejected");
        assert_eq!(
            reduced
                .rules
                .iter()
                .filter(|rule| rule.head.predicate == "p" && !rule.body.is_empty())
                .count(),
            1,
            "a constrained support clause cannot found the wider self-support domain"
        );
    }

    #[test]
    fn derived_predicate_source_arity_collisions_are_rejected() {
        let fixtures = [
            r#"
                #pragma epistemic_mode = faeel
                unary(a).
                binary(a, b).
                result(X) :- unary(X), know unary(X).
                result(X, Y) :- binary(X, Y), know binary(X, Y).
            "#,
            r#"
                #pragma epistemic_mode = faeel
                node(key).
                edge(key, 5000000000).
                result(X) :- node(X), know edge(X, Y).
                ?- result(A, B).
            "#,
            r#"
                #pragma epistemic_mode = faeel
                node(key).
                edge(key, 5000000000).
                result(X) :- node(X), know edge(X, Y).
                observer(X) :- result(X, Y).
            "#,
        ];

        for source in fixtures {
            let program = parse_program(source).expect("parse source-arity collision fixture");
            let errors = [
                plan_epistemic_gpu_execution(&program)
                    .expect_err("GPU planning must reject derived source-arity collisions"),
                reduce_epistemic_program_to_ordinary(&program)
                    .expect_err("execution reduction must reject source-arity collisions"),
                reduce_epistemic_program_to_ordinary_for_stratified_schema(&program)
                    .expect_err("schema reduction must reject source-arity collisions"),
            ];
            for error in errors {
                let message = error.to_string();
                assert!(
                    message.contains("epistemic derived predicate schema"),
                    "{message}"
                );
                assert!(message.contains("result"), "{message}");
                assert!(message.contains("{1, 2}"), "{message}");
            }
        }
    }

    #[test]
    fn constrained_augmented_head_queries_are_rejected() {
        let fixtures = [
            r#"
                #pragma epistemic_mode = faeel
                node(key).
                edge(key, 5000000000).
                result(X) :- node(X), know edge(X, Y).
                ?- result(other).
            "#,
            r#"
                #pragma epistemic_mode = faeel
                pair(left, right).
                edge(left, 5000000000).
                result(X, Y) :- pair(X, Y), know edge(X, Z).
                ?- result(Value, Value).
            "#,
            r#"
                #pragma epistemic_mode = faeel
                node(key).
                edge(key, 5000000000).
                allowed(5000000000).
                result(X) :- node(X), edge(X, Y), know allowed(Y).
                ?- result(key).
            "#,
        ];

        for source in fixtures {
            let program = parse_program(source).expect("parse constrained-query fixture");
            let errors = [
                plan_epistemic_gpu_execution(&program)
                    .expect_err("GPU planning must reject a constrained augmented query"),
                reduce_epistemic_program_to_ordinary(&program)
                    .expect_err("execution reduction must reject a constrained augmented query"),
                reduce_epistemic_program_to_ordinary_for_stratified_schema(&program)
                    .expect_err("schema reduction must reject a constrained augmented query"),
            ];
            for error in errors {
                let message = error.to_string();
                assert!(
                    message.contains("epistemic augmented head query"),
                    "{message}"
                );
                assert!(message.contains("distinct named variables"), "{message}");
            }
        }
    }

    #[test]
    fn removed_unfounded_sibling_does_not_create_an_augmented_arity_conflict() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = faeel
                pred node(symbol).
                pred edge(symbol, i64).
                pred result(symbol).
                node(key).
                edge(key, 5000000000).
                result(X) :- node(X), know edge(X, Y).
                result(key) :- possible result(key).
            "#,
        )
        .expect("parse foundedness fixture");

        let reduced = reduce_epistemic_program_to_ordinary(&program)
            .expect("removed unfounded support must not affect the surviving relation shape");
        let result_rules = reduced
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "result")
            .collect::<Vec<_>>();
        assert_eq!(result_rules.len(), 1);
        assert_eq!(result_rules[0].head.arity(), 1);
    }

    #[test]
    fn ordinary_bound_appended_columns_participate_in_shape_validation() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = faeel
                pred node(symbol).
                pred edge(symbol, i64).
                pred allowed(i64).
                pred result(symbol).
                node(key).
                edge(key, 5000000000).
                allowed(5000000000).
                result(X) :- node(X).
                result(X) :- node(X), edge(X, Y), know allowed(Y).
                ?- result(X).
            "#,
        )
        .expect("parse ordinary-bound augmentation fixture");

        let errors = [
            plan_epistemic_gpu_execution(&program)
                .expect_err("GPU planning must reject divergent internal arities"),
            reduce_epistemic_program_to_ordinary(&program)
                .expect_err("execution reduction must reject divergent internal arities"),
            reduce_epistemic_program_to_ordinary_for_stratified_schema(&program)
                .expect_err("schema reduction must reject divergent internal arities"),
        ];
        for error in errors {
            let message = error.to_string();
            assert!(
                message.contains("epistemic augmented predicate schema"),
                "{message}"
            );
            assert!(message.contains("result/1"), "{message}");
            assert!(message.contains("{1, 2}"), "{message}");
        }
    }

    #[test]
    fn ordinary_bound_appended_columns_widen_the_internal_declaration() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = faeel
                pred node(symbol).
                pred edge(symbol, i64).
                pred allowed(i64).
                pred result(symbol).
                node(key).
                edge(key, 5000000000).
                allowed(5000000000).
                result(X) :- node(X), edge(X, Y), know allowed(Y).
                ?- result(X).
            "#,
        )
        .expect("parse ordinary-bound declaration fixture");

        let reduced = reduce_epistemic_program_to_ordinary(&program)
            .expect("uniform ordinary-bound augmentation should reduce");
        let declaration = reduced
            .predicates
            .iter()
            .find(|declaration| declaration.name == "result")
            .expect("missing result declaration");
        assert_eq!(declaration.arity(), 2);
        crate::compile::Compiler::new()
            .compile_program(&reduced)
            .expect("reconciled ordinary-bound augmentation should compile");
    }

    #[test]
    fn stratified_schema_reduction_uses_the_recursive_stratum_reducer() {
        let program = parse_program(
            r#"
                #pragma epistemic_mode = faeel
                pred node(u32).
                pred edge(u32, u32).
                pred accepted_edge(u32, u32).
                pred reach(u32, u32).
                node(1).
                node(2).
                node(3).
                edge(1, 2).
                edge(2, 3).
                accepted_edge(X, Y) :- node(X), node(Y), know edge(X, Y).
                reach(X, Y) :- node(X), node(Y), know accepted_edge(X, Y).
                reach(X, Z) :- reach(X, Y), node(Z), know accepted_edge(Y, Z).
                ?- reach(X, Z).
            "#,
        )
        .expect("parse stratified recursive fixture");

        let plan = try_plan_stratified_epistemic_program(&program)
            .expect("stratified planning should succeed")
            .expect("fixture requires stratified execution");
        assert_eq!(plan.strata.len(), 2);

        let reduced = reduce_epistemic_program_to_ordinary_for_stratified_schema(&program)
            .expect("schema reduction should follow each stratum's executable reducer");
        assert!(reduced
            .rules
            .iter()
            .filter(|rule| rule.head.predicate == "reach")
            .all(|rule| rule.head.arity() == 2));
        assert_eq!(
            reduced
                .predicates
                .iter()
                .find(|declaration| declaration.name == "reach")
                .expect("missing reach declaration")
                .arity(),
            2
        );
        crate::compile::Compiler::new()
            .compile_program(&reduced)
            .expect("path-aligned stratified schema program should compile");
    }

    #[test]
    fn high_arity_epistemic_adapter_reduction_is_not_wcoj_required() {
        let program = parse_program(
            r#"
            pred case_variant(u32, u32).
            pred case_domain_variant(u32, u32, u32).
            pred domain_adapter_root(u32, u32, u32).
            pred domain_adapter_intervention(u32, u32, u32).
            pred domain_candidate_seed(u32, u32, u32, u32).
            pred heldout_label_seed(u32, u32).
            pred blocked_candidate(u32, u32, u32).
            pred generated_candidate(u32, u32, u32, u32, u32).

            generated_candidate(Case, Variant, Candidate, Root, Intervention) :-
                case_domain_variant(Case, Variant, Domain),
                domain_adapter_root(Domain, Candidate, Root),
                domain_adapter_intervention(Domain, Candidate, Intervention),
                domain_candidate_seed(Domain, Candidate, Root, Intervention),
                know domain_candidate_seed(Domain, Candidate, Root, Intervention),
                possible case_variant(Case, Variant),
                not know heldout_label_seed(Case, Candidate),
                not possible blocked_candidate(Case, Variant, Candidate).
            "#,
        )
        .expect("parse high-arity adapter epistemic program");

        let plan = plan_epistemic_gpu_execution(&program)
            .expect("plan high-arity adapter epistemic program");

        assert_eq!(plan.reductions.len(), 1);
        assert_eq!(
            plan.reductions[0].wcoj_status,
            EpistemicWcojReductionStatus::NotWcojCandidate
        );
    }

    #[test]
    fn binary_triangle_epistemic_reduction_still_requires_wcoj() {
        let program = parse_program(
            r#"
            pred xy(u32, u32).
            pred yz(u32, u32).
            pred xz(u32, u32).
            pred tri(u32, u32, u32).

            tri(X, Y, Z) :-
                xy(X, Y),
                yz(Y, Z),
                xz(X, Z),
                know xy(X, Y).
            "#,
        )
        .expect("parse binary triangle epistemic program");

        let plan =
            plan_epistemic_gpu_execution(&program).expect("plan binary triangle epistemic program");

        assert_eq!(plan.reductions.len(), 1);
        assert_eq!(
            plan.reductions[0].wcoj_status,
            EpistemicWcojReductionStatus::RequiresPlannerEligibility
        );
    }

    #[test]
    fn binary_eight_clique_epistemic_reduction_requires_wcoj() {
        let program = parse_program(
            r#"
            pred edge(u32, u32).
            pred clique8(u32, u32, u32, u32, u32, u32, u32, u32).

            clique8(A, B, C, D, E, F, G, H) :-
                edge(A, B),
                edge(A, C),
                edge(A, D),
                edge(A, E),
                edge(A, F),
                edge(A, G),
                edge(A, H),
                edge(B, C),
                edge(B, D),
                edge(B, E),
                edge(B, F),
                edge(B, G),
                edge(B, H),
                edge(C, D),
                edge(C, E),
                edge(C, F),
                edge(C, G),
                edge(C, H),
                edge(D, E),
                edge(D, F),
                edge(D, G),
                edge(D, H),
                edge(E, F),
                edge(E, G),
                edge(E, H),
                edge(F, G),
                edge(F, H),
                edge(G, H),
                know edge(A, B).
            "#,
        )
        .expect("parse binary eight-clique epistemic program");

        let plan = plan_epistemic_gpu_execution(&program)
            .expect("plan binary eight-clique epistemic program");

        assert_eq!(plan.reductions.len(), 1);
        assert_eq!(
            plan.reductions[0].wcoj_status,
            EpistemicWcojReductionStatus::RequiresPlannerEligibility
        );
    }
}
