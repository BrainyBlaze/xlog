//! Epistemic mode helpers for compatibility fixtures.

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
/// required device buffers, WCOJ planning obligations, and zero CPU fallback
/// counters.
pub fn plan_epistemic_gpu_execution(program: &Program) -> Result<EpistemicGpuPlan> {
    reject_recursive_epistemic_program(program)?;
    let eir = build_eir(program)?;
    // FAEEL unfounded modal self-support is NOT rejected here: it is a defined FAEEL
    // result (the unfounded head is simply absent from the founded model). The
    // structural foundedness decision drives the reduced-base drop in
    // `faeel_unfounded_self_support_rule_indices`; the founded extension is then
    // computed by the GPU world-view validation over the reduced base. See
    // `reduce_epistemic_program_to_ordinary_inner`.
    let mut epistemic_literals = Vec::new();
    let mut reductions = Vec::new();
    let mut tuple_membership_bindings = Vec::new();
    let mut solver_assumption_bindings = Vec::new();

    for (rule_index, rule) in eir.rules.iter().enumerate() {
        let mut rule_epistemic_literals = Vec::new();
        let mut relational_body_atoms = 0usize;
        let mut has_negated_relational_atom = false;

        for lit in &rule.body {
            match lit {
                EirBodyLiteral::Relational { negated, .. } => {
                    if *negated {
                        has_negated_relational_atom = true;
                    } else {
                        relational_body_atoms += 1;
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
            relational_body_atoms,
            wcoj_status: wcoj_status_for_reduction(
                relational_body_atoms,
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
    for (constraint_index, constraint) in eir.constraints.iter().enumerate() {
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
                        // bind no variable, no quantifier flip — the EGB-04 path).
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
    /// The program has no ordinary recursion among epistemic rules; the existing
    /// single-pass epistemic world-view executor handles it.
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
    /// target is admitted as Case B IFF the negation is STRATIFIED (the reduced ordinary
    /// `not R` has no cycle through negation, so the target sits in a strictly lower
    /// stratum); a genuine negation cycle stays rejected (3-valued well-founded
    /// semantics needs a host-side solver, precluded by the no-host-solver lock). A
    /// `possible` modal over a co-evolving target is admitted ONLY under FAEEL (where
    /// the least fixpoint equals the founded model). G91 `possible` recursion is the
    /// autoepistemic self-fulfilling fixpoint, which the monotone resolve cannot
    /// express, so it stays rejected as an honest wall.
    CaseB,
}

/// Reject epistemic programs that contain ordinary (non-modal) recursion before the
/// SINGLE-PASS GPU world-view planner.
///
/// [`plan_epistemic_gpu_execution`] builds a single-pass plan that evaluates each
/// candidate world view exactly once; it cannot iterate a recursive fixpoint, so ANY
/// ordinary recursion fails closed here — including the admissible Case-A fragment,
/// which is handled by a SEPARATE path
/// ([`try_reduce_case_a_recursive_epistemic_program`]) that delegates to the ordinary
/// recursive engine and intercepts Case-A programs before this planner is reached. In
/// production this guard therefore only ever sees non-recursive programs; it remains
/// defense-in-depth for direct callers of the single-pass planner.
///
/// Self-support THROUGH a modal literal (e.g. `p() :- possible p().`) is NOT ordinary
/// recursion: the modal edge is excluded from the dependency walk, so FAEEL/G91
/// foundedness still governs those cases. Under FAEEL the unfounded head is excluded
/// from the founded model by [`faeel_unfounded_self_support_rule_indices`] (the reduced
/// base drops the circular self-support rule); under G91 the circular form is accepted.
fn reject_recursive_epistemic_program(program: &Program) -> Result<()> {
    match classify_recursive_epistemic_program(program) {
        Ok(RecursiveEpistemicClass::NonRecursive) => Ok(()),
        Ok(RecursiveEpistemicClass::CaseA | RecursiveEpistemicClass::CaseB) => {
            Err(recursive_epistemic_rejection(
                "an epistemic program contains ordinary recursion; the single-pass epistemic GPU \
                 planner cannot iterate a recursive fixpoint. Case-A/Case-B recursive epistemic \
                 programs are executed through the ordinary recursive engine via \
                 `try_reduce_case_a_recursive_epistemic_program`, not this planner.",
            ))
        }
        // Recursive shapes outside the admissible fragment already carry a specific
        // typed diagnostic (negated-modal recursion, G91 `possible` recursion).
        Err(err) => Err(err),
    }
}

/// Classify an epistemic program's ordinary recursion as non-recursive or Case A.
///
/// Returns a typed [`XlogError::UnsupportedEpistemicConstruct`] for any recursive
/// shape outside Case A (recursion through a derived/recursive or epistemic relation,
/// a negated modal literal in a recursion-participating rule, etc.).
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

    // Dependency edges from ORDINARY (positive/negated) body literals only; modal,
    // comparison, and arithmetic literals do not contribute recursion edges here.
    let mut deps: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for rule in &program.rules {
        let entry = deps.entry(rule.head.predicate.as_str()).or_default();
        for lit in &rule.body {
            if let BodyLiteral::Positive(atom) | BodyLiteral::Negated(atom) = lit {
                entry.insert(atom.predicate.as_str());
            }
        }
    }

    fn reaches<'a>(
        start: &'a str,
        target: &str,
        deps: &BTreeMap<&'a str, BTreeSet<&'a str>>,
        seen: &mut BTreeSet<&'a str>,
    ) -> bool {
        let Some(next) = deps.get(start) else {
            return false;
        };
        for &pred in next {
            if pred == target {
                return true;
            }
            if seen.insert(pred) && reaches(pred, target, deps, seen) {
                return true;
            }
        }
        false
    }

    // Collect the set of ordinary-recursive predicates (predicates that ordinarily
    // depend on themselves through positive/negated body literals).
    let recursive_predicates: BTreeSet<&str> = deps
        .keys()
        .copied()
        .filter(|pred| reaches(pred, pred, &deps, &mut BTreeSet::new()))
        .collect();

    if recursive_predicates.is_empty() {
        return Ok(RecursiveEpistemicClass::NonRecursive);
    }

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
    // Both classes use the SAME reduction (positive modal → positive ordinary atom,
    // `reduce_case_a_epistemic_program_to_ordinary`), so the only structural difference
    // is whether the resolved relation is fixed (A) or part of the SCC (B). The whole
    // program is scanned (not only recursion-participating rules) because that blanket
    // reduction rewrites EVERY modal literal.
    //
    // SOUNDNESS FLOOR (stays rejected):
    //   * a NEGATED modal over a non-invariant target whose reduced `not R` forms a
    //     CYCLE THROUGH NEGATION — genuinely non-stratified; the sound 3-valued
    //     well-founded model requires a host-side WFS / stable-model solver (precluded
    //     by the no-host-solver lock). A STRATIFIED negated modal (target in a strictly
    //     lower stratum, even if recursive/non-invariant) IS admitted as Case B and runs
    //     as ordinary stratified negation on the GPU semi-naive engine.
    //   * a `possible` modal over a co-evolving target under G91 — G91 `possible` is the
    //     autoepistemic SELF-FULFILLING fixpoint, which the monotone least-fixpoint
    //     resolve cannot express. FAEEL `possible` IS the founded least fixpoint, so it
    //     is admitted. (A non-recursive `possible` self-support stays NonRecursive and
    //     is handled by the single-pass founded-extension path — item B.)
    let mode = program.directives.epistemic_mode_or_default();
    let invariant = InvariantRelations::analyze(program);
    let mut saw_case_b = false;
    // A NEGATED modal over a NON-invariant target is admissible IFF the negation is
    // STRATIFIED after reduction (no cycle through negation). The decision is DEFERRED
    // to a reduce + stratification check that runs AFTER this scan, so the immediate
    // per-literal rejections below (G91 `possible` self-fulfilling) still win when they
    // co-occur. Recording it as a candidate keeps the soundness floor under the SOLE
    // arbiter of the existing stratification analysis on the reduced program.
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
                // A NEGATED modal over a NON-invariant relation is the WALL A1 case.
                // SOUNDNESS ARGUMENT (why stratification decides it): when the reduced
                // ordinary program (`not know R` -> `not R`, `know R` -> `R`) is
                // STRATIFIED, its perfect model is TOTAL and 2-valued. A total
                // 2-valued model makes every modal target R 2-valued, so under FAEEL
                // `know R == possible R == R` and `not know R == not possible R == not
                // R` (the modal op stops mattering once R is determined -- the same
                // equivalence example 29 proves for DETERMINED targets, generalized
                // here to STRATIFIED targets). Replacing each modal by its ordinary
                // atom therefore preserves truth values, so the stratified perfect
                // model of the reduced program IS the FAEEL model. The 2-valued
                // (stratified) property is the linchpin.
                //
                // When the reduced program is NOT stratified (a cycle through the
                // negation), the sound semantics is the 3-valued WELL-FOUNDED model
                // (R partly UNDEFINED), which needs the host-side WFS / stable-model
                // solver -- precluded by the no-host-solver lock. Decision deferred to
                // the post-scan reduce + stratification check.
                saw_negated_non_invariant_modal = true;
                saw_case_b = true;
                continue;
            }

            if modal.op == EpistemicOp::Possible && mode == EpistemicMode::G91 {
                // G91 `possible` over a co-evolving target is the autoepistemic
                // self-fulfilling fixpoint; the monotone least-fixpoint resolve would
                // silently compute the FAEEL founded answer under a G91 pragma. Fail
                // closed as an honest wall rather than return a silently-wrong result.
                return Err(recursive_epistemic_rejection(&format!(
                    "rule for `{}` uses the modal literal `possible {}` over a relation that \
                     CO-EVOLVES with the program's ordinary recursion, under G91 mode. G91 \
                     `possible` recursion is the autoepistemic self-fulfilling fixpoint, which \
                     the monotone founded-least-fixpoint reduction cannot express. Use FAEEL \
                     mode (founded least fixpoint) or remove the recursion.",
                    rule.head.predicate, modal.atom.predicate,
                )));
            }

            // POSITIVE `know` (any mode) or FAEEL `possible` over a co-evolving target:
            // admissible Case B. The founded least fixpoint computes the result.
            saw_case_b = true;
        }
    }

    // WALL A1 DISCRIMINATOR: a deferred negated-modal-over-non-invariant is admissible
    // IFF the negation is STRATIFIED. Decide it by the SOLE arbiter of the existing
    // stratification analysis on the REDUCED ordinary program: the reduction maps
    // `not know R` -> ordinary `not R` and `know R` -> `R`, so running
    // `analyze_stratification` over the reduced program detects whether any negation
    // edge closes a cycle. No negation cycle => stratified => admit (Case B, founded =
    // perfect = well-founded model). A negation cycle => non-stratified => the sound
    // 3-valued well-founded model requires the host-only WFS solver (no-host-solver
    // lock), so fail closed with the formal architectural bound. The reduce here is
    // pure analysis (the same reduction the executor applies) and is only reached when
    // a negated non-invariant modal was actually seen.
    if saw_negated_non_invariant_modal {
        let reduced = reduce_case_a_epistemic_program_to_ordinary(program);
        let strat = crate::stratify::analyze_stratification(&reduced);
        if !strat.non_monotone_sccs.is_empty() {
            // Identify a predicate in a negation cycle for a precise diagnostic. Pick the
            // lexicographically smallest predicate across all non-monotone SCCs so the
            // diagnostic is DETERMINISTIC (the SCC set is HashSet-derived, so naive
            // iteration order is non-deterministic).
            let cyclic_pred = strat
                .non_monotone_sccs
                .iter()
                .filter_map(|&scc_idx| strat.sccs.get(scc_idx))
                .flatten()
                .min()
                .cloned()
                .unwrap_or_else(|| "<unknown>".to_string());
            return Err(recursive_epistemic_rejection(&format!(
                "a NEGATED modal literal over a non-invariant relation forms a cycle through \
                 negation (predicate `{cyclic_pred}` participates in a non-monotone SCC of the \
                 reduced program); the recursion is NOT stratified. Its sound semantics is the \
                 3-valued well-founded model (atoms partly UNDEFINED), which requires a \
                 host-side well-founded / stable-model solver. The GPU production path provides \
                 no host-side semantic solver (the no-host-solver architectural constraint), so \
                 this case is bounded by that constraint, not by an unimplemented feature. \
                 Stratify the program (give the negated modal target a strictly lower stratum) \
                 or remove the negation cycle."
            )));
        }
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
    // programs -- examples 10/34/35/36 -- never reach here; they classify NonRecursive
    // and run the constraint kernel on the single-pass path.)
    let has_epistemic_constraint = program.constraints.iter().any(|constraint| {
        constraint
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
    });
    if has_epistemic_constraint {
        return Err(recursive_epistemic_rejection(
            "a recursive epistemic program carries an epistemic integrity constraint \
             (`:- know ...` / `:- not know ...`). Recursive epistemic programs execute \
             through the ordinary semi-naive engine, which does not run the world-view \
             constraint kernel, and the recursive reduction would silently DROP the \
             modal constraint -- yielding a result that ignores it. To keep results \
             sound this fails closed rather than silently dropping the constraint. \
             Remove the recursion or express the integrity constraint over a \
             non-recursive (single-pass) epistemic relation.",
        ));
    }

    if saw_case_b {
        Ok(RecursiveEpistemicClass::CaseB)
    } else {
        Ok(RecursiveEpistemicClass::CaseA)
    }
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
        if !self.derived_heads.contains(predicate) {
            // Pure EDB relation: invariant by construction.
            return true;
        }
        if self.epistemic_heads.contains(predicate) {
            // Definition itself uses a modal literal: not a fixed lower stratum.
            return false;
        }
        match self.ordinary_deps.get(predicate) {
            None => true,
            Some(deps) => deps.iter().all(|dep| self.is_invariant_inner(dep, seen)),
        }
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
        EirBodyLiteral::Constraint | EirBodyLiteral::Binding => true,
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
/// This preserves the W2.x/W38-B planner contract for reduced ordinary bodies:
/// cardinality, selectivity, heat, prefix-degree, sorted-layout, and
/// helper-splitting decisions are owned by the existing production compiler
/// pipeline rather than by an epistemic side planner.
pub fn compile_epistemic_gpu_execution_with_stats_snapshot(
    program: &Program,
    stats_snapshot: Option<&StatsSnapshot>,
) -> Result<EpistemicExecutablePlan> {
    compile_epistemic_gpu_execution_inner(program, stats_snapshot, false)
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
    let gpu_plan = plan_epistemic_gpu_execution(program)?;
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
    let reduced_program = reduce_epistemic_program_to_ordinary(program);
    let mut compiler = Compiler::new();
    let reduced_runtime_plan =
        compiler.compile_program_with_stats_snapshot(&reduced_program, stats_snapshot)?;
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

/// Validate a Case-A recursive epistemic program and return its ordinary reduction.
///
/// This is the Case-A counterpart to [`compile_epistemic_gpu_execution`]: instead of
/// building a single-pass GPU world-view plan, it proves the program is admissible
/// Case A and resolves it to an ordinary recursive program for the existing fixpoint
/// engine. Validation still flows through the EIR boundary ([`build_eir`]) via
/// [`classify_recursive_epistemic_program`], which already requires EVERY modal literal
/// to range over an INVARIANT relation. A direct modal self-support over the recursive
/// head (`possible p` with `p` the recursive/derived head) ranges over a NON-invariant
/// relation and is therefore rejected as non-Case-A upstream — so unfounded modal
/// self-support never reaches this reduction. Only EXECUTION routes through the
/// ordinary engine.
///
/// Returns `Ok(Some(reduced))` when the program is admissible Case A, `Ok(None)` when
/// the program has no ordinary recursion (the caller should use the single-pass
/// epistemic path), and a typed error for any non-Case-A recursive shape.
pub fn try_reduce_case_a_recursive_epistemic_program(program: &Program) -> Result<Option<Program>> {
    match classify_recursive_epistemic_program(program)? {
        RecursiveEpistemicClass::NonRecursive => Ok(None),
        // Case A and Case B share the same reduction: each positive `know`/`possible`
        // modal is resolved to its ordinary atom. In Case A that atom is invariant (a
        // fixed gated relation); in Case B it co-evolves inside the recursive SCC, so
        // the semi-naive least fixpoint computes the founded co-evolving result. The
        // reduction is identical — only the dependency shape of the resolved relation
        // differs — so both route through the ordinary recursive engine.
        RecursiveEpistemicClass::CaseA | RecursiveEpistemicClass::CaseB => {
            Ok(Some(reduce_case_a_epistemic_program_to_ordinary(program)))
        }
    }
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
    for (constraint_index, constraint) in program.constraints.iter().enumerate() {
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
/// Genuinely UNBOUNDED or UNTYPED structured forms (a `cons` with a non-list
/// tail, a nested structure, a `predref`, or an `aggregate`) carry no fixed,
/// typed column set and stay rejected with a precise finiteness/resource
/// diagnostic -- NOT an "unsupported construct".
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
                return Err(XlogError::ResourceExhausted {
                    context: format!(
                        "modal tuple-key for {predicate} uses a `cons` pattern whose tail length \
                         is not statically fixed, so it has no finite, typed GPU key-column set; \
                         bind it to a fixed-arity list literal `[a, b, ...]` instead"
                    ),
                    estimated_bytes: 0,
                    budget_bytes: 0,
                });
            }
            EirTerm::PredRef(name) => {
                return Err(XlogError::ResourceExhausted {
                    context: format!(
                        "modal tuple-key for {predicate} uses predref `{name}`, which has no \
                         finite, typed GPU key-column encoding"
                    ),
                    estimated_bytes: 0,
                    budget_bytes: 0,
                });
            }
            EirTerm::Aggregate { op, variable } => {
                return Err(XlogError::ResourceExhausted {
                    context: format!(
                        "modal tuple-key for {predicate} uses aggregate `{op}({variable})`, whose \
                         value is not a finite, typed GPU key-column tuple"
                    ),
                    estimated_bytes: 0,
                    budget_bytes: 0,
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
/// cannot express, so it is rejected with a precise finiteness diagnostic.
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
            return Err(XlogError::ResourceExhausted {
                context: format!(
                    "modal tuple-key for {predicate} nests a non-scalar element {element:?} inside \
                     a {shape}; only fixed-arity structures of scalar/Symbol-typed elements have a \
                     finite, typed GPU key-column encoding"
                ),
                estimated_bytes: 0,
                budget_bytes: 0,
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

/// Indices (into `program.rules`) of FAEEL rules that are unfounded by circular modal
/// self-support and must be excluded from the reduced founded-model base.
///
/// A rule qualifies when (a) the program is in FAEEL mode, (b) the rule body contains a
/// modal literal `possible p`/`know p` over the rule's OWN head predicate/arity
/// (direct self-support), (c) that head has NO independent founded support
/// ([`has_independent_founded_support`]) and NO tuple-level founded support
/// ([`has_tuple_level_independent_founded_support`]), and (d) excluding the rule does
/// NOT silently elide a mode-independent safety failure — i.e. the head carries no
/// variable bound ONLY by the self-supporting modal. Condition (d) preserves the clean
/// `UnsafeVariable` honest-exit for pure nonzero self-support (`p(X) :- possible p(X)`)
/// in EVERY mode (G91 rejects it identically): dropping such a rule would replace a
/// precise safety diagnostic with a confusing materialization error.
///
/// Returns indices in ASCENDING order; callers must remove in DESCENDING order to keep
/// the remaining indices stable.
///
/// This is the structural foundedness DECISION; the founded EXTENSION is then computed
/// by the existing GPU world-view validation over the reduced base (no CPU semantic
/// solver). G91 mode never drops, so circular self-support stays accepted there — the
/// drop is exactly the FAEEL-vs-G91 mode difference.
fn faeel_unfounded_self_support_rule_indices(program: &Program) -> Vec<usize> {
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
            // Direct modal self-support over the rule's own head.
            if modal.atom.predicate != eir_rule.head.predicate
                || modal.atom.arity != eir_rule.head.arity
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

/// Return the ordinary runtime program produced after epistemic GPU planning.
///
/// Epistemic literals are removed only for the reduced production runtime
/// dispatch; callers must still plan and certify the explicit epistemic GPU
/// contract before using this reduced program.
///
/// The augmenting positive-modal resolve is gated on INVARIANT targets only (see the
/// body comment): for an invariant `R`, `know R`/`possible R` ranges exactly over
/// `R`'s extension, so resolving the modal into an ordinary join binds the augmented
/// output column WITHOUT leaking — and the GPU EGB-02 membership filter re-gates
/// post hoc. A determined-but-not-invariant target (an epistemic-derived head like a
/// multi-column `r`) is NOT resolved here, so its augmenting output variable stays
/// unbound and the reduced program fails closed at this strict (execution) entry
/// point. See [`reduce_epistemic_program_to_ordinary_for_stratified_schema`] for the
/// schema-only relaxation used by the stratified driver.
pub fn reduce_epistemic_program_to_ordinary(program: &Program) -> Program {
    reduce_epistemic_program_to_ordinary_inner(program, &BTreeSet::new())
}

/// Schema-only reduction for the stratified epistemic driver.
///
/// Identical to [`reduce_epistemic_program_to_ordinary`] EXCEPT it also resolves an
/// augmenting positive modal whose target is epistemically DETERMINED (per
/// [`EpistemicallyDeterminedPredicates::analyze`]) but not invariant — e.g. a
/// multi-column determined head `r` in `out(X) :- node(X), know r(X, Y)`, where the
/// modal binds the augmented output column `Y`. This is used SOLELY to compute the
/// plan-wide relation SCHEMAS (column types/arities) for an
/// [`crate::EpistemicStratifiedPlan`]; the resolved positive atom over `r` supplies
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
pub fn reduce_epistemic_program_to_ordinary_for_stratified_schema(program: &Program) -> Program {
    let determined = EpistemicallyDeterminedPredicates::analyze(program);
    reduce_epistemic_program_to_ordinary_inner(program, &determined.determined)
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
) -> Program {
    let mut reduced = program.clone();

    // FAEEL FOUNDED-MODEL EXTENSION: a rule whose head is supported ONLY by circular
    // modal self-support (`possible p`/`know p` over its own head, with no independent
    // founded derivation) contributes nothing to the FAEEL founded model. Excluding the
    // rule from the reduced ordinary base is precisely the founded/equilibrium
    // semantics: the unfounded head is absent from the model rather than fabricated by
    // the stripped-modal `1=1` filler (which would wrongly found it, the G91 answer).
    //
    // This is the structural foundedness DECISION (compile-time, reusing the exact
    // `has_independent_founded_support` / `has_tuple_level_independent_founded_support`
    // predicates the legacy guard used) driving the EXTENSION COMPUTATION on the
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
    for index in faeel_unfounded_self_support_rule_indices(program)
        .into_iter()
        .rev()
    {
        reduced.rules.remove(index);
    }

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
    // EGB-02 membership filter then re-gates them against the accepted world view.
    //
    // STRICTLY SCOPED to keep the prohibition on resolving over still-modal relations
    // machine-checked: only POSITIVE modals (negated `not know`/`not possible` is an
    // anti-join that does NOT range-restrict, so it is never resolved) over INVARIANT
    // targets (a still-modal / epistemic-derived target is NOT invariant, so it is
    // never resolved — its augmenting variable stays unbound and the reduced program
    // fails closed). Non-augmenting modals keep the original strip-and-gate path, so
    // every existing single/joint pilot (16/18/09/19/21) is untouched.
    let invariant = InvariantRelations::analyze(program);

    // Heads where a positive-invariant modal was ACTUALLY resolved into a positive
    // ordinary atom (i.e. a modal-only-bound output variable was genuinely augmented).
    // ONLY these heads' declarations/queries are reconciled to the augmented arity.
    // `append_body_local_tuple_key_variables_to_head` may spuriously append a
    // modal-local variable that is ALSO positively bound (e.g. `Y` in the recursive
    // `reach(X,Z) :- reach(X,Y), vertex(Z), know a(Y,Z)`), which must NOT trigger a
    // declaration bump — that head is materialized at its original arity by the
    // (Case-A) recursive engine, so bumping its declaration would corrupt the schema.
    let mut resolved_augmented_heads: BTreeSet<String> = BTreeSet::new();

    for rule in &mut reduced.rules {
        // Head variables that NO non-epistemic positive body literal binds. After the
        // modal is stripped, an output (head) variable bound ONLY by the modal would
        // be unsafe in the reduced ordinary program. Computed BEFORE the head is
        // mutated by augmentation. (`append_body_local_tuple_key_variables_to_head`
        // appends modal-local variables to the head, so both already-present head
        // variables like `Y` in `pair(X,Y) :- ..possible edge(X,Y)` AND augmented
        // variables like `Y` in `one_hop(X) :- ..know edge(X,Y)` are covered here.)
        let modal_only_output_variables = modal_only_bound_output_variables(rule);
        append_body_local_tuple_key_variables_to_head(rule);
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
        // candidate tuples and the GPU EGB-02 filter re-gates against the accepted
        // world view. A NEGATED modal (anti-join) never binds and is never resolved; a
        // still-modal / epistemic-derived target is NOT invariant and is never
        // resolved, so its unbound output variable correctly fails closed downstream.
        let mut resolved_here = false;
        for lit in &mut rule.body {
            if let BodyLiteral::Epistemic(modal) = lit {
                // The target is resolvable when it is INVARIANT (always — proven-sound
                // for both schema and execution), OR — for SCHEMA inference only — when
                // it is epistemically DETERMINED. The determined relaxation is empty for
                // the strict execution reduce, so an execution-path reduce never
                // resolves a modal over a still-derived (un-gated) relation.
                let resolvable_target = invariant.is_invariant(&modal.atom.predicate)
                    || schema_only_determined_resolve.contains(&modal.atom.predicate);
                if !modal.negated
                    && resolvable_target
                    && modal_atom_binds_output_variable(modal, &modal_only_output_variables)
                {
                    *lit = BodyLiteral::Positive(modal.atom.clone());
                    resolved_here = true;
                }
            }
        }
        if resolved_here {
            resolved_augmented_heads.insert(rule.head.predicate.clone());
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
    // a schema mismatch. SCOPED to heads where the resolve actually fired (so a
    // spuriously-appended-but-positively-bound recursive head like `reach` is NOT
    // bumped). Infer each appended column's type from the resolved body atom.
    let augmented_heads =
        reconcile_augmented_head_declarations(&mut reduced, &resolved_augmented_heads);

    // Drop reduced-program queries that reference an AUGMENTED head: the reduced
    // relation is now arity-bumped, so an original arity-N query over it would union
    // the arity-N query projection against the augmented relation and fail with a
    // schema mismatch. The user-visible query results for epistemic heads are
    // surfaced separately from the GPU gated buffers (`epistemic_result_to_query_
    // results`, projected to public arity), and the surfacing gate
    // (`queried_predicates`) reads the ORIGINAL program's queries, so dropping the
    // redundant reduced query here is inert for display and only removes the crash.
    // Non-augmented epistemic heads keep their arity-matched reduced queries untouched.
    if !augmented_heads.is_empty() {
        reduced
            .queries
            .retain(|query| !augmented_heads.contains(&query.atom.predicate));
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

    reduced
}

/// Reduce a Case-A recursive epistemic program to an equivalent ordinary recursive
/// program for the existing fixpoint engine.
///
/// Unlike [`reduce_epistemic_program_to_ordinary`] (which strips modal literals and
/// gates the single-pass result post hoc), this RESOLVES each positive `know`/
/// `possible` literal to its gated relation by rewriting it into an ordinary positive
/// body atom over the same predicate. Because the modal relation is invariant (EDB or
/// a lower non-recursive, non-epistemic stratum — proved by
/// [`classify_recursive_epistemic_program`]), its extension is the accepted world
/// view's extension, so the rewrite preserves modal semantics while letting the
/// recursive/semi-naive engine join the recursion against the gated relation at every
/// iteration. The modal atom's variables become ordinary join variables (no hidden
/// head columns are appended), which fixes both the missing in-loop gate and the
/// arity mismatch that make the post-hoc reduction single-pass-only.
///
/// Callers MUST first prove the program is Case A via
/// [`classify_recursive_epistemic_program`]; this function assumes that contract.
pub fn reduce_case_a_epistemic_program_to_ordinary(program: &Program) -> Program {
    let mut reduced = program.clone();
    for rule in &mut reduced.rules {
        for lit in &mut rule.body {
            if let BodyLiteral::Epistemic(modal) = lit {
                // Case A admits modal literals over invariant relations. A positive
                // `know`/`possible` resolves to a positive ordinary atom over its
                // (invariant) gated relation; a NEGATED `not know`/`not possible`
                // over an invariant relation equals ordinary `not R` (an anti-join,
                // no modal gating), so resolve it to a negated ordinary atom.
                *lit = if modal.negated {
                    BodyLiteral::Negated(modal.atom.clone())
                } else {
                    BodyLiteral::Positive(modal.atom.clone())
                };
            }
        }
    }
    // World-view integrity constraints have no place in a Case-A ordinary program: the
    // recursion already joins against the gated relations. Drop any constraint that
    // still references a modal literal (purely relational constraints are retained).
    reduced.constraints.retain(|constraint| {
        !constraint
            .body
            .iter()
            .any(|lit| matches!(lit, BodyLiteral::Epistemic(_)))
    });
    reduced
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
    // Variables bound by a positive non-epistemic body literal (positive atoms,
    // `is`-expressions, and univ all introduce bindings; comparisons and negated atoms
    // do not range-restrict).
    let mut positively_bound: BTreeSet<&str> = BTreeSet::new();
    for lit in &rule.body {
        if let BodyLiteral::Positive(atom) = lit {
            for term in &atom.terms {
                if let Term::Variable(name) = term {
                    positively_bound.insert(name.as_str());
                }
            }
        }
    }

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

/// Widen each predicate's declaration to the maximum arity of its (now possibly
/// augmented) defining rule heads, inferring appended column types from the positive
/// body atoms that bind the augmented head variables.
///
/// Augmentation appends modal-local columns to a rule head; without widening the
/// matching `PredDecl`, the runtime would union the augmented rule output against the
/// narrow declared (empty) relation stub and fail with a schema mismatch.
///
/// Only heads in `resolved_augmented_heads` (where a positive-invariant modal was
/// genuinely resolved into a positive atom, range-restricting a modal-only-bound
/// output variable) are reconciled; a head that merely had a positively-bound
/// modal-local variable spuriously appended is NOT widened.
///
/// Returns the set of head predicates whose declaration was widened (i.e. whose rule
/// heads were augmented beyond the original declared arity).
fn reconcile_augmented_head_declarations(
    reduced: &mut Program,
    resolved_augmented_heads: &BTreeSet<String>,
) -> BTreeSet<String> {
    use crate::ast::{PredColumn, TypeRef};

    let mut augmented_heads = BTreeSet::new();

    // Per head predicate: the maximum rule-head arity and, per column position, an
    // inferred type from a positive body atom (the resolved modal or any binder).
    let mut max_arity: BTreeMap<String, usize> = BTreeMap::new();
    let mut inferred_types: BTreeMap<String, Vec<Option<TypeRef>>> = BTreeMap::new();

    // Map predicate -> declared column types (for type inference of body atom columns).
    let declared_types: BTreeMap<String, Vec<TypeRef>> = reduced
        .predicates
        .iter()
        .map(|decl| (decl.name.clone(), decl.types.clone()))
        .collect();

    for rule in &reduced.rules {
        if rule.body.is_empty() {
            continue;
        }
        // Only heads where the invariant-resolve genuinely fired are reconciled.
        if !resolved_augmented_heads.contains(&rule.head.predicate) {
            continue;
        }
        let head = rule.head.predicate.as_str();
        let arity = rule.head.terms.len();
        let entry = max_arity.entry(head.to_string()).or_insert(0);
        if arity > *entry {
            *entry = arity;
        }
        let types = inferred_types
            .entry(head.to_string())
            .or_insert_with(|| vec![None; arity]);
        if types.len() < arity {
            types.resize(arity, None);
        }
        // Infer each head variable's type from a positive body atom that binds it.
        for (col, term) in rule.head.terms.iter().enumerate() {
            if types[col].is_some() {
                continue;
            }
            let Term::Variable(head_var) = term else {
                continue;
            };
            for lit in &rule.body {
                let BodyLiteral::Positive(atom) = lit else {
                    continue;
                };
                let Some(pos) = atom
                    .terms
                    .iter()
                    .position(|t| matches!(t, Term::Variable(name) if name == head_var))
                else {
                    continue;
                };
                if let Some(decl_types) = declared_types.get(&atom.predicate) {
                    if let Some(typ) = decl_types.get(pos) {
                        types[col] = Some(typ.clone());
                        break;
                    }
                }
            }
        }
    }

    for decl in &mut reduced.predicates {
        let Some(&target_arity) = max_arity.get(&decl.name) else {
            continue;
        };
        if target_arity <= decl.types.len() {
            continue;
        }
        let inferred = inferred_types.get(&decl.name);
        for col in decl.types.len()..target_arity {
            let typ = inferred
                .and_then(|types| types.get(col))
                .and_then(|t| t.clone())
                // Default appended columns to U32 (the modal relation key column type).
                .unwrap_or(TypeRef::Scalar(xlog_core::ScalarType::U32));
            decl.types.push(typ.clone());
            decl.columns.push(PredColumn { name: None, typ });
        }
        augmented_heads.insert(decl.name.clone());
    }

    augmented_heads
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
    relational_body_atoms: usize,
    has_negated_relational_atom: bool,
) -> EpistemicWcojReductionStatus {
    if relational_body_atoms >= 3 && !has_negated_relational_atom {
        EpistemicWcojReductionStatus::RequiresPlannerEligibility
    } else {
        EpistemicWcojReductionStatus::NotWcojCandidate
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
/// independently of one another (K3 split diagnostics).
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
    /// Original bounded split certificate.
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
    reject_gpt_epistemic_constraints(program)?;
    if candidates.len() > config.max_candidates {
        return Err(xlog_core::XlogError::ResourceExhausted {
            context: "epistemic GPT candidate guard".to_string(),
            estimated_bytes: candidates.len() as u64,
            budget_bytes: config.max_candidates as u64,
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
        match evaluate_epistemic_candidate(program, candidate, mode)? {
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
/// lower stratum's head reads the GATED extension through the EXISTING EGB-02
/// membership filter (no resolve-into-body, no double-gating).
#[derive(Debug, Clone)]
pub struct EpistemicStratifiedPlan {
    /// Strata in execution (topological) order.
    pub strata: Vec<EpistemicStratum>,
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
        // base `r` via the existing EGB-02 membership filter. The acyclicity guard in
        // `head_is_determined` (self-reference returns false) plus the fixpoint's
        // monotonicity keep every recursive predicate OUT of `determined`, so a
        // circular `know reach` in a recursive SCC (example 22) is never determined
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
/// via the existing EGB-02 membership filter — never via resolve-into-body.
///
/// Returns:
/// - `Ok(Some(plan))` when the program genuinely needs (and admits) stratification:
///   at least one modal literal ranges over an epistemically-determined derived
///   head, and a sound stratification exists.
/// - `Ok(None)` when no modal ranges over a determined derived head (the existing
///   joint/split/single paths own the program — e.g. example 18's shared base
///   modal, where the modal target `q` is EDB, not a determined derived head), OR
///   when the nested target is NOT determined (circular modality / recursion /
///   unfounded self-support is handed back to the recursive + FAEEL/G91 guards,
///   which keep ownership and fail closed there).
pub fn try_plan_stratified_epistemic_program(
    program: &Program,
) -> Result<Option<EpistemicStratifiedPlan>> {
    let determined = EpistemicallyDeterminedPredicates::analyze(program);

    // A stratification is needed only when some modal literal ranges over a
    // DETERMINED epistemic-derived head. (A modal over a base/EDB predicate is the
    // ordinary single/joint path — example 18 — and must NOT be intercepted.)
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

    Ok(Some(EpistemicStratifiedPlan { strata }))
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

    // Keep only the queries whose predicate this stratum materializes, so each
    // stratum's executable surfaces its own head(s).
    let head_set: BTreeSet<&str> = head_predicates.iter().map(String::as_str).collect();
    stratum.queries = program
        .queries
        .iter()
        .filter(|query| head_set.contains(query.atom.predicate.as_str()))
        .cloned()
        .collect();

    // Drop constraints that reference predicates this stratum does not own, to keep
    // the sub-program self-contained.
    stratum.constraints = program
        .constraints
        .iter()
        .filter(|constraint| {
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
    // EGB-06: rules that couple more than one distinct epistemic body predicate
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
