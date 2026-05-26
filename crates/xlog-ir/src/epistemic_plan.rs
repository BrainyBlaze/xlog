//! GPU-native epistemic execution planning contracts.

use std::collections::{BTreeMap, BTreeSet};

use xlog_core::RelId;

use crate::eir::{EirEpistemicLiteral, EirEpistemicMode, EirEpistemicOp, EirTerm};
use crate::plan::ExecutionPlan;

/// Generate-Propagate-Test hot-path phase that must execute on GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicGpuHotPathPhase {
    /// Candidate epistemic assumptions are generated on device.
    CandidateGeneration,
    /// Candidate assumptions are propagated into reduced programs on device.
    Propagation,
    /// Candidate bitsets are validated on device before production dispatch.
    CandidateValidation,
    /// Stable-model tuple membership is populated on device.
    ModelMembership,
    /// Reduced-program stable models are checked against world-view guesses on device.
    WorldViewValidation,
    /// Accepted world views and query results are materialized from device buffers.
    ResultMaterialization,
    /// Final result flags are materialized from device-side output metadata.
    FinalResultMaterialization,
    /// Final query tuples are materialized into a device-resident output buffer.
    FinalTupleMaterialization,
}

/// GPU-resident buffer category required by accepted epistemic execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicGpuBufferKind {
    /// Candidate assumption bitsets.
    CandidateAssumptions,
    /// Accepted and candidate world-view bitsets.
    WorldViews,
    /// Per-model membership checks used by `know` and `possible`.
    ModelMembership,
    /// Structured rejection reasons for failed candidates.
    RejectionReasons,
}

/// WCOJ status for a reduced ordinary program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpistemicWcojReductionStatus {
    /// The reduced body is too small or otherwise not a WCOJ candidate.
    NotWcojCandidate,
    /// The reduced body must be submitted to the production WCOJ planner.
    RequiresPlannerEligibility,
}

/// CPU fallback counters that must remain zero on the accepted hot path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EpistemicCpuFallbackCounters {
    /// CPU candidate enumeration count.
    pub candidate_enumeration: u64,
    /// CPU world-view validation count.
    pub world_view_validation: u64,
    /// CPU SAT/MaxSAT search count.
    pub solver_search: u64,
    /// CPU-only probabilistic recomputation count.
    pub probabilistic_recompute: u64,
}

impl EpistemicCpuFallbackCounters {
    /// Return true when every forbidden CPU fallback counter is zero.
    pub fn is_zero(&self) -> bool {
        self.candidate_enumeration == 0
            && self.world_view_validation == 0
            && self.solver_search == 0
            && self.probabilistic_recompute == 0
    }
}

/// One epistemic rule's reduced ordinary-program planning summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicReductionPlan {
    /// Source-order rule index.
    pub rule_index: usize,
    /// Head predicate materialized by the reduced production runtime plan.
    pub head_predicate: String,
    /// Positive relational body atom count after removing epistemic literals.
    pub relational_body_atoms: usize,
    /// WCOJ planner status for the reduced ordinary body.
    pub wcoj_status: EpistemicWcojReductionStatus,
}

/// Binding from an epistemic literal to reduced stable-model tuple evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicTupleMembershipBinding {
    /// Index of the epistemic literal in `EpistemicGpuPlan::epistemic_literals`.
    pub literal_index: usize,
    /// Index of the reduced rule in `EpistemicGpuPlan::reductions`.
    pub reduction_index: usize,
    /// Predicate whose stable-model tuples must be checked.
    pub predicate: String,
    /// Predicate arity whose stable-model tuples must be checked.
    pub arity: usize,
    /// Source relation columns that form the tuple key for this epistemic atom.
    pub key_columns: Vec<usize>,
    /// Source atom terms that must be matched against the stable-model tuple key.
    pub key_terms: Vec<EirTerm>,
    /// Reduced output column for each variable tuple-key term.
    ///
    /// Ground terms use `None`; variable terms use `Some(column_index)`.
    pub bound_output_columns: Vec<Option<usize>>,
    /// Epistemic operator whose membership semantics are being checked.
    pub op: EirEpistemicOp,
    /// Whether the epistemic literal is explicitly negated.
    pub negated: bool,
}

/// Solver production capability required by accepted epistemic execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EpistemicSolverCapability {
    /// Incremental SAT solve calls with pushed assumptions.
    IncrementalSat,
    /// Explicit push, solve, retract assumption lifecycle.
    AssumptionLifecycle,
    /// Learned-clause publication and reuse across valid incremental calls.
    LearnedClauseTransfer,
    /// Weighted MaxSAT soft-constraint solving.
    WeightedMaxSat,
    /// GPU-backed SAT/MaxSAT portfolio dispatch.
    PortfolioSatMaxSat,
}

/// Solver status kind that must cross the epistemic boundary distinctly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EpistemicSolverStatusKind {
    /// Satisfiable solver result.
    Sat,
    /// Unsatisfiable solver result.
    Unsat,
    /// Inconclusive solver result.
    Unknown,
    /// Budget-exhausted solver result.
    Timeout,
}

/// Binding from an epistemic literal to a solver assumption obligation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicSolverAssumptionBinding {
    /// Index of the epistemic literal in `EpistemicGpuPlan::epistemic_literals`.
    pub literal_index: usize,
    /// Index of the reduced rule in `EpistemicGpuPlan::reductions`.
    pub reduction_index: usize,
    /// Predicate whose epistemic truth becomes a solver assumption.
    pub predicate: String,
    /// Predicate arity for the solver assumption.
    pub arity: usize,
    /// Source atom terms that define the solver assumption key.
    pub terms: Vec<EirTerm>,
    /// Epistemic operator represented by the assumption.
    pub op: EirEpistemicOp,
    /// Whether the epistemic literal is explicitly negated.
    pub negated: bool,
}

/// Solver-service contract exported from the epistemic semantic plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicSolverServiceContract {
    /// Per-literal solver assumptions that must be pushed and retracted.
    pub assumption_bindings: Vec<EpistemicSolverAssumptionBinding>,
    /// Production solver capabilities required before this plan can count as accepted.
    pub required_capabilities: Vec<EpistemicSolverCapability>,
    /// Solver statuses that must remain distinct across the interface.
    pub required_statuses: Vec<EpistemicSolverStatusKind>,
}

impl EpistemicSolverServiceContract {
    /// Build the v0.9 production solver contract for the provided assumptions.
    pub fn production_default(assumption_bindings: Vec<EpistemicSolverAssumptionBinding>) -> Self {
        Self {
            assumption_bindings,
            required_capabilities: vec![
                EpistemicSolverCapability::IncrementalSat,
                EpistemicSolverCapability::AssumptionLifecycle,
                EpistemicSolverCapability::LearnedClauseTransfer,
                EpistemicSolverCapability::WeightedMaxSat,
                EpistemicSolverCapability::PortfolioSatMaxSat,
            ],
            required_statuses: vec![
                EpistemicSolverStatusKind::Sat,
                EpistemicSolverStatusKind::Unsat,
                EpistemicSolverStatusKind::Unknown,
                EpistemicSolverStatusKind::Timeout,
            ],
        }
    }

    /// Count distinct required solver capabilities.
    pub fn distinct_required_capability_count(&self) -> usize {
        self.required_capabilities
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Count distinct solver statuses that must cross the semantic boundary.
    pub fn distinct_required_status_count(&self) -> usize {
        self.required_statuses
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
    }
}

/// Production-facing GPU execution contract for an epistemic program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicGpuPlan {
    /// Selected epistemic semantics mode.
    pub mode: EirEpistemicMode,
    /// Epistemic literals preserved from EIR.
    pub epistemic_literals: Vec<EirEpistemicLiteral>,
    /// Coarse Generate-Propagate-Test phases required by the hot path.
    pub required_phases: Vec<EpistemicGpuHotPathPhase>,
    /// Concrete GPU kernel phases required by accepted production execution.
    pub required_kernel_phases: Vec<EpistemicGpuHotPathPhase>,
    /// GPU buffer classes required by the hot path.
    pub required_buffers: Vec<EpistemicGpuBufferKind>,
    /// Reduced ordinary-program planning summaries.
    pub reductions: Vec<EpistemicReductionPlan>,
    /// Per-literal stable-model tuple membership bindings.
    pub tuple_membership_bindings: Vec<EpistemicTupleMembershipBinding>,
    /// Reduced-output columns copied into the public final output.
    /// `None` means identity/all columns; `Some([])` is a real zero-arity projection.
    pub final_output_columns: Option<Vec<usize>>,
    /// Solver-service obligations exported by the epistemic semantic plan.
    pub solver_contract: EpistemicSolverServiceContract,
    /// Forbidden CPU fallback counters. Release certification must keep these zero.
    pub cpu_fallbacks: EpistemicCpuFallbackCounters,
}

impl EpistemicGpuPlan {
    /// Create a plan with the standard GPU hot-path phase and buffer requirements.
    pub fn new(
        mode: EirEpistemicMode,
        epistemic_literals: Vec<EirEpistemicLiteral>,
        reductions: Vec<EpistemicReductionPlan>,
    ) -> Self {
        let tuple_membership_bindings = epistemic_literals
            .iter()
            .enumerate()
            .map(|(literal_index, literal)| EpistemicTupleMembershipBinding {
                literal_index,
                reduction_index: literal_index.min(reductions.len().saturating_sub(1)),
                predicate: literal.atom.predicate.clone(),
                arity: literal.atom.arity,
                key_columns: (0..literal.atom.arity).collect(),
                key_terms: literal.atom.terms.clone(),
                bound_output_columns: vec![None; literal.atom.arity],
                op: literal.op,
                negated: literal.negated,
            })
            .collect();
        let solver_assumption_bindings = epistemic_literals
            .iter()
            .enumerate()
            .map(
                |(literal_index, literal)| EpistemicSolverAssumptionBinding {
                    literal_index,
                    reduction_index: literal_index.min(reductions.len().saturating_sub(1)),
                    predicate: literal.atom.predicate.clone(),
                    arity: literal.atom.arity,
                    terms: literal.atom.terms.clone(),
                    op: literal.op,
                    negated: literal.negated,
                },
            )
            .collect();

        Self {
            mode,
            epistemic_literals,
            required_phases: vec![
                EpistemicGpuHotPathPhase::CandidateGeneration,
                EpistemicGpuHotPathPhase::Propagation,
                EpistemicGpuHotPathPhase::WorldViewValidation,
                EpistemicGpuHotPathPhase::ResultMaterialization,
            ],
            required_kernel_phases: vec![
                EpistemicGpuHotPathPhase::CandidateGeneration,
                EpistemicGpuHotPathPhase::Propagation,
                EpistemicGpuHotPathPhase::CandidateValidation,
                EpistemicGpuHotPathPhase::ModelMembership,
                EpistemicGpuHotPathPhase::WorldViewValidation,
                EpistemicGpuHotPathPhase::ResultMaterialization,
                EpistemicGpuHotPathPhase::FinalResultMaterialization,
                EpistemicGpuHotPathPhase::FinalTupleMaterialization,
            ],
            required_buffers: vec![
                EpistemicGpuBufferKind::CandidateAssumptions,
                EpistemicGpuBufferKind::WorldViews,
                EpistemicGpuBufferKind::ModelMembership,
                EpistemicGpuBufferKind::RejectionReasons,
            ],
            reductions,
            tuple_membership_bindings,
            final_output_columns: None,
            solver_contract: EpistemicSolverServiceContract::production_default(
                solver_assumption_bindings,
            ),
            cpu_fallbacks: EpistemicCpuFallbackCounters::default(),
        }
    }

    /// Replace inferred tuple-membership bindings with planner-derived bindings.
    pub fn with_tuple_membership_bindings(
        mut self,
        tuple_membership_bindings: Vec<EpistemicTupleMembershipBinding>,
    ) -> Self {
        self.tuple_membership_bindings = tuple_membership_bindings;
        self
    }

    /// Set the public projection applied after GPU tuple membership row filtering.
    pub fn with_final_output_columns(mut self, final_output_columns: Option<Vec<usize>>) -> Self {
        self.final_output_columns = final_output_columns;
        self
    }

    /// Replace inferred solver obligations with planner-derived obligations.
    pub fn with_solver_contract(mut self, solver_contract: EpistemicSolverServiceContract) -> Self {
        self.solver_contract = solver_contract;
        self
    }

    /// Validate that solver obligations match the epistemic semantic boundary.
    pub fn validate_solver_contract(&self) -> xlog_core::Result<()> {
        let contract = &self.solver_contract;
        if contract.assumption_bindings.len() != self.epistemic_literals.len() {
            return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic solver service contract".to_string(),
                context: format!(
                    "expected {} solver assumption bindings for epistemic literals, found {}",
                    self.epistemic_literals.len(),
                    contract.assumption_bindings.len()
                ),
            });
        }

        let distinct_capability_count = contract.distinct_required_capability_count();
        if distinct_capability_count != contract.required_capabilities.len() {
            return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic solver service contract".to_string(),
                context: format!(
                    "solver capability requirements must be distinct, got {} entries but {} distinct",
                    contract.required_capabilities.len(),
                    distinct_capability_count
                ),
            });
        }

        let distinct_status_count = contract.distinct_required_status_count();
        if distinct_status_count != contract.required_statuses.len() {
            return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic solver service contract".to_string(),
                context: format!(
                    "solver status requirements must be distinct, got {} entries but {} distinct",
                    contract.required_statuses.len(),
                    distinct_status_count
                ),
            });
        }

        for required in [
            EpistemicSolverCapability::IncrementalSat,
            EpistemicSolverCapability::AssumptionLifecycle,
            EpistemicSolverCapability::LearnedClauseTransfer,
            EpistemicSolverCapability::WeightedMaxSat,
            EpistemicSolverCapability::PortfolioSatMaxSat,
        ] {
            if !contract.required_capabilities.contains(&required) {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic solver service contract".to_string(),
                    context: format!("missing required solver capability {required:?}"),
                });
            }
        }

        for required in [
            EpistemicSolverStatusKind::Sat,
            EpistemicSolverStatusKind::Unsat,
            EpistemicSolverStatusKind::Unknown,
            EpistemicSolverStatusKind::Timeout,
        ] {
            if !contract.required_statuses.contains(&required) {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic solver service contract".to_string(),
                    context: format!("missing required solver status {required:?}"),
                });
            }
        }

        let mut seen_literals = vec![false; self.epistemic_literals.len()];
        for binding in &contract.assumption_bindings {
            if binding.literal_index >= self.epistemic_literals.len() {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic solver service contract".to_string(),
                    context: format!(
                        "literal_index {} exceeds literal count {}",
                        binding.literal_index,
                        self.epistemic_literals.len()
                    ),
                });
            }
            if seen_literals[binding.literal_index] {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic solver service contract".to_string(),
                    context: format!(
                        "duplicate solver assumption for literal_index {}",
                        binding.literal_index
                    ),
                });
            }
            seen_literals[binding.literal_index] = true;

            if binding.reduction_index >= self.reductions.len() {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic solver service contract".to_string(),
                    context: format!(
                        "reduction_index {} exceeds reduction count {}",
                        binding.reduction_index,
                        self.reductions.len()
                    ),
                });
            }

            let literal = &self.epistemic_literals[binding.literal_index];
            let tuple_binding =
                self.tuple_membership_bindings
                    .iter()
                    .find(|tuple_binding| tuple_binding.literal_index == binding.literal_index)
                    .ok_or_else(|| {
                        xlog_core::XlogError::UnsupportedEpistemicConstruct {
                            construct: "epistemic solver service contract".to_string(),
                            context: format!(
                                "solver assumption for literal_index {} has no matching tuple-membership binding",
                                binding.literal_index
                            ),
                        }
                    })?;
            if binding.reduction_index != tuple_binding.reduction_index {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic solver service contract".to_string(),
                    context: format!(
                        "solver assumption for literal_index {} uses reduction_index {}, but tuple membership uses {}",
                        binding.literal_index,
                        binding.reduction_index,
                        tuple_binding.reduction_index
                    ),
                });
            }
            if binding.predicate != literal.atom.predicate
                || binding.arity != literal.atom.arity
                || binding.terms != literal.atom.terms
                || binding.op != literal.op
                || binding.negated != literal.negated
            {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic solver service contract".to_string(),
                    context: format!(
                        "solver assumption for literal_index {} does not match epistemic literal",
                        binding.literal_index
                    ),
                });
            }
        }

        Ok(())
    }

    /// Validate that every epistemic literal has a matching tuple-membership binding.
    pub fn validate_tuple_membership_bindings(&self) -> xlog_core::Result<()> {
        if self.tuple_membership_bindings.len() != self.epistemic_literals.len() {
            return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                construct: "epistemic GPU tuple membership binding".to_string(),
                context: format!(
                    "expected {} bindings for epistemic literals, found {}",
                    self.epistemic_literals.len(),
                    self.tuple_membership_bindings.len()
                ),
            });
        }

        let mut seen_literals = vec![false; self.epistemic_literals.len()];

        for binding in &self.tuple_membership_bindings {
            if binding.literal_index >= self.epistemic_literals.len() {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU tuple membership binding".to_string(),
                    context: format!(
                        "literal_index {} exceeds literal count {}",
                        binding.literal_index,
                        self.epistemic_literals.len()
                    ),
                });
            }
            if seen_literals[binding.literal_index] {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU tuple membership binding".to_string(),
                    context: format!(
                        "duplicate literal_index {} in tuple-membership bindings",
                        binding.literal_index
                    ),
                });
            }
            seen_literals[binding.literal_index] = true;

            if binding.reduction_index >= self.reductions.len() {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU tuple membership binding".to_string(),
                    context: format!(
                        "reduction_index {} exceeds reduction count {}",
                        binding.reduction_index,
                        self.reductions.len()
                    ),
                });
            }

            let literal = &self.epistemic_literals[binding.literal_index];
            if binding.predicate != literal.atom.predicate
                || binding.arity != literal.atom.arity
                || binding.op != literal.op
                || binding.negated != literal.negated
            {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU tuple membership binding".to_string(),
                    context: format!(
                        "binding for literal_index {} does not match epistemic literal",
                        binding.literal_index
                    ),
                });
            }

            if binding.key_columns.len() != binding.arity {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU tuple membership binding".to_string(),
                    context: format!(
                        "binding for literal_index {} has {} key columns for arity {}",
                        binding.literal_index,
                        binding.key_columns.len(),
                        binding.arity
                    ),
                });
            }

            if binding.key_terms.len() != binding.arity {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU tuple membership binding".to_string(),
                    context: format!(
                        "binding for literal_index {} has {} key terms for arity {}",
                        binding.literal_index,
                        binding.key_terms.len(),
                        binding.arity
                    ),
                });
            }

            if binding.key_terms != literal.atom.terms {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU tuple membership binding".to_string(),
                    context: format!(
                        "key terms for literal_index {} do not match epistemic literal",
                        binding.literal_index
                    ),
                });
            }

            if binding.bound_output_columns.len() != binding.arity {
                return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                    construct: "epistemic GPU tuple membership binding".to_string(),
                    context: format!(
                        "binding for literal_index {} has {} bound output columns for arity {}",
                        binding.literal_index,
                        binding.bound_output_columns.len(),
                        binding.arity
                    ),
                });
            }

            for (term, bound_col) in binding
                .key_terms
                .iter()
                .zip(binding.bound_output_columns.iter())
            {
                match (term, bound_col) {
                    (EirTerm::Variable(_), Some(_)) => {}
                    (EirTerm::Variable(variable), None) => {
                        return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                            construct: "epistemic GPU tuple membership binding".to_string(),
                            context: format!(
                                "variable tuple key {variable} for literal_index {} is missing a \
                                 reduced output column",
                                binding.literal_index
                            ),
                        });
                    }
                    (_, None) => {}
                    (_, Some(bound_col)) => {
                        return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                            construct: "epistemic GPU tuple membership binding".to_string(),
                            context: format!(
                                "ground tuple key for literal_index {} unexpectedly binds \
                                 reduced output column {}",
                                binding.literal_index, bound_col
                            ),
                        });
                    }
                }
            }

            let mut seen_key_columns = vec![false; binding.arity];
            for &key_col in &binding.key_columns {
                if key_col >= binding.arity {
                    return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                        construct: "epistemic GPU tuple membership binding".to_string(),
                        context: format!(
                            "key column {} exceeds arity {} for literal_index {}",
                            key_col, binding.arity, binding.literal_index
                        ),
                    });
                }
                if seen_key_columns[key_col] {
                    return Err(xlog_core::XlogError::UnsupportedEpistemicConstruct {
                        construct: "epistemic GPU tuple membership binding".to_string(),
                        context: format!(
                            "duplicate key column {} for literal_index {}",
                            key_col, binding.literal_index
                        ),
                    });
                }
                seen_key_columns[key_col] = true;
            }
        }

        Ok(())
    }
}

/// Production-facing executable plan for accepted epistemic lowering.
#[derive(Debug, Clone)]
pub struct EpistemicExecutablePlan {
    /// GPU semantic contract for the epistemic hot path.
    pub gpu_plan: EpistemicGpuPlan,
    /// Predicate-to-relation ID map produced by the reduced production compiler.
    pub relation_ids: BTreeMap<String, RelId>,
    /// Ordinary reduced program compiled through the production runtime pipeline.
    pub reduced_runtime_plan: ExecutionPlan,
}
