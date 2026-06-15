//! Bounded solver-service interface for v0.9 semantics.

use std::cell::RefCell;

use xlog_core::{Result, XlogError};

use crate::{Clause, Literal, Objective, SolveInstance};

/// Status returned by the solver-service interface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolverServiceStatus {
    /// The hard constraints are satisfiable.
    Sat,
    /// The hard constraints are unsatisfiable.
    Unsat,
    /// Search was not attempted or no authoritative backend is available.
    Unknown,
    /// Search was bounded before any assignment could be inspected.
    Timeout,
    /// MaxSAT optimum as an integer score.
    Optimal(u64),
}

/// Search budget for bounded service solves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverServiceBudget {
    /// Do not search; return `Unknown`.
    NoSearch,
    /// Inspect at most this many assignments.
    AssignmentLimit(u64),
    /// Exhaustively inspect the assignment space.
    Exhaustive,
}

/// Trace counters for service-level behavior.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SolverServiceTrace {
    /// Number of learned clauses received from another service.
    pub learned_clause_transfers: usize,
    /// CPU assignment candidates inspected by the semantic-oracle facade.
    pub cpu_assignment_enumerations: u64,
    /// CPU MaxSAT candidates inspected by the semantic-oracle facade.
    pub cpu_maxsat_enumerations: u64,
}

impl SolverServiceTrace {
    /// Reject attempts to use the CPU semantic-oracle facade as production metric evidence.
    pub fn require_production_metric_eligibility(&self) -> Result<()> {
        Err(XlogError::UnsupportedEpistemicConstruct {
            construct: "CPU semantic-oracle solver service".to_string(),
            context: format!(
                "SolverService is fixture-only and cannot satisfy production solver metrics; \
                 cpu_assignment_enumerations={} cpu_maxsat_enumerations={} \
                 learned_clause_transfers={}",
                self.cpu_assignment_enumerations,
                self.cpu_maxsat_enumerations,
                self.learned_clause_transfers
            ),
        })
    }
}

/// Result returned by the solver service.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverServiceResult {
    /// Solver-service status.
    pub status: SolverServiceStatus,
    /// Assignment when available.
    pub assignment: Option<Vec<bool>>,
}

/// Learned-clause transfer result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LearnedClauseTransfer {
    /// Number of clauses transferred.
    pub clauses: usize,
}

/// GPU portfolio status for the semantic-oracle facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverPortfolioStatus {
    /// GPU portfolio solving is unavailable with rationale.
    Deferred {
        /// Deferral rationale.
        reason: &'static str,
    },
}

/// Incremental SAT/MaxSAT service facade.
pub struct SolverService {
    instance: SolveInstance,
    assumptions: Vec<Option<Literal>>,
    learned_clauses: RefCell<Vec<ScopedLearnedClause>>,
    trace: RefCell<SolverServiceTrace>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ScopedLearnedClause {
    clause: Clause,
    assumption_scope: Vec<Literal>,
}

impl SolverService {
    /// Create a service for an instance.
    pub fn new(instance: SolveInstance) -> Self {
        Self {
            instance,
            assumptions: Vec::new(),
            learned_clauses: RefCell::new(Vec::new()),
            trace: RefCell::new(SolverServiceTrace::default()),
        }
    }

    /// Add an assumption and return its token.
    pub fn assume(&mut self, literal: Literal) -> usize {
        let token = self.assumptions.len();
        self.assumptions.push(Some(literal));
        token
    }

    /// Retract an assumption by token.
    pub fn retract_assumption(&mut self, token: usize) -> bool {
        let Some(slot) = self.assumptions.get_mut(token) else {
            return false;
        };
        let existed = slot.is_some();
        *slot = None;
        existed
    }

    /// Solve with exhaustive search.
    pub fn solve(&self) -> SolverServiceResult {
        self.solve_with_budget(SolverServiceBudget::Exhaustive)
    }

    /// Solve with a bounded search budget.
    pub fn solve_with_budget(&self, budget: SolverServiceBudget) -> SolverServiceResult {
        match budget {
            SolverServiceBudget::NoSearch => SolverServiceResult {
                status: SolverServiceStatus::Unknown,
                assignment: None,
            },
            SolverServiceBudget::AssignmentLimit(0) => SolverServiceResult {
                status: SolverServiceStatus::Timeout,
                assignment: None,
            },
            SolverServiceBudget::AssignmentLimit(limit) => self.solve_assignments(Some(limit)),
            SolverServiceBudget::Exhaustive => self.solve_assignments(None),
        }
    }

    /// Transfer learned clauses to another service.
    pub fn transfer_learned_clauses_to(&self, target: &mut SolverService) -> LearnedClauseTransfer {
        let learned = self.learned_clauses.borrow();
        let count = learned.len();
        target
            .learned_clauses
            .borrow_mut()
            .extend(learned.iter().cloned());
        target.trace.borrow_mut().learned_clause_transfers += count;
        LearnedClauseTransfer { clauses: count }
    }

    /// Return current service trace.
    pub fn trace(&self) -> SolverServiceTrace {
        self.trace.borrow().clone()
    }

    /// Report that the semantic-oracle facade is not the GPU portfolio path.
    pub fn gpu_portfolio_status(&self) -> SolverPortfolioStatus {
        SolverPortfolioStatus::Deferred {
            reason: "GPU portfolio solving is not implemented in the semantic-oracle facade and blocks solver-service integration closure",
        }
    }

    fn solve_assignments(&self, limit: Option<u64>) -> SolverServiceResult {
        let max = 1u64.checked_shl(self.instance.num_vars).unwrap_or(u64::MAX);
        let search_max = limit.map_or(max, |limit| limit.min(max));
        let hard_clauses = self.hard_clauses();

        if self.instance.objective == Objective::MaxSat {
            return self.solve_maxsat(
                search_max,
                limit.is_some() && search_max < max,
                &hard_clauses,
            );
        }

        for mask in 0..search_max {
            let assignment = assignment_from_mask(self.instance.num_vars, mask);
            if hard_clauses
                .iter()
                .all(|clause| clause.is_satisfied(&assignment))
            {
                self.record_cpu_assignment_enumerations(mask.saturating_add(1));
                return SolverServiceResult {
                    status: SolverServiceStatus::Sat,
                    assignment: Some(assignment),
                };
            }
        }
        self.record_cpu_assignment_enumerations(search_max);

        if limit.is_some() && search_max < max {
            return SolverServiceResult {
                status: SolverServiceStatus::Timeout,
                assignment: None,
            };
        }

        self.record_unsat_learning(&hard_clauses);
        SolverServiceResult {
            status: SolverServiceStatus::Unsat,
            assignment: None,
        }
    }

    fn solve_maxsat(
        &self,
        search_max: u64,
        timed_out: bool,
        hard_clauses: &[Clause],
    ) -> SolverServiceResult {
        let mut best_assignment = None;
        let mut best_score = f64::NEG_INFINITY;
        for mask in 0..search_max {
            let assignment = assignment_from_mask(self.instance.num_vars, mask);
            if !hard_clauses
                .iter()
                .all(|clause| clause.is_satisfied(&assignment))
            {
                continue;
            }
            let score = self.instance.weighted_satisfaction(&assignment);
            if score > best_score {
                best_score = score;
                best_assignment = Some(assignment);
            }
        }
        self.record_cpu_maxsat_enumerations(search_max);

        if timed_out {
            return SolverServiceResult {
                status: SolverServiceStatus::Timeout,
                assignment: None,
            };
        }

        match best_assignment {
            Some(assignment) => SolverServiceResult {
                status: SolverServiceStatus::Optimal(best_score as u64),
                assignment: Some(assignment),
            },
            None => SolverServiceResult {
                status: SolverServiceStatus::Unsat,
                assignment: None,
            },
        }
    }

    fn hard_clauses(&self) -> Vec<Clause> {
        let mut clauses = if self.instance.objective == Objective::MaxSat {
            Vec::new()
        } else {
            self.instance.clauses.clone()
        };
        let active_assumptions = self.active_assumptions();
        clauses.extend(active_assumptions.iter().copied().map(Clause::unit));
        clauses.extend(
            self.learned_clauses
                .borrow()
                .iter()
                .filter(|learned| {
                    learned
                        .assumption_scope
                        .iter()
                        .all(|literal| active_assumptions.contains(literal))
                })
                .map(|learned| learned.clause.clone()),
        );
        clauses
    }

    fn record_unsat_learning(&self, hard_clauses: &[Clause]) {
        let mut learned = self.learned_clauses.borrow_mut();
        if learned.is_empty() && !hard_clauses.is_empty() {
            learned.push(ScopedLearnedClause {
                clause: Clause::new(Vec::new()),
                assumption_scope: self.active_assumptions(),
            });
        }
    }

    fn record_cpu_assignment_enumerations(&self, count: u64) {
        let mut trace = self.trace.borrow_mut();
        trace.cpu_assignment_enumerations = trace.cpu_assignment_enumerations.saturating_add(count);
    }

    fn record_cpu_maxsat_enumerations(&self, count: u64) {
        let mut trace = self.trace.borrow_mut();
        trace.cpu_maxsat_enumerations = trace.cpu_maxsat_enumerations.saturating_add(count);
    }

    fn active_assumptions(&self) -> Vec<Literal> {
        self.assumptions.iter().flatten().copied().collect()
    }
}

fn assignment_from_mask(num_vars: u32, mask: u64) -> Vec<bool> {
    (0..num_vars)
        .map(|var| (mask & (1u64 << var)) != 0)
        .collect()
}
