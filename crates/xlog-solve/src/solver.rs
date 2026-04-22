//! Continuous Local Search (CLS) solver implementation.
//!
//! This module implements a CLS-based SAT/MaxSAT solver that treats SAT as continuous
//! optimization. Variables are relaxed from {0,1} to [0,1] and gradient descent with
//! momentum is used to minimize the unsatisfied clause penalty.
//!
//! # Algorithm Overview
//!
//! The CLS algorithm works as follows:
//! 1. Initialize variables with random values in [0,1]
//! 2. For each iteration:
//!    - Compute gradients of the unsatisfied clause penalty
//!    - Update assignments using momentum-based gradient descent
//!    - Clamp values to [0,1]
//!    - Check if the discretized assignment satisfies all clauses
//! 3. Return SAT if satisfied, or best effort approximation otherwise
//!
//! # Example
//!
//! ```
//! use xlog_solve::{SolveInstance, Clause, Literal, Solver};
//!
//! // Create a SAT instance: (x0) AND (NOT x0 OR x1)
//! let instance = SolveInstance::new(2, vec![
//!     Clause::new(vec![Literal::positive(0)]),
//!     Clause::new(vec![Literal::negative(0), Literal::positive(1)]),
//! ]);
//!
//! // Solve
//! let solver = Solver::new_cpu();
//! let result = solver.solve(instance);
//! ```

use crate::instance::SolveInstance;
use crate::proof::{SolveProof, SolveResult, SolveStats, SolveStatus};

// =============================================================================
// SolverConfig - Configuration parameters for the CLS solver
// =============================================================================

/// Configuration parameters for the CLS solver.
///
/// These parameters control the behavior of the continuous local search algorithm:
/// - `max_iterations`: Maximum number of gradient descent iterations
/// - `learning_rate`: Step size for gradient updates
/// - `momentum`: Momentum coefficient for velocity accumulation
/// - `discretize_threshold`: Threshold for converting continuous values to boolean
///
/// # Default Values
///
/// ```
/// use xlog_solve::SolverConfig;
///
/// let config = SolverConfig::default();
/// assert_eq!(config.max_iterations, 10000);
/// assert_eq!(config.learning_rate, 0.1);
/// assert_eq!(config.momentum, 0.9);
/// assert_eq!(config.discretize_threshold, 0.5);
/// ```
///
/// # Example
///
/// ```
/// use xlog_solve::SolverConfig;
///
/// let mut config = SolverConfig::default();
/// config.max_iterations = 5000;
/// config.learning_rate = 0.05;
/// config.momentum = 0.95;
/// config.discretize_threshold = 0.5;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SolverConfig {
    /// Maximum number of iterations before giving up.
    ///
    /// The solver will terminate and return the best-effort result if this
    /// limit is reached without finding a satisfying assignment.
    pub max_iterations: u32,

    /// Learning rate for gradient descent updates.
    ///
    /// Controls the step size when updating variable assignments.
    /// Larger values lead to faster convergence but may overshoot.
    /// Typical values are in the range [0.01, 0.5].
    pub learning_rate: f32,

    /// Momentum coefficient for velocity accumulation.
    ///
    /// Controls how much of the previous velocity is retained.
    /// Helps escape local minima and smooths the optimization trajectory.
    /// Values close to 1.0 give more weight to history.
    /// Typical values are in the range [0.8, 0.99].
    pub momentum: f32,

    /// Threshold for discretizing continuous values to boolean.
    ///
    /// Values >= threshold become `true`, values < threshold become `false`.
    /// The standard value is 0.5 for symmetric treatment.
    pub discretize_threshold: f32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10000,
            learning_rate: 0.1,
            momentum: 0.9,
            discretize_threshold: 0.5,
        }
    }
}

impl SolverConfig {
    /// Creates a new configuration with specified parameters.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - Maximum number of iterations
    /// * `learning_rate` - Step size for gradient updates
    /// * `momentum` - Momentum coefficient
    /// * `discretize_threshold` - Threshold for boolean conversion
    #[inline]
    pub const fn new(
        max_iterations: u32,
        learning_rate: f32,
        momentum: f32,
        discretize_threshold: f32,
    ) -> Self {
        Self {
            max_iterations,
            learning_rate,
            momentum,
            discretize_threshold,
        }
    }

    /// Creates a configuration optimized for fast convergence on small instances.
    ///
    /// Uses higher learning rate and fewer iterations.
    #[inline]
    pub const fn fast() -> Self {
        Self {
            max_iterations: 1000,
            learning_rate: 0.2,
            momentum: 0.9,
            discretize_threshold: 0.5,
        }
    }

    /// Creates a configuration optimized for thorough search on hard instances.
    ///
    /// Uses more iterations and lower learning rate for better exploration.
    #[inline]
    pub const fn thorough() -> Self {
        Self {
            max_iterations: 50000,
            learning_rate: 0.05,
            momentum: 0.95,
            discretize_threshold: 0.5,
        }
    }

    /// Sets the maximum iterations, consuming and returning self.
    #[inline]
    pub const fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets the learning rate, consuming and returning self.
    #[inline]
    pub const fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the momentum, consuming and returning self.
    #[inline]
    pub const fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets the discretize threshold, consuming and returning self.
    #[inline]
    pub const fn with_discretize_threshold(mut self, threshold: f32) -> Self {
        self.discretize_threshold = threshold;
        self
    }
}

// =============================================================================
// SolverState - Internal state during CLS optimization
// =============================================================================

/// Internal state during CLS optimization.
///
/// Tracks the continuous variable assignments, momentum velocities, and
/// computed gradients throughout the optimization process.
///
/// # Memory Layout
///
/// Each vector has length equal to the number of variables:
/// - `assignments`: Current continuous values in [0,1]
/// - `velocities`: Momentum velocities (accumulated gradient history)
/// - `gradients`: Computed gradients for the current iteration
#[derive(Debug, Clone)]
pub struct SolverState {
    /// Continuous variable assignments in [0,1].
    ///
    /// Index i corresponds to variable i. Values close to 1.0 indicate
    /// the variable should be true, values close to 0.0 indicate false.
    pub assignments: Vec<f32>,

    /// Momentum velocities for each variable.
    ///
    /// Accumulates gradient history to help escape local minima
    /// and smooth the optimization trajectory.
    pub velocities: Vec<f32>,

    /// Computed gradients for the current iteration.
    ///
    /// The gradient of the unsatisfied clause penalty with respect
    /// to each variable. Negative gradients indicate the variable
    /// should increase to reduce unsatisfaction.
    pub gradients: Vec<f32>,
}

impl SolverState {
    /// Creates a new solver state for the given number of variables.
    ///
    /// Variables are initialized with values near 0.5 using a deterministic
    /// pseudo-random pattern based on index to break symmetry while
    /// maintaining reproducibility.
    ///
    /// # Arguments
    ///
    /// * `num_vars` - Number of variables in the SAT instance
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolverState;
    ///
    /// let state = SolverState::new(10);
    /// assert_eq!(state.assignments.len(), 10);
    /// assert_eq!(state.velocities.len(), 10);
    /// assert_eq!(state.gradients.len(), 10);
    /// ```
    pub fn new(num_vars: u32) -> Self {
        let n = num_vars as usize;

        // Initialize assignments with pseudo-random values near 0.5
        // This breaks symmetry while being deterministic
        let assignments: Vec<f32> = (0..n)
            .map(|i| {
                // Simple deterministic pseudo-random initialization
                // Uses golden ratio for good distribution
                let phi = 1.618033988749895_f64;
                let val = ((i as f64 + 1.0) * phi).fract() as f32;
                // Keep values in [0.3, 0.7] to avoid starting at extremes
                0.3 + val * 0.4
            })
            .collect();

        Self {
            assignments,
            velocities: vec![0.0; n],
            gradients: vec![0.0; n],
        }
    }

    /// Creates a solver state with specific initial assignments.
    ///
    /// Useful for warm-starting from a known assignment or for testing.
    ///
    /// # Arguments
    ///
    /// * `assignments` - Initial continuous assignments in [0,1]
    pub fn with_assignments(assignments: Vec<f32>) -> Self {
        let n = assignments.len();
        Self {
            assignments,
            velocities: vec![0.0; n],
            gradients: vec![0.0; n],
        }
    }

    /// Discretizes the continuous assignments to boolean values.
    ///
    /// Values >= threshold become `true`, values < threshold become `false`.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold for boolean conversion (typically 0.5)
    ///
    /// # Returns
    ///
    /// A vector of boolean values, one per variable.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolverState;
    ///
    /// let mut state = SolverState::new(3);
    /// state.assignments = vec![0.3, 0.7, 0.5];
    /// let discrete = state.discretize(0.5);
    /// assert_eq!(discrete, vec![false, true, true]);
    /// ```
    #[inline]
    pub fn discretize(&self, threshold: f32) -> Vec<bool> {
        self.assignments
            .iter()
            .map(|&val| val >= threshold)
            .collect()
    }

    /// Returns the number of variables in this state.
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.assignments.len()
    }

    /// Resets all velocities to zero.
    ///
    /// Useful for restarting the optimization with a fresh momentum state.
    #[inline]
    pub fn reset_velocities(&mut self) {
        self.velocities.fill(0.0);
    }

    /// Clears the gradients buffer.
    #[inline]
    pub fn clear_gradients(&mut self) {
        self.gradients.fill(0.0);
    }
}

// =============================================================================
// Solver - The main CLS solver
// =============================================================================

/// The CLS-based SAT/MaxSAT solver.
///
/// Implements Continuous Local Search, treating SAT as continuous optimization.
/// Variables are relaxed from {0,1} to [0,1] and gradient descent with momentum
/// is used to minimize the unsatisfied clause penalty.
///
/// # Algorithm
///
/// The solver minimizes the "unsatisfied-ness" of clauses:
/// - For each clause, compute the product of (1 - lit_value) for all literals
/// - This product is 0 when any literal is satisfied, approaching 1 when all fail
/// - Gradient descent moves variables to minimize this penalty
///
/// # Completeness
///
/// CLS is an incomplete solver - it can find satisfying assignments efficiently
/// but cannot prove unsatisfiability. For unsatisfiable instances, it will
/// return `Unknown` status with the best-effort assignment found.
///
/// # Example
///
/// ```
/// use xlog_solve::{SolveInstance, Clause, Literal, Solver, SolverConfig};
///
/// // Create instance: (x0 OR x1) AND (NOT x0 OR x1)
/// let instance = SolveInstance::new(2, vec![
///     Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
///     Clause::new(vec![Literal::negative(0), Literal::positive(1)]),
/// ]);
///
/// // Solve with default config
/// let solver = Solver::new_cpu();
/// let result = solver.solve(instance.clone());
///
/// // Or with custom config
/// let config = SolverConfig::default().with_max_iterations(5000);
/// let solver = Solver::with_config_cpu(config);
/// let result = solver.solve(instance);
/// ```
#[derive(Debug, Clone)]
pub struct Solver {
    /// Configuration parameters for the solver.
    config: SolverConfig,
}

impl Solver {
    /// Creates a new CPU-based CLS solver with default configuration.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::Solver;
    ///
    /// let solver = Solver::new_cpu();
    /// ```
    #[inline]
    pub fn new_cpu() -> Self {
        Self {
            config: SolverConfig::default(),
        }
    }

    /// Creates a new CPU-based CLS solver with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The solver configuration
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::{Solver, SolverConfig};
    ///
    /// let config = SolverConfig::default()
    ///     .with_max_iterations(5000)
    ///     .with_learning_rate(0.05);
    /// let solver = Solver::with_config_cpu(config);
    /// ```
    #[inline]
    pub fn with_config_cpu(config: SolverConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the solver configuration.
    #[inline]
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// Solves the given SAT instance.
    ///
    /// # Arguments
    ///
    /// * `instance` - The SAT/MaxSAT instance to solve
    ///
    /// # Returns
    ///
    /// A `SolveResult` containing:
    /// - `Sat` status with satisfying assignment if found
    /// - `Unknown` status with best-effort assignment if not found
    ///
    /// Note: CLS cannot prove unsatisfiability, so `Unsat` is never returned.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::{SolveInstance, Clause, Literal, Solver, SolveStatus};
    ///
    /// let instance = SolveInstance::new(1, vec![
    ///     Clause::new(vec![Literal::positive(0)]),
    /// ]);
    ///
    /// let solver = Solver::new_cpu();
    /// let result = solver.solve(instance);
    ///
    /// assert!(matches!(result.status, SolveStatus::Sat));
    /// ```
    pub fn solve(&self, instance: SolveInstance) -> SolveResult {
        let start = std::time::Instant::now();

        // Handle edge cases
        if instance.num_vars == 0 {
            // Empty instance is trivially satisfiable if no clauses
            // or trivially unsatisfiable if there are empty clauses
            let has_empty_clause = instance.clauses.iter().any(|c| c.is_empty());
            if has_empty_clause {
                // Empty clause can never be satisfied
                return SolveResult {
                    status: SolveStatus::Unknown,
                    proof: SolveProof::approximate(vec![], 0, instance.clauses.len() as u32, 0),
                    stats: SolveStats::new(0, start.elapsed().as_micros() as u64, 0),
                };
            }
            // No variables and no empty clauses - trivially SAT
            return SolveResult::satisfiable(vec![]).with_stats(SolveStats::new(
                0,
                start.elapsed().as_micros() as u64,
                0,
            ));
        }

        if instance.clauses.is_empty() {
            // No clauses - any assignment works
            let assignment = vec![false; instance.num_vars as usize];
            return SolveResult::satisfiable(assignment).with_stats(SolveStats::new(
                0,
                start.elapsed().as_micros() as u64,
                0,
            ));
        }

        // Check for empty clauses (impossible to satisfy)
        if instance.clauses.iter().any(|c| c.is_empty()) {
            return SolveResult {
                status: SolveStatus::Unknown,
                proof: SolveProof::approximate(
                    vec![false; instance.num_vars as usize],
                    instance.count_satisfied(&vec![false; instance.num_vars as usize]) as u32,
                    instance.clauses.len() as u32,
                    0,
                ),
                stats: SolveStats::new(0, start.elapsed().as_micros() as u64, 0),
            };
        }

        let mut state = SolverState::new(instance.num_vars);

        // Track best solution found
        let mut best_assignment: Option<Vec<bool>> = None;
        let mut best_satisfied: u32 = 0;

        for iter in 0..self.config.max_iterations {
            // Compute gradients
            self.compute_gradients(&instance, &mut state);

            // Update with momentum
            self.update_assignments(&mut state);

            // Check if solved
            let discrete = state.discretize(self.config.discretize_threshold);
            let satisfied = instance.count_satisfied(&discrete) as u32;

            // Track best solution
            if satisfied > best_satisfied {
                best_satisfied = satisfied;
                best_assignment = Some(discrete.clone());
            }

            if instance.is_satisfied(&discrete) {
                return SolveResult::satisfiable(discrete).with_stats(SolveStats {
                    iterations: iter + 1,
                    duration_us: start.elapsed().as_micros() as u64,
                    peak_memory: 0,
                });
            }
        }

        // Return best effort
        let final_discrete =
            best_assignment.unwrap_or_else(|| state.discretize(self.config.discretize_threshold));
        let final_satisfied = instance.count_satisfied(&final_discrete) as u32;

        SolveResult {
            status: SolveStatus::Unknown,
            proof: SolveProof::approximate(
                final_discrete,
                final_satisfied,
                instance.clauses.len() as u32,
                self.config.max_iterations,
            ),
            stats: SolveStats {
                iterations: self.config.max_iterations,
                duration_us: start.elapsed().as_micros() as u64,
                peak_memory: 0,
            },
        }
    }

    /// Computes gradients of the unsatisfied clause penalty.
    ///
    /// For each clause, the "unsatisfied-ness" is the product of (1 - lit_value)
    /// for all literals. The gradient is computed using the product rule:
    ///
    /// d(clause_unsat)/d(var) = d/d(var)[ prod_i (1 - lit_i) ]
    ///                        = sign(lit) * prod_{j != i} (1 - lit_j)
    ///
    /// where sign(lit) = -1 for positive literals, +1 for negative literals.
    ///
    /// # Arguments
    ///
    /// * `instance` - The SAT instance
    /// * `state` - The solver state to update with computed gradients
    fn compute_gradients(&self, instance: &SolveInstance, state: &mut SolverState) {
        state.gradients.fill(0.0);

        for clause in &instance.clauses {
            // Compute clause unsatisfaction: prod_i (1 - lit_val_i)
            // This is 0 when any literal is satisfied (lit_val = 1)
            // and approaches 1 when all literals are unsatisfied (lit_val = 0)
            let mut clause_unsat = 1.0f32;
            for lit in &clause.literals {
                let val = state.assignments[lit.var as usize];
                // lit_val is the "truth value" of the literal:
                // - For positive literal: lit_val = val
                // - For negative literal: lit_val = 1 - val
                let lit_val = if lit.negated { 1.0 - val } else { val };
                clause_unsat *= 1.0 - lit_val;
            }

            // Skip nearly satisfied clauses (gradient contribution negligible)
            // This is an important optimization: when a clause is satisfied,
            // its gradient contribution is essentially zero, so we skip it
            if clause_unsat < 0.001 {
                continue;
            }

            // Compute gradient contribution for each literal in this clause
            // Using the product rule: d/dx[f(x)*g(y)] = g(y) * df/dx
            // The gradient of (1 - lit_val) with respect to var is:
            // - For positive literal: d/dvar[1 - var] = -1
            // - For negative literal: d/dvar[1 - (1-var)] = d/dvar[var] = +1
            for lit in &clause.literals {
                let var = lit.var as usize;

                // Compute product of other terms (excluding this literal)
                // This gives us the coefficient when applying product rule
                let mut other_product = 1.0f32;
                for other_lit in &clause.literals {
                    if other_lit.var != lit.var {
                        let other_val = state.assignments[other_lit.var as usize];
                        let lit_val = if other_lit.negated {
                            1.0 - other_val
                        } else {
                            other_val
                        };
                        other_product *= 1.0 - lit_val;
                    }
                }

                // The derivative of (1 - lit_val) with respect to var:
                // - Positive literal: lit_val = var, so d(1-var)/dvar = -1
                // - Negative literal: lit_val = 1-var, so d(1-(1-var))/dvar = d(var)/dvar = +1
                // But we want to minimize unsatisfaction, so gradient points to increase satisfaction
                let sign = if lit.negated { 1.0 } else { -1.0 };
                state.gradients[var] += sign * other_product;
            }
        }
    }

    /// Updates variable assignments using momentum-based gradient descent.
    ///
    /// The update rule is:
    /// ```text
    /// velocity[i] = momentum * velocity[i] - learning_rate * gradient[i]
    /// assignment[i] = clamp(assignment[i] + velocity[i], 0.0, 1.0)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `state` - The solver state to update
    fn update_assignments(&self, state: &mut SolverState) {
        for i in 0..state.assignments.len() {
            // Momentum update: accumulate velocity
            state.velocities[i] = self.config.momentum * state.velocities[i]
                - self.config.learning_rate * state.gradients[i];

            // Update assignment
            state.assignments[i] += state.velocities[i];

            // Clamp to valid range [0, 1]
            state.assignments[i] = state.assignments[i].clamp(0.0, 1.0);
        }
    }
}

impl Default for Solver {
    fn default() -> Self {
        Self::new_cpu()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{Clause, Literal};

    // ==========================================================================
    // SolverConfig Tests
    // ==========================================================================

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert_eq!(config.max_iterations, 10000);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.momentum, 0.9);
        assert_eq!(config.discretize_threshold, 0.5);
    }

    #[test]
    fn test_solver_config_new() {
        let config = SolverConfig::new(5000, 0.05, 0.95, 0.4);
        assert_eq!(config.max_iterations, 5000);
        assert_eq!(config.learning_rate, 0.05);
        assert_eq!(config.momentum, 0.95);
        assert_eq!(config.discretize_threshold, 0.4);
    }

    #[test]
    fn test_solver_config_fast() {
        let config = SolverConfig::fast();
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.learning_rate, 0.2);
    }

    #[test]
    fn test_solver_config_thorough() {
        let config = SolverConfig::thorough();
        assert_eq!(config.max_iterations, 50000);
        assert_eq!(config.learning_rate, 0.05);
    }

    #[test]
    fn test_solver_config_builders() {
        let config = SolverConfig::default()
            .with_max_iterations(2000)
            .with_learning_rate(0.2)
            .with_momentum(0.8)
            .with_discretize_threshold(0.6);

        assert_eq!(config.max_iterations, 2000);
        assert_eq!(config.learning_rate, 0.2);
        assert_eq!(config.momentum, 0.8);
        assert_eq!(config.discretize_threshold, 0.6);
    }

    // ==========================================================================
    // SolverState Tests
    // ==========================================================================

    #[test]
    fn test_solver_state_new() {
        let state = SolverState::new(5);
        assert_eq!(state.assignments.len(), 5);
        assert_eq!(state.velocities.len(), 5);
        assert_eq!(state.gradients.len(), 5);

        // All velocities and gradients should be zero
        assert!(state.velocities.iter().all(|&v| v == 0.0));
        assert!(state.gradients.iter().all(|&g| g == 0.0));

        // Assignments should be in [0.3, 0.7]
        for &val in &state.assignments {
            assert!(val >= 0.3 && val <= 0.7);
        }
    }

    #[test]
    fn test_solver_state_with_assignments() {
        let assignments = vec![0.1, 0.5, 0.9];
        let state = SolverState::with_assignments(assignments.clone());
        assert_eq!(state.assignments, assignments);
        assert!(state.velocities.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_solver_state_discretize() {
        let mut state = SolverState::new(4);
        state.assignments = vec![0.2, 0.5, 0.6, 0.9];

        let discrete = state.discretize(0.5);
        assert_eq!(discrete, vec![false, true, true, true]);

        let discrete_high = state.discretize(0.7);
        assert_eq!(discrete_high, vec![false, false, false, true]);
    }

    #[test]
    fn test_solver_state_num_vars() {
        let state = SolverState::new(10);
        assert_eq!(state.num_vars(), 10);
    }

    #[test]
    fn test_solver_state_reset_velocities() {
        let mut state = SolverState::new(3);
        state.velocities = vec![1.0, 2.0, 3.0];
        state.reset_velocities();
        assert!(state.velocities.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_solver_state_clear_gradients() {
        let mut state = SolverState::new(3);
        state.gradients = vec![1.0, 2.0, 3.0];
        state.clear_gradients();
        assert!(state.gradients.iter().all(|&g| g == 0.0));
    }

    // ==========================================================================
    // Solver Construction Tests
    // ==========================================================================

    #[test]
    fn test_solver_new_cpu() {
        let solver = Solver::new_cpu();
        assert_eq!(solver.config().max_iterations, 10000);
    }

    #[test]
    fn test_solver_with_config_cpu() {
        let config = SolverConfig::fast();
        let solver = Solver::with_config_cpu(config);
        assert_eq!(solver.config().max_iterations, 1000);
    }

    #[test]
    fn test_solver_default() {
        let solver = Solver::default();
        assert_eq!(solver.config().max_iterations, 10000);
    }

    // ==========================================================================
    // Core Algorithm Tests (from task spec)
    // ==========================================================================

    #[test]
    fn test_solver_simple_sat() {
        // (x0) - trivially satisfiable
        let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);
        assert!(matches!(result.status, SolveStatus::Sat));
        if let Some(assignment) = result.assignment() {
            assert!(assignment[0]); // x0 must be true
        }
    }

    #[test]
    fn test_solver_two_clause() {
        // (x0 OR x1) AND (NOT x0 OR x1) - x1 must be true
        let instance = SolveInstance::new(
            2,
            vec![
                Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
                Clause::new(vec![Literal::negative(0), Literal::positive(1)]),
            ],
        );
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);
        assert!(matches!(result.status, SolveStatus::Sat));
        if let Some(assignment) = result.assignment() {
            assert!(assignment[1]); // x1 must be true
        }
    }

    #[test]
    fn test_solver_unsat() {
        // (x0) AND (NOT x0) - unsatisfiable
        let instance = SolveInstance::new(
            1,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::negative(0)]),
            ],
        );
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);
        // CLS is incomplete for UNSAT, so Unknown is acceptable
        assert!(matches!(
            result.status,
            SolveStatus::Unsat | SolveStatus::Unknown
        ));
    }

    // ==========================================================================
    // Additional Algorithm Tests
    // ==========================================================================

    #[test]
    fn test_solver_empty_instance() {
        // No clauses - any assignment works
        let instance = SolveInstance::new(3, vec![]);
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);
        assert!(matches!(result.status, SolveStatus::Sat));
    }

    #[test]
    fn test_solver_no_variables() {
        // No variables and no clauses
        let instance = SolveInstance::new(0, vec![]);
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);
        assert!(matches!(result.status, SolveStatus::Sat));
    }

    #[test]
    fn test_solver_unit_propagation() {
        // Unit clause forces x0=true, which makes second clause satisfied via x1
        // (x0) AND (NOT x0 OR x1)
        let instance = SolveInstance::new(
            2,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::negative(0), Literal::positive(1)]),
            ],
        );
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);
        assert!(matches!(result.status, SolveStatus::Sat));
        if let Some(assignment) = result.assignment() {
            assert!(assignment[0]); // x0 must be true
        }
    }

    #[test]
    fn test_solver_negative_unit() {
        // (NOT x0) - must set x0=false
        let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);
        assert!(matches!(result.status, SolveStatus::Sat));
        if let Some(assignment) = result.assignment() {
            assert!(!assignment[0]); // x0 must be false
        }
    }

    #[test]
    fn test_solver_three_vars() {
        // (x0 OR x1) AND (NOT x1 OR x2) AND (NOT x2 OR x0)
        // Satisfiable: x0=true, x1=true, x2=true
        let instance = SolveInstance::new(
            3,
            vec![
                Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
                Clause::new(vec![Literal::negative(1), Literal::positive(2)]),
                Clause::new(vec![Literal::negative(2), Literal::positive(0)]),
            ],
        );
        let solver = Solver::new_cpu();
        let result = solver.solve(instance.clone());
        assert!(matches!(result.status, SolveStatus::Sat));
        if let Some(assignment) = result.assignment() {
            assert!(instance.is_satisfied(assignment));
        }
    }

    #[test]
    fn test_solver_all_positive() {
        // (x0) AND (x1) AND (x2) - all must be true
        let instance = SolveInstance::new(
            3,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::positive(1)]),
                Clause::new(vec![Literal::positive(2)]),
            ],
        );
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);
        assert!(matches!(result.status, SolveStatus::Sat));
        if let Some(assignment) = result.assignment() {
            assert!(assignment.iter().all(|&v| v));
        }
    }

    #[test]
    fn test_solver_all_negative() {
        // (NOT x0) AND (NOT x1) AND (NOT x2) - all must be false
        let instance = SolveInstance::new(
            3,
            vec![
                Clause::new(vec![Literal::negative(0)]),
                Clause::new(vec![Literal::negative(1)]),
                Clause::new(vec![Literal::negative(2)]),
            ],
        );
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);
        assert!(matches!(result.status, SolveStatus::Sat));
        if let Some(assignment) = result.assignment() {
            assert!(assignment.iter().all(|&v| !v));
        }
    }

    #[test]
    fn test_solver_xor_like() {
        // (x0 OR x1) AND (NOT x0 OR NOT x1) - XOR: exactly one must be true
        let instance = SolveInstance::new(
            2,
            vec![
                Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
                Clause::new(vec![Literal::negative(0), Literal::negative(1)]),
            ],
        );
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);
        assert!(matches!(result.status, SolveStatus::Sat));
        if let Some(assignment) = result.assignment() {
            // Exactly one should be true
            assert!(assignment[0] != assignment[1]);
        }
    }

    #[test]
    fn test_solver_binary_clause() {
        // (x0 OR x1) - at least one true
        let instance = SolveInstance::new(
            2,
            vec![Clause::new(vec![
                Literal::positive(0),
                Literal::positive(1),
            ])],
        );
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);
        assert!(matches!(result.status, SolveStatus::Sat));
        if let Some(assignment) = result.assignment() {
            assert!(assignment[0] || assignment[1]);
        }
    }

    #[test]
    fn test_solver_ternary_clause() {
        // (x0 OR x1 OR x2) - at least one true
        let instance = SolveInstance::new(
            3,
            vec![Clause::new(vec![
                Literal::positive(0),
                Literal::positive(1),
                Literal::positive(2),
            ])],
        );
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);
        assert!(matches!(result.status, SolveStatus::Sat));
        if let Some(assignment) = result.assignment() {
            assert!(assignment[0] || assignment[1] || assignment[2]);
        }
    }

    // ==========================================================================
    // Statistics Tests
    // ==========================================================================

    #[test]
    fn test_solver_stats() {
        let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);

        // Should have iteration count > 0
        assert!(result.stats.iterations > 0);
        // Duration should be recorded
        // Note: Very fast executions might have 0 microseconds
        assert!(result.stats.iterations <= solver.config().max_iterations);
    }

    #[test]
    fn test_solver_stats_iterations_limited() {
        // Test that iterations are limited by config
        let config = SolverConfig::default().with_max_iterations(10);
        let solver = Solver::with_config_cpu(config);

        // Unsatisfiable instance will run to max iterations
        let instance = SolveInstance::new(
            1,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::negative(0)]),
            ],
        );
        let result = solver.solve(instance);

        assert!(result.stats.iterations <= 10);
    }

    // ==========================================================================
    // Gradient Computation Tests
    // ==========================================================================

    #[test]
    fn test_compute_gradients_single_positive() {
        // (x0) - gradient should push x0 towards 1
        let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
        let solver = Solver::new_cpu();
        let mut state = SolverState::with_assignments(vec![0.5]);

        solver.compute_gradients(&instance, &mut state);

        // Gradient should be negative (to increase x0 when subtracted)
        assert!(state.gradients[0] < 0.0);
    }

    #[test]
    fn test_compute_gradients_single_negative() {
        // (NOT x0) - gradient should push x0 towards 0
        let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::negative(0)])]);
        let solver = Solver::new_cpu();
        let mut state = SolverState::with_assignments(vec![0.5]);

        solver.compute_gradients(&instance, &mut state);

        // Gradient should be positive (to decrease x0 when subtracted)
        assert!(state.gradients[0] > 0.0);
    }

    #[test]
    fn test_compute_gradients_satisfied_clause() {
        // (x0) with x0=1.0 - clause is satisfied, gradient should be near zero
        let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
        let solver = Solver::new_cpu();
        let mut state = SolverState::with_assignments(vec![1.0]);

        solver.compute_gradients(&instance, &mut state);

        // Gradient should be very small (clause is satisfied)
        assert!(state.gradients[0].abs() < 0.01);
    }

    // ==========================================================================
    // Update Assignments Tests
    // ==========================================================================

    #[test]
    fn test_update_assignments_clamps() {
        let solver = Solver::with_config_cpu(SolverConfig::default().with_learning_rate(10.0));
        let mut state = SolverState::with_assignments(vec![0.5]);
        state.gradients = vec![-1.0]; // Large negative gradient

        solver.update_assignments(&mut state);

        // Should be clamped to [0, 1]
        assert!(state.assignments[0] >= 0.0);
        assert!(state.assignments[0] <= 1.0);
    }

    #[test]
    fn test_update_assignments_momentum() {
        let solver = Solver::with_config_cpu(
            SolverConfig::default()
                .with_learning_rate(0.1)
                .with_momentum(0.5),
        );
        let mut state = SolverState::with_assignments(vec![0.5]);
        state.velocities = vec![0.1]; // Previous velocity
        state.gradients = vec![-0.1]; // Gradient

        solver.update_assignments(&mut state);

        // Velocity should incorporate both momentum and gradient
        // v = 0.5 * 0.1 - 0.1 * (-0.1) = 0.05 + 0.01 = 0.06
        let expected_velocity = 0.5 * 0.1 - 0.1 * (-0.1);
        assert!((state.velocities[0] - expected_velocity).abs() < 1e-6);
    }

    // ==========================================================================
    // Edge Case Tests
    // ==========================================================================

    #[test]
    fn test_solver_empty_clause() {
        // Empty clause can never be satisfied
        let instance = SolveInstance::new(1, vec![Clause::new(vec![])]);
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);

        // Should return Unknown (CLS cannot prove UNSAT)
        assert!(matches!(result.status, SolveStatus::Unknown));
    }

    #[test]
    fn test_solver_large_clause() {
        // Large clause (10 literals)
        let literals: Vec<Literal> = (0..10).map(Literal::positive).collect();
        let instance = SolveInstance::new(10, vec![Clause::new(literals)]);
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);

        assert!(matches!(result.status, SolveStatus::Sat));
    }

    #[test]
    fn test_solver_many_clauses() {
        // Many unit clauses (20 variables, each must be true)
        let clauses: Vec<Clause> = (0..20)
            .map(|i| Clause::new(vec![Literal::positive(i)]))
            .collect();
        let instance = SolveInstance::new(20, clauses);
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);

        assert!(matches!(result.status, SolveStatus::Sat));
        if let Some(assignment) = result.assignment() {
            assert!(assignment.iter().all(|&v| v));
        }
    }

    #[test]
    fn test_solver_pigeon_hole_small() {
        // 2 pigeons, 1 hole - unsatisfiable
        // p1_h1: pigeon 1 in hole 1
        // p2_h1: pigeon 2 in hole 1
        // Clauses:
        // (p1_h1) - pigeon 1 must be somewhere
        // (p2_h1) - pigeon 2 must be somewhere
        // (NOT p1_h1 OR NOT p2_h1) - hole 1 can have at most one pigeon
        let instance = SolveInstance::new(
            2,
            vec![
                Clause::new(vec![Literal::positive(0)]), // pigeon 1 in hole 1
                Clause::new(vec![Literal::positive(1)]), // pigeon 2 in hole 1
                Clause::new(vec![Literal::negative(0), Literal::negative(1)]), // at most one
            ],
        );
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);

        // CLS cannot prove UNSAT, so should return Unknown
        assert!(matches!(result.status, SolveStatus::Unknown));
    }

    #[test]
    fn test_solver_deterministic() {
        // Same instance should give same result (solver is deterministic)
        let instance = SolveInstance::new(
            2,
            vec![
                Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
                Clause::new(vec![Literal::negative(0), Literal::positive(1)]),
            ],
        );

        let solver = Solver::new_cpu();
        let result1 = solver.solve(instance.clone());
        let result2 = solver.solve(instance);

        assert_eq!(result1.status, result2.status);
        assert_eq!(result1.assignment(), result2.assignment());
    }

    #[test]
    fn test_solver_with_fast_config() {
        // Fast config should still solve simple instances
        let instance = SolveInstance::new(
            2,
            vec![Clause::new(vec![
                Literal::positive(0),
                Literal::positive(1),
            ])],
        );
        let solver = Solver::with_config_cpu(SolverConfig::fast());
        let result = solver.solve(instance);

        assert!(matches!(result.status, SolveStatus::Sat));
    }

    #[test]
    fn test_solver_implication_chain() {
        // x0 -> x1 -> x2 -> x3, with x0=true
        // (x0) AND (NOT x0 OR x1) AND (NOT x1 OR x2) AND (NOT x2 OR x3)
        // Should result in all variables true
        let instance = SolveInstance::new(
            4,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::negative(0), Literal::positive(1)]),
                Clause::new(vec![Literal::negative(1), Literal::positive(2)]),
                Clause::new(vec![Literal::negative(2), Literal::positive(3)]),
            ],
        );
        let solver = Solver::new_cpu();
        let result = solver.solve(instance);

        assert!(matches!(result.status, SolveStatus::Sat));
        if let Some(assignment) = result.assignment() {
            // All should be true due to implications
            assert!(assignment.iter().all(|&v| v));
        }
    }
}
