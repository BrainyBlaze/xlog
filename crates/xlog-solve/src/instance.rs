//! SAT instance representation types.
//!
//! This module provides the core types for representing SAT and MaxSAT instances
//! in Conjunctive Normal Form (CNF). The representation is designed for efficient
//! evaluation and GPU-friendly memory layouts.
//!
//! # Types
//!
//! - [`Literal`]: A propositional variable with optional negation
//! - [`Clause`]: A disjunction (OR) of literals
//! - [`Objective`]: Optimization objective (satisfaction, MaxSAT, etc.)
//! - [`SolveInstance`]: A complete SAT/MaxSAT instance
//!
//! # Example
//!
//! ```
//! use xlog_solve::{SolveInstance, Clause, Literal};
//!
//! // Encode: (x0 OR NOT x1) AND (x1 OR x2)
//! let instance = SolveInstance::new(3, vec![
//!     Clause::new(vec![Literal::positive(0), Literal::negative(1)]),
//!     Clause::new(vec![Literal::positive(1), Literal::positive(2)]),
//! ]);
//!
//! // Check if an assignment satisfies the formula
//! let assignment = vec![true, false, true];
//! assert!(instance.is_satisfied(&assignment));
//! ```

use std::cmp::Ordering;

// =============================================================================
// Literal - A propositional variable with optional negation
// =============================================================================

/// A propositional variable with optional negation.
///
/// Literals are the atomic units of SAT formulas. A literal is either a variable
/// (positive literal) or the negation of a variable (negative literal).
///
/// Variables are 0-indexed internally but convert to 1-indexed for DIMACS format.
///
/// # Memory Layout
///
/// The struct is designed for efficient GPU transfer with a compact representation:
/// - `var`: 4 bytes (u32 variable index)
/// - `negated`: 1 byte (bool), with potential padding
///
/// For bulk GPU operations, consider using the packed representation via `to_packed()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Literal {
    /// The variable index (0-indexed).
    pub var: u32,
    /// Whether this literal is negated.
    pub negated: bool,
}

impl Literal {
    /// Creates a positive literal (variable without negation).
    ///
    /// # Arguments
    ///
    /// * `var` - The variable index (0-indexed)
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::Literal;
    /// let lit = Literal::positive(3);
    /// assert_eq!(lit.var, 3);
    /// assert!(!lit.negated);
    /// ```
    #[inline]
    pub const fn positive(var: u32) -> Self {
        Self {
            var,
            negated: false,
        }
    }

    /// Creates a negative literal (negated variable).
    ///
    /// # Arguments
    ///
    /// * `var` - The variable index (0-indexed)
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::Literal;
    /// let lit = Literal::negative(3);
    /// assert_eq!(lit.var, 3);
    /// assert!(lit.negated);
    /// ```
    #[inline]
    pub const fn negative(var: u32) -> Self {
        Self { var, negated: true }
    }

    /// Creates a literal from a variable and sign.
    ///
    /// # Arguments
    ///
    /// * `var` - The variable index (0-indexed)
    /// * `negated` - Whether the literal is negated
    #[inline]
    pub const fn new(var: u32, negated: bool) -> Self {
        Self { var, negated }
    }

    /// Returns the negation of this literal.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::Literal;
    /// let pos = Literal::positive(5);
    /// let neg = pos.negate();
    /// assert!(neg.negated);
    /// assert_eq!(neg.var, 5);
    /// ```
    #[inline]
    pub const fn negate(self) -> Self {
        Self {
            var: self.var,
            negated: !self.negated,
        }
    }

    /// Evaluates this literal under a given assignment.
    ///
    /// # Arguments
    ///
    /// * `assignment` - A slice of boolean values where index i represents variable i
    ///
    /// # Returns
    ///
    /// `true` if the literal is satisfied by the assignment, `false` otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `self.var >= assignment.len()` (variable index out of bounds).
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::Literal;
    /// let lit = Literal::positive(1);
    /// assert!(lit.eval(&[false, true, false]));  // x1 = true
    /// ```
    #[inline]
    pub fn eval(self, assignment: &[bool]) -> bool {
        let value = assignment[self.var as usize];
        if self.negated {
            !value
        } else {
            value
        }
    }

    /// Converts this literal to DIMACS format.
    ///
    /// DIMACS format uses 1-indexed variables, with negative numbers for negated literals.
    ///
    /// # Returns
    ///
    /// A signed integer where:
    /// - Positive values represent positive literals (var + 1)
    /// - Negative values represent negative literals (-(var + 1))
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::Literal;
    /// assert_eq!(Literal::positive(0).to_dimacs(), 1);
    /// assert_eq!(Literal::negative(2).to_dimacs(), -3);
    /// ```
    #[inline]
    pub fn to_dimacs(self) -> i32 {
        let var_1indexed = (self.var + 1) as i32;
        if self.negated {
            -var_1indexed
        } else {
            var_1indexed
        }
    }

    /// Creates a literal from DIMACS format.
    ///
    /// DIMACS format uses 1-indexed variables, with negative numbers for negated literals.
    ///
    /// # Arguments
    ///
    /// * `dimacs` - A non-zero signed integer in DIMACS format
    ///
    /// # Panics
    ///
    /// Panics if `dimacs == 0` (zero is not a valid DIMACS literal).
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::Literal;
    /// let pos = Literal::from_dimacs(3);
    /// assert_eq!(pos.var, 2);
    /// assert!(!pos.negated);
    ///
    /// let neg = Literal::from_dimacs(-5);
    /// assert_eq!(neg.var, 4);
    /// assert!(neg.negated);
    /// ```
    #[inline]
    pub fn from_dimacs(dimacs: i32) -> Self {
        assert!(dimacs != 0, "DIMACS literal cannot be zero");
        let negated = dimacs < 0;
        let var = dimacs.unsigned_abs() - 1;
        Self { var, negated }
    }

    /// Returns a packed 32-bit representation suitable for GPU transfer.
    ///
    /// The packed format stores the variable in the lower 31 bits and the
    /// negation flag in the most significant bit.
    ///
    /// # Returns
    ///
    /// A u32 where bit 31 is the negation flag and bits 0-30 are the variable.
    #[inline]
    pub fn to_packed(self) -> u32 {
        if self.negated {
            self.var | 0x8000_0000
        } else {
            self.var
        }
    }

    /// Creates a literal from a packed 32-bit representation.
    ///
    /// # Arguments
    ///
    /// * `packed` - A u32 with negation in bit 31 and variable in bits 0-30
    #[inline]
    pub fn from_packed(packed: u32) -> Self {
        let negated = (packed & 0x8000_0000) != 0;
        let var = packed & 0x7FFF_FFFF;
        Self { var, negated }
    }
}

impl PartialOrd for Literal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Literal {
    /// Orders literals first by variable index, then by negation (positive before negative).
    fn cmp(&self, other: &Self) -> Ordering {
        match self.var.cmp(&other.var) {
            Ordering::Equal => self.negated.cmp(&other.negated),
            ord => ord,
        }
    }
}

// =============================================================================
// Clause - A disjunction (OR) of literals
// =============================================================================

/// A clause is a disjunction (OR) of literals.
///
/// In CNF (Conjunctive Normal Form), a formula is a conjunction (AND) of clauses,
/// where each clause is a disjunction (OR) of literals.
///
/// # Properties
///
/// - An empty clause is always unsatisfied (represents `false`)
/// - A clause is satisfied if at least one of its literals is satisfied
/// - Unit clauses (single literal) are important for unit propagation
///
/// # Memory Layout
///
/// The clause stores literals in a `Vec` which allows efficient iteration
/// and provides good cache locality for clause evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Clause {
    /// The literals in this clause.
    pub literals: Vec<Literal>,
}

impl Clause {
    /// Creates a new clause from a vector of literals.
    ///
    /// # Arguments
    ///
    /// * `literals` - The literals forming this disjunction
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::{Clause, Literal};
    /// // Create clause: (x0 OR NOT x1)
    /// let clause = Clause::new(vec![
    ///     Literal::positive(0),
    ///     Literal::negative(1),
    /// ]);
    /// ```
    #[inline]
    pub fn new(literals: Vec<Literal>) -> Self {
        Self { literals }
    }

    /// Creates a unit clause (single literal).
    ///
    /// Unit clauses are important in SAT solving for unit propagation.
    ///
    /// # Arguments
    ///
    /// * `literal` - The single literal in this clause
    #[inline]
    pub fn unit(literal: Literal) -> Self {
        Self {
            literals: vec![literal],
        }
    }

    /// Creates a binary clause (two literals).
    ///
    /// Binary clauses are common in many SAT encodings (implications, at-most-one, etc.).
    ///
    /// # Arguments
    ///
    /// * `a` - The first literal
    /// * `b` - The second literal
    #[inline]
    pub fn binary(a: Literal, b: Literal) -> Self {
        Self {
            literals: vec![a, b],
        }
    }

    /// Creates a ternary clause (three literals).
    ///
    /// # Arguments
    ///
    /// * `a` - The first literal
    /// * `b` - The second literal
    /// * `c` - The third literal
    #[inline]
    pub fn ternary(a: Literal, b: Literal, c: Literal) -> Self {
        Self {
            literals: vec![a, b, c],
        }
    }

    /// Returns the number of literals in this clause.
    #[inline]
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Returns true if this clause has no literals.
    ///
    /// An empty clause is always unsatisfied.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Checks if this clause is satisfied by the given assignment.
    ///
    /// A clause is satisfied if at least one of its literals evaluates to true.
    /// An empty clause is never satisfied.
    ///
    /// # Arguments
    ///
    /// * `assignment` - A slice of boolean values for each variable
    ///
    /// # Returns
    ///
    /// `true` if the clause is satisfied, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::{Clause, Literal};
    /// let clause = Clause::new(vec![
    ///     Literal::positive(0),
    ///     Literal::negative(1),
    /// ]);
    /// assert!(clause.is_satisfied(&[true, true]));  // x0=true satisfies
    /// assert!(!clause.is_satisfied(&[false, true])); // neither satisfied
    /// ```
    #[inline]
    pub fn is_satisfied(&self, assignment: &[bool]) -> bool {
        self.literals.iter().any(|lit| lit.eval(assignment))
    }

    /// Counts how many literals in this clause are satisfied by the assignment.
    ///
    /// This is useful for CLS solvers and MaxSAT evaluation.
    ///
    /// # Arguments
    ///
    /// * `assignment` - A slice of boolean values for each variable
    ///
    /// # Returns
    ///
    /// The number of literals that evaluate to true.
    #[inline]
    pub fn count_satisfied(&self, assignment: &[bool]) -> usize {
        self.literals
            .iter()
            .filter(|lit| lit.eval(assignment))
            .count()
    }

    /// Returns an iterator over the literals in this clause.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Literal> {
        self.literals.iter()
    }

    /// Returns an iterator over the variable indices in this clause.
    #[inline]
    pub fn vars(&self) -> impl Iterator<Item = u32> + '_ {
        self.literals.iter().map(|lit| lit.var)
    }
}

impl IntoIterator for Clause {
    type Item = Literal;
    type IntoIter = std::vec::IntoIter<Literal>;

    fn into_iter(self) -> Self::IntoIter {
        self.literals.into_iter()
    }
}

impl<'a> IntoIterator for &'a Clause {
    type Item = &'a Literal;
    type IntoIter = std::slice::Iter<'a, Literal>;

    fn into_iter(self) -> Self::IntoIter {
        self.literals.iter()
    }
}

impl FromIterator<Literal> for Clause {
    fn from_iter<I: IntoIterator<Item = Literal>>(iter: I) -> Self {
        Self {
            literals: iter.into_iter().collect(),
        }
    }
}

// =============================================================================
// Objective - Optimization objective for solving
// =============================================================================

/// The optimization objective for a SAT/MaxSAT instance.
///
/// This determines what the solver should optimize for:
/// - `Satisfaction`: Find any satisfying assignment (standard SAT)
/// - `MaxSat`: Maximize the number/weight of satisfied clauses
/// - `MinUnsat`: Minimize the number/weight of unsatisfied clauses
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Objective {
    /// Find any satisfying assignment (decision problem).
    ///
    /// The solver returns SAT if an assignment exists that satisfies all clauses,
    /// or UNSAT if no such assignment exists.
    #[default]
    Satisfaction,

    /// Maximize the total weight of satisfied clauses (optimization).
    ///
    /// For unweighted instances, this maximizes the count of satisfied clauses.
    /// The solver finds an assignment that maximizes the objective.
    MaxSat,

    /// Minimize the total weight of unsatisfied clauses (optimization).
    ///
    /// This is equivalent to MaxSAT for complete solvers but may differ
    /// for incomplete/approximate solvers.
    MinUnsat,
}

// =============================================================================
// SolveInstance - A complete SAT/MaxSAT instance
// =============================================================================

/// A SAT or MaxSAT instance in Conjunctive Normal Form (CNF).
///
/// This struct represents the complete problem to be solved:
/// - A set of propositional variables (indexed 0 to num_vars-1)
/// - A conjunction of clauses (each clause is a disjunction of literals)
/// - Optional weights for MaxSAT
/// - An optimization objective
///
/// # CNF Semantics
///
/// The formula represented is:
/// ```text
/// clause[0] AND clause[1] AND ... AND clause[n-1]
/// ```
///
/// where each clause is:
/// ```text
/// lit[0] OR lit[1] OR ... OR lit[k-1]
/// ```
///
/// # Memory Layout
///
/// The instance is designed for efficient GPU transfer:
/// - Clauses can be flattened to a contiguous literal array with offset indices
/// - Weights align with clause indices for vectorized operations
///
/// # Example
///
/// ```
/// use xlog_solve::{SolveInstance, Clause, Literal, Objective};
///
/// // (x0 OR NOT x1) AND (x1 OR x2)
/// let instance = SolveInstance::new(3, vec![
///     Clause::new(vec![Literal::positive(0), Literal::negative(1)]),
///     Clause::new(vec![Literal::positive(1), Literal::positive(2)]),
/// ]);
///
/// assert_eq!(instance.num_vars, 3);
/// assert_eq!(instance.clauses.len(), 2);
/// assert_eq!(instance.objective, Objective::Satisfaction);
/// ```
#[derive(Debug, Clone)]
pub struct SolveInstance {
    /// Number of propositional variables in this instance.
    ///
    /// Variables are indexed from 0 to `num_vars - 1`.
    pub num_vars: u32,

    /// The clauses forming this CNF formula.
    ///
    /// The formula is satisfied when ALL clauses are satisfied.
    pub clauses: Vec<Clause>,

    /// Optional weights for each clause (for weighted MaxSAT).
    ///
    /// If `Some`, must have the same length as `clauses`.
    /// If `None`, all clauses have implicit weight 1.0.
    pub weights: Option<Vec<f64>>,

    /// The optimization objective.
    pub objective: Objective,
}

impl SolveInstance {
    /// Creates a new SAT instance with the given variables and clauses.
    ///
    /// The instance uses `Objective::Satisfaction` (standard SAT).
    ///
    /// # Arguments
    ///
    /// * `num_vars` - The number of propositional variables
    /// * `clauses` - The clauses forming the CNF formula
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::{SolveInstance, Clause, Literal};
    ///
    /// let instance = SolveInstance::new(3, vec![
    ///     Clause::new(vec![Literal::positive(0), Literal::negative(1)]),
    ///     Clause::new(vec![Literal::positive(1), Literal::positive(2)]),
    /// ]);
    /// ```
    #[inline]
    pub fn new(num_vars: u32, clauses: Vec<Clause>) -> Self {
        Self {
            num_vars,
            clauses,
            weights: None,
            objective: Objective::Satisfaction,
        }
    }

    /// Creates a weighted MaxSAT instance.
    ///
    /// # Arguments
    ///
    /// * `num_vars` - The number of propositional variables
    /// * `clauses` - The clauses forming the CNF formula
    /// * `weights` - Weights for each clause (must match clause count)
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != clauses.len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::{SolveInstance, Clause, Literal, Objective};
    ///
    /// let instance = SolveInstance::with_weights(
    ///     2,
    ///     vec![
    ///         Clause::new(vec![Literal::positive(0)]),
    ///         Clause::new(vec![Literal::positive(1)]),
    ///     ],
    ///     vec![1.0, 2.0],  // Clause 1 has twice the weight
    /// );
    ///
    /// assert_eq!(instance.objective, Objective::MaxSat);
    /// ```
    #[inline]
    pub fn with_weights(num_vars: u32, clauses: Vec<Clause>, weights: Vec<f64>) -> Self {
        assert_eq!(
            clauses.len(),
            weights.len(),
            "Number of weights ({}) must match number of clauses ({})",
            weights.len(),
            clauses.len()
        );
        Self {
            num_vars,
            clauses,
            weights: Some(weights),
            objective: Objective::MaxSat,
        }
    }

    /// Adds a clause to this instance.
    ///
    /// # Arguments
    ///
    /// * `clause` - The clause to add
    ///
    /// # Note
    ///
    /// If the instance has weights, the new clause gets weight 1.0.
    #[inline]
    pub fn add_clause(&mut self, clause: Clause) {
        self.clauses.push(clause);
        if let Some(ref mut weights) = self.weights {
            weights.push(1.0);
        }
    }

    /// Adds a weighted clause to this instance.
    ///
    /// If the instance doesn't have weights yet, this initializes weights
    /// with 1.0 for all existing clauses.
    ///
    /// # Arguments
    ///
    /// * `clause` - The clause to add
    /// * `weight` - The weight for this clause
    #[inline]
    pub fn add_weighted_clause(&mut self, clause: Clause, weight: f64) {
        if self.weights.is_none() {
            self.weights = Some(vec![1.0; self.clauses.len()]);
        }
        self.clauses.push(clause);
        self.weights.as_mut().unwrap().push(weight);
    }

    /// Returns the number of clauses in this instance.
    #[inline]
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Checks if all clauses are satisfied by the given assignment.
    ///
    /// # Arguments
    ///
    /// * `assignment` - A slice of boolean values, one per variable
    ///
    /// # Returns
    ///
    /// `true` if all clauses are satisfied, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::{SolveInstance, Clause, Literal};
    ///
    /// let instance = SolveInstance::new(2, vec![
    ///     Clause::new(vec![Literal::positive(0)]),
    ///     Clause::new(vec![Literal::positive(1)]),
    /// ]);
    ///
    /// assert!(instance.is_satisfied(&[true, true]));
    /// assert!(!instance.is_satisfied(&[true, false]));
    /// ```
    #[inline]
    pub fn is_satisfied(&self, assignment: &[bool]) -> bool {
        self.clauses
            .iter()
            .all(|clause| clause.is_satisfied(assignment))
    }

    /// Counts how many clauses are satisfied by the given assignment.
    ///
    /// # Arguments
    ///
    /// * `assignment` - A slice of boolean values, one per variable
    ///
    /// # Returns
    ///
    /// The number of satisfied clauses.
    #[inline]
    pub fn count_satisfied(&self, assignment: &[bool]) -> usize {
        self.clauses
            .iter()
            .filter(|clause| clause.is_satisfied(assignment))
            .count()
    }

    /// Computes the weighted satisfaction score for the given assignment.
    ///
    /// If weights are specified, returns the sum of weights of satisfied clauses.
    /// If no weights are specified, each clause has implicit weight 1.0.
    ///
    /// # Arguments
    ///
    /// * `assignment` - A slice of boolean values, one per variable
    ///
    /// # Returns
    ///
    /// The total weight of satisfied clauses.
    #[inline]
    pub fn weighted_satisfaction(&self, assignment: &[bool]) -> f64 {
        match &self.weights {
            Some(weights) => self
                .clauses
                .iter()
                .zip(weights.iter())
                .filter(|(clause, _)| clause.is_satisfied(assignment))
                .map(|(_, weight)| *weight)
                .sum(),
            None => self.count_satisfied(assignment) as f64,
        }
    }

    /// Returns the total weight of all clauses.
    ///
    /// For unweighted instances, returns the number of clauses.
    #[inline]
    pub fn total_weight(&self) -> f64 {
        match &self.weights {
            Some(weights) => weights.iter().sum(),
            None => self.clauses.len() as f64,
        }
    }

    /// Returns the satisfaction ratio (satisfied weight / total weight).
    ///
    /// # Arguments
    ///
    /// * `assignment` - A slice of boolean values, one per variable
    ///
    /// # Returns
    ///
    /// A value in [0.0, 1.0] representing the fraction of weight satisfied.
    /// Returns 0.0 for empty instances (no clauses).
    #[inline]
    pub fn satisfaction_ratio(&self, assignment: &[bool]) -> f64 {
        let total = self.total_weight();
        if total == 0.0 {
            0.0
        } else {
            self.weighted_satisfaction(assignment) / total
        }
    }

    /// Returns an iterator over all clauses.
    #[inline]
    pub fn iter_clauses(&self) -> impl Iterator<Item = &Clause> {
        self.clauses.iter()
    }

    /// Returns an iterator over clauses with their weights.
    ///
    /// If no weights are specified, uses 1.0 for each clause.
    pub fn iter_weighted(&self) -> impl Iterator<Item = (&Clause, f64)> {
        let weights = self.weights.as_ref();
        self.clauses.iter().enumerate().map(move |(i, clause)| {
            let weight = weights.map_or(1.0, |w| w[i]);
            (clause, weight)
        })
    }

    /// Validates that all variable indices in clauses are within bounds.
    ///
    /// # Returns
    ///
    /// `true` if all variable indices are less than `num_vars`, `false` otherwise.
    pub fn validate(&self) -> bool {
        self.clauses
            .iter()
            .all(|clause| clause.literals.iter().all(|lit| lit.var < self.num_vars))
    }

    /// Returns the maximum variable index used in any clause.
    ///
    /// Returns `None` if there are no clauses or all clauses are empty.
    pub fn max_var(&self) -> Option<u32> {
        self.clauses
            .iter()
            .flat_map(|c| c.literals.iter())
            .map(|lit| lit.var)
            .max()
    }

    /// Returns the total number of literals across all clauses.
    pub fn total_literals(&self) -> usize {
        self.clauses.iter().map(|c| c.len()).sum()
    }
}

impl Default for SolveInstance {
    fn default() -> Self {
        Self {
            num_vars: 0,
            clauses: Vec::new(),
            weights: None,
            objective: Objective::Satisfaction,
        }
    }
}

// Tests first (TDD approach)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_new() {
        let pos = Literal::positive(5);
        assert_eq!(pos.var, 5);
        assert!(!pos.negated);

        let neg = Literal::negative(3);
        assert_eq!(neg.var, 3);
        assert!(neg.negated);
    }

    #[test]
    fn test_literal_eval() {
        let pos = Literal::positive(1);
        let neg = Literal::negative(1);

        // Assignment: x0=true, x1=false, x2=true
        let assignment = vec![true, false, true];

        // x1 is false, so positive literal evaluates to false
        assert!(!pos.eval(&assignment));
        // NOT x1 evaluates to true (since x1=false)
        assert!(neg.eval(&assignment));

        // Test with x2 (index 2, value true)
        let pos2 = Literal::positive(2);
        let neg2 = Literal::negative(2);
        assert!(pos2.eval(&assignment));
        assert!(!neg2.eval(&assignment));
    }

    #[test]
    fn test_literal_to_dimacs() {
        let pos = Literal::positive(3);
        let neg = Literal::negative(7);

        // DIMACS uses 1-indexed variables
        assert_eq!(pos.to_dimacs(), 4); // var 3 -> 4 in DIMACS (1-indexed)
        assert_eq!(neg.to_dimacs(), -8); // var 7 negated -> -8 in DIMACS
    }

    #[test]
    fn test_literal_from_dimacs() {
        // DIMACS variable 4 (positive) -> var index 3
        let pos = Literal::from_dimacs(4);
        assert_eq!(pos.var, 3);
        assert!(!pos.negated);

        // DIMACS variable -8 (negative) -> var index 7
        let neg = Literal::from_dimacs(-8);
        assert_eq!(neg.var, 7);
        assert!(neg.negated);
    }

    #[test]
    fn test_clause_new() {
        let clause = Clause::new(vec![Literal::positive(1), Literal::negative(2)]);
        assert_eq!(clause.literals.len(), 2);
    }

    #[test]
    fn test_clause_is_satisfied() {
        // Clause: (x0 OR NOT x1)
        let clause = Clause::new(vec![Literal::positive(0), Literal::negative(1)]);

        // x0=true, x1=false -> satisfied (both literals true)
        assert!(clause.is_satisfied(&[true, false]));

        // x0=true, x1=true -> satisfied (x0 is true)
        assert!(clause.is_satisfied(&[true, true]));

        // x0=false, x1=false -> satisfied (NOT x1 is true)
        assert!(clause.is_satisfied(&[false, false]));

        // x0=false, x1=true -> NOT satisfied (both literals false)
        assert!(!clause.is_satisfied(&[false, true]));
    }

    #[test]
    fn test_clause_count_satisfied() {
        // Clause: (x0 OR NOT x1 OR x2)
        let clause = Clause::new(vec![
            Literal::positive(0),
            Literal::negative(1),
            Literal::positive(2),
        ]);

        // x0=true, x1=false, x2=true -> all 3 literals satisfied
        assert_eq!(clause.count_satisfied(&[true, false, true]), 3);

        // x0=true, x1=true, x2=false -> only x0 satisfied
        assert_eq!(clause.count_satisfied(&[true, true, false]), 1);

        // x0=false, x1=true, x2=false -> none satisfied
        assert_eq!(clause.count_satisfied(&[false, true, false]), 0);
    }

    #[test]
    fn test_clause_empty() {
        // Empty clause is always unsatisfied (false)
        let empty = Clause::new(vec![]);
        assert!(!empty.is_satisfied(&[true, false, true]));
        assert_eq!(empty.count_satisfied(&[true, false, true]), 0);
    }

    #[test]
    fn test_clause_unit() {
        // Unit clause (single literal)
        let unit_pos = Clause::unit(Literal::positive(0));
        assert!(unit_pos.is_satisfied(&[true]));
        assert!(!unit_pos.is_satisfied(&[false]));

        let unit_neg = Clause::unit(Literal::negative(0));
        assert!(!unit_neg.is_satisfied(&[true]));
        assert!(unit_neg.is_satisfied(&[false]));
    }

    #[test]
    fn test_objective_default() {
        let obj = Objective::default();
        assert!(matches!(obj, Objective::Satisfaction));
    }

    #[test]
    fn test_instance_from_cnf() {
        // (x1 OR NOT x2) AND (x2 OR x3)
        let instance = SolveInstance::new(
            3,
            vec![
                Clause::new(vec![Literal::positive(0), Literal::negative(1)]),
                Clause::new(vec![Literal::positive(1), Literal::positive(2)]),
            ],
        );
        assert_eq!(instance.num_vars, 3);
        assert_eq!(instance.clauses.len(), 2);
    }

    #[test]
    fn test_instance_is_satisfied() {
        let instance = SolveInstance::new(
            3,
            vec![
                Clause::new(vec![Literal::positive(0), Literal::negative(1)]),
                Clause::new(vec![Literal::positive(1), Literal::positive(2)]),
            ],
        );

        // x0=true, x1=false, x2=true should satisfy both clauses
        let assignment = vec![true, false, true];
        assert!(instance.is_satisfied(&assignment));

        // x0=false, x1=true, x2=false should fail clause 2
        let assignment2 = vec![false, true, false];
        assert!(!instance.is_satisfied(&assignment2));
    }

    #[test]
    fn test_instance_count_satisfied() {
        let instance = SolveInstance::new(
            3,
            vec![
                Clause::new(vec![Literal::positive(0), Literal::negative(1)]),
                Clause::new(vec![Literal::positive(1), Literal::positive(2)]),
                Clause::new(vec![Literal::negative(0), Literal::negative(2)]),
            ],
        );

        // x0=true, x1=false, x2=true
        // Clause 1: x0=T or NOT x1=T -> satisfied
        // Clause 2: x1=F or x2=T -> satisfied
        // Clause 3: NOT x0=F or NOT x2=F -> NOT satisfied
        assert_eq!(instance.count_satisfied(&[true, false, true]), 2);

        // x0=false, x1=false, x2=false
        // Clause 1: x0=F or NOT x1=T -> satisfied
        // Clause 2: x1=F or x2=F -> NOT satisfied
        // Clause 3: NOT x0=T or NOT x2=T -> satisfied
        assert_eq!(instance.count_satisfied(&[false, false, false]), 2);
    }

    #[test]
    fn test_instance_with_weights() {
        let instance = SolveInstance::with_weights(
            3,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::positive(1)]),
                Clause::new(vec![Literal::positive(2)]),
            ],
            vec![1.0, 2.0, 3.0],
        );

        assert!(instance.weights.is_some());
        assert_eq!(instance.weights.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_instance_weighted_satisfaction() {
        let instance = SolveInstance::with_weights(
            3,
            vec![
                Clause::new(vec![Literal::positive(0)]), // weight 1.0
                Clause::new(vec![Literal::positive(1)]), // weight 2.0
                Clause::new(vec![Literal::positive(2)]), // weight 3.0
            ],
            vec![1.0, 2.0, 3.0],
        );

        // All true: 1.0 + 2.0 + 3.0 = 6.0
        assert_eq!(instance.weighted_satisfaction(&[true, true, true]), 6.0);

        // Only x1=true: 2.0
        assert_eq!(instance.weighted_satisfaction(&[false, true, false]), 2.0);

        // None satisfied: 0.0
        assert_eq!(instance.weighted_satisfaction(&[false, false, false]), 0.0);
    }

    #[test]
    fn test_instance_weighted_satisfaction_unweighted() {
        // When no weights provided, each clause has implicit weight 1.0
        let instance = SolveInstance::new(
            2,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::positive(1)]),
            ],
        );

        assert_eq!(instance.weighted_satisfaction(&[true, true]), 2.0);
        assert_eq!(instance.weighted_satisfaction(&[true, false]), 1.0);
    }

    #[test]
    fn test_instance_empty() {
        let instance = SolveInstance::new(0, vec![]);
        assert!(instance.is_satisfied(&[]));
        assert_eq!(instance.count_satisfied(&[]), 0);
        assert_eq!(instance.weighted_satisfaction(&[]), 0.0);
    }

    #[test]
    fn test_literal_negate() {
        let pos = Literal::positive(5);
        let neg = pos.negate();
        assert_eq!(neg.var, 5);
        assert!(neg.negated);

        let pos_again = neg.negate();
        assert_eq!(pos_again.var, 5);
        assert!(!pos_again.negated);
    }

    #[test]
    fn test_clause_binary() {
        let clause = Clause::binary(Literal::positive(0), Literal::negative(1));
        assert_eq!(clause.literals.len(), 2);
        assert!(clause.is_satisfied(&[true, true]));
        assert!(clause.is_satisfied(&[true, false]));
        assert!(clause.is_satisfied(&[false, false]));
        assert!(!clause.is_satisfied(&[false, true]));
    }

    #[test]
    fn test_clause_ternary() {
        let clause = Clause::ternary(
            Literal::positive(0),
            Literal::positive(1),
            Literal::positive(2),
        );
        assert_eq!(clause.literals.len(), 3);

        // Only satisfied when at least one is true
        assert!(clause.is_satisfied(&[true, false, false]));
        assert!(clause.is_satisfied(&[false, true, false]));
        assert!(clause.is_satisfied(&[false, false, true]));
        assert!(!clause.is_satisfied(&[false, false, false]));
    }

    #[test]
    fn test_instance_add_clause() {
        let mut instance = SolveInstance::new(3, vec![Clause::new(vec![Literal::positive(0)])]);
        assert_eq!(instance.clauses.len(), 1);

        instance.add_clause(Clause::new(vec![Literal::positive(1)]));
        assert_eq!(instance.clauses.len(), 2);
    }

    #[test]
    fn test_instance_total_weight() {
        let instance = SolveInstance::with_weights(
            3,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::positive(1)]),
                Clause::new(vec![Literal::positive(2)]),
            ],
            vec![1.0, 2.0, 3.0],
        );
        assert_eq!(instance.total_weight(), 6.0);

        // Unweighted instance: total weight = number of clauses
        let unweighted = SolveInstance::new(
            2,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::positive(1)]),
            ],
        );
        assert_eq!(unweighted.total_weight(), 2.0);
    }

    #[test]
    fn test_instance_satisfaction_ratio() {
        let instance = SolveInstance::new(
            3,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::positive(1)]),
                Clause::new(vec![Literal::positive(2)]),
                Clause::new(vec![Literal::negative(0)]),
            ],
        );

        // x0=true satisfies clauses 0, but not 3 -> 1 satisfied
        // x1, x2 satisfy clauses 1, 2
        // Total: 3 out of 4
        assert_eq!(instance.satisfaction_ratio(&[true, true, true]), 0.75);
    }

    #[test]
    fn test_literal_ordering() {
        let a = Literal::positive(1);
        let b = Literal::positive(2);
        let c = Literal::negative(1);

        assert!(a < b); // Lower var comes first
        assert!(a < c); // Same var, positive before negative
    }

    #[test]
    fn test_clause_len_and_is_empty() {
        let empty = Clause::new(vec![]);
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let unit = Clause::unit(Literal::positive(0));
        assert!(!unit.is_empty());
        assert_eq!(unit.len(), 1);

        let binary = Clause::binary(Literal::positive(0), Literal::positive(1));
        assert_eq!(binary.len(), 2);
    }

    #[test]
    fn test_instance_num_clauses() {
        let instance = SolveInstance::new(
            3,
            vec![
                Clause::new(vec![Literal::positive(0)]),
                Clause::new(vec![Literal::positive(1)]),
            ],
        );
        assert_eq!(instance.num_clauses(), 2);
    }
}

// Implementation will go here after tests fail
