//! Proof artifacts and solve results for SAT/MaxSAT solving.
//!
//! This module provides types for representing solver outputs:
//! - [`SolveProof`]: Proof artifacts with checksums for verification
//! - [`SolveStatus`]: Result status (SAT, UNSAT, Unknown, Optimal)
//! - [`SolveStats`]: Solver statistics (iterations, timing, memory)
//! - [`SolveResult`]: Complete solver result combining status, proof, and stats
//!
//! # Proof Checksums
//!
//! Satisfying assignments include FNV-1a checksums to enable lightweight verification
//! that an assignment hasn't been corrupted during transfer or storage. This is
//! particularly useful for GPU-based solving where results are transferred across
//! memory boundaries.
//!
//! # Example
//!
//! ```
//! use xlog_solve::{SolveResult, SolveStatus, SolveStats};
//!
//! // Create a satisfying result
//! let result = SolveResult::satisfiable(vec![true, false, true]);
//! assert!(result.is_sat());
//!
//! // Add statistics
//! let stats = SolveStats::new(100, 5000, 1024);
//! let result = result.with_stats(stats);
//! assert_eq!(result.stats.iterations, 100);
//! ```

// =============================================================================
// FNV-1a Checksum Helper
// =============================================================================

/// Computes an FNV-1a checksum for a boolean assignment.
///
/// FNV-1a (Fowler-Noll-Vo) is a fast, non-cryptographic hash function
/// suitable for integrity verification. The implementation incorporates
/// both the index and value to ensure position-sensitivity.
///
/// # Arguments
///
/// * `assignment` - The boolean assignment to checksum
///
/// # Returns
///
/// A 64-bit FNV-1a hash of the assignment.
///
/// # Example
///
/// ```ignore
/// let checksum = compute_checksum(&[true, false, true]);
/// assert_ne!(checksum, 0);
/// ```
#[inline]
pub fn compute_checksum(assignment: &[bool]) -> u64 {
    // FNV-1a constants for 64-bit hash
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for (i, &val) in assignment.iter().enumerate() {
        // Combine index (upper 32 bits) and value (lower bit) for position-sensitivity
        hash ^= (i as u64) << 32 | (val as u64);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// =============================================================================
// SolveProof - Proof artifact from solver
// =============================================================================

/// Proof artifact from a SAT/MaxSAT solver.
///
/// Proofs provide evidence of the solver's conclusion. For satisfiable instances,
/// this includes the satisfying assignment with a checksum for verification.
/// For unsatisfiable instances, a proof of unsatisfiability is included.
///
/// # Variants
///
/// - [`Satisfying`](SolveProof::Satisfying): Contains a satisfying assignment with checksum
/// - [`Unsatisfiable`](SolveProof::Unsatisfiable): Proof that no satisfying assignment exists
/// - [`Approximate`](SolveProof::Approximate): Best-effort assignment that may not satisfy all clauses
/// - [`None`](SolveProof::None): No proof available (e.g., timeout before any result)
///
/// # Checksums
///
/// The checksum field uses FNV-1a hashing to enable lightweight verification that
/// an assignment hasn't been corrupted. This is particularly useful for:
/// - Verifying GPU-to-CPU transfers
/// - Detecting memory corruption
/// - Validating serialized/deserialized results
///
/// # Example
///
/// ```
/// use xlog_solve::SolveProof;
///
/// // Create a satisfying proof with automatic checksum
/// let proof = SolveProof::satisfying(vec![true, false, true]);
/// assert!(proof.is_satisfying());
///
/// // Extract the assignment
/// if let Some(assignment) = proof.assignment() {
///     assert_eq!(assignment, &[true, false, true]);
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum SolveProof {
    /// The instance is satisfiable with the given assignment.
    ///
    /// The checksum can be used to verify the assignment's integrity.
    Satisfying {
        /// The satisfying assignment (one boolean per variable).
        assignment: Vec<bool>,
        /// FNV-1a checksum of the assignment for integrity verification.
        checksum: u64,
    },

    /// The instance is unsatisfiable.
    ///
    /// The checksum can be used to verify the proof's integrity
    /// (e.g., if the solver produces resolution proofs).
    Unsatisfiable {
        /// Checksum for proof integrity verification.
        checksum: u64,
    },

    /// An approximate solution from an incomplete solver.
    ///
    /// This is typically produced by local search or heuristic solvers
    /// that may not find a complete satisfying assignment.
    Approximate {
        /// The best assignment found.
        assignment: Vec<bool>,
        /// Number of clauses satisfied by this assignment.
        satisfied_clauses: u32,
        /// Total number of clauses in the instance.
        total_clauses: u32,
        /// Number of solver iterations performed.
        iterations: u32,
    },

    /// No proof available.
    ///
    /// This occurs when the solver times out or is interrupted
    /// before producing any meaningful result.
    #[default]
    None,
}

impl SolveProof {
    /// Creates a satisfying proof with automatic checksum computation.
    ///
    /// # Arguments
    ///
    /// * `assignment` - The satisfying assignment
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveProof;
    ///
    /// let proof = SolveProof::satisfying(vec![true, false, true]);
    /// assert!(proof.is_satisfying());
    /// ```
    #[inline]
    pub fn satisfying(assignment: Vec<bool>) -> Self {
        let checksum = compute_checksum(&assignment);
        Self::Satisfying {
            assignment,
            checksum,
        }
    }

    /// Creates an unsatisfiability proof.
    ///
    /// The checksum is computed as a sentinel value for the empty proof.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveProof;
    ///
    /// let proof = SolveProof::unsatisfiable();
    /// assert!(!proof.is_satisfying());
    /// ```
    #[inline]
    pub fn unsatisfiable() -> Self {
        // Use a sentinel value indicating UNSAT - we compute checksum of empty
        // assignment and XOR with a sentinel to distinguish from SAT with empty formula
        let checksum = compute_checksum(&[]).wrapping_mul(0xDEADBEEFCAFEBABE);
        Self::Unsatisfiable { checksum }
    }

    /// Creates an approximate solution proof.
    ///
    /// This is typically used by incomplete solvers (like local search)
    /// that may not find a fully satisfying assignment.
    ///
    /// # Arguments
    ///
    /// * `assignment` - The best assignment found
    /// * `satisfied_clauses` - Number of clauses satisfied
    /// * `total_clauses` - Total number of clauses
    /// * `iterations` - Number of solver iterations performed
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveProof;
    ///
    /// let proof = SolveProof::approximate(vec![true, false], 8, 10, 1000);
    /// if let SolveProof::Approximate { satisfied_clauses, total_clauses, .. } = proof {
    ///     assert_eq!(satisfied_clauses, 8);
    ///     assert_eq!(total_clauses, 10);
    /// }
    /// ```
    #[inline]
    pub fn approximate(
        assignment: Vec<bool>,
        satisfied_clauses: u32,
        total_clauses: u32,
        iterations: u32,
    ) -> Self {
        Self::Approximate {
            assignment,
            satisfied_clauses,
            total_clauses,
            iterations,
        }
    }

    /// Returns true if this is a satisfying proof.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveProof;
    ///
    /// assert!(SolveProof::satisfying(vec![true]).is_satisfying());
    /// assert!(!SolveProof::unsatisfiable().is_satisfying());
    /// ```
    #[inline]
    pub fn is_satisfying(&self) -> bool {
        matches!(self, Self::Satisfying { .. })
    }

    /// Returns the assignment if this proof contains one.
    ///
    /// Returns `Some` for [`Satisfying`](SolveProof::Satisfying) and
    /// [`Approximate`](SolveProof::Approximate) variants, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveProof;
    ///
    /// let proof = SolveProof::satisfying(vec![true, false]);
    /// assert_eq!(proof.assignment(), Some(&[true, false][..]));
    ///
    /// let unsat = SolveProof::unsatisfiable();
    /// assert_eq!(unsat.assignment(), None);
    /// ```
    #[inline]
    pub fn assignment(&self) -> Option<&[bool]> {
        match self {
            Self::Satisfying { assignment, .. } => Some(assignment),
            Self::Approximate { assignment, .. } => Some(assignment),
            Self::Unsatisfiable { .. } | Self::None => None,
        }
    }

    /// Returns the checksum if this proof has one.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveProof;
    ///
    /// let proof = SolveProof::satisfying(vec![true]);
    /// assert!(proof.checksum().is_some());
    /// ```
    #[inline]
    pub fn checksum(&self) -> Option<u64> {
        match self {
            Self::Satisfying { checksum, .. } => Some(*checksum),
            Self::Unsatisfiable { checksum } => Some(*checksum),
            Self::Approximate { .. } | Self::None => None,
        }
    }

    /// Verifies the integrity of a satisfying assignment by recomputing its checksum.
    ///
    /// Returns `true` if the checksum matches, `false` if corrupted.
    /// Returns `None` for non-satisfying proofs.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveProof;
    ///
    /// let proof = SolveProof::satisfying(vec![true, false]);
    /// assert_eq!(proof.verify_checksum(), Some(true));
    /// ```
    #[inline]
    pub fn verify_checksum(&self) -> Option<bool> {
        match self {
            Self::Satisfying {
                assignment,
                checksum,
            } => Some(compute_checksum(assignment) == *checksum),
            _ => None,
        }
    }

    /// Returns the satisfaction ratio for approximate proofs.
    ///
    /// Returns `Some(ratio)` for [`Approximate`](SolveProof::Approximate) where
    /// ratio = satisfied_clauses / total_clauses. Returns `None` for other variants.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveProof;
    ///
    /// let proof = SolveProof::approximate(vec![true], 8, 10, 100);
    /// assert_eq!(proof.satisfaction_ratio(), Some(0.8));
    /// ```
    #[inline]
    pub fn satisfaction_ratio(&self) -> Option<f64> {
        match self {
            Self::Approximate {
                satisfied_clauses,
                total_clauses,
                ..
            } => {
                if *total_clauses == 0 {
                    Some(0.0)
                } else {
                    Some(*satisfied_clauses as f64 / *total_clauses as f64)
                }
            }
            _ => None,
        }
    }
}

// =============================================================================
// SolveStatus - Result status from solver
// =============================================================================

/// Status of a solve operation.
///
/// Represents the outcome of a SAT/MaxSAT solver execution.
///
/// # Variants
///
/// - [`Sat`](SolveStatus::Sat): Instance is satisfiable
/// - [`Unsat`](SolveStatus::Unsat): Instance is unsatisfiable
/// - [`Unknown`](SolveStatus::Unknown): Could not determine (timeout/resource limit)
/// - [`Optimal`](SolveStatus::Optimal): Found optimal solution with given value (MaxSAT)
///
/// # Example
///
/// ```
/// use xlog_solve::SolveStatus;
///
/// let status = SolveStatus::Sat;
/// assert!(status.is_satisfiable());
/// assert!(status.is_determined());
///
/// let optimal = SolveStatus::Optimal(42);
/// assert_eq!(optimal.optimal_value(), Some(42));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SolveStatus {
    /// Instance is satisfiable.
    ///
    /// A satisfying assignment exists and should be available in the proof.
    Sat,

    /// Instance is unsatisfiable.
    ///
    /// No assignment can satisfy all clauses.
    Unsat,

    /// Could not determine satisfiability.
    ///
    /// The solver did not reach a definitive conclusion, typically due to:
    /// - Timeout
    /// - Memory limit
    /// - Iteration limit
    /// - Incomplete search (for approximate solvers)
    #[default]
    Unknown,

    /// Found optimal solution with given value.
    ///
    /// For MaxSAT problems, this indicates the optimal objective value
    /// (e.g., maximum number of satisfied clauses or weighted sum).
    Optimal(u64),
}

impl SolveStatus {
    /// Returns true if the status indicates satisfiability.
    ///
    /// Returns `true` for [`Sat`](SolveStatus::Sat) and [`Optimal`](SolveStatus::Optimal).
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveStatus;
    ///
    /// assert!(SolveStatus::Sat.is_satisfiable());
    /// assert!(SolveStatus::Optimal(5).is_satisfiable());
    /// assert!(!SolveStatus::Unsat.is_satisfiable());
    /// ```
    #[inline]
    pub fn is_satisfiable(&self) -> bool {
        matches!(self, Self::Sat | Self::Optimal(_))
    }

    /// Returns true if the status indicates unsatisfiability.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveStatus;
    ///
    /// assert!(SolveStatus::Unsat.is_unsatisfiable());
    /// assert!(!SolveStatus::Sat.is_unsatisfiable());
    /// ```
    #[inline]
    pub fn is_unsatisfiable(&self) -> bool {
        matches!(self, Self::Unsat)
    }

    /// Returns true if the solver reached a definitive conclusion.
    ///
    /// Returns `true` for all variants except [`Unknown`](SolveStatus::Unknown).
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveStatus;
    ///
    /// assert!(SolveStatus::Sat.is_determined());
    /// assert!(SolveStatus::Unsat.is_determined());
    /// assert!(SolveStatus::Optimal(5).is_determined());
    /// assert!(!SolveStatus::Unknown.is_determined());
    /// ```
    #[inline]
    pub fn is_determined(&self) -> bool {
        !matches!(self, Self::Unknown)
    }

    /// Returns the optimal value if this is an [`Optimal`](SolveStatus::Optimal) status.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveStatus;
    ///
    /// assert_eq!(SolveStatus::Optimal(42).optimal_value(), Some(42));
    /// assert_eq!(SolveStatus::Sat.optimal_value(), None);
    /// ```
    #[inline]
    pub fn optimal_value(&self) -> Option<u64> {
        match self {
            Self::Optimal(v) => Some(*v),
            _ => None,
        }
    }
}

// =============================================================================
// SolveStats - Statistics from solving
// =============================================================================

/// Statistics from a solve operation.
///
/// Tracks performance metrics from solver execution including
/// iteration count, timing, and memory usage.
///
/// # Builder Pattern
///
/// `SolveStats` supports a builder pattern for convenient construction:
///
/// ```
/// use xlog_solve::SolveStats;
///
/// let stats = SolveStats::default()
///     .with_iterations(100)
///     .with_duration_us(5000)
///     .with_peak_memory(1024);
///
/// assert_eq!(stats.iterations, 100);
/// assert_eq!(stats.duration_us, 5000);
/// assert_eq!(stats.peak_memory, 1024);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SolveStats {
    /// Number of iterations performed by the solver.
    ///
    /// For iterative solvers (like CLS), this is the number of
    /// gradient descent steps. For CDCL solvers, this might be
    /// the number of decisions or conflicts.
    pub iterations: u32,

    /// Wall-clock time spent solving, in microseconds.
    ///
    /// This includes all solver operations but typically excludes
    /// instance setup and result extraction.
    pub duration_us: u64,

    /// Peak memory usage during solving, in bytes.
    ///
    /// This tracks the maximum memory allocated by the solver,
    /// useful for profiling and resource management.
    pub peak_memory: u64,
}

impl SolveStats {
    /// Creates new statistics with all fields specified.
    ///
    /// # Arguments
    ///
    /// * `iterations` - Number of solver iterations
    /// * `duration_us` - Solve time in microseconds
    /// * `peak_memory` - Peak memory usage in bytes
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveStats;
    ///
    /// let stats = SolveStats::new(100, 5000, 1024);
    /// assert_eq!(stats.iterations, 100);
    /// ```
    #[inline]
    pub const fn new(iterations: u32, duration_us: u64, peak_memory: u64) -> Self {
        Self {
            iterations,
            duration_us,
            peak_memory,
        }
    }

    /// Sets the iteration count, consuming and returning self.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveStats;
    ///
    /// let stats = SolveStats::default().with_iterations(100);
    /// assert_eq!(stats.iterations, 100);
    /// ```
    #[inline]
    pub const fn with_iterations(mut self, iterations: u32) -> Self {
        self.iterations = iterations;
        self
    }

    /// Sets the duration in microseconds, consuming and returning self.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveStats;
    ///
    /// let stats = SolveStats::default().with_duration_us(5000);
    /// assert_eq!(stats.duration_us, 5000);
    /// ```
    #[inline]
    pub const fn with_duration_us(mut self, duration_us: u64) -> Self {
        self.duration_us = duration_us;
        self
    }

    /// Sets the peak memory usage in bytes, consuming and returning self.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveStats;
    ///
    /// let stats = SolveStats::default().with_peak_memory(1024);
    /// assert_eq!(stats.peak_memory, 1024);
    /// ```
    #[inline]
    pub const fn with_peak_memory(mut self, peak_memory: u64) -> Self {
        self.peak_memory = peak_memory;
        self
    }

    /// Returns the duration in milliseconds (rounded down).
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveStats;
    ///
    /// let stats = SolveStats::new(0, 5500, 0);
    /// assert_eq!(stats.duration_ms(), 5);
    /// ```
    #[inline]
    pub const fn duration_ms(&self) -> u64 {
        self.duration_us / 1000
    }

    /// Returns the duration in seconds as a floating-point value.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveStats;
    ///
    /// let stats = SolveStats::new(0, 1_500_000, 0);
    /// assert_eq!(stats.duration_secs(), 1.5);
    /// ```
    #[inline]
    pub fn duration_secs(&self) -> f64 {
        self.duration_us as f64 / 1_000_000.0
    }

    /// Returns iterations per second (throughput).
    ///
    /// Returns 0.0 if duration is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveStats;
    ///
    /// let stats = SolveStats::new(1000, 1_000_000, 0);
    /// assert_eq!(stats.iterations_per_sec(), 1000.0);
    /// ```
    #[inline]
    pub fn iterations_per_sec(&self) -> f64 {
        if self.duration_us == 0 {
            0.0
        } else {
            (self.iterations as f64 * 1_000_000.0) / self.duration_us as f64
        }
    }
}

// =============================================================================
// SolveResult - Complete solver result
// =============================================================================

/// Complete result from a solve operation.
///
/// Combines the solve status, proof artifact, and statistics into
/// a single comprehensive result structure.
///
/// # Construction
///
/// Use the constructor methods for common result types:
///
/// ```
/// use xlog_solve::{SolveResult, SolveStats};
///
/// // Satisfiable result
/// let sat = SolveResult::satisfiable(vec![true, false, true]);
/// assert!(sat.is_sat());
///
/// // Unsatisfiable result
/// let unsat = SolveResult::unsatisfiable();
/// assert!(unsat.is_unsat());
///
/// // Unknown result (e.g., timeout)
/// let unknown = SolveResult::unknown(1000);
/// assert!(!unknown.is_sat());
///
/// // With custom statistics
/// let stats = SolveStats::new(100, 5000, 1024);
/// let result = SolveResult::satisfiable(vec![true]).with_stats(stats);
/// ```
///
/// # Example
///
/// ```
/// use xlog_solve::SolveResult;
///
/// let result = SolveResult::satisfiable(vec![true, false]);
///
/// if result.is_sat() {
///     if let Some(assignment) = result.assignment() {
///         println!("Found assignment: {:?}", assignment);
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct SolveResult {
    /// The solve status (SAT, UNSAT, Unknown, Optimal).
    pub status: SolveStatus,

    /// The proof artifact from solving.
    pub proof: SolveProof,

    /// Statistics from the solve operation.
    pub stats: SolveStats,
}

impl SolveResult {
    /// Creates a satisfiable result with the given assignment.
    ///
    /// Automatically computes a checksum for the assignment.
    ///
    /// # Arguments
    ///
    /// * `assignment` - The satisfying assignment
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveResult;
    ///
    /// let result = SolveResult::satisfiable(vec![true, false, true]);
    /// assert!(result.is_sat());
    /// ```
    #[inline]
    pub fn satisfiable(assignment: Vec<bool>) -> Self {
        Self {
            status: SolveStatus::Sat,
            proof: SolveProof::satisfying(assignment),
            stats: SolveStats::default(),
        }
    }

    /// Creates an unsatisfiable result.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveResult;
    ///
    /// let result = SolveResult::unsatisfiable();
    /// assert!(result.is_unsat());
    /// ```
    #[inline]
    pub fn unsatisfiable() -> Self {
        Self {
            status: SolveStatus::Unsat,
            proof: SolveProof::unsatisfiable(),
            stats: SolveStats::default(),
        }
    }

    /// Creates an unknown result (e.g., timeout).
    ///
    /// # Arguments
    ///
    /// * `iterations` - Number of iterations performed before giving up
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveResult;
    ///
    /// let result = SolveResult::unknown(1000);
    /// assert!(!result.is_sat());
    /// assert!(!result.is_unsat());
    /// ```
    #[inline]
    pub fn unknown(iterations: u32) -> Self {
        Self {
            status: SolveStatus::Unknown,
            proof: SolveProof::None,
            stats: SolveStats::default().with_iterations(iterations),
        }
    }

    /// Creates an optimal result (for MaxSAT).
    ///
    /// # Arguments
    ///
    /// * `assignment` - The optimal assignment
    /// * `optimal_value` - The optimal objective value
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::{SolveResult, SolveStatus};
    ///
    /// let result = SolveResult::optimal(vec![true, false], 42);
    /// assert!(matches!(result.status, SolveStatus::Optimal(42)));
    /// ```
    #[inline]
    pub fn optimal(assignment: Vec<bool>, optimal_value: u64) -> Self {
        Self {
            status: SolveStatus::Optimal(optimal_value),
            proof: SolveProof::satisfying(assignment),
            stats: SolveStats::default(),
        }
    }

    /// Creates an approximate result from an incomplete solver.
    ///
    /// # Arguments
    ///
    /// * `assignment` - The best assignment found
    /// * `satisfied_clauses` - Number of satisfied clauses
    /// * `total_clauses` - Total number of clauses
    /// * `iterations` - Number of solver iterations
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveResult;
    ///
    /// let result = SolveResult::approximate(vec![true, false], 8, 10, 1000);
    /// assert!(!result.is_sat());  // Not definitively SAT
    /// ```
    #[inline]
    pub fn approximate(
        assignment: Vec<bool>,
        satisfied_clauses: u32,
        total_clauses: u32,
        iterations: u32,
    ) -> Self {
        Self {
            status: SolveStatus::Unknown,
            proof: SolveProof::approximate(
                assignment,
                satisfied_clauses,
                total_clauses,
                iterations,
            ),
            stats: SolveStats::default().with_iterations(iterations),
        }
    }

    /// Updates the statistics, consuming and returning self.
    ///
    /// # Arguments
    ///
    /// * `stats` - The new statistics
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::{SolveResult, SolveStats};
    ///
    /// let stats = SolveStats::new(100, 5000, 1024);
    /// let result = SolveResult::satisfiable(vec![true]).with_stats(stats);
    /// assert_eq!(result.stats.iterations, 100);
    /// ```
    #[inline]
    pub fn with_stats(mut self, stats: SolveStats) -> Self {
        self.stats = stats;
        self
    }

    /// Returns true if the result is satisfiable.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveResult;
    ///
    /// assert!(SolveResult::satisfiable(vec![true]).is_sat());
    /// assert!(!SolveResult::unsatisfiable().is_sat());
    /// ```
    #[inline]
    pub fn is_sat(&self) -> bool {
        self.status.is_satisfiable()
    }

    /// Returns true if the result is unsatisfiable.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveResult;
    ///
    /// assert!(SolveResult::unsatisfiable().is_unsat());
    /// assert!(!SolveResult::satisfiable(vec![true]).is_unsat());
    /// ```
    #[inline]
    pub fn is_unsat(&self) -> bool {
        self.status.is_unsatisfiable()
    }

    /// Returns the assignment if one is available.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveResult;
    ///
    /// let result = SolveResult::satisfiable(vec![true, false]);
    /// assert_eq!(result.assignment(), Some(&[true, false][..]));
    /// ```
    #[inline]
    pub fn assignment(&self) -> Option<&[bool]> {
        self.proof.assignment()
    }

    /// Returns the optimal value if this is a MaxSAT result.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveResult;
    ///
    /// let result = SolveResult::optimal(vec![true], 42);
    /// assert_eq!(result.optimal_value(), Some(42));
    /// ```
    #[inline]
    pub fn optimal_value(&self) -> Option<u64> {
        self.status.optimal_value()
    }

    /// Verifies the integrity of the proof checksum.
    ///
    /// Returns `Some(true)` if the checksum is valid, `Some(false)` if corrupted,
    /// or `None` if verification is not applicable.
    ///
    /// # Example
    ///
    /// ```
    /// use xlog_solve::SolveResult;
    ///
    /// let result = SolveResult::satisfiable(vec![true, false]);
    /// assert_eq!(result.verify_proof(), Some(true));
    /// ```
    #[inline]
    pub fn verify_proof(&self) -> Option<bool> {
        self.proof.verify_checksum()
    }
}

impl Default for SolveResult {
    fn default() -> Self {
        Self {
            status: SolveStatus::Unknown,
            proof: SolveProof::None,
            stats: SolveStats::default(),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // SolveProof Tests
    // ==========================================================================

    #[test]
    fn test_solve_proof_satisfying() {
        let assignment = vec![true, false, true];
        let proof = SolveProof::satisfying(assignment.clone());
        match proof {
            SolveProof::Satisfying {
                assignment: a,
                checksum,
            } => {
                assert_eq!(a, assignment);
                assert_ne!(checksum, 0); // Checksum should be computed
            }
            _ => panic!("Expected Satisfying proof"),
        }
    }

    #[test]
    fn test_solve_proof_unsatisfiable() {
        let proof = SolveProof::unsatisfiable();
        match proof {
            SolveProof::Unsatisfiable { checksum } => {
                // Checksum for empty unsatisfiability proof should be consistent
                assert_ne!(checksum, 0);
            }
            _ => panic!("Expected Unsatisfiable proof"),
        }
    }

    #[test]
    fn test_solve_proof_approximate() {
        let assignment = vec![true, false];
        let proof = SolveProof::approximate(assignment.clone(), 5, 10, 100);
        match proof {
            SolveProof::Approximate {
                assignment: a,
                satisfied_clauses,
                total_clauses,
                iterations,
            } => {
                assert_eq!(a, assignment);
                assert_eq!(satisfied_clauses, 5);
                assert_eq!(total_clauses, 10);
                assert_eq!(iterations, 100);
            }
            _ => panic!("Expected Approximate proof"),
        }
    }

    #[test]
    fn test_solve_proof_none() {
        let proof = SolveProof::None;
        assert!(matches!(proof, SolveProof::None));
    }

    #[test]
    fn test_solve_proof_checksum_deterministic() {
        // Same assignment should produce same checksum
        let assignment = vec![true, false, true, false];
        let proof1 = SolveProof::satisfying(assignment.clone());
        let proof2 = SolveProof::satisfying(assignment);

        let checksum1 = match proof1 {
            SolveProof::Satisfying { checksum, .. } => checksum,
            _ => panic!("Expected Satisfying"),
        };
        let checksum2 = match proof2 {
            SolveProof::Satisfying { checksum, .. } => checksum,
            _ => panic!("Expected Satisfying"),
        };

        assert_eq!(checksum1, checksum2);
    }

    #[test]
    fn test_solve_proof_checksum_different_assignments() {
        // Different assignments should produce different checksums
        let proof1 = SolveProof::satisfying(vec![true, false]);
        let proof2 = SolveProof::satisfying(vec![false, true]);

        let checksum1 = match proof1 {
            SolveProof::Satisfying { checksum, .. } => checksum,
            _ => panic!("Expected Satisfying"),
        };
        let checksum2 = match proof2 {
            SolveProof::Satisfying { checksum, .. } => checksum,
            _ => panic!("Expected Satisfying"),
        };

        assert_ne!(checksum1, checksum2);
    }

    #[test]
    fn test_solve_proof_checksum() {
        // Test from the task spec
        let proof = SolveProof::Satisfying {
            assignment: vec![true, false],
            checksum: 12345,
        };
        match proof {
            SolveProof::Satisfying { checksum, .. } => {
                assert_eq!(checksum, 12345);
            }
            _ => panic!("Wrong proof type"),
        }
    }

    #[test]
    fn test_solve_proof_is_satisfying() {
        let sat_proof = SolveProof::satisfying(vec![true]);
        assert!(sat_proof.is_satisfying());

        let unsat_proof = SolveProof::unsatisfiable();
        assert!(!unsat_proof.is_satisfying());

        let approx_proof = SolveProof::approximate(vec![true], 1, 1, 1);
        assert!(!approx_proof.is_satisfying());

        assert!(!SolveProof::None.is_satisfying());
    }

    #[test]
    fn test_solve_proof_assignment() {
        let assignment = vec![true, false, true];
        let sat_proof = SolveProof::satisfying(assignment.clone());
        assert_eq!(sat_proof.assignment(), Some(&assignment[..]));

        let approx_proof = SolveProof::approximate(assignment.clone(), 2, 3, 10);
        assert_eq!(approx_proof.assignment(), Some(&assignment[..]));

        let unsat_proof = SolveProof::unsatisfiable();
        assert_eq!(unsat_proof.assignment(), None);

        assert_eq!(SolveProof::None.assignment(), None);
    }

    #[test]
    fn test_solve_proof_verify_checksum() {
        let proof = SolveProof::satisfying(vec![true, false, true]);
        assert_eq!(proof.verify_checksum(), Some(true));

        // Manually create a corrupted proof
        let corrupted = SolveProof::Satisfying {
            assignment: vec![true, false, true],
            checksum: 12345, // Wrong checksum
        };
        assert_eq!(corrupted.verify_checksum(), Some(false));
    }

    #[test]
    fn test_solve_proof_satisfaction_ratio() {
        let proof = SolveProof::approximate(vec![true], 8, 10, 100);
        assert_eq!(proof.satisfaction_ratio(), Some(0.8));

        let full = SolveProof::approximate(vec![true], 10, 10, 100);
        assert_eq!(full.satisfaction_ratio(), Some(1.0));

        let empty = SolveProof::approximate(vec![], 0, 0, 100);
        assert_eq!(empty.satisfaction_ratio(), Some(0.0));

        assert_eq!(
            SolveProof::satisfying(vec![true]).satisfaction_ratio(),
            None
        );
    }

    #[test]
    fn test_solve_proof_default() {
        let proof = SolveProof::default();
        assert!(matches!(proof, SolveProof::None));
    }

    // ==========================================================================
    // SolveStatus Tests
    // ==========================================================================

    #[test]
    fn test_solve_status_sat() {
        let status = SolveStatus::Sat;
        assert!(status.is_satisfiable());
        assert!(!status.is_unsatisfiable());
        assert!(status.is_determined());
    }

    #[test]
    fn test_solve_status_unsat() {
        let status = SolveStatus::Unsat;
        assert!(!status.is_satisfiable());
        assert!(status.is_unsatisfiable());
        assert!(status.is_determined());
    }

    #[test]
    fn test_solve_status_unknown() {
        let status = SolveStatus::Unknown;
        assert!(!status.is_satisfiable());
        assert!(!status.is_unsatisfiable());
        assert!(!status.is_determined());
    }

    #[test]
    fn test_solve_status_optimal() {
        let status = SolveStatus::Optimal(42);
        assert!(status.is_satisfiable());
        assert!(!status.is_unsatisfiable());
        assert!(status.is_determined());
        assert_eq!(status.optimal_value(), Some(42));
    }

    #[test]
    fn test_solve_status_optimal_value() {
        assert_eq!(SolveStatus::Sat.optimal_value(), None);
        assert_eq!(SolveStatus::Unsat.optimal_value(), None);
        assert_eq!(SolveStatus::Unknown.optimal_value(), None);
        assert_eq!(SolveStatus::Optimal(100).optimal_value(), Some(100));
    }

    #[test]
    fn test_solve_status_eq() {
        assert_eq!(SolveStatus::Sat, SolveStatus::Sat);
        assert_eq!(SolveStatus::Unsat, SolveStatus::Unsat);
        assert_eq!(SolveStatus::Unknown, SolveStatus::Unknown);
        assert_eq!(SolveStatus::Optimal(5), SolveStatus::Optimal(5));
        assert_ne!(SolveStatus::Optimal(5), SolveStatus::Optimal(6));
        assert_ne!(SolveStatus::Sat, SolveStatus::Unsat);
    }

    #[test]
    fn test_solve_status_default() {
        let status = SolveStatus::default();
        assert!(matches!(status, SolveStatus::Unknown));
    }

    // ==========================================================================
    // SolveStats Tests
    // ==========================================================================

    #[test]
    fn test_solve_stats_default() {
        let stats = SolveStats::default();
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.duration_us, 0);
        assert_eq!(stats.peak_memory, 0);
    }

    #[test]
    fn test_solve_stats_new() {
        let stats = SolveStats::new(100, 5000, 1024);
        assert_eq!(stats.iterations, 100);
        assert_eq!(stats.duration_us, 5000);
        assert_eq!(stats.peak_memory, 1024);
    }

    #[test]
    fn test_solve_stats_with_iterations() {
        let stats = SolveStats::default().with_iterations(500);
        assert_eq!(stats.iterations, 500);
        assert_eq!(stats.duration_us, 0);
        assert_eq!(stats.peak_memory, 0);
    }

    #[test]
    fn test_solve_stats_with_duration_us() {
        let stats = SolveStats::default().with_duration_us(10000);
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.duration_us, 10000);
        assert_eq!(stats.peak_memory, 0);
    }

    #[test]
    fn test_solve_stats_with_peak_memory() {
        let stats = SolveStats::default().with_peak_memory(2048);
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.duration_us, 0);
        assert_eq!(stats.peak_memory, 2048);
    }

    #[test]
    fn test_solve_stats_chained_builders() {
        let stats = SolveStats::default()
            .with_iterations(100)
            .with_duration_us(5000)
            .with_peak_memory(1024);
        assert_eq!(stats.iterations, 100);
        assert_eq!(stats.duration_us, 5000);
        assert_eq!(stats.peak_memory, 1024);
    }

    #[test]
    fn test_solve_stats_duration_ms() {
        let stats = SolveStats::new(0, 5500, 0);
        assert_eq!(stats.duration_ms(), 5);
    }

    #[test]
    fn test_solve_stats_duration_secs() {
        let stats = SolveStats::new(0, 1_500_000, 0);
        assert_eq!(stats.duration_secs(), 1.5);
    }

    #[test]
    fn test_solve_stats_iterations_per_sec() {
        let stats = SolveStats::new(1000, 1_000_000, 0);
        assert_eq!(stats.iterations_per_sec(), 1000.0);

        let zero = SolveStats::new(1000, 0, 0);
        assert_eq!(zero.iterations_per_sec(), 0.0);
    }

    // ==========================================================================
    // SolveResult Tests
    // ==========================================================================

    #[test]
    fn test_solve_result_sat() {
        let result = SolveResult::satisfiable(vec![true, false, true]);
        assert!(matches!(result.status, SolveStatus::Sat));
        assert!(result.proof.is_satisfying());
    }

    #[test]
    fn test_solve_result_unsat() {
        let result = SolveResult::unsatisfiable();
        assert!(matches!(result.status, SolveStatus::Unsat));
        assert!(matches!(result.proof, SolveProof::Unsatisfiable { .. }));
    }

    #[test]
    fn test_solve_result_unknown() {
        let result = SolveResult::unknown(500);
        assert!(matches!(result.status, SolveStatus::Unknown));
        assert_eq!(result.stats.iterations, 500);
    }

    #[test]
    fn test_solve_result_optimal() {
        let result = SolveResult::optimal(vec![true, false], 42);
        assert!(matches!(result.status, SolveStatus::Optimal(42)));
        assert!(result.proof.is_satisfying());
    }

    #[test]
    fn test_solve_result_with_stats() {
        let stats = SolveStats::new(100, 5000, 1024);
        let result = SolveResult::satisfiable(vec![true]).with_stats(stats);
        assert_eq!(result.stats.iterations, 100);
        assert_eq!(result.stats.duration_us, 5000);
        assert_eq!(result.stats.peak_memory, 1024);
    }

    #[test]
    fn test_solve_result_is_sat() {
        assert!(SolveResult::satisfiable(vec![true]).is_sat());
        assert!(!SolveResult::unsatisfiable().is_sat());
        assert!(!SolveResult::unknown(0).is_sat());
        assert!(SolveResult::optimal(vec![true], 5).is_sat());
    }

    #[test]
    fn test_solve_result_is_unsat() {
        assert!(!SolveResult::satisfiable(vec![true]).is_unsat());
        assert!(SolveResult::unsatisfiable().is_unsat());
        assert!(!SolveResult::unknown(0).is_unsat());
        assert!(!SolveResult::optimal(vec![true], 5).is_unsat());
    }

    #[test]
    fn test_solve_result_assignment() {
        let assignment = vec![true, false, true];
        let sat_result = SolveResult::satisfiable(assignment.clone());
        assert_eq!(sat_result.assignment(), Some(&assignment[..]));

        let unsat_result = SolveResult::unsatisfiable();
        assert_eq!(unsat_result.assignment(), None);
    }

    #[test]
    fn test_solve_result_approximate() {
        let result = SolveResult::approximate(vec![true, false], 8, 10, 1000);
        assert!(matches!(result.status, SolveStatus::Unknown));
        match &result.proof {
            SolveProof::Approximate {
                satisfied_clauses,
                total_clauses,
                iterations,
                ..
            } => {
                assert_eq!(*satisfied_clauses, 8);
                assert_eq!(*total_clauses, 10);
                assert_eq!(*iterations, 1000);
            }
            _ => panic!("Expected Approximate proof"),
        }
    }

    #[test]
    fn test_solve_result_optimal_value() {
        let result = SolveResult::optimal(vec![true], 42);
        assert_eq!(result.optimal_value(), Some(42));

        let sat = SolveResult::satisfiable(vec![true]);
        assert_eq!(sat.optimal_value(), None);
    }

    #[test]
    fn test_solve_result_verify_proof() {
        let result = SolveResult::satisfiable(vec![true, false]);
        assert_eq!(result.verify_proof(), Some(true));
    }

    #[test]
    fn test_solve_result_default() {
        let result = SolveResult::default();
        assert!(matches!(result.status, SolveStatus::Unknown));
        assert!(matches!(result.proof, SolveProof::None));
    }

    // ==========================================================================
    // FNV-1a Checksum Tests
    // ==========================================================================

    #[test]
    fn test_compute_checksum_empty() {
        let checksum = compute_checksum(&[]);
        // FNV-1a offset basis
        assert_eq!(checksum, 0xcbf29ce484222325);
    }

    #[test]
    fn test_compute_checksum_single() {
        let checksum_true = compute_checksum(&[true]);
        let checksum_false = compute_checksum(&[false]);
        assert_ne!(checksum_true, checksum_false);
    }

    #[test]
    fn test_compute_checksum_deterministic() {
        let assignment = vec![true, false, true, false, true];
        let c1 = compute_checksum(&assignment);
        let c2 = compute_checksum(&assignment);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_compute_checksum_order_sensitive() {
        // Different order should produce different checksum
        let c1 = compute_checksum(&[true, false]);
        let c2 = compute_checksum(&[false, true]);
        assert_ne!(c1, c2);
    }
}
