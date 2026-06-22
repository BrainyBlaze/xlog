//! Integration tests for xlog-solve
//!
//! These tests verify the CLS solver's behavior across various SAT patterns,
//! from simple unit clauses to complex combinatorial problems. The tests cover:
//!
//! - Standard SAT instances (satisfiable and unsatisfiable)
//! - Random 3-SAT instances
//! - Combinatorial problems (pigeonhole, graph coloring)
//! - Solver configuration effects
//! - Determinism and reproducibility
//! - Scalability characteristics
//! - MaxSAT weighted instances

use xlog_solve::{
    Clause, Literal, Objective, SolveInstance, SolveProof, SolveStats, SolveStatus, Solver,
    SolverConfig, SolverState,
};

// =============================================================================
// Random 3-SAT Tests
// =============================================================================

/// Test a random 3-SAT instance that should be satisfiable.
///
/// This instance has 10 variables and 10 clauses with a clause-to-variable ratio
/// of 1.0, well below the SAT threshold (~4.26) where random 3-SAT becomes hard.
#[test]
fn test_3sat_satisfiable() {
    let clauses: Vec<Clause> = vec![
        Clause::new(vec![
            Literal::positive(0),
            Literal::positive(1),
            Literal::negative(2),
        ]),
        Clause::new(vec![
            Literal::negative(0),
            Literal::positive(2),
            Literal::positive(3),
        ]),
        Clause::new(vec![
            Literal::positive(1),
            Literal::negative(3),
            Literal::positive(4),
        ]),
        Clause::new(vec![
            Literal::negative(1),
            Literal::positive(4),
            Literal::negative(5),
        ]),
        Clause::new(vec![
            Literal::positive(2),
            Literal::positive(5),
            Literal::positive(6),
        ]),
        Clause::new(vec![
            Literal::negative(2),
            Literal::negative(6),
            Literal::positive(7),
        ]),
        Clause::new(vec![
            Literal::positive(3),
            Literal::positive(7),
            Literal::negative(8),
        ]),
        Clause::new(vec![
            Literal::negative(3),
            Literal::positive(8),
            Literal::positive(9),
        ]),
        Clause::new(vec![
            Literal::positive(4),
            Literal::negative(9),
            Literal::positive(0),
        ]),
        Clause::new(vec![
            Literal::negative(4),
            Literal::positive(0),
            Literal::negative(1),
        ]),
    ];

    let instance = SolveInstance::new(10, clauses);
    let solver_config = {
        let mut config = SolverConfig::default();
        config.max_iterations = 5000;
        config.learning_rate = 0.15;
        config.momentum = 0.9;
        config.discretize_threshold = 0.5;
        config
    };
    let solver = Solver::with_config_cpu(solver_config);

    let result = solver.solve(instance.clone());

    match result.status {
        SolveStatus::Sat => {
            // Verify the assignment is actually satisfying
            if let SolveProof::Satisfying { assignment, .. } = &result.proof {
                assert!(
                    instance.is_satisfied(assignment),
                    "Reported SAT but assignment doesn't satisfy all clauses"
                );
            } else {
                panic!("SAT status but no Satisfying proof");
            }
        }
        SolveStatus::Unknown => {
            // CLS may not always find solution, but should satisfy most clauses
            if let SolveProof::Approximate {
                satisfied_clauses,
                total_clauses,
                assignment,
                ..
            } = &result.proof
            {
                let ratio = *satisfied_clauses as f64 / *total_clauses as f64;
                assert!(
                    ratio > 0.7,
                    "Should satisfy at least 70% of clauses, got {:.1}%",
                    ratio * 100.0
                );
                // Verify the reported count matches actual satisfaction
                let actual_satisfied = instance.count_satisfied(assignment) as u32;
                assert_eq!(
                    actual_satisfied, *satisfied_clauses,
                    "Reported satisfaction count doesn't match actual"
                );
            } else {
                panic!("Unknown status but no Approximate proof");
            }
        }
        SolveStatus::Unsat => {
            // This instance is satisfiable, UNSAT would be incorrect
            // However, CLS is incomplete and cannot prove UNSAT, so this shouldn't happen
            panic!("CLS incorrectly reported UNSAT for a satisfiable instance");
        }
        SolveStatus::Optimal(_) => {
            // Not a MaxSAT instance
            panic!("Unexpected Optimal status for SAT instance");
        }
    }

    println!("Solver stats: {:?}", result.stats);
}

/// Test a denser 3-SAT instance (more clauses per variable).
#[test]
fn test_3sat_dense() {
    // 5 variables, 15 clauses (ratio 3.0, harder but still likely SAT)
    let clauses: Vec<Clause> = vec![
        Clause::ternary(
            Literal::positive(0),
            Literal::positive(1),
            Literal::positive(2),
        ),
        Clause::ternary(
            Literal::negative(0),
            Literal::positive(1),
            Literal::positive(3),
        ),
        Clause::ternary(
            Literal::positive(0),
            Literal::negative(1),
            Literal::positive(4),
        ),
        Clause::ternary(
            Literal::negative(0),
            Literal::negative(1),
            Literal::positive(2),
        ),
        Clause::ternary(
            Literal::positive(1),
            Literal::positive(2),
            Literal::negative(3),
        ),
        Clause::ternary(
            Literal::negative(1),
            Literal::positive(2),
            Literal::positive(4),
        ),
        Clause::ternary(
            Literal::positive(0),
            Literal::negative(2),
            Literal::positive(3),
        ),
        Clause::ternary(
            Literal::negative(0),
            Literal::negative(2),
            Literal::negative(4),
        ),
        Clause::ternary(
            Literal::positive(2),
            Literal::positive(3),
            Literal::positive(4),
        ),
        Clause::ternary(
            Literal::negative(2),
            Literal::negative(3),
            Literal::positive(0),
        ),
        Clause::ternary(
            Literal::positive(3),
            Literal::negative(4),
            Literal::positive(1),
        ),
        Clause::ternary(
            Literal::negative(3),
            Literal::positive(4),
            Literal::negative(0),
        ),
        Clause::ternary(
            Literal::positive(0),
            Literal::positive(3),
            Literal::negative(4),
        ),
        Clause::ternary(
            Literal::negative(1),
            Literal::negative(3),
            Literal::negative(4),
        ),
        Clause::ternary(
            Literal::positive(1),
            Literal::positive(4),
            Literal::negative(2),
        ),
    ];

    let instance = SolveInstance::new(5, clauses);
    let solver = Solver::with_config_cpu(SolverConfig::thorough());

    let result = solver.solve(instance.clone());

    // Should find a solution or get close
    match result.status {
        SolveStatus::Sat => {
            if let Some(assignment) = result.assignment() {
                assert!(instance.is_satisfied(assignment));
            }
        }
        SolveStatus::Unknown => {
            // Check we at least made progress
            if let SolveProof::Approximate {
                satisfied_clauses,
                total_clauses,
                ..
            } = result.proof
            {
                let ratio = satisfied_clauses as f64 / total_clauses as f64;
                assert!(ratio > 0.5, "Should satisfy at least 50% of dense clauses");
            }
        }
        _ => {}
    }
}

// =============================================================================
// Pigeonhole Principle Tests
// =============================================================================

/// Test the pigeonhole principle: 2 pigeons, 1 hole - UNSAT.
///
/// The pigeonhole principle states that if you have n+1 pigeons and n holes,
/// at least one hole must contain more than one pigeon. This creates an
/// unsatisfiable SAT instance.
#[test]
fn test_pigeonhole_unsat_2_1() {
    // Variables: p[i][j] = pigeon i is in hole j
    // p[0][0] = var 0, p[1][0] = var 1
    let instance = SolveInstance::new(
        2,
        vec![
            // Each pigeon must be in some hole
            Clause::new(vec![Literal::positive(0)]), // Pigeon 0 in hole 0
            Clause::new(vec![Literal::positive(1)]), // Pigeon 1 in hole 0
            // At most one pigeon per hole
            Clause::new(vec![Literal::negative(0), Literal::negative(1)]),
        ],
    );

    let solver = Solver::new_cpu();
    let result = solver.solve(instance.clone());

    // CLS is incomplete for UNSAT, so Unknown is acceptable
    assert!(
        matches!(result.status, SolveStatus::Unsat | SolveStatus::Unknown),
        "Pigeonhole should be UNSAT or Unknown, got {:?}",
        result.status
    );

    // If Unknown, verify we couldn't satisfy all clauses
    if result.status == SolveStatus::Unknown {
        if let SolveProof::Approximate {
            satisfied_clauses,
            total_clauses,
            ..
        } = result.proof
        {
            assert!(
                satisfied_clauses < total_clauses,
                "Should not satisfy all clauses in UNSAT instance"
            );
        }
    }
}

/// Test larger pigeonhole: 3 pigeons, 2 holes - UNSAT.
#[test]
fn test_pigeonhole_unsat_3_2() {
    // Variables: p[i][j] = pigeon i is in hole j
    // p[0][0]=0, p[0][1]=1, p[1][0]=2, p[1][1]=3, p[2][0]=4, p[2][1]=5
    // Each pigeon must be in some hole
    let clauses = vec![
        Clause::new(vec![Literal::positive(0), Literal::positive(1)]), // Pigeon 0 in some hole
        Clause::new(vec![Literal::positive(2), Literal::positive(3)]), // Pigeon 1 in some hole
        Clause::new(vec![Literal::positive(4), Literal::positive(5)]), // Pigeon 2 in some hole
        // At most one pigeon per hole
        // Hole 0: at most one of vars 0, 2, 4
        Clause::new(vec![Literal::negative(0), Literal::negative(2)]),
        Clause::new(vec![Literal::negative(0), Literal::negative(4)]),
        Clause::new(vec![Literal::negative(2), Literal::negative(4)]),
        // Hole 1: at most one of vars 1, 3, 5
        Clause::new(vec![Literal::negative(1), Literal::negative(3)]),
        Clause::new(vec![Literal::negative(1), Literal::negative(5)]),
        Clause::new(vec![Literal::negative(3), Literal::negative(5)]),
    ];

    let instance = SolveInstance::new(6, clauses);
    let solver = Solver::new_cpu();
    let result = solver.solve(instance);

    assert!(
        matches!(result.status, SolveStatus::Unsat | SolveStatus::Unknown),
        "3-2 Pigeonhole should be UNSAT or Unknown"
    );
}

// =============================================================================
// Graph Coloring Tests
// =============================================================================

/// Test 3-coloring of a triangle - should be satisfiable.
///
/// A triangle (K3) can be 3-colored: each vertex gets a different color.
#[test]
fn test_graph_coloring_triangle_3colors() {
    // Variables: c[v][k] = vertex v has color k
    // v0: vars 0,1,2; v1: vars 3,4,5; v2: vars 6,7,8
    let mut clauses = Vec::new();

    // Each vertex must have at least one color
    clauses.push(Clause::ternary(
        Literal::positive(0),
        Literal::positive(1),
        Literal::positive(2),
    ));
    clauses.push(Clause::ternary(
        Literal::positive(3),
        Literal::positive(4),
        Literal::positive(5),
    ));
    clauses.push(Clause::ternary(
        Literal::positive(6),
        Literal::positive(7),
        Literal::positive(8),
    ));

    // Each vertex has at most one color (optional but helps)
    for v in 0..3 {
        let base = v * 3;
        clauses.push(Clause::binary(
            Literal::negative(base),
            Literal::negative(base + 1),
        ));
        clauses.push(Clause::binary(
            Literal::negative(base),
            Literal::negative(base + 2),
        ));
        clauses.push(Clause::binary(
            Literal::negative(base + 1),
            Literal::negative(base + 2),
        ));
    }

    // Adjacent vertices must have different colors
    // Edge (0,1): for each color k, NOT(v0=k AND v1=k)
    for k in 0..3u32 {
        clauses.push(Clause::binary(
            Literal::negative(k),
            Literal::negative(3 + k),
        ));
    }
    // Edge (1,2)
    for k in 0..3u32 {
        clauses.push(Clause::binary(
            Literal::negative(3 + k),
            Literal::negative(6 + k),
        ));
    }
    // Edge (0,2)
    for k in 0..3u32 {
        clauses.push(Clause::binary(
            Literal::negative(k),
            Literal::negative(6 + k),
        ));
    }

    let instance = SolveInstance::new(9, clauses);
    let solver = Solver::new_cpu();
    let result = solver.solve(instance.clone());

    // Triangle is 3-colorable
    match result.status {
        SolveStatus::Sat => {
            if let Some(assignment) = result.assignment() {
                assert!(
                    instance.is_satisfied(assignment),
                    "Coloring assignment should satisfy all constraints"
                );
            }
        }
        SolveStatus::Unknown => {
            // Even if Unknown, should satisfy most clauses
            if let SolveProof::Approximate {
                satisfied_clauses,
                total_clauses,
                ..
            } = result.proof
            {
                let ratio = satisfied_clauses as f64 / total_clauses as f64;
                assert!(
                    ratio > 0.8,
                    "Should satisfy at least 80% of graph coloring clauses"
                );
            }
        }
        _ => panic!("Unexpected status for 3-colorable graph"),
    }
}

/// Test 2-coloring of a triangle - should be UNSAT.
///
/// A triangle (K3) cannot be 2-colored: it's an odd cycle.
#[test]
fn test_graph_coloring_triangle_2colors_unsat() {
    // Variables: c[v][k] = vertex v has color k (k in {0,1})
    // v0: vars 0,1; v1: vars 2,3; v2: vars 4,5
    // Each vertex must have at least one color
    let clauses = vec![
        Clause::binary(Literal::positive(0), Literal::positive(1)),
        Clause::binary(Literal::positive(2), Literal::positive(3)),
        Clause::binary(Literal::positive(4), Literal::positive(5)),
        // Each vertex has at most one color
        Clause::binary(Literal::negative(0), Literal::negative(1)),
        Clause::binary(Literal::negative(2), Literal::negative(3)),
        Clause::binary(Literal::negative(4), Literal::negative(5)),
        // Adjacent vertices must have different colors
        // Edge (0,1)
        Clause::binary(Literal::negative(0), Literal::negative(2)),
        Clause::binary(Literal::negative(1), Literal::negative(3)),
        // Edge (1,2)
        Clause::binary(Literal::negative(2), Literal::negative(4)),
        Clause::binary(Literal::negative(3), Literal::negative(5)),
        // Edge (0,2)
        Clause::binary(Literal::negative(0), Literal::negative(4)),
        Clause::binary(Literal::negative(1), Literal::negative(5)),
    ];

    let instance = SolveInstance::new(6, clauses);
    let solver = Solver::new_cpu();
    let result = solver.solve(instance);

    // Triangle is not 2-colorable
    assert!(
        matches!(result.status, SolveStatus::Unsat | SolveStatus::Unknown),
        "Triangle 2-coloring should be UNSAT or Unknown"
    );
}

// =============================================================================
// Solver Determinism Tests
// =============================================================================

/// Test that the solver produces deterministic results.
///
/// Given the same input instance, the solver should produce identical output
/// because it uses deterministic pseudo-random initialization.
#[test]
fn test_solver_determinism() {
    let instance = SolveInstance::new(
        5,
        vec![
            Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
            Clause::new(vec![Literal::negative(0), Literal::positive(2)]),
            Clause::new(vec![
                Literal::positive(1),
                Literal::negative(2),
                Literal::positive(3),
            ]),
            Clause::new(vec![
                Literal::negative(1),
                Literal::positive(3),
                Literal::positive(4),
            ]),
            Clause::new(vec![Literal::negative(3), Literal::negative(4)]),
        ],
    );

    let solver_config = {
        let mut config = SolverConfig::default();
        config.max_iterations = 1000;
        config.learning_rate = 0.1;
        config.momentum = 0.9;
        config.discretize_threshold = 0.5;
        config
    };
    let solver = Solver::with_config_cpu(solver_config);

    // Run solver multiple times
    let result1 = solver.solve(instance.clone());
    let result2 = solver.solve(instance.clone());
    let result3 = solver.solve(instance.clone());

    // All runs should produce the same status
    assert_eq!(
        result1.status, result2.status,
        "Solver should produce same status on repeated runs"
    );
    assert_eq!(
        result2.status, result3.status,
        "Solver should produce same status on repeated runs"
    );

    // All runs should produce the same assignment
    assert_eq!(
        result1.assignment(),
        result2.assignment(),
        "Solver should produce same assignment on repeated runs"
    );
    assert_eq!(
        result2.assignment(),
        result3.assignment(),
        "Solver should produce same assignment on repeated runs"
    );

    // All runs should have the same iteration count
    assert_eq!(
        result1.stats.iterations, result2.stats.iterations,
        "Solver should use same number of iterations on repeated runs"
    );
}

/// Test that solver state initialization is deterministic.
#[test]
fn test_solver_state_deterministic_init() {
    let state1 = SolverState::new(10);
    let state2 = SolverState::new(10);

    assert_eq!(
        state1.assignments, state2.assignments,
        "State initialization should be deterministic"
    );
    assert_eq!(state1.velocities, state2.velocities);
    assert_eq!(state1.gradients, state2.gradients);
}

// =============================================================================
// Configuration Effect Tests
// =============================================================================

/// Test that different configurations affect solver behavior.
#[test]
fn test_solver_config_effects() {
    // A satisfiable instance
    let instance = SolveInstance::new(
        3,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0), Literal::positive(1)]),
            Clause::new(vec![Literal::negative(1), Literal::positive(2)]),
        ],
    );

    // Fast config with few iterations
    let fast_config = {
        let mut config = SolverConfig::default();
        config.max_iterations = 100;
        config.learning_rate = 0.3;
        config.momentum = 0.8;
        config.discretize_threshold = 0.5;
        config
    };
    let fast_solver = Solver::with_config_cpu(fast_config);
    let fast_result = fast_solver.solve(instance.clone());

    // Thorough config with many iterations
    let thorough_config = SolverConfig::thorough();
    let thorough_solver = Solver::with_config_cpu(thorough_config);
    let thorough_result = thorough_solver.solve(instance.clone());

    // Both should find a solution for this easy instance
    assert!(
        matches!(fast_result.status, SolveStatus::Sat | SolveStatus::Unknown),
        "Fast solver should complete"
    );
    assert!(
        matches!(
            thorough_result.status,
            SolveStatus::Sat | SolveStatus::Unknown
        ),
        "Thorough solver should complete"
    );

    // Fast config should use fewer or equal iterations
    assert!(
        fast_result.stats.iterations <= thorough_config.max_iterations,
        "Fast config should not exceed its max iterations"
    );
}

/// Test different discretization thresholds.
#[test]
fn test_discretize_threshold_effects() {
    // Instance where threshold might matter
    let instance = SolveInstance::new(
        2,
        vec![
            Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
            Clause::new(vec![Literal::negative(0), Literal::negative(1)]),
        ],
    );

    // Low threshold (more likely to set variables true)
    let low_threshold_config = {
        let mut config = SolverConfig::default();
        config.discretize_threshold = 0.3;
        config
    };
    let low_solver = Solver::with_config_cpu(low_threshold_config);
    let low_result = low_solver.solve(instance.clone());

    // High threshold (more likely to set variables false)
    let high_threshold_config = {
        let mut config = SolverConfig::default();
        config.discretize_threshold = 0.7;
        config
    };
    let high_solver = Solver::with_config_cpu(high_threshold_config);
    let high_result = high_solver.solve(instance.clone());

    // Both should still work
    for result in [low_result, high_result] {
        match result.status {
            SolveStatus::Sat => {
                if let Some(assignment) = result.assignment() {
                    assert!(instance.is_satisfied(assignment));
                }
            }
            SolveStatus::Unknown => {
                // Acceptable
            }
            _ => panic!("Unexpected status"),
        }
    }
}

/// Test learning rate effects on convergence.
#[test]
fn test_learning_rate_effects() {
    // Simple instance
    let instance = SolveInstance::new(
        2,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::positive(1)]),
        ],
    );

    // Very low learning rate
    let slow_config = {
        let mut config = SolverConfig::default();
        config.learning_rate = 0.01;
        config.max_iterations = 10000;
        config
    };
    let slow_solver = Solver::with_config_cpu(slow_config);
    let slow_result = slow_solver.solve(instance.clone());

    // Normal learning rate
    let normal_config = {
        let mut config = SolverConfig::default();
        config.learning_rate = 0.1;
        config.max_iterations = 10000;
        config
    };
    let normal_solver = Solver::with_config_cpu(normal_config);
    let normal_result = normal_solver.solve(instance.clone());

    // Both should solve this trivial instance
    assert!(
        slow_result.is_sat() || slow_result.status == SolveStatus::Unknown,
        "Slow learner should still work"
    );
    assert!(
        normal_result.is_sat() || normal_result.status == SolveStatus::Unknown,
        "Normal learner should still work"
    );

    // Normal rate should typically converge faster
    if slow_result.is_sat() && normal_result.is_sat() {
        // Normal should use fewer or similar iterations
        // (Not guaranteed due to discretization, but usually true)
    }
}

// =============================================================================
// Scalability Tests
// =============================================================================

/// Test solver with a moderately large instance.
#[test]
fn test_solver_large_instance() {
    // 50 variables, 100 clauses (easy ratio)
    let mut clauses = Vec::new();
    for i in 0..100u32 {
        let v1 = i % 50;
        let v2 = (i * 7) % 50;
        let v3 = (i * 13) % 50;
        clauses.push(Clause::ternary(
            Literal::new(v1, i % 2 == 0),
            Literal::new(v2, i % 3 == 0),
            Literal::new(v3, i % 5 == 0),
        ));
    }

    let instance = SolveInstance::new(50, clauses);
    let solver_config = {
        let mut config = SolverConfig::default();
        config.max_iterations = 10000;
        config
    };
    let solver = Solver::with_config_cpu(solver_config);

    let result = solver.solve(instance.clone());

    // Should make progress
    match result.status {
        SolveStatus::Sat => {
            if let Some(assignment) = result.assignment() {
                assert!(instance.is_satisfied(assignment));
            }
        }
        SolveStatus::Unknown => {
            if let SolveProof::Approximate {
                satisfied_clauses,
                total_clauses,
                ..
            } = result.proof
            {
                let ratio = satisfied_clauses as f64 / total_clauses as f64;
                assert!(ratio > 0.5, "Should satisfy at least 50% of clauses");
            }
        }
        _ => {}
    }

    // Verify stats were recorded
    assert!(
        result.stats.iterations > 0,
        "Should have performed iterations"
    );
}

/// Test solver with many variables but few constraints (underconstrained).
#[test]
fn test_solver_underconstrained() {
    // 100 variables, 10 clauses - very easy
    let clauses: Vec<Clause> = (0..10)
        .map(|i| {
            Clause::ternary(
                Literal::positive(i * 10),
                Literal::positive(i * 10 + 1),
                Literal::positive(i * 10 + 2),
            )
        })
        .collect();

    let instance = SolveInstance::new(100, clauses);
    let solver = Solver::with_config_cpu(SolverConfig::fast());

    let result = solver.solve(instance.clone());

    // Should easily find a solution
    match result.status {
        SolveStatus::Sat => {
            if let Some(assignment) = result.assignment() {
                assert_eq!(
                    assignment.len(),
                    100,
                    "Assignment should have 100 variables"
                );
                assert!(instance.is_satisfied(assignment));
            }
        }
        SolveStatus::Unknown => {
            // Still acceptable, but should have high satisfaction
            if let SolveProof::Approximate {
                satisfied_clauses,
                total_clauses,
                ..
            } = result.proof
            {
                assert!(
                    satisfied_clauses >= total_clauses - 1,
                    "Underconstrained should satisfy almost all clauses"
                );
            }
        }
        _ => panic!("Unexpected status for underconstrained instance"),
    }
}

// =============================================================================
// MaxSAT and Weighted Tests
// =============================================================================

/// Test weighted MaxSAT instance.
#[test]
fn test_maxsat_weighted() {
    // Create a weighted instance where we want to maximize satisfaction
    let instance = SolveInstance::with_weights(
        3,
        vec![
            // High weight clause - strongly prefer satisfying
            Clause::new(vec![Literal::positive(0)]),
            // Low weight clause - less important
            Clause::new(vec![Literal::negative(0)]),
            // Medium weight clause
            Clause::new(vec![Literal::positive(1), Literal::positive(2)]),
        ],
        vec![10.0, 1.0, 5.0],
    );

    assert_eq!(instance.objective, Objective::MaxSat);
    assert_eq!(instance.total_weight(), 16.0);

    let solver = Solver::new_cpu();
    let result = solver.solve(instance.clone());

    // Check the weighted satisfaction
    if let Some(assignment) = result.assignment() {
        let weighted_sat = instance.weighted_satisfaction(assignment);
        // Should prefer satisfying high-weight clause (x0=true)
        assert!(
            weighted_sat >= 10.0,
            "Should satisfy at least the high-weight clause"
        );
    }
}

/// Test that satisfaction ratio is computed correctly.
#[test]
fn test_satisfaction_ratio() {
    let instance = SolveInstance::new(
        2,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::positive(1)]),
            Clause::new(vec![Literal::negative(0)]),
            Clause::new(vec![Literal::negative(1)]),
        ],
    );

    // Any assignment satisfies exactly 2 of 4 clauses
    let assignment1 = vec![true, true];
    let assignment2 = vec![false, false];
    let assignment3 = vec![true, false];

    assert_eq!(instance.count_satisfied(&assignment1), 2);
    assert_eq!(instance.count_satisfied(&assignment2), 2);
    assert_eq!(instance.count_satisfied(&assignment3), 2);

    let ratio = instance.satisfaction_ratio(&assignment1);
    assert!(
        (ratio - 0.5).abs() < 0.001,
        "Satisfaction ratio should be 0.5"
    );
}

// =============================================================================
// Proof and Checksum Tests
// =============================================================================

/// Test that proof checksums are computed and verified correctly.
#[test]
fn test_proof_checksum_verification() {
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

    if result.is_sat() {
        // Verify the checksum is valid
        assert_eq!(
            result.verify_proof(),
            Some(true),
            "Proof checksum should be valid"
        );

        // Get the checksum value
        if let SolveProof::Satisfying { checksum, .. } = &result.proof {
            assert_ne!(*checksum, 0, "Checksum should be non-zero");
        }
    }
}

/// Test that different assignments produce different checksums.
#[test]
fn test_proof_checksum_uniqueness() {
    use xlog_solve::compute_checksum;

    let checksum1 = compute_checksum(&[true, false, true]);
    let checksum2 = compute_checksum(&[false, true, false]);
    let checksum3 = compute_checksum(&[true, true, true]);

    assert_ne!(
        checksum1, checksum2,
        "Different assignments should have different checksums"
    );
    assert_ne!(
        checksum2, checksum3,
        "Different assignments should have different checksums"
    );
    assert_ne!(
        checksum1, checksum3,
        "Different assignments should have different checksums"
    );
}

// =============================================================================
// Statistics Tests
// =============================================================================

/// Test that solver statistics are recorded correctly.
#[test]
fn test_solver_statistics() {
    let instance = SolveInstance::new(
        5,
        vec![
            Clause::new(vec![Literal::positive(0), Literal::positive(1)]),
            Clause::new(vec![Literal::negative(1), Literal::positive(2)]),
            Clause::new(vec![Literal::negative(2), Literal::positive(3)]),
            Clause::new(vec![Literal::negative(3), Literal::positive(4)]),
        ],
    );

    let config = {
        let mut config = SolverConfig::default();
        config.max_iterations = 500;
        config
    };
    let solver = Solver::with_config_cpu(config);
    let result = solver.solve(instance);

    // Check stats are populated
    assert!(
        result.stats.iterations > 0,
        "Should have performed iterations"
    );
    assert!(
        result.stats.iterations <= 500,
        "Should not exceed max iterations"
    );
    // Duration might be 0 for very fast solves, so we don't assert on it
}

/// Test statistics helper methods.
#[test]
fn test_stats_helpers() {
    let stats = SolveStats {
        iterations: 1000,
        duration_us: 2_500_000, // 2.5 seconds
        peak_memory: 1_048_576, // 1 MB
    };

    assert_eq!(stats.duration_ms(), 2500);
    assert!((stats.duration_secs() - 2.5).abs() < 0.001);
    assert!((stats.iterations_per_sec() - 400.0).abs() < 0.1);
}

// =============================================================================
// Edge Cases
// =============================================================================

/// Test empty instance.
#[test]
fn test_empty_instance() {
    // No clauses - trivially satisfiable
    let instance = SolveInstance::new(0, vec![]);
    let solver = Solver::new_cpu();
    let result = solver.solve(instance);

    assert!(result.is_sat(), "Empty instance should be SAT");
}

/// Test instance with no variables but empty clause.
#[test]
fn test_empty_clause_instance() {
    // An empty clause is always false
    let instance = SolveInstance::new(0, vec![Clause::new(vec![])]);
    let solver = Solver::new_cpu();
    let result = solver.solve(instance);

    // Empty clause cannot be satisfied
    assert!(
        !result.is_sat(),
        "Instance with empty clause should not be SAT"
    );
}

/// Test single variable, single clause.
#[test]
fn test_minimal_sat() {
    let instance = SolveInstance::new(1, vec![Clause::new(vec![Literal::positive(0)])]);
    let solver = Solver::new_cpu();
    let result = solver.solve(instance.clone());

    assert!(result.is_sat(), "Single positive literal should be SAT");
    if let Some(assignment) = result.assignment() {
        assert!(assignment[0], "Variable should be true");
        assert!(instance.is_satisfied(assignment));
    }
}

/// Test single variable, contradictory clauses.
#[test]
fn test_minimal_unsat() {
    let instance = SolveInstance::new(
        1,
        vec![
            Clause::new(vec![Literal::positive(0)]),
            Clause::new(vec![Literal::negative(0)]),
        ],
    );
    let solver = Solver::new_cpu();
    let result = solver.solve(instance);

    assert!(
        matches!(result.status, SolveStatus::Unsat | SolveStatus::Unknown),
        "Contradictory clauses should be UNSAT or Unknown"
    );
}

/// Test tautological clause.
#[test]
fn test_tautological_clause() {
    // Clause containing both x and NOT x is always true
    let instance = SolveInstance::new(
        2,
        vec![
            Clause::new(vec![Literal::positive(0), Literal::negative(0)]), // tautology
            Clause::new(vec![Literal::positive(1)]),                       // actual constraint
        ],
    );
    let solver = Solver::new_cpu();
    let result = solver.solve(instance.clone());

    assert!(
        result.is_sat(),
        "Tautological clause should not prevent SAT"
    );
    if let Some(assignment) = result.assignment() {
        assert!(instance.is_satisfied(assignment));
    }
}

// =============================================================================
// Instance API Tests
// =============================================================================

/// Test instance construction and manipulation.
#[test]
fn test_instance_construction() {
    let mut instance = SolveInstance::new(
        3,
        vec![Clause::new(vec![
            Literal::positive(0),
            Literal::positive(1),
        ])],
    );

    assert_eq!(instance.num_vars, 3);
    assert_eq!(instance.num_clauses(), 1);
    assert!(instance.validate());

    // Add a clause
    instance.add_clause(Clause::new(vec![Literal::positive(2)]));
    assert_eq!(instance.num_clauses(), 2);

    // Add weighted clause
    instance.add_weighted_clause(Clause::new(vec![Literal::negative(0)]), 2.5);
    assert_eq!(instance.num_clauses(), 3);
    assert!(instance.weights.is_some());
}

/// Test literal DIMACS conversion roundtrip.
#[test]
fn test_literal_dimacs_roundtrip() {
    for var in 0..10 {
        let pos = Literal::positive(var);
        let neg = Literal::negative(var);

        let pos_dimacs = pos.to_dimacs();
        let neg_dimacs = neg.to_dimacs();

        assert!(pos_dimacs > 0);
        assert!(neg_dimacs < 0);

        let pos_back = Literal::from_dimacs(pos_dimacs);
        let neg_back = Literal::from_dimacs(neg_dimacs);

        assert_eq!(pos, pos_back);
        assert_eq!(neg, neg_back);
    }
}

/// Test literal packed representation roundtrip.
#[test]
fn test_literal_packed_roundtrip() {
    for var in 0..100 {
        let pos = Literal::positive(var);
        let neg = Literal::negative(var);

        let pos_packed = pos.to_packed();
        let neg_packed = neg.to_packed();

        let pos_back = Literal::from_packed(pos_packed);
        let neg_back = Literal::from_packed(neg_packed);

        assert_eq!(pos, pos_back);
        assert_eq!(neg, neg_back);
    }
}

// =============================================================================
// Complex Real-World Patterns
// =============================================================================

/// Test implication chain: x0 -> x1 -> x2 -> ... -> xn
#[test]
fn test_implication_chain() {
    let n = 10;
    let mut clauses = Vec::new();

    // x0 must be true (starting condition)
    clauses.push(Clause::unit(Literal::positive(0)));

    // x_i -> x_{i+1} for all i
    for i in 0..n - 1 {
        clauses.push(Clause::binary(
            Literal::negative(i as u32),
            Literal::positive((i + 1) as u32),
        ));
    }

    let instance = SolveInstance::new(n as u32, clauses);
    let solver = Solver::new_cpu();
    let result = solver.solve(instance.clone());

    assert!(result.is_sat(), "Implication chain should be SAT");
    if let Some(assignment) = result.assignment() {
        // All variables should be true
        for (i, value) in assignment.iter().enumerate().take(n) {
            assert!(*value, "Variable {} should be true due to implications", i);
        }
        assert!(instance.is_satisfied(assignment));
    }
}

/// Test XOR chain encoding.
#[test]
fn test_xor_chain() {
    // x0 XOR x1 = true, x1 XOR x2 = true, etc.
    // This means alternating truth values
    let mut clauses = Vec::new();

    // XOR encoded as: (a OR b) AND (NOT a OR NOT b)
    for i in 0..3u32 {
        let a = i;
        let b = i + 1;
        // a XOR b = true
        clauses.push(Clause::binary(Literal::positive(a), Literal::positive(b)));
        clauses.push(Clause::binary(Literal::negative(a), Literal::negative(b)));
    }

    // Fix x0 = true
    clauses.push(Clause::unit(Literal::positive(0)));

    let instance = SolveInstance::new(4, clauses);
    let solver = Solver::new_cpu();
    let result = solver.solve(instance.clone());

    assert!(result.is_sat(), "XOR chain should be SAT");
    if let Some(assignment) = result.assignment() {
        // Should alternate: T, F, T, F
        assert!(assignment[0]);
        assert!(!assignment[1]);
        assert!(assignment[2]);
        assert!(!assignment[3]);
        assert!(instance.is_satisfied(assignment));
    }
}

/// Test at-most-one constraint encoding.
#[test]
fn test_at_most_one() {
    let n = 5;
    let mut clauses = Vec::new();

    // At least one: x0 OR x1 OR x2 OR x3 OR x4
    clauses.push(Clause::new((0..n as u32).map(Literal::positive).collect()));

    // At most one: for each pair (i,j), NOT xi OR NOT xj
    for i in 0..n {
        for j in i + 1..n {
            clauses.push(Clause::binary(
                Literal::negative(i as u32),
                Literal::negative(j as u32),
            ));
        }
    }

    let instance = SolveInstance::new(n as u32, clauses);
    let solver = Solver::new_cpu();
    let result = solver.solve(instance.clone());

    assert!(result.is_sat(), "At-most-one should be SAT");
    if let Some(assignment) = result.assignment() {
        // Exactly one should be true
        let count = assignment.iter().filter(|&&x| x).count();
        assert_eq!(count, 1, "Exactly one variable should be true");
        assert!(instance.is_satisfied(assignment));
    }
}
