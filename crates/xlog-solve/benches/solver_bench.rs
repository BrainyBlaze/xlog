//! Performance benchmarks for the xlog-solve crate.
//!
//! Run with: `cargo bench -p xlog-solve`
//!
//! These benchmarks measure the performance of:
//! - SAT solving for various instance sizes
//! - Gradient computation
//! - State updates
//! - Both easy and hard instances

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use xlog_solve::{Clause, Literal, SolveInstance, Solver, SolverConfig, SolverState};

// =============================================================================
// Instance Generators
// =============================================================================

/// Generates a satisfiable random 3-SAT instance.
///
/// Creates clauses with exactly 3 literals each, with random variable
/// assignments that guarantee satisfiability.
fn generate_random_3sat(num_vars: u32, num_clauses: u32, seed: u64) -> SolveInstance {
    // Simple LCG for deterministic pseudo-random numbers
    let mut rng_state = seed;
    let mut next_rand = || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        rng_state
    };

    let mut clauses = Vec::with_capacity(num_clauses as usize);

    for _ in 0..num_clauses {
        let mut literals = Vec::with_capacity(3);
        for _ in 0..3 {
            let var = (next_rand() % num_vars as u64) as u32;
            let negated = next_rand() % 2 == 0;
            literals.push(Literal::new(var, negated));
        }
        clauses.push(Clause::new(literals));
    }

    SolveInstance::new(num_vars, clauses)
}

/// Generates an "easy" satisfiable instance with mostly unit clauses.
fn generate_easy_instance(num_vars: u32) -> SolveInstance {
    let mut clauses = Vec::with_capacity(num_vars as usize);

    // Create unit clauses that are easy to satisfy
    for i in 0..num_vars {
        clauses.push(Clause::unit(Literal::positive(i)));
    }

    SolveInstance::new(num_vars, clauses)
}

/// Generates a "hard" unsatisfiable instance (pigeon-hole style).
///
/// Creates constraints that force more values than possible into limited slots.
fn generate_hard_instance(num_vars: u32) -> SolveInstance {
    let mut clauses = Vec::new();

    // Create constraints that are difficult to satisfy
    // Each variable must be true AND false at the same time (in different clauses)
    // This creates contradictions that CLS will struggle with
    for i in 0..num_vars {
        // Variable i must be true OR the next variable must be true
        clauses.push(Clause::new(vec![
            Literal::positive(i),
            Literal::positive((i + 1) % num_vars),
        ]));
        // Variable i must be false OR the next variable must be false
        clauses.push(Clause::new(vec![
            Literal::negative(i),
            Literal::negative((i + 1) % num_vars),
        ]));
        // At least one of three consecutive variables must be true
        clauses.push(Clause::new(vec![
            Literal::positive(i),
            Literal::positive((i + 1) % num_vars),
            Literal::positive((i + 2) % num_vars),
        ]));
    }

    SolveInstance::new(num_vars, clauses)
}

/// Generates an implication chain: x0 -> x1 -> x2 -> ... -> x_{n-1}
fn generate_implication_chain(num_vars: u32) -> SolveInstance {
    let mut clauses = Vec::with_capacity(num_vars as usize);

    // Force x0 to be true
    clauses.push(Clause::unit(Literal::positive(0)));

    // Each variable implies the next
    for i in 0..num_vars.saturating_sub(1) {
        clauses.push(Clause::binary(
            Literal::negative(i),
            Literal::positive(i + 1),
        ));
    }

    SolveInstance::new(num_vars, clauses)
}

// =============================================================================
// Solver Benchmarks
// =============================================================================

fn bench_solve_easy(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_easy");

    for size in [10, 50, 100, 500, 1000].iter() {
        let instance = generate_easy_instance(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("vars", size), size, |b, _| {
            let solver = Solver::with_config_cpu(SolverConfig::fast());
            b.iter(|| solver.solve(black_box(instance.clone())));
        });
    }

    group.finish();
}

fn bench_solve_implication_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_implication_chain");

    for size in [10, 50, 100, 500].iter() {
        let instance = generate_implication_chain(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("vars", size), size, |b, _| {
            let solver = Solver::with_config_cpu(SolverConfig::fast());
            b.iter(|| solver.solve(black_box(instance.clone())));
        });
    }

    group.finish();
}

fn bench_solve_random_3sat(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_random_3sat");

    // Test at phase transition: ~4.26 clauses per variable
    for num_vars in [20, 50, 100, 200].iter() {
        let num_clauses = (*num_vars as f64 * 4.26) as u32;
        let instance = generate_random_3sat(*num_vars, num_clauses, 42);

        group.throughput(Throughput::Elements(*num_vars as u64));
        group.bench_with_input(
            BenchmarkId::new("vars", num_vars),
            num_vars,
            |b, _| {
                let solver = Solver::with_config_cpu(
                    SolverConfig::default().with_max_iterations(1000),
                );
                b.iter(|| solver.solve(black_box(instance.clone())));
            },
        );
    }

    group.finish();
}

fn bench_solve_hard(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_hard");

    // Hard instances with limited iterations
    for size in [10, 20, 50].iter() {
        let instance = generate_hard_instance(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("vars", size), size, |b, _| {
            let solver = Solver::with_config_cpu(
                SolverConfig::default().with_max_iterations(500),
            );
            b.iter(|| solver.solve(black_box(instance.clone())));
        });
    }

    group.finish();
}

// =============================================================================
// Gradient Computation Benchmarks
// =============================================================================

/// Internal helper to expose gradient computation for benchmarking.
///
/// This reimplements the gradient computation from the solver since the
/// method is private. In a real scenario, we might expose a benchmark-only
/// API or use inline benchmarking.
fn compute_gradients_benchmark(instance: &SolveInstance, state: &mut SolverState) {
    state.gradients.fill(0.0);

    for clause in &instance.clauses {
        // Compute clause unsatisfaction
        let mut clause_unsat = 1.0f32;
        for lit in &clause.literals {
            let val = state.assignments[lit.var as usize];
            let lit_val = if lit.negated { 1.0 - val } else { val };
            clause_unsat *= 1.0 - lit_val;
        }

        if clause_unsat < 0.001 {
            continue;
        }

        for lit in &clause.literals {
            let var = lit.var as usize;
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
            let sign = if lit.negated { 1.0 } else { -1.0 };
            state.gradients[var] += sign * other_product;
        }
    }
}

fn bench_gradient_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_computation");

    for num_vars in [50, 100, 500, 1000].iter() {
        let num_clauses = (*num_vars as f64 * 4.0) as u32;
        let instance = generate_random_3sat(*num_vars, num_clauses, 123);
        let mut state = SolverState::new(*num_vars);

        group.throughput(Throughput::Elements(num_clauses as u64));
        group.bench_with_input(
            BenchmarkId::new("clauses", num_clauses),
            &num_clauses,
            |b, _| {
                b.iter(|| {
                    compute_gradients_benchmark(black_box(&instance), black_box(&mut state));
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// State Update Benchmarks
// =============================================================================

fn bench_state_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_creation");

    for size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("vars", size), size, |b, &size| {
            b.iter(|| SolverState::new(black_box(size)));
        });
    }

    group.finish();
}

fn bench_state_discretize(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_discretize");

    for size in [100, 1000, 10000, 100000].iter() {
        let state = SolverState::new(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("vars", size), size, |b, _| {
            b.iter(|| state.discretize(black_box(0.5)));
        });
    }

    group.finish();
}

fn bench_state_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_operations");

    for size in [1000, 10000].iter() {
        let size_u32 = *size as u32;

        group.bench_with_input(
            BenchmarkId::new("reset_velocities", size),
            size,
            |b, _| {
                let mut state = SolverState::new(size_u32);
                state.velocities.fill(1.0);
                b.iter(|| {
                    black_box(&mut state).reset_velocities();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("clear_gradients", size),
            size,
            |b, _| {
                let mut state = SolverState::new(size_u32);
                state.gradients.fill(1.0);
                b.iter(|| {
                    black_box(&mut state).clear_gradients();
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// Instance Benchmarks
// =============================================================================

fn bench_instance_satisfaction_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("instance_satisfaction");

    for num_vars in [100, 500, 1000].iter() {
        let num_clauses = (*num_vars as f64 * 4.0) as u32;
        let instance = generate_random_3sat(*num_vars, num_clauses, 999);
        let assignment: Vec<bool> = (0..*num_vars).map(|i| i % 2 == 0).collect();

        group.throughput(Throughput::Elements(num_clauses as u64));
        group.bench_with_input(
            BenchmarkId::new("is_satisfied", num_clauses),
            &num_clauses,
            |b, _| {
                b.iter(|| instance.is_satisfied(black_box(&assignment)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("count_satisfied", num_clauses),
            &num_clauses,
            |b, _| {
                b.iter(|| instance.count_satisfied(black_box(&assignment)));
            },
        );
    }

    group.finish();
}

fn bench_clause_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("clause_evaluation");

    for clause_size in [2, 3, 5, 10].iter() {
        let literals: Vec<Literal> = (0..*clause_size as u32)
            .map(|i| {
                if i % 2 == 0 {
                    Literal::positive(i)
                } else {
                    Literal::negative(i)
                }
            })
            .collect();
        let clause = Clause::new(literals);
        let assignment: Vec<bool> = (0..*clause_size).map(|i| i % 3 == 0).collect();

        group.bench_with_input(
            BenchmarkId::new("is_satisfied", clause_size),
            clause_size,
            |b, _| {
                b.iter(|| clause.is_satisfied(black_box(&assignment)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("count_satisfied", clause_size),
            clause_size,
            |b, _| {
                b.iter(|| clause.count_satisfied(black_box(&assignment)));
            },
        );
    }

    group.finish();
}

// =============================================================================
// Literal Benchmarks
// =============================================================================

fn bench_literal_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("literal_operations");

    // Batch operations
    let num_literals = 10000;
    let literals: Vec<Literal> = (0..num_literals as u32)
        .map(|i| {
            if i % 2 == 0 {
                Literal::positive(i)
            } else {
                Literal::negative(i)
            }
        })
        .collect();
    let assignment: Vec<bool> = (0..num_literals).map(|i| i % 3 == 0).collect();

    group.throughput(Throughput::Elements(num_literals as u64));

    group.bench_function("eval_batch", |b| {
        b.iter(|| {
            literals
                .iter()
                .map(|lit| lit.eval(black_box(&assignment)))
                .count()
        });
    });

    group.bench_function("to_packed_batch", |b| {
        b.iter(|| literals.iter().map(|lit| lit.to_packed()).sum::<u32>());
    });

    group.bench_function("from_packed_batch", |b| {
        let packed: Vec<u32> = literals.iter().map(|lit| lit.to_packed()).collect();
        b.iter(|| {
            packed
                .iter()
                .map(|&p| Literal::from_packed(black_box(p)))
                .count()
        });
    });

    group.bench_function("to_dimacs_batch", |b| {
        b.iter(|| literals.iter().map(|lit| lit.to_dimacs()).sum::<i32>());
    });

    group.finish();
}

// =============================================================================
// Criterion Groups
// =============================================================================

criterion_group!(
    solver_benches,
    bench_solve_easy,
    bench_solve_implication_chain,
    bench_solve_random_3sat,
    bench_solve_hard,
);

criterion_group!(
    gradient_benches,
    bench_gradient_computation,
);

criterion_group!(
    state_benches,
    bench_state_creation,
    bench_state_discretize,
    bench_state_operations,
);

criterion_group!(
    instance_benches,
    bench_instance_satisfaction_check,
    bench_clause_evaluation,
    bench_literal_operations,
);

criterion_main!(solver_benches, gradient_benches, state_benches, instance_benches);
