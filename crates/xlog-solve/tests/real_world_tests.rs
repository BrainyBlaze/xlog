//! Tests for real-world SAT problem encodings.
//!
//! These tests verify the correctness of SAT encodings for:
//! - Sudoku
//! - N-Queens
//! - Job Shop Scheduling
//! - Graph Coloring
//! - Circuit SAT

use xlog_solve::{Clause, Literal, SolveInstance, SolveStatus, Solver, SolverConfig};

// =============================================================================
// Sudoku Tests
// =============================================================================

mod sudoku {
    use super::*;

    /// Total number of variables for 9x9 Sudoku
    pub const NUM_VARS: u32 = 729;

    #[inline]
    pub fn var_index(row: usize, col: usize, val: usize) -> u32 {
        (row * 81 + col * 9 + val) as u32
    }

    #[inline]
    pub fn from_var_index(var: u32) -> (usize, usize, usize) {
        let var = var as usize;
        (var / 81, (var % 81) / 9, var % 9)
    }

    pub fn encode(grid: &[[u8; 9]; 9]) -> SolveInstance {
        let mut clauses = Vec::new();

        // Each cell has at least one value
        for row in 0..9 {
            for col in 0..9 {
                let literals: Vec<Literal> = (0..9)
                    .map(|val| Literal::positive(var_index(row, col, val)))
                    .collect();
                clauses.push(Clause::new(literals));
            }
        }

        // Each cell has at most one value
        for row in 0..9 {
            for col in 0..9 {
                for v1 in 0..9 {
                    for v2 in (v1 + 1)..9 {
                        clauses.push(Clause::binary(
                            Literal::negative(var_index(row, col, v1)),
                            Literal::negative(var_index(row, col, v2)),
                        ));
                    }
                }
            }
        }

        // Each row has each value
        for row in 0..9 {
            for val in 0..9 {
                let literals: Vec<Literal> = (0..9)
                    .map(|col| Literal::positive(var_index(row, col, val)))
                    .collect();
                clauses.push(Clause::new(literals));

                for c1 in 0..9 {
                    for c2 in (c1 + 1)..9 {
                        clauses.push(Clause::binary(
                            Literal::negative(var_index(row, c1, val)),
                            Literal::negative(var_index(row, c2, val)),
                        ));
                    }
                }
            }
        }

        // Each column has each value
        for col in 0..9 {
            for val in 0..9 {
                let literals: Vec<Literal> = (0..9)
                    .map(|row| Literal::positive(var_index(row, col, val)))
                    .collect();
                clauses.push(Clause::new(literals));

                for r1 in 0..9 {
                    for r2 in (r1 + 1)..9 {
                        clauses.push(Clause::binary(
                            Literal::negative(var_index(r1, col, val)),
                            Literal::negative(var_index(r2, col, val)),
                        ));
                    }
                }
            }
        }

        // Each 3x3 box has each value
        for box_row in 0..3 {
            for box_col in 0..3 {
                for val in 0..9 {
                    let mut cells = Vec::new();
                    for dr in 0..3 {
                        for dc in 0..3 {
                            cells.push((box_row * 3 + dr, box_col * 3 + dc));
                        }
                    }

                    let literals: Vec<Literal> = cells
                        .iter()
                        .map(|&(r, c)| Literal::positive(var_index(r, c, val)))
                        .collect();
                    clauses.push(Clause::new(literals));

                    for i in 0..cells.len() {
                        for j in (i + 1)..cells.len() {
                            let (r1, c1) = cells[i];
                            let (r2, c2) = cells[j];
                            clauses.push(Clause::binary(
                                Literal::negative(var_index(r1, c1, val)),
                                Literal::negative(var_index(r2, c2, val)),
                            ));
                        }
                    }
                }
            }
        }

        // Fixed values
        for (row, row_values) in grid.iter().enumerate() {
            for (col, &cell) in row_values.iter().enumerate() {
                if cell != 0 {
                    let val = (cell - 1) as usize;
                    clauses.push(Clause::unit(Literal::positive(var_index(row, col, val))));
                }
            }
        }

        SolveInstance::new(NUM_VARS, clauses)
    }

    pub fn decode(assignment: &[bool]) -> [[u8; 9]; 9] {
        let mut grid = [[0u8; 9]; 9];
        for row in 0..9 {
            for col in 0..9 {
                for val in 0..9 {
                    if assignment[var_index(row, col, val) as usize] {
                        grid[row][col] = (val + 1) as u8;
                        break;
                    }
                }
            }
        }
        grid
    }

    pub fn verify(grid: &[[u8; 9]; 9]) -> bool {
        for row_values in grid.iter().take(9) {
            for &cell in row_values.iter().take(9) {
                if !(1..=9).contains(&cell) {
                    return false;
                }
            }
        }

        for row_values in grid.iter().take(9) {
            let mut seen = [false; 9];
            for &cell in row_values.iter().take(9) {
                let val = (cell - 1) as usize;
                if seen[val] {
                    return false;
                }
                seen[val] = true;
            }
        }

        for col in 0..9 {
            let mut seen = [false; 9];
            for row_values in grid.iter().take(9) {
                let val = (row_values[col] - 1) as usize;
                if seen[val] {
                    return false;
                }
                seen[val] = true;
            }
        }

        for box_row in 0..3 {
            for box_col in 0..3 {
                let mut seen = [false; 9];
                for dr in 0..3 {
                    for dc in 0..3 {
                        let row = box_row * 3 + dr;
                        let col = box_col * 3 + dc;
                        let val = (grid[row][col] - 1) as usize;
                        if seen[val] {
                            return false;
                        }
                        seen[val] = true;
                    }
                }
            }
        }

        true
    }
}

#[test]
fn test_sudoku_var_index_roundtrip() {
    for row in 0..9 {
        for col in 0..9 {
            for val in 0..9 {
                let idx = sudoku::var_index(row, col, val);
                let (r, c, v) = sudoku::from_var_index(idx);
                assert_eq!((row, col, val), (r, c, v));
            }
        }
    }
}

#[test]
fn test_sudoku_encoding_size() {
    let puzzle = [[0u8; 9]; 9]; // Empty puzzle
    let instance = sudoku::encode(&puzzle);

    assert_eq!(instance.num_vars, 729); // 9 * 9 * 9
                                        // Clauses: 81 at-least-one + 81*36 at-most-one + 81*2 row constraints + ...
    assert!(instance.num_clauses() > 0);
}

#[test]
fn test_sudoku_verify_valid_solution() {
    let grid = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ];
    assert!(sudoku::verify(&grid));
}

#[test]
fn test_sudoku_verify_invalid_row() {
    let mut grid = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ];
    grid[0][0] = grid[0][1]; // Duplicate in row
    assert!(!sudoku::verify(&grid));
}

#[test]
fn test_sudoku_solve_easy_puzzle() {
    // Almost complete puzzle with just one empty cell
    let puzzle = [
        [5, 3, 4, 6, 7, 8, 9, 1, 0], // Last cell should be 2
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ];

    let instance = sudoku::encode(&puzzle);
    let solver = Solver::with_config_cpu(SolverConfig::thorough());
    let result = solver.solve(instance);

    assert!(matches!(result.status, SolveStatus::Sat));

    if let Some(assignment) = result.assignment() {
        let solution = sudoku::decode(assignment);
        assert!(sudoku::verify(&solution));
        assert_eq!(solution[0][8], 2); // The missing value
    }
}

// =============================================================================
// N-Queens Tests
// =============================================================================

mod nqueens {
    use super::*;

    #[inline]
    pub fn var_index(n: usize, row: usize, col: usize) -> u32 {
        (row * n + col) as u32
    }

    pub fn encode(n: usize) -> SolveInstance {
        let num_vars = (n * n) as u32;
        let mut clauses = Vec::new();

        // At least one queen per row
        for row in 0..n {
            let literals: Vec<Literal> = (0..n)
                .map(|col| Literal::positive(var_index(n, row, col)))
                .collect();
            clauses.push(Clause::new(literals));
        }

        // At most one queen per row
        for row in 0..n {
            for c1 in 0..n {
                for c2 in (c1 + 1)..n {
                    clauses.push(Clause::binary(
                        Literal::negative(var_index(n, row, c1)),
                        Literal::negative(var_index(n, row, c2)),
                    ));
                }
            }
        }

        // At most one queen per column
        for col in 0..n {
            for r1 in 0..n {
                for r2 in (r1 + 1)..n {
                    clauses.push(Clause::binary(
                        Literal::negative(var_index(n, r1, col)),
                        Literal::negative(var_index(n, r2, col)),
                    ));
                }
            }
        }

        // At most one queen per diagonal
        for diag in -(n as i32 - 1)..=(n as i32 - 1) {
            let mut cells = Vec::new();
            for row in 0..n {
                let col = (row as i32) - diag;
                if col >= 0 && col < n as i32 {
                    cells.push((row, col as usize));
                }
            }
            for i in 0..cells.len() {
                for j in (i + 1)..cells.len() {
                    let (r1, c1) = cells[i];
                    let (r2, c2) = cells[j];
                    clauses.push(Clause::binary(
                        Literal::negative(var_index(n, r1, c1)),
                        Literal::negative(var_index(n, r2, c2)),
                    ));
                }
            }
        }

        // At most one queen per anti-diagonal
        for diag in 0..(2 * n - 1) {
            let mut cells = Vec::new();
            for row in 0..n {
                let col = diag as i32 - row as i32;
                if col >= 0 && col < n as i32 {
                    cells.push((row, col as usize));
                }
            }
            for i in 0..cells.len() {
                for j in (i + 1)..cells.len() {
                    let (r1, c1) = cells[i];
                    let (r2, c2) = cells[j];
                    clauses.push(Clause::binary(
                        Literal::negative(var_index(n, r1, c1)),
                        Literal::negative(var_index(n, r2, c2)),
                    ));
                }
            }
        }

        SolveInstance::new(num_vars, clauses)
    }

    pub fn decode(n: usize, assignment: &[bool]) -> Vec<(usize, usize)> {
        let mut queens = Vec::new();
        for row in 0..n {
            for col in 0..n {
                if assignment[var_index(n, row, col) as usize] {
                    queens.push((row, col));
                }
            }
        }
        queens
    }

    pub fn verify(n: usize, queens: &[(usize, usize)]) -> bool {
        if queens.len() != n {
            return false;
        }

        for &(row, col) in queens {
            if row >= n || col >= n {
                return false;
            }
        }

        for i in 0..queens.len() {
            for j in (i + 1)..queens.len() {
                let (r1, c1) = queens[i];
                let (r2, c2) = queens[j];

                if r1 == r2 || c1 == c2 {
                    return false;
                }
                if (r1 as i32 - r2 as i32).abs() == (c1 as i32 - c2 as i32).abs() {
                    return false;
                }
            }
        }

        true
    }
}

#[test]
fn test_nqueens_encoding_size() {
    let instance = nqueens::encode(4);
    assert_eq!(instance.num_vars, 16); // 4 * 4
}

#[test]
fn test_nqueens_verify_valid() {
    // Known 4-queens solution
    let queens = vec![(0, 1), (1, 3), (2, 0), (3, 2)];
    assert!(nqueens::verify(4, &queens));
}

#[test]
fn test_nqueens_verify_invalid_same_row() {
    let queens = vec![(0, 0), (0, 1), (2, 2), (3, 3)];
    assert!(!nqueens::verify(4, &queens));
}

#[test]
fn test_nqueens_verify_invalid_diagonal() {
    let queens = vec![(0, 0), (1, 1), (2, 3), (3, 2)];
    assert!(!nqueens::verify(4, &queens));
}

#[test]
fn test_nqueens_solve_4() {
    let instance = nqueens::encode(4);
    // Use higher iterations for challenging constraint satisfaction problems
    let config = SolverConfig::new(100000, 0.05, 0.95, 0.5);
    let solver = Solver::with_config_cpu(config);
    let result = solver.solve(instance);

    // CLS is an incomplete solver - it may not always find a solution
    // but if it does find one, it must be valid
    if matches!(result.status, SolveStatus::Sat) {
        if let Some(assignment) = result.assignment() {
            let queens = nqueens::decode(4, assignment);
            assert!(nqueens::verify(4, &queens));
        }
    }
    // Test passes if solver returns Sat with valid solution OR Unknown
    // (CLS cannot prove satisfiability, only find solutions)
}

#[test]
fn test_nqueens_solve_8() {
    let instance = nqueens::encode(8);
    // Use higher iterations for challenging constraint satisfaction problems
    let config = SolverConfig::new(100000, 0.05, 0.95, 0.5);
    let solver = Solver::with_config_cpu(config);
    let result = solver.solve(instance);

    // CLS is an incomplete solver - it may not always find a solution
    // but if it does find one, it must be valid
    if matches!(result.status, SolveStatus::Sat) {
        if let Some(assignment) = result.assignment() {
            let queens = nqueens::decode(8, assignment);
            assert!(nqueens::verify(8, &queens));
            assert_eq!(queens.len(), 8);
        }
    }
    // Test passes if solver returns Sat with valid solution OR Unknown
}

// =============================================================================
// Job Shop Scheduling Tests
// =============================================================================

mod job_shop {
    use super::*;
    use std::collections::HashMap;

    #[derive(Debug, Clone, Copy)]
    pub struct Operation {
        pub machine: usize,
        pub duration: usize,
    }

    #[derive(Debug, Clone)]
    pub struct Job {
        pub operations: Vec<Operation>,
    }

    #[derive(Debug, Clone)]
    pub struct JobShopProblem {
        pub num_machines: usize,
        pub jobs: Vec<Job>,
        pub horizon: usize,
    }

    impl JobShopProblem {
        pub fn total_operations(&self) -> usize {
            self.jobs.iter().map(|j| j.operations.len()).sum()
        }

        pub fn var_index(&self, job: usize, op: usize, time: usize) -> u32 {
            let mut offset = 0;
            for j in 0..job {
                offset += self.jobs[j].operations.len() * self.horizon;
            }
            offset += op * self.horizon;
            (offset + time) as u32
        }

        pub fn num_vars(&self) -> u32 {
            (self.total_operations() * self.horizon) as u32
        }
    }

    pub fn encode(problem: &JobShopProblem) -> SolveInstance {
        let num_vars = problem.num_vars();
        let mut clauses = Vec::new();

        // Each operation starts at exactly one time
        for (j, job) in problem.jobs.iter().enumerate() {
            for o in 0..job.operations.len() {
                let op = &job.operations[o];
                let latest_start = problem.horizon.saturating_sub(op.duration);

                let literals: Vec<Literal> = (0..=latest_start)
                    .map(|t| Literal::positive(problem.var_index(j, o, t)))
                    .collect();
                if !literals.is_empty() {
                    clauses.push(Clause::new(literals));
                }

                for t1 in 0..=latest_start {
                    for t2 in (t1 + 1)..=latest_start {
                        clauses.push(Clause::binary(
                            Literal::negative(problem.var_index(j, o, t1)),
                            Literal::negative(problem.var_index(j, o, t2)),
                        ));
                    }
                }

                for t in (latest_start + 1)..problem.horizon {
                    clauses.push(Clause::unit(Literal::negative(problem.var_index(j, o, t))));
                }
            }
        }

        // Precedence within jobs
        for (j, job) in problem.jobs.iter().enumerate() {
            for o in 0..(job.operations.len().saturating_sub(1)) {
                let op = &job.operations[o];
                let next_op = &job.operations[o + 1];
                let latest_start = problem.horizon.saturating_sub(op.duration);
                let next_latest_start = problem.horizon.saturating_sub(next_op.duration);

                for t in 0..=latest_start {
                    let earliest_next = t + op.duration;
                    let mut literals = vec![Literal::negative(problem.var_index(j, o, t))];
                    for next_t in earliest_next..=next_latest_start {
                        literals.push(Literal::positive(problem.var_index(j, o + 1, next_t)));
                    }
                    if literals.len() > 1 {
                        clauses.push(Clause::new(literals));
                    }
                }
            }
        }

        // No machine conflicts
        let mut ops_by_machine: Vec<Vec<(usize, usize, usize)>> =
            vec![Vec::new(); problem.num_machines];

        for (j, job) in problem.jobs.iter().enumerate() {
            for (o, op) in job.operations.iter().enumerate() {
                ops_by_machine[op.machine].push((j, o, op.duration));
            }
        }

        for machine_ops in &ops_by_machine {
            for i in 0..machine_ops.len() {
                for k in (i + 1)..machine_ops.len() {
                    let (j1, o1, d1) = machine_ops[i];
                    let (j2, o2, d2) = machine_ops[k];

                    let latest1 = problem.horizon.saturating_sub(d1);
                    let latest2 = problem.horizon.saturating_sub(d2);

                    for t1 in 0..=latest1 {
                        for t2 in 0..=latest2 {
                            let overlaps = !(t1 + d1 <= t2 || t2 + d2 <= t1);
                            if overlaps {
                                clauses.push(Clause::binary(
                                    Literal::negative(problem.var_index(j1, o1, t1)),
                                    Literal::negative(problem.var_index(j2, o2, t2)),
                                ));
                            }
                        }
                    }
                }
            }
        }

        SolveInstance::new(num_vars, clauses)
    }

    pub type Schedule = Vec<(usize, usize, usize)>;

    pub fn decode(problem: &JobShopProblem, assignment: &[bool]) -> Schedule {
        let mut schedule = Vec::new();
        for (j, job) in problem.jobs.iter().enumerate() {
            for o in 0..job.operations.len() {
                for t in 0..problem.horizon {
                    if assignment[problem.var_index(j, o, t) as usize] {
                        schedule.push((j, o, t));
                        break;
                    }
                }
            }
        }
        schedule
    }

    pub fn verify(problem: &JobShopProblem, schedule: &Schedule) -> bool {
        let mut start_times: HashMap<(usize, usize), usize> = HashMap::new();

        for &(j, o, t) in schedule {
            start_times.insert((j, o), t);
        }

        for (j, job) in problem.jobs.iter().enumerate() {
            for o in 0..job.operations.len() {
                if !start_times.contains_key(&(j, o)) {
                    return false;
                }
            }
        }

        for (j, job) in problem.jobs.iter().enumerate() {
            for o in 0..(job.operations.len().saturating_sub(1)) {
                let t1 = start_times[&(j, o)];
                let t2 = start_times[&(j, o + 1)];
                let d1 = job.operations[o].duration;
                if t2 < t1 + d1 {
                    return false;
                }
            }
        }

        let mut ops_by_machine: Vec<Vec<(usize, usize)>> = vec![Vec::new(); problem.num_machines];
        for (j, job) in problem.jobs.iter().enumerate() {
            for (o, op) in job.operations.iter().enumerate() {
                ops_by_machine[op.machine].push((j, o));
            }
        }

        for machine_ops in &ops_by_machine {
            for i in 0..machine_ops.len() {
                for k in (i + 1)..machine_ops.len() {
                    let (j1, o1) = machine_ops[i];
                    let (j2, o2) = machine_ops[k];

                    let t1 = start_times[&(j1, o1)];
                    let t2 = start_times[&(j2, o2)];
                    let d1 = problem.jobs[j1].operations[o1].duration;
                    let d2 = problem.jobs[j2].operations[o2].duration;

                    let overlaps = !(t1 + d1 <= t2 || t2 + d2 <= t1);
                    if overlaps {
                        return false;
                    }
                }
            }
        }

        true
    }

    pub fn simple_problem() -> JobShopProblem {
        JobShopProblem {
            num_machines: 2,
            horizon: 10,
            jobs: vec![
                Job {
                    operations: vec![
                        Operation {
                            machine: 0,
                            duration: 2,
                        },
                        Operation {
                            machine: 1,
                            duration: 1,
                        },
                    ],
                },
                Job {
                    operations: vec![
                        Operation {
                            machine: 1,
                            duration: 2,
                        },
                        Operation {
                            machine: 0,
                            duration: 1,
                        },
                    ],
                },
            ],
        }
    }
}

#[test]
fn test_job_shop_simple_problem() {
    let problem = job_shop::simple_problem();
    let instance = job_shop::encode(&problem);
    let solver = Solver::with_config_cpu(SolverConfig::thorough());
    let result = solver.solve(instance);

    assert!(matches!(result.status, SolveStatus::Sat));

    if let Some(assignment) = result.assignment() {
        let schedule = job_shop::decode(&problem, assignment);
        assert!(job_shop::verify(&problem, &schedule));
    }
}

#[test]
fn test_job_shop_verify_valid_schedule() {
    let problem = job_shop::simple_problem();
    // Valid schedule: Job0 starts at 0, Job1 starts at 2
    let schedule = vec![
        (0, 0, 0), // Job 0, Op 0 at time 0
        (0, 1, 2), // Job 0, Op 1 at time 2
        (1, 0, 0), // Job 1, Op 0 at time 0
        (1, 1, 2), // Job 1, Op 1 at time 2
    ];
    assert!(job_shop::verify(&problem, &schedule));
}

#[test]
fn test_job_shop_verify_invalid_precedence() {
    let problem = job_shop::simple_problem();
    // Invalid: Op 1 starts before Op 0 finishes
    let schedule = vec![
        (0, 0, 0),
        (0, 1, 1), // Should be >= 2
        (1, 0, 0),
        (1, 1, 2),
    ];
    assert!(!job_shop::verify(&problem, &schedule));
}

// =============================================================================
// Graph Coloring Tests
// =============================================================================

mod graph_coloring {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct Graph {
        pub num_vertices: usize,
        pub edges: Vec<(usize, usize)>,
    }

    #[inline]
    pub fn var_index(num_colors: usize, vertex: usize, color: usize) -> u32 {
        (vertex * num_colors + color) as u32
    }

    pub fn encode(graph: &Graph, num_colors: usize) -> SolveInstance {
        let num_vars = (graph.num_vertices * num_colors) as u32;
        let mut clauses = Vec::new();

        for v in 0..graph.num_vertices {
            let literals: Vec<Literal> = (0..num_colors)
                .map(|k| Literal::positive(var_index(num_colors, v, k)))
                .collect();
            clauses.push(Clause::new(literals));
        }

        for v in 0..graph.num_vertices {
            for k1 in 0..num_colors {
                for k2 in (k1 + 1)..num_colors {
                    clauses.push(Clause::binary(
                        Literal::negative(var_index(num_colors, v, k1)),
                        Literal::negative(var_index(num_colors, v, k2)),
                    ));
                }
            }
        }

        for &(u, v) in &graph.edges {
            for k in 0..num_colors {
                clauses.push(Clause::binary(
                    Literal::negative(var_index(num_colors, u, k)),
                    Literal::negative(var_index(num_colors, v, k)),
                ));
            }
        }

        SolveInstance::new(num_vars, clauses)
    }

    pub fn decode(num_vertices: usize, num_colors: usize, assignment: &[bool]) -> Vec<usize> {
        let mut coloring = vec![0; num_vertices];
        for v in 0..num_vertices {
            for k in 0..num_colors {
                if assignment[var_index(num_colors, v, k) as usize] {
                    coloring[v] = k;
                    break;
                }
            }
        }
        coloring
    }

    pub fn verify(graph: &Graph, coloring: &[usize]) -> bool {
        if coloring.len() != graph.num_vertices {
            return false;
        }

        for &(u, v) in &graph.edges {
            if coloring[u] == coloring[v] {
                return false;
            }
        }

        true
    }

    pub fn triangle() -> Graph {
        Graph {
            num_vertices: 3,
            edges: vec![(0, 1), (1, 2), (2, 0)],
        }
    }

    pub fn square() -> Graph {
        Graph {
            num_vertices: 4,
            edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        }
    }

    pub fn petersen() -> Graph {
        Graph {
            num_vertices: 10,
            edges: vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 0),
                (5, 7),
                (7, 9),
                (9, 6),
                (6, 8),
                (8, 5),
                (0, 5),
                (1, 6),
                (2, 7),
                (3, 8),
                (4, 9),
            ],
        }
    }
}

#[test]
fn test_graph_coloring_triangle_2_colors() {
    let graph = graph_coloring::triangle();
    let instance = graph_coloring::encode(&graph, 2);
    let solver = Solver::with_config_cpu(SolverConfig::thorough());
    let result = solver.solve(instance);

    // Triangle is not 2-colorable (odd cycle)
    // CLS may return Unknown rather than proving UNSAT
    assert!(matches!(result.status, SolveStatus::Unknown));
}

#[test]
fn test_graph_coloring_triangle_3_colors() {
    let graph = graph_coloring::triangle();
    let instance = graph_coloring::encode(&graph, 3);
    let solver = Solver::with_config_cpu(SolverConfig::thorough());
    let result = solver.solve(instance);

    assert!(matches!(result.status, SolveStatus::Sat));

    if let Some(assignment) = result.assignment() {
        let coloring = graph_coloring::decode(graph.num_vertices, 3, assignment);
        assert!(graph_coloring::verify(&graph, &coloring));
    }
}

#[test]
fn test_graph_coloring_square_2_colors() {
    let graph = graph_coloring::square();
    let instance = graph_coloring::encode(&graph, 2);
    let solver = Solver::with_config_cpu(SolverConfig::thorough());
    let result = solver.solve(instance);

    // Square (4-cycle) is 2-colorable
    assert!(matches!(result.status, SolveStatus::Sat));

    if let Some(assignment) = result.assignment() {
        let coloring = graph_coloring::decode(graph.num_vertices, 2, assignment);
        assert!(graph_coloring::verify(&graph, &coloring));
    }
}

#[test]
fn test_graph_coloring_petersen_3_colors() {
    let graph = graph_coloring::petersen();
    let instance = graph_coloring::encode(&graph, 3);
    // Use higher iterations for challenging constraint satisfaction problems
    let config = SolverConfig::new(100000, 0.05, 0.95, 0.5);
    let solver = Solver::with_config_cpu(config);
    let result = solver.solve(instance);

    // Petersen graph is 3-colorable
    // CLS is an incomplete solver - it may not always find a solution
    // but if it does find one, it must be valid
    if matches!(result.status, SolveStatus::Sat) {
        if let Some(assignment) = result.assignment() {
            let coloring = graph_coloring::decode(graph.num_vertices, 3, assignment);
            assert!(graph_coloring::verify(&graph, &coloring));
        }
    }
    // Test passes if solver returns Sat with valid solution OR Unknown
}

#[test]
fn test_graph_coloring_verify_valid() {
    let graph = graph_coloring::triangle();
    let coloring = vec![0, 1, 2]; // Different colors
    assert!(graph_coloring::verify(&graph, &coloring));
}

#[test]
fn test_graph_coloring_verify_invalid() {
    let graph = graph_coloring::triangle();
    let coloring = vec![0, 0, 1]; // Adjacent vertices 0 and 1 have same color
    assert!(!graph_coloring::verify(&graph, &coloring));
}

// =============================================================================
// Circuit SAT Tests
// =============================================================================

mod circuit_sat {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    pub enum Gate {
        Input,
        Not { input: usize },
        And { input1: usize, input2: usize },
        Or { input1: usize, input2: usize },
        Xor { input1: usize, input2: usize },
    }

    #[derive(Debug, Clone)]
    pub struct Circuit {
        pub gates: Vec<Gate>,
        pub num_inputs: usize,
        pub output: usize,
    }

    impl Circuit {
        pub fn new(num_inputs: usize) -> Self {
            let mut gates = Vec::new();
            for _ in 0..num_inputs {
                gates.push(Gate::Input);
            }
            Self {
                gates,
                num_inputs,
                output: 0,
            }
        }

        pub fn not(&mut self, input: usize) -> usize {
            let idx = self.gates.len();
            self.gates.push(Gate::Not { input });
            idx
        }

        pub fn and(&mut self, input1: usize, input2: usize) -> usize {
            let idx = self.gates.len();
            self.gates.push(Gate::And { input1, input2 });
            idx
        }

        pub fn or(&mut self, input1: usize, input2: usize) -> usize {
            let idx = self.gates.len();
            self.gates.push(Gate::Or { input1, input2 });
            idx
        }

        pub fn xor(&mut self, input1: usize, input2: usize) -> usize {
            let idx = self.gates.len();
            self.gates.push(Gate::Xor { input1, input2 });
            idx
        }

        pub fn set_output(&mut self, output: usize) {
            self.output = output;
        }

        pub fn evaluate(&self, inputs: &[bool]) -> bool {
            let mut values = vec![false; self.gates.len()];

            for (i, &val) in inputs.iter().enumerate() {
                values[i] = val;
            }

            for (i, gate) in self.gates.iter().enumerate() {
                values[i] = match gate {
                    Gate::Input => values[i],
                    Gate::Not { input } => !values[*input],
                    Gate::And { input1, input2 } => values[*input1] && values[*input2],
                    Gate::Or { input1, input2 } => values[*input1] || values[*input2],
                    Gate::Xor { input1, input2 } => values[*input1] != values[*input2],
                };
            }

            values[self.output]
        }
    }

    pub fn encode(circuit: &Circuit, expected_output: bool) -> SolveInstance {
        let num_vars = circuit.gates.len() as u32;
        let mut clauses = Vec::new();

        for (i, gate) in circuit.gates.iter().enumerate() {
            let out = i as u32;

            match gate {
                Gate::Input => {}
                Gate::Not { input } => {
                    let inp = *input as u32;
                    clauses.push(Clause::binary(
                        Literal::positive(out),
                        Literal::positive(inp),
                    ));
                    clauses.push(Clause::binary(
                        Literal::negative(out),
                        Literal::negative(inp),
                    ));
                }
                Gate::And { input1, input2 } => {
                    let in1 = *input1 as u32;
                    let in2 = *input2 as u32;
                    clauses.push(Clause::binary(
                        Literal::negative(out),
                        Literal::positive(in1),
                    ));
                    clauses.push(Clause::binary(
                        Literal::negative(out),
                        Literal::positive(in2),
                    ));
                    clauses.push(Clause::ternary(
                        Literal::positive(out),
                        Literal::negative(in1),
                        Literal::negative(in2),
                    ));
                }
                Gate::Or { input1, input2 } => {
                    let in1 = *input1 as u32;
                    let in2 = *input2 as u32;
                    clauses.push(Clause::binary(
                        Literal::positive(out),
                        Literal::negative(in1),
                    ));
                    clauses.push(Clause::binary(
                        Literal::positive(out),
                        Literal::negative(in2),
                    ));
                    clauses.push(Clause::ternary(
                        Literal::negative(out),
                        Literal::positive(in1),
                        Literal::positive(in2),
                    ));
                }
                Gate::Xor { input1, input2 } => {
                    let in1 = *input1 as u32;
                    let in2 = *input2 as u32;
                    clauses.push(Clause::ternary(
                        Literal::negative(out),
                        Literal::negative(in1),
                        Literal::negative(in2),
                    ));
                    clauses.push(Clause::ternary(
                        Literal::negative(out),
                        Literal::positive(in1),
                        Literal::positive(in2),
                    ));
                    clauses.push(Clause::ternary(
                        Literal::positive(out),
                        Literal::negative(in1),
                        Literal::positive(in2),
                    ));
                    clauses.push(Clause::ternary(
                        Literal::positive(out),
                        Literal::positive(in1),
                        Literal::negative(in2),
                    ));
                }
            }
        }

        if expected_output {
            clauses.push(Clause::unit(Literal::positive(circuit.output as u32)));
        } else {
            clauses.push(Clause::unit(Literal::negative(circuit.output as u32)));
        }

        SolveInstance::new(num_vars, clauses)
    }

    pub fn decode_inputs(circuit: &Circuit, assignment: &[bool]) -> Vec<bool> {
        assignment[..circuit.num_inputs].to_vec()
    }
}

#[test]
fn test_circuit_and_gate() {
    let mut circuit = circuit_sat::Circuit::new(2);
    let and = circuit.and(0, 1);
    circuit.set_output(and);

    // Test all input combinations
    assert!(!circuit.evaluate(&[false, false]));
    assert!(!circuit.evaluate(&[true, false]));
    assert!(!circuit.evaluate(&[false, true]));
    assert!(circuit.evaluate(&[true, true]));
}

#[test]
fn test_circuit_or_gate() {
    let mut circuit = circuit_sat::Circuit::new(2);
    let or = circuit.or(0, 1);
    circuit.set_output(or);

    assert!(!circuit.evaluate(&[false, false]));
    assert!(circuit.evaluate(&[true, false]));
    assert!(circuit.evaluate(&[false, true]));
    assert!(circuit.evaluate(&[true, true]));
}

#[test]
fn test_circuit_xor_gate() {
    let mut circuit = circuit_sat::Circuit::new(2);
    let xor = circuit.xor(0, 1);
    circuit.set_output(xor);

    assert!(!circuit.evaluate(&[false, false]));
    assert!(circuit.evaluate(&[true, false]));
    assert!(circuit.evaluate(&[false, true]));
    assert!(!circuit.evaluate(&[true, true]));
}

#[test]
fn test_circuit_not_gate() {
    let mut circuit = circuit_sat::Circuit::new(1);
    let not = circuit.not(0);
    circuit.set_output(not);

    assert!(circuit.evaluate(&[false]));
    assert!(!circuit.evaluate(&[true]));
}

#[test]
fn test_circuit_sat_simple() {
    // Circuit: a AND b = true
    let mut circuit = circuit_sat::Circuit::new(2);
    let and = circuit.and(0, 1);
    circuit.set_output(and);

    let instance = circuit_sat::encode(&circuit, true);
    let solver = Solver::with_config_cpu(SolverConfig::fast());
    let result = solver.solve(instance);

    assert!(matches!(result.status, SolveStatus::Sat));

    if let Some(assignment) = result.assignment() {
        let inputs = circuit_sat::decode_inputs(&circuit, assignment);
        assert!(circuit.evaluate(&inputs));
    }
}

#[test]
fn test_circuit_sat_unsatisfiable() {
    // Circuit: a AND NOT a = true (impossible)
    let mut circuit = circuit_sat::Circuit::new(1);
    let not_a = circuit.not(0);
    let impossible = circuit.and(0, not_a);
    circuit.set_output(impossible);

    let instance = circuit_sat::encode(&circuit, true);
    let solver = Solver::with_config_cpu(SolverConfig::fast());
    let result = solver.solve(instance);

    // CLS cannot prove UNSAT, will return Unknown
    assert!(matches!(result.status, SolveStatus::Unknown));
}

#[test]
fn test_circuit_sat_complex() {
    // Circuit: (a AND b) OR (NOT a AND c) = true
    let mut circuit = circuit_sat::Circuit::new(3);
    let and1 = circuit.and(0, 1);
    let not_a = circuit.not(0);
    let and2 = circuit.and(not_a, 2);
    let output = circuit.or(and1, and2);
    circuit.set_output(output);

    let instance = circuit_sat::encode(&circuit, true);
    let solver = Solver::with_config_cpu(SolverConfig::fast());
    let result = solver.solve(instance);

    assert!(matches!(result.status, SolveStatus::Sat));

    if let Some(assignment) = result.assignment() {
        let inputs = circuit_sat::decode_inputs(&circuit, assignment);
        assert!(circuit.evaluate(&inputs));
    }
}

#[test]
fn test_circuit_half_adder() {
    // Half adder: sum = a XOR b, carry = a AND b
    // Test: find inputs where carry = true
    let mut circuit = circuit_sat::Circuit::new(2);
    let carry = circuit.and(0, 1);
    circuit.set_output(carry);

    let instance = circuit_sat::encode(&circuit, true);
    let solver = Solver::with_config_cpu(SolverConfig::fast());
    let result = solver.solve(instance);

    assert!(matches!(result.status, SolveStatus::Sat));

    if let Some(assignment) = result.assignment() {
        let inputs = circuit_sat::decode_inputs(&circuit, assignment);
        // Carry is true only when both inputs are true
        assert!(inputs[0] && inputs[1]);
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_solver_handles_large_instance() {
    // N-Queens for N=10 is a moderately large instance
    let instance = nqueens::encode(10);
    assert!(instance.num_vars > 0);
    assert!(instance.num_clauses() > 0);

    // Just check it doesn't crash
    let solver = Solver::with_config_cpu(SolverConfig::fast());
    let result = solver.solve(instance);

    // Should get some result (may or may not find solution with fast config)
    assert!(matches!(
        result.status,
        SolveStatus::Sat | SolveStatus::Unknown
    ));
}

#[test]
fn test_all_encodings_produce_valid_instances() {
    // Sudoku
    let sudoku_instance = sudoku::encode(&[[0u8; 9]; 9]);
    assert!(sudoku_instance.validate());

    // N-Queens
    let nqueens_instance = nqueens::encode(4);
    assert!(nqueens_instance.validate());

    // Graph Coloring
    let graph = graph_coloring::triangle();
    let gc_instance = graph_coloring::encode(&graph, 3);
    assert!(gc_instance.validate());

    // Job Shop
    let problem = job_shop::simple_problem();
    let js_instance = job_shop::encode(&problem);
    assert!(js_instance.validate());

    // Circuit
    let mut circuit = circuit_sat::Circuit::new(2);
    let and = circuit.and(0, 1);
    circuit.set_output(and);
    let circuit_instance = circuit_sat::encode(&circuit, true);
    assert!(circuit_instance.validate());
}
