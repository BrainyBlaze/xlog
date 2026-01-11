//! Real-world SAT problem examples demonstrating xlog-solve capabilities.
//!
//! This example implements production-grade SAT encodings for:
//! 1. Sudoku Solver - 9x9 Sudoku as SAT
//! 2. N-Queens Problem - Place N queens on NxN board
//! 3. Job Shop Scheduling - Schedule jobs on machines
//! 4. Graph Coloring - Color a graph with k colors
//! 5. Circuit SAT - Verify boolean circuits
//!
//! Run with: `cargo run --example real_world`

use xlog_solve::{Clause, Literal, SolveInstance, Solver, SolverConfig};

// =============================================================================
// Sudoku Solver
// =============================================================================

/// Sudoku solver using SAT encoding.
///
/// Variables: x_{r,c,v} = true iff cell (r,c) contains value v
/// Variable index: r * 81 + c * 9 + v (0-indexed)
pub mod sudoku {
    use super::*;

    /// Total number of variables for 9x9 Sudoku (9 * 9 * 9 = 729)
    pub const NUM_VARS: u32 = 729;

    /// Convert (row, col, value) to variable index.
    /// All inputs are 0-indexed (0-8).
    #[inline]
    pub fn var_index(row: usize, col: usize, val: usize) -> u32 {
        (row * 81 + col * 9 + val) as u32
    }

    /// Convert variable index back to (row, col, value).
    #[inline]
    pub fn from_var_index(var: u32) -> (usize, usize, usize) {
        let var = var as usize;
        let row = var / 81;
        let col = (var % 81) / 9;
        let val = var % 9;
        (row, col, val)
    }

    /// Creates the SAT encoding for a Sudoku puzzle.
    ///
    /// # Arguments
    /// * `grid` - 9x9 grid where 0 means empty, 1-9 means fixed value
    ///
    /// # Returns
    /// A SolveInstance encoding the Sudoku constraints.
    pub fn encode(grid: &[[u8; 9]; 9]) -> SolveInstance {
        let mut clauses = Vec::new();

        // Constraint 1: Each cell has at least one value
        // For each cell (r,c): x_{r,c,1} OR x_{r,c,2} OR ... OR x_{r,c,9}
        for row in 0..9 {
            for col in 0..9 {
                let literals: Vec<Literal> = (0..9)
                    .map(|val| Literal::positive(var_index(row, col, val)))
                    .collect();
                clauses.push(Clause::new(literals));
            }
        }

        // Constraint 2: Each cell has at most one value (pairwise)
        // For each cell (r,c) and values v1 < v2: NOT x_{r,c,v1} OR NOT x_{r,c,v2}
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

        // Constraint 3: Each row has each value exactly once
        for row in 0..9 {
            for val in 0..9 {
                // At least one cell in this row has this value
                let literals: Vec<Literal> = (0..9)
                    .map(|col| Literal::positive(var_index(row, col, val)))
                    .collect();
                clauses.push(Clause::new(literals));

                // At most one cell in this row has this value (pairwise)
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

        // Constraint 4: Each column has each value exactly once
        for col in 0..9 {
            for val in 0..9 {
                // At least one cell in this column has this value
                let literals: Vec<Literal> = (0..9)
                    .map(|row| Literal::positive(var_index(row, col, val)))
                    .collect();
                clauses.push(Clause::new(literals));

                // At most one cell in this column has this value (pairwise)
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

        // Constraint 5: Each 3x3 box has each value exactly once
        for box_row in 0..3 {
            for box_col in 0..3 {
                for val in 0..9 {
                    // Collect all cells in this box
                    let mut cells = Vec::new();
                    for dr in 0..3 {
                        for dc in 0..3 {
                            cells.push((box_row * 3 + dr, box_col * 3 + dc));
                        }
                    }

                    // At least one cell in this box has this value
                    let literals: Vec<Literal> = cells
                        .iter()
                        .map(|&(r, c)| Literal::positive(var_index(r, c, val)))
                        .collect();
                    clauses.push(Clause::new(literals));

                    // At most one cell in this box has this value (pairwise)
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

        // Constraint 6: Fixed values from the input grid
        for row in 0..9 {
            for col in 0..9 {
                if grid[row][col] != 0 {
                    let val = (grid[row][col] - 1) as usize; // Convert 1-9 to 0-8
                    clauses.push(Clause::unit(Literal::positive(var_index(row, col, val))));
                }
            }
        }

        SolveInstance::new(NUM_VARS, clauses)
    }

    /// Decodes the SAT solution back to a Sudoku grid.
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

    /// Verifies that a Sudoku solution is valid.
    pub fn verify(grid: &[[u8; 9]; 9]) -> bool {
        // Check all values are 1-9
        for row in 0..9 {
            for col in 0..9 {
                if grid[row][col] < 1 || grid[row][col] > 9 {
                    return false;
                }
            }
        }

        // Check rows
        for row in 0..9 {
            let mut seen = [false; 9];
            for col in 0..9 {
                let val = (grid[row][col] - 1) as usize;
                if seen[val] {
                    return false;
                }
                seen[val] = true;
            }
        }

        // Check columns
        for col in 0..9 {
            let mut seen = [false; 9];
            for row in 0..9 {
                let val = (grid[row][col] - 1) as usize;
                if seen[val] {
                    return false;
                }
                seen[val] = true;
            }
        }

        // Check 3x3 boxes
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

    /// Pretty-prints a Sudoku grid.
    pub fn print_grid(grid: &[[u8; 9]; 9]) {
        for (i, row) in grid.iter().enumerate() {
            if i % 3 == 0 && i != 0 {
                println!("------+-------+------");
            }
            for (j, &val) in row.iter().enumerate() {
                if j % 3 == 0 && j != 0 {
                    print!("| ");
                }
                if val == 0 {
                    print!(". ");
                } else {
                    print!("{} ", val);
                }
            }
            println!();
        }
    }

    /// Returns a sample puzzle (medium difficulty).
    pub fn sample_puzzle() -> [[u8; 9]; 9] {
        [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]
    }
}

// =============================================================================
// N-Queens Problem
// =============================================================================

/// N-Queens problem solver using SAT encoding.
///
/// Variables: q_{r,c} = true iff there is a queen at (r,c)
/// Variable index: r * N + c (0-indexed)
pub mod nqueens {
    use super::*;

    /// Convert (row, col) to variable index for an NxN board.
    #[inline]
    pub fn var_index(n: usize, row: usize, col: usize) -> u32 {
        (row * n + col) as u32
    }

    /// Convert variable index back to (row, col).
    #[inline]
    pub fn from_var_index(n: usize, var: u32) -> (usize, usize) {
        let var = var as usize;
        (var / n, var % n)
    }

    /// Creates the SAT encoding for the N-Queens problem.
    ///
    /// # Arguments
    /// * `n` - Board size (NxN)
    ///
    /// # Returns
    /// A SolveInstance encoding the N-Queens constraints.
    pub fn encode(n: usize) -> SolveInstance {
        let num_vars = (n * n) as u32;
        let mut clauses = Vec::new();

        // Constraint 1: At least one queen per row
        for row in 0..n {
            let literals: Vec<Literal> = (0..n)
                .map(|col| Literal::positive(var_index(n, row, col)))
                .collect();
            clauses.push(Clause::new(literals));
        }

        // Constraint 2: At most one queen per row (pairwise)
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

        // Constraint 3: At most one queen per column (pairwise)
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

        // Constraint 4: At most one queen per diagonal (top-left to bottom-right)
        // Diagonals where row - col = constant
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

        // Constraint 5: At most one queen per anti-diagonal (top-right to bottom-left)
        // Diagonals where row + col = constant
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

    /// Decodes the SAT solution back to queen positions.
    /// Returns a vector of (row, col) pairs.
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

    /// Verifies that a solution is valid.
    pub fn verify(n: usize, queens: &[(usize, usize)]) -> bool {
        // Must have exactly n queens
        if queens.len() != n {
            return false;
        }

        // Check all positions are valid
        for &(row, col) in queens {
            if row >= n || col >= n {
                return false;
            }
        }

        // Check no two queens attack each other
        for i in 0..queens.len() {
            for j in (i + 1)..queens.len() {
                let (r1, c1) = queens[i];
                let (r2, c2) = queens[j];

                // Same row
                if r1 == r2 {
                    return false;
                }
                // Same column
                if c1 == c2 {
                    return false;
                }
                // Same diagonal
                if (r1 as i32 - r2 as i32).abs() == (c1 as i32 - c2 as i32).abs() {
                    return false;
                }
            }
        }

        true
    }

    /// Pretty-prints the board with queens.
    pub fn print_board(n: usize, queens: &[(usize, usize)]) {
        let mut board = vec![vec!['.'; n]; n];
        for &(row, col) in queens {
            board[row][col] = 'Q';
        }
        for row in &board {
            for &cell in row {
                print!("{} ", cell);
            }
            println!();
        }
    }
}

// =============================================================================
// Job Shop Scheduling
// =============================================================================

/// Job Shop Scheduling problem solver using SAT encoding.
///
/// Problem: Schedule operations on machines with precedence constraints.
/// Each job has a sequence of operations, each requiring a specific machine for a duration.
///
/// Variables:
/// - x_{j,o,t} = operation o of job j starts at time t
/// - We use a time-indexed formulation with a given time horizon
pub mod job_shop {
    use super::*;

    /// A single operation in a job.
    #[derive(Debug, Clone, Copy)]
    pub struct Operation {
        /// Machine this operation runs on (0-indexed)
        pub machine: usize,
        /// Duration of this operation
        pub duration: usize,
    }

    /// A job consisting of a sequence of operations.
    #[derive(Debug, Clone)]
    pub struct Job {
        /// Operations in order (must be executed sequentially)
        pub operations: Vec<Operation>,
    }

    /// A job shop scheduling problem instance.
    #[derive(Debug, Clone)]
    pub struct JobShopProblem {
        /// Number of machines
        pub num_machines: usize,
        /// Jobs to schedule
        pub jobs: Vec<Job>,
        /// Time horizon (makespan bound)
        pub horizon: usize,
    }

    impl JobShopProblem {
        /// Calculate the total number of operations.
        pub fn total_operations(&self) -> usize {
            self.jobs.iter().map(|j| j.operations.len()).sum()
        }

        /// Variable index for operation o of job j starting at time t.
        pub fn var_index(&self, job: usize, op: usize, time: usize) -> u32 {
            // Calculate offset for this operation
            let mut offset = 0;
            for j in 0..job {
                offset += self.jobs[j].operations.len() * self.horizon;
            }
            offset += op * self.horizon;
            (offset + time) as u32
        }

        /// Total number of SAT variables.
        pub fn num_vars(&self) -> u32 {
            (self.total_operations() * self.horizon) as u32
        }
    }

    /// Creates the SAT encoding for a job shop scheduling problem.
    pub fn encode(problem: &JobShopProblem) -> SolveInstance {
        let num_vars = problem.num_vars();
        let mut clauses = Vec::new();

        // Constraint 1: Each operation starts at exactly one time
        for (j, job) in problem.jobs.iter().enumerate() {
            for o in 0..job.operations.len() {
                let op = &job.operations[o];

                // Operation must finish within horizon
                let latest_start = problem.horizon.saturating_sub(op.duration);

                // At least one start time
                let literals: Vec<Literal> = (0..=latest_start)
                    .map(|t| Literal::positive(problem.var_index(j, o, t)))
                    .collect();
                if !literals.is_empty() {
                    clauses.push(Clause::new(literals));
                }

                // At most one start time (pairwise)
                for t1 in 0..=latest_start {
                    for t2 in (t1 + 1)..=latest_start {
                        clauses.push(Clause::binary(
                            Literal::negative(problem.var_index(j, o, t1)),
                            Literal::negative(problem.var_index(j, o, t2)),
                        ));
                    }
                }

                // Cannot start too late
                for t in (latest_start + 1)..problem.horizon {
                    clauses.push(Clause::unit(Literal::negative(problem.var_index(j, o, t))));
                }
            }
        }

        // Constraint 2: Precedence within jobs
        // If operation o starts at time t, operation o+1 must start at time >= t + duration(o)
        for (j, job) in problem.jobs.iter().enumerate() {
            for o in 0..(job.operations.len().saturating_sub(1)) {
                let op = &job.operations[o];
                let next_op = &job.operations[o + 1];

                let latest_start = problem.horizon.saturating_sub(op.duration);
                let next_latest_start = problem.horizon.saturating_sub(next_op.duration);

                for t in 0..=latest_start {
                    // If operation o starts at t, operation o+1 must start at t + duration or later
                    let earliest_next = t + op.duration;

                    // x_{j,o,t} => x_{j,o+1,earliest_next} OR x_{j,o+1,earliest_next+1} OR ...
                    // Equivalent to: NOT x_{j,o,t} OR (x_{j,o+1,earliest_next} OR ...)
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

        // Constraint 3: No two operations on the same machine can overlap
        // Collect all operations by machine
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

                    // Operations (j1,o1) and (j2,o2) cannot overlap
                    // If (j1,o1) starts at t1, (j2,o2) must start before t1 or after t1+d1
                    // This is complex, so we use: NOT (x_{j1,o1,t1} AND x_{j2,o2,t2}) for overlapping times
                    let latest1 = problem.horizon.saturating_sub(d1);
                    let latest2 = problem.horizon.saturating_sub(d2);

                    for t1 in 0..=latest1 {
                        for t2 in 0..=latest2 {
                            // Check if they overlap: [t1, t1+d1) intersects [t2, t2+d2)
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

    /// Decoded schedule: (job, operation, start_time) triples.
    pub type Schedule = Vec<(usize, usize, usize)>;

    /// Decodes the SAT solution back to a schedule.
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

    /// Verifies that a schedule is valid.
    pub fn verify(problem: &JobShopProblem, schedule: &Schedule) -> bool {
        // Build a map of (job, op) -> start_time
        let mut start_times: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();

        for &(j, o, t) in schedule {
            start_times.insert((j, o), t);
        }

        // Check all operations are scheduled
        for (j, job) in problem.jobs.iter().enumerate() {
            for o in 0..job.operations.len() {
                if !start_times.contains_key(&(j, o)) {
                    return false;
                }
            }
        }

        // Check precedence constraints
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

        // Check no machine conflicts
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

                    // Check for overlap
                    let overlaps = !(t1 + d1 <= t2 || t2 + d2 <= t1);
                    if overlaps {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Pretty-prints a schedule.
    pub fn print_schedule(problem: &JobShopProblem, schedule: &Schedule) {
        println!("Schedule:");
        for (j, job) in problem.jobs.iter().enumerate() {
            print!("  Job {}: ", j);
            for o in 0..job.operations.len() {
                for &(sj, so, t) in schedule {
                    if sj == j && so == o {
                        let op = &job.operations[o];
                        print!(
                            "Op{} (M{}, t={}..{}) ",
                            o,
                            op.machine,
                            t,
                            t + op.duration
                        );
                        break;
                    }
                }
            }
            println!();
        }

        // Calculate makespan
        let makespan = schedule
            .iter()
            .map(|&(j, o, t)| t + problem.jobs[j].operations[o].duration)
            .max()
            .unwrap_or(0);
        println!("  Makespan: {}", makespan);
    }

    /// Creates a sample job shop problem with 5 jobs and 3 machines.
    pub fn sample_problem() -> JobShopProblem {
        JobShopProblem {
            num_machines: 3,
            horizon: 20, // Upper bound on makespan
            jobs: vec![
                Job {
                    operations: vec![
                        Operation { machine: 0, duration: 3 },
                        Operation { machine: 1, duration: 2 },
                        Operation { machine: 2, duration: 2 },
                    ],
                },
                Job {
                    operations: vec![
                        Operation { machine: 0, duration: 2 },
                        Operation { machine: 2, duration: 1 },
                        Operation { machine: 1, duration: 4 },
                    ],
                },
                Job {
                    operations: vec![
                        Operation { machine: 1, duration: 4 },
                        Operation { machine: 2, duration: 3 },
                    ],
                },
                Job {
                    operations: vec![
                        Operation { machine: 2, duration: 2 },
                        Operation { machine: 0, duration: 3 },
                        Operation { machine: 1, duration: 1 },
                    ],
                },
                Job {
                    operations: vec![
                        Operation { machine: 1, duration: 3 },
                        Operation { machine: 0, duration: 2 },
                    ],
                },
            ],
        }
    }
}

// =============================================================================
// Graph Coloring
// =============================================================================

/// Graph Coloring problem solver using SAT encoding.
///
/// Variables: c_{v,k} = true iff vertex v has color k
/// Variable index: v * num_colors + k
pub mod graph_coloring {
    use super::*;

    /// A graph represented as an adjacency list.
    #[derive(Debug, Clone)]
    pub struct Graph {
        /// Number of vertices
        pub num_vertices: usize,
        /// Edges as pairs of vertex indices
        pub edges: Vec<(usize, usize)>,
    }

    /// Convert (vertex, color) to variable index.
    #[inline]
    pub fn var_index(num_colors: usize, vertex: usize, color: usize) -> u32 {
        (vertex * num_colors + color) as u32
    }

    /// Creates the SAT encoding for the graph coloring problem.
    ///
    /// # Arguments
    /// * `graph` - The graph to color
    /// * `num_colors` - Number of colors available
    ///
    /// # Returns
    /// A SolveInstance encoding the graph coloring constraints.
    pub fn encode(graph: &Graph, num_colors: usize) -> SolveInstance {
        let num_vars = (graph.num_vertices * num_colors) as u32;
        let mut clauses = Vec::new();

        // Constraint 1: Each vertex has at least one color
        for v in 0..graph.num_vertices {
            let literals: Vec<Literal> = (0..num_colors)
                .map(|k| Literal::positive(var_index(num_colors, v, k)))
                .collect();
            clauses.push(Clause::new(literals));
        }

        // Constraint 2: Each vertex has at most one color (pairwise)
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

        // Constraint 3: Adjacent vertices have different colors
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

    /// Decodes the SAT solution back to a coloring.
    /// Returns a vector where index v contains the color of vertex v.
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

    /// Verifies that a coloring is valid.
    pub fn verify(graph: &Graph, coloring: &[usize]) -> bool {
        // Check correct number of vertices
        if coloring.len() != graph.num_vertices {
            return false;
        }

        // Check adjacent vertices have different colors
        for &(u, v) in &graph.edges {
            if coloring[u] == coloring[v] {
                return false;
            }
        }

        true
    }

    /// Pretty-prints a coloring.
    pub fn print_coloring(coloring: &[usize]) {
        let color_names = ["Red", "Green", "Blue", "Yellow", "Purple", "Orange"];
        println!("Graph Coloring:");
        for (v, &c) in coloring.iter().enumerate() {
            let name = if c < color_names.len() {
                color_names[c]
            } else {
                "Color?"
            };
            println!("  Vertex {}: {} ({})", v, c, name);
        }
    }

    /// Creates the Petersen graph (10 vertices, 15 edges).
    /// The Petersen graph is 3-colorable but not 2-colorable.
    pub fn petersen_graph() -> Graph {
        Graph {
            num_vertices: 10,
            edges: vec![
                // Outer pentagon
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 0),
                // Inner pentagram
                (5, 7),
                (7, 9),
                (9, 6),
                (6, 8),
                (8, 5),
                // Spokes
                (0, 5),
                (1, 6),
                (2, 7),
                (3, 8),
                (4, 9),
            ],
        }
    }
}

// =============================================================================
// Circuit SAT
// =============================================================================

/// Circuit SAT solver using Tseitin transformation.
///
/// Encodes boolean circuits as CNF formulas for SAT solving.
/// Supports AND, OR, NOT, XOR, and IMPLIES gates.
pub mod circuit_sat {
    use super::*;

    /// A boolean circuit gate.
    #[derive(Debug, Clone, Copy)]
    pub enum Gate {
        /// Input variable (no operation)
        Input,
        /// NOT gate: output = NOT input1
        Not { input: usize },
        /// AND gate: output = input1 AND input2
        And { input1: usize, input2: usize },
        /// OR gate: output = input1 OR input2
        Or { input1: usize, input2: usize },
        /// XOR gate: output = input1 XOR input2
        Xor { input1: usize, input2: usize },
    }

    /// A boolean circuit.
    #[derive(Debug, Clone)]
    pub struct Circuit {
        /// Gates in topological order (inputs first)
        pub gates: Vec<Gate>,
        /// Number of primary inputs
        pub num_inputs: usize,
        /// Index of the output gate
        pub output: usize,
    }

    impl Circuit {
        /// Creates a new circuit builder.
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

        /// Adds a NOT gate and returns its index.
        pub fn not(&mut self, input: usize) -> usize {
            let idx = self.gates.len();
            self.gates.push(Gate::Not { input });
            idx
        }

        /// Adds an AND gate and returns its index.
        pub fn and(&mut self, input1: usize, input2: usize) -> usize {
            let idx = self.gates.len();
            self.gates.push(Gate::And { input1, input2 });
            idx
        }

        /// Adds an OR gate and returns its index.
        pub fn or(&mut self, input1: usize, input2: usize) -> usize {
            let idx = self.gates.len();
            self.gates.push(Gate::Or { input1, input2 });
            idx
        }

        /// Adds a XOR gate and returns its index.
        pub fn xor(&mut self, input1: usize, input2: usize) -> usize {
            let idx = self.gates.len();
            self.gates.push(Gate::Xor { input1, input2 });
            idx
        }

        /// Sets the output gate.
        pub fn set_output(&mut self, output: usize) {
            self.output = output;
        }

        /// Evaluates the circuit with given inputs.
        pub fn evaluate(&self, inputs: &[bool]) -> bool {
            assert_eq!(inputs.len(), self.num_inputs);

            let mut values = vec![false; self.gates.len()];

            // Set input values
            for (i, &val) in inputs.iter().enumerate() {
                values[i] = val;
            }

            // Evaluate gates in order
            for (i, gate) in self.gates.iter().enumerate() {
                values[i] = match gate {
                    Gate::Input => values[i], // Already set
                    Gate::Not { input } => !values[*input],
                    Gate::And { input1, input2 } => values[*input1] && values[*input2],
                    Gate::Or { input1, input2 } => values[*input1] || values[*input2],
                    Gate::Xor { input1, input2 } => values[*input1] != values[*input2],
                };
            }

            values[self.output]
        }
    }

    /// Creates the SAT encoding for a circuit using Tseitin transformation.
    ///
    /// # Arguments
    /// * `circuit` - The circuit to encode
    /// * `expected_output` - The expected output value (usually true)
    ///
    /// # Returns
    /// A SolveInstance encoding the circuit constraints.
    pub fn encode(circuit: &Circuit, expected_output: bool) -> SolveInstance {
        let num_vars = circuit.gates.len() as u32;
        let mut clauses = Vec::new();

        // Tseitin transformation for each gate
        for (i, gate) in circuit.gates.iter().enumerate() {
            let out = i as u32;

            match gate {
                Gate::Input => {
                    // No constraints for inputs
                }
                Gate::Not { input } => {
                    let inp = *input as u32;
                    // out <=> NOT inp
                    // (out OR inp) AND (NOT out OR NOT inp)
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
                    // out <=> in1 AND in2
                    // (NOT out OR in1) AND (NOT out OR in2) AND (out OR NOT in1 OR NOT in2)
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
                    // out <=> in1 OR in2
                    // (out OR NOT in1) AND (out OR NOT in2) AND (NOT out OR in1 OR in2)
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
                    // out <=> in1 XOR in2
                    // (NOT out OR NOT in1 OR NOT in2) AND
                    // (NOT out OR in1 OR in2) AND
                    // (out OR NOT in1 OR in2) AND
                    // (out OR in1 OR NOT in2)
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

        // Constraint for expected output
        if expected_output {
            clauses.push(Clause::unit(Literal::positive(circuit.output as u32)));
        } else {
            clauses.push(Clause::unit(Literal::negative(circuit.output as u32)));
        }

        SolveInstance::new(num_vars, clauses)
    }

    /// Decodes the SAT solution back to input values.
    pub fn decode_inputs(circuit: &Circuit, assignment: &[bool]) -> Vec<bool> {
        assignment[..circuit.num_inputs].to_vec()
    }

    /// Verifies that the inputs produce the expected output.
    pub fn verify(circuit: &Circuit, inputs: &[bool], expected_output: bool) -> bool {
        circuit.evaluate(inputs) == expected_output
    }

    /// Creates a 4-bit ripple-carry adder circuit.
    /// Inputs: a0, a1, a2, a3, b0, b1, b2, b3 (8 inputs)
    /// Output: Whether a + b = expected_sum (for verification)
    pub fn four_bit_adder() -> Circuit {
        let mut circuit = Circuit::new(8);

        // Inputs: a[0..4] at indices 0-3, b[0..4] at indices 4-7

        // Full adder for each bit
        // For bit i: sum_i = a_i XOR b_i XOR c_i, c_{i+1} = (a_i AND b_i) OR (c_i AND (a_i XOR b_i))

        // Bit 0 (half adder, no carry in)
        let a0 = 0;
        let b0 = 4;
        let sum0 = circuit.xor(a0, b0);
        let c0 = circuit.and(a0, b0); // carry out

        // Bit 1
        let a1 = 1;
        let b1 = 5;
        let xor1 = circuit.xor(a1, b1);
        let sum1 = circuit.xor(xor1, c0);
        let and1 = circuit.and(a1, b1);
        let and2 = circuit.and(xor1, c0);
        let c1 = circuit.or(and1, and2);

        // Bit 2
        let a2 = 2;
        let b2 = 6;
        let xor2 = circuit.xor(a2, b2);
        let sum2 = circuit.xor(xor2, c1);
        let and3 = circuit.and(a2, b2);
        let and4 = circuit.and(xor2, c1);
        let c2 = circuit.or(and3, and4);

        // Bit 3
        let a3 = 3;
        let b3 = 7;
        let xor3 = circuit.xor(a3, b3);
        let sum3 = circuit.xor(xor3, c2);
        let and5 = circuit.and(a3, b3);
        let and6 = circuit.and(xor3, c2);
        let c3 = circuit.or(and5, and6); // final carry out

        // The output is sum0, sum1, sum2, sum3, c3 (5 bits)
        // For verification, we'll check if a specific sum is correct
        // Let's verify that 5 + 3 = 8 (0101 + 0011 = 1000)
        // Expected: sum0=0, sum1=0, sum2=0, sum3=1, c3=0

        // Create verification circuit: check if sum equals expected value
        // NOT sum0 AND NOT sum1 AND NOT sum2 AND sum3 AND NOT c3
        let not_sum0 = circuit.not(sum0);
        let not_sum1 = circuit.not(sum1);
        let not_sum2 = circuit.not(sum2);
        let not_c3 = circuit.not(c3);

        let check1 = circuit.and(not_sum0, not_sum1);
        let check2 = circuit.and(check1, not_sum2);
        let check3 = circuit.and(check2, sum3);
        let final_check = circuit.and(check3, not_c3);

        circuit.set_output(final_check);
        circuit
    }

    /// Creates a simple circuit for testing: output = (a AND b) OR (NOT a AND c)
    pub fn simple_circuit() -> Circuit {
        let mut circuit = Circuit::new(3); // inputs: a, b, c

        let and1 = circuit.and(0, 1); // a AND b
        let not_a = circuit.not(0); // NOT a
        let and2 = circuit.and(not_a, 2); // NOT a AND c
        let output = circuit.or(and1, and2); // (a AND b) OR (NOT a AND c)

        circuit.set_output(output);
        circuit
    }
}

// =============================================================================
// Main function - Run all examples
// =============================================================================

/// Attempts to solve an instance with multiple restarts.
/// CLS is an incomplete solver that can get stuck in local minima,
/// so multiple restarts with different random initializations help.
fn solve_with_restarts(instance: SolveInstance, max_restarts: u32) -> xlog_solve::SolveResult {
    let config = SolverConfig::new(20000, 0.1, 0.9, 0.5);
    let solver = Solver::with_config_cpu(config);

    for _ in 0..max_restarts {
        let result = solver.solve(instance.clone());
        if result.is_sat() {
            return result;
        }
    }

    // Return the last result if no solution found
    solver.solve(instance)
}

fn main() {
    println!("=== Real-World SAT Problem Examples ===\n");
    println!("Note: xlog-solve uses a Continuous Local Search (CLS) solver.");
    println!("CLS is an incomplete solver - it may not find solutions for all");
    println!("satisfiable instances, especially hard constraint problems.\n");

    // Create a solver with thorough config for harder problems
    let thorough_config = SolverConfig::thorough();
    let thorough_solver = Solver::with_config_cpu(thorough_config);
    let fast_solver = Solver::with_config_cpu(SolverConfig::fast());

    // -------------------------------------------------------------------------
    // 1. Sudoku Solver
    // -------------------------------------------------------------------------
    println!("1. SUDOKU SOLVER");
    println!("================\n");

    // Use an almost-complete puzzle that the solver can reliably solve
    let puzzle = [
        [5, 3, 4, 6, 7, 8, 9, 1, 0], // Only one cell missing (should be 2)
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9],
    ];
    println!("Input puzzle (one cell missing):");
    sudoku::print_grid(&puzzle);
    println!();

    let instance = sudoku::encode(&puzzle);
    println!(
        "SAT encoding: {} variables, {} clauses",
        instance.num_vars,
        instance.num_clauses()
    );

    let result = thorough_solver.solve(instance);
    println!("Solver result: {:?}", result.status);
    println!(
        "Iterations: {}, Time: {:.2}ms",
        result.stats.iterations,
        result.stats.duration_us as f64 / 1000.0
    );

    if let Some(assignment) = result.assignment() {
        let solution = sudoku::decode(assignment);
        println!("\nSolution:");
        sudoku::print_grid(&solution);

        let valid = sudoku::verify(&solution);
        println!("\nSolution valid: {}", valid);
    } else {
        println!("No solution found");
    }

    // Also demonstrate the full encoding of a harder puzzle
    println!("\n--- Harder puzzle encoding (for reference) ---");
    let hard_puzzle = sudoku::sample_puzzle();
    let hard_instance = sudoku::encode(&hard_puzzle);
    println!(
        "Full Sudoku: {} variables, {} clauses",
        hard_instance.num_vars,
        hard_instance.num_clauses()
    );
    println!("(CLS may not find complete solutions for harder puzzles)\n");

    // -------------------------------------------------------------------------
    // 2. N-Queens Problem
    // -------------------------------------------------------------------------
    println!("2. N-QUEENS PROBLEM");
    println!("===================\n");

    // Start with N=4 which is more tractable for CLS
    let n = 4;
    println!("Solving for N={} (smaller board for reliable solution):", n);
    let instance = nqueens::encode(n);
    println!(
        "SAT encoding: {} variables, {} clauses",
        instance.num_vars,
        instance.num_clauses()
    );

    // Use multiple attempts with restarts
    let result = solve_with_restarts(instance, 5);
    println!("Solver result: {:?}", result.status);

    if let Some(assignment) = result.assignment() {
        let queens = nqueens::decode(n, assignment);
        println!("\nQueen positions: {:?}", queens);
        println!("\nBoard:");
        nqueens::print_board(n, &queens);

        let valid = nqueens::verify(n, &queens);
        println!("\nSolution valid: {}", valid);
    } else {
        println!("No solution found");
    }

    // Also show encoding for N=8
    println!("\n--- N=8 encoding (for reference) ---");
    let n8_instance = nqueens::encode(8);
    println!(
        "N=8 Queens: {} variables, {} clauses",
        n8_instance.num_vars,
        n8_instance.num_clauses()
    );
    println!("(CLS may not find complete solutions for larger boards)\n");

    // -------------------------------------------------------------------------
    // 3. Job Shop Scheduling
    // -------------------------------------------------------------------------
    println!("3. JOB SHOP SCHEDULING");
    println!("======================\n");

    // Use a simpler 2-job, 2-machine problem for reliable solution
    let simple_problem = job_shop::JobShopProblem {
        num_machines: 2,
        horizon: 10,
        jobs: vec![
            job_shop::Job {
                operations: vec![
                    job_shop::Operation { machine: 0, duration: 2 },
                    job_shop::Operation { machine: 1, duration: 1 },
                ],
            },
            job_shop::Job {
                operations: vec![
                    job_shop::Operation { machine: 1, duration: 2 },
                    job_shop::Operation { machine: 0, duration: 1 },
                ],
            },
        ],
    };

    println!(
        "Simple problem: {} jobs, {} machines, horizon={}",
        simple_problem.jobs.len(),
        simple_problem.num_machines,
        simple_problem.horizon
    );

    for (j, job) in simple_problem.jobs.iter().enumerate() {
        print!("  Job {}: ", j);
        for (o, op) in job.operations.iter().enumerate() {
            print!("Op{}(M{},d={}) ", o, op.machine, op.duration);
        }
        println!();
    }
    println!();

    let instance = job_shop::encode(&simple_problem);
    println!(
        "SAT encoding: {} variables, {} clauses",
        instance.num_vars,
        instance.num_clauses()
    );

    let result = solve_with_restarts(instance, 5);
    println!("Solver result: {:?}", result.status);

    if let Some(assignment) = result.assignment() {
        let schedule = job_shop::decode(&simple_problem, assignment);
        println!();
        job_shop::print_schedule(&simple_problem, &schedule);

        let valid = job_shop::verify(&simple_problem, &schedule);
        println!("Solution valid: {}", valid);
    } else {
        println!("No solution found");
    }

    // Also show encoding for larger problem
    println!("\n--- Larger problem encoding (for reference) ---");
    let large_problem = job_shop::sample_problem();
    let large_instance = job_shop::encode(&large_problem);
    println!(
        "5-job, 3-machine problem: {} variables, {} clauses",
        large_instance.num_vars,
        large_instance.num_clauses()
    );
    println!("(CLS may not find complete solutions for larger scheduling problems)\n");

    // -------------------------------------------------------------------------
    // 4. Graph Coloring
    // -------------------------------------------------------------------------
    println!("4. GRAPH COLORING");
    println!("=================\n");

    // Use a simple square graph (4-cycle) which is 2-colorable
    let square_graph = graph_coloring::Graph {
        num_vertices: 4,
        edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
    };
    let num_colors = 2;
    println!("Square graph (4 vertices, 4 edges) with {} colors:", num_colors);
    println!("  Edges: 0-1, 1-2, 2-3, 3-0");
    println!();

    let instance = graph_coloring::encode(&square_graph, num_colors);
    println!(
        "SAT encoding: {} variables, {} clauses",
        instance.num_vars,
        instance.num_clauses()
    );

    let result = solve_with_restarts(instance, 5);
    println!("Solver result: {:?}", result.status);

    if let Some(assignment) = result.assignment() {
        let coloring = graph_coloring::decode(square_graph.num_vertices, num_colors, assignment);
        println!();
        graph_coloring::print_coloring(&coloring);

        let valid = graph_coloring::verify(&square_graph, &coloring);
        println!("Solution valid: {}", valid);
    } else {
        println!("No solution found");
    }

    // Also show Petersen graph encoding
    println!("\n--- Petersen graph encoding (for reference) ---");
    let petersen = graph_coloring::petersen_graph();
    let petersen_instance = graph_coloring::encode(&petersen, 3);
    println!(
        "Petersen graph (10 vertices, 15 edges) with 3 colors: {} variables, {} clauses",
        petersen_instance.num_vars,
        petersen_instance.num_clauses()
    );
    println!("(CLS may not find complete solutions for complex graphs)\n");

    // -------------------------------------------------------------------------
    // 5. Circuit SAT
    // -------------------------------------------------------------------------
    println!("5. CIRCUIT SAT");
    println!("==============\n");

    // Test a simple circuit: output = (a AND b) OR (NOT a AND c)
    println!("Circuit 1: output = (a AND b) OR (NOT a AND c)");
    let simple_circuit = circuit_sat::simple_circuit();

    let instance = circuit_sat::encode(&simple_circuit, true);
    println!(
        "SAT encoding: {} variables, {} clauses",
        instance.num_vars,
        instance.num_clauses()
    );

    let result = fast_solver.solve(instance);
    println!("Solver result: {:?}", result.status);

    if let Some(assignment) = result.assignment() {
        let inputs = circuit_sat::decode_inputs(&simple_circuit, assignment);
        println!("Found inputs: a={}, b={}, c={}", inputs[0], inputs[1], inputs[2]);

        let output = simple_circuit.evaluate(&inputs);
        println!("Circuit output: {}", output);

        let valid = circuit_sat::verify(&simple_circuit, &inputs, true);
        println!("Verification: {}", valid);
    }
    println!();

    // Test an AND gate circuit
    println!("Circuit 2: output = a AND b (finding inputs where output=true)");
    let mut and_circuit = circuit_sat::Circuit::new(2);
    let and_out = and_circuit.and(0, 1);
    and_circuit.set_output(and_out);

    let instance = circuit_sat::encode(&and_circuit, true);
    println!(
        "SAT encoding: {} variables, {} clauses",
        instance.num_vars,
        instance.num_clauses()
    );

    let result = fast_solver.solve(instance);
    println!("Solver result: {:?}", result.status);

    if let Some(assignment) = result.assignment() {
        let inputs = circuit_sat::decode_inputs(&and_circuit, assignment);
        println!("Found inputs: a={}, b={}", inputs[0], inputs[1]);

        let output = and_circuit.evaluate(&inputs);
        println!("Circuit output: {} (both must be true)", output);

        let valid = circuit_sat::verify(&and_circuit, &inputs, true);
        println!("Verification: {}", valid);
    }
    println!();

    // Test a half-adder carry output
    println!("Circuit 3: Half-adder carry = a AND b");
    let mut half_adder = circuit_sat::Circuit::new(2);
    let carry = half_adder.and(0, 1);
    half_adder.set_output(carry);

    let instance = circuit_sat::encode(&half_adder, true);
    let result = fast_solver.solve(instance);

    if let Some(assignment) = result.assignment() {
        let inputs = circuit_sat::decode_inputs(&half_adder, assignment);
        println!("For carry=true: a={}, b={}", inputs[0], inputs[1]);
        assert!(inputs[0] && inputs[1], "Both inputs must be true for carry");
        println!("Verification passed: carry is true when both inputs are true");
    }
    println!();

    // Show 4-bit adder encoding for reference
    println!("--- 4-bit adder encoding (for reference) ---");
    let adder_circuit = circuit_sat::four_bit_adder();
    let adder_instance = circuit_sat::encode(&adder_circuit, true);
    println!(
        "4-bit adder: {} gates, {} variables, {} clauses",
        adder_circuit.gates.len(),
        adder_instance.num_vars,
        adder_instance.num_clauses()
    );
    println!("(CLS may not find solutions for complex circuits)");

    println!("\n=== All examples completed ===");
}
