# Solver Services (xlog-solve)

This document describes XLOG's GPU-native SAT/MaxSAT solver based on Continuous Local Search (CLS), inspired by FastFourierSAT.

## Overview

`xlog-solve` provides solver services for satisfiability and optimization problems. It treats SAT as continuous optimization, enabling GPU parallelism.

## Core Concept

Traditional SAT solvers work with discrete boolean assignments. CLS instead:

1. Represents variables as continuous values in [0, 1]
2. Expresses clauses as differentiable loss functions
3. Uses gradient-like updates computed in parallel on GPU
4. Discretizes to boolean when a solution is found

## Data Structures

### Solve Instance

```rust
pub struct SolveInstance {
    pub num_vars: u32,
    pub clauses: Vec<Clause>,           // CNF clauses
    pub weights: Option<Vec<f64>>,      // For MaxSAT/weighted
    pub objective: Objective,           // SAT, MaxSAT, or MinUnsat
}

pub struct Clause {
    pub literals: Vec<Literal>,         // Variable + polarity
}

pub struct Literal {
    pub var: u32,
    pub negated: bool,
}
```

### GPU Solver State

```rust
pub struct SolverState {
    pub assignments: CudaSlice<f32>,    // Continuous [0,1] per variable
    pub velocities: CudaSlice<f32>,     // Momentum for updates
    pub clause_sat: CudaSlice<f32>,     // Satisfaction degree per clause
    pub gradients: CudaSlice<f32>,      // dL/d(var)
}
```

### Configuration

```rust
pub struct SolverConfig {
    pub max_iterations: u32,            // Default: 10,000
    pub learning_rate: f32,             // Step size
    pub momentum: f32,                  // Velocity decay
    pub discretize_threshold: f32,      // When to snap to 0/1
}
```

## Loss Function

Each clause contributes to the loss:

```
clause_loss(c) = prod(1 - lit_value(l)) for l in c
```

Where `lit_value(l) = assignments[var]` if positive, or `1 - assignments[var]` if negated.

- Loss = 0 when clause is satisfied (at least one literal is 1)
- Loss > 0 when clause is violated

Total loss = sum of clause losses (for SAT) or weighted sum (for MaxSAT).

## GPU Kernels

The solver uses three phases per iteration:

### Phase 1: Evaluate Clause Satisfaction

```cuda
__global__ void cls_evaluate_clauses(
    const float* assignments,      // [num_vars]
    const int32_t* clause_lits,    // Packed literals
    const uint32_t* clause_offsets,// Start of each clause
    uint32_t num_clauses,
    float* clause_sat              // Output: satisfaction [0,1]
) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_clauses) return;

    float product = 1.0f;
    for (uint32_t i = clause_offsets[c]; i < clause_offsets[c+1]; i++) {
        int32_t lit = clause_lits[i];
        uint32_t var = abs(lit) - 1;
        float val = (lit > 0) ? assignments[var] : 1.0f - assignments[var];
        product *= (1.0f - val);
    }
    clause_sat[c] = product;
}
```

### Phase 2: Compute Gradients

```cuda
__global__ void cls_compute_gradients(
    const float* assignments,
    const float* clause_sat,
    const int32_t* var_clauses,    // Which clauses mention var
    const uint32_t* var_offsets,
    uint32_t num_vars,
    float* gradients               // Output: dL/d(var)
) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vars) return;

    float grad = 0.0f;
    for (uint32_t i = var_offsets[v]; i < var_offsets[v+1]; i++) {
        // Compute partial derivative contribution from each clause
        grad += partial_derivative(v, var_clauses[i], ...);
    }
    gradients[v] = grad;
}
```

### Phase 3: Update Assignments

```cuda
__global__ void cls_update_assignments(
    float* assignments,
    float* velocities,
    const float* gradients,
    float learning_rate,
    float momentum,
    uint32_t num_vars
) {
    uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vars) return;

    velocities[v] = momentum * velocities[v] - learning_rate * gradients[v];
    assignments[v] = clamp(assignments[v] + velocities[v], 0.0f, 1.0f);
}
```

## Proof Generation

The solver produces verifiable proofs:

```rust
pub enum SolveProof {
    /// SAT: satisfying assignment is the proof
    Satisfying {
        assignment: Vec<bool>,
        checksum: u64,
    },

    /// UNSAT: resolution proof
    Unsatisfiable {
        learned_clauses: Vec<Clause>,
        resolution_chain: Vec<ResolutionStep>,
        checksum: u64,
    },

    /// Optimum: assignment + bound proof
    Optimal {
        assignment: Vec<bool>,
        objective_value: f64,
        bound_certificate: BoundCert,
    },

    /// Approximate: best-effort with quality metrics
    Approximate {
        assignment: Vec<bool>,
        satisfied_clauses: u32,
        total_clauses: u32,
        iterations: u32,
    },
}
```

### GPU-Accelerated Verification

```rust
pub fn verify_proof_gpu(
    instance: &SolveInstance,
    proof: &SolveProof,
    provider: &CudaKernelProvider,
) -> Result<VerifyResult> {
    match proof {
        SolveProof::Satisfying { assignment, checksum } => {
            let sat_count = provider.count_satisfied_clauses(
                &instance.clauses,
                assignment,
            )?;
            Ok(sat_count == instance.clauses.len())
        }
        // ... other proof types
    }
}
```

## Solver API

```rust
pub struct Solver {
    provider: Arc<CudaKernelProvider>,
    stats: Arc<StatsManager>,
    config: SolverConfig,
}

impl Solver {
    /// Solve a SAT/MaxSAT instance
    pub fn solve(&self, instance: SolveInstance) -> Result<SolveResult>;

    /// Solve with timeout
    pub fn solve_with_timeout(
        &self,
        instance: SolveInstance,
        timeout: Duration,
    ) -> Result<SolveResult>;

    /// Batch solve multiple instances
    pub fn solve_batch(
        &self,
        instances: Vec<SolveInstance>,
    ) -> Result<Vec<SolveResult>>;

    /// Incremental solving (add clauses to existing state)
    pub fn solve_incremental(
        &self,
        state: &mut IncrementalState,
        new_clauses: Vec<Clause>,
    ) -> Result<SolveResult>;
}

pub struct SolveResult {
    pub status: SolveStatus,
    pub proof: SolveProof,
    pub stats: SolveStats,
}

pub enum SolveStatus {
    Sat,
    Unsat,
    Unknown,
    Optimal(f64),
}
```

## Integration with XLOG

### xlog-prob Integration

The solver supports knowledge compilation workflows:

```
PIR → CNF → Solver (preprocessing) → D4 (compilation) → XGCF
```

Solver preprocessing can simplify CNF before compilation.

### xlog-elp Integration (Planned)

For epistemic logic, the solver handles:
- Guess validation (SAT checks for world-view candidates)
- Propagation (unit propagation, pure literal elimination)
- Test phase (brave/cautious consequence checking)

## Performance Characteristics

| Metric | Target |
|--------|--------|
| Variables | 10,000+ |
| Clauses | 100,000+ |
| Iterations/sec | 1,000+ |
| GPU utilization | >80% |

## Limitations

Current implementation:
- CLS is incomplete (may not find UNSAT proof)
- Best for satisfiable instances
- UNSAT detection requires fallback to CDCL (planned)

## See Also

- [Probabilistic Tier](xlog-prob.md) — Uses solver for preprocessing
- [GPU Execution](gpu-execution.md) — Shared GPU infrastructure
