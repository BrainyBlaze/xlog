# GPU-Native Knowledge Compilation Design (GPU D4 + GPU CDCL Verifier)
**Date:** January 22, 2026  
**Status:** Revised for 100% GPU-native design  
**Target Release:** v0.5.0

---

## Executive Summary

This design defines a **100% GPU-native knowledge compilation pipeline** for XLOG with **zero CPU data transfers**. The runtime path is **single‑route**:

1. **GPU D4 compiler** (BFS-parallelized) builds an exact d-DNNF.
2. **GPU CDCL verifier** proves equivalence by checking **two UNSAT queries**:
   `UNSAT(φ ∧ ¬C)` and `UNSAT(C ∧ ¬φ)`.
3. **GPU XGCF** is emitted and executed by existing GPU kernels.

There is **no CPU fallback**, no tensor path, and no randomized validation. Correctness is guaranteed by
the GPU verifier. The CPU is used only for **control-plane kernel launches** and stream management, not
for **data-plane** movement of CNF/circuit/weights/gradients.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)  
2. [GPU D4 Compiler](#2-gpu-d4-compiler)  
3. [GPU CDCL Equivalence Verifier](#3-gpu-cdcl-equivalence-verifier)  
4. [GPU-Resident Cache & Memory Model](#4-gpu-resident-cache--memory-model)  
5. [Integration Points](#5-integration-points)  
6. [Testing Strategy (GPU-Only)](#6-testing-strategy-gpu-only)  
7. [Performance Targets](#7-performance-targets)  
8. [Implementation Roadmap](#8-implementation-roadmap)  
9. [Risks & Mitigations](#9-risks--mitigations)

---

## 1. Architecture Overview

### 1.1 Goals & Constraints

- **100% GPU-native**: all CNF, circuits, and verification data stay on device.
- **Zero CPU transfers (data-plane)**: no device↔host copies of CNF/circuit/weights/gradients in the
  steady-state compilation+evaluation path.
- **Soundness**: correctness proven by a GPU CDCL equivalence check.
- **Compatibility**: output is XGCF, evaluated by existing GPU kernels.

**Definition: “zero CPU transfers” in XLOG**
- Inputs (CNF, weights, neural tensors) are provided as **device-resident buffers** (e.g., via DLPack).
- Outputs (logZ, grads) are produced as **device-resident buffers** and can be
  exported via DLPack.
- The host is allowed to launch kernels and manage CUDA streams, but it should not need to memcpy
  the data-plane arrays listed above.

### 1.2 End-to-End Flow

```
GPU-resident CNF (φ)
        │
        ▼
GPU D4 Compiler (BFS parallel)
        │  produces candidate circuit C (d-DNNF)
        ▼
GPU CDCL Verifier
  check UNSAT(φ ∧ ¬C) and UNSAT(C ∧ ¬φ)
        │
   UNSAT? ✔
        ▼
GPU XGCF (device-resident)
        │
        ▼
Existing GPU evaluation kernels
```

**Key property:** The verifier is complete. If both checks are UNSAT, the circuit is equivalent and safe.
If either check is SAT, the circuit is rejected.

For the **GPU D4 exact path**, a SAT result indicates a compiler bug; production behavior is **fail-fast**
via a GPU trap / CUDA error (no silent fallback).

---

## 2. GPU D4 Compiler

### 2.1 BFS Parallelization

D4’s recursive search is transformed into a **breadth‑first GPU work queue**. Each BFS level processes a batch of CNF subproblems in parallel:

```
Level 0:   [φ]
Level 1:   [φ|x]   [φ|¬x]
Level 2:   [φ|x,y] [φ|x,¬y] [φ|¬x,y] [φ|¬x,¬y] ...
```

### 2.2 GPU Data Structures

```rust
struct GpuCnf {
    // Host-known capacities (buffers are allocated to these sizes).
    var_cap: u32,
    clause_cap: u32,
    lit_cap: u32,

    // Device-resident exact counts (len = 1 each).
    num_vars: DeviceSlice<u32>,
    num_clauses: DeviceSlice<u32>,
    num_lits: DeviceSlice<u32>,

    // CSR buffers sized by capacity.
    clause_offsets: DeviceSlice<u32>, // len = clause_cap + 1
    literals: DeviceSlice<i32>,       // len = lit_cap, signed DIMACS: ±var_id (1-based)
}

/// Per-work-item assignments live in a pooled buffer to avoid device malloc.
/// The concrete encoding may be **dense tri-state bytes** or **compressed bitsets** (see 2.2.1).
/// Layout shown here is the dense form: `assignments[work_id * stride + var]`,
/// with `var` 1-indexed to match DIMACS.
struct GpuAssignmentPool {
    assignments: DeviceSlice<u8>, // 0=unassigned,1=true,2=false
    stride: u32,                  // = num_vars + 1
}

struct D4WorkItem {
    subproblem_id: u32,
    parent_node: u32,
    branch: u8,
    depth: u16,
    assignment_offset: u32, // base = subproblem_id * stride
}

struct GpuWorkQueue {
    items: DeviceSlice<D4WorkItem>,
    capacity: u32,
    /// Device-resident counter (implemented as a 1-element device slice updated via atomics).
    size: DeviceSlice<u32>,
}

struct GpuComponentCache {
    keys: DeviceSlice<u64>,
    values: DeviceSlice<u32>,
    table_size: u32,
}

struct GpuCircuitBuilder {
    node_type: DeviceSlice<u8>,
    child_offsets: DeviceSlice<u32>,
    child_indices: DeviceSlice<u32>,
    lit: DeviceSlice<i32>,
    decision_var: DeviceSlice<u32>,
    decision_child_false: DeviceSlice<u32>,
    decision_child_true: DeviceSlice<u32>,
}
```

**Type note:** In this document, `DeviceSlice<T>` is shorthand for a GPU-resident,
owned, tracked buffer such as `xlog_cuda::memory::TrackedCudaSlice<T>`.

#### 2.2.1 Assignment Representation (Robust + Bandwidth-Conscious)

The naive dense layout `assignments[subproblem_id][var]` is simple and fast for lookups, but it can be
too large to copy/initialize at scale. The GPU D4 design therefore standardizes **two formats**:

1) **Dense tri-state bytes (simple, fast)**  
`u8` per var: `0=unassigned, 1=true, 2=false`. Best for small/medium `num_vars` and small frontiers.

2) **Compressed bitsets (recommended default)**  
Two bitsets per subproblem: `true_bits[var]=1` and `false_bits[var]=1`. Unassigned iff both are 0.
This reduces assignment memory by ~4x compared to `u8` and enables fast set/copy operations.

To avoid O(num_vars) device bandwidth per branch, the compiler uses a **frontier+worker** pattern:
- Expand BFS only to a fixed depth `frontier_depth` (small, e.g., 6–10) to create enough independent work.
- Store each frontier subproblem’s assignment in global memory (compressed bitsets).
- For each frontier item, run **local DFS D4** inside one GPU block (shared-memory stack + propagation),
  emitting circuit nodes into the global builder.

This keeps parallelism (many frontier items) while avoiding repeated dense assignment copies at deeper levels.

### 2.3 Core Kernels (GPU)

- **Unit Propagation**: simplify each CNF instance.
- **Component Decomposition**: union‑find over clauses sharing unassigned variables.
- **Variable Selection**: GPU heuristic (e.g., VSIDS approximation).
- **CNF Restriction**: branch on variable; emit children into `next_queue`.
- **Circuit Assembly**: write nodes into device‑resident builder.

### 2.4 Output

The compiler produces **device‑resident XGCF** (not host XGCF). A new constructor will be required:

```
GpuXgcf::from_device(builder: GpuCircuitBuilder, layout: GpuCircuitLayout)
```

This avoids CPU copies and keeps the circuit resident for evaluation.

### 2.5 GPU Post-Processing Requirements (Smoothness + Free Vars)

The current CPU path performs two important correctness steps after compilation:

1) **Smoothness for random variables**  
`crates/xlog-prob/src/xgcf.rs` implements `Xgcf::smooth_random_vars(&is_random_var)` to ensure
the circuit is smooth w.r.t. random variables. This is required for correct WMC evaluation and
gradients when evidence/queries force variables along different OR/DECISION branches.

**GPU-native requirement:** either:
- GPU D4 emits circuits that are already smooth w.r.t. random vars, or
- a GPU smoothing pass exists and is applied before evaluation (and ideally before verification).

2) **Free-variable correction factor**  
The current exact evaluator computes `free_vars` (vars absent from the compiled circuit and absent
from CNF clauses) and adds `logsumexp2(t, f)` for each such var in `eval_log_z`.

**GPU-native requirement:** compute and apply the same correction factor on GPU, or enforce a
compiler invariant that every CNF variable appears in the circuit (making `free_vars` empty).

#### 2.5.1 GPU Smoothing Pass (Port of `Xgcf::smooth_random_vars`)

XLOG’s reference smoother (`crates/xlog-prob/src/xgcf.rs`) works by:
1) computing each node’s **random support set** (which random vars occur below it), and
2) wrapping OR/DECISION children with ANDs that include missing vars as **tautology decisions**
   (`Decision(var, Const1, Const1)`), so every branch mentions the same random vars.

GPU-native implementation strategy:
- Represent random supports as **bitsets over random vars only** (not all CNF vars).
  - Build a device table `random_var_list[]` (CNF var ids) and `random_var_to_bit[var] -> bit_index`.
- Compute support bitsets bottom-up in level order:
  - `Lit(var)` / `Decision(var, ..)` add the bit for `var` iff `is_random_var[var]`.
  - `And/Or` take bitwise OR of children supports.
- Smooth:
  - For each OR/DECISION node, compute `missing = parent_support \ child_support` and, if non-empty,
    wrap the child in `And(child, tautology(var_1), ..., tautology(var_k))`.

To keep the pass linear-time and allocation-free:
- Pre-create one tautology node per random var in a **tautology table** (device-resident), so wrapping does
  not require hashing/interning.
- Use a two-pass emitter:
  - pass A: count how many wrapper AND nodes each child needs (bitcount of missing), prefix-sum to size buffers
  - pass B: emit the augmented circuit into the `GpuCircuitBuilder` pools.

#### 2.5.2 GPU Free-Var Correction (Matches `exact.rs`)

After smoothing, compute the exact evaluator’s “free var” correction on device:

- Compute `vars_in_clauses[var]` from `GpuCnf` literals (mark abs(lit)).
- Compute `vars_in_circuit[var]` from XGCF `Lit` and `Decision` nodes.
- `free_var[var] = !vars_in_clauses[var] && !vars_in_circuit[var]`.

Then apply:
- `logZ += sum_{free var v} logsumexp2(log_true[v], log_false[v])`
- `grad_true[v] += softmax_true(log_true[v], log_false[v])`
- `grad_false[v] += softmax_false(log_true[v], log_false[v])`

This is implemented as a GPU reduction over `v=1..max_var` with deterministic summation (pairwise reduction),
so correctness does not depend on host post-processing.

---

## 3. GPU CDCL Equivalence Verifier

### 3.1 Verification Problem

We must prove `φ ≡ C`.

**Operational form:** implement equivalence checking as **two UNSAT queries**:

```
SAT( φ ∧ ¬C ) == UNSAT   and   SAT( C ∧ ¬φ ) == UNSAT
```

This works even when the CNF encoding does not assert a root; `φ` denotes the satisfaction predicate
of the whole CNF constraint set.

### 3.2 GPU CDCL Solver Requirements

The verifier is a **complete SAT solver** (unlike CLS in `crates/xlog-solve`, which is incomplete and CPU‑only). It must:

- Implement watched literals and clause learning.
- Perform conflict analysis (1‑UIP).
- Support restart and clause database management.
- For **SAT**, return a device‑resident assignment and **validate it on GPU** (all clauses satisfied).
- For **UNSAT**, return a **GPU‑checkable certificate** and **validate it on GPU** before reporting UNSAT.

#### 3.2.5 UNSAT Certificate (Mandatory): Resolution‑Trace Proof

XLOG requires a **device‑side UNSAT check** with **no CPU proof checking**. DRAT/DRUP checking is not a good fit for GPU
as a production baseline, so the verifier emits a **resolution‑trace proof** that can be replayed on GPU:

- Each learned clause `L_i` stores:
  - `conflict_clause_id` (base or earlier learned clause id), and
  - `steps[] = [(var, reason_clause_id), ...]` corresponding to the conflict analysis resolution sequence.
- The GPU proof checker replays resolution:
  - Start from `conflict_clause_id`,
  - for each step `(var, reason_clause_id)`, resolve on `var` with `reason_clause_id`,
  - and verify the final resolvent equals the stored learned clause `L_i` (set‑equality, no tautologies).

The **UNSAT certificate** is accepted only if the **last learned clause is empty** and all stored traces check on GPU.

#### 3.2.1 Execution Model (GPU)

Equivalence verification runs only **a few** SAT instances per compilation (two checks), but each instance
can be large. The design therefore favors **per-instance locality** over massive cross-instance batching:

- **One thread-block per SAT instance** (or one block per component, if the instance decomposes).
- A **persistent CDCL loop** inside the block (no host round-trips, no mid-solve device↔host copies).
- A warp-synchronous implementation for the hot operations (BCP scan, reason inspection) to control divergence.

This matches XLOG’s design goal: keep compilation+verification on GPU even when the control flow is irregular.

#### 3.2.2 Clause DB Layout (GPU-Friendly)

Use a CSR-like clause database that mirrors how XLOG already stores CNFs/circuits on GPU:

- `clause_offsets[clause_id]..clause_offsets[clause_id+1]` indexes into `literals[]`.
- Literals are stored as signed `i32` DIMACS (`+v` / `-v`), with `0` unused.
- Learned clauses live in a separate arena with the same layout and an eviction policy (e.g., LBD + activity).

Watched literals are represented as:

- `watch_pos[clause_id][0..2)` = indices into the clause’s literal slice (two watched positions)
- `watch_lists[lit]` = list of clause_ids currently watching `lit` (device-side adjacency)

To keep the solver deterministic and memory-bounded, all arenas are fixed-capacity and allocated from
`GpuMemoryManager` pools; overflow is a hard error.

#### 3.2.3 Determinism Contract

XLOG’s GPU tier has explicit determinism testing (`crates/xlog-cuda-tests`, e.g. C15). The CDCL verifier must:

- Use deterministic tie-breaking (literal selection, clause bump order, restart schedule).
- Avoid nondeterministic “first writer wins” patterns for learned clause insertion.
- When SAT is requested, return a deterministic device-resident assignment (useful for debugging solver/compiler bugs).

#### 3.2.4 SAT Subsystem (`crates/xlog-solve`) Status

As of **January 25, 2026**, `crates/xlog-solve` provides:

- A **CPU Continuous Local Search (CLS)** solver (heuristic, incomplete) that can be used only as an *optional*
  accelerator (e.g., SAT witness finding, or future CDCL seeding).
- A **GPU-native CDCL solver** (complete SAT/UNSAT) used as the **production verifier**.

The GPU CDCL verifier is implemented in `kernels/sat.cu` (compiled to `kernels/sat.ptx`), loaded by
`crates/xlog-cuda`, and exposed as `xlog_solve::GpuCdclSolver` for verifier integrations (e.g., equivalence checking
in `crates/xlog-prob`).

### 3.3 GPU-Resident Equivalence Construction

We construct the equivalence-check SAT instances on GPU:

1. Encode `C` into CNF (device‑side Tseitin).
2. Solve `SAT(φ ∧ ¬C)` with GPU CDCL.
3. Solve `SAT(C ∧ ¬φ)` with GPU CDCL.

All of this happens **on device** with no host transfer.

**Verifier-grade constraint:** the equivalence path performs **zero device→host copies**, including scalar status/size reads.
Exact CNF sizes are computed on GPU into device-resident scalars; buffers are allocated to host-known capacities and validated on device.

#### 3.3.1 Encoding details (linear size, GPU-friendly)

Let `φ` be the original CNF (CSR clause list). Let `C` be the compiled circuit with root node id `c_root`.
Let `v_root` be the **Tseitin variable id** introduced for the circuit root when encoding `C` into CNF.

- **CNF(C):** encode each circuit node with a Tseitin variable and local clauses.
  Add a unit clause to force the desired polarity of the root:
  - for `¬C`: add `(¬v_root)`
  - for `C`: add `(v_root)`

- **CNF(¬φ):** `¬φ` is the disjunction of unsatisfied clauses:
  `¬(∧ clause_j) = ∨ (¬clause_j)`.
  Encode this in CNF by introducing clause-satisfaction variables `s_j`:

  - For each clause `j` with literals `(l1 ∨ ... ∨ lk)`:
    - `(¬s_j ∨ l1 ∨ ... ∨ lk)`     // `s_j -> clause_j`
    - For each literal `li`: `(s_j ∨ ¬li)`   // `li -> s_j`
  - Add one clause: `(¬s_1 ∨ ¬s_2 ∨ ... ∨ ¬s_m)`  // at least one clause is unsatisfied

This yields a **linear-size** encoding in total literals of `φ` and nodes of `C`.

#### 3.3.2 Incremental GPU CDCL (recommended)

For performance, run both checks using the same GPU-resident solver state:
- load shared clauses for `φ` and `CNF(C)` once,
- then solve with different **assumptions** (root polarity and the `¬φ` clause),
so learned clauses are reused across both checks.

### 3.4 Safety Contract

- **UNSAT** → accept circuit.
- **SAT** → reject circuit. For the exact GPU D4 path, treat this as a **compiler bug** and fail-fast (GPU trap / CUDA error).
- **UNKNOWN** is not allowed in production; GPU CDCL must be complete.

**Device-side validation rules (mandatory):**
- Every **SAT** result must pass an on‑GPU model check for the solved CNF.
- Every **UNSAT** result must pass an on‑GPU proof check (empty‑clause certificate).

This is the sole correctness guarantee in the design.

---

## 4. GPU-Resident Cache & Memory Model

### 4.1 Circuit Cache (Device-Resident)

Cache keyed by CNF hash (computed on GPU). The cache stores:

- `GpuXgcf` circuits
- Optional metadata (compile time, node count, verification status)

Eviction uses GPU LRU with a fixed memory budget.

### 4.2 Memory Pools

All allocations use GPU memory pools to avoid expensive device malloc:

- Work queues
- Circuit builders
- Clause DB for CDCL
- Temporary buffers for CNF restriction

### 4.3 Deterministic Budgeting & Overflow Semantics (No Fallback)

Because the design forbids CPU fallback, every memory limit must have a deterministic outcome:

- All major arenas are **pre-sized** from `GpuCompileConfig` and the device `MemoryBudget`.
- If any arena overflows (frontier queue, circuit node pool, learned clause pool, PIR intern table):
  - return a hard `XlogError::Kernel`/`XlogError::Compilation` equivalent (no partial results),
  - optionally preserve a small device-resident “failure record” (reason + counters) for debugging.

This makes failure modes reproducible and prevents silent correctness degradation.

---

## 5. Integration Points

### 5.1 Exact Inference Path

Replace the CPU D4 call in `crates/xlog-prob/src/exact.rs:538`:

```rust
// OLD: CPU D4 + file IO
// d4.compile_ddnnf(&cnf_path, &out_path)?;

// NEW: GPU-native compile + verify (device-resident)
let gpu_circuit = compile_gpu_d4_and_verify(&gpu_cnf, provider.as_ref(), &config)?;
```

### 5.2 GPU CNF Generation

To maintain **zero CPU transfers**, CNF creation must be GPU-native:

- Either move PIR → CNF encoding to GPU, or
- Ensure PIR is already GPU‑resident (e.g., via GPU datalog execution).

**Runtime requirement:** `GpuCnf` is built on device, not uploaded from host.

#### 5.2.0 GPU Provenance → GPU PIR (How PIR Becomes Device-Resident)

Today, provenance extraction (`crates/xlog-prob/src/provenance.rs`) is CPU-only and uses a `PirBuilder`
that performs hash-consing + simplifications for `And/Or/Decision/Lit/NegLit`.

For the GPU-native design, provenance becomes a **first-class GPU column** in the relational executor:

- Each derived tuple carries an extra `prov_id` column (a `PirNodeId::as_u32()`).
- Relational operators update `prov_id` using the provenance semiring:
  - **Join (rule body conjunction):** `prov_out = AND(prov_left, prov_right, ...)`
  - **Union of derivations:** `prov_out = OR(prov_existing, prov_new)` (when dedup detects duplicates)
  - **Deterministic facts:** `prov = Const(true)`
  - **Probabilistic facts:** `prov = Lit(leaf_id)`
  - **Annotated disjunctions:** `prov = AND(choice_lits...)` where each `choice_lit` is a `Decision`
    node `Decision(var, ConstF, ConstT)` or `Decision(var, ConstT, ConstF)` matching
    `PirBuilder::choice_lit` and `compile_annotated_disjunction`.
  - **Negation:** use the CPU semantics exactly:
    `not(A)` is implemented as `negate_provenance( OR(all matching provs for A) )`,
    with De Morgan pushing to leaves (`Lit ↔ NegLit`, `And ↔ Or`, and branchwise `Decision` negation).

**GPU hash-consing (device-side interning)**

To keep PIR canonical and deterministic, every batch of newly created PIR nodes is interned on GPU:

1) Build node keys in SoA form (`tag`, `payload`, `children range`) with children already sorted/deduped.
2) Hash keys (stable hash) and sort by `(hash, key)` using existing GPU sort/dedup primitives.
3) Assign new node ids for unique keys via prefix-sum and write them into the `GpuPirGraph` pools.
4) Scatter the resulting node ids back to the `prov_id` output column.

This makes PIR construction look like another relational operator: **pure GPU, batched, deterministic**.

#### 5.2.1 GPU PIR layout aligned with `pir::PirNode`

The current PIR is an enum in `crates/xlog-prob/src/pir.rs`:

```
PirNode::Const(bool)
PirNode::Lit { leaf: LeafId }
PirNode::NegLit { leaf: LeafId }
PirNode::And { children: Vec<PirNodeId> }
PirNode::Or { children: Vec<PirNodeId> }
PirNode::Decision { var: ChoiceVarId, child_false, child_true }
```

The GPU mirror must preserve this exact shape. Use a **structure‑of‑arrays (SoA)** layout so kernels can read fields without indirection.

Proposed device layout (new module: `crates/xlog-prob/src/compilation/gpu_pir.rs`):

```rust
/// Node type tags matching `PirNode` variants.
pub const PIR_CONST: u8 = 0;
pub const PIR_LIT: u8 = 1;
pub const PIR_NEG_LIT: u8 = 2;
pub const PIR_AND: u8 = 3;
pub const PIR_OR: u8 = 4;
pub const PIR_DECISION: u8 = 5;

/// GPU-resident PIR graph (device-side mirror of `pir::PirGraph`).
pub struct GpuPirGraph {
    pub node_type: DeviceSlice<u8>,        // length = num_nodes
    pub child_offsets: DeviceSlice<u32>,   // CSR for AND/OR children, len = num_nodes + 1
    pub children: DeviceSlice<u32>,        // child node ids (PirNodeId::as_u32)
    pub leaf_id: DeviceSlice<u32>,         // LeafId::as_u32 for LIT/NEG_LIT, else 0
    pub decision_var: DeviceSlice<u32>,    // ChoiceVarId::as_u32 for DECISION, else 0
    pub decision_child_false: DeviceSlice<u32>,
    pub decision_child_true: DeviceSlice<u32>,
}

/// GPU-resident PIR root list.
pub struct GpuPirRoots {
    pub roots: DeviceSlice<u32>, // PirNodeId::as_u32
}
```

**Invariants:**
- `child_offsets[i]..child_offsets[i+1]` is valid only for AND/OR nodes.
- `leaf_id[i]` is non‑zero only for LIT/NEG_LIT nodes.
- `decision_*[i]` is non‑zero only for DECISION nodes.

**Mapping table (host PIR → GPU SoA):**

| `PirNode` variant | `node_type[i]` | `children` | `child_offsets` | `leaf_id[i]` | `decision_var[i]` | `decision_child_false[i]` | `decision_child_true[i]` |
|---|---|---|---|---|---|---|---|
| `Const(true/false)` | `PIR_CONST` | none | `child_offsets[i]=child_offsets[i+1]` | 0 | 0 | 0 | 0 |
| `Lit { leaf }` | `PIR_LIT` | none | `child_offsets[i]=child_offsets[i+1]` | `leaf.as_u32()` | 0 | 0 | 0 |
| `NegLit { leaf }` | `PIR_NEG_LIT` | none | `child_offsets[i]=child_offsets[i+1]` | `leaf.as_u32()` | 0 | 0 | 0 |
| `And { children }` | `PIR_AND` | CSR range | `child_offsets[i]..child_offsets[i+1]` | 0 | 0 | 0 | 0 |
| `Or { children }` | `PIR_OR` | CSR range | `child_offsets[i]..child_offsets[i+1]` | 0 | 0 | 0 | 0 |
| `Decision { var, child_false, child_true }` | `PIR_DECISION` | none | `child_offsets[i]=child_offsets[i+1]` | 0 | `var.as_u32()` | `child_false.as_u32()` | `child_true.as_u32()` |

#### 5.2.2 GPU CNF API (tied to current encoder)

The host CNF encoder (`encode_cnf(&PirGraph, roots)`) is re‑implemented on GPU to eliminate host transfers.

```rust
/// GPU-resident CNF (device-side mirror of `cnf::CnfFormula`).
pub struct GpuCnf {
    pub var_cap: u32,
    pub clause_cap: u32,
    pub lit_cap: u32,

    pub num_vars: DeviceSlice<u32>,    // len=1
    pub num_clauses: DeviceSlice<u32>, // len=1
    pub num_lits: DeviceSlice<u32>,    // len=1

    pub clause_offsets: DeviceSlice<u32>, // len = clause_cap + 1
    pub literals: DeviceSlice<i32>,       // len = lit_cap, signed DIMACS
}

/// GPU-resident CNF encoding bundle (CNF + var tables).
pub struct GpuCnfEncoding {
    pub cnf: GpuCnf,
    pub vars: GpuCnfVarTables,
}

/// Encode CNF directly on GPU from a GPU PIR graph.
pub fn encode_cnf_gpu(
    pir: &GpuPirGraph,
    roots: &GpuPirRoots,
    provider: &CudaKernelProvider,
) -> Result<GpuCnfEncoding>;
```

**Notes:**
- `encode_cnf_gpu` must perform **Tseitin encoding** for non‑literal nodes exactly as the CPU version does.
- The GPU encoder produces the same variable numbering semantics (`DIMACS 1‑indexed`).
- This removes all host file IO (`in.cnf`, `out.nnf`) from the exact path.

**GPU-side Tseitin encoding outline (mirrors `encode_cnf`):**

```
// Inputs: GpuPirGraph + roots
// Outputs: GpuCnf (CSR), plus node_var/leaf_var/choice_var tables on device

1) Assign CNF variable IDs:
   a) leaf_var[leaf_id] = next_var++
   b) choice_var[var_id] = next_var++
   c) node_var[node_id] =
        - leaf_var[leaf] for LIT
        - new var for NEG_LIT (constrained to !leaf_var)
        - new var for CONST/AND/OR/DECISION

2) Emit clauses per node in topological (level) order:
   Const(true):   (v)
   Const(false):  (¬v)

   Lit(leaf):     // no clauses (uses leaf_var)

   NegLit(leaf):  v ↔ ¬leaf_var
      ( v ∨  leaf_var)
      (¬v ∨ ¬leaf_var)

   And(children):
      For each child c: (¬v ∨ c)
      (v ∨ ¬c1 ∨ ¬c2 ∨ ... ∨ ¬ck)

   Or(children):
      For each child c: (v ∨ ¬c)
      (¬v ∨ c1 ∨ c2 ∨ ... ∨ ck)

   Decision(var, f, t):
      v ↔ ITE(var, t, f)
      (¬var ∨ ¬t ∨ v)
      ( var ∨ ¬f ∨ v)
      (¬var ∨ t ∨ ¬v)
      ( var ∨ f ∨ ¬v)

3) No root assertion clause in XLOG.
   Roots select which PIR subgraph is encoded; evidence/query are imposed by
   forcing the corresponding CNF variables via weights (e.g., setting one side
   to -INFINITY).
```

**Implementation detail:** all clause emission must be device‑side with
prefix‑sum allocation into the CSR buffers to avoid host coordination.

**Clause-count prepass (exact CSR sizing):**

Let `deg(i)` be the number of children for node `i` (0 for non‑AND/OR).
Let `is_const(i)`, `is_lit(i)`, `is_neglit(i)`, `is_and(i)`, `is_or(i)`, `is_dec(i)` be indicator functions.
There are no asserted roots in the CNF encoding used by exact inference.

Per node, the number of CNF clauses emitted is:

```
count(i) =
  is_const(i) * 1 +
  is_neglit(i) * 2 +
  is_and(i) * (deg(i) + 1) +
  is_or(i) * (deg(i) + 1) +
  is_dec(i) * 4
```

Total clauses:

```
num_clauses = sum_i count(i)
```

Total literal entries (CSR nnz) is:

```
nnz(i) =
  is_const(i) * 1 +
  is_neglit(i) * 4 +              // two 2-literal clauses
  is_and(i) * (3*deg(i) + 1) +
  is_or(i) * (3*deg(i) + 1) +
  is_dec(i) * 12                  // four 3-literal clauses

nnz_total = sum_i nnz(i)
```

This prepass runs entirely on GPU to allocate `clause_offsets` and `literals`.

**GPU kernel sequence for Tseitin encoding:**

```
// K0: Assign CNF variable IDs
//   - leaf_var[leaf_id]
//   - choice_var[var_id]
//   - node_var[node_id]

// K1: Count clauses + nnz per node
//   - write count(i) and nnz(i) to device arrays

// K2: Prefix-sum per-node counts to compute:
//   - base clause index per node
//   - base literal index per node
//   - total num_clauses and nnz_total

// K3: Emit clauses into CSR
//   - each node writes its own clauses at the precomputed offsets
//   - uses node_var/leaf_var/choice_var for literals
//   - fills clause_offsets for the per-node clause block and writes literals
```

**Notes:**
- K2 can reuse existing scan kernels in `kernels/scan.cu`.
- K3 writes fixed‑width clauses for CONST/NEG_LIT/DECISION and variable‑width for AND/OR.

#### 5.2.3 GPU-native weight tables

Exact evaluation currently builds `var_log_true/false` on CPU (`exact.rs`) and uploads to GPU.
To keep **zero transfers**, weights must also be GPU-resident:

```rust
/// Device-resident tables produced by `encode_cnf_gpu` for mapping PIR ids -> CNF var ids.
pub struct GpuCnfVarTables {
    pub node_var: DeviceSlice<u32>,   // PirNodeId -> CNF var id (DIMACS)
    pub leaf_var: DeviceSlice<u32>,   // LeafId -> CNF var id (DIMACS)
    pub choice_var: DeviceSlice<u32>, // ChoiceVarId -> CNF var id (DIMACS)
    pub max_var: u32,
}

pub struct GpuWeights {
    pub log_true: DeviceSlice<f64>,
    pub log_false: DeviceSlice<f64>,
}

pub fn build_weights_gpu(
    vars: &GpuCnfVarTables,
    leaf_probs: &DeviceSlice<f64>,          // indexed by LeafId
    choice_probs: &DeviceSlice<(f64, f64)>, // indexed by ChoiceVarId (binary choice vars)
    evidence: &DeviceSlice<u8>,             // indexed by CNF var id (DIMACS), 0/1/2
    provider: &CudaKernelProvider,
) -> Result<GpuWeights>;
```

**Annotated disjunction correctness requirement (matches `compile_annotated_disjunction`)**

The binary `choice_probs[var] = (p_true, p_false)` must match the conditional Bernoulli probabilities used
by the current lowering in `crates/xlog-prob/src/provenance.rs`:

```
p_true[i] = p_outcome[i] / remaining[i]
p_false[i] = 1 - p_true[i]
remaining[i+1] = remaining[i] - p_outcome[i]
```

For neural predicates that provide an `L`-way softmax, the GPU runtime must therefore:
- (optionally) scale the softmax so `sum(p_outcome) < 1` and create an explicit “none” branch,
- compute the conditional chain `p_true[i]` on GPU, and
- write `(ln p_true[i], ln(1-p_true[i]))` into `(log_true[var], log_false[var])` for the corresponding `ChoiceVarId`.

This integrates naturally with DLPack: neural outputs arrive as device tensors; the weight builder is a GPU kernel.

#### 5.2.4 Core GPU compile/verify API (exact signatures)

```rust
/// Compile CNF on GPU, then verify equivalence with GPU CDCL.
pub fn compile_gpu_d4_and_verify(
    cnf: &GpuCnf,
    provider: &CudaKernelProvider,
    config: &GpuCompileConfig,
) -> Result<GpuXgcf>;

/// Configuration for GPU D4 + GPU CDCL.
pub struct GpuCompileConfig {
    /// BFS expansion depth before handing each frontier item to a per-block DFS worker.
    pub frontier_depth: u16,
    /// Hard cap on the number of frontier work items (overflow is a hard error).
    pub max_frontier_items: u32,
    /// Absolute depth cap (defensive); exceeding this is a hard error (no UNKNOWN).
    pub max_depth: u16,

    /// CDCL restart cadence (deterministic).
    pub cdcl_restart_interval: u32,
    /// Learned clause arena size (bytes) for the verifier instance.
    pub cdcl_learned_bytes: u64,
    /// Optional conflict budget for debug/profiling only; production must be unbounded.
    pub cdcl_conflict_budget: Option<u64>,
}

/// Build a device-resident XGCF directly from device circuit buffers.
impl GpuXgcf {
    pub fn from_device(
        builder: GpuCircuitBuilder,
        layout: GpuCircuitLayout,
        provider: &CudaKernelProvider,
    ) -> Result<GpuXgcf>;
}

/// Device layout metadata for XGCF construction.
pub struct GpuCircuitLayout {
    pub num_nodes: u32,
    pub num_levels: u32,
    pub level_offsets: DeviceSlice<u32>,
    pub level_nodes: DeviceSlice<u32>,
    pub root: u32,
    pub max_var: u32,
}
```

---

### 5.3 GPU Neural Fast-Path (Template Circuits + Device Slot Mapping)

“GPU neural fast-path” in XLOG means: **compile once, then only update weights and run forward/backward**.
This is the critical path for training workloads where the logic structure is fixed but neural predictions
change per batch.

Current status (v0.3.x):
- `crates/pyxlog/src/lib.rs` caches circuits, but still:
  - compiles via **CPU D4** (expanded source text), and
  - builds weights / gradient maps on CPU (including an AD-chain mismatch).

GPU-native design:

1) **Template compilation (once per shape)**  
Key by a template signature (predicate, arity, neural label count, etc.). Cache stores:
- `GpuXgcf` (device-resident circuit),
- `GpuCnfVarTables` (for weights),
- a device-resident **slot map** from neural outputs to `ChoiceVarId`/CNF vars.

2) **Device slot map**

```rust
/// One slot per neural “label probability” produced by a neural predicate instance.
/// Slots are grouped by input_position (0..num_inputs).
pub struct GpuWeightSlots {
    pub group_offsets: DeviceSlice<u32>, // len = num_groups + 1
    pub slot_cnf_var: DeviceSlice<u32>,  // len = num_slots, each is a DIMACS CNF var id
}
```

3) **GPU weight fill (correct AD chain)**

Given a device tensor `p[label]` per group (typically softmax output), we must match
`compile_annotated_disjunction`’s conditional chain:

- Choose an always-present “none” branch with small `eps` and scale: `p_outcome = (1-eps) * p`.
  This guarantees `sum(p_outcome) < 1` and yields exactly `L` binary choice vars for `L` labels.
- Compute `remaining[i] = 1 - sum_{k<i} p_outcome[k]`.
- Compute `q[i] = p_outcome[i] / remaining[i]` and write:
  - `log_true[var_i]  = ln(q[i])`
  - `log_false[var_i] = ln(1-q[i])`

All of this is a GPU kernel; no `.tolist()` and no host-side loops.

4) **GPU gradient scatter (correct chain rule; uses both grad_true and grad_false)**

The XGCF backward pass returns gradients w.r.t **independent log-weights**:
`g_true[var] = ∂logP/∂ln(q)` and `g_false[var] = ∂logP/∂ln(1-q)`.

To backprop into neural probabilities, compute on GPU:

- `dlogP/dq[i] = g_true[var_i] * (1/q[i]) + g_false[var_i] * (-1/(1-q[i]))`
- `dlogP/dp_outcome[i] = (dlogP/dq[i])*(1/remaining[i]) + Σ_{j>i} (dlogP/dq[j])*(p_outcome[j]/remaining[j]^2)`
- For NLL loss `L = -logP`, the probability-space gradient is `dL/dp = -dlogP/dp`.
- Undo the `(1-eps)` scaling: `dL/dp_softmax = (1-eps) * dL/dp_outcome`.

Finally, export the per-group gradient tensors via DLPack and call `output.backward(grad)` in Python without
any CPU data transfers.

This fast-path is fully compatible with the “GPU D4 + GPU CDCL verifier” compiler: the verifier guarantees the
cached circuit is correct; the training loop only mutates weights.

## 6. Testing Strategy (GPU-Only)

All tests validate correctness **on GPU only**:

- GPU D4 unit tests (small CNFs, component decomposition, branching).
- GPU CDCL unit tests (SAT/UNSAT correctness).
- End‑to‑end tests:
  - Compile + verify + evaluate.
  - For small `num_vars` (e.g., <= 20): brute-force enumerate assignments on GPU to compute exact WMC and compare.
  - For larger instances: rely on the verifier (UNSAT) plus sanity checks and strict fail-fast on any SAT result.

No CPU comparison is used in the runtime path or CI gating for production builds.

---

## 7. Performance Targets

- **Compilation (GPU D4 + verification)**:
  - 1K clauses: 5–20 ms
  - 10K clauses: 20–100 ms
- **Verification**: dominated by GPU CDCL, typically 5–50 ms
- **Warm cache**: 0 ms compile, immediate evaluation

These are realistic targets given GPU CDCL cost.

---

## 8. Implementation Roadmap

### Phase 1: GPU D4 Core
- BFS work queue + unit propagation + branching
- GPU circuit builder and device‑resident XGCF output

### Phase 2: GPU CDCL Verifier
- Clause DB, watched literals, conflict analysis
- Equivalence checks for `φ` vs `C` (e.g., `SAT(φ ∧ ¬C)` and `SAT(C ∧ ¬φ)`)
  - **Status:** Implemented (Jan 25, 2026): GPU CDCL + on-GPU model/proof validation + zero-host-read equivalence queries

### Phase 3: GPU CNF Builder
- Move PIR → CNF encoding to GPU
- Ensure CNF never leaves device

### Phase 4: Cache + Integration
- Device‑resident circuit cache
- Replace CPU D4 call in `exact.rs`

---

## 9. Risks & Mitigations

1. **GPU CDCL Complexity**
   - Mitigation: start with a correctness-first CDCL (watched literals + 1-UIP + deterministic restarts),
     then optimize; learned-clause eviction may be used, but the solver must remain complete (no “UNKNOWN”).

2. **GPU Memory Pressure**
   - Mitigation: strict memory pools + LRU circuit eviction.

3. **No CPU Fallback**
   - Mitigation: verifier completeness is mandatory; reject any unknown outcome.

4. **Annotated Disjunction (AD) Weight Semantics**
   - Risk: incorrect mapping between multi-class probabilities and the binary conditional chain used by CNF lowering.
   - Mitigation: implement the chain mapping on GPU exactly as `compile_annotated_disjunction` does, and add GPU-only
     tests that compare the implied categorical distribution against expected probabilities.

5. **Neural Gradient Mapping**
   - Risk: using only `grad_true` or ignoring the chain rule yields incorrect training signals.
   - Mitigation: GPU kernels compute `dL/dp` using both `grad_true` and `grad_false` and the conditional-chain Jacobian
     (see 5.3), with unit tests on small circuits.

6. **GPU Provenance (Negation / WFS) Complexity**
   - Mitigation: implement provenance as a GPU column with batched hash-consing; stage non-monotone/WFS support behind
     a feature gate until verified end-to-end on GPU.

---

**Summary:** This design delivers a **strictly GPU-native, safe, and exact** compilation pipeline by combining **GPU D4**
and a **GPU CDCL equivalence verifier**, with no CPU fallback paths and no data-plane transfers of CNF/circuit/weights/gradients.

---

## Appendix A: Compatibility Impact (File-Level Changes)

This appendix lists the concrete source files that must change to support the GPU-native design.

1. `crates/xlog-prob/src/exact.rs`  
   - Replace CPU D4 invocation at line ~538 with `compile_gpu_d4_and_verify(&GpuCnf, &CudaKernelProvider, &GpuCompileConfig)`.
   - Remove temp file IO (`in.cnf`, `out.nnf`) and DDNNF parsing on CPU.

2. `crates/xlog-prob/src/cnf.rs`  
   - Keep host encoder for tooling/tests only.
   - Add GPU PIR + CNF encoders in:
     - `crates/xlog-prob/src/compilation/gpu_pir.rs`
     - `crates/xlog-prob/src/compilation/gpu_cnf.rs`

3. `crates/xlog-prob/src/compilation/mod.rs`  
   - Export `encode_cnf_gpu`, `compile_gpu_d4_and_verify`, and GPU CDCL verifier entrypoint.

4. `crates/xlog-prob/src/gpu.rs`  
   - Add `GpuXgcf::from_device(...)` constructor to build device-resident circuits.
   - Add evaluation path that accepts `GpuWeights` without host uploads.

5. `crates/xlog-prob/src/provenance.rs` and `crates/xlog-prob/src/pir.rs`  
   - Add GPU-resident PIR graph (`GpuPirGraph`) and GPU extraction path.
   - Ensure provenance extraction can emit GPU PIR directly or via GPU executor.

6. `crates/xlog-cuda/src/provider.rs` and `kernels/`  
   - Add new PTX modules for GPU D4 kernels and GPU CDCL verifier.
   - Wire kernel entry points into the provider (similar to existing circuit kernels).

7. `crates/xlog-solve/`  
   - CLS solver remains CPU-only and **not used** for verification.
   - GPU CNF + GPU CDCL verifier live here (`GpuCnf`, `GpuCdclSolver`) and are used by `xlog-prob` for equivalence
     checking.

8. `crates/pyxlog/src/lib.rs`  
   - Current template circuit cache stores `ExactDdnnfProgram` compiled via CPU D4.
   - Must be updated to call the GPU-native compiler path and to avoid any host-side CNF/DDNNF materialization.
   - Must make the neural cache path truly GPU-native:
     - no `.tolist()` (CPU readback) for neural outputs; use DLPack device tensors,
     - compute annotated-disjunction conditional chain probabilities on GPU (matches `compile_annotated_disjunction`),
     - backprop using both `grad_true` and `grad_false` with the correct chain rule (see 5.3).

9. `crates/xlog-cli/src/main.rs`  
   - `xlog prob --prob-engine exact_ddnnf` currently uses the CPU D4 path via `ExactDdnnfProgram`.
   - Must be updated to use the GPU-native compiler path.

10. Tests  
   - Add GPU-only integration tests in `crates/xlog-prob/tests/` for:
     - GPU D4 compile + GPU CDCL verify
     - Device-resident CNF builder
     - Device-resident weight tables
   - Optionally add CUDA certification categories for D4/CDCL kernels.
