# P3: Incremental Verifier Interface — Design

**Date:** 2026-03-08
**Status:** Implemented (2026-03-08, branch feat/p3-incremental-verifier)
**Depends on:** v0.5.0-phase1 (tagged 2026-03-06, commit 407d8bab)

## Goal

Amortize GPU memory allocation across the q1/q2 equivalence check pair by
reusing a pre-allocated solver workspace, eliminating redundant arena
allocation/deallocation between the two solves.

## Non-goals

- Incremental SAT semantics (learned clause transfer between solves)
- Assumption-based solving (q1 and q2 have different variable spaces)
- `add_clauses()` API (deferred to future version if needed)
- Python-level exposure (controlled entirely via Rust config)

---

## Section 1: Architecture

### Naming

`GpuCdclWorkspace` — not "Session" or "Arena". v1 is persistent allocation
with resettable solver state, not true incremental SAT.

### Ownership

Workspace is standalone. Not stored inside `GpuCdclSolver`. All existing
`solve_*` methods remain `&self`; workspace is passed explicitly as `&mut`.

### Construction

```rust
let mut ws = solver.new_workspace(max_var_cap, max_clause_cap)?;
```

Created via `solver.new_workspace()` to prevent provider/config mismatch.
The solver uses its own `GpuCdclConfig` (max_learned_clauses,
max_learned_lits, max_proof_u32) for learned-clause and proof buffer sizing.

### Sizing contract

- `max_var_cap`: `std::cmp::max(q1.var_cap, q2.var_cap)`
- `max_clause_cap`: `std::cmp::max(q1.clause_cap, q2.clause_cap)`

Uses explicit max even though q2 is currently larger by construction.

### Solver methods

New `_ws` variants mirror existing methods:

- `solve_expect_unsat_with_branch_limit_ws(&self, ws: &mut GpuCdclWorkspace, ...)`
- `solve_expect_unsat_with_branch_limit_gated_ws(&self, ws: &mut GpuCdclWorkspace, ...)`
- `solve_expect_unsat_with_decision_ranges_ws(&self, ws: &mut GpuCdclWorkspace, ...)`
- `solve_expect_unsat_with_decision_ranges_gated_ws(&self, ws: &mut GpuCdclWorkspace, ...)`

Naming convention: `_ws` before `_gated` would break the existing suffix
convention where `_gated` ends the name. Follow the existing pattern:
`_ws_gated` is wrong; `_gated_ws` is wrong. The `_ws` variants are separate
methods that accept workspace, and `_gated` variants of those append `_gated`
at the end: `*_ws` and `*_ws_gated` — but to stay consistent with the
existing codebase where `_gated` is always the terminal suffix, use
`*_gated_ws` only if the codebase already does this, otherwise follow
whatever suffix ordering the implementation plan determines from the existing
code. The implementation plan should verify the convention.

### Error handling

If the CNF exceeds workspace capacity (var_cap or clause_cap), return
`XlogError::Kernel` with a descriptive message. No silent fallback to
per-call allocation.

---

## Section 2: Workspace Internals

### Buffer inventory

All 30 arena buffers from `gpu_cdcl.rs:147-188`, with exact types:

**Variable state** (sized `var_cap + 1`):

| Buffer | Type |
|--------|------|
| assign | i8 |
| level | u32 |
| reason | i32 |
| var_activity | u32 |
| var_phase | i8 |
| decision_heap | u32 |
| decision_heap_pos | u32 |

**Trail** (sized `var_cap + 1`):

| Buffer | Type |
|--------|------|
| trail | i32 |
| trail_lim | u32 |

**Analysis scratch** (sized `var_cap + 1`):

| Buffer | Type |
|--------|------|
| seen | u8 |
| learnt_tmp | i32 |
| proof_vars_tmp | u32 |
| proof_reason_tmp | u32 |

**Watch lists**:

| Buffer | Type | Size |
|--------|------|------|
| watch0_pos | u32 | clause_total_cap |
| watch1_pos | u32 | clause_total_cap |
| watch_head | i32 | 2 * var_cap |
| watch_next | i32 | 2 * clause_total_cap |
| watch_prev | i32 | 2 * clause_total_cap |

Where `clause_total_cap = max_clause_cap + config.max_learned_clauses`.

**Learned clause storage**:

| Buffer | Type | Size |
|--------|------|------|
| learned_offsets | u32 | max_learned_clauses + 1 |
| learned_lits | i32 | max_learned_lits |
| learned_deleted | u8 | max_learned_clauses |
| learned_lbd | u32 | max_learned_clauses |
| learned_activity | u32 | max_learned_clauses |
| learned_locked | u8 | max_learned_clauses |

**Proof**:

| Buffer | Type | Size |
|--------|------|------|
| proof_offsets | u32 | max_learned_clauses + 1 |
| proof_data | u32 | max_proof_u32 |

**Scalar outputs**:

| Buffer | Type | Size |
|--------|------|------|
| out_status | i32 | 1 |
| out_error | i32 | 1 |
| out_learned_count | u32 | 1 |

### Reset policy

`reset_for_solve()` is a **no-op**. The `sat_cdcl_solve` kernel initializes
all mutable state internally:

- `sat.cu:1220` — variable state (assign, level, reason, activity, phase,
  heap)
- `sat.cu:1293` — watch lists (watch_head, watch0_pos, watch1_pos,
  watch_next, watch_prev)
- `sat.cu:1329` — learned clause metadata
- `sat.cu:1341` — proof, trail, analysis scratch, outputs

External zeroing would double-pay initialization cost.

### Workspace does NOT own

- CNF storage (`clause_offsets`, `literals`) — stays on `GpuCnf`
- Provider reference — workspace stores raw device pointers only

---

## Section 3: Opt-in and Propagation

### Config layer

`GpuEquivalenceConfig` (validation.rs:21) gains `reuse_workspace`:

```rust
pub struct GpuEquivalenceConfig {
    pub cdcl: GpuCdclConfig,
    pub reuse_workspace: bool,  // default false
}
```

### Propagation

`GpuCompileConfig` (gpu_d4.rs:49) gains `incremental_verify: bool` (default
`false`). Since `GpuCompileConfig` has no `Default` impl, every struct
literal site must add `incremental_verify: false`. The implementation plan
must enumerate all sites (at minimum: `gpu_d4.rs`, test files in
`crates/xlog-prob/tests/`).

The two call sites in `mod.rs` map it:

```rust
// mod.rs:135 (non-cached path)
GpuEquivalenceConfig { cdcl, reuse_workspace: config.incremental_verify }

// mod.rs:305 (cached path)
GpuEquivalenceConfig { cdcl, reuse_workspace: config.incremental_verify }
```

### Integration in validation.rs

`check_equivalence_gpu_gated` (validation.rs:812-904):

```rust
if equiv_config.reuse_workspace {
    let max_var_cap = std::cmp::max(q1.var_cap, q2.var_cap);
    let max_clause_cap = std::cmp::max(q1.clause_cap, q2.clause_cap);
    let mut ws = solver.new_workspace(max_var_cap, max_clause_cap)?;
    // q1: branch-limit variant
    solver.solve_expect_unsat_with_branch_limit_gated_ws(&mut ws, &q1_cnf, ...)?;
    // q2: decision-ranges variant
    solver.solve_expect_unsat_with_decision_ranges_gated_ws(&mut ws, &q2_cnf, ...)?;
} else {
    solver.solve_expect_unsat_with_branch_limit_gated(&q1_cnf, ...)?;
    solver.solve_expect_unsat_with_decision_ranges_gated(&q2_cnf, ...)?;
}
```

Same pattern for non-gated `check_equivalence_gpu` (without `_gated` suffix).

### Gated variant contract

`_gated_ws` (or `_ws_gated` — see Section 1 naming note) methods accept the
same `compile_needed: &TrackedCudaSlice<u32>` as existing `_gated` methods.
When the gate is 0, kernel early-returns at `sat.cu:1137` / `sat.cu:2598`.
Workspace buffers remain untouched, which is correct since the solver would
not have written to them.

---

## Section 4: Testing

1. **Unit test** (`crates/xlog-solve/tests/`): Create workspace via
   `solver.new_workspace()`, solve a known-UNSAT CNF twice via `_ws`
   variants, verify both return UNSAT. Confirm workspace device pointers are
   identical across calls (reuse, not realloc).

2. **Capacity overflow test** (`crates/xlog-solve/tests/`): Create workspace
   with tiny `max_var_cap`, attempt to solve a CNF that exceeds it — verify
   clean `XlogError::Kernel`, not panic or silent corruption.

3. **Integration test** (`crates/xlog-prob/tests/`): Compile a small CNF
   with `incremental_verify: true`, verify equivalence check passes
   identically to `incremental_verify: false`. No Python exposure needed.

4. **Regression**: `cargo test --workspace --all-targets --exclude pyxlog
   --release` and `pytest python/tests/` pass unchanged (default is `false`,
   no behavioral change).

---

## Design Decisions Log

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Name: GpuCdclWorkspace, not Session | v1 is allocation reuse, not incremental SAT |
| D2 | Standalone, not stored in solver | Solver methods are `&self`; storing workspace requires `&mut self` or interior mutability |
| D3 | Created via `solver.new_workspace()` | Prevents provider/config mismatch |
| D4 | No-op `reset_for_solve()` | Kernel initializes all mutable state (sat.cu:1220, 1293, 1329, 1341) |
| D5 | Sizing uses explicit `max(q1, q2)` | Future-proof even though q2 > q1 by construction today |
| D6 | Error on capacity overflow, no fallback | Keeps workspace contract simple and predictable |
| D7 | Opt-in on GpuEquivalenceConfig | Direct consumer; GpuCompileConfig maps into it |
| D8 | Tests in Rust, not Python | Feature is Rust-internal, no Python exposure needed |
| D9 | Defer `add_clauses()` | No proven use case for v1; keeps scope minimal |
