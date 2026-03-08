# P3: Incremental Verifier Interface — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Amortize GPU arena allocation across the q1/q2 equivalence check pair by reusing a pre-allocated `GpuCdclWorkspace`.

**Architecture:** `GpuCdclWorkspace` owns the 30 device buffers currently allocated per solve call. Created via `solver.new_workspace(max_var_cap, max_clause_cap)`. Passed as `&mut` to new `_ws` solver method variants. `reset_for_solve()` is a no-op — the kernel initializes all mutable state. Opt-in via `GpuCompileConfig.incremental_verify` → `GpuEquivalenceConfig.reuse_workspace`.

**Tech Stack:** Rust (xlog-solve, xlog-prob), CUDA kernels (sat.cu — no kernel changes needed)

**Design doc:** `docs/plans/2026-03-08-p3-incremental-verifier-design.md`

---

### Task 1: Add `GpuCdclWorkspace` struct and `new_workspace` constructor

**Files:**
- Modify: `crates/xlog-solve/src/gpu_cdcl.rs` (add struct + constructor after line 58)

**Step 1: Add the `GpuCdclWorkspace` struct**

After `GpuCdclSolver` (line 58), add:

```rust
/// Pre-allocated solver arena for reuse across multiple CDCL solves.
///
/// Owns the 30 device buffers that `launch_cdcl_with_decision_ranges_gated` normally
/// allocates per call. Does NOT own CNF storage (clause_offsets/literals stay on GpuCnf).
///
/// Created via [`GpuCdclSolver::new_workspace`]. Passed as `&mut` to `_ws` solver methods.
///
/// `reset_for_solve()` is intentionally a no-op: the `sat_cdcl_solve` kernel initializes
/// all mutable state at launch (sat.cu:1220, 1293, 1329, 1341).
pub struct GpuCdclWorkspace {
    // Capacity limits (used for overflow checks)
    pub(crate) var_cap: usize,
    pub(crate) clause_total_cap: usize,

    // Variable state (var_cap + 1 each)
    pub(crate) assign: TrackedCudaSlice<i8>,
    pub(crate) level: TrackedCudaSlice<u32>,
    pub(crate) reason: TrackedCudaSlice<i32>,
    pub(crate) var_activity: TrackedCudaSlice<u32>,
    pub(crate) var_phase: TrackedCudaSlice<i8>,
    pub(crate) decision_heap: TrackedCudaSlice<u32>,
    pub(crate) decision_heap_pos: TrackedCudaSlice<u32>,

    // Trail (var_cap + 1 each)
    pub(crate) trail: TrackedCudaSlice<i32>,
    pub(crate) trail_lim: TrackedCudaSlice<u32>,

    // Analysis scratch (var_cap + 1 each)
    pub(crate) seen: TrackedCudaSlice<u8>,
    pub(crate) learnt_tmp: TrackedCudaSlice<i32>,
    pub(crate) proof_vars_tmp: TrackedCudaSlice<u32>,
    pub(crate) proof_reason_tmp: TrackedCudaSlice<u32>,

    // Watch lists
    pub(crate) watch0_pos: TrackedCudaSlice<u32>,     // clause_total_cap
    pub(crate) watch1_pos: TrackedCudaSlice<u32>,     // clause_total_cap
    pub(crate) watch_head: TrackedCudaSlice<i32>,     // 2 * var_cap
    pub(crate) watch_next: TrackedCudaSlice<i32>,     // 2 * clause_total_cap
    pub(crate) watch_prev: TrackedCudaSlice<i32>,     // 2 * clause_total_cap

    // Learned clause arena
    pub(crate) learned_offsets: TrackedCudaSlice<u32>,  // max_learned_clauses + 1
    pub(crate) learned_lits: TrackedCudaSlice<i32>,     // max_learned_lits
    pub(crate) learned_deleted: TrackedCudaSlice<u8>,   // max_learned_clauses
    pub(crate) learned_lbd: TrackedCudaSlice<u32>,      // max_learned_clauses
    pub(crate) learned_activity: TrackedCudaSlice<u32>, // max_learned_clauses
    pub(crate) learned_locked: TrackedCudaSlice<u8>,    // max_learned_clauses

    // Proof trace
    pub(crate) proof_offsets: TrackedCudaSlice<u32>,  // max_learned_clauses + 1
    pub(crate) proof_data: TrackedCudaSlice<u32>,     // max_proof_u32

    // Scalar outputs
    pub(crate) out_status: TrackedCudaSlice<i32>,       // 1
    pub(crate) out_error: TrackedCudaSlice<i32>,        // 1
    pub(crate) out_learned_count: TrackedCudaSlice<u32>, // 1
}

impl GpuCdclWorkspace {
    /// No-op: the sat_cdcl_solve kernel initializes all mutable state at launch.
    #[inline]
    pub fn reset_for_solve(&mut self) {
        // Intentionally empty. See sat.cu:1220, 1293, 1329, 1341.
    }
}
```

**Step 2: Add `new_workspace` on `GpuCdclSolver`**

Inside the `impl GpuCdclSolver` block, after `new()` (line 73), add:

```rust
    /// Pre-allocate a reusable solver arena.
    ///
    /// `max_var_cap` and `max_clause_cap` must be >= the `var_cap` / `clause_cap` of any
    /// `GpuCnf` that will be solved with this workspace. If a solve call exceeds these
    /// capacities, it returns `XlogError::Kernel`.
    pub fn new_workspace(
        &self,
        max_var_cap: u32,
        max_clause_cap: u32,
    ) -> Result<GpuCdclWorkspace> {
        let num_vars_cap = max_var_cap as usize;
        let num_clauses_cap = max_clause_cap as usize;
        let max_learned_clauses = self.config.max_learned_clauses as usize;
        let max_learned_lits = self.config.max_learned_lits as usize;
        let max_proof_u32 = self.config.max_proof_u32 as usize;

        let max_total_clauses = num_clauses_cap
            .checked_add(max_learned_clauses)
            .ok_or_else(|| XlogError::Kernel("SAT clause capacity overflow".to_string()))?;

        let memory = self.provider.memory();

        Ok(GpuCdclWorkspace {
            var_cap: num_vars_cap,
            clause_total_cap: max_total_clauses,

            // Variable state
            assign: memory.alloc::<i8>(num_vars_cap + 1)?,
            level: memory.alloc::<u32>(num_vars_cap + 1)?,
            reason: memory.alloc::<i32>(num_vars_cap + 1)?,
            var_activity: memory.alloc::<u32>(num_vars_cap + 1)?,
            var_phase: memory.alloc::<i8>(num_vars_cap + 1)?,
            decision_heap: memory.alloc::<u32>(num_vars_cap + 1)?,
            decision_heap_pos: memory.alloc::<u32>(num_vars_cap + 1)?,

            // Trail
            trail: memory.alloc::<i32>(num_vars_cap + 1)?,
            trail_lim: memory.alloc::<u32>(num_vars_cap + 1)?,

            // Analysis scratch
            seen: memory.alloc::<u8>(num_vars_cap + 1)?,
            learnt_tmp: memory.alloc::<i32>(num_vars_cap + 1)?,
            proof_vars_tmp: memory.alloc::<u32>(num_vars_cap + 1)?,
            proof_reason_tmp: memory.alloc::<u32>(num_vars_cap + 1)?,

            // Watch lists
            watch0_pos: memory.alloc::<u32>(max_total_clauses)?,
            watch1_pos: memory.alloc::<u32>(max_total_clauses)?,
            watch_head: memory.alloc::<i32>(2 * num_vars_cap)?,
            watch_next: memory.alloc::<i32>(2 * max_total_clauses)?,
            watch_prev: memory.alloc::<i32>(2 * max_total_clauses)?,

            // Learned
            learned_offsets: memory.alloc::<u32>(max_learned_clauses + 1)?,
            learned_lits: memory.alloc::<i32>(max_learned_lits)?,
            learned_deleted: memory.alloc::<u8>(max_learned_clauses)?,
            learned_lbd: memory.alloc::<u32>(max_learned_clauses)?,
            learned_activity: memory.alloc::<u32>(max_learned_clauses)?,
            learned_locked: memory.alloc::<u8>(max_learned_clauses)?,

            // Proof
            proof_offsets: memory.alloc::<u32>(max_learned_clauses + 1)?,
            proof_data: memory.alloc::<u32>(max_proof_u32)?,

            // Outputs
            out_status: memory.alloc::<i32>(1)?,
            out_error: memory.alloc::<i32>(1)?,
            out_learned_count: memory.alloc::<u32>(1)?,
        })
    }
```

**Step 3: Run workspace compilation check**

Run: `cargo check -p xlog-solve --release 2>&1 | head -20`
Expected: compiles clean (workspace is defined but not yet used)

**Step 4: Commit**

```bash
git add crates/xlog-solve/src/gpu_cdcl.rs
git commit -m "feat(solve): add GpuCdclWorkspace struct and new_workspace constructor"
```

---

### Task 2: Add `launch_cdcl_ws_gated` internal method

The existing `launch_cdcl_with_decision_ranges_gated` (gpu_cdcl.rs:87) allocates all 30 buffers internally. We need a parallel method that uses workspace buffers instead.

**Files:**
- Modify: `crates/xlog-solve/src/gpu_cdcl.rs`

**Step 1: Add capacity-check + workspace launch method**

After the existing `launch_cdcl_with_decision_ranges_gated` method (ends around line 280), add:

```rust
    /// Launch CDCL using pre-allocated workspace buffers.
    ///
    /// Returns a `GpuCdclRun` that borrows from the workspace (the workspace must outlive
    /// the run). Errors if the CNF exceeds workspace capacity.
    fn launch_cdcl_with_workspace_gated(
        &self,
        ws: &mut GpuCdclWorkspace,
        cnf: &GpuCnf,
        compile_needed: &TrackedCudaSlice<u32>,
        decision_base_limit: &TrackedCudaSlice<u32>,
        decision_extra_base: &TrackedCudaSlice<u32>,
        decision_extra_count: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        let num_vars_cap = cnf.var_cap as usize;
        let num_clauses_cap = cnf.clause_cap as usize;

        if num_vars_cap > ws.var_cap {
            return Err(XlogError::Kernel(format!(
                "CNF var_cap {} exceeds workspace var_cap {}",
                num_vars_cap, ws.var_cap
            )));
        }
        let max_total_clauses = num_clauses_cap
            .checked_add(self.config.max_learned_clauses as usize)
            .ok_or_else(|| XlogError::Kernel("SAT clause capacity overflow".to_string()))?;
        if max_total_clauses > ws.clause_total_cap {
            return Err(XlogError::Kernel(format!(
                "CNF clause_total {} exceeds workspace clause_total_cap {}",
                max_total_clauses, ws.clause_total_cap
            )));
        }

        if cnf.var_cap == 0 {
            return Err(XlogError::Compilation(
                "GpuCdclSolver requires num_vars > 0".to_string(),
            ));
        }
        if decision_base_limit.len() != 1 {
            return Err(XlogError::Compilation(format!(
                "GpuCdclSolver requires decision_base_limit len=1, got {}",
                decision_base_limit.len()
            )));
        }
        if decision_extra_base.len() != 1 {
            return Err(XlogError::Compilation(format!(
                "GpuCdclSolver requires decision_extra_base len=1, got {}",
                decision_extra_base.len()
            )));
        }
        if decision_extra_count.len() != 1 {
            return Err(XlogError::Compilation(format!(
                "GpuCdclSolver requires decision_extra_count len=1, got {}",
                decision_extra_count.len()
            )));
        }
        if self.config.max_learned_clauses == 0 {
            return Err(XlogError::Compilation(
                "GpuCdclSolver requires max_learned_clauses > 0".to_string(),
            ));
        }
        if self.config.max_learned_lits == 0 {
            return Err(XlogError::Compilation(
                "GpuCdclSolver requires max_learned_lits > 0".to_string(),
            ));
        }
        if self.config.max_proof_u32 < 2 {
            return Err(XlogError::Compilation(
                "GpuCdclSolver requires max_proof_u32 >= 2".to_string(),
            ));
        }

        ws.reset_for_solve(); // no-op, documented

        let sat_fn = self
            .provider
            .device()
            .inner()
            .get_func(SAT_MODULE, sat_kernels::SAT_CDCL_SOLVE)
            .ok_or_else(|| XlogError::Kernel("sat_cdcl_solve kernel not found".to_string()))?;

        let cnf_var_cap = cnf.var_cap;
        let cnf_clause_cap = cnf.clause_cap;
        let cfg_max_learned_clauses = self.config.max_learned_clauses;
        let cfg_max_learned_lits = self.config.max_learned_lits;
        let cfg_max_proof_u32 = self.config.max_proof_u32;
        let cfg_restart_base = self.config.restart_base;
        let cfg_reduce_interval = self.config.reduce_interval;

        let mut params: Vec<*mut c_void> = vec![
            compile_needed.as_kernel_param(),
            (&cnf.clause_offsets).as_kernel_param(),
            (&cnf.literals).as_kernel_param(),
            (&cnf.num_vars).as_kernel_param(),
            (&cnf.num_clauses).as_kernel_param(),
            decision_base_limit.as_kernel_param(),
            decision_extra_base.as_kernel_param(),
            decision_extra_count.as_kernel_param(),
            cnf_var_cap.as_kernel_param(),
            cnf_clause_cap.as_kernel_param(),
            cfg_max_learned_clauses.as_kernel_param(),
            cfg_max_learned_lits.as_kernel_param(),
            cfg_max_proof_u32.as_kernel_param(),
            cfg_restart_base.as_kernel_param(),
            cfg_reduce_interval.as_kernel_param(),
            (&mut ws.assign).as_kernel_param(),
            (&mut ws.level).as_kernel_param(),
            (&mut ws.reason).as_kernel_param(),
            (&mut ws.var_activity).as_kernel_param(),
            (&mut ws.var_phase).as_kernel_param(),
            (&mut ws.decision_heap).as_kernel_param(),
            (&mut ws.decision_heap_pos).as_kernel_param(),
            (&mut ws.trail).as_kernel_param(),
            (&mut ws.trail_lim).as_kernel_param(),
            (&mut ws.seen).as_kernel_param(),
            (&mut ws.learnt_tmp).as_kernel_param(),
            (&mut ws.proof_vars_tmp).as_kernel_param(),
            (&mut ws.proof_reason_tmp).as_kernel_param(),
            (&mut ws.watch0_pos).as_kernel_param(),
            (&mut ws.watch1_pos).as_kernel_param(),
            (&mut ws.watch_head).as_kernel_param(),
            (&mut ws.watch_next).as_kernel_param(),
            (&mut ws.watch_prev).as_kernel_param(),
            (&mut ws.learned_offsets).as_kernel_param(),
            (&mut ws.learned_lits).as_kernel_param(),
            (&mut ws.learned_deleted).as_kernel_param(),
            (&mut ws.learned_lbd).as_kernel_param(),
            (&mut ws.learned_activity).as_kernel_param(),
            (&mut ws.learned_locked).as_kernel_param(),
            (&mut ws.proof_offsets).as_kernel_param(),
            (&mut ws.proof_data).as_kernel_param(),
            (&mut ws.out_status).as_kernel_param(),
            (&mut ws.out_error).as_kernel_param(),
            (&mut ws.out_learned_count).as_kernel_param(),
        ];

        unsafe {
            sat_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut params,
            )
        }
        .map_err(|e| XlogError::Kernel(format!("Failed to launch SAT solver kernel: {}", e)))?;

        Ok(())
    }
```

**Important difference from `launch_cdcl_with_decision_ranges_gated`:** This method returns `Result<()>` instead of `Result<GpuCdclRun>`. The caller reads `ws.out_status` / `ws.out_error` / `ws.out_learned_count` directly from the workspace. No ownership transfer needed.

**Step 2: Run compilation check**

Run: `cargo check -p xlog-solve --release 2>&1 | head -20`
Expected: compiles clean

**Step 3: Commit**

```bash
git add crates/xlog-solve/src/gpu_cdcl.rs
git commit -m "feat(solve): add launch_cdcl_with_workspace_gated internal method"
```

---

### Task 3: Add public `_ws` solver method variants

Only the 4 methods actually used by the verifier need `_ws` variants:

- `solve_expect_unsat_with_branch_limit_ws` (for q1, non-gated)
- `solve_expect_unsat_with_branch_limit_gated_ws` (for q1, gated)
- `solve_expect_unsat_with_decision_ranges_ws` (for q2, non-gated)
- `solve_expect_unsat_with_decision_ranges_gated_ws` (for q2, gated)

**Files:**
- Modify: `crates/xlog-solve/src/gpu_cdcl.rs`

**Step 1: Add the 4 `_ws` methods**

Add after `solve_expect_unsat_with_decision_ranges_gated` (line ~637). The gated variants are the real implementations; the non-gated ones delegate (same pattern as existing code).

```rust
    // ── Workspace variants ──────────────────────────────────────────────────

    /// Like `solve_expect_unsat_with_branch_limit`, but reuses a pre-allocated workspace.
    pub fn solve_expect_unsat_with_branch_limit_ws(
        &self,
        ws: &mut GpuCdclWorkspace,
        cnf: &GpuCnf,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        let compile_needed = self.alloc_u32_scalar(1)?;
        self.solve_expect_unsat_with_branch_limit_gated_ws(
            ws, cnf, &compile_needed, branch_var_limit,
        )
    }

    /// Like `solve_expect_unsat_with_branch_limit_gated`, but reuses a pre-allocated workspace.
    pub fn solve_expect_unsat_with_branch_limit_gated_ws(
        &self,
        ws: &mut GpuCdclWorkspace,
        cnf: &GpuCnf,
        compile_needed: &TrackedCudaSlice<u32>,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        let zero = self.alloc_u32_scalar(0)?;
        self.solve_expect_unsat_with_decision_ranges_gated_ws(
            ws,
            cnf,
            compile_needed,
            branch_var_limit,
            &zero,
            &zero,
        )
    }

    /// Like `solve_expect_unsat_with_decision_ranges`, but reuses a pre-allocated workspace.
    pub fn solve_expect_unsat_with_decision_ranges_ws(
        &self,
        ws: &mut GpuCdclWorkspace,
        cnf: &GpuCnf,
        decision_base_limit: &TrackedCudaSlice<u32>,
        decision_extra_base: &TrackedCudaSlice<u32>,
        decision_extra_count: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        let compile_needed = self.alloc_u32_scalar(1)?;
        self.solve_expect_unsat_with_decision_ranges_gated_ws(
            ws,
            cnf,
            &compile_needed,
            decision_base_limit,
            decision_extra_base,
            decision_extra_count,
        )
    }

    /// Like `solve_expect_unsat_with_decision_ranges_gated`, but reuses a pre-allocated workspace.
    pub fn solve_expect_unsat_with_decision_ranges_gated_ws(
        &self,
        ws: &mut GpuCdclWorkspace,
        cnf: &GpuCnf,
        compile_needed: &TrackedCudaSlice<u32>,
        decision_base_limit: &TrackedCudaSlice<u32>,
        decision_extra_base: &TrackedCudaSlice<u32>,
        decision_extra_count: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        #[cfg(debug_assertions)]
        let trace = std::env::var_os("XLOG_CDCL_TRACE").is_some();
        #[cfg(debug_assertions)]
        let t0 = std::time::Instant::now();

        self.launch_cdcl_with_workspace_gated(
            ws,
            cnf,
            compile_needed,
            decision_base_limit,
            decision_extra_base,
            decision_extra_count,
        )?;

        let device = self.provider.device().inner();
        let memory = self.provider.memory();

        let assert_status_fn = device
            .get_func(SAT_MODULE, sat_kernels::SAT_ASSERT_STATUS)
            .ok_or_else(|| XlogError::Kernel("sat_assert_status kernel not found".to_string()))?;
        unsafe {
            assert_status_fn
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        compile_needed,
                        &ws.out_status,
                        &ws.out_error,
                        SAT_STATUS_UNSAT,
                    ),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("Failed to launch sat_assert_status: {}", e))
                })?;
        }
        self.provider.device().synchronize()?;
        #[cfg(debug_assertions)]
        if trace {
            eprintln!("[xlog-solve] cdcl_ws(unsat) time: {:?}", t0.elapsed());
        }

        // Proof check — same logic as solve_expect_unsat_with_decision_ranges_gated,
        // but reads learned clause / proof buffers from workspace.
        let mut out_ok = memory.alloc::<i32>(1)?;
        device
            .htod_sync_copy_into(&[1i32], &mut out_ok)
            .map_err(|e| XlogError::Kernel(format!("Failed to init proof out_ok: {}", e)))?;

        let scratch_cap_u32 = cnf
            .var_cap
            .checked_add(1)
            .ok_or_else(|| XlogError::Kernel("Proof scratch capacity overflow".to_string()))?;
        let scratch_cap = scratch_cap_u32 as usize;

        let mut proof_blocks: usize = 1;
        let mut scratch_a = None;
        let mut scratch_b = None;
        let mut scratch_map = None;
        let mut last_alloc_err: Option<XlogError> = None;
        for blocks in [512usize, 256, 128, 64, 32, 16, 8, 4, 2, 1] {
            let len = match scratch_cap.checked_mul(blocks) {
                Some(v) => v,
                None => {
                    last_alloc_err = Some(XlogError::Kernel(
                        "Proof scratch allocation length overflow".to_string(),
                    ));
                    continue;
                }
            };

            let a = match memory.alloc::<i32>(len) {
                Ok(buf) => buf,
                Err(e) => { last_alloc_err = Some(e); continue; }
            };
            let b = match memory.alloc::<i32>(len) {
                Ok(buf) => buf,
                Err(e) => { last_alloc_err = Some(e); drop(a); continue; }
            };
            let m = match memory.alloc::<u32>(len) {
                Ok(buf) => buf,
                Err(e) => { last_alloc_err = Some(e); drop(a); drop(b); continue; }
            };

            proof_blocks = blocks;
            scratch_a = Some(a);
            scratch_b = Some(b);
            scratch_map = Some(m);
            break;
        }
        let mut scratch_a = scratch_a.ok_or_else(|| {
            last_alloc_err.unwrap_or_else(|| {
                XlogError::Kernel("Failed to allocate proof scratch buffers".to_string())
            })
        })?;
        let mut scratch_b = scratch_b.unwrap();
        let mut scratch_map = scratch_map.unwrap();

        // NOTE: The proof check kernel call here must reference ws.learned_offsets,
        // ws.learned_lits, ws.proof_offsets, ws.proof_data, ws.out_learned_count
        // instead of the GpuCdclRun fields. The implementer must copy the exact
        // proof_check launch from solve_expect_unsat_with_decision_ranges_gated
        // (gpu_cdcl.rs:757-810), replacing `run.learned_offsets` → `ws.learned_offsets`
        // etc. The kernel function name is sat_kernels::SAT_PROOF_CHECK.
        //
        // The implementer should read gpu_cdcl.rs:757-810 and replicate the launch,
        // substituting workspace fields for GpuCdclRun fields.

        // [IMPLEMENTER: Copy the sat_proof_check launch block from
        //  solve_expect_unsat_with_decision_ranges_gated, lines ~757-810,
        //  replacing run.X with ws.X for: learned_offsets, learned_lits,
        //  proof_offsets, proof_data, out_learned_count.
        //  Also replace run.assignment with ws.assign.
        //  Keep scratch_a, scratch_b, scratch_map, out_ok as local.]

        Ok(())
    }
```

**Important implementation note:** The code block marked `[IMPLEMENTER]` above is a placeholder. The implementer **must** read the proof-check launch at `gpu_cdcl.rs:757-810` (the `sat_proof_check` kernel launch in `solve_expect_unsat_with_decision_ranges_gated`) and replicate it exactly, substituting `ws.` fields for `run.` fields. Do not skip proof checking — it is a correctness requirement.

**Step 2: Run compilation check**

Run: `cargo check -p xlog-solve --release 2>&1 | head -30`
Expected: compiles (warnings about unused variables from the placeholder are OK)

**Step 3: Commit**

```bash
git add crates/xlog-solve/src/gpu_cdcl.rs
git commit -m "feat(solve): add _ws public solver method variants for workspace reuse"
```

---

### Task 4: Unit test — workspace reuse correctness

**Files:**
- Create: `crates/xlog-solve/tests/gpu_cdcl_workspace.rs`

**Step 1: Write the test**

```rust
//! Tests for GpuCdclWorkspace reuse across multiple CDCL solves.

use std::sync::Arc;
use xlog_cuda::CudaKernelProvider;
use xlog_solve::gpu_cdcl::{GpuCdclConfig, GpuCdclSolver, GpuCdclWorkspace};
use xlog_solve::gpu_cnf::GpuCnf;

fn make_provider() -> Arc<CudaKernelProvider> {
    // Use device 0 with 64 MB — same as other solver tests.
    Arc::new(CudaKernelProvider::new(0, 64 * 1024 * 1024).expect("CUDA provider"))
}

/// Build a trivially UNSAT CNF: (x1) AND (NOT x1).
/// var_cap=1, clause_cap=2, lit_cap=2.
fn build_trivial_unsat(provider: &Arc<CudaKernelProvider>) -> GpuCnf {
    let memory = provider.memory();
    let device = provider.device().inner();

    let mut num_vars = memory.alloc::<u32>(1).unwrap();
    device.htod_sync_copy_into(&[1u32], &mut num_vars).unwrap();
    let mut num_clauses = memory.alloc::<u32>(1).unwrap();
    device.htod_sync_copy_into(&[2u32], &mut num_clauses).unwrap();
    let mut num_lits = memory.alloc::<u32>(1).unwrap();
    device.htod_sync_copy_into(&[2u32], &mut num_lits).unwrap();

    // CSR offsets: [0, 1, 2]
    let mut clause_offsets = memory.alloc::<u32>(3).unwrap();
    device.htod_sync_copy_into(&[0u32, 1, 2], &mut clause_offsets).unwrap();
    // Literals: [1, -1] (DIMACS 1-based)
    let mut literals = memory.alloc::<i32>(2).unwrap();
    device.htod_sync_copy_into(&[1i32, -1], &mut literals).unwrap();

    GpuCnf {
        var_cap: 1,
        clause_cap: 2,
        lit_cap: 2,
        num_vars,
        num_clauses,
        num_lits,
        clause_offsets,
        literals,
    }
}

#[test]
fn test_workspace_reuse_two_solves() {
    let provider = make_provider();
    let config = GpuCdclConfig::default();
    let solver = GpuCdclSolver::new(provider.clone(), config);

    let cnf = build_trivial_unsat(&provider);

    // Create workspace with capacity >= cnf caps.
    let mut ws = solver.new_workspace(cnf.var_cap, cnf.clause_cap).unwrap();

    // Record device pointer of assign buffer to verify reuse.
    let assign_ptr_before = ws.assign.device_ptr().clone();

    // First solve.
    let branch_limit = {
        let memory = provider.memory();
        let mut bl = memory.alloc::<u32>(1).unwrap();
        provider.device().inner().htod_sync_copy_into(&[1u32], &mut bl).unwrap();
        bl
    };
    solver
        .solve_expect_unsat_with_branch_limit_ws(&mut ws, &cnf, &branch_limit)
        .expect("first solve should return UNSAT");

    // Verify same device pointer (workspace reused, not reallocated).
    let assign_ptr_after = ws.assign.device_ptr().clone();
    assert_eq!(
        assign_ptr_before, assign_ptr_after,
        "workspace should reuse the same device buffers across solves"
    );

    // Second solve (same workspace, same CNF).
    solver
        .solve_expect_unsat_with_branch_limit_ws(&mut ws, &cnf, &branch_limit)
        .expect("second solve should return UNSAT");
}

#[test]
fn test_workspace_capacity_overflow() {
    let provider = make_provider();
    let config = GpuCdclConfig::default();
    let solver = GpuCdclSolver::new(provider.clone(), config);

    // Create workspace with tiny capacity (var_cap=1, clause_cap=1).
    let mut ws = solver.new_workspace(1, 1).unwrap();

    // Build a CNF that exceeds var_cap.
    let cnf = {
        let memory = provider.memory();
        let device = provider.device().inner();
        let mut num_vars = memory.alloc::<u32>(1).unwrap();
        device.htod_sync_copy_into(&[10u32], &mut num_vars).unwrap();
        let mut num_clauses = memory.alloc::<u32>(1).unwrap();
        device.htod_sync_copy_into(&[1u32], &mut num_clauses).unwrap();
        let mut num_lits = memory.alloc::<u32>(1).unwrap();
        device.htod_sync_copy_into(&[1u32], &mut num_lits).unwrap();
        let mut clause_offsets = memory.alloc::<u32>(11 + 1).unwrap();
        device.htod_sync_copy_into(&vec![0u32; 12], &mut clause_offsets).unwrap();
        let mut literals = memory.alloc::<i32>(1).unwrap();
        device.htod_sync_copy_into(&[1i32], &mut literals).unwrap();
        GpuCnf {
            var_cap: 10,  // exceeds workspace var_cap of 1
            clause_cap: 1,
            lit_cap: 1,
            num_vars,
            num_clauses,
            num_lits,
            clause_offsets,
            literals,
        }
    };

    let branch_limit = {
        let memory = provider.memory();
        let mut bl = memory.alloc::<u32>(1).unwrap();
        provider.device().inner().htod_sync_copy_into(&[1u32], &mut bl).unwrap();
        bl
    };

    let result = solver.solve_expect_unsat_with_branch_limit_ws(&mut ws, &cnf, &branch_limit);
    assert!(result.is_err(), "should error on capacity overflow");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("exceeds workspace"),
        "error should mention workspace overflow, got: {}",
        msg
    );
}
```

**Step 2: Run tests to verify**

Run: `cargo test -p xlog-solve --test gpu_cdcl_workspace --release -- --nocapture 2>&1 | tail -20`
Expected: 2 tests pass

**Step 3: Commit**

```bash
git add crates/xlog-solve/tests/gpu_cdcl_workspace.rs
git commit -m "test(solve): workspace reuse correctness and capacity overflow tests"
```

---

### Task 5: Add `incremental_verify` to `GpuCompileConfig` and `reuse_workspace` to `GpuEquivalenceConfig`

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_d4.rs:49-67` (add field to `GpuCompileConfig`)
- Modify: `crates/xlog-prob/src/compilation/validation.rs:21` (add field to `GpuEquivalenceConfig`)
- Modify: `crates/xlog-prob/src/compilation/mod.rs:135,305` (propagate)
- Modify: All `GpuCompileConfig { ... }` struct literal sites (add `incremental_verify: false`)

**Step 1: Add `incremental_verify` to `GpuCompileConfig`**

In `crates/xlog-prob/src/compilation/gpu_d4.rs`, after `cdcl_conflict_budget` (line 66), add:

```rust
    /// Enable workspace reuse in the equivalence verifier (amortizes arena allocation).
    pub incremental_verify: bool,
```

**Step 2: Add `reuse_workspace` to `GpuEquivalenceConfig`**

In `crates/xlog-prob/src/compilation/validation.rs`, change line 21 from:

```rust
pub struct GpuEquivalenceConfig {
    pub cdcl: GpuCdclConfig,
}
```

to:

```rust
pub struct GpuEquivalenceConfig {
    pub cdcl: GpuCdclConfig,
    pub reuse_workspace: bool,
}
```

**Step 3: Update the two propagation sites in `mod.rs`**

Change line 135 from:
```rust
        GpuEquivalenceConfig { cdcl },
```
to:
```rust
        GpuEquivalenceConfig { cdcl, reuse_workspace: config.incremental_verify },
```

Change line 305 from:
```rust
        GpuEquivalenceConfig { cdcl },
```
to:
```rust
        GpuEquivalenceConfig { cdcl, reuse_workspace: config.incremental_verify },
```

**Step 4: Update all `GpuCompileConfig` struct literal sites**

Add `incremental_verify: false` to every struct literal. Complete list (18 sites):

| File | Line(s) |
|------|---------|
| `crates/xlog-prob/src/compilation/gpu_d4.rs` | 1310, 1735, 2028, 2328, 2617, 2903, 3184, 3462 |
| `crates/xlog-prob/src/exact.rs` | 1801 |
| `crates/xlog-prob/tests/gpu_cache_compile_and_verify.rs` | 30 |
| `crates/xlog-prob/tests/cdcl_q2_status.rs` | 134 |
| `crates/xlog-prob/tests/gpu_d4_var_presence.rs` | 58, 251, 370, 503 |
| `crates/xlog-prob/tests/cdcl_q1_status_simple.rs` | 74 |
| `crates/xlog-prob/tests/gpu_d4_compile_and_verify.rs` | 39 |

For each, add `incremental_verify: false,` as the last field. Example for `gpu_d4.rs:1310`:

```rust
let config = super::GpuCompileConfig {
    frontier_depth: 0,
    max_frontier_items: 1,
    max_depth: 8,
    cdcl_restart_interval: 128,
    cdcl_learned_bytes: 1 << 20,
    cdcl_conflict_budget: None,
    smooth_node_cap: 256,
    smooth_edge_cap: 512,
    incremental_verify: false,  // ← add this
};
```

**Step 5: Run compilation check**

Run: `cargo check -p xlog-prob --release 2>&1 | head -20`
Expected: compiles clean

**Step 6: Commit**

```bash
git add crates/xlog-prob/src/compilation/gpu_d4.rs \
       crates/xlog-prob/src/compilation/validation.rs \
       crates/xlog-prob/src/compilation/mod.rs \
       crates/xlog-prob/src/exact.rs \
       crates/xlog-prob/tests/
git commit -m "feat(prob): add incremental_verify config and reuse_workspace propagation"
```

---

### Task 6: Integrate workspace reuse in `check_equivalence_gpu` and `check_equivalence_gpu_gated`

**Files:**
- Modify: `crates/xlog-prob/src/compilation/validation.rs` (both functions)

**Step 1: Update `check_equivalence_gpu_gated` (lines 812-904)**

Replace lines 876-895 (the solver creation and two solve calls) with:

```rust
    let solver = GpuCdclSolver::new(provider.clone(), config.cdcl);
    if config.reuse_workspace {
        let max_var_cap = std::cmp::max(q1.var_cap, q2.var_cap);
        let max_clause_cap = std::cmp::max(q1.clause_cap, q2.clause_cap);
        let mut ws = solver.new_workspace(max_var_cap, max_clause_cap)?;
        solver.solve_expect_unsat_with_branch_limit_gated_ws(
            &mut ws,
            &q1,
            compile_needed,
            phi_decision_var_limit,
        )?;
        #[cfg(debug_assertions)]
        {
            provider.device().synchronize().map_err(|e| {
                XlogError::Kernel(format!("sync after solve_expect_unsat(q1) failed: {}", e))
            })?;
            eprintln!("[xlog-prob] equivalence: solve_expect_unsat q2");
        }
        solver.solve_expect_unsat_with_decision_ranges_gated_ws(
            &mut ws,
            &q2,
            compile_needed,
            phi_decision_var_limit,
            &q2_unsat_var_base,
            &phi.num_clauses,
        )?;
    } else {
        solver.solve_expect_unsat_with_branch_limit_gated(
            &q1,
            compile_needed,
            phi_decision_var_limit,
        )?;
        #[cfg(debug_assertions)]
        {
            provider.device().synchronize().map_err(|e| {
                XlogError::Kernel(format!("sync after solve_expect_unsat(q1) failed: {}", e))
            })?;
            eprintln!("[xlog-prob] equivalence: solve_expect_unsat q2");
        }
        solver.solve_expect_unsat_with_decision_ranges_gated(
            &q2,
            compile_needed,
            phi_decision_var_limit,
            &q2_unsat_var_base,
            &phi.num_clauses,
        )?;
    }
```

**Step 2: Update `check_equivalence_gpu` (lines 747-774)**

Same pattern. Replace lines 763-772 with:

```rust
    let solver = GpuCdclSolver::new(provider.clone(), config.cdcl);
    if config.reuse_workspace {
        let max_var_cap = std::cmp::max(queries.q1.var_cap, queries.q2.var_cap);
        let max_clause_cap = std::cmp::max(queries.q1.clause_cap, queries.q2.clause_cap);
        let mut ws = solver.new_workspace(max_var_cap, max_clause_cap)?;
        solver.solve_expect_unsat_with_branch_limit_ws(
            &mut ws, &queries.q1, phi_decision_var_limit,
        )?;
        solver.solve_expect_unsat_with_decision_ranges_ws(
            &mut ws,
            &queries.q2,
            phi_decision_var_limit,
            &queries.q2_unsat_var_base,
            &phi.num_clauses,
        )?;
    } else {
        solver.solve_expect_unsat_with_branch_limit(&queries.q1, phi_decision_var_limit)?;
        solver.solve_expect_unsat_with_decision_ranges(
            &queries.q2,
            phi_decision_var_limit,
            &queries.q2_unsat_var_base,
            &phi.num_clauses,
        )?;
    }
```

**Step 3: Add import**

At the top of `validation.rs`, ensure `GpuCdclWorkspace` does not need to be imported (it's only constructed internally by `solver.new_workspace()` and used locally).

However, `std::cmp::max` must be in scope. Check if it's already imported; if not, add `use std::cmp;` and use `cmp::max(...)`.

**Step 4: Run compilation check**

Run: `cargo check -p xlog-prob --release 2>&1 | head -20`
Expected: compiles clean

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/compilation/validation.rs
git commit -m "feat(prob): integrate workspace reuse in equivalence checker"
```

---

### Task 7: Integration test — workspace-enabled compilation

**Files:**
- Create: `crates/xlog-prob/tests/gpu_workspace_verify.rs`

**Step 1: Write the test**

Model this after `crates/xlog-prob/tests/gpu_d4_compile_and_verify.rs` (line 39). Use the same small CNF but set `incremental_verify: true`.

```rust
//! Integration test: equivalence verification with workspace reuse enabled.

use std::sync::Arc;
use xlog_cuda::CudaKernelProvider;
use xlog_prob::compilation::{compile_gpu_d4_and_verify, GpuCompileConfig};
use xlog_prob::cnf::build_gpu_cnf;
// The test needs a small PIR → CNF → compile → verify pipeline.
// Reference: gpu_d4_compile_and_verify.rs for the setup pattern.

#[test]
fn test_compile_and_verify_with_workspace_reuse() {
    // NOTE: The implementer must follow the exact setup from
    // gpu_d4_compile_and_verify.rs (provider creation, CNF construction,
    // compile_gpu_d4_and_verify call) but with incremental_verify: true.
    //
    // The goal is to verify that the workspace-enabled path produces the
    // same result (UNSAT on both q1 and q2) as the default path.
    //
    // Steps:
    // 1. Copy the provider/CNF setup from gpu_d4_compile_and_verify.rs
    // 2. Set incremental_verify: true in GpuCompileConfig
    // 3. Call compile_gpu_d4_and_verify
    // 4. Assert it succeeds (no panic, no error)
    //
    // Then run the same test with incremental_verify: false to confirm
    // both paths produce identical success.

    // [IMPLEMENTER: Read gpu_d4_compile_and_verify.rs and replicate the
    //  test setup. The only change is adding incremental_verify: true.]
}
```

**Step 2: Run the test**

Run: `cargo test -p xlog-prob --test gpu_workspace_verify --release -- --nocapture 2>&1 | tail -20`
Expected: 1 test passes

**Step 3: Commit**

```bash
git add crates/xlog-prob/tests/gpu_workspace_verify.rs
git commit -m "test(prob): integration test for workspace-enabled equivalence verification"
```

---

### Task 8: Regression suite + changelog

**Files:**
- Modify: `CHANGELOG.md`

**Step 1: Run full Rust test suite**

Run: `LD_LIBRARY_PATH=/usr/lib/wsl/lib cargo test --workspace --all-targets --exclude pyxlog --release 2>&1 | tail -30`
Expected: all tests pass, 0 failures

**Step 2: Run Python test suite**

Run: `.venv/bin/python -m pytest python/tests/ -v --tb=short 2>&1 | tail -30`
Expected: all tests pass (no behavioral change — default is `incremental_verify: false`)

**Step 3: Update CHANGELOG.md**

Add under the latest unreleased section:

```markdown
### Added
- `GpuCdclWorkspace` for reusable solver arena across multiple CDCL solves (P3)
- `GpuCdclSolver::new_workspace()` constructor
- `solve_expect_unsat_*_ws` method variants for workspace-backed solving
- `GpuCompileConfig.incremental_verify` opt-in for workspace reuse in equivalence verification
- `GpuEquivalenceConfig.reuse_workspace` config field
```

**Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add P3 incremental verifier entries to changelog"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | `GpuCdclWorkspace` struct + `new_workspace()` | `gpu_cdcl.rs` |
| 2 | `launch_cdcl_with_workspace_gated` internal | `gpu_cdcl.rs` |
| 3 | 4 public `_ws` solver methods | `gpu_cdcl.rs` |
| 4 | Unit tests (reuse + overflow) | `gpu_cdcl_workspace.rs` |
| 5 | Config fields + propagation (18 struct sites) | `gpu_d4.rs`, `validation.rs`, `mod.rs`, 6 test files |
| 6 | Integration in `check_equivalence_gpu*` | `validation.rs` |
| 7 | Integration test | `gpu_workspace_verify.rs` |
| 8 | Regression suite + changelog | `CHANGELOG.md` |

**Dependencies:** Tasks 1→2→3 (sequential, same file). Task 4 depends on 3. Task 5 is independent of 1-4. Task 6 depends on 3+5. Task 7 depends on 6. Task 8 depends on all.
