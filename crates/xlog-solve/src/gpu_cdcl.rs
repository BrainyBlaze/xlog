use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{sat_kernels, SAT_MODULE};
use xlog_cuda::CudaKernelProvider;

use crate::gpu_cnf::GpuCnf;

// Must match kernels/sat.cu.
const SAT_STATUS_UNSAT: i32 = 0;
const SAT_STATUS_SAT: i32 = 1;

struct GpuCdclRun {
    assignment: TrackedCudaSlice<i8>,
    // Scratch buffers used only by sat_cdcl_solve, but must stay alive until the solver kernel completes.
    #[allow(dead_code)]
    decision_heap: TrackedCudaSlice<u32>,
    #[allow(dead_code)]
    decision_heap_pos: TrackedCudaSlice<u32>,

    learned_offsets: TrackedCudaSlice<u32>,
    learned_lits: TrackedCudaSlice<i32>,
    proof_offsets: TrackedCudaSlice<u32>,
    proof_data: TrackedCudaSlice<u32>,

    out_status: TrackedCudaSlice<i32>,
    out_error: TrackedCudaSlice<i32>,
    out_learned_count: TrackedCudaSlice<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct GpuCdclConfig {
    pub max_learned_clauses: u32,
    pub max_learned_lits: u32,
    pub max_proof_u32: u32,
    pub restart_base: u32,
    pub reduce_interval: u32,
}

impl Default for GpuCdclConfig {
    fn default() -> Self {
        Self {
            max_learned_clauses: 32_768,
            max_learned_lits: 262_144,
            max_proof_u32: 1_048_576,
            restart_base: 100,
            reduce_interval: 2000,
        }
    }
}

pub struct GpuCdclSolver {
    provider: Arc<CudaKernelProvider>,
    config: GpuCdclConfig,
}

/// Raw CDCL outputs (device-resident) for debugging and research.
///
/// Production verifier paths should prefer `solve_expect_sat*` / `solve_expect_unsat*`,
/// which validate results on GPU and enforce expectations without host reads.
pub struct GpuCdclRawOutput {
    pub assignment: TrackedCudaSlice<i8>,
    pub out_status: TrackedCudaSlice<i32>,
    pub out_error: TrackedCudaSlice<i32>,
    pub out_learned_count: TrackedCudaSlice<u32>,
}

impl GpuCdclSolver {
    pub fn new(provider: Arc<CudaKernelProvider>, config: GpuCdclConfig) -> Self {
        Self { provider, config }
    }

    fn alloc_u32_scalar(&self, value: u32) -> Result<TrackedCudaSlice<u32>> {
        let memory = self.provider.memory();
        let mut gate = memory.alloc::<u32>(1)?;
        self.provider
            .device()
            .inner()
            .htod_sync_copy_into(&[value], &mut gate)
            .map_err(|e| XlogError::Kernel(format!("GpuCdclSolver gate upload failed: {}", e)))?;
        Ok(gate)
    }

    fn launch_cdcl_with_decision_ranges_gated(
        &self,
        cnf: &GpuCnf,
        compile_needed: &TrackedCudaSlice<u32>,
        decision_base_limit: &TrackedCudaSlice<u32>,
        decision_extra_base: &TrackedCudaSlice<u32>,
        decision_extra_count: &TrackedCudaSlice<u32>,
    ) -> Result<GpuCdclRun> {
        let num_vars_cap = cnf.var_cap as usize;
        let num_clauses_cap = cnf.clause_cap as usize;

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

        let max_learned_clauses = self.config.max_learned_clauses as usize;
        let max_learned_lits = self.config.max_learned_lits as usize;
        let max_proof_u32 = self.config.max_proof_u32 as usize;

        let max_total_clauses = num_clauses_cap
            .checked_add(max_learned_clauses)
            .ok_or_else(|| XlogError::Kernel("SAT clause capacity overflow".to_string()))?;

        let memory = self.provider.memory();

        // Variable state
        let mut assign = memory.alloc::<i8>(num_vars_cap + 1)?;
        let mut level = memory.alloc::<u32>(num_vars_cap + 1)?;
        let mut reason = memory.alloc::<i32>(num_vars_cap + 1)?;
        let mut var_activity = memory.alloc::<u32>(num_vars_cap + 1)?;
        let mut var_phase = memory.alloc::<i8>(num_vars_cap + 1)?;
        let mut decision_heap = memory.alloc::<u32>(num_vars_cap + 1)?;
        let mut decision_heap_pos = memory.alloc::<u32>(num_vars_cap + 1)?;

        // Trail / levels
        let mut trail = memory.alloc::<i32>(num_vars_cap + 1)?;
        let mut trail_lim = memory.alloc::<u32>(num_vars_cap + 1)?;

        // Analysis scratch
        let mut seen = memory.alloc::<u8>(num_vars_cap + 1)?;
        let mut learnt_tmp = memory.alloc::<i32>(num_vars_cap + 1)?;
        let mut proof_vars_tmp = memory.alloc::<u32>(num_vars_cap + 1)?;
        let mut proof_reason_tmp = memory.alloc::<u32>(num_vars_cap + 1)?;

        // Watched literals
        let mut watch0_pos = memory.alloc::<u32>(max_total_clauses)?;
        let mut watch1_pos = memory.alloc::<u32>(max_total_clauses)?;
        let mut watch_head = memory.alloc::<i32>(2 * num_vars_cap)?;
        let mut watch_next = memory.alloc::<i32>(2 * max_total_clauses)?;
        let mut watch_prev = memory.alloc::<i32>(2 * max_total_clauses)?;

        // Learned clause arena
        let mut learned_offsets = memory.alloc::<u32>(max_learned_clauses + 1)?;
        let mut learned_lits = memory.alloc::<i32>(max_learned_lits)?;
        let mut learned_deleted = memory.alloc::<u8>(max_learned_clauses)?;
        let mut learned_lbd = memory.alloc::<u32>(max_learned_clauses)?;
        let mut learned_activity = memory.alloc::<u32>(max_learned_clauses)?;
        let mut learned_locked = memory.alloc::<u8>(max_learned_clauses)?;

        // Proof trace arena
        let mut proof_offsets = memory.alloc::<u32>(max_learned_clauses + 1)?;
        let mut proof_data = memory.alloc::<u32>(max_proof_u32)?;

        // Device-resident outputs
        let mut out_status = memory.alloc::<i32>(1)?;
        let mut out_error = memory.alloc::<i32>(1)?;
        let mut out_learned_count = memory.alloc::<u32>(1)?;

        let sat_fn = self
            .provider
            .device()
            .inner()
            .get_func(SAT_MODULE, sat_kernels::SAT_CDCL_SOLVE)
            .ok_or_else(|| XlogError::Kernel("sat_cdcl_solve kernel not found".to_string()))?;

        // IMPORTANT: When launching with an explicit `Vec<*mut c_void>` parameter list, scalar
        // kernel arguments MUST be backed by stable host storage until `cuLaunchKernel` copies
        // them. Do not pass temporaries like `self.config.restart_base.as_kernel_param()`.
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
            (&mut assign).as_kernel_param(),
            (&mut level).as_kernel_param(),
            (&mut reason).as_kernel_param(),
            (&mut var_activity).as_kernel_param(),
            (&mut var_phase).as_kernel_param(),
            (&mut decision_heap).as_kernel_param(),
            (&mut decision_heap_pos).as_kernel_param(),
            (&mut trail).as_kernel_param(),
            (&mut trail_lim).as_kernel_param(),
            (&mut seen).as_kernel_param(),
            (&mut learnt_tmp).as_kernel_param(),
            (&mut proof_vars_tmp).as_kernel_param(),
            (&mut proof_reason_tmp).as_kernel_param(),
            (&mut watch0_pos).as_kernel_param(),
            (&mut watch1_pos).as_kernel_param(),
            (&mut watch_head).as_kernel_param(),
            (&mut watch_next).as_kernel_param(),
            (&mut watch_prev).as_kernel_param(),
            (&mut learned_offsets).as_kernel_param(),
            (&mut learned_lits).as_kernel_param(),
            (&mut learned_deleted).as_kernel_param(),
            (&mut learned_lbd).as_kernel_param(),
            (&mut learned_activity).as_kernel_param(),
            (&mut learned_locked).as_kernel_param(),
            (&mut proof_offsets).as_kernel_param(),
            (&mut proof_data).as_kernel_param(),
            (&mut out_status).as_kernel_param(),
            (&mut out_error).as_kernel_param(),
            (&mut out_learned_count).as_kernel_param(),
        ];

        unsafe {
            sat_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    // One block per SAT instance; use a full block so sat_cdcl_solve can do
                    // block-parallel propagation and initialization.
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut params,
            )
        }
        .map_err(|e| XlogError::Kernel(format!("Failed to launch SAT solver kernel: {}", e)))?;

        Ok(GpuCdclRun {
            assignment: assign,
            decision_heap,
            decision_heap_pos,
            learned_offsets,
            learned_lits,
            proof_offsets,
            proof_data,
            out_status,
            out_error,
            out_learned_count,
        })
    }

    /// Launch CDCL and return raw device outputs without enforcing SAT/UNSAT on device.
    ///
    /// This is intentionally **not** used in production verifier paths. It exists so tests and
    /// debugging tools can inspect `out_status/out_error` without modifying kernel behavior.
    pub fn solve_raw_with_branch_limit(
        &self,
        cnf: &GpuCnf,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<GpuCdclRawOutput> {
        let compile_needed = self.alloc_u32_scalar(1)?;
        self.solve_raw_with_branch_limit_gated(cnf, &compile_needed, branch_var_limit)
    }

    /// Gated variant of `solve_raw_with_branch_limit`.
    pub fn solve_raw_with_branch_limit_gated(
        &self,
        cnf: &GpuCnf,
        compile_needed: &TrackedCudaSlice<u32>,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<GpuCdclRawOutput> {
        let zero = self.alloc_u32_scalar(0)?;
        let run = self.launch_cdcl_with_decision_ranges_gated(
            cnf,
            compile_needed,
            branch_var_limit,
            &zero,
            &zero,
        )?;
        // Ensure kernel completion so `out_*` are valid for inspection.
        self.provider.device().synchronize()?;

        let GpuCdclRun {
            assignment,
            out_status,
            out_error,
            out_learned_count,
            ..
        } = run;

        Ok(GpuCdclRawOutput {
            assignment,
            out_status,
            out_error,
            out_learned_count,
        })
    }

    /// Launch CDCL and return raw device outputs without enforcing SAT/UNSAT on device, using an
    /// explicit decision variable set:
    /// - decision vars include all `v` in `1..=decision_base_limit[0]` and
    ///   `decision_extra_base[0]..(decision_extra_base[0] + decision_extra_count[0] - 1)`.
    ///
    /// Production verifier paths should prefer `solve_expect_*` methods which enforce results on
    /// GPU without host reads.
    pub fn solve_raw_with_decision_ranges(
        &self,
        cnf: &GpuCnf,
        decision_base_limit: &TrackedCudaSlice<u32>,
        decision_extra_base: &TrackedCudaSlice<u32>,
        decision_extra_count: &TrackedCudaSlice<u32>,
    ) -> Result<GpuCdclRawOutput> {
        let compile_needed = self.alloc_u32_scalar(1)?;
        self.solve_raw_with_decision_ranges_gated(
            cnf,
            &compile_needed,
            decision_base_limit,
            decision_extra_base,
            decision_extra_count,
        )
    }

    /// Gated variant of `solve_raw_with_decision_ranges`.
    pub fn solve_raw_with_decision_ranges_gated(
        &self,
        cnf: &GpuCnf,
        compile_needed: &TrackedCudaSlice<u32>,
        decision_base_limit: &TrackedCudaSlice<u32>,
        decision_extra_base: &TrackedCudaSlice<u32>,
        decision_extra_count: &TrackedCudaSlice<u32>,
    ) -> Result<GpuCdclRawOutput> {
        let run = self.launch_cdcl_with_decision_ranges_gated(
            cnf,
            compile_needed,
            decision_base_limit,
            decision_extra_base,
            decision_extra_count,
        )?;
        // Ensure kernel completion so `out_*` are valid for inspection.
        self.provider.device().synchronize()?;

        let GpuCdclRun {
            assignment,
            out_status,
            out_error,
            out_learned_count,
            ..
        } = run;

        Ok(GpuCdclRawOutput {
            assignment,
            out_status,
            out_error,
            out_learned_count,
        })
    }

    /// Solve and enforce SAT entirely on GPU (no device->host reads).
    pub fn solve_expect_sat(&self, cnf: &GpuCnf) -> Result<TrackedCudaSlice<i8>> {
        let compile_needed = self.alloc_u32_scalar(1)?;
        self.solve_expect_sat_gated(cnf, &compile_needed)
    }

    /// Solve and enforce SAT entirely on GPU (no device->host reads),
    /// skipping all GPU work if `compile_needed` is 0.
    pub fn solve_expect_sat_gated(
        &self,
        cnf: &GpuCnf,
        compile_needed: &TrackedCudaSlice<u32>,
    ) -> Result<TrackedCudaSlice<i8>> {
        self.solve_expect_sat_with_branch_limit_gated(cnf, compile_needed, &cnf.num_vars)
    }

    /// Solve and enforce SAT entirely on GPU (no device->host reads), using an explicit decision
    /// variable set:
    /// - decision vars include all `v` in `1..=decision_base_limit[0]` and
    ///   `decision_extra_base[0]..(decision_extra_base[0] + decision_extra_count[0] - 1)`.
    pub fn solve_expect_sat_with_decision_ranges(
        &self,
        cnf: &GpuCnf,
        decision_base_limit: &TrackedCudaSlice<u32>,
        decision_extra_base: &TrackedCudaSlice<u32>,
        decision_extra_count: &TrackedCudaSlice<u32>,
    ) -> Result<TrackedCudaSlice<i8>> {
        let compile_needed = self.alloc_u32_scalar(1)?;
        self.solve_expect_sat_with_decision_ranges_gated(
            cnf,
            &compile_needed,
            decision_base_limit,
            decision_extra_base,
            decision_extra_count,
        )
    }

    /// Gated variant of `solve_expect_sat_with_decision_ranges`.
    pub fn solve_expect_sat_with_decision_ranges_gated(
        &self,
        cnf: &GpuCnf,
        compile_needed: &TrackedCudaSlice<u32>,
        decision_base_limit: &TrackedCudaSlice<u32>,
        decision_extra_base: &TrackedCudaSlice<u32>,
        decision_extra_count: &TrackedCudaSlice<u32>,
    ) -> Result<TrackedCudaSlice<i8>> {
        #[cfg(debug_assertions)]
        let trace = std::env::var_os("XLOG_CDCL_TRACE").is_some();
        #[cfg(debug_assertions)]
        let t0 = std::time::Instant::now();

        let run = self.launch_cdcl_with_decision_ranges_gated(
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
                    (compile_needed, &run.out_status, &run.out_error, SAT_STATUS_SAT),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("Failed to launch sat_assert_status: {}", e))
                })?;
        }
        // Fail-fast if the solver did not produce SAT.
        self.provider.device().synchronize()?;
        #[cfg(debug_assertions)]
        if trace {
            eprintln!("[xlog-solve] cdcl(sat) time: {:?}", t0.elapsed());
        }

        let mut out_ok = memory.alloc::<i32>(1)?;
        let check_fn = device
            .get_func(SAT_MODULE, sat_kernels::SAT_CHECK_MODEL)
            .ok_or_else(|| XlogError::Kernel("sat_check_model kernel not found".to_string()))?;
        unsafe {
            check_fn
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        compile_needed,
                        &cnf.clause_offsets,
                        &cnf.literals,
                        &cnf.num_clauses,
                        &run.assignment,
                        &mut out_ok,
                    ),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("Failed to launch SAT model check: {}", e))
                })?;
        }

        let assert_ok_fn = device
            .get_func(SAT_MODULE, sat_kernels::SAT_ASSERT_OK)
            .ok_or_else(|| XlogError::Kernel("sat_assert_ok kernel not found".to_string()))?;
        unsafe {
            assert_ok_fn
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (compile_needed, &out_ok),
                )
                .map_err(|e| XlogError::Kernel(format!("Failed to launch sat_assert_ok: {}", e)))?;
        }
        self.provider.device().synchronize()?;
        #[cfg(debug_assertions)]
        if trace {
            eprintln!("[xlog-solve] cdcl(sat)+model_check time: {:?}", t0.elapsed());
        }

        Ok(run.assignment)
    }

    /// Solve and enforce SAT entirely on GPU (no device->host reads),
    /// restricting branching to variables in `1..=branch_var_limit[0]`.
    pub fn solve_expect_sat_with_branch_limit(
        &self,
        cnf: &GpuCnf,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<TrackedCudaSlice<i8>> {
        let compile_needed = self.alloc_u32_scalar(1)?;
        self.solve_expect_sat_with_branch_limit_gated(cnf, &compile_needed, branch_var_limit)
    }

    /// Solve and enforce SAT entirely on GPU (no device->host reads),
    /// skipping all GPU work if `compile_needed` is 0, and restricting branching to
    /// variables in `1..=branch_var_limit[0]`.
    pub fn solve_expect_sat_with_branch_limit_gated(
        &self,
        cnf: &GpuCnf,
        compile_needed: &TrackedCudaSlice<u32>,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<TrackedCudaSlice<i8>> {
        let zero = self.alloc_u32_scalar(0)?;
        self.solve_expect_sat_with_decision_ranges_gated(
            cnf,
            compile_needed,
            branch_var_limit,
            &zero,
            &zero,
        )
    }

    /// Solve and enforce UNSAT entirely on GPU (no device->host reads).
    pub fn solve_expect_unsat(&self, cnf: &GpuCnf) -> Result<()> {
        let compile_needed = self.alloc_u32_scalar(1)?;
        self.solve_expect_unsat_gated(cnf, &compile_needed)
    }

    /// Solve and enforce UNSAT entirely on GPU (no device->host reads),
    /// skipping all GPU work if `compile_needed` is 0.
    pub fn solve_expect_unsat_gated(
        &self,
        cnf: &GpuCnf,
        compile_needed: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        self.solve_expect_unsat_with_branch_limit_gated(cnf, compile_needed, &cnf.num_vars)
    }

    /// Solve and enforce UNSAT entirely on GPU (no device->host reads),
    /// restricting branching to variables in `1..=branch_var_limit[0]`.
    pub fn solve_expect_unsat_with_branch_limit(
        &self,
        cnf: &GpuCnf,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        let compile_needed = self.alloc_u32_scalar(1)?;
        self.solve_expect_unsat_with_branch_limit_gated(cnf, &compile_needed, branch_var_limit)
    }

    /// Solve and enforce UNSAT entirely on GPU (no device->host reads), using an explicit decision
    /// variable set:
    /// - decision vars include all `v` in `1..=decision_base_limit[0]` and
    ///   `decision_extra_base[0]..(decision_extra_base[0] + decision_extra_count[0] - 1)`.
    pub fn solve_expect_unsat_with_decision_ranges(
        &self,
        cnf: &GpuCnf,
        decision_base_limit: &TrackedCudaSlice<u32>,
        decision_extra_base: &TrackedCudaSlice<u32>,
        decision_extra_count: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        let compile_needed = self.alloc_u32_scalar(1)?;
        self.solve_expect_unsat_with_decision_ranges_gated(
            cnf,
            &compile_needed,
            decision_base_limit,
            decision_extra_base,
            decision_extra_count,
        )
    }

    /// Solve and enforce UNSAT entirely on GPU (no device->host reads),
    /// skipping all GPU work if `compile_needed` is 0, and restricting branching to
    /// variables in `1..=branch_var_limit[0]`.
    pub fn solve_expect_unsat_with_branch_limit_gated(
        &self,
        cnf: &GpuCnf,
        compile_needed: &TrackedCudaSlice<u32>,
        branch_var_limit: &TrackedCudaSlice<u32>,
    ) -> Result<()> {
        let zero = self.alloc_u32_scalar(0)?;
        self.solve_expect_unsat_with_decision_ranges_gated(
            cnf,
            compile_needed,
            branch_var_limit,
            &zero,
            &zero,
        )
    }

    /// Solve and enforce UNSAT entirely on GPU (no device->host reads), using an explicit decision
    /// variable set:
    /// - decision vars include all `v` in `1..=decision_base_limit[0]` and
    ///   `decision_extra_base[0]..(decision_extra_base[0] + decision_extra_count[0] - 1)`.
    pub fn solve_expect_unsat_with_decision_ranges_gated(
        &self,
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

        let run = self.launch_cdcl_with_decision_ranges_gated(
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
                    (compile_needed, &run.out_status, &run.out_error, SAT_STATUS_UNSAT),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("Failed to launch sat_assert_status: {}", e))
                })?;
        }
        // Fail-fast if the solver did not produce UNSAT.
        self.provider.device().synchronize()?;
        #[cfg(debug_assertions)]
        if trace {
            eprintln!("[xlog-solve] cdcl(unsat) time: {:?}", t0.elapsed());
        }

        let mut out_ok = memory.alloc::<i32>(1)?;
        device
            .htod_sync_copy_into(&[1i32], &mut out_ok)
            .map_err(|e| XlogError::Kernel(format!("Failed to init proof out_ok: {}", e)))?;

        // sat_proof_check uses scratch buffers sized to `scratch_cap` per verifier block. To keep
        // proof checking fast on large instances, allocate multiple scratch regions and verify
        // learned clauses in parallel across blocks.
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
                Err(e) => {
                    last_alloc_err = Some(e);
                    continue;
                }
            };
            let b = match memory.alloc::<i32>(len) {
                Ok(buf) => buf,
                Err(e) => {
                    last_alloc_err = Some(e);
                    // Drop `a` before retrying with a smaller configuration.
                    drop(a);
                    continue;
                }
            };
            let m = match memory.alloc::<u32>(len) {
                Ok(buf) => buf,
                Err(e) => {
                    last_alloc_err = Some(e);
                    drop(a);
                    drop(b);
                    continue;
                }
            };

            proof_blocks = blocks;
            scratch_a = Some(a);
            scratch_b = Some(b);
            scratch_map = Some(m);
            break;
        }
        let mut scratch_a =
            scratch_a.ok_or_else(|| last_alloc_err.unwrap_or_else(|| {
                XlogError::Kernel("Failed to allocate proof scratch buffers".to_string())
            }))?;
        let mut scratch_b =
            scratch_b.ok_or_else(|| XlogError::Kernel("Missing proof scratch buffer".to_string()))?;
        let mut scratch_map =
            scratch_map.ok_or_else(|| XlogError::Kernel("Missing proof scratch map buffer".to_string()))?;
        device
            .memset_zeros(&mut scratch_map)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero proof scratch map: {}", e)))?;
        #[cfg(debug_assertions)]
        if trace {
            eprintln!("[xlog-solve] proof_check blocks: {}", proof_blocks);
        }
        #[cfg(debug_assertions)]
        let t_mark = std::time::Instant::now();

        let needed_cap_u32 = self.config.max_learned_clauses;
        let needed_cap = needed_cap_u32 as usize;
        let mut needed = memory.alloc::<u8>(needed_cap)?;
        device
            .memset_zeros(&mut needed)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero proof needed mask: {}", e)))?;

        let mark_needed_fn = device
            .get_func(SAT_MODULE, sat_kernels::SAT_PROOF_MARK_NEEDED)
            .ok_or_else(|| {
                XlogError::Kernel("sat_proof_mark_needed kernel not found".to_string())
            })?;
        let mut mark_params: Vec<*mut c_void> = vec![
            compile_needed.as_kernel_param(),
            (&cnf.num_clauses).as_kernel_param(),
            (&run.out_learned_count).as_kernel_param(),
            (&run.proof_offsets).as_kernel_param(),
            (&run.proof_data).as_kernel_param(),
            needed_cap_u32.as_kernel_param(),
            (&mut needed).as_kernel_param(),
        ];
        unsafe {
            mark_needed_fn
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut mark_params,
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("Failed to launch sat_proof_mark_needed: {}", e))
                })?;
        }
        self.provider.device().synchronize()?;
        #[cfg(debug_assertions)]
        if trace {
            eprintln!("[xlog-solve] proof_mark_needed time: {:?}", t_mark.elapsed());
        }

        let proof_fn = device
            .get_func(SAT_MODULE, sat_kernels::SAT_PROOF_CHECK)
            .ok_or_else(|| XlogError::Kernel("sat_proof_check kernel not found".to_string()))?;
        #[cfg(debug_assertions)]
        let t_proof = std::time::Instant::now();
        let proof_blocks_u32 = u32::try_from(proof_blocks).map_err(|_| {
            XlogError::Kernel("Proof check grid dim exceeds u32::MAX".to_string())
        })?;
        let mut proof_params: Vec<*mut c_void> = vec![
            compile_needed.as_kernel_param(),
            (&cnf.clause_offsets).as_kernel_param(),
            (&cnf.literals).as_kernel_param(),
            (&cnf.num_clauses).as_kernel_param(),
            (&run.learned_offsets).as_kernel_param(),
            (&run.learned_lits).as_kernel_param(),
            (&run.out_learned_count).as_kernel_param(),
            (&run.proof_offsets).as_kernel_param(),
            (&run.proof_data).as_kernel_param(),
            (&needed).as_kernel_param(),
            needed_cap_u32.as_kernel_param(),
            (&mut scratch_a).as_kernel_param(),
            (&mut scratch_b).as_kernel_param(),
            (&mut scratch_map).as_kernel_param(),
            scratch_cap_u32.as_kernel_param(),
            (&mut out_ok).as_kernel_param(),
        ];
        unsafe {
            proof_fn
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (proof_blocks_u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut proof_params,
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("Failed to launch SAT proof check: {}", e))
                })?;
        }

        let assert_ok_fn = device
            .get_func(SAT_MODULE, sat_kernels::SAT_ASSERT_OK)
            .ok_or_else(|| XlogError::Kernel("sat_assert_ok kernel not found".to_string()))?;
        unsafe {
            assert_ok_fn
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (compile_needed, &out_ok),
                )
                .map_err(|e| XlogError::Kernel(format!("Failed to launch sat_assert_ok: {}", e)))?;
        }
        self.provider.device().synchronize()?;
        #[cfg(debug_assertions)]
        if trace {
            eprintln!("[xlog-solve] proof_check time: {:?}", t_proof.elapsed());
            eprintln!("[xlog-solve] cdcl(unsat)+proof_check time: {:?}", t0.elapsed());
        }

        Ok(())
    }
}
