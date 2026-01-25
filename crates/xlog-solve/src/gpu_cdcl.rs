use std::sync::Arc;

use std::ffi::c_void;

use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::CudaKernelProvider;
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{SAT_MODULE, sat_kernels};

use crate::gpu_cnf::GpuCnf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSolveStatus {
    Sat,
    Unsat,
}

pub struct GpuCdclResult {
    pub status: GpuSolveStatus,
    /// Device-resident assignment (len = num_vars + 1, values in {-1,0,1}).
    ///
    /// For SAT, this is a total model. For UNSAT, contents are unspecified.
    pub assignment: TrackedCudaSlice<i8>,
    /// Number of learned clauses produced by the solver kernel.
    pub learned_count: u32,
}

impl std::fmt::Debug for GpuCdclResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuCdclResult")
            .field("status", &self.status)
            .field("learned_count", &self.learned_count)
            .finish()
    }
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

impl GpuCdclSolver {
    pub fn new(provider: Arc<CudaKernelProvider>, config: GpuCdclConfig) -> Self {
        Self { provider, config }
    }

    pub fn solve(&self, cnf: &GpuCnf) -> Result<GpuCdclResult> {
        let num_vars = cnf.num_vars as usize;
        let num_clauses = cnf.num_clauses as usize;

        if num_vars == 0 {
            return Err(XlogError::Compilation("GpuCdclSolver requires num_vars > 0".to_string()));
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

        let max_total_clauses = num_clauses
            .checked_add(max_learned_clauses)
            .ok_or_else(|| XlogError::Kernel("SAT clause capacity overflow".to_string()))?;

        let memory = self.provider.memory();

        // Variable state
        let mut assign = memory.alloc::<i8>(num_vars + 1)?;
        let mut level = memory.alloc::<u32>(num_vars + 1)?;
        let mut reason = memory.alloc::<i32>(num_vars + 1)?;
        let mut var_activity = memory.alloc::<u32>(num_vars + 1)?;
        let mut var_phase = memory.alloc::<i8>(num_vars + 1)?;

        // Trail / levels
        let mut trail = memory.alloc::<i32>(num_vars + 1)?;
        let mut trail_lim = memory.alloc::<u32>(num_vars + 1)?;

        // Analysis scratch
        let mut seen = memory.alloc::<u8>(num_vars + 1)?;
        let mut learnt_tmp = memory.alloc::<i32>(num_vars + 1)?;
        let mut proof_vars_tmp = memory.alloc::<u32>(num_vars + 1)?;
        let mut proof_reason_tmp = memory.alloc::<u32>(num_vars + 1)?;

        // Watched literals
        let mut watch0_pos = memory.alloc::<u32>(max_total_clauses)?;
        let mut watch1_pos = memory.alloc::<u32>(max_total_clauses)?;
        let mut watch_head = memory.alloc::<i32>(2 * num_vars)?;
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

        // Outputs
        let mut out_status = memory.alloc::<i32>(1)?;
        let mut out_error = memory.alloc::<i32>(1)?;
        let mut out_learned_count = memory.alloc::<u32>(1)?;

        let sat_fn = self
            .provider
            .device()
            .inner()
            .get_func(SAT_MODULE, sat_kernels::SAT_CDCL_SOLVE)
            .ok_or_else(|| XlogError::Kernel("sat_cdcl_solve kernel not found".to_string()))?;

        let num_vars_u32 = cnf.num_vars;
        let num_clauses_u32 = cnf.num_clauses;
        let max_learned_clauses_u32 = self.config.max_learned_clauses;
        let max_learned_lits_u32 = self.config.max_learned_lits;
        let max_proof_u32_u32 = self.config.max_proof_u32;
        let restart_base_u32 = self.config.restart_base;
        let reduce_interval_u32 = self.config.reduce_interval;

        let mut params: Vec<*mut c_void> = vec![
            (&cnf.clause_offsets).as_kernel_param(),
            (&cnf.literals).as_kernel_param(),
            num_vars_u32.as_kernel_param(),
            num_clauses_u32.as_kernel_param(),
            max_learned_clauses_u32.as_kernel_param(),
            max_learned_lits_u32.as_kernel_param(),
            max_proof_u32_u32.as_kernel_param(),
            restart_base_u32.as_kernel_param(),
            reduce_interval_u32.as_kernel_param(),
            (&mut assign).as_kernel_param(),
            (&mut level).as_kernel_param(),
            (&mut reason).as_kernel_param(),
            (&mut var_activity).as_kernel_param(),
            (&mut var_phase).as_kernel_param(),
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
            sat_fn
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut params,
                )
        }
        .map_err(|e| XlogError::Kernel(format!("Failed to launch SAT solver kernel: {}", e)))?;
        self.provider.device().synchronize()?;

        let mut status_host = [0i32];
        let mut error_host = [0i32];
        let mut learned_count_host = [0u32];
        self.provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&out_status, &mut status_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy SAT status: {}", e)))?;
        self.provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&out_error, &mut error_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy SAT error: {}", e)))?;
        self.provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&out_learned_count, &mut learned_count_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy learned_count: {}", e)))?;

        let status_host = status_host[0];
        let error_host = error_host[0];
        let learned_count = learned_count_host[0];

        if error_host != 0 || status_host == -1 {
            return Err(XlogError::Kernel(format!(
                "GPU CDCL solver failed: status={} error_code={}",
                status_host, error_host
            )));
        }

        match status_host {
            1 => {
                // SAT: validate model on GPU.
                let mut out_ok = memory.alloc::<i32>(1)?;
                let check_fn = self
                    .provider
                    .device()
                    .inner()
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
                                &cnf.clause_offsets,
                                &cnf.literals,
                                cnf.num_clauses,
                                &assign,
                                &mut out_ok,
                            ),
                        )
                        .map_err(|e| XlogError::Kernel(format!("Failed to launch SAT model-check kernel: {}", e)))?;
                }
                self.provider.device().synchronize()?;
                let mut ok_host = [0i32];
                self.provider
                    .device()
                    .inner()
                    .dtoh_sync_copy_into(&out_ok, &mut ok_host)
                    .map_err(|e| XlogError::Kernel(format!("Failed to copy SAT model-check result: {}", e)))?;
                if ok_host[0] != 1 {
                    return Err(XlogError::Kernel(
                        "GPU CDCL solver returned SAT but model check failed".to_string(),
                    ));
                }
                Ok(GpuCdclResult {
                    status: GpuSolveStatus::Sat,
                    assignment: assign,
                    learned_count,
                })
            }
            0 => {
                // UNSAT: validate proof on GPU.
                let mut out_ok = memory.alloc::<i32>(1)?;
                let scratch_cap = (cnf.num_vars as usize) + 1;
                let mut scratch_a = memory.alloc::<i32>(scratch_cap)?;
                let mut scratch_b = memory.alloc::<i32>(scratch_cap)?;

                let proof_fn = self
                    .provider
                    .device()
                    .inner()
                    .get_func(SAT_MODULE, sat_kernels::SAT_PROOF_CHECK)
                    .ok_or_else(|| XlogError::Kernel("sat_proof_check kernel not found".to_string()))?;
                unsafe {
                    proof_fn
                        .clone()
                        .launch(
                            LaunchConfig {
                                grid_dim: (1, 1, 1),
                                block_dim: (1, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &cnf.clause_offsets,
                                &cnf.literals,
                                cnf.num_clauses,
                                &learned_offsets,
                                &learned_lits,
                                &out_learned_count,
                                &proof_offsets,
                                &proof_data,
                                &mut scratch_a,
                                &mut scratch_b,
                                scratch_cap as u32,
                                &mut out_ok,
                            ),
                        )
                        .map_err(|e| XlogError::Kernel(format!("Failed to launch SAT proof-check kernel: {}", e)))?;
                }
                self.provider.device().synchronize()?;
                let mut ok_host = [0i32];
                self.provider
                    .device()
                    .inner()
                    .dtoh_sync_copy_into(&out_ok, &mut ok_host)
                    .map_err(|e| XlogError::Kernel(format!("Failed to copy SAT proof-check result: {}", e)))?;
                if ok_host[0] != 1 {
                    return Err(XlogError::Kernel(
                        "GPU CDCL solver returned UNSAT but proof check failed".to_string(),
                    ));
                }
                Ok(GpuCdclResult {
                    status: GpuSolveStatus::Unsat,
                    assignment: assign,
                    learned_count,
                })
            }
            other => Err(XlogError::Kernel(format!(
                "GPU CDCL solver returned invalid status {}",
                other
            ))),
        }
    }
}
