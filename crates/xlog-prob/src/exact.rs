//! Exact probabilistic inference via Decision-DNNF (D4) + weighted model counting.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use cudarc::driver::LaunchConfig;
use xlog_core::{MemoryBudget, Result, ScalarType, XlogError};
use xlog_cuda::LaunchAsync;
use xlog_logic::ast::Program;

use crate::compilation::gpu_cache::{
    GpuCircuitCache, GpuCircuitCacheConfig, GpuCircuitCacheHandle,
};
use crate::compilation::gpu_cnf::GpuCnfVarTables;
#[cfg(feature = "host-io")]
use crate::compilation::gpu_weights::map_nodes_to_vars_gpu;
use crate::compilation::gpu_weights::{build_evidence_by_var_gpu, build_weights_gpu};
use crate::compilation::{
    compile_gpu_d4_and_verify_cached, encode_cnf_gpu, CircuitCompileProfile, DeviceRandomVarList,
    GpuCompileConfig, GpuPirGraph, GpuPirRoots,
};
use crate::neural_fast_path::{GpuWeightSlots, NeuralFastPathConfig};
use crate::provenance::{extract_from_program, extract_from_source, GroundAtom, Provenance};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{
    arith_kernels, filter_kernels, neural_kernels, weights_kernels, ARITH_MODULE, FILTER_MODULE,
    NEURAL_MODULE, WEIGHTS_MODULE,
};
use xlog_cuda::{CudaBuffer, CudaDevice, CudaKernelProvider, GpuMemoryManager};

#[derive(Debug, Clone)]
pub struct QueryProbability {
    pub atom: GroundAtom,
    pub log_prob: f64,
    pub prob: f64,
}

#[derive(Debug, Clone)]
pub struct ExactResult {
    pub log_z_e: f64,
    pub query_probs: Vec<QueryProbability>,
}

#[derive(Debug, Clone)]
pub struct QueryGradients {
    pub atom: GroundAtom,
    pub log_prob: f64,
    pub prob: f64,
    pub grad_true: Vec<f64>,
    pub grad_false: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ExactResultWithGrads {
    pub log_z_e: f64,
    pub query_grads: Vec<QueryGradients>,
}

#[derive(Debug, Clone)]
struct QuerySpec {
    #[cfg_attr(not(feature = "host-io"), allow(dead_code))]
    atom: GroundAtom,
    var: Option<u32>,
}

struct GpuExactState {
    provider: Arc<CudaKernelProvider>,
    cache: Mutex<GpuCircuitCache>,
    handle: GpuCircuitCacheHandle,
}

/// GPU device selection and memory budget for probabilistic inference.
///
/// Use [`GpuConfig::default()`] and override individual fields as needed.
#[derive(Debug, Clone, Copy)]
pub struct GpuConfig {
    /// CUDA device ordinal (0-based).
    pub device_ordinal: usize,
    /// Device memory budget in bytes (clamped to available memory at runtime).
    pub memory_bytes: u64,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            memory_bytes: 32 * 1024 * 1024 * 1024, // 32 GB — clamped to available device memory by GpuMemoryManager at runtime.
        }
    }
}

impl GpuExactState {
    fn new(
        provider: Arc<CudaKernelProvider>,
        cache: GpuCircuitCache,
        handle: GpuCircuitCacheHandle,
    ) -> Result<Self> {
        Ok(Self {
            provider,
            cache: Mutex::new(cache),
            handle,
        })
    }

    fn provider(&self) -> &Arc<CudaKernelProvider> {
        &self.provider
    }

    fn handle(&self) -> &GpuCircuitCacheHandle {
        &self.handle
    }
}

#[derive(Clone)]
pub struct ExactDdnnfProgram {
    gpu: Option<Arc<GpuExactState>>,
    queries: Vec<QuerySpec>,
    #[cfg_attr(not(feature = "host-io"), allow(dead_code))]
    random_vars: Option<Arc<DeviceRandomVarList>>,
    max_var: u32,
    #[allow(dead_code)] // retained: config is stored for future re-compilation paths
    gpu_config: GpuConfig,
    /// Latest circuit compilation profile (populated on cache miss when profiling).
    last_compile_profile: Option<CircuitCompileProfile>,
}

impl ExactDdnnfProgram {
    pub fn compile_source(source: &str) -> Result<Self> {
        let provenance = extract_from_source(source)?;
        Self::compile_provenance_with_gpu(provenance, GpuConfig::default())
    }

    pub fn compile_source_with_gpu(source: &str, config: GpuConfig) -> Result<Self> {
        let provenance = extract_from_source(source)?;
        Self::compile_provenance_with_gpu(provenance, config)
    }

    pub fn compile_from_program(program: &Program, config: GpuConfig) -> Result<Self> {
        let provenance = extract_from_program(program)?;
        Self::compile_provenance_with_gpu(provenance, config)
    }

    #[allow(dead_code)] // retained: accessor for future re-compilation paths
    pub(crate) fn gpu_config(&self) -> GpuConfig {
        self.gpu_config
    }

    /// Get the latest circuit compilation profile (populated when XLOG_WARMUP_PROFILE=1).
    pub fn last_compile_profile(&self) -> Option<&CircuitCompileProfile> {
        self.last_compile_profile.as_ref()
    }

    #[cfg(feature = "host-io")]
    pub fn evaluate(&self) -> Result<ExactResult> {
        if self.gpu.is_none() {
            let mut query_probs: Vec<QueryProbability> = Vec::with_capacity(self.queries.len());
            for query in &self.queries {
                query_probs.push(QueryProbability {
                    atom: query.atom.clone(),
                    log_prob: f64::NEG_INFINITY,
                    prob: 0.0,
                });
            }
            return Ok(ExactResult {
                log_z_e: 0.0,
                query_probs,
            });
        }

        let log_z_e = self.eval_log_z_gpu(None)?;
        if log_z_e.is_infinite() && log_z_e.is_sign_negative() {
            return Err(XlogError::Execution(
                "Exact inference error: evidence is inconsistent (P(E)=0)".to_string(),
            ));
        }

        let mut query_probs: Vec<QueryProbability> = Vec::with_capacity(self.queries.len());
        for query in &self.queries {
            let (log_prob, prob) = match query.var {
                None => (f64::NEG_INFINITY, 0.0),
                Some(var) => {
                    let log_z_eq = self.eval_log_z_gpu(Some(var))?;
                    let log_prob = log_z_eq - log_z_e;
                    let mut prob = if log_prob.is_infinite() && log_prob.is_sign_negative() {
                        0.0
                    } else {
                        log_prob.exp()
                    };
                    if prob.is_nan() {
                        return Err(XlogError::Execution(
                            "Exact inference error: NaN probability encountered".to_string(),
                        ));
                    }
                    if prob < 0.0 {
                        prob = 0.0;
                    } else if prob > 1.0 {
                        prob = 1.0;
                    }
                    (log_prob, prob)
                }
            };

            query_probs.push(QueryProbability {
                atom: query.atom.clone(),
                log_prob,
                prob,
            });
        }

        Ok(ExactResult {
            log_z_e,
            query_probs,
        })
    }

    pub fn num_vars(&self) -> usize {
        if self.max_var == 0 {
            0
        } else {
            (self.max_var as usize) + 1
        }
    }

    /// Returns the indices of random (probabilistic) variables in order.
    ///
    /// Random variables are those with non-trivial weights (not (0.0, 0.0)).
    /// These correspond to annotated disjunctions in the source program.
    /// The order matches the order variables were assigned during CNF encoding.
    #[cfg(feature = "host-io")]
    pub fn random_var_indices(&self) -> Vec<u32> {
        let Some(state) = self.gpu.as_ref() else {
            return Vec::new();
        };
        let Some(random_vars) = self.random_vars.as_ref() else {
            return Vec::new();
        };
        if random_vars.is_empty() {
            return Vec::new();
        }
        let count = random_vars.count() as usize;
        let mut host = vec![0u32; count];
        let view = random_vars.list().slice(0..count);
        if let Err(e) = state
            .provider()
            .device()
            .inner()
            .dtoh_sync_copy_into(&view, &mut host)
        {
            eprintln!("Failed to read random var list: {}", e);
            return Vec::new();
        }
        host
    }

    /// CNF variable id for the `idx`-th query formula (DIMACS, 1-based), if present.
    pub(crate) fn query_var(&self, idx: usize) -> Option<u32> {
        self.queries.get(idx).and_then(|q| q.var)
    }

    /// GPU neural fast-path: compute NLL gradients w.r.t. probability tensors (no host reads).
    ///
    /// This implements the design in `docs/design/2026-01-22-gpu-native-compilation-design.md` §5.3:
    /// - Fill AD conditional-chain log-weights from device-resident `p[label]`.
    /// - Run XGCF forward+backward on GPU.
    /// - Scatter gradients back into probability-space via the correct chain rule (uses both grad_true + grad_false).
    ///
    /// The output gradient buffers are updated in-place:
    /// - Base run: `out = dlogZ_base/dp`
    /// - Query-forced run: `out -= dlogZ_query/dp`
    /// Result: `out = dL/dp` for `L = -log P(query | evidence)` (NLL).
    pub fn neural_backward_nll_buffers(
        &self,
        slots: &GpuWeightSlots,
        query_idx: usize,
        probs: &[CudaBuffer],
        out_grads: &mut [CudaBuffer],
        cfg: NeuralFastPathConfig,
    ) -> Result<()> {
        self.neural_backward_nll_buffers_inner(slots, query_idx, probs, out_grads, cfg, None, true)
    }

    /// Same as [`Self::neural_backward_nll_buffers`], but also returns the device-resident scalar NLL loss:
    /// `L = -log P(query | evidence)`.
    ///
    /// The returned slice has length 1 and is written on GPU (no device->host reads).
    pub fn neural_backward_nll_buffers_with_device_loss(
        &self,
        slots: &GpuWeightSlots,
        query_idx: usize,
        probs: &[CudaBuffer],
        out_grads: &mut [CudaBuffer],
        cfg: NeuralFastPathConfig,
        expected_true: bool,
    ) -> Result<TrackedCudaSlice<f64>> {
        let state = self.gpu_state()?;
        let mut loss = state.provider.memory().alloc::<f64>(1)?;
        self.neural_backward_nll_buffers_inner(
            slots,
            query_idx,
            probs,
            out_grads,
            cfg,
            Some(&mut loss),
            expected_true,
        )?;
        Ok(loss)
    }

    /// Batched variant of [`Self::neural_backward_nll_buffers_with_device_loss`].
    ///
    /// Computes NLL gradients for `batch` queries that share one compiled circuit
    /// template and returns a device-resident vector of `batch` scalar losses.
    ///
    /// On circuits that require free-variable correction, this falls back to the
    /// existing per-query path for correctness.
    pub fn neural_backward_nll_buffers_batch_with_device_loss(
        &self,
        slots: &GpuWeightSlots,
        query_indices: &[usize],
        probs_batch: &[Vec<CudaBuffer>],
        out_grads_batch: &mut [Vec<CudaBuffer>],
        cfg: NeuralFastPathConfig,
        expected_true: bool,
    ) -> Result<TrackedCudaSlice<f64>> {
        let batch = query_indices.len();
        if batch == 0 {
            return Err(XlogError::Execution(
                "Neural fast-path batch error: empty query batch".to_string(),
            ));
        }
        if probs_batch.len() != batch || out_grads_batch.len() != batch {
            return Err(XlogError::Compilation(format!(
                "Neural fast-path batch error: query/prob/grad batch mismatch ({}/{}/{})",
                batch,
                probs_batch.len(),
                out_grads_batch.len()
            )));
        }

        let state = self.gpu_state()?;
        let batch_u32 = u32::try_from(batch).map_err(|_| {
            XlogError::Compilation("Neural fast-path batch size exceeds u32".to_string())
        })?;
        let device = state.provider.device().inner();

        // Fallback for circuits that currently require per-query free-var correction.
        {
            let cache = state
                .cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if cache.has_any_free_var_mask() {
                drop(cache);
                let mut losses = state.provider.memory().alloc::<f64>(batch)?;
                for q in 0..batch {
                    let loss_q = self.neural_backward_nll_buffers_with_device_loss(
                        slots,
                        query_indices[q],
                        &probs_batch[q],
                        &mut out_grads_batch[q],
                        cfg,
                        expected_true,
                    )?;
                    let mut dst = losses.slice_mut(q..(q + 1));
                    device.dtod_copy(&loss_q, &mut dst).map_err(|e| {
                        XlogError::Kernel(format!(
                            "Failed to copy fallback batch loss to output: {}",
                            e
                        ))
                    })?;
                }
                return Ok(losses);
            }
        }

        let fill = device
            .get_func(NEURAL_MODULE, neural_kernels::NEURAL_FILL_AD_CHAIN_F32)
            .ok_or_else(|| {
                XlogError::Kernel("neural_fill_ad_chain_f32 kernel not found".to_string())
            })?;
        let scatter = device
            .get_func(
                NEURAL_MODULE,
                neural_kernels::NEURAL_SCATTER_AD_CHAIN_GRADS_F32,
            )
            .ok_or_else(|| {
                XlogError::Kernel("neural_scatter_ad_chain_grads_f32 kernel not found".to_string())
            })?;
        let binary_f64 = device
            .get_func(ARITH_MODULE, arith_kernels::ARITH_BINARY_F64)
            .ok_or_else(|| XlogError::Kernel("arith_binary_f64 kernel not found".to_string()))?;
        let apply_query_false_batched = device
            .get_func(
                WEIGHTS_MODULE,
                weights_kernels::WEIGHTS_APPLY_QUERY_VARS_FALSE_BATCHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel(
                    "weights_apply_query_vars_false_batched kernel not found".to_string(),
                )
            })?;
        let apply_query_true_batched = device
            .get_func(
                WEIGHTS_MODULE,
                weights_kernels::WEIGHTS_APPLY_QUERY_VARS_TRUE_BATCHED,
            )
            .ok_or_else(|| {
                XlogError::Kernel(
                    "weights_apply_query_vars_true_batched kernel not found".to_string(),
                )
            })?;

        let mut cache = state
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let var_stride = cache.var_stride()?;
        let var_stride_usize = var_stride as usize;
        let node_stride = cache.node_stride();
        let node_stride_usize = node_stride as usize;

        let mut var_log_true_batch = state
            .provider
            .memory()
            .alloc::<f64>(batch * var_stride_usize)?;
        let mut var_log_false_batch = state
            .provider
            .memory()
            .alloc::<f64>(batch * var_stride_usize)?;
        cache.copy_slot_weights_to_batch(
            state.handle(),
            &mut var_log_true_batch,
            &mut var_log_false_batch,
            batch_u32,
        )?;

        let mut values_batch = state
            .provider
            .memory()
            .alloc::<f64>(batch * node_stride_usize)?;
        let mut adj_batch = state
            .provider
            .memory()
            .alloc::<f64>(batch * node_stride_usize)?;
        let mut grad_true_batch = state
            .provider
            .memory()
            .alloc::<f64>(batch * var_stride_usize)?;
        let mut grad_false_batch = state
            .provider
            .memory()
            .alloc::<f64>(batch * var_stride_usize)?;
        let mut base_roots = state.provider.memory().alloc::<f64>(batch)?;
        let mut query_roots = state.provider.memory().alloc::<f64>(batch)?;
        let mut losses = state.provider.memory().alloc::<f64>(batch)?;
        let mut query_vars = state.provider.memory().alloc::<u32>(batch)?;
        let mut force_saved = state.provider.memory().alloc::<f64>(batch)?;

        let mut query_vars_host: Vec<u32> = Vec::with_capacity(batch);

        // Fill per-query var weight rows from device-resident probability tensors.
        for q in 0..batch {
            if probs_batch[q].len() != out_grads_batch[q].len() {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path batch error: probs len {} != out_grads len {} for query {}",
                    probs_batch[q].len(),
                    out_grads_batch[q].len(),
                    q
                )));
            }
            if probs_batch[q].len() as u32 != slots.num_groups() {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path batch error: expected {} groups, got {} for query {}",
                    slots.num_groups(),
                    probs_batch[q].len(),
                    q
                )));
            }

            let query_var = self.query_var(query_indices[q]).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Neural fast-path batch error: query {} has no CNF var",
                    query_indices[q]
                ))
            })?;
            if query_var == 0 || query_var > self.max_var {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path batch error: query var {} out of bounds (max_var={})",
                    query_var, self.max_var
                )));
            }
            query_vars_host.push(query_var);

            let row_start = q
                .checked_mul(var_stride_usize)
                .ok_or_else(|| XlogError::Compilation("Neural batch row overflow".to_string()))?;
            let row_end = row_start + var_stride_usize;

            for (g, prob_buf) in probs_batch[q].iter().enumerate() {
                if prob_buf.arity() != 1 {
                    return Err(XlogError::Compilation(
                        "Neural fast-path expects 1-column prob buffers".to_string(),
                    ));
                }
                let ty = prob_buf.schema().column_type(0).ok_or_else(|| {
                    XlogError::Compilation("Missing prob buffer schema".to_string())
                })?;
                if ty != ScalarType::F32 {
                    return Err(XlogError::Compilation(format!(
                        "Neural fast-path expects prob dtype F32, got {:?}",
                        ty
                    )));
                }

                let slot_vars = slots.group_slot_cnf_var(g)?;
                let labels = slot_vars.len() as u32;
                if prob_buf.num_rows() != labels as u64 {
                    return Err(XlogError::Compilation(format!(
                        "Neural fast-path prob rows {} != labels {}",
                        prob_buf.num_rows(),
                        labels
                    )));
                }
                if out_grads_batch[q][g].num_rows() != labels as u64 {
                    return Err(XlogError::Compilation(format!(
                        "Neural fast-path grad rows {} != labels {}",
                        out_grads_batch[q][g].num_rows(),
                        labels
                    )));
                }

                let prob_col = prob_buf.column(0).ok_or_else(|| {
                    XlogError::Compilation("Neural fast-path missing prob column".to_string())
                })?;
                let mut q_true = var_log_true_batch.slice_mut(row_start..row_end);
                let mut q_false = var_log_false_batch.slice_mut(row_start..row_end);

                unsafe {
                    fill.clone().launch(
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            prob_col,
                            labels,
                            &slot_vars,
                            cfg.eps,
                            cfg.min_p,
                            &mut q_true,
                            &mut q_false,
                        ),
                    )
                }
                .map_err(|e| {
                    XlogError::Kernel(format!("neural_fill_ad_chain_f32 failed: {}", e))
                })?;
            }
        }

        // Base pass (all queries): grads = dlogZ_base/dp, roots = logZ_base.
        cache.eval_grads_inplace_fused_batched(
            state.handle(),
            &var_log_true_batch,
            &var_log_false_batch,
            &mut values_batch,
            &mut adj_batch,
            &mut grad_true_batch,
            &mut grad_false_batch,
            batch_u32,
        )?;
        cache.copy_root_batched_from_values(
            state.handle(),
            &values_batch,
            &mut base_roots,
            batch_u32,
        )?;

        // Scatter base gradients into output buffers.
        for q in 0..batch {
            let row_start = q
                .checked_mul(var_stride_usize)
                .ok_or_else(|| XlogError::Compilation("Neural batch row overflow".to_string()))?;
            let row_end = row_start + var_stride_usize;
            let q_grad_true = grad_true_batch.slice(row_start..row_end);
            let q_grad_false = grad_false_batch.slice(row_start..row_end);

            for (g, prob_buf) in probs_batch[q].iter().enumerate() {
                let slot_vars = slots.group_slot_cnf_var(g)?;
                let labels = slot_vars.len() as u32;
                let prob_col = prob_buf.column(0).ok_or_else(|| {
                    XlogError::Compilation("Neural fast-path missing prob column".to_string())
                })?;
                let out_col = out_grads_batch[q][g]
                    .columns
                    .get_mut(0)
                    .ok_or_else(|| XlogError::Compilation("Missing grad column".to_string()))?;

                let shared_bytes: u32 = 3u64
                    .checked_mul(labels as u64)
                    .and_then(|n| n.checked_mul(std::mem::size_of::<f64>() as u64))
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| {
                        XlogError::Kernel("Neural scatter shared memory overflow".to_string())
                    })?;

                unsafe {
                    scatter.clone().launch(
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: shared_bytes,
                        },
                        (
                            prob_col,
                            labels,
                            &slot_vars,
                            cfg.eps,
                            cfg.min_p,
                            &q_grad_true,
                            &q_grad_false,
                            0u8,
                            out_col,
                        ),
                    )
                }
                .map_err(|e| XlogError::Kernel(format!("neural_scatter (base) failed: {}", e)))?;
            }
        }

        device
            .htod_sync_copy_into(&query_vars_host, &mut query_vars)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to upload batched query vars: {}", e))
            })?;
        let force_grid = if batch_u32 == 0 {
            0
        } else {
            (batch_u32 + 255) / 256
        };
        if force_grid != 0 {
            if expected_true {
                unsafe {
                    apply_query_false_batched.clone().launch(
                        LaunchConfig {
                            grid_dim: (force_grid, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &query_vars,
                            batch_u32,
                            self.max_var,
                            var_stride,
                            &mut var_log_false_batch,
                            &mut force_saved,
                        ),
                    )
                }
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "weights_apply_query_vars_false_batched failed: {}",
                        e
                    ))
                })?;
            } else {
                unsafe {
                    apply_query_true_batched.clone().launch(
                        LaunchConfig {
                            grid_dim: (force_grid, 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &query_vars,
                            batch_u32,
                            self.max_var,
                            var_stride,
                            &mut var_log_true_batch,
                            &mut force_saved,
                        ),
                    )
                }
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "weights_apply_query_vars_true_batched failed: {}",
                        e
                    ))
                })?;
            }
        }

        // Query-forced pass (all queries): grads = dlogZ_query/dp, roots = logZ_query.
        cache.eval_grads_inplace_fused_batched(
            state.handle(),
            &var_log_true_batch,
            &var_log_false_batch,
            &mut values_batch,
            &mut adj_batch,
            &mut grad_true_batch,
            &mut grad_false_batch,
            batch_u32,
        )?;
        cache.copy_root_batched_from_values(
            state.handle(),
            &values_batch,
            &mut query_roots,
            batch_u32,
        )?;

        let loss_grid = if batch_u32 == 0 {
            0
        } else {
            (batch_u32 + 255) / 256
        };
        if loss_grid != 0 {
            unsafe {
                binary_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (loss_grid, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&base_roots, &query_roots, batch_u32, 1u8, &mut losses),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("Failed to compute batched NLL loss: {}", e)))?;
        }

        // Scatter query gradients with subtract mode.
        for q in 0..batch {
            let row_start = q
                .checked_mul(var_stride_usize)
                .ok_or_else(|| XlogError::Compilation("Neural batch row overflow".to_string()))?;
            let row_end = row_start + var_stride_usize;
            let q_grad_true = grad_true_batch.slice(row_start..row_end);
            let q_grad_false = grad_false_batch.slice(row_start..row_end);

            for (g, prob_buf) in probs_batch[q].iter().enumerate() {
                let slot_vars = slots.group_slot_cnf_var(g)?;
                let labels = slot_vars.len() as u32;
                let prob_col = prob_buf.column(0).ok_or_else(|| {
                    XlogError::Compilation("Neural fast-path missing prob column".to_string())
                })?;
                let out_col = out_grads_batch[q][g]
                    .columns
                    .get_mut(0)
                    .ok_or_else(|| XlogError::Compilation("Missing grad column".to_string()))?;

                let shared_bytes: u32 = 3u64
                    .checked_mul(labels as u64)
                    .and_then(|n| n.checked_mul(std::mem::size_of::<f64>() as u64))
                    .and_then(|n| u32::try_from(n).ok())
                    .ok_or_else(|| {
                        XlogError::Kernel("Neural scatter shared memory overflow".to_string())
                    })?;

                unsafe {
                    scatter.clone().launch(
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: shared_bytes,
                        },
                        (
                            prob_col,
                            labels,
                            &slot_vars,
                            cfg.eps,
                            cfg.min_p,
                            &q_grad_true,
                            &q_grad_false,
                            1u8,
                            out_col,
                        ),
                    )
                }
                .map_err(|e| XlogError::Kernel(format!("neural_scatter (query) failed: {}", e)))?;
            }
        }

        Ok(losses)
    }

    fn neural_backward_nll_buffers_inner(
        &self,
        slots: &GpuWeightSlots,
        query_idx: usize,
        probs: &[CudaBuffer],
        out_grads: &mut [CudaBuffer],
        cfg: NeuralFastPathConfig,
        out_loss: Option<&mut TrackedCudaSlice<f64>>,
        expected_true: bool,
    ) -> Result<()> {
        if self.gpu.is_none() {
            return Err(XlogError::Execution(
                "Neural fast-path error: program has no compiled circuit".to_string(),
            ));
        }

        let query_var = self.query_var(query_idx).ok_or_else(|| {
            XlogError::Execution(format!(
                "Neural fast-path error: query {} has no CNF var",
                query_idx
            ))
        })?;

        if probs.len() != out_grads.len() {
            return Err(XlogError::Compilation(format!(
                "Neural fast-path error: probs len {} != out_grads len {}",
                probs.len(),
                out_grads.len()
            )));
        }
        if probs.len() as u32 != slots.num_groups() {
            return Err(XlogError::Compilation(format!(
                "Neural fast-path error: expected {} groups, got {}",
                slots.num_groups(),
                probs.len()
            )));
        }

        let state = self.gpu_state()?;
        let device = state.provider.device().inner();

        let fill = device
            .get_func(NEURAL_MODULE, neural_kernels::NEURAL_FILL_AD_CHAIN_F32)
            .ok_or_else(|| {
                XlogError::Kernel("neural_fill_ad_chain_f32 kernel not found".to_string())
            })?;
        let scatter = device
            .get_func(
                NEURAL_MODULE,
                neural_kernels::NEURAL_SCATTER_AD_CHAIN_GRADS_F32,
            )
            .ok_or_else(|| {
                XlogError::Kernel("neural_scatter_ad_chain_grads_f32 kernel not found".to_string())
            })?;
        let binary_f64 = device
            .get_func(ARITH_MODULE, arith_kernels::ARITH_BINARY_F64)
            .ok_or_else(|| XlogError::Kernel("arith_binary_f64 kernel not found".to_string()))?;

        let mut cache = state
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        let root_idx = state.handle().root() as usize;

        // If the caller requested the scalar loss, keep the base logZ on device so we can compute
        // loss = logZ_base - logZ_query without any host reads.
        let mut base_log_z: Option<TrackedCudaSlice<f64>> = if out_loss.is_some() {
            Some(state.provider.memory().alloc::<f64>(1)?)
        } else {
            None
        };

        // 1) Update AD chain weights from device-resident p[label].
        for (g, prob_buf) in probs.iter().enumerate() {
            if prob_buf.arity() != 1 {
                return Err(XlogError::Compilation(
                    "Neural fast-path expects 1-column prob buffers".to_string(),
                ));
            }
            let ty = prob_buf
                .schema()
                .column_type(0)
                .ok_or_else(|| XlogError::Compilation("Missing prob buffer schema".to_string()))?;
            if ty != ScalarType::F32 {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path expects prob dtype F32, got {:?}",
                    ty
                )));
            }

            let slot_vars = slots.group_slot_cnf_var(g)?;
            let labels = slot_vars.len() as u32;

            if prob_buf.num_rows() != labels as u64 {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path prob rows {} != labels {}",
                    prob_buf.num_rows(),
                    labels
                )));
            }

            let prob_col = prob_buf.column(0).ok_or_else(|| {
                XlogError::Compilation("Neural fast-path missing prob column".to_string())
            })?;

            let (var_log_true, var_log_false) = cache.var_log_weights_mut();

            unsafe {
                fill.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        prob_col,
                        labels,
                        &slot_vars,
                        cfg.eps,
                        cfg.min_p,
                        var_log_true,
                        var_log_false,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("neural_fill_ad_chain_f32 failed: {}", e)))?;
        }

        // 2) Base run: out = dlogZ_base/dp
        cache.eval_grads_inplace_fused(state.handle())?;
        if let Some(base) = base_log_z.as_mut() {
            let root_view = cache.values().slice(root_idx..(root_idx + 1));
            device.dtod_copy(&root_view, base).map_err(|e| {
                XlogError::Kernel(format!("Failed to copy base logZ on GPU: {}", e))
            })?;
        }
        for (g, prob_buf) in probs.iter().enumerate() {
            let slot_vars = slots.group_slot_cnf_var(g)?;
            let labels = slot_vars.len() as u32;

            let out_buf = out_grads.get_mut(g).ok_or_else(|| {
                XlogError::Compilation("Neural fast-path missing output grad buffer".to_string())
            })?;
            if out_buf.arity() != 1 {
                return Err(XlogError::Compilation(
                    "Neural fast-path expects 1-column grad buffers".to_string(),
                ));
            }
            let out_ty = out_buf
                .schema()
                .column_type(0)
                .ok_or_else(|| XlogError::Compilation("Missing grad buffer schema".to_string()))?;
            if out_ty != ScalarType::F32 {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path expects grad dtype F32, got {:?}",
                    out_ty
                )));
            }
            if out_buf.num_rows() != labels as u64 {
                return Err(XlogError::Compilation(format!(
                    "Neural fast-path grad rows {} != labels {}",
                    out_buf.num_rows(),
                    labels
                )));
            }

            let prob_col = prob_buf.column(0).ok_or_else(|| {
                XlogError::Compilation("Neural fast-path missing prob column".to_string())
            })?;
            let out_col = out_buf
                .columns
                .get_mut(0)
                .ok_or_else(|| XlogError::Compilation("Missing grad column".to_string()))?;

            let shared_bytes: u32 = 3u64
                .checked_mul(labels as u64)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f64>() as u64))
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| {
                    XlogError::Kernel("Neural scatter shared memory overflow".to_string())
                })?;

            unsafe {
                scatter.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: shared_bytes,
                    },
                    (
                        prob_col,
                        labels,
                        &slot_vars,
                        cfg.eps,
                        cfg.min_p,
                        cache.grad_true(),
                        cache.grad_false(),
                        0u8,
                        out_col,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("neural_scatter (base) failed: {}", e)))?;
        }

        // 3) Query run: out -= dlogZ_query/dp
        if query_var == 0 || query_var > self.max_var {
            return Err(XlogError::Compilation(format!(
                "Neural fast-path error: query var {} out of bounds (max_var={})",
                query_var, self.max_var
            )));
        }

        let mut restore = state.provider.memory().alloc::<f64>(1)?;
        if expected_true {
            {
                let (_, var_log_false) = cache.var_log_weights_mut();
                force_query_var_false(state.provider(), var_log_false, query_var, &mut restore)?;
            }
        } else {
            {
                let (var_log_true, _) = cache.var_log_weights_mut();
                force_query_var_true(state.provider(), var_log_true, query_var, &mut restore)?;
            }
        }

        cache.eval_grads_inplace_fused(state.handle())?;
        if let Some(out) = out_loss {
            let base = base_log_z
                .as_ref()
                .expect("base_log_z allocated when out_loss requested");
            let root_view = cache.values().slice(root_idx..(root_idx + 1));
            unsafe {
                binary_f64.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (base, &root_view, 1u32, 1u8, out),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("Failed to compute NLL loss on GPU: {}", e)))?;
        }
        for (g, prob_buf) in probs.iter().enumerate() {
            let slot_vars = slots.group_slot_cnf_var(g)?;
            let labels = slot_vars.len() as u32;

            let prob_col = prob_buf.column(0).ok_or_else(|| {
                XlogError::Compilation("Neural fast-path missing prob column".to_string())
            })?;
            let out_col = out_grads[g]
                .columns
                .get_mut(0)
                .ok_or_else(|| XlogError::Compilation("Missing grad column".to_string()))?;

            let shared_bytes: u32 = 3u64
                .checked_mul(labels as u64)
                .and_then(|n| n.checked_mul(std::mem::size_of::<f64>() as u64))
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| {
                    XlogError::Kernel("Neural scatter shared memory overflow".to_string())
                })?;

            unsafe {
                scatter.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: shared_bytes,
                    },
                    (
                        prob_col,
                        labels,
                        &slot_vars,
                        cfg.eps,
                        cfg.min_p,
                        cache.grad_true(),
                        cache.grad_false(),
                        1u8,
                        out_col,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("neural_scatter (query) failed: {}", e)))?;
        }
        if expected_true {
            {
                let (_, var_log_false) = cache.var_log_weights_mut();
                restore_query_var_false(state.provider(), var_log_false, query_var, &restore)?;
            }
        } else {
            {
                let (var_log_true, _) = cache.var_log_weights_mut();
                restore_query_var_true(state.provider(), var_log_true, query_var, &restore)?;
            }
        }

        Ok(())
    }

    #[cfg(feature = "host-io")]
    pub fn evaluate_gpu_with_grads(&self) -> Result<ExactResultWithGrads> {
        if self.gpu.is_none() {
            return Ok(ExactResultWithGrads {
                log_z_e: 0.0,
                query_grads: Vec::new(),
            });
        }

        let weights_len = if self.max_var == 0 {
            0
        } else {
            (self.max_var as usize) + 1
        };

        let (log_z_e, grad_true_e, grad_false_e) = self.eval_log_z_and_grads_gpu_cached(None)?;

        if log_z_e.is_infinite() && log_z_e.is_sign_negative() {
            return Err(XlogError::Execution(
                "Exact inference error: evidence is inconsistent (P(E)=0)".to_string(),
            ));
        }

        let mut query_grads: Vec<QueryGradients> = Vec::with_capacity(self.queries.len());

        for query in &self.queries {
            let Some(var) = query.var else {
                query_grads.push(QueryGradients {
                    atom: query.atom.clone(),
                    log_prob: f64::NEG_INFINITY,
                    prob: 0.0,
                    grad_true: vec![0.0; weights_len],
                    grad_false: vec![0.0; weights_len],
                });
                continue;
            };

            let idx = var as usize;
            if idx >= weights_len {
                return Err(XlogError::Compilation(format!(
                    "Exact inference error: query var {} out of bounds (len={})",
                    var, weights_len
                )));
            }

            let (log_z_eq, grad_true_eq, grad_false_eq) =
                self.eval_log_z_and_grads_gpu_cached(Some(var))?;

            let log_prob = log_z_eq - log_z_e;
            let mut prob = if log_prob.is_infinite() && log_prob.is_sign_negative() {
                0.0
            } else {
                log_prob.exp()
            };
            if prob.is_nan() {
                return Err(XlogError::Execution(
                    "Exact inference error: NaN probability encountered".to_string(),
                ));
            }
            if prob < 0.0 {
                prob = 0.0;
            } else if prob > 1.0 {
                prob = 1.0;
            }

            if grad_true_eq.len() != grad_true_e.len() || grad_false_eq.len() != grad_false_e.len()
            {
                return Err(XlogError::Execution(
                    "Exact inference error: gradient length mismatch".to_string(),
                ));
            }

            let mut grad_true: Vec<f64> = grad_true_eq;
            let mut grad_false: Vec<f64> = grad_false_eq;
            for i in 0..grad_true.len() {
                grad_true[i] -= grad_true_e[i];
                grad_false[i] -= grad_false_e[i];
            }

            query_grads.push(QueryGradients {
                atom: query.atom.clone(),
                log_prob,
                prob,
                grad_true,
                grad_false,
            });
        }

        Ok(ExactResultWithGrads {
            log_z_e,
            query_grads,
        })
    }

    fn compile_provenance_with_gpu(provenance: Provenance, config: GpuConfig) -> Result<Self> {
        if config.memory_bytes == 0 {
            return Err(XlogError::Kernel(
                "GPU memory budget must be non-zero".to_string(),
            ));
        }

        let mut roots_set: HashSet<crate::pir::PirNodeId> = HashSet::new();

        let mut evidence_formulas: Vec<(crate::pir::PirNodeId, bool, GroundAtom)> = Vec::new();
        let mut evidence_atoms: std::collections::HashMap<GroundAtom, bool> =
            std::collections::HashMap::new();
        for (atom, value) in &provenance.evidence {
            if let Some(prev) = evidence_atoms.insert(atom.clone(), *value) {
                if prev != *value {
                    return Err(XlogError::Execution(format!(
                        "Exact inference error: conflicting evidence for {}",
                        display_atom(atom)
                    )));
                }
            }

            let formula = provenance.query_formula(&atom.predicate, &atom.args);
            match formula {
                Some(id) => {
                    roots_set.insert(id);
                    evidence_formulas.push((id, *value, atom.clone()));
                }
                None => {
                    if *value {
                        return Err(XlogError::Execution(format!(
                            "Exact inference error: evidence atom is never derivable: {}",
                            display_atom(atom)
                        )));
                    }
                }
            }
        }

        let mut queries: Vec<QuerySpec> = Vec::new();
        #[cfg(feature = "host-io")]
        let mut query_nodes: Vec<(usize, crate::pir::PirNodeId)> = Vec::new();
        for atom in &provenance.queries {
            let formula = provenance.query_formula(&atom.predicate, &atom.args);
            if let Some(id) = formula {
                roots_set.insert(id);
                #[cfg(feature = "host-io")]
                {
                    query_nodes.push((queries.len(), id));
                }
            }
            queries.push(QuerySpec {
                atom: atom.clone(),
                var: None,
            });
        }

        // Ensure ALL probabilistic variable nodes (Decision, Lit, NegLit) are reachable
        // so they get CNF variables. This is required for the template/neural fast-path
        // where GpuWeightSlots expects one CNF variable per ChoiceVarId/LeafId.
        for (idx, node) in provenance.pir.nodes().iter().enumerate() {
            match node {
                crate::pir::PirNode::Decision { .. }
                | crate::pir::PirNode::Lit { .. }
                | crate::pir::PirNode::NegLit { .. } => {
                    roots_set.insert(crate::pir::PirNodeId::from_u32(idx as u32));
                }
                _ => {}
            }
        }

        let mut roots: Vec<crate::pir::PirNodeId> = roots_set.into_iter().collect();
        roots.sort();

        if roots.is_empty() {
            return Ok(Self {
                gpu: None,
                queries,
                random_vars: None,
                max_var: 0,
                gpu_config: config,
                last_compile_profile: None,
            });
        }

        let device = Arc::new(CudaDevice::new(config.device_ordinal)?);
        let memory = Arc::new(GpuMemoryManager::new(
            device.clone(),
            MemoryBudget::with_limit(config.memory_bytes),
        ));
        let provider = Arc::new(CudaKernelProvider::new(device, memory)?);

        let canonical_cnf_hash = crate::cnf::canonical_pir_hash(&provenance.pir, &roots)?;
        let gpu_pir = GpuPirGraph::from_host(&provenance.pir, &provider)?;
        let gpu_roots = GpuPirRoots::from_host(&roots, &provider)?;
        let encoding = encode_cnf_gpu(&gpu_pir, &gpu_roots, &provider)?;
        if encoding.vars.max_var != encoding.cnf.var_cap {
            return Err(XlogError::Compilation(format!(
                "Exact inference error: CNF var_cap {} != vars.max_var {}",
                encoding.cnf.var_cap, encoding.vars.max_var
            )));
        }

        let (leaf_probs_host, choice_true_host, choice_false_host) =
            build_weight_sources(&provenance)?;

        let leaf_probs = upload_f64(&provider, &leaf_probs_host)?;
        let choice_true = upload_f64(&provider, &choice_true_host)?;
        let choice_false = upload_f64(&provider, &choice_false_host)?;

        let evidence_by_var = if evidence_formulas.is_empty() {
            let mut evidence = provider
                .memory()
                .alloc::<u8>((encoding.vars.max_var as usize) + 1)?;
            provider
                .device()
                .inner()
                .memset_zeros(&mut evidence)
                .map_err(|e| XlogError::Kernel(format!("Failed to zero evidence buffer: {}", e)))?;
            evidence
        } else {
            let mut nodes: Vec<u32> = Vec::with_capacity(evidence_formulas.len());
            let mut vals: Vec<u8> = Vec::with_capacity(evidence_formulas.len());
            for (node, value, _atom) in &evidence_formulas {
                nodes.push(node.as_u32());
                vals.push(if *value { 1u8 } else { 2u8 });
            }
            let evidence_nodes = upload_u32(&provider, &nodes)?;
            let evidence_vals = upload_u8(&provider, &vals)?;
            build_evidence_by_var_gpu(
                &encoding.vars.node_var,
                &evidence_nodes,
                &evidence_vals,
                encoding.vars.max_var,
                &provider,
            )?
        };

        let weights = build_weights_gpu(
            &encoding.vars,
            &leaf_probs,
            &choice_true,
            &choice_false,
            &evidence_by_var,
            &provider,
        )?;
        let random_var_count = leaf_probs_host
            .len()
            .checked_add(choice_true_host.len())
            .ok_or_else(|| XlogError::Compilation("random var count overflow".to_string()))?;
        let random_var_count = u32::try_from(random_var_count)
            .map_err(|_| XlogError::Compilation("random var count exceeds u32".to_string()))?;
        let num_leaf_probs = u32::try_from(leaf_probs_host.len())
            .map_err(|_| XlogError::Compilation("leaf_probs count exceeds u32".to_string()))?;
        let num_choice_probs = u32::try_from(choice_true_host.len())
            .map_err(|_| XlogError::Compilation("choice_probs count exceeds u32".to_string()))?;
        let (random_var_list, actual_random_var_count) = collect_random_vars_device(
            &provider,
            &encoding.vars,
            num_leaf_probs,
            num_choice_probs,
            random_var_count,
        )?;
        let random_vars =
            DeviceRandomVarList::from_device(random_var_list, actual_random_var_count)?;

        let compile_config = default_compile_config(&encoding.cnf, config.memory_bytes)?;
        let cache_config = default_cache_config(&encoding.cnf, &compile_config)?;

        let mut cache = GpuCircuitCache::new(&provider, cache_config)?;
        let (handle, compile_profile) = compile_gpu_d4_and_verify_cached(
            &encoding.cnf,
            &encoding.decision_var_limit,
            &provider,
            &compile_config,
            &mut cache,
            &random_vars,
            Some(canonical_cnf_hash),
        )?;
        cache.store_weights(&handle, &weights.log_true, &weights.log_false)?;

        #[cfg(feature = "host-io")]
        if !query_nodes.is_empty() {
            let mut node_ids: Vec<u32> = Vec::with_capacity(query_nodes.len());
            for (_idx, node) in &query_nodes {
                node_ids.push(node.as_u32());
            }
            let node_ids_device = upload_u32(&provider, &node_ids)?;
            let vars_device = map_nodes_to_vars_gpu(
                &encoding.vars.node_var,
                &node_ids_device,
                encoding.vars.max_var,
                &provider,
            )?;

            let mut vars_host = vec![0u32; vars_device.len()];
            provider
                .device()
                .inner()
                .dtoh_sync_copy_into(&vars_device, &mut vars_host)
                .map_err(|e| XlogError::Kernel(format!("Failed to read query vars: {}", e)))?;

            for (i, (query_idx, _)) in query_nodes.iter().enumerate() {
                let var = vars_host[i];
                queries[*query_idx].var = Some(var);
            }
        }

        let state = GpuExactState::new(provider, cache, handle)?;

        Ok(Self {
            gpu: Some(Arc::new(state)),
            queries,
            random_vars: Some(Arc::new(random_vars)),
            max_var: encoding.vars.max_var,
            gpu_config: config,
            last_compile_profile: compile_profile,
        })
    }

    #[cfg(feature = "host-io")]
    fn eval_log_z_gpu(&self, query_true: Option<u32>) -> Result<f64> {
        let state = self.gpu_state()?;
        let mut cache = state
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some(var) = query_true {
            if var == 0 || var > self.max_var {
                return Err(XlogError::Compilation(format!(
                    "Exact inference error: query var {} out of bounds (max_var={})",
                    var, self.max_var
                )));
            }
        }

        let mut restore = None;
        if let Some(var) = query_true {
            let mut buf = state.provider.memory().alloc::<f64>(1)?;
            {
                let (_, var_log_false) = cache.var_log_weights_mut();
                force_query_var_false(state.provider(), var_log_false, var, &mut buf)?;
            }
            restore = Some((var, buf));
        }

        let mut out_log_z = state.provider.memory().alloc::<f64>(1)?;
        let eval_result = cache.eval_log_wmc_device_inplace(state.handle(), &mut out_log_z);

        if let Some((var, buf)) = restore {
            let (_, var_log_false) = cache.var_log_weights_mut();
            let restore_result =
                restore_query_var_false(state.provider(), var_log_false, var, &buf);
            if let Err(err) = eval_result {
                restore_result?;
                return Err(err);
            }
            restore_result?;
        } else {
            eval_result?;
        }

        let mut host = [0.0f64];
        state
            .provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&out_log_z, &mut host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read logZ: {}", e)))?;
        Ok(host[0])
    }

    fn gpu_state(&self) -> Result<Arc<GpuExactState>> {
        self.gpu.clone().ok_or_else(|| {
            XlogError::Execution(
                "Exact inference GPU error: program has no compiled circuit".to_string(),
            )
        })
    }

    #[cfg(feature = "host-io")]
    fn eval_log_z_and_grads_gpu_cached(
        &self,
        query_true: Option<u32>,
    ) -> Result<(f64, Vec<f64>, Vec<f64>)> {
        let state = self.gpu_state()?;
        let mut cache = state
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some(var) = query_true {
            if var == 0 || var > self.max_var {
                return Err(XlogError::Compilation(format!(
                    "Exact inference error: query var {} out of bounds (max_var={})",
                    var, self.max_var
                )));
            }
        }

        let mut restore = None;
        if let Some(var) = query_true {
            let mut buf = state.provider.memory().alloc::<f64>(1)?;
            {
                let (_, var_log_false) = cache.var_log_weights_mut();
                force_query_var_false(state.provider(), var_log_false, var, &mut buf)?;
            }
            restore = Some((var, buf));
        }

        let eval_result = cache.eval_grads_inplace(state.handle());

        if let Some((var, buf)) = restore {
            let (_, var_log_false) = cache.var_log_weights_mut();
            let restore_result =
                restore_query_var_false(state.provider(), var_log_false, var, &buf);
            if let Err(err) = eval_result {
                restore_result?;
                return Err(err);
            }
            restore_result?;
        } else {
            eval_result?;
        }

        let weights_len = if self.max_var == 0 {
            0
        } else {
            (self.max_var as usize) + 1
        };

        let device = state.provider.device().inner();
        let mut host_grad_true: Vec<f64> = vec![0.0; weights_len];
        let mut host_grad_false: Vec<f64> = vec![0.0; weights_len];

        let root_idx = state.handle().root() as usize;
        let root_view = cache.values().slice(root_idx..(root_idx + 1));
        let mut log_z = [0.0_f64];
        device
            .dtoh_sync_copy_into(&root_view, &mut log_z)
            .map_err(|e| XlogError::Kernel(format!("Failed to read logZ: {}", e)))?;

        // Gradient buffers are multi-slot: [slot0_var0..slot0_varN, slot1_var0..].
        // Slice into the correct slot to download only this circuit's gradients.
        let var_stride = cache.var_stride()? as usize;
        let slot = state.handle().slot_index() as usize;
        let grad_start = slot * var_stride;
        let grad_end = grad_start + weights_len;
        let grad_true_slot = cache.grad_true().slice(grad_start..grad_end);
        let grad_false_slot = cache.grad_false().slice(grad_start..grad_end);
        device
            .dtoh_sync_copy_into(&grad_true_slot, &mut host_grad_true)
            .map_err(|e| XlogError::Kernel(format!("Failed to download grad_true: {}", e)))?;
        device
            .dtoh_sync_copy_into(&grad_false_slot, &mut host_grad_false)
            .map_err(|e| XlogError::Kernel(format!("Failed to download grad_false: {}", e)))?;

        Ok((log_z[0], host_grad_true, host_grad_false))
    }
}

fn force_query_var_false(
    provider: &Arc<CudaKernelProvider>,
    log_false: &mut TrackedCudaSlice<f64>,
    var: u32,
    restore: &mut TrackedCudaSlice<f64>,
) -> Result<()> {
    let device = provider.device().inner();
    let func = device
        .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_FORCE_VAR_FALSE)
        .ok_or_else(|| XlogError::Kernel("weights_force_var_false kernel not found".to_string()))?;
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (var, log_false, restore),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("weights_force_var_false failed: {}", e)))?;
    Ok(())
}

fn restore_query_var_false(
    provider: &Arc<CudaKernelProvider>,
    log_false: &mut TrackedCudaSlice<f64>,
    var: u32,
    restore: &TrackedCudaSlice<f64>,
) -> Result<()> {
    let device = provider.device().inner();
    let func = device
        .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_RESTORE_VAR_FALSE)
        .ok_or_else(|| {
            XlogError::Kernel("weights_restore_var_false kernel not found".to_string())
        })?;
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (var, log_false, restore),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("weights_restore_var_false failed: {}", e)))?;
    Ok(())
}

fn force_query_var_true(
    provider: &Arc<CudaKernelProvider>,
    log_true: &mut TrackedCudaSlice<f64>,
    var: u32,
    restore: &mut TrackedCudaSlice<f64>,
) -> Result<()> {
    let device = provider.device().inner();
    let func = device
        .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_FORCE_VAR_TRUE)
        .ok_or_else(|| XlogError::Kernel("weights_force_var_true kernel not found".to_string()))?;
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (var, log_true, restore),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("weights_force_var_true failed: {}", e)))?;
    Ok(())
}

fn restore_query_var_true(
    provider: &Arc<CudaKernelProvider>,
    log_true: &mut TrackedCudaSlice<f64>,
    var: u32,
    restore: &TrackedCudaSlice<f64>,
) -> Result<()> {
    let device = provider.device().inner();
    let func = device
        .get_func(WEIGHTS_MODULE, weights_kernels::WEIGHTS_RESTORE_VAR_TRUE)
        .ok_or_else(|| {
            XlogError::Kernel("weights_restore_var_true kernel not found".to_string())
        })?;
    unsafe {
        func.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (var, log_true, restore),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("weights_restore_var_true failed: {}", e)))?;
    Ok(())
}

pub(crate) fn default_compile_config(
    cnf: &xlog_solve::GpuCnf,
    memory_bytes: u64,
) -> Result<GpuCompileConfig> {
    // Must match the default GPU D4 configuration expected by the Python training paths.
    // Sizing is conservative and strictly bounded by `GpuCompileConfig::{smooth_node_cap,smooth_edge_cap}`.
    let frontier_depth: u16 = 6;

    let var_cap = cnf.var_cap.max(1);
    let trail_bytes_per_item = (var_cap as u64)
        .checked_add(1)
        .and_then(|v| v.checked_mul(std::mem::size_of::<i32>() as u64))
        .ok_or_else(|| XlogError::Compilation("trail size overflow".to_string()))?;
    let denom = trail_bytes_per_item.saturating_mul(8).max(1);
    let max_items_by_trail = memory_bytes / denom;
    let max_frontier_items = max_items_by_trail.clamp(8, 4096).min(u64::from(u32::MAX)) as u32;

    // The Phase 1 D4 compiler emits one leaf circuit per frontier item; caps must scale with the
    // maximum frontier size (up to 2^frontier_depth, bounded by max_frontier_items).
    let frontier_cap_factor = (1u64
        .checked_shl(frontier_depth as u32)
        .unwrap_or(u64::from(u32::MAX)))
    .min(u64::from(max_frontier_items)) as u32;

    let per_item_nodes = cnf
        .var_cap
        .checked_mul(5)
        .ok_or_else(|| XlogError::Compilation("smooth_node_cap overflow".to_string()))?
        .max(1024);
    let smooth_node_cap = per_item_nodes
        .checked_mul(frontier_cap_factor)
        .ok_or_else(|| XlogError::Compilation("smooth_node_cap overflow".to_string()))?;

    // Edge capacity scales with node capacity; AND/OR fanout grows edges but stays within a small
    // multiple of nodes for the compiler's Phase 1 emission patterns.
    let mut smooth_edge_cap = smooth_node_cap
        .checked_mul(2)
        .ok_or_else(|| XlogError::Compilation("smooth_edge_cap overflow".to_string()))?;
    if smooth_edge_cap < max_frontier_items {
        smooth_edge_cap = max_frontier_items;
    }

    // The verifier's UNSAT certificate (resolution trace) can be large even when the source CNF
    // is small, because equivalence checking builds CNF(C) with many Tseitin variables/clauses.
    // Allocate a larger share of the budget to the GPU CDCL arenas to avoid deterministic
    // overflow errors in production verifier paths.
    let mut cdcl_learned_bytes = memory_bytes / 8;
    if cdcl_learned_bytes < 4 * 1024 * 1024 {
        cdcl_learned_bytes = 4 * 1024 * 1024;
    }

    Ok(GpuCompileConfig {
        frontier_depth,
        max_frontier_items,
        max_depth: 128,
        smooth_node_cap,
        smooth_edge_cap,
        cdcl_restart_interval: 64,
        cdcl_learned_bytes,
        cdcl_conflict_budget: None,
        incremental_verify: false,
    })
}

pub(crate) fn default_cache_config(
    cnf: &xlog_solve::GpuCnf,
    compile: &GpuCompileConfig,
) -> Result<GpuCircuitCacheConfig> {
    if compile.smooth_node_cap == 0 || compile.smooth_edge_cap == 0 {
        return Err(XlogError::Compilation(
            "GPU cache config requires non-zero smoothing caps".to_string(),
        ));
    }
    Ok(GpuCircuitCacheConfig {
        num_slots: 4, // Hold 4 circuit templates; power-of-2 hash table.
        table_size: 8,
        node_cap: compile.smooth_node_cap,
        edge_cap: compile.smooth_edge_cap,
        level_cap: compile.smooth_node_cap,
        var_cap: cnf.var_cap,
    })
}

pub(crate) fn build_weight_sources(
    provenance: &Provenance,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let max_leaf = provenance.leaf_probs.keys().map(|leaf| leaf.as_u32()).max();
    let leaf_len = max_leaf.map(|v| v as usize + 1).unwrap_or(0);
    let mut leaf_probs = vec![0.0f64; leaf_len];
    let mut leaf_seen = vec![false; leaf_len];
    for (leaf, p) in &provenance.leaf_probs {
        let idx = leaf.as_u32() as usize;
        if idx >= leaf_len {
            return Err(XlogError::Compilation(
                "leaf probability index out of bounds".to_string(),
            ));
        }
        leaf_probs[idx] = *p;
        leaf_seen[idx] = true;
    }
    if let Some((idx, _)) = leaf_seen.iter().enumerate().find(|(_, seen)| !**seen) {
        return Err(XlogError::Compilation(format!(
            "missing probability for leaf {}",
            idx
        )));
    }

    let max_choice = provenance
        .choice_probs
        .keys()
        .map(|choice| choice.as_u32())
        .max();
    let choice_len = max_choice.map(|v| v as usize + 1).unwrap_or(0);
    let mut choice_true = vec![0.0f64; choice_len];
    let mut choice_false = vec![0.0f64; choice_len];
    let mut choice_seen = vec![false; choice_len];
    for (choice, (pt, pf)) in &provenance.choice_probs {
        let idx = choice.as_u32() as usize;
        if idx >= choice_len {
            return Err(XlogError::Compilation(
                "choice probability index out of bounds".to_string(),
            ));
        }
        choice_true[idx] = *pt;
        choice_false[idx] = *pf;
        choice_seen[idx] = true;
    }
    if let Some((idx, _)) = choice_seen.iter().enumerate().find(|(_, seen)| !**seen) {
        return Err(XlogError::Compilation(format!(
            "missing probability for choice {}",
            idx
        )));
    }

    Ok((leaf_probs, choice_true, choice_false))
}

pub(crate) fn upload_u32(
    provider: &Arc<CudaKernelProvider>,
    host: &[u32],
) -> Result<TrackedCudaSlice<u32>> {
    let memory = provider.memory();
    let mut buf = memory.alloc::<u32>(host.len())?;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(host, &mut buf)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload u32 buffer: {}", e)))?;
    Ok(buf)
}

pub(crate) fn upload_u8(
    provider: &Arc<CudaKernelProvider>,
    host: &[u8],
) -> Result<TrackedCudaSlice<u8>> {
    let memory = provider.memory();
    let mut buf = memory.alloc::<u8>(host.len())?;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(host, &mut buf)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload u8 buffer: {}", e)))?;
    Ok(buf)
}

pub(crate) fn upload_f64(
    provider: &Arc<CudaKernelProvider>,
    host: &[f64],
) -> Result<TrackedCudaSlice<f64>> {
    let memory = provider.memory();
    let mut buf = memory.alloc::<f64>(host.len())?;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(host, &mut buf)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload f64 buffer: {}", e)))?;
    Ok(buf)
}

fn capture_compact_count_device(
    provider: &Arc<CudaKernelProvider>,
    prefix_sum: &TrackedCudaSlice<u32>,
    mask: &TrackedCudaSlice<u8>,
    n: u32,
) -> Result<TrackedCudaSlice<u32>> {
    let mut out = provider.memory().alloc::<u32>(1)?;
    let device = provider.device().inner();
    let capture_fn = device
        .get_func(FILTER_MODULE, filter_kernels::CAPTURE_COMPACT_COUNT)
        .ok_or_else(|| XlogError::Kernel("capture_compact_count kernel not found".to_string()))?;
    unsafe {
        capture_fn.clone().launch(
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            },
            (prefix_sum, mask, n, &mut out),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("capture_compact_count failed: {}", e)))?;
    Ok(out)
}

pub(crate) fn collect_random_vars_device(
    provider: &Arc<CudaKernelProvider>,
    vars: &GpuCnfVarTables,
    num_leaf_probs: u32,
    num_choice_probs: u32,
    _expected_count: u32,
) -> Result<(TrackedCudaSlice<u32>, u32)> {
    let device = provider.device().inner();
    let memory = provider.memory();

    let mask_len = vars
        .max_var
        .checked_add(1)
        .ok_or_else(|| XlogError::Compilation("random var mask_len overflow".to_string()))?;
    let mask_len_usize = usize::try_from(mask_len)
        .map_err(|_| XlogError::Compilation("random var mask_len exceeds usize".to_string()))?;

    let mut mask = memory.alloc::<u8>(mask_len_usize)?;
    device
        .memset_zeros(&mut mask)
        .map_err(|e| XlogError::Kernel(format!("Failed to zero random var mask: {}", e)))?;

    let mut iota = memory.alloc::<u32>(mask_len_usize)?;
    let fill_iota = device
        .get_func(FILTER_MODULE, filter_kernels::FILL_U32_IOTA)
        .ok_or_else(|| XlogError::Kernel("fill_u32_iota kernel not found".to_string()))?;
    let block_size = 256u32;
    let grid = (mask_len + block_size - 1) / block_size;
    unsafe {
        fill_iota.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (&mut iota, mask_len, 0u32),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("fill_u32_iota failed: {}", e)))?;

    // Only iterate over the probabilistic entries — leaf_var and choice_var are allocated
    // to num_nodes but only the first num_leaf_probs / num_choice_probs entries correspond
    // to variables with actual probabilities. Non-probabilistic PIR leaf nodes also get
    // CNF variables but must NOT be marked as random.
    let leaf_len = num_leaf_probs;
    let choice_len = num_choice_probs;

    let mark_kernel = device
        .get_func(FILTER_MODULE, filter_kernels::MARK_RANDOM_VARS)
        .ok_or_else(|| XlogError::Kernel("mark_random_vars kernel not found".to_string()))?;
    let mark_n = leaf_len.max(choice_len);
    if mark_n > 0 {
        let grid = (mark_n + block_size - 1) / block_size;
        unsafe {
            mark_kernel.clone().launch(
                LaunchConfig {
                    grid_dim: (grid, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &vars.leaf_var,
                    &vars.choice_var,
                    leaf_len,
                    choice_len,
                    &mut mask,
                    mask_len,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("mark_random_vars failed: {}", e)))?;
    }

    let prefix_sum = provider.scan_u8_mask_device(&mask, mask_len)?;
    let count_device = capture_compact_count_device(provider, &prefix_sum, &mask, mask_len)?;

    // Read the actual random var count from device (the GPU scan result is authoritative).
    // The host-side expected_count can be wrong when some ChoiceVarIds are unreachable
    // from query/evidence roots and don't get assigned CNF variables.
    let actual_count = {
        let mut buf = vec![0u32; 1];
        device
            .dtoh_sync_copy_into(&count_device, &mut buf)
            .map_err(|e| XlogError::Kernel(format!("dtoh count_device failed: {}", e)))?;
        buf[0]
    };

    if actual_count == 0 {
        // No random variables in the circuit — return empty list.
        let out = provider.memory().alloc::<u32>(0)?;
        return Ok((out, 0));
    }

    let mut out = memory.alloc::<u32>(mask_len_usize)?;
    let compact_fn = device
        .get_func(FILTER_MODULE, filter_kernels::COMPACT_U32_BY_MASK)
        .ok_or_else(|| XlogError::Kernel("compact_u32_by_mask kernel not found".to_string()))?;
    unsafe {
        compact_fn.clone().launch(
            LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (&iota, &mask, &prefix_sum, mask_len, &mut out),
        )
    }
    .map_err(|e| XlogError::Kernel(format!("compact_u32_by_mask failed: {}", e)))?;

    Ok((out, actual_count))
}

fn display_atom(atom: &GroundAtom) -> String {
    if atom.args.is_empty() {
        format!("{}()", atom.predicate)
    } else {
        format!("{}({} args)", atom.predicate, atom.args.len())
    }
}

#[cfg(all(test, feature = "host-io"))]
mod tests {
    use super::*;
    use xlog_cuda::CudaDevice;

    #[test]
    fn test_exact_negation_probability() {
        if CudaDevice::new(0).is_err() {
            eprintln!("Skipping test: CUDA runtime unavailable");
            return;
        }
        // 0.3::rain(). dry() :- not rain().
        // P(dry) = P(not rain) = 1 - 0.3 = 0.7
        let source = r#"
0.3::rain().
dry() :- not rain().
query(dry()).
"#;

        let program = ExactDdnnfProgram::compile_source(source).unwrap();
        let result = program.evaluate().unwrap();

        assert_eq!(result.query_probs.len(), 1);
        let dry_prob = result.query_probs[0].prob;
        assert!(
            (dry_prob - 0.7).abs() < 1e-6,
            "P(dry) should be 0.7, got {}",
            dry_prob
        );
    }

    #[test]
    fn test_exact_multi_layer_negation() {
        if CudaDevice::new(0).is_err() {
            eprintln!("Skipping test: CUDA runtime unavailable");
            return;
        }
        // 0.4::c(). b() :- not c(). a() :- not b().
        // P(b) = P(not c) = 0.6
        // P(a) = P(not b) = 0.4
        let source = r#"
0.4::c().
b() :- not c().
a() :- not b().
query(a()).
"#;

        let program = ExactDdnnfProgram::compile_source(source).unwrap();
        let result = program.evaluate().unwrap();

        assert_eq!(result.query_probs.len(), 1);
        let a_prob = result.query_probs[0].prob;
        assert!(
            (a_prob - 0.4).abs() < 1e-6,
            "P(a) should be 0.4, got {}",
            a_prob
        );
    }

    #[test]
    fn test_eval_log_z_changes_for_sprinkler_given_wet() {
        if CudaDevice::new(0).is_err() {
            eprintln!("Skipping test: CUDA runtime unavailable");
            return;
        }

        let source = r#"
0.7::rain().
0.2::sprinkler().
wet() :- rain().
wet() :- sprinkler().
evidence(wet(), true).
query(rain()).
query(sprinkler()).
"#;

        let program = ExactDdnnfProgram::compile_source(source).unwrap();
        let log_z_e = program.eval_log_z_gpu(None).unwrap();
        let sprinkler_var = program.query_var(1).unwrap();
        let log_z_eq = program.eval_log_z_gpu(Some(sprinkler_var)).unwrap();

        let state = program.gpu_state().unwrap();
        let mut cache = state
            .cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (_, var_log_false) = cache.var_log_weights_mut();

        let mut before = [0.0f64];
        let view = var_log_false.slice(sprinkler_var as usize..(sprinkler_var as usize + 1));
        state
            .provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&view, &mut before)
            .unwrap();

        let mut restore = state.provider.memory().alloc::<f64>(1).unwrap();
        force_query_var_false(state.provider(), var_log_false, sprinkler_var, &mut restore)
            .unwrap();

        let mut after = [0.0f64];
        let view_after = var_log_false.slice(sprinkler_var as usize..(sprinkler_var as usize + 1));
        state
            .provider
            .device()
            .inner()
            .dtoh_sync_copy_into(&view_after, &mut after)
            .unwrap();

        restore_query_var_false(state.provider(), var_log_false, sprinkler_var, &restore).unwrap();

        assert!(
            before[0].is_finite(),
            "expected finite log_false before forcing"
        );
        assert!(
            after[0].is_infinite() && after[0].is_sign_negative(),
            "expected -inf log_false after forcing, got {}",
            after[0]
        );
        assert!(
            log_z_eq < log_z_e,
            "conditioning on sprinkler should reduce logZ (log_z_e={}, log_z_eq={})",
            log_z_e,
            log_z_eq
        );
    }
}
