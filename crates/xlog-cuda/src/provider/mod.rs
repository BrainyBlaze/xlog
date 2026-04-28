//! CUDA kernel provider implementation
//!
//! This module provides the `CudaKernelProvider` which manages pre-compiled
//! PTX kernels for GPU execution of relational operations (join, dedup, groupby).

use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use std::ffi::c_void;
use xlog_core::{Result, Schema, XlogError};

use crate::{
    cuda_compat::{
        AsKernelParam, DeviceParamStorage, DevicePtr, DeviceRepr, DeviceSlice,
        IntoKernelParamStorage, LaunchAsync, LaunchConfig,
    },
    memory::{validate_logical_row_count, CudaColumn, TrackedCudaSlice},
    CudaBuffer, CudaDevice, CudaStream, CudaViewMut, GpuMemoryManager,
};

mod arithmetic;
mod filter;
mod groupby;
mod ilp;
mod ilp_exact;
mod io;
mod kernel_loading;
pub mod kernel_paths;
mod probabilistic;
mod relational;
mod transfer;

/// Per-module PTX load timing (populated only when XLOG_WARMUP_PROFILE=1).
#[derive(Debug, Clone, Default)]
pub struct PtxLoadProfile {
    pub total_sec: f64,
    pub per_module_sec: Vec<(String, f64)>,
    pub cubin_loaded: u32,
    pub ptx_fallback: u32,
}

fn warmup_profiling_enabled() -> bool {
    std::env::var("XLOG_WARMUP_PROFILE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Detect device compute capability as a two-digit number (e.g. 75, 80, 120).
fn detect_compute_capability(device: &Arc<CudaDevice>) -> Result<u32> {
    let major = device
        .inner()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .map_err(|e| XlogError::Kernel(format!("Failed to query SM major: {}", e)))?;
    let minor = device
        .inner()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .map_err(|e| XlogError::Kernel(format!("Failed to query SM minor: {}", e)))?;
    Ok((major as u32) * 10 + (minor as u32))
}

#[cfg(test)]
fn resolve_module_path(name: &str, cc: u32) -> Option<(std::path::PathBuf, bool)> {
    kernel_paths::KernelArtifactLocator::from_env().resolve_module_path(name, cc)
}

#[derive(Debug)]
pub(crate) enum KernelModuleSource {
    File { path: PathBuf, is_cubin: bool },
    EmbeddedPortablePtx { ptx: &'static str },
}

fn resolve_module_source_with_locator(
    name: &str,
    cc: u32,
    locator: &kernel_paths::KernelArtifactLocator,
) -> Option<KernelModuleSource> {
    if let Some((path, is_cubin)) = locator.resolve_module_path(name, cc) {
        return Some(KernelModuleSource::File { path, is_cubin });
    }

    crate::embedded_kernel_data::portable_ptx(name)
        .map(|ptx| KernelModuleSource::EmbeddedPortablePtx { ptx })
}

fn resolve_module_source(name: &str, cc: u32) -> Option<KernelModuleSource> {
    let locator = kernel_paths::KernelArtifactLocator::from_env();
    resolve_module_source_with_locator(name, cc, &locator)
}

/// Resolve a kernel module from sidecar artifacts or embedded portable PTX.
///
/// Asserts (in debug builds) that `name` is present in the kernel manifest,
/// catching name/order drift between the manifest and provider load blocks.
fn load_module_source(name: &str, cc: u32) -> Result<KernelModuleSource> {
    debug_assert!(
        crate::kernel_manifest_data::KERNEL_CU_NAMES.contains(&name),
        "kernel module '{name}' is not in KERNEL_CU_NAMES manifest — update kernel_manifest_data.rs"
    );
    resolve_module_source(name, cc).ok_or_else(|| {
        XlogError::Kernel(format!(
            "{name}: no cubin, sidecar portable PTX, or embedded portable PTX found"
        ))
    })
}

#[derive(Clone)]
pub(crate) struct RawCudaView<'a, T> {
    ptr: cudarc::driver::sys::CUdeviceptr,
    len: usize,
    stream: Arc<CudaStream>,
    _marker: PhantomData<&'a [T]>,
}

impl<'a, T> DeviceSlice<T> for RawCudaView<'a, T> {
    fn len(&self) -> usize {
        self.len
    }

    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl<'a, T> DevicePtr<T> for RawCudaView<'a, T> {
    fn device_ptr<'b>(
        &'b self,
        _stream: &'b CudaStream,
    ) -> (
        cudarc::driver::sys::CUdeviceptr,
        cudarc::driver::SyncOnDrop<'b>,
    ) {
        (self.ptr, cudarc::driver::SyncOnDrop::Sync(None))
    }
}

impl<'a, T> RawCudaView<'a, T> {
    pub fn device_ptr(&self) -> &cudarc::driver::sys::CUdeviceptr {
        &self.ptr
    }
}

impl<'a, T: DeviceRepr> AsKernelParam for &RawCudaView<'a, T> {
    fn as_kernel_param(&self) -> *mut c_void {
        ((*self).device_ptr() as *const cudarc::driver::sys::CUdeviceptr)
            .cast_mut()
            .cast()
    }
}

impl<'a, T: DeviceRepr> IntoKernelParamStorage for &'a RawCudaView<'a, T> {
    type Storage = DeviceParamStorage<'a>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        DeviceParamStorage::unsynced(self.ptr)
    }
}

/// Scratch buffers for stable radix sorting of u32 key/value pairs.
pub struct RadixSortScratch {
    keys_b: TrackedCudaSlice<u32>,
    values_b: TrackedCudaSlice<u32>,
    hist: TrackedCudaSlice<u32>,
    prefix: TrackedCudaSlice<u32>,
    ranks: TrackedCudaSlice<u32>,
    len: u32,
}

impl RadixSortScratch {
    pub fn new(provider: &CudaKernelProvider, n: u32) -> Result<Self> {
        let memory = provider.memory();
        let len = n.max(1);
        let keys_b = memory.alloc::<u32>(len as usize)?;
        let values_b = memory.alloc::<u32>(len as usize)?;
        let ranks = memory.alloc::<u32>(len as usize)?;
        let block_size = CudaKernelProvider::SORT_BLOCK_SIZE;
        let grid_size = ((len + block_size - 1) / block_size).max(1);
        let hist = memory.alloc::<u32>((grid_size as usize) * 16)?;
        let prefix = memory.alloc::<u32>(16)?;
        Ok(Self {
            keys_b,
            values_b,
            hist,
            prefix,
            ranks,
            len,
        })
    }

    pub fn ensure_capacity(&mut self, provider: &CudaKernelProvider, n: u32) -> Result<()> {
        if n <= self.len {
            return Ok(());
        }
        *self = Self::new(provider, n)?;
        Ok(())
    }
}

/// Module names for loaded PTX modules
pub const JOIN_MODULE: &str = "xlog_join";
pub const DEDUP_MODULE: &str = "xlog_dedup";
pub const GROUPBY_MODULE: &str = "xlog_groupby";
pub const SCAN_MODULE: &str = "xlog_scan";
pub const SORT_MODULE: &str = "xlog_sort";
pub const FILTER_MODULE: &str = "xlog_filter";
pub const SET_OPS_MODULE: &str = "xlog_set_ops";
pub const PACK_MODULE: &str = "xlog_pack";
pub const CIRCUIT_MODULE: &str = "xlog_circuit";
pub const MC_SAMPLE_MODULE: &str = "xlog_mc_sample";
pub const MC_EVAL_MODULE: &str = "xlog_mc_eval";
pub const ARITH_MODULE: &str = "xlog_arith";
pub const SAT_MODULE: &str = "xlog_sat";
pub const D4_MODULE: &str = "xlog_d4";
pub const NEURAL_MODULE: &str = "xlog_neural";
pub const PIR_MODULE: &str = "xlog_pir";
pub const CNF_MODULE: &str = "xlog_cnf";
pub const CACHE_MODULE: &str = "xlog_cache";
pub const WEIGHTS_MODULE: &str = "xlog_weights";
pub const ILP_MODULE: &str = "xlog_ilp";
pub const ILP_CREDIT_MODULE: &str = "xlog_ilp_credit";
pub const ILP_EXACT_MODULE: &str = "xlog_ilp_exact";

// Compile-time check: kernel manifest lists exactly 22 modules.
const _: () = assert!(crate::kernel_manifest_data::KERNEL_CU_NAMES.len() == 22);

/// Kernel function names in the Monte Carlo sampling module
pub mod mc_sample_kernels {
    pub const MC_SAMPLE_BERNOULLI: &str = "mc_sample_bernoulli";
}

/// Kernel function names in the Monte Carlo evaluation module
pub mod mc_eval_kernels {
    pub const MC_EVAL_MASK_VAR: &str = "mc_eval_mask_var";
    pub const MC_EVAL_MASK_AD: &str = "mc_eval_mask_ad_choice";
    pub const MC_EVAL_QUERY_EVIDENCE_TRUTH: &str = "mc_eval_query_evidence_truth";
    pub const MC_EVAL_ACCUMULATE_COUNTS: &str = "mc_accumulate_counts";
}

/// Kernel function names in the arithmetic module
pub mod arith_kernels {
    pub const ARITH_BINARY_I64: &str = "arith_binary_i64";
    pub const ARITH_BINARY_I32: &str = "arith_binary_i32";
    pub const ARITH_BINARY_U64: &str = "arith_binary_u64";
    pub const ARITH_BINARY_U32: &str = "arith_binary_u32";
    pub const ARITH_BINARY_F64: &str = "arith_binary_f64";
    pub const ARITH_BINARY_F32: &str = "arith_binary_f32";
    pub const ARITH_ABS_I64: &str = "arith_abs_i64";
    pub const ARITH_ABS_I32: &str = "arith_abs_i32";
    pub const ARITH_ABS_F64: &str = "arith_abs_f64";
    pub const ARITH_ABS_F32: &str = "arith_abs_f32";
    pub const ARITH_POW_F64: &str = "arith_pow_f64";
    pub const ARITH_CAST: &str = "arith_cast";
    pub const ARITH_FILL_CONST_U32: &str = "arith_fill_const_u32";
    pub const ARITH_FILL_CONST_U64: &str = "arith_fill_const_u64";
    pub const ARITH_FILL_CONST_I64: &str = "arith_fill_const_i64";
    pub const ARITH_FILL_CONST_I32: &str = "arith_fill_const_i32";
    pub const ARITH_FILL_CONST_F64: &str = "arith_fill_const_f64";
    pub const ARITH_FILL_CONST_F32: &str = "arith_fill_const_f32";
    pub const ARITH_FILL_CONST_U8: &str = "arith_fill_const_u8";
    // Conditional select kernels
    pub const ARITH_SELECT_I64: &str = "arith_select_i64";
    pub const ARITH_SELECT_I32: &str = "arith_select_i32";
    pub const ARITH_SELECT_U64: &str = "arith_select_u64";
    pub const ARITH_SELECT_U32: &str = "arith_select_u32";
    pub const ARITH_SELECT_F64: &str = "arith_select_f64";
    pub const ARITH_SELECT_F32: &str = "arith_select_f32";
}

/// Kernel function names in the neural fast-path module.
pub mod neural_kernels {
    pub const NEURAL_FILL_AD_CHAIN_F32: &str = "neural_fill_ad_chain_f32";
    pub const NEURAL_SCATTER_AD_CHAIN_GRADS_F32: &str = "neural_scatter_ad_chain_grads_f32";
}

/// Kernel function names in the ILP module.
pub mod ilp_kernels {
    pub const EXTRACT_NONZERO_INDICES: &str = "extract_nonzero_indices";
    pub const ILP_MARK_SELECTED_IDS_U32: &str = "ilp_mark_selected_ids_u32";
    pub const ILP_MARK_SELECTED_IDS_I32: &str = "ilp_mark_selected_ids_i32";
    pub const ILP_MARK_SELECTED_IDS_I64: &str = "ilp_mark_selected_ids_i64";
    pub const ILP_MARK_SELECTED_IDS_U64: &str = "ilp_mark_selected_ids_u64";
    pub const ILP_VALIDATE_SELECTED_IDS_U32: &str = "ilp_validate_selected_ids_u32";
    pub const ILP_VALIDATE_SELECTED_IDS_I32: &str = "ilp_validate_selected_ids_i32";
    pub const ILP_VALIDATE_SELECTED_IDS_I64: &str = "ilp_validate_selected_ids_i64";
    pub const ILP_VALIDATE_SELECTED_IDS_U64: &str = "ilp_validate_selected_ids_u64";
    pub const ILP_BROADCAST_CANDIDATE_FLAG: &str = "ilp_broadcast_candidate_flag";
    pub const ILP_COO_FILL_FROM_MASK: &str = "ilp_coo_fill_from_mask";
    pub const ILP_CSR_HISTOGRAM: &str = "ilp_csr_histogram";
    pub const ILP_REDUCE_SUM_F32: &str = "ilp_reduce_sum_f32";
    pub const ILP_REDUCE_SUM_F64: &str = "ilp_reduce_sum_f64";
}

/// Kernel function names in the ILP credit module.
pub mod ilp_credit_kernels {
    pub const ILP_COO_FILL: &str = "ilp_coo_fill";
    pub const ILP_CREDIT_FORWARD_F32: &str = "ilp_credit_forward_f32";
    pub const ILP_CREDIT_FORWARD_F64: &str = "ilp_credit_forward_f64";
    pub const ILP_CREDIT_BACKWARD_F32: &str = "ilp_credit_backward_f32";
    pub const ILP_CREDIT_BACKWARD_F64: &str = "ilp_credit_backward_f64";
}

/// Kernel function names in the ILP exact-induction module (M8 Phase 1).
pub mod ilp_exact_kernels {
    pub const ILP_EXACT_SCORE: &str = "ilp_exact_score";
}

/// Kernel function names in the PIR interning module.
pub mod pir_kernels {
    pub const PIR_PACK_KEYS: &str = "pir_pack_keys";
    pub const PIR_HASH_KEYS: &str = "pir_hash_keys";
    pub const PIR_MARK_UNIQUE: &str = "pir_mark_unique";
    pub const PIR_FIND_EXISTING: &str = "pir_find_existing";
    pub const PIR_MARK_NEW_GROUPS: &str = "pir_mark_new_groups";
    pub const PIR_BUILD_GROUP_IDS: &str = "pir_build_group_ids";
    pub const PIR_FILL_CHILD_PARENTS: &str = "pir_fill_child_parents";
    pub const PIR_MARK_UNIQUE_PAIRS: &str = "pir_mark_unique_pairs";
    pub const PIR_COMPACT_PAIRS: &str = "pir_compact_pairs";
    pub const PIR_COUNT_CHILDREN: &str = "pir_count_children";
    pub const PIR_WRITE_CHILD_OFFSETS: &str = "pir_write_child_offsets";
    pub const PIR_GATHER_CHILDREN: &str = "pir_gather_children";
    pub const PIR_BUILD_GRAPH_CHILD_COUNTS: &str = "pir_build_graph_child_counts";
    pub const PIR_SUM_COUNTS: &str = "pir_sum_counts";
    pub const PIR_EMIT_NODES_AND_IDS: &str = "pir_emit_nodes_and_ids";
    pub const PIR_UPDATE_COUNTS: &str = "pir_update_counts";
}

/// Kernel function names in the GPU CNF encoder module.
pub mod cnf_kernels {
    pub const CNF_REACHABILITY_INIT: &str = "cnf_reachability_init";
    pub const CNF_REACHABILITY_BFS: &str = "cnf_reachability_bfs";
    pub const CNF_MARK_LEAF_CHOICE: &str = "cnf_mark_leaf_choice";
    pub const CNF_ASSIGN_LEAF_VAR: &str = "cnf_assign_leaf_var";
    pub const CNF_ASSIGN_CHOICE_VAR: &str = "cnf_assign_choice_var";
    pub const CNF_MARK_NODE_VARS: &str = "cnf_mark_node_vars";
    pub const CNF_COUNT_CLAUSES: &str = "cnf_count_clauses";
    pub const CNF_CAPTURE_LAST_COUNTS: &str = "cnf_capture_last_counts";
    pub const CNF_COMPUTE_LEAF_CHOICE_TOTALS: &str = "cnf_compute_leaf_choice_totals";
    pub const CNF_COMPUTE_TOTALS: &str = "cnf_compute_totals";
    pub const CNF_ASSIGN_NODE_VAR: &str = "cnf_assign_node_var";
    pub const CNF_EMIT_CLAUSES: &str = "cnf_emit_clauses";
    pub const CNF_SET_CLAUSE_END: &str = "cnf_set_clause_end";
}

/// Kernel function names in the weights module.
pub mod weights_kernels {
    pub const WEIGHTS_FILL_LEAF: &str = "weights_fill_leaf";
    pub const WEIGHTS_FILL_CHOICE: &str = "weights_fill_choice";
    pub const WEIGHTS_SET_EVIDENCE_FROM_NODES: &str = "weights_set_evidence_from_nodes";
    pub const WEIGHTS_APPLY_EVIDENCE: &str = "weights_apply_evidence";
    pub const WEIGHTS_MAP_NODES_TO_VARS: &str = "weights_map_nodes_to_vars";
    pub const WEIGHTS_FORCE_VAR_FALSE: &str = "weights_force_var_false";
    pub const WEIGHTS_RESTORE_VAR_FALSE: &str = "weights_restore_var_false";
    pub const WEIGHTS_FORCE_VAR_TRUE: &str = "weights_force_var_true";
    pub const WEIGHTS_RESTORE_VAR_TRUE: &str = "weights_restore_var_true";
    pub const WEIGHTS_COPY_SLOT_TO_BATCH: &str = "weights_copy_slot_to_batch";
    pub const WEIGHTS_APPLY_QUERY_VARS: &str = "weights_apply_query_vars";
    pub const WEIGHTS_RESTORE_QUERY_VARS: &str = "weights_restore_query_vars";
    pub const WEIGHTS_APPLY_QUERY_VARS_FALSE_BATCHED: &str =
        "weights_apply_query_vars_false_batched";
    pub const WEIGHTS_RESTORE_QUERY_VARS_FALSE_BATCHED: &str =
        "weights_restore_query_vars_false_batched";
    pub const WEIGHTS_APPLY_QUERY_VARS_TRUE_BATCHED: &str = "weights_apply_query_vars_true_batched";
    pub const WEIGHTS_RESTORE_QUERY_VARS_TRUE_BATCHED: &str =
        "weights_restore_query_vars_true_batched";
}

/// Kernel function names in the GPU D4 module (CNF validation + circuit levelization).
pub mod d4_kernels {
    pub const D4_VALIDATE_CNF: &str = "d4_validate_cnf";
    pub const D4_LEVELIZE_COUNTS: &str = "d4_levelize_counts";
    pub const D4_LEVELIZE_EMIT: &str = "d4_levelize_emit";
    // Task 4: BFS frontier expansion + unit propagation.
    pub const D4_FRONTIER_PREPARE: &str = "d4_frontier_prepare";
    pub const D4_FRONTIER_EXPAND: &str = "d4_frontier_expand";
    pub const D4_FRONTIER_PREPARE_DENSE: &str = "d4_frontier_prepare_dense";
    pub const D4_FRONTIER_EXPAND_DENSE: &str = "d4_frontier_expand_dense";
    // Task 5: per-frontier D4 DFS worker (count+emit).
    pub const D4_COMPILE_COUNT: &str = "d4_compile_count";
    pub const D4_COMPILE_EMIT: &str = "d4_compile_emit";
    pub const D4_CAPTURE_EMIT_META: &str = "d4_capture_emit_meta";
    // Task 6: GPU smoothing (random-var support + wrapper emission).
    pub const D4_SUPPORT_LEVEL: &str = "d4_support_level";
    pub const D4_SUPPORT_SET_ROOT_BITS: &str = "d4_support_set_root_bits";
    pub const D4_SMOOTH_COUNT: &str = "d4_smooth_count";
    pub const D4_SMOOTH_WRAPPER_COUNTS: &str = "d4_smooth_wrapper_counts";
    pub const D4_SMOOTH_WRAPPER_EDGE_COUNTS_OR: &str = "d4_smooth_wrapper_edge_counts_or";
    pub const D4_SMOOTH_WRAPPER_EDGE_COUNTS_DEC: &str = "d4_smooth_wrapper_edge_counts_dec";
    pub const D4_SMOOTH_INIT_NODES: &str = "d4_smooth_init_nodes";
    pub const D4_SMOOTH_EMIT_LEVEL: &str = "d4_smooth_emit_level";
    pub const D4_SMOOTH_CHECK_EDGE_CAP: &str = "d4_smooth_check_edge_cap";
    // Task 6: GPU free-var mask (vars in clauses vs circuit).
    pub const D4_MARK_VARS_IN_CLAUSES: &str = "d4_mark_vars_in_clauses";
    pub const D4_MARK_VARS_IN_CIRCUIT: &str = "d4_mark_vars_in_circuit";
    pub const D4_BUILD_FREE_VAR_MASK: &str = "d4_build_free_var_mask";
    // GPU-only assertions (tests + invariant enforcement without host reads).
    pub const D4_ASSERT_U32_EQ: &str = "d4_assert_u32_eq";
    pub const D4_ASSERT_BITSET_VAR: &str = "d4_assert_bitset_var";
    pub const D4_ASSERT_DENSE_VAR: &str = "d4_assert_dense_var";
    pub const D4_ASSERT_LEAF_ROOT_AND_DEGREE: &str = "d4_assert_leaf_root_and_degree";
}

/// Kernel function names in the join module
pub mod join_kernels {
    pub const HASH_JOIN_BUILD: &str = "hash_join_build";
    pub const HASH_JOIN_PROBE: &str = "hash_join_probe";
    // V2 kernels for multi-column joins
    pub const COMPUTE_COMPOSITE_HASH: &str = "compute_composite_hash";
    pub const HASH_JOIN_BUCKET_COUNT_V2: &str = "hash_join_bucket_count_v2";
    pub const HASH_JOIN_SCATTER_V2: &str = "hash_join_scatter_v2";
    pub const HASH_JOIN_PROBE_V2: &str = "hash_join_probe_v2";
    pub const HASH_JOIN_SEMI: &str = "hash_join_semi";
    pub const HASH_JOIN_ANTI: &str = "hash_join_anti";
    pub const INIT_HASH_TABLE: &str = "init_hash_table";
}

/// Kernel function names in the dedup module
pub mod dedup_kernels {
    pub const MARK_DUPLICATES: &str = "mark_duplicates";
    pub const MARK_UNIQUE_COLUMNAR: &str = "mark_unique_columnar";
    pub const MARK_UNIQUE_AND_SCAN_COLUMNAR: &str = "mark_unique_and_scan_columnar";
    pub const COMPACT_ROWS: &str = "compact_rows";
    pub const MARK_UNIQUE_FULL_ROW_BYTEWISE: &str = "mark_unique_full_row_bytewise";
    pub const MARK_DIFF_FULL_ROW_TYPED_SORTED: &str = "mark_diff_full_row_typed_sorted";
}

/// Kernel function names in the groupby module
pub mod groupby_kernels {
    pub const DETECT_GROUP_BOUNDARIES: &str = "detect_group_boundaries";
    pub const DETECT_BOUNDARIES: &str = "detect_boundaries";
    pub const EXTRACT_GROUP_KEYS: &str = "extract_group_keys";
    pub const GROUP_IDS_FROM_BOUNDARIES: &str = "group_ids_from_boundaries";
    pub const GROUP_START_INDICES: &str = "group_start_indices";
    pub const CAPTURE_NUM_GROUPS: &str = "capture_num_groups";
    pub const GROUPBY_COUNT: &str = "groupby_count";
    pub const GROUPBY_SUM: &str = "groupby_sum";
    pub const GROUPBY_MIN: &str = "groupby_min";
    pub const GROUPBY_MAX: &str = "groupby_max";
    pub const GROUPBY_LOGSUMEXP_MAX: &str = "groupby_logsumexp_max";
    pub const GROUPBY_LOGSUMEXP_SUMEXP: &str = "groupby_logsumexp_sumexp";
    pub const GROUPBY_LOGSUMEXP_FINAL: &str = "groupby_logsumexp_final";
}

/// Kernel function names in the scan module
pub mod scan_kernels {
    pub const BLOCK_INCLUSIVE_SCAN: &str = "block_inclusive_scan";
    pub const ADD_BLOCK_OFFSETS: &str = "add_block_offsets";
    pub const EXCLUSIVE_SCAN_MASK: &str = "exclusive_scan_mask";
    pub const COUNT_MASK: &str = "count_mask";
    // Multi-block scan kernels for large prefix sums
    pub const MULTIBLOCK_SCAN_PHASE1: &str = "multiblock_scan_phase1";
    pub const MULTIBLOCK_SCAN_U32_PHASE1: &str = "multiblock_scan_u32_phase1";
    pub const MULTIBLOCK_SCAN_PHASE2: &str = "multiblock_scan_phase2";
    pub const MULTIBLOCK_SCAN_PHASE3: &str = "multiblock_scan_phase3";
}

/// Kernel function names in the sort module
pub mod sort_kernels {
    pub const RADIX_HISTOGRAM: &str = "radix_histogram";
    pub const RADIX_SCATTER: &str = "radix_scatter";
    pub const COMPUTE_RANKS: &str = "compute_ranks";
    pub const RADIX_SCATTER_STABLE: &str = "radix_scatter_stable";
    pub const COMPUTE_DIGIT_PREFIX_SUMS: &str = "compute_digit_prefix_sums";
    pub const INIT_INDICES: &str = "init_indices";
    pub const APPLY_PERMUTATION_U32: &str = "apply_permutation_u32";
    pub const APPLY_PERMUTATION_BYTES: &str = "apply_permutation_bytes";

    pub const GATHER_KEYS_I32_ORDERED_U32: &str = "gather_keys_i32_ordered_u32";
    pub const GATHER_KEYS_F32_ORDERED_U32: &str = "gather_keys_f32_ordered_u32";
    pub const GATHER_KEYS_BOOL_ORDERED_U32: &str = "gather_keys_bool_ordered_u32";

    pub const GATHER_KEYS_U64_LO_U32: &str = "gather_keys_u64_lo_u32";
    pub const GATHER_KEYS_U64_HI_U32: &str = "gather_keys_u64_hi_u32";

    pub const GATHER_KEYS_I64_LO_U32: &str = "gather_keys_i64_lo_u32";
    pub const GATHER_KEYS_I64_HI_U32: &str = "gather_keys_i64_hi_u32";

    pub const GATHER_KEYS_F64_LO_U32: &str = "gather_keys_f64_lo_u32";
    pub const GATHER_KEYS_F64_HI_U32: &str = "gather_keys_f64_hi_u32";
}

/// Kernel function names in the filter module
pub mod filter_kernels {
    pub const FILTER_COMPARE_U32: &str = "filter_compare_u32";
    pub const FILTER_COMPARE_I64: &str = "filter_compare_i64";
    pub const FILTER_COMPARE_F64: &str = "filter_compare_f64";
    pub const FILTER_COMPARE_I32: &str = "filter_compare_i32";
    pub const FILTER_COMPARE_U64: &str = "filter_compare_u64";
    pub const FILTER_COMPARE_F32: &str = "filter_compare_f32";
    pub const FILTER_COMPARE_U8: &str = "filter_compare_u8";
    pub const FILTER_COMPARE_U32_SCAN_PHASE1: &str = "filter_compare_u32_scan_phase1";
    pub const FILTER_COMPARE_F64_SCAN_PHASE1: &str = "filter_compare_f64_scan_phase1";
    pub const FILTER_COMPARE_F32_SCAN_PHASE1: &str = "filter_compare_f32_scan_phase1";
    pub const FILTER_COMPARE_U32_COL: &str = "filter_compare_u32_col";
    pub const FILTER_COMPARE_I32_COL: &str = "filter_compare_i32_col";
    pub const FILTER_COMPARE_I64_COL: &str = "filter_compare_i64_col";
    pub const FILTER_COMPARE_U64_COL: &str = "filter_compare_u64_col";
    pub const FILTER_COMPARE_F32_COL: &str = "filter_compare_f32_col";
    pub const FILTER_COMPARE_F64_COL: &str = "filter_compare_f64_col";
    pub const FILTER_COMPARE_U8_COL: &str = "filter_compare_u8_col";
    pub const FILL_U32_IOTA: &str = "fill_u32_iota";
    pub const FILL_U32_CONST: &str = "fill_u32_const";
    pub const MARK_RANDOM_VARS: &str = "mark_random_vars";
    pub const RANDOM_VAR_TO_BIT_FROM_LIST: &str = "random_var_to_bit_from_list";
    pub const CHECK_RANDOM_VAR_COUNT: &str = "check_random_var_count";
    pub const COMPACT_U32_BY_MASK: &str = "compact_u32_by_mask";
    pub const COMPACT_I64_BY_MASK: &str = "compact_i64_by_mask";
    pub const COMPACT_F64_BY_MASK: &str = "compact_f64_by_mask";
    pub const COMPACT_BYTES_BY_MASK: &str = "compact_bytes_by_mask";
    pub const CAPTURE_COMPACT_COUNT: &str = "capture_compact_count";
    pub const MASK_CLAMP_ROWS: &str = "mask_clamp_rows";
    pub const MASK_AND: &str = "mask_and";
    pub const MASK_OR: &str = "mask_or";
    pub const MASK_NOT: &str = "mask_not";
}

/// Kernel function names in the set_ops module
pub mod set_ops_kernels {
    pub const CONCAT_U32: &str = "concat_u32";
    pub const CONCAT_BYTES: &str = "concat_bytes";
    pub const SORTED_DIFF_MARK: &str = "sorted_diff_mark";
}

/// Kernel function names in the pack module (GPU-side key packing)
pub mod pack_kernels {
    /// Pack multiple columns into row-major byte array
    pub const PACK_KEYS: &str = "pack_keys";
    /// Compute FNV-1a hash from packed keys
    pub const HASH_PACKED_KEYS: &str = "hash_packed_keys";
    /// Fused pack + hash in single pass (optimal for join key preparation)
    pub const PACK_AND_HASH_KEYS: &str = "pack_and_hash_keys";
    /// Fused pack + hash for arbitrary key column counts
    pub const PACK_AND_HASH_KEYS_GENERIC: &str = "pack_and_hash_keys_generic";
    /// Vectorized pack for 8-byte aligned columns
    pub const PACK_KEYS_ALIGNED: &str = "pack_keys_aligned";
    /// Unpack single column from packed row data
    pub const UNPACK_COLUMN: &str = "unpack_column";
    /// Unpack single column with device-resident row count
    pub const UNPACK_COLUMN_COUNTED: &str = "unpack_column_counted";
    /// Gather rows from packed data based on index array
    pub const GATHER_PACKED_ROWS: &str = "gather_packed_rows";
    /// Gather rows with device-resident row count
    pub const GATHER_PACKED_ROWS_COUNTED: &str = "gather_packed_rows_counted";
    /// Scatter write: distribute packed rows to non-contiguous output positions
    pub const SCATTER_PACKED_ROWS: &str = "scatter_packed_rows";
    /// Compare packed keys for equality
    pub const COMPARE_PACKED_KEYS: &str = "compare_packed_keys";
    /// Pack u8 bools into Arrow bitmap bytes
    pub const PACK_BOOLS_TO_BITMAP: &str = "pack_bools_to_bitmap";
}

/// Kernel function names in the circuit module
pub mod circuit_kernels {
    pub const XGCF_FORWARD_LEVEL: &str = "xgcf_forward_level";
    pub const XGCF_BACKWARD_LEVEL_PROPAGATE: &str = "xgcf_backward_level_propagate";
    pub const XGCF_BACKWARD_LEVEL_DECISION_GRAD: &str = "xgcf_backward_level_decision_grad";
    pub const XGCF_BACKWARD_LEVEL_LIT_GRAD: &str = "xgcf_backward_level_lit_grad";
    pub const XGCF_FREE_VAR_APPLY_GRAD: &str = "xgcf_free_var_apply_grad";
    pub const XGCF_FREE_VAR_REDUCE_STAGE: &str = "xgcf_free_var_reduce_stage";
    pub const XGCF_ADD_SCALAR: &str = "xgcf_add_scalar";
    pub const XGCF_FORWARD_LEVEL_CACHED: &str = "xgcf_forward_level_cached";
    pub const XGCF_EVAL_ALL_LEVELS_CACHED: &str = "xgcf_eval_all_levels_cached";
    pub const XGCF_EVAL_ALL_LEVELS_CACHED_BATCHED: &str = "xgcf_eval_all_levels_cached_batched";
    pub const XGCF_BACKWARD_LEVEL_PROPAGATE_CACHED: &str = "xgcf_backward_level_propagate_cached";
    pub const XGCF_BACKWARD_LEVEL_DECISION_GRAD_CACHED: &str =
        "xgcf_backward_level_decision_grad_cached";
    pub const XGCF_BACKWARD_LEVEL_LIT_GRAD_CACHED: &str = "xgcf_backward_level_lit_grad_cached";
    pub const XGCF_BACKWARD_ALL_LEVELS_CACHED: &str = "xgcf_backward_all_levels_cached";
    pub const XGCF_BACKWARD_ALL_LEVELS_CACHED_BATCHED: &str =
        "xgcf_backward_all_levels_cached_batched";
    pub const XGCF_FREE_VAR_APPLY_GRAD_CACHED: &str = "xgcf_free_var_apply_grad_cached";
    pub const XGCF_FREE_VAR_REDUCE_STAGE_CACHED: &str = "xgcf_free_var_reduce_stage_cached";
    pub const XGCF_ADD_SCALAR_CACHED: &str = "xgcf_add_scalar_cached";
    pub const XGCF_SET_ROOT_ADJ_CACHED_BATCHED: &str = "xgcf_set_root_adj_cached_batched";
    pub const XGCF_COPY_ROOT_CACHED: &str = "xgcf_copy_root_cached";
    pub const XGCF_COPY_ROOT_CACHED_META: &str = "xgcf_copy_root_cached_meta";
    pub const XGCF_COPY_ROOT_CACHED_META_BATCHED: &str = "xgcf_copy_root_cached_meta_batched";
}

/// Kernel function names in the cache module
pub mod cache_kernels {
    pub const CACHE_CNF_HASH: &str = "cache_cnf_hash";
    pub const CACHE_LOOKUP_OR_INSERT: &str = "cache_lookup_or_insert";
    pub const CACHE_EVICT_LRU: &str = "cache_evict_lru";
    pub const CACHE_STORE_U8: &str = "cache_store_u8";
    pub const CACHE_STORE_U32: &str = "cache_store_u32";
    pub const CACHE_STORE_I32: &str = "cache_store_i32";
    pub const CACHE_STORE_F64: &str = "cache_store_f64";
    pub const CACHE_STORE_META: &str = "cache_store_meta";
}

/// Kernel function names in the SAT module
pub mod sat_kernels {
    pub const SAT_CDCL_SOLVE: &str = "sat_cdcl_solve";
    pub const SAT_CHECK_MODEL: &str = "sat_check_model";
    pub const SAT_PROOF_MARK_NEEDED: &str = "sat_proof_mark_needed";
    pub const SAT_PROOF_CHECK: &str = "sat_proof_check";
    pub const SAT_ASSERT_STATUS: &str = "sat_assert_status";
    pub const SAT_ASSERT_OK: &str = "sat_assert_ok";
    pub const SAT_XGCF_CNF_COUNTS: &str = "sat_xgcf_cnf_counts";
    pub const SAT_XGCF_CNF_EMIT: &str = "sat_xgcf_cnf_emit";
    pub const SAT_XGCF_CNF_CAPTURE_LAST_COUNTS: &str = "sat_xgcf_cnf_capture_last_counts";
    pub const SAT_XGCF_CNF_COMPUTE_TOTALS: &str = "sat_xgcf_cnf_compute_totals";
    pub const SAT_CNF_WRITE_TERMINATOR: &str = "sat_cnf_write_terminator";
    pub const SAT_CNF_COPY_INTO: &str = "sat_cnf_copy_into";
    pub const SAT_SHIFT_OFFSETS: &str = "sat_shift_offsets";
    pub const SAT_XGCF_WRITE_ROOT_UNIT_CLAUSE: &str = "sat_xgcf_write_root_unit_clause";
    pub const SAT_NOT_PHI_COUNTS: &str = "sat_not_phi_counts";
    pub const SAT_EMIT_NOT_PHI: &str = "sat_emit_not_phi";
}

/// Default maximum output size for join operations.
/// This prevents memory overflow when joining large tables with high cardinality matches.
pub const DEFAULT_JOIN_MAX_OUTPUT: usize = 1_000_000;

/// Comparison operators for filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompareOp {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}

/// Join types for hash_join_v2
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join: return rows where keys match on both sides
    Inner,
    /// Semi join: return left rows that have any match in right (no right columns)
    Semi,
    /// Anti join: return left rows that have NO match in right
    Anti,
    /// Left outer join: return all left rows, with nulls for non-matching right
    LeftOuter,
}

/// Result of packing key columns and computing hashes for join operations
struct PackedKeyData {
    /// Computed hash values (one per row)
    hashes: crate::memory::TrackedCudaSlice<u64>,
    /// Packed key data in row-major format
    packed_keys: crate::memory::TrackedCudaSlice<u8>,
    /// Total bytes per row (key stride)
    key_bytes: u32,
}

struct JoinHashTableV2 {
    bucket_counts: crate::memory::TrackedCudaSlice<u32>,
    bucket_offsets: crate::memory::TrackedCudaSlice<u32>,
    bucket_entries: crate::memory::TrackedCudaSlice<u32>,
    bucket_entry_hashes: crate::memory::TrackedCudaSlice<u64>,
    bucket_mask: u32,
}

/// Bucketed hash table for u64 hashes.
pub struct HashTableU64 {
    pub bucket_counts: crate::memory::TrackedCudaSlice<u32>,
    pub bucket_offsets: crate::memory::TrackedCudaSlice<u32>,
    pub bucket_entries: crate::memory::TrackedCudaSlice<u32>,
    pub bucket_entry_hashes: crate::memory::TrackedCudaSlice<u64>,
    pub bucket_mask: u32,
}

/// Cached build-side join index for v2 hash join.
///
/// This captures the packed key bytes and bucketed hash table layout for the build (right) side,
/// enabling reuse across repeated joins on the same relation + key columns.
pub struct JoinIndexV2 {
    right_num_rows: u32,
    right_keys: Vec<usize>,
    key_bytes: u32,
    packed_keys: crate::memory::TrackedCudaSlice<u8>,
    table: JoinHashTableV2,
}

impl JoinIndexV2 {
    /// Key columns (indices) this index was built for.
    pub fn right_keys(&self) -> &[usize] {
        &self.right_keys
    }

    /// Row count of the build-side buffer at index build time.
    pub fn right_num_rows(&self) -> u32 {
        self.right_num_rows
    }

    /// Approximate device memory used by this cached index.
    pub fn estimated_bytes(&self) -> u64 {
        let mut bytes = 0u64;
        bytes = bytes.saturating_add(self.packed_keys.len() as u64);
        bytes = bytes.saturating_add(self.table.bucket_counts.len() as u64 * 4);
        bytes = bytes.saturating_add(self.table.bucket_offsets.len() as u64 * 4);
        bytes = bytes.saturating_add(self.table.bucket_entries.len() as u64 * 4);
        bytes = bytes.saturating_add(self.table.bucket_entry_hashes.len() as u64 * 8);
        bytes
    }
}

/// CUDA kernel provider for xlog GPU operations
///
/// Manages pre-compiled PTX modules for relational operations:
/// - **Join**: Hash join with build/probe phases
/// - **Dedup**: Sort-based deduplication with prefix-sum compaction
/// - **GroupBy**: Sorted-input group aggregation (count, sum, min, max)
///
/// PTX modules are loaded at construction time and stored in the CUDA device.
/// Kernel functions can be retrieved using `device.get_func()`.
///
/// # Example
/// ```ignore
/// use std::sync::Arc;
/// use xlog_cuda::{CudaDevice, GpuMemoryManager, CudaKernelProvider};
/// use xlog_core::MemoryBudget;
///
/// let device = Arc::new(CudaDevice::new(0)?);
/// let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::default()));
/// let provider = CudaKernelProvider::new(device, memory)?;
/// ```
pub struct CudaKernelProvider {
    /// The CUDA device with loaded PTX modules
    device: Arc<CudaDevice>,
    /// GPU memory manager for kernel allocations
    memory: Arc<GpuMemoryManager>,
    /// Tracked host transfers for diagnostics
    transfer_tracker: HostTransferTracker,
    /// PTX load profiling data (populated only when XLOG_WARMUP_PROFILE=1)
    ptx_load_profile: Option<PtxLoadProfile>,
    /// Column-level D2H transfer counter (incremented by each download_column_* call)
    d2h_transfer_count: AtomicU64,
    /// Strict deterministic-Datalog D2H gate. When `true`, any data-plane D2H
    /// transfer (column downloads or `dtoh_sync_copy_into_tracked`) increments
    /// the violation counter and returns `XlogError::Execution` from the
    /// originating call. Metadata reads via `dtoh_scalar_untracked` are NOT
    /// gated. See [`CudaKernelProvider::enable_strict_deterministic_d2h`].
    strict_deterministic_d2h: AtomicBool,
    /// Cumulative count of deterministic-D2H gate violations observed since
    /// the last reset. Increments even on the failing path (the originating
    /// call still returns `Err`); kept for telemetry and tests.
    deterministic_d2h_violations: AtomicU64,
}

#[derive(Default)]
struct HostTransferTracker {
    dtoh_bytes: AtomicU64,
    htod_bytes: AtomicU64,
    dtoh_calls: AtomicU64,
    htod_calls: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
pub struct HostTransferStats {
    pub dtoh_bytes: u64,
    pub htod_bytes: u64,
    pub dtoh_calls: u64,
    pub htod_calls: u64,
}

impl HostTransferTracker {
    fn record_dtoh(&self, bytes: u64) {
        self.dtoh_calls.fetch_add(1, Ordering::Relaxed);
        self.dtoh_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_htod(&self, bytes: u64) {
        self.htod_calls.fetch_add(1, Ordering::Relaxed);
        self.htod_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn snapshot(&self) -> HostTransferStats {
        HostTransferStats {
            dtoh_bytes: self.dtoh_bytes.load(Ordering::Relaxed),
            htod_bytes: self.htod_bytes.load(Ordering::Relaxed),
            dtoh_calls: self.dtoh_calls.load(Ordering::Relaxed),
            htod_calls: self.htod_calls.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.dtoh_bytes.store(0, Ordering::Relaxed);
        self.htod_bytes.store(0, Ordering::Relaxed);
        self.dtoh_calls.store(0, Ordering::Relaxed);
        self.htod_calls.store(0, Ordering::Relaxed);
    }
}

impl CudaKernelProvider {
    /// Create a new CUDA kernel provider
    ///
    /// Loads all kernel modules into the CUDA device.
    /// Prefers cubin for the detected SM arch, falls back to portable PTX (sm_75+).
    ///
    /// # Arguments
    /// * `device` - The CUDA device to load modules into
    /// * `memory` - The GPU memory manager for kernel allocations
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if PTX loading fails
    ///
    /// # Example
    /// ```ignore
    /// let device = Arc::new(CudaDevice::new(0)?);
    /// let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::default()));
    /// let provider = CudaKernelProvider::new(device, memory)?;
    /// ```
    pub fn new(device: Arc<CudaDevice>, memory: Arc<GpuMemoryManager>) -> Result<Self> {
        let profiling = warmup_profiling_enabled();
        let ptx_load_profile = Self::load_all_kernel_modules(&device, profiling)?;

        Ok(Self {
            device,
            memory,
            transfer_tracker: HostTransferTracker::default(),
            ptx_load_profile,
            d2h_transfer_count: AtomicU64::new(0),
            strict_deterministic_d2h: AtomicBool::new(false),
            deterministic_d2h_violations: AtomicU64::new(0),
        })
    }

    /// Construct a provider whose `GpuMemoryManager` must already
    /// have a v0.6 [`crate::device_runtime::XlogDeviceRuntime`]
    /// attached via [`GpuMemoryManager::with_runtime`].
    ///
    /// Equivalent to [`Self::new`] in every respect — same kernel
    /// loading, same field initialization — but **rejects** managers
    /// that lack a runtime. This guards against the misconfiguration
    /// in which a caller asks for runtime-routed provider semantics
    /// (by calling `with_runtime`) but supplies a legacy manager
    /// built via [`GpuMemoryManager::new`]; without the check, the
    /// resulting provider would silently keep using the cudarc
    /// default allocator and the runtime budget/logging stack would
    /// never observe the allocations the caller expected to be
    /// routed through it.
    ///
    /// Note: a runtime-routed manager passed to [`Self::new`] still
    /// routes correctly — `alloc::<T>` and `alloc_raw` consult
    /// `memory.runtime()` regardless of which provider constructor
    /// was used. `with_runtime` exists for callers that want the
    /// requirement enforced at construction time, not for
    /// correctness of the routing itself.
    ///
    /// This is the **opt-in** runtime entry point for providers.
    /// `Self::new` continues to accept managers without a runtime
    /// (the legacy default) and remains the production constructor
    /// until the runtime stack is certified end-to-end.
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if `memory.runtime()` is `None`,
    /// or anything `Self::new` would return.
    ///
    /// # Example
    /// ```ignore
    /// let device = Arc::new(CudaDevice::new(0)?);
    /// let runtime = Arc::new(XlogDeviceRuntime::with_resource(
    ///     Arc::clone(&device),
    ///     0,
    ///     Arc::new(StreamPool::with_defaults(Arc::clone(&device))),
    ///     Box::new(AsyncCudaResource::new(/* ... */)),
    /// ));
    /// let memory = Arc::new(GpuMemoryManager::with_runtime(
    ///     Arc::clone(&device),
    ///     MemoryBudget::default(),
    ///     runtime,
    /// ));
    /// let provider = CudaKernelProvider::with_runtime(device, memory)?;
    /// ```
    pub fn with_runtime(device: Arc<CudaDevice>, memory: Arc<GpuMemoryManager>) -> Result<Self> {
        if memory.runtime().is_none() {
            return Err(XlogError::Kernel(
                "CudaKernelProvider::with_runtime requires a GpuMemoryManager built via \
                 GpuMemoryManager::with_runtime; got a manager with no runtime attached"
                    .to_string(),
            ));
        }
        Self::new(device, memory)
    }

    /// Get the CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get the GPU memory manager
    pub fn memory(&self) -> &Arc<GpuMemoryManager> {
        &self.memory
    }

    /// Get PTX load profiling data (only populated when XLOG_WARMUP_PROFILE=1).
    pub fn ptx_load_profile(&self) -> Option<&PtxLoadProfile> {
        self.ptx_load_profile.as_ref()
    }

    /// Reset tracked host transfer statistics.
    pub fn reset_host_transfer_stats(&self) {
        self.transfer_tracker.reset();
    }

    /// Snapshot tracked host transfer statistics.
    pub fn host_transfer_stats(&self) -> HostTransferStats {
        self.transfer_tracker.snapshot()
    }

    /// Read the column-level D2H transfer counter.
    ///
    /// This counter increments once per `download_column_*` call, enabling
    /// callers (e.g. the ILP trainer) to assert that no column downloads
    /// occurred during a performance-critical section.
    pub fn d2h_transfer_count(&self) -> u64 {
        self.d2h_transfer_count.load(Ordering::Relaxed)
    }

    /// Reset the column-level D2H transfer counter to zero.
    pub fn reset_d2h_transfer_count(&self) {
        self.d2h_transfer_count.store(0, Ordering::Relaxed);
    }

    /// Enable the strict deterministic-Datalog D2H gate.
    ///
    /// While enabled, any data-plane device-to-host transfer (column downloads
    /// via `download_column` / `download_column_untracked`, and any internal
    /// transfer routed through `dtoh_sync_copy_into_tracked`) increments
    /// [`CudaKernelProvider::deterministic_d2h_violation_count`] and returns
    /// `XlogError::Execution` from the originating call.
    ///
    /// Metadata reads via [`CudaKernelProvider::dtoh_scalar_untracked`] are
    /// allowed and never trip the gate.
    ///
    /// Default is `false`; the runtime opts in via
    /// `RuntimeConfig::strict_deterministic_d2h`. v0.5.5 ships the gate
    /// opt-in only — known-violating relational paths (set difference,
    /// join count/materialize) are scheduled for replacement before the
    /// default flips.
    pub fn enable_strict_deterministic_d2h(&self) {
        self.strict_deterministic_d2h.store(true, Ordering::Relaxed);
    }

    /// Disable the strict deterministic-Datalog D2H gate.
    pub fn disable_strict_deterministic_d2h(&self) {
        self.strict_deterministic_d2h
            .store(false, Ordering::Relaxed);
    }

    /// Returns whether the strict deterministic-Datalog D2H gate is enabled.
    pub fn strict_deterministic_d2h_enabled(&self) -> bool {
        self.strict_deterministic_d2h.load(Ordering::Relaxed)
    }

    /// Cumulative deterministic-D2H gate violations since the last reset.
    pub fn deterministic_d2h_violation_count(&self) -> u64 {
        self.deterministic_d2h_violations.load(Ordering::Relaxed)
    }

    /// Reset the deterministic-D2H violation counter to zero.
    pub fn reset_deterministic_d2h_violations(&self) {
        self.deterministic_d2h_violations
            .store(0, Ordering::Relaxed);
    }

    /// Chokepoint for the deterministic-D2H gate.
    ///
    /// If the gate is enabled, increments the violation counter and returns
    /// `XlogError::Execution` naming the offending operation and byte count.
    /// If the gate is disabled, returns `Ok(())` cheaply.
    pub(crate) fn check_deterministic_d2h(&self, op: &'static str, bytes: u64) -> Result<()> {
        if self.strict_deterministic_d2h.load(Ordering::Relaxed) {
            self.deterministic_d2h_violations
                .fetch_add(1, Ordering::Relaxed);
            return Err(XlogError::Execution(format!(
                "deterministic D2H gate: {} attempted to copy {} bytes from device to host",
                op, bytes
            )));
        }
        Ok(())
    }

    fn dtoh_sync_copy_into_tracked<T: DeviceRepr, Src: DevicePtr<T>>(
        &self,
        src: &Src,
        dst: &mut [T],
    ) -> Result<()> {
        let bytes = std::mem::size_of::<T>()
            .checked_mul(dst.len())
            .ok_or_else(|| XlogError::Kernel("dtoh size overflow".to_string()))?;
        self.check_deterministic_d2h("dtoh_sync_copy_into_tracked", bytes as u64)?;
        self.transfer_tracker.record_dtoh(bytes as u64);
        self.device
            .inner()
            .dtoh_sync_copy_into(src, dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy from device: {}", e)))
    }

    /// Read a single scalar from device to host WITHOUT updating the
    /// D2H transfer tracker. Use ONLY for metadata reads (e.g. total_nnz
    /// after an exclusive scan), never for data-plane transfers.
    ///
    /// This makes the "metadata != data-plane" contract explicit and
    /// auditable: callers that bypass tracking must call this method
    /// (which is grep-able) rather than reaching for device().inner().
    pub fn dtoh_scalar_untracked<T: DeviceRepr + Default + Copy>(
        &self,
        src: &crate::memory::TrackedCudaSlice<T>,
        index: usize,
    ) -> Result<T> {
        if index >= src.len() {
            return Err(XlogError::Kernel(format!(
                "dtoh_scalar_untracked: index={} >= len={}",
                index,
                src.len()
            )));
        }
        let slice = src.try_slice(index..index + 1).ok_or_else(|| {
            XlogError::Kernel(format!(
                "dtoh_scalar_untracked: slice failed at index={}",
                index
            ))
        })?;
        let mut buf = [T::default()];
        self.device
            .inner()
            .dtoh_sync_copy_into(&slice, &mut buf)
            .map_err(|e| XlogError::Kernel(format!("dtoh_scalar_untracked: copy failed: {}", e)))?;
        Ok(buf[0])
    }

    fn htod_sync_copy_into_tracked<T: DeviceRepr, Dst: cudarc::driver::DevicePtrMut<T>>(
        &self,
        src: &[T],
        dst: &mut Dst,
    ) -> Result<()> {
        let bytes = std::mem::size_of::<T>()
            .checked_mul(src.len())
            .ok_or_else(|| XlogError::Kernel("htod size overflow".to_string()))?;
        self.transfer_tracker.record_htod(bytes as u64);
        self.device
            .inner()
            .htod_sync_copy_into(src, dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy to device: {}", e)))
    }

    /// Compute exclusive prefix sum of u8 mask, returns (prefix_sum_vec, total_count)
    ///
    /// This is useful for compaction operations where we need to know:
    /// 1. The output position for each input element (prefix sum)
    /// 2. The total number of elements that pass the mask (count)
    ///
    /// # Arguments
    /// * `mask` - A slice of u8 values (0 or non-zero)
    ///
    /// # Returns
    /// A tuple of:
    /// - `Vec<u32>` containing the exclusive prefix sum
    /// - `u32` containing the total count of non-zero mask elements
    ///
    /// # Example
    /// ```ignore
    /// let mask = vec![1u8, 0, 1, 1, 0, 1];
    /// let (prefix_sum, count) = provider.prefix_sum_mask(&mask)?;
    /// // prefix_sum = [0, 1, 1, 2, 3, 3]
    /// // count = 4
    /// ```
    ///
    /// # Note
    /// For small inputs (<=256 elements), a CPU scan is used for efficiency.
    /// For larger inputs, a three-phase multi-block GPU scan is used.
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails
    pub fn exclusive_scan_u32_inplace(
        &self,
        data: &mut crate::memory::TrackedCudaSlice<u32>,
        n: u32,
    ) -> Result<()> {
        if n as usize > data.len() {
            return Err(XlogError::Kernel(format!(
                "exclusive_scan_u32_inplace: n={} exceeds slice len={}",
                n,
                data.len()
            )));
        }
        self.multiblock_scan_u32_inplace(data, n)
    }

    fn multiblock_scan_u32_inplace(
        &self,
        data: &mut crate::memory::TrackedCudaSlice<u32>,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }

        let device = self.device.inner();
        let block_size = 256u32;

        if n <= block_size {
            let phase2_fn = device
                .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE2)
                .ok_or_else(|| {
                    XlogError::Kernel("Failed to get multiblock_scan_phase2 kernel".to_string())
                })?;

            // SAFETY: multiblock_scan_phase2(uint32_t* block_sums, uint32_t num_blocks)
            unsafe {
                phase2_fn.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut *data, n),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase2 failed: {}", e)))?;

            self.device.synchronize()?;
            return Ok(());
        }

        let num_blocks = (n + block_size - 1) / block_size;
        let mut block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

        let phase1_u32_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_U32_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_u32_phase1 kernel".to_string())
            })?;

        // SAFETY: multiblock_scan_u32_phase1(uint32_t* data, uint32_t* block_sums, uint32_t n)
        unsafe {
            phase1_u32_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &mut block_sums, n),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_u32_phase1 failed: {}", e)))?;
        self.device.synchronize()?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace(&mut block_sums, num_blocks)?;
        }

        let phase3_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE3)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase3 kernel".to_string())
            })?;

        // SAFETY: multiblock_scan_phase3(uint32_t* prefix_sum, const uint32_t* block_offsets, uint32_t n)
        unsafe {
            phase3_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &block_sums, n),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;

        self.device.synchronize()?;
        Ok(())
    }

    fn multiblock_scan_u32_view_inplace(
        &self,
        data: &mut CudaViewMut<'_, u32>,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }

        let device = self.device.inner();
        let block_size = 256u32;

        if n <= block_size {
            let phase2_fn = device
                .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE2)
                .ok_or_else(|| {
                    XlogError::Kernel("Failed to get multiblock_scan_phase2 kernel".to_string())
                })?;

            // SAFETY: multiblock_scan_phase2(uint32_t* block_sums, uint32_t num_blocks)
            unsafe {
                phase2_fn.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (data, n),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase2 failed: {}", e)))?;

            self.device.synchronize()?;
            return Ok(());
        }

        let num_blocks = (n + block_size - 1) / block_size;
        let mut block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

        let phase1_u32_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_U32_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_u32_phase1 kernel".to_string())
            })?;

        // SAFETY: multiblock_scan_u32_phase1(uint32_t* data, uint32_t* block_sums, uint32_t n)
        unsafe {
            phase1_u32_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &mut block_sums, n),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_u32_phase1 failed: {}", e)))?;
        self.device.synchronize()?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace(&mut block_sums, num_blocks)?;
        }

        let phase3_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE3)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase3 kernel".to_string())
            })?;

        // SAFETY: multiblock_scan_phase3(uint32_t* prefix_sum, const uint32_t* block_offsets, uint32_t n)
        unsafe {
            phase3_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &block_sums, n),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;

        self.device.synchronize()?;
        Ok(())
    }

    /// (GPU-asserted). If they diverge, the GPU asserts and the export fails.
    /// - Download fails

    // ============== Internal Helper Methods ==============

    /// Read a buffer's logical row count, using the host cache when available
    /// and falling back to a metadata-only device-to-host read when needed.
    pub fn device_row_count(&self, buffer: &CudaBuffer) -> Result<usize> {
        if let Some(n) = buffer.cached_row_count() {
            return Ok(n as usize);
        }
        let mut host_rows = [0u32];
        self.device
            .inner()
            .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
            .map_err(|e| XlogError::Kernel(format!("Failed to read row count: {}", e)))?;
        buffer.set_cached_row_count_if_unset(host_rows[0]);
        Ok(host_rows[0] as usize)
    }

    /// Read and validate a buffer's logical row count for outward-facing APIs.
    ///
    /// This keeps exported/query-visible lengths tied to the device logical row
    /// count while still rejecting impossible metadata (`logical_rows > row_cap`).
    pub fn validated_logical_row_count(&self, buffer: &CudaBuffer) -> Result<usize> {
        let logical_rows = self.device_row_count(buffer)?;
        validate_logical_row_count(buffer.num_rows(), logical_rows)
    }

    fn clone_device_row_count(&self, buffer: &CudaBuffer) -> Result<TrackedCudaSlice<u32>> {
        let mut d_num_rows = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .dtod_copy(buffer.num_rows_device(), &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy row count: {}", e)))?;
        Ok(d_num_rows)
    }

    fn upload_device_row_count(&self, row_count: u32) -> Result<TrackedCudaSlice<u32>> {
        let mut d_num_rows = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .htod_sync_copy_into(&[row_count], &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload row count: {}", e)))?;
        Ok(d_num_rows)
    }

    fn buffer_from_columns_with_device_count(
        &self,
        columns: Vec<CudaColumn>,
        row_cap: u64,
        schema: Schema,
        src: &CudaBuffer,
    ) -> Result<CudaBuffer> {
        let d_num_rows = self.clone_device_row_count(src)?;
        Ok(CudaBuffer::from_columns(
            columns, row_cap, d_num_rows, schema,
        ))
    }

    fn column_bytes_view<'a>(
        &self,
        col: &'a CudaColumn,
        num_bytes: usize,
    ) -> Result<RawCudaView<'a, u8>> {
        if col.num_bytes() < num_bytes {
            return Err(XlogError::Kernel(format!(
                "Column has {} bytes but {} required",
                col.num_bytes(),
                num_bytes
            )));
        }
        let ptr = *col.device_ptr();
        Ok(RawCudaView {
            ptr,
            len: num_bytes,
            stream: col.stream().clone(),
            _marker: PhantomData,
        })
    }

    fn bytes_as_u32_view<'a>(
        &self,
        bytes: &'a TrackedCudaSlice<u8>,
        num_elements: usize,
    ) -> Result<RawCudaView<'a, u32>> {
        let required_bytes = num_elements * std::mem::size_of::<u32>();
        if bytes.len() < required_bytes {
            return Err(XlogError::Kernel(format!(
                "Packed keys have {} bytes but {} required for {} u32 elements",
                bytes.len(),
                required_bytes,
                num_elements
            )));
        }
        let ptr = *bytes.device_ptr();
        if (ptr as usize) % std::mem::align_of::<u32>() != 0 {
            return Err(XlogError::Kernel(
                "Packed keys device pointer is not u32-aligned".to_string(),
            ));
        }
        Ok(RawCudaView {
            ptr,
            len: num_elements,
            stream: bytes.stream().clone(),
            _marker: PhantomData,
        })
    }

    /// Reinterpret a `CudaBuffer` column as a `u32` slice for kernel access.
    fn column_as_u32_view<'a>(
        &self,
        col: &'a CudaColumn,
        num_elements: usize,
    ) -> Result<RawCudaView<'a, u32>> {
        let required_bytes = num_elements * std::mem::size_of::<u32>();
        if col.num_bytes() < required_bytes {
            return Err(XlogError::Kernel(format!(
                "Column has {} bytes but {} required for {} u32 elements",
                col.num_bytes(),
                required_bytes,
                num_elements
            )));
        }
        let ptr = *col.device_ptr();
        if (ptr as usize) % std::mem::align_of::<u32>() != 0 {
            return Err(XlogError::Kernel(
                "Column device pointer is not u32-aligned".to_string(),
            ));
        }
        Ok(RawCudaView {
            ptr,
            len: num_elements,
            stream: col.stream().clone(),
            _marker: PhantomData,
        })
    }

    fn column_as_u64_view<'a>(
        &self,
        col: &'a CudaColumn,
        num_elements: usize,
    ) -> Result<RawCudaView<'a, u64>> {
        let required_bytes = num_elements * std::mem::size_of::<u64>();
        if col.num_bytes() < required_bytes {
            return Err(XlogError::Kernel(format!(
                "Column has {} bytes but {} required for {} u64 elements",
                col.num_bytes(),
                required_bytes,
                num_elements
            )));
        }
        let ptr = *col.device_ptr();
        if (ptr as usize) % std::mem::align_of::<u64>() != 0 {
            return Err(XlogError::Kernel(
                "Column device pointer is not u64-aligned".to_string(),
            ));
        }
        Ok(RawCudaView {
            ptr,
            len: num_elements,
            stream: col.stream().clone(),
            _marker: PhantomData,
        })
    }

    /// Reinterpret a `CudaBuffer` column as an `f64` slice for kernel access.
    fn column_as_f64_view<'a>(
        &self,
        col: &'a CudaColumn,
        num_elements: usize,
    ) -> Result<RawCudaView<'a, f64>> {
        let required_bytes = num_elements * std::mem::size_of::<f64>();
        if col.num_bytes() < required_bytes {
            return Err(XlogError::Kernel(format!(
                "Column has {} bytes but {} required for {} f64 elements",
                col.num_bytes(),
                required_bytes,
                num_elements
            )));
        }
        let ptr = *col.device_ptr();
        if (ptr as usize) % std::mem::align_of::<f64>() != 0 {
            return Err(XlogError::Kernel(
                "Column device pointer is not f64-aligned".to_string(),
            ));
        }
        Ok(RawCudaView {
            ptr,
            len: num_elements,
            stream: col.stream().clone(),
            _marker: PhantomData,
        })
    }

    /// Create an empty buffer with the given schema (all columns are empty slices)
    ///
    /// # Arguments
    /// * `schema` - The schema for the empty buffer
    ///
    /// # Returns
    /// A new CudaBuffer with zero rows
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if allocation fails
    pub fn create_empty_buffer(&self, schema: Schema) -> Result<CudaBuffer> {
        let mut columns = Vec::with_capacity(schema.arity());
        for _ in 0..schema.arity() {
            // Allocate zero-length column
            columns.push(self.memory.alloc::<u8>(0)?.into());
        }
        self.buffer_from_columns(columns, 0, schema)
    }

    pub(crate) fn buffer_from_columns(
        &self,
        columns: Vec<CudaColumn>,
        row_cap: u64,
        schema: Schema,
    ) -> Result<CudaBuffer> {
        let row_u32 = u32::try_from(row_cap)
            .map_err(|_| XlogError::Kernel(format!("Row capacity {} exceeds u32::MAX", row_cap)))?;
        let mut d_num_rows = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .htod_sync_copy_into(&[row_u32], &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("Failed to set row count: {}", e)))?;
        Ok(CudaBuffer::from_columns_with_host_count(
            columns, row_cap, d_num_rows, schema, row_u32,
        ))
    }

    /// Combine schemas from left and right buffers for join result
    fn combine_schemas(&self, left: &Schema, right: &Schema) -> Schema {
        let mut columns = left.columns.clone();
        columns.extend(right.columns.iter().cloned());
        Schema::new(columns)
    }

    /// Check if two schemas have compatible types (same arity and column types)
    ///
    /// This ignores column names, which is useful for Datalog operations where
    /// projected relations may have different column names but the same types.
    fn schemas_type_compatible(&self, a: &Schema, b: &Schema) -> bool {
        if a.arity() != b.arity() {
            return false;
        }
        for i in 0..a.arity() {
            if a.column_type(i) != b.column_type(i) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::{AggOp, MemoryBudget, ScalarType};

    fn has_cuda_device() -> bool {
        CudaDevice::new(0).is_ok()
    }

    #[test]
    fn test_kernel_artifact_locator_precedence_order() {
        use super::kernel_paths::KernelArtifactLocator;
        use std::fs;
        use std::path::PathBuf;

        let root = std::env::temp_dir().join(format!(
            "xlog-kernel-paths-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before UNIX_EPOCH")
                .as_nanos()
        ));
        let cubin_dir = root.join("cubin");
        let package_dir = root.join("bin").join("kernels");
        let out_dir = root.join("out");
        fs::create_dir_all(&cubin_dir).expect("create cubin dir");
        fs::create_dir_all(&package_dir).expect("create package kernels dir");
        fs::create_dir_all(&out_dir).expect("create out dir");

        let name = "xlog_join";
        let cc = 75;
        let cubin_path = cubin_dir.join(format!("{name}.sm_{cc}.cubin"));
        let package_path = package_dir.join(format!("{name}.sm_{cc}.cubin"));
        let out_path = out_dir.join(format!("{name}.sm_{cc}.cubin"));
        fs::write(&cubin_path, b"cubin").expect("write cubin file");
        fs::write(&package_path, b"package").expect("write package file");
        fs::write(&out_path, b"out").expect("write out file");

        let locator = KernelArtifactLocator::new(
            Some(cubin_dir.clone()),
            Some(package_dir.clone()),
            Some(out_dir.clone()),
        );

        let (path, is_cubin) = locator
            .resolve_module_path(name, cc)
            .expect("expected a kernel artifact");
        assert_eq!(path, cubin_path);
        assert!(is_cubin);

        fs::remove_file(&cubin_path).expect("remove cubin file");
        let (path, is_cubin) = locator
            .resolve_module_path(name, cc)
            .expect("expected package kernel artifact");
        assert_eq!(path, package_path);
        assert!(is_cubin);

        fs::remove_file(&package_path).expect("remove package file");
        let (path, is_cubin) = locator
            .resolve_module_path(name, cc)
            .expect("expected out dir kernel artifact");
        assert_eq!(path, out_path);
        assert!(is_cubin);

        let _ = fs::remove_dir_all(PathBuf::from(&root));
    }

    #[test]
    fn test_module_resolution_finds_portable_ptx() {
        // Verify resolve_module_path finds portable PTX for all modules.
        // Uses a dummy cc (999) so cubin won't match — only portable PTX.
        for name in crate::kernel_manifest_data::KERNEL_CU_NAMES {
            let result = resolve_module_path(name, 999);
            assert!(
                result.is_some(),
                "resolve_module_path({name}, 999) should find portable PTX"
            );
            let (path, is_cubin) = result.unwrap();
            assert!(
                !is_cubin,
                "{name}: expected portable PTX fallback, got cubin"
            );
            assert!(
                path.to_str().unwrap().ends_with(".portable.ptx"),
                "{name}: path should end with .portable.ptx, got {:?}",
                path
            );
        }
    }

    #[test]
    fn test_module_resolution_falls_back_to_embedded_portable_ptx() {
        use super::kernel_paths::KernelArtifactLocator;

        let locator = KernelArtifactLocator::new(None, None, None);
        for name in crate::kernel_manifest_data::KERNEL_CU_NAMES {
            let source = resolve_module_source_with_locator(name, 999, &locator)
                .unwrap_or_else(|| panic!("{name}: expected embedded portable PTX fallback"));

            match source {
                KernelModuleSource::EmbeddedPortablePtx { ptx } => {
                    assert!(
                        ptx.contains(".entry"),
                        "{name}: embedded PTX should contain CUDA entry points"
                    );
                }
                KernelModuleSource::File { path, .. } => {
                    panic!(
                        "{name}: expected embedded portable PTX fallback, got file {}",
                        path.display()
                    );
                }
            }
        }
    }

    #[test]
    fn test_embedded_portable_ptx_manifest_matches_kernel_manifest() {
        let embedded_names: std::collections::BTreeSet<_> =
            crate::embedded_kernel_data::EMBEDDED_PORTABLE_PTX
                .iter()
                .map(|artifact| artifact.name)
                .collect();
        let manifest_names: std::collections::BTreeSet<_> =
            crate::kernel_manifest_data::KERNEL_CU_NAMES
                .iter()
                .copied()
                .collect();

        assert_eq!(
            embedded_names, manifest_names,
            "embedded portable PTX table should cover every runtime kernel module"
        );
    }

    #[test]
    fn test_kernel_provider_creation() {
        if !has_cuda_device() {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = Arc::new(CudaDevice::new(0).expect("Failed to create device"));
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GB
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));

        let provider = CudaKernelProvider::new(device.clone(), memory.clone());
        assert!(
            provider.is_ok(),
            "Failed to create kernel provider: {:?}",
            provider.err()
        );

        let provider = provider.unwrap();
        assert!(Arc::ptr_eq(provider.device(), &device));
        assert!(Arc::ptr_eq(provider.memory(), &memory));
    }

    #[test]
    fn test_kernel_functions_accessible() {
        if !has_cuda_device() {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = Arc::new(CudaDevice::new(0).expect("Failed to create device"));
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));

        let _provider =
            CudaKernelProvider::new(device.clone(), memory).expect("Failed to create provider");

        // Verify all kernel functions can be retrieved
        let inner = device.inner();

        // Join kernels
        let build_fn = inner.get_func(JOIN_MODULE, join_kernels::HASH_JOIN_BUILD);
        assert!(
            build_fn.is_some(),
            "hash_join_build function should be accessible"
        );

        let probe_fn = inner.get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE);
        assert!(
            probe_fn.is_some(),
            "hash_join_probe function should be accessible"
        );

        // Dedup kernels
        let mark_fn = inner.get_func(DEDUP_MODULE, dedup_kernels::MARK_DUPLICATES);
        assert!(
            mark_fn.is_some(),
            "mark_duplicates function should be accessible"
        );

        let compact_fn = inner.get_func(DEDUP_MODULE, dedup_kernels::COMPACT_ROWS);
        assert!(
            compact_fn.is_some(),
            "compact_rows function should be accessible"
        );

        // GroupBy kernels
        let boundaries_fn =
            inner.get_func(GROUPBY_MODULE, groupby_kernels::DETECT_GROUP_BOUNDARIES);
        assert!(
            boundaries_fn.is_some(),
            "detect_group_boundaries function should be accessible"
        );

        let count_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_COUNT);
        assert!(
            count_fn.is_some(),
            "groupby_count function should be accessible"
        );

        let sum_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_SUM);
        assert!(
            sum_fn.is_some(),
            "groupby_sum function should be accessible"
        );

        let min_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MIN);
        assert!(
            min_fn.is_some(),
            "groupby_min function should be accessible"
        );

        let max_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MAX);
        assert!(
            max_fn.is_some(),
            "groupby_max function should be accessible"
        );

        // Circuit kernels (XGCF forward/backward)
        let xgcf_forward = inner.get_func(CIRCUIT_MODULE, "xgcf_forward_level");
        assert!(
            xgcf_forward.is_some(),
            "xgcf_forward_level function should be accessible"
        );

        let xgcf_backward_propagate =
            inner.get_func(CIRCUIT_MODULE, "xgcf_backward_level_propagate");
        assert!(
            xgcf_backward_propagate.is_some(),
            "xgcf_backward_level_propagate function should be accessible"
        );

        let xgcf_backward_decision_grad =
            inner.get_func(CIRCUIT_MODULE, "xgcf_backward_level_decision_grad");
        assert!(
            xgcf_backward_decision_grad.is_some(),
            "xgcf_backward_level_decision_grad function should be accessible"
        );

        let xgcf_backward_lit_grad = inner.get_func(CIRCUIT_MODULE, "xgcf_backward_level_lit_grad");
        assert!(
            xgcf_backward_lit_grad.is_some(),
            "xgcf_backward_level_lit_grad function should be accessible"
        );

        // Neural fast-path kernels (AD chain weight fill + gradient scatter)
        let neural_fill = inner.get_func("xlog_neural", "neural_fill_ad_chain_f32");
        assert!(
            neural_fill.is_some(),
            "neural_fill_ad_chain_f32 function should be accessible"
        );
        let neural_scatter = inner.get_func("xlog_neural", "neural_scatter_ad_chain_grads_f32");
        assert!(
            neural_scatter.is_some(),
            "neural_scatter_ad_chain_grads_f32 function should be accessible"
        );
    }

    #[test]
    fn test_module_names_unique() {
        // Ensure module names don't collide
        assert_ne!(JOIN_MODULE, DEDUP_MODULE);
        assert_ne!(JOIN_MODULE, GROUPBY_MODULE);
        assert_ne!(DEDUP_MODULE, GROUPBY_MODULE);
    }

    // Helper function to create test provider
    fn create_test_provider() -> Option<CudaKernelProvider> {
        if !has_cuda_device() {
            return None;
        }
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        CudaKernelProvider::new(device, memory).ok()
    }

    // Helper function to create a CudaBuffer with U32 data
    fn create_test_buffer(
        provider: &CudaKernelProvider,
        data: &[u32],
        col_name: &str,
    ) -> CudaBuffer {
        let schema = Schema::new(vec![(col_name.to_string(), ScalarType::U32)]);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut col = provider.memory().alloc::<u8>(bytes.len()).expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .expect("htod");

        provider
            .buffer_from_columns(vec![col.into()], data.len() as u64, schema)
            .expect("buffer")
    }

    // Helper function to create an empty buffer with correct column count
    fn create_empty_test_buffer(provider: &CudaKernelProvider, schema: Schema) -> CudaBuffer {
        let mut columns = Vec::with_capacity(schema.arity());
        for _ in 0..schema.arity() {
            columns.push(provider.memory().alloc::<u8>(0).expect("alloc").into());
        }
        provider
            .buffer_from_columns(columns, 0, schema)
            .expect("buffer")
    }

    // Helper function to read U32 data from CudaBuffer
    fn read_buffer_u32(provider: &CudaKernelProvider, buffer: &CudaBuffer, col: usize) -> Vec<u32> {
        if buffer.is_empty() || buffer.column(col).is_none() {
            return vec![];
        }
        let num_rows = buffer.num_rows() as usize;
        let mut bytes = vec![0u8; num_rows * 4];
        provider
            .device()
            .inner()
            .dtoh_sync_copy_into(buffer.column(col).unwrap(), &mut bytes)
            .expect("dtoh");
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn test_compact_device_mask_respects_mask_len_smaller_than_row_cap() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
        let base = create_test_buffer(&provider, &[1, 2, 3, 4, 5, 6, 7, 8], "id");

        let row_cap = 16u64;
        let data: Vec<u32> = (0..row_cap as u32).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut col = provider.memory().alloc::<u8>(bytes.len()).expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .expect("htod");
        let expanded = provider
            .buffer_from_columns_with_device_count(vec![col.into()], row_cap, schema, &base)
            .expect("buffer");

        let mask: Vec<u8> = vec![1, 0, 1, 0, 1, 0, 1, 0];
        let (prefix_sum, count) = provider.prefix_sum_mask(&mask).expect("prefix sum");

        let mut d_mask = provider.memory().alloc::<u8>(mask.len()).expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&mask, &mut d_mask)
            .expect("mask htod");

        let mut d_prefix = provider
            .memory()
            .alloc::<u32>(prefix_sum.len())
            .expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&prefix_sum, &mut d_prefix)
            .expect("prefix htod");

        let mut d_out_count = provider.memory().alloc::<u32>(1).expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&[count], &mut d_out_count)
            .expect("count htod");

        let compacted = provider
            .compact_buffer_by_device_mask_device_count(&expanded, &d_mask, &d_prefix, d_out_count)
            .expect("compact");

        assert_eq!(compacted.num_rows(), mask.len() as u64);
        let device_rows = provider.device_row_count(&compacted).expect("row count");
        assert_eq!(device_rows as u32, count);
    }

    #[test]
    fn test_clone_buffer_preserves_device_count() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
        let ids: Vec<u32> = vec![10, 20, 30];
        let buffer = provider
            .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
            .unwrap();

        let cloned = provider.clone_buffer(&buffer).unwrap();

        let mut host_count = [0u32];
        provider
            .device()
            .inner()
            .dtoh_sync_copy_into(cloned.num_rows_device(), &mut host_count)
            .unwrap();
        assert_eq!(host_count[0], 3);
    }

    /// `clone_buffer` must propagate the host-side `cached_row_count` so
    /// downstream code can read the row count without a D2H round-trip.
    /// Without this propagation, buffers flowed through the relation store
    /// (`CompiledIlpProgram::put_relation` calls `clone_buffer` before
    /// storing) lose their host-visible count, forcing consumers to choose
    /// between an extra D2H (blowing the D2H-budget gates used by M8/Phase 1
    /// and beyond) and a hard error. This test pins the cache-propagation
    /// contract directly.
    #[test]
    fn test_clone_buffer_preserves_cached_row_count() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
        let ids: Vec<u32> = vec![7, 11, 13, 17];
        let source = provider
            .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
            .unwrap();
        // Source's cache is populated by the `create_buffer_from_*` path;
        // verify the precondition so a regression in that path shows up here
        // rather than silently passing the real assertion below.
        assert_eq!(
            source.cached_row_count(),
            Some(4),
            "source buffer should have its cached row count populated by \
             create_buffer_from_slices"
        );

        let cloned = provider.clone_buffer(&source).unwrap();

        assert_eq!(
            cloned.cached_row_count(),
            Some(4),
            "clone_buffer must propagate cached_row_count from source to clone",
        );
    }

    // ============== Hash Join Tests ==============

    #[test]
    fn test_hash_join_empty_inputs() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = create_empty_test_buffer(&provider, schema.clone());

        // Join empty with empty
        let result = provider.hash_join(&empty, &empty, &[0], &[0]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_hash_join_validation() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let left = create_test_buffer(&provider, &[1, 2, 3], "left_key");
        let right = create_test_buffer(&provider, &[2, 3, 4], "right_key");

        // Empty key columns
        let result = provider.hash_join(&left, &right, &[], &[0]);
        assert!(result.is_err());

        // Mismatched key lengths
        let result = provider.hash_join(&left, &right, &[0], &[0, 0]);
        assert!(result.is_err());
    }

    // ============== Dedup Tests ==============

    #[test]
    fn test_dedup_empty_input() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = create_empty_test_buffer(&provider, schema);

        let result = provider.dedup(&empty, &[0]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_dedup_validation() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&provider, &[1, 1, 2, 2, 3], "key");

        // Empty key columns
        let result = provider.dedup(&buffer, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dedup_with_duplicates() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Test dedup with duplicates: [3, 1, 2, 1, 3, 2]
        let buffer = create_test_buffer(&provider, &[3, 1, 2, 1, 3, 2], "key");
        let deduped = provider.dedup(&buffer, &[0]).unwrap();

        let dedup_count = provider
            .device_row_count(&deduped)
            .expect("read dedup row count");
        assert_eq!(dedup_count, 3, "Should have 3 unique values");

        let result = provider.download_column::<u32>(&deduped, 0).unwrap();
        // Result should be sorted and deduped
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_dedup_larger_input() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create input with duplicates: 0..500 ++ 250..750 = 1000 elements, 750 unique
        let a: Vec<u32> = (0..500).collect();
        let b: Vec<u32> = (250..750).collect();
        let input: Vec<u32> = a.iter().chain(b.iter()).copied().collect();

        let buffer = create_test_buffer(&provider, &input, "key");
        let deduped = provider.dedup(&buffer, &[0]).unwrap();

        let dedup_count = provider
            .device_row_count(&deduped)
            .expect("read dedup row count");
        assert_eq!(dedup_count, 750, "Should have 750 unique values (0..750)");

        // Verify output is sorted
        let result = provider.download_column::<u32>(&deduped, 0).unwrap();
        let is_sorted = result.windows(2).all(|w| w[0] <= w[1]);
        assert!(is_sorted, "Output should be sorted");

        // Verify expected values
        let expected: Vec<u32> = (0..750).collect();
        assert_eq!(result, expected);
    }

    // ============== Union Tests ==============

    #[test]
    fn test_union_empty_inputs() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = create_empty_test_buffer(&provider, schema.clone());

        // Empty union empty
        let result = provider.union(&empty, &empty);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());

        // Non-empty union empty
        let a = create_test_buffer(&provider, &[1, 2, 3], "key");
        let empty2 = create_empty_test_buffer(&provider, schema);
        let result = provider.union(&a, &empty2);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_union_schema_type_mismatch() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let a = create_test_buffer(&provider, &[1, 2], "col_a");
        let b = create_test_buffer(&provider, &[3, 4], "col_b");

        // Different column names but same types should succeed (Datalog union semantics)
        let result = provider.union(&a, &b);
        assert!(result.is_ok());

        // Different arity should fail - create a 2-column buffer
        let two_col_schema = Schema::new(vec![
            ("x".to_string(), ScalarType::U32),
            ("y".to_string(), ScalarType::U32),
        ]);
        let c = provider
            .create_buffer_from_u32_columns(&[&[1, 2], &[3, 4]], two_col_schema)
            .unwrap();
        let result = provider.union(&a, &c);
        assert!(result.is_err());
    }

    // ============== Diff Tests ==============

    #[test]
    fn test_diff_empty_inputs() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = create_empty_test_buffer(&provider, schema.clone());

        // Empty diff empty
        let result = provider.diff(&empty, &empty);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());

        // Non-empty diff empty should return all of a
        let a = create_test_buffer(&provider, &[1, 2, 3], "key");
        let empty2 = create_empty_test_buffer(&provider, schema);
        let result = provider.diff(&a, &empty2);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_diff_basic() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let a = create_test_buffer(&provider, &[1, 2, 3, 4, 5], "key");
        let b = create_test_buffer(&provider, &[2, 4], "key");

        let result = provider.diff(&a, &b);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3); // 1, 3, 5

        let values = read_buffer_u32(&provider, &result, 0);
        assert_eq!(values, vec![1, 3, 5]);
    }

    #[test]
    fn test_diff_all_removed() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let a = create_test_buffer(&provider, &[1, 2, 3], "key");
        let b = create_test_buffer(&provider, &[1, 2, 3, 4, 5], "key");

        let result = provider.diff(&a, &b);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_diff_schema_mismatch() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Different column names with same types should work (Datalog semantics)
        let a = create_test_buffer(&provider, &[1, 2], "col_a");
        let b = create_test_buffer(&provider, &[1, 2], "col_b");
        let result = provider.diff(&a, &b);
        assert!(
            result.is_ok(),
            "Same types with different names should succeed"
        );

        // Create buffers with different arities (this should fail)
        let schema_2col = Schema::new(vec![
            ("c0".to_string(), ScalarType::U32),
            ("c1".to_string(), ScalarType::U32),
        ]);

        let bytes_2col: Vec<u8> = [1u32, 2, 3, 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut col0 = provider
            .memory()
            .alloc::<u8>(bytes_2col.len() / 2)
            .expect("alloc");
        let mut col1 = provider
            .memory()
            .alloc::<u8>(bytes_2col.len() / 2)
            .expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&bytes_2col[..8], &mut col0)
            .expect("htod");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&bytes_2col[8..], &mut col1)
            .expect("htod");
        let buffer_2col = provider
            .buffer_from_columns(vec![col0.into(), col1.into()], 2, schema_2col)
            .expect("buffer");

        let buffer_1col = create_test_buffer(&provider, &[1, 2], "c0");

        let result = provider.diff(&buffer_2col, &buffer_1col);
        assert!(result.is_err(), "Different arities should fail");
    }

    // ============== GroupBy Aggregation Tests ==============

    #[test]
    fn test_groupby_empty_input() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = create_empty_test_buffer(&provider, schema);

        let result = provider.groupby_agg(&empty, &[0], AggOp::Count, 0);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_groupby_validation() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&provider, &[1, 1, 2, 2, 3], "key");

        // Empty key columns
        let result = provider.groupby_agg(&buffer, &[], AggOp::Count, 0);
        assert!(result.is_err());

        // Value column out of bounds
        let result = provider.groupby_agg(&buffer, &[0], AggOp::Count, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_groupby_logsumexp() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create buffer with U32 keys and F64 values
        // Group 0 (key=1): values 1.0, 2.0 -> logsumexp = log(e^1 + e^2) ≈ 2.31326
        // Group 1 (key=2): values 3.0, 4.0 -> logsumexp = log(e^3 + e^4) ≈ 4.31326
        let keys: Vec<u32> = vec![1, 1, 2, 2];
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

        let schema = Schema::new(vec![
            ("key".to_string(), ScalarType::U32),
            ("value".to_string(), ScalarType::F64),
        ]);

        // Create key column
        let key_bytes: Vec<u8> = keys.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut key_col = provider
            .memory()
            .alloc::<u8>(key_bytes.len())
            .expect("alloc key");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&key_bytes, &mut key_col)
            .expect("upload key");

        // Create value column
        let val_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut val_col = provider
            .memory()
            .alloc::<u8>(val_bytes.len())
            .expect("alloc val");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&val_bytes, &mut val_col)
            .expect("upload val");

        let buffer = provider
            .buffer_from_columns(vec![key_col.into(), val_col.into()], 4, schema)
            .expect("buffer");

        // Run LogSumExp aggregation grouped by key column (0), aggregating value column (1)
        let result = provider.groupby_agg(&buffer, &[0], AggOp::LogSumExp, 1);
        assert!(
            result.is_ok(),
            "groupby_agg with LogSumExp should succeed: {:?}",
            result.err()
        );

        let result = result.unwrap();
        let group_count = provider
            .device_row_count(&result)
            .expect("read group count");
        assert_eq!(group_count, 2, "Should have 2 groups");

        // Download results
        let result_values = provider
            .download_column::<f64>(&result, 1)
            .expect("download result");

        // Expected values:
        // logsumexp(1.0, 2.0) = 2.0 + log(exp(1.0-2.0) + exp(2.0-2.0)) = 2.0 + log(e^-1 + 1) ≈ 2.31326
        // logsumexp(3.0, 4.0) = 4.0 + log(exp(3.0-4.0) + exp(4.0-4.0)) = 4.0 + log(e^-1 + 1) ≈ 4.31326
        let expected_0 = 2.0_f64 + ((-1.0_f64).exp() + 1.0_f64).ln(); // ≈ 2.31326
        let expected_1 = 4.0_f64 + ((-1.0_f64).exp() + 1.0_f64).ln(); // ≈ 4.31326

        let tolerance = 1e-5;
        assert!(
            (result_values[0] - expected_0).abs() < tolerance,
            "Group 0 logsumexp mismatch: got {}, expected {}",
            result_values[0],
            expected_0
        );
        assert!(
            (result_values[1] - expected_1).abs() < tolerance,
            "Group 1 logsumexp mismatch: got {}, expected {}",
            result_values[1],
            expected_1
        );
    }

    // ============== Schema Helper Tests ==============

    #[test]
    fn test_combine_schemas() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let left = Schema::new(vec![("a".to_string(), ScalarType::U32)]);
        let right = Schema::new(vec![("b".to_string(), ScalarType::U64)]);

        let combined = provider.combine_schemas(&left, &right);
        assert_eq!(combined.arity(), 2);
        assert_eq!(combined.column_type(0), Some(ScalarType::U32));
        assert_eq!(combined.column_type(1), Some(ScalarType::U64));
    }

    #[test]
    fn test_groupby_result_schema() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let input = Schema::new(vec![
            ("key".to_string(), ScalarType::U32),
            ("value".to_string(), ScalarType::U32),
        ]);

        // Count result schema (u64 to match predicate declarations)
        let count_schema =
            provider.groupby_multi_agg_result_schema(&input, &[0], &[(1, AggOp::Count)]);
        assert_eq!(count_schema.arity(), 2);
        assert_eq!(count_schema.column_type(1), Some(ScalarType::U64));

        // Sum result schema
        let sum_schema = provider.groupby_multi_agg_result_schema(&input, &[0], &[(1, AggOp::Sum)]);
        assert_eq!(sum_schema.arity(), 2);
        assert_eq!(sum_schema.column_type(1), Some(ScalarType::U64));

        // Min/Max result schema
        let min_schema = provider.groupby_multi_agg_result_schema(&input, &[0], &[(1, AggOp::Min)]);
        assert_eq!(min_schema.arity(), 2);
        assert_eq!(min_schema.column_type(1), Some(ScalarType::U32));
    }

    #[test]
    fn test_groupby_multi_agg_sum_returns_u64_schema() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device");
                return;
            }
        };

        let schema = Schema::new(vec![
            ("key".to_string(), ScalarType::U32),
            ("val".to_string(), ScalarType::U32),
        ]);

        let result_schema =
            provider.groupby_multi_agg_result_schema(&schema, &[0], &[(1, AggOp::Sum)]);

        // Sum should return U64 to prevent overflow
        assert_eq!(
            result_schema.column_type(1),
            Some(ScalarType::U64),
            "Sum aggregation should return U64 type, not U32"
        );
    }

    #[test]
    fn test_join_custom_max_output() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create buffers that produce more than 10 results when joined
        // Left: [1, 1, 1, 1, 2, 2, 2, 2] - 4 copies of 1, 4 copies of 2
        // Right: [1, 1, 1, 2, 2, 2] - 3 copies of 1, 3 copies of 2
        // Join produces: 4*3 + 4*3 = 24 results
        let left = create_test_buffer(&provider, &[1, 1, 1, 1, 2, 2, 2, 2], "left_key");
        let right = create_test_buffer(&provider, &[1, 1, 1, 2, 2, 2], "right_key");

        // Test with limit of 10 - should get at most 10
        let result_limited = provider
            .hash_join_v2_with_limit(&left, &right, &[0], &[0], JoinType::Inner, Some(10))
            .expect("join with limit should succeed");
        assert!(
            result_limited.num_rows() <= 10,
            "With limit 10, got {} rows but expected at most 10",
            result_limited.num_rows()
        );

        // Test with None (default) - should get all 24 results
        let result_unlimited = provider
            .hash_join_v2_with_limit(&left, &right, &[0], &[0], JoinType::Inner, None)
            .expect("join without limit should succeed");
        assert_eq!(
            result_unlimited.num_rows(),
            24,
            "Without limit, expected 24 rows but got {}",
            result_unlimited.num_rows()
        );

        // Test legacy API still works (backward compatibility)
        let result_legacy = provider
            .hash_join_v2(&left, &right, &[0], &[0], JoinType::Inner)
            .expect("legacy hash_join_v2 should succeed");
        assert_eq!(
            result_legacy.num_rows(),
            24,
            "Legacy API without limit, expected 24 rows but got {}",
            result_legacy.num_rows()
        );
    }

    // ============== Arithmetic Operation Tests ==============

    /// Helper to create a test provider for arithmetic tests
    fn create_arith_test_provider() -> Option<CudaKernelProvider> {
        if !has_cuda_device() {
            return None;
        }
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        CudaKernelProvider::new(device, memory).ok()
    }

    /// Helper to create an i64 buffer for arithmetic tests
    fn create_i64_buffer(provider: &CudaKernelProvider, data: &[i64]) -> CudaBuffer {
        let schema = Schema::new(vec![("col".to_string(), ScalarType::I64)]);
        provider
            .create_buffer_from_slice::<i64>(data, schema)
            .unwrap()
    }

    /// Helper to create an f64 buffer for arithmetic tests
    fn create_f64_buffer(provider: &CudaKernelProvider, data: &[f64]) -> CudaBuffer {
        let schema = Schema::new(vec![("col".to_string(), ScalarType::F64)]);
        provider
            .create_buffer_from_slice::<f64>(data, schema)
            .unwrap()
    }

    #[test]
    fn test_add_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[1, 2, 3, 4, 5]);
        let b = create_i64_buffer(&provider, &[10, 20, 30, 40, 50]);

        let result = provider.add_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![11, 22, 33, 44, 55]);
    }

    #[test]
    fn test_sub_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[10, 20, 30, 40, 50]);
        let b = create_i64_buffer(&provider, &[1, 2, 3, 4, 5]);

        let result = provider.sub_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![9, 18, 27, 36, 45]);
    }

    #[test]
    fn test_mul_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[2, 3, 4, 5, 6]);
        let b = create_i64_buffer(&provider, &[3, 4, 5, 6, 7]);

        let result = provider.mul_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![6, 12, 20, 30, 42]);
    }

    #[test]
    fn test_div_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[100, 200, 300, 400]);
        let b = create_i64_buffer(&provider, &[10, 20, 30, 40]);

        let result = provider.div_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![10, 10, 10, 10]);
    }

    #[test]
    fn test_div_columns_by_zero() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[10, 20, 30]);
        let b = create_i64_buffer(&provider, &[2, 0, 3]); // Note: division by zero

        let result = provider.div_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        // Division by zero returns i64::MAX
        assert_eq!(values, vec![5, i64::MAX, 10]);
    }

    #[test]
    fn test_mod_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[17, 23, 100, 7]);
        let b = create_i64_buffer(&provider, &[5, 7, 30, 3]);

        let result = provider.mod_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![2, 2, 10, 1]);
    }

    #[test]
    fn test_mod_columns_by_zero() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[10, 20]);
        let b = create_i64_buffer(&provider, &[3, 0]); // Note: mod by zero

        let result = provider.mod_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        // Mod by zero returns 0
        assert_eq!(values, vec![1, 0]);
    }

    #[test]
    fn test_abs_column_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[-5, 10, -15, 20, 0]);

        let result = provider.abs_column(&a).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![5, 10, 15, 20, 0]);
    }

    #[test]
    fn test_min_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[5, 10, 15, 20]);
        let b = create_i64_buffer(&provider, &[3, 12, 10, 25]);

        let result = provider.min_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![3, 10, 10, 20]);
    }

    #[test]
    fn test_max_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[5, 10, 15, 20]);
        let b = create_i64_buffer(&provider, &[3, 12, 10, 25]);

        let result = provider.max_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![5, 12, 15, 25]);
    }

    #[test]
    fn test_add_columns_f64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_f64_buffer(&provider, &[1.5, 2.5, 3.5]);
        let b = create_f64_buffer(&provider, &[0.5, 1.5, 2.5]);

        let result = provider.add_columns(&a, &b).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        assert_eq!(values, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_mul_columns_f64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_f64_buffer(&provider, &[2.0, 3.0, 4.0]);
        let b = create_f64_buffer(&provider, &[1.5, 2.0, 2.5]);

        let result = provider.mul_columns(&a, &b).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        assert_eq!(values, vec![3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_div_columns_f64_by_zero() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_f64_buffer(&provider, &[1.0, -1.0, 0.0]);
        let b = create_f64_buffer(&provider, &[0.0, 0.0, 0.0]);

        let result = provider.div_columns(&a, &b).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        // IEEE 754: 1.0/0.0 = Inf, -1.0/0.0 = -Inf, 0.0/0.0 = NaN
        assert!(values[0].is_infinite() && values[0].is_sign_positive());
        assert!(values[1].is_infinite() && values[1].is_sign_negative());
        assert!(values[2].is_nan());
    }

    #[test]
    fn test_pow_columns() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let base = create_i64_buffer(&provider, &[2, 3, 4, 5]);
        let exp = create_i64_buffer(&provider, &[3, 2, 2, 1]);

        let result = provider.pow_columns(&base, &exp).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        // pow always returns f64
        assert_eq!(values, vec![8.0, 9.0, 16.0, 5.0]);
    }

    #[test]
    fn test_pow_columns_fractional_exp() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let base = create_f64_buffer(&provider, &[4.0, 9.0, 27.0]);
        let exp = create_f64_buffer(&provider, &[0.5, 0.5, 1.0 / 3.0]);

        let result = provider.pow_columns(&base, &exp).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        // sqrt(4) = 2, sqrt(9) = 3, cbrt(27) = 3
        assert!((values[0] - 2.0).abs() < 1e-10);
        assert!((values[1] - 3.0).abs() < 1e-10);
        assert!((values[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cast_i64_to_f64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[1, 2, 3, 4, 5]);

        let result = provider.cast_column(&a, ScalarType::F64).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_cast_f64_to_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_f64_buffer(&provider, &[1.9, 2.1, 3.5, 4.0, 5.7]);

        let result = provider.cast_column(&a, ScalarType::I64).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        // Truncation towards zero
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_cast_i64_to_i32() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[1, 2, 3, 100, 200]);

        let result = provider.cast_column(&a, ScalarType::I32).unwrap();
        let values = provider.download_column::<i32>(&result, 0).unwrap();

        assert_eq!(values, vec![1, 2, 3, 100, 200]);
    }

    #[test]
    fn test_arithmetic_row_count_mismatch() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[1, 2, 3]);
        let b = create_i64_buffer(&provider, &[1, 2]); // Different size

        let result = provider.add_columns(&a, &b);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("Row count mismatch"));
    }

    #[test]
    fn test_arithmetic_empty_buffers() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[]);
        let b = create_i64_buffer(&provider, &[]);

        let result = provider.add_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, Vec::<i64>::new());
    }

    #[test]
    fn test_wrapping_arithmetic_overflow() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[i64::MAX, i64::MIN]);
        let b = create_i64_buffer(&provider, &[1, -1]);

        // Addition should wrap
        let add_result = provider.add_columns(&a, &b).unwrap();
        let add_values = provider.download_column::<i64>(&add_result, 0).unwrap();
        assert_eq!(add_values[0], i64::MIN); // MAX + 1 wraps to MIN
        assert_eq!(add_values[1], i64::MAX); // MIN - 1 wraps to MAX
    }

    #[test]
    fn test_abs_column_f64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_f64_buffer(&provider, &[-1.5, 2.5, -3.5, 0.0]);

        let result = provider.abs_column(&a).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        assert_eq!(values, vec![1.5, 2.5, 3.5, 0.0]);
    }

    #[test]
    fn test_min_max_columns_f64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_f64_buffer(&provider, &[1.5, 5.0, 3.0]);
        let b = create_f64_buffer(&provider, &[2.0, 3.0, 4.0]);

        let min_result = provider.min_columns(&a, &b).unwrap();
        let min_values = provider.download_column::<f64>(&min_result, 0).unwrap();
        assert_eq!(min_values, vec![1.5, 3.0, 3.0]);

        let max_result = provider.max_columns(&a, &b).unwrap();
        let max_values = provider.download_column::<f64>(&max_result, 0).unwrap();
        assert_eq!(max_values, vec![2.0, 5.0, 4.0]);
    }
}
