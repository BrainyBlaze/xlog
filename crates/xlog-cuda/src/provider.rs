//! CUDA kernel provider implementation
//!
//! This module provides the `CudaKernelProvider` which manages pre-compiled
//! PTX kernels for GPU execution of relational operations (join, dedup, groupby).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::marker::PhantomData;

use cudarc::driver::{
    CudaViewMut, DevicePtr, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig,
};
use cudarc::nvrtc::Ptx;
use std::ffi::c_void;
use xlog_core::{AggOp, Result, ScalarType, Schema, XlogError};

use crate::{memory::{CudaColumn, TrackedCudaSlice}, CudaBuffer, CudaDevice, GpuMemoryManager};

// Embedded PTX sources (pre-compiled from .cu files with nvcc -ptx -arch=sm_70)
const JOIN_PTX: &str = include_str!("../../../kernels/join.ptx");
const DEDUP_PTX: &str = include_str!("../../../kernels/dedup.ptx");
const GROUPBY_PTX: &str = include_str!("../../../kernels/groupby.ptx");
const SCAN_PTX: &str = include_str!("../../../kernels/scan.ptx");
const SORT_PTX: &str = include_str!("../../../kernels/sort.ptx");
const FILTER_PTX: &str = include_str!("../../../kernels/filter.ptx");
const SET_OPS_PTX: &str = include_str!("../../../kernels/set_ops.ptx");
const PACK_PTX: &str = include_str!("../../../kernels/pack.ptx");
const CIRCUIT_PTX: &str = include_str!("../../../kernels/circuit.ptx");
const MC_SAMPLE_PTX: &str = include_str!("../../../kernels/mc_sample.ptx");
const ARITH_PTX: &str = include_str!("../../../kernels/arith.ptx");
const SAT_PTX: &str = include_str!("../../../kernels/sat.ptx");
const NEURAL_PTX: &str = include_str!("../../../kernels/neural.ptx");

#[derive(Clone, Copy)]
struct RawCudaView<'a, T> {
    ptr: cudarc::driver::sys::CUdeviceptr,
    len: usize,
    _marker: PhantomData<&'a [T]>,
}

impl<'a, T> DeviceSlice<T> for RawCudaView<'a, T> {
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T> DevicePtr<T> for RawCudaView<'a, T> {
    fn device_ptr(&self) -> &cudarc::driver::sys::CUdeviceptr {
        &self.ptr
    }
}

unsafe impl<'a, T: DeviceRepr> DeviceRepr for &RawCudaView<'a, T> {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut c_void {
        let ptr = DevicePtr::device_ptr(*self);
        ptr as *const cudarc::driver::sys::CUdeviceptr as *mut c_void
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
pub const ARITH_MODULE: &str = "xlog_arith";
pub const SAT_MODULE: &str = "xlog_sat";
pub const NEURAL_MODULE: &str = "xlog_neural";

/// Kernel function names in the Monte Carlo sampling module
pub mod mc_sample_kernels {
    pub const MC_SAMPLE_BERNOULLI: &str = "mc_sample_bernoulli";
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
}

/// Kernel function names in the groupby module
pub mod groupby_kernels {
    pub const DETECT_GROUP_BOUNDARIES: &str = "detect_group_boundaries";
    pub const DETECT_BOUNDARIES: &str = "detect_boundaries";
    pub const EXTRACT_GROUP_KEYS: &str = "extract_group_keys";
    pub const GROUP_IDS_FROM_BOUNDARIES: &str = "group_ids_from_boundaries";
    pub const GROUP_START_INDICES: &str = "group_start_indices";
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
    pub const COMPACT_U32_BY_MASK: &str = "compact_u32_by_mask";
    pub const COMPACT_I64_BY_MASK: &str = "compact_i64_by_mask";
    pub const COMPACT_F64_BY_MASK: &str = "compact_f64_by_mask";
    pub const COMPACT_BYTES_BY_MASK: &str = "compact_bytes_by_mask";
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
    /// Vectorized pack for 8-byte aligned columns
    pub const PACK_KEYS_ALIGNED: &str = "pack_keys_aligned";
    /// Unpack single column from packed row data
    pub const UNPACK_COLUMN: &str = "unpack_column";
    /// Gather rows from packed data based on index array
    pub const GATHER_PACKED_ROWS: &str = "gather_packed_rows";
    /// Scatter write: distribute packed rows to non-contiguous output positions
    pub const SCATTER_PACKED_ROWS: &str = "scatter_packed_rows";
    /// Compare packed keys for equality
    pub const COMPARE_PACKED_KEYS: &str = "compare_packed_keys";
}

/// Kernel function names in the circuit module
pub mod circuit_kernels {
    pub const XGCF_FORWARD_LEVEL: &str = "xgcf_forward_level";
    pub const XGCF_BACKWARD_LEVEL_PROPAGATE: &str = "xgcf_backward_level_propagate";
    pub const XGCF_BACKWARD_LEVEL_DECISION_GRAD: &str = "xgcf_backward_level_decision_grad";
    pub const XGCF_BACKWARD_LEVEL_LIT_GRAD: &str = "xgcf_backward_level_lit_grad";
}

/// Kernel function names in the SAT module
pub mod sat_kernels {
    pub const SAT_CDCL_SOLVE: &str = "sat_cdcl_solve";
    pub const SAT_CHECK_MODEL: &str = "sat_check_model";
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
    /// Loads all PTX modules into the CUDA device.
    /// The modules are compiled for sm_70 (Volta) and above.
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
        // Load join module
        device
            .inner()
            .load_ptx(
                Ptx::from_src(JOIN_PTX),
                JOIN_MODULE,
                &[
                    join_kernels::HASH_JOIN_BUILD,
                    join_kernels::HASH_JOIN_PROBE,
                    join_kernels::COMPUTE_COMPOSITE_HASH,
                    join_kernels::HASH_JOIN_BUCKET_COUNT_V2,
                    join_kernels::HASH_JOIN_SCATTER_V2,
                    join_kernels::HASH_JOIN_PROBE_V2,
                    join_kernels::HASH_JOIN_SEMI,
                    join_kernels::HASH_JOIN_ANTI,
                    join_kernels::INIT_HASH_TABLE,
                ],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load join PTX: {}", e)))?;

        // Load dedup module
        device
            .inner()
            .load_ptx(
                Ptx::from_src(DEDUP_PTX),
                DEDUP_MODULE,
                &[
                    dedup_kernels::MARK_DUPLICATES,
                    dedup_kernels::MARK_UNIQUE_COLUMNAR,
                    dedup_kernels::MARK_UNIQUE_AND_SCAN_COLUMNAR,
                    dedup_kernels::COMPACT_ROWS,
                ],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load dedup PTX: {}", e)))?;

        // Load groupby module
        device
            .inner()
            .load_ptx(
                Ptx::from_src(GROUPBY_PTX),
                GROUPBY_MODULE,
                &[
                    groupby_kernels::DETECT_GROUP_BOUNDARIES,
                    groupby_kernels::DETECT_BOUNDARIES,
                    groupby_kernels::EXTRACT_GROUP_KEYS,
                    groupby_kernels::GROUP_IDS_FROM_BOUNDARIES,
                    groupby_kernels::GROUP_START_INDICES,
                    groupby_kernels::GROUPBY_COUNT,
                    groupby_kernels::GROUPBY_SUM,
                    groupby_kernels::GROUPBY_MIN,
                    groupby_kernels::GROUPBY_MAX,
                    groupby_kernels::GROUPBY_LOGSUMEXP_MAX,
                    groupby_kernels::GROUPBY_LOGSUMEXP_SUMEXP,
                    groupby_kernels::GROUPBY_LOGSUMEXP_FINAL,
                ],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load groupby PTX: {}", e)))?;

        // Load scan module
        device
            .inner()
            .load_ptx(
                Ptx::from_src(SCAN_PTX),
                SCAN_MODULE,
                &[
                    scan_kernels::BLOCK_INCLUSIVE_SCAN,
                    scan_kernels::ADD_BLOCK_OFFSETS,
                    scan_kernels::EXCLUSIVE_SCAN_MASK,
                    scan_kernels::COUNT_MASK,
                    scan_kernels::MULTIBLOCK_SCAN_PHASE1,
                    scan_kernels::MULTIBLOCK_SCAN_U32_PHASE1,
                    scan_kernels::MULTIBLOCK_SCAN_PHASE2,
                    scan_kernels::MULTIBLOCK_SCAN_PHASE3,
                ],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load scan PTX: {}", e)))?;

        // Load sort module
        device
            .inner()
            .load_ptx(
                Ptx::from_src(SORT_PTX),
                SORT_MODULE,
                &[
                    sort_kernels::RADIX_HISTOGRAM,
                    sort_kernels::RADIX_SCATTER,
                    sort_kernels::COMPUTE_RANKS,
                    sort_kernels::RADIX_SCATTER_STABLE,
                    sort_kernels::COMPUTE_DIGIT_PREFIX_SUMS,
                    sort_kernels::INIT_INDICES,
                    sort_kernels::APPLY_PERMUTATION_U32,
                    sort_kernels::APPLY_PERMUTATION_BYTES,
                    sort_kernels::GATHER_KEYS_I32_ORDERED_U32,
                    sort_kernels::GATHER_KEYS_F32_ORDERED_U32,
                    sort_kernels::GATHER_KEYS_BOOL_ORDERED_U32,
                    sort_kernels::GATHER_KEYS_U64_LO_U32,
                    sort_kernels::GATHER_KEYS_U64_HI_U32,
                    sort_kernels::GATHER_KEYS_I64_LO_U32,
                    sort_kernels::GATHER_KEYS_I64_HI_U32,
                    sort_kernels::GATHER_KEYS_F64_LO_U32,
                    sort_kernels::GATHER_KEYS_F64_HI_U32,
                ],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load sort PTX: {}", e)))?;

        // Load filter module
        device
            .inner()
            .load_ptx(
                Ptx::from_src(FILTER_PTX),
                FILTER_MODULE,
                &[
                    filter_kernels::FILTER_COMPARE_U32,
                    filter_kernels::FILTER_COMPARE_I64,
                    filter_kernels::FILTER_COMPARE_F64,
                    filter_kernels::FILTER_COMPARE_I32,
                    filter_kernels::FILTER_COMPARE_U64,
                    filter_kernels::FILTER_COMPARE_F32,
                    filter_kernels::FILTER_COMPARE_U8,
                    filter_kernels::FILTER_COMPARE_U32_SCAN_PHASE1,
                    filter_kernels::FILTER_COMPARE_F64_SCAN_PHASE1,
                    filter_kernels::FILTER_COMPARE_F32_SCAN_PHASE1,
                    filter_kernels::FILTER_COMPARE_U32_COL,
                    filter_kernels::FILTER_COMPARE_I32_COL,
                    filter_kernels::FILTER_COMPARE_I64_COL,
                    filter_kernels::FILTER_COMPARE_U64_COL,
                    filter_kernels::FILTER_COMPARE_F32_COL,
                    filter_kernels::FILTER_COMPARE_F64_COL,
                    filter_kernels::FILTER_COMPARE_U8_COL,
                    filter_kernels::COMPACT_U32_BY_MASK,
                    filter_kernels::COMPACT_I64_BY_MASK,
                    filter_kernels::COMPACT_F64_BY_MASK,
                    filter_kernels::COMPACT_BYTES_BY_MASK,
                    filter_kernels::MASK_AND,
                    filter_kernels::MASK_OR,
                    filter_kernels::MASK_NOT,
                ],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load filter PTX: {}", e)))?;

        // Load set_ops module
        device
            .inner()
            .load_ptx(
                Ptx::from_src(SET_OPS_PTX),
                SET_OPS_MODULE,
                &[
                    set_ops_kernels::CONCAT_U32,
                    set_ops_kernels::CONCAT_BYTES,
                    set_ops_kernels::SORTED_DIFF_MARK,
                ],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load set_ops PTX: {}", e)))?;

        // Load pack module (GPU-side key packing for multi-column joins)
        device
            .inner()
            .load_ptx(
                Ptx::from_src(PACK_PTX),
                PACK_MODULE,
                &[
                    pack_kernels::PACK_KEYS,
                    pack_kernels::HASH_PACKED_KEYS,
                    pack_kernels::PACK_AND_HASH_KEYS,
                    pack_kernels::PACK_KEYS_ALIGNED,
                    pack_kernels::UNPACK_COLUMN,
                    pack_kernels::GATHER_PACKED_ROWS,
                    pack_kernels::SCATTER_PACKED_ROWS,
                    pack_kernels::COMPARE_PACKED_KEYS,
                ],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load pack PTX: {}", e)))?;

        // Load circuit module (XGCF circuit evaluation)
        device
            .inner()
            .load_ptx(
                Ptx::from_src(CIRCUIT_PTX),
                CIRCUIT_MODULE,
                &[
                    circuit_kernels::XGCF_FORWARD_LEVEL,
                    circuit_kernels::XGCF_BACKWARD_LEVEL_PROPAGATE,
                    circuit_kernels::XGCF_BACKWARD_LEVEL_DECISION_GRAD,
                    circuit_kernels::XGCF_BACKWARD_LEVEL_LIT_GRAD,
                ],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load circuit PTX: {}", e)))?;

        // Load Monte Carlo sampling module (Bernoulli sampler)
        device
            .inner()
            .load_ptx(
                Ptx::from_src(MC_SAMPLE_PTX),
                MC_SAMPLE_MODULE,
                &[mc_sample_kernels::MC_SAMPLE_BERNOULLI],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load MC sample PTX: {}", e)))?;

        // Load arithmetic module
        let arith_module = Ptx::from_src(ARITH_PTX);
        let _arith = device
            .inner()
            .load_ptx(
                arith_module,
                ARITH_MODULE,
                &[
                    arith_kernels::ARITH_BINARY_I64,
                    arith_kernels::ARITH_BINARY_I32,
                    arith_kernels::ARITH_BINARY_U64,
                    arith_kernels::ARITH_BINARY_U32,
                    arith_kernels::ARITH_BINARY_F64,
                    arith_kernels::ARITH_BINARY_F32,
                    arith_kernels::ARITH_ABS_I64,
                    arith_kernels::ARITH_ABS_I32,
                    arith_kernels::ARITH_ABS_F64,
                    arith_kernels::ARITH_ABS_F32,
                    arith_kernels::ARITH_POW_F64,
                    arith_kernels::ARITH_CAST,
                    arith_kernels::ARITH_FILL_CONST_U32,
                    arith_kernels::ARITH_FILL_CONST_U64,
                    arith_kernels::ARITH_FILL_CONST_I64,
                    arith_kernels::ARITH_FILL_CONST_I32,
                    arith_kernels::ARITH_FILL_CONST_F64,
                    arith_kernels::ARITH_FILL_CONST_F32,
                    arith_kernels::ARITH_FILL_CONST_U8,
                    arith_kernels::ARITH_SELECT_I64,
                    arith_kernels::ARITH_SELECT_I32,
                    arith_kernels::ARITH_SELECT_U64,
                    arith_kernels::ARITH_SELECT_U32,
                    arith_kernels::ARITH_SELECT_F64,
                    arith_kernels::ARITH_SELECT_F32,
                ],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load arith PTX: {}", e)))?;

        // Load SAT module (GPU CDCL + on-GPU validation)
        device
            .inner()
            .load_ptx(
                Ptx::from_src(SAT_PTX),
                SAT_MODULE,
                &[
                    sat_kernels::SAT_CDCL_SOLVE,
                    sat_kernels::SAT_CHECK_MODEL,
                    sat_kernels::SAT_PROOF_CHECK,
                    sat_kernels::SAT_ASSERT_STATUS,
                    sat_kernels::SAT_ASSERT_OK,
                    sat_kernels::SAT_XGCF_CNF_COUNTS,
                    sat_kernels::SAT_XGCF_CNF_EMIT,
                    sat_kernels::SAT_XGCF_CNF_CAPTURE_LAST_COUNTS,
                    sat_kernels::SAT_XGCF_CNF_COMPUTE_TOTALS,
                    sat_kernels::SAT_CNF_WRITE_TERMINATOR,
                    sat_kernels::SAT_CNF_COPY_INTO,
                    sat_kernels::SAT_SHIFT_OFFSETS,
                    sat_kernels::SAT_XGCF_WRITE_ROOT_UNIT_CLAUSE,
                    sat_kernels::SAT_NOT_PHI_COUNTS,
                    sat_kernels::SAT_EMIT_NOT_PHI,
                ],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load SAT PTX: {}", e)))?;

        // Load neural fast-path module (AD chain weight fill + gradient scatter)
        device
            .inner()
            .load_ptx(
                Ptx::from_src(NEURAL_PTX),
                NEURAL_MODULE,
                &[
                    neural_kernels::NEURAL_FILL_AD_CHAIN_F32,
                    neural_kernels::NEURAL_SCATTER_AD_CHAIN_GRADS_F32,
                ],
            )
            .map_err(|e| XlogError::Kernel(format!("Failed to load neural PTX: {}", e)))?;

        Ok(Self {
            device,
            memory,
            transfer_tracker: HostTransferTracker::default(),
        })
    }

    /// Get the CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get the GPU memory manager
    pub fn memory(&self) -> &Arc<GpuMemoryManager> {
        &self.memory
    }

    /// Reset tracked host transfer statistics.
    pub fn reset_host_transfer_stats(&self) {
        self.transfer_tracker.reset();
    }

    /// Snapshot tracked host transfer statistics.
    pub fn host_transfer_stats(&self) -> HostTransferStats {
        self.transfer_tracker.snapshot()
    }

    fn dtoh_sync_copy_into_tracked<T: DeviceRepr, Src: DevicePtr<T>>(
        &self,
        src: &Src,
        dst: &mut [T],
    ) -> Result<()> {
        let bytes = std::mem::size_of::<T>()
            .checked_mul(dst.len())
            .ok_or_else(|| XlogError::Kernel("dtoh size overflow".to_string()))?;
        self.transfer_tracker.record_dtoh(bytes as u64);
        self.device
            .inner()
            .dtoh_sync_copy_into(src, dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy from device: {}", e)))
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

    /// Sample independent Bernoulli variables on the GPU.
    ///
    /// Returns a row-major `(sample, var)` matrix as a flat `Vec<u8>` of length
    /// `num_samples * probs.len()`, where each entry is 0/1.
    pub fn sample_bernoulli_matrix(
        &self,
        probs: &[f32],
        num_samples: usize,
        seed: u64,
    ) -> Result<Vec<u8>> {
        if probs.is_empty() || num_samples == 0 {
            return Ok(Vec::new());
        }

        let num_vars_u32: u32 = probs.len().try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "sample_bernoulli_matrix: num_vars {} exceeds u32::MAX",
                probs.len()
            ))
        })?;
        let num_samples_u32: u32 = num_samples.try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "sample_bernoulli_matrix: num_samples {} exceeds u32::MAX",
                num_samples
            ))
        })?;

        let total = probs.len().checked_mul(num_samples).ok_or_else(|| {
            XlogError::Kernel("sample_bernoulli_matrix: size overflow".to_string())
        })?;

        let device = self.device.inner();

        let mut d_probs = self.memory.alloc::<f32>(probs.len())?;
        device
            .htod_sync_copy_into(probs, &mut d_probs)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload Bernoulli probs: {}", e)))?;

        let mut d_out = self.memory.alloc::<u8>(total)?;

        let block_size = 256u32;
        let total_u32: u32 = total.try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "sample_bernoulli_matrix: total {} exceeds u32::MAX",
                total
            ))
        })?;
        let num_blocks = (total_u32 + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let kernel = device
            .get_func(MC_SAMPLE_MODULE, mc_sample_kernels::MC_SAMPLE_BERNOULLI)
            .ok_or_else(|| XlogError::Kernel("mc_sample_bernoulli kernel not found".to_string()))?;

        // SAFETY: mc_sample_bernoulli(out, probs, num_vars, num_samples, seed)
        unsafe {
            kernel
                .clone()
                .launch(config, (&mut d_out, &d_probs, num_vars_u32, num_samples_u32, seed))
        }
        .map_err(|e| XlogError::Kernel(format!("Failed to launch mc_sample_bernoulli: {}", e)))?;

        let mut host: Vec<u8> = vec![0u8; total];
        device
            .dtoh_sync_copy_into(&d_out, &mut host)
            .map_err(|e| XlogError::Kernel(format!("Failed to download Bernoulli samples: {}", e)))?;

        Ok(host)
    }

    /// Perform a hash join between two buffers
    ///
    /// Uses a two-phase hash join:
    /// 1. Build phase: Insert keys from `right` into a hash table
    /// 2. Probe phase: Match keys from `left` against the hash table
    ///
    /// # Arguments
    /// * `left` - The left (probe) buffer
    /// * `right` - The right (build) buffer
    /// * `left_keys` - Column indices for join keys in left buffer
    /// * `right_keys` - Column indices for join keys in right buffer
    ///
    /// # Returns
    /// A buffer containing the joined rows with columns from both inputs
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails
    pub fn hash_join(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
    ) -> Result<CudaBuffer> {
        self.hash_join_with_limit(left, right, left_keys, right_keys, None)
    }

    /// Hash join with configurable maximum output size
    ///
    /// Uses a two-phase hash join:
    /// 1. Build phase: Insert keys from `right` into a hash table
    /// 2. Probe phase: Match keys from `left` against the hash table
    ///
    /// # Arguments
    /// * `left` - The left (probe) buffer
    /// * `right` - The right (build) buffer
    /// * `left_keys` - Column indices for join keys in left buffer
    /// * `right_keys` - Column indices for join keys in right buffer
    /// * `max_output` - Maximum number of output rows (defaults to DEFAULT_JOIN_MAX_OUTPUT)
    ///
    /// # Returns
    /// A buffer containing the joined rows with columns from both inputs
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails
    pub fn hash_join_with_limit(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        max_output: Option<usize>,
    ) -> Result<CudaBuffer> {
        let max_output_limit = max_output.unwrap_or(DEFAULT_JOIN_MAX_OUTPUT);

        // Validate key columns early (even for empty inputs).
        if left_keys.is_empty() || right_keys.is_empty() {
            return Err(XlogError::Kernel(
                "Join requires at least one key column".to_string(),
            ));
        }
        if left_keys.len() != right_keys.len() {
            return Err(XlogError::Kernel(
                "Left and right key columns must have same length".to_string(),
            ));
        }
        for (&left_idx, &right_idx) in left_keys.iter().zip(right_keys.iter()) {
            if left_idx >= left.arity() {
                return Err(XlogError::Kernel(format!(
                    "Left key column index {} out of bounds (arity {})",
                    left_idx,
                    left.arity()
                )));
            }
            if right_idx >= right.arity() {
                return Err(XlogError::Kernel(format!(
                    "Right key column index {} out of bounds (arity {})",
                    right_idx,
                    right.arity()
                )));
            }
        }

        // Natural-join output: all left columns + right non-key columns.
        let right_key_set: std::collections::HashSet<usize> = right_keys.iter().copied().collect();
        let mut result_columns_schema = left.schema().columns.clone();
        for (idx, col) in right.schema().columns.iter().enumerate() {
            if !right_key_set.contains(&idx) {
                result_columns_schema.push(col.clone());
            }
        }
        let result_schema = Schema::new(result_columns_schema);

        // Handle empty inputs
        if left.is_empty() || right.is_empty() {
            return self.create_empty_buffer(result_schema);
        }

        // Delegate to the v2 implementation for correctness across key types and cardinalities.
        let combined = self.hash_join_v2_with_limit(
            left,
            right,
            left_keys,
            right_keys,
            JoinType::Inner,
            Some(max_output_limit),
        )?;

        if combined.is_empty() {
            return self.create_empty_buffer(result_schema);
        }

        let left_arity = left.arity();
        let right_arity = right.arity();

        let CudaBuffer {
            columns: combined_columns,
            num_rows,
            schema: _,
        } = combined;

        if combined_columns.len() != left_arity + right_arity {
            return Err(XlogError::Kernel(format!(
                "Join internal error: expected {} columns, got {}",
                left_arity + right_arity,
                combined_columns.len()
            )));
        }

        let mut output_columns = Vec::with_capacity(result_schema.arity());
        let mut it = combined_columns.into_iter();

        // Left columns (all preserved)
        for _ in 0..left_arity {
            let col = it.next().ok_or_else(|| {
                XlogError::Kernel("Join internal error: missing left columns".to_string())
            })?;
            output_columns.push(col);
        }

        // Right columns, excluding join keys
        for (right_col_idx, col) in it.enumerate() {
            if !right_key_set.contains(&right_col_idx) {
                output_columns.push(col);
            }
        }

        Ok(CudaBuffer::from_columns(
            output_columns,
            num_rows,
            result_schema,
        ))
    }

    /// Remove duplicate rows based on key columns
    ///
    /// Sorts the input by the provided key columns, then removes adjacent duplicates.
    ///
    /// # Arguments
    /// * `input` - The input buffer
    /// * `key_cols` - Column indices to use for duplicate detection
    ///
    /// # Returns
    /// A buffer with duplicate rows removed
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails
    pub fn dedup(&self, input: &CudaBuffer, key_cols: &[usize]) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema().clone());
        }

        if key_cols.is_empty() {
            return Err(XlogError::Kernel(
                "Dedup requires at least one key column".to_string(),
            ));
        }

        let sorted = self.sort(input, key_cols)?;
        self.dedup_sorted(&sorted, key_cols)
    }

    /// Remove duplicate rows from a buffer that is already sorted by key columns
    ///
    /// This is an optimized version of `dedup` that skips the sorting step.
    /// The caller must ensure the input is already sorted by the key columns.
    ///
    /// # Arguments
    /// * `input` - The input buffer (must be sorted by key columns)
    /// * `key_cols` - Column indices to use for duplicate detection
    ///
    /// # Returns
    /// A buffer with duplicate rows removed
    pub fn dedup_sorted(&self, input: &CudaBuffer, key_cols: &[usize]) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema().clone());
        }

        if key_cols.is_empty() {
            return Err(XlogError::Kernel(
                "Dedup requires at least one key column".to_string(),
            ));
        }

        if input.num_rows() <= 1 {
            return self.clone_buffer(input);
        }

        if input.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Dedup supports at most {} rows, got {}",
                u32::MAX,
                input.num_rows()
            )));
        }

        fn scalar_type_code(ty: ScalarType) -> u8 {
            match ty {
                ScalarType::U32 => 0,
                ScalarType::U64 => 1,
                ScalarType::I32 => 2,
                ScalarType::I64 => 3,
                ScalarType::F32 => 4,
                ScalarType::F64 => 5,
                ScalarType::Bool => 6,
                ScalarType::Symbol => 7,
            }
        }

        let device = self.device.inner();
        let num_rows = input.num_rows() as u32;

        let mut col_ptrs_host: Vec<u64> = Vec::with_capacity(key_cols.len());
        let mut col_sizes_host: Vec<u32> = Vec::with_capacity(key_cols.len());
        let mut col_types_host: Vec<u8> = Vec::with_capacity(key_cols.len());

        for &key_col in key_cols {
            if key_col >= input.arity() {
                return Err(XlogError::Kernel(format!(
                    "Key column {} out of bounds (arity {})",
                    key_col,
                    input.arity()
                )));
            }

            let col = input
                .column(key_col)
                .ok_or_else(|| XlogError::Kernel(format!("Key column {} not found", key_col)))?;
            let ty = input.schema().column_type(key_col).ok_or_else(|| {
                XlogError::Kernel(format!("Key column {} type not found in schema", key_col))
            })?;

            let elem_size = ty.size_bytes();
            let expected_bytes = (num_rows as usize) * elem_size;
            if col.num_bytes() != expected_bytes {
                return Err(XlogError::Kernel(format!(
                    "Key column {} has {} bytes but expected {} (num_rows={}, elem_size={})",
                    key_col,
                    col.num_bytes(),
                    expected_bytes,
                    num_rows,
                    elem_size
                )));
            }

            let ptr = *cudarc::driver::DevicePtr::device_ptr(col) as u64;
            col_ptrs_host.push(ptr);
            col_sizes_host.push(elem_size as u32);
            col_types_host.push(scalar_type_code(ty));
        }

        let num_key_cols = key_cols.len() as u32;
        let mut d_col_ptrs = self.memory.alloc::<u64>(key_cols.len())?;
        let mut d_col_sizes = self.memory.alloc::<u32>(key_cols.len())?;
        let mut d_col_types = self.memory.alloc::<u8>(key_cols.len())?;

        device
            .htod_sync_copy_into(&col_ptrs_host, &mut d_col_ptrs)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload key column ptrs: {}", e)))?;
        device
            .htod_sync_copy_into(&col_sizes_host, &mut d_col_sizes)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload key column sizes: {}", e)))?;
        device
            .htod_sync_copy_into(&col_types_host, &mut d_col_types)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload key column types: {}", e)))?;

        let block_size = 256u32;
        let num_blocks = (num_rows + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let d_unique_mask = self.memory.alloc::<u8>(num_rows as usize)?;
        let d_prefix_sum = self.memory.alloc::<u32>(num_rows as usize)?;
        let mut d_block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

        let mark_and_scan_fn = device
            .get_func(DEDUP_MODULE, dedup_kernels::MARK_UNIQUE_AND_SCAN_COLUMNAR)
            .ok_or_else(|| {
                XlogError::Kernel("mark_unique_and_scan_columnar kernel not found".to_string())
            })?;

        // SAFETY: mark_unique_and_scan_columnar(col_ptrs, col_sizes, col_types, num_key_cols, num_rows, unique_mask, prefix_sum, block_sums)
        unsafe {
            mark_and_scan_fn.clone().launch(
                config,
                (
                    &d_col_ptrs,
                    &d_col_sizes,
                    &d_col_types,
                    num_key_cols,
                    num_rows,
                    &d_unique_mask,
                    &d_prefix_sum,
                    &d_block_sums,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("mark_unique_and_scan_columnar failed: {}", e)))?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace(&mut d_block_sums, num_blocks)?;

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
                    (&d_prefix_sum, &d_block_sums, num_rows),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        let last_idx = (num_rows as usize)
            .checked_sub(1)
            .ok_or_else(|| XlogError::Kernel("Dedup: unexpected empty input".to_string()))?;
        let last_prefix_view = d_prefix_sum.slice(last_idx..(last_idx + 1));
        let last_mask_view = d_unique_mask.slice(last_idx..(last_idx + 1));

        let mut last_prefix_host = [0u32];
        let mut last_mask_host = [0u8];
        device
            .dtoh_sync_copy_into(&last_prefix_view, &mut last_prefix_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read last prefix sum: {}", e)))?;
        device
            .dtoh_sync_copy_into(&last_mask_view, &mut last_mask_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read last unique mask: {}", e)))?;

        let output_count = (last_prefix_host[0] as u64) + (last_mask_host[0] as u64);

        if output_count == 0 {
            return self.create_empty_buffer(input.schema().clone());
        }

        self.compact_buffer_by_device_mask(input, &d_unique_mask, &d_prefix_sum, output_count)
    }

    /// Compute union of two buffers
    ///
    /// MVP: Simple concatenation without deduplication.
    /// A full implementation would dedup the result.
    ///
    /// # Arguments
    /// * `a` - First buffer
    /// * `b` - Second buffer
    ///
    /// # Returns
    /// A buffer containing all rows from both inputs
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if schemas don't match or operation fails
    pub fn union(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        // Handle empty cases
        if a.is_empty() && b.is_empty() {
            return self.create_empty_buffer(a.schema().clone());
        }
        if a.is_empty() {
            return self.clone_buffer(b);
        }
        if b.is_empty() {
            return self.clone_buffer(a);
        }

        // Verify schemas have compatible types (ignore column names for Datalog union)
        if !self.schemas_type_compatible(a.schema(), b.schema()) {
            return Err(XlogError::Kernel(format!(
                "Union requires compatible schemas: {:?} vs {:?}",
                a.schema(),
                b.schema()
            )));
        }

        self.concat_buffers_gpu(a, b)
    }

    fn concat_buffers_gpu(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        if a.is_empty() && b.is_empty() {
            return self.create_empty_buffer(a.schema().clone());
        }
        if a.is_empty() {
            return self.clone_buffer(b);
        }
        if b.is_empty() {
            return self.clone_buffer(a);
        }

        if !self.schemas_type_compatible(a.schema(), b.schema()) {
            return Err(XlogError::Kernel(format!(
                "Concat requires compatible schemas: {:?} vs {:?}",
                a.schema(),
                b.schema()
            )));
        }

        let schema = a.schema().clone();
        let total_rows = a.num_rows() + b.num_rows();
        if total_rows > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Concat supports at most {} rows, got {}",
                u32::MAX,
                total_rows
            )));
        }

        let device = self.device.inner();
        let concat_fn = device
            .get_func(SET_OPS_MODULE, set_ops_kernels::CONCAT_BYTES)
            .ok_or_else(|| XlogError::Kernel("concat_bytes kernel not found".to_string()))?;

        let block_size = 256u32;

        let a_rows = usize::try_from(a.num_rows()).map_err(|_| {
            XlogError::Kernel(format!("Concat: a has too many rows: {}", a.num_rows()))
        })?;
        let b_rows = usize::try_from(b.num_rows()).map_err(|_| {
            XlogError::Kernel(format!("Concat: b has too many rows: {}", b.num_rows()))
        })?;

        let mut result_columns = Vec::with_capacity(schema.arity());
        for col_idx in 0..schema.arity() {
            let elem_size = schema
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            let a_bytes = a_rows
                .checked_mul(elem_size)
                .ok_or_else(|| XlogError::Kernel("Concat: a_bytes overflow".to_string()))?;
            let b_bytes = b_rows
                .checked_mul(elem_size)
                .ok_or_else(|| XlogError::Kernel("Concat: b_bytes overflow".to_string()))?;
            let total_bytes = a_bytes
                .checked_add(b_bytes)
                .ok_or_else(|| XlogError::Kernel("Concat: total_bytes overflow".to_string()))?;

            let a_bytes_u32 = u32::try_from(a_bytes).map_err(|_| {
                XlogError::Kernel(format!("Concat: a_bytes too large: {}", a_bytes))
            })?;
            let b_bytes_u32 = u32::try_from(b_bytes).map_err(|_| {
                XlogError::Kernel(format!("Concat: b_bytes too large: {}", b_bytes))
            })?;
            let total_bytes_u32 = u32::try_from(total_bytes).map_err(|_| {
                XlogError::Kernel(format!("Concat: total_bytes too large: {}", total_bytes))
            })?;

            let a_col = a
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("A column {} not found", col_idx)))?;
            let b_col = b
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("B column {} not found", col_idx)))?;

            let mut out_col = self.memory.alloc::<u8>(total_bytes)?;

            if total_bytes_u32 > 0 {
                let grid_size = (total_bytes_u32 + block_size - 1) / block_size;
                let config = LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };

                // SAFETY: concat_bytes(const uint8_t* a, uint32_t a_bytes, const uint8_t* b, uint32_t b_bytes, uint8_t* output)
                unsafe {
                    concat_fn.clone().launch(
                        config,
                        (a_col, a_bytes_u32, b_col, b_bytes_u32, &mut out_col),
                    )
                }
                .map_err(|e| XlogError::Kernel(format!("concat_bytes failed: {}", e)))?;
            }

            result_columns.push(out_col.into());
        }

        self.device.synchronize()?;

        Ok(CudaBuffer::from_columns(result_columns, total_rows, schema))
    }

    /// Compute set difference (a - b)
    ///
    /// Returns rows from `a` that don't exist in `b`.
    /// Uses hash-based approach: build hash table from b, probe with a.
    ///
    /// # Arguments
    /// * `a` - Source buffer
    /// * `b` - Buffer to subtract
    ///
    /// # Returns
    /// A buffer containing rows in `a` but not in `b`
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if schemas don't match or operation fails
    pub fn diff(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        // Handle empty cases
        if a.is_empty() {
            return self.create_empty_buffer(a.schema().clone());
        }
        if b.is_empty() {
            return self.clone_buffer(a);
        }

        // Verify schemas have compatible types (ignore column names for Datalog negation)
        if !self.schemas_type_compatible(a.schema(), b.schema()) {
            return Err(XlogError::Kernel(format!(
                "Diff requires compatible schemas: {:?} vs {:?}",
                a.schema(),
                b.schema()
            )));
        }

        // Use first column as key for hash-based diff
        if a.arity() == 0 {
            return Err(XlogError::Kernel(
                "Diff requires at least one column".to_string(),
            ));
        }

        let num_b = b.num_rows() as u32;
        let num_a = a.num_rows() as u32;

        // Build hash table from b
        let hash_table_size = (num_b as usize * 2).max(1024) as u32;
        let hash_table_alloc_size = (hash_table_size * 3) as usize;
        let mut hash_table = self.memory.alloc::<u32>(hash_table_alloc_size)?;
        let mut next_ptrs = self.memory.alloc::<u32>(num_b as usize)?;

        // Initialize all hash table entries
        let init_val = 0xFFFFFFFFu32;
        self.device
            .inner()
            .htod_sync_copy_into(&vec![init_val; hash_table_alloc_size], &mut hash_table)
            .map_err(|e| XlogError::Kernel(format!("Failed to init hash table: {}", e)))?;
        self.device
            .inner()
            .htod_sync_copy_into(&vec![init_val; num_b as usize], &mut next_ptrs)
            .map_err(|e| XlogError::Kernel(format!("Failed to init next pointers: {}", e)))?;

        // Build phase with b's keys using transmute for direct GPU access
        let build_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_BUILD)
            .ok_or_else(|| XlogError::Kernel("hash_join_build kernel not found".to_string()))?;

        let b_key_col = b
            .column(0)
            .ok_or_else(|| XlogError::Kernel("B key column not found".to_string()))?;
        let b_keys_view = self.column_as_u32_view(b_key_col, num_b as usize)?;

        let block_size = 256u32;
        let build_grid = (num_b + block_size - 1) / block_size;
        let build_config = LaunchConfig {
            grid_dim: (build_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Kernel parameters match expected signature
        unsafe {
            build_func
                .clone()
                .launch(
                    build_config,
                    (
                        &b_keys_view,
                        &b_keys_view, // payload = key for diff
                        num_b,
                        &hash_table,
                        &next_ptrs,
                        hash_table_size,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("Build kernel failed: {}", e)))?;
        }

        // Synchronize and build lookup set for filtering
        self.device.synchronize()?;

        // Get a's keys using transmute
        let a_key_col = a
            .column(0)
            .ok_or_else(|| XlogError::Kernel("A key column not found".to_string()))?;

        // Read keys to host for filtering (set difference requires iterating)
        let mut a_keys_host = vec![0u8; (num_a as usize) * 4];
        self.device
            .inner()
            .dtoh_sync_copy_into(a_key_col, &mut a_keys_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read a keys: {}", e)))?;

        let mut b_keys_host = vec![0u8; (num_b as usize) * 4];
        self.device
            .inner()
            .dtoh_sync_copy_into(b_key_col, &mut b_keys_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read b keys: {}", e)))?;

        // Build lookup set from b
        let b_keys_set: std::collections::HashSet<u32> = b_keys_host
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Find indices of a rows not in b
        let diff_indices: Vec<usize> = a_keys_host
            .chunks_exact(4)
            .enumerate()
            .map(|(i, chunk)| {
                (
                    i,
                    u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                )
            })
            .filter(|(_, k)| !b_keys_set.contains(k))
            .map(|(i, _)| i)
            .collect();

        let diff_count = diff_indices.len() as u64;

        if diff_count == 0 {
            return self.create_empty_buffer(a.schema().clone());
        }

        // Build result by selecting rows
        let schema = a.schema().clone();
        let mut result_columns = Vec::with_capacity(schema.arity());

        for col_idx in 0..schema.arity() {
            let col_type_size = schema
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let result_bytes = (diff_count as usize) * col_type_size;

            if let Some(a_col) = a.column(col_idx) {
                // Read column data
                let a_col_bytes = (num_a as usize) * col_type_size;
                let mut a_col_host = vec![0u8; a_col_bytes];
                self.device
                    .inner()
                    .dtoh_sync_copy_into(a_col, &mut a_col_host)
                    .map_err(|e| XlogError::Kernel(format!("Failed to read column: {}", e)))?;

                // Select rows matching diff indices
                let mut result_host = Vec::with_capacity(result_bytes);
                for &idx in &diff_indices {
                    let start = idx * col_type_size;
                    let end = start + col_type_size;
                    result_host.extend_from_slice(&a_col_host[start..end]);
                }

                // Upload result
                let mut result_col = self.memory.alloc::<u8>(result_bytes)?;
                self.device
                    .inner()
                    .htod_sync_copy_into(&result_host, &mut result_col)
                    .map_err(|e| XlogError::Kernel(format!("Failed to upload result: {}", e)))?;

                result_columns.push(result_col.into());
            }
        }

        Ok(CudaBuffer::from_columns(result_columns, diff_count, schema))
    }

    // ============== GPU-Native Set Operations ==============

    /// GPU-native union (no host roundtrip)
    ///
    /// Computes the union of two buffers entirely on the GPU using:
    /// 1. Concatenate arrays using concat_u32 kernel
    /// 2. Sort the concatenated result
    /// 3. Deduplicate using existing dedup()
    ///
    /// # Arguments
    /// * `a` - First buffer
    /// * `b` - Second buffer
    ///
    /// # Returns
    /// A buffer containing deduplicated union of both inputs, sorted
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if schemas don't match or operation fails
    pub fn union_gpu(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        if a.is_empty() && b.is_empty() {
            return self.create_empty_buffer(a.schema().clone());
        }

        // Verify schemas have compatible types (ignore column names for Datalog union).
        if !self.schemas_type_compatible(a.schema(), b.schema()) {
            return Err(XlogError::Kernel(format!(
                "Union requires compatible schemas: {:?} vs {:?}",
                a.schema(),
                b.schema()
            )));
        }

        let schema = a.schema().clone();
        let key_cols: Vec<usize> = (0..schema.arity()).collect();
        if key_cols.is_empty() {
            return Err(XlogError::Kernel(
                "Union requires at least one column".to_string(),
            ));
        }

        // Set semantics require dedup even when one side is empty.
        if a.is_empty() {
            return self.dedup(b, &key_cols);
        }
        if b.is_empty() {
            return self.dedup(a, &key_cols);
        }

        let concat = self.concat_buffers_gpu(a, b)?;
        let sorted = self.sort(&concat, &key_cols)?;
        self.dedup_sorted(&sorted, &key_cols)
    }

    /// GPU-native set difference (no host roundtrip)
    ///
    /// Computes a - b (elements in a but not in b) entirely on GPU using:
    /// 1. Sort both inputs
    /// 2. Dedup both inputs
    /// 3. Mark elements in a not in b using sorted_diff_mark kernel (binary search)
    /// 4. Filter using existing filter_by_mask()
    ///
    /// # Arguments
    /// * `a` - Source buffer
    /// * `b` - Buffer to subtract
    ///
    /// # Returns
    /// A buffer containing elements in a but not in b, sorted and deduped
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if schemas don't match or operation fails
    pub fn diff_gpu(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        if a.is_empty() {
            return self.create_empty_buffer(a.schema().clone());
        }

        // Verify schemas have compatible types (ignore column names for Datalog negation)
        if !self.schemas_type_compatible(a.schema(), b.schema()) {
            return Err(XlogError::Kernel(format!(
                "Diff requires compatible schemas: {:?} vs {:?}",
                a.schema(),
                b.schema()
            )));
        }

        if a.arity() == 0 {
            return Err(XlogError::Kernel(
                "Diff requires at least one column".to_string(),
            ));
        }

        let col_type = a
            .schema()
            .column_type(0)
            .ok_or_else(|| XlogError::Kernel("No columns".to_string()))?;

        // Keep the single-column U32 fast path; all other cases use hash-based anti-join.
        if a.arity() == 1 && matches!(col_type, ScalarType::U32) && !b.is_empty() {
            return self.diff_gpu_u32(a, b);
        }

        self.diff_via_anti_join(a, b)
    }

    /// U32-optimized diff using GPU sort
    fn diff_gpu_u32(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        // For multi-column U32 buffers, use anti-join on all columns
        if a.arity() != 1 {
            return self.diff_via_anti_join(a, b);
        }

        // Step 1: Sort and dedup both inputs (sorted, so use dedup_sorted)
        let sorted_a = self.sort(a, &[0])?;
        let deduped_a = self.dedup_sorted(&sorted_a, &[0])?;

        let sorted_b = self.sort(b, &[0])?;
        let deduped_b = self.dedup_sorted(&sorted_b, &[0])?;

        if deduped_a.is_empty() {
            return self.create_empty_buffer(a.schema().clone());
        }

        let num_a = deduped_a.num_rows() as u32;
        let num_b = deduped_b.num_rows() as u32;

        // Step 2: Mark elements in a not in b using sorted_diff_mark kernel
        let diff_mark_fn = self
            .device
            .inner()
            .get_func(SET_OPS_MODULE, set_ops_kernels::SORTED_DIFF_MARK)
            .ok_or_else(|| XlogError::Kernel("sorted_diff_mark kernel not found".to_string()))?;

        // Get column data as u32 views
        let a_col = deduped_a
            .column(0)
            .ok_or_else(|| XlogError::Kernel("A column 0 not found".to_string()))?;
        let b_col = deduped_b
            .column(0)
            .ok_or_else(|| XlogError::Kernel("B column 0 not found".to_string()))?;

        let a_view = self.column_as_u32_view(a_col, num_a as usize)?;
        let b_view = self.column_as_u32_view(b_col, num_b as usize)?;

        // Allocate mask for diff marking
        let diff_mask = self.memory.alloc::<u8>(num_a as usize)?;

        // Launch diff mark kernel
        let block_size = 256u32;
        let grid_size = (num_a + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Kernel signature matches: sorted_diff_mark(a, a_len, b, b_len, in_diff)
        unsafe {
            diff_mark_fn
                .clone()
                .launch(config, (&a_view, num_a, &b_view, num_b, &diff_mask))
        }
        .map_err(|e| XlogError::Kernel(format!("sorted_diff_mark failed: {}", e)))?;

        // Compute prefix sum of diff mask on GPU.
        let device = self.device.inner();
        let num_blocks = grid_size;
        let d_prefix_sum = self.memory.alloc::<u32>(num_a as usize)?;
        let mut d_block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

        let phase1_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase1 kernel".to_string())
            })?;

        // SAFETY: multiblock_scan_phase1(const uint8_t* mask, uint32_t* prefix_sum, uint32_t* block_sums, uint32_t n)
        unsafe {
            phase1_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&diff_mask, &d_prefix_sum, &d_block_sums, num_a),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase1 failed: {}", e)))?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace(&mut d_block_sums, num_blocks)?;

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
                    (&d_prefix_sum, &d_block_sums, num_a),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        let last_idx = (num_a as usize)
            .checked_sub(1)
            .ok_or_else(|| XlogError::Kernel("Diff: unexpected empty input".to_string()))?;

        let last_prefix_view = d_prefix_sum.slice(last_idx..(last_idx + 1));
        let last_mask_view = diff_mask.slice(last_idx..(last_idx + 1));

        let mut last_prefix_host = [0u32];
        let mut last_mask_host = [0u8];
        device
            .dtoh_sync_copy_into(&last_prefix_view, &mut last_prefix_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read last prefix sum: {}", e)))?;
        device
            .dtoh_sync_copy_into(&last_mask_view, &mut last_mask_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read last diff mask: {}", e)))?;

        let output_count = (last_prefix_host[0] as u64) + (last_mask_host[0] as u64);

        if output_count == 0 {
            return self.create_empty_buffer(a.schema().clone());
        }

        self.compact_buffer_by_device_mask(&deduped_a, &diff_mask, &d_prefix_sum, output_count)
    }

    /// Multi-column diff using anti-join
    ///
    /// Uses hash-based anti-join on all columns to compute set difference.
    /// This is the appropriate algorithm for multi-column Datalog negation.
    fn diff_via_anti_join(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        // Use all columns as join keys for set difference
        let all_cols: Vec<usize> = (0..a.arity()).collect();

        // Dedup both inputs first (set difference requires set semantics)
        let deduped_a = self.dedup(a, &all_cols)?;
        let deduped_b = self.dedup(b, &all_cols)?;

        if deduped_a.is_empty() {
            return self.create_empty_buffer(a.schema().clone());
        }

        if deduped_b.is_empty() {
            return Ok(deduped_a);
        }

        // Anti-join: return rows from a that don't match any row in b
        // Since we're doing set difference, all columns are keys
        self.hash_join_v2(&deduped_a, &deduped_b, &all_cols, &all_cols, JoinType::Anti)
    }

    /// Perform groupby aggregation
    ///
    /// Assumes input is sorted by key columns (sorting is complex, deferred).
    ///
    /// # Arguments
    /// * `input` - The input buffer
    /// * `key_cols` - Column indices for grouping
    /// * `agg` - Aggregation operation to perform
    /// * `value_col` - Column index for the value to aggregate
    ///
    /// # Returns
    /// A buffer with one row per group, containing key columns and aggregated value
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails
    pub fn groupby_agg(
        &self,
        input: &CudaBuffer,
        key_cols: &[usize],
        agg: AggOp,
        value_col: usize,
    ) -> Result<CudaBuffer> {
        if input.is_empty() {
            // Return empty buffer with appropriate schema
            let result_schema = self.groupby_result_schema(input.schema(), key_cols, agg);
            return self.create_empty_buffer(result_schema);
        }

        if key_cols.is_empty() {
            return Err(XlogError::Kernel(
                "GroupBy requires at least one key column".to_string(),
            ));
        }

        // Verify value column exists
        if value_col >= input.arity() {
            return Err(XlogError::Kernel(format!(
                "Value column {} out of bounds",
                value_col
            )));
        }

        let num_rows = input.num_rows() as u32;
        let num_key_cols = key_cols.len() as u32;
        // For column-based storage with single key column, row_stride = 1
        // since we pass a single contiguous column view
        let row_stride = 1u32;

        // Get boundary detection kernel
        let boundary_func = self
            .device
            .inner()
            .get_func(GROUPBY_MODULE, groupby_kernels::DETECT_GROUP_BOUNDARIES)
            .ok_or_else(|| {
                XlogError::Kernel("detect_group_boundaries kernel not found".to_string())
            })?;

        // Allocate is_boundary mask
        let is_boundary = self.memory.alloc::<u8>(num_rows as usize)?;

        // Prepare sorted keys buffer (MVP: use first key column)
        let sorted_keys = input
            .column(key_cols[0])
            .ok_or_else(|| XlogError::Kernel("Key column not found".to_string()))?;
        let sorted_keys_view = self.column_as_u32_view(sorted_keys, num_rows as usize)?;

        // Launch boundary detection
        let block_size = 256u32;
        let grid_size = (num_rows + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Kernel parameters match expected signature
        unsafe {
            boundary_func
                .clone()
                .launch(
                    config,
                    (
                        &sorted_keys_view,
                        num_rows,
                        num_key_cols,
                        row_stride,
                        &is_boundary,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("detect_group_boundaries failed: {}", e)))?;
        }

        // Compute group IDs from boundaries (CPU for MVP)
        self.device.synchronize()?;

        let mut boundary_host = vec![0u8; num_rows as usize];
        self.device
            .inner()
            .dtoh_sync_copy_into(&is_boundary, &mut boundary_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read boundaries: {}", e)))?;

        // Compute group IDs: cumulative sum of boundaries - 1
        let mut group_ids_host = vec![0u32; num_rows as usize];
        let mut current_group = 0u32;
        for i in 0..num_rows as usize {
            if i > 0 && boundary_host[i] == 1 {
                current_group += 1;
            }
            group_ids_host[i] = current_group;
        }
        let num_groups = current_group + 1;

        // Upload group IDs
        let mut group_ids = self.memory.alloc::<u32>(num_rows as usize)?;
        self.device
            .inner()
            .htod_sync_copy_into(&group_ids_host, &mut group_ids)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload group IDs: {}", e)))?;

        // Get value column
        let values = input
            .column(value_col)
            .ok_or_else(|| XlogError::Kernel("Value column not found".to_string()))?;

        // Allocate and initialize output based on aggregation type
        // Return output as bytes via host for consistent handling
        let (output_bytes, _result_type): (Vec<u8>, ScalarType) = match agg {
            AggOp::Count => {
                let mut output = self.memory.alloc::<u64>(num_groups as usize)?;
                // Initialize to 0
                self.device
                    .inner()
                    .htod_sync_copy_into(&vec![0u64; num_groups as usize], &mut output)
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to init count output: {}", e))
                    })?;

                let count_func = self
                    .device
                    .inner()
                    .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_COUNT)
                    .ok_or_else(|| {
                        XlogError::Kernel("groupby_count kernel not found".to_string())
                    })?;

                // SAFETY: Kernel parameters match expected signature
                unsafe {
                    count_func
                        .clone()
                        .launch(config, (&is_boundary, &group_ids, num_rows, &output))
                        .map_err(|e| XlogError::Kernel(format!("groupby_count failed: {}", e)))?;
                }

                self.device.synchronize()?;

                // Read back as bytes
                let mut host_output = vec![0u64; num_groups as usize];
                self.device
                    .inner()
                    .dtoh_sync_copy_into(&output, &mut host_output)
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to read count output: {}", e))
                    })?;
                let bytes: Vec<u8> = host_output.iter().flat_map(|v| v.to_le_bytes()).collect();
                (bytes, ScalarType::U64)
            }
            AggOp::Sum => {
                let mut output = self.memory.alloc::<u64>(num_groups as usize)?;
                // Initialize to 0
                self.device
                    .inner()
                    .htod_sync_copy_into(&vec![0u64; num_groups as usize], &mut output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to init sum output: {}", e)))?;

                let sum_func = self
                    .device
                    .inner()
                    .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_SUM)
                    .ok_or_else(|| XlogError::Kernel("groupby_sum kernel not found".to_string()))?;

                // SAFETY: Kernel parameters match expected signature
                unsafe {
                    sum_func
                        .clone()
                        .launch(config, (values, &group_ids, num_rows, &output))
                        .map_err(|e| XlogError::Kernel(format!("groupby_sum failed: {}", e)))?;
                }

                self.device.synchronize()?;

                // Read back as bytes
                let mut host_output = vec![0u64; num_groups as usize];
                self.device
                    .inner()
                    .dtoh_sync_copy_into(&output, &mut host_output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to read sum output: {}", e)))?;
                let bytes: Vec<u8> = host_output.iter().flat_map(|v| v.to_le_bytes()).collect();
                (bytes, ScalarType::U64)
            }
            AggOp::Min => {
                let mut output = self.memory.alloc::<u32>(num_groups as usize)?;
                // Initialize to MAX
                self.device
                    .inner()
                    .htod_sync_copy_into(&vec![u32::MAX; num_groups as usize], &mut output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to init min output: {}", e)))?;

                let min_func = self
                    .device
                    .inner()
                    .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MIN)
                    .ok_or_else(|| XlogError::Kernel("groupby_min kernel not found".to_string()))?;

                // SAFETY: Kernel parameters match expected signature
                unsafe {
                    min_func
                        .clone()
                        .launch(config, (values, &group_ids, num_rows, &output))
                        .map_err(|e| XlogError::Kernel(format!("groupby_min failed: {}", e)))?;
                }

                self.device.synchronize()?;

                // Read back as bytes
                let mut host_output = vec![0u32; num_groups as usize];
                self.device
                    .inner()
                    .dtoh_sync_copy_into(&output, &mut host_output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to read min output: {}", e)))?;
                let bytes: Vec<u8> = host_output.iter().flat_map(|v| v.to_le_bytes()).collect();
                (bytes, ScalarType::U32)
            }
            AggOp::Max => {
                let mut output = self.memory.alloc::<u32>(num_groups as usize)?;
                // Initialize to 0
                self.device
                    .inner()
                    .htod_sync_copy_into(&vec![0u32; num_groups as usize], &mut output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to init max output: {}", e)))?;

                let max_func = self
                    .device
                    .inner()
                    .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MAX)
                    .ok_or_else(|| XlogError::Kernel("groupby_max kernel not found".to_string()))?;

                // SAFETY: Kernel parameters match expected signature
                unsafe {
                    max_func
                        .clone()
                        .launch(config, (values, &group_ids, num_rows, &output))
                        .map_err(|e| XlogError::Kernel(format!("groupby_max failed: {}", e)))?;
                }

                self.device.synchronize()?;

                // Read back as bytes
                let mut host_output = vec![0u32; num_groups as usize];
                self.device
                    .inner()
                    .dtoh_sync_copy_into(&output, &mut host_output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to read max output: {}", e)))?;
                let bytes: Vec<u8> = host_output.iter().flat_map(|v| v.to_le_bytes()).collect();
                (bytes, ScalarType::U32)
            }
            AggOp::LogSumExp => {
                // LogSumExp requires F64 values, use 3-pass algorithm:
                // 1. Find max per group
                // 2. Compute sum of exp(x - max) per group
                // 3. Compute log(sum) + max per group

                // Get values as f64 view
                let values_f64 = self.column_as_f64_view(values, num_rows as usize)?;

                // Allocate buffers for intermediate results
                let mut maxs = self.memory.alloc::<f64>(num_groups as usize)?;
                let mut sumexps = self.memory.alloc::<f64>(num_groups as usize)?;
                let results = self.memory.alloc::<f64>(num_groups as usize)?;

                // Initialize maxs to -INFINITY
                self.device
                    .inner()
                    .htod_sync_copy_into(&vec![f64::NEG_INFINITY; num_groups as usize], &mut maxs)
                    .map_err(|e| XlogError::Kernel(format!("Failed to init maxs: {}", e)))?;

                // Initialize sumexps to 0.0
                self.device
                    .inner()
                    .htod_sync_copy_into(&vec![0.0f64; num_groups as usize], &mut sumexps)
                    .map_err(|e| XlogError::Kernel(format!("Failed to init sumexps: {}", e)))?;

                // Pass 1: Find max per group
                let max_func = self
                    .device
                    .inner()
                    .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_LOGSUMEXP_MAX)
                    .ok_or_else(|| {
                        XlogError::Kernel("groupby_logsumexp_max kernel not found".to_string())
                    })?;

                // SAFETY: Kernel parameters match expected signature
                unsafe {
                    max_func
                        .clone()
                        .launch(config, (&values_f64, &group_ids, num_rows, &maxs))
                        .map_err(|e| {
                            XlogError::Kernel(format!("groupby_logsumexp_max failed: {}", e))
                        })?;
                }
                self.device.synchronize()?;

                // Pass 2: Compute sum of exp(x - max) per group
                let sumexp_func = self
                    .device
                    .inner()
                    .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_LOGSUMEXP_SUMEXP)
                    .ok_or_else(|| {
                        XlogError::Kernel("groupby_logsumexp_sumexp kernel not found".to_string())
                    })?;

                // SAFETY: Kernel parameters match expected signature
                unsafe {
                    sumexp_func
                        .clone()
                        .launch(config, (&values_f64, &group_ids, &maxs, num_rows, &sumexps))
                        .map_err(|e| {
                            XlogError::Kernel(format!("groupby_logsumexp_sumexp failed: {}", e))
                        })?;
                }
                self.device.synchronize()?;

                // Pass 3: Compute final result = max + log(sumexp)
                let final_config = LaunchConfig::for_num_elems(num_groups);
                let final_func = self
                    .device
                    .inner()
                    .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_LOGSUMEXP_FINAL)
                    .ok_or_else(|| {
                        XlogError::Kernel("groupby_logsumexp_final kernel not found".to_string())
                    })?;

                // SAFETY: Kernel parameters match expected signature
                unsafe {
                    final_func
                        .clone()
                        .launch(final_config, (&maxs, &sumexps, num_groups, &results))
                        .map_err(|e| {
                            XlogError::Kernel(format!("groupby_logsumexp_final failed: {}", e))
                        })?;
                }
                self.device.synchronize()?;

                // Read back as bytes
                let mut host_output = vec![0.0f64; num_groups as usize];
                self.device
                    .inner()
                    .dtoh_sync_copy_into(&results, &mut host_output)
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to read logsumexp output: {}", e))
                    })?;
                let bytes: Vec<u8> = host_output.iter().flat_map(|v| v.to_le_bytes()).collect();
                (bytes, ScalarType::F64)
            }
        };

        self.device.synchronize()?;

        // Extract first key value for each group (CPU for MVP)
        let mut key_col_host = vec![0u8; (num_rows as usize) * 4];
        self.device
            .inner()
            .dtoh_sync_copy_into(sorted_keys, &mut key_col_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read keys: {}", e)))?;

        // Get first row index of each group
        let mut group_first_indices = vec![0usize; num_groups as usize];
        for (i, &boundary) in boundary_host.iter().enumerate() {
            if boundary == 1 {
                let group = group_ids_host[i] as usize;
                group_first_indices[group] = i;
            }
        }
        // First element is always a boundary
        group_first_indices[0] = 0;

        // Extract key values for result
        let mut result_keys = Vec::with_capacity((num_groups as usize) * 4);
        for &idx in &group_first_indices {
            let start = idx * 4;
            let end = start + 4;
            result_keys.extend_from_slice(&key_col_host[start..end]);
        }

        // Upload result key column
        let mut result_key_col = self.memory.alloc::<u8>(result_keys.len())?;
        self.device
            .inner()
            .htod_sync_copy_into(&result_keys, &mut result_key_col)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload result keys: {}", e)))?;

        // Upload output bytes as u8 column
        let mut output_col_u8 = self.memory.alloc::<u8>(output_bytes.len())?;
        self.device
            .inner()
            .htod_sync_copy_into(&output_bytes, &mut output_col_u8)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload output: {}", e)))?;

        // Build result schema: key columns + aggregation result
        let result_schema = self.groupby_result_schema(input.schema(), key_cols, agg);

        Ok(CudaBuffer::from_columns(
            vec![result_key_col.into(), output_col_u8.into()],
            num_groups as u64,
            result_schema,
        ))
    }

    /// Multi-aggregation groupby
    ///
    /// Performs groupby with multiple aggregation operations at once.
    /// This is more efficient than running separate groupby operations
    /// because it only sorts and computes group boundaries once.
    ///
    /// # Arguments
    /// * `buffer` - The input buffer
    /// * `key_cols` - Column indices for grouping (currently only single-column supported)
    /// * `aggs` - A slice of (value_col, AggOp) pairs specifying which aggregations to perform
    ///
    /// # Returns
    /// A buffer with one row per group, containing key columns followed by aggregated values
    /// in the same order as the `aggs` parameter
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails
    ///
    /// # Example
    /// ```ignore
    /// let result = provider.groupby_multi_agg(
    ///     &buffer,
    ///     &[0],  // group by column 0
    ///     &[(1, AggOp::Sum), (1, AggOp::Count), (1, AggOp::Min)],
    /// )?;
    /// // result has columns: key, sum, count, min
    /// ```
    pub fn groupby_multi_agg(
        &self,
        buffer: &CudaBuffer,
        key_cols: &[usize],
        aggs: &[(usize, AggOp)],
    ) -> Result<CudaBuffer> {
        // Handle empty input
        if buffer.is_empty() {
            let result_schema =
                self.groupby_multi_agg_result_schema(buffer.schema(), key_cols, aggs);
            return self.create_empty_buffer(result_schema);
        }

        // Validate inputs
        if key_cols.is_empty() {
            return Err(XlogError::Kernel(
                "GroupBy requires at least one key column".to_string(),
            ));
        }
        if aggs.is_empty() {
            return Err(XlogError::Kernel(
                "GroupBy requires at least one aggregation".to_string(),
            ));
        }

        // Validate key columns exist
        for &key_col in key_cols {
            if key_col >= buffer.arity() {
                return Err(XlogError::Kernel(format!(
                    "Key column {} out of bounds (arity {})",
                    key_col,
                    buffer.arity()
                )));
            }
        }

        // Validate all value columns exist and basic dtype constraints for current kernels.
        for &(value_col, agg_op) in aggs {
            if value_col >= buffer.arity() {
                return Err(XlogError::Kernel(format!(
                    "Value column {} out of bounds (arity {})",
                    value_col,
                    buffer.arity()
                )));
            }

            let value_ty = buffer
                .schema()
                .column_type(value_col)
                .ok_or_else(|| XlogError::Kernel("Value column has no type".to_string()))?;
            match agg_op {
                AggOp::Count => {}
                AggOp::Sum | AggOp::Min | AggOp::Max => {
                    if value_ty != ScalarType::U32 {
                        return Err(XlogError::Kernel(format!(
                            "{:?} currently requires U32 values, got {:?}",
                            agg_op, value_ty
                        )));
                    }
                }
                AggOp::LogSumExp => {
                    if value_ty != ScalarType::F64 {
                        return Err(XlogError::Kernel(format!(
                            "LogSumExp requires F64 values, got {:?}",
                            value_ty
                        )));
                    }
                }
            }
        }

        let num_rows = buffer.num_rows() as u32;

        // Step 1: Sort buffer by key columns
        let sorted = self.sort(buffer, key_cols)?;

        // Step 2: Detect boundaries using detect_group_boundaries kernel over packed key bytes
        let boundary_func = self
            .device
            .inner()
            .get_func(GROUPBY_MODULE, groupby_kernels::DETECT_GROUP_BOUNDARIES)
            .ok_or_else(|| {
                XlogError::Kernel("detect_group_boundaries kernel not found".to_string())
            })?;

        // Allocate boundaries mask
        let boundaries = self.memory.alloc::<u8>(num_rows as usize)?;

        let packed = self.compute_hashes_and_pack_keys(&sorted, key_cols)?;
        if packed.key_bytes == 0 || packed.key_bytes % 4 != 0 {
            return Err(XlogError::Kernel(format!(
                "GroupBy key packing produced {} bytes per row (expected multiple of 4); Bool keys are not supported",
                packed.key_bytes
            )));
        }

        let segments_per_row = (packed.key_bytes / 4) as usize;
        let total_segments = (num_rows as usize) * segments_per_row;
        let packed_u32 = self.bytes_as_u32_view(&packed.packed_keys, total_segments)?;

        // Launch boundary detection
        let block_size = 256u32;
        let grid_size = (num_rows + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Kernel parameters match expected signature
        unsafe {
            boundary_func
                .clone()
                .launch(
                    config,
                    (
                        &packed_u32,
                        num_rows,
                        segments_per_row as u32,
                        segments_per_row as u32,
                        &boundaries,
                    ),
                )
        }
        .map_err(|e| XlogError::Kernel(format!("detect_group_boundaries failed: {}", e)))?;

        self.device.synchronize()?;

        // Step 3: Compute group IDs on-device using prefix sum over boundaries.
        let device = self.device.inner();
        let num_blocks = grid_size;
        let d_boundary_pos = self.memory.alloc::<u32>(num_rows as usize)?;
        let mut d_block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

        let phase1_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase1 kernel".to_string())
            })?;

        // SAFETY: multiblock_scan_phase1(const uint8_t* mask, uint32_t* prefix_sum, uint32_t* block_sums, uint32_t n)
        unsafe {
            phase1_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&boundaries, &d_boundary_pos, &d_block_sums, num_rows),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase1 failed: {}", e)))?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace(&mut d_block_sums, num_blocks)?;

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
                    (&d_boundary_pos, &d_block_sums, num_rows),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        let last_idx = (num_rows as usize).checked_sub(1).ok_or_else(|| {
            XlogError::Kernel("GroupBy: unexpected empty input after boundary scan".to_string())
        })?;

        let last_prefix_view = d_boundary_pos.slice(last_idx..(last_idx + 1));
        let last_boundary_view = boundaries.slice(last_idx..(last_idx + 1));
        let mut last_prefix = [0u32];
        let mut last_boundary = [0u8];
        self.dtoh_sync_copy_into_tracked(&last_prefix_view, &mut last_prefix)
            .map_err(|e| XlogError::Kernel(format!("Failed to read last prefix sum: {}", e)))?;
        self.dtoh_sync_copy_into_tracked(&last_boundary_view, &mut last_boundary)
            .map_err(|e| XlogError::Kernel(format!("Failed to read last boundary: {}", e)))?;

        let num_groups = (last_prefix[0] as u64) + (last_boundary[0] as u64);
        if num_groups == 0 {
            let result_schema =
                self.groupby_multi_agg_result_schema(buffer.schema(), key_cols, aggs);
            return self.create_empty_buffer(result_schema);
        }

        let num_groups_u32 = u32::try_from(num_groups).map_err(|_| {
            XlogError::Kernel(format!(
                "GroupBy produced {} groups, exceeding u32::MAX",
                num_groups
            ))
        })?;
        let num_groups_usize = num_groups_u32 as usize;

        let mut group_ids = self.memory.alloc::<u32>(num_rows as usize)?;
        let mut group_first_idx = self.memory.alloc::<u32>(num_groups_usize)?;

        let group_ids_fn = device
            .get_func(GROUPBY_MODULE, groupby_kernels::GROUP_IDS_FROM_BOUNDARIES)
            .ok_or_else(|| {
                XlogError::Kernel("group_ids_from_boundaries kernel not found".to_string())
            })?;
        let group_start_fn = device
            .get_func(GROUPBY_MODULE, groupby_kernels::GROUP_START_INDICES)
            .ok_or_else(|| {
                XlogError::Kernel("group_start_indices kernel not found".to_string())
            })?;

        // SAFETY: group_ids_from_boundaries(boundaries, boundary_pos, num_rows, group_ids)
        unsafe {
            group_ids_fn
                .clone()
                .launch(config, (&boundaries, &d_boundary_pos, num_rows, &mut group_ids))
        }
        .map_err(|e| XlogError::Kernel(format!("group_ids_from_boundaries failed: {}", e)))?;

        // SAFETY: group_start_indices(boundaries, boundary_pos, num_rows, group_first_idx)
        unsafe {
            group_start_fn
                .clone()
                .launch(
                    config,
                    (&boundaries, &d_boundary_pos, num_rows, &mut group_first_idx),
                )
        }
        .map_err(|e| XlogError::Kernel(format!("group_start_indices failed: {}", e)))?;

        self.device.synchronize()?;

        // Step 4: For each (value_col, op) pair, run the appropriate kernel on-device.
        let mut agg_columns: Vec<CudaColumn> = Vec::with_capacity(aggs.len());

        for &(value_col, agg_op) in aggs {
            let values = sorted
                .column(value_col)
                .ok_or_else(|| XlogError::Kernel("Value column not found".to_string()))?;

            match agg_op {
                AggOp::Count => {
                    let output_bytes = num_groups_usize
                        .checked_mul(std::mem::size_of::<u64>())
                        .ok_or_else(|| XlogError::Kernel("Count output size overflow".to_string()))?;
                    let mut output = self.memory.alloc::<u8>(output_bytes)?;
                    if num_groups_u32 > 0 {
                        device
                            .memset_zeros(&mut output)
                            .map_err(|e| {
                                XlogError::Kernel(format!("Failed to zero count output: {}", e))
                            })?;
                    }

                    let count_func = device
                        .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_COUNT)
                        .ok_or_else(|| {
                            XlogError::Kernel("groupby_count kernel not found".to_string())
                        })?;

                    // SAFETY: groupby_count(boundaries, group_ids, num_rows, counts)
                    unsafe {
                        count_func
                            .clone()
                            .launch(config, (&boundaries, &group_ids, num_rows, &output))
                    }
                    .map_err(|e| XlogError::Kernel(format!("groupby_count failed: {}", e)))?;

                    self.device.synchronize()?;
                    agg_columns.push(output.into());
                }
                AggOp::Sum => {
                    let values_view = self.column_as_u32_view(values, num_rows as usize)?;
                    let output_bytes = num_groups_usize
                        .checked_mul(std::mem::size_of::<u64>())
                        .ok_or_else(|| XlogError::Kernel("Sum output size overflow".to_string()))?;
                    let mut output = self.memory.alloc::<u8>(output_bytes)?;
                    if num_groups_u32 > 0 {
                        device
                            .memset_zeros(&mut output)
                            .map_err(|e| {
                                XlogError::Kernel(format!("Failed to zero sum output: {}", e))
                            })?;
                    }

                    let sum_func = device
                        .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_SUM)
                        .ok_or_else(|| {
                            XlogError::Kernel("groupby_sum kernel not found".to_string())
                        })?;

                    // SAFETY: groupby_sum(values, group_ids, num_rows, sums)
                    unsafe {
                        sum_func
                            .clone()
                            .launch(config, (&values_view, &group_ids, num_rows, &output))
                    }
                    .map_err(|e| XlogError::Kernel(format!("groupby_sum failed: {}", e)))?;

                    self.device.synchronize()?;
                    agg_columns.push(output.into());
                }
                AggOp::Min => {
                    let values_view = self.column_as_u32_view(values, num_rows as usize)?;
                    let output_bytes = num_groups_usize
                        .checked_mul(std::mem::size_of::<u32>())
                        .ok_or_else(|| XlogError::Kernel("Min output size overflow".to_string()))?;
                    let mut output = self.memory.alloc::<u8>(output_bytes)?;
                    let fill_fn = device
                        .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_U32)
                        .ok_or_else(|| {
                            XlogError::Kernel("arith_fill_const_u32 not found".to_string())
                        })?;
                    let fill_config = LaunchConfig::for_num_elems(num_groups_u32);
                    unsafe {
                        fill_fn
                            .clone()
                            .launch(fill_config, (u32::MAX, num_groups_u32, &mut output))
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to init min output: {}", e))
                    })?;

                    let min_func = device
                        .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MIN)
                        .ok_or_else(|| {
                            XlogError::Kernel("groupby_min kernel not found".to_string())
                        })?;

                    // SAFETY: groupby_min(values, group_ids, num_rows, mins)
                    unsafe {
                        min_func
                            .clone()
                            .launch(config, (&values_view, &group_ids, num_rows, &output))
                    }
                    .map_err(|e| XlogError::Kernel(format!("groupby_min failed: {}", e)))?;

                    self.device.synchronize()?;
                    agg_columns.push(output.into());
                }
                AggOp::Max => {
                    let values_view = self.column_as_u32_view(values, num_rows as usize)?;
                    let output_bytes = num_groups_usize
                        .checked_mul(std::mem::size_of::<u32>())
                        .ok_or_else(|| XlogError::Kernel("Max output size overflow".to_string()))?;
                    let mut output = self.memory.alloc::<u8>(output_bytes)?;
                    if num_groups_u32 > 0 {
                        device
                            .memset_zeros(&mut output)
                            .map_err(|e| {
                                XlogError::Kernel(format!("Failed to zero max output: {}", e))
                            })?;
                    }

                    let max_func = device
                        .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MAX)
                        .ok_or_else(|| {
                            XlogError::Kernel("groupby_max kernel not found".to_string())
                        })?;

                    // SAFETY: groupby_max(values, group_ids, num_rows, maxs)
                    unsafe {
                        max_func
                            .clone()
                            .launch(config, (&values_view, &group_ids, num_rows, &output))
                    }
                    .map_err(|e| XlogError::Kernel(format!("groupby_max failed: {}", e)))?;

                    self.device.synchronize()?;
                    agg_columns.push(output.into());
                }
                AggOp::LogSumExp => {
                    let values_f64 = self.column_as_f64_view(values, num_rows as usize)?;
                    let output_bytes = num_groups_usize
                        .checked_mul(std::mem::size_of::<f64>())
                        .ok_or_else(|| {
                            XlogError::Kernel("LogSumExp output size overflow".to_string())
                        })?;
                    let mut maxs = self.memory.alloc::<u8>(output_bytes)?;
                    let mut sumexps = self.memory.alloc::<u8>(output_bytes)?;
                    let results = self.memory.alloc::<u8>(output_bytes)?;

                    let fill_f64 = device
                        .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_F64)
                        .ok_or_else(|| {
                            XlogError::Kernel("arith_fill_const_f64 not found".to_string())
                        })?;
                    let fill_config = LaunchConfig::for_num_elems(num_groups_u32);
                    unsafe {
                        fill_f64.clone().launch(
                            fill_config,
                            (f64::NEG_INFINITY, num_groups_u32, &mut maxs),
                        )
                    }
                    .map_err(|e| XlogError::Kernel(format!("Failed to init maxs: {}", e)))?;
                    if num_groups_u32 > 0 {
                        device
                            .memset_zeros(&mut sumexps)
                            .map_err(|e| {
                                XlogError::Kernel(format!("Failed to init sumexps: {}", e))
                            })?;
                    }

                    let max_func = device
                        .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_LOGSUMEXP_MAX)
                        .ok_or_else(|| {
                            XlogError::Kernel("groupby_logsumexp_max kernel not found".to_string())
                        })?;

                    unsafe {
                        max_func
                            .clone()
                            .launch(config, (&values_f64, &group_ids, num_rows, &maxs))
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!("groupby_logsumexp_max failed: {}", e))
                    })?;

                    self.device.synchronize()?;

                    let sumexp_func = device
                        .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_LOGSUMEXP_SUMEXP)
                        .ok_or_else(|| {
                            XlogError::Kernel(
                                "groupby_logsumexp_sumexp kernel not found".to_string(),
                            )
                        })?;

                    unsafe {
                        sumexp_func
                            .clone()
                            .launch(
                                config,
                                (&values_f64, &group_ids, &maxs, num_rows, &sumexps),
                            )
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!("groupby_logsumexp_sumexp failed: {}", e))
                    })?;

                    self.device.synchronize()?;

                    let final_config = LaunchConfig::for_num_elems(num_groups_u32);
                    let final_func = device
                        .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_LOGSUMEXP_FINAL)
                        .ok_or_else(|| {
                            XlogError::Kernel(
                                "groupby_logsumexp_final kernel not found".to_string(),
                            )
                        })?;

                    unsafe {
                        final_func
                            .clone()
                            .launch(final_config, (&maxs, &sumexps, num_groups_u32, &results))
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!("groupby_logsumexp_final failed: {}", e))
                    })?;

                    self.device.synchronize()?;
                    agg_columns.push(results.into());
                }
            }
        }

        // Step 5: Build output buffer with keys and aggregated values.
        let mut result_columns: Vec<CudaColumn> =
            Vec::with_capacity(key_cols.len() + aggs.len());

        let group_packed_bytes = (num_groups_usize)
            .checked_mul(packed.key_bytes as usize)
            .ok_or_else(|| XlogError::Kernel("GroupBy packed size overflow".to_string()))?;
        let mut group_packed = self.memory.alloc::<u8>(group_packed_bytes)?;

        let gather_fn = device
            .get_func(PACK_MODULE, pack_kernels::GATHER_PACKED_ROWS)
            .ok_or_else(|| {
                XlogError::Kernel("gather_packed_rows kernel not found".to_string())
            })?;
        let gather_config = LaunchConfig::for_num_elems(num_groups_u32);

        // SAFETY: gather_packed_rows(src_packed, row_size, indices, num_indices, dst_packed)
        unsafe {
            gather_fn.clone().launch(
                gather_config,
                (
                    &packed.packed_keys,
                    packed.key_bytes,
                    &group_first_idx,
                    num_groups_u32,
                    &mut group_packed,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("gather_packed_rows failed: {}", e)))?;

        let mut col_offsets: Vec<u32> = Vec::with_capacity(key_cols.len());
        let mut col_sizes: Vec<u32> = Vec::with_capacity(key_cols.len());
        let mut offset = 0u32;
        for &key_col in key_cols {
            let size = buffer
                .schema()
                .column_type(key_col)
                .map(|t| t.size_bytes() as u32)
                .unwrap_or(4);
            col_offsets.push(offset);
            col_sizes.push(size);
            offset = offset
                .checked_add(size)
                .ok_or_else(|| XlogError::Kernel("GroupBy key size overflow".to_string()))?;
        }

        let unpack_fn = device
            .get_func(PACK_MODULE, pack_kernels::UNPACK_COLUMN)
            .ok_or_else(|| XlogError::Kernel("unpack_column kernel not found".to_string()))?;
        let unpack_config = LaunchConfig::for_num_elems(num_groups_u32);

        for idx in 0..key_cols.len() {
            let col_size = col_sizes[idx];
            let col_offset = col_offsets[idx];
            let out_bytes = (num_groups_usize)
                .checked_mul(col_size as usize)
                .ok_or_else(|| XlogError::Kernel("GroupBy key column overflow".to_string()))?;
            let mut out_col = self.memory.alloc::<u8>(out_bytes)?;

            // SAFETY: unpack_column(packed_input, row_size, col_offset, col_size, num_rows, col_output)
            unsafe {
                unpack_fn.clone().launch(
                    unpack_config,
                    (
                        &group_packed,
                        packed.key_bytes,
                        col_offset,
                        col_size,
                        num_groups_u32,
                        &mut out_col,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("unpack_column failed: {}", e)))?;

            result_columns.push(out_col.into());
        }

        result_columns.extend(agg_columns);

        let result_schema = self.groupby_multi_agg_result_schema(buffer.schema(), key_cols, aggs);

        Ok(CudaBuffer::from_columns(
            result_columns,
            num_groups as u64,
            result_schema,
        ))
    }

    /// Create result schema for multi-aggregation groupby
    fn groupby_multi_agg_result_schema(
        &self,
        input: &Schema,
        key_cols: &[usize],
        aggs: &[(usize, AggOp)],
    ) -> Schema {
        let mut columns: Vec<(String, ScalarType)> = key_cols
            .iter()
            .filter_map(|&i| input.columns.get(i).cloned())
            .collect();

        for (i, &(_value_col, agg_op)) in aggs.iter().enumerate() {
            let agg_name = match agg_op {
                AggOp::Count => format!("count_{}", i),
                AggOp::Sum => format!("sum_{}", i),
                AggOp::Min => format!("min_{}", i),
                AggOp::Max => format!("max_{}", i),
                AggOp::LogSumExp => format!("logsumexp_{}", i),
            };
            // Return correct types for each aggregation
            // Count and Sum use u64 to match predicate declarations and prevent overflow
            let agg_type = match agg_op {
                AggOp::Count => ScalarType::U64,
                AggOp::Sum => ScalarType::U64,
                AggOp::Min | AggOp::Max => ScalarType::U32,
                AggOp::LogSumExp => ScalarType::F64,
            };
            columns.push((agg_name, agg_type));
        }

        Schema::new(columns)
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

    pub fn prefix_sum_mask(&self, mask: &[u8]) -> Result<(Vec<u32>, u32)> {
        if mask.is_empty() {
            return Ok((vec![], 0));
        }

        let n = mask.len();
        if n > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Mask length {} exceeds u32::MAX",
                n
            )));
        }

        // For small inputs, use CPU scan (faster than kernel launch overhead)
        if n <= 256 {
            return self.prefix_sum_mask_cpu(mask);
        }

        // For larger inputs, use multi-block GPU scan
        self.prefix_sum_mask_gpu_multiblock(mask)
    }

    /// CPU implementation for small prefix sums (avoids kernel launch overhead)
    fn prefix_sum_mask_cpu(&self, mask: &[u8]) -> Result<(Vec<u32>, u32)> {
        let mut prefix_sum = Vec::with_capacity(mask.len());
        let mut sum = 0u32;
        for &m in mask {
            prefix_sum.push(sum);
            sum += m as u32;
        }
        Ok((prefix_sum, sum))
    }

    /// Multi-block GPU implementation for large prefix sums
    /// Uses three-phase algorithm:
    /// 1. Each block computes local exclusive scan and outputs block total
    /// 2. Scan the block totals to get block offsets
    /// 3. Add block offsets to each element
    fn prefix_sum_mask_gpu_multiblock(&self, mask: &[u8]) -> Result<(Vec<u32>, u32)> {
        let n = mask.len();
        let device = self.device.inner();
        let block_size = 256u32;
        let num_blocks = ((n as u32) + block_size - 1) / block_size;

        // Upload mask to GPU
        let d_mask = device
            .htod_sync_copy(mask)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload mask: {}", e)))?;

        // Allocate output for prefix sum (using memory manager for budget enforcement)
        let d_prefix_sum = self.memory.alloc::<u32>(n)?;

        // Allocate block sums array (using memory manager for budget enforcement)
        let mut d_block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

        // Phase 1: Block-level exclusive scans + collect block totals
        let phase1_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase1 kernel".to_string())
            })?;

        // SAFETY: Kernel parameters match expected signature:
        // multiblock_scan_phase1(const uint8_t* mask, uint32_t* prefix_sum, uint32_t* block_sums, uint32_t n)
        unsafe {
            phase1_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&d_mask, &d_prefix_sum, &d_block_sums, n as u32),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!("Failed to launch multiblock_scan_phase1: {}", e))
        })?;

        // Phase 2: Scan block sums (only if we have more than 1 block)
        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace(&mut d_block_sums, num_blocks)?;

            // Phase 3: Add block offsets
            let phase3_fn = device
                .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE3)
                .ok_or_else(|| {
                    XlogError::Kernel("Failed to get multiblock_scan_phase3 kernel".to_string())
                })?;

            // SAFETY: Kernel parameters match expected signature:
            // multiblock_scan_phase3(uint32_t* prefix_sum, const uint32_t* block_offsets, uint32_t n)
            unsafe {
                phase3_fn.clone().launch(
                    LaunchConfig {
                        grid_dim: (num_blocks, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&d_prefix_sum, &d_block_sums, n as u32),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to launch multiblock_scan_phase3: {}", e))
            })?;
        }

        // Synchronize and download results
        self.device.synchronize()?;

        let prefix_sum = device
            .dtoh_sync_copy(&d_prefix_sum)
            .map_err(|e| XlogError::Kernel(format!("Failed to download prefix_sum: {}", e)))?;

        // Compute count from the last prefix sum value + last mask value
        let count = prefix_sum[n - 1] + mask[n - 1] as u32;

        Ok((prefix_sum, count))
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

        Ok(())
    }

    // ============== Sort Methods ==============

    const SORT_BLOCK_SIZE: u32 = 256;

    /// Sort buffer by key columns.
    ///
    /// Computes a stable row permutation on the GPU (supports multi-column and all scalar types),
    /// then applies the permutation on the GPU to reorder all columns.
    ///
    /// # Arguments
    /// * `input` - The input buffer to sort
    /// * `key_cols` - Column indices to use for sorting (lexicographic, first key is most significant)
    ///
    /// # Returns
    /// A new buffer with rows sorted by the key columns
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - `key_cols` is empty or out of bounds
    /// - Input has more than `u32::MAX` rows
    /// - Download/upload or kernel execution fails
    pub fn sort(&self, input: &CudaBuffer, key_cols: &[usize]) -> Result<CudaBuffer> {
        if input.num_rows == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }

        if key_cols.is_empty() {
            return Err(XlogError::Kernel(
                "Sort requires at least one key column".to_string(),
            ));
        }

        if input.num_rows > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Sort supports at most {} rows, got {}",
                u32::MAX,
                input.num_rows
            )));
        }

        for &key_col in key_cols {
            if key_col >= input.arity() {
                return Err(XlogError::Kernel(format!(
                    "Key column index {} out of bounds (arity {})",
                    key_col,
                    input.arity()
                )));
            }
        }

        let n = input.num_rows as u32;
        let device = self.device.inner();

        let block_size = Self::SORT_BLOCK_SIZE;
        let grid_size = (n + block_size - 1) / block_size;
        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Allocate and initialize identity permutation.
        let init_fn = device
            .get_func(SORT_MODULE, sort_kernels::INIT_INDICES)
            .ok_or_else(|| XlogError::Kernel("init_indices kernel not found".to_string()))?;

        let mut indices_a = self.memory.alloc::<u32>(n as usize)?;
        let mut indices_b = self.memory.alloc::<u32>(n as usize)?;

        // SAFETY: init_indices(uint32_t* indices, uint32_t n)
        unsafe { init_fn.clone().launch(launch_config, (&mut indices_a, n)) }
            .map_err(|e| XlogError::Kernel(format!("init_indices failed: {}", e)))?;

        // Working key buffers (u32 words).
        let mut keys_a = self.memory.alloc::<u32>(n as usize)?;
        let mut keys_b = self.memory.alloc::<u32>(n as usize)?;

        // Radix-sort scratch.
        let mut d_hist = self.memory.alloc::<u32>((grid_size as usize) * 16)?;
        let mut d_prefix = self.memory.alloc::<u32>(16)?;
        let mut d_ranks = self.memory.alloc::<u32>(n as usize)?;

        // Process key columns from least-significant to most-significant (stable LSD).
        for &col_idx in key_cols.iter().rev() {
            let ty = input.schema.column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Key column {} type not found in schema", col_idx))
            })?;

            let col = input
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Key column {} not found", col_idx)))?;

            match ty {
                ScalarType::U32 | ScalarType::Symbol => {
                    let col_view = self.column_as_u32_view(col, n as usize)?;
                    let gather_fn = device
                        .get_func(SORT_MODULE, sort_kernels::APPLY_PERMUTATION_U32)
                        .ok_or_else(|| {
                            XlogError::Kernel("apply_permutation_u32 kernel not found".to_string())
                        })?;

                    // SAFETY: apply_permutation_u32(input, output, permutation, n)
                    unsafe {
                        gather_fn
                            .clone()
                            .launch(launch_config, (&col_view, &mut keys_a, &indices_a, n))
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!("apply_permutation_u32 failed: {}", e))
                    })?;

                    self.radix_sort_u32_pairs_with_scratch(
                        &mut keys_a,
                        &mut keys_b,
                        &mut indices_a,
                        &mut indices_b,
                        &mut d_hist,
                        &mut d_prefix,
                        &mut d_ranks,
                        n,
                    )?;
                }
                ScalarType::I32 => {
                    let col_bits = self.column_as_u32_view(col, n as usize)?;
                    let gather_fn = device
                        .get_func(SORT_MODULE, sort_kernels::GATHER_KEYS_I32_ORDERED_U32)
                        .ok_or_else(|| {
                            XlogError::Kernel(
                                "gather_keys_i32_ordered_u32 kernel not found".to_string(),
                            )
                        })?;

                    // SAFETY: gather_keys_i32_ordered_u32(i32_bits, permutation, num_rows, out_keys)
                    unsafe {
                        gather_fn
                            .clone()
                            .launch(launch_config, (&col_bits, &indices_a, n, &mut keys_a))
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!("gather_keys_i32_ordered_u32 failed: {}", e))
                    })?;

                    self.radix_sort_u32_pairs_with_scratch(
                        &mut keys_a,
                        &mut keys_b,
                        &mut indices_a,
                        &mut indices_b,
                        &mut d_hist,
                        &mut d_prefix,
                        &mut d_ranks,
                        n,
                    )?;
                }
                ScalarType::F32 => {
                    let col_bits = self.column_as_u32_view(col, n as usize)?;
                    let gather_fn = device
                        .get_func(SORT_MODULE, sort_kernels::GATHER_KEYS_F32_ORDERED_U32)
                        .ok_or_else(|| {
                            XlogError::Kernel(
                                "gather_keys_f32_ordered_u32 kernel not found".to_string(),
                            )
                        })?;

                    // SAFETY: gather_keys_f32_ordered_u32(f32_bits, permutation, num_rows, out_keys)
                    unsafe {
                        gather_fn
                            .clone()
                            .launch(launch_config, (&col_bits, &indices_a, n, &mut keys_a))
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!("gather_keys_f32_ordered_u32 failed: {}", e))
                    })?;

                    self.radix_sort_u32_pairs_with_scratch(
                        &mut keys_a,
                        &mut keys_b,
                        &mut indices_a,
                        &mut indices_b,
                        &mut d_hist,
                        &mut d_prefix,
                        &mut d_ranks,
                        n,
                    )?;
                }
                ScalarType::Bool => {
                    if col.num_bytes() < n as usize {
                        return Err(XlogError::Kernel(format!(
                            "Bool column {} has {} bytes but expected {}",
                            col_idx,
                            col.num_bytes(),
                            n
                        )));
                    }

                    let gather_fn = device
                        .get_func(SORT_MODULE, sort_kernels::GATHER_KEYS_BOOL_ORDERED_U32)
                        .ok_or_else(|| {
                            XlogError::Kernel(
                                "gather_keys_bool_ordered_u32 kernel not found".to_string(),
                            )
                        })?;

                    // SAFETY: gather_keys_bool_ordered_u32(bools, permutation, num_rows, out_keys)
                    unsafe {
                        gather_fn
                            .clone()
                            .launch(launch_config, (col, &indices_a, n, &mut keys_a))
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!("gather_keys_bool_ordered_u32 failed: {}", e))
                    })?;

                    self.radix_sort_u32_pairs_with_scratch(
                        &mut keys_a,
                        &mut keys_b,
                        &mut indices_a,
                        &mut indices_b,
                        &mut d_hist,
                        &mut d_prefix,
                        &mut d_ranks,
                        n,
                    )?;
                }
                ScalarType::U64 => {
                    let col_bits = self.column_as_u64_view(col, n as usize)?;
                    for &word in &[
                        sort_kernels::GATHER_KEYS_U64_LO_U32,
                        sort_kernels::GATHER_KEYS_U64_HI_U32,
                    ] {
                        let gather_fn = device.get_func(SORT_MODULE, word).ok_or_else(|| {
                            XlogError::Kernel(format!("{} kernel not found", word))
                        })?;

                        // SAFETY: gather_keys_u64_*_u32(vals, permutation, num_rows, out_keys)
                        unsafe {
                            gather_fn
                                .clone()
                                .launch(launch_config, (&col_bits, &indices_a, n, &mut keys_a))
                        }
                        .map_err(|e| XlogError::Kernel(format!("{} failed: {}", word, e)))?;

                        self.radix_sort_u32_pairs_with_scratch(
                            &mut keys_a,
                            &mut keys_b,
                            &mut indices_a,
                            &mut indices_b,
                            &mut d_hist,
                            &mut d_prefix,
                            &mut d_ranks,
                            n,
                        )?;
                    }
                }
                ScalarType::I64 => {
                    let col_bits = self.column_as_u64_view(col, n as usize)?;
                    for &word in &[
                        sort_kernels::GATHER_KEYS_I64_LO_U32,
                        sort_kernels::GATHER_KEYS_I64_HI_U32,
                    ] {
                        let gather_fn = device.get_func(SORT_MODULE, word).ok_or_else(|| {
                            XlogError::Kernel(format!("{} kernel not found", word))
                        })?;

                        // SAFETY: gather_keys_i64_*_u32(i64_bits, permutation, num_rows, out_keys)
                        unsafe {
                            gather_fn
                                .clone()
                                .launch(launch_config, (&col_bits, &indices_a, n, &mut keys_a))
                        }
                        .map_err(|e| XlogError::Kernel(format!("{} failed: {}", word, e)))?;

                        self.radix_sort_u32_pairs_with_scratch(
                            &mut keys_a,
                            &mut keys_b,
                            &mut indices_a,
                            &mut indices_b,
                            &mut d_hist,
                            &mut d_prefix,
                            &mut d_ranks,
                            n,
                        )?;
                    }
                }
                ScalarType::F64 => {
                    let col_bits = self.column_as_u64_view(col, n as usize)?;
                    for &word in &[
                        sort_kernels::GATHER_KEYS_F64_LO_U32,
                        sort_kernels::GATHER_KEYS_F64_HI_U32,
                    ] {
                        let gather_fn = device.get_func(SORT_MODULE, word).ok_or_else(|| {
                            XlogError::Kernel(format!("{} kernel not found", word))
                        })?;

                        // SAFETY: gather_keys_f64_*_u32(f64_bits, permutation, num_rows, out_keys)
                        unsafe {
                            gather_fn
                                .clone()
                                .launch(launch_config, (&col_bits, &indices_a, n, &mut keys_a))
                        }
                        .map_err(|e| XlogError::Kernel(format!("{} failed: {}", word, e)))?;

                        self.radix_sort_u32_pairs_with_scratch(
                            &mut keys_a,
                            &mut keys_b,
                            &mut indices_a,
                            &mut indices_b,
                            &mut d_hist,
                            &mut d_prefix,
                            &mut d_ranks,
                            n,
                        )?;
                    }
                }
            }
        }

        self.apply_permutation_gpu(input, &indices_a)
    }

    fn radix_sort_u32_pairs_with_scratch(
        &self,
        keys_a: &mut crate::memory::TrackedCudaSlice<u32>,
        keys_b: &mut crate::memory::TrackedCudaSlice<u32>,
        indices_a: &mut crate::memory::TrackedCudaSlice<u32>,
        indices_b: &mut crate::memory::TrackedCudaSlice<u32>,
        hist: &mut crate::memory::TrackedCudaSlice<u32>,
        prefix: &mut crate::memory::TrackedCudaSlice<u32>,
        ranks: &mut crate::memory::TrackedCudaSlice<u32>,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }

        let device = self.device.inner();
        let block_size = Self::SORT_BLOCK_SIZE;
        let grid_size = (n + block_size - 1) / block_size;

        let sort_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let histogram_fn = device
            .get_func(SORT_MODULE, sort_kernels::RADIX_HISTOGRAM)
            .ok_or_else(|| XlogError::Kernel("radix_histogram kernel not found".to_string()))?;
        let prefix_fn = device
            .get_func(SORT_MODULE, sort_kernels::COMPUTE_DIGIT_PREFIX_SUMS)
            .ok_or_else(|| {
                XlogError::Kernel("compute_digit_prefix_sums kernel not found".to_string())
            })?;
        let ranks_fn = device
            .get_func(SORT_MODULE, sort_kernels::COMPUTE_RANKS)
            .ok_or_else(|| XlogError::Kernel("compute_ranks kernel not found".to_string()))?;
        let scatter_fn = device
            .get_func(SORT_MODULE, sort_kernels::RADIX_SCATTER_STABLE)
            .ok_or_else(|| {
                XlogError::Kernel("radix_scatter_stable kernel not found".to_string())
            })?;

        let prefix_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut in_a = true;
        for pass in 0..8u32 {
            let shift = pass * 4;

            let (keys_in, indices_in, keys_out, indices_out) = if in_a {
                (&*keys_a, &*indices_a, &mut *keys_b, &mut *indices_b)
            } else {
                (&*keys_b, &*indices_b, &mut *keys_a, &mut *indices_a)
            };

            // Histogram (digit-major): hist[digit * grid_size + block] = count
            // SAFETY: radix_histogram(keys, num_rows, histograms, shift)
            unsafe {
                histogram_fn
                    .clone()
                    .launch(sort_config, (keys_in, n, &mut *hist, shift))
            }
            .map_err(|e| XlogError::Kernel(format!("radix_histogram failed: {}", e)))?;

            // Compute global digit prefix sums.
            // SAFETY: compute_digit_prefix_sums(histograms, grid_size, prefix_sums)
            unsafe {
                prefix_fn
                    .clone()
                    .launch(prefix_config, (&*hist, grid_size, &mut *prefix))
            }
            .map_err(|e| XlogError::Kernel(format!("compute_digit_prefix_sums failed: {}", e)))?;

            // Convert per-block histograms to per-block exclusive offsets (in-place scan per digit).
            for digit in 0..16u32 {
                let start = (digit * grid_size) as usize;
                let end = start + (grid_size as usize);
                let mut digit_slice = hist.slice_mut(start..end);
                self.multiblock_scan_u32_view_inplace(&mut digit_slice, grid_size)?;
            }

            // Compute per-element ranks for stability.
            // SAFETY: compute_ranks(keys, num_rows, ranks, shift)
            unsafe {
                ranks_fn
                    .clone()
                    .launch(sort_config, (keys_in, n, &mut *ranks, shift))
            }
            .map_err(|e| XlogError::Kernel(format!("compute_ranks failed: {}", e)))?;

            // Stable scatter using digit prefix + per-block offsets + ranks.
            // SAFETY: radix_scatter_stable(keys_in, indices_in, ranks, keys_out, indices_out, prefix_sums, block_offsets, num_rows, shift)
            unsafe {
                scatter_fn.clone().launch(
                    sort_config,
                    (
                        keys_in,
                        indices_in,
                        &*ranks,
                        keys_out,
                        indices_out,
                        &*prefix,
                        &*hist,
                        n,
                        shift,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("radix_scatter_stable failed: {}", e)))?;

            in_a = !in_a;
        }

        // 8 passes (32b/4b) => even number of swaps => sorted data ends in A.
        if !in_a {
            return Err(XlogError::Kernel(
                "Unexpected radix-sort buffer parity (expected even number of passes)".to_string(),
            ));
        }

        Ok(())
    }

    /// Apply permutation to reorder all columns in buffer using GPU
    fn apply_permutation_gpu(
        &self,
        input: &CudaBuffer,
        permutation: &cudarc::driver::CudaSlice<u32>,
    ) -> Result<CudaBuffer> {
        let n = input.num_rows as u32;
        let device = self.device.inner();

        let grid_size = (n + Self::SORT_BLOCK_SIZE - 1) / Self::SORT_BLOCK_SIZE;
        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (Self::SORT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let apply_perm_fn = device
            .get_func(SORT_MODULE, sort_kernels::APPLY_PERMUTATION_BYTES)
            .ok_or_else(|| {
                XlogError::Kernel("apply_permutation_bytes kernel not found".to_string())
            })?;

        let mut new_columns = Vec::with_capacity(input.columns.len());

        for col_idx in 0..input.columns.len() {
            let src_col = input
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

            let elem_size = input
                .schema
                .column_type(col_idx)
                .ok_or_else(|| {
                    XlogError::Kernel(format!("Schema type for column {} not found", col_idx))
                })?
                .size_bytes() as u32;

            let output_bytes = (n as usize) * (elem_size as usize);
            if src_col.num_bytes() != output_bytes {
                return Err(XlogError::Kernel(format!(
                    "Column {} has {} bytes but expected {} (num_rows={}, elem_size={})",
                    col_idx,
                    src_col.num_bytes(),
                    output_bytes,
                    n,
                    elem_size
                )));
            }
            let dst_col = self.memory.alloc::<u8>(output_bytes)?;

            // SAFETY: Kernel signature matches: apply_permutation_bytes(input, output, permutation, num_rows, elem_size)
            unsafe {
                apply_perm_fn.clone().launch(
                    launch_config,
                    (src_col, &dst_col, permutation, n, elem_size),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("apply_permutation_bytes failed: {}", e)))?;

            new_columns.push(dst_col.into());
        }

        self.device.synchronize()?;

        Ok(CudaBuffer {
            columns: new_columns,
            num_rows: input.num_rows,
            schema: input.schema.clone(),
        })
    }

    /// Gather rows by explicit indices on GPU: output[i] = input[indices[i]].
    ///
    /// This is like `apply_permutation_gpu`, but the input can be larger than the output
    /// (i.e. `output_rows` < input.num_rows()).
    fn gather_buffer_by_indices(
        &self,
        input: &CudaBuffer,
        indices: &cudarc::driver::CudaSlice<u32>,
        output_rows: u32,
    ) -> Result<CudaBuffer> {
        if output_rows == 0 {
            return self.create_empty_buffer(input.schema().clone());
        }

        if input.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "GPU gather supports at most {} input rows, got {}",
                u32::MAX,
                input.num_rows()
            )));
        }

        let device = self.device.inner();
        let block_size = 256u32;
        let grid_size = (output_rows + block_size - 1) / block_size;
        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let gather_fn = device
            .get_func(SORT_MODULE, sort_kernels::APPLY_PERMUTATION_BYTES)
            .ok_or_else(|| {
                XlogError::Kernel("apply_permutation_bytes kernel not found".to_string())
            })?;

        let mut new_columns = Vec::with_capacity(input.columns.len());
        for col_idx in 0..input.columns.len() {
            let src_col = input
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

            let elem_size = input
                .schema
                .column_type(col_idx)
                .ok_or_else(|| {
                    XlogError::Kernel(format!("Schema type for column {} not found", col_idx))
                })?
                .size_bytes() as u32;

            let expected_src_bytes = (input.num_rows as usize) * (elem_size as usize);
            if src_col.num_bytes() != expected_src_bytes {
                return Err(XlogError::Kernel(format!(
                    "Column {} has {} bytes but expected {} (num_rows={}, elem_size={})",
                    col_idx,
                    src_col.num_bytes(),
                    expected_src_bytes,
                    input.num_rows,
                    elem_size
                )));
            }

            let dst_bytes = (output_rows as usize) * (elem_size as usize);
            let dst_col = self.memory.alloc::<u8>(dst_bytes)?;

            // SAFETY: Kernel signature matches: apply_permutation_bytes(input, output, permutation, num_rows, elem_size)
            unsafe {
                gather_fn.clone().launch(
                    launch_config,
                    (src_col, &dst_col, indices, output_rows, elem_size),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("apply_permutation_bytes failed: {}", e)))?;

            new_columns.push(dst_col.into());
        }

        self.device.synchronize()?;

        Ok(CudaBuffer {
            columns: new_columns,
            num_rows: output_rows as u64,
            schema: input.schema.clone(),
        })
    }

    // ============== Filter Methods ==============

    /// Filter buffer where u32 column equals constant
    ///
    /// # Arguments
    /// * `input` - The input buffer to filter
    /// * `col` - Column index to filter on
    /// * `value` - Value to compare against
    ///
    /// # Returns
    /// A new buffer containing only rows where `input[col] == value`
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails
    pub fn filter_u32_eq(&self, input: &CudaBuffer, col: usize, value: u32) -> Result<CudaBuffer> {
        self.filter_u32(input, col, value, CompareOp::Eq)
    }

    /// Filter buffer where u32 column is greater than constant
    ///
    /// # Arguments
    /// * `input` - The input buffer to filter
    /// * `col` - Column index to filter on
    /// * `value` - Value to compare against
    ///
    /// # Returns
    /// A new buffer containing only rows where `input[col] > value`
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails
    pub fn filter_u32_gt(&self, input: &CudaBuffer, col: usize, value: u32) -> Result<CudaBuffer> {
        self.filter_u32(input, col, value, CompareOp::Gt)
    }

    /// Filter u32 column with comparison operator.
    ///
    /// # Arguments
    /// * `input` - Buffer to filter
    /// * `col` - Column index to compare
    /// * `value` - Constant to compare against
    /// * `op` - Comparison operator
    ///
    /// # Errors
    /// Returns error if column index is out of bounds or the input exceeds the u32 row limit.
    pub fn filter_u32(
        &self,
        input: &CudaBuffer,
        col: usize,
        value: u32,
        op: CompareOp,
    ) -> Result<CudaBuffer> {
        if input.num_rows == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }

        if input.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "filter_u32 supports at most {} rows, got {}",
                u32::MAX,
                input.num_rows()
            )));
        }

        let n = input.num_rows as usize;
        let num_rows = input.num_rows as u32;
        let device = self.device.inner();

        // Validate column index
        if col >= input.arity() {
            return Err(XlogError::Kernel(format!(
                "Column index {} out of bounds (arity {})",
                col,
                input.arity()
            )));
        }

        // Validate column type is U32 (or Symbol which is u32-backed)
        let col_type = input.schema().column_type(col);
        if col_type != Some(ScalarType::U32) && col_type != Some(ScalarType::Symbol) {
            return Err(XlogError::Kernel(format!(
                "Column {} is {:?} (expected U32/Symbol for filter_u32)",
                col, col_type
            )));
        }

        // Get the filter column as u32 view
        let col_data = input
            .column(col)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col)))?;
        let col_view = self.column_as_u32_view(col_data, n)?;

        let block_size = 256u32;
        let num_blocks = (num_rows + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Fused compare + scan phase1.
        let d_mask = self.memory.alloc::<u8>(n)?;
        let d_prefix_sum = self.memory.alloc::<u32>(n)?;
        let mut d_block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

        let filter_scan_fn = device
            .get_func(
                FILTER_MODULE,
                filter_kernels::FILTER_COMPARE_U32_SCAN_PHASE1,
            )
            .ok_or_else(|| {
                XlogError::Kernel("filter_compare_u32_scan_phase1 kernel not found".to_string())
            })?;

        // SAFETY: filter_compare_u32_scan_phase1(column, constant, num_rows, op, mask, prefix_sum, block_sums)
        unsafe {
            filter_scan_fn.clone().launch(
                config,
                (
                    &col_view,
                    value,
                    num_rows,
                    op as u8,
                    &d_mask,
                    &d_prefix_sum,
                    &d_block_sums,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("filter_compare_u32_scan_phase1 failed: {}", e)))?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace(&mut d_block_sums, num_blocks)?;

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
                    (&d_prefix_sum, &d_block_sums, num_rows),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        let last_idx = (n as usize)
            .checked_sub(1)
            .ok_or_else(|| XlogError::Kernel("filter_u32: unexpected empty input".to_string()))?;
        let last_prefix_view = d_prefix_sum.slice(last_idx..(last_idx + 1));
        let last_mask_view = d_mask.slice(last_idx..(last_idx + 1));

        let mut last_prefix_host = [0u32];
        let mut last_mask_host = [0u8];
        device
            .dtoh_sync_copy_into(&last_prefix_view, &mut last_prefix_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read last prefix sum: {}", e)))?;
        device
            .dtoh_sync_copy_into(&last_mask_view, &mut last_mask_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read last mask: {}", e)))?;

        let output_count = (last_prefix_host[0] as u64) + (last_mask_host[0] as u64);
        if output_count == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }

        self.compact_buffer_by_device_mask(input, &d_mask, &d_prefix_sum, output_count)
    }

    /// Filter i32 column with comparison operator.
    pub fn filter_i32(
        &self,
        input: &CudaBuffer,
        col: usize,
        value: i32,
        op: CompareOp,
    ) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema.clone());
        }

        let mask = self.compare_const_mask::<i32>(
            input,
            col,
            value,
            op,
            &[ScalarType::I32],
            filter_kernels::FILTER_COMPARE_I32,
        )?;
        self.filter_by_device_mask(input, &mask)
    }

    /// Filter u64 column with comparison operator.
    pub fn filter_u64(
        &self,
        input: &CudaBuffer,
        col: usize,
        value: u64,
        op: CompareOp,
    ) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema.clone());
        }

        let mask = self.compare_const_mask::<u64>(
            input,
            col,
            value,
            op,
            &[ScalarType::U64],
            filter_kernels::FILTER_COMPARE_U64,
        )?;
        self.filter_by_device_mask(input, &mask)
    }

    /// Filter f32 column with comparison operator.
    pub fn filter_f32(
        &self,
        input: &CudaBuffer,
        col: usize,
        value: f32,
        op: CompareOp,
    ) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema.clone());
        }

        let mask = self.compare_const_mask::<f32>(
            input,
            col,
            value,
            op,
            &[ScalarType::F32],
            filter_kernels::FILTER_COMPARE_F32,
        )?;
        self.filter_by_device_mask(input, &mask)
    }

    /// Filter bool column with comparison operator.
    pub fn filter_bool(
        &self,
        input: &CudaBuffer,
        col: usize,
        value: bool,
        op: CompareOp,
    ) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema.clone());
        }

        let value_u8 = if value { 1u8 } else { 0u8 };
        let mask = self.compare_const_mask::<u8>(
            input,
            col,
            value_u8,
            op,
            &[ScalarType::Bool],
            filter_kernels::FILTER_COMPARE_U8,
        )?;
        self.filter_by_device_mask(input, &mask)
    }

    /// Compare two u32 columns and return a device mask.
    pub fn compare_columns_u32(
        &self,
        input: &CudaBuffer,
        left: usize,
        right: usize,
        op: CompareOp,
    ) -> Result<TrackedCudaSlice<u8>> {
        self.compare_columns_mask::<u32>(
            input,
            left,
            right,
            op,
            &[ScalarType::U32, ScalarType::Symbol],
            filter_kernels::FILTER_COMPARE_U32_COL,
        )
    }

    /// Compare two i32 columns and return a device mask.
    pub fn compare_columns_i32(
        &self,
        input: &CudaBuffer,
        left: usize,
        right: usize,
        op: CompareOp,
    ) -> Result<TrackedCudaSlice<u8>> {
        self.compare_columns_mask::<i32>(
            input,
            left,
            right,
            op,
            &[ScalarType::I32],
            filter_kernels::FILTER_COMPARE_I32_COL,
        )
    }

    /// Compare two i64 columns and return a device mask.
    pub fn compare_columns_i64(
        &self,
        input: &CudaBuffer,
        left: usize,
        right: usize,
        op: CompareOp,
    ) -> Result<TrackedCudaSlice<u8>> {
        self.compare_columns_mask::<i64>(
            input,
            left,
            right,
            op,
            &[ScalarType::I64],
            filter_kernels::FILTER_COMPARE_I64_COL,
        )
    }

    /// Compare two u64 columns and return a device mask.
    pub fn compare_columns_u64(
        &self,
        input: &CudaBuffer,
        left: usize,
        right: usize,
        op: CompareOp,
    ) -> Result<TrackedCudaSlice<u8>> {
        self.compare_columns_mask::<u64>(
            input,
            left,
            right,
            op,
            &[ScalarType::U64],
            filter_kernels::FILTER_COMPARE_U64_COL,
        )
    }

    /// Compare two f32 columns and return a device mask.
    pub fn compare_columns_f32(
        &self,
        input: &CudaBuffer,
        left: usize,
        right: usize,
        op: CompareOp,
    ) -> Result<TrackedCudaSlice<u8>> {
        self.compare_columns_mask::<f32>(
            input,
            left,
            right,
            op,
            &[ScalarType::F32],
            filter_kernels::FILTER_COMPARE_F32_COL,
        )
    }

    /// Compare two f64 columns and return a device mask.
    pub fn compare_columns_f64(
        &self,
        input: &CudaBuffer,
        left: usize,
        right: usize,
        op: CompareOp,
    ) -> Result<TrackedCudaSlice<u8>> {
        self.compare_columns_mask::<f64>(
            input,
            left,
            right,
            op,
            &[ScalarType::F64],
            filter_kernels::FILTER_COMPARE_F64_COL,
        )
    }

    /// Compare two u8 columns and return a device mask.
    pub fn compare_columns_u8(
        &self,
        input: &CudaBuffer,
        left: usize,
        right: usize,
        op: CompareOp,
    ) -> Result<TrackedCudaSlice<u8>> {
        self.compare_columns_mask::<u8>(
            input,
            left,
            right,
            op,
            &[ScalarType::Bool],
            filter_kernels::FILTER_COMPARE_U8_COL,
        )
    }

    fn compare_const_mask<T: DeviceRepr + Copy>(
        &self,
        input: &CudaBuffer,
        col: usize,
        value: T,
        op: CompareOp,
        allowed_types: &[ScalarType],
        kernel: &str,
    ) -> Result<TrackedCudaSlice<u8>> {
        if input.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Filter supports at most {} rows, got {}",
                u32::MAX,
                input.num_rows()
            )));
        }
        if col >= input.arity() {
            return Err(XlogError::Kernel(format!(
                "Column index {} out of bounds (arity {})",
                col,
                input.arity()
            )));
        }

        if input.is_empty() {
            return Ok(self.memory.alloc::<u8>(0)?);
        }

        let col_type = input
            .schema()
            .column_type(col)
            .ok_or_else(|| XlogError::Kernel("Missing column type".into()))?;
        if !allowed_types.contains(&col_type) {
            return Err(XlogError::Kernel(format!(
                "Column {} is {:?} (expected {:?})",
                col, col_type, allowed_types
            )));
        }

        let num_rows = input.num_rows() as u32;
        let expected_bytes = (num_rows as usize)
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| XlogError::Kernel("filter compare size overflow".into()))?;
        let col_data = input
            .column(col)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col)))?;
        if col_data.num_bytes() != expected_bytes {
            return Err(XlogError::Kernel(format!(
                "Column {} has {} bytes but expected {} for {} rows",
                col,
                col_data.num_bytes(),
                expected_bytes,
                input.num_rows()
            )));
        }

        let block_size = 256u32;
        let num_blocks = (num_rows + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut d_mask = self.memory.alloc::<u8>(num_rows as usize)?;
        let func = self
            .device
            .inner()
            .get_func(FILTER_MODULE, kernel)
            .ok_or_else(|| XlogError::Kernel("filter compare kernel not found".into()))?;

        unsafe { func.clone().launch(config, (col_data, value, num_rows, op as u8, &mut d_mask)) }
            .map_err(|e| XlogError::Kernel(format!("filter compare failed: {}", e)))?;

        Ok(d_mask)
    }

    fn compare_columns_mask<T: DeviceRepr>(
        &self,
        input: &CudaBuffer,
        left: usize,
        right: usize,
        op: CompareOp,
        allowed_types: &[ScalarType],
        kernel: &str,
    ) -> Result<TrackedCudaSlice<u8>> {
        if input.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Filter supports at most {} rows, got {}",
                u32::MAX,
                input.num_rows()
            )));
        }
        if left >= input.arity() || right >= input.arity() {
            return Err(XlogError::Kernel(format!(
                "Column indices {} or {} out of bounds (arity {})",
                left,
                right,
                input.arity()
            )));
        }

        if input.is_empty() {
            return Ok(self.memory.alloc::<u8>(0)?);
        }

        let left_type = input
            .schema()
            .column_type(left)
            .ok_or_else(|| XlogError::Kernel("Missing left column type".into()))?;
        let right_type = input
            .schema()
            .column_type(right)
            .ok_or_else(|| XlogError::Kernel("Missing right column type".into()))?;

        if left_type != right_type {
            return Err(XlogError::Kernel(
                "Column-column compare requires matching types".into(),
            ));
        }
        if !allowed_types.contains(&left_type) {
            return Err(XlogError::Kernel(format!(
                "Column type {:?} not supported for compare",
                left_type
            )));
        }

        let num_rows = input.num_rows() as u32;
        let expected_bytes = (num_rows as usize)
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| XlogError::Kernel("compare columns size overflow".into()))?;

        let left_col = input
            .column(left)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", left)))?;
        let right_col = input
            .column(right)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", right)))?;

        if left_col.num_bytes() != expected_bytes || right_col.num_bytes() != expected_bytes {
            return Err(XlogError::Kernel(format!(
                "Compare columns expect {} bytes per column for {} rows",
                expected_bytes,
                input.num_rows()
            )));
        }

        let block_size = 256u32;
        let num_blocks = (num_rows + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut d_mask = self.memory.alloc::<u8>(num_rows as usize)?;
        let func = self
            .device
            .inner()
            .get_func(FILTER_MODULE, kernel)
            .ok_or_else(|| XlogError::Kernel("filter compare kernel not found".into()))?;

        unsafe {
            func.clone()
                .launch(config, (left_col, right_col, num_rows, op as u8, &mut d_mask))
        }
        .map_err(|e| XlogError::Kernel(format!("filter compare failed: {}", e)))?;

        Ok(d_mask)
    }

    /// Filter buffer where f64 column equals constant
    ///
    /// # Arguments
    /// * `input` - The input buffer to filter
    /// * `col` - Column index to filter on
    /// * `value` - Value to compare against
    ///
    /// # Returns
    /// A new buffer containing only rows where `input[col] == value`
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails or column type is not F64
    pub fn filter_f64_eq(&self, input: &CudaBuffer, col: usize, value: f64) -> Result<CudaBuffer> {
        self.filter_f64(input, col, value, CompareOp::Eq)
    }

    /// Filter buffer where f64 column is greater than constant
    ///
    /// # Arguments
    /// * `input` - The input buffer to filter
    /// * `col` - Column index to filter on
    /// * `value` - Value to compare against
    ///
    /// # Returns
    /// A new buffer containing only rows where `input[col] > value`
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails or column type is not F64
    pub fn filter_f64_gt(&self, input: &CudaBuffer, col: usize, value: f64) -> Result<CudaBuffer> {
        self.filter_f64(input, col, value, CompareOp::Gt)
    }

    /// Filter buffer where f64 column is less than constant
    ///
    /// # Arguments
    /// * `input` - The input buffer to filter
    /// * `col` - Column index to filter on
    /// * `value` - Value to compare against
    ///
    /// # Returns
    /// A new buffer containing only rows where `input[col] < value`
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails or column type is not F64
    pub fn filter_f64_lt(&self, input: &CudaBuffer, col: usize, value: f64) -> Result<CudaBuffer> {
        self.filter_f64(input, col, value, CompareOp::Lt)
    }

    /// Filter f64 column with comparison operator.
    ///
    /// # Arguments
    /// * `input` - Buffer to filter
    /// * `col` - Column index to compare
    /// * `value` - Constant to compare against
    /// * `op` - Comparison operator
    ///
    /// # Errors
    /// Returns error if column index is out of bounds or column type is not F64.
    pub fn filter_f64(
        &self,
        input: &CudaBuffer,
        col: usize,
        value: f64,
        op: CompareOp,
    ) -> Result<CudaBuffer> {
        if input.num_rows == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }

        if input.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "filter_f64 supports at most {} rows, got {}",
                u32::MAX,
                input.num_rows()
            )));
        }

        let n = input.num_rows as usize;
        let num_rows = input.num_rows as u32;
        let device = self.device.inner();

        // Validate column index
        if col >= input.arity() {
            return Err(XlogError::Kernel(format!(
                "Column index {} out of bounds (arity {})",
                col,
                input.arity()
            )));
        }

        // Validate column type is F64
        if input.schema().column_type(col) != Some(ScalarType::F64) {
            return Err(XlogError::Kernel(format!(
                "Column {} is not F64 (expected F64 for filter_f64)",
                col
            )));
        }

        // Get the filter column as f64 view
        let col_data = input
            .column(col)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col)))?;
        let col_view = self.column_as_f64_view(col_data, n)?;

        let block_size = 256u32;
        let num_blocks = (num_rows + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Fused compare + scan phase1.
        let d_mask = self.memory.alloc::<u8>(n)?;
        let d_prefix_sum = self.memory.alloc::<u32>(n)?;
        let mut d_block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

        let filter_scan_fn = device
            .get_func(
                FILTER_MODULE,
                filter_kernels::FILTER_COMPARE_F64_SCAN_PHASE1,
            )
            .ok_or_else(|| {
                XlogError::Kernel("filter_compare_f64_scan_phase1 kernel not found".to_string())
            })?;

        // SAFETY: filter_compare_f64_scan_phase1(column, constant, num_rows, op, mask, prefix_sum, block_sums)
        unsafe {
            filter_scan_fn.clone().launch(
                config,
                (
                    &col_view,
                    value,
                    num_rows,
                    op as u8,
                    &d_mask,
                    &d_prefix_sum,
                    &d_block_sums,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("filter_compare_f64_scan_phase1 failed: {}", e)))?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace(&mut d_block_sums, num_blocks)?;

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
                    (&d_prefix_sum, &d_block_sums, num_rows),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        let last_idx = (n as usize)
            .checked_sub(1)
            .ok_or_else(|| XlogError::Kernel("filter_f64: unexpected empty input".to_string()))?;
        let last_prefix_view = d_prefix_sum.slice(last_idx..(last_idx + 1));
        let last_mask_view = d_mask.slice(last_idx..(last_idx + 1));

        let mut last_prefix_host = [0u32];
        let mut last_mask_host = [0u8];
        device
            .dtoh_sync_copy_into(&last_prefix_view, &mut last_prefix_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read last prefix sum: {}", e)))?;
        device
            .dtoh_sync_copy_into(&last_mask_view, &mut last_mask_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read last mask: {}", e)))?;

        let output_count = (last_prefix_host[0] as u64) + (last_mask_host[0] as u64);
        if output_count == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }

        self.compact_buffer_by_device_mask(input, &d_mask, &d_prefix_sum, output_count)
    }

    /// Filter buffer by pre-computed mask.
    ///
    /// # Arguments
    /// * `input` - The input buffer to filter
    /// * `mask` - Mask slice where non-zero means keep the row
    ///
    /// # Errors
    /// Returns error if mask length doesn't match buffer rows.
    pub fn filter_by_mask(&self, input: &CudaBuffer, mask: &[u8]) -> Result<CudaBuffer> {
        if input.num_rows == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }

        let n = input.num_rows as usize;
        if mask.len() != n {
            return Err(XlogError::Kernel(format!(
                "Mask length {} doesn't match buffer rows {}",
                mask.len(),
                n
            )));
        }

        // Compute prefix sum and count
        let (prefix_sum, count) = self.prefix_sum_mask(mask)?;

        if count == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }

        // Compact all columns using mask
        self.compact_buffer_by_mask(input, mask, &prefix_sum, count as u64)
    }

    /// Compact buffer columns using mask and prefix sum indices
    fn compact_buffer_by_mask(
        &self,
        input: &CudaBuffer,
        mask: &[u8],
        prefix_sum: &[u32],
        output_count: u64,
    ) -> Result<CudaBuffer> {
        let device = self.device.inner();

        // Upload mask and prefix sum to GPU
        let d_mask = device
            .htod_sync_copy(mask)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload mask: {}", e)))?;
        let d_prefix_sum = device
            .htod_sync_copy(prefix_sum)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload prefix_sum: {}", e)))?;

        self.compact_buffer_by_device_mask(input, &d_mask, &d_prefix_sum, output_count)
    }

    fn compact_buffer_by_device_mask(
        &self,
        input: &CudaBuffer,
        d_mask: &cudarc::driver::CudaSlice<u8>,
        d_prefix_sum: &cudarc::driver::CudaSlice<u32>,
        output_count: u64,
    ) -> Result<CudaBuffer> {
        let n = input.num_rows as u32;
        let device = self.device.inner();

        // Get compact kernel
        let compact_fn = device
            .get_func(FILTER_MODULE, filter_kernels::COMPACT_BYTES_BY_MASK)
            .ok_or_else(|| {
                XlogError::Kernel("compact_bytes_by_mask kernel not found".to_string())
            })?;

        let block_size = 256u32;
        let grid_size = (n + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Compact each column
        let mut new_columns = Vec::with_capacity(input.columns.len());
        for col_idx in 0..input.columns.len() {
            let src_col = input
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

            let elem_size = input
                .schema
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4) as u32;

            let output_bytes = (output_count as usize) * (elem_size as usize);
            let dst_col = self.memory.alloc::<u8>(output_bytes)?;

            // SAFETY: Kernel signature matches:
            // compact_bytes_by_mask(input, mask, prefix_sum, num_rows, elem_size, output)
            unsafe {
                compact_fn.clone().launch(
                    config,
                    (src_col, d_mask, d_prefix_sum, n, elem_size, &dst_col),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("compact_bytes_by_mask failed: {}", e)))?;

            new_columns.push(dst_col.into());
        }

        self.device.synchronize()?;

        Ok(CudaBuffer {
            columns: new_columns,
            num_rows: output_count,
            schema: input.schema.clone(),
        })
    }

    pub fn filter_by_device_mask(
        &self,
        input: &CudaBuffer,
        d_mask: &cudarc::driver::CudaSlice<u8>,
    ) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema().clone());
        }

        if input.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Device-mask filtering supports at most {} rows, got {}",
                u32::MAX,
                input.num_rows()
            )));
        }

        let n = input.num_rows() as u32;
        let device = self.device.inner();

        let block_size = 256u32;
        let num_blocks = (n + block_size - 1) / block_size;

        let d_prefix_sum = self.memory.alloc::<u32>(n as usize)?;
        let mut d_block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

        let phase1_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase1 kernel".to_string())
            })?;

        // SAFETY: multiblock_scan_phase1(const uint8_t* mask, uint32_t* prefix_sum, uint32_t* block_sums, uint32_t n)
        unsafe {
            phase1_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (d_mask, &d_prefix_sum, &d_block_sums, n),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase1 failed: {}", e)))?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace(&mut d_block_sums, num_blocks)?;

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
                    (&d_prefix_sum, &d_block_sums, n),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        let last_idx = (n as usize).checked_sub(1).ok_or_else(|| {
            XlogError::Kernel("Device-mask filtering: unexpected empty input".to_string())
        })?;

        let last_prefix_view = d_prefix_sum.slice(last_idx..(last_idx + 1));
        let last_mask_view = d_mask.slice(last_idx..(last_idx + 1));

        let mut last_prefix_host = [0u32];
        let mut last_mask_host = [0u8];
        device
            .dtoh_sync_copy_into(&last_prefix_view, &mut last_prefix_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read last prefix sum: {}", e)))?;
        device
            .dtoh_sync_copy_into(&last_mask_view, &mut last_mask_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read last mask: {}", e)))?;

        let output_count = (last_prefix_host[0] as u64) + (last_mask_host[0] as u64);

        if output_count == 0 {
            return self.create_empty_buffer(input.schema().clone());
        }

        self.compact_buffer_by_device_mask(input, d_mask, &d_prefix_sum, output_count)
    }

    // ============== Buffer Helper Methods ==============

    /// Create a CudaBuffer from a slice of u32 values (single column)
    ///
    /// # Arguments
    /// * `data` - The u32 values to upload
    /// * `schema` - The schema for the buffer (should have one U32 column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the data
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if upload fails
    pub fn create_buffer_from_u32_slice(&self, data: &[u32], schema: Schema) -> Result<CudaBuffer> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut col = self.memory.alloc::<u8>(bytes.len())?;
        self.device
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload data: {}", e)))?;

        Ok(CudaBuffer::from_columns(
            vec![col.into()],
            data.len() as u64,
            schema,
        ))
    }

    /// Create a CudaBuffer from multiple u32 column slices
    ///
    /// # Arguments
    /// * `columns` - Slice of column data slices (each column as &[u32])
    /// * `schema` - The schema for the buffer
    ///
    /// # Returns
    /// A new CudaBuffer containing all columns
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if upload fails or columns have mismatched lengths
    pub fn create_buffer_from_u32_columns(
        &self,
        columns: &[&[u32]],
        schema: Schema,
    ) -> Result<CudaBuffer> {
        if columns.is_empty() {
            return self.create_empty_buffer(schema);
        }

        let num_rows = columns[0].len();
        for (i, col) in columns.iter().enumerate() {
            if col.len() != num_rows {
                return Err(XlogError::Kernel(format!(
                    "Column {} has {} rows but expected {}",
                    i,
                    col.len(),
                    num_rows
                )));
            }
        }

        let mut cuda_columns = Vec::with_capacity(columns.len());
        for col_data in columns {
            let bytes: Vec<u8> = col_data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut col = self.memory.alloc::<u8>(bytes.len())?;
            self.device
                .inner()
                .htod_sync_copy_into(&bytes, &mut col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload column: {}", e)))?;
            cuda_columns.push(col.into());
        }

        Ok(CudaBuffer::from_columns(
            cuda_columns,
            num_rows as u64,
            schema,
        ))
    }

    /// Create a CudaBuffer from a u64 slice (single column)
    ///
    /// # Arguments
    /// * `data` - The u64 data slice
    /// * `schema` - The schema for the buffer
    ///
    /// # Returns
    /// A new CudaBuffer containing the data as a single column
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if upload fails
    pub fn create_buffer_from_u64_slice(&self, data: &[u64], schema: Schema) -> Result<CudaBuffer> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut col = self.memory.alloc::<u8>(bytes.len())?;
        self.device
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload data: {}", e)))?;

        Ok(CudaBuffer::from_columns(
            vec![col.into()],
            data.len() as u64,
            schema,
        ))
    }

    /// Create a CudaBuffer from an i64 slice (single column)
    ///
    /// # Arguments
    /// * `data` - The i64 data slice
    /// * `schema` - The schema for the buffer
    ///
    /// # Returns
    /// A new CudaBuffer containing the data as a single column
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if upload fails
    pub fn create_buffer_from_i64_slice(&self, data: &[i64], schema: Schema) -> Result<CudaBuffer> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut col = self.memory.alloc::<u8>(bytes.len())?;
        self.device
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload data: {}", e)))?;

        Ok(CudaBuffer::from_columns(
            vec![col.into()],
            data.len() as u64,
            schema,
        ))
    }

    /// Create a CudaBuffer from an i32 slice (single column)
    ///
    /// # Arguments
    /// * `data` - The i32 data slice
    /// * `schema` - The schema for the buffer
    ///
    /// # Returns
    /// A new CudaBuffer containing the data as a single column
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if upload fails
    pub fn create_buffer_from_i32_slice(&self, data: &[i32], schema: Schema) -> Result<CudaBuffer> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut col = self.memory.alloc::<u8>(bytes.len())?;
        self.device
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload data: {}", e)))?;

        Ok(CudaBuffer::from_columns(
            vec![col.into()],
            data.len() as u64,
            schema,
        ))
    }

    /// Create a CudaBuffer from an f64 slice (single column)
    ///
    /// # Arguments
    /// * `data` - The f64 data slice
    /// * `schema` - The schema for the buffer
    ///
    /// # Returns
    /// A new CudaBuffer containing the data as a single column
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if upload fails
    pub fn create_buffer_from_f64_slice(&self, data: &[f64], schema: Schema) -> Result<CudaBuffer> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut col = self.memory.alloc::<u8>(bytes.len())?;
        self.device
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload data: {}", e)))?;

        Ok(CudaBuffer::from_columns(
            vec![col.into()],
            data.len() as u64,
            schema,
        ))
    }

    /// Create a CudaBuffer from an f32 slice (single column)
    ///
    /// # Arguments
    /// * `data` - The f32 data slice
    /// * `schema` - The schema for the buffer
    ///
    /// # Returns
    /// A new CudaBuffer containing the data as a single column
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if upload fails
    pub fn create_buffer_from_f32_slice(&self, data: &[f32], schema: Schema) -> Result<CudaBuffer> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut col = self.memory.alloc::<u8>(bytes.len())?;
        self.device
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload data: {}", e)))?;

        Ok(CudaBuffer::from_columns(
            vec![col.into()],
            data.len() as u64,
            schema,
        ))
    }

    /// Create a CudaBuffer from a u8 slice (single column)
    ///
    /// # Arguments
    /// * `data` - The u8 data slice
    /// * `schema` - The schema for the buffer
    ///
    /// # Returns
    /// A new CudaBuffer containing the data as a single column
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if upload fails
    pub fn create_buffer_from_u8_slice(&self, data: &[u8], schema: Schema) -> Result<CudaBuffer> {
        let mut col = self.memory.alloc::<u8>(data.len())?;
        self.device
            .inner()
            .htod_sync_copy_into(data, &mut col)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload data: {}", e)))?;

        Ok(CudaBuffer::from_columns(
            vec![col.into()],
            data.len() as u64,
            schema,
        ))
    }

    /// Create a buffer from multiple column slices (raw bytes)
    ///
    /// This is a generic version that works with any column type by accepting
    /// raw byte slices. Each slice should contain the column data in little-endian
    /// format with the correct size for the column's type.
    ///
    /// # Arguments
    /// * `slices` - Slice of raw byte slices, one per column
    /// * `schema` - The schema for the buffer
    ///
    /// # Returns
    /// A new CudaBuffer containing all columns
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Number of slices doesn't match schema arity
    /// - Upload fails
    pub fn create_buffer_from_slices(
        &self,
        slices: &[&[u8]],
        schema: Schema,
    ) -> Result<CudaBuffer> {
        if slices.len() != schema.arity() {
            return Err(XlogError::Kernel(format!(
                "Slice count {} doesn't match schema arity {}",
                slices.len(),
                schema.arity()
            )));
        }

        if slices.is_empty() {
            return self.create_empty_buffer(schema);
        }

        let first_col_size = schema.column_type(0).map(|t| t.size_bytes()).unwrap_or(4);
        let num_rows = slices[0].len() / first_col_size;

        // Validate that all columns have consistent row counts
        for (i, slice) in slices.iter().enumerate() {
            let col_size = schema.column_type(i).map(|t| t.size_bytes()).unwrap_or(4);
            let col_rows = slice.len() / col_size;
            if col_rows != num_rows {
                return Err(XlogError::Kernel(format!(
                    "Column {} has {} rows but expected {} rows (based on first column)",
                    i, col_rows, num_rows
                )));
            }
            // Also verify the slice length is exactly divisible by the type size
            if slice.len() % col_size != 0 {
                return Err(XlogError::Kernel(format!(
                    "Column {} slice length {} is not divisible by type size {}",
                    i,
                    slice.len(),
                    col_size
                )));
            }
        }

        let mut columns = Vec::with_capacity(slices.len());

        for (i, slice) in slices.iter().enumerate() {
            let mut col = self.memory.alloc::<u8>(slice.len())?;
            self.device
                .inner()
                .htod_sync_copy_into(slice, &mut col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload column {}: {}", i, e)))?;
            columns.push(col.into());
        }

        Ok(CudaBuffer::from_columns(columns, num_rows as u64, schema))
    }

    /// Export CudaBuffer to Arrow RecordBatch
    ///
    /// Downloads data from GPU and converts it to an Arrow RecordBatch for
    /// interoperability with Arrow-based tools like cuDF, Polars, or DuckDB.
    ///
    /// # Arguments
    /// * `buffer` - The CudaBuffer to export
    ///
    /// # Returns
    /// An Arrow RecordBatch containing all columns from the buffer
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Column download fails
    /// - RecordBatch creation fails
    pub fn to_arrow_record_batch(
        &self,
        buffer: &CudaBuffer,
    ) -> Result<arrow::record_batch::RecordBatch> {
        use arrow::array::*;
        use arrow::datatypes::{Field, Schema as ArrowSchema};

        let num_rows = buffer.num_rows as usize;

        let fields: Vec<Field> = buffer
            .schema
            .columns
            .iter()
            .map(|(name, scalar_type)| Field::new(name, scalar_type.to_arrow_type(), false))
            .collect();
        let arrow_schema = Arc::new(ArrowSchema::new(fields));

        let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(buffer.arity());

        for (col_idx, (_, scalar_type)) in buffer.schema.columns.iter().enumerate() {
            let col = buffer
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

            // Handle empty buffer case
            if num_rows == 0 {
                let array: Arc<dyn Array> = match scalar_type {
                    ScalarType::Bool => Arc::new(BooleanArray::from(Vec::<bool>::new())),
                    ScalarType::U32 => Arc::new(UInt32Array::from(Vec::<u32>::new())),
                    ScalarType::Symbol => Arc::new(xlog_core::symbol::to_arrow(&[])),
                    ScalarType::I32 => Arc::new(Int32Array::from(Vec::<i32>::new())),
                    ScalarType::U64 => Arc::new(UInt64Array::from(Vec::<u64>::new())),
                    ScalarType::I64 => Arc::new(Int64Array::from(Vec::<i64>::new())),
                    ScalarType::F32 => Arc::new(Float32Array::from(Vec::<f32>::new())),
                    ScalarType::F64 => Arc::new(Float64Array::from(Vec::<f64>::new())),
                };
                arrays.push(array);
                continue;
            }

            let mut bytes = vec![0u8; col.len()];
            self.device
                .inner()
                .dtoh_sync_copy_into(col, &mut bytes)
                .map_err(|e| XlogError::Kernel(format!("Failed to download column: {}", e)))?;

            let array: Arc<dyn Array> = match scalar_type {
                ScalarType::Bool => Arc::new(BooleanArray::from(
                    bytes.iter().map(|&b| b != 0).collect::<Vec<_>>(),
                )),
                ScalarType::U32 => {
                    let values: Vec<u32> = bytes
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Arc::new(UInt32Array::from(values))
                }
                ScalarType::Symbol => {
                    let values: Vec<u32> = bytes
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Arc::new(xlog_core::symbol::to_arrow(&values))
                }
                ScalarType::I32 => {
                    let values: Vec<i32> = bytes
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Arc::new(Int32Array::from(values))
                }
                ScalarType::U64 => {
                    let values: Vec<u64> = bytes
                        .chunks_exact(8)
                        .map(|c| {
                            u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect();
                    Arc::new(UInt64Array::from(values))
                }
                ScalarType::I64 => {
                    let values: Vec<i64> = bytes
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect();
                    Arc::new(Int64Array::from(values))
                }
                ScalarType::F32 => {
                    let values: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Arc::new(Float32Array::from(values))
                }
                ScalarType::F64 => {
                    let values: Vec<f64> = bytes
                        .chunks_exact(8)
                        .map(|c| {
                            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect();
                    Arc::new(Float64Array::from(values))
                }
            };

            arrays.push(array);
        }

        arrow::record_batch::RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| XlogError::Kernel(format!("Failed to create RecordBatch: {}", e)))
    }

    /// Import Arrow RecordBatch to CudaBuffer
    ///
    /// Uploads Arrow data to GPU memory.
    ///
    /// # Arguments
    /// * `record_batch` - The Arrow RecordBatch to import
    ///
    /// # Returns
    /// A new CudaBuffer with the data on GPU
    ///
    /// # Errors
    /// Returns error if Arrow type is not supported or upload fails
    pub fn from_arrow_record_batch(
        &self,
        record_batch: &arrow::record_batch::RecordBatch,
    ) -> Result<CudaBuffer> {
        use arrow::array::*;
        use arrow::datatypes::DataType;

        let num_rows = record_batch.num_rows() as u64;

        if num_rows == 0 {
            let columns: Vec<(String, ScalarType)> = record_batch
                .schema()
                .fields()
                .iter()
                .filter_map(|f| {
                    ScalarType::from_arrow_type(f.data_type()).map(|st| (f.name().clone(), st))
                })
                .collect();
            return self.create_empty_buffer(Schema::new(columns));
        }

        let device = self.device.inner();
        let mut columns = Vec::with_capacity(record_batch.num_columns());
        let mut schema_cols = Vec::with_capacity(record_batch.num_columns());

        for (col_idx, field) in record_batch.schema().fields().iter().enumerate() {
            let array = record_batch.column(col_idx);
            let scalar_type = ScalarType::from_arrow_type(field.data_type()).ok_or_else(|| {
                XlogError::Kernel(format!("Unsupported Arrow type: {:?}", field.data_type()))
            })?;

            let bytes: Vec<u8> = match field.data_type() {
                DataType::Boolean => {
                    let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                    arr.iter()
                        .map(|v| if v.unwrap_or(false) { 1u8 } else { 0u8 })
                        .collect()
                }
                DataType::UInt32 => {
                    let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
                    arr.values().iter().flat_map(|v| v.to_le_bytes()).collect()
                }
                DataType::Int32 => {
                    let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
                    arr.values().iter().flat_map(|v| v.to_le_bytes()).collect()
                }
                DataType::UInt64 => {
                    let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
                    arr.values().iter().flat_map(|v| v.to_le_bytes()).collect()
                }
                DataType::Int64 => {
                    let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
                    arr.values().iter().flat_map(|v| v.to_le_bytes()).collect()
                }
                DataType::Float32 => {
                    let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
                    arr.values().iter().flat_map(|v| v.to_le_bytes()).collect()
                }
                DataType::Float64 => {
                    let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
                    arr.values().iter().flat_map(|v| v.to_le_bytes()).collect()
                }
                _ => {
                    return Err(XlogError::Kernel(format!(
                        "Unsupported Arrow type: {:?}",
                        field.data_type()
                    )))
                }
            };

            let mut d_col = self.memory.alloc::<u8>(bytes.len())?;
            device
                .htod_sync_copy_into(&bytes, &mut d_col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload column: {}", e)))?;

            columns.push(d_col.into());
            schema_cols.push((field.name().clone(), scalar_type));
        }

        Ok(CudaBuffer::from_columns(
            columns,
            num_rows,
            Schema::new(schema_cols),
        ))
    }

    /// Export a CudaBuffer to an Arrow IPC stream (RecordBatchStream) as bytes.
    ///
    /// This is a convenience wrapper around `to_arrow_record_batch` that enables
    /// interoperability with tools like cuDF via standard Arrow IPC readers.
    ///
    /// Note: This is not zero-copy; data is downloaded from GPU to host memory.
    pub fn to_arrow_ipc_stream(&self, buffer: &CudaBuffer) -> Result<Vec<u8>> {
        use arrow::ipc::writer::StreamWriter;

        let batch = self.to_arrow_record_batch(buffer)?;
        let mut out = Vec::new();
        let mut writer = StreamWriter::try_new(&mut out, &batch.schema())
            .map_err(|e| XlogError::Kernel(format!("Failed to create Arrow IPC writer: {}", e)))?;
        writer
            .write(&batch)
            .map_err(|e| XlogError::Kernel(format!("Failed to write Arrow RecordBatch: {}", e)))?;
        writer
            .finish()
            .map_err(|e| XlogError::Kernel(format!("Failed to finish Arrow IPC stream: {}", e)))?;
        Ok(out)
    }

    /// Import a single-batch Arrow IPC stream (RecordBatchStream) into a CudaBuffer.
    ///
    /// Note: This uploads Arrow data from host to GPU memory.
    pub fn from_arrow_ipc_stream(&self, ipc: &[u8]) -> Result<CudaBuffer> {
        use arrow::ipc::reader::StreamReader;
        use std::io::Cursor;

        let cursor = Cursor::new(ipc);
        let mut reader = StreamReader::try_new(cursor, None)
            .map_err(|e| XlogError::Kernel(format!("Failed to create Arrow IPC reader: {}", e)))?;

        let batches = reader.by_ref();
        let first = batches
            .next()
            .ok_or_else(|| {
                XlogError::Kernel("Arrow IPC stream contained no record batches".to_string())
            })?
            .map_err(|e| XlogError::Kernel(format!("Failed to read Arrow RecordBatch: {}", e)))?;

        if batches.next().is_some() {
            return Err(XlogError::Kernel(
                "Arrow IPC stream contains multiple record batches; this API expects exactly one"
                    .to_string(),
            ));
        }

        self.from_arrow_record_batch(&first)
    }

    /// Write a CudaBuffer to a file as an Arrow IPC stream (RecordBatchStream).
    pub fn write_arrow_ipc_stream_file<P: AsRef<std::path::Path>>(
        &self,
        buffer: &CudaBuffer,
        path: P,
    ) -> Result<()> {
        let bytes = self.to_arrow_ipc_stream(buffer)?;
        std::fs::write(&path, bytes).map_err(|e| {
            XlogError::Kernel(format!(
                "Failed to write Arrow IPC stream to {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        Ok(())
    }

    /// Read a CudaBuffer from a file containing an Arrow IPC stream (RecordBatchStream).
    pub fn read_arrow_ipc_stream_file<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<CudaBuffer> {
        let bytes = std::fs::read(&path).map_err(|e| {
            XlogError::Kernel(format!(
                "Failed to read Arrow IPC stream from {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        self.from_arrow_ipc_stream(&bytes)
    }

    /// Download a column from a CudaBuffer as Vec<u32>
    ///
    /// # Arguments
    /// * `buffer` - The buffer to download from
    /// * `col_idx` - The column index to download
    ///
    /// # Returns
    /// A Vec<u32> containing the column data
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Column index is out of bounds
    /// - Download fails
    pub fn download_column_u32(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<u32>> {
        let col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

        if buffer.num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = (buffer.num_rows as usize) * std::mem::size_of::<u32>();
        if col.num_bytes() != num_bytes {
            return Err(XlogError::Kernel(format!(
                "Column {} has {} bytes but expected {} for {} rows",
                col_idx,
                col.num_bytes(),
                num_bytes,
                buffer.num_rows
            )));
        }
        let mut bytes = vec![0u8; num_bytes];
        self.device
            .inner()
            .dtoh_sync_copy_into(col, &mut bytes)
            .map_err(|e| XlogError::Kernel(format!("Failed to download column: {}", e)))?;

        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Download a column from a CudaBuffer as Vec<u8>
    ///
    /// # Arguments
    /// * `buffer` - The buffer to download from
    /// * `col_idx` - The column index to download
    ///
    /// # Returns
    /// A Vec<u8> containing the column data
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Column index is out of bounds
    /// - Download fails
    pub fn download_column_u8(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<u8>> {
        let col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

        if buffer.num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = buffer.num_rows as usize;
        if col.num_bytes() != num_bytes {
            return Err(XlogError::Kernel(format!(
                "Column {} has {} bytes but expected {} for {} rows",
                col_idx,
                col.num_bytes(),
                num_bytes,
                buffer.num_rows
            )));
        }

        let mut bytes = vec![0u8; num_bytes];
        self.device
            .inner()
            .dtoh_sync_copy_into(col, &mut bytes)
            .map_err(|e| XlogError::Kernel(format!("Failed to download column: {}", e)))?;

        Ok(bytes)
    }

    /// Download a column from GPU memory as u64 values
    ///
    /// # Arguments
    /// * `buffer` - The CudaBuffer containing the column
    /// * `col_idx` - Index of the column to download
    ///
    /// # Returns
    /// A Vec<u64> containing the column data
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Column index is out of bounds
    /// - Download fails
    pub fn download_column_u64(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<u64>> {
        let col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

        if buffer.num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = (buffer.num_rows as usize) * std::mem::size_of::<u64>();
        if col.num_bytes() != num_bytes {
            return Err(XlogError::Kernel(format!(
                "Column {} has {} bytes but expected {} for {} rows",
                col_idx,
                col.num_bytes(),
                num_bytes,
                buffer.num_rows
            )));
        }
        let mut bytes = vec![0u8; num_bytes];
        self.device
            .inner()
            .dtoh_sync_copy_into(col, &mut bytes)
            .map_err(|e| XlogError::Kernel(format!("Failed to download column: {}", e)))?;

        Ok(bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    /// Download an F64 column from GPU to host memory
    ///
    /// # Arguments
    /// * `buffer` - The CudaBuffer containing the column
    /// * `col_idx` - The column index to download
    ///
    /// # Returns
    /// A Vec<f64> containing the column data
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Column index is out of bounds
    /// - Download fails
    pub fn download_column_f64(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<f64>> {
        let col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

        if buffer.num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = (buffer.num_rows as usize) * std::mem::size_of::<f64>();
        if col.num_bytes() != num_bytes {
            return Err(XlogError::Kernel(format!(
                "Column {} has {} bytes but expected {} for {} rows",
                col_idx,
                col.num_bytes(),
                num_bytes,
                buffer.num_rows
            )));
        }
        let mut bytes = vec![0u8; num_bytes];
        self.device
            .inner()
            .dtoh_sync_copy_into(col, &mut bytes)
            .map_err(|e| XlogError::Kernel(format!("Failed to download column: {}", e)))?;

        Ok(bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    /// Download a Bool column from GPU to host memory
    ///
    /// # Arguments
    /// * `buffer` - The CudaBuffer containing the column
    /// * `col_idx` - The column index to download
    ///
    /// # Returns
    /// A Vec<bool> containing the column data
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Column index is out of bounds
    /// - Download fails
    pub fn download_column_bool(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<bool>> {
        let col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

        if buffer.num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = buffer.num_rows as usize;
        if col.num_bytes() != num_bytes {
            return Err(XlogError::Kernel(format!(
                "Column {} has {} bytes but expected {} for {} rows",
                col_idx,
                col.num_bytes(),
                num_bytes,
                buffer.num_rows
            )));
        }
        let mut bytes = vec![0u8; num_bytes];
        self.device
            .inner()
            .dtoh_sync_copy_into(col, &mut bytes)
            .map_err(|e| XlogError::Kernel(format!("Failed to download column: {}", e)))?;

        Ok(bytes.into_iter().map(|b| b != 0).collect())
    }

    /// Download an I32 column from GPU to host memory
    ///
    /// # Arguments
    /// * `buffer` - The CudaBuffer containing the column
    /// * `col_idx` - The column index to download
    ///
    /// # Returns
    /// A Vec<i32> containing the column data
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Column index is out of bounds
    /// - Download fails
    pub fn download_column_i32(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<i32>> {
        let col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

        if buffer.num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = (buffer.num_rows as usize) * std::mem::size_of::<i32>();
        if col.num_bytes() != num_bytes {
            return Err(XlogError::Kernel(format!(
                "Column {} has {} bytes but expected {} for {} rows",
                col_idx,
                col.num_bytes(),
                num_bytes,
                buffer.num_rows
            )));
        }
        let mut bytes = vec![0u8; num_bytes];
        self.device
            .inner()
            .dtoh_sync_copy_into(col, &mut bytes)
            .map_err(|e| XlogError::Kernel(format!("Failed to download column: {}", e)))?;

        Ok(bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    /// Download an I64 column from GPU to host memory
    ///
    /// # Arguments
    /// * `buffer` - The CudaBuffer containing the column
    /// * `col_idx` - The column index to download
    ///
    /// # Returns
    /// A Vec<i64> containing the column data
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Column index is out of bounds
    /// - Download fails
    pub fn download_column_i64(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<i64>> {
        let col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

        if buffer.num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = (buffer.num_rows as usize) * std::mem::size_of::<i64>();
        if col.num_bytes() != num_bytes {
            return Err(XlogError::Kernel(format!(
                "Column {} has {} bytes but expected {} for {} rows",
                col_idx,
                col.num_bytes(),
                num_bytes,
                buffer.num_rows
            )));
        }
        let mut bytes = vec![0u8; num_bytes];
        self.device
            .inner()
            .dtoh_sync_copy_into(col, &mut bytes)
            .map_err(|e| XlogError::Kernel(format!("Failed to download column: {}", e)))?;

        Ok(bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    /// Download an F32 column from GPU to host memory
    ///
    /// # Arguments
    /// * `buffer` - The CudaBuffer containing the column
    /// * `col_idx` - The column index to download
    ///
    /// # Returns
    /// A Vec<f32> containing the column data
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Column index is out of bounds
    /// - Download fails
    pub fn download_column_f32(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<Vec<f32>> {
        let col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

        if buffer.num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = (buffer.num_rows as usize) * std::mem::size_of::<f32>();
        if col.num_bytes() != num_bytes {
            return Err(XlogError::Kernel(format!(
                "Column {} has {} bytes but expected {} for {} rows",
                col_idx,
                col.num_bytes(),
                num_bytes,
                buffer.num_rows
            )));
        }
        let mut bytes = vec![0u8; num_bytes];
        self.device
            .inner()
            .dtoh_sync_copy_into(col, &mut bytes)
            .map_err(|e| XlogError::Kernel(format!("Failed to download column: {}", e)))?;

        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    // ============== Internal Helper Methods ==============

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
            _marker: PhantomData,
        })
    }

    /// Reinterpret a `CudaBuffer` column as a `u32` slice for kernel access.
    fn column_as_u32_view<'a>(&self, col: &'a CudaColumn, num_elements: usize) -> Result<RawCudaView<'a, u32>> {
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
            _marker: PhantomData,
        })
    }

    fn column_as_u64_view<'a>(&self, col: &'a CudaColumn, num_elements: usize) -> Result<RawCudaView<'a, u64>> {
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
            _marker: PhantomData,
        })
    }

    /// Reinterpret a `CudaBuffer` column as an `f64` slice for kernel access.
    fn column_as_f64_view<'a>(&self, col: &'a CudaColumn, num_elements: usize) -> Result<RawCudaView<'a, f64>> {
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
        Ok(CudaBuffer::from_columns(columns, 0, schema))
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

    /// Create result schema for groupby aggregation
    fn groupby_result_schema(&self, input: &Schema, key_cols: &[usize], agg: AggOp) -> Schema {
        let mut columns: Vec<(String, ScalarType)> = key_cols
            .iter()
            .filter_map(|&i| input.columns.get(i).cloned())
            .collect();

        // Count and Sum use u64 to match predicate declarations and prevent overflow
        let agg_type = match agg {
            AggOp::Count => ScalarType::U64,
            AggOp::Sum => ScalarType::U64,
            AggOp::Min | AggOp::Max => ScalarType::U32,
            AggOp::LogSumExp => ScalarType::F64,
        };

        let agg_name = match agg {
            AggOp::Count => "count",
            AggOp::Sum => "sum",
            AggOp::Min => "min",
            AggOp::Max => "max",
            AggOp::LogSumExp => "logsumexp",
        };

        columns.push((agg_name.to_string(), agg_type));
        Schema::new(columns)
    }

    // ============== Hash Join V2 Implementation ==============

    /// Multi-column hash join with support for different join types.
    ///
    /// # Arguments
    /// * `left` - The left (probe) buffer
    /// * `right` - The right (build) buffer
    /// * `left_keys` - Column indices for join keys in left buffer
    /// * `right_keys` - Column indices for join keys in right buffer
    /// * `join_type` - Type of join to perform (Inner, Semi, Anti, LeftOuter)
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails or parameters are invalid
    pub fn hash_join_v2(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        join_type: JoinType,
    ) -> Result<CudaBuffer> {
        self.hash_join_v2_with_limit(left, right, left_keys, right_keys, join_type, None)
    }

    /// V2 hash join with configurable maximum output size
    ///
    /// Multi-column join with typed key comparison, supporting different join types.
    /// Uses composite hashing (FNV-1a) for multi-column keys with full key verification.
    ///
    /// # Arguments
    /// * `left` - The left (probe) buffer
    /// * `right` - The right (build) buffer
    /// * `left_keys` - Column indices for join keys in left buffer
    /// * `right_keys` - Column indices for join keys in right buffer
    /// * `join_type` - Type of join to perform (Inner, Semi, Anti, LeftOuter)
    /// * `max_output` - Optional maximum number of output rows (None = unlimited, subject to memory budget)
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails or parameters are invalid
    pub fn hash_join_v2_with_limit(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        join_type: JoinType,
        max_output: Option<usize>,
    ) -> Result<CudaBuffer> {
        match join_type {
            JoinType::Inner => {
                self.hash_join_inner_v2(left, right, left_keys, right_keys, max_output)
            }
            JoinType::Semi => self.hash_join_semi_impl(left, right, left_keys, right_keys),
            JoinType::Anti => self.hash_join_anti_impl(left, right, left_keys, right_keys),
            JoinType::LeftOuter => {
                self.hash_join_left_outer_impl(left, right, left_keys, right_keys, max_output)
            }
        }
    }

    /// Build a cached join index for the right/build side of v2 hash join.
    pub fn build_join_index_v2(&self, right: &CudaBuffer, right_keys: &[usize]) -> Result<JoinIndexV2> {
        if right.is_empty() {
            return Err(XlogError::Kernel(
                "Cannot build join index for empty relation".to_string(),
            ));
        }
        if right_keys.is_empty() {
            return Err(XlogError::Kernel(
                "Join requires at least one key column".to_string(),
            ));
        }
        for &k in right_keys {
            if k >= right.arity() {
                return Err(XlogError::Kernel(format!(
                    "Right key column index {} out of bounds (arity {})",
                    k,
                    right.arity()
                )));
            }
        }

        let num_right = right.num_rows() as u32;
        let right_packed = self.compute_hashes_and_pack_keys(right, right_keys)?;
        let table = self.build_hash_table_v2(&right_packed.hashes, num_right)?;

        Ok(JoinIndexV2 {
            right_num_rows: num_right,
            right_keys: right_keys.to_vec(),
            key_bytes: right_packed.key_bytes,
            packed_keys: right_packed.packed_keys,
            table,
        })
    }

    /// Hash join using a cached build-side join index.
    ///
    /// The `index` must have been built for the same `right` buffer and `right_keys`.
    pub fn hash_join_v2_with_index(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        join_type: JoinType,
        index: &JoinIndexV2,
        max_output: Option<usize>,
    ) -> Result<CudaBuffer> {
        // Handle empty inputs early.
        if left.is_empty() {
            return match join_type {
                JoinType::Inner | JoinType::LeftOuter => {
                    let combined_schema = self.combine_schemas(left.schema(), right.schema());
                    self.create_empty_buffer(combined_schema)
                }
                JoinType::Semi | JoinType::Anti => self.create_empty_buffer(left.schema().clone()),
            };
        }
        if right.is_empty() {
            return match join_type {
                JoinType::Inner => {
                    let combined_schema = self.combine_schemas(left.schema(), right.schema());
                    self.create_empty_buffer(combined_schema)
                }
                JoinType::Semi => self.create_empty_buffer(left.schema().clone()),
                JoinType::Anti => self.clone_buffer(left),
                JoinType::LeftOuter => self.left_outer_with_nulls(left, right),
            };
        }

        // Validate key columns.
        if left_keys.is_empty() || right_keys.is_empty() {
            return Err(XlogError::Kernel(
                "Join requires at least one key column".to_string(),
            ));
        }
        if left_keys.len() != right_keys.len() {
            return Err(XlogError::Kernel(
                "Left and right key columns must have same length".to_string(),
            ));
        }
        for (&left_idx, &right_idx) in left_keys.iter().zip(right_keys.iter()) {
            if left_idx >= left.arity() {
                return Err(XlogError::Kernel(format!(
                    "Left key column index {} out of bounds (arity {})",
                    left_idx,
                    left.arity()
                )));
            }
            if right_idx >= right.arity() {
                return Err(XlogError::Kernel(format!(
                    "Right key column index {} out of bounds (arity {})",
                    right_idx,
                    right.arity()
                )));
            }
            let left_type = left.schema().column_type(left_idx);
            let right_type = right.schema().column_type(right_idx);
            if left_type != right_type {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    left_idx, left_type, right_idx, right_type
                )));
            }
        }

        // Validate index matches the right side.
        if index.right_num_rows != right.num_rows() as u32 {
            return Err(XlogError::Kernel(
                "Join index row count does not match right relation".to_string(),
            ));
        }
        if index.right_keys.as_slice() != right_keys {
            return Err(XlogError::Kernel(
                "Join index key columns do not match requested right_keys".to_string(),
            ));
        }

        match join_type {
            JoinType::Inner => self.hash_join_inner_v2_indexed(left, right, left_keys, index, max_output),
            JoinType::Semi => self.hash_join_semi_indexed(left, left_keys, index),
            JoinType::Anti => self.hash_join_anti_indexed(left, right, left_keys, index),
            JoinType::LeftOuter => {
                self.hash_join_left_outer_indexed(left, right, left_keys, index, max_output)
            }
        }
    }

    /// Pack key columns on GPU and compute hashes (no host roundtrip).
    ///
    /// Uses the fused `pack_and_hash_keys` kernel for optimal performance when both
    /// packed keys and hashes are needed. This eliminates the host roundtrip that
    /// was previously required for column-major to row-major conversion.
    ///
    /// # Arguments
    /// * `buffer` - Source buffer with columns to pack
    /// * `key_cols` - Indices of columns to pack as keys (max 4)
    ///
    /// # Returns
    /// `PackedKeyData` containing GPU-resident packed keys and hashes
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - No key columns specified
    /// - More than 4 key columns specified (kernel limitation)
    /// - Column index is out of bounds
    /// - Kernel launch fails
    fn pack_keys_gpu(&self, buffer: &CudaBuffer, key_cols: &[usize]) -> Result<PackedKeyData> {
        if key_cols.is_empty() {
            return Err(XlogError::Kernel(
                "pack_keys_gpu: no key columns specified".into(),
            ));
        }
        if key_cols.len() > 4 {
            return Err(XlogError::Kernel(
                "pack_keys_gpu: max 4 key columns supported".into(),
            ));
        }

        let num_rows = buffer.num_rows() as u32;
        if num_rows == 0 {
            // Handle empty buffer case
            return Ok(PackedKeyData {
                hashes: self.memory.alloc::<u64>(0)?,
                packed_keys: self.memory.alloc::<u8>(0)?,
                key_bytes: 0,
            });
        }

        // Calculate column sizes and total row size
        let mut col_sizes: Vec<u32> = Vec::with_capacity(key_cols.len());
        let mut row_size: u32 = 0;
        for &col_idx in key_cols {
            let col_type = buffer
                .schema()
                .column_type(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Invalid column index: {}", col_idx)))?;
            let size = col_type.size_bytes() as u32;
            col_sizes.push(size);
            row_size += size;
        }

        // Allocate output buffers on GPU
        let packed_bytes = (num_rows as u64) * (row_size as u64);
        let packed_slice = self.memory.alloc::<u8>(packed_bytes as usize)?;
        let hash_slice = self.memory.alloc::<u64>(num_rows as usize)?;

        // Upload column sizes to GPU
        let mut col_sizes_slice = self.memory.alloc::<u32>(col_sizes.len())?;
        self.htod_sync_copy_into_tracked(&col_sizes, &mut col_sizes_slice)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload col_sizes: {}", e)))?;

        // Get column device pointers as u64 values for the kernel
        // The kernel expects raw pointers as u64
        let mut col_ptrs: [u64; 4] = [0; 4];
        for (i, &col_idx) in key_cols.iter().enumerate() {
            let col = buffer
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Key column {} not found", col_idx)))?;
            // Get the device pointer as a raw u64 value
            col_ptrs[i] = *col.device_ptr() as u64;
        }

        // Get the kernel function
        let func = self
            .device
            .inner()
            .get_func(PACK_MODULE, pack_kernels::PACK_AND_HASH_KEYS)
            .ok_or_else(|| XlogError::Kernel("pack_and_hash_keys kernel not found".to_string()))?;

        // Launch configuration
        let block_size = 256u32;
        let grid_size = (num_rows + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch the fused pack+hash kernel
        // SAFETY: Kernel signature matches pack_and_hash_keys in pack.cu:
        // pack_and_hash_keys(col0, col1, col2, col3, col_sizes, num_cols, num_rows, row_size, packed_output, hashes)
        // Column pointers are passed as CudaSlice references - the kernel sees raw device pointers.
        // We pass column data as raw pointers cast to u8* in the kernel.
        unsafe {
            func.clone()
                .launch(
                    config,
                    (
                        col_ptrs[0],
                        col_ptrs[1],
                        col_ptrs[2],
                        col_ptrs[3],
                        &col_sizes_slice,
                        key_cols.len() as u32,
                        num_rows,
                        row_size,
                        &packed_slice,
                        &hash_slice,
                    ),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("pack_and_hash_keys launch failed: {}", e))
                })?;
        }

        self.device.synchronize()?;

        Ok(PackedKeyData {
            hashes: hash_slice,
            packed_keys: packed_slice,
            key_bytes: row_size,
        })
    }

    /// Compute composite hashes AND return packed key data for key verification.
    ///
    /// Uses GPU-side packing via `pack_keys_gpu` when possible (1-4 key columns).
    /// Falls back to CPU packing for edge cases or when GPU packing fails.
    ///
    /// Uses FNV-1a hash to combine all key columns into a single u64 hash per row.
    /// Also returns the packed key data for byte-by-byte comparison in join kernels.
    fn compute_hashes_and_pack_keys(
        &self,
        buffer: &CudaBuffer,
        key_cols: &[usize],
    ) -> Result<PackedKeyData> {
        // Use GPU-side packing for 1-4 key columns (optimal path)
        if !key_cols.is_empty() && key_cols.len() <= 4 {
            return self.pack_keys_gpu(buffer, key_cols);
        }

        // Fallback path for empty key_cols or >4 key columns
        // (>4 key columns is rare in practice and uses CPU packing)
        self.compute_hashes_and_pack_keys_cpu(buffer, key_cols)
    }

    /// CPU-based key packing fallback for edge cases.
    ///
    /// This is used when GPU packing is not applicable (e.g., >4 key columns).
    fn compute_hashes_and_pack_keys_cpu(
        &self,
        buffer: &CudaBuffer,
        key_cols: &[usize],
    ) -> Result<PackedKeyData> {
        let num_rows = buffer.num_rows() as u32;
        if num_rows == 0 {
            return Ok(PackedKeyData {
                hashes: self.memory.alloc::<u64>(0)?,
                packed_keys: self.memory.alloc::<u8>(0)?,
                key_bytes: 0,
            });
        }

        // For column-major storage, we need to pack data row-major for the kernel
        // CPU approach: copy key columns to host, pack row-major, upload

        let num_key_cols = key_cols.len();
        let mut col_sizes: Vec<u32> = Vec::with_capacity(num_key_cols);
        let mut col_data: Vec<Vec<u8>> = Vec::with_capacity(num_key_cols);

        // Read key columns to host
        for &col_idx in key_cols {
            let col = buffer
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Key column {} not found", col_idx)))?;

            let elem_size = buffer
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            let bytes = (num_rows as usize) * elem_size;
            let mut host_data = vec![0u8; bytes];
            self.device
                .inner()
                .dtoh_sync_copy_into(col, &mut host_data)
                .map_err(|e| XlogError::Kernel(format!("Failed to read key column: {}", e)))?;

            col_sizes.push(elem_size as u32);
            col_data.push(host_data);
        }

        // Compute row stride (total bytes per row)
        let row_stride: u32 = col_sizes.iter().sum();

        // Pack data row-major: for each row, concatenate key column values
        let total_bytes = (num_rows as usize) * (row_stride as usize);
        let mut packed = vec![0u8; total_bytes];

        for row in 0..num_rows as usize {
            let mut offset = row * (row_stride as usize);
            for (col_idx, elem_size) in col_sizes.iter().enumerate() {
                let elem_size = *elem_size as usize;
                let src_start = row * elem_size;
                let src_end = src_start + elem_size;
                packed[offset..offset + elem_size]
                    .copy_from_slice(&col_data[col_idx][src_start..src_end]);
                offset += elem_size;
            }
        }

        // Upload packed data
        let mut d_data = self.memory.alloc::<u8>(packed.len())?;
        self.device
            .inner()
            .htod_sync_copy_into(&packed, &mut d_data)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload packed data: {}", e)))?;

        // Compute column offsets within each row
        let mut col_offsets = vec![0u32; num_key_cols];
        let mut offset = 0u32;
        for i in 0..num_key_cols {
            col_offsets[i] = offset;
            offset += col_sizes[i];
        }

        let d_col_offsets = self
            .device
            .inner()
            .htod_sync_copy(&col_offsets)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload col_offsets: {}", e)))?;

        let d_col_sizes = self
            .device
            .inner()
            .htod_sync_copy(&col_sizes)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload col_sizes: {}", e)))?;

        // Allocate output hashes
        let d_hashes = self.memory.alloc::<u64>(num_rows as usize)?;

        // Get kernel
        let hash_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::COMPUTE_COMPOSITE_HASH)
            .ok_or_else(|| {
                XlogError::Kernel("compute_composite_hash kernel not found".to_string())
            })?;

        // Launch kernel
        let block_size = 256u32;
        let grid_size = (num_rows + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Kernel signature:
        // compute_composite_hash(data, col_offsets, col_sizes, num_key_cols, num_rows, row_stride, hashes)
        unsafe {
            hash_func
                .clone()
                .launch(
                    config,
                    (
                        &d_data,
                        &d_col_offsets,
                        &d_col_sizes,
                        num_key_cols as u32,
                        num_rows,
                        row_stride,
                        &d_hashes,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("compute_composite_hash failed: {}", e)))?;
        }

        self.device.synchronize()?;
        Ok(PackedKeyData {
            hashes: d_hashes,
            packed_keys: d_data,
            key_bytes: row_stride,
        })
    }

    /// Build a cache-friendly hash table from u64 hashes (v2).
    ///
    /// The table uses a bucketed CSR layout (counts + offsets + entries), avoiding linked-list
    /// pointer chasing during probe.
    fn build_hash_table_v2(
        &self,
        hashes: &cudarc::driver::CudaSlice<u64>,
        num_rows: u32,
    ) -> Result<JoinHashTableV2> {
        let device = self.device.inner();

        // Number of buckets: next power-of-two >= max(2*num_rows, 1024)
        let target = (num_rows as u64).saturating_mul(2).max(1024);
        let num_buckets_u64 = target.next_power_of_two();
        let num_buckets = u32::try_from(num_buckets_u64).map_err(|_| {
            XlogError::Kernel(format!(
                "Join hash table too large: num_buckets={}",
                num_buckets_u64
            ))
        })?;
        let bucket_mask = num_buckets
            .checked_sub(1)
            .ok_or_else(|| XlogError::Kernel("Join hash table size underflow".to_string()))?;

        let mut bucket_counts = self.memory.alloc::<u32>(num_buckets as usize)?;
        if num_buckets > 0 {
            device
                .memset_zeros(&mut bucket_counts)
                .map_err(|e| XlogError::Kernel(format!("Failed to zero bucket_counts: {}", e)))?;
        }

        let block_size = 256u32;
        let grid_size = (num_rows + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let count_fn = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_BUCKET_COUNT_V2)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_bucket_count_v2 kernel not found".to_string())
            })?;

        // SAFETY: hash_join_bucket_count_v2(hashes, num_rows, bucket_counts, bucket_mask)
        unsafe {
            count_fn
                .clone()
                .launch(config, (hashes, num_rows, &bucket_counts, bucket_mask))
                .map_err(|e| {
                    XlogError::Kernel(format!("hash_join_bucket_count_v2 failed: {}", e))
                })?;
        }

        // bucket_offsets = exclusive scan(bucket_counts)
        let mut bucket_offsets = self.memory.alloc::<u32>(num_buckets as usize)?;
        if num_buckets > 0 {
            device
                .dtod_copy(&bucket_counts, &mut bucket_offsets)
                .map_err(|e| XlogError::Kernel(format!("Failed to copy bucket_counts: {}", e)))?;
            self.multiblock_scan_u32_inplace(&mut bucket_offsets, num_buckets)?;
        }

        // bucket_cursors = bucket_offsets (then atomically incremented during scatter)
        let mut bucket_cursors = self.memory.alloc::<u32>(num_buckets as usize)?;
        if num_buckets > 0 {
            device
                .dtod_copy(&bucket_offsets, &mut bucket_cursors)
                .map_err(|e| XlogError::Kernel(format!("Failed to copy bucket_offsets: {}", e)))?;
        }

        let bucket_entries = self.memory.alloc::<u32>(num_rows as usize)?;
        let bucket_entry_hashes = self.memory.alloc::<u64>(num_rows as usize)?;

        let scatter_fn = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_SCATTER_V2)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_scatter_v2 kernel not found".to_string())
            })?;

        // SAFETY: hash_join_scatter_v2(hashes, num_rows, bucket_cursors, bucket_mask, bucket_entries, bucket_entry_hashes)
        unsafe {
            scatter_fn
                .clone()
                .launch(
                    config,
                    (
                        hashes,
                        num_rows,
                        &bucket_cursors,
                        bucket_mask,
                        &bucket_entries,
                        &bucket_entry_hashes,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_scatter_v2 failed: {}", e)))?;
        }

        self.device.synchronize()?;
        Ok(JoinHashTableV2 {
            bucket_counts,
            bucket_offsets,
            bucket_entries,
            bucket_entry_hashes,
            bucket_mask,
        })
    }

    /// Inner join implementation using v2 kernels
    fn hash_join_inner_v2(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        max_output: Option<usize>,
    ) -> Result<CudaBuffer> {
        // Handle empty inputs
        if left.is_empty() || right.is_empty() {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        // Validate key columns
        if left_keys.is_empty() || right_keys.is_empty() {
            return Err(XlogError::Kernel(
                "Join requires at least one key column".to_string(),
            ));
        }
        if left_keys.len() != right_keys.len() {
            return Err(XlogError::Kernel(
                "Left and right key columns must have same length".to_string(),
            ));
        }

        // Validate key column types match
        for (&left_idx, &right_idx) in left_keys.iter().zip(right_keys.iter()) {
            let left_type = left.schema().column_type(left_idx);
            let right_type = right.schema().column_type(right_idx);
            if left_type != right_type {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    left_idx, left_type, right_idx, right_type
                )));
            }
        }

        let num_left = left.num_rows() as u32;
        let num_right = right.num_rows() as u32;

        // Compute composite hashes and pack keys for both sides
        let left_packed = self.compute_hashes_and_pack_keys(left, left_keys)?;
        let right_packed = self.compute_hashes_and_pack_keys(right, right_keys)?;

        // Build hash table from right side (cache-friendly bucket layout).
        let table = self.build_hash_table_v2(&right_packed.hashes, num_right)?;

        // Count join output (no truncation) to size buffers precisely.
        //
        // The probe kernel always increments output_count, even when max_output==0,
        // so we can run a first pass with max_output=0 to get the full match count.
        let probe_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2)
            .ok_or_else(|| XlogError::Kernel("hash_join_probe_v2 kernel not found".to_string()))?;

        let block_size = 256u32;
        let probe_grid = (num_left + block_size - 1) / block_size;
        let probe_config = LaunchConfig {
            grid_dim: (probe_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let d_count_only = self
            .device
            .inner()
            .htod_sync_copy(&[0u32])
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc output count: {}", e)))?;
        let d_dummy_left = self.memory.alloc::<u32>(1)?;
        let d_dummy_right = self.memory.alloc::<u32>(1)?;
        let max_output_count_only = 0u32;

        // SAFETY: hash_join_probe_v2(probe_hashes, num_probe,
        //                            bucket_offsets, bucket_counts, bucket_entries, bucket_entry_hashes, bucket_mask,
        //                            probe_keys, build_keys, key_bytes,
        //                            output_left, output_right, output_count, max_output)
        // Note: Using raw pointer launch because tuple exceeds 12-element limit
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&*left_packed.hashes).as_kernel_param(),
                (&num_left).as_kernel_param(),
                (&*table.bucket_offsets).as_kernel_param(),
                (&*table.bucket_counts).as_kernel_param(),
                (&*table.bucket_entries).as_kernel_param(),
                (&*table.bucket_entry_hashes).as_kernel_param(),
                (&table.bucket_mask).as_kernel_param(),
                (&*left_packed.packed_keys).as_kernel_param(),
                (&*right_packed.packed_keys).as_kernel_param(),
                (&left_packed.key_bytes).as_kernel_param(),
                (&*d_dummy_left).as_kernel_param(),
                (&*d_dummy_right).as_kernel_param(),
                (&d_count_only).as_kernel_param(),
                (&max_output_count_only).as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(probe_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("hash_join_probe_v2 (count) failed: {}", e))
                })?;
        }

        self.device.synchronize()?;

        let mut count_host = vec![0u32];
        self.device
            .inner()
            .dtoh_sync_copy_into(&d_count_only, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read output count: {}", e)))?;

        let full_count = count_host[0] as u64;
        let requested = max_output
            .map(|limit| (limit as u64).min(full_count))
            .unwrap_or(full_count);

        if requested == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        if requested > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Join produced {} rows which exceeds the u32 index limit",
                requested
            )));
        }

        // Allocate output buffers for row index pairs and rerun probe to materialize results.
        let max_output = requested as u32;
        let d_output_left = self.memory.alloc::<u32>(max_output as usize)?;
        let d_output_right = self.memory.alloc::<u32>(max_output as usize)?;
        let d_output_count = self
            .device
            .inner()
            .htod_sync_copy(&[0u32])
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc output count: {}", e)))?;

        // SAFETY: hash_join_probe_v2(probe_hashes, num_probe,
        //                            bucket_offsets, bucket_counts, bucket_entries, bucket_entry_hashes, bucket_mask,
        //                            probe_keys, build_keys, key_bytes,
        //                            output_left, output_right, output_count, max_output)
        // Note: Using raw pointer launch because tuple exceeds 12-element limit
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&*left_packed.hashes).as_kernel_param(),
                (&num_left).as_kernel_param(),
                (&*table.bucket_offsets).as_kernel_param(),
                (&*table.bucket_counts).as_kernel_param(),
                (&*table.bucket_entries).as_kernel_param(),
                (&*table.bucket_entry_hashes).as_kernel_param(),
                (&table.bucket_mask).as_kernel_param(),
                (&*left_packed.packed_keys).as_kernel_param(),
                (&*right_packed.packed_keys).as_kernel_param(),
                (&left_packed.key_bytes).as_kernel_param(),
                (&*d_output_left).as_kernel_param(),
                (&*d_output_right).as_kernel_param(),
                (&d_output_count).as_kernel_param(),
                (&max_output).as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(probe_config, &mut params)
                .map_err(|e| XlogError::Kernel(format!("hash_join_probe_v2 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        // Get output count
        let mut count_host = vec![0u32];
        self.device
            .inner()
            .dtoh_sync_copy_into(&d_output_count, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read output count: {}", e)))?;
        // Clamp result count to max_output to prevent buffer overflow
        // (kernel atomically increments before bounds check, so count can exceed max_output)
        let result_count = (count_host[0] as u64).min(max_output as u64);

        if result_count == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        let output_rows = result_count as u32;

        // Gather join results fully on-GPU (avoid host index download + host gather).
        let gathered_left = self.gather_buffer_by_indices(left, &d_output_left, output_rows)?;
        let gathered_right = self.gather_buffer_by_indices(right, &d_output_right, output_rows)?;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        result_columns.extend(gathered_left.columns.into_iter());
        result_columns.extend(gathered_right.columns.into_iter());

        Ok(CudaBuffer::from_columns(
            result_columns,
            result_count,
            combined_schema,
        ))
    }

    fn hash_join_inner_v2_indexed(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        index: &JoinIndexV2,
        max_output: Option<usize>,
    ) -> Result<CudaBuffer> {
        // Handle empty inputs.
        if left.is_empty() || right.is_empty() {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        let num_left = left.num_rows() as u32;

        // Compute composite hashes and pack probe keys.
        let left_packed = self.compute_hashes_and_pack_keys(left, left_keys)?;
        if left_packed.key_bytes != index.key_bytes {
            return Err(XlogError::Kernel(
                "Join key byte width mismatch between probe and cached index".to_string(),
            ));
        }

        let table = &index.table;

        // Count join output (no truncation) to size buffers precisely.
        let probe_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2)
            .ok_or_else(|| XlogError::Kernel("hash_join_probe_v2 kernel not found".to_string()))?;

        let block_size = 256u32;
        let probe_grid = (num_left + block_size - 1) / block_size;
        let probe_config = LaunchConfig {
            grid_dim: (probe_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let d_count_only = self
            .device
            .inner()
            .htod_sync_copy(&[0u32])
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc output count: {}", e)))?;
        let d_dummy_left = self.memory.alloc::<u32>(1)?;
        let d_dummy_right = self.memory.alloc::<u32>(1)?;
        let max_output_count_only = 0u32;

        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&*left_packed.hashes).as_kernel_param(),
                (&num_left).as_kernel_param(),
                (&*table.bucket_offsets).as_kernel_param(),
                (&*table.bucket_counts).as_kernel_param(),
                (&*table.bucket_entries).as_kernel_param(),
                (&*table.bucket_entry_hashes).as_kernel_param(),
                (&table.bucket_mask).as_kernel_param(),
                (&*left_packed.packed_keys).as_kernel_param(),
                (&*index.packed_keys).as_kernel_param(),
                (&index.key_bytes).as_kernel_param(),
                (&*d_dummy_left).as_kernel_param(),
                (&*d_dummy_right).as_kernel_param(),
                (&d_count_only).as_kernel_param(),
                (&max_output_count_only).as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(probe_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("hash_join_probe_v2 (count) failed: {}", e))
                })?;
        }

        self.device.synchronize()?;

        let mut count_host = vec![0u32];
        self.device
            .inner()
            .dtoh_sync_copy_into(&d_count_only, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read output count: {}", e)))?;

        let full_count = count_host[0] as u64;
        let requested = max_output
            .map(|limit| (limit as u64).min(full_count))
            .unwrap_or(full_count);

        if requested == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        if requested > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Join produced {} rows which exceeds the u32 index limit",
                requested
            )));
        }

        // Allocate output buffers for row index pairs and rerun probe to materialize results.
        let max_output = requested as u32;
        let d_output_left = self.memory.alloc::<u32>(max_output as usize)?;
        let d_output_right = self.memory.alloc::<u32>(max_output as usize)?;
        let d_output_count = self
            .device
            .inner()
            .htod_sync_copy(&[0u32])
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc output count: {}", e)))?;

        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&*left_packed.hashes).as_kernel_param(),
                (&num_left).as_kernel_param(),
                (&*table.bucket_offsets).as_kernel_param(),
                (&*table.bucket_counts).as_kernel_param(),
                (&*table.bucket_entries).as_kernel_param(),
                (&*table.bucket_entry_hashes).as_kernel_param(),
                (&table.bucket_mask).as_kernel_param(),
                (&*left_packed.packed_keys).as_kernel_param(),
                (&*index.packed_keys).as_kernel_param(),
                (&index.key_bytes).as_kernel_param(),
                (&*d_output_left).as_kernel_param(),
                (&*d_output_right).as_kernel_param(),
                (&d_output_count).as_kernel_param(),
                (&max_output).as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(probe_config, &mut params)
                .map_err(|e| XlogError::Kernel(format!("hash_join_probe_v2 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        // Get output count.
        let mut count_host = vec![0u32];
        self.device
            .inner()
            .dtoh_sync_copy_into(&d_output_count, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read output count: {}", e)))?;

        let result_count = (count_host[0] as u64).min(max_output as u64);

        if result_count == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        let output_rows = result_count as u32;

        let gathered_left = self.gather_buffer_by_indices(left, &d_output_left, output_rows)?;
        let gathered_right = self.gather_buffer_by_indices(right, &d_output_right, output_rows)?;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        result_columns.extend(gathered_left.columns.into_iter());
        result_columns.extend(gathered_right.columns.into_iter());

        Ok(CudaBuffer::from_columns(
            result_columns,
            result_count,
            combined_schema,
        ))
    }

    /// Semi-join implementation: return left rows that have matches in right
    fn hash_join_semi_impl(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
    ) -> Result<CudaBuffer> {
        // Handle empty inputs
        if left.is_empty() {
            return self.create_empty_buffer(left.schema().clone());
        }
        if right.is_empty() {
            // No matches possible - return empty with left schema
            return self.create_empty_buffer(left.schema().clone());
        }

        // Validate key columns
        if left_keys.is_empty() || right_keys.is_empty() {
            return Err(XlogError::Kernel(
                "Join requires at least one key column".to_string(),
            ));
        }
        if left_keys.len() != right_keys.len() {
            return Err(XlogError::Kernel(
                "Left and right key columns must have same length".to_string(),
            ));
        }

        // Validate key column types match
        for (&left_idx, &right_idx) in left_keys.iter().zip(right_keys.iter()) {
            let left_type = left.schema().column_type(left_idx);
            let right_type = right.schema().column_type(right_idx);
            if left_type != right_type {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    left_idx, left_type, right_idx, right_type
                )));
            }
        }

        let num_left = left.num_rows() as u32;
        let num_right = right.num_rows() as u32;

        // Compute composite hashes and pack keys for both sides
        let left_packed = self.compute_hashes_and_pack_keys(left, left_keys)?;
        let right_packed = self.compute_hashes_and_pack_keys(right, right_keys)?;

        // Build hash table from right side (cache-friendly bucket layout).
        let table = self.build_hash_table_v2(&right_packed.hashes, num_right)?;

        // Allocate output mask
        let d_has_match = self.memory.alloc::<u8>(num_left as usize)?;

        // Launch semi-join kernel
        let semi_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_SEMI)
            .ok_or_else(|| XlogError::Kernel("hash_join_semi kernel not found".to_string()))?;

        let block_size = 256u32;
        let grid_size = (num_left + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: hash_join_semi(probe_hashes, num_probe,
        //                        bucket_offsets, bucket_counts, bucket_entries, bucket_entry_hashes, bucket_mask,
        //                        probe_keys, build_keys, key_bytes, has_match)
        unsafe {
            semi_func
                .clone()
                .launch(
                    config,
                    (
                        &left_packed.hashes,
                        num_left,
                        &table.bucket_offsets,
                        &table.bucket_counts,
                        &table.bucket_entries,
                        &table.bucket_entry_hashes,
                        table.bucket_mask,
                        &left_packed.packed_keys,
                        &right_packed.packed_keys,
                        left_packed.key_bytes,
                        &d_has_match,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_semi failed: {}", e)))?;
        }

        self.filter_by_device_mask(left, &d_has_match)
    }

    fn hash_join_semi_indexed(
        &self,
        left: &CudaBuffer,
        left_keys: &[usize],
        index: &JoinIndexV2,
    ) -> Result<CudaBuffer> {
        // Handle empty inputs.
        if left.is_empty() {
            return self.create_empty_buffer(left.schema().clone());
        }
        if index.right_num_rows == 0 {
            return self.create_empty_buffer(left.schema().clone());
        }

        let num_left = left.num_rows() as u32;

        let left_packed = self.compute_hashes_and_pack_keys(left, left_keys)?;
        if left_packed.key_bytes != index.key_bytes {
            return Err(XlogError::Kernel(
                "Join key byte width mismatch between probe and cached index".to_string(),
            ));
        }

        let table = &index.table;

        // Allocate output mask.
        let d_has_match = self.memory.alloc::<u8>(num_left as usize)?;

        let semi_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_SEMI)
            .ok_or_else(|| XlogError::Kernel("hash_join_semi kernel not found".to_string()))?;

        let block_size = 256u32;
        let grid_size = (num_left + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            semi_func
                .clone()
                .launch(
                    config,
                    (
                        &left_packed.hashes,
                        num_left,
                        &table.bucket_offsets,
                        &table.bucket_counts,
                        &table.bucket_entries,
                        &table.bucket_entry_hashes,
                        table.bucket_mask,
                        &left_packed.packed_keys,
                        &index.packed_keys,
                        index.key_bytes,
                        &d_has_match,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_semi failed: {}", e)))?;
        }

        self.filter_by_device_mask(left, &d_has_match)
    }

    /// Anti-join implementation: return left rows that have NO matches in right
    fn hash_join_anti_impl(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
    ) -> Result<CudaBuffer> {
        // Handle empty inputs
        if left.is_empty() {
            return self.create_empty_buffer(left.schema().clone());
        }
        if right.is_empty() {
            // No matches possible - return all left rows
            return self.clone_buffer(left);
        }

        // Validate key columns
        if left_keys.is_empty() || right_keys.is_empty() {
            return Err(XlogError::Kernel(
                "Join requires at least one key column".to_string(),
            ));
        }
        if left_keys.len() != right_keys.len() {
            return Err(XlogError::Kernel(
                "Left and right key columns must have same length".to_string(),
            ));
        }

        // Validate key column types match
        for (&left_idx, &right_idx) in left_keys.iter().zip(right_keys.iter()) {
            let left_type = left.schema().column_type(left_idx);
            let right_type = right.schema().column_type(right_idx);
            if left_type != right_type {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    left_idx, left_type, right_idx, right_type
                )));
            }
        }

        let num_left = left.num_rows() as u32;
        let num_right = right.num_rows() as u32;

        // Compute composite hashes and pack keys for both sides
        let left_packed = self.compute_hashes_and_pack_keys(left, left_keys)?;
        let right_packed = self.compute_hashes_and_pack_keys(right, right_keys)?;

        // Build hash table from right side (cache-friendly bucket layout).
        let table = self.build_hash_table_v2(&right_packed.hashes, num_right)?;

        // Allocate output mask
        let d_no_match = self.memory.alloc::<u8>(num_left as usize)?;

        // Launch anti-join kernel
        let anti_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_ANTI)
            .ok_or_else(|| XlogError::Kernel("hash_join_anti kernel not found".to_string()))?;

        let block_size = 256u32;
        let grid_size = (num_left + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: hash_join_anti(probe_hashes, num_probe,
        //                        bucket_offsets, bucket_counts, bucket_entries, bucket_entry_hashes, bucket_mask,
        //                        probe_keys, build_keys, key_bytes, no_match)
        unsafe {
            anti_func
                .clone()
                .launch(
                    config,
                    (
                        &left_packed.hashes,
                        num_left,
                        &table.bucket_offsets,
                        &table.bucket_counts,
                        &table.bucket_entries,
                        &table.bucket_entry_hashes,
                        table.bucket_mask,
                        &left_packed.packed_keys,
                        &right_packed.packed_keys,
                        left_packed.key_bytes,
                        &d_no_match,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_anti failed: {}", e)))?;
        }

        self.filter_by_device_mask(left, &d_no_match)
    }

    fn hash_join_anti_indexed(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        index: &JoinIndexV2,
    ) -> Result<CudaBuffer> {
        if left.is_empty() {
            return self.create_empty_buffer(left.schema().clone());
        }
        if right.is_empty() {
            return self.clone_buffer(left);
        }

        let num_left = left.num_rows() as u32;

        let left_packed = self.compute_hashes_and_pack_keys(left, left_keys)?;
        if left_packed.key_bytes != index.key_bytes {
            return Err(XlogError::Kernel(
                "Join key byte width mismatch between probe and cached index".to_string(),
            ));
        }

        let table = &index.table;

        let d_no_match = self.memory.alloc::<u8>(num_left as usize)?;

        let anti_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_ANTI)
            .ok_or_else(|| XlogError::Kernel("hash_join_anti kernel not found".to_string()))?;

        let block_size = 256u32;
        let grid_size = (num_left + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            anti_func
                .clone()
                .launch(
                    config,
                    (
                        &left_packed.hashes,
                        num_left,
                        &table.bucket_offsets,
                        &table.bucket_counts,
                        &table.bucket_entries,
                        &table.bucket_entry_hashes,
                        table.bucket_mask,
                        &left_packed.packed_keys,
                        &index.packed_keys,
                        index.key_bytes,
                        &d_no_match,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_anti failed: {}", e)))?;
        }

        self.filter_by_device_mask(left, &d_no_match)
    }

    fn hash_join_left_outer_indexed(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        index: &JoinIndexV2,
        max_output: Option<usize>,
    ) -> Result<CudaBuffer> {
        // Handle empty left - return empty with combined schema.
        if left.is_empty() {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }
        // Handle empty right - return left rows with null right columns.
        if right.is_empty() {
            return self.left_outer_with_nulls(left, right);
        }

        let num_left = left.num_rows() as u32;

        let left_packed = self.compute_hashes_and_pack_keys(left, left_keys)?;
        if left_packed.key_bytes != index.key_bytes {
            return Err(XlogError::Kernel(
                "Join key byte width mismatch between probe and cached index".to_string(),
            ));
        }

        let table = &index.table;

        // Allocate mask for semi-join to check which left rows have matches.
        let d_has_match = self.memory.alloc::<u8>(num_left as usize)?;

        let semi_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_SEMI)
            .ok_or_else(|| XlogError::Kernel("hash_join_semi kernel not found".to_string()))?;

        let block_size = 256u32;
        let grid_size = (num_left + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            semi_func
                .clone()
                .launch(
                    config,
                    (
                        &left_packed.hashes,
                        num_left,
                        &table.bucket_offsets,
                        &table.bucket_counts,
                        &table.bucket_entries,
                        &table.bucket_entry_hashes,
                        table.bucket_mask,
                        &left_packed.packed_keys,
                        &index.packed_keys,
                        index.key_bytes,
                        &d_has_match,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_semi failed: {}", e)))?;
        }

        let probe_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2)
            .ok_or_else(|| XlogError::Kernel("hash_join_probe_v2 kernel not found".to_string()))?;

        // Count inner-join matches to size buffers precisely.
        let d_count_only = self
            .device
            .inner()
            .htod_sync_copy(&[0u32])
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc output count: {}", e)))?;
        let d_dummy_left = self.memory.alloc::<u32>(1)?;
        let d_dummy_right = self.memory.alloc::<u32>(1)?;
        let max_output_count_only = 0u32;

        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                (&num_left).as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                (&table.bucket_mask).as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&index.packed_keys).as_kernel_param(),
                (&index.key_bytes).as_kernel_param(),
                (&d_dummy_left).as_kernel_param(),
                (&d_dummy_right).as_kernel_param(),
                (&d_count_only).as_kernel_param(),
                (&max_output_count_only).as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("hash_join_probe_v2 (count) failed: {}", e))
                })?;
        }

        self.device.synchronize()?;

        let mut count_host = vec![0u32];
        self.device
            .inner()
            .dtoh_sync_copy_into(&d_count_only, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read output count: {}", e)))?;

        let full_inner = count_host[0] as u64;
        let requested_inner = max_output
            .map(|limit| (limit as u64).min(full_inner))
            .unwrap_or(full_inner);

        if requested_inner > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Join produced {} rows which exceeds the u32 index limit",
                requested_inner
            )));
        }

        let max_output = requested_inner as u32;
        let alloc_len = (requested_inner.max(1)) as usize;
        let d_output_left = self.memory.alloc::<u32>(alloc_len)?;
        let d_output_right = self.memory.alloc::<u32>(alloc_len)?;
        let d_output_count = self
            .device
            .inner()
            .htod_sync_copy(&[0u32])
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc output count: {}", e)))?;

        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                (&num_left).as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                (&table.bucket_mask).as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&index.packed_keys).as_kernel_param(),
                (&index.key_bytes).as_kernel_param(),
                (&d_output_left).as_kernel_param(),
                (&d_output_right).as_kernel_param(),
                (&d_output_count).as_kernel_param(),
                (&max_output).as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(config, &mut params)
                .map_err(|e| XlogError::Kernel(format!("hash_join_probe_v2 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        let device = self.device.inner();

        let mut count_host = vec![0u32];
        device
            .dtoh_sync_copy_into(&d_output_count, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read output count: {}", e)))?;
        let inner_count = count_host[0].min(max_output) as u32;

        let mask_not_fn = device
            .get_func(FILTER_MODULE, filter_kernels::MASK_NOT)
            .ok_or_else(|| XlogError::Kernel("mask_not kernel not found".to_string()))?;

        let mut d_no_match = self.memory.alloc::<u8>(num_left as usize)?;

        unsafe {
            mask_not_fn
                .clone()
                .launch(config, (&d_has_match, &mut d_no_match, num_left))
        }
        .map_err(|e| XlogError::Kernel(format!("mask_not failed: {}", e)))?;

        let unmatched_left = self.filter_by_device_mask(left, &d_no_match)?;

        let unmatched_rows = unmatched_left.num_rows();
        let total_rows = (inner_count as u64) + unmatched_rows;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());

        if total_rows == 0 {
            return self.create_empty_buffer(combined_schema);
        }

        let inner_left = self.gather_buffer_by_indices(left, &d_output_left, inner_count)?;
        let inner_right = self.gather_buffer_by_indices(right, &d_output_right, inner_count)?;

        if unmatched_rows == 0 {
            let mut result_columns = Vec::with_capacity(combined_schema.arity());
            result_columns.extend(inner_left.columns.into_iter());
            result_columns.extend(inner_right.columns.into_iter());
            return Ok(CudaBuffer::from_columns(
                result_columns,
                inner_count as u64,
                combined_schema,
            ));
        }

        if inner_count == 0 {
            let mut result_columns = Vec::with_capacity(combined_schema.arity());
            result_columns.extend(unmatched_left.columns.into_iter());

            for col_idx in 0..right.arity() {
                let elem_size = right
                    .schema()
                    .column_type(col_idx)
                    .map(|t| t.size_bytes())
                    .unwrap_or(4);

                let bytes = (unmatched_rows as usize)
                    .checked_mul(elem_size)
                    .ok_or_else(|| {
                        XlogError::Kernel(
                            "Left outer join: right column byte size overflow".to_string(),
                        )
                    })?;

                let mut dst_col = self.memory.alloc::<u8>(bytes)?;
                if bytes > 0 {
                    device.memset_zeros(&mut dst_col).map_err(|e| {
                        XlogError::Kernel(format!("Failed to zero null right column: {}", e))
                    })?;
                }
                result_columns.push(dst_col.into());
            }

            self.device.synchronize()?;
            return Ok(CudaBuffer::from_columns(
                result_columns,
                unmatched_rows,
                combined_schema,
            ));
        }

        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        let inner_rows = inner_count as u64;

        for (col_idx, (inner_col, unmatched_col)) in inner_left
            .columns
            .into_iter()
            .zip(unmatched_left.columns.into_iter())
            .enumerate()
        {
            let elem_size = left
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            let inner_bytes = (inner_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("Left outer join: inner_bytes overflow".to_string())
                })?;
            let unmatched_bytes = (unmatched_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("Left outer join: unmatched_bytes overflow".to_string())
                })?;
            let total_bytes = inner_bytes.checked_add(unmatched_bytes).ok_or_else(|| {
                XlogError::Kernel("Left outer join: total_bytes overflow".to_string())
            })?;

            let mut out_col = self.memory.alloc::<u8>(total_bytes)?;

            if inner_bytes > 0 {
                let mut out_view = out_col.slice_mut(0..inner_bytes);
                device.dtod_copy(&inner_col, &mut out_view).map_err(|e| {
                    XlogError::Kernel(format!("Failed to copy inner left column: {}", e))
                })?;
            }
            if unmatched_bytes > 0 {
                let mut out_view = out_col.slice_mut(inner_bytes..total_bytes);
                device
                    .dtod_copy(&unmatched_col, &mut out_view)
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to copy unmatched left column: {}", e))
                    })?;
            }

            result_columns.push(out_col.into());
        }

        for (col_idx, inner_col) in inner_right.columns.into_iter().enumerate() {
            let elem_size = right
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            let inner_bytes = (inner_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("Left outer join: inner_bytes overflow".to_string())
                })?;
            let unmatched_bytes = (unmatched_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("Left outer join: unmatched_bytes overflow".to_string())
                })?;
            let total_bytes = inner_bytes.checked_add(unmatched_bytes).ok_or_else(|| {
                XlogError::Kernel("Left outer join: total_bytes overflow".to_string())
            })?;

            let mut out_col = self.memory.alloc::<u8>(total_bytes)?;

            if total_bytes > 0 {
                device.memset_zeros(&mut out_col).map_err(|e| {
                    XlogError::Kernel(format!("Failed to zero right outer column: {}", e))
                })?;
            }

            if inner_bytes > 0 {
                let mut out_view = out_col.slice_mut(0..inner_bytes);
                device.dtod_copy(&inner_col, &mut out_view).map_err(|e| {
                    XlogError::Kernel(format!("Failed to copy inner right column: {}", e))
                })?;
            }

            result_columns.push(out_col.into());
        }

        self.device.synchronize()?;

        Ok(CudaBuffer::from_columns(
            result_columns,
            total_rows,
            combined_schema,
        ))
    }

    /// Left outer join implementation: return all left rows with matched right columns or nulls
    fn hash_join_left_outer_impl(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        max_output: Option<usize>,
    ) -> Result<CudaBuffer> {
        // Handle empty left - return empty with combined schema
        if left.is_empty() {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        // Handle empty right - return left rows with null right columns
        if right.is_empty() {
            return self.left_outer_with_nulls(left, right);
        }

        // Validate key columns
        if left_keys.is_empty() || right_keys.is_empty() {
            return Err(XlogError::Kernel(
                "Join requires at least one key column".to_string(),
            ));
        }
        if left_keys.len() != right_keys.len() {
            return Err(XlogError::Kernel(
                "Left and right key columns must have same length".to_string(),
            ));
        }

        // Validate key column types match
        for (&left_idx, &right_idx) in left_keys.iter().zip(right_keys.iter()) {
            let left_type = left.schema().column_type(left_idx);
            let right_type = right.schema().column_type(right_idx);
            if left_type != right_type {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    left_idx, left_type, right_idx, right_type
                )));
            }
        }

        let num_left = left.num_rows() as u32;
        let num_right = right.num_rows() as u32;

        // Compute composite hashes and pack keys for both sides
        let left_packed = self.compute_hashes_and_pack_keys(left, left_keys)?;
        let right_packed = self.compute_hashes_and_pack_keys(right, right_keys)?;

        // Build hash table from right side (cache-friendly bucket layout).
        let table = self.build_hash_table_v2(&right_packed.hashes, num_right)?;

        // Allocate mask for semi-join to check which left rows have matches
        let d_has_match = self.memory.alloc::<u8>(num_left as usize)?;

        // Launch semi-join kernel to get match information
        let semi_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_SEMI)
            .ok_or_else(|| XlogError::Kernel("hash_join_semi kernel not found".to_string()))?;

        let block_size = 256u32;
        let grid_size = (num_left + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: hash_join_semi(probe_hashes, num_probe,
        //                        bucket_offsets, bucket_counts, bucket_entries, bucket_entry_hashes, bucket_mask,
        //                        probe_keys, build_keys, key_bytes, has_match)
        unsafe {
            semi_func
                .clone()
                .launch(
                    config,
                    (
                        &left_packed.hashes,
                        num_left,
                        &table.bucket_offsets,
                        &table.bucket_counts,
                        &table.bucket_entries,
                        &table.bucket_entry_hashes,
                        table.bucket_mask,
                        &left_packed.packed_keys,
                        &right_packed.packed_keys,
                        left_packed.key_bytes,
                        &d_has_match,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_semi failed: {}", e)))?;
        }

        let probe_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2)
            .ok_or_else(|| XlogError::Kernel("hash_join_probe_v2 kernel not found".to_string()))?;

        // Count inner-join matches to size buffers precisely.
        let d_count_only = self
            .device
            .inner()
            .htod_sync_copy(&[0u32])
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc output count: {}", e)))?;
        let d_dummy_left = self.memory.alloc::<u32>(1)?;
        let d_dummy_right = self.memory.alloc::<u32>(1)?;
        let max_output_count_only = 0u32;

        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                (&num_left).as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                (&table.bucket_mask).as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&right_packed.packed_keys).as_kernel_param(),
                (&left_packed.key_bytes).as_kernel_param(),
                (&d_dummy_left).as_kernel_param(),
                (&d_dummy_right).as_kernel_param(),
                (&d_count_only).as_kernel_param(),
                (&max_output_count_only).as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("hash_join_probe_v2 (count) failed: {}", e))
                })?;
        }

        self.device.synchronize()?;

        let mut count_host = vec![0u32];
        self.device
            .inner()
            .dtoh_sync_copy_into(&d_count_only, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read output count: {}", e)))?;

        let full_inner = count_host[0] as u64;
        let requested_inner = max_output
            .map(|limit| (limit as u64).min(full_inner))
            .unwrap_or(full_inner);

        if requested_inner > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Join produced {} rows which exceeds the u32 index limit",
                requested_inner
            )));
        }

        let max_output = requested_inner as u32;
        let alloc_len = (requested_inner.max(1)) as usize;
        let d_output_left = self.memory.alloc::<u32>(alloc_len)?;
        let d_output_right = self.memory.alloc::<u32>(alloc_len)?;
        let d_output_count = self
            .device
            .inner()
            .htod_sync_copy(&[0u32])
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc output count: {}", e)))?;

        // SAFETY: hash_join_probe_v2(probe_hashes, num_probe,
        //                            bucket_offsets, bucket_counts, bucket_entries, bucket_entry_hashes, bucket_mask,
        //                            probe_keys, build_keys, key_bytes,
        //                            output_left, output_right, output_count, max_output)
        // Note: Using raw pointer launch because tuple exceeds 12-element limit
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                (&num_left).as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                (&table.bucket_mask).as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&right_packed.packed_keys).as_kernel_param(),
                (&left_packed.key_bytes).as_kernel_param(),
                (&d_output_left).as_kernel_param(),
                (&d_output_right).as_kernel_param(),
                (&d_output_count).as_kernel_param(),
                (&max_output).as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(config, &mut params)
                .map_err(|e| XlogError::Kernel(format!("hash_join_probe_v2 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        let device = self.device.inner();

        // Read inner join result count.
        let mut count_host = vec![0u32];
        device
            .dtoh_sync_copy_into(&d_output_count, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read output count: {}", e)))?;
        // Clamp inner count to max_output to prevent buffer overflow
        // (kernel atomically increments before bounds check, so count can exceed max_output).
        let inner_count = count_host[0].min(max_output) as u32;

        // Build unmatched-left buffer by inverting has_match mask and compacting on-GPU.
        let mask_not_fn = device
            .get_func(FILTER_MODULE, filter_kernels::MASK_NOT)
            .ok_or_else(|| XlogError::Kernel("mask_not kernel not found".to_string()))?;

        let mut d_no_match = self.memory.alloc::<u8>(num_left as usize)?;

        // SAFETY: mask_not(const uint8_t* a, uint8_t* out, uint32_t n)
        unsafe {
            mask_not_fn
                .clone()
                .launch(config, (&d_has_match, &mut d_no_match, num_left))
        }
        .map_err(|e| XlogError::Kernel(format!("mask_not failed: {}", e)))?;

        let unmatched_left = self.filter_by_device_mask(left, &d_no_match)?;

        let unmatched_rows = unmatched_left.num_rows();
        let total_rows = (inner_count as u64) + unmatched_rows;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());

        if total_rows == 0 {
            return self.create_empty_buffer(combined_schema);
        }

        // Gather matched rows (if any) using on-GPU indices.
        let inner_left = self.gather_buffer_by_indices(left, &d_output_left, inner_count)?;
        let inner_right = self.gather_buffer_by_indices(right, &d_output_right, inner_count)?;

        if unmatched_rows == 0 {
            let mut result_columns = Vec::with_capacity(combined_schema.arity());
            result_columns.extend(inner_left.columns.into_iter());
            result_columns.extend(inner_right.columns.into_iter());
            return Ok(CudaBuffer::from_columns(
                result_columns,
                inner_count as u64,
                combined_schema,
            ));
        }

        if inner_count == 0 {
            let mut result_columns = Vec::with_capacity(combined_schema.arity());
            result_columns.extend(unmatched_left.columns.into_iter());

            for col_idx in 0..right.arity() {
                let elem_size = right
                    .schema()
                    .column_type(col_idx)
                    .map(|t| t.size_bytes())
                    .unwrap_or(4);

                let bytes = (unmatched_rows as usize)
                    .checked_mul(elem_size)
                    .ok_or_else(|| {
                        XlogError::Kernel(
                            "Left outer join: right column byte size overflow".to_string(),
                        )
                    })?;

                let mut dst_col = self.memory.alloc::<u8>(bytes)?;
                if bytes > 0 {
                    device.memset_zeros(&mut dst_col).map_err(|e| {
                        XlogError::Kernel(format!("Failed to zero null right column: {}", e))
                    })?;
                }
                result_columns.push(dst_col.into());
            }

            self.device.synchronize()?;
            return Ok(CudaBuffer::from_columns(
                result_columns,
                unmatched_rows,
                combined_schema,
            ));
        }

        // Concatenate: matched rows followed by unmatched rows (null-extended on right).
        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        let inner_rows = inner_count as u64;

        // Left columns: inner-left then unmatched-left.
        for (col_idx, (inner_col, unmatched_col)) in inner_left
            .columns
            .into_iter()
            .zip(unmatched_left.columns.into_iter())
            .enumerate()
        {
            let elem_size = left
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            let inner_bytes = (inner_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("Left outer join: inner_bytes overflow".to_string())
                })?;
            let unmatched_bytes = (unmatched_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("Left outer join: unmatched_bytes overflow".to_string())
                })?;
            let total_bytes = inner_bytes.checked_add(unmatched_bytes).ok_or_else(|| {
                XlogError::Kernel("Left outer join: total_bytes overflow".to_string())
            })?;

            let mut out_col = self.memory.alloc::<u8>(total_bytes)?;

            if inner_bytes > 0 {
                let mut out_view = out_col.slice_mut(0..inner_bytes);
                device.dtod_copy(&inner_col, &mut out_view).map_err(|e| {
                    XlogError::Kernel(format!("Failed to copy inner left column: {}", e))
                })?;
            }
            if unmatched_bytes > 0 {
                let mut out_view = out_col.slice_mut(inner_bytes..total_bytes);
                device
                    .dtod_copy(&unmatched_col, &mut out_view)
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to copy unmatched left column: {}", e))
                    })?;
            }

            result_columns.push(out_col.into());
        }

        // Right columns: inner-right then zeros.
        for (col_idx, inner_col) in inner_right.columns.into_iter().enumerate() {
            let elem_size = right
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            let inner_bytes = (inner_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("Left outer join: inner_bytes overflow".to_string())
                })?;
            let unmatched_bytes = (unmatched_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("Left outer join: unmatched_bytes overflow".to_string())
                })?;
            let total_bytes = inner_bytes.checked_add(unmatched_bytes).ok_or_else(|| {
                XlogError::Kernel("Left outer join: total_bytes overflow".to_string())
            })?;

            let mut out_col = self.memory.alloc::<u8>(total_bytes)?;

            if total_bytes > 0 {
                device.memset_zeros(&mut out_col).map_err(|e| {
                    XlogError::Kernel(format!("Failed to zero right outer column: {}", e))
                })?;
            }

            if inner_bytes > 0 {
                let mut out_view = out_col.slice_mut(0..inner_bytes);
                device.dtod_copy(&inner_col, &mut out_view).map_err(|e| {
                    XlogError::Kernel(format!("Failed to copy inner right column: {}", e))
                })?;
            }

            result_columns.push(out_col.into());
        }

        self.device.synchronize()?;

        Ok(CudaBuffer::from_columns(
            result_columns,
            total_rows,
            combined_schema,
        ))
    }

    /// Helper for left outer join with empty right: all left rows with null right columns
    fn left_outer_with_nulls(&self, left: &CudaBuffer, right: &CudaBuffer) -> Result<CudaBuffer> {
        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let num_rows = left.num_rows();
        let device = self.device.inner();

        let mut result_columns = Vec::with_capacity(combined_schema.arity());

        // Copy all left columns device-to-device
        for col_idx in 0..left.arity() {
            let col = left
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Left column {} not found", col_idx)))?;

            let elem_size = left
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            let bytes = (num_rows as usize) * elem_size;
            let mut dst_col = self.memory.alloc::<u8>(bytes)?;
            if bytes > 0 {
                device
                    .dtod_copy(col, &mut dst_col)
                    .map_err(|e| XlogError::Kernel(format!("Failed to copy left column: {}", e)))?;
            }

            result_columns.push(dst_col.into());
        }

        // Create null (zero) columns for right side on-device
        for col_idx in 0..right.arity() {
            let elem_size = right
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            let bytes = (num_rows as usize) * elem_size;
            let mut dst_col = self.memory.alloc::<u8>(bytes)?;
            if bytes > 0 {
                device
                    .memset_zeros(&mut dst_col)
                    .map_err(|e| XlogError::Kernel(format!("Failed to zero null column: {}", e)))?;
            }

            result_columns.push(dst_col.into());
        }

        self.device.synchronize()?;

        Ok(CudaBuffer::from_columns(
            result_columns,
            num_rows,
            combined_schema,
        ))
    }

    /// Clone a buffer (deep copy) via host (MVP simplification)
    fn clone_buffer(&self, buffer: &CudaBuffer) -> Result<CudaBuffer> {
        let mut result_columns = Vec::with_capacity(buffer.arity());

        for col_idx in 0..buffer.arity() {
            let col_type_size = buffer
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let bytes = (buffer.num_rows() as usize) * col_type_size;

            if let Some(src_col) = buffer.column(col_idx) {
                // Copy via host to avoid type mismatch
                let mut host_data = vec![0u8; bytes];
                self.device
                    .inner()
                    .dtoh_sync_copy_into(src_col, &mut host_data)
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to read column for clone: {}", e))
                    })?;

                let mut dst_col = self.memory.alloc::<u8>(bytes)?;
                self.device
                    .inner()
                    .htod_sync_copy_into(&host_data, &mut dst_col)
                    .map_err(|e| XlogError::Kernel(format!("Failed to clone column: {}", e)))?;
                result_columns.push(dst_col.into());
            }
        }

        Ok(CudaBuffer::from_columns(
            result_columns,
            buffer.num_rows(),
            buffer.schema().clone(),
        ))
    }

    // ============== Arithmetic Operations (GPU) ==============

    /// Extract a single column from a buffer as a new single-column buffer
    ///
    /// # Arguments
    /// * `buffer` - The source buffer
    /// * `col_idx` - The column index to extract
    ///
    /// # Returns
    /// A new single-column CudaBuffer containing just the specified column
    pub fn extract_column(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<CudaBuffer> {
        if buffer.is_empty() {
            let col_type = buffer
                .schema()
                .column_type(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;
            let schema = Schema::new(vec![("col".to_string(), col_type)]);
            return self.create_empty_buffer(schema);
        }

        let col_type = buffer
            .schema()
            .column_type(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;
        let col_type_size = col_type.size_bytes();
        let num_rows = buffer.num_rows();
        let bytes = (num_rows as usize) * col_type_size;

        let src_col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found in buffer", col_idx)))?;

        // Copy via host
        let mut host_data = vec![0u8; bytes];
        self.device
            .inner()
            .dtoh_sync_copy_into(src_col, &mut host_data)
            .map_err(|e| XlogError::Kernel(format!("Failed to read column: {}", e)))?;

        let mut dst_col = self.memory.alloc::<u8>(bytes)?;
        self.device
            .inner()
            .htod_sync_copy_into(&host_data, &mut dst_col)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy column: {}", e)))?;

        let schema = Schema::new(vec![("col".to_string(), col_type)]);
        Ok(CudaBuffer::from_columns(vec![dst_col.into()], num_rows, schema))
    }

    /// Create a column filled with a constant value
    ///
    /// # Arguments
    /// * `value_bytes` - The raw bytes of the constant value (in little-endian format)
    /// * `col_type` - The ScalarType of the column
    /// * `num_rows` - Number of rows to create
    ///
    /// # Returns
    /// A new single-column CudaBuffer filled with the constant value
    pub fn create_constant_column(
        &self,
        value_bytes: &[u8],
        col_type: ScalarType,
        num_rows: u64,
    ) -> Result<CudaBuffer> {
        if num_rows == 0 {
            let schema = Schema::new(vec![("const".to_string(), col_type)]);
            return self.create_empty_buffer(schema);
        }

        let elem_size = col_type.size_bytes();
        if value_bytes.len() != elem_size {
            return Err(XlogError::Kernel(format!(
                "Value bytes length {} doesn't match type size {}",
                value_bytes.len(),
                elem_size
            )));
        }

        if num_rows > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Constant column supports at most {} rows, got {}",
                u32::MAX,
                num_rows
            )));
        }

        let total_bytes = (num_rows as usize)
            .checked_mul(elem_size)
            .ok_or_else(|| XlogError::Kernel("Constant column size overflow".to_string()))?;

        let mut dst_col = self.memory.alloc::<u8>(total_bytes)?;
        let n = num_rows as u32;

        macro_rules! launch_fill_const {
            ($kernel:expr, $value:expr) => {{
                let func = self
                    .device
                    .inner()
                    .get_func(ARITH_MODULE, $kernel)
                    .ok_or_else(|| {
                        XlogError::Kernel("arith fill kernel not found".to_string())
                    })?;
                let config = LaunchConfig::for_num_elems(n);
                unsafe { func.clone().launch(config, ($value, n, &mut dst_col)) }
                    .map_err(|e| XlogError::Kernel(format!("fill const failed: {}", e)))?;
            }};
        }

        match col_type {
            ScalarType::U32 | ScalarType::Symbol => {
                let value = u32::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_U32, value);
            }
            ScalarType::U64 => {
                let value = u64::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_U64, value);
            }
            ScalarType::I64 => {
                let value = i64::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_I64, value);
            }
            ScalarType::I32 => {
                let value = i32::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_I32, value);
            }
            ScalarType::F64 => {
                let value = f64::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_F64, value);
            }
            ScalarType::F32 => {
                let value = f32::from_le_bytes(value_bytes.try_into().unwrap());
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_F32, value);
            }
            ScalarType::Bool => {
                let value = value_bytes[0];
                launch_fill_const!(arith_kernels::ARITH_FILL_CONST_U8, value);
            }
        }

        self.device.synchronize()?;

        let schema = Schema::new(vec![("const".to_string(), col_type)]);
        Ok(CudaBuffer::from_columns(vec![dst_col.into()], num_rows, schema))
    }

    /// Element-wise addition of two single-column buffers
    ///
    /// Performs element-wise addition using GPU kernels.
    /// Uses wrapping arithmetic for integer overflow.
    ///
    /// # Arguments
    /// * `a` - First operand buffer (single column)
    /// * `b` - Second operand buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise sum
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn add_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => self.binary_arith_op_device::<i64>(
                a,
                b,
                0,
                arith_kernels::ARITH_BINARY_I64,
            ),
            Some(ScalarType::I32) => self.binary_arith_op_device::<i32>(
                a,
                b,
                0,
                arith_kernels::ARITH_BINARY_I32,
            ),
            Some(ScalarType::U64) => self.binary_arith_op_device::<u64>(
                a,
                b,
                0,
                arith_kernels::ARITH_BINARY_U64,
            ),
            Some(ScalarType::U32 | ScalarType::Symbol) => self.binary_arith_op_device::<u32>(
                a,
                b,
                0,
                arith_kernels::ARITH_BINARY_U32,
            ),
            Some(ScalarType::F64) => self.binary_arith_op_device::<f64>(
                a,
                b,
                0,
                arith_kernels::ARITH_BINARY_F64,
            ),
            Some(ScalarType::F32) => self.binary_arith_op_device::<f32>(
                a,
                b,
                0,
                arith_kernels::ARITH_BINARY_F32,
            ),
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise subtraction of two single-column buffers
    ///
    /// Performs element-wise subtraction using GPU kernels.
    /// Uses wrapping arithmetic for integer overflow.
    ///
    /// # Arguments
    /// * `a` - First operand buffer (single column)
    /// * `b` - Second operand buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise difference
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn sub_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => self.binary_arith_op_device::<i64>(
                a,
                b,
                1,
                arith_kernels::ARITH_BINARY_I64,
            ),
            Some(ScalarType::I32) => self.binary_arith_op_device::<i32>(
                a,
                b,
                1,
                arith_kernels::ARITH_BINARY_I32,
            ),
            Some(ScalarType::U64) => self.binary_arith_op_device::<u64>(
                a,
                b,
                1,
                arith_kernels::ARITH_BINARY_U64,
            ),
            Some(ScalarType::U32 | ScalarType::Symbol) => self.binary_arith_op_device::<u32>(
                a,
                b,
                1,
                arith_kernels::ARITH_BINARY_U32,
            ),
            Some(ScalarType::F64) => self.binary_arith_op_device::<f64>(
                a,
                b,
                1,
                arith_kernels::ARITH_BINARY_F64,
            ),
            Some(ScalarType::F32) => self.binary_arith_op_device::<f32>(
                a,
                b,
                1,
                arith_kernels::ARITH_BINARY_F32,
            ),
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise multiplication of two single-column buffers
    ///
    /// Performs element-wise multiplication using GPU kernels.
    /// Uses wrapping arithmetic for integer overflow.
    ///
    /// # Arguments
    /// * `a` - First operand buffer (single column)
    /// * `b` - Second operand buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise product
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn mul_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => self.binary_arith_op_device::<i64>(
                a,
                b,
                2,
                arith_kernels::ARITH_BINARY_I64,
            ),
            Some(ScalarType::I32) => self.binary_arith_op_device::<i32>(
                a,
                b,
                2,
                arith_kernels::ARITH_BINARY_I32,
            ),
            Some(ScalarType::U64) => self.binary_arith_op_device::<u64>(
                a,
                b,
                2,
                arith_kernels::ARITH_BINARY_U64,
            ),
            Some(ScalarType::U32 | ScalarType::Symbol) => self.binary_arith_op_device::<u32>(
                a,
                b,
                2,
                arith_kernels::ARITH_BINARY_U32,
            ),
            Some(ScalarType::F64) => self.binary_arith_op_device::<f64>(
                a,
                b,
                2,
                arith_kernels::ARITH_BINARY_F64,
            ),
            Some(ScalarType::F32) => self.binary_arith_op_device::<f32>(
                a,
                b,
                2,
                arith_kernels::ARITH_BINARY_F32,
            ),
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise division of two single-column buffers
    ///
    /// Performs element-wise division using GPU kernels.
    /// For signed integers, division by zero returns i64::MAX/i32::MAX.
    /// For unsigned integers, division by zero returns u64::MAX/u32::MAX.
    /// For floats, division by zero produces Inf/NaN as per IEEE 754.
    ///
    /// # Arguments
    /// * `a` - Dividend buffer (single column)
    /// * `b` - Divisor buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise quotient
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn div_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => self.binary_arith_op_device::<i64>(
                a,
                b,
                3,
                arith_kernels::ARITH_BINARY_I64,
            ),
            Some(ScalarType::I32) => self.binary_arith_op_device::<i32>(
                a,
                b,
                3,
                arith_kernels::ARITH_BINARY_I32,
            ),
            Some(ScalarType::U64) => self.binary_arith_op_device::<u64>(
                a,
                b,
                3,
                arith_kernels::ARITH_BINARY_U64,
            ),
            Some(ScalarType::U32 | ScalarType::Symbol) => self.binary_arith_op_device::<u32>(
                a,
                b,
                3,
                arith_kernels::ARITH_BINARY_U32,
            ),
            Some(ScalarType::F64) => self.binary_arith_op_device::<f64>(
                a,
                b,
                3,
                arith_kernels::ARITH_BINARY_F64,
            ),
            Some(ScalarType::F32) => self.binary_arith_op_device::<f32>(
                a,
                b,
                3,
                arith_kernels::ARITH_BINARY_F32,
            ),
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise modulo of two single-column buffers
    ///
    /// Performs element-wise modulo using GPU kernels.
    /// For integers, modulo by zero returns 0.
    /// For floats, modulo by zero produces NaN as per IEEE 754.
    ///
    /// # Arguments
    /// * `a` - Dividend buffer (single column)
    /// * `b` - Divisor buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise remainder
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn mod_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => self.binary_arith_op_device::<i64>(
                a,
                b,
                4,
                arith_kernels::ARITH_BINARY_I64,
            ),
            Some(ScalarType::I32) => self.binary_arith_op_device::<i32>(
                a,
                b,
                4,
                arith_kernels::ARITH_BINARY_I32,
            ),
            Some(ScalarType::U64) => self.binary_arith_op_device::<u64>(
                a,
                b,
                4,
                arith_kernels::ARITH_BINARY_U64,
            ),
            Some(ScalarType::U32 | ScalarType::Symbol) => self.binary_arith_op_device::<u32>(
                a,
                b,
                4,
                arith_kernels::ARITH_BINARY_U32,
            ),
            Some(ScalarType::F64) => self.binary_arith_op_device::<f64>(
                a,
                b,
                4,
                arith_kernels::ARITH_BINARY_F64,
            ),
            Some(ScalarType::F32) => self.binary_arith_op_device::<f32>(
                a,
                b,
                4,
                arith_kernels::ARITH_BINARY_F32,
            ),
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise absolute value of a single-column buffer
    ///
    /// Performs element-wise absolute value using GPU kernels.
    ///
    /// # Arguments
    /// * `a` - Input buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the absolute values
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Buffer is not single-column
    /// - Type is not supported for arithmetic
    pub fn abs_column(&self, a: &CudaBuffer) -> Result<CudaBuffer> {
        if a.arity() != 1 {
            return Err(XlogError::Kernel(
                "Arithmetic requires single-column buffers".into(),
            ));
        }

        if a.num_rows() == 0 {
            return self.create_empty_buffer(a.schema().clone());
        }

        let n: u32 = a.num_rows().try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "abs_column: row count {} exceeds u32::MAX",
                a.num_rows()
            ))
        })?;
        let col = a
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing column 0".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        match a.schema().column_type(0) {
            Some(ScalarType::I64) => {
                let expected_bytes = (n as usize)
                    .checked_mul(std::mem::size_of::<i64>())
                    .ok_or_else(|| XlogError::Kernel("abs_column size overflow".into()))?;
                if col.num_bytes() != expected_bytes {
                    return Err(XlogError::Kernel(format!(
                        "Column 0 has {} bytes but expected {} for {} rows",
                        col.num_bytes(),
                        expected_bytes,
                        a.num_rows()
                    )));
                }
                let mut out = self.memory.alloc::<u8>(expected_bytes)?;
                let func = self
                    .device
                    .inner()
                    .get_func(ARITH_MODULE, arith_kernels::ARITH_ABS_I64)
                    .ok_or_else(|| XlogError::Kernel("arith_abs_i64 not found".into()))?;
                unsafe { func.clone().launch(config, (col, n, &mut out)) }
                    .map_err(|e| XlogError::Kernel(format!("abs_i64 failed: {}", e)))?;
                self.device.synchronize()?;
                Ok(CudaBuffer::from_columns(vec![out.into()], a.num_rows(), a.schema.clone()))
            }
            Some(ScalarType::I32) => {
                let expected_bytes = (n as usize)
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or_else(|| XlogError::Kernel("abs_column size overflow".into()))?;
                if col.num_bytes() != expected_bytes {
                    return Err(XlogError::Kernel(format!(
                        "Column 0 has {} bytes but expected {} for {} rows",
                        col.num_bytes(),
                        expected_bytes,
                        a.num_rows()
                    )));
                }
                let mut out = self.memory.alloc::<u8>(expected_bytes)?;
                let func = self
                    .device
                    .inner()
                    .get_func(ARITH_MODULE, arith_kernels::ARITH_ABS_I32)
                    .ok_or_else(|| XlogError::Kernel("arith_abs_i32 not found".into()))?;
                unsafe { func.clone().launch(config, (col, n, &mut out)) }
                    .map_err(|e| XlogError::Kernel(format!("abs_i32 failed: {}", e)))?;
                self.device.synchronize()?;
                Ok(CudaBuffer::from_columns(vec![out.into()], a.num_rows(), a.schema.clone()))
            }
            Some(ScalarType::F64) => {
                let expected_bytes = (n as usize)
                    .checked_mul(std::mem::size_of::<f64>())
                    .ok_or_else(|| XlogError::Kernel("abs_column size overflow".into()))?;
                if col.num_bytes() != expected_bytes {
                    return Err(XlogError::Kernel(format!(
                        "Column 0 has {} bytes but expected {} for {} rows",
                        col.num_bytes(),
                        expected_bytes,
                        a.num_rows()
                    )));
                }
                let mut out = self.memory.alloc::<u8>(expected_bytes)?;
                let func = self
                    .device
                    .inner()
                    .get_func(ARITH_MODULE, arith_kernels::ARITH_ABS_F64)
                    .ok_or_else(|| XlogError::Kernel("arith_abs_f64 not found".into()))?;
                unsafe { func.clone().launch(config, (col, n, &mut out)) }
                    .map_err(|e| XlogError::Kernel(format!("abs_f64 failed: {}", e)))?;
                self.device.synchronize()?;
                Ok(CudaBuffer::from_columns(vec![out.into()], a.num_rows(), a.schema.clone()))
            }
            Some(ScalarType::F32) => {
                let expected_bytes = (n as usize)
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| XlogError::Kernel("abs_column size overflow".into()))?;
                if col.num_bytes() != expected_bytes {
                    return Err(XlogError::Kernel(format!(
                        "Column 0 has {} bytes but expected {} for {} rows",
                        col.num_bytes(),
                        expected_bytes,
                        a.num_rows()
                    )));
                }
                let mut out = self.memory.alloc::<u8>(expected_bytes)?;
                let func = self
                    .device
                    .inner()
                    .get_func(ARITH_MODULE, arith_kernels::ARITH_ABS_F32)
                    .ok_or_else(|| XlogError::Kernel("arith_abs_f32 not found".into()))?;
                unsafe { func.clone().launch(config, (col, n, &mut out)) }
                    .map_err(|e| XlogError::Kernel(format!("abs_f32 failed: {}", e)))?;
                self.device.synchronize()?;
                Ok(CudaBuffer::from_columns(vec![out.into()], a.num_rows(), a.schema.clone()))
            }
            Some(ScalarType::U32 | ScalarType::U64 | ScalarType::Bool | ScalarType::Symbol) => {
                self.clone_buffer(a)
            }
            other => Err(XlogError::Kernel(format!(
                "Abs not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise minimum of two single-column buffers
    ///
    /// Performs element-wise minimum using GPU kernels.
    ///
    /// # Arguments
    /// * `a` - First operand buffer (single column)
    /// * `b` - Second operand buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise minimums
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn min_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => self.binary_arith_op_device::<i64>(
                a,
                b,
                5,
                arith_kernels::ARITH_BINARY_I64,
            ),
            Some(ScalarType::I32) => self.binary_arith_op_device::<i32>(
                a,
                b,
                5,
                arith_kernels::ARITH_BINARY_I32,
            ),
            Some(ScalarType::U64) => self.binary_arith_op_device::<u64>(
                a,
                b,
                5,
                arith_kernels::ARITH_BINARY_U64,
            ),
            Some(ScalarType::U32 | ScalarType::Symbol) => self.binary_arith_op_device::<u32>(
                a,
                b,
                5,
                arith_kernels::ARITH_BINARY_U32,
            ),
            Some(ScalarType::F64) => self.binary_arith_op_device::<f64>(
                a,
                b,
                5,
                arith_kernels::ARITH_BINARY_F64,
            ),
            Some(ScalarType::F32) => self.binary_arith_op_device::<f32>(
                a,
                b,
                5,
                arith_kernels::ARITH_BINARY_F32,
            ),
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise maximum of two single-column buffers
    ///
    /// Performs element-wise maximum using GPU kernels.
    ///
    /// # Arguments
    /// * `a` - First operand buffer (single column)
    /// * `b` - Second operand buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise maximums
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn max_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        match a.schema().column_type(0) {
            Some(ScalarType::I64) => self.binary_arith_op_device::<i64>(
                a,
                b,
                6,
                arith_kernels::ARITH_BINARY_I64,
            ),
            Some(ScalarType::I32) => self.binary_arith_op_device::<i32>(
                a,
                b,
                6,
                arith_kernels::ARITH_BINARY_I32,
            ),
            Some(ScalarType::U64) => self.binary_arith_op_device::<u64>(
                a,
                b,
                6,
                arith_kernels::ARITH_BINARY_U64,
            ),
            Some(ScalarType::U32 | ScalarType::Symbol) => self.binary_arith_op_device::<u32>(
                a,
                b,
                6,
                arith_kernels::ARITH_BINARY_U32,
            ),
            Some(ScalarType::F64) => self.binary_arith_op_device::<f64>(
                a,
                b,
                6,
                arith_kernels::ARITH_BINARY_F64,
            ),
            Some(ScalarType::F32) => self.binary_arith_op_device::<f32>(
                a,
                b,
                6,
                arith_kernels::ARITH_BINARY_F32,
            ),
            other => Err(XlogError::Kernel(format!(
                "Arithmetic not supported for {:?}",
                other
            ))),
        }
    }

    /// Element-wise power of two single-column buffers
    ///
    /// Converts both operands to f64, computes x^y on the GPU, and returns f64 result.
    /// This matches the behavior of most database systems where pow() returns a float.
    ///
    /// # Arguments
    /// * `base` - Base values buffer (single column)
    /// * `exp` - Exponent values buffer (single column)
    ///
    /// # Returns
    /// A new CudaBuffer containing the element-wise powers as f64
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Type is not supported for arithmetic
    pub fn pow_columns(&self, base: &CudaBuffer, exp: &CudaBuffer) -> Result<CudaBuffer> {
        if base.num_rows() != exp.num_rows() {
            return Err(XlogError::Kernel("Row count mismatch".into()));
        }
        if base.arity() != 1 || exp.arity() != 1 {
            return Err(XlogError::Kernel(
                "Arithmetic requires single-column buffers".into(),
            ));
        }

        if base.num_rows() == 0 {
            let schema = Schema::new(vec![("result".to_string(), ScalarType::F64)]);
            return self.create_empty_buffer(schema);
        }

        let n: u32 = base.num_rows().try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "pow_columns: row count {} exceeds u32::MAX",
                base.num_rows()
            ))
        })?;

        let base_f64_buf = if base.schema().column_type(0) == Some(ScalarType::F64) {
            None
        } else {
            Some(self.cast_column(base, ScalarType::F64)?)
        };
        let base_buf = base_f64_buf.as_ref().unwrap_or(base);

        let exp_f64_buf = if exp.schema().column_type(0) == Some(ScalarType::F64) {
            None
        } else {
            Some(self.cast_column(exp, ScalarType::F64)?)
        };
        let exp_buf = exp_f64_buf.as_ref().unwrap_or(exp);

        let base_col = base_buf
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing base column".into()))?;
        let exp_col = exp_buf
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing exp column".into()))?;

        let expected_bytes = (n as usize)
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| XlogError::Kernel("pow_columns size overflow".into()))?;
        if base_col.num_bytes() != expected_bytes || exp_col.num_bytes() != expected_bytes {
            return Err(XlogError::Kernel(format!(
                "pow_columns: expected {} bytes for {} rows",
                expected_bytes,
                base.num_rows()
            )));
        }

        let mut out = self.memory.alloc::<u8>(expected_bytes)?;
        let func = self
            .device
            .inner()
            .get_func(ARITH_MODULE, arith_kernels::ARITH_POW_F64)
            .ok_or_else(|| XlogError::Kernel("arith_pow_f64 not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        unsafe { func.clone().launch(config, (base_col, exp_col, n, &mut out)) }
            .map_err(|e| XlogError::Kernel(format!("pow_f64 failed: {}", e)))?;

        self.device.synchronize()?;

        let schema = Schema::new(vec![("result".to_string(), ScalarType::F64)]);
        Ok(CudaBuffer::from_columns(
            vec![out.into()],
            base.num_rows(),
            schema,
        ))
    }

    /// Conditional select between two single-column buffers based on a boolean mask.
    ///
    /// For each row: out[i] = mask[i] ? then_vals[i] : else_vals[i]
    ///
    /// # Arguments
    /// * `mask` - Boolean mask buffer (single column, type Bool/u8)
    /// * `then_vals` - Values to select when mask is true
    /// * `else_vals` - Values to select when mask is false
    ///
    /// # Returns
    /// A new CudaBuffer with values selected based on the mask
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Row counts don't match
    /// - Buffers are not single-column
    /// - Types of then/else values don't match
    pub fn select_columns(
        &self,
        mask: &CudaBuffer,
        then_vals: &CudaBuffer,
        else_vals: &CudaBuffer,
    ) -> Result<CudaBuffer> {
        if mask.num_rows() != then_vals.num_rows() || mask.num_rows() != else_vals.num_rows() {
            return Err(XlogError::Kernel("Row count mismatch in select".into()));
        }
        if mask.arity() != 1 || then_vals.arity() != 1 || else_vals.arity() != 1 {
            return Err(XlogError::Kernel(
                "Select requires single-column buffers".into(),
            ));
        }

        let then_type = then_vals.schema().column_type(0);
        let else_type = else_vals.schema().column_type(0);
        if then_type != else_type {
            return Err(XlogError::Kernel(format!(
                "Type mismatch in select: then={:?}, else={:?}",
                then_type, else_type
            )));
        }

        if mask.num_rows() == 0 {
            let result_type = then_type.unwrap_or(ScalarType::I64);
            let schema = Schema::new(vec![("result".to_string(), result_type)]);
            return self.create_empty_buffer(schema);
        }

        let n: u32 = mask.num_rows().try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "select_columns: row count {} exceeds u32::MAX",
                mask.num_rows()
            ))
        })?;

        let mask_col = mask
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing mask column".into()))?;
        let then_col = then_vals
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing then column".into()))?;
        let else_col = else_vals
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing else column".into()))?;

        let result_type = then_type.unwrap_or(ScalarType::I64);
        let elem_size = result_type.size_bytes();
        let expected_bytes = (n as usize)
            .checked_mul(elem_size)
            .ok_or_else(|| XlogError::Kernel("select_columns size overflow".into()))?;

        let mut out = self.memory.alloc::<u8>(expected_bytes)?;

        let kernel_name = match result_type {
            ScalarType::I64 => arith_kernels::ARITH_SELECT_I64,
            ScalarType::I32 => arith_kernels::ARITH_SELECT_I32,
            ScalarType::U64 => arith_kernels::ARITH_SELECT_U64,
            ScalarType::U32 | ScalarType::Symbol => arith_kernels::ARITH_SELECT_U32,
            ScalarType::F64 => arith_kernels::ARITH_SELECT_F64,
            ScalarType::F32 => arith_kernels::ARITH_SELECT_F32,
            ScalarType::Bool => {
                // Bool is stored as u8, treat as u8 select (use fill + mask trick)
                // For simplicity, cast to u32 and back
                return self.select_columns_bool(mask, then_vals, else_vals);
            }
        };

        let func = self
            .device
            .inner()
            .get_func(ARITH_MODULE, kernel_name)
            .ok_or_else(|| XlogError::Kernel(format!("{} not found", kernel_name)))?;
        let config = LaunchConfig::for_num_elems(n);

        unsafe { func.clone().launch(config, (mask_col, then_col, else_col, n, &mut out)) }
            .map_err(|e| XlogError::Kernel(format!("select kernel failed: {}", e)))?;

        self.device.synchronize()?;

        let schema = Schema::new(vec![("result".to_string(), result_type)]);
        Ok(CudaBuffer::from_columns(
            vec![out.into()],
            mask.num_rows(),
            schema,
        ))
    }

    /// Helper for select_columns when result type is Bool
    fn select_columns_bool(
        &self,
        mask: &CudaBuffer,
        then_vals: &CudaBuffer,
        else_vals: &CudaBuffer,
    ) -> Result<CudaBuffer> {
        // Cast bool columns to u32, select, then cast back
        let then_u32 = self.cast_column(then_vals, ScalarType::U32)?;
        let else_u32 = self.cast_column(else_vals, ScalarType::U32)?;
        let result_u32 = self.select_columns(mask, &then_u32, &else_u32)?;
        self.cast_column(&result_u32, ScalarType::Bool)
    }

    /// Cast a single-column buffer to a different type
    ///
    /// Casts data on the GPU using the arithmetic cast kernel.
    ///
    /// # Arguments
    /// * `a` - Input buffer (single column)
    /// * `target` - Target scalar type
    ///
    /// # Returns
    /// A new CudaBuffer with the cast values
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Buffer is not single-column
    /// - Source or target type is not supported for casting
    pub fn cast_column(&self, a: &CudaBuffer, target: ScalarType) -> Result<CudaBuffer> {
        if a.arity() != 1 {
            return Err(XlogError::Kernel(
                "Cast requires single-column buffer".into(),
            ));
        }

        let source_type = a
            .schema()
            .column_type(0)
            .ok_or_else(|| XlogError::Kernel("Missing column type".into()))?;

        let schema = Schema::new(vec![("result".to_string(), target)]);

        if a.num_rows() == 0 {
            return self.create_empty_buffer(schema);
        }

        let n: u32 = a.num_rows().try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "cast_column: row count {} exceeds u32::MAX",
                a.num_rows()
            ))
        })?;

        let src_col = a
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing column 0".into()))?;
        let src_bytes = (n as usize)
            .checked_mul(source_type.size_bytes())
            .ok_or_else(|| XlogError::Kernel("cast_column size overflow".into()))?;
        if src_col.num_bytes() != src_bytes {
            return Err(XlogError::Kernel(format!(
                "Column 0 has {} bytes but expected {} for {} rows",
                src_col.num_bytes(),
                src_bytes,
                a.num_rows()
            )));
        }

        let dst_bytes = (n as usize)
            .checked_mul(target.size_bytes())
            .ok_or_else(|| XlogError::Kernel("cast_column size overflow".into()))?;
        let mut out = self.memory.alloc::<u8>(dst_bytes)?;

        let func = self
            .device
            .inner()
            .get_func(ARITH_MODULE, arith_kernels::ARITH_CAST)
            .ok_or_else(|| XlogError::Kernel("arith_cast not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        unsafe {
            func.clone()
                .launch(config, (src_col, &mut out, n, source_type.to_code(), target.to_code()))
        }
        .map_err(|e| XlogError::Kernel(format!("cast failed: {}", e)))?;

        self.device.synchronize()?;

        Ok(CudaBuffer::from_columns(vec![out.into()], a.num_rows(), schema))
    }

    /// Helper for binary arithmetic operations on device.
    fn binary_arith_op_device<T: DeviceRepr>(
        &self,
        a: &CudaBuffer,
        b: &CudaBuffer,
        op: u8,
        kernel: &str,
    ) -> Result<CudaBuffer> {
        if a.num_rows() != b.num_rows() {
            return Err(XlogError::Kernel("Row count mismatch".into()));
        }
        if a.arity() != 1 || b.arity() != 1 {
            return Err(XlogError::Kernel(
                "Arithmetic requires single-column buffers".into(),
            ));
        }
        if a.schema().column_type(0) != b.schema().column_type(0) {
            return Err(XlogError::Kernel(
                "Arithmetic requires matching column types".into(),
            ));
        }
        if a.num_rows() == 0 {
            return self.create_empty_buffer(a.schema.clone());
        }

        let n: u32 = a.num_rows().try_into().map_err(|_| {
            XlogError::Kernel(format!(
                "arith: row count {} exceeds u32::MAX",
                a.num_rows()
            ))
        })?;

        let expected_bytes = (n as usize)
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| XlogError::Kernel("arith output size overflow".into()))?;

        let col_a = a
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing column 0".into()))?;
        let col_b = b
            .column(0)
            .ok_or_else(|| XlogError::Kernel("Missing column 0".into()))?;

        if col_a.num_bytes() != expected_bytes || col_b.num_bytes() != expected_bytes {
            return Err(XlogError::Kernel(format!(
                "Arithmetic expects {} bytes per column for {} rows",
                expected_bytes,
                a.num_rows()
            )));
        }

        let mut out = self.memory.alloc::<u8>(expected_bytes)?;
        let func = self
            .device
            .inner()
            .get_func(ARITH_MODULE, kernel)
            .ok_or_else(|| XlogError::Kernel("arith kernel not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        unsafe { func.clone().launch(config, (col_a, col_b, n, op, &mut out)) }
            .map_err(|e| XlogError::Kernel(format!("arith binary failed: {}", e)))?;

        self.device.synchronize()?;
        Ok(CudaBuffer::from_columns(
            vec![out.into()],
            a.num_rows(),
            a.schema.clone(),
        ))
    }

    /// Combine multiple single-column buffers into a multi-column buffer
    ///
    /// # Arguments
    /// * `columns` - Vector of single-column CudaBuffers to combine
    /// * `types` - Vector of ScalarTypes for each column
    ///
    /// # Returns
    /// A new CudaBuffer with all columns combined
    pub fn combine_columns(
        &self,
        columns: Vec<CudaBuffer>,
        types: Vec<ScalarType>,
    ) -> Result<CudaBuffer> {
        if columns.is_empty() {
            return Ok(CudaBuffer::empty());
        }

        let num_rows = columns[0].num_rows();

        // Verify all columns have the same number of rows
        for (i, col) in columns.iter().enumerate() {
            if col.num_rows() != num_rows {
                return Err(XlogError::Kernel(format!(
                    "Column {} has {} rows, expected {}",
                    i,
                    col.num_rows(),
                    num_rows
                )));
            }
        }

        if num_rows == 0 {
            let schema_cols: Vec<(String, ScalarType)> = types
                .iter()
                .enumerate()
                .map(|(i, t)| (format!("col_{}", i), *t))
                .collect();
            let schema = Schema::new(schema_cols);
            return self.create_empty_buffer(schema);
        }

        // Extract each single-column buffer's data
        let mut result_columns = Vec::with_capacity(columns.len());
        for (i, col_buf) in columns.iter().enumerate() {
            let col_type = types.get(i).copied().unwrap_or(ScalarType::U64);
            let col_type_size = col_type.size_bytes();
            let bytes = (num_rows as usize) * col_type_size;

            // Each input buffer should have exactly one column
            let src_col = col_buf
                .column(0)
                .ok_or_else(|| XlogError::Kernel(format!("Column {} buffer has no data", i)))?;

            // Copy via host
            let mut host_data = vec![0u8; bytes];
            self.device
                .inner()
                .dtoh_sync_copy_into(src_col, &mut host_data)
                .map_err(|e| XlogError::Kernel(format!("Failed to read column {}: {}", i, e)))?;

            let mut dst_col = self.memory.alloc::<u8>(bytes)?;
            self.device
                .inner()
                .htod_sync_copy_into(&host_data, &mut dst_col)
                .map_err(|e| XlogError::Kernel(format!("Failed to copy column {}: {}", i, e)))?;

            result_columns.push(dst_col.into());
        }

        let schema_cols: Vec<(String, ScalarType)> = types
            .iter()
            .enumerate()
            .map(|(i, t)| (format!("col_{}", i), *t))
            .collect();
        let schema = Schema::new(schema_cols);

        Ok(CudaBuffer::from_columns(result_columns, num_rows, schema))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::MemoryBudget;

    fn has_cuda_device() -> bool {
        CudaDevice::new(0).is_ok()
    }

    #[test]
    fn test_ptx_embedded() {
        // Verify PTX sources are embedded and non-empty
        assert!(!JOIN_PTX.is_empty(), "JOIN_PTX should not be empty");
        assert!(!DEDUP_PTX.is_empty(), "DEDUP_PTX should not be empty");
        assert!(!GROUPBY_PTX.is_empty(), "GROUPBY_PTX should not be empty");
        assert!(!CIRCUIT_PTX.is_empty(), "CIRCUIT_PTX should not be empty");
        assert!(!MC_SAMPLE_PTX.is_empty(), "MC_SAMPLE_PTX should not be empty");
        assert!(!SAT_PTX.is_empty(), "SAT_PTX should not be empty");
        assert!(!NEURAL_PTX.is_empty(), "NEURAL_PTX should not be empty");

        // Verify PTX contains expected kernel names
        assert!(
            JOIN_PTX.contains("hash_join_build"),
            "JOIN_PTX should contain hash_join_build"
        );
        assert!(
            JOIN_PTX.contains("hash_join_probe"),
            "JOIN_PTX should contain hash_join_probe"
        );
        assert!(
            DEDUP_PTX.contains("mark_duplicates"),
            "DEDUP_PTX should contain mark_duplicates"
        );
        assert!(
            DEDUP_PTX.contains("compact_rows"),
            "DEDUP_PTX should contain compact_rows"
        );
        assert!(
            GROUPBY_PTX.contains("detect_group_boundaries"),
            "GROUPBY_PTX should contain detect_group_boundaries"
        );
        assert!(
            GROUPBY_PTX.contains("groupby_count"),
            "GROUPBY_PTX should contain groupby_count"
        );
        assert!(
            GROUPBY_PTX.contains("groupby_sum"),
            "GROUPBY_PTX should contain groupby_sum"
        );
        assert!(
            GROUPBY_PTX.contains("groupby_min"),
            "GROUPBY_PTX should contain groupby_min"
        );
        assert!(
            GROUPBY_PTX.contains("groupby_max"),
            "GROUPBY_PTX should contain groupby_max"
        );
        assert!(
            CIRCUIT_PTX.contains("xgcf_forward_level"),
            "CIRCUIT_PTX should contain xgcf_forward_level"
        );
        assert!(
            CIRCUIT_PTX.contains("xgcf_backward_level_propagate"),
            "CIRCUIT_PTX should contain xgcf_backward_level_propagate"
        );
        assert!(
            CIRCUIT_PTX.contains("xgcf_backward_level_decision_grad"),
            "CIRCUIT_PTX should contain xgcf_backward_level_decision_grad"
        );
        assert!(
            CIRCUIT_PTX.contains("xgcf_backward_level_lit_grad"),
            "CIRCUIT_PTX should contain xgcf_backward_level_lit_grad"
        );
        assert!(
            MC_SAMPLE_PTX.contains("mc_sample_bernoulli"),
            "MC_SAMPLE_PTX should contain mc_sample_bernoulli"
        );
        assert!(
            SAT_PTX.contains("sat_cdcl_solve"),
            "SAT_PTX should contain sat_cdcl_solve"
        );
        assert!(
            SAT_PTX.contains("sat_check_model"),
            "SAT_PTX should contain sat_check_model"
        );
        assert!(
            SAT_PTX.contains("sat_proof_check"),
            "SAT_PTX should contain sat_proof_check"
        );
        assert!(
            SAT_PTX.contains("sat_assert_status"),
            "SAT_PTX should contain sat_assert_status"
        );
        assert!(
            SAT_PTX.contains("sat_assert_ok"),
            "SAT_PTX should contain sat_assert_ok"
        );
        assert!(
            SAT_PTX.contains("sat_xgcf_cnf_counts"),
            "SAT_PTX should contain sat_xgcf_cnf_counts"
        );
        assert!(
            SAT_PTX.contains("sat_xgcf_cnf_emit"),
            "SAT_PTX should contain sat_xgcf_cnf_emit"
        );
        assert!(
            SAT_PTX.contains("sat_xgcf_cnf_capture_last_counts"),
            "SAT_PTX should contain sat_xgcf_cnf_capture_last_counts"
        );
        assert!(
            SAT_PTX.contains("sat_xgcf_cnf_compute_totals"),
            "SAT_PTX should contain sat_xgcf_cnf_compute_totals"
        );
        assert!(
            SAT_PTX.contains("sat_cnf_write_terminator"),
            "SAT_PTX should contain sat_cnf_write_terminator"
        );
        assert!(
            SAT_PTX.contains("sat_cnf_copy_into"),
            "SAT_PTX should contain sat_cnf_copy_into"
        );
        assert!(
            SAT_PTX.contains("sat_shift_offsets"),
            "SAT_PTX should contain sat_shift_offsets"
        );
        assert!(
            SAT_PTX.contains("sat_xgcf_write_root_unit_clause"),
            "SAT_PTX should contain sat_xgcf_write_root_unit_clause"
        );
        assert!(
            SAT_PTX.contains("sat_not_phi_counts"),
            "SAT_PTX should contain sat_not_phi_counts"
        );
        assert!(
            SAT_PTX.contains("sat_emit_not_phi"),
            "SAT_PTX should contain sat_emit_not_phi"
        );

        assert!(
            NEURAL_PTX.contains("neural_fill_ad_chain_f32"),
            "NEURAL_PTX should contain neural_fill_ad_chain_f32"
        );
        assert!(
            NEURAL_PTX.contains("neural_scatter_ad_chain_grads_f32"),
            "NEURAL_PTX should contain neural_scatter_ad_chain_grads_f32"
        );
    }

    #[test]
    fn test_ptx_target_architecture() {
        // Verify PTX is compiled for a valid CUDA architecture (sm_70 or later)
        // Note: The actual target may vary based on the build environment
        let valid_targets = [".target sm_70", ".target sm_80", ".target sm_90"];

        assert!(
            valid_targets.iter().any(|t| JOIN_PTX.contains(t)),
            "JOIN_PTX should target sm_70 or later, got: {:?}",
            JOIN_PTX.lines().find(|l| l.starts_with(".target"))
        );
        assert!(
            valid_targets.iter().any(|t| DEDUP_PTX.contains(t)),
            "DEDUP_PTX should target sm_70 or later"
        );
        assert!(
            valid_targets.iter().any(|t| GROUPBY_PTX.contains(t)),
            "GROUPBY_PTX should target sm_70 or later"
        );
        assert!(
            valid_targets.iter().any(|t| CIRCUIT_PTX.contains(t)),
            "CIRCUIT_PTX should target sm_70 or later"
        );
        assert!(
            valid_targets.iter().any(|t| MC_SAMPLE_PTX.contains(t)),
            "MC_SAMPLE_PTX should target sm_70 or later"
        );
        assert!(
            valid_targets.iter().any(|t| SAT_PTX.contains(t)),
            "SAT_PTX should target sm_70 or later"
        );
        assert!(
            valid_targets.iter().any(|t| NEURAL_PTX.contains(t)),
            "NEURAL_PTX should target sm_70 or later"
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

        let xgcf_backward_propagate = inner.get_func(CIRCUIT_MODULE, "xgcf_backward_level_propagate");
        assert!(
            xgcf_backward_propagate.is_some(),
            "xgcf_backward_level_propagate function should be accessible"
        );

        let xgcf_backward_decision_grad = inner.get_func(CIRCUIT_MODULE, "xgcf_backward_level_decision_grad");
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

        CudaBuffer::from_columns(vec![col.into()], data.len() as u64, schema)
    }

    // Helper function to create an empty buffer with correct column count
    fn create_empty_test_buffer(provider: &CudaKernelProvider, schema: Schema) -> CudaBuffer {
        let mut columns = Vec::with_capacity(schema.arity());
        for _ in 0..schema.arity() {
            columns.push(provider.memory().alloc::<u8>(0).expect("alloc").into());
        }
        CudaBuffer::from_columns(columns, 0, schema)
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

        assert_eq!(deduped.num_rows(), 3, "Should have 3 unique values");

        let result = provider.download_column_u32(&deduped, 0).unwrap();
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

        assert_eq!(
            deduped.num_rows(),
            750,
            "Should have 750 unique values (0..750)"
        );

        // Verify output is sorted
        let result = provider.download_column_u32(&deduped, 0).unwrap();
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
        let buffer_2col = CudaBuffer::from_columns(vec![col0.into(), col1.into()], 2, schema_2col);

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

        let buffer = CudaBuffer::from_columns(vec![key_col.into(), val_col.into()], 4, schema);

        // Run LogSumExp aggregation grouped by key column (0), aggregating value column (1)
        let result = provider.groupby_agg(&buffer, &[0], AggOp::LogSumExp, 1);
        assert!(
            result.is_ok(),
            "groupby_agg with LogSumExp should succeed: {:?}",
            result.err()
        );

        let result = result.unwrap();
        assert_eq!(result.num_rows(), 2, "Should have 2 groups");

        // Download results
        let result_values = provider
            .download_column_f64(&result, 1)
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
        let count_schema = provider.groupby_result_schema(&input, &[0], AggOp::Count);
        assert_eq!(count_schema.arity(), 2);
        assert_eq!(count_schema.column_type(1), Some(ScalarType::U64));

        // Sum result schema
        let sum_schema = provider.groupby_result_schema(&input, &[0], AggOp::Sum);
        assert_eq!(sum_schema.arity(), 2);
        assert_eq!(sum_schema.column_type(1), Some(ScalarType::U64));

        // Min/Max result schema
        let min_schema = provider.groupby_result_schema(&input, &[0], AggOp::Min);
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
        provider.create_buffer_from_i64_slice(data, schema).unwrap()
    }

    /// Helper to create an f64 buffer for arithmetic tests
    fn create_f64_buffer(provider: &CudaKernelProvider, data: &[f64]) -> CudaBuffer {
        let schema = Schema::new(vec![("col".to_string(), ScalarType::F64)]);
        provider.create_buffer_from_f64_slice(data, schema).unwrap()
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
        let values = provider.download_column_i64(&result, 0).unwrap();

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
        let values = provider.download_column_i64(&result, 0).unwrap();

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
        let values = provider.download_column_i64(&result, 0).unwrap();

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
        let values = provider.download_column_i64(&result, 0).unwrap();

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
        let values = provider.download_column_i64(&result, 0).unwrap();

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
        let values = provider.download_column_i64(&result, 0).unwrap();

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
        let values = provider.download_column_i64(&result, 0).unwrap();

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
        let values = provider.download_column_i64(&result, 0).unwrap();

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
        let values = provider.download_column_i64(&result, 0).unwrap();

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
        let values = provider.download_column_i64(&result, 0).unwrap();

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
        let values = provider.download_column_f64(&result, 0).unwrap();

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
        let values = provider.download_column_f64(&result, 0).unwrap();

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
        let values = provider.download_column_f64(&result, 0).unwrap();

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
        let values = provider.download_column_f64(&result, 0).unwrap();

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
        let values = provider.download_column_f64(&result, 0).unwrap();

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
        let values = provider.download_column_f64(&result, 0).unwrap();

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
        let values = provider.download_column_i64(&result, 0).unwrap();

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
        let values = provider.download_column_i32(&result, 0).unwrap();

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
        let values = provider.download_column_i64(&result, 0).unwrap();

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
        let add_values = provider.download_column_i64(&add_result, 0).unwrap();
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
        let values = provider.download_column_f64(&result, 0).unwrap();

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
        let min_values = provider.download_column_f64(&min_result, 0).unwrap();
        assert_eq!(min_values, vec![1.5, 3.0, 3.0]);

        let max_result = provider.max_columns(&a, &b).unwrap();
        let max_values = provider.download_column_f64(&max_result, 0).unwrap();
        assert_eq!(max_values, vec![2.0, 5.0, 4.0]);
    }
}
