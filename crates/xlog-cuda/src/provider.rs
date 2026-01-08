//! CUDA kernel provider implementation
//!
//! This module provides the `CudaKernelProvider` which manages pre-compiled
//! PTX kernels for GPU execution of relational operations (join, dedup, groupby).

use std::sync::Arc;

use cudarc::driver::{CudaView, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use xlog_core::{AggOp, Result, Schema, ScalarType, XlogError};

use crate::{CudaBuffer, CudaDevice, GpuMemoryManager};

// Embedded PTX sources (pre-compiled from .cu files with nvcc -ptx --gpu-architecture=sm_90)
const JOIN_PTX: &str = include_str!("../../../kernels/join.ptx");
const DEDUP_PTX: &str = include_str!("../../../kernels/dedup.ptx");
const GROUPBY_PTX: &str = include_str!("../../../kernels/groupby.ptx");
const SCAN_PTX: &str = include_str!("../../../kernels/scan.ptx");
const SORT_PTX: &str = include_str!("../../../kernels/sort.ptx");
const FILTER_PTX: &str = include_str!("../../../kernels/filter.ptx");

/// Module names for loaded PTX modules
pub const JOIN_MODULE: &str = "xlog_join";
pub const DEDUP_MODULE: &str = "xlog_dedup";
pub const GROUPBY_MODULE: &str = "xlog_groupby";
pub const SCAN_MODULE: &str = "xlog_scan";
pub const SORT_MODULE: &str = "xlog_sort";
pub const FILTER_MODULE: &str = "xlog_filter";

/// Kernel function names in the join module
pub mod join_kernels {
    pub const HASH_JOIN_BUILD: &str = "hash_join_build";
    pub const HASH_JOIN_PROBE: &str = "hash_join_probe";
    // V2 kernels for multi-column joins
    pub const COMPUTE_COMPOSITE_HASH: &str = "compute_composite_hash";
    pub const HASH_JOIN_BUILD_V2: &str = "hash_join_build_v2";
    pub const HASH_JOIN_PROBE_V2: &str = "hash_join_probe_v2";
    pub const HASH_JOIN_SEMI: &str = "hash_join_semi";
    pub const HASH_JOIN_ANTI: &str = "hash_join_anti";
    pub const INIT_HASH_TABLE: &str = "init_hash_table";
}

/// Kernel function names in the dedup module
pub mod dedup_kernels {
    pub const MARK_DUPLICATES: &str = "mark_duplicates";
    pub const COMPACT_ROWS: &str = "compact_rows";
}

/// Kernel function names in the groupby module
pub mod groupby_kernels {
    pub const DETECT_GROUP_BOUNDARIES: &str = "detect_group_boundaries";
    pub const GROUPBY_COUNT: &str = "groupby_count";
    pub const GROUPBY_SUM: &str = "groupby_sum";
    pub const GROUPBY_MIN: &str = "groupby_min";
    pub const GROUPBY_MAX: &str = "groupby_max";
}

/// Kernel function names in the scan module
pub mod scan_kernels {
    pub const EXCLUSIVE_SCAN_MASK: &str = "exclusive_scan_mask";
    pub const COUNT_MASK: &str = "count_mask";
}

/// Kernel function names in the sort module
pub mod sort_kernels {
    pub const RADIX_HISTOGRAM: &str = "radix_histogram";
    pub const RADIX_SCATTER: &str = "radix_scatter";
    pub const INIT_INDICES: &str = "init_indices";
    pub const APPLY_PERMUTATION_U32: &str = "apply_permutation_u32";
    pub const APPLY_PERMUTATION_BYTES: &str = "apply_permutation_bytes";
}

/// Kernel function names in the filter module
pub mod filter_kernels {
    pub const FILTER_COMPARE_U32: &str = "filter_compare_u32";
    pub const FILTER_COMPARE_I64: &str = "filter_compare_i64";
    pub const FILTER_COMPARE_F64: &str = "filter_compare_f64";
    pub const COMPACT_U32_BY_MASK: &str = "compact_u32_by_mask";
    pub const COMPACT_I64_BY_MASK: &str = "compact_i64_by_mask";
    pub const COMPACT_F64_BY_MASK: &str = "compact_f64_by_mask";
    pub const COMPACT_BYTES_BY_MASK: &str = "compact_bytes_by_mask";
    pub const MASK_AND: &str = "mask_and";
    pub const MASK_OR: &str = "mask_or";
    pub const MASK_NOT: &str = "mask_not";
}

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
}

impl CudaKernelProvider {
    /// Create a new CUDA kernel provider
    ///
    /// Loads all PTX modules (join, dedup, groupby) into the CUDA device.
    /// The modules are compiled for sm_90 (H200/Hopper architecture).
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
                    join_kernels::HASH_JOIN_BUILD_V2,
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
                &[dedup_kernels::MARK_DUPLICATES, dedup_kernels::COMPACT_ROWS],
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
                    groupby_kernels::GROUPBY_COUNT,
                    groupby_kernels::GROUPBY_SUM,
                    groupby_kernels::GROUPBY_MIN,
                    groupby_kernels::GROUPBY_MAX,
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
                    scan_kernels::EXCLUSIVE_SCAN_MASK,
                    scan_kernels::COUNT_MASK,
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
                    sort_kernels::INIT_INDICES,
                    sort_kernels::APPLY_PERMUTATION_U32,
                    sort_kernels::APPLY_PERMUTATION_BYTES,
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

        Ok(Self { device, memory })
    }

    /// Get the CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get the GPU memory manager
    pub fn memory(&self) -> &Arc<GpuMemoryManager> {
        &self.memory
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
        // Handle empty inputs
        if left.is_empty() || right.is_empty() {
            // Return empty buffer with combined schema
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        // Validate key columns
        if left_keys.is_empty() || right_keys.is_empty() {
            return Err(XlogError::Kernel("Join requires at least one key column".to_string()));
        }
        if left_keys.len() != right_keys.len() {
            return Err(XlogError::Kernel("Left and right key columns must have same length".to_string()));
        }

        // For MVP, we support single-column U32 keys only
        if left_keys.len() != 1 {
            return Err(XlogError::Kernel("MVP: Only single-column joins supported".to_string()));
        }

        let left_key_col = left_keys[0];
        let right_key_col = right_keys[0];

        // Verify column types are U32
        if left.schema().column_type(left_key_col) != Some(ScalarType::U32)
            || right.schema().column_type(right_key_col) != Some(ScalarType::U32)
        {
            return Err(XlogError::Kernel("MVP: Only U32 key columns supported".to_string()));
        }

        let num_build = right.num_rows() as u32;
        let num_probe = left.num_rows() as u32;

        // Hash table size: 2x build size for good load factor
        let hash_table_size = (num_build as usize * 2).max(1024) as u32;

        // Allocate hash table: 3 entries per slot (key, payload, head pointer)
        let hash_table_alloc_size = (hash_table_size * 3) as usize;
        let mut hash_table = self.memory.alloc::<u32>(hash_table_alloc_size)?;
        // Initialize all hash table entries to 0xFFFFFFFF (empty)
        let init_val = 0xFFFFFFFFu32;
        self.device.inner().htod_sync_copy_into(
            &vec![init_val; hash_table_alloc_size],
            &mut hash_table,
        ).map_err(|e| XlogError::Kernel(format!("Failed to init hash table: {}", e)))?;

        // Allocate next pointers for linked list collision handling
        let mut next_ptrs = self.memory.alloc::<u32>(num_build as usize)?;
        // Initialize to 0xFFFFFFFF
        self.device.inner().htod_sync_copy_into(
            &vec![init_val; num_build as usize],
            &mut next_ptrs,
        ).map_err(|e| XlogError::Kernel(format!("Failed to init next pointers: {}", e)))?;

        // Get kernel functions
        let build_func = self.device.inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_BUILD)
            .ok_or_else(|| XlogError::Kernel("hash_join_build kernel not found".to_string()))?;
        let probe_func = self.device.inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE)
            .ok_or_else(|| XlogError::Kernel("hash_join_probe kernel not found".to_string()))?;

        // For MVP, we use the key column as both key and payload
        // In a full implementation, we'd project out the specific columns
        let build_keys = right.column(right_key_col)
            .ok_or_else(|| XlogError::Kernel("Build key column not found".to_string()))?;
        let build_payloads = build_keys; // Use row index as payload for simplicity

        let probe_keys = left.column(left_key_col)
            .ok_or_else(|| XlogError::Kernel("Probe key column not found".to_string()))?;
        let probe_payloads = probe_keys;

        // Launch build kernel
        let block_size = 256u32;
        let build_grid_size = (num_build + block_size - 1) / block_size;
        let build_config = LaunchConfig {
            grid_dim: (build_grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Kernel parameters match the expected signature in join.cu
        unsafe {
            build_func.clone().launch(build_config, (
                build_keys,
                build_payloads,
                num_build,
                &hash_table,
                &next_ptrs,
                hash_table_size,
            )).map_err(|e| XlogError::Kernel(format!("Build kernel failed: {}", e)))?;
        }

        // Allocate output buffers (worst case: cross product)
        let max_output = (num_build as u64 * num_probe as u64).min(1_000_000) as usize;
        let output_left = self.memory.alloc::<u32>(max_output)?;
        let output_right = self.memory.alloc::<u32>(max_output)?;
        let mut output_count = self.memory.alloc::<u32>(1)?;
        // Initialize count to 0
        self.device.inner().htod_sync_copy_into(
            &vec![0u32],
            &mut output_count,
        ).map_err(|e| XlogError::Kernel(format!("Failed to init output count: {}", e)))?;

        // Launch probe kernel
        let probe_grid_size = (num_probe + block_size - 1) / block_size;
        let probe_config = LaunchConfig {
            grid_dim: (probe_grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Kernel parameters match the expected signature in join.cu
        unsafe {
            probe_func.clone().launch(probe_config, (
                probe_keys,
                probe_payloads,
                num_probe,
                &hash_table,
                build_keys,
                build_payloads,
                &next_ptrs,
                hash_table_size,
                &output_left,
                &output_right,
                &output_count,
            )).map_err(|e| XlogError::Kernel(format!("Probe kernel failed: {}", e)))?;
        }

        // Synchronize and get result count
        self.device.synchronize()?;

        let mut count_host = vec![0u32; 1];
        self.device.inner().dtoh_sync_copy_into(&output_count, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read output count: {}", e)))?;
        let result_count = count_host[0] as u64;

        // Build result schema: combine left and right schemas
        let result_schema = self.combine_schemas(left.schema(), right.schema());

        // For MVP, return left/right key columns as result
        // Copy via host to convert between types (MVP simplification)
        let left_bytes = (result_count as usize) * std::mem::size_of::<u32>();
        let right_bytes = left_bytes;

        if result_count == 0 {
            return self.create_empty_buffer(result_schema);
        }

        // Copy via host: read u32, write as u8
        let mut left_host = vec![0u32; result_count as usize];
        let mut right_host = vec![0u32; result_count as usize];

        self.device.inner().dtoh_sync_copy_into(&output_left, &mut left_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read left output: {}", e)))?;
        self.device.inner().dtoh_sync_copy_into(&output_right, &mut right_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read right output: {}", e)))?;

        // Convert to bytes and upload as u8 slices
        let left_bytes_vec: Vec<u8> = left_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let right_bytes_vec: Vec<u8> = right_host.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut output_left_u8 = self.memory.alloc::<u8>(left_bytes)?;
        let mut output_right_u8 = self.memory.alloc::<u8>(right_bytes)?;

        self.device.inner().htod_sync_copy_into(&left_bytes_vec, &mut output_left_u8)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload left output: {}", e)))?;
        self.device.inner().htod_sync_copy_into(&right_bytes_vec, &mut output_right_u8)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload right output: {}", e)))?;

        Ok(CudaBuffer::from_columns(
            vec![output_left_u8, output_right_u8],
            result_count,
            result_schema,
        ))
    }

    /// Remove duplicate rows based on key columns
    ///
    /// Assumes input is already sorted by key columns (sorting is complex,
    /// deferred to future implementation).
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
            return Err(XlogError::Kernel("Dedup requires at least one key column".to_string()));
        }

        let num_rows = input.num_rows() as u32;
        let num_key_cols = key_cols.len() as u32;
        let row_stride = input.arity() as u32;

        // Get kernels
        let mark_func = self.device.inner()
            .get_func(DEDUP_MODULE, dedup_kernels::MARK_DUPLICATES)
            .ok_or_else(|| XlogError::Kernel("mark_duplicates kernel not found".to_string()))?;
        let compact_func = self.device.inner()
            .get_func(DEDUP_MODULE, dedup_kernels::COMPACT_ROWS)
            .ok_or_else(|| XlogError::Kernel("compact_rows kernel not found".to_string()))?;

        // Allocate unique mask (1 = unique, 0 = duplicate)
        let unique_mask = self.memory.alloc::<u8>(num_rows as usize)?;

        // For MVP, combine all columns into a single contiguous buffer for the kernel
        // This is a simplification; proper implementation would handle multi-column layout
        let total_elements = (num_rows as usize) * (row_stride as usize);
        let mut sorted_keys = self.memory.alloc::<u32>(total_elements)?;

        // Copy input columns into contiguous buffer (row-major for kernel)
        // For MVP, we assume first column contains the sorted keys
        // Copy via host to avoid type mismatch (MVP simplification)
        if let Some(first_col) = input.column(0) {
            let col_bytes = (num_rows as usize) * std::mem::size_of::<u32>();
            let mut host_bytes = vec![0u8; col_bytes];
            self.device.inner().dtoh_sync_copy_into(first_col, &mut host_bytes)
                .map_err(|e| XlogError::Kernel(format!("Failed to read input: {}", e)))?;

            // Convert bytes to u32 and upload
            let host_u32: Vec<u32> = host_bytes.chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            self.device.inner().htod_sync_copy_into(&host_u32, &mut sorted_keys)
                .map_err(|e| XlogError::Kernel(format!("Failed to copy input: {}", e)))?;
        }

        // Launch mark_duplicates kernel
        let block_size = 256u32;
        let grid_size = (num_rows + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Kernel parameters match expected signature
        unsafe {
            mark_func.clone().launch(config, (
                &sorted_keys,
                num_rows,
                num_key_cols,
                row_stride,
                &unique_mask,
            )).map_err(|e| XlogError::Kernel(format!("mark_duplicates failed: {}", e)))?;
        }

        // Compute prefix sum on CPU (GPU prefix sum is complex, defer to later)
        self.device.synchronize()?;

        let mut mask_host = vec![0u8; num_rows as usize];
        self.device.inner().dtoh_sync_copy_into(&unique_mask, &mut mask_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read mask: {}", e)))?;

        // Compute exclusive prefix sum
        let mut prefix_sum_host = vec![0u32; num_rows as usize];
        let mut running_sum = 0u32;
        for i in 0..num_rows as usize {
            prefix_sum_host[i] = running_sum;
            running_sum += mask_host[i] as u32;
        }
        let unique_count = running_sum as u64;

        if unique_count == 0 {
            return self.create_empty_buffer(input.schema().clone());
        }

        // Upload prefix sum
        let mut prefix_sum = self.memory.alloc::<u32>(num_rows as usize)?;
        self.device.inner().htod_sync_copy_into(&prefix_sum_host, &mut prefix_sum)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload prefix sum: {}", e)))?;

        // Allocate output
        let output_elements = (unique_count as usize) * (row_stride as usize);
        let output = self.memory.alloc::<u32>(output_elements)?;

        // Launch compact_rows kernel
        // SAFETY: Kernel parameters match expected signature
        unsafe {
            compact_func.clone().launch(config, (
                &sorted_keys,
                &unique_mask,
                &prefix_sum,
                num_rows,
                row_stride,
                &output,
            )).map_err(|e| XlogError::Kernel(format!("compact_rows failed: {}", e)))?;
        }

        self.device.synchronize()?;

        // Build output columns
        let bytes_per_col = (unique_count as usize) * std::mem::size_of::<u32>();
        let mut result_columns = Vec::new();

        // Copy output to result columns via host (MVP simplification)
        // Read output as u32, convert to bytes, upload as u8
        let mut output_host = vec![0u32; output_elements];
        self.device.inner().dtoh_sync_copy_into(&output, &mut output_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read output: {}", e)))?;

        for col_idx in 0..input.arity() {
            let mut col = self.memory.alloc::<u8>(bytes_per_col)?;
            // For MVP, copy first column's worth of data to each column
            // (simplified: real implementation would handle multi-column properly)
            let col_start = col_idx * unique_count as usize;
            let col_end = col_start + unique_count as usize;
            let col_data: Vec<u8> = output_host[col_start.min(output_host.len())..col_end.min(output_host.len())]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            if !col_data.is_empty() {
                self.device.inner().htod_sync_copy_into(&col_data, &mut col)
                    .map_err(|e| XlogError::Kernel(format!("Failed to upload output column: {}", e)))?;
            }
            result_columns.push(col);
        }

        Ok(CudaBuffer::from_columns(
            result_columns,
            unique_count,
            input.schema().clone(),
        ))
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

        // Verify schemas match
        if a.schema() != b.schema() {
            return Err(XlogError::Kernel("Union requires matching schemas".to_string()));
        }

        let total_rows = a.num_rows() + b.num_rows();
        let schema = a.schema().clone();

        // Concatenate each column via host (MVP simplification)
        let mut result_columns = Vec::with_capacity(schema.arity());
        for col_idx in 0..schema.arity() {
            let col_type_size = schema.column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let a_bytes = (a.num_rows() as usize) * col_type_size;
            let b_bytes = (b.num_rows() as usize) * col_type_size;
            let total_bytes = a_bytes + b_bytes;

            // Read both columns to host
            let mut result_host = Vec::with_capacity(total_bytes);

            // Copy a's column data
            if let Some(a_col) = a.column(col_idx) {
                if a_bytes > 0 {
                    let mut a_host = vec![0u8; a_bytes];
                    self.device.inner().dtoh_sync_copy_into(a_col, &mut a_host)
                        .map_err(|e| XlogError::Kernel(format!("Failed to read column from a: {}", e)))?;
                    result_host.extend_from_slice(&a_host);
                }
            }

            // Copy b's column data after a's
            if let Some(b_col) = b.column(col_idx) {
                if b_bytes > 0 {
                    let mut b_host = vec![0u8; b_bytes];
                    self.device.inner().dtoh_sync_copy_into(b_col, &mut b_host)
                        .map_err(|e| XlogError::Kernel(format!("Failed to read column from b: {}", e)))?;
                    result_host.extend_from_slice(&b_host);
                }
            }

            // Upload concatenated result
            let mut result_col = self.memory.alloc::<u8>(total_bytes)?;
            self.device.inner().htod_sync_copy_into(&result_host, &mut result_col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload concatenated column: {}", e)))?;

            result_columns.push(result_col);
        }

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

        // Verify schemas match
        if a.schema() != b.schema() {
            return Err(XlogError::Kernel("Diff requires matching schemas".to_string()));
        }

        // Use first column as key for hash-based diff
        if a.arity() == 0 {
            return Err(XlogError::Kernel("Diff requires at least one column".to_string()));
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
        self.device.inner().htod_sync_copy_into(
            &vec![init_val; hash_table_alloc_size],
            &mut hash_table,
        ).map_err(|e| XlogError::Kernel(format!("Failed to init hash table: {}", e)))?;
        self.device.inner().htod_sync_copy_into(
            &vec![init_val; num_b as usize],
            &mut next_ptrs,
        ).map_err(|e| XlogError::Kernel(format!("Failed to init next pointers: {}", e)))?;

        // Build phase with b's keys using transmute for direct GPU access
        let build_func = self.device.inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_BUILD)
            .ok_or_else(|| XlogError::Kernel("hash_join_build kernel not found".to_string()))?;

        let b_key_col = b.column(0)
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
            build_func.clone().launch(build_config, (
                &b_keys_view,
                &b_keys_view, // payload = key for diff
                num_b,
                &hash_table,
                &next_ptrs,
                hash_table_size,
            )).map_err(|e| XlogError::Kernel(format!("Build kernel failed: {}", e)))?;
        }

        // Synchronize and build lookup set for filtering
        self.device.synchronize()?;

        // Get a's keys using transmute
        let a_key_col = a.column(0)
            .ok_or_else(|| XlogError::Kernel("A key column not found".to_string()))?;

        // Read keys to host for filtering (set difference requires iterating)
        let mut a_keys_host = vec![0u8; (num_a as usize) * 4];
        self.device.inner().dtoh_sync_copy_into(a_key_col, &mut a_keys_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read a keys: {}", e)))?;

        let mut b_keys_host = vec![0u8; (num_b as usize) * 4];
        self.device.inner().dtoh_sync_copy_into(b_key_col, &mut b_keys_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read b keys: {}", e)))?;

        // Build lookup set from b
        let b_keys_set: std::collections::HashSet<u32> = b_keys_host.chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Find indices of a rows not in b
        let diff_indices: Vec<usize> = a_keys_host.chunks_exact(4)
            .enumerate()
            .map(|(i, chunk)| (i, u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
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
            let col_type_size = schema.column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let result_bytes = (diff_count as usize) * col_type_size;

            if let Some(a_col) = a.column(col_idx) {
                // Read column data
                let a_col_bytes = (num_a as usize) * col_type_size;
                let mut a_col_host = vec![0u8; a_col_bytes];
                self.device.inner().dtoh_sync_copy_into(a_col, &mut a_col_host)
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
                self.device.inner().htod_sync_copy_into(&result_host, &mut result_col)
                    .map_err(|e| XlogError::Kernel(format!("Failed to upload result: {}", e)))?;

                result_columns.push(result_col);
            }
        }

        Ok(CudaBuffer::from_columns(result_columns, diff_count, schema))
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
            return Err(XlogError::Kernel("GroupBy requires at least one key column".to_string()));
        }

        // Verify value column exists
        if value_col >= input.arity() {
            return Err(XlogError::Kernel(format!("Value column {} out of bounds", value_col)));
        }

        let num_rows = input.num_rows() as u32;
        let num_key_cols = key_cols.len() as u32;
        let row_stride = input.arity() as u32;

        // Get boundary detection kernel
        let boundary_func = self.device.inner()
            .get_func(GROUPBY_MODULE, groupby_kernels::DETECT_GROUP_BOUNDARIES)
            .ok_or_else(|| XlogError::Kernel("detect_group_boundaries kernel not found".to_string()))?;

        // Allocate is_boundary mask
        let is_boundary = self.memory.alloc::<u8>(num_rows as usize)?;

        // Prepare sorted keys buffer (MVP: use first key column)
        let sorted_keys = input.column(key_cols[0])
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
            boundary_func.clone().launch(config, (
                &sorted_keys_view,
                num_rows,
                num_key_cols,
                row_stride,
                &is_boundary,
            )).map_err(|e| XlogError::Kernel(format!("detect_group_boundaries failed: {}", e)))?;
        }

        // Compute group IDs from boundaries (CPU for MVP)
        self.device.synchronize()?;

        let mut boundary_host = vec![0u8; num_rows as usize];
        self.device.inner().dtoh_sync_copy_into(&is_boundary, &mut boundary_host)
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
        self.device.inner().htod_sync_copy_into(&group_ids_host, &mut group_ids)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload group IDs: {}", e)))?;

        // Get value column
        let values = input.column(value_col)
            .ok_or_else(|| XlogError::Kernel("Value column not found".to_string()))?;

        // Allocate and initialize output based on aggregation type
        // Return output as bytes via host for consistent handling
        let (output_bytes, _result_type): (Vec<u8>, ScalarType) = match agg {
            AggOp::Count => {
                let mut output = self.memory.alloc::<u32>(num_groups as usize)?;
                // Initialize to 0
                self.device.inner().htod_sync_copy_into(&vec![0u32; num_groups as usize], &mut output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to init count output: {}", e)))?;

                let count_func = self.device.inner()
                    .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_COUNT)
                    .ok_or_else(|| XlogError::Kernel("groupby_count kernel not found".to_string()))?;

                // SAFETY: Kernel parameters match expected signature
                unsafe {
                    count_func.clone().launch(config, (
                        &is_boundary,
                        &group_ids,
                        num_rows,
                        &output,
                    )).map_err(|e| XlogError::Kernel(format!("groupby_count failed: {}", e)))?;
                }

                self.device.synchronize()?;

                // Read back as bytes
                let mut host_output = vec![0u32; num_groups as usize];
                self.device.inner().dtoh_sync_copy_into(&output, &mut host_output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to read count output: {}", e)))?;
                let bytes: Vec<u8> = host_output.iter().flat_map(|v| v.to_le_bytes()).collect();
                (bytes, ScalarType::U32)
            }
            AggOp::Sum => {
                let mut output = self.memory.alloc::<u64>(num_groups as usize)?;
                // Initialize to 0
                self.device.inner().htod_sync_copy_into(&vec![0u64; num_groups as usize], &mut output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to init sum output: {}", e)))?;

                let sum_func = self.device.inner()
                    .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_SUM)
                    .ok_or_else(|| XlogError::Kernel("groupby_sum kernel not found".to_string()))?;

                // SAFETY: Kernel parameters match expected signature
                unsafe {
                    sum_func.clone().launch(config, (
                        values,
                        &group_ids,
                        num_rows,
                        &output,
                    )).map_err(|e| XlogError::Kernel(format!("groupby_sum failed: {}", e)))?;
                }

                self.device.synchronize()?;

                // Read back as bytes
                let mut host_output = vec![0u64; num_groups as usize];
                self.device.inner().dtoh_sync_copy_into(&output, &mut host_output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to read sum output: {}", e)))?;
                let bytes: Vec<u8> = host_output.iter().flat_map(|v| v.to_le_bytes()).collect();
                (bytes, ScalarType::U64)
            }
            AggOp::Min => {
                let mut output = self.memory.alloc::<u32>(num_groups as usize)?;
                // Initialize to MAX
                self.device.inner().htod_sync_copy_into(&vec![u32::MAX; num_groups as usize], &mut output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to init min output: {}", e)))?;

                let min_func = self.device.inner()
                    .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MIN)
                    .ok_or_else(|| XlogError::Kernel("groupby_min kernel not found".to_string()))?;

                // SAFETY: Kernel parameters match expected signature
                unsafe {
                    min_func.clone().launch(config, (
                        values,
                        &group_ids,
                        num_rows,
                        &output,
                    )).map_err(|e| XlogError::Kernel(format!("groupby_min failed: {}", e)))?;
                }

                self.device.synchronize()?;

                // Read back as bytes
                let mut host_output = vec![0u32; num_groups as usize];
                self.device.inner().dtoh_sync_copy_into(&output, &mut host_output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to read min output: {}", e)))?;
                let bytes: Vec<u8> = host_output.iter().flat_map(|v| v.to_le_bytes()).collect();
                (bytes, ScalarType::U32)
            }
            AggOp::Max => {
                let mut output = self.memory.alloc::<u32>(num_groups as usize)?;
                // Initialize to 0
                self.device.inner().htod_sync_copy_into(&vec![0u32; num_groups as usize], &mut output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to init max output: {}", e)))?;

                let max_func = self.device.inner()
                    .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MAX)
                    .ok_or_else(|| XlogError::Kernel("groupby_max kernel not found".to_string()))?;

                // SAFETY: Kernel parameters match expected signature
                unsafe {
                    max_func.clone().launch(config, (
                        values,
                        &group_ids,
                        num_rows,
                        &output,
                    )).map_err(|e| XlogError::Kernel(format!("groupby_max failed: {}", e)))?;
                }

                self.device.synchronize()?;

                // Read back as bytes
                let mut host_output = vec![0u32; num_groups as usize];
                self.device.inner().dtoh_sync_copy_into(&output, &mut host_output)
                    .map_err(|e| XlogError::Kernel(format!("Failed to read max output: {}", e)))?;
                let bytes: Vec<u8> = host_output.iter().flat_map(|v| v.to_le_bytes()).collect();
                (bytes, ScalarType::U32)
            }
            AggOp::LogSumExp => {
                // LogSumExp not implemented in kernels yet
                return Err(XlogError::Kernel("LogSumExp not yet implemented".to_string()));
            }
        };

        self.device.synchronize()?;

        // Extract first key value for each group (CPU for MVP)
        let mut key_col_host = vec![0u8; (num_rows as usize) * 4];
        self.device.inner().dtoh_sync_copy_into(sorted_keys, &mut key_col_host)
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
        self.device.inner().htod_sync_copy_into(&result_keys, &mut result_key_col)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload result keys: {}", e)))?;

        // Upload output bytes as u8 column
        let mut output_col_u8 = self.memory.alloc::<u8>(output_bytes.len())?;
        self.device.inner().htod_sync_copy_into(&output_bytes, &mut output_col_u8)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload output: {}", e)))?;

        // Build result schema: key columns + aggregation result
        let result_schema = self.groupby_result_schema(input.schema(), key_cols, agg);

        Ok(CudaBuffer::from_columns(
            vec![result_key_col, output_col_u8],
            num_groups as u64,
            result_schema,
        ))
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
    /// # Limitations
    /// The current implementation uses a single-block scan kernel which only
    /// works correctly for masks with up to 256 elements. For larger masks,
    /// a multi-pass algorithm would be needed.
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails
    pub fn prefix_sum_mask(&self, mask: &[u8]) -> Result<(Vec<u32>, u32)> {
        if mask.is_empty() {
            return Ok((vec![], 0));
        }

        // Check for single-block limitation
        if mask.len() > 256 {
            return Err(XlogError::Kernel(format!(
                "prefix_sum_mask currently limited to 256 elements, got {}",
                mask.len()
            )));
        }

        let n = mask.len();
        let device = self.device.inner();

        // Upload mask to GPU
        let d_mask = device
            .htod_sync_copy(mask)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload mask: {}", e)))?;

        // Allocate output for prefix sum
        let d_prefix_sum = unsafe { device.alloc::<u32>(n) }
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc prefix_sum: {}", e)))?;

        // Allocate count (initialize to 0)
        let d_count = device
            .htod_sync_copy(&[0u32])
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc count: {}", e)))?;

        // Launch exclusive_scan_mask kernel
        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;

        let scan_fn = device
            .get_func(SCAN_MODULE, scan_kernels::EXCLUSIVE_SCAN_MASK)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get exclusive_scan_mask kernel".to_string())
            })?;

        // SAFETY: Kernel parameters match expected signature:
        // exclusive_scan_mask(const uint8_t* mask, uint32_t* prefix_sum, uint32_t n)
        unsafe {
            scan_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&d_mask, &d_prefix_sum, n as u32),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("Failed to launch exclusive_scan_mask: {}", e)))?;

        // Launch count_mask kernel
        let count_fn = device
            .get_func(SCAN_MODULE, scan_kernels::COUNT_MASK)
            .ok_or_else(|| XlogError::Kernel("Failed to get count_mask kernel".to_string()))?;

        // SAFETY: Kernel parameters match expected signature:
        // count_mask(const uint8_t* mask, uint32_t n, uint32_t* count)
        unsafe {
            count_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&d_mask, n as u32, &d_count),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("Failed to launch count_mask: {}", e)))?;

        // Synchronize and download results
        self.device.synchronize()?;

        let prefix_sum = device
            .dtoh_sync_copy(&d_prefix_sum)
            .map_err(|e| XlogError::Kernel(format!("Failed to download prefix_sum: {}", e)))?;

        let count_vec = device
            .dtoh_sync_copy(&d_count)
            .map_err(|e| XlogError::Kernel(format!("Failed to download count: {}", e)))?;

        Ok((prefix_sum, count_vec[0]))
    }

    // ============== Sort Methods ==============

    // Radix sort constants
    const RADIX_BITS: u32 = 4;
    const RADIX_SIZE: u32 = 16; // 1 << RADIX_BITS
    const SORT_BLOCK_SIZE: u32 = 256;

    /// Sort buffer by key columns using GPU radix sort
    /// Currently supports single U32 key column
    ///
    /// Uses 4-bit radix sort with 8 passes (32 bits / 4 bits = 8 passes).
    /// Each pass: histogram -> prefix sum -> scatter
    ///
    /// # Arguments
    /// * `input` - The input buffer to sort
    /// * `key_cols` - Column indices to use for sorting (currently only single column supported)
    ///
    /// # Returns
    /// A new buffer with rows sorted by the key columns
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if:
    /// - Multi-column sort is requested (not yet implemented)
    /// - Kernel execution fails
    pub fn sort(&self, input: &CudaBuffer, key_cols: &[usize]) -> Result<CudaBuffer> {
        if input.num_rows == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }

        if key_cols.len() != 1 {
            return Err(XlogError::Kernel(
                "Multi-column sort not yet implemented".into(),
            ));
        }

        let key_col = key_cols[0];
        let n = input.num_rows as u32;
        let device = self.device.inner();

        // Compute grid size
        let grid_size = (n + Self::SORT_BLOCK_SIZE - 1) / Self::SORT_BLOCK_SIZE;

        // Get kernel functions
        let histogram_fn = device
            .get_func(SORT_MODULE, sort_kernels::RADIX_HISTOGRAM)
            .ok_or_else(|| XlogError::Kernel("radix_histogram kernel not found".to_string()))?;
        let scatter_fn = device
            .get_func(SORT_MODULE, sort_kernels::RADIX_SCATTER)
            .ok_or_else(|| XlogError::Kernel("radix_scatter kernel not found".to_string()))?;
        let init_indices_fn = device
            .get_func(SORT_MODULE, sort_kernels::INIT_INDICES)
            .ok_or_else(|| XlogError::Kernel("init_indices kernel not found".to_string()))?;

        // Allocate double buffers for keys and indices
        let mut keys_a = self.memory.alloc::<u32>(n as usize)?;
        let mut keys_b = self.memory.alloc::<u32>(n as usize)?;
        let mut indices_a = self.memory.alloc::<u32>(n as usize)?;
        let mut indices_b = self.memory.alloc::<u32>(n as usize)?;

        // Copy key column to keys_a (need to transmute from u8 to u32)
        let key_col_data = input.column(key_col)
            .ok_or_else(|| XlogError::Kernel("Key column not found".to_string()))?;
        let key_bytes = (n as usize) * std::mem::size_of::<u32>();
        let mut key_host = vec![0u8; key_bytes];
        device
            .dtoh_sync_copy_into(key_col_data, &mut key_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read keys: {}", e)))?;
        let key_u32: Vec<u32> = key_host
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        device
            .htod_sync_copy_into(&key_u32, &mut keys_a)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload keys: {}", e)))?;

        // Initialize indices to identity permutation
        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (Self::SORT_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Kernel signature matches: init_indices(uint32_t* indices, uint32_t n)
        unsafe {
            init_indices_fn.clone().launch(launch_config, (&indices_a, n))
        }
        .map_err(|e| XlogError::Kernel(format!("init_indices failed: {}", e)))?;

        // Allocate histograms buffer: [grid_size * RADIX_SIZE]
        let hist_size = (grid_size * Self::RADIX_SIZE) as usize;
        let histograms = self.memory.alloc::<u32>(hist_size)?;

        // Allocate prefix sums buffer once (reused each pass)
        let mut prefix_sums = self.memory.alloc::<u32>(Self::RADIX_SIZE as usize)?;

        // Perform 8 radix sort passes (4 bits per pass, 32 bits total)
        let mut use_a = true;
        for pass in 0..8u32 {
            let shift = pass * Self::RADIX_BITS;

            let (keys_in, keys_out) = if use_a {
                (&keys_a, &mut keys_b)
            } else {
                (&keys_b, &mut keys_a)
            };
            let (indices_in, indices_out) = if use_a {
                (&indices_a, &mut indices_b)
            } else {
                (&indices_b, &mut indices_a)
            };

            // Step 1: Compute histograms
            // SAFETY: Kernel signature matches: radix_histogram(const uint32_t* keys, uint32_t num_rows, uint32_t* histograms, uint32_t shift)
            unsafe {
                histogram_fn.clone().launch(launch_config, (keys_in, n, &histograms, shift))
            }
            .map_err(|e| XlogError::Kernel(format!("radix_histogram failed: {}", e)))?;

            // Step 2: Compute global prefix sums from histograms (on CPU for simplicity)
            // Download histograms, compute prefix sums, upload
            self.device.synchronize()?;
            let mut hist_host = vec![0u32; hist_size];
            device
                .dtoh_sync_copy_into(&histograms, &mut hist_host)
                .map_err(|e| XlogError::Kernel(format!("Failed to read histograms: {}", e)))?;

            // Sum across blocks for each digit to get global counts
            let mut global_counts = vec![0u32; Self::RADIX_SIZE as usize];
            for block in 0..grid_size as usize {
                for digit in 0..Self::RADIX_SIZE as usize {
                    global_counts[digit] += hist_host[block * Self::RADIX_SIZE as usize + digit];
                }
            }

            // Compute exclusive prefix sum of global counts
            let mut prefix_host = vec![0u32; Self::RADIX_SIZE as usize];
            let mut running_sum = 0u32;
            for digit in 0..Self::RADIX_SIZE as usize {
                prefix_host[digit] = running_sum;
                running_sum += global_counts[digit];
            }

            // Upload prefix sums (reusing pre-allocated buffer)
            device
                .htod_sync_copy_into(&prefix_host, &mut prefix_sums)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload prefix sums: {}", e)))?;

            // Step 3: Scatter keys to sorted positions
            // SAFETY: Kernel signature matches: radix_scatter(keys_in, indices_in, keys_out, indices_out, prefix_sums, local_offsets, num_rows, shift)
            unsafe {
                scatter_fn.clone().launch(
                    launch_config,
                    (keys_in, indices_in, keys_out, indices_out, &prefix_sums, &histograms, n, shift),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("radix_scatter failed: {}", e)))?;

            use_a = !use_a;
        }

        // Synchronize to ensure sort is complete
        self.device.synchronize()?;

        // Get final permutation (indices are in indices_a or indices_b depending on use_a)
        let final_indices = if use_a { &indices_a } else { &indices_b };

        // Apply permutation to all columns using GPU kernel
        self.apply_permutation_gpu(input, final_indices)
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
            .ok_or_else(|| XlogError::Kernel("apply_permutation_bytes kernel not found".to_string()))?;

        let mut new_columns = Vec::with_capacity(input.columns.len());

        for col_idx in 0..input.columns.len() {
            let src_col = input.column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;

            let elem_size = input.schema.column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4) as u32;

            let output_bytes = (n as usize) * (elem_size as usize);
            let dst_col = self.memory.alloc::<u8>(output_bytes)?;

            // SAFETY: Kernel signature matches: apply_permutation_bytes(input, output, permutation, num_rows, elem_size)
            unsafe {
                apply_perm_fn.clone().launch(
                    launch_config,
                    (src_col, &dst_col, permutation, n, elem_size),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("apply_permutation_bytes failed: {}", e)))?;

            new_columns.push(dst_col);
        }

        self.device.synchronize()?;

        Ok(CudaBuffer {
            columns: new_columns,
            num_rows: input.num_rows,
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
    /// # Limitations
    /// Currently limited to buffers with <= 256 rows due to single-block prefix sum.
    /// TODO: Implement multi-block prefix sum for larger datasets.
    ///
    /// # Errors
    /// Returns error if column index is out of bounds or buffer has > 256 rows.
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

        let n = input.num_rows as usize;
        let device = self.device.inner();

        // Validate column index
        if col >= input.arity() {
            return Err(XlogError::Kernel(format!(
                "Column index {} out of bounds (arity {})",
                col,
                input.arity()
            )));
        }

        // Get the filter column as u32 view
        let col_data = input
            .column(col)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col)))?;
        let col_view = self.column_as_u32_view(col_data, n)?;

        // Allocate mask buffer
        let d_mask = self.memory.alloc::<u8>(n)?;

        // Launch filter_compare_u32 kernel
        let block_size = 256u32;
        let grid_size = ((n as u32) + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let filter_fn = device
            .get_func(FILTER_MODULE, filter_kernels::FILTER_COMPARE_U32)
            .ok_or_else(|| XlogError::Kernel("filter_compare_u32 kernel not found".to_string()))?;

        // SAFETY: Kernel parameters match expected signature:
        // filter_compare_u32(const uint32_t* column, uint32_t constant, uint32_t num_rows, uint8_t op, uint8_t* mask)
        unsafe {
            filter_fn
                .clone()
                .launch(config, (&col_view, value, n as u32, op as u8, &d_mask))
        }
        .map_err(|e| XlogError::Kernel(format!("filter_compare_u32 failed: {}", e)))?;

        // Synchronize and download mask to compute prefix sum
        self.device.synchronize()?;

        let mut mask_host = vec![0u8; n];
        device
            .dtoh_sync_copy_into(&d_mask, &mut mask_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to download mask: {}", e)))?;

        // Compute prefix sum and count
        let (prefix_sum, count) = self.prefix_sum_mask(&mask_host)?;

        if count == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }

        // Compact all columns using mask
        self.compact_buffer_by_mask(input, &mask_host, &prefix_sum, count as u64)
    }

    /// Filter buffer by pre-computed mask.
    ///
    /// # Arguments
    /// * `input` - The input buffer to filter
    /// * `mask` - Mask slice where non-zero means keep the row
    ///
    /// # Limitations
    /// Currently limited to buffers with <= 256 rows due to single-block prefix sum.
    /// TODO: Implement multi-block prefix sum for larger datasets.
    ///
    /// # Errors
    /// Returns error if mask length doesn't match buffer rows or buffer has > 256 rows.
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
        let n = input.num_rows as u32;
        let device = self.device.inner();

        // Upload mask and prefix sum to GPU
        let d_mask = device
            .htod_sync_copy(mask)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload mask: {}", e)))?;
        let d_prefix_sum = device
            .htod_sync_copy(prefix_sum)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload prefix_sum: {}", e)))?;

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
                    (src_col, &d_mask, &d_prefix_sum, n, elem_size, &dst_col),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("compact_bytes_by_mask failed: {}", e)))?;

            new_columns.push(dst_col);
        }

        self.device.synchronize()?;

        Ok(CudaBuffer {
            columns: new_columns,
            num_rows: output_count,
            schema: input.schema.clone(),
        })
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
    pub fn create_buffer_from_u32_slice(
        &self,
        data: &[u32],
        schema: Schema,
    ) -> Result<CudaBuffer> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut col = self.memory.alloc::<u8>(bytes.len())?;
        self.device
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload data: {}", e)))?;

        Ok(CudaBuffer::from_columns(
            vec![col],
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
            cuda_columns.push(col);
        }

        Ok(CudaBuffer::from_columns(
            cuda_columns,
            num_rows as u64,
            schema,
        ))
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
        let col = buffer.column(col_idx).ok_or_else(|| {
            XlogError::Kernel(format!("Column {} not found", col_idx))
        })?;

        if buffer.num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = (buffer.num_rows as usize) * std::mem::size_of::<u32>();
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

    // ============== Internal Helper Methods ==============

    /// Transmute a CudaSlice<u8> column to a CudaView<u32> for kernel access
    ///
    /// # Safety
    /// Caller must ensure:
    /// - The column contains valid u32 data (4-byte aligned, correct endianness)
    /// - The column length is a multiple of 4 bytes
    fn column_as_u32_view<'a>(
        &self,
        col: &'a cudarc::driver::CudaSlice<u8>,
        num_elements: usize,
    ) -> Result<CudaView<'a, u32>> {
        let required_bytes = num_elements * std::mem::size_of::<u32>();
        if col.num_bytes() < required_bytes {
            return Err(XlogError::Kernel(format!(
                "Column has {} bytes but {} required for {} u32 elements",
                col.num_bytes(),
                required_bytes,
                num_elements
            )));
        }
        // SAFETY: We've verified the column has enough bytes
        unsafe { col.transmute(num_elements) }
            .ok_or_else(|| XlogError::Kernel("Failed to transmute column to u32".to_string()))
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
            columns.push(self.memory.alloc::<u8>(0)?);
        }
        Ok(CudaBuffer::from_columns(columns, 0, schema))
    }

    /// Combine schemas from left and right buffers for join result
    fn combine_schemas(&self, left: &Schema, right: &Schema) -> Schema {
        let mut columns = left.columns.clone();
        columns.extend(right.columns.iter().cloned());
        Schema::new(columns)
    }

    /// Create result schema for groupby aggregation
    fn groupby_result_schema(&self, input: &Schema, key_cols: &[usize], agg: AggOp) -> Schema {
        let mut columns: Vec<(String, ScalarType)> = key_cols
            .iter()
            .filter_map(|&i| input.columns.get(i).cloned())
            .collect();

        let agg_type = match agg {
            AggOp::Count => ScalarType::U32,
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
        match join_type {
            JoinType::Inner => self.hash_join_inner_v2(left, right, left_keys, right_keys),
            JoinType::Semi => self.hash_join_semi_impl(left, right, left_keys, right_keys),
            JoinType::Anti => self.hash_join_anti_impl(left, right, left_keys, right_keys),
            JoinType::LeftOuter => {
                self.hash_join_left_outer_impl(left, right, left_keys, right_keys)
            }
        }
    }

    /// Compute composite hashes for key columns of a buffer
    ///
    /// Uses FNV-1a hash to combine all key columns into a single u64 hash per row.
    fn compute_composite_hashes(
        &self,
        buffer: &CudaBuffer,
        key_cols: &[usize],
    ) -> Result<cudarc::driver::CudaSlice<u64>> {
        let num_rows = buffer.num_rows() as u32;
        if num_rows == 0 {
            return self.memory.alloc::<u64>(0);
        }

        // For column-major storage, we need to pack data row-major for the kernel
        // However, the current compute_composite_hash kernel expects row-major layout
        // with byte offsets within each row.
        //
        // MVP approach: copy key columns to host, pack row-major, upload

        let num_key_cols = key_cols.len();
        let mut col_sizes: Vec<u32> = Vec::with_capacity(num_key_cols);
        let mut col_data: Vec<Vec<u8>> = Vec::with_capacity(num_key_cols);

        // Read key columns to host
        for &col_idx in key_cols {
            let col = buffer.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Key column {} not found", col_idx))
            })?;

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
        let d_data = self
            .device
            .inner()
            .htod_sync_copy(&packed)
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
        Ok(d_hashes)
    }

    /// Build hash table from u64 hashes
    ///
    /// Returns (hash_table, next_ptrs) for probing.
    fn build_hash_table_v2(
        &self,
        hashes: &cudarc::driver::CudaSlice<u64>,
        num_rows: u32,
    ) -> Result<(cudarc::driver::CudaSlice<u32>, cudarc::driver::CudaSlice<u32>)> {
        // Hash table size: 2x build size for good load factor
        let hash_table_size = ((num_rows as usize) * 2).max(1024) as u32;

        // Allocate hash table and next pointers
        let d_hash_table = self.memory.alloc::<u32>(hash_table_size as usize)?;
        let d_next_ptrs = self.memory.alloc::<u32>(num_rows as usize)?;

        // Initialize hash table to 0xFFFFFFFF using init_hash_table kernel
        let init_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::INIT_HASH_TABLE)
            .ok_or_else(|| XlogError::Kernel("init_hash_table kernel not found".to_string()))?;

        let block_size = 256u32;
        let init_grid = (hash_table_size + block_size - 1) / block_size;
        let init_config = LaunchConfig {
            grid_dim: (init_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: init_hash_table(hash_table, size)
        unsafe {
            init_func
                .clone()
                .launch(init_config, (&d_hash_table, hash_table_size))
                .map_err(|e| XlogError::Kernel(format!("init_hash_table failed: {}", e)))?;
        }

        // Initialize next_ptrs to 0xFFFFFFFF
        let next_init_grid = (num_rows + block_size - 1) / block_size;
        let next_init_config = LaunchConfig {
            grid_dim: (next_init_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            init_func
                .clone()
                .launch(next_init_config, (&d_next_ptrs, num_rows))
                .map_err(|e| XlogError::Kernel(format!("init next_ptrs failed: {}", e)))?;
        }

        // Build hash table
        let build_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_BUILD_V2)
            .ok_or_else(|| XlogError::Kernel("hash_join_build_v2 kernel not found".to_string()))?;

        let build_grid = (num_rows + block_size - 1) / block_size;
        let build_config = LaunchConfig {
            grid_dim: (build_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: hash_join_build_v2(hashes, num_rows, hash_table, next_ptrs, hash_table_size)
        unsafe {
            build_func
                .clone()
                .launch(
                    build_config,
                    (hashes, num_rows, &d_hash_table, &d_next_ptrs, hash_table_size),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_build_v2 failed: {}", e)))?;
        }

        self.device.synchronize()?;
        Ok((d_hash_table, d_next_ptrs))
    }

    /// Inner join implementation using v2 kernels
    fn hash_join_inner_v2(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
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

        let num_left = left.num_rows() as u32;
        let num_right = right.num_rows() as u32;

        // Compute composite hashes for both sides
        let left_hashes = self.compute_composite_hashes(left, left_keys)?;
        let right_hashes = self.compute_composite_hashes(right, right_keys)?;

        // Build hash table from right side
        let hash_table_size = ((num_right as usize) * 2).max(1024) as u32;
        let (d_hash_table, d_next_ptrs) = self.build_hash_table_v2(&right_hashes, num_right)?;

        // Allocate output buffers for row index pairs
        let max_output = ((num_left as u64) * (num_right as u64)).min(1_000_000) as u32;
        let d_output_left = self.memory.alloc::<u32>(max_output as usize)?;
        let d_output_right = self.memory.alloc::<u32>(max_output as usize)?;
        let d_output_count = self
            .device
            .inner()
            .htod_sync_copy(&[0u32])
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc output count: {}", e)))?;

        // Launch probe kernel
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

        // SAFETY: hash_join_probe_v2(probe_hashes, num_probe, hash_table, build_hashes,
        //                            next_ptrs, hash_table_size, output_left, output_right,
        //                            output_count, max_output)
        unsafe {
            probe_func
                .clone()
                .launch(
                    probe_config,
                    (
                        &left_hashes,
                        num_left,
                        &d_hash_table,
                        &right_hashes,
                        &d_next_ptrs,
                        hash_table_size,
                        &d_output_left,
                        &d_output_right,
                        &d_output_count,
                        max_output,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_probe_v2 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        // Get output count
        let mut count_host = vec![0u32];
        self.device
            .inner()
            .dtoh_sync_copy_into(&d_output_count, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read output count: {}", e)))?;
        let result_count = count_host[0] as u64;

        if result_count == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        // Download row index pairs - need to slice the GPU buffer to match result_count
        let mut left_indices = vec![0u32; result_count as usize];
        let mut right_indices = vec![0u32; result_count as usize];

        // Use slice to get only the valid portion of the output
        let left_slice = d_output_left.slice(0..result_count as usize);
        let right_slice = d_output_right.slice(0..result_count as usize);

        self.device
            .inner()
            .dtoh_sync_copy_into(&left_slice, &mut left_indices)
            .map_err(|e| XlogError::Kernel(format!("Failed to read left indices: {}", e)))?;
        self.device
            .inner()
            .dtoh_sync_copy_into(&right_slice, &mut right_indices)
            .map_err(|e| XlogError::Kernel(format!("Failed to read right indices: {}", e)))?;

        // Gather rows from both sides
        self.gather_join_result(left, right, &left_indices, &right_indices)
    }

    /// Gather join result rows from left and right buffers using index pairs
    fn gather_join_result(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_indices: &[u32],
        right_indices: &[u32],
    ) -> Result<CudaBuffer> {
        let result_count = left_indices.len() as u64;
        let combined_schema = self.combine_schemas(left.schema(), right.schema());

        let mut result_columns = Vec::with_capacity(combined_schema.arity());

        // Gather left columns
        for col_idx in 0..left.arity() {
            let col = left.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Left column {} not found", col_idx))
            })?;

            let elem_size = left
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            // Read column to host
            let src_bytes = (left.num_rows() as usize) * elem_size;
            let mut src_host = vec![0u8; src_bytes];
            self.device
                .inner()
                .dtoh_sync_copy_into(col, &mut src_host)
                .map_err(|e| XlogError::Kernel(format!("Failed to read left column: {}", e)))?;

            // Gather values
            let dst_bytes = (result_count as usize) * elem_size;
            let mut dst_host = vec![0u8; dst_bytes];
            for (i, &idx) in left_indices.iter().enumerate() {
                let src_start = (idx as usize) * elem_size;
                let dst_start = i * elem_size;
                dst_host[dst_start..dst_start + elem_size]
                    .copy_from_slice(&src_host[src_start..src_start + elem_size]);
            }

            // Upload gathered column
            let mut dst_col = self.memory.alloc::<u8>(dst_bytes)?;
            self.device
                .inner()
                .htod_sync_copy_into(&dst_host, &mut dst_col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload left column: {}", e)))?;

            result_columns.push(dst_col);
        }

        // Gather right columns
        for col_idx in 0..right.arity() {
            let col = right.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Right column {} not found", col_idx))
            })?;

            let elem_size = right
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            // Read column to host
            let src_bytes = (right.num_rows() as usize) * elem_size;
            let mut src_host = vec![0u8; src_bytes];
            self.device
                .inner()
                .dtoh_sync_copy_into(col, &mut src_host)
                .map_err(|e| XlogError::Kernel(format!("Failed to read right column: {}", e)))?;

            // Gather values
            let dst_bytes = (result_count as usize) * elem_size;
            let mut dst_host = vec![0u8; dst_bytes];
            for (i, &idx) in right_indices.iter().enumerate() {
                let src_start = (idx as usize) * elem_size;
                let dst_start = i * elem_size;
                dst_host[dst_start..dst_start + elem_size]
                    .copy_from_slice(&src_host[src_start..src_start + elem_size]);
            }

            // Upload gathered column
            let mut dst_col = self.memory.alloc::<u8>(dst_bytes)?;
            self.device
                .inner()
                .htod_sync_copy_into(&dst_host, &mut dst_col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload right column: {}", e)))?;

            result_columns.push(dst_col);
        }

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

        let num_left = left.num_rows() as u32;
        let num_right = right.num_rows() as u32;

        // Compute composite hashes
        let left_hashes = self.compute_composite_hashes(left, left_keys)?;
        let right_hashes = self.compute_composite_hashes(right, right_keys)?;

        // Build hash table from right side
        let hash_table_size = ((num_right as usize) * 2).max(1024) as u32;
        let (d_hash_table, d_next_ptrs) = self.build_hash_table_v2(&right_hashes, num_right)?;

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

        // SAFETY: hash_join_semi(probe_hashes, num_probe, hash_table, build_hashes,
        //                        next_ptrs, hash_table_size, has_match)
        unsafe {
            semi_func
                .clone()
                .launch(
                    config,
                    (
                        &left_hashes,
                        num_left,
                        &d_hash_table,
                        &right_hashes,
                        &d_next_ptrs,
                        hash_table_size,
                        &d_has_match,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_semi failed: {}", e)))?;
        }

        self.device.synchronize()?;

        // Download mask
        let mut mask_host = vec![0u8; num_left as usize];
        self.device
            .inner()
            .dtoh_sync_copy_into(&d_has_match, &mut mask_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read semi-join mask: {}", e)))?;

        // Filter left buffer by mask using gather-based approach (no row limit)
        self.gather_rows_by_mask(left, &mask_host)
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

        let num_left = left.num_rows() as u32;
        let num_right = right.num_rows() as u32;

        // Compute composite hashes
        let left_hashes = self.compute_composite_hashes(left, left_keys)?;
        let right_hashes = self.compute_composite_hashes(right, right_keys)?;

        // Build hash table from right side
        let hash_table_size = ((num_right as usize) * 2).max(1024) as u32;
        let (d_hash_table, d_next_ptrs) = self.build_hash_table_v2(&right_hashes, num_right)?;

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

        // SAFETY: hash_join_anti(probe_hashes, num_probe, hash_table, build_hashes,
        //                        next_ptrs, hash_table_size, no_match)
        unsafe {
            anti_func
                .clone()
                .launch(
                    config,
                    (
                        &left_hashes,
                        num_left,
                        &d_hash_table,
                        &right_hashes,
                        &d_next_ptrs,
                        hash_table_size,
                        &d_no_match,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_anti failed: {}", e)))?;
        }

        self.device.synchronize()?;

        // Download mask
        let mut mask_host = vec![0u8; num_left as usize];
        self.device
            .inner()
            .dtoh_sync_copy_into(&d_no_match, &mut mask_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read anti-join mask: {}", e)))?;

        // Filter left buffer by mask using gather-based approach (no row limit)
        self.gather_rows_by_mask(left, &mask_host)
    }

    /// Left outer join implementation: return all left rows with matched right columns or nulls
    fn hash_join_left_outer_impl(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
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

        let num_left = left.num_rows() as u32;
        let num_right = right.num_rows() as u32;

        // Compute composite hashes
        let left_hashes = self.compute_composite_hashes(left, left_keys)?;
        let right_hashes = self.compute_composite_hashes(right, right_keys)?;

        // Build hash table from right side
        let hash_table_size = ((num_right as usize) * 2).max(1024) as u32;
        let (d_hash_table, d_next_ptrs) = self.build_hash_table_v2(&right_hashes, num_right)?;

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

        unsafe {
            semi_func
                .clone()
                .launch(
                    config,
                    (
                        &left_hashes,
                        num_left,
                        &d_hash_table,
                        &right_hashes,
                        &d_next_ptrs,
                        hash_table_size,
                        &d_has_match,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_semi failed: {}", e)))?;
        }

        // Also do inner join to get matching pairs
        let max_output = ((num_left as u64) * (num_right as u64)).min(1_000_000) as u32;
        let d_output_left = self.memory.alloc::<u32>(max_output as usize)?;
        let d_output_right = self.memory.alloc::<u32>(max_output as usize)?;
        let d_output_count = self
            .device
            .inner()
            .htod_sync_copy(&[0u32])
            .map_err(|e| XlogError::Kernel(format!("Failed to alloc output count: {}", e)))?;

        let probe_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2)
            .ok_or_else(|| XlogError::Kernel("hash_join_probe_v2 kernel not found".to_string()))?;

        unsafe {
            probe_func
                .clone()
                .launch(
                    config,
                    (
                        &left_hashes,
                        num_left,
                        &d_hash_table,
                        &right_hashes,
                        &d_next_ptrs,
                        hash_table_size,
                        &d_output_left,
                        &d_output_right,
                        &d_output_count,
                        max_output,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_probe_v2 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        // Download match mask
        let mut has_match_host = vec![0u8; num_left as usize];
        self.device
            .inner()
            .dtoh_sync_copy_into(&d_has_match, &mut has_match_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read has_match: {}", e)))?;

        // Download inner join result count
        let mut count_host = vec![0u32];
        self.device
            .inner()
            .dtoh_sync_copy_into(&d_output_count, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read output count: {}", e)))?;
        let inner_count = count_host[0] as usize;

        // Download inner join indices
        let mut left_indices = vec![0u32; inner_count];
        let mut right_indices = vec![0u32; inner_count];

        if inner_count > 0 {
            let left_slice = d_output_left.slice(0..inner_count);
            let right_slice = d_output_right.slice(0..inner_count);

            self.device
                .inner()
                .dtoh_sync_copy_into(&left_slice, &mut left_indices)
                .map_err(|e| XlogError::Kernel(format!("Failed to read left indices: {}", e)))?;
            self.device
                .inner()
                .dtoh_sync_copy_into(&right_slice, &mut right_indices)
                .map_err(|e| XlogError::Kernel(format!("Failed to read right indices: {}", e)))?;
        }

        // Count unmatched left rows
        let unmatched_count = has_match_host.iter().filter(|&&m| m == 0).count();

        // Total result: inner join results + unmatched left rows
        let total_count = inner_count + unmatched_count;

        if total_count == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        // Build result by gathering from inner join results and adding unmatched rows with nulls
        self.gather_left_outer_result(
            left,
            right,
            &left_indices,
            &right_indices,
            &has_match_host,
            total_count as u64,
        )
    }

    /// Helper for left outer join: gather result rows including nulls for unmatched
    fn gather_left_outer_result(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_indices: &[u32],
        right_indices: &[u32],
        has_match: &[u8],
        total_count: u64,
    ) -> Result<CudaBuffer> {
        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let inner_count = left_indices.len();

        // Collect unmatched left row indices
        let unmatched_left: Vec<u32> = has_match
            .iter()
            .enumerate()
            .filter(|(_, &m)| m == 0)
            .map(|(i, _)| i as u32)
            .collect();

        let mut result_columns = Vec::with_capacity(combined_schema.arity());

        // Gather left columns: first inner join rows, then unmatched rows
        for col_idx in 0..left.arity() {
            let col = left.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Left column {} not found", col_idx))
            })?;

            let elem_size = left
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            // Read column to host
            let src_bytes = (left.num_rows() as usize) * elem_size;
            let mut src_host = vec![0u8; src_bytes];
            self.device
                .inner()
                .dtoh_sync_copy_into(col, &mut src_host)
                .map_err(|e| XlogError::Kernel(format!("Failed to read left column: {}", e)))?;

            // Gather values: inner join rows first, then unmatched
            let dst_bytes = (total_count as usize) * elem_size;
            let mut dst_host = vec![0u8; dst_bytes];

            // Inner join rows
            for (i, &idx) in left_indices.iter().enumerate() {
                let src_start = (idx as usize) * elem_size;
                let dst_start = i * elem_size;
                dst_host[dst_start..dst_start + elem_size]
                    .copy_from_slice(&src_host[src_start..src_start + elem_size]);
            }

            // Unmatched rows
            for (i, &idx) in unmatched_left.iter().enumerate() {
                let src_start = (idx as usize) * elem_size;
                let dst_start = (inner_count + i) * elem_size;
                dst_host[dst_start..dst_start + elem_size]
                    .copy_from_slice(&src_host[src_start..src_start + elem_size]);
            }

            // Upload gathered column
            let mut dst_col = self.memory.alloc::<u8>(dst_bytes)?;
            self.device
                .inner()
                .htod_sync_copy_into(&dst_host, &mut dst_col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload left column: {}", e)))?;

            result_columns.push(dst_col);
        }

        // Gather right columns: matched rows from inner join, zeros for unmatched
        for col_idx in 0..right.arity() {
            let col = right.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Right column {} not found", col_idx))
            })?;

            let elem_size = right
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            // Read column to host
            let src_bytes = (right.num_rows() as usize) * elem_size;
            let mut src_host = vec![0u8; src_bytes];
            self.device
                .inner()
                .dtoh_sync_copy_into(col, &mut src_host)
                .map_err(|e| XlogError::Kernel(format!("Failed to read right column: {}", e)))?;

            // Gather values: matched rows, then zeros for unmatched
            let dst_bytes = (total_count as usize) * elem_size;
            let mut dst_host = vec![0u8; dst_bytes]; // Zeros for unmatched by default

            // Matched rows from inner join
            for (i, &idx) in right_indices.iter().enumerate() {
                let src_start = (idx as usize) * elem_size;
                let dst_start = i * elem_size;
                dst_host[dst_start..dst_start + elem_size]
                    .copy_from_slice(&src_host[src_start..src_start + elem_size]);
            }

            // Unmatched rows already have zeros from vec initialization

            // Upload gathered column
            let mut dst_col = self.memory.alloc::<u8>(dst_bytes)?;
            self.device
                .inner()
                .htod_sync_copy_into(&dst_host, &mut dst_col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload right column: {}", e)))?;

            result_columns.push(dst_col);
        }

        Ok(CudaBuffer::from_columns(
            result_columns,
            total_count,
            combined_schema,
        ))
    }

    /// Helper for left outer join with empty right: all left rows with null right columns
    fn left_outer_with_nulls(&self, left: &CudaBuffer, right: &CudaBuffer) -> Result<CudaBuffer> {
        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let num_rows = left.num_rows();

        let mut result_columns = Vec::with_capacity(combined_schema.arity());

        // Copy all left columns
        for col_idx in 0..left.arity() {
            let col = left.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Left column {} not found", col_idx))
            })?;

            let elem_size = left
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            let bytes = (num_rows as usize) * elem_size;
            let mut host_data = vec![0u8; bytes];
            self.device
                .inner()
                .dtoh_sync_copy_into(col, &mut host_data)
                .map_err(|e| XlogError::Kernel(format!("Failed to read left column: {}", e)))?;

            let mut dst_col = self.memory.alloc::<u8>(bytes)?;
            self.device
                .inner()
                .htod_sync_copy_into(&host_data, &mut dst_col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload left column: {}", e)))?;

            result_columns.push(dst_col);
        }

        // Create null (zero) columns for right side
        for col_idx in 0..right.arity() {
            let elem_size = right
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            let bytes = (num_rows as usize) * elem_size;
            let zeros = vec![0u8; bytes];

            let mut dst_col = self.memory.alloc::<u8>(bytes)?;
            self.device
                .inner()
                .htod_sync_copy_into(&zeros, &mut dst_col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload null column: {}", e)))?;

            result_columns.push(dst_col);
        }

        Ok(CudaBuffer::from_columns(
            result_columns,
            num_rows,
            combined_schema,
        ))
    }

    /// Gather rows from buffer based on mask (host-based filtering with no row limit)
    ///
    /// This is an alternative to filter_by_mask that doesn't have the 256-row limitation
    /// because it does gathering on the host side.
    fn gather_rows_by_mask(&self, buffer: &CudaBuffer, mask: &[u8]) -> Result<CudaBuffer> {
        // Count matching rows and collect indices
        let indices: Vec<u32> = mask
            .iter()
            .enumerate()
            .filter(|(_, &m)| m != 0)
            .map(|(i, _)| i as u32)
            .collect();

        let result_count = indices.len() as u64;

        if result_count == 0 {
            return self.create_empty_buffer(buffer.schema().clone());
        }

        // Gather columns using indices
        let mut result_columns = Vec::with_capacity(buffer.arity());

        for col_idx in 0..buffer.arity() {
            let col = buffer.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Column {} not found", col_idx))
            })?;

            let elem_size = buffer
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);

            // Read column to host
            let src_bytes = (buffer.num_rows() as usize) * elem_size;
            let mut src_host = vec![0u8; src_bytes];
            self.device
                .inner()
                .dtoh_sync_copy_into(col, &mut src_host)
                .map_err(|e| XlogError::Kernel(format!("Failed to read column: {}", e)))?;

            // Gather values using indices
            let dst_bytes = (result_count as usize) * elem_size;
            let mut dst_host = vec![0u8; dst_bytes];
            for (i, &idx) in indices.iter().enumerate() {
                let src_start = (idx as usize) * elem_size;
                let dst_start = i * elem_size;
                dst_host[dst_start..dst_start + elem_size]
                    .copy_from_slice(&src_host[src_start..src_start + elem_size]);
            }

            // Upload gathered column
            let mut dst_col = self.memory.alloc::<u8>(dst_bytes)?;
            self.device
                .inner()
                .htod_sync_copy_into(&dst_host, &mut dst_col)
                .map_err(|e| XlogError::Kernel(format!("Failed to upload column: {}", e)))?;

            result_columns.push(dst_col);
        }

        Ok(CudaBuffer::from_columns(
            result_columns,
            result_count,
            buffer.schema().clone(),
        ))
    }

    /// Clone a buffer (deep copy) via host (MVP simplification)
    fn clone_buffer(&self, buffer: &CudaBuffer) -> Result<CudaBuffer> {
        let mut result_columns = Vec::with_capacity(buffer.arity());

        for col_idx in 0..buffer.arity() {
            let col_type_size = buffer.schema().column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let bytes = (buffer.num_rows() as usize) * col_type_size;

            if let Some(src_col) = buffer.column(col_idx) {
                // Copy via host to avoid type mismatch
                let mut host_data = vec![0u8; bytes];
                self.device.inner().dtoh_sync_copy_into(src_col, &mut host_data)
                    .map_err(|e| XlogError::Kernel(format!("Failed to read column for clone: {}", e)))?;

                let mut dst_col = self.memory.alloc::<u8>(bytes)?;
                self.device.inner().htod_sync_copy_into(&host_data, &mut dst_col)
                    .map_err(|e| XlogError::Kernel(format!("Failed to clone column: {}", e)))?;
                result_columns.push(dst_col);
            }
        }

        Ok(CudaBuffer::from_columns(
            result_columns,
            buffer.num_rows(),
            buffer.schema().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::MemoryBudget;

    fn has_cuda_device() -> bool {
        cudarc::driver::CudaDevice::count().unwrap_or(0) > 0
    }

    #[test]
    fn test_ptx_embedded() {
        // Verify PTX sources are embedded and non-empty
        assert!(!JOIN_PTX.is_empty(), "JOIN_PTX should not be empty");
        assert!(!DEDUP_PTX.is_empty(), "DEDUP_PTX should not be empty");
        assert!(!GROUPBY_PTX.is_empty(), "GROUPBY_PTX should not be empty");

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
    }

    #[test]
    fn test_ptx_target_architecture() {
        // Verify PTX is compiled for sm_90
        assert!(
            JOIN_PTX.contains(".target sm_90"),
            "JOIN_PTX should target sm_90"
        );
        assert!(
            DEDUP_PTX.contains(".target sm_90"),
            "DEDUP_PTX should target sm_90"
        );
        assert!(
            GROUPBY_PTX.contains(".target sm_90"),
            "GROUPBY_PTX should target sm_90"
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

        let _provider = CudaKernelProvider::new(device.clone(), memory).expect("Failed to create provider");

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
        let boundaries_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::DETECT_GROUP_BOUNDARIES);
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
        assert!(sum_fn.is_some(), "groupby_sum function should be accessible");

        let min_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MIN);
        assert!(min_fn.is_some(), "groupby_min function should be accessible");

        let max_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MAX);
        assert!(max_fn.is_some(), "groupby_max function should be accessible");
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

        CudaBuffer::from_columns(vec![col], data.len() as u64, schema)
    }

    // Helper function to create an empty buffer with correct column count
    fn create_empty_test_buffer(provider: &CudaKernelProvider, schema: Schema) -> CudaBuffer {
        let mut columns = Vec::with_capacity(schema.arity());
        for _ in 0..schema.arity() {
            columns.push(provider.memory().alloc::<u8>(0).expect("alloc"));
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
    fn test_union_schema_mismatch() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let a = create_test_buffer(&provider, &[1, 2], "col_a");
        let b = create_test_buffer(&provider, &[3, 4], "col_b");

        // Different column names should fail
        let result = provider.union(&a, &b);
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

        let a = create_test_buffer(&provider, &[1, 2], "col_a");
        let b = create_test_buffer(&provider, &[1, 2], "col_b");

        let result = provider.diff(&a, &b);
        assert!(result.is_err());
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
    fn test_groupby_logsumexp_not_implemented() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&provider, &[1, 1, 2], "key");

        let result = provider.groupby_agg(&buffer, &[0], AggOp::LogSumExp, 0);
        assert!(result.is_err());
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

        // Count result schema
        let count_schema = provider.groupby_result_schema(&input, &[0], AggOp::Count);
        assert_eq!(count_schema.arity(), 2);
        assert_eq!(count_schema.column_type(1), Some(ScalarType::U32));

        // Sum result schema
        let sum_schema = provider.groupby_result_schema(&input, &[0], AggOp::Sum);
        assert_eq!(sum_schema.arity(), 2);
        assert_eq!(sum_schema.column_type(1), Some(ScalarType::U64));

        // Min/Max result schema
        let min_schema = provider.groupby_result_schema(&input, &[0], AggOp::Min);
        assert_eq!(min_schema.arity(), 2);
        assert_eq!(min_schema.column_type(1), Some(ScalarType::U32));
    }
}
