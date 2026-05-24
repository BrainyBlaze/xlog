//! Relational operations: join, dedup, union, diff, sort, and related helpers.

use std::ffi::c_void;
use std::sync::atomic::Ordering;

use crate::{
    cuda_graph::{CapturedCudaGraph, CsmCudaGraphKey, CudaGraphNodeKind},
    AsKernelParam, DeviceSlice, LaunchAsync, LaunchConfig,
};
use xlog_core::{Result, ScalarType, Schema, XlogError};

use super::{
    dedup_kernels, filter_kernels, ilp_kernels, join_kernels, pack_kernels, scan_kernels,
    set_ops_kernels, sort_kernels, CsmCudaGraphEntry, CsmCudaGraphNodes, HashTableU64,
    JoinHashTableV2, JoinIndexV2, JoinType, PackedKeyData, RadixSortScratch, DEDUP_MODULE,
    DEFAULT_JOIN_MAX_OUTPUT, FILTER_MODULE, ILP_MODULE, JOIN_MODULE, NESTED_LOOP_TOTAL_THRESHOLD,
    PACK_MODULE, SCAN_MODULE, SET_OPS_MODULE, SORT_MODULE,
};
use crate::device_runtime::{Access, BlockId, StreamId};
use crate::launch::LaunchRecorder;
use crate::memory::{CudaColumn, TrackedCudaSlice};
use crate::CudaBuffer;

// Per-column scalar-type encoding used by the deterministic full-row
// dedup/diff kernels. Must match the `XLOG_TY_*` defines in
// `kernels/dedup.cu`. Centralized here as named constants and one helper
// so all full-row callers in this module share one source of truth.
const XLOG_TY_U32: u8 = 0;
const XLOG_TY_U64: u8 = 1;
const XLOG_TY_I32: u8 = 2;
const XLOG_TY_I64: u8 = 3;
const XLOG_TY_F32: u8 = 4;
const XLOG_TY_F64: u8 = 5;
const XLOG_TY_BOOL: u8 = 6;
const XLOG_TY_SYMBOL: u8 = 7;
const W66_SMALL_FULL_ROW_SORT_MAX_ROWS: usize = 1024;

#[inline]
fn scalar_type_code_dedup(ty: ScalarType) -> u8 {
    match ty {
        ScalarType::U32 => XLOG_TY_U32,
        ScalarType::U64 => XLOG_TY_U64,
        ScalarType::I32 => XLOG_TY_I32,
        ScalarType::I64 => XLOG_TY_I64,
        ScalarType::F32 => XLOG_TY_F32,
        ScalarType::F64 => XLOG_TY_F64,
        ScalarType::Bool => XLOG_TY_BOOL,
        ScalarType::Symbol => XLOG_TY_SYMBOL,
    }
}

impl super::CudaKernelProvider {
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
        let mut result_sort_labels = left.schema().sort_labels().to_vec();
        for (idx, col) in right.schema().columns.iter().enumerate() {
            if !right_key_set.contains(&idx) {
                result_columns_schema.push(col.clone());
                result_sort_labels.push(
                    right
                        .schema()
                        .column_sort_label(idx)
                        .unwrap_or(&col.0)
                        .to_string(),
                );
            }
        }
        let result_schema = Schema::new(result_columns_schema)
            .with_sort_labels(result_sort_labels)
            .expect("natural join sort labels match result schema arity");

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
            row_cap,
            d_num_rows,
            schema: _,
            ..
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
            row_cap,
            d_num_rows,
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
    /// A buffer containing one row per duplicate-equivalence class
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails
    pub fn dedup(&self, input: &CudaBuffer, key_cols: &[usize]) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema().clone());
        }

        if key_cols.is_empty() {
            if input.arity() == 0 {
                // A 0-arity relation is either empty or {()}, and dedup collapses any
                // non-empty multiplicity to a single empty tuple.
                let rows = self.device_row_count(input)?;
                if rows == 0 {
                    return self.create_empty_buffer(input.schema().clone());
                }
                return self.buffer_from_columns(Vec::new(), 1, input.schema().clone());
            }
            return Err(XlogError::Kernel(
                "Dedup requires at least one key column".to_string(),
            ));
        }

        if Self::is_full_row_key(key_cols, input.arity()) && input.arity() > 1 {
            return self.dedup_full_row_deterministic(input);
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
    /// A buffer containing one row per duplicate-equivalence class
    pub fn dedup_sorted(&self, input: &CudaBuffer, key_cols: &[usize]) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema().clone());
        }

        if key_cols.is_empty() {
            if input.arity() == 0 {
                let rows = self.device_row_count(input)?;
                if rows == 0 {
                    return self.create_empty_buffer(input.schema().clone());
                }
                return self.buffer_from_columns(Vec::new(), 1, input.schema().clone());
            }
            return Err(XlogError::Kernel(
                "Dedup requires at least one key column".to_string(),
            ));
        }

        if Self::is_full_row_key(key_cols, input.arity()) && input.arity() > 1 {
            return self.dedup_full_row_deterministic(input);
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

        // Use the module-level `scalar_type_code_dedup` so the host
        // encoding stays in lockstep with the `XLOG_TY_*` defines in
        // `kernels/dedup.cu`.
        let scalar_type_code = scalar_type_code_dedup;

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

            let ptr = *col.device_ptr();
            col_ptrs_host.push(ptr);
            col_sizes_host.push(elem_size as u32);
            col_types_host.push(scalar_type_code(ty));
        }

        let num_key_cols = key_cols.len() as u32;
        let mut d_col_ptrs = self.memory.alloc::<u64>(key_cols.len())?;
        let mut d_col_sizes = self.memory.alloc::<u32>(key_cols.len())?;
        let mut d_col_types = self.memory.alloc::<u8>(key_cols.len())?;

        self.htod_launch_metadata_sync_copy_into(&col_ptrs_host, &mut d_col_ptrs)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload key column ptrs: {}", e)))?;
        self.htod_launch_metadata_sync_copy_into(&col_sizes_host, &mut d_col_sizes)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload key column sizes: {}", e)))?;
        self.htod_launch_metadata_sync_copy_into(&col_types_host, &mut d_col_types)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload key column types: {}", e)))?;

        let block_size = 256u32;
        let num_blocks = num_rows.div_ceil(block_size);
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

        // SAFETY: mark_unique_and_scan_columnar(col_ptrs, col_sizes, col_types, num_key_cols,
        //                                       num_rows_device, row_cap,
        //                                       unique_mask, prefix_sum, block_sums)
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            mark_and_scan_fn.clone().launch(
                config,
                (
                    &d_col_ptrs,
                    &d_col_sizes,
                    &d_col_types,
                    num_key_cols,
                    input.num_rows_device(),
                    num_rows,
                    &d_unique_mask,
                    &d_prefix_sum,
                    &d_block_sums,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("mark_unique_and_scan_columnar failed: {}", e)))?;
        self.device.synchronize()?;

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
            self.device.synchronize()?;
        }

        self.device.synchronize()?;

        let d_out_count = self.capture_compact_count(&d_prefix_sum, &d_unique_mask, num_rows)?;
        self.compact_buffer_by_device_mask_device_count(
            input,
            &d_unique_mask,
            &d_prefix_sum,
            d_out_count,
        )
    }
    /// Compute union of two buffers (GPU-native, deduped)
    ///
    /// # Arguments
    /// * `a` - First buffer
    /// * `b` - Second buffer
    ///
    /// # Returns
    /// A buffer containing the deduplicated union of both inputs
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if schemas don't match or operation fails
    pub fn union(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        self.union_gpu(a, b)
    }

    fn concat_buffers_gpu(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        if !self.schemas_type_compatible(a.schema(), b.schema()) {
            return Err(XlogError::Kernel(format!(
                "Concat requires compatible schemas: {:?} vs {:?}",
                a.schema(),
                b.schema()
            )));
        }

        let schema = a.schema().clone();
        let a_rows = self.device_row_count(a)? as u64;
        let b_rows = self.device_row_count(b)? as u64;

        if a_rows == 0 && b_rows == 0 {
            return self.create_empty_buffer(schema);
        }
        if a_rows == 0 {
            return self.clone_buffer(b);
        }
        if b_rows == 0 {
            return self.clone_buffer(a);
        }

        let total_rows = a_rows + b_rows;
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

        let a_rows = usize::try_from(a_rows)
            .map_err(|_| XlogError::Kernel(format!("Concat: a has too many rows: {}", a_rows)))?;
        let b_rows = usize::try_from(b_rows)
            .map_err(|_| XlogError::Kernel(format!("Concat: b has too many rows: {}", b_rows)))?;

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
                let grid_size = total_bytes_u32.div_ceil(block_size);
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

        self.buffer_from_columns(result_columns, total_rows, schema)
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
        let num_a = self.device_row_count(a)?;
        let num_b = self.device_row_count(b)?;
        if num_a > u32::MAX as usize || num_b > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Diff supports at most {} rows per side (a={}, b={})",
                u32::MAX,
                num_a,
                num_b
            )));
        }

        // Handle empty cases
        if num_a == 0 {
            return self.create_empty_buffer(a.schema().clone());
        }
        if num_b == 0 {
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

        let num_b = num_b as u32;
        let num_a = num_a as u32;

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
        let build_grid = num_b.div_ceil(block_size);
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
        self.dtoh_sync_copy_into_tracked(a_key_col, &mut a_keys_host)
            .map_err(|e| XlogError::Kernel(format!("Failed to read a keys: {}", e)))?;

        let mut b_keys_host = vec![0u8; (num_b as usize) * 4];
        self.dtoh_sync_copy_into_tracked(b_key_col, &mut b_keys_host)
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
                self.dtoh_sync_copy_into_tracked(a_col, &mut a_col_host)
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

        self.buffer_from_columns(result_columns, diff_count, schema)
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
        // Verify schemas have compatible types (ignore column names for Datalog union).
        if !self.schemas_type_compatible(a.schema(), b.schema()) {
            return Err(XlogError::Kernel(format!(
                "Union requires compatible schemas: {:?} vs {:?}",
                a.schema(),
                b.schema()
            )));
        }

        let schema = a.schema().clone();
        let a_rows = self.device_row_count(a)?;
        let b_rows = self.device_row_count(b)?;
        if schema.arity() == 0 {
            // 0-arity set union: empty ∪ empty = empty; otherwise {()}.
            if a_rows == 0 && b_rows == 0 {
                return self.create_empty_buffer(schema);
            }
            return self.buffer_from_columns(Vec::new(), 1, schema);
        }

        let key_cols: Vec<usize> = (0..schema.arity()).collect();
        if a_rows == 0 && b_rows == 0 {
            return self.create_empty_buffer(schema);
        }

        // Set semantics require dedup even when one side is empty.
        if a_rows == 0 {
            return self.dedup(b, &key_cols);
        }
        if b_rows == 0 {
            return self.dedup(a, &key_cols);
        }

        let concat = self.concat_buffers_gpu(a, b)?;
        if Self::use_csm_cuda_graph_env()
            && schema.arity() > 1
            && a_rows.saturating_add(b_rows) <= W66_SMALL_FULL_ROW_SORT_MAX_ROWS
        {
            return self.dedup_full_row_deterministic(&concat);
        }

        let sorted = self.sort(&concat, &key_cols)?;
        self.dedup_sorted(&sorted, &key_cols)
    }

    /// Set difference (a - b) with deterministic set semantics.
    ///
    /// Single-column `u32` buffers use a GPU sorted-diff fast path. General
    /// multi-column buffers use a byte-exact host set fallback after GPU dedup;
    /// the hash anti-join implementation is intentionally not used for Datalog
    /// delta subtraction because its unordered parallel probe path can leak
    /// nondeterminism into recursive fixed-point convergence.
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
        let num_a = self.device_row_count(a)?;
        let num_b = self.device_row_count(b)?;
        if num_a > u32::MAX as usize || num_b > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Diff supports at most {} rows per side (a={}, b={})",
                u32::MAX,
                num_a,
                num_b
            )));
        }

        if num_a == 0 {
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
            // 0-arity set difference: {()} - empty = {()}, {()} - {()} = empty.
            if num_b == 0 {
                return self.buffer_from_columns(Vec::new(), 1, a.schema().clone());
            }
            return self.create_empty_buffer(a.schema().clone());
        }

        let col_type = a
            .schema()
            .column_type(0)
            .ok_or_else(|| XlogError::Kernel("No columns".to_string()))?;

        // Keep the single-column U32 fast path; all other cases use the
        // deterministic byte-exact set-difference fallback.
        if a.arity() == 1 && matches!(col_type, ScalarType::U32) && num_b != 0 {
            return self.diff_gpu_u32(a, b);
        }

        self.diff_via_deterministic_set(a, b)
    }

    /// U32-optimized diff using GPU sort
    fn diff_gpu_u32(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        // For multi-column U32 buffers, use deterministic set difference on all columns.
        if a.arity() != 1 {
            return self.diff_via_deterministic_set(a, b);
        }

        // Step 1: Sort and dedup both inputs (sorted, so use dedup_sorted)
        let sorted_a = self.sort(a, &[0])?;
        let deduped_a = self.dedup_sorted(&sorted_a, &[0])?;

        let sorted_b = self.sort(b, &[0])?;
        let deduped_b = self.dedup_sorted(&sorted_b, &[0])?;

        let num_a = self.device_row_count(&deduped_a)?;
        let num_b = self.device_row_count(&deduped_b)?;
        if num_a > u32::MAX as usize || num_b > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Diff supports at most {} rows per side (a={}, b={})",
                u32::MAX,
                num_a,
                num_b
            )));
        }

        if num_a == 0 {
            return self.create_empty_buffer(a.schema().clone());
        }

        let num_a = num_a as u32;
        let num_b = num_b as u32;

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
        let grid_size = num_a.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Kernel signature matches:
        // sorted_diff_mark(a, a_len_device, a_cap, b, b_len_device, b_cap, in_diff)
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            diff_mark_fn.clone().launch(
                config,
                (
                    &a_view,
                    deduped_a.num_rows_device(),
                    num_a,
                    &b_view,
                    deduped_b.num_rows_device(),
                    num_b,
                    &diff_mask,
                ),
            )
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

        let d_out_count = self.capture_compact_count(&d_prefix_sum, &diff_mask, num_a)?;
        self.compact_buffer_by_device_mask_device_count(
            &deduped_a,
            &diff_mask,
            &d_prefix_sum,
            d_out_count,
        )
    }

    /// General-arity deterministic full-row set difference (a \ b) on the GPU.
    ///
    /// Pipeline:
    ///   1. Dedup both sides to set semantics (`a` and `b` may carry
    ///      duplicates from upstream union/concat steps).
    ///   2. Sort `b` by all columns using the typed multi-column sort.
    ///   3. Per-row binary search of each `a` row against sorted `b` using
    ///      the same typed comparator — `mark_diff_full_row_typed_sorted`.
    ///   4. Multi-block exclusive scan on the keep mask.
    ///   5. Column-wise gather via the existing
    ///      `compact_buffer_by_device_mask_device_count` helper.
    ///
    /// The typed comparator agrees with the multi-column sort's order
    /// convention (signed-int sign-flip; float total-order normalization),
    /// so the binary search converges. Equality under the typed comparator
    /// is bytewise equality, which is the same set semantics used by the
    /// host-side `BTreeSet<Vec<u8>>` fallback this method replaces.
    ///
    /// Recursive Datalog evaluation relies on stable delta subtraction for
    /// fixpoint convergence. This path deliberately avoids the GPU hash
    /// anti-join (whose unordered parallel probe path can leak
    /// nondeterminism into recursive fixed-point convergence).
    fn diff_via_deterministic_set(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        // Step 1: dedup both inputs to set semantics. Route through
        // `dedup_full_row_deterministic` regardless of arity so the
        // dedup step and the diff-probe typed comparator agree on
        // equality even for single-column float buffers — the legacy
        // `dedup` single-column kernel uses IEEE `==` (collapses
        // +0/-0), which would mismatch the totalOrder-key probe and
        // could drop one of {+0, -0} silently from the result.
        let deduped_a = self.dedup_full_row_deterministic(a)?;
        let deduped_b = self.dedup_full_row_deterministic(b)?;

        let a_rows = self.device_row_count(&deduped_a)? as u32;
        let b_rows = self.device_row_count(&deduped_b)? as u32;
        if a_rows == 0 {
            return self.create_empty_buffer(a.schema().clone());
        }
        if b_rows == 0 {
            return Ok(deduped_a);
        }
        let arity = deduped_a.arity();
        if arity == 0 {
            // 0-arity: {()} - {()} = empty.
            return self.create_empty_buffer(a.schema().clone());
        }

        // Step 2: `dedup_full_row_deterministic` already returns `b`
        // sorted by the same typed multi-column sort the diff probe's
        // comparator agrees with, so reuse the deduplicated buffer
        // directly — re-sorting would double the cost on a hot path
        // (negation / delta subtraction).
        let sorted_b = deduped_b;

        // Step 3: build the typed per-column descriptor arrays for the
        // diff kernel — ptrs and sizes per column for both a and b, plus
        // the type code per column (used by the typed comparator).
        let schema = deduped_a.schema().clone();
        let device = self.device.inner();

        let mut a_col_ptrs: Vec<u64> = Vec::with_capacity(arity);
        let mut b_col_ptrs: Vec<u64> = Vec::with_capacity(arity);
        let mut col_sizes: Vec<u32> = Vec::with_capacity(arity);
        let mut col_types: Vec<u8> = Vec::with_capacity(arity);
        for col_idx in 0..arity {
            let a_col = deduped_a.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("diff_full_row: a column {} missing", col_idx))
            })?;
            let b_col = sorted_b.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("diff_full_row: b column {} missing", col_idx))
            })?;
            let ty = schema.column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("diff_full_row: column {} type missing", col_idx))
            })?;
            a_col_ptrs.push(*a_col.device_ptr());
            b_col_ptrs.push(*b_col.device_ptr());
            col_sizes.push(ty.size_bytes() as u32);
            col_types.push(scalar_type_code_dedup(ty));
        }

        let mut d_a_ptrs = self.memory.alloc::<u64>(arity)?;
        let mut d_b_ptrs = self.memory.alloc::<u64>(arity)?;
        let mut d_sizes = self.memory.alloc::<u32>(arity)?;
        let mut d_types = self.memory.alloc::<u8>(arity)?;
        self.htod_launch_metadata_sync_copy_into(&a_col_ptrs, &mut d_a_ptrs)
            .map_err(|e| XlogError::Kernel(format!("diff_full_row a ptr upload: {}", e)))?;
        self.htod_launch_metadata_sync_copy_into(&b_col_ptrs, &mut d_b_ptrs)
            .map_err(|e| XlogError::Kernel(format!("diff_full_row b ptr upload: {}", e)))?;
        self.htod_launch_metadata_sync_copy_into(&col_sizes, &mut d_sizes)
            .map_err(|e| XlogError::Kernel(format!("diff_full_row size upload: {}", e)))?;
        self.htod_launch_metadata_sync_copy_into(&col_types, &mut d_types)
            .map_err(|e| XlogError::Kernel(format!("diff_full_row type upload: {}", e)))?;

        let block_size = 256u32;
        let grid = a_rows.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let d_keep_mask = self.memory.alloc::<u8>(a_rows as usize)?;
        let diff_fn = device
            .get_func(DEDUP_MODULE, dedup_kernels::MARK_DIFF_FULL_ROW_TYPED_SORTED)
            .ok_or_else(|| {
                XlogError::Kernel("mark_diff_full_row_typed_sorted kernel not found".to_string())
            })?;
        // SAFETY: kernel signature matches:
        //   mark_diff_full_row_typed_sorted(a_col_ptrs, b_col_ptrs, col_sizes,
        //       col_types, num_cols, num_a_device, num_b, a_cap, keep_mask)
        unsafe {
            diff_fn.clone().launch(
                cfg,
                (
                    &d_a_ptrs,
                    &d_b_ptrs,
                    &d_sizes,
                    &d_types,
                    arity as u32,
                    deduped_a.num_rows_device(),
                    b_rows,
                    a_rows,
                    &d_keep_mask,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("mark_diff_full_row_typed_sorted launch: {}", e)))?;
        self.device.synchronize()?;

        // Step 4 + 5: scan + column-wise gather of kept a rows.
        let (d_prefix_sum, d_out_count) =
            self.scan_mask_to_prefix_with_count(&d_keep_mask, a_rows)?;

        self.compact_buffer_by_device_mask_device_count(
            &deduped_a,
            &d_keep_mask,
            &d_prefix_sum,
            d_out_count,
        )
    }

    /// Public deterministic full-row dedup with totalOrder-bytewise
    /// equality semantics for *all* arities (including single-column
    /// float buffers).
    ///
    /// Differs from `dedup(input, &[0])` for single-column float
    /// columns: the legacy single-column GPU kernel collapses +0/-0
    /// (IEEE `==` says they're equal) and treats two NaNs with
    /// different payloads as distinct. `dedup_full_row` instead uses
    /// totalOrder-bijective bytewise equality, so:
    ///
    ///   * `+0.0` and `-0.0` are distinct.
    ///   * Two NaNs collapse iff bit-identical.
    ///
    /// Routing today (post-PR 2):
    ///   * `dedup(input, &all_cols)` with `arity > 1` routes to the
    ///     full-row pipeline (same semantics as this method).
    ///   * `dedup(input, &[0])` with `arity == 1` keeps the legacy
    ///     single-column GPU kernel — IEEE `==` for floats, so +0/-0
    ///     collapse and NaNs collapse iff bit-identical-or-IEEE-eq.
    ///   * `dedup_full_row(input)` always uses bytewise totalOrder
    ///     equality for *all* arities, so single-column float
    ///     callers must use this method explicitly to get the
    ///     totalOrder semantics.
    ///
    /// Multi-column callers that pass the all-columns key vector to
    /// `dedup` already route through the same deterministic full-row
    /// pipeline; single-column callers that want totalOrder semantics
    /// must call `dedup_full_row` directly.
    pub fn dedup_full_row(&self, input: &CudaBuffer) -> Result<CudaBuffer> {
        // Env-gated recorded dispatch. `dedup_full_row_recorded`
        // (slice #5) requires every column to be U32 / Symbol;
        // mixed-type schemas fall through to the legacy path.
        if Self::use_recorded_dedup_env() && input.num_rows() > 1 && input.arity() > 0 {
            if let Some(launch_stream) = self.recorded_op_stream_or_init() {
                let recorded_compatible = (0..input.arity()).all(|c| {
                    matches!(
                        input.schema.column_type(c),
                        Some(ScalarType::U32) | Some(ScalarType::Symbol)
                    )
                });
                if recorded_compatible {
                    return self.dedup_full_row_recorded(input, launch_stream);
                }
            }
        }
        self.dedup_full_row_deterministic(input)
    }

    /// Public deterministic full-row set difference. Equivalent to
    /// `diff_gpu(a, b)` for the multi-column path but named explicitly so
    /// callers cannot mistake it for the older first-column-key `diff`.
    /// `a` and `b` must have type-compatible schemas.
    pub fn diff_full_row(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
        // Single-column types still go through `diff_gpu` so the existing
        // u32 fast path is preserved; the deterministic-set fallback is
        // now the GPU pipeline regardless.
        self.diff_gpu(a, b)
    }

    /// Read a binary-join output-count scalar from device memory.
    ///
    /// **Why this is metadata, not data-plane:** the value is a single
    /// `u32` produced by an atomic-increment counter inside the join
    /// kernel, used solely to size the next allocation (in the
    /// count-only pass) or to drive the result buffer's logical row
    /// count (in the post-materialize pass). It is control-plane state
    /// in the same sense as a relation's row count — never tuple data.
    ///
    /// The strict deterministic-Datalog D2H gate explicitly allows this
    /// category via `dtoh_scalar_untracked`, which is the auditable,
    /// single-purpose API for metadata reads.
    ///
    /// The v0.5.5 metadata-read hardening replaced eight
    /// `dtoh_sync_copy_into_tracked` reads of these scalars with this
    /// helper, so binary-join materialization runs strict-clean.
    /// **This does not make binary-join materialization fully
    /// GPU-resident**: the count is still a host scalar, used by host
    /// code to drive allocation. A future worst-case-bounded
    /// GPU-resident output buffer can localize the upgrade here once a
    /// memory-budget-aware upper bound exists.
    fn read_join_output_count_metadata(&self, d_count: &TrackedCudaSlice<u32>) -> Result<u32> {
        // Avoid double-prefixing the error string. `XlogError` Display
        // already prefixes its variants (e.g. "Kernel error: ..."), so
        // wrapping the whole error would produce
        // "Kernel error: Failed to read output count: Kernel error: ...".
        // Extract the inner message for the common Kernel variant; fall
        // back to Display for everything else.
        self.dtoh_scalar_untracked::<u32>(d_count, 0)
            .map_err(|e| match e {
                XlogError::Kernel(message) => {
                    XlogError::Kernel(format!("Failed to read output count: {}", message))
                }
                other => XlogError::Kernel(format!("Failed to read output count: {}", other)),
            })
    }

    fn is_full_row_key(key_cols: &[usize], arity: usize) -> bool {
        key_cols.len() == arity
            && key_cols
                .iter()
                .copied()
                .enumerate()
                .all(|(expected, actual)| expected == actual)
    }

    /// Deterministic GPU full-row dedup pipeline.
    ///
    /// Pipeline: typed multi-column sort → bytewise per-column adjacent
    /// equality mask → multi-block exclusive prefix scan → column-wise
    /// gather (via `compact_buffer_by_device_mask_device_count`).
    ///
    /// Bytewise equality matches the host-fallback `BTreeSet<Vec<u8>>`
    /// semantics. For floats it agrees with IEEE-754 totalOrder equality
    /// under the project's `f{32,64}_to_ordered_u{32,64}` normalization
    /// (kernels/sort.cu): the normalization is bijective, so distinct bit
    /// patterns map to distinct ordered keys, so bytewise eq on the
    /// post-sort buffer is the same membership relation. +0/-0 stay
    /// distinct; two NaNs collapse iff bit-identical.
    ///
    /// Replaces the host-side `BTreeSet<Vec<u8>>` fallback that the
    /// strict deterministic-Datalog D2H gate flags as a violator.
    fn dedup_full_row_deterministic(&self, input: &CudaBuffer) -> Result<CudaBuffer> {
        let row_count = self.device_row_count(input)?;
        if row_count == 0 {
            return self.create_empty_buffer(input.schema().clone());
        }
        if row_count == 1 {
            return self.clone_buffer(input);
        }
        if row_count > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "dedup_full_row supports at most {} rows, got {}",
                u32::MAX,
                row_count
            )));
        }
        let arity = input.arity();
        if arity == 0 {
            // 0-arity, non-empty: collapse to {()}.
            return self.buffer_from_columns(Vec::new(), 1, input.schema().clone());
        }

        // Step 1: typed multi-column sort. Float columns use total-order
        // normalization; signed integers use sign-flipped unsigned compare.
        let sorted =
            if Self::use_csm_cuda_graph_env() && row_count <= W66_SMALL_FULL_ROW_SORT_MAX_ROWS {
                self.small_sort_full_row_deterministic(input, row_count)?
            } else {
                let all_cols: Vec<usize> = (0..arity).collect();
                self.sort(input, &all_cols)?
            };

        // Step 2: bytewise adjacent-equality mask on the sorted buffer.
        let n = self.device_row_count(&sorted)? as u32;
        if n <= 1 {
            return Ok(sorted);
        }

        let device = self.device.inner();
        let mut col_ptrs_host: Vec<u64> = Vec::with_capacity(arity);
        let mut col_sizes_host: Vec<u32> = Vec::with_capacity(arity);
        for col_idx in 0..arity {
            let col = sorted
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Sorted column {} not found", col_idx)))?;
            let ty = sorted.schema().column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Sorted column {} type missing", col_idx))
            })?;
            col_ptrs_host.push(*col.device_ptr());
            col_sizes_host.push(ty.size_bytes() as u32);
        }

        let mut d_col_ptrs = self.memory.alloc::<u64>(arity)?;
        let mut d_col_sizes = self.memory.alloc::<u32>(arity)?;
        self.htod_launch_metadata_sync_copy_into(&col_ptrs_host, &mut d_col_ptrs)
            .map_err(|e| XlogError::Kernel(format!("dedup_full_row_gpu col ptr upload: {}", e)))?;
        self.htod_launch_metadata_sync_copy_into(&col_sizes_host, &mut d_col_sizes)
            .map_err(|e| XlogError::Kernel(format!("dedup_full_row_gpu col size upload: {}", e)))?;

        let block_size = 256u32;
        let grid = n.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let d_unique_mask = self.memory.alloc::<u8>(n as usize)?;
        let mark_fn = device
            .get_func(DEDUP_MODULE, dedup_kernels::MARK_UNIQUE_FULL_ROW_BYTEWISE)
            .ok_or_else(|| {
                XlogError::Kernel("mark_unique_full_row_bytewise kernel not found".to_string())
            })?;

        // SAFETY: kernel signature matches:
        //   mark_unique_full_row_bytewise(col_ptrs, col_sizes, num_cols,
        //                                 num_rows_device, row_cap, unique_mask)
        unsafe {
            mark_fn.clone().launch(
                cfg,
                (
                    &d_col_ptrs,
                    &d_col_sizes,
                    arity as u32,
                    sorted.num_rows_device(),
                    n,
                    &d_unique_mask,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "mark_unique_full_row_bytewise launch failed: {}",
                e
            ))
        })?;
        self.device.synchronize()?;

        // Step 3: exclusive prefix scan over the mask.
        let (d_prefix_sum, d_out_count) = self.scan_mask_to_prefix_with_count(&d_unique_mask, n)?;

        // Step 4: gather the kept rows using the existing column-wise
        // compaction helper. This reuses the same machinery the
        // existing GPU dedup_sorted typed-columnar path uses.
        self.compact_buffer_by_device_mask_device_count(
            &sorted,
            &d_unique_mask,
            &d_prefix_sum,
            d_out_count,
        )
    }

    fn small_sort_full_row_deterministic(
        &self,
        input: &CudaBuffer,
        row_count: usize,
    ) -> Result<CudaBuffer> {
        if row_count > W66_SMALL_FULL_ROW_SORT_MAX_ROWS {
            return Err(XlogError::Kernel(format!(
                "small full-row sort supports at most {} rows, got {}",
                W66_SMALL_FULL_ROW_SORT_MAX_ROWS, row_count
            )));
        }
        if row_count == 0 {
            return self.create_empty_buffer(input.schema().clone());
        }
        if row_count == 1 {
            return self.clone_buffer(input);
        }

        let arity = input.arity();
        let device = self.device.inner();
        let mut col_ptrs_host: Vec<u64> = Vec::with_capacity(arity);
        let mut col_sizes_host: Vec<u32> = Vec::with_capacity(arity);
        let mut col_types_host: Vec<u8> = Vec::with_capacity(arity);
        for col_idx in 0..arity {
            let col = input.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("small full-row sort: column {} missing", col_idx))
            })?;
            let ty = input.schema().column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!(
                    "small full-row sort: column {} type missing",
                    col_idx
                ))
            })?;
            let elem_size = ty.size_bytes();
            let expected_bytes_u64 =
                input
                    .num_rows()
                    .checked_mul(elem_size as u64)
                    .ok_or_else(|| {
                        XlogError::Kernel(
                            "small full-row sort: column byte-size overflow".to_string(),
                        )
                    })?;
            let expected_bytes = usize::try_from(expected_bytes_u64).map_err(|_| {
                XlogError::Kernel(format!(
                    "small full-row sort: expected byte size {} exceeds usize::MAX",
                    expected_bytes_u64
                ))
            })?;
            if col.num_bytes() != expected_bytes {
                return Err(XlogError::Kernel(format!(
                    "small full-row sort: column {} has {} bytes but expected {}",
                    col_idx,
                    col.num_bytes(),
                    expected_bytes
                )));
            }
            col_ptrs_host.push(*col.device_ptr());
            col_sizes_host.push(elem_size as u32);
            col_types_host.push(scalar_type_code_dedup(ty));
        }

        let mut d_col_ptrs = self.memory.alloc::<u64>(arity)?;
        let mut d_col_sizes = self.memory.alloc::<u32>(arity)?;
        let mut d_col_types = self.memory.alloc::<u8>(arity)?;
        self.htod_launch_metadata_sync_copy_into(&col_ptrs_host, &mut d_col_ptrs)
            .map_err(|e| XlogError::Kernel(format!("small full-row sort ptr upload: {}", e)))?;
        self.htod_launch_metadata_sync_copy_into(&col_sizes_host, &mut d_col_sizes)
            .map_err(|e| XlogError::Kernel(format!("small full-row sort size upload: {}", e)))?;
        self.htod_launch_metadata_sync_copy_into(&col_types_host, &mut d_col_types)
            .map_err(|e| XlogError::Kernel(format!("small full-row sort type upload: {}", e)))?;

        let mut d_indices = self.memory.alloc::<u32>(row_count)?;
        let sort_fn = device
            .get_func(
                DEDUP_MODULE,
                dedup_kernels::SMALL_SORT_FULL_ROW_INDICES_TYPED,
            )
            .ok_or_else(|| {
                XlogError::Kernel("small_sort_full_row_indices_typed kernel not found".to_string())
            })?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (W66_SMALL_FULL_ROW_SORT_MAX_ROWS as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: kernel signature matches:
        //   small_sort_full_row_indices_typed(col_ptrs, col_sizes, col_types,
        //       num_cols, num_rows_device, row_cap, out_indices)
        unsafe {
            sort_fn.clone().launch(
                cfg,
                (
                    &d_col_ptrs,
                    &d_col_sizes,
                    &d_col_types,
                    arity as u32,
                    input.num_rows_device(),
                    row_count as u32,
                    &mut d_indices,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "small_sort_full_row_indices_typed launch failed: {}",
                e
            ))
        })?;
        self.device.synchronize()?;
        self.small_full_row_sort_invocations
            .fetch_add(1, Ordering::Relaxed);

        self.gather_buffer_by_indices(input, &d_indices, row_count as u32)
    }

    /// Run the multi-block exclusive-scan pipeline on a u8 mask of length
    /// `n` and return the per-row prefix-sum buffer plus a device-resident
    /// scalar with the total number of marked rows. Mirrors the helper
    /// pattern already used by `diff_gpu_u32` and `dedup_sorted`.
    fn scan_mask_to_prefix_with_count(
        &self,
        d_mask: &cudarc::driver::CudaSlice<u8>,
        n: u32,
    ) -> Result<(
        crate::memory::TrackedCudaSlice<u32>,
        crate::memory::TrackedCudaSlice<u32>,
    )> {
        let device = self.device.inner();
        let block_size = 256u32;
        let num_blocks = n.div_ceil(block_size);

        let d_prefix_sum = self.memory.alloc::<u32>(n as usize)?;
        let mut d_block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

        let phase1_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase1 kernel".to_string())
            })?;
        // SAFETY: kernel signature matches multiblock_scan_phase1.
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
            // SAFETY: kernel signature matches multiblock_scan_phase3.
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

        let d_out_count = self.capture_compact_count(&d_prefix_sum, d_mask, n)?;
        Ok((d_prefix_sum, d_out_count))
    }

    // ============== Sort Methods ==============

    pub(super) const SORT_BLOCK_SIZE: u32 = 256;

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
        // Env-gated recorded dispatch. Eligibility check
        // mirrors `sort_recorded`'s validation (slice #5):
        // U32 / Symbol key columns only. Other types fall
        // through to the legacy multi-type path.
        if Self::use_recorded_sort_env() && !key_cols.is_empty() && input.num_rows() > 0 {
            if let Some(launch_stream) = self.recorded_op_stream_or_init() {
                let recorded_compatible = key_cols.iter().all(|&k| {
                    matches!(
                        input.schema.column_type(k),
                        Some(ScalarType::U32) | Some(ScalarType::Symbol)
                    )
                });
                if recorded_compatible {
                    return self.sort_recorded(input, key_cols, launch_stream);
                }
            }
        }

        if input.num_rows() == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }

        if key_cols.is_empty() {
            return Err(XlogError::Kernel(
                "Sort requires at least one key column".to_string(),
            ));
        }

        if input.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Sort supports at most {} rows, got {}",
                u32::MAX,
                input.num_rows()
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

        let n = input.num_rows() as u32;
        let d_num_rows = input.num_rows_device();
        let device = self.device.inner();

        let block_size = Self::SORT_BLOCK_SIZE;
        let grid_size = n.div_ceil(block_size);
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

        // SAFETY: init_indices(indices, num_rows_device, row_cap)
        unsafe {
            init_fn
                .clone()
                .launch(launch_config, (&mut indices_a, d_num_rows, n))
        }
        .map_err(|e| XlogError::Kernel(format!("init_indices failed: {}", e)))?;
        self.device.synchronize()?;

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

                    // SAFETY: apply_permutation_u32(input, output, permutation, num_rows_device, row_cap)
                    unsafe {
                        gather_fn.clone().launch(
                            launch_config,
                            (&col_view, &mut keys_a, &indices_a, d_num_rows, n),
                        )
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
                        d_num_rows,
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

                    // SAFETY: gather_keys_i32_ordered_u32(i32_bits, permutation, num_rows_device, row_cap, out_keys)
                    unsafe {
                        gather_fn.clone().launch(
                            launch_config,
                            (&col_bits, &indices_a, d_num_rows, n, &mut keys_a),
                        )
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
                        d_num_rows,
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

                    // SAFETY: gather_keys_f32_ordered_u32(f32_bits, permutation, num_rows_device, row_cap, out_keys)
                    unsafe {
                        gather_fn.clone().launch(
                            launch_config,
                            (&col_bits, &indices_a, d_num_rows, n, &mut keys_a),
                        )
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
                        d_num_rows,
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

                    // SAFETY: gather_keys_bool_ordered_u32(bools, permutation, num_rows_device, row_cap, out_keys)
                    unsafe {
                        gather_fn
                            .clone()
                            .launch(launch_config, (col, &indices_a, d_num_rows, n, &mut keys_a))
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
                        d_num_rows,
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

                        // SAFETY: gather_keys_u64_*_u32(vals, permutation, num_rows_device, row_cap, out_keys)
                        unsafe {
                            gather_fn.clone().launch(
                                launch_config,
                                (&col_bits, &indices_a, d_num_rows, n, &mut keys_a),
                            )
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
                            d_num_rows,
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

                        // SAFETY: gather_keys_i64_*_u32(i64_bits, permutation, num_rows_device, row_cap, out_keys)
                        unsafe {
                            gather_fn.clone().launch(
                                launch_config,
                                (&col_bits, &indices_a, d_num_rows, n, &mut keys_a),
                            )
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
                            d_num_rows,
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

                        // SAFETY: gather_keys_f64_*_u32(f64_bits, permutation, num_rows_device, row_cap, out_keys)
                        unsafe {
                            gather_fn.clone().launch(
                                launch_config,
                                (&col_bits, &indices_a, d_num_rows, n, &mut keys_a),
                            )
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
                            d_num_rows,
                            n,
                        )?;
                    }
                }
            }
        }

        self.apply_permutation_gpu(input, &indices_a)
    }

    #[allow(clippy::too_many_arguments)]
    fn radix_sort_u32_pairs_with_scratch(
        &self,
        keys_a: &mut crate::memory::TrackedCudaSlice<u32>,
        keys_b: &mut crate::memory::TrackedCudaSlice<u32>,
        indices_a: &mut crate::memory::TrackedCudaSlice<u32>,
        indices_b: &mut crate::memory::TrackedCudaSlice<u32>,
        hist: &mut crate::memory::TrackedCudaSlice<u32>,
        prefix: &mut crate::memory::TrackedCudaSlice<u32>,
        ranks: &mut crate::memory::TrackedCudaSlice<u32>,
        num_rows_device: &crate::memory::TrackedCudaSlice<u32>,
        row_cap: u32,
    ) -> Result<()> {
        if row_cap == 0 {
            return Ok(());
        }
        self.device.synchronize()?;

        let device = self.device.inner();
        let block_size = Self::SORT_BLOCK_SIZE;
        let grid_size = row_cap.div_ceil(block_size);

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
            // SAFETY: radix_histogram(keys, num_rows_device, row_cap, histograms, shift)
            unsafe {
                histogram_fn.clone().launch(
                    sort_config,
                    (keys_in, num_rows_device, row_cap, &mut *hist, shift),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("radix_histogram failed: {}", e)))?;
            self.device.synchronize()?;

            // Compute global digit prefix sums.
            // SAFETY: compute_digit_prefix_sums(histograms, grid_size, prefix_sums)
            unsafe {
                prefix_fn
                    .clone()
                    .launch(prefix_config, (&*hist, grid_size, &mut *prefix))
            }
            .map_err(|e| XlogError::Kernel(format!("compute_digit_prefix_sums failed: {}", e)))?;
            self.device.synchronize()?;

            // Convert per-block histograms to per-block exclusive offsets (in-place scan per digit).
            for digit in 0..16u32 {
                let start = (digit * grid_size) as usize;
                let end = start + (grid_size as usize);
                let mut digit_slice = hist.slice_mut(start..end);
                self.multiblock_scan_u32_view_inplace(&mut digit_slice, grid_size)?;
            }
            self.device.synchronize()?;

            // Compute per-element ranks for stability.
            // SAFETY: compute_ranks(keys, num_rows_device, row_cap, ranks, shift)
            unsafe {
                ranks_fn.clone().launch(
                    sort_config,
                    (keys_in, num_rows_device, row_cap, &mut *ranks, shift),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("compute_ranks failed: {}", e)))?;
            self.device.synchronize()?;

            // Stable scatter using digit prefix + per-block offsets + ranks.
            // SAFETY: radix_scatter_stable(keys_in, indices_in, ranks, keys_out, indices_out, prefix_sums, block_offsets, num_rows_device, row_cap, shift)
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
                        num_rows_device,
                        row_cap,
                        shift,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("radix_scatter_stable failed: {}", e)))?;
            self.device.synchronize()?;

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
    /// Initialize indices array with 0..n-1 on device.
    pub fn init_indices(
        &self,
        indices: &mut crate::memory::TrackedCudaSlice<u32>,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if n as usize > indices.len() {
            return Err(XlogError::Kernel(format!(
                "init_indices: n={} exceeds indices len={}",
                n,
                indices.len()
            )));
        }
        let device = self.device.inner();
        let block_size = Self::SORT_BLOCK_SIZE;
        let grid_size = n.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let init_fn = device
            .get_func(SORT_MODULE, sort_kernels::INIT_INDICES)
            .ok_or_else(|| XlogError::Kernel("init_indices kernel not found".to_string()))?;
        let d_num_rows = self.upload_device_row_count(n)?;
        // SAFETY: init_indices(indices, num_rows_device, row_cap)
        unsafe {
            init_fn
                .clone()
                .launch(config, (&mut *indices, &d_num_rows, n))
        }
        .map_err(|e| XlogError::Kernel(format!("init_indices failed: {}", e)))?;
        Ok(())
    }

    /// Gather u32 keys by permutation: out[i] = input[indices[i]].
    pub fn gather_u32_by_indices(
        &self,
        input: &crate::memory::TrackedCudaSlice<u32>,
        indices: &crate::memory::TrackedCudaSlice<u32>,
        output: &mut crate::memory::TrackedCudaSlice<u32>,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if n as usize > output.len() {
            return Err(XlogError::Kernel(format!(
                "gather_u32_by_indices: n={} exceeds output len={}",
                n,
                output.len()
            )));
        }
        let device = self.device.inner();
        let block_size = Self::SORT_BLOCK_SIZE;
        let grid_size = n.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let gather_fn = device
            .get_func(SORT_MODULE, sort_kernels::APPLY_PERMUTATION_U32)
            .ok_or_else(|| {
                XlogError::Kernel("apply_permutation_u32 kernel not found".to_string())
            })?;
        let d_num_rows = self.upload_device_row_count(n)?;
        // SAFETY: apply_permutation_u32(input, output, permutation, num_rows_device, row_cap)
        unsafe {
            gather_fn
                .clone()
                .launch(config, (input, output, indices, &d_num_rows, n))
        }
        .map_err(|e| XlogError::Kernel(format!("gather_u32_by_indices failed: {}", e)))?;
        Ok(())
    }

    /// Gather u8 values by permutation: out[i] = input[indices[i]].
    pub fn gather_u8_by_indices(
        &self,
        input: &crate::memory::TrackedCudaSlice<u8>,
        indices: &crate::memory::TrackedCudaSlice<u32>,
        output: &mut crate::memory::TrackedCudaSlice<u8>,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if n as usize > output.len() {
            return Err(XlogError::Kernel(format!(
                "gather_u8_by_indices: n={} exceeds output len={}",
                n,
                output.len()
            )));
        }
        let device = self.device.inner();
        let block_size = Self::SORT_BLOCK_SIZE;
        let grid_size = n.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let gather_fn = device
            .get_func(SORT_MODULE, sort_kernels::APPLY_PERMUTATION_BYTES)
            .ok_or_else(|| {
                XlogError::Kernel("apply_permutation_bytes kernel not found".to_string())
            })?;
        let d_num_rows = self.upload_device_row_count(n)?;
        // SAFETY: apply_permutation_bytes(input, output, permutation, num_rows_device, row_cap, elem_size)
        unsafe {
            gather_fn
                .clone()
                .launch(config, (input, output, indices, &d_num_rows, n, 1u32))
        }
        .map_err(|e| XlogError::Kernel(format!("gather_u8_by_indices failed: {}", e)))?;
        Ok(())
    }

    /// Gather low 32 bits of u64 values by permutation.
    pub fn gather_u64_lo_by_indices(
        &self,
        input: &crate::memory::TrackedCudaSlice<u64>,
        indices: &crate::memory::TrackedCudaSlice<u32>,
        output: &mut crate::memory::TrackedCudaSlice<u32>,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let device = self.device.inner();
        let block_size = Self::SORT_BLOCK_SIZE;
        let grid_size = n.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let gather_fn = device
            .get_func(SORT_MODULE, sort_kernels::GATHER_KEYS_U64_LO_U32)
            .ok_or_else(|| XlogError::Kernel("gather_keys_u64_lo_u32 not found".to_string()))?;
        let d_num_rows = self.upload_device_row_count(n)?;
        // SAFETY: gather_keys_u64_lo_u32(vals, permutation, num_rows_device, row_cap, out_keys)
        unsafe {
            gather_fn
                .clone()
                .launch(config, (input, indices, &d_num_rows, n, output))
        }
        .map_err(|e| XlogError::Kernel(format!("gather_u64_lo_by_indices failed: {}", e)))?;
        Ok(())
    }

    /// Gather high 32 bits of u64 values by permutation.
    pub fn gather_u64_hi_by_indices(
        &self,
        input: &crate::memory::TrackedCudaSlice<u64>,
        indices: &crate::memory::TrackedCudaSlice<u32>,
        output: &mut crate::memory::TrackedCudaSlice<u32>,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let device = self.device.inner();
        let block_size = Self::SORT_BLOCK_SIZE;
        let grid_size = n.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let gather_fn = device
            .get_func(SORT_MODULE, sort_kernels::GATHER_KEYS_U64_HI_U32)
            .ok_or_else(|| XlogError::Kernel("gather_keys_u64_hi_u32 not found".to_string()))?;
        let d_num_rows = self.upload_device_row_count(n)?;
        // SAFETY: gather_keys_u64_hi_u32(vals, permutation, num_rows_device, row_cap, out_keys)
        unsafe {
            gather_fn
                .clone()
                .launch(config, (input, indices, &d_num_rows, n, output))
        }
        .map_err(|e| XlogError::Kernel(format!("gather_u64_hi_by_indices failed: {}", e)))?;
        Ok(())
    }

    /// Stable radix sort of (key, value) u32 pairs using reusable scratch.
    pub fn radix_sort_u32_pairs(
        &self,
        keys: &mut crate::memory::TrackedCudaSlice<u32>,
        values: &mut crate::memory::TrackedCudaSlice<u32>,
        n: u32,
        scratch: &mut RadixSortScratch,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        scratch.ensure_capacity(self, n)?;
        let d_num_rows = self.upload_device_row_count(n)?;
        self.radix_sort_u32_pairs_with_scratch(
            keys,
            &mut scratch.keys_b,
            values,
            &mut scratch.values_b,
            &mut scratch.hist,
            &mut scratch.prefix,
            &mut scratch.ranks,
            &d_num_rows,
            n,
        )
    }
    /// Compute exclusive prefix sum of u8 mask on device (no host reads).
    pub fn scan_u8_mask_device(
        &self,
        mask: &crate::memory::TrackedCudaSlice<u8>,
        n: u32,
    ) -> Result<crate::memory::TrackedCudaSlice<u32>> {
        if n == 0 {
            return self.memory.alloc::<u32>(0);
        }
        if n as usize > mask.len() {
            return Err(XlogError::Kernel(format!(
                "scan_u8_mask_device: n={} exceeds mask len={}",
                n,
                mask.len()
            )));
        }
        let device = self.device.inner();
        let block_size = 256u32;
        let num_blocks = n.div_ceil(block_size);

        let mut prefix_sum = self.memory.alloc::<u32>(n as usize)?;
        let mut block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

        let phase1_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("multiblock_scan_phase1 kernel not found".to_string())
            })?;

        // SAFETY: multiblock_scan_phase1(mask, prefix_sum, block_sums, n)
        unsafe {
            phase1_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (mask, &mut prefix_sum, &mut block_sums, n),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase1 failed: {}", e)))?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace(&mut block_sums, num_blocks)?;

            let phase3_fn = device
                .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE3)
                .ok_or_else(|| {
                    XlogError::Kernel("multiblock_scan_phase3 kernel not found".to_string())
                })?;

            // SAFETY: multiblock_scan_phase3(prefix_sum, block_offsets, n)
            unsafe {
                phase3_fn.clone().launch(
                    LaunchConfig {
                        grid_dim: (num_blocks, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut prefix_sum, &block_sums, n),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;
        }

        Ok(prefix_sum)
    }

    /// Count non-zero entries in a u8 mask on device (no host reads).
    ///
    /// Returns a 1-element device buffer containing the count.
    pub fn count_mask_device(
        &self,
        mask: &crate::memory::TrackedCudaSlice<u8>,
        n: u32,
    ) -> Result<crate::memory::TrackedCudaSlice<u32>> {
        let mut d_count = self.memory.alloc::<u32>(1)?;
        self.htod_launch_metadata_sync_copy_into(&[0u32], &mut d_count)
            .map_err(|e| {
                XlogError::Kernel(format!("count_mask_device: zero init failed: {}", e))
            })?;

        if n == 0 {
            return Ok(d_count);
        }

        let device = self.device.inner();
        let block_size = 256u32;
        let grid_size = n.div_ceil(block_size);

        let count_fn = device
            .get_func(SCAN_MODULE, scan_kernels::COUNT_MASK)
            .ok_or_else(|| XlogError::Kernel("count_mask kernel not found".to_string()))?;

        // SAFETY: count_mask(mask, n, count)
        unsafe {
            count_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (mask, n, &mut d_count),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("count_mask kernel failed: {}", e)))?;

        self.device.synchronize()?;

        Ok(d_count)
    }

    /// Count 1-bits in `mask[0..n]` and write the result into
    /// `task_counts[slot_idx]` via the existing `count_mask` kernel.
    ///
    /// The caller MUST ensure `task_counts[slot_idx]` is zero before
    /// calling (e.g. by zeroing the whole array once).
    ///
    /// This avoids allocating a fresh 1-element device buffer per call,
    /// which matters when iterating over hundreds of tasks.
    pub fn count_mask_into_slot(
        &self,
        mask: &crate::memory::TrackedCudaSlice<u8>,
        n: u32,
        task_counts: &mut crate::memory::TrackedCudaSlice<u32>,
        slot_idx: usize,
    ) -> Result<()> {
        if n == 0 {
            // Slot is already zero (caller pre-zeroed); nothing to do.
            return Ok(());
        }
        if slot_idx >= task_counts.len() {
            return Err(XlogError::Kernel(format!(
                "count_mask_into_slot: slot_idx={} >= len={}",
                slot_idx,
                task_counts.len()
            )));
        }

        let device = self.device.inner();
        let block_size = 256u32;
        let grid_size = n.div_ceil(block_size);

        let count_fn = device
            .get_func(SCAN_MODULE, scan_kernels::COUNT_MASK)
            .ok_or_else(|| XlogError::Kernel("count_mask kernel not found".to_string()))?;

        // Get a mutable sub-slice pointing at task_counts[slot_idx..slot_idx+1].
        let mut slot = task_counts.slice_mut(slot_idx..slot_idx + 1);

        // SAFETY: count_mask(mask, n, count) — writes atomicAdd into count ptr.
        // The slot was pre-zeroed by the caller.
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            count_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (mask, n, &mut slot),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("count_mask_into_slot kernel failed: {}", e)))?;

        Ok(())
    }
    /// Apply permutation to reorder all columns in buffer using GPU
    fn apply_permutation_gpu(
        &self,
        input: &CudaBuffer,
        permutation: &cudarc::driver::CudaSlice<u32>,
    ) -> Result<CudaBuffer> {
        let row_cap = input.num_rows() as u32;
        let d_num_rows = input.num_rows_device();
        let device = self.device.inner();

        let grid_size = row_cap.div_ceil(Self::SORT_BLOCK_SIZE);
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

            let output_bytes = (row_cap as usize) * (elem_size as usize);
            if src_col.num_bytes() != output_bytes {
                return Err(XlogError::Kernel(format!(
                    "Column {} has {} bytes but expected {} (num_rows={}, elem_size={})",
                    col_idx,
                    src_col.num_bytes(),
                    output_bytes,
                    row_cap,
                    elem_size
                )));
            }
            let dst_col = self.memory.alloc::<u8>(output_bytes)?;

            // SAFETY: Kernel signature matches: apply_permutation_bytes(input, output, permutation, num_rows_device, row_cap, elem_size)
            unsafe {
                apply_perm_fn.clone().launch(
                    launch_config,
                    (
                        src_col,
                        &dst_col,
                        permutation,
                        d_num_rows,
                        row_cap,
                        elem_size,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("apply_permutation_bytes failed: {}", e)))?;

            new_columns.push(dst_col.into());
        }

        self.device.synchronize()?;

        self.buffer_from_columns_with_device_count(
            new_columns,
            input.num_rows(),
            input.schema.clone(),
            input,
        )
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

        let d_output_rows = self.upload_device_row_count(output_rows)?;
        let device = self.device.inner();
        let block_size = 256u32;
        let grid_size = output_rows.div_ceil(block_size);
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

            let expected_src_bytes = (input.num_rows() as usize) * (elem_size as usize);
            if src_col.num_bytes() != expected_src_bytes {
                return Err(XlogError::Kernel(format!(
                    "Column {} has {} bytes but expected {} (num_rows={}, elem_size={})",
                    col_idx,
                    src_col.num_bytes(),
                    expected_src_bytes,
                    input.num_rows(),
                    elem_size
                )));
            }

            let dst_bytes = (output_rows as usize) * (elem_size as usize);
            let dst_col = self.memory.alloc::<u8>(dst_bytes)?;

            // SAFETY: Kernel signature matches: apply_permutation_bytes(input, output, permutation, num_rows_device, row_cap, elem_size)
            unsafe {
                gather_fn.clone().launch(
                    launch_config,
                    (
                        src_col,
                        &dst_col,
                        indices,
                        &d_output_rows,
                        output_rows,
                        elem_size,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("apply_permutation_bytes failed: {}", e)))?;

            new_columns.push(dst_col.into());
        }

        self.device.synchronize()?;

        Ok(CudaBuffer::from_columns(
            new_columns,
            output_rows as u64,
            d_output_rows,
            input.schema.clone(),
        ))
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
        // Env-gated recorded dispatch. The `pack_keys`
        // constraint inherited by `hash_join_v2_recorded`
        // requires `left_keys.len() <= 4`. Mismatch falls
        // through to the legacy path.
        if Self::use_recorded_hash_join_env()
            && !left_keys.is_empty()
            && left_keys.len() == right_keys.len()
            && left_keys.len() <= 4
        {
            if let Some(launch_stream) = self.recorded_op_stream_or_init() {
                return self.hash_join_v2_recorded(
                    left,
                    right,
                    left_keys,
                    right_keys,
                    join_type,
                    max_output,
                    launch_stream,
                );
            }
        }
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

    /// W4.2 nested-loop inner join (emit-pairs design).
    ///
    /// Drop-in compatible with `hash_join_v2(_, _, &[left_key],
    /// &[right_key], JoinType::Inner)`: same input types, same
    /// output schema (`combine_schemas(left, right)`), same row
    /// set. Caller (the executor's dispatch site, Step 5) is
    /// responsible for choosing between `hash_join_v2` and this
    /// fn based on the eligibility predicate + threshold check;
    /// this fn validates the same contract fail-closed and
    /// returns `Err` if a caller violates it.
    ///
    /// # Eligibility (validated inside; `Err` on violation)
    ///
    /// * `left.arity() > left_key && right.arity() > right_key`.
    /// * Left and right key columns share the same `ScalarType`,
    ///   and that shared type is `U32` or `Symbol` (Symbol is
    ///   `u32` at the byte level — same kernel applies).
    /// * Each key column's allocation is at least `num_rows * 4`
    ///   bytes (preflight lower-bound validation; mirrors the
    ///   `crates/xlog-cuda/src/provider/ilp.rs:18` codebase idiom
    ///   `col.num_bytes() < required_bytes`). `CudaColumn::num_bytes()`
    ///   reports the allocation size, which can exceed
    ///   `num_rows * 4` when the buffer has spare capacity
    ///   (`row_cap > num_rows`); strict-equality validation would
    ///   false-positive-reject normal over-allocated buffers
    ///   reaching this path through `Executor::execute_node`.
    /// * `num_left * num_right <= NESTED_LOOP_TOTAL_THRESHOLD`
    ///   (computed via `checked_mul`; release-mode wrapping
    ///   multiply is forbidden).
    ///
    /// # Implementation outline
    ///
    /// 1. Read logical row counts via `device_row_count` (NOT
    ///    `row_cap`).
    /// 2. Empty-input fast path: if either side is empty, return
    ///    `create_empty_buffer(combine_schemas(...))` — mirrors
    ///    `hash_join_inner_v2`'s pattern at
    ///    `crates/xlog-cuda/src/provider/relational.rs:3165-3170`.
    /// 3. Validate eligibility (above).
    /// 4. Allocate two `u32` index arrays of length
    ///    `num_left * num_right` (bounded at 32 MB total under
    ///    the threshold).
    /// 5. Launch `nested_loop_join_inner_u32_1key_pairs` with
    ///    `&CudaColumn` key pointers (variant-agnostic).
    /// 6. D2H the output count.
    /// 7. Materialize via `gather_buffer_by_indices` for both
    ///    sides + concatenate columns.
    pub fn nested_loop_join_v2_inner_u32_1key(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_key: usize,
        right_key: usize,
    ) -> Result<CudaBuffer> {
        // ----- 1. Logical row counts (NOT row_cap) -----
        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;

        // ----- 2. Empty-input fast path (inner-join schema) -----
        if num_left == 0 || num_right == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        // ----- 3. Eligibility validation -----
        if left.arity() <= left_key {
            return Err(XlogError::Kernel(format!(
                "nested_loop: left_key={} out of bounds (arity={})",
                left_key,
                left.arity()
            )));
        }
        if right.arity() <= right_key {
            return Err(XlogError::Kernel(format!(
                "nested_loop: right_key={} out of bounds (arity={})",
                right_key,
                right.arity()
            )));
        }
        let lt = left.schema().column_type(left_key);
        let rt = right.schema().column_type(right_key);
        if lt != rt || !matches!(lt, Some(ScalarType::U32) | Some(ScalarType::Symbol)) {
            return Err(XlogError::Kernel(format!(
                "nested_loop: key types must be equal U32/Symbol; got left={:?} right={:?}",
                lt, rt
            )));
        }
        let left_col = left
            .column(left_key)
            .ok_or_else(|| XlogError::Kernel(format!("nested_loop: left.column({})", left_key)))?;
        let right_col = right.column(right_key).ok_or_else(|| {
            XlogError::Kernel(format!("nested_loop: right.column({})", right_key))
        })?;
        // Byte-length lower-bound check (per F-W42-14, corrected
        // to ≥-semantic). The codebase convention is that
        // `CudaColumn::num_bytes()` reports the ALLOCATION size,
        // which can exceed `num_rows * sizeof(T)` when the buffer
        // has spare capacity (row_cap > num_rows). The check
        // must therefore be a lower-bound (column has AT LEAST
        // enough bytes for the kernel's `num_rows` reads), NOT
        // strict equality. Mirrors the `ilp.rs:18` idiom in this
        // codebase (`col.num_bytes() < required_bytes` for the
        // failure case). Strict equality would falsely reject
        // any normal buffer with spare allocation — surfaced as
        // a regression in `test_simple_join` and
        // `test_transitive_closure` after the Step-5 dispatch
        // wiring routed those joins through this path.
        let required_left_bytes = num_left
            .checked_mul(4)
            .ok_or_else(|| XlogError::Kernel("nested_loop: left byte-count overflow".into()))?;
        let required_right_bytes = num_right
            .checked_mul(4)
            .ok_or_else(|| XlogError::Kernel("nested_loop: right byte-count overflow".into()))?;
        if left_col.num_bytes() < required_left_bytes {
            return Err(XlogError::Kernel(format!(
                "nested_loop: left key column has {} bytes; \
                 require at least {} ({} rows × 4) — buffer allocation \
                 is smaller than logical row count",
                left_col.num_bytes(),
                required_left_bytes,
                num_left
            )));
        }
        if right_col.num_bytes() < required_right_bytes {
            return Err(XlogError::Kernel(format!(
                "nested_loop: right key column has {} bytes; \
                 require at least {} ({} rows × 4) — buffer allocation \
                 is smaller than logical row count",
                right_col.num_bytes(),
                required_right_bytes,
                num_right
            )));
        }

        // ----- 4. Fail-closed threshold check via checked_mul -----
        let upper_bound: u64 = (num_left as u64)
            .checked_mul(num_right as u64)
            .ok_or_else(|| XlogError::Kernel("nested_loop: row-count product overflow".into()))?;
        if upper_bound > NESTED_LOOP_TOTAL_THRESHOLD {
            return Err(XlogError::Kernel(format!(
                "nested_loop: caller violated eligibility threshold: \
                 num_left * num_right = {} > {} (NESTED_LOOP_TOTAL_THRESHOLD)",
                upper_bound, NESTED_LOOP_TOTAL_THRESHOLD
            )));
        }

        // ----- 5. Allocate index arrays + counter -----
        let upper_bound_usize = upper_bound as usize;
        let mut d_output_left_idx = self.memory.alloc::<u32>(upper_bound_usize)?;
        let mut d_output_right_idx = self.memory.alloc::<u32>(upper_bound_usize)?;
        let mut d_output_count = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .memset_zeros(&mut d_output_count)
            .map_err(|e| XlogError::Kernel(format!("nested_loop: counter zero failed: {}", e)))?;

        // ----- 6. Launch kernel (variant-agnostic column refs) -----
        let func = self
            .device
            .inner()
            .get_func(
                JOIN_MODULE,
                join_kernels::NESTED_LOOP_JOIN_INNER_U32_1KEY_PAIRS,
            )
            .ok_or_else(|| {
                XlogError::Kernel("nested_loop_join_inner_u32_1key_pairs kernel not found".into())
            })?;

        let num_left_u32 = num_left as u32;
        let num_right_u32 = num_right as u32;
        let upper_bound_u32 = upper_bound as u32;
        let block_size = 256u32;
        let grid_size = num_left_u32.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: kernel signature matches PTX:
        //   nested_loop_join_inner_u32_1key_pairs(
        //     const uint32_t* left_keys, const uint32_t* right_keys,
        //     uint32_t num_left, uint32_t num_right,
        //     uint32_t* output_left_idx, uint32_t* output_right_idx,
        //     uint32_t* output_count, uint32_t output_capacity)
        // Byte lengths validated above; counts fit in u32 by the
        // threshold; allocations sized to upper_bound; counter
        // pre-zeroed.
        unsafe {
            func.clone()
                .launch(
                    config,
                    (
                        left_col,
                        right_col,
                        num_left_u32,
                        num_right_u32,
                        &mut d_output_left_idx,
                        &mut d_output_right_idx,
                        &mut d_output_count,
                        upper_bound_u32,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("nested_loop launch failed: {}", e)))?;
        }

        self.device.synchronize()?;

        // ----- 7. D2H the output count (single u32) -----
        let output_rows = self.dtoh_scalar_untracked(&d_output_count, 0)?;
        // Defense-in-depth: kernel guarantees output_rows ≤
        // upper_bound by the in-kernel atomic-cap branch, but
        // double-check to surface contract violations early.
        if (output_rows as u64) > upper_bound {
            return Err(XlogError::Kernel(format!(
                "nested_loop: kernel reported {} output rows > upper_bound {}",
                output_rows, upper_bound
            )));
        }

        // ----- 8. Gather both sides via existing GPU machinery -----
        let gathered_left = self.gather_buffer_by_indices(left, &d_output_left_idx, output_rows)?;
        let gathered_right =
            self.gather_buffer_by_indices(right, &d_output_right_idx, output_rows)?;

        // ----- 9. Combine columns + return drop-in result -----
        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        result_columns.extend(gathered_left.columns);
        result_columns.extend(gathered_right.columns);
        // `buffer_from_columns` takes `row_cap: u64` — see
        // `crates/xlog-cuda/src/provider/mod.rs:2133-2138`.
        self.buffer_from_columns(result_columns, output_rows as u64, combined_schema)
    }

    /// W4.3 sortedness-detection wrapper. Returns `Ok(true)` iff
    /// the column at `key_col` of `buf` is sorted ascending
    /// (`keys[i] <= keys[i+1]` for all i in `[0, num_rows-1)`),
    /// `Ok(false)` if a violation is detected, `Err(_)` on
    /// kernel-launch / D2H failure.
    ///
    /// **Empty / single-row fast path** (per W4.3 plan iter-4 D1
    /// + F-W43-4): `n < 2` returns `Ok(true)` BEFORE allocation
    ///   or kernel launch. The detection kernel's grid `(n + 255)
    ///   / 256` is undefined for `n == 0`; single-row sequences
    ///   are trivially sorted. This is the load-bearing
    ///   invariant Cert G's empty-input subcases verify.
    ///
    /// Validation:
    ///   * Key column index within arity bounds.
    ///   * Key column type is `U32` or `Symbol`
    ///     (byte-identical at the kernel level).
    ///   * Key column allocation `>= num_rows * 4` bytes
    ///     (mirrors W4.2 F-W42-14 byte-length lower-bound idiom).
    ///
    /// **Caller surface (per W4.3 plan iter-6 F-W43-14)**: this
    /// fn has NO executor-dispatch caller after iteration-6
    /// unwiring — its only callers are operator-level certs
    /// (Cert G's empty-input subcases) and the Step 12
    /// production bench (sort-merge-with-detection Path 1
    /// timing). The provider returns the honest `Result<bool>`
    /// — the kernel can fail (allocation, launch, D2H), and
    /// `Err(_)` is preserved so callers can log or surface it
    /// at their abstraction level. There is no fail-closed
    /// dispatch contract anymore. Earlier fail-closed callers used
    /// `matches!(_, Ok(true))`; after the dispatch site was unwired,
    /// any later caller must decide its own Err-handling policy.
    pub fn is_sorted_ascending_u32(&self, buf: &CudaBuffer, key_col: usize) -> Result<bool> {
        // Empty / single-row fast path (per F-W43-4).
        let n = self.device_row_count(buf)?;
        if n < 2 {
            return Ok(true);
        }

        // Validate key column.
        if buf.arity() <= key_col {
            return Err(XlogError::Kernel(format!(
                "is_sorted_ascending_u32: key_col={} out of bounds (arity={})",
                key_col,
                buf.arity()
            )));
        }
        let kt = buf.schema().column_type(key_col);
        if !matches!(kt, Some(ScalarType::U32) | Some(ScalarType::Symbol)) {
            return Err(XlogError::Kernel(format!(
                "is_sorted_ascending_u32: key column must be U32 or Symbol; got {:?}",
                kt
            )));
        }
        let key_column = buf.column(key_col).ok_or_else(|| {
            XlogError::Kernel(format!(
                "is_sorted_ascending_u32: column({}) missing",
                key_col
            ))
        })?;
        let required_bytes = n
            .checked_mul(4)
            .ok_or_else(|| XlogError::Kernel("is_sorted_ascending_u32: byte overflow".into()))?;
        if key_column.num_bytes() < required_bytes {
            return Err(XlogError::Kernel(format!(
                "is_sorted_ascending_u32: key column has {} bytes; require at least {} ({} rows × 4)",
                key_column.num_bytes(),
                required_bytes,
                n
            )));
        }

        // Allocate result flag, initialize to 1 (sorted by
        // default; kernel atomically writes 0 only on
        // detected violation).
        let mut d_result = self.memory.alloc::<u32>(1)?;
        self.htod_launch_metadata_sync_copy_into(&[1u32], &mut d_result)
            .map_err(|e| {
                XlogError::Kernel(format!("is_sorted_ascending_u32: htod result init: {}", e))
            })?;

        // Launch detection kernel.
        let func = self
            .device
            .inner()
            .get_func(SORT_MODULE, sort_kernels::CHECK_ASCENDING_SORTED_U32)
            .ok_or_else(|| {
                XlogError::Kernel("check_ascending_sorted_u32 kernel not found".into())
            })?;
        let n_u32 = n as u32;
        let block_size = 256u32;
        let grid_size = n_u32.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: kernel signature
        //   check_ascending_sorted_u32(
        //     const uint32_t* keys, uint32_t num_rows,
        //     uint32_t* result)
        // Byte length validated above; `n` fits in u32 by
        // device_row_count's u32 underlying representation;
        // result allocation is 1 u32, initialized to 1.
        unsafe {
            func.clone()
                .launch(config, (key_column, n_u32, &mut d_result))
                .map_err(|e| {
                    XlogError::Kernel(format!("check_ascending_sorted_u32 launch: {}", e))
                })?;
        }

        self.device.synchronize()?;
        let result = self.dtoh_scalar_untracked(&d_result, 0)?;
        Ok(result == 1)
    }

    /// W4.3 sort-merge inner join (caller-asserted pre-sorted
    /// inputs). Drop-in compatible with `hash_join_v2(_, _,
    /// &[left_key], &[right_key], JoinType::Inner)`: same
    /// input types, same output schema
    /// (`combine_schemas(left, right)`), same row set.
    ///
    /// **Caller surface (per W4.3 plan iter-6 F-W43-14)**: this
    /// fn has NO executor-dispatch caller after iteration-6
    /// unwiring. Step 12 production evidence rejected D2 precedence
    /// and D7 #8 for the iteration-1–5 dispatch site at
    /// `execute_join`; this fn remains graduated operator work for
    /// direct provider callers and certs.
    /// Current callers: operator-level certs in
    /// `crates/xlog-integration/tests/test_w43_sort_merge_dispatch.rs`
    /// (Cert A/E/F/G provider parity tests) and the Step 12
    /// production bench at
    /// `crates/xlog-integration/benches/w43_production_sort_merge_bench.rs`
    /// (sort-merge-with-detection Path 1 timing).
    ///
    /// **Caller contract**: both inputs are pre-sorted ascending
    /// by their respective key column. The kernel does NOT
    /// detect or enforce sortedness; callers may pre-check via
    /// `is_sorted_ascending_u32`. On unsorted inputs the row-set
    /// output is undefined (operator-level UB per iter-6 D4 —
    /// the iteration-1–5 dispatch-site fall-back path no longer
    /// exists).
    ///
    /// # Eligibility (validated inside; `Err` on violation)
    ///
    /// * `left.arity() > left_key && right.arity() > right_key`.
    /// * Left and right key columns share the same `ScalarType`,
    ///   and that shared type is `U32` or `Symbol`.
    /// * Each key column's allocation is at least `num_rows * 4`
    ///   bytes (lower-bound check, mirrors W4.2 F-W42-14).
    /// * `num_left * num_right <= NESTED_LOOP_TOTAL_THRESHOLD`
    ///   (shared with W4.2 nested-loop per W4.3 D3; computed via
    ///   `checked_mul`; release-mode wrapping multiply is
    ///   forbidden).
    ///
    /// # Implementation outline
    ///
    /// Mirrors `nested_loop_join_v2_inner_u32_1key` literal
    /// idioms verbatim per W4.2 F-W42-13/14/15/16/17 (empty
    /// fast path with no `?`, byte-length lower-bound `<`
    /// check, `checked_mul` for threshold, `as u64` for
    /// `row_cap`, variant-agnostic `&CudaColumn` launch).
    ///
    /// 1. Read logical row counts via `device_row_count` (NOT
    ///    `row_cap`).
    /// 2. Empty-input fast path: if either side is empty,
    ///    return `create_empty_buffer(combine_schemas(...))` —
    ///    mirrors `hash_join_inner_v2` at `relational.rs:3165-3170`
    ///    AND `nested_loop_join_v2_inner_u32_1key`'s identical
    ///    pattern.
    /// 3. Validate eligibility (above).
    /// 4. Allocate two `u32` index arrays of length
    ///    `num_left * num_right` (bounded at 32 MB total under
    ///    the shared threshold).
    /// 5. Launch `sort_merge_join_inner_u32_1key_pairs` with
    ///    `&CudaColumn` key pointers (variant-agnostic per
    ///    W4.2 F-W42-11).
    /// 6. D2H the output count.
    /// 7. Materialize via `gather_buffer_by_indices` for both
    ///    sides + concatenate columns.
    pub fn sort_merge_join_v2_inner_u32_1key(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_key: usize,
        right_key: usize,
    ) -> Result<CudaBuffer> {
        // ----- 1. Logical row counts (NOT row_cap) -----
        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;

        // ----- 2. Empty-input fast path (inner-join schema) -----
        if num_left == 0 || num_right == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        // ----- 3. Eligibility validation -----
        if left.arity() <= left_key {
            return Err(XlogError::Kernel(format!(
                "sort_merge: left_key={} out of bounds (arity={})",
                left_key,
                left.arity()
            )));
        }
        if right.arity() <= right_key {
            return Err(XlogError::Kernel(format!(
                "sort_merge: right_key={} out of bounds (arity={})",
                right_key,
                right.arity()
            )));
        }
        let lt = left.schema().column_type(left_key);
        let rt = right.schema().column_type(right_key);
        if lt != rt || !matches!(lt, Some(ScalarType::U32) | Some(ScalarType::Symbol)) {
            return Err(XlogError::Kernel(format!(
                "sort_merge: key types must be equal U32/Symbol; got left={:?} right={:?}",
                lt, rt
            )));
        }
        let left_col = left
            .column(left_key)
            .ok_or_else(|| XlogError::Kernel(format!("sort_merge: left.column({})", left_key)))?;
        let right_col = right
            .column(right_key)
            .ok_or_else(|| XlogError::Kernel(format!("sort_merge: right.column({})", right_key)))?;
        let required_left_bytes = num_left
            .checked_mul(4)
            .ok_or_else(|| XlogError::Kernel("sort_merge: left byte overflow".into()))?;
        let required_right_bytes = num_right
            .checked_mul(4)
            .ok_or_else(|| XlogError::Kernel("sort_merge: right byte overflow".into()))?;
        if left_col.num_bytes() < required_left_bytes {
            return Err(XlogError::Kernel(format!(
                "sort_merge: left key column has {} bytes; \
                 require at least {} ({} rows × 4)",
                left_col.num_bytes(),
                required_left_bytes,
                num_left
            )));
        }
        if right_col.num_bytes() < required_right_bytes {
            return Err(XlogError::Kernel(format!(
                "sort_merge: right key column has {} bytes; \
                 require at least {} ({} rows × 4)",
                right_col.num_bytes(),
                required_right_bytes,
                num_right
            )));
        }

        // ----- 4. Fail-closed threshold check via checked_mul -----
        let upper_bound: u64 = (num_left as u64)
            .checked_mul(num_right as u64)
            .ok_or_else(|| XlogError::Kernel("sort_merge: row-count product overflow".into()))?;
        if upper_bound > NESTED_LOOP_TOTAL_THRESHOLD {
            return Err(XlogError::Kernel(format!(
                "sort_merge: caller violated eligibility threshold: \
                 num_left * num_right = {} > {} (NESTED_LOOP_TOTAL_THRESHOLD)",
                upper_bound, NESTED_LOOP_TOTAL_THRESHOLD
            )));
        }

        // ----- 5. Allocate index arrays + counter -----
        let upper_bound_usize = upper_bound as usize;
        let mut d_output_left_idx = self.memory.alloc::<u32>(upper_bound_usize)?;
        let mut d_output_right_idx = self.memory.alloc::<u32>(upper_bound_usize)?;
        let mut d_output_count = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .memset_zeros(&mut d_output_count)
            .map_err(|e| XlogError::Kernel(format!("sort_merge: counter zero: {}", e)))?;

        // ----- 6. Launch kernel (variant-agnostic column refs) -----
        let func = self
            .device
            .inner()
            .get_func(
                JOIN_MODULE,
                join_kernels::SORT_MERGE_JOIN_INNER_U32_1KEY_PAIRS,
            )
            .ok_or_else(|| {
                XlogError::Kernel("sort_merge_join_inner_u32_1key_pairs kernel not found".into())
            })?;

        let num_left_u32 = num_left as u32;
        let num_right_u32 = num_right as u32;
        let upper_bound_u32 = upper_bound as u32;
        let block_size = 256u32;
        let grid_size = num_left_u32.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: kernel signature matches PTX:
        //   sort_merge_join_inner_u32_1key_pairs(
        //     const uint32_t* left_keys (sorted ascending),
        //     const uint32_t* right_keys (sorted ascending),
        //     uint32_t num_left, uint32_t num_right,
        //     uint32_t* output_left_idx, uint32_t* output_right_idx,
        //     uint32_t* output_count, uint32_t output_capacity)
        // Byte length validated above; sortedness is a caller-
        // supplied invariant (per iter-6 D4 — no dispatch-site
        // pre-check exists after F-W43-14 unwiring); counts fit
        // in u32 by caller-supplied input-size bound; allocations
        // sized to upper_bound; counter pre-zeroed.
        unsafe {
            func.clone()
                .launch(
                    config,
                    (
                        left_col,
                        right_col,
                        num_left_u32,
                        num_right_u32,
                        &mut d_output_left_idx,
                        &mut d_output_right_idx,
                        &mut d_output_count,
                        upper_bound_u32,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("sort_merge launch: {}", e)))?;
        }

        self.device.synchronize()?;

        // ----- 7. D2H the output count (single u32) -----
        let output_rows = self.dtoh_scalar_untracked(&d_output_count, 0)?;
        // Defense-in-depth: kernel's atomic-cap branch ensures
        // output_rows ≤ upper_bound, but double-check.
        if (output_rows as u64) > upper_bound {
            return Err(XlogError::Kernel(format!(
                "sort_merge: kernel reported {} output rows > upper_bound {}",
                output_rows, upper_bound
            )));
        }

        // ----- 8. Gather both sides via existing GPU machinery -----
        let gathered_left = self.gather_buffer_by_indices(left, &d_output_left_idx, output_rows)?;
        let gathered_right =
            self.gather_buffer_by_indices(right, &d_output_right_idx, output_rows)?;

        // ----- 9. Combine columns + return drop-in result -----
        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        result_columns.extend(gathered_left.columns);
        result_columns.extend(gathered_right.columns);
        self.buffer_from_columns(result_columns, output_rows as u64, combined_schema)
    }

    /// Sorted-chain variant of [`Self::sort_merge_join_v2_inner_u32_1key`].
    ///
    /// The W4.3 operator is product-thresholded because it allocates
    /// `|left| * |right|` candidate pairs. W6.3 chain routing uses this
    /// bounded variant only for sorted large inputs where the expected
    /// fanout is one-to-one; capacity is caller supplied and the kernel's
    /// logical output counter is checked after launch. If duplicates make
    /// the true output exceed `output_capacity`, this returns an error so
    /// the caller can fail closed to the hash fallback.
    pub fn sort_merge_join_v2_inner_u32_1key_bounded(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_key: usize,
        right_key: usize,
        output_capacity: usize,
    ) -> Result<CudaBuffer> {
        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;

        if num_left == 0 || num_right == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "sort_merge_bounded: row counts exceed u32 surface: left={} right={}",
                num_left, num_right
            )));
        }
        if output_capacity == 0 || output_capacity > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "sort_merge_bounded: invalid output capacity {}",
                output_capacity
            )));
        }
        if left.arity() <= left_key {
            return Err(XlogError::Kernel(format!(
                "sort_merge_bounded: left_key={} out of bounds (arity={})",
                left_key,
                left.arity()
            )));
        }
        if right.arity() <= right_key {
            return Err(XlogError::Kernel(format!(
                "sort_merge_bounded: right_key={} out of bounds (arity={})",
                right_key,
                right.arity()
            )));
        }
        let lt = left.schema().column_type(left_key);
        let rt = right.schema().column_type(right_key);
        if lt != rt || !matches!(lt, Some(ScalarType::U32) | Some(ScalarType::Symbol)) {
            return Err(XlogError::Kernel(format!(
                "sort_merge_bounded: key types must be equal U32/Symbol; got left={:?} right={:?}",
                lt, rt
            )));
        }

        let left_col = left.column(left_key).ok_or_else(|| {
            XlogError::Kernel(format!("sort_merge_bounded: left.column({})", left_key))
        })?;
        let right_col = right.column(right_key).ok_or_else(|| {
            XlogError::Kernel(format!("sort_merge_bounded: right.column({})", right_key))
        })?;
        let required_left_bytes = num_left
            .checked_mul(4)
            .ok_or_else(|| XlogError::Kernel("sort_merge_bounded: left byte overflow".into()))?;
        let required_right_bytes = num_right
            .checked_mul(4)
            .ok_or_else(|| XlogError::Kernel("sort_merge_bounded: right byte overflow".into()))?;
        if left_col.num_bytes() < required_left_bytes {
            return Err(XlogError::Kernel(format!(
                "sort_merge_bounded: left key column has {} bytes; require at least {}",
                left_col.num_bytes(),
                required_left_bytes
            )));
        }
        if right_col.num_bytes() < required_right_bytes {
            return Err(XlogError::Kernel(format!(
                "sort_merge_bounded: right key column has {} bytes; require at least {}",
                right_col.num_bytes(),
                required_right_bytes
            )));
        }

        let mut d_output_left_idx = self.memory.alloc::<u32>(output_capacity)?;
        let mut d_output_right_idx = self.memory.alloc::<u32>(output_capacity)?;
        let mut d_output_count = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .memset_zeros(&mut d_output_count)
            .map_err(|e| XlogError::Kernel(format!("sort_merge_bounded: counter zero: {}", e)))?;

        let func = self
            .device
            .inner()
            .get_func(
                JOIN_MODULE,
                join_kernels::SORT_MERGE_JOIN_INNER_U32_1KEY_PAIRS,
            )
            .ok_or_else(|| {
                XlogError::Kernel("sort_merge_join_inner_u32_1key_pairs kernel not found".into())
            })?;

        let num_left_u32 = num_left as u32;
        let num_right_u32 = num_right as u32;
        let output_capacity_u32 = output_capacity as u32;
        let block_size = 256u32;
        let grid_size = num_left_u32.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.clone()
                .launch(
                    config,
                    (
                        left_col,
                        right_col,
                        num_left_u32,
                        num_right_u32,
                        &mut d_output_left_idx,
                        &mut d_output_right_idx,
                        &mut d_output_count,
                        output_capacity_u32,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("sort_merge_bounded launch: {}", e)))?;
        }

        self.device.synchronize()?;
        let output_rows = self.dtoh_scalar_untracked(&d_output_count, 0)?;
        if output_rows as usize > output_capacity {
            return Err(XlogError::Kernel(format!(
                "sort_merge_bounded: output {} exceeded bounded capacity {}",
                output_rows, output_capacity
            )));
        }

        let gathered_left = self.gather_buffer_by_indices(left, &d_output_left_idx, output_rows)?;
        let gathered_right =
            self.gather_buffer_by_indices(right, &d_output_right_idx, output_rows)?;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        result_columns.extend(gathered_left.columns);
        result_columns.extend(gathered_right.columns);
        self.buffer_from_columns(result_columns, output_rows as u64, combined_schema)
    }

    /// Build a cached join index for the right/build side of v2 hash join.
    pub fn build_join_index_v2(
        &self,
        right: &CudaBuffer,
        right_keys: &[usize],
    ) -> Result<JoinIndexV2> {
        let num_right = self.device_row_count(right)?;
        if num_right == 0 {
            return Err(XlogError::Kernel(
                "Cannot build join index for empty relation".to_string(),
            ));
        }
        if num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join index supports at most {} rows, got {}",
                u32::MAX,
                num_right
            )));
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

        let num_right = num_right as u32;
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

    /// Build a cached join index for background persistent-index mode.
    ///
    /// When recorded hash joins are enabled and the provider has a runtime-backed
    /// manager, the build is enqueued on the provider's recorded operation stream
    /// and dependency-recorded like the indexed join consumer path. Otherwise this
    /// falls back to the legacy synchronous builder.
    pub fn build_join_index_v2_background(
        &self,
        right: &CudaBuffer,
        right_keys: &[usize],
    ) -> Result<JoinIndexV2> {
        if Self::use_recorded_hash_join_env()
            && !right_keys.is_empty()
            && right_keys.len() <= 4
            && right.num_rows() > 0
        {
            if let Some(launch_stream) = self.recorded_op_stream_or_init() {
                return self.build_join_index_v2_recorded(right, right_keys, launch_stream);
            }
        }

        self.build_join_index_v2(right, right_keys)
    }

    /// Recorded-stream join-index builder used by persistent background builds.
    ///
    /// The build side is packed and bucketized on `launch_stream`; the returned
    /// `JoinIndexV2` carries runtime-tracked buffers whose writes were committed
    /// through the launch recorder / stream dependency machinery.
    pub fn build_join_index_v2_recorded(
        &self,
        right: &CudaBuffer,
        right_keys: &[usize],
        launch_stream: StreamId,
    ) -> Result<JoinIndexV2> {
        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "build_join_index_v2_recorded requires a runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "build_join_index_v2_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        let num_right = self.device_row_count(right)?;
        if num_right == 0 {
            return Err(XlogError::Kernel(
                "Cannot build join index for empty relation".to_string(),
            ));
        }
        if num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join index supports at most {} rows, got {}",
                u32::MAX,
                num_right
            )));
        }
        if right_keys.is_empty() {
            return Err(XlogError::Kernel(
                "Join requires at least one key column".to_string(),
            ));
        }
        if right_keys.len() > 4 {
            return Err(XlogError::Kernel(
                "build_join_index_v2_recorded: max 4 key columns supported".to_string(),
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

        let num_right = num_right as u32;
        let right_packed =
            self.pack_keys_gpu_on_stream(right, right_keys, &cu_stream, launch_stream, runtime)?;
        let table = self.build_hash_table_v2_on_stream(
            &right_packed.hashes,
            num_right,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

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
    #[allow(clippy::too_many_arguments)]
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
        // Env-gated recorded dispatch. Same `≤4 key column`
        // constraint as the non-indexed variant.
        if Self::use_recorded_hash_join_env()
            && !left_keys.is_empty()
            && left_keys.len() == right_keys.len()
            && left_keys.len() <= 4
        {
            if let Some(launch_stream) = self.recorded_op_stream_or_init() {
                return self.hash_join_v2_with_index_recorded(
                    left,
                    right,
                    left_keys,
                    right_keys,
                    join_type,
                    index,
                    max_output,
                    launch_stream,
                );
            }
        }
        let left_rows = self.device_row_count(left)?;
        let right_rows = self.device_row_count(right)?;
        if left_rows > u32::MAX as usize || right_rows > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                left_rows,
                right_rows
            )));
        }

        // Handle empty inputs early.
        if left_rows == 0 {
            return match join_type {
                JoinType::Inner | JoinType::LeftOuter => {
                    let combined_schema = self.combine_schemas(left.schema(), right.schema());
                    self.create_empty_buffer(combined_schema)
                }
                JoinType::Semi | JoinType::Anti => self.create_empty_buffer(left.schema().clone()),
            };
        }
        if right_rows == 0 {
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
        if index.right_num_rows != right_rows as u32 {
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
            JoinType::Inner => {
                self.hash_join_inner_v2_indexed(left, right, left_keys, index, max_output)
            }
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

        let num_rows = self.device_row_count(buffer)?;
        if num_rows > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "pack_keys_gpu supports at most {} rows, got {}",
                u32::MAX,
                num_rows
            )));
        }
        let num_rows = num_rows as u32;
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

        // Get column device pointers as u64 values for the kernel
        // The kernel expects raw pointers as u64
        let mut col_ptrs: [u64; 4] = [0; 4];
        for (i, &col_idx) in key_cols.iter().enumerate() {
            let col = buffer
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Key column {} not found", col_idx)))?;
            // Get the device pointer as a raw u64 value
            col_ptrs[i] = *col.device_ptr();
        }
        let mut packed_col_sizes = 0u64;
        for (i, size) in col_sizes.iter().copied().enumerate() {
            if size > u16::MAX as u32 {
                return Err(XlogError::Kernel(format!(
                    "pack_keys_gpu: column element size {} exceeds 16-bit kernel argument",
                    size
                )));
            }
            packed_col_sizes |= (size as u64) << (i * 16);
        }

        // Get the kernel function
        let func = self
            .device
            .inner()
            .get_func(PACK_MODULE, pack_kernels::PACK_AND_HASH_KEYS)
            .ok_or_else(|| XlogError::Kernel("pack_and_hash_keys kernel not found".to_string()))?;

        // Launch configuration
        let block_size = 256u32;
        let grid_size = num_rows.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch the fused pack+hash kernel
        // SAFETY: Kernel signature matches pack_and_hash_keys in pack.cu:
        // pack_and_hash_keys(col0, col1, col2, col3, packed_col_sizes, num_cols, num_rows, row_size, packed_output, hashes)
        // Column pointers are passed as CudaSlice references - the kernel sees raw device pointers.
        // We pass column data as raw pointers cast to u8* in the kernel.
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone()
                .launch(
                    config,
                    (
                        col_ptrs[0],
                        col_ptrs[1],
                        col_ptrs[2],
                        col_ptrs[3],
                        packed_col_sizes,
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

    /// Pack key columns on GPU and compute hashes for arbitrary column counts.
    fn pack_keys_gpu_generic(
        &self,
        buffer: &CudaBuffer,
        key_cols: &[usize],
    ) -> Result<PackedKeyData> {
        if key_cols.is_empty() {
            return Err(XlogError::Kernel(
                "pack_keys_gpu_generic: no key columns specified".into(),
            ));
        }

        let num_rows = self.device_row_count(buffer)?;
        if num_rows > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "pack_keys_gpu_generic supports at most {} rows, got {}",
                u32::MAX,
                num_rows
            )));
        }
        let num_rows = num_rows as u32;
        if num_rows == 0 {
            return Ok(PackedKeyData {
                hashes: self.memory.alloc::<u64>(0)?,
                packed_keys: self.memory.alloc::<u8>(0)?,
                key_bytes: 0,
            });
        }

        let mut col_sizes: Vec<u32> = Vec::with_capacity(key_cols.len());
        let mut col_ptrs: Vec<u64> = Vec::with_capacity(key_cols.len());
        let mut row_size: u32 = 0;

        for &col_idx in key_cols {
            let col_type = buffer
                .schema()
                .column_type(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Invalid column index: {}", col_idx)))?;
            let size = col_type.size_bytes() as u32;
            row_size = row_size
                .checked_add(size)
                .ok_or_else(|| XlogError::Kernel("Row size overflow".to_string()))?;
            col_sizes.push(size);

            let col = buffer
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Key column {} not found", col_idx)))?;
            col_ptrs.push(*col.device_ptr());
        }

        let packed_bytes = (num_rows as u64)
            .checked_mul(row_size as u64)
            .ok_or_else(|| XlogError::Kernel("Packed key byte size overflow".to_string()))?;
        let packed_slice = self.memory.alloc::<u8>(packed_bytes as usize)?;
        let hash_slice = self.memory.alloc::<u64>(num_rows as usize)?;

        let mut d_col_sizes = self.memory.alloc::<u32>(col_sizes.len())?;
        self.htod_sync_copy_into_tracked(&col_sizes, &mut d_col_sizes)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload col_sizes: {}", e)))?;

        let mut d_col_ptrs = self.memory.alloc::<u64>(col_ptrs.len())?;
        self.htod_sync_copy_into_tracked(&col_ptrs, &mut d_col_ptrs)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload col_ptrs: {}", e)))?;

        let func = self
            .device
            .inner()
            .get_func(PACK_MODULE, pack_kernels::PACK_AND_HASH_KEYS_GENERIC)
            .ok_or_else(|| {
                XlogError::Kernel("pack_and_hash_keys_generic kernel not found".to_string())
            })?;

        let block_size = 256u32;
        let grid_size = num_rows.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone()
                .launch(
                    config,
                    (
                        &d_col_ptrs,
                        &d_col_sizes,
                        key_cols.len() as u32,
                        num_rows,
                        row_size,
                        &packed_slice,
                        &hash_slice,
                    ),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("pack_and_hash_keys_generic launch failed: {}", e))
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
    pub(super) fn compute_hashes_and_pack_keys(
        &self,
        buffer: &CudaBuffer,
        key_cols: &[usize],
    ) -> Result<PackedKeyData> {
        if key_cols.is_empty() {
            return Err(XlogError::Kernel(
                "compute_hashes_and_pack_keys: no key columns specified".to_string(),
            ));
        }

        if key_cols.len() <= 4 {
            self.pack_keys_gpu(buffer, key_cols)
        } else {
            self.pack_keys_gpu_generic(buffer, key_cols)
        }
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
            self.device.synchronize()?;
        }

        let block_size = 256u32;
        let grid_size = num_rows.div_ceil(block_size);
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
        self.device.synchronize()?;

        // bucket_offsets = exclusive scan(bucket_counts)
        let mut bucket_offsets = self.memory.alloc::<u32>(num_buckets as usize)?;
        if num_buckets > 0 {
            device
                .dtod_copy(&bucket_counts, &mut bucket_offsets)
                .map_err(|e| XlogError::Kernel(format!("Failed to copy bucket_counts: {}", e)))?;
            self.device.synchronize()?;
            self.multiblock_scan_u32_inplace(&mut bucket_offsets, num_buckets)?;
            self.device.synchronize()?;
        }

        // bucket_cursors = bucket_offsets (then atomically incremented during scatter)
        let mut bucket_cursors = self.memory.alloc::<u32>(num_buckets as usize)?;
        if num_buckets > 0 {
            device
                .dtod_copy(&bucket_offsets, &mut bucket_cursors)
                .map_err(|e| XlogError::Kernel(format!("Failed to copy bucket_offsets: {}", e)))?;
            self.device.synchronize()?;
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

    /// Build a bucketed hash table from a u64 hash array.
    pub fn build_hash_table_u64(
        &self,
        hashes: &crate::memory::TrackedCudaSlice<u64>,
        num_rows: u32,
    ) -> Result<HashTableU64> {
        let JoinHashTableV2 {
            bucket_counts,
            bucket_offsets,
            bucket_entries,
            bucket_entry_hashes,
            bucket_mask,
        } = self.build_hash_table_v2(hashes, num_rows)?;
        Ok(HashTableU64 {
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
        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }

        // Handle empty inputs
        if num_left == 0 || num_right == 0 {
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

        let num_left = num_left as u32;
        let num_right = num_right as u32;

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
        let probe_grid = num_left.div_ceil(block_size);
        let probe_config = LaunchConfig {
            grid_dim: (probe_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut d_count_only = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .memset_zeros(&mut d_count_only)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero output count: {}", e)))?;
        self.device.synchronize()?;
        let d_dummy_left = self.memory.alloc::<u32>(1)?;
        let d_dummy_right = self.memory.alloc::<u32>(1)?;
        let max_output_count_only = 0u32;

        // SAFETY: hash_join_probe_v2(probe_hashes, num_probe,
        //                            bucket_offsets, bucket_counts, bucket_entries, bucket_entry_hashes, bucket_mask,
        //                            probe_keys, build_keys, key_bytes,
        //                            output_left, output_right, output_count, max_output)
        // Note: Using raw pointer launch because tuple exceeds 12-element limit
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&right_packed.packed_keys).as_kernel_param(),
                left_packed.key_bytes.as_kernel_param(),
                (&d_dummy_left).as_kernel_param(),
                (&d_dummy_right).as_kernel_param(),
                (&d_count_only).as_kernel_param(),
                max_output_count_only.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(probe_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("hash_join_probe_v2 (count) failed: {}", e))
                })?;
        }

        self.device.synchronize()?;

        // Metadata read: this u32 is the join's count-only result, used to
        // size the next allocation. See `read_join_output_count_metadata`
        // for the metadata-vs-data-plane rationale.
        let full_count = self.read_join_output_count_metadata(&d_count_only)? as u64;
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
        let mut d_output_count = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .memset_zeros(&mut d_output_count)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero output count: {}", e)))?;
        self.device.synchronize()?;

        // SAFETY: hash_join_probe_v2(probe_hashes, num_probe,
        //                            bucket_offsets, bucket_counts, bucket_entries, bucket_entry_hashes, bucket_mask,
        //                            probe_keys, build_keys, key_bytes,
        //                            output_left, output_right, output_count, max_output)
        // Note: Using raw pointer launch because tuple exceeds 12-element limit
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&right_packed.packed_keys).as_kernel_param(),
                left_packed.key_bytes.as_kernel_param(),
                (&d_output_left).as_kernel_param(),
                (&d_output_right).as_kernel_param(),
                (&d_output_count).as_kernel_param(),
                max_output.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(probe_config, &mut params)
                .map_err(|e| XlogError::Kernel(format!("hash_join_probe_v2 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        // Metadata read: post-materialize device-side atomic count.
        // Used as the result buffer's logical row count after clamping
        // to the host-allocated upper bound. See
        // `read_join_output_count_metadata` for the rationale.
        // Clamp to max_output to prevent buffer overflow (kernel atomically
        // increments before bounds check, so count can exceed max_output).
        let result_count =
            (self.read_join_output_count_metadata(&d_output_count)? as u64).min(max_output as u64);

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
        result_columns.extend(gathered_left.columns);
        result_columns.extend(gathered_right.columns);

        self.buffer_from_columns(result_columns, result_count, combined_schema)
    }

    fn hash_join_inner_v2_indexed(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        index: &JoinIndexV2,
        max_output: Option<usize>,
    ) -> Result<CudaBuffer> {
        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }

        // Handle empty inputs.
        if num_left == 0 || num_right == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        let num_left = num_left as u32;

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
        let probe_grid = num_left.div_ceil(block_size);
        let probe_config = LaunchConfig {
            grid_dim: (probe_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut d_count_only = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .memset_zeros(&mut d_count_only)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero output count: {}", e)))?;
        self.device.synchronize()?;
        let d_dummy_left = self.memory.alloc::<u32>(1)?;
        let d_dummy_right = self.memory.alloc::<u32>(1)?;
        let max_output_count_only = 0u32;

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&index.packed_keys).as_kernel_param(),
                index.key_bytes.as_kernel_param(),
                (&d_dummy_left).as_kernel_param(),
                (&d_dummy_right).as_kernel_param(),
                (&d_count_only).as_kernel_param(),
                max_output_count_only.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(probe_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("hash_join_probe_v2 (count) failed: {}", e))
                })?;
        }

        self.device.synchronize()?;

        // Metadata read: this u32 is the join's count-only result, used to
        // size the next allocation. See `read_join_output_count_metadata`
        // for the metadata-vs-data-plane rationale.
        let full_count = self.read_join_output_count_metadata(&d_count_only)? as u64;
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
        let mut d_output_count = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .memset_zeros(&mut d_output_count)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero output count: {}", e)))?;
        self.device.synchronize()?;

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&index.packed_keys).as_kernel_param(),
                index.key_bytes.as_kernel_param(),
                (&d_output_left).as_kernel_param(),
                (&d_output_right).as_kernel_param(),
                (&d_output_count).as_kernel_param(),
                max_output.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(probe_config, &mut params)
                .map_err(|e| XlogError::Kernel(format!("hash_join_probe_v2 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        // Metadata read: post-materialize device-side atomic count, used
        // as the result buffer's logical row count after clamping.
        let result_count =
            (self.read_join_output_count_metadata(&d_output_count)? as u64).min(max_output as u64);

        if result_count == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        let output_rows = result_count as u32;

        let gathered_left = self.gather_buffer_by_indices(left, &d_output_left, output_rows)?;
        let gathered_right = self.gather_buffer_by_indices(right, &d_output_right, output_rows)?;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        result_columns.extend(gathered_left.columns);
        result_columns.extend(gathered_right.columns);

        self.buffer_from_columns(result_columns, result_count, combined_schema)
    }

    /// Semi-join implementation: return left rows that have matches in right
    fn hash_join_semi_impl(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
    ) -> Result<CudaBuffer> {
        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }

        // Handle empty inputs
        if num_left == 0 {
            return self.create_empty_buffer(left.schema().clone());
        }
        if num_right == 0 {
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

        let num_left = num_left as u32;
        let num_right = num_right as u32;

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
        let grid_size = num_left.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: hash_join_semi(probe_hashes, num_probe,
        //                        bucket_offsets, bucket_counts, bucket_entries, bucket_entry_hashes, bucket_mask,
        //                        probe_keys, build_keys, key_bytes, has_match)
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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

        self.device.synchronize()?;
        self.filter_by_device_mask(left, &d_has_match)
    }

    /// Compute a per-row membership mask on device: for each row in `probe`,
    /// check whether a matching row exists in `build` (by the specified key
    /// columns).  Returns a `TrackedCudaSlice<u8>` of length = probe row count
    /// that stays GPU-resident (no D2H transfer).
    pub fn membership_mask_device(
        &self,
        probe: &CudaBuffer,
        build: &CudaBuffer,
        probe_keys: &[usize],
        build_keys: &[usize],
    ) -> Result<TrackedCudaSlice<u8>> {
        let num_probe = self.device_row_count(probe)?;
        let num_build = self.device_row_count(build)?;

        // Edge case: empty probe → empty device allocation
        if num_probe == 0 {
            return self.memory.alloc::<u8>(0);
        }

        // Edge case: empty build → no matches possible, return zeroed mask
        if num_build == 0 {
            let mut d_mask = self.memory.alloc::<u8>(num_probe)?;
            self.device.inner().memset_zeros(&mut d_mask).map_err(|e| {
                XlogError::Kernel(format!(
                    "Failed to zero membership mask for empty build: {}",
                    e
                ))
            })?;
            return Ok(d_mask);
        }

        if num_probe > u32::MAX as usize || num_build > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "membership_mask supports at most {} rows per side (probe={}, build={})",
                u32::MAX,
                num_probe,
                num_build
            )));
        }

        // Validate key columns
        if probe_keys.is_empty() || build_keys.is_empty() {
            return Err(XlogError::Kernel(
                "membership_mask requires at least one key column".to_string(),
            ));
        }
        if probe_keys.len() != build_keys.len() {
            return Err(XlogError::Kernel(
                "Probe and build key columns must have same length".to_string(),
            ));
        }

        // Validate key column types match
        for (&p_idx, &b_idx) in probe_keys.iter().zip(build_keys.iter()) {
            let p_type = probe.schema().column_type(p_idx);
            let b_type = build.schema().column_type(b_idx);
            if p_type != b_type {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: probe[{}]={:?}, build[{}]={:?}",
                    p_idx, p_type, b_idx, b_type
                )));
            }
        }

        let num_probe_u32 = num_probe as u32;
        let num_build_u32 = num_build as u32;

        // Compute composite hashes and pack keys for both sides
        let probe_packed = self.compute_hashes_and_pack_keys(probe, probe_keys)?;
        let build_packed = self.compute_hashes_and_pack_keys(build, build_keys)?;

        // Build hash table from build side
        let table = self.build_hash_table_v2(&build_packed.hashes, num_build_u32)?;

        // Allocate output mask on device
        let d_has_match = self.memory.alloc::<u8>(num_probe)?;

        // Launch semi-join kernel to populate the mask
        let semi_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_SEMI)
            .ok_or_else(|| XlogError::Kernel("hash_join_semi kernel not found".to_string()))?;

        let block_size = 256u32;
        let grid_size = num_probe_u32.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: hash_join_semi(probe_hashes, num_probe,
        //                        bucket_offsets, bucket_counts, bucket_entries,
        //                        bucket_entry_hashes, bucket_mask,
        //                        probe_keys, build_keys, key_bytes, has_match)
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            semi_func
                .clone()
                .launch(
                    config,
                    (
                        &probe_packed.hashes,
                        num_probe_u32,
                        &table.bucket_offsets,
                        &table.bucket_counts,
                        &table.bucket_entries,
                        &table.bucket_entry_hashes,
                        table.bucket_mask,
                        &probe_packed.packed_keys,
                        &build_packed.packed_keys,
                        probe_packed.key_bytes,
                        &d_has_match,
                    ),
                )
                .map_err(|e| XlogError::Kernel(format!("hash_join_semi failed: {}", e)))?;
        }

        Ok(d_has_match)
    }

    /// Compute a per-row membership mask: for each row in `probe`, check whether
    /// a matching row exists in `build` (by the specified key columns).
    /// Returns a `Vec<bool>` of length = probe row count.
    /// This downloads only num_probe bytes (the mask), NOT column data.
    pub fn membership_mask(
        &self,
        probe: &CudaBuffer,
        build: &CudaBuffer,
        probe_keys: &[usize],
        build_keys: &[usize],
    ) -> Result<Vec<bool>> {
        let d_has_match = self.membership_mask_device(probe, build, probe_keys, build_keys)?;
        let num_probe = d_has_match.len();
        if num_probe == 0 {
            return Ok(Vec::new());
        }
        let mut host_mask = vec![0u8; num_probe];
        self.device
            .inner()
            .dtoh_sync_copy_into(&d_has_match, &mut host_mask)
            .map_err(|e| XlogError::Kernel(format!("Failed to download membership mask: {}", e)))?;
        Ok(host_mask.into_iter().map(|b| b != 0).collect())
    }

    fn hash_join_semi_indexed(
        &self,
        left: &CudaBuffer,
        left_keys: &[usize],
        index: &JoinIndexV2,
    ) -> Result<CudaBuffer> {
        let num_left = self.device_row_count(left)?;
        if num_left > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows on left side (left={})",
                u32::MAX,
                num_left
            )));
        }

        // Handle empty inputs.
        if num_left == 0 {
            return self.create_empty_buffer(left.schema().clone());
        }
        if index.right_num_rows == 0 {
            return self.create_empty_buffer(left.schema().clone());
        }

        let num_left = num_left as u32;

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
        let grid_size = num_left.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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

        self.device.synchronize()?;
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
        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }

        // Handle empty inputs
        if num_left == 0 {
            return self.create_empty_buffer(left.schema().clone());
        }
        if num_right == 0 {
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

        let num_left = num_left as u32;
        let num_right = num_right as u32;

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
        let grid_size = num_left.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: hash_join_anti(probe_hashes, num_probe,
        //                        bucket_offsets, bucket_counts, bucket_entries, bucket_entry_hashes, bucket_mask,
        //                        probe_keys, build_keys, key_bytes, no_match)
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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

        self.device.synchronize()?;
        self.filter_by_device_mask(left, &d_no_match)
    }

    fn hash_join_anti_indexed(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        index: &JoinIndexV2,
    ) -> Result<CudaBuffer> {
        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }
        if num_left == 0 {
            return self.create_empty_buffer(left.schema().clone());
        }
        if num_right == 0 {
            return self.clone_buffer(left);
        }

        let num_left = num_left as u32;

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
        let grid_size = num_left.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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

        self.device.synchronize()?;
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
        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }

        // Handle empty left - return empty with combined schema.
        if num_left == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }
        // Handle empty right - return left rows with null right columns.
        if num_right == 0 {
            return self.left_outer_with_nulls(left, right);
        }

        let num_left = num_left as u32;

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
        let grid_size = num_left.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
        let mut d_count_only = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .memset_zeros(&mut d_count_only)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero output count: {}", e)))?;
        let d_dummy_left = self.memory.alloc::<u32>(1)?;
        let d_dummy_right = self.memory.alloc::<u32>(1)?;
        let max_output_count_only = 0u32;

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&index.packed_keys).as_kernel_param(),
                index.key_bytes.as_kernel_param(),
                (&d_dummy_left).as_kernel_param(),
                (&d_dummy_right).as_kernel_param(),
                (&d_count_only).as_kernel_param(),
                max_output_count_only.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("hash_join_probe_v2 (count) failed: {}", e))
                })?;
        }

        self.device.synchronize()?;

        // Metadata read: this u32 is the join's count-only result, used
        // to size the next allocation.
        let full_inner = self.read_join_output_count_metadata(&d_count_only)? as u64;
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
        let mut d_output_count = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .memset_zeros(&mut d_output_count)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero output count: {}", e)))?;

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&index.packed_keys).as_kernel_param(),
                index.key_bytes.as_kernel_param(),
                (&d_output_left).as_kernel_param(),
                (&d_output_right).as_kernel_param(),
                (&d_output_count).as_kernel_param(),
                max_output.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(config, &mut params)
                .map_err(|e| XlogError::Kernel(format!("hash_join_probe_v2 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        let device = self.device.inner();

        // Metadata read: post-materialize device-side atomic count, used
        // as the inner-join result's logical row count after clamping.
        // Clamp to max_output to prevent buffer overflow (kernel atomically
        // increments before bounds check, so count can exceed max_output).
        let inner_count = self
            .read_join_output_count_metadata(&d_output_count)?
            .min(max_output);

        let mask_not_fn = device
            .get_func(FILTER_MODULE, filter_kernels::MASK_NOT)
            .ok_or_else(|| XlogError::Kernel("mask_not kernel not found".to_string()))?;

        let mut d_no_match = self.memory.alloc::<u8>(num_left as usize)?;

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            mask_not_fn
                .clone()
                .launch(config, (&d_has_match, &mut d_no_match, num_left))
        }
        .map_err(|e| XlogError::Kernel(format!("mask_not failed: {}", e)))?;

        let unmatched_left = self.filter_by_device_mask(left, &d_no_match)?;

        let unmatched_rows = self.device_row_count(&unmatched_left)? as u64;
        let total_rows = (inner_count as u64) + unmatched_rows;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());

        if total_rows == 0 {
            return self.create_empty_buffer(combined_schema);
        }

        let inner_left = self.gather_buffer_by_indices(left, &d_output_left, inner_count)?;
        let inner_right = self.gather_buffer_by_indices(right, &d_output_right, inner_count)?;

        if unmatched_rows == 0 {
            let mut result_columns = Vec::with_capacity(combined_schema.arity());
            result_columns.extend(inner_left.columns);
            result_columns.extend(inner_right.columns);
            return self.buffer_from_columns(result_columns, inner_count as u64, combined_schema);
        }

        if inner_count == 0 {
            let mut result_columns = Vec::with_capacity(combined_schema.arity());
            result_columns.extend(unmatched_left.columns);

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
            return self.buffer_from_columns(result_columns, unmatched_rows, combined_schema);
        }

        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        let inner_rows = inner_count as u64;

        for (col_idx, (inner_col, unmatched_col)) in inner_left
            .columns
            .into_iter()
            .zip(unmatched_left.columns)
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
                let unmatched_view = self.column_bytes_view(&unmatched_col, unmatched_bytes)?;
                device
                    .dtod_copy(&unmatched_view, &mut out_view)
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

        self.buffer_from_columns(result_columns, total_rows, combined_schema)
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
        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }

        // Handle empty left - return empty with combined schema
        if num_left == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }

        // Handle empty right - return left rows with null right columns
        if num_right == 0 {
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

        let num_left = num_left as u32;
        let num_right = num_right as u32;

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
        let grid_size = num_left.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: hash_join_semi(probe_hashes, num_probe,
        //                        bucket_offsets, bucket_counts, bucket_entries, bucket_entry_hashes, bucket_mask,
        //                        probe_keys, build_keys, key_bytes, has_match)
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
        let mut d_count_only = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .memset_zeros(&mut d_count_only)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero output count: {}", e)))?;
        let d_dummy_left = self.memory.alloc::<u32>(1)?;
        let d_dummy_right = self.memory.alloc::<u32>(1)?;
        let max_output_count_only = 0u32;

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&right_packed.packed_keys).as_kernel_param(),
                left_packed.key_bytes.as_kernel_param(),
                (&d_dummy_left).as_kernel_param(),
                (&d_dummy_right).as_kernel_param(),
                (&d_count_only).as_kernel_param(),
                max_output_count_only.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("hash_join_probe_v2 (count) failed: {}", e))
                })?;
        }

        self.device.synchronize()?;

        // Metadata read: this u32 is the join's count-only result, used
        // to size the next allocation.
        let full_inner = self.read_join_output_count_metadata(&d_count_only)? as u64;
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
        let mut d_output_count = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .memset_zeros(&mut d_output_count)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero output count: {}", e)))?;

        // SAFETY: hash_join_probe_v2(probe_hashes, num_probe,
        //                            bucket_offsets, bucket_counts, bucket_entries, bucket_entry_hashes, bucket_mask,
        //                            probe_keys, build_keys, key_bytes,
        //                            output_left, output_right, output_count, max_output)
        // Note: Using raw pointer launch because tuple exceeds 12-element limit
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&right_packed.packed_keys).as_kernel_param(),
                left_packed.key_bytes.as_kernel_param(),
                (&d_output_left).as_kernel_param(),
                (&d_output_right).as_kernel_param(),
                (&d_output_count).as_kernel_param(),
                max_output.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch(config, &mut params)
                .map_err(|e| XlogError::Kernel(format!("hash_join_probe_v2 failed: {}", e)))?;
        }

        self.device.synchronize()?;

        let device = self.device.inner();

        // Metadata read: post-materialize device-side atomic count, used
        // as the inner-join result's logical row count after clamping.
        // Clamp to max_output to prevent buffer overflow (kernel atomically
        // increments before bounds check, so count can exceed max_output).
        let inner_count = self
            .read_join_output_count_metadata(&d_output_count)?
            .min(max_output);

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

        let unmatched_rows = self.device_row_count(&unmatched_left)? as u64;
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
            result_columns.extend(inner_left.columns);
            result_columns.extend(inner_right.columns);
            return self.buffer_from_columns(result_columns, inner_count as u64, combined_schema);
        }

        if inner_count == 0 {
            let mut result_columns = Vec::with_capacity(combined_schema.arity());
            result_columns.extend(unmatched_left.columns);

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
            return self.buffer_from_columns(result_columns, unmatched_rows, combined_schema);
        }

        // Concatenate: matched rows followed by unmatched rows (null-extended on right).
        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        let inner_rows = inner_count as u64;

        // Left columns: inner-left then unmatched-left.
        for (col_idx, (inner_col, unmatched_col)) in inner_left
            .columns
            .into_iter()
            .zip(unmatched_left.columns)
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
                let unmatched_view = self.column_bytes_view(&unmatched_col, unmatched_bytes)?;
                device
                    .dtod_copy(&unmatched_view, &mut out_view)
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

        self.buffer_from_columns(result_columns, total_rows, combined_schema)
    }

    /// Helper for left outer join with empty right: all left rows with null right columns
    fn left_outer_with_nulls(&self, left: &CudaBuffer, right: &CudaBuffer) -> Result<CudaBuffer> {
        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let num_rows = self.device_row_count(left)? as u64;
        if num_rows == 0 {
            return self.create_empty_buffer(combined_schema);
        }
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
                let src_view = self.column_bytes_view(col, bytes)?;
                device
                    .dtod_copy(&src_view, &mut dst_col)
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

        self.buffer_from_columns(result_columns, num_rows, combined_schema)
    }

    /// Clone a buffer (deep copy) on-device.
    ///
    /// This is primarily used when a caller needs owned buffer state for a
    /// separate runtime object while preserving the original relation store.
    pub fn clone_buffer(&self, buffer: &CudaBuffer) -> Result<CudaBuffer> {
        let mut result_columns = Vec::with_capacity(buffer.arity());
        let device = self.device.inner();

        for col_idx in 0..buffer.arity() {
            let src_col = buffer
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;
            let mut dst_col = self.memory.alloc::<u8>(src_col.len())?;
            if !src_col.is_empty() {
                device
                    .dtod_copy(src_col, &mut dst_col)
                    .map_err(|e| XlogError::Kernel(format!("Failed to clone column: {}", e)))?;
            }
            result_columns.push(dst_col.into());
        }

        let mut d_num_rows = self.memory.alloc::<u32>(1)?;
        device
            .dtod_copy(buffer.num_rows_device(), &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("Failed to clone row count: {}", e)))?;

        let cloned = CudaBuffer::from_columns(
            result_columns,
            buffer.row_cap,
            d_num_rows,
            buffer.schema().clone(),
        );
        // Preserve the host-side row-count cache so downstream code can avoid
        // a D2H read of num_rows_device() just to learn the row count.
        if let Some(cached) = buffer.cached_row_count() {
            cloned.set_cached_row_count_if_unset(cached);
        }
        Ok(cloned)
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
        let src_col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found in buffer", col_idx)))?;
        let mut dst_col = self.memory.alloc::<u8>(src_col.len())?;
        let device = self.device.inner();
        if !src_col.is_empty() {
            device
                .dtod_copy(src_col, &mut dst_col)
                .map_err(|e| XlogError::Kernel(format!("Failed to copy column: {}", e)))?;
        }

        let mut d_num_rows = self.memory.alloc::<u32>(1)?;
        device
            .dtod_copy(buffer.num_rows_device(), &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy row count: {}", e)))?;
        self.device.synchronize()?;

        let schema = Schema::new(vec![("col".to_string(), col_type)]);
        Ok(CudaBuffer::from_columns(
            vec![dst_col.into()],
            buffer.row_cap,
            d_num_rows,
            schema,
        ))
    }

    /// Extract active (i,j,k) rule indices from a flattened N×N×N mask.
    /// Returns up to `max_active` entries sorted by soft-mask priority.
    pub fn extract_active_rule_indices(
        &self,
        mask_hard: &CudaBuffer,
        mask_soft: &CudaBuffer,
        n: usize,
        max_active: usize,
    ) -> Result<Vec<(u32, u32, u32)>> {
        let total = n * n * n;
        let block_size = 256usize;
        let grid_size = total.div_ceil(block_size);

        let mut out_i = self.memory().alloc::<u32>(total)?;
        let mut out_j = self.memory().alloc::<u32>(total)?;
        let mut out_k = self.memory().alloc::<u32>(total)?;
        let mut out_p = self.memory().alloc::<f32>(total)?;
        let mut count = self.memory().alloc::<u32>(1)?;

        self.htod_launch_metadata_sync_copy_into(&[0u32], &mut count)
            .map_err(|e| XlogError::Kernel(format!("ILP htod count: {}", e)))?;

        let hard_col = mask_hard
            .column(0)
            .ok_or_else(|| XlogError::Kernel("ILP hard mask has no column".into()))?;
        let soft_col = mask_soft
            .column(0)
            .ok_or_else(|| XlogError::Kernel("ILP soft mask has no column".into()))?;

        let kernel = self
            .device()
            .inner()
            .get_func(ILP_MODULE, ilp_kernels::EXTRACT_NONZERO_INDICES)
            .ok_or_else(|| XlogError::Kernel("extract_nonzero_indices kernel not found".into()))?;

        let hard_bytes = total * std::mem::size_of::<f32>();
        let soft_bytes = total * std::mem::size_of::<f32>();
        let hard_view = self.column_bytes_view(hard_col, hard_bytes)?;
        let soft_view = self.column_bytes_view(soft_col, soft_bytes)?;

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            kernel
                .clone()
                .launch(
                    cudarc::driver::LaunchConfig {
                        grid_dim: (grid_size as u32, 1, 1),
                        block_dim: (block_size as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &hard_view, &soft_view, n as u32, &mut out_i, &mut out_j, &mut out_k,
                        &mut out_p, &mut count,
                    ),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("Failed to launch extract_nonzero_indices: {}", e))
                })?;
        }

        let mut count_host = [0u32];
        self.device()
            .inner()
            .dtoh_sync_copy_into(&count, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh count: {}", e)))?;
        let active_count = count_host[0] as usize;

        if active_count == 0 {
            return Ok(Vec::new());
        }

        let mut i_host = vec![0u32; active_count];
        let mut j_host = vec![0u32; active_count];
        let mut k_host = vec![0u32; active_count];
        let mut p_host = vec![0f32; active_count];

        let out_i_view = out_i
            .try_slice(0..active_count)
            .ok_or_else(|| XlogError::Kernel("ILP slice i out of bounds".into()))?;
        let out_j_view = out_j
            .try_slice(0..active_count)
            .ok_or_else(|| XlogError::Kernel("ILP slice j out of bounds".into()))?;
        let out_k_view = out_k
            .try_slice(0..active_count)
            .ok_or_else(|| XlogError::Kernel("ILP slice k out of bounds".into()))?;
        let out_p_view = out_p
            .try_slice(0..active_count)
            .ok_or_else(|| XlogError::Kernel("ILP slice p out of bounds".into()))?;

        self.device()
            .inner()
            .dtoh_sync_copy_into(&out_i_view, &mut i_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh i: {}", e)))?;
        self.device()
            .inner()
            .dtoh_sync_copy_into(&out_j_view, &mut j_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh j: {}", e)))?;
        self.device()
            .inner()
            .dtoh_sync_copy_into(&out_k_view, &mut k_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh k: {}", e)))?;
        self.device()
            .inner()
            .dtoh_sync_copy_into(&out_p_view, &mut p_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh p: {}", e)))?;

        let mut indices: Vec<(f32, u32, u32, u32)> = (0..active_count)
            .map(|idx| (p_host[idx], i_host[idx], j_host[idx], k_host[idx]))
            .collect();
        indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        indices.truncate(max_active);

        Ok(indices.into_iter().map(|(_, i, j, k)| (i, j, k)).collect())
    }

    // ============== Recorded sort + dedup_full_row (v0.6 slice #5) ==============
    //
    // Strict-recorder, launch_stream-routed siblings of `sort` and
    // `dedup_full_row`. Scope-narrow per the slice directive:
    //   * `sort_recorded` accepts only u32 / Symbol key columns; other key
    //     types return XlogError::Kernel
    //     before any kernel work is queued.
    //   * `dedup_full_row_recorded` requires every column to be u32 / Symbol
    //     (it composes sort_recorded internally). Mixed-type full-row dedup
    //     remains on the legacy `dedup_full_row`.
    //
    // No legacy default-routed code is touched. Existing callers are
    // unchanged. Runtime/provider opt-in wiring is NOT included in this
    // slice — callers that want recorded sort/dedup must invoke the
    // recorded methods directly with a launch_stream.

    /// Stream-aware variant of
    /// [`Self::radix_sort_u32_pairs_with_scratch`]. Same kernel
    /// chain but every launch is `launch_on_stream` and there
    /// are no internal `device.synchronize()` calls. Caller-owned
    /// scratch (`keys_a/b`, `indices_a/b`, `hist`, `prefix`,
    /// `ranks`) is recorded by the caller; intermediate
    /// `block_sums` allocations created by the inner scan are
    /// recorded directly inside
    /// [`Self::multiblock_scan_u32_view_inplace_on_stream`].
    #[allow(clippy::too_many_arguments)]
    fn radix_sort_u32_pairs_with_scratch_on_stream(
        &self,
        keys_a: &mut TrackedCudaSlice<u32>,
        keys_b: &mut TrackedCudaSlice<u32>,
        indices_a: &mut TrackedCudaSlice<u32>,
        indices_b: &mut TrackedCudaSlice<u32>,
        hist: &mut TrackedCudaSlice<u32>,
        prefix: &mut TrackedCudaSlice<u32>,
        ranks: &mut TrackedCudaSlice<u32>,
        num_rows_device: &TrackedCudaSlice<u32>,
        row_cap: u32,
        cu_stream: &cudarc::driver::CudaStream,
        launch_stream: StreamId,
        runtime: &crate::device_runtime::XlogDeviceRuntime,
    ) -> Result<()> {
        if row_cap == 0 {
            return Ok(());
        }
        let device = self.device.inner();
        let block_size = Self::SORT_BLOCK_SIZE;
        let grid_size = row_cap.div_ceil(block_size);
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

            // SAFETY: radix_histogram(keys, num_rows_device, row_cap, histograms, shift)
            unsafe {
                histogram_fn.clone().launch_on_stream(
                    cu_stream,
                    sort_config,
                    (keys_in, num_rows_device, row_cap, &mut *hist, shift),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("radix_histogram (on_stream) failed: {}", e)))?;

            // SAFETY: compute_digit_prefix_sums(histograms, grid_size, prefix_sums)
            unsafe {
                prefix_fn.clone().launch_on_stream(
                    cu_stream,
                    prefix_config,
                    (&*hist, grid_size, &mut *prefix),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "compute_digit_prefix_sums (on_stream) failed: {}",
                    e
                ))
            })?;

            // Per-digit per-block exclusive offsets — in-place
            // scan on a 16-strided view of `hist`.
            for digit in 0..16u32 {
                let start = (digit * grid_size) as usize;
                let end = start + (grid_size as usize);
                let mut digit_slice = hist.slice_mut(start..end);
                self.multiblock_scan_u32_view_inplace_on_stream(
                    &mut digit_slice,
                    grid_size,
                    cu_stream,
                    launch_stream,
                    runtime,
                )?;
            }

            // SAFETY: compute_ranks(keys, num_rows_device, row_cap, ranks, shift)
            unsafe {
                ranks_fn.clone().launch_on_stream(
                    cu_stream,
                    sort_config,
                    (keys_in, num_rows_device, row_cap, &mut *ranks, shift),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("compute_ranks (on_stream) failed: {}", e)))?;

            // SAFETY: radix_scatter_stable(keys_in, indices_in, ranks, keys_out,
            // indices_out, prefix_sums, block_offsets, num_rows_device, row_cap, shift)
            unsafe {
                scatter_fn.clone().launch_on_stream(
                    cu_stream,
                    sort_config,
                    (
                        keys_in,
                        indices_in,
                        &*ranks,
                        keys_out,
                        indices_out,
                        &*prefix,
                        &*hist,
                        num_rows_device,
                        row_cap,
                        shift,
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("radix_scatter_stable (on_stream) failed: {}", e))
            })?;

            in_a = !in_a;
        }

        if !in_a {
            return Err(XlogError::Kernel(
                "Unexpected radix-sort buffer parity (expected even number of passes)".to_string(),
            ));
        }
        Ok(())
    }

    /// Stream-aware variant of
    /// [`Self::apply_permutation_gpu`]. Permutes every input
    /// column on `launch_stream` into caller-allocated
    /// `dst_cols`. No internal sync; caller records both the
    /// permutation slice and `dst_cols`.
    fn apply_permutation_gpu_on_stream(
        &self,
        input: &CudaBuffer,
        permutation: &TrackedCudaSlice<u32>,
        dst_cols: &mut [TrackedCudaSlice<u8>],
        cu_stream: &cudarc::driver::CudaStream,
    ) -> Result<()> {
        let row_cap = input.num_rows() as u32;
        let d_num_rows = input.num_rows_device();
        let device = self.device.inner();

        let grid_size = row_cap.div_ceil(Self::SORT_BLOCK_SIZE);
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

        if dst_cols.len() != input.columns.len() {
            return Err(XlogError::Kernel(format!(
                "apply_permutation_gpu_on_stream: dst_cols.len()={} mismatches input.cols={}",
                dst_cols.len(),
                input.columns.len()
            )));
        }

        for (col_idx, dst_col) in dst_cols.iter_mut().enumerate() {
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
            let output_bytes = (row_cap as usize) * (elem_size as usize);
            if src_col.num_bytes() != output_bytes {
                return Err(XlogError::Kernel(format!(
                    "Column {} has {} bytes but expected {} (num_rows={}, elem_size={})",
                    col_idx,
                    src_col.num_bytes(),
                    output_bytes,
                    row_cap,
                    elem_size
                )));
            }
            // SAFETY: apply_permutation_bytes(input, output, permutation,
            // num_rows_device, row_cap, elem_size)
            unsafe {
                apply_perm_fn.clone().launch_on_stream(
                    cu_stream,
                    launch_config,
                    (
                        src_col,
                        &mut *dst_col,
                        permutation,
                        d_num_rows,
                        row_cap,
                        elem_size,
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("apply_permutation_bytes (on_stream) failed: {}", e))
            })?;
        }
        Ok(())
    }

    /// Strict-recorder variant of [`Self::sort`] — narrow to
    /// `u32` / `Symbol` key columns. The whole sort chain
    /// (init → LSD radix passes → multi-column gather) runs on
    /// the caller-supplied `launch_stream`; every input column
    /// and the input row-count buffer are recorded as reads
    /// before preflight; every fresh runtime-backed allocation
    /// (scratch + output columns + output `d_num_rows`) is
    /// recorded via `write` BEFORE preflight (snapshot drops the borrow so kernel `&mut` borrows after preflight remain valid)
    /// enqueue.
    ///
    /// # Errors
    ///   * Manager not runtime-backed.
    ///   * `launch_stream` does not resolve.
    ///   * Empty `key_cols` or out-of-bounds index.
    ///   * Any key column type other than `U32` / `Symbol`
    ///     (multi-type recorded sort is outside this API surface).
    ///   * Preflight / kernel / commit failures.
    pub fn sort_recorded(
        &self,
        input: &CudaBuffer,
        key_cols: &[usize],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "sort_recorded requires a runtime-backed GpuMemoryManager (with_runtime)"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "sort_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        if input.num_rows() == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }
        if key_cols.is_empty() {
            return Err(XlogError::Kernel(
                "Sort requires at least one key column".to_string(),
            ));
        }
        if input.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Sort supports at most {} rows, got {}",
                u32::MAX,
                input.num_rows()
            )));
        }
        for &k in key_cols {
            if k >= input.arity() {
                return Err(XlogError::Kernel(format!(
                    "Key column index {} out of bounds (arity {})",
                    k,
                    input.arity()
                )));
            }
            let ty = input.schema.column_type(k).ok_or_else(|| {
                XlogError::Kernel(format!("Key column {} type not found in schema", k))
            })?;
            if !matches!(ty, ScalarType::U32 | ScalarType::Symbol | ScalarType::U64) {
                return Err(XlogError::Kernel(format!(
                    "sort_recorded supports only U32 / Symbol / U64 key columns; \
                     got {:?} for column {}",
                    ty, k
                )));
            }
        }

        let n = input.num_rows() as u32;
        let block_size = Self::SORT_BLOCK_SIZE;
        let grid_size = n.div_ceil(block_size);
        let device = self.device.inner();
        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Pre-allocate ALL fresh runtime-backed buffers BEFORE
        // recorder construction (Rust drop order).
        let mut indices_a = self.memory.alloc::<u32>(n as usize)?;
        let mut indices_b = self.memory.alloc::<u32>(n as usize)?;
        let mut keys_a = self.memory.alloc::<u32>(n as usize)?;
        let mut keys_b = self.memory.alloc::<u32>(n as usize)?;
        let mut d_hist = self.memory.alloc::<u32>((grid_size as usize) * 16)?;
        let mut d_prefix = self.memory.alloc::<u32>(16)?;
        let mut d_ranks = self.memory.alloc::<u32>(n as usize)?;
        // Output's d_num_rows must reflect input's LOGICAL
        // row count (not row_cap). The legacy `sort` clones
        // `input.num_rows_device()` via `dtod_copy`; recorded
        // sort does the same on launch_stream so downstream
        // consumers see the correct logical count.
        let output_d_num_rows = self.memory.alloc::<u32>(1)?;

        let mut dst_cols: Vec<TrackedCudaSlice<u8>> = Vec::with_capacity(input.columns.len());
        for col_idx in 0..input.columns.len() {
            let elem_size = input
                .schema
                .column_type(col_idx)
                .ok_or_else(|| {
                    XlogError::Kernel(format!("Schema type for column {} not found", col_idx))
                })?
                .size_bytes();
            dst_cols.push(self.memory.alloc::<u8>((n as usize) * elem_size)?);
        }

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(input.num_rows_device());
        for col_idx in 0..input.columns.len() {
            let c = input
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;
            rec.read_column(c);
        }
        // Pre-launch fresh writes: the recorder snapshots block
        // identity at record time and drops the slice borrow,
        // so kernel `&mut` borrows after preflight are unaffected.
        rec.write(&indices_a);
        rec.write(&indices_b);
        rec.write(&keys_a);
        rec.write(&keys_b);
        rec.write(&d_hist);
        rec.write(&d_prefix);
        rec.write(&d_ranks);
        rec.write(&output_d_num_rows);
        for dst_col in &dst_cols {
            rec.write(dst_col);
        }
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("sort_recorded: preflight failed: {}", e)))?;

        // Step 1: init_indices.
        let init_fn = device
            .get_func(SORT_MODULE, sort_kernels::INIT_INDICES)
            .ok_or_else(|| XlogError::Kernel("init_indices kernel not found".to_string()))?;
        // SAFETY: init_indices(indices, num_rows_device, row_cap)
        unsafe {
            init_fn.clone().launch_on_stream(
                &cu_stream,
                launch_config,
                (&mut indices_a, input.num_rows_device(), n),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("init_indices (on_stream) failed: {}", e)))?;

        // Step 2: LSD radix passes per key column. U32 / Symbol
        // are 4-byte → one radix pass per column. U64 keys use
        // the hi/lo gather pair (mirrors legacy `sort()`'s
        // strategy at line ~1691): one radix pass per half,
        // lo-first then hi, so the stable LSD ordering is
        // hi-most-significant.
        for &col_idx in key_cols.iter().rev() {
            let col = input
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Key column {} not found", col_idx)))?;
            let ty = input.schema.column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Key column {} type not found in schema", col_idx))
            })?;
            match ty {
                ScalarType::U32 | ScalarType::Symbol => {
                    let col_view = self.column_as_u32_view(col, n as usize)?;
                    let gather_fn = device
                        .get_func(SORT_MODULE, sort_kernels::APPLY_PERMUTATION_U32)
                        .ok_or_else(|| {
                            XlogError::Kernel("apply_permutation_u32 kernel not found".to_string())
                        })?;
                    // SAFETY: apply_permutation_u32(input, output, permutation,
                    // num_rows_device, row_cap)
                    unsafe {
                        gather_fn.clone().launch_on_stream(
                            &cu_stream,
                            launch_config,
                            (
                                &col_view,
                                &mut keys_a,
                                &indices_a,
                                input.num_rows_device(),
                                n,
                            ),
                        )
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "apply_permutation_u32 (on_stream) failed: {}",
                            e
                        ))
                    })?;

                    self.radix_sort_u32_pairs_with_scratch_on_stream(
                        &mut keys_a,
                        &mut keys_b,
                        &mut indices_a,
                        &mut indices_b,
                        &mut d_hist,
                        &mut d_prefix,
                        &mut d_ranks,
                        input.num_rows_device(),
                        n,
                        &cu_stream,
                        launch_stream,
                        runtime,
                    )?;
                }
                ScalarType::U64 => {
                    let col_view = self.column_as_u64_view(col, n as usize)?;
                    for &word in &[
                        sort_kernels::GATHER_KEYS_U64_LO_U32,
                        sort_kernels::GATHER_KEYS_U64_HI_U32,
                    ] {
                        let gather_fn = device.get_func(SORT_MODULE, word).ok_or_else(|| {
                            XlogError::Kernel(format!("{} kernel not found", word))
                        })?;
                        // SAFETY: gather_keys_u64_*_u32(vals, permutation,
                        // num_rows_device, row_cap, out_keys)
                        unsafe {
                            gather_fn.clone().launch_on_stream(
                                &cu_stream,
                                launch_config,
                                (
                                    &col_view,
                                    &indices_a,
                                    input.num_rows_device(),
                                    n,
                                    &mut keys_a,
                                ),
                            )
                        }
                        .map_err(|e| {
                            XlogError::Kernel(format!("{} (on_stream) failed: {}", word, e))
                        })?;

                        self.radix_sort_u32_pairs_with_scratch_on_stream(
                            &mut keys_a,
                            &mut keys_b,
                            &mut indices_a,
                            &mut indices_b,
                            &mut d_hist,
                            &mut d_prefix,
                            &mut d_ranks,
                            input.num_rows_device(),
                            n,
                            &cu_stream,
                            launch_stream,
                            runtime,
                        )?;
                    }
                }
                other => {
                    return Err(XlogError::Kernel(format!(
                        "sort_recorded: column {} unexpected type {:?} after guard",
                        col_idx, other
                    )));
                }
            }
        }

        // Step 3: gather all input columns by the final permutation.
        self.apply_permutation_gpu_on_stream(input, &indices_a, &mut dst_cols, &cu_stream)?;

        // Step 4: copy input's logical d_num_rows into the
        // output's slot via dtod-async on launch_stream.
        // Sort preserves row count, so this matches what
        // legacy `apply_permutation_gpu` does via
        // `clone_device_row_count`.
        // SAFETY: runtime-backed buffers, 4-byte u32 copy.
        unsafe {
            let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                *output_d_num_rows.device_ptr(),
                *input.num_rows_device().device_ptr(),
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "sort_recorded: cuMemcpyDtoDAsync (output_d_num_rows) failed: {:?}",
                    res
                )));
            }
        }

        rec.commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("sort_recorded: commit failed: {}", e)))?;

        let new_columns: Vec<CudaColumn> = dst_cols.into_iter().map(|s| s.into()).collect();
        Ok(CudaBuffer::from_columns(
            new_columns,
            input.num_rows(),
            output_d_num_rows,
            input.schema.clone(),
        ))
    }

    /// Strict-recorder variant of [`Self::dedup_full_row`] —
    /// narrow to U32 / Symbol / U64 columns.
    ///
    /// Composes [`Self::sort_recorded`] (typed multi-column
    /// sort) → on-stream `mark_unique_full_row_bytewise` →
    /// [`Self::compact_buffer_by_device_mask_counted_recorded`]
    /// (gather kept rows). All three primitives commit
    /// independently; the runtime's record-all + wait-all
    /// `last_use_events: Vec<CudaEvent>` semantics chain the
    /// deallocate safety end-to-end.
    pub fn dedup_full_row_recorded(
        &self,
        input: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "dedup_full_row_recorded requires a runtime-backed GpuMemoryManager".to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "dedup_full_row_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        let row_count = input.num_rows() as usize;
        if row_count == 0 {
            return self.create_empty_buffer(input.schema().clone());
        }
        if row_count == 1 {
            return self.clone_buffer(input);
        }
        if row_count > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "dedup_full_row_recorded supports at most {} rows, got {}",
                u32::MAX,
                row_count
            )));
        }
        let arity = input.arity();
        if arity == 0 {
            return self.buffer_from_columns(Vec::new(), 1, input.schema().clone());
        }
        for col_idx in 0..arity {
            let ty = input.schema.column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Column {} type not found in schema", col_idx))
            })?;
            if !matches!(ty, ScalarType::U32 | ScalarType::Symbol | ScalarType::U64) {
                return Err(XlogError::Kernel(format!(
                    "dedup_full_row_recorded supports only U32 / Symbol / U64 columns; \
                     got {:?} for column {}",
                    ty, col_idx
                )));
            }
        }

        // Step 1: typed sort on launch_stream.
        let all_cols: Vec<usize> = (0..arity).collect();
        let sorted = self.sort_recorded(input, &all_cols, launch_stream)?;
        let n = sorted.num_rows() as u32;
        if n <= 1 {
            return Ok(sorted);
        }

        // Step 2: bytewise adjacent-equality mask. Allocate
        // d_col_ptrs / d_col_sizes / d_unique_mask up front
        // before the recorder. col_ptrs/sizes are populated
        // synchronously as launch metadata (ordered before the
        // launch_stream kernel sees them).
        let device = self.device.inner();
        let mut col_ptrs_host: Vec<u64> = Vec::with_capacity(arity);
        let mut col_sizes_host: Vec<u32> = Vec::with_capacity(arity);
        for col_idx in 0..arity {
            let c = sorted
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Sorted column {} not found", col_idx)))?;
            let ty = sorted.schema().column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("Sorted column {} type missing", col_idx))
            })?;
            col_ptrs_host.push(*c.device_ptr());
            col_sizes_host.push(ty.size_bytes() as u32);
        }
        let mut d_col_ptrs = self.memory.alloc::<u64>(arity)?;
        let mut d_col_sizes = self.memory.alloc::<u32>(arity)?;
        self.htod_launch_metadata_sync_copy_into(&col_ptrs_host, &mut d_col_ptrs)
            .map_err(|e| {
                XlogError::Kernel(format!("dedup_full_row_recorded col ptr upload: {}", e))
            })?;
        self.htod_launch_metadata_sync_copy_into(&col_sizes_host, &mut d_col_sizes)
            .map_err(|e| {
                XlogError::Kernel(format!("dedup_full_row_recorded col size upload: {}", e))
            })?;
        let d_unique_mask = self.memory.alloc::<u8>(n as usize)?;

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        for col_idx in 0..arity {
            let c = sorted
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Sorted column {} not found", col_idx)))?;
            rec.read_column(c);
        }
        rec.read(sorted.num_rows_device());
        rec.write(&d_col_ptrs);
        rec.write(&d_col_sizes);
        rec.write(&d_unique_mask);
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "dedup_full_row_recorded: mark_unique preflight failed: {}",
                e
            ))
        })?;

        let block_size = 256u32;
        let grid = n.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let mark_fn = device
            .get_func(DEDUP_MODULE, dedup_kernels::MARK_UNIQUE_FULL_ROW_BYTEWISE)
            .ok_or_else(|| {
                XlogError::Kernel("mark_unique_full_row_bytewise kernel not found".to_string())
            })?;
        // SAFETY: mark_unique_full_row_bytewise(col_ptrs, col_sizes,
        // num_cols, num_rows_device, row_cap, unique_mask)
        unsafe {
            mark_fn.clone().launch_on_stream(
                &cu_stream,
                cfg,
                (
                    &d_col_ptrs,
                    &d_col_sizes,
                    arity as u32,
                    sorted.num_rows_device(),
                    n,
                    &d_unique_mask,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "mark_unique_full_row_bytewise (on_stream) failed: {}",
                e
            ))
        })?;

        rec.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "dedup_full_row_recorded: mark_unique commit failed: {}",
                e
            ))
        })?;

        // Step 3: gather kept rows via the recorded compact tail.
        self.compact_buffer_by_device_mask_counted_recorded(&sorted, &d_unique_mask, launch_stream)
    }

    // ============== Recorded hash join — slice #7A: inner only ==============
    //
    // Strict-recorder, launch_stream-routed sibling of
    // `hash_join_inner_v2`. Composes the existing recorded
    // pack helper (slice #6, `pack_keys_gpu_on_stream`) with
    // two new on-stream helpers — `build_hash_table_v2_on_stream`
    // and `gather_buffer_by_indices_on_stream` — and runs the
    // probe kernel + count + materialize chain entirely on
    // launch_stream. Existing `hash_join_v2_*` callers keep
    // their bit-for-bit semantics; runtime/planner wiring is
    // NOT in this slice.
    //
    // Scope:
    //   * `JoinType::Inner` only. Semi/Anti/LeftOuter and the
    //     indexed variant (`hash_join_v2_with_index`) are outside
    //     this recorded provider surface.
    //   * Pack-keys constraint of ≤4 columns inherits from
    //     `pack_keys_gpu_on_stream`.
    //   * Algorithm is unchanged: count-then-materialize
    //     (two probe passes); the GPU-resident
    //     count-prefix-materialize prototype is not reintroduced here.

    /// Stream-aware variant of `build_hash_table_v2`. Mirrors
    /// the legacy bucket-count → exclusive-scan → scatter chain
    /// on the caller-supplied `launch_stream` (no internal
    /// `device.synchronize()`). Each fresh scratch allocation is
    /// fenced via `prepare_first_use(Access::Write)` immediately
    /// after alloc so the first cross-stream consumer (memset /
    /// dtod-copy / kernel) waits for cuMemAllocAsync to complete;
    /// at exit, every block that escapes (the four returned
    /// bucket buffers) plus the internal `bucket_cursors` scratch
    /// is finalized with `finish_block_use(Access::Write)` so
    /// end-of-scope drops are correctly serialized.
    fn build_hash_table_v2_on_stream(
        &self,
        hashes: &TrackedCudaSlice<u64>,
        num_rows: u32,
        cu_stream: &cudarc::driver::CudaStream,
        launch_stream: StreamId,
        runtime: &crate::device_runtime::XlogDeviceRuntime,
    ) -> Result<crate::provider::JoinHashTableV2> {
        let device = self.device.inner();

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

        let bucket_counts = self.memory.alloc::<u32>(num_buckets as usize)?;
        // Fence the alloc-ready event from `bucket_counts`'s
        // alloc_stream onto `launch_stream` BEFORE the memset
        // below — the memset would otherwise execute against a
        // stream that has not waited on cuMemAllocAsync's
        // completion event, producing garbage / pool-recycled
        // bytes when the alloc and use streams differ.
        runtime
            .prepare_first_use(&bucket_counts, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "build_hash_table_v2_on_stream: prepare bucket_counts failed: {}",
                    e
                ))
            })?;
        // Async u32 zero-fill on launch_stream.
        if num_buckets > 0 {
            // SAFETY: bucket_counts is runtime-backed for
            // num_buckets * 4 bytes; cu_stream is a valid
            // stream the runtime owns.
            unsafe {
                let res = cudarc::driver::sys::cuMemsetD8Async(
                    *bucket_counts.device_ptr(),
                    0,
                    (num_buckets as usize) * std::mem::size_of::<u32>(),
                    cu_stream.cu_stream(),
                );
                if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "cuMemsetD8Async (bucket_counts) failed: {:?}",
                        res
                    )));
                }
            }
        }

        let block_size = 256u32;
        let grid_size = num_rows.div_ceil(block_size);
        let cfg = LaunchConfig {
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
            count_fn.clone().launch_on_stream(
                cu_stream,
                cfg,
                (hashes, num_rows, &bucket_counts, bucket_mask),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_bucket_count_v2 (on_stream) failed: {}",
                e
            ))
        })?;

        let mut bucket_offsets = self.memory.alloc::<u32>(num_buckets as usize)?;
        // See `bucket_counts` rationale above: fence
        // alloc-ready → launch_stream before the dtod-copy.
        runtime
            .prepare_first_use(&bucket_offsets, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "build_hash_table_v2_on_stream: prepare bucket_offsets failed: {}",
                    e
                ))
            })?;
        if num_buckets > 0 {
            // dtod copy bucket_counts → bucket_offsets on launch_stream.
            // SAFETY: both buffers are runtime-backed for the
            // same num_buckets * 4 bytes; cu_stream is valid.
            unsafe {
                let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    *bucket_offsets.device_ptr(),
                    *bucket_counts.device_ptr(),
                    (num_buckets as usize) * std::mem::size_of::<u32>(),
                    cu_stream.cu_stream(),
                );
                if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "cuMemcpyDtoDAsync (bucket_counts → bucket_offsets) failed: {:?}",
                        res
                    )));
                }
            }
            self.multiblock_scan_u32_inplace_on_stream(
                &mut bucket_offsets,
                num_buckets,
                cu_stream,
                launch_stream,
                runtime,
            )?;
        }

        let bucket_cursors = self.memory.alloc::<u32>(num_buckets as usize)?;
        // Fence alloc-ready → launch_stream for cursors before
        // the dtod-copy.
        runtime
            .prepare_first_use(&bucket_cursors, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "build_hash_table_v2_on_stream: prepare bucket_cursors failed: {}",
                    e
                ))
            })?;
        if num_buckets > 0 {
            // dtod copy bucket_offsets → bucket_cursors on launch_stream.
            // SAFETY: same shape and size constraints as above.
            unsafe {
                let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    *bucket_cursors.device_ptr(),
                    *bucket_offsets.device_ptr(),
                    (num_buckets as usize) * std::mem::size_of::<u32>(),
                    cu_stream.cu_stream(),
                );
                if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "cuMemcpyDtoDAsync (bucket_offsets → bucket_cursors) failed: {:?}",
                        res
                    )));
                }
            }
        }

        let bucket_entries = self.memory.alloc::<u32>(num_rows as usize)?;
        let bucket_entry_hashes = self.memory.alloc::<u64>(num_rows as usize)?;
        // Fence alloc-ready → launch_stream for both before the
        // scatter kernel writes them.
        runtime
            .prepare_first_use(&bucket_entries, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "build_hash_table_v2_on_stream: prepare bucket_entries failed: {}",
                    e
                ))
            })?;
        runtime
            .prepare_first_use(&bucket_entry_hashes, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "build_hash_table_v2_on_stream: prepare bucket_entry_hashes failed: {}",
                    e
                ))
            })?;

        let scatter_fn = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_SCATTER_V2)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_scatter_v2 kernel not found".to_string())
            })?;
        // SAFETY: hash_join_scatter_v2(hashes, num_rows, bucket_cursors, bucket_mask, bucket_entries, bucket_entry_hashes)
        unsafe {
            scatter_fn.clone().launch_on_stream(
                cu_stream,
                cfg,
                (
                    hashes,
                    num_rows,
                    &bucket_cursors,
                    bucket_mask,
                    &bucket_entries,
                    &bucket_entry_hashes,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!("hash_join_scatter_v2 (on_stream) failed: {}", e))
        })?;

        // Record uses on launch_stream:
        // * bucket_cursors drops at end of helper — must be
        //   recorded so the runtime defers its free behind
        //   the scatter kernel.
        // * bucket_counts / bucket_offsets / bucket_entries /
        //   bucket_entry_hashes escape via JoinHashTableV2;
        //   record so downstream drops are gated.
        for blk in [
            bucket_counts.runtime_block(),
            bucket_offsets.runtime_block(),
            bucket_cursors.runtime_block(),
            bucket_entries.runtime_block(),
            bucket_entry_hashes.runtime_block(),
        ] {
            if let Some(b) = blk {
                runtime
                    .finish_block_use(BlockId::from_block(b), launch_stream, Access::Write)
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "build_hash_table_v2_on_stream: finish_block_use failed: {}",
                            e
                        ))
                    })?;
            } else {
                return Err(XlogError::Kernel(
                    "build_hash_table_v2_on_stream: buffer has no runtime block — \
                     caller must use a runtime-backed manager"
                        .to_string(),
                ));
            }
        }

        Ok(crate::provider::JoinHashTableV2 {
            bucket_counts,
            bucket_offsets,
            bucket_entries,
            bucket_entry_hashes,
            bucket_mask,
        })
    }

    /// Stream-aware variant of `gather_buffer_by_indices`.
    /// Allocates output column storage, runs
    /// `apply_permutation_bytes` per input column on
    /// `launch_stream`, and assembles a `CudaBuffer`. Records
    /// every fresh allocation directly via the runtime so the
    /// returned buffer is safe to drop before any pending
    /// gather kernel completes.
    fn gather_buffer_by_indices_on_stream(
        &self,
        input: &CudaBuffer,
        indices: &TrackedCudaSlice<u32>,
        output_rows: u32,
        cu_stream: &cudarc::driver::CudaStream,
        launch_stream: StreamId,
        runtime: &crate::device_runtime::XlogDeviceRuntime,
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

        let d_output_rows = self.upload_device_row_count(output_rows)?;
        // `upload_device_row_count` initializes this scalar on
        // the manager/default stream. Publish that write into
        // the runtime dependency state, then fence launch_stream
        // before the gather kernels read it. The scalar is local
        // scratch, so we also finish a read after the kernels so
        // its drop/free waits for launch_stream completion.
        runtime
            .finish_first_use(&d_output_rows, StreamId::DEFAULT, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "gather_buffer_by_indices_on_stream: record d_output_rows upload failed: {}",
                    e
                ))
            })?;
        runtime
            .prepare_first_use(&d_output_rows, launch_stream, Access::Read)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "gather_buffer_by_indices_on_stream: prepare d_output_rows failed: {}",
                    e
                ))
            })?;
        let device = self.device.inner();
        let block_size = 256u32;
        let grid_size = output_rows.div_ceil(block_size);
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

        let mut dst_cols: Vec<TrackedCudaSlice<u8>> = Vec::with_capacity(input.columns.len());
        for col_idx in 0..input.columns.len() {
            let elem_size = input
                .schema
                .column_type(col_idx)
                .ok_or_else(|| {
                    XlogError::Kernel(format!("Schema type for column {} not found", col_idx))
                })?
                .size_bytes() as u32;
            let dst_bytes = (output_rows as usize) * (elem_size as usize);
            let dst = self.memory.alloc::<u8>(dst_bytes)?;
            // Fence alloc-ready → launch_stream for each fresh
            // dst_col before the gather kernel writes it.
            runtime
                .prepare_first_use(&dst, launch_stream, Access::Write)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "gather_buffer_by_indices_on_stream: prepare dst_col {} failed: {}",
                        col_idx, e
                    ))
                })?;
            dst_cols.push(dst);
        }

        for (col_idx, dst_col) in dst_cols.iter_mut().enumerate() {
            let src_col = input
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;
            let elem_size = input
                .schema
                .column_type(col_idx)
                .map(|t| t.size_bytes() as u32)
                .unwrap_or(4);
            // SAFETY: apply_permutation_bytes(input, output, permutation, num_rows_device, row_cap, elem_size)
            unsafe {
                gather_fn.clone().launch_on_stream(
                    cu_stream,
                    launch_config,
                    (
                        src_col,
                        &mut *dst_col,
                        indices,
                        &d_output_rows,
                        output_rows,
                        elem_size,
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("apply_permutation_bytes (on_stream) failed: {}", e))
            })?;
        }

        runtime
            .finish_first_use(&d_output_rows, launch_stream, Access::Read)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "gather_buffer_by_indices_on_stream: record d_output_rows read failed: {}",
                    e
                ))
            })?;

        // Record uses on launch_stream for buffers we wrote
        // (the dst_cols escape via the returned CudaBuffer).
        // input.column[i] reads will be recorded by the
        // caller's outer LaunchRecorder.
        for dst_col in &dst_cols {
            if let Some(b) = dst_col.runtime_block() {
                runtime
                    .finish_block_use(BlockId::from_block(b), launch_stream, Access::Write)
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "gather_buffer_by_indices_on_stream: finish_block_use \
                         (dst_col) failed: {}",
                            e
                        ))
                    })?;
            } else {
                return Err(XlogError::Kernel(
                    "gather_buffer_by_indices_on_stream: dst_col has no runtime block".to_string(),
                ));
            }
        }

        let new_columns: Vec<CudaColumn> = dst_cols.into_iter().map(|s| s.into()).collect();
        Ok(CudaBuffer::from_columns(
            new_columns,
            output_rows as u64,
            d_output_rows,
            input.schema.clone(),
        ))
    }

    /// Strict-recorder variant of `hash_join_inner_v2`.
    /// `JoinType::Inner` only. Same count-then-materialize
    /// algorithm as the legacy variant, but every kernel
    /// runs on the caller-supplied `launch_stream` and host
    /// scalar reads of the join output count are explicitly
    /// ordered against the stream.
    pub fn hash_join_inner_v2_recorded(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        max_output: Option<usize>,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        use crate::launch::LaunchRecorder;

        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "hash_join_inner_v2_recorded requires a runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "hash_join_inner_v2_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }
        if num_left == 0 || num_right == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }
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
        if left_keys.len() > 4 {
            return Err(XlogError::Kernel(
                "hash_join_inner_v2_recorded: max 4 key columns supported (pack_keys constraint)"
                    .to_string(),
            ));
        }
        for (&l, &r) in left_keys.iter().zip(right_keys.iter()) {
            let lt = left.schema().column_type(l);
            let rt = right.schema().column_type(r);
            if lt != rt {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    l, lt, r, rt
                )));
            }
        }

        let num_left = num_left as u32;
        let num_right = num_right as u32;

        // Step 1+2: pack keys for both sides on launch_stream.
        let left_packed =
            self.pack_keys_gpu_on_stream(left, left_keys, &cu_stream, launch_stream, runtime)?;
        let right_packed =
            self.pack_keys_gpu_on_stream(right, right_keys, &cu_stream, launch_stream, runtime)?;

        // Step 3: build hash table on launch_stream.
        let table = self.build_hash_table_v2_on_stream(
            &right_packed.hashes,
            num_right,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        let probe_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2)
            .ok_or_else(|| XlogError::Kernel("hash_join_probe_v2 kernel not found".to_string()))?;
        let block_size = 256u32;
        let probe_grid = num_left.div_ceil(block_size);
        let probe_config = LaunchConfig {
            grid_dim: (probe_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Step 4: count-only pass. Allocate count + dummy
        // output buffers up front. The recorder's preflight will
        // queue the alloc-ready waits before either the memset
        // OR the kernel runs on launch_stream.
        let d_count_only = self.memory.alloc::<u32>(1)?;
        let d_dummy_left = self.memory.alloc::<u32>(1)?;
        let d_dummy_right = self.memory.alloc::<u32>(1)?;

        // Build the recorder for the probe / output stage.
        // Reads BEFORE preflight: hashes + packed_keys (already
        // recorded by pack_keys_gpu_on_stream as writes on
        // launch_stream — recording reads here adds the next
        // event in the chain), table buckets, dummy buffers.
        let max_output_count_only = 0u32;
        let mut rec_count = LaunchRecorder::new_strict(launch_stream);
        rec_count.read(&left_packed.hashes);
        rec_count.read(&left_packed.packed_keys);
        rec_count.read(&right_packed.packed_keys);
        rec_count.read(&table.bucket_offsets);
        rec_count.read(&table.bucket_counts);
        rec_count.read(&table.bucket_entries);
        rec_count.read(&table.bucket_entry_hashes);
        rec_count.write(&d_count_only);
        rec_count.write(&d_dummy_left);
        rec_count.write(&d_dummy_right);
        rec_count.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_inner_v2_recorded: count-pass preflight failed: {}",
                e
            ))
        })?;

        // Zero-init d_count_only via async memset on
        // launch_stream — runs AFTER preflight has queued the
        // alloc-ready waits, so the memset is correctly fenced
        // behind cuMemAllocAsync's completion.
        // SAFETY: d_count_only is runtime-backed for 4 bytes.
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_count_only.device_ptr(),
                0,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "cuMemsetD8Async (d_count_only) failed: {:?}",
                    res
                )));
            }
        }

        // SAFETY: hash_join_probe_v2 14-arg signature — see
        // legacy hash_join_inner_v2 for the canonical
        // documentation. Tuple exceeds 12-element limit, so
        // we use the raw-pointer launch path.
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&right_packed.packed_keys).as_kernel_param(),
                left_packed.key_bytes.as_kernel_param(),
                (&d_dummy_left).as_kernel_param(),
                (&d_dummy_right).as_kernel_param(),
                (&d_count_only).as_kernel_param(),
                max_output_count_only.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch_on_stream(&cu_stream, probe_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "hash_join_probe_v2 (count, on_stream) failed: {}",
                        e
                    ))
                })?;
        }

        rec_count.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_inner_v2_recorded: count-pass commit failed: {}",
                e
            ))
        })?;

        // Explicit barrier before host scalar read of full_count.
        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_inner_v2_recorded: launch_stream sync (count read) failed: {}",
                e
            ))
        })?;
        let full_count = self.read_join_output_count_metadata(&d_count_only)? as u64;
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
        let max_output_u32 = requested as u32;

        // Step 5: materialize pass. Allocate output index
        // buffers + count. Memset runs AFTER preflight has
        // queued the alloc-ready waits.
        let d_output_left = self.memory.alloc::<u32>(max_output_u32 as usize)?;
        let d_output_right = self.memory.alloc::<u32>(max_output_u32 as usize)?;
        let d_output_count = self.memory.alloc::<u32>(1)?;

        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        rec_mat.read(&left_packed.hashes);
        rec_mat.read(&left_packed.packed_keys);
        rec_mat.read(&right_packed.packed_keys);
        rec_mat.read(&table.bucket_offsets);
        rec_mat.read(&table.bucket_counts);
        rec_mat.read(&table.bucket_entries);
        rec_mat.read(&table.bucket_entry_hashes);
        rec_mat.write(&d_output_left);
        rec_mat.write(&d_output_right);
        rec_mat.write(&d_output_count);
        rec_mat.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_inner_v2_recorded: materialize-pass preflight failed: {}",
                e
            ))
        })?;

        // Zero-init d_output_count via async memset on
        // launch_stream — fenced behind alloc-ready waits.
        // SAFETY: runtime-backed 4-byte buffer.
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_output_count.device_ptr(),
                0,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "cuMemsetD8Async (d_output_count) failed: {:?}",
                    res
                )));
            }
        }

        // SAFETY: same 14-arg probe signature.
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&right_packed.packed_keys).as_kernel_param(),
                left_packed.key_bytes.as_kernel_param(),
                (&d_output_left).as_kernel_param(),
                (&d_output_right).as_kernel_param(),
                (&d_output_count).as_kernel_param(),
                max_output_u32.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch_on_stream(&cu_stream, probe_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "hash_join_probe_v2 (materialize, on_stream) failed: {}",
                        e
                    ))
                })?;
        }

        rec_mat.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_inner_v2_recorded: materialize-pass commit failed: {}",
                e
            ))
        })?;

        // Explicit barrier before host scalar read of result_count.
        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_inner_v2_recorded: launch_stream sync (mat read) failed: {}",
                e
            ))
        })?;
        let result_count = (self.read_join_output_count_metadata(&d_output_count)? as u64)
            .min(max_output_u32 as u64);
        if result_count == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }
        let output_rows = result_count as u32;

        // Step 6: gather both sides on launch_stream. Each
        // gather records reads of input.column[i] via its own
        // outer LaunchRecorder — set up below.
        let mut rec_gather = LaunchRecorder::new_strict(launch_stream);
        for col_idx in 0..left.columns.len() {
            let c = left
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Left column {} not found", col_idx)))?;
            rec_gather.read_column(c);
        }
        for col_idx in 0..right.columns.len() {
            let c = right
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Right column {} not found", col_idx)))?;
            rec_gather.read_column(c);
        }
        rec_gather.read(&d_output_left);
        rec_gather.read(&d_output_right);
        rec_gather.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_inner_v2_recorded: gather preflight failed: {}",
                e
            ))
        })?;

        let gathered_left = self.gather_buffer_by_indices_on_stream(
            left,
            &d_output_left,
            output_rows,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        let gathered_right = self.gather_buffer_by_indices_on_stream(
            right,
            &d_output_right,
            output_rows,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        rec_gather.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_inner_v2_recorded: gather commit failed: {}",
                e
            ))
        })?;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        result_columns.extend(gathered_left.columns);
        result_columns.extend(gathered_right.columns);
        self.buffer_from_columns(result_columns, result_count, combined_schema)
    }

    /// Strict-recorder, deterministic-ordering Inner hash
    /// join (binary-join retake, sub-slice #1).
    ///
    /// Algorithm: count → exclusive scan → device-resident
    /// total → host scalar read → materialize with
    /// per-probe-row offsets. Each probe row writes its
    /// `local`-th match to
    /// `output[per_probe_offsets[tid] + local]` directly —
    /// no global `atomicAdd(output_count)` on the
    /// materialize pass, so the output ordering is a
    /// deterministic function of (probe-row index,
    /// per-row match discovery order). Compare to
    /// [`Self::hash_join_inner_v2_recorded`] which uses the
    /// legacy count-then-atomic-materialize chain (correct
    /// but with atomic-induced order non-determinism across
    /// threads/blocks).
    ///
    /// Sourced from the archived `archive/gpu-resident-binary-join-prototype-*`
    /// branches — three new kernels migrated:
    /// `hash_join_probe_v2_count_per_row`,
    /// `hash_join_probe_v2_materialize`,
    /// `hash_join_total_from_scan`. LeftOuter / Semi / Anti
    /// / indexed variants from the prototype are
    /// intentionally NOT migrated in this slice.
    ///
    /// Reuses every recorded helper from prior slices:
    /// `pack_keys_gpu_on_stream` (slice #6),
    /// `build_hash_table_v2_on_stream` (slice #7A),
    /// `multiblock_scan_u32_inplace_on_stream` (slice #4),
    /// `gather_buffer_by_indices_on_stream` (slice #7A).
    /// Inherits the slice-8 / `f0942448` compact / pack
    /// fixes via composition.
    pub fn hash_join_inner_v2_count_scan_materialize_recorded(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        max_output: Option<usize>,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        if Self::use_csm_cuda_graph_env() {
            if let Some(result) = self
                .hash_join_inner_v2_count_scan_materialize_cuda_graph_recorded(
                    left,
                    right,
                    left_keys,
                    right_keys,
                    max_output,
                    launch_stream,
                )?
            {
                return Ok(result);
            }
            self.csm_cuda_graph_fallbacks
                .fetch_add(1, Ordering::Relaxed);
        }

        use crate::launch::LaunchRecorder;

        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "hash_join_inner_v2_count_scan_materialize_recorded requires a \
                 runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "hash_join_inner_v2_count_scan_materialize_recorded: launch_stream \
                 StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        // Validation (mirrors `hash_join_inner_v2_recorded`).
        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }
        if num_left == 0 || num_right == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }
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
        if left_keys.len() > 4 {
            return Err(XlogError::Kernel(
                "hash_join_inner_v2_count_scan_materialize_recorded: max 4 key \
                 columns supported (pack_keys constraint)"
                    .to_string(),
            ));
        }
        for (&l, &r) in left_keys.iter().zip(right_keys.iter()) {
            let lt = left.schema().column_type(l);
            let rt = right.schema().column_type(r);
            if lt != rt {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    l, lt, r, rt
                )));
            }
        }

        let _num_left = num_left as u32;
        let probe_cap = left.num_rows() as u32;

        // Steps 1+2: pack + table on launch_stream.
        let left_packed =
            self.pack_keys_gpu_on_stream(left, left_keys, &cu_stream, launch_stream, runtime)?;
        let right_packed =
            self.pack_keys_gpu_on_stream(right, right_keys, &cu_stream, launch_stream, runtime)?;
        let table = self.build_hash_table_v2_on_stream(
            &right_packed.hashes,
            num_right as u32,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        let device = self.device.inner();
        let block_size = 256u32;
        let probe_grid = probe_cap.div_ceil(block_size);
        let probe_config = LaunchConfig {
            grid_dim: (probe_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Allocate count + offsets + total scalar + overflow flag.
        let per_probe_count = self.memory.alloc::<u32>(probe_cap as usize)?;
        let mut per_probe_offsets = self.memory.alloc::<u32>(probe_cap as usize)?;
        let d_logical_count = self.memory.alloc::<u32>(1)?;
        let d_overflow = self.memory.alloc::<u8>(1)?;
        // Fence alloc-ready → launch_stream for both before
        // the memset writes them. The recorder below will
        // attach further dependencies, but the memset runs
        // ahead of the recorder's preflight so we need this
        // direct fence.
        runtime
            .prepare_first_use(&d_overflow, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "hash_join_inner_v2_count_scan_materialize_recorded: prepare d_overflow \
                     failed: {}",
                    e
                ))
            })?;
        runtime
            .prepare_first_use(&d_logical_count, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "hash_join_inner_v2_count_scan_materialize_recorded: prepare d_logical_count \
                     failed: {}",
                    e
                ))
            })?;
        // Zero-init overflow + logical_count on launch_stream.
        // SAFETY: 1-byte and 4-byte runtime-backed buffers.
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_overflow.device_ptr(),
                0,
                1,
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "cuMemsetD8Async (d_overflow init) failed: {:?}",
                    res
                )));
            }
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_logical_count.device_ptr(),
                0,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "cuMemsetD8Async (d_logical_count init) failed: {:?}",
                    res
                )));
            }
        }

        // Build the count/scan recorder. Reads on inputs that
        // outlive this recorder (left/right packed + table)
        // BEFORE preflight; fresh writes on per_probe_count /
        // per_probe_offsets / d_logical_count / d_overflow
        // AFTER kernels enqueue.
        let count_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2_COUNT_PER_ROW)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_probe_v2_count_per_row kernel not found".to_string())
            })?;
        let total_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_TOTAL_FROM_SCAN)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_total_from_scan kernel not found".to_string())
            })?;

        let mut rec_count = LaunchRecorder::new_strict(launch_stream);
        rec_count.read(&left_packed.hashes);
        rec_count.read(&left_packed.packed_keys);
        rec_count.read(&right_packed.packed_keys);
        rec_count.read(&table.bucket_offsets);
        rec_count.read(&table.bucket_counts);
        rec_count.read(&table.bucket_entries);
        rec_count.read(&table.bucket_entry_hashes);
        rec_count.read(left.num_rows_device());
        rec_count.write(&per_probe_count);
        rec_count.write(&per_probe_offsets);
        rec_count.write(&d_logical_count);
        rec_count.write(&d_overflow);
        rec_count.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!("csm inner: count/scan preflight failed: {}", e))
        })?;

        // Step 3: count_per_row.
        // SAFETY: 12-arg signature matches the PTX kernel.
        unsafe {
            count_func.clone().launch_on_stream(
                &cu_stream,
                probe_config,
                (
                    &left_packed.hashes,
                    left.num_rows_device(),
                    probe_cap,
                    &table.bucket_offsets,
                    &table.bucket_counts,
                    &table.bucket_entries,
                    &table.bucket_entry_hashes,
                    table.bucket_mask,
                    &left_packed.packed_keys,
                    &right_packed.packed_keys,
                    left_packed.key_bytes,
                    &per_probe_count,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_probe_v2_count_per_row (on_stream) failed: {}",
                e
            ))
        })?;

        // Step 4: dtod-async copy per_probe_count → per_probe_offsets,
        // then exclusive in-place scan.
        // SAFETY: same length, both runtime-backed u32 buffers.
        unsafe {
            let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                *per_probe_offsets.device_ptr(),
                *per_probe_count.device_ptr(),
                (probe_cap as usize) * std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "csm inner: cuMemcpyDtoDAsync (per_probe_count → offsets) failed: {:?}",
                    res
                )));
            }
        }
        self.multiblock_scan_u32_inplace_on_stream(
            &mut per_probe_offsets,
            probe_cap,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        // Step 5: total_from_scan — writes d_logical_count + d_overflow.
        // SAFETY: 7-arg signature. capacity = probe_cap *
        // num_right is the worst-case bound (cross-product);
        // in practice the chain caps the actual write count
        // after the host scalar read below sizes the output
        // index buffers exactly.
        let materialize_capacity_bound: u64 = (probe_cap as u64).saturating_mul(num_right as u64);
        let materialize_capacity_u32 = materialize_capacity_bound.min(u32::MAX as u64) as u32;
        unsafe {
            total_func.clone().launch_on_stream(
                &cu_stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &per_probe_offsets,
                    &per_probe_count,
                    left.num_rows_device(),
                    probe_cap,
                    materialize_capacity_u32,
                    &d_logical_count,
                    &d_overflow,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_total_from_scan (on_stream) failed: {}",
                e
            ))
        })?;

        rec_count.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("csm inner: count/scan commit failed: {}", e))
        })?;

        // Sync + host scalar read of total. dtoh_scalar_untracked
        // is the sanctioned metadata-read API.
        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!("csm inner: sync (total read) failed: {}", e))
        })?;
        let total = self.read_join_output_count_metadata(&d_logical_count)? as u64;
        let requested = max_output
            .map(|limit| (limit as u64).min(total))
            .unwrap_or(total);
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
        let output_capacity = requested as u32;

        // Step 6: materialize. Allocate index outputs sized to
        // `output_capacity` (the user-clamped total). If
        // `requested < total`, the kernel suppresses writes
        // past `output_capacity` and raises d_overflow — a
        // separate metadata flag the caller can inspect via
        // a future helper. (For this slice we trust the
        // tail of the result is the deterministic "last
        // requested" rows.)
        let d_output_left = self.memory.alloc::<u32>(output_capacity as usize)?;
        let d_output_right = self.memory.alloc::<u32>(output_capacity as usize)?;

        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        rec_mat.read(&left_packed.hashes);
        rec_mat.read(&left_packed.packed_keys);
        rec_mat.read(&right_packed.packed_keys);
        rec_mat.read(&table.bucket_offsets);
        rec_mat.read(&table.bucket_counts);
        rec_mat.read(&table.bucket_entries);
        rec_mat.read(&table.bucket_entry_hashes);
        rec_mat.read(&per_probe_offsets);
        rec_mat.read(left.num_rows_device());
        rec_mat.write(&d_output_left);
        rec_mat.write(&d_output_right);
        // d_overflow is consumed by the materialize kernel — recorder must own it through commit.
        rec_mat.write(&d_overflow);
        rec_mat.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!("csm inner: materialize preflight failed: {}", e))
        })?;

        let materialize_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2_MATERIALIZE)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_probe_v2_materialize kernel not found".to_string())
            })?;
        // SAFETY: 16-arg signature; tuple form supports up to
        // 12 elements, so we use the raw-param launch path.
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                left.num_rows_device().as_kernel_param(),
                probe_cap.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&right_packed.packed_keys).as_kernel_param(),
                left_packed.key_bytes.as_kernel_param(),
                (&per_probe_offsets).as_kernel_param(),
                output_capacity.as_kernel_param(),
                (&d_output_left).as_kernel_param(),
                (&d_output_right).as_kernel_param(),
                (&d_overflow).as_kernel_param(),
            ];
            materialize_func
                .clone()
                .launch_on_stream(&cu_stream, probe_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "hash_join_probe_v2_materialize (on_stream) failed: {}",
                        e
                    ))
                })?;
        }

        rec_mat.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("csm inner: materialize commit failed: {}", e))
        })?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!("csm inner: sync (post-materialize) failed: {}", e))
        })?;

        // Step 7: gather both sides on launch_stream.
        let mut rec_gather = LaunchRecorder::new_strict(launch_stream);
        for col_idx in 0..left.columns.len() {
            let c = left
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Left column {} not found", col_idx)))?;
            rec_gather.read_column(c);
        }
        for col_idx in 0..right.columns.len() {
            let c = right
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Right column {} not found", col_idx)))?;
            rec_gather.read_column(c);
        }
        rec_gather.read(&d_output_left);
        rec_gather.read(&d_output_right);
        rec_gather
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("csm inner: gather preflight failed: {}", e)))?;
        let gathered_left = self.gather_buffer_by_indices_on_stream(
            left,
            &d_output_left,
            output_capacity,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        let gathered_right = self.gather_buffer_by_indices_on_stream(
            right,
            &d_output_right,
            output_capacity,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        rec_gather
            .commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("csm inner: gather commit failed: {}", e)))?;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        result_columns.extend(gathered_left.columns);
        result_columns.extend(gathered_right.columns);
        self.buffer_from_columns(result_columns, output_capacity as u64, combined_schema)
    }

    fn hash_join_inner_v2_count_scan_materialize_cuda_graph_recorded(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        max_output: Option<usize>,
        launch_stream: StreamId,
    ) -> Result<Option<CudaBuffer>> {
        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "hash_join_inner_v2_count_scan_materialize_cuda_graph_recorded requires a \
                 runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "hash_join_inner_v2_count_scan_materialize_cuda_graph_recorded: \
                     launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }
        if num_left == 0 || num_right == 0 || max_output == Some(0) {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema).map(Some);
        }
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
        if left_keys.len() > 4 {
            return Err(XlogError::Kernel(
                "hash_join_inner_v2_count_scan_materialize_cuda_graph_recorded: max 4 key \
                 columns supported (pack_keys constraint)"
                    .to_string(),
            ));
        }
        for (&l, &r) in left_keys.iter().zip(right_keys.iter()) {
            let lt = left.schema().column_type(l);
            let rt = right.schema().column_type(r);
            if lt != rt {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    l, lt, r, rt
                )));
            }
        }

        let logical_probe_cap = left.num_rows() as u32;
        let probe_cap = crate::cuda_graph::graph_capacity_class_u32(logical_probe_cap);
        let Some(output_capacity) =
            Self::csm_cuda_graph_output_capacity(logical_probe_cap, num_right as u32, max_output)?
        else {
            return Ok(None);
        };
        if output_capacity == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema).map(Some);
        }

        let left_packed =
            self.pack_keys_gpu_on_stream(left, left_keys, &cu_stream, launch_stream, runtime)?;
        let right_packed =
            self.pack_keys_gpu_on_stream(right, right_keys, &cu_stream, launch_stream, runtime)?;
        let graph_key = CsmCudaGraphKey::inner(
            left_keys.len(),
            left_packed.key_bytes,
            probe_cap,
            output_capacity,
        )?;
        let table = self.build_hash_table_v2_on_stream(
            &right_packed.hashes,
            num_right as u32,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        let device = self.device.inner();
        let block_size = 256u32;
        let probe_grid = probe_cap.div_ceil(block_size);
        let probe_config = LaunchConfig {
            grid_dim: (probe_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let materialize_capacity_bound: u64 = (probe_cap as u64).saturating_mul(num_right as u64);
        let materialize_capacity_u32 = materialize_capacity_bound.min(u32::MAX as u64) as u32;

        {
            let mut cache = self.csm_cuda_graph_cache.lock().map_err(|e| {
                XlogError::Kernel(format!("csm CUDA Graph cache lock poisoned: {}", e))
            })?;
            if let Some(entry) = cache.get_mut(&graph_key) {
                let result = self.launch_csm_cuda_graph_entry(
                    entry,
                    left,
                    right,
                    &left_packed,
                    &right_packed,
                    &table,
                    max_output,
                    materialize_capacity_u32,
                    probe_config,
                    &cu_stream,
                    launch_stream,
                    runtime,
                )?;
                self.csm_cuda_graph_cache_hits
                    .fetch_add(1, Ordering::Relaxed);
                return Ok(Some(result));
            }
        }

        let per_probe_count = self.memory.alloc::<u32>(probe_cap as usize)?;
        let mut per_probe_offsets = self.memory.alloc::<u32>(probe_cap as usize)?;
        let d_logical_count = self.memory.alloc::<u32>(1)?;
        let d_overflow = self.memory.alloc::<u8>(1)?;
        let d_output_left = self.memory.alloc::<u32>(output_capacity as usize)?;
        let d_output_right = self.memory.alloc::<u32>(output_capacity as usize)?;
        let mut scan_scratch = self.multiblock_scan_u32_scratch_for_len(probe_cap)?;

        let count_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2_COUNT_PER_ROW)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_probe_v2_count_per_row kernel not found".to_string())
            })?;
        let total_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_TOTAL_FROM_SCAN)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_total_from_scan kernel not found".to_string())
            })?;
        let materialize_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2_MATERIALIZE)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_probe_v2_materialize kernel not found".to_string())
            })?;

        let graph = CapturedCudaGraph::capture_on_stream(&cu_stream, || {
            // SAFETY: graph capture records these writes; replay preflight orders
            // the runtime-backed buffers before the graph is launched.
            unsafe {
                let res = cudarc::driver::sys::cuMemsetD8Async(
                    *d_overflow.device_ptr(),
                    0,
                    1,
                    cu_stream.cu_stream(),
                );
                if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "csm inner graph: cuMemsetD8Async (d_overflow) failed: {:?}",
                        res
                    )));
                }
                let res = cudarc::driver::sys::cuMemsetD8Async(
                    *d_logical_count.device_ptr(),
                    0,
                    std::mem::size_of::<u32>(),
                    cu_stream.cu_stream(),
                );
                if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "csm inner graph: cuMemsetD8Async (d_logical_count) failed: {:?}",
                        res
                    )));
                }
            }

            // SAFETY: 12-arg signature matches the PTX kernel.
            unsafe {
                count_func.clone().launch_on_stream(
                    &cu_stream,
                    probe_config,
                    (
                        &left_packed.hashes,
                        left.num_rows_device(),
                        probe_cap,
                        &table.bucket_offsets,
                        &table.bucket_counts,
                        &table.bucket_entries,
                        &table.bucket_entry_hashes,
                        table.bucket_mask,
                        &left_packed.packed_keys,
                        &right_packed.packed_keys,
                        left_packed.key_bytes,
                        &per_probe_count,
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("csm inner graph: count_per_row failed: {}", e))
            })?;

            // SAFETY: same length, both runtime-backed u32 buffers.
            unsafe {
                let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                    *per_probe_offsets.device_ptr(),
                    *per_probe_count.device_ptr(),
                    (probe_cap as usize) * std::mem::size_of::<u32>(),
                    cu_stream.cu_stream(),
                );
                if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "csm inner graph: cuMemcpyDtoDAsync (count -> offsets) failed: {:?}",
                        res
                    )));
                }
            }
            self.multiblock_scan_u32_inplace_on_stream_with_scratch(
                &mut per_probe_offsets,
                probe_cap,
                &cu_stream,
                &mut scan_scratch,
            )?;

            // SAFETY: 7-arg signature matches the PTX kernel.
            unsafe {
                total_func.clone().launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &per_probe_offsets,
                        &per_probe_count,
                        left.num_rows_device(),
                        probe_cap,
                        materialize_capacity_u32,
                        &d_logical_count,
                        &d_overflow,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("csm inner graph: total failed: {}", e)))?;

            // SAFETY: 16-arg signature; tuple form supports up to 12 elements, so use raw params.
            unsafe {
                let mut params: Vec<*mut c_void> = vec![
                    (&left_packed.hashes).as_kernel_param(),
                    left.num_rows_device().as_kernel_param(),
                    probe_cap.as_kernel_param(),
                    (&table.bucket_offsets).as_kernel_param(),
                    (&table.bucket_counts).as_kernel_param(),
                    (&table.bucket_entries).as_kernel_param(),
                    (&table.bucket_entry_hashes).as_kernel_param(),
                    table.bucket_mask.as_kernel_param(),
                    (&left_packed.packed_keys).as_kernel_param(),
                    (&right_packed.packed_keys).as_kernel_param(),
                    left_packed.key_bytes.as_kernel_param(),
                    (&per_probe_offsets).as_kernel_param(),
                    output_capacity.as_kernel_param(),
                    (&d_output_left).as_kernel_param(),
                    (&d_output_right).as_kernel_param(),
                    (&d_overflow).as_kernel_param(),
                ];
                materialize_func
                    .clone()
                    .launch_on_stream(&cu_stream, probe_config, &mut params)
                    .map_err(|e| {
                        XlogError::Kernel(format!("csm inner graph: materialize failed: {}", e))
                    })?;
            }
            Ok(())
        })?;
        let nodes = Self::csm_cuda_graph_nodes(&graph)?;
        let mut entry = CsmCudaGraphEntry {
            graph,
            nodes,
            per_probe_count,
            per_probe_offsets,
            d_logical_count,
            d_overflow,
            d_output_left,
            d_output_right,
            scan_scratch,
            probe_capacity: probe_cap,
            output_capacity,
        };
        self.csm_cuda_graph_captures.fetch_add(1, Ordering::Relaxed);

        let result = self.launch_csm_cuda_graph_entry(
            &mut entry,
            left,
            right,
            &left_packed,
            &right_packed,
            &table,
            max_output,
            materialize_capacity_u32,
            probe_config,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        self.csm_cuda_graph_cache
            .lock()
            .map_err(|e| XlogError::Kernel(format!("csm CUDA Graph cache lock poisoned: {}", e)))?
            .insert(graph_key, entry);
        Ok(Some(result))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_csm_cuda_graph_entry(
        &self,
        entry: &mut CsmCudaGraphEntry,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_packed: &PackedKeyData,
        right_packed: &PackedKeyData,
        table: &JoinHashTableV2,
        max_output: Option<usize>,
        materialize_capacity_u32: u32,
        probe_config: LaunchConfig,
        cu_stream: &cudarc::driver::CudaStream,
        launch_stream: StreamId,
        runtime: &crate::device_runtime::XlogDeviceRuntime,
    ) -> Result<CudaBuffer> {
        let mut rec_graph = LaunchRecorder::new_strict(launch_stream);
        rec_graph.read(&left_packed.hashes);
        rec_graph.read(&left_packed.packed_keys);
        rec_graph.read(&right_packed.packed_keys);
        rec_graph.read(&table.bucket_offsets);
        rec_graph.read(&table.bucket_counts);
        rec_graph.read(&table.bucket_entries);
        rec_graph.read(&table.bucket_entry_hashes);
        rec_graph.read(left.num_rows_device());
        rec_graph.read_write(&entry.per_probe_count);
        rec_graph.read_write(&entry.per_probe_offsets);
        rec_graph.read_write(&entry.d_logical_count);
        rec_graph.read_write(&entry.d_overflow);
        rec_graph.write(&entry.d_output_left);
        rec_graph.write(&entry.d_output_right);
        for level in entry.scan_scratch.levels() {
            rec_graph.read_write(level);
        }
        rec_graph
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("csm inner graph: preflight failed: {}", e)))?;

        let probe_cap = entry.probe_capacity;
        let output_capacity = entry.output_capacity;
        if probe_config.grid_dim.0 != probe_cap.div_ceil(probe_config.block_dim.0) {
            return Err(XlogError::Kernel(format!(
                "csm CUDA Graph replay probe grid mismatch: graph probe_cap={}, grid={:?}",
                probe_cap, probe_config.grid_dim
            )));
        }
        if entry.nodes.node_count < 5 {
            return Err(XlogError::Kernel(format!(
                "csm CUDA Graph replay node inventory too small: {}",
                entry.nodes.node_count
            )));
        }

        let mut count_params = entry.graph.kernel_node_params(entry.nodes.count)?;
        let mut total_params = entry.graph.kernel_node_params(entry.nodes.total)?;
        let mut materialize_params = entry.graph.kernel_node_params(entry.nodes.materialize)?;
        let mut count_args: Vec<*mut c_void> = vec![
            (&left_packed.hashes).as_kernel_param(),
            left.num_rows_device().as_kernel_param(),
            probe_cap.as_kernel_param(),
            (&table.bucket_offsets).as_kernel_param(),
            (&table.bucket_counts).as_kernel_param(),
            (&table.bucket_entries).as_kernel_param(),
            (&table.bucket_entry_hashes).as_kernel_param(),
            table.bucket_mask.as_kernel_param(),
            (&left_packed.packed_keys).as_kernel_param(),
            (&right_packed.packed_keys).as_kernel_param(),
            left_packed.key_bytes.as_kernel_param(),
            (&entry.per_probe_count).as_kernel_param(),
        ];
        let mut total_args: Vec<*mut c_void> = vec![
            (&entry.per_probe_offsets).as_kernel_param(),
            (&entry.per_probe_count).as_kernel_param(),
            left.num_rows_device().as_kernel_param(),
            probe_cap.as_kernel_param(),
            materialize_capacity_u32.as_kernel_param(),
            (&entry.d_logical_count).as_kernel_param(),
            (&entry.d_overflow).as_kernel_param(),
        ];
        let mut materialize_args: Vec<*mut c_void> = vec![
            (&left_packed.hashes).as_kernel_param(),
            left.num_rows_device().as_kernel_param(),
            probe_cap.as_kernel_param(),
            (&table.bucket_offsets).as_kernel_param(),
            (&table.bucket_counts).as_kernel_param(),
            (&table.bucket_entries).as_kernel_param(),
            (&table.bucket_entry_hashes).as_kernel_param(),
            table.bucket_mask.as_kernel_param(),
            (&left_packed.packed_keys).as_kernel_param(),
            (&right_packed.packed_keys).as_kernel_param(),
            left_packed.key_bytes.as_kernel_param(),
            (&entry.per_probe_offsets).as_kernel_param(),
            output_capacity.as_kernel_param(),
            (&entry.d_output_left).as_kernel_param(),
            (&entry.d_output_right).as_kernel_param(),
            (&entry.d_overflow).as_kernel_param(),
        ];
        count_params.kernelParams = count_args.as_mut_ptr();
        count_params.extra = std::ptr::null_mut();
        total_params.kernelParams = total_args.as_mut_ptr();
        total_params.extra = std::ptr::null_mut();
        materialize_params.kernelParams = materialize_args.as_mut_ptr();
        materialize_params.extra = std::ptr::null_mut();
        unsafe {
            entry
                .graph
                .set_kernel_node_params(entry.nodes.count, &count_params)?;
            entry
                .graph
                .set_kernel_node_params(entry.nodes.total, &total_params)?;
            entry
                .graph
                .set_kernel_node_params(entry.nodes.materialize, &materialize_params)?;
        }

        entry.graph.launch(cu_stream)?;
        self.csm_cuda_graph_launches.fetch_add(1, Ordering::Relaxed);
        rec_graph
            .commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("csm inner graph: commit failed: {}", e)))?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!("csm inner graph: sync (total read) failed: {}", e))
        })?;
        let total = self.read_join_output_count_metadata(&entry.d_logical_count)? as u64;
        let requested = max_output
            .map(|limit| (limit as u64).min(total))
            .unwrap_or(total);
        if requested == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }
        if requested > output_capacity as u64 {
            return Err(XlogError::Kernel(format!(
                "csm inner graph produced {} rows but graph output capacity is {}",
                requested, output_capacity
            )));
        }
        let output_rows = requested as u32;

        let mut rec_gather = LaunchRecorder::new_strict(launch_stream);
        for col_idx in 0..left.columns.len() {
            let c = left
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Left column {} not found", col_idx)))?;
            rec_gather.read_column(c);
        }
        for col_idx in 0..right.columns.len() {
            let c = right
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Right column {} not found", col_idx)))?;
            rec_gather.read_column(c);
        }
        rec_gather.read(&entry.d_output_left);
        rec_gather.read(&entry.d_output_right);
        rec_gather.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!("csm inner graph: gather preflight failed: {}", e))
        })?;
        let gathered_left = self.gather_buffer_by_indices_on_stream(
            left,
            &entry.d_output_left,
            output_rows,
            cu_stream,
            launch_stream,
            runtime,
        )?;
        let gathered_right = self.gather_buffer_by_indices_on_stream(
            right,
            &entry.d_output_right,
            output_rows,
            cu_stream,
            launch_stream,
            runtime,
        )?;
        rec_gather.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("csm inner graph: gather commit failed: {}", e))
        })?;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        result_columns.extend(gathered_left.columns);
        result_columns.extend(gathered_right.columns);
        self.buffer_from_columns(result_columns, output_rows as u64, combined_schema)
    }

    fn csm_cuda_graph_nodes(graph: &CapturedCudaGraph) -> Result<CsmCudaGraphNodes> {
        let nodes = graph.nodes()?;
        if nodes.len() < 5 {
            return Err(XlogError::Kernel(format!(
                "csm inner graph captured too few nodes: {}",
                nodes.len()
            )));
        }
        let kernel_nodes: Vec<_> = nodes
            .iter()
            .copied()
            .filter(|n| n.kind == CudaGraphNodeKind::Kernel)
            .collect();
        if kernel_nodes.len() < 3 {
            return Err(XlogError::Kernel(format!(
                "csm inner graph captured too few kernel nodes: {}",
                kernel_nodes.len()
            )));
        }
        Ok(CsmCudaGraphNodes {
            count: kernel_nodes[0],
            total: kernel_nodes[kernel_nodes.len() - 2],
            materialize: kernel_nodes[kernel_nodes.len() - 1],
            node_count: nodes.len(),
        })
    }

    fn csm_cuda_graph_output_capacity(
        probe_cap: u32,
        num_right: u32,
        max_output: Option<usize>,
    ) -> Result<Option<u32>> {
        if let Some(limit) = max_output {
            let limit = u32::try_from(limit).map_err(|_| {
                XlogError::Kernel(format!(
                    "csm CUDA Graph max_output {} exceeds u32::MAX",
                    limit
                ))
            })?;
            return Ok(Some(crate::cuda_graph::graph_capacity_class_u32(limit)));
        }

        let worst_case = (probe_cap as u64).saturating_mul(num_right as u64);
        if worst_case > u32::MAX as u64 {
            return Ok(None);
        }
        let auto_cap = std::env::var("XLOG_CSM_CUDA_GRAPH_AUTO_OUTPUT_CAP")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1_000_000);
        if worst_case <= auto_cap {
            Ok(Some(crate::cuda_graph::graph_capacity_class_u32(
                worst_case as u32,
            )))
        } else {
            Ok(None)
        }
    }

    /// Non-indexed LeftOuter CSM (binary-join retake sub-slice 3).
    ///
    /// Deterministic count → scan → materialize chain producing
    /// MATCHED `(left_idx, right_idx)` pairs first (Inner CSM
    /// machinery), then a per-probe-row unmatched mask
    /// (`hash_join_csm_unmatched_mask`) compacted via the
    /// recorded compact tail to produce `unmatched_left`. The
    /// final result is `inner_left | unmatched_left` per left
    /// column and `inner_right | zeros` per right column —
    /// matching the legacy `hash_join_left_outer_v2_recorded`
    /// row-ordering invariant downstream consumers depend on.
    ///
    /// This slice does NOT adopt the archived prototype's
    /// `hash_join_left_outer_count_per_row` /
    /// `hash_join_left_outer_materialize` design — those
    /// kernels interleave matched and null-sentinel rows by
    /// probe-row index, which would change the legacy
    /// LeftOuter ordering downstream consumers depend on.
    ///
    /// # Errors
    ///   * Manager not runtime-backed.
    ///   * `launch_stream` does not resolve.
    ///   * `left_keys`/`right_keys` empty, mismatched length,
    ///     or > 4 (pack_keys constraint).
    ///   * Key column type mismatch.
    ///   * Preflight / kernel / commit failures.
    pub fn hash_join_left_outer_v2_count_scan_materialize_recorded(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        max_output: Option<usize>,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        use crate::launch::LaunchRecorder;

        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "hash_join_left_outer_v2_count_scan_materialize_recorded requires a \
                 runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "csm left_outer: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        // Validation (mirrors hash_join_left_outer_v2_recorded).
        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }
        if num_left == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }
        if num_right == 0 {
            // Empty right → all left rows with zero-filled right
            // columns. Same legacy fallback as
            // `hash_join_left_outer_v2_recorded` — host-sync,
            // no launch_stream work queued.
            return self.left_outer_with_nulls(left, right);
        }
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
        if left_keys.len() > 4 {
            return Err(XlogError::Kernel(
                "csm left_outer: max 4 key columns supported (pack_keys constraint)".to_string(),
            ));
        }
        for (&l, &r) in left_keys.iter().zip(right_keys.iter()) {
            let lt = left.schema().column_type(l);
            let rt = right.schema().column_type(r);
            if lt != rt {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    l, lt, r, rt
                )));
            }
        }

        // Base `probe_cap` on the validated logical row count
        // (`num_left`) with a checked cast — `left.num_rows()` is
        // the row capacity and could over-allocate per-probe
        // scratch when `row_cap > num_left`, and silently
        // truncate if either exceeded `u32::MAX`. The earlier
        // validation already rejects `num_left > u32::MAX as
        // usize`, but make the cast explicit at the use site.
        let probe_cap = u32::try_from(num_left).map_err(|_| {
            XlogError::Kernel("csm left_outer: left row count exceeds u32::MAX".to_string())
        })?;
        let num_right_u32 = u32::try_from(num_right).map_err(|_| {
            XlogError::Kernel("csm left_outer: right row count exceeds u32::MAX".to_string())
        })?;

        // Steps 1+2: pack + table on launch_stream.
        let left_packed =
            self.pack_keys_gpu_on_stream(left, left_keys, &cu_stream, launch_stream, runtime)?;
        let right_packed =
            self.pack_keys_gpu_on_stream(right, right_keys, &cu_stream, launch_stream, runtime)?;
        let table = self.build_hash_table_v2_on_stream(
            &right_packed.hashes,
            num_right_u32,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        let device = self.device.inner();
        let block_size = 256u32;
        let probe_grid = probe_cap.div_ceil(block_size);
        let probe_config = LaunchConfig {
            grid_dim: (probe_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Phase A: count + scan + total (Inner CSM machinery).
        let per_probe_count = self.memory.alloc::<u32>(probe_cap as usize)?;
        let mut per_probe_offsets = self.memory.alloc::<u32>(probe_cap as usize)?;
        let d_logical_count = self.memory.alloc::<u32>(1)?;
        let d_overflow = self.memory.alloc::<u8>(1)?;
        // Fence alloc-ready → launch_stream for the scalars
        // before the memsets below run (memsets enqueue ahead
        // of any preflight that registers them).
        runtime
            .prepare_first_use(&d_overflow, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!("csm left_outer: prepare d_overflow failed: {}", e))
            })?;
        runtime
            .prepare_first_use(&d_logical_count, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "csm left_outer: prepare d_logical_count failed: {}",
                    e
                ))
            })?;
        // Zero-init overflow + logical_count on launch_stream.
        // SAFETY: 1-byte and 4-byte runtime-backed buffers.
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_overflow.device_ptr(),
                0,
                1,
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "csm left_outer: cuMemsetD8Async (d_overflow) failed: {:?}",
                    res
                )));
            }
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_logical_count.device_ptr(),
                0,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "csm left_outer: cuMemsetD8Async (d_logical_count) failed: {:?}",
                    res
                )));
            }
        }

        let count_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2_COUNT_PER_ROW)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_probe_v2_count_per_row kernel not found".to_string())
            })?;
        let total_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_TOTAL_FROM_SCAN)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_total_from_scan kernel not found".to_string())
            })?;

        let mut rec_count = LaunchRecorder::new_strict(launch_stream);
        rec_count.read(&left_packed.hashes);
        rec_count.read(&left_packed.packed_keys);
        rec_count.read(&right_packed.packed_keys);
        rec_count.read(&table.bucket_offsets);
        rec_count.read(&table.bucket_counts);
        rec_count.read(&table.bucket_entries);
        rec_count.read(&table.bucket_entry_hashes);
        rec_count.read(left.num_rows_device());
        rec_count.write(&per_probe_count);
        rec_count.write(&per_probe_offsets);
        rec_count.write(&d_logical_count);
        rec_count.write(&d_overflow);
        rec_count.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "csm left_outer: count/scan preflight failed: {}",
                e
            ))
        })?;

        // Step A1: count_per_row.
        // SAFETY: 12-arg signature.
        unsafe {
            count_func.clone().launch_on_stream(
                &cu_stream,
                probe_config,
                (
                    &left_packed.hashes,
                    left.num_rows_device(),
                    probe_cap,
                    &table.bucket_offsets,
                    &table.bucket_counts,
                    &table.bucket_entries,
                    &table.bucket_entry_hashes,
                    table.bucket_mask,
                    &left_packed.packed_keys,
                    &right_packed.packed_keys,
                    left_packed.key_bytes,
                    &per_probe_count,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_probe_v2_count_per_row (csm left_outer) failed: {}",
                e
            ))
        })?;

        // Step A2: dtod copy per_probe_count → per_probe_offsets,
        // then exclusive in-place scan.
        // SAFETY: same length, both runtime-backed u32.
        unsafe {
            let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                *per_probe_offsets.device_ptr(),
                *per_probe_count.device_ptr(),
                (probe_cap as usize) * std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "csm left_outer: cuMemcpyDtoDAsync (count → offsets) failed: {:?}",
                    res
                )));
            }
        }
        self.multiblock_scan_u32_inplace_on_stream(
            &mut per_probe_offsets,
            probe_cap,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        // Step A3: total_from_scan — writes d_logical_count + d_overflow.
        let materialize_capacity_bound: u64 = (probe_cap as u64).saturating_mul(num_right as u64);
        let materialize_capacity_u32 = materialize_capacity_bound.min(u32::MAX as u64) as u32;
        // SAFETY: 7-arg signature.
        unsafe {
            total_func.clone().launch_on_stream(
                &cu_stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &per_probe_offsets,
                    &per_probe_count,
                    left.num_rows_device(),
                    probe_cap,
                    materialize_capacity_u32,
                    &d_logical_count,
                    &d_overflow,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_total_from_scan (csm left_outer) failed: {}",
                e
            ))
        })?;

        rec_count.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("csm left_outer: count/scan commit failed: {}", e))
        })?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!("csm left_outer: sync (count read) failed: {}", e))
        })?;
        let inner_total = self.read_join_output_count_metadata(&d_logical_count)? as u64;
        let inner_clamped = max_output
            .map(|limit| (limit as u64).min(inner_total))
            .unwrap_or(inner_total);
        if inner_clamped > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Join produced {} matched rows which exceeds the u32 index limit",
                inner_clamped
            )));
        }
        let inner_count_u32 = inner_clamped as u32;

        // Phase B: materialize matched index pairs.
        let materialize_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2_MATERIALIZE)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_probe_v2_materialize kernel not found".to_string())
            })?;
        let d_output_left = self.memory.alloc::<u32>(inner_count_u32.max(1) as usize)?;
        let d_output_right = self.memory.alloc::<u32>(inner_count_u32.max(1) as usize)?;

        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        rec_mat.read(&left_packed.hashes);
        rec_mat.read(&left_packed.packed_keys);
        rec_mat.read(&right_packed.packed_keys);
        rec_mat.read(&table.bucket_offsets);
        rec_mat.read(&table.bucket_counts);
        rec_mat.read(&table.bucket_entries);
        rec_mat.read(&table.bucket_entry_hashes);
        rec_mat.read(&per_probe_offsets);
        rec_mat.read(left.num_rows_device());
        rec_mat.write(&d_output_left);
        rec_mat.write(&d_output_right);
        // d_overflow is consumed by the materialize kernel — recorder must own it through commit.
        rec_mat.write(&d_overflow);
        rec_mat.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "csm left_outer: materialize preflight failed: {}",
                e
            ))
        })?;
        if inner_count_u32 > 0 {
            // SAFETY: 16-arg signature; raw-param launch.
            unsafe {
                let mut params: Vec<*mut c_void> = vec![
                    (&left_packed.hashes).as_kernel_param(),
                    left.num_rows_device().as_kernel_param(),
                    probe_cap.as_kernel_param(),
                    (&table.bucket_offsets).as_kernel_param(),
                    (&table.bucket_counts).as_kernel_param(),
                    (&table.bucket_entries).as_kernel_param(),
                    (&table.bucket_entry_hashes).as_kernel_param(),
                    table.bucket_mask.as_kernel_param(),
                    (&left_packed.packed_keys).as_kernel_param(),
                    (&right_packed.packed_keys).as_kernel_param(),
                    left_packed.key_bytes.as_kernel_param(),
                    (&per_probe_offsets).as_kernel_param(),
                    inner_count_u32.as_kernel_param(),
                    (&d_output_left).as_kernel_param(),
                    (&d_output_right).as_kernel_param(),
                    (&d_overflow).as_kernel_param(),
                ];
                materialize_func
                    .clone()
                    .launch_on_stream(&cu_stream, probe_config, &mut params)
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "hash_join_probe_v2_materialize (csm left_outer) failed: {}",
                            e
                        ))
                    })?;
            }
        }
        rec_mat.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("csm left_outer: materialize commit failed: {}", e))
        })?;

        // Phase C: unmatched-left mask + recorded compact tail.
        let d_unmatched_mask = self.memory.alloc::<u8>(probe_cap as usize)?;
        let unmatched_mask_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_CSM_UNMATCHED_MASK)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_csm_unmatched_mask kernel not found".to_string())
            })?;
        let mut rec_um = LaunchRecorder::new_strict(launch_stream);
        rec_um.read(&per_probe_count);
        rec_um.read(left.num_rows_device());
        rec_um.write(&d_unmatched_mask);
        rec_um.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "csm left_outer: unmatched mask preflight failed: {}",
                e
            ))
        })?;
        // SAFETY: 4-arg signature.
        unsafe {
            unmatched_mask_func.clone().launch_on_stream(
                &cu_stream,
                probe_config,
                (
                    &per_probe_count,
                    left.num_rows_device(),
                    probe_cap,
                    &d_unmatched_mask,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_csm_unmatched_mask (on_stream) failed: {}",
                e
            ))
        })?;
        rec_um.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "csm left_outer: unmatched mask commit failed: {}",
                e
            ))
        })?;

        let unmatched_left = self.compact_buffer_by_device_mask_counted_recorded(
            left,
            &d_unmatched_mask,
            launch_stream,
        )?;
        let unmatched_rows = self.device_row_count(&unmatched_left)? as u64;
        let total_rows = (inner_count_u32 as u64) + unmatched_rows;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        if total_rows == 0 {
            return self.create_empty_buffer(combined_schema);
        }

        // Phase D: gather matched left + right (only if inner_count > 0).
        let inner_left_buf;
        let inner_right_buf;
        if inner_count_u32 > 0 {
            let mut rec_gather = LaunchRecorder::new_strict(launch_stream);
            for col_idx in 0..left.columns.len() {
                let c = left.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("Left column {} not found", col_idx))
                })?;
                rec_gather.read_column(c);
            }
            for col_idx in 0..right.columns.len() {
                let c = right.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("Right column {} not found", col_idx))
                })?;
                rec_gather.read_column(c);
            }
            rec_gather.read(&d_output_left);
            rec_gather.read(&d_output_right);
            rec_gather.preflight(runtime).map_err(|e| {
                XlogError::Kernel(format!("csm left_outer: gather preflight failed: {}", e))
            })?;
            inner_left_buf = Some(self.gather_buffer_by_indices_on_stream(
                left,
                &d_output_left,
                inner_count_u32,
                &cu_stream,
                launch_stream,
                runtime,
            )?);
            inner_right_buf = Some(self.gather_buffer_by_indices_on_stream(
                right,
                &d_output_right,
                inner_count_u32,
                &cu_stream,
                launch_stream,
                runtime,
            )?);
            rec_gather.commit(runtime).map_err(|e| {
                XlogError::Kernel(format!("csm left_outer: gather commit failed: {}", e))
            })?;
        } else {
            inner_left_buf = None;
            inner_right_buf = None;
        }

        // Phase E: per-column dtod-async concat.
        // Same step-D pattern as `hash_join_left_outer_v2_recorded`.
        let mut rec_d = LaunchRecorder::new_strict(launch_stream);
        for col_idx in 0..unmatched_left.columns.len() {
            let c = unmatched_left.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("unmatched_left col {} not found", col_idx))
            })?;
            rec_d.read_column(c);
        }
        if let Some(b) = inner_left_buf.as_ref() {
            for col_idx in 0..b.columns.len() {
                let c = b.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("inner_left col {} not found", col_idx))
                })?;
                rec_d.read_column(c);
            }
        }
        if let Some(b) = inner_right_buf.as_ref() {
            for col_idx in 0..b.columns.len() {
                let c = b.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("inner_right col {} not found", col_idx))
                })?;
                rec_d.read_column(c);
            }
        }
        rec_d.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!("csm left_outer: phase-E preflight failed: {}", e))
        })?;

        let inner_rows = inner_count_u32 as u64;
        let mut result_columns: Vec<CudaColumn> = Vec::with_capacity(combined_schema.arity());

        // Per-left-column: inner_left | unmatched_left.
        for col_idx in 0..left.arity() {
            let elem_size = left
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let inner_bytes = (inner_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| XlogError::Kernel("csm left_outer: inner_bytes overflow".into()))?;
            let unmatched_bytes = (unmatched_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("csm left_outer: unmatched_bytes overflow".into())
                })?;
            let total_bytes = inner_bytes
                .checked_add(unmatched_bytes)
                .ok_or_else(|| XlogError::Kernel("csm left_outer: total_bytes overflow".into()))?;
            let out_col = self.memory.alloc::<u8>(total_bytes)?;
            let dst_ptr = *out_col.device_ptr();
            // Fence alloc-ready → launch_stream for out_col.
            runtime
                .prepare_first_use(&out_col, launch_stream, Access::Write)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "csm left_outer: prepare left out_col {} failed: {}",
                        col_idx, e
                    ))
                })?;
            if inner_bytes > 0 {
                let src_col = inner_left_buf
                    .as_ref()
                    .expect("inner_count > 0")
                    .column(col_idx)
                    .ok_or_else(|| XlogError::Kernel("inner_left col missing".into()))?;
                // SAFETY: dtod async on cu_stream.
                unsafe {
                    let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_ptr,
                        *src_col.device_ptr(),
                        inner_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "csm left_outer: dtod inner_left col {} failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if unmatched_bytes > 0 {
                let src_col = unmatched_left.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("unmatched_left col {} not found", col_idx))
                })?;
                // SAFETY: bounded by total_bytes.
                unsafe {
                    let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_ptr + inner_bytes as u64,
                        *src_col.device_ptr(),
                        unmatched_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "csm left_outer: dtod unmatched col {} failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if let Some(b) = out_col.runtime_block() {
                runtime
                    .finish_block_use(BlockId::from_block(b), launch_stream, Access::Write)
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "csm left_outer: finish_block_use (left col {}) failed: {}",
                            col_idx, e
                        ))
                    })?;
            }
            result_columns.push(out_col.into());
        }

        // Per-right-column: inner_right | zeros.
        for col_idx in 0..right.arity() {
            let elem_size = right
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let inner_bytes = (inner_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("csm left_outer: right inner_bytes overflow".into())
                })?;
            let unmatched_bytes = (unmatched_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("csm left_outer: right unmatched_bytes overflow".into())
                })?;
            let total_bytes = inner_bytes.checked_add(unmatched_bytes).ok_or_else(|| {
                XlogError::Kernel("csm left_outer: right total_bytes overflow".into())
            })?;
            let out_col = self.memory.alloc::<u8>(total_bytes)?;
            let dst_ptr = *out_col.device_ptr();
            // Fence alloc-ready → launch_stream for out_col.
            runtime
                .prepare_first_use(&out_col, launch_stream, Access::Write)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "csm left_outer: prepare right out_col {} failed: {}",
                        col_idx, e
                    ))
                })?;
            if total_bytes > 0 {
                // SAFETY: zero-fill whole column on cu_stream.
                unsafe {
                    let res = cudarc::driver::sys::cuMemsetD8Async(
                        dst_ptr,
                        0,
                        total_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "csm left_outer: zero-fill right col {} failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if inner_bytes > 0 {
                let src_col = inner_right_buf
                    .as_ref()
                    .expect("inner_count > 0")
                    .column(col_idx)
                    .ok_or_else(|| XlogError::Kernel("inner_right col missing".into()))?;
                // SAFETY: dtod async on cu_stream.
                unsafe {
                    let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_ptr,
                        *src_col.device_ptr(),
                        inner_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "csm left_outer: dtod inner_right col {} failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if let Some(b) = out_col.runtime_block() {
                runtime
                    .finish_block_use(BlockId::from_block(b), launch_stream, Access::Write)
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "csm left_outer: finish_block_use (right col {}) failed: {}",
                            col_idx, e
                        ))
                    })?;
            }
            result_columns.push(out_col.into());
        }

        rec_d.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("csm left_outer: phase-E commit failed: {}", e))
        })?;

        // Guard the u32 metadata cast: `inner_count_u32` is
        // already u32, but `unmatched_rows` is read from the
        // device row count of `unmatched_left` and could push
        // `total_rows` past u32::MAX in pathological inputs.
        // Truncating `total_rows as u32` would corrupt the
        // host-side row-count cache and the device-side
        // `d_num_rows` scalar, leading to OOB reads in
        // downstream consumers.
        if total_rows > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "csm left_outer: output row count {} exceeds u32::MAX",
                total_rows
            )));
        }
        let total_rows_u32 = total_rows as u32;
        let d_num_rows = self.upload_device_row_count(total_rows_u32)?;
        Ok(CudaBuffer::from_columns_with_host_count(
            result_columns,
            total_rows,
            d_num_rows,
            combined_schema,
            total_rows_u32,
        ))
    }

    /// Indexed-Inner CSM (binary-join retake sub-slice 2).
    ///
    /// Same deterministic count→scan→materialize algorithm as
    /// [`Self::hash_join_inner_v2_count_scan_materialize_recorded`]
    /// but skips pack-right + table-build — the cached
    /// [`crate::provider::JoinIndexV2`] supplies
    /// `index.packed_keys` and `&index.table`. Only the probe
    /// (left) side is packed on `launch_stream`.
    ///
    /// Reuses the three CSM kernels migrated in sub-slice 1
    /// (`hash_join_probe_v2_count_per_row`,
    /// `hash_join_probe_v2_materialize`,
    /// `hash_join_total_from_scan`) — no new kernel additions.
    /// Composes `pack_keys_gpu_on_stream`,
    /// `multiblock_scan_u32_inplace_on_stream`, and
    /// `gather_buffer_by_indices_on_stream` unchanged from
    /// prior slices.
    ///
    /// Index buffers (packed_keys + 4 table buckets) are
    /// owned by the caller and recorded as reads on
    /// `launch_stream` for the count and materialize
    /// recorders — dropping the index after the call returns
    /// is correctly serialized through the runtime's
    /// record-all + wait-all event chain.
    #[allow(clippy::too_many_arguments)]
    pub fn hash_join_inner_v2_with_index_count_scan_materialize_recorded(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        index: &crate::provider::JoinIndexV2,
        max_output: Option<usize>,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        use crate::launch::LaunchRecorder;

        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "hash_join_inner_v2_with_index_count_scan_materialize_recorded requires \
                 a runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "indexed CSM inner: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        // Validation (mirror legacy hash_join_v2_with_index +
        // CSM constraints).
        let left_rows = self.device_row_count(left)?;
        let right_rows = self.device_row_count(right)?;
        if left_rows > u32::MAX as usize || right_rows > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                left_rows,
                right_rows
            )));
        }
        if left_rows == 0 || right_rows == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }
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
        if left_keys.len() > 4 {
            return Err(XlogError::Kernel(
                "indexed CSM inner: max 4 key columns supported (pack_keys constraint)".to_string(),
            ));
        }
        for (&l, &r) in left_keys.iter().zip(right_keys.iter()) {
            if l >= left.arity() {
                return Err(XlogError::Kernel(format!(
                    "Left key column index {} out of bounds (arity {})",
                    l,
                    left.arity()
                )));
            }
            if r >= right.arity() {
                return Err(XlogError::Kernel(format!(
                    "Right key column index {} out of bounds (arity {})",
                    r,
                    right.arity()
                )));
            }
            let lt = left.schema().column_type(l);
            let rt = right.schema().column_type(r);
            if lt != rt {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    l, lt, r, rt
                )));
            }
        }
        if index.right_num_rows() != right_rows as u32 {
            return Err(XlogError::Kernel(
                "Join index row count does not match right relation".to_string(),
            ));
        }
        if index.right_keys() != right_keys {
            return Err(XlogError::Kernel(
                "Join index key columns do not match requested right_keys".to_string(),
            ));
        }

        let probe_cap = left.num_rows() as u32;
        let table = &index.table;

        // Pack only LEFT on launch_stream. Build side comes
        // from the cached index.
        let left_packed =
            self.pack_keys_gpu_on_stream(left, left_keys, &cu_stream, launch_stream, runtime)?;
        if left_packed.key_bytes != index.key_bytes {
            return Err(XlogError::Kernel(
                "Join key byte width mismatch between probe and cached index".to_string(),
            ));
        }

        let device = self.device.inner();
        let block_size = 256u32;
        let probe_grid = probe_cap.div_ceil(block_size);
        let probe_config = LaunchConfig {
            grid_dim: (probe_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Allocate count + offsets + total scalar + overflow flag.
        let per_probe_count = self.memory.alloc::<u32>(probe_cap as usize)?;
        let mut per_probe_offsets = self.memory.alloc::<u32>(probe_cap as usize)?;
        let d_logical_count = self.memory.alloc::<u32>(1)?;
        let d_overflow = self.memory.alloc::<u8>(1)?;
        // Fence alloc-ready → launch_stream for both before memset.
        runtime
            .prepare_first_use(&d_overflow, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "indexed CSM inner: prepare d_overflow failed: {}",
                    e
                ))
            })?;
        runtime
            .prepare_first_use(&d_logical_count, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "indexed CSM inner: prepare d_logical_count failed: {}",
                    e
                ))
            })?;
        // Zero-init both scalars on launch_stream.
        // SAFETY: 1-byte and 4-byte runtime-backed buffers.
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_overflow.device_ptr(),
                0,
                1,
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "indexed CSM inner: cuMemsetD8Async (d_overflow) failed: {:?}",
                    res
                )));
            }
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_logical_count.device_ptr(),
                0,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "indexed CSM inner: cuMemsetD8Async (d_logical_count) failed: {:?}",
                    res
                )));
            }
        }

        // Count/scan recorder. Reads on left_packed + index
        // buffers + left.num_rows_device BEFORE preflight;
        // post-preflight fresh writes for the four newly
        // allocated buffers.
        let count_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2_COUNT_PER_ROW)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_probe_v2_count_per_row kernel not found".to_string())
            })?;
        let total_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_TOTAL_FROM_SCAN)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_total_from_scan kernel not found".to_string())
            })?;

        let mut rec_count = LaunchRecorder::new_strict(launch_stream);
        rec_count.read(&left_packed.hashes);
        rec_count.read(&left_packed.packed_keys);
        rec_count.read(&index.packed_keys);
        rec_count.read(&table.bucket_offsets);
        rec_count.read(&table.bucket_counts);
        rec_count.read(&table.bucket_entries);
        rec_count.read(&table.bucket_entry_hashes);
        rec_count.read(left.num_rows_device());
        rec_count.write(&per_probe_count);
        rec_count.write(&per_probe_offsets);
        rec_count.write(&d_logical_count);
        rec_count.write(&d_overflow);
        rec_count.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed CSM inner: count/scan preflight failed: {}",
                e
            ))
        })?;

        // Step 3: count_per_row.
        // SAFETY: 12-arg signature matches the PTX kernel.
        unsafe {
            count_func.clone().launch_on_stream(
                &cu_stream,
                probe_config,
                (
                    &left_packed.hashes,
                    left.num_rows_device(),
                    probe_cap,
                    &table.bucket_offsets,
                    &table.bucket_counts,
                    &table.bucket_entries,
                    &table.bucket_entry_hashes,
                    table.bucket_mask,
                    &left_packed.packed_keys,
                    &index.packed_keys,
                    index.key_bytes,
                    &per_probe_count,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_probe_v2_count_per_row (on_stream, indexed) failed: {}",
                e
            ))
        })?;

        // Step 4: dtod-async copy per_probe_count → per_probe_offsets,
        // then exclusive in-place scan.
        // SAFETY: same length, both runtime-backed u32 buffers.
        unsafe {
            let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                *per_probe_offsets.device_ptr(),
                *per_probe_count.device_ptr(),
                (probe_cap as usize) * std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "indexed CSM inner: cuMemcpyDtoDAsync (count → offsets) failed: {:?}",
                    res
                )));
            }
        }
        self.multiblock_scan_u32_inplace_on_stream(
            &mut per_probe_offsets,
            probe_cap,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        // Step 5: total_from_scan.
        let materialize_capacity_bound: u64 = (probe_cap as u64).saturating_mul(right_rows as u64);
        let materialize_capacity_u32 = materialize_capacity_bound.min(u32::MAX as u64) as u32;
        // SAFETY: 7-arg signature.
        unsafe {
            total_func.clone().launch_on_stream(
                &cu_stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &per_probe_offsets,
                    &per_probe_count,
                    left.num_rows_device(),
                    probe_cap,
                    materialize_capacity_u32,
                    &d_logical_count,
                    &d_overflow,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_total_from_scan (on_stream, indexed) failed: {}",
                e
            ))
        })?;

        rec_count.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed CSM inner: count/scan commit failed: {}",
                e
            ))
        })?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "indexed CSM inner: sync (total read) failed: {}",
                e
            ))
        })?;
        let total = self.read_join_output_count_metadata(&d_logical_count)? as u64;
        let requested = max_output
            .map(|limit| (limit as u64).min(total))
            .unwrap_or(total);
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
        let output_capacity = requested as u32;

        // Step 6: materialize.
        let d_output_left = self.memory.alloc::<u32>(output_capacity as usize)?;
        let d_output_right = self.memory.alloc::<u32>(output_capacity as usize)?;

        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        rec_mat.read(&left_packed.hashes);
        rec_mat.read(&left_packed.packed_keys);
        rec_mat.read(&index.packed_keys);
        rec_mat.read(&table.bucket_offsets);
        rec_mat.read(&table.bucket_counts);
        rec_mat.read(&table.bucket_entries);
        rec_mat.read(&table.bucket_entry_hashes);
        rec_mat.read(&per_probe_offsets);
        rec_mat.read(left.num_rows_device());
        rec_mat.write(&d_output_left);
        rec_mat.write(&d_output_right);
        // d_overflow is consumed by the materialize kernel — recorder must own it through commit.
        rec_mat.write(&d_overflow);
        rec_mat.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed CSM inner: materialize preflight failed: {}",
                e
            ))
        })?;

        let materialize_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2_MATERIALIZE)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_probe_v2_materialize kernel not found".to_string())
            })?;
        // SAFETY: 16-arg signature; raw-param launch path.
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                left.num_rows_device().as_kernel_param(),
                probe_cap.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&index.packed_keys).as_kernel_param(),
                index.key_bytes.as_kernel_param(),
                (&per_probe_offsets).as_kernel_param(),
                output_capacity.as_kernel_param(),
                (&d_output_left).as_kernel_param(),
                (&d_output_right).as_kernel_param(),
                (&d_overflow).as_kernel_param(),
            ];
            materialize_func
                .clone()
                .launch_on_stream(&cu_stream, probe_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "hash_join_probe_v2_materialize (on_stream, indexed) failed: {}",
                        e
                    ))
                })?;
        }

        rec_mat.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed CSM inner: materialize commit failed: {}",
                e
            ))
        })?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "indexed CSM inner: sync (post-materialize) failed: {}",
                e
            ))
        })?;

        // Step 7: gather both sides on launch_stream.
        let mut rec_gather = LaunchRecorder::new_strict(launch_stream);
        for col_idx in 0..left.columns.len() {
            let c = left
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Left column {} not found", col_idx)))?;
            rec_gather.read_column(c);
        }
        for col_idx in 0..right.columns.len() {
            let c = right
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Right column {} not found", col_idx)))?;
            rec_gather.read_column(c);
        }
        rec_gather.read(&d_output_left);
        rec_gather.read(&d_output_right);
        rec_gather.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!("indexed CSM inner: gather preflight failed: {}", e))
        })?;
        let gathered_left = self.gather_buffer_by_indices_on_stream(
            left,
            &d_output_left,
            output_capacity,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        let gathered_right = self.gather_buffer_by_indices_on_stream(
            right,
            &d_output_right,
            output_capacity,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        rec_gather.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("indexed CSM inner: gather commit failed: {}", e))
        })?;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        result_columns.extend(gathered_left.columns);
        result_columns.extend(gathered_right.columns);
        self.buffer_from_columns(result_columns, output_capacity as u64, combined_schema)
    }

    /// Indexed LeftOuter CSM (binary-join retake sub-slice 4).
    ///
    /// Combines the indexed-Inner CSM Phases A+B (probe-only
    /// pack on `launch_stream`; cached
    /// [`crate::provider::JoinIndexV2`] supplies the build
    /// side's `packed_keys` and `&index.table`) with the
    /// non-indexed LeftOuter CSM Phases C–E (per-probe
    /// unmatched-mask via `hash_join_csm_unmatched_mask` →
    /// recorded compact tail → gather matched left + right →
    /// per-column `inner | unmatched` / `inner | zeros`
    /// concat). Same row-ordering invariant as
    /// [`Self::hash_join_left_outer_v2_count_scan_materialize_recorded`]:
    /// matched rows first, unmatched-with-zero-right second.
    ///
    /// No new kernels — reuses the four already-migrated CSM
    /// kernels plus `hash_join_csm_unmatched_mask` from
    /// sub-slice 3.
    ///
    /// # Errors
    ///   * Manager not runtime-backed.
    ///   * `launch_stream` does not resolve.
    ///   * `left_keys`/`right_keys` empty, mismatched length,
    ///     or > 4 (pack_keys constraint).
    ///   * Key column type mismatch.
    ///   * `index.right_num_rows()` mismatches the right
    ///     buffer's logical row count.
    ///   * `index.right_keys()` mismatches the requested
    ///     `right_keys`.
    ///   * `left_packed.key_bytes` mismatches `index.key_bytes`.
    ///   * Preflight / kernel / commit failures.
    #[allow(clippy::too_many_arguments)]
    pub fn hash_join_left_outer_v2_with_index_count_scan_materialize_recorded(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        index: &crate::provider::JoinIndexV2,
        max_output: Option<usize>,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        use crate::launch::LaunchRecorder;

        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "hash_join_left_outer_v2_with_index_count_scan_materialize_recorded requires \
                 a runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "indexed csm left_outer: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        // Validation (mirror hash_join_inner_v2_with_index_count_scan_materialize_recorded
        // + non-indexed LeftOuter CSM).
        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }
        if num_left == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }
        if num_right == 0 {
            // Empty right → all left rows with zero-filled
            // right columns. Same legacy fallback as the
            // non-indexed LeftOuter CSM and
            // `hash_join_left_outer_v2_recorded`.
            return self.left_outer_with_nulls(left, right);
        }
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
        if left_keys.len() > 4 {
            return Err(XlogError::Kernel(
                "indexed csm left_outer: max 4 key columns supported (pack_keys constraint)"
                    .to_string(),
            ));
        }
        for (&l, &r) in left_keys.iter().zip(right_keys.iter()) {
            if l >= left.arity() {
                return Err(XlogError::Kernel(format!(
                    "Left key column index {} out of bounds (arity {})",
                    l,
                    left.arity()
                )));
            }
            if r >= right.arity() {
                return Err(XlogError::Kernel(format!(
                    "Right key column index {} out of bounds (arity {})",
                    r,
                    right.arity()
                )));
            }
            let lt = left.schema().column_type(l);
            let rt = right.schema().column_type(r);
            if lt != rt {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    l, lt, r, rt
                )));
            }
        }
        if index.right_num_rows() != num_right as u32 {
            return Err(XlogError::Kernel(
                "Join index row count does not match right relation".to_string(),
            ));
        }
        if index.right_keys() != right_keys {
            return Err(XlogError::Kernel(
                "Join index key columns do not match requested right_keys".to_string(),
            ));
        }

        // Base `probe_cap` on the validated logical row count.
        let probe_cap = u32::try_from(num_left).map_err(|_| {
            XlogError::Kernel("indexed csm left_outer: left row count exceeds u32::MAX".to_string())
        })?;

        let table = &index.table;

        // Pack only LEFT on launch_stream. Build side comes from
        // the cached index.
        let left_packed =
            self.pack_keys_gpu_on_stream(left, left_keys, &cu_stream, launch_stream, runtime)?;
        if left_packed.key_bytes != index.key_bytes {
            return Err(XlogError::Kernel(
                "Join key byte width mismatch between probe and cached index".to_string(),
            ));
        }

        let device = self.device.inner();
        let block_size = 256u32;
        let probe_grid = probe_cap.div_ceil(block_size);
        let probe_config = LaunchConfig {
            grid_dim: (probe_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Phase A: count + scan + total.
        let per_probe_count = self.memory.alloc::<u32>(probe_cap as usize)?;
        let mut per_probe_offsets = self.memory.alloc::<u32>(probe_cap as usize)?;
        let d_logical_count = self.memory.alloc::<u32>(1)?;
        let d_overflow = self.memory.alloc::<u8>(1)?;
        runtime
            .prepare_first_use(&d_overflow, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "indexed csm left_outer: prepare d_overflow failed: {}",
                    e
                ))
            })?;
        runtime
            .prepare_first_use(&d_logical_count, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "indexed csm left_outer: prepare d_logical_count failed: {}",
                    e
                ))
            })?;
        // Zero-init scalars on launch_stream.
        // SAFETY: 1-byte and 4-byte runtime-backed buffers.
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_overflow.device_ptr(),
                0,
                1,
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "indexed csm left_outer: cuMemsetD8Async (d_overflow) failed: {:?}",
                    res
                )));
            }
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_logical_count.device_ptr(),
                0,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "indexed csm left_outer: cuMemsetD8Async (d_logical_count) failed: {:?}",
                    res
                )));
            }
        }

        let count_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2_COUNT_PER_ROW)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_probe_v2_count_per_row kernel not found".to_string())
            })?;
        let total_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_TOTAL_FROM_SCAN)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_total_from_scan kernel not found".to_string())
            })?;

        let mut rec_count = LaunchRecorder::new_strict(launch_stream);
        rec_count.read(&left_packed.hashes);
        rec_count.read(&left_packed.packed_keys);
        rec_count.read(&index.packed_keys);
        rec_count.read(&table.bucket_offsets);
        rec_count.read(&table.bucket_counts);
        rec_count.read(&table.bucket_entries);
        rec_count.read(&table.bucket_entry_hashes);
        rec_count.read(left.num_rows_device());
        rec_count.write(&per_probe_count);
        rec_count.write(&per_probe_offsets);
        rec_count.write(&d_logical_count);
        rec_count.write(&d_overflow);
        rec_count.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed csm left_outer: count/scan preflight failed: {}",
                e
            ))
        })?;

        // Step A1: count_per_row.
        // SAFETY: 12-arg signature.
        unsafe {
            count_func.clone().launch_on_stream(
                &cu_stream,
                probe_config,
                (
                    &left_packed.hashes,
                    left.num_rows_device(),
                    probe_cap,
                    &table.bucket_offsets,
                    &table.bucket_counts,
                    &table.bucket_entries,
                    &table.bucket_entry_hashes,
                    table.bucket_mask,
                    &left_packed.packed_keys,
                    &index.packed_keys,
                    index.key_bytes,
                    &per_probe_count,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_probe_v2_count_per_row (indexed csm left_outer) failed: {}",
                e
            ))
        })?;

        // Step A2: dtod copy + exclusive scan.
        // SAFETY: same length, both runtime-backed u32.
        unsafe {
            let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                *per_probe_offsets.device_ptr(),
                *per_probe_count.device_ptr(),
                (probe_cap as usize) * std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "indexed csm left_outer: cuMemcpyDtoDAsync (count → offsets) failed: {:?}",
                    res
                )));
            }
        }
        self.multiblock_scan_u32_inplace_on_stream(
            &mut per_probe_offsets,
            probe_cap,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        // Step A3: total_from_scan — writes d_logical_count + d_overflow.
        let materialize_capacity_bound: u64 = (probe_cap as u64).saturating_mul(num_right as u64);
        let materialize_capacity_u32 = materialize_capacity_bound.min(u32::MAX as u64) as u32;
        // SAFETY: 7-arg signature.
        unsafe {
            total_func.clone().launch_on_stream(
                &cu_stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &per_probe_offsets,
                    &per_probe_count,
                    left.num_rows_device(),
                    probe_cap,
                    materialize_capacity_u32,
                    &d_logical_count,
                    &d_overflow,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_total_from_scan (indexed csm left_outer) failed: {}",
                e
            ))
        })?;

        rec_count.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed csm left_outer: count/scan commit failed: {}",
                e
            ))
        })?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "indexed csm left_outer: sync (count read) failed: {}",
                e
            ))
        })?;
        let inner_total = self.read_join_output_count_metadata(&d_logical_count)? as u64;
        let inner_clamped = max_output
            .map(|limit| (limit as u64).min(inner_total))
            .unwrap_or(inner_total);
        if inner_clamped > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Join produced {} matched rows which exceeds the u32 index limit",
                inner_clamped
            )));
        }
        let inner_count_u32 = inner_clamped as u32;

        // Phase B: materialize matched index pairs.
        let materialize_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2_MATERIALIZE)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_probe_v2_materialize kernel not found".to_string())
            })?;
        let d_output_left = self.memory.alloc::<u32>(inner_count_u32.max(1) as usize)?;
        let d_output_right = self.memory.alloc::<u32>(inner_count_u32.max(1) as usize)?;

        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        rec_mat.read(&left_packed.hashes);
        rec_mat.read(&left_packed.packed_keys);
        rec_mat.read(&index.packed_keys);
        rec_mat.read(&table.bucket_offsets);
        rec_mat.read(&table.bucket_counts);
        rec_mat.read(&table.bucket_entries);
        rec_mat.read(&table.bucket_entry_hashes);
        rec_mat.read(&per_probe_offsets);
        rec_mat.read(left.num_rows_device());
        rec_mat.write(&d_output_left);
        rec_mat.write(&d_output_right);
        // d_overflow is consumed by the materialize kernel — recorder must own it through commit.
        rec_mat.write(&d_overflow);
        rec_mat.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed csm left_outer: materialize preflight failed: {}",
                e
            ))
        })?;
        if inner_count_u32 > 0 {
            // SAFETY: 16-arg signature; raw-param launch.
            unsafe {
                let mut params: Vec<*mut c_void> = vec![
                    (&left_packed.hashes).as_kernel_param(),
                    left.num_rows_device().as_kernel_param(),
                    probe_cap.as_kernel_param(),
                    (&table.bucket_offsets).as_kernel_param(),
                    (&table.bucket_counts).as_kernel_param(),
                    (&table.bucket_entries).as_kernel_param(),
                    (&table.bucket_entry_hashes).as_kernel_param(),
                    table.bucket_mask.as_kernel_param(),
                    (&left_packed.packed_keys).as_kernel_param(),
                    (&index.packed_keys).as_kernel_param(),
                    index.key_bytes.as_kernel_param(),
                    (&per_probe_offsets).as_kernel_param(),
                    inner_count_u32.as_kernel_param(),
                    (&d_output_left).as_kernel_param(),
                    (&d_output_right).as_kernel_param(),
                    (&d_overflow).as_kernel_param(),
                ];
                materialize_func
                    .clone()
                    .launch_on_stream(&cu_stream, probe_config, &mut params)
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "hash_join_probe_v2_materialize (indexed csm left_outer) failed: {}",
                            e
                        ))
                    })?;
            }
        }
        rec_mat.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed csm left_outer: materialize commit failed: {}",
                e
            ))
        })?;

        // Phase C: unmatched-left mask + recorded compact tail.
        let d_unmatched_mask = self.memory.alloc::<u8>(probe_cap as usize)?;
        let unmatched_mask_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_CSM_UNMATCHED_MASK)
            .ok_or_else(|| {
                XlogError::Kernel("hash_join_csm_unmatched_mask kernel not found".to_string())
            })?;
        let mut rec_um = LaunchRecorder::new_strict(launch_stream);
        rec_um.read(&per_probe_count);
        rec_um.read(left.num_rows_device());
        rec_um.write(&d_unmatched_mask);
        rec_um.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed csm left_outer: unmatched mask preflight failed: {}",
                e
            ))
        })?;
        // SAFETY: 4-arg signature.
        unsafe {
            unmatched_mask_func.clone().launch_on_stream(
                &cu_stream,
                probe_config,
                (
                    &per_probe_count,
                    left.num_rows_device(),
                    probe_cap,
                    &d_unmatched_mask,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_csm_unmatched_mask (indexed csm left_outer) failed: {}",
                e
            ))
        })?;
        rec_um.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed csm left_outer: unmatched mask commit failed: {}",
                e
            ))
        })?;

        let unmatched_left = self.compact_buffer_by_device_mask_counted_recorded(
            left,
            &d_unmatched_mask,
            launch_stream,
        )?;
        let unmatched_rows = self.device_row_count(&unmatched_left)? as u64;
        let total_rows = (inner_count_u32 as u64) + unmatched_rows;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        if total_rows == 0 {
            return self.create_empty_buffer(combined_schema);
        }

        // Phase D: gather matched left + right (only if inner_count > 0).
        let inner_left_buf;
        let inner_right_buf;
        if inner_count_u32 > 0 {
            let mut rec_gather = LaunchRecorder::new_strict(launch_stream);
            for col_idx in 0..left.columns.len() {
                let c = left.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("Left column {} not found", col_idx))
                })?;
                rec_gather.read_column(c);
            }
            for col_idx in 0..right.columns.len() {
                let c = right.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("Right column {} not found", col_idx))
                })?;
                rec_gather.read_column(c);
            }
            rec_gather.read(&d_output_left);
            rec_gather.read(&d_output_right);
            rec_gather.preflight(runtime).map_err(|e| {
                XlogError::Kernel(format!(
                    "indexed csm left_outer: gather preflight failed: {}",
                    e
                ))
            })?;
            inner_left_buf = Some(self.gather_buffer_by_indices_on_stream(
                left,
                &d_output_left,
                inner_count_u32,
                &cu_stream,
                launch_stream,
                runtime,
            )?);
            inner_right_buf = Some(self.gather_buffer_by_indices_on_stream(
                right,
                &d_output_right,
                inner_count_u32,
                &cu_stream,
                launch_stream,
                runtime,
            )?);
            rec_gather.commit(runtime).map_err(|e| {
                XlogError::Kernel(format!(
                    "indexed csm left_outer: gather commit failed: {}",
                    e
                ))
            })?;
        } else {
            inner_left_buf = None;
            inner_right_buf = None;
        }

        // Phase E: per-column dtod-async concat.
        let mut rec_d = LaunchRecorder::new_strict(launch_stream);
        for col_idx in 0..unmatched_left.columns.len() {
            let c = unmatched_left.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("unmatched_left col {} not found", col_idx))
            })?;
            rec_d.read_column(c);
        }
        if let Some(b) = inner_left_buf.as_ref() {
            for col_idx in 0..b.columns.len() {
                let c = b.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("inner_left col {} not found", col_idx))
                })?;
                rec_d.read_column(c);
            }
        }
        if let Some(b) = inner_right_buf.as_ref() {
            for col_idx in 0..b.columns.len() {
                let c = b.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("inner_right col {} not found", col_idx))
                })?;
                rec_d.read_column(c);
            }
        }
        rec_d.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed csm left_outer: phase-E preflight failed: {}",
                e
            ))
        })?;

        let inner_rows = inner_count_u32 as u64;
        let mut result_columns: Vec<CudaColumn> = Vec::with_capacity(combined_schema.arity());

        // Per-left-column: inner_left | unmatched_left.
        for col_idx in 0..left.arity() {
            let elem_size = left
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let inner_bytes = (inner_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("indexed csm left_outer: inner_bytes overflow".into())
                })?;
            let unmatched_bytes = (unmatched_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("indexed csm left_outer: unmatched_bytes overflow".into())
                })?;
            let total_bytes = inner_bytes.checked_add(unmatched_bytes).ok_or_else(|| {
                XlogError::Kernel("indexed csm left_outer: total_bytes overflow".into())
            })?;
            let out_col = self.memory.alloc::<u8>(total_bytes)?;
            let dst_ptr = *out_col.device_ptr();
            runtime
                .prepare_first_use(&out_col, launch_stream, Access::Write)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "indexed csm left_outer: prepare left out_col {} failed: {}",
                        col_idx, e
                    ))
                })?;
            if inner_bytes > 0 {
                let src_col = inner_left_buf
                    .as_ref()
                    .expect("inner_count > 0")
                    .column(col_idx)
                    .ok_or_else(|| XlogError::Kernel("inner_left col missing".into()))?;
                // SAFETY: dtod async on cu_stream.
                unsafe {
                    let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_ptr,
                        *src_col.device_ptr(),
                        inner_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "indexed csm left_outer: dtod inner_left col {} failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if unmatched_bytes > 0 {
                let src_col = unmatched_left.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("unmatched_left col {} not found", col_idx))
                })?;
                // SAFETY: bounded by total_bytes.
                unsafe {
                    let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_ptr + inner_bytes as u64,
                        *src_col.device_ptr(),
                        unmatched_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "indexed csm left_outer: dtod unmatched col {} failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if let Some(b) = out_col.runtime_block() {
                runtime
                    .finish_block_use(BlockId::from_block(b), launch_stream, Access::Write)
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "indexed csm left_outer: finish_block_use (left col {}) failed: {}",
                            col_idx, e
                        ))
                    })?;
            }
            result_columns.push(out_col.into());
        }

        // Per-right-column: inner_right | zeros.
        for col_idx in 0..right.arity() {
            let elem_size = right
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let inner_bytes = (inner_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("indexed csm left_outer: right inner_bytes overflow".into())
                })?;
            let unmatched_bytes = (unmatched_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "indexed csm left_outer: right unmatched_bytes overflow".into(),
                    )
                })?;
            let total_bytes = inner_bytes.checked_add(unmatched_bytes).ok_or_else(|| {
                XlogError::Kernel("indexed csm left_outer: right total_bytes overflow".into())
            })?;
            let out_col = self.memory.alloc::<u8>(total_bytes)?;
            let dst_ptr = *out_col.device_ptr();
            runtime
                .prepare_first_use(&out_col, launch_stream, Access::Write)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "indexed csm left_outer: prepare right out_col {} failed: {}",
                        col_idx, e
                    ))
                })?;
            if total_bytes > 0 {
                // SAFETY: zero-fill whole column on cu_stream.
                unsafe {
                    let res = cudarc::driver::sys::cuMemsetD8Async(
                        dst_ptr,
                        0,
                        total_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "indexed csm left_outer: zero-fill right col {} failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if inner_bytes > 0 {
                let src_col = inner_right_buf
                    .as_ref()
                    .expect("inner_count > 0")
                    .column(col_idx)
                    .ok_or_else(|| XlogError::Kernel("inner_right col missing".into()))?;
                // SAFETY: dtod async on cu_stream.
                unsafe {
                    let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_ptr,
                        *src_col.device_ptr(),
                        inner_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "indexed csm left_outer: dtod inner_right col {} failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if let Some(b) = out_col.runtime_block() {
                runtime
                    .finish_block_use(BlockId::from_block(b), launch_stream, Access::Write)
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "indexed csm left_outer: finish_block_use (right col {}) failed: {}",
                            col_idx, e
                        ))
                    })?;
            }
            result_columns.push(out_col.into());
        }

        rec_d.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed csm left_outer: phase-E commit failed: {}",
                e
            ))
        })?;

        // Guard u32 metadata cast.
        if total_rows > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "indexed csm left_outer: output row count {} exceeds u32::MAX",
                total_rows
            )));
        }
        let total_rows_u32 = total_rows as u32;
        let d_num_rows = self.upload_device_row_count(total_rows_u32)?;
        Ok(CudaBuffer::from_columns_with_host_count(
            result_columns,
            total_rows,
            d_num_rows,
            combined_schema,
            total_rows_u32,
        ))
    }

    /// Strict-recorder, launch_stream-routed variant of
    /// `hash_join_v2`. Covers all four join types
    /// (`Inner` / `Semi` / `Anti` / `LeftOuter`) via dedicated
    /// per-type recorded methods.
    ///
    /// When [`Self::use_recorded_csm_env`] is on, `Inner` and
    /// `LeftOuter` route through the CSM (count-scan-materialize)
    /// methods; otherwise they route through the legacy recorded
    /// methods. `Semi` / `Anti` always route through their
    /// existing recorded methods — no CSM implementation exists
    /// for them. All eligibility checks (runtime-backed manager,
    /// ≤4 keys, key-type match, row-count caps) are validated
    /// upstream by the public `hash_join_v2_with_limit` and inside
    /// each per-type method.
    #[allow(clippy::too_many_arguments)]
    pub fn hash_join_v2_recorded(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        join_type: JoinType,
        max_output: Option<usize>,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let csm_on = Self::use_recorded_csm_env();
        match join_type {
            JoinType::Inner => {
                if csm_on {
                    self.csm_invocations.fetch_add(1, Ordering::Relaxed);
                    self.hash_join_inner_v2_count_scan_materialize_recorded(
                        left,
                        right,
                        left_keys,
                        right_keys,
                        max_output,
                        launch_stream,
                    )
                } else {
                    self.hash_join_inner_v2_recorded(
                        left,
                        right,
                        left_keys,
                        right_keys,
                        max_output,
                        launch_stream,
                    )
                }
            }
            JoinType::Semi => self.hash_join_semi_or_anti_v2_recorded(
                left,
                right,
                left_keys,
                right_keys,
                false,
                launch_stream,
            ),
            JoinType::Anti => self.hash_join_semi_or_anti_v2_recorded(
                left,
                right,
                left_keys,
                right_keys,
                true,
                launch_stream,
            ),
            JoinType::LeftOuter => {
                if csm_on {
                    self.csm_invocations.fetch_add(1, Ordering::Relaxed);
                    self.hash_join_left_outer_v2_count_scan_materialize_recorded(
                        left,
                        right,
                        left_keys,
                        right_keys,
                        max_output,
                        launch_stream,
                    )
                } else {
                    self.hash_join_left_outer_v2_recorded(
                        left,
                        right,
                        left_keys,
                        right_keys,
                        max_output,
                        launch_stream,
                    )
                }
            }
        }
    }

    /// Strict-recorder LeftOuter hash join (slice #7C).
    ///
    /// Mirrors the legacy `hash_join_left_outer_impl` chain on
    /// `launch_stream`:
    ///   1. pack keys both sides + build hash table on stream
    ///      (via the slice #6 + #7A helpers).
    ///   2. SEMI kernel → `d_has_match` mask.
    ///   3. PROBE count + materialize → inner-join indices.
    ///   4. `mask_not` → `d_no_match`; recorded compact tail
    ///      filters `left` to `unmatched_left`.
    ///   5. Gather inner left + inner right on stream.
    ///   6. Concatenate: per-left-column `inner | unmatched`,
    ///      per-right-column `inner | zeros`. All copies and
    ///      zero-fills go on `cu_stream` via
    ///      `cuMemcpyDtoDAsync_v2` / `cuMemsetD8Async`.
    ///
    /// Empty-right edge case keeps the legacy
    /// `left_outer_with_nulls` (synchronous on default stream)
    /// — no launch_stream work is queued for that path, which
    /// is the correct semantic.
    fn hash_join_left_outer_v2_recorded(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        max_output: Option<usize>,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        use crate::launch::LaunchRecorder;

        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "hash_join_v2_recorded (left_outer) requires a runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                "hash_join_v2_recorded (left_outer): launch_stream StreamId({}) does not resolve",
                launch_stream.0
            ))
            })?;

        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }
        if num_left == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }
        if num_right == 0 {
            // Empty right: all left rows, null right columns.
            // Falls back to legacy default-stream path. No
            // launch_stream work is queued; caller drops are
            // safe because legacy syncs before returning.
            return self.left_outer_with_nulls(left, right);
        }
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
        if left_keys.len() > 4 {
            return Err(XlogError::Kernel(
                "hash_join_v2_recorded (left_outer): max 4 key columns supported \
                 (pack_keys constraint)"
                    .to_string(),
            ));
        }
        for (&l, &r) in left_keys.iter().zip(right_keys.iter()) {
            let lt = left.schema().column_type(l);
            let rt = right.schema().column_type(r);
            if lt != rt {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    l, lt, r, rt
                )));
            }
        }

        let num_left = num_left as u32;
        let num_right = num_right as u32;

        let left_packed =
            self.pack_keys_gpu_on_stream(left, left_keys, &cu_stream, launch_stream, runtime)?;
        let right_packed =
            self.pack_keys_gpu_on_stream(right, right_keys, &cu_stream, launch_stream, runtime)?;
        let table = self.build_hash_table_v2_on_stream(
            &right_packed.hashes,
            num_right,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        let device = self.device.inner();
        let block_size = 256u32;
        let grid_size = num_left.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Step A: SEMI mask (d_has_match) — used for unmatched
        // detection later. Plus PROBE count + materialize for
        // inner-join row indices.
        let d_has_match = self.memory.alloc::<u8>(num_left as usize)?;
        let d_count_only = self.memory.alloc::<u32>(1)?;
        let d_dummy_left = self.memory.alloc::<u32>(1)?;
        let d_dummy_right = self.memory.alloc::<u32>(1)?;
        // Fence alloc-ready → launch_stream for d_count_only
        // before the memset writes it (the memset runs ahead
        // of the recorder's preflight below).
        runtime
            .prepare_first_use(&d_count_only, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "left_outer recorded: prepare d_count_only failed: {}",
                    e
                ))
            })?;
        // SAFETY: runtime-backed 4-byte buffer.
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_count_only.device_ptr(),
                0,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "cuMemsetD8Async (left_outer d_count_only) failed: {:?}",
                    res
                )));
            }
        }

        let semi_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_SEMI)
            .ok_or_else(|| XlogError::Kernel("hash_join_semi kernel not found".to_string()))?;
        let probe_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2)
            .ok_or_else(|| XlogError::Kernel("hash_join_probe_v2 kernel not found".to_string()))?;

        let mut rec_a = LaunchRecorder::new_strict(launch_stream);
        rec_a.read(&left_packed.hashes);
        rec_a.read(&left_packed.packed_keys);
        rec_a.read(&right_packed.packed_keys);
        rec_a.read(&table.bucket_offsets);
        rec_a.read(&table.bucket_counts);
        rec_a.read(&table.bucket_entries);
        rec_a.read(&table.bucket_entry_hashes);
        rec_a.write(&d_has_match);
        rec_a.write(&d_count_only);
        rec_a.write(&d_dummy_left);
        rec_a.write(&d_dummy_right);
        rec_a.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_v2_recorded (left_outer): semi/count preflight failed: {}",
                e
            ))
        })?;

        // SAFETY: hash_join_semi 11-arg signature.
        unsafe {
            semi_func.clone().launch_on_stream(
                &cu_stream,
                cfg,
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
        }
        .map_err(|e| XlogError::Kernel(format!("hash_join_semi (on_stream) failed: {}", e)))?;

        let max_output_count_only = 0u32;
        // SAFETY: hash_join_probe_v2 14-arg signature; raw-param launch.
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&right_packed.packed_keys).as_kernel_param(),
                left_packed.key_bytes.as_kernel_param(),
                (&d_dummy_left).as_kernel_param(),
                (&d_dummy_right).as_kernel_param(),
                (&d_count_only).as_kernel_param(),
                max_output_count_only.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch_on_stream(&cu_stream, cfg, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "hash_join_probe_v2 (count, on_stream, left_outer) failed: {}",
                        e
                    ))
                })?;
        }

        rec_a.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_v2_recorded (left_outer): semi/count commit failed: {}",
                e
            ))
        })?;

        // Sync + read inner-count.
        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_v2_recorded (left_outer): sync (count read) failed: {}",
                e
            ))
        })?;
        let full_inner = self.read_join_output_count_metadata(&d_count_only)? as u64;
        let requested_inner = max_output
            .map(|limit| (limit as u64).min(full_inner))
            .unwrap_or(full_inner);
        if requested_inner > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Join produced {} rows which exceeds the u32 index limit",
                requested_inner
            )));
        }
        let max_output_u32 = requested_inner as u32;
        let alloc_len = (requested_inner.max(1)) as usize;

        let d_output_left = self.memory.alloc::<u32>(alloc_len)?;
        let d_output_right = self.memory.alloc::<u32>(alloc_len)?;
        let d_output_count = self.memory.alloc::<u32>(1)?;
        // Fence alloc-ready → launch_stream for d_output_count
        // before the memset (memset runs ahead of preflight).
        runtime
            .prepare_first_use(&d_output_count, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "left_outer recorded: prepare d_output_count failed: {}",
                    e
                ))
            })?;
        // SAFETY: runtime-backed 4-byte buffer.
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_output_count.device_ptr(),
                0,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "cuMemsetD8Async (left_outer d_output_count) failed: {:?}",
                    res
                )));
            }
        }

        let mut rec_b = LaunchRecorder::new_strict(launch_stream);
        rec_b.read(&left_packed.hashes);
        rec_b.read(&left_packed.packed_keys);
        rec_b.read(&right_packed.packed_keys);
        rec_b.read(&table.bucket_offsets);
        rec_b.read(&table.bucket_counts);
        rec_b.read(&table.bucket_entries);
        rec_b.read(&table.bucket_entry_hashes);
        rec_b.write(&d_output_left);
        rec_b.write(&d_output_right);
        rec_b.write(&d_output_count);
        rec_b.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_v2_recorded (left_outer): materialize preflight failed: {}",
                e
            ))
        })?;

        // SAFETY: hash_join_probe_v2 14-arg materialize.
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&right_packed.packed_keys).as_kernel_param(),
                left_packed.key_bytes.as_kernel_param(),
                (&d_output_left).as_kernel_param(),
                (&d_output_right).as_kernel_param(),
                (&d_output_count).as_kernel_param(),
                max_output_u32.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch_on_stream(&cu_stream, cfg, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "hash_join_probe_v2 (materialize, on_stream, left_outer) failed: {}",
                        e
                    ))
                })?;
        }

        rec_b.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_v2_recorded (left_outer): materialize commit failed: {}",
                e
            ))
        })?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_v2_recorded (left_outer): sync (materialize read) failed: {}",
                e
            ))
        })?;
        let inner_count = self
            .read_join_output_count_metadata(&d_output_count)?
            .min(max_output_u32);

        // Step B: mask_not(d_has_match) → d_no_match, then
        // recorded compact tail filters `left` to unmatched_left.
        let d_no_match = self.memory.alloc::<u8>(num_left as usize)?;
        let mask_not_fn = device
            .get_func(FILTER_MODULE, filter_kernels::MASK_NOT)
            .ok_or_else(|| XlogError::Kernel("mask_not kernel not found".to_string()))?;

        let mut rec_c = LaunchRecorder::new_strict(launch_stream);
        rec_c.read(&d_has_match);
        rec_c.write(&d_no_match);
        rec_c.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_v2_recorded (left_outer): mask_not preflight failed: {}",
                e
            ))
        })?;
        // SAFETY: mask_not(in_mask, out_mask, num_rows)
        unsafe {
            mask_not_fn.clone().launch_on_stream(
                &cu_stream,
                cfg,
                (&d_has_match, &d_no_match, num_left),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("mask_not (on_stream) failed: {}", e)))?;
        rec_c.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_v2_recorded (left_outer): mask_not commit failed: {}",
                e
            ))
        })?;

        let unmatched_left =
            self.compact_buffer_by_device_mask_counted_recorded(left, &d_no_match, launch_stream)?;
        let unmatched_rows = self.device_row_count(&unmatched_left)? as u64;
        let total_rows = (inner_count as u64) + unmatched_rows;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        if total_rows == 0 {
            return self.create_empty_buffer(combined_schema);
        }

        // Step C: gather inner-left and inner-right on stream
        // (only when there are inner matches). Wrap the
        // gather kernels in a recorder so reads of
        // `left.column[i]`, `right.column[i]`, and the index
        // buffers `d_output_{left,right}` are registered on
        // launch_stream — without it, dropping `left` /
        // `right` after this method returns could race the
        // still-pending gather reads.
        let inner_count_u32 = inner_count;
        let inner_left_buf;
        let inner_right_buf;
        if inner_count > 0 {
            let mut rec_gather = LaunchRecorder::new_strict(launch_stream);
            for col_idx in 0..left.columns.len() {
                let c = left.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("Left column {} not found", col_idx))
                })?;
                rec_gather.read_column(c);
            }
            for col_idx in 0..right.columns.len() {
                let c = right.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("Right column {} not found", col_idx))
                })?;
                rec_gather.read_column(c);
            }
            rec_gather.read(&d_output_left);
            rec_gather.read(&d_output_right);
            rec_gather.preflight(runtime).map_err(|e| {
                XlogError::Kernel(format!(
                    "hash_join_v2_recorded (left_outer): gather preflight failed: {}",
                    e
                ))
            })?;
            inner_left_buf = Some(self.gather_buffer_by_indices_on_stream(
                left,
                &d_output_left,
                inner_count_u32,
                &cu_stream,
                launch_stream,
                runtime,
            )?);
            inner_right_buf = Some(self.gather_buffer_by_indices_on_stream(
                right,
                &d_output_right,
                inner_count_u32,
                &cu_stream,
                launch_stream,
                runtime,
            )?);
            rec_gather.commit(runtime).map_err(|e| {
                XlogError::Kernel(format!(
                    "hash_join_v2_recorded (left_outer): gather commit failed: {}",
                    e
                ))
            })?;
        } else {
            inner_left_buf = None;
            inner_right_buf = None;
        }

        // Step D: concatenate per-column on launch_stream.
        // Left columns: inner_left | unmatched_left.
        // Right columns: inner_right | zeros.
        //
        // The dtod copies queue AFTER the gather/compact
        // commits, so the events those commits recorded on
        // the source buffers (`unmatched_left.column[i]`,
        // `inner_*_buf.column[i]`) do NOT cover the still-
        // pending dtod copies. We open a new recorder around
        // the entire concat block, record reads on every
        // source column, preflight, run the dtod copies and
        // zero-fills, and commit. The commit's event is
        // recorded AFTER all dtod copies are queued — so a
        // subsequent drop of `unmatched_left` /
        // `inner_*_buf` correctly waits for the dtod copies
        // to complete.
        let mut rec_d = LaunchRecorder::new_strict(launch_stream);
        for col_idx in 0..unmatched_left.columns.len() {
            let c = unmatched_left.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("unmatched_left col {} not found", col_idx))
            })?;
            rec_d.read_column(c);
        }
        if let Some(b) = inner_left_buf.as_ref() {
            for col_idx in 0..b.columns.len() {
                let c = b.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("inner_left col {} not found", col_idx))
                })?;
                rec_d.read_column(c);
            }
        }
        if let Some(b) = inner_right_buf.as_ref() {
            for col_idx in 0..b.columns.len() {
                let c = b.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("inner_right col {} not found", col_idx))
                })?;
                rec_d.read_column(c);
            }
        }
        rec_d.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_v2_recorded (left_outer): step-D preflight failed: {}",
                e
            ))
        })?;

        let mut result_columns: Vec<CudaColumn> = Vec::with_capacity(combined_schema.arity());
        let inner_rows = inner_count as u64;

        // Per-left-column concat.
        for col_idx in 0..left.arity() {
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

            let out_col = self.memory.alloc::<u8>(total_bytes)?;
            let dst_ptr = *out_col.device_ptr();
            // Fence alloc-ready → launch_stream for out_col
            // before the dtod-copies write it.
            runtime
                .prepare_first_use(&out_col, launch_stream, Access::Write)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "left_outer recorded: prepare left out_col {} failed: {}",
                        col_idx, e
                    ))
                })?;

            if inner_bytes > 0 {
                let src_col = inner_left_buf
                    .as_ref()
                    .expect("inner_count > 0 but inner_left_buf is None")
                    .column(col_idx)
                    .ok_or_else(|| {
                        XlogError::Kernel(format!("inner_left col {} not found", col_idx))
                    })?;
                // SAFETY: cuMemcpyDtoDAsync_v2 on cu_stream.
                unsafe {
                    let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_ptr,
                        *src_col.device_ptr(),
                        inner_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "cuMemcpyDtoDAsync (left_outer inner_left col {}) failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if unmatched_bytes > 0 {
                let src_col = unmatched_left.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("unmatched_left col {} not found", col_idx))
                })?;
                // SAFETY: dst_ptr + inner_bytes is in-bounds
                // (inner_bytes + unmatched_bytes == total_bytes).
                unsafe {
                    let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_ptr + inner_bytes as u64,
                        *src_col.device_ptr(),
                        unmatched_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "cuMemcpyDtoDAsync (left_outer unmatched_left col {}) failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }

            // Record use on launch_stream so end-of-scope drop
            // (when result_columns goes out of scope down the
            // line via output buffer drop) defers correctly.
            if let Some(b) = out_col.runtime_block() {
                runtime
                    .finish_block_use(BlockId::from_block(b), launch_stream, Access::Write)
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "hash_join_v2_recorded (left_outer): finish_block_use \
                         (left col {}) failed: {}",
                            col_idx, e
                        ))
                    })?;
            }
            result_columns.push(out_col.into());
        }

        // Per-right-column: inner_right | zeros.
        for col_idx in 0..right.arity() {
            let elem_size = right
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let inner_bytes = (inner_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("Left outer join: right inner_bytes overflow".to_string())
                })?;
            let unmatched_bytes = (unmatched_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    XlogError::Kernel("Left outer join: right unmatched_bytes overflow".to_string())
                })?;
            let total_bytes = inner_bytes.checked_add(unmatched_bytes).ok_or_else(|| {
                XlogError::Kernel("Left outer join: right total_bytes overflow".to_string())
            })?;

            let out_col = self.memory.alloc::<u8>(total_bytes)?;
            let dst_ptr = *out_col.device_ptr();
            // Fence alloc-ready → launch_stream for out_col
            // before the memset / dtod-copy write it.
            runtime
                .prepare_first_use(&out_col, launch_stream, Access::Write)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "left_outer recorded: prepare right out_col {} failed: {}",
                        col_idx, e
                    ))
                })?;

            // Zero whole column (unmatched portion will stay
            // zero; inner portion will be overwritten by the
            // dtod copy below if inner_bytes > 0).
            if total_bytes > 0 {
                // SAFETY: out_col has total_bytes bytes; cu_stream is valid.
                unsafe {
                    let res = cudarc::driver::sys::cuMemsetD8Async(
                        dst_ptr,
                        0,
                        total_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "cuMemsetD8Async (left_outer right col {}) failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if inner_bytes > 0 {
                let src_col = inner_right_buf
                    .as_ref()
                    .expect("inner_count > 0 but inner_right_buf is None")
                    .column(col_idx)
                    .ok_or_else(|| {
                        XlogError::Kernel(format!("inner_right col {} not found", col_idx))
                    })?;
                // SAFETY: cuMemcpyDtoDAsync_v2 on cu_stream.
                unsafe {
                    let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_ptr,
                        *src_col.device_ptr(),
                        inner_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "cuMemcpyDtoDAsync (left_outer inner_right col {}) failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }

            if let Some(b) = out_col.runtime_block() {
                runtime
                    .finish_block_use(BlockId::from_block(b), launch_stream, Access::Write)
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "hash_join_v2_recorded (left_outer): finish_block_use \
                         (right col {}) failed: {}",
                            col_idx, e
                        ))
                    })?;
            }
            result_columns.push(out_col.into());
        }

        // Commit the step-D recorder NOW that every dtod
        // copy is queued. The recorded event captures up to
        // commit time, so a subsequent drop of any source
        // buffer (unmatched_left / inner_*_buf) correctly
        // waits.
        rec_d.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_v2_recorded (left_outer): step-D commit failed: {}",
                e
            ))
        })?;

        // d_num_rows scalar for the output buffer (uploaded
        // synchronously; no launch_stream work touches it).
        let d_num_rows = self.upload_device_row_count(total_rows as u32)?;
        Ok(CudaBuffer::from_columns_with_host_count(
            result_columns,
            total_rows,
            d_num_rows,
            combined_schema,
            total_rows as u32,
        ))
    }

    /// Strict-recorder Semi/Anti hash join (slice #7B).
    /// Single helper parametrized by `anti`: both share the
    /// kernel-arg shape and chain — pack keys for both sides,
    /// build the hash table, run the `hash_join_semi` /
    /// `hash_join_anti` kernel to produce a per-left-row mask,
    /// then compose with
    /// `compact_buffer_by_device_mask_counted_recorded` to
    /// filter `left` by the mask. Semi mask is "has match",
    /// Anti mask is "no match"; the recorded compact tail keeps
    /// rows where the mask byte is non-zero either way.
    fn hash_join_semi_or_anti_v2_recorded(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        anti: bool,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        use crate::launch::LaunchRecorder;

        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "hash_join_v2_recorded (semi/anti) requires a runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                "hash_join_v2_recorded (semi/anti): launch_stream StreamId({}) does not resolve",
                launch_stream.0
            ))
            })?;

        let num_left = self.device_row_count(left)?;
        let num_right = self.device_row_count(right)?;
        if num_left > u32::MAX as usize || num_right > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                num_left,
                num_right
            )));
        }
        if num_left == 0 {
            return self.create_empty_buffer(left.schema().clone());
        }
        if num_right == 0 {
            // No matches possible.
            //   * Semi: empty result.
            //   * Anti: keep all left rows.
            //
            // The Anti edge case is a copy of `left`. We use
            // legacy `clone_buffer` here, which runs on the
            // default stream and synchronizes before
            // returning. No launch_stream work is queued, so
            // there are no recorded events on the original
            // input columns from this call — which is the
            // correct semantic: we did not touch them on
            // launch_stream.
            return if anti {
                self.clone_buffer(left)
            } else {
                self.create_empty_buffer(left.schema().clone())
            };
        }
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
        if left_keys.len() > 4 {
            return Err(XlogError::Kernel(
                "hash_join_v2_recorded (semi/anti): max 4 key columns supported (pack_keys constraint)"
                    .to_string(),
            ));
        }
        for (&l, &r) in left_keys.iter().zip(right_keys.iter()) {
            let lt = left.schema().column_type(l);
            let rt = right.schema().column_type(r);
            if lt != rt {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    l, lt, r, rt
                )));
            }
        }

        let num_left = num_left as u32;
        let num_right = num_right as u32;

        let left_packed =
            self.pack_keys_gpu_on_stream(left, left_keys, &cu_stream, launch_stream, runtime)?;
        let right_packed =
            self.pack_keys_gpu_on_stream(right, right_keys, &cu_stream, launch_stream, runtime)?;
        let table = self.build_hash_table_v2_on_stream(
            &right_packed.hashes,
            num_right,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        let d_mask = self.memory.alloc::<u8>(num_left as usize)?;

        let kernel_name = if anti {
            join_kernels::HASH_JOIN_ANTI
        } else {
            join_kernels::HASH_JOIN_SEMI
        };
        let func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, kernel_name)
            .ok_or_else(|| XlogError::Kernel(format!("{} kernel not found", kernel_name)))?;

        let block_size = 256u32;
        let grid_size = num_left.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(&left_packed.hashes);
        rec.read(&left_packed.packed_keys);
        rec.read(&right_packed.packed_keys);
        rec.read(&table.bucket_offsets);
        rec.read(&table.bucket_counts);
        rec.read(&table.bucket_entries);
        rec.read(&table.bucket_entry_hashes);
        rec.write(&d_mask);
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_v2_recorded (semi/anti): preflight failed: {}",
                e
            ))
        })?;

        // SAFETY: hash_join_{semi,anti}(probe_hashes, num_probe,
        //   bucket_offsets, bucket_counts, bucket_entries,
        //   bucket_entry_hashes, bucket_mask, probe_keys,
        //   build_keys, key_bytes, mask). 11 args.
        unsafe {
            func.clone().launch_on_stream(
                &cu_stream,
                cfg,
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
                    &d_mask,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("{} (on_stream) failed: {}", kernel_name, e)))?;

        rec.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_v2_recorded (semi/anti): commit failed: {}",
                e
            ))
        })?;

        // Filter `left` by the mask via the recorded compact
        // tail. Its own LaunchRecorder records reads on
        // left.column[i] and left.num_rows_device against
        // launch_stream — so dropping `left` after this
        // method returns is correctly serialized through the
        // runtime's record-all + wait-all event chain.
        self.compact_buffer_by_device_mask_counted_recorded(left, &d_mask, launch_stream)
    }

    // ============== Recorded indexed hash join — slice #7D ==============
    //
    // Strict-recorder, launch_stream-routed sibling of
    // `hash_join_v2_with_index`. Covers Inner, Semi, Anti, and
    // LeftOuter via a single dispatcher. The build-side
    // packed keys + hash table come from the cached
    // `JoinIndexV2`; the probe (left) side is packed on
    // launch_stream via `pack_keys_gpu_on_stream`. Recorded
    // gather / compact / mask_not helpers from earlier slices
    // are reused unchanged.
    //
    // Existing legacy `hash_join_v2_with_index*` paths are
    // unchanged; runtime/planner wiring is NOT included.

    /// Strict-recorder, launch_stream-routed variant of
    /// `hash_join_v2_with_index`. Supports all four join
    /// types — the indexed variants share the same
    /// `(packed_keys, table)` shape, so a single recorded
    /// surface covers them.
    ///
    /// When [`Self::use_recorded_csm_env`] is on, `Inner` and
    /// `LeftOuter` route through the indexed CSM
    /// (count-scan-materialize) methods; otherwise they route
    /// through the legacy indexed recorded methods. `Semi` /
    /// `Anti` always route through their existing indexed
    /// recorded methods — no CSM implementation exists for them.
    #[allow(clippy::too_many_arguments)]
    pub fn hash_join_v2_with_index_recorded(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        join_type: JoinType,
        index: &crate::provider::JoinIndexV2,
        max_output: Option<usize>,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "hash_join_v2_with_index_recorded requires a runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        // Resolve once; sub-helpers re-resolve as needed.
        runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "hash_join_v2_with_index_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        // Validate inputs (mirror legacy hash_join_v2_with_index).
        let left_rows = self.device_row_count(left)?;
        let right_rows = self.device_row_count(right)?;
        if left_rows > u32::MAX as usize || right_rows > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "Join supports at most {} rows per side (left={}, right={})",
                u32::MAX,
                left_rows,
                right_rows
            )));
        }
        if left_rows == 0 {
            return match join_type {
                JoinType::Inner | JoinType::LeftOuter => {
                    let combined_schema = self.combine_schemas(left.schema(), right.schema());
                    self.create_empty_buffer(combined_schema)
                }
                JoinType::Semi | JoinType::Anti => self.create_empty_buffer(left.schema().clone()),
            };
        }
        if right_rows == 0 {
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
        if left_keys.len() > 4 {
            return Err(XlogError::Kernel(
                "hash_join_v2_with_index_recorded: max 4 key columns supported \
                 (pack_keys constraint)"
                    .to_string(),
            ));
        }
        for (&l, &r) in left_keys.iter().zip(right_keys.iter()) {
            if l >= left.arity() {
                return Err(XlogError::Kernel(format!(
                    "Left key column index {} out of bounds (arity {})",
                    l,
                    left.arity()
                )));
            }
            if r >= right.arity() {
                return Err(XlogError::Kernel(format!(
                    "Right key column index {} out of bounds (arity {})",
                    r,
                    right.arity()
                )));
            }
            let lt = left.schema().column_type(l);
            let rt = right.schema().column_type(r);
            if lt != rt {
                return Err(XlogError::Kernel(format!(
                    "Key column type mismatch: left[{}]={:?}, right[{}]={:?}",
                    l, lt, r, rt
                )));
            }
        }
        if index.right_num_rows() != right_rows as u32 {
            return Err(XlogError::Kernel(
                "Join index row count does not match right relation".to_string(),
            ));
        }
        if index.right_keys() != right_keys {
            return Err(XlogError::Kernel(
                "Join index key columns do not match requested right_keys".to_string(),
            ));
        }

        let csm_on = Self::use_recorded_csm_env();
        match join_type {
            JoinType::Inner => {
                if csm_on {
                    self.csm_invocations.fetch_add(1, Ordering::Relaxed);
                    self.hash_join_inner_v2_with_index_count_scan_materialize_recorded(
                        left,
                        right,
                        left_keys,
                        right_keys,
                        index,
                        max_output,
                        launch_stream,
                    )
                } else {
                    self.hash_join_inner_v2_with_index_recorded(
                        left,
                        right,
                        left_keys,
                        index,
                        max_output,
                        launch_stream,
                    )
                }
            }
            JoinType::Semi => self.hash_join_semi_or_anti_v2_with_index_recorded(
                left,
                left_keys,
                index,
                false,
                launch_stream,
            ),
            JoinType::Anti => self.hash_join_semi_or_anti_v2_with_index_recorded(
                left,
                left_keys,
                index,
                true,
                launch_stream,
            ),
            JoinType::LeftOuter => {
                if csm_on {
                    self.csm_invocations.fetch_add(1, Ordering::Relaxed);
                    self.hash_join_left_outer_v2_with_index_count_scan_materialize_recorded(
                        left,
                        right,
                        left_keys,
                        right_keys,
                        index,
                        max_output,
                        launch_stream,
                    )
                } else {
                    self.hash_join_left_outer_v2_with_index_recorded(
                        left,
                        right,
                        left_keys,
                        index,
                        max_output,
                        launch_stream,
                    )
                }
            }
        }
    }

    /// Indexed-Inner recorded. Mirrors `hash_join_inner_v2_recorded`
    /// minus the right-side pack + hash-table build (the
    /// cached `JoinIndexV2` provides `index.packed_keys` and
    /// `&index.table`). Probe count + materialize + gather all
    /// run on `launch_stream`.
    fn hash_join_inner_v2_with_index_recorded(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        index: &crate::provider::JoinIndexV2,
        max_output: Option<usize>,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        use crate::launch::LaunchRecorder;

        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "hash_join_v2_with_index_recorded (inner) requires runtime-backed manager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel("indexed inner: launch_stream does not resolve".to_string())
            })?;

        let num_left = left.num_rows() as u32;
        let table = &index.table;

        // Pack left only on launch_stream.
        let left_packed =
            self.pack_keys_gpu_on_stream(left, left_keys, &cu_stream, launch_stream, runtime)?;
        if left_packed.key_bytes != index.key_bytes {
            return Err(XlogError::Kernel(
                "Join key byte width mismatch between probe and cached index".to_string(),
            ));
        }

        let probe_func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2)
            .ok_or_else(|| XlogError::Kernel("hash_join_probe_v2 kernel not found".to_string()))?;
        let block_size = 256u32;
        let probe_grid = num_left.div_ceil(block_size);
        let probe_config = LaunchConfig {
            grid_dim: (probe_grid, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Count pass.
        let d_count_only = self.memory.alloc::<u32>(1)?;
        let d_dummy_left = self.memory.alloc::<u32>(1)?;
        let d_dummy_right = self.memory.alloc::<u32>(1)?;
        // Fence alloc-ready → launch_stream for d_count_only
        // before the memset (memset runs ahead of preflight).
        runtime
            .prepare_first_use(&d_count_only, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!("indexed inner: prepare d_count_only failed: {}", e))
            })?;
        // SAFETY: 4-byte runtime-backed buffer.
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_count_only.device_ptr(),
                0,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "cuMemsetD8Async (indexed inner d_count_only) failed: {:?}",
                    res
                )));
            }
        }

        let max_output_count_only = 0u32;
        let mut rec_count = LaunchRecorder::new_strict(launch_stream);
        rec_count.read(&left_packed.hashes);
        rec_count.read(&left_packed.packed_keys);
        rec_count.read(&index.packed_keys);
        rec_count.read(&table.bucket_offsets);
        rec_count.read(&table.bucket_counts);
        rec_count.read(&table.bucket_entries);
        rec_count.read(&table.bucket_entry_hashes);
        rec_count.write(&d_count_only);
        rec_count.write(&d_dummy_left);
        rec_count.write(&d_dummy_right);
        rec_count.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!("indexed inner: count-pass preflight failed: {}", e))
        })?;
        // SAFETY: 14-arg probe via raw-param launch.
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&index.packed_keys).as_kernel_param(),
                index.key_bytes.as_kernel_param(),
                (&d_dummy_left).as_kernel_param(),
                (&d_dummy_right).as_kernel_param(),
                (&d_count_only).as_kernel_param(),
                max_output_count_only.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch_on_stream(&cu_stream, probe_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "hash_join_probe_v2 (indexed count, on_stream) failed: {}",
                        e
                    ))
                })?;
        }
        rec_count.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("indexed inner: count-pass commit failed: {}", e))
        })?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!("indexed inner: sync (count read) failed: {}", e))
        })?;
        let full_count = self.read_join_output_count_metadata(&d_count_only)? as u64;
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
        let max_output_u32 = requested as u32;

        // Materialize pass.
        let d_output_left = self.memory.alloc::<u32>(max_output_u32 as usize)?;
        let d_output_right = self.memory.alloc::<u32>(max_output_u32 as usize)?;
        let d_output_count = self.memory.alloc::<u32>(1)?;
        // Fence alloc-ready → launch_stream for d_output_count
        // before the memset.
        runtime
            .prepare_first_use(&d_output_count, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "indexed inner: prepare d_output_count failed: {}",
                    e
                ))
            })?;
        // SAFETY: 4-byte runtime-backed buffer.
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_output_count.device_ptr(),
                0,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "cuMemsetD8Async (indexed inner d_output_count) failed: {:?}",
                    res
                )));
            }
        }

        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        rec_mat.read(&left_packed.hashes);
        rec_mat.read(&left_packed.packed_keys);
        rec_mat.read(&index.packed_keys);
        rec_mat.read(&table.bucket_offsets);
        rec_mat.read(&table.bucket_counts);
        rec_mat.read(&table.bucket_entries);
        rec_mat.read(&table.bucket_entry_hashes);
        rec_mat.write(&d_output_left);
        rec_mat.write(&d_output_right);
        rec_mat.write(&d_output_count);
        rec_mat.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed inner: materialize preflight failed: {}",
                e
            ))
        })?;
        // SAFETY: 14-arg probe via raw-param launch.
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&index.packed_keys).as_kernel_param(),
                index.key_bytes.as_kernel_param(),
                (&d_output_left).as_kernel_param(),
                (&d_output_right).as_kernel_param(),
                (&d_output_count).as_kernel_param(),
                max_output_u32.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch_on_stream(&cu_stream, probe_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "hash_join_probe_v2 (indexed mat, on_stream) failed: {}",
                        e
                    ))
                })?;
        }
        rec_mat.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("indexed inner: materialize commit failed: {}", e))
        })?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!("indexed inner: sync (mat read) failed: {}", e))
        })?;
        let result_count = (self.read_join_output_count_metadata(&d_output_count)? as u64)
            .min(max_output_u32 as u64);
        if result_count == 0 {
            let combined_schema = self.combine_schemas(left.schema(), right.schema());
            return self.create_empty_buffer(combined_schema);
        }
        let output_rows = result_count as u32;

        // Gather both sides on launch_stream.
        let mut rec_gather = LaunchRecorder::new_strict(launch_stream);
        for col_idx in 0..left.columns.len() {
            let c = left
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Left column {} not found", col_idx)))?;
            rec_gather.read_column(c);
        }
        for col_idx in 0..right.columns.len() {
            let c = right
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Right column {} not found", col_idx)))?;
            rec_gather.read_column(c);
        }
        rec_gather.read(&d_output_left);
        rec_gather.read(&d_output_right);
        rec_gather.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!("indexed inner: gather preflight failed: {}", e))
        })?;
        let gathered_left = self.gather_buffer_by_indices_on_stream(
            left,
            &d_output_left,
            output_rows,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        let gathered_right = self.gather_buffer_by_indices_on_stream(
            right,
            &d_output_right,
            output_rows,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        rec_gather.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("indexed inner: gather commit failed: {}", e))
        })?;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        let mut result_columns = Vec::with_capacity(combined_schema.arity());
        result_columns.extend(gathered_left.columns);
        result_columns.extend(gathered_right.columns);
        self.buffer_from_columns(result_columns, result_count, combined_schema)
    }

    /// Indexed Semi/Anti recorded. Mirrors
    /// `hash_join_semi_or_anti_v2_recorded` minus the
    /// right-side pack + table build. Composes pack-left →
    /// SEMI/ANTI kernel → recorded compact tail. Anti-empty-
    /// right edge case is handled by the dispatcher.
    fn hash_join_semi_or_anti_v2_with_index_recorded(
        &self,
        left: &CudaBuffer,
        left_keys: &[usize],
        index: &crate::provider::JoinIndexV2,
        anti: bool,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        use crate::launch::LaunchRecorder;

        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "hash_join_v2_with_index_recorded (semi/anti) requires runtime-backed manager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel("indexed semi/anti: launch_stream does not resolve".to_string())
            })?;

        let num_left = left.num_rows() as u32;
        let table = &index.table;

        let left_packed =
            self.pack_keys_gpu_on_stream(left, left_keys, &cu_stream, launch_stream, runtime)?;
        if left_packed.key_bytes != index.key_bytes {
            return Err(XlogError::Kernel(
                "Join key byte width mismatch between probe and cached index".to_string(),
            ));
        }

        let d_mask = self.memory.alloc::<u8>(num_left as usize)?;
        let kernel_name = if anti {
            join_kernels::HASH_JOIN_ANTI
        } else {
            join_kernels::HASH_JOIN_SEMI
        };
        let func = self
            .device
            .inner()
            .get_func(JOIN_MODULE, kernel_name)
            .ok_or_else(|| XlogError::Kernel(format!("{} kernel not found", kernel_name)))?;
        let block_size = 256u32;
        let grid_size = num_left.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(&left_packed.hashes);
        rec.read(&left_packed.packed_keys);
        rec.read(&index.packed_keys);
        rec.read(&table.bucket_offsets);
        rec.read(&table.bucket_counts);
        rec.read(&table.bucket_entries);
        rec.read(&table.bucket_entry_hashes);
        rec.write(&d_mask);
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!("indexed semi/anti: preflight failed: {}", e))
        })?;
        // SAFETY: 11-arg semi/anti.
        unsafe {
            func.clone().launch_on_stream(
                &cu_stream,
                cfg,
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
                    &d_mask,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "{} (on_stream, indexed) failed: {}",
                kernel_name, e
            ))
        })?;
        rec.commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("indexed semi/anti: commit failed: {}", e)))?;

        self.compact_buffer_by_device_mask_counted_recorded(left, &d_mask, launch_stream)
    }

    /// Indexed LeftOuter recorded. Mirrors
    /// `hash_join_left_outer_v2_recorded` minus the right-side
    /// pack + table build. Same chain shape: SEMI mask + PROBE
    /// count/materialize + mask_not + recorded compact for
    /// unmatched + gather inner + per-column dtod-async concat.
    fn hash_join_left_outer_v2_with_index_recorded(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        index: &crate::provider::JoinIndexV2,
        max_output: Option<usize>,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        use crate::launch::LaunchRecorder;

        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "hash_join_v2_with_index_recorded (left_outer) requires runtime-backed manager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel("indexed left_outer: launch_stream does not resolve".to_string())
            })?;

        let num_left = left.num_rows() as u32;
        let table = &index.table;

        let left_packed =
            self.pack_keys_gpu_on_stream(left, left_keys, &cu_stream, launch_stream, runtime)?;
        if left_packed.key_bytes != index.key_bytes {
            return Err(XlogError::Kernel(
                "Join key byte width mismatch between probe and cached index".to_string(),
            ));
        }

        let device = self.device.inner();
        let block_size = 256u32;
        let grid_size = num_left.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Step A: SEMI mask + PROBE count.
        let d_has_match = self.memory.alloc::<u8>(num_left as usize)?;
        let d_count_only = self.memory.alloc::<u32>(1)?;
        let d_dummy_left = self.memory.alloc::<u32>(1)?;
        let d_dummy_right = self.memory.alloc::<u32>(1)?;
        // Fence alloc-ready → launch_stream for d_count_only
        // before the memset.
        runtime
            .prepare_first_use(&d_count_only, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "indexed left_outer: prepare d_count_only failed: {}",
                    e
                ))
            })?;
        // SAFETY: 4-byte runtime-backed buffer.
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_count_only.device_ptr(),
                0,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "cuMemsetD8Async (indexed left_outer d_count_only) failed: {:?}",
                    res
                )));
            }
        }

        let semi_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_SEMI)
            .ok_or_else(|| XlogError::Kernel("hash_join_semi kernel not found".to_string()))?;
        let probe_func = device
            .get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE_V2)
            .ok_or_else(|| XlogError::Kernel("hash_join_probe_v2 kernel not found".to_string()))?;

        let mut rec_a = LaunchRecorder::new_strict(launch_stream);
        rec_a.read(&left_packed.hashes);
        rec_a.read(&left_packed.packed_keys);
        rec_a.read(&index.packed_keys);
        rec_a.read(&table.bucket_offsets);
        rec_a.read(&table.bucket_counts);
        rec_a.read(&table.bucket_entries);
        rec_a.read(&table.bucket_entry_hashes);
        rec_a.write(&d_has_match);
        rec_a.write(&d_count_only);
        rec_a.write(&d_dummy_left);
        rec_a.write(&d_dummy_right);
        rec_a.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed left_outer: semi/count preflight failed: {}",
                e
            ))
        })?;
        // SAFETY: hash_join_semi 11-arg.
        unsafe {
            semi_func.clone().launch_on_stream(
                &cu_stream,
                cfg,
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
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "hash_join_semi (on_stream, indexed left_outer) failed: {}",
                e
            ))
        })?;

        let max_output_count_only = 0u32;
        // SAFETY: hash_join_probe_v2 14-arg count pass.
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&index.packed_keys).as_kernel_param(),
                index.key_bytes.as_kernel_param(),
                (&d_dummy_left).as_kernel_param(),
                (&d_dummy_right).as_kernel_param(),
                (&d_count_only).as_kernel_param(),
                max_output_count_only.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch_on_stream(&cu_stream, cfg, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "hash_join_probe_v2 (count, on_stream, indexed left_outer) failed: {}",
                        e
                    ))
                })?;
        }
        rec_a.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed left_outer: semi/count commit failed: {}",
                e
            ))
        })?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "indexed left_outer: sync (count read) failed: {}",
                e
            ))
        })?;
        let full_inner = self.read_join_output_count_metadata(&d_count_only)? as u64;
        let requested_inner = max_output
            .map(|limit| (limit as u64).min(full_inner))
            .unwrap_or(full_inner);
        if requested_inner > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "Join produced {} rows which exceeds the u32 index limit",
                requested_inner
            )));
        }
        let max_output_u32 = requested_inner as u32;
        let alloc_len = (requested_inner.max(1)) as usize;

        // PROBE materialize.
        let d_output_left = self.memory.alloc::<u32>(alloc_len)?;
        let d_output_right = self.memory.alloc::<u32>(alloc_len)?;
        let d_output_count = self.memory.alloc::<u32>(1)?;
        // Fence alloc-ready → launch_stream for d_output_count
        // before the memset.
        runtime
            .prepare_first_use(&d_output_count, launch_stream, Access::Write)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "indexed left_outer: prepare d_output_count failed: {}",
                    e
                ))
            })?;
        // SAFETY: 4-byte runtime-backed buffer.
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8Async(
                *d_output_count.device_ptr(),
                0,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "cuMemsetD8Async (indexed left_outer d_output_count) failed: {:?}",
                    res
                )));
            }
        }

        let mut rec_b = LaunchRecorder::new_strict(launch_stream);
        rec_b.read(&left_packed.hashes);
        rec_b.read(&left_packed.packed_keys);
        rec_b.read(&index.packed_keys);
        rec_b.read(&table.bucket_offsets);
        rec_b.read(&table.bucket_counts);
        rec_b.read(&table.bucket_entries);
        rec_b.read(&table.bucket_entry_hashes);
        rec_b.write(&d_output_left);
        rec_b.write(&d_output_right);
        rec_b.write(&d_output_count);
        rec_b.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed left_outer: materialize preflight failed: {}",
                e
            ))
        })?;
        // SAFETY: hash_join_probe_v2 14-arg materialize.
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&left_packed.hashes).as_kernel_param(),
                num_left.as_kernel_param(),
                (&table.bucket_offsets).as_kernel_param(),
                (&table.bucket_counts).as_kernel_param(),
                (&table.bucket_entries).as_kernel_param(),
                (&table.bucket_entry_hashes).as_kernel_param(),
                table.bucket_mask.as_kernel_param(),
                (&left_packed.packed_keys).as_kernel_param(),
                (&index.packed_keys).as_kernel_param(),
                index.key_bytes.as_kernel_param(),
                (&d_output_left).as_kernel_param(),
                (&d_output_right).as_kernel_param(),
                (&d_output_count).as_kernel_param(),
                max_output_u32.as_kernel_param(),
            ];
            probe_func
                .clone()
                .launch_on_stream(&cu_stream, cfg, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "hash_join_probe_v2 (mat, on_stream, indexed left_outer) failed: {}",
                        e
                    ))
                })?;
        }
        rec_b.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed left_outer: materialize commit failed: {}",
                e
            ))
        })?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!("indexed left_outer: sync (mat read) failed: {}", e))
        })?;
        let inner_count = self
            .read_join_output_count_metadata(&d_output_count)?
            .min(max_output_u32);

        // Step B: mask_not → unmatched filter via recorded compact tail.
        let d_no_match = self.memory.alloc::<u8>(num_left as usize)?;
        let mask_not_fn = device
            .get_func(FILTER_MODULE, filter_kernels::MASK_NOT)
            .ok_or_else(|| XlogError::Kernel("mask_not kernel not found".to_string()))?;
        let mut rec_c = LaunchRecorder::new_strict(launch_stream);
        rec_c.read(&d_has_match);
        rec_c.write(&d_no_match);
        rec_c.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed left_outer: mask_not preflight failed: {}",
                e
            ))
        })?;
        // SAFETY: mask_not(in, out, n).
        unsafe {
            mask_not_fn.clone().launch_on_stream(
                &cu_stream,
                cfg,
                (&d_has_match, &d_no_match, num_left),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "mask_not (on_stream, indexed left_outer) failed: {}",
                e
            ))
        })?;
        rec_c.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("indexed left_outer: mask_not commit failed: {}", e))
        })?;

        let unmatched_left =
            self.compact_buffer_by_device_mask_counted_recorded(left, &d_no_match, launch_stream)?;
        let unmatched_rows = self.device_row_count(&unmatched_left)? as u64;
        let total_rows = (inner_count as u64) + unmatched_rows;

        let combined_schema = self.combine_schemas(left.schema(), right.schema());
        if total_rows == 0 {
            return self.create_empty_buffer(combined_schema);
        }

        // Step C: gather inner sides. Same outer-recorder
        // wrapping as the non-indexed LeftOuter — registers
        // launch_stream reads on left/right columns and the
        // probe-output index buffers so the caller's drop of
        // those inputs is correctly serialized.
        let inner_count_u32 = inner_count;
        let inner_left_buf;
        let inner_right_buf;
        if inner_count > 0 {
            let mut rec_gather = LaunchRecorder::new_strict(launch_stream);
            for col_idx in 0..left.columns.len() {
                let c = left.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("Left column {} not found", col_idx))
                })?;
                rec_gather.read_column(c);
            }
            for col_idx in 0..right.columns.len() {
                let c = right.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("Right column {} not found", col_idx))
                })?;
                rec_gather.read_column(c);
            }
            rec_gather.read(&d_output_left);
            rec_gather.read(&d_output_right);
            rec_gather.preflight(runtime).map_err(|e| {
                XlogError::Kernel(format!(
                    "indexed left_outer: gather preflight failed: {}",
                    e
                ))
            })?;
            inner_left_buf = Some(self.gather_buffer_by_indices_on_stream(
                left,
                &d_output_left,
                inner_count_u32,
                &cu_stream,
                launch_stream,
                runtime,
            )?);
            inner_right_buf = Some(self.gather_buffer_by_indices_on_stream(
                right,
                &d_output_right,
                inner_count_u32,
                &cu_stream,
                launch_stream,
                runtime,
            )?);
            rec_gather.commit(runtime).map_err(|e| {
                XlogError::Kernel(format!("indexed left_outer: gather commit failed: {}", e))
            })?;
        } else {
            inner_left_buf = None;
            inner_right_buf = None;
        }

        // Step D: concatenate per-column on launch_stream.
        // Same step-D recorder discipline as the non-indexed
        // LeftOuter: re-record source columns AFTER the dtod
        // copies are queued, so a drop of `unmatched_left` /
        // `inner_*_buf` waits on the correct event.
        let mut rec_d = LaunchRecorder::new_strict(launch_stream);
        for col_idx in 0..unmatched_left.columns.len() {
            let c = unmatched_left.column(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!("unmatched_left col {} not found", col_idx))
            })?;
            rec_d.read_column(c);
        }
        if let Some(b) = inner_left_buf.as_ref() {
            for col_idx in 0..b.columns.len() {
                let c = b.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("inner_left col {} not found", col_idx))
                })?;
                rec_d.read_column(c);
            }
        }
        if let Some(b) = inner_right_buf.as_ref() {
            for col_idx in 0..b.columns.len() {
                let c = b.column(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!("inner_right col {} not found", col_idx))
                })?;
                rec_d.read_column(c);
            }
        }
        rec_d.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "indexed left_outer: step-D preflight failed: {}",
                e
            ))
        })?;

        let mut result_columns: Vec<CudaColumn> = Vec::with_capacity(combined_schema.arity());
        let inner_rows = inner_count as u64;

        for col_idx in 0..left.arity() {
            let elem_size = left
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let inner_bytes = (inner_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| XlogError::Kernel("inner_bytes overflow".to_string()))?;
            let unmatched_bytes = (unmatched_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| XlogError::Kernel("unmatched_bytes overflow".to_string()))?;
            let total_bytes = inner_bytes
                .checked_add(unmatched_bytes)
                .ok_or_else(|| XlogError::Kernel("total_bytes overflow".to_string()))?;
            let out_col = self.memory.alloc::<u8>(total_bytes)?;
            let dst_ptr = *out_col.device_ptr();
            // Fence alloc-ready → launch_stream for out_col.
            runtime
                .prepare_first_use(&out_col, launch_stream, Access::Write)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "indexed left_outer: prepare left out_col {} failed: {}",
                        col_idx, e
                    ))
                })?;
            if inner_bytes > 0 {
                let src_col = inner_left_buf
                    .as_ref()
                    .expect("inner_count > 0")
                    .column(col_idx)
                    .ok_or_else(|| XlogError::Kernel("inner_left col missing".to_string()))?;
                // SAFETY: dtod async on cu_stream.
                unsafe {
                    let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_ptr,
                        *src_col.device_ptr(),
                        inner_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "indexed left_outer: dtod copy inner_left col {} failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if unmatched_bytes > 0 {
                let src_col = unmatched_left
                    .column(col_idx)
                    .ok_or_else(|| XlogError::Kernel("unmatched col missing".to_string()))?;
                // SAFETY: bounded by total_bytes.
                unsafe {
                    let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_ptr + inner_bytes as u64,
                        *src_col.device_ptr(),
                        unmatched_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "indexed left_outer: dtod copy unmatched col {} failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if let Some(b) = out_col.runtime_block() {
                runtime
                    .finish_block_use(BlockId::from_block(b), launch_stream, Access::Write)
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "indexed left_outer: finish_block_use (left col {}) failed: {}",
                            col_idx, e
                        ))
                    })?;
            }
            result_columns.push(out_col.into());
        }

        for col_idx in 0..right.arity() {
            let elem_size = right
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let inner_bytes = (inner_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| XlogError::Kernel("right inner_bytes overflow".to_string()))?;
            let unmatched_bytes = (unmatched_rows as usize)
                .checked_mul(elem_size)
                .ok_or_else(|| XlogError::Kernel("right unmatched_bytes overflow".to_string()))?;
            let total_bytes = inner_bytes
                .checked_add(unmatched_bytes)
                .ok_or_else(|| XlogError::Kernel("right total_bytes overflow".to_string()))?;
            let out_col = self.memory.alloc::<u8>(total_bytes)?;
            let dst_ptr = *out_col.device_ptr();
            // Fence alloc-ready → launch_stream for out_col.
            runtime
                .prepare_first_use(&out_col, launch_stream, Access::Write)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "indexed left_outer: prepare right out_col {} failed: {}",
                        col_idx, e
                    ))
                })?;
            if total_bytes > 0 {
                // SAFETY: zero-fill the whole column.
                unsafe {
                    let res = cudarc::driver::sys::cuMemsetD8Async(
                        dst_ptr,
                        0,
                        total_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "indexed left_outer: zero-fill right col {} failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if inner_bytes > 0 {
                let src_col = inner_right_buf
                    .as_ref()
                    .expect("inner_count > 0")
                    .column(col_idx)
                    .ok_or_else(|| XlogError::Kernel("inner_right col missing".to_string()))?;
                // SAFETY: dtod async on cu_stream.
                unsafe {
                    let res = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                        dst_ptr,
                        *src_col.device_ptr(),
                        inner_bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "indexed left_outer: dtod copy inner_right col {} failed: {:?}",
                            col_idx, res
                        )));
                    }
                }
            }
            if let Some(b) = out_col.runtime_block() {
                runtime
                    .finish_block_use(BlockId::from_block(b), launch_stream, Access::Write)
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "indexed left_outer: finish_block_use (right col {}) failed: {}",
                            col_idx, e
                        ))
                    })?;
            }
            result_columns.push(out_col.into());
        }

        // Commit step-D recorder; see non-indexed LeftOuter
        // for the rationale.
        rec_d.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("indexed left_outer: step-D commit failed: {}", e))
        })?;

        let d_num_rows = self.upload_device_row_count(total_rows as u32)?;
        Ok(CudaBuffer::from_columns_with_host_count(
            result_columns,
            total_rows,
            d_num_rows,
            combined_schema,
            total_rows as u32,
        ))
    }
}
