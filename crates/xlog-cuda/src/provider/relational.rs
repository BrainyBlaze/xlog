//! Relational operations: join, dedup, union, diff, sort, and related helpers.

use std::ffi::c_void;

use cudarc::driver::{DevicePtr, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, ScalarType, Schema, XlogError};

use super::{
    dedup_kernels, filter_kernels, ilp_kernels, join_kernels,
    pack_kernels, scan_kernels, set_ops_kernels, sort_kernels,
    HashTableU64, JoinHashTableV2, JoinIndexV2, JoinType, PackedKeyData,
    RadixSortScratch,
    DEFAULT_JOIN_MAX_OUTPUT,
    DEDUP_MODULE, FILTER_MODULE, ILP_MODULE, JOIN_MODULE,
    PACK_MODULE, SCAN_MODULE, SET_OPS_MODULE, SORT_MODULE,
};
use crate::memory::TrackedCudaSlice;
use crate::CudaBuffer;

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
    /// A buffer with duplicate rows removed
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

        // SAFETY: mark_unique_and_scan_columnar(col_ptrs, col_sizes, col_types, num_key_cols,
        //                                       num_rows_device, row_cap,
        //                                       unique_mask, prefix_sum, block_sums)
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

        // Keep the single-column U32 fast path; all other cases use hash-based anti-join.
        if a.arity() == 1 && matches!(col_type, ScalarType::U32) && num_b != 0 {
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
        let grid_size = (num_a + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Kernel signature matches:
        // sorted_diff_mark(a, a_len_device, a_cap, b, b_len_device, b_cap, in_diff)
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

        let a_rows = self.device_row_count(&deduped_a)?;
        let b_rows = self.device_row_count(&deduped_b)?;
        if a_rows == 0 {
            return self.create_empty_buffer(a.schema().clone());
        }

        if b_rows == 0 {
            return Ok(deduped_a);
        }

        // Anti-join: return rows from a that don't match any row in b
        // Since we're doing set difference, all columns are keys
        self.hash_join_v2(&deduped_a, &deduped_b, &all_cols, &all_cols, JoinType::Anti)
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

        // SAFETY: init_indices(indices, num_rows_device, row_cap)
        unsafe {
            init_fn
                .clone()
                .launch(launch_config, (&mut indices_a, d_num_rows, n))
        }
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

        let device = self.device.inner();
        let block_size = Self::SORT_BLOCK_SIZE;
        let grid_size = (row_cap + block_size - 1) / block_size;

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
            // SAFETY: compute_ranks(keys, num_rows_device, row_cap, ranks, shift)
            unsafe {
                ranks_fn.clone().launch(
                    sort_config,
                    (keys_in, num_rows_device, row_cap, &mut *ranks, shift),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("compute_ranks failed: {}", e)))?;

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
        let grid_size = (n + block_size - 1) / block_size;
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
        let grid_size = (n + block_size - 1) / block_size;
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
        let grid_size = (n + block_size - 1) / block_size;
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
        let grid_size = (n + block_size - 1) / block_size;
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
        let grid_size = (n + block_size - 1) / block_size;
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
        let num_blocks = (n + block_size - 1) / block_size;

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
        self.device
            .inner()
            .htod_sync_copy_into(&[0u32], &mut d_count)
            .map_err(|e| XlogError::Kernel(format!("count_mask_device: zero init failed: {}", e)))?;

        if n == 0 {
            return Ok(d_count);
        }

        let device = self.device.inner();
        let block_size = 256u32;
        let grid_size = (n + block_size - 1) / block_size;

        let count_fn = device
            .get_func(SCAN_MODULE, scan_kernels::COUNT_MASK)
            .ok_or_else(|| {
                XlogError::Kernel("count_mask kernel not found".to_string())
            })?;

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
        let grid_size = (n + block_size - 1) / block_size;

        let count_fn = device
            .get_func(SCAN_MODULE, scan_kernels::COUNT_MASK)
            .ok_or_else(|| {
                XlogError::Kernel("count_mask kernel not found".to_string())
            })?;

        // Get a mutable sub-slice pointing at task_counts[slot_idx..slot_idx+1].
        let mut slot = task_counts.slice_mut(slot_idx..slot_idx + 1);

        // SAFETY: count_mask(mask, n, count) — writes atomicAdd into count ptr.
        // The slot was pre-zeroed by the caller.
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

        let grid_size = (row_cap + Self::SORT_BLOCK_SIZE - 1) / Self::SORT_BLOCK_SIZE;
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
            col_ptrs.push(*col.device_ptr() as u64);
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
        let grid_size = (num_rows + block_size - 1) / block_size;
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
        } = self.build_hash_table_v2(&*hashes, num_rows)?;
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
        self.dtoh_sync_copy_into_tracked(&d_count_only, &mut count_host)
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
        self.dtoh_sync_copy_into_tracked(&d_output_count, &mut count_host)
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
        self.dtoh_sync_copy_into_tracked(&d_count_only, &mut count_host)
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
        self.dtoh_sync_copy_into_tracked(&d_output_count, &mut count_host)
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
            self.device
                .inner()
                .memset_zeros(&mut d_mask)
                .map_err(|e| {
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
        let grid_size = (num_probe_u32 + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: hash_join_semi(probe_hashes, num_probe,
        //                        bucket_offsets, bucket_counts, bucket_entries,
        //                        bucket_entry_hashes, bucket_mask,
        //                        probe_keys, build_keys, key_bytes, has_match)
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
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to download membership mask: {}", e))
            })?;
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
        self.dtoh_sync_copy_into_tracked(&d_count_only, &mut count_host)
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
        self.dtoh_sync_copy_into_tracked(&d_output_count, &mut count_host)
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
            result_columns.extend(inner_left.columns.into_iter());
            result_columns.extend(inner_right.columns.into_iter());
            return self.buffer_from_columns(result_columns, inner_count as u64, combined_schema);
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
            return self.buffer_from_columns(result_columns, unmatched_rows, combined_schema);
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
        self.dtoh_sync_copy_into_tracked(&d_count_only, &mut count_host)
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
        self.dtoh_sync_copy_into_tracked(&d_output_count, &mut count_host)
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
            result_columns.extend(inner_left.columns.into_iter());
            result_columns.extend(inner_right.columns.into_iter());
            return self.buffer_from_columns(result_columns, inner_count as u64, combined_schema);
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
            return self.buffer_from_columns(result_columns, unmatched_rows, combined_schema);
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

    /// Clone a buffer (deep copy) on-device
    pub(super) fn clone_buffer(&self, buffer: &CudaBuffer) -> Result<CudaBuffer> {
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

        Ok(CudaBuffer::from_columns(
            result_columns,
            buffer.row_cap,
            d_num_rows,
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
        let grid_size = (total + block_size - 1) / block_size;

        let mut out_i = self.memory().alloc::<u32>(total)?;
        let mut out_j = self.memory().alloc::<u32>(total)?;
        let mut out_k = self.memory().alloc::<u32>(total)?;
        let mut out_p = self.memory().alloc::<f32>(total)?;
        let mut count = self.memory().alloc::<u32>(1)?;

        self.device()
            .inner()
            .htod_sync_copy_into(&[0u32], &mut count)
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
            .ok_or_else(|| {
                XlogError::Kernel("extract_nonzero_indices kernel not found".into())
            })?;

        let hard_bytes = total * std::mem::size_of::<f32>();
        let soft_bytes = total * std::mem::size_of::<f32>();
        let hard_view = self.column_bytes_view(hard_col, hard_bytes)?;
        let soft_view = self.column_bytes_view(soft_col, soft_bytes)?;

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
                        &hard_view,
                        &soft_view,
                        n as u32,
                        &mut out_i,
                        &mut out_j,
                        &mut out_k,
                        &mut out_p,
                        &mut count,
                    ),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "Failed to launch extract_nonzero_indices: {}",
                        e
                    ))
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
        indices.sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(max_active);

        Ok(indices.into_iter().map(|(_, i, j, k)| (i, j, k)).collect())
    }
}
