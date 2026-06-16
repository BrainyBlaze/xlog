//! Groupby aggregation operations: groupby_agg, groupby_multi_agg.

use crate::{LaunchAsync, LaunchConfig};
use xlog_core::{AggOp, Result, ScalarType, Schema, XlogError};

use super::{
    arith_kernels, groupby_kernels, pack_kernels, scan_kernels, ARITH_MODULE, GROUPBY_MODULE,
    PACK_MODULE, SCAN_MODULE,
};
use crate::memory::{CudaColumn, TrackedCudaSlice};
use crate::CudaBuffer;

impl super::CudaKernelProvider {
    /// Perform groupby aggregation
    ///
    /// Assumes input is already sorted by key columns.
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
        self.groupby_multi_agg(input, key_cols, &[(value_col, agg)])
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
        // Env-gated recorded dispatch. `groupby_multi_agg_recorded`
        // is narrow to U32 / Symbol keys + Count /
        // Sum / Min / Max aggs + ≤4 key columns. Mismatch
        // (any other key type, LogSumExp, or >4 keys) falls
        // through to the legacy path.
        if Self::use_recorded_groupby_env()
            && !key_cols.is_empty()
            && !aggs.is_empty()
            && key_cols.len() <= 4
        {
            if let Some(launch_stream) = self.recorded_op_stream_or_init() {
                let keys_compatible = key_cols.iter().all(|&k| {
                    matches!(
                        buffer.schema.column_type(k),
                        Some(ScalarType::U32) | Some(ScalarType::Symbol)
                    )
                });
                let aggs_compatible = aggs.iter().all(|&(_, op)| {
                    matches!(op, AggOp::Count | AggOp::Sum | AggOp::Min | AggOp::Max)
                });
                if keys_compatible && aggs_compatible {
                    return self.groupby_multi_agg_recorded(buffer, key_cols, aggs, launch_stream);
                }
            }
        }
        let num_rows = self.device_row_count(buffer)?;
        if num_rows == 0 {
            let result_schema =
                self.groupby_multi_agg_result_schema(buffer.schema(), key_cols, aggs);
            return self.create_empty_buffer(result_schema);
        }
        if num_rows > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "GroupBy supports at most {} rows, got {}",
                u32::MAX,
                num_rows
            )));
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
                // U64 value-column widening: values reduce through the u64-value
                // kernels (the legacy groupby is the unfused baseline for
                // u64-key WCOJ relations, whose value columns are U64).
                AggOp::Sum | AggOp::Min | AggOp::Max => {
                    if !matches!(value_ty, ScalarType::U32 | ScalarType::U64) {
                        return Err(XlogError::Kernel(format!(
                            "{:?} currently requires U32 or U64 values, got {:?}",
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

        // Step 1: Sort buffer by key columns
        let sorted = self.sort(buffer, key_cols)?;
        let num_rows = self.device_row_count(&sorted)?;
        if num_rows > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "GroupBy supports at most {} rows, got {}",
                u32::MAX,
                num_rows
            )));
        }
        let num_rows = num_rows as u32;

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
        let grid_size = num_rows.div_ceil(block_size);
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Kernel parameters match expected signature
        unsafe {
            boundary_func.clone().launch(
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
        let d_num_groups = self.capture_num_groups(&d_boundary_pos, &boundaries, num_rows)?;
        let row_cap = num_rows as u64;
        let row_cap_usize = num_rows as usize;
        let row_cap_u32 = num_rows;

        let mut group_ids = self.memory.alloc::<u32>(num_rows as usize)?;
        let mut group_first_idx = self.memory.alloc::<u32>(row_cap_usize)?;

        let group_ids_fn = device
            .get_func(GROUPBY_MODULE, groupby_kernels::GROUP_IDS_FROM_BOUNDARIES)
            .ok_or_else(|| {
                XlogError::Kernel("group_ids_from_boundaries kernel not found".to_string())
            })?;
        let group_start_fn = device
            .get_func(GROUPBY_MODULE, groupby_kernels::GROUP_START_INDICES)
            .ok_or_else(|| XlogError::Kernel("group_start_indices kernel not found".to_string()))?;

        // SAFETY: group_ids_from_boundaries(boundaries, boundary_pos, num_rows, group_ids)
        unsafe {
            group_ids_fn.clone().launch(
                config,
                (&boundaries, &d_boundary_pos, num_rows, &mut group_ids),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("group_ids_from_boundaries failed: {}", e)))?;

        // SAFETY: group_start_indices(boundaries, boundary_pos, num_rows, group_first_idx)
        unsafe {
            group_start_fn.clone().launch(
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
                    let output_bytes = row_cap_usize
                        .checked_mul(std::mem::size_of::<u64>())
                        .ok_or_else(|| {
                            XlogError::Kernel("Count output size overflow".to_string())
                        })?;
                    let mut output = self.memory.alloc::<u8>(output_bytes)?;
                    device.memset_zeros(&mut output).map_err(|e| {
                        XlogError::Kernel(format!("Failed to zero count output: {}", e))
                    })?;

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
                    let value_ty = sorted
                        .schema()
                        .column_type(value_col)
                        .ok_or_else(|| XlogError::Kernel("Value column has no type".to_string()))?;
                    let output_bytes = row_cap_usize
                        .checked_mul(std::mem::size_of::<u64>())
                        .ok_or_else(|| XlogError::Kernel("Sum output size overflow".to_string()))?;
                    let mut output = self.memory.alloc::<u8>(output_bytes)?;
                    device.memset_zeros(&mut output).map_err(|e| {
                        XlogError::Kernel(format!("Failed to zero sum output: {}", e))
                    })?;

                    // U64 value-column widening: value columns reduce through the
                    // u64-value sum kernel (same u64 accumulator).
                    if value_ty == ScalarType::U64 {
                        let values_view = self.column_as_u64_view(values, num_rows as usize)?;
                        let sum_func = device
                            .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_SUM_U64)
                            .ok_or_else(|| {
                                XlogError::Kernel("groupby_sum_u64 kernel not found".to_string())
                            })?;
                        // SAFETY: groupby_sum_u64(values, group_ids, num_rows, sums)
                        unsafe {
                            sum_func
                                .clone()
                                .launch(config, (&values_view, &group_ids, num_rows, &output))
                        }
                        .map_err(|e| XlogError::Kernel(format!("groupby_sum_u64 failed: {}", e)))?;
                    } else {
                        let values_view = self.column_as_u32_view(values, num_rows as usize)?;
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
                    }

                    self.device.synchronize()?;
                    agg_columns.push(output.into());
                }
                AggOp::Min => {
                    let value_ty = sorted
                        .schema()
                        .column_type(value_col)
                        .ok_or_else(|| XlogError::Kernel("Value column has no type".to_string()))?;
                    if value_ty == ScalarType::U64 {
                        // U64 value-column min path (output U64,
                        // identity u64::MAX).
                        let values_view = self.column_as_u64_view(values, num_rows as usize)?;
                        let output_bytes = row_cap_usize
                            .checked_mul(std::mem::size_of::<u64>())
                            .ok_or_else(|| {
                                XlogError::Kernel("Min output size overflow".to_string())
                            })?;
                        let mut output = self.memory.alloc::<u8>(output_bytes)?;
                        let fill_fn = device
                            .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_U64)
                            .ok_or_else(|| {
                                XlogError::Kernel("arith_fill_const_u64 not found".to_string())
                            })?;
                        let fill_config = LaunchConfig::for_num_elems(row_cap_u32);
                        // SAFETY: arith_fill_const_u64(value, n, output)
                        unsafe {
                            fill_fn
                                .clone()
                                .launch(fill_config, (u64::MAX, row_cap_u32, &mut output))
                        }
                        .map_err(|e| {
                            XlogError::Kernel(format!("Failed to init min output: {}", e))
                        })?;

                        let min_func = device
                            .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MIN_U64)
                            .ok_or_else(|| {
                                XlogError::Kernel("groupby_min_u64 kernel not found".to_string())
                            })?;
                        // SAFETY: groupby_min_u64(values, group_ids, num_rows, mins)
                        unsafe {
                            min_func
                                .clone()
                                .launch(config, (&values_view, &group_ids, num_rows, &output))
                        }
                        .map_err(|e| XlogError::Kernel(format!("groupby_min_u64 failed: {}", e)))?;

                        self.device.synchronize()?;
                        agg_columns.push(output.into());
                    } else {
                        let values_view = self.column_as_u32_view(values, num_rows as usize)?;
                        let output_bytes = row_cap_usize
                            .checked_mul(std::mem::size_of::<u32>())
                            .ok_or_else(|| {
                                XlogError::Kernel("Min output size overflow".to_string())
                            })?;
                        let mut output = self.memory.alloc::<u8>(output_bytes)?;
                        let fill_fn = device
                            .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_U32)
                            .ok_or_else(|| {
                                XlogError::Kernel("arith_fill_const_u32 not found".to_string())
                            })?;
                        let fill_config = LaunchConfig::for_num_elems(row_cap_u32);
                        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                        unsafe {
                            fill_fn
                                .clone()
                                .launch(fill_config, (u32::MAX, row_cap_u32, &mut output))
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
                }
                AggOp::Max => {
                    let value_ty = sorted
                        .schema()
                        .column_type(value_col)
                        .ok_or_else(|| XlogError::Kernel("Value column has no type".to_string()))?;
                    if value_ty == ScalarType::U64 {
                        // U64 value-column max path (output U64,
                        // identity 0).
                        let values_view = self.column_as_u64_view(values, num_rows as usize)?;
                        let output_bytes = row_cap_usize
                            .checked_mul(std::mem::size_of::<u64>())
                            .ok_or_else(|| {
                                XlogError::Kernel("Max output size overflow".to_string())
                            })?;
                        let mut output = self.memory.alloc::<u8>(output_bytes)?;
                        device.memset_zeros(&mut output).map_err(|e| {
                            XlogError::Kernel(format!("Failed to zero max output: {}", e))
                        })?;

                        let max_func = device
                            .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MAX_U64)
                            .ok_or_else(|| {
                                XlogError::Kernel("groupby_max_u64 kernel not found".to_string())
                            })?;
                        // SAFETY: groupby_max_u64(values, group_ids, num_rows, maxs)
                        unsafe {
                            max_func
                                .clone()
                                .launch(config, (&values_view, &group_ids, num_rows, &output))
                        }
                        .map_err(|e| XlogError::Kernel(format!("groupby_max_u64 failed: {}", e)))?;

                        self.device.synchronize()?;
                        agg_columns.push(output.into());
                    } else {
                        let values_view = self.column_as_u32_view(values, num_rows as usize)?;
                        let output_bytes = row_cap_usize
                            .checked_mul(std::mem::size_of::<u32>())
                            .ok_or_else(|| {
                                XlogError::Kernel("Max output size overflow".to_string())
                            })?;
                        let mut output = self.memory.alloc::<u8>(output_bytes)?;
                        device.memset_zeros(&mut output).map_err(|e| {
                            XlogError::Kernel(format!("Failed to zero max output: {}", e))
                        })?;

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
                }
                AggOp::LogSumExp => {
                    let values_f64 = self.column_as_f64_view(values, num_rows as usize)?;
                    let output_bytes = row_cap_usize
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
                    let fill_config = LaunchConfig::for_num_elems(row_cap_u32);
                    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                    unsafe {
                        fill_f64
                            .clone()
                            .launch(fill_config, (f64::NEG_INFINITY, row_cap_u32, &mut maxs))
                    }
                    .map_err(|e| XlogError::Kernel(format!("Failed to init maxs: {}", e)))?;
                    device
                        .memset_zeros(&mut sumexps)
                        .map_err(|e| XlogError::Kernel(format!("Failed to init sumexps: {}", e)))?;

                    let max_func = device
                        .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_LOGSUMEXP_MAX)
                        .ok_or_else(|| {
                            XlogError::Kernel("groupby_logsumexp_max kernel not found".to_string())
                        })?;

                    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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

                    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                    unsafe {
                        sumexp_func
                            .clone()
                            .launch(config, (&values_f64, &group_ids, &maxs, num_rows, &sumexps))
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!("groupby_logsumexp_sumexp failed: {}", e))
                    })?;

                    self.device.synchronize()?;

                    let final_config = LaunchConfig::for_num_elems(row_cap_u32);
                    let final_func = device
                        .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_LOGSUMEXP_FINAL)
                        .ok_or_else(|| {
                            XlogError::Kernel(
                                "groupby_logsumexp_final kernel not found".to_string(),
                            )
                        })?;

                    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                    unsafe {
                        final_func.clone().launch(
                            final_config,
                            (&maxs, &sumexps, &d_num_groups, row_cap_u32, &results),
                        )
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
        let mut result_columns: Vec<CudaColumn> = Vec::with_capacity(key_cols.len() + aggs.len());

        let group_packed_bytes = row_cap_usize
            .checked_mul(packed.key_bytes as usize)
            .ok_or_else(|| XlogError::Kernel("GroupBy packed size overflow".to_string()))?;
        let mut group_packed = self.memory.alloc::<u8>(group_packed_bytes)?;

        let gather_fn = device
            .get_func(PACK_MODULE, pack_kernels::GATHER_PACKED_ROWS_COUNTED)
            .ok_or_else(|| {
                XlogError::Kernel("gather_packed_rows_counted kernel not found".to_string())
            })?;
        let gather_config = LaunchConfig::for_num_elems(row_cap_u32);

        // SAFETY: gather_packed_rows_counted(src_packed, row_size, indices, num_rows, capacity_rows, dst_packed)
        unsafe {
            gather_fn.clone().launch(
                gather_config,
                (
                    &packed.packed_keys,
                    packed.key_bytes,
                    &group_first_idx,
                    &d_num_groups,
                    row_cap_u32,
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
            .get_func(PACK_MODULE, pack_kernels::UNPACK_COLUMN_COUNTED)
            .ok_or_else(|| {
                XlogError::Kernel("unpack_column_counted kernel not found".to_string())
            })?;
        let unpack_config = LaunchConfig::for_num_elems(row_cap_u32);

        for idx in 0..key_cols.len() {
            let col_size = col_sizes[idx];
            let col_offset = col_offsets[idx];
            let out_bytes = row_cap_usize
                .checked_mul(col_size as usize)
                .ok_or_else(|| XlogError::Kernel("GroupBy key column overflow".to_string()))?;
            let mut out_col = self.memory.alloc::<u8>(out_bytes)?;

            // SAFETY: unpack_column_counted(packed_input, row_size, col_offset, col_size, num_rows, capacity_rows, col_output)
            unsafe {
                unpack_fn.clone().launch(
                    unpack_config,
                    (
                        &group_packed,
                        packed.key_bytes,
                        col_offset,
                        col_size,
                        &d_num_groups,
                        row_cap_u32,
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
            row_cap,
            d_num_groups,
            result_schema,
        ))
    }

    fn capture_num_groups(
        &self,
        boundary_pos: &TrackedCudaSlice<u32>,
        boundaries: &TrackedCudaSlice<u8>,
        num_rows: u32,
    ) -> Result<TrackedCudaSlice<u32>> {
        let mut d_num_groups = self.memory.alloc::<u32>(1)?;
        let capture_fn = self
            .device
            .inner()
            .get_func(GROUPBY_MODULE, groupby_kernels::CAPTURE_NUM_GROUPS)
            .ok_or_else(|| XlogError::Kernel("capture_num_groups kernel not found".to_string()))?;
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            capture_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (boundary_pos, boundaries, num_rows, &mut d_num_groups),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("capture_num_groups failed: {}", e)))?;
        Ok(d_num_groups)
    }

    /// Create result schema for multi-aggregation groupby
    pub(crate) fn groupby_multi_agg_result_schema(
        &self,
        input: &Schema,
        key_cols: &[usize],
        aggs: &[(usize, AggOp)],
    ) -> Schema {
        let mut columns: Vec<(String, ScalarType)> = key_cols
            .iter()
            .filter_map(|&i| input.columns.get(i).cloned())
            .collect();
        let mut sort_labels: Vec<String> = key_cols
            .iter()
            .filter_map(|&i| {
                input
                    .column_sort_label(i)
                    .map(ToString::to_string)
                    .or_else(|| input.columns.get(i).map(|(name, _)| name.clone()))
            })
            .collect();

        for (i, &(value_col, agg_op)) in aggs.iter().enumerate() {
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
                // Value-width preserving min/max: preserve the value column's width
                // (U64 values reduce to U64; everything else stays U32).
                AggOp::Min | AggOp::Max => match input.columns.get(value_col).map(|(_, ty)| *ty) {
                    Some(ScalarType::U64) => ScalarType::U64,
                    _ => ScalarType::U32,
                },
                AggOp::LogSumExp => ScalarType::F64,
            };
            columns.push((agg_name, agg_type));
            sort_labels.push(format!("aggregate_{}", i));
        }

        Schema::new(columns)
            .with_sort_labels(sort_labels)
            .expect("groupby result sort labels match column arity")
    }

    // ======================================================================
    // Recorded GroupBy provider-level path
    //
    // Strict-recorder, launch_stream-routed sibling of `groupby_multi_agg`.
    // Scope-narrow per the slice directive:
    //   * U32 / Symbol key columns only (delegates to sort_recorded which has
    //     the same constraint).
    //   * Aggs: Count, Sum, Min, Max only. LogSumExp is a multi-kernel
    //     chain (max → sumexp → final) and is outside this API surface.
    //   * No legacy default-routed code is touched. The legacy
    //     `groupby_multi_agg` and `groupby_agg` keep their semantics
    //     bit-for-bit; runtime/planner wiring is NOT included.
    // ======================================================================

    /// Stream-aware variant of `pack_keys_gpu` (≤4 columns).
    /// Mirrors the legacy fused `pack_and_hash_keys` launch on
    /// `launch_stream`, then records the kernel's intermediate
    /// returned outputs (`packed_keys`, `hashes`) against the runtime so that
    /// downstream consumers / drops are correctly serialized
    /// against `launch_stream`.
    pub(super) fn pack_keys_gpu_on_stream(
        &self,
        buffer: &CudaBuffer,
        key_cols: &[usize],
        cu_stream: &cudarc::driver::CudaStream,
        launch_stream: crate::device_runtime::StreamId,
        runtime: &crate::device_runtime::XlogDeviceRuntime,
    ) -> Result<crate::provider::PackedKeyData> {
        use crate::launch::LaunchRecorder;

        if key_cols.is_empty() {
            return Err(XlogError::Kernel(
                "pack_keys_gpu_on_stream: no key columns specified".to_string(),
            ));
        }
        if key_cols.len() > 4 {
            return Err(XlogError::Kernel(
                "pack_keys_gpu_on_stream: max 4 key columns supported".to_string(),
            ));
        }
        let num_rows = self.device_row_count(buffer)?;
        if num_rows > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "pack_keys_gpu_on_stream supports at most {} rows, got {}",
                u32::MAX,
                num_rows
            )));
        }
        let num_rows = num_rows as u32;

        let mut col_sizes_host: Vec<u32> = Vec::with_capacity(key_cols.len());
        let mut row_size: u32 = 0;
        for &col_idx in key_cols {
            let ty = buffer
                .schema()
                .column_type(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Invalid column index: {}", col_idx)))?;
            let s = ty.size_bytes() as u32;
            col_sizes_host.push(s);
            row_size += s;
        }

        if num_rows == 0 {
            return Ok(crate::provider::PackedKeyData {
                hashes: self.memory.alloc::<u64>(0)?,
                packed_keys: self.memory.alloc::<u8>(0)?,
                key_bytes: row_size,
            });
        }

        let packed_bytes = (num_rows as u64) * (row_size as u64);
        let packed_slice = self.memory.alloc::<u8>(packed_bytes as usize)?;
        let hash_slice = self.memory.alloc::<u64>(num_rows as usize)?;

        let mut col_ptrs: [u64; 4] = [0; 4];
        for (i, &col_idx) in key_cols.iter().enumerate() {
            let col = buffer
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Key column {} not found", col_idx)))?;
            col_ptrs[i] = *col.device_ptr();
        }
        let mut packed_col_sizes = 0u64;
        for (i, size) in col_sizes_host.iter().copied().enumerate() {
            if size > u16::MAX as u32 {
                return Err(XlogError::Kernel(format!(
                    "pack_keys_gpu_on_stream: column element size {} exceeds 16-bit kernel argument",
                    size
                )));
            }
            packed_col_sizes |= (size as u64) << (i * 16);
        }

        // The pack kernel takes raw column pointers (`u64`)
        // rather than typed `CudaColumn` kernel params, so the
        // generic launch recorder cannot infer source-column
        // lifetimes from the argument list. Record those reads
        // explicitly before queueing the launch; this also
        // enforces the strict external-memory policy for
        // recorded paths.
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        for &col_idx in key_cols {
            let col = buffer
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Key column {} not found", col_idx)))?;
            rec.read_column(col);
        }
        rec.write(&packed_slice);
        rec.write(&hash_slice);
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "pack_keys_gpu_on_stream: launch recorder preflight failed: {}",
                e
            ))
        })?;

        let func = self
            .device
            .inner()
            .get_func(PACK_MODULE, pack_kernels::PACK_AND_HASH_KEYS)
            .ok_or_else(|| XlogError::Kernel("pack_and_hash_keys kernel not found".to_string()))?;
        let block_size = 256u32;
        let grid_size = num_rows.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: pack_and_hash_keys signature.
        unsafe {
            func.clone().launch_on_stream(
                cu_stream,
                cfg,
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
        }
        .map_err(|e| XlogError::Kernel(format!("pack_and_hash_keys (on_stream) failed: {}", e)))?;

        // Record uses for buffers touched on launch_stream.
        // `packed_slice` / `hash_slice` are fresh outputs that
        // escape to the caller. The post-preflight-fresh path is
        // valid because they were allocated by this helper before
        // preflight and first used by the queued pack launch.
        rec.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "pack_keys_gpu_on_stream: launch recorder commit failed: {}",
                e
            ))
        })?;

        Ok(crate::provider::PackedKeyData {
            hashes: hash_slice,
            packed_keys: packed_slice,
            key_bytes: row_size,
        })
    }

    /// Async u8 zero-fill on `cu_stream` via `cuMemsetD8Async`.
    /// Used by recorded GroupBy aggregations that need a
    /// freshly zeroed output buffer (Count, Sum, Max).
    fn memset_zeros_u8_on_stream(
        &self,
        buf: &mut TrackedCudaSlice<u8>,
        cu_stream: &cudarc::driver::CudaStream,
    ) -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let ptr = *buf.device_ptr();
        let len = <TrackedCudaSlice<u8> as crate::DeviceSlice<u8>>::len(buf);
        // SAFETY: ptr is a live runtime-backed device pointer
        // for `len` bytes, cu_stream is a valid CUDA stream
        // owned by the runtime's pool. cuMemsetD8Async queues
        // and returns immediately.
        unsafe {
            let res = cudarc::driver::sys::cuMemsetD8Async(ptr, 0, len, cu_stream.cu_stream());
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "cuMemsetD8Async (groupby init) failed: {:?}",
                    res
                )));
            }
        }
        Ok(())
    }

    /// Strict-recorder variant of [`Self::groupby_multi_agg`].
    ///
    /// Sort + pack + boundary detect + scan + capture-num-groups
    /// + group-id derivation + per-aggregation kernels + key
    ///   gather/unpack — every kernel runs on the caller-supplied
    ///   `launch_stream` via `launch_on_stream`. Composition with
    ///   existing recorded primitives:
    ///   * `sort_recorded` does the typed multi-column
    ///     sort and commits its own LaunchRecorder.
    ///   * `pack_keys_gpu_on_stream` runs the fused
    ///     pack+hash kernel on launch_stream and records its
    ///     buffers directly via `record_block_use`.
    ///   * `multiblock_scan_u32_inplace_on_stream`
    ///     drives the boundary-position scan tail.
    ///   * The groupby-specific chain has its own LaunchRecorder
    ///     for the boundary mask, group ids, group_first
    ///     indices, num_groups scalar, per-aggregation outputs,
    ///     and key gather/unpack outputs.
    ///
    /// Composition correctness: each recorder commits
    /// independently; the runtime's record-all + wait-all
    /// `last_use_events: Vec<CudaEvent>` semantics chain the
    /// deallocate safety end-to-end across the four primitive
    /// commits.
    ///
    /// # Scope (narrow)
    /// * U32 / Symbol key columns only (sort_recorded
    ///   constraint).
    /// * Aggs: Count, Sum, Min, Max. LogSumExp is rejected with
    ///   a structured error — its multi-kernel chain is
    ///   outside this recorded provider surface.
    /// * Manager must be runtime-backed.
    pub fn groupby_multi_agg_recorded(
        &self,
        buffer: &CudaBuffer,
        key_cols: &[usize],
        aggs: &[(usize, AggOp)],
        launch_stream: crate::device_runtime::StreamId,
    ) -> Result<CudaBuffer> {
        use crate::launch::LaunchRecorder;

        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "groupby_multi_agg_recorded requires a runtime-backed GpuMemoryManager".to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "groupby_multi_agg_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        let num_rows = self.device_row_count(buffer)?;
        if num_rows == 0 {
            let result_schema =
                self.groupby_multi_agg_result_schema(buffer.schema(), key_cols, aggs);
            return self.create_empty_buffer(result_schema);
        }
        if num_rows > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "GroupBy supports at most {} rows, got {}",
                u32::MAX,
                num_rows
            )));
        }
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
        if key_cols.len() > 4 {
            return Err(XlogError::Kernel(
                "groupby_multi_agg_recorded: max 4 key columns supported (pack_keys constraint)"
                    .to_string(),
            ));
        }
        for &k in key_cols {
            if k >= buffer.arity() {
                return Err(XlogError::Kernel(format!(
                    "Key column {} out of bounds (arity {})",
                    k,
                    buffer.arity()
                )));
            }
            let ty = buffer
                .schema()
                .column_type(k)
                .ok_or_else(|| XlogError::Kernel("Key column has no type".to_string()))?;
            if !matches!(ty, ScalarType::U32 | ScalarType::Symbol) {
                return Err(XlogError::Kernel(format!(
                    "groupby_multi_agg_recorded: key column type {:?} unsupported (U32 / Symbol \
                     only); multi-type sort_recorded is deferred",
                    ty
                )));
            }
        }
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
                AggOp::Sum => {
                    // Recorded groupby U64 value-column sum path: values reduce through the u64-value
                    // sum kernel (same u64 accumulator as the U32 path).
                    if !matches!(value_ty, ScalarType::U32 | ScalarType::U64) {
                        return Err(XlogError::Kernel(format!(
                            "Sum currently requires U32 or U64 values, got {:?}",
                            value_ty
                        )));
                    }
                }
                AggOp::Min | AggOp::Max => {
                    // Recorded groupby U64 value-column min/max path: values reduce through the
                    // u64-value min/max kernels (result preserves the
                    // value width, mirroring the legacy path).
                    if !matches!(value_ty, ScalarType::U32 | ScalarType::U64) {
                        return Err(XlogError::Kernel(format!(
                            "{:?} currently requires U32 or U64 values, got {:?}",
                            agg_op, value_ty
                        )));
                    }
                }
                AggOp::LogSumExp => {
                    return Err(XlogError::Kernel(
                        "groupby_multi_agg_recorded: LogSumExp not yet supported in the \
                        recorded path (multi-kernel chain deferred to a future implementation)"
                            .to_string(),
                    ));
                }
            }
        }

        // Step 1: sort by key columns (recorded sort, U32/Symbol only).
        let sorted = self.sort_recorded(buffer, key_cols, launch_stream)?;
        let num_rows = self.device_row_count(&sorted)?;
        if num_rows > u32::MAX as usize {
            return Err(XlogError::Kernel(format!(
                "GroupBy supports at most {} rows, got {}",
                u32::MAX,
                num_rows
            )));
        }
        let num_rows = num_rows as u32;
        let row_cap_usize = num_rows as usize;
        let row_cap_u32 = num_rows;
        let row_cap_u64 = num_rows as u64;

        // Step 2: pack keys on launch_stream.
        let packed =
            self.pack_keys_gpu_on_stream(&sorted, key_cols, &cu_stream, launch_stream, runtime)?;
        if packed.key_bytes == 0 || packed.key_bytes % 4 != 0 {
            return Err(XlogError::Kernel(format!(
                "GroupBy key packing produced {} bytes per row (expected multiple of 4); \
                 Bool keys are not supported",
                packed.key_bytes
            )));
        }
        let segments_per_row = (packed.key_bytes / 4) as usize;
        let total_segments = row_cap_usize * segments_per_row;
        let packed_u32 = self.bytes_as_u32_view(&packed.packed_keys, total_segments)?;

        // Step 3: allocate ALL fresh runtime-backed buffers
        // BEFORE the GroupBy recorder (Rust drop order — the
        // recorder's 'b lifetime must outlive every borrow it
        // holds via post_preflight_fresh).
        let boundaries = self.memory.alloc::<u8>(row_cap_usize)?;
        let block_size = 256u32;
        let num_blocks = num_rows.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let d_boundary_pos = self.memory.alloc::<u32>(row_cap_usize)?;
        let mut d_block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;
        let mut d_num_groups = self.memory.alloc::<u32>(1)?;
        let mut group_ids = self.memory.alloc::<u32>(row_cap_usize)?;
        let mut group_first_idx = self.memory.alloc::<u32>(row_cap_usize)?;

        // Per-aggregation outputs (allocated up front).
        let mut agg_outputs: Vec<TrackedCudaSlice<u8>> = Vec::with_capacity(aggs.len());
        for &(value_col, agg_op) in aggs {
            let elem_size = match agg_op {
                AggOp::Count | AggOp::Sum => std::mem::size_of::<u64>(),
                // Value-width preserving min/max: preserve the value column's width.
                AggOp::Min | AggOp::Max => match sorted.schema().column_type(value_col) {
                    Some(ScalarType::U64) => std::mem::size_of::<u64>(),
                    _ => std::mem::size_of::<u32>(),
                },
                AggOp::LogSumExp => unreachable!("rejected above"),
            };
            let bytes = row_cap_usize
                .checked_mul(elem_size)
                .ok_or_else(|| XlogError::Kernel("groupby agg output size overflow".to_string()))?;
            agg_outputs.push(self.memory.alloc::<u8>(bytes)?);
        }

        // Key gather + unpack outputs.
        let group_packed_bytes = row_cap_usize
            .checked_mul(packed.key_bytes as usize)
            .ok_or_else(|| XlogError::Kernel("GroupBy packed size overflow".to_string()))?;
        let mut group_packed = self.memory.alloc::<u8>(group_packed_bytes)?;

        let mut col_offsets: Vec<u32> = Vec::with_capacity(key_cols.len());
        let mut col_sizes: Vec<u32> = Vec::with_capacity(key_cols.len());
        let mut offset = 0u32;
        for &key_col in key_cols {
            let s = buffer
                .schema()
                .column_type(key_col)
                .map(|t| t.size_bytes() as u32)
                .unwrap_or(4);
            col_offsets.push(offset);
            col_sizes.push(s);
            offset = offset
                .checked_add(s)
                .ok_or_else(|| XlogError::Kernel("GroupBy key size overflow".to_string()))?;
        }
        let mut key_unpacked: Vec<TrackedCudaSlice<u8>> = Vec::with_capacity(key_cols.len());
        for &col_size in &col_sizes {
            let bytes = row_cap_usize
                .checked_mul(col_size as usize)
                .ok_or_else(|| XlogError::Kernel("GroupBy key column overflow".to_string()))?;
            key_unpacked.push(self.memory.alloc::<u8>(bytes)?);
        }

        // Build the recorder. Reads BEFORE preflight: the
        // sorted buffer's value columns + num_rows_device, plus
        // the packed_keys produced by pack_keys_on_stream
        // (which already recorded its own writes against
        // launch_stream — we record reads here so the chain
        // ordering is explicit).
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(sorted.num_rows_device());
        // sort_recorded already recorded reads on every input
        // column on launch_stream; packed_keys is the new
        // launch_stream-resident input to the boundary chain.
        rec.read(&packed.packed_keys);
        for &(value_col, _) in aggs {
            let c = sorted.column(value_col).ok_or_else(|| {
                XlogError::Kernel(format!("Value column {} not found", value_col))
            })?;
            rec.read_column(c);
        }
        rec.write(&boundaries);
        rec.write(&d_boundary_pos);
        rec.write(&d_block_sums);
        rec.write(&d_num_groups);
        rec.write(&group_ids);
        rec.write(&group_first_idx);
        rec.write(&group_packed);
        for o in &agg_outputs {
            rec.write(o);
        }
        for k in &key_unpacked {
            rec.write(k);
        }
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "groupby_multi_agg_recorded: preflight failed: {}",
                e
            ))
        })?;

        let device = self.device.inner();

        // Step 4: detect_group_boundaries on launch_stream.
        let boundary_func = device
            .get_func(GROUPBY_MODULE, groupby_kernels::DETECT_GROUP_BOUNDARIES)
            .ok_or_else(|| {
                XlogError::Kernel("detect_group_boundaries kernel not found".to_string())
            })?;
        // SAFETY: detect_group_boundaries(packed_u32, num_rows, segments_per_row, segments_per_row, boundaries)
        unsafe {
            boundary_func.clone().launch_on_stream(
                &cu_stream,
                cfg,
                (
                    &packed_u32,
                    num_rows,
                    segments_per_row as u32,
                    segments_per_row as u32,
                    &boundaries,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!("detect_group_boundaries (on_stream) failed: {}", e))
        })?;

        // Step 5: multi-block scan over boundary mask (yielding boundary positions).
        let phase1_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase1 kernel".to_string())
            })?;
        // SAFETY: multiblock_scan_phase1(mask, prefix_sum, block_sums, n)
        unsafe {
            phase1_fn.clone().launch_on_stream(
                &cu_stream,
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&boundaries, &d_boundary_pos, &d_block_sums, num_rows),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!("multiblock_scan_phase1 (on_stream) failed: {}", e))
        })?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace_on_stream(
                &mut d_block_sums,
                num_blocks,
                &cu_stream,
                launch_stream,
                runtime,
            )?;
            let phase3_fn = device
                .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE3)
                .ok_or_else(|| {
                    XlogError::Kernel("Failed to get multiblock_scan_phase3 kernel".to_string())
                })?;
            // SAFETY: multiblock_scan_phase3(prefix_sum, block_offsets, n)
            unsafe {
                phase3_fn.clone().launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (num_blocks, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&d_boundary_pos, &d_block_sums, num_rows),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("multiblock_scan_phase3 (on_stream) failed: {}", e))
            })?;
        }

        // Step 6: capture_num_groups on launch_stream.
        let capture_fn = device
            .get_func(GROUPBY_MODULE, groupby_kernels::CAPTURE_NUM_GROUPS)
            .ok_or_else(|| XlogError::Kernel("capture_num_groups kernel not found".to_string()))?;
        // SAFETY: capture_num_groups(boundary_pos, boundaries, num_rows, num_groups)
        unsafe {
            capture_fn.clone().launch_on_stream(
                &cu_stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&d_boundary_pos, &boundaries, num_rows, &mut d_num_groups),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("capture_num_groups (on_stream) failed: {}", e)))?;

        // Step 7: derive group_ids + group_first_idx on launch_stream.
        let group_ids_fn = device
            .get_func(GROUPBY_MODULE, groupby_kernels::GROUP_IDS_FROM_BOUNDARIES)
            .ok_or_else(|| {
                XlogError::Kernel("group_ids_from_boundaries kernel not found".to_string())
            })?;
        let group_start_fn = device
            .get_func(GROUPBY_MODULE, groupby_kernels::GROUP_START_INDICES)
            .ok_or_else(|| XlogError::Kernel("group_start_indices kernel not found".to_string()))?;
        // SAFETY: matches kernel signatures.
        unsafe {
            group_ids_fn.clone().launch_on_stream(
                &cu_stream,
                cfg,
                (&boundaries, &d_boundary_pos, num_rows, &mut group_ids),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "group_ids_from_boundaries (on_stream) failed: {}",
                e
            ))
        })?;
        unsafe {
            group_start_fn.clone().launch_on_stream(
                &cu_stream,
                cfg,
                (&boundaries, &d_boundary_pos, num_rows, &mut group_first_idx),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("group_start_indices (on_stream) failed: {}", e)))?;

        // Step 8: per-aggregation kernels.
        for ((value_col, agg_op), output) in aggs.iter().zip(agg_outputs.iter_mut()) {
            let values = sorted.column(*value_col).ok_or_else(|| {
                XlogError::Kernel(format!("Value column {} not found", value_col))
            })?;
            match agg_op {
                AggOp::Count => {
                    self.memset_zeros_u8_on_stream(output, &cu_stream)?;
                    let count_func = device
                        .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_COUNT)
                        .ok_or_else(|| {
                            XlogError::Kernel("groupby_count kernel not found".to_string())
                        })?;
                    // SAFETY: groupby_count(boundaries, group_ids, num_rows, counts)
                    unsafe {
                        count_func.clone().launch_on_stream(
                            &cu_stream,
                            cfg,
                            (&boundaries, &group_ids, num_rows, &*output),
                        )
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!("groupby_count (on_stream) failed: {}", e))
                    })?;
                }
                AggOp::Sum => {
                    self.memset_zeros_u8_on_stream(output, &cu_stream)?;
                    let value_ty = sorted
                        .schema()
                        .column_type(*value_col)
                        .ok_or_else(|| XlogError::Kernel("Value column has no type".to_string()))?;
                    if value_ty == ScalarType::U64 {
                        let values_view = self.column_as_u64_view(values, row_cap_usize)?;
                        let sum_func = device
                            .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_SUM_U64)
                            .ok_or_else(|| {
                                XlogError::Kernel("groupby_sum_u64 kernel not found".to_string())
                            })?;
                        // SAFETY: groupby_sum_u64(values, group_ids, num_rows, sums)
                        unsafe {
                            sum_func.clone().launch_on_stream(
                                &cu_stream,
                                cfg,
                                (&values_view, &group_ids, num_rows, &*output),
                            )
                        }
                        .map_err(|e| {
                            XlogError::Kernel(format!("groupby_sum_u64 (on_stream) failed: {}", e))
                        })?;
                    } else {
                        let values_view = self.column_as_u32_view(values, row_cap_usize)?;
                        let sum_func = device
                            .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_SUM)
                            .ok_or_else(|| {
                                XlogError::Kernel("groupby_sum kernel not found".to_string())
                            })?;
                        // SAFETY: groupby_sum(values, group_ids, num_rows, sums)
                        unsafe {
                            sum_func.clone().launch_on_stream(
                                &cu_stream,
                                cfg,
                                (&values_view, &group_ids, num_rows, &*output),
                            )
                        }
                        .map_err(|e| {
                            XlogError::Kernel(format!("groupby_sum (on_stream) failed: {}", e))
                        })?;
                    }
                }
                AggOp::Min => {
                    let value_ty = sorted
                        .schema()
                        .column_type(*value_col)
                        .ok_or_else(|| XlogError::Kernel("Value column has no type".to_string()))?;
                    let fill_config = LaunchConfig::for_num_elems(row_cap_u32);
                    if value_ty == ScalarType::U64 {
                        // U64 value-column min path (output U64,
                        // identity u64::MAX).
                        let fill_fn = device
                            .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_U64)
                            .ok_or_else(|| {
                                XlogError::Kernel("arith_fill_const_u64 not found".to_string())
                            })?;
                        // SAFETY: arith_fill_const_u64(value, n, output)
                        unsafe {
                            fill_fn.clone().launch_on_stream(
                                &cu_stream,
                                fill_config,
                                (u64::MAX, row_cap_u32, &mut *output),
                            )
                        }
                        .map_err(|e| {
                            XlogError::Kernel(format!(
                                "arith_fill_const_u64 (on_stream) failed: {}",
                                e
                            ))
                        })?;
                        let values_view = self.column_as_u64_view(values, row_cap_usize)?;
                        let min_func = device
                            .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MIN_U64)
                            .ok_or_else(|| {
                                XlogError::Kernel("groupby_min_u64 kernel not found".to_string())
                            })?;
                        // SAFETY: groupby_min_u64(values, group_ids, num_rows, mins)
                        unsafe {
                            min_func.clone().launch_on_stream(
                                &cu_stream,
                                cfg,
                                (&values_view, &group_ids, num_rows, &*output),
                            )
                        }
                        .map_err(|e| {
                            XlogError::Kernel(format!("groupby_min_u64 (on_stream) failed: {}", e))
                        })?;
                    } else {
                        let fill_fn = device
                            .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_U32)
                            .ok_or_else(|| {
                                XlogError::Kernel("arith_fill_const_u32 not found".to_string())
                            })?;
                        // SAFETY: arith_fill_const_u32(value, n, output)
                        unsafe {
                            fill_fn.clone().launch_on_stream(
                                &cu_stream,
                                fill_config,
                                (u32::MAX, row_cap_u32, &mut *output),
                            )
                        }
                        .map_err(|e| {
                            XlogError::Kernel(format!(
                                "arith_fill_const_u32 (on_stream) failed: {}",
                                e
                            ))
                        })?;
                        let values_view = self.column_as_u32_view(values, row_cap_usize)?;
                        let min_func = device
                            .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MIN)
                            .ok_or_else(|| {
                                XlogError::Kernel("groupby_min kernel not found".to_string())
                            })?;
                        // SAFETY: groupby_min(values, group_ids, num_rows, mins)
                        unsafe {
                            min_func.clone().launch_on_stream(
                                &cu_stream,
                                cfg,
                                (&values_view, &group_ids, num_rows, &*output),
                            )
                        }
                        .map_err(|e| {
                            XlogError::Kernel(format!("groupby_min (on_stream) failed: {}", e))
                        })?;
                    }
                }
                AggOp::Max => {
                    self.memset_zeros_u8_on_stream(output, &cu_stream)?;
                    let value_ty = sorted
                        .schema()
                        .column_type(*value_col)
                        .ok_or_else(|| XlogError::Kernel("Value column has no type".to_string()))?;
                    if value_ty == ScalarType::U64 {
                        // U64 value-column max path (output U64,
                        // identity 0).
                        let values_view = self.column_as_u64_view(values, row_cap_usize)?;
                        let max_func = device
                            .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MAX_U64)
                            .ok_or_else(|| {
                                XlogError::Kernel("groupby_max_u64 kernel not found".to_string())
                            })?;
                        // SAFETY: groupby_max_u64(values, group_ids, num_rows, maxs)
                        unsafe {
                            max_func.clone().launch_on_stream(
                                &cu_stream,
                                cfg,
                                (&values_view, &group_ids, num_rows, &*output),
                            )
                        }
                        .map_err(|e| {
                            XlogError::Kernel(format!("groupby_max_u64 (on_stream) failed: {}", e))
                        })?;
                    } else {
                        let values_view = self.column_as_u32_view(values, row_cap_usize)?;
                        let max_func = device
                            .get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MAX)
                            .ok_or_else(|| {
                                XlogError::Kernel("groupby_max kernel not found".to_string())
                            })?;
                        // SAFETY: groupby_max(values, group_ids, num_rows, maxs)
                        unsafe {
                            max_func.clone().launch_on_stream(
                                &cu_stream,
                                cfg,
                                (&values_view, &group_ids, num_rows, &*output),
                            )
                        }
                        .map_err(|e| {
                            XlogError::Kernel(format!("groupby_max (on_stream) failed: {}", e))
                        })?;
                    }
                }
                AggOp::LogSumExp => unreachable!("rejected above"),
            }
        }

        // Step 9: gather packed key rows by group_first_idx.
        let gather_fn = device
            .get_func(PACK_MODULE, pack_kernels::GATHER_PACKED_ROWS_COUNTED)
            .ok_or_else(|| {
                XlogError::Kernel("gather_packed_rows_counted kernel not found".to_string())
            })?;
        let gather_config = LaunchConfig::for_num_elems(row_cap_u32);
        // SAFETY: gather_packed_rows_counted(src_packed, row_size, indices, num_rows, capacity_rows, dst_packed)
        unsafe {
            gather_fn.clone().launch_on_stream(
                &cu_stream,
                gather_config,
                (
                    &packed.packed_keys,
                    packed.key_bytes,
                    &group_first_idx,
                    &d_num_groups,
                    row_cap_u32,
                    &mut group_packed,
                ),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "gather_packed_rows_counted (on_stream) failed: {}",
                e
            ))
        })?;

        // Step 10: unpack each key column from the gathered packed rows.
        let unpack_fn = device
            .get_func(PACK_MODULE, pack_kernels::UNPACK_COLUMN_COUNTED)
            .ok_or_else(|| {
                XlogError::Kernel("unpack_column_counted kernel not found".to_string())
            })?;
        let unpack_config = LaunchConfig::for_num_elems(row_cap_u32);
        for idx in 0..key_cols.len() {
            let col_size = col_sizes[idx];
            let col_offset = col_offsets[idx];
            // SAFETY: unpack_column_counted(packed, row_size, col_offset, col_size,
            // num_rows, capacity_rows, col_output)
            unsafe {
                unpack_fn.clone().launch_on_stream(
                    &cu_stream,
                    unpack_config,
                    (
                        &group_packed,
                        packed.key_bytes,
                        col_offset,
                        col_size,
                        &d_num_groups,
                        row_cap_u32,
                        &mut key_unpacked[idx],
                    ),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("unpack_column_counted (on_stream) failed: {}", e))
            })?;
        }

        // Record fresh writes via post-preflight escape hatch.
        rec.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("groupby_multi_agg_recorded: commit failed: {}", e))
        })?;

        // Step 11: build the result CudaBuffer (keys then aggs).
        let mut result_columns: Vec<CudaColumn> = Vec::with_capacity(key_cols.len() + aggs.len());
        for k in key_unpacked {
            result_columns.push(k.into());
        }
        for o in agg_outputs {
            result_columns.push(o.into());
        }
        let result_schema = self.groupby_multi_agg_result_schema(buffer.schema(), key_cols, aggs);
        Ok(CudaBuffer::from_columns(
            result_columns,
            row_cap_u64,
            d_num_groups,
            result_schema,
        ))
    }

    /// Convenience single-aggregation entry, mirrors
    /// [`Self::groupby_agg`]. Forwards to
    /// [`Self::groupby_multi_agg_recorded`].
    pub fn groupby_agg_recorded(
        &self,
        input: &CudaBuffer,
        key_cols: &[usize],
        agg: AggOp,
        value_col: usize,
        launch_stream: crate::device_runtime::StreamId,
    ) -> Result<CudaBuffer> {
        self.groupby_multi_agg_recorded(input, key_cols, &[(value_col, agg)], launch_stream)
    }
}
