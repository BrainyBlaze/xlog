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

        let num_rows = num_rows as u32;

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
                    let values_view = self.column_as_u32_view(values, num_rows as usize)?;
                    let output_bytes = row_cap_usize
                        .checked_mul(std::mem::size_of::<u64>())
                        .ok_or_else(|| XlogError::Kernel("Sum output size overflow".to_string()))?;
                    let mut output = self.memory.alloc::<u8>(output_bytes)?;
                    device.memset_zeros(&mut output).map_err(|e| {
                        XlogError::Kernel(format!("Failed to zero sum output: {}", e))
                    })?;

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
                    let output_bytes = row_cap_usize
                        .checked_mul(std::mem::size_of::<u32>())
                        .ok_or_else(|| XlogError::Kernel("Min output size overflow".to_string()))?;
                    let mut output = self.memory.alloc::<u8>(output_bytes)?;
                    let fill_fn = device
                        .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_U32)
                        .ok_or_else(|| {
                            XlogError::Kernel("arith_fill_const_u32 not found".to_string())
                        })?;
                    let fill_config = LaunchConfig::for_num_elems(row_cap_u32);
                    unsafe {
                        fill_fn
                            .clone()
                            .launch(fill_config, (u32::MAX, row_cap_u32, &mut output))
                    }
                    .map_err(|e| XlogError::Kernel(format!("Failed to init min output: {}", e)))?;

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
                    let output_bytes = row_cap_usize
                        .checked_mul(std::mem::size_of::<u32>())
                        .ok_or_else(|| XlogError::Kernel("Max output size overflow".to_string()))?;
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
}
