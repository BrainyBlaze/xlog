//! Filter and compare operations on GPU buffers.
//!
//! Generic versions of `filter`, `compare_columns`, plus the shared helpers
//! `compare_const_mask` and `compare_columns_mask` (moved from mod.rs).

use std::marker::PhantomData;

use crate::{DeviceRepr, DeviceSlice, KernelScalar, LaunchAsync, LaunchConfig};
use xlog_core::{Result, ScalarType, XlogError};

use super::{filter_kernels, scan_kernels, RawCudaView, FILTER_MODULE, SCAN_MODULE};
use crate::device_runtime::StreamId;
use crate::launch::LaunchRecorder;
use crate::memory::{CudaColumn, TrackedCudaSlice};
use crate::type_seam::GpuScalar;
use crate::{CompareOp, CudaBuffer};

impl super::CudaKernelProvider {
    // ------------------------------------------------------------------
    // Generic public API
    // ------------------------------------------------------------------

    /// Generic compare-columns: produce a device mask for `left <op> right`.
    ///
    /// Replaces: `compare_columns_u32`, `compare_columns_i32`, `compare_columns_i64`,
    /// `compare_columns_u64`, `compare_columns_f32`, `compare_columns_f64`,
    /// `compare_columns_u8`.
    pub fn compare_columns<T: GpuScalar>(
        &self,
        input: &CudaBuffer,
        left: usize,
        right: usize,
        op: CompareOp,
    ) -> Result<TrackedCudaSlice<u8>> {
        self.compare_columns_mask::<T>(
            input,
            left,
            right,
            op,
            T::allowed_scalar_types(),
            T::compare_col_kernel(),
        )
    }

    /// Generic filter: keep rows where `column[col] <op> value`.
    ///
    /// Dispatches between fused compare+scan+compact (u32, f64) and
    /// mask+compact (all other types).
    ///
    /// Replaces: `filter_u32`, `filter_f64`, `filter_i32`, `filter_u64`,
    /// `filter_f32`, `filter_bool`.
    pub fn filter<T: GpuScalar>(
        &self,
        input: &CudaBuffer,
        col: usize,
        value: T,
        op: CompareOp,
    ) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema.clone());
        }

        if T::filter_scan_phase1_kernel().is_some() {
            // Fused compare+scan+compact path (u32, f64).
            self.filter_fused_scan::<T>(input, col, value, op)
        } else {
            // Mask + compact path (i32, u64, f32, i64, u8, bool).
            let mask = self.compare_const_mask::<T>(
                input,
                col,
                value,
                op,
                T::allowed_scalar_types(),
                T::filter_compare_kernel(),
            )?;
            self.filter_by_device_mask(input, &mask)
        }
    }

    // ------------------------------------------------------------------
    // Private helpers (moved from mod.rs)
    // ------------------------------------------------------------------

    /// Generate a device mask by comparing a column against a constant value.
    ///
    /// Each output byte is 1 if the comparison holds, 0 otherwise.
    pub(crate) fn compare_const_mask<T: KernelScalar>(
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
            return self.memory.alloc::<u8>(0);
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
        let num_blocks = num_rows.div_ceil(block_size);
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

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone()
                .launch(config, (col_data, value, num_rows, op as u8, &mut d_mask))
        }
        .map_err(|e| XlogError::Kernel(format!("filter compare failed: {}", e)))?;

        Ok(d_mask)
    }

    /// Strict-recorder variant of [`Self::compare_const_mask`].
    ///
    /// Runs the filter compare kernel on the caller-supplied
    /// `launch_stream` and threads the column read through the
    /// runtime via [`LaunchRecorder`]. This is the second
    /// migrated launch path (after `memset_recorded`) and the
    /// first kernel-driven one — it is intentionally a sibling
    /// of the legacy [`Self::compare_const_mask`] rather than a
    /// replacement. Existing callers stay on the legacy path
    /// until the broader filter migration lands.
    ///
    /// # Strict-mode contract
    /// * Requires the provider's manager to be built via
    ///   [`crate::GpuMemoryManager::with_runtime`]; otherwise
    ///   returns `XlogError::Kernel` before any allocation.
    /// * `input.column(col)` is recorded as a read; external
    ///   (`CudaColumn::Dlpack` / `CudaColumn::ArrowDevice`)
    ///   columns are rejected at preflight, before the kernel
    ///   is enqueued.
    /// * `d_mask` is freshly allocated through the same
    ///   runtime-backed manager. By construction its
    ///   `runtime_block()` is `Some`, so its write recording
    ///   cannot strict-reject. The write is therefore noted
    ///   AFTER the kernel is enqueued — this sidesteps the
    ///   borrow conflict between `&mut d_mask` (cudarc kernel
    ///   param) and `&d_mask` (recorder). A future migration
    ///   that may write to a buffer of unknown provenance must
    ///   instead capture identity pre-launch (e.g. via a raw
    ///   view) so strict rejection happens at preflight.
    ///
    /// # Errors
    ///   * `XlogError::Kernel` if the manager has no runtime,
    ///     or if `launch_stream` does not resolve.
    ///   * `XlogError::Kernel` from preflight (external column,
    ///     unsupported active resource).
    ///   * `XlogError::Kernel` from the underlying CUDA launch.
    ///   * `XlogError::Kernel` from commit on transient
    ///     `record_block_use` failure.
    pub fn compare_const_mask_recorded<T: GpuScalar>(
        &self,
        input: &CudaBuffer,
        col: usize,
        value: T,
        op: CompareOp,
        launch_stream: StreamId,
    ) -> Result<TrackedCudaSlice<u8>> {
        let allowed_types = T::allowed_scalar_types();
        let kernel = T::filter_compare_kernel();
        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "compare_const_mask_recorded requires a runtime-backed GpuMemoryManager \
                 (constructed via with_runtime)"
                    .to_string(),
            )
        })?;
        let pool = runtime.stream_pool();
        let cu_stream = pool.resolve(launch_stream).ok_or_else(|| {
            XlogError::Kernel(format!(
                "compare_const_mask_recorded: launch_stream StreamId({}) does not resolve",
                launch_stream.0
            ))
        })?;

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
            return self.memory.alloc::<u8>(0);
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
        let num_blocks = num_rows.div_ceil(block_size);
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

        // Strict recorder + PREFLIGHT before the kernel queues.
        // External columns (DLPack / Arrow) and unsupported
        // active resources are caught here without any CUDA
        // work in flight.
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read_column(col_data);
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "compare_const_mask_recorded: launch recorder preflight failed: {}",
                e
            ))
        })?;

        // SAFETY: PTX kernel signature matches the params tuple;
        // col_data is validated above and lives through the
        // launch (held by `input`); d_mask was allocated by the
        // same runtime-backed manager and matches `num_rows`.
        // launch_on_stream queues on `cu_stream` and returns
        // immediately.
        unsafe {
            func.clone().launch_on_stream(
                &cu_stream,
                config,
                (col_data, value, num_rows, op as u8, &mut d_mask),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!("compare_const_mask_recorded launch failed: {}", e))
        })?;

        // Record the write AFTER the launch enqueues, using the
        // explicit escape hatch. d_mask is the freshly-allocated
        // runtime-backed output of THIS call; the kernel-param
        // borrow rules force this ordering. See the
        // "Strict-mode contract" on this method.
        rec.write_post_preflight_fresh(&d_mask);
        rec.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "compare_const_mask_recorded: launch recorder commit failed: {}",
                e
            ))
        })?;

        Ok(d_mask)
    }

    /// Generate a device mask by comparing two columns element-wise.
    pub(crate) fn compare_columns_mask<T: DeviceRepr>(
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

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                config,
                (left_col, right_col, num_rows, op as u8, &mut d_mask),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("filter compare failed: {}", e)))?;

        Ok(d_mask)
    }

    /// Strict-recorder variant of [`Self::compare_columns_mask`].
    ///
    /// Runs the column-column compare kernel on the
    /// caller-supplied `launch_stream` and threads BOTH column
    /// reads through the runtime via [`LaunchRecorder`]. Sibling
    /// of the legacy [`Self::compare_columns_mask`]; existing
    /// callers stay on the legacy path.
    ///
    /// # Strict-mode contract
    /// * Requires the provider's manager to be built via
    ///   [`crate::GpuMemoryManager::with_runtime`]; otherwise
    ///   returns `XlogError::Kernel` before any allocation.
    /// * `input.column(left)` and `input.column(right)` are both
    ///   recorded as reads BEFORE preflight. External (DLPack /
    ///   Arrow) columns on either side are rejected at preflight,
    ///   before the kernel is enqueued.
    /// * `d_mask` is freshly allocated by the same runtime-backed
    ///   manager; its write is recorded via the
    ///   `write_post_preflight_fresh` escape hatch (the
    ///   kernel-param borrow rules force this ordering — see
    ///   `compare_const_mask_recorded` for the same pattern).
    ///
    /// # Errors
    ///   * `XlogError::Kernel` if the manager has no runtime,
    ///     or if `launch_stream` does not resolve.
    ///   * `XlogError::Kernel` from preflight (external column
    ///     on either side, unsupported active resource).
    ///   * `XlogError::Kernel` from the underlying CUDA launch.
    ///   * `XlogError::Kernel` from commit on transient
    ///     `record_block_use` failure.
    pub fn compare_columns_mask_recorded<T: GpuScalar>(
        &self,
        input: &CudaBuffer,
        left: usize,
        right: usize,
        op: CompareOp,
        launch_stream: StreamId,
    ) -> Result<TrackedCudaSlice<u8>> {
        let allowed_types = T::allowed_scalar_types();
        let kernel = T::compare_col_kernel();

        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "compare_columns_mask_recorded requires a runtime-backed GpuMemoryManager \
                 (constructed via with_runtime)"
                    .to_string(),
            )
        })?;
        let pool = runtime.stream_pool();
        let cu_stream = pool.resolve(launch_stream).ok_or_else(|| {
            XlogError::Kernel(format!(
                "compare_columns_mask_recorded: launch_stream StreamId({}) does not resolve",
                launch_stream.0
            ))
        })?;

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
            return self.memory.alloc::<u8>(0);
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
        let num_blocks = num_rows.div_ceil(block_size);
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

        // Record BOTH column reads BEFORE preflight. Strict mode
        // catches external columns on either side here, before
        // any CUDA work is queued.
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read_column(left_col);
        rec.read_column(right_col);
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "compare_columns_mask_recorded: launch recorder preflight failed: {}",
                e
            ))
        })?;

        // SAFETY: PTX kernel signature matches the params tuple;
        // both columns are validated above and live through the
        // launch (held by `input`); d_mask was allocated by the
        // same runtime-backed manager and matches `num_rows`.
        // launch_on_stream queues on `cu_stream` and returns
        // immediately.
        unsafe {
            func.clone().launch_on_stream(
                &cu_stream,
                config,
                (left_col, right_col, num_rows, op as u8, &mut d_mask),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "compare_columns_mask_recorded launch failed: {}",
                e
            ))
        })?;

        // Record d_mask write AFTER the launch enqueues, via the
        // explicit escape hatch — d_mask is the freshly-allocated
        // runtime-backed output of THIS call.
        rec.write_post_preflight_fresh(&d_mask);
        rec.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "compare_columns_mask_recorded: launch recorder commit failed: {}",
                e
            ))
        })?;

        Ok(d_mask)
    }

    // ------------------------------------------------------------------
    // Fused compare+scan+compact path (generic over T)
    // ------------------------------------------------------------------

    /// Fused compare+scan+compact filter. Used for types with a dedicated
    /// `filter_scan_phase1` kernel (u32 and f64).
    fn filter_fused_scan<T: GpuScalar>(
        &self,
        input: &CudaBuffer,
        col: usize,
        value: T,
        op: CompareOp,
    ) -> Result<CudaBuffer> {
        if input.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Kernel(format!(
                "filter supports at most {} rows, got {}",
                u32::MAX,
                input.num_rows()
            )));
        }

        let n = input.num_rows() as usize;
        let num_rows = input.num_rows() as u32;
        let device = self.device.inner();

        // Validate column index
        if col >= input.arity() {
            return Err(XlogError::Kernel(format!(
                "Column index {} out of bounds (arity {})",
                col,
                input.arity()
            )));
        }

        // Validate column type
        let col_type = input
            .schema()
            .column_type(col)
            .ok_or_else(|| XlogError::Kernel("Missing column type".into()))?;
        if !T::allowed_scalar_types().contains(&col_type) {
            return Err(XlogError::Kernel(format!(
                "Column {} is {:?} (expected one of {:?})",
                col,
                col_type,
                T::allowed_scalar_types()
            )));
        }

        // Get the filter column as a typed view
        let col_data = input
            .column(col)
            .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col)))?;
        let col_view = Self::column_as_typed_view::<T>(col_data, n)?;

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

        let scan_kernel_name = T::filter_scan_phase1_kernel()
            .expect("filter_fused_scan called without scan phase1 kernel");
        let filter_scan_fn = device
            .get_func(FILTER_MODULE, scan_kernel_name)
            .ok_or_else(|| XlogError::Kernel(format!("{} kernel not found", scan_kernel_name)))?;

        // SAFETY: filter_compare_*_scan_phase1(column, constant, num_rows, num_rows_device, op, mask, prefix_sum, block_sums)
        unsafe {
            filter_scan_fn.clone().launch(
                config,
                (
                    &col_view,
                    value,
                    num_rows,
                    input.num_rows_device(),
                    op as u8,
                    &d_mask,
                    &d_prefix_sum,
                    &d_block_sums,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("{} failed: {}", scan_kernel_name, e)))?;

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

        let d_out_count = self.capture_compact_count(&d_prefix_sum, &d_mask, num_rows)?;
        self.compact_buffer_by_device_mask_device_count(input, &d_mask, &d_prefix_sum, d_out_count)
    }

    // ------------------------------------------------------------------
    // Generic column view helper
    // ------------------------------------------------------------------

    /// Reinterpret a `CudaColumn` as a typed `RawCudaView<T>` for kernel access.
    ///
    /// This is the generic equivalent of `column_as_u32_view`, `column_as_f64_view`, etc.
    fn column_as_typed_view<'a, T: GpuScalar>(
        col: &'a CudaColumn,
        num_elements: usize,
    ) -> Result<RawCudaView<'a, T>> {
        let required_bytes = num_elements * T::BYTE_WIDTH;
        if col.num_bytes() < required_bytes {
            return Err(XlogError::Kernel(format!(
                "Column has {} bytes but {} required for {} elements of size {}",
                col.num_bytes(),
                required_bytes,
                num_elements,
                T::BYTE_WIDTH,
            )));
        }
        let ptr = *col.device_ptr();
        if T::BYTE_WIDTH > 1 && (ptr as usize) % T::BYTE_WIDTH != 0 {
            return Err(XlogError::Kernel(format!(
                "Column device pointer is not {}-byte aligned",
                T::BYTE_WIDTH,
            )));
        }
        Ok(RawCudaView {
            ptr,
            len: num_elements,
            stream: col.stream().clone(),
            source_block: col.runtime_block(),
            _marker: PhantomData,
        })
    }

    // ------------------------------------------------------------------
    // Host-mask prefix sum + filter/compact infrastructure
    // (moved from mod.rs)
    // ------------------------------------------------------------------

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
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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
            // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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

    /// Filter buffer by pre-computed mask.
    ///
    /// # Arguments
    /// * `input` - The input buffer to filter
    /// * `mask` - Mask slice where non-zero means keep the row
    ///
    /// # Errors
    /// Returns error if mask length doesn't match buffer rows.
    pub fn filter_by_mask(&self, input: &CudaBuffer, mask: &[u8]) -> Result<CudaBuffer> {
        if input.num_rows() == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }

        let n = input.num_rows() as usize;
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

    /// Compact a buffer using a device-resident mask.
    ///
    /// Computes prefix sum and output count fully on-device.
    /// Strict-recorder variant of
    /// [`Self::compact_buffer_by_device_mask_counted`] — the
    /// first migrated COMPACT path.
    ///
    /// The compact pipeline is a multi-kernel chain:
    /// `mask_clamp_rows` → `multiblock_scan_phase1` →
    /// `multiblock_scan_u32_inplace_on_stream` (recursive,
    /// only when `num_blocks > 1`) → `multiblock_scan_phase3` →
    /// `capture_compact_count` → host scalar read of
    /// `d_out_count` → per-column `compact_bytes_by_mask`.
    /// **Every kernel runs on the same explicit `launch_stream`
    /// via `launch_on_stream`**, and the host scalar read at
    /// the chain's middle is explicitly ordered by
    /// `cu_stream.synchronize()` — non-blocking streams do
    /// NOT get default-stream implicit ordering.
    ///
    /// # Strict-mode contract
    /// * Requires the provider's manager to be built via
    ///   [`crate::GpuMemoryManager::with_runtime`]; otherwise
    ///   returns `XlogError::Kernel` before any allocation.
    /// * `d_mask` is recorded as a read.
    /// * `input.num_rows_device()` is recorded as a read.
    /// * Each `input.column(i)` is recorded as a read; external
    ///   columns on any side are rejected at preflight, before
    ///   any CUDA work is enqueued.
    /// * Every fresh runtime-backed allocation that this
    ///   function makes (`d_mask_clamped`, `d_prefix_sum`,
    ///   `d_block_sums`, `d_out_count`, each `dst_col`) is
    ///   recorded via `write_post_preflight_fresh` AFTER the
    ///   kernel chain enqueues. Locals that drop at end-of-scope
    ///   (`d_mask_clamped`, `d_prefix_sum`, `d_block_sums`)
    ///   stay safe because the runtime's deallocate queues
    ///   `cuStreamWaitEvent(alloc_stream, recorded_event)`
    ///   BEFORE `cuMemFreeAsync`, gating the free on the
    ///   launch_stream chain.
    /// * Intermediate `block_sums` allocations created by the
    ///   recursive scan helper are recorded directly inside the
    ///   helper (they don't outlive the helper call).
    ///
    /// # Errors
    ///   * `XlogError::Kernel` if the manager has no runtime,
    ///     or if `launch_stream` does not resolve.
    ///   * `XlogError::Kernel` from preflight (external column
    ///     on any side, unsupported active resource).
    ///   * `XlogError::Kernel` from any underlying CUDA launch
    ///     or from the launch_stream synchronize before the
    ///     host scalar read.
    ///   * `XlogError::Kernel` from commit on transient
    ///     `record_block_use` failure.
    pub fn compact_buffer_by_device_mask_counted_recorded(
        &self,
        input: &CudaBuffer,
        d_mask: &TrackedCudaSlice<u8>,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let runtime = self.memory.runtime().ok_or_else(|| {
            XlogError::Kernel(
                "compact_buffer_by_device_mask_counted_recorded requires a \
                 runtime-backed GpuMemoryManager (constructed via with_runtime)"
                    .to_string(),
            )
        })?;
        let pool = runtime.stream_pool();
        let cu_stream = pool.resolve(launch_stream).ok_or_else(|| {
            XlogError::Kernel(format!(
                "compact_buffer_by_device_mask_counted_recorded: launch_stream \
                 StreamId({}) does not resolve",
                launch_stream.0
            ))
        })?;

        let n = input.num_rows() as u32;
        if n == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }
        if (n as usize) > d_mask.len() {
            return Err(XlogError::Kernel(format!(
                "compact_buffer_by_device_mask_counted_recorded: mask len {} < rows {}",
                d_mask.len(),
                n
            )));
        }

        let device = self.device.inner();
        let block_size = 256u32;
        let num_blocks = n.div_ceil(block_size);
        let row_cap = u64::from(n);

        // Allocate ALL fresh runtime-backed buffers up front,
        // BEFORE the recorder is constructed. This is required
        // by the borrow checker: `rec` borrows each of these
        // (via `write_post_preflight_fresh`) for its lifetime
        // `'b`, which must outlive all its borrows. With Rust's
        // reverse-of-declaration drop order, anything declared
        // after `rec` cannot be borrowed by it. Output column
        // sizes are known up front from `row_cap = n`, so this
        // is sound — the host scalar read of `d_out_count` only
        // tells us `output_rows`, which we use as metadata, not
        // for sizing.
        let mut d_mask_clamped = self.memory.alloc::<u8>(n as usize)?;
        let d_prefix_sum = self.memory.alloc::<u32>(n as usize)?;
        let mut d_block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;
        let mut d_out_count = self.memory.alloc::<u32>(1)?;

        let mut dst_cols: Vec<TrackedCudaSlice<u8>> = Vec::with_capacity(input.columns.len());
        for col_idx in 0..input.columns.len() {
            let elem_size = input
                .schema
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let output_bytes = (row_cap as usize) * elem_size;
            dst_cols.push(self.memory.alloc::<u8>(output_bytes)?);
        }

        // Build recorder, record reads BEFORE preflight.
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(d_mask);
        rec.read(input.num_rows_device());
        for col_idx in 0..input.columns.len() {
            let src_col = input
                .column(col_idx)
                .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;
            rec.read_column(src_col);
        }
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "compact_buffer_by_device_mask_counted_recorded: launch recorder \
                 preflight failed: {}",
                e
            ))
        })?;

        // Step 1: mask_clamp_rows on launch_stream.
        let clamp_fn = device
            .get_func(FILTER_MODULE, filter_kernels::MASK_CLAMP_ROWS)
            .ok_or_else(|| XlogError::Kernel("mask_clamp_rows kernel not found".to_string()))?;
        // SAFETY: mask_clamp_rows(in_mask, num_rows_device, row_cap, out_mask)
        unsafe {
            clamp_fn.clone().launch_on_stream(
                &cu_stream,
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (d_mask, input.num_rows_device(), n, &mut d_mask_clamped),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("mask_clamp_rows (on_stream) failed: {}", e)))?;

        // Step 2: multiblock_scan_phase1 on launch_stream.
        let phase1_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase1 kernel".to_string())
            })?;
        // SAFETY: multiblock_scan_phase1(const u8 mask, u32 prefix_sum, u32 block_sums, u32 n)
        unsafe {
            phase1_fn.clone().launch_on_stream(
                &cu_stream,
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&d_mask_clamped, &d_prefix_sum, &d_block_sums, n),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!("multiblock_scan_phase1 (on_stream) failed: {}", e))
        })?;

        // Step 3: scan inplace on block_sums + phase3 propagate
        // (only when there is more than one block).
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
                    (&d_prefix_sum, &d_block_sums, n),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("multiblock_scan_phase3 (on_stream) failed: {}", e))
            })?;
        }

        // Step 4: capture_compact_count on launch_stream.
        // (`d_out_count` was pre-allocated up front; see header.)
        let capture_fn = device
            .get_func(FILTER_MODULE, filter_kernels::CAPTURE_COMPACT_COUNT)
            .ok_or_else(|| {
                XlogError::Kernel("capture_compact_count kernel not found".to_string())
            })?;
        // SAFETY: capture_compact_count(prefix_sum, mask, n, out_count)
        unsafe {
            capture_fn.clone().launch_on_stream(
                &cu_stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&d_prefix_sum, d_mask, n, &mut d_out_count),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!("capture_compact_count (on_stream) failed: {}", e))
        })?;

        // Explicit ordering for the host scalar read of
        // `d_out_count`. `dtoh_scalar_untracked` routes its
        // copy through the device's default cudarc stream,
        // which does NOT get implicit synchronization with the
        // non-blocking `launch_stream`. Without this barrier
        // we would race the still-pending capture kernel.
        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "compact_buffer_by_device_mask_counted_recorded: launch_stream \
                 synchronize before host scalar read failed: {}",
                e
            ))
        })?;

        let output_rows = self.dtoh_scalar_untracked(&d_out_count, 0)? as u64;

        // Step 5: per-column compact_bytes_by_mask on
        // launch_stream. Only run when output_rows > 0; an
        // empty mask still allocates row_cap-sized columns
        // (matching legacy) but skips the kernel.
        if output_rows > 0 {
            let compact_fn = device
                .get_func(FILTER_MODULE, filter_kernels::COMPACT_BYTES_BY_MASK)
                .ok_or_else(|| {
                    XlogError::Kernel("compact_bytes_by_mask kernel not found".to_string())
                })?;
            let grid_size = n.div_ceil(block_size);
            let cfg = LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            for (col_idx, dst_col) in dst_cols.iter().enumerate() {
                let src_col = input
                    .column(col_idx)
                    .ok_or_else(|| XlogError::Kernel(format!("Column {} not found", col_idx)))?;
                let elem_size = input
                    .schema
                    .column_type(col_idx)
                    .map(|t| t.size_bytes())
                    .unwrap_or(4) as u32;
                // SAFETY: compact_bytes_by_mask(input, mask, prefix_sum, n, elem_size, output)
                unsafe {
                    compact_fn.clone().launch_on_stream(
                        &cu_stream,
                        cfg,
                        (src_col, d_mask, &d_prefix_sum, n, elem_size, dst_col),
                    )
                }
                .map_err(|e| {
                    XlogError::Kernel(format!("compact_bytes_by_mask (on_stream) failed: {}", e))
                })?;
            }
        }

        // Record fresh writes via the post-preflight escape
        // hatch. ALL fresh runtime-backed allocations made by
        // this function are recorded so that drops at
        // end-of-scope (or on the returned buffer's drop) are
        // correctly serialized with the launch_stream chain.
        rec.write_post_preflight_fresh(&d_mask_clamped);
        rec.write_post_preflight_fresh(&d_prefix_sum);
        rec.write_post_preflight_fresh(&d_block_sums);
        rec.write_post_preflight_fresh(&d_out_count);
        for dst_col in &dst_cols {
            rec.write_post_preflight_fresh(dst_col);
        }
        rec.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "compact_buffer_by_device_mask_counted_recorded: launch recorder \
                 commit failed: {}",
                e
            ))
        })?;

        let new_columns: Vec<CudaColumn> = dst_cols.into_iter().map(|s| s.into()).collect();
        Ok(CudaBuffer::from_columns_with_host_count(
            new_columns,
            row_cap,
            d_out_count,
            input.schema.clone(),
            output_rows as u32,
        ))
    }

    pub fn compact_buffer_by_device_mask_counted(
        &self,
        input: &CudaBuffer,
        d_mask: &TrackedCudaSlice<u8>,
    ) -> Result<CudaBuffer> {
        let n = input.num_rows() as u32;
        if n == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }
        if n as usize > d_mask.len() {
            return Err(XlogError::Kernel(format!(
                "compact_buffer_by_device_mask_counted: mask len {} < rows {}",
                d_mask.len(),
                n
            )));
        }

        let device = self.device.inner();
        let block_size = 256u32;
        let num_blocks = (n + block_size - 1) / block_size;

        let mut d_mask_clamped = self.memory.alloc::<u8>(n as usize)?;
        let clamp_fn = device
            .get_func(FILTER_MODULE, filter_kernels::MASK_CLAMP_ROWS)
            .ok_or_else(|| XlogError::Kernel("mask_clamp_rows kernel not found".to_string()))?;

        // SAFETY: mask_clamp_rows(in_mask, num_rows_device, row_cap, out_mask)
        unsafe {
            clamp_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (d_mask, input.num_rows_device(), n, &mut d_mask_clamped),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("mask_clamp_rows failed: {}", e)))?;

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
                (&d_mask_clamped, &d_prefix_sum, &d_block_sums, n),
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

            // SAFETY: multiblock_scan_phase3(prefix_sum, block_offsets, n)
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

        let d_out_count = self.capture_compact_count(&d_prefix_sum, d_mask, n)?;

        self.compact_buffer_by_device_mask_device_count(input, d_mask, &d_prefix_sum, d_out_count)
    }

    pub(crate) fn capture_compact_count(
        &self,
        d_prefix_sum: &cudarc::driver::CudaSlice<u32>,
        d_mask: &cudarc::driver::CudaSlice<u8>,
        n: u32,
    ) -> Result<TrackedCudaSlice<u32>> {
        let mut d_out_count = self.memory.alloc::<u32>(1)?;
        let device = self.device.inner();
        let capture_fn = device
            .get_func(FILTER_MODULE, filter_kernels::CAPTURE_COMPACT_COUNT)
            .ok_or_else(|| {
                XlogError::Kernel("capture_compact_count kernel not found".to_string())
            })?;
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            capture_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (d_prefix_sum, d_mask, n, &mut d_out_count),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("capture_compact_count failed: {}", e)))?;
        Ok(d_out_count)
    }

    pub(crate) fn compact_buffer_by_device_mask_device_count(
        &self,
        input: &CudaBuffer,
        d_mask: &cudarc::driver::CudaSlice<u8>,
        d_prefix_sum: &cudarc::driver::CudaSlice<u32>,
        d_out_count: TrackedCudaSlice<u32>,
    ) -> Result<CudaBuffer> {
        let mask_len = u32::try_from(d_mask.len()).map_err(|_| {
            XlogError::Kernel(format!(
                "compact_buffer_by_device_mask_device_count: mask len {} exceeds u32::MAX",
                d_mask.len()
            ))
        })?;
        let prefix_len = u32::try_from(d_prefix_sum.len()).map_err(|_| {
            XlogError::Kernel(format!(
                "compact_buffer_by_device_mask_device_count: prefix sum len {} exceeds u32::MAX",
                d_prefix_sum.len()
            ))
        })?;
        if prefix_len < mask_len {
            return Err(XlogError::Kernel(format!(
                "compact_buffer_by_device_mask_device_count: prefix sum len {} < mask len {}",
                prefix_len, mask_len
            )));
        }
        if mask_len as u64 > input.num_rows() {
            return Err(XlogError::Kernel(format!(
                "compact_buffer_by_device_mask_device_count: mask len {} > row cap {}",
                mask_len,
                input.num_rows()
            )));
        }
        if mask_len == 0 {
            return self.create_empty_buffer(input.schema.clone());
        }
        let n = mask_len;
        let device = self.device.inner();

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

        // The compact kernel only writes `output_rows` elements, but downstream
        // metadata must still remember the logical span covered by the mask.
        // This keeps host-visible `row_cap` aligned with the masked prefix while
        // the device-side row count tracks the actual compacted cardinality.
        let output_rows = self.dtoh_scalar_untracked(&d_out_count, 0)? as u64;
        let row_cap = u64::from(n);

        if output_rows == 0 {
            let mut new_columns = Vec::with_capacity(input.columns.len());
            for col_idx in 0..input.columns.len() {
                let elem_size = input
                    .schema
                    .column_type(col_idx)
                    .map(|t| t.size_bytes())
                    .unwrap_or(4);
                let output_bytes = (row_cap as usize) * elem_size;
                new_columns.push(self.memory.alloc::<u8>(output_bytes)?.into());
            }
            let d_zero_rows = self.upload_device_row_count(0)?;
            return Ok(CudaBuffer::from_columns_with_host_count(
                new_columns,
                row_cap,
                d_zero_rows,
                input.schema.clone(),
                0,
            ));
        }

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

            let output_bytes = (row_cap as usize) * (elem_size as usize);
            let dst_col = self.memory.alloc::<u8>(output_bytes)?;

            // SAFETY: Kernel signature matches:
            // compact_bytes_by_mask(input, mask, prefix_sum, num_rows, elem_size, output)
            // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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

        Ok(CudaBuffer::from_columns_with_host_count(
            new_columns,
            row_cap,
            d_out_count,
            input.schema.clone(),
            output_rows as u32,
        ))
    }

    fn compact_buffer_by_device_mask(
        &self,
        input: &CudaBuffer,
        d_mask: &cudarc::driver::CudaSlice<u8>,
        d_prefix_sum: &cudarc::driver::CudaSlice<u32>,
        output_count: u64,
    ) -> Result<CudaBuffer> {
        let n = input.num_rows() as u32;
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
            // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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

        self.buffer_from_columns(new_columns, output_count, input.schema.clone())
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

        let mut d_mask_clamped = self.memory.alloc::<u8>(n as usize)?;
        let clamp_fn = device
            .get_func(FILTER_MODULE, filter_kernels::MASK_CLAMP_ROWS)
            .ok_or_else(|| XlogError::Kernel("mask_clamp_rows kernel not found".to_string()))?;

        // SAFETY: mask_clamp_rows(in_mask, num_rows_device, row_cap, out_mask)
        unsafe {
            clamp_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (d_mask, input.num_rows_device(), n, &mut d_mask_clamped),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("mask_clamp_rows failed: {}", e)))?;

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
                (&d_mask_clamped, &d_prefix_sum, &d_block_sums, n),
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

        let d_out_count = self.capture_compact_count(&d_prefix_sum, &d_mask_clamped, n)?;
        self.compact_buffer_by_device_mask_device_count(
            input,
            &d_mask_clamped,
            &d_prefix_sum,
            d_out_count,
        )
    }
}
