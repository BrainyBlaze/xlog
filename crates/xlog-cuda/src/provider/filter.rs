//! Filter and compare operations on GPU buffers.
//!
//! Generic versions of `filter`, `compare_columns`, plus the shared helpers
//! `compare_const_mask` and `compare_columns_mask` (moved from mod.rs).

use std::marker::PhantomData;

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, ScalarType, XlogError};

use super::{scan_kernels, RawCudaView, FILTER_MODULE, SCAN_MODULE};
use crate::memory::{CudaColumn, TrackedCudaSlice};
use crate::type_seam::GpuScalar;
use crate::{CudaBuffer, CompareOp};

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
    pub(crate) fn compare_const_mask<T: DeviceRepr + Copy>(
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

        unsafe {
            func.clone()
                .launch(config, (col_data, value, num_rows, op as u8, &mut d_mask))
        }
        .map_err(|e| XlogError::Kernel(format!("filter compare failed: {}", e)))?;

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

        unsafe {
            func.clone().launch(
                config,
                (left_col, right_col, num_rows, op as u8, &mut d_mask),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("filter compare failed: {}", e)))?;

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
                col, col_type, T::allowed_scalar_types()
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
            .ok_or_else(|| {
                XlogError::Kernel(format!("{} kernel not found", scan_kernel_name))
            })?;

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
        let ptr = *cudarc::driver::DevicePtr::device_ptr(col);
        if T::BYTE_WIDTH > 1 && (ptr as usize) % T::BYTE_WIDTH != 0 {
            return Err(XlogError::Kernel(format!(
                "Column device pointer is not {}-byte aligned",
                T::BYTE_WIDTH,
            )));
        }
        Ok(RawCudaView {
            ptr,
            len: num_elements,
            _marker: PhantomData,
        })
    }
}
