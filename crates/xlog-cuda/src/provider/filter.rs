//! Filter and compare operations on GPU buffers.
//!
//! Generic versions of `filter`, `compare_columns`, plus the shared helpers
//! `compare_const_mask` and `compare_columns_mask` (moved from mod.rs).

use std::marker::PhantomData;

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, ScalarType, XlogError};

use super::{filter_kernels, scan_kernels, RawCudaView, FILTER_MODULE, SCAN_MODULE};
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

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
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

        let output_rows = self.dtoh_scalar_untracked(&d_out_count, 0)? as u64;
        if output_rows == 0 {
            return self.create_empty_buffer(input.schema.clone());
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

            let output_bytes = (output_rows as usize) * (elem_size as usize);
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

        Ok(CudaBuffer::from_columns(
            new_columns,
            output_rows,
            d_out_count,
            input.schema.clone(),
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
