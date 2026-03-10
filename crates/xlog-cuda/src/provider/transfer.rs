//! Host <-> device transfer operations.
//!
//! Generic versions of `download_column` and `create_buffer_from_slice`,
//! replacing the 15 type-specialized functions with 3 generics.

use std::sync::atomic::Ordering;
use xlog_core::{Result, Schema, XlogError};

use crate::type_seam::GpuScalar;
use crate::CudaBuffer;

impl super::CudaKernelProvider {
    /// Download a single column from GPU to host as `Vec<T>`.
    ///
    /// Replaces: `download_column_u32`, `download_column_u64`,
    /// `download_column_i32`, `download_column_i64`, `download_column_f32`,
    /// `download_column_f64`, `download_column_u8`, `download_column_bool`.
    ///
    /// Increments the D2H transfer counter (gate-tracked).
    pub fn download_column<T: GpuScalar>(
        &self,
        buffer: &CudaBuffer,
        col_idx: usize,
    ) -> Result<Vec<T>> {
        self.d2h_transfer_count.fetch_add(1, Ordering::Relaxed);
        self.download_column_inner::<T>(buffer, col_idx)
    }

    /// Download a column WITHOUT incrementing the D2H transfer counter.
    /// Records in `transfer_tracker` for profiling stats but not for the D2H gate.
    ///
    /// Replaces: `download_f64_untracked` (now generic over `T`).
    pub fn download_column_untracked<T: GpuScalar>(
        &self,
        buffer: &CudaBuffer,
        col_idx: usize,
    ) -> Result<Vec<T>> {
        let col = buffer
            .column(col_idx)
            .ok_or_else(|| {
                XlogError::kernel_ctx(
                    "download_column_untracked",
                    "column not found",
                    &col_idx,
                )
            })?;

        let num_rows = self.device_row_count(buffer)?;
        if num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = num_rows.checked_mul(T::BYTE_WIDTH).ok_or_else(|| {
            XlogError::kernel_ctx(
                "download_column_untracked",
                "byte size overflow",
                &num_rows,
            )
        })?;
        let col_view = self.column_bytes_view(col, num_bytes)?;
        let mut bytes = vec![0u8; num_bytes];
        self.dtoh_sync_copy_into_tracked(&col_view, &mut bytes)?;

        Ok(bytes
            .chunks_exact(T::BYTE_WIDTH)
            .map(|c| T::from_le_bytes(c))
            .collect())
    }

    /// Shared implementation for tracked column downloads.
    ///
    /// Uses `device.inner().dtoh_sync_copy_into()` directly (no stats recording),
    /// which matches the existing `download_column_u32` pattern. The caller
    /// (`download_column`) is responsible for incrementing `d2h_transfer_count`.
    fn download_column_inner<T: GpuScalar>(
        &self,
        buffer: &CudaBuffer,
        col_idx: usize,
    ) -> Result<Vec<T>> {
        let col = buffer
            .column(col_idx)
            .ok_or_else(|| {
                XlogError::kernel_ctx("download_column", "column not found", &col_idx)
            })?;

        let num_rows = self.device_row_count(buffer)?;
        if num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = num_rows.checked_mul(T::BYTE_WIDTH).ok_or_else(|| {
            XlogError::kernel_ctx("download_column", "byte size overflow", &num_rows)
        })?;
        let col_view = self.column_bytes_view(col, num_bytes)?;
        let mut bytes = vec![0u8; num_bytes];
        self.device
            .inner()
            .dtoh_sync_copy_into(&col_view, &mut bytes)
            .map_err(|e| {
                XlogError::kernel_ctx("download_column", "dtoh copy failed", &e)
            })?;

        Ok(bytes
            .chunks_exact(T::BYTE_WIDTH)
            .map(|c| T::from_le_bytes(c))
            .collect())
    }

    /// Upload a typed slice as a single-column GPU buffer.
    ///
    /// Replaces: `create_buffer_from_u32_slice`, `create_buffer_from_u64_slice`,
    /// `create_buffer_from_i32_slice`, `create_buffer_from_i64_slice`,
    /// `create_buffer_from_f32_slice`, `create_buffer_from_f64_slice`,
    /// `create_buffer_from_u8_slice`.
    pub fn create_buffer_from_slice<T: GpuScalar>(
        &self,
        data: &[T],
        schema: Schema,
    ) -> Result<CudaBuffer> {
        let num_bytes = data.len().checked_mul(T::BYTE_WIDTH).ok_or_else(|| {
            XlogError::kernel_ctx(
                "create_buffer_from_slice",
                "byte size overflow",
                &data.len(),
            )
        })?;

        let mut bytes = vec![0u8; num_bytes];
        for (i, val) in data.iter().enumerate() {
            let offset = i * T::BYTE_WIDTH;
            val.to_le_bytes_into(&mut bytes[offset..offset + T::BYTE_WIDTH]);
        }

        let mut col = self.memory.alloc::<u8>(bytes.len())?;
        self.device
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .map_err(|e| {
                XlogError::kernel_ctx("create_buffer_from_slice", "htod copy failed", &e)
            })?;

        self.buffer_from_columns(vec![col.into()], data.len() as u64, schema)
    }
}
