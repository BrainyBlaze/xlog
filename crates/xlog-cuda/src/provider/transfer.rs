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
    /// Increments `d2h_transfer_count` (the per-call ILP-style counter)
    /// and is checked by the strict deterministic-Datalog D2H guard when
    /// it is enabled (see `enable_strict_deterministic_d2h`).
    pub fn download_column<T: GpuScalar>(
        &self,
        buffer: &CudaBuffer,
        col_idx: usize,
    ) -> Result<Vec<T>> {
        // Gate first so a deterministic-strict run fails before any host
        // allocation or counter mutation. Zero-row downloads are a no-op
        // in `download_column_inner` and never issue a D2H transfer, so
        // the gate must not fire for them. The row count we resolve here
        // is threaded into the inner helper so it does not look it up
        // again — the cache makes a second call cheap, but the explicit
        // hand-off keeps the contract clear.
        let num_rows = self.device_row_count(buffer)?;
        if num_rows > 0 {
            self.gate_column_download::<T>("download_column", num_rows)?;
            self.d2h_transfer_count.fetch_add(1, Ordering::Relaxed);
        }
        self.download_column_inner_with_rows::<T>(buffer, col_idx, num_rows)
    }

    /// Download a column WITHOUT incrementing the per-call
    /// `d2h_transfer_count` (the ILP-style counter). Records in
    /// `transfer_tracker` for byte/call profiling stats.
    ///
    /// IS still checked by the strict deterministic-Datalog D2H guard when
    /// it is enabled — "untracked" only refers to `d2h_transfer_count`,
    /// not to the deterministic gate. Use `dtoh_scalar_untracked` for
    /// metadata reads that must remain allowed under the gate.
    ///
    /// Replaces: `download_f64_untracked` (now generic over `T`).
    pub fn download_column_untracked<T: GpuScalar>(
        &self,
        buffer: &CudaBuffer,
        col_idx: usize,
    ) -> Result<Vec<T>> {
        let col = buffer.column(col_idx).ok_or_else(|| {
            XlogError::kernel_ctx("download_column_untracked", "column not found", &col_idx)
        })?;

        let num_rows = self.device_row_count(buffer)?;
        if num_rows == 0 {
            return Ok(vec![]);
        }

        // Gate first so a deterministic-strict run fails before any host
        // allocation, mirroring `download_column`. The downstream
        // `dtoh_sync_copy_into_tracked` would also gate, but we stop earlier
        // and with a more specific op label.
        self.gate_column_download::<T>("download_column_untracked", num_rows)?;

        let num_bytes = num_rows.checked_mul(T::BYTE_WIDTH).ok_or_else(|| {
            XlogError::kernel_ctx("download_column_untracked", "byte size overflow", &num_rows)
        })?;
        let col_view = self.column_bytes_view(col, num_bytes)?;
        let mut bytes = vec![0u8; num_bytes];
        self.dtoh_sync_copy_into_tracked(&col_view, &mut bytes)?;

        Ok(bytes
            .chunks_exact(T::BYTE_WIDTH)
            .map(|c| T::from_le_bytes(c))
            .collect())
    }

    /// Shared implementation for tracked column downloads, with the row
    /// count threaded in by the caller so we do not look it up twice.
    ///
    /// Uses the provider's tracked D2H chokepoint so transfer-budget traces
    /// observe column downloads. The caller is responsible for the early
    /// strict deterministic-D2H gate check and for incrementing
    /// `d2h_transfer_count`; this helper assumes both have already happened.
    fn download_column_inner_with_rows<T: GpuScalar>(
        &self,
        buffer: &CudaBuffer,
        col_idx: usize,
        num_rows: usize,
    ) -> Result<Vec<T>> {
        let col = buffer.column(col_idx).ok_or_else(|| {
            XlogError::kernel_ctx("download_column", "column not found", &col_idx)
        })?;

        if num_rows == 0 {
            return Ok(vec![]);
        }

        let num_bytes = num_rows.checked_mul(T::BYTE_WIDTH).ok_or_else(|| {
            XlogError::kernel_ctx("download_column", "byte size overflow", &num_rows)
        })?;
        let col_view = self.column_bytes_view(col, num_bytes)?;
        let mut bytes = vec![0u8; num_bytes];
        self.dtoh_sync_copy_into_tracked(&col_view, &mut bytes)
            .map_err(|e| XlogError::kernel_ctx("download_column", "dtoh copy failed", &e))?;

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
        self.htod_sync_copy_into_tracked(&bytes, &mut col)
            .map_err(|e| {
                XlogError::kernel_ctx("create_buffer_from_slice", "htod copy failed", &e)
            })?;

        self.buffer_from_columns(vec![col.into()], data.len() as u64, schema)
    }

    /// Probe the deterministic-D2H gate for a column download of `num_rows`
    /// rows of scalar `T`. Returns `Err` and increments the violation counter
    /// when the gate is enabled. Used by `download_column` and
    /// `download_column_untracked` so the gate fires before any host buffer
    /// is allocated or counter mutated, mirroring the chokepoint in
    /// `dtoh_sync_copy_into_tracked`.
    fn gate_column_download<T: GpuScalar>(&self, op: &'static str, num_rows: usize) -> Result<()> {
        let bytes = num_rows
            .checked_mul(T::BYTE_WIDTH)
            .ok_or_else(|| XlogError::kernel_ctx(op, "byte size overflow", &num_rows))?;
        self.check_deterministic_d2h(op, bytes as u64)
    }
}
