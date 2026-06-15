//! Owned, recorded helpers for WCOJ variable-ordering column
//! projection on `CudaBuffer`s.
//!
//! Two helpers, both with the same shape:
//!
//! * [`CudaKernelProvider::wcoj_project_2col_swap_recorded`] —
//!   2-col-only swap helper used by the variable-ordering dispatcher when
//!   triangle non-default leaders need a `(col1, col0)` rotated
//!   lookup atom before [`CudaKernelProvider::wcoj_layout_u32_recorded`]
//!   can sort it.
//!
//! * [`CudaKernelProvider::wcoj_project_output_columns_recorded`] —
//!   N-col arbitrary-permutation helper that the variable-ordering dispatcher
//!   applies to the kernel-direct output buffer to remap kernel
//!   columns into the rule's head order.
//!
//! Both helpers:
//!
//! 1. Build owned `TrackedCudaSlice<u8>` columns sized to the
//!    source's `row_cap × dtype.size_bytes()`.
//! 2. Allocate a fresh `num_rows_device` device scalar.
//! 3. Use `LaunchRecorder::new_strict` + `preflight` to declare
//!    reads (source columns + source `num_rows_device`) and
//!    writes (new columns + new `num_rows_device`).
//! 4. Issue per-column DtoD-async copies on the launch stream,
//!    plus one DtoD-async copy for `num_rows_device`.
//! 5. **Failure-drain**: any error after the first queued copy
//!    must `cu_stream.synchronize()` before returning, because
//!    partially-allocated owned buffers are about to drop and
//!    an in-flight DtoD copy would race the runtime dealloc.
//!    Mirrors `wcoj.rs ≈ 2140` (skew-score histogram
//!    queued-result discipline).
//! 6. Carry `cached_row_count` from `src` unchanged — logical
//!    row count is invariant under column permutation.

use cudarc::driver::sys;
use xlog_core::{Result, Schema, XlogError};

use crate::cuda_compat::DeviceSlice;
use crate::device_runtime::StreamId;
use crate::launch::LaunchRecorder;
use crate::memory::{CudaColumn, TrackedCudaSlice};
use crate::CudaBuffer;

use super::CudaKernelProvider;

impl CudaKernelProvider {
    /// Produce an owned 2-col `CudaBuffer` whose columns are
    /// `[src.col(1), src.col(0)]`. See module docs for the full
    /// recorded / failure-drain contract.
    ///
    /// Used by the variable-ordering dispatcher when a triangle non-default
    /// leader requires a col-swap before
    /// [`Self::wcoj_layout_u32_recorded`] sorts the result. The
    /// 4-cycle path is rotation-only and never invokes this
    /// helper.
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime, the
    ///   stream doesn't resolve, the input isn't 2-col, or any
    ///   queued DtoD copy fails. On any failure after the first
    ///   queued copy, the launch stream is synchronized before
    ///   the function returns.
    pub fn wcoj_project_2col_swap_recorded(
        &self,
        src: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "wcoj_project_2col_swap_recorded requires a runtime-backed \
                 GpuMemoryManager (constructed via with_runtime)"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_project_2col_swap_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        if src.arity() != 2 {
            return Err(XlogError::Kernel(format!(
                "wcoj_project_2col_swap_recorded: src must be 2-column, got arity {}",
                src.arity()
            )));
        }

        // Build the swapped schema. The (name, type) pairs are
        // re-ordered; key_columns recomputes via Schema::new.
        let swapped_schema = Schema::new(vec![
            src.schema.columns[1].clone(),
            src.schema.columns[0].clone(),
        ])
        .with_sort_labels(vec![
            src.schema
                .column_sort_label(1)
                .unwrap_or("col1")
                .to_string(),
            src.schema
                .column_sort_label(0)
                .unwrap_or("col0")
                .to_string(),
        ])
        .expect("swapped sort labels match schema arity");

        // Empty buffer fast-path: arity=2 with row_cap=0 produces
        // an empty owned buffer with the swapped schema. No DtoD
        // copy is queued; failure-drain is moot.
        if src.row_cap == 0 {
            return self.create_empty_buffer(swapped_schema);
        }

        // Allocate the two output column buffers (sized in bytes
        // from the source) plus a fresh num_rows_device scalar.
        // SAFETY: src.column(0/1) are guaranteed Some (arity == 2);
        // their byte length matches row_cap × dtype.size_bytes()
        // by CudaBuffer construction invariants.
        let bytes_col0 = src.column(0).expect("src.col0").len();
        let bytes_col1 = src.column(1).expect("src.col1").len();
        let new_col0: TrackedCudaSlice<u8> = self.memory.alloc::<u8>(bytes_col1)?;
        let new_col1: TrackedCudaSlice<u8> = self.memory.alloc::<u8>(bytes_col0)?;
        let new_num_rows: TrackedCudaSlice<u32> = self.memory.alloc::<u32>(1)?;

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read_column(src.column(0).expect("src.col0"));
        rec.read_column(src.column(1).expect("src.col1"));
        rec.read(src.num_rows_device());
        rec.write(&new_col0);
        rec.write(&new_col1);
        rec.write(&new_num_rows);
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_project_2col_swap_recorded: launch recorder preflight failed: {}",
                e
            ))
        })?;

        // Failure-drain: from here on, any error must synchronize
        // the stream before returning so partially-issued copies
        // drain before the runtime deallocs `new_col0` /
        // `new_col1` / `new_num_rows` on Err drop.
        let queued_result: Result<()> = (|| {
            // SAFETY: all DtoD copies are between live device
            // pointers on a stream the runtime owns. Sizes match
            // the source columns exactly. cuMemcpyDtoDAsync_v2 is
            // genuinely stream-asynchronous.
            unsafe {
                let res = sys::cuMemcpyDtoDAsync_v2(
                    *new_col0.device_ptr(),
                    *src.column(1).expect("src.col1").device_ptr(),
                    bytes_col1,
                    cu_stream.cu_stream(),
                );
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_project_2col_swap_recorded: dtod col1 → new_col0 failed: {:?}",
                        res
                    )));
                }
                let res = sys::cuMemcpyDtoDAsync_v2(
                    *new_col1.device_ptr(),
                    *src.column(0).expect("src.col0").device_ptr(),
                    bytes_col0,
                    cu_stream.cu_stream(),
                );
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_project_2col_swap_recorded: dtod col0 → new_col1 failed: {:?}",
                        res
                    )));
                }
                let res = sys::cuMemcpyDtoDAsync_v2(
                    *new_num_rows.device_ptr(),
                    *src.num_rows_device().device_ptr(),
                    std::mem::size_of::<u32>(),
                    cu_stream.cu_stream(),
                );
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_project_2col_swap_recorded: dtod num_rows_device failed: {:?}",
                        res
                    )));
                }
            }
            Ok(())
        })();

        if let Err(e) = queued_result {
            let _ = cu_stream.synchronize();
            return Err(e);
        }

        rec.commit(runtime).map_err(|e| {
            // commit happens after CUDA work is queued; if commit
            // fails, drain so partially-issued copies finish
            // before the buffers drop.
            let _ = cu_stream.synchronize();
            XlogError::Kernel(format!(
                "wcoj_project_2col_swap_recorded: launch recorder commit failed: {}",
                e
            ))
        })?;

        // Build the output buffer. Carry cached_row_count from src
        // (logical row count is invariant under column permutation).
        let columns: Vec<CudaColumn> = vec![new_col0.into(), new_col1.into()];
        let buf = match src.cached_row_count() {
            Some(host_count) => CudaBuffer::from_columns_with_host_count(
                columns,
                src.row_cap,
                new_num_rows,
                swapped_schema,
                host_count,
            ),
            None => CudaBuffer::from_columns(columns, src.row_cap, new_num_rows, swapped_schema),
        };
        Ok(buf)
    }

    /// Produce an owned N-col `CudaBuffer` with columns
    /// reordered per `perm` and the schema replaced with
    /// `head_schema`.
    ///
    /// `perm[i]` is the source column index that becomes output
    /// column `i`. The dispatcher uses this post-kernel to remap
    /// the kernel-direct output (in leader's `(a, b, c[, d])`
    /// order) into the rule's head order.
    ///
    /// See module docs for the full recorded / failure-drain
    /// contract.
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime, the
    ///   stream doesn't resolve, `perm.len() != head_schema.arity()`,
    ///   any `perm` index is ≥ `src.arity()`, or any queued DtoD
    ///   copy fails. Failure-drain on Err.
    pub fn wcoj_project_output_columns_recorded(
        &self,
        src: &CudaBuffer,
        perm: &[usize],
        head_schema: Schema,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "wcoj_project_output_columns_recorded requires a runtime-backed \
                 GpuMemoryManager (constructed via with_runtime)"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_project_output_columns_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        if perm.len() != head_schema.arity() {
            return Err(XlogError::Kernel(format!(
                "wcoj_project_output_columns_recorded: perm len {} must equal head_schema arity {}",
                perm.len(),
                head_schema.arity()
            )));
        }
        for (i, &p) in perm.iter().enumerate() {
            if p >= src.arity() {
                return Err(XlogError::Kernel(format!(
                    "wcoj_project_output_columns_recorded: perm[{}] = {} out of bounds (src arity {})",
                    i,
                    p,
                    src.arity()
                )));
            }
        }

        // Empty-output short-circuit. WCOJ legitimately produces
        // empty output — the helper must NOT divide-by-zero or
        // refuse to materialize. cached_row_count == Some(0) +
        // num_rows_device == 0 are produced via create_empty_buffer
        // (sized to head_schema; its num_rows_device starts at 0).
        if src.row_cap == 0 {
            return self.create_empty_buffer(head_schema);
        }

        // Allocate one new column buffer per output position +
        // a fresh num_rows_device scalar.
        let mut new_columns: Vec<TrackedCudaSlice<u8>> = Vec::with_capacity(perm.len());
        for &p in perm {
            let bytes = src.column(p).expect("src column").len();
            new_columns.push(self.memory.alloc::<u8>(bytes)?);
        }
        let new_num_rows: TrackedCudaSlice<u32> = self.memory.alloc::<u32>(1)?;

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        for i in 0..src.arity() {
            rec.read_column(src.column(i).expect("src.col"));
        }
        rec.read(src.num_rows_device());
        for c in &new_columns {
            rec.write(c);
        }
        rec.write(&new_num_rows);
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_project_output_columns_recorded: launch recorder preflight failed: {}",
                e
            ))
        })?;

        let queued_result: Result<()> = (|| {
            for (i, &p) in perm.iter().enumerate() {
                let src_col = src.column(p).expect("src column");
                let bytes = src_col.len();
                // SAFETY: device pointer is live, len matches source
                // column. Stream is owned by the runtime.
                unsafe {
                    let res = sys::cuMemcpyDtoDAsync_v2(
                        *new_columns[i].device_ptr(),
                        *src_col.device_ptr(),
                        bytes,
                        cu_stream.cu_stream(),
                    );
                    if res != sys::cudaError_enum::CUDA_SUCCESS {
                        return Err(XlogError::Kernel(format!(
                            "wcoj_project_output_columns_recorded: dtod perm[{}] = src col {} failed: {:?}",
                            i, p, res
                        )));
                    }
                }
            }
            // num_rows_device DtoD-copy.
            unsafe {
                let res = sys::cuMemcpyDtoDAsync_v2(
                    *new_num_rows.device_ptr(),
                    *src.num_rows_device().device_ptr(),
                    std::mem::size_of::<u32>(),
                    cu_stream.cu_stream(),
                );
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_project_output_columns_recorded: dtod num_rows_device failed: {:?}",
                        res
                    )));
                }
            }
            Ok(())
        })();

        if let Err(e) = queued_result {
            let _ = cu_stream.synchronize();
            return Err(e);
        }

        rec.commit(runtime).map_err(|e| {
            let _ = cu_stream.synchronize();
            XlogError::Kernel(format!(
                "wcoj_project_output_columns_recorded: launch recorder commit failed: {}",
                e
            ))
        })?;

        let columns: Vec<CudaColumn> = new_columns.into_iter().map(|c| c.into()).collect();
        let buf = match src.cached_row_count() {
            Some(host_count) => CudaBuffer::from_columns_with_host_count(
                columns,
                src.row_cap,
                new_num_rows,
                head_schema,
                host_count,
            ),
            None => CudaBuffer::from_columns(columns, src.row_cap, new_num_rows, head_schema),
        };
        Ok(buf)
    }
}
