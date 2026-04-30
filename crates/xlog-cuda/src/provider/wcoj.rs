//! v0.6.2 GPU 3-way Worst-Case Optimal Join — provider entry.
//!
//! Single public method:
//! [`CudaKernelProvider::wcoj_triangle_u32_recorded`]. The slice is
//! intentionally narrow:
//!
//!   * **U32 only.** Caller's three relations must each have
//!     a 2-column [`xlog_core::Schema`] of `(U32, U32)`.
//!   * **Sorted, deduped inputs.** Caller-supplied:
//!     - `e_xy` lex-sorted+deduped by (X, Y),
//!     - `e_yz` lex-sorted+deduped by (Y, Z),
//!     - `e_xz` lex-sorted+deduped by (X, Z).
//!     Physical layout construction is a separate slice — this
//!     entry assumes the caller has already arranged input layout.
//!   * **Two-phase count → host-scan → materialize.** Mirrors
//!     SRDatalog (Sun et al., arXiv 2604.20073) Section 4's
//!     deterministic two-phase pipeline, simplified to a host-side
//!     prefix sum for v1. Output offsets are computed before
//!     materialization, so the kernel uses no runtime atomics on
//!     the output.
//!   * **Strict [`LaunchRecorder`] discipline.** Two recorders run
//!     sequentially on the caller-supplied launch stream:
//!     1. count recorder: reads `e_xy` / `e_yz` / `e_xz` columns
//!        + their `d_num_rows`; writes the per-row count buffer.
//!     2. materialize recorder: reads the same inputs + the
//!        host-built offsets buffer; writes the three output
//!        columns + output `d_num_rows`.
//!   * **Output deterministic and lex-sorted by (X, Y, Z).** Locked
//!     by [`tests/test_wcoj_triangle_u32.rs`].
//!   * **Set semantics on deduped input.** If the caller violates
//!     the dedup contract on an input, the kernel may emit
//!     duplicates; the test suite documents this as caller
//!     responsibility.
//!
//! What this slice deliberately does NOT do:
//!
//!   * No executor integration, no planner dispatch wiring.
//!   * No recursion / fixpoint.
//!   * No cost model.
//!   * No Symbol or u64 column variants.
//!   * No histogram-guided block dispatch (SRDatalog §5 skew opt).
//!     v1 is one thread per row of `e_xy`; sufficient for the
//!     u32 cert tests, deferred for power-law inputs.

use std::ffi::c_void;

use cudarc::driver::sys;
use xlog_core::{Result, ScalarType, Schema, XlogError};

use super::{wcoj_kernels, CudaKernelProvider, WCOJ_MODULE};
use crate::device_runtime::StreamId;
use crate::launch::LaunchRecorder;
use crate::memory::{CudaColumn, TrackedCudaSlice};
use crate::CudaBuffer;
use crate::{AsKernelParam, LaunchAsync, LaunchConfig};

const BLOCK_SIZE: u32 = 256;

impl CudaKernelProvider {
    /// Evaluate `tri(X, Y, Z) :- e_xy(X,Y), e_yz(Y,Z), e_xz(X,Z)`
    /// on already-sorted, already-deduped binary u32 relations.
    /// See module-level docs for the full contract.
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime
    ///   (`with_runtime` is required), the launch stream does
    ///   not resolve, an input is not 2-column u32, or any kernel
    ///   launch fails.
    pub fn wcoj_triangle_u32_recorded(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "wcoj_triangle_u32_recorded requires a runtime-backed \
                 GpuMemoryManager (constructed via with_runtime)"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_triangle_u32_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        // ---------------------------------------------------------
        // Validate input shape: every relation must be 2-column u32.
        // ---------------------------------------------------------
        for (label, buf) in [("e_xy", e_xy), ("e_yz", e_yz), ("e_xz", e_xz)] {
            if buf.arity() != 2 {
                return Err(XlogError::Kernel(format!(
                    "wcoj_triangle_u32_recorded: {} must be 2-column, got arity {}",
                    label,
                    buf.arity()
                )));
            }
            for col_idx in 0..2 {
                let ty = buf.schema.column_type(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!(
                        "wcoj_triangle_u32_recorded: {} column {} type missing",
                        label, col_idx
                    ))
                })?;
                if !matches!(ty, ScalarType::U32) {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_triangle_u32_recorded: {} column {} must be U32, got {:?}",
                        label, col_idx, ty
                    )));
                }
            }
        }

        // Output schema is fixed: (X: U32, Y: U32, Z: U32).
        let out_schema = Schema::new(vec![
            ("col0".to_string(), ScalarType::U32),
            ("col1".to_string(), ScalarType::U32),
            ("col2".to_string(), ScalarType::U32),
        ]);

        let n_xy = e_xy.num_rows() as u32;
        let n_yz = e_yz.num_rows() as u32;
        let n_xz = e_xz.num_rows() as u32;

        // Empty input on any side → empty result, no kernel launches.
        if n_xy == 0 || n_yz == 0 || n_xz == 0 {
            return self.create_empty_buffer(out_schema);
        }

        // ---------------------------------------------------------
        // Phase 1 (count). Pre-allocate count buffer BEFORE the
        // recorder so kernel `&mut` borrows are unaffected by
        // recorder lifetimes.
        // ---------------------------------------------------------
        let mut count_buf = self.memory.alloc::<u32>(n_xy as usize)?;

        let xy_col0 = column_u32(e_xy, 0)?;
        let xy_col1 = column_u32(e_xy, 1)?;
        let yz_col0 = column_u32(e_yz, 0)?;
        let yz_col1 = column_u32(e_yz, 1)?;
        let xz_col0 = column_u32(e_xz, 0)?;
        let xz_col1 = column_u32(e_xz, 1)?;

        let mut rec_count = LaunchRecorder::new_strict(launch_stream);
        rec_count.read(e_xy.num_rows_device());
        rec_count.read(e_yz.num_rows_device());
        rec_count.read(e_xz.num_rows_device());
        rec_count.read_column(e_xy.column(0).expect("xy.col0"));
        rec_count.read_column(e_xy.column(1).expect("xy.col1"));
        rec_count.read_column(e_yz.column(0).expect("yz.col0"));
        rec_count.read_column(e_yz.column(1).expect("yz.col1"));
        rec_count.read_column(e_xz.column(0).expect("xz.col0"));
        rec_count.read_column(e_xz.column(1).expect("xz.col1"));
        rec_count.write(&count_buf);
        rec_count.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_triangle_u32_recorded: count preflight failed: {}",
                e
            ))
        })?;

        let device = self.device.inner();
        let count_kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_COUNT)
            .ok_or_else(|| XlogError::Kernel("wcoj_triangle_count kernel not found".to_string()))?;
        let grid = (n_xy + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // SAFETY:
        //   wcoj_triangle_count(
        //     const u32* xy_col0, const u32* xy_col1, u32 n_xy,
        //     const u32* yz_col0, const u32* yz_col1, u32 n_yz,
        //     const u32* xz_col0, const u32* xz_col1, u32 n_xz,
        //     u32* out_counts)
        // All buffers are runtime-backed device memory; pointers
        // are owned by the inputs / our pre-allocated count_buf;
        // the launch is queued on launch_stream; preflight has
        // already verified cross-stream tracking.
        unsafe {
            count_kernel
                .clone()
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        xy_col0,
                        xy_col1,
                        n_xy,
                        yz_col0,
                        yz_col1,
                        n_yz,
                        xz_col0,
                        xz_col1,
                        n_xz,
                        &mut count_buf,
                    ),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("wcoj_triangle_count launch failed: {}", e))
                })?;
        }

        rec_count.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_triangle_u32_recorded: count commit failed: {}",
                e
            ))
        })?;

        // ---------------------------------------------------------
        // Host-side prefix sum + total. Drains the count kernel
        // first so the host read is well-defined; the H2D of
        // offsets is queued on launch_stream so the materialize
        // kernel sees the correct offsets.
        //
        // For v1 we accept a control-plane D2H/H2D round trip on
        // the count array. A device-side recorded scan is a later
        // optimization slice.
        // ---------------------------------------------------------
        unsafe {
            let res = sys::cuStreamSynchronize(cu_stream.cu_stream());
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "wcoj_triangle_u32_recorded: stream sync after count failed: {:?}",
                    res
                )));
            }
        }
        let mut host_counts: Vec<u32> = vec![0; n_xy as usize];
        unsafe {
            let res = sys::cuMemcpyDtoH_v2(
                host_counts.as_mut_ptr() as *mut c_void,
                *count_buf.device_ptr(),
                (n_xy as usize) * std::mem::size_of::<u32>(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "wcoj_triangle_u32_recorded: D2H counts failed: {:?}",
                    res
                )));
            }
        }
        // Exclusive prefix sum + total.
        let mut total: u64 = 0;
        let mut host_offsets: Vec<u32> = Vec::with_capacity(n_xy as usize);
        for &c in &host_counts {
            if total > u32::MAX as u64 {
                return Err(XlogError::Kernel(
                    "wcoj_triangle_u32_recorded: total triangle count overflows u32".to_string(),
                ));
            }
            host_offsets.push(total as u32);
            total += c as u64;
        }
        if total > u32::MAX as u64 {
            return Err(XlogError::Kernel(
                "wcoj_triangle_u32_recorded: total triangle count overflows u32".to_string(),
            ));
        }
        let total_rows = total as u32;

        if total_rows == 0 {
            return self.create_empty_buffer(out_schema);
        }

        // ---------------------------------------------------------
        // Phase 2 (materialize). Allocate output (now that
        // total_rows is known on host) + offsets buf, build
        // recorder, H2D offsets on launch_stream, launch
        // materialize kernel on launch_stream.
        // ---------------------------------------------------------
        let offsets_buf = self.memory.alloc::<u32>(n_xy as usize)?;
        let bytes_per_col = (total_rows as usize) * std::mem::size_of::<u32>();
        let mut out_x = self.memory.alloc::<u8>(bytes_per_col)?;
        let mut out_y = self.memory.alloc::<u8>(bytes_per_col)?;
        let mut out_z = self.memory.alloc::<u8>(bytes_per_col)?;
        let out_d_num_rows = self.memory.alloc::<u32>(1)?;

        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        rec_mat.read(e_xy.num_rows_device());
        rec_mat.read(e_yz.num_rows_device());
        rec_mat.read(e_xz.num_rows_device());
        rec_mat.read_column(e_xy.column(0).expect("xy.col0"));
        rec_mat.read_column(e_xy.column(1).expect("xy.col1"));
        rec_mat.read_column(e_yz.column(0).expect("yz.col0"));
        rec_mat.read_column(e_yz.column(1).expect("yz.col1"));
        rec_mat.read_column(e_xz.column(0).expect("xz.col0"));
        rec_mat.read_column(e_xz.column(1).expect("xz.col1"));
        rec_mat.write(&offsets_buf);
        rec_mat.write(&out_x);
        rec_mat.write(&out_y);
        rec_mat.write(&out_z);
        rec_mat.write(&out_d_num_rows);
        rec_mat.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_triangle_u32_recorded: materialize preflight failed: {}",
                e
            ))
        })?;

        // H2D offsets on launch_stream, ordered before the
        // materialize kernel (same stream).
        unsafe {
            let res = sys::cuMemcpyHtoDAsync_v2(
                *offsets_buf.device_ptr(),
                host_offsets.as_ptr() as *const c_void,
                (n_xy as usize) * std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "wcoj_triangle_u32_recorded: H2D offsets failed: {:?}",
                    res
                )));
            }
        }
        // H2D the output row count too, on launch_stream — the
        // returned CudaBuffer's d_num_rows must reflect the
        // host-known total.
        unsafe {
            let res = sys::cuMemcpyHtoDAsync_v2(
                *out_d_num_rows.device_ptr(),
                &total_rows as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "wcoj_triangle_u32_recorded: H2D out_d_num_rows failed: {:?}",
                    res
                )));
            }
        }

        let materialize_kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_MATERIALIZE)
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_triangle_materialize kernel not found".to_string())
            })?;

        // Reinterpret the per-byte output slices as u32 slices for
        // the kernel's typed signature.
        let out_x_u32: &mut TrackedCudaSlice<u32> = unsafe { reinterpret_u8_as_u32(&mut out_x) };
        let out_y_u32: &mut TrackedCudaSlice<u32> = unsafe { reinterpret_u8_as_u32(&mut out_y) };
        let out_z_u32: &mut TrackedCudaSlice<u32> = unsafe { reinterpret_u8_as_u32(&mut out_z) };

        // SAFETY:
        //   wcoj_triangle_materialize(
        //     const u32* xy_col0, const u32* xy_col1, u32 n_xy,
        //     const u32* yz_col0, const u32* yz_col1, u32 n_yz,
        //     const u32* xz_col0, const u32* xz_col1, u32 n_xz,
        //     const u32* out_offsets, u32 total_rows,
        //     u32* out_x, u32* out_y, u32* out_z)
        // 14 args exceeds the LaunchAsync tuple bound (13), so we
        // use the raw param-vec form. Same launch_on_stream
        // pattern as the multi-arg hash-join probe kernels.
        let mat_config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                xy_col0.as_kernel_param(),
                xy_col1.as_kernel_param(),
                n_xy.as_kernel_param(),
                yz_col0.as_kernel_param(),
                yz_col1.as_kernel_param(),
                n_yz.as_kernel_param(),
                xz_col0.as_kernel_param(),
                xz_col1.as_kernel_param(),
                n_xz.as_kernel_param(),
                (&offsets_buf).as_kernel_param(),
                total_rows.as_kernel_param(),
                out_x_u32.as_kernel_param(),
                out_y_u32.as_kernel_param(),
                out_z_u32.as_kernel_param(),
            ];
            materialize_kernel
                .clone()
                .launch_on_stream(&cu_stream, mat_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("wcoj_triangle_materialize launch failed: {}", e))
                })?;
        }

        rec_mat.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_triangle_u32_recorded: materialize commit failed: {}",
                e
            ))
        })?;

        let columns: Vec<CudaColumn> = vec![out_x.into(), out_y.into(), out_z.into()];
        Ok(CudaBuffer::from_columns_with_host_count(
            columns,
            total_rows as u64,
            out_d_num_rows,
            out_schema,
            total_rows,
        ))
    }
}

/// Borrow a u32 column out of a 2-column u32 [`CudaBuffer`] as a
/// typed `TrackedCudaSlice<u32>` reference for kernel binding.
///
/// Each `CudaColumn` is stored as raw bytes (`TrackedCudaSlice<u8>`
/// for owned columns); the kernel signature expects `u32*`. This
/// helper does the reinterpretation behind a single chokepoint
/// so the unsafety is colocated.
fn column_u32<'a>(buf: &'a CudaBuffer, col_idx: usize) -> Result<&'a TrackedCudaSlice<u32>> {
    let col = buf.column(col_idx).ok_or_else(|| {
        XlogError::Kernel(format!(
            "wcoj_triangle_u32_recorded: column {} not found",
            col_idx
        ))
    })?;
    match col {
        CudaColumn::Owned(slice) => {
            // SAFETY: The buffer was validated to have ScalarType::U32
            // at this column; the byte slice's len is therefore a
            // multiple of 4 and aligned. The transmute below
            // changes the element type from u8 to u32 without
            // touching the underlying raw_ptr / runtime_block,
            // both of which are agnostic to the element type.
            // The returned reference's lifetime is tied to `buf`,
            // so the original column outlives every use.
            unsafe { Ok(&*(slice as *const TrackedCudaSlice<u8> as *const TrackedCudaSlice<u32>)) }
        }
        _ => Err(XlogError::Kernel(
            "wcoj_triangle_u32_recorded: u32 columns must be Owned-variant CudaColumns".to_string(),
        )),
    }
}

/// `&mut` reinterpret of an owned u8 slice (e.g. a freshly-
/// allocated output column) as a u32 slice for kernel binding.
/// Mirrors [`column_u32`] but for the writable path.
unsafe fn reinterpret_u8_as_u32(slice: &mut TrackedCudaSlice<u8>) -> &mut TrackedCudaSlice<u32> {
    &mut *(slice as *mut TrackedCudaSlice<u8> as *mut TrackedCudaSlice<u32>)
}
