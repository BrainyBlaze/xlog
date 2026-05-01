//! v0.6.2 GPU 3-way Worst-Case Optimal Join — provider entries.
//!
//! Public methods (parallel u32 and u64 entries — see
//! [`CudaKernelProvider::wcoj_triangle_u32_recorded`] and
//! [`CudaKernelProvider::wcoj_triangle_u64_recorded`]):
//!
//!   * **U32 / Symbol** entry takes 2-column inputs whose columns
//!     may be [`xlog_core::ScalarType::U32`] or
//!     [`xlog_core::ScalarType::Symbol`] (both share the same
//!     4-byte physical layout). The kernel does bit-equality
//!     joins; cross-relation type compatibility is enforced by
//!     the planner upstream.
//!   * **U64** entry takes 2-column inputs whose columns are
//!     [`xlog_core::ScalarType::U64`] only. Backed by parallel
//!     `_u64` count + materialize kernels in `wcoj.cu`; counters
//!     and the `wcoj_compute_total` reducer are reused unchanged
//!     (they're bounded by `u32::MAX` rows).
//!   * Mixed-width inputs in the same triangle are rejected at
//!     the provider level — each entry's schema guard requires
//!     all three relations match its width.
//!   * **Sorted, deduped inputs.** Caller-supplied:
//!     - `e_xy` lex-sorted+deduped by (X, Y),
//!     - `e_yz` lex-sorted+deduped by (Y, Z),
//!     - `e_xz` lex-sorted+deduped by (X, Z).
//!     Physical layout construction is a separate slice — this
//!     entry assumes the caller has already arranged input layout.
//!   * **Two-phase count → device-scan → materialize.** Mirrors
//!     SRDatalog (Sun et al., arXiv 2604.20073) Section 4's
//!     deterministic two-phase pipeline. Per-row counts are
//!     prefix-summed *on device* via the existing recorded
//!     `multiblock_scan_u32_inplace_on_stream` helper; the only
//!     host visit between the two phases is a single 4-byte
//!     `dtoh_scalar_untracked` of the inclusive total
//!     (sanctioned metadata read, exempt from the strict
//!     deterministic-D2H gate). The v1 count-vector D2H + host
//!     prefix sum + offsets H2D round trip is gone.
//!   * **Strict [`LaunchRecorder`] discipline.** Two recorders
//!     run sequentially on the caller-supplied launch stream:
//!     1. count+scan recorder: reads `e_xy` / `e_yz` / `e_xz`
//!        columns + their `d_num_rows`; writes `count_buf`,
//!        `offsets_buf`, `d_total`. Spans the count kernel,
//!        the dtod copy `count_buf → offsets_buf`, the
//!        device-side prefix-sum on `offsets_buf`, and
//!        `wcoj_compute_total`.
//!     2. materialize recorder: reads same inputs + `offsets_buf`;
//!        writes the three output columns + output `d_num_rows`.
//!   * **Output deterministic and lex-sorted by (X, Y, Z).** Locked
//!     by [`tests/test_wcoj_triangle_u32.rs`].
//!   * **Set semantics on deduped input.** If the caller violates
//!     the dedup contract on an input, the kernel may emit
//!     duplicates; the test suite documents this as caller
//!     responsibility.
//!
//! What this slice deliberately does NOT do:
//!
//!   * No executor integration, no planner dispatch wiring (the
//!     AST/RIR dispatch helpers live in `xlog-integration` and
//!     `xlog-runtime` and gain U64 admission in commit 3 of this
//!     slice).
//!   * No recursion / fixpoint.
//!   * No cost model.
//!   * No histogram-guided block dispatch (SRDatalog §5 skew opt).
//!     v1 is one thread per row of `e_xy`; sufficient for the
//!     correctness cert tests, deferred for power-law inputs.

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
    /// Build the sorted+deduped WCOJ physical layout for a 2-column
    /// u32 relation.
    ///
    /// Output: a 2-column u32 [`CudaBuffer`] sorted lexicographically
    /// by `(col0, col1)` and deduplicated. The output is suitable for
    /// direct consumption by [`Self::wcoj_triangle_u32_recorded`] in
    /// any of the three slot positions (`e_xy`, `e_yz`, `e_xz`); the
    /// caller chooses which logical relation each input represents
    /// by the slot it passes the layout into.
    ///
    /// Composes existing recorded primitives end-to-end:
    /// [`Self::dedup_full_row_recorded`] internally invokes
    /// [`Self::sort_recorded`] (typed multi-column radix sort on
    /// `(col0, col1)`) followed by an on-stream
    /// `mark_unique_full_row_bytewise` mask + counted compaction —
    /// each primitive carries its own [`crate::launch::LaunchRecorder`]
    /// commit and the runtime's record-all + wait-all
    /// `last_use_events` chains the dealloc safety end-to-end.
    ///
    /// This entry exists for two reasons:
    ///   1. Narrowing the input contract to 2-column u32 lets the
    ///      WCOJ-specific call site fail fast with a clear error
    ///      rather than the more generic dedup error if the caller
    ///      passes the wrong arity / type.
    ///   2. Naming the WCOJ pipeline boundary makes downstream
    ///      callers (planner / executor wiring, cert harness)
    ///      target the WCOJ-specific layout API rather than the
    ///      general-purpose dedup primitive — separating concerns
    ///      that may diverge as the WCOJ stack grows.
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime
    ///   (`with_runtime` is required), the input is not 2-column,
    ///   any column is not [`ScalarType::U32`] or
    ///   [`ScalarType::Symbol`] (both share the same 4-byte
    ///   physical layout, so the underlying sort/dedup primitives
    ///   handle either with no kernel changes), or any inner
    ///   sort/dedup primitive fails.
    pub fn wcoj_layout_u32_recorded(
        &self,
        input: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        // Manager must be runtime-backed — the inner
        // dedup_full_row_recorded enforces the same constraint, but
        // checking here gives a WCOJ-specific error message.
        if self.memory().runtime().is_none() {
            return Err(XlogError::Kernel(
                "wcoj_layout_u32_recorded requires a runtime-backed \
                 GpuMemoryManager (constructed via with_runtime)"
                    .to_string(),
            ));
        }
        // 2-column 4-byte-key contract: U32 or Symbol per column.
        // Both share the same 4-byte physical layout
        // (`ScalarType::size_bytes` == 4); the sort+dedup
        // primitives we delegate to already accept either.
        if input.arity() != 2 {
            return Err(XlogError::Kernel(format!(
                "wcoj_layout_u32_recorded: input must be 2-column, got arity {}",
                input.arity()
            )));
        }
        for col_idx in 0..2 {
            let ty = input.schema.column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_layout_u32_recorded: column {} type missing",
                    col_idx
                ))
            })?;
            if !matches!(ty, ScalarType::U32 | ScalarType::Symbol) {
                return Err(XlogError::Kernel(format!(
                    "wcoj_layout_u32_recorded: column {} must be U32 or Symbol, got {:?}",
                    col_idx, ty
                )));
            }
        }
        // Delegate to the existing typed sort + full-row dedup.
        // Both primitives are fully recorder-disciplined; the
        // resulting CudaBuffer is sorted lex by (col0, col1) and
        // deduplicated.
        self.dedup_full_row_recorded(input, launch_stream)
    }

    /// Evaluate `tri(X, Y, Z) :- e_xy(X,Y), e_yz(Y,Z), e_xz(X,Z)`
    /// on already-sorted, already-deduped binary 4-byte-key
    /// relations. See module-level docs for the full contract.
    ///
    /// Each column may be [`ScalarType::U32`] or
    /// [`ScalarType::Symbol`] — both share the same 4-byte
    /// physical layout, so the kernel reads the bits unchanged.
    /// Cross-relation type compatibility (e.g., that Y is the
    /// same type in `e_xy.col1` and `e_yz.col0`) is the
    /// planner's responsibility upstream; this entry only
    /// enforces width.
    ///
    /// The output schema preserves per-head-position scalar types
    /// from the inputs:
    ///   * `out.col0` = `e_xy.col0` type (X)
    ///   * `out.col1` = `e_xy.col1` type (Y)
    ///   * `out.col2` = `e_yz.col1` type (Z)
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime
    ///   (`with_runtime` is required), the launch stream does
    ///   not resolve, an input is not 2-column with U32/Symbol
    ///   columns, or any kernel launch fails.
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
        // Validate input shape: every relation must be 2-column
        // with U32 or Symbol columns (both 4-byte physical).
        // Cross-relation type compatibility is enforced upstream
        // by the planner / hypergraph type inference.
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
                if !matches!(ty, ScalarType::U32 | ScalarType::Symbol) {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_triangle_u32_recorded: {} column {} must be U32 or Symbol, got {:?}",
                        label, col_idx, ty
                    )));
                }
            }
        }

        // Output schema mirrors the inputs' per-head-position
        // scalar types: out.col0 = X = e_xy.col0; out.col1 = Y =
        // e_xy.col1; out.col2 = Z = e_yz.col1. Cross-relation
        // type compatibility (X type same in e_xz.col0, Y same
        // in e_yz.col0, Z same in e_xz.col1) is the planner's
        // job; we trust it here.
        let x_type = e_xy.schema.column_type(0).expect("validated above");
        let y_type = e_xy.schema.column_type(1).expect("validated above");
        let z_type = e_yz.schema.column_type(1).expect("validated above");
        let out_schema = Schema::new(vec![
            ("col0".to_string(), x_type),
            ("col1".to_string(), y_type),
            ("col2".to_string(), z_type),
        ]);

        // CudaBuffer::num_rows() returns `row_cap` (allocation
        // capacity), which can exceed the logical row count for
        // primitives that compact in place (e.g. the deduped
        // outputs of `wcoj_layout_u32_recorded`). The kernel must
        // grid-dispatch over the LOGICAL row count, not the cap,
        // so we prefer the host-cached count and fall back to a
        // 4-byte `dtoh_scalar_untracked` of `d_num_rows`.
        // `dtoh_scalar_untracked` is the sanctioned metadata-read
        // path the strict deterministic-D2H gate whitelists.
        let n_xy = self.logical_row_count_u32(e_xy)?;
        let n_yz = self.logical_row_count_u32(e_yz)?;
        let n_xz = self.logical_row_count_u32(e_xz)?;

        // Empty input on any side → empty result, no kernel launches.
        if n_xy == 0 || n_yz == 0 || n_xz == 0 {
            return self.create_empty_buffer(out_schema);
        }

        // ---------------------------------------------------------
        // Phase 1 (count + device scan + total). Pre-allocate
        // every fresh buffer BEFORE the recorder so kernel `&mut`
        // borrows are unaffected by recorder lifetimes.
        //   * `count_buf`   — per-row triangle counts (preserved
        //     for `wcoj_compute_total` to read counts[n-1]).
        //   * `offsets_buf` — receives a dtod copy of `count_buf`,
        //     then is exclusive-scanned in place to become the
        //     per-row write offsets.
        //   * `d_total`     — 1-element scalar holding the inclusive
        //     total triangle count, written by `wcoj_compute_total`,
        //     read by the host via `dtoh_scalar_untracked`.
        // ---------------------------------------------------------
        let mut count_buf = self.memory.alloc::<u32>(n_xy as usize)?;
        let mut offsets_buf = self.memory.alloc::<u32>(n_xy as usize)?;
        let d_total = self.memory.alloc::<u32>(1)?;

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
        rec_count.write(&offsets_buf);
        rec_count.write(&d_total);
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

        // dtod-async copy `count_buf → offsets_buf` on launch_stream.
        // The scan helper runs in place; we keep `count_buf` intact
        // so `wcoj_compute_total` can read counts[n-1] directly.
        // SAFETY: both buffers are runtime-backed u32 slices of
        // length `n_xy`, both sized `n_xy * 4` bytes.
        let bytes_count = (n_xy as usize) * std::mem::size_of::<u32>();
        unsafe {
            let res = sys::cuMemcpyDtoDAsync_v2(
                *offsets_buf.device_ptr(),
                *count_buf.device_ptr(),
                bytes_count,
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "wcoj_triangle_u32_recorded: dtod count_buf → offsets_buf failed: {:?}",
                    res
                )));
            }
        }

        // Device-side exclusive prefix-sum on `offsets_buf` over
        // `[0..n_xy)`. The helper is `pub(crate)` and recorded
        // against `launch_stream` via the supplied `cu_stream` /
        // `runtime` arguments — no host involvement.
        self.multiblock_scan_u32_inplace_on_stream(
            &mut offsets_buf,
            n_xy,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        // Reduce the two last elements into `d_total`.
        // SAFETY: 4-arg signature
        //   wcoj_compute_total(const u32* counts, const u32* offsets,
        //                      u32 n, u32* total)
        let total_kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_COMPUTE_TOTAL)
            .ok_or_else(|| XlogError::Kernel("wcoj_compute_total kernel not found".to_string()))?;
        unsafe {
            total_kernel
                .clone()
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&count_buf, &offsets_buf, n_xy, &d_total),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("wcoj_compute_total launch failed: {}", e))
                })?;
        }

        rec_count.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_triangle_u32_recorded: count+scan+total commit failed: {}",
                e
            ))
        })?;

        // Sync + sanctioned scalar D2H of the total.
        // `dtoh_scalar_untracked` is the metadata-read path; the
        // strict deterministic-D2H gate explicitly whitelists it.
        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_triangle_u32_recorded: stream sync after total failed: {}",
                e
            ))
        })?;
        let total_rows = self
            .dtoh_scalar_untracked::<u32>(&d_total, 0)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "wcoj_triangle_u32_recorded: read d_total failed: {}",
                    e
                ))
            })?;

        if total_rows == 0 {
            return self.create_empty_buffer(out_schema);
        }

        // ---------------------------------------------------------
        // Phase 2 (materialize). Allocate output (now that
        // total_rows is known on host via the scalar metadata
        // read), build recorder, launch materialize on launch_stream.
        // ---------------------------------------------------------
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
        rec_mat.read(&offsets_buf);
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

        // H2D the output row count on launch_stream — the returned
        // CudaBuffer's d_num_rows must reflect the host-known total.
        // This is a 4-byte H2D (the *only* host→device transfer in
        // the path); it does not involve any column-sized data.
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

impl CudaKernelProvider {
    /// Resolve a [`CudaBuffer`]'s logical row count to a `u32`.
    ///
    /// Prefers the cached host count when set (no host syscalls).
    /// Falls back to a 4-byte `dtoh_scalar_untracked` of the
    /// device-resident `d_num_rows` slot — the sanctioned
    /// metadata-read path the strict deterministic-D2H gate
    /// explicitly whitelists.
    ///
    /// `CudaBuffer::num_rows()` is intentionally not used here:
    /// it returns `row_cap` (allocation capacity), which can
    /// exceed the logical count for compact-in-place primitives
    /// like `dedup_full_row_recorded` / `wcoj_layout_u32_recorded`.
    fn logical_row_count_u32(&self, buf: &CudaBuffer) -> Result<u32> {
        if let Some(c) = buf.cached_row_count() {
            return Ok(c);
        }
        self.dtoh_scalar_untracked::<u32>(buf.num_rows_device(), 0)
    }
}

// ===============================================================
// U64 entries — parallel to the u32/Symbol pair above. Counters /
// per-row counts / output offsets / total-rows scalar all remain
// u32 (bounded by `u32::MAX` rows). Only the join-key reads and
// output columns are 8-byte; the prefix-scan helper, the
// total-reducer kernel, the dtoh-scalar fast-path, and the
// strict-D2H gate all reuse the u32 plumbing unchanged.
// ===============================================================

impl CudaKernelProvider {
    /// Build the sorted+deduped WCOJ physical layout for a 2-column
    /// U64 relation. Output: a 2-column U64 [`CudaBuffer`] sorted
    /// lexicographically by `(col0, col1)` and deduplicated. Suitable
    /// for direct consumption by [`Self::wcoj_triangle_u64_recorded`].
    ///
    /// Composition mirrors [`Self::wcoj_layout_u32_recorded`]:
    /// delegates to [`Self::dedup_full_row_recorded`], which
    /// gained U64 admission in commit 1 of this slice (its
    /// internal `sort_recorded` ports the legacy `sort()`'s hi/lo
    /// radix-pass strategy into the recorded path).
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime, the
    ///   input is not 2-column, any column is not
    ///   [`ScalarType::U64`], or any inner sort/dedup primitive
    ///   fails.
    pub fn wcoj_layout_u64_recorded(
        &self,
        input: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        if self.memory().runtime().is_none() {
            return Err(XlogError::Kernel(
                "wcoj_layout_u64_recorded requires a runtime-backed \
                 GpuMemoryManager (constructed via with_runtime)"
                    .to_string(),
            ));
        }
        if input.arity() != 2 {
            return Err(XlogError::Kernel(format!(
                "wcoj_layout_u64_recorded: input must be 2-column, got arity {}",
                input.arity()
            )));
        }
        for col_idx in 0..2 {
            let ty = input.schema.column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_layout_u64_recorded: column {} type missing",
                    col_idx
                ))
            })?;
            if !matches!(ty, ScalarType::U64) {
                return Err(XlogError::Kernel(format!(
                    "wcoj_layout_u64_recorded: column {} must be U64, got {:?}",
                    col_idx, ty
                )));
            }
        }
        // dedup_full_row_recorded internally invokes sort_recorded
        // (U64-aware after commit 1) and the bytewise mask kernel
        // (already width-generic via col_sizes upload).
        self.dedup_full_row_recorded(input, launch_stream)
    }

    /// Evaluate `tri(X, Y, Z) :- e_xy(X,Y), e_yz(Y,Z), e_xz(X,Z)`
    /// on already-sorted, already-deduped binary U64 relations.
    /// Mirrors [`Self::wcoj_triangle_u32_recorded`]'s contract;
    /// the only differences are the 8-byte join-key reads/writes
    /// and the U64-specific count/materialize kernels. Counters
    /// and the total reducer remain u32.
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime, the
    ///   launch stream does not resolve, an input is not 2-column
    ///   with U64 columns, or any kernel launch fails.
    pub fn wcoj_triangle_u64_recorded(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "wcoj_triangle_u64_recorded requires a runtime-backed \
                 GpuMemoryManager (constructed via with_runtime)"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_triangle_u64_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        // Validate: every relation 2-column U64.
        for (label, buf) in [("e_xy", e_xy), ("e_yz", e_yz), ("e_xz", e_xz)] {
            if buf.arity() != 2 {
                return Err(XlogError::Kernel(format!(
                    "wcoj_triangle_u64_recorded: {} must be 2-column, got arity {}",
                    label,
                    buf.arity()
                )));
            }
            for col_idx in 0..2 {
                let ty = buf.schema.column_type(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!(
                        "wcoj_triangle_u64_recorded: {} column {} type missing",
                        label, col_idx
                    ))
                })?;
                if !matches!(ty, ScalarType::U64) {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_triangle_u64_recorded: {} column {} must be U64, got {:?}",
                        label, col_idx, ty
                    )));
                }
            }
        }

        // Output schema preserves per-head-position scalar types
        // (all U64 by construction at this point).
        let out_schema = Schema::new(vec![
            ("col0".to_string(), ScalarType::U64),
            ("col1".to_string(), ScalarType::U64),
            ("col2".to_string(), ScalarType::U64),
        ]);

        let n_xy = self.logical_row_count_u32(e_xy)?;
        let n_yz = self.logical_row_count_u32(e_yz)?;
        let n_xz = self.logical_row_count_u32(e_xz)?;

        if n_xy == 0 || n_yz == 0 || n_xz == 0 {
            return self.create_empty_buffer(out_schema);
        }

        // Phase 1: count + scan + total. Counter buffers stay u32.
        let mut count_buf = self.memory.alloc::<u32>(n_xy as usize)?;
        let mut offsets_buf = self.memory.alloc::<u32>(n_xy as usize)?;
        let d_total = self.memory.alloc::<u32>(1)?;

        let xy_col0 = column_u64(e_xy, 0)?;
        let xy_col1 = column_u64(e_xy, 1)?;
        let yz_col0 = column_u64(e_yz, 0)?;
        let yz_col1 = column_u64(e_yz, 1)?;
        let xz_col0 = column_u64(e_xz, 0)?;
        let xz_col1 = column_u64(e_xz, 1)?;

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
        rec_count.write(&offsets_buf);
        rec_count.write(&d_total);
        rec_count.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_triangle_u64_recorded: count preflight failed: {}",
                e
            ))
        })?;

        let device = self.device.inner();
        let count_kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_COUNT_U64)
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_triangle_count_u64 kernel not found".to_string())
            })?;
        let grid = (n_xy + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // SAFETY: same 10-arg signature as the u32 count kernel
        // but with u64 join-key buffers; counts stay u32.
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
                    XlogError::Kernel(format!("wcoj_triangle_count_u64 launch failed: {}", e))
                })?;
        }

        // dtod-async copy `count_buf → offsets_buf` (u32-sized).
        let bytes_count = (n_xy as usize) * std::mem::size_of::<u32>();
        unsafe {
            let res = sys::cuMemcpyDtoDAsync_v2(
                *offsets_buf.device_ptr(),
                *count_buf.device_ptr(),
                bytes_count,
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "wcoj_triangle_u64_recorded: dtod count_buf → offsets_buf failed: {:?}",
                    res
                )));
            }
        }

        // Device-side exclusive prefix-sum on offsets_buf — u32 plumbing
        // unchanged from the u32 path.
        self.multiblock_scan_u32_inplace_on_stream(
            &mut offsets_buf,
            n_xy,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        // Reduce two last elements into d_total. Reused unchanged
        // since counters stay u32.
        let total_kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_COMPUTE_TOTAL)
            .ok_or_else(|| XlogError::Kernel("wcoj_compute_total kernel not found".to_string()))?;
        unsafe {
            total_kernel
                .clone()
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&count_buf, &offsets_buf, n_xy, &d_total),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("wcoj_compute_total launch failed: {}", e))
                })?;
        }

        rec_count.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_triangle_u64_recorded: count+scan+total commit failed: {}",
                e
            ))
        })?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_triangle_u64_recorded: stream sync after total failed: {}",
                e
            ))
        })?;
        let total_rows = self
            .dtoh_scalar_untracked::<u32>(&d_total, 0)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "wcoj_triangle_u64_recorded: read d_total failed: {}",
                    e
                ))
            })?;

        if total_rows == 0 {
            return self.create_empty_buffer(out_schema);
        }

        // Phase 2: materialize. Output columns are 8 bytes/row.
        let bytes_per_col = (total_rows as usize) * std::mem::size_of::<u64>();
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
        rec_mat.read(&offsets_buf);
        rec_mat.write(&out_x);
        rec_mat.write(&out_y);
        rec_mat.write(&out_z);
        rec_mat.write(&out_d_num_rows);
        rec_mat.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_triangle_u64_recorded: materialize preflight failed: {}",
                e
            ))
        })?;

        // 4-byte H2D of total_rows into out_d_num_rows. Identical
        // to u32 path — d_num_rows is u32 regardless of key width.
        unsafe {
            let res = sys::cuMemcpyHtoDAsync_v2(
                *out_d_num_rows.device_ptr(),
                &total_rows as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "wcoj_triangle_u64_recorded: H2D out_d_num_rows failed: {:?}",
                    res
                )));
            }
        }

        let materialize_kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_MATERIALIZE_U64)
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_triangle_materialize_u64 kernel not found".to_string())
            })?;

        let out_x_u64: &mut TrackedCudaSlice<u64> = unsafe { reinterpret_u8_as_u64(&mut out_x) };
        let out_y_u64: &mut TrackedCudaSlice<u64> = unsafe { reinterpret_u8_as_u64(&mut out_y) };
        let out_z_u64: &mut TrackedCudaSlice<u64> = unsafe { reinterpret_u8_as_u64(&mut out_z) };

        // SAFETY: 14-arg materialize_u64 (u64* keys + u32 counters
        // + u64* outputs). Same param-vec form as the u32 path —
        // exceeds the LaunchAsync tuple bound (13).
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
                out_x_u64.as_kernel_param(),
                out_y_u64.as_kernel_param(),
                out_z_u64.as_kernel_param(),
            ];
            materialize_kernel
                .clone()
                .launch_on_stream(&cu_stream, mat_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "wcoj_triangle_materialize_u64 launch failed: {}",
                        e
                    ))
                })?;
        }

        rec_mat.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_triangle_u64_recorded: materialize commit failed: {}",
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

// ===============================================================
// v0.6.2 A2-lite — skew classifier provider entries.
//
// Single-launch combined histogram across the three triangle
// join-key columns (e1.col1, e2.col0, e3.col0). Returns
// `Ok(Some(score))` on success or `Ok(None)` on any classifier-
// side failure (silent fallback per slice spec — classifier is
// optimization, not correctness).
// ===============================================================

const WCOJ_SKEW_NUM_BUCKETS: u32 = 64;

impl CudaKernelProvider {
    /// Compute the WCOJ adaptive-dispatch skew score for a u32
    /// (or Symbol) triangle. Returns `Ok(Some(score))` on success
    /// where `score = max(max_bucket_i / row_count_i)` over the
    /// three join-key columns. Threshold convention (consumed in
    /// commit B): `score >= 0.10` → dispatch WCOJ, else fall back.
    ///
    /// Returns `Ok(None)` on any failure (no runtime, validation
    /// failure, kernel error, D2H error, zero rows on any input).
    /// Caller silently falls back to binary join.
    pub fn wcoj_triangle_skew_score_u32(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<Option<f64>> {
        match self.try_compute_skew_score_u32(e_xy, e_yz, e_xz, launch_stream) {
            Ok(score) => Ok(Some(score)),
            Err(_) => Ok(None),
        }
    }

    /// Compute the WCOJ adaptive-dispatch skew score for a U64
    /// triangle. See [`Self::wcoj_triangle_skew_score_u32`].
    pub fn wcoj_triangle_skew_score_u64(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<Option<f64>> {
        match self.try_compute_skew_score_u64(e_xy, e_yz, e_xz, launch_stream) {
            Ok(score) => Ok(Some(score)),
            Err(_) => Ok(None),
        }
    }

    fn try_compute_skew_score_u32(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<f64> {
        let runtime = self
            .memory()
            .runtime()
            .ok_or_else(|| XlogError::Kernel("skew_score: no runtime".to_string()))?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| XlogError::Kernel("skew_score: stream resolve failed".to_string()))?;

        // Validate: 2-col with U32 or Symbol on each input.
        for (label, buf) in [("e_xy", e_xy), ("e_yz", e_yz), ("e_xz", e_xz)] {
            if buf.arity() != 2 {
                return Err(XlogError::Kernel(format!(
                    "skew_score: {label} must be 2-column"
                )));
            }
            for col_idx in 0..2 {
                let ty = buf
                    .schema
                    .column_type(col_idx)
                    .ok_or_else(|| XlogError::Kernel("skew_score: type missing".to_string()))?;
                if !matches!(ty, ScalarType::U32 | ScalarType::Symbol) {
                    return Err(XlogError::Kernel(format!(
                        "skew_score: {label}.{col_idx} not U32/Symbol"
                    )));
                }
            }
        }

        let n_xy = self.logical_row_count_u32(e_xy)?;
        let n_yz = self.logical_row_count_u32(e_yz)?;
        let n_xz = self.logical_row_count_u32(e_xz)?;
        if n_xy == 0 || n_yz == 0 || n_xz == 0 {
            return Err(XlogError::Kernel("skew_score: zero rows".to_string()));
        }

        // Classifier reads e_xy.col1 (Y), e_yz.col0 (Y), e_xz.col0 (X)
        // — the three columns the count kernel binary-searches on.
        let xy_col1 = column_u32(e_xy, 1)?;
        let yz_col0 = column_u32(e_yz, 0)?;
        let xz_col0 = column_u32(e_xz, 0)?;

        // Pre-allocate the 3 × 64 histogram buffer BEFORE the
        // recorder. Total = 768 bytes — under the
        // `dtoh_small_metadata_untracked` 4 KB cap.
        let hist_len = (3 * WCOJ_SKEW_NUM_BUCKETS) as usize;
        let mut hist_buf = self.memory.alloc::<u32>(hist_len)?;

        // Block / grid sizing: one row per thread, 256 threads
        // per block. Grid is `3 * blocks_per_col` so each third
        // of the grid handles one column.
        let max_n = n_xy.max(n_yz).max(n_xz);
        let blocks_per_col = (max_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let total_blocks = blocks_per_col.saturating_mul(3);

        // Resolve the kernel handle BEFORE queueing any
        // launch-stream work. If the kernel module isn't loaded,
        // failing here drops `hist_buf` cleanly — no queued
        // memset/launch is in flight.
        let device = self.device.inner();
        let kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_SKEW_HISTOGRAM_U32)
            .ok_or_else(|| {
                XlogError::Kernel("skew_score: histogram_u32 kernel not found".to_string())
            })?;

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e_xy.num_rows_device());
        rec.read(e_yz.num_rows_device());
        rec.read(e_xz.num_rows_device());
        rec.read_column(e_xy.column(1).expect("e_xy.col1"));
        rec.read_column(e_yz.column(0).expect("e_yz.col0"));
        rec.read_column(e_xz.column(0).expect("e_xz.col0"));
        rec.write(&hist_buf);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("skew_score: preflight failed: {e}")))?;

        // From here on, any error path before `rec.commit(runtime)`
        // succeeds must drain the launch stream before returning,
        // because `hist_buf` is about to drop and the runtime
        // dealloc would otherwise race the queued memset/kernel.
        // We capture the in-flight closure in a local helper and
        // sync on the failure branch.
        let queued_result: Result<()> = (|| {
            // Step 1: zero the histogram on launch_stream. CUDA does
            // not provide a global barrier inside one regular kernel,
            // so an in-kernel "zero pass + atomic merge" can race;
            // the on-stream memset is the correctness-preserving
            // sequencing primitive. Bytes = 192 * 4 = 768.
            let hist_bytes = hist_len * std::mem::size_of::<u32>();
            unsafe {
                let res = sys::cuMemsetD8Async(
                    *hist_buf.device_ptr(),
                    0,
                    hist_bytes,
                    cu_stream.cu_stream(),
                );
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "skew_score: cuMemsetD8Async failed: {res:?}"
                    )));
                }
            }

            // Step 2: combined histogram kernel.
            // SAFETY: 8-arg signature
            //   wcoj_triangle_skew_histogram_u32(
            //     const u32* xy_col1, u32 n_xy,
            //     const u32* yz_col0, u32 n_yz,
            //     const u32* xz_col0, u32 n_xz,
            //     u32 blocks_per_col,
            //     u32* histograms)
            unsafe {
                kernel
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (total_blocks, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            xy_col1,
                            n_xy,
                            yz_col0,
                            n_yz,
                            xz_col0,
                            n_xz,
                            blocks_per_col,
                            &mut hist_buf,
                        ),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("skew_score: histogram launch failed: {e}"))
                    })?;
            }
            Ok(())
        })();

        // If the queued-work helper failed, best-effort sync the
        // stream so any partially-issued memset / launch drains
        // before `hist_buf` goes out of scope. Sync errors are
        // swallowed — the original error is more informative for
        // the caller, and Ok(None) wraps either way.
        if let Err(e) = queued_result {
            let _ = cu_stream.synchronize();
            return Err(e);
        }

        // commit() can also fail after queued work landed. Same
        // sync-before-return discipline.
        if let Err(e) = rec.commit(runtime) {
            let _ = cu_stream.synchronize();
            return Err(XlogError::Kernel(format!("skew_score: commit failed: {e}")));
        }

        // Step 3: sync before D2H — host must see the histogram
        // kernel's writes before reading them.
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("skew_score: sync failed: {e}")))?;

        // Step 4: D2H the 192-element histogram via the
        // metadata-only chokepoint.
        let host_hist = self
            .dtoh_small_metadata_untracked::<u32>(&hist_buf, hist_len)
            .map_err(|e| XlogError::Kernel(format!("skew_score: dtoh failed: {e}")))?;

        Ok(score_from_histograms(&host_hist, n_xy, n_yz, n_xz))
    }

    fn try_compute_skew_score_u64(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<f64> {
        let runtime = self
            .memory()
            .runtime()
            .ok_or_else(|| XlogError::Kernel("skew_score: no runtime".to_string()))?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| XlogError::Kernel("skew_score: stream resolve failed".to_string()))?;

        for (label, buf) in [("e_xy", e_xy), ("e_yz", e_yz), ("e_xz", e_xz)] {
            if buf.arity() != 2 {
                return Err(XlogError::Kernel(format!(
                    "skew_score: {label} must be 2-column"
                )));
            }
            for col_idx in 0..2 {
                let ty = buf
                    .schema
                    .column_type(col_idx)
                    .ok_or_else(|| XlogError::Kernel("skew_score: type missing".to_string()))?;
                if !matches!(ty, ScalarType::U64) {
                    return Err(XlogError::Kernel(format!(
                        "skew_score: {label}.{col_idx} not U64"
                    )));
                }
            }
        }

        let n_xy = self.logical_row_count_u32(e_xy)?;
        let n_yz = self.logical_row_count_u32(e_yz)?;
        let n_xz = self.logical_row_count_u32(e_xz)?;
        if n_xy == 0 || n_yz == 0 || n_xz == 0 {
            return Err(XlogError::Kernel("skew_score: zero rows".to_string()));
        }

        let xy_col1 = column_u64(e_xy, 1)?;
        let yz_col0 = column_u64(e_yz, 0)?;
        let xz_col0 = column_u64(e_xz, 0)?;

        let hist_len = (3 * WCOJ_SKEW_NUM_BUCKETS) as usize;
        let mut hist_buf = self.memory.alloc::<u32>(hist_len)?;

        let max_n = n_xy.max(n_yz).max(n_xz);
        let blocks_per_col = (max_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let total_blocks = blocks_per_col.saturating_mul(3);

        // Resolve kernel BEFORE queuing any launch-stream work
        // (mirror of the u32 path's failure-safety hygiene).
        let device = self.device.inner();
        let kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_SKEW_HISTOGRAM_U64)
            .ok_or_else(|| {
                XlogError::Kernel("skew_score: histogram_u64 kernel not found".to_string())
            })?;

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e_xy.num_rows_device());
        rec.read(e_yz.num_rows_device());
        rec.read(e_xz.num_rows_device());
        rec.read_column(e_xy.column(1).expect("e_xy.col1"));
        rec.read_column(e_yz.column(0).expect("e_yz.col0"));
        rec.read_column(e_xz.column(0).expect("e_xz.col0"));
        rec.write(&hist_buf);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("skew_score: preflight failed: {e}")))?;

        // Queued-work block. Any failure here must drain the
        // launch stream before `hist_buf` drops, otherwise
        // runtime dealloc may race in-flight memset/launch.
        let queued_result: Result<()> = (|| {
            let hist_bytes = hist_len * std::mem::size_of::<u32>();
            unsafe {
                let res = sys::cuMemsetD8Async(
                    *hist_buf.device_ptr(),
                    0,
                    hist_bytes,
                    cu_stream.cu_stream(),
                );
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "skew_score: cuMemsetD8Async failed: {res:?}"
                    )));
                }
            }

            unsafe {
                kernel
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (total_blocks, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            xy_col1,
                            n_xy,
                            yz_col0,
                            n_yz,
                            xz_col0,
                            n_xz,
                            blocks_per_col,
                            &mut hist_buf,
                        ),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("skew_score: histogram launch failed: {e}"))
                    })?;
            }
            Ok(())
        })();

        if let Err(e) = queued_result {
            let _ = cu_stream.synchronize();
            return Err(e);
        }

        if let Err(e) = rec.commit(runtime) {
            let _ = cu_stream.synchronize();
            return Err(XlogError::Kernel(format!("skew_score: commit failed: {e}")));
        }

        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("skew_score: sync failed: {e}")))?;

        let host_hist = self
            .dtoh_small_metadata_untracked::<u32>(&hist_buf, hist_len)
            .map_err(|e| XlogError::Kernel(format!("skew_score: dtoh failed: {e}")))?;

        Ok(score_from_histograms(&host_hist, n_xy, n_yz, n_xz))
    }
}

/// Reduce the 192-element histogram into a scalar skew score.
/// Returns `max(max_bucket_i / row_count_i)` over the three
/// columns. Bucket layout matches the kernel:
///   `host_hist[col_idx * 64 .. col_idx * 64 + 64]`
fn score_from_histograms(host_hist: &[u32], n_xy: u32, n_yz: u32, n_xz: u32) -> f64 {
    let buckets = WCOJ_SKEW_NUM_BUCKETS as usize;
    let column_score = |col_idx: usize, n: u32| -> f64 {
        if n == 0 {
            return 0.0;
        }
        let start = col_idx * buckets;
        let end = start + buckets;
        let max = host_hist[start..end].iter().copied().max().unwrap_or(0);
        max as f64 / n as f64
    };
    let s0 = column_score(0, n_xy);
    let s1 = column_score(1, n_yz);
    let s2 = column_score(2, n_xz);
    s0.max(s1).max(s2)
}

/// Borrow a 4-byte-key column out of a 2-column [`CudaBuffer`] as
/// a typed `TrackedCudaSlice<u32>` reference for kernel binding.
///
/// Each `CudaColumn` is stored as raw bytes (`TrackedCudaSlice<u8>`
/// for owned columns); the kernel signature expects `u32*`. This
/// helper does the reinterpretation behind a single chokepoint
/// so the unsafety is colocated. Accepts U32 *or* Symbol columns
/// — both share the same 4-byte physical layout, and the kernel
/// performs equality joins on bit-identical values regardless of
/// the logical scalar type. The cross-relation type-compatibility
/// check (so symbol IDs only join symbol IDs, not raw u32s with
/// the same bit pattern) is enforced by the planner upstream.
fn column_u32<'a>(buf: &'a CudaBuffer, col_idx: usize) -> Result<&'a TrackedCudaSlice<u32>> {
    let col = buf.column(col_idx).ok_or_else(|| {
        XlogError::Kernel(format!(
            "wcoj_triangle_u32_recorded: column {} not found",
            col_idx
        ))
    })?;
    match col {
        CudaColumn::Owned(slice) => {
            // SAFETY: The buffer was validated to have either
            // ScalarType::U32 or ScalarType::Symbol at this
            // column. Both have `size_bytes() == 4`, so the
            // byte slice's len is a multiple of 4 and aligned.
            // The transmute below changes the element type
            // from u8 to u32 without touching the underlying
            // raw_ptr / runtime_block, both of which are
            // agnostic to the element type. The returned
            // reference's lifetime is tied to `buf`, so the
            // original column outlives every use.
            unsafe { Ok(&*(slice as *const TrackedCudaSlice<u8> as *const TrackedCudaSlice<u32>)) }
        }
        _ => Err(XlogError::Kernel(
            "wcoj_triangle_u32_recorded: 4-byte-key columns must be Owned-variant CudaColumns"
                .to_string(),
        )),
    }
}

/// `&mut` reinterpret of an owned u8 slice (e.g. a freshly-
/// allocated output column) as a u32 slice for kernel binding.
/// Mirrors [`column_u32`] but for the writable path.
unsafe fn reinterpret_u8_as_u32(slice: &mut TrackedCudaSlice<u8>) -> &mut TrackedCudaSlice<u32> {
    &mut *(slice as *mut TrackedCudaSlice<u8> as *mut TrackedCudaSlice<u32>)
}

/// Borrow an 8-byte-key column out of a 2-column [`CudaBuffer`]
/// as a typed `TrackedCudaSlice<u64>` reference for kernel
/// binding. Mirrors [`column_u32`] but for U64 keys; the caller
/// (`wcoj_triangle_u64_recorded`) has already validated that the
/// schema is U64.
fn column_u64<'a>(buf: &'a CudaBuffer, col_idx: usize) -> Result<&'a TrackedCudaSlice<u64>> {
    let col = buf.column(col_idx).ok_or_else(|| {
        XlogError::Kernel(format!(
            "wcoj_triangle_u64_recorded: column {} not found",
            col_idx
        ))
    })?;
    match col {
        CudaColumn::Owned(slice) => {
            // SAFETY: Schema validated to be ScalarType::U64
            // (size_bytes == 8). CUDA allocations are 256-byte
            // aligned so u64 alignment is satisfied. The
            // transmute changes element type from u8 to u64
            // without touching raw_ptr / runtime_block. The
            // returned reference's lifetime is tied to `buf`.
            unsafe { Ok(&*(slice as *const TrackedCudaSlice<u8> as *const TrackedCudaSlice<u64>)) }
        }
        _ => Err(XlogError::Kernel(
            "wcoj_triangle_u64_recorded: 8-byte-key columns must be Owned-variant CudaColumns"
                .to_string(),
        )),
    }
}

/// `&mut` reinterpret of an owned u8 slice as a u64 slice for
/// kernel binding. Mirrors [`reinterpret_u8_as_u32`].
unsafe fn reinterpret_u8_as_u64(slice: &mut TrackedCudaSlice<u8>) -> &mut TrackedCudaSlice<u64> {
    &mut *(slice as *mut TrackedCudaSlice<u8> as *mut TrackedCudaSlice<u64>)
}
