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

// ===============================================================
// Per-phase CUDA-event timing (feature `wcoj-phase-timing` only).
//
// Records 6 events on the launch_stream around the 4 GPU phases
// of `wcoj_triangle_*_recorded`:
//
//   e0 — start of `wcoj_triangle_count` kernel
//   e1 — end of count + dtod copy (= start of scan)
//   e2 — end of scan (= start of `wcoj_compute_total` kernel)
//   e3 — end of `wcoj_compute_total` kernel
//          [host work: sync, dtoh_scalar, alloc, H2D, recorder]
//   e4 — start of `wcoj_triangle_materialize` kernel
//   e5 — end of materialize kernel
//
// Phase deltas (`f32` ms via `CudaEvent::elapsed_ms`):
//   count_ms       = elapsed(e0, e1)
//   scan_ms        = elapsed(e1, e2)
//   total_ms       = elapsed(e2, e3)
//   materialize_ms = elapsed(e4, e5)
//
// The host-side work between e3 and e4 is intentionally NOT
// captured by these events — the caller's residual bucket
// (wall-clock minus measured GPU phases) accounts for it.
//
// Under feature-off, `PhaseTimer` is a unit type with no-op
// methods that the optimizer eliminates entirely; the
// `wcoj_triangle_*_recorded` hot path is unchanged.

#[cfg(feature = "wcoj-phase-timing")]
struct PhaseTimer {
    events: [cudarc::driver::CudaEvent; 6],
}

#[cfg(feature = "wcoj-phase-timing")]
impl PhaseTimer {
    fn new(stream: &cudarc::driver::CudaStream) -> Result<Self> {
        use cudarc::driver::sys::CUevent_flags;
        let ctx = stream.context();
        let mk = || {
            ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| XlogError::Kernel(format!("phase event create: {e}")))
        };
        Ok(Self {
            events: [mk()?, mk()?, mk()?, mk()?, mk()?, mk()?],
        })
    }

    fn record(&self, idx: usize, stream: &cudarc::driver::CudaStream) -> Result<()> {
        self.events[idx]
            .record(stream)
            .map_err(|e| XlogError::Kernel(format!("phase event record {idx}: {e}")))
    }

    fn record_after_queued_work(
        &self,
        idx: usize,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<()> {
        if let Err(e) = self.events[idx].record(stream) {
            // The event itself is diagnostic, but at this point
            // CUDA work has already been queued against buffers
            // owned by this call. Drain before returning so an
            // event-record failure cannot turn into a diagnostic
            // feature-only use-after-free.
            let _ = stream.synchronize();
            return Err(XlogError::Kernel(format!("phase event record {idx}: {e}")));
        }
        Ok(())
    }

    fn finish(self) -> Result<crate::wcoj_phase_timing::WcojTrianglePhaseTiming> {
        let elapsed = |a: usize, b: usize| -> Result<f32> {
            self.events[a]
                .elapsed_ms(&self.events[b])
                .map_err(|e| XlogError::Kernel(format!("phase elapsed_ms({a}, {b}): {e}")))
        };
        Ok(crate::wcoj_phase_timing::WcojTrianglePhaseTiming {
            count_ms: elapsed(0, 1)?,
            scan_ms: elapsed(1, 2)?,
            total_ms: elapsed(2, 3)?,
            materialize_ms: elapsed(4, 5)?,
        })
    }
}

/// No-op stub for production builds (feature off).
#[cfg(not(feature = "wcoj-phase-timing"))]
struct PhaseTimer;

#[cfg(not(feature = "wcoj-phase-timing"))]
impl PhaseTimer {
    #[inline(always)]
    fn new(_stream: &cudarc::driver::CudaStream) -> Result<Self> {
        Ok(Self)
    }
    #[inline(always)]
    fn record(&self, _idx: usize, _stream: &cudarc::driver::CudaStream) -> Result<()> {
        Ok(())
    }
    #[inline(always)]
    fn record_after_queued_work(
        &self,
        _idx: usize,
        _stream: &cudarc::driver::CudaStream,
    ) -> Result<()> {
        Ok(())
    }
}

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
    /// Fast-path: if the input is already strictly lex-sorted and
    /// full-row unique, a recorded checker proves that property and
    /// the method returns a recorded device-side clone. Otherwise it
    /// falls back to [`Self::dedup_full_row_recorded`], which invokes
    /// [`Self::sort_recorded`] (typed multi-column radix sort on
    /// `(col0, col1)`) followed by an on-stream
    /// `mark_unique_full_row_bytewise` mask + counted compaction.
    /// Both paths are launch-recorder disciplined and preserve the
    /// sorted+deduped output contract.
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
        // Layout fast-path: if the input is already strictly
        // lex-sorted AND full-row unique, we can skip the
        // (expensive) sort + mark-unique + compact pipeline
        // and emit a recorded device-side clone. The phase
        // report (docs/evidence/2026-05-01-wcoj-bench-baseline/phase-timing-report.md)
        // measured layout at 91-97% of WCOJ adaptive dispatch
        // wall clock; this branch is the targeted overhead
        // reduction.
        match self.try_wcoj_layout_fast_path_u32(input, launch_stream) {
            Ok(Some(out)) => {
                self.record_wcoj_layout_fast_path_hit();
                return Ok(out);
            }
            Ok(None) => {
                // Empty input handled by dedup_full_row_recorded
                // (which has its own n==0 short-circuit returning
                // create_empty_buffer). Fall through.
            }
            Err(_) => {
                // Checker failed unexpectedly. Fall through to
                // the safe path; correctness is preserved.
            }
        }
        // Fall through: the input wasn't proven sorted+unique.
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

        // Phase-timing scaffolding (no-op when feature off).
        let phase_timer = PhaseTimer::new(&cu_stream)?;
        phase_timer.record(0, &cu_stream)?;

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
        phase_timer.record_after_queued_work(1, &cu_stream)?;

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
        phase_timer.record_after_queued_work(2, &cu_stream)?;

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
        phase_timer.record_after_queued_work(3, &cu_stream)?;

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
        phase_timer.record_after_queued_work(4, &cu_stream)?;
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
        phase_timer.record_after_queued_work(5, &cu_stream)?;

        rec_mat.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_triangle_u32_recorded: materialize commit failed: {}",
                e
            ))
        })?;

        // Sync once more (the rec_mat commit + recorder events
        // chain don't guarantee the materialize event has been
        // observed yet on the host side). Then capture phase
        // timings into the provider's diagnostic slot — no-op
        // when feature is off.
        #[cfg(feature = "wcoj-phase-timing")]
        {
            cu_stream.synchronize().map_err(|e| {
                XlogError::Kernel(format!(
                    "wcoj_triangle_u32_recorded: stream sync after materialize failed: {}",
                    e
                ))
            })?;
            let timing = phase_timer.finish()?;
            self.put_wcoj_triangle_phase_timing(timing);
        }
        #[cfg(not(feature = "wcoj-phase-timing"))]
        let _ = phase_timer; // suppress unused warning when feature off

        let columns: Vec<CudaColumn> = vec![out_x.into(), out_y.into(), out_z.into()];
        Ok(CudaBuffer::from_columns_with_host_count(
            columns,
            total_rows as u64,
            out_d_num_rows,
            out_schema,
            total_rows,
        ))
    }

    /// Evaluate `cyc4(W, X, Y, Z) :- e1(W,X), e2(X,Y), e3(Y,Z), e4(Z,W)`
    /// on already-sorted, already-deduped binary 4-byte-key
    /// relations. Structural mirror of [`Self::wcoj_triangle_u32_recorded`]
    /// for the 4-cycle case; see that entry's contract and the
    /// module-level docs for the shared two-phase recorder
    /// discipline.
    ///
    /// Each column may be [`ScalarType::U32`] or
    /// [`ScalarType::Symbol`] — both share the same 4-byte
    /// physical layout, so the kernel reads the bits unchanged.
    /// Cross-relation type compatibility (e.g., that X is the
    /// same type in `e1.col1` and `e2.col0`) is the planner's
    /// responsibility upstream; this entry only enforces width.
    ///
    /// The 4-cycle slot order is `[e1(W,X), e2(X,Y), e3(Y,Z), e4(Z,W)]`.
    /// The output schema preserves per-head-position scalar types
    /// from the inputs:
    ///   * `out.col0` = `e1.col0` type (W)
    ///   * `out.col1` = `e1.col1` type (X)
    ///   * `out.col2` = `e2.col1` type (Y)
    ///   * `out.col3` = `e3.col1` type (Z)
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime
    ///   (`with_runtime` is required), the launch stream does
    ///   not resolve, an input is not 2-column with U32/Symbol
    ///   columns, or any kernel launch fails.
    pub fn wcoj_4cycle_u32_recorded(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "wcoj_4cycle_u32_recorded requires a runtime-backed \
                 GpuMemoryManager (constructed via with_runtime)"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_4cycle_u32_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        // ---------------------------------------------------------
        // Validate input shape: every relation must be 2-column
        // with U32 or Symbol columns (both 4-byte physical).
        // Cross-relation type compatibility is enforced upstream
        // by the planner / hypergraph type inference.
        // ---------------------------------------------------------
        for (label, buf) in [("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)] {
            if buf.arity() != 2 {
                return Err(XlogError::Kernel(format!(
                    "wcoj_4cycle_u32_recorded: {} must be 2-column, got arity {}",
                    label,
                    buf.arity()
                )));
            }
            for col_idx in 0..2 {
                let ty = buf.schema.column_type(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!(
                        "wcoj_4cycle_u32_recorded: {} column {} type missing",
                        label, col_idx
                    ))
                })?;
                if !matches!(ty, ScalarType::U32 | ScalarType::Symbol) {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_4cycle_u32_recorded: {} column {} must be U32 or Symbol, got {:?}",
                        label, col_idx, ty
                    )));
                }
            }
        }

        // Output schema mirrors the inputs' per-head-position
        // scalar types: out.col0 = W = e1.col0; out.col1 = X =
        // e1.col1; out.col2 = Y = e2.col1; out.col3 = Z = e3.col1.
        // Cross-relation type compatibility (X same in e2.col0,
        // Y same in e3.col0, Z same in e4.col0, W same in e4.col1)
        // is the planner's job; we trust it here.
        let w_type = e1.schema.column_type(0).expect("validated above");
        let x_type = e1.schema.column_type(1).expect("validated above");
        let y_type = e2.schema.column_type(1).expect("validated above");
        let z_type = e3.schema.column_type(1).expect("validated above");
        let out_schema = Schema::new(vec![
            ("col0".to_string(), w_type),
            ("col1".to_string(), x_type),
            ("col2".to_string(), y_type),
            ("col3".to_string(), z_type),
        ]);

        // See `wcoj_triangle_u32_recorded` for why we use the
        // logical row count (not `num_rows()`) and route through
        // `dtoh_scalar_untracked` on cache miss.
        let n_e1 = self.logical_row_count_u32(e1)?;
        let n_e2 = self.logical_row_count_u32(e2)?;
        let n_e3 = self.logical_row_count_u32(e3)?;
        let n_e4 = self.logical_row_count_u32(e4)?;

        // Empty input on any side → empty result, no kernel launches.
        if n_e1 == 0 || n_e2 == 0 || n_e3 == 0 || n_e4 == 0 {
            return self.create_empty_buffer(out_schema);
        }

        // ---------------------------------------------------------
        // Phase 1 (count + device scan + total). Pre-allocate
        // every fresh buffer BEFORE the recorder so kernel `&mut`
        // borrows are unaffected by recorder lifetimes. Grid is
        // dispatched over `n_e1` (one thread per row of e1) — the
        // count/materialize kernels iterate the e2 / e3 / e4
        // chain inside each thread.
        // ---------------------------------------------------------
        let mut count_buf = self.memory.alloc::<u32>(n_e1 as usize)?;
        let mut offsets_buf = self.memory.alloc::<u32>(n_e1 as usize)?;
        let d_total = self.memory.alloc::<u32>(1)?;

        let e1_col0 = column_u32(e1, 0)?;
        let e1_col1 = column_u32(e1, 1)?;
        let e2_col0 = column_u32(e2, 0)?;
        let e2_col1 = column_u32(e2, 1)?;
        let e3_col0 = column_u32(e3, 0)?;
        let e3_col1 = column_u32(e3, 1)?;
        let e4_col0 = column_u32(e4, 0)?;
        let e4_col1 = column_u32(e4, 1)?;

        let mut rec_count = LaunchRecorder::new_strict(launch_stream);
        rec_count.read(e1.num_rows_device());
        rec_count.read(e2.num_rows_device());
        rec_count.read(e3.num_rows_device());
        rec_count.read(e4.num_rows_device());
        rec_count.read_column(e1.column(0).expect("e1.col0"));
        rec_count.read_column(e1.column(1).expect("e1.col1"));
        rec_count.read_column(e2.column(0).expect("e2.col0"));
        rec_count.read_column(e2.column(1).expect("e2.col1"));
        rec_count.read_column(e3.column(0).expect("e3.col0"));
        rec_count.read_column(e3.column(1).expect("e3.col1"));
        rec_count.read_column(e4.column(0).expect("e4.col0"));
        rec_count.read_column(e4.column(1).expect("e4.col1"));
        rec_count.write(&count_buf);
        rec_count.write(&offsets_buf);
        rec_count.write(&d_total);
        rec_count.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_4cycle_u32_recorded: count preflight failed: {}",
                e
            ))
        })?;

        let device = self.device.inner();
        let count_kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_4CYCLE_COUNT)
            .ok_or_else(|| XlogError::Kernel("wcoj_4cycle_count kernel not found".to_string()))?;
        let grid = (n_e1 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Phase-timing scaffolding (no-op when feature off).
        let phase_timer = PhaseTimer::new(&cu_stream)?;
        phase_timer.record(0, &cu_stream)?;

        // SAFETY:
        //   wcoj_4cycle_count(
        //     const u32* e1_col0, const u32* e1_col1, u32 n_e1,
        //     const u32* e2_col0, const u32* e2_col1, u32 n_e2,
        //     const u32* e3_col0, const u32* e3_col1, u32 n_e3,
        //     const u32* e4_col0, const u32* e4_col1, u32 n_e4,
        //     u32* out_counts)
        // 13 args exceeds the LaunchAsync tuple bound (12), so we
        // use the raw param-vec form. All buffers are runtime-backed
        // device memory; the launch is queued on launch_stream;
        // preflight has already verified cross-stream tracking.
        let count_config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                e1_col0.as_kernel_param(),
                e1_col1.as_kernel_param(),
                n_e1.as_kernel_param(),
                e2_col0.as_kernel_param(),
                e2_col1.as_kernel_param(),
                n_e2.as_kernel_param(),
                e3_col0.as_kernel_param(),
                e3_col1.as_kernel_param(),
                n_e3.as_kernel_param(),
                e4_col0.as_kernel_param(),
                e4_col1.as_kernel_param(),
                n_e4.as_kernel_param(),
                (&mut count_buf).as_kernel_param(),
            ];
            count_kernel
                .clone()
                .launch_on_stream(&cu_stream, count_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("wcoj_4cycle_count launch failed: {}", e))
                })?;
        }

        // dtod-async copy `count_buf → offsets_buf` on launch_stream.
        // The scan helper runs in place; we keep `count_buf` intact
        // so `wcoj_compute_total` can read counts[n-1] directly.
        // SAFETY: both buffers are runtime-backed u32 slices of
        // length `n_e1`, both sized `n_e1 * 4` bytes.
        let bytes_count = (n_e1 as usize) * std::mem::size_of::<u32>();
        unsafe {
            let res = sys::cuMemcpyDtoDAsync_v2(
                *offsets_buf.device_ptr(),
                *count_buf.device_ptr(),
                bytes_count,
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "wcoj_4cycle_u32_recorded: dtod count_buf → offsets_buf failed: {:?}",
                    res
                )));
            }
        }
        phase_timer.record_after_queued_work(1, &cu_stream)?;

        // Device-side exclusive prefix-sum on `offsets_buf` over
        // `[0..n_e1)`. Recorded against `launch_stream` via the
        // supplied `cu_stream` / `runtime` arguments — no host
        // involvement.
        self.multiblock_scan_u32_inplace_on_stream(
            &mut offsets_buf,
            n_e1,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        phase_timer.record_after_queued_work(2, &cu_stream)?;

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
                    (&count_buf, &offsets_buf, n_e1, &d_total),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("wcoj_compute_total launch failed: {}", e))
                })?;
        }
        phase_timer.record_after_queued_work(3, &cu_stream)?;

        rec_count.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_4cycle_u32_recorded: count+scan+total commit failed: {}",
                e
            ))
        })?;

        // Sync + sanctioned scalar D2H of the total.
        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_4cycle_u32_recorded: stream sync after total failed: {}",
                e
            ))
        })?;
        let total_rows = self
            .dtoh_scalar_untracked::<u32>(&d_total, 0)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "wcoj_4cycle_u32_recorded: read d_total failed: {}",
                    e
                ))
            })?;

        if total_rows == 0 {
            return self.create_empty_buffer(out_schema);
        }

        // ---------------------------------------------------------
        // Phase 2 (materialize). Allocate output (4 columns now —
        // W, X, Y, Z), build recorder, launch materialize on
        // launch_stream.
        // ---------------------------------------------------------
        let bytes_per_col = (total_rows as usize) * std::mem::size_of::<u32>();
        let mut out_w = self.memory.alloc::<u8>(bytes_per_col)?;
        let mut out_x = self.memory.alloc::<u8>(bytes_per_col)?;
        let mut out_y = self.memory.alloc::<u8>(bytes_per_col)?;
        let mut out_z = self.memory.alloc::<u8>(bytes_per_col)?;
        let out_d_num_rows = self.memory.alloc::<u32>(1)?;

        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        rec_mat.read(e1.num_rows_device());
        rec_mat.read(e2.num_rows_device());
        rec_mat.read(e3.num_rows_device());
        rec_mat.read(e4.num_rows_device());
        rec_mat.read_column(e1.column(0).expect("e1.col0"));
        rec_mat.read_column(e1.column(1).expect("e1.col1"));
        rec_mat.read_column(e2.column(0).expect("e2.col0"));
        rec_mat.read_column(e2.column(1).expect("e2.col1"));
        rec_mat.read_column(e3.column(0).expect("e3.col0"));
        rec_mat.read_column(e3.column(1).expect("e3.col1"));
        rec_mat.read_column(e4.column(0).expect("e4.col0"));
        rec_mat.read_column(e4.column(1).expect("e4.col1"));
        rec_mat.read(&offsets_buf);
        rec_mat.write(&out_w);
        rec_mat.write(&out_x);
        rec_mat.write(&out_y);
        rec_mat.write(&out_z);
        rec_mat.write(&out_d_num_rows);
        rec_mat.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_4cycle_u32_recorded: materialize preflight failed: {}",
                e
            ))
        })?;

        // H2D the output row count on launch_stream — the returned
        // CudaBuffer's d_num_rows must reflect the host-known total.
        // 4-byte H2D, the only host→device transfer in the path.
        unsafe {
            let res = sys::cuMemcpyHtoDAsync_v2(
                *out_d_num_rows.device_ptr(),
                &total_rows as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "wcoj_4cycle_u32_recorded: H2D out_d_num_rows failed: {:?}",
                    res
                )));
            }
        }

        let materialize_kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_4CYCLE_MATERIALIZE)
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_4cycle_materialize kernel not found".to_string())
            })?;

        // Reinterpret the per-byte output slices as u32 slices for
        // the kernel's typed signature.
        let out_w_u32: &mut TrackedCudaSlice<u32> = unsafe { reinterpret_u8_as_u32(&mut out_w) };
        let out_x_u32: &mut TrackedCudaSlice<u32> = unsafe { reinterpret_u8_as_u32(&mut out_x) };
        let out_y_u32: &mut TrackedCudaSlice<u32> = unsafe { reinterpret_u8_as_u32(&mut out_y) };
        let out_z_u32: &mut TrackedCudaSlice<u32> = unsafe { reinterpret_u8_as_u32(&mut out_z) };

        // SAFETY:
        //   wcoj_4cycle_materialize(
        //     const u32* e1_col0, const u32* e1_col1, u32 n_e1,
        //     const u32* e2_col0, const u32* e2_col1, u32 n_e2,
        //     const u32* e3_col0, const u32* e3_col1, u32 n_e3,
        //     const u32* e4_col0, const u32* e4_col1, u32 n_e4,
        //     const u32* out_offsets, u32 total_rows,
        //     u32* out_w, u32* out_x, u32* out_y, u32* out_z)
        // 17 args exceeds the LaunchAsync tuple bound (13), so we
        // use the raw param-vec form.
        let mat_config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        phase_timer.record_after_queued_work(4, &cu_stream)?;
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                e1_col0.as_kernel_param(),
                e1_col1.as_kernel_param(),
                n_e1.as_kernel_param(),
                e2_col0.as_kernel_param(),
                e2_col1.as_kernel_param(),
                n_e2.as_kernel_param(),
                e3_col0.as_kernel_param(),
                e3_col1.as_kernel_param(),
                n_e3.as_kernel_param(),
                e4_col0.as_kernel_param(),
                e4_col1.as_kernel_param(),
                n_e4.as_kernel_param(),
                (&offsets_buf).as_kernel_param(),
                total_rows.as_kernel_param(),
                out_w_u32.as_kernel_param(),
                out_x_u32.as_kernel_param(),
                out_y_u32.as_kernel_param(),
                out_z_u32.as_kernel_param(),
            ];
            materialize_kernel
                .clone()
                .launch_on_stream(&cu_stream, mat_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("wcoj_4cycle_materialize launch failed: {}", e))
                })?;
        }
        phase_timer.record_after_queued_work(5, &cu_stream)?;

        rec_mat.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_4cycle_u32_recorded: materialize commit failed: {}",
                e
            ))
        })?;

        // Sync once more then capture phase timings into the
        // provider's diagnostic slot — no-op when feature is off.
        // The 4-cycle pipeline shape matches triangle's (count,
        // dtod-copy, scan, compute_total, materialize), so we
        // reuse the triangle phase-timing slot.
        #[cfg(feature = "wcoj-phase-timing")]
        {
            cu_stream.synchronize().map_err(|e| {
                XlogError::Kernel(format!(
                    "wcoj_4cycle_u32_recorded: stream sync after materialize failed: {}",
                    e
                ))
            })?;
            let timing = phase_timer.finish()?;
            self.put_wcoj_triangle_phase_timing(timing);
        }
        #[cfg(not(feature = "wcoj-phase-timing"))]
        let _ = phase_timer; // suppress unused warning when feature off

        let columns: Vec<CudaColumn> = vec![out_w.into(), out_x.into(), out_y.into(), out_z.into()];
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
    /// already sorted+unique inputs take the recorded fast-path clone;
    /// other inputs fall back to [`Self::dedup_full_row_recorded`],
    /// whose U64 `sort_recorded` path ports the legacy `sort()` hi/lo
    /// radix-pass strategy into recorded launch discipline.
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
        // Fast-path: see u32 entry for rationale + measurement
        // basis. Strictly lex-sorted AND full-row unique inputs
        // skip dedup_full_row_recorded.
        match self.try_wcoj_layout_fast_path_u64(input, launch_stream) {
            Ok(Some(out)) => {
                self.record_wcoj_layout_fast_path_hit();
                return Ok(out);
            }
            Ok(None) | Err(_) => {}
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

        // Phase-timing scaffolding (no-op when feature off).
        let phase_timer = PhaseTimer::new(&cu_stream)?;
        phase_timer.record(0, &cu_stream)?;

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
        phase_timer.record_after_queued_work(1, &cu_stream)?;

        // Device-side exclusive prefix-sum on offsets_buf — u32 plumbing
        // unchanged from the u32 path.
        self.multiblock_scan_u32_inplace_on_stream(
            &mut offsets_buf,
            n_xy,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        phase_timer.record_after_queued_work(2, &cu_stream)?;

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
        phase_timer.record_after_queued_work(3, &cu_stream)?;

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
        phase_timer.record_after_queued_work(4, &cu_stream)?;
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
        phase_timer.record_after_queued_work(5, &cu_stream)?;

        rec_mat.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_triangle_u64_recorded: materialize commit failed: {}",
                e
            ))
        })?;

        // Sync + capture phase timings (no-op when feature off).
        #[cfg(feature = "wcoj-phase-timing")]
        {
            cu_stream.synchronize().map_err(|e| {
                XlogError::Kernel(format!(
                    "wcoj_triangle_u64_recorded: stream sync after materialize failed: {}",
                    e
                ))
            })?;
            let timing = phase_timer.finish()?;
            self.put_wcoj_triangle_phase_timing(timing);
        }
        #[cfg(not(feature = "wcoj-phase-timing"))]
        let _ = phase_timer;

        let columns: Vec<CudaColumn> = vec![out_x.into(), out_y.into(), out_z.into()];
        Ok(CudaBuffer::from_columns_with_host_count(
            columns,
            total_rows as u64,
            out_d_num_rows,
            out_schema,
            total_rows,
        ))
    }

    /// Evaluate
    /// `cycle4(W, X, Y, Z) :- e1(W,X), e2(X,Y), e3(Y,Z), e4(Z,W)`
    /// on already-sorted, already-deduped binary U64 relations.
    /// Mirrors [`Self::wcoj_4cycle_u32_recorded`]'s contract; the
    /// only differences are the 8-byte join-key reads/writes and
    /// the U64-specific count/materialize kernels. Counters and
    /// the total reducer remain u32 (bounded by the upstream host-
    /// side row-count guard).
    ///
    /// 4-cycle slot order:
    /// `[e1(W,X), e2(X,Y), e3(Y,Z), e4(Z,W)]`.
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime, the
    ///   launch stream does not resolve, an input is not 2-column
    ///   with U64 columns, or any kernel launch fails.
    pub fn wcoj_4cycle_u64_recorded(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "wcoj_4cycle_u64_recorded requires a runtime-backed \
                 GpuMemoryManager (constructed via with_runtime)"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_4cycle_u64_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        // Validate: every relation 2-column U64.
        for (label, buf) in [("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)] {
            if buf.arity() != 2 {
                return Err(XlogError::Kernel(format!(
                    "wcoj_4cycle_u64_recorded: {} must be 2-column, got arity {}",
                    label,
                    buf.arity()
                )));
            }
            for col_idx in 0..2 {
                let ty = buf.schema.column_type(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!(
                        "wcoj_4cycle_u64_recorded: {} column {} type missing",
                        label, col_idx
                    ))
                })?;
                if !matches!(ty, ScalarType::U64) {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_4cycle_u64_recorded: {} column {} must be U64, got {:?}",
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
            ("col3".to_string(), ScalarType::U64),
        ]);

        let n_e1 = self.logical_row_count_u32(e1)?;
        let n_e2 = self.logical_row_count_u32(e2)?;
        let n_e3 = self.logical_row_count_u32(e3)?;
        let n_e4 = self.logical_row_count_u32(e4)?;

        if n_e1 == 0 || n_e2 == 0 || n_e3 == 0 || n_e4 == 0 {
            return self.create_empty_buffer(out_schema);
        }

        // Phase 1: count + scan + total. Counter buffers stay u32.
        let mut count_buf = self.memory.alloc::<u32>(n_e1 as usize)?;
        let mut offsets_buf = self.memory.alloc::<u32>(n_e1 as usize)?;
        let d_total = self.memory.alloc::<u32>(1)?;

        let e1_col0 = column_u64(e1, 0)?;
        let e1_col1 = column_u64(e1, 1)?;
        let e2_col0 = column_u64(e2, 0)?;
        let e2_col1 = column_u64(e2, 1)?;
        let e3_col0 = column_u64(e3, 0)?;
        let e3_col1 = column_u64(e3, 1)?;
        let e4_col0 = column_u64(e4, 0)?;
        let e4_col1 = column_u64(e4, 1)?;

        let mut rec_count = LaunchRecorder::new_strict(launch_stream);
        rec_count.read(e1.num_rows_device());
        rec_count.read(e2.num_rows_device());
        rec_count.read(e3.num_rows_device());
        rec_count.read(e4.num_rows_device());
        rec_count.read_column(e1.column(0).expect("e1.col0"));
        rec_count.read_column(e1.column(1).expect("e1.col1"));
        rec_count.read_column(e2.column(0).expect("e2.col0"));
        rec_count.read_column(e2.column(1).expect("e2.col1"));
        rec_count.read_column(e3.column(0).expect("e3.col0"));
        rec_count.read_column(e3.column(1).expect("e3.col1"));
        rec_count.read_column(e4.column(0).expect("e4.col0"));
        rec_count.read_column(e4.column(1).expect("e4.col1"));
        rec_count.write(&count_buf);
        rec_count.write(&offsets_buf);
        rec_count.write(&d_total);
        rec_count.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_4cycle_u64_recorded: count preflight failed: {}",
                e
            ))
        })?;

        let device = self.device.inner();
        let count_kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_4CYCLE_COUNT_U64)
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_4cycle_count_u64 kernel not found".to_string())
            })?;
        let grid = (n_e1 + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Phase-timing scaffolding (no-op when feature off).
        let phase_timer = PhaseTimer::new(&cu_stream)?;
        phase_timer.record(0, &cu_stream)?;

        // SAFETY: 13-arg count_u64 (u64* keys + u32 counters).
        // Uses the raw param-vec form — exceeds the LaunchAsync
        // tuple bound (12). Same pattern as the u32 4-cycle entry.
        let count_config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                e1_col0.as_kernel_param(),
                e1_col1.as_kernel_param(),
                n_e1.as_kernel_param(),
                e2_col0.as_kernel_param(),
                e2_col1.as_kernel_param(),
                n_e2.as_kernel_param(),
                e3_col0.as_kernel_param(),
                e3_col1.as_kernel_param(),
                n_e3.as_kernel_param(),
                e4_col0.as_kernel_param(),
                e4_col1.as_kernel_param(),
                n_e4.as_kernel_param(),
                (&mut count_buf).as_kernel_param(),
            ];
            count_kernel
                .clone()
                .launch_on_stream(&cu_stream, count_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("wcoj_4cycle_count_u64 launch failed: {}", e))
                })?;
        }

        // dtod-async copy `count_buf → offsets_buf` (u32-sized).
        let bytes_count = (n_e1 as usize) * std::mem::size_of::<u32>();
        unsafe {
            let res = sys::cuMemcpyDtoDAsync_v2(
                *offsets_buf.device_ptr(),
                *count_buf.device_ptr(),
                bytes_count,
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "wcoj_4cycle_u64_recorded: dtod count_buf → offsets_buf failed: {:?}",
                    res
                )));
            }
        }
        phase_timer.record_after_queued_work(1, &cu_stream)?;

        // Device-side exclusive prefix-sum on offsets_buf — u32 plumbing
        // unchanged from the u32 path.
        self.multiblock_scan_u32_inplace_on_stream(
            &mut offsets_buf,
            n_e1,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        phase_timer.record_after_queued_work(2, &cu_stream)?;

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
                    (&count_buf, &offsets_buf, n_e1, &d_total),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("wcoj_compute_total launch failed: {}", e))
                })?;
        }
        phase_timer.record_after_queued_work(3, &cu_stream)?;

        rec_count.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_4cycle_u64_recorded: count+scan+total commit failed: {}",
                e
            ))
        })?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_4cycle_u64_recorded: stream sync after total failed: {}",
                e
            ))
        })?;
        let total_rows = self
            .dtoh_scalar_untracked::<u32>(&d_total, 0)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "wcoj_4cycle_u64_recorded: read d_total failed: {}",
                    e
                ))
            })?;

        if total_rows == 0 {
            return self.create_empty_buffer(out_schema);
        }

        // Phase 2: materialize. Output columns are 8 bytes/row.
        let bytes_per_col = (total_rows as usize) * std::mem::size_of::<u64>();
        let mut out_w = self.memory.alloc::<u8>(bytes_per_col)?;
        let mut out_x = self.memory.alloc::<u8>(bytes_per_col)?;
        let mut out_y = self.memory.alloc::<u8>(bytes_per_col)?;
        let mut out_z = self.memory.alloc::<u8>(bytes_per_col)?;
        let out_d_num_rows = self.memory.alloc::<u32>(1)?;

        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        rec_mat.read(e1.num_rows_device());
        rec_mat.read(e2.num_rows_device());
        rec_mat.read(e3.num_rows_device());
        rec_mat.read(e4.num_rows_device());
        rec_mat.read_column(e1.column(0).expect("e1.col0"));
        rec_mat.read_column(e1.column(1).expect("e1.col1"));
        rec_mat.read_column(e2.column(0).expect("e2.col0"));
        rec_mat.read_column(e2.column(1).expect("e2.col1"));
        rec_mat.read_column(e3.column(0).expect("e3.col0"));
        rec_mat.read_column(e3.column(1).expect("e3.col1"));
        rec_mat.read_column(e4.column(0).expect("e4.col0"));
        rec_mat.read_column(e4.column(1).expect("e4.col1"));
        rec_mat.read(&offsets_buf);
        rec_mat.write(&out_w);
        rec_mat.write(&out_x);
        rec_mat.write(&out_y);
        rec_mat.write(&out_z);
        rec_mat.write(&out_d_num_rows);
        rec_mat.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_4cycle_u64_recorded: materialize preflight failed: {}",
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
                    "wcoj_4cycle_u64_recorded: H2D out_d_num_rows failed: {:?}",
                    res
                )));
            }
        }

        let materialize_kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_4CYCLE_MATERIALIZE_U64)
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_4cycle_materialize_u64 kernel not found".to_string())
            })?;

        let out_w_u64: &mut TrackedCudaSlice<u64> = unsafe { reinterpret_u8_as_u64(&mut out_w) };
        let out_x_u64: &mut TrackedCudaSlice<u64> = unsafe { reinterpret_u8_as_u64(&mut out_x) };
        let out_y_u64: &mut TrackedCudaSlice<u64> = unsafe { reinterpret_u8_as_u64(&mut out_y) };
        let out_z_u64: &mut TrackedCudaSlice<u64> = unsafe { reinterpret_u8_as_u64(&mut out_z) };

        // SAFETY: 17-arg materialize_u64 (u64* keys + u32 counters
        // + u64* outputs). Same param-vec form as the u32 path —
        // exceeds the LaunchAsync tuple bound (13).
        let mat_config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        phase_timer.record_after_queued_work(4, &cu_stream)?;
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                e1_col0.as_kernel_param(),
                e1_col1.as_kernel_param(),
                n_e1.as_kernel_param(),
                e2_col0.as_kernel_param(),
                e2_col1.as_kernel_param(),
                n_e2.as_kernel_param(),
                e3_col0.as_kernel_param(),
                e3_col1.as_kernel_param(),
                n_e3.as_kernel_param(),
                e4_col0.as_kernel_param(),
                e4_col1.as_kernel_param(),
                n_e4.as_kernel_param(),
                (&offsets_buf).as_kernel_param(),
                total_rows.as_kernel_param(),
                out_w_u64.as_kernel_param(),
                out_x_u64.as_kernel_param(),
                out_y_u64.as_kernel_param(),
                out_z_u64.as_kernel_param(),
            ];
            materialize_kernel
                .clone()
                .launch_on_stream(&cu_stream, mat_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("wcoj_4cycle_materialize_u64 launch failed: {}", e))
                })?;
        }
        phase_timer.record_after_queued_work(5, &cu_stream)?;

        rec_mat.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_4cycle_u64_recorded: materialize commit failed: {}",
                e
            ))
        })?;

        // Sync + capture phase timings (no-op when feature off).
        // The 4-cycle pipeline shape matches triangle's (count,
        // dtod-copy, scan, compute_total, materialize), so we
        // reuse the triangle phase-timing slot.
        #[cfg(feature = "wcoj-phase-timing")]
        {
            cu_stream.synchronize().map_err(|e| {
                XlogError::Kernel(format!(
                    "wcoj_4cycle_u64_recorded: stream sync after materialize failed: {}",
                    e
                ))
            })?;
            let timing = phase_timer.finish()?;
            self.put_wcoj_triangle_phase_timing(timing);
        }
        #[cfg(not(feature = "wcoj-phase-timing"))]
        let _ = phase_timer;

        let columns: Vec<CudaColumn> = vec![out_w.into(), out_x.into(), out_y.into(), out_z.into()];
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

    /// Compute the WCOJ adaptive-dispatch skew score for a u32 (or
    /// Symbol) 4-cycle. Returns `Ok(Some(score))` on success where
    /// `score = max(max_bucket_i / row_count_i)` over the four
    /// join-key columns (col1 of each input — the X/Y/Z/W variables
    /// in their respective second positions). Mirrors the triangle
    /// classifier; the same threshold convention applies (caller
    /// uses `score >= 0.10` to dispatch WCOJ, else fall back).
    ///
    /// Returns `Ok(None)` on any failure (no runtime, validation
    /// failure, kernel error, D2H error, zero rows on any input).
    /// Caller silently falls back to binary join.
    pub fn wcoj_4cycle_skew_score_u32(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<Option<f64>> {
        match self.try_compute_skew_score_4cycle_u32(e1, e2, e3, e4, launch_stream) {
            Ok(score) => Ok(Some(score)),
            Err(_) => Ok(None),
        }
    }

    /// Compute the WCOJ adaptive-dispatch skew score for a U64
    /// 4-cycle. See [`Self::wcoj_4cycle_skew_score_u32`].
    pub fn wcoj_4cycle_skew_score_u64(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<Option<f64>> {
        match self.try_compute_skew_score_4cycle_u64(e1, e2, e3, e4, launch_stream) {
            Ok(score) => Ok(Some(score)),
            Err(_) => Ok(None),
        }
    }

    fn try_compute_skew_score_4cycle_u32(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
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
        for (label, buf) in [("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)] {
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

        let n_e1 = self.logical_row_count_u32(e1)?;
        let n_e2 = self.logical_row_count_u32(e2)?;
        let n_e3 = self.logical_row_count_u32(e3)?;
        let n_e4 = self.logical_row_count_u32(e4)?;
        if n_e1 == 0 || n_e2 == 0 || n_e3 == 0 || n_e4 == 0 {
            return Err(XlogError::Kernel("skew_score: zero rows".to_string()));
        }

        // Classifier reads col0 of each input — the lookup-key
        // sides for the count kernel's binary searches:
        //   * e1.col0 = W (iteration partition + closing-edge W)
        //   * e2.col0 = X (e1 → e2 lookup key)
        //   * e3.col0 = Y (e2 → e3 lookup key)
        //   * e4.col0 = Z (e3 → e4 lookup key)
        // Histogramming col1 instead would miss skew that exists
        // only on the lookup-key side.
        let e1_col0 = column_u32(e1, 0)?;
        let e2_col0 = column_u32(e2, 0)?;
        let e3_col0 = column_u32(e3, 0)?;
        let e4_col0 = column_u32(e4, 0)?;

        // Pre-allocate the 4 × 64 histogram buffer BEFORE the
        // recorder. Total = 1024 bytes — still under the
        // `dtoh_small_metadata_untracked` 4 KB cap.
        let hist_len = (4 * WCOJ_SKEW_NUM_BUCKETS) as usize;
        let mut hist_buf = self.memory.alloc::<u32>(hist_len)?;

        // Block / grid sizing: one row per thread, 256 threads
        // per block. Grid is `4 * blocks_per_col` so each fourth
        // of the grid handles one column.
        let max_n = n_e1.max(n_e2).max(n_e3).max(n_e4);
        let blocks_per_col = (max_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let total_blocks = blocks_per_col.saturating_mul(4);

        // Resolve the kernel handle BEFORE queueing any
        // launch-stream work. If the kernel module isn't loaded,
        // failing here drops `hist_buf` cleanly — no queued
        // memset/launch is in flight.
        let device = self.device.inner();
        let kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_4CYCLE_SKEW_HISTOGRAM_U32)
            .ok_or_else(|| {
                XlogError::Kernel("skew_score: histogram_u32 kernel not found".to_string())
            })?;

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e1.num_rows_device());
        rec.read(e2.num_rows_device());
        rec.read(e3.num_rows_device());
        rec.read(e4.num_rows_device());
        rec.read_column(e1.column(0).expect("e1.col0"));
        rec.read_column(e2.column(0).expect("e2.col0"));
        rec.read_column(e3.column(0).expect("e3.col0"));
        rec.read_column(e4.column(0).expect("e4.col0"));
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
            // sequencing primitive. Bytes = 256 * 4 = 1024.
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
            // SAFETY: 10-arg signature
            //   wcoj_4cycle_skew_histogram_u32(
            //     const u32* e1_col0, u32 n_e1,
            //     const u32* e2_col0, u32 n_e2,
            //     const u32* e3_col0, u32 n_e3,
            //     const u32* e4_col0, u32 n_e4,
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
                            e1_col0,
                            n_e1,
                            e2_col0,
                            n_e2,
                            e3_col0,
                            n_e3,
                            e4_col0,
                            n_e4,
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

        // Step 4: D2H the 256-element histogram via the
        // metadata-only chokepoint.
        let host_hist = self
            .dtoh_small_metadata_untracked::<u32>(&hist_buf, hist_len)
            .map_err(|e| XlogError::Kernel(format!("skew_score: dtoh failed: {e}")))?;

        Ok(score_from_histograms_4cycle(
            &host_hist, n_e1, n_e2, n_e3, n_e4,
        ))
    }

    fn try_compute_skew_score_4cycle_u64(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
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

        for (label, buf) in [("e1", e1), ("e2", e2), ("e3", e3), ("e4", e4)] {
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

        let n_e1 = self.logical_row_count_u32(e1)?;
        let n_e2 = self.logical_row_count_u32(e2)?;
        let n_e3 = self.logical_row_count_u32(e3)?;
        let n_e4 = self.logical_row_count_u32(e4)?;
        if n_e1 == 0 || n_e2 == 0 || n_e3 == 0 || n_e4 == 0 {
            return Err(XlogError::Kernel("skew_score: zero rows".to_string()));
        }

        // Classifier reads col0 of each input — the lookup-key
        // sides for the count kernel's binary searches (mirrors
        // the u32 path's choice; see comment there for rationale).
        let e1_col0 = column_u64(e1, 0)?;
        let e2_col0 = column_u64(e2, 0)?;
        let e3_col0 = column_u64(e3, 0)?;
        let e4_col0 = column_u64(e4, 0)?;

        let hist_len = (4 * WCOJ_SKEW_NUM_BUCKETS) as usize;
        let mut hist_buf = self.memory.alloc::<u32>(hist_len)?;

        let max_n = n_e1.max(n_e2).max(n_e3).max(n_e4);
        let blocks_per_col = (max_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let total_blocks = blocks_per_col.saturating_mul(4);

        // Resolve kernel BEFORE queuing any launch-stream work
        // (mirror of the u32 path's failure-safety hygiene).
        let device = self.device.inner();
        let kernel = device
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_4CYCLE_SKEW_HISTOGRAM_U64)
            .ok_or_else(|| {
                XlogError::Kernel("skew_score: histogram_u64 kernel not found".to_string())
            })?;

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e1.num_rows_device());
        rec.read(e2.num_rows_device());
        rec.read(e3.num_rows_device());
        rec.read(e4.num_rows_device());
        rec.read_column(e1.column(0).expect("e1.col0"));
        rec.read_column(e2.column(0).expect("e2.col0"));
        rec.read_column(e3.column(0).expect("e3.col0"));
        rec.read_column(e4.column(0).expect("e4.col0"));
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
                            e1_col0,
                            n_e1,
                            e2_col0,
                            n_e2,
                            e3_col0,
                            n_e3,
                            e4_col0,
                            n_e4,
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

        Ok(score_from_histograms_4cycle(
            &host_hist, n_e1, n_e2, n_e3, n_e4,
        ))
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

/// Reduce the 256-element histogram into a scalar skew score for a
/// 4-cycle. Returns `max(max_bucket_i / row_count_i)` over the four
/// join-key columns. Bucket layout matches the kernel:
///   `host_hist[col_idx * 64 .. col_idx * 64 + 64]` for col_idx in 0..4
fn score_from_histograms_4cycle(
    host_hist: &[u32],
    n_e1: u32,
    n_e2: u32,
    n_e3: u32,
    n_e4: u32,
) -> f64 {
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
    let s0 = column_score(0, n_e1);
    let s1 = column_score(1, n_e2);
    let s2 = column_score(2, n_e3);
    let s3 = column_score(3, n_e4);
    s0.max(s1).max(s2).max(s3)
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

// ===============================================================
// v0.6.2 — WCOJ layout fast-path implementation.
//
// Goal: when an input is already strictly lex-sorted AND full-row
// unique, skip `dedup_full_row_recorded` (sort + mark-unique +
// compact) and emit a recorded device-side clone instead. The
// existing layout API surface is unchanged; the fast-path is a
// purely additive optimization with proof-based correctness.
//
// Flow:
//   1. Validate (caller already did this).
//   2. Resolve LOGICAL row count via `logical_row_count_u32`
//      (NOT `input.num_rows()` — that returns row_cap on
//      compacted buffers).
//   3. n == 0  → return Ok(None); caller falls through to the
//      existing `dedup_full_row_recorded` n==0 short-circuit
//      (`create_empty_buffer`). We don't mint an empty clone
//      here; we preserve existing semantics exactly.
//   4. n == 1  → recorded clone (trivially sorted+unique).
//   5. n >= 2  → launch the checker kernel under a fresh
//      `LaunchRecorder`; sync; D2H the 4-byte flag.
//   6. Flag == 1 → recorded clone. Flag == 0 → return
//      Ok(None); caller falls through to the dedup path.
//
// Strict-D2H: the 4-byte flag read uses
// `dtoh_scalar_untracked::<u32>` (whitelisted by the strict
// gate, same class as `wcoj_compute_total`'s d_total read).
// ===============================================================

impl CudaKernelProvider {
    /// Try to short-circuit a u32/Symbol layout call by proving
    /// the input is already sorted+unique. Returns:
    ///   * `Ok(Some(out))` — fast-path hit; `out` is the layout.
    ///   * `Ok(None)`      — fast-path missed (n==0 or proof
    ///                       failed). Caller falls through to
    ///                       `dedup_full_row_recorded`.
    ///   * `Err(e)`        — checker pipeline error. Caller
    ///                       treats this as "fall through" to
    ///                       preserve correctness.
    fn try_wcoj_layout_fast_path_u32(
        &self,
        input: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<Option<CudaBuffer>> {
        let runtime = self
            .memory()
            .runtime()
            .ok_or_else(|| XlogError::Kernel("wcoj_layout fast-path: no runtime".to_string()))?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_layout fast-path: stream resolve".to_string())
            })?;

        // LOGICAL row count, not row_cap: compacted buffers
        // (e.g. dedup outputs) have row_cap > logical.
        let n = self.logical_row_count_u32(input)?;
        if n == 0 {
            // Preserve existing semantics: dedup_full_row_recorded's
            // n==0 path returns create_empty_buffer(schema). Don't
            // shadow that here.
            return Ok(None);
        }
        if n == 1 {
            // Trivially sorted+unique; skip the checker entirely.
            return Ok(Some(self.recorded_clone_2col_4byte(
                input,
                n,
                launch_stream,
                &cu_stream,
                runtime,
            )?));
        }

        // n >= 2: run the checker. Output flag in u32 (4 bytes).
        let mut flag_buf = self.memory.alloc::<u32>(1)?;

        let col0 = column_u32(input, 0)?;
        let col1 = column_u32(input, 1)?;

        // Resolve the kernel before queueing launch-stream work.
        // If the module lookup fails, `flag_buf` can drop without
        // racing any in-flight H2D / kernel work.
        let device = self.device.inner();
        let kernel = device
            .get_func(
                WCOJ_MODULE,
                wcoj_kernels::WCOJ_LAYOUT_CHECK_SORTED_UNIQUE_U32,
            )
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_layout_check_sorted_unique_u32 kernel not found".into())
            })?;

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(input.num_rows_device());
        rec.read_column(input.column(0).expect("col0"));
        rec.read_column(input.column(1).expect("col1"));
        rec.write(&flag_buf);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("wcoj_layout fast-path: preflight {e}")))?;

        // Initialize flag = 1 on stream via cuMemsetD32Async-
        // equivalent. The simplest portable path is a 4-byte
        // host->device async copy (sequenced before the kernel
        // by stream order). Doing it as part of the recorded
        // window keeps the dealloc-safety chain intact.
        let one: u32 = 1;
        let grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let queued_result: Result<()> = (|| {
            unsafe {
                let res = sys::cuMemcpyHtoDAsync_v2(
                    *flag_buf.device_ptr(),
                    &one as *const u32 as *const c_void,
                    std::mem::size_of::<u32>(),
                    cu_stream.cu_stream(),
                );
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_layout fast-path: H2D flag init failed: {res:?}"
                    )));
                }
            }

            // SAFETY: 4-arg signature
            //   wcoj_layout_check_sorted_unique_u32(
            //     const u32* col0, const u32* col1, u32 n, u32* flag)
            unsafe {
                kernel
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (col0, col1, n, &mut flag_buf),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "wcoj_layout_check_sorted_unique_u32 launch: {e}"
                        ))
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
            return Err(XlogError::Kernel(format!(
                "wcoj_layout fast-path: commit {e}"
            )));
        }

        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("wcoj_layout fast-path: sync {e}")))?;

        let flag_val = self.dtoh_scalar_untracked::<u32>(&flag_buf, 0)?;
        if flag_val == 1 {
            Ok(Some(self.recorded_clone_2col_4byte(
                input,
                n,
                launch_stream,
                &cu_stream,
                runtime,
            )?))
        } else {
            Ok(None)
        }
    }

    /// U64 variant. Mirrors `try_wcoj_layout_fast_path_u32`.
    fn try_wcoj_layout_fast_path_u64(
        &self,
        input: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<Option<CudaBuffer>> {
        let runtime = self
            .memory()
            .runtime()
            .ok_or_else(|| XlogError::Kernel("wcoj_layout fast-path: no runtime".to_string()))?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_layout fast-path: stream resolve".to_string())
            })?;

        let n = self.logical_row_count_u32(input)?;
        if n == 0 {
            return Ok(None);
        }
        if n == 1 {
            return Ok(Some(self.recorded_clone_2col_8byte(
                input,
                n,
                launch_stream,
                &cu_stream,
                runtime,
            )?));
        }

        let mut flag_buf = self.memory.alloc::<u32>(1)?;
        let col0 = column_u64(input, 0)?;
        let col1 = column_u64(input, 1)?;

        let device = self.device.inner();
        let kernel = device
            .get_func(
                WCOJ_MODULE,
                wcoj_kernels::WCOJ_LAYOUT_CHECK_SORTED_UNIQUE_U64,
            )
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_layout_check_sorted_unique_u64 kernel not found".into())
            })?;

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(input.num_rows_device());
        rec.read_column(input.column(0).expect("col0"));
        rec.read_column(input.column(1).expect("col1"));
        rec.write(&flag_buf);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("wcoj_layout fast-path u64: preflight {e}")))?;

        let one: u32 = 1;
        let grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let queued_result: Result<()> = (|| {
            unsafe {
                let res = sys::cuMemcpyHtoDAsync_v2(
                    *flag_buf.device_ptr(),
                    &one as *const u32 as *const c_void,
                    std::mem::size_of::<u32>(),
                    cu_stream.cu_stream(),
                );
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_layout fast-path u64: H2D flag init failed: {res:?}"
                    )));
                }
            }

            unsafe {
                kernel
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (col0, col1, n, &mut flag_buf),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "wcoj_layout_check_sorted_unique_u64 launch: {e}"
                        ))
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
            return Err(XlogError::Kernel(format!(
                "wcoj_layout fast-path u64: commit {e}"
            )));
        }

        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("wcoj_layout fast-path u64: sync {e}")))?;

        let flag_val = self.dtoh_scalar_untracked::<u32>(&flag_buf, 0)?;
        if flag_val == 1 {
            Ok(Some(self.recorded_clone_2col_8byte(
                input,
                n,
                launch_stream,
                &cu_stream,
                runtime,
            )?))
        } else {
            Ok(None)
        }
    }

    /// Recorded device-side clone of a 2-column 4-byte-per-key
    /// buffer, sized to `n` logical rows. Allocates fresh
    /// columns + d_num_rows on the runtime allocator and copies
    /// via `cuMemcpyDtoDAsync_v2` on `launch_stream` under a
    /// `LaunchRecorder` window. NOT a view: the output buffer
    /// owns its bytes; input lifetime independence is preserved.
    fn recorded_clone_2col_4byte(
        &self,
        input: &CudaBuffer,
        n: u32,
        launch_stream: StreamId,
        cu_stream: &cudarc::driver::CudaStream,
        runtime: &std::sync::Arc<crate::device_runtime::XlogDeviceRuntime>,
    ) -> Result<CudaBuffer> {
        let bpc = (n as usize) * 4;
        let out_col0 = self.memory.alloc::<u8>(bpc)?;
        let out_col1 = self.memory.alloc::<u8>(bpc)?;
        let out_d_num_rows = self.memory.alloc::<u32>(1)?;

        let src_col0 = input.column(0).expect("col0");
        let src_col1 = input.column(1).expect("col1");

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(input.num_rows_device());
        rec.read_column(src_col0);
        rec.read_column(src_col1);
        rec.write(&out_col0);
        rec.write(&out_col1);
        rec.write(&out_d_num_rows);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("wcoj_layout clone 4B: preflight {e}")))?;

        let queued_result: Result<()> = (|| {
            unsafe {
                let r0 = sys::cuMemcpyDtoDAsync_v2(
                    *out_col0.device_ptr(),
                    *src_col0.device_ptr(),
                    bpc,
                    cu_stream.cu_stream(),
                );
                if r0 != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_layout clone 4B: dtod col0 failed: {r0:?}"
                    )));
                }
                let r1 = sys::cuMemcpyDtoDAsync_v2(
                    *out_col1.device_ptr(),
                    *src_col1.device_ptr(),
                    bpc,
                    cu_stream.cu_stream(),
                );
                if r1 != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_layout clone 4B: dtod col1 failed: {r1:?}"
                    )));
                }
                let r2 = sys::cuMemcpyHtoDAsync_v2(
                    *out_d_num_rows.device_ptr(),
                    &n as *const u32 as *const c_void,
                    std::mem::size_of::<u32>(),
                    cu_stream.cu_stream(),
                );
                if r2 != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_layout clone 4B: H2D d_num_rows failed: {r2:?}"
                    )));
                }
            }
            Ok(())
        })();
        if let Err(e) = queued_result {
            let _ = cu_stream.synchronize();
            return Err(e);
        }

        if let Err(e) = rec.commit(runtime) {
            let _ = cu_stream.synchronize();
            return Err(XlogError::Kernel(format!(
                "wcoj_layout clone 4B: commit {e}"
            )));
        }

        Ok(CudaBuffer::from_columns_with_host_count(
            vec![out_col0.into(), out_col1.into()],
            n as u64,
            out_d_num_rows,
            input.schema().clone(),
            n,
        ))
    }

    /// 8-byte-per-key sibling. Same recorder discipline.
    fn recorded_clone_2col_8byte(
        &self,
        input: &CudaBuffer,
        n: u32,
        launch_stream: StreamId,
        cu_stream: &cudarc::driver::CudaStream,
        runtime: &std::sync::Arc<crate::device_runtime::XlogDeviceRuntime>,
    ) -> Result<CudaBuffer> {
        let bpc = (n as usize) * 8;
        let out_col0 = self.memory.alloc::<u8>(bpc)?;
        let out_col1 = self.memory.alloc::<u8>(bpc)?;
        let out_d_num_rows = self.memory.alloc::<u32>(1)?;

        let src_col0 = input.column(0).expect("col0");
        let src_col1 = input.column(1).expect("col1");

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(input.num_rows_device());
        rec.read_column(src_col0);
        rec.read_column(src_col1);
        rec.write(&out_col0);
        rec.write(&out_col1);
        rec.write(&out_d_num_rows);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("wcoj_layout clone 8B: preflight {e}")))?;

        let queued_result: Result<()> = (|| {
            unsafe {
                let r0 = sys::cuMemcpyDtoDAsync_v2(
                    *out_col0.device_ptr(),
                    *src_col0.device_ptr(),
                    bpc,
                    cu_stream.cu_stream(),
                );
                if r0 != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_layout clone 8B: dtod col0 failed: {r0:?}"
                    )));
                }
                let r1 = sys::cuMemcpyDtoDAsync_v2(
                    *out_col1.device_ptr(),
                    *src_col1.device_ptr(),
                    bpc,
                    cu_stream.cu_stream(),
                );
                if r1 != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_layout clone 8B: dtod col1 failed: {r1:?}"
                    )));
                }
                let r2 = sys::cuMemcpyHtoDAsync_v2(
                    *out_d_num_rows.device_ptr(),
                    &n as *const u32 as *const c_void,
                    std::mem::size_of::<u32>(),
                    cu_stream.cu_stream(),
                );
                if r2 != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "wcoj_layout clone 8B: H2D d_num_rows failed: {r2:?}"
                    )));
                }
            }
            Ok(())
        })();
        if let Err(e) = queued_result {
            let _ = cu_stream.synchronize();
            return Err(e);
        }

        if let Err(e) = rec.commit(runtime) {
            let _ = cu_stream.synchronize();
            return Err(XlogError::Kernel(format!(
                "wcoj_layout clone 8B: commit {e}"
            )));
        }

        Ok(CudaBuffer::from_columns_with_host_count(
            vec![out_col0.into(), out_col1.into()],
            n as u64,
            out_d_num_rows,
            input.schema().clone(),
            n,
        ))
    }
}
