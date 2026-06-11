//! v0.6.2 GPU 3-way Worst-Case Optimal Join — provider entries.
//!
//! Public methods (parallel u32 and u64 entries — see
//! [`CudaKernelProvider::wcoj_triangle_u32_recorded`] and
//! [`CudaKernelProvider::wcoj_triangle_u64_recorded`]):
//!
//!   * **U32 / Symbol** entry takes 2-column inputs whose columns
//!     may be [`xlog_core::ScalarType::U32`] or
//!     [`xlog_core::ScalarType::Symbol`] (both share the same
//!     4-byte physical layout). It routes through the
//!     histogram-guided block-slice triangle path.
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
//!       Physical layout construction is a separate slice — this
//!       entry assumes the caller has already arranged input layout.
//!   * **Two-phase count → device-scan → materialize.** Mirrors
//!     SRDatalog (Sun et al., arXiv 2604.20073) Section 4's
//!     deterministic two-phase pipeline. Row counts are
//!     prefix-summed on device; the only host visit between the
//!     two phases is a single 4-byte `dtoh_scalar_untracked` of
//!     the inclusive total (sanctioned metadata read, exempt from
//!     the strict deterministic-D2H gate).
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
use std::ffi::c_void;
use std::time::Instant;

use cudarc::driver::sys;
use xlog_core::{Result, ScalarType, Schema, XlogError};

use super::{wcoj_kernels, CudaKernelProvider, WCOJ_MODULE};
use crate::device_runtime::StreamId;
use crate::launch::LaunchRecorder;
use crate::memory::{CudaColumn, TrackedCudaSlice};
use crate::wcoj_metadata::WcojRelationMetadata;
use crate::CudaBuffer;
use crate::{AsKernelParam, LaunchAsync, LaunchConfig};

const BLOCK_SIZE: u32 = 256;

fn column_u32(input: &CudaBuffer, col_idx: usize) -> Result<&TrackedCudaSlice<u32>> {
    let col = input.column(col_idx).ok_or_else(|| {
        XlogError::Kernel(format!(
            "wcoj_layout_u32_recorded: column {col_idx} not found"
        ))
    })?;
    match col {
        CudaColumn::Owned(slice) => unsafe {
            Ok(&*(slice as *const TrackedCudaSlice<u8> as *const TrackedCudaSlice<u32>))
        },
        _ => Err(XlogError::Kernel(
            "wcoj_layout_u32_recorded: input column must be owned".to_string(),
        )),
    }
}

fn column_u64(input: &CudaBuffer, col_idx: usize) -> Result<&TrackedCudaSlice<u64>> {
    let col = input.column(col_idx).ok_or_else(|| {
        XlogError::Kernel(format!(
            "wcoj_layout_u64_recorded: column {col_idx} not found"
        ))
    })?;
    match col {
        CudaColumn::Owned(slice) => unsafe {
            Ok(&*(slice as *const TrackedCudaSlice<u8> as *const TrackedCudaSlice<u64>))
        },
        _ => Err(XlogError::Kernel(
            "wcoj_layout_u64_recorded: input column must be owned".to_string(),
        )),
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

    /// W3.1 — generic full-row sort+dedup for relations of any
    /// arity ≥ 2 in the 4-byte width-class (`U32`, `Symbol`,
    /// mixable within the class).
    ///
    /// **Design (per W3.1 plan iteration 6, D1)**: this is a NEW
    /// entry point. The existing arity-2
    /// [`Self::wcoj_layout_u32_recorded`] is **unchanged** for
    /// the triangle / 4-cycle / project-then-layout callers — it
    /// retains its arity-2-specific fast-path branch. W3.1's
    /// generic surface delegates straight to
    /// [`Self::dedup_full_row_recorded`] for any arity ≥ 2.
    ///
    /// **Validation order** (`runtime → arity ≥ 2 → per-column
    /// width-class → delegate`):
    ///   1. Manager runtime-backed.
    ///   2. `input.arity() >= 2`.
    ///   3. Every column type ∈ `{U32, Symbol}` (4-byte
    ///      width-class). Mixed `U32` + `Symbol` within one
    ///      relation is permitted; `U64` is rejected — use
    ///      [`Self::wcoj_layout_sort_u64_recorded`] instead.
    ///   4. Delegate to `dedup_full_row_recorded(input, launch_stream)`.
    ///
    /// Stream resolution is owned by `dedup_full_row_recorded`
    /// and is NOT in this entry point's validation list. The
    /// `n == 0` short-circuit (returns
    /// `create_empty_buffer(input.schema().clone())`) is also
    /// owned downstream — single source of truth, no duplicated
    /// empty-buffer semantics.
    ///
    /// **Composition**: `dedup_full_row_recorded` only — there
    /// is no fast-path branch for arity ≥ 3 in W3.1 (the
    /// existing arity-2 fast-path stays untouched and reachable
    /// only via `wcoj_layout_u32_recorded`).
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime
    ///   (`with_runtime` is required).
    /// * `XlogError::Kernel` if `input.arity() < 2`.
    /// * `XlogError::Kernel` if any column is not `U32` /
    ///   `Symbol`.
    /// * Whatever `dedup_full_row_recorded` returns for
    ///   stream-resolution / kernel-launch failures.
    pub fn wcoj_layout_sort_u32_recorded(
        &self,
        input: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.record_wcoj_layout_sort_invocation();
        if self.memory().runtime().is_none() {
            return Err(XlogError::Kernel(
                "wcoj_layout_sort_u32_recorded requires a runtime-backed \
                 GpuMemoryManager (constructed via with_runtime)"
                    .to_string(),
            ));
        }
        if input.arity() < 2 {
            return Err(XlogError::Kernel(format!(
                "wcoj_layout_sort_u32_recorded: input must have arity >= 2, got {}",
                input.arity()
            )));
        }
        for col_idx in 0..input.arity() {
            let ty = input.schema.column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_layout_sort_u32_recorded: column {} type missing",
                    col_idx
                ))
            })?;
            if !matches!(ty, ScalarType::U32 | ScalarType::Symbol) {
                return Err(XlogError::Kernel(format!(
                    "wcoj_layout_sort_u32_recorded: column {} must be U32 or Symbol \
                     (4-byte width-class), got {:?}",
                    col_idx, ty
                )));
            }
        }
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
        self.wcoj_triangle_hg_u32_recorded(
            e_xy,
            e_yz,
            e_xz,
            crate::wcoj_metadata::WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
            launch_stream,
        )
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
        self.wcoj_4cycle_hg_u32_recorded(
            e1,
            e2,
            e3,
            e4,
            crate::wcoj_metadata::WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
            launch_stream,
        )
    }

    fn logical_row_count_u32(&self, buf: &CudaBuffer) -> Result<u32> {
        if let Some(c) = buf.cached_row_count() {
            return Ok(c);
        }
        self.dtoh_scalar_untracked::<u32>(buf.num_rows_device(), 0)
    }

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
        if let Ok(Some(out)) = self.try_wcoj_layout_fast_path_u64(input, launch_stream) {
            self.record_wcoj_layout_fast_path_hit();
            return Ok(out);
        }
        // dedup_full_row_recorded internally invokes sort_recorded
        // (U64-aware after commit 1) and the bytewise mask kernel
        // (already width-generic via col_sizes upload).
        self.dedup_full_row_recorded(input, launch_stream)
    }

    /// W3.1 — generic full-row sort+dedup for relations of any
    /// arity ≥ 2 in the 8-byte width-class (`U64` only).
    ///
    /// **Design (per W3.1 plan iteration 6, D1)**: NEW entry
    /// point. The existing arity-2
    /// [`Self::wcoj_layout_u64_recorded`] is **unchanged** for
    /// the existing 2-column callers — it retains its
    /// arity-2-specific fast-path branch. W3.1's generic surface
    /// delegates straight to [`Self::dedup_full_row_recorded`]
    /// for any arity ≥ 2.
    ///
    /// Mirrors [`Self::wcoj_layout_sort_u32_recorded`]'s contract
    /// at the 8-byte width-class — see that entry's doc for the
    /// validation order, stream-resolution ownership, n==0
    /// semantics, and "no fast-path for arity ≥ 3" lock.
    ///
    /// **Validation order** (`runtime → arity ≥ 2 → per-column
    /// width-class → delegate`):
    ///   1. Manager runtime-backed.
    ///   2. `input.arity() >= 2`.
    ///   3. Every column type = `U64`. `U32` / `Symbol` are
    ///      rejected — use [`Self::wcoj_layout_sort_u32_recorded`]
    ///      instead.
    ///   4. Delegate to `dedup_full_row_recorded(input, launch_stream)`.
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime
    ///   (`with_runtime` is required).
    /// * `XlogError::Kernel` if `input.arity() < 2`.
    /// * `XlogError::Kernel` if any column is not `U64`.
    /// * Whatever `dedup_full_row_recorded` returns for
    ///   stream-resolution / kernel-launch failures.
    pub fn wcoj_layout_sort_u64_recorded(
        &self,
        input: &CudaBuffer,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.record_wcoj_layout_sort_invocation();
        if self.memory().runtime().is_none() {
            return Err(XlogError::Kernel(
                "wcoj_layout_sort_u64_recorded requires a runtime-backed \
                 GpuMemoryManager (constructed via with_runtime)"
                    .to_string(),
            ));
        }
        if input.arity() < 2 {
            return Err(XlogError::Kernel(format!(
                "wcoj_layout_sort_u64_recorded: input must have arity >= 2, got {}",
                input.arity()
            )));
        }
        for col_idx in 0..input.arity() {
            let ty = input.schema.column_type(col_idx).ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_layout_sort_u64_recorded: column {} type missing",
                    col_idx
                ))
            })?;
            if !matches!(ty, ScalarType::U64) {
                return Err(XlogError::Kernel(format!(
                    "wcoj_layout_sort_u64_recorded: column {} must be U64 \
                     (8-byte width-class), got {:?}",
                    col_idx, ty
                )));
            }
        }
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
        self.wcoj_triangle_hg_u64_recorded(
            e_xy,
            e_yz,
            e_xz,
            crate::wcoj_metadata::WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
            launch_stream,
        )
    }

    /// Evaluate
    /// `cycle4(W, X, Y, Z) :- e1(W,X), e2(X,Y), e3(Y,Z), e4(Z,W)`
    /// on already-sorted, already-deduped binary U64 relations.
    /// Mirrors [`Self::wcoj_4cycle_u32_recorded`]'s contract; the
    /// only differences are the 8-byte join-key reads/writes and
    /// the U64 HG planner/count/materialize kernels. Counters and
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
        self.wcoj_4cycle_hg_u64_recorded(
            e1,
            e2,
            e3,
            e4,
            crate::wcoj_metadata::WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT,
            launch_stream,
        )
    }
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
    ///     failed). Caller falls through to
    ///     `dedup_full_row_recorded`.
    ///   * `Err(e)`        — checker pipeline error. Caller
    ///     treats this as "fall through" to preserve correctness.
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
        let grid = n.div_ceil(BLOCK_SIZE);
        let queued_result: Result<()> = (|| {
            self.htod_launch_metadata_async_copy_one(
                &one,
                &flag_buf,
                &cu_stream,
                "wcoj_layout fast-path flag init",
            )?;

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
        let grid = n.div_ceil(BLOCK_SIZE);
        let queued_result: Result<()> = (|| {
            self.htod_launch_metadata_async_copy_one(
                &one,
                &flag_buf,
                &cu_stream,
                "wcoj_layout fast-path u64 flag init",
            )?;

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
            }
            self.htod_launch_metadata_async_copy_one(
                &n,
                &out_d_num_rows,
                cu_stream,
                "wcoj_layout clone 4B d_num_rows",
            )?;
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
            }
            self.htod_launch_metadata_async_copy_one(
                &n,
                &out_d_num_rows,
                cu_stream,
                "wcoj_layout clone 8B d_num_rows",
            )?;
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

// ===============================================================
// W3.2/W6.4 — General-arity clique WCOJ (k = 5..8) provider.
//
// Thin public methods (k=5..8 × u32/u64) delegate to a
// single generic helper `wcoj_clique_recorded_inner`. Width-class
// (4-byte = U32+Symbol mixable, 8-byte = U64) and K
// drive kernel-name selection and per-row element-size; otherwise
// the orchestration is identical: validate → upload edge-pointer
// arrays → count → scan → total → materialize → output.
//
// Each public entry assumes sorted+deduped input as a
// pre-condition (same contract as `wcoj_triangle_u32_recorded` /
// `wcoj_4cycle_u32_recorded`); the runtime dispatcher routes
// every edge through W3.1's `wcoj_layout_sort_*_recorded` before
// invoking these entries.
// ===============================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CliqueWidthClass {
    FourByte,  // U32 / Symbol
    EightByte, // U64
}

impl CliqueWidthClass {
    fn elem_bytes(self) -> usize {
        match self {
            CliqueWidthClass::FourByte => 4,
            CliqueWidthClass::EightByte => 8,
        }
    }
    fn label(self) -> &'static str {
        match self {
            CliqueWidthClass::FourByte => "u32",
            CliqueWidthClass::EightByte => "u64",
        }
    }
    fn validate_col_type(self, ty: ScalarType) -> bool {
        match self {
            CliqueWidthClass::FourByte => matches!(ty, ScalarType::U32 | ScalarType::Symbol),
            CliqueWidthClass::EightByte => matches!(ty, ScalarType::U64),
        }
    }
}

/// Resolve the kernel name for a given `(K, count_or_materialize, width_class)`.
fn clique_kernel_name(k: usize, materialize: bool, w: CliqueWidthClass) -> &'static str {
    match (k, materialize, w) {
        (5, false, CliqueWidthClass::FourByte) => wcoj_kernels::WCOJ_CLIQUE5_COUNT_HG_U32,
        (5, true, CliqueWidthClass::FourByte) => wcoj_kernels::WCOJ_CLIQUE5_MATERIALIZE_HG_U32,
        (5, false, CliqueWidthClass::EightByte) => wcoj_kernels::WCOJ_CLIQUE5_COUNT_HG_U64,
        (5, true, CliqueWidthClass::EightByte) => wcoj_kernels::WCOJ_CLIQUE5_MATERIALIZE_HG_U64,
        (6, false, CliqueWidthClass::FourByte) => wcoj_kernels::WCOJ_CLIQUE6_COUNT_HG_U32,
        (6, true, CliqueWidthClass::FourByte) => wcoj_kernels::WCOJ_CLIQUE6_MATERIALIZE_HG_U32,
        (6, false, CliqueWidthClass::EightByte) => wcoj_kernels::WCOJ_CLIQUE6_COUNT_HG_U64,
        (6, true, CliqueWidthClass::EightByte) => wcoj_kernels::WCOJ_CLIQUE6_MATERIALIZE_HG_U64,
        (7, false, CliqueWidthClass::FourByte) => wcoj_kernels::WCOJ_CLIQUE7_COUNT_HG_U32,
        (7, true, CliqueWidthClass::FourByte) => wcoj_kernels::WCOJ_CLIQUE7_MATERIALIZE_HG_U32,
        (7, false, CliqueWidthClass::EightByte) => wcoj_kernels::WCOJ_CLIQUE7_COUNT_HG_U64,
        (7, true, CliqueWidthClass::EightByte) => wcoj_kernels::WCOJ_CLIQUE7_MATERIALIZE_HG_U64,
        (8, false, CliqueWidthClass::FourByte) => wcoj_kernels::WCOJ_CLIQUE8_COUNT_HG_U32,
        (8, true, CliqueWidthClass::FourByte) => wcoj_kernels::WCOJ_CLIQUE8_MATERIALIZE_HG_U32,
        (8, false, CliqueWidthClass::EightByte) => wcoj_kernels::WCOJ_CLIQUE8_COUNT_HG_U64,
        (8, true, CliqueWidthClass::EightByte) => wcoj_kernels::WCOJ_CLIQUE8_MATERIALIZE_HG_U64,
        _ => panic!("clique_kernel_name: K must be 5..8, got {}", k),
    }
}

enum CliqueLeaderMetadata {
    U32(WcojRelationMetadata<u32>),
    U64(WcojRelationMetadata<u64>),
}

impl CliqueLeaderMetadata {
    fn total_rows_u32(&self, entry_label: &str) -> Result<u32> {
        let total = match self {
            CliqueLeaderMetadata::U32(metadata) => metadata.total,
            CliqueLeaderMetadata::U64(metadata) => metadata.total,
        };
        u32::try_from(total).map_err(|_| {
            XlogError::Kernel(format!(
                "{}: leader metadata total {} exceeds u32 kernel surface",
                entry_label, total
            ))
        })
    }

    fn key_count(&self) -> u32 {
        match self {
            CliqueLeaderMetadata::U32(metadata) => metadata.key_count,
            CliqueLeaderMetadata::U64(metadata) => metadata.key_count,
        }
    }
}

fn validate_clique_metadata_leader<'a>(
    k: usize,
    edges: &'a [&CudaBuffer],
    leader_edge_idx: u32,
    width_class: CliqueWidthClass,
    entry_label: &str,
) -> Result<&'a CudaBuffer> {
    if !(5..=8).contains(&k) {
        return Err(XlogError::Kernel(format!(
            "{}: k must be 5..8, got {}",
            entry_label, k
        )));
    }
    let expected_edges = k * (k - 1) / 2;
    if edges.len() != expected_edges {
        return Err(XlogError::Kernel(format!(
            "{}: expected {} edges (= C({}, 2)), got {}",
            entry_label,
            expected_edges,
            k,
            edges.len()
        )));
    }
    let leader_slot = usize::try_from(leader_edge_idx)
        .ok()
        .filter(|idx| *idx < expected_edges)
        .ok_or_else(|| {
            XlogError::Kernel(format!(
                "{}: leader_edge_idx {} out of range for {} edges",
                entry_label, leader_edge_idx, expected_edges
            ))
        })?;
    let leader = edges[leader_slot];
    if leader.arity() != 2 {
        return Err(XlogError::Kernel(format!(
            "{}: leader edge must be 2-column, got arity {}",
            entry_label,
            leader.arity()
        )));
    }
    let ty = leader.schema.column_type(0).ok_or_else(|| {
        XlogError::Kernel(format!(
            "{}: leader edge column 0 type missing",
            entry_label
        ))
    })?;
    if !width_class.validate_col_type(ty) {
        return Err(XlogError::Kernel(format!(
            "{}: leader edge column 0 type {:?} not in {} width-class",
            entry_label,
            ty,
            width_class.label()
        )));
    }
    Ok(leader)
}

impl CudaKernelProvider {
    fn wcoj_clique_metadata_recorded_u32_inner(
        &self,
        k: usize,
        edges: &[&CudaBuffer],
        leader_edge_idx: u32,
        launch_stream: StreamId,
        entry_label: &str,
    ) -> Result<WcojRelationMetadata<u32>> {
        let leader = validate_clique_metadata_leader(
            k,
            edges,
            leader_edge_idx,
            CliqueWidthClass::FourByte,
            entry_label,
        )?;
        self.wcoj_build_metadata_u32_recorded(leader, 0, launch_stream)
    }

    fn wcoj_clique_metadata_recorded_u64_inner(
        &self,
        k: usize,
        edges: &[&CudaBuffer],
        leader_edge_idx: u32,
        launch_stream: StreamId,
        entry_label: &str,
    ) -> Result<WcojRelationMetadata<u64>> {
        let leader = validate_clique_metadata_leader(
            k,
            edges,
            leader_edge_idx,
            CliqueWidthClass::EightByte,
            entry_label,
        )?;
        self.wcoj_build_metadata_u64_recorded(leader, 0, launch_stream)
    }

    /// Build leader-edge runtime metadata for a 5-clique 4-byte-width dispatch.
    pub fn wcoj_clique5_metadata_recorded_u32(
        &self,
        edges: &[&CudaBuffer; 10],
        leader_edge_idx: u32,
        launch_stream: StreamId,
    ) -> Result<WcojRelationMetadata<u32>> {
        self.wcoj_clique_metadata_recorded_u32_inner(
            5,
            edges,
            leader_edge_idx,
            launch_stream,
            "wcoj_clique5_metadata_recorded_u32",
        )
    }

    /// Build leader-edge runtime metadata for a 5-clique 8-byte-width dispatch.
    pub fn wcoj_clique5_metadata_recorded_u64(
        &self,
        edges: &[&CudaBuffer; 10],
        leader_edge_idx: u32,
        launch_stream: StreamId,
    ) -> Result<WcojRelationMetadata<u64>> {
        self.wcoj_clique_metadata_recorded_u64_inner(
            5,
            edges,
            leader_edge_idx,
            launch_stream,
            "wcoj_clique5_metadata_recorded_u64",
        )
    }

    /// Build leader-edge runtime metadata for a 6-clique 4-byte-width dispatch.
    pub fn wcoj_clique6_metadata_recorded_u32(
        &self,
        edges: &[&CudaBuffer; 15],
        leader_edge_idx: u32,
        launch_stream: StreamId,
    ) -> Result<WcojRelationMetadata<u32>> {
        self.wcoj_clique_metadata_recorded_u32_inner(
            6,
            edges,
            leader_edge_idx,
            launch_stream,
            "wcoj_clique6_metadata_recorded_u32",
        )
    }

    /// Build leader-edge runtime metadata for a 6-clique 8-byte-width dispatch.
    pub fn wcoj_clique6_metadata_recorded_u64(
        &self,
        edges: &[&CudaBuffer; 15],
        leader_edge_idx: u32,
        launch_stream: StreamId,
    ) -> Result<WcojRelationMetadata<u64>> {
        self.wcoj_clique_metadata_recorded_u64_inner(
            6,
            edges,
            leader_edge_idx,
            launch_stream,
            "wcoj_clique6_metadata_recorded_u64",
        )
    }

    /// W3.2 — generic clique provider helper. Orchestrates count
    /// → scan → total → materialize for K-clique on K*(K-1)/2
    /// 2-column edges in the given width-class.
    ///
    /// Caller pre-conditions:
    ///   * Manager runtime-backed (validated here too).
    ///   * `K ∈ {5, 6, 7, 8}` (validated; panic-free `Err` otherwise).
    ///   * `edges.len() == K * (K - 1) / 2`.
    ///   * Each edge is 2-column with all columns in `width_class`.
    ///   * Each edge is lex-sorted+deduped on `(col0, col1)` —
    ///     same contract as `wcoj_triangle_*_recorded`. The
    ///     runtime dispatcher (W3.2 step 7) routes every edge
    ///     through W3.1's `wcoj_layout_sort_*_recorded` before
    ///     calling here; provider does NOT layout-sort itself.
    #[allow(clippy::too_many_arguments)]
    fn wcoj_clique_recorded_inner(
        &self,
        k: usize,
        edges: &[&CudaBuffer],
        leader_edge_idx: u32,
        edge_order: Option<&[u8]>,
        iteration_order: Option<&[u8]>,
        width_class: CliqueWidthClass,
        launch_stream: StreamId,
        entry_label: &str,
    ) -> Result<CudaBuffer> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(format!(
                "{} requires a runtime-backed GpuMemoryManager (with_runtime)",
                entry_label
            ))
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "{}: launch_stream StreamId({}) does not resolve",
                    entry_label, launch_stream.0
                ))
            })?;
        if !(5..=8).contains(&k) {
            return Err(XlogError::Kernel(format!(
                "{}: k must be 5..8, got {}",
                entry_label, k
            )));
        }
        let expected_edges = k * (k - 1) / 2;
        if edges.len() != expected_edges {
            return Err(XlogError::Kernel(format!(
                "{}: expected {} edges (= C({}, 2)), got {}",
                entry_label,
                expected_edges,
                k,
                edges.len()
            )));
        }
        if usize::try_from(leader_edge_idx)
            .ok()
            .is_none_or(|idx| idx >= expected_edges)
        {
            return Err(XlogError::Kernel(format!(
                "{}: leader_edge_idx {} out of range for {} edges",
                entry_label, leader_edge_idx, expected_edges
            )));
        }
        match (edge_order, iteration_order) {
            (Some(edge_order), Some(iteration_order)) => {
                validate_clique_u8_permutation(
                    edge_order,
                    expected_edges,
                    "edge_order",
                    entry_label,
                )?;
                validate_clique_u8_permutation(iteration_order, k, "iteration_order", entry_label)?;
            }
            (None, None) => {}
            _ => {
                return Err(XlogError::Kernel(format!(
                    "{}: edge_order and iteration_order must both be present or both be omitted",
                    entry_label
                )));
            }
        }

        // Validate every edge: 2-column, in width-class.
        for (i, buf) in edges.iter().enumerate() {
            if buf.arity() != 2 {
                return Err(XlogError::Kernel(format!(
                    "{}: edge[{}] must be 2-column, got arity {}",
                    entry_label,
                    i,
                    buf.arity()
                )));
            }
            for col_idx in 0..2 {
                let ty = buf.schema.column_type(col_idx).ok_or_else(|| {
                    XlogError::Kernel(format!(
                        "{}: edge[{}] column {} type missing",
                        entry_label, i, col_idx
                    ))
                })?;
                if !width_class.validate_col_type(ty) {
                    return Err(XlogError::Kernel(format!(
                        "{}: edge[{}] column {} type {:?} not in {} width-class",
                        entry_label,
                        i,
                        col_idx,
                        ty,
                        width_class.label()
                    )));
                }
            }
        }

        // Build output schema in kernel binding order. The runtime
        // projects this buffer back to rule-head order when the plan
        // chooses a non-identity variable order.
        let mut head_types = Vec::with_capacity(k);
        let leader_slot = edge_order.map(|order| order[0] as usize).unwrap_or(0);
        head_types.push(edges[leader_slot].schema.column_type(0).expect("validated"));
        head_types.push(edges[leader_slot].schema.column_type(1).expect("validated"));
        for i in 2..k {
            let logical_edge = i - 1;
            let edge_slot = edge_order
                .map(|order| order[logical_edge] as usize)
                .unwrap_or(logical_edge);
            head_types.push(edges[edge_slot].schema.column_type(1).expect("validated"));
        }
        let out_schema = Schema::new(
            head_types
                .iter()
                .enumerate()
                .map(|(i, t)| (format!("col{}", i), *t))
                .collect(),
        );

        let leader_slot = usize::try_from(leader_edge_idx).expect("validated");
        let n_leader = self.logical_row_count_u32(edges[leader_slot])?;
        if n_leader == 0 {
            return self.create_empty_buffer(out_schema);
        }
        // Paper §5 Algorithm 1 Phase 1: Histograms maintained alongside data; refreshed during Merge per Authorization 5 (2026-05-17)
        let metadata_start = Instant::now();
        let leader_metadata = match width_class {
            CliqueWidthClass::FourByte => {
                CliqueLeaderMetadata::U32(self.wcoj_clique_metadata_recorded_u32_inner(
                    k,
                    edges,
                    leader_edge_idx,
                    launch_stream,
                    entry_label,
                )?)
            }
            CliqueWidthClass::EightByte => {
                CliqueLeaderMetadata::U64(self.wcoj_clique_metadata_recorded_u64_inner(
                    k,
                    edges,
                    leader_edge_idx,
                    launch_stream,
                    entry_label,
                )?)
            }
        };
        self.record_kclique_metadata_build_nanos(metadata_start.elapsed().as_nanos());
        let leader_work_total = leader_metadata.total_rows_u32(entry_label)?;
        if leader_work_total != n_leader {
            return Err(XlogError::Kernel(format!(
                "{}: leader metadata total {} does not match leader row count {}",
                entry_label, leader_work_total, n_leader
            )));
        }
        let leader_metadata_key_count = leader_metadata.key_count();
        if leader_metadata_key_count == 0 {
            return self.create_empty_buffer(out_schema);
        }

        // Build host-side per-edge pointer arrays + row-count
        // array. These get htod'd to small device buffers that
        // the kernel reads as `const T* const* edge_col0` etc.
        let mut edge_col0_ptrs: Vec<u64> = Vec::with_capacity(expected_edges);
        let mut edge_col1_ptrs: Vec<u64> = Vec::with_capacity(expected_edges);
        let mut edge_n_host: Vec<u32> = Vec::with_capacity(expected_edges);
        for buf in edges.iter() {
            let col0 = buf.column(0).expect("validated");
            let col1 = buf.column(1).expect("validated");
            edge_col0_ptrs.push(*col0.device_ptr());
            edge_col1_ptrs.push(*col1.device_ptr());
            edge_n_host.push(self.logical_row_count_u32(buf)?);
        }

        // Allocate device-side pointer arrays + row counts.
        let mut d_edge_col0 = self.memory.alloc::<u64>(expected_edges)?;
        let mut d_edge_col1 = self.memory.alloc::<u64>(expected_edges)?;
        let mut d_edge_n = self.memory.alloc::<u32>(expected_edges)?;
        let device = self.device.inner();
        self.htod_launch_metadata_sync_copy_into(&edge_col0_ptrs, &mut d_edge_col0)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "{}: htod edge_col0_ptrs failed: {}",
                    entry_label, e
                ))
            })?;
        self.htod_launch_metadata_sync_copy_into(&edge_col1_ptrs, &mut d_edge_col1)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "{}: htod edge_col1_ptrs failed: {}",
                    entry_label, e
                ))
            })?;
        self.htod_launch_metadata_sync_copy_into(&edge_n_host, &mut d_edge_n)
            .map_err(|e| {
                XlogError::Kernel(format!("{}: htod edge_n failed: {}", entry_label, e))
            })?;
        let d_edge_order = if let Some(edge_order) = edge_order {
            let mut buf = self.memory.alloc::<u8>(expected_edges)?;
            self.htod_launch_metadata_sync_copy_into(edge_order, &mut buf)
                .map_err(|e| {
                    XlogError::Kernel(format!("{}: htod edge_order failed: {}", entry_label, e))
                })?;
            Some(buf)
        } else {
            None
        };
        let d_iteration_order = if let Some(iteration_order) = iteration_order {
            let mut buf = self.memory.alloc::<u8>(k)?;
            self.htod_launch_metadata_sync_copy_into(iteration_order, &mut buf)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "{}: htod iteration_order failed: {}",
                        entry_label, e
                    ))
                })?;
            Some(buf)
        } else {
            None
        };

        // Phase 1: HG block counts + scan + total.
        let block_work_unit = crate::wcoj_metadata::WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT;
        let grid = leader_work_total.div_ceil(block_work_unit);
        let count_buf = self.memory.alloc::<u32>(grid as usize)?;
        let thread_counts_buf = self
            .memory
            .alloc::<u32>((grid as usize) * (BLOCK_SIZE as usize))?;
        let mut offsets_buf = self.memory.alloc::<u32>(grid as usize)?;
        let d_total = self.memory.alloc::<u32>(1)?;

        let mut rec_count = LaunchRecorder::new_strict(launch_stream);
        for buf in edges.iter() {
            rec_count.read(buf.num_rows_device());
            rec_count.read_column(buf.column(0).expect("validated"));
            rec_count.read_column(buf.column(1).expect("validated"));
        }
        rec_count.read(&d_edge_col0);
        rec_count.read(&d_edge_col1);
        rec_count.read(&d_edge_n);
        if let Some(buf) = d_edge_order.as_ref() {
            rec_count.read(buf);
        }
        if let Some(buf) = d_iteration_order.as_ref() {
            rec_count.read(buf);
        }
        match &leader_metadata {
            CliqueLeaderMetadata::U32(leader_metadata) => {
                rec_count.read(&leader_metadata.unique_keys);
                rec_count.read(&leader_metadata.fan_out);
                rec_count.read(&leader_metadata.prefix_sum);
            }
            CliqueLeaderMetadata::U64(leader_metadata) => {
                rec_count.read(&leader_metadata.unique_keys);
                rec_count.read(&leader_metadata.fan_out);
                rec_count.read(&leader_metadata.prefix_sum);
            }
        }
        rec_count.write(&count_buf);
        rec_count.write(&thread_counts_buf);
        rec_count.write(&offsets_buf);
        rec_count.write(&d_total);
        rec_count.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!("{}: count preflight failed: {}", entry_label, e))
        })?;

        let count_kernel = device
            .get_func(WCOJ_MODULE, clique_kernel_name(k, false, width_class))
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "{}: count kernel '{}' not found",
                    entry_label,
                    clique_kernel_name(k, false, width_class)
                ))
            })?;
        let count_config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: kernel signature
        //   wcoj_clique{K}_count_*(
        //     const T* const* edge_col0,
        //     const T* const* edge_col1,
        //     const u32* edge_n,
        //     u32 leader_edge_idx,
        //     const u8* edge_order,
        //     const u8* iteration_order,
        //     u32 leader_count,
        //     const T* unique_keys,
        //     const u32* fan_out,
        //     const u32* prefix_sum,
        //     u32 metadata_key_count,
        //     u32 block_work_unit,
        //     u32* out_block_counts,
        //     u32* out_thread_counts)
        // Pointers all device-resident; preflight verified
        // cross-stream tracking. Raw params are required because
        // the metadata-extended ABI exceeds the tuple-launch arity.
        let null_order_ptr = 0_u64;
        let edge_order_param = match d_edge_order.as_ref() {
            Some(buf) => buf.as_kernel_param(),
            None => null_order_ptr.as_kernel_param(),
        };
        let iteration_order_param = match d_iteration_order.as_ref() {
            Some(buf) => buf.as_kernel_param(),
            None => null_order_ptr.as_kernel_param(),
        };
        unsafe {
            let mut params: Vec<*mut c_void> = match &leader_metadata {
                CliqueLeaderMetadata::U32(leader_metadata) => vec![
                    (&d_edge_col0).as_kernel_param(),
                    (&d_edge_col1).as_kernel_param(),
                    (&d_edge_n).as_kernel_param(),
                    leader_edge_idx.as_kernel_param(),
                    edge_order_param,
                    iteration_order_param,
                    n_leader.as_kernel_param(),
                    (&leader_metadata.unique_keys).as_kernel_param(),
                    (&leader_metadata.fan_out).as_kernel_param(),
                    (&leader_metadata.prefix_sum).as_kernel_param(),
                    leader_metadata_key_count.as_kernel_param(),
                    block_work_unit.as_kernel_param(),
                    (&count_buf).as_kernel_param(),
                    (&thread_counts_buf).as_kernel_param(),
                ],
                CliqueLeaderMetadata::U64(leader_metadata) => vec![
                    (&d_edge_col0).as_kernel_param(),
                    (&d_edge_col1).as_kernel_param(),
                    (&d_edge_n).as_kernel_param(),
                    leader_edge_idx.as_kernel_param(),
                    edge_order_param,
                    iteration_order_param,
                    n_leader.as_kernel_param(),
                    (&leader_metadata.unique_keys).as_kernel_param(),
                    (&leader_metadata.fan_out).as_kernel_param(),
                    (&leader_metadata.prefix_sum).as_kernel_param(),
                    leader_metadata_key_count.as_kernel_param(),
                    block_work_unit.as_kernel_param(),
                    (&count_buf).as_kernel_param(),
                    (&thread_counts_buf).as_kernel_param(),
                ],
            };
            count_kernel
                .clone()
                .launch_on_stream(&cu_stream, count_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "{}: count kernel launch failed: {}",
                        entry_label, e
                    ))
                })?;
        }

        // dtod count → offsets (scan modifies offsets in place).
        let bytes_count = (grid as usize) * std::mem::size_of::<u32>();
        unsafe {
            let res = sys::cuMemcpyDtoDAsync_v2(
                *offsets_buf.device_ptr(),
                *count_buf.device_ptr(),
                bytes_count,
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "{}: dtod count → offsets failed: {:?}",
                    entry_label, res
                )));
            }
        }

        // Device-side exclusive prefix-sum on offsets.
        self.multiblock_scan_u32_inplace_on_stream(
            &mut offsets_buf,
            grid,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        // Compute total = counts[n-1] + offsets[n-1].
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
                    (&count_buf, &offsets_buf, grid, &d_total),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("wcoj_compute_total launch failed: {}", e))
                })?;
        }

        rec_count.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "{}: count+scan+total commit failed: {}",
                entry_label, e
            ))
        })?;

        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "{}: stream sync after total failed: {}",
                entry_label, e
            ))
        })?;
        let total_rows = self
            .dtoh_scalar_untracked::<u32>(&d_total, 0)
            .map_err(|e| {
                XlogError::Kernel(format!("{}: read d_total failed: {}", entry_label, e))
            })?;

        if total_rows == 0 {
            return self.create_empty_buffer(out_schema);
        }

        // Phase 2: materialize. Allocate K output column buffers.
        let elem_bytes = width_class.elem_bytes();
        let bytes_per_col = (total_rows as usize) * elem_bytes;
        let mut out_col_bufs: Vec<TrackedCudaSlice<u8>> = Vec::with_capacity(k);
        let mut out_col_ptrs: Vec<u64> = Vec::with_capacity(k);
        for _ in 0..k {
            let buf = self.memory.alloc::<u8>(bytes_per_col)?;
            out_col_ptrs.push(*buf.device_ptr());
            out_col_bufs.push(buf);
        }
        let mut d_out_cols = self.memory.alloc::<u64>(k)?;
        self.htod_launch_metadata_sync_copy_into(&out_col_ptrs, &mut d_out_cols)
            .map_err(|e| {
                XlogError::Kernel(format!("{}: htod out_col_ptrs failed: {}", entry_label, e))
            })?;
        let out_d_num_rows = self.memory.alloc::<u32>(1)?;

        // H2D output row count.
        self.htod_launch_metadata_async_copy_one(
            &total_rows,
            &out_d_num_rows,
            &cu_stream,
            &format!("{entry_label}: out_d_num_rows"),
        )?;

        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        for buf in edges.iter() {
            rec_mat.read(buf.num_rows_device());
            rec_mat.read_column(buf.column(0).expect("validated"));
            rec_mat.read_column(buf.column(1).expect("validated"));
        }
        rec_mat.read(&d_edge_col0);
        rec_mat.read(&d_edge_col1);
        rec_mat.read(&d_edge_n);
        if let Some(buf) = d_edge_order.as_ref() {
            rec_mat.read(buf);
        }
        if let Some(buf) = d_iteration_order.as_ref() {
            rec_mat.read(buf);
        }
        match &leader_metadata {
            CliqueLeaderMetadata::U32(leader_metadata) => {
                rec_mat.read(&leader_metadata.unique_keys);
                rec_mat.read(&leader_metadata.fan_out);
                rec_mat.read(&leader_metadata.prefix_sum);
            }
            CliqueLeaderMetadata::U64(leader_metadata) => {
                rec_mat.read(&leader_metadata.unique_keys);
                rec_mat.read(&leader_metadata.fan_out);
                rec_mat.read(&leader_metadata.prefix_sum);
            }
        }
        rec_mat.read(&thread_counts_buf);
        rec_mat.read(&offsets_buf);
        rec_mat.read(&d_out_cols);
        for buf in out_col_bufs.iter() {
            rec_mat.write(buf);
        }
        rec_mat.write(&out_d_num_rows);
        rec_mat.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "{}: materialize preflight failed: {}",
                entry_label, e
            ))
        })?;

        let materialize_kernel = device
            .get_func(WCOJ_MODULE, clique_kernel_name(k, true, width_class))
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "{}: materialize kernel '{}' not found",
                    entry_label,
                    clique_kernel_name(k, true, width_class)
                ))
            })?;

        // SAFETY: kernel signature
        //   wcoj_clique{K}_materialize_*(
        //     const T* const* edge_col0,
        //     const T* const* edge_col1,
        //     const u32* edge_n,
        //     u32 leader_edge_idx,
        //     const u8* edge_order,
        //     const u8* iteration_order,
        //     u32 leader_count,
        //     const T* unique_keys,
        //     const u32* fan_out,
        //     const u32* prefix_sum,
        //     u32 metadata_key_count,
        //     u32 block_work_unit,
        //     const u32* thread_counts,
        //     const u32* block_offsets,
        //     u32 total_rows,
        //     T* const* out_cols)
        // Raw params are required because the metadata-extended ABI
        // exceeds the tuple-launch arity.
        let mat_config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut params: Vec<*mut c_void> = match &leader_metadata {
                CliqueLeaderMetadata::U32(leader_metadata) => vec![
                    (&d_edge_col0).as_kernel_param(),
                    (&d_edge_col1).as_kernel_param(),
                    (&d_edge_n).as_kernel_param(),
                    leader_edge_idx.as_kernel_param(),
                    edge_order_param,
                    iteration_order_param,
                    n_leader.as_kernel_param(),
                    (&leader_metadata.unique_keys).as_kernel_param(),
                    (&leader_metadata.fan_out).as_kernel_param(),
                    (&leader_metadata.prefix_sum).as_kernel_param(),
                    leader_metadata_key_count.as_kernel_param(),
                    block_work_unit.as_kernel_param(),
                    (&thread_counts_buf).as_kernel_param(),
                    (&offsets_buf).as_kernel_param(),
                    total_rows.as_kernel_param(),
                    (&d_out_cols).as_kernel_param(),
                ],
                CliqueLeaderMetadata::U64(leader_metadata) => vec![
                    (&d_edge_col0).as_kernel_param(),
                    (&d_edge_col1).as_kernel_param(),
                    (&d_edge_n).as_kernel_param(),
                    leader_edge_idx.as_kernel_param(),
                    edge_order_param,
                    iteration_order_param,
                    n_leader.as_kernel_param(),
                    (&leader_metadata.unique_keys).as_kernel_param(),
                    (&leader_metadata.fan_out).as_kernel_param(),
                    (&leader_metadata.prefix_sum).as_kernel_param(),
                    leader_metadata_key_count.as_kernel_param(),
                    block_work_unit.as_kernel_param(),
                    (&thread_counts_buf).as_kernel_param(),
                    (&offsets_buf).as_kernel_param(),
                    total_rows.as_kernel_param(),
                    (&d_out_cols).as_kernel_param(),
                ],
            };
            materialize_kernel
                .clone()
                .launch_on_stream(&cu_stream, mat_config, &mut params)
                .map_err(|e| {
                    XlogError::Kernel(format!("{}: materialize launch failed: {}", entry_label, e))
                })?;
        }

        rec_mat.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!("{}: materialize commit failed: {}", entry_label, e))
        })?;

        let columns: Vec<CudaColumn> = out_col_bufs.into_iter().map(|b| b.into()).collect();
        Ok(CudaBuffer::from_columns_with_host_count(
            columns,
            total_rows as u64,
            out_d_num_rows,
            out_schema,
            total_rows,
        ))
    }

    /// W3.2 — 5-clique WCOJ at 4-byte width-class.
    ///
    /// `edges` must contain exactly **10** 2-column buffers in
    /// canonical lex `(i, j)` order (i < j): `(0,1), (0,2), (0,3),
    /// (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)`. Each
    /// column may be `U32` or `Symbol` (mixable within the
    /// 4-byte width-class). All edges must be lex-sorted+deduped
    /// on `(col0, col1)` — the runtime dispatcher routes through
    /// `wcoj_layout_sort_u32_recorded` before calling here.
    pub fn wcoj_clique5_u32_recorded(
        &self,
        edges: &[&CudaBuffer; 10],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            5,
            edges,
            0,
            None,
            None,
            CliqueWidthClass::FourByte,
            launch_stream,
            "wcoj_clique5_u32_recorded",
        )
    }

    /// 5-clique WCOJ at 4-byte width-class using plan-derived launch params.
    pub fn wcoj_clique5_u32_recorded_planned(
        &self,
        edges: &[&CudaBuffer; 10],
        leader_edge_idx: u32,
        edge_order: &[u8],
        iteration_order: &[u8],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            5,
            edges,
            leader_edge_idx,
            Some(edge_order),
            Some(iteration_order),
            CliqueWidthClass::FourByte,
            launch_stream,
            "wcoj_clique5_u32_recorded_planned",
        )
    }

    /// W3.2 — 5-clique WCOJ at 8-byte width-class (U64 only).
    pub fn wcoj_clique5_u64_recorded(
        &self,
        edges: &[&CudaBuffer; 10],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            5,
            edges,
            0,
            None,
            None,
            CliqueWidthClass::EightByte,
            launch_stream,
            "wcoj_clique5_u64_recorded",
        )
    }

    /// 5-clique WCOJ at 8-byte width-class using plan-derived launch params.
    pub fn wcoj_clique5_u64_recorded_planned(
        &self,
        edges: &[&CudaBuffer; 10],
        leader_edge_idx: u32,
        edge_order: &[u8],
        iteration_order: &[u8],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            5,
            edges,
            leader_edge_idx,
            Some(edge_order),
            Some(iteration_order),
            CliqueWidthClass::EightByte,
            launch_stream,
            "wcoj_clique5_u64_recorded_planned",
        )
    }

    /// W3.2 — 6-clique WCOJ at 4-byte width-class.
    ///
    /// `edges` must contain exactly **15** 2-column buffers in
    /// canonical lex `(i, j)` order. Width-class + sort+dedup
    /// pre-condition match `wcoj_clique5_u32_recorded`.
    pub fn wcoj_clique6_u32_recorded(
        &self,
        edges: &[&CudaBuffer; 15],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            6,
            edges,
            0,
            None,
            None,
            CliqueWidthClass::FourByte,
            launch_stream,
            "wcoj_clique6_u32_recorded",
        )
    }

    /// 6-clique WCOJ at 4-byte width-class using plan-derived launch params.
    pub fn wcoj_clique6_u32_recorded_planned(
        &self,
        edges: &[&CudaBuffer; 15],
        leader_edge_idx: u32,
        edge_order: &[u8],
        iteration_order: &[u8],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            6,
            edges,
            leader_edge_idx,
            Some(edge_order),
            Some(iteration_order),
            CliqueWidthClass::FourByte,
            launch_stream,
            "wcoj_clique6_u32_recorded_planned",
        )
    }

    /// W3.2 — 6-clique WCOJ at 8-byte width-class (U64 only).
    pub fn wcoj_clique6_u64_recorded(
        &self,
        edges: &[&CudaBuffer; 15],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            6,
            edges,
            0,
            None,
            None,
            CliqueWidthClass::EightByte,
            launch_stream,
            "wcoj_clique6_u64_recorded",
        )
    }

    /// 6-clique WCOJ at 8-byte width-class using plan-derived launch params.
    pub fn wcoj_clique6_u64_recorded_planned(
        &self,
        edges: &[&CudaBuffer; 15],
        leader_edge_idx: u32,
        edge_order: &[u8],
        iteration_order: &[u8],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            6,
            edges,
            leader_edge_idx,
            Some(edge_order),
            Some(iteration_order),
            CliqueWidthClass::EightByte,
            launch_stream,
            "wcoj_clique6_u64_recorded_planned",
        )
    }

    /// W6.4 — 7-clique WCOJ at 4-byte width-class.
    pub fn wcoj_clique7_u32_recorded(
        &self,
        edges: &[&CudaBuffer; 21],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            7,
            edges,
            0,
            None,
            None,
            CliqueWidthClass::FourByte,
            launch_stream,
            "wcoj_clique7_u32_recorded",
        )
    }

    /// W6.4 — 7-clique WCOJ at 4-byte width-class using plan-derived launch params.
    pub fn wcoj_clique7_u32_recorded_planned(
        &self,
        edges: &[&CudaBuffer; 21],
        leader_edge_idx: u32,
        edge_order: &[u8],
        iteration_order: &[u8],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            7,
            edges,
            leader_edge_idx,
            Some(edge_order),
            Some(iteration_order),
            CliqueWidthClass::FourByte,
            launch_stream,
            "wcoj_clique7_u32_recorded_planned",
        )
    }

    /// W6.4 — 7-clique WCOJ at 8-byte width-class (U64 only).
    pub fn wcoj_clique7_u64_recorded(
        &self,
        edges: &[&CudaBuffer; 21],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            7,
            edges,
            0,
            None,
            None,
            CliqueWidthClass::EightByte,
            launch_stream,
            "wcoj_clique7_u64_recorded",
        )
    }

    /// W6.4 — 7-clique WCOJ at 8-byte width-class using plan-derived launch params.
    pub fn wcoj_clique7_u64_recorded_planned(
        &self,
        edges: &[&CudaBuffer; 21],
        leader_edge_idx: u32,
        edge_order: &[u8],
        iteration_order: &[u8],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            7,
            edges,
            leader_edge_idx,
            Some(edge_order),
            Some(iteration_order),
            CliqueWidthClass::EightByte,
            launch_stream,
            "wcoj_clique7_u64_recorded_planned",
        )
    }

    /// W6.4 — 8-clique WCOJ at 4-byte width-class.
    pub fn wcoj_clique8_u32_recorded(
        &self,
        edges: &[&CudaBuffer; 28],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            8,
            edges,
            0,
            None,
            None,
            CliqueWidthClass::FourByte,
            launch_stream,
            "wcoj_clique8_u32_recorded",
        )
    }

    /// W6.4 — 8-clique WCOJ at 4-byte width-class using plan-derived launch params.
    pub fn wcoj_clique8_u32_recorded_planned(
        &self,
        edges: &[&CudaBuffer; 28],
        leader_edge_idx: u32,
        edge_order: &[u8],
        iteration_order: &[u8],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            8,
            edges,
            leader_edge_idx,
            Some(edge_order),
            Some(iteration_order),
            CliqueWidthClass::FourByte,
            launch_stream,
            "wcoj_clique8_u32_recorded_planned",
        )
    }

    /// W6.4 — 8-clique WCOJ at 8-byte width-class (U64 only).
    pub fn wcoj_clique8_u64_recorded(
        &self,
        edges: &[&CudaBuffer; 28],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            8,
            edges,
            0,
            None,
            None,
            CliqueWidthClass::EightByte,
            launch_stream,
            "wcoj_clique8_u64_recorded",
        )
    }

    /// W6.4 — 8-clique WCOJ at 8-byte width-class using plan-derived launch params.
    pub fn wcoj_clique8_u64_recorded_planned(
        &self,
        edges: &[&CudaBuffer; 28],
        leader_edge_idx: u32,
        edge_order: &[u8],
        iteration_order: &[u8],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_recorded_inner(
            8,
            edges,
            leader_edge_idx,
            Some(edge_order),
            Some(iteration_order),
            CliqueWidthClass::EightByte,
            launch_stream,
            "wcoj_clique8_u64_recorded_planned",
        )
    }

    /// S1e — aggregate-fused K-clique count-by-root (u32 width-class,
    /// K ∈ {5, 6}). For `q(R, count(*)) :- <complete K-clique body>`
    /// grouped by the plan's position-0 root variable, computes the
    /// (root, count) row set WITHOUT materializing the clique rows.
    ///
    /// Pipeline (mirrors `wcoj_4cycle_groupby_root_count_u32_recorded`):
    ///   1. Layout-normalize every edge per dispatch (sorted-fast-path
    ///      clone when already lex-sorted + unique) — the fused path must
    ///      give the same guarantee as the unfused pipeline instead of
    ///      trusting store-buffer sortedness.
    ///   2. Build the leader-edge runtime metadata, htod the per-edge
    ///      pointer arrays + plan orders (same surface as the unfused
    ///      planned clique entries).
    ///   3. `wcoj_clique{K}_groupby_root_count_hg_u32` accumulates, per
    ///      leader-edge row, the row's clique completion count via
    ///      atomicAdd (order-insensitive, deterministic values). The
    ///      row's group key is the oriented leader edge's col0 — the
    ///      kernel's binding[0] root.
    ///   4. Staging (root, count) over the n_leader input rows, compact
    ///      count>0, reduce per root with the recorded groupby Sum.
    ///
    /// All reduction work is O(n_leader) — input-sized, never
    /// join-output-sized. Output schema (root: U32/Symbol, count: U64)
    /// matches the unfused materialize+groupby baseline.
    #[allow(clippy::too_many_arguments)]
    fn wcoj_clique_groupby_root_count_recorded_inner(
        &self,
        k: usize,
        edges: &[&CudaBuffer],
        leader_edge_idx: u32,
        edge_order: &[u8],
        iteration_order: &[u8],
        launch_stream: StreamId,
        entry_label: &str,
    ) -> Result<CudaBuffer> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(format!(
                "{} requires a runtime-backed GpuMemoryManager (with_runtime)",
                entry_label
            ))
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "{}: launch_stream StreamId({}) does not resolve",
                    entry_label, launch_stream.0
                ))
            })?;
        if !matches!(k, 5 | 6) {
            return Err(XlogError::Kernel(format!(
                "{}: fused count-by-root supports k 5..6, got {}",
                entry_label, k
            )));
        }
        let expected_edges = k * (k - 1) / 2;
        if edges.len() != expected_edges {
            return Err(XlogError::Kernel(format!(
                "{}: expected {} edges (= C({}, 2)), got {}",
                entry_label,
                expected_edges,
                k,
                edges.len()
            )));
        }
        let leader_slot = usize::try_from(leader_edge_idx)
            .ok()
            .filter(|idx| *idx < expected_edges)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "{}: leader_edge_idx {} out of range for {} edges",
                    entry_label, leader_edge_idx, expected_edges
                ))
            })?;
        validate_clique_u8_permutation(edge_order, expected_edges, "edge_order", entry_label)?;
        validate_clique_u8_permutation(iteration_order, k, "iteration_order", entry_label)?;

        // Layout-normalize per dispatch (commit 31b0ccf0 contract for
        // ALL fused group-by-root entries). Also enforces the 2-column
        // 4-byte width-class per edge.
        let mut laid_out: Vec<CudaBuffer> = Vec::with_capacity(expected_edges);
        for buf in edges {
            laid_out.push(self.wcoj_layout_u32_recorded(buf, launch_stream)?);
        }
        let edges: Vec<&CudaBuffer> = laid_out.iter().collect();
        let leader = edges[leader_slot];

        let w_type = leader.schema.column_type(0).ok_or_else(|| {
            XlogError::Kernel(format!(
                "{}: leader edge column 0 type missing",
                entry_label
            ))
        })?;
        let out_schema = Schema::new(vec![
            ("root".to_string(), w_type),
            ("count".to_string(), ScalarType::U64),
        ]);
        let n_leader = self.logical_row_count_u32(leader)?;
        if n_leader == 0 {
            return self.create_empty_buffer(out_schema);
        }

        let leader_metadata = self.wcoj_clique_metadata_recorded_u32_inner(
            k,
            &edges,
            leader_edge_idx,
            launch_stream,
            entry_label,
        )?;
        let leader_work_total =
            u32::try_from(leader_metadata.total).map_err(|_| {
                XlogError::Kernel(format!(
                    "{}: leader metadata total {} exceeds u32 kernel surface",
                    entry_label, leader_metadata.total
                ))
            })?;
        if leader_work_total != n_leader {
            return Err(XlogError::Kernel(format!(
                "{}: leader metadata total {} does not match leader row count {}",
                entry_label, leader_work_total, n_leader
            )));
        }
        if leader_metadata.key_count == 0 {
            return self.create_empty_buffer(out_schema);
        }

        // Host-side per-edge pointer arrays + row counts, htod'd to
        // small device buffers (same surface as the unfused planned
        // clique entries).
        let mut edge_col0_ptrs: Vec<u64> = Vec::with_capacity(expected_edges);
        let mut edge_col1_ptrs: Vec<u64> = Vec::with_capacity(expected_edges);
        let mut edge_n_host: Vec<u32> = Vec::with_capacity(expected_edges);
        for buf in edges.iter() {
            let col0 = buf.column(0).ok_or_else(|| {
                XlogError::Kernel(format!("{}: edge column 0 missing", entry_label))
            })?;
            let col1 = buf.column(1).ok_or_else(|| {
                XlogError::Kernel(format!("{}: edge column 1 missing", entry_label))
            })?;
            edge_col0_ptrs.push(*col0.device_ptr());
            edge_col1_ptrs.push(*col1.device_ptr());
            edge_n_host.push(self.logical_row_count_u32(buf)?);
        }
        let mut d_edge_col0 = self.memory.alloc::<u64>(expected_edges)?;
        let mut d_edge_col1 = self.memory.alloc::<u64>(expected_edges)?;
        let mut d_edge_n = self.memory.alloc::<u32>(expected_edges)?;
        self.htod_launch_metadata_sync_copy_into(&edge_col0_ptrs, &mut d_edge_col0)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "{}: htod edge_col0_ptrs failed: {}",
                    entry_label, e
                ))
            })?;
        self.htod_launch_metadata_sync_copy_into(&edge_col1_ptrs, &mut d_edge_col1)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "{}: htod edge_col1_ptrs failed: {}",
                    entry_label, e
                ))
            })?;
        self.htod_launch_metadata_sync_copy_into(&edge_n_host, &mut d_edge_n)
            .map_err(|e| {
                XlogError::Kernel(format!("{}: htod edge_n failed: {}", entry_label, e))
            })?;
        let mut d_edge_order = self.memory.alloc::<u8>(expected_edges)?;
        self.htod_launch_metadata_sync_copy_into(edge_order, &mut d_edge_order)
            .map_err(|e| {
                XlogError::Kernel(format!("{}: htod edge_order failed: {}", entry_label, e))
            })?;
        let mut d_iteration_order = self.memory.alloc::<u8>(k)?;
        self.htod_launch_metadata_sync_copy_into(iteration_order, &mut d_iteration_order)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "{}: htod iteration_order failed: {}",
                    entry_label, e
                ))
            })?;

        // Per-leader-row match counters, zero-initialized. Allocated as
        // the u8-backed column layout so the array doubles as the
        // staging buffer's count column after the kernel fills it.
        let mut row_counts = self
            .memory()
            .alloc::<u8>(n_leader as usize * std::mem::size_of::<u32>())?;
        self.device()
            .inner()
            .memset_zeros(&mut row_counts)
            .map_err(|e| XlogError::Kernel(format!("{}: zero row counts failed: {}", entry_label, e)))?;

        let block_work_unit = crate::wcoj_metadata::WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT;
        let grid = leader_work_total.div_ceil(block_work_unit);
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        for buf in edges.iter() {
            rec.read(buf.num_rows_device());
            rec.read_column(buf.column(0).expect("validated"));
            rec.read_column(buf.column(1).expect("validated"));
        }
        rec.read(&d_edge_col0);
        rec.read(&d_edge_col1);
        rec.read(&d_edge_n);
        rec.read(&d_edge_order);
        rec.read(&d_iteration_order);
        rec.read(&leader_metadata.unique_keys);
        rec.read(&leader_metadata.fan_out);
        rec.read(&leader_metadata.prefix_sum);
        rec.write(&row_counts);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{}: preflight failed: {}", entry_label, e)))?;

        let kernel_name = match k {
            5 => wcoj_kernels::WCOJ_CLIQUE5_GROUPBY_ROOT_COUNT_HG_U32,
            _ => wcoj_kernels::WCOJ_CLIQUE6_GROUPBY_ROOT_COUNT_HG_U32,
        };
        let kernel = self
            .device()
            .inner()
            .get_func(WCOJ_MODULE, kernel_name)
            .ok_or_else(|| {
                XlogError::Kernel(format!("{}: kernel '{}' not found", entry_label, kernel_name))
            })?;
        // SAFETY: kernel signature
        //   wcoj_clique{K}_groupby_root_count_hg_u32(
        //     const u32* const* edge_col0,
        //     const u32* const* edge_col1,
        //     const u32* edge_n,
        //     u32 leader_edge_idx,
        //     const u8* edge_order,
        //     const u8* iteration_order,
        //     u32 leader_count,
        //     const u32* unique_keys,
        //     const u32* fan_out,
        //     const u32* prefix_sum,
        //     u32 metadata_key_count,
        //     u32 block_work_unit,
        //     u32* out_row_counts)
        // Pointers all device-resident; preflight verified cross-stream
        // tracking. Raw params are required because the
        // metadata-extended ABI exceeds the tuple-launch arity.
        unsafe {
            let mut params: Vec<*mut c_void> = vec![
                (&d_edge_col0).as_kernel_param(),
                (&d_edge_col1).as_kernel_param(),
                (&d_edge_n).as_kernel_param(),
                leader_edge_idx.as_kernel_param(),
                (&d_edge_order).as_kernel_param(),
                (&d_iteration_order).as_kernel_param(),
                n_leader.as_kernel_param(),
                (&leader_metadata.unique_keys).as_kernel_param(),
                (&leader_metadata.fan_out).as_kernel_param(),
                (&leader_metadata.prefix_sum).as_kernel_param(),
                leader_metadata.key_count.as_kernel_param(),
                block_work_unit.as_kernel_param(),
                (&row_counts).as_kernel_param(),
            ];
            kernel
                .clone()
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    &mut params,
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "{}: groupby-count launch failed: {}",
                        entry_label, e
                    ))
                })?;
        }
        rec.commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{}: commit failed: {}", entry_label, e)))?;

        // Staging buffer (root, count) over the n_leader input rows:
        // root is a device-to-device copy of the oriented leader edge's
        // col0; the count column is the kernel-filled array. Rows stay
        // lex-sorted by root.
        let root_src = match leader.column(0).expect("validated") {
            CudaColumn::Owned(slice) => slice,
            _ => {
                return Err(XlogError::Kernel(format!(
                    "{}: leader.col0 must be an owned CudaColumn",
                    entry_label
                )))
            }
        };
        let root_copy = self
            .memory()
            .alloc::<u8>(n_leader as usize * std::mem::size_of::<u32>())?;
        // Explicit-length copy: layout-normalized columns are allocated at
        // capacity, which can exceed the logical n_leader * 4 bytes a
        // full-slice typed copy would assert on.
        unsafe {
            let res = sys::cuMemcpyDtoD_v2(
                *root_copy.device_ptr(),
                *root_src.device_ptr(),
                n_leader as usize * std::mem::size_of::<u32>(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "{}: copy root column failed: {:?}",
                    entry_label, res
                )));
            }
        }
        let mut d_num_rows = self.memory().alloc::<u32>(1)?;
        self.device()
            .inner()
            .dtod_copy(leader.num_rows_device(), &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("{}: copy row count failed: {}", entry_label, e)))?;
        let staging_schema = Schema::new(vec![
            ("root".to_string(), w_type),
            ("count".to_string(), ScalarType::U32),
        ]);
        let staging = CudaBuffer::from_columns_with_host_count(
            vec![root_copy.into(), row_counts.into()],
            n_leader as u64,
            d_num_rows,
            staging_schema,
            n_leader,
        );

        // Keep only roots with at least one completed clique, then
        // reduce per root. Both steps run over input-sized data.
        let mask = self.compare_const_mask_recorded::<u32>(
            &staging,
            1,
            0u32,
            crate::CompareOp::Gt,
            launch_stream,
        )?;
        let compacted =
            self.compact_buffer_by_device_mask_counted_recorded(&staging, &mask, launch_stream)?;
        self.groupby_multi_agg_recorded(
            &compacted,
            &[0],
            &[(1, xlog_core::AggOp::Sum)],
            launch_stream,
        )
    }

    /// S1e — fused 5-clique count-by-root at the 4-byte width-class,
    /// using plan-derived launch params. See
    /// [`Self::wcoj_clique_groupby_root_count_recorded_inner`].
    pub fn wcoj_clique5_groupby_root_count_u32_recorded_planned(
        &self,
        edges: &[&CudaBuffer; 10],
        leader_edge_idx: u32,
        edge_order: &[u8],
        iteration_order: &[u8],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_groupby_root_count_recorded_inner(
            5,
            edges,
            leader_edge_idx,
            edge_order,
            iteration_order,
            launch_stream,
            "wcoj_clique5_groupby_root_count_u32_recorded_planned",
        )
    }

    /// S1e — fused 6-clique count-by-root at the 4-byte width-class,
    /// using plan-derived launch params. See
    /// [`Self::wcoj_clique_groupby_root_count_recorded_inner`].
    pub fn wcoj_clique6_groupby_root_count_u32_recorded_planned(
        &self,
        edges: &[&CudaBuffer; 15],
        leader_edge_idx: u32,
        edge_order: &[u8],
        iteration_order: &[u8],
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.wcoj_clique_groupby_root_count_recorded_inner(
            6,
            edges,
            leader_edge_idx,
            edge_order,
            iteration_order,
            launch_stream,
            "wcoj_clique6_groupby_root_count_u32_recorded_planned",
        )
    }
}

fn validate_clique_u8_permutation(
    values: &[u8],
    len: usize,
    label: &str,
    entry_label: &str,
) -> Result<()> {
    if values.len() != len {
        return Err(XlogError::Kernel(format!(
            "{}: {} length {} must equal {}",
            entry_label,
            label,
            values.len(),
            len
        )));
    }
    let mut seen = vec![false; len];
    for &value in values {
        let idx = usize::from(value);
        if idx >= len {
            return Err(XlogError::Kernel(format!(
                "{}: {} value {} out of range 0..{}",
                entry_label, label, value, len
            )));
        }
        if seen[idx] {
            return Err(XlogError::Kernel(format!(
                "{}: {} duplicates value {}",
                entry_label, label, value
            )));
        }
        seen[idx] = true;
    }
    Ok(())
}
