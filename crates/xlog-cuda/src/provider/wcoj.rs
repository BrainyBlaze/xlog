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
//!     Physical layout construction is a separate slice — this
//!     entry assumes the caller has already arranged input layout.
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

use cudarc::driver::sys;
use xlog_core::{Result, ScalarType, Schema, XlogError};

use super::{wcoj_kernels, CudaKernelProvider, WCOJ_MODULE};
use crate::device_runtime::StreamId;
use crate::launch::LaunchRecorder;
use crate::memory::{CudaColumn, TrackedCudaSlice};
use crate::CudaBuffer;
use crate::{LaunchAsync, LaunchConfig};

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

// ===============================================================
// W3.2 — General-arity clique WCOJ (k = 5, k = 6) provider.
//
// Four thin public methods (k=5/k=6 × u32/u64) delegate to a
// single generic helper `wcoj_clique_recorded_inner`. Width-class
// (4-byte = U32+Symbol mixable, 8-byte = U64) and K (5 or 6)
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
        _ => panic!("clique_kernel_name: K must be 5 or 6, got {}", k),
    }
}

impl CudaKernelProvider {
    /// W3.2 — generic clique provider helper. Orchestrates count
    /// → scan → total → materialize for K-clique on K*(K-1)/2
    /// 2-column edges in the given width-class.
    ///
    /// Caller pre-conditions:
    ///   * Manager runtime-backed (validated here too).
    ///   * `K ∈ {5, 6}` (validated; panic-free `Err` otherwise).
    ///   * `edges.len() == K * (K - 1) / 2` (10 for k=5, 15 for k=6).
    ///   * Each edge is 2-column with all columns in `width_class`.
    ///   * Each edge is lex-sorted+deduped on `(col0, col1)` —
    ///     same contract as `wcoj_triangle_*_recorded`. The
    ///     runtime dispatcher (W3.2 step 7) routes every edge
    ///     through W3.1's `wcoj_layout_sort_*_recorded` before
    ///     calling here; provider does NOT layout-sort itself.
    fn wcoj_clique_recorded_inner(
        &self,
        k: usize,
        edges: &[&CudaBuffer],
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
        if !(k == 5 || k == 6) {
            return Err(XlogError::Kernel(format!(
                "{}: k must be 5 or 6, got {}",
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

        // Build output schema: K columns, types taken from a
        // canonical-edge column. For vertex i (head var i), pick
        // the type from the canonical edge containing i with i in
        // first position: edge (0, i) for i >= 1, edge (0, 1) for
        // i == 0. Locked per D6 canonical edge layout.
        // Edge (0, i) has its col1 = vertex i for i >= 1; col0 =
        // vertex 0. Edge (0, 1) has col0 = vertex 0, col1 = vertex 1.
        let mut head_types = Vec::with_capacity(k);
        // vertex 0 type from edge (0, 1).col0
        head_types.push(edges[0].schema.column_type(0).expect("validated"));
        // vertex 1 type from edge (0, 1).col1
        head_types.push(edges[0].schema.column_type(1).expect("validated"));
        // vertex i (for i >= 2) type from edge (0, i).col1.
        // Edge index for (0, i) = (i - 1) under canonical lex.
        for i in 2..k {
            let edge_idx_0_i = i - 1; // (0, i) for i >= 1 has edge index i-1
            head_types.push(
                edges[edge_idx_0_i]
                    .schema
                    .column_type(1)
                    .expect("validated"),
            );
        }
        let out_schema = Schema::new(
            head_types
                .iter()
                .enumerate()
                .map(|(i, t)| (format!("col{}", i), *t))
                .collect(),
        );

        // n_leader = row count of edge (0, 1) = edges[0].
        let n_leader = self.logical_row_count_u32(edges[0])?;
        if n_leader == 0 {
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
            edge_col0_ptrs.push(*col0.device_ptr() as u64);
            edge_col1_ptrs.push(*col1.device_ptr() as u64);
            edge_n_host.push(self.logical_row_count_u32(buf)?);
        }

        // Allocate device-side pointer arrays + row counts.
        let mut d_edge_col0 = self.memory.alloc::<u64>(expected_edges)?;
        let mut d_edge_col1 = self.memory.alloc::<u64>(expected_edges)?;
        let mut d_edge_n = self.memory.alloc::<u32>(expected_edges)?;
        let device = self.device.inner();
        device
            .htod_sync_copy_into(&edge_col0_ptrs, &mut d_edge_col0)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "{}: htod edge_col0_ptrs failed: {}",
                    entry_label, e
                ))
            })?;
        device
            .htod_sync_copy_into(&edge_col1_ptrs, &mut d_edge_col1)
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "{}: htod edge_col1_ptrs failed: {}",
                    entry_label, e
                ))
            })?;
        device
            .htod_sync_copy_into(&edge_n_host, &mut d_edge_n)
            .map_err(|e| {
                XlogError::Kernel(format!("{}: htod edge_n failed: {}", entry_label, e))
            })?;

        // Phase 1: HG block counts + scan + total.
        let block_work_unit = crate::wcoj_metadata::WCOJ_HG_BLOCK_WORK_UNIT_DEFAULT;
        let grid = n_leader.div_ceil(block_work_unit);
        let mut count_buf = self.memory.alloc::<u32>(grid as usize)?;
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
        rec_count.write(&count_buf);
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
        //     u32 leader_count,
        //     u32 block_work_unit,
        //     u32* out_block_counts)
        // Pointers all device-resident; preflight verified
        // cross-stream tracking.
        unsafe {
            count_kernel
                .clone()
                .launch_on_stream(
                    &cu_stream,
                    count_config,
                    (
                        &d_edge_col0,
                        &d_edge_col1,
                        &d_edge_n,
                        n_leader,
                        block_work_unit,
                        &mut count_buf,
                    ),
                )
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
            out_col_ptrs.push(*buf.device_ptr() as u64);
            out_col_bufs.push(buf);
        }
        let mut d_out_cols = self.memory.alloc::<u64>(k)?;
        device
            .htod_sync_copy_into(&out_col_ptrs, &mut d_out_cols)
            .map_err(|e| {
                XlogError::Kernel(format!("{}: htod out_col_ptrs failed: {}", entry_label, e))
            })?;
        let out_d_num_rows = self.memory.alloc::<u32>(1)?;

        // H2D output row count.
        unsafe {
            let res = sys::cuMemcpyHtoDAsync_v2(
                *out_d_num_rows.device_ptr(),
                &total_rows as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "{}: H2D out_d_num_rows failed: {:?}",
                    entry_label, res
                )));
            }
        }

        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        for buf in edges.iter() {
            rec_mat.read(buf.num_rows_device());
            rec_mat.read_column(buf.column(0).expect("validated"));
            rec_mat.read_column(buf.column(1).expect("validated"));
        }
        rec_mat.read(&d_edge_col0);
        rec_mat.read(&d_edge_col1);
        rec_mat.read(&d_edge_n);
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
        //     u32 leader_count,
        //     u32 block_work_unit,
        //     const u32* block_offsets,
        //     u32 total_rows,
        //     T* const* out_cols)
        // 8 args, fits the LaunchAsync tuple-launch.
        let mat_config = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            materialize_kernel
                .clone()
                .launch_on_stream(
                    &cu_stream,
                    mat_config,
                    (
                        &d_edge_col0,
                        &d_edge_col1,
                        &d_edge_n,
                        n_leader,
                        block_work_unit,
                        &offsets_buf,
                        total_rows,
                        &d_out_cols,
                    ),
                )
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
            CliqueWidthClass::FourByte,
            launch_stream,
            "wcoj_clique5_u32_recorded",
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
            CliqueWidthClass::EightByte,
            launch_stream,
            "wcoj_clique5_u64_recorded",
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
            CliqueWidthClass::FourByte,
            launch_stream,
            "wcoj_clique6_u32_recorded",
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
            CliqueWidthClass::EightByte,
            launch_stream,
            "wcoj_clique6_u64_recorded",
        )
    }
}
