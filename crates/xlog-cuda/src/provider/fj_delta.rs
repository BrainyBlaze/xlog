//! D3 — factorized recursive delta (S3 spike): fused novel-set
//! evaluation for semi-naive transitive closure at the provider level.
//!
//! Design: `docs/plans/2026-06-12-d3-factorized-delta-design.md`. The
//! per-source novel set of one TC iteration,
//!
//! ```text
//!   novel[x] = (∪_{y ∈ delta[x]} edge[y]) \ R[x]
//! ```
//!
//! is a union of flat sorted-range trie nodes (the D2 substrate) minus
//! the stable relation's rows. Instead of materializing the
//! witness-multiplied flat join and diffing it afterwards (the
//! production `hash_join_v2` → `diff_gpu` pipeline), this entry
//! evaluates the union–diff over a dense-domain characteristic
//! bitvector — `domain` bits per source, `domain²/8` bytes total —
//! so rediscoveries and duplicate witnesses collapse without ever
//! being written out. Only surviving novel tuples are flattened
//! (output-linear), and the emit order makes the result lex-sorted
//! and full-row-deduped by construction: it is simultaneously the
//! next iteration's delta and the `union_gpu` input.
//!
//! Spike invariants:
//!   * `edge` must be layout-normalized (lex-sorted (y, z), full-row
//!     deduped — `wcoj_layout_u32_recorded`); `delta` and `full_r`
//!     need no ordering (the bitmap is order-insensitive).
//!   * u32/Symbol width class only; ids must be `< domain`, enforced
//!     fail-closed by an in-kernel error flag, and
//!     `domain ≤ FJ_DELTA_MAX_DOMAIN` (dense-domain spike bound; the
//!     sparse-domain generalization is gated Phase B work).
//!   * zero tracked transfers — host reads are the sanctioned
//!     `dtoh_scalar_untracked` metadata scalars (scan totals, error
//!     flag); recorded launches throughout.

use xlog_core::{Result, ScalarType, XlogError};

use super::{wcoj_kernels, CudaKernelProvider, WCOJ_MODULE};
use crate::device_runtime::StreamId;
use crate::launch::LaunchRecorder;
use crate::CudaBuffer;
use crate::{LaunchAsync, LaunchConfig};

const BLOCK_SIZE: u32 = 256;

/// Dense-domain bound for the spike bitmap (`domain²/8` bytes = 512 MB
/// at the bound; gate fixtures use `domain ≤ 2^13`).
pub const FJ_DELTA_MAX_DOMAIN: u32 = 1 << 16;

fn require_binary_u32_class(buf: &CudaBuffer, name: &str, ctx: &str) -> Result<()> {
    if buf.arity() != 2 {
        return Err(XlogError::Kernel(format!(
            "{ctx}: {name} must be arity-2, got {}",
            buf.arity()
        )));
    }
    for idx in 0..2 {
        match buf.schema().column_type(idx) {
            Some(ScalarType::U32) | Some(ScalarType::Symbol) => {}
            other => {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: {name} column {idx} must be U32/Symbol, got {other:?}"
                )));
            }
        }
    }
    Ok(())
}

impl CudaKernelProvider {
    /// One factorized semi-naive delta step for transitive closure:
    /// returns the lex-sorted, full-row-deduped novel set
    /// `{(x, z) : (x, y) ∈ delta, (y, z) ∈ edge, (x, z) ∉ full_r}`.
    ///
    /// `edge` must be layout-normalized (lex-sorted, deduped); `delta`
    /// and `full_r` are order-insensitive. All ids must be `< domain`.
    /// The output schema is `full_r`'s schema (union-compatible with
    /// the stable relation by construction).
    pub fn fj_delta_novel_u32_recorded(
        &self,
        delta: &CudaBuffer,
        edge: &CudaBuffer,
        full_r: &CudaBuffer,
        domain: u32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "fj_delta_novel_u32_recorded";
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(format!(
                "{ctx} requires a runtime-backed GpuMemoryManager \
                 (constructed via with_runtime)"
            ))
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "{ctx}: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        require_binary_u32_class(delta, "delta", ctx)?;
        require_binary_u32_class(edge, "edge", ctx)?;
        require_binary_u32_class(full_r, "full_r", ctx)?;
        if domain == 0 || domain > FJ_DELTA_MAX_DOMAIN {
            return Err(XlogError::Kernel(format!(
                "{ctx}: domain {domain} outside (0, {FJ_DELTA_MAX_DOMAIN}] \
                 (dense-domain spike bound)"
            )));
        }

        let row_count = |buf: &CudaBuffer| -> Result<u32> {
            match buf.cached_row_count() {
                Some(c) => Ok(c),
                None => self.dtoh_scalar_untracked::<u32>(buf.num_rows_device(), 0),
            }
        };
        let n_delta = row_count(delta)?;
        let n_edge = row_count(edge)?;
        let n_r = row_count(full_r)?;

        let out_schema = full_r.schema().clone();
        if n_delta == 0 || n_edge == 0 {
            return self.create_empty_buffer(out_schema);
        }

        let words_per_row = domain.div_ceil(32);
        let n_words = u64::from(domain) * u64::from(words_per_row);
        // domain ≤ 2^16 keeps n_words ≤ 2^27 — well inside u32.
        let n_words = n_words as u32;

        let delta_x = delta
            .column(0)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: delta column 0 missing")))?;
        let delta_y = delta
            .column(1)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: delta column 1 missing")))?;
        let edge_y = edge
            .column(0)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: edge column 0 missing")))?;
        let edge_z = edge
            .column(1)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: edge column 1 missing")))?;
        let r_x = full_r
            .column(0)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: full_r column 0 missing")))?;
        let r_z = full_r
            .column(1)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: full_r column 1 missing")))?;
        let delta_x_v = self.column_as_u32_view(delta_x, n_delta as usize)?;
        let delta_y_v = self.column_as_u32_view(delta_y, n_delta as usize)?;
        let edge_y_v = self.column_as_u32_view(edge_y, n_edge as usize)?;
        let edge_z_v = self.column_as_u32_view(edge_z, n_edge as usize)?;

        // ---- Phase 1: per-delta-row trie ranges + work prefix.
        let range_lo = self.memory().alloc::<u32>(n_delta as usize)?;
        let mut wp = self.memory().alloc::<u32>(n_delta as usize + 1)?;
        {
            let mut rec = LaunchRecorder::new_strict(launch_stream);
            rec.read_column(delta_y);
            rec.read_column(edge_y);
            rec.write(&range_lo);
            rec.write(&wp);
            rec.preflight(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: range preflight failed: {e}")))?;
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::FJ_DELTA_RANGE_U32)
                .ok_or_else(|| {
                    XlogError::Kernel("fj_delta_range_u32 kernel not found".to_string())
                })?;
            let grid = n_delta.div_ceil(BLOCK_SIZE);
            // SAFETY: fj_delta_range_u32(delta_y, n_delta, edge_y,
            // n_edge, range_lo, work_prefix); buffers are
            // device-resident and preflighted.
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
                        (&delta_y_v, n_delta, &edge_y_v, n_edge, &range_lo, &mut wp),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("fj_delta_range_u32 launch failed: {e}"))
                    })?;
            }
            self.multiblock_scan_u32_inplace_on_stream(
                &mut wp,
                n_delta + 1,
                &cu_stream,
                launch_stream,
                runtime,
            )?;
            rec.commit(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: range commit failed: {e}")))?;
        }
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: range sync failed: {e}")))?;
        let total_work = u64::from(self.dtoh_scalar_untracked::<u32>(&wp, n_delta as usize)?);
        if total_work == 0 {
            return self.create_empty_buffer(out_schema);
        }
        if total_work > u64::from(u32::MAX - 1) {
            return Err(XlogError::Kernel(format!(
                "{ctx}: candidate work {total_work} exceeds the u32 work-index space"
            )));
        }
        let total_work = total_work as u32;

        // ---- Phase 2: bitmap mark (candidates) + subtract (stable R).
        let mut bitmap = self.memory().alloc::<u32>(n_words as usize)?;
        self.device()
            .inner()
            .memset_zeros(&mut bitmap)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero bitmap failed: {e}")))?;
        let mut error_flag = self.memory().alloc::<u32>(1)?;
        self.device()
            .inner()
            .memset_zeros(&mut error_flag)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero error flag failed: {e}")))?;
        {
            let mut rec = LaunchRecorder::new_strict(launch_stream);
            rec.read_column(delta_x);
            rec.read_column(edge_z);
            rec.read(&range_lo);
            rec.read(&wp);
            rec.read_write(&bitmap);
            rec.write(&error_flag);
            rec.preflight(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: mark preflight failed: {e}")))?;
            let mark = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::FJ_DELTA_MARK_U32)
                .ok_or_else(|| {
                    XlogError::Kernel("fj_delta_mark_u32 kernel not found".to_string())
                })?;
            let grid = total_work.div_ceil(BLOCK_SIZE);
            // SAFETY: fj_delta_mark_u32(delta_x, n_delta, range_lo,
            // work_prefix, total_work, edge_z, bitmap, words_per_row,
            // domain, error_flag); buffers preflighted.
            unsafe {
                mark.clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &delta_x_v,
                            n_delta,
                            &range_lo,
                            &wp,
                            total_work,
                            &edge_z_v,
                            &mut bitmap,
                            words_per_row,
                            domain,
                            &mut error_flag,
                        ),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("fj_delta_mark_u32 launch failed: {e}"))
                    })?;
            }
            if n_r > 0 {
                let r_x_v = self.column_as_u32_view(r_x, n_r as usize)?;
                let r_z_v = self.column_as_u32_view(r_z, n_r as usize)?;
                rec.read_column(r_x);
                rec.read_column(r_z);
                let subtract = self
                    .device()
                    .inner()
                    .get_func(WCOJ_MODULE, wcoj_kernels::FJ_DELTA_SUBTRACT_U32)
                    .ok_or_else(|| {
                        XlogError::Kernel("fj_delta_subtract_u32 kernel not found".to_string())
                    })?;
                let grid = n_r.div_ceil(BLOCK_SIZE);
                // SAFETY: fj_delta_subtract_u32(r_x, r_z, n_r, bitmap,
                // words_per_row, domain, error_flag); same-stream launch
                // orders subtract after mark.
                unsafe {
                    subtract
                        .clone()
                        .launch_on_stream(
                            &cu_stream,
                            LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (BLOCK_SIZE, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &r_x_v,
                                &r_z_v,
                                n_r,
                                &mut bitmap,
                                words_per_row,
                                domain,
                                &mut error_flag,
                            ),
                        )
                        .map_err(|e| {
                            XlogError::Kernel(format!("fj_delta_subtract_u32 launch failed: {e}"))
                        })?;
                }
            }
            rec.commit(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: mark commit failed: {e}")))?;
        }
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: mark sync failed: {e}")))?;
        if self.dtoh_scalar_untracked::<u32>(&error_flag, 0)? != 0 {
            return Err(XlogError::Kernel(format!(
                "{ctx}: id outside domain {domain} (fail-closed; raise domain or \
                 renumber the fixture)"
            )));
        }

        // ---- Phase 3: popcount → scan → emit at scanned offsets.
        let mut counts = self.memory().alloc::<u32>(n_words as usize + 1)?;
        {
            let mut rec = LaunchRecorder::new_strict(launch_stream);
            rec.read(&bitmap);
            rec.write(&counts);
            rec.preflight(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: count preflight failed: {e}")))?;
            let popcount = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::FJ_DELTA_POPCOUNT)
                .ok_or_else(|| {
                    XlogError::Kernel("fj_delta_popcount kernel not found".to_string())
                })?;
            let grid = n_words.div_ceil(BLOCK_SIZE);
            // SAFETY: fj_delta_popcount(bitmap, n_words, counts).
            unsafe {
                popcount
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&bitmap, n_words, &mut counts),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("fj_delta_popcount launch failed: {e}"))
                    })?;
            }
            self.multiblock_scan_u32_inplace_on_stream(
                &mut counts,
                n_words + 1,
                &cu_stream,
                launch_stream,
                runtime,
            )?;
            rec.commit(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: count commit failed: {e}")))?;
        }
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: count sync failed: {e}")))?;
        let total_novel = self.dtoh_scalar_untracked::<u32>(&counts, n_words as usize)?;
        if total_novel == 0 {
            return self.create_empty_buffer(out_schema);
        }

        let out_x = self.memory().alloc::<u32>(total_novel as usize)?;
        let out_z = self.memory().alloc::<u32>(total_novel as usize)?;
        {
            let mut rec = LaunchRecorder::new_strict(launch_stream);
            rec.read(&bitmap);
            rec.read(&counts);
            rec.write(&out_x);
            rec.write(&out_z);
            rec.preflight(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: emit preflight failed: {e}")))?;
            let emit = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::FJ_DELTA_EMIT_U32)
                .ok_or_else(|| {
                    XlogError::Kernel("fj_delta_emit_u32 kernel not found".to_string())
                })?;
            let grid = n_words.div_ceil(BLOCK_SIZE);
            // SAFETY: fj_delta_emit_u32(bitmap, words_per_row, n_words,
            // offsets, out_x, out_z); offsets are the scanned counts.
            unsafe {
                emit.clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&bitmap, words_per_row, n_words, &counts, &out_x, &out_z),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("fj_delta_emit_u32 launch failed: {e}"))
                    })?;
            }
            rec.commit(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: emit commit failed: {e}")))?;
        }
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: emit sync failed: {e}")))?;

        let d_nr = self.memory().alloc::<u32>(1)?;
        self.htod_launch_metadata_async_copy_one(
            &total_novel,
            &d_nr,
            &cu_stream,
            &format!("{ctx}: result num_rows"),
        )?;
        Ok(CudaBuffer::from_columns_with_host_count(
            vec![out_x.into_bytes().into(), out_z.into_bytes().into()],
            u64::from(total_novel),
            d_nr,
            out_schema,
            total_novel,
        ))
    }
}
