//! D3 sparse-domain spike — factorized novel set via a GPU
//! open-addressing hash set, for large/sparse domains where the dense
//! bitvector (`domain²/8` bytes) is infeasible.
//!
//! Design: `docs/plans/2026-06-14-d3-sparse-domain-spike.md`. Same
//! semantics as the dense `fj_delta_novel_u32_recorded`
//! (`novel = delta ⋈ edge \ R`, full-row deduped) but evaluated over a
//! hash set keyed by `(x<<32)|z` instead of a characteristic
//! bitvector: duplicate witnesses and rediscoveries collapse at the
//! slot, so no witness-multiplied intermediate is materialized and
//! there is no `domain²` term. Output is unordered (slot scan);
//! callers needing lex order sort downstream (`union_gpu` does).
//!
//! Wired into the executor's domain-based router (sparse route above
//! the dense cap). Table capacity is `2×(|R| + distinct-candidate
//! estimate)`, where the estimate comes from a fixed 8 MiB estimator
//! bitmap (Phase 1b) — sized to DISTINCT keys, not the witness count,
//! so a witness-blowup workload does not over-provision the table.
//! Over the caller's `max_table_bytes` → `Ok(None)` (legacy fallback);
//! an estimate that under-sizes the table is caught by the
//! overflow-safe insert, which also declines to legacy.

use xlog_core::{Result, ScalarType, XlogError};

use super::{wcoj_kernels, CudaKernelProvider, WCOJ_MODULE};
use super::fj_delta::FjDeltaCols;
use crate::device_runtime::StreamId;
use crate::launch::LaunchRecorder;
use crate::CudaBuffer;
use crate::{LaunchAsync, LaunchConfig};

const BLOCK_SIZE: u32 = 256;

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
    /// Sparse-domain twin of [`Self::fj_delta_novel_u32_recorded`]: one
    /// factorized semi-naive delta step over a hash set, with no domain
    /// cap. Forbids the single key `(u32::MAX, u32::MAX)` (its packed
    /// `key+1` overflows the empty sentinel) — fails closed if present.
    ///
    /// Returns `Ok(None)` when the distinct-sized hash table
    /// (`2×(|R| + distinct-candidate estimate)`, power of two) would
    /// exceed `max_table_bytes`, or when an insert overflows an
    /// under-sized table — both are clean route-declines so the caller
    /// falls back to the legacy path. `max_table_bytes == 0` disables
    /// the budget guard (standalone spike/parity tests).
    pub fn fj_delta_sparse_novel_u32_recorded(
        &self,
        delta: &CudaBuffer,
        edge: &CudaBuffer,
        full_r: &CudaBuffer,
        cols: FjDeltaCols,
        max_table_bytes: u64,
        launch_stream: StreamId,
    ) -> Result<Option<CudaBuffer>> {
        let ctx = "fj_delta_sparse_novel_u32_recorded";
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(format!(
                "{ctx} requires a runtime-backed GpuMemoryManager (with_runtime)"
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
            return Ok(Some(self.create_empty_buffer(out_schema)?));
        }

        let delta_x = delta.column(cols.delta_carry).ok_or_else(|| {
            XlogError::Kernel(format!("{ctx}: delta column {} missing", cols.delta_carry))
        })?;
        let delta_y = delta.column(cols.delta_key).ok_or_else(|| {
            XlogError::Kernel(format!("{ctx}: delta column {} missing", cols.delta_key))
        })?;
        let edge_y = edge
            .column(0)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: edge column 0 missing")))?;
        let edge_z = edge
            .column(1)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: edge column 1 missing")))?;
        let r_x = full_r.column(cols.r_carry).ok_or_else(|| {
            XlogError::Kernel(format!("{ctx}: full_r column {} missing", cols.r_carry))
        })?;
        let r_z = full_r.column(cols.r_value).ok_or_else(|| {
            XlogError::Kernel(format!("{ctx}: full_r column {} missing", cols.r_value))
        })?;
        let delta_y_v = self.column_as_u32_view(delta_y, n_delta as usize)?;
        let edge_y_v = self.column_as_u32_view(edge_y, n_edge as usize)?;

        // ---- Phase 1: per-delta-row edge ranges + work prefix
        // (reuses the dense path's range kernel).
        let range_lo = self.memory().alloc::<u32>(n_delta as usize)?;
        let mut wp = self.memory().alloc::<u32>(n_delta as usize + 1)?;
        {
            let mut rec = LaunchRecorder::new_strict(launch_stream);
            rec.read_column(delta_y);
            rec.read_column(edge_y);
            rec.write(&range_lo);
            rec.write(&wp);
            rec.preflight(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: range preflight: {e}")))?;
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::FJ_DELTA_RANGE_U32)
                .ok_or_else(|| XlogError::Kernel("fj_delta_range_u32 not found".to_string()))?;
            let grid = n_delta.div_ceil(BLOCK_SIZE);
            // SAFETY: fj_delta_range_u32(delta_y, n_delta, edge_y, n_edge, range_lo, wp).
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
                    .map_err(|e| XlogError::Kernel(format!("fj_delta_range_u32 launch: {e}")))?;
            }
            self.multiblock_scan_u32_inplace_on_stream(&mut wp, n_delta + 1, &cu_stream, launch_stream, runtime)?;
            rec.commit(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: range commit: {e}")))?;
        }
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: range sync: {e}")))?;
        let total_work = u64::from(self.dtoh_scalar_untracked::<u32>(&wp, n_delta as usize)?);
        if total_work == 0 {
            return Ok(Some(self.create_empty_buffer(out_schema)?));
        }
        if total_work > u64::from(u32::MAX - 1) {
            return Err(XlogError::Kernel(format!(
                "{ctx}: candidate work {total_work} exceeds u32 work-index space"
            )));
        }
        let total_work = total_work as u32;

        // ---- Phase 1b: distinct-candidate estimate. Hash every
        // candidate key into a fixed 8 MiB estimator bitmap (2²⁶ bits)
        // and popcount it: that approximates the number of DISTINCT
        // (x,z) candidates, which (with |R|) sizes the real table —
        // sizing to the witness count `total_work` instead would
        // over-provision by the multiplicity factor (the parked S4
        // peak regression). Collisions undercount slightly; the host
        // adds margin and the overflow-safe insert is the backstop.
        const EST_BITS: u32 = 1 << 26;
        const EST_WORDS: u32 = EST_BITS / 32;
        let est_bit_mask = EST_BITS - 1;
        let mut est = self.memory().alloc::<u32>(EST_WORDS as usize)?;
        self.device()
            .inner()
            .memset_zeros(&mut est)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero estimator: {e}")))?;
        let mut est_counts = self.memory().alloc::<u32>(EST_WORDS as usize + 1)?;
        {
            let mut rec = LaunchRecorder::new_strict(launch_stream);
            rec.read_column(delta_x);
            rec.read_column(edge_z);
            rec.read(&range_lo);
            rec.read(&wp);
            rec.read_write(&est);
            rec.write(&est_counts);
            rec.preflight(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: estimate preflight: {e}")))?;
            let delta_x_v = self.column_as_u32_view(delta_x, n_delta as usize)?;
            let edge_z_v = self.column_as_u32_view(edge_z, n_edge as usize)?;
            let estimate = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::FJ_DELTA_SPARSE_ESTIMATE)
                .ok_or_else(|| {
                    XlogError::Kernel("fj_delta_sparse_estimate not found".to_string())
                })?;
            let grid = total_work.div_ceil(BLOCK_SIZE);
            // SAFETY: fj_delta_sparse_estimate(delta_x, n_delta, range_lo,
            // wp, total_work, edge_z, est_bitmap, est_bit_mask).
            unsafe {
                estimate
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&delta_x_v, n_delta, &range_lo, &wp, total_work, &edge_z_v, &mut est, est_bit_mask),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("fj_delta_sparse_estimate launch: {e}"))
                    })?;
            }
            // popcount the estimator (reuses the dense popcount kernel)
            // → exclusive scan → total set bits at [EST_WORDS].
            let popcount = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::FJ_DELTA_POPCOUNT)
                .ok_or_else(|| XlogError::Kernel("fj_delta_popcount not found".to_string()))?;
            let pgrid = EST_WORDS.div_ceil(BLOCK_SIZE);
            // SAFETY: fj_delta_popcount(bitmap, n_words, counts).
            unsafe {
                popcount
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (pgrid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&est, EST_WORDS, &mut est_counts),
                    )
                    .map_err(|e| XlogError::Kernel(format!("estimator popcount launch: {e}")))?;
            }
            self.multiblock_scan_u32_inplace_on_stream(
                &mut est_counts,
                EST_WORDS + 1,
                &cu_stream,
                launch_stream,
                runtime,
            )?;
            rec.commit(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: estimate commit: {e}")))?;
        }
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: estimate sync: {e}")))?;
        let distinct_est = self.dtoh_scalar_untracked::<u32>(&est_counts, EST_WORDS as usize)?;
        // Free the estimator + its scan before sizing the real table so
        // they don't inflate peak (they are sizing scaffolding only).
        drop(est);
        drop(est_counts);

        // ---- Table sizing: power-of-two capacity ≥ 2×(|R| + distinct
        // estimate × margin). Margin 3/2 absorbs estimator-collision
        // undercount; load factor ≤ 0.5. Sized to DISTINCT keys, not
        // the witness count — this is the fix for the parked S4 peak
        // regression. The overflow-safe insert backstops a bad estimate.
        let est_margined = (u64::from(distinct_est) * 3) / 2;
        let upper = u64::from(n_r) + est_margined + 1;
        let want = upper
            .checked_mul(2)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: table size overflow")))?;
        let mut cap: u64 = 1;
        while cap < want {
            cap <<= 1;
        }
        if cap > u64::from(u32::MAX) {
            return Err(XlogError::Kernel(format!(
                "{ctx}: hash table capacity {cap} exceeds u32 slot space (workload too large \
                 for the spike's single-table sizing)"
            )));
        }
        // Route-decline guard: the table (u64 keys + u8 is_r) plus the
        // scan counts (u32, cap+1) must fit the caller's budget; over
        // budget → decline so the caller uses the legacy path.
        if max_table_bytes != 0 {
            let table_bytes = u64::from(cap)
                .saturating_mul(8 + 1 + 4)
                .saturating_add(4);
            if table_bytes > max_table_bytes {
                return Ok(None);
            }
        }
        let cap = cap as u32;
        let mask = cap - 1;

        let mut table = self.memory().alloc::<u64>(cap as usize)?;
        let mut is_r = self.memory().alloc::<u8>(cap as usize)?;
        self.device()
            .inner()
            .memset_zeros(&mut table)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero table: {e}")))?;
        self.device()
            .inner()
            .memset_zeros(&mut is_r)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero is_r: {e}")))?;
        let mut overflow = self.memory().alloc::<u32>(1)?;
        self.device()
            .inner()
            .memset_zeros(&mut overflow)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero overflow: {e}")))?;

        // ---- Phase 2: load R (marks is_r), then insert candidates.
        {
            let mut rec = LaunchRecorder::new_strict(launch_stream);
            rec.read_column(delta_x);
            rec.read_column(edge_z);
            rec.read(&range_lo);
            rec.read(&wp);
            rec.read_write(&table);
            rec.read_write(&is_r);
            rec.write(&overflow);
            if n_r > 0 {
                rec.read_column(r_x);
                rec.read_column(r_z);
            }
            rec.preflight(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: insert preflight: {e}")))?;

            if n_r > 0 {
                let r_x_v = self.column_as_u32_view(r_x, n_r as usize)?;
                let r_z_v = self.column_as_u32_view(r_z, n_r as usize)?;
                let load_r = self
                    .device()
                    .inner()
                    .get_func(WCOJ_MODULE, wcoj_kernels::FJ_DELTA_SPARSE_LOAD_R)
                    .ok_or_else(|| {
                        XlogError::Kernel("fj_delta_sparse_load_r not found".to_string())
                    })?;
                let grid = n_r.div_ceil(BLOCK_SIZE);
                // SAFETY: fj_delta_sparse_load_r(r_x, r_z, n_r, table, is_r, mask, overflow).
                unsafe {
                    load_r
                        .clone()
                        .launch_on_stream(
                            &cu_stream,
                            LaunchConfig {
                                grid_dim: (grid, 1, 1),
                                block_dim: (BLOCK_SIZE, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (&r_x_v, &r_z_v, n_r, &mut table, &mut is_r, mask, &mut overflow),
                        )
                        .map_err(|e| {
                            XlogError::Kernel(format!("fj_delta_sparse_load_r launch: {e}"))
                        })?;
                }
            }

            let delta_x_v = self.column_as_u32_view(delta_x, n_delta as usize)?;
            let edge_z_v = self.column_as_u32_view(edge_z, n_edge as usize)?;
            let insert = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::FJ_DELTA_SPARSE_INSERT_CANDIDATES)
                .ok_or_else(|| {
                    XlogError::Kernel("fj_delta_sparse_insert_candidates not found".to_string())
                })?;
            let grid = total_work.div_ceil(BLOCK_SIZE);
            // SAFETY: fj_delta_sparse_insert_candidates(delta_x, n_delta,
            // range_lo, wp, total_work, edge_z, table, mask, overflow).
            unsafe {
                insert
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&delta_x_v, n_delta, &range_lo, &wp, total_work, &edge_z_v, &mut table, mask, &mut overflow),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("fj_delta_sparse_insert_candidates launch: {e}"))
                    })?;
            }
            rec.commit(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: insert commit: {e}")))?;
        }
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: insert sync: {e}")))?;

        // Overflow backstop: if the distinct estimate under-sized the
        // table and an insert exhausted its probe budget, the table may
        // hold partial results — decline to the legacy path rather than
        // emit a wrong (incomplete) novel set.
        if self.dtoh_scalar_untracked::<u32>(&overflow, 0)? != 0 {
            return Ok(None);
        }

        // ---- Phase 3: mark novel slots → scan → emit.
        let mut counts = self.memory().alloc::<u32>(cap as usize + 1)?;
        {
            let mut rec = LaunchRecorder::new_strict(launch_stream);
            rec.read(&table);
            rec.read(&is_r);
            rec.write(&counts);
            rec.preflight(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: mark preflight: {e}")))?;
            let mark = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::FJ_DELTA_SPARSE_MARK)
                .ok_or_else(|| XlogError::Kernel("fj_delta_sparse_mark not found".to_string()))?;
            let grid = cap.div_ceil(BLOCK_SIZE);
            // SAFETY: fj_delta_sparse_mark(table, is_r, cap, counts).
            unsafe {
                mark.clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&table, &is_r, cap, &mut counts),
                    )
                    .map_err(|e| XlogError::Kernel(format!("fj_delta_sparse_mark launch: {e}")))?;
            }
            self.multiblock_scan_u32_inplace_on_stream(&mut counts, cap + 1, &cu_stream, launch_stream, runtime)?;
            rec.commit(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: mark commit: {e}")))?;
        }
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: mark sync: {e}")))?;
        let total_novel = self.dtoh_scalar_untracked::<u32>(&counts, cap as usize)?;
        if total_novel == 0 {
            return Ok(Some(self.create_empty_buffer(out_schema)?));
        }

        let out_x = self.memory().alloc::<u32>(total_novel as usize)?;
        let out_z = self.memory().alloc::<u32>(total_novel as usize)?;
        {
            let mut rec = LaunchRecorder::new_strict(launch_stream);
            rec.read(&table);
            rec.read(&is_r);
            rec.read(&counts);
            rec.write(&out_x);
            rec.write(&out_z);
            rec.preflight(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: emit preflight: {e}")))?;
            let emit = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::FJ_DELTA_SPARSE_EMIT)
                .ok_or_else(|| XlogError::Kernel("fj_delta_sparse_emit not found".to_string()))?;
            let grid = cap.div_ceil(BLOCK_SIZE);
            // SAFETY: fj_delta_sparse_emit(table, is_r, offsets, cap, out_x, out_z).
            unsafe {
                emit.clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&table, &is_r, &counts, cap, &out_x, &out_z),
                    )
                    .map_err(|e| XlogError::Kernel(format!("fj_delta_sparse_emit launch: {e}")))?;
            }
            rec.commit(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: emit commit: {e}")))?;
        }
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: emit sync: {e}")))?;

        let d_nr = self.memory().alloc::<u32>(1)?;
        self.htod_launch_metadata_async_copy_one(
            &total_novel,
            &d_nr,
            &cu_stream,
            &format!("{ctx}: result num_rows"),
        )?;
        let columns = if cols.r_carry == 0 {
            vec![out_x.into_bytes().into(), out_z.into_bytes().into()]
        } else {
            vec![out_z.into_bytes().into(), out_x.into_bytes().into()]
        };
        Ok(Some(CudaBuffer::from_columns_with_host_count(
            columns,
            u64::from(total_novel),
            d_nr,
            out_schema,
            total_novel,
        )))
    }
}
