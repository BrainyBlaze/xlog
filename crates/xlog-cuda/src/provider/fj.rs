//! GPU Free Join provider: level-synchronous factorized join execution.
//!
//! Design: `docs/plans/2026-06-12-d2-free-join-design.md`. The paper's
//! (Wang/Willsey/Suciu, SIGMOD 2023) depth-first recursion over lazy
//! hash tries is replaced by:
//!
//!   * **Flat sorted-range tries** (§2.1): every input is
//!     layout-normalized (lex-sorted + deduped, the existing WCOJ
//!     layout); a trie node is a contiguous `[lo, hi)` row range and
//!     `get(key)` is a binary-search refinement of that range on the
//!     next column. No per-level structure is ever built.
//!   * **Level-synchronous frontier execution** (§2.2): a bindings
//!     frontier (SoA u32 columns: bound variables plus per-live-atom
//!     `(lo, hi)` range pairs) is rebuilt per plan node by bulk
//!     two-phase EXPAND (count → device scan → emit) over the node's
//!     cover subatom, followed by one PROBE refinement kernel per
//!     probe subatom and a single mask compaction (reusing the
//!     existing mask + scan + gather kernels).
//!
//! Invariants (§2.3, non-negotiable):
//!   * all inputs layout-normalized per dispatch;
//!   * no atomics in any emit path — output positions come from
//!     exclusive scans, so output order is deterministic (parent-row
//!     order × lex order of the plan's variable sequence);
//!   * set semantics by construction: distinct cover groups over
//!     deduped inputs keep the frontier duplicate-free; a final dedup
//!     runs only when the head projects a strict subset of the bound
//!     variables;
//!   * zero tracked transfers — host reads are limited to the
//!     sanctioned `dtoh_scalar_untracked` metadata scalars (scan
//!     totals, compaction counts); recorded launches throughout.
//!
//! Width classes: u32/Symbol (`free_join_execute_u32_recorded`) and
//! u64 (`free_join_execute_u64_recorded`) share one
//! width-parameterized pipeline — frontier VAR columns carry
//! width-sized data values while RANGE columns are u32 row indices in
//! every width class (the staging/compaction/projection helpers are
//! schema-driven per column, so mixed-width frontiers need no special
//! casing). Full expansion runs at the last node (the factorized
//! trailing-range enumeration of §2.4 remains future work).

use std::ffi::c_void;

use xlog_core::{Result, ScalarType, Schema, XlogError};

use super::{wcoj_kernels, CudaKernelProvider, WCOJ_MODULE};
use crate::device_runtime::StreamId;
use crate::launch::LaunchRecorder;
use crate::memory::{CudaColumn, TrackedCudaSlice};
use crate::CudaBuffer;
use crate::{AsKernelParam, LaunchAsync, LaunchConfig};

const BLOCK_SIZE: u32 = 256;

/// One subatom: an atom (`input_idx`) restricted to the variables its
/// next `var_positions.len()` physical columns bind/probe. Across the
/// whole plan, each atom's subatoms consume its columns in order and
/// must partition them exactly (design §3).
#[derive(Debug, Clone)]
pub struct FjSubAtom {
    /// Index into the `inputs` slice of
    /// [`CudaKernelProvider::free_join_execute_u32_recorded`].
    pub input_idx: usize,
    /// Global variable ids bound (cover) or matched (probe) by this
    /// subatom's columns, in column order.
    pub var_positions: Vec<usize>,
}

/// One plan node: iterate the cover subatom (bulk EXPAND over the
/// whole frontier), then refine every probe subatom (PROBE +
/// compaction). Probe variables must already be bound.
#[derive(Debug, Clone)]
pub struct FjNode {
    pub cover: FjSubAtom,
    pub probes: Vec<FjSubAtom>,
}

/// A host-side Free Join plan over `inputs` (design §3). Callers hand-build
/// the plan today; planner construction from binary joins (`binary2fj`) is a
/// downstream integration surface.
#[derive(Debug, Clone)]
pub struct FjPlan {
    /// Number of distinct join variables (ids `0..num_vars`).
    pub num_vars: usize,
    /// Plan nodes, executed in order.
    pub nodes: Vec<FjNode>,
    /// Head projection: variable ids in output column order. When
    /// this is a strict subset of the bound variables the result is
    /// deduplicated (set semantics).
    pub output_vars: Vec<usize>,
}

/// Frontier column tag: which logical value a SoA column holds. The
/// live-range set is tracked statically per node (design §5.1) —
/// exhausted atoms' range columns are dropped, untouched atoms have
/// no columns at all (their range is the constant `[0, n)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ColTag {
    /// Bound value of a join variable.
    Var(usize),
    /// Current trie-range lower bound for an atom.
    RangeLo(usize),
    /// Current trie-range upper bound for an atom.
    RangeHi(usize),
}

type FrontierCol = (ColTag, TrackedCudaSlice<u8>);

/// Data width class of a Free Join execution. Row indices, trie
/// ranges, work prefixes, and group marks are u32 in every class;
/// only DATA columns (cover/probe/var values) take this width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FjWidth {
    U32,
    U64,
}

impl FjWidth {
    fn var_bytes(self) -> usize {
        match self {
            Self::U32 => std::mem::size_of::<u32>(),
            Self::U64 => std::mem::size_of::<u64>(),
        }
    }

    fn var_type(self) -> ScalarType {
        match self {
            Self::U32 => ScalarType::U32,
            Self::U64 => ScalarType::U64,
        }
    }

    fn count_kernel(self) -> &'static str {
        match self {
            Self::U32 => wcoj_kernels::FJ_EXPAND_COUNT_U32,
            Self::U64 => wcoj_kernels::FJ_EXPAND_COUNT_U64,
        }
    }

    fn emit_kernel(self) -> &'static str {
        match self {
            Self::U32 => wcoj_kernels::FJ_EXPAND_EMIT_U32,
            Self::U64 => wcoj_kernels::FJ_EXPAND_EMIT_U64,
        }
    }

    fn probe_kernel(self) -> &'static str {
        match self {
            Self::U32 => wcoj_kernels::FJ_PROBE_REFINE_U32,
            Self::U64 => wcoj_kernels::FJ_PROBE_REFINE_U64,
        }
    }
}

/// Per-tag column type within a frontier of the given width: VAR
/// columns are width-sized data, RANGE columns are u32 row indices.
fn tag_type(tag: ColTag, width: FjWidth) -> ScalarType {
    match tag {
        ColTag::Var(_) => width.var_type(),
        ColTag::RangeLo(_) | ColTag::RangeHi(_) => ScalarType::U32,
    }
}

/// What the pipeline produces after the last plan node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FjMode {
    /// Materialize the projected row set (`output_vars` columns).
    Materialize,
    /// Design §2.4 factorized count: reduce to
    /// `(output_vars[0], count)` where each frontier row contributes
    /// the PRODUCT of its remaining live trie-range lengths
    /// (unconsumed trailing columns never expand). Plans may
    /// partially consume atoms, but every atom must be touched.
    CountByRoot,
}

fn owned_col_ptr(buf: &CudaBuffer, idx: usize, ctx: &str) -> Result<u64> {
    match buf.column(idx) {
        Some(CudaColumn::Owned(s)) => Ok(*s.device_ptr()),
        Some(_) => Err(XlogError::Kernel(format!(
            "{ctx}: input column {idx} must be owned"
        ))),
        None => Err(XlogError::Kernel(format!(
            "{ctx}: input column {idx} not found"
        ))),
    }
}

fn find_col<'a>(cols: &'a [FrontierCol], tag: ColTag, ctx: &str) -> Result<&'a TrackedCudaSlice<u8>> {
    cols.iter()
        .find(|(t, _)| *t == tag)
        .map(|(_, s)| s)
        .ok_or_else(|| XlogError::Kernel(format!("{ctx}: frontier column {tag:?} missing")))
}

/// Validate the plan against the input arities. Returns the variable
/// binding order (covers' variables in plan order).
fn validate_plan(
    plan: &FjPlan,
    arities: &[usize],
    mode: FjMode,
    ctx: &str,
) -> Result<Vec<usize>> {
    if plan.nodes.is_empty() {
        return Err(XlogError::Kernel(format!("{ctx}: plan has no nodes")));
    }
    let mut bound = vec![false; plan.num_vars];
    let mut bind_order: Vec<usize> = Vec::new();
    let mut consumed = vec![0usize; arities.len()];
    let check_sub = |sub: &FjSubAtom, consumed: &[usize], what: &str| -> Result<()> {
        if sub.input_idx >= arities.len() {
            return Err(XlogError::Kernel(format!(
                "{ctx}: {what} input_idx {} out of bounds ({} inputs)",
                sub.input_idx,
                arities.len()
            )));
        }
        if sub.var_positions.is_empty() {
            return Err(XlogError::Kernel(format!(
                "{ctx}: {what} on input {} has no variables",
                sub.input_idx
            )));
        }
        if consumed[sub.input_idx] + sub.var_positions.len() > arities[sub.input_idx] {
            return Err(XlogError::Kernel(format!(
                "{ctx}: {what} over-consumes input {} (arity {}, consumed {}, +{})",
                sub.input_idx,
                arities[sub.input_idx],
                consumed[sub.input_idx],
                sub.var_positions.len()
            )));
        }
        for &v in &sub.var_positions {
            if v >= plan.num_vars {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: {what} variable {v} out of bounds (num_vars {})",
                    plan.num_vars
                )));
            }
        }
        Ok(())
    };
    for (k, node) in plan.nodes.iter().enumerate() {
        check_sub(&node.cover, &consumed, "cover")?;
        for &v in &node.cover.var_positions {
            if bound[v] {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: node {k} cover rebinds variable {v}"
                )));
            }
            bound[v] = true;
            bind_order.push(v);
        }
        consumed[node.cover.input_idx] += node.cover.var_positions.len();
        let mut seen_atoms = vec![node.cover.input_idx];
        for probe in &node.probes {
            check_sub(probe, &consumed, "probe")?;
            if seen_atoms.contains(&probe.input_idx) {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: node {k} touches input {} more than once",
                    probe.input_idx
                )));
            }
            seen_atoms.push(probe.input_idx);
            for &v in &probe.var_positions {
                if !bound[v] {
                    return Err(XlogError::Kernel(format!(
                        "{ctx}: node {k} probes unbound variable {v}"
                    )));
                }
            }
            consumed[probe.input_idx] += probe.var_positions.len();
        }
    }
    for (i, (&used, &arity)) in consumed.iter().zip(arities.iter()).enumerate() {
        match mode {
            FjMode::Materialize => {
                if used != arity {
                    return Err(XlogError::Kernel(format!(
                        "{ctx}: plan consumes {used}/{arity} columns of input {i} \
                         (materialization requires full consumption)"
                    )));
                }
            }
            // §2.4 factorized counting: unconsumed trailing columns
            // contribute their live range lengths as multiplicities,
            // but an untouched atom has no range to read — reject.
            FjMode::CountByRoot => {
                if used == 0 {
                    return Err(XlogError::Kernel(format!(
                        "{ctx}: count plan never touches input {i} \
                         (untouched atoms have no live range)"
                    )));
                }
            }
        }
    }
    if plan.output_vars.is_empty() {
        return Err(XlogError::Kernel(format!("{ctx}: empty output_vars")));
    }
    if mode == FjMode::CountByRoot && plan.output_vars.len() != 1 {
        return Err(XlogError::Kernel(format!(
            "{ctx}: count plans take exactly one output (group) variable, got {}",
            plan.output_vars.len()
        )));
    }
    for &v in &plan.output_vars {
        if v >= plan.num_vars || !bound[v] {
            return Err(XlogError::Kernel(format!(
                "{ctx}: output variable {v} is never bound"
            )));
        }
    }
    Ok(bind_order)
}

impl CudaKernelProvider {
    /// Execute a hand-built Free Join plan over u32/Symbol relations
    /// via the level-synchronous frontier engine. See the module docs
    /// for the algorithm and invariants; the plan contract is
    /// documented on [`FjPlan`].
    ///
    /// Inputs are layout-normalized per dispatch (sorted + deduped via
    /// the existing WCOJ layout entries — already-normalized inputs
    /// take the recorded fast-path check). The output contains one
    /// column per `output_vars` entry (all `U32`) holding the join's
    /// projected row set under set semantics.
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime, the launch
    ///   stream does not resolve, an input violates the u32
    ///   width-class layout contract, the plan is invalid (unbound
    ///   probe variables, over/under-consumed atom columns, rebound
    ///   variables, unknown output variables), the frontier exceeds
    ///   the u32 work-index space, or any kernel launch fails.
    pub fn free_join_execute_u32_recorded(
        &self,
        inputs: &[&CudaBuffer],
        plan: &FjPlan,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.free_join_execute_recorded_impl(
            inputs,
            plan,
            launch_stream,
            FjWidth::U32,
            FjMode::Materialize,
            "free_join_execute_u32_recorded",
        )
    }

    /// u64 width-class twin of [`Self::free_join_execute_u32_recorded`]:
    /// identical pipeline, contract, and invariants; every
    /// input column must be `U64` and the output columns are `U64`.
    pub fn free_join_execute_u64_recorded(
        &self,
        inputs: &[&CudaBuffer],
        plan: &FjPlan,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.free_join_execute_recorded_impl(
            inputs,
            plan,
            launch_stream,
            FjWidth::U64,
            FjMode::Materialize,
            "free_join_execute_u64_recorded",
        )
    }

    /// Design §2.4 factorized count-by-root over the Free Join
    /// frontier: runs the same pipeline but reduces to
    /// `(group, count)` instead of materializing rows. The plan's
    /// `output_vars` must be exactly `[group_var]`; atoms may be
    /// PARTIALLY consumed — each surviving frontier row contributes
    /// the product of its remaining live trie-range lengths (the
    /// d-representation count), so trailing private variables never
    /// expand the frontier. Output schema: `(group: U32, count: U64)`.
    ///
    /// u32/Symbol width-class only: the reduction reuses the recorded
    /// groupby, whose KEY columns are bounded engine-wide to
    /// U32/Symbol (multi-type recorded sort is deferred there) — u64
    /// bodies stay on the materialize path.
    pub fn free_join_count_by_root_u32_recorded(
        &self,
        inputs: &[&CudaBuffer],
        plan: &FjPlan,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        self.free_join_execute_recorded_impl(
            inputs,
            plan,
            launch_stream,
            FjWidth::U32,
            FjMode::CountByRoot,
            "free_join_count_by_root_u32_recorded",
        )
    }

    #[allow(clippy::too_many_lines)]
    fn free_join_execute_recorded_impl(
        &self,
        inputs: &[&CudaBuffer],
        plan: &FjPlan,
        launch_stream: StreamId,
        width: FjWidth,
        mode: FjMode,
        ctx: &str,
    ) -> Result<CudaBuffer> {
        if self.memory().runtime().is_none() {
            return Err(XlogError::Kernel(format!(
                "{ctx} requires a runtime-backed GpuMemoryManager \
                 (constructed via with_runtime)"
            )));
        }
        if inputs.is_empty() {
            return Err(XlogError::Kernel(format!("{ctx}: no inputs")));
        }
        let arities: Vec<usize> = inputs.iter().map(|b| b.arity()).collect();
        let bind_order = validate_plan(plan, &arities, mode, ctx)?;

        // Layout-normalize every input per dispatch. Arity-2 inputs
        // go through the triangle-grade
        // entry (it has the sorted+unique recorded fast-path);
        // wider inputs use the generic full-row WCOJ sort+dedup entry.
        let mut norm: Vec<CudaBuffer> = Vec::with_capacity(inputs.len());
        for input in inputs {
            let normalized = match (width, input.arity()) {
                (FjWidth::U32, 2) => self.wcoj_layout_u32_recorded(input, launch_stream)?,
                (FjWidth::U32, _) => self.wcoj_layout_sort_u32_recorded(input, launch_stream)?,
                (FjWidth::U64, 2) => self.wcoj_layout_u64_recorded(input, launch_stream)?,
                (FjWidth::U64, _) => self.wcoj_layout_sort_u64_recorded(input, launch_stream)?,
            };
            norm.push(normalized);
        }
        let mut n_rows: Vec<u32> = Vec::with_capacity(norm.len());
        for buf in &norm {
            let n = match buf.cached_row_count() {
                Some(c) => c,
                None => self.dtoh_scalar_untracked::<u32>(buf.num_rows_device(), 0)?,
            };
            n_rows.push(n);
        }

        let out_schema = match mode {
            FjMode::Materialize => Schema::new(
                plan.output_vars
                    .iter()
                    .map(|v| (format!("v{v}"), width.var_type()))
                    .collect(),
            ),
            FjMode::CountByRoot => Schema::new(vec![
                (format!("v{}", plan.output_vars[0]), width.var_type()),
                ("count".to_string(), ScalarType::U64),
            ]),
        };
        // Inner-join semantics: any empty atom empties the result.
        if n_rows.iter().any(|&n| n == 0) {
            return self.create_empty_buffer(out_schema);
        }

        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(format!("{ctx} requires a runtime-backed GpuMemoryManager"))
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

        // Frontier: starts as the single empty binding with every
        // atom untouched (constant range [0, n)).
        let mut frontier: Vec<FrontierCol> = Vec::new();
        let mut count: u32 = 1;
        // Column CAPACITY of the current frontier (the last node's
        // n_children): compaction shrinks the logical count without
        // reallocating, and every CudaBuffer built over frontier
        // columns must carry row_cap == capacity (columns are
        // row_cap × elem bytes by contract), with the logical count
        // riding on num_rows_device.
        let mut frontier_cap: u32 = 1;
        let mut consumed = vec![0usize; inputs.len()];

        for node in &plan.nodes {
            let a = node.cover.input_idx;
            let c = node.cover.var_positions.len();
            let depth = consumed[a];
            let cover_live = frontier.iter().any(|(t, _)| *t == ColTag::RangeLo(a));

            // ---- EXPAND phase 0: total candidate work. Live cover
            // atoms need a per-row work prefix (range lengths →
            // exclusive scan); untouched cover atoms share the
            // constant range, so the mapping is uniform and the
            // total is known on host.
            let (total_work, work_prefix) = if cover_live {
                let mut wp = self.memory().alloc::<u32>(count as usize + 1)?;
                let lo_col = find_col(&frontier, ColTag::RangeLo(a), ctx)?;
                let hi_col = find_col(&frontier, ColTag::RangeHi(a), ctx)?;
                let mut rec = LaunchRecorder::new_strict(launch_stream);
                rec.read(lo_col);
                rec.read(hi_col);
                rec.write(&wp);
                rec.preflight(runtime)
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: wp preflight failed: {e}")))?;
                let kernel = self
                    .device()
                    .inner()
                    .get_func(WCOJ_MODULE, wcoj_kernels::FJ_EXPAND_WORK_PREFIX_U32)
                    .ok_or_else(|| {
                        XlogError::Kernel("fj_expand_work_prefix_u32 kernel not found".to_string())
                    })?;
                let grid = count.div_ceil(BLOCK_SIZE);
                // SAFETY: fj_expand_work_prefix_u32(parent_lo,
                // parent_hi, n_frontier, work_prefix); buffers are
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
                            (lo_col, hi_col, count, &mut wp),
                        )
                        .map_err(|e| {
                            XlogError::Kernel(format!(
                                "fj_expand_work_prefix_u32 launch failed: {e}"
                            ))
                        })?;
                }
                self.multiblock_scan_u32_inplace_on_stream(
                    &mut wp,
                    count + 1,
                    &cu_stream,
                    launch_stream,
                    runtime,
                )?;
                rec.commit(runtime)
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: wp commit failed: {e}")))?;
                cu_stream
                    .synchronize()
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: wp sync failed: {e}")))?;
                let total = self.dtoh_scalar_untracked::<u32>(&wp, count as usize)?;
                (u64::from(total), Some(wp))
            } else {
                ((count as u64) * (n_rows[a] as u64), None)
            };
            if total_work == 0 {
                return self.create_empty_buffer(out_schema);
            }
            if total_work > u64::from(u32::MAX - 1) {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: expansion work {total_work} exceeds the u32 work-index \
                     space (frontier budget)"
                )));
            }
            let total_work = total_work as u32;

            // ---- EXPAND phase 1: mark distinct cover-prefix group
            // starts and scan them into output offsets.
            //
            // Identity-group fast path: when the cover consumes through
            // the atom's LAST column, rows within any trie range have
            // distinct column suffixes (inputs are full-row deduped and
            // the range fixes all preceding columns), so every candidate
            // position is its own group — the marks pass, its device
            // scan, and its host sync are skipped and
            // n_children == total_work (the emit kernel takes its
            // out == w branch via a null group_offsets pointer).
            let identity = depth + c >= arities[a];
            // ---- Fused-probe analysis: a probe folds into the count
            // pass iff (a) its key variables are all bound by THIS
            // node's cover (the kernel reads keys from cover_cols at
            // the candidate position; earlier bindings are already
            // encoded in the probe's carried range), and (b) it
            // consumes through its atom's last column (existence-only —
            // no refined range survives for later nodes). Fused probes
            // skip the separate probe kernel and the mask compaction
            // entirely: the emit pass materializes exactly the
            // surviving children.
            let is_fusable = |pr: &FjSubAtom| {
                consumed[pr.input_idx] + pr.var_positions.len() >= arities[pr.input_idx]
                    && pr
                        .var_positions
                        .iter()
                        .all(|v| node.cover.var_positions.contains(v))
            };
            let fused: Vec<&FjSubAtom> = node.probes.iter().filter(|p| is_fusable(p)).collect();
            // The count pass runs when groups need marking (non-identity
            // cover) OR fused probes need evaluating.
            let count_ran = !identity || !fused.is_empty();
            // Pack one descriptor per fused probe, sequentially:
            // [n_cols, has_range, in_lo_ptr, in_hi_ptr, n_atom_rows,
            //  data_col_ptr * n_cols, cover_var_idx * n_cols].
            let mut fused_desc: Vec<u64> = Vec::new();
            for pr in &fused {
                let p = pr.input_idx;
                let live = frontier.iter().any(|(t, _)| *t == ColTag::RangeLo(p));
                fused_desc.push(pr.var_positions.len() as u64);
                fused_desc.push(u64::from(live));
                if live {
                    fused_desc.push(*find_col(&frontier, ColTag::RangeLo(p), ctx)?.device_ptr());
                    fused_desc.push(*find_col(&frontier, ColTag::RangeHi(p), ctx)?.device_ptr());
                } else {
                    fused_desc.push(0);
                    fused_desc.push(0);
                }
                fused_desc.push(u64::from(n_rows[p]));
                for i in consumed[p]..consumed[p] + pr.var_positions.len() {
                    fused_desc.push(owned_col_ptr(&norm[p], i, ctx)?);
                }
                for v in &pr.var_positions {
                    fused_desc.push(
                        node.cover
                            .var_positions
                            .iter()
                            .position(|cv| cv == v)
                            .expect("fusable probe keys are cover variables")
                            as u64,
                    );
                }
            }
            let d_fused_desc: Option<TrackedCudaSlice<u64>> = if fused_desc.is_empty() {
                None
            } else {
                let mut tbl = self.memory().alloc::<u64>(fused_desc.len())?;
                self.htod_launch_metadata_sync_copy_into(&fused_desc, &mut tbl)
                    .map_err(|e| {
                        XlogError::Kernel(format!("{ctx}: htod fused-probe table failed: {e}"))
                    })?;
                Some(tbl)
            };
            let cover_ptrs: Vec<u64> = (depth..depth + c)
                .map(|i| owned_col_ptr(&norm[a], i, ctx))
                .collect::<Result<_>>()?;
            let mut d_cover_tbl = self.memory().alloc::<u64>(c)?;
            self.htod_launch_metadata_sync_copy_into(&cover_ptrs, &mut d_cover_tbl)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: htod cover table failed: {e}")))?;
            let mut marks = self.memory().alloc::<u32>(total_work as usize + 1)?;
            let has_parent_range: u32 = u32::from(cover_live);
            let const_lo: u32 = 0;
            let const_hi: u32 = n_rows[a];
            let null_ptr: u64 = 0;
            if count_ran {
                let mut rec = LaunchRecorder::new_strict(launch_stream);
                rec.read(norm[a].num_rows_device());
                for i in depth..depth + c {
                    rec.read_column(norm[a].column(i).expect("validated cover column"));
                }
                rec.read(&d_cover_tbl);
                if let Some(wp) = work_prefix.as_ref() {
                    rec.read(wp);
                    rec.read(find_col(&frontier, ColTag::RangeLo(a), ctx)?);
                }
                if let Some(d) = d_fused_desc.as_ref() {
                    rec.read(d);
                    for pr in &fused {
                        let p = pr.input_idx;
                        rec.read(norm[p].num_rows_device());
                        for i in consumed[p]..consumed[p] + pr.var_positions.len() {
                            rec.read_column(norm[p].column(i).expect("validated probe column"));
                        }
                        if frontier.iter().any(|(t, _)| *t == ColTag::RangeLo(p)) {
                            rec.read(find_col(&frontier, ColTag::RangeLo(p), ctx)?);
                            rec.read(find_col(&frontier, ColTag::RangeHi(p), ctx)?);
                        }
                    }
                }
                rec.write(&marks);
                rec.preflight(runtime).map_err(|e| {
                    XlogError::Kernel(format!("{ctx}: count preflight failed: {e}"))
                })?;
                let kernel = self
                    .device()
                    .inner()
                    .get_func(WCOJ_MODULE, width.count_kernel())
                    .ok_or_else(|| {
                        XlogError::Kernel(format!("{} kernel not found", width.count_kernel()))
                    })?;
                let grid = total_work.div_ceil(BLOCK_SIZE);
                let c_u32 = c as u32;
                let identity_u32: u32 = u32::from(identity);
                let n_fused_u32: u32 = fused.len() as u32;
                // SAFETY: fj_expand_count_u32(cover_cols,
                // n_cover_cols, parent_lo, work_prefix,
                // has_parent_range, const_lo, const_hi, n_frontier,
                // total_work, group_marks). parent_lo/work_prefix
                // are null when the cover atom is untouched; the
                // kernel never dereferences them on that branch.
                unsafe {
                    let parent_lo_param = match work_prefix.as_ref() {
                        Some(_) => find_col(&frontier, ColTag::RangeLo(a), ctx)?.as_kernel_param(),
                        None => null_ptr.as_kernel_param(),
                    };
                    let wp_param = match work_prefix.as_ref() {
                        Some(wp) => wp.as_kernel_param(),
                        None => null_ptr.as_kernel_param(),
                    };
                    let mut params: Vec<*mut c_void> = vec![
                        (&d_cover_tbl).as_kernel_param(),
                        c_u32.as_kernel_param(),
                        parent_lo_param,
                        wp_param,
                        has_parent_range.as_kernel_param(),
                        const_lo.as_kernel_param(),
                        const_hi.as_kernel_param(),
                        count.as_kernel_param(),
                        total_work.as_kernel_param(),
                        identity_u32.as_kernel_param(),
                        match d_fused_desc.as_ref() {
                            Some(d) => d.as_kernel_param(),
                            None => null_ptr.as_kernel_param(),
                        },
                        n_fused_u32.as_kernel_param(),
                        (&marks).as_kernel_param(),
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
                                "{} launch failed: {e}",
                                width.count_kernel()
                            ))
                        })?;
                }
                self.multiblock_scan_u32_inplace_on_stream(
                    &mut marks,
                    total_work + 1,
                    &cu_stream,
                    launch_stream,
                    runtime,
                )?;
                rec.commit(runtime)
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: count commit failed: {e}")))?;
            }
            let n_children = if count_ran {
                cu_stream
                    .synchronize()
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: count sync failed: {e}")))?;
                self.dtoh_scalar_untracked::<u32>(&marks, total_work as usize)?
            } else {
                total_work
            };
            if n_children == 0 {
                return self.create_empty_buffer(out_schema);
            }

            // ---- EXPAND phase 2 + PROBE: allocate the child
            // frontier (copied parent columns minus the cover's
            // ranges, new cover variables, refined cover range when
            // the cover atom keeps unconsumed columns), the probe
            // range outputs, and the survival mask — all before the
            // recorder, per the established discipline.
            // VAR columns are width-sized data; RANGE columns are u32
            // row indices in every width class. The emit kernel takes
            // the two copy groups separately so one launch shape
            // serves both widths.
            let var_bytes = (n_children as usize) * width.var_bytes();
            let range_bytes = (n_children as usize) * std::mem::size_of::<u32>();
            let mut parent_copy_var_ptrs: Vec<u64> = Vec::new();
            let mut child_copy_var_ptrs: Vec<u64> = Vec::new();
            let mut parent_copy_range_ptrs: Vec<u64> = Vec::new();
            let mut child_copy_range_ptrs: Vec<u64> = Vec::new();
            let mut child_cols: Vec<FrontierCol> = Vec::new();
            for (tag, slice) in &frontier {
                if matches!(tag, ColTag::RangeLo(x) | ColTag::RangeHi(x) if *x == a) {
                    continue; // cover range is refined, not copied
                }
                let is_var = matches!(tag, ColTag::Var(_));
                let dst = self
                    .memory()
                    .alloc::<u8>(if is_var { var_bytes } else { range_bytes })?;
                if is_var {
                    parent_copy_var_ptrs.push(*slice.device_ptr());
                    child_copy_var_ptrs.push(*dst.device_ptr());
                } else {
                    parent_copy_range_ptrs.push(*slice.device_ptr());
                    child_copy_range_ptrs.push(*dst.device_ptr());
                }
                child_cols.push((*tag, dst));
            }
            let n_copy_var = parent_copy_var_ptrs.len();
            let n_copy_range = parent_copy_range_ptrs.len();
            let mut child_var_ptrs: Vec<u64> = Vec::with_capacity(c);
            for &v in &node.cover.var_positions {
                let dst = self.memory().alloc::<u8>(var_bytes)?;
                child_var_ptrs.push(*dst.device_ptr());
                child_cols.push((ColTag::Var(v), dst));
            }
            let keep_cover = depth + c < arities[a];
            if keep_cover {
                let lo = self.memory().alloc::<u8>(range_bytes)?;
                let hi = self.memory().alloc::<u8>(range_bytes)?;
                child_cols.push((ColTag::RangeLo(a), lo));
                child_cols.push((ColTag::RangeHi(a), hi));
            }
            // Pointer tables (launch metadata; bounded by plan width).
            let upload_tbl = |ptrs: &[u64]| -> Result<TrackedCudaSlice<u64>> {
                let mut tbl = self.memory().alloc::<u64>(ptrs.len().max(1))?;
                if !ptrs.is_empty() {
                    self.htod_launch_metadata_sync_copy_into(ptrs, &mut tbl)
                        .map_err(|e| {
                            XlogError::Kernel(format!("{ctx}: htod pointer table failed: {e}"))
                        })?;
                }
                Ok(tbl)
            };
            let d_parent_copy_var_tbl = upload_tbl(&parent_copy_var_ptrs)?;
            let d_child_copy_var_tbl = upload_tbl(&child_copy_var_ptrs)?;
            let d_parent_copy_range_tbl = upload_tbl(&parent_copy_range_ptrs)?;
            let d_child_copy_range_tbl = upload_tbl(&child_copy_range_ptrs)?;
            let d_child_var_tbl = upload_tbl(&child_var_ptrs)?;

            // Probe pre-allocations (key tables, data tables,
            // refined ranges, mask).
            struct ProbePlan {
                input_idx: usize,
                n_cols: u32,
                data_tbl: TrackedCudaSlice<u64>,
                key_tbl: TrackedCudaSlice<u64>,
                live: bool,
                keep: bool,
                out_lo: Option<TrackedCudaSlice<u8>>,
                out_hi: Option<TrackedCudaSlice<u8>>,
            }
            let mut probe_plans: Vec<ProbePlan> = Vec::with_capacity(node.probes.len());
            for probe in node.probes.iter().filter(|pr| !is_fusable(pr)) {
                let p = probe.input_idx;
                let p_len = probe.var_positions.len();
                let p_depth = consumed[p];
                let data_ptrs: Vec<u64> = (p_depth..p_depth + p_len)
                    .map(|i| owned_col_ptr(&norm[p], i, ctx))
                    .collect::<Result<_>>()?;
                let key_ptrs: Vec<u64> = probe
                    .var_positions
                    .iter()
                    .map(|&v| Ok(*find_col(&child_cols, ColTag::Var(v), ctx)?.device_ptr()))
                    .collect::<Result<_>>()?;
                let live = child_cols.iter().any(|(t, _)| *t == ColTag::RangeLo(p));
                let keep = p_depth + p_len < arities[p];
                let (out_lo, out_hi) = if keep {
                    (
                        Some(self.memory().alloc::<u8>(range_bytes)?),
                        Some(self.memory().alloc::<u8>(range_bytes)?),
                    )
                } else {
                    (None, None)
                };
                probe_plans.push(ProbePlan {
                    input_idx: p,
                    n_cols: p_len as u32,
                    data_tbl: upload_tbl(&data_ptrs)?,
                    key_tbl: upload_tbl(&key_ptrs)?,
                    live,
                    keep,
                    out_lo,
                    out_hi,
                });
            }
            let mask: Option<TrackedCudaSlice<u8>> = if probe_plans.is_empty() {
                None
            } else {
                Some(self.memory().alloc::<u8>(n_children as usize)?)
            };

            {
                let mut rec = LaunchRecorder::new_strict(launch_stream);
                for i in depth..depth + c {
                    rec.read_column(norm[a].column(i).expect("validated cover column"));
                }
                rec.read(&d_cover_tbl);
                if count_ran {
                    rec.read(&marks);
                }
                if let Some(wp) = work_prefix.as_ref() {
                    rec.read(wp);
                    rec.read(find_col(&frontier, ColTag::RangeLo(a), ctx)?);
                    rec.read(find_col(&frontier, ColTag::RangeHi(a), ctx)?);
                }
                for (_, slice) in &frontier {
                    rec.read(slice);
                }
                rec.read(&d_parent_copy_var_tbl);
                rec.read(&d_child_copy_var_tbl);
                rec.read(&d_parent_copy_range_tbl);
                rec.read(&d_child_copy_range_tbl);
                rec.read(&d_child_var_tbl);
                for (_, slice) in &child_cols {
                    rec.write(slice);
                }
                for pp in probe_plans.iter() {
                    let p = pp.input_idx;
                    for i in consumed[p]..consumed[p] + pp.n_cols as usize {
                        rec.read_column(norm[p].column(i).expect("validated probe column"));
                    }
                    rec.read(norm[p].num_rows_device());
                    rec.read(&pp.data_tbl);
                    rec.read(&pp.key_tbl);
                    if let Some(lo) = pp.out_lo.as_ref() {
                        rec.write(lo);
                    }
                    if let Some(hi) = pp.out_hi.as_ref() {
                        rec.write(hi);
                    }
                }
                if let Some(m) = mask.as_ref() {
                    rec.write(m);
                }
                rec.preflight(runtime)
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: emit preflight failed: {e}")))?;

                let emit_kernel = self
                    .device()
                    .inner()
                    .get_func(WCOJ_MODULE, width.emit_kernel())
                    .ok_or_else(|| {
                        XlogError::Kernel(format!("{} kernel not found", width.emit_kernel()))
                    })?;
                let grid = total_work.div_ceil(BLOCK_SIZE);
                let c_u32 = c as u32;
                let n_copy_var_u32 = n_copy_var as u32;
                let n_copy_range_u32 = n_copy_range as u32;
                let keep_cover_u32 = u32::from(keep_cover);
                // SAFETY: fj_expand_emit_{u32,u64}(cover_cols,
                // n_cover_cols, parent_lo, parent_hi, work_prefix,
                // has_parent_range, const_lo, const_hi, n_frontier,
                // total_work, group_offsets, parent_copy_var_cols,
                // child_copy_var_cols, n_copy_var_cols,
                // parent_copy_range_cols, child_copy_range_cols,
                // n_copy_range_cols, child_var_cols, keep_cover_range,
                // child_cover_lo, child_cover_hi). Nullable pointers
                // are only dereferenced behind their flags.
                unsafe {
                    let parent_lo_param = match work_prefix.as_ref() {
                        Some(_) => find_col(&frontier, ColTag::RangeLo(a), ctx)?.as_kernel_param(),
                        None => null_ptr.as_kernel_param(),
                    };
                    let parent_hi_param = match work_prefix.as_ref() {
                        Some(_) => find_col(&frontier, ColTag::RangeHi(a), ctx)?.as_kernel_param(),
                        None => null_ptr.as_kernel_param(),
                    };
                    let wp_param = match work_prefix.as_ref() {
                        Some(wp) => wp.as_kernel_param(),
                        None => null_ptr.as_kernel_param(),
                    };
                    let cover_lo_param = if keep_cover {
                        find_col(&child_cols, ColTag::RangeLo(a), ctx)?.as_kernel_param()
                    } else {
                        null_ptr.as_kernel_param()
                    };
                    let cover_hi_param = if keep_cover {
                        find_col(&child_cols, ColTag::RangeHi(a), ctx)?.as_kernel_param()
                    } else {
                        null_ptr.as_kernel_param()
                    };
                    let mut params: Vec<*mut c_void> = vec![
                        (&d_cover_tbl).as_kernel_param(),
                        c_u32.as_kernel_param(),
                        parent_lo_param,
                        parent_hi_param,
                        wp_param,
                        has_parent_range.as_kernel_param(),
                        const_lo.as_kernel_param(),
                        const_hi.as_kernel_param(),
                        count.as_kernel_param(),
                        total_work.as_kernel_param(),
                        // No-count path: null offsets select the kernel's
                        // out == w branch (every position is its own group).
                        if count_ran {
                            (&marks).as_kernel_param()
                        } else {
                            null_ptr.as_kernel_param()
                        },
                        (&d_parent_copy_var_tbl).as_kernel_param(),
                        (&d_child_copy_var_tbl).as_kernel_param(),
                        n_copy_var_u32.as_kernel_param(),
                        (&d_parent_copy_range_tbl).as_kernel_param(),
                        (&d_child_copy_range_tbl).as_kernel_param(),
                        n_copy_range_u32.as_kernel_param(),
                        (&d_child_var_tbl).as_kernel_param(),
                        keep_cover_u32.as_kernel_param(),
                        cover_lo_param,
                        cover_hi_param,
                    ];
                    emit_kernel
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
                                "{} launch failed: {e}",
                                width.emit_kernel()
                            ))
                        })?;
                }

                // PROBE refinements over the expanded frontier.
                let probe_kernel = self
                    .device()
                    .inner()
                    .get_func(WCOJ_MODULE, width.probe_kernel())
                    .ok_or_else(|| {
                        XlogError::Kernel(format!("{} kernel not found", width.probe_kernel()))
                    })?;
                let probe_grid = n_children.div_ceil(BLOCK_SIZE);
                for (probe_idx, pp) in probe_plans.iter().enumerate() {
                    let p = pp.input_idx;
                    let has_range = u32::from(pp.live);
                    let p_const_lo: u32 = 0;
                    let p_const_hi: u32 = n_rows[p];
                    let keep_u32 = u32::from(pp.keep);
                    let combine: u32 = u32::from(probe_idx > 0);
                    // SAFETY: fj_probe_refine_u32(probe_cols,
                    // n_probe_cols, key_cols, in_lo, in_hi, has_range,
                    // const_lo, const_hi, n_frontier, keep_range,
                    // out_lo, out_hi, mask, combine_mask). Nullable
                    // pointers only dereferenced behind their flags.
                    unsafe {
                        let in_lo_param = if pp.live {
                            find_col(&child_cols, ColTag::RangeLo(p), ctx)?.as_kernel_param()
                        } else {
                            null_ptr.as_kernel_param()
                        };
                        let in_hi_param = if pp.live {
                            find_col(&child_cols, ColTag::RangeHi(p), ctx)?.as_kernel_param()
                        } else {
                            null_ptr.as_kernel_param()
                        };
                        let out_lo_param = match pp.out_lo.as_ref() {
                            Some(lo) => lo.as_kernel_param(),
                            None => null_ptr.as_kernel_param(),
                        };
                        let out_hi_param = match pp.out_hi.as_ref() {
                            Some(hi) => hi.as_kernel_param(),
                            None => null_ptr.as_kernel_param(),
                        };
                        let mask_ref = mask.as_ref().expect("mask exists when probes exist");
                        let mut params: Vec<*mut c_void> = vec![
                            (&pp.data_tbl).as_kernel_param(),
                            pp.n_cols.as_kernel_param(),
                            (&pp.key_tbl).as_kernel_param(),
                            in_lo_param,
                            in_hi_param,
                            has_range.as_kernel_param(),
                            p_const_lo.as_kernel_param(),
                            p_const_hi.as_kernel_param(),
                            n_children.as_kernel_param(),
                            keep_u32.as_kernel_param(),
                            out_lo_param,
                            out_hi_param,
                            mask_ref.as_kernel_param(),
                            combine.as_kernel_param(),
                        ];
                        probe_kernel
                            .clone()
                            .launch_on_stream(
                                &cu_stream,
                                LaunchConfig {
                                    grid_dim: (probe_grid, 1, 1),
                                    block_dim: (BLOCK_SIZE, 1, 1),
                                    shared_mem_bytes: 0,
                                },
                                &mut params,
                            )
                            .map_err(|e| {
                                XlogError::Kernel(format!(
                                    "{} launch failed: {e}",
                                    width.probe_kernel()
                                ))
                            })?;
                    }
                }
                rec.commit(runtime)
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: emit commit failed: {e}")))?;
            }

            // ---- Bookkeeping: consumed columns, live-range set.
            consumed[a] += c;
            for probe in &node.probes {
                consumed[probe.input_idx] += probe.var_positions.len();
            }
            // Fused probes exhaust their atoms in the count pass: drop
            // their stale copied ranges from the child frontier.
            for pr in &fused {
                let p = pr.input_idx;
                child_cols.retain(|(t, _)| {
                    !matches!(t, ColTag::RangeLo(x) | ColTag::RangeHi(x) if *x == p)
                });
            }
            // Replace probed atoms' stale (copied) ranges with the
            // refined outputs; exhausted atoms drop their ranges.
            for pp in &mut probe_plans {
                let p = pp.input_idx;
                child_cols.retain(|(t, _)| {
                    !matches!(t, ColTag::RangeLo(x) | ColTag::RangeHi(x) if *x == p)
                });
                if pp.keep {
                    child_cols.push((
                        ColTag::RangeLo(p),
                        pp.out_lo.take().expect("keep implies out_lo"),
                    ));
                    child_cols.push((
                        ColTag::RangeHi(p),
                        pp.out_hi.take().expect("keep implies out_hi"),
                    ));
                }
            }

            // ---- Compaction (single mask pass per node).
            if let Some(mask) = mask {
                let tags: Vec<ColTag> = child_cols.iter().map(|(t, _)| *t).collect();
                // Per-tag column types: the compaction helper sizes
                // its per-column copies from the schema, so the mixed
                // VAR/RANGE width classes need no special casing.
                let schema = Schema::new(
                    tags.iter()
                        .enumerate()
                        .map(|(i, t)| (format!("f{i}"), tag_type(*t, width)))
                        .collect(),
                );
                let d_nr = self.memory().alloc::<u32>(1)?;
                self.htod_launch_metadata_async_copy_one(
                    &n_children,
                    &d_nr,
                    &cu_stream,
                    &format!("{ctx}: frontier num_rows"),
                )?;
                let columns: Vec<CudaColumn> =
                    child_cols.drain(..).map(|(_, s)| s.into()).collect();
                let staging = CudaBuffer::from_columns_with_host_count(
                    columns,
                    u64::from(n_children),
                    d_nr,
                    schema,
                    n_children,
                );
                let compacted = self.compact_buffer_by_device_mask_counted_recorded(
                    &staging,
                    &mask,
                    launch_stream,
                )?;
                let new_count = compacted.cached_row_count().ok_or_else(|| {
                    XlogError::Kernel(format!("{ctx}: compaction lost its row count"))
                })?;
                if new_count == 0 {
                    return self.create_empty_buffer(out_schema);
                }
                let mut new_frontier: Vec<FrontierCol> = Vec::with_capacity(tags.len());
                for (tag, col) in tags.into_iter().zip(compacted.columns.into_iter()) {
                    let CudaColumn::Owned(slice) = col else {
                        return Err(XlogError::Kernel(format!(
                            "{ctx}: compaction produced a non-owned column"
                        )));
                    };
                    new_frontier.push((tag, slice));
                }
                frontier = new_frontier;
                count = new_count;
            } else {
                frontier = child_cols;
                count = n_children;
            }
            frontier_cap = n_children;
        }

        // ---- COUNT epilogue (§2.4): per-row multiplicity = product
        // of remaining live trie-range lengths, then the existing
        // recorded groupby Sum reduces (group, multiplicity) to
        // (group, count). Unconsumed trailing columns never expand
        // the frontier — this is the d-representation count.
        if mode == FjMode::CountByRoot {
            let group_var = plan.output_vars[0];
            let mut lo_ptrs: Vec<u64> = Vec::new();
            let mut hi_ptrs: Vec<u64> = Vec::new();
            for (t, s) in &frontier {
                if let ColTag::RangeLo(x) = t {
                    lo_ptrs.push(*s.device_ptr());
                    hi_ptrs.push(*find_col(&frontier, ColTag::RangeHi(*x), ctx)?.device_ptr());
                }
            }
            let upload_tbl = |ptrs: &[u64]| -> Result<TrackedCudaSlice<u64>> {
                let mut tbl = self.memory().alloc::<u64>(ptrs.len().max(1))?;
                if !ptrs.is_empty() {
                    self.htod_launch_metadata_sync_copy_into(ptrs, &mut tbl)
                        .map_err(|e| {
                            XlogError::Kernel(format!("{ctx}: htod range table failed: {e}"))
                        })?;
                }
                Ok(tbl)
            };
            let d_lo_tbl = upload_tbl(&lo_ptrs)?;
            let d_hi_tbl = upload_tbl(&hi_ptrs)?;
            // Sized to the frontier CAPACITY so the staging buffer's
            // row_cap invariant holds; only the logical `count`
            // prefix is written/read.
            let mut mult = self
                .memory()
                .alloc::<u8>(frontier_cap as usize * std::mem::size_of::<u64>())?;
            {
                let mut rec = LaunchRecorder::new_strict(launch_stream);
                for (t, s) in &frontier {
                    if matches!(t, ColTag::RangeLo(_) | ColTag::RangeHi(_)) {
                        rec.read(s);
                    }
                }
                rec.read(&d_lo_tbl);
                rec.read(&d_hi_tbl);
                rec.write(&mult);
                rec.preflight(runtime).map_err(|e| {
                    XlogError::Kernel(format!("{ctx}: multiplicity preflight failed: {e}"))
                })?;
                let kernel = self
                    .device()
                    .inner()
                    .get_func(WCOJ_MODULE, wcoj_kernels::FJ_COUNT_MULTIPLICITY)
                    .ok_or_else(|| {
                        XlogError::Kernel("fj_count_multiplicity kernel not found".to_string())
                    })?;
                let grid = count.div_ceil(BLOCK_SIZE);
                let n_ranges = lo_ptrs.len() as u32;
                // SAFETY: fj_count_multiplicity(range_lo_cols,
                // range_hi_cols, n_ranges, n_frontier, mult);
                // device-resident, preflighted.
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
                            (&d_lo_tbl, &d_hi_tbl, n_ranges, count, &mut mult),
                        )
                        .map_err(|e| {
                            XlogError::Kernel(format!(
                                "fj_count_multiplicity launch failed: {e}"
                            ))
                        })?;
                }
                rec.commit(runtime).map_err(|e| {
                    XlogError::Kernel(format!("{ctx}: multiplicity commit failed: {e}"))
                })?;
            }
            let key_idx = frontier
                .iter()
                .position(|(t, _)| *t == ColTag::Var(group_var))
                .ok_or_else(|| {
                    XlogError::Kernel(format!("{ctx}: group variable {group_var} missing"))
                })?;
            let (_, key_col) = frontier.swap_remove(key_idx);
            let d_nr = self.memory().alloc::<u32>(1)?;
            self.htod_launch_metadata_async_copy_one(
                &count,
                &d_nr,
                &cu_stream,
                &format!("{ctx}: staging num_rows"),
            )?;
            let staging_schema = Schema::new(vec![
                (format!("v{group_var}"), width.var_type()),
                ("count".to_string(), ScalarType::U64),
            ]);
            let staging = CudaBuffer::from_columns_with_host_count(
                vec![key_col.into(), mult.into()],
                u64::from(frontier_cap),
                d_nr,
                staging_schema,
                count,
            );
            return self.groupby_multi_agg_recorded(
                &staging,
                &[0],
                &[(1, xlog_core::AggOp::Sum)],
                launch_stream,
            );
        }

        // ---- Final materialization: project the head variables out
        // of the frontier (recorded dtod copies). The frontier holds
        // exactly the bound-variable columns at this point (all atoms
        // exhausted ⇒ no range columns survive).
        let perm: Vec<usize> = plan
            .output_vars
            .iter()
            .map(|&v| {
                frontier
                    .iter()
                    .position(|(t, _)| *t == ColTag::Var(v))
                    .ok_or_else(|| {
                        XlogError::Kernel(format!("{ctx}: output variable {v} missing"))
                    })
            })
            .collect::<Result<_>>()?;
        let schema = Schema::new(
            frontier
                .iter()
                .enumerate()
                .map(|(i, (t, _))| (format!("f{i}"), tag_type(*t, width)))
                .collect(),
        );
        let d_nr = self.memory().alloc::<u32>(1)?;
        self.htod_launch_metadata_async_copy_one(
            &count,
            &d_nr,
            &cu_stream,
            &format!("{ctx}: result num_rows"),
        )?;
        let columns: Vec<CudaColumn> = frontier.into_iter().map(|(_, s)| s.into()).collect();
        // row_cap = frontier CAPACITY (compaction shrinks the logical
        // count without reallocating columns); the logical count rides
        // on num_rows_device + the host cache.
        let src = CudaBuffer::from_columns_with_host_count(
            columns,
            u64::from(frontier_cap),
            d_nr,
            schema,
            count,
        );
        let projected =
            self.wcoj_project_output_columns_recorded(&src, &perm, out_schema, launch_stream)?;
        // Set semantics: projecting a strict subset of the bound
        // variables can introduce duplicates — dedup exactly then
        // (the full projection is duplicate-free by construction).
        let distinct_outputs: std::collections::BTreeSet<usize> =
            plan.output_vars.iter().copied().collect();
        if distinct_outputs.len() < bind_order.len() {
            return self.dedup_full_row_recorded(&projected, launch_stream);
        }
        Ok(projected)
    }
}
