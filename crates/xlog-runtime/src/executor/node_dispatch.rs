//! RIR node dispatch and per-node execution handlers.

use std::collections::HashMap;

use xlog_core::{AggOp, RelId, Result, ScalarType, Schema, XlogError};
use xlog_cuda::provider::NESTED_LOOP_TOTAL_THRESHOLD;
use xlog_cuda::{CudaBuffer, JoinType as CudaJoinType};
use xlog_ir::{JoinType, ProjectExpr, RirNode};

use crate::ilp_registry::{read_device_row_count, IlpMask, IlpTagEntry, IlpTaggedResult};

use super::join_cache::{estimate_join_index_bytes, JoinIndexKey};
use super::Executor;

/// W4.2 eligibility predicate for nested-loop join dispatch.
///
/// Returns `true` iff the join shape is admissible for the
/// `nested_loop_join_v2_inner_u32_1key` provider entry point.
/// The predicate is intentionally narrow per the W4.2
/// iteration-4 plan D1:
///   * `JoinType::Inner` only (Semi / Anti / LeftOuter fall back
///     to hash).
///   * Exactly one key column on each side.
///   * Both key columns share the same `ScalarType` AND that
///     shared type is `U32` or `Symbol` (Symbol is `u32` at the
///     byte level — same kernel applies). U32-on-Symbol or
///     other type mismatches return `false`, mirroring
///     `hash_join_v2`'s own type-mismatch rejection at
///     `crates/xlog-cuda/src/provider/relational.rs:3567-3576`.
///
/// Out-of-bounds key indices yield `Schema::column_type(_) = None`,
/// which fails the `matches!(...)` guard — falling back to hash
/// without a separate bounds check.
///
/// Cheap O(1) — no kernel launches, no row-count reads, no D2H.
/// The threshold check (`num_left * num_right <=
/// NESTED_LOOP_TOTAL_THRESHOLD`) is performed at the dispatch
/// site (W4.2 Step 5), not in this predicate.
//
fn eligible_for_nested_loop(
    left: &CudaBuffer,
    right: &CudaBuffer,
    left_keys: &[usize],
    right_keys: &[usize],
    join_type: JoinType,
) -> bool {
    if join_type != JoinType::Inner {
        return false;
    }
    if left_keys.len() != 1 || right_keys.len() != 1 {
        return false;
    }
    let lt = left.schema().column_type(left_keys[0]);
    let rt = right.schema().column_type(right_keys[0]);
    lt == rt && matches!(lt, Some(ScalarType::U32) | Some(ScalarType::Symbol))
}

/// W4.3 eligibility predicate for sort-merge join dispatch.
///
/// Returns `true` iff the join shape is admissible for the
/// `sort_merge_join_v2_inner_u32_1key` provider entry point.
/// The predicate is intentionally narrow per the W4.3
/// iteration-4 plan D4 — same envelope as
/// `eligible_for_nested_loop` so the dispatch decision tree
/// (D2: sort-merge > nested-loop > hash) can route between
/// operators with symmetric eligibility checks:
///   * `JoinType::Inner` only (Semi / Anti / LeftOuter fall back
///     to nested-loop or hash).
///   * Exactly one key column on each side.
///   * Both key columns share the same `ScalarType` AND that
///     shared type is `U32` or `Symbol` (Symbol is `u32` at the
///     byte level — same kernel applies).
///
/// **Sortedness detection is NOT in this predicate** (per W4.3
/// D1 + Step 4 lock): runtime sortedness detection requires a
/// kernel launch + D2H, which is a runtime cost not appropriate
/// for a static eligibility predicate. The dispatch site (Step
/// 5) calls `provider.is_sorted_ascending_u32` AFTER this
/// predicate accepts AND the threshold check passes, so
/// detection cost is paid only on candidates.
///
/// **Threshold check is NOT in this predicate** either — it
/// requires `device_row_count` (an O(1) read but cheap-relative-
/// to-kernel and conventionally placed at the dispatch site
/// per W4.2's pattern).
///
/// Out-of-bounds key indices yield `Schema::column_type(_) = None`,
/// which fails the `matches!(...)` guard — falling back to
/// hash without a separate bounds check.
///
/// Cheap O(1) — no kernel launches, no row-count reads, no D2H.
//
// W4.3 plan iter-4 Step 4: predicate is added in this commit;
// the call site that wires it into `execute_join` (BEFORE the
// W4.2 nested-loop branch per D2 precedence) lives in Step 5.
// Until Step 5 lands the predicate is intentionally unused.
#[allow(dead_code)]
fn eligible_for_sort_merge(
    left: &CudaBuffer,
    right: &CudaBuffer,
    left_keys: &[usize],
    right_keys: &[usize],
    join_type: JoinType,
) -> bool {
    if join_type != JoinType::Inner {
        return false;
    }
    if left_keys.len() != 1 || right_keys.len() != 1 {
        return false;
    }
    let lt = left.schema().column_type(left_keys[0]);
    let rt = right.schema().column_type(right_keys[0]);
    lt == rt && matches!(lt, Some(ScalarType::U32) | Some(ScalarType::Symbol))
}

impl Executor {
    /// Execute a Scan node — looks up the relation by RelId and returns a clone.
    pub(super) fn execute_scan(&mut self, rel: RelId) -> Result<CudaBuffer> {
        let name = self
            .get_rel_name(rel)
            .ok_or_else(|| XlogError::Execution(format!("Unknown relation: RelId({})", rel.0)))?;

        let buffer = self
            .store
            .get(name)
            .ok_or_else(|| XlogError::Execution(format!("Relation not found: {}", name)))?;

        self.stats.record_access(rel);
        self.stats.update_cardinality(rel, buffer.num_rows());
        self.stats.update_byte_size(rel, buffer.estimated_bytes());

        self.clone_buffer(buffer)
    }

    /// Execute a single RIR node tree
    ///
    /// Recursively evaluates the node and its children, returning
    /// the result as a GPU buffer.
    ///
    /// # Arguments
    /// * `node` - The RIR node to execute
    ///
    /// # Returns
    /// A CudaBuffer containing the result of the node execution
    ///
    /// # Errors
    /// Returns an error if the node execution fails
    pub fn execute_node(&mut self, node: &RirNode) -> Result<CudaBuffer> {
        match node {
            RirNode::Unit => {
                // Materialize the relational "unit" ({()}) as a 0-arity buffer with one row.
                let mut d_num_rows = self.provider.memory().alloc::<u32>(1)?;
                self.provider
                    .device()
                    .inner()
                    .htod_sync_copy_into(&[1u32], &mut d_num_rows)
                    .map_err(|e| {
                        XlogError::Kernel(format!("Failed to create unit row count: {}", e))
                    })?;
                Ok(CudaBuffer::from_columns(
                    Vec::new(),
                    1,
                    d_num_rows,
                    Schema::new(vec![]),
                ))
            }

            RirNode::Scan { rel } => {
                let start = self.profiler.start_op();
                let result = self.execute_scan(*rel)?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("scan", 0, result.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }
                Ok(result)
            }

            RirNode::Filter { input, predicate } => {
                let input_buf = self.execute_node(input)?;
                let input_rows = input_buf.num_rows();
                let start = self.profiler.start_op();
                let result = self.execute_filter(&input_buf, predicate)?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("filter", input_rows, result.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }
                Ok(result)
            }

            RirNode::Project { input, columns } => {
                let input_buf = self.execute_node(input)?;
                let input_rows = input_buf.num_rows();
                let start = self.profiler.start_op();
                let result = self.execute_project(&input_buf, columns)?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("project", input_rows, result.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }
                Ok(result)
            }

            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
            } => {
                let left_rel = match left.as_ref() {
                    RirNode::Scan { rel } => Some(*rel),
                    _ => None,
                };
                let right_rel = match right.as_ref() {
                    RirNode::Scan { rel } => Some(*rel),
                    _ => None,
                };
                let left_buf = self.execute_node(left)?;
                let right_buf = self.execute_node(right)?;
                let input_rows = left_buf.num_rows() + right_buf.num_rows();
                let start = self.profiler.start_op();
                let result = self.execute_join(
                    &left_buf, &right_buf, left_keys, right_keys, *join_type, left_rel, right_rel,
                )?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("join", input_rows, result.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }
                Ok(result)
            }

            RirNode::GroupBy {
                input,
                key_cols,
                aggs,
            } => {
                let input_buf = self.execute_node(input)?;
                let input_rows = input_buf.num_rows();
                let start = self.profiler.start_op();
                let result = self.execute_groupby(&input_buf, key_cols, aggs)?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("groupby", input_rows, result.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }
                Ok(result)
            }

            RirNode::Union { inputs } => {
                let mut buffers = Vec::with_capacity(inputs.len());
                let mut input_rows = 0u64;
                for input in inputs {
                    let buf = self.execute_node(input)?;
                    input_rows += buf.num_rows();
                    buffers.push(buf);
                }
                let start = self.profiler.start_op();
                let result = self.execute_union(&buffers)?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("union", input_rows, result.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }
                Ok(result)
            }

            RirNode::Distinct { input, key_cols } => {
                let input_buf = self.execute_node(input)?;
                let input_rows = input_buf.num_rows();
                let start = self.profiler.start_op();
                let result = self.execute_distinct(&input_buf, key_cols)?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("dedup", input_rows, result.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }
                Ok(result)
            }

            RirNode::Diff { left, right } => {
                let left_buf = self.execute_node(left)?;
                let right_buf = self.execute_node(right)?;
                let input_rows = left_buf.num_rows() + right_buf.num_rows();
                let start = self.profiler.start_op();
                let result = self.execute_diff(&left_buf, &right_buf)?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("diff", input_rows, result.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }
                Ok(result)
            }

            RirNode::Fixpoint {
                scc_id,
                base,
                recursive,
                delta_rel,
                full_rel,
            } => {
                // Semi-naive fixpoint iteration
                self.execute_fixpoint(*scc_id, base, recursive, *delta_rel, *full_rel)
            }
            RirNode::TensorMaskedJoin {
                mask_name,
                schema_size,
                left_keys,
                right_keys,
                rel_index,
                head_rel_name,
                max_active_rules,
                head_projection,
                ..
            } => self.execute_tensor_masked_join(
                mask_name,
                *schema_size,
                left_keys,
                right_keys,
                rel_index,
                head_rel_name,
                *max_active_rules,
                head_projection,
            ),
            // v0.6.5 slice 1: defensive fallback descent for any
            // `execute_node` caller that bypasses the WCOJ dispatch
            // hook (probabilistic eval, neural store walks, etc.).
            // The non-recursive arm in `recursive.rs` short-circuits
            // dispatch-eligible bodies before reaching here; this
            // arm is the safety net for everyone else.
            RirNode::MultiWayJoin { fallback, .. } => self.execute_node(fallback),
        }
    }

    /// Execute a Join node
    ///
    /// Delegates to the kernel provider's hash_join_v2 which supports all join types natively.
    fn execute_join(
        &mut self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
        join_type: JoinType,
        left_rel: Option<RelId>,
        right_rel: Option<RelId>,
    ) -> Result<CudaBuffer> {
        // Convert IR JoinType to CUDA JoinType (used by adaptive
        // indexing and the hash fallback below).
        let cuda_join_type = match join_type {
            JoinType::Inner => CudaJoinType::Inner,
            JoinType::Semi => CudaJoinType::Semi,
            JoinType::Anti => CudaJoinType::Anti,
            JoinType::LeftOuter => CudaJoinType::LeftOuter,
        };

        // Output buffer — set by W4.2 nested-loop dispatch,
        // adaptive indexing, or the hash fallback. All three
        // paths flow through the shared `record_join_result`
        // feedback block at the end of this fn.
        let mut out: Option<CudaBuffer> = None;

        // W4.2 nested-loop dispatch (precedes adaptive indexing
        // and hash fallback). On predicate + threshold pass,
        // route to `nested_loop_join_v2_inner_u32_1key` and bump
        // the dispatch counter; do NOT early-return — leave the
        // result in `out` so the shared feedback block at fn-end
        // observes it. Otherwise leave `out = None` and fall
        // through.
        //
        // Threshold check uses logical row counts via
        // `provider.device_row_count(...)` (NOT `row_cap`), with
        // `checked_mul` fail-closed on overflow per W4.2
        // iteration-4 F-W42-15.
        if eligible_for_nested_loop(left, right, left_keys, right_keys, join_type) {
            let num_left = self.provider.device_row_count(left)? as u64;
            let num_right = self.provider.device_row_count(right)? as u64;
            let in_threshold = num_left
                .checked_mul(num_right)
                .map(|p| p <= NESTED_LOOP_TOTAL_THRESHOLD)
                .unwrap_or(false);
            if in_threshold {
                out = Some(self.provider.nested_loop_join_v2_inner_u32_1key(
                    left,
                    right,
                    left_keys[0],
                    right_keys[0],
                )?);
                self.nested_loop_dispatch_count += 1;
            }
        }

        // Adaptive indexing: opportunistically reuse cached
        // build-side hash tables when the right side is a base
        // relation scan and has become "hot" in runtime
        // statistics. Only runs if W4.2 didn't dispatch.
        if out.is_none() {
            if let Some(build_rel) = right_rel {
                let build_heat = self
                    .stats
                    .get_relation_stats(build_rel)
                    .map(|s| s.heat)
                    .unwrap_or(0.0);
                let est_index_bytes = estimate_join_index_bytes(right, right_keys);
                let budget_bytes = self.provider.memory().budget().device_bytes;
                let remaining_bytes = self.provider.memory().remaining_bytes();

                let should_index = self.join_index_cache.should_build(
                    est_index_bytes,
                    build_heat,
                    remaining_bytes,
                    budget_bytes,
                );

                if let Some(build_name) = self.get_rel_name(build_rel).map(|s| s.to_string()) {
                    if let Some(version) = self.store.version(&build_name) {
                        let key = JoinIndexKey {
                            rel: build_rel,
                            version,
                            key_cols: right_keys.to_vec(),
                        };

                        if let Some(index) = self.join_index_cache.get(&key) {
                            out = Some(self.provider.hash_join_v2_with_index(
                                left,
                                right,
                                left_keys,
                                right_keys,
                                cuda_join_type,
                                index,
                                None,
                            )?);
                        } else if should_index {
                            if let Some(build_buf) = self.store.get(&build_name) {
                                match self.provider.build_join_index_v2(build_buf, right_keys) {
                                    Ok(index) => {
                                        let joined = self.provider.hash_join_v2_with_index(
                                            left,
                                            right,
                                            left_keys,
                                            right_keys,
                                            cuda_join_type,
                                            &index,
                                            None,
                                        )?;
                                        self.join_index_cache.insert(key, index);
                                        if let Some(stats) =
                                            self.stats.get_relation_stats_mut(build_rel)
                                        {
                                            stats.has_index = true;
                                        }
                                        out = Some(joined);
                                    }
                                    Err(_) => {
                                        // If indexing fails (e.g., memory pressure), fall back to normal join.
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } // end `if out.is_none()` (adaptive-indexing gate added by W4.2 Step 5 patch)

        let out = match out {
            Some(buf) => buf,
            None => {
                self.provider
                    .hash_join_v2(left, right, left_keys, right_keys, cuda_join_type)?
            }
        };

        if let (Some(l), Some(r)) = (left_rel, right_rel) {
            let input_rows = left.num_rows().saturating_mul(right.num_rows());
            self.stats.record_join_result(
                l,
                r,
                left_keys.to_vec(),
                right_keys.to_vec(),
                input_rows,
                out.num_rows(),
            );
        }

        Ok(out)
    }

    /// Execute a GroupBy node
    ///
    /// Delegates to the kernel provider's groupby_multi_agg for multi-aggregation support.
    fn execute_groupby(
        &self,
        input: &CudaBuffer,
        key_cols: &[usize],
        aggs: &[(usize, AggOp)],
    ) -> Result<CudaBuffer> {
        if aggs.is_empty() {
            // No aggregations: just distinct on key columns
            return self.provider.dedup(input, key_cols);
        }

        // Use multi-aggregation groupby
        self.provider.groupby_multi_agg(input, key_cols, aggs)
    }

    /// Execute a Union node
    ///
    /// Combines multiple input buffers into one using GPU-native operation.
    pub(super) fn execute_union(&self, inputs: &[CudaBuffer]) -> Result<CudaBuffer> {
        if inputs.is_empty() {
            return self.provider.create_empty_buffer(Schema::new(vec![]));
        }

        if inputs.len() == 1 {
            return self.clone_buffer(&inputs[0]);
        }

        // Pairwise union using GPU-native operation
        let mut result = self.clone_buffer(&inputs[0])?;
        for input in inputs.iter().skip(1) {
            result = self.provider.union_gpu(&result, input)?;
        }

        Ok(result)
    }

    /// Execute a Distinct node
    ///
    /// Removes duplicate rows based on key columns.
    pub(super) fn execute_distinct(
        &self,
        input: &CudaBuffer,
        key_cols: &[usize],
    ) -> Result<CudaBuffer> {
        self.provider.dedup(input, key_cols)
    }

    /// Execute a Diff node
    ///
    /// Returns rows in left that are not in right using GPU-native operation.
    pub(super) fn execute_diff(&self, left: &CudaBuffer, right: &CudaBuffer) -> Result<CudaBuffer> {
        self.provider.diff_gpu(left, right)
    }

    fn execute_tensor_masked_join(
        &mut self,
        mask_name: &str,
        schema_size: usize,
        left_keys: &[usize],
        right_keys: &[usize],
        rel_index: &[(RelId, String)],
        head_rel_name: &str,
        max_active_rules: usize,
        head_projection: &[usize],
    ) -> Result<CudaBuffer> {
        // RD-12: No-op when no mask registered. Return empty buffer with
        // the head relation's schema (not Schema::new(vec![])) to prevent
        // schema corruption when execute_non_recursive_scc stores the result.
        let ilp_mask = match self.ilp_registry.get_mask(mask_name) {
            Some(mask) => mask,
            None => {
                self.ilp_last_result = Some(IlpTaggedResult {
                    entries: Vec::new(),
                });
                // RD-37: Fail hard if head relation missing from store.
                let schema = self
                    .store
                    .get(head_rel_name)
                    .map(|buf| buf.schema().clone())
                    .ok_or_else(|| {
                        XlogError::Execution(format!(
                            "TensorMaskedJoin: head relation '{}' not found in store \
                         (was load_facts_into_store called?)",
                            head_rel_name
                        ))
                    })?;
                return self.provider.create_empty_buffer(schema);
            }
        };

        let start = self.profiler.start_op();

        let mut tag_entries: Vec<IlpTagEntry> = Vec::new();
        let mut process_rule = |i: u32,
                                j: u32,
                                k: u32,
                                strict_candidate_idx: Option<usize>,
                                strict_flags: Option<&CudaBuffer>|
         -> Result<()> {
            let (_, left_name) = &rel_index[i as usize];
            let (_, right_name) = &rel_index[j as usize];

            let left_buf = match self.store.get(left_name) {
                Some(buf) if buf.arity() > 0 => buf,
                _ => return Ok(()),
            };
            let right_buf = match self.store.get(right_name) {
                Some(buf) if buf.arity() > 0 => buf,
                _ => return Ok(()),
            };

            // Skip arity-mismatched relations: the join keys are fixed by
            // the learnable rule template, so the mapped relation must have
            // enough columns for every key index. Relations with matching
            // arity but different semantic column meanings will join without
            // error; semantic correctness of the mask is the optimizer's
            // responsibility (RD-37).
            let left_max_key = left_keys.iter().copied().max().unwrap_or(0);
            let right_max_key = right_keys.iter().copied().max().unwrap_or(0);
            if left_buf.arity() <= left_max_key || right_buf.arity() <= right_max_key {
                return Ok(());
            }

            let joined = self.provider.hash_join_v2(
                left_buf,
                right_buf,
                left_keys,
                right_keys,
                CudaJoinType::Inner,
            )?;

            // Project join result to head schema columns if projection is specified.
            // The join produces [left_cols..., right_cols...] but the head may only
            // need a subset (e.g. reach(X,Y) from b1(X,Z) join b2(Z,Y) needs cols 0,3).
            let projected = if !head_projection.is_empty() && head_projection.len() < joined.arity()
            {
                let proj_exprs: Vec<ProjectExpr> = head_projection
                    .iter()
                    .map(|&col| ProjectExpr::Column(col))
                    .collect();
                self.execute_project(&joined, &proj_exprs)?
            } else {
                joined
            };

            let projected = if let (Some(candidate_idx), Some(active_flags)) =
                (strict_candidate_idx, strict_flags)
            {
                self.provider.filter_buffer_by_candidate_flag(
                    &projected,
                    active_flags,
                    candidate_idx,
                )?
            } else {
                projected
            };

            // RD-22: Use public helper instead of private device_row_count
            let num_rows = read_device_row_count(&self.provider, &projected)? as u32;

            if num_rows > 0 {
                tag_entries.push(IlpTagEntry {
                    i,
                    j,
                    k,
                    num_rows,
                    buffer: Some(projected),
                });
            }
            Ok(())
        };

        let active_rule_count = match ilp_mask {
            IlpMask::Dense { hard, soft, .. } => {
                let active_rules = self.provider.extract_active_rule_indices(
                    hard,
                    soft,
                    schema_size,
                    max_active_rules,
                )?;
                let count = active_rules.len() as u64;
                for &(i, j, k) in &active_rules {
                    process_rule(i, j, k, None, None)?;
                }
                count
            }
            IlpMask::Sparse { active_entries, .. } => {
                let limit = max_active_rules.min(active_entries.len());
                for &(i, j, k) in &active_entries[..limit] {
                    process_rule(i, j, k, None, None)?;
                }
                limit as u64
            }
            IlpMask::SparseDevice {
                candidate_order,
                active_flags,
                selected_count,
                ..
            } => {
                if *selected_count > 0 {
                    for (candidate_idx, &(i, j, k)) in candidate_order.iter().enumerate() {
                        process_rule(i, j, k, Some(candidate_idx), Some(active_flags))?;
                    }
                }
                (*selected_count).min(max_active_rules) as u64
            }
        };

        // 4. Phase 2: Union results by k, borrowing buffers from tag_entries
        let mut bufs_by_k: HashMap<u32, Vec<&CudaBuffer>> = HashMap::new();
        for entry in &tag_entries {
            if let Some(ref buf) = entry.buffer {
                bufs_by_k.entry(entry.k).or_default().push(buf);
            }
        }

        for (k, buffers) in bufs_by_k {
            let (_, target_name) = &rel_index[k as usize];

            // Chain-union all buffers (union_gpu takes &CudaBuffer refs)
            let union_buf = if buffers.len() == 1 {
                // Single buffer: union with an empty buffer to produce a copy
                let empty = self
                    .provider
                    .create_empty_buffer(buffers[0].schema().clone())?;
                self.provider.union_gpu(buffers[0], &empty)?
            } else {
                let mut acc = self.provider.union_gpu(buffers[0], buffers[1])?;
                for buf in &buffers[2..] {
                    acc = self.provider.union_gpu(&acc, buf)?;
                }
                acc
            };

            // Diff against existing and merge
            if let Some(existing) = self.store.get(target_name) {
                let delta = self.provider.diff_gpu(&union_buf, existing)?;
                if !delta.is_empty() {
                    let merged = self.provider.union_gpu(existing, &delta)?;
                    self.store_put(target_name, merged);
                }
            } else {
                let key_cols: Vec<usize> = (0..union_buf.arity()).collect();
                let deduped = self.provider.dedup(&union_buf, &key_cols)?;
                self.store_put(target_name, deduped);
            }
        }

        // 5. Phase 3: Store tag entries (with retained buffers)
        self.ilp_last_result = Some(IlpTaggedResult {
            entries: tag_entries,
        });

        if let Some(start) = start {
            let mem = self.provider.memory().allocated_bytes();
            self.profiler
                .record_op("TensorMaskedJoin", 0, active_rule_count, start, mem);
        }

        // Return empty with head schema (results routed via store).
        let schema = self
            .store
            .get(head_rel_name)
            .map(|buf| buf.schema().clone())
            .ok_or_else(|| {
                XlogError::Execution(format!(
                    "TensorMaskedJoin: head relation '{}' not found in store \
                 (was load_facts_into_store called?)",
                    head_rel_name
                ))
            })?;
        self.provider.create_empty_buffer(schema)
    }
}
