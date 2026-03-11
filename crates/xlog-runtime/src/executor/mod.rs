//! Query executor for RIR nodes
//!
//! The executor interprets RIR (Relational IR) nodes using the CUDA kernel provider
//! to execute GPU-accelerated relational operations.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use xlog_core::{RelId, Result, RuntimeConfig, ScalarType, Schema, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_ir::{ConstValue, ExecutionPlan, Expr, JoinType, RirNode, Stratum};
#[cfg(test)]
use xlog_ir::{CompareOp, ProjectExpr};
use xlog_stats::{StatsManager, StatsSnapshot};

use crate::ilp_registry::{IlpRegistry, IlpTaggedResult};
use crate::profiler::{ExecutionStats, Profiler};
use crate::RelationStore;

mod delta;
mod expression;
mod join_cache;
mod node_dispatch;
mod recursive;
use join_cache::JoinIndexCache;

/// Incremental update for a base relation.
pub struct RelationDelta {
    pub insert: Option<CudaBuffer>,
    pub delete: Option<CudaBuffer>,
}

impl RelationDelta {
    pub fn new(insert: Option<CudaBuffer>, delete: Option<CudaBuffer>) -> Self {
        Self { insert, delete }
    }
}

/// Query executor that interprets RIR nodes using GPU kernels
///
/// The executor processes execution plans by iterating through strata and
/// executing RIR node trees. It maintains a relation store for intermediate
/// and final results.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use xlog_runtime::Executor;
/// use xlog_cuda::CudaKernelProvider;
///
/// let provider = Arc::new(CudaKernelProvider::new(device, memory)?);
/// let mut executor = Executor::new(provider);
///
/// // Execute a plan
/// let result = executor.execute_plan(&plan)?;
/// ```
pub struct Executor {
    /// CUDA kernel provider for GPU operations
    provider: Arc<CudaKernelProvider>,
    /// Storage for named relations
    store: RelationStore,
    /// Mapping from RelId to relation name
    rel_names: HashMap<RelId, String>,
    /// Mapping from relation name to RelId
    name_to_rel: HashMap<String, RelId>,
    /// Runtime statistics for adaptive optimization
    stats: StatsManager,
    /// Cached build-side join indexes (adaptive indexing)
    join_index_cache: JoinIndexCache,
    /// Runtime configuration
    config: RuntimeConfig,
    /// Performance profiler for --stats output
    profiler: Profiler,
    /// ILP tensor mask registry
    ilp_registry: IlpRegistry,
    /// Last ILP tagged result metadata
    ilp_last_result: Option<IlpTaggedResult>,
}

impl Executor {
    /// Create a new executor with the given kernel provider
    ///
    /// # Arguments
    /// * `provider` - The CUDA kernel provider for GPU operations
    pub fn new(provider: Arc<CudaKernelProvider>) -> Self {
        Self::new_with_config(provider, RuntimeConfig::default())
    }

    /// Create a new executor with the given kernel provider and runtime config
    pub fn new_with_config(provider: Arc<CudaKernelProvider>, config: RuntimeConfig) -> Self {
        const DEFAULT_JOIN_INDEX_CACHE_BYTES: u64 = 256 * 1024 * 1024;
        let max_index_cache_bytes =
            (provider.memory().budget().device_bytes / 4).min(DEFAULT_JOIN_INDEX_CACHE_BYTES);
        Self {
            provider: provider.clone(),
            store: RelationStore::new(provider.clone()),
            rel_names: HashMap::new(),
            name_to_rel: HashMap::new(),
            stats: StatsManager::new(),
            join_index_cache: JoinIndexCache::new(max_index_cache_bytes),
            config,
            profiler: Profiler::default(),
            ilp_registry: IlpRegistry::new(),
            ilp_last_result: None,
        }
    }

    /// Enable or disable the performance profiler
    ///
    /// When enabled, execution statistics will be collected for --stats output.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiler = Profiler::new(enabled);
        if enabled {
            self.profiler
                .set_memory_budget(self.provider.memory().budget().device_bytes);
        }
    }

    /// Check if profiling is enabled
    pub fn is_profiling(&self) -> bool {
        self.profiler.is_enabled()
    }

    /// Get execution statistics
    ///
    /// Returns collected statistics if profiling was enabled.
    pub fn execution_stats(&self, total_output_rows: u64) -> ExecutionStats {
        self.profiler.execution_stats(total_output_rows)
    }

    /// Get a reference to the relation store
    pub fn store(&self) -> &RelationStore {
        &self.store
    }

    /// Get a mutable reference to the relation store
    pub fn store_mut(&mut self) -> &mut RelationStore {
        &mut self.store
    }

    /// Get a mutable reference to the ILP registry (RD-35).
    pub fn ilp_registry_mut(&mut self) -> &mut IlpRegistry {
        &mut self.ilp_registry
    }

    /// Get the last ILP tagged result (RD-35).
    pub fn ilp_last_result(&self) -> Option<&IlpTaggedResult> {
        self.ilp_last_result.as_ref()
    }

    /// Store a relation buffer and invalidate join indices.
    pub fn put_relation(&mut self, name: &str, buffer: CudaBuffer) {
        self.store_put(name, buffer);
    }

    /// Get a reference to the runtime statistics manager
    pub fn stats(&self) -> &StatsManager {
        &self.stats
    }

    /// Reset executor state for Monte Carlo sampling.
    ///
    /// Clears relation storage and join index cache while preserving relation registrations.
    pub fn reset_for_mc(&mut self) {
        self.store.clear();
        self.join_index_cache.clear();
    }

    /// Targeted MC reset: preserve base/static relations and clear dynamic ones.
    ///
    /// Unlike [`reset_for_mc`] which drops all relations, this method keeps the
    /// relations listed in `preserve` untouched, removes every other relation,
    /// then re-creates the relations specified in `clear_to_empty` as empty
    /// GPU buffers with the given schemas.  The join-index cache is fully
    /// invalidated because dynamic relations have changed.
    ///
    /// # Arguments
    /// * `preserve` - Relation names to keep as-is (base/static facts).
    /// * `clear_to_empty` - `(name, schema)` pairs for dynamic relations that
    ///   should be present but empty after the reset.
    pub fn reset_for_mc_relations(
        &mut self,
        preserve: &[&str],
        clear_to_empty: &[(&str, Schema)],
    ) -> Result<()> {
        let preserve_set: HashSet<&str> = preserve.iter().copied().collect();
        let existing_names: Vec<String> = self.store.names().map(|s| s.to_string()).collect();

        for name in &existing_names {
            if !preserve_set.contains(name.as_str()) {
                self.store.remove(name);
            }
        }

        for (name, schema) in clear_to_empty {
            let empty = self.provider.create_empty_buffer(schema.clone())?;
            self.store.put(name, empty);
        }

        self.join_index_cache.clear();
        Ok(())
    }

    /// Reset executor state for ILP attempt reuse.
    ///
    /// Clears ILP registry (masks + tagged results), relation storage,
    /// join index cache, stats, and profiler. Preserves relation name
    /// registrations (rel_names, name_to_rel) since those are immutable
    /// compile artifacts.
    pub fn reset_for_ilp(&mut self) {
        self.ilp_registry.clear();
        self.ilp_last_result = None;
        self.store.clear();
        self.join_index_cache.clear();
        self.stats = StatsManager::new();
        self.profiler = Profiler::default();
    }

    /// Get a mutable reference to the runtime statistics manager
    pub fn stats_mut(&mut self) -> &mut StatsManager {
        &mut self.stats
    }

    /// Capture a runtime statistics snapshot, including predicate name mappings.
    ///
    /// Use this snapshot to seed the compiler/optimizer on subsequent compilations.
    pub fn stats_snapshot(&self) -> StatsSnapshot {
        let mut snapshot = self.stats.snapshot();
        snapshot.rel_names = self
            .rel_names
            .iter()
            .map(|(id, name)| (*id, name.clone()))
            .collect();
        snapshot
    }

    fn store_put(&mut self, name: &str, buffer: CudaBuffer) {
        self.store.put(name, buffer);
        if let Some(&rel_id) = self.name_to_rel.get(name) {
            self.join_index_cache.invalidate_rel(rel_id);
        }
    }

    fn store_remove(&mut self, name: &str) -> Option<CudaBuffer> {
        if let Some(&rel_id) = self.name_to_rel.get(name) {
            self.join_index_cache.invalidate_rel(rel_id);
        }
        self.store.remove(name)
    }

    /// Register a relation name for a RelId
    ///
    /// This mapping is used when executing Scan nodes to look up relations
    /// by their RelId.
    ///
    /// # Arguments
    /// * `rel_id` - The relation identifier
    /// * `name` - The name to associate with the relation
    pub fn register_relation(&mut self, rel_id: RelId, name: &str) {
        self.rel_names.insert(rel_id, name.to_string());
        self.name_to_rel.insert(name.to_string(), rel_id);
        self.stats.register_relation(rel_id);
    }

    /// Get the relation name for a RelId
    fn get_rel_name(&self, rel_id: RelId) -> Option<&str> {
        self.rel_names.get(&rel_id).map(|s| s.as_str())
    }

    /// Execute a complete execution plan
    ///
    /// Iterates through strata in order, executing each one.
    /// Returns the result of the final query if present, or an empty buffer.
    ///
    /// # Arguments
    /// * `plan` - The execution plan to execute
    ///
    /// # Returns
    /// The result buffer from executing the plan
    ///
    /// # Errors
    /// Returns an error if any stratum or query execution fails
    pub fn execute_plan(&mut self, plan: &ExecutionPlan) -> Result<CudaBuffer> {
        // Execute strata in order
        for (idx, stratum) in plan.strata.iter().enumerate() {
            // Count rules and check if recursive
            let (num_rules, is_recursive) = stratum
                .sccs
                .iter()
                .map(|&scc_id| {
                    let rules = plan
                        .rules_by_scc
                        .get(scc_id as usize)
                        .map(|r| r.len())
                        .unwrap_or(0);
                    let recursive = plan
                        .sccs
                        .get(scc_id as usize)
                        .map(|s| s.is_recursive)
                        .unwrap_or(false);
                    (rules, recursive)
                })
                .fold((0, false), |(r, rec), (nr, nrec)| (r + nr, rec || nrec));

            self.profiler.begin_stratum(idx, num_rules, is_recursive);
            self.execute_stratum_impl(stratum, plan)?;

            // Record peak memory after stratum
            let mem_bytes = self.provider.memory().allocated_bytes();
            self.profiler.record_peak_memory(mem_bytes);

            self.profiler.end_stratum();
        }

        // Ensure all GPU work completes before returning control to callers.
        self.provider.device().synchronize()?;

        // If there are no strata, return empty buffer
        self.provider.create_empty_buffer(Schema::new(vec![]))
    }

    /// Apply base-relation deltas and recompute affected SCCs (no recompilation).
    ///
    /// This provides correctness for both insertions and deletions by recomputing any SCCs that
    /// depend (directly or transitively) on the changed relations.
    pub fn apply_deltas_and_recompute(
        &mut self,
        plan: &ExecutionPlan,
        deltas: &HashMap<String, RelationDelta>,
    ) -> Result<()> {
        if deltas.is_empty() {
            return Ok(());
        }

        let has_deletes = deltas
            .values()
            .any(|d| d.delete.as_ref().map(|b| !b.is_empty()).unwrap_or(false));

        // 1) Apply EDB updates.
        for (name, delta) in deltas {
            let existing = self.store.get(name);

            let base_schema = existing
                .map(|b| b.schema().clone())
                .or_else(|| delta.insert.as_ref().map(|b| b.schema().clone()))
                .or_else(|| delta.delete.as_ref().map(|b| b.schema().clone()))
                .ok_or_else(|| {
                    XlogError::Execution(format!(
                        "Delta update for {} has no existing relation and no schema",
                        name
                    ))
                })?;

            let mut updated = if let Some(buf) = existing {
                self.clone_buffer(buf)?
            } else {
                self.create_empty_buffer(base_schema)?
            };

            if let Some(delete_buf) = &delta.delete {
                updated = self.provider.diff_gpu(&updated, delete_buf)?;
            }
            if let Some(insert_buf) = &delta.insert {
                updated = self.provider.union_gpu(&updated, insert_buf)?;
            }

            self.store_put(name, updated);
        }

        // 2) Compute affected SCC closure.
        let changed_preds: HashSet<&str> = deltas.keys().map(|s| s.as_str()).collect();

        let mut pred_to_scc: HashMap<&str, u32> = HashMap::new();
        for scc in &plan.sccs {
            for pred in &scc.predicates {
                pred_to_scc.insert(pred.as_str(), scc.id);
            }
        }

        let mut dependents: HashMap<u32, Vec<u32>> = HashMap::new();
        for (scc_id, rules) in plan.rules_by_scc.iter().enumerate() {
            let scc_id = scc_id as u32;
            for rule in rules {
                let mut rels = Vec::new();
                Self::collect_scan_rels(&rule.body, &mut rels);
                for rel in rels {
                    let Some(name) = self.get_rel_name(rel) else {
                        continue;
                    };
                    let Some(&dep_scc) = pred_to_scc.get(name) else {
                        continue;
                    };
                    if dep_scc == scc_id {
                        continue;
                    }
                    dependents.entry(dep_scc).or_default().push(scc_id);
                }
            }
        }

        let mut affected: HashSet<u32> = HashSet::new();
        let mut queue: Vec<u32> = Vec::new();
        for pred in &changed_preds {
            if let Some(&scc) = pred_to_scc.get(*pred) {
                affected.insert(scc);
                queue.push(scc);
            }
        }

        while let Some(scc) = queue.pop() {
            if let Some(deps) = dependents.get(&scc) {
                for &next in deps {
                    if affected.insert(next) {
                        queue.push(next);
                    }
                }
            }
        }

        if affected.is_empty() {
            return Ok(());
        }

        fn contains_non_monotonic_ops(node: &RirNode) -> bool {
            match node {
                RirNode::Unit | RirNode::Scan { .. } => false,
                RirNode::Filter { input, .. }
                | RirNode::Project { input, .. }
                | RirNode::Distinct { input, .. } => contains_non_monotonic_ops(input),
                RirNode::Union { inputs } => inputs.iter().any(contains_non_monotonic_ops),
                RirNode::GroupBy { .. } | RirNode::Diff { .. } => true,
                RirNode::Join {
                    left,
                    right,
                    join_type,
                    ..
                } => {
                    matches!(join_type, JoinType::Anti | JoinType::LeftOuter)
                        || contains_non_monotonic_ops(left)
                        || contains_non_monotonic_ops(right)
                }
                RirNode::Fixpoint {
                    base, recursive, ..
                } => contains_non_monotonic_ops(base) || contains_non_monotonic_ops(recursive),
                RirNode::TensorMaskedJoin { .. } => false,
            }
        }

        // 3) Decide which SCCs must be recomputed (cleared first).
        //
        // If there are deletes, we always recompute for correctness.
        // If there are only inserts, we can incrementally update SCCs that are monotone w.r.t.
        // insertion (no anti-joins, diffs, or aggregates) and do a targeted recompute for the rest.
        let mut recompute_sccs: HashSet<u32> = HashSet::new();
        if has_deletes {
            recompute_sccs = affected.clone();
        } else {
            for &scc_id in &affected {
                if let Some(rules) = plan.rules_by_scc.get(scc_id as usize) {
                    if rules.iter().any(|r| contains_non_monotonic_ops(&r.body)) {
                        recompute_sccs.insert(scc_id);
                    }
                }
            }

            // If any SCC is recomputed due to non-monotonic ops, all dependents must also be
            // recomputed because their prior outputs may now be invalid.
            let mut queue: Vec<u32> = recompute_sccs.iter().copied().collect();
            while let Some(scc) = queue.pop() {
                if let Some(deps) = dependents.get(&scc) {
                    for &next in deps {
                        if !affected.contains(&next) {
                            continue;
                        }
                        if recompute_sccs.insert(next) {
                            queue.push(next);
                        }
                    }
                }
            }
        }

        // 4) Clear IDB relations for SCCs we are recomputing (but never clear directly-updated bases).
        for scc_id in &recompute_sccs {
            let Some(scc) = plan.sccs.iter().find(|s| s.id == *scc_id) else {
                continue;
            };

            for pred in &scc.predicates {
                if changed_preds.contains(pred.as_str()) {
                    continue;
                }
                let schema = self
                    .store
                    .get(pred)
                    .map(|b| b.schema().clone())
                    .or_else(|| {
                        plan.rules_by_scc
                            .get(*scc_id as usize)
                            .and_then(|rules| rules.iter().find(|r| r.head == pred.as_str()))
                            .and_then(|r| {
                                let schema = r.meta.schema.clone();
                                if schema.arity() > 0 {
                                    Some(schema)
                                } else {
                                    None
                                }
                            })
                    })
                    .ok_or_else(|| {
                        XlogError::Execution(format!(
                            "Missing schema for predicate {} during recompute",
                            pred
                        ))
                    })?;

                let empty = self.create_empty_buffer(schema)?;
                self.store_put(pred, empty);
            }
        }

        // 5) Re-execute affected SCCs in plan order (incremental for insert-only monotone SCCs).
        for stratum in &plan.strata {
            for &scc_id in &stratum.sccs {
                if !affected.contains(&scc_id) {
                    continue;
                }
                let rules = plan.rules_by_scc.get(scc_id as usize).ok_or_else(|| {
                    XlogError::Execution(format!("Missing rules for SCC {}", scc_id))
                })?;
                let is_recursive = plan
                    .sccs
                    .iter()
                    .find(|s| s.id == scc_id)
                    .map(|s| s.is_recursive)
                    .unwrap_or(false);

                if is_recursive {
                    self.execute_recursive_scc(rules)?;
                } else {
                    self.execute_non_recursive_scc(rules)?;
                }
            }
        }

        Ok(())
    }

    fn collect_scan_rels(node: &RirNode, out: &mut Vec<RelId>) {
        match node {
            RirNode::Unit => {}
            RirNode::Scan { rel } => out.push(*rel),
            RirNode::Filter { input, .. } | RirNode::Project { input, .. } => {
                Self::collect_scan_rels(input, out);
            }
            RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                Self::collect_scan_rels(left, out);
                Self::collect_scan_rels(right, out);
            }
            RirNode::GroupBy { input, .. } | RirNode::Distinct { input, .. } => {
                Self::collect_scan_rels(input, out);
            }
            RirNode::Union { inputs } => {
                for input in inputs {
                    Self::collect_scan_rels(input, out);
                }
            }
            RirNode::Fixpoint {
                base, recursive, ..
            } => {
                Self::collect_scan_rels(base, out);
                Self::collect_scan_rels(recursive, out);
            }
            RirNode::TensorMaskedJoin { rel_index, .. } => {
                for (rel_id, _) in rel_index {
                    out.push(*rel_id);
                }
            }
        }
    }

    fn rewrite_scan_nth(
        node: &RirNode,
        target: RelId,
        nth: usize,
        replacement: RelId,
    ) -> Option<RirNode> {
        let mut remaining = nth;
        let (rewritten, replaced) =
            Self::rewrite_scan_nth_impl(node, target, &mut remaining, replacement);
        replaced.then_some(rewritten)
    }

    fn rewrite_scan_nth_impl(
        node: &RirNode,
        target: RelId,
        remaining: &mut usize,
        replacement: RelId,
    ) -> (RirNode, bool) {
        match node {
            RirNode::Unit => (RirNode::Unit, false),
            RirNode::Scan { rel } => {
                if *rel == target {
                    if *remaining == 0 {
                        return (RirNode::Scan { rel: replacement }, true);
                    }
                    *remaining -= 1;
                }
                (node.clone(), false)
            }

            RirNode::Filter { input, predicate } => {
                let (new_input, replaced) =
                    Self::rewrite_scan_nth_impl(input, target, remaining, replacement);
                (
                    RirNode::Filter {
                        input: Box::new(new_input),
                        predicate: predicate.clone(),
                    },
                    replaced,
                )
            }

            RirNode::Project { input, columns } => {
                let (new_input, replaced) =
                    Self::rewrite_scan_nth_impl(input, target, remaining, replacement);
                (
                    RirNode::Project {
                        input: Box::new(new_input),
                        columns: columns.clone(),
                    },
                    replaced,
                )
            }

            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
            } => {
                let (new_left, replaced_left) =
                    Self::rewrite_scan_nth_impl(left, target, remaining, replacement);
                if replaced_left {
                    return (
                        RirNode::Join {
                            left: Box::new(new_left),
                            right: right.clone(),
                            left_keys: left_keys.clone(),
                            right_keys: right_keys.clone(),
                            join_type: *join_type,
                        },
                        true,
                    );
                }
                let (new_right, replaced_right) =
                    Self::rewrite_scan_nth_impl(right, target, remaining, replacement);
                (
                    RirNode::Join {
                        left: Box::new(new_left),
                        right: Box::new(new_right),
                        left_keys: left_keys.clone(),
                        right_keys: right_keys.clone(),
                        join_type: *join_type,
                    },
                    replaced_right,
                )
            }

            RirNode::GroupBy {
                input,
                key_cols,
                aggs,
            } => {
                let (new_input, replaced) =
                    Self::rewrite_scan_nth_impl(input, target, remaining, replacement);
                (
                    RirNode::GroupBy {
                        input: Box::new(new_input),
                        key_cols: key_cols.clone(),
                        aggs: aggs.clone(),
                    },
                    replaced,
                )
            }

            RirNode::Union { inputs } => {
                let mut replaced_any = false;
                let mut new_inputs = Vec::with_capacity(inputs.len());
                for input in inputs {
                    let (new_input, replaced) =
                        Self::rewrite_scan_nth_impl(input, target, remaining, replacement);
                    replaced_any |= replaced;
                    new_inputs.push(new_input);
                }
                (RirNode::Union { inputs: new_inputs }, replaced_any)
            }

            RirNode::Distinct { input, key_cols } => {
                let (new_input, replaced) =
                    Self::rewrite_scan_nth_impl(input, target, remaining, replacement);
                (
                    RirNode::Distinct {
                        input: Box::new(new_input),
                        key_cols: key_cols.clone(),
                    },
                    replaced,
                )
            }

            RirNode::Diff { left, right } => {
                let (new_left, replaced_left) =
                    Self::rewrite_scan_nth_impl(left, target, remaining, replacement);
                if replaced_left {
                    return (
                        RirNode::Diff {
                            left: Box::new(new_left),
                            right: right.clone(),
                        },
                        true,
                    );
                }
                let (new_right, replaced_right) =
                    Self::rewrite_scan_nth_impl(right, target, remaining, replacement);
                (
                    RirNode::Diff {
                        left: Box::new(new_left),
                        right: Box::new(new_right),
                    },
                    replaced_right,
                )
            }

            RirNode::Fixpoint {
                scc_id,
                base,
                recursive,
                delta_rel,
                full_rel,
            } => {
                let (new_base, replaced_base) =
                    Self::rewrite_scan_nth_impl(base, target, remaining, replacement);
                if replaced_base {
                    return (
                        RirNode::Fixpoint {
                            scc_id: *scc_id,
                            base: Box::new(new_base),
                            recursive: recursive.clone(),
                            delta_rel: *delta_rel,
                            full_rel: *full_rel,
                        },
                        true,
                    );
                }
                let (new_recursive, replaced_recursive) =
                    Self::rewrite_scan_nth_impl(recursive, target, remaining, replacement);
                (
                    RirNode::Fixpoint {
                        scc_id: *scc_id,
                        base: Box::new(new_base),
                        recursive: Box::new(new_recursive),
                        delta_rel: *delta_rel,
                        full_rel: *full_rel,
                    },
                    replaced_recursive,
                )
            }
            RirNode::TensorMaskedJoin { .. } => {
                // TensorMaskedJoin is a leaf node — no child scans to rewrite.
                (node.clone(), false)
            }
        }
    }

    /// Execute a stratum (public API)
    ///
    /// This method cannot be called directly because stratum execution requires
    /// access to the full ExecutionPlan (for rules_by_scc mapping). Use
    /// `execute_plan` instead, which processes all strata with proper context.
    ///
    /// # Arguments
    /// * `_stratum` - The stratum (unused - see error)
    ///
    /// # Returns
    /// Always returns an error indicating this method should not be called directly
    ///
    /// # Errors
    /// Always returns an error. Use `execute_plan` instead.
    pub fn execute_stratum(&mut self, _stratum: &Stratum) -> Result<()> {
        Err(XlogError::Execution(
            "execute_stratum cannot be called directly; use execute_plan instead which provides \
             the required rules_by_scc context"
                .to_string(),
        ))
    }

    // ============== Node execution implementations ==============

    /// Execute a Scan node — stays in mod.rs (simple store lookup).
    ///
    /// Looks up the relation by RelId and returns a clone of its buffer.
    fn execute_scan(&mut self, rel: RelId) -> Result<CudaBuffer> {
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

        // Clone the buffer
        self.clone_buffer(buffer)
    }

    /// Evaluate a predicate expression for a single row
    #[cfg(test)]
    fn evaluate_predicate(
        expr: &Expr,
        columns: &[Vec<u8>],
        row_idx: usize,
        schema: &Schema,
    ) -> Result<bool> {
        match expr {
            Expr::Column(col_idx) => {
                // Interpret column value as boolean
                let col_type = schema.column_type(*col_idx);
                if let Some(ScalarType::Bool) = col_type {
                    Ok(columns
                        .get(*col_idx)
                        .map(|c| c.get(row_idx).copied().unwrap_or(0) != 0)
                        .unwrap_or(false))
                } else {
                    // Non-bool columns: check if non-zero
                    Ok(true)
                }
            }

            Expr::Const(ConstValue::Bool(b)) => Ok(*b),
            Expr::Const(_) => Ok(true), // Non-bool constants are truthy

            Expr::Compare { left, op, right } => {
                let use_float =
                    Self::expr_may_be_float(left, schema) || Self::expr_may_be_float(right, schema);

                if use_float {
                    let left_val = Self::evaluate_expr_as_f64(left, columns, row_idx, schema)?;
                    let right_val = Self::evaluate_expr_as_f64(right, columns, row_idx, schema)?;

                    Ok(match op {
                        CompareOp::Eq => left_val == right_val,
                        CompareOp::Ne => left_val != right_val,
                        CompareOp::Lt => left_val < right_val,
                        CompareOp::Le => left_val <= right_val,
                        CompareOp::Gt => left_val > right_val,
                        CompareOp::Ge => left_val >= right_val,
                    })
                } else {
                    let left_val = Self::evaluate_expr_as_i64(left, columns, row_idx, schema)?;
                    let right_val = Self::evaluate_expr_as_i64(right, columns, row_idx, schema)?;

                    Ok(match op {
                        CompareOp::Eq => left_val == right_val,
                        CompareOp::Ne => left_val != right_val,
                        CompareOp::Lt => left_val < right_val,
                        CompareOp::Le => left_val <= right_val,
                        CompareOp::Gt => left_val > right_val,
                        CompareOp::Ge => left_val >= right_val,
                    })
                }
            }

            Expr::And(exprs) => {
                for e in exprs {
                    if !Self::evaluate_predicate(e, columns, row_idx, schema)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }

            Expr::Or(exprs) => {
                for e in exprs {
                    if Self::evaluate_predicate(e, columns, row_idx, schema)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }

            Expr::Not(inner) => Ok(!Self::evaluate_predicate(inner, columns, row_idx, schema)?),

            // Arithmetic expressions are not used as predicates directly
            Expr::Add(_, _)
            | Expr::Sub(_, _)
            | Expr::Mul(_, _)
            | Expr::Div(_, _)
            | Expr::Mod(_, _)
            | Expr::Abs(_)
            | Expr::Min(_, _)
            | Expr::Max(_, _)
            | Expr::Pow(_, _)
            | Expr::Cast(_, _)
            | Expr::Conditional { .. } => Err(XlogError::Execution(
                "Arithmetic expression cannot be evaluated as boolean predicate".into(),
            )),
        }
    }

    pub(crate) fn expr_may_be_float(expr: &Expr, schema: &Schema) -> bool {
        match expr {
            Expr::Column(col_idx) => matches!(
                schema.column_type(*col_idx),
                Some(ScalarType::F32 | ScalarType::F64)
            ),
            Expr::Const(ConstValue::F32(_) | ConstValue::F64(_)) => true,
            Expr::Cast(_, ScalarType::F32 | ScalarType::F64) => true,
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Mod(l, r)
            | Expr::Min(l, r)
            | Expr::Max(l, r)
            | Expr::Pow(l, r) => {
                Self::expr_may_be_float(l, schema) || Self::expr_may_be_float(r, schema)
            }
            Expr::Abs(inner) | Expr::Cast(inner, _) => Self::expr_may_be_float(inner, schema),
            _ => false,
        }
    }

    #[cfg(test)]
    fn evaluate_expr_as_f64(
        expr: &Expr,
        columns: &[Vec<u8>],
        row_idx: usize,
        schema: &Schema,
    ) -> Result<f64> {
        match expr {
            Expr::Column(col_idx) => {
                let col_type = schema.column_type(*col_idx).unwrap_or(ScalarType::U32);
                let col_data = columns
                    .get(*col_idx)
                    .ok_or_else(|| XlogError::Execution(format!("Column {} not found", col_idx)))?;

                let type_size = col_type.size_bytes();
                let start = row_idx * type_size;

                Ok(match col_type {
                    ScalarType::F64 => {
                        let bytes = &col_data[start..start + 8];
                        f64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ])
                    }
                    ScalarType::F32 => {
                        let bytes = &col_data[start..start + 4];
                        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
                    }
                    ScalarType::U32 | ScalarType::Symbol => {
                        let bytes = &col_data[start..start + 4];
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
                    }
                    ScalarType::I32 => {
                        let bytes = &col_data[start..start + 4];
                        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
                    }
                    ScalarType::U64 => {
                        let bytes = &col_data[start..start + 8];
                        u64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ]) as f64
                    }
                    ScalarType::I64 => {
                        let bytes = &col_data[start..start + 8];
                        i64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ]) as f64
                    }
                    ScalarType::Bool => col_data.get(start).copied().unwrap_or(0) as f64,
                })
            }

            Expr::Const(val) => Ok(match val {
                ConstValue::U32(v) => *v as f64,
                ConstValue::I32(v) => *v as f64,
                ConstValue::U64(v) => *v as f64,
                ConstValue::I64(v) => *v as f64,
                ConstValue::Bool(b) => {
                    if *b {
                        1.0
                    } else {
                        0.0
                    }
                }
                ConstValue::F32(f) => *f as f64,
                ConstValue::F64(f) => *f,
                ConstValue::Symbol(_) => {
                    return Err(XlogError::Execution(
                        "Cannot evaluate Symbol constant as f64".to_string(),
                    ));
                }
            }),

            Expr::Add(l, r) => Ok(Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?
                + Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?),
            Expr::Sub(l, r) => Ok(Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?
                - Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?),
            Expr::Mul(l, r) => Ok(Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?
                * Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?),
            Expr::Div(l, r) => {
                let left_val = Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?;
                if right_val == 0.0 {
                    return Err(XlogError::Execution("Division by zero".to_string()));
                }
                Ok(left_val / right_val)
            }
            Expr::Mod(l, r) => {
                let left_val = Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?;
                if right_val == 0.0 {
                    return Err(XlogError::Execution("Modulo by zero".to_string()));
                }
                Ok(left_val % right_val)
            }
            Expr::Abs(inner) => {
                Ok(Self::evaluate_expr_as_f64(inner, columns, row_idx, schema)?.abs())
            }
            Expr::Min(l, r) => Ok(Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?
                .min(Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?)),
            Expr::Max(l, r) => Ok(Self::evaluate_expr_as_f64(l, columns, row_idx, schema)?
                .max(Self::evaluate_expr_as_f64(r, columns, row_idx, schema)?)),
            Expr::Pow(base, exp) => Ok(Self::evaluate_expr_as_f64(base, columns, row_idx, schema)?
                .powf(Self::evaluate_expr_as_f64(exp, columns, row_idx, schema)?)),
            Expr::Cast(inner, target_type) => match target_type {
                ScalarType::F64 => Self::evaluate_expr_as_f64(inner, columns, row_idx, schema),
                ScalarType::F32 => {
                    Ok(Self::evaluate_expr_as_f64(inner, columns, row_idx, schema)? as f32 as f64)
                }
                _ => Ok(Self::evaluate_expr_as_i64(inner, columns, row_idx, schema)? as f64),
            },

            _ => Err(XlogError::Execution(
                "Cannot evaluate compound expression as f64".to_string(),
            )),
        }
    }

    /// Evaluate an expression as an i64 value
    #[cfg(test)]
    fn evaluate_expr_as_i64(
        expr: &Expr,
        columns: &[Vec<u8>],
        row_idx: usize,
        schema: &Schema,
    ) -> Result<i64> {
        match expr {
            Expr::Column(col_idx) => {
                let col_type = schema.column_type(*col_idx).unwrap_or(ScalarType::U32);
                let col_data = columns
                    .get(*col_idx)
                    .ok_or_else(|| XlogError::Execution(format!("Column {} not found", col_idx)))?;

                let type_size = col_type.size_bytes();
                let start = row_idx * type_size;

                Ok(match col_type {
                    ScalarType::U32 => {
                        let bytes = &col_data[start..start + 4];
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
                    }
                    ScalarType::I32 => {
                        let bytes = &col_data[start..start + 4];
                        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
                    }
                    ScalarType::U64 => {
                        let bytes = &col_data[start..start + 8];
                        u64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ]) as i64
                    }
                    ScalarType::I64 => {
                        let bytes = &col_data[start..start + 8];
                        i64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ])
                    }
                    ScalarType::Bool => col_data.get(start).copied().unwrap_or(0) as i64,
                    ScalarType::Symbol => {
                        let bytes = &col_data[start..start + 4];
                        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
                    }
                    ScalarType::F32 => {
                        let bytes = &col_data[start..start + 4];
                        let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                        val as i64
                    }
                    ScalarType::F64 => {
                        let bytes = &col_data[start..start + 8];
                        let val = f64::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ]);
                        val as i64
                    }
                })
            }

            Expr::Const(val) => Ok(match val {
                ConstValue::U32(v) => *v as i64,
                ConstValue::I32(v) => *v as i64,
                ConstValue::U64(v) => *v as i64,
                ConstValue::I64(v) => *v,
                ConstValue::Bool(b) => *b as i64,
                ConstValue::F32(f) => *f as i64,
                ConstValue::F64(f) => *f as i64,
                ConstValue::Symbol(_) => 0,
            }),

            // Arithmetic expressions - evaluate them and return the result
            Expr::Add(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                Ok(left_val.wrapping_add(right_val))
            }
            Expr::Sub(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                Ok(left_val.wrapping_sub(right_val))
            }
            Expr::Mul(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                Ok(left_val.wrapping_mul(right_val))
            }
            Expr::Div(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                if right_val == 0 {
                    return Err(XlogError::Execution("Division by zero".to_string()));
                }
                Ok(left_val / right_val)
            }
            Expr::Mod(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                if right_val == 0 {
                    return Err(XlogError::Execution("Modulo by zero".to_string()));
                }
                Ok(left_val % right_val)
            }
            Expr::Abs(inner) => {
                let val = Self::evaluate_expr_as_i64(inner, columns, row_idx, schema)?;
                Ok(val.abs())
            }
            Expr::Min(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                Ok(left_val.min(right_val))
            }
            Expr::Max(l, r) => {
                let left_val = Self::evaluate_expr_as_i64(l, columns, row_idx, schema)?;
                let right_val = Self::evaluate_expr_as_i64(r, columns, row_idx, schema)?;
                Ok(left_val.max(right_val))
            }
            Expr::Pow(base, exp) => {
                let base_val = Self::evaluate_expr_as_i64(base, columns, row_idx, schema)?;
                let exp_val = Self::evaluate_expr_as_i64(exp, columns, row_idx, schema)?;
                if exp_val < 0 {
                    return Err(XlogError::Execution(
                        "Negative exponent in integer pow".to_string(),
                    ));
                } else if exp_val > u32::MAX as i64 {
                    // Exponent too large - would overflow anyway
                    Ok(i64::MAX)
                } else {
                    Ok(base_val.pow(exp_val as u32))
                }
            }
            Expr::Cast(inner, _target_type) => {
                // For i64 evaluation, cast is a no-op since we evaluate everything as i64
                Self::evaluate_expr_as_i64(inner, columns, row_idx, schema)
            }

            _ => Err(XlogError::Execution(
                "Cannot evaluate compound expression as value".to_string(),
            )),
        }
    }

    /// Get the relation name for a RelId, creating a default name if not registered
    fn get_or_create_rel_name(&mut self, rel_id: RelId, default: &str) -> String {
        if let Some(name) = self.rel_names.get(&rel_id) {
            name.clone()
        } else {
            self.register_relation(rel_id, default);
            default.to_string()
        }
    }

    // ============== Helper methods ==============

    /// Create an empty buffer with the given schema
    fn create_empty_buffer(&self, schema: Schema) -> Result<CudaBuffer> {
        self.provider.create_empty_buffer(schema)
    }

    /// Clone a buffer (device-to-device copy)
    fn clone_buffer(&self, buffer: &CudaBuffer) -> Result<CudaBuffer> {
        if buffer.is_empty() {
            return self.create_empty_buffer(buffer.schema().clone());
        }

        let mut result_columns = Vec::with_capacity(buffer.arity());

        for col_idx in 0..buffer.arity() {
            let col_type_size = buffer
                .schema()
                .column_type(col_idx)
                .map(|t| t.size_bytes())
                .unwrap_or(4);
            let bytes = (buffer.num_rows() as usize) * col_type_size;

            if let Some(src_col) = buffer.column(col_idx) {
                let mut dst_col = self.provider.memory().alloc::<u8>(bytes)?;
                if bytes > 0 {
                    self.provider
                        .device()
                        .inner()
                        .dtod_copy(src_col, &mut dst_col)
                        .map_err(|e| {
                            XlogError::Execution(format!("Failed to clone column on device: {}", e))
                        })?;
                }
                result_columns.push(dst_col.into());
            }
        }

        let d_num_rows = self.clone_device_row_count(buffer)?;
        Ok(CudaBuffer::from_columns(
            result_columns,
            buffer.num_rows(),
            d_num_rows,
            buffer.schema().clone(),
        ))
    }

    fn clone_device_row_count(&self, buffer: &CudaBuffer) -> Result<TrackedCudaSlice<u32>> {
        let mut d_num_rows = self.provider.memory().alloc::<u32>(1)?;
        self.provider
            .device()
            .inner()
            .dtod_copy(buffer.num_rows_device(), &mut d_num_rows)
            .map_err(|e| XlogError::Execution(format!("Failed to copy row count: {}", e)))?;
        Ok(d_num_rows)
    }

    fn buffer_row_count(&self, buffer: &CudaBuffer) -> Result<u32> {
        if let Some(n) = buffer.cached_row_count() {
            return Ok(n);
        }
        let mut host_rows = [0u32];
        self.provider
            .device()
            .inner()
            .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
            .map_err(|e| XlogError::Execution(format!("Failed to read row count: {}", e)))?;
        buffer.set_cached_row_count_if_unset(host_rows[0]);
        Ok(host_rows[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::MemoryBudget;
    use xlog_cuda::{CudaDevice, GpuMemoryManager};
    use xlog_ir::{CompiledRule, RirMeta, Scc};

    fn has_cuda_device() -> bool {
        // Check if CUDA device is available using CudaDevice wrapper
        CudaDevice::new(0).is_ok()
    }

    fn create_test_executor() -> Option<Executor> {
        if !has_cuda_device() {
            return None;
        }
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GB
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        let provider = Arc::new(CudaKernelProvider::new(device, memory).ok()?);
        Some(Executor::new(provider))
    }

    fn device_row_count(executor: &Executor, rows: u64) -> TrackedCudaSlice<u32> {
        let rows_u32 = u32::try_from(rows).expect("row count fits u32");
        let mut d_num_rows = executor.provider.memory().alloc::<u32>(1).expect("alloc");
        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&[rows_u32], &mut d_num_rows)
            .expect("htod");
        d_num_rows
    }

    fn create_test_buffer(executor: &Executor, data: &[u32], col_name: &str) -> CudaBuffer {
        let schema = Schema::new(vec![(col_name.to_string(), ScalarType::U32)]);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut col = executor
            .provider
            .memory()
            .alloc::<u8>(bytes.len())
            .expect("alloc");
        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .expect("htod");

        let rows = data.len() as u64;
        let d_num_rows = device_row_count(executor, rows);
        CudaBuffer::from_columns(vec![col.into()], rows, d_num_rows, schema)
    }

    fn read_buffer_u32(executor: &Executor, buffer: &CudaBuffer, col: usize) -> Vec<u32> {
        executor
            .provider
            .download_column::<u32>(buffer, col)
            .unwrap_or_default()
    }

    fn buffer_row_count(executor: &Executor, buffer: &CudaBuffer) -> u32 {
        if let Some(n) = buffer.cached_row_count() {
            return n;
        }
        let mut host_rows = [0u32];
        executor
            .provider
            .device()
            .inner()
            .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
            .expect("dtoh row count");
        buffer.set_cached_row_count_if_unset(host_rows[0]);
        host_rows[0]
    }

    fn to_f64_column_bytes(values: &[f64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn to_f32_column_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    // ============== Basic Executor Tests ==============

    #[test]
    fn test_executor_creation() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        assert!(executor.store().is_empty());
    }

    #[test]
    fn test_predicate_f64_comparisons() {
        let schema = Schema::new(vec![("x".to_string(), ScalarType::F64)]);
        let values = [1.0f64, 2.0, 3.0, f64::NAN];
        let columns = vec![to_f64_column_bytes(&values)];

        let gt_two = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Gt,
            right: Box::new(Expr::Const(ConstValue::F64(2.0))),
        };

        let results: Vec<bool> = (0..values.len())
            .map(|row| Executor::evaluate_predicate(&gt_two, &columns, row, &schema).unwrap())
            .collect();
        assert_eq!(results, vec![false, false, true, false]);

        let eq_nan = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Eq,
            right: Box::new(Expr::Const(ConstValue::F64(f64::NAN))),
        };
        let results: Vec<bool> = (0..values.len())
            .map(|row| Executor::evaluate_predicate(&eq_nan, &columns, row, &schema).unwrap())
            .collect();
        assert_eq!(results, vec![false, false, false, false]);

        let ne_nan = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Ne,
            right: Box::new(Expr::Const(ConstValue::F64(f64::NAN))),
        };
        let results: Vec<bool> = (0..values.len())
            .map(|row| Executor::evaluate_predicate(&ne_nan, &columns, row, &schema).unwrap())
            .collect();
        assert_eq!(results, vec![true, true, true, true]);
    }

    #[test]
    fn test_predicate_f32_comparisons() {
        let schema = Schema::new(vec![("x".to_string(), ScalarType::F32)]);
        let values = [1.0f32, 2.0, 3.0, f32::NAN];
        let columns = vec![to_f32_column_bytes(&values)];

        let le_two = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Le,
            right: Box::new(Expr::Const(ConstValue::F32(2.0))),
        };

        let results: Vec<bool> = (0..values.len())
            .map(|row| Executor::evaluate_predicate(&le_two, &columns, row, &schema).unwrap())
            .collect();
        assert_eq!(results, vec![true, true, false, false]);
    }

    #[test]
    fn test_predicate_mixed_float_int_comparisons() {
        let schema = Schema::new(vec![
            ("x".to_string(), ScalarType::F64),
            ("y".to_string(), ScalarType::U32),
        ]);

        let x = [1.5f64, 2.0, 2.5];
        let y = [1u32, 2, 3];
        let columns = vec![
            to_f64_column_bytes(&x),
            y.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ];

        let x_gt_2 = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Gt,
            right: Box::new(Expr::Const(ConstValue::U32(2))),
        };
        let results: Vec<bool> = (0..x.len())
            .map(|row| Executor::evaluate_predicate(&x_gt_2, &columns, row, &schema).unwrap())
            .collect();
        assert_eq!(results, vec![false, false, true]);

        let y_lt_2_5 = Expr::Compare {
            left: Box::new(Expr::Column(1)),
            op: CompareOp::Lt,
            right: Box::new(Expr::Const(ConstValue::F64(2.5))),
        };
        let results: Vec<bool> = (0..y.len())
            .map(|row| Executor::evaluate_predicate(&y_lt_2_5, &columns, row, &schema).unwrap())
            .collect();
        assert_eq!(results, vec![true, true, false]);
    }

    #[test]
    fn test_register_and_get_relation() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Register a relation
        executor.register_relation(RelId(1), "test_rel");

        // Verify mapping
        assert_eq!(executor.get_rel_name(RelId(1)), Some("test_rel"));
        assert_eq!(executor.get_rel_name(RelId(2)), None);
    }

    // ============== Scan Node Tests ==============

    #[test]
    fn test_execute_scan_not_found() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        executor.register_relation(RelId(1), "missing_rel");

        let node = RirNode::Scan { rel: RelId(1) };
        let result = executor.execute_node(&node);

        assert!(result.is_err());
    }

    #[test]
    fn test_execute_scan_success() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create and store a buffer
        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        executor.store_mut().put("test_rel", buffer);
        executor.register_relation(RelId(1), "test_rel");

        // Execute scan
        let node = RirNode::Scan { rel: RelId(1) };
        let result = executor.execute_node(&node);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 5);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    // ============== Filter Node Tests ==============

    #[test]
    fn test_execute_filter_empty_input() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = executor.create_empty_buffer(schema).unwrap();

        let predicate = Expr::Const(ConstValue::Bool(true));
        let result = executor.execute_filter(&empty, &predicate);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
    }

    #[test]
    fn test_execute_filter_all_match() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        let predicate = Expr::Const(ConstValue::Bool(true));

        let result = executor.execute_filter(&buffer, &predicate);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 5);
    }

    #[test]
    fn test_execute_filter_none_match() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        let predicate = Expr::Const(ConstValue::Bool(false));

        let result = executor.execute_filter(&buffer, &predicate);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
    }

    #[test]
    fn test_execute_filter_comparison() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");

        // Filter: key > 3
        let predicate = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Gt,
            right: Box::new(Expr::Const(ConstValue::U32(3))),
        };

        let result = executor.execute_filter(&buffer, &predicate);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 2);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![4, 5]);
    }

    #[test]
    fn test_execute_filter_and() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");

        // Filter: key >= 2 AND key <= 4
        let predicate = Expr::And(vec![
            Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Ge,
                right: Box::new(Expr::Const(ConstValue::U32(2))),
            },
            Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Le,
                right: Box::new(Expr::Const(ConstValue::U32(4))),
            },
        ]);

        let result = executor.execute_filter(&buffer, &predicate);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![2, 3, 4]);
    }

    // ============== Project Node Tests ==============

    #[test]
    fn test_execute_project_empty_input() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U32),
        ]);
        let empty = executor.create_empty_buffer(schema).unwrap();

        let result = executor.execute_project(&empty, &[ProjectExpr::Column(0)]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
        assert_eq!(result.arity(), 1);
    }

    #[test]
    fn test_execute_project_reorder() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create a 2-column buffer
        let schema = Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U32),
        ]);

        let a_data: Vec<u8> = [1u32, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_data: Vec<u8> = [10u32, 20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let mut col_a = executor
            .provider
            .memory()
            .alloc::<u8>(a_data.len())
            .unwrap();
        let mut col_b = executor
            .provider
            .memory()
            .alloc::<u8>(b_data.len())
            .unwrap();

        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&a_data, &mut col_a)
            .unwrap();
        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&b_data, &mut col_b)
            .unwrap();

        let d_num_rows = device_row_count(&executor, 3);
        let buffer =
            CudaBuffer::from_columns(vec![col_a.into(), col_b.into()], 3, d_num_rows, schema);

        // Project: [b, a] (reverse order)
        let result =
            executor.execute_project(&buffer, &[ProjectExpr::Column(1), ProjectExpr::Column(0)]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);
        assert_eq!(result.arity(), 2);

        // First column should be b's values
        let col0 = read_buffer_u32(&executor, &result, 0);
        assert_eq!(col0, vec![10, 20, 30]);

        // Second column should be a's values
        let col1 = read_buffer_u32(&executor, &result, 1);
        assert_eq!(col1, vec![1, 2, 3]);
    }

    #[test]
    fn test_execute_computed_projection_wiring() {
        // Test that ProjectExpr::Computed is handled correctly
        // Even if arithmetic stubs return errors, verify the flow is correct
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create a 2-column buffer
        let schema = Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U32),
        ]);

        let a_data: Vec<u8> = [10u32, 20, 30]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let b_data: Vec<u8> = [1u32, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut col_a = executor
            .provider
            .memory()
            .alloc::<u8>(a_data.len())
            .unwrap();
        let mut col_b = executor
            .provider
            .memory()
            .alloc::<u8>(b_data.len())
            .unwrap();

        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&a_data, &mut col_a)
            .unwrap();
        executor
            .provider
            .device()
            .inner()
            .htod_sync_copy_into(&b_data, &mut col_b)
            .unwrap();

        let d_num_rows = device_row_count(&executor, 3);
        let buffer =
            CudaBuffer::from_columns(vec![col_a.into(), col_b.into()], 3, d_num_rows, schema);

        // Project with computed expression: a + b
        let add_expr = Expr::Add(Box::new(Expr::Column(0)), Box::new(Expr::Column(1)));
        let projections = vec![
            ProjectExpr::Column(0),                           // Pass through column a
            ProjectExpr::Computed(add_expr, ScalarType::U32), // Compute a + b
        ];

        let result = executor.execute_project(&buffer, &projections);

        // The wiring should be correct - result depends on whether CUDA arithmetic kernels are available
        // If available: result has 2 columns with computed values
        // If not available: may return error from provider stubs
        match result {
            Ok(res) => {
                // Wiring worked and arithmetic kernels are available
                assert_eq!(buffer_row_count(&executor, &res), 3);
                assert_eq!(res.arity(), 2);

                // First column should be a's values (pass-through)
                let col0 = read_buffer_u32(&executor, &res, 0);
                assert_eq!(col0, vec![10, 20, 30]);

                // Second column should be a + b = [11, 22, 33]
                let col1 = read_buffer_u32(&executor, &res, 1);
                assert_eq!(col1, vec![11, 22, 33]);
            }
            Err(e) => {
                // Arithmetic kernels not available - that's OK for this test
                // The important thing is that the wiring reached the provider
                let err_msg = format!("{}", e);
                assert!(
                    err_msg.contains("not implemented")
                        || err_msg.contains("not yet implemented")
                        || err_msg.contains("not supported")
                        || err_msg.contains("stub")
                        || err_msg.contains("Unsupported")
                        || err_msg.contains("arithmetic kernels"),
                    "Unexpected error: {}. Expected arithmetic kernel stub error.",
                    err_msg
                );
            }
        }
    }

    // ============== Union Node Tests ==============

    #[test]
    fn test_execute_union_empty_inputs() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let result = executor.execute_union(&[]);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
    }

    #[test]
    fn test_execute_union_single_input() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2, 3], "key");

        let result = executor.execute_union(&[buffer]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_execute_union_multiple_inputs() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer1 = create_test_buffer(&executor, &[1, 2], "key");
        let buffer2 = create_test_buffer(&executor, &[3, 4], "key");
        let buffer3 = create_test_buffer(&executor, &[5], "key");

        let result = executor.execute_union(&[buffer1, buffer2, buffer3]);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 5);
    }

    // ============== Distinct Node Tests ==============

    #[test]
    fn test_execute_distinct_empty() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = executor.create_empty_buffer(schema).unwrap();

        let result = executor.execute_distinct(&empty, &[0]);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
    }

    // ============== Diff Node Tests ==============

    #[test]
    fn test_execute_diff() {
        let executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let left = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        let right = create_test_buffer(&executor, &[2, 4], "key");

        let result = executor.execute_diff(&left, &right);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![1, 3, 5]);
    }

    // ============== Fixpoint Tests ==============

    #[test]
    fn test_execute_fixpoint_base_only() {
        // Test fixpoint with a base case that reaches fixpoint immediately
        // (recursive step produces nothing new)
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create base relation
        let buffer = create_test_buffer(&executor, &[1, 2, 3], "key");
        executor.store_mut().put("base_rel", buffer);
        executor.register_relation(RelId(1), "base_rel");

        // Create an empty recursive relation (simulating a recursive step that produces nothing)
        let empty_schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty_buffer = executor.create_empty_buffer(empty_schema).unwrap();
        executor.store_mut().put("empty_rel", empty_buffer);
        executor.register_relation(RelId(4), "empty_rel");

        // Base: scan base_rel
        // Recursive: scan empty_rel (produces nothing new)
        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        // Should return base case since recursive produces nothing
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);
        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_execute_fixpoint_empty_base() {
        // Test fixpoint with empty base case
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create empty base relation
        let empty_schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty_buffer = executor.create_empty_buffer(empty_schema.clone()).unwrap();
        executor.store_mut().put("empty_base", empty_buffer);
        executor.register_relation(RelId(1), "empty_base");

        // Create recursive relation (won't be used since base is empty)
        let rec_buffer = create_test_buffer(&executor, &[4, 5, 6], "key");
        executor.store_mut().put("rec_rel", rec_buffer);
        executor.register_relation(RelId(4), "rec_rel");

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        // Should return empty since base is empty
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
    }

    #[test]
    fn test_execute_fixpoint_one_iteration() {
        // Test fixpoint that converges after one iteration
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Base: [1, 2]
        let base_buffer = create_test_buffer(&executor, &[1, 2], "key");
        executor.store_mut().put("base_rel", base_buffer);
        executor.register_relation(RelId(1), "base_rel");

        // Recursive produces [1, 2, 3] - after diff with R, only [3] remains
        let rec_buffer = create_test_buffer(&executor, &[1, 2, 3], "key");
        executor.store_mut().put("rec_rel", rec_buffer);
        executor.register_relation(RelId(4), "rec_rel");

        // After first iteration, R = [1, 2, 3], recursive produces [1, 2, 3] again
        // diff([1, 2, 3], [1, 2, 3]) = empty -> fixpoint reached

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        let result = result.unwrap();
        // Result should be [1, 2, 3]
        assert_eq!(buffer_row_count(&executor, &result), 3);
    }

    #[test]
    fn test_execute_fixpoint_multiple_iterations() {
        // Test fixpoint that requires multiple iterations to converge
        // This simulates transitive closure behavior
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Base: [1]
        let base_buffer = create_test_buffer(&executor, &[1], "key");
        executor.store_mut().put("base_rel", base_buffer);
        executor.register_relation(RelId(1), "base_rel");

        // For this test, we need a recursive rule that can expand
        // Since we can't easily simulate join-based recursion without complex setup,
        // we'll test a simpler case where recursive produces cumulative data

        // Recursive relation will produce [1, 2] in first iteration,
        // then [1, 2, 3] in second iteration, etc.
        // This requires a more complex setup, so let's test the basic convergence

        // Simplified test: recursive produces union of base with [2]
        // First iteration: R=[1], rec produces [1, 2] -> delta_new = [2]
        // Second iteration: R=[1, 2], rec produces [1, 2] -> delta_new = empty
        let rec_buffer = create_test_buffer(&executor, &[1, 2], "key");
        executor.store_mut().put("rec_rel", rec_buffer);
        executor.register_relation(RelId(4), "rec_rel");

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        let result = result.unwrap();
        // Result should be union of [1] and [2] = [1, 2]
        assert_eq!(buffer_row_count(&executor, &result), 2);
    }

    #[test]
    fn test_execute_fixpoint_via_node() {
        // Test fixpoint through execute_node to ensure the match arm works
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create and store a base buffer
        let buffer = create_test_buffer(&executor, &[1, 2, 3], "key");
        executor.store_mut().put("base_rel", buffer);
        executor.register_relation(RelId(1), "base_rel");

        // Empty recursive means immediate fixpoint
        let empty_schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty_buffer = executor.create_empty_buffer(empty_schema).unwrap();
        executor.store_mut().put("empty_rel", empty_buffer);
        executor.register_relation(RelId(4), "empty_rel");

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);
    }

    #[test]
    fn test_fixpoint_cleanup() {
        // Test that fixpoint properly cleans up delta and full relations
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&executor, &[1, 2], "key");
        executor.store_mut().put("base_rel", buffer);
        executor.register_relation(RelId(1), "base_rel");

        let empty_schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty_buffer = executor.create_empty_buffer(empty_schema).unwrap();
        executor.store_mut().put("empty_rel", empty_buffer);
        executor.register_relation(RelId(4), "empty_rel");

        // Register names for delta and full relations to check cleanup
        executor.register_relation(RelId(2), "__delta_test");
        executor.register_relation(RelId(3), "__full_test");

        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(4) });

        let node = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(2),
            full_rel: RelId(3),
        };

        let result = executor.execute_node(&node);
        assert!(result.is_ok());

        // After fixpoint, the delta and full relations should be cleaned up
        assert!(!executor.store().contains("__delta_test"));
        assert!(!executor.store().contains("__full_test"));
    }

    // ============== Execute Plan Tests ==============

    #[test]
    fn test_execute_plan_empty() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let plan = ExecutionPlan::new(vec![]);

        let result = executor.execute_plan(&plan);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 0);
    }

    #[test]
    fn test_execute_plan_with_stratum() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create input relation
        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        executor.store_mut().put("input", buffer);
        executor.register_relation(RelId(1), "input");

        // Build a simple plan
        let scc = Scc {
            id: 0,
            predicates: vec!["output".to_string()],
            is_recursive: false,
        };

        let rule = CompiledRule {
            head: "output".to_string(),
            body: RirNode::Scan { rel: RelId(1) },
            meta: RirMeta::default(),
        };

        let stratum = Stratum {
            id: 0,
            sccs: vec![0],
        };

        let plan = ExecutionPlan {
            sccs: vec![scc],
            strata: vec![stratum],
            rules_by_scc: vec![vec![rule]],
            est_memory_peak: 0,
        };

        let result = executor.execute_plan(&plan);
        assert!(result.is_ok());

        // Verify output relation was created
        assert!(executor.store().contains("output"));
        let output = executor.store().get("output").unwrap();
        assert_eq!(buffer_row_count(&executor, output), 5);
    }

    #[test]
    fn test_apply_deltas_and_recompute_updates_dependents() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let input = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        executor.store_mut().put("input", input);
        executor.register_relation(RelId(1), "input");

        // SCC0: identity rule for input (mirrors how compiled facts appear as scan rules).
        // SCC1: output depends on input.
        let scc0 = Scc {
            id: 0,
            predicates: vec!["input".to_string()],
            is_recursive: false,
        };
        let scc1 = Scc {
            id: 1,
            predicates: vec!["output".to_string()],
            is_recursive: false,
        };

        let input_rule = CompiledRule {
            head: "input".to_string(),
            body: RirNode::Scan { rel: RelId(1) },
            meta: RirMeta::default(),
        };

        let output_rule = CompiledRule {
            head: "output".to_string(),
            body: RirNode::Filter {
                input: Box::new(RirNode::Scan { rel: RelId(1) }),
                predicate: Expr::Compare {
                    left: Box::new(Expr::Column(0)),
                    op: CompareOp::Gt,
                    right: Box::new(Expr::Const(ConstValue::U32(2))),
                },
            },
            meta: RirMeta::default(),
        };

        let stratum = Stratum {
            id: 0,
            sccs: vec![0, 1],
        };

        let plan = ExecutionPlan {
            sccs: vec![scc0, scc1],
            strata: vec![stratum],
            rules_by_scc: vec![vec![input_rule], vec![output_rule]],
            est_memory_peak: 0,
        };

        executor.execute_plan(&plan).expect("initial execute_plan");
        let initial_out = executor.store().get("output").expect("output missing");
        let initial_vals = read_buffer_u32(&executor, initial_out, 0);
        assert_eq!(initial_vals, vec![3, 4, 5]);

        let delete_buf = create_test_buffer(&executor, &[5], "key");
        let insert_buf = create_test_buffer(&executor, &[10], "key");

        let mut deltas = HashMap::new();
        deltas.insert(
            "input".to_string(),
            RelationDelta::new(Some(insert_buf), Some(delete_buf)),
        );

        executor
            .apply_deltas_and_recompute(&plan, &deltas)
            .expect("apply_deltas_and_recompute");

        let out = executor
            .store()
            .get("output")
            .expect("output missing after recompute");
        let vals = read_buffer_u32(&executor, out, 0);
        assert_eq!(vals, vec![3, 4, 10]);
    }

    #[test]
    fn test_apply_deltas_and_recompute_insert_only_recomputes_anti_join() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let lhs = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        executor.store_mut().put("lhs", lhs);
        executor.register_relation(RelId(1), "lhs");

        let blocked = create_test_buffer(&executor, &[], "key");
        executor.store_mut().put("blocked", blocked);
        executor.register_relation(RelId(2), "blocked");

        // SCC0: lhs identity rule
        // SCC1: blocked identity rule
        // SCC2: out = lhs \ blocked (anti-join)
        let scc0 = Scc {
            id: 0,
            predicates: vec!["lhs".to_string()],
            is_recursive: false,
        };
        let scc1 = Scc {
            id: 1,
            predicates: vec!["blocked".to_string()],
            is_recursive: false,
        };
        let scc2 = Scc {
            id: 2,
            predicates: vec!["out".to_string()],
            is_recursive: false,
        };

        let lhs_rule = CompiledRule {
            head: "lhs".to_string(),
            body: RirNode::Scan { rel: RelId(1) },
            meta: RirMeta::default(),
        };
        let blocked_rule = CompiledRule {
            head: "blocked".to_string(),
            body: RirNode::Scan { rel: RelId(2) },
            meta: RirMeta::default(),
        };
        let out_rule = CompiledRule {
            head: "out".to_string(),
            body: RirNode::Join {
                left: Box::new(RirNode::Scan { rel: RelId(1) }),
                right: Box::new(RirNode::Scan { rel: RelId(2) }),
                left_keys: vec![0],
                right_keys: vec![0],
                join_type: JoinType::Anti,
            },
            meta: RirMeta::default(),
        };

        let stratum = Stratum {
            id: 0,
            sccs: vec![0, 1, 2],
        };

        let plan = ExecutionPlan {
            sccs: vec![scc0, scc1, scc2],
            strata: vec![stratum],
            rules_by_scc: vec![vec![lhs_rule], vec![blocked_rule], vec![out_rule]],
            est_memory_peak: 0,
        };

        executor.execute_plan(&plan).expect("initial execute_plan");
        let initial = executor.store().get("out").expect("out missing");
        let initial_vals = read_buffer_u32(&executor, initial, 0);
        assert_eq!(initial_vals, vec![1, 2, 3, 4, 5]);

        // Insert into the "blocked" relation: output should shrink.
        let insert_buf = create_test_buffer(&executor, &[2, 4], "key");
        let mut deltas = HashMap::new();
        deltas.insert(
            "blocked".to_string(),
            RelationDelta::new(Some(insert_buf), None),
        );

        executor
            .apply_deltas_and_recompute(&plan, &deltas)
            .expect("apply_deltas_and_recompute");

        let out = executor
            .store()
            .get("out")
            .expect("out missing after update");
        let vals = read_buffer_u32(&executor, out, 0);
        assert_eq!(vals, vec![1, 3, 5]);
    }

    // ============== RIR Node Composition Tests ==============

    #[test]
    fn test_execute_filter_project_chain() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create input relation
        let buffer = create_test_buffer(&executor, &[1, 2, 3, 4, 5], "key");
        executor.store_mut().put("input", buffer);
        executor.register_relation(RelId(1), "input");

        // Build: Project(Filter(Scan))
        let scan = RirNode::Scan { rel: RelId(1) };
        let filter = RirNode::Filter {
            input: Box::new(scan),
            predicate: Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Gt,
                right: Box::new(Expr::Const(ConstValue::U32(2))),
            },
        };
        let project = RirNode::Project {
            input: Box::new(filter),
            columns: vec![ProjectExpr::Column(0)],
        };

        let result = executor.execute_node(&project);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(buffer_row_count(&executor, &result), 3);

        let values = read_buffer_u32(&executor, &result, 0);
        assert_eq!(values, vec![3, 4, 5]);
    }

    // ============== MC Relation Reset Tests ==============

    #[test]
    fn test_reset_for_mc_relations_preserves_static_and_clears_dynamic() {
        let mut executor = match create_test_executor() {
            Some(e) => e,
            None => {
                eprintln!("Skipping: no CUDA device");
                return;
            }
        };

        executor.register_relation(RelId(1), "base_rel");
        executor.register_relation(RelId(2), "dyn_rel");

        let schema = Schema::new(vec![("x".to_string(), ScalarType::U32)]);
        let base = create_test_buffer(&executor, &[1u32], "x");
        let dyn_buf = create_test_buffer(&executor, &[9u32], "x");
        executor.put_relation("base_rel", base);
        executor.put_relation("dyn_rel", dyn_buf);

        executor
            .reset_for_mc_relations(&["base_rel"], &[("dyn_rel", schema.clone())])
            .unwrap();

        assert_eq!(
            buffer_row_count(&executor, executor.store().get("base_rel").unwrap()),
            1
        );
        assert_eq!(
            buffer_row_count(&executor, executor.store().get("dyn_rel").unwrap()),
            0
        );
    }
}
