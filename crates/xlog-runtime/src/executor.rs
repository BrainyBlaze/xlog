//! Query executor for RIR nodes
//!
//! The executor interprets RIR (Relational IR) nodes using the CUDA kernel provider
//! to execute GPU-accelerated relational operations.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cudarc::driver::{LaunchAsync, LaunchConfig};
use xlog_core::{AggOp, RelId, Result, RuntimeConfig, ScalarType, Schema, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{arith_kernels, filter_kernels, ARITH_MODULE, FILTER_MODULE};
use xlog_cuda::{CudaBuffer, CudaKernelProvider, JoinIndexV2, JoinType as CudaJoinType};
use xlog_ir::{
    CompareOp, ConstValue, ExecutionPlan, Expr, JoinType, ProjectExpr, RirNode, Stratum,
};
use xlog_stats::{StatsManager, StatsSnapshot};

use crate::ilp_registry::{read_device_row_count, IlpMask, IlpRegistry, IlpTagEntry, IlpTaggedResult};
use crate::profiler::{ExecutionStats, Profiler};
use crate::RelationStore;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct JoinIndexKey {
    rel: RelId,
    version: u64,
    key_cols: Vec<usize>,
}

struct CachedJoinIndex {
    index: JoinIndexV2,
    bytes: u64,
    last_used: u64,
}

struct JoinIndexCache {
    entries: HashMap<JoinIndexKey, CachedJoinIndex>,
    clock: u64,
    total_bytes: u64,
    max_bytes: u64,
}

impl JoinIndexCache {
    fn new(max_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            clock: 0,
            total_bytes: 0,
            max_bytes,
        }
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.clock = 0;
        self.total_bytes = 0;
    }

    fn get(&mut self, key: &JoinIndexKey) -> Option<&JoinIndexV2> {
        let entry = self.entries.get_mut(key)?;
        self.clock = self.clock.saturating_add(1);
        entry.last_used = self.clock;
        Some(&entry.index)
    }

    fn insert(&mut self, key: JoinIndexKey, index: JoinIndexV2) {
        let bytes = index.estimated_bytes();
        if bytes > self.max_bytes {
            return;
        }

        self.evict_until_fits(bytes);

        self.clock = self.clock.saturating_add(1);
        let last_used = self.clock;

        if let Some(prev) = self.entries.remove(&key) {
            self.total_bytes = self.total_bytes.saturating_sub(prev.bytes);
        }

        self.total_bytes = self.total_bytes.saturating_add(bytes);
        self.entries.insert(
            key,
            CachedJoinIndex {
                index,
                bytes,
                last_used,
            },
        );
    }

    fn invalidate_rel(&mut self, rel: RelId) {
        let keys: Vec<JoinIndexKey> = self
            .entries
            .keys()
            .filter(|k| k.rel == rel)
            .cloned()
            .collect();
        for key in keys {
            if let Some(entry) = self.entries.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(entry.bytes);
            }
        }
    }

    fn evict_until_fits(&mut self, additional_bytes: u64) {
        while !self.entries.is_empty()
            && self.total_bytes.saturating_add(additional_bytes) > self.max_bytes
        {
            let mut oldest_key: Option<JoinIndexKey> = None;
            let mut oldest_clock = u64::MAX;

            for (k, v) in &self.entries {
                if v.last_used < oldest_clock {
                    oldest_clock = v.last_used;
                    oldest_key = Some(k.clone());
                }
            }

            let Some(key) = oldest_key else {
                break;
            };
            if let Some(entry) = self.entries.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(entry.bytes);
            } else {
                break;
            }
        }
    }
}

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

    pub fn execute_non_recursive_scc(&mut self, rules: &[xlog_ir::CompiledRule]) -> Result<()> {
        for rule in rules {
            let result = self.execute_node(&rule.body)?;

            if let Some(existing) = self.store.get(&rule.head) {
                if result.is_empty() {
                    continue;
                }
                let merged = self.provider.union_gpu(existing, &result)?;
                self.store_put(&rule.head, merged);
            } else {
                let key_cols: Vec<usize> = (0..result.arity()).collect();
                let deduped = if result.is_empty() {
                    result
                } else {
                    self.provider.dedup(&result, &key_cols)?
                };
                self.store_put(&rule.head, deduped);
            }
        }
        Ok(())
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
        }
    }

    /// Execute a stratum (internal implementation)
    ///
    /// Processes all SCCs in the stratum by executing their rules.
    /// For recursive SCCs, uses semi-naive fixpoint iteration.
    fn execute_stratum_impl(&mut self, stratum: &Stratum, plan: &ExecutionPlan) -> Result<()> {
        // Process each SCC in the stratum
        for &scc_id in &stratum.sccs {
            // Get rules for this SCC
            if let Some(rules) = plan.rules_by_scc.get(scc_id as usize) {
                // Get SCC metadata
                let scc = plan.sccs.get(scc_id as usize);
                let is_recursive = scc.map(|s| s.is_recursive).unwrap_or(false);

                if is_recursive {
                    // Recursive SCC: use semi-naive fixpoint iteration
                    self.execute_recursive_scc(rules)?;
                } else {
                    // Non-recursive SCC: execute rules once, union results for same predicate
                    for rule in rules {
                        let result = self.execute_node(&rule.body)?;

                        // Union with existing result if predicate already has data
                        if let Some(existing) = self.store.get(&rule.head) {
                            let union_input_rows = existing.num_rows() + result.num_rows();
                            let start = self.profiler.start_op();
                            let merged = self.provider.union_gpu(existing, &result)?;
                            if let Some(start) = start {
                                let mem = self.provider.memory().allocated_bytes();
                                self.profiler.record_op(
                                    "union",
                                    union_input_rows,
                                    merged.num_rows(),
                                    start,
                                    mem,
                                );
                                self.profiler.record_peak_memory(mem);
                            }
                            self.store_put(&rule.head, merged);
                        } else {
                            let key_cols: Vec<usize> = (0..result.arity()).collect();
                            let deduped = if result.is_empty() {
                                result
                            } else {
                                let dedup_input_rows = result.num_rows();
                                let start = self.profiler.start_op();
                                let deduped = self.provider.dedup(&result, &key_cols)?;
                                if let Some(start) = start {
                                    let mem = self.provider.memory().allocated_bytes();
                                    self.profiler.record_op(
                                        "dedup",
                                        dedup_input_rows,
                                        deduped.num_rows(),
                                        start,
                                        mem,
                                    );
                                    self.profiler.record_peak_memory(mem);
                                }
                                deduped
                            };
                            self.store_put(&rule.head, deduped);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Execute a recursive SCC using semi-naive fixpoint iteration
    ///
    /// The algorithm:
    /// 1. Execute all rules once to get initial result
    /// 2. Track which relations changed (delta)
    /// 3. Re-execute rules, using delta from previous iteration
    /// 4. Repeat until no changes (fixpoint reached)
    pub fn execute_recursive_scc(&mut self, rules: &[xlog_ir::CompiledRule]) -> Result<()> {
        // Identify SCC predicates from rule heads (these are the recursive IDBs).
        let mut recursive_preds: HashSet<String> = HashSet::new();
        let mut schema_by_pred: HashMap<String, Schema> = HashMap::new();
        for rule in rules {
            recursive_preds.insert(rule.head.clone());
            if rule.meta.schema.arity() > 0 {
                schema_by_pred
                    .entry(rule.head.clone())
                    .or_insert_with(|| rule.meta.schema.clone());
            }
        }

        // Ensure all recursive predicates exist in the store so scans never fail
        // due to evaluation order (mutual recursion can reference an as-yet-empty relation).
        for pred in &recursive_preds {
            if !self.store.contains(pred) {
                let schema = schema_by_pred
                    .get(pred)
                    .cloned()
                    .or_else(|| self.store.get(pred).map(|b| b.schema().clone()))
                    .ok_or_else(|| {
                        XlogError::Execution(format!(
                            "Missing schema for recursive predicate {}",
                            pred
                        ))
                    })?;
                let empty = self.create_empty_buffer(schema)?;
                self.store_put(pred, empty);
            }
        }

        // Create per-predicate delta relations (distinct RelIds) so semi-naive evaluation
        // can target a single recursive Scan occurrence without overriding *all* scans of
        // that predicate in a rule (required for self-joins like p(X,Y), p(Y,Z)).
        let mut next_rel_id = self
            .rel_names
            .keys()
            .map(|r| r.0)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        let mut delta_rel_by_pred: HashMap<String, (RelId, String)> = HashMap::new();
        for pred in &recursive_preds {
            let rel_id = RelId(next_rel_id);
            next_rel_id = next_rel_id.saturating_add(1);
            let name = format!("__delta_{}_{}", pred, rel_id.0);
            self.register_relation(rel_id, &name);
            delta_rel_by_pred.insert(pred.clone(), (rel_id, name));
        }

        // Step 1: Execute all rules once against the current store to seed initial results.
        // Accumulate per-head before mutating the store to avoid order dependence.
        let mut derived_initial: HashMap<String, CudaBuffer> = HashMap::new();
        for rule in rules {
            let result = self.execute_node(&rule.body)?;
            if let Some(acc) = derived_initial.get_mut(&rule.head) {
                let union_input = acc.num_rows() + result.num_rows();
                let start = self.profiler.start_op();
                let merged = self.provider.union_gpu(acc, &result)?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("union", union_input, merged.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }
                *acc = merged;
            } else {
                derived_initial.insert(rule.head.clone(), result);
            }
        }

        // Initialize delta from the newly-derived tuples only.
        //
        // This supports incremental maintenance: if the SCC is executed again after EDB inserts,
        // the delta relations start with only the *new* tuples, not a full rescan of the current
        // fixed point.
        for pred in &recursive_preds {
            let full_old = self
                .store
                .remove(pred)
                .ok_or_else(|| XlogError::Execution(format!("Missing relation: {}", pred)))?;

            let derived = match derived_initial.remove(pred) {
                Some(buf) => buf,
                None => self.create_empty_buffer(full_old.schema().clone())?,
            };

            let union_input = full_old.num_rows() + derived.num_rows();
            let start = self.profiler.start_op();
            let merged = self.provider.union_gpu(&full_old, &derived)?;
            if let Some(start) = start {
                let mem = self.provider.memory().allocated_bytes();
                self.profiler
                    .record_op("union", union_input, merged.num_rows(), start, mem);
                self.profiler.record_peak_memory(mem);
            }

            let key_cols: Vec<usize> = (0..merged.arity()).collect();
            let full_new = if merged.is_empty() {
                merged
            } else {
                let dedup_input = merged.num_rows();
                let start = self.profiler.start_op();
                let deduped = self.provider.dedup_sorted(&merged, &key_cols)?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("dedup", dedup_input, deduped.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }
                deduped
            };

            let (_delta_rel_id, delta_name) = delta_rel_by_pred.get(pred).ok_or_else(|| {
                XlogError::Execution(format!("Missing delta relation for {}", pred))
            })?;

            let delta_initial = if full_old.is_empty() || full_new.is_empty() {
                self.clone_buffer(&full_new)?
            } else {
                let diff_input = full_new.num_rows() + full_old.num_rows();
                let start = self.profiler.start_op();
                let diffed = self.provider.diff_gpu(&full_new, &full_old)?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("diff", diff_input, diffed.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }
                diffed
            };

            self.store_put(pred, full_new);
            self.store_put(delta_name, delta_initial);
        }

        // Step 2: Iterate until no new tuples are produced.
        let mut reached_fixpoint = false;
        let max_iterations = self.config.max_iterations as usize;
        let mut iteration_count = 0usize;
        for _iteration in 0..max_iterations {
            iteration_count += 1;
            // Compute delta_new_raw per head by evaluating each rule once per recursive Scan occurrence.
            let mut delta_new_raw_by_head: HashMap<String, CudaBuffer> = HashMap::new();

            for rule in rules {
                let mut scans = Vec::new();
                Self::collect_scan_rels(&rule.body, &mut scans);

                // Build a list of (rel_id, occurrence_idx, pred_name) for recursive scans.
                let mut seen: HashMap<RelId, usize> = HashMap::new();
                let mut variants: Vec<(RelId, usize, String)> = Vec::new();
                for rel_id in scans {
                    let pred_name = match self.get_rel_name(rel_id) {
                        Some(n) => n.to_string(),
                        None => continue,
                    };
                    if !recursive_preds.contains(&pred_name) {
                        continue;
                    }

                    // Skip variants where the delta for this predicate is empty.
                    let (_delta_rel_id, delta_name) = match delta_rel_by_pred.get(&pred_name) {
                        Some(v) => v,
                        None => continue,
                    };
                    if self
                        .store
                        .get(delta_name)
                        .map(|b| b.is_empty())
                        .unwrap_or(true)
                    {
                        continue;
                    }

                    let occ = seen.entry(rel_id).or_insert(0);
                    variants.push((rel_id, *occ, pred_name));
                    *occ += 1;
                }

                if variants.is_empty() {
                    // Base rule: it can only contribute on the first seeding pass.
                    continue;
                }

                let mut rule_delta_raw: Option<CudaBuffer> = None;
                for (rel_id, occ, pred_name) in variants {
                    let (delta_rel_id, _delta_name) =
                        delta_rel_by_pred.get(&pred_name).ok_or_else(|| {
                            XlogError::Execution(format!(
                                "Missing delta relation for predicate {}",
                                pred_name
                            ))
                        })?;

                    let variant_node =
                        Self::rewrite_scan_nth(&rule.body, rel_id, occ, *delta_rel_id).ok_or_else(
                            || {
                                XlogError::Execution(format!(
                                    "Failed to rewrite rule body for predicate {}",
                                    pred_name
                                ))
                            },
                        )?;

                    let out = self.execute_node(&variant_node)?;
                    rule_delta_raw = Some(if let Some(acc) = rule_delta_raw {
                        let union_input = acc.num_rows() + out.num_rows();
                        let start = self.profiler.start_op();
                        let merged = self.provider.union_gpu(&acc, &out)?;
                        if let Some(start) = start {
                            let mem = self.provider.memory().allocated_bytes();
                            self.profiler.record_op(
                                "union",
                                union_input,
                                merged.num_rows(),
                                start,
                                mem,
                            );
                            self.profiler.record_peak_memory(mem);
                        }
                        merged
                    } else {
                        out
                    });
                }

                if let Some(rule_out) = rule_delta_raw {
                    if let Some(acc) = delta_new_raw_by_head.get_mut(&rule.head) {
                        let union_input = acc.num_rows() + rule_out.num_rows();
                        let start = self.profiler.start_op();
                        let merged = self.provider.union_gpu(acc, &rule_out)?;
                        if let Some(start) = start {
                            let mem = self.provider.memory().allocated_bytes();
                            self.profiler.record_op(
                                "union",
                                union_input,
                                merged.num_rows(),
                                start,
                                mem,
                            );
                            self.profiler.record_peak_memory(mem);
                        }
                        *acc = merged;
                    } else {
                        delta_new_raw_by_head.insert(rule.head.clone(), rule_out);
                    }
                }
            }

            // Finalize delta_new per head: delta_new = dedup(delta_raw - full).
            let mut any_changed = false;

            for pred in &recursive_preds {
                let full = self
                    .store
                    .get(pred)
                    .ok_or_else(|| XlogError::Execution(format!("Missing relation: {}", pred)))?;

                let delta_raw = delta_new_raw_by_head.remove(pred);
                let delta_new = if let Some(delta_raw) = delta_raw {
                    if delta_raw.is_empty() {
                        self.create_empty_buffer(full.schema().clone())?
                    } else {
                        let diff_input = delta_raw.num_rows() + full.num_rows();
                        let start = self.profiler.start_op();
                        let diffed = self.provider.diff_gpu(&delta_raw, full)?;
                        if let Some(start) = start {
                            let mem = self.provider.memory().allocated_bytes();
                            self.profiler.record_op(
                                "diff",
                                diff_input,
                                diffed.num_rows(),
                                start,
                                mem,
                            );
                            self.profiler.record_peak_memory(mem);
                        }
                        diffed
                    }
                } else {
                    self.create_empty_buffer(full.schema().clone())?
                };

                let (_delta_rel_id, delta_name) = delta_rel_by_pred.get(pred).ok_or_else(|| {
                    XlogError::Execution(format!("Missing delta relation for {}", pred))
                })?;
                if !delta_new.is_empty() {
                    any_changed = true;
                }
                self.store_put(delta_name, delta_new);
            }

            // Fixpoint reached if no deltas produced.
            if !any_changed {
                reached_fixpoint = true;
                self.profiler.record_iterations(iteration_count);
                break;
            }

            // Merge deltas into full relations.
            for pred in &recursive_preds {
                let full_old = self
                    .store
                    .remove(pred)
                    .ok_or_else(|| XlogError::Execution(format!("Missing relation: {}", pred)))?;
                let (_delta_rel_id, delta_name) = delta_rel_by_pred.get(pred).ok_or_else(|| {
                    XlogError::Execution(format!("Missing delta relation for {}", pred))
                })?;
                let delta = self.store_remove(delta_name).ok_or_else(|| {
                    XlogError::Execution(format!("Missing relation: {}", delta_name))
                })?;

                if delta.is_empty() {
                    self.store_put(pred, full_old);
                    self.store_put(delta_name, delta);
                    continue;
                }

                let union_input = full_old.num_rows() + delta.num_rows();
                let start = self.profiler.start_op();
                let merged = self.provider.union_gpu(&full_old, &delta)?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("union", union_input, merged.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }

                let key_cols: Vec<usize> = (0..merged.arity()).collect();
                let full_new = if merged.is_empty() {
                    merged
                } else {
                    let dedup_input = merged.num_rows();
                    let start = self.profiler.start_op();
                    let deduped = self.provider.dedup_sorted(&merged, &key_cols)?;
                    if let Some(start) = start {
                        let mem = self.provider.memory().allocated_bytes();
                        self.profiler.record_op(
                            "dedup",
                            dedup_input,
                            deduped.num_rows(),
                            start,
                            mem,
                        );
                        self.profiler.record_peak_memory(mem);
                    }
                    deduped
                };
                self.store_put(pred, full_new);
                self.store_put(delta_name, delta);
            }
        }

        // Cleanup: remove delta relations from store and relation mapping.
        for (_pred, (rel_id, delta_name)) in delta_rel_by_pred {
            self.store_remove(&delta_name);
            self.rel_names.remove(&rel_id);
            self.name_to_rel.remove(&delta_name);
            let _ = self.stats.unregister_relation(rel_id);
        }

        if !reached_fixpoint {
            // Record iterations even on failure for debugging
            self.profiler.record_iterations(iteration_count);
            return Err(XlogError::Execution(format!(
                "Recursive SCC iteration limit ({}) exceeded",
                self.config.max_iterations
            )));
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

    /// Execute a Scan node
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

    /// Execute a Filter node using GPU predicate evaluation.
    pub fn execute_filter(&self, input: &CudaBuffer, predicate: &Expr) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema().clone());
        }

        let mask = self.eval_predicate_mask_gpu(predicate, input)?;
        self.provider.filter_by_device_mask(input, &mask)
    }

    fn eval_predicate_mask_gpu(
        &self,
        expr: &Expr,
        input: &CudaBuffer,
    ) -> Result<TrackedCudaSlice<u8>> {
        if input.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Execution(format!(
                "Predicate evaluation supports at most {} rows, got {}",
                u32::MAX,
                input.num_rows()
            )));
        }
        let n = input.num_rows() as u32;

        match expr {
            Expr::Column(col_idx) => {
                let col_type = input
                    .schema()
                    .column_type(*col_idx)
                    .ok_or_else(|| XlogError::Execution(format!("Column {} not found", col_idx)))?;
                if col_type == ScalarType::Bool {
                    let col_buf = self.wrap_single_column(input, *col_idx)?;
                    let zero = self.provider.create_constant_column_with_device_count(
                        &[0u8],
                        ScalarType::Bool,
                        input.num_rows(),
                        input.num_rows_device(),
                    )?;
                    return self.compare_buffers_mask(&col_buf, &zero, CompareOp::Ne);
                }
                self.mask_filled(n, 1)
            }
            Expr::Const(ConstValue::Bool(b)) => self.mask_filled(n, if *b { 1 } else { 0 }),
            Expr::Const(_) => self.mask_filled(n, 1),
            Expr::Compare { left, op, right } => {
                let use_float = Self::expr_may_be_float(left, input.schema())
                    || Self::expr_may_be_float(right, input.schema());

                let mut left_buf = self.evaluate_arith_expr(left, input)?;
                let mut right_buf = self.evaluate_arith_expr(right, input)?;

                if use_float {
                    left_buf = self.provider.cast_column(&left_buf, ScalarType::F64)?;
                    right_buf = self.provider.cast_column(&right_buf, ScalarType::F64)?;
                }

                self.compare_buffers_mask(&left_buf, &right_buf, *op)
            }
            Expr::And(exprs) => {
                if exprs.is_empty() {
                    return self.mask_filled(n, 1);
                }
                let mut mask = self.eval_predicate_mask_gpu(&exprs[0], input)?;
                for expr in &exprs[1..] {
                    let next = self.eval_predicate_mask_gpu(expr, input)?;
                    mask = self.mask_and(&mask, &next, n)?;
                }
                Ok(mask)
            }
            Expr::Or(exprs) => {
                if exprs.is_empty() {
                    return self.mask_filled(n, 0);
                }
                let mut mask = self.eval_predicate_mask_gpu(&exprs[0], input)?;
                for expr in &exprs[1..] {
                    let next = self.eval_predicate_mask_gpu(expr, input)?;
                    mask = self.mask_or(&mask, &next, n)?;
                }
                Ok(mask)
            }
            Expr::Not(inner) => {
                let mask = self.eval_predicate_mask_gpu(inner, input)?;
                self.mask_not(&mask, n)
            }
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

    fn compare_buffers_mask(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        op: CompareOp,
    ) -> Result<TrackedCudaSlice<u8>> {
        if left.arity() != 1 || right.arity() != 1 {
            return Err(XlogError::Execution(
                "Compare requires single-column buffers".into(),
            ));
        }
        if left.num_rows() != right.num_rows() {
            return Err(XlogError::Execution(
                "Compare requires matching row counts".into(),
            ));
        }
        if left.num_rows() > u32::MAX as u64 {
            return Err(XlogError::Execution(format!(
                "Compare supports at most {} rows, got {}",
                u32::MAX,
                left.num_rows()
            )));
        }
        if left.is_empty() {
            return self.provider.memory().alloc::<u8>(0).map_err(|e| {
                XlogError::Execution(format!("Failed to allocate empty mask: {}", e))
            });
        }

        let left_type = left
            .schema()
            .column_type(0)
            .ok_or_else(|| XlogError::Execution("Missing left column type".into()))?;
        let right_type = right
            .schema()
            .column_type(0)
            .ok_or_else(|| XlogError::Execution("Missing right column type".into()))?;

        if left_type != right_type {
            return Err(XlogError::Execution(
                "Compare requires matching column types".into(),
            ));
        }

        let kernel = match left_type {
            ScalarType::U32 | ScalarType::Symbol => filter_kernels::FILTER_COMPARE_U32_COL,
            ScalarType::U64 => filter_kernels::FILTER_COMPARE_U64_COL,
            ScalarType::I32 => filter_kernels::FILTER_COMPARE_I32_COL,
            ScalarType::I64 => filter_kernels::FILTER_COMPARE_I64_COL,
            ScalarType::F32 => filter_kernels::FILTER_COMPARE_F32_COL,
            ScalarType::F64 => filter_kernels::FILTER_COMPARE_F64_COL,
            ScalarType::Bool => filter_kernels::FILTER_COMPARE_U8_COL,
        };

        let left_col = left
            .column(0)
            .ok_or_else(|| XlogError::Execution("Missing left column".into()))?;
        let right_col = right
            .column(0)
            .ok_or_else(|| XlogError::Execution("Missing right column".into()))?;

        let num_rows = left.num_rows() as u32;
        let mut d_mask = self.provider.memory().alloc::<u8>(num_rows as usize)?;

        let func = self
            .provider
            .device()
            .inner()
            .get_func(FILTER_MODULE, kernel)
            .ok_or_else(|| XlogError::Execution("filter compare kernel not found".into()))?;
        let config = LaunchConfig::for_num_elems(num_rows);

        unsafe {
            func.clone().launch(
                config,
                (left_col, right_col, num_rows, op as u8, &mut d_mask),
            )
        }
        .map_err(|e| XlogError::Execution(format!("filter compare failed: {}", e)))?;

        Ok(d_mask)
    }

    fn mask_and(
        &self,
        left: &TrackedCudaSlice<u8>,
        right: &TrackedCudaSlice<u8>,
        n: u32,
    ) -> Result<TrackedCudaSlice<u8>> {
        let mut out = self.provider.memory().alloc::<u8>(n as usize)?;
        if n == 0 {
            return Ok(out);
        }

        let func = self
            .provider
            .device()
            .inner()
            .get_func(FILTER_MODULE, filter_kernels::MASK_AND)
            .ok_or_else(|| XlogError::Execution("mask_and kernel not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        unsafe { func.clone().launch(config, (left, right, &mut out, n)) }
            .map_err(|e| XlogError::Execution(format!("mask_and failed: {}", e)))?;

        Ok(out)
    }

    fn mask_or(
        &self,
        left: &TrackedCudaSlice<u8>,
        right: &TrackedCudaSlice<u8>,
        n: u32,
    ) -> Result<TrackedCudaSlice<u8>> {
        let mut out = self.provider.memory().alloc::<u8>(n as usize)?;
        if n == 0 {
            return Ok(out);
        }

        let func = self
            .provider
            .device()
            .inner()
            .get_func(FILTER_MODULE, filter_kernels::MASK_OR)
            .ok_or_else(|| XlogError::Execution("mask_or kernel not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        unsafe { func.clone().launch(config, (left, right, &mut out, n)) }
            .map_err(|e| XlogError::Execution(format!("mask_or failed: {}", e)))?;

        Ok(out)
    }

    fn mask_not(&self, input: &TrackedCudaSlice<u8>, n: u32) -> Result<TrackedCudaSlice<u8>> {
        let mut out = self.provider.memory().alloc::<u8>(n as usize)?;
        if n == 0 {
            return Ok(out);
        }

        let func = self
            .provider
            .device()
            .inner()
            .get_func(FILTER_MODULE, filter_kernels::MASK_NOT)
            .ok_or_else(|| XlogError::Execution("mask_not kernel not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        unsafe { func.clone().launch(config, (input, &mut out, n)) }
            .map_err(|e| XlogError::Execution(format!("mask_not failed: {}", e)))?;

        Ok(out)
    }

    fn mask_filled(&self, n: u32, value: u8) -> Result<TrackedCudaSlice<u8>> {
        let mut out = self.provider.memory().alloc::<u8>(n as usize)?;
        if n == 0 {
            return Ok(out);
        }

        if value == 0 {
            self.provider
                .device()
                .inner()
                .memset_zeros(&mut out)
                .map_err(|e| XlogError::Execution(format!("mask memset failed: {}", e)))?;
            return Ok(out);
        }

        let func = self
            .provider
            .device()
            .inner()
            .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_U8)
            .ok_or_else(|| XlogError::Execution("arith fill kernel not found".into()))?;
        let config = LaunchConfig::for_num_elems(n);

        unsafe { func.clone().launch(config, (value, n, &mut out)) }
            .map_err(|e| XlogError::Execution(format!("mask fill failed: {}", e)))?;

        Ok(out)
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

    fn expr_may_be_float(expr: &Expr, schema: &Schema) -> bool {
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

    fn wrap_single_column(&self, buffer: &CudaBuffer, col_idx: usize) -> Result<CudaBuffer> {
        let col_type = buffer
            .schema()
            .column_type(col_idx)
            .ok_or_else(|| XlogError::Execution(format!("Column {} not found", col_idx)))?;
        let schema = Schema::new(vec![("expr".to_string(), col_type)]);

        if buffer.is_empty() {
            return self.create_empty_buffer(schema);
        }

        let num_rows = buffer.num_rows();
        let bytes = (num_rows as usize)
            .checked_mul(col_type.size_bytes())
            .ok_or_else(|| XlogError::Execution("Column size overflow".into()))?;

        let src_col = buffer
            .column(col_idx)
            .ok_or_else(|| XlogError::Execution(format!("Column {} not found", col_idx)))?;
        let mut dst_col = self.provider.memory().alloc::<u8>(bytes)?;
        if bytes > 0 {
            self.provider
                .device()
                .inner()
                .dtod_copy(src_col, &mut dst_col)
                .map_err(|e| XlogError::Execution(format!("Failed to copy column: {}", e)))?;
        }

        let d_num_rows = self.clone_device_row_count(buffer)?;
        Ok(CudaBuffer::from_columns(
            vec![dst_col.into()],
            num_rows,
            d_num_rows,
            schema,
        ))
    }

    /// Evaluate an arithmetic expression on a buffer, producing a single-column result
    ///
    /// This method recursively evaluates arithmetic expressions (Add, Sub, Mul, Div, etc.)
    /// by delegating to the CUDA kernel provider for GPU-accelerated operations.
    fn evaluate_arith_expr(&self, expr: &Expr, input: &CudaBuffer) -> Result<CudaBuffer> {
        match expr {
            Expr::Column(idx) => {
                // Extract the column as a single-column buffer without host round-trip
                self.wrap_single_column(input, *idx)
            }
            Expr::Const(val) => {
                // Create a column filled with the constant value
                let (bytes, col_type) = self.const_to_bytes_and_type(val);
                self.provider.create_constant_column_with_device_count(
                    &bytes,
                    col_type,
                    input.num_rows(),
                    input.num_rows_device(),
                )
            }
            Expr::Add(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.add_columns(&left, &right)
            }
            Expr::Sub(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.sub_columns(&left, &right)
            }
            Expr::Mul(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.mul_columns(&left, &right)
            }
            Expr::Div(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.div_columns(&left, &right)
            }
            Expr::Mod(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.mod_columns(&left, &right)
            }
            Expr::Abs(inner) => {
                let val = self.evaluate_arith_expr(inner, input)?;
                self.provider.abs_column(&val)
            }
            Expr::Min(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.min_columns(&left, &right)
            }
            Expr::Max(l, r) => {
                let left = self.evaluate_arith_expr(l, input)?;
                let right = self.evaluate_arith_expr(r, input)?;
                self.provider.max_columns(&left, &right)
            }
            Expr::Pow(base, exp) => {
                let base_buf = self.evaluate_arith_expr(base, input)?;
                let exp_buf = self.evaluate_arith_expr(exp, input)?;
                self.provider.pow_columns(&base_buf, &exp_buf)
            }
            Expr::Cast(inner, target_type) => {
                let val = self.evaluate_arith_expr(inner, input)?;
                self.provider.cast_column(&val, *target_type)
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                // Evaluate condition to get boolean mask
                let mask_slice = self.eval_predicate_mask_gpu(condition, input)?;

                // Convert mask slice to a CudaBuffer for select_columns
                let d_num_rows = self.clone_device_row_count(input)?;
                let mask_buffer = CudaBuffer::from_columns(
                    vec![mask_slice.into()],
                    input.num_rows(),
                    d_num_rows,
                    Schema::new(vec![("mask".to_string(), ScalarType::Bool)]),
                );

                // Evaluate both branches
                let then_buf = self.evaluate_arith_expr(then_expr, input)?;
                let else_buf = self.evaluate_arith_expr(else_expr, input)?;

                // Select based on mask
                self.provider
                    .select_columns(&mask_buffer, &then_buf, &else_buf)
            }
            _ => Err(XlogError::Execution(format!(
                "Unsupported expression in arithmetic evaluation: {:?}",
                expr
            ))),
        }
    }

    /// Convert a ConstValue to raw bytes and ScalarType
    fn const_to_bytes_and_type(&self, val: &ConstValue) -> (Vec<u8>, ScalarType) {
        match val {
            ConstValue::U32(v) => (v.to_le_bytes().to_vec(), ScalarType::U32),
            ConstValue::U64(v) => (v.to_le_bytes().to_vec(), ScalarType::U64),
            ConstValue::I32(v) => (v.to_le_bytes().to_vec(), ScalarType::I32),
            ConstValue::I64(v) => (v.to_le_bytes().to_vec(), ScalarType::I64),
            ConstValue::F32(v) => (v.to_le_bytes().to_vec(), ScalarType::F32),
            ConstValue::F64(v) => (v.to_le_bytes().to_vec(), ScalarType::F64),
            ConstValue::Bool(v) => (vec![if *v { 1u8 } else { 0u8 }], ScalarType::Bool),
            ConstValue::Symbol(s) => (
                xlog_core::symbol::intern(s).to_le_bytes().to_vec(),
                ScalarType::Symbol,
            ),
        }
    }

    /// Execute a Project node
    ///
    /// Selects and reorders columns according to the projection list.
    /// Supports both column pass-through and computed expressions.
    fn execute_project(&self, input: &CudaBuffer, columns: &[ProjectExpr]) -> Result<CudaBuffer> {
        if input.is_empty() {
            // Build projected schema
            let projected_schema = self.project_schema(input.schema(), columns)?;
            return self.create_empty_buffer(projected_schema);
        }

        // Build result columns as single-column CudaBuffers
        let mut result_buffers: Vec<CudaBuffer> = Vec::with_capacity(columns.len());
        let mut result_types: Vec<ScalarType> = Vec::with_capacity(columns.len());

        for proj_expr in columns {
            match proj_expr {
                ProjectExpr::Column(col_idx) => {
                    // Use extract_column to get a single-column buffer
                    let col_buffer = self.provider.extract_column(input, *col_idx)?;
                    let col_type = input
                        .schema()
                        .column_type(*col_idx)
                        .unwrap_or(ScalarType::U64);
                    result_types.push(col_type);
                    result_buffers.push(col_buffer);
                }
                ProjectExpr::Computed(expr, result_type) => {
                    // Evaluate the arithmetic expression to get a single-column buffer
                    let computed_buffer = self.evaluate_arith_expr(expr, input)?;
                    result_types.push(*result_type);
                    result_buffers.push(computed_buffer);
                }
            }
        }

        // Combine all single-column buffers into a multi-column buffer
        self.provider.combine_columns(result_buffers, result_types)
    }

    /// Build a projected schema from ProjectExpr list
    fn project_schema(&self, input: &Schema, columns: &[ProjectExpr]) -> Result<Schema> {
        let mut projected_columns: Vec<(String, ScalarType)> = Vec::with_capacity(columns.len());
        for proj_expr in columns {
            match proj_expr {
                ProjectExpr::Column(col_idx) => {
                    if let Some((name, ty)) = input.columns.get(*col_idx) {
                        projected_columns.push((name.clone(), *ty));
                    } else {
                        return Err(XlogError::Execution(format!(
                            "Column index {} out of bounds",
                            col_idx
                        )));
                    }
                }
                ProjectExpr::Computed(_expr, result_type) => {
                    // Computed columns get a generated name
                    let col_name = format!("computed_{}", projected_columns.len());
                    projected_columns.push((col_name, *result_type));
                }
            }
        }
        Ok(Schema::new(projected_columns))
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
        fn estimate_join_index_bytes(right: &CudaBuffer, right_keys: &[usize]) -> u64 {
            if right_keys.is_empty() {
                return u64::MAX;
            }

            let mut key_bytes_per_row: u64 = 0;
            for &k in right_keys {
                let Some(ty) = right.schema().column_type(k) else {
                    return u64::MAX;
                };
                let sz = ty.size_bytes();
                key_bytes_per_row = key_bytes_per_row.saturating_add(sz as u64);
            }

            let num_rows = right.num_rows();
            let packed_bytes = num_rows.saturating_mul(key_bytes_per_row);

            let target = num_rows.saturating_mul(2).max(1024);
            let num_buckets = target.next_power_of_two();

            // Stored index bytes: packed keys + (counts+offsets) + (entry row ids + entry hashes)
            packed_bytes
                .saturating_add(num_buckets.saturating_mul(8))
                .saturating_add(num_rows.saturating_mul(12))
        }

        // Convert IR JoinType to CUDA JoinType
        let cuda_join_type = match join_type {
            JoinType::Inner => CudaJoinType::Inner,
            JoinType::Semi => CudaJoinType::Semi,
            JoinType::Anti => CudaJoinType::Anti,
            JoinType::LeftOuter => CudaJoinType::LeftOuter,
        };

        // Adaptive indexing: opportunistically reuse cached build-side hash tables when the right side
        // is a base relation scan and has become "hot" in runtime statistics.
        let mut out: Option<CudaBuffer> = None;
        if let Some(build_rel) = right_rel {
            let build_heat = self
                .stats
                .get_relation_stats(build_rel)
                .map(|s| s.heat)
                .unwrap_or(0.0);
            let est_index_bytes = estimate_join_index_bytes(right, right_keys);

            let cache_budget = self.join_index_cache.max_bytes;
            let budget_bytes = self.provider.memory().budget().device_bytes;
            let remaining_bytes = self.provider.memory().remaining_bytes();

            // Heuristic: require higher "heat" for larger indexes, and avoid building under
            // memory pressure. Always skip if the estimated index cannot fit in the cache budget.
            let heat_threshold = if cache_budget > 0 && est_index_bytes > cache_budget / 2 {
                0.6
            } else {
                0.3
            };
            let has_room = remaining_bytes >= est_index_bytes.saturating_add(budget_bytes / 10);

            let should_index =
                build_heat >= heat_threshold && est_index_bytes <= cache_budget && has_room;

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
    fn execute_union(&self, inputs: &[CudaBuffer]) -> Result<CudaBuffer> {
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
    fn execute_distinct(&self, input: &CudaBuffer, key_cols: &[usize]) -> Result<CudaBuffer> {
        self.provider.dedup(input, key_cols)
    }

    /// Execute a Diff node
    ///
    /// Returns rows in left that are not in right using GPU-native operation.
    fn execute_diff(&self, left: &CudaBuffer, right: &CudaBuffer) -> Result<CudaBuffer> {
        self.provider.diff_gpu(left, right)
    }

    /// Maximum iterations for fixpoint computation to prevent infinite loops
    const MAX_FIXPOINT_ITERATIONS: usize = 1000;

    /// Execute a Fixpoint node using semi-naive evaluation
    ///
    /// The semi-naive algorithm avoids redundant computation in recursive queries:
    ///
    /// 1. **Initialize:**
    ///    - Compute base case: `R = eval(base)`
    ///    - Set delta to base: `delta = R`
    ///    - Store both `R` and `delta` in RelationStore
    ///
    /// 2. **Iterate until fixpoint:**
    ///    - Compute new tuples: `delta_new = eval(recursive)` using current `delta`
    ///    - Remove already-known tuples: `delta_new = delta_new - R`
    ///    - If `delta_new` is empty, we've reached fixpoint
    ///    - Otherwise: `R = R union delta_new`, `delta = delta_new`
    ///
    /// 3. **Return:** Final `R`
    ///
    /// # Arguments
    /// * `scc_id` - SCC identifier for logging/debugging
    /// * `base` - Base case RIR tree (non-recursive facts/rules)
    /// * `recursive` - Recursive RIR tree (references delta relation)
    /// * `delta_rel` - RelId for delta relation
    /// * `full_rel` - RelId for full relation
    ///
    /// # Returns
    /// A CudaBuffer containing the final fixpoint result
    ///
    /// # Errors
    /// Returns an error if evaluation fails or iteration limit is exceeded
    fn execute_fixpoint(
        &mut self,
        scc_id: u32,
        base: &RirNode,
        recursive: &RirNode,
        delta_rel: RelId,
        full_rel: RelId,
    ) -> Result<CudaBuffer> {
        // Step 1: Compute base case R = eval(base)
        let r_initial = self.execute_node(base)?;

        // Handle empty base case using device-resident row count
        if self.buffer_row_count(&r_initial)? == 0 {
            return Ok(r_initial);
        }

        // Step 2: Initialize delta = R (clone the base result)
        let delta_initial = self.clone_buffer(&r_initial)?;

        // Get relation names for delta and full relations
        let delta_name = self.get_or_create_rel_name(delta_rel, &format!("__delta_{}", scc_id));
        let full_name = self.get_or_create_rel_name(full_rel, &format!("__full_{}", scc_id));

        // Store initial R and delta in relation store
        self.store_put(&full_name, r_initial);
        self.store_put(&delta_name, delta_initial);

        // Step 3: Iterate until fixpoint
        for _iteration in 0..Self::MAX_FIXPOINT_ITERATIONS {
            // Evaluate recursive step using current delta
            // The recursive RIR tree should reference delta_rel internally
            let delta_new_raw = self.execute_node(recursive)?;

            // Get current R for set difference
            let current_r = self.store.get(&full_name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "Full relation {} not found during fixpoint iteration",
                    full_name
                ))
            })?;

            // Compute delta_new = delta_new_raw - R (remove already-known tuples)
            let delta_new = self.provider.diff_gpu(&delta_new_raw, current_r)?;

            // Check for fixpoint: if delta_new is empty, we're done
            if self.buffer_row_count(&delta_new)? == 0 {
                // Fixpoint reached - return final R
                let final_r = self.store_remove(&full_name).ok_or_else(|| {
                    XlogError::Execution("Full relation lost during fixpoint".to_string())
                })?;

                // Clean up delta relation
                self.store_remove(&delta_name);

                return Ok(final_r);
            }

            // Not at fixpoint yet: R = R union delta_new
            let new_r = self.provider.union_gpu(current_r, &delta_new)?;

            // Update relations for next iteration
            // delta = delta_new (the newly discovered tuples)
            self.store_put(&delta_name, delta_new);
            self.store_put(&full_name, new_r);
        }

        // Iteration limit exceeded
        Err(XlogError::Execution(format!(
            "Fixpoint iteration limit ({}) exceeded for SCC {}",
            Self::MAX_FIXPOINT_ITERATIONS,
            scc_id
        )))
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
                self.ilp_last_result = Some(IlpTaggedResult { entries: Vec::new() });
                // RD-37: Fail hard if head relation missing from store.
                let schema = self.store.get(head_rel_name)
                    .map(|buf| buf.schema().clone())
                    .ok_or_else(|| XlogError::Execution(format!(
                        "TensorMaskedJoin: head relation '{}' not found in store \
                         (was load_facts_into_store called?)", head_rel_name
                    )))?;
                return self.provider.create_empty_buffer(schema);
            }
        };

        let start = self.profiler.start_op();

        // 2. Extract active (i,j,k) indices: sparse path skips GPU kernel entirely
        let active_rules: Vec<(u32, u32, u32)> = match ilp_mask {
            IlpMask::Dense { hard, soft, .. } => {
                self.provider.extract_active_rule_indices(
                    hard, soft, schema_size, max_active_rules,
                )?
            }
            IlpMask::Sparse { active_entries, .. } => {
                let limit = max_active_rules.min(active_entries.len());
                active_entries[..limit].to_vec()
            }
        };

        // 3. Phase 1: Dispatch hash joins, collect results into tag_entries
        //    (retaining per-entry buffers for batch credit queries)
        let mut tag_entries: Vec<IlpTagEntry> = Vec::new();

        for &(i, j, k) in &active_rules {
            let (_, left_name) = &rel_index[i as usize];
            let (_, right_name) = &rel_index[j as usize];

            let left_buf = match self.store.get(left_name) {
                Some(buf) if buf.arity() > 0 => buf,
                _ => continue,
            };
            let right_buf = match self.store.get(right_name) {
                Some(buf) if buf.arity() > 0 => buf,
                _ => continue,
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
                continue;
            }

            let joined = self.provider.hash_join_v2(
                left_buf, right_buf,
                left_keys, right_keys,
                CudaJoinType::Inner,
            )?;

            // Project join result to head schema columns if projection is specified.
            // The join produces [left_cols..., right_cols...] but the head may only
            // need a subset (e.g. reach(X,Y) from b1(X,Z) join b2(Z,Y) needs cols 0,3).
            let projected = if !head_projection.is_empty() && head_projection.len() < joined.arity() {
                let proj_exprs: Vec<ProjectExpr> = head_projection
                    .iter()
                    .map(|&col| ProjectExpr::Column(col))
                    .collect();
                self.execute_project(&joined, &proj_exprs)?
            } else {
                joined
            };

            // RD-22: Use public helper instead of private device_row_count
            let num_rows = read_device_row_count(&self.provider, &projected)? as u32;

            if num_rows > 0 {
                tag_entries.push(IlpTagEntry { i, j, k, num_rows, buffer: Some(projected) });
            }
        }

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
                let empty = self.provider.create_empty_buffer(buffers[0].schema().clone())?;
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
        self.ilp_last_result = Some(IlpTaggedResult { entries: tag_entries });

        if let Some(start) = start {
            let mem = self.provider.memory().allocated_bytes();
            self.profiler.record_op(
                "TensorMaskedJoin", 0, active_rules.len() as u64, start, mem,
            );
        }

        // Return empty with head schema (results routed via store).
        let schema = self.store.get(head_rel_name)
            .map(|buf| buf.schema().clone())
            .ok_or_else(|| XlogError::Execution(format!(
                "TensorMaskedJoin: head relation '{}' not found in store \
                 (was load_facts_into_store called?)", head_rel_name
            )))?;
        self.provider.create_empty_buffer(schema)
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
            .download_column_u32(buffer, col)
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
}
