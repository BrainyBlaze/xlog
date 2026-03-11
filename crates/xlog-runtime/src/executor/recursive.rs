//! Recursive SCC execution using semi-naive fixpoint iteration.

use std::collections::{HashMap, HashSet};

use xlog_core::{RelId, Result, Schema, XlogError};
use xlog_cuda::CudaBuffer;
use xlog_ir::{ExecutionPlan, RirNode, Stratum};

use super::delta::DeltaRelationTracker;
use super::Executor;

impl Executor {
    /// Maximum iterations for fixpoint computation to prevent infinite loops
    const MAX_FIXPOINT_ITERATIONS: usize = 1000;

    /// Stub: always returns an error directing callers to use `execute_plan` instead.
    pub fn execute_stratum(&mut self, _stratum: &Stratum) -> Result<()> {
        Err(XlogError::Execution(
            "execute_stratum cannot be called directly; use execute_plan instead which provides \
             the required rules_by_scc context"
                .to_string(),
        ))
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

    /// Execute a stratum (internal implementation)
    ///
    /// Processes all SCCs in the stratum by executing their rules.
    /// For recursive SCCs, uses semi-naive fixpoint iteration.
    pub(super) fn execute_stratum_impl(
        &mut self,
        stratum: &Stratum,
        plan: &ExecutionPlan,
    ) -> Result<()> {
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

        let mut delta_tracker = DeltaRelationTracker::new();
        for pred in &recursive_preds {
            let rel_id = RelId(next_rel_id);
            next_rel_id = next_rel_id.saturating_add(1);
            let name = format!("__delta_{}_{}", pred, rel_id.0);
            self.register_relation(rel_id, &name);
            delta_tracker.insert(pred.clone(), rel_id, name);
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

            let (_delta_rel_id, delta_name) = delta_tracker.get(pred).ok_or_else(|| {
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
                    let (_delta_rel_id, delta_name) = match delta_tracker.get(&pred_name) {
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
                        delta_tracker.get(&pred_name).ok_or_else(|| {
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

                let (_delta_rel_id, delta_name) = delta_tracker.get(pred).ok_or_else(|| {
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
                let (_delta_rel_id, delta_name) = delta_tracker.get(pred).ok_or_else(|| {
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
        for (_pred, (rel_id, delta_name)) in delta_tracker.into_inner() {
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

    /// Execute a Fixpoint node using semi-naive evaluation
    ///
    /// The semi-naive algorithm avoids redundant computation in recursive queries:
    ///
    /// 1. **Initialize:**
    ///    - Compute base case: `R = base_result`
    ///    - Set delta to base: `delta = R`
    ///    - Store both `R` and `delta` in RelationStore
    ///
    /// 2. **Iterate until fixpoint:**
    ///    - Compute new tuples: `delta_new = recursive_result` using current `delta`
    ///    - Remove already-known tuples: `delta_new = delta_new - R`
    ///    - If `delta_new` is empty, we have reached fixpoint
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
    /// Returns an error if iteration limit is exceeded
    pub(super) fn execute_fixpoint(
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

            // Check for fixpoint: if delta_new is empty, we are done
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
}
