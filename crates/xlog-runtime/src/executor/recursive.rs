//! Recursive SCC execution using semi-naive fixpoint iteration.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::time::Instant;

use xlog_core::{RelId, Result, Schema, XlogError};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_ir::{ExecutionPlan, RirNode, Stratum};

use crate::profiler::Profiler;

use super::delta::DeltaRelationTracker;
use super::Executor;

impl Executor {
    /// Maximum iterations for fixpoint computation to prevent infinite loops
    const MAX_FIXPOINT_ITERATIONS: usize = 1000;

    /// Union a batch of same-head contributions in one multiway pass,
    /// recording a single profiled "union" op for the whole batch.
    ///
    /// Takes the provider and profiler as explicit arguments (instead of
    /// `&mut self`) so call sites can hold `self.store` borrows across the
    /// call; the field borrows stay disjoint.
    fn union_batch_profiled(
        provider: &CudaKernelProvider,
        profiler: &mut Profiler,
        inputs: &[&CudaBuffer],
    ) -> Result<CudaBuffer> {
        let union_input: u64 = inputs.iter().map(|b| b.num_rows()).sum();
        let start = profiler.start_op();
        let merged = provider.union_many_gpu(inputs)?;
        if let Some(start) = start {
            let mem = provider.memory().allocated_bytes();
            profiler.record_op("union", union_input, merged.num_rows(), start, mem);
            profiler.record_peak_memory(mem);
        }
        Ok(merged)
    }

    /// For a `MultiWayJoin` or `ChainJoin` body, try the specialized WCOJ
    /// dispatchers first; on decline, fall back to the embedded fallback
    /// subtree via `execute_node`. For any other RIR variant, defer to
    /// `execute_node` directly.
    ///
    /// Used at two sites in the recursive engine: the seeding pass, where
    /// stable rules and recursive rules get their initial dispatch on the full
    /// body, and the per-variant loop, where each recursive scan with a
    /// non-empty delta is rewritten to its delta `RelId` for one dispatch.
    /// Multi-recursive bodies, including distinct recursive predicates and
    /// same-predicate self-recursive bodies, reach a `MultiWayJoin` here after
    /// the promoter admits bodies with more than one recursive scan; the
    /// per-variant rewrite loop builds one variant per recursive occurrence
    /// with a non-empty delta and dispatches each via this helper.
    ///
    /// Counter semantics: `wcoj_*_dispatch_count` increments once per
    /// successful WCOJ kernel result: once per recursive rule, iteration, and
    /// variant. Non-recursive dispatch sites increment once per rule per call.
    fn execute_wcoj_or_fallback_node(&mut self, node: &RirNode) -> Result<CudaBuffer> {
        if let RirNode::ChainJoin { .. } = node {
            if let Some(buf) = self.try_dispatch_chain_on_body(node)? {
                return Ok(buf);
            }
            return self.execute_node(node);
        }
        if let RirNode::MultiWayJoin { .. } = node {
            // Triangle, 4-cycle, then K-clique. A body cannot
            // match more than one paper-derived shape (different
            // atom counts). The dispatcher's own gate handles
            // env-var / config / adaptive decisions; this site is
            // purely structural.
            if let Some(buf) = self.try_dispatch_wcoj_triangle_on_body(node)? {
                return Ok(buf);
            }
            if let Some(buf) = self.try_dispatch_wcoj_4cycle_on_body(node)? {
                return Ok(buf);
            }
            // Recursive clique bodies use the same launch-local metadata
            // builders as non-recursive K-clique dispatch, so rewritten
            // semi-naive variants are eligible here too.
            if let Some(buf) = self.try_dispatch_wcoj_clique5_on_body(node)? {
                return Ok(buf);
            }
            if let Some(buf) = self.try_dispatch_wcoj_clique6_on_body(node)? {
                return Ok(buf);
            }
            if let Some(buf) = self.try_dispatch_wcoj_clique7_on_body(node)? {
                return Ok(buf);
            }
            if let Some(buf) = self.try_dispatch_wcoj_clique8_on_body(node)? {
                return Ok(buf);
            }
            // Generalized Free Join dispatch for every multiway shape the
            // dedicated dispatchers declined. The hook re-checks dedicated
            // shapes structurally, so it only fires on general bodies.
            if let Some(buf) = self.try_dispatch_free_join(node)? {
                return Ok(buf);
            }
        }
        self.execute_node(node)
    }

    fn refresh_kclique_edge_metadata_after_merge(
        &mut self,
        rules: &[xlog_ir::CompiledRule],
        pred: &str,
    ) {
        let start = Instant::now();
        let affected_rules = rules
            .iter()
            .filter(|rule| self.kclique_body_mentions_pred(&rule.body, pred))
            .count() as u64;
        self.record_kclique_histogram_refresh_time(start, affected_rules);
    }

    fn record_kclique_histogram_refresh_time(&mut self, start: Instant, affected_rules: u64) {
        if affected_rules == 0 {
            return;
        }
        self.kclique_histogram_refresh_count = self
            .kclique_histogram_refresh_count
            .saturating_add(affected_rules);
        self.kclique_histogram_refresh_nanos = self
            .kclique_histogram_refresh_nanos
            .saturating_add(start.elapsed().as_nanos());
    }

    fn kclique_body_mentions_pred(&self, node: &RirNode, pred: &str) -> bool {
        let RirNode::MultiWayJoin {
            inputs, var_order, ..
        } = node
        else {
            return false;
        };
        let Some(order) = var_order.as_ref().and_then(|order| order.kclique.as_ref()) else {
            return false;
        };
        if !matches!(order.k, 5..=8) {
            return false;
        }
        inputs.iter().any(|input| {
            let RirNode::Scan { rel } = input else {
                return false;
            };
            self.rel_names.get(rel).is_some_and(|name| name == pred)
        })
    }

    /// Stub: always returns an error directing callers to use `execute_plan` instead.
    pub fn execute_stratum(&mut self, _stratum: &Stratum) -> Result<()> {
        Err(XlogError::Execution(
            "execute_stratum cannot be called directly; use execute_plan instead which provides \
             the required rules_by_scc context"
                .to_string(),
        ))
    }

    /// Execute all rules in a non-recursive strongly connected component once.
    ///
    /// Each contiguous run of same-head rules is merged into the store with
    /// one multiway union instead of one union per rule, so many-rule heads
    /// stay linear in their total rows. Flushing on every head switch (not
    /// once per SCC group) preserves rule-order dataflow: promoter-generated
    /// helper rules share an SCC group with their consumer, so a helper's
    /// head must be installed before the next rule reads it.
    pub fn execute_non_recursive_scc(&mut self, rules: &[xlog_ir::CompiledRule]) -> Result<()> {
        let mut pending_head: Option<&str> = None;
        let mut pending: Vec<CudaBuffer> = Vec::new();
        for rule in rules {
            if pending_head != Some(rule.head.as_str()) {
                if let Some(head) = pending_head.take() {
                    let batch = std::mem::take(&mut pending);
                    self.install_plain_head_batch(head, batch)?;
                }
                pending_head = Some(rule.head.as_str());
            } else if !pending.is_empty() && self.body_reads_own_head(rule) {
                // A same-head rule that reads its own head (a non-monotone
                // singleton SCC, admitted under the probabilistic profile)
                // must observe prior same-head contributions exactly as the
                // sequential per-rule path did: flush before evaluating it.
                let batch = std::mem::take(&mut pending);
                self.install_plain_head_batch(&rule.head, batch)?;
            }
            let result = self.execute_node(&rule.body)?;
            pending.push(result);
        }
        if let Some(head) = pending_head {
            self.install_plain_head_batch(head, pending)?;
        }
        Ok(())
    }

    /// Whether the rule's body reads its own head relation in a way that is
    /// sensitive to same-pass installs. A body that IS a bare scan of the
    /// head (`h :- h`, the planner's identity/carry rule for fact
    /// predicates) is exempt: its scan result is always a subset of the
    /// final merged relation, so batching cannot change the outcome. Any
    /// other self-reading shape (negation, joins, or projections over the
    /// head — possible only for non-monotone singleton SCCs admitted under
    /// the probabilistic profile) must flush for sequential parity.
    fn body_reads_own_head(&self, rule: &xlog_ir::CompiledRule) -> bool {
        if let RirNode::Scan { rel } = &rule.body {
            if self.get_rel_name(*rel).is_some_and(|n| n == rule.head) {
                return false;
            }
        }
        let mut scans = Vec::new();
        Self::collect_scan_rels(&rule.body, &mut scans);
        scans
            .into_iter()
            .any(|rel| self.get_rel_name(rel).is_some_and(|n| n == rule.head))
    }

    /// Install one head's batched results, skipping empty contributions.
    /// Mirrors the pre-batching per-rule behavior: an existing relation is
    /// left untouched when every contribution is empty, and a lone fresh
    /// non-empty result records a single-input union (internally one dedup)
    /// like the dispatched install path, so `--stats` accounting stays
    /// uniform across both installers. The store is only mutated after the
    /// merge succeeds, so a failed union leaves the existing relation
    /// intact.
    fn install_plain_head_batch(&mut self, head: &str, results: Vec<CudaBuffer>) -> Result<()> {
        let non_empty: Vec<&CudaBuffer> = results.iter().filter(|r| !r.is_empty()).collect();

        // An existing empty relation (e.g. a pre-seeded schema buffer)
        // carries no rows to merge, so it takes the fresh-install path,
        // matching the dispatched installer.
        let existing = self.store.get(head).filter(|buf| !buf.is_empty());
        if let Some(existing) = existing {
            if non_empty.is_empty() {
                // No new rows for this head: leave the relation untouched.
                return Ok(());
            }
            let mut union_inputs = Vec::with_capacity(non_empty.len() + 1);
            union_inputs.push(existing);
            union_inputs.extend(non_empty);
            let merged =
                Self::union_batch_profiled(&self.provider, &mut self.profiler, &union_inputs)?;
            self.store_put(head, merged);
        } else if non_empty.is_empty() {
            // All contributions are empty: an absent head gets an empty
            // relation with the result schema; an existing (empty) head is
            // left untouched.
            if self.store.get(head).is_none() {
                let first = results.into_iter().next().ok_or_else(|| {
                    XlogError::Execution(format!("No results collected for head {}", head))
                })?;
                self.store_put(head, first);
            }
        } else {
            let merged =
                Self::union_batch_profiled(&self.provider, &mut self.profiler, &non_empty)?;
            self.store_put(head, merged);
        }
        Ok(())
    }

    /// Install one head's batched non-recursive results with a single
    /// profiled multiway union. Each entry's flag records whether the
    /// route's output is already sorted+deduped, so a lone fresh WCOJ
    /// result installs without a redundant dedup pass, mirroring the
    /// per-route install behavior. The store is only mutated after the
    /// merge succeeds, so a failed union leaves the existing relation
    /// intact.
    fn install_dispatched_head_batch(
        &mut self,
        head: &str,
        results: Vec<(CudaBuffer, bool)>,
    ) -> Result<()> {
        // Union with existing result if the predicate already has rows. An
        // existing empty relation (e.g. a pre-seeded schema buffer)
        // contributes nothing, so it takes the fresh-install path — which
        // is what lets a lone already-deduped WCOJ result skip the
        // redundant dedup in production runs.
        let existing = self.store.get(head).filter(|buf| !buf.is_empty());
        if let Some(existing) = existing {
            let mut union_inputs: Vec<&CudaBuffer> = Vec::with_capacity(results.len() + 1);
            union_inputs.push(existing);
            union_inputs.extend(results.iter().map(|(buf, _)| buf));
            let merged =
                Self::union_batch_profiled(&self.provider, &mut self.profiler, &union_inputs)?;
            self.store_put(head, merged);
        } else if results.len() == 1 {
            let (result, already_deduped) = results.into_iter().next().expect("len checked");
            if already_deduped || result.is_empty() {
                self.store_put(head, result);
            } else {
                // Set semantics for a lone raw result: a single-input
                // multiway union (internally one dedup), recorded as a
                // "union" op like the union-with-empty install it replaces.
                let merged =
                    Self::union_batch_profiled(&self.provider, &mut self.profiler, &[&result])?;
                self.store_put(head, merged);
            }
        } else {
            let union_inputs: Vec<&CudaBuffer> = results.iter().map(|(buf, _)| buf).collect();
            let merged =
                Self::union_batch_profiled(&self.provider, &mut self.profiler, &union_inputs)?;
            self.store_put(head, merged);
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
                    // Recursive SCC: use semi-naive fixpoint iteration. The
                    // recursive engine invokes WCOJ dispatch via
                    // `execute_wcoj_or_fallback_node` on both the seeding
                    // pass and per-variant evaluation when the promoted body
                    // shape is eligible.
                    self.execute_recursive_scc(rules)?;
                } else {
                    // Non-recursive SCC: execute rules once, merging each
                    // contiguous run of same-head results with one multiway
                    // union instead of one union per rule, so many-rule heads
                    // stay linear in their total rows. Flushing on every head
                    // switch (not once per SCC group) preserves rule-order
                    // dataflow: promoter-generated helper rules share an SCC
                    // group with their consumer, so a helper's head must be
                    // installed before the next rule dispatches against it.
                    let mut pending_head: Option<&str> = None;
                    let mut pending: Vec<(CudaBuffer, bool)> = Vec::new();
                    for rule in rules {
                        if pending_head != Some(rule.head.as_str()) {
                            if let Some(head) = pending_head.take() {
                                let batch = std::mem::take(&mut pending);
                                self.install_dispatched_head_batch(head, batch)?;
                            }
                            pending_head = Some(rule.head.as_str());
                        } else if !pending.is_empty() && self.body_reads_own_head(rule) {
                            // A same-head rule that reads its own head (a
                            // non-monotone singleton SCC, admitted under the
                            // probabilistic profile) must observe prior
                            // same-head contributions exactly as the
                            // sequential per-rule path did: flush before
                            // evaluating it.
                            let batch = std::mem::take(&mut pending);
                            self.install_dispatched_head_batch(&rule.head, batch)?;
                        }

                        // Route two-atom ChainJoin bodies before the
                        // triangle/4-cycle/KC attempts. The dispatcher
                        // silently declines on non-chain bodies or when
                        // the env gate disables the route.
                        let entry = if let Some(chain_result) =
                            self.try_dispatch_chain_on_body(&rule.body)?
                        {
                            (chain_result, false)
                        }
                        // WCOJ triangle dispatch, gated by runtime configuration.
                        // Try to short-circuit the rule via the GPU
                        // 3-way kernel. On Some(_), record the result
                        // and skip the binary-join path for this rule.
                        // On None (gate off, shape mismatch, missing
                        // input, kernel error), fall through silently.
                        // See `wcoj_dispatch::try_dispatch_wcoj_triangle`
                        // for the full match contract. WCOJ output is
                        // already sorted+deduped, so a lone fresh
                        // install needs no dedup pass.
                        else if let Some(wcoj_result) = self.try_dispatch_wcoj_triangle(rule)? {
                            (wcoj_result, true)
                        }
                        // WCOJ 4-cycle dispatch.
                        // Same pattern as triangle. Order is a doc
                        // anchor — a body cannot match both shapes
                        // (different atom counts), so triangle's
                        // earlier attempt always returns None on a
                        // 4-cycle body and vice versa.
                        else if let Some(wcoj_result) = self.try_dispatch_wcoj_4cycle(rule)? {
                            (wcoj_result, true)
                        }
                        // K-clique dispatch for k=5..k=8.
                        // Same shape-gated default-dispatch
                        // pattern as triangle / 4-cycle; silent
                        // fallback to MultiWayJoin.fallback on
                        // dispatcher decline or kernel error.
                        else if let Some(wcoj_result) = self.try_dispatch_wcoj_clique5(rule)? {
                            (wcoj_result, true)
                        } else if let Some(wcoj_result) = self.try_dispatch_wcoj_clique6(rule)? {
                            (wcoj_result, true)
                        } else if let Some(wcoj_result) = self.try_dispatch_wcoj_clique7(rule)? {
                            (wcoj_result, true)
                        } else if let Some(wcoj_result) = self.try_dispatch_wcoj_clique8(rule)? {
                            (wcoj_result, true)
                        }
                        // Generalized Free Join dispatch for every multiway
                        // shape the dedicated dispatchers above declined. The
                        // dispatcher re-checks those shapes structurally, so
                        // it only fires on general bodies. Unlike the
                        // dedicated kernels, the frontier engine emits one row
                        // per derivation path, so its output still needs the
                        // dedup the per-head merge (or the fresh-install
                        // dedup) provides.
                        else if let Some(fj_result) = self.try_dispatch_free_join(&rule.body)? {
                            (fj_result, false)
                        } else {
                            // When WCOJ dispatch declines on a `MultiWayJoin`
                            // body (gate off, kernel error, adaptive score below
                            // threshold, ...), execute the embedded `fallback`,
                            // the post-optimizer binary-join tree the promoter
                            // captured. `execute_node`'s `MultiWayJoin` arm is the
                            // defensive safety net; explicit destructuring here
                            // keeps the intent visible at the dispatch site.
                            let body_to_execute = match &rule.body {
                                xlog_ir::RirNode::MultiWayJoin { fallback, .. }
                                | xlog_ir::RirNode::ChainJoin { fallback, .. } => fallback.as_ref(),
                                other => other,
                            };
                            (self.execute_node(body_to_execute)?, false)
                        };

                        pending.push(entry);
                    }
                    if let Some(head) = pending_head {
                        self.install_dispatched_head_batch(head, pending)?;
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
        // Reset the per-iteration stats trace at SCC entry so tests see a
        // fresh trace per invocation. Gated on the `recursive-stats-trace`
        // feature; default OFF.
        #[cfg(feature = "recursive-stats-trace")]
        {
            self.last_recursive_stats_trace.entries.clear();
        }
        // Identify SCC predicates from rule heads (these are the recursive IDBs).
        let mut recursive_pred_names: BTreeSet<String> = BTreeSet::new();
        let mut schema_by_pred: HashMap<String, Schema> = HashMap::new();
        for rule in rules {
            recursive_pred_names.insert(rule.head.clone());
            if rule.meta.schema.arity() > 0 {
                schema_by_pred
                    .entry(rule.head.clone())
                    .or_insert_with(|| rule.meta.schema.clone());
            }
        }
        let recursive_pred_lookup: HashSet<String> = recursive_pred_names.iter().cloned().collect();
        let recursive_preds: Vec<String> = recursive_pred_names.into_iter().collect();

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

        // Execute all rules once against the current store to seed initial results.
        // Accumulate per-head before mutating the store to avoid order dependence.
        //
        // Route through `execute_wcoj_or_fallback_node` so promoted
        // MultiWayJoin bodies for stable and linear-recursive triangles or
        // 4-cycles get a chance at WCOJ dispatch on the seeding pass. Stable
        // rules with zero recursive scans only run here, so without this hook
        // they would never see a kernel.
        let mut derived_initial: HashMap<String, Vec<CudaBuffer>> = HashMap::new();
        for rule in rules {
            let result = self.execute_wcoj_or_fallback_node(&rule.body)?;
            derived_initial
                .entry(rule.head.clone())
                .or_default()
                .push(result);
        }

        // Initialize delta from the newly-derived tuples only.
        //
        // This supports incremental maintenance: if the SCC is executed again after EDB inserts,
        // the delta relations start with only the *new* tuples, not a full rescan of the current
        // fixed point.
        for pred in &recursive_preds {
            // Read the prior full relation in place: the store is only
            // mutated after the merge succeeds, so a failed union leaves the
            // relation (and its version counter) intact.
            let full_old = self
                .store
                .get(pred)
                .ok_or_else(|| XlogError::Execution(format!("Missing relation: {}", pred)))?;

            let derived = derived_initial.remove(pred).unwrap_or_default();

            // One multiway union per head: the prior full relation and every
            // same-head seed contribution are concatenated, sorted, and
            // deduplicated in a single pass instead of one union per rule.
            let mut union_inputs: Vec<&CudaBuffer> = Vec::with_capacity(derived.len() + 1);
            union_inputs.push(full_old);
            union_inputs.extend(derived.iter());
            let full_new =
                Self::union_batch_profiled(&self.provider, &mut self.profiler, &union_inputs)?;
            drop(union_inputs);
            drop(derived);

            let delta_name = delta_tracker.delta_name(pred)?;

            let full_old_rows = self.buffer_row_count(full_old)?;
            let full_new_rows = self.buffer_row_count(&full_new)?;
            let delta_initial = if full_new_rows == 0 {
                self.create_empty_buffer(full_new.schema().clone())?
            } else if full_old_rows == 0 {
                self.clone_buffer(&full_new)?
            } else {
                let diff_input = full_new.num_rows() + full_old.num_rows();
                let start = self.profiler.start_op();
                let diffed = self.provider.diff_gpu(&full_new, full_old)?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("diff", diff_input, diffed.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }
                diffed
            };

            // Seed-iteration cardinality refresh. Capture the actual
            // `delta_initial` row count before the `store_put` move; after the
            // move, the buffer is gone.
            let delta_initial_rows = self.buffer_row_count(&delta_initial)? as u64;
            let seed_full_rows = full_new_rows as u64;
            // Pre-resolve rel_id lookups before the &mut self stats
            // borrow below.
            let full_rel_opt = self.name_to_rel_id(pred);
            let delta_rel = delta_tracker.delta_rel_id(pred)?;

            self.store_put(pred, full_new);
            self.store_put(delta_name, delta_initial);

            // Stats updates fire whether or not WCOJ ran on the seed
            // pass. update_cardinality is a no-op for unregistered
            // rel_ids (defensive: tests that don't register an IDB
            // head get a no-op for the full_rel write).
            if let Some(full_rel) = full_rel_opt {
                self.stats.update_cardinality(full_rel, seed_full_rows);
            }
            self.stats.update_cardinality(delta_rel, delta_initial_rows);

            // Seed stats trace entry, gated on `recursive-stats-trace`.
            #[cfg(feature = "recursive-stats-trace")]
            self.last_recursive_stats_trace
                .entries
                .push(super::RecursiveStatsTraceEntry {
                    iteration: 0,
                    pred: pred.clone(),
                    full_rel: full_rel_opt.unwrap_or(RelId(u32::MAX)),
                    delta_rel,
                    full_rows: seed_full_rows,
                    delta_rows: delta_initial_rows,
                    phase: super::RecursiveStatsPhase::Seed,
                    binary_est_for_variant: None,
                });
        }

        // Iterate until no new tuples are produced.
        let mut reached_fixpoint = false;
        let max_iterations = self.config.max_iterations as usize;
        let mut iteration_count = 0usize;
        // D3 — per-fixpoint dispatch context for the factorized delta
        // (domain bounds + normalized EDB statics are cached across
        // iterations).
        let mut fd_ctx = super::wcoj_dispatch::FactorizedDeltaCtx::default();
        for _iteration in 0..max_iterations {
            iteration_count += 1;
            // Compute delta_new_raw per head by evaluating each rule once per recursive Scan occurrence.
            // Contributions are collected unmerged; the per-head finalize
            // below unions each head's batch in one multiway pass instead of
            // one union per rule.
            let mut delta_new_raw_by_head: HashMap<String, Vec<CudaBuffer>> = HashMap::new();
            // D3 — factorized novel sets per head: already diffed
            // against the stable relation and full-row deduped at
            // dispatch time. Kept separate from the raw accumulator so
            // all-factorized heads can skip the legacy diff entirely.
            let mut delta_novel_by_head: HashMap<String, Vec<CudaBuffer>> = HashMap::new();

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
                    if !recursive_pred_lookup.contains(&pred_name) {
                        continue;
                    }

                    // Skip variants where the delta for this predicate is empty.
                    let delta_name = match delta_tracker.get(&pred_name) {
                        Some((_rel_id, name)) => name.as_str(),
                        None => continue,
                    };
                    let delta_is_empty = match self.store.get(delta_name) {
                        Some(delta) => self.buffer_row_count(delta)? == 0,
                        None => true,
                    };
                    if delta_is_empty {
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

                let mut rule_delta_raw: Vec<CudaBuffer> = Vec::new();
                let mut rule_delta_novel: Vec<CudaBuffer> = Vec::new();
                for (rel_id, occ, pred_name) in variants {
                    let delta_rel_id = delta_tracker.delta_rel_id(&pred_name)?;

                    let variant_node =
                        Self::rewrite_scan_nth(&rule.body, rel_id, occ, delta_rel_id).ok_or_else(
                            || {
                                XlogError::Execution(format!(
                                    "Failed to rewrite rule body for predicate {}",
                                    pred_name
                                ))
                            },
                        )?;

                    // Try the factorized delta pipeline first: a qualifying
                    // ChainJoin variant returns the novel set directly
                    // (already diffed against the head's stable relation and
                    // deduped). Declines are silent and fall through to the
                    // legacy path.
                    if let Some(novel) = self.try_dispatch_factorized_delta(
                        &variant_node,
                        delta_rel_id,
                        &rule.head,
                        &recursive_pred_lookup,
                        &mut fd_ctx,
                    )? {
                        rule_delta_novel.push(novel);
                        continue;
                    }

                    // Try WCOJ on the rewritten variant body before falling
                    // back to the binary-join walker.
                    // For a linear-recursive triangle/4-cycle, the
                    // variant has one Scan's RelId swapped to its
                    // delta — the kernel reads from the delta store
                    // entry transparently, no special-case dispatch
                    // logic needed.
                    let out = self.execute_wcoj_or_fallback_node(&variant_node)?;
                    rule_delta_raw.push(out);
                }

                // D3 — a rule with BOTH factorized and legacy variant
                // outputs folds its novel rows into the raw batch
                // (the legacy diff is a no-op on novel rows, so this is
                // sound); an all-factorized rule keeps its novel rows on
                // the diff-free track.
                if !rule_delta_raw.is_empty() {
                    rule_delta_raw.append(&mut rule_delta_novel);
                    delta_new_raw_by_head
                        .entry(rule.head.clone())
                        .or_default()
                        .append(&mut rule_delta_raw);
                } else if !rule_delta_novel.is_empty() {
                    delta_novel_by_head
                        .entry(rule.head.clone())
                        .or_default()
                        .append(&mut rule_delta_novel);
                }
            }

            // Finalize delta_new per head: delta_new = dedup(delta_raw - full).
            delta_tracker.begin_iteration();

            for pred in &recursive_preds {
                let full = self
                    .store
                    .get(pred)
                    .ok_or_else(|| XlogError::Execution(format!("Missing relation: {}", pred)))?;
                // Capture the current full row count for the trace's
                // `full_rows` field before this iteration's delta relation is
                // replaced. Gated on `recursive-stats-trace` so production
                // builds do not compute it.
                #[cfg(feature = "recursive-stats-trace")]
                let pre_phase4_full_rows = self.buffer_row_count(full)? as u64;

                let mut raw_bufs = delta_new_raw_by_head.remove(pred).unwrap_or_default();
                let mut novel_bufs = delta_novel_by_head.remove(pred).unwrap_or_default();
                // D3 — when a head received both raw and factorized
                // contributions (different rules), fold the novel rows
                // into the raw batch before the legacy diff (sound: the
                // diff is a no-op on novel rows). An all-factorized
                // head skips the diff entirely — its novel rows are
                // already diffed and deduped by construction.
                if !raw_bufs.is_empty() {
                    raw_bufs.append(&mut novel_bufs);
                }
                let delta_new = if !novel_bufs.is_empty() {
                    if novel_bufs.len() == 1 {
                        novel_bufs.pop().expect("len checked")
                    } else {
                        // Novel sets are deduped per rule, not across rules,
                        // so a multi-rule head still unions its novel batch.
                        let union_inputs: Vec<&CudaBuffer> = novel_bufs.iter().collect();
                        Self::union_batch_profiled(
                            &self.provider,
                            &mut self.profiler,
                            &union_inputs,
                        )?
                    }
                } else if !raw_bufs.is_empty() {
                    let delta_raw = if raw_bufs.len() == 1 {
                        raw_bufs.pop().expect("len checked")
                    } else {
                        // One multiway union per head instead of one union
                        // per rule contribution.
                        let union_inputs: Vec<&CudaBuffer> = raw_bufs.iter().collect();
                        Self::union_batch_profiled(
                            &self.provider,
                            &mut self.profiler,
                            &union_inputs,
                        )?
                    };
                    drop(raw_bufs);
                    if self.buffer_row_count(&delta_raw)? == 0 {
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

                let delta_name = delta_tracker.delta_name(pred)?.to_string();
                let delta_new_rows = self.buffer_row_count(&delta_new)? as u64;
                if delta_new_rows != 0 {
                    delta_tracker.mark_changed();
                }
                // Pre-resolve rel_id lookups before the &mut self
                // store_put + stats update below. `full_rel_opt` is
                // only used by the trace under the
                // `recursive-stats-trace` feature.
                #[cfg(feature = "recursive-stats-trace")]
                let full_rel_opt = self.name_to_rel_id(pred);
                let delta_rel = delta_tracker.delta_rel_id(pred)?;
                self.store_put(&delta_name, delta_new);

                // Refresh the delta relation cardinality after computing this
                // iteration's delta. The full relation cardinality is not
                // updated here because the full relation has not changed yet
                // this iteration; the merge step owns that.
                self.stats.update_cardinality(delta_rel, delta_new_rows);

                // Delta stats trace entry, gated on `recursive-stats-trace`.
                // binary_est_for_variant captures the cost model's
                // first-binary-hop estimate for the linear-recursive fixtures
                // (`pred == "e1"` rewrites
                // Scan(e1) → Scan(delta_e1); first hop is
                // `delta_e1.col1 ⋈ e2.col0`). Populated inline because
                // delta_rel is unregistered at fixpoint exit, so the
                // test cannot recompute after `execute_plan` returns.
                #[cfg(feature = "recursive-stats-trace")]
                let binary_est_for_variant: Option<u64> = if pred == "e1" {
                    self.name_to_rel_id("e2").map(|e2_rel| {
                        self.stats
                            .estimate_join_cardinality(delta_rel, e2_rel, &[1], &[0])
                    })
                } else {
                    None
                };
                #[cfg(feature = "recursive-stats-trace")]
                self.last_recursive_stats_trace
                    .entries
                    .push(super::RecursiveStatsTraceEntry {
                        iteration: iteration_count,
                        pred: pred.clone(),
                        full_rel: full_rel_opt.unwrap_or(RelId(u32::MAX)),
                        delta_rel,
                        full_rows: pre_phase4_full_rows,
                        delta_rows: delta_new_rows,
                        phase: super::RecursiveStatsPhase::Phase2Delta,
                        binary_est_for_variant,
                    });
            }

            // Fixpoint reached if no deltas produced.
            if delta_tracker.is_converged() {
                reached_fixpoint = true;
                self.profiler.record_iterations(iteration_count);
                break;
            }

            // Merge deltas into full relations.
            for pred in &recursive_preds {
                let dn = delta_tracker.delta_name(pred)?.to_string();
                // Read both relations in place: the store is only mutated
                // after the merge succeeds, so a failed union leaves the
                // full relation (and its version counter) intact, and the
                // unchanged delta never needs a remove/re-put round trip.
                let full_old = self
                    .store
                    .get(pred)
                    .ok_or_else(|| XlogError::Execution(format!("Missing relation: {}", pred)))?;
                let delta = self
                    .store
                    .get(&dn)
                    .ok_or_else(|| XlogError::Execution(format!("Missing relation: {}", dn)))?;

                if self.buffer_row_count(delta)? == 0 {
                    // Zero-delta short-circuit: full and delta are unchanged
                    // this iteration. The delta relation record with zero
                    // rows stands, and the full relation record from the prior
                    // merge stands. No additional update is needed.
                    continue;
                }

                let union_input = full_old.num_rows() + delta.num_rows();
                let start = self.profiler.start_op();
                let merged = self.provider.union_gpu(full_old, delta)?;
                if let Some(start) = start {
                    let mem = self.provider.memory().allocated_bytes();
                    self.profiler
                        .record_op("union", union_input, merged.num_rows(), start, mem);
                    self.profiler.record_peak_memory(mem);
                }

                let full_new = merged;
                // Capture `full_new`'s row count before the `store_put` move
                // and pre-resolve `full_rel_opt` before the mutable stats
                // borrow. The delta row count and delta relation id are only
                // used by the trace under the `recursive-stats-trace` feature.
                let full_new_rows_phase4 = self.buffer_row_count(&full_new)? as u64;
                #[cfg(feature = "recursive-stats-trace")]
                let delta_rows_phase4 = self.buffer_row_count(delta)? as u64;
                let full_rel_opt = self.name_to_rel_id(pred);
                #[cfg(feature = "recursive-stats-trace")]
                let delta_rel = delta_tracker.delta_rel_id(pred)?;
                self.store_put(pred, full_new);

                // Record the full relation's new cardinality. The delta
                // relation was already recorded for this iteration.
                if let Some(full_rel) = full_rel_opt {
                    self.stats
                        .update_cardinality(full_rel, full_new_rows_phase4);
                }
                self.refresh_kclique_edge_metadata_after_merge(rules, pred);

                // Full-relation stats trace entry, gated on `recursive-stats-trace`.
                #[cfg(feature = "recursive-stats-trace")]
                self.last_recursive_stats_trace
                    .entries
                    .push(super::RecursiveStatsTraceEntry {
                        iteration: iteration_count,
                        pred: pred.clone(),
                        full_rel: full_rel_opt.unwrap_or(RelId(u32::MAX)),
                        delta_rel,
                        full_rows: full_new_rows_phase4,
                        delta_rows: delta_rows_phase4,
                        phase: super::RecursiveStatsPhase::Phase4Full,
                        binary_est_for_variant: None,
                    });
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
        // Compute base case R = eval(base)
        let r_initial = self.execute_node(base)?;

        // Handle empty base case using device-resident row count
        if self.buffer_row_count(&r_initial)? == 0 {
            return Ok(r_initial);
        }

        // Initialize delta = R (clone the base result)
        let delta_initial = self.clone_buffer(&r_initial)?;

        // Get relation names for delta and full relations
        let delta_name = self.get_or_create_rel_name(delta_rel, &format!("__delta_{}", scc_id));
        let full_name = self.get_or_create_rel_name(full_rel, &format!("__full_{}", scc_id));

        // Store initial R and delta in relation store
        self.store_put(&full_name, r_initial);
        self.store_put(&delta_name, delta_initial);

        // Iterate until fixpoint
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
