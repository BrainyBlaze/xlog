//! Recursive SCC execution using semi-naive fixpoint iteration.

use std::collections::{BTreeSet, HashMap, HashSet};

use xlog_core::{RelId, Result, Schema, XlogError};
use xlog_cuda::CudaBuffer;
use xlog_ir::{ExecutionPlan, RirNode, Stratum};

use super::delta::DeltaRelationTracker;
use super::Executor;

impl Executor {
    /// Maximum iterations for fixpoint computation to prevent infinite loops
    const MAX_FIXPOINT_ITERATIONS: usize = 1000;

    /// v0.6.5 slice 4 helper. For a `MultiWayJoin` body (produced
    /// by the slice 1–2 promoter), try WCOJ dispatch via the
    /// triangle/4-cycle entry points; on decline, fall back to
    /// the embedded fallback subtree via `execute_node`. For any
    /// other RIR variant, defer to `execute_node` directly.
    ///
    /// Used at TWO sites in the recursive engine — the seeding
    /// pass (where stable rules with zero recursive Scans get
    /// their only chance to dispatch WCOJ) and the per-variant
    /// loop (where linear-recursive rules see one Scan rewritten
    /// to its delta RelId on each iteration). Multi-recursive
    /// bodies never reach a `MultiWayJoin` here because the slice
    /// 4 promoter gate skips them; the helper falls through to
    /// `execute_node` and the binary-join tree.
    ///
    /// Counter semantics: `wcoj_*_dispatch_count` increments per
    /// successful WCOJ kernel result — once per (rule, iteration,
    /// variant). Slice 1–3 non-recursive sites still increment
    /// once per rule per call.
    fn execute_wcoj_or_fallback_node(&mut self, node: &RirNode) -> Result<CudaBuffer> {
        if let RirNode::MultiWayJoin { .. } = node {
            // Triangle first, then 4-cycle. Slice 1 ordering — a
            // body cannot match both shapes (different atom
            // counts). The dispatcher's own gate handles env-var
            // / config / adaptive decisions; this site is purely
            // structural. The dispatcher increments
            // `wcoj_*_dispatch_count` internally on a successful
            // kernel result, so the helper just returns the
            // buffer and lets the caller fold it into the rule's
            // output.
            if let Some(buf) = self.try_dispatch_wcoj_triangle_on_body(node)? {
                return Ok(buf);
            }
            if let Some(buf) = self.try_dispatch_wcoj_4cycle_on_body(node)? {
                return Ok(buf);
            }
            // W3.2 plan §177: the recursive WCOJ helper is NOT
            // extended for clique-keyed dispatch. Recursive
            // clique bodies are rejected at the promoter level
            // (`promote_multiway` gates clique promotion on
            // `recursive_scan_count == 0`), so they fall through
            // to the binary-join path here. No `try_dispatch_wcoj_clique*`
            // call site in this helper.
        }
        self.execute_node(node)
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
                    // Recursive SCC: use semi-naive fixpoint iteration.
                    // v0.6.5 slice 4: the recursive engine now invokes
                    // WCOJ dispatch via `execute_wcoj_or_fallback_node`
                    // on both the seeding pass and per-variant
                    // evaluation, gated by the slice 4 promoter
                    // (recursive-Scan count ≤ 1).
                    self.execute_recursive_scc(rules)?;
                } else {
                    // Non-recursive SCC: execute rules once, union results for same predicate.
                    for rule in rules {
                        // v0.6.2 WCOJ triangle dispatch — env-gated.
                        // Try to short-circuit the rule via the GPU
                        // 3-way kernel. On Some(_), install the
                        // result and skip the binary-join path for
                        // this rule. On None (gate off, shape
                        // mismatch, missing input, kernel error),
                        // fall through silently. See
                        // `wcoj_dispatch::try_dispatch_wcoj_triangle`
                        // for the full match contract.
                        if let Some(wcoj_result) = self.try_dispatch_wcoj_triangle(rule)? {
                            // Mirrors the binary-join arm below:
                            // union with existing result if predicate
                            // already has data; otherwise install
                            // directly. WCOJ output is already
                            // sorted+deduped, so the dedup pass on
                            // the else branch is unnecessary here.
                            if let Some(existing) = self.store.get(&rule.head) {
                                let merged = self.provider.union_gpu(existing, &wcoj_result)?;
                                self.store_put(&rule.head, merged);
                            } else {
                                self.store_put(&rule.head, wcoj_result);
                            }
                            continue;
                        }

                        // v0.6.5 slice 2: WCOJ 4-cycle dispatch.
                        // Same pattern as triangle. Order is a doc
                        // anchor — a body cannot match both shapes
                        // (different atom counts), so triangle's
                        // earlier attempt always returns None on a
                        // 4-cycle body and vice versa.
                        if let Some(wcoj_result) = self.try_dispatch_wcoj_4cycle(rule)? {
                            if let Some(existing) = self.store.get(&rule.head) {
                                let merged = self.provider.union_gpu(existing, &wcoj_result)?;
                                self.store_put(&rule.head, merged);
                            } else {
                                self.store_put(&rule.head, wcoj_result);
                            }
                            continue;
                        }

                        // W3.2 — k=5 / k=6 clique dispatch.
                        // Same shape-gated default-dispatch
                        // pattern as triangle / 4-cycle; silent
                        // fallback to MultiWayJoin.fallback on
                        // dispatcher decline or kernel error.
                        if let Some(wcoj_result) = self.try_dispatch_wcoj_clique5(rule)? {
                            if let Some(existing) = self.store.get(&rule.head) {
                                let merged = self.provider.union_gpu(existing, &wcoj_result)?;
                                self.store_put(&rule.head, merged);
                            } else {
                                self.store_put(&rule.head, wcoj_result);
                            }
                            continue;
                        }
                        if let Some(wcoj_result) = self.try_dispatch_wcoj_clique6(rule)? {
                            if let Some(existing) = self.store.get(&rule.head) {
                                let merged = self.provider.union_gpu(existing, &wcoj_result)?;
                                self.store_put(&rule.head, merged);
                            } else {
                                self.store_put(&rule.head, wcoj_result);
                            }
                            continue;
                        }

                        // v0.6.5 slice 1: when WCOJ dispatch declines on
                        // a `MultiWayJoin` body (gate off, kernel error,
                        // adaptive score below threshold, …), execute
                        // the embedded `fallback` — the post-optimizer
                        // binary-join tree the promoter captured. This
                        // preserves byte-identical behavior with v0.6.2.
                        // `execute_node`'s `MultiWayJoin` arm is the
                        // defensive safety net; explicit destructuring
                        // here keeps the intent visible at the dispatch
                        // site.
                        let body_to_execute = match &rule.body {
                            xlog_ir::RirNode::MultiWayJoin { fallback, .. } => fallback.as_ref(),
                            other => other,
                        };
                        let result = self.execute_node(body_to_execute)?;

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
        // W2.3: reset the per-iteration stats trace at SCC entry so
        // tests see a fresh trace per invocation. Gated on the
        // `recursive-stats-trace` feature; default OFF.
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

        // Step 1: Execute all rules once against the current store to seed initial results.
        // Accumulate per-head before mutating the store to avoid order dependence.
        //
        // v0.6.5 slice 4: route through `execute_wcoj_or_fallback_node`
        // so MultiWayJoin bodies (slice 4 promoter output for stable
        // and linear-recursive triangles / 4-cycles) get a chance at
        // WCOJ dispatch on the seeding pass. Stable rules — bodies
        // with zero recursive Scans — only run here, so without this
        // hook they'd never see a kernel.
        let mut derived_initial: HashMap<String, CudaBuffer> = HashMap::new();
        for rule in rules {
            let result = self.execute_wcoj_or_fallback_node(&rule.body)?;
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
            let full_new = if self.buffer_row_count(&merged)? == 0 {
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

            let delta_name = delta_tracker.delta_name(pred)?;

            let full_old_rows = self.buffer_row_count(&full_old)?;
            let full_new_rows = self.buffer_row_count(&full_new)?;
            let delta_initial = if full_new_rows == 0 {
                self.create_empty_buffer(full_new.schema().clone())?
            } else if full_old_rows == 0 {
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

            // W2.3 step 4 — seed-iteration cardinality refresh.
            // Capture the actual delta_initial row count BEFORE the
            // `store_put` move (after the move, the buffer is gone).
            // `full_new_rows` was captured at line 356 above.
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

            // W2.3 trace seam — gated on `recursive-stats-trace`.
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

                let mut rule_delta_raw: Option<CudaBuffer> = None;
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

                    // v0.6.5 slice 4: try WCOJ on the rewritten variant
                    // body before falling back to the binary-join walker.
                    // For a linear-recursive triangle/4-cycle, the
                    // variant has one Scan's RelId swapped to its
                    // delta — the kernel reads from the delta store
                    // entry transparently, no special-case dispatch
                    // logic needed.
                    let out = self.execute_wcoj_or_fallback_node(&variant_node)?;
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
            delta_tracker.begin_iteration();

            for pred in &recursive_preds {
                let full = self
                    .store
                    .get(pred)
                    .ok_or_else(|| XlogError::Execution(format!("Missing relation: {}", pred)))?;
                // W2.3 step 5: capture the pre-Phase-4 full row count
                // for the trace's full_rows field at this Phase 2 site.
                // Gated on `recursive-stats-trace` so production builds
                // don't compute it.
                #[cfg(feature = "recursive-stats-trace")]
                let pre_phase4_full_rows = self.buffer_row_count(full)? as u64;

                let delta_raw = delta_new_raw_by_head.remove(pred);
                let delta_new = if let Some(delta_raw) = delta_raw {
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

                // W2.3 step 5 — Phase 2: refresh delta_rel card.
                // full_rel card is NOT updated here (full hasn't
                // changed yet this iteration; Phase 4 owns that).
                self.stats.update_cardinality(delta_rel, delta_new_rows);

                // W2.3 trace seam — gated on `recursive-stats-trace`.
                // binary_est_for_variant captures the cost model's
                // first-binary-hop estimate for the slice-4
                // linear-recursive fixtures (`pred == "e1"` rewrites
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
                let full_old = self
                    .store
                    .remove(pred)
                    .ok_or_else(|| XlogError::Execution(format!("Missing relation: {}", pred)))?;
                let dn = delta_tracker.delta_name(pred)?.to_string();
                let delta = self
                    .store_remove(&dn)
                    .ok_or_else(|| XlogError::Execution(format!("Missing relation: {}", dn)))?;

                if self.buffer_row_count(&delta)? == 0 {
                    // W2.3: zero-delta short-circuit — full and delta
                    // unchanged this iteration. Phase 2's delta_rel
                    // record (with rows == 0) stands; full_rel record
                    // from a prior iteration's Phase 4 stands. No
                    // additional update.
                    self.store_put(pred, full_old);
                    self.store_put(&dn, delta);
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
                let full_new = if self.buffer_row_count(&merged)? == 0 {
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
                // W2.3 step 6 — Phase 4: capture full_new's row count
                // BEFORE the store_put move; pre-resolve full_rel_opt
                // before the &mut self stats borrow. delta_rows_phase4
                // and delta_rel are only used by the trace under the
                // `recursive-stats-trace` feature.
                let full_new_rows_phase4 = self.buffer_row_count(&full_new)? as u64;
                #[cfg(feature = "recursive-stats-trace")]
                let delta_rows_phase4 = self.buffer_row_count(&delta)? as u64;
                let full_rel_opt = self.name_to_rel_id(pred);
                #[cfg(feature = "recursive-stats-trace")]
                let delta_rel = delta_tracker.delta_rel_id(pred)?;
                self.store_put(pred, full_new);
                self.store_put(&dn, delta);

                // Record full_rel's new card. (Phase 2 already
                // recorded delta_rel for this iteration.)
                if let Some(full_rel) = full_rel_opt {
                    self.stats
                        .update_cardinality(full_rel, full_new_rows_phase4);
                }

                // W2.3 trace seam — gated on `recursive-stats-trace`.
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
