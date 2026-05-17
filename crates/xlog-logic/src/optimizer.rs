//! Query optimizer for join ordering and predicate pushdown.
//!
//! This module provides cost-based query optimization for XLOG's relational IR.
//! It uses GPU-resident statistics from [`xlog_stats::StatsManager`] to make
//! informed decisions about:
//!
//! - **Predicate pushdown**: Moving filter predicates closer to base scans to
//!   reduce intermediate result sizes early in the pipeline.
//! - **Cost estimation**: Computing expected row counts, CPU costs, GPU memory
//!   usage, and data transfer counts for plan nodes.
//! - **Join ordering**: (Future) Reordering joins based on selectivity estimates
//!   to minimize intermediate result sizes.
//!
//! # Usage
//!
//! ```ignore
//! use std::sync::Arc;
//! use xlog_logic::optimizer::{Optimizer, OptimizerConfig, PlanCost};
//! use xlog_stats::StatsManager;
//!
//! let stats = Arc::new(StatsManager::new());
//! let optimizer = Optimizer::new(stats);
//!
//! // Optimize a query plan
//! let optimized_plan = optimizer.optimize(original_plan);
//!
//! // Get cost estimates
//! let cost = optimizer.estimate_cost(&optimized_plan);
//! println!("Estimated rows: {}, GPU memory: {} bytes", cost.rows, cost.gpu_mem);
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use xlog_core::{RelId, Schema};
use xlog_ir::{CompareOp, Expr, JoinType, RirNode};
use xlog_stats::StatsManager;

/// Configuration for query optimization.
///
/// Controls optimizer behavior including thresholds for algorithm selection
/// and feature toggles.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OptimizerConfig {
    /// Maximum number of relations for exhaustive dynamic programming.
    ///
    /// When a query involves more relations than this threshold, the optimizer
    /// switches to a greedy algorithm for join ordering to avoid exponential
    /// time complexity. Default: 10 relations.
    pub dp_threshold: usize,

    /// Heat threshold for recommending index creation.
    ///
    /// Relations with access heat above this threshold are candidates for
    /// index building to accelerate future queries. Default: 0.7.
    pub index_heat_threshold: f32,

    /// Enable predicate pushdown optimization.
    ///
    /// When enabled, filter predicates are pushed down through projections
    /// and joins to be applied as early as possible. Default: true.
    pub enable_pushdown: bool,

    /// Default selectivity for filters when no statistics are available.
    ///
    /// Used as a fallback when column statistics cannot provide a better
    /// estimate. Default: 0.1 (10% selectivity).
    pub default_filter_selectivity: f64,

    /// Cost multiplier for GPU-to-host data transfers.
    ///
    /// Transfers are expensive operations; this multiplier reflects the
    /// relative cost compared to local GPU operations. Default: 100.0.
    pub transfer_cost_multiplier: f64,

    /// Bytes per row used for GPU memory estimation when schema is unknown.
    ///
    /// Default: 32 bytes (assumes 4 columns at 8 bytes each on average).
    pub default_bytes_per_row: u64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            dp_threshold: 10,
            index_heat_threshold: 0.7,
            enable_pushdown: true,
            default_filter_selectivity: 0.1,
            transfer_cost_multiplier: 100.0,
            default_bytes_per_row: 32,
        }
    }
}

/// Cost estimate for a query plan node.
///
/// Captures the multi-dimensional cost of executing a plan node, enabling
/// the optimizer to make informed decisions based on available resources.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlanCost {
    /// Estimated number of output rows.
    pub rows: u64,

    /// Estimated CPU cost (arbitrary units, relative comparisons only).
    ///
    /// This represents processing overhead that cannot be parallelized on
    /// the GPU, such as coordination, scheduling, and result materialization.
    pub cpu_cost: f64,

    /// Estimated GPU memory usage in bytes.
    ///
    /// Includes both input buffers and intermediate storage required for
    /// the operation.
    pub gpu_mem: u64,

    /// Number of GPU-to-host or host-to-GPU data transfers.
    ///
    /// Transfers are typically the most expensive operations in GPU computing
    /// and should be minimized.
    pub transfers: u32,
}

impl PlanCost {
    /// Creates a new cost estimate with specified row count.
    pub fn with_rows(rows: u64) -> Self {
        Self {
            rows,
            ..Default::default()
        }
    }

    /// Computes a scalar cost value for comparison purposes.
    ///
    /// The formula weights different cost components:
    /// - CPU cost is taken directly
    /// - GPU memory is scaled by 0.001 (1GB = 1M cost units)
    /// - Transfers are heavily weighted due to their high latency
    ///
    /// # Arguments
    ///
    /// * `transfer_weight` - Weight multiplier for transfer costs
    pub fn total_cost(&self, transfer_weight: f64) -> f64 {
        self.cpu_cost + (self.gpu_mem as f64 * 0.001) + (self.transfers as f64 * transfer_weight)
    }

    /// Combines two costs representing sequential operations.
    ///
    /// Row count comes from the second (later) operation; other costs are summed.
    pub fn then(self, other: PlanCost) -> PlanCost {
        PlanCost {
            rows: other.rows,
            cpu_cost: self.cpu_cost + other.cpu_cost,
            gpu_mem: self.gpu_mem.max(other.gpu_mem), // Peak memory usage
            transfers: self.transfers + other.transfers,
        }
    }
}

/// Query optimizer using statistics for cost-based decisions.
///
/// The optimizer transforms query plans to improve execution efficiency
/// by applying rewrites like predicate pushdown and using statistics to
/// estimate costs for different plan alternatives.
pub struct Optimizer {
    stats: Arc<StatsManager>,
    config: OptimizerConfig,
    /// Schemas for relations, keyed by RelId
    schemas: HashMap<RelId, Schema>,
}

impl Optimizer {
    /// Creates a new optimizer with default configuration.
    ///
    /// # Arguments
    ///
    /// * `stats` - Shared statistics manager for cardinality and selectivity estimates
    pub fn new(stats: Arc<StatsManager>) -> Self {
        Self {
            stats,
            config: OptimizerConfig::default(),
            schemas: HashMap::new(),
        }
    }

    /// Creates a new optimizer with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `stats` - Shared statistics manager
    /// * `config` - Custom optimizer configuration
    pub fn with_config(stats: Arc<StatsManager>, config: OptimizerConfig) -> Self {
        Self {
            stats,
            config,
            schemas: HashMap::new(),
        }
    }

    /// Sets the schemas for relations.
    ///
    /// This information is used by the optimizer to accurately determine
    /// column widths during predicate pushdown.
    pub fn set_schemas(&mut self, schemas: HashMap<RelId, Schema>) {
        self.schemas = schemas;
    }

    /// Returns a reference to the current configuration.
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// Returns a reference to the statistics manager.
    pub fn stats(&self) -> &Arc<StatsManager> {
        &self.stats
    }

    /// Optimizes an execution plan by applying transformation rules.
    ///
    /// Currently applies:
    /// - Predicate pushdown (if enabled)
    ///
    /// Future optimizations may include:
    /// - Join reordering based on cardinality estimates
    /// - Projection pushdown
    /// - Common subexpression elimination
    ///
    /// # Arguments
    ///
    /// * `node` - The plan to optimize
    ///
    /// # Returns
    ///
    /// An optimized plan that is semantically equivalent to the input
    pub fn optimize(&self, node: RirNode) -> RirNode {
        if self.config.enable_pushdown {
            self.predicate_pushdown(node)
        } else {
            node
        }
    }

    /// Pushes filter predicates closer to scan nodes.
    ///
    /// This transformation reduces intermediate result sizes by applying
    /// filters as early as possible in the query pipeline. The rules are:
    ///
    /// - Filters can be pushed through projections (with column remapping)
    /// - Filters can be pushed into one or both sides of a join if the
    ///   predicate references only columns from that side
    /// - Filters on join keys can inform join selectivity estimates
    ///
    /// # Arguments
    ///
    /// * `node` - The plan node to transform
    ///
    /// # Returns
    ///
    /// The transformed plan with predicates pushed down where beneficial
    fn predicate_pushdown(&self, node: RirNode) -> RirNode {
        match node {
            // Base case: scan nodes cannot be transformed further
            RirNode::Unit => RirNode::Unit,
            RirNode::Scan { rel } => RirNode::Scan { rel },

            // Filter on top of another node: try to push down
            RirNode::Filter { input, predicate } => {
                // First, recursively optimize the input
                let optimized_input = self.predicate_pushdown(*input);

                match optimized_input {
                    // Filter on Filter: merge predicates
                    RirNode::Filter {
                        input: inner_input,
                        predicate: inner_pred,
                    } => {
                        let merged = Expr::And(vec![inner_pred, predicate]);
                        RirNode::Filter {
                            input: inner_input,
                            predicate: merged,
                        }
                    }

                    // Filter on Project: push through if possible
                    RirNode::Project {
                        input: proj_input,
                        columns,
                    } => {
                        // Check if predicate only references pass-through columns
                        if let Some(remapped) =
                            self.remap_predicate_through_project(&predicate, &columns)
                        {
                            // Push the remapped predicate below the projection
                            RirNode::Project {
                                input: Box::new(RirNode::Filter {
                                    input: proj_input,
                                    predicate: remapped,
                                }),
                                columns,
                            }
                        } else {
                            // Cannot push: keep filter above
                            RirNode::Filter {
                                input: Box::new(RirNode::Project {
                                    input: proj_input,
                                    columns,
                                }),
                                predicate,
                            }
                        }
                    }

                    // Filter on Join: try to push to appropriate side
                    RirNode::Join {
                        left,
                        right,
                        left_keys,
                        right_keys,
                        join_type,
                    } => {
                        let left_width = self.estimate_width(&left);
                        let (left_preds, right_preds, remaining) =
                            self.split_predicate_for_join(&predicate, left_width);

                        // Apply pushed predicates to each side
                        let new_left = if !left_preds.is_empty() {
                            Box::new(RirNode::Filter {
                                input: left,
                                predicate: Self::conjoin(left_preds),
                            })
                        } else {
                            left
                        };

                        let new_right = if !right_preds.is_empty() {
                            Box::new(RirNode::Filter {
                                input: right,
                                predicate: Self::conjoin(right_preds),
                            })
                        } else {
                            right
                        };

                        let join_node = RirNode::Join {
                            left: new_left,
                            right: new_right,
                            left_keys,
                            right_keys,
                            join_type,
                        };

                        // Apply remaining predicates that couldn't be pushed
                        if !remaining.is_empty() {
                            RirNode::Filter {
                                input: Box::new(join_node),
                                predicate: Self::conjoin(remaining),
                            }
                        } else {
                            join_node
                        }
                    }

                    // Default: cannot push further
                    other => RirNode::Filter {
                        input: Box::new(other),
                        predicate,
                    },
                }
            }

            // Project: recursively optimize input
            RirNode::Project { input, columns } => RirNode::Project {
                input: Box::new(self.predicate_pushdown(*input)),
                columns,
            },

            // Join: recursively optimize both sides
            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
            } => RirNode::Join {
                left: Box::new(self.predicate_pushdown(*left)),
                right: Box::new(self.predicate_pushdown(*right)),
                left_keys,
                right_keys,
                join_type,
            },

            // GroupBy: recursively optimize input
            RirNode::GroupBy {
                input,
                key_cols,
                aggs,
            } => RirNode::GroupBy {
                input: Box::new(self.predicate_pushdown(*input)),
                key_cols,
                aggs,
            },

            // Union: recursively optimize all inputs
            RirNode::Union { inputs } => RirNode::Union {
                inputs: inputs
                    .into_iter()
                    .map(|i| self.predicate_pushdown(i))
                    .collect(),
            },

            // Distinct: recursively optimize input
            RirNode::Distinct { input, key_cols } => RirNode::Distinct {
                input: Box::new(self.predicate_pushdown(*input)),
                key_cols,
            },

            // Diff: recursively optimize both sides
            RirNode::Diff { left, right } => RirNode::Diff {
                left: Box::new(self.predicate_pushdown(*left)),
                right: Box::new(self.predicate_pushdown(*right)),
            },

            // Fixpoint: recursively optimize base and recursive parts
            RirNode::Fixpoint {
                scc_id,
                base,
                recursive,
                delta_rel,
                full_rel,
            } => RirNode::Fixpoint {
                scc_id,
                base: Box::new(self.predicate_pushdown(*base)),
                recursive: Box::new(self.predicate_pushdown(*recursive)),
                delta_rel,
                full_rel,
            },

            RirNode::TensorMaskedJoin { .. } => node, // Leaf-like: no pushdown

            // v0.6.5: `MultiWayJoin` is produced by `xlog-logic::promote`
            // *after* the optimizer runs, so this arm is unreachable in
            // production. Required for compile safety and as a no-op
            // fallback if the call order ever changes.
            RirNode::MultiWayJoin { .. } => node,
        }
    }

    /// Attempts to remap a predicate through a projection.
    ///
    /// Returns `Some(remapped_predicate)` if all column references in the
    /// predicate can be traced back through pass-through columns.
    /// Returns `None` if the predicate references computed columns.
    fn remap_predicate_through_project(
        &self,
        predicate: &Expr,
        columns: &[xlog_ir::ProjectExpr],
    ) -> Option<Expr> {
        // Build a mapping from output column index to input column index
        // Only for pass-through columns
        let mut output_to_input: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        for (out_idx, proj_expr) in columns.iter().enumerate() {
            if let xlog_ir::ProjectExpr::Column(in_idx) = proj_expr {
                output_to_input.insert(out_idx, *in_idx);
            }
        }

        self.remap_expr(predicate, &output_to_input)
    }

    /// Recursively remaps column references in an expression.
    fn remap_expr(
        &self,
        expr: &Expr,
        mapping: &std::collections::HashMap<usize, usize>,
    ) -> Option<Expr> {
        match expr {
            Expr::Column(idx) => mapping.get(idx).map(|&new_idx| Expr::Column(new_idx)),

            Expr::Const(val) => Some(Expr::Const(val.clone())),

            Expr::Compare { left, op, right } => {
                let new_left = self.remap_expr(left, mapping)?;
                let new_right = self.remap_expr(right, mapping)?;
                Some(Expr::Compare {
                    left: Box::new(new_left),
                    op: *op,
                    right: Box::new(new_right),
                })
            }

            Expr::And(exprs) => {
                let remapped: Option<Vec<_>> =
                    exprs.iter().map(|e| self.remap_expr(e, mapping)).collect();
                remapped.map(Expr::And)
            }

            Expr::Or(exprs) => {
                let remapped: Option<Vec<_>> =
                    exprs.iter().map(|e| self.remap_expr(e, mapping)).collect();
                remapped.map(Expr::Or)
            }

            Expr::Not(inner) => {
                let remapped = self.remap_expr(inner, mapping)?;
                Some(Expr::Not(Box::new(remapped)))
            }

            // Arithmetic operations
            Expr::Add(l, r) => {
                let new_l = self.remap_expr(l, mapping)?;
                let new_r = self.remap_expr(r, mapping)?;
                Some(Expr::Add(Box::new(new_l), Box::new(new_r)))
            }
            Expr::Sub(l, r) => {
                let new_l = self.remap_expr(l, mapping)?;
                let new_r = self.remap_expr(r, mapping)?;
                Some(Expr::Sub(Box::new(new_l), Box::new(new_r)))
            }
            Expr::Mul(l, r) => {
                let new_l = self.remap_expr(l, mapping)?;
                let new_r = self.remap_expr(r, mapping)?;
                Some(Expr::Mul(Box::new(new_l), Box::new(new_r)))
            }
            Expr::Div(l, r) => {
                let new_l = self.remap_expr(l, mapping)?;
                let new_r = self.remap_expr(r, mapping)?;
                Some(Expr::Div(Box::new(new_l), Box::new(new_r)))
            }
            Expr::Mod(l, r) => {
                let new_l = self.remap_expr(l, mapping)?;
                let new_r = self.remap_expr(r, mapping)?;
                Some(Expr::Mod(Box::new(new_l), Box::new(new_r)))
            }

            // Built-in functions
            Expr::Abs(inner) => {
                let remapped = self.remap_expr(inner, mapping)?;
                Some(Expr::Abs(Box::new(remapped)))
            }
            Expr::Min(l, r) => {
                let new_l = self.remap_expr(l, mapping)?;
                let new_r = self.remap_expr(r, mapping)?;
                Some(Expr::Min(Box::new(new_l), Box::new(new_r)))
            }
            Expr::Max(l, r) => {
                let new_l = self.remap_expr(l, mapping)?;
                let new_r = self.remap_expr(r, mapping)?;
                Some(Expr::Max(Box::new(new_l), Box::new(new_r)))
            }
            Expr::Pow(l, r) => {
                let new_l = self.remap_expr(l, mapping)?;
                let new_r = self.remap_expr(r, mapping)?;
                Some(Expr::Pow(Box::new(new_l), Box::new(new_r)))
            }
            Expr::Cast(inner, scalar_type) => {
                let remapped = self.remap_expr(inner, mapping)?;
                Some(Expr::Cast(Box::new(remapped), *scalar_type))
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let new_condition = self.remap_expr(condition, mapping)?;
                let new_then = self.remap_expr(then_expr, mapping)?;
                let new_else = self.remap_expr(else_expr, mapping)?;
                Some(Expr::Conditional {
                    condition: Box::new(new_condition),
                    then_expr: Box::new(new_then),
                    else_expr: Box::new(new_else),
                })
            }
        }
    }

    /// Estimates the output width (number of columns) of a plan node.
    fn estimate_width(&self, node: &RirNode) -> usize {
        match node {
            RirNode::Unit => 0,
            RirNode::Scan { rel } => {
                // Use schema if available, otherwise stats, otherwise default
                if let Some(schema) = self.schemas.get(rel) {
                    schema.arity()
                } else if let Some(stats) = self.stats.get_relation_stats(*rel) {
                    stats.column_stats.len().max(1)
                } else {
                    4 // Default assumption
                }
            }
            RirNode::Filter { input, .. } => self.estimate_width(input),
            RirNode::Project { columns, .. } => columns.len(),
            RirNode::Join { left, right, .. } => {
                self.estimate_width(left) + self.estimate_width(right)
            }
            RirNode::GroupBy { key_cols, aggs, .. } => key_cols.len() + aggs.len(),
            RirNode::Union { inputs } => {
                inputs.first().map(|i| self.estimate_width(i)).unwrap_or(0)
            }
            RirNode::Distinct { input, .. } => self.estimate_width(input),
            RirNode::Diff { left, .. } => self.estimate_width(left),
            RirNode::Fixpoint { base, .. } => self.estimate_width(base),
            // RD-27: Optimizer schemas are HashMap<RelId, Schema>.
            // Use head_rel_id (not head_rel_name) for lookup.
            RirNode::TensorMaskedJoin { head_rel_id, .. } => self
                .schemas
                .get(head_rel_id)
                .map(|s| s.arity())
                .unwrap_or(2),
            // v0.6.5: `MultiWayJoin` post-promoter only — width equals
            // the head projection arity, mirroring the Project arm.
            RirNode::MultiWayJoin { output_columns, .. } => output_columns.len(),
        }
    }

    /// Splits a predicate into parts pushable to left, right, or neither side of a join.
    ///
    /// Returns (left_predicates, right_predicates, remaining_predicates).
    fn split_predicate_for_join(
        &self,
        predicate: &Expr,
        left_width: usize,
    ) -> (Vec<Expr>, Vec<Expr>, Vec<Expr>) {
        let mut left_preds = Vec::new();
        let mut right_preds = Vec::new();
        let mut remaining = Vec::new();

        // Flatten AND expressions
        let conjuncts = Self::flatten_and(predicate);

        for conj in conjuncts {
            let cols = Self::collect_columns(&conj);
            let max_col = cols.iter().copied().max().unwrap_or(0);
            let min_col = cols.iter().copied().min().unwrap_or(0);

            if cols.is_empty() {
                // No columns referenced, can push to either side
                left_preds.push(conj);
            } else if max_col < left_width {
                // All columns from left side
                left_preds.push(conj);
            } else if min_col >= left_width {
                // All columns from right side - need to remap
                let remapped = Self::remap_columns(&conj, |c| c - left_width);
                right_preds.push(remapped);
            } else {
                // References both sides, cannot push
                remaining.push(conj);
            }
        }

        (left_preds, right_preds, remaining)
    }

    /// Flattens nested AND expressions into a list of conjuncts.
    fn flatten_and(expr: &Expr) -> Vec<Expr> {
        match expr {
            Expr::And(exprs) => exprs.iter().flat_map(Self::flatten_and).collect(),
            other => vec![other.clone()],
        }
    }

    /// Collects all column indices referenced in an expression.
    fn collect_columns(expr: &Expr) -> Vec<usize> {
        match expr {
            Expr::Column(idx) => vec![*idx],
            Expr::Const(_) => vec![],
            Expr::Compare { left, right, .. } => {
                let mut cols = Self::collect_columns(left);
                cols.extend(Self::collect_columns(right));
                cols
            }
            Expr::And(exprs) | Expr::Or(exprs) => {
                exprs.iter().flat_map(Self::collect_columns).collect()
            }
            Expr::Not(inner) | Expr::Abs(inner) | Expr::Cast(inner, _) => {
                Self::collect_columns(inner)
            }
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Mod(l, r)
            | Expr::Min(l, r)
            | Expr::Max(l, r)
            | Expr::Pow(l, r) => {
                let mut cols = Self::collect_columns(l);
                cols.extend(Self::collect_columns(r));
                cols
            }
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let mut cols = Self::collect_columns(condition);
                cols.extend(Self::collect_columns(then_expr));
                cols.extend(Self::collect_columns(else_expr));
                cols
            }
        }
    }

    /// Remaps column references in an expression using a transformation function.
    fn remap_columns<F: Fn(usize) -> usize + Copy>(expr: &Expr, f: F) -> Expr {
        match expr {
            Expr::Column(idx) => Expr::Column(f(*idx)),
            Expr::Const(v) => Expr::Const(v.clone()),
            Expr::Compare { left, op, right } => Expr::Compare {
                left: Box::new(Self::remap_columns(left, f)),
                op: *op,
                right: Box::new(Self::remap_columns(right, f)),
            },
            Expr::And(exprs) => {
                Expr::And(exprs.iter().map(|e| Self::remap_columns(e, f)).collect())
            }
            Expr::Or(exprs) => Expr::Or(exprs.iter().map(|e| Self::remap_columns(e, f)).collect()),
            Expr::Not(inner) => Expr::Not(Box::new(Self::remap_columns(inner, f))),
            Expr::Add(l, r) => Expr::Add(
                Box::new(Self::remap_columns(l, f)),
                Box::new(Self::remap_columns(r, f)),
            ),
            Expr::Sub(l, r) => Expr::Sub(
                Box::new(Self::remap_columns(l, f)),
                Box::new(Self::remap_columns(r, f)),
            ),
            Expr::Mul(l, r) => Expr::Mul(
                Box::new(Self::remap_columns(l, f)),
                Box::new(Self::remap_columns(r, f)),
            ),
            Expr::Div(l, r) => Expr::Div(
                Box::new(Self::remap_columns(l, f)),
                Box::new(Self::remap_columns(r, f)),
            ),
            Expr::Mod(l, r) => Expr::Mod(
                Box::new(Self::remap_columns(l, f)),
                Box::new(Self::remap_columns(r, f)),
            ),
            Expr::Abs(inner) => Expr::Abs(Box::new(Self::remap_columns(inner, f))),
            Expr::Min(l, r) => Expr::Min(
                Box::new(Self::remap_columns(l, f)),
                Box::new(Self::remap_columns(r, f)),
            ),
            Expr::Max(l, r) => Expr::Max(
                Box::new(Self::remap_columns(l, f)),
                Box::new(Self::remap_columns(r, f)),
            ),
            Expr::Pow(l, r) => Expr::Pow(
                Box::new(Self::remap_columns(l, f)),
                Box::new(Self::remap_columns(r, f)),
            ),
            Expr::Cast(inner, t) => Expr::Cast(Box::new(Self::remap_columns(inner, f)), *t),
            Expr::Conditional {
                condition,
                then_expr,
                else_expr,
            } => Expr::Conditional {
                condition: Box::new(Self::remap_columns(condition, f)),
                then_expr: Box::new(Self::remap_columns(then_expr, f)),
                else_expr: Box::new(Self::remap_columns(else_expr, f)),
            },
        }
    }

    /// Combines a list of predicates into a single AND expression.
    fn conjoin(predicates: Vec<Expr>) -> Expr {
        debug_assert!(!predicates.is_empty());
        if predicates.len() == 1 {
            predicates.into_iter().next().unwrap()
        } else {
            Expr::And(predicates)
        }
    }

    /// Estimates the cost of executing a plan node.
    ///
    /// Recursively computes cost estimates for the entire plan tree,
    /// using statistics when available and falling back to heuristics.
    ///
    /// # Arguments
    ///
    /// * `node` - The plan node to estimate
    ///
    /// # Returns
    ///
    /// A [`PlanCost`] with estimated rows, CPU cost, GPU memory, and transfers
    pub fn estimate_cost(&self, node: &RirNode) -> PlanCost {
        match node {
            RirNode::Unit => PlanCost {
                rows: 1,
                cpu_cost: 0.0,
                gpu_mem: 0,
                transfers: 0,
            },
            RirNode::Scan { rel } => self.estimate_scan_cost(*rel),

            RirNode::Filter { input, predicate } => {
                let input_cost = self.estimate_cost(input);
                self.estimate_filter_cost(input_cost, predicate, input)
            }

            RirNode::Project { input, columns } => {
                let input_cost = self.estimate_cost(input);
                self.estimate_project_cost(input_cost, columns)
            }

            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
            } => {
                let left_cost = self.estimate_cost(left);
                let right_cost = self.estimate_cost(right);
                self.estimate_join_cost(
                    left_cost, right_cost, left, right, left_keys, right_keys, *join_type,
                )
            }

            RirNode::GroupBy {
                input,
                key_cols,
                aggs,
            } => {
                let input_cost = self.estimate_cost(input);
                self.estimate_groupby_cost(input_cost, key_cols, aggs)
            }

            RirNode::Union { inputs } => {
                let costs: Vec<_> = inputs.iter().map(|i| self.estimate_cost(i)).collect();
                self.estimate_union_cost(costs)
            }

            RirNode::Distinct { input, key_cols } => {
                let input_cost = self.estimate_cost(input);
                self.estimate_distinct_cost(input_cost, key_cols)
            }

            RirNode::Diff { left, right } => {
                let left_cost = self.estimate_cost(left);
                let right_cost = self.estimate_cost(right);
                self.estimate_diff_cost(left_cost, right_cost)
            }

            RirNode::Fixpoint {
                base, recursive, ..
            } => {
                let base_cost = self.estimate_cost(base);
                let recursive_cost = self.estimate_cost(recursive);
                self.estimate_fixpoint_cost(base_cost, recursive_cost)
            }

            RirNode::TensorMaskedJoin {
                max_active_rules, ..
            } => PlanCost {
                rows: *max_active_rules as u64,
                cpu_cost: *max_active_rules as f64 * 100.0,
                gpu_mem: *max_active_rules as u64 * 1024,
                transfers: 1,
            },
            // v0.6.5: `MultiWayJoin` cost is the sum of input scan costs.
            // Heuristic only — the post-promoter dispatch decides whether
            // to run the WCOJ kernel or fall back; cost-model integration
            // for the multiway operator itself is later-slice work.
            RirNode::MultiWayJoin { inputs, .. } => {
                let mut total = PlanCost::default();
                for inp in inputs {
                    let c = self.estimate_cost(inp);
                    total.rows = total.rows.saturating_add(c.rows);
                    total.cpu_cost += c.cpu_cost;
                    total.gpu_mem = total.gpu_mem.saturating_add(c.gpu_mem);
                    total.transfers = total.transfers.saturating_add(c.transfers);
                }
                total
            }
        }
    }

    /// Estimates cost for a base relation scan.
    fn estimate_scan_cost(&self, rel: RelId) -> PlanCost {
        if let Some(stats) = self.stats.get_relation_stats(rel) {
            PlanCost {
                rows: stats.cardinality,
                cpu_cost: stats.cardinality as f64 * 0.01, // Minimal per-row CPU cost
                gpu_mem: stats
                    .byte_size
                    .max(stats.cardinality * self.config.default_bytes_per_row),
                transfers: 0, // Data already on GPU
            }
        } else {
            // Default estimates for unknown relations
            let default_rows = 1000;
            PlanCost {
                rows: default_rows,
                cpu_cost: default_rows as f64 * 0.01,
                gpu_mem: default_rows * self.config.default_bytes_per_row,
                transfers: 0,
            }
        }
    }

    /// Estimates cost for a filter operation.
    fn estimate_filter_cost(
        &self,
        input_cost: PlanCost,
        predicate: &Expr,
        input: &RirNode,
    ) -> PlanCost {
        let selectivity = self.estimate_predicate_selectivity(predicate, input);
        let output_rows = ((input_cost.rows as f64 * selectivity) as u64).max(1);

        PlanCost {
            rows: output_rows,
            cpu_cost: input_cost.cpu_cost + input_cost.rows as f64 * 0.02, // Predicate eval cost
            gpu_mem: input_cost.gpu_mem, // Filter reuses input memory
            transfers: input_cost.transfers,
        }
    }

    /// Estimates cost for a projection operation.
    fn estimate_project_cost(
        &self,
        input_cost: PlanCost,
        columns: &[xlog_ir::ProjectExpr],
    ) -> PlanCost {
        // Count computed vs pass-through columns
        let computed_count = columns
            .iter()
            .filter(|c| matches!(c, xlog_ir::ProjectExpr::Computed(_, _)))
            .count();

        // Computed columns add CPU cost
        let compute_cost = computed_count as f64 * input_cost.rows as f64 * 0.05;

        // Output size may be smaller if fewer columns
        let output_width_ratio = columns.len() as f64 / (columns.len() + 2) as f64; // Rough estimate

        PlanCost {
            rows: input_cost.rows,
            cpu_cost: input_cost.cpu_cost + compute_cost,
            gpu_mem: (input_cost.gpu_mem as f64 * output_width_ratio) as u64,
            transfers: input_cost.transfers,
        }
    }

    /// Estimates cost for a join operation.
    #[allow(clippy::too_many_arguments)]
    fn estimate_join_cost(
        &self,
        left_cost: PlanCost,
        right_cost: PlanCost,
        left: &RirNode,
        right: &RirNode,
        left_keys: &[usize],
        right_keys: &[usize],
        join_type: JoinType,
    ) -> PlanCost {
        // Semi and Anti joins always produce at most left_cost.rows
        // Handle these specially before checking stats
        let output_rows = match join_type {
            JoinType::Semi => {
                // At most left side rows, estimate 50% match
                ((left_cost.rows as f64 * 0.5) as u64).max(1)
            }
            JoinType::Anti => {
                // At most left side rows, estimate 50% don't match
                ((left_cost.rows as f64 * 0.5) as u64).max(1)
            }
            JoinType::Inner | JoinType::LeftOuter => {
                // Get relation IDs for selectivity lookup
                let left_rels = left.referenced_relations();
                let right_rels = right.referenced_relations();

                if left_rels.len() == 1 && right_rels.len() == 1 {
                    // Simple join between two base relations
                    let estimated = self.stats.estimate_join_cardinality(
                        left_rels[0],
                        right_rels[0],
                        left_keys,
                        right_keys,
                    );

                    match join_type {
                        JoinType::LeftOuter => estimated.max(left_cost.rows),
                        _ => estimated,
                    }
                } else {
                    // Multi-way or complex join: use heuristic
                    match join_type {
                        JoinType::Inner => {
                            // Assume 10% selectivity for inner joins
                            ((left_cost.rows as f64 * right_cost.rows as f64 * 0.1) as u64).max(1)
                        }
                        JoinType::LeftOuter => {
                            // At least left side rows
                            left_cost.rows.max(
                                ((left_cost.rows as f64 * right_cost.rows as f64 * 0.1) as u64)
                                    .max(1),
                            )
                        }
                        _ => unreachable!(),
                    }
                }
            }
        };

        // Join CPU cost: hash build + probe
        let build_cost = right_cost.rows as f64 * 1.0; // Build hash table
        let probe_cost = left_cost.rows as f64 * 0.5; // Probe operations
        let cpu_cost = left_cost.cpu_cost + right_cost.cpu_cost + build_cost + probe_cost;

        // GPU memory: both inputs plus hash table overhead
        let hash_table_overhead = right_cost.gpu_mem / 2; // Approximate hash table size
        let gpu_mem = left_cost.gpu_mem + right_cost.gpu_mem + hash_table_overhead;

        PlanCost {
            rows: output_rows,
            cpu_cost,
            gpu_mem,
            transfers: left_cost.transfers + right_cost.transfers,
        }
    }

    /// Estimates cost for a group-by with aggregation.
    fn estimate_groupby_cost(
        &self,
        input_cost: PlanCost,
        key_cols: &[usize],
        _aggs: &[(usize, xlog_core::AggOp)],
    ) -> PlanCost {
        // Estimate distinct groups based on key columns
        // Heuristic: sqrt(input_rows) for unknown cardinality
        let estimated_groups = if key_cols.is_empty() {
            1 // No grouping = single result
        } else {
            // Rough estimate: assume good reduction
            ((input_cost.rows as f64).sqrt() as u64).max(1)
        };

        PlanCost {
            rows: estimated_groups,
            cpu_cost: input_cost.cpu_cost + input_cost.rows as f64 * 0.5, // Aggregation cost
            gpu_mem: input_cost.gpu_mem + estimated_groups * self.config.default_bytes_per_row,
            transfers: input_cost.transfers,
        }
    }

    /// Estimates cost for a union operation.
    fn estimate_union_cost(&self, input_costs: Vec<PlanCost>) -> PlanCost {
        let total_rows: u64 = input_costs.iter().map(|c| c.rows).sum();
        let total_cpu: f64 = input_costs.iter().map(|c| c.cpu_cost).sum();
        let max_gpu: u64 = input_costs.iter().map(|c| c.gpu_mem).max().unwrap_or(0);
        let total_transfers: u32 = input_costs.iter().map(|c| c.transfers).sum();

        PlanCost {
            rows: total_rows,
            cpu_cost: total_cpu + total_rows as f64 * 0.01, // Concatenation cost
            gpu_mem: max_gpu,                               // Can process sequentially
            transfers: total_transfers,
        }
    }

    /// Estimates cost for a distinct operation.
    fn estimate_distinct_cost(&self, input_cost: PlanCost, _key_cols: &[usize]) -> PlanCost {
        // Heuristic: distinct reduces rows by some factor
        let estimated_distinct = (input_cost.rows as f64 * 0.7) as u64;

        PlanCost {
            rows: estimated_distinct.max(1),
            cpu_cost: input_cost.cpu_cost + input_cost.rows as f64 * 0.3, // Hash-based dedup
            gpu_mem: input_cost.gpu_mem + input_cost.rows * 8,            // Hash set overhead
            transfers: input_cost.transfers,
        }
    }

    /// Estimates cost for a set difference operation.
    fn estimate_diff_cost(&self, left_cost: PlanCost, right_cost: PlanCost) -> PlanCost {
        // Diff removes matching rows from left
        let estimated_remaining = (left_cost.rows as f64 * 0.5) as u64;

        PlanCost {
            rows: estimated_remaining.max(1),
            cpu_cost: left_cost.cpu_cost + right_cost.cpu_cost + right_cost.rows as f64 * 0.5,
            gpu_mem: left_cost.gpu_mem + right_cost.gpu_mem,
            transfers: left_cost.transfers + right_cost.transfers,
        }
    }

    /// Estimates cost for a fixpoint (recursive) operation.
    fn estimate_fixpoint_cost(&self, base_cost: PlanCost, recursive_cost: PlanCost) -> PlanCost {
        // Fixpoint cost depends on number of iterations
        // Heuristic: assume log2(base_rows) iterations
        let estimated_iterations = ((base_cost.rows as f64).log2().ceil() as u64).max(1);

        PlanCost {
            rows: base_cost.rows * estimated_iterations, // Output accumulates
            cpu_cost: base_cost.cpu_cost + recursive_cost.cpu_cost * estimated_iterations as f64,
            gpu_mem: (base_cost.gpu_mem + recursive_cost.gpu_mem) * 2, // Need delta and full
            transfers: base_cost.transfers + recursive_cost.transfers * estimated_iterations as u32,
        }
    }

    /// Estimates selectivity of a predicate expression.
    fn estimate_predicate_selectivity(&self, predicate: &Expr, input: &RirNode) -> f64 {
        match predicate {
            Expr::Compare { left, op, right } => {
                self.estimate_compare_selectivity(left, *op, right, input)
            }
            Expr::And(exprs) => {
                // Multiply selectivities (independence assumption)
                exprs
                    .iter()
                    .map(|e| self.estimate_predicate_selectivity(e, input))
                    .product()
            }
            Expr::Or(exprs) => {
                // P(A or B) = P(A) + P(B) - P(A)P(B) for independent events
                // Simplified: max of selectivities as lower bound
                exprs
                    .iter()
                    .map(|e| self.estimate_predicate_selectivity(e, input))
                    .fold(0.0, f64::max)
            }
            Expr::Not(inner) => 1.0 - self.estimate_predicate_selectivity(inner, input),
            _ => self.config.default_filter_selectivity,
        }
    }

    /// Estimates selectivity for a comparison predicate.
    fn estimate_compare_selectivity(
        &self,
        left: &Expr,
        op: CompareOp,
        right: &Expr,
        input: &RirNode,
    ) -> f64 {
        // Try to get column statistics if comparing column to constant
        if let (Expr::Column(col_idx), Expr::Const(_)) | (Expr::Const(_), Expr::Column(col_idx)) =
            (left, right)
        {
            // Find the base relation for this column
            if let Some(rel_id) = self.find_column_relation(input, *col_idx) {
                if let Some(stats) = self.stats.get_relation_stats(rel_id) {
                    if let Some(col_stats) = stats.get_column(*col_idx) {
                        return match op {
                            CompareOp::Eq => col_stats.equality_selectivity(stats.cardinality),
                            CompareOp::Ne => {
                                1.0 - col_stats.equality_selectivity(stats.cardinality)
                            }
                            CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => {
                                // Range predicates: estimate ~33% selectivity
                                0.33
                            }
                        };
                    }
                }
            }
        }

        // Default selectivities by operator
        match op {
            CompareOp::Eq => 0.1, // 10% for equality
            CompareOp::Ne => 0.9, // 90% for inequality
            CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => 0.33, // 33% for ranges
        }
    }

    /// Finds the base relation that provides a given column.
    fn find_column_relation(&self, node: &RirNode, col_idx: usize) -> Option<RelId> {
        match node {
            RirNode::Scan { rel } => Some(*rel),
            RirNode::Filter { input, .. } => self.find_column_relation(input, col_idx),
            RirNode::Project { input, columns } => {
                // Trace column through projection
                if col_idx < columns.len() {
                    if let xlog_ir::ProjectExpr::Column(src_idx) = &columns[col_idx] {
                        return self.find_column_relation(input, *src_idx);
                    }
                }
                None
            }
            RirNode::Join { left, right, .. } => {
                let left_width = self.estimate_width(left);
                if col_idx < left_width {
                    self.find_column_relation(left, col_idx)
                } else {
                    self.find_column_relation(right, col_idx - left_width)
                }
            }
            // v0.6.5: per slice 1 guardrail — return None for
            // `MultiWayJoin`. The promoter runs after the optimizer,
            // so this arm is unreachable in production. A half-mapped
            // implementation that walked `inputs` via `slot_vars` would
            // be more dangerous than `None` for this slice.
            RirNode::MultiWayJoin { .. } => None,
            _ => None, // Complex cases: give up
        }
    }

    /// Returns relations that should have indexes built based on access heat.
    ///
    /// This is useful for adaptive query processing where frequently accessed
    /// relations benefit from index structures.
    pub fn recommend_indexes(&self) -> Vec<RelId> {
        self.stats.hot_relations(self.config.index_heat_threshold)
    }

    /// Returns true if the query involves more relations than the DP threshold.
    ///
    /// Used to decide between exhaustive and greedy join ordering algorithms.
    pub fn should_use_greedy(&self, node: &RirNode) -> bool {
        let rels = node.referenced_relations();
        let unique_rels: std::collections::HashSet<_> = rels.iter().collect();
        unique_rels.len() > self.config.dp_threshold
    }
}

/// v0.6.5 slice 3 — selectivity-aware optimizer pass.
///
/// **No-op by default.** Slice 3 lays the seam; slices 4 / 5 may
/// add real reordering logic that consults `stats` to pick join
/// orderings on selectivity.
///
/// Walks `plan.rules_by_scc[*].body` and rewrites nodes in place.
/// The default no-op preserves every existing plan tree
/// byte-for-byte. Tests assert structural equality (Debug-format
/// snapshot before/after) for triangle, 4-cycle, and recursive-
/// SCC plans.
///
/// **Compile-pipeline ordering** (locked by slice 3): runs
/// between `Optimizer::optimize` and `xlog_logic::promote::promote_multiway`.
/// The slice 1 invariant — promoter sees the post-optimizer tree —
/// is preserved.
pub mod selectivity_pass {
    //! v0.6.5 W2.2 — selectivity-driven join reordering for
    //! canonical lowered triangle and 4-cycle bodies.
    //!
    //! ## Behavior
    //!
    //! For each rule body that matches the canonical lowered
    //! triangle or 4-cycle shape, the pass enumerates the valid
    //! candidate inner pairings (3 for triangle, 2 for 4-cycle),
    //! computes each candidate's
    //! `StatsManager::estimate_join_cardinality` with
    //! **pair-derived join keys from the shared-variable
    //! mapping**, and rewrites the body so the smallest-cost
    //! choice is materialized first. Tie → keep the optimizer's
    //! existing order (deterministic no-op).
    //!
    //! ## Safety floor
    //!
    //! If any input atom for a recognized body has no
    //! `StatsManager` entry OR `cardinality == 0`, the body is
    //! left unchanged. Recursive deltas / freshly-uploaded
    //! relations / unseeded predicates therefore stay on the
    //! optimizer's default order until stats are populated.
    //!
    //! ## Default-fallback edge case
    //!
    //! `StatsManager::estimate_join_cardinality` returns `u64`
    //! with no provenance — the caller cannot tell whether the
    //! estimate came from the cached `JoinSelectivity` table,
    //! the column-distinct heuristic, or the 10% default
    //! fallback. When all input atoms have populated
    //! cardinalities but no column statistics, the per-pair
    //! estimates may all collapse to the same fallback ratio,
    //! making the chosen pairing uninformative. **This is an
    //! accepted trade-off**: row-set parity holds regardless of
    //! selectivity quality (the rewrite preserves semantics);
    //! the integration certs gate on row-set + WCOJ-dispatch
    //! correctness, not on optimal pair choice.
    //!
    //! ## Promoter coordination
    //!
    //! The slice 1 / slice 2 promoters were extended in W2.2
    //! step 2a to accept the canonical *semantic* shape with
    //! any valid key combination — they emit
    //! `MultiWayJoin.inputs` and `slot_vars` in canonical
    //! semantic order regardless of the body's positional
    //! layout. Reordered bodies therefore still promote and
    //! still dispatch the WCOJ kernel correctly.
    use std::collections::HashMap;
    use xlog_core::RelId;
    use xlog_ir::ExecutionPlan;
    use xlog_stats::StatsManager;

    /// W2.2: selectivity-driven join reordering for canonical
    /// triangle + 4-cycle bodies. See module-level doc.
    ///
    /// `rel_ids` is the predicate-name → RelId map used to
    /// resolve body Scans against `StatsManager` lookups.
    /// Production callers pass `Compiler::lowerer().rel_ids()`.
    /// Test callers can pass an empty map; with no
    /// `StatsManager` entries either, the safety floor leaves
    /// every body unchanged (legacy no-op behavior preserved).
    pub fn run(plan: &mut ExecutionPlan, stats: &StatsManager, rel_ids: &HashMap<String, RelId>) {
        // `rel_ids` is reserved for future shape-extension
        // work; the current rewriters operate on RelIds
        // directly from the body's Scans, so the map isn't
        // consulted here. Production callers still pass it
        // so the API surface is forward-compatible.
        let _ = rel_ids;
        for rules in plan.rules_by_scc.iter_mut() {
            for rule in rules.iter_mut() {
                if let Some(rewritten) = super::reorder::try_reorder_triangle(&rule.body, stats) {
                    rule.body = rewritten;
                    continue;
                }
                if let Some(rewritten) = super::reorder::try_reorder_4cycle(&rule.body, stats) {
                    rule.body = rewritten;
                }
            }
        }
    }
}

/// W3.7 AOT helper-relation splitting for deep joins with buried skew.
pub mod helper_split_pass {
    use std::collections::{HashMap, HashSet};

    use xlog_core::{RelId, ScalarType, Schema};
    use xlog_ir::{CompiledRule, ExecutionPlan, JoinType, ProjectExpr, RirMeta, RirNode, Scc};
    use xlog_stats::StatsManager;

    const HEAVY_SKEW_RATIO: f64 = 10.0;

    /// Description of a helper relation introduced by the pass.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct HelperRelationSpec {
        /// Predicate name allocated for the helper relation.
        pub name: String,
        /// Relation identifier allocated for the helper relation.
        pub rel_id: RelId,
        /// Output schema of the helper relation.
        pub schema: Schema,
        /// Pair of source relations extracted into the helper body.
        pub source_rels: [RelId; 2],
    }

    struct JoinStep {
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
    }

    struct LinearBody {
        leaves: Vec<RelId>,
        leaf_classes: Vec<Vec<u32>>,
        joins: Vec<JoinStep>,
        project: Vec<ProjectExpr>,
        final_classes: Vec<u32>,
    }

    struct FlatJoin {
        leaves: Vec<RelId>,
        output_cols: Vec<usize>,
        equalities: Vec<(usize, usize)>,
    }

    struct Candidate {
        pair_start: usize,
        helper_schema: Schema,
        helper_project: Vec<ProjectExpr>,
        helper_join_left_keys: Vec<usize>,
        helper_join_right_keys: Vec<usize>,
        exposed_classes: Vec<u32>,
    }

    struct Rewrite {
        helper_body: RirNode,
        outer_body: RirNode,
        spec: HelperRelationSpec,
    }

    /// Rewrite eligible rules in-place and return the helper relations introduced.
    pub fn run<F>(
        plan: &mut ExecutionPlan,
        schemas: &HashMap<RelId, Schema>,
        stats: &StatsManager,
        mut allocate: F,
    ) -> Vec<HelperRelationSpec>
    where
        F: FnMut(Schema) -> (String, RelId),
    {
        let mut specs = Vec::new();
        for scc_idx in 0..plan.rules_by_scc.len() {
            let mut rule_idx = 0;
            while rule_idx < plan.rules_by_scc[scc_idx].len() {
                let rewrite = {
                    let rule = &plan.rules_by_scc[scc_idx][rule_idx];
                    try_rewrite_rule(rule, schemas, stats, &mut allocate)
                };
                if let Some(rewrite) = rewrite {
                    let helper_rule = CompiledRule {
                        head: rewrite.spec.name.clone(),
                        body: rewrite.helper_body,
                        meta: RirMeta::with_schema(rewrite.spec.schema.clone()),
                    };
                    plan.rules_by_scc[scc_idx].insert(rule_idx, helper_rule);
                    rule_idx += 1;
                    plan.rules_by_scc[scc_idx][rule_idx].body = rewrite.outer_body;
                    add_helper_to_scc(&mut plan.sccs, scc_idx, &rewrite.spec.name);
                    specs.push(rewrite.spec);
                }
                rule_idx += 1;
            }
        }
        specs
    }

    fn add_helper_to_scc(sccs: &mut [Scc], scc_idx: usize, helper: &str) {
        if let Some(scc) = sccs.get_mut(scc_idx) {
            if !scc.predicates.iter().any(|p| p == helper) {
                scc.predicates.push(helper.to_string());
            }
        }
    }

    fn try_rewrite_rule<F>(
        rule: &CompiledRule,
        schemas: &HashMap<RelId, Schema>,
        stats: &StatsManager,
        allocate: &mut F,
    ) -> Option<Rewrite>
    where
        F: FnMut(Schema) -> (String, RelId),
    {
        let linear = linearize_project_body(&rule.body, schemas)?;
        let candidate = choose_candidate(&linear, schemas, stats)?;
        let (helper_name, helper_rel) = allocate(candidate.helper_schema.clone());
        let helper_body = build_helper_body(&linear, &candidate);
        let outer_body = build_outer_body(&linear, &candidate, helper_rel)?;
        Some(Rewrite {
            helper_body,
            outer_body,
            spec: HelperRelationSpec {
                name: helper_name,
                rel_id: helper_rel,
                schema: candidate.helper_schema,
                source_rels: [
                    linear.leaves[candidate.pair_start],
                    linear.leaves[candidate.pair_start + 1],
                ],
            },
        })
    }

    fn linearize_project_body(
        body: &RirNode,
        schemas: &HashMap<RelId, Schema>,
    ) -> Option<LinearBody> {
        let RirNode::Project { input, columns } = body else {
            return None;
        };
        let flat = collect_join_graph(input, schemas)?;
        if flat.leaves.len() < 6 {
            return None;
        }
        let mut offsets = Vec::with_capacity(flat.leaves.len());
        let mut total_cols = 0usize;
        for rel in &flat.leaves {
            offsets.push(total_cols);
            total_cols += schemas.get(rel)?.arity();
        }
        let mut uf = UnionFind::new(total_cols);
        for (left, right) in flat.equalities {
            if left >= total_cols || right >= total_cols {
                return None;
            }
            uf.union(left, right);
        }
        let mut leaf_classes: Vec<Vec<u32>> = Vec::with_capacity(flat.leaves.len());
        for (leaf_idx, rel) in flat.leaves.iter().enumerate() {
            let arity = schemas.get(rel)?.arity();
            let offset = offsets[leaf_idx];
            leaf_classes.push((0..arity).map(|col| uf.find(offset + col) as u32).collect());
        }
        let final_classes = flat
            .output_cols
            .iter()
            .map(|col| uf.find(*col) as u32)
            .collect();
        let joins = derive_left_deep_steps(&leaf_classes)?;
        Some(LinearBody {
            leaves: flat.leaves,
            leaf_classes,
            joins,
            project: columns.clone(),
            final_classes,
        })
    }

    fn collect_join_graph(node: &RirNode, schemas: &HashMap<RelId, Schema>) -> Option<FlatJoin> {
        match node {
            RirNode::Scan { rel } => Some(FlatJoin {
                leaves: vec![*rel],
                output_cols: (0..schemas.get(rel)?.arity()).collect(),
                equalities: Vec::new(),
            }),
            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
            } if *join_type == JoinType::Inner => {
                let left_flat = collect_join_graph(left, schemas)?;
                let right_flat = collect_join_graph(right, schemas)?;
                if left_keys.len() != right_keys.len() {
                    return None;
                }
                let right_shift = total_width(&left_flat.leaves, schemas)?;
                let mut leaves = left_flat.leaves;
                leaves.extend(right_flat.leaves);
                let right_output_cols: Vec<usize> = right_flat
                    .output_cols
                    .iter()
                    .map(|col| col + right_shift)
                    .collect();
                let mut equalities = left_flat.equalities;
                equalities.extend(
                    right_flat
                        .equalities
                        .iter()
                        .map(|(left, right)| (left + right_shift, right + right_shift)),
                );
                for (&left_key, &right_key) in left_keys.iter().zip(right_keys.iter()) {
                    equalities.push((
                        *left_flat.output_cols.get(left_key)?,
                        *right_output_cols.get(right_key)?,
                    ));
                }
                let mut output_cols = left_flat.output_cols;
                output_cols.extend(right_output_cols);
                Some(FlatJoin {
                    leaves,
                    output_cols,
                    equalities,
                })
            }
            _ => None,
        }
    }

    fn total_width(leaves: &[RelId], schemas: &HashMap<RelId, Schema>) -> Option<usize> {
        leaves
            .iter()
            .map(|rel| schemas.get(rel).map(Schema::arity))
            .try_fold(0usize, |acc, width| width.map(|width| acc + width))
    }

    fn derive_left_deep_steps(leaf_classes: &[Vec<u32>]) -> Option<Vec<JoinStep>> {
        let mut joins = Vec::with_capacity(leaf_classes.len().saturating_sub(1));
        let mut current = leaf_classes.first()?.clone();
        for classes in leaf_classes.iter().skip(1) {
            let mut left_keys = Vec::new();
            let mut right_keys = Vec::new();
            for (right_col, class) in classes.iter().enumerate() {
                if let Some(left_col) = current
                    .iter()
                    .position(|current_class| current_class == class)
                {
                    left_keys.push(left_col);
                    right_keys.push(right_col);
                }
            }
            if left_keys.is_empty() {
                return None;
            }
            joins.push(JoinStep {
                left_keys,
                right_keys,
            });
            current.extend(classes.iter().copied());
        }
        Some(joins)
    }

    fn choose_candidate(
        linear: &LinearBody,
        schemas: &HashMap<RelId, Schema>,
        stats: &StatsManager,
    ) -> Option<Candidate> {
        for pair_start in 3..linear.leaves.len().saturating_sub(1) {
            let candidate = build_candidate(linear, schemas, pair_start)?;
            if skew_ratio_for_candidate(linear, stats, &candidate) >= HEAVY_SKEW_RATIO {
                return Some(candidate);
            }
        }
        None
    }

    fn build_candidate(
        linear: &LinearBody,
        schemas: &HashMap<RelId, Schema>,
        pair_start: usize,
    ) -> Option<Candidate> {
        let left_rel = linear.leaves[pair_start];
        let right_rel = linear.leaves[pair_start + 1];
        let left_schema = schemas.get(&left_rel)?;
        let right_schema = schemas.get(&right_rel)?;
        let internal_step = linear.joins.get(pair_start)?;
        let mut helper_left_keys = Vec::new();
        let mut helper_right_keys = Vec::new();
        for (&left_key, &right_key) in internal_step
            .left_keys
            .iter()
            .zip(internal_step.right_keys.iter())
        {
            let class = class_at_state(linear, pair_start + 1, left_key)?;
            let left_col = linear.leaf_classes[pair_start]
                .iter()
                .position(|c| *c == class)?;
            helper_left_keys.push(left_col);
            helper_right_keys.push(right_key);
        }
        let internal: HashSet<u32> = helper_left_keys
            .iter()
            .map(|col| linear.leaf_classes[pair_start][*col])
            .collect();
        let outside = outside_classes(linear, pair_start);
        let output = projected_classes(linear)?;
        let mut exposed_classes = Vec::new();
        let mut helper_project = Vec::new();
        let mut helper_columns = Vec::new();
        for (col, class) in linear.leaf_classes[pair_start].iter().copied().enumerate() {
            if !internal.contains(&class)
                && (outside.contains(&class) || output.contains(&class))
                && !exposed_classes.contains(&class)
            {
                exposed_classes.push(class);
                helper_project.push(ProjectExpr::Column(col));
                let ty = left_schema.column_type(col).unwrap_or(ScalarType::U32);
                helper_columns.push((format!("c{}", helper_columns.len()), ty));
            }
        }
        let right_offset = left_schema.arity();
        for (col, class) in linear.leaf_classes[pair_start + 1]
            .iter()
            .copied()
            .enumerate()
        {
            if !internal.contains(&class)
                && (outside.contains(&class) || output.contains(&class))
                && !exposed_classes.contains(&class)
            {
                exposed_classes.push(class);
                helper_project.push(ProjectExpr::Column(right_offset + col));
                let ty = right_schema.column_type(col).unwrap_or(ScalarType::U32);
                helper_columns.push((format!("c{}", helper_columns.len()), ty));
            }
        }
        if exposed_classes.len() != 2 {
            return None;
        }
        Some(Candidate {
            pair_start,
            helper_schema: Schema::new(helper_columns),
            helper_project,
            helper_join_left_keys: helper_left_keys,
            helper_join_right_keys: helper_right_keys,
            exposed_classes,
        })
    }

    fn class_at_state(linear: &LinearBody, leaf_count: usize, col: usize) -> Option<u32> {
        let mut idx = col;
        for leaf_idx in 0..leaf_count {
            let classes = &linear.leaf_classes[leaf_idx];
            if idx < classes.len() {
                return Some(classes[idx]);
            }
            idx -= classes.len();
        }
        None
    }

    fn outside_classes(linear: &LinearBody, pair_start: usize) -> HashSet<u32> {
        linear
            .leaf_classes
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != pair_start && *idx != pair_start + 1)
            .flat_map(|(_, classes)| classes.iter().copied())
            .collect()
    }

    fn projected_classes(linear: &LinearBody) -> Option<HashSet<u32>> {
        let mut out = HashSet::new();
        for expr in &linear.project {
            let ProjectExpr::Column(col) = expr else {
                return None;
            };
            out.insert(*linear.final_classes.get(*col)?);
        }
        Some(out)
    }

    fn skew_ratio_for_candidate(
        linear: &LinearBody,
        stats: &StatsManager,
        candidate: &Candidate,
    ) -> f64 {
        let rel = linear.leaves[candidate.pair_start];
        let Some(rel_stats) = stats.get_relation_stats(rel) else {
            return 0.0;
        };
        let mut ratio: f64 = 0.0;
        for (col, class) in linear.leaf_classes[candidate.pair_start]
            .iter()
            .copied()
            .enumerate()
        {
            if !candidate.exposed_classes.contains(&class) {
                continue;
            }
            let Some(col_stats) = rel_stats.get_column(col) else {
                continue;
            };
            if col_stats.distinct_estimate == 0 {
                continue;
            }
            ratio = ratio.max(rel_stats.cardinality as f64 / col_stats.distinct_estimate as f64);
        }
        ratio
    }

    fn build_helper_body(linear: &LinearBody, candidate: &Candidate) -> RirNode {
        let left = RirNode::Scan {
            rel: linear.leaves[candidate.pair_start],
        };
        let right = RirNode::Scan {
            rel: linear.leaves[candidate.pair_start + 1],
        };
        RirNode::Project {
            input: Box::new(RirNode::Join {
                left: Box::new(left),
                right: Box::new(right),
                left_keys: candidate.helper_join_left_keys.clone(),
                right_keys: candidate.helper_join_right_keys.clone(),
                join_type: JoinType::Inner,
            }),
            columns: candidate.helper_project.clone(),
        }
    }

    fn build_outer_body(
        linear: &LinearBody,
        candidate: &Candidate,
        helper_rel: RelId,
    ) -> Option<RirNode> {
        let mut node = RirNode::Scan {
            rel: linear.leaves[0],
        };
        let mut classes = linear.leaf_classes[0].clone();
        for leaf_idx in 1..candidate.pair_start {
            let step = &linear.joins[leaf_idx - 1];
            node = RirNode::Join {
                left: Box::new(node),
                right: Box::new(RirNode::Scan {
                    rel: linear.leaves[leaf_idx],
                }),
                left_keys: step.left_keys.clone(),
                right_keys: step.right_keys.clone(),
                join_type: JoinType::Inner,
            };
            classes.extend(linear.leaf_classes[leaf_idx].iter().copied());
        }
        let prefix_step = &linear.joins[candidate.pair_start - 1];
        let mut helper_right_keys = Vec::new();
        for &rk in &prefix_step.right_keys {
            let class = linear.leaf_classes[candidate.pair_start][rk];
            helper_right_keys.push(candidate.exposed_classes.iter().position(|c| *c == class)?);
        }
        node = RirNode::Join {
            left: Box::new(node),
            right: Box::new(RirNode::Scan { rel: helper_rel }),
            left_keys: prefix_step.left_keys.clone(),
            right_keys: helper_right_keys,
            join_type: JoinType::Inner,
        };
        classes.extend(candidate.exposed_classes.iter().copied());
        for leaf_idx in candidate.pair_start + 2..linear.leaves.len() {
            let step = &linear.joins[leaf_idx - 1];
            let mut left_keys = Vec::new();
            for &lk in &step.left_keys {
                let class = class_at_state(linear, leaf_idx, lk)?;
                left_keys.push(classes.iter().position(|c| *c == class)?);
            }
            node = RirNode::Join {
                left: Box::new(node),
                right: Box::new(RirNode::Scan {
                    rel: linear.leaves[leaf_idx],
                }),
                left_keys,
                right_keys: step.right_keys.clone(),
                join_type: JoinType::Inner,
            };
            classes.extend(linear.leaf_classes[leaf_idx].iter().copied());
        }
        let mut project = Vec::with_capacity(linear.project.len());
        for expr in &linear.project {
            let ProjectExpr::Column(col) = expr else {
                return None;
            };
            let class = *linear.final_classes.get(*col)?;
            let mapped = classes.iter().position(|c| *c == class)?;
            project.push(ProjectExpr::Column(mapped));
        }
        Some(RirNode::Project {
            input: Box::new(node),
            columns: project,
        })
    }

    struct UnionFind {
        parent: Vec<usize>,
    }

    impl UnionFind {
        fn new(len: usize) -> Self {
            Self {
                parent: (0..len).collect(),
            }
        }

        fn find(&mut self, x: usize) -> usize {
            let p = self.parent[x];
            if p == x {
                x
            } else {
                let root = self.find(p);
                self.parent[x] = root;
                root
            }
        }

        fn union(&mut self, a: usize, b: usize) {
            let ra = self.find(a);
            let rb = self.find(b);
            if ra != rb {
                self.parent[rb] = ra;
            }
        }
    }
}

#[path = "optimizer/stream_schedule_pass.rs"]
pub mod stream_schedule_pass;

#[cfg(test)]
mod helper_split_pass_tests {
    use std::collections::HashMap;

    use super::helper_split_pass;
    use xlog_core::{RelId, ScalarType, Schema};
    use xlog_ir::{CompiledRule, ExecutionPlan, JoinType, ProjectExpr, RirMeta, RirNode, Scc};
    use xlog_stats::{ColumnStats, StatsManager};

    fn edge_schema() -> Schema {
        Schema::new(vec![
            ("c0".to_string(), ScalarType::U32),
            ("c1".to_string(), ScalarType::U32),
        ])
    }

    fn helper_schema() -> Schema {
        Schema::new(vec![
            ("c0".to_string(), ScalarType::U32),
            ("c1".to_string(), ScalarType::U32),
        ])
    }

    fn schemas() -> HashMap<RelId, Schema> {
        (0..6)
            .map(|idx| (RelId(idx), edge_schema()))
            .collect::<HashMap<_, _>>()
    }

    fn left_deep_fixture_body() -> RirNode {
        let ab_bc = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(0) }),
            right: Box::new(RirNode::Scan { rel: RelId(1) }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let with_cd = RirNode::Join {
            left: Box::new(ab_bc),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![3],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let with_de = RirNode::Join {
            left: Box::new(with_cd),
            right: Box::new(RirNode::Scan { rel: RelId(3) }),
            left_keys: vec![5],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let with_ef = RirNode::Join {
            left: Box::new(with_de),
            right: Box::new(RirNode::Scan { rel: RelId(4) }),
            left_keys: vec![7],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let with_af = RirNode::Join {
            left: Box::new(with_ef),
            right: Box::new(RirNode::Scan { rel: RelId(5) }),
            left_keys: vec![0, 9],
            right_keys: vec![0, 1],
            join_type: JoinType::Inner,
        };
        RirNode::Project {
            input: Box::new(with_af),
            columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
                ProjectExpr::Column(5),
                ProjectExpr::Column(9),
            ],
        }
    }

    fn plan() -> ExecutionPlan {
        ExecutionPlan {
            sccs: vec![Scc {
                id: 0,
                predicates: vec!["out".to_string()],
                is_recursive: false,
            }],
            strata: vec![],
            rules_by_scc: vec![vec![CompiledRule {
                head: "out".to_string(),
                body: left_deep_fixture_body(),
                meta: RirMeta::with_schema(Schema::new(vec![
                    ("a".to_string(), ScalarType::U32),
                    ("b".to_string(), ScalarType::U32),
                    ("c".to_string(), ScalarType::U32),
                    ("d".to_string(), ScalarType::U32),
                    ("f".to_string(), ScalarType::U32),
                ])),
            }]],
            est_memory_peak: 0,
        }
    }

    fn stats_for_de(distinct_d: u64) -> StatsManager {
        let mut stats = StatsManager::new();
        for idx in 0..6 {
            stats.register_relation(RelId(idx));
            stats.update_cardinality(RelId(idx), 8192);
        }
        let mut d_col = ColumnStats::new(0, ScalarType::U32);
        d_col.update_distinct(distinct_d);
        stats.add_column_stats(RelId(3), d_col);
        stats
    }

    fn contains_scan(node: &RirNode, rel: RelId) -> bool {
        match node {
            RirNode::Scan { rel: scan_rel } => *scan_rel == rel,
            RirNode::Join { left, right, .. } => {
                contains_scan(left, rel) || contains_scan(right, rel)
            }
            RirNode::Project { input, .. }
            | RirNode::Filter { input, .. }
            | RirNode::Distinct { input, .. }
            | RirNode::GroupBy { input, .. } => contains_scan(input, rel),
            RirNode::Union { inputs } => inputs.iter().any(|input| contains_scan(input, rel)),
            RirNode::Diff { left, right } => contains_scan(left, rel) || contains_scan(right, rel),
            RirNode::Fixpoint {
                base, recursive, ..
            } => contains_scan(base, rel) || contains_scan(recursive, rel),
            RirNode::MultiWayJoin { inputs, .. } => {
                inputs.iter().any(|input| contains_scan(input, rel))
            }
            RirNode::TensorMaskedJoin { rel_index, .. } => {
                rel_index.iter().any(|(input_rel, _)| *input_rel == rel)
            }
            RirNode::Unit => false,
        }
    }

    #[test]
    fn helper_split_extracts_buried_pair() {
        let mut plan = plan();
        let schemas = schemas();
        let stats = stats_for_de(1);
        let specs = helper_split_pass::run(&mut plan, &schemas, &stats, |_| {
            ("__w37_helper_6".to_string(), RelId(6))
        });

        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "__w37_helper_6");
        assert_eq!(specs[0].rel_id, RelId(6));
        assert_eq!(specs[0].schema, helper_schema());
        assert_eq!(specs[0].source_rels, [RelId(3), RelId(4)]);
        assert_eq!(plan.rules_by_scc[0].len(), 2);
        assert_eq!(plan.rules_by_scc[0][0].head, "__w37_helper_6");
        assert_eq!(plan.rules_by_scc[0][1].head, "out");
        assert!(contains_scan(&plan.rules_by_scc[0][1].body, RelId(6)));
        assert!(plan.sccs[0]
            .predicates
            .iter()
            .any(|predicate| predicate == "__w37_helper_6"));
    }

    #[test]
    fn helper_split_ignores_flat_distribution() {
        let mut plan = plan();
        let schemas = schemas();
        let stats = stats_for_de(8192);
        let specs = helper_split_pass::run(&mut plan, &schemas, &stats, |_| {
            ("__w37_helper_6".to_string(), RelId(6))
        });

        assert!(specs.is_empty());
        assert_eq!(plan.rules_by_scc[0].len(), 1);
        assert!(!contains_scan(&plan.rules_by_scc[0][0].body, RelId(6)));
    }
}

/// W2.2 — selectivity-driven body rewriters for triangle and
/// 4-cycle canonical lowered shapes. `pub(super)` so
/// `selectivity_pass::run` can dispatch into them.
mod reorder {
    use std::collections::HashMap;
    use xlog_core::RelId;
    use xlog_ir::rir::ProjectExpr;
    use xlog_ir::{JoinType, RirNode};
    use xlog_stats::StatsManager;

    fn ac3(atom: u8, col: u8) -> u8 {
        atom * 2 + col
    }
    fn ac4(atom: u8, col: u8) -> u8 {
        atom * 2 + col
    }
    fn uf_find_n<const N: usize>(parent: &mut [u8; N], x: u8) -> u8 {
        let mut root = x;
        while parent[root as usize] != root {
            root = parent[root as usize];
        }
        let mut cur = x;
        while parent[cur as usize] != root {
            let next = parent[cur as usize];
            parent[cur as usize] = root;
            cur = next;
        }
        root
    }
    fn uf_union_n<const N: usize>(parent: &mut [u8; N], a: u8, b: u8) {
        let ra = uf_find_n(parent, a);
        let rb = uf_find_n(parent, b);
        if ra != rb {
            parent[rb as usize] = ra;
        }
    }

    fn populated_card(stats: &StatsManager, rel: RelId) -> Option<u64> {
        stats
            .get_relation_stats(rel)
            .map(|s| s.cardinality)
            .filter(|c| *c > 0)
    }

    // ---------------------------------------------------------
    // Triangle rewriter
    // ---------------------------------------------------------

    struct TriangleSemantics {
        rel_xy: RelId,
        rel_yz: RelId,
        rel_xz: RelId,
    }

    fn match_and_infer_triangle(body: &RirNode) -> Option<TriangleSemantics> {
        let RirNode::Project {
            input: outer_input,
            columns,
        } = body
        else {
            return None;
        };
        let RirNode::Join {
            left: l1,
            right: r1,
            left_keys: lk1,
            right_keys: rk1,
            join_type: jt1,
        } = outer_input.as_ref()
        else {
            return None;
        };
        if !matches!(jt1, JoinType::Inner) {
            return None;
        }
        let RirNode::Scan { rel: rel_third } = r1.as_ref() else {
            return None;
        };
        let RirNode::Join {
            left: l2,
            right: r2,
            left_keys: lk2,
            right_keys: rk2,
            join_type: jt2,
        } = l1.as_ref()
        else {
            return None;
        };
        if !matches!(jt2, JoinType::Inner) {
            return None;
        }
        let RirNode::Scan { rel: rel_inner_l } = l2.as_ref() else {
            return None;
        };
        let RirNode::Scan { rel: rel_inner_r } = r2.as_ref() else {
            return None;
        };
        if lk2.len() != 1 || rk2.len() != 1 || lk1.len() != 2 || rk1.len() != 2 {
            return None;
        }
        if columns.len() != 3 {
            return None;
        }
        if lk2[0] >= 2 || rk2[0] >= 2 {
            return None;
        }
        if lk1.iter().any(|k| *k >= 4) || rk1.iter().any(|k| *k >= 2) {
            return None;
        }

        let mut parent = [0u8, 1, 2, 3, 4, 5];
        uf_union_n::<6>(&mut parent, ac3(0, lk2[0] as u8), ac3(1, rk2[0] as u8));
        for i in 0..2 {
            let inner_ac = match lk1[i] {
                0 => (0u8, 0u8),
                1 => (0, 1),
                2 => (1, 0),
                3 => (1, 1),
                _ => return None,
            };
            uf_union_n::<6>(
                &mut parent,
                ac3(inner_ac.0, inner_ac.1),
                ac3(2, rk1[i] as u8),
            );
        }
        let roots: [u8; 6] = std::array::from_fn(|i| uf_find_n::<6>(&mut parent, i as u8));
        let mut counts: HashMap<u8, u8> = HashMap::new();
        for r in &roots {
            *counts.entry(*r).or_insert(0) += 1;
        }
        if counts.len() != 3 || counts.values().any(|c| *c != 2) {
            return None;
        }
        let mut head_classes: [u8; 3] = [0; 3];
        for (i, pc) in columns.iter().enumerate() {
            let ProjectExpr::Column(k) = pc else {
                return None;
            };
            let outer_ac = match *k {
                0 => (0u8, 0u8),
                1 => (0, 1),
                2 => (1, 0),
                3 => (1, 1),
                4 => (2, 0),
                5 => (2, 1),
                _ => return None,
            };
            head_classes[i] = uf_find_n::<6>(&mut parent, ac3(outer_ac.0, outer_ac.1));
        }
        if head_classes[0] == head_classes[1]
            || head_classes[0] == head_classes[2]
            || head_classes[1] == head_classes[2]
        {
            return None;
        }
        let x_class = head_classes[0];
        let y_class = head_classes[1];
        let z_class = head_classes[2];
        let atom_classes = |a: u8| (roots[ac3(a, 0) as usize], roots[ac3(a, 1) as usize]);
        let atom_rels = [*rel_inner_l, *rel_inner_r, *rel_third];
        let mut rel_xy = None;
        let mut rel_yz = None;
        let mut rel_xz = None;
        for atom_idx in 0..3u8 {
            let (c0, c1) = atom_classes(atom_idx);
            let bx = c0 == x_class || c1 == x_class;
            let by = c0 == y_class || c1 == y_class;
            let bz = c0 == z_class || c1 == z_class;
            match (bx, by, bz) {
                (true, true, false) => rel_xy = Some(atom_rels[atom_idx as usize]),
                (false, true, true) => rel_yz = Some(atom_rels[atom_idx as usize]),
                (true, false, true) => rel_xz = Some(atom_rels[atom_idx as usize]),
                _ => return None,
            }
        }
        Some(TriangleSemantics {
            rel_xy: rel_xy?,
            rel_yz: rel_yz?,
            rel_xz: rel_xz?,
        })
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TriangleInnerPair {
        YShared,
        XShared,
        ZShared,
    }

    fn build_triangle_body(s: &TriangleSemantics, inner_pair: TriangleInnerPair) -> RirNode {
        let mk_scan = |r: RelId| RirNode::Scan { rel: r };
        match inner_pair {
            TriangleInnerPair::YShared => {
                let inner = RirNode::Join {
                    left: Box::new(mk_scan(s.rel_xy)),
                    right: Box::new(mk_scan(s.rel_yz)),
                    left_keys: vec![1],
                    right_keys: vec![0],
                    join_type: JoinType::Inner,
                };
                let outer = RirNode::Join {
                    left: Box::new(inner),
                    right: Box::new(mk_scan(s.rel_xz)),
                    left_keys: vec![0, 3],
                    right_keys: vec![0, 1],
                    join_type: JoinType::Inner,
                };
                RirNode::Project {
                    input: Box::new(outer),
                    columns: vec![
                        ProjectExpr::Column(0),
                        ProjectExpr::Column(1),
                        ProjectExpr::Column(3),
                    ],
                }
            }
            TriangleInnerPair::XShared => {
                let inner = RirNode::Join {
                    left: Box::new(mk_scan(s.rel_xy)),
                    right: Box::new(mk_scan(s.rel_xz)),
                    left_keys: vec![0],
                    right_keys: vec![0],
                    join_type: JoinType::Inner,
                };
                let outer = RirNode::Join {
                    left: Box::new(inner),
                    right: Box::new(mk_scan(s.rel_yz)),
                    left_keys: vec![1, 3],
                    right_keys: vec![0, 1],
                    join_type: JoinType::Inner,
                };
                RirNode::Project {
                    input: Box::new(outer),
                    columns: vec![
                        ProjectExpr::Column(0),
                        ProjectExpr::Column(1),
                        ProjectExpr::Column(3),
                    ],
                }
            }
            TriangleInnerPair::ZShared => {
                let inner = RirNode::Join {
                    left: Box::new(mk_scan(s.rel_xz)),
                    right: Box::new(mk_scan(s.rel_yz)),
                    left_keys: vec![1],
                    right_keys: vec![1],
                    join_type: JoinType::Inner,
                };
                let outer = RirNode::Join {
                    left: Box::new(inner),
                    right: Box::new(mk_scan(s.rel_xy)),
                    left_keys: vec![0, 2],
                    right_keys: vec![0, 1],
                    join_type: JoinType::Inner,
                };
                RirNode::Project {
                    input: Box::new(outer),
                    columns: vec![
                        ProjectExpr::Column(0),
                        ProjectExpr::Column(2),
                        ProjectExpr::Column(3),
                    ],
                }
            }
        }
    }

    pub fn try_reorder_triangle(body: &RirNode, stats: &StatsManager) -> Option<RirNode> {
        let s = match_and_infer_triangle(body)?;
        let _ = (
            populated_card(stats, s.rel_xy)?,
            populated_card(stats, s.rel_yz)?,
            populated_card(stats, s.rel_xz)?,
        );
        let est_y = stats.estimate_join_cardinality(s.rel_xy, s.rel_yz, &[1], &[0]);
        let est_x = stats.estimate_join_cardinality(s.rel_xy, s.rel_xz, &[0], &[0]);
        let est_z = stats.estimate_join_cardinality(s.rel_yz, s.rel_xz, &[1], &[1]);
        let mut best = (TriangleInnerPair::YShared, est_y);
        if est_x < best.1 {
            best = (TriangleInnerPair::XShared, est_x);
        }
        if est_z < best.1 {
            best = (TriangleInnerPair::ZShared, est_z);
        }
        let candidate = build_triangle_body(&s, best.0);
        // Skip when the candidate is structurally identical to
        // the input (no-op rewrite). RirNode doesn't impl
        // PartialEq, so compare via Debug — bodies are small
        // (≤ 6 Scans + 2 Joins + 1 Project) so the cost is
        // negligible relative to the optimizer's broader work.
        if format!("{:?}", candidate) == format!("{:?}", body) {
            return None;
        }
        Some(candidate)
    }

    // ---------------------------------------------------------
    // 4-cycle rewriter
    // ---------------------------------------------------------

    struct Cycle4Semantics {
        rel_wx: RelId,
        rel_xy: RelId,
        rel_yz: RelId,
        rel_zw: RelId,
    }

    fn match_and_infer_4cycle(body: &RirNode) -> Option<Cycle4Semantics> {
        let RirNode::Project {
            input: outer_input,
            columns,
        } = body
        else {
            return None;
        };
        let RirNode::Join {
            left: outer_l,
            right: outer_r,
            left_keys: olk,
            right_keys: ork,
            join_type: ojt,
        } = outer_input.as_ref()
        else {
            return None;
        };
        if !matches!(ojt, JoinType::Inner) {
            return None;
        }
        let RirNode::Join {
            left: ll,
            right: lr,
            left_keys: ilk_l,
            right_keys: irk_l,
            join_type: ijt_l,
        } = outer_l.as_ref()
        else {
            return None;
        };
        if !matches!(ijt_l, JoinType::Inner) {
            return None;
        }
        let RirNode::Scan { rel: rel_ll } = ll.as_ref() else {
            return None;
        };
        let RirNode::Scan { rel: rel_lr } = lr.as_ref() else {
            return None;
        };
        let RirNode::Join {
            left: rl,
            right: rr,
            left_keys: ilk_r,
            right_keys: irk_r,
            join_type: ijt_r,
        } = outer_r.as_ref()
        else {
            return None;
        };
        if !matches!(ijt_r, JoinType::Inner) {
            return None;
        }
        let RirNode::Scan { rel: rel_rl } = rl.as_ref() else {
            return None;
        };
        let RirNode::Scan { rel: rel_rr } = rr.as_ref() else {
            return None;
        };
        if ilk_l.len() != 1 || irk_l.len() != 1 || ilk_r.len() != 1 || irk_r.len() != 1 {
            return None;
        }
        if olk.len() != 2 || ork.len() != 2 || columns.len() != 4 {
            return None;
        }
        if ilk_l[0] >= 2 || irk_l[0] >= 2 || ilk_r[0] >= 2 || irk_r[0] >= 2 {
            return None;
        }
        if olk.iter().any(|k| *k >= 4) || ork.iter().any(|k| *k >= 4) {
            return None;
        }

        let mut parent = [0u8, 1, 2, 3, 4, 5, 6, 7];
        uf_union_n::<8>(&mut parent, ac4(0, ilk_l[0] as u8), ac4(1, irk_l[0] as u8));
        uf_union_n::<8>(&mut parent, ac4(2, ilk_r[0] as u8), ac4(3, irk_r[0] as u8));
        for i in 0..2 {
            let l_ac = match olk[i] {
                0 => (0u8, 0u8),
                1 => (0, 1),
                2 => (1, 0),
                3 => (1, 1),
                _ => return None,
            };
            let r_ac = match ork[i] {
                0 => (2u8, 0u8),
                1 => (2, 1),
                2 => (3, 0),
                3 => (3, 1),
                _ => return None,
            };
            uf_union_n::<8>(&mut parent, ac4(l_ac.0, l_ac.1), ac4(r_ac.0, r_ac.1));
        }
        let roots: [u8; 8] = std::array::from_fn(|i| uf_find_n::<8>(&mut parent, i as u8));
        let mut counts: HashMap<u8, u8> = HashMap::new();
        for r in &roots {
            *counts.entry(*r).or_insert(0) += 1;
        }
        if counts.len() != 4 || counts.values().any(|c| *c != 2) {
            return None;
        }

        let mut head_classes: [u8; 4] = [0; 4];
        for (i, pc) in columns.iter().enumerate() {
            let ProjectExpr::Column(k) = pc else {
                return None;
            };
            let ac = match *k {
                0 => (0u8, 0u8),
                1 => (0, 1),
                2 => (1, 0),
                3 => (1, 1),
                4 => (2, 0),
                5 => (2, 1),
                6 => (3, 0),
                7 => (3, 1),
                _ => return None,
            };
            head_classes[i] = uf_find_n::<8>(&mut parent, ac4(ac.0, ac.1));
        }
        for i in 0..4 {
            for j in (i + 1)..4 {
                if head_classes[i] == head_classes[j] {
                    return None;
                }
            }
        }
        let w_class = head_classes[0];
        let x_class = head_classes[1];
        let y_class = head_classes[2];
        let z_class = head_classes[3];
        let atom_classes = |a: u8| (roots[ac4(a, 0) as usize], roots[ac4(a, 1) as usize]);
        let atom_rels = [*rel_ll, *rel_lr, *rel_rl, *rel_rr];
        let mut rel_wx = None;
        let mut rel_xy = None;
        let mut rel_yz = None;
        let mut rel_zw = None;
        for atom_idx in 0..4u8 {
            let (c0, c1) = atom_classes(atom_idx);
            let bw = c0 == w_class || c1 == w_class;
            let bx = c0 == x_class || c1 == x_class;
            let by = c0 == y_class || c1 == y_class;
            let bz = c0 == z_class || c1 == z_class;
            match (bw, bx, by, bz) {
                (true, true, false, false) => rel_wx = Some(atom_rels[atom_idx as usize]),
                (false, true, true, false) => rel_xy = Some(atom_rels[atom_idx as usize]),
                (false, false, true, true) => rel_yz = Some(atom_rels[atom_idx as usize]),
                (true, false, false, true) => rel_zw = Some(atom_rels[atom_idx as usize]),
                _ => return None,
            }
        }
        Some(Cycle4Semantics {
            rel_wx: rel_wx?,
            rel_xy: rel_xy?,
            rel_yz: rel_yz?,
            rel_zw: rel_zw?,
        })
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Cycle4Grouping {
        Default,
        Alt,
    }

    fn build_4cycle_body(s: &Cycle4Semantics, g: Cycle4Grouping) -> RirNode {
        let mk_scan = |r: RelId| RirNode::Scan { rel: r };
        match g {
            Cycle4Grouping::Default => {
                let il = RirNode::Join {
                    left: Box::new(mk_scan(s.rel_wx)),
                    right: Box::new(mk_scan(s.rel_xy)),
                    left_keys: vec![1],
                    right_keys: vec![0],
                    join_type: JoinType::Inner,
                };
                let ir = RirNode::Join {
                    left: Box::new(mk_scan(s.rel_yz)),
                    right: Box::new(mk_scan(s.rel_zw)),
                    left_keys: vec![1],
                    right_keys: vec![0],
                    join_type: JoinType::Inner,
                };
                let outer = RirNode::Join {
                    left: Box::new(il),
                    right: Box::new(ir),
                    left_keys: vec![0, 3],
                    right_keys: vec![3, 0],
                    join_type: JoinType::Inner,
                };
                RirNode::Project {
                    input: Box::new(outer),
                    columns: vec![
                        ProjectExpr::Column(0),
                        ProjectExpr::Column(1),
                        ProjectExpr::Column(3),
                        ProjectExpr::Column(5),
                    ],
                }
            }
            Cycle4Grouping::Alt => {
                let il = RirNode::Join {
                    left: Box::new(mk_scan(s.rel_xy)),
                    right: Box::new(mk_scan(s.rel_yz)),
                    left_keys: vec![1],
                    right_keys: vec![0],
                    join_type: JoinType::Inner,
                };
                let ir = RirNode::Join {
                    left: Box::new(mk_scan(s.rel_zw)),
                    right: Box::new(mk_scan(s.rel_wx)),
                    left_keys: vec![1],
                    right_keys: vec![0],
                    join_type: JoinType::Inner,
                };
                let outer = RirNode::Join {
                    left: Box::new(il),
                    right: Box::new(ir),
                    left_keys: vec![0, 3],
                    right_keys: vec![3, 0],
                    join_type: JoinType::Inner,
                };
                RirNode::Project {
                    input: Box::new(outer),
                    columns: vec![
                        ProjectExpr::Column(5),
                        ProjectExpr::Column(0),
                        ProjectExpr::Column(1),
                        ProjectExpr::Column(3),
                    ],
                }
            }
        }
    }

    pub fn try_reorder_4cycle(body: &RirNode, stats: &StatsManager) -> Option<RirNode> {
        let s = match_and_infer_4cycle(body)?;
        let _ = (
            populated_card(stats, s.rel_wx)?,
            populated_card(stats, s.rel_xy)?,
            populated_card(stats, s.rel_yz)?,
            populated_card(stats, s.rel_zw)?,
        );
        let est_default = stats
            .estimate_join_cardinality(s.rel_wx, s.rel_xy, &[1], &[0])
            .saturating_add(stats.estimate_join_cardinality(s.rel_yz, s.rel_zw, &[1], &[0]));
        let est_alt = stats
            .estimate_join_cardinality(s.rel_xy, s.rel_yz, &[1], &[0])
            .saturating_add(stats.estimate_join_cardinality(s.rel_zw, s.rel_wx, &[1], &[0]));
        let chosen = if est_alt < est_default {
            Cycle4Grouping::Alt
        } else {
            Cycle4Grouping::Default
        };
        let candidate = build_4cycle_body(&s, chosen);
        if format!("{:?}", candidate) == format!("{:?}", body) {
            return None;
        }
        Some(candidate)
    }
}

#[cfg(test)]
mod selectivity_pass_tests {
    use super::selectivity_pass;
    use crate::Compiler;
    use xlog_stats::StatsManager;

    fn body_snapshots(plan: &xlog_ir::ExecutionPlan) -> Vec<String> {
        plan.rules_by_scc
            .iter()
            .flatten()
            .map(|r| format!("{:?}", r.body))
            .collect()
    }

    #[test]
    fn selectivity_pass_is_noop_for_triangle_plan() {
        let mut compiler = Compiler::new();
        let plan = compiler
            .compile("tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z).")
            .expect("compile");
        let before = body_snapshots(&plan);
        let stats = StatsManager::new();
        let mut plan2 = plan.clone();
        selectivity_pass::run(&mut plan2, &stats, &std::collections::HashMap::new());
        let after = body_snapshots(&plan2);
        assert_eq!(
            before, after,
            "selectivity_pass must preserve every triangle rule body byte-for-byte"
        );
    }

    #[test]
    fn selectivity_pass_is_noop_for_4cycle_plan() {
        let mut compiler = Compiler::new();
        let plan = compiler
            .compile("cycle4(W, X, Y, Z) :- e1(W, X), e2(X, Y), e3(Y, Z), e4(Z, W).")
            .expect("compile");
        let before = body_snapshots(&plan);
        let stats = StatsManager::new();
        let mut plan2 = plan.clone();
        selectivity_pass::run(&mut plan2, &stats, &std::collections::HashMap::new());
        let after = body_snapshots(&plan2);
        assert_eq!(
            before, after,
            "selectivity_pass must preserve every 4-cycle rule body byte-for-byte"
        );
    }

    #[test]
    fn selectivity_pass_is_noop_for_recursive_scc() {
        let mut compiler = Compiler::new();
        let plan = compiler
            .compile(
                "edge(1, 2). edge(2, 3). \
                 reach(X, Y) :- edge(X, Y). \
                 reach(X, Z) :- reach(X, Y), edge(Y, Z).",
            )
            .expect("compile");
        let before = body_snapshots(&plan);
        let stats = StatsManager::new();
        let mut plan2 = plan.clone();
        selectivity_pass::run(&mut plan2, &stats, &std::collections::HashMap::new());
        let after = body_snapshots(&plan2);
        assert_eq!(
            before, after,
            "selectivity_pass must preserve recursive SCC bodies byte-for-byte"
        );
    }

    // ---------------------------------------------------------
    // W2.2 — selectivity-driven reordering tests
    // ---------------------------------------------------------

    use xlog_core::RelId;
    use xlog_ir::plan::{CompiledRule, PlanBuilder, Scc};
    use xlog_ir::rir::ProjectExpr;
    use xlog_ir::{ExecutionPlan, JoinType, RirNode};

    /// Build a hand-crafted canonical lowered triangle plan
    /// with three Scans at RelId(1), RelId(2), RelId(3) for
    /// (e_xy, e_yz, e_xz). Bypasses the optimizer entirely so
    /// the W2.2 cert is a clean stats-→-pair-choice
    /// observation, not a confounded test of optimizer + W2.2.
    ///
    /// Default canonical shape (Y-shared inner): inner keys
    /// `[1]/[0]`, outer keys `[0,3]/[0,1]`, project `[0,1,3]`.
    fn synth_triangle_plan() -> ExecutionPlan {
        let inner = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let outer = RirNode::Join {
            left: Box::new(inner),
            right: Box::new(RirNode::Scan { rel: RelId(3) }),
            left_keys: vec![0, 3],
            right_keys: vec![0, 1],
            join_type: JoinType::Inner,
        };
        let body = RirNode::Project {
            input: Box::new(outer),
            columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
            ],
        };
        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["tri".to_string()],
            is_recursive: false,
        });
        builder.add_rule(
            0,
            CompiledRule {
                head: "tri".to_string(),
                body,
                meta: Default::default(),
            },
        );
        builder.build()
    }

    /// Seed a `StatsManager` with three triangle-edge
    /// cardinalities at the conventional RelIds (1, 2, 3) used
    /// by `synth_triangle_plan`.
    fn seed_triangle_stats(c1: u64, c2: u64, c3: u64) -> StatsManager {
        let mut stats = StatsManager::new();
        for (rid, card) in [(RelId(1), c1), (RelId(2), c2), (RelId(3), c3)] {
            stats.register_relation(rid);
            stats.update_cardinality(rid, card);
        }
        stats
    }

    /// Inspect the (left RelId, right RelId) of the inner Join
    /// in a canonical lowered triangle body. Used by W2.2
    /// reordering certs.
    ///
    /// After `compile()` the body is a `MultiWayJoin` whose
    /// `fallback` field holds the post-selectivity-pass
    /// pre-promotion shape — that's where the inner-pair
    /// signature lives. The helper unwraps `MultiWayJoin →
    /// fallback` if needed before drilling into the binary
    /// Join structure.
    fn inspect_triangle_inner_pair(plan: &xlog_ir::ExecutionPlan) -> Option<(RelId, RelId)> {
        let body = &plan.rules_by_scc.iter().flatten().next()?.body;
        let body = match body {
            xlog_ir::RirNode::MultiWayJoin { fallback, .. } => fallback.as_ref(),
            other => other,
        };
        let xlog_ir::RirNode::Project { input, .. } = body else {
            return None;
        };
        let xlog_ir::RirNode::Join { left, .. } = input.as_ref() else {
            return None;
        };
        let xlog_ir::RirNode::Join {
            left: l2,
            right: r2,
            ..
        } = left.as_ref()
        else {
            return None;
        };
        let xlog_ir::RirNode::Scan { rel: rel_l } = l2.as_ref() else {
            return None;
        };
        let xlog_ir::RirNode::Scan { rel: rel_r } = r2.as_ref() else {
            return None;
        };
        Some((*rel_l, *rel_r))
    }

    /// W2.2 — snapshot 1: cards favor `(e1, e2)` Y-shared inner.
    /// Triangle rule: `tri(X, Y, Z) :- e1(X, Y), e2(Y, Z), e3(X, Z)`.
    /// To make Y-shared smallest, give e1 + e2 small cards and
    /// e3 a large card so all pair products are dominated by
    /// pairs containing e3 — except the pair (e1, e2) which
    /// is the smallest product.
    #[test]
    fn selectivity_pass_picks_y_shared_inner_when_e1_e2_smallest() {
        let mut plan = synth_triangle_plan();
        // e1=10, e2=10, e3=100_000 → Y-shared (e1⋈e2) smallest.
        let stats = seed_triangle_stats(10, 10, 100_000);
        selectivity_pass::run(&mut plan, &stats, &std::collections::HashMap::new());
        let pair = inspect_triangle_inner_pair(&plan).expect("inner pair");
        // Y-shared inner = (e_xy, e_yz) = (RelId(1), RelId(2)).
        assert!(
            pair == (RelId(1), RelId(2)) || pair == (RelId(2), RelId(1)),
            "expected (RelId(1), RelId(2)) for Y-shared; got {:?}",
            pair
        );
    }

    /// W2.2 — snapshot 2: cards favor `(e1, e3)` X-shared inner.
    /// e1 + e3 small, e2 large.
    #[test]
    fn selectivity_pass_picks_x_shared_inner_when_e1_e3_smallest() {
        let mut plan = synth_triangle_plan();
        // e1=10, e2=100_000, e3=10 → X-shared (e1⋈e3) smallest.
        let stats = seed_triangle_stats(10, 100_000, 10);
        selectivity_pass::run(&mut plan, &stats, &std::collections::HashMap::new());
        let pair = inspect_triangle_inner_pair(&plan).expect("inner pair");
        // X-shared inner = (e_xy, e_xz) = (RelId(1), RelId(3)).
        assert!(
            pair == (RelId(1), RelId(3)) || pair == (RelId(3), RelId(1)),
            "expected (RelId(1), RelId(3)) for X-shared; got {:?}",
            pair
        );
    }

    /// W2.2 — snapshot 3: cards favor `(e2, e3)` Z-shared inner.
    /// e2 + e3 small, e1 large.
    #[test]
    fn selectivity_pass_picks_z_shared_inner_when_e2_e3_smallest() {
        let mut plan = synth_triangle_plan();
        // e1=100_000, e2=10, e3=10 → Z-shared (e2⋈e3) smallest.
        let stats = seed_triangle_stats(100_000, 10, 10);
        selectivity_pass::run(&mut plan, &stats, &std::collections::HashMap::new());
        let pair = inspect_triangle_inner_pair(&plan).expect("inner pair");
        // Z-shared inner = (e_yz, e_xz) = (RelId(2), RelId(3)).
        assert!(
            pair == (RelId(2), RelId(3)) || pair == (RelId(3), RelId(2)),
            "expected (RelId(2), RelId(3)) for Z-shared; got {:?}",
            pair
        );
    }

    /// W2.2 — two snapshots produce different inner pairs. Pins
    /// "stats drive the order, not deterministic
    /// canonicalization." Deterministic canonicalization that
    /// ignores stats CANNOT pass this gate.
    #[test]
    fn selectivity_pass_two_snapshots_produce_different_inner_pairs() {
        let mut plan_a = synth_triangle_plan();
        let stats_a = seed_triangle_stats(10, 10, 100_000); // Y-shared
        selectivity_pass::run(&mut plan_a, &stats_a, &std::collections::HashMap::new());
        let pair_a = inspect_triangle_inner_pair(&plan_a).expect("snapshot A pair");

        let mut plan_b = synth_triangle_plan();
        let stats_b = seed_triangle_stats(100_000, 10, 10); // Z-shared
        selectivity_pass::run(&mut plan_b, &stats_b, &std::collections::HashMap::new());
        let pair_b = inspect_triangle_inner_pair(&plan_b).expect("snapshot B pair");

        let normalize = |(a, b): (RelId, RelId)| -> (RelId, RelId) {
            if a.0 <= b.0 {
                (a, b)
            } else {
                (b, a)
            }
        };
        assert_ne!(
            normalize(pair_a),
            normalize(pair_b),
            "two different stats snapshots must produce different inner pairs; \
             got A = {:?}, B = {:?}",
            pair_a,
            pair_b
        );
    }

    /// W2.2 — fallback edge case: relation cards present but no
    /// column statistics. The 10% default fallback inside
    /// `estimate_join_cardinality` means all three pair
    /// estimates collapse to roughly the same ratio. The pass
    /// either picks SOME pair or leaves the body unchanged;
    /// the test is tolerant by design and documents the
    /// uninformative-fallback case explicitly.
    #[test]
    fn selectivity_pass_with_only_relation_cards_may_pick_arbitrary_pair() {
        let mut plan = synth_triangle_plan();
        // All three cards equal — no column stats to break ties.
        let stats = seed_triangle_stats(100, 100, 100);
        selectivity_pass::run(&mut plan, &stats, &std::collections::HashMap::new());
        // Either a triangle inner pair is identifiable (any of
        // the three) or the body stays unchanged. Both are OK.
        let _ = inspect_triangle_inner_pair(&plan);
    }

    // ---------------------------------------------------------
    // W2.2 — 4-cycle compile-time reordering tests
    // ---------------------------------------------------------

    /// Build a hand-crafted canonical lowered 4-cycle plan
    /// with four Scans at RelId(1), RelId(2), RelId(3), RelId(4)
    /// for (e_wx, e_xy, e_yz, e_zw). Bypasses the optimizer.
    /// Default canonical bushy shape: inner-left
    /// `(e_wx ⋈ e_xy)` on X, inner-right `(e_yz ⋈ e_zw)` on Z,
    /// outer keys `[0, 3] / [3, 0]`, project `[0, 1, 3, 5]`.
    fn synth_4cycle_plan() -> ExecutionPlan {
        let inner_left = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let inner_right = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(3) }),
            right: Box::new(RirNode::Scan { rel: RelId(4) }),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let outer = RirNode::Join {
            left: Box::new(inner_left),
            right: Box::new(inner_right),
            left_keys: vec![0, 3],
            right_keys: vec![3, 0],
            join_type: JoinType::Inner,
        };
        let body = RirNode::Project {
            input: Box::new(outer),
            columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
                ProjectExpr::Column(5),
            ],
        };
        let mut builder = PlanBuilder::new();
        builder.add_scc(Scc {
            id: 0,
            predicates: vec!["cyc".to_string()],
            is_recursive: false,
        });
        builder.add_rule(
            0,
            CompiledRule {
                head: "cyc".to_string(),
                body,
                meta: Default::default(),
            },
        );
        builder.build()
    }

    fn seed_4cycle_stats(c1: u64, c2: u64, c3: u64, c4: u64) -> StatsManager {
        let mut stats = StatsManager::new();
        for (rid, card) in [
            (RelId(1), c1),
            (RelId(2), c2),
            (RelId(3), c3),
            (RelId(4), c4),
        ] {
            stats.register_relation(rid);
            stats.update_cardinality(rid, card);
        }
        stats
    }

    /// Recover the 4-cycle inner-grouping signature: `(left_left,
    /// left_right, right_left, right_right)` Scan RelIds. Used
    /// to identify which grouping the rewriter chose.
    fn inspect_4cycle_grouping(
        plan: &xlog_ir::ExecutionPlan,
    ) -> Option<(RelId, RelId, RelId, RelId)> {
        let body = &plan.rules_by_scc.iter().flatten().next()?.body;
        let body = match body {
            xlog_ir::RirNode::MultiWayJoin { fallback, .. } => fallback.as_ref(),
            other => other,
        };
        let xlog_ir::RirNode::Project { input, .. } = body else {
            return None;
        };
        let xlog_ir::RirNode::Join { left, right, .. } = input.as_ref() else {
            return None;
        };
        let xlog_ir::RirNode::Join {
            left: ll,
            right: lr,
            ..
        } = left.as_ref()
        else {
            return None;
        };
        let xlog_ir::RirNode::Join {
            left: rl,
            right: rr,
            ..
        } = right.as_ref()
        else {
            return None;
        };
        let xlog_ir::RirNode::Scan { rel: r_ll } = ll.as_ref() else {
            return None;
        };
        let xlog_ir::RirNode::Scan { rel: r_lr } = lr.as_ref() else {
            return None;
        };
        let xlog_ir::RirNode::Scan { rel: r_rl } = rl.as_ref() else {
            return None;
        };
        let xlog_ir::RirNode::Scan { rel: r_rr } = rr.as_ref() else {
            return None;
        };
        Some((*r_ll, *r_lr, *r_rl, *r_rr))
    }

    /// W2.2 — 4-cycle: cards favor Default grouping
    /// `(e_wx⋈e_xy on X) + (e_yz⋈e_zw on Z)`. Default cost is
    /// `est(WX⋈XY)+est(YZ⋈ZW) = 0.1*c1*c2 + 0.1*c3*c4`.
    /// Alt cost is `0.1*c2*c3 + 0.1*c4*c1`. Default smaller
    /// when `c1*c2 + c3*c4 < c2*c3 + c4*c1`. With
    /// (c1=10, c2=10, c3=100_000, c4=100_000):
    ///   default = 100 + 10^10 ≈ 10^10.
    ///   alt = 10^6 + 10^6 ≈ 2*10^6.
    /// → alt is smaller, so this fixture actually favors Alt.
    /// Use (c1=10, c2=10, c3=10, c4=10_000_000) instead:
    ///   default = 100 + 10^8 = 10^8.
    ///   alt = 100 + 10^8 = 10^8 (same).
    /// Need uneven c4 vs others: (c1=10, c2=10, c3=10_000_000, c4=10):
    ///   default = 100 + 10^8 = 10^8.
    ///   alt = 10^8 + 100 = 10^8 (same).
    /// Default favored when c1*c2 << c2*c3 AND c3*c4 << c4*c1.
    /// I.e., c1 small and c4 small relative to c2 and c3.
    /// (c1=10, c2=10_000, c3=10_000, c4=10):
    ///   default = 0.1*100_000 + 0.1*100_000 = 20_000.
    ///   alt = 0.1*100_000_000 + 0.1*100 = 10_000_010.
    /// → Default smaller. ✓
    #[test]
    fn selectivity_pass_4cycle_picks_default_grouping_when_corners_smallest() {
        let mut plan = synth_4cycle_plan();
        let stats = seed_4cycle_stats(10, 10_000, 10_000, 10);
        selectivity_pass::run(&mut plan, &stats, &std::collections::HashMap::new());
        let (ll, lr, rl, rr) = inspect_4cycle_grouping(&plan).expect("grouping");
        // Default: (e_wx, e_xy, e_yz, e_zw) = (RelId(1..4)).
        assert_eq!(
            (ll, lr, rl, rr),
            (RelId(1), RelId(2), RelId(3), RelId(4)),
            "expected Default grouping"
        );
    }

    /// W2.2 — 4-cycle: cards favor Alt grouping
    /// `(e_xy⋈e_yz on Y) + (e_zw⋈e_wx on W)`. Alt smaller when
    /// `c2*c3 + c4*c1 < c1*c2 + c3*c4`. Use
    /// (c1=10_000, c2=10, c3=10, c4=10_000):
    ///   default = 0.1*100_000 + 0.1*100_000 = 20_000.
    ///   alt = 0.1*100 + 0.1*10^8 = 10_000_010.
    /// → Default still wins. Need c1*c2 LARGE and c3*c4 LARGE
    /// while c2*c3 SMALL and c4*c1 SMALL. Try
    /// (c1=10_000, c2=10_000, c3=10, c4=10):
    ///   default = 0.1*10^8 + 0.1*100 = 10_000_010.
    ///   alt = 0.1*100_000 + 0.1*100_000 = 20_000.
    /// → Alt smaller. ✓
    #[test]
    fn selectivity_pass_4cycle_picks_alt_grouping_when_diagonals_smallest() {
        let mut plan = synth_4cycle_plan();
        let stats = seed_4cycle_stats(10_000, 10_000, 10, 10);
        selectivity_pass::run(&mut plan, &stats, &std::collections::HashMap::new());
        let (ll, lr, rl, rr) = inspect_4cycle_grouping(&plan).expect("grouping");
        // Alt: (e_xy, e_yz, e_zw, e_wx) = (RelId(2), RelId(3), RelId(4), RelId(1)).
        assert_eq!(
            (ll, lr, rl, rr),
            (RelId(2), RelId(3), RelId(4), RelId(1)),
            "expected Alt grouping"
        );
    }

    /// W2.2 — same plan, two stats snapshots → two different
    /// 4-cycle groupings. Pins "stats drive the choice" for
    /// 4-cycle.
    #[test]
    fn selectivity_pass_4cycle_two_snapshots_produce_different_groupings() {
        let mut plan_a = synth_4cycle_plan();
        let stats_a = seed_4cycle_stats(10, 10_000, 10_000, 10); // Default.
        selectivity_pass::run(&mut plan_a, &stats_a, &std::collections::HashMap::new());
        let g_a = inspect_4cycle_grouping(&plan_a).expect("grouping a");

        let mut plan_b = synth_4cycle_plan();
        let stats_b = seed_4cycle_stats(10_000, 10_000, 10, 10); // Alt.
        selectivity_pass::run(&mut plan_b, &stats_b, &std::collections::HashMap::new());
        let g_b = inspect_4cycle_grouping(&plan_b).expect("grouping b");

        assert_ne!(
            g_a, g_b,
            "two different stats snapshots must produce different 4-cycle groupings; \
             got A = {:?}, B = {:?}",
            g_a, g_b
        );
    }

    /// W2.2 — 4-cycle missing-stats safety floor: any unseeded
    /// relation → body unchanged.
    #[test]
    fn selectivity_pass_4cycle_skips_when_card_missing() {
        let mut plan = synth_4cycle_plan();
        // Only seed 3 of 4.
        let mut stats = StatsManager::new();
        for rid in [RelId(1), RelId(2), RelId(3)] {
            stats.register_relation(rid);
            stats.update_cardinality(rid, 100);
        }
        let before = format!("{:?}", plan.rules_by_scc[0][0].body);
        selectivity_pass::run(&mut plan, &stats, &std::collections::HashMap::new());
        let after = format!("{:?}", plan.rules_by_scc[0][0].body);
        assert_eq!(
            before, after,
            "missing-stats safety floor must leave body unchanged"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::ScalarType;
    use xlog_ir::{ConstValue, ProjectExpr};
    use xlog_stats::ColumnStats;

    fn make_stats_manager() -> Arc<StatsManager> {
        let mut mgr = StatsManager::new();

        // Register test relations with realistic statistics
        mgr.register_relation(RelId(1));
        mgr.update_cardinality(RelId(1), 10_000);
        mgr.update_byte_size(RelId(1), 320_000); // ~32 bytes per row

        mgr.register_relation(RelId(2));
        mgr.update_cardinality(RelId(2), 5_000);
        mgr.update_byte_size(RelId(2), 160_000);

        mgr.register_relation(RelId(3));
        mgr.update_cardinality(RelId(3), 1_000);
        mgr.update_byte_size(RelId(3), 32_000);

        // Add column statistics for relation 1
        let mut col0 = ColumnStats::new(0, ScalarType::I64);
        col0.update_distinct(1000);
        col0.update_range(0, 10000);
        mgr.add_column_stats(RelId(1), col0);

        let mut col1 = ColumnStats::new(1, ScalarType::I64);
        col1.update_distinct(100);
        mgr.add_column_stats(RelId(1), col1);

        Arc::new(mgr)
    }

    #[test]
    fn test_optimizer_new() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        assert_eq!(optimizer.config().dp_threshold, 10);
        assert!(optimizer.config().enable_pushdown);
    }

    #[test]
    fn test_optimizer_with_config() {
        let stats = make_stats_manager();
        let mut config = OptimizerConfig::default();
        config.dp_threshold = 5;
        config.enable_pushdown = false;
        let optimizer = Optimizer::with_config(stats, config);

        assert_eq!(optimizer.config().dp_threshold, 5);
        assert!(!optimizer.config().enable_pushdown);
    }

    #[test]
    fn test_estimate_scan_cost() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let scan = RirNode::Scan { rel: RelId(1) };
        let cost = optimizer.estimate_cost(&scan);

        assert_eq!(cost.rows, 10_000);
        assert!(cost.gpu_mem > 0);
        assert_eq!(cost.transfers, 0); // Data on GPU
    }

    #[test]
    fn test_estimate_scan_cost_unknown_relation() {
        let stats = Arc::new(StatsManager::new());
        let optimizer = Optimizer::new(stats);

        let scan = RirNode::Scan { rel: RelId(999) };
        let cost = optimizer.estimate_cost(&scan);

        // Should use defaults
        assert_eq!(cost.rows, 1000);
    }

    #[test]
    fn test_estimate_filter_cost() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let filter = RirNode::Filter {
            input: Box::new(RirNode::Scan { rel: RelId(1) }),
            predicate: Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Eq,
                right: Box::new(Expr::Const(ConstValue::I64(42))),
            },
        };

        let cost = optimizer.estimate_cost(&filter);

        // Filter should reduce row count
        assert!(cost.rows < 10_000);
        assert!(cost.rows >= 1);
    }

    #[test]
    fn test_estimate_join_cost() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let join = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };

        let cost = optimizer.estimate_cost(&join);

        // Should have positive estimates
        assert!(cost.rows > 0);
        assert!(cost.cpu_cost > 0.0);
        assert!(cost.gpu_mem > 0);
    }

    #[test]
    fn test_estimate_join_cost_with_selectivity() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));
        mgr.update_cardinality(RelId(1), 1000);
        mgr.update_cardinality(RelId(2), 500);

        // Record a join result to cache selectivity
        mgr.record_join_result(RelId(1), RelId(2), vec![0], vec![0], 500_000, 2500);

        let optimizer = Optimizer::new(Arc::new(mgr));

        let join = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };

        let cost = optimizer.estimate_cost(&join);

        // Should use cached selectivity for estimate
        assert!(cost.rows > 0);
    }

    #[test]
    fn test_predicate_pushdown_simple_scan() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let scan = RirNode::Scan { rel: RelId(1) };
        let optimized = optimizer.optimize(scan);

        // Scan should pass through unchanged
        assert!(matches!(optimized, RirNode::Scan { rel: RelId(1) }));
    }

    #[test]
    fn test_predicate_pushdown_filter_on_scan() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let filter = RirNode::Filter {
            input: Box::new(RirNode::Scan { rel: RelId(1) }),
            predicate: Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Eq,
                right: Box::new(Expr::Const(ConstValue::I64(42))),
            },
        };

        let optimized = optimizer.optimize(filter);

        // Filter on scan should stay in place
        assert!(matches!(optimized, RirNode::Filter { .. }));
    }

    #[test]
    fn test_predicate_pushdown_merges_filters() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let nested_filter = RirNode::Filter {
            input: Box::new(RirNode::Filter {
                input: Box::new(RirNode::Scan { rel: RelId(1) }),
                predicate: Expr::Compare {
                    left: Box::new(Expr::Column(0)),
                    op: CompareOp::Gt,
                    right: Box::new(Expr::Const(ConstValue::I64(0))),
                },
            }),
            predicate: Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Lt,
                right: Box::new(Expr::Const(ConstValue::I64(100))),
            },
        };

        let optimized = optimizer.optimize(nested_filter);

        // Filters should be merged into AND
        if let RirNode::Filter { predicate, .. } = optimized {
            assert!(matches!(predicate, Expr::And(_)));
        } else {
            panic!("Expected Filter node");
        }
    }

    #[test]
    fn test_predicate_pushdown_through_project() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        // Filter on projected column that's a pass-through
        let plan = RirNode::Filter {
            input: Box::new(RirNode::Project {
                input: Box::new(RirNode::Scan { rel: RelId(1) }),
                columns: vec![ProjectExpr::Column(0), ProjectExpr::Column(1)],
            }),
            predicate: Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Eq,
                right: Box::new(Expr::Const(ConstValue::I64(42))),
            },
        };

        let optimized = optimizer.optimize(plan);

        // Filter should be pushed below project
        assert!(matches!(optimized, RirNode::Project { .. }));
        if let RirNode::Project { input, .. } = optimized {
            assert!(matches!(*input, RirNode::Filter { .. }));
        }
    }

    #[test]
    fn test_predicate_pushdown_into_join() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        // Filter on left side column only
        let plan = RirNode::Filter {
            input: Box::new(RirNode::Join {
                left: Box::new(RirNode::Scan { rel: RelId(1) }),
                right: Box::new(RirNode::Scan { rel: RelId(2) }),
                left_keys: vec![0],
                right_keys: vec![0],
                join_type: JoinType::Inner,
            }),
            predicate: Expr::Compare {
                left: Box::new(Expr::Column(0)), // Left side column
                op: CompareOp::Eq,
                right: Box::new(Expr::Const(ConstValue::I64(42))),
            },
        };

        let optimized = optimizer.optimize(plan);

        // Filter should be pushed into left side of join
        if let RirNode::Join { left, .. } = optimized {
            assert!(matches!(*left, RirNode::Filter { .. }));
        } else {
            panic!("Expected Join node");
        }
    }

    #[test]
    fn test_plan_cost_total() {
        let cost = PlanCost {
            rows: 1000,
            cpu_cost: 100.0,
            gpu_mem: 1_000_000,
            transfers: 2,
        };

        let total = cost.total_cost(100.0);

        // cpu_cost + gpu_mem*0.001 + transfers*100
        // 100.0 + 1000.0 + 200.0 = 1300.0
        assert!((total - 1300.0).abs() < 0.001);
    }

    #[test]
    fn test_plan_cost_then() {
        let cost1 = PlanCost {
            rows: 1000,
            cpu_cost: 50.0,
            gpu_mem: 500,
            transfers: 1,
        };

        let cost2 = PlanCost {
            rows: 500,
            cpu_cost: 25.0,
            gpu_mem: 800,
            transfers: 1,
        };

        let combined = cost1.then(cost2);

        assert_eq!(combined.rows, 500); // Takes output rows from second
        assert_eq!(combined.cpu_cost, 75.0);
        assert_eq!(combined.gpu_mem, 800); // Peak memory
        assert_eq!(combined.transfers, 2);
    }

    #[test]
    fn test_optimizer_config_default() {
        let config = OptimizerConfig::default();

        assert_eq!(config.dp_threshold, 10);
        assert!((config.index_heat_threshold - 0.7).abs() < 0.001);
        assert!(config.enable_pushdown);
        assert!((config.default_filter_selectivity - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_should_use_greedy() {
        let stats = make_stats_manager();
        let mut config = OptimizerConfig::default();
        config.dp_threshold = 2;
        let optimizer = Optimizer::with_config(stats, config);

        // Single relation: should NOT use greedy
        let single = RirNode::Scan { rel: RelId(1) };
        assert!(!optimizer.should_use_greedy(&single));

        // Three relations: should use greedy (threshold is 2)
        let multi = RirNode::Join {
            left: Box::new(RirNode::Join {
                left: Box::new(RirNode::Scan { rel: RelId(1) }),
                right: Box::new(RirNode::Scan { rel: RelId(2) }),
                left_keys: vec![0],
                right_keys: vec![0],
                join_type: JoinType::Inner,
            }),
            right: Box::new(RirNode::Scan { rel: RelId(3) }),
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        assert!(optimizer.should_use_greedy(&multi));
    }

    #[test]
    fn test_recommend_indexes() {
        let mut mgr = StatsManager::new();
        mgr.register_relation(RelId(1));
        mgr.register_relation(RelId(2));

        // Heat up relation 1 extensively
        for _ in 0..50 {
            mgr.record_access(RelId(1));
        }

        let optimizer = Optimizer::new(Arc::new(mgr));
        let recommendations = optimizer.recommend_indexes();

        assert!(recommendations.contains(&RelId(1)));
        assert!(!recommendations.contains(&RelId(2)));
    }

    #[test]
    fn test_estimate_groupby_cost() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let groupby = RirNode::GroupBy {
            input: Box::new(RirNode::Scan { rel: RelId(1) }),
            key_cols: vec![0],
            aggs: vec![(1, xlog_core::AggOp::Sum)],
        };

        let cost = optimizer.estimate_cost(&groupby);

        // GroupBy should reduce row count
        assert!(cost.rows < 10_000);
        assert!(cost.rows >= 1);
    }

    #[test]
    fn test_estimate_union_cost() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let union = RirNode::Union {
            inputs: vec![
                RirNode::Scan { rel: RelId(1) },
                RirNode::Scan { rel: RelId(2) },
            ],
        };

        let cost = optimizer.estimate_cost(&union);

        // Union sums row counts
        assert_eq!(cost.rows, 15_000); // 10000 + 5000
    }

    #[test]
    fn test_estimate_distinct_cost() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let distinct = RirNode::Distinct {
            input: Box::new(RirNode::Scan { rel: RelId(1) }),
            key_cols: vec![0],
        };

        let cost = optimizer.estimate_cost(&distinct);

        // Distinct reduces rows
        assert!(cost.rows <= 10_000);
        assert!(cost.rows >= 1);
    }

    #[test]
    fn test_estimate_diff_cost() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let diff = RirNode::Diff {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
        };

        let cost = optimizer.estimate_cost(&diff);

        // Diff reduces left side
        assert!(cost.rows <= 10_000);
        assert!(cost.rows >= 1);
    }

    #[test]
    fn test_estimate_fixpoint_cost() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let fixpoint = RirNode::Fixpoint {
            scc_id: 0,
            base: Box::new(RirNode::Scan { rel: RelId(1) }),
            recursive: Box::new(RirNode::Scan { rel: RelId(1) }),
            delta_rel: RelId(10),
            full_rel: RelId(11),
        };

        let cost = optimizer.estimate_cost(&fixpoint);

        // Fixpoint accumulates rows across iterations
        assert!(cost.rows >= 10_000);
    }

    #[test]
    fn test_predicate_selectivity_equality() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let scan = RirNode::Scan { rel: RelId(1) };

        // Equality predicate
        let eq_pred = Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Eq,
            right: Box::new(Expr::Const(ConstValue::I64(42))),
        };

        let selectivity = optimizer.estimate_predicate_selectivity(&eq_pred, &scan);

        // With 1000 distinct values, selectivity should be ~0.001
        assert!(selectivity < 0.01);
        assert!(selectivity > 0.0);
    }

    #[test]
    fn test_predicate_selectivity_and() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let scan = RirNode::Scan { rel: RelId(1) };

        // AND of two predicates
        let and_pred = Expr::And(vec![
            Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Gt,
                right: Box::new(Expr::Const(ConstValue::I64(0))),
            },
            Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Lt,
                right: Box::new(Expr::Const(ConstValue::I64(100))),
            },
        ]);

        let selectivity = optimizer.estimate_predicate_selectivity(&and_pred, &scan);

        // Product of individual selectivities (0.33 * 0.33 ≈ 0.11)
        assert!(selectivity < 0.5);
        assert!(selectivity > 0.0);
    }

    #[test]
    fn test_predicate_selectivity_not() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let scan = RirNode::Scan { rel: RelId(1) };

        // NOT of equality
        let not_pred = Expr::Not(Box::new(Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Eq,
            right: Box::new(Expr::Const(ConstValue::I64(42))),
        }));

        let selectivity = optimizer.estimate_predicate_selectivity(&not_pred, &scan);

        // NOT(equality) should have high selectivity
        assert!(selectivity > 0.9);
    }

    #[test]
    fn test_join_type_semi() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let semi_join = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: JoinType::Semi,
        };

        let cost = optimizer.estimate_cost(&semi_join);

        // Semi join outputs at most left side rows
        assert!(cost.rows <= 10_000);
    }

    #[test]
    fn test_join_type_anti() {
        let stats = make_stats_manager();
        let optimizer = Optimizer::new(stats);

        let anti_join = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: JoinType::Anti,
        };

        let cost = optimizer.estimate_cost(&anti_join);

        // Anti join outputs at most left side rows
        assert!(cost.rows <= 10_000);
    }

    #[test]
    fn test_pushdown_disabled() {
        let stats = make_stats_manager();
        let mut config = OptimizerConfig::default();
        config.enable_pushdown = false;
        let optimizer = Optimizer::with_config(stats, config);

        // Filter that could be pushed
        let plan = RirNode::Filter {
            input: Box::new(RirNode::Filter {
                input: Box::new(RirNode::Scan { rel: RelId(1) }),
                predicate: Expr::Compare {
                    left: Box::new(Expr::Column(0)),
                    op: CompareOp::Gt,
                    right: Box::new(Expr::Const(ConstValue::I64(0))),
                },
            }),
            predicate: Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Lt,
                right: Box::new(Expr::Const(ConstValue::I64(100))),
            },
        };

        let optimized = optimizer.optimize(plan.clone());

        // With pushdown disabled, structure should remain the same
        // (outer filter, inner filter, scan)
        if let RirNode::Filter { input, .. } = optimized {
            assert!(matches!(*input, RirNode::Filter { .. }));
        } else {
            panic!("Expected Filter node");
        }
    }

    #[test]
    fn test_collect_columns() {
        let expr = Expr::And(vec![
            Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Eq,
                right: Box::new(Expr::Column(2)),
            },
            Expr::Compare {
                left: Box::new(Expr::Column(1)),
                op: CompareOp::Gt,
                right: Box::new(Expr::Const(ConstValue::I64(0))),
            },
        ]);

        let cols = Optimizer::collect_columns(&expr);

        assert!(cols.contains(&0));
        assert!(cols.contains(&1));
        assert!(cols.contains(&2));
    }

    #[test]
    fn test_flatten_and() {
        let nested = Expr::And(vec![
            Expr::And(vec![
                Expr::Compare {
                    left: Box::new(Expr::Column(0)),
                    op: CompareOp::Eq,
                    right: Box::new(Expr::Const(ConstValue::I64(1))),
                },
                Expr::Compare {
                    left: Box::new(Expr::Column(1)),
                    op: CompareOp::Eq,
                    right: Box::new(Expr::Const(ConstValue::I64(2))),
                },
            ]),
            Expr::Compare {
                left: Box::new(Expr::Column(2)),
                op: CompareOp::Eq,
                right: Box::new(Expr::Const(ConstValue::I64(3))),
            },
        ]);

        let flattened = Optimizer::flatten_and(&nested);

        assert_eq!(flattened.len(), 3);
    }

    #[test]
    fn test_conjoin_single() {
        let single = vec![Expr::Compare {
            left: Box::new(Expr::Column(0)),
            op: CompareOp::Eq,
            right: Box::new(Expr::Const(ConstValue::I64(42))),
        }];

        let result = Optimizer::conjoin(single);

        assert!(matches!(result, Expr::Compare { .. }));
    }

    #[test]
    fn test_conjoin_multiple() {
        let multiple = vec![
            Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Eq,
                right: Box::new(Expr::Const(ConstValue::I64(1))),
            },
            Expr::Compare {
                left: Box::new(Expr::Column(1)),
                op: CompareOp::Eq,
                right: Box::new(Expr::Const(ConstValue::I64(2))),
            },
        ];

        let result = Optimizer::conjoin(multiple);

        assert!(matches!(result, Expr::And(_)));
    }

    #[test]
    fn test_predicate_pushdown_with_schemas() {
        // Regression test: ensure predicate pushdown uses schemas for accurate width estimation.
        // Without schemas, the optimizer could incorrectly remap column indices.
        let stats = make_stats_manager();
        let mut optimizer = Optimizer::new(stats);

        // Set up schemas: left has 3 columns, right has 3 columns
        let left_schema = Schema::new(vec![
            ("c0".to_string(), xlog_core::ScalarType::Symbol),
            ("c1".to_string(), xlog_core::ScalarType::Symbol),
            ("c2".to_string(), xlog_core::ScalarType::Symbol),
        ]);
        let right_schema = Schema::new(vec![
            ("c0".to_string(), xlog_core::ScalarType::Symbol),
            ("c1".to_string(), xlog_core::ScalarType::Symbol),
            ("c2".to_string(), xlog_core::ScalarType::U32),
        ]);

        let mut schemas = HashMap::new();
        schemas.insert(RelId(1), left_schema);
        schemas.insert(RelId(2), right_schema);
        optimizer.set_schemas(schemas);

        // Filter on Column(5) which is in the right side (left_width=3, so column 5-3=2 in right)
        let plan = RirNode::Filter {
            input: Box::new(RirNode::Join {
                left: Box::new(RirNode::Scan { rel: RelId(1) }),
                right: Box::new(RirNode::Scan { rel: RelId(2) }),
                left_keys: vec![0],
                right_keys: vec![0],
                join_type: JoinType::Inner,
            }),
            predicate: Expr::Compare {
                left: Box::new(Expr::Column(5)), // Right side column (index 5 = 3 + 2)
                op: CompareOp::Ge,
                right: Box::new(Expr::Const(ConstValue::U32(4))),
            },
        };

        let optimized = optimizer.optimize(plan);

        // Filter should be pushed into right side of join with Column(2) (remapped from 5-3=2)
        if let RirNode::Join { right, .. } = optimized {
            if let RirNode::Filter { predicate, .. } = *right {
                if let Expr::Compare { left, .. } = predicate {
                    if let Expr::Column(idx) = *left {
                        assert_eq!(
                            idx, 2,
                            "Column should be remapped to 2 (5 - left_width(3) = 2)"
                        );
                    } else {
                        panic!("Expected Column expression");
                    }
                } else {
                    panic!("Expected Compare predicate");
                }
            } else {
                panic!("Expected Filter on right side of join");
            }
        } else {
            panic!("Expected Join node");
        }
    }

    /// v0.6.5 slice 1: optimizer arms for `MultiWayJoin`.
    ///
    /// The promoter runs after `Optimizer::optimize` in `Compiler`, so
    /// these arms are unreachable in production. They exist for compile
    /// safety and to pin the documented semantics: `optimize` returns
    /// the node unchanged, `estimate_width` reports the head arity from
    /// `output_columns`, `estimate_cost` is the sum of input costs, and
    /// `find_column_relation` returns `None` (per slice 1 guardrail).
    ///
    /// v0.6.5 slice 2 (D5) extends each test below to also exercise a
    /// synthesized 4-input `MultiWayJoin` via [`build_4input_multiway`].
    /// This pins shape-agnosticism: the arms must NOT hard-code
    /// `inputs.len() == 3` or `output_columns.len() == 3`. Slice 2a
    /// (4-way) will produce real 4-input bodies through the promoter;
    /// these tests are the load-bearing guard against silent regression.
    fn build_canonical_triangle_multiway() -> RirNode {
        let scan_xy = RirNode::Scan { rel: RelId(1) };
        let scan_yz = RirNode::Scan { rel: RelId(2) };
        let scan_xz = RirNode::Scan { rel: RelId(3) };
        let inner_join = RirNode::Join {
            left: Box::new(scan_xy.clone()),
            right: Box::new(scan_yz.clone()),
            left_keys: vec![1],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        let outer_join = RirNode::Join {
            left: Box::new(inner_join),
            right: Box::new(scan_xz.clone()),
            left_keys: vec![0, 3],
            right_keys: vec![0, 1],
            join_type: JoinType::Inner,
        };
        let fallback = RirNode::Project {
            input: Box::new(outer_join),
            columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
            ],
        };
        RirNode::MultiWayJoin {
            inputs: vec![scan_xy, scan_yz, scan_xz],
            slot_vars: vec![
                vec![Some(0), Some(1)],
                vec![Some(1), Some(2)],
                vec![Some(0), Some(2)],
            ],
            output_columns: vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(3),
            ],
            fallback: Box::new(fallback),
            plan: None,
            var_order: None,
        }
    }

    /// v0.6.5 slice 2 (D5): synthesized 4-input `MultiWayJoin` for
    /// shape-agnosticism testing. Slice 1's promoter is triangle-only,
    /// so this shape never reaches `Optimizer` through the production
    /// pipeline; the tests below exercise the optimizer arms directly.
    ///
    /// Inputs reuse `RelId(1, 2, 3, 1)` — RelId(1) repeats — so the
    /// stats manager registered in `make_stats_manager` covers all
    /// four scans. Cost floor is `2*10_000 + 5_000 + 1_000 = 26_000`.
    fn build_4input_multiway() -> RirNode {
        let scans = [RelId(1), RelId(2), RelId(3), RelId(1)]
            .map(|rel| RirNode::Scan { rel })
            .to_vec();
        // 4-cycle slot_vars [[A,B],[B,C],[C,D],[A,D]].
        let slot_vars = vec![
            vec![Some(0u32), Some(1)],
            vec![Some(1u32), Some(2)],
            vec![Some(2u32), Some(3)],
            vec![Some(0u32), Some(3)],
        ];
        // 4-arity head projection (no real semantic meaning — the
        // synthesized fallback is a stub).
        let output_columns = vec![
            ProjectExpr::Column(0),
            ProjectExpr::Column(1),
            ProjectExpr::Column(2),
            ProjectExpr::Column(3),
        ];
        // Stub fallback: the optimizer arms do not execute fallback,
        // so any RirNode is fine. Use Unit to keep the fixture small.
        let fallback = RirNode::Unit;
        RirNode::MultiWayJoin {
            inputs: scans,
            slot_vars,
            output_columns,
            fallback: Box::new(fallback),
            plan: None,
            var_order: None,
        }
    }

    #[test]
    fn optimize_returns_multiway_unchanged() {
        let optimizer = Optimizer::new(make_stats_manager());
        for node in [build_canonical_triangle_multiway(), build_4input_multiway()] {
            let optimized = optimizer.optimize(node.clone());
            match (&node, &optimized) {
                (
                    RirNode::MultiWayJoin {
                        inputs: a_in,
                        output_columns: a_out,
                        ..
                    },
                    RirNode::MultiWayJoin {
                        inputs: b_in,
                        output_columns: b_out,
                        ..
                    },
                ) => {
                    assert_eq!(a_in.len(), b_in.len());
                    assert_eq!(a_out.len(), b_out.len());
                }
                _ => panic!("optimize() must return a MultiWayJoin"),
            }
        }
    }

    #[test]
    fn estimate_width_uses_output_columns_arity() {
        let optimizer = Optimizer::new(make_stats_manager());
        // Canonical triangle: 3 head columns.
        assert_eq!(
            optimizer.estimate_width(&build_canonical_triangle_multiway()),
            3
        );
        // 4-input synthesized: 4 head columns. Locks shape-
        // agnosticism — the arm must use output_columns.len(),
        // not a hard-coded 3.
        assert_eq!(optimizer.estimate_width(&build_4input_multiway()), 4);
    }

    #[test]
    fn estimate_cost_sums_input_costs() {
        let optimizer = Optimizer::new(make_stats_manager());

        // Canonical triangle: rels 1, 2, 3 with cardinalities
        // 10_000 + 5_000 + 1_000 = 16_000.
        let cost_tri = optimizer.estimate_cost(&build_canonical_triangle_multiway());
        assert!(
            cost_tri.rows >= 16_000,
            "expected cost.rows >= 16000, got {}",
            cost_tri.rows
        );

        // 4-input synthesized: rels 1, 2, 3, 1 → 2*10_000 + 5_000 +
        // 1_000 = 26_000. The arm sums all four inputs; cost grows.
        // Locks shape-agnosticism — the arm must walk every entry
        // in `inputs`, not a hard-coded 3.
        let cost_4 = optimizer.estimate_cost(&build_4input_multiway());
        assert!(
            cost_4.rows >= 26_000,
            "expected 4-input cost.rows >= 26000, got {}",
            cost_4.rows
        );
        assert!(
            cost_4.rows > cost_tri.rows,
            "4-input cost ({}) must exceed triangle cost ({})",
            cost_4.rows,
            cost_tri.rows
        );
    }

    #[test]
    fn find_column_relation_returns_none_for_multiway() {
        let optimizer = Optimizer::new(make_stats_manager());
        // Per slice 1 guardrail: no column-to-input mapping in this
        // slice. Half-mapped is more dangerous than None. The arm
        // must return None regardless of arity — slice 2 strengthens
        // this to also check the 4-input synthesized shape so a
        // future "let's just return inputs[col_idx % len]" patch
        // gets caught.
        for node in [build_canonical_triangle_multiway(), build_4input_multiway()] {
            for col in 0..node.referenced_relations().len() {
                assert!(
                    optimizer.find_column_relation(&node, col).is_none(),
                    "find_column_relation must return None for any \
                     MultiWayJoin column (col={})",
                    col,
                );
            }
        }
    }
}
