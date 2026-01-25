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
        let config = OptimizerConfig {
            dp_threshold: 5,
            enable_pushdown: false,
            ..Default::default()
        };
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
        let config = OptimizerConfig {
            dp_threshold: 2,
            ..Default::default()
        };
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
        let config = OptimizerConfig {
            enable_pushdown: false,
            ..Default::default()
        };
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
}
