//! Variable-ordering interface for multiway-join planning.
//!
//! The variable order is the sequence in which a multiway evaluator
//! binds variables. Different orders produce identical *results* but
//! can vary widely in *cost* (intermediate sizes, work per step). PR
//! 1 defines the trait shape and ships one trivial implementation
//! ([`AppearanceOrder`]) so the rest of the planner has something
//! deterministic to call. Cost-aware implementations slot in here in
//! later PRs without breaking the trait.
//!
//! ## Trait signature rationale
//!
//! [`VariableOrder::order`] takes the full [`HypergraphRule`] (not
//! just a `&[Vertex]`) on purpose: future selectivity-aware
//! implementations need to inspect hyperedge structure to weigh
//! orderings. Taking the whole IR now means PR 1's trivial impl and
//! PR 3's selectivity-aware impl share one signature.

use super::ir::{HypergraphRule, VertexId};
use xlog_core::RelId;
use xlog_ir::rir::HelperSplitSpec;
use xlog_stats::StatsSnapshot;

const DEFAULT_BURIED_SKEW_THRESHOLD: f64 = 3.0;

/// Compute a variable order for a [`HypergraphRule`].
///
/// Returned vectors must:
///   * contain every [`VertexId`] in `hg.vertex_ids()` exactly once,
///   * be deterministic for a given input (same `hg` → same output),
///   * not depend on hidden mutable state (e.g. process-wide RNG).
///
/// Determinism is the contract that lets the explain output be
/// snapshot-tested. Implementations that want randomness should
/// expose a seeded constructor and document the seeding policy.
pub trait VariableOrder {
    /// Stable identifier for this order's strategy. Used by the
    /// explain output (e.g. `"appearance"`, `"selectivity-greedy"`).
    fn name(&self) -> &'static str;

    /// Compute the order. See trait-level contract for invariants.
    fn order(&self, hg: &HypergraphRule) -> Vec<VertexId>;
}

/// Trivial variable order: variables in their first-appearance
/// order across the body. Already the construction order produced
/// by [`HypergraphRule::from_rule`], so this is just an
/// `IntoIterator` over `hg.vertex_ids()`.
///
/// Useful as the default order for tests, and as a baseline that
/// future cost-aware implementations can be compared against.
#[derive(Debug, Clone, Copy, Default)]
pub struct AppearanceOrder;

impl VariableOrder for AppearanceOrder {
    fn name(&self) -> &'static str {
        "appearance"
    }

    fn order(&self, hg: &HypergraphRule) -> Vec<VertexId> {
        hg.vertex_ids().collect()
    }
}

/// Relation-level statistics required by the full-variable WCOJ planner.
///
/// The trait intentionally reads the existing `xlog-stats` snapshot surface
/// instead of introducing a planner-private stats accumulator. Implementations
/// should return `None` for missing or unseeded observations so planning can
/// decline incomplete stats without panicking.
pub trait StatsSource {
    /// Returns the relation cardinality if it is known and nonzero.
    fn relation_cardinality(&self, rel_id: RelId) -> Option<u64>;

    /// Returns the distinct-value estimate for a relation column.
    fn column_ndv(&self, rel_id: RelId, col_idx: usize) -> Option<u64>;

    /// Returns the observed selectivity between two relation columns.
    fn join_selectivity(
        &self,
        left_rel: RelId,
        right_rel: RelId,
        left_col: usize,
        right_col: usize,
    ) -> Option<f64>;

    /// Returns average and maximum prefix degree for a relation column.
    fn prefix_degree(&self, rel_id: RelId, col_idx: usize) -> Option<(f64, f64)>;

    /// Returns heat and skew factor for a relation column.
    fn key_heat(&self, rel_id: RelId, col_idx: usize) -> Option<(f64, f64)>;
}

impl StatsSource for StatsSnapshot {
    fn relation_cardinality(&self, rel_id: RelId) -> Option<u64> {
        let card = self
            .relations
            .iter()
            .find(|rel| rel.rel_id == rel_id)?
            .cardinality;
        (card > 0).then_some(card)
    }

    fn column_ndv(&self, rel_id: RelId, col_idx: usize) -> Option<u64> {
        let ndv = self
            .relations
            .iter()
            .find(|rel| rel.rel_id == rel_id)?
            .get_column(col_idx)?
            .distinct_estimate;
        (ndv > 0).then_some(ndv)
    }

    fn join_selectivity(
        &self,
        left_rel: RelId,
        right_rel: RelId,
        left_col: usize,
        right_col: usize,
    ) -> Option<f64> {
        self.join_selectivities.iter().find_map(|sel| {
            let direct = sel.left_rel == left_rel
                && sel.right_rel == right_rel
                && sel.left_keys.as_slice() == [left_col]
                && sel.right_keys.as_slice() == [right_col];
            let swapped = sel.left_rel == right_rel
                && sel.right_rel == left_rel
                && sel.left_keys.as_slice() == [right_col]
                && sel.right_keys.as_slice() == [left_col];
            (direct || swapped).then_some(sel.selectivity)
        })
    }

    fn prefix_degree(&self, rel_id: RelId, col_idx: usize) -> Option<(f64, f64)> {
        let degree = self
            .relations
            .iter()
            .find(|rel| rel.rel_id == rel_id)?
            .get_prefix_degree(col_idx)?;
        Some((degree.avg_degree, degree.max_degree))
    }

    fn key_heat(&self, rel_id: RelId, col_idx: usize) -> Option<(f64, f64)> {
        let heat = self
            .relations
            .iter()
            .find(|rel| rel.rel_id == rel_id)?
            .get_key_heat(col_idx)?;
        Some((heat.heat, heat.skew_factor))
    }
}

/// Binary relation edge in a planned WCOJ shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KCliqueEdge {
    /// Relation backing this binary edge.
    pub rel_id: RelId,
    /// Left endpoint variable.
    pub left: VertexId,
    /// Right endpoint variable.
    pub right: VertexId,
    /// Column in `rel_id` containing [`KCliqueEdge::left`].
    pub left_col: usize,
    /// Column in `rel_id` containing [`KCliqueEdge::right`].
    pub right_col: usize,
}

impl KCliqueEdge {
    /// True when the two edges share at least one endpoint variable.
    pub fn touches(&self, other: &KCliqueEdge) -> bool {
        self.left == other.left
            || self.left == other.right
            || self.right == other.left
            || self.right == other.right
    }

    fn endpoint_col(&self, vertex: VertexId) -> Option<usize> {
        if self.left == vertex {
            Some(self.left_col)
        } else if self.right == vertex {
            Some(self.right_col)
        } else {
            None
        }
    }

    fn other_endpoint(&self, vertex: VertexId) -> Option<VertexId> {
        if self.left == vertex {
            Some(self.right)
        } else if self.right == vertex {
            Some(self.left)
        } else {
            None
        }
    }
}

/// Shape consumed by the full-variable WCOJ planner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KCliqueShape {
    variable_count: u8,
    edges: Vec<KCliqueEdge>,
}

impl KCliqueShape {
    /// Creates a complete `K_k` binary-edge shape with deterministic relation IDs.
    pub fn complete(variable_count: u8, first_rel_id: RelId) -> Option<Self> {
        valid_variable_count(variable_count)?;
        let mut edges = Vec::new();
        let mut next_rel = first_rel_id.0;

        for left in 0..variable_count {
            for right in (left + 1)..variable_count {
                edges.push(KCliqueEdge {
                    rel_id: RelId(next_rel),
                    left: VertexId(usize::from(left)),
                    right: VertexId(usize::from(right)),
                    left_col: 0,
                    right_col: 1,
                });
                next_rel = next_rel.checked_add(1)?;
            }
        }

        Some(Self {
            variable_count,
            edges,
        })
    }

    /// Creates a K-clique-family shape from explicit relation edges.
    ///
    /// Used by the production promoter after it has validated the lowered RIR
    /// as a complete clique and recovered the actual relation IDs from scan
    /// leaves. The edge list must already be in deterministic shape order.
    pub fn from_edges(variable_count: u8, edges: Vec<KCliqueEdge>) -> Option<Self> {
        valid_variable_count(variable_count)?;
        (!edges.is_empty()).then_some(())?;
        Some(Self {
            variable_count,
            edges,
        })
    }

    /// Creates the W5.2 4-cycle shape used by the predecessor WCOJ path.
    pub fn cycle4(first_rel_id: RelId) -> Option<Self> {
        let variable_count = 4;
        valid_variable_count(variable_count)?;
        let endpoints = [(0, 1), (1, 2), (2, 3), (3, 0)];
        let mut edges = Vec::new();

        for (idx, (left, right)) in endpoints.iter().enumerate() {
            edges.push(KCliqueEdge {
                rel_id: RelId(first_rel_id.0.checked_add(idx as u32)?),
                left: VertexId(*left),
                right: VertexId(*right),
                left_col: 0,
                right_col: 1,
            });
        }

        Some(Self {
            variable_count,
            edges,
        })
    }

    /// Number of variables in the shape.
    pub fn variable_count(&self) -> u8 {
        self.variable_count
    }

    /// Relation edges in canonical shape order.
    pub fn edges(&self) -> &[KCliqueEdge] {
        &self.edges
    }

    fn variables(&self) -> impl Iterator<Item = VertexId> + '_ {
        (0..self.variable_count).map(|idx| VertexId(usize::from(idx)))
    }
}

/// Planned all-variable order and edge permutation for a WCOJ shape.
#[derive(Debug, Clone, PartialEq)]
pub struct FullVariableOrder {
    /// Variable binding order consumed by the future RIR plan.
    pub variable_order: Vec<VertexId>,
    /// Edge order sorted by the time each edge becomes fully bound.
    pub edge_permutation: Vec<usize>,
    /// HyperCube-derived per-variable share allocation.
    pub variable_share_allocation: Vec<VariableShare>,
    /// Cost record used for dispatch-gate certification.
    pub cost_prediction: CostPredictionRecord,
    /// Predicted winner for the measured W5.2-style path comparison.
    pub predicted_winner: PredictedWinner,
    /// Helper-relation split requests for buried inner-variable skew.
    pub helper_split_specs: Vec<HelperSplitSpec>,
}

/// Per-variable share allocated by the planner.
#[derive(Debug, Clone, PartialEq)]
pub struct VariableShare {
    /// Variable receiving this share.
    pub variable: VertexId,
    /// Relative share; larger values receive more block-slice budget.
    pub share: f64,
}

/// Cost-model output for WCOJ-vs-hash prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct CostPredictionRecord {
    /// Estimated WCOJ work under the selected full variable order.
    pub wcoj_cost: f64,
    /// Estimated hash-chain work under the existing fallback path.
    pub hash_cost: f64,
}

/// Predicted lower-cost path for a benchmark cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictedWinner {
    /// GPU WCOJ path is predicted to win.
    WcojPath,
    /// Existing hash-chain path is predicted to win.
    HashPath,
}

/// Plans a full variable order for a K-clique-family WCOJ shape.
///
/// The implementation ports the HoneyComb-style planning idea at the algorithm
/// level: estimate per-variable pessimistic work from cardinality, NDV,
/// selectivity, prefix degree, and key heat; allocate shares for all variables;
/// then emit a deterministic binding order plus an edge permutation. Missing
/// stats return `None` rather than guessing.
pub fn plan_kclique_var_order<S: StatsSource>(
    shape: &KCliqueShape,
    stats: &S,
) -> Option<FullVariableOrder> {
    if shape.edges.is_empty() {
        return None;
    }
    valid_variable_count(shape.variable_count)?;
    require_complete_stats(shape, stats)?;

    let root_scores = root_scores(shape, stats)?;
    let variable_order = greedy_variable_order(shape, &root_scores);
    let edge_permutation = edge_permutation(shape, &variable_order);
    let variable_share_allocation = share_allocation(&root_scores);
    let cost_prediction = cost_prediction(shape, stats, &root_scores)?;
    let predicted_winner = if cost_prediction.wcoj_cost <= cost_prediction.hash_cost {
        PredictedWinner::WcojPath
    } else {
        PredictedWinner::HashPath
    };
    let helper_split_specs = helper_split_specs_for_buried_skew(shape, stats, &variable_order)?;

    Some(FullVariableOrder {
        variable_order,
        edge_permutation,
        variable_share_allocation,
        cost_prediction,
        predicted_winner,
        helper_split_specs,
    })
}

fn valid_variable_count(variable_count: u8) -> Option<()> {
    (2..=8).contains(&variable_count).then_some(())
}

fn require_complete_stats<S: StatsSource>(shape: &KCliqueShape, stats: &S) -> Option<()> {
    for edge in shape.edges() {
        stats.relation_cardinality(edge.rel_id)?;
        checked_ndv(stats, edge.rel_id, edge.left_col)?;
        checked_ndv(stats, edge.rel_id, edge.right_col)?;
        checked_prefix(stats, edge.rel_id, edge.left_col)?;
        checked_prefix(stats, edge.rel_id, edge.right_col)?;
        checked_heat(stats, edge.rel_id, edge.left_col)?;
        checked_heat(stats, edge.rel_id, edge.right_col)?;
    }

    for (left_idx, left_edge) in shape.edges().iter().enumerate() {
        for right_edge in shape.edges().iter().skip(left_idx + 1) {
            if left_edge.touches(right_edge) {
                checked_selectivity(
                    stats,
                    left_edge.rel_id,
                    right_edge.rel_id,
                    left_edge.left_col,
                    right_edge.left_col,
                )?;
            }
        }
    }

    Some(())
}

fn checked_ndv<S: StatsSource>(stats: &S, rel_id: RelId, col_idx: usize) -> Option<u64> {
    let ndv = stats.column_ndv(rel_id, col_idx)?;
    (ndv > 0).then_some(ndv)
}

fn checked_prefix<S: StatsSource>(stats: &S, rel_id: RelId, col_idx: usize) -> Option<(f64, f64)> {
    let (avg, max) = stats.prefix_degree(rel_id, col_idx)?;
    (avg.is_finite() && max.is_finite() && avg > 0.0 && max >= avg).then_some((avg, max))
}

fn checked_heat<S: StatsSource>(stats: &S, rel_id: RelId, col_idx: usize) -> Option<(f64, f64)> {
    let (heat, skew) = stats.key_heat(rel_id, col_idx)?;
    (heat.is_finite() && skew.is_finite() && heat >= 0.0 && skew >= 0.0).then_some((heat, skew))
}

fn checked_selectivity<S: StatsSource>(
    stats: &S,
    left_rel: RelId,
    right_rel: RelId,
    left_col: usize,
    right_col: usize,
) -> Option<f64> {
    let selectivity = stats.join_selectivity(left_rel, right_rel, left_col, right_col)?;
    (selectivity.is_finite() && (0.0..=1.0).contains(&selectivity)).then_some(selectivity)
}

fn root_scores<S: StatsSource>(shape: &KCliqueShape, stats: &S) -> Option<Vec<(VertexId, f64)>> {
    let mut scores = Vec::new();

    for variable in shape.variables() {
        let mut score = 0.0;
        for edge in shape.edges() {
            let Some(col_idx) = edge.endpoint_col(variable) else {
                continue;
            };
            let card = stats.relation_cardinality(edge.rel_id)? as f64;
            let ndv = checked_ndv(stats, edge.rel_id, col_idx)? as f64;
            let (avg_degree, max_degree) = checked_prefix(stats, edge.rel_id, col_idx)?;
            let (heat, skew) = checked_heat(stats, edge.rel_id, col_idx)?;
            let prefix_skew = (max_degree / avg_degree).max(1.0);
            let density = card / ndv;
            score += density
                * avg_degree.powi(2)
                * (1.0 + heat).powi(2)
                * (1.0 + 0.25 * skew)
                * prefix_skew.sqrt();
        }
        scores.push((variable, score.max(f64::EPSILON)));
    }

    scores.sort_by_key(|entry| entry.0);
    Some(scores)
}

fn greedy_variable_order(shape: &KCliqueShape, root_scores: &[(VertexId, f64)]) -> Vec<VertexId> {
    let mut remaining: Vec<VertexId> = root_scores.iter().map(|(var, _)| *var).collect();
    let mut order = Vec::with_capacity(remaining.len());

    while !remaining.is_empty() {
        let best_pos = remaining
            .iter()
            .enumerate()
            .min_by(|(_, left), (_, right)| {
                let left_score = marginal_score(shape, root_scores, **left, &order);
                let right_score = marginal_score(shape, root_scores, **right, &order);
                left_score
                    .total_cmp(&right_score)
                    .then_with(|| left.cmp(right))
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        order.push(remaining.remove(best_pos));
    }

    order
}

fn marginal_score(
    shape: &KCliqueShape,
    root_scores: &[(VertexId, f64)],
    variable: VertexId,
    bound: &[VertexId],
) -> f64 {
    let root = root_scores
        .iter()
        .find(|(var, _)| *var == variable)
        .map(|(_, score)| *score)
        .unwrap_or(f64::MAX);
    let bound_edges = shape
        .edges()
        .iter()
        .filter(|edge| {
            edge.endpoint_col(variable).is_some()
                && edge
                    .other_endpoint(variable)
                    .is_some_and(|other| bound.contains(&other))
        })
        .count() as f64;

    root / (1.0 + bound_edges).powi(2)
}

fn edge_permutation(shape: &KCliqueShape, variable_order: &[VertexId]) -> Vec<usize> {
    let mut positions = vec![0usize; usize::from(shape.variable_count())];
    for (pos, variable) in variable_order.iter().enumerate() {
        positions[variable.0] = pos;
    }

    let mut indexed: Vec<(usize, usize, usize, RelId)> = shape
        .edges()
        .iter()
        .enumerate()
        .map(|(idx, edge)| {
            let left = positions[edge.left.0];
            let right = positions[edge.right.0];
            (idx, left.max(right), left.min(right), edge.rel_id)
        })
        .collect();
    indexed.sort_by_key(|(_, max_pos, min_pos, rel_id)| (*max_pos, *min_pos, *rel_id));
    indexed.into_iter().map(|(idx, _, _, _)| idx).collect()
}

fn helper_split_specs_for_buried_skew<S: StatsSource>(
    shape: &KCliqueShape,
    stats: &S,
    variable_order: &[VertexId],
) -> Option<Vec<HelperSplitSpec>> {
    let leader = *variable_order.first()?;
    let variable_heat = per_variable_heat(shape, stats)?;
    let leader_heat = variable_heat
        .iter()
        .find(|(variable, _)| *variable == leader)
        .map(|(_, heat)| *heat)?
        .max(f64::EPSILON);
    let (hot_variable, hot_heat) = variable_heat
        .iter()
        .copied()
        .filter(|(variable, _)| *variable != leader)
        .max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })?;
    let threshold = buried_skew_threshold();
    if hot_heat / leader_heat < threshold {
        return Some(Vec::new());
    }

    let helper_vertices: Vec<VertexId> = variable_order
        .iter()
        .copied()
        .filter(|variable| *variable != hot_variable)
        .take(2)
        .collect();
    if helper_vertices.len() != 2 {
        return Some(Vec::new());
    }
    let edge_hot_left = clique_edge_idx_for_vars(shape, hot_variable, helper_vertices[0])?;
    let edge_hot_right = clique_edge_idx_for_vars(shape, hot_variable, helper_vertices[1])?;
    let leader_edge = clique_edge_idx_for_vars(shape, helper_vertices[0], helper_vertices[1])?;

    Some(vec![HelperSplitSpec {
        helper_id: 0,
        variable: u8::try_from(hot_variable.0).ok()?,
        edge_slots: vec![
            u8::try_from(edge_hot_left).ok()?,
            u8::try_from(edge_hot_right).ok()?,
            u8::try_from(leader_edge).ok()?,
        ],
    }])
}

fn buried_skew_threshold() -> f64 {
    std::env::var("XLOG_BURIED_SKEW_THRESHOLD")
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(DEFAULT_BURIED_SKEW_THRESHOLD)
}

fn per_variable_heat<S: StatsSource>(
    shape: &KCliqueShape,
    stats: &S,
) -> Option<Vec<(VertexId, f64)>> {
    let mut heats = Vec::new();
    for variable in shape.variables() {
        let mut heat = 0.0f64;
        for edge in shape.edges() {
            let Some(col_idx) = edge.endpoint_col(variable) else {
                continue;
            };
            let (key_heat, skew_factor) = checked_heat(stats, edge.rel_id, col_idx)?;
            heat = heat.max(key_heat.max(skew_factor));
        }
        heats.push((variable, heat.max(f64::EPSILON)));
    }
    Some(heats)
}

fn clique_edge_idx_for_vars(
    shape: &KCliqueShape,
    left: VertexId,
    right: VertexId,
) -> Option<usize> {
    let (left, right) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    shape
        .edges()
        .iter()
        .position(|edge| edge.left == left && edge.right == right)
}

fn share_allocation(root_scores: &[(VertexId, f64)]) -> Vec<VariableShare> {
    let inverse_sum: f64 = root_scores.iter().map(|(_, score)| 1.0 / *score).sum();
    let mut shares: Vec<VariableShare> = root_scores
        .iter()
        .map(|(variable, score)| VariableShare {
            variable: *variable,
            share: (1.0 / *score) / inverse_sum,
        })
        .collect();
    shares.sort_by_key(|share| share.variable);
    shares
}

fn cost_prediction<S: StatsSource>(
    shape: &KCliqueShape,
    stats: &S,
    root_scores: &[(VertexId, f64)],
) -> Option<CostPredictionRecord> {
    let wcoj_cost = root_scores.iter().map(|(_, score)| *score).sum::<f64>();
    let mut hash_cost = 0.0;

    for edge in shape.edges() {
        hash_cost += stats.relation_cardinality(edge.rel_id)? as f64;
    }

    let avg_selectivity = average_touching_selectivity(shape, stats)?;
    let hash_cost = hash_cost * (1.0 + avg_selectivity);

    Some(CostPredictionRecord {
        wcoj_cost,
        hash_cost,
    })
}

fn average_touching_selectivity<S: StatsSource>(shape: &KCliqueShape, stats: &S) -> Option<f64> {
    let mut total = 0.0;
    let mut count = 0usize;

    for (left_idx, left_edge) in shape.edges().iter().enumerate() {
        for right_edge in shape.edges().iter().skip(left_idx + 1) {
            if left_edge.touches(right_edge) {
                total += checked_selectivity(
                    stats,
                    left_edge.rel_id,
                    right_edge.rel_id,
                    left_edge.left_col,
                    right_edge.left_col,
                )?;
                count += 1;
            }
        }
    }

    if count == 0 {
        Some(1.0)
    } else {
        Some(total / count as f64)
    }
}
