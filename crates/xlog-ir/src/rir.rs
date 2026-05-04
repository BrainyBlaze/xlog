//! Relational IR node definitions

use xlog_core::{AggOp, RelId, ScalarType};

/// Join type variants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Standard inner join
    Inner,
    /// Left outer join
    LeftOuter,
    /// Semi join (exists check)
    Semi,
    /// Anti join (not exists / negation)
    Anti,
}

/// Expression in filter predicates
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Column reference by index
    Column(usize),
    /// Constant value
    Const(ConstValue),
    /// Binary comparison
    Compare {
        /// Left-hand side expression.
        left: Box<Expr>,
        /// Comparison operator.
        op: CompareOp,
        /// Right-hand side expression.
        right: Box<Expr>,
    },
    /// Logical AND
    And(Vec<Expr>),
    /// Logical OR
    Or(Vec<Expr>),
    /// Logical NOT
    Not(Box<Expr>),

    // Arithmetic operations
    /// Addition
    Add(Box<Expr>, Box<Expr>),
    /// Subtraction
    Sub(Box<Expr>, Box<Expr>),
    /// Multiplication
    Mul(Box<Expr>, Box<Expr>),
    /// Division
    Div(Box<Expr>, Box<Expr>),
    /// Modulo
    Mod(Box<Expr>, Box<Expr>),

    // Built-in functions
    /// Absolute value
    Abs(Box<Expr>),
    /// Minimum of two values
    Min(Box<Expr>, Box<Expr>),
    /// Maximum of two values
    Max(Box<Expr>, Box<Expr>),
    /// Power (base, exponent)
    Pow(Box<Expr>, Box<Expr>),
    /// Type cast
    Cast(Box<Expr>, ScalarType),

    /// Conditional expression: if condition then then_expr else else_expr
    /// The condition is a boolean comparison expression.
    /// Used for UDF conditionals like: if X >= 100 then 1 else 2
    Conditional {
        /// Boolean condition (should evaluate to bool)
        condition: Box<Expr>,
        /// Expression to evaluate when condition is true
        then_expr: Box<Expr>,
        /// Expression to evaluate when condition is false
        else_expr: Box<Expr>,
    },
}

/// Projection expression -- either a pass-through column reference or a computed value.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectExpr {
    /// Pass through column at given index.
    Column(usize),
    /// Compute an expression whose result has the given scalar type.
    Computed(Expr, ScalarType),
}

/// Per-lookup-input permutation for W2.1 variable-ordering.
///
/// When a non-default leader is chosen, the dispatcher rotates kernel
/// inputs and may swap the two columns of selected lookup atoms (triangle
/// only — the 4-cycle has rotational symmetry and never needs col-swap).
/// `swap_cols == true` means the dispatcher must materialize an owned
/// 2-col view with cols swapped before calling the layout helper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupPerm {
    /// Index into the leader-rotated input list (NOT the original
    /// promoter input list). `0` is the leader slot.
    pub input_idx: u8,
    /// Whether to swap col0 ↔ col1 on this input before the layout
    /// helper sees it.
    pub swap_cols: bool,
}

/// Variable-ordering decision attached to a `MultiWayJoin`.
///
/// `None` on the parent variant preserves slice 1/2/4/W2.2 dispatch
/// behavior bit-identically (default leader, no col-swap, no kernel
/// projection — `output_columns` carries the binary-fallback projection
/// as before).
///
/// When `Some`, the dispatcher consumes `leader_idx` to rotate the
/// kernel `inputs`, applies any `lookup_perms` col-swaps, and
/// post-projects the kernel-direct output buffer through
/// `kernel_output_cols`. `MultiWayJoin::output_columns` stays untouched
/// so binary-fallback consumers continue reading it directly.
#[derive(Debug, Clone, PartialEq)]
pub struct VariableOrder {
    /// Selected leader's index in the canonical promoter input order
    /// (e.g., for triangle: 0=e_xy, 1=e_yz, 2=e_xz). `0` reproduces
    /// the default leader.
    pub leader_idx: u8,
    /// One entry per non-leader lookup input, in dispatcher slot order.
    pub lookup_perms: Vec<LookupPerm>,
    /// Permutation applied to the kernel-direct output buffer to
    /// produce head-ordered columns. For default leader this would be
    /// identity but the field is omitted (`var_order = None`) — slice
    /// 1/2 keeps using `MultiWayJoin::output_columns` directly.
    pub kernel_output_cols: Vec<ProjectExpr>,
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    /// Equal (`==`)
    Eq,
    /// Not equal (`!=`)
    Ne,
    /// Less than (`<`)
    Lt,
    /// Less than or equal (`<=`)
    Le,
    /// Greater than (`>`)
    Gt,
    /// Greater than or equal (`>=`)
    Ge,
}

/// Constant values in expressions
#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    /// Unsigned 32-bit integer constant.
    U32(u32),
    /// Unsigned 64-bit integer constant.
    U64(u64),
    /// Signed 32-bit integer constant.
    I32(i32),
    /// Signed 64-bit integer constant.
    I64(i64),
    /// 32-bit float constant.
    F32(f32),
    /// 64-bit float constant.
    F64(f64),
    /// Boolean constant.
    Bool(bool),
    /// Interned symbol string constant.
    Symbol(String),
}

/// Relational IR node types
#[derive(Debug, Clone)]
pub enum RirNode {
    /// A 0-arity relation containing exactly one empty tuple ({()}).
    ///
    /// This is the identity element for joins and the natural seed for rules whose bodies
    /// contain no positive atoms (e.g. `p() :- not q().`), allowing negation-only rules to
    /// be lowered as set difference against a unit relation.
    Unit,

    /// Scan a base relation
    Scan {
        /// Relation identifier to scan.
        rel: RelId,
    },

    /// Filter rows by predicate
    Filter {
        /// Input relation subtree to filter.
        input: Box<RirNode>,
        /// Boolean predicate applied to each row.
        predicate: Expr,
    },

    /// Project specific columns (pass-through or computed)
    Project {
        /// Input relation subtree to project from.
        input: Box<RirNode>,
        /// Output projection expressions in result-column order.
        columns: Vec<ProjectExpr>,
    },

    /// Join two relations
    Join {
        /// Left-hand input relation.
        left: Box<RirNode>,
        /// Right-hand input relation.
        right: Box<RirNode>,
        /// Column indices from the left input used as join keys.
        left_keys: Vec<usize>,
        /// Column indices from the right input used as join keys.
        right_keys: Vec<usize>,
        /// Join semantics to apply.
        join_type: JoinType,
    },

    /// Group by with aggregation
    GroupBy {
        /// Input relation subtree to aggregate.
        input: Box<RirNode>,
        /// Column indices preserved as grouping keys.
        key_cols: Vec<usize>,
        /// (value_column, aggregation_op)
        aggs: Vec<(usize, AggOp)>,
    },

    /// Union multiple inputs
    Union {
        /// Input subtrees whose rows are concatenated together.
        inputs: Vec<RirNode>,
    },

    /// Remove duplicates
    Distinct {
        /// Input relation subtree to deduplicate.
        input: Box<RirNode>,
        /// Column indices defining tuple identity.
        key_cols: Vec<usize>,
    },

    /// Set difference (left - right)
    Diff {
        /// Left-hand input relation.
        left: Box<RirNode>,
        /// Right-hand input relation whose rows are removed from the left input.
        right: Box<RirNode>,
    },

    /// Fixpoint iteration for recursion
    Fixpoint {
        /// SCC identifier
        scc_id: u32,
        /// Base case computation
        base: Box<RirNode>,
        /// Recursive step computation
        recursive: Box<RirNode>,
        /// Relation for delta (new tuples)
        delta_rel: RelId,
        /// Relation for full result
        full_rel: RelId,
    },

    /// A multi-way conjunctive join that the executor MAY dispatch to a
    /// specialized physical operator (e.g. GPU WCOJ). When the dispatch
    /// declines, the executor falls through to `fallback`, which is the
    /// IR-equivalent binary-join plan captured at promotion time.
    ///
    /// **Invariant** (upheld by `xlog-logic::promote::promote_multiway`):
    /// executing `fallback` produces the same row set as a successful
    /// specialized dispatch.
    ///
    /// v0.6.5 slice 1 only emits this for the certified triangle shape;
    /// 4-way and general-arity admission are deferred to later slices.
    ///
    /// # Walker contract
    ///
    /// Generic walkers and visitors that handle `MultiWayJoin` MUST be
    /// shape-agnostic over `inputs`, `slot_vars`, and `output_columns`
    /// — no walker may assume a fixed arity or a specific
    /// variable-class layout. Only matchers/promoters whose name
    /// carries an explicit shape qualifier (e.g.
    /// `match_multiway_triangle`, `try_promote_triangle`) may lock to
    /// a specific shape.
    MultiWayJoin {
        /// Input scans, in physical-plan slot order. For the v0.6.5
        /// initial promoter, this is exactly `[Scan(rel_xy), Scan(rel_yz),
        /// Scan(rel_xz)]` for a recognized triangle. Each input MUST be
        /// `RirNode::Scan { rel }` in v1.
        inputs: Vec<RirNode>,
        /// Per-slot, per-column variable-class id. Same id across slots →
        /// join on that variable. For the canonical triangle this is
        /// `[[Some(0), Some(1)], [Some(1), Some(2)], [Some(0), Some(2)]]`.
        /// `None` is reserved for constant-bound or don't-care columns;
        /// the v1 promoter never emits `None`.
        slot_vars: Vec<Vec<Option<u32>>>,
        /// Output projection in head-tuple order, identical to what the
        /// equivalent `Project { input: Join { ... } }` carries. For the
        /// triangle: `[Column(0), Column(1), Column(3)]`. The executor
        /// re-validates this; a malformed or rotated projection is
        /// treated as ineligible (no dispatch).
        output_columns: Vec<ProjectExpr>,
        /// IR-equivalent binary-join plan. Executed verbatim on dispatch
        /// decline. Captured from the post-optimizer tree by the
        /// promoter; never synthesized.
        fallback: Box<RirNode>,
        /// Optional W2.1 variable-ordering decision.
        ///
        /// `None` preserves slice 1/2/4/W2.2 behavior bit-identically:
        /// dispatcher uses default leader, no col-swap, post-kernel
        /// projection is the existing `output_columns`.
        ///
        /// `Some(VariableOrder)` instructs the dispatcher to rotate
        /// kernel inputs to put `leader_idx` at slot 0, apply
        /// `lookup_perms` col-swaps, and post-project via
        /// `kernel_output_cols`. `output_columns` is NOT consulted on
        /// the W2.1 path; binary-fallback consumers still read it.
        var_order: Option<VariableOrder>,
    },

    /// Tensorized ILP super-graph join. A DLPack mask tensor selects which
    /// (body_i, body_j) → head_k rule combinations are active.
    TensorMaskedJoin {
        /// Name of the mask tensor registered in the runtime.
        mask_name: String,
        /// Arity of the relation schema participating in the tensorized join.
        schema_size: usize,
        /// Left-side join key columns within the body schema.
        left_keys: Vec<usize>,
        /// Right-side join key columns within the body schema.
        right_keys: Vec<usize>,
        /// Mapping from tensor dimension index → (RelId, relation name).
        /// Sorted by RelId for deterministic ordering (RD-36).
        rel_index: Vec<(RelId, String)>,
        /// Head relation name (for store lookup in executor, RD-12).
        head_rel_name: String,
        /// Head relation ID (for optimizer schema lookup, keyed by RelId, RD-27).
        head_rel_id: RelId,
        /// Maximum active rules to process (budget cap, RD-6).
        max_active_rules: usize,
        /// Column indices from the join result to project into the head schema.
        /// Maps head column `i` to join result column `head_projection[i]`.
        /// Join result columns are: [left_col_0..left_col_n, right_col_0..right_col_m].
        head_projection: Vec<usize>,
    },
}

impl RirNode {
    /// Check if this node is a leaf (Scan)
    pub fn is_leaf(&self) -> bool {
        matches!(self, RirNode::Scan { .. })
    }

    /// Get all relation IDs referenced in this subtree
    pub fn referenced_relations(&self) -> Vec<RelId> {
        let mut rels = Vec::new();
        self.collect_relations(&mut rels);
        rels
    }

    fn collect_relations(&self, rels: &mut Vec<RelId>) {
        match self {
            RirNode::Unit => {}
            RirNode::Scan { rel } => rels.push(*rel),
            RirNode::Filter { input, .. } | RirNode::Project { input, .. } => {
                input.collect_relations(rels);
            }
            RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
                left.collect_relations(rels);
                right.collect_relations(rels);
            }
            RirNode::Union { inputs } => {
                for input in inputs {
                    input.collect_relations(rels);
                }
            }
            RirNode::GroupBy { input, .. } | RirNode::Distinct { input, .. } => {
                input.collect_relations(rels);
            }
            RirNode::Fixpoint {
                base,
                recursive,
                delta_rel,
                full_rel,
                ..
            } => {
                base.collect_relations(rels);
                recursive.collect_relations(rels);
                rels.push(*delta_rel);
                rels.push(*full_rel);
            }
            RirNode::TensorMaskedJoin { rel_index, .. } => {
                for (rel_id, _) in rel_index {
                    rels.push(*rel_id);
                }
            }
            RirNode::MultiWayJoin { inputs, .. } => {
                // Recurse into `inputs` only. The `fallback` references
                // the same set by promoter invariant; walking both would
                // double-count.
                for input in inputs {
                    input.collect_relations(rels);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::ScalarType;

    #[test]
    fn test_scan_node() {
        let node = RirNode::Scan { rel: RelId(1) };
        assert!(matches!(node, RirNode::Scan { rel: RelId(1) }));
        assert!(node.is_leaf());
    }

    #[test]
    fn test_join_node() {
        let left = Box::new(RirNode::Scan { rel: RelId(1) });
        let right = Box::new(RirNode::Scan { rel: RelId(2) });
        let join = RirNode::Join {
            left,
            right,
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };
        assert!(matches!(join, RirNode::Join { .. }));
        let rels = join.referenced_relations();
        assert!(rels.contains(&RelId(1)));
        assert!(rels.contains(&RelId(2)));
    }

    #[test]
    fn test_fixpoint_node() {
        let base = Box::new(RirNode::Scan { rel: RelId(1) });
        let recursive = Box::new(RirNode::Scan { rel: RelId(2) });
        let fp = RirNode::Fixpoint {
            scc_id: 0,
            base,
            recursive,
            delta_rel: RelId(3),
            full_rel: RelId(4),
        };
        assert!(matches!(fp, RirNode::Fixpoint { scc_id: 0, .. }));
    }

    #[test]
    fn test_anti_join() {
        let left = Box::new(RirNode::Scan { rel: RelId(1) });
        let right = Box::new(RirNode::Scan { rel: RelId(2) });
        let anti = RirNode::Join {
            left,
            right,
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: JoinType::Anti,
        };
        if let RirNode::Join { join_type, .. } = anti {
            assert_eq!(join_type, JoinType::Anti);
        }
    }

    #[test]
    fn test_expr_arithmetic() {
        let expr = Expr::Add(
            Box::new(Expr::Column(0)),
            Box::new(Expr::Const(ConstValue::I64(1))),
        );
        assert!(matches!(expr, Expr::Add(_, _)));
    }

    #[test]
    fn test_project_expr_computed() {
        let proj = ProjectExpr::Computed(
            Expr::Add(
                Box::new(Expr::Column(0)),
                Box::new(Expr::Const(ConstValue::I64(1))),
            ),
            ScalarType::I64,
        );
        assert!(matches!(proj, ProjectExpr::Computed(_, _)));
    }
}
