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
        left: Box<Expr>,
        op: CompareOp,
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

/// Projection expression - either pass-through column or computed value
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectExpr {
    /// Pass through column at given index
    Column(usize),
    /// Compute expression, result has given type
    Computed(Expr, ScalarType),
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Constant values in expressions
#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
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
    Scan { rel: RelId },

    /// Filter rows by predicate
    Filter {
        input: Box<RirNode>,
        predicate: Expr,
    },

    /// Project specific columns (pass-through or computed)
    Project {
        input: Box<RirNode>,
        columns: Vec<ProjectExpr>,
    },

    /// Join two relations
    Join {
        left: Box<RirNode>,
        right: Box<RirNode>,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        join_type: JoinType,
    },

    /// Group by with aggregation
    GroupBy {
        input: Box<RirNode>,
        key_cols: Vec<usize>,
        /// (value_column, aggregation_op)
        aggs: Vec<(usize, AggOp)>,
    },

    /// Union multiple inputs
    Union { inputs: Vec<RirNode> },

    /// Remove duplicates
    Distinct {
        input: Box<RirNode>,
        key_cols: Vec<usize>,
    },

    /// Set difference (left - right)
    Diff {
        left: Box<RirNode>,
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

    /// Tensorized ILP super-graph join. A DLPack mask tensor selects which
    /// (body_i, body_j) → head_k rule combinations are active.
    TensorMaskedJoin {
        mask_name: String,
        schema_size: usize,
        left_keys: Vec<usize>,
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
