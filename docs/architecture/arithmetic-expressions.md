# Arithmetic Expressions

This document describes how XLOG compiles and executes arithmetic expressions on the GPU. It is the architectural companion to the user-facing syntax reference in [`docs/language-reference.md`](../language-reference.md) §Arithmetic Expressions — start there if you are looking for `is`-expression syntax, operator precedence, built-in functions, or worked examples. The material below focuses on the AST, type inference, lowering to RIR, and GPU execution strategy.

## AST representation

```rust
/// Arithmetic expression tree produced by the parser.
pub enum ArithExpr {
    Variable(String),
    Integer(i64),
    Float(f64),

    // Binary operations
    Add(Box<ArithExpr>, Box<ArithExpr>),
    Sub(Box<ArithExpr>, Box<ArithExpr>),
    Mul(Box<ArithExpr>, Box<ArithExpr>),
    Div(Box<ArithExpr>, Box<ArithExpr>),
    Mod(Box<ArithExpr>, Box<ArithExpr>),

    // Built-in functions
    Abs(Box<ArithExpr>),
    Min(Box<ArithExpr>, Box<ArithExpr>),
    Max(Box<ArithExpr>, Box<ArithExpr>),
    Pow(Box<ArithExpr>, Box<ArithExpr>),

    // Type cast: cast(expr, target_type)
    Cast(Box<ArithExpr>, ScalarType),

    // Call to a user-defined function
    FuncCall { name: String, args: Vec<ArithExpr> },
}

/// Is-expression for variable binding (rule bodies).
pub struct IsExpr {
    pub target: String,      // Must be fresh (unbound) variable
    pub expr: ArithExpr,
}
```

User-defined function calls (`FuncCall`) are inlined into the caller's expression tree by `crates/xlog-logic/src/expand.rs` **before** lowering, so the RIR never sees a residual `FuncCall` node.

## Type inference rules

| Expression | Rule | Result type |
|------------|------|-------------|
| `X + Y` / `X - Y` / `X * Y` / `X / Y` | `type(X) == type(Y)` required | `type(X)` |
| `X % Y` | Must be integer types | `type(X)` |
| `abs(X)` | Any numeric | `type(X)` |
| `min(X, Y)` / `max(X, Y)` | `type(X) == type(Y)` required | `type(X)` |
| `pow(X, Y)` | Any numeric | `f64` always |
| `cast(X, T)` | Any numeric | `T` |
| Integer literal | — | `i64` |
| Float literal | — | `f64` |

Type inference is a single pass over the AST; a mismatch is reported with a source-located error directing the user to `cast()`. See `crates/xlog-logic/src/typeinfer.rs`.

## Lowering to RIR

The lowerer transforms `IsExpr` into a computed projection extending the current column set:

```rust
fn lower_is_expr(
    &mut self,
    is_expr: &IsExpr,
    input: RirNode,
    var_env: &mut VariableEnv,
) -> Result<RirNode> {
    // 1. Target must be fresh
    if var_env.contains(&is_expr.target) {
        return Err(XlogError::Compilation(format!(
            "Variable {} already bound; 'is' requires fresh variable",
            is_expr.target
        )));
    }

    // 2. Convert expression (all referenced vars must be bound)
    let expr = self.arith_to_expr(&is_expr.expr, var_env)?;

    // 3. Infer result type
    let result_type = self.infer_arith_type(&is_expr.expr, var_env)?;

    // 4. Extend projection
    let new_col = var_env.column_count();
    var_env.bind(&is_expr.target, new_col, result_type);

    Ok(RirNode::Project {
        input: Box::new(input),
        columns: self.extend_projection_with(expr, result_type),
    })
}
```

### ProjectExpr

```rust
pub enum ProjectExpr {
    Column(usize),                    // Pass-through column
    Computed(Expr, ScalarType),       // Computed from expression
}
```

The optimizer runs common-subexpression elimination and predicate pushdown over these `Computed` nodes; UDF calls, now inlined, participate transparently.

## GPU execution

Arithmetic `Expr` nodes compile to GPU kernels via `CudaKernelProvider`'s arithmetic module.

```
ArithExpr tree
    │ (expand.rs inlines UDF calls)
    ▼
Expr (column refs + operations, no FuncCall variant)
    │
    ▼
Materialize intermediate columns as GPU buffers
    │
    ▼
Apply operations using CudaKernelProvider arithmetic helpers
    │
    ▼
Output column appended to the projected CudaBuffer
```

## Error behavior

| Condition | Integer behavior | Float behavior |
|-----------|------------------|----------------|
| Division by zero | Result is `INT64_MAX` (sentinel) | `NaN` or `±Inf` per IEEE 754 |
| Overflow | Wraps | IEEE 754 |

Errors propagate as special values; callers that need to guard against them must filter explicitly:

```xlog
safe_ratio(X, Y, R) :-
    value(X), value(Y),
    Y != 0,
    R is X / Y.
```

See [Float Predicates and IEEE 754 Semantics](../language-reference.md#float-predicates-and-ieee-754-semantics) in the language reference for the float-comparison contract that arithmetic results flow into.

## Scoping

- `is` bindings are **body-only** — they cannot appear in rule heads.
- The target variable must be **fresh** (unbound at the point of the `is`).
- All variables referenced by the right-hand side must already be bound (by prior body atoms, `is` expressions, or UDF argument positions).
- Evaluation order within a rule body is left-to-right.

## See also

- [`docs/language-reference.md`](../language-reference.md) — user-facing syntax, operators, conditionals, worked examples
- [GPU execution](gpu-execution.md) — kernel-level details of mask-DAG filters and GroupBy
- [Query optimizer](query-optimizer.md) — projection and CSE optimization
