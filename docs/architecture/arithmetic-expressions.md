# Arithmetic Expressions

This document describes how XLOG compiles and executes arithmetic expressions on
the GPU. It complements the user-facing syntax reference in
[`../language-reference.md`](../language-reference.md): start there if you need
surface syntax, precedence, built-ins, or worked examples.

## AST Representation

```rust
/// Arithmetic expression tree
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

    // User-defined function call (inlined before lowering)
    FuncCall { name: String, args: Vec<ArithExpr> },
}

/// Is-expression for variable binding
pub struct IsExpr {
    pub target: String,      // Must be fresh (unbound) variable
    pub expr: ArithExpr,
}
```

## Type Inference Rules

| Expression | Rule | Result Type |
|------------|------|-------------|
| `X + Y` | `type(X) == type(Y)` required | `type(X)` |
| `X % Y` | Must be integer types | `type(X)` |
| `abs(X)` | Any numeric | `type(X)` |
| `min(X, Y)` | `type(X) == type(Y)` required | `type(X)` |
| `pow(X, Y)` | Any numeric | `f64` always |
| `cast(X, T)` | Any numeric | `T` |
| Integer literal | - | `i64` |
| Float literal | - | `f64` |

Type mismatches are rejected during inference with a source-located diagnostic
that directs users to `cast()` when needed.

## Lowering to RIR

The lowerer transforms `IsExpr` into computed projections:

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

    // 2. Convert expression (all vars must be bound)
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

## GPU Execution

Arithmetic expressions are evaluated on the GPU through the arithmetic support
inside `CudaKernelProvider`.

### Evaluation Strategy

```
ArithExpr Tree
    │
    ▼
Compile to Expr (column references + operations)
    │
    ▼
Materialize intermediate columns as GPU buffers
    │
    ▼
Apply operations using CudaKernelProvider arithmetic helpers
    │
    ▼
Output column in projected CudaBuffer
```

### Error Handling

| Condition | Integer Behavior | Float Behavior |
|-----------|------------------|----------------|
| Division by zero | `INT64_MAX` | `NaN` / `Inf` |
| Overflow | Wraps (standard semantics) | IEEE 754 |

Errors propagate as special values; use explicit filtering to handle them:

```prolog
safe_ratio(X, Y, R) :-
    value(X), value(Y),
    Y != 0,
    R is X / Y.
```

## Scoping Rules

- `is` bindings are body-only and cannot appear in rule heads.
- The target variable must be fresh at the point of the `is`.
- All variables referenced by the expression must already be bound.

## See Also

- [`../language-reference.md`](../language-reference.md) — syntax, operators, and examples
- [GPU Execution](gpu-execution.md) — execution details for computed expressions
- [Query Optimizer](query-optimizer.md) — projection and pushdown behavior
