# Arithmetic Expressions

This document describes XLOG's arithmetic expression support using Prolog/Datalog-style `is` syntax, enabling computation within logic programs while maintaining GPU-accelerated execution.

## Syntax

XLOG uses the standard Prolog `is` syntax for arithmetic:

```prolog
% Bind fresh variable Z to the result of expression
Z is X + Y
```

The left-hand side must be a **fresh (unbound) variable**. The right-hand side is an arithmetic expression referencing bound variables.

## Supported Operations

| Category | Operations | Notes |
|----------|------------|-------|
| Binary | `+`, `-`, `*`, `/`, `%` | Modulo (`%`) requires integer types |
| Unary | `abs` | Absolute value |
| Binary functions | `min`, `max`, `pow` | `pow` always returns `f64` |
| Type conversion | `cast` | Explicit type casting |

### Operator Precedence

- `*`, `/`, `%` bind tighter than `+`, `-`
- Parentheses override precedence

## Grammar

```pest
arith_op = { "+" | "-" | "*" | "/" | "%" }
builtin_fn = { "abs" | "min" | "max" | "pow" | "cast" }

arith_primary = {
    builtin_fn ~ "(" ~ arith_expr ~ ("," ~ (arith_expr | type_spec))* ~ ")" |
    "(" ~ arith_expr ~ ")" |
    variable |
    integer |
    float_num
}

arith_term = { arith_primary ~ (("*" | "/" | "%") ~ arith_primary)* }
arith_expr = { arith_term ~ (("+" | "-") ~ arith_term)* }

is_expr = { variable ~ "is" ~ arith_expr }
```

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

**Type mismatch**: Clear error message guides users to use `cast()`.

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

Arithmetic expressions are evaluated on the GPU using existing arithmetic kernels.

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

## Examples

### Basic Arithmetic

```prolog
increment(X, Y) :- value(X), Y is X + 1.
```

### Complex Expression

```prolog
distance(A, B, D) :-
    point(A, X1, Y1),
    point(B, X2, Y2),
    DX is X2 - X1,
    DY is Y2 - Y1,
    D is pow(DX * DX + DY * DY, 0.5).
```

### Type Casting

```prolog
ratio(X, R) :-
    count(X, N),
    total(T),
    R is cast(N, f64) / cast(T, f64).
```

### Recursive with Arithmetic (Fibonacci)

```prolog
fib(0, 1).
fib(1, 1).
fib(N, F) :-
    fib(N1, F1),
    fib(N2, F2),
    N is N1 + 1,
    N1 is N2 + 1,
    F is F1 + F2,
    N < 10.
```

## Scoping Rules

- `is` bindings are **body-only** — cannot appear in rule heads
- Target variable must be **fresh** (unbound before the `is`)
- All variables in the expression must be **bound** (from prior atoms or `is` expressions)
- Left-to-right evaluation order within rule body

## Limitations

Not supported:
- Arithmetic in rule heads (use `is` in body instead)
- String operations (separate feature)
- Automatic type coercion (use explicit `cast`)
- User-defined functions (future work)

## See Also

- [GPU Execution](gpu-execution.md) — How arithmetic is evaluated on GPU
- [Query Optimizer](query-optimizer.md) — Projection optimization
