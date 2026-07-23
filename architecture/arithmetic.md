# Arithmetic Expressions

How XLOG compiles and executes arithmetic expressions on the GPU, from AST representation through type inference, lowering, and evaluation.

<Note>
For contributors — how XLOG compiles and runs arithmetic internally. If you only
need the surface language (syntax, precedence, built-ins, worked examples), read
the user-facing arithmetic syntax reference in the language reference instead.
</Note>

This page traces one arithmetic expression through the compiler and onto the GPU.
The path has four stages: the parser builds a tree, the type checker assigns a
type to every node, the lowerer turns the tree into a column computation, and the
GPU evaluates it. Each section below covers one stage.

## AST Representation

The parser represents an arithmetic expression as an abstract syntax tree (AST) —
a tree where each node is one operation or value. `ArithExpr` is that tree type,
and `IsExpr` wraps a single `is` binding (a fresh variable plus the expression
that computes it).

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

Before lowering, every expression node gets a scalar type. The rules below are
applied bottom-up over the tree. Most binary operations require both sides to
share a type; a few (like `pow` and `cast`) fix the result type regardless of the
inputs.

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

Lowering turns the type-checked tree into a query-plan node. XLOG's internal
query plan is the relational intermediate representation (RIR) — the form the
compiler works with after parsing and before it hits the GPU. Here the lowerer
converts each `IsExpr` into a computed projection: it appends one new column to
the current plan, and that column holds the value of the expression.

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

Each column in a projection is either passed through unchanged or computed from an
expression. `ProjectExpr` is the enum that captures that choice:

```rust
pub enum ProjectExpr {
    Column(usize),                    // Pass-through column
    Computed(Expr, ScalarType),       // Computed from expression
}
```

## GPU Execution

Arithmetic expressions are evaluated on the GPU through the arithmetic support
inside `CudaKernelProvider` (the component that runs XLOG kernels on CUDA).

### Evaluation Strategy

The expression tree is compiled into column references and operations, its inputs
are staged as GPU buffers, and the CUDA helpers apply the operations to produce
one output column.

```text
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

Arithmetic faults do not raise an error at runtime. Instead they produce special
sentinel values, which flow through like any other data. The table shows what
each fault yields for integer versus float columns.

| Condition | Integer Behavior | Float Behavior |
|-----------|------------------|----------------|
| Division by zero | `INT64_MAX` | `NaN` / `Inf` |
| Overflow | Wraps (standard semantics) | IEEE 754 |

Because faults become values rather than errors, filter them out explicitly when
you need to exclude them:

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

- [GPU Execution](/architecture/gpu-execution) — execution details for computed expressions
- [Query Optimizer](/architecture/query-optimizer) — projection and pushdown behavior
