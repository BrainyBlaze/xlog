# Arithmetic Expressions Design

**Date:** 2026-01-10
**Status:** Implemented
**Author:** Claude + User collaboration

## Overview

Add arithmetic expression support to xlog using Prolog/Datalog-style `is` syntax, enabling computation within logic programs while maintaining GPU-accelerated execution.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Syntax | `Z is X + Y` (Prolog-style) | Familiar to logic programmers, clear semantics |
| Operations | `+ - * / % abs min max pow cast` | Practical coverage without bloat |
| Type handling | Strict same-type with explicit cast | Catches bugs, no silent precision loss |
| Scoping | Body only, fresh variable binding | Clear left-to-right dataflow |
| GPU execution | Fused into projection kernels | Minimal overhead, no extra launches |
| Error handling | Propagate special values | GPU-friendly, explicit filtering |

## Grammar Extensions

```pest
// Arithmetic operators with precedence
arith_op = { "+" | "-" | "*" | "/" | "%" }

// Built-in functions
builtin_fn = { "abs" | "min" | "max" | "pow" | "cast" }

// Arithmetic expressions (recursive, precedence-aware)
arith_primary = {
    builtin_fn ~ "(" ~ arith_expr ~ ("," ~ (arith_expr | type_spec))* ~ ")" |
    "(" ~ arith_expr ~ ")" |
    variable |
    integer |
    float_num
}

arith_term = { arith_primary ~ (("*" | "/" | "%") ~ arith_primary)* }
arith_expr = { arith_term ~ (("+" | "-") ~ arith_term)* }

// The 'is' construct - binds fresh variable to expression result
is_expr = { variable ~ "is" ~ arith_expr }

// Extended body literal
body_literal = { negated_atom | atom | comparison | is_expr }
```

**Precedence:** `* / %` binds tighter than `+ -`. Parentheses override.

## AST Extensions

```rust
/// Arithmetic expression tree
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
pub struct IsExpr {
    pub target: String,      // Must be fresh (unbound) variable
    pub expr: ArithExpr,
}

/// Extended body literal
pub enum BodyLiteral {
    Positive(Atom),
    Negated(Atom),
    Comparison(Comparison),
    IsExpr(IsExpr),          // NEW
}
```

## IR Extensions (xlog-ir)

```rust
/// Extended expression enum
pub enum Expr {
    Column(usize),
    Const(ConstValue),

    // Existing
    Compare { left: Box<Expr>, op: CompareOp, right: Box<Expr> },
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Not(Box<Expr>),

    // NEW: Arithmetic operations
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),

    // NEW: Built-in functions
    Abs(Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Cast(Box<Expr>, ScalarType),
}

/// Projection expression - column pass-through or computed
pub enum ProjectExpr {
    Column(usize),
    Computed(Expr, ScalarType),
}

/// Extended RirNode
pub enum RirNode {
    // ... existing variants ...

    Project {
        input: Box<RirNode>,
        columns: Vec<ProjectExpr>,  // Now supports computed expressions
    },
}
```

## Lowering Logic

The lowerer transforms `IsExpr` into extended projections:

1. **Validate fresh variable**: Target must NOT be already bound
2. **Convert expression**: All variables in expression must be bound
3. **Infer result type**: Validate operand type compatibility
4. **Extend projection**: Add computed column to output schema

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

## Type Inference Rules

| Expression | Rule | Result Type |
|------------|------|-------------|
| `X + Y` | `type(X) == type(Y)` required | `type(X)` |
| `X % Y` | Must be integer types | `type(X)` |
| `abs(X)` | Any numeric | `type(X)` |
| `min(X, Y)` | `type(X) == type(Y)` required | `type(X)` |
| `pow(X, Y)` | Any numeric | `F64` always |
| `cast(X, T)` | Any numeric | `T` |
| Integer literal | - | `I64` |
| Float literal | - | `F64` |

**Error on mismatch:** Clear message guides users to use `cast()`.

## GPU Execution

Arithmetic is fused into projection kernels using bytecode evaluation:

```cuda
// Expression opcodes
enum ExprOp : uint8_t {
    OP_COL, OP_CONST,
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD,
    OP_ABS, OP_MIN, OP_MAX, OP_POW, OP_CAST,
};

// Stack-based evaluator (no GPU recursion)
__device__ int64_t eval_expr_i64(
    const uint8_t* bytecode,
    const int64_t* constants,
    const int64_t* row,
    int num_cols
) {
    int64_t stack[16];
    int sp = 0;

    for (int pc = 0; bytecode[pc] != 0; pc++) {
        switch (bytecode[pc]) {
            case OP_COL:   stack[sp++] = row[bytecode[++pc]]; break;
            case OP_CONST: stack[sp++] = constants[bytecode[++pc]]; break;
            case OP_ADD:   stack[sp-2] += stack[sp-1]; sp--; break;
            case OP_DIV:
                stack[sp-2] = (stack[sp-1] == 0)
                    ? INT64_MAX
                    : stack[sp-2] / stack[sp-1];
                sp--;
                break;
            // ... other ops
        }
    }
    return stack[0];
}
```

**Error handling:**
- Division by zero: `INT64_MAX` for integers, `NaN`/`Inf` for floats
- Overflow: Wraps (standard integer semantics)

## Examples

### Basic Arithmetic
```datalog
increment(X, Y) :- value(X), Y is X + 1.
```

### Complex Expression
```datalog
distance(A, B, D) :-
    point(A, X1, Y1),
    point(B, X2, Y2),
    DX is X2 - X1,
    DY is Y2 - Y1,
    D is pow(DX * DX + DY * DY, 0.5).
```

### Type Casting
```datalog
ratio(X, R) :-
    count(X, N),
    total(T),
    R is cast(N, f64) / cast(T, f64).
```

### Recursive with Arithmetic (Fibonacci)
```datalog
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

## Files to Modify

| File | Changes |
|------|---------|
| `crates/xlog-logic/src/grammar.pest` | Add arithmetic grammar rules |
| `crates/xlog-logic/src/ast.rs` | Add `ArithExpr`, `IsExpr`, extend `BodyLiteral` |
| `crates/xlog-logic/src/parser.rs` | Parse arithmetic expressions and `is` |
| `crates/xlog-logic/src/lower.rs` | `arith_to_expr`, `infer_arith_type`, `lower_is_expr` |
| `crates/xlog-ir/src/rir.rs` | Add arithmetic to `Expr`, add `ProjectExpr` |
| `crates/xlog-runtime/src/executor.rs` | Execute `ProjectExpr::Computed` |
| `crates/xlog-cuda/src/provider.rs` | Compile expressions to bytecode |
| `crates/xlog-cuda/kernels/project.cu` | NEW: Projection with expression evaluation |

## Testing Strategy

1. **Parser tests**: Precedence, parentheses, all operators, error cases
2. **Type inference tests**: Mismatch detection, cast behavior, pow returns f64
3. **Lowering tests**: Fresh variable requirement, variable binding order
4. **GPU tests**: Division by zero, overflow, all ops for i64 and f64
5. **Integration tests**: Fibonacci, distance calculation, aggregation + arithmetic

## Non-Goals

- Arithmetic in rule heads (use `is` in body)
- String operations (separate feature)
- Automatic type coercion (explicit cast required)
- User-defined functions (future work)
