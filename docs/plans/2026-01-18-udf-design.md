# User-Defined Functions Design (v0.3.2)

> **Status:** Approved
> **Author:** Claude + Human
> **Date:** 2026-01-18
> **Target:** v0.3.2 (Language Core)
> **Depends on:** Module System (2.3)

---

## Overview

This document specifies user-defined functions (UDFs) for XLOG, enabling reusable expressions and predicate-based computations. Functions are inline-expanded at compile time and support recursion with safety mechanisms.

---

## Core Syntax

### Function Definition

Functions are defined with `func` keyword. Body can be arithmetic expressions or predicate-based:

```prolog
% Simple arithmetic functions
func square(X) = X * X.
func cube(X) = X * X * X.

% With optional type annotations
func dist(X1: f64, Y1: f64, X2: f64, Y2: f64) -> f64 =
    pow(square(X2 - X1) + square(Y2 - Y1), 0.5).

% Predicate-based functions (inline views)
func get_parent(X) = P :- parent(X, P).
func get_ancestor(X) = A :- ancestor(X, A).  % may return multiple values
```

### Function Calls

Functions can be called two ways:

```prolog
% Via `is` expression (binds to variable)
result(Y) :- input(X), Y is square(X).
close(A, B) :- point(A, X1, Y1), point(B, X2, Y2),
               D is dist(X1, Y1, X2, Y2), D < 10.0.

% Direct in terms (inline)
result(square(X)) :- input(X).
close(A, B) :- point(A, X1, Y1), point(B, X2, Y2),
               dist(X1, Y1, X2, Y2) < 10.0.
```

### Semantics

Functions expand inline at compile time. Predicate-based functions are non-deterministic — they expand to all matching results (like a join):

```prolog
func get_value(K) = V :- mapping(K, V).

% This rule:
result(X) :- X is get_value(1).

% Expands to:
result(X) :- mapping(1, X).
```

---

## Recursive Functions

### Syntax

Recursive functions must use conditional form with explicit base case:

```prolog
% Factorial - base case required
func factorial(N) = if N <= 1 then 1 else N * factorial(N - 1).

% Fibonacci
func fib(N) = if N <= 1 then N else fib(N - 1) + fib(N - 2).

% GCD using recursion
func gcd(A, B) = if B == 0 then A else gcd(B, A % B).
```

### Safety Mechanisms

All three mechanisms are enforced:

**1. Required conditional form** — Compile error if recursive call has no base case:

```
error[E0501]: recursive function without base case
  --> math.xlog:3:1
   |
 3 | func bad(N) = bad(N - 1).
   |      ^^^ recursive call with no termination condition
   |
   = help: use conditional form: `if <base_condition> then <base_value> else <recursive_call>`
```

**2. Static base-case analysis** — Warning if base case seems unreachable:

```
warning[W0502]: potentially infinite recursion
  --> math.xlog:5:1
   |
 5 | func risky(N) = if N == 0 then 1 else risky(N + 1).
   |                                       ^^^^^^^^^^^ N increases, may never reach base case
   |
   = note: base case requires N == 0, but recursive call uses N + 1
```

**3. Runtime depth limit** — Configurable maximum recursion depth:

```prolog
#pragma max_recursion_depth = 1000.   % default is 1000

% At runtime, exceeding depth produces error:
% error: maximum recursion depth (1000) exceeded in function `fib`
```

---

## Type System

### Type Inference

Types are inferred from usage when not annotated:

```prolog
func square(X) = X * X.
% X inferred from call site:
%   square(3)     → X: i64, returns i64
%   square(3.0)   → X: f64, returns f64

func add_one(X) = X + 1.
% Literal 1 is polymorphic, resolved by context
```

### Optional Type Annotations

Explicit types for documentation or disambiguation:

```prolog
% Full annotation
func square(X: f64) -> f64 = X * X.

% Parameter types only (return inferred)
func cube(X: f64) = X * X * X.

% Return type only (parameters inferred)
func double(X) -> i64 = X * 2.
```

### Type Errors

Clear messages when types don't match:

```
error[E0503]: type mismatch in function `square`
  --> math.xlog:5:20
   |
 5 | func square(X: i64) -> f64 = X * X.
   |                    ^^^       ^^^^^ expression has type i64
   |                    |
   |                    expected f64
   |
   = help: add explicit cast: `cast(X * X, f64)`
```

---

## Module Integration

### Visibility

Functions follow the same visibility rules as predicates — public by default:

```prolog
% math.xlog
func square(X) = X * X.                    % public
func dist(X1, Y1, X2, Y2) = ...            % public
private func internal_helper(X) = X + 1.   % private, not importable
```

### Importing Functions

Functions are imported with the same syntax as predicates:

```prolog
% main.xlog
use math.                         % imports all public predicates AND functions
use math::{square, dist}.         % selective import

result(Y) :- input(X), Y is square(X).
```

### Name Conflicts

Same rules as predicates — error on conflict, resolve with selective imports:

```
error[E0402]: ambiguous import `abs`
  --> main.xlog:2:1
   |
 1 | use math.
   |     ---- `abs` first imported here (function)
 2 | use utils.
   |     ^^^^^ `abs` also exported by utils (function)
   |
   = help: use selective imports: `use math::{abs}.`
```

### Function vs Predicate Disambiguation

Functions and predicates share the same namespace within a module. You cannot have both:

```prolog
% ERROR: name collision
func edge(X) = X + 1.
pred edge(u32, u32).   % error: `edge` already defined as function
```

---

## Grammar Changes

Additions to `grammar.pest`:

```pest
// Type annotations for functions
type_annotation = { ":" ~ type_spec }
return_type = { "->" ~ type_spec }

// Function parameters
func_param = { ident ~ type_annotation? }
func_params = { func_param ~ ("," ~ func_param)* }

// Conditional expression (for recursive functions)
cond_expr = { "if" ~ cond_test ~ "then" ~ func_body ~ "else" ~ func_body }
cond_test = { arith_expr ~ cmp_op ~ arith_expr }

// Function body - arithmetic, conditional, or predicate-based
func_body_arith = { cond_expr | arith_expr }
func_body_pred = { variable ~ ":-" ~ body }
func_body = { func_body_pred | func_body_arith }

// Function definition
func_def = {
    private_mod? ~ "func" ~ ident ~ "(" ~ func_params? ~ ")" ~ return_type?
    ~ "=" ~ func_body ~ "."
}

// Function call in expressions (extends arith_primary)
func_call = { ident ~ "(" ~ (arith_expr ~ ("," ~ arith_expr)*)? ~ ")" }
arith_primary = {
    func_call |                  // NEW: user-defined function calls
    builtin_fn ~ "(" ~ ... |
    "(" ~ arith_expr ~ ")" |
    variable |
    float_num |
    integer
}

// Pragma for recursion depth
pragma_max_recursion = { "#pragma" ~ "max_recursion_depth" ~ "=" ~ integer }
pragma = { pragma_prob_engine | pragma_prob_cache | pragma_max_recursion }

// Update statement to include func_def
statement = {
    func_def              // NEW
    | use_stmt
    | domain_decl
    | pred_decl
    | pragma
    | rule_def
    | prob_fact
    | annotated_disjunction
    | evidence_stmt
    | prob_query
    | fact
    | constraint
    | query
}
```

---

## AST Changes

Additions to `ast.rs`:

```rust
/// Function parameter with optional type
#[derive(Debug, Clone, PartialEq)]
pub struct FuncParam {
    pub name: String,
    pub typ: Option<ScalarType>,
}

/// Conditional expression for recursive functions
#[derive(Debug, Clone, PartialEq)]
pub struct CondExpr {
    pub condition: (ArithExpr, CompOp, ArithExpr),
    pub then_branch: Box<FuncBody>,
    pub else_branch: Box<FuncBody>,
}

/// Function body - either arithmetic/conditional or predicate-based
#[derive(Debug, Clone, PartialEq)]
pub enum FuncBody {
    Arithmetic(ArithExpr),
    Conditional(CondExpr),
    Predicate { result: String, body: Vec<BodyLiteral> },
}

/// Function definition
#[derive(Debug, Clone, PartialEq)]
pub struct FuncDef {
    pub name: String,
    pub params: Vec<FuncParam>,
    pub return_type: Option<ScalarType>,
    pub body: FuncBody,
    pub is_private: bool,
    pub is_recursive: bool,  // detected during parsing
}

/// Updated Program
pub struct Program {
    pub imports: Vec<UseDecl>,
    pub functions: Vec<FuncDef>,   // NEW
    pub domains: Vec<DomainDecl>,
    pub predicates: Vec<PredDecl>,
    pub rules: Vec<Rule>,
    pub constraints: Vec<Constraint>,
    pub queries: Vec<Query>,
    pub prob_facts: Vec<ProbFact>,
    pub annotated_disjunctions: Vec<AnnotatedDisjunction>,
    pub evidence: Vec<Evidence>,
    pub prob_queries: Vec<ProbQuery>,
    pub directives: Directives,
}
```

---

## Compilation Pipeline

### Function Registry

New module `function.rs` manages function definitions and expansion:

```rust
pub struct FunctionRegistry {
    functions: HashMap<String, FuncDef>,
    call_graph: HashMap<String, HashSet<String>>,  // for recursion detection
}

impl FunctionRegistry {
    /// Register a function, checking for duplicates
    pub fn register(&mut self, func: FuncDef) -> Result<(), FuncError>;

    /// Check for recursive cycles, validate base cases
    pub fn validate_recursion(&self) -> Result<(), RecursionError>;

    /// Expand a function call to its body (inline)
    pub fn expand(&self, name: &str, args: &[ArithExpr]) -> Result<FuncBody, FuncError>;
}
```

### Inline Expansion

Functions are expanded at lowering time, before IR generation:

```prolog
% Original
func square(X) = X * X.
result(Y) :- input(X), Y is square(X).

% After expansion
result(Y) :- input(X), Y is X * X.
```

Predicate-based functions expand to joins:

```prolog
% Original
func get_parent(C) = P :- parent(C, P).
grandparent(G) :- child(C), P is get_parent(C), G is get_parent(P).

% After expansion
grandparent(G) :- child(C), parent(C, P), parent(P, G).
```

### Recursive Function Handling

Recursive functions expand with depth tracking:

```rust
pub struct RecursionContext {
    depth: usize,
    max_depth: usize,  // from pragma or default 1000
}

impl FunctionRegistry {
    pub fn expand_recursive(
        &self,
        name: &str,
        args: &[ArithExpr],
        ctx: &mut RecursionContext
    ) -> Result<ArithExpr, RecursionError>;
}
```

### Compilation Phases (Updated)

1. **Parse** — Parse functions into AST
2. **Register** — Build function registry, detect recursion
3. **Validate** — Check base cases, type consistency
4. **Resolve** — Module imports (includes functions)
5. **Expand** — Inline all function calls
6. **Lower** — Existing lowering (no functions remain)
7. **Compile** — Unchanged

---

## Testing Strategy

### Test Cases

| Category | Test |
|----------|------|
| Basic arithmetic | `func square(X) = X * X.` expands correctly |
| Nested calls | `func f(X) = g(g(X)).` chains properly |
| Type inference | Untyped function infers from call site |
| Type annotations | Explicit types checked and enforced |
| Predicate body | `func f(X) = Y :- p(X, Y).` expands to join |
| Multi-result | Predicate function returns all matches |
| Conditional | `if-then-else` evaluates correctly |
| Recursion base | `func fib(N) = if N <= 1 then N else ...` terminates |
| Recursion depth | Exceeding max depth produces clear error |
| No base case | Missing base case rejected at compile time |
| Private function | Cannot import `private func` |
| Module import | Functions imported with `use module.` |
| Name conflict | Same function name from two modules errors |
| Direct call | `result(square(X))` syntax works |
| Via `is` | `Y is square(X)` syntax works |

### Test File Structure

```
crates/xlog-logic/tests/
  functions/
    basic.xlog           # simple arithmetic functions
    nested.xlog          # function composition
    typed.xlog           # type annotations
    predicate.xlog       # predicate-based functions
    recursive.xlog       # recursive with base cases
    recursive_bad.xlog   # should fail: no base case
    visibility.xlog      # private functions
    module_a.xlog        # function exports
    module_b.xlog        # function imports
```

### Error Message Tests

```rust
#[test]
fn test_no_base_case_error() {
    let src = r#"func bad(N) = bad(N - 1)."#;
    let err = parse_and_validate(src).unwrap_err();
    assert!(err.contains("recursive function without base case"));
    assert!(err.contains("help: use conditional form"));
}

#[test]
fn test_depth_exceeded_error() {
    let src = r#"
        func deep(N) = if N <= 0 then 0 else deep(N - 1).
        result(X) :- X is deep(10000).
    "#;
    let err = execute(src).unwrap_err();
    assert!(err.contains("maximum recursion depth"));
}
```

---

## Implementation Plan

### Files to Create

| File | Purpose |
|------|---------|
| `crates/xlog-logic/src/function.rs` | `FuncDef`, `FunctionRegistry`, expansion logic |
| `crates/xlog-logic/tests/function_tests.rs` | Function integration tests |
| `crates/xlog-logic/tests/functions/` | Test .xlog files |

### Files to Modify

| File | Changes |
|------|---------|
| `grammar.pest` | Add `func_def`, `func_param`, `cond_expr`, `func_call`, `func_body` |
| `ast.rs` | Add `FuncDef`, `FuncParam`, `FuncBody`, `CondExpr`, update `Program` |
| `parser.rs` | Parse function definitions and calls |
| `lib.rs` | Export `function` module |
| `lower.rs` | Inline function expansion before lowering |
| `compile.rs` | Integrate function registry, validate recursion |
| `resolver.rs` | Include functions in module exports |

### Implementation Order

1. Grammar + AST — function syntax parses (no semantics yet)
2. `function.rs` — `FuncDef`, `FunctionRegistry` types
3. Parser — wire up function parsing
4. Arithmetic expansion — inline simple functions
5. Conditional expansion — handle `if-then-else`
6. Predicate expansion — expand to joins
7. Recursion detection — call graph analysis
8. Base case validation — static analysis
9. Depth limiting — runtime tracking
10. Type inference — infer parameter/return types
11. Type checking — validate annotations
12. Module integration — visibility, imports
13. Tests — comprehensive test suite

### Dependencies

- Depends on Module System (2.3) for visibility/imports
- No new external crates needed
- Extends existing `ArithExpr` evaluation

---

## Non-Goals (Deferred)

- **Higher-order functions** — functions as arguments
- **Closures** — capturing variables from outer scope
- **Memoization** — caching function results
- **Pattern matching** — multi-clause function definitions

---

## Summary

User-defined functions provide:

- **Two body types** — arithmetic expressions or predicate-based (inline views)
- **Non-deterministic semantics** — predicate functions expand to all results
- **Optional type annotations** — inferred when omitted
- **Flexible call syntax** — both `is` expression and direct in terms
- **Safe recursion** — conditional form required, base case analysis, depth limit
- **Module integration** — same visibility/import rules as predicates
- **Inline expansion** — compiled away before IR generation
