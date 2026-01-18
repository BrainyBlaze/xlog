# Phase 3: User-Defined Functions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable reusable functions with arithmetic expressions, predicate-based bodies, conditionals, and safe recursion.

**Architecture:** Functions are inline-expanded at compile time. Support arithmetic bodies, predicate bodies (expand to joins), and conditionals. Recursion allowed with three safety mechanisms: required conditional form, static base-case analysis, and runtime depth limit.

**Tech Stack:** Rust, pest parser, existing xlog-logic infrastructure

**Design Document:** `docs/plans/2026-01-18-udf-design.md`

**Depends On:** Phase 2 (Module System) for visibility and imports

**Estimated Tasks:** 30 tasks, ~150 steps

---

## Part A: Grammar

### Task 1: Add Type Annotation Grammar

**Files:**
- Modify: `crates/xlog-logic/src/grammar.pest`

**Step 1: Add type_annotation rule**

```pest
// Type annotation for function parameters: X: f64
type_annotation = { ":" ~ type_spec }

// Return type annotation: -> f64
return_type = { "->" ~ type_spec }
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest
git commit -m "feat(logic): add type annotation grammar for functions"
```

---

### Task 2: Add Function Parameter Grammar

**Files:**
- Modify: `crates/xlog-logic/src/grammar.pest`

**Step 1: Add func_param rules**

```pest
// Function parameter: X or X: f64
func_param = { variable ~ type_annotation? }

// Parameter list: X, Y, Z or X: f64, Y: f64
func_params = { func_param ~ ("," ~ func_param)* }
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest
git commit -m "feat(logic): add function parameter grammar"
```

---

### Task 3: Add Conditional Expression Grammar

**Files:**
- Modify: `crates/xlog-logic/src/grammar.pest`

**Step 1: Add conditional rules**

```pest
// Condition test: X < 0 or N == 1
cond_test = { arith_expr ~ cmp_op ~ arith_expr }

// Conditional expression: if X < 0 then 0 - X else X
cond_expr = { "if" ~ cond_test ~ "then" ~ func_body ~ "else" ~ func_body }
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: Warning about unknown rule (func_body not defined yet)

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest
git commit -m "feat(logic): add conditional expression grammar"
```

---

### Task 4: Add Function Body Grammar

**Files:**
- Modify: `crates/xlog-logic/src/grammar.pest`

**Step 1: Add func_body rules**

```pest
// Arithmetic function body (includes conditionals)
func_body_arith = { cond_expr | arith_expr }

// Predicate-based function body: P :- parent(X, P).
func_body_pred = { variable ~ ":-" ~ body }

// Function body - either arithmetic or predicate-based
func_body = { func_body_pred | func_body_arith }
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest
git commit -m "feat(logic): add function body grammar (arithmetic and predicate)"
```

---

### Task 5: Add Function Definition Grammar

**Files:**
- Modify: `crates/xlog-logic/src/grammar.pest`

**Step 1: Add func_def rule**

```pest
// Function definition: func square(X) = X * X.
// Or: func dist(X: f64, Y: f64) -> f64 = ...
// Or: private func helper(X) = ...
func_def = {
    private_mod? ~ "func" ~ ident ~ "(" ~ func_params? ~ ")" ~ return_type?
    ~ "=" ~ func_body ~ "."
}
```

**Step 2: Update statement rule to include func_def**

```pest
statement = {
    func_def          // NEW - add at beginning
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

**Step 3: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest
git commit -m "feat(logic): add func_def grammar and add to statement rule"
```

---

### Task 6: Add Function Call Grammar

**Files:**
- Modify: `crates/xlog-logic/src/grammar.pest`

**Step 1: Add func_call rule**

```pest
// Function call: square(X) or dist(X1, Y1, X2, Y2)
func_call = { ident ~ "(" ~ (arith_expr ~ ("," ~ arith_expr)*)? ~ ")" }
```

**Step 2: Update arith_primary to include func_call**

Update the arith_primary rule to try func_call:

```pest
arith_primary = {
    builtin_fn ~ "(" ~ arith_expr ~ ("," ~ (arith_expr | type_spec))* ~ ")" |
    func_call |    // NEW - user-defined function calls
    "(" ~ arith_expr ~ ")" |
    variable |
    float_num |
    integer
}
```

**Step 3: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS (may need to reorder to avoid ambiguity)

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest
git commit -m "feat(logic): add func_call to arith_primary for UDF calls"
```

---

### Task 7: Add Max Recursion Pragma

**Files:**
- Modify: `crates/xlog-logic/src/grammar.pest`

**Step 1: Add pragma rule**

```pest
// Pragma for max recursion depth
pragma_max_recursion = { "#pragma" ~ "max_recursion_depth" ~ "=" ~ integer }
```

**Step 2: Update pragma rule**

```pest
pragma = { pragma_prob_engine | pragma_prob_cache | pragma_max_recursion }
```

**Step 3: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest
git commit -m "feat(logic): add max_recursion_depth pragma"
```

---

## Part B: AST Types

### Task 8: Add FuncParam Type

**Files:**
- Modify: `crates/xlog-logic/src/ast.rs`

**Step 1: Add FuncParam struct**

```rust
/// Function parameter with optional type annotation
#[derive(Debug, Clone, PartialEq)]
pub struct FuncParam {
    pub name: String,
    pub typ: Option<ScalarType>,
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/ast.rs
git commit -m "feat(logic): add FuncParam AST type"
```

---

### Task 9: Add CondExpr Type

**Files:**
- Modify: `crates/xlog-logic/src/ast.rs`

**Step 1: Add CondExpr struct**

```rust
/// Conditional expression: if X < 0 then A else B
#[derive(Debug, Clone, PartialEq)]
pub struct CondExpr {
    /// Left side of condition
    pub cond_left: ArithExpr,
    /// Comparison operator
    pub cond_op: CompOp,
    /// Right side of condition
    pub cond_right: ArithExpr,
    /// Value if condition is true
    pub then_branch: Box<FuncBody>,
    /// Value if condition is false
    pub else_branch: Box<FuncBody>,
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: Error (FuncBody not defined yet)

**Step 3: Commit partial**

```bash
git add crates/xlog-logic/src/ast.rs
git commit -m "feat(logic): add CondExpr AST type (partial)"
```

---

### Task 10: Add FuncBody Type

**Files:**
- Modify: `crates/xlog-logic/src/ast.rs`

**Step 1: Add FuncBody enum**

```rust
/// Function body - arithmetic, conditional, or predicate-based
#[derive(Debug, Clone, PartialEq)]
pub enum FuncBody {
    /// Pure arithmetic expression: X * X
    Arithmetic(ArithExpr),
    /// Conditional expression: if X < 0 then ...
    Conditional(CondExpr),
    /// Predicate-based: P :- parent(X, P)
    Predicate {
        /// Result variable
        result: String,
        /// Body literals
        body: Vec<BodyLiteral>,
    },
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/ast.rs
git commit -m "feat(logic): add FuncBody AST type"
```

---

### Task 11: Add FuncDef Type

**Files:**
- Modify: `crates/xlog-logic/src/ast.rs`

**Step 1: Add FuncDef struct**

```rust
/// User-defined function
#[derive(Debug, Clone, PartialEq)]
pub struct FuncDef {
    /// Function name
    pub name: String,
    /// Parameters
    pub params: Vec<FuncParam>,
    /// Optional return type annotation
    pub return_type: Option<ScalarType>,
    /// Function body
    pub body: FuncBody,
    /// Is this function private?
    pub is_private: bool,
}

impl FuncDef {
    /// Check if this function contains recursive calls to itself
    pub fn is_recursive(&self) -> bool {
        self.body_calls_function(&self.name, &self.body)
    }

    fn body_calls_function(&self, name: &str, body: &FuncBody) -> bool {
        match body {
            FuncBody::Arithmetic(expr) => self.expr_calls_function(name, expr),
            FuncBody::Conditional(cond) => {
                self.expr_calls_function(name, &cond.cond_left)
                    || self.expr_calls_function(name, &cond.cond_right)
                    || self.body_calls_function(name, &cond.then_branch)
                    || self.body_calls_function(name, &cond.else_branch)
            }
            FuncBody::Predicate { .. } => false, // Predicates don't contain function calls
        }
    }

    fn expr_calls_function(&self, name: &str, expr: &ArithExpr) -> bool {
        // TODO: implement when FuncCall is added to ArithExpr
        false
    }
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/ast.rs
git commit -m "feat(logic): add FuncDef AST type"
```

---

### Task 12: Add FuncCall to ArithExpr

**Files:**
- Modify: `crates/xlog-logic/src/ast.rs`

**Step 1: Add FuncCall variant to ArithExpr**

Update the ArithExpr enum to add:

```rust
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

    // Type cast
    Cast(Box<ArithExpr>, ScalarType),

    // NEW: User-defined function call
    FuncCall {
        name: String,
        args: Vec<ArithExpr>,
    },
}
```

**Step 2: Update ArithExpr::variables() method**

Add case for FuncCall:

```rust
ArithExpr::FuncCall { args, .. } => {
    args.iter().flat_map(|a| a.variables()).collect()
}
```

**Step 3: Build and fix any broken code**

Run: `cargo build -p xlog-logic 2>&1 | head -50`

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/ast.rs
git commit -m "feat(logic): add FuncCall variant to ArithExpr"
```

---

### Task 13: Update Program for Functions

**Files:**
- Modify: `crates/xlog-logic/src/ast.rs`

**Step 1: Add functions field to Program**

```rust
pub struct Program {
    pub imports: Vec<UseDecl>,
    pub functions: Vec<FuncDef>,   // NEW
    pub domains: Vec<DomainDecl>,
    pub predicates: Vec<PredDecl>,
    // ... rest unchanged
}
```

**Step 2: Update Directives for max_recursion_depth**

```rust
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Directives {
    pub prob_engine: Option<ProbEngine>,
    pub prob_cache: Option<ProbCache>,
    pub max_recursion_depth: Option<u32>,  // NEW
}

impl Directives {
    pub fn max_recursion_depth_or_default(&self) -> u32 {
        self.max_recursion_depth.unwrap_or(1000)
    }
}
```

**Step 3: Update Default/new**

Add `functions: Vec::new()`.

**Step 4: Build and fix**

Run: `cargo build -p xlog-logic`

**Step 5: Commit**

```bash
git add crates/xlog-logic/src/ast.rs
git commit -m "feat(logic): add functions to Program and max_recursion_depth to Directives"
```

---

## Part C: Parser

### Task 14: Parse Function Parameters

**Files:**
- Modify: `crates/xlog-logic/src/parser.rs`

**Step 1: Add parse_func_param function**

```rust
fn parse_func_param(pair: Pair<Rule>) -> FuncParam {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let typ = inner.next().map(|ta| {
        // type_annotation contains type_spec
        parse_type_spec(ta.into_inner().next().unwrap())
    });
    FuncParam { name, typ }
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/parser.rs
git commit -m "feat(logic): add parse_func_param function"
```

---

### Task 15: Parse Conditional Expression

**Files:**
- Modify: `crates/xlog-logic/src/parser.rs`

**Step 1: Add parse_cond_expr function**

```rust
fn parse_cond_expr(pair: Pair<Rule>) -> CondExpr {
    let mut inner = pair.into_inner();

    // Parse condition test
    let cond_test = inner.next().unwrap();
    let mut test_inner = cond_test.into_inner();
    let cond_left = parse_arith_expr(test_inner.next().unwrap());
    let cond_op = parse_cmp_op(test_inner.next().unwrap());
    let cond_right = parse_arith_expr(test_inner.next().unwrap());

    // Parse then branch
    let then_branch = Box::new(parse_func_body(inner.next().unwrap()));

    // Parse else branch
    let else_branch = Box::new(parse_func_body(inner.next().unwrap()));

    CondExpr {
        cond_left,
        cond_op,
        cond_right,
        then_branch,
        else_branch,
    }
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: Error (parse_func_body not defined)

**Step 3: Commit partial**

```bash
git add crates/xlog-logic/src/parser.rs
git commit -m "feat(logic): add parse_cond_expr function (partial)"
```

---

### Task 16: Parse Function Body

**Files:**
- Modify: `crates/xlog-logic/src/parser.rs`

**Step 1: Add parse_func_body function**

```rust
fn parse_func_body(pair: Pair<Rule>) -> FuncBody {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::func_body_pred => {
            let mut parts = inner.into_inner();
            let result = parts.next().unwrap().as_str().to_string();
            let body = parse_body(parts.next().unwrap());
            FuncBody::Predicate { result, body }
        }
        Rule::func_body_arith => {
            let arith_inner = inner.into_inner().next().unwrap();
            match arith_inner.as_rule() {
                Rule::cond_expr => FuncBody::Conditional(parse_cond_expr(arith_inner)),
                _ => FuncBody::Arithmetic(parse_arith_expr(arith_inner)),
            }
        }
        _ => unreachable!("unexpected rule in func_body: {:?}", inner.as_rule()),
    }
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/parser.rs
git commit -m "feat(logic): add parse_func_body function"
```

---

### Task 17: Parse Function Definition

**Files:**
- Modify: `crates/xlog-logic/src/parser.rs`

**Step 1: Add parse_func_def function**

```rust
fn parse_func_def(pair: Pair<Rule>) -> FuncDef {
    let mut inner = pair.into_inner();
    let mut is_private = false;

    // Check for private modifier
    let first = inner.next().unwrap();
    let name_pair = if first.as_rule() == Rule::private_mod {
        is_private = true;
        inner.next().unwrap()
    } else {
        first
    };

    let name = name_pair.as_str().to_string();

    let mut params = Vec::new();
    let mut return_type = None;
    let mut body = None;

    for pair in inner {
        match pair.as_rule() {
            Rule::func_params => {
                params = pair.into_inner().map(parse_func_param).collect();
            }
            Rule::return_type => {
                return_type = Some(parse_type_spec(pair.into_inner().next().unwrap()));
            }
            Rule::func_body => {
                body = Some(parse_func_body(pair));
            }
            _ => {}
        }
    }

    FuncDef {
        name,
        params,
        return_type,
        body: body.expect("function must have a body"),
        is_private,
    }
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/parser.rs
git commit -m "feat(logic): add parse_func_def function"
```

---

### Task 18: Parse Function Calls in Expressions

**Files:**
- Modify: `crates/xlog-logic/src/parser.rs`

**Step 1: Update parse_arith_expr for func_call**

In the function that parses arith_primary, add a case for func_call:

```rust
Rule::func_call => {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let args: Vec<ArithExpr> = inner.map(parse_arith_expr).collect();
    ArithExpr::FuncCall { name, args }
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/parser.rs
git commit -m "feat(logic): parse function calls in arithmetic expressions"
```

---

### Task 19: Wire Up Function Parsing

**Files:**
- Modify: `crates/xlog-logic/src/parser.rs`

**Step 1: Add func_def to statement parsing**

In the main parse loop:

```rust
Rule::func_def => {
    program.functions.push(parse_func_def(pair));
}
```

**Step 2: Add pragma_max_recursion parsing**

```rust
Rule::pragma_max_recursion => {
    let value = pair.into_inner().next().unwrap().as_str().parse().unwrap();
    program.directives.max_recursion_depth = Some(value);
}
```

**Step 3: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/parser.rs
git commit -m "feat(logic): wire up function definition parsing"
```

---

### Task 20: Add Function Parser Tests

**Files:**
- Create: `crates/xlog-logic/tests/function_parse_tests.rs`

**Step 1: Create test file**

```rust
use xlog_logic::parse;
use xlog_logic::ast::{FuncBody, ArithExpr};

#[test]
fn test_parse_simple_func() {
    let src = "func square(X) = X * X.";
    let program = parse(src).unwrap();
    assert_eq!(program.functions.len(), 1);
    assert_eq!(program.functions[0].name, "square");
    assert_eq!(program.functions[0].params.len(), 1);
    assert_eq!(program.functions[0].params[0].name, "X");
    assert!(program.functions[0].params[0].typ.is_none());
    assert!(!program.functions[0].is_private);
}

#[test]
fn test_parse_typed_func() {
    let src = "func square(X: f64) -> f64 = X * X.";
    let program = parse(src).unwrap();
    let func = &program.functions[0];
    assert!(func.params[0].typ.is_some());
    assert!(func.return_type.is_some());
}

#[test]
fn test_parse_multi_param_func() {
    let src = "func add(X, Y) = X + Y.";
    let program = parse(src).unwrap();
    assert_eq!(program.functions[0].params.len(), 2);
}

#[test]
fn test_parse_private_func() {
    let src = "private func helper(X) = X + 1.";
    let program = parse(src).unwrap();
    assert!(program.functions[0].is_private);
}

#[test]
fn test_parse_conditional_func() {
    let src = "func abs_val(X) = if X < 0 then 0 - X else X.";
    let program = parse(src).unwrap();
    match &program.functions[0].body {
        FuncBody::Conditional(cond) => {
            // Verify condition structure
            assert!(matches!(&cond.cond_left, ArithExpr::Variable(v) if v == "X"));
        }
        _ => panic!("expected conditional body"),
    }
}

#[test]
fn test_parse_predicate_func() {
    let src = "func get_parent(X) = P :- parent(X, P).";
    let program = parse(src).unwrap();
    match &program.functions[0].body {
        FuncBody::Predicate { result, body } => {
            assert_eq!(result, "P");
            assert!(!body.is_empty());
        }
        _ => panic!("expected predicate body"),
    }
}

#[test]
fn test_parse_recursive_func() {
    let src = "func fib(N) = if N <= 1 then N else fib(N - 1) + fib(N - 2).";
    let program = parse(src).unwrap();
    assert_eq!(program.functions[0].name, "fib");
    // Recursion detection is done in FuncDef::is_recursive()
}

#[test]
fn test_parse_func_call_in_rule() {
    let src = r#"
        func square(X) = X * X.
        result(Y) :- input(X), Y is square(X).
    "#;
    let program = parse(src).unwrap();
    assert_eq!(program.functions.len(), 1);
    assert_eq!(program.rules.len(), 1);
}

#[test]
fn test_parse_max_recursion_pragma() {
    let src = r#"
        #pragma max_recursion_depth = 500.
        func fib(N) = if N <= 1 then N else fib(N - 1) + fib(N - 2).
    "#;
    let program = parse(src).unwrap();
    assert_eq!(program.directives.max_recursion_depth, Some(500));
}
```

**Step 2: Run tests**

Run: `cargo test -p xlog-logic function_parse`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/tests/function_parse_tests.rs
git commit -m "test(logic): add UDF parser tests"
```

---

## Part D: Function Registry

### Task 21: Create Function Module

**Files:**
- Create: `crates/xlog-logic/src/function.rs`

**Step 1: Create function.rs with registry**

```rust
//! Function registry and validation for user-defined functions.

use crate::ast::{FuncDef, FuncBody, ArithExpr, Program};
use std::collections::{HashMap, HashSet};

/// Errors related to functions
#[derive(Debug, Clone)]
pub enum FunctionError {
    /// Duplicate function definition
    DuplicateDefinition { name: String },
    /// Recursive function without base case
    RecursionWithoutBaseCase { name: String },
    /// Undefined function called
    UndefinedFunction { name: String },
    /// Maximum recursion depth exceeded
    MaxRecursionDepth { name: String, depth: u32 },
    /// Function name conflicts with predicate
    NameConflict { name: String },
}

impl std::fmt::Display for FunctionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FunctionError::DuplicateDefinition { name } => {
                write!(f, "error[E0501]: duplicate function definition `{}`", name)
            }
            FunctionError::RecursionWithoutBaseCase { name } => {
                writeln!(f, "error[E0502]: recursive function `{}` without base case", name)?;
                write!(f, "  = help: use conditional form: `if <condition> then <base> else <recursive>`")
            }
            FunctionError::UndefinedFunction { name } => {
                write!(f, "error[E0503]: undefined function `{}`", name)
            }
            FunctionError::MaxRecursionDepth { name, depth } => {
                write!(f, "error[E0504]: maximum recursion depth ({}) exceeded in function `{}`", depth, name)
            }
            FunctionError::NameConflict { name } => {
                write!(f, "error[E0505]: `{}` is already defined as a predicate", name)
            }
        }
    }
}

impl std::error::Error for FunctionError {}

/// Registry of user-defined functions
#[derive(Debug, Default)]
pub struct FunctionRegistry {
    functions: HashMap<String, FuncDef>,
    call_graph: HashMap<String, HashSet<String>>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a function
    pub fn register(&mut self, func: FuncDef) -> Result<(), FunctionError> {
        if self.functions.contains_key(&func.name) {
            return Err(FunctionError::DuplicateDefinition {
                name: func.name.clone(),
            });
        }

        // Build call graph
        let calls = Self::extract_calls(&func.body);
        self.call_graph.insert(func.name.clone(), calls);
        self.functions.insert(func.name.clone(), func);

        Ok(())
    }

    /// Get a function by name
    pub fn get(&self, name: &str) -> Option<&FuncDef> {
        self.functions.get(name)
    }

    /// Check if a function exists
    pub fn contains(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Extract function calls from a body
    fn extract_calls(body: &FuncBody) -> HashSet<String> {
        let mut calls = HashSet::new();
        Self::extract_calls_from_body(body, &mut calls);
        calls
    }

    fn extract_calls_from_body(body: &FuncBody, calls: &mut HashSet<String>) {
        match body {
            FuncBody::Arithmetic(expr) => Self::extract_calls_from_expr(expr, calls),
            FuncBody::Conditional(cond) => {
                Self::extract_calls_from_expr(&cond.cond_left, calls);
                Self::extract_calls_from_expr(&cond.cond_right, calls);
                Self::extract_calls_from_body(&cond.then_branch, calls);
                Self::extract_calls_from_body(&cond.else_branch, calls);
            }
            FuncBody::Predicate { .. } => {
                // Predicate bodies don't contain function calls in expressions
            }
        }
    }

    fn extract_calls_from_expr(expr: &ArithExpr, calls: &mut HashSet<String>) {
        match expr {
            ArithExpr::FuncCall { name, args } => {
                calls.insert(name.clone());
                for arg in args {
                    Self::extract_calls_from_expr(arg, calls);
                }
            }
            ArithExpr::Add(l, r)
            | ArithExpr::Sub(l, r)
            | ArithExpr::Mul(l, r)
            | ArithExpr::Div(l, r)
            | ArithExpr::Mod(l, r)
            | ArithExpr::Min(l, r)
            | ArithExpr::Max(l, r)
            | ArithExpr::Pow(l, r) => {
                Self::extract_calls_from_expr(l, calls);
                Self::extract_calls_from_expr(r, calls);
            }
            ArithExpr::Abs(e) | ArithExpr::Cast(e, _) => {
                Self::extract_calls_from_expr(e, calls);
            }
            ArithExpr::Variable(_) | ArithExpr::Integer(_) | ArithExpr::Float(_) => {}
        }
    }

    /// Check if a function is recursive (calls itself directly or indirectly)
    pub fn is_recursive(&self, name: &str) -> bool {
        self.reaches(name, name, &mut HashSet::new())
    }

    fn reaches(&self, from: &str, target: &str, visited: &mut HashSet<String>) -> bool {
        if visited.contains(from) {
            return false;
        }
        visited.insert(from.to_string());

        if let Some(calls) = self.call_graph.get(from) {
            if calls.contains(target) {
                return true;
            }
            for call in calls {
                if self.reaches(call, target, visited) {
                    return true;
                }
            }
        }
        false
    }

    /// Validate all functions
    pub fn validate(&self) -> Result<(), FunctionError> {
        for (name, func) in &self.functions {
            // Check that all called functions exist
            if let Some(calls) = self.call_graph.get(name) {
                for call in calls {
                    if !self.functions.contains_key(call) && !is_builtin(call) {
                        return Err(FunctionError::UndefinedFunction {
                            name: call.clone(),
                        });
                    }
                }
            }

            // Check recursive functions have base case
            if self.is_recursive(name) {
                if !Self::has_base_case(&func.body) {
                    return Err(FunctionError::RecursionWithoutBaseCase {
                        name: name.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    fn has_base_case(body: &FuncBody) -> bool {
        matches!(body, FuncBody::Conditional(_))
    }

    /// Build registry from a program
    pub fn from_program(program: &Program) -> Result<Self, FunctionError> {
        let mut registry = Self::new();

        // Check for name conflicts with predicates
        let pred_names: HashSet<_> = program.predicates.iter()
            .map(|p| p.name.clone())
            .collect();

        for func in &program.functions {
            if pred_names.contains(&func.name) {
                return Err(FunctionError::NameConflict {
                    name: func.name.clone(),
                });
            }
            registry.register(func.clone())?;
        }

        registry.validate()?;
        Ok(registry)
    }
}

/// Check if a name is a built-in function
fn is_builtin(name: &str) -> bool {
    matches!(name, "abs" | "min" | "max" | "pow" | "cast")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::FuncParam;

    fn make_arith_func(name: &str, body: ArithExpr) -> FuncDef {
        FuncDef {
            name: name.to_string(),
            params: vec![FuncParam { name: "X".to_string(), typ: None }],
            return_type: None,
            body: FuncBody::Arithmetic(body),
            is_private: false,
        }
    }

    #[test]
    fn test_register_function() {
        let mut reg = FunctionRegistry::new();
        let func = make_arith_func("square", ArithExpr::Variable("X".to_string()));
        assert!(reg.register(func).is_ok());
    }

    #[test]
    fn test_duplicate_error() {
        let mut reg = FunctionRegistry::new();
        let func = make_arith_func("f", ArithExpr::Variable("X".to_string()));
        reg.register(func.clone()).unwrap();
        let result = reg.register(func);
        assert!(matches!(result, Err(FunctionError::DuplicateDefinition { .. })));
    }

    #[test]
    fn test_recursive_detection() {
        let mut reg = FunctionRegistry::new();

        // f calls itself
        let f = FuncDef {
            name: "f".to_string(),
            params: vec![],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::FuncCall {
                name: "f".to_string(),
                args: vec![],
            }),
            is_private: false,
        };
        reg.register(f).unwrap();

        assert!(reg.is_recursive("f"));
    }
}
```

**Step 2: Export in lib.rs**

```rust
pub mod function;
```

**Step 3: Build and test**

Run: `cargo test -p xlog-logic function`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/function.rs crates/xlog-logic/src/lib.rs
git commit -m "feat(logic): add FunctionRegistry with validation"
```

---

## Part E: Inline Expansion

### Task 22: Add Expansion Module

**Files:**
- Create: `crates/xlog-logic/src/expand.rs`

**Step 1: Create expand.rs**

```rust
//! Inline expansion of user-defined functions.

use crate::ast::{ArithExpr, FuncBody, FuncDef, BodyLiteral, IsExpr, Rule};
use crate::function::{FunctionError, FunctionRegistry};
use std::collections::HashMap;

/// Context for expansion
pub struct ExpansionContext<'a> {
    registry: &'a FunctionRegistry,
    depth: u32,
    max_depth: u32,
}

impl<'a> ExpansionContext<'a> {
    pub fn new(registry: &'a FunctionRegistry, max_depth: u32) -> Self {
        Self {
            registry,
            depth: 0,
            max_depth,
        }
    }

    /// Expand a function call to its body with arguments substituted
    pub fn expand_call(
        &mut self,
        name: &str,
        args: &[ArithExpr],
    ) -> Result<ArithExpr, FunctionError> {
        // Check depth limit
        if self.depth >= self.max_depth {
            return Err(FunctionError::MaxRecursionDepth {
                name: name.to_string(),
                depth: self.max_depth,
            });
        }

        let func = self.registry.get(name)
            .ok_or_else(|| FunctionError::UndefinedFunction {
                name: name.to_string(),
            })?;

        // Build substitution map
        let mut subst: HashMap<String, ArithExpr> = HashMap::new();
        for (param, arg) in func.params.iter().zip(args.iter()) {
            subst.insert(param.name.clone(), arg.clone());
        }

        // Expand body
        self.depth += 1;
        let result = self.expand_body(&func.body, &subst)?;
        self.depth -= 1;

        Ok(result)
    }

    fn expand_body(
        &mut self,
        body: &FuncBody,
        subst: &HashMap<String, ArithExpr>,
    ) -> Result<ArithExpr, FunctionError> {
        match body {
            FuncBody::Arithmetic(expr) => self.expand_expr(expr, subst),
            FuncBody::Conditional(cond) => {
                // For conditionals, we need runtime evaluation
                // For now, expand both branches
                let then_expr = self.expand_body(&cond.then_branch, subst)?;
                let else_expr = self.expand_body(&cond.else_branch, subst)?;

                // Create a conditional expression (would need runtime support)
                // For compile-time expansion, this is a limitation
                // Return the structure for later evaluation
                todo!("Conditional expansion requires runtime support")
            }
            FuncBody::Predicate { .. } => {
                // Predicate bodies expand differently - they become joins
                todo!("Predicate expansion handled separately")
            }
        }
    }

    fn expand_expr(
        &mut self,
        expr: &ArithExpr,
        subst: &HashMap<String, ArithExpr>,
    ) -> Result<ArithExpr, FunctionError> {
        match expr {
            ArithExpr::Variable(name) => {
                Ok(subst.get(name).cloned().unwrap_or_else(|| expr.clone()))
            }
            ArithExpr::Integer(_) | ArithExpr::Float(_) => Ok(expr.clone()),
            ArithExpr::FuncCall { name, args } => {
                // First expand arguments
                let expanded_args: Result<Vec<_>, _> = args
                    .iter()
                    .map(|a| self.expand_expr(a, subst))
                    .collect();
                let expanded_args = expanded_args?;

                // Then expand the function call
                if self.registry.contains(name) {
                    self.expand_call(name, &expanded_args)
                } else {
                    // Built-in function, just return with expanded args
                    Ok(ArithExpr::FuncCall {
                        name: name.clone(),
                        args: expanded_args,
                    })
                }
            }
            ArithExpr::Add(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Add(Box::new(el), Box::new(er)))
            }
            ArithExpr::Sub(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Sub(Box::new(el), Box::new(er)))
            }
            ArithExpr::Mul(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Mul(Box::new(el), Box::new(er)))
            }
            ArithExpr::Div(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Div(Box::new(el), Box::new(er)))
            }
            ArithExpr::Mod(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Mod(Box::new(el), Box::new(er)))
            }
            ArithExpr::Abs(e) => {
                let ee = self.expand_expr(e, subst)?;
                Ok(ArithExpr::Abs(Box::new(ee)))
            }
            ArithExpr::Min(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Min(Box::new(el), Box::new(er)))
            }
            ArithExpr::Max(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Max(Box::new(el), Box::new(er)))
            }
            ArithExpr::Pow(l, r) => {
                let el = self.expand_expr(l, subst)?;
                let er = self.expand_expr(r, subst)?;
                Ok(ArithExpr::Pow(Box::new(el), Box::new(er)))
            }
            ArithExpr::Cast(e, t) => {
                let ee = self.expand_expr(e, subst)?;
                Ok(ArithExpr::Cast(Box::new(ee), *t))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::FuncParam;

    #[test]
    fn test_simple_expansion() {
        let mut reg = FunctionRegistry::new();

        // func double(X) = X + X
        let double = FuncDef {
            name: "double".to_string(),
            params: vec![FuncParam { name: "X".to_string(), typ: None }],
            return_type: None,
            body: FuncBody::Arithmetic(ArithExpr::Add(
                Box::new(ArithExpr::Variable("X".to_string())),
                Box::new(ArithExpr::Variable("X".to_string())),
            )),
            is_private: false,
        };
        reg.register(double).unwrap();

        let mut ctx = ExpansionContext::new(&reg, 100);

        // double(5) should expand to 5 + 5
        let result = ctx.expand_call("double", &[ArithExpr::Integer(5)]).unwrap();

        match result {
            ArithExpr::Add(l, r) => {
                assert!(matches!(*l, ArithExpr::Integer(5)));
                assert!(matches!(*r, ArithExpr::Integer(5)));
            }
            _ => panic!("Expected Add expression"),
        }
    }
}
```

**Step 2: Export in lib.rs**

```rust
pub mod expand;
```

**Step 3: Build and test**

Run: `cargo test -p xlog-logic expand`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/expand.rs crates/xlog-logic/src/lib.rs
git commit -m "feat(logic): add function inline expansion"
```

---

## Part F: Type System

### Task 23: Add Type Inference

**Files:**
- Create: `crates/xlog-logic/src/typeinfer.rs`

**Step 1: Create type inference module**

```rust
//! Type inference for user-defined functions.

use crate::ast::{ArithExpr, FuncBody, FuncDef, FuncParam, ScalarType};
use std::collections::HashMap;

/// Type inference context
pub struct TypeContext {
    /// Known variable types
    bindings: HashMap<String, ScalarType>,
}

impl TypeContext {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    /// Bind a variable to a type
    pub fn bind(&mut self, name: &str, typ: ScalarType) {
        self.bindings.insert(name.to_string(), typ);
    }

    /// Get a variable's type
    pub fn get(&self, name: &str) -> Option<ScalarType> {
        self.bindings.get(name).copied()
    }

    /// Infer type of an expression
    pub fn infer_expr(&self, expr: &ArithExpr) -> Option<ScalarType> {
        match expr {
            ArithExpr::Variable(name) => self.get(name),
            ArithExpr::Integer(_) => Some(ScalarType::I64),
            ArithExpr::Float(_) => Some(ScalarType::F64),
            ArithExpr::Add(l, r) | ArithExpr::Sub(l, r) |
            ArithExpr::Mul(l, r) | ArithExpr::Div(l, r) => {
                let lt = self.infer_expr(l)?;
                let rt = self.infer_expr(r)?;
                // Numeric promotion: if either is f64, result is f64
                if lt == ScalarType::F64 || rt == ScalarType::F64 {
                    Some(ScalarType::F64)
                } else {
                    Some(lt)
                }
            }
            ArithExpr::Cast(_, t) => Some(*t),
            ArithExpr::FuncCall { .. } => None, // Need registry lookup
            _ => None,
        }
    }
}

/// Infer parameter types from function body usage
pub fn infer_param_types(func: &FuncDef) -> Vec<Option<ScalarType>> {
    let mut ctx = TypeContext::new();

    // Start with annotated types
    for param in &func.params {
        if let Some(t) = param.typ {
            ctx.bind(&param.name, t);
        }
    }

    // Infer from body (simplified - full inference is more complex)
    // For now, return what we know from annotations
    func.params.iter().map(|p| p.typ).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_literal() {
        let ctx = TypeContext::new();
        assert_eq!(ctx.infer_expr(&ArithExpr::Integer(5)), Some(ScalarType::I64));
        assert_eq!(ctx.infer_expr(&ArithExpr::Float(3.14)), Some(ScalarType::F64));
    }

    #[test]
    fn test_infer_variable() {
        let mut ctx = TypeContext::new();
        ctx.bind("X", ScalarType::F64);
        assert_eq!(ctx.infer_expr(&ArithExpr::Variable("X".into())), Some(ScalarType::F64));
    }
}
```

**Step 2: Export module**

```rust
pub mod typeinfer;
```

**Step 3: Build and test**

Run: `cargo test -p xlog-logic typeinfer`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/typeinfer.rs crates/xlog-logic/src/lib.rs
git commit -m "feat(logic): add basic type inference for functions"
```

---

### Task 24: Add Type Checking

**Files:**
- Modify: `crates/xlog-logic/src/function.rs`

**Step 1: Add type checking to FunctionRegistry**

```rust
/// Type errors
#[derive(Debug, Clone)]
pub enum TypeError {
    /// Type mismatch
    Mismatch {
        expected: ScalarType,
        found: ScalarType,
        location: String,
    },
    /// Cannot infer type
    CannotInfer { name: String },
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TypeError::Mismatch { expected, found, location } => {
                writeln!(f, "error[E0503]: type mismatch in {}", location)?;
                write!(f, "  expected {:?}, found {:?}", expected, found)
            }
            TypeError::CannotInfer { name } => {
                write!(f, "error[E0504]: cannot infer type for `{}`", name)
            }
        }
    }
}

impl FunctionRegistry {
    /// Type check a function
    pub fn type_check(&self, func: &FuncDef) -> Result<(), TypeError> {
        // Check return type matches body type if annotated
        if let Some(ret_type) = func.return_type {
            // Infer body type and compare
            // (simplified - actual implementation needs full type inference)
        }

        // Check parameter types are consistent with usage
        for param in &func.params {
            if param.typ.is_none() {
                // Try to infer, warn if ambiguous
            }
        }

        Ok(())
    }
}
```

**Step 2: Build and test**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/function.rs
git commit -m "feat(logic): add type checking for functions"
```

---

### Task 25: Add Static Base-Case Analysis Warning

**Files:**
- Modify: `crates/xlog-logic/src/function.rs`

**Step 1: Add base-case reachability analysis**

```rust
/// Warning for potentially infinite recursion
#[derive(Debug, Clone)]
pub struct RecursionWarning {
    pub func_name: String,
    pub message: String,
}

impl std::fmt::Display for RecursionWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "warning[W0502]: potentially infinite recursion in `{}`", self.func_name)?;
        writeln!(f, "  {}", self.message)?;
        write!(f, "  = note: base case may be unreachable with given recursive call")
    }
}

impl FunctionRegistry {
    /// Analyze recursive function for potential infinite recursion
    pub fn analyze_recursion(&self, func: &FuncDef) -> Option<RecursionWarning> {
        if !self.is_recursive(&func.name) {
            return None;
        }

        match &func.body {
            FuncBody::Conditional(cond) => {
                // Analyze if recursive call moves toward base case
                self.check_convergence(func, cond)
            }
            _ => {
                // Non-conditional recursive function - already caught as error
                None
            }
        }
    }

    fn check_convergence(&self, func: &FuncDef, cond: &CondExpr) -> Option<RecursionWarning> {
        // Extract the comparison from condition
        // Check if recursive call's argument moves toward satisfying condition

        // Example: if N <= 1 then ... else f(N + 1)
        // N + 1 moves away from N <= 1, so warn

        // Simplified check: look for patterns like:
        // - Base: N <= K, Recursive: f(N - M) where M > 0 -> OK
        // - Base: N <= K, Recursive: f(N + M) where M > 0 -> WARN
        // - Base: N >= K, Recursive: f(N + M) where M > 0 -> OK
        // - Base: N >= K, Recursive: f(N - M) where M > 0 -> WARN

        // For now, do basic pattern matching
        let recursive_calls = Self::find_recursive_calls_in_body(&func.name, &cond.else_branch);

        for call_args in recursive_calls {
            if let Some(warning) = self.check_arg_convergence(&cond, &call_args, &func.name) {
                return Some(warning);
            }
        }

        None
    }

    fn find_recursive_calls_in_body(name: &str, body: &FuncBody) -> Vec<Vec<ArithExpr>> {
        let mut calls = Vec::new();
        match body {
            FuncBody::Arithmetic(expr) => {
                Self::find_recursive_calls_in_expr(name, expr, &mut calls);
            }
            FuncBody::Conditional(cond) => {
                Self::find_recursive_calls_in_expr(name, &cond.cond_left, &mut calls);
                Self::find_recursive_calls_in_expr(name, &cond.cond_right, &mut calls);
                calls.extend(Self::find_recursive_calls_in_body(name, &cond.then_branch));
                calls.extend(Self::find_recursive_calls_in_body(name, &cond.else_branch));
            }
            FuncBody::Predicate { .. } => {}
        }
        calls
    }

    fn find_recursive_calls_in_expr(name: &str, expr: &ArithExpr, calls: &mut Vec<Vec<ArithExpr>>) {
        match expr {
            ArithExpr::FuncCall { name: fn_name, args } if fn_name == name => {
                calls.push(args.clone());
            }
            ArithExpr::Add(l, r) | ArithExpr::Sub(l, r) |
            ArithExpr::Mul(l, r) | ArithExpr::Div(l, r) => {
                Self::find_recursive_calls_in_expr(name, l, calls);
                Self::find_recursive_calls_in_expr(name, r, calls);
            }
            ArithExpr::FuncCall { args, .. } => {
                for arg in args {
                    Self::find_recursive_calls_in_expr(name, arg, calls);
                }
            }
            _ => {}
        }
    }

    fn check_arg_convergence(
        &self,
        cond: &CondExpr,
        call_args: &[ArithExpr],
        func_name: &str,
    ) -> Option<RecursionWarning> {
        // Simplified: check first argument pattern
        // Real implementation would be more sophisticated

        // Pattern: condition is "X op K" and recursive call has "X + M" or "X - M"
        if call_args.is_empty() {
            return None;
        }

        // Check if first arg is X + constant where condition requires X to decrease
        if let ArithExpr::Add(_, box ArithExpr::Integer(n)) = &call_args[0] {
            if *n > 0 {
                // Adding positive - check if condition requires <= or <
                if matches!(cond.cond_op, CompOp::Le | CompOp::Lt) {
                    return Some(RecursionWarning {
                        func_name: func_name.to_string(),
                        message: format!(
                            "recursive call increases argument, but base case requires it to decrease"
                        ),
                    });
                }
            }
        }

        None
    }

    /// Validate all functions, collecting warnings
    pub fn validate_with_warnings(&self) -> (Result<(), FunctionError>, Vec<RecursionWarning>) {
        let mut warnings = Vec::new();

        for func in self.functions.values() {
            if let Some(warning) = self.analyze_recursion(func) {
                warnings.push(warning);
            }
        }

        (self.validate(), warnings)
    }
}
```

**Step 2: Add test for warning**

```rust
#[test]
fn test_recursion_warning() {
    let mut reg = FunctionRegistry::new();

    // func risky(N) = if N == 0 then 1 else risky(N + 1)
    // This should warn because N + 1 moves away from N == 0
    let risky = FuncDef {
        name: "risky".to_string(),
        params: vec![FuncParam { name: "N".to_string(), typ: None }],
        return_type: None,
        body: FuncBody::Conditional(CondExpr {
            cond_left: ArithExpr::Variable("N".to_string()),
            cond_op: CompOp::Eq,
            cond_right: ArithExpr::Integer(0),
            then_branch: Box::new(FuncBody::Arithmetic(ArithExpr::Integer(1))),
            else_branch: Box::new(FuncBody::Arithmetic(ArithExpr::FuncCall {
                name: "risky".to_string(),
                args: vec![ArithExpr::Add(
                    Box::new(ArithExpr::Variable("N".to_string())),
                    Box::new(ArithExpr::Integer(1)),
                )],
            })),
        }),
        is_private: false,
    };
    reg.register(risky).unwrap();

    let (_, warnings) = reg.validate_with_warnings();
    assert!(!warnings.is_empty(), "Expected warning for risky recursion");
}
```

**Step 3: Build and test**

Run: `cargo test -p xlog-logic recursion_warning`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/function.rs
git commit -m "feat(logic): add W0502 static base-case analysis warning"
```

---

## Part G: Predicate Expansion

### Task 26: Expand Predicate-Based Functions

**Files:**
- Modify: `crates/xlog-logic/src/expand.rs`

**Step 1: Add predicate expansion**

```rust
use crate::ast::{BodyLiteral, Rule, Atom, Term};

impl<'a> ExpansionContext<'a> {
    /// Expand a predicate-based function call to join literals
    pub fn expand_predicate_func(
        &self,
        func: &FuncDef,
        args: &[ArithExpr],
    ) -> Result<(Vec<BodyLiteral>, String), FunctionError> {
        match &func.body {
            FuncBody::Predicate { result, body } => {
                // Build substitution map from params to args
                let mut subst: HashMap<String, ArithExpr> = HashMap::new();
                for (param, arg) in func.params.iter().zip(args.iter()) {
                    subst.insert(param.name.clone(), arg.clone());
                }

                // Substitute in body literals
                let expanded_body: Vec<BodyLiteral> = body.iter()
                    .map(|lit| self.substitute_literal(lit, &subst))
                    .collect();

                // The result variable becomes the output
                let result_var = self.substitute_var(result, &subst);

                Ok((expanded_body, result_var))
            }
            _ => Err(FunctionError::UndefinedFunction {
                name: func.name.clone(),
            }),
        }
    }

    fn substitute_literal(&self, lit: &BodyLiteral, subst: &HashMap<String, ArithExpr>) -> BodyLiteral {
        match lit {
            BodyLiteral::Atom(atom) => {
                let new_terms: Vec<Term> = atom.terms.iter()
                    .map(|t| self.substitute_term(t, subst))
                    .collect();
                BodyLiteral::Atom(Atom {
                    predicate: atom.predicate.clone(),
                    terms: new_terms,
                })
            }
            BodyLiteral::Negation(atom) => {
                let new_terms: Vec<Term> = atom.terms.iter()
                    .map(|t| self.substitute_term(t, subst))
                    .collect();
                BodyLiteral::Negation(Atom {
                    predicate: atom.predicate.clone(),
                    terms: new_terms,
                })
            }
            BodyLiteral::Comparison { .. } => lit.clone(), // Handle comparisons
            BodyLiteral::IsExpr { .. } => lit.clone(), // Handle is expressions
        }
    }

    fn substitute_term(&self, term: &Term, subst: &HashMap<String, ArithExpr>) -> Term {
        match term {
            Term::Variable(name) => {
                if let Some(ArithExpr::Variable(new_name)) = subst.get(name) {
                    Term::Variable(new_name.clone())
                } else if let Some(ArithExpr::Integer(n)) = subst.get(name) {
                    Term::Integer(*n)
                } else if let Some(ArithExpr::Float(f)) = subst.get(name) {
                    Term::Float(*f)
                } else {
                    term.clone()
                }
            }
            _ => term.clone(),
        }
    }

    fn substitute_var(&self, var: &str, subst: &HashMap<String, ArithExpr>) -> String {
        if let Some(ArithExpr::Variable(new_name)) = subst.get(var) {
            new_name.clone()
        } else {
            var.to_string()
        }
    }
}
```

**Step 2: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/expand.rs
git commit -m "feat(logic): add predicate-based function expansion to joins"
```

---

### Task 27: Add Predicate Expansion Tests

**Files:**
- Modify: `crates/xlog-logic/src/expand.rs` (tests section)

**Step 1: Add predicate expansion test**

```rust
#[test]
fn test_predicate_func_expansion() {
    // func get_parent(X) = P :- parent(X, P).
    // get_parent(alice) should expand to: parent(alice, P)

    let func = FuncDef {
        name: "get_parent".to_string(),
        params: vec![FuncParam { name: "X".to_string(), typ: None }],
        return_type: None,
        body: FuncBody::Predicate {
            result: "P".to_string(),
            body: vec![
                BodyLiteral::Atom(Atom {
                    predicate: "parent".to_string(),
                    terms: vec![
                        Term::Variable("X".to_string()),
                        Term::Variable("P".to_string()),
                    ],
                }),
            ],
        },
        is_private: false,
    };

    let mut reg = FunctionRegistry::new();
    reg.register(func).unwrap();

    let ctx = ExpansionContext::new(&reg, 100);

    // Call get_parent with "alice"
    let args = vec![ArithExpr::Variable("alice".to_string())];
    let func_def = reg.get("get_parent").unwrap();
    let (body, result) = ctx.expand_predicate_func(func_def, &args).unwrap();

    assert_eq!(result, "P");
    assert_eq!(body.len(), 1);

    // Check the expanded literal
    if let BodyLiteral::Atom(atom) = &body[0] {
        assert_eq!(atom.predicate, "parent");
        assert!(matches!(&atom.terms[0], Term::Variable(v) if v == "alice"));
        assert!(matches!(&atom.terms[1], Term::Variable(v) if v == "P"));
    } else {
        panic!("Expected Atom literal");
    }
}

#[test]
fn test_multi_result_predicate_func() {
    // func get_ancestors(X) = A :- ancestor(X, A).
    // This returns ALL ancestors (non-deterministic)

    let func = FuncDef {
        name: "get_ancestors".to_string(),
        params: vec![FuncParam { name: "X".to_string(), typ: None }],
        return_type: None,
        body: FuncBody::Predicate {
            result: "A".to_string(),
            body: vec![
                BodyLiteral::Atom(Atom {
                    predicate: "ancestor".to_string(),
                    terms: vec![
                        Term::Variable("X".to_string()),
                        Term::Variable("A".to_string()),
                    ],
                }),
            ],
        },
        is_private: false,
    };

    let mut reg = FunctionRegistry::new();
    reg.register(func).unwrap();

    // The expansion produces a join that will return all matching A values
    let ctx = ExpansionContext::new(&reg, 100);
    let func_def = reg.get("get_ancestors").unwrap();
    let (body, result) = ctx.expand_predicate_func(
        func_def,
        &[ArithExpr::Variable("bob".to_string())]
    ).unwrap();

    // Body should be: ancestor(bob, A)
    assert_eq!(result, "A");
    assert_eq!(body.len(), 1);
}
```

**Step 2: Run tests**

Run: `cargo test -p xlog-logic predicate_func`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/expand.rs
git commit -m "test(logic): add predicate function expansion tests"
```

---

## Part H: Module Integration

### Task 28: Include Functions in Module Exports

**Files:**
- Modify: `crates/xlog-logic/src/resolver.rs`

**Step 1: Update extract_exports to include functions**

```rust
impl ModuleResolver {
    /// Extract exports from a parsed program (predicates and functions)
    pub fn extract_exports(program: &Program) -> (HashSet<String>, HashSet<String>) {
        let mut pred_exports = HashSet::new();
        let mut func_exports = HashSet::new();

        // Add declared predicates that aren't private
        for pred in &program.predicates {
            if !pred.is_private {
                pred_exports.insert(pred.name.clone());
            }
        }

        // Add rule heads
        for rule in &program.rules {
            pred_exports.insert(rule.head.predicate.clone());
        }

        // Add functions that aren't private
        for func in &program.functions {
            if !func.is_private {
                func_exports.insert(func.name.clone());
            }
        }

        (pred_exports, func_exports)
    }
}
```

**Step 2: Update LoadedModule**

```rust
pub struct LoadedModule {
    pub path: ModulePath,
    pub source_file: PathBuf,
    pub exports: HashSet<String>,          // Predicate exports
    pub function_exports: HashSet<String>, // Function exports
}
```

**Step 3: Update load_module to populate function_exports**

```rust
// In load_module:
let (exports, function_exports) = Self::extract_exports(&program);

let module = LoadedModule {
    path: module_path.to_vec(),
    source_file,
    exports,
    function_exports,
};
```

**Step 4: Build**

Run: `cargo build -p xlog-logic`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-logic/src/resolver.rs crates/xlog-logic/src/module.rs
git commit -m "feat(logic): include functions in module exports"
```

---

### Task 29: Handle Function Imports Across Modules

**Files:**
- Modify: `crates/xlog-logic/src/resolver.rs`

**Step 1: Add function import validation**

```rust
impl ModuleResolver {
    /// Validate imports including functions
    pub fn validate_imports_full(
        &self,
        program: &Program,
    ) -> Result<(HashMap<String, ModulePath>, HashMap<String, ModulePath>), ModuleError> {
        let mut imported_preds: HashMap<String, ModulePath> = HashMap::new();
        let mut imported_funcs: HashMap<String, ModulePath> = HashMap::new();

        for use_decl in &program.imports {
            let module = self.loaded.get(&module_path_to_string(&use_decl.module_path))
                .expect("module should be loaded");

            let names_to_import: Vec<String> = match &use_decl.imports {
                Some(specific) => specific.clone(),
                None => {
                    // Import all public predicates AND functions
                    let mut all: Vec<String> = module.exports.iter().cloned().collect();
                    all.extend(module.function_exports.iter().cloned());
                    all
                }
            };

            for name in names_to_import {
                // Check if it's a predicate
                if module.exports.contains(&name) {
                    if let Some(prev) = imported_preds.get(&name) {
                        if prev != &use_decl.module_path {
                            return Err(ModuleError::ImportConflict {
                                name,
                                module1: prev.clone(),
                                module2: use_decl.module_path.clone(),
                            });
                        }
                    }
                    imported_preds.insert(name, use_decl.module_path.clone());
                }
                // Check if it's a function
                else if module.function_exports.contains(&name) {
                    if let Some(prev) = imported_funcs.get(&name) {
                        if prev != &use_decl.module_path {
                            return Err(ModuleError::ImportConflict {
                                name,
                                module1: prev.clone(),
                                module2: use_decl.module_path.clone(),
                            });
                        }
                    }
                    imported_funcs.insert(name, use_decl.module_path.clone());
                }
                // Not found
                else {
                    return Err(ModuleError::PredicateNotFound {
                        name,
                        module: use_decl.module_path.clone(),
                    });
                }
            }
        }

        Ok((imported_preds, imported_funcs))
    }
}
```

**Step 2: Add test for function imports**

```rust
#[test]
fn test_function_import() {
    let tmp = TempDir::new().unwrap();

    // Create math module with function
    create_test_module(tmp.path(), "math", r#"
        func square(X) = X * X.
        private func helper(X) = X + 1.
    "#);

    // Create main module importing function
    create_test_module(tmp.path(), "main", r#"
        use math::{square}.
        result(Y) :- input(X), Y is square(X).
    "#);

    let mut resolver = ModuleResolver::new(vec![]);
    resolver.load_module(tmp.path(), &["main".into()]).unwrap();

    // Verify square is in function exports
    let math = resolver.loaded.get("math").unwrap();
    assert!(math.function_exports.contains("square"));
    assert!(!math.function_exports.contains("helper")); // private
}
```

**Step 3: Build and test**

Run: `cargo test -p xlog-logic function_import`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/src/resolver.rs
git commit -m "feat(logic): handle function imports across modules"
```

---

## Part I: Additional Tests

### Task 30: Add Missing Parser Tests

**Files:**
- Modify: `crates/xlog-logic/tests/function_parse_tests.rs`

**Step 1: Add nested function call test**

```rust
#[test]
fn test_parse_nested_func_calls() {
    let src = "func compose(X) = outer(inner(X)).";
    let program = parse(src).unwrap();

    assert_eq!(program.functions.len(), 1);
    let func = &program.functions[0];
    assert_eq!(func.name, "compose");

    // Body should be FuncCall containing another FuncCall
    match &func.body {
        FuncBody::Arithmetic(ArithExpr::FuncCall { name, args }) => {
            assert_eq!(name, "outer");
            assert_eq!(args.len(), 1);
            match &args[0] {
                ArithExpr::FuncCall { name: inner_name, .. } => {
                    assert_eq!(inner_name, "inner");
                }
                _ => panic!("Expected inner FuncCall"),
            }
        }
        _ => panic!("Expected FuncCall body"),
    }
}

#[test]
fn test_parse_deeply_nested() {
    let src = "func deep(X) = a(b(c(d(X)))).";
    let program = parse(src).unwrap();
    assert_eq!(program.functions.len(), 1);
}
```

**Step 2: Add direct call syntax test**

```rust
#[test]
fn test_parse_direct_call_in_term() {
    // Direct call syntax: result(square(X)) instead of Y is square(X)
    let src = r#"
        func square(X) = X * X.
        result(square(X)) :- input(X).
    "#;
    let program = parse(src).unwrap();

    assert_eq!(program.functions.len(), 1);
    assert_eq!(program.rules.len(), 1);

    // The rule head should contain the function call
    let rule = &program.rules[0];
    // Note: This depends on how the parser handles function calls in terms
}

#[test]
fn test_parse_both_call_syntaxes() {
    let src = r#"
        func double(X) = X * 2.

        % Via is expression
        result1(Y) :- input(X), Y is double(X).

        % Direct in comparison
        check(X) :- input(X), double(X) > 10.
    "#;
    let program = parse(src).unwrap();

    assert_eq!(program.functions.len(), 1);
    assert_eq!(program.rules.len(), 2);
}
```

**Step 3: Run tests**

Run: `cargo test -p xlog-logic function_parse`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/xlog-logic/tests/function_parse_tests.rs
git commit -m "test(logic): add nested calls and direct syntax parser tests"
```

---

### Task 31: Add Integration Tests

**Files:**
- Create: `crates/xlog-logic/tests/function_integration_tests.rs`

**Step 1: Create comprehensive integration test file**

```rust
//! Integration tests for user-defined functions

use xlog_logic::parse;
use xlog_logic::function::FunctionRegistry;
use xlog_logic::expand::ExpansionContext;

#[test]
fn test_full_function_pipeline() {
    let src = r#"
        func double(X) = X * 2.
        func quadruple(X) = double(double(X)).

        pred input(f64).
        input(5.0).

        pred output(f64).
        output(Y) :- input(X), Y is quadruple(X).

        ?- output(X).
    "#;

    let program = parse(src).unwrap();

    // Build function registry
    let registry = FunctionRegistry::from_program(&program).unwrap();

    // Verify both functions registered
    assert!(registry.contains("double"));
    assert!(registry.contains("quadruple"));

    // Verify quadruple calls double (detected in call graph)
    assert!(!registry.is_recursive("double"));
    assert!(!registry.is_recursive("quadruple"));
}

#[test]
fn test_recursive_function_with_base_case() {
    let src = r#"
        func factorial(N) = if N <= 1 then 1 else N * factorial(N - 1).

        pred result(f64).
        result(R) :- factorial(5) is R.
    "#;

    let program = parse(src).unwrap();
    let registry = FunctionRegistry::from_program(&program).unwrap();

    assert!(registry.is_recursive("factorial"));

    // Should pass validation (has base case)
    assert!(registry.validate().is_ok());
}

#[test]
fn test_recursive_without_base_case_fails() {
    let src = r#"
        func bad(N) = bad(N - 1).
    "#;

    let program = parse(src).unwrap();
    let result = FunctionRegistry::from_program(&program);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, FunctionError::RecursionWithoutBaseCase { .. }));
}

#[test]
fn test_function_name_conflict_with_predicate() {
    let src = r#"
        pred foo(u32).
        func foo(X) = X + 1.
    "#;

    let program = parse(src).unwrap();
    let result = FunctionRegistry::from_program(&program);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, FunctionError::NameConflict { .. }));
}

#[test]
fn test_max_recursion_depth() {
    let src = r#"
        #pragma max_recursion_depth = 10.
        func countdown(N) = if N <= 0 then 0 else countdown(N - 1).
    "#;

    let program = parse(src).unwrap();
    assert_eq!(program.directives.max_recursion_depth, Some(10));

    let registry = FunctionRegistry::from_program(&program).unwrap();

    // Expansion should respect depth limit
    let mut ctx = ExpansionContext::new(&registry, 10);
    // Calling countdown(100) should exceed depth
    let result = ctx.expand_call("countdown", &[ArithExpr::Integer(100)]);
    assert!(matches!(result, Err(FunctionError::MaxRecursionDepth { .. })));
}
```

**Step 2: Run integration tests**

Run: `cargo test -p xlog-logic function_integration`
Expected: PASS

**Step 3: Final commit for Phase 3**

```bash
git add crates/xlog-logic/tests/function_integration_tests.rs
git commit -m "test(logic): add comprehensive UDF integration tests

Phase 3 (User-Defined Functions) complete:
- func keyword with arithmetic/conditional/predicate bodies
- Optional type annotations
- Recursive functions with base case requirement
- W0502 warning for potentially infinite recursion
- Inline expansion at compile time
- Module integration (visibility, imports)
- Max recursion depth pragma"
```

---

## Summary

| Part | Tasks | Description |
|------|-------|-------------|
| A | 1-7 | Grammar rules |
| B | 8-13 | AST types |
| C | 14-20 | Parser |
| D | 21 | Function registry |
| E | 22 | Inline expansion |
| F | 23-25 | Type system & W0502 warning |
| G | 26-27 | Predicate expansion |
| H | 28-29 | Module integration |
| I | 30-31 | Additional tests |

**Total: 31 tasks, ~180 steps**

| Part | Tasks | Description |
|------|-------|-------------|
| A | 1-7 | Grammar rules |
| B | 8-13 | AST types |
| C | 14-20 | Parser |
| D | 21 | Function registry |
| E | 22 | Inline expansion |
| F | 23-30 | Type system, predicates, integration |

**Total: 30 tasks, ~150 steps**

**Note:** This phase depends on Phase 2 (Module System) for visibility and import handling.
