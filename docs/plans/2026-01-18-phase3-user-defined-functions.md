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

## Part F: Remaining Tasks (Summary)

Due to length constraints, the remaining tasks are summarized:

### Tasks 23-25: Type System
- **Task 23:** Add type inference for function parameters
- **Task 24:** Add type checking for function bodies
- **Task 25:** Add type error messages

### Tasks 26-27: Predicate Expansion
- **Task 26:** Expand predicate-based functions to joins
- **Task 27:** Add tests for predicate expansion

### Tasks 28-29: Module Integration
- **Task 28:** Include functions in module exports
- **Task 29:** Handle function imports across modules

### Task 30: Integration Tests
- Create comprehensive end-to-end tests

---

## Summary

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
