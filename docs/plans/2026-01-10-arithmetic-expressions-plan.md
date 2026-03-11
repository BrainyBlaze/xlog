# Arithmetic Expressions Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Prolog-style `is` arithmetic expressions to xlog, enabling computation within Datalog rules.

**Architecture:** Extend grammar/AST for arithmetic, lower `is` expressions to computed projections in IR, execute via bytecode-based GPU kernel evaluation.

**Tech Stack:** Pest (parser), Rust (AST/IR/lowering), CUDA (GPU kernels)

---

## Task 1: Extend Grammar with Arithmetic

**Files:**
- Modify: `crates/xlog-logic/src/grammar.pest`

**Step 1: Write failing parse test**

Create test in `crates/xlog-logic/src/parser.rs`:

```rust
#[test]
fn test_parse_is_expr() {
    let input = "result(X, Z) :- input(X, Y), Z is Y + 1.";
    let result = parse(input);
    assert!(result.is_ok(), "Failed to parse is expression: {:?}", result.err());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-logic test_parse_is_expr -- --nocapture`
Expected: FAIL with parse error

**Step 3: Add arithmetic grammar rules**

Add to `grammar.pest` after the `comparison` rule:

```pest
// Arithmetic operators
arith_op_mul = { "*" | "/" | "%" }
arith_op_add = { "+" | "-" }

// Built-in functions
builtin_fn = { "abs" | "min" | "max" | "pow" | "cast" }

// Arithmetic expressions with precedence
arith_primary = {
    builtin_fn ~ "(" ~ arith_expr ~ ("," ~ (arith_expr | type_spec))* ~ ")" |
    "(" ~ arith_expr ~ ")" |
    variable |
    integer |
    float_num
}

arith_term = { arith_primary ~ (arith_op_mul ~ arith_primary)* }
arith_expr = { arith_term ~ (arith_op_add ~ arith_term)* }

// The 'is' construct
is_expr = { variable ~ "is" ~ arith_expr }
```

**Step 4: Update body_literal rule**

Change:
```pest
body_literal = { negated_atom | atom | comparison }
```
To:
```pest
body_literal = { negated_atom | atom | comparison | is_expr }
```

**Step 5: Run test to verify it passes**

Run: `cargo test -p xlog-logic test_parse_is_expr -- --nocapture`
Expected: PASS (parser accepts the syntax)

**Step 6: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest crates/xlog-logic/src/parser.rs
git commit -m "feat(parser): add arithmetic expression grammar"
```

---

## Task 2: Extend AST with ArithExpr and IsExpr

**Files:**
- Modify: `crates/xlog-logic/src/ast.rs`

**Step 1: Write failing test**

Add to `crates/xlog-logic/src/ast.rs` in the tests module:

```rust
#[test]
fn test_arith_expr_structure() {
    let expr = ArithExpr::Add(
        Box::new(ArithExpr::Variable("X".to_string())),
        Box::new(ArithExpr::Integer(1)),
    );
    assert!(matches!(expr, ArithExpr::Add(_, _)));
}

#[test]
fn test_is_expr_structure() {
    let is_expr = IsExpr {
        target: "Z".to_string(),
        expr: ArithExpr::Variable("Y".to_string()),
    };
    assert_eq!(is_expr.target, "Z");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-logic test_arith_expr_structure -- --nocapture`
Expected: FAIL with "cannot find type ArithExpr"

**Step 3: Add ArithExpr enum**

Add after the `AggOp` enum:

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

    // Type cast
    Cast(Box<ArithExpr>, ScalarType),
}

impl ArithExpr {
    /// Get all variable names used in this expression
    pub fn variables(&self) -> Vec<&str> {
        match self {
            ArithExpr::Variable(name) => vec![name.as_str()],
            ArithExpr::Integer(_) | ArithExpr::Float(_) => vec![],
            ArithExpr::Add(l, r) | ArithExpr::Sub(l, r) | ArithExpr::Mul(l, r)
            | ArithExpr::Div(l, r) | ArithExpr::Mod(l, r)
            | ArithExpr::Min(l, r) | ArithExpr::Max(l, r) | ArithExpr::Pow(l, r) => {
                let mut vars = l.variables();
                vars.extend(r.variables());
                vars
            }
            ArithExpr::Abs(e) | ArithExpr::Cast(e, _) => e.variables(),
        }
    }
}
```

**Step 4: Add IsExpr struct**

Add after ArithExpr:

```rust
/// Is-expression for variable binding: Z is X + Y
#[derive(Debug, Clone, PartialEq)]
pub struct IsExpr {
    pub target: String,      // Must be fresh (unbound) variable
    pub expr: ArithExpr,
}
```

**Step 5: Extend BodyLiteral enum**

Change:
```rust
pub enum BodyLiteral {
    Positive(Atom),
    Negated(Atom),
    Comparison(Comparison),
}
```
To:
```rust
pub enum BodyLiteral {
    Positive(Atom),
    Negated(Atom),
    Comparison(Comparison),
    IsExpr(IsExpr),
}
```

**Step 6: Update BodyLiteral::variables() method**

Add case to the match in `variables()`:

```rust
BodyLiteral::IsExpr(is_expr) => {
    let mut vars = is_expr.expr.variables();
    vars.push(is_expr.target.as_str());
    vars
}
```

**Step 7: Run tests to verify they pass**

Run: `cargo test -p xlog-logic test_arith_expr_structure test_is_expr_structure -- --nocapture`
Expected: PASS

**Step 8: Commit**

```bash
git add crates/xlog-logic/src/ast.rs
git commit -m "feat(ast): add ArithExpr and IsExpr types"
```

---

## Task 3: Implement Parser for Arithmetic Expressions

**Files:**
- Modify: `crates/xlog-logic/src/parser.rs`

**Step 1: Write comprehensive parse tests**

Add to parser tests:

```rust
#[test]
fn test_parse_arithmetic_precedence() {
    // Multiplication before addition
    let input = "r(X, Z) :- p(X, A, B), Z is A + B * 2.";
    let result = parse(input).unwrap();
    let rule = &result.rules[0];
    assert_eq!(rule.body.len(), 2);
    assert!(matches!(&rule.body[1], BodyLiteral::IsExpr(_)));
}

#[test]
fn test_parse_arithmetic_parentheses() {
    let input = "r(X, Z) :- p(X, A, B), Z is (A + B) * 2.";
    assert!(parse(input).is_ok());
}

#[test]
fn test_parse_arithmetic_builtins() {
    let inputs = [
        "r(X, Z) :- p(X, Y), Z is abs(Y).",
        "r(X, Z) :- p(X, A, B), Z is min(A, B).",
        "r(X, Z) :- p(X, A, B), Z is max(A, B).",
        "r(X, Z) :- p(X, A, B), Z is pow(A, B).",
        "r(X, Z) :- p(X, Y), Z is cast(Y, f64).",
    ];
    for input in inputs {
        assert!(parse(input).is_ok(), "Failed to parse: {}", input);
    }
}

#[test]
fn test_parse_arithmetic_nested() {
    let input = "r(X, Z) :- p(X, A, B, C), Z is abs(A - B) + min(B, C) * 2.";
    assert!(parse(input).is_ok());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p xlog-logic test_parse_arithmetic -- --nocapture`
Expected: FAIL (parser doesn't know how to build ArithExpr)

**Step 3: Add build_arith_expr function**

Add to parser.rs:

```rust
fn build_arith_expr(pair: Pair<Rule>) -> Result<ArithExpr, XlogError> {
    match pair.as_rule() {
        Rule::arith_expr => {
            let mut inner = pair.into_inner();
            let first = build_arith_term(inner.next().unwrap())?;

            let mut result = first;
            while let Some(op_pair) = inner.next() {
                let op = op_pair.as_str();
                let right = build_arith_term(inner.next().unwrap())?;
                result = match op {
                    "+" => ArithExpr::Add(Box::new(result), Box::new(right)),
                    "-" => ArithExpr::Sub(Box::new(result), Box::new(right)),
                    _ => return Err(XlogError::Parse(format!("Unknown additive op: {}", op))),
                };
            }
            Ok(result)
        }
        _ => Err(XlogError::Parse(format!("Expected arith_expr, got {:?}", pair.as_rule()))),
    }
}

fn build_arith_term(pair: Pair<Rule>) -> Result<ArithExpr, XlogError> {
    let mut inner = pair.into_inner();
    let first = build_arith_primary(inner.next().unwrap())?;

    let mut result = first;
    while let Some(op_pair) = inner.next() {
        let op = op_pair.as_str();
        let right = build_arith_primary(inner.next().unwrap())?;
        result = match op {
            "*" => ArithExpr::Mul(Box::new(result), Box::new(right)),
            "/" => ArithExpr::Div(Box::new(result), Box::new(right)),
            "%" => ArithExpr::Mod(Box::new(result), Box::new(right)),
            _ => return Err(XlogError::Parse(format!("Unknown multiplicative op: {}", op))),
        };
    }
    Ok(result)
}

fn build_arith_primary(pair: Pair<Rule>) -> Result<ArithExpr, XlogError> {
    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::variable => Ok(ArithExpr::Variable(inner.as_str().to_string())),
        Rule::integer => {
            let val: i64 = inner.as_str().parse()
                .map_err(|e| XlogError::Parse(format!("Invalid integer: {}", e)))?;
            Ok(ArithExpr::Integer(val))
        }
        Rule::float_num => {
            let val: f64 = inner.as_str().parse()
                .map_err(|e| XlogError::Parse(format!("Invalid float: {}", e)))?;
            Ok(ArithExpr::Float(val))
        }
        Rule::arith_expr => build_arith_expr(inner),
        Rule::builtin_fn => {
            let mut fn_inner = inner.into_inner();
            // This is handled specially - the builtin_fn rule captures the function call
            Err(XlogError::Parse("builtin_fn should be handled in arith_primary".into()))
        }
        Rule::arith_primary => {
            // Handle builtin function calls
            let mut parts = inner.into_inner();
            let first = parts.next().unwrap();

            if first.as_rule() == Rule::builtin_fn {
                let fn_name = first.as_str();
                let arg1 = build_arith_expr(parts.next().unwrap())?;

                match fn_name {
                    "abs" => Ok(ArithExpr::Abs(Box::new(arg1))),
                    "min" => {
                        let arg2 = build_arith_expr(parts.next().unwrap())?;
                        Ok(ArithExpr::Min(Box::new(arg1), Box::new(arg2)))
                    }
                    "max" => {
                        let arg2 = build_arith_expr(parts.next().unwrap())?;
                        Ok(ArithExpr::Max(Box::new(arg1), Box::new(arg2)))
                    }
                    "pow" => {
                        let arg2 = build_arith_expr(parts.next().unwrap())?;
                        Ok(ArithExpr::Pow(Box::new(arg1), Box::new(arg2)))
                    }
                    "cast" => {
                        let type_pair = parts.next().unwrap();
                        let scalar_type = build_type_spec(type_pair)?;
                        Ok(ArithExpr::Cast(Box::new(arg1), scalar_type))
                    }
                    _ => Err(XlogError::Parse(format!("Unknown builtin: {}", fn_name))),
                }
            } else {
                build_arith_primary(first)
            }
        }
        _ => Err(XlogError::Parse(format!("Unexpected in arith_primary: {:?}", inner.as_rule()))),
    }
}

fn build_is_expr(pair: Pair<Rule>) -> Result<IsExpr, XlogError> {
    let mut inner = pair.into_inner();
    let target = inner.next().unwrap().as_str().to_string();
    let expr = build_arith_expr(inner.next().unwrap())?;
    Ok(IsExpr { target, expr })
}
```

**Step 4: Update build_body_literal to handle is_expr**

Add case to the match:

```rust
Rule::is_expr => Ok(BodyLiteral::IsExpr(build_is_expr(inner)?)),
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p xlog-logic test_parse_arithmetic -- --nocapture`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/xlog-logic/src/parser.rs
git commit -m "feat(parser): implement arithmetic expression parsing"
```

---

## Task 4: Extend IR with Arithmetic Expressions

**Files:**
- Modify: `crates/xlog-ir/src/rir.rs`

**Step 1: Write failing test**

Add to `crates/xlog-ir/src/rir.rs`:

```rust
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
        Expr::Add(Box::new(Expr::Column(0)), Box::new(Expr::Const(ConstValue::I64(1)))),
        ScalarType::I64,
    );
    assert!(matches!(proj, ProjectExpr::Computed(_, _)));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-ir test_expr_arithmetic -- --nocapture`
Expected: FAIL with "no variant named Add"

**Step 3: Extend Expr enum with arithmetic operations**

Add to Expr enum:

```rust
pub enum Expr {
    Column(usize),
    Const(ConstValue),
    Compare { left: Box<Expr>, op: CompareOp, right: Box<Expr> },
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Not(Box<Expr>),

    // Arithmetic operations
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Mod(Box<Expr>, Box<Expr>),

    // Built-in functions
    Abs(Box<Expr>),
    Min(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Cast(Box<Expr>, ScalarType),
}
```

**Step 4: Add ProjectExpr enum**

Add after Expr:

```rust
/// Projection expression - either pass-through column or computed value
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectExpr {
    /// Pass through column at given index
    Column(usize),
    /// Compute expression, result has given type
    Computed(Expr, ScalarType),
}
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p xlog-ir test_expr_arithmetic test_project_expr_computed -- --nocapture`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/xlog-ir/src/rir.rs
git commit -m "feat(ir): add arithmetic expressions to Expr and ProjectExpr"
```

---

## Task 5: Implement Type Inference for Arithmetic

**Files:**
- Modify: `crates/xlog-logic/src/lower.rs`

**Step 1: Write failing test**

Add to lower.rs tests:

```rust
#[test]
fn test_arith_type_inference_same_type() {
    // X + Y where both are i64 should succeed
    let lowerer = Lowerer::new();
    let mut var_env = VariableEnv::new();
    var_env.bind("X", 0, ScalarType::I64);
    var_env.bind("Y", 1, ScalarType::I64);

    let expr = ArithExpr::Add(
        Box::new(ArithExpr::Variable("X".to_string())),
        Box::new(ArithExpr::Variable("Y".to_string())),
    );
    let result = lowerer.infer_arith_type(&expr, &var_env);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ScalarType::I64);
}

#[test]
fn test_arith_type_inference_mismatch() {
    // X + Y where X is i64 and Y is f64 should fail
    let lowerer = Lowerer::new();
    let mut var_env = VariableEnv::new();
    var_env.bind("X", 0, ScalarType::I64);
    var_env.bind("Y", 1, ScalarType::F64);

    let expr = ArithExpr::Add(
        Box::new(ArithExpr::Variable("X".to_string())),
        Box::new(ArithExpr::Variable("Y".to_string())),
    );
    let result = lowerer.infer_arith_type(&expr, &var_env);
    assert!(result.is_err());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-logic test_arith_type_inference -- --nocapture`
Expected: FAIL with "no method named infer_arith_type"

**Step 3: Implement infer_arith_type**

Add to Lowerer impl:

```rust
/// Infer the result type of an arithmetic expression (strict same-type)
pub fn infer_arith_type(
    &self,
    expr: &ArithExpr,
    var_env: &VariableEnv,
) -> Result<ScalarType> {
    match expr {
        ArithExpr::Variable(name) => {
            var_env.get_type(name).ok_or_else(||
                XlogError::Compilation(format!("Unknown variable {} in arithmetic", name)))
        }
        ArithExpr::Integer(_) => Ok(ScalarType::I64),
        ArithExpr::Float(_) => Ok(ScalarType::F64),

        ArithExpr::Add(l, r) | ArithExpr::Sub(l, r) |
        ArithExpr::Mul(l, r) | ArithExpr::Div(l, r) => {
            let lt = self.infer_arith_type(l, var_env)?;
            let rt = self.infer_arith_type(r, var_env)?;

            if lt != rt {
                return Err(XlogError::Compilation(format!(
                    "Type mismatch in arithmetic: {:?} vs {:?}. Use cast() for conversion.",
                    lt, rt
                )));
            }

            if !Self::is_numeric_type(&lt) {
                return Err(XlogError::Compilation(format!(
                    "Arithmetic requires numeric type, got {:?}", lt
                )));
            }

            Ok(lt)
        }

        ArithExpr::Mod(l, r) => {
            let lt = self.infer_arith_type(l, var_env)?;
            let rt = self.infer_arith_type(r, var_env)?;

            if lt != rt {
                return Err(XlogError::Compilation(format!(
                    "Type mismatch in mod: {:?} vs {:?}", lt, rt
                )));
            }

            if matches!(lt, ScalarType::F32 | ScalarType::F64) {
                return Err(XlogError::Compilation(
                    "Modulo (%) not supported for floating point".into()
                ));
            }

            Ok(lt)
        }

        ArithExpr::Abs(inner) => self.infer_arith_type(inner, var_env),

        ArithExpr::Min(l, r) | ArithExpr::Max(l, r) => {
            let lt = self.infer_arith_type(l, var_env)?;
            let rt = self.infer_arith_type(r, var_env)?;

            if lt != rt {
                return Err(XlogError::Compilation(format!(
                    "Type mismatch in min/max: {:?} vs {:?}", lt, rt
                )));
            }

            Ok(lt)
        }

        ArithExpr::Pow(_, _) => {
            // pow always returns f64 (standard math behavior)
            Ok(ScalarType::F64)
        }

        ArithExpr::Cast(_, target) => Ok(*target),
    }
}

fn is_numeric_type(t: &ScalarType) -> bool {
    matches!(t,
        ScalarType::I32 | ScalarType::I64 |
        ScalarType::U32 | ScalarType::U64 |
        ScalarType::F32 | ScalarType::F64
    )
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p xlog-logic test_arith_type_inference -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/xlog-logic/src/lower.rs
git commit -m "feat(lower): implement arithmetic type inference"
```

---

## Task 6: Implement Lowering for IsExpr

**Files:**
- Modify: `crates/xlog-logic/src/lower.rs`

**Step 1: Write failing integration test**

Add to `crates/xlog-logic/tests/real_world_tests.rs`:

```rust
#[test]
fn test_arithmetic_basic() {
    let program = r#"
        input(1, 10).
        input(2, 20).
        input(3, 30).
        output(X, Z) :- input(X, Y), Z is Y + 5.
        ?- output(X, Z).
    "#;

    let result = run_xlog(program).unwrap();
    let output = result.get("output").unwrap();

    // Should have (1, 15), (2, 25), (3, 35)
    assert_eq!(output.len(), 3);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-logic test_arithmetic_basic -- --nocapture`
Expected: FAIL (lowerer doesn't handle IsExpr)

**Step 3: Implement arith_to_expr**

Add to Lowerer impl:

```rust
/// Convert ArithExpr to IR Expr
fn arith_to_expr(&self, arith: &ArithExpr, var_env: &VariableEnv) -> Result<Expr> {
    match arith {
        ArithExpr::Variable(name) => {
            let col = var_env.get_column(name)
                .ok_or_else(|| XlogError::Compilation(format!(
                    "Variable {} not bound before use in arithmetic", name
                )))?;
            Ok(Expr::Column(col))
        }
        ArithExpr::Integer(i) => Ok(Expr::Const(ConstValue::I64(*i))),
        ArithExpr::Float(f) => Ok(Expr::Const(ConstValue::F64(*f))),

        ArithExpr::Add(l, r) => Ok(Expr::Add(
            Box::new(self.arith_to_expr(l, var_env)?),
            Box::new(self.arith_to_expr(r, var_env)?),
        )),
        ArithExpr::Sub(l, r) => Ok(Expr::Sub(
            Box::new(self.arith_to_expr(l, var_env)?),
            Box::new(self.arith_to_expr(r, var_env)?),
        )),
        ArithExpr::Mul(l, r) => Ok(Expr::Mul(
            Box::new(self.arith_to_expr(l, var_env)?),
            Box::new(self.arith_to_expr(r, var_env)?),
        )),
        ArithExpr::Div(l, r) => Ok(Expr::Div(
            Box::new(self.arith_to_expr(l, var_env)?),
            Box::new(self.arith_to_expr(r, var_env)?),
        )),
        ArithExpr::Mod(l, r) => Ok(Expr::Mod(
            Box::new(self.arith_to_expr(l, var_env)?),
            Box::new(self.arith_to_expr(r, var_env)?),
        )),

        ArithExpr::Abs(e) => Ok(Expr::Abs(Box::new(self.arith_to_expr(e, var_env)?))),
        ArithExpr::Min(l, r) => Ok(Expr::Min(
            Box::new(self.arith_to_expr(l, var_env)?),
            Box::new(self.arith_to_expr(r, var_env)?),
        )),
        ArithExpr::Max(l, r) => Ok(Expr::Max(
            Box::new(self.arith_to_expr(l, var_env)?),
            Box::new(self.arith_to_expr(r, var_env)?),
        )),
        ArithExpr::Pow(l, r) => Ok(Expr::Pow(
            Box::new(self.arith_to_expr(l, var_env)?),
            Box::new(self.arith_to_expr(r, var_env)?),
        )),
        ArithExpr::Cast(e, t) => Ok(Expr::Cast(
            Box::new(self.arith_to_expr(e, var_env)?),
            *t,
        )),
    }
}
```

**Step 4: Implement lower_is_expr**

Add to Lowerer impl:

```rust
/// Lower an is-expression: binds fresh variable to computed expression
fn lower_is_expr(
    &mut self,
    is_expr: &IsExpr,
    input: RirNode,
    var_env: &mut VariableEnv,
) -> Result<RirNode> {
    // 1. Verify target is NOT already bound
    if var_env.contains(&is_expr.target) {
        return Err(XlogError::Compilation(format!(
            "Variable {} already bound; 'is' requires fresh variable",
            is_expr.target
        )));
    }

    // 2. Verify all variables in expression are bound
    for var in is_expr.expr.variables() {
        if !var_env.contains(var) {
            return Err(XlogError::Compilation(format!(
                "Variable {} used in arithmetic but not bound",
                var
            )));
        }
    }

    // 3. Convert expression
    let expr = self.arith_to_expr(&is_expr.expr, var_env)?;

    // 4. Infer result type
    let result_type = self.infer_arith_type(&is_expr.expr, var_env)?;

    // 5. Build projection that passes through all existing columns plus computed
    let mut proj_exprs: Vec<ProjectExpr> = (0..var_env.column_count())
        .map(ProjectExpr::Column)
        .collect();
    proj_exprs.push(ProjectExpr::Computed(expr, result_type));

    // 6. Bind the new variable
    let new_col = var_env.column_count();
    var_env.bind(&is_expr.target, new_col, result_type);

    // 7. Build new schema
    let mut new_schema = input.schema().clone();
    new_schema.add_column(&is_expr.target, result_type);

    Ok(RirNode::Project {
        input: Box::new(input),
        columns: proj_exprs,
        schema: new_schema,
    })
}
```

**Step 5: Update lower_body_literal to handle IsExpr**

Add case to the match in lower_body_literal:

```rust
BodyLiteral::IsExpr(is_expr) => {
    self.lower_is_expr(is_expr, current, var_env)
}
```

**Step 6: Run test to verify compilation works**

Run: `cargo test -p xlog-logic test_arithmetic_basic -- --nocapture`
Expected: May still fail (executor doesn't handle ProjectExpr::Computed yet)

**Step 7: Commit**

```bash
git add crates/xlog-logic/src/lower.rs
git commit -m "feat(lower): implement is-expression lowering"
```

---

## Task 7: Update Executor to Handle Computed Projections

**Files:**
- Modify: `crates/xlog-runtime/src/executor.rs`

**Step 1: Write failing test**

Add to executor tests:

```rust
#[test]
fn test_execute_computed_projection() {
    // Test that computed projections work
    let provider = create_test_provider();
    let executor = Executor::new(provider);

    // Create a simple scan + computed projection
    let input = create_test_buffer(&[1, 2, 3], "x");
    let plan = RirNode::Project {
        input: Box::new(RirNode::Scan { relation: "test".into() }),
        columns: vec![
            ProjectExpr::Column(0),
            ProjectExpr::Computed(
                Expr::Add(
                    Box::new(Expr::Column(0)),
                    Box::new(Expr::Const(ConstValue::I64(10))),
                ),
                ScalarType::I64,
            ),
        ],
        schema: Schema::new(vec![
            ("x".into(), ScalarType::I64),
            ("y".into(), ScalarType::I64),
        ]),
    };

    let result = executor.execute(&plan).unwrap();
    // Should have [(1, 11), (2, 12), (3, 13)]
    assert_eq!(result.num_rows(), 3);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-runtime test_execute_computed_projection -- --nocapture`
Expected: FAIL (executor doesn't handle ProjectExpr::Computed)

**Step 3: Update execute_project to handle computed expressions**

This is the most complex part. The executor needs to:
1. For Column projections: just pass through
2. For Computed projections: evaluate expression and append column

Add expression evaluation to executor:

```rust
fn evaluate_expr(&self, expr: &Expr, buffer: &CudaBuffer) -> Result<CudaBuffer> {
    // For now, evaluate on CPU (later: GPU bytecode)
    match expr {
        Expr::Column(idx) => {
            // Extract single column as new buffer
            self.provider.project_columns(buffer, &[*idx])
        }
        Expr::Const(val) => {
            // Create constant column with buffer's row count
            self.provider.create_constant_column(val, buffer.num_rows())
        }
        Expr::Add(l, r) => {
            let left = self.evaluate_expr(l, buffer)?;
            let right = self.evaluate_expr(r, buffer)?;
            self.provider.add_columns(&left, &right)
        }
        // Similar for Sub, Mul, Div, Mod, Abs, Min, Max, Pow, Cast
        _ => Err(XlogError::Execution(format!("Unsupported expr: {:?}", expr))),
    }
}

fn execute_project(&mut self, input: &RirNode, columns: &[ProjectExpr], schema: &Schema) -> Result<CudaBuffer> {
    let input_buf = self.execute_node(input)?;

    // Separate column pass-throughs from computed expressions
    let mut result_columns = Vec::new();

    for proj in columns {
        match proj {
            ProjectExpr::Column(idx) => {
                let col = self.provider.extract_column(&input_buf, *idx)?;
                result_columns.push(col);
            }
            ProjectExpr::Computed(expr, _) => {
                let computed = self.evaluate_expr(expr, &input_buf)?;
                result_columns.push(computed);
            }
        }
    }

    self.provider.combine_columns(result_columns, schema.clone())
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-runtime test_execute_computed_projection -- --nocapture`
Expected: PASS (or may need provider methods implemented)

**Step 5: Commit**

```bash
git add crates/xlog-runtime/src/executor.rs
git commit -m "feat(executor): implement computed projection evaluation"
```

---

## Task 8: Add Provider Methods for Arithmetic

**Files:**
- Modify: `crates/xlog-cuda/src/provider/mod.rs`

**Step 1: Write failing test**

Add to provider tests:

```rust
#[test]
fn test_add_columns() {
    let provider = create_test_provider().unwrap();

    let a = create_test_buffer_i64(&provider, &[1, 2, 3]);
    let b = create_test_buffer_i64(&provider, &[10, 20, 30]);

    let result = provider.add_columns(&a, &b).unwrap();
    let values = provider.download_column_i64(&result, 0).unwrap();
    assert_eq!(values, vec![11, 22, 33]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-cuda test_add_columns -- --nocapture`
Expected: FAIL with "no method named add_columns"

**Step 3: Implement arithmetic column operations**

Add to CudaKernelProvider impl:

```rust
/// Add two single-column buffers element-wise
pub fn add_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
    self.binary_arith_op(a, b, ArithOp::Add)
}

pub fn sub_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
    self.binary_arith_op(a, b, ArithOp::Sub)
}

pub fn mul_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
    self.binary_arith_op(a, b, ArithOp::Mul)
}

pub fn div_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
    self.binary_arith_op(a, b, ArithOp::Div)
}

pub fn mod_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
    self.binary_arith_op(a, b, ArithOp::Mod)
}

fn binary_arith_op(&self, a: &CudaBuffer, b: &CudaBuffer, op: ArithOp) -> Result<CudaBuffer> {
    if a.num_rows() != b.num_rows() {
        return Err(XlogError::Kernel("Row count mismatch in arithmetic".into()));
    }
    if a.arity() != 1 || b.arity() != 1 {
        return Err(XlogError::Kernel("Arithmetic requires single-column buffers".into()));
    }

    let n = a.num_rows() as usize;
    let col_type = a.schema().column_type(0);

    // For now: CPU implementation, will add GPU kernel later
    match col_type {
        ScalarType::I64 => {
            let a_vals = self.download_column_i64(a, 0)?;
            let b_vals = self.download_column_i64(b, 0)?;

            let result: Vec<i64> = a_vals.iter().zip(b_vals.iter())
                .map(|(x, y)| match op {
                    ArithOp::Add => x.wrapping_add(*y),
                    ArithOp::Sub => x.wrapping_sub(*y),
                    ArithOp::Mul => x.wrapping_mul(*y),
                    ArithOp::Div => if *y == 0 { i64::MAX } else { x / y },
                    ArithOp::Mod => if *y == 0 { 0 } else { x % y },
                })
                .collect();

            self.create_buffer_from_i64_slice(&result, a.schema().clone())
        }
        ScalarType::F64 => {
            let a_vals = self.download_column_f64(a, 0)?;
            let b_vals = self.download_column_f64(b, 0)?;

            let result: Vec<f64> = a_vals.iter().zip(b_vals.iter())
                .map(|(x, y)| match op {
                    ArithOp::Add => x + y,
                    ArithOp::Sub => x - y,
                    ArithOp::Mul => x * y,
                    ArithOp::Div => x / y,  // Produces Inf/NaN for div-by-zero
                    ArithOp::Mod => x % y,
                })
                .collect();

            self.create_buffer_from_f64_slice(&result, a.schema().clone())
        }
        _ => Err(XlogError::Kernel(format!("Arithmetic not supported for {:?}", col_type))),
    }
}

enum ArithOp { Add, Sub, Mul, Div, Mod }
```

**Step 4: Add unary and other operations**

```rust
pub fn abs_column(&self, a: &CudaBuffer) -> Result<CudaBuffer> {
    // Implementation similar to above
}

pub fn min_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
    // Element-wise min
}

pub fn max_columns(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<CudaBuffer> {
    // Element-wise max
}

pub fn pow_columns(&self, base: &CudaBuffer, exp: &CudaBuffer) -> Result<CudaBuffer> {
    // Compute base^exp, returns f64
}

pub fn cast_column(&self, a: &CudaBuffer, target: ScalarType) -> Result<CudaBuffer> {
    // Type conversion
}

pub fn create_constant_column(&self, val: &ConstValue, num_rows: u64) -> Result<CudaBuffer> {
    // Create column filled with constant value
}
```

**Step 5: Run tests**

Run: `cargo test -p xlog-cuda test_add_columns -- --nocapture`
Expected: PASS

**Step 6: Commit**

```bash
git add crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(provider): implement arithmetic column operations"
```

---

## Task 9: Integration Tests

**Files:**
- Modify: `crates/xlog-logic/tests/real_world_tests.rs`

**Step 1: Add comprehensive arithmetic tests**

```rust
#[test]
fn test_arithmetic_all_ops() {
    let program = r#"
        val(10).
        val(20).
        val(30).

        // Basic operations
        plus5(X, Y) :- val(X), Y is X + 5.
        minus5(X, Y) :- val(X), Y is X - 5.
        times2(X, Y) :- val(X), Y is X * 2.
        div2(X, Y) :- val(X), Y is X / 2.
        mod7(X, Y) :- val(X), Y is X % 7.

        ?- plus5(X, Y).
        ?- minus5(X, Y).
        ?- times2(X, Y).
        ?- div2(X, Y).
        ?- mod7(X, Y).
    "#;

    let result = run_xlog(program).unwrap();

    assert_eq!(result.get("plus5").unwrap().len(), 3);   // (10,15), (20,25), (30,35)
    assert_eq!(result.get("minus5").unwrap().len(), 3);  // (10,5), (20,15), (30,25)
    assert_eq!(result.get("times2").unwrap().len(), 3);  // (10,20), (20,40), (30,60)
    assert_eq!(result.get("div2").unwrap().len(), 3);    // (10,5), (20,10), (30,15)
    assert_eq!(result.get("mod7").unwrap().len(), 3);    // (10,3), (20,6), (30,2)
}

#[test]
fn test_arithmetic_chained() {
    let program = r#"
        point(0, 0).
        point(3, 4).

        // Distance from origin: sqrt(x^2 + y^2)
        distance(X, Y, D) :-
            point(X, Y),
            X2 is X * X,
            Y2 is Y * Y,
            Sum is X2 + Y2,
            D is pow(cast(Sum, f64), 0.5).

        ?- distance(X, Y, D).
    "#;

    let result = run_xlog(program).unwrap();
    // (0,0,0.0), (3,4,5.0)
    assert_eq!(result.get("distance").unwrap().len(), 2);
}

#[test]
fn test_arithmetic_type_error() {
    let program = r#"
        mixed(1, 1.5).
        // This should fail: X is i64, Y is f64
        bad(Z) :- mixed(X, Y), Z is X + Y.
        ?- bad(Z).
    "#;

    let result = compile_xlog(program);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Type mismatch"));
}

#[test]
fn test_arithmetic_fresh_var_error() {
    let program = r#"
        val(10).
        // Z is already bound from val
        bad(Z) :- val(Z), Z is Z + 1.
        ?- bad(Z).
    "#;

    let result = compile_xlog(program);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already bound"));
}
```

**Step 2: Run all tests**

Run: `cargo test -p xlog-logic --test real_world_tests -- --nocapture`
Expected: All PASS

**Step 3: Commit**

```bash
git add crates/xlog-logic/tests/real_world_tests.rs
git commit -m "test: add comprehensive arithmetic expression tests"
```

---

## Task 10: Documentation

**Files:**
- Modify: `docs/plans/2026-01-10-arithmetic-expressions-design.md` (mark as implemented)

**Step 1: Update design doc status**

Change:
```markdown
**Status:** Approved
```
To:
```markdown
**Status:** Implemented
```

**Step 2: Commit**

```bash
git add docs/plans/2026-01-10-arithmetic-expressions-design.md
git commit -m "docs: mark arithmetic expressions as implemented"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Grammar extensions | grammar.pest |
| 2 | AST types | ast.rs |
| 3 | Parser implementation | parser.rs |
| 4 | IR extensions | rir.rs |
| 5 | Type inference | lower.rs |
| 6 | IsExpr lowering | lower.rs |
| 7 | Executor updates | executor.rs |
| 8 | Provider arithmetic ops | provider.rs |
| 9 | Integration tests | real_world_tests.rs |
| 10 | Documentation | design doc |

**Total commits:** 10
**Estimated complexity:** Medium-high (parser + lowerer + executor changes)
