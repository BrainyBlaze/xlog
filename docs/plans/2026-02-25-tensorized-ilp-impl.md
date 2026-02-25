# Tensorized Differentiable ILP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add tensorized differentiable ILP to XLOG — parse `learnable` rules, lower to `TensorMaskedJoin` IR nodes, execute via GPU-accelerated mask extraction + hash joins, and expose to Python via PyO3.

**Architecture:** Learnable rules are parsed into a new `LearnableRule` AST type, lowered to a `TensorMaskedJoin` RirNode variant, and executed by the runtime via a CUDA kernel that extracts active (i,j,k) indices from a DLPack mask tensor, then dispatches native `hash_join_v2` for each active rule. Gradient flow happens entirely in PyTorch via Straight-Through Gumbel-Softmax — xlog-prob is not modified.

**Tech Stack:** Rust (pest parser, xlog-ir, xlog-runtime), CUDA C (extraction kernel), PyO3 (Python bindings), cudarc 0.19, DLPack

**RFC:** `docs/ilp/rfc-tensorized-ilp.md` (v4.4, 37 RDs)

---

## Task 1: Grammar — Add `learnable_rule` Production

**Files:**
- Modify: `crates/xlog-logic/src/grammar.pest:160-177`

**Step 1: Add the grammar production**

Open `crates/xlog-logic/src/grammar.pest`. Before the `statement` rule (line 162), add:

```pest
// Learnable rule: parameterized by a named tensor mask
learnable_rule = {
    "learnable" ~ "(" ~ ident ~ ")" ~ "::" ~ head ~ ":-" ~ body ~ "."
}
```

Then add `learnable_rule` to the `statement` alternatives (line 162). Insert it before `rule_def` (order matters for PEG — `learnable` keyword disambiguates):

```pest
statement = {
    func_def
    | use_stmt
    | domain_decl
    | pred_decl
    | pragma
    | neural_pred_decl
    | learnable_rule
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

**Step 2: Verify it compiles**

Run: `cargo check -p xlog-logic 2>&1 | head -30`

Expected: Compile succeeds (pest_derive auto-generates `Rule::learnable_rule` variant). There will be a warning about unmatched `Rule::learnable_rule` in the parser — that's expected and fixed in Task 2.

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/grammar.pest
git commit -m "feat(ilp): add learnable_rule grammar production"
```

---

## Task 2: AST — Add `LearnableRule` Type and `Program` Field

**Files:**
- Modify: `crates/xlog-logic/src/ast.rs:458-472`

**Step 1: Add the `LearnableRule` struct**

After the existing `NeuralPredDecl` struct (around line 349), add:

```rust
/// A learnable rule template parameterized by a named tensor mask.
/// Used for differentiable ILP — the mask selects which (body1, body2, head)
/// combinations are active during execution.
#[derive(Debug, Clone)]
pub struct LearnableRule {
    pub mask_name: String,
    pub head: Atom,
    pub body: Vec<BodyLiteral>,
}
```

**Step 2: Add the field to `Program`**

In the `Program` struct (line ~458), add after `neural_predicates`:

```rust
    pub learnable_rules: Vec<LearnableRule>,
```

**Step 3: Initialize the field in `Program::new()`**

In `Program::new()` (or `Default` impl), add:

```rust
    learnable_rules: Vec::new(),
```

**Step 4: Verify it compiles**

Run: `cargo check -p xlog-logic 2>&1 | head -30`

Expected: Compile succeeds. May have warnings about unused field.

**Step 5: Commit**

```bash
git add crates/xlog-logic/src/ast.rs
git commit -m "feat(ilp): add LearnableRule AST type and Program field"
```

---

## Task 3: Parser — Dispatch and Build Learnable Rules

**Files:**
- Modify: `crates/xlog-logic/src/parser.rs:69-122` (dispatch) and add `build_learnable_rule` function
- Test: `crates/xlog-logic/tests/logic/learnable.xlog` (new)
- Test: `crates/xlog-logic/tests/integration_tests.rs` (add test)

**Step 1: Create test fixture file**

Create `crates/xlog-logic/tests/logic/learnable.xlog`:

```
// Learnable ILP test program
edge(1, 2).
edge(2, 3).
edge(3, 4).

learnable(W_mask) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).

?- reach(1, N).
```

**Step 2: Write the failing test**

Add to `crates/xlog-logic/tests/integration_tests.rs`:

```rust
// =============================================================================
// Learnable Rule Tests (ILP)
// =============================================================================

#[test]
fn test_parse_learnable_rule() {
    let input = include_str!("logic/learnable.xlog");
    let result = parse_program(input);
    assert!(
        result.is_ok(),
        "Failed to parse learnable program: {:?}",
        result.err()
    );

    let program = result.unwrap();
    assert_eq!(program.learnable_rules.len(), 1);

    let lr = &program.learnable_rules[0];
    assert_eq!(lr.mask_name, "W_mask");
    assert_eq!(lr.head.predicate, "reach");
    assert_eq!(lr.head.terms.len(), 2);
    assert_eq!(lr.body.len(), 2);
}

#[test]
fn test_parse_learnable_rule_preserves_normal_rules() {
    let input = include_str!("logic/learnable.xlog");
    let program = parse_program(input).unwrap();

    // 3 facts (edge) are in program.rules
    assert_eq!(program.facts().count(), 3);
    // The learnable rule is NOT in program.rules
    assert_eq!(program.proper_rules().count(), 0);
    // It's in learnable_rules
    assert_eq!(program.learnable_rules.len(), 1);
    // Query still parsed
    assert_eq!(program.queries.len(), 1);
}
```

**Step 3: Run test to verify it fails**

Run: `cargo test -p xlog-logic --test integration_tests test_parse_learnable 2>&1 | tail -20`

Expected: FAIL — `Rule::learnable_rule` is not matched in `build_statement`.

**Step 4: Implement parser dispatch and builder**

In `crates/xlog-logic/src/parser.rs`, add the dispatch arm in `build_statement` (line ~117, before the `_ => {}` catch-all):

```rust
            Rule::learnable_rule => {
                program.learnable_rules.push(build_learnable_rule(inner)?);
            }
```

Add the builder function (after `build_neural_pred_decl` or at end of file):

```rust
/// Build a learnable rule from a parsed pair.
/// Grammar: learnable_rule = { "learnable" ~ "(" ~ ident ~ ")" ~ "::" ~ head ~ ":-" ~ body ~ "." }
/// RD-33: Uses build_head (not build_atom) because grammar produces `head` pair.
fn build_learnable_rule(pair: Pair<'_, Rule>) -> Result<LearnableRule> {
    let mut inner = pair.into_inner();
    let mask_name = inner
        .next()
        .ok_or_else(|| XlogError::Parse("Missing learnable mask name".into()))?
        .as_str()
        .to_string();
    let head = build_head(
        inner
            .next()
            .ok_or_else(|| XlogError::Parse("Missing learnable head".into()))?,
    )?;
    let body = build_body(
        inner
            .next()
            .ok_or_else(|| XlogError::Parse("Missing learnable body".into()))?,
    )?;
    Ok(LearnableRule {
        mask_name,
        head,
        body,
    })
}
```

Add the import for `LearnableRule` at the top of parser.rs (in the `use` block for ast types):

```rust
use crate::ast::LearnableRule;
```

**Step 5: Run test to verify it passes**

Run: `cargo test -p xlog-logic --test integration_tests test_parse_learnable 2>&1 | tail -20`

Expected: 2 tests PASS.

**Step 6: Commit**

```bash
git add crates/xlog-logic/src/parser.rs crates/xlog-logic/tests/logic/learnable.xlog crates/xlog-logic/tests/integration_tests.rs
git commit -m "feat(ilp): parse learnable rules into LearnableRule AST"
```

---

## Task 4: Stratification — Add Learnable Rule Edges

**Files:**
- Modify: `crates/xlog-logic/src/stratify.rs:51-80`

**Step 1: Write the failing test**

Add to `crates/xlog-logic/tests/integration_tests.rs`:

```rust
#[test]
fn test_stratify_with_learnable_rule() {
    let input = r#"
        edge(1, 2).
        edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;
    let program = parse_program(input).unwrap();
    let result = stratify(&program);
    assert!(
        result.is_ok(),
        "Stratification failed: {:?}",
        result.err()
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-logic --test integration_tests test_stratify_with_learnable 2>&1 | tail -20`

Expected: May pass or fail depending on whether stratify ignores unknown predicates. Run it to find out.

**Step 3: Add learnable rule edges to `build_dependency_graph`**

In `crates/xlog-logic/src/stratify.rs`, in the `build_dependency_graph` function (around line 80, after the existing rule iteration), add:

```rust
    // Learnable rules: head depends on body predicates.
    // At runtime, TensorMaskedJoin dynamically selects which relations
    // to join, but for stratification we conservatively register the
    // template's body predicates as positive dependencies.
    for lr in &program.learnable_rules {
        let head = &lr.head.predicate;
        graph.add_predicate(head.clone());
        for body_lit in &lr.body {
            if let Some(atom) = body_lit.atom() {
                graph.add_predicate(atom.predicate.clone());
                graph.add_edge(head.clone(), atom.predicate.clone(), DepType::Positive);
            }
        }
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-logic --test integration_tests test_stratify_with_learnable 2>&1 | tail -20`

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-logic/src/stratify.rs crates/xlog-logic/tests/integration_tests.rs
git commit -m "feat(ilp): add learnable rule edges to dependency graph"
```

---

## Task 5: IR — Add `TensorMaskedJoin` RirNode Variant

**Files:**
- Modify: `crates/xlog-ir/src/rir.rs:110-226`

**Step 1: Add the variant**

In `crates/xlog-ir/src/rir.rs`, add to the `RirNode` enum (after `Fixpoint`, before the closing `}`):

```rust
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
```

**Step 2: Add `collect_relations` arm**

In `collect_relations` (line ~193), add the arm for `TensorMaskedJoin`. This match is exhaustive with no `_` wildcard, so it will fail to compile without this arm:

```rust
            RirNode::TensorMaskedJoin { rel_index, .. } => {
                for (rel_id, _) in rel_index {
                    rels.push(*rel_id);
                }
            }
```

**Step 3: Verify it compiles**

Run: `cargo check -p xlog-ir 2>&1 | head -20`

Expected: Compile succeeds for xlog-ir. Downstream crates (xlog-logic, xlog-runtime) will fail due to their exhaustive matches — that's expected and fixed in Tasks 6 and 8.

**Step 4: Commit**

```bash
git add crates/xlog-ir/src/rir.rs
git commit -m "feat(ilp): add TensorMaskedJoin RirNode variant"
```

---

## Task 6: Optimizer — Add Pass-Through Arms for TensorMaskedJoin

**Files:**
- Modify: `crates/xlog-logic/src/optimizer.rs`

The optimizer has 3 exhaustive match functions that must handle every `RirNode` variant. `TensorMaskedJoin` is leaf-like — no pushdown, fixed width, fixed cost.

**Step 1: Add arms to all 3 functions**

In `crates/xlog-logic/src/optimizer.rs`:

**predicate_pushdown** (line ~258, in the match on `node`): Add before the closing `}` of the match:

```rust
            RirNode::TensorMaskedJoin { .. } => node, // Leaf-like: no pushdown
```

**estimate_width** (line ~569, in the match on `node`): Add before the closing `}`:

```rust
            // RD-27: Optimizer schemas are HashMap<RelId, Schema>.
            // Use head_rel_id (not head_rel_name) for lookup.
            RirNode::TensorMaskedJoin { head_rel_id, .. } => self
                .schemas
                .get(head_rel_id)
                .map(|s| s.arity())
                .unwrap_or(2),
```

**estimate_cost** (line ~768, in the match on `node`): Add before the closing `}`:

```rust
            RirNode::TensorMaskedJoin {
                max_active_rules, ..
            } => PlanCost {
                rows: *max_active_rules as u64,
                cpu_cost: *max_active_rules as f64 * 100.0,
                gpu_mem: *max_active_rules as u64 * 1024,
                transfers: 1,
            },
```

Also check if there's a `find_column_relation` function with a `_ => None` catch-all — if it's exhaustive, add a `TensorMaskedJoin` arm there too. Based on the RFC, it already has `_ => None`.

**Step 2: Verify it compiles**

Run: `cargo check -p xlog-logic 2>&1 | head -20`

Expected: xlog-logic compiles successfully.

**Step 3: Commit**

```bash
git add crates/xlog-logic/src/optimizer.rs
git commit -m "feat(ilp): add TensorMaskedJoin arms in optimizer"
```

---

## Task 7: Lowering — Lower `LearnableRule` to `TensorMaskedJoin`

**Files:**
- Modify: `crates/xlog-logic/src/lower.rs`
- Test: `crates/xlog-logic/tests/integration_tests.rs`

**Step 1: Write the failing test**

Add to `crates/xlog-logic/tests/integration_tests.rs`:

```rust
#[test]
fn test_compile_learnable_rule_produces_tmj() {
    let input = r#"
        edge(1, 2).
        edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;
    let mut compiler = Compiler::new();
    let program = parse_program(input).unwrap();
    let plan = compiler.compile_program(&program);
    assert!(
        plan.is_ok(),
        "Compilation failed: {:?}",
        plan.err()
    );

    // Verify we can find a TensorMaskedJoin in the plan
    let plan = plan.unwrap();
    let has_tmj = plan.rules_by_scc.iter().flatten().any(|rule| {
        matches!(rule.body, xlog_ir::rir::RirNode::TensorMaskedJoin { .. })
    });
    assert!(has_tmj, "Expected TensorMaskedJoin in compiled plan");
}

#[test]
fn test_learnable_rule_body_validation() {
    // Body must have exactly 2 positive atoms
    let input = r#"
        learnable(W) :: h(X) :- b1(X, Z).
    "#;
    let mut compiler = Compiler::new();
    let program = parse_program(input).unwrap();
    let result = compiler.compile_program(&program);
    assert!(result.is_err(), "Should reject single-body learnable rule");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p xlog-logic --test integration_tests test_compile_learnable 2>&1 | tail -20`

Run: `cargo test -p xlog-logic --test integration_tests test_learnable_rule_body 2>&1 | tail -20`

Expected: Both FAIL.

**Step 3: Implement `lower_learnable_rule` and wire into `lower_program`**

In `crates/xlog-logic/src/lower.rs`:

Add the `lower_learnable_rule` method to the `Compiler` impl (or whichever struct owns lowering):

```rust
/// Lower a learnable rule template into a TensorMaskedJoin node.
/// RD-34: Validates body has exactly 2 positive atoms.
/// RD-36: Sorts rel_index by RelId for deterministic tensor dimension mapping.
/// RD-30: Uses get_or_create_rel_id for head (handles head-only predicates).
fn lower_learnable_rule(&mut self, rule: &LearnableRule) -> Result<RirNode> {
    // RD-34: Validate body shape
    if rule.body.len() != 2 {
        return Err(XlogError::Compilation(format!(
            "learnable rule '{}' requires exactly 2 body literals, got {}",
            rule.mask_name,
            rule.body.len()
        )));
    }
    for (idx, lit) in rule.body.iter().enumerate() {
        match lit {
            BodyLiteral::Positive(_) => {}
            _ => {
                return Err(XlogError::Compilation(format!(
                    "learnable rule '{}' body[{}]: only positive atoms allowed",
                    rule.mask_name, idx
                )));
            }
        }
    }

    // RD-36: Sort by RelId for deterministic mapping
    let mut rel_index: Vec<(RelId, String)> = self
        .rel_ids()
        .iter()
        .map(|(name, id)| (*id, name.clone()))
        .collect();
    rel_index.sort_by_key(|(id, _)| id.0);
    let schema_size = rel_index.len();

    let (left_keys, right_keys) =
        self.extract_template_join_keys(&rule.body[0], &rule.body[1])?;

    let head_rel_name = rule.head.predicate.clone();
    // RD-30: Allocate lazily — head-only predicates may not have a RelId yet
    let head_rel_id = self.get_or_create_rel_id(&head_rel_name);

    Ok(RirNode::TensorMaskedJoin {
        mask_name: rule.mask_name.clone(),
        schema_size,
        left_keys,
        right_keys,
        rel_index,
        head_rel_name,
        head_rel_id,
        max_active_rules: 32,
    })
}
```

Add the `extract_template_join_keys` helper:

```rust
/// Extract join keys from two body literals' shared variables.
/// For `b1(X, Z), b2(Z, Y)`, the shared variable Z gives left_keys=[1], right_keys=[0].
fn extract_template_join_keys(
    &self,
    left: &BodyLiteral,
    right: &BodyLiteral,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let left_atom = left.atom().ok_or_else(|| {
        XlogError::Compilation("Learnable body[0] is not an atom".into())
    })?;
    let right_atom = right.atom().ok_or_else(|| {
        XlogError::Compilation("Learnable body[1] is not an atom".into())
    })?;

    let mut left_keys = Vec::new();
    let mut right_keys = Vec::new();

    for (li, lt) in left_atom.terms.iter().enumerate() {
        if let Some(lname) = lt.variable_name() {
            for (ri, rt) in right_atom.terms.iter().enumerate() {
                if let Some(rname) = rt.variable_name() {
                    if lname == rname {
                        left_keys.push(li);
                        right_keys.push(ri);
                    }
                }
            }
        }
    }

    Ok((left_keys, right_keys))
}
```

Wire into `lower_program` — add before `Ok(builder.build())` (line ~367):

```rust
        // Lower learnable rules (RD-32)
        // Pre-allocate RelIds for ALL learnable predicates (heads + bodies)
        // so every lower_learnable_rule snapshot is complete.
        for learnable in &program.learnable_rules {
            self.get_or_create_rel_id(&learnable.head.predicate);
            for lit in &learnable.body {
                if let BodyLiteral::Positive(atom) = lit {
                    self.get_or_create_rel_id(&atom.predicate);
                }
            }
        }
        for learnable in &program.learnable_rules {
            let head_pred = &learnable.head.predicate;
            let scc_id = self.find_scc_for_predicate(head_pred);
            let body = self.lower_learnable_rule(learnable)?;
            let meta = self.create_meta_for_predicate(head_pred);
            builder.add_rule(
                scc_id,
                CompiledRule {
                    head: head_pred.clone(),
                    body,
                    meta,
                },
            );
        }
```

Add the import at the top of lower.rs:

```rust
use crate::ast::LearnableRule;
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p xlog-logic --test integration_tests test_compile_learnable 2>&1 | tail -20`

Run: `cargo test -p xlog-logic --test integration_tests test_learnable_rule_body 2>&1 | tail -20`

Expected: Both PASS.

**Step 5: Commit**

```bash
git add crates/xlog-logic/src/lower.rs crates/xlog-logic/tests/integration_tests.rs
git commit -m "feat(ilp): lower LearnableRule to TensorMaskedJoin via lower_program"
```

---

## Task 8: Runtime — ILP Registry and Executor Match Arms

**Files:**
- Create: `crates/xlog-runtime/src/ilp_registry.rs`
- Modify: `crates/xlog-runtime/src/lib.rs`
- Modify: `crates/xlog-runtime/src/executor.rs`

**Step 1: Create the ILP registry module**

Create `crates/xlog-runtime/src/ilp_registry.rs`:

```rust
//! ILP (Inductive Logic Programming) registry for tensor mask management.

use std::collections::HashMap;
use xlog_core::XlogError;
use xlog_cuda::{CudaBuffer, CudaKernelProvider};

/// Reads the device-side row count using only public APIs (RD-22).
/// `CudaKernelProvider::device_row_count` is private (provider.rs:6904).
pub fn read_device_row_count(
    provider: &CudaKernelProvider,
    buffer: &CudaBuffer,
) -> Result<usize, XlogError> {
    let mut host_rows = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to read row count: {}", e)))?;
    Ok(host_rows[0] as usize)
}

/// Registry for ILP tensor masks.
pub struct IlpRegistry {
    masks: HashMap<String, IlpMask>,
}

/// A registered ILP mask pair (hard + soft) with schema size.
pub struct IlpMask {
    /// Flat 1D CudaBuffer (N*N*N f32 elements), imported via DLPack.
    /// Python flattens 3D tensor to 1D before export (RD-16).
    pub hard: CudaBuffer,
    pub soft: CudaBuffer,
    pub schema_size: usize,
}

/// Tag metadata from TensorMaskedJoin execution.
/// Does NOT store CudaBuffer clones (RD-17: CudaBuffer is not Clone).
pub struct IlpTaggedResult {
    pub entries: Vec<IlpTagEntry>,
}

/// Metadata for a single active rule (i,j,k) and its result cardinality.
pub struct IlpTagEntry {
    pub i: u32,
    pub j: u32,
    pub k: u32,
    pub num_rows: u32,
}

impl IlpRegistry {
    pub fn new() -> Self {
        Self {
            masks: HashMap::new(),
        }
    }

    pub fn insert_mask(
        &mut self,
        name: String,
        hard: CudaBuffer,
        soft: CudaBuffer,
        schema_size: usize,
    ) {
        self.masks
            .insert(name, IlpMask { hard, soft, schema_size });
    }

    pub fn get_mask(&self, name: &str) -> Option<&IlpMask> {
        self.masks.get(name)
    }
}
```

**Step 2: Export from `lib.rs`**

In `crates/xlog-runtime/src/lib.rs`, add:

```rust
pub mod ilp_registry;
pub use ilp_registry::{IlpRegistry, IlpTagEntry, IlpTaggedResult, read_device_row_count};
```

**Step 3: Add fields and accessors to Executor**

In `crates/xlog-runtime/src/executor.rs`:

Add imports at top:

```rust
use crate::ilp_registry::{IlpRegistry, IlpTaggedResult};
```

Add fields to the `Executor` struct (after `profiler`):

```rust
    ilp_registry: IlpRegistry,
    ilp_last_result: Option<IlpTaggedResult>,
```

Initialize in `new_with_config` (in the `Self { ... }` block):

```rust
            ilp_registry: IlpRegistry::new(),
            ilp_last_result: None,
```

Add accessor methods (after `store_mut`):

```rust
    /// Get a mutable reference to the ILP registry (RD-35).
    pub fn ilp_registry_mut(&mut self) -> &mut IlpRegistry {
        &mut self.ilp_registry
    }

    /// Get the last ILP tagged result (RD-35).
    pub fn ilp_last_result(&self) -> Option<&IlpTaggedResult> {
        self.ilp_last_result.as_ref()
    }
```

**Step 4: Add exhaustive match arms**

There are 4 match sites that must handle `TensorMaskedJoin`:

**`contains_non_monotonic_ops`** (line ~465):
```rust
            RirNode::TensorMaskedJoin { .. } => false,
```

**`collect_scan_rels`** (line ~1248):
```rust
            RirNode::TensorMaskedJoin { rel_index, .. } => {
                for (rel_id, _) in rel_index {
                    out.push(*rel_id);
                }
            }
```

**`rewrite_scan_nth_impl`** (line ~1293):
```rust
            RirNode::TensorMaskedJoin { .. } => (node.clone(), false),
```

**`execute_node`** (line ~624): Add a placeholder that will be fully implemented in Task 10:
```rust
            RirNode::TensorMaskedJoin {
                mask_name,
                schema_size,
                left_keys,
                right_keys,
                rel_index,
                head_rel_name,
                max_active_rules,
                ..
            } => self.execute_tensor_masked_join(
                mask_name,
                *schema_size,
                left_keys,
                right_keys,
                rel_index,
                head_rel_name,
                *max_active_rules,
            ),
```

Add the stub method (we implement fully in Task 10):

```rust
    /// Execute a TensorMaskedJoin node (ILP).
    /// Stub — full implementation in Task 10 after CUDA kernel is available.
    fn execute_tensor_masked_join(
        &mut self,
        _mask_name: &str,
        _schema_size: usize,
        _left_keys: &[usize],
        _right_keys: &[usize],
        _rel_index: &[(RelId, String)],
        head_rel_name: &str,
        _max_active_rules: usize,
    ) -> Result<CudaBuffer> {
        // RD-12: No-op — return empty buffer with head schema.
        self.ilp_last_result = Some(IlpTaggedResult { entries: Vec::new() });
        let schema = self
            .store
            .get(head_rel_name)
            .map(|buf| buf.schema().clone())
            .ok_or_else(|| {
                XlogError::Execution(format!(
                    "TensorMaskedJoin: head relation '{}' not found in store",
                    head_rel_name
                ))
            })?;
        self.provider.create_empty_buffer(schema)
    }
```

**Step 5: Verify it compiles**

Run: `cargo check -p xlog-runtime 2>&1 | head -30`

Expected: Compile succeeds.

**Step 6: Run existing tests to confirm no regressions**

Run: `cargo test -p xlog-runtime 2>&1 | tail -20`

Expected: All existing tests still pass.

**Step 7: Commit**

```bash
git add crates/xlog-runtime/src/ilp_registry.rs crates/xlog-runtime/src/lib.rs crates/xlog-runtime/src/executor.rs
git commit -m "feat(ilp): add IlpRegistry, executor fields, and match arms for TensorMaskedJoin"
```

---

## Task 9: CUDA Kernel — `extract_nonzero_indices` and Provider Wrapper

**Files:**
- Create: `kernels/ilp.cu`
- Modify: `crates/xlog-cuda/src/kernel_manifest_data.rs`
- Modify: `crates/xlog-cuda/src/provider.rs`

**Step 1: Create the CUDA kernel**

Create `kernels/ilp.cu`:

```c
#include <stdint.h>

/// Extract nonzero indices from a flat N*N*N binary mask.
/// For each element where mask_hard[idx] > 0.5, outputs (i, j, k) and
/// the soft-mask priority value. Caller sorts by priority and truncates.
extern "C" __global__ void extract_nonzero_indices(
    const float* mask_hard,
    const float* mask_soft,
    uint32_t N,
    uint32_t* out_i,
    uint32_t* out_j,
    uint32_t* out_k,
    float*    out_priority,
    uint32_t* active_count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = N * N * N;
    if (idx >= total) return;
    if (mask_hard[idx] > 0.5f) {
        uint32_t pos = atomicAdd(active_count, 1);
        out_i[pos] = idx / (N * N);
        out_j[pos] = (idx / N) % N;
        out_k[pos] = idx % N;
        out_priority[pos] = mask_soft[idx];
    }
}
```

**Step 2: Update kernel manifest**

In `crates/xlog-cuda/src/kernel_manifest_data.rs`, add `"ilp"` to the array and update the comment:

Change:
```rust
/// Module names matching the .cu filenames (without extension).
/// Order matches provider.rs load order. All 19 modules listed.
pub const KERNEL_CU_NAMES: &[&str] = &[
```

To:
```rust
/// Module names matching the .cu filenames (without extension).
/// Order matches provider.rs load order. All 20 modules listed.
pub const KERNEL_CU_NAMES: &[&str] = &[
```

Add `"ilp",` after `"neural",` (at the end of the array).

**Step 3: Update the assertion in provider.rs**

Find the assertion that checks `KERNEL_CU_NAMES.len() == 19` in `crates/xlog-cuda/src/provider.rs` and change it to `== 20`.

Search for: `assert!(` or `debug_assert!` or `const _: () = assert!` near the kernel manifest usage.

**Step 4a: Add re-exports to `crates/xlog-cuda/src/lib.rs`**

The existing lib.rs (line 18-23) re-exports module constants and kernel name modules from `provider`. Add `ILP_MODULE` and `ilp_kernels` to the re-export list:

```rust
pub use provider::{
    circuit_kernels, dedup_kernels, filter_kernels, groupby_kernels, join_kernels, pack_kernels,
    pir_kernels, scan_kernels, set_ops_kernels, sort_kernels, ilp_kernels, CompareOp, CudaKernelProvider,
    JoinIndexV2, JoinType, CIRCUIT_MODULE, DEDUP_MODULE, FILTER_MODULE, GROUPBY_MODULE,
    JOIN_MODULE, PACK_MODULE, PIR_MODULE, SCAN_MODULE, SET_OPS_MODULE, SORT_MODULE, ILP_MODULE,
};
```

**Step 4b: Add module constant and kernel names in provider.rs**

Near the existing module constants (lines ~182-199), add:

```rust
pub const ILP_MODULE: &str = "xlog_ilp";

pub mod ilp_kernels {
    pub const EXTRACT_NONZERO_INDICES: &str = "extract_nonzero_indices";
}
```

**Step 5: Add PTX load block in provider.rs**

Find where the last module is loaded (the `neural` block). After it, add the ILP load block following the exact same pattern:

```rust
        // ILP module
        {
            let t0 = if profiling { Some(Instant::now()) } else { None };
            let (ptx, is_cubin) = load_module_from_file("ilp", cc)?;
            device
                .inner()
                .load_ptx(
                    ptx,
                    ILP_MODULE,
                    &[ilp_kernels::EXTRACT_NONZERO_INDICES],
                )
                .map_err(|e| {
                    XlogError::Kernel(format!("Failed to load ILP module: {}", e))
                })?;
            if let Some(t0) = t0 {
                if profiling {
                    device
                        .inner()
                        .synchronize()
                        .map_err(|e| {
                            XlogError::Kernel(format!("sync after ILP load: {}", e))
                        })?;
                }
                let elapsed = t0.elapsed().as_secs_f64();
                profile.per_module_sec.push(("ilp".to_string(), elapsed));
                profile.total_sec += elapsed;
                if is_cubin {
                    profile.cubin_loaded += 1;
                } else {
                    profile.ptx_fallback += 1;
                }
            }
        }
```

**Step 6: Add the provider wrapper method**

Add the `extract_active_rule_indices` method to `CudaKernelProvider`:

```rust
    /// Extract active (i,j,k) rule indices from a flattened N×N×N mask.
    /// Returns up to `max_active` entries sorted by soft-mask priority.
    /// RD-19: Uses actual provider APIs (memory.alloc, dtoh_sync_copy_into, etc.)
    /// RD-23: try_slice returns Option (cudarc 0.19), uses ok_or_else.
    pub fn extract_active_rule_indices(
        &self,
        mask_hard: &CudaBuffer,
        mask_soft: &CudaBuffer,
        n: usize,
        max_active: usize,
    ) -> Result<Vec<(u32, u32, u32)>> {
        let total = n * n * n;
        let block_size = 256usize;
        let grid_size = (total + block_size - 1) / block_size;

        let mut out_i = self.memory().alloc::<u32>(total)?;
        let mut out_j = self.memory().alloc::<u32>(total)?;
        let mut out_k = self.memory().alloc::<u32>(total)?;
        let mut out_p = self.memory().alloc::<f32>(total)?;
        let mut count = self.memory().alloc::<u32>(1)?;

        self.device()
            .inner()
            .htod_sync_copy_into(&[0u32], &mut count)
            .map_err(|e| XlogError::Kernel(format!("ILP htod count: {}", e)))?;

        let hard_col = mask_hard
            .column(0)
            .ok_or_else(|| XlogError::Kernel("ILP hard mask has no column".into()))?;
        let soft_col = mask_soft
            .column(0)
            .ok_or_else(|| XlogError::Kernel("ILP soft mask has no column".into()))?;

        let kernel = self
            .device()
            .inner()
            .get_func(ILP_MODULE, ilp_kernels::EXTRACT_NONZERO_INDICES)
            .ok_or_else(|| {
                XlogError::Kernel("extract_nonzero_indices kernel not found".into())
            })?;

        let hard_bytes = total * std::mem::size_of::<f32>();
        let soft_bytes = total * std::mem::size_of::<f32>();
        let hard_view = self.column_bytes_view(hard_col, hard_bytes)?;
        let soft_view = self.column_bytes_view(soft_col, soft_bytes)?;

        unsafe {
            kernel
                .clone()
                .launch(
                    cudarc::driver::LaunchConfig {
                        grid_dim: (grid_size as u32, 1, 1),
                        block_dim: (block_size as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &hard_view,
                        &soft_view,
                        n as u32,
                        &mut out_i,
                        &mut out_j,
                        &mut out_k,
                        &mut out_p,
                        &mut count,
                    ),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "Failed to launch extract_nonzero_indices: {}",
                        e
                    ))
                })?;
        }

        let mut count_host = [0u32];
        self.device()
            .inner()
            .dtoh_sync_copy_into(&count, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh count: {}", e)))?;
        let active_count = count_host[0] as usize;

        if active_count == 0 {
            return Ok(Vec::new());
        }

        let mut i_host = vec![0u32; active_count];
        let mut j_host = vec![0u32; active_count];
        let mut k_host = vec![0u32; active_count];
        let mut p_host = vec![0f32; active_count];

        let out_i_view = out_i
            .try_slice(0..active_count)
            .ok_or_else(|| XlogError::Kernel("ILP slice i out of bounds".into()))?;
        let out_j_view = out_j
            .try_slice(0..active_count)
            .ok_or_else(|| XlogError::Kernel("ILP slice j out of bounds".into()))?;
        let out_k_view = out_k
            .try_slice(0..active_count)
            .ok_or_else(|| XlogError::Kernel("ILP slice k out of bounds".into()))?;
        let out_p_view = out_p
            .try_slice(0..active_count)
            .ok_or_else(|| XlogError::Kernel("ILP slice p out of bounds".into()))?;

        self.device()
            .inner()
            .dtoh_sync_copy_into(&out_i_view, &mut i_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh i: {}", e)))?;
        self.device()
            .inner()
            .dtoh_sync_copy_into(&out_j_view, &mut j_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh j: {}", e)))?;
        self.device()
            .inner()
            .dtoh_sync_copy_into(&out_k_view, &mut k_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh k: {}", e)))?;
        self.device()
            .inner()
            .dtoh_sync_copy_into(&out_p_view, &mut p_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh p: {}", e)))?;

        let mut indices: Vec<(f32, u32, u32, u32)> = (0..active_count)
            .map(|idx| (p_host[idx], i_host[idx], j_host[idx], k_host[idx]))
            .collect();
        indices.sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(max_active);

        Ok(indices.into_iter().map(|(_, i, j, k)| (i, j, k)).collect())
    }
```

**Step 7: Build (CUDA compilation required)**

Run: `cargo build -p xlog-cuda 2>&1 | tail -30`

Expected: Compile succeeds. The build.rs will compile `ilp.cu` to PTX and optionally cubin.

**Step 8: Commit**

```bash
git add kernels/ilp.cu crates/xlog-cuda/src/kernel_manifest_data.rs crates/xlog-cuda/src/provider.rs
git commit -m "feat(ilp): add extract_nonzero_indices CUDA kernel and provider wrapper"
```

---

## Task 10: Executor — Full `execute_tensor_masked_join` Implementation

**Files:**
- Modify: `crates/xlog-runtime/src/executor.rs`

**Step 1: Replace the stub with full implementation**

Replace the stub `execute_tensor_masked_join` from Task 8 with the full implementation. Reference the RFC at `docs/ilp/rfc-tensorized-ilp.md` §4.8 for the complete code.

Key logic:
1. Check for registered mask → no-op if none (RD-12)
2. Call `extract_active_rule_indices` to get (i,j,k) tuples
3. For each active rule: look up `rel_index[i]` and `rel_index[j]`, call `hash_join_v2`
4. Union results per target relation k, diff against existing, merge into store
5. Store tag metadata in `ilp_last_result`
6. Return empty buffer with head schema

The full implementation is in the RFC — copy it directly, adjusting for the actual `import` paths and `Result` types.

**Step 2: Verify it compiles**

Run: `cargo check -p xlog-runtime 2>&1 | head -20`

Expected: Compile succeeds.

**Step 3: Commit**

```bash
git add crates/xlog-runtime/src/executor.rs
git commit -m "feat(ilp): implement execute_tensor_masked_join with hash-join dispatch"
```

---

## Task 11: CUDA Kernel Test — `extract_nonzero_indices` Validation

**Files:**
- Create: `crates/xlog-cuda/tests/ilp_kernel_tests.rs`

**Step 1: Write the kernel test**

Create `crates/xlog-cuda/tests/ilp_kernel_tests.rs`:

```rust
//! Tests for the ILP CUDA kernel (extract_nonzero_indices)

use std::sync::Arc;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

fn setup_provider() -> Option<CudaKernelProvider> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    CudaKernelProvider::new(device, memory).ok()
}

fn make_mask_buffer(
    provider: &CudaKernelProvider,
    data: &[f32],
) -> xlog_cuda::CudaBuffer {
    let schema = Schema::new(vec![("c0".to_string(), ScalarType::F32)]);
    provider
        .create_buffer_from_slices(
            &[bytemuck::cast_slice(data)],
            schema,
        )
        .expect("create mask buffer")
}

#[test]
fn test_extract_nonzero_3x3x3_single_active() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let n = 3;
    let total = n * n * n; // 27
    let mut hard = vec![0.0f32; total];
    let mut soft = vec![0.0f32; total];

    // Set (i=0, j=1, k=2) active: flat index = 0*9 + 1*3 + 2 = 5
    hard[5] = 1.0;
    soft[5] = 0.9;

    let hard_buf = make_mask_buffer(&provider, &hard);
    let soft_buf = make_mask_buffer(&provider, &soft);

    let result = provider
        .extract_active_rule_indices(&hard_buf, &soft_buf, n, 32)
        .expect("kernel launch");

    assert_eq!(result.len(), 1);
    assert_eq!(result[0], (0, 1, 2));
}

#[test]
fn test_extract_nonzero_budget_cap_top_priority() {
    // RFC T2.3: 50 non-zeros, max=10 → top 10 by soft-mask priority.
    // We use N=4 (64 total) with 50 active, cap at 10, and verify the
    // returned entries are exactly the 10 with highest soft-mask values.
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let n = 4;
    let total = n * n * n; // 64
    let mut hard = vec![0.0f32; total];
    // Assign distinct priorities so top-K is deterministic
    let mut soft = vec![0.0f32; total];
    // Activate exactly 50 elements (indices 0..50)
    for idx in 0..50 {
        hard[idx] = 1.0;
        soft[idx] = (idx + 1) as f32; // priority 1..50
    }

    let hard_buf = make_mask_buffer(&provider, &hard);
    let soft_buf = make_mask_buffer(&provider, &soft);

    // Budget cap = 10
    let result = provider
        .extract_active_rule_indices(&hard_buf, &soft_buf, n, 10)
        .expect("kernel launch");

    assert_eq!(result.len(), 10, "Budget cap must truncate to 10");

    // The top 10 by priority should be flat indices 40..49 (priority 41..50).
    // Convert returned (i,j,k) back to flat indices and verify they are
    // the 10 highest-priority entries.
    let flat_indices: Vec<usize> = result
        .iter()
        .map(|(i, j, k)| (*i as usize) * n * n + (*j as usize) * n + (*k as usize))
        .collect();
    for &fi in &flat_indices {
        assert!(
            fi >= 40 && fi < 50,
            "Expected top-10 entries (flat indices 40..49), got {}",
            fi
        );
    }
}

#[test]
fn test_extract_nonzero_empty_mask() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let n = 3;
    let total = n * n * n;
    let hard = vec![0.0f32; total];
    let soft = vec![0.0f32; total];

    let hard_buf = make_mask_buffer(&provider, &hard);
    let soft_buf = make_mask_buffer(&provider, &soft);

    let result = provider
        .extract_active_rule_indices(&hard_buf, &soft_buf, n, 32)
        .expect("kernel launch");

    assert!(result.is_empty());
}
```

**Step 2: Run tests**

Run: `cargo test -p xlog-cuda --test ilp_kernel_tests 2>&1 | tail -20`

Expected: All 3 tests PASS (on GPU machine) or SKIP (on CI without GPU).

**Step 3: Commit**

```bash
git add crates/xlog-cuda/tests/ilp_kernel_tests.rs
git commit -m "test(ilp): add CUDA kernel extraction tests"
```

---

## Task 12: End-to-End Integration Tests — Parse Through Execute (RFC T3.1–T3.3)

**Files:**
- Create: `crates/xlog-runtime/tests/ilp_integration_tests.rs`

**Step 1: Write the integration tests**

These tests verify the full pipeline including fact loading (RD-26). They cover:
- T3.3: No mask registered → no-op with correct head schema
- T3.2: Empty mask → no derivations
- T3.1: Identity mask → correct join results

Create `crates/xlog-runtime/tests/ilp_integration_tests.rs`:

```rust
//! Integration tests for ILP TensorMaskedJoin execution.
//!
//! These tests require a CUDA device to run.
//! Covers RFC T3.1 (identity mask), T3.2 (empty mask), T3.3 (no mask noop).

#![allow(clippy::arc_with_non_send_sync)]

use std::collections::HashMap;
use std::sync::Arc;
use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_ir::rir::RirNode;
use xlog_logic::{parse_program, Compiler};
use xlog_runtime::{Executor, read_device_row_count};

fn setup() -> Option<(Arc<CudaKernelProvider>, Compiler)> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    let provider = Arc::new(CudaKernelProvider::new(device, memory).ok()?);
    let compiler = Compiler::new();
    Some((provider, compiler))
}

/// Helper: full relation setup per RD-15 + fact loading per RD-26.
/// Mirrors the working pattern in xlog-gpu/logic.rs:112-137.
fn setup_executor_with_facts(
    provider: &Arc<CudaKernelProvider>,
    compiler: &Compiler,
    ast: &xlog_logic::ast::Program,
    executor: &mut Executor,
) {
    // Step 1: Register relations
    for (name, rel_id) in compiler.rel_ids() {
        executor.register_relation(*rel_id, name);
    }

    // Step 2: Pre-seed schemas
    for (name, schema) in compiler.schemas() {
        let empty = provider.create_empty_buffer(schema.clone()).unwrap();
        executor.store_mut().put(name, empty);
    }

    // Step 3: Load base facts (RD-26)
    // Iterate ast.facts(), serialize to columns, upload, union into store.
    // This replicates load_facts_into_store from the RFC pyxlog helpers.
    // For Rust-side tests, we group facts by predicate, serialize terms,
    // and use create_buffer_from_slices + union.
    load_test_facts(provider, compiler, ast, executor);
}

/// Simplified test-side fact loader for integration tests.
/// Groups facts by predicate, serializes terms to column bytes,
/// uploads via create_buffer_from_slices, unions into store.
fn load_test_facts(
    provider: &Arc<CudaKernelProvider>,
    compiler: &Compiler,
    ast: &xlog_logic::ast::Program,
    executor: &mut Executor,
) {
    use std::collections::HashMap;
    let mut fact_groups: HashMap<String, Vec<Vec<i64>>> = HashMap::new();
    for rule in ast.facts() {
        let pred = &rule.head.predicate;
        let terms: Vec<i64> = rule.head.terms.iter().map(|t| {
            // AST defines Term::Integer(i64) directly (ast.rs:11),
            // NOT Term::Constant(Constant::Integer(...)).
            match t {
                xlog_logic::ast::Term::Integer(v) => *v,
                xlog_logic::ast::Term::Symbol(id) => *id as i64,
                _ => panic!("Test helper only handles Integer/Symbol terms"),
            }
        }).collect();
        fact_groups.entry(pred.clone()).or_default().push(terms);
    }

    for (pred, rows) in &fact_groups {
        if rows.is_empty() { continue; }
        let arity = rows[0].len();

        // Build column-major byte slices
        let mut columns: Vec<Vec<u8>> = vec![Vec::new(); arity];
        for row in rows {
            for (col_idx, val) in row.iter().enumerate() {
                columns[col_idx].extend_from_slice(&(*val as i64).to_ne_bytes());
            }
        }

        let schema = compiler.schemas().get(pred)
            .cloned()
            .unwrap_or_else(|| Schema::new(
                (0..arity).map(|i| (format!("c{}", i), ScalarType::I64)).collect()
            ));

        let col_refs: Vec<&[u8]> = columns.iter().map(|c| c.as_slice()).collect();
        let buf = provider.create_buffer_from_slices(&col_refs, schema).unwrap();

        // Union with existing (which is the empty seed)
        if let Some(existing) = executor.store().get(pred) {
            let merged = provider.union_gpu(existing, &buf).unwrap();
            executor.store_mut().put(pred, merged);
        } else {
            executor.store_mut().put(pred, buf);
        }
    }
}

/// T3.3: No mask registered — TensorMaskedJoin returns no-op with correct
/// head schema, no store corruption.
#[test]
fn test_tmj_no_mask_noop() {
    let Some((provider, mut compiler)) = setup() else {
        return;
    };

    let input = r#"
        edge(1, 2).
        edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;

    let ast = parse_program(input).unwrap();
    let plan = compiler.compile_program(&ast).unwrap();

    let mut executor = Executor::new(provider.clone());
    setup_executor_with_facts(&provider, &compiler, &ast, &mut executor);

    // Execute without setting a mask — should no-op gracefully (RD-12)
    let result = executor.execute_plan(&plan);
    assert!(result.is_ok(), "No-mask execution failed: {:?}", result.err());

    // Last ILP result should be empty
    let tagged = executor.ilp_last_result();
    assert!(tagged.is_some());
    assert!(tagged.unwrap().entries.is_empty());

    // Verify edge facts were loaded (RD-26 fact loading worked)
    let edge_buf = executor.store().get("edge");
    assert!(edge_buf.is_some(), "edge relation should exist in store");
    let edge_rows = read_device_row_count(&provider, edge_buf.unwrap()).unwrap();
    assert_eq!(edge_rows, 2, "edge should have 2 facts");
}

/// T3.2: Empty mask (all zeros) → no derivations.
#[test]
fn test_tmj_empty_mask_no_derivations() {
    let Some((provider, mut compiler)) = setup() else {
        return;
    };

    let input = r#"
        edge(1, 2).
        edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;

    let ast = parse_program(input).unwrap();
    let plan = compiler.compile_program(&ast).unwrap();

    let mut executor = Executor::new(provider.clone());
    setup_executor_with_facts(&provider, &compiler, &ast, &mut executor);

    // Determine schema size from compiled rel_ids
    let n = compiler.rel_ids().len();
    let total = n * n * n;

    // Create all-zero mask (no active rules)
    let hard_data = vec![0.0f32; total];
    let soft_data = vec![0.0f32; total];
    let schema_1d = Schema::new(vec![("c0".to_string(), ScalarType::F32)]);
    let hard_buf = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&hard_data)], schema_1d.clone())
        .unwrap();
    let soft_buf = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&soft_data)], schema_1d)
        .unwrap();

    executor
        .ilp_registry_mut()
        .insert_mask("W".to_string(), hard_buf, soft_buf, n);

    let result = executor.execute_plan(&plan);
    assert!(result.is_ok(), "Empty-mask execution failed: {:?}", result.err());

    let tagged = executor.ilp_last_result().unwrap();
    assert!(tagged.entries.is_empty(), "Empty mask should produce no results");
}

/// T3.1: Identity mask (edge→edge join for reach) → correct join results.
/// Sets mask so that reach(X,Y) :- edge(X,Z), edge(Z,Y) is active.
#[test]
fn test_tmj_identity_mask_correct_join() {
    let Some((provider, mut compiler)) = setup() else {
        return;
    };

    let input = r#"
        edge(1, 2).
        edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;

    let ast = parse_program(input).unwrap();
    let plan = compiler.compile_program(&ast).unwrap();

    let mut executor = Executor::new(provider.clone());
    setup_executor_with_facts(&provider, &compiler, &ast, &mut executor);

    // Build rel_index to find edge and reach indices
    let mut rel_index: Vec<(xlog_ir::RelId, String)> = compiler
        .rel_ids()
        .iter()
        .map(|(name, id)| (*id, name.clone()))
        .collect();
    rel_index.sort_by_key(|(id, _)| id.0);
    let n = rel_index.len();

    let edge_idx = rel_index.iter().position(|(_, name)| name == "edge").unwrap();
    let reach_idx = rel_index.iter().position(|(_, name)| name == "reach").unwrap();

    // Create mask: activate (edge, edge) → reach
    // Flat index = edge_idx * N^2 + edge_idx * N + reach_idx
    let total = n * n * n;
    let mut hard_data = vec![0.0f32; total];
    let mut soft_data = vec![0.0f32; total];
    let flat_idx = edge_idx * n * n + edge_idx * n + reach_idx;
    hard_data[flat_idx] = 1.0;
    soft_data[flat_idx] = 0.95;

    let schema_1d = Schema::new(vec![("c0".to_string(), ScalarType::F32)]);
    let hard_buf = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&hard_data)], schema_1d.clone())
        .unwrap();
    let soft_buf = provider
        .create_buffer_from_slices(&[bytemuck::cast_slice(&soft_data)], schema_1d)
        .unwrap();

    executor
        .ilp_registry_mut()
        .insert_mask("W".to_string(), hard_buf, soft_buf, n);

    let result = executor.execute_plan(&plan);
    assert!(result.is_ok(), "Mask execution failed: {:?}", result.err());

    // Tag metadata should have one entry: (edge, edge) → reach
    let tagged = executor.ilp_last_result().unwrap();
    assert!(!tagged.entries.is_empty(), "Identity mask should produce results");

    let entry = &tagged.entries[0];
    assert_eq!(entry.i as usize, edge_idx);
    assert_eq!(entry.j as usize, edge_idx);
    assert_eq!(entry.k as usize, reach_idx);

    // edge ⋈ edge on shared variable gives reach(1,3)
    // (edge(1,2) ⋈ edge(2,3) → reach(1,3))
    assert!(entry.num_rows >= 1, "Join should produce at least 1 row");
}
```

**Step 2: Run tests**

Run: `cargo test -p xlog-runtime --test ilp_integration_tests 2>&1 | tail -30`

Expected: All 3 tests PASS on GPU machine, SKIP otherwise.

**Step 3: Commit**

```bash
git add crates/xlog-runtime/tests/ilp_integration_tests.rs
git commit -m "test(ilp): add end-to-end integration tests (T3.1-T3.3) with fact loading"
```

---

## Task 13: Python Bindings — `CompiledIlpProgram` PyClass

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`

This is the largest single task. It adds the `IlpProgramFactory` and `CompiledIlpProgram` PyO3 classes.

**Step 1: Add imports**

At the top of `crates/pyxlog/src/lib.rs`, add:

```rust
use xlog_logic::ast::Program as AstProgram;  // RD-29: avoid collision with #[pyclass] Program
use xlog_runtime::ilp_registry::read_device_row_count;
```

**Step 2: Add helper functions**

Add the helper functions near the bottom of the file (before the module init function):

- `load_facts_into_store` — see RFC §4.9 (`CompiledIlpProgram` compile helper) for full implementation
- `extract_tmj_keys` — see RFC §4.9 for full implementation
- `push_term_bytes` — duplicate from `xlog-gpu/logic.rs:282` (~40 lines, RD-31)
- `strip_learnable_declarations` — simple line filter
- `extract_learnable_declarations` — simple line filter

**Step 3: Add `IlpProgramFactory` and `CompiledIlpProgram`**

Add the full `#[pyclass]` structs and `#[pymethods]` impls. The complete code is in RFC §4.9 and §4.10. Key methods:

- `IlpProgramFactory::compile()` — static method, returns `CompiledIlpProgram`
- `CompiledIlpProgram::set_rule_mask()` — inject DLPack tensors
- `CompiledIlpProgram::evaluate()` — re-execute with mask
- `CompiledIlpProgram::get_tagged_results()` — return tag metadata
- `CompiledIlpProgram::fact_exists()` — host-side membership check
- `CompiledIlpProgram::tagged_entries_containing_fact()` — per-fact credit
- `CompiledIlpProgram::commit_induced_rule()` — harden a discovered rule
- `CompiledIlpProgram::ilp_schema_size()` — return N
- `CompiledIlpProgram::ilp_relation_names()` — return relation names

**Step 4: Register in module init**

In the `#[pymodule]` init function, add:

```rust
    m.add_class::<IlpProgramFactory>()?;
    m.add_class::<CompiledIlpProgram>()?;
```

**Step 5: Verify it compiles**

Run: `cargo check -p pyxlog 2>&1 | head -30`

Expected: Compile succeeds.

**Step 6: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "feat(ilp): add CompiledIlpProgram PyO3 bindings for ILP"
```

---

## Task 14: Python Integration Test

**Files:**
- Create: `crates/pyxlog/tests/test_ilp.py` (or `tests/test_ilp.py` at project root)

**Step 1: Write the Python test**

```python
"""Integration test for tensorized ILP via pyxlog."""

import pytest

try:
    import pyxlog
    import torch
    import torch.nn.functional as F
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="pyxlog or torch not available")


def test_ilp_compile_and_schema():
    """Test basic ILP compilation returns correct schema."""
    source = """
        edge(1, 2).
        edge(2, 3).
        edge(3, 4).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)
    n = prog.ilp_schema_size()
    assert n > 0
    names = prog.ilp_relation_names()
    assert "edge" in names
    assert "reach" in names


def test_ilp_set_mask_and_evaluate():
    """Test mask injection and evaluation."""
    source = """
        edge(1, 2).
        edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)
    n = prog.ilp_schema_size()

    # Create mask: all zeros (no active rules)
    W = torch.zeros((n, n, n), device='cuda')
    M_hard = W.contiguous().view(-1)
    M_soft = W.contiguous().view(-1)

    prog.set_rule_mask("W", M_hard, M_soft, n)
    prog.evaluate()

    results = prog.get_tagged_results()
    assert len(results) == 0  # No active rules, no results


def test_ilp_gradient_flow():
    """Test that gradients flow through the ILP mask (RFC T4.1).

    Uses the RFC's per-fact surrogate credit architecture (§3 Gradient Flow):
    - 3D Gumbel-Softmax with dim=-1 (not flattened)
    - Per-fact credit via tagged_entries_containing_fact (RD-24)
    - Differentiable missed-positive penalty (RD-21)
    """
    source = """
        edge(1, 2).
        edge(2, 3).
        edge(3, 4).
        learnable(W_mask) :: reach(X, Y) :- body1(X, Z), body2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()

    W = torch.randn((n, n, n), requires_grad=True, device='cuda')

    # RFC §4.10: Per-(i,j) Gumbel-Softmax with dim=-1 on 3D tensor
    M_soft = F.gumbel_softmax(W, tau=0.5, hard=False, dim=-1)
    index = M_soft.max(dim=-1, keepdim=True)[1]
    M_hard = torch.zeros_like(M_soft).scatter_(-1, index, 1.0)
    M = (M_hard - M_soft).detach() + M_soft  # Straight-through

    # RD-16: Flatten to 1D for DLPack ndim==1 compliance
    prog.set_rule_mask(
        "W_mask",
        M_hard.contiguous().view(-1),
        M_soft.contiguous().view(-1),
        n,
    )
    prog.evaluate()

    # RFC §3: Per-fact surrogate credit assignment
    positive_examples = [("reach", [1, 3])]  # edge(1,2)⋈edge(2,3) should derive this
    loss = torch.tensor(0.0, device='cuda')

    for rel_name, values in positive_examples:
        contributing = prog.tagged_entries_containing_fact(rel_name, values)
        if contributing:
            # Per-fact credit: sum M_soft for all (i,j,k) that derived this fact (RD-24)
            credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
            loss = loss + (-torch.log(credit.clamp(min=1e-8)))
        else:
            # RD-21: Differentiable missed-positive penalty
            k_idx = rel_names.index(rel_name)
            penalty = -M_soft[:, :, k_idx].sum() / (n * n)
            loss = loss + penalty

    loss.backward()
    assert W.grad is not None, "Gradients must flow through ST-Gumbel-Softmax"
    assert W.grad.abs().sum().item() > 0, "Non-zero gradient expected (T4.1)"
```

**Step 2: Run test**

Run: `cd /home/dev/projects/xlog && python -m pytest crates/pyxlog/tests/test_ilp.py -v 2>&1 | tail -30`

(Adjust path based on where pytest discovers tests for this project.)

Expected: All tests PASS on GPU machine.

**Step 3: Commit**

```bash
git add crates/pyxlog/tests/test_ilp.py
git commit -m "test(ilp): add Python integration tests for ILP gradient flow"
```

---

## Task 15: Additional RFC Test Coverage (Towards M1–M4 Gates)

The RFC requires 35 tests across 6 milestones. Tasks 3, 4, 7, 11, 12, and 14 already cover a core subset. This task adds gap tests to improve coverage. 10 tests requiring advanced fixtures (recursive programs, finite-difference checks, profiler APIs, convergence benchmarks) are explicitly deferred for incremental addition during implementation.

**Files:**
- Modify: `crates/xlog-logic/tests/integration_tests.rs` (add M1 gap tests)
- Modify: `crates/xlog-cuda/tests/ilp_kernel_tests.rs` (add M2 gap tests)
- Modify: `crates/xlog-runtime/tests/ilp_integration_tests.rs` (add M3 gap tests)
- Modify: `crates/pyxlog/tests/test_ilp.py` (add M4/M5 gap tests)

### Step 1: M1 Gap Tests — Syntax & IR (xlog-logic)

Add to `crates/xlog-logic/tests/integration_tests.rs`:

```rust
// T1.2: Parse failure on malformed learnable rule
#[test]
fn test_parse_learnable_malformed_fails() {
    // Missing mask name
    let input = "learnable() :: h(X) :- b1(X, Z), b2(Z, Y).";
    assert!(parse_program(input).is_err());

    // Missing :: separator
    let input2 = "learnable(W) h(X,Y) :- b1(X,Z), b2(Z,Y).";
    assert!(parse_program(input2).is_err());
}

// T1.4: referenced_relations() includes all rel_index entries.
// Note: collect_relations is private (rir.rs:193); the public API is
// referenced_relations() (rir.rs:187) which delegates internally.
#[test]
fn test_tmj_referenced_relations_complete() {
    let input = r#"
        edge(1,2).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;
    let mut compiler = Compiler::new();
    let program = parse_program(input).unwrap();
    let plan = compiler.compile_program(&program).unwrap();

    // Find the TensorMaskedJoin node and check referenced_relations
    for rule in plan.rules_by_scc.iter().flatten() {
        if let xlog_ir::rir::RirNode::TensorMaskedJoin { rel_index, .. } = &rule.body {
            let collected = rule.body.referenced_relations();
            for (rel_id, _) in rel_index {
                assert!(
                    collected.contains(rel_id),
                    "referenced_relations missing RelId {:?}",
                    rel_id
                );
            }
            return; // Found and validated
        }
    }
    panic!("No TensorMaskedJoin found in compiled plan");
}

// T1.8: Optimizer handles TensorMaskedJoin without panic
#[test]
fn test_optimizer_handles_tmj() {
    // If compilation succeeds, the optimizer passed all 3 functions
    // (predicate_pushdown, estimate_width, estimate_cost) without panicking.
    let input = r#"
        edge(1,2).
        edge(2,3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
        ?- reach(1, N).
    "#;
    let mut compiler = Compiler::new();
    let program = parse_program(input).unwrap();
    let result = compiler.compile_program(&program);
    assert!(result.is_ok(), "Optimizer should handle TensorMaskedJoin: {:?}", result.err());
}
```

### Step 2: M2 Gap Tests — DLPack Bridge & Kernel (xlog-cuda)

Add to `crates/xlog-cuda/tests/ilp_kernel_tests.rs`:

```rust
// T2.4: ILP module loads successfully.
// ILP_MODULE and ilp_kernels are re-exported from xlog_cuda root
// (added in Task 9 Step 4a). Full path used here for robustness.
#[test]
fn test_ilp_module_loads() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };
    // If provider construction succeeded, ilp module is loaded.
    // Verify the kernel function is accessible.
    use xlog_cuda::provider::{ILP_MODULE, ilp_kernels};
    let func = provider.device().inner().get_func(
        ILP_MODULE,
        ilp_kernels::EXTRACT_NONZERO_INDICES,
    );
    assert!(func.is_some(), "extract_nonzero_indices kernel must be loadable");
}

// T2.2: Multi-element extraction (supplement to single_active test)
#[test]
fn test_extract_nonzero_multiple_active() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let n = 3;
    let total = n * n * n;
    let mut hard = vec![0.0f32; total];
    let mut soft = vec![0.0f32; total];

    // Activate 3 entries with distinct priorities
    // (0,1,2) at flat 5, priority 0.9
    hard[5] = 1.0; soft[5] = 0.9;
    // (1,0,1) at flat 10, priority 0.5
    hard[10] = 1.0; soft[10] = 0.5;
    // (2,2,0) at flat 24, priority 0.8
    hard[24] = 1.0; soft[24] = 0.8;

    let hard_buf = make_mask_buffer(&provider, &hard);
    let soft_buf = make_mask_buffer(&provider, &soft);

    let result = provider
        .extract_active_rule_indices(&hard_buf, &soft_buf, n, 32)
        .expect("kernel launch");

    assert_eq!(result.len(), 3);
    // Sorted by priority descending: (0,1,2)=0.9, (2,2,0)=0.8, (1,0,1)=0.5
    assert_eq!(result[0], (0, 1, 2));
    assert_eq!(result[1], (2, 2, 0));
    assert_eq!(result[2], (1, 0, 1));
}
```

### Step 3: M3 Gap Tests — Executor Integration (xlog-runtime)

Add to `crates/xlog-runtime/tests/ilp_integration_tests.rs`:

```rust
// T3.4: Tag metadata matches active rules
#[test]
fn test_tmj_tag_metadata_correct() {
    // Reuses the identity mask test setup but validates tag metadata fields
    let Some((provider, mut compiler)) = setup() else { return; };

    let input = r#"
        edge(1, 2).
        edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;
    let ast = parse_program(input).unwrap();
    let plan = compiler.compile_program(&ast).unwrap();
    let mut executor = Executor::new(provider.clone());
    setup_executor_with_facts(&provider, &compiler, &ast, &mut executor);

    let mut rel_index: Vec<(xlog_ir::RelId, String)> = compiler
        .rel_ids().iter()
        .map(|(name, id)| (*id, name.clone())).collect();
    rel_index.sort_by_key(|(id, _)| id.0);
    let n = rel_index.len();

    let edge_idx = rel_index.iter().position(|(_, name)| name == "edge").unwrap();
    let reach_idx = rel_index.iter().position(|(_, name)| name == "reach").unwrap();

    let total = n * n * n;
    let mut hard_data = vec![0.0f32; total];
    let mut soft_data = vec![0.0f32; total];
    let flat_idx = edge_idx * n * n + edge_idx * n + reach_idx;
    hard_data[flat_idx] = 1.0;
    soft_data[flat_idx] = 0.95;

    let schema_1d = Schema::new(vec![("c0".to_string(), ScalarType::F32)]);
    let hard_buf = provider.create_buffer_from_slices(
        &[bytemuck::cast_slice(&hard_data)], schema_1d.clone()).unwrap();
    let soft_buf = provider.create_buffer_from_slices(
        &[bytemuck::cast_slice(&soft_data)], schema_1d).unwrap();
    executor.ilp_registry_mut().insert_mask("W".to_string(), hard_buf, soft_buf, n);

    executor.execute_plan(&plan).unwrap();

    let tagged = executor.ilp_last_result().unwrap();
    assert_eq!(tagged.entries.len(), 1);
    let e = &tagged.entries[0];
    assert_eq!(e.i as usize, edge_idx, "Tag i should be edge index");
    assert_eq!(e.j as usize, edge_idx, "Tag j should be edge index");
    assert_eq!(e.k as usize, reach_idx, "Tag k should be reach index");
    assert!(e.num_rows > 0, "Tag num_rows should be positive");
}

// T3.7: Diff against existing facts — only new facts added
#[test]
fn test_tmj_diff_no_duplicate_facts() {
    // Run twice with same mask — second run should not double the facts
    let Some((provider, mut compiler)) = setup() else { return; };

    let input = r#"
        edge(1, 2).
        edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    "#;
    let ast = parse_program(input).unwrap();
    let plan = compiler.compile_program(&ast).unwrap();
    let mut executor = Executor::new(provider.clone());
    setup_executor_with_facts(&provider, &compiler, &ast, &mut executor);

    let mut rel_index: Vec<(xlog_ir::RelId, String)> = compiler
        .rel_ids().iter()
        .map(|(name, id)| (*id, name.clone())).collect();
    rel_index.sort_by_key(|(id, _)| id.0);
    let n = rel_index.len();

    let edge_idx = rel_index.iter().position(|(_, name)| name == "edge").unwrap();
    let reach_idx = rel_index.iter().position(|(_, name)| name == "reach").unwrap();

    let total = n * n * n;
    let mut hard_data = vec![0.0f32; total];
    let mut soft_data = vec![0.0f32; total];
    hard_data[edge_idx * n * n + edge_idx * n + reach_idx] = 1.0;
    soft_data[edge_idx * n * n + edge_idx * n + reach_idx] = 0.95;

    let schema_1d = Schema::new(vec![("c0".to_string(), ScalarType::F32)]);

    // First execution
    let h1 = provider.create_buffer_from_slices(
        &[bytemuck::cast_slice(&hard_data)], schema_1d.clone()).unwrap();
    let s1 = provider.create_buffer_from_slices(
        &[bytemuck::cast_slice(&soft_data)], schema_1d.clone()).unwrap();
    executor.ilp_registry_mut().insert_mask("W".to_string(), h1, s1, n);
    executor.execute_plan(&plan).unwrap();

    let reach_rows_1 = executor.store().get("reach")
        .map(|b| read_device_row_count(&provider, b).unwrap())
        .unwrap_or(0);

    // Second execution with same mask — diff should prevent duplication
    let h2 = provider.create_buffer_from_slices(
        &[bytemuck::cast_slice(&hard_data)], schema_1d.clone()).unwrap();
    let s2 = provider.create_buffer_from_slices(
        &[bytemuck::cast_slice(&soft_data)], schema_1d).unwrap();
    executor.ilp_registry_mut().insert_mask("W".to_string(), h2, s2, n);
    executor.execute_plan(&plan).unwrap();

    let reach_rows_2 = executor.store().get("reach")
        .map(|b| read_device_row_count(&provider, b).unwrap())
        .unwrap_or(0);

    assert_eq!(reach_rows_1, reach_rows_2,
        "Re-execution with same mask must not duplicate facts (diff_gpu)");
}
```

### Step 4: M4 Gap Tests — Gradient Flow (Python)

Add to `crates/pyxlog/tests/test_ilp.py`:

```python
def test_ilp_missed_positive_penalty():
    """T4.5: Missed-positive penalty produces non-zero gradient (RD-21)."""
    source = """
        edge(1, 2).
        edge(2, 3).
        learnable(W_mask) :: reach(X, Y) :- body1(X, Z), body2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()

    W = torch.randn((n, n, n), requires_grad=True, device='cuda')

    # Create a mask that does NOT produce reach(1,4)
    M_soft = F.gumbel_softmax(W, tau=0.5, hard=False, dim=-1)
    index = M_soft.max(dim=-1, keepdim=True)[1]
    M_hard = torch.zeros_like(M_soft).scatter_(-1, index, 1.0)
    M = (M_hard - M_soft).detach() + M_soft

    prog.set_rule_mask("W_mask",
                        M_hard.contiguous().view(-1),
                        M_soft.contiguous().view(-1), n)
    prog.evaluate()

    # Ask for a fact that almost certainly won't be derived
    contributing = prog.tagged_entries_containing_fact("reach", [99, 99])
    assert len(contributing) == 0, "Sanity: this fact shouldn't be derived"

    # RD-21: Differentiable missed-positive penalty
    k_idx = rel_names.index("reach")
    penalty = -M_soft[:, :, k_idx].sum() / (n * n)
    penalty.backward()
    assert W.grad is not None, "Missed-positive penalty must produce gradients"
    assert W.grad.abs().sum().item() > 0, "Non-zero gradient from penalty (T4.5)"


def test_ilp_temperature_annealing():
    """T4.4: Temperature annealing produces increasingly discrete M."""
    n = 4
    W = torch.randn((n, n, n), device='cuda')

    discreteness = []
    for tau in [2.0, 1.0, 0.5, 0.1]:
        M_soft = F.gumbel_softmax(W, tau=tau, hard=False, dim=-1)
        # Measure discreteness: max along dim=-1 approaches 1.0
        max_vals = M_soft.max(dim=-1)[0]
        discreteness.append(max_vals.mean().item())

    # Lower temperature → more discrete → higher mean-max
    for i in range(len(discreteness) - 1):
        assert discreteness[i] <= discreteness[i + 1] + 0.05, \
            f"Temperature annealing: tau decrease should increase discreteness"
```

### Step 5: M5 Gap Tests — ILP Benchmark Smoke (Python)

Add to `crates/pyxlog/tests/test_ilp.py`:

```python
def test_ilp_predecessor_benchmark_smoke():
    """T5.1 smoke: Predecessor benchmark setup compiles and runs 5 steps."""
    source = """
        zero(0). succ(0, 1). succ(1, 2). succ(2, 3). succ(3, 4).
        learnable(W_mask) :: pred(X, Y) :- body1(X, Z), body2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)
    n = prog.ilp_schema_size()

    W = torch.randn((n, n, n), requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([W], lr=0.1)

    for step in range(5):
        optimizer.zero_grad()
        M_soft = F.gumbel_softmax(W, tau=0.5, hard=False, dim=-1)
        index = M_soft.max(dim=-1, keepdim=True)[1]
        M_hard = torch.zeros_like(M_soft).scatter_(-1, index, 1.0)
        M = (M_hard - M_soft).detach() + M_soft

        prog.set_rule_mask("W_mask",
                            M_hard.contiguous().view(-1),
                            M_soft.contiguous().view(-1), n)
        prog.evaluate()

        # Simple loss: does pred(1, 0) exist?
        results = prog.get_tagged_results()
        loss = torch.tensor(0.0, device='cuda')
        for (i, j, k, nr) in results:
            if nr > 0:
                loss = loss + M_soft[i, j, k]
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

    # Smoke: completed without crash
    assert True


def test_ilp_commit_rule():
    """T5.5 + T5.6: Rule commit removes learnable, post-commit matches."""
    source = """
        edge(1, 2). edge(2, 3).
        learnable(W_mask) :: reach(X, Y) :- body1(X, Z), body2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)

    # Commit a concrete rule
    prog.commit_induced_rule("reach(X, Y) :- edge(X, Z), edge(Z, Y).")

    # After commit, the program should work without a mask
    # (T5.6: no TensorMaskedJoin in recompiled plan)
    # Evaluate should succeed without set_rule_mask
    prog.evaluate()

    # T5.5: Post-commit evaluation produces correct results
    assert prog.fact_exists("reach", [1, 3]), "reach(1,3) should be derived post-commit"
```

### Step 6: Run full test suite and verify no regressions

**Step 6a: Rust workspace tests**

Run: `cargo test --workspace --exclude pyxlog --release 2>&1 | tail -30`

Expected: All tests pass.

**Step 6b: xlog-logic tests specifically**

Run: `cargo test -p xlog-logic --test integration_tests 2>&1 | tail -30`

Expected: All learnable rule tests pass (M1 gate).

**Step 6c: xlog-cuda tests specifically**

Run: `cargo test -p xlog-cuda --test ilp_kernel_tests 2>&1 | tail -30`

Expected: All kernel tests pass (M2 gate).

**Step 6d: xlog-runtime tests specifically**

Run: `cargo test -p xlog-runtime --test ilp_integration_tests 2>&1 | tail -30`

Expected: All integration tests pass (M3 gate).

**Step 6e: Python tests**

Run: `cd /home/dev/projects/xlog && python -m pytest crates/pyxlog/tests/test_ilp.py -v 2>&1 | tail -40`

Expected: All Python tests pass (M4/M5 gates).

**Step 7: Commit (if any fixes were needed)**

```bash
git add -A
git commit -m "test(ilp): complete RFC milestone gate tests M1-M6"
```

### RFC Test Coverage Map

After Task 15, the plan covers:

| RFC Test | Plan Task | Status |
|----------|-----------|--------|
| T1.1 | Task 3 (`test_parse_learnable_rule`) | Covered |
| T1.2 | Task 15 (`test_parse_learnable_malformed_fails`) | **Added** |
| T1.3 | Task 7 (`test_compile_learnable_rule_produces_tmj`) | Covered |
| T1.4 | Task 15 (`test_tmj_collect_relations_complete`) | **Added** |
| T1.5 | Deferred (needs recursive program fixture) | Deferred |
| T1.6 | Task 3 (`test_parse_learnable_rule`) | Covered |
| T1.7 | Task 4 (`test_stratify_with_learnable_rule`) | Covered |
| T1.8 | Task 15 (`test_optimizer_handles_tmj`) | **Added** |
| T1.9 | Task 8 (compile check covers match arms) | Covered |
| T2.1 | Task 12 (`test_tmj_empty_mask_no_derivations`) | **Added** |
| T2.2 | Task 11 (`test_extract_nonzero_3x3x3_single_active` + `multiple_active`) | Covered/**Added** |
| T2.3 | Task 11 (`test_extract_nonzero_budget_cap_top_priority`) | **Fixed** |
| T2.4 | Task 15 (`test_ilp_module_loads`) | **Added** |
| T2.5 | Deferred (needs DLPack round-trip infra) | Deferred |
| T3.1 | Task 12 (`test_tmj_identity_mask_correct_join`) | **Added** |
| T3.2 | Task 12 (`test_tmj_empty_mask_no_derivations`) | **Added** |
| T3.3 | Task 12 (`test_tmj_no_mask_noop`) | Covered |
| T3.4 | Task 15 (`test_tmj_tag_metadata_correct`) | **Added** |
| T3.5 | Task 14 (`test_ilp_compile_and_schema` + `fact_exists` usage) | Covered |
| T3.6 | Task 14 (`tagged_entries_containing_fact` usage) | Covered |
| T3.7 | Task 15 (`test_tmj_diff_no_duplicate_facts`) | **Added** |
| T3.8 | Deferred (needs 3+ active rules fixture) | Deferred |
| T3.9 | Deferred (needs profiler query API) | Deferred |
| T3.10 | Deferred (needs recursive fixture) | Deferred |
| T4.1 | Task 14 (`test_ilp_gradient_flow`) | **Fixed** |
| T4.2 | Deferred (needs ground-truth rule indices) | Deferred |
| T4.3 | Deferred (finite-difference check) | Deferred |
| T4.4 | Task 15 (`test_ilp_temperature_annealing`) | **Added** |
| T4.5 | Task 15 (`test_ilp_missed_positive_penalty`) | **Added** |
| T5.1 | Task 15 (`test_ilp_predecessor_benchmark_smoke`) | **Added** |
| T5.2 | Deferred (convergence test) | Deferred |
| T5.3 | Deferred (mutual recursion) | Deferred |
| T5.4 | Deferred (transitive closure) | Deferred |
| T5.5 | Task 15 (`test_ilp_commit_rule`) | **Added** |
| T5.6 | Task 15 (`test_ilp_commit_rule`) | **Added** |

**Coverage: 25/35 RFC tests addressed (71%). This does NOT satisfy the RFC's full-pass milestone gates (M1-M6), which require all 35 tests passing. The 10 deferred tests require advanced fixtures (recursive programs, finite-difference infrastructure, profiler APIs, convergence benchmarks) and will be added incrementally during implementation as the infrastructure matures. Full milestone gate compliance is a post-implementation validation step, not a plan deliverable.**

---

## Dependency Graph

```
Task 1 (grammar) → Task 2 (AST) → Task 3 (parser) → Task 4 (stratify)
                                                    ↘
Task 5 (IR) → Task 6 (optimizer) → Task 7 (lowering) → Task 8 (executor stubs)
                                                           ↓
Task 9 (CUDA kernel) → Task 10 (executor full) → Task 11 (kernel test)
                                                → Task 12 (integration tests T3.1-T3.3)
                                                → Task 13 (pyxlog) → Task 14 (Python test)
                                                                   ↘
                                                Task 15 (additional RFC test coverage)
```

Tasks 1-8 can proceed sequentially without a GPU (pure Rust compilation).
Tasks 9-15 require a CUDA device for testing.
Task 15 depends on ALL prior tasks — it adds gap tests across all crates and validates milestone gates.
