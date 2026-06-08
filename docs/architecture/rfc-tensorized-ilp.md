# RFC: Tensorized Differentiable ILP in XLOG

> **Implementation status:** The design in this RFC is **shipped** (since v0.5.0; current as of v0.8.6). See `docs/architecture/dilp-training.md` for the current architecture and `docs/architecture/dilp-showcase-report.md` for validation. This document is retained as the design reference that motivates and records the resolved decisions (RD-1 through RD-27).
>
> **Entry points:** Python trainer at `crates/pyxlog/python/pyxlog/ilp/trainer.py` (`train_only`, `train_and_promote`). Rust registry at `crates/xlog-runtime/src/ilp_registry.rs`. GPU kernels at `kernels/ilp.cu`, `kernels/ilp_credit.cu`.

**Status:** Design reference (v5, implemented in v0.5.0; current as of v0.8.6)
**Date:** 2026-03-05
**Supersedes:** earlier RFC drafts v1–v4.4

### v5 Changes (2026-03-05)

This revision reconciles RFC text with implemented behavior post-v0.4.0-GA:

- **RD-6 updated**: Per-(i,j) argmax replaced by global 1D Gumbel-softmax/top-k over sparse candidate list. Reflects actual `SparseMaskBackend` semantics.
- **Performance target scoped**: Zero host round-trip target now applies to the loss/grad compute path (`compute_ilp_loss_grad_gpu`). Witness/convergence checks (`batch_fact_membership`) still perform bounded D2H reads; these are outside the inner gradient step.
- **RD-17 updated**: `IlpTagEntry` retains `buffer: Option<CudaBuffer>` for batch credit queries. The no-buffer constraint is removed.
- **Benchmark gates updated**: Phase 5 acceptance tests now use reach/grandparent/colleague/plus2 (4 stages x 5 seeds = 20/20 reliability gate), replacing the original Predecessor/Even-Odd/Ancestor set.
- **TIIPS phase**: Remains optional and unimplemented. No change in status.
- **GPU-resident loss path**: New §4.8 documents `compute_ilp_loss_grad_gpu` with zero-D2H guarantee (non-chunked), strict mode gate, and COO memory cap.

---

## 1. Objective

Enable XLOG to dynamically induce logical rules during execution by treating
the discrete rule space as a continuous Adjacency Tensor (W) trained via
gradient descent. Bypass the AST-compilation bottleneck by executing a
**Tensorized Super-Graph** natively on the GPU, gated by a Straight-Through
(ST) Gumbel-Softmax binary mask (M) provided via DLPack.

Optionally accelerate convergence via TIIPS-inspired **Selective Transductive
Guidance**, where a neural backbone predicts intermediate subgoals when
inductive gradient descent plateaus.

### Performance Target

The GPU loss/gradient compute path (`compute_ilp_loss_grad_gpu`) must
complete without host round-trips (zero D2H) in the non-chunked path.
Mask application and evaluation run on-device. Witness/convergence checks
(`batch_fact_membership`) perform bounded D2H reads (bool mask bytes) and
are outside the inner gradient step. The chunked COO fallback (activated
when COO allocation exceeds `coo_memory_cap`) uses bounded D2H by design;
`set_strict_zero_dtoh(True)` rejects this fallback for CI gates.

---

## 2. Resolved Design Decisions

### RD-1: API Naming

Standardize on `set_rule_mask()`, matching existing pyxlog naming
(`set_train_mode`, `set_active_tensor_source`).

### RD-2: TensorMaskedJoin Schema

Canonical definition in §4.2.

### RD-3: Executor Access Pattern

`TensorMaskedJoin` uses name-based `RelationStore` with bidirectional
`rel_names`/`name_to_rel` mappings (same as `execute_scan()`).

### RD-4: Kernel File Location

New `kernels/ilp.cu`. Manifest updated from 19 to 20 modules.

### RD-5: Gradient Mechanism

ILP gradients flow through **PyTorch's ST-Gumbel-Softmax**. xlog-prob
is not involved. See §3.

### RD-6: Cardinality of Active Rules

~~Per-(i,j) argmax selects one k, producing at most N^2 active rules.~~
**(v5):** `SparseMaskBackend` uses 1D candidate logits with global
Gumbel-softmax and top-k selection over a flat candidate list, producing
at most `budget` active rules (default: `max_active_rules`). The per-(i,j)
argmax semantics from v4 are superseded. `DenseMaskBackend` (debug fallback)
retains the N^3 dense path for parity testing.

### RD-7–RD-11: (Unchanged from v3)

See prior versions for full text of notation, hardware scope, stratum
placement, pyxlog integration target, and rule commit mechanism.

### RD-12: Lifecycle — TensorMaskedJoin Without Mask

`execute_tensor_masked_join()` returns an empty buffer with the **head
relation's schema** (not `Schema::new(vec![])`) when no mask is registered.
The `execute_non_recursive_scc` path at executor.rs:597-604 stores this
into the head relation's slot; using the correct schema prevents corruption.
The head relation name is stored in `TensorMaskedJoin.head_rel_name` and
the schema is looked up from the pre-seeded store (see RD-15). This avoids
the fragile `rel_index.first()` fallback which may not correspond to the
actual head relation.

### RD-13: Commit Semantics — Base Source Separation

`CompiledIlpProgram` stores `base_source` (without learnable declarations)
and `learnable_source` separately. `commit_induced_rule()` appends to
`base_source` and recompiles without learnable template.

### RD-14: Executor Constructor

`Executor::new(provider: Arc<CudaKernelProvider>) -> Self`. No RuntimeConfig
argument (it defaults internally).

### RD-15: Relation Setup Before Execution (v4 — was Blocker #5)

RFC v3 called `executor.execute_plan()` without prior relation registration
or schema seeding. The actual working pattern (from `xlog-gpu/logic.rs:112-137`)
requires three setup steps before execution:

1. `executor.register_relation(rel_id, name)` for every compiled relation
2. `executor.store_mut().put(name, provider.create_empty_buffer(schema)?)` to
   pre-seed schemas
3. Load base facts (execute fact rules that populate EDB relations)

Then `executor.execute_plan(&plan)` succeeds because all relations have
registered names and seeded schemas.

### RD-16: DLPack 3D Tensor Incompatibility (v4 — was Blocker #2)

`from_dlpack_tensors()` enforces `ndim == 1` (dlpack.rs:221). A 3D PyTorch
tensor cannot be imported directly. **Resolution:** Python flattens the mask
to 1D before DLPack export: `M_hard.contiguous().view(-1)`. The ILP registry
stores `schema_size` N so the flat N^3-element buffer is interpreted as
N×N×N by the extraction kernel (which already indexes as
`i = idx/(N*N), j = (idx/N)%N, k = idx%N`).

### RD-17: CudaBuffer Is Not Clone (v4 — was Blocker #4)

`CudaBuffer` does not implement `Clone`. Tagged results cannot store clones.
**Resolution:** `IlpTagEntry` stores only metadata `(i, j, k, num_rows)`.
For per-fact membership checks, the surrogate loss computation re-queries the
executor's store for the head relation and uses the new
`download_and_check_membership()` helper (which uses existing
`download_column_u32()`/`download_column_i64()` methods).

### RD-18: `contains_tuple` Does Not Exist (v4 — was Blocker #3)

No such method exists on `CudaKernelProvider`. **Resolution:** Implement
`contains_tuple_host()` as a new method on `CompiledIlpProgram` that
downloads the relevant columns via existing `download_column_*` helpers
and checks membership on the host. For small result sets (typical in ILP),
this is negligible. The API boundary is in pyxlog, not in xlog-cuda.

### RD-19: Provider Wrapper API Corrections (v4 — was Blocker #1)

RFC v3 used `column(0)?`, `.to_host()`, `.slice()` on `TrackedCudaSlice`.
These do not exist. The actual patterns:
- `memory.alloc::<T>(len)` returns `TrackedCudaSlice<T>`
- Host copy: `device.inner().dtoh_sync_copy_into(&slice, &mut host_vec)`
- `CudaBuffer::column(idx)` returns `Option<&CudaColumn>`, not `Result`
- For downloading columns: `provider.download_column_u32(buf, col_idx)`

The extraction kernel wrapper is rewritten using only these actual APIs.

### RD-20: Exhaustive RirNode Match Sites (v4 — was High #7)

Adding `TensorMaskedJoin` to `RirNode` requires arms in **all** exhaustive
matches. The complete list:

| File | Function | Required Arm |
|------|----------|-------------|
| `executor.rs:463` | `contains_non_monotonic_ops` | `false` (monotonic) |
| `executor.rs:620` | `execute_node` | dispatch to `execute_tensor_masked_join` |
| `executor.rs:1245` | `collect_scan_rels` | push all `rel_index` RelIds |
| `executor.rs:1291` | `rewrite_scan_nth_impl` | `(node.clone(), false)` |
| `optimizer.rs:258` | `predicate_pushdown` | pass-through (leaf-like) |
| `optimizer.rs:569` | `estimate_width` | head relation arity |
| `optimizer.rs:768` | `estimate_cost` | fixed small cost |
| `optimizer.rs:1127` | `find_column_relation` | `None` (already `_ => None`) |
| `rir.rs:193` | `collect_relations` | push all `rel_index` RelIds (Vec-based) |

---

## 3. Architecture Overview

### Pipeline Separation

ILP lives **entirely in the xlog-runtime executor pipeline**. The xlog-prob
static circuit pipeline is not involved.

```
Pipeline A: xlog-prob (NOT used for ILP)
├─ Static PIR graph compiled at Program::compile() time
├─ Immutable d-DNNF circuit
├─ Neural predicates: swap weight slots, not graph structure
└─ Used for: probabilistic inference, neural-symbolic training

Pipeline B: xlog-runtime (USED for ILP)
├─ Executor evaluates RirNode trees including TensorMaskedJoin
├─ Dynamic: mask tensor changes between evaluate() calls
├─ Returns tagged metadata {(i,j,k,num_rows)}
└─ Used for: logic evaluation, ILP rule execution
```

### Gradient Flow Architecture

Gradients do NOT flow through XLOG. They flow through PyTorch's autograd:

```
PyTorch Autograd Graph              XLOG Executor (non-differentiable)
─────────────────────               ──────────────────────────────────
W (requires_grad=True)
  │
  ├─ Per-(i,j) Gumbel-Softmax (dim=-1 on 3D tensor)
  │   → M_soft (N x N x N, differentiable)
  │
  ├─ ST hard snap (argmax per (i,j) slice)
  │   → M_hard (N x N x N, detached)
  │   → M = (M_hard - M_soft).detach() + M_soft
  │
  ├─ DLPack(M_hard.view(-1)) ──────► Executor receives flat 1D mask
  │  DLPack(M_soft.view(-1))          (RD-16: flattened for ndim==1)
  │                                   extract_nonzero_indices
  │                                   hash_join_v2 dispatches
  │                                   → tag metadata {(i,j,k,n_rows)}
  │           ◄────────────────────
  │
  ├─ Surrogate loss (per-fact credit assignment):
  │   For each positive example (x,y) in relation k:
  │     contributing_rules = program.tagged_entries_with_fact(k, (x,y))
  │     if contributing_rules:
  │       credit = sum(M_soft[i,j,k] for (i,j,k) in contributing_rules)
  │       loss += -log(clamp(credit, 1e-8))
  │     else:
  │       # Differentiable missed-positive penalty (RD-21):
  │       # Push M_soft towards target k for all active (i,j) pairs
  │       k_idx = relation_index(k)
  │       loss += -sum(log(clamp(M_soft[i,j,k_idx], 1e-8))
  │                    for i,j where M_hard[i,j,:].sum() > 0) / N^2
  │
  └─ loss.backward()
       → dL/dM_soft → dL/dW (via Gumbel-Softmax chain rule)
```

### RD-21: Missed-Positive Penalty Must Be Differentiable (v4 — was Medium #10)

RFC v3 used `loss += 10.0` for missed positives, which is constant w.r.t. W
and produces no gradient. The v4 fix uses a differentiable penalty that
pushes M_soft towards the target relation k for all active (i,j) pairs.
This creates gradient signal that encourages the mask to route some (i,j)
pair towards the correct head relation.

### RD-22: Device Row Count Is Private (v4.1 — v4 finding #1)

`CudaKernelProvider::device_row_count()` is `fn` (not `pub fn`) at
provider/mod.rs (private method). New ILP code cannot call it directly. The fix is a
standalone helper that inlines the same pattern using only public APIs:
`provider.device()`, `buffer.num_rows_device()`, and cudarc's
`dtoh_sync_copy_into`. All existing public methods
like `download_column::<T>()` call `device_row_count` internally, so they still
work; this helper is only needed where we need the count without downloading
column data.

### RD-23: `try_slice` Returns `Option`, Not `Result` (v4.1 — v4 finding #2)

In cudarc 0.19, `TrackedCudaSlice::try_slice(range)` returns
`Option<CudaView<T>>`, not `Result`. All `.map_err()` calls on `try_slice`
must use `.ok_or_else()` instead.

### RD-24: Per-Fact Credit Must Be Per-Join, Not Per-Relation (v4.1 — v4 finding #4)

RFC v4's `tagged_entries_containing_fact` checked if a fact exists in the
final relation, then credited ALL (i,j,k) entries targeting k — collapsing
to relation-level granularity. The v4.1 fix re-executes each candidate
join (`hash_join_v2` on rel_i ⋈ rel_j`) and checks per-fact membership
in each individual join result. `hash_join_v2` takes `&self` on provider
(line 7147), so this is safe to call from `CompiledIlpProgram`. The
re-execution cost is negligible for ILP's typical scale (few active rules,
small relations).

### RD-25: `fact_exists` Must Be Schema-Aware (v4.1 — v4 finding #5)

RFC v4 hardcoded `download_column_i64` for all columns. Non-i64 relations
(e.g., u32 symbol IDs) would fail. The fix dispatches on
`buf.schema().column_type(col_idx)` (types.rs:130) to the appropriate
`download_column_*` variant, then widens each value to i64 for uniform
comparison. This works losslessly for U32, I32, Symbol, Bool. U64 truncate-casts
to i64 (safe for typical Datalog symbol IDs). f32/f64 columns are
rejected with an error (ILP fact comparison is integer-only).

### RD-26: Base Facts Must Be Explicitly Loaded (v4.2 — v4.1 finding #4)

`execute_plan()` does NOT load ground facts into the store. Facts are
lowered to `Scan` nodes that read from the store (lower.rs:331,
executor.rs:1490). The working path in `logic.rs:137` explicitly calls
`load_facts()` between schema seeding and plan execution. The RFC must
replicate this: `load_facts_into_store()` iterates `ast.facts()`,
serializes terms to column bytes via `push_term_bytes`, uploads via
`create_buffer_from_slices`, and merges into the pre-seeded store via
`provider.union()`. This applies to `compile()`, `evaluate()`, and
`commit_induced_rule()`.

### RD-27: Optimizer Schemas Keyed by RelId (v4.2 — v4.1 finding #1)

`Optimizer.schemas` is `HashMap<RelId, Schema>` (optimizer.rs:167), not
`HashMap<String, Schema>`. The `TensorMaskedJoin` node carries both
`head_rel_name: String` (for executor store lookups) and
`head_rel_id: RelId` (for optimizer schema lookups).

### RD-28: `extract_tmj_keys` Must Match Actual IR Shapes (v4.2 — v4.1 finding #3)

`ExecutionPlan` has no `.nodes()` method. Access rules via
`plan.rules_by_scc: Vec<Vec<CompiledRule>>` where each `CompiledRule.body`
is a `RirNode`. `RirNode::Fixpoint` has `base`/`recursive` fields (not
`body`). `RirNode::Union` has `inputs: Vec<RirNode>` (not `left`/`right`).

### RD-29: `Program` Type Collision in pyxlog (v4.3 — v4.2 finding #1)

`Program` is already the `#[pyclass]` PyO3 struct at pyxlog/lib.rs:433.
The AST type `xlog_logic::ast::Program` is NOT imported in pyxlog (lib.rs:18
lists `Atom, BodyLiteral, ProbEngine, Rule, Term` but not `Program`).
Fix: `use xlog_logic::ast::Program as AstProgram;` in pyxlog. All ILP
code uses `AstProgram` for AST references.

### RD-30: Lowering Must Use `get_or_create_rel_id` (v4.3 — v4.2 findings #2+#3)

`lower.rs` uses `xlog_core::Result`/`XlogError` (lower.rs:15), not `anyhow`.
Additionally, `rel_ids` are allocated lazily from body scans/facts
(lower.rs:97,401), so a head-only predicate may not yet have a RelId.
`get_or_create_rel_id(&name)` (lower.rs:97) allocates if missing,
returning a valid `RelId` without errors.

### RD-31: `push_term_bytes` Duplication (v4.3 — v4.2 finding #4)

`push_term_bytes` is private in `xlog-gpu/logic.rs:282`. It is already
duplicated in `xlog-logic/examples/xlog_run.rs:99` (~40 lines). The ILP
implementation duplicates it in `pyxlog/src/lib.rs`. The function depends
only on `xlog_core::ScalarType` and `xlog_logic::ast::Term` (no
xlog-gpu-specific deps), so the duplication is safe and self-contained.

### RD-32: Learnable Rules Must Be Wired into `lower_program` (v4.4 — v4.3 finding #1)

`lower_program` (lower.rs:321–368) iterates `program.proper_rules()` and
`program.facts()` only. Learnable rules live in `program.learnable_rules`
and require an explicit loop calling `lower_learnable_rule`, placed after
proper-rule lowering so body-relation RelIds are already allocated.

### RD-33: Parser Must Use `build_head`, Not `build_atom` (v4.4 — v4.3 finding #2)

The grammar production `learnable_rule` uses `head` (grammar.pest:81),
which expects `head_term_list`. `build_atom` (parser.rs:727) expects
`term_list` and will fail on `head` pairs. `build_head` (parser.rs:652)
is the correct parser function.

### RD-34: Learnable Body Shape Validation (v4.4 — v4.3 finding #3)

The grammar allows negation, comparisons, is-exprs, and arbitrary body
arity (grammar.pest:75). Learnable templates require exactly two positive
atoms. `lower_learnable_rule` must validate this before indexing
`rule.body[0]`/`rule.body[1]`, returning `XlogError::Compilation` on violation.

### RD-35: Executor Accessor Methods for ILP State (v4.4 — v4.3 finding #4)

`ilp_registry_mut()` and `ilp_last_result()` are called by
`CompiledIlpProgram` in pyxlog but were never defined on `Executor`.
Added as trivial accessor methods in the executor changes section.

### RD-36: Deterministic `rel_index` Ordering (v4.4 — v4.3 finding #5)

`rel_index` is built from `HashMap::iter()` which has nondeterministic
ordering. Must sort by `RelId` (`.sort_by_key(|(id, _)| id.0)`) so tensor
dimension → relation mapping is stable across recompiles. Applied in both
`lower_learnable_rule` and the pyxlog `compile()` path.

### RD-37: Fail Hard on Missing Head Schema (v4.4 — v4.3 finding #6)

The `Schema::new(vec![])` fallback silently produces a 0-column buffer
that corrupts the store when `execute_non_recursive_scc` writes it back.
Both fallback paths in `execute_tensor_masked_join` (no-mask early-return
and end-of-function return) now return `XlogError::Execution` if the head
relation is not found in the store.

### Crates Involved

```
xlog-logic   : LearnableRule AST + lowering to TensorMaskedJoin
               + parser.rs dispatch + stratify.rs dependency graph
               + optimizer.rs pass-through arms (3 functions)
xlog-ir      : TensorMaskedJoin RirNode variant + collect_relations
xlog-cuda    : extract_nonzero_indices kernel, kernel manifest update,
               ILP_MODULE const + load_ptx block
xlog-runtime : IlpRegistry, execute_tensor_masked_join(), tag metadata
               + arms in contains_non_monotonic_ops, collect_scan_rels,
                 rewrite_scan_nth_impl
pyxlog       : CompiledIlpProgram class, host-side membership check
```

xlog-prob is **not modified**.

---

## 4. Implementation Specification

### Phase 1: Syntax & IR Representation

**Crates:** `xlog-logic`, `xlog-ir`

#### 4.1 Grammar & AST

**File: `crates/xlog-logic/src/grammar.pest`**

Add a `learnable_rule` production alongside the existing `rule_def`. Use the
existing symbols `head` and `body` (not `rule_head`/`rule_body`):

```pest
learnable_rule = {
    "learnable" ~ "(" ~ ident ~ ")" ~ "::" ~ head ~ ":-" ~ body ~ "."
}
```

Integrate into the existing statement dispatch (alongside `rule_def`,
`prob_fact`, `neural_pred_decl`, etc.).

**File: `crates/xlog-logic/src/ast.rs`**

Add to the `Program` struct:

```rust
pub struct Program {
    // ... existing fields ...
    pub learnable_rules: Vec<LearnableRule>,
}

pub struct LearnableRule {
    pub mask_name: String,
    pub head: Atom,
    pub body: Vec<BodyLiteral>,
}
```

**File: `crates/xlog-logic/src/parser.rs`**

Add match arm alongside `Rule::rule_def`, `Rule::fact`, etc.:

```rust
Rule::learnable_rule => {
    program.learnable_rules.push(build_learnable_rule(inner)?);
}
```

```rust
fn build_learnable_rule(pair: Pair<Rule>) -> Result<LearnableRule> {
    let mut inner = pair.into_inner();
    let mask_name = inner.next()
        .ok_or_else(|| XlogError::Parse("Missing learnable mask name".into()))?
        .as_str().to_string();
    // RD-33: Grammar produces `head` (grammar.pest:81), which uses
    // head_term_list (supporting aggregates). build_head (parser.rs:652)
    // is the correct parser; build_atom expects term_list and will fail.
    let head = build_head(inner.next()
        .ok_or_else(|| XlogError::Parse("Missing learnable head".into()))?)?;
    let body = build_body(inner.next()
        .ok_or_else(|| XlogError::Parse("Missing learnable body".into()))?)?;
    Ok(LearnableRule { mask_name, head, body })
}
```

**File: `crates/xlog-logic/src/stratify.rs`**

Add learnable rule edges in `build_dependency_graph`. The current code
uses `add_edge(from, to, DepType)` (not `add_dependency`), and body
literals are accessed via `BodyLiteral::Positive(atom)` / `Negated(atom)`
pattern matching (the `.atom()` method returns `Option<&Atom>`):

```rust
pub fn build_dependency_graph(program: &Program) -> DependencyGraph {
    let mut graph = DependencyGraph::new();

    for rule in &program.rules {
        // ... existing logic ...
    }

    // Learnable rules: head depends on body predicates.
    // At runtime, TensorMaskedJoin dynamically selects which relations
    // to join, but for stratification we conservatively register the
    // template's body predicates as positive dependencies.
    for lr in &program.learnable_rules {
        let head = &lr.head.predicate;
        graph.add_predicate(head.clone());
        for body_lit in &lr.body {
            if let Some(atom) = body_lit.atom() {
                graph.add_edge(
                    head.clone(),
                    atom.predicate.clone(),
                    DepType::Positive,
                );
            }
        }
    }

    graph
}
```

**File: `crates/xlog-logic/src/optimizer.rs`**

Add `TensorMaskedJoin` arms in **three** exhaustive match functions:

```rust
// In predicate_pushdown (line ~258):
RirNode::TensorMaskedJoin { .. } => node, // Leaf-like, no pushdown

// In estimate_width (line ~569):
// Optimizer schemas are HashMap<RelId, Schema> (optimizer.rs:167).
// Use head_rel_id (not head_rel_name) for lookup.
RirNode::TensorMaskedJoin { head_rel_id, .. } => {
    self.schemas.get(head_rel_id)
        .map(|s| s.arity())
        .unwrap_or(2)
}

// In estimate_cost (line ~768):
RirNode::TensorMaskedJoin { max_active_rules, .. } => PlanCost {
    rows: *max_active_rules as u64,
    cpu_cost: *max_active_rules as f64 * 100.0, // hash-join per rule
    gpu_mem: *max_active_rules as u64 * 1024,   // join buffers
    transfers: 1,                                // mask D2H for indices
},
```

Note: `find_column_relation` already has `_ => None` catch-all.

#### 4.2 Relational IR

**File: `crates/xlog-ir/src/rir.rs`**

Add a new `RirNode` variant:

```rust
pub enum RirNode {
    // ... existing 10 variants ...

    TensorMaskedJoin {
        mask_name: String,
        schema_size: usize,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        /// Mapping from tensor dimension index -> (RelId, relation name)
        rel_index: Vec<(RelId, String)>,
        /// Head relation name (for store lookup in executor, RD-12)
        head_rel_name: String,
        /// Head relation ID (for optimizer schema lookup, keyed by RelId)
        head_rel_id: RelId,
        max_active_rules: usize,
    },
}
```

Update `collect_relations` (note: uses `Vec<RelId>`, not `HashSet`).
**Must be committed atomically with the variant addition above** — the
existing match at rir.rs:193 is exhaustive with no `_` wildcard:

```rust
fn collect_relations(&self, rels: &mut Vec<RelId>) {
    match self {
        // ... existing arms ...

        RirNode::TensorMaskedJoin { rel_index, .. } => {
            for (rel_id, _) in rel_index {
                rels.push(*rel_id);
            }
        }
    }
}
```

#### 4.3 Lowering

**File: `crates/xlog-logic/src/lower.rs`**

```rust
fn lower_learnable_rule(&mut self, rule: &LearnableRule) -> Result<RirNode> {
    // RD-34: Validate body shape before indexing.
    // Grammar allows negation, comparisons, is-exprs, and arbitrary arity
    // (grammar.pest:75). Learnable templates require exactly two positive atoms.
    if rule.body.len() != 2 {
        return Err(XlogError::Compilation(format!(
            "learnable rule '{}' requires exactly 2 body literals, got {}",
            rule.mask_name, rule.body.len()
        )));
    }
    for (idx, lit) in rule.body.iter().enumerate() {
        match lit {
            BodyLiteral::Positive(_) => {}
            _ => return Err(XlogError::Compilation(format!(
                "learnable rule '{}' body[{}]: only positive atoms allowed, got {:?}",
                rule.mask_name, idx, lit
            ))),
        }
    }

    // RD-36: Sort by RelId to keep tensor dimension → relation mapping
    // deterministic across recompiles. HashMap::iter() is nondeterministic.
    let mut rel_index: Vec<(RelId, String)> = self.rel_ids().iter()
        .map(|(name, id)| (*id, name.clone()))
        .collect();
    rel_index.sort_by_key(|(id, _)| id.0);
    let schema_size = rel_index.len();
    let (left_keys, right_keys) = self.extract_template_join_keys(
        &rule.body[0], &rule.body[1]
    )?;
    let head_rel_name = rule.head.predicate.clone();
    // RD-30: Use get_or_create_rel_id (lower.rs:97), NOT rel_ids().get().
    // Head-only predicates may not yet have a RelId allocated
    // (rel_ids are populated lazily from body scans/facts at lower.rs:401).
    let head_rel_id = self.get_or_create_rel_id(&head_rel_name);
    Ok(RirNode::TensorMaskedJoin {
        mask_name: rule.mask_name.clone(),
        schema_size, left_keys, right_keys, rel_index,
        head_rel_name, head_rel_id, max_active_rules: 32,
    })
}
```

**Wiring into `lower_program`** (RD-32):

The existing `lower_program` (lower.rs:321–368) iterates only
`program.proper_rules()` (lower.rs:323) and `program.facts()` (lower.rs:331).
Learnable rules live in `program.learnable_rules` and must be lowered
separately, after proper rules (so all body-relation RelIds are already
allocated). Add this block before `Ok(builder.build())` at lower.rs:367:

```rust
// Lower learnable rules (RD-32)
// Placed after proper rules so body-relation RelIds are already allocated.
// Pre-allocate RelIds for ALL learnable rule predicates (heads AND bodies)
// so that every lower_learnable_rule snapshot (rel_index) is complete.
// Body predicates may appear only in learnable rules (e.g. runtime-declared
// input relations with no facts/proper rules), so they won't have been
// allocated by prior lowering passes.
for learnable in &program.learnable_rules {
    self.get_or_create_rel_id(&learnable.head.predicate);
    for lit in &learnable.body {
        if let BodyLiteral::Positive(atom) = lit {
            self.get_or_create_rel_id(&atom.predicate);
        }
    }
}
// Each learnable rule becomes a TensorMaskedJoin node in the head predicate's SCC.
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

---

### Phase 2: ILP Registry & DLPack Bridge

**Crates:** `xlog-runtime`, `xlog-cuda`

#### 4.4 ILP Registry

**New file: `crates/xlog-runtime/src/ilp_registry.rs`**

```rust
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use std::collections::HashMap;

/// Helper: read device-side row count using only public APIs (RD-22).
/// `device_row_count` is private on CudaKernelProvider (provider/mod.rs).
/// This inlines the same pattern: dtoh_sync_copy_into from num_rows_device().
pub fn read_device_row_count(
    provider: &CudaKernelProvider,
    buffer: &CudaBuffer,
) -> Result<usize> {
    let mut host_rows = [0u32];
    provider.device().inner()
        .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
        .map_err(|e| XlogError::Kernel(format!("Failed to read row count: {}", e)))?;
    Ok(host_rows[0] as usize)
}

pub struct IlpRegistry {
    masks: HashMap<String, IlpMask>,
}

pub struct IlpMask {
    /// Flat 1D CudaBuffer (N*N*N f32 elements), imported via DLPack.
    /// Python flattens 3D tensor to 1D before export (RD-16).
    pub hard: CudaBuffer,
    pub soft: CudaBuffer,
    pub schema_size: usize,
}

/// Tag metadata from TensorMaskedJoin execution.
/// (v5) IlpTagEntry retains buffer for batch credit queries (RD-17 relaxed).
pub struct IlpTaggedResult {
    pub entries: Vec<IlpTagEntry>,
}

pub struct IlpTagEntry {
    pub i: u32,
    pub j: u32,
    pub k: u32,
    pub num_rows: u32,
    /// The projected join result buffer, retained for batch credit queries.
    pub buffer: Option<CudaBuffer>,
}

impl IlpRegistry {
    pub fn new() -> Self { Self { masks: HashMap::new() } }

    pub fn insert_mask(&mut self, name: String, hard: CudaBuffer,
                       soft: CudaBuffer, schema_size: usize) {
        self.masks.insert(name, IlpMask { hard, soft, schema_size });
    }

    pub fn get_mask(&self, name: &str) -> Option<&IlpMask> {
        self.masks.get(name)
    }
}
```

**File: `crates/xlog-runtime/src/executor.rs`**

Add fields on `Executor` and arms in exhaustive matches:

```rust
pub struct Executor {
    // ... existing fields ...
    ilp_registry: IlpRegistry,
    ilp_last_result: Option<IlpTaggedResult>,
}

// RD-35: Accessor methods for pyxlog to reach ILP state.
// Called by CompiledIlpProgram.set_rule_mask() and .get_tagged_results().
// Note: store() and store_mut() are already pub on Executor (executor.rs:231,236).
impl Executor {
    pub fn ilp_registry_mut(&mut self) -> &mut IlpRegistry {
        &mut self.ilp_registry
    }

    pub fn ilp_last_result(&self) -> Option<&IlpTaggedResult> {
        self.ilp_last_result.as_ref()
    }
}
```

Additional match arms required (RD-20):

```rust
// In contains_non_monotonic_ops (line ~463):
RirNode::TensorMaskedJoin { .. } => false,

// In collect_scan_rels (line ~1245):
RirNode::TensorMaskedJoin { rel_index, .. } => {
    for (rel_id, _) in rel_index {
        out.push(*rel_id);
    }
}

// In rewrite_scan_nth_impl (line ~1291):
RirNode::TensorMaskedJoin { .. } => (node.clone(), false),

// In execute_node (line ~620): dispatch with all fields
// (head_rel_id only needed by optimizer, not executor)
RirNode::TensorMaskedJoin {
    mask_name, schema_size, left_keys, right_keys,
    rel_index, head_rel_name, max_active_rules, ..
} => self.execute_tensor_masked_join(
    mask_name, *schema_size, left_keys, right_keys,
    rel_index, head_rel_name, *max_active_rules,
),
```

#### 4.5 CUDA Kernel

**New file: `kernels/ilp.cu`**

```c
#include <stdint.h>

extern "C" __global__ void extract_nonzero_indices(
    const float* mask_hard,     // Flat N*N*N f32 (RD-16: 1D)
    const float* mask_soft,     // Flat N*N*N f32 (for priority)
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

#### 4.6 Build System Changes

**File: `crates/xlog-cuda/src/kernel_manifest_data.rs`**

```rust
pub const KERNEL_CU_NAMES: &[&str] = &[
    "join", "dedup", "groupby", "scan", "sort", "filter", "set_ops",
    "pack", "pir", "cnf", "cache", "weights", "circuit", "mc_sample",
    "mc_eval", "arith", "sat", "d4", "neural",
    "ilp",  // <-- new
];
```

**File: `crates/xlog-cuda/src/provider/mod.rs`** (post-Wave-2: kernel constants remain in mod.rs)

```rust
// Update assertion
const _: () = assert!(crate::kernel_manifest_data::KERNEL_CU_NAMES.len() == 20);

// Add module constant and kernel names
pub const ILP_MODULE: &str = "xlog_ilp";

pub mod ilp_kernels {
    pub const EXTRACT_NONZERO_INDICES: &str = "extract_nonzero_indices";
}
```

Add load_ptx block (follows exact pattern of all other module blocks):

```rust
// Load ILP module (mask non-zero extraction)
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
        .map_err(|e| XlogError::Kernel(format!("Failed to load ILP module: {}", e)))?;
    if let Some(t0) = t0 {
        if profiling {
            device.inner().synchronize()
                .map_err(|e| XlogError::Kernel(format!("sync after ILP load: {}", e)))?;
        }
        let elapsed = t0.elapsed().as_secs_f64();
        profile.per_module_sec.push(("ilp".to_string(), elapsed));
        profile.total_sec += elapsed;
        if is_cubin { profile.cubin_loaded += 1; } else { profile.ptx_fallback += 1; }
    }
}
```

#### 4.7 Provider Wrapper

**File: `crates/xlog-cuda/src/provider/ilp.rs`** (post-Wave-2: ILP methods in ilp.rs submodule)

Uses actual APIs only (RD-19): `memory.alloc()`, `htod_sync_copy_into`,
`dtoh_sync_copy_into`, `device.inner().get_func()`.

```rust
impl CudaKernelProvider {
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

        // Allocate output buffers via GpuMemoryManager
        let mut out_i = self.memory().alloc::<u32>(total)?;
        let mut out_j = self.memory().alloc::<u32>(total)?;
        let mut out_k = self.memory().alloc::<u32>(total)?;
        let mut out_p = self.memory().alloc::<f32>(total)?;
        let mut count = self.memory().alloc::<u32>(1)?;

        // Zero-init the counter
        self.device().inner()
            .htod_sync_copy_into(&[0u32], &mut count)
            .map_err(|e| XlogError::Kernel(format!("ILP htod count: {}", e)))?;

        // Get raw device pointers for mask columns.
        // mask_hard and mask_soft are single-column f32 CudaBuffers (RD-16).
        let hard_col = mask_hard.column(0)
            .ok_or_else(|| XlogError::Kernel("ILP hard mask has no column".into()))?;
        let soft_col = mask_soft.column(0)
            .ok_or_else(|| XlogError::Kernel("ILP soft mask has no column".into()))?;

        let device = self.device();
        let kernel = device.inner()
            .get_func(ILP_MODULE, ilp_kernels::EXTRACT_NONZERO_INDICES)
            .ok_or_else(|| XlogError::Kernel(
                "extract_nonzero_indices kernel not found".into()
            ))?;

        // Get byte-level views for passing to kernel.
        // CudaColumn stores raw bytes; the kernel interprets them as f32.
        let hard_num_bytes = total * std::mem::size_of::<f32>();
        let soft_num_bytes = total * std::mem::size_of::<f32>();
        let hard_view = self.column_bytes_view(hard_col, hard_num_bytes)?;
        let soft_view = self.column_bytes_view(soft_col, soft_num_bytes)?;

        unsafe {
            kernel.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size as u32, 1, 1),
                    block_dim: (block_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&hard_view, &soft_view, n as u32,
                 &mut out_i, &mut out_j, &mut out_k, &mut out_p, &mut count),
            ).map_err(|e| XlogError::Kernel(
                format!("Failed to launch extract_nonzero_indices: {}", e)
            ))?;
        }

        // Download counter to host
        let mut count_host = [0u32];
        device.inner()
            .dtoh_sync_copy_into(&count, &mut count_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh count: {}", e)))?;
        let active_count = count_host[0] as usize;

        if active_count == 0 {
            return Ok(Vec::new());
        }

        // Download active indices to host
        let mut i_host = vec![0u32; active_count];
        let mut j_host = vec![0u32; active_count];
        let mut k_host = vec![0u32; active_count];
        let mut p_host = vec![0f32; active_count];

        // Slice output buffers to active_count elements before download.
        // RD-23: try_slice returns Option<CudaView<T>> in cudarc 0.19,
        // NOT Result. Use .ok_or_else() instead of .map_err().
        let out_i_slice = out_i.try_slice(0..active_count)
            .ok_or_else(|| XlogError::Kernel("ILP slice i out of bounds".into()))?;
        let out_j_slice = out_j.try_slice(0..active_count)
            .ok_or_else(|| XlogError::Kernel("ILP slice j out of bounds".into()))?;
        let out_k_slice = out_k.try_slice(0..active_count)
            .ok_or_else(|| XlogError::Kernel("ILP slice k out of bounds".into()))?;
        let out_p_slice = out_p.try_slice(0..active_count)
            .ok_or_else(|| XlogError::Kernel("ILP slice p out of bounds".into()))?;

        device.inner().dtoh_sync_copy_into(&out_i_slice, &mut i_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh i: {}", e)))?;
        device.inner().dtoh_sync_copy_into(&out_j_slice, &mut j_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh j: {}", e)))?;
        device.inner().dtoh_sync_copy_into(&out_k_slice, &mut k_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh k: {}", e)))?;
        device.inner().dtoh_sync_copy_into(&out_p_slice, &mut p_host)
            .map_err(|e| XlogError::Kernel(format!("ILP dtoh p: {}", e)))?;

        // Sort by priority descending, truncate
        let mut indices: Vec<(f32, u32, u32, u32)> = (0..active_count)
            .map(|idx| (p_host[idx], i_host[idx], j_host[idx], k_host[idx]))
            .collect();
        indices.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        indices.truncate(max_active);

        Ok(indices.into_iter().map(|(_, i, j, k)| (i, j, k)).collect())
    }
}
```

**Note on `try_slice` / dtoh pattern:** `TrackedCudaSlice<T>` wraps
`cudarc::driver::CudaSlice<T>` which provides `.try_slice()` returning
`Option<CudaView<T>>` (cudarc 0.19, core.rs:1456). The
`device.inner().dtoh_sync_copy_into()` method is the standard download
path used throughout the codebase (e.g., `provider/transfer.rs`, `provider/io.rs`).

---

### Phase 3: Executor Integration

**Crate:** `xlog-runtime`

#### 4.8 TensorMaskedJoin Execution

**File: `crates/xlog-runtime/src/executor.rs`**

```rust
fn execute_tensor_masked_join(
    &mut self,
    mask_name: &str,
    schema_size: usize,
    left_keys: &[usize],
    right_keys: &[usize],
    rel_index: &[(RelId, String)],
    head_rel_name: &str,
    max_active_rules: usize,
) -> Result<CudaBuffer> {
    // RD-12: No-op when no mask registered. Return empty buffer with
    // the head relation's schema (not Schema::new(vec![])) to prevent
    // schema corruption when execute_non_recursive_scc stores the result.
    // The head schema comes from the pre-seeded store via head_rel_name,
    // NOT from rel_index.first() which may not match the head (RD-12 v4.1).
    let ilp_mask = match self.ilp_registry.get_mask(mask_name) {
        Some(mask) => mask,
        None => {
            self.ilp_last_result = Some(IlpTaggedResult { entries: Vec::new() });
            // RD-37: Fail hard if head relation missing from store.
            // Schema::new(vec![]) would silently produce a 0-column buffer that
            // corrupts the store when execute_non_recursive_scc writes it back.
            // The store must have been seeded in compile() / evaluate().
            let schema = self.store.get(head_rel_name)
                .map(|buf| buf.schema().clone())
                .ok_or_else(|| XlogError::Execution(format!(
                    "TensorMaskedJoin: head relation '{}' not found in store \
                     (was load_facts_into_store called?)", head_rel_name
                )))?;
            return self.provider.create_empty_buffer(schema);
        }
    };

    let start = self.profiler.start_op();

    // 2. Extract active (i,j,k) indices via GPU kernel
    let active_rules = self.provider.extract_active_rule_indices(
        &ilp_mask.hard, &ilp_mask.soft,
        schema_size, max_active_rules,
    )?;

    // 3. Dispatch hash joins, collect tag metadata
    let mut tag_entries: Vec<IlpTagEntry> = Vec::new();
    let mut results_by_k: HashMap<u32, Vec<CudaBuffer>> = HashMap::new();

    for &(i, j, k) in &active_rules {
        let (_, left_name) = &rel_index[i as usize];
        let (_, right_name) = &rel_index[j as usize];

        let left_buf = match self.store.get(left_name) {
            Some(buf) => buf,
            None => continue,
        };
        let right_buf = match self.store.get(right_name) {
            Some(buf) => buf,
            None => continue,
        };

        let joined = self.provider.hash_join_v2(
            left_buf, right_buf,
            left_keys, right_keys,
            JoinType::Inner,
        )?;

        // RD-22: Use public helper instead of private device_row_count
        let num_rows = read_device_row_count(&self.provider, &joined)? as u32;

        if num_rows > 0 {
            // RD-17: Store only metadata, not CudaBuffer clones
            tag_entries.push(IlpTagEntry { i, j, k, num_rows });
            results_by_k.entry(k).or_default().push(joined);
        }
    }

    // 4. Union results per target relation and store
    for (k, buffers) in results_by_k {
        let (_, target_name) = &rel_index[k as usize];

        // Chain-union all buffers
        let union_buf = if buffers.len() == 1 {
            buffers.into_iter().next().unwrap()
        } else {
            let mut acc = self.provider.union_gpu(&buffers[0], &buffers[1])?;
            for buf in &buffers[2..] {
                acc = self.provider.union_gpu(&acc, buf)?;
            }
            acc
        };

        // Diff against existing and merge
        if let Some(existing) = self.store.get(target_name) {
            let delta = self.provider.diff_gpu(&union_buf, existing)?;
            if !delta.is_empty() {
                let merged = self.provider.union_gpu(existing, &delta)?;
                self.store_put(target_name, merged);
            }
        } else {
            let key_cols: Vec<usize> = (0..union_buf.arity()).collect();
            let deduped = self.provider.dedup(&union_buf, &key_cols)?;
            self.store_put(target_name, deduped);
        }
    }

    // 5. Store tag metadata
    self.ilp_last_result = Some(IlpTaggedResult { entries: tag_entries });

    if let Some(start) = start {
        let mem = self.provider.memory().allocated_bytes();
        self.profiler.record_op(
            "TensorMaskedJoin", 0, active_rules.len() as u64, start, mem,
        );
    }

    // Return empty with head schema (results routed via store).
    // RD-37: Fail hard — same rationale as no-mask path above.
    let schema = self.store.get(head_rel_name)
        .map(|buf| buf.schema().clone())
        .ok_or_else(|| XlogError::Execution(format!(
            "TensorMaskedJoin: head relation '{}' not found in store \
             (was load_facts_into_store called?)", head_rel_name
        )))?;
    self.provider.create_empty_buffer(schema)
}
```

---

### Phase 4: Python API & Orchestration

**Crate:** `pyxlog`

#### 4.9 New CompiledIlpProgram Class

**File: `crates/pyxlog/src/lib.rs`**

```rust
// RD-29: `Program` is already the PyO3 class (lib.rs:433).
// Alias the AST type to avoid collision. The existing import line
// (lib.rs:18) does NOT import Program; add it with an alias:
use xlog_logic::ast::Program as AstProgram;

#[pyclass]
pub struct IlpProgramFactory;

#[pymethods]
impl IlpProgramFactory {
    #[staticmethod]
    pub fn compile(
        source: &str,
        device: usize,
        memory_mb: u64,
    ) -> PyResult<CompiledIlpProgram> {
        let ast = xlog_logic::parse_program(source)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Separate base source from learnable declarations (RD-13)
        let base_source = strip_learnable_declarations(source);
        let learnable_source = extract_learnable_declarations(source);

        let mut compiler = xlog_logic::Compiler::new();
        let plan = compiler.compile_program(&ast)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // RD-36: Sort by RelId for deterministic tensor dimension mapping.
        let mut rel_index: Vec<(RelId, String)> = compiler.rel_ids().iter()
            .map(|(name, id)| (*id, name.clone()))
            .collect();
        rel_index.sort_by_key(|(id, _)| id.0);
        let schemas = compiler.schemas().clone();

        // Create provider (RD-14: correct helper name)
        let config = GpuConfig {
            device_ordinal: device,
            memory_bytes: memory_mb * 1024 * 1024,
        };
        let provider = Arc::new(
            provider_from_config(config)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        );

        // RD-15: Full relation setup before execution
        // (mirrors xlog-gpu/logic.rs:112-137)
        let mut executor = Executor::new(provider.clone());

        // Step 1: Register all relations with their RelIds
        for (name, rel_id) in compiler.rel_ids() {
            executor.register_relation(*rel_id, name);
        }

        // Step 2: Pre-seed all relations with empty buffers (correct schema)
        for (name, schema) in &schemas {
            let empty = provider.create_empty_buffer(schema.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            executor.store_mut().put(name, empty);
        }

        // Step 3: Load base facts into store (RD-26).
        // execute_plan does NOT load facts — facts are lowered to Scan
        // which reads from the store, so they must be loaded first.
        // Mirrors xlog-gpu/logic.rs:137 (load_facts before execute_plan).
        load_facts_into_store(&ast, &provider, &mut executor, &schemas)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Step 4: Execute plan (TensorMaskedJoin no-ops gracefully
        // per RD-12 since no mask is registered yet)
        executor.execute_plan(&plan)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        // Extract join keys from the plan's TensorMaskedJoin node (RD-24)
        let (left_keys, right_keys) = extract_tmj_keys(&plan);

        Ok(CompiledIlpProgram {
            base_source, learnable_source, ast, executor, provider,
            plan, rel_index, schemas, left_keys, right_keys,
        })
    }
}

#[pyclass]
pub struct CompiledIlpProgram {
    base_source: String,
    learnable_source: String,
    /// Retained for fact loading — AstProgram.facts() yields ground facts
    /// that must be explicitly loaded into the store before execution.
    /// (execute_plan does NOT load facts; see logic.rs:137 pattern.)
    ast: AstProgram,
    executor: Executor,
    provider: Arc<CudaKernelProvider>,
    plan: ExecutionPlan,
    rel_index: Vec<(RelId, String)>,
    /// String-keyed schemas for fact loading and store seeding.
    /// NOT the optimizer's schemas (which are HashMap<RelId, Schema>,
    /// see optimizer.rs:167). The optimizer receives its RelId-keyed
    /// schemas via Compiler and operates on the plan before execution.
    /// This field is only used in load_facts_into_store and create_empty_buffer.
    schemas: HashMap<String, Schema>,
    /// Cached from TensorMaskedJoin node for per-fact credit (RD-24)
    left_keys: Vec<usize>,
    right_keys: Vec<usize>,
}

#[pymethods]
impl CompiledIlpProgram {
    /// Register mask tensors. Python MUST flatten to 1D first (RD-16).
    /// Usage: program.set_rule_mask("W_mask",
    ///            M_hard.contiguous().view(-1),
    ///            M_soft.contiguous().view(-1), N)
    pub fn set_rule_mask(
        &mut self,
        name: String,
        mask_hard_flat: &Bound<'_, PyAny>,
        mask_soft_flat: &Bound<'_, PyAny>,
        schema_size: usize,
    ) -> PyResult<()> {
        // dlpack_from_py returns DlpackManagedTensor (ndim must be 1)
        let hard_dmt = dlpack_from_py(mask_hard_flat)?;
        let soft_dmt = dlpack_from_py(mask_soft_flat)?;

        // from_dlpack_tensors: Vec<DlpackManagedTensor> -> CudaBuffer
        let hard_buf = self.provider.from_dlpack_tensors(vec![hard_dmt])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let soft_buf = self.provider.from_dlpack_tensors(vec![soft_dmt])
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        self.executor.ilp_registry_mut().insert_mask(
            name, hard_buf, soft_buf, schema_size,
        );
        Ok(())
    }

    pub fn evaluate(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            // Reset derived facts, re-seed schemas, reload facts, re-execute
            self.executor.reset_for_mc();
            // Re-seed empty buffers (reset_for_mc clears store)
            for (name, schema) in &self.schemas {
                let empty = self.provider.create_empty_buffer(schema.clone())?;
                self.executor.store_mut().put(name, empty);
            }
            // RD-26: Reload base facts before execution
            load_facts_into_store(
                &self.ast, &self.provider, &mut self.executor, &self.schemas,
            )?;
            self.executor.execute_plan(&self.plan)
        }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Get tag metadata from last TensorMaskedJoin: [(i, j, k, n_rows)]
    pub fn get_tagged_results(&self) -> PyResult<Vec<(u32, u32, u32, u32)>> {
        match self.executor.ilp_last_result() {
            Some(result) => Ok(result.entries.iter()
                .map(|e| (e.i, e.j, e.k, e.num_rows))
                .collect()),
            None => Ok(Vec::new()),
        }
    }

    /// Host-side membership check (RD-18: no contains_tuple on provider).
    /// Downloads the relation's columns and checks for the tuple.
    /// RD-25: Schema-aware column download (dispatches by ScalarType).
    /// RD-22: Uses read_device_row_count helper (device_row_count is private).
    pub fn fact_exists(
        &self,
        relation: &str,
        values: Vec<i64>,
    ) -> PyResult<bool> {
        let buf = self.executor.store().get(relation)
            .ok_or_else(|| PyValueError::new_err(
                format!("Relation '{}' not found", relation)
            ))?;

        Self::fact_exists_in_buffer(&self.provider, buf, &values)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Shared helper: check if a tuple exists in a CudaBuffer.
    /// Uses schema-aware column download (RD-25) and public row count (RD-22).
    fn fact_exists_in_buffer(
        provider: &CudaKernelProvider,
        buf: &CudaBuffer,
        values: &[i64],
    ) -> Result<bool> {
        let num_rows = read_device_row_count(provider, buf)?;
        if num_rows == 0 { return Ok(false); }
        if values.len() != buf.arity() { return Ok(false); }

        // RD-25: Download each column using the correct type, widen to i64.
        let schema = buf.schema();
        let mut columns: Vec<Vec<i64>> = Vec::new();
        for col_idx in 0..buf.arity() {
            let col_type = schema.column_type(col_idx)
                .ok_or_else(|| XlogError::Kernel(
                    format!("Column {} type not found in schema", col_idx)
                ))?;
            let col_i64: Vec<i64> = match col_type {
                ScalarType::I64 => provider.download_column_i64(buf, col_idx)?,
                ScalarType::I32 => provider.download_column_i32(buf, col_idx)?
                    .into_iter().map(|v| v as i64).collect(),
                ScalarType::U32 | ScalarType::Symbol => {
                    provider.download_column_u32(buf, col_idx)?
                        .into_iter().map(|v| v as i64).collect()
                }
                ScalarType::U64 => {
                    // u64 values >i64::MAX are unlikely in Datalog symbol IDs,
                    // but we truncate-cast here. If exact u64 fidelity is needed,
                    // change Python-side values to pass u64 directly.
                    let col_u64 = provider.download_column_u64(buf, col_idx)?;
                    col_u64.into_iter().map(|v| v as i64).collect()
                }
                ScalarType::Bool => {
                    // Bool is 1 byte per row on GPU. Use download_column_bool
                    // (returns Vec<bool>), NOT download_column_u8.
                    provider.download_column_bool(buf, col_idx)?
                        .into_iter().map(|v| if v { 1i64 } else { 0i64 }).collect()
                }
                ScalarType::F32 | ScalarType::F64 => {
                    return Err(XlogError::Kernel(
                        format!("fact_exists does not support float column type {:?}", col_type)
                    ));
                }
            };
            columns.push(col_i64);
        }

        // For typical ILP relations (arity 2, few hundred rows),
        // host-side scan is negligible.
        for row in 0..num_rows {
            let mut matches = true;
            for (col_idx, val) in values.iter().enumerate() {
                if columns[col_idx][row] != *val {
                    matches = false;
                    break;
                }
            }
            if matches { return Ok(true); }
        }
        Ok(false)
    }

    /// For per-fact credit: which (i,j,k) rules produced a given fact?
    /// RD-24: Re-executes each candidate join (rel_i ⋈ rel_j) and checks
    /// per-fact membership in the individual join result. This provides
    /// true per-join granularity, not just per-relation credit.
    /// hash_join_v2 takes &self (provider/relational.rs), safe to call here.
    pub fn tagged_entries_containing_fact(
        &self,
        relation: &str,
        values: Vec<i64>,
    ) -> PyResult<Vec<(u32, u32, u32)>> {
        let k_idx = self.rel_index.iter()
            .position(|(_, name)| name == relation)
            .ok_or_else(|| PyValueError::new_err(
                format!("Relation '{}' not in ILP schema", relation)
            ))? as u32;

        let tagged = match self.executor.ilp_last_result() {
            Some(t) => t,
            None => return Ok(Vec::new()),
        };

        let mut result = Vec::new();
        for entry in &tagged.entries {
            if entry.k != k_idx || entry.num_rows == 0 {
                continue;
            }

            // Re-execute this specific join to check per-fact membership
            let (_, left_name) = &self.rel_index[entry.i as usize];
            let (_, right_name) = &self.rel_index[entry.j as usize];

            let left_buf = match self.executor.store().get(left_name) {
                Some(buf) => buf,
                None => continue,
            };
            let right_buf = match self.executor.store().get(right_name) {
                Some(buf) => buf,
                None => continue,
            };

            // Re-execute the join (cheap for ILP-scale relations)
            let joined = self.provider.hash_join_v2(
                left_buf, right_buf,
                &self.left_keys, &self.right_keys,
                JoinType::Inner,
            ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            // Check if THIS specific join produced the target fact
            let found = Self::fact_exists_in_buffer(
                &self.provider, &joined, &values,
            ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            if found {
                result.push((entry.i, entry.j, entry.k));
            }
        }
        Ok(result)
    }

    pub fn ilp_schema_size(&self) -> usize { self.rel_index.len() }

    pub fn ilp_relation_names(&self) -> Vec<String> {
        self.rel_index.iter().map(|(_, name)| name.clone()).collect()
    }

    pub fn commit_induced_rule(&mut self, rule_source: &str) -> PyResult<()> {
        let new_base = format!("{}\n{}", self.base_source, rule_source);

        let ast = xlog_logic::parse_program(&new_base)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mut compiler = xlog_logic::Compiler::new();
        let plan = compiler.compile_program(&ast)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let schemas = compiler.schemas().clone();

        // Full re-setup (RD-15)
        self.executor.reset_for_mc();
        for (name, rel_id) in compiler.rel_ids() {
            self.executor.register_relation(*rel_id, name);
        }
        for (name, schema) in &schemas {
            let empty = self.provider.create_empty_buffer(schema.clone())
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            self.executor.store_mut().put(name, empty);
        }
        // RD-26: Load base facts before execution
        load_facts_into_store(&ast, &self.provider, &mut self.executor, &schemas)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.executor.execute_plan(&plan)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        self.base_source = new_base;
        self.ast = ast;
        let (lk, rk) = extract_tmj_keys(&plan);
        self.left_keys = lk;
        self.right_keys = rk;
        self.plan = plan;
        self.schemas = schemas;
        Ok(())
    }
}
```

Helper functions:

```rust
/// Load base facts from the parsed AST into the executor's store (RD-26).
/// Mirrors xlog-gpu/logic.rs:171-221 (GpuLogicEvaluator::load_facts).
///
/// Facts are rules with empty bodies (program.facts() filters for is_fact()).
/// Each fact's terms are serialized to column bytes, uploaded to GPU, and
/// merged (union) into the pre-seeded relation buffer.
///
/// NOTE: `push_term_bytes` is currently a private fn in xlog-gpu/logic.rs:282.
/// Resolution: Duplicate it in pyxlog/src/lib.rs (~40 lines). This is the
/// least-invasive option — the function is already duplicated in
/// xlog-logic/examples/xlog_run.rs:99 and has no xlog-gpu-specific deps
/// (it only uses xlog_core::ScalarType and xlog_logic::ast::Term).
/// `create_buffer_from_slices` and `union` are both pub on CudaKernelProvider.
fn load_facts_into_store(
    ast: &AstProgram,
    provider: &CudaKernelProvider,
    executor: &mut Executor,
    schemas: &HashMap<String, Schema>,
) -> Result<()> {
    use std::collections::HashMap as StdMap;

    // Group fact rows by predicate name
    let mut rows_by_pred: StdMap<&str, Vec<&[Term]>> = StdMap::new();
    for fact in ast.facts() {
        rows_by_pred
            .entry(fact.head.predicate.as_str())
            .or_default()
            .push(&fact.head.terms);
    }

    for (pred, rows) in rows_by_pred {
        let schema = schemas.get(pred).ok_or_else(|| {
            XlogError::Execution(format!(
                "Missing schema for fact predicate {}", pred
            ))
        })?;

        if rows.iter().any(|r| r.len() != schema.arity()) {
            return Err(XlogError::Execution(format!(
                "Fact arity mismatch for {} (expected {})", pred, schema.arity()
            )));
        }

        // Serialize terms to column byte vectors
        let mut columns: Vec<Vec<u8>> = vec![Vec::new(); schema.arity()];
        for row in &rows {
            for (col_idx, term) in row.iter().enumerate() {
                let typ = schema.column_type(col_idx).ok_or_else(|| {
                    XlogError::Execution(format!("Missing type for col {}", col_idx))
                })?;
                push_term_bytes(&mut columns[col_idx], term, typ)?;
            }
        }

        let slices: Vec<&[u8]> = columns.iter().map(|c| c.as_slice()).collect();
        let fact_buf = provider.create_buffer_from_slices(&slices, schema.clone())?;

        // Merge into existing store buffer
        let existing = executor.store().get(pred).ok_or_else(|| {
            XlogError::Execution(format!(
                "Missing base relation {} while loading facts", pred
            ))
        })?;
        let merged = provider.union(existing, &fact_buf)?;
        executor.store_mut().put(pred, merged);
    }
    Ok(())
}

/// Walk the execution plan to find the TensorMaskedJoin and extract its keys.
/// Used to cache left_keys/right_keys for per-fact credit re-execution (RD-24).
///
/// ExecutionPlan has: rules_by_scc: Vec<Vec<CompiledRule>> (plan.rs:45)
/// CompiledRule has: body: RirNode (plan.rs:32)
/// RirNode::Fixpoint has: base, recursive (not body) (rir.rs:166-172)
/// RirNode::Union has: inputs: Vec<RirNode> (not left/right) (rir.rs:151)
fn extract_tmj_keys(plan: &ExecutionPlan) -> (Vec<usize>, Vec<usize>) {
    fn walk(node: &RirNode) -> Option<(Vec<usize>, Vec<usize>)> {
        match node {
            RirNode::TensorMaskedJoin { left_keys, right_keys, .. } => {
                Some((left_keys.clone(), right_keys.clone()))
            }
            RirNode::Fixpoint { base, recursive, .. } => {
                walk(base).or_else(|| walk(recursive))
            }
            RirNode::Union { inputs } => {
                inputs.iter().find_map(walk)
            }
            RirNode::Filter { input, .. }
            | RirNode::Project { input, .. }
            | RirNode::Distinct { input, .. }
            | RirNode::GroupBy { input, .. } => walk(input),
            RirNode::Join { left, right, .. }
            | RirNode::Diff { left, right } => {
                walk(left).or_else(|| walk(right))
            }
            _ => None, // Unit, Scan: leaf nodes
        }
    }
    // ExecutionPlan.rules_by_scc: Vec<Vec<CompiledRule>>
    // CompiledRule.body: RirNode
    for scc_rules in &plan.rules_by_scc {
        for rule in scc_rules {
            if let Some(keys) = walk(&rule.body) {
                return keys;
            }
        }
    }
    (vec![], vec![]) // No TensorMaskedJoin found (shouldn't happen for ILP plans)
}

fn strip_learnable_declarations(source: &str) -> String {
    source.lines()
        .filter(|line| !line.trim_start().starts_with("learnable("))
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_learnable_declarations(source: &str) -> String {
    source.lines()
        .filter(|line| line.trim_start().starts_with("learnable("))
        .collect::<Vec<_>>()
        .join("\n")
}
```

#### 4.10 Complete Python Orchestration Example

```python
import pyxlog
import torch
import torch.nn.functional as F

# 1. Compile
program = pyxlog.IlpProgramFactory.compile("""
    edge(1, 2). edge(2, 3). edge(3, 4).
    learnable(W_mask) :: reach(X, Y) :- body1(X, Z), body2(Z, Y).
""", device=0, memory_mb=4096)

N = program.ilp_schema_size()
rel_names = program.ilp_relation_names()

# 2. Training examples
positive_examples = [("reach", [1, 4]), ("reach", [1, 3])]
negative_examples = [("reach", [4, 1]), ("reach", [3, 1])]

# 3. Learnable tensor
W = torch.randn((N, N, N), requires_grad=True, device='cuda')
optimizer = torch.optim.Adam([W], lr=0.1)

# 4. Training loop
for step in range(200):
    optimizer.zero_grad()

    # --- ST Gumbel-Softmax (per-(i,j) slice, dim=-1 on 3D) ---
    tau = max(0.1, 1.0 - step * 0.009)
    M_soft = F.gumbel_softmax(W, tau=tau, hard=False, dim=-1)
    index = M_soft.max(dim=-1, keepdim=True)[1]
    M_hard = torch.zeros_like(M_soft).scatter_(-1, index, 1.0)
    M = (M_hard - M_soft).detach() + M_soft

    # --- XLOG Execution ---
    # RD-16: Flatten to 1D for DLPack ndim==1 compliance
    program.set_rule_mask("W_mask",
                          M_hard.contiguous().view(-1),
                          M_soft.contiguous().view(-1), N)
    program.evaluate()

    # --- Per-Fact Surrogate Loss ---
    loss = torch.tensor(0.0, device='cuda')

    for rel_name, values in positive_examples:
        contributing = program.tagged_entries_containing_fact(rel_name, values)
        if contributing:
            credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
            loss = loss + (-torch.log(credit.clamp(min=1e-8)))
        else:
            # RD-21: Differentiable missed-positive penalty.
            # Push M_soft towards target k for all active (i,j) pairs.
            k_idx = rel_names.index(rel_name)
            penalty = -M_soft[:, :, k_idx].sum() / (N * N)
            loss = loss + penalty

    for rel_name, values in negative_examples:
        contributing = program.tagged_entries_containing_fact(rel_name, values)
        if contributing:
            credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
            loss = loss + (-torch.log((1.0 - credit).clamp(min=1e-8)))

    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

# 5. Extract converged rule
with torch.no_grad():
    for i in range(N):
        for j in range(N):
            k = W[i, j, :].argmax().item()
            if W[i, j, k].item() > 2.0:
                rule = f"{rel_names[k]}(X,Y) :- {rel_names[i]}(X,Z), {rel_names[j]}(Z,Y)."
                print(f"Learned: {rule}")
                program.commit_induced_rule(rule)
```

---

### Phase 5: TIIPS Selective Transductive Guidance (Optional)

Add a method to `CompiledIlpProgram` that exports current relation
contents as DLPack tensors:

```rust
pub fn extract_relation_state(
    &self,
    py: Python<'_>,
) -> PyResult<HashMap<String, PyObject>> {
    let mut state = HashMap::new();
    for (_, name) in &self.rel_index {
        if let Some(buf) = self.executor.store().get(name) {
            // Use provider.to_dlpack_table() (RD-9 naming fix)
            let table = self.provider.to_dlpack_table(/* need owned buf */);
            // Note: to_dlpack_table takes owned CudaBuffer.
            // For read-only export, we'd need to clone the buffer or
            // use download_column_* + upload to PyTorch.
            // Implementation deferred to Phase 5 since this is optional.
            todo!("Phase 5: DLPack export of relation state")
        }
    }
    Ok(state)
}
```

---

#### 4.11 GPU-Resident Loss/Gradient Path (v5)

**(Added v5.)** `compute_ilp_loss_grad_gpu` is a single Rust/CUDA call that
replaces the Python-side `_compute_loss_from_candidates()` loop.

**Pipeline:**
```
Python trainer
  └─ prog.compute_ilp_loss_grad_gpu(pos, neg, cand_probs)
       └─ Rust (lib.rs)
            ├─ Phase A: Parse pos/neg facts, resolve to compacted fact indices
            ├─ Phase B: For each candidate: build task (d_mask, fact_indices, cidx)
            ├─ Phase C: COO fill — scan + fill kernel per task (fully on-device)
            ├─ Phase D: Sort COO → CSR via radix sort + histogram kernel + prefix-sum
            ├─ Phase E: Forward kernel → credit + loss_contrib
            │           Reduce kernel → device-resident total loss
            │           Backward kernel → device-resident grad
            └─ Return (loss, grad) as DLPack capsules (zero-copy GPU→PyTorch)
```

**CUDA kernels:**
- `ilp_coo_fill_from_mask` (`kernels/ilp.cu`): Fill COO arrays from device-side mask + prefix-sum
- `ilp_csr_histogram` (`kernels/ilp.cu`): Histogram of fact indices for CSR row_offsets
- `ilp_reduce_sum_f32` / `ilp_reduce_sum_f64` (`kernels/ilp.cu`): Block-level sum reduction
- Forward/backward credit kernels (`kernels/ilp_credit.cu`): f32/f64 variants

**D2H transfer guarantee:**
- Non-chunked path: zero D2H transfers (strict byte-level accounting via `host_transfer_stats()`)
- Chunked fallback (COO exceeds `coo_memory_cap`): bounded D2H per chunk, merge on host
- `set_strict_zero_dtoh(True)`: raises instead of falling back to chunked path

**Python API:**
```python
prog.set_candidate_map([(i, j, k), ...])       # Upload candidate → index mapping
prog.set_coo_memory_cap(bytes)                  # Default 16 MB
prog.set_strict_zero_dtoh(True)                 # Reject chunked fallback
loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(positives, negatives, cand_probs)
loss = torch.from_dlpack(loss_dl)               # Zero-copy device tensor
grad = torch.from_dlpack(grad_dl)               # Zero-copy device tensor
```

---

## 5. Test Plan with Paper-Derived Benchmarks

### 5.1 Phase 1 Tests: Syntax & IR (No GPU Required)

| Test | Description | Pass Criterion |
|------|-------------|----------------|
| T1.1 | Parse `learnable(W) :: h(X,Y) :- b1(X,Z), b2(Z,Y).` | AST `learnable_rules` non-empty |
| T1.2 | Parse failure on malformed learnable rule | Specific error message |
| T1.3 | Lower LearnableRule to TensorMaskedJoin | `rel_index` populated from `rel_ids()` |
| T1.4 | `collect_relations()` includes all `rel_index` entries | Vec contains all RelIds |
| T1.5 | Learnable rule in recursive SCC | Fixpoint wraps TensorMaskedJoin |
| T1.6 | `parser.rs` `Rule::learnable_rule` arm | Parsed `Program.learnable_rules` has entry |
| T1.7 | `stratify.rs` includes learnable head in dep graph | `add_edge` called with `DepType::Positive` |
| T1.8 | `optimizer.rs` handles TensorMaskedJoin | All 3 functions pass through without panic |
| T1.9 | `executor.rs` exhaustive matches | `contains_non_monotonic_ops`, `collect_scan_rels`, `rewrite_scan_nth_impl` all handle variant |

### 5.2 Phase 2 Tests: DLPack Bridge & Kernel

| Test | Description | Pass Criterion |
|------|-------------|----------------|
| T2.1 | `set_rule_mask()` with flattened 1D tensors | Mask in IlpRegistry |
| T2.2 | `extract_nonzero_indices` kernel | 3x3x3 mask → correct (i,j,k) |
| T2.3 | Budget-cap truncation | 50 non-zeros, max=10 → top 10 |
| T2.4 | ILP module loads | Provider `get_func(ILP_MODULE, ...)` returns Some |
| T2.5 | DLPack round-trip | Flat PyTorch tensor → CudaBuffer → download matches |

### 5.3 Phase 3 Tests: Executor Integration

| Test | Description | Pass Criterion |
|------|-------------|----------------|
| T3.1 | TensorMaskedJoin with identity mask | Correct join results |
| T3.2 | TensorMaskedJoin with empty mask | No derivations |
| T3.3 | No mask registered (RD-12) | No-op, correct head schema, no corruption |
| T3.4 | Tag metadata matches | `get_tagged_results()` correct |
| T3.5 | `fact_exists()` host-side check | True/false correct |
| T3.6 | `tagged_entries_containing_fact()` | Returns correct (i,j,k) |
| T3.7 | Diff against existing facts | Only new facts added |
| T3.8 | Chain-union >2 buffers | Correct union of 3+ |
| T3.9 | Profiler records ILP ops | Stats include TensorMaskedJoin |
| T3.10 | TensorMaskedJoin inside Fixpoint | Semi-naive recursion works |

### 5.4 Phase 4 Tests: End-to-End Gradient Flow

| Test | Description | Pass Criterion |
|------|-------------|----------------|
| T4.1 | Loss differentiable w.r.t. M_soft | Non-zero `W.grad` |
| T4.2 | Gradient direction correct | Correct (i,j,k) gets positive gradient |
| T4.3 | Finite-difference check | Relative error < 1e-2 |
| T4.4 | Temperature annealing | Increasingly discrete M |
| T4.5 | Missed-positive penalty (RD-21) | Non-zero gradient for missed examples |

### 5.5 Phase 5 Tests: End-to-End ILP Benchmarks

**(v5):** Benchmark gate set updated to match implemented reliability suite.

| Test | Benchmark | Source | Pass Criterion |
|------|-----------|--------|----------------|
| T5.1 | Reach (transitive closure) | Custom | Correct rule within default step budget |
| T5.2 | Grandparent | Custom | Two-hop derivation rule |
| T5.3 | Colleague | Custom | Multi-predicate join rule |
| T5.4 | Plus2 | Custom | Arithmetic successor chain |
| T5.5 | Rule Commit | Custom | Post-commit evaluation matches |
| T5.6 | Commit removes learnable | Custom | No TensorMaskedJoin in recompiled plan |

**Reliability gate:** 4 stages (T5.1–T5.4) x 5 seeds = 20/20 with sparse backend.
**GA gate:** 4 stages x 50 seeds = 200/200 (Clopper-Pearson lower95 >= 0.98).

### 5.6 Validation Milestones

| Milestone | Tests | Gate |
|-----------|-------|------|
| **M1: Syntax** | T1.1-T1.9 | All pass |
| **M2: Bridge** | T2.1-T2.5 | All pass |
| **M3: Executor** | T3.1-T3.10 | All pass |
| **M4: Gradients** | T4.1-T4.5 | All pass |
| **M5: ILP Basic** | T5.1-T5.2 | Converge |
| **M6: ILP Complete** | T5.1-T5.6 | All pass |
| **M7: GPU Loss** | 16/16 credit tests + strict D2H gate | All pass |
| **M8: Reliability** | 4 stages x 5 seeds (beta) / 50 seeds (GA) | 20/20 / 200/200 |

---

## 6. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Surrogate loss noise | Slow convergence | Temperature annealing + per-fact credit |
| `fact_exists()` host round-trip | Training bottleneck | Batch into single download per relation |
| DLPack 1D flattening constraint | Python complexity | Clear API docs; could extend dlpack.rs later |
| N^3 memory for large schemas | N=1000 → 4GB | Filter by arity/type signature |
| Semi-naive re-derivation | Wasted joins | Explicit `diff_gpu` before store |
| `try_slice` returns `Option` | Cudarc 0.19 returns `Option`, not `Result` | All call sites use `.ok_or_else()` (RD-23) |
| `device_row_count` private | Cannot call from new code | `read_device_row_count` helper using public APIs (RD-22) |
| Per-fact credit re-executes joins | Extra GPU work per credit check | ILP-scale relations are small; negligible cost (RD-24) |

---

## 7. Files Changed Summary

| File | Change | Description |
|------|--------|-------------|
| `crates/xlog-logic/src/grammar.pest` | Modify | `learnable_rule` production |
| `crates/xlog-logic/src/ast.rs` | Modify | `LearnableRule` struct, `Program.learnable_rules` |
| `crates/xlog-logic/src/parser.rs` | Modify | `Rule::learnable_rule` arm + `build_learnable_rule()` |
| `crates/xlog-logic/src/lower.rs` | Modify | `lower_learnable_rule()` |
| `crates/xlog-logic/src/stratify.rs` | Modify | Learnable rule edges in `build_dependency_graph()` |
| `crates/xlog-logic/src/optimizer.rs` | Modify | TensorMaskedJoin arms in `predicate_pushdown`, `estimate_width`, `estimate_cost` |
| `crates/xlog-ir/src/rir.rs` | Modify | `TensorMaskedJoin` variant + `collect_relations` arm |
| `crates/xlog-runtime/src/ilp_registry.rs` | **New** | `IlpRegistry`, `IlpMask`, `IlpTaggedResult`, `IlpTagEntry` |
| `crates/xlog-runtime/src/executor.rs` | Modify | ILP fields + `execute_tensor_masked_join()` + 3 exhaustive match arms |
| `crates/xlog-runtime/src/lib.rs` | Modify | Export `ilp_registry` |
| `kernels/ilp.cu` | **New** | `extract_nonzero_indices` kernel |
| `crates/xlog-cuda/src/kernel_manifest_data.rs` | Modify | Add `"ilp"` |
| `crates/xlog-cuda/src/provider/mod.rs` + `provider/ilp.rs` | Modify | Assertion, `ILP_MODULE`, `load_ptx`, `extract_active_rule_indices()` |
| `crates/pyxlog/src/lib.rs` | Modify | `IlpProgramFactory`, `CompiledIlpProgram`, helpers |

**Not modified:** `crates/xlog-prob/`.

---

## 8. Architectural Decision Record

### Why xlog-runtime, not xlog-prob?

ILP requires **parametric relation selection** (second-order). xlog-prob
compiles first-order fixed-structure circuits. Neural predicates provide
variable weights for fixed structure; ILP requires variable structure.

### Why surrogate loss, not XGCF gradients?

XGCF circuits are immutable, handle probability not relational algebra,
and ILP needs gradients through discrete set operations. ST-Gumbel-Softmax
sidesteps all three via PyTorch autograd.

### Why host-side membership check?

`CudaKernelProvider` has no `contains_tuple` method. For ILP's typical
scale (relations with hundreds of rows, arity 2), downloading columns and
scanning on the host is negligible. A GPU kernel can be added later if
profiling shows this is a bottleneck.

### Why flatten to 1D?

The DLPack import path (`from_dlpack_tensors`, dlpack.rs:221) enforces
`ndim == 1`. Changing this would touch a well-tested, safety-critical code
path shared by neural predicates. Flattening in Python is trivial
(`.contiguous().view(-1)`) and the kernel already indexes linearly.

---

## 9. Changelog

### v4.3 → v4.4

| # | v4.3 Finding | Fix |
|---|-------------|-----|
| 1 | `lower_learnable_rule` defined but never called from `lower_program` — learnable rules never emitted (Blocker) | RD-32: Explicit loop over `program.learnable_rules` before `builder.build()`, after proper rules |
| 2 | `build_learnable_rule` uses `build_atom` but grammar produces `head` pair (Blocker) | RD-33: Changed to `build_head` (parser.rs:652), which handles `head_term_list` |
| 3 | `rule.body[0]`/`rule.body[1]` indexed without validation — panics on non-binary or negated bodies (High) | RD-34: Arity check (exactly 2) + positive-only check with `XlogError::Compilation` on violation |
| 4 | `ilp_registry_mut()` and `ilp_last_result()` called by pyxlog but never defined on Executor (High) | RD-35: Accessor methods added to Executor impl block in §4.4 |
| 5 | `rel_index` built from `HashMap::iter()` — nondeterministic tensor dimension mapping (Medium) | RD-36: `.sort_by_key(\|(id, _)\| id.0)` in both `lower_learnable_rule` and pyxlog `compile()` |
| 6 | `Schema::new(vec![])` fallback silently produces 0-column buffer (Medium) | RD-37: Both paths now `ok_or_else` → `XlogError::Execution` with diagnostic message |

### v4.2 → v4.3

| # | v4.2 Finding | Fix |
|---|-------------|-----|
| 1 | `Program` type collision — pyxlog already has `#[pyclass] pub struct Program` (Blocker) | RD-29: `use xlog_logic::ast::Program as AstProgram;` — all ILP code uses `AstProgram` |
| 2 | `anyhow!` in lowering — lower.rs uses `xlog_core::Result`/`XlogError`, not anyhow (Blocker) | RD-30: Removed `anyhow!`; uses `get_or_create_rel_id` which never errors |
| 3 | `head_rel_id` lookup can fail for head-only predicates not yet in `rel_ids` (Blocker) | RD-30: `self.get_or_create_rel_id(&head_rel_name)` (lower.rs:97) allocates lazily |
| 4 | `push_term_bytes` resolution left as open choice (High) | RD-31: Deterministic — duplicate in pyxlog/lib.rs (~40 lines, already duplicated in xlog-logic/examples) |

### v4.1 → v4.2

| # | v4.1 Finding | Fix |
|---|-------------|-----|
| 1 | `estimate_width` uses `self.schemas.get(head_rel_name)` but optimizer schemas keyed by `RelId` (Blocker) | RD-27: Added `head_rel_id: RelId` to `TensorMaskedJoin`; `estimate_width` uses `self.schemas.get(head_rel_id)` |
| 2 | `ScalarType::U8` does not exist in `xlog-core` (Blocker) | Removed `ScalarType::U8` branch; Bool uses `download_column_bool` → `if v { 1i64 } else { 0i64 }` |
| 3 | `extract_tmj_keys` uses wrong shapes: `Fixpoint { body }`, `Union { left, right }`, `plan.nodes()` (Blocker) | RD-28: Fixpoint→`base`/`recursive`, Union→`inputs`, plan→`rules_by_scc.iter().flatten().body`; full recursive walk of all node types |
| 4 | Base facts never loaded — `execute_plan` doesn't populate store from AST facts (Blocker) | RD-26: `load_facts_into_store()` helper mirrors `logic.rs:171-221`; called in `compile()`, `evaluate()`, `commit_induced_rule()`. `AstProgram` stored on `CompiledIlpProgram`. |

### v4 → v4.1

| # | v4 Finding | Fix |
|---|-----------|-----|
| 1 | `device_row_count` is private (`fn` not `pub fn`) in `provider/mod.rs` (Blocker) | RD-22: `read_device_row_count` helper inlines same pattern using public APIs (`provider.device()`, `buffer.num_rows_device()`, `dtoh_sync_copy_into`) |
| 2 | `try_slice` returns `Option`, not `Result` — `.map_err()` won't compile (Blocker) | RD-23: All `.try_slice()` calls changed from `.map_err()` to `.ok_or_else()` |
| 3 | RD-12 schema fallback uses `rel_index.first()` which may not match head (High) | Added `head_rel_name: String` field to `TensorMaskedJoin`; all schema lookups (no-mask path, return path, `estimate_width`) now use `head_rel_name` instead of `rel_index.first()` |
| 4 | Per-fact credit regressed to relation-level — credits ALL entries targeting k (High) | RD-24: `tagged_entries_containing_fact` re-executes `hash_join_v2` per candidate entry and checks per-fact membership individually |
| 5 | `fact_exists` hardcodes `download_column_i64` — non-i64 relations fail (Medium) | RD-25: Schema-aware dispatch via `buf.schema().column_type()` → `download_column::<T>()` (turbofish generics post-Wave-2) with i64 widening |

### v3 → v4

| # | v3 Finding | Fix |
|---|-----------|-----|
| 1 | Provider uses `column(0)?`, `.to_host()`, `.slice()` (Blocker) | RD-19: Rewritten with `memory.alloc`, `dtoh_sync_copy_into`, `column_bytes_view`, `try_slice` |
| 2 | DLPack ndim==1 blocks 3D tensors (Blocker) | RD-16: Python flattens to 1D before DLPack export |
| 3 | `contains_tuple` undefined (Blocker) | RD-18: Host-side `fact_exists()` using `download_column_*` |
| 4 | `CudaBuffer::clone()` assumed (Blocker) | RD-17: Tags store metadata only `(i,j,k,num_rows)`, not buffer clones |
| 5 | Compile path misses relation setup (Blocker) | RD-15: Full 3-step setup (register, seed schemas, execute) matching logic.rs |
| 6 | No-mask no-op returns wrong schema (High) | RD-12 updated: returns head relation schema from store |
| 7 | Under-scoped exhaustive match sites (High) | RD-20: Complete list of 9 match sites with required arms |
| 8 | Stratify API mismatch (High) | Fixed: `add_edge(from, to, DepType::Positive)`, `body_lit.atom()` |
| 9 | API drift: `make_provider`, `DlpackTable::new` (Medium) | Fixed: `provider_from_config`, `provider.to_dlpack_table()` |
| 10 | Missed-positive `loss += 10.0` has no gradient (Medium) | RD-21: Differentiable penalty using `M_soft[:,:,k_idx].sum()` |

### v2 → v3

(See v3 document for that changelog.)

---

## Appendix A: Relationship to Reference Papers

### A.1 DeepMind Differentiable ILP (JAIR)

| Paper | Implementation | Improvement |
|-------|---------------|-------------|
| Soft rule eval (all clauses) | Hard mask + sparse dispatch | Eliminates O(N^3) memory |
| Product T-Norm | `M_soft * indicator` product | Equivalent gradient flow |
| Template clause generation | Universal join template | Reuses hash_join_v2 |
| Forward chaining | Semi-naive fixpoint | Delta-only processing |

### A.2 TIIPS (Transductive Guidance)

| Paper | Implementation | Notes |
|-------|---------------|-------|
| Transductive Teacher | `extract_relation_state()` | Optional Phase 5 |
| Selective guidance | Plateau detection in Python | No core changes |
| Cross-entropy against subgoal | Additional positive examples | Same loss mechanism |
