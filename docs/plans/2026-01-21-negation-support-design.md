# Negation Support in Probabilistic Programs with Gradient Computation

## Overview

Add full negation support to the exact d-DNNF inference engine with gradient computation, using stratification-first semantics with Well-Founded Semantics (WFS) fallback for non-stratified programs.

**Status**: Design approved, ready for implementation

## Goals

1. Support negation in probabilistic programs (`not p(X)`)
2. Provide exact gradients for neural-symbolic training
3. Handle both stratified and non-monotone (cyclic) negation
4. Maintain compatibility with existing MC engine semantics

## Architecture

```
Source Program
     │
     ▼
┌─────────────────────┐
│ Stratification      │ ◄── Detect if program has cycles through negation
│ Analysis            │     Classify SCCs as stratified or non-monotone
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Provenance          │ ◄── Modified to handle BodyLiteral::Negated
│ Extraction          │     NNF transformation for negated leaves
└─────────────────────┘     WFS fixed-point for non-stratified SCCs
     │
     ▼
┌─────────────────────┐
│ PIR (Provenance IR) │ ◄── Add NegLit node type for negated leaves
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ CNF Encoding &      │ ◄── Handle NegLit in Tseitin transformation
│ d-DNNF Compilation  │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Gradient            │ ◄── Negated leaf gradients: flip sign
│ Computation         │     WFS: gradients only for defined atoms
└─────────────────────┘
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Semantics | Stratification-first + WFS fallback | Covers 90%+ cases efficiently; WFS for complex cases |
| Implementation | NNF transformation | Minimal PIR changes; clean gradient flow |
| Phasing | Single complete implementation | User requirement |
| Validation | MC + ProbLog + formal tests | Comprehensive correctness verification |

## PIR Changes

Add a `NegLit` variant to represent negated probabilistic leaves:

```rust
// In pir.rs
pub enum PirNode {
    Const(bool),
    Lit { leaf: LeafId },           // Positive leaf: weight (p, 1-p)
    NegLit { leaf: LeafId },        // Negated leaf: weight (1-p, p)
    And { children: Vec<PirNodeId> },
    Or { children: Vec<PirNodeId> },
    Decision { var: ChoiceVarId, child_false: PirNodeId, child_true: PirNodeId },
}
```

**Why `NegLit` instead of `Not { child }`?**

1. Keeps circuit in NNF - negations only at leaves
2. Avoids nested negation handling (`not not p`)
3. Clean weight semantics: `NegLit` uses complemented weight directly
4. Simpler CNF encoding - no De Morgan transformations needed

**PirBuilder Extension**:

```rust
impl PirGraph {
    pub fn neg_lit(&mut self, leaf: LeafId) -> PirNodeId {
        self.push_node(PirNode::NegLit { leaf })
    }
}
```

**Weight Interpretation**:

| Node | True Weight | False Weight |
|------|-------------|--------------|
| `Lit { leaf: i }` | `p_i` | `1 - p_i` |
| `NegLit { leaf: i }` | `1 - p_i` | `p_i` |

## Stratification Analysis

**Purpose**: Detect whether a program can be evaluated layer-by-layer (stratified) or has cycles through negation (non-monotone) requiring WFS.

**Data Structure**:

```rust
// In stratify.rs or new file
pub struct StratificationResult {
    /// SCCs in evaluation order
    pub sccs: Vec<Vec<String>>,
    /// Which SCCs contain cycles through negation
    pub non_monotone_sccs: HashSet<usize>,
    /// Stratum number for each predicate (if stratified)
    pub strata: HashMap<String, usize>,
}

pub fn analyze_stratification(program: &Program) -> StratificationResult {
    // 1. Build dependency graph with edge polarity (positive/negative)
    // 2. Find SCCs
    // 3. Check each SCC for negative internal edges (non-monotone)
    // 4. For stratified SCCs, assign stratum numbers
}
```

**Classification Rules**:

- **Stratified SCC**: No negative edges within the SCC. Can use simple two-valued evaluation.
- **Non-monotone SCC**: Has cycle through negation. Requires WFS fixed-point.

## Provenance Extraction for Negation

**Remove Current Blocker** (provenance.rs:691-696):

```rust
// DELETE THIS
BodyLiteral::Negated(atom) => {
    return Err(XlogError::Compilation(format!(
        "Negation not supported in provenance extraction (found not {})",
        atom.predicate
    )));
}
```

**Replacement - Stratified Case**:

```rust
BodyLiteral::Negated(atom) => {
    let relation = store.get(&atom.predicate).ok_or_else(|| {
        XlogError::Compilation(format!("Unknown predicate: {}", atom.predicate))
    })?;

    // Try to unify and get the tuple's provenance
    for (binding, prov) in states.drain(..) {
        match unify_and_lookup(atom, &binding, relation) {
            Some((new_binding, tuple_prov)) => {
                // Tuple exists - negate its provenance
                let neg_prov = negate_provenance(tuple_prov, builder);
                next_states.push((new_binding, builder.and(vec![prov, neg_prov])));
            }
            None => {
                // Tuple doesn't exist - negation succeeds (const true)
                next_states.push((binding, prov));
            }
        }
    }
}
```

**`negate_provenance()` Function**:

```rust
fn negate_provenance(prov: PirNodeId, builder: &mut PirBuilder) -> PirNodeId {
    match builder.node(prov) {
        PirNode::Const(b) => builder.push_node(PirNode::Const(!b)),
        PirNode::Lit { leaf } => builder.neg_lit(*leaf),  // Key: use NegLit
        PirNode::NegLit { leaf } => builder.lit(*leaf),   // Double negation
        // For compound nodes: apply De Morgan (rare in practice)
        PirNode::And { children } => { /* Or of negated children */ }
        PirNode::Or { children } => { /* And of negated children */ }
    }
}
```

## Well-Founded Semantics for Non-Monotone SCCs

**When WFS Applies**: SCCs identified as non-monotone (cycles through negation) use WFS instead of simple iteration.

**WFS Three-Valued Logic**:

| Value | Meaning | Probability Treatment |
|-------|---------|----------------------|
| True | Definitely derivable | Normal probability |
| False | Definitely not derivable | Probability = 0 |
| Undefined | In cycle, neither provable | Excluded from computation |

**WFS Algorithm (Alternating Fixed-Point)**:

```rust
fn evaluate_wfs_scc(
    scc_predicates: &[String],
    rules: &[Rule],
    store: &mut HashMap<String, Relation>,
    builder: &mut PirBuilder,
) -> Result<WfsResult> {
    // Initialize: all atoms in SCC are undefined
    let mut true_set: HashMap<GroundAtom, PirNodeId> = HashMap::new();
    let mut false_set: HashSet<GroundAtom> = HashSet::new();

    loop {
        // Unfounded set computation: find atoms that cannot be supported
        let unfounded = compute_unfounded_set(&true_set, rules, store);
        false_set.extend(unfounded);

        // Consequence operator: derive what must be true given false_set
        let new_true = derive_consequences(rules, &true_set, &false_set, store, builder);

        if new_true.is_empty() && unfounded.is_empty() {
            break; // Fixed point reached
        }

        true_set.extend(new_true);
    }

    // Atoms not in true_set or false_set remain undefined
    Ok(WfsResult { true_set, false_set })
}
```

**Gradient Treatment for Undefined Atoms**:

- Queries on undefined atoms return probability 0 with zero gradient
- This is conservative: no gradient signal for genuinely ambiguous cases
- Matches ProbLog's behavior

## CNF Encoding & Gradient Computation

**CNF Encoding for NegLit**:

```rust
// In cnf.rs - extend encode_node()
fn encode_node(node: &PirNode, ...) -> CnfVar {
    match node {
        PirNode::Lit { leaf } => {
            // Positive literal: variable for leaf
            get_or_create_leaf_var(*leaf, polarity: Positive)
        }
        PirNode::NegLit { leaf } => {
            // Negated literal: same variable, opposite polarity
            get_or_create_leaf_var(*leaf, polarity: Negative)
        }
        // And, Or, Decision unchanged...
    }
}
```

**Weight Table**:

```rust
// For leaf i with probability p:
//   Lit { leaf: i }    uses weights (log(p), log(1-p))
//   NegLit { leaf: i } uses weights (log(1-p), log(p))  // Swapped!
```

**Gradient Computation**:

```rust
// In gradient computation (exact.rs)
fn compute_leaf_gradient(leaf: LeafId, is_negated: bool, grad_true: f64, grad_false: f64) -> f64 {
    if is_negated {
        // d/dp of using (1-p, p) = negative of normal gradient
        grad_false - grad_true  // Swapped and sign flipped
    } else {
        grad_true - grad_false  // Normal case
    }
}
```

**Key Insight**: The gradient w.r.t. the original probability `p` flips sign for negated leaves because `∂(1-p)/∂p = -1`.

## Testing Strategy

**Test Categories**:

```
python/tests/test_negation.py
├── TestStratifiedNegation
│   ├── test_simple_negation          # dry() :- not rain().
│   ├── test_multi_layer_stratified   # a :- not b. b :- not c.
│   ├── test_negation_with_rules      # path + not blocked
│   └── test_negation_gradient_flow   # verify gradients reach networks
│
├── TestNonMonotoneWFS
│   ├── test_simple_cycle             # p :- not q. q :- not p.
│   ├── test_wfs_undefined            # atoms in cycle → undefined
│   ├── test_wfs_partial_definition   # some atoms defined, some not
│   └── test_wfs_gradient_zero        # undefined atoms have zero gradient
│
├── TestGradientCorrectness
│   ├── test_finite_difference_*      # compare analytic vs numeric gradients
│   └── test_gradient_sign_flip       # negated leaf gradient is negative
│
└── TestReferenceComparison
    ├── test_mc_probability_match     # exact vs MC within confidence interval
    └── test_problog_compatibility    # match ProbLog on reference programs
```

**Finite Difference Gradient Check**:

```python
def test_finite_difference_negation():
    """Verify gradients by numerical differentiation."""
    program = create_program_with_negation()

    eps = 1e-5
    for param in network.parameters():
        # Compute analytic gradient
        loss = program.forward_backward(query)
        analytic_grad = param.grad.clone()

        # Compute numeric gradient
        param.data += eps
        loss_plus = program.nll_loss(query)
        param.data -= 2 * eps
        loss_minus = program.nll_loss(query)
        numeric_grad = (loss_plus - loss_minus) / (2 * eps)

        assert torch.allclose(analytic_grad, numeric_grad, rtol=1e-3)
```

## Implementation Plan

**Files to Modify**:

| File | Changes |
|------|---------|
| `crates/xlog-prob/src/pir.rs` | Add `NegLit { leaf: LeafId }` variant, `neg_lit()` builder method |
| `crates/xlog-prob/src/provenance.rs` | Remove negation blocker, add `negate_provenance()`, handle `BodyLiteral::Negated` |
| `crates/xlog-prob/src/cnf.rs` | Handle `NegLit` in Tseitin encoding with swapped polarity |
| `crates/xlog-prob/src/exact.rs` | Track leaf polarity, flip gradient sign for negated leaves |
| `crates/xlog-logic/src/stratify.rs` | Add `analyze_stratification()` with edge polarity tracking |

**New Files**:

| File | Purpose |
|------|---------|
| `crates/xlog-prob/src/wfs.rs` | Well-Founded Semantics implementation for non-monotone SCCs |
| `python/tests/test_negation.py` | Comprehensive negation test suite |

**Implementation Order**:

1. **PIR extension** - Add `NegLit` node (~5 lines)
2. **Stratification analysis** - Detect non-monotone SCCs (~100 lines)
3. **Provenance extraction** - Handle negated literals for stratified case (~50 lines)
4. **CNF encoding** - Handle `NegLit` with swapped polarity (~20 lines)
5. **Gradient computation** - Sign flip for negated leaves (~30 lines)
6. **WFS implementation** - Alternating fixed-point for non-monotone (~200 lines)
7. **Tests** - Full test suite (~300 lines)

**Estimated Total**: ~700 lines of Rust + ~300 lines of Python tests

## Success Criteria

1. All existing tests continue to pass
2. New negation tests pass for both stratified and non-monotone cases
3. Gradients verified via finite difference checks
4. Probabilities match MC engine within confidence intervals
5. Probabilities match ProbLog on reference programs
