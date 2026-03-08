# Provenance Primitives Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose retained provenance metadata (leaf atoms, choice sources, formula atoms) so external Rust consumers can resolve PIR nodes back to source atoms and annotated-disjunction metadata.

**Architecture:** Inline retention at existing extraction allocation sites in `extract_from_program()`. New `ChoiceSource` type. Three read-only accessors on `Provenance`. Top-level re-exports from `xlog-prob`. All Rust-only, no Python bindings.

**Tech Stack:** Rust, xlog-prob crate, BTreeMap

**Design doc:** `docs/plans/2026-03-08-provenance-primitives-design.md`

---

### Task 1: Add `ChoiceSource` type and make `GroundAtom::new()` public

**Files:**
- Modify: `crates/xlog-prob/src/provenance.rs:44-57` (GroundAtom impl block)

**Step 1: Make `GroundAtom::new()` public**

In `crates/xlog-prob/src/provenance.rs:51`, change:

```rust
    fn new(predicate: impl Into<String>, args: Vec<Value>) -> Self {
```

to:

```rust
    pub fn new(predicate: impl Into<String>, args: Vec<Value>) -> Self {
```

**Step 2: Add `ChoiceSource` type**

In `crates/xlog-prob/src/provenance.rs`, after the `GroundAtom` impl block (after line 57),
add:

```rust
/// Metadata for a single Bernoulli decision stage in an annotated disjunction.
#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceSource {
    /// Explicit heads of the annotated disjunction, paired with probabilities.
    /// Does not include the synthetic implicit "none" branch.
    pub choices: Vec<(GroundAtom, f64)>,
    /// Position of this ChoiceVarId in the m-1 Bernoulli decision chain.
    pub choice_index: usize,
    /// Enclosing annotated-disjunction identity. `None` in v1.
    pub source_id: Option<usize>,
}
```

**Step 3: Verify it compiles**

Run: `cargo check -p xlog-prob`
Expected: compiles with no errors (new type is not yet used)

**Step 4: Commit**

```bash
git add crates/xlog-prob/src/provenance.rs
git commit -m "feat(prob): add ChoiceSource type and make GroundAtom::new() public"
```

---

### Task 2: Add new fields to `Provenance` and update construction

**Files:**
- Modify: `crates/xlog-prob/src/provenance.rs:280-287` (Provenance struct)
- Modify: `crates/xlog-prob/src/provenance.rs:438-445` (Provenance construction in extract_from_program)
- Modify: `crates/xlog-prob/src/provenance.rs:308-309` (local variables in extract_from_program)

**Step 1: Add fields to `Provenance` struct**

In `crates/xlog-prob/src/provenance.rs`, the `Provenance` struct at line 280 currently is:

```rust
pub struct Provenance {
    pub pir: PirGraph,
    pub leaf_probs: BTreeMap<LeafId, f64>,
    pub choice_probs: BTreeMap<ChoiceVarId, (f64, f64)>,
    tuple_formulas: BTreeMap<GroundAtom, PirNodeId>,
    pub queries: Vec<GroundAtom>,
    pub evidence: Vec<(GroundAtom, bool)>,
}
```

Change to:

```rust
pub struct Provenance {
    pub pir: PirGraph,
    pub leaf_probs: BTreeMap<LeafId, f64>,
    pub choice_probs: BTreeMap<ChoiceVarId, (f64, f64)>,
    tuple_formulas: BTreeMap<GroundAtom, PirNodeId>,
    pub queries: Vec<GroundAtom>,
    pub evidence: Vec<(GroundAtom, bool)>,
    pub leaf_atoms: BTreeMap<LeafId, GroundAtom>,
    pub choice_sources: BTreeMap<ChoiceVarId, ChoiceSource>,
}
```

**Step 2: Initialize new maps in `extract_from_program()`**

At line 308-309, after the existing map declarations:

```rust
    let mut leaf_probs: BTreeMap<LeafId, f64> = BTreeMap::new();
    let mut choice_probs: BTreeMap<ChoiceVarId, (f64, f64)> = BTreeMap::new();
```

Add:

```rust
    let mut leaf_atoms: BTreeMap<LeafId, GroundAtom> = BTreeMap::new();
    let mut choice_sources: BTreeMap<ChoiceVarId, ChoiceSource> = BTreeMap::new();
```

**Step 3: Include new fields in Provenance construction**

At line 438-445, the `Ok(Provenance { ... })` block currently is:

```rust
    Ok(Provenance {
        pir: builder.finish(),
        leaf_probs,
        choice_probs,
        tuple_formulas,
        queries,
        evidence,
    })
```

Change to:

```rust
    Ok(Provenance {
        pir: builder.finish(),
        leaf_probs,
        choice_probs,
        tuple_formulas,
        queries,
        evidence,
        leaf_atoms,
        choice_sources,
    })
```

**Step 4: Verify it compiles**

Run: `cargo check -p xlog-prob`
Expected: compiles (maps are empty but structurally valid)

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/provenance.rs
git commit -m "feat(prob): add leaf_atoms and choice_sources fields to Provenance"
```

---

### Task 3: Retain leaf atoms during extraction

**Files:**
- Modify: `crates/xlog-prob/src/provenance.rs:322-335` (probabilistic facts loop)

**Step 1: Write the failing test**

Create file `crates/xlog-prob/tests/test_provenance_primitives.rs`:

```rust
use xlog_prob::pir::LeafId;
use xlog_prob::provenance::{Provenance, Value};

#[test]
fn leaf_atom_resolves_prob_facts() {
    let src = r#"
        0.3::rain.
        0.7::sprinkler.
        query(rain).
    "#;
    let prov: Provenance = xlog_prob::provenance::extract_from_source(src).unwrap();

    // Two prob facts → two leaves.
    assert_eq!(prov.leaf_probs.len(), 2);
    assert_eq!(prov.leaf_atoms.len(), 2);

    // LeafId(0) should map to rain, LeafId(1) to sprinkler.
    let atom0 = prov.leaf_atom(LeafId::new(0)).unwrap();
    assert_eq!(atom0.predicate, "rain");
    assert!(atom0.args.is_empty());

    let atom1 = prov.leaf_atom(LeafId::new(1)).unwrap();
    assert_eq!(atom1.predicate, "sprinkler");
    assert!(atom1.args.is_empty());
}

#[test]
fn leaf_atom_returns_none_for_invalid() {
    let src = r#"
        0.5::edge(1,2).
        query(edge(1,2)).
    "#;
    let prov: Provenance = xlog_prob::provenance::extract_from_source(src).unwrap();
    assert!(prov.leaf_atom(LeafId::new(9999)).is_none());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob --test test_provenance_primitives -- --nocapture`
Expected: FAIL — `leaf_atom` method does not exist yet

**Step 3: Add `leaf_atom()` accessor and retain leaf atoms in extraction**

In `crates/xlog-prob/src/provenance.rs`, add to the `impl Provenance` block (after
`query_formula`, around line 294):

```rust
    pub fn leaf_atom(&self, leaf: LeafId) -> Option<&GroundAtom> {
        self.leaf_atoms.get(&leaf)
    }
```

In the probabilistic facts loop (lines 322-335), after `leaf_probs.insert(leaf, pf.prob);`
(line 329), add:

```rust
        leaf_atoms.insert(leaf, key.clone());
```

The full loop should now read:

```rust
    // Probabilistic facts.
    let mut next_leaf: u32 = 0;
    for pf in &program.prob_facts {
        validate_prob(pf.prob, "probabilistic fact")?;
        let key = atom_key_from_ground_atom(&pf.atom)?;
        let leaf = LeafId::new(next_leaf);
        next_leaf = next_leaf.saturating_add(1);
        leaf_probs.insert(leaf, pf.prob);
        leaf_atoms.insert(leaf, key.clone());

        let rel = store
            .entry(key.predicate.clone())
            .or_insert_with(Relation::new);
        rel.insert_or(key.args.clone(), builder.lit(leaf), &mut builder);
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob --test test_provenance_primitives -- --nocapture`
Expected: PASS (both tests)

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/provenance.rs crates/xlog-prob/tests/test_provenance_primitives.rs
git commit -m "feat(prob): retain leaf_atoms during extraction with leaf_atom() accessor"
```

---

### Task 4: Retain choice sources during extraction

**Files:**
- Modify: `crates/xlog-prob/src/provenance.rs:484-555` (compile_annotated_disjunction)
- Modify: `crates/xlog-prob/src/provenance.rs:337-356` (AD loop in extract_from_program)
- Modify: `crates/xlog-prob/tests/test_provenance_primitives.rs`

**Step 1: Write the failing test**

Append to `crates/xlog-prob/tests/test_provenance_primitives.rs`:

```rust
use xlog_prob::pir::ChoiceVarId;
use xlog_prob::provenance::ChoiceSource;

#[test]
fn choice_source_resolves_ad_metadata() {
    // 2 explicit heads → 1 choice variable (m-1 = 1).
    let src = r#"
        0.3::color(red) ; 0.7::color(blue).
        query(color(red)).
    "#;
    let prov: Provenance = xlog_prob::provenance::extract_from_source(src).unwrap();

    assert_eq!(prov.choice_sources.len(), prov.choice_probs.len());

    let cs = prov.choice_source(ChoiceVarId::new(0)).unwrap();

    // choices: explicit heads only (no implicit none branch).
    assert_eq!(cs.choices.len(), 2);
    assert_eq!(cs.choices[0].0.predicate, "color");
    assert_eq!(cs.choices[0].1, 0.3);
    assert_eq!(cs.choices[1].0.predicate, "color");
    assert_eq!(cs.choices[1].1, 0.7);

    // choice_index = 0 (first and only decision stage).
    assert_eq!(cs.choice_index, 0);

    // source_id = None in v1.
    assert!(cs.source_id.is_none());
}

#[test]
fn choice_source_returns_none_for_invalid() {
    let src = r#"
        0.3::color(red) ; 0.7::color(blue).
        query(color(red)).
    "#;
    let prov: Provenance = xlog_prob::provenance::extract_from_source(src).unwrap();
    assert!(prov.choice_source(ChoiceVarId::new(9999)).is_none());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob --test test_provenance_primitives -- choice_source --nocapture`
Expected: FAIL — `choice_source` method does not exist yet

**Step 3: Add `choice_source()` accessor and retain choice sources in extraction**

In `crates/xlog-prob/src/provenance.rs`, add to the `impl Provenance` block:

```rust
    pub fn choice_source(&self, var: ChoiceVarId) -> Option<&ChoiceSource> {
        self.choice_sources.get(&var)
    }
```

Modify `compile_annotated_disjunction()` signature (line 484) to accept the
`choice_sources` map:

```rust
fn compile_annotated_disjunction(
    ad: &xlog_logic::ast::AnnotatedDisjunction,
    next_choice: &mut u32,
    choice_probs: &mut BTreeMap<ChoiceVarId, (f64, f64)>,
    choice_sources: &mut BTreeMap<ChoiceVarId, ChoiceSource>,
    builder: &mut PirBuilder,
) -> Result<(Vec<ChoiceVarId>, Vec<PirNodeId>)> {
```

Inside `compile_annotated_disjunction()`, build the `choices` vec from the AD's explicit
heads. Add this after the existing `validate_prob` loop (after line 493, before the `probs`
vec construction):

```rust
    let explicit_choices: Vec<(GroundAtom, f64)> = ad
        .choices
        .iter()
        .map(|pf| {
            let atom = atom_key_from_ground_atom(&pf.atom).unwrap();
            (atom, pf.prob)
        })
        .collect();
```

Note: the `atom_key_from_ground_atom` call here is safe because the same validation was
already done in the loop above (line 492). The `.unwrap()` cannot panic.

Inside the `for i in 0..(m - 1)` loop (lines 519-533), after `choice_probs.insert(...)`:

```rust
        choice_sources.insert(var, ChoiceSource {
            choices: explicit_choices.clone(),
            choice_index: i,
            source_id: None,
        });
```

Update the call site in `extract_from_program()` (line 345-346). Change:

```rust
        let (vars, outcome_formulas) =
            compile_annotated_disjunction(ad, &mut next_choice, &mut choice_probs, &mut builder)?;
```

to:

```rust
        let (vars, outcome_formulas) =
            compile_annotated_disjunction(ad, &mut next_choice, &mut choice_probs, &mut choice_sources, &mut builder)?;
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob --test test_provenance_primitives -- --nocapture`
Expected: PASS (all 4 tests)

**Step 5: Run full workspace tests to check for regressions**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`
Expected: all pass

**Step 6: Commit**

```bash
git add crates/xlog-prob/src/provenance.rs crates/xlog-prob/tests/test_provenance_primitives.rs
git commit -m "feat(prob): retain choice_sources during extraction with choice_source() accessor"
```

---

### Task 5: Add `atoms_with_formulas()` accessor

**Files:**
- Modify: `crates/xlog-prob/src/provenance.rs` (Provenance impl block)
- Modify: `crates/xlog-prob/tests/test_provenance_primitives.rs`

**Step 1: Write the failing test**

Append to `crates/xlog-prob/tests/test_provenance_primitives.rs`:

```rust
use xlog_prob::pir::PirNodeId;

#[test]
fn atoms_with_formulas_returns_all_formula_atoms() {
    let src = r#"
        wet :- rain.
        wet :- sprinkler.
        0.3::rain.
        0.7::sprinkler.
        query(wet).
    "#;
    let prov: Provenance = xlog_prob::provenance::extract_from_source(src).unwrap();

    let formula_atoms: Vec<_> = prov.atoms_with_formulas().collect();

    // Should contain: rain, sprinkler (prob facts) + wet (derived).
    assert_eq!(formula_atoms.len(), 3);

    // Every atom returned should have a valid PirNodeId.
    for (_atom, node_id) in &formula_atoms {
        assert!(prov.pir.node(*node_id).is_some());
    }

    // Check that specific atoms are present.
    let predicates: Vec<&str> = formula_atoms.iter().map(|(a, _)| a.predicate.as_str()).collect();
    assert!(predicates.contains(&"rain"));
    assert!(predicates.contains(&"sprinkler"));
    assert!(predicates.contains(&"wet"));

    // Verify consistency with query_formula.
    for (atom, node_id) in &formula_atoms {
        let looked_up = prov.query_formula(&atom.predicate, &atom.args).unwrap();
        assert_eq!(looked_up, *node_id);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob --test test_provenance_primitives -- atoms_with_formulas --nocapture`
Expected: FAIL — `atoms_with_formulas` method does not exist yet

**Step 3: Add `atoms_with_formulas()` accessor**

In `crates/xlog-prob/src/provenance.rs`, add to the `impl Provenance` block:

```rust
    pub fn atoms_with_formulas(&self) -> impl Iterator<Item = (&GroundAtom, PirNodeId)> + '_ {
        self.tuple_formulas.iter().map(|(atom, &id)| (atom, id))
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p xlog-prob --test test_provenance_primitives -- --nocapture`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/provenance.rs crates/xlog-prob/tests/test_provenance_primitives.rs
git commit -m "feat(prob): add atoms_with_formulas() accessor on Provenance"
```

---

### Task 6: Add top-level re-exports in `xlog-prob`

**Files:**
- Modify: `crates/xlog-prob/src/lib.rs`

**Step 1: Add re-exports**

In `crates/xlog-prob/src/lib.rs`, after the existing `pub mod` declarations, add:

```rust
pub use pir::{ChoiceVarId, LeafId, PirGraph, PirNode, PirNodeId};
pub use provenance::{ChoiceSource, GroundAtom, Provenance, Value};
```

The full file should read:

```rust
//! Probabilistic reasoning tier for XLOG (Phase 4).

pub mod cnf;
pub mod compilation;
pub mod exact;
pub mod exact_gpu;
pub mod gpu;
pub mod kc;
pub mod mc;
pub mod neural_fast_path;
pub mod pir;
pub mod provenance;
pub mod wfs;
pub mod xgcf;

pub use pir::{ChoiceVarId, LeafId, PirGraph, PirNode, PirNodeId};
pub use provenance::{ChoiceSource, GroundAtom, Provenance, Value};
```

**Step 2: Verify it compiles**

Run: `cargo check -p xlog-prob`
Expected: compiles

**Step 3: Update the test file to use top-level imports**

In `crates/xlog-prob/tests/test_provenance_primitives.rs`, change the imports at the top
from:

```rust
use xlog_prob::pir::LeafId;
use xlog_prob::provenance::{Provenance, Value};
```

and later:

```rust
use xlog_prob::pir::ChoiceVarId;
use xlog_prob::provenance::ChoiceSource;
```

and:

```rust
use xlog_prob::pir::PirNodeId;
```

to a single top-level import block:

```rust
use xlog_prob::{ChoiceSource, ChoiceVarId, GroundAtom, LeafId, PirNodeId, Provenance, Value};
```

Remove the now-redundant `use` statements scattered through the file.

**Step 4: Run all tests to verify nothing broke**

Run: `cargo test -p xlog-prob --test test_provenance_primitives -- --nocapture`
Expected: PASS (all 5 tests)

**Step 5: Verify existing provenance test still works with submodule imports**

Run: `cargo test -p xlog-prob --test provenance_tc -- --nocapture`
Expected: PASS (existing test uses `xlog_prob::pir::*` and `xlog_prob::provenance::*`
which still work — re-exports don't break submodule paths)

**Step 6: Run full workspace tests**

Run: `cargo test --workspace --all-targets --exclude pyxlog --release`
Expected: all pass

**Step 7: Commit**

```bash
git add crates/xlog-prob/src/lib.rs crates/xlog-prob/tests/test_provenance_primitives.rs
git commit -m "feat(prob): add top-level re-exports for provenance and PIR types"
```

---

### Task 7 (Optional): Add `GroundAtom::new()` test

Only if `GroundAtom::new()` was made public in Task 1.

**Files:**
- Modify: `crates/xlog-prob/tests/test_provenance_primitives.rs`

**Step 1: Write the test**

Append to `crates/xlog-prob/tests/test_provenance_primitives.rs`:

```rust
#[test]
fn ground_atom_new_public() {
    let atom = GroundAtom::new("reach", vec![Value::I64(1), Value::I64(3)]);
    assert_eq!(atom.predicate, "reach");
    assert_eq!(atom.args, vec![Value::I64(1), Value::I64(3)]);
}
```

**Step 2: Run test**

Run: `cargo test -p xlog-prob --test test_provenance_primitives -- ground_atom_new --nocapture`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/xlog-prob/tests/test_provenance_primitives.rs
git commit -m "test(prob): verify GroundAtom::new() public constructor"
```
