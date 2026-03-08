# Provenance Primitives — Design

**Date:** 2026-03-08
**Status:** Approved
**Change Request:** `docs/xlog-change-request-provenance.md`

## Goal

Expose retained provenance metadata so an external Rust consumer (DTS) can
resolve PIR leaf nodes and decision nodes back to their source atoms and
annotated-disjunction metadata. All data is already computed during
extraction — this design retains it and adds narrow read-only accessors.

## Non-goals

- Python bindings (Rust-only in v1)
- Populating `source_id` on `ChoiceSource` (stays `None` in v1)
- Post-hoc reconstruction or extra extraction passes
- Internal deduplication of `ChoiceSource` across `ChoiceVarId`s
- Inference-path changes or new computation

---

## Section 1: New Type — `ChoiceSource`

```rust
// In crates/xlog-prob/src/provenance.rs

#[derive(Debug, Clone, PartialEq)]
pub struct ChoiceSource {
    /// Explicit heads of the annotated disjunction, paired with probabilities.
    /// Does not include the synthetic implicit "none" branch.
    pub choices: Vec<(GroundAtom, f64)>,
    /// Position of this ChoiceVarId in the m-1 Bernoulli decision chain.
    pub choice_index: usize,
    /// Enclosing annotated-disjunction identity. `None` in v1.
    /// If populated later: AD ordinal in program.annotated_disjunctions,
    /// stable within one extraction, not a cross-program persistent id.
    pub source_id: Option<usize>,
}
```

Fields are public. No constructor — struct literal construction.

---

## Section 2: Provenance Struct Changes

Two new public fields, mirroring existing `leaf_probs` / `choice_probs`:

```rust
pub struct Provenance {
    // existing — unchanged
    pub pir: PirGraph,
    pub leaf_probs: BTreeMap<LeafId, f64>,
    pub choice_probs: BTreeMap<ChoiceVarId, (f64, f64)>,
    tuple_formulas: BTreeMap<GroundAtom, PirNodeId>,  // stays private
    pub queries: Vec<GroundAtom>,
    pub evidence: Vec<(GroundAtom, bool)>,

    // new
    pub leaf_atoms: BTreeMap<LeafId, GroundAtom>,
    pub choice_sources: BTreeMap<ChoiceVarId, ChoiceSource>,
}
```

Entry counts: `leaf_atoms` has the same cardinality as `leaf_probs`;
`choice_sources` has the same cardinality as `choice_probs` (naive
duplication per `ChoiceVarId`, acceptable for small `m`).

---

## Section 3: Accessors

Three new methods on `Provenance`, plus making `GroundAtom::new()` public:

```rust
impl GroundAtom {
    // existing private fn → make pub
    pub fn new(predicate: impl Into<String>, args: Vec<Value>) -> Self { ... }
}

impl Provenance {
    // existing
    pub fn query_formula(&self, predicate: &str, args: &[Value]) -> Option<PirNodeId> { ... }

    // new: iterate all atoms with provenance formulas (base facts + derived)
    pub fn atoms_with_formulas(&self) -> impl Iterator<Item = (&GroundAtom, PirNodeId)> + '_ {
        self.tuple_formulas.iter().map(|(atom, &id)| (atom, id))
    }

    // new: resolve leaf to source atom
    pub fn leaf_atom(&self, leaf: LeafId) -> Option<&GroundAtom> {
        self.leaf_atoms.get(&leaf)
    }

    // new: resolve choice var to AD metadata
    pub fn choice_source(&self, var: ChoiceVarId) -> Option<&ChoiceSource> {
        self.choice_sources.get(&var)
    }
}
```

`atoms_with_formulas()` exposes `tuple_formulas` through a read-only
iterator without making the field public.

---

## Section 4: Extraction Retention

Inline retention at existing allocation sites in `extract_from_program()`.
No new passes, no post-hoc reconstruction.

**Site 1: Probabilistic facts (leaf atoms)**

```rust
// existing
let leaf = LeafId::new(next_leaf_id);
leaf_probs.insert(leaf, pf.prob);

// new — retain the atom alongside its probability
leaf_atoms.insert(leaf, atom.clone());
```

**Site 2: Annotated disjunctions (choice sources)**

Inside the m-1 loop over Bernoulli decision stages in
`compile_annotated_disjunction()`:

```rust
// existing
let var = ChoiceVarId::new(next_choice_id);
choice_probs.insert(var, (p_true, p_false));

// new — retain AD metadata per choice var
choice_sources.insert(var, ChoiceSource {
    choices: choices.clone(),   // explicit heads only, no implicit none
    choice_index: i,
    source_id: None,            // v1: not populated
});
```

Naive duplication of `choices` across `m-1` entries is acceptable.

---

## Section 5: Re-exports

`xlog-prob`'s `lib.rs` currently declares modules only — no top-level
`pub use` re-exports exist. Add a re-export surface so DTS imports from
`xlog_prob::{...}` without submodule coupling:

```rust
// crates/xlog-prob/src/lib.rs — new re-export surface

pub use pir::{ChoiceVarId, LeafId, PirGraph, PirNode, PirNodeId};
pub use provenance::{ChoiceSource, GroundAtom, Provenance, Value};
```

Covers existing public types DTS needs plus the new `ChoiceSource`.

---

## Section 6: Testing

Integration tests in `crates/xlog-prob/tests/test_provenance_primitives.rs`
(exercises public API only):

| # | Test | Required | Verifies |
|---|------|----------|----------|
| 1 | `atoms_with_formulas_returns_all_formula_atoms` | Yes | Iterator yields every atom (base facts + derived) with correct `PirNodeId` |
| 2 | `leaf_atom_resolves_prob_facts` | Yes | Each `LeafId` maps to correct `GroundAtom` |
| 3 | `choice_source_resolves_ad_metadata` | Yes | Explicit heads only, correct `choice_index`, `source_id: None` |
| 4 | `leaf_atom_returns_none_for_invalid` | Yes | `leaf_atom(LeafId::new(9999))` returns `None` |
| 5 | `choice_source_returns_none_for_invalid` | Yes | `choice_source(ChoiceVarId::new(9999))` returns `None` |
| 6 | `ground_atom_new_public` | Optional | Only if `GroundAtom::new()` is made public |

Tests construct programs with known prob facts and annotated disjunctions,
extract provenance, and assert against expected mappings.

---

## Section 7: Files

| File | Action |
|------|--------|
| `crates/xlog-prob/src/provenance.rs` | Add `ChoiceSource` type, two new fields on `Provenance`, three accessors, make `GroundAtom::new()` pub |
| `crates/xlog-prob/src/lib.rs` | Add top-level re-export surface for provenance + PIR types |
| `crates/xlog-prob/tests/test_provenance_primitives.rs` | New — 5 required + 1 optional integration test |

No other crates modified. No Python bindings. No breaking changes.

---

## Design Decisions Log

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Rust-only, no Python bindings | DTS is a Rust consumer; Python bindings add surface area for no current user |
| D2 | `source_id: None` in v1 | `choice_index` alone suffices for DTS reconstruction; avoids premature semantic commitment on AD ordinal identity |
| D3 | Naive duplication in `choice_sources` | `m` is small (2-5 heads per AD); dedup adds complexity for negligible memory savings (YAGNI) |
| D4 | Inline retention at allocation sites | Data is available during extraction; post-hoc reconstruction would reverse-engineer discarded data |
| D5 | `GroundAtom::new()` public — optional | Ergonomic sugar; fields are already public so struct literal works; not a required provenance primitive |
| D6 | `atoms_with_formulas()` returns iterator, not ref to map | Preserves `tuple_formulas` encapsulation; read-only access through narrow surface |
| D7 | `choices` excludes implicit none branch | Only explicit AD heads; synthetic none branch is an internal lowering detail |
| D8 | Top-level re-exports in xlog-prob | DTS should import from `xlog_prob::{...}` not reach into submodules |
