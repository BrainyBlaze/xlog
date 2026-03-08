# xlog Change Request: Provenance Primitives

> **Supersedes** the earlier version of this document which asked xlog to implement
> `derivation_tree()`, `certificate()`, and DTS-specific output types.
> Redesigned as a minimal provenance-primitive ask.
> This request is self-contained.

## Ownership Boundary

- **xlog** exposes retained provenance primitives (accessor methods over data already computed during extraction).
- **DTS** owns derivation-tree reconstruction and certificate construction, using xlog's existing public PIR graph traversal APIs.

## DTS Use Case

DTS Phase 1 reconstruct stage traces how each committed fact was derived. This requires resolving PIR leaf nodes and decision nodes back to their source atoms and rules. The PIR graph traversal APIs are already public — only the semantic mappings are missing.

Rule induction via `pyxlog.ilp.train_only()` already works (real CUDA roundtrip). Only provenance primitives are blocked.

## Requested Changes (4 required + 1 optional ergonomic addition)

### 1. `GroundAtom::new()` — optional ergonomic addition

`GroundAtom` is already a public type with public fields, so DTS can construct it with a struct literal today. `query_formula()` also takes `predicate, args` directly. This means a public constructor is not a dependency for the provenance-primitives API.

If xlog maintainers want the ergonomic improvement and a cleaner public constructor surface, the existing constructor can be made public:

```rust
pub fn new(predicate: impl Into<String>, args: Vec<Value>) -> Self
```

This is optional. The core provenance request does not depend on it.

### 2. `Provenance::atoms_with_formulas()` — new accessor

Expose the set of ground atoms that currently have provenance formulas, paired with their PIR root
node.

```rust
pub fn atoms_with_formulas(&self) -> /* accessor over (&GroundAtom, PirNodeId) pairs */
```

Semantics:

- one entry per atom currently stored in `tuple_formulas`
- includes all atoms with provenance formulas, including base facts
- order unspecified
- concrete return type at maintainer discretion

This is an accessor over existing data, not a new provenance computation.

### 3. `Provenance::leaf_atom()` — new accessor + retained mapping

During extraction (`extract_from_program`, lines 323-335), each `LeafId` is created alongside a `GroundAtom`, but only the probability is stored in `leaf_probs`. The atom is used to populate the internal `store` and then discarded.

Ask: retain a `BTreeMap<LeafId, GroundAtom>` during extraction and expose it.

```rust
pub fn leaf_atom(&self, leaf: LeafId) -> Option<&GroundAtom>
```

Extraction change: after `leaf_probs.insert(leaf, pf.prob)`, also insert into the new map.

### 4. `Provenance::choice_source()` — new accessor + retained mapping

During annotated-disjunction extraction, each `ChoiceVarId` is created as one stage in the ordered
lowering chain for an annotated disjunction. DTS needs to resolve each `PirNode::Decision { var, ... }`
back to:

- the enclosing annotated disjunction choices and probabilities
- the exact decision position for that `ChoiceVarId` within the lowering chain

```rust
pub fn choice_source(&self, var: ChoiceVarId) -> Option<&ChoiceSource>
```

Logical retained mapping:

- `ChoiceVarId -> ChoiceSource`

This is enough for DTS to reconstruct decision nodes precisely during PIR traversal.

### 5. `ChoiceSource` — new type

```rust
pub struct ChoiceSource {
    pub choices: Vec<(GroundAtom, f64)>,
    /// Position of this ChoiceVarId in the annotated-disjunction lowering chain.
    ///
    /// For an annotated disjunction with m heads, xlog currently lowers to an
    /// ordered chain of m - 1 Bernoulli choice variables. `choice_index`
    /// identifies which decision stage this ChoiceVarId represents.
    pub choice_index: usize,
    /// v1 semantics:
    /// - `None` is acceptable and expected if no stable source id is retained.
    /// - If xlog later chooses to retain one, it should define it explicitly,
    ///   e.g. as the annotated-disjunction ordinal in `program.annotated_disjunctions`.
    pub source_id: Option<usize>,
}
```

Notes:

- `choices` is preferred over parallel atoms / probabilities vectors because it avoids a public
  parallel-vector invariant.
- `choice_index` is required for precise reconstruction of `PirNode::Decision { var, ... }`.
- `source_id` identifies the enclosing annotated disjunction if retained; `choice_index`
  identifies the exact decision stage within that disjunction.
- `source_id` is optional in `v1`. DTS does not require a non-`None` source id to proceed.

## Storage / Retention Impact

Two retained provenance mappings are needed:

| Field | Logical Role | Entry Count |
|-------|--------------|-------------|
| `leaf_atoms` | `LeafId -> GroundAtom` | Same as `leaf_probs` |
| `choice_sources` | `ChoiceVarId -> ChoiceSource` | Same as `choice_probs` |

Important nuance:

- `leaf_atoms` is a straight one-to-one retention of data already available during extraction.
- `choice_sources` has one accessor-visible entry per `ChoiceVarId`, matching `choice_probs`.
- For an annotated disjunction with `m` heads, xlog currently emits `m - 1` choice variables.
- A naive implementation would therefore duplicate the same annotated-disjunction metadata across
  `m - 1` entries, differing only in `choice_index`.
- That duplication is acceptable for this minimal API, but the accessor does not require a naive
  representation. xlog may deduplicate internally and preserve the same public accessor behavior.

No new provenance computation is requested. This is metadata retention plus narrow accessors.

## DTS-Side Traversal Example

Shows why each accessor is needed:

```rust
fn reconstruct(prov: &Provenance, predicate: &str, args: &[Value]) -> Option<Derivation> {
    // Find PIR root for this atom — EXISTING API
    let root_id = prov.query_formula(predicate, args)?;
    // Walk PIR graph — EXISTING API (PirGraph::node())
    walk_node(&prov.pir, prov, root_id)
}

fn walk_node(graph: &PirGraph, prov: &Provenance, id: PirNodeId) -> Option<Derivation> {
    match graph.node(id)? {
        PirNode::Lit { leaf } => {
            // Resolve leaf to source atom — NEW (item 3)
            let atom = prov.leaf_atom(*leaf)?;
            Some(Derivation::leaf(atom.clone()))
        }
        PirNode::Decision { var, child_true, child_false } => {
            // Resolve choice to rule/disjunction — NEW (item 4)
            let source = prov.choice_source(*var)?;
            let t = walk_node(graph, prov, *child_true);
            let f = walk_node(graph, prov, *child_false);
            Some(Derivation::choice(source, t, f))
        }
        PirNode::And { children } => {
            let sub: Vec<_> = children.iter()
                .filter_map(|c| walk_node(graph, prov, *c))
                .collect();
            Some(Derivation::conjunction(sub))
        }
        // Or, NegLit, Const handled similarly
    }
}

// Enumerate all atoms with formulas — NEW (item 2)
for (atom, root_id) in prov.atoms_with_formulas() {
    println!("{:?} -> PIR root {}", atom, root_id.as_u32());
}

// Construct a query atom — struct literal (public fields, no dependency on item 1)
let atom = GroundAtom { predicate: "reach".into(), args: vec![Value::I64(1), Value::I64(3)] };
```

## Summary

| Item | xlog Effort |
|------|-------------|
| `GroundAtom::new()` public | Optional ergonomic addition, not a required provenance primitive |
| `atoms_with_formulas()` accessor | Small — iterate `tuple_formulas` |
| `leaf_atom()` + retained map | Small — one insert per prob fact during extraction |
| `choice_source()` + retained map | Small — one insert per choice var during extraction |
| `ChoiceSource` type (`choices`, `choice_index`, `source_id`) | Small |
