# P2a: Term Embeddings — Design

**Date:** 2026-03-08
**Status:** Implemented (2026-03-08, branch feat/p2a-term-embeddings)
**Depends on:** v0.5.0-phase1 (tagged 2026-03-06, commit 407d8bab)

## Goal

Enable embedding predicates (`labels.is_none()`) to produce tensors for
explicit PyTorch-side training. The AST already parses
`nn(encoder, [X], Embedding) :: embed(X, Embedding).` but no execution path
handles the `labels.is_none()` case. P2a adds three capabilities:
`register_embedding()` for payload registration with cross-registration
validation, `forward_embedding()` for batched tensor lookup with autograd
support, and compile-time form validation for network names. Inference
through rules (dot/cosine evaluation, grounded query API) is deferred to
v0.5.1+.

## Non-goals

- GPU-native embedding weights or CUDA lookup kernels (v0.6.0+)
- Implicit loss from rule success/failure (requires tensor-aware rule engine)
- Foreign tensor predicates (`dot`, `cosine`, etc.) in rule evaluation
  (requires arithmetic function lowering in lower.rs + executor.rs; v0.5.1+)
- Grounded-facts `query()` API for embedding inference (no such API exists
  today; v0.5.1+)
- Integration with P2b optimizer/scheduler APIs
- Artifact persistence for embedding modules
- Symbol/string lookup keys (integer IDs only)
- Train/eval mode switching for embeddings (user-managed in v0.5)

## Deferred: Embedding Inference (v0.5.1+)

The inference path — rules like `similar(X, Y) :- embed(X, EX), embed(Y, EY),
Score is dot(EX, EY), Score > 0.8` evaluated through `query()` — is deferred.
It requires three features that do not exist today:

1. **Arithmetic function lowering:** `dot(EX, EY)` parses as
   `ArithExpr::FuncCall` (parser.rs:1074) but lowering rejects all function
   calls (lower.rs:1868) and the executor handles only lowered `Expr::*`
   (executor.rs:2048). Making `dot`/`cosine` work in `is` expressions
   requires special-casing these as built-in arithmetic functions across
   the lowerer and executor.
2. **Grounded-facts query API:** Current `CompiledProgram` exposes
   `evaluate_query_probabilities()` (pyxlog/src/lib.rs:3017) which returns
   probabilities, not variable bindings. A `query()` method returning
   grounded bindings is a new API surface.
3. **Embedding cache in rule evaluation:** The rule engine must resolve
   `embed(X, EX)` by calling the embedding module and caching results
   per-query, keyed on `(decl_name, entity_id)`.

These are bounded but separate from P2a's training scope.

---

## Section 1: Registration API

### `register_embedding`

Separate method on `CompiledProgram`, distinct from `register_network`:

```python
program.register_embedding(
    name="entity_embed",        # must match nn() declaration name
    module_or_tensor=module,    # nn.Embedding or torch.Tensor
    trainable=True,             # only valid for nn.Embedding
)
```

### Accepted payloads

| Payload | Trainable | Gradient flow |
|---------|-----------|---------------|
| `nn.Embedding` | Yes | Through `.weight` via autograd |
| `torch.Tensor` | No (enforced) | None — frozen lookup |

If `trainable=True` is passed with a raw `torch.Tensor`, raise a hard error.

### Cross-registration validation

Cross-registration requires a by-network index. Today,
`NeuralPredicateRegistry` (neural_registry.rs:18) is indexed by predicate
name, and `CompiledProgram` stores only a `HashSet<String>` of declared
network names (lib.rs:596). Neither provides a network-name-to-form lookup.

**Compile-time rule:** Every network name must map to exactly one form
(classification or embedding) across all `nn()` declarations. Mixed use of
the same network name — some declarations with labels, some without — is a
compile error. Error: "network '{name}' is declared as both classification
and embedding; each network name must have a single form."

**Implementation:** At compile time, build a `HashMap<String, bool>` mapping
each declared network name to whether it is an embedding
(`labels.is_none()`). Store this alongside `declared_networks` in
`CompiledProgram`. Iterate all `nn()` declarations; if a name appears with
conflicting forms, reject before compilation proceeds.

Validation at registration:
- If the declaration has labels, reject `register_embedding`. Error:
  "declaration '{name}' is a classification network; use register_network()".
- If the declaration lacks labels, reject `register_network`. Error:
  "declaration '{name}' is an embedding; use register_embedding()".

### Registration-time checks

1. Declared name exists in program's neural predicate declarations.
2. Declaration form matches (embedding vs classification) via by-network index.
3. Payload shape: `nn.Embedding` — `.weight` is 2D `[vocab_size, dim]`, float
   dtype. `torch.Tensor` — rank 2, float dtype.

### Optimizer ownership

User-managed. `register_embedding` stores no optimizer or scheduler. The user
creates their own optimizer over `embedding.parameters()` and calls `.step()`
in their training loop. P2b's `get_lr`/`set_lr`/`scheduler_step` do not cover
embeddings in v0.5.

---

## Section 2: Rust-Side Handle

### `EmbeddingHandle`

New struct in `crates/xlog-neural/src/handle.rs`:

```rust
pub struct EmbeddingHandle {
    pub name: String,
    #[cfg(feature = "python")]
    pub module: Option<PyObject>,  // nn.Embedding or tensor
    pub trainable: bool,
    pub dim: usize,                // embedding dimension
    pub vocab_size: usize,         // number of entries
}
```

### Registry integration

`NetworkRegistry` gains a parallel `HashMap<String, EmbeddingHandle>` for
embeddings. Methods: `register_embedding`, `get_embedding`,
`get_embedding_mut`, `contains_embedding`.

Train/eval mode switching for embeddings is **not part of v0.5**. The user
manages module state directly. Future versions may integrate embeddings into
the registry's `set_train_mode()` if needed.

---

## Section 3: Forward Embedding API

### `forward_embedding()`

New method on `CompiledProgram`. This is a standalone API, separate from
the existing `forward_backward_*` classification path. It does not hook
into the circuit-backed forward/backward pipeline.

```python
results = program.forward_embedding("entity_embed", [0, 5, 42])
# results -> torch.Tensor [3, dim]
# grad-enabled if nn.Embedding; no grad if frozen torch.Tensor
```

- Accepts: embedding declaration name + list of integer IDs.
- Returns: batched `torch.Tensor`.
- For `nn.Embedding`: calls `module(torch.tensor(ids))`. Autograd graph
  intact. User computes loss and calls `loss.backward()`.
- For `torch.Tensor`: indexes `tensor[ids]`. No gradient flow.

### Lookup semantics

The method calls the embedding module directly — no embedding cache, no
rule evaluation, no interaction with the logic engine. The cache design
(keyed on `(decl_name, entity_id)`) is deferred to the inference path.

### Training workflow

```python
embedding = torch.nn.Embedding(100, 64)
optimizer = torch.optim.Adam(embedding.parameters())

program.register_embedding("entity_embed", embedding, trainable=True)

# Training loop (user-managed)
for batch_ids, targets in dataloader:
    optimizer.zero_grad()
    vectors = program.forward_embedding("entity_embed", batch_ids.tolist())
    loss = my_loss_fn(vectors, targets)
    loss.backward()
    optimizer.step()
```

---

## Section 4: Testing

Five tests cover the contract surface:

| # | Test | Verifies |
|---|------|----------|
| 1 | `register_embedding` with `nn.Embedding` | `forward_embedding` returns correct shape `[n, dim]` and expected values |
| 2 | `register_embedding` with frozen `torch.Tensor` | Returns correct values; `trainable=True` with raw tensor is rejected |
| 3 | Cross-registration errors (both directions) | Embedding decl + `register_network` raises error; labeled decl + `register_embedding` raises error |
| 4 | Gradient flow | `forward_embedding` then external loss then `.backward()` then `optimizer.step()` changes `nn.Embedding` weights |
| 5 | Frozen output is non-trainable | `forward_embedding` on frozen tensor returns tensor with `requires_grad=False` |

All tests in `python/tests/test_embeddings.py`.

---

## Section 5: Files

### New

- `python/tests/test_embeddings.py` — 5 tests

### Modified

- `crates/xlog-neural/src/handle.rs` — add `EmbeddingHandle` struct
- `crates/xlog-neural/src/lib.rs` — re-export `EmbeddingHandle`
- `crates/xlog-neural/src/registry.rs` — add embedding storage and methods
- `crates/pyxlog/src/lib.rs` — by-network form index at compile time,
  `register_embedding`, `forward_embedding`, cross-registration validation
  in `register_network`

---

## Design Decisions Log

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Scope: training-only in v0.5, no inference | dot/cosine require lowerer+executor changes; grounded query API does not exist; both deferred to v0.5.1+ |
| D2 | Separate `register_embedding`, not unified with `register_network` | Different parameters; cross-registration validation catches real mistakes |
| D3 | `nn.Embedding` + `torch.Tensor` payloads; raw tensors always frozen | `nn.Embedding` owns autograd; raw tensors have no optimizer owner |
| D4 | By-network form index for cross-registration | `NeuralPredicateRegistry` is by-predicate; need network-name-to-form lookup; mixed-form same-name rejected at compile time |
| D5 | External loss only; no implicit loss from rule success | `.item()` detaches from autograd; implicit loss requires tensor-aware rule engine |
| D6 | `forward_embedding` is standalone, not in forward_backward_* | forward_backward_* is circuit-backed classification; embeddings have no circuit path in v0.5 |
| D7 | Integer IDs only for lookup keys | `nn.Embedding` takes integers natively; symbol interning is a runtime internal |
| D8 | User-managed optimizer | Keeps P2a scoped to execution, not optimizer orchestration; P2b APIs do not cover embeddings |
| D9 | No train/eval mode switching in v0.5 | User manages module state directly; registry integration deferred |
