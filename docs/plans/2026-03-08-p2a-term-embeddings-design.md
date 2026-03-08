# P2a: Term Embeddings — Design

**Date:** 2026-03-08
**Status:** Approved
**Depends on:** v0.5.0-phase1 (tagged 2026-03-06, commit 407d8bab)

## Goal

Enable embedding predicates (`labels.is_none()`) to execute. The AST already
parses `nn(encoder, [X], Embedding) :: embed(X, Embedding).` but the
forward/backward path only handles classification (softmax over a label list).
P2a adds the execution path for embeddings: registration, forward dispatch,
inference through rules, and explicit training via Python.

## Non-goals

- GPU-native embedding weights or CUDA lookup kernels (v0.6.0+)
- Implicit loss from rule success/failure (requires tensor-aware rule engine)
- Full foreign tensor predicate set (only `dot`/`cosine` in v0.5)
- Integration with P2b optimizer/scheduler APIs
- Artifact persistence for embedding modules
- Symbol/string lookup keys (integer IDs only)

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

- If the declaration has labels (`labels.is_some()`), reject
  `register_embedding`. Error: "declaration '{name}' is a classification
  network; use register_network()".
- If the declaration lacks labels (`labels.is_none()`), reject
  `register_network`. Error: "declaration '{name}' is an embedding; use
  register_embedding()".

### Registration-time checks

1. Declared name exists in program's neural predicate declarations.
2. Declaration form matches (embedding vs classification).
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
`get_embedding_mut`, `contains_embedding`. Train/inference mode switching
applies to trainable embeddings (calls `module.train()` / `module.eval()`
on the PyTorch side).

---

## Section 3: Forward Dispatch

### Per-query embedding cache

When the rule engine resolves `embed(X, EX)`, the embedding module produces
a tensor. A per-query cache stores these tensors to avoid redundant lookups:

- **Key:** `(embedding_decl_name: &str, entity_id: u32)`
- **Value:** `torch.Tensor` of shape `[dim]`
- **Lifetime:** one query (forward or forward/backward)
- **Behavior:** memoized. Repeated `embed(X, EX)` for the same entity reuses
  the same tensor node in the autograd graph.

Variable names are not used as keys — they are rule-local and unstable across
joins. Predicate name + entity ID is the stable identity.

### Dispatch branch

In `forward_backward_*` (pyxlog/src/lib.rs), when encountering a neural
predicate with `labels.is_none()`:

1. Look up `EmbeddingHandle` by name (not `NetworkHandle`).
2. Call `module(torch.tensor([entity_id]))` for `nn.Embedding`, or
   `tensor[entity_id]` for raw `torch.Tensor`.
3. Store result in the embedding cache.
4. Return the cached tensor for downstream consumption.

### Score semantics

`dot(EX, EY, Score)` and `cosine(EX, EY, Score)` produce normal f64 scalar
terms bound into the relation. The value enters the arithmetic/comparison
pipeline like any other f64.

The scalar is extracted via `.item()` (detached from autograd graph) for logic
evaluation. The underlying tensor computation is retained in the graph for the
training path.

---

## Section 4: Two Execution Modes

### Inference: `query()`

Embedding predicates work inside rules:

```prolog
similar(X, Y) :-
    embed(X, EX),
    embed(Y, EY),
    Score is dot(EX, EY),
    Score > 0.8.
```

```python
results = program.query("similar(X, Y)")
# Returns grounded facts only. No tensor exposure.
```

The rule engine resolves embeddings internally: cache lookup, tensor op,
`.item()`, scalar binding, comparison. Tensors never appear in query results.

### Training: `forward_embedding()`

New method on `CompiledProgram`:

```python
results = program.forward_embedding("entity_embed", [0, 5, 42])
# results -> torch.Tensor [3, dim], grad-enabled for nn.Embedding
```

- Accepts: embedding declaration name + list of integer IDs.
- Returns: batched `torch.Tensor` with autograd graph intact.
- The user computes loss in Python and calls `loss.backward()`.

This is the only gradient-carrying embedding API in v0.5.

### Boundary rule

If a query path uses embedding-derived values only inside logic comparisons
and never exposes tensor outputs to Python, it is inference-only. No gradient
semantics. Normal query results never expose tensor-valued variables.

---

## Section 5: Tensor Predicates (Inference Path)

### `dot(EX, EY, Score)`

Computes `torch.dot(EX, EY).item()` -> f64. Both EX and EY must be in the
embedding cache. If either is missing, the predicate fails (no derivation).

### `cosine(EX, EY, Score)`

Computes `torch.cosine_similarity(EX, EY, dim=0).item()` -> f64. Same cache
contract as `dot`.

### Implementation

These predicates dispatch as foreign predicates in the rule engine. They pull
tensors from the embedding cache by variable binding, compute, and return a
scalar. They do not appear in the AST as neural predicates — they are built-in
arithmetic operators over cached embedding values.

---

## Section 6: Testing

Five tests cover the contract surface:

| # | Test | Verifies |
|---|------|----------|
| 1 | `register_embedding` with `nn.Embedding` | Forward produces correct dim vector via `forward_embedding` |
| 2 | `register_embedding` with frozen `torch.Tensor` | Forward produces correct values; `trainable=True` with tensor is rejected |
| 3 | `dot` / `cosine` produce correct scalars | Two embeddings produce correct dot product and cosine similarity |
| 4 | Cross-registration errors (both directions) | Embedding decl + `register_network` raises error; labeled decl + `register_embedding` raises error |
| 5 | Gradient flow | `forward_embedding` then external loss then `.backward()` updates `nn.Embedding` weights |

All tests in `python/tests/test_embeddings.py`.

---

## Section 7: Files

### New

- `python/tests/test_embeddings.py` — 5 tests

### Modified

- `crates/xlog-neural/src/handle.rs` — add `EmbeddingHandle` struct
- `crates/xlog-neural/src/registry.rs` — add embedding storage and methods
- `crates/pyxlog/src/lib.rs` — `register_embedding`, `forward_embedding`,
  embedding dispatch in forward/backward, `dot`/`cosine` foreign predicates

---

## Design Decisions Log

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Scope A: PyTorch-side execution, no CUDA kernels | Smallest change that enables embeddings; defers GPU-native to v0.6.0+ |
| D2 | Separate `register_embedding`, not unified with `register_network` | Different parameters; cross-registration validation catches real mistakes |
| D3 | `nn.Embedding` + `torch.Tensor` payloads; raw tensors always frozen | `nn.Embedding` owns autograd; raw tensors have no optimizer owner |
| D4 | Per-query embedding cache keyed on `(decl_name, entity_id)` | Preserves scalar unification model; memoizes lookups; stable identity |
| D5 | Score is normal f64 scalar, not soft confidence | `dot` returns a value, not a probability; arithmetic needs a real binding |
| D6 | External loss only; no implicit loss from rule success | `.item()` detaches from autograd; implicit loss requires tensor-aware rule engine |
| D7 | Separate `forward_embedding` training API | Keeps `query()` results scalar-only; training API returns live tensors |
| D8 | Integer IDs only for lookup keys | `nn.Embedding` takes integers natively; symbol interning is a runtime internal |
| D9 | User-managed optimizer | Keeps P2a scoped to execution, not optimizer orchestration; P2b APIs do not cover embeddings |
| D10 | Inference path in v0.5 (query with embedding rules) | Core value proposition: similar(X, Y) rules in the logic language |
| D11 | `dot` and `cosine` only; other tensor ops deferred | Covers similarity and KG embedding use cases; minimal dispatch surface |
