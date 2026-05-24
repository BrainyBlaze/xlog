# Bounded Exact Induction

This document describes XLOG's **bounded exact-induction engine** — the non-gradient,
GPU-native path for enumerating all `(left, right)` candidate pairs across four
fixed 2-body Datalog rule templates and returning top-K per template with full
structured metadata.

Introduced for DTS's M8 Phase 1 as the production replacement for
`pyxlog.ilp.induce_exact(backend="python")`, whose host-orchestrated
`set_rule_mask`/`evaluate`/`batch_fact_membership_device` loop is documented as a
throwaway prototype.

- **Crate:** `crates/xlog-induce/`
- **CUDA kernels:** `kernels/ilp_exact.cu` (module `xlog_ilp_exact`, entries
  `ilp_exact_score` for `U64` and `ilp_exact_score_u32` for `U32` /
  `Symbol`)
- **Provider launcher:** `crates/xlog-cuda/src/provider/ilp_exact.rs`
- **pyxlog bridge:** `crates/pyxlog/src/ilp_exact.rs` +
  `crates/pyxlog/python/pyxlog/ilp/exact_induce.py`
- **Parity tests:** `python/tests/test_ilp_exact_induce.py`
- **Internal kernel design note:** `docs/plans/2026-04-17-m8-ilp-exact-kernel-design.md`
- **Related:** [Differentiable ILP (dILP)](../ROADMAP.md) — the gradient-trained
  counterpart; this engine is exact / bounded and does not use learnable masks.

## Design Goals

1. **Zero host marshalling in the scoring loop.** Setup H2D/D2D uploads are
   permitted but constant-size; the `(topology, L, R)` sweep itself performs
   no host/device round trips. The production `ilp_exact_score_topk` path
   reduces on device and the tracked D2H counter on `CudaKernelProvider`
   ticks exactly **once per `induce_exact` call** for the compact selected-row
   export, independent of candidate count.
2. **Per-topology semantic isolation.** Each `(topology, L, R)` triple is
   scored against the topology's rule template in isolation. There is no
   interference across topologies — a direct fix for the Python prototype's
   stale-mask contamination bug (see [Semantics](#semantics) below).
3. **Deterministic output.** Integer counts only; each kernel block owns one
   unique output slot (no cross-block atomics); host-side reduction is a
   lex sort that is already locked bit-for-bit by 16 unit tests.
4. **Surgical crate boundary.** `xlog-induce` speaks `RelId` +
   `CudaBuffer` handles; name resolution and DLPack unwrapping happen at
   the pyxlog boundary.

## Topologies

The engine scores against four fixed 2-body rule templates. For a head
relation `H(X, Y)` and candidate relations `L` (left body) and `R` (right
body), a query pair `(qx, qy)` is covered by a topology as follows:

| Topology | Rule | Coverage predicate |
|---|---|---|
| `chain`  | `H(X,Y) :- L(X,Z), R(Z,Y)` | ∃ *z* such that `(qx, z) ∈ L ∧ (z, qy) ∈ R` |
| `star`   | `H(X,Y) :- L(X,Y), R(X,Y)` | `(qx, qy) ∈ L ∧ (qx, qy) ∈ R` |
| `fanout` | `H(X,Y) :- L(X,Z), R(X,Y)` | ∃ _ such that `(qx, _) ∈ L ∧ (qx, qy) ∈ R` |
| `fanin`  | `H(X,Y) :- L(X,Y), R(Z,Y)` | `(qx, qy) ∈ L ∧ ∃ _ such that `(_, qy) ∈ R` |

The kernel implements these predicates directly — it does **not** go
through `evaluate()` / `set_rule_mask`. This is what guarantees per-topology
isolation.

## Architecture

```
Python user
    │
    ▼
pyxlog.ilp.induce_exact(prog, backend="native", ...)
    │   (name resolution, DLPack unwrap, empty-negatives synthesis)
    ▼
CompiledIlpProgram::induce_exact_native  (crates/pyxlog/src/ilp_exact.rs)
    │   (build InduceExactRequest with RelId + &CudaBuffer handles)
    ▼
xlog_induce::induce_exact(provider, &request)  (crates/xlog-induce/src/lib.rs)
    │
    ├──► validate::classify_request   (empty cands / zero positives → empty result)
    │
    ▼
CudaKernelProvider::ilp_exact_score_topk(...)  (crates/xlog-cuda/src/provider/ilp_exact.rs)
    │
    ├── Setup (not D2H-counted):
    │     - D2D concat candidate arg0/arg1 columns
    │     - H2D upload cand_offsets (small prefix-sum array)
    │     - Alloc output pos_covered/neg_covered arrays
    │
    ├── Launch typed kernel  (kernels/ilp_exact.cu)
    │     U64    → `ilp_exact_score`
    │     U32    → `ilp_exact_score_u32`
    │     Symbol → `ilp_exact_score_u32` with logical schema preserved
    │     grid  = (C, C, 4)       — one block per (L, R, topology)
    │     block = (256, 1, 1)
    │     Each block writes exactly one slot in pos_covered and neg_covered.
    │
    ├── Device top-K selection over pos_covered / neg_covered
    │
    └── Download (1 D2H, tracked):
          compact selected top-K rows as u32 field tuples
    │
    ▼
xlog_induce::reduce_per_topology  (host-side, deterministic lex sort)
    │
    ▼
ExactInductionResult { candidates: Vec<ScoredCandidate>, total_scored, ... }
    │
    ▼
pyxlog wrapper repackages into Python ExactInductionResult / ScoredCandidate
```

## Launch Geometry

- **Grid dims:** `(C, C, 4)` blocks addressed as `(L, R, topology)`, where
  `C` is the number of candidate relations in the request and topology
  enumeration is `chain=0, star=1, fanout=2, fanin=3`.
- **Block dims:** `(256, 1, 1)` threads. Each block cooperatively scans all
  positive and negative query pairs for its single `(topology, L, R)`
  triple.
- **Output slot:** `slot = topology * (C * C) + L * C + R`. Each block
  writes exactly one slot in each of `pos_covered` and `neg_covered`, so
  there are no cross-block atomics on the scoring path. Shared-memory
  pair-halving reduction is used *within* a block to sum per-thread
  coverage counts.

At DTS-scale sizes (C ≈ 5–20, |P|+|N| ≈ 20–50) the launch is modest
(≤ 1 600 blocks × 256 threads) but ample for occupancy. The inner
per-thread work is O(|L|·|R|) for `chain` and O(|L|+|R|) for the other
three topologies — microseconds per block in practice.

## Data Layout

| Buffer | Type / length | Contents |
|---|---|---|
| `cand_arg0`, `cand_arg1` | `u64` or `u32` × total_rows | Concatenated (D2D-copied) arg0 / arg1 columns of all candidate relations |
| `cand_offsets` | `u32` × (C+1) | Exclusive prefix-sum of candidate row counts; H2D uploaded once per call |
| `pos_arg0`, `pos_arg1` | `u64` or `u32` × num_pos | Device-resident positive query pairs (DLPack-imported from the caller's torch tensors) |
| `neg_arg0`, `neg_arg1` | `u64` or `u32` × num_neg | Same for negatives. When the caller passes no negatives, the engine materializes a zero-row pair buffer with the positive pair type so the kernel signature stays uniform. |
| `pos_covered`, `neg_covered` | `u32` × (4·C·C) | Output count arrays; kernel writes each slot exactly once. |

Column type dispatch is explicit at both native boundaries:

- `xlog_induce::induce_exact` validates that positives, negatives, and every
  candidate buffer are arity-2 pairs with one uniform logical type: `U64`,
  `U32`, or `Symbol`.
- `CudaKernelProvider::ilp_exact_score` repeats that validation at the provider
  seam before choosing the physical kernel.
- `U64` buffers launch `ilp_exact_score` over `uint64_t` columns.
- `U32` buffers launch `ilp_exact_score_u32` over `uint32_t` columns.
- `Symbol` buffers also launch `ilp_exact_score_u32`, but the `CudaBuffer`
  schema remains `Symbol` and mixed `U32`/`Symbol` requests fail with typed
  diagnostics rather than silently narrowing.

The pyxlog DLPack layer accepts physical 32-bit tensor IDs for `symbol`
schemas only through schema-checked import. This preserves public logical
types while allowing CUDA-resident symbol-id tensors to reach the native scorer
without host widening.

## Semantics

The Python prototype's `backend="python"` loop calls `set_rule_mask(mask_name,
...)` **per inner iteration**, updating only the current topology's mask
while leaving the other three topologies' masks at whatever state they
last held. Because `evaluate()` derives facts from *all* currently-set
masks, stale masks from earlier outer-loop iterations leak derivations
into later topologies' coverage counts.

The native kernel bypasses `evaluate()` and `set_rule_mask` entirely:
it implements the four topology predicates directly on the candidate
row sets. Per-topology isolation is a structural property, not a
runtime hygiene requirement.

The Python prototype gained an opt-in `strict_per_topology: bool = False`
parameter that zeroes the three "other" topology masks before each
topology's inner loop, yielding the same per-topology-isolated scoring
that the kernel produces by construction. **Default is `False`** for
backward compatibility with DTS Phase 0 callers that are calibrated
against the historical prototype numbers. The parity test sets
`strict_per_topology=True` explicitly to match the kernel.

## D2H Budget

The kernel is designed around the xlog-native D2H transfer counter
(`CudaKernelProvider::d2h_transfer_count`, exposed to Python as
`prog.d2h_transfer_count()`).

- **Counted transfers per production `induce_exact` call: 1** (one compact
  selected-row export) regardless of candidate count, query count, or topology count. The
  parity test `test_induce_exact_native_does_not_scale_d2h_with_candidate_pairs`
  enforces `large.d2h_transfer_count ≤ small.d2h_transfer_count + 2`,
  which passes trivially.
- Setup H2D (`cand_offsets`) and D2D (candidate column concatenation)
  are not D2H-counted.
- Setup reads that would otherwise require a D2H (e.g. relation row
  counts) go through the host-side `CudaBuffer::cached_row_count()`
  cache. This is a load-bearing invariant — see the next section.

### `clone_buffer` cached-row-count propagation

Every relation buffer in `CompiledIlpProgram`'s executor store was
deep-copied via `CudaKernelProvider::clone_buffer` on insertion
(`CompiledIlpProgram::put_relation` clones `live_buffer` before
`executor.put_relation`). The previous `clone_buffer` implementation
built the clone via `CudaBuffer::from_columns`, which does **not**
populate the host-side `cached_row_count`. Downstream consumers that
needed the row count were forced to either D2H-read `num_rows_device()`
or trust the (misleading) `num_rows()` capacity accessor.

`clone_buffer` now calls `set_cached_row_count_if_unset(source.cached_row_count())`
on the clone when the source has a populated cache, preserving the
host-visible count across clones. Pinned by
`test_clone_buffer_preserves_cached_row_count`.

## Validation Contract

Buffer-level invariants enforced by `xlog_induce::induce_exact`:

| Check | Location | Failure mode |
|---|---|---|
| `candidates.is_empty()` | Engine top of `induce_exact` | Returns default `ExactInductionResult` (all zeros) — matches Python reference's `if not body_indices: return …` |
| Buffer arity == 2 | `validate_pair_buffer` | `XlogError::Execution` |
| Column type is exactly one of `U64`, `U32`, `Symbol` | `validate_pair_buffer` | `XlogError::Type` |
| Positive, negative, and candidate pair types match exactly | `require_pair_type` / provider layout check | `XlogError::Type` / `XlogError::Kernel` |
| `cached_row_count()` populated | `cached_rows` helper | `XlogError::Execution` — caller fed a buffer that skipped the DLPack ingest path |
| `positive_count == 0` | `classify_request` (pure) | Returns result with zero candidates but retained neg/candidate counts |

Trivial dead-end classification (`classify_request`) is pure and
unit-tested without CUDA via 5 tests in `xlog-induce/src/validate.rs`.

## Public Types

```rust
// crates/xlog-induce/src/lib.rs
pub struct InduceExactRequest<'a> {
    pub head_rel_idx: RelId,
    pub candidates: &'a [(RelId, &'a CudaBuffer)],
    pub positives: &'a CudaBuffer,
    pub negatives: Option<&'a CudaBuffer>,
    pub config: ExactInductionConfig,
}

pub fn induce_exact(
    provider: &CudaKernelProvider,
    request: &InduceExactRequest<'_>,
) -> Result<ExactInductionResult>;
```

```rust
// crates/xlog-induce/src/types.rs
pub enum Topology { Chain, Star, Fanout, Fanin }

pub struct ExactInductionConfig {
    pub k_per_topology: u32,
    pub deterministic: bool,  // reserved — engine is inherently deterministic
}

pub struct ScoredCandidate {
    pub topology: Topology,
    pub head_rel_idx: RelId,
    pub left_rel_idx: RelId,
    pub right_rel_idx: RelId,
    pub positives_covered: u32,
    pub negatives_covered: u32,
    pub local_rank: u32,
    pub next_positives_covered: u32,
    pub next_negatives_covered: u32,
    pub tie_class_size: u32,
}

pub struct ExactInductionResult {
    pub candidates: Vec<ScoredCandidate>,
    pub total_scored: u32,
    pub candidate_count: u32,
    pub positive_count: u32,
    pub negative_count: u32,
}
```

## Python Surface

```python
from pyxlog.ilp import induce_exact

result = induce_exact(
    prog,                                   # CompiledIlpProgram
    head_relation="p_A",
    candidate_relations=["p_B", "p_C", "p_D"],
    positive_arg0=pos_a0,                   # 1-D device torch tensor
    positive_arg1=pos_a1,
    negative_arg0=neg_a0,                   # optional
    negative_arg1=neg_a1,
    k_per_topology=2,
    deterministic=True,
    backend="native",                       # or "python" (prototype)
    # strict_per_topology=True,             # only meaningful for backend="python"
)
```

Returns an `ExactInductionResult` dataclass with `candidates: list[ScoredCandidate]`.

## Testing

- **Host-only unit tests** (`cargo test -p xlog-induce --lib`):
  23 tests — 16 reduce-layer, 5 classify_request, 2 topology string/order.
  No CUDA required.
- **CUDA-gated launcher tests** (`cargo test -p xlog-cuda --lib ilp_exact`):
  `ilp_exact_score_matches_hand_computed_fixture` (hand-derived coverage
  against C=2 candidates), `ilp_exact_score_is_deterministic_across_runs`,
  `ilp_exact_score_handles_empty_negatives`,
  `ilp_exact_score_accepts_u32_pair_buffers`,
  `ilp_exact_score_accepts_symbol_pair_buffers`, and
  `ilp_exact_score_rejects_mixed_pair_types`. Skip with `eprintln!` when no
  CUDA device is present.
- **Python parity test** (`python -m pytest python/tests/test_ilp_exact_induce.py`):
  `test_induce_exact_native_matches_python_reference` and
  `test_induce_exact_native_does_not_scale_d2h_with_candidate_pairs`.
  Uses `strict_per_topology=True` against the Python reference so both
  backends compute clean per-topology coverage.
- **v0.8.6 typed parity tests**
  (`python -m pytest python/tests/test_v086_exact_types_runtime.py`):
  `U32` and `Symbol` fixtures match the Python reference, preserve relation
  type annotations, keep D2H at exactly two count-array transfers, and reject
  mixed logical pair types.

## Type Dispatch And Packaging Policy

- `U64`: supported through `ilp_exact_score`.
- `U32`: supported through `ilp_exact_score_u32`.
- `Symbol`: supported through the same physical `u32` kernel while retaining
  `Symbol` schema identity and rejecting mixed `U32`/`Symbol` requests.
- Chain topology shared-memory caching: supported through
  `ilp_exact_score_chain_smem` and `ilp_exact_score_chain_smem_u32` when
  `XLOG_ILP_EXACT_CHAIN_SMEM` is enabled and the candidate-row threshold is
  met. Non-chain topologies inside those entry points continue to call the
  baseline matcher.
- PTX policy: `kernels/ilp_exact.cu` is checked in; generated
  `ilp_exact.portable.ptx` and architecture-specific `.cubin` files are build
  and packaging artifacts, not source artifacts. `crates/xlog-cuda/build.rs`
  generates portable PTX for every manifest module, including `ilp_exact`;
  `scripts/stage_pyxlog_kernels.sh` stages those artifacts into
  `pyxlog/kernels/`; `scripts/install_pyxlog_for_python.py` rejects a wheel
  that lacks portable PTX. This matches the current ILP-family convention for
  `ilp.cu` and `ilp_credit.cu`.

## Profile-Gated Chain Shared Memory

The v0.8.6 G086_CHAIN_SMEM node adds an A/B-controlled chain scorer that tiles
left-relation rows into dynamic shared memory for the strict chain topology.
It is enabled by default only for candidate relations with at least
`XLOG_ILP_EXACT_CHAIN_SMEM_MIN_ROWS` rows, defaulting to `256`, and can be
disabled with `XLOG_ILP_EXACT_CHAIN_SMEM=0`.

The optimization preserves the public exact-induction contract:

- chain-smem and baseline dispatch produce identical coverage signatures on
  the certified small and chain-hot fixtures;
- median runtime improves by more than `1.2x` on the certified chain-hot
  fixture;
- small fixtures do not regress by more than five percent;
- the D2H budget remains the same two count-array transfers used by the
  baseline native exact-induction path.

Evidence: `docs/evidence/2026-05-19-v086-chain-smem/`.

## See Also

- `docs/plans/2026-04-17-m8-ilp-exact-kernel-design.md` — the original
  internal kernel design note (launch geometry, data layout, D2H accounting).
- `docs/architecture/cuda-certification.md` — CUDA certification suite
  (C01–C25 + G01–G08); `ilp_exact` is not yet in the formal certification
  registry because its PTX is not committed (see above).
- `ROADMAP.md` → "Bounded Exact Induction (`xlog-induce`) — DTS M8
  Phase 1" — milestone-level status, planned DTS-side integration.
