# Strict GPU-Native Substrate Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the last two strict-mode host escapes in xlog so the strict Python ILP training/runtime path is genuinely GPU-native end-to-end, with explicit compatibility boundaries instead of hidden host fallbacks.

**Architecture:** Fix this in two layers. First, replace the current host-backed strict sparse mask update substrate with a device-backed runtime mask path so `set_rule_mask_sparse_selected_device(...)` never downloads selected candidate IDs to host. Second, split strict result finalization from compatibility artifact export so strict training no longer auto-materializes argmax, rule strings, logits, soft probabilities, or ranking summaries on host. Compatibility export must become an explicit, gated boundary.

**Tech Stack:** Rust (`pyxlog`, `xlog-runtime`, `xlog-cuda`, PyO3), Python (`pyxlog.ilp`, PyTorch), `pytest`, `cargo test`, `maturin`.

## Confirmed Remaining Gaps

1. Strict sparse mask updates still materialize selected candidate IDs on host in [crates/pyxlog/src/ilp.rs](/home/dev/projects/xlog/crates/pyxlog/src/ilp.rs#L1189).
2. Strict sparse masks are still represented as host `Vec<(u32, u32, u32)>` in [crates/xlog-runtime/src/ilp_registry.rs](/home/dev/projects/xlog/crates/xlog-runtime/src/ilp_registry.rs#L21) and consumed from host entries in [crates/xlog-runtime/src/executor/node_dispatch.rs](/home/dev/projects/xlog/crates/xlog-runtime/src/executor/node_dispatch.rs#L445).
3. Strict trainer finalization still materializes host data in [_strict_selected_hard](/home/dev/projects/xlog/crates/pyxlog/python/pyxlog/ilp/trainer.py#L830), [_strict_witness_coverage](/home/dev/projects/xlog/crates/pyxlog/python/pyxlog/ilp/trainer.py#L840), [_finalize_strict_attempt](/home/dev/projects/xlog/crates/pyxlog/python/pyxlog/ilp/trainer.py#L859), [_fill_metrics](/home/dev/projects/xlog/crates/pyxlog/python/pyxlog/ilp/trainer.py#L1157), and sparse `decode_argmax()` in [crates/pyxlog/python/pyxlog/ilp/backend.py](/home/dev/projects/xlog/crates/pyxlog/python/pyxlog/ilp/backend.py#L240).
4. The current strict `TrainResult` / `LearnedArtifact` contract is host-shaped by construction in [crates/pyxlog/python/pyxlog/ilp/types.py](/home/dev/projects/xlog/crates/pyxlog/python/pyxlog/ilp/types.py#L127) and [crates/pyxlog/python/pyxlog/ilp/types.py](/home/dev/projects/xlog/crates/pyxlog/python/pyxlog/ilp/types.py#L276), and is assembled unconditionally in [crates/pyxlog/python/pyxlog/ilp/trainer.py](/home/dev/projects/xlog/crates/pyxlog/python/pyxlog/ilp/trainer.py#L1243).

## Recommended Approach

### Recommended: Device-backed sparse masks plus explicit compatibility export

Implement a new strict-only sparse mask substrate in the runtime, then make strict trainer finalization return a strict/device result shape that does not auto-export host summaries. Add an explicit compatibility export method or helper for callers that want JSON/list artifacts.

Why this is the right approach:
- It removes the actual substrate cause rather than hiding the current DTOH call.
- It gives strict mode a real invariant: no implicit host materialization.
- It leaves compatibility workflows available, but only behind explicit, testable gates.

### Rejected shortcut 1: Only patch `ilp.rs:1189`

Not sufficient. Even if the DTOH download moves out of `set_rule_mask_sparse_selected_device(...)`, the runtime still stores sparse masks as host tuples today. That would just relocate the same escape.

### Rejected shortcut 2: Keep strict return values host-shaped because “the hot loop is already clean”

Not acceptable. The user already called out this exact loophole. Hidden host result packaging after the loop is still a strict-path escape.

## Full Definition of Done

### DoD A: Strict Sparse Mask Substrate

- `set_rule_mask_sparse_selected_device(...)` performs zero host materialization when `strict_zero_dtoh=True`.
- No call path reachable from strict sparse mask updates may invoke provider host-download helpers such as `download_column_untracked`, `download_column`, or equivalent DTOH helpers.
- Sparse strict masks are no longer represented only as host `Vec<(u32, u32, u32)>`.
- The runtime has a device-backed strict sparse mask representation and an insertion path that accepts device-selected candidate IDs or device-gathered `(i,j,k)` rows without downloading them to host first.
- The executor can consume the strict sparse representation without a compatibility fallback that silently host-materializes selected candidates.
- If a compatibility-only sparse mask path remains, it is explicitly gated out of strict mode with a deterministic error mentioning the compatibility API by name.

### DoD B: Strict Trainer Finalization

- `train_only(..., strict_gpu_native=True)` does not call `.cpu()`, `.item()`, `.tolist()`, or `.numpy()` anywhere in the strict success path, including finalization and returned result construction.
- Strict finalization does not decode argmax to a Python `int` by default.
- Strict finalization does not auto-build `discovered_rule` strings by default.
- Strict finalization does not auto-populate host `logits`, `soft_probs`, `selected_hard`, `confidence_margin`, or `top_k_concentration`.
- Any host-shaped summary or artifact export is explicit and named as such, for example `export_compat_artifact(...)`, `materialize_host_summary(...)`, or equivalent.
- The default strict return type must be visibly strict-safe: either a separate strict result/artifact type, or the existing type with strict-only fields populated and compatibility fields left unset / empty by design.

### DoD C: Compatibility Boundaries

- `strict_gpu_native=False` preserves current compatibility behavior unless intentionally tightened and documented.
- `LearnedArtifact.save()` and any JSON serialization remain valid for compatibility artifacts.
- Strict artifacts/results cannot be silently serialized into compatibility JSON. They must either:
  - raise with a clear error that explicit compatibility export is required, or
  - require a separate explicit export method that performs host materialization outside strict mode.
- `holdout.py` and `promoter.py` remain hard-gated out of strict mode unless they are fully rewritten to be strict-safe in this task. Do not loosen their current strict rejection.

### DoD D: Verification and Evidence

- All new tests are written before the corresponding implementation change.
- Native changes rebuild cleanly with:
  - `maturin develop --release -m crates/pyxlog/Cargo.toml`
- Fresh verification includes, at minimum:
  - `python -m pytest python/tests/test_ilp_sparse.py python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_types.py -q`
  - `python -m pytest python/tests/test_ilp_d2h_gate.py -q`
  - `python -m pytest python/tests/test_ilp_device_queries.py -q -x`
  - `python -m pytest python/tests/test_ilp_credit_gpu.py -q -x`
  - `python -m pytest python/tests/test_ilp_holdout.py -q -x`
  - `python -m pytest python/tests/test_ilp_promoter.py -q -x`
  - `python -m pytest python/tests/test_ilp_trainer.py -q`
- If Rust runtime tests are added for the new sparse mask substrate, run the exact targeted `cargo test` commands for those modules and record fresh results.
- The evidence doc must be updated with issue, root cause, fix, strict vs compatibility classification, verification commands, and exact results.

## Anti-Shit / Anti-Gaming Gates

These are acceptance gates, not suggestions.

### Gate 1: No hidden host materialization in strict sparse mask updates

- Behavioral gate: strict sparse selected-device updates must succeed with the strict host-materialization guard enabled.
- Structural gate: all host downloads in `crates/pyxlog/src/ilp.rs` must go through a small named wrapper that enforces `strict_zero_dtoh`.
- Code review gate: no raw provider download helper is allowed in the strict sparse selected-device call graph.
- Static belt-and-suspenders gate: add a source-level regression test that the body of `set_rule_mask_sparse_selected_device(...)` contains no `download_` calls after the refactor.

### Gate 2: No Python host-sync primitives in strict finalization

- Behavioral gate: add a strict trainer test that monkeypatches or otherwise traps Python-side `.cpu()`, `.item()`, `.tolist()`, and `.numpy()` usage in the strict success path. The strict training run must still pass.
- Structural gate: compatibility-only materialization must live in clearly named helpers such as `_materialize_compat_metrics(...)` or `_export_compat_artifact(...)`. Strict helpers may not call them.
- Static belt-and-suspenders gate: add source-level tests that strict helper bodies do not contain `.cpu(`, `.item(`, `.tolist(`, or `.numpy(`.

### Gate 3: No fake compliance by dropping fields after materializing them

- Behavioral gate: strict result construction must pass even when Python host-sync primitives are trapped. This prevents “materialize then discard” cheating.
- Structural gate: `TrainResult` assembly for strict mode must not call the compatibility artifact builder at all.
- API gate: strict result fields that require host export must be absent, `None`, or empty by contract, not computed and then hidden.

### Gate 4: No fake compliance by moving work into a hidden compatibility path

- Strict mode must not silently flip to `strict_gpu_native=False`, disable `strict_zero_dtoh`, or route through compatibility APIs.
- Add explicit negative tests that strict mode rejects any remaining compatibility-only export or serialization entry points.
- Any compatibility-only fallback must raise a strict-mode error naming the blocked API and the explicit compatibility alternative.

### Gate 5: No “counter says zero” loophole

- Do not rely only on `d2h_transfer_count()` if some host reads are untracked today.
- The implementation must either:
  - make all host materialization helpers used by ILP strict paths strict-aware and fail closed, or
  - track them in the strict accounting path so the regression tests can catch them.
- Acceptance is based on behavior and strict guard enforcement, not on a partially scoped counter.

## Task 1: Write the failing strict sparse substrate tests

**Files:**
- Modify: `python/tests/test_ilp_sparse_guard.py`
- Modify: `python/tests/test_ilp_d2h_gate.py`
- Modify: `crates/pyxlog/src/ilp.rs`

**Step 1: Write the failing Python regression tests**

Add tests that cover:
- strict selected-device sparse mask update through the public API
- strict training path using sparse backend after the substrate change
- strict rejection of any remaining compatibility-only sparse update path

Example test shape:

```python
def test_strict_selected_device_sparse_path_performs_no_host_materialization():
    prog = pyxlog.IlpProgramFactory.compile(REACH_SOURCE, device=0, memory_mb=64)
    prog.set_strict_zero_dtoh(True)
    prog.set_candidate_map([(i_edge, i_edge, k_reach)])
    ids = torch.tensor([0], device="cuda", dtype=torch.int64)
    soft = torch.tensor([1.0], device="cuda", dtype=torch.float64)
    prog.set_rule_mask_sparse_selected_device("W_reach", ids, soft, False)
    prog.evaluate()
```

**Step 2: Add the failing strict guard around ILP host materialization helpers**

Create or extend a small helper in `crates/pyxlog/src/ilp.rs` so strict mode fails closed on host downloads used by ILP APIs. The new test should fail before the substrate refactor is complete.

**Step 3: Run the targeted tests to verify RED**

Run:
- `python -m pytest python/tests/test_ilp_sparse_guard.py -q`
- `python -m pytest python/tests/test_ilp_d2h_gate.py -q -k sparse`

Expected:
- at least one new strict sparse test fails because the current implementation still materializes selected IDs on host

## Task 2: Implement the device-backed strict sparse mask substrate

**Files:**
- Modify: `crates/pyxlog/src/ilp.rs`
- Modify: `crates/pyxlog/src/lib.rs`
- Modify: `crates/xlog-runtime/src/ilp_registry.rs`
- Modify: `crates/xlog-runtime/src/executor/node_dispatch.rs`
- Modify: `crates/xlog-cuda/src/provider/relational.rs` or another provider module if a new gather helper is needed
- Test: `python/tests/test_ilp_sparse_guard.py`

**Step 1: Add a strict sparse device mask representation**

Introduce a runtime mask variant for strict sparse masks, for example:

```rust
pub enum IlpMask {
    Dense { hard: CudaBuffer, soft: CudaBuffer, schema_size: usize },
    Sparse { active_entries: Vec<(u32, u32, u32)>, schema_size: usize },
    SparseDevice { active_entries_ijk: CudaBuffer, schema_size: usize, count: usize },
}
```

Do not bikeshed the exact field names. The point is that the strict variant stays device-backed.

**Step 2: Keep candidate order on device**

When `set_candidate_map(...)` is called from Python, keep enough device-side data to gather selected `(i,j,k)` rows from selected candidate IDs without downloading IDs to host first.

**Step 3: Insert strict sparse masks without DTOH**

Refactor `set_rule_mask_sparse_selected_device(...)` so it:
- imports selected IDs and soft probabilities from DLPack
- resolves selected rows on device
- inserts a device-backed sparse mask into the runtime

**Step 4: Teach the executor to consume the new strict sparse mask**

The executor must not force a silent conversion of `SparseDevice` back to host tuples in strict mode. If one narrow host bridge remains truly unavoidable, stop and re-scope before merging. The acceptance target for this task is zero host materialization for the strict sparse update path.

**Step 5: Run the sparse strict tests to verify GREEN**

Run:
- `python -m pytest python/tests/test_ilp_sparse.py python/tests/test_ilp_sparse_guard.py -q`
- `python -m pytest python/tests/test_ilp_d2h_gate.py -q -k sparse`

Expected:
- all targeted sparse and strict D2H tests pass

## Task 3: Write the failing strict finalization tests

**Files:**
- Modify: `python/tests/test_ilp_trainer.py`
- Modify: `python/tests/test_ilp_types.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/backend.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/types.py`

**Step 1: Add a behavioral trap test for Python host-sync primitives**

Add a strict training test that traps Python-side host materialization calls. The strict path must pass without touching them.

Example test shape:

```python
def test_train_only_strict_gpu_native_avoids_python_host_materialization(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("host materialization called")

    monkeypatch.setattr(torch.Tensor, "cpu", _boom, raising=False)
    monkeypatch.setattr(torch.Tensor, "item", _boom, raising=False)
    monkeypatch.setattr(torch.Tensor, "tolist", _boom, raising=False)

    result = train_only(... strict config ...)
    assert result is not None
```

If monkeypatching `torch.Tensor` directly is too blunt, trap the narrow trainer/backend helpers instead. The test must still be behavioral, not just source inspection.

**Step 2: Add strict result contract tests**

Add tests that assert strict mode does not auto-populate compatibility-only fields, for example:

```python
def test_train_only_strict_result_does_not_auto_materialize_compat_artifact():
    result = train_only(... strict config ...)
    assert result.discovered_rule is None
    assert result.artifact.logits == []
    assert result.artifact.soft_probs == []
    assert result.artifact.selected_hard == []
```

Adjust exact expectations if you introduce a separate strict result type, but keep the contract explicit and testable.

**Step 3: Run the targeted tests to verify RED**

Run:
- `python -m pytest python/tests/test_ilp_trainer.py -q -k strict`
- `python -m pytest python/tests/test_ilp_types.py -q`

Expected:
- at least one new strict finalization test fails before the refactor

## Task 4: Split strict result construction from compatibility export

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/backend.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/types.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/promoter.py` if needed for explicit compatibility export usage
- Test: `python/tests/test_ilp_trainer.py`
- Test: `python/tests/test_ilp_types.py`

**Step 1: Remove Python host materialization from strict helpers**

Strict helpers must stop calling:
- `_strict_selected_hard(...)`
- `_strict_witness_coverage(...)` if it returns host floats via `.cpu()`
- `_fill_metrics(...)`
- `SparseMaskBackend.decode_argmax()` if it returns `W.argmax().item()`

Either delete the strict-only host helpers or move them behind explicit compatibility export helpers.

**Step 2: Introduce an explicit compatibility export boundary**

One acceptable shape:

```python
@dataclass
class StrictLearnedArtifact:
    candidate_map: list[CandidateMapEntry]
    top_candidate_id: torch.Tensor | None = None
    cand_probs: torch.Tensor | None = None
    logits: torch.Tensor | None = None

def export_compat_artifact(strict_artifact: StrictLearnedArtifact, ... ) -> LearnedArtifact:
    ...
```

Another acceptable shape:
- reuse `TrainResult`, but keep compatibility-only fields unset in strict mode and require an explicit export method

Either shape is fine. Hidden host export is not.

**Step 3: Make discovered rule decoding explicit**

Strict mode must not auto-build a rule string from candidate IDs. If a caller wants a human-readable discovered rule, that should come from the explicit compatibility export or a named host-materialization method.

**Step 4: Preserve compatibility mode**

`strict_gpu_native=False` must still populate the current host-facing `TrainResult` / `LearnedArtifact` contract unless an intentional API migration is documented and tested.

**Step 5: Run the targeted tests to verify GREEN**

Run:
- `python -m pytest python/tests/test_ilp_trainer.py -q`
- `python -m pytest python/tests/test_ilp_types.py -q`
- `python -m pytest python/tests/test_ilp_holdout.py -q -x`
- `python -m pytest python/tests/test_ilp_promoter.py -q -x`

Expected:
- all touched trainer/type/holdout/promoter tests pass

## Task 5: Rebuild, run full touched verification, and update evidence

**Files:**
- Modify: `docs/evidence/2026-03-17-strict-ilp-gpu-native-audit.md`

**Step 1: Rebuild native bindings**

Run:
- `maturin develop --release -m crates/pyxlog/Cargo.toml`

Expected:
- successful rebuild and editable install

**Step 2: Run fresh verification**

Run:
- `python -m pytest python/tests/test_ilp_sparse.py python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_types.py -q`
- `python -m pytest python/tests/test_ilp_d2h_gate.py -q`
- `python -m pytest python/tests/test_ilp_device_queries.py -q -x`
- `python -m pytest python/tests/test_ilp_credit_gpu.py -q -x`
- `python -m pytest python/tests/test_ilp_holdout.py -q -x`
- `python -m pytest python/tests/test_ilp_promoter.py -q -x`
- `python -m pytest python/tests/test_ilp_trainer.py -q`

If Rust unit tests were added:
- run the exact targeted `cargo test` commands for those new tests

**Step 3: Update the evidence doc**

Replace the current residual-risk section only if the gaps are actually gone. The evidence doc must record:
- issue
- root cause
- fix
- strict vs compatibility classification
- verification commands and fresh results

**Step 4: Commit**

```bash
git add crates/pyxlog/src/ilp.rs \
        crates/pyxlog/src/lib.rs \
        crates/xlog-runtime/src/ilp_registry.rs \
        crates/xlog-runtime/src/executor/node_dispatch.rs \
        crates/pyxlog/python/pyxlog/ilp/backend.py \
        crates/pyxlog/python/pyxlog/ilp/trainer.py \
        crates/pyxlog/python/pyxlog/ilp/types.py \
        python/tests/test_ilp_sparse.py \
        python/tests/test_ilp_sparse_guard.py \
        python/tests/test_ilp_d2h_gate.py \
        python/tests/test_ilp_trainer.py \
        python/tests/test_ilp_types.py \
        docs/evidence/2026-03-17-strict-ilp-gpu-native-audit.md \
        docs/plans/2026-03-17-strict-gpu-native-substrate-closure.md
git commit -m "fix(ilp): close remaining strict GPU-native substrate gaps"
```

## Explicit Non-Goals

- Do not rewrite holdout or promoter to be strict-safe in this task unless all hidden host paths are fully removed.
- Do not add CPU workarounds.
- Do not weaken or delete strict regression tests.
- Do not “solve” the task by reclassifying strict finalization as out-of-scope.
- Do not rely on comments or docs as substitutes for code changes.

## Handoff Summary

If the implementation is correct, the final state should be:

- strict sparse mask selection remains device-backed all the way through the update substrate
- strict training returns without implicit Python host materialization
- compatibility export remains available, but only explicitly and outside strict mode
- the existing evidence doc no longer lists either of these two gaps as residual risks
