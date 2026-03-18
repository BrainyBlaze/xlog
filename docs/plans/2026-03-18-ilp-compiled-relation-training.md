# ILP Compiled Relation Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a strict-compatible ILP training API that consumes pre-uploaded tensor-backed relations and DLPack-backed example relations without reconstructing host source facts or host example tuples.

**Architecture:** Keep `train_only(source, ...)` as the existing compile-from-source path. Add persistent ILP-side relation overrides on `CompiledIlpProgram`, plus a new strict compiled-program training entry point that accepts positive/negative example relations as DLPack-backed columns. Reuse the existing strict trainer loop and zero-D2H runtime path; do not route strict callers through compatibility code.

**Tech Stack:** PyO3 Rust bindings, `xlog_runtime::RelationStore`, CUDA/DLPack ingestion, Python strict trainer (`pyxlog.ilp.trainer`), pytest.

### Task 1: Write the compiled-program strict API regressions

**Files:**
- Modify: `python/tests/test_ilp_trainer.py`
- Modify: `python/tests/test_ilp_d2h_gate.py`
- Modify: `python/tests/test_ilp_sparse_guard.py`

**Step 1: Write the failing strict caller-path test**

Add a test that:
- compiles an ILP program from a source template with no inline base facts
- uploads base facts with `CompiledIlpProgram.put_relation(...)`
- passes positive/negative examples as DLPack-backed relation columns
- monkeypatches `torch.Tensor.cpu`, `torch.Tensor.tolist`, and `torch.Tensor.item` to explode
- calls the new compiled-program training API with `strict_gpu_native=True`
- asserts it returns `StrictTrainResult`

**Step 2: Run the test to verify RED**

Run: `python -m pytest python/tests/test_ilp_trainer.py -q -k compiled_relation`

Expected: FAIL because the API does not exist yet.

**Step 3: Write the failing DTS-equivalence regression**

Add a test that:
- builds a source template with no inline facts
- uploads the same `edge` facts via `put_relation(...)`
- trains through the new compiled-program strict API
- trains through the existing `train_only(source_with_inline_facts, ...)` strict path
- asserts both paths complete and expose the same strict attempt-count / total-step shape, with explicit compatibility export either unavailable or explicitly documented

**Step 4: Run the targeted test to verify RED**

Run: `python -m pytest python/tests/test_ilp_trainer.py -q -k dts_or_compiled_relation`

Expected: FAIL because the compiled-program path does not exist yet.

**Step 5: Write the guard regressions**

Add tests that:
- strict compiled-program API rejects `strict_gpu_native=False` if the new API is strict-only
- removed compatibility fallbacks remain unavailable in strict mode
- `CompiledIlpProgram.put_relation(...)` accepts DLPack-backed columns and does not expose host-only helper behavior

**Step 6: Run the guard tests to verify RED**

Run: `python -m pytest python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_d2h_gate.py -q -k compiled_relation`

Expected: FAIL because the new API and guards are not implemented yet.

### Task 2: Add persistent ILP-side relation ingestion

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`
- Modify: `crates/pyxlog/src/ilp.rs`

**Step 1: Add persistent relation override state to `CompiledIlpProgram`**

Store a `RelationStore` or equivalent ILP-side persistent override store alongside the executor so uploaded relations survive `reset_runtime()` and strict reevaluation.

**Step 2: Add `CompiledIlpProgram.put_relation(...)`**

Mirror the logic-session DLPack ingestion path:
- validate relation name
- validate schema against declared ILP schemas
- import DLPack columns with existing zero-copy helpers
- store the buffer in the persistent override store
- clone it into the live executor store

**Step 3: Reapply overrides wherever ILP runtime state is rebuilt**

Update:
- compile initialization if needed
- `reset_runtime()`
- strict reevaluation helpers that rebuild executor state

Use device-side clones only. No source reconstruction.

**Step 4: Run the new `put_relation` tests**

Run: `python -m pytest python/tests/test_ilp_sparse_guard.py -q -k put_relation`

Expected: PASS.

### Task 3: Add relation-native strict example ingestion

**Files:**
- Modify: `crates/pyxlog/src/ilp.rs`

**Step 1: Add a strict GPU loss/grad entry point that accepts example relations as DLPack-backed columns**

Implement a new Rust binding that:
- takes `target_predicate`
- imports positive/negative example relations from DLPack columns
- validates schema against the target relation schema
- performs the same strict sparse evaluation / loss-grad path without materializing example tuples on host

**Step 2: Keep the new path strict-compatible only**

No silent fallback to:
- source reconstruction
- host example lists
- compatibility-only export helpers

**Step 3: Run the new loss/grad example tests**

Run: `python -m pytest python/tests/test_ilp_d2h_gate.py -q -k compiled_relation`

Expected: PASS.

### Task 4: Expose the compiled-program strict trainer entry point

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/__init__.py`

**Step 1: Add the public Python API**

Add a new entry point with a clear strict-native name, for example:

```python
def train_on_compiled_relations(
    prog,
    mask_name: str,
    target_predicate: str,
    positive_columns,
    negative_columns,
    config: TrainConfig = TrainConfig(),
) -> StrictTrainResult:
    ...
```

Use `prog.base_source` only for compatibility metadata where unavoidable; do not reconstruct source facts from uploaded relations.

**Step 2: Keep it strict-only if that keeps the change narrow**

If compatibility mode is unsupported here, raise a clear `IlpConfigError`.

**Step 3: Reuse the existing strict trainer loop**

Wire the new API into the strict compiled-program path, using the new Rust relation-native loss/grad method instead of host example tuples.

**Step 4: Run the new trainer tests**

Run: `python -m pytest python/tests/test_ilp_trainer.py -q -k compiled_relation`

Expected: PASS.

### Task 5: Update evidence and strict-path documentation

**Files:**
- Modify: `docs/evidence/2026-03-17-strict-ilp-gpu-native-audit.md`

**Step 1: Document the new strict-native API**

Record:
- exact API DTS should call
- what is now zero-D2H / zero-host-transfer
- what compatibility-only path still exists
- which APIs are strict-only vs compatibility-only

**Step 2: Record fresh verification commands and outputs**

Include exact command lines and fresh results.

### Task 6: Rebuild and verify

**Files:**
- None

**Step 1: Rebuild native bindings**

Run: `maturin develop --release -m crates/pyxlog/Cargo.toml`

**Step 2: Run the strict ILP test surface**

Run:
- `python -m pytest python/tests/test_ilp_trainer.py -q`
- `python -m pytest python/tests/test_ilp_d2h_gate.py -q`
- `python -m pytest python/tests/test_ilp_sparse_guard.py -q`
- `python -m pytest python/tests/test_ilp_credit_gpu.py -q -x`
- any additional focused suites touched by the new relation-native API

**Step 3: Run a broader touched-surface sanity pass**

Run:
- `python -m pytest python/tests/test_ilp_backend.py python/tests/test_ilp_artifact.py python/tests/test_ilp_reset.py python/tests/test_ilp_multistart.py python/tests/test_ilp_robustness.py -q`

**Step 4: Verify clean outputs and update documentation**

Record the exact outputs in the evidence doc before claiming completion.
