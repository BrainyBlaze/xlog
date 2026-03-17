# Persistent Python Relation Store API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose a persistent Python relation-store API that lets callers register, replace, remove, clear, evaluate, and export named GPU relations through DLPack without recompiling on every call.

**Architecture:** Add an explicit stateful logic-session surface instead of mutating the existing stateless `CompiledLogicProgram.evaluate(dlpack_inputs=...)` contract. The session owns a persistent base relation store, validates schema/arity on import, reuses the existing runtime executor for evaluation, and exports relations back through the existing zero-copy DLPack table path.

**Tech Stack:** Rust (`xlog-gpu`, `xlog-runtime`, `pyxlog`), PyO3, DLPack, Python `pytest`, PyTorch CUDA tensors

### Task 1: Add the failing Python integration test for a persistent relation session

**Files:**
- Create: `python/tests/test_logic_relation_store.py`
- Test: `python/tests/test_logic_relation_store.py`

**Step 1: Write the failing test**

Add tests that define the intended Python API:
- `CompiledLogicProgram.session()` returns a persistent session object
- `session.put_relation(name, [dlpack columns])` stores or replaces a named relation
- `session.evaluate()` reuses the stored relations
- `session.export_relation(name)` returns DLPack columns
- `session.remove_relation(name)` and `session.clear_relations()` update the store
- schema mismatches fail explicitly

Use a minimal deterministic logic program such as:

```python
SOURCE = """
pred edge(u32, u32).
pred reach(u32, u32).
reach(X, Y) :- edge(X, Y).
?- reach(X, Y).
"""
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest python/tests/test_logic_relation_store.py -v`
Expected: FAIL because `CompiledLogicProgram.session()` and the session methods do not exist yet.

**Step 3: Write minimal implementation**

No production code in this task.

**Step 4: Re-run the focused test**

Run: `python -m pytest python/tests/test_logic_relation_store.py -v`
Expected: still FAIL for the missing session API.

**Step 5: Commit**

```bash
git add python/tests/test_logic_relation_store.py
git commit -m "test(pyxlog): define persistent logic relation-store API"
```

### Task 2: Add a reusable stateful logic-session core in Rust

**Files:**
- Modify: `crates/xlog-gpu/src/logic.rs`
- Test: `python/tests/test_logic_relation_store.py`

**Step 1: Write the failing test**

Reuse the test from Task 1. Do not add production code until the failure is observed.

**Step 2: Run test to verify it fails**

Run: `python -m pytest python/tests/test_logic_relation_store.py -v`
Expected: FAIL on the missing session creation/evaluation path.

**Step 3: Write minimal implementation**

In `crates/xlog-gpu/src/logic.rs`:
- add a stateful `LogicSession` type or equivalent helper owned by `pyxlog`
- initialize persistent base relations from compiled program facts
- provide helpers to:
  - create a fresh executor from the stored base relations
  - run the compiled plan against that executor
  - move persistent relations back into the session after evaluation if needed
- keep the existing stateless `evaluate(...)` API unchanged

Prefer extracting common setup from `evaluate_with_options(...)` instead of duplicating fact-loading logic.

**Step 4: Re-run the focused test**

Run: `python -m pytest python/tests/test_logic_relation_store.py -v`
Expected: FAIL moves forward to the missing PyO3 surface or individual session methods, not the core runtime path.

**Step 5: Commit**

```bash
git add crates/xlog-gpu/src/logic.rs python/tests/test_logic_relation_store.py
git commit -m "feat(xlog-gpu): add persistent logic session core"
```

### Task 3: Expose the session and DLPack relation-store methods through PyO3

**Files:**
- Modify: `crates/pyxlog/src/lib.rs`
- Modify: `crates/pyxlog/src/logic.rs`
- Test: `python/tests/test_logic_relation_store.py`

**Step 1: Write the failing test**

Extend the test expectations only if needed after observing the prior failure:
- `CompiledLogicProgram.session()`
- `LogicRelationSession.put_relation(...)`
- `LogicRelationSession.export_relation(...)`
- `LogicRelationSession.remove_relation(...)`
- `LogicRelationSession.clear_relations()`
- `LogicRelationSession.evaluate()`

**Step 2: Run test to verify it fails**

Run: `python -m pytest python/tests/test_logic_relation_store.py -v`
Expected: FAIL on the missing PyO3 registration or missing methods.

**Step 3: Write minimal implementation**

In `crates/pyxlog/src/lib.rs`:
- add a new `#[pyclass]` for the persistent logic session
- register it in the module

In `crates/pyxlog/src/logic.rs`:
- add `CompiledLogicProgram.session()`
- implement DLPack-backed `put_relation`
- use `from_dlpack_tensors_with_schema(...)` for schema/type checking
- implement `export_relation` with `to_dlpack_table(...)`
- implement `remove_relation` and `clear_relations`
- keep return shapes simple and aligned with existing logic query DLPack patterns

**Step 4: Re-run the focused test**

Run: `python -m pytest python/tests/test_logic_relation_store.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/pyxlog/src/lib.rs crates/pyxlog/src/logic.rs python/tests/test_logic_relation_store.py
git commit -m "feat(pyxlog): expose persistent DLPack relation-store session"
```

### Task 4: Cover replacement, clearing, and explicit schema-failure cases

**Files:**
- Modify: `python/tests/test_logic_relation_store.py`
- Test: `python/tests/test_logic_relation_store.py`

**Step 1: Write the failing test**

Add focused tests for:
- replacing an existing relation updates the next evaluation result
- clearing relations removes previously stored data
- exporting an unknown relation fails clearly
- wrong arity / wrong dtype input fails clearly

**Step 2: Run test to verify it fails**

Run: `python -m pytest python/tests/test_logic_relation_store.py -v`
Expected: FAIL on the new edge-case assertions.

**Step 3: Write minimal implementation**

Adjust the session validation and error messages in:
- `crates/pyxlog/src/logic.rs`
- `crates/xlog-gpu/src/logic.rs` only if runtime support is required

**Step 4: Re-run the focused test**

Run: `python -m pytest python/tests/test_logic_relation_store.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/pyxlog/src/logic.rs crates/xlog-gpu/src/logic.rs python/tests/test_logic_relation_store.py
git commit -m "test(pyxlog): cover persistent relation replacement and validation"
```

### Task 5: Update Python binding documentation

**Files:**
- Modify: `docs/architecture/python-bindings.md`
- Test: `python/tests/test_logic_relation_store.py`

**Step 1: Write the failing test**

No new executable test. Treat the already-passing integration test as the contract anchor.

**Step 2: Verify current docs are stale**

Run: `rg -n "dlpack_inputs|LogicProgram" docs/architecture/python-bindings.md`
Expected: only transient `evaluate(dlpack_inputs=...)` usage is documented.

**Step 3: Write minimal implementation**

Add a short documented example for:
- `program.session()`
- `put_relation(...)`
- `evaluate()`
- `export_relation(...)`

Keep the existing stateless `evaluate(dlpack_inputs=...)` example so the API addition is clearly additive.

**Step 4: Run focused verification**

Run: `python -m pytest python/tests/test_logic_relation_store.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/architecture/python-bindings.md
git commit -m "docs(pyxlog): document persistent relation-store session"
```
