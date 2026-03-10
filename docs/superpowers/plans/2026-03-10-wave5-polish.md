# Wave 5: Polish

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address low-priority cosmetic improvements opportunistically.

**Architecture:** No structural changes. Documentation, type hints, and minor idiom improvements.

**Tech Stack:** Rust, Python. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-10-codebase-refactoring-design.md` (L1–L9)

**Prerequisite:** None. These can be done at any time, ideally when touching nearby code.

---

### Task 1: Review `allow(dead_code)` suppressions (L1)

- [ ] Search for all `#[allow(dead_code)]` in the workspace
- [ ] Verify each is still justified (the code may have become used since the suppression was added)
- [ ] Remove any that are no longer needed

### Task 2: Python type hints (L2)

- [ ] Add type hints to internal functions in `crates/pyxlog/python/pyxlog/ilp/`
- [ ] Focus on function signatures, not every local variable

### Task 3: Python docstrings (L3)

- [ ] Add docstrings to backend classes in `ilp/`
- [ ] Focus on public-facing methods

### Task 4: Test assertion specificity (L4)

- [ ] Review test assertions for overly broad checks (e.g., `assert!(result.is_ok())` instead of checking the actual value)
- [ ] Improve where the assertion message would help debugging

### Task 5: Feature flag documentation (L5)

- [ ] Add comments to `Cargo.toml` files documenting what each feature flag enables
- [ ] Focus on flags that aren't self-explanatory

### Task 6: CUDA kernel constant namespacing (L6)

- [ ] Audit CUDA header constants for potential namespace collisions
- [ ] Add prefixes where needed (e.g., `XLOG_` prefix)

### Task 7: Config struct documentation (L7)

- [ ] Add `/// field docs` to public Config struct fields
- [ ] Focus on fields whose purpose isn't obvious from the name

### Task 8: Module re-export cleanup (L8)

- [ ] Review `pub use` re-exports in `lib.rs` files
- [ ] Remove re-exports of types that consumers should import from their defining module

### Task 9: Schema Iterator trait (L9)

- [ ] Implement `Iterator` for schema column iteration
- [ ] Replace manual indexing loops with iterator patterns
