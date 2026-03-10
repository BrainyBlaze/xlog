# Wave 3: API Coherence

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize naming conventions across the public API and consolidate the CUDA test harness.

**Architecture:** Rename public functions to follow consistent verb conventions. Test harness was already consolidated in Wave 2 Task 23.

**Tech Stack:** Rust (rename + deprecate pattern). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-10-codebase-refactoring-design.md` (H6, H7, H8)

**Prerequisite:** Wave 2 (module splits) must be complete.

---

### Task 1: Define and document naming conventions (H6)

**Files:**
- Create: `docs/conventions/api-naming.md`

- [ ] **Step 1: Write the naming convention document**

```markdown
# xlog API Naming Conventions

| Pattern | Convention | Example |
|---------|-----------|---------|
| GPU→CPU transfer | `read_*` | `read_column::<u32>()` |
| CPU→GPU transfer | `write_*` | `write_buffer_from_slice::<u32>()` |
| Cheap accessor | `get_*` | `get_device()`, `get_memory()` |
| Expensive computation | `compute_*` / `evaluate_*` | `compute_ilp_loss()` |
| Simple setter | `set_*` | `set_candidate_map()` |
| Multi-field setup | `configure_*` | `configure_training()` |
| Row count from device | `read_row_count()` | |
| Row count upload | `write_row_count()` | |
```

- [ ] **Step 2: Commit**

```bash
git add docs/conventions/
git commit -m "docs: establish API naming conventions"
```

### Task 2: Rename download_* to read_* in provider

**Files:**
- Modify: `crates/xlog-cuda/src/provider/memory.rs`
- Modify: All callers across workspace

**Approach:** Since Wave 2 added a generic `download_column<T>`, rename it to `read_column<T>`. Keep the deprecated `download_column_*` functions pointing to the new name.

- [ ] **Step 1: Add `read_column<T>` as the canonical name**

In `provider/memory.rs`, rename `download_column` to `read_column`. Add a deprecated wrapper:

```rust
#[deprecated(note = "Renamed to read_column")]
pub fn download_column<T: DeviceRepr>(&self, buf: &CudaBuffer, col: usize) -> Result<Vec<T>> {
    self.read_column(buf, col)
}
```

- [ ] **Step 2: Update callers**

Search and replace `download_column` → `read_column` across the workspace. The deprecated wrapper ensures nothing breaks if a caller is missed.

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "refactor(cuda): rename download_column to read_column per naming conventions"
```

### Task 3: Rename upload/create functions to write_*

- [ ] **Step 1:** Rename `create_buffer_from_slice<T>` → `write_buffer_from_slice<T>` (or keep `create_` for constructors — discuss with team).

**NOTE:** `create_*` is acceptable for factory methods. The convention applies to data transfer verbs. `create_buffer_from_slice` creates a new buffer, so `create_` may be appropriate. Only rename if the team agrees. Add deprecated wrappers if renaming.

- [ ] **Step 2: Run tests, commit**

### Task 4: Rename accessor methods

- [ ] **Step 1:** Audit and rename methods that don't follow `get_*` convention for cheap accessors.

Current: `device()`, `memory()`, `ptx_load_profile()`
These are already fine — Rust convention is to omit `get_` for simple accessors. No changes needed.

- [ ] **Step 2: Document the exception**

Add to `api-naming.md`: "Rust-idiomatic simple accessors (returning references) omit the `get_` prefix per Rust API guidelines."

### Task 5: Final validation

- [ ] **Step 1:** Run full workspace tests.
- [ ] **Step 2:** Run CUDA certification.
- [ ] **Step 3:** Run Python tests.
