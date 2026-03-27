# xlog v0.5.0 Pre-Release Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit and improve the xlog codebase for publication readiness (crates.io + PyPI), then produce a technical whitepaper.

**Architecture:** Six sequential phases — cleanup, API review, test validation, docs review, pre-publish dry-runs, whitepaper. Each phase produces commits on the `audit/v0.5.0-prerelease` branch.

**Tech Stack:** Rust (cargo, clippy, rustdoc, cargo-audit, cargo-udeps), Python (maturin, ruff, mypy, pytest), CUDA (requires GPU — manual steps marked with `[MANUAL/GPU]`).

**Constraints:** No GPU hardware in this session. CUDA-dependent steps are marked `[MANUAL/GPU]` and must be run by the user on a machine with an NVIDIA GPU.

---

## Task 1: Version Synchronization

**Files:**
- Modify: `Cargo.toml` (root workspace)
- Modify: `crates/xlog-cuda-tests/Cargo.toml`
- Modify: `crates/pyxlog/pyproject.toml`

**Context:** Three different versions exist right now:
- Root `Cargo.toml` workspace version: `0.3.2`
- `pyproject.toml` (Python package): `0.4.0`
- README / CHANGELOG / git tags: `v0.5.0`

All 13 crates except `xlog-cuda-tests` inherit from the workspace version, so updating the root fixes most crates. `xlog-cuda-tests` has a hardcoded `0.2.0`.

- [ ] **Step 1: Update root workspace version**

In `Cargo.toml` (root), change `version = "0.3.2"` to `version = "0.5.0"` under `[workspace.package]`.

- [ ] **Step 2: Update xlog-cuda-tests version**

In `crates/xlog-cuda-tests/Cargo.toml`, change `version = "0.2.0"` to `version = "0.5.0"`.

- [ ] **Step 3: Update pyproject.toml version**

In `crates/pyxlog/pyproject.toml`, change `version = "0.4.0"` to `version = "0.5.0"`.

- [ ] **Step 4: Verify all crates report 0.5.0**

Run:
```bash
cargo metadata --no-deps --format-version 1 | python -c "
import json, sys
meta = json.load(sys.stdin)
for p in sorted(meta['packages'], key=lambda x: x['name']):
    if p['name'].startswith('xlog') or p['name'] == 'pyxlog':
        print(f\"{p['name']:25s} {p['version']}\")
"
```

Expected: Every xlog-* and pyxlog crate shows `0.5.0`.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/xlog-cuda-tests/Cargo.toml crates/pyxlog/pyproject.toml
git commit -m "chore: sync all crate versions to 0.5.0"
```

---

## Task 2: Warning Cleanup — Rust

**Files:**
- Modify: Various `.rs` files flagged by clippy

**Context:** `cargo clippy` is Rust's official linter (like pylint/flake8 for Python). It catches common mistakes, style issues, and potential bugs. Zero warnings is the standard for published crates.

- [ ] **Step 1: Run clippy and capture output**

```bash
cargo clippy --workspace --all-targets 2>&1 | tee clippy-report.txt
```

Review the output. Warnings fall into categories:
- `needless_borrow` — using `&x` where `x` already works
- `redundant_clone` — cloning when a move would suffice
- `unused_imports` — leftover `use` statements
- `dead_code` — functions/types never called

- [ ] **Step 2: Fix all clippy warnings**

Fix each warning. Common fixes:
- Remove `&` from `needless_borrow`
- Remove `.clone()` from `redundant_clone`
- Delete unused `use` lines
- For `dead_code`: if genuinely unused, remove it. If it's a public API intended for users, add `#[allow(dead_code)]` with a comment explaining why.

- [ ] **Step 3: Re-run clippy to confirm zero warnings**

```bash
cargo clippy --workspace --all-targets 2>&1 | grep -c "^warning"
```

Expected: `0`

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "chore: fix all clippy warnings"
```

---

## Task 3: Warning Cleanup — Rustdoc

**Files:**
- Modify: Various `.rs` files with broken doc links or syntax

**Context:** `cargo doc` generates HTML documentation from `///` comments. Warnings here mean broken links, invalid code examples, or missing references — all of which show up on docs.rs when you publish.

- [ ] **Step 1: Run cargo doc and capture warnings**

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps 2>&1 | tee doc-warnings.txt
```

The `-D warnings` flag turns warnings into errors so they're easy to count.

- [ ] **Step 2: Fix all doc warnings**

Common fixes:
- Broken intra-doc links: `[SomeType]` where `SomeType` isn't in scope — fix the path or remove the link
- Code blocks that don't compile: add `# ` prefix to hide setup lines, or mark as ```` ```no_run ```` / ```` ```ignore ````
- Missing backticks around type names in prose

- [ ] **Step 3: Re-run to confirm zero warnings**

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps 2>&1 | grep -c "^error"
```

Expected: `0`

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "chore: fix all rustdoc warnings"
```

---

## Task 4: Warning Cleanup — Python

**Files:**
- Modify: Files under `python/`

**Context:** `ruff` is a fast Python linter (replaces flake8/isort/pyflakes). `mypy` checks type annotations. Both should be clean before PyPI publish.

- [ ] **Step 1: Check if ruff and mypy are installed**

```bash
pip install ruff mypy 2>/dev/null
```

- [ ] **Step 2: Run ruff on python/ tree**

```bash
ruff check python/ 2>&1 | tee ruff-report.txt
```

- [ ] **Step 3: Fix ruff issues (auto-fix where possible)**

```bash
ruff check python/ --fix
```

Review remaining issues manually and fix.

- [ ] **Step 4: Run mypy on python/ tree**

```bash
mypy python/ --ignore-missing-imports 2>&1 | tee mypy-report.txt
```

`--ignore-missing-imports` skips errors from missing type stubs for third-party packages (like pyxlog itself, which has no .pyi files yet).

- [ ] **Step 5: Fix mypy issues**

Fix type annotation issues flagged by mypy. Common fixes:
- Add type annotations to function signatures
- Fix type mismatches
- Add `# type: ignore` only as last resort with a comment explaining why

- [ ] **Step 6: Commit**

```bash
git add -u python/
git commit -m "chore: fix Python linting and type issues"
```

---

## Task 5: Dead Code & Unused Dependencies

**Files:**
- Modify: Various `Cargo.toml` and `.rs` files

**Context:** `cargo +nightly udeps` uses the nightly compiler to detect dependencies listed in `Cargo.toml` that your code never imports. Removing them speeds up compilation and shrinks the dependency tree.

- [ ] **Step 1: Install cargo-udeps (if not present)**

```bash
cargo install cargo-udeps --locked 2>/dev/null
```

Requires nightly Rust: `rustup install nightly`

- [ ] **Step 2: Run udeps**

```bash
cargo +nightly udeps --workspace 2>&1 | tee udeps-report.txt
```

- [ ] **Step 3: Remove unused dependencies**

For each flagged dependency, remove it from the relevant `Cargo.toml`. Be careful:
- Some deps are only used behind feature flags — check `[features]` sections
- Some deps are only used in tests — check `[dev-dependencies]`

- [ ] **Step 4: Audit `#[allow(dead_code)]` directives**

There are 33 of these across 20 files. For each one:

```bash
grep -rn "allow(dead_code)" crates/
```

Decide:
- If the code is genuinely unused and not part of the public API → remove the code
- If it's a public API reserved for users → keep, but add a doc comment
- If it's used only behind a feature flag → replace with `#[cfg(feature = "...")]`

- [ ] **Step 5: Verify the project still builds**

```bash
cargo build --workspace
```

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "chore: remove unused deps and dead code"
```

---

## Task 6: Dependency Security Audit

**Files:**
- Possibly modify: `Cargo.toml` files if vulnerable deps need updating

**Context:** `cargo audit` checks your dependency tree against the RustSec Advisory Database — a community-maintained list of known vulnerabilities in Rust crates (like npm audit or pip-audit).

- [ ] **Step 1: Install cargo-audit (if not present)**

```bash
cargo install cargo-audit --locked 2>/dev/null
```

- [ ] **Step 2: Run audit**

```bash
cargo audit 2>&1 | tee audit-report.txt
```

- [ ] **Step 3: Address findings**

For each advisory:
- If a patch version exists → update the dependency in `Cargo.toml`
- If no patch exists → document the risk and whether it affects xlog's use case
- Run `cargo update` after changes to refresh `Cargo.lock`

- [ ] **Step 4: Verify build still works**

```bash
cargo build --workspace
```

- [ ] **Step 5: Commit (if changes made)**

```bash
git add Cargo.toml Cargo.lock crates/*/Cargo.toml
git commit -m "chore: address dependency security advisories"
```

---

## Task 7: Public API Inventory

**Files:**
- Modify: Various `lib.rs` and module files across crates

**Context:** Every `pub` item becomes your API contract when published to crates.io. Changing or removing a `pub` item later is a breaking change (requires a major version bump). This task checks that only intentional items are public.

The crates with the largest public surface are:
- `xlog-prob`: 194 pub fns, 53 pub structs
- `xlog-cuda-tests`: 158 pub fns (this is a test crate — should most of this be pub?)
- `xlog-cuda`: 148 pub fns, 24 pub structs
- `xlog-neural`: 105 pub fns

- [ ] **Step 1: Generate public API report**

For each library crate, list all pub items:

```bash
for crate_dir in crates/xlog-*/src/lib.rs crates/pyxlog/src/lib.rs; do
    echo "=== $(dirname $(dirname $crate_dir)) ==="
    grep -rn "^pub " "$(dirname $crate_dir)/" | head -50
    echo ""
done
```

- [ ] **Step 2: Review xlog-cuda-tests visibility**

This is a test/certification crate. Most of its 158 pub fns are likely test helpers. Unless other crates depend on them, downgrade to `pub(crate)`:
- Check: `grep -r "xlog-cuda-tests" crates/*/Cargo.toml` — is anything depending on it?
- If not, make internal items `pub(crate)` instead of `pub`

- [ ] **Step 3: Check for missing `#[non_exhaustive]`**

Search for public enums and structs that users might match on:

```bash
grep -rn "^pub enum\|^pub struct" crates/*/src/
```

For each one, ask: "Will we ever add variants/fields?" If yes, add `#[non_exhaustive]`.

Key candidates:
- Error enums (you'll add new error variants as the project grows)
- Configuration structs (you'll add new fields)
- Public enums representing categories/modes

- [ ] **Step 4: Tighten visibility where needed**

Change accidentally-public items to:
- `pub(crate)` — visible within the crate only
- `pub(super)` — visible to the parent module only
- Remove `pub` entirely if it's a private helper

- [ ] **Step 5: Verify build**

```bash
cargo build --workspace
```

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "refactor: tighten public API visibility"
```

---

## Task 8: Rustdoc Coverage — Enable `missing_docs` Warning

**Files:**
- Modify: `lib.rs` in each library crate

**Context:** Adding `#![warn(missing_docs)]` at the top of `lib.rs` makes the compiler warn whenever a public item lacks a `///` doc comment. This is the standard way to enforce documentation in Rust. Currently NONE of the 14 crates have this enabled.

This is a large task. Prioritize the user-facing crates (`xlog-gpu`, `xlog-cli`, `pyxlog`) and the core crates (`xlog-core`, `xlog-ir`). Internal crates can be done incrementally.

- [ ] **Step 1: Enable `missing_docs` on xlog-core**

Add `#![warn(missing_docs)]` as the first line in `crates/xlog-core/src/lib.rs`.

```bash
cargo doc --no-deps -p xlog-core 2>&1 | grep "missing documentation"
```

Count the warnings. For each public item, add a `///` doc comment explaining what it does.

- [ ] **Step 2: Enable `missing_docs` on xlog-ir**

Same process for `crates/xlog-ir/src/lib.rs`.

- [ ] **Step 3: Enable `missing_docs` on xlog-gpu**

Same for `crates/xlog-gpu/src/lib.rs`. This is the main user-facing Rust API — docs here matter most.

- [ ] **Step 4: Enable `missing_docs` on xlog-cli**

Same for `crates/xlog-cli/src/main.rs` (or `lib.rs` if it has one). CLI crates may have less public API.

- [ ] **Step 5: Enable `missing_docs` on remaining crates**

Work through: `xlog-cuda`, `xlog-logic`, `xlog-runtime`, `xlog-prob`, `xlog-solve`, `xlog-stats`, `xlog-neural`.

For crates with very large public surfaces (xlog-prob has 194 pub fns), it's acceptable to:
- Add doc comments to the most important items
- Use `#[allow(missing_docs)]` on internal-but-pub items with a `// TODO(v0.6): document or make pub(crate)` note

- [ ] **Step 6: Verify zero doc warnings on priority crates**

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p xlog-core -p xlog-ir -p xlog-gpu 2>&1
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add -u
git commit -m "docs: enable missing_docs lint, add rustdoc to public API"
```

---

## Task 9: Unsafe Block Audit

**Files:**
- Modify: Various `.rs` files containing `unsafe` blocks

**Context:** Rust's `unsafe` keyword opts out of the compiler's memory safety guarantees. Every `unsafe` block should have a `// SAFETY:` comment explaining why the operation is safe. There are 467 unsafe blocks across 51 files — mostly in GPU/FFI code, which is expected.

This task does NOT rewrite unsafe code — it documents the safety invariants.

- [ ] **Step 1: Find unsafe blocks missing SAFETY comments**

```bash
# This finds `unsafe` blocks that don't have a SAFETY comment on the preceding line
grep -rn "unsafe {" crates/ --include="*.rs" | while read line; do
    file=$(echo "$line" | cut -d: -f1)
    lineno=$(echo "$line" | cut -d: -f2)
    prev=$((lineno - 1))
    if ! sed -n "${prev}p" "$file" | grep -qi "safety"; then
        echo "$line"
    fi
done 2>&1 | tee unsafe-audit.txt
```

- [ ] **Step 2: Add SAFETY comments to high-priority files**

Focus on files with the most unsafe blocks:
- `crates/xlog-cuda/src/provider/relational.rs` (52 blocks)
- `crates/xlog-prob/src/compilation/gpu_d4/frontier.rs` (41 blocks)
- `crates/xlog-prob/src/compilation/gpu_d4/build.rs` (24 blocks)
- `crates/xlog-prob/src/exact.rs` (18 blocks)
- `crates/xlog-cuda/src/provider/ilp.rs` (18 blocks)

For each unsafe block, add a comment above it:
```rust
// SAFETY: <why this is safe>
// - pointer is valid because <reason>
// - alignment is guaranteed by <reason>
// - no aliasing because <reason>
unsafe { ... }
```

Common patterns in GPU code:
- Raw pointer from `cudarc` device allocation → safe because cudarc manages lifetime
- FFI call to CUDA kernel → safe because arguments match kernel signature
- Transmute for DLPack/Arrow interop → safe because layout is guaranteed by spec

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "docs: add SAFETY comments to unsafe blocks"
```

---

## Task 10: Python Bindings Audit

**Files:**
- Create: `crates/pyxlog/python/pyxlog/_native.pyi` (type stubs)
- Modify: `crates/pyxlog/src/lib.rs` (GIL handling review)

**Context:** Python users get no IDE autocomplete or type checking without `.pyi` stub files. These are type declaration files (like `.d.ts` in TypeScript) that describe the Python-visible API.

- [ ] **Step 1: Generate type stubs from PyO3 definitions**

Read through `crates/pyxlog/src/lib.rs` and all its submodules. For each `#[pyfunction]` and `#[pymethods]` block, write the corresponding Python type stub.

Create `crates/pyxlog/python/pyxlog/_native.pyi`:

```python
"""Type stubs for the pyxlog native module (Rust/PyO3)."""
from typing import Optional, Dict, List, Any

# For each #[pyfunction] in lib.rs, add a stub:
# def function_name(arg: type, ...) -> return_type: ...
```

The exact contents depend on what's in the PyO3 code — read `lib.rs` and its submodules first.

- [ ] **Step 2: Create `pyxlog/py.typed` marker**

This file (empty) tells Python tools like mypy that this package supports type checking:

```bash
touch crates/pyxlog/python/pyxlog/py.typed
```

- [ ] **Step 3: Review GIL handling**

Search for long-running operations that should release the GIL:

```bash
grep -rn "py.allow_threads\|Python::with_gil\|#\[pyfunction\]" crates/pyxlog/src/
```

Any function that calls GPU operations (kernel launches, memory transfers) should use `py.allow_threads(|| { ... })` to release the GIL during the GPU work. Otherwise Python threads block.

- [ ] **Step 4: Review error mapping**

Check that Rust errors become meaningful Python exceptions:

```bash
grep -rn "PyErr\|PyResult\|to_pyerr\|pyo3::exceptions" crates/pyxlog/src/
```

Ensure errors include the Rust error message (not just "internal error").

- [ ] **Step 5: Commit**

```bash
git add crates/pyxlog/python/pyxlog/_native.pyi crates/pyxlog/python/pyxlog/py.typed
git add -u
git commit -m "feat: add Python type stubs and review bindings quality"
```

---

## Task 11: Test Validation — Rust

**Files:**
- No modifications expected (unless tests fail)

- [ ] **Step 1: Run full Rust test suite**

```bash
cargo test --workspace 2>&1 | tee test-report.txt
```

This runs unit tests, integration tests, and doc tests across all 14 crates. Some tests may be skipped if they require CUDA — that's expected.

- [ ] **Step 2: Check for ignored/skipped tests**

```bash
grep -rn "#\[ignore\]" crates/ --include="*.rs"
```

Review each ignored test: is it ignored for a good reason (e.g., requires GPU), or is it a forgotten broken test?

- [ ] **Step 3: `[MANUAL/GPU]` Run CUDA certification suite**

On a machine with an NVIDIA GPU:

```bash
cargo test -p xlog-cuda-tests --release 2>&1 | tee cuda-cert-report.txt
```

Expected: 206/206 tests pass.

- [ ] **Step 4: `[MANUAL/GPU]` Run full workspace tests with GPU**

```bash
cargo test --workspace --release 2>&1 | tee full-test-report.txt
```

- [ ] **Step 5: Commit test reports (if desired)**

```bash
git add *-report.txt
git commit -m "docs: add test validation reports for v0.5.0 audit"
```

---

## Task 12: Test Validation — Python

**Files:**
- No modifications expected (unless tests fail)

- [ ] **Step 1: `[MANUAL/GPU]` Run Python test suite**

On a machine with NVIDIA GPU + pyxlog wheel installed:

```bash
cd crates/pyxlog && maturin develop --release && cd ../..
pytest python/tests/ -v 2>&1 | tee pytest-report.txt
```

Expected: 109+ tests pass (the count may have grown since v0.4.0-alpha).

- [ ] **Step 2: `[MANUAL/GPU]` Validate neural examples**

Run each of the 6 neural examples to confirm they execute:

```bash
for ex in examples/neural/0{1,2,3,4,5,6}_*; do
    echo "=== $ex ==="
    python "$ex/train.py" --epochs 1 --smoke-test 2>&1 | tail -5
done
```

- [ ] **Step 3: Validate deterministic examples**

```bash
python scripts/validate_examples.py 2>&1 | tee example-validation.txt
```

Or run the CLI directly on each `.xlog` file:

```bash
for f in examples/xlog/**/*.xlog; do
    echo "=== $f ==="
    cargo run --release -- run "$f" 2>&1 | tail -3
done
```

---

## Task 13: Benchmark Sanity Check

**Files:**
- Modify: `docs/BENCHMARKS.md` if baselines are stale

- [ ] **Step 1: `[MANUAL/GPU]` Check benchmarks compile**

```bash
cargo bench --workspace --no-run 2>&1
```

This compiles benchmarks without running them — catches build issues.

- [ ] **Step 2: `[MANUAL/GPU]` Run benchmarks**

```bash
cargo bench --workspace 2>&1 | tee bench-report.txt
```

- [ ] **Step 3: Compare against BENCHMARKS.md baselines**

Read `docs/BENCHMARKS.md` and compare the reported numbers against the fresh run. If the 5-wave refactoring changed performance characteristics, update the baselines.

- [ ] **Step 4: Commit (if baselines updated)**

```bash
git add docs/BENCHMARKS.md
git commit -m "docs: update benchmark baselines post-refactoring"
```

---

## Task 14: README Review

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Verify quick-start commands**

The README shows a transitive closure example. Run it:

```bash
cargo run --release -- run examples/xlog/00-basics/transitive-closure.xlog 2>&1
```

Confirm the output matches what README claims.

- [ ] **Step 2: Check feature list accuracy**

Read README section by section. For each claimed feature, verify it exists:
- "Term embeddings" → `register_embedding` in the codebase
- "GPU CDCL verifier" → `gpu_cdcl` module exists
- "DLPack" → `dlpack` module exists
- etc.

Flag any feature listed that doesn't work or doesn't exist yet.

- [ ] **Step 3: Validate all links**

```bash
grep -oP 'https?://[^\s\)]+' README.md | while read url; do
    status=$(curl -o /dev/null -s -w "%{http_code}" "$url" 2>/dev/null)
    echo "$status $url"
done
```

Fix or remove broken links.

- [ ] **Step 4: Check installation instructions**

Verify the stated requirements match reality:
- Rust version: check `rust-toolchain.toml` or `Cargo.toml` edition
- CUDA version: check cudarc feature flag
- OS: confirm Linux-only or if Windows/Mac partially work

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: update README for v0.5.0 accuracy"
```

---

## Task 15: CHANGELOG Review

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Resolve the "Unreleased" section**

There are 42 commits past the v0.5.0 tag. The CHANGELOG has an "Unreleased" section with:
- MC runtime optimization
- Evidence clamping
- Provenance primitives
- 5-wave refactoring (57 commits)
- ILP reliability gate optimization
- Strict winner metadata
- Recursive GPU row-count fix

Decision needed: Are these part of v0.5.0 or v0.5.1/v0.6.0?

If they're v0.5.0 → merge them into the v0.5.0 section and re-tag.
If they're post-v0.5.0 → keep in Unreleased, clearly labeled.

- [ ] **Step 2: Check for breaking changes**

Search the v0.5.0 section for renames, removals, or behavior changes:
- `coo_memory_cap` → `coo_chunk_budget` (renamed parameter — breaking!)
- Artifact schema `beta-v1` → `beta-v2` (migration — potentially breaking)

Add a `### Breaking Changes` subsection if not present.

- [ ] **Step 3: Add migration guidance**

Users upgrading from v0.3.2 (last published version) to v0.5.0 need guidance. Add a section:

```markdown
### Migrating from v0.3.2

- `coo_memory_cap` parameter renamed to `coo_chunk_budget`
- Artifact format upgraded to `beta-v2` (run migration tool: ...)
- [List other breaking changes]
```

- [ ] **Step 4: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: finalize CHANGELOG for v0.5.0 release"
```

---

## Task 16: ARCHITECTURE.md Review

**Files:**
- Modify: `docs/ARCHITECTURE.md`

- [ ] **Step 1: Verify crate descriptions match reality**

The 5-wave refactoring restructured 5 major modules into 27 submodules. Read the ARCHITECTURE.md crate descriptions and cross-reference with actual file structure:

```bash
# For each crate, list its module structure
for crate in crates/xlog-*/src/; do
    echo "=== $crate ==="
    find "$crate" -name "*.rs" | sort
done
```

Flag descriptions that reference old file names or module structures.

- [ ] **Step 2: Update stale sections**

Fix any references to:
- Old module names (pre-refactoring)
- Removed files or functions
- Outdated dependency relationships

- [ ] **Step 3: Commit**

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: update ARCHITECTURE.md for post-refactoring structure"
```

---

## Task 17: Language Reference Review

**Files:**
- Modify: `docs/language-reference.md`

**Context:** This file says "v0.3.2" and "Last Updated: January 2026" — it's at least 2 months stale and 2 versions behind.

- [ ] **Step 1: Update version and date**

Change the header to reflect v0.5.0 and current date.

- [ ] **Step 2: Check for new v0.5.0 syntax**

Features added since v0.3.2 that need documentation:
- `register_embedding()` / `forward_embedding()` (term embeddings)
- `coo_chunk_budget` parameter
- `host_transfer_stats()` API
- `McSamplingMethod::EvidenceClamping`
- Any new pragmas or builtins

Search the CHANGELOG for syntax/API additions and ensure each is in the reference.

- [ ] **Step 3: Verify all examples compile**

For each code example in the reference, try running it:

```bash
# Extract .xlog examples and test them
cargo run --release -- run <example>
```

- [ ] **Step 4: Commit**

```bash
git add docs/language-reference.md
git commit -m "docs: update language reference to v0.5.0"
```

---

## Task 18: ROADMAP.md Review

**Files:**
- Modify: `docs/ROADMAP.md`

- [ ] **Step 1: Mark shipped features as done**

Read ROADMAP.md and cross-reference against CHANGELOG v0.5.0. Any feature marked "in progress" or "planned" that has been shipped should be marked "done" or "shipped in v0.5.0".

- [ ] **Step 2: Remove or re-date abandoned items**

If any planned feature is no longer intended, remove it or add a note.

- [ ] **Step 3: Commit**

```bash
git add docs/ROADMAP.md
git commit -m "docs: update ROADMAP.md for v0.5.0 status"
```

---

## Task 19: Pre-Publish Dry-Run — Crates.io

**Files:**
- Possibly modify: `Cargo.toml` files if metadata is missing

**Context:** `cargo publish --dry-run` packages your crate exactly as it would be uploaded, but doesn't actually upload. It catches:
- Missing required fields (`description`, `license`, `repository`)
- Files exceeding crates.io size limits (10MB per crate)
- Dependency resolution issues

- [ ] **Step 1: Determine publish order**

Crates must be published in dependency order (leaf crates first):

```bash
cargo metadata --no-deps --format-version 1 | python -c "
import json, sys
meta = json.load(sys.stdin)
for p in sorted(meta['packages'], key=lambda x: x['name']):
    if p['name'].startswith('xlog') or p['name'] == 'pyxlog':
        deps = [d['name'] for d in p['dependencies'] if d['name'].startswith('xlog')]
        print(f\"{p['name']:25s} depends on: {', '.join(deps) or '(none)'}\")
"
```

Expected publish order (leaf → root):
1. xlog-core
2. xlog-ir, xlog-stats
3. xlog-cuda, xlog-solve
4. xlog-logic, xlog-runtime
5. xlog-prob, xlog-neural
6. xlog-gpu
7. xlog-cli, pyxlog
8. xlog-cuda-tests (if published at all)

- [ ] **Step 2: Check metadata completeness**

Each crate needs in its `Cargo.toml`:
- `description` — one-line summary
- `license` — SPDX expression (should inherit "MIT OR Apache-2.0")
- `repository` — GitHub URL
- `readme` — path to README (can point to root README)
- `keywords` — up to 5 crates.io search terms
- `categories` — crates.io categories

```bash
for toml in crates/*/Cargo.toml; do
    echo "=== $toml ==="
    grep -E "^description|^license|^repository|^readme|^keywords|^categories" "$toml"
    echo ""
done
```

- [ ] **Step 3: Run dry-run on each crate**

```bash
for crate in xlog-core xlog-ir xlog-stats xlog-cuda xlog-solve xlog-logic xlog-runtime xlog-prob xlog-neural xlog-gpu xlog-cli; do
    echo "=== $crate ==="
    cargo publish --dry-run -p "$crate" 2>&1 | tail -5
    echo ""
done
```

Fix any errors (usually missing metadata fields).

- [ ] **Step 4: Check crate sizes**

```bash
for crate in crates/xlog-*/; do
    echo "=== $crate ==="
    du -sh "$crate"
done
```

Crates.io limit is 10MB per crate. If any exceed, add files to `.cargo` exclude list or a crate-level `exclude` in `Cargo.toml`.

- [ ] **Step 5: Check crate name availability**

```bash
for name in xlog-core xlog-ir xlog-cuda xlog-runtime xlog-logic xlog-prob xlog-solve xlog-stats xlog-neural xlog-gpu xlog-cli pyxlog; do
    status=$(curl -s -o /dev/null -w "%{http_code}" "https://crates.io/api/v1/crates/$name")
    echo "$name: $status (200=taken, 404=available)"
done
```

- [ ] **Step 6: Commit metadata fixes**

```bash
git add crates/*/Cargo.toml Cargo.toml
git commit -m "chore: complete crates.io metadata for all crates"
```

---

## Task 20: Pre-Publish Dry-Run — PyPI

**Files:**
- Possibly modify: `crates/pyxlog/pyproject.toml`

- [ ] **Step 1: `[MANUAL/GPU]` Build the wheel**

```bash
cd crates/pyxlog && maturin build --release 2>&1 | tee wheel-build.txt && cd ../..
```

- [ ] **Step 2: Check wheel metadata**

```bash
pip install pkginfo
python -c "
from pkginfo import Wheel
w = Wheel('target/wheels/pyxlog-0.5.0-*.whl')
print(f'Name: {w.name}')
print(f'Version: {w.version}')
print(f'Summary: {w.summary}')
print(f'License: {w.license}')
print(f'Requires-Python: {w.requires_python}')
print(f'Classifiers: {w.classifiers}')
"
```

- [ ] **Step 3: Add PyPI classifiers**

In `crates/pyxlog/pyproject.toml`, add classifiers if missing:

```toml
classifiers = [
    "Development Status :: 4 - Beta",
    "Programming Language :: Rust",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "License :: OSI Approved :: Apache Software License",
]
```

- [ ] **Step 4: `[MANUAL/GPU]` Smoke test the wheel in a fresh virtualenv**

```bash
python -m venv /tmp/pyxlog-test
source /tmp/pyxlog-test/bin/activate
pip install target/wheels/pyxlog-0.5.0-*.whl
python -c "import pyxlog; print(pyxlog.__version__)"
deactivate
```

- [ ] **Step 5: Check PyPI name availability**

```bash
curl -s -o /dev/null -w "%{http_code}" "https://pypi.org/pypi/pyxlog/json"
# 200 = taken, 404 = available
```

- [ ] **Step 6: Commit**

```bash
git add crates/pyxlog/pyproject.toml
git commit -m "chore: complete PyPI metadata for pyxlog"
```

---

## Task 21: License Scan

**Files:**
- None expected (unless issues found)

- [ ] **Step 1: Check all dependency licenses**

```bash
cargo install cargo-license --locked 2>/dev/null
cargo license --avoid-build-deps --avoid-dev-deps 2>&1 | tee license-report.txt
```

`cargo-license` lists the license of every dependency. Look for:
- `GPL` or `AGPL` — these are copyleft and may be incompatible with MIT/Apache-2.0
- `UNKNOWN` — needs manual investigation
- `Proprietary` — cannot be redistributed

- [ ] **Step 2: Verify xlog's own license files**

```bash
ls LICENSE-*
# Should see LICENSE-MIT and LICENSE-APACHE
```

Check that both files exist, are non-empty, and have the correct copyright holder.

- [ ] **Step 3: Check for vendored code**

```bash
find . -name "vendor" -o -name "third_party" -o -name "external" 2>/dev/null
```

If any vendored code exists, verify its license is compatible.

---

## Task 22: Technical Whitepaper — Research & Outline

**Files:**
- Create: `docs/whitepaper-v050.md`

**Context:** This is the last task. By now, every claim in the whitepaper can be verified against audited, cleaned-up code and docs.

- [ ] **Step 1: Gather benchmark data**

Collect from:
- `docs/BENCHMARKS.md` — existing baselines
- CHANGELOG performance claims (MC 8.6% improvement, ILP 4.6x compilation speedup, etc.)
- Neural example results in `examples/neural/results/`
- Any comparison data against DeepProbLog in `examples/neural/baseline/`

- [ ] **Step 2: Write outline**

```markdown
# xlog: A GPU-Accelerated Datalog Engine for Neural-Symbolic AI

## 1. Introduction (~500 words)
- The gap between neural and symbolic AI
- Why Datalog as the symbolic substrate
- Why GPU acceleration matters (dataset scale, training loop integration)

## 2. Architecture (~1500 words)
- Crate structure and design philosophy
- Data flow: source text → AST → RIR → GPU execution plan → results
- GPU execution model: semi-naive fixpoint on device
- Memory model: device-resident relations, zero-copy interop

## 3. Key Innovations (~2000 words)
### 3.1 GPU-Native Semi-Naive Evaluation
- Hash joins, radix sort, dedup on GPU
- Stratum scheduling, recursive fixpoint

### 3.2 Probabilistic Inference on GPU
- Knowledge compilation (D4) on device
- GPU CDCL SAT verifier
- Monte Carlo sampling with evidence clamping

### 3.3 Neural-Symbolic Bridge
- Neural predicates: PyTorch networks as Datalog predicates
- Autograd through the symbolic layer
- Term embeddings (register_embedding / forward_embedding)

### 3.4 Differentiable ILP
- Sparse GPU mask API
- Promotion pipeline with 6 gates
- Hard-negative mining

## 4. Performance (~1000 words)
- Benchmark methodology
- Comparison with DeepProbLog
- GPU speedups: exact inference, MC sampling, training
- Scaling characteristics

## 5. Usage Examples (~500 words)
- Deterministic: transitive closure / supply chain BOM
- Probabilistic: wet grass / reachability
- Neural-symbolic: MNIST addition training

## 6. Limitations & Future Work (~500 words)
- Linux-only, NVIDIA GPU required
- Epistemic logic (xlog-elp) planned but not shipped
- Python batch query limited to U32 entity IDs
- Roadmap: v0.6.0 goals
```

- [ ] **Step 3: Write the whitepaper**

Fill in each section based on the codebase, ARCHITECTURE.md, CHANGELOG, benchmark data, and example code. Target 5000–7000 words.

Tone guidelines:
- Systems developers: explain GPU pipeline, memory model, Rust design choices
- ML researchers: explain neural predicates, comparison with DeepProbLog, dILP training
- Use diagrams where possible (ASCII art or mermaid syntax)
- Include code snippets from actual examples (not made-up code)

- [ ] **Step 4: Self-review the whitepaper**

Check:
- Every performance claim has a source (benchmark data, commit, CHANGELOG entry)
- Every feature claim is verified against the audited codebase
- No features are described that don't exist or don't work
- Code examples are copy-pasteable and correct

- [ ] **Step 5: Commit**

```bash
git add docs/whitepaper-v050.md
git commit -m "docs: add v0.5.0 technical whitepaper"
```

---

## Summary

| Task | Phase | GPU Required | Estimated Effort |
|------|-------|-------------|-----------------|
| 1. Version sync | Cleanup | No | 10 min |
| 2. Clippy warnings | Cleanup | No | 30 min |
| 3. Rustdoc warnings | Cleanup | No | 30 min |
| 4. Python linting | Cleanup | No | 30 min |
| 5. Dead code / unused deps | Cleanup | No | 1 hr |
| 6. Dependency audit | Cleanup | No | 30 min |
| 7. Public API inventory | Code Review | No | 2 hr |
| 8. Rustdoc coverage | Code Review | No | 4 hr |
| 9. Unsafe block audit | Code Review | No | 2 hr |
| 10. Python bindings | Code Review | No | 2 hr |
| 11. Rust tests | Test | GPU for full | 30 min |
| 12. Python tests | Test | Yes | 1 hr |
| 13. Benchmark sanity | Test | Yes | 1 hr |
| 14. README review | Docs | No | 1 hr |
| 15. CHANGELOG review | Docs | No | 1 hr |
| 16. ARCHITECTURE.md review | Docs | No | 2 hr |
| 17. Language reference | Docs | No | 2 hr |
| 18. ROADMAP review | Docs | No | 30 min |
| 19. Crates.io dry-run | Pre-Publish | No | 1 hr |
| 20. PyPI dry-run | Pre-Publish | GPU for build | 1 hr |
| 21. License scan | Pre-Publish | No | 30 min |
| 22. Whitepaper | Whitepaper | No | 4-6 hr |

**Total estimated: ~28-30 hours of work across 1-2 weeks**
