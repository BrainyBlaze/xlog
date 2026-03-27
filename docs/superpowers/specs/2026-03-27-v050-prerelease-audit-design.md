# xlog v0.5.0 Pre-Release Audit — Design Spec

**Date**: 2026-03-27
**Timeline**: 1–2 weeks
**Branch**: `audit/v0.5.0-prerelease`
**Targets**: crates.io (Rust) + PyPI (pyxlog) — improvements now, publish later

---

## Phase 1: Cleanup & Consistency

### Version synchronization
- Update root `Cargo.toml` workspace version from `0.3.2` to `0.5.0`
- Verify all 14 crate `Cargo.toml` files inherit or match
- Ensure `pyproject.toml` / maturin config matches for PyPI

### TODO/FIXME audit
- Review all 21 markers across the codebase
- Resolve completed items, re-defer with issue links, or remove stale ones
- Items marked "deferred to v0.5.0" must be checked: done or re-deferred?

### Warning cleanup
- `cargo clippy --workspace --all-targets` — zero warnings
- `cargo doc --workspace --no-deps` — zero warnings
- Python: `ruff check` / `mypy` on the `python/` tree

### Dead code & unused dependencies
- `cargo +nightly udeps` to find unused deps
- Check for `#[allow(dead_code)]` masking real issues
- Remove leftover debug/checkpoint code from 5-wave refactoring

---

## Phase 2: Code Review — API Surface Audit

### Public API inventory
- Enumerate every `pub` item in each crate's `lib.rs`
- Check for accidentally-public internals (the 5-wave refactoring tightened 71 items — verify nothing slipped back)
- Ensure `#[non_exhaustive]` on public enums/structs where forward-compatibility matters

### Rustdoc coverage
- Enable `#![warn(missing_docs)]` on library crates, fix all gaps
- Every public function, struct, enum, trait needs a doc comment
- Add rustdoc examples for key entry points (`xlog-gpu`, `xlog-cli`, `pyxlog`)

### Safety audit (CUDA-specific)
- All `unsafe` blocks have `// SAFETY:` comments explaining the invariant
- FFI boundary between Rust and CUDA kernels: pointer lifetimes clear?
- GPU memory management in `xlog-cuda`: check for leaks, double-free paths
- DLPack / Arrow C Data Interface: verify zero-copy ownership semantics

### Python bindings (pyxlog)
- `.pyi` type stub files present and accurate
- Rust errors surface as meaningful Python exceptions
- Long-running GPU ops release the GIL via `py.allow_threads()`

### Dependency audit
- `cargo audit` for known CVEs
- Check pinned versions vs. ranges
- CUDA version compatibility: `cudarc 0.12` + `cuda-12060` matches documented sm_70+ requirement

---

## Phase 3: Test Validation

### Full test suite
- `cargo test --workspace` — all unit, integration, and doc tests pass
- CUDA certification suite (G01–G06, 206 tests) — run and capture results
- `pytest python/tests/` — all 109 Python tests pass

### Coverage gaps
- New v0.5.0 features have tests: term embeddings, `coo_chunk_budget`, `host_transfer_stats()`
- All 6 neural examples (`examples/neural/01_minimal` through `06_clutrr`) run end-to-end
- All 4 showcase examples (enterprise, knowledge-graph, game-analytics, supply-chain) parse and execute

### Benchmark sanity
- `cargo bench --workspace` compiles and runs
- Baseline in `BENCHMARKS.md` still representative after 5-wave refactoring

---

## Phase 4: Documentation Review

### README.md
- Quick-start commands work when copy-pasted
- Feature list matches what v0.5.0 actually ships
- Installation instructions accurate (CUDA version, Rust version, OS)
- No broken links or dead URLs

### CHANGELOG.md
- v0.5.0 entry is complete (everything on HEAD since v0.4.0-beta accounted for)
- "Unreleased" section resolved: folded into v0.5.0 or clearly marked post-v0.5.0
- Breaking changes called out in a dedicated subsection
- Migration guidance for users upgrading from v0.3.2

### ARCHITECTURE.md
- Module descriptions match post-refactoring crate/file structure
- Diagrams reflect current data flow
- Cross-references to files/functions still resolve

### language-reference.md
- Every syntax feature has a working example
- New v0.5.0 features documented (`register_embedding`, `forward_embedding`, `coo_chunk_budget`)
- No references to removed or renamed APIs

### ROADMAP.md
- Items shipped in v0.5.0 marked done
- Nothing marked "in progress" that is actually complete or abandoned

---

## Phase 5: Pre-Publish Validation (dry-run only)

- `cargo publish --dry-run` across crates in dependency order — surfaces missing metadata, oversized files, broken dependency paths
- `maturin build` — confirms the wheel builds cleanly
- License scan — no incompatible vendored code in the tree
- URL check — `repository`, `homepage`, `documentation` links in Cargo.toml are valid and reachable

---

## Phase 6: Technical Whitepaper

### Audience
Dual: systems developers (Rust/GPU) and AI/ML researchers (neural-symbolic).

### Structure
1. **Motivation** — why GPU-accelerated Datalog, the neural-symbolic gap
2. **Architecture overview** — crate stack, data flow: parse → compile → GPU execution
3. **Key innovations**:
   - GPU-native semi-naive fixpoint evaluation
   - GPU CDCL SAT verifier (zero host transfers)
   - Knowledge compilation (D4) on device
   - Neural-symbolic bridge (neural predicates, PyTorch autograd)
   - dILP training pipeline with promotion gates
4. **Benchmarks** — performance vs. existing systems, GPU speedups
5. **Usage examples** — 2–3 worked examples (deterministic, probabilistic, neural-symbolic)
6. **Current limitations & roadmap**

### Constraints
- ~5000–7000 words / 15–20 pages
- Technical but accessible — no Rust knowledge required for ML sections, no Datalog knowledge required for systems sections
- Written last, reflecting verified, cleaned-up reality
