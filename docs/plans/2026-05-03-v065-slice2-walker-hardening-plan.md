# v0.6.5 Slice 2 — `MultiWayJoin` Walker Hardening (amended)

**Date:** 2026-05-03
**Branch (proposed):** `feat/v065-multiway-walker-hardening`
**Worktree (proposed):** `.worktrees/v065-multiway-walker-hardening`
**Baseline commit:** `038e22f6` (origin/main HEAD)
**Status:** Plan, post-review amendments. Approved as doc + tests only — no production behavior changes, no public API, no CUDA, no crate-type changes.

## Goal

Lock every `RirNode` walker's behavior around `MultiWayJoin` with explicit arms (slice 1 already added them), shape-agnosticism guarantees pinned by tests, and concise inline documentation of the contract. Slice 2a (4-way kernels) and 2b (cost model) will add new `MultiWayJoin` shapes; they need the foundation hardened.

This slice is **defensive only**.

## Scope

Five deliverables. All tests live in their natural home crate (no synthetic dependencies).

1. **D1 — Inline `MultiWayJoin` walker contract** in `xlog-ir::rir`. Short and stable; no audit table.
2. **D2 — Cross-crate runtime fallback tests** in `xlog-integration` (executor paths only).
3. **D3 — `xlog-prob` MC sampling fallback test** in `xlog-prob/tests/`.
4. **D4 — Shape-agnosticism tests** in walker home crates using synthesized 4-input IR.
5. **D5 — Strengthen existing optimizer arm tests** to use non-canonical 4-input/4-output IR.

## Audit Findings (Risks)

### R1. `Executor::execute_non_recursive_scc` is the WCOJ-bypass path used by `xlog-prob::mc::sampling` for monotone SCCs

`xlog-runtime/src/executor/recursive.rs:26` — `pub fn execute_non_recursive_scc(...)` is a separate entry point from `execute_stratum_impl`. It calls `self.execute_node(&rule.body)` directly without invoking `try_dispatch_wcoj_triangle`. After slice 1, eligible triangle bodies are `MultiWayJoin`, so the safety-net arm (`MultiWayJoin { fallback, .. } => execute_node(fallback)`) in `node_dispatch.rs` carries correctness for this entire path.

`xlog-prob/src/mc/sampling.rs:46,81` calls `executor.execute_non_recursive_scc(rules)` for monotone SCCs in MC sampling. **No test exercises this with a triangle body.** If a future refactor changes the safety-net arm to `unimplemented!()` or `return Err(...)` (because "the dispatch hook handles this"), MC sampling on triangle programs breaks silently.

### R2. Shape assumptions hard-coded in walker arms — risk for slice 2a (4-way)

Every `MultiWayJoin` arm landed in slice 1 is shape-agnostic *today* (verified by inspection), but no test pins this. When slice 2a generalizes the promoter to emit 4-input `MultiWayJoin` nodes, an arm that accidentally added `inputs.len() == 3` or `output_columns.len() == 3` along the way would silently regress.

### R3. `pyxlog::ilp::walk_tmj` is unreachable from a normal cross-crate test

`pyxlog/Cargo.toml`: `crate-type = ["cdylib"]`, `test = false`. `walk_tmj` is private. We cannot link a normal Rust integration test against it, and we will not change the crate type or expose new API for a defensive slice. Coverage for this arm is **source-contract only** (see D2).

### R4. `Optimizer::estimate_width` and `find_column_relation` are private

These are pub(crate)/private inside `xlog_logic::optimizer`. They cannot be called from `xlog-integration` tests. Optimizer arm coverage stays in `optimizer.rs`'s `mod tests`.

### R5. The promoter is still triangle-only

A 4-input `MultiWayJoin` cannot come from `Compiler::compile()` yet. All 4-input shape-agnosticism tests **must synthesize the IR directly**.

## Deliverables

### D1 — Inline contract doc in `xlog-ir::rir`

Add a short doc block above the `MultiWayJoin` variant (NO file-by-file audit table):

> **Walker contract.** Generic walkers and visitors that handle
> `MultiWayJoin` MUST be shape-agnostic over `inputs`, `slot_vars`,
> and `output_columns` — no walker may assume a fixed arity or a
> specific variable-class layout. Only matchers/promoters whose
> name carries an explicit shape qualifier (e.g.
> `match_multiway_triangle`, `try_promote_triangle`) may lock to a
> specific shape.

Stable, short, discoverable at the variant definition site. No code change.

### D2 — Cross-crate runtime fallback tests in `xlog-integration`

New file: `crates/xlog-integration/tests/test_multiway_walker_contract.rs`.

| ID | Test | What it locks |
|---|---|---|
| C1 | `executor::execute_plan` with WCOJ force-on, triangle program → row set correct AND `wcoj_triangle_dispatch_count() == 1` | Slice 1 happy path |
| C2 | `executor::execute_node` invoked directly on a `MultiWayJoin` body → produces same row set as the embedded `fallback` | Safety-net arm in `node_dispatch.rs` |
| C3 | `executor::execute_plan` with WCOJ kill switch ON, triangle program → row set correct AND counter == 0 | Fallback descent in `execute_stratum_impl` |
| C4 | `executor::execute_plan` with adaptive-only fixture below threshold → row set correct AND counter == 0 | Adaptive fall-back to `fallback` |
| **C5** | `executor::execute_non_recursive_scc` invoked directly with a rules slice carrying a triangle `MultiWayJoin` body → row set correct, no panic | **R1 — locks the path xlog-prob MC sampling actually uses** |
| C6 | **Source-contract test**: `include_str!("../../pyxlog/src/ilp.rs")` and assert the file contains the literal substring `RirNode::MultiWayJoin { fallback, .. } => walk_tmj(fallback, target_mask)` | **R3 — pyxlog cannot be linked, so we pin the source** |

C1–C5 require a CUDA runtime; they skip cleanly when one isn't available (mirrors `test_wcoj_executor_wiring.rs`'s skip pattern). C6 is a pure-Rust source-string check, runs unconditionally, will fail visibly if anyone removes the explicit `walk_tmj` arm.

C5 requires constructing rules with a synthesized `MultiWayJoin` body whose `fallback` is a real binary-join tree the executor can run end-to-end. The fixture mirrors the v0.6.2 cert tests' `tri(X,Y,Z) :- e1(X,Y), e2(Y,Z), e3(X,Z)` shape, but instead of going through `Compiler::compile`, the test programmatically wraps the body in `MultiWayJoin { inputs, slot_vars, output_columns, fallback }` to exercise the path WITHOUT relying on the promoter (which already does this for the regular path).

### D3 — `xlog-prob` MC sampling fallback test (DROPPED)

**Investigated and dropped during slice 2 implementation.** MC's
pre-compilation type inference (`xlog-prob/src/mc/buffers.rs::ensure_predicate_decls`)
rejects multi-arity rules with shared variables. The relevant
inference rule (`infer_term_scalar_type`):

* `Term::Variable(_)` → `ScalarType::U64`
* `Term::Integer(i)` with `0 <= i <= u32::MAX` → `ScalarType::U32`
* `Term::Integer(i)` with `i > u32::MAX` → `ScalarType::I64`

There is no integer literal that infers as `U64`. So a triangle
rule's `tri(X, Y, Z) :- e1(X, Y), …` (variables → U64) cannot
coexist with binary integer facts `e1(2, 3).` (constants → U32):
`ensure_predicate_decls` raises `Inconsistent predicate types for e1`
before the program ever reaches `Compiler::compile_program`.

Consequence: **no triangle program can flow through the MC
pipeline**. The promoter never produces `MultiWayJoin` in MC's
compilation path; the safety-net arm is never exercised by MC
sampling on triangle bodies in production. The R1 risk this
deliverable was meant to lock is real for the runtime path
(other callers of `Executor::execute_non_recursive_scc`) but
not for MC sampling specifically.

C5 in D2 covers `Executor::execute_non_recursive_scc` directly
with a synthesized `MultiWayJoin` body — the load-bearing R1
guard. D3 is dropped without coverage loss.

This is a pre-existing MC type-inference limitation, orthogonal
to slice 2's walker hardening goal. Logged here so future slices
that touch MC compilation know to revisit.

### D4 — Shape-agnosticism tests using synthesized IR

| Crate | Test file | Test name | Asserts |
|---|---|---|---|
| `xlog-ir` | `tests/test_multiway_rir.rs` (extend) | `referenced_relations_handles_4_inputs` | 4-input synthesized `MultiWayJoin` reports 4 distinct `RelId`s |
| `xlog-runtime` | `executor/rewrite.rs` mod tests (extend) | `collect_scan_rels_handles_4_inputs` | 4-input rewrite collects 4 rels |
| `xlog-runtime` | `executor/rewrite.rs` mod tests (extend) | `rewrite_scan_nth_handles_4_inputs_and_fallback` | 4-input rewrite touches both inputs and fallback |

These tests synthesize a non-canonical 4-input `MultiWayJoin` directly (no Compiler, no promoter — R5). They do NOT execute the synthesized IR through the runtime; they only exercise the walker arm in isolation. **Load-bearing slice 2a guard**: if a future author writes `assert!(inputs.len() == 3)` somewhere, these catch it.

Optimizer arm coverage stays in `xlog-logic::optimizer::tests` (R4) — see D5.

### D5 — Strengthen existing optimizer arm tests

`crates/xlog-logic/src/optimizer.rs` already contains four MultiWayJoin arm tests landed in slice 1:

* `optimize_returns_multiway_unchanged`
* `estimate_width_uses_output_columns_arity`
* `estimate_cost_sums_input_costs`
* `find_column_relation_returns_none_for_multiway`

These currently use a 3-input/3-output canonical-triangle fixture (`build_canonical_triangle_multiway`). **Strengthen them by extending each to also cover a synthesized 4-input/4-output `MultiWayJoin`.** The simplest shape: take the existing helper and add a sibling `build_4input_multiway()` returning a 4-input fixture with synthetic 4-arity output_columns. Each existing test asserts the v0.6.2 canonical case AND the 4-input case.

`find_column_relation_returns_none_for_multiway` continues to assert `None` for any arity — locking the slice 1 guardrail across shapes. Slice 2b (cost model) may relax this; doing so will require updating this test, which is the correct signal.

## Out of Scope (locked, restating)

* No CUDA, no kernel changes.
* No cost model.
* No promoter generalization (still triangle-only eligibility).
* No new `RirNode` variants.
* No new walker arms (every site already has one).
* No production-code behavior changes.
* No new public API.
* No env/config changes.
* No crate-type changes (pyxlog stays cdylib + test = false).
* No release tag.

## Build Sequence (as implemented)

1. Spawn worktree `feat/v065-multiway-walker-hardening` off `038e22f6`.
2. **D1**: doc-only commit on `xlog-ir`. Cargo green; no behavior change.
3. **D4**: shape-agnosticism tests in walker home crates. Cargo green.
4. **D5**: strengthen existing optimizer arm tests with 4-input fixture. Cargo green.
5. **D2**: cross-crate runtime fallback tests in `xlog-integration` (C1–C6). Cargo green.
6. **D3**: investigated and dropped (see D3 section above for rationale). C5 in D2 is the load-bearing R1 guard.
7. Workspace gate: `cargo test --workspace --release`, real_world tests under `XLOG_USE_DEVICE_RUNTIME=1`, CUDA cert suite, fmt, build.
8. FF-merge to local main. STOP. No push, no tag.

Each step is its own commit. Total: 5 commits (D1, D4, D5, D2, plan amendment).

## Acceptance

* All existing tests remain green (slice 1 contract preserved).
* All new tests pass (or skip cleanly when no CUDA, where applicable).
* Workspace build clean, no warnings.
* `cargo fmt --all -- --check` clean.
* `wcoj_triangle_dispatch_count` numbers match v0.6.2 baseline on cert tests (regression check via existing tests; slice 2 doesn't touch dispatch logic).
* No new public API. No production-code behavior changes. `pyxlog/Cargo.toml` unchanged.

## Risk and Mitigation

| Risk | Mitigation |
|---|---|
| C5 fixture (synthesized rules with `MultiWayJoin` body) requires the same upload/runtime scaffolding as the cert tests. | Reuse the `make_runtime_fixture()` + `triangle_fixture()` + `upload_binary_u32` helpers already shared across the WCOJ cert and wiring tests. Same skip-on-no-CUDA pattern. |
| C6 source-contract test is brittle: any reformat of the matching arm breaks it without indicating a real regression. | Match a tolerant substring (the unique `walk_tmj(fallback, target_mask)` call), not the entire arm verbatim. If `cargo fmt` ever reflows that line non-trivially, update the test. Cost: occasional false positive. Benefit: catches accidental deletion. |
| D3 test depends on CUDA + a non-trivial fixture. | **Materialized as: D3 dropped.** MC's pre-compilation type inference rejects multi-arity rules with shared variables (Variable→U64 vs Integer→U32 mismatch), so the path through MC never reaches the safety-net arm. C5 in D2 covers `Executor::execute_non_recursive_scc` directly with a synthesized body — the actual R1 lock. |
| D5 strengthening could reveal shape assumptions in the optimizer arms today. | Unlikely (each arm uses only `output_columns.len()` or per-`inputs` recursion), but if it does, the test failure is the load-bearing signal — treat as a real bug to fix in this slice. |

## Open Plan-Review Questions (final)

None. All previous open questions resolved by the amendment:

* C5 placement: `xlog-integration` (public Executor API) ✓
* C6 placement: source-contract in `xlog-integration` (pyxlog can't be linked) ✓
* D3 placement: `xlog-prob/tests/` using public MC API ✓
* D4 granularity: home-crate mod tests ✓
* D1 form: short stable contract, no audit table ✓
* D5 form: strengthen existing tests, do not duplicate ✓

Implementation may proceed.
