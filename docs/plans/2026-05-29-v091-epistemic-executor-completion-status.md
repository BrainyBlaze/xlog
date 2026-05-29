# v0.9.1 Epistemic Executor — Completion Status

**Date:** 2026-05-29.
**Branch:** `feat/v091-epistemic-executor-completion` (base `38ea1a34`).
**Scope:** Resolution status for the seven completion bundles defined in
`docs/plans/2026-05-28-v090-epistemic-executor-goal-bundles.md`.

This document records which epistemic semantics are now **accepted and verified on
the production/device path** and which fragments remain **typed fail-closed** (scoped
out). The cross-cutting locks (no hidden CPU fallback, no fake predicate rewriting,
EIR as the semantic boundary, raw RIR lowering rejected, typed fail-closed, real
runtime pilots) held for every accepted item.

## Completed bundles

| Bundle | Status | What landed | Evidence |
|---|---|---|---|
| EGB-02 tuple-key/bound-value membership | DONE | ground / single-bound / multi-bound / repeated-variable / anonymous-wildcard / arity-0 membership on the GPU device path; fixed a global-gate soundness bug (ground/anon/nullary modal literals were ungated) | 15 device pilots; `tuple_source_key_column_device_reads>0`; zero CPU-fallback counters |
| EGB-01 EIR candidate enumeration | DONE | candidate worlds derived from EIR (full `2^N` lattice on device), generated/propagated/tested/accepted/rejected/reason trace counts; empty-accepted-world-view distinguished from failure; resource fail-closed before partial exec | 3 device pilots; determinism reruns; pre-existing 4-literal enumeration pilot |
| EGB-07 FAEEL founded self-support | DONE | per-tuple-key foundedness; FAEEL rejects unfounded self-support; G91 self-support kept separate; precise missing-foundation diagnostics | 9 production-path pilots + G91 separation pilots |
| EGB-04 epistemic integrity constraints | DONE | `:- know g().` / `:- possible g().` / `:- not possible g().` prune candidate world views via a new GPU constraint kernel (reason code 6); constraints dropped from the reduced ordinary program (no RIR rewrite) | 7 device pilots; zero CPU-fallback; no `__xlog_constraint_*` relation |
| EGB-05 safe split semantics | DONE | split/coalesce/reject decisions explained via typed `EpistemicComponentMergeReason`; paired split-vs-unsplit equivalence; recomposition covers each source rule exactly once | 18 split pilots + device equivalence pilot |
| EGB-06 joint multi-epistemic solving | DONE | rules coupling ≥2 distinct-name epistemic predicates (any operator mix incl. negated modal) solved jointly over the candidate world view; matches unsplit | 6 device pilots + operator-combination matrix |
| EGB-03 nested modal operators | DONE (milestone scope) | nested modal forms (`know possible p()`, `not`-interspersed) recognized explicitly and rejected with a **stable typed diagnostic**; no parser-precedence accident; no fake flattening | negative pilots; stable `UnsupportedEpistemicConstruct` across all probed forms |

## Regression fixed during integration

`fix(v091): materialize nullary EDB facts as present (1 row)` — nullary facts
(`pred().`) were materialized as 0 rows (read as **absent**), pre-existing at base
`38ea1a34`. This broke ordinary nullary queries and ground/nullary modal membership
once EGB-02 stopped the old no-op gate from leaking the output row. Fixed at the
materialization layer (`create_zero_arity_buffer`); no epistemic-only special casing.

## Scoped-out fragments (remain typed fail-closed)

These are deliberate, sound over-approximations — they fail closed with typed
diagnostics, never produce unsound results:

- **Mixed per-row + global modal literal in one rule** (EGB-02): e.g. a bound-variable
  modal literal combined with a ground/anonymous/nullary one — `UnsupportedEpistemicConstruct`.
- **Nested modal semantics** (EGB-03 K1/K3): nested forms are not executable; truth
  tables and FAEEL-vs-G91 nested behavior are out of scope for this milestone.
- **Epistemic constraints beyond ground modal atoms** (EGB-04): variable-keyed
  constraints, constraints mixing relational/comparison literals with modal literals,
  constraint-only programs, and epistemic constraints in the split/GPT paths. Per-constraint
  rejection attribution is class-level (reason code 6), not per-constraint index.
- **Same-name multi-arity modal coupling** (EGB-06): `p/1` + `p/2` is unrepresentable
  in the name-keyed relation store; fails closed identically on split and unsplit paths.
- **Cross-component epistemic coupling beyond single-rule joint solving** (EGB-05/06).
- **Aggregate / compound / list / predref tuple keys** in modal atoms (EGB-02).

## Verification (feat HEAD `9f6eee45`)

- `XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-runtime --test test_epistemic_gpu_workspace --release --features epistemic-logic-tests` → **116 passed**
- `cargo test -p xlog-cli --test run_cli_tests --release test_xlog_run_epistemic_examples` → **green** (all 5 `examples/epistemic/*.xlog` through `xlog run`)
- epistemic logic suites (split / faeel / g91 / eir / world_view / gpt / examples / executable_plan) → **74 passed, 0 failed**
- `cargo test -p xlog-cuda --test set_ops_tests --release` → **35 passed** (incl. zero-arity union/diff)
- `cargo test -p xlog-cuda-tests --test certification_suite --release` → **206-cert suite passed**
- `xlog-gpu`, `xlog-cli` full suites → green
