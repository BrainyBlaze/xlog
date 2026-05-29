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
| EGB-04 epistemic integrity constraints | DONE | `:- know g().` / `:- possible g().` / `:- not possible g().` prune candidate world views via a GPU constraint kernel; constraints dropped from the reduced ordinary program (no RIR rewrite). **K2 met:** a parallel `constraint_violation_index` device buffer records *which* constraint fired per candidate (reason code 6 unchanged), surfaced as `result.semantic_trace.constraint_violation_indices`. | 8 device pilots incl. `egb04_constraint_specific_reason_identifies_firing_constraint` (asserts the specific firing index); zero CPU-fallback |
| EGB-05 safe split semantics | DONE | split/coalesce/reject decisions explained via typed `EpistemicComponentMergeReason`; paired split-vs-unsplit equivalence; recomposition covers each source rule exactly once | 18 split pilots + device equivalence pilot |
| EGB-06 joint multi-epistemic solving | DONE | rules coupling ≥2 distinct-name epistemic predicates (any operator mix incl. negated modal) solved jointly over the candidate world view; matches unsplit | 6 device pilots + operator-combination matrix |
| EGB-03 nested modal operators | DONE (milestone scope) | nested modal forms (`know possible p()`, `not`-interspersed) recognized explicitly and rejected with a **stable typed diagnostic**; no parser-precedence accident; no fake flattening | negative pilots; stable `UnsupportedEpistemicConstruct` across all probed forms |

## Regression fixed during integration

`fix(v091): materialize nullary EDB facts as present (1 row)` — nullary facts
(`pred().`) were materialized as 0 rows (read as **absent**), pre-existing at base
`38ea1a34`. This broke ordinary nullary queries and ground/nullary modal membership
once EGB-02 stopped the old no-op gate from leaking the output row. Fixed at the
materialization layer (`create_zero_arity_buffer`); no epistemic-only special casing.

## Category A — In-spec typed fail-closed (REQUIRED by the goal, NOT debt)

These are mandated by the goal's own "Expected Rejected Behavior" sections and by
cross-cutting lock #5 ("Typed fail-closed behavior remains required ... not silent
partial execution"). They are correct rejection-by-design, verified by negative
pilots — accepting them would violate the no-fake / no-CPU-fallback locks.

- **Nested modal semantics** — EGB-03 Expected Rejected: *"If the accepted fragment
  remains single-level for a milestone, nested forms must continue to fail closed
  with typed diagnostics"*; Bundle Ordering: EGB-03 lands only after single-level is
  complete. Truth tables / FAEEL-vs-G91 nested behavior are out of scope by design.
- **Aggregate / compound / list / predref modal tuple keys** — EGB-02 Expected
  Rejected lists these verbatim as fail-closed.
- **Epistemic constraints with variable tuple keys, nested-modal bodies, or
  CPU-only world-view scans** — EGB-04 Expected Rejected lists these verbatim.
- **Unsafe same-name multi-arity modal coupling** (`p/1` + `p/2` unsafely bound) —
  EGB-06 Expected Rejected: unsupported-tuple-key joint conditions fail closed.
  Safe (bound) cross-arity coupling IS accepted (EGB-05/06).

## Category B — Genuine follow-up (NOT goal-mandated; tracked, not "done")

- **B1 — EGB-04.K2 constraint-specific reasons — CLOSED (commit `e39bcd33`).** The
  kernel now records the firing constraint index in a parallel
  `constraint_violation_index` buffer (reason code 6 unchanged), surfaced as
  `result.semantic_trace.constraint_violation_indices`; verified by
  `egb04_constraint_specific_reason_identifies_firing_constraint`. KPI met.
- **B2 — Mixed per-row + global modal literal in one rule (EGB-02).** NOT in EGB-02's
  Expected Rejected list — introduced by the EGB-02 implementation to replace a
  *silently wrong* result (the global-gate bug) with sound fail-closed, as lock #1
  forced. Sound, but a new boundary the goal did not explicitly require; either
  implement the mixed path or keep fail-closed by explicit decision.
- **B3 — Cross-component epistemic coupling beyond single-rule joint solving**
  (EGB-05/06): tracked as future work, currently fail-closed.

## Verification (feat HEAD `e1c9cb07`)

- `XLOG_USE_DEVICE_RUNTIME=1 cargo test -p xlog-runtime --test test_epistemic_gpu_workspace --release --features epistemic-logic-tests` → **117 passed** (incl. EGB-04.K2 specific-constraint test)
- `cargo test -p xlog-cli --test run_cli_tests --release test_xlog_run_epistemic_examples` → **green** (all 5 `examples/epistemic/*.xlog` through `xlog run`)
- epistemic logic suites (split / faeel / g91 / eir / world_view / gpt / examples / executable_plan) → **74 passed, 0 failed**
- `cargo test -p xlog-cuda --test set_ops_tests --release` → **35 passed** (incl. zero-arity union/diff)
- `cargo test -p xlog-cuda-tests --test certification_suite --release` → **206-cert suite passed**
- `xlog-gpu`, `xlog-cli` full suites → green
