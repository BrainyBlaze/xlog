# v0.6.5 Slice 4 — Recursive WCOJ Evidence

**Date:** 2026-05-03
**Branch:** `feat/v065-recursive-wcoj`
**Base:** `main` at `616ec628` (slice 3 amendment)
**Plan:** `docs/plans/2026-05-03-v065-slice4-recursive-wcoj-plan.md`

## Slice Summary

WCOJ triangle and 4-cycle dispatch now fires inside recursive
SCCs, gated per-rule on the count of body Scans whose RelId
resolves to the rule's head SCC predicate set:

| Recursive Scans in body | Slice 4 behavior          |
|-------------------------|---------------------------|
| 0 (stable rule)         | Promote → WCOJ on seeding |
| 1 (linear recursion)    | Promote → WCOJ per variant |
| ≥ 2 (multi-recursion)   | Skip — defer to slice 4.2 |

Counter semantics: `wcoj_*_dispatch_count` increments per
successful WCOJ kernel result — once per (rule, iteration,
variant) on dispatch.

## Acceptance Gates

All seven gates from the slice 4 plan are met.

| # | Gate | Status |
|---|------|--------|
| 1 | Stable triangle in recursive SCC dispatches WCOJ on seeding (counter == 1) | PASS — `stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding` |
| 2 | Linear-recursive triangle dispatches per iteration; row set matches binary-join | PASS — `linear_recursive_triangle_dispatches_on_seeding_and_per_variant` (counter ≥ 2; row set matches binary-join) |
| 3 | Multi-recursive triangle skips WCOJ (counter == 0) and matches binary-join | PASS — `multirec_triangle_skips_wcoj_and_matches_binary_join` |
| 4a | Stable 4-cycle in recursive SCC dispatches WCOJ on seeding (counter == 1) | PASS — `stable_4cycle_in_recursive_scc_dispatches_wcoj_on_seeding` |
| 4b | Linear-recursive 4-cycle dispatches per iteration; row set matches binary-join | PASS — `linear_recursive_4cycle_dispatches_on_seeding_and_per_variant` (counter ≥ 2; row set matches binary-join) |
| 5 | Adaptive classifier makes same decision in recursive vs non-recursive arm | PASS — `adaptive_dispatches_in_recursive_scc_on_superhub` |
| 6 | Stale doc claims about "recursive WCOJ excluded" are removed | PASS — `wcoj_dispatch.rs` header + `recursive.rs:106-110` comment + `promote.rs` header all updated |
| 7 | Workspace + CUDA cert + WCOJ regression no regression | PASS — see "Workspace tally" below |

## Cert Test Results

```
cargo test -p xlog-integration --release --test test_wcoj_recursive_dispatch
running 6 tests
test adaptive_dispatches_in_recursive_scc_on_superhub ... ok
test stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding ... ok
test linear_recursive_triangle_dispatches_on_seeding_and_per_variant ... ok
test linear_recursive_4cycle_dispatches_on_seeding_and_per_variant ... ok
test stable_4cycle_in_recursive_scc_dispatches_wcoj_on_seeding ... ok
test multirec_triangle_skips_wcoj_and_matches_binary_join ... ok
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured
```

Per-test counter assertions:

| Test | Counter | Reference rows match |
|------|---------|----------------------|
| `stable_triangle_in_recursive_scc_dispatches_wcoj_on_seeding` | `wcoj_triangle_dispatch_count() == 1` | binary-join row-for-row |
| `stable_4cycle_in_recursive_scc_dispatches_wcoj_on_seeding`   | `wcoj_4cycle_dispatch_count() == 1`   | binary-join row-for-row |
| `linear_recursive_triangle_dispatches_on_seeding_and_per_variant` | `wcoj_triangle_dispatch_count() >= 2` (seeding + ≥ 1 variant) | binary-join row-for-row |
| `linear_recursive_4cycle_dispatches_on_seeding_and_per_variant`   | `wcoj_4cycle_dispatch_count() >= 2`   (seeding + ≥ 1 variant) | binary-join row-for-row |
| `multirec_triangle_skips_wcoj_and_matches_binary_join`         | `wcoj_triangle_dispatch_count() == 0` | binary-join row-for-row |
| `adaptive_dispatches_in_recursive_scc_on_superhub`             | `wcoj_triangle_dispatch_count() >= 1` | n/a (counter assertion) |

## Linear-recursive coverage

End-to-end coverage of count == 1 is provided by the two
linear-recursive cert tests above. Each test:

1. Builds a fixture with one in-SCC body Scan (`e1` in the
   triangle case, `e1` in the 4-cycle case) fed back via a
   dedicated recursive rule (`e1(...) :- tri(...)` / `e1(...)
   :- cyc(...)`). Other body atoms are extensional.
2. Designs the EDB so the recursive chain advances at least
   once: seeding produces an initial result; the recursive
   `e1` rule projects new `e1` rows from that result; the
   promoted triangle/4-cycle body is variant-rewritten to use
   `e1_delta` and dispatches WCOJ; that variant produces NEW
   result rows; the chain may continue or terminate per data.
3. Asserts `wcoj_*_dispatch_count() >= 2` — strictly more than
   the seeding-only case — and asserts the row set matches
   the binary-join reference.

The fixture chain (triangle): `e1_seed(1,2)` →
`tri(1,2,3)` (seeding) → recursive `e1(1,3)` → variant fires
WCOJ on `e1_delta=(1,3)` ⋈ `e2` ⋈ `e3` → `tri(1,3,4)` (new).

Promoter-side coverage in `xlog-logic::promote::tests`:

```
test promote::tests::promotes_linear_recursive_triangle ... ok
test promote::tests::promotes_linear_recursive_4cycle ... ok
test promote::tests::skips_multirec_triangle_in_recursive_scc ... ok
test promote::tests::skips_multirec_4cycle_in_recursive_scc ... ok
test promote::tests::promotes_stable_triangle_in_recursive_scc ... ok
test promote::tests::promotes_stable_4cycle_in_recursive_scc ... ok
test promote::tests::promotes_linear_rec_and_non_rec_sccs_in_mixed_plan ... ok
```

## Workspace Tally

| Crate                | PASS | FAIL | IGN |
|----------------------|------|------|-----|
| `xlog-cuda`          | 507  | 0    | 6   |
| `xlog-runtime`       | 125  | 0    | 2   |
| `xlog-logic`         | 503  | 0    | 5   |
| `xlog-integration`   | 116  | 0    | 0   |
| `xlog-cuda-tests` (cert) | 1 (cert pass) | 0 | 0 |
| Other crates (sum)   | 503  | 0    | 4   |
| **Workspace**        | **1755+1 cert** | **0** | **17** |

Slice 1–3 WCOJ regression confirmed bit-identical:

* 69 cuda WCOJ tests pass (slice 1–3 baseline)
* 59 integration WCOJ tests pass (slice 1–3 baseline)
* 39 xlog-runtime lib WCOJ tests pass

## Code-Level Changes

| File | Change |
|------|--------|
| `crates/xlog-logic/src/promote.rs` | `promote_multiway(plan, rel_ids)` signature; `recursive_scan_count` helper; per-rule gate (`count <= 1`); 6 new + 1 renamed tests |
| `crates/xlog-logic/src/compile.rs` | Pass `self.lowerer.rel_ids()` into `promote_multiway` |
| `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` | Add `try_dispatch_wcoj_*_on_body(&RirNode)` body-keyed entry points; rule-keyed wrappers stay byte-identical |
| `crates/xlog-runtime/src/executor/recursive.rs` | Add `execute_wcoj_or_fallback_node` helper; wire into seeding pass + per-variant evaluation |
| `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs` | New cert file — 4 tests covering stable triangle, stable 4-cycle, multi-rec fallback, adaptive parity |

## Risks / Out-of-Slice

* **Q1 — Per-iteration classifier cost.** The recursive arm
  runs the classifier on each iteration with unchanged buffers.
  Documented as a perf opportunity for slice 5 / v0.6.6. No
  phase-timing evidence in slice 4 (correctness-only scope per
  user lock).
* **Multi-recursive WCOJ** (count ≥ 2) — deferred to slice 4.2.
  Per-variant union+dedup interaction with WCOJ is the gating
  open question. The slice 4 promoter gate enforces the
  exclusion at the IR level so the recursive engine never sees
  a multi-rec MultiWayJoin.

## Test-Fixture Friction (informational)

The recursive engine's per-iteration union compares schemas
strictly. EDB uploads must use column names matching the
compiler's `c0/c1` convention, AND the program needs explicit
`pred name(u32, ...)` declarations to anchor U32 typing.

Inline facts also work for typing but perturb the optimizer's
cardinality estimates: with inline facts, the optimizer chose a
right-deep `Project { Join { Scan, Join { Scan, Scan } } }`
shape that the slice 1 promoter doesn't recognize (it expects
left-deep `Project { Join { Join { Scan, Scan }, Scan } }`).
The cert tests in this slice use `pred` declarations to keep
the lowering shape canonical.

This is documented at the test-file header for future slice
authors.
