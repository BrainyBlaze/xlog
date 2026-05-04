# W3.1 Evidence — Sorted Relation Accessors Beyond the Triangle Layout Helper

**Closes board item: W3.1 only.**
**Date:** 2026-05-04
**Branch:** `feat/w31-sorted-relation-accessors`
**Base:** `main` at `475774ef` (W2.6 closure commit).
**Plan:** `docs/plans/2026-05-04-w31-sorted-relation-accessors-plan.md`
(approved iteration 6).
**Head:** branch tip — `git log 475774ef..HEAD --oneline` is the
source of truth for the actual commit sequence.

## Summary

`CudaKernelProvider` gains two new public methods for full-row
sort + dedup at any arity ≥ 2:

* `wcoj_layout_sort_u32_recorded(input, launch_stream)` — 4-byte
  width-class. Accepts `U32` / `Symbol` per column (mixable
  within the class). Rejects mixed 4-byte + 8-byte.
* `wcoj_layout_sort_u64_recorded(input, launch_stream)` — 8-byte
  width-class. Accepts `U64` per column.

Both delegate to the existing arity-agnostic
`dedup_full_row_recorded` (which composes `sort_recorded` +
mark-unique + compact under the hood). No new CUDA `.cu` source.
No new fast-path for arity ≥ 3.

The existing arity-2 entry points
(`wcoj_layout_u32_recorded` / `wcoj_layout_u64_recorded`) and
their typed fast-path branches are **bit-identical pre-W3.1** —
the triangle / 4-cycle / project-then-layout dispatchers route
through the unchanged paths.

## Acceptance Properties (82 tests across validation + round-trip)

| Part | # tests | Location | What it locks |
|------|---------|----------|---------------|
| Width-class validation U32 | 5 | `crates/xlog-cuda/tests/test_wcoj_layout_sort_u32.rs` | arity-2 + U64 rejection; arity-3 + mixed-4+8 rejection; arity-4 + mixed `(U32, Symbol)` acceptance with schema preservation; arity-1 rejection; runtime-backed-required. |
| Width-class validation U64 | 5 | `crates/xlog-cuda/tests/test_wcoj_layout_sort_u64.rs` | Mirror at the 8-byte width-class. |
| Round-trip grid | 72 | `crates/xlog-cuda/tests/test_wcoj_layout_sort_roundtrip.rs` | 3 shapes (empty / already-sorted / unsorted+duplicated) × 4 width-class fixtures (U32 / U64 / Symbol / mixed-4-byte alternating `(U32, Symbol, ...)`) × 6 arities `{2, 3, 4, 5, 6, 7}`. Per-cell asserts: row count, sortedness, full-row uniqueness, set equality vs input modulo dedup, schema preservation bit-for-bit. |

**W3.1 acceptance total: 82 tests, 82/82 PASS.**

### Round-trip per-cell asserts (locked)

For each cell in the 72-cell grid:

1. **Row count**: `out.num_rows() == |unique(input rows)|`.
2. **Sortedness**: every output row is lex-≤ its successor.
3. **Full-row uniqueness**: no two consecutive rows are equal.
4. **Set equality**: the set of output rows equals the set of
   input rows (no row gain/loss except dedup).
5. **Schema preservation**: every output column's `ScalarType`
   matches the input schema bit-for-bit. No width-class
   promotion (U32 → U64) or Symbol → U32 collapse.

### Why the arity-7 sentinel matters

The plan locked arity coverage at `{2, 3, 4, 5, 6, 7}`. The
`= 7` cell is a sentinel — testing only through 6 risks silently
implementing a W3.2-shaped cap at the kernel level. Round-trip
PASS at arity 7 proves the new entry's `arity ≥ 2` validation +
`dedup_full_row_recorded` composition has no implicit upper
bound; runtime arity is unbounded by W3.1's surface.

### Why the mixed-4-byte fixture covers every arity

User direction in plan iteration 2: "U32 + Symbol mixing is part
of W3.1's width-class contract — don't under-test it." Mixed
fixtures (`U32, Symbol, U32, Symbol, ...`) appear at every
arity from 2 to 7, not just at one representative arity.

## Cert Test Results

```
cargo test -p xlog-cuda --release --test test_wcoj_layout_sort_u32
running 5 tests
test arity_2_rejects_u64_column ... ok
test arity_3_rejects_mixed_4byte_8byte ... ok
test arity_4_accepts_mixed_u32_symbol ... ok
test arity_below_2_rejected ... ok
test runtime_backed_required ... ok
test result: ok. 5 passed; 0 failed; 0 ignored

cargo test -p xlog-cuda --release --test test_wcoj_layout_sort_u64
running 5 tests
test arity_2_rejects_u32_column ... ok
test arity_3_rejects_mixed_8byte_4byte ... ok
test arity_4_accepts_uniform_u64 ... ok
test arity_below_2_rejected ... ok
test runtime_backed_required ... ok
test result: ok. 5 passed; 0 failed; 0 ignored

cargo test -p xlog-cuda --release --test test_wcoj_layout_sort_roundtrip
running 72 tests
... (54 cells for U32 / Symbol / Mixed4 across arities 2-7)
... (18 cells for U64 across arities 2-7)
test result: ok. 72 passed; 0 failed; 0 ignored
```

## Workspace Tally

| Suite | PASS | FAIL | IGN | Pre-W3.1 baseline |
|-------|------|------|-----|-------------------|
| Workspace tests (default features, lib + integration) — `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` | **1957** | 0 | 17 | 1875 (post-W2.6) |
| W3.1 layout sort u32 | 5 | 0 | 0 | (new) |
| W3.1 layout sort u64 | 5 | 0 | 0 | (new) |
| W3.1 round-trip grid | 72 | 0 | 0 | (new) |
| Existing layout u32 (arity-2, unchanged) | 9 | 0 | 0 | 9 |
| Existing layout u64 (arity-2, unchanged) | 6 | 0 | 0 | 6 |
| W2.1 cert | 11 | 0 | 0 | 11 |
| W2.3 trace cert (`recursive-stats-trace`) | 10 | 0 | 0 | 10 |
| W2.4 cert | 3 | 0 | 0 | 3 |
| W2.6 cert | 7 | 0 | 0 | 7 |
| Slice-4 cert | 6 | 0 | 0 | 6 |
| CUDA certification suite (`run_full_certification` meta-cert, 206 sub-tests) | 1 | 0 | 0 | 1 |
| `cargo fmt --check --all` | clean | — | — | clean |

**Workspace pass-count delta: +82** (1875 → 1957) — exactly
matches the locked plan's `+82` acceptance grid (10 validation
+ 72 round-trip).

## Code-Level Changes

| File | Change |
|------|--------|
| `crates/xlog-cuda/src/provider/wcoj.rs` | Add `wcoj_layout_sort_u32_recorded` + `wcoj_layout_sort_u64_recorded` public methods on `CudaKernelProvider`. Total +146 lines (doc-comments + bodies). No existing function modified. |
| `crates/xlog-cuda/tests/test_wcoj_layout_sort_u32.rs` | NEW. 5 width-class validation tests for the 4-byte entry. |
| `crates/xlog-cuda/tests/test_wcoj_layout_sort_u64.rs` | NEW. 5 width-class validation tests for the 8-byte entry. |
| `crates/xlog-cuda/tests/test_wcoj_layout_sort_roundtrip.rs` | NEW. 72 round-trip tests covering the 3 shapes × 4 width-class fixtures × 6 arities grid. |
| `docs/plans/2026-05-04-w31-sorted-relation-accessors-plan.md` | NEW (committed as branch commit #1 per iteration-2 user direction; not on main). |
| (this file) | NEW. Evidence README. |

**No changes** to:
* `wcoj_dispatch.rs` (triangle + 4-cycle dispatchers — still call the original arity-2 entry points).
* `wcoj_project.rs` (project-then-layout pipeline — unchanged 2-arity call site).
* `relational.rs` (`sort_recorded` and `dedup_full_row_recorded` were already arity-agnostic; W3.1 only consumes them).
* Any `.cu` file under `crates/xlog-cuda/cuda/`.
* The two existing arity-2 fast-path checkers `try_wcoj_layout_fast_path_u32` / `_u64`.

## Decision Mapping (D1–D6 from approved plan iteration 6)

| Decision | Locked answer | Rationale |
|----------|---------------|-----------|
| D1 | New entry points; existing arity-2 helpers stay bit-identical. | Minimal blast radius; production triangle / 4-cycle hot path is untouched. |
| D2 | Full-row keys (internally `0..arity`). | Matches the existing 2-arity helpers' "every column is a key" semantic; removes `key_cols` parameter from caller's surface. |
| D3 | Uniform width-class per call (4-byte = U32 + Symbol mixable; 8-byte = U64). Reject mixed 4+8. | Preserves Symbol parity with U32 (existing kernel contract). Mixed-width WCOJ semantics are out of scope for W3.1. |
| D4 | No fast-path for arity ≥ 3. Out of scope for W3.1 with no closure credit. | Correctness-first; the perf path can be opened on its own merits at a later time, with no W3.1-implied commitment. |
| D5 | Cert at arities `{2, 3, 4, 5, 6, 7}`. | Arity-7 sentinel proves no silent W3.2-shaped cap. |
| D6 | Branch from main `475774ef`; plan as branch commit #1. No runtime rerouting. No `.cu` changes. No push, no tag. No self-mark DONE. | Scope discipline; user-gated closure approval. |

## Audit (Step 1, no code change)

Five findings from the read-only audit, all confirmed:

1. `sort_recorded` (`crates/xlog-cuda/src/provider/relational.rs:5309`) accepts any arity input, any `key_cols`, with per-key-column type ∈ `{U32, Symbol, U64}`. Arity-agnostic.
2. `dedup_full_row_recorded` (`crates/xlog-cuda/src/provider/relational.rs:5601`) owns runtime-backed check, stream resolution, n==0 short-circuit, n==1 short-circuit, per-column type validation, and the `sort_recorded(input, &(0..arity).collect(), launch_stream)` composition. Arity-agnostic.
3. Existing arity-2 layout entries `wcoj_layout_u32_recorded` / `_u64_recorded` (`wcoj.rs:229`, `wcoj.rs:1224`) do NOT call `stream_pool().resolve(...)` themselves — confirmed by `grep`. Stream resolution is delegated downstream through the fast-path checker and `dedup_full_row_recorded`. New W3.1 entries inherit that exact pattern.
4. `wcoj_dispatch.rs` and `wcoj_project.rs` call only the existing arity-2 entry points. New W3.1 entries have no current callers (intentional — W3.1 is API surface only).
5. No CUDA `.cu` file under `crates/xlog-cuda/cuda/` mentions an arity bound > 4 that W3.1 would silently widen. Sort kernels operate per-column; arity is host-side iteration.

## Process Rule Compliance

* Process rule #1: this slice does **not** self-mark W3.1 DONE.
* Process rule #2: every commit references W3.1.
* Process rule #3: plan opens with "Closes W3.1 only."
* Process rule #5: no `v0.6.6` references; no punt-to-later wording — out-of-scope items are owned by W3.2+ board items, named at the point of reference.
* Process rule #6: no push, no tag.

## Closure Board Update Proposal

After explicit user "mark W3.1 DONE" approval, a follow-up
commit applies:

* `docs/v065-closure-board.md` — W3.1 status `OPEN → DONE`,
  status tally `DONE: 5 → 6; OPEN: 14 → 13` (verify counts at
  apply-time).
* `docs/v065-closure-board.md` "Completed" section gets a W3.1
  entry referencing the branch commits (full sequence via
  `git log 475774ef..HEAD --oneline` — README does not pin
  commit hashes to avoid amend-circularity).
* FF-merge `feat/w31-sorted-relation-accessors` into local
  `main`. No tag, no push.
