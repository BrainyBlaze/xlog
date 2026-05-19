# W3.1 — Sorted Relation Accessors Beyond the Triangle Layout Helper

**Closes W3.1 only.** General-arity WCOJ kernel and runtime
rerouting are **out of scope** for W3.1; those concerns are owned
by W3.2+ board items. No CUDA `.cu` source changes. No push, no
tag. No `v0.6.6` references. Plan-first; no implementation until
iteration is approved by the user.

**Plan iteration:** 6 (text fix per user iteration-5 review).
**Base:** `main` at `475774ef` (W2.6 closure commit).
**Worktree:** `.worktrees/w31-sorted-relation-accessors`.
**Branch:** `feat/w31-sorted-relation-accessors`.
**Closure board:** `docs/v065-closure-board.md:85` (W3.1 row, OPEN).

## Acceptance line (locked from board)

> xlog-cuda cert: empty / already-sorted / unsorted+duplicated
> all round-trip via the new accessors at u32, u64, Symbol widths.

Plan extends the arity coverage per the user's iteration-0
direction: certify at **arities {2, 3, 4, 5, 6, 7}** so the >6
sentinel proves runtime arity is unbounded (no silent W3.2-shaped
cap). Three round-trip shapes (empty / already-sorted /
unsorted+duplicated) × **four width-class fixtures** (U32, U64,
Symbol, mixed-4-byte alternating `(U32, Symbol, ...)`) × six
arities = **72 cert cells**; combined with 10 width-class
validation tests, the W3.1 acceptance total is **82 tests**.
Plan's Part B locks each cell.

## Direction (locked with user, iteration 0)

| # | Decision | Locked answer |
|---|----------|---------------|
| D1 | API shape | **New entry point.** Add `wcoj_layout_sort_u32_recorded` (4-byte width-class — accepts U32 / Symbol per column, mixed within the class allowed) and `wcoj_layout_sort_u64_recorded` (8-byte width-class). Existing `wcoj_layout_u32_recorded` / `wcoj_layout_u64_recorded` stay bit-identical for current triangle / 4-cycle callers — no behavioral drift on the dispatcher hotpath. |
| D2 | Key-column policy | **Full-row keys only.** Internally derive `key_cols = (0..arity).collect()`. No caller-supplied `key_cols` parameter. |
| D3 | Width-class policy | **Uniform width-class per call**, not exact-uniform-type. 4-byte class = U32 + Symbol (mixable within); 8-byte class = U64. Reject mixed 4-byte + 8-byte in one relation. Preserves Symbol parity without opening mixed-width WCOJ semantics. |
| D4 | Fast-path | **No new fast-path for arity ≥ 3.** Existing arity-2 fast-path (`try_wcoj_layout_fast_path_u32` / `_u64`) stays untouched — exclusively reachable from the existing arity-2 entry points. Generic path is correctness-first via `sort_recorded` + `dedup_full_row_recorded` only. Arity ≥ 3 fast-path is **out of scope for W3.1 and receives no closure credit in this slice.** No future-item invention. |
| D5 | Test arity coverage | **{2, 3, 4, 5, 6, 7}.** The arity=7 sentinel proves the implementation does not silently cap at 6 (the W3.2 k-bound). |
| D6 | Scope discipline | No runtime rerouting; general-arity dispatch and W3.2's kernel template are **out of scope, owned by W3.2+ board items**; no `.cu` source changes; no push; no tag; no self-mark DONE; no `v0.6.6` references; branch from `475774ef`; the plan commit lands as branch commit #1 in `feat/w31-sorted-relation-accessors` (not on `main`). |

## Code-level surface (read-only audit, no edits in this plan)

What already exists and is **untouched** by W3.1:

* `crates/xlog-cuda/src/provider/relational.rs:5309` — `sort_recorded(input, key_cols, launch_stream)` — accepts any arity input, any `key_cols`, U32 / Symbol / U64 per-column. Already arity-agnostic; W3.1 reuses verbatim.
* `crates/xlog-cuda/src/provider/relational.rs:5601` — `dedup_full_row_recorded(input, launch_stream)` — composes sort + mark-unique + compact. Already arity-agnostic; W3.1 reuses verbatim.
* `crates/xlog-cuda/src/provider/wcoj.rs:229` — `wcoj_layout_u32_recorded` (arity-2 entry point with fast-path branch).
* `crates/xlog-cuda/src/provider/wcoj.rs:1224` — `wcoj_layout_u64_recorded` (arity-2 entry point with fast-path branch).
* `crates/xlog-cuda/src/provider/wcoj.rs:2939` / `:3078` — `try_wcoj_layout_fast_path_u32` / `_u64` (arity-2-specific fast-path checkers).
* `wcoj_dispatch.rs:1079-1093` (triangle dispatcher, 3 calls) and the 4-cycle dispatcher (4 calls) — only callers of the arity-2 entry points. Untouched.
* `wcoj_project.rs:9, 54` — project → layout pipeline; layout call site is arity-2. Untouched.

What W3.1 **adds** (no existing code modified):

* New `wcoj_layout_sort_u32_recorded(input, launch_stream)` and
  `wcoj_layout_sort_u64_recorded(input, launch_stream)` in
  `crates/xlog-cuda/src/provider/wcoj.rs`. Each runs validation
  in this order, then delegates:
  1. Validates manager is runtime-backed (same error contract as
     existing entry points).
  2. Validates `input.arity() >= 2` (no upper bound).
  3. For each of `0..arity`, validates the column's `ScalarType`
     is in the entry point's width-class:
     * `_u32_` variant: `U32` or `Symbol`.
     * `_u64_` variant: `U64`.
  4. Delegates unconditionally to:
     ```
     dedup_full_row_recorded(input, launch_stream)
     ```
     which has its own `n == 0` short-circuit returning
     `create_empty_buffer(input.schema.clone())` and which
     internally calls `sort_recorded(input, &(0..arity).collect::<Vec<_>>(), launch_stream)`
     for `n > 0` — full-row sort + dedup. The output's schema is
     preserved bit-for-bit from input.

**Stream resolution is owned by `dedup_full_row_recorded`**, not
the new entry points. W3.1 entries do NOT call
`runtime.stream_pool().resolve(launch_stream)` themselves —
duplicating that resolution would fork the
"launch_stream does not resolve" error message between two
layers. Single source of truth for stream-resolution errors is
the existing `dedup_full_row_recorded` (and beneath it,
`sort_recorded`). The cost: the new entries' error messages for
runtime-not-backed and arity / width-class violations are
W3.1-prefixed, but stream-resolution failures will surface with
the existing helper's prefix. This is the same trade existing
arity-2 layout entries already make (per audit of
`wcoj_layout_u32_recorded` / `_u64_recorded` — neither
re-resolves the stream itself; resolution is delegated through
`dedup_full_row_recorded`).

W3.1's new entry points do **not** duplicate the `n == 0` check
themselves; doing so would fork the empty-buffer semantics
between two layers. The single source of truth is
`dedup_full_row_recorded`.

W3.1 does **not** add a fast-path branch in the new generic
entries. The arity-2 fast-path is a separate code path reachable
only via the existing arity-2 entry points.

## Step plan (9 steps)

### Step 1 — Audit (read-only)

Confirm via `grep` / `Read`:

* `sort_recorded` validates `key_cols` indices are in-bounds and
  per-column types are `U32 | Symbol | U64` — no other type ever
  reaches the radix kernel.
* `dedup_full_row_recorded` composes `sort_recorded(input, &(0..arity).collect(), stream)`
  internally — full-row keys is its semantic.
* Existing 2-arity entry points and their fast-path branches do
  NOT share a normalize-arity helper that we'd accidentally
  perturb.
* `wcoj_dispatch.rs` and `wcoj_project.rs` call only
  `wcoj_layout_u32_recorded` / `wcoj_layout_u64_recorded` — never
  reach for a sort_recorded directly. W3.1's new entry points
  have no current callers (intentional — W3.1 is the API surface
  only; W3.2 will be the first consumer).
* No CUDA `.cu` file under `crates/xlog-cuda/cuda/` mentions an
  arity bound > 4 that W3.1 would silently widen. Sort kernels
  operate per-column; arity is a host-side iteration count.

**Output**: a 5-bullet audit note in the evidence README confirming
each finding (no code change in step 1).

### Step 2 — `wcoj_layout_sort_u32_recorded` entry point

Add to `crates/xlog-cuda/src/provider/wcoj.rs`. Doc-comment must:

* Open with "W3.1 — generic full-row sort+dedup for relations of
  any arity ≥ 2 in the 4-byte width-class (U32, Symbol, mixable
  within the class)."
* Reference the locked design (D1: new entry, not in-place
  generalization) and explicitly state that the existing
  `wcoj_layout_u32_recorded` is unchanged for arity-2 callers.
* Document the n==0 short-circuit, validation order (runtime →
  arity ≥ 2 → per-column width-class → delegate), and the
  composition (`dedup_full_row_recorded` only, no fast-path).
  Stream resolution is owned by `dedup_full_row_recorded` and is
  not in this entry point's validation list.

Signature:
```rust
pub fn wcoj_layout_sort_u32_recorded(
    &self,
    input: &CudaBuffer,
    launch_stream: StreamId,
) -> Result<CudaBuffer>
```

Body skeleton (locked):
```rust
if self.memory().runtime().is_none() {
    return Err(XlogError::Kernel(
        "wcoj_layout_sort_u32_recorded requires a runtime-backed \
         GpuMemoryManager (constructed via with_runtime)".into(),
    ));
}
// stream resolution defers to dedup_full_row_recorded which
// re-resolves from the same StreamId — single source of truth
// for the resolution error.

if input.arity() < 2 {
    return Err(XlogError::Kernel(format!(
        "wcoj_layout_sort_u32_recorded: input must have arity >= 2, got {}",
        input.arity()
    )));
}
for col_idx in 0..input.arity() {
    let ty = input.schema.column_type(col_idx).ok_or_else(|| {
        XlogError::Kernel(format!(
            "wcoj_layout_sort_u32_recorded: column {} type missing",
            col_idx
        ))
    })?;
    if !matches!(ty, ScalarType::U32 | ScalarType::Symbol) {
        return Err(XlogError::Kernel(format!(
            "wcoj_layout_sort_u32_recorded: column {} must be U32 or Symbol \
             (4-byte width-class), got {:?}",
            col_idx, ty
        )));
    }
}
// dedup_full_row_recorded handles n==0 → create_empty_buffer
// internally; we do NOT short-circuit before it because that
// would duplicate semantics. Recorder discipline preserved.
self.dedup_full_row_recorded(input, launch_stream)
```

### Step 3 — `wcoj_layout_sort_u64_recorded` entry point

Mirror of step 2 with `ScalarType::U64` validation. Same doc-comment
template (s/4-byte/8-byte/, s/U32 or Symbol/U64/).

### Step 4 — Manifest / re-export check

Verify the new entry points are reachable from `xlog-cuda`'s public
surface (`crates/xlog-cuda/src/lib.rs`'s re-exports if applicable;
otherwise reachable via `CudaKernelProvider` directly). No public
manifest registration — these are pure provider methods that
delegate to existing manifest-registered kernels.

### Step 5 — Cert: width-class validation tests

`crates/xlog-cuda/tests/test_wcoj_layout_sort_u32.rs`:

* `arity_2_rejects_u64_column` — explicit width-class rejection.
* `arity_3_rejects_mixed_4byte_8byte` — relation `(U32, U64, U32)` rejected.
* `arity_4_accepts_mixed_u32_symbol` — relation `(U32, Symbol, U32, Symbol)` accepted.
* `arity_below_2_rejected` — arity-1 (single-column) rejected.
* `runtime_backed_required` — non-runtime manager returns the `with_runtime` error.

`crates/xlog-cuda/tests/test_wcoj_layout_sort_u64.rs`:

* `arity_2_rejects_u32_column` — converse of the above.
* `arity_3_rejects_mixed_8byte_4byte` — relation `(U64, U32, U64)` rejected.
* `arity_4_accepts_uniform_u64` — relation `(U64, U64, U64, U64)` accepted.
* `arity_below_2_rejected`.
* `runtime_backed_required`.

**Total step 5: 10 tests.** Error-message asserts use `contains`
on stable semantic fragments only (e.g. `contains("must be U32 or
Symbol")`, `contains("4-byte width-class")`,
`contains("with_runtime")`, `contains("arity >= 2")`). Exact
error-tail matches are churn-prone and add no meaningful contract
value beyond the semantic fragment.

### Step 6 — Cert: round-trip Part B grid (empty / sorted / unsorted+dup)

`crates/xlog-cuda/tests/test_wcoj_layout_sort_roundtrip.rs`. Three
shapes × **four** width-class fixtures × six arities = **72 tests**.

Per-arity fixture builder:

```rust
fn build_input_u32(arity: usize, kind: InputKind) -> CudaBuffer;          // (U32, U32, ..., U32)
fn build_input_u64(arity: usize, kind: InputKind) -> CudaBuffer;          // (U64, U64, ..., U64)
fn build_input_symbol(arity: usize, kind: InputKind, sym_table: &SymbolTable) -> CudaBuffer;
                                                                            // (Symbol, Symbol, ..., Symbol)
fn build_input_mixed_4byte(arity: usize, kind: InputKind, sym_table: &SymbolTable) -> CudaBuffer;
                                                                            // alternating (U32, Symbol, U32, Symbol, ...)
```

Symbol fixtures use **real symbol-table-allocated identifiers**
(via the project's `SymbolTable` / equivalent — exact API to be
confirmed in step 1's audit). Allocate ≥ `2 × arity` distinct
symbols per fixture so the AlreadySorted and
UnsortedWithDuplicates seeds have enough vocabulary; arbitrary u32
bits are not sufficient because Symbol parity is the contract
under test.

Where `InputKind` ∈ {`Empty`, `AlreadySorted`, `UnsortedWithDuplicates`}:

* **Empty**: 0 rows, schema set per width-class.
* **AlreadySorted**: rows pre-sorted lex by full row, no duplicates. Locked seed pattern: deterministic `(i, i+1, i+2, ...)` style sequences extended/truncated for the per-arity column count; for Symbol, the i-th symbol in allocation order; for mixed-4-byte, alternating per-column type.
* **UnsortedWithDuplicates**: same row content as AlreadySorted, plus 2 duplicate rows + 2 rows in reverse order. Locked seed.

Per-cell test asserts:
1. The output's `num_rows()` matches the expected (deduped, sorted) row count.
2. Each output row is lex-≤ its successor (sortedness).
3. No two consecutive rows are equal (full-row uniqueness).
4. The set of output rows equals the set of input rows (semantics
   preserved — no row gain/loss except dedup).
5. Each output column's `ScalarType` is preserved exactly from the
   input schema (no width-class promotion / collapse).

Schemas, all six arities {2, 3, 4, 5, 6, 7}:
* **U32 fixtures**: every column `ScalarType::U32`.
* **U64 fixtures**: every column `ScalarType::U64`.
* **Symbol fixtures**: every column `ScalarType::Symbol`, contents are real `SymbolTable`-allocated IDs.
* **Mixed-4-byte fixtures**: alternating `(U32, Symbol, U32, Symbol, ...)`, i.e. column `i` has type `U32` if `i` even, `Symbol` if `i` odd. Symbol IDs come from the same allocator as the Symbol-only fixtures so the comparison kernel sees real identifiers.

For mixed-4-byte at arity 2: `(U32, Symbol)`. At arity 3:
`(U32, Symbol, U32)`. ... arity 7: `(U32, Symbol, U32, Symbol, U32, Symbol, U32)`.
Mixed coverage at every arity (per user direction — under-testing
the U32 + Symbol mix would leave the width-class contract under-
exercised).

**Total step 6: 72 round-trip tests** = 3 shapes × 4 width-classes
× 6 arities. Routes:
* U32 + Symbol + Mixed-4-byte → `wcoj_layout_sort_u32_recorded`.
* U64 → `wcoj_layout_sort_u64_recorded`.

### Step 7 — Workspace gate

* `cargo fmt --check --all` clean.
* `cargo test -p xlog-cuda --release --test test_wcoj_layout_sort_u32` (5/5).
* `cargo test -p xlog-cuda --release --test test_wcoj_layout_sort_u64` (5/5).
* `cargo test -p xlog-cuda --release --test test_wcoj_layout_sort_roundtrip` (72/72).
* `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` — pass count must increase by **+82** (the new W3.1 tests). 0 fail. Ignored unchanged. (Symbolic delta only — the global pre-W3.1 baseline is reported in evidence at execution time but not pinned in the assertion, since unrelated test additions on `main` would otherwise drift the gate.)
* `cargo test -p xlog-cuda-tests --test certification_suite --release` — `run_full_certification` PASS (1/1, the 206-sub-test meta-cert).
* `cargo test -p xlog-cuda --release --test test_wcoj_layout_u32` and `test_wcoj_layout_u64` — pass unchanged (no behavioral drift on existing arity-2 entry points).
* `cargo test -p xlog-integration --release --test test_wcoj_record_join_result_feedback` (W2.4) — 3/3 pass unchanged.
* `cargo test -p xlog-integration --release --test test_wcoj_recursive_dispatch` (slice-4) — 6/6 pass unchanged.
* `cargo test -p xlog-integration --release --test test_w26_heat_selectivity` (W2.6) — 7/7 pass unchanged.

### Step 8 — Evidence README

`docs/evidence/2026-05-04-w31-sorted-relation-accessors/README.md`
following the W2.6 README's structure:

* Header: "Closes W3.1 only", branch, base hash, plan reference.
* Summary: what the new entry points do; what's untouched.
* Acceptance properties table mapping each board-line claim to the
  cert tests that lock it.
* Cert results: full `cargo test` output for the 5 + 5 + 72 = 82
  new tests.
* Workspace tally with the post-W3.1 numbers.
* Code-level changes table (only `wcoj.rs` + 3 new test files).
* Decision mapping (D1–D6, with rationale).
* Process rule compliance.
* Closure proposal (gated on user approval): board update
  proposal `OPEN → DONE`, tally `DONE: 5 → 6; OPEN: 14 → 13`
  (verify counts at apply-time, the same way W2.6's evidence did).

### Step 9 — Closure proposal + FF-merge (gated on user approval)

* Do NOT modify `docs/v065-closure-board.md` until user approves.
* Do NOT FF-merge until user approves.
* Do NOT push, do NOT tag.

## Test counts summary (locked)

| Part | Description | # tests |
|------|-------------|--------:|
| Width-class validation U32 | Step 5 first half | 5 |
| Width-class validation U64 | Step 5 second half | 5 |
| Round-trip grid (3 shapes × 4 width-classes × 6 arities) | Step 6 | 72 |
| **W3.1 acceptance total** | | **82** |

## Process Rule Compliance

* Process rule #1: this slice does **not** self-mark W3.1 DONE.
* Process rule #2: every commit references W3.1.
* Process rule #3: this plan opens with "Closes W3.1 only."
* Process rule #5: no `v0.6.6` references; no "punt-to-later"
  wording — out-of-scope items are owned by W3.2+ board items,
  named at the point of reference, not pre-named here as W3.1's
  responsibility.
* Process rule #6: no push, no tag.

## Iteration-2 Direction (Locked)

Five points the user locked in the iteration 1 → 2 review; all
incorporated into the body of this plan:

1. **Mixed 4-byte coverage**: every arity {2, 3, 4, 5, 6, 7}.
   Step 6 grid expanded from 3 width-classes × 6 arities = 54
   cells to 4 width-classes × 6 arities = 72 cells, with
   `(U32, Symbol, U32, Symbol, ...)` at every arity.
2. **Symbol fixture content**: real `SymbolTable`-allocated IDs,
   not arbitrary u32 bits — Symbol parity is the contract.
3. **Error-message asserts**: `contains` on stable semantic
   fragments, not exact-tail matches.
4. **Workspace delta**: symbolic +82 (was +67 in iteration 1; the
   change reflects the Step-6 grid expansion). No global
   pass-count pin.
5. **Plan commit timing**: branch commit #1 in
   `feat/w31-sorted-relation-accessors`. Plan does NOT land on
   `main` (departs from W2.6's `d3ef4cda` pattern; per user
   direction, all W3.1 evidence stays inside the worktree branch
   until closure approval).

## Iteration 5 → 6 Text Fix

One live contradiction the user flagged in iteration-5 review:

* **Step 2 doc-comment validation order.** Step 2 told the
  implementer that the new entry's doc-comment must document
  validation order as `runtime → stream → arity ≥ 2 →
  per-column width-class`. That order had stream resolution
  inside the entry point's validation list, contradicting the
  iteration-4 lock that stream resolution is owned by
  `dedup_full_row_recorded` (not the new entry). Patched to
  `runtime → arity ≥ 2 → per-column width-class → delegate`,
  with an explicit second sentence stating that stream
  resolution is owned by `dedup_full_row_recorded` and is not
  in this entry point's validation list. Step 3's "mirror of
  step 2" wording inherits the fix automatically — no separate
  edit needed for `wcoj_layout_sort_u64_recorded`.

## Iteration 4 → 5 Text Fixes

Two text issues the user flagged in iteration-4 review:

1. **Step 8 evidence test count.** Iteration 4's Step 8 still
   said "Cert results: full `cargo test` output for the
   5 + 5 + 57 = 67 new tests" — the 57/67 numbers were
   iteration-1 holdovers; iteration 2 expanded Step 6 to 72
   round-trip tests (3 shapes × 4 fixtures × 6 arities). Patched
   to "5 + 5 + 72 = 82 new tests" so Step 8 matches the
   acceptance grid the rest of the plan locks.
2. **Banned punt-to-later wording.** Process rule #5 forbids the
   particular descriptor; iteration 4's rule text and the
   iteration-1 → 2 diff table both quoted that descriptor
   literally, which counts as live use in the plan body. Process
   rule rewritten to call the banned pattern "punt-to-later"
   wording instead of quoting the banned token, and the
   diff-table cell rewritten to describe the iteration-1 phrasing
   without naming it. The banned token now appears nowhere in
   this plan file.

## Iteration 3 → 4 Text Fixes

Two live contradictions the user flagged in iteration-3 review:

1. **Acceptance line — width-class fixture count.** Iteration-3
   line 24 still said "three width-classes (U32, U64, Symbol)"
   while iteration 2 had locked four (U32, U64, Symbol,
   mixed-4-byte) at every arity, with the cert grid math at
   72 cells. Patched: acceptance paragraph now says **four
   width-class fixtures** with the (3 shapes × 4 fixtures × 6
   arities = 72 cells) breakdown spelled out, plus the +10
   validation tests for the **82-test acceptance total**.
2. **Stream-resolution ownership.** Iteration-3's
   "What W3.1 adds" prose (line 60) listed
   "Resolves `launch_stream` (same error contract)" as step 2
   of the entry-point validation; the body skeleton (line ~139)
   delegated stream resolution to `dedup_full_row_recorded`.
   Patched per user recommendation: W3.1 entries validate
   runtime-backed + arity + per-column width-class, then
   delegate to `dedup_full_row_recorded`. **Stream resolution
   stays owned by `dedup_full_row_recorded`** — the new entries
   do NOT call `stream_pool().resolve(...)` themselves. Audited
   the existing `wcoj_layout_u32_recorded` /
   `wcoj_layout_u64_recorded` to confirm they follow the same
   pattern (`grep` of each function body shows no
   `stream_pool` / `resolve` call; both delegate stream-resolution
   downstream through `dedup_full_row_recorded` and the fast-path
   checker). New W3.1 entries inherit that pattern.

## Iteration 2 → 3 Defect Patches

Three blocking items the user flagged in iteration-2 review:

1. **D4 wording** — iteration 2 said "owned by a future
   perf-targeted board item (not pre-named here)", which still
   invents an untracked future slice. Replaced with the user's
   verbatim direction: "out of scope for W3.1 and receives no
   closure credit in this slice. No future-item invention." No
   follow-up commentary about hypothetical future items.
2. **Step 2 signature** — user's review excerpt showed two
   `&self` parameters on the signature. Grep of the plan file
   confirms the on-disk version has exactly one `&self` (line
   120). No edit was needed — verified clean. Logged here so the
   patch is auditable.
3. **`n == 0` contradiction** — iteration 2 had inconsistent
   n==0 handling (the "What W3.1 adds" prose said
   "short-circuit via `create_empty_buffer`"; the body skeleton
   said "delegate, dedup_full_row_recorded handles it").
   Resolved per user recommendation: keep the skeleton,
   delegate unconditionally to `dedup_full_row_recorded`. The
   "What W3.1 adds" prose is rewritten to match. Single source
   of truth for empty-buffer semantics is the existing
   `dedup_full_row_recorded` n==0 branch.

## Iteration 1 → 2 Diff (for review)

| Field | Iteration 1 | Iteration 2 |
|-------|-------------|-------------|
| Plan commit lands on | `main` | branch commit #1 |
| Mixed 4-byte arities | only arity 4 | all of {2,3,4,5,6,7} |
| Symbol fixture content | u32 bits stored as Symbol | real `SymbolTable` IDs |
| Error-message asserts | exact-tail match | `contains` semantic fragments |
| Workspace pass-count assertion | exact post-W3.1 number | symbolic +N delta only |
| Out-of-scope wording | iteration-1 phrasing punted scope to a later slice | "out of scope, owned by W3.2+ board items" |
| `v0.6.6` references | not mentioned (implicit) | explicitly forbidden in process rule #5 |
| Step 6 test count | 57 | 72 |
| W3.1 acceptance total | 67 | 82 |

## Open Questions for Iteration 7

Iteration 6 closes the iteration-5 stream-order contradiction.
No structural or procedural ambiguities remain. The user has
already pre-approved the worktree-creation procedure (per
iteration-2 reply): on iteration-6 approval, create
`.worktrees/w31-sorted-relation-accessors` from main at
`475774ef`, copy the plan file there, commit it as branch commit
#1 of `feat/w31-sorted-relation-accessors`, and remove the
untracked plan from `main`. No code changes until iteration 6 is
explicitly approved.
