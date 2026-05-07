# W4.2 Nested-Loop Join Operator — Plan (iteration 2 canonical)

**Plan iteration:** 2 (amendment after iteration-1 review surfaced F-W42-1..6 — 3 blocking + 2 major + 1 minor).
**Worktree:** `.worktrees/w42-nested-loop-join` on branch `feat/w42-nested-loop-join` (off local `main` `20dd96a5`).
**Spike evidence:** `bench-spike/w42-nested-loop` HEAD `9c0cefc6` (unmerged); evidence at `docs/evidence/2026-05-07-w42-bench-spike/README.md`.
**Date:** 2026-05-07.

## Acceptance Line (locked from board)

From `docs/v065-closure-board.md`:

> W4.2 | ROADMAP item #14 | OPEN | — | Nested-loop join operator for small relations. Adaptive selection: when both sides are below a threshold, nested-loop is cheaper than hash. | Cert: small × small fixture picks nested-loop; large × small picks hash; row-set agreement.

## Paper-alignment note

W4.2 has **no direct paper claim** in arXiv:2604.20073 (the SRDatalog paper is about WCOJ + recursive Datalog, not binary-join operator selection). This is internal optimization work, not paper-grounded closure. The W4.1 paper-alignment discipline does not apply; standard correctness + perf-evidence discipline does.

## Process Rule Compliance

* User-locked at iteration approval (not in this plan):
  1. Fresh branch off local `main` `20dd96a5`. ✅
  2. Spike branch preserved unmerged. ✅
  3. Production eligibility narrow (inner join, U32/Symbol single-key, small row-count). Encoded in D2.
  4. Threshold conservative, evidence-grounded, ≠ 1000 unchanged, capped below untested crossover. Encoded in D4.
  5. Correctness certs A/B/C/D required. Encoded in D5.
  6. Post-implementation bench step required. Encoded in Step 12.
  7. No board edit / DONE marking until gates pass and closure separately approved. Encoded in D8.

## Direction (locked, iteration 1)

| ID | Topic | Direction |
|----|-------|-----------|
| **D1** | **Eligibility predicate (production-narrow).** | A join is eligible for nested-loop dispatch iff ALL hold: (a) `JoinType::Inner` (only Inner; Semi/Anti/LeftOuter fall back to hash); (b) exactly **1 key column** on each side (`left_keys.len() == 1 && right_keys.len() == 1`); (c) the key column is `ScalarType::U32` OR `ScalarType::Symbol` (both 4-byte unsigned with identical kernel-level treatment); (d) size threshold (D4) is met. ANY other shape (multi-key, non-U32/Symbol key, non-Inner) MUST fall back to hash. The eligibility check lives at the dispatch site and is bit-cheap (no buffer reads, no kernel launches). |
| **D2** | **Production kernel scope (per F-W42-1, F-W42-2).** | Hardened nested-loop kernel `nested_loop_join_inner_u32_1key_pairs`. Kernel emits **matched (left_idx, right_idx) index pairs** as two parallel `u32` arrays — it does NOT materialize output rows. Output materialization happens AFTER the kernel via `Self::gather_buffer_by_indices` (the same gather pattern `hash_join_v2` uses at `crates/xlog-cuda/src/provider/relational.rs:3344-3354`). Output schema: **full concatenation** `[left_cols, right_cols]` via `combine_schemas(left, right)` — drop-in compatible with `hash_join_v2`'s actual output (verified: hash gathers BOTH sides' full column lists; there is no key-column dropping). Symbol keys reuse the U32 kernel byte-identically (Symbol is `u32` at the `ScalarType` byte level). The kernel reads ONLY the key columns from each side as `*const uint32_t` pointers (one per side); payload columns are NOT touched by the kernel — they reach the output through `gather_buffer_by_indices`. Per F-W42-2: this is consistent with `CudaBuffer`'s columnar layout (`crates/xlog-cuda/src/memory.rs:1041-1055`) — `CudaBuffer.columns` is a `Vec<CudaColumn>`, each column its own `CudaSlice<u8>`; the kernel takes per-column pointers, not row-major raw bytes. The spike's 1-col kernel does NOT graduate to production. |
| **D3** | **Provider API surface (per F-W42-2, F-W42-5).** | Add `pub fn nested_loop_join_v2_inner_u32_1key(left, right, left_key, right_key) -> Result<CudaBuffer>` on `CudaKernelProvider`. Body sequence: (1) validate eligibility (1 key col, U32 OR Symbol type, key cols within arity); (2) read row counts (`device_row_count`); (3) compute upper-bound output rows = `num_left * num_right` (in `u64` to avoid overflow at the validation step); (4) allocate the two index arrays as `TrackedCudaSlice<u32>` of length `upper_bound`; allocate counter; (5) launch kernel with key-column pointers; (6) `device.synchronize()`; (7) D2H counter via `dtoh_scalar_untracked` (single u32 — same metadata-only D2H profile as `hash_join_v2`); (8) call `gather_buffer_by_indices(left, &output_left_idx, output_rows)` and same for right (the existing GPU-side gather machinery); (9) construct result `CudaBuffer` via `buffer_from_columns` with `combine_schemas(left, right)`. **No silent capacity clamp**: per F-W42-5, the eligibility predicate (D1+D4) caps `num_left * num_right <= 4_000_000`, so the index-array allocation upper bound is bounded at 32 MB total (4M × 4 bytes × 2 arrays). The provider validates `num_left * num_right <= NESTED_LOOP_TOTAL_THRESHOLD` defensively and returns `Err(XlogError::Kernel)` if the caller violates the contract — fail-closed, never truncate. NO API entry for fall-back to hash inside the provider — fallback decisions are made by the executor's dispatch site, not by the provider. |
| **D4** | **Threshold (Cartesian product, conservative from spike, per F-W42-4).** | Dispatch nested-loop iff `left_rows * right_rows <= NESTED_LOOP_TOTAL_THRESHOLD` (note: **`<=`**, inclusive — per F-W42-4 boundary fix; `2000 * 2000 = 4_000_000` is admitted, matching the spike's largest verified-winning symmetric cell) where `NESTED_LOOP_TOTAL_THRESHOLD = 4_000_000` (4M Cartesian rows). **Rationale from spike (`docs/evidence/2026-05-07-w42-bench-spike/README.md`):** the largest symmetric tested cell `L=R=2000` → 4M total, NL win 5.41×; the next tested cell `L=R=5000` → 25M total, still NL win 4.28×; the algorithmic crossover is extrapolated to ~10000×10000 = 100M; 4M is well below the untested zone with 6× margin to absorb the F3 caveat (production multi-col kernel may have higher per-row cost than the spike's 1-col kernel). The Cartesian-product semantic (`left * right`) replaces the existing dead `right_rows < 1000` semantic (`crates/xlog-runtime/src/statistics.rs:22`) — the spike showed `right_rows`-only is insufficient because L=5000×R=50 wins the same as L=50×R=5000. **The existing `JoinStrategy::NESTED_LOOP_THRESHOLD = 1000` is NOT shipped unchanged**: W4.2 introduces a NEW constant `NESTED_LOOP_TOTAL_THRESHOLD = 4_000_000` and leaves the existing dead-code enum untouched (its cleanup is out of W4.2 scope). The threshold also serves as the **memory-safety cap** (per F-W42-5 + D3): within eligibility, the index-array allocation is bounded at 32 MB total. |
| **D5** | **Test surface (correctness certs, per F-W42-3, F-W42-6).** | Five certs in `crates/xlog-integration/tests/test_w42_nested_loop_dispatch.rs` (new file): **(A)** small×small dispatch — eligible inputs at `L=100, R=100` (10K total, well below threshold); assert `nested_loop_dispatch_count >= 1` AND row-set parity vs a reference computed by **direct provider call to `provider.hash_join_v2`** on the same uploaded `CudaBuffer` inputs (per F-W42-6: do NOT use `wcoj_*` counters as hash evidence — wcoj is unrelated; the parity assertion is the correctness witness). **(B)** **large × small fallback (board acceptance line)** — asymmetric above-threshold fixture **`L=50_000, R=100`** (5M total, above 4M threshold) with bounded matches (e.g., right keys ⊆ left keys with controlled cardinality so the join output stays small enough for parity comparison); assert `nested_loop_dispatch_count == 0` AND row-set parity vs `provider.hash_join_v2` reference. (Per F-W42-3: the iteration-1 `L=R=10000` was symmetric, not "large × small" as the board acceptance line requires.) **(C)** multi-col key fallback — `L=100, R=100`, `left_keys = [0, 1]`, `right_keys = [0, 1]` (2-col composite key, eligibility predicate disqualifies); assert `nested_loop_dispatch_count == 0` AND row-set parity. **(C')** non-Inner fallback — `L=100, R=100`, `JoinType::Semi`; assert `nested_loop_dispatch_count == 0` AND semi-join row set correct. **(E)** Symbol-typed key dispatch — Symbol-keyed inner join with row counts in eligible range; assert `nested_loop_dispatch_count >= 1` AND row-set parity. **(D)** row-set parity is folded as tail `BTreeSet<row>` comparison into A/B/C/C'/E. |
| **D6** | **Dispatch counter.** | Add `nested_loop_dispatch_count: AtomicU64` to `Executor` (mirrors the existing `wcoj_triangle_dispatch_count` / `wcoj_4cycle_dispatch_count` pattern at `crates/xlog-runtime/src/executor/mod.rs`). Increments on every successful nested-loop launch from `execute_join`. Accessor `pub fn nested_loop_dispatch_count(&self) -> u64`. NO `RuntimeConfig` field, NO env knob (per D8 process locks). The counter is observability for tests; runtime always dispatches via the eligibility predicate. |
| **D7** | **Acceptance gates (locked).** | (1) Cert A PASS (small×small dispatch + parity); (2) Cert B PASS (large×small hash fallback + parity); (3) Cert C PASS (unsupported-shape fallback for multi-col key + non-Inner); (4) Cert D PASS (row-set parity built into A/B/C); (5) Cert E PASS (Symbol-typed dispatch); (6) Post-implementation bench (Step 12) shows nested-loop wins by **≥ 2×** vs hash on the eligible-envelope fixture (multi-col, single-key, U32/Symbol, ≤ 4M total); (7) all other slice-1/2/4 + W4.1 tests PASS (no regressions); (8) zero workspace warnings on touched files; (9) `cargo fmt --check --all` clean; (10) `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` exit 0; (11) `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1; (12) post-impl bench evidence committed to `docs/evidence/2026-05-07-w42-production-bench/README.md`. |
| **D8** | **Process locks.** | No board edit. No DONE marking. No FF-merge. No `v0.6.6` references. No env-knob additions (`XLOG_NESTED_LOOP_*` etc. forbidden). No `RuntimeConfig` field additions. The threshold `NESTED_LOOP_TOTAL_THRESHOLD` is a `const` in code, not config-tunable in v0.6.5. The existing dead `JoinStrategy` enum at `crates/xlog-runtime/src/statistics.rs:7-44` is NOT touched (its cleanup is out of W4.2 scope; W4.2 introduces a parallel constant + eligibility predicate). The bench spike branch (`bench-spike/w42-nested-loop`) stays unmerged — W4.2 does not graduate spike code to production. |

## Read-Only Surface (recon results)

* **Existing dead-code design layer** (W4.2 leaves untouched per D8):
  * `crates/xlog-runtime/src/statistics.rs:7-44` — `JoinStrategy` enum + `JoinStrategy::select` static helper with hardcoded `NESTED_LOOP_THRESHOLD = 1000`. Zero production consumers; only `crates/xlog-runtime/tests/statistics_tests.rs` exercises it.
* **Production hash-join dispatch site** (W4.2 wires here):
  * `crates/xlog-runtime/src/executor/node_dispatch.rs:246-339` — `execute_join`. Currently always calls `hash_join_v2` or `hash_join_v2_with_index`. W4.2 adds a pre-hash branch for nested-loop eligibility.
* **GPU kernel infrastructure**:
  * `crates/xlog-cuda/kernels/join.cu` — hash-join family. W4.2 appends `nested_loop_join_inner_u32_1key`.
  * `crates/xlog-cuda/src/kernel_manifest_data.rs:50-66` — kernel registration list for the `"join"` module.
  * `crates/xlog-cuda/src/provider/relational.rs:2498` — `hash_join_v2` reference impl for ownership/error/output-shape conventions.
* **Existing dispatch-counter pattern** (W4.2 mirrors):
  * `crates/xlog-runtime/src/executor/mod.rs` — `wcoj_triangle_dispatch_count: AtomicU64` + accessor. W4.2 adds an analogous `nested_loop_dispatch_count`.
* **Cert template** (W4.2 mirrors):
  * `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs` — gate-off reference + gate-on dispatched + row-set parity pattern. W4.2's certs at `test_w42_nested_loop_dispatch.rs` follow the same shape.

## Step-by-Step Execution Plan (12 steps)

### Step 1 — Plan iteration commit (this commit)

This iteration-1 plan, on `feat/w42-nested-loop-join`. No code yet. The agent does NOT advance to Step 2 until the user explicitly approves iteration 1. (Subsequent iterations may add F-W42-N findings.)

Commit subject: `docs(plan): W4.2 iteration 1 — nested-loop join (recon + spike-grounded direction)`.

### Step 2 — Production kernel (emit-pairs design)

File: `crates/xlog-cuda/kernels/join.cu` (append).

Add `extern "C" __global__ void nested_loop_join_inner_u32_1key_pairs(...)` that emits matched **(left_idx, right_idx) index pairs** (NOT row-major bytes). Per F-W42-2 this matches `CudaBuffer`'s columnar layout (`crates/xlog-cuda/src/memory.rs:1041-1055`).

Signature:

```cuda
extern "C" __global__ void nested_loop_join_inner_u32_1key_pairs(
    const uint32_t* __restrict__ left_keys,    // pointer to left's key column data
    const uint32_t* __restrict__ right_keys,   // pointer to right's key column data
    uint32_t num_left,
    uint32_t num_right,
    uint32_t* __restrict__ output_left_idx,    // pre-allocated, capacity = num_left * num_right
    uint32_t* __restrict__ output_right_idx,
    uint32_t* __restrict__ output_count,
    uint32_t output_capacity                   // = num_left * num_right (no clamp; eligibility caps this at 4M)
);
```

Each thread takes one left-row; iterates over all right-rows; on `right_keys[r] == left_keys[tid]`, `atomicAdd` to `output_count` and write `(tid, r)` to the index arrays. Per F-W42-5: there is no silent clamp — the eligibility predicate (D1+D4) caps `num_left * num_right <= 4_000_000`, so allocation is bounded and `out_idx < output_capacity` is by-construction. The kernel WILL still guard with `if (out_idx < output_capacity)` defensively (cheap branch); on contract violation the provider returns `Err` BEFORE the launch.

Symbol-typed keys reuse this kernel byte-identically (Symbol IS u32). Register kernel name in `crates/xlog-cuda/src/kernel_manifest_data.rs::"join"` module and add a constant in `crates/xlog-cuda/src/provider/mod.rs::join_kernels`.

Commit subject: `feat(w42): add nested-loop emit-pairs kernel (multi-col-compatible)`.

### Step 3 — Provider API (emit-pairs + columnar gather)

File: `crates/xlog-cuda/src/provider/nested_loop.rs` (new file).

`pub fn nested_loop_join_v2_inner_u32_1key(left: &CudaBuffer, right: &CudaBuffer, left_key: usize, right_key: usize) -> Result<CudaBuffer>`:

1. Validate eligibility:
   - `left.arity() > left_key` and `right.arity() > right_key` (key indices within arity bounds).
   - `left.schema().column_type(left_key)` and right's match `ScalarType::U32` OR `ScalarType::Symbol`.
   - `num_left * num_right <= NESTED_LOOP_TOTAL_THRESHOLD`. If exceeded, return `Err(XlogError::Kernel("nested_loop: caller violated eligibility threshold"))` — fail-closed per F-W42-5.
2. Compute `upper_bound = (num_left as u64) * (num_right as u64)`. Cast to `u32` after the threshold check (4M fits in u32).
3. Allocate `d_output_left_idx: TrackedCudaSlice<u32>` of length `upper_bound`. Same for `d_output_right_idx`. Allocate `d_output_count: TrackedCudaSlice<u32>` of length 1; zero it via `memset_zeros`.
4. Read key-column pointers from `left.column(left_key)` and `right.column(right_key)`. (Both are `CudaColumn::Owned(TrackedCudaSlice<u8>)` for U32/Symbol — pass directly to the kernel as the `*const uint32_t` parameter.)
5. Launch the kernel with grid `(num_left + 255) / 256`, block `256`.
6. `device.synchronize()`.
7. D2H the counter via `self.dtoh_scalar_untracked(&d_output_count, 0)?` (single u32 — same metadata-only D2H pattern as `hash_join_v2`).
8. Materialize the output via the existing GPU-side gather machinery: call `self.gather_buffer_by_indices(left, &d_output_left_idx, output_rows)?` to get `gathered_left`, same for `gathered_right`. (This is the same pattern `hash_join_inner_v2` uses at `relational.rs:3344-3354`.)
9. Combine via `let combined_schema = self.combine_schemas(left.schema(), right.schema());` then `self.buffer_from_columns(result_columns, output_rows as usize, combined_schema)`. Per F-W42-1 + D2: output schema is the FULL `[left_cols, right_cols]` concatenation, drop-in compatible with `hash_join_v2`.

Register in `provider/mod.rs::mod nested_loop;`. Add kernel-name constant. Build verifies clean compile.

Commit subject: `feat(w42): add nested_loop_join_v2_inner_u32_1key provider fn (gather-based)`.

### Step 4 — Eligibility predicate

File: `crates/xlog-runtime/src/executor/node_dispatch.rs` (edit).

Add a private fn `eligible_for_nested_loop(left, right, left_keys, right_keys, join_type) -> bool` that returns `true` iff D1's predicate holds. Cheap O(1) check: enum-match on `JoinType::Inner`, key-col count, key-col `ScalarType` lookup via `left.schema().column_type(left_keys[0])`. NO row-count read here — the threshold check is separate (Step 5) and is also O(1) but reads the cached `host_row_count` if available, else `device_row_count` (single u32 D2H, same metadata-only cost as hash_v2's own row-count reads).

Commit subject: `feat(w42): add eligible_for_nested_loop predicate`.

### Step 5 — Dispatch counter + dispatch wiring

Files:
* `crates/xlog-runtime/src/executor/mod.rs` — add `nested_loop_dispatch_count: AtomicU64` field + accessor + reset hook (mirror `wcoj_*_dispatch_count`).
* `crates/xlog-runtime/src/executor/node_dispatch.rs` — at the top of `execute_join` (BEFORE the existing adaptive-indexing branch), check eligibility + threshold; if both met, call `provider.nested_loop_join_v2_inner_u32_1key` + increment counter + return; else fall through to the existing hash path (unchanged).

Constant: `const NESTED_LOOP_TOTAL_THRESHOLD: u64 = 4_000_000;` colocated with the dispatch site for visibility.

Commit subject: `feat(w42): wire nested-loop dispatch + counter at execute_join`.

### Step 6 — Cert A: small×small dispatch (per F-W42-6)

File: `crates/xlog-integration/tests/test_w42_nested_loop_dispatch.rs` (new).

Test `small_small_dispatches_nested_loop_and_matches_hash`:
* Fixture: `L=100, R=100`, multi-col (e.g., arity 2 with key at col 0, payload at col 1), single-key U32, unique-keyed.
* **Reference row set:** computed by direct provider call `provider.hash_join_v2(left, right, &[0], &[0], JoinType::Inner)` on the same uploaded `CudaBuffer` inputs — bypasses the executor's dispatch path. Convert to `BTreeSet<Row>`.
* **Dispatched row set:** computed via `Executor::execute_plan` with default config (which routes through the new dispatch wiring). Assert:
  - `executor.nested_loop_dispatch_count() >= 1` — confirms the new path fired.
  - `BTreeSet<Row>` equals the reference set — correctness witness.
* **No `wcoj_*` counter assertions** — per F-W42-6, WCOJ counters are unrelated to hash/nested-loop dispatch and are not evidence here. The pair `(nested_loop_dispatch_count >= 1, row-set parity)` is sufficient evidence: the dispatch path fired AND produced the right answer.

Commit subject: `test(w42): cert A — small×small dispatches nested-loop with hash parity`.

### Step 7 — Cert B: large × small hash fallback (per F-W42-3, board acceptance line)

Test `large_times_small_falls_back_to_hash_above_threshold`:
* Fixture: **asymmetric large×small** `L=50_000, R=100`. Cartesian product = 5_000_000 > `NESTED_LOOP_TOTAL_THRESHOLD = 4_000_000` → ineligible. Per F-W42-3, this matches the board's "large × small picks hash" acceptance line; iteration-1's symmetric `L=R=10000` was the wrong shape.
* Bounded matches: right keys ⊆ `[0..100)`, left keys = sequential repeats so each left row matches at most one right key. Output ≈ 100 rows total — small enough for `BTreeSet<Row>` parity comparison.
* Single run via `Executor::execute_plan`, default config. Assert:
  - `executor.nested_loop_dispatch_count() == 0` — confirms the eligibility predicate refused the dispatch.
  - `BTreeSet<Row>` equals the reference computed by direct `provider.hash_join_v2(left, right, &[0], &[0], JoinType::Inner)`.

Commit subject: `test(w42): cert B — large × small falls back to hash above threshold`.

### Step 8 — Cert C: unsupported-shape fallback

Test `multi_col_key_falls_back_to_hash` and `semi_join_falls_back_to_hash`:
* Fixture: small (eligible row count) but with `left_keys = [0, 1]`, `right_keys = [0, 1]` (2-col composite key).
* Default config; assert `nested_loop_dispatch_count == 0` (multi-col key disqualifies despite small size).
* Second sub-test: same small fixture, but `JoinType::Semi`. Assert `nested_loop_dispatch_count == 0` (non-Inner disqualifies).

Commit subject: `test(w42): cert C — multi-col key + non-Inner fall back to hash`.

### Step 9 — Cert D: row-set parity (built into A/B/C)

This step is verification-only: confirm that A/B/C all carry `BTreeSet<Row>` parity assertions vs a hash-only reference run. No new test file. If any cert lacks the parity tail, this step adds it.

Commit subject (only if patches needed): `test(w42): cert D — strengthen row-set parity assertions`.

### Step 10 — Cert E: Symbol-typed dispatch

Test `symbol_typed_key_dispatches_nested_loop`:
* Fixture: small, single-col Symbol-typed inputs. Symbol is u32-shaped at the kernel level, so the same kernel applies but the eligibility predicate must accept `ScalarType::Symbol` alongside `ScalarType::U32`.
* Assert `nested_loop_dispatch_count >= 1`, row-set parity.

Commit subject: `test(w42): cert E — Symbol-typed key dispatches nested-loop`.

### Step 11 — Workspace gate (mid-W4.2)

Run the full gate set BEFORE the post-impl bench:
* `cargo fmt --check --all` clean.
* `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` exit 0; pass-count delta = +5 (5 new W4.2 cert fns: A, B, C-multicol, C-semi, E; D folded into A/B/C parity tails).
* `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1.
* Zero warnings on touched files.

Commit subject (if any cleanup): `chore(w42): workspace gate green pre-bench`.

### Step 12 — Post-implementation bench

File: `crates/xlog-integration/benches/w42_production_nested_loop_bench.rs` (new).

Bench the production kernel + dispatch path (NOT the spike kernel) on multi-col fixtures matching the production eligibility envelope:
* Multi-col arity (e.g., 3 cols with key at col 0).
* Single-key, U32.
* Cartesian-product matrix: `(L, R)` ∈ {(100,100), (500,500), (1000,1000), (2000,2000)} (all inside the 4M threshold), plus 2 above-threshold cells {(5000,5000), (10000,1000)} for hash-fallback comparison.
* Provider-direct envelope-parity vs `hash_join_v2`.
* Pre-cell row-set parity check.
* Output: `docs/evidence/2026-05-07-w42-production-bench/README.md` with median timings + speedup table.

D7 acceptance criterion #6: nested-loop must win by **≥ 2×** vs hash on the eligible cells. (Spike showed 4–6×; F3 caveat may reduce this for the production multi-col kernel; ≥ 2× is the minimum-viable signal that the threshold is correctly placed.)

Commit subject: `feat(w42): add production nested-loop bench + evidence`.

### Step 13 — Closure proposal (no DONE marking)

Plan-iteration commit + Steps 2–12 commits on `feat/w42-nested-loop-join`. No `docs/v065-closure-board.md` edit. No FF-merge. No advance.

Per D8: closure proposal text describes the work + acceptance evidence; the board edit + FF-merge + tally update happen ONLY on separate user approval, not as part of this plan's execution.

Commit subject (text-only): N/A (no commit).

## Acceptance Grid (iteration-1 canonical)

| Cell | Count | Test file | Acceptance criterion |
|------|-------|-----------|----------------------|
| **Cert A — small×small dispatch + parity** | 1 | `test_w42_nested_loop_dispatch.rs` (new) | `nested_loop_dispatch_count >= 1`; `BTreeSet<Row>` parity vs hash reference |
| **Cert B — large × small hash fallback + parity** | 1 | `test_w42_nested_loop_dispatch.rs` | Asymmetric `L=50_000 R=100` (5M total > 4M threshold); `nested_loop_dispatch_count == 0`; row-set parity vs `provider.hash_join_v2` reference |
| **Cert C — multi-col key fallback** | 1 | `test_w42_nested_loop_dispatch.rs` | `nested_loop_dispatch_count == 0`; row-set parity |
| **Cert C' — non-Inner (Semi) fallback** | 1 | `test_w42_nested_loop_dispatch.rs` | `nested_loop_dispatch_count == 0`; semi-join row set correct |
| **Cert E — Symbol-typed dispatch** | 1 | `test_w42_nested_loop_dispatch.rs` | `nested_loop_dispatch_count >= 1`; row-set parity |
| **Post-impl bench** | 1 | `w42_production_nested_loop_bench.rs` (new) | Nested-loop wins ≥ 2× vs hash on eligible cells |
| **Workspace pass-count delta** | **+5** | — | Five new test cells (A, B, C-multicol, C-semi, E). D is folded into A/B/C parity tails. Step 12 bench is non-test. |

## Source-of-Truth References (iteration-1 canonical)

* **Spike evidence**: `docs/evidence/2026-05-07-w42-bench-spike/README.md` (on `bench-spike/w42-nested-loop` branch); `9c0cefc6` HEAD.
* **Existing dead-code design**: `crates/xlog-runtime/src/statistics.rs:7-44` (`JoinStrategy` enum, untouched by W4.2).
* **Hash dispatch site**: `crates/xlog-runtime/src/executor/node_dispatch.rs:246-339`.
* **Hash provider reference**: `crates/xlog-cuda/src/provider/relational.rs:2498` (`hash_join_v2`).
* **Kernel manifest**: `crates/xlog-cuda/src/kernel_manifest_data.rs:50-66`.
* **Counter pattern**: `wcoj_*_dispatch_count` in `crates/xlog-runtime/src/executor/mod.rs`.
* **Cert template**: `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs`.

## Risk Register (informational, iteration 1)

| Risk | Mitigation |
|------|------------|
| Production multi-col kernel has higher per-row cost than spike's 1-col kernel (F3 caveat) → 4M Cartesian threshold may overshoot | Step 12 post-impl bench validates the threshold against production-shape fixtures. If post-bench shows < 2× win at 4M, a subsequent plan iteration amends the threshold downward. The emit-pairs design (D2 per F-W42-2) keeps the kernel's per-row cost low: only key columns are touched in the inner loop; payload arrives via columnar gather AFTER the kernel, so per-row kernel work is independent of arity. |
| `hash_join_v2`'s ~2.7 ms launch-overhead floor (F2 caveat) means measured wins partly attributable to overhead, not algorithm | Out of scope for W4.2 per user direction "Hash launch-overhead reduction is separate work". Recorded in Risk Register; W4.2 does NOT optimize hash. |
| Eligibility predicate misses an edge case (e.g., empty inputs, key column index out-of-bounds) → silent dispatch error | Step 4's predicate is fail-closed: any unrecognized type / out-of-bounds / arity mismatch returns `false` → falls back to hash. Cert C (multi-col + non-Inner) verifies the negative direction. |
| Symbol-typed inputs handled differently than U32 in the kernel | Symbol IS u32 at the byte level in xlog's `ScalarType` representation. Cert E directly verifies. If Symbol byte representation diverges in any subtle way, the cert fails before merge. |
| Threshold `4_000_000` is a magic number; future maintainers won't know why | Constant has a doc-comment citing the spike evidence path + iteration-1 plan ref. Bench evidence (Step 12 + spike) is the empirical basis. |
| Existing `JoinStrategy` dead code adds confusion | NOT touched by W4.2 (per D8). A separate cleanup task can delete it later. W4.2's parallel constant + dispatch live in the executor + provider, NOT in the dead `statistics.rs` enum. |
| Cert A's "nested-loop dispatched" assertion needs a way to force hash for the reference run | Two options: (a) add a `RuntimeConfig::with_nested_loop_dispatch(Some(false))` knob (REJECTED per D8 — no `RuntimeConfig` field additions); (b) capture hash row set via direct provider call in the test, bypass the executor for the reference. Use (b). Cert A's reference run calls `provider.hash_join_v2` directly on the same uploaded buffers; dispatched run uses `Executor::execute_plan`. |
| Cert B fixture upload time concerns (per F-W42-3 amendment to asymmetric large×small) | `L=50_000 R=100` is 5M Cartesian rows but only 50K + 100 = ~50K input rows uploaded — a fraction of the iteration-1 `L=R=10000` (100K input rows). Hash-join wall time on this fixture is dominated by the ~2.7 ms launch-overhead floor (per spike F2), not by the input size. Test budget < 5s on CUDA. |

## Plan-Approval Gate (iteration 2)

This plan is **iteration 2 draft** (iteration 1 had F-W42-1..6 surfaced; live D-table + Step plan + Acceptance Grid + Risk Register rewritten in place). The agent does NOT advance to Step 2 until the user explicitly states "Iteration 2 is approved" (or equivalent). Subsequent iterations may add further F-W42-N findings; the live D-table + Step plan + Acceptance Grid above are the canonical source of truth.

Before iteration approval, the user may:
* Push back on threshold value (e.g., reduce 4M to 2M or 1M for more conservatism).
* Push back on Cartesian-product semantics (e.g., revert to `right_rows < THRESHOLD` for simplicity).
* Push back on Cert E (Symbol scope) — could be deferred to W4.2 iteration 2 if scope creep concern.
* Push back on Step 12's `≥ 2×` criterion (e.g., raise to ≥ 3× or lower to ≥ 1.5×).
* Push back on the spike kernel NOT graduating to production (e.g., insist on graduating to save kernel-write time).
* Add/remove certs in D5.
* Adjust Step ordering or commit-subject conventions.
* Anything else.

The agent does NOT modify the live D-table / Step plan / Acceptance Grid based on chat alone — every amendment lands as a new iteration commit (`docs(plan): W4.2 iteration 2 amendment — F-W42-N findings`).

## Iteration 1 Notes (historical / superseded)

* Plan length: ~370 lines (intentionally tighter than W4.1's 757-line iteration-7 final). Subsequent iterations may expand if F-W42-N findings warrant.
* No paper-claim (P1-P5) alignment is required for W4.2 — the SRDatalog paper does not cover binary-join operator selection. W4.2 is internal-optimization closure work.
* Spike evidence is treated as **load-bearing input**: the threshold value (4M) and the post-impl bench acceptance (≥ 2×) are both derived from spike measurements. If subsequent W4.2 iterations contradict the spike, the spike evidence README is the canonical reference.

## Iteration-2 Amendment Log

User review of iteration 1 surfaced 3 blocking + 2 major + 1 minor findings. The live D-table, Step plan, Acceptance Grid, and Risk Register above are rewritten in place to be **iteration-2 canonical**. Each finding's before/after state is recorded here for traceability.

| ID | Severity | Finding | Iteration-1 (wrong) | Iteration-2 (corrected) |
|----|----------|---------|---------------------|--------------------------|
| **F-W42-1** | Blocking | Output schema is `[left_cols, right_cols]` not `[left_cols, right_cols_minus_key]` | D2 said `[left_cols, right_cols_minus_key]` | D2 + D3 + Step 3 say full concatenation `[left_cols, right_cols]` via `combine_schemas`; matches `hash_join_v2` (verified at `crates/xlog-cuda/src/provider/mod.rs:2151` `combine_schemas` extends both sides; `crates/xlog-cuda/src/provider/relational.rs:3344-3354` gathers full left + full right). Drop-in compatible. |
| **F-W42-2** | Blocking | Step 2's row-major kernel shape contradicts `CudaBuffer`'s columnar layout | Step 2 said `left_data[tid * left_arity + left_key_col]` (row-major bytes) | Step 2 + D2 + D3 redesigned to **emit (left_idx, right_idx) index pairs** + reuse `gather_buffer_by_indices` for columnar materialization. Kernel takes per-column pointers (`*const uint32_t left_keys`, `*const uint32_t right_keys`); payload columns are NOT touched by the kernel. Matches `CudaBuffer` columnar layout at `crates/xlog-cuda/src/memory.rs:1041-1055`. |
| **F-W42-3** | Blocking | Cert B fixture didn't match board's "large × small picks hash" | Cert B was `L=R=10000` (symmetric) | Cert B is now `L=50_000, R=100` (asymmetric large × small, 5M total > 4M threshold). Bounded matches via right keys ⊆ left keys. |
| **F-W42-4** | Major | Threshold boundary inconsistent: `<` vs `<=` for `2000 * 2000 = 4_000_000` | D4 said `< 4_000_000`; Step 12 listed `(2000,2000)` as eligible | D4 corrected to **`<= 4_000_000`** (inclusive). `2000 * 2000 = 4_000_000` is admitted, matching the spike's largest verified-winning symmetric cell. |
| **F-W42-5** | Major | "Capacity-clamp same as spike" is unsafe in production (silent truncation violates row-set parity) | D3 said "capacity-clamp same as spike (256M entries)" | D3 + Step 3 + Step 2 redesigned: provider validates `num_left * num_right <= NESTED_LOOP_TOTAL_THRESHOLD` and returns `Err(XlogError::Kernel)` on contract violation BEFORE any allocation (fail-closed). Within eligibility, allocation is exact (`upper_bound = num_left * num_right`); kernel cannot overflow because output_capacity equals the upper bound. The 4M threshold IS the safety cap; index-array allocation is bounded at 32 MB total. |
| **F-W42-6** | Minor | Cert A's "hash was NOT used" wording mentioned wcoj_* counters, which are unrelated | Step 6 cert A said "assert wcoj_*_dispatch_count == 0" as hash-evidence | Step 6 + D5 corrected: Cert A relies on `nested_loop_dispatch_count >= 1` (positive evidence the new path fired) AND row-set parity vs a reference computed by direct `provider.hash_join_v2` call (correctness witness). No `wcoj_*` assertions; no hash counter introduced. |

**Net effect:** D2, D3, D4, D5 rewritten in place. Step 2 and Step 3 redesigned. Step 6 (Cert A), Step 7 (Cert B) updated. Risk Register row about Cert B fixture updated. Acceptance Grid row about Cert B updated. Header iteration tag bumped from 1 to 2.

**Process note:** Per the W4.1 plan-iteration discipline (which W4.2 inherits), all amendments are surfaced as F-W42-N findings with explicit before/after states. The agent does NOT modify the live D-table / Step plan based on chat alone — every amendment lands as a new iteration commit. This iteration-2 commit (`docs(plan): W4.2 iteration 2 amendment — F-W42-1..6 (3 blocking, 2 major, 1 minor)`) is the corresponding artifact.
