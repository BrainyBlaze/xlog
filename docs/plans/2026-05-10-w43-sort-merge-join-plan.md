# W4.3 Sort-Merge Join Operator — Plan (iteration 5 canonical)

**Plan iteration:** 5 (post-execution amendment after Step 11 verification surfaced F-W43-12 — workspace-test gate exception for pre-existing `test_wcoj_layout_fast_path` flake — alongside retroactive logging of F-W43-11 from Step 5; both are execution-discovered findings, distinct from the pre-execution review pattern of iterations 1–4).
**Worktree:** `.worktrees/w43-sort-merge-join` on branch `feat/w43-sort-merge-join` (off local `main` `19f7bc5d`).
**Spike evidence:** `bench-spike/w43-sort-merge` HEAD `fadc2700` (unmerged); evidence at `docs/evidence/2026-05-10-w43-bench-spike/README.md`.
**Recon predecessor:** `docs/plans/2026-05-08-w43-sort-merge-join-recon.md`.
**Date:** 2026-05-10.

## Acceptance Line (locked from board)

From `docs/v065-closure-board.md`:

> W4.3 | ROADMAP item #15 | OPEN | — | General sort-merge join operator for pre-sorted binary relations. Triangle-layout helper is a special case; this is the generic path. | Cert: pre-sorted binary join skips the sort step, matches reference output.

## Paper-alignment note

W4.3 has **no direct paper claim** in arXiv:2604.20073 — the SRDatalog paper covers WCOJ + recursive Datalog, not binary-join operator selection. Same status as W4.2: internal optimization work, standard correctness + perf-evidence discipline.

## Process Rule Compliance

* Spike-first per `feedback_perf_bench_spike_first.md`: ✅ done; `bench-spike/w43-sort-merge` preserved unmerged at `fadc2700`.
* Spike decision-gate ≥ 2× win on tested matrix: ✅ met (range 2.52×–3.25×).
* Five mandatory iteration-1 locks per user direction: encoded in D1–D5 below.

## Direction (locked, iteration 5 canonical)

| ID | Lock | Direction |
|----|------|-----------|
| **D1** | **Sortedness detection mechanism (per F-W43-4 empty-input handling)**. | **Option B (runtime detection kernel)** per recon. New kernel `check_ascending_sorted_u32` in `crates/xlog-cuda/kernels/sort.cu` — single-pass scan, returns `1` if `keys[i] <= keys[i+1]` for all `i`, else `0`. Provider fn `provider.is_sorted_ascending_u32(buf, key) -> Result<bool>` wraps the kernel. **Empty / single-row fast path (per F-W43-4)**: provider fn checks `device_row_count(buf)? < 2` BEFORE any allocation or kernel launch and returns `Ok(true)` (a 0- or 1-row sequence is trivially sorted). Empty inputs reach the dispatch site through threshold check `0 * 0 = 0 <= 4M = true`, so detection MUST short-circuit empties without launching the kernel — otherwise the kernel grid `(0 + 255) / 256 = 0` is undefined. Same fast-path semantic as `hash_join_inner_v2`'s empty handling at `relational.rs:3165-3170`. For `n >= 2` rows the kernel runs, reads the u32 result via `dtoh_scalar_untracked` (single-u32 D2H, same metadata-only profile as `hash_join_v2`'s row-count reads). The dispatch site at `execute_join` calls this fn ONLY when the join is otherwise eligible for sort-merge (Inner + 1-key + matching U32/Symbol + size-eligible per D3); detection cost is paid only on candidates, NOT on every join. Mirrors the established WCOJ layout fast-path pattern at `crates/xlog-cuda/src/provider/wcoj.rs:3137-3187` (u32) and `:3265-3307` (u64), but checks "sorted ascending" only — duplicates allowed (sort-merge handles run-length). The kernel does NOT check uniqueness. **Out of scope for W4.3**: producer-side metadata tracking (option C) and IR-level annotation (option D); both are larger structural changes that can be considered in v0.6.6+ if benchmark data justifies eliminating the per-dispatch detection cost. |
| **D2** | **Dispatch precedence vs W4.2 nested-loop (PROVISIONAL per F-W43-2)**. | **Sort-merge takes precedence when sorted**, nested-loop when not sorted (and size-eligible), hash otherwise. Decision tree at `execute_join`:<br>1. Eligible for sort-merge envelope (Inner + 1-key + matching U32/Symbol)?<br>&nbsp;&nbsp;a. Yes AND size-eligible (D3)?<br>&nbsp;&nbsp;&nbsp;&nbsp;i. Both inputs detected sorted via D1 kernel (Ok(true))? → **sort-merge**.<br>&nbsp;&nbsp;&nbsp;&nbsp;ii. Either Ok(false), Err(_), or precondition failure? → fall through (fail-closed per D5).<br>&nbsp;&nbsp;b. Not size-eligible? → fall through.<br>2. Eligible for nested-loop envelope (W4.2 D1 + size-eligible)? → **nested-loop**.<br>3. Else → **hash**.<br><br>**Precedence is PROVISIONAL** (per F-W43-2): the spike measured sort-merge-vs-hash and W4.2 measured nested-loop-vs-hash, but no benchmark has DIRECTLY compared sort-merge vs nested-loop on overlapping eligibility (sorted + size-eligible inputs) WITH detection cost included. The Step-12 production bench MUST include side-by-side overlap cells where both operators are eligible, so the precedence decision can be empirically validated. If the bench shows nested-loop wins on the overlap (e.g., because detection-kernel cost erodes sort-merge's advantage on tiny cells), iteration-2+ amends D2 to nested-loop-first. Until Step 12 evidence lands, the precedence is the working hypothesis based on indirect spike data only. |
| **D3** | **Memory-safe output sizing**. | **Cartesian-style threshold matching W4.2's** `NESTED_LOOP_TOTAL_THRESHOLD = 4_000_000`. Sort-merge dispatches iff `(L as u64).checked_mul(R as u64).map(\|p\| p <= NESTED_LOOP_TOTAL_THRESHOLD).unwrap_or(false)`. Output worst-case is `L * R` (when all keys identical → full Cartesian explosion); the threshold caps allocation at 4M output rows × 4 bytes × 2 index arrays = 32 MB total intermediate, plus the gather pass output. **Constant-sharing decision**: both W4.2 (nested-loop) and W4.3 (sort-merge) use the SAME constant. No new constant introduced. **Rationale**: spike data shows sort-merge wins through L=R=5000 (25M Cartesian) — bigger than the 4M threshold — but the worst-case OUTPUT for arbitrary key distributions can be `L * R`, not just `min(L, R) * dup_rate`. The 4M cap is conservative-but-bench-grounded for both operators; consistency between them simplifies the dispatch + makes the memory budget at the dispatch site predictable. **Future iteration**: a higher sort-merge-specific threshold could be introduced in v0.6.6+ if production traffic shows expected-output is much smaller than worst-case (e.g., empirical-distribution-based dynamic threshold). Not in W4.3 scope. |
| **D4** | **Schema/key-type admissibility (production-narrow)**. | A join is eligible for sort-merge dispatch iff ALL hold: (a) `JoinType::Inner` (Semi/Anti/LeftOuter fall back); (b) exactly **1 key column** on each side (`left_keys.len() == 1 && right_keys.len() == 1`); (c) **left and right key column types are EQUAL** AND that shared type is `ScalarType::U32` OR `ScalarType::Symbol`; (d) size threshold (D3) met; (e) sortedness detected on BOTH sides via D1 kernel. Mirrors W4.2's narrow envelope so the operator-precedence decision tree (D2) is symmetric across sort-merge and nested-loop's eligibility checks. Multi-key, non-Inner, non-U32/Symbol, mismatched types, above-threshold, OR unsorted → fall back to nested-loop or hash per D2. |
| **D5** | **Hash-fallback policy on detection failure (per F-W43-1 fail-closed lock)**. | **Fall through to W4.2 nested-loop OR hash** per D2's decision tree. Sort-then-merge (sort the inputs first, then merge) is NOT in W4.3 scope. Rationale: the sort kernel itself has multi-launch overhead (~7-step radix-sort family at `crates/xlog-cuda/kernels/sort.cu`); paying that upfront would erode the sort-merge speedup such that the operator may net-lose vs hash. Spike does not measure sort-then-merge; without empirical data, sort-then-merge is speculative and out of scope. The dispatch is **FAIL-CLOSED** on detection: the dispatch site MUST handle `is_sorted_ascending_u32`'s return as `Result<bool>` and fall through on **both** `Ok(false)` AND `Err(_)`. Specifically, the dispatch site MUST NOT use the `?` operator on the detection call — propagating the Err to the caller would violate the fail-closed contract by erroring out a join that COULD have succeeded via nested-loop or hash. The pseudocode in Step 5 reflects this: `match is_sorted_ascending_u32(...) { Ok(true) => proceed, _ => fall through }`. Detection NEVER causes an error to propagate to the caller. |
| **D6** | **Dispatch counter**. | Add `pub(super) sort_merge_dispatch_count: u64` to `Executor` (mirrors W4.2 `nested_loop_dispatch_count` plain-`u64` convention; methods take `&mut self` so atomic synchronization is unnecessary). Increments on every successful sort-merge launch from `execute_join` via `self.sort_merge_dispatch_count += 1`. Public accessor `pub fn sort_merge_dispatch_count(&self) -> u64`. NO `RuntimeConfig` field, NO env knob (per process locks). The counter is observability for tests; runtime always dispatches via the eligibility predicate + detection kernel. |
| **D7** | **Acceptance gates (locked, per F-W43-3 timed-region clarification + F-W43-4 empty-input cert + F-W43-12 workspace-test gate exception).** | (1) Cert A PASS — pre-sorted small-Cartesian dispatch + parity vs hash + `sort_merge_dispatch_count >= 1` + selectivity feedback wired (mirrors W4.2 Cert A); (2) Cert B PASS — UNSORTED-but-otherwise-eligible inputs fall back (`sort_merge_dispatch_count == 0`, `nested_loop_dispatch_count >= 1` or hash, row-set parity); (3) Cert C PASS — above-threshold sorted inputs fall back to hash (`sort_merge_dispatch_count == 0`, parity); (4) Cert D PASS — multi-col composite key fallback (`sort_merge_dispatch_count == 0`); (5) Cert D' PASS — non-Inner (Semi) fallback; (6) Cert E PASS — Symbol-typed key dispatch; (7) Cert F PASS — duplicate-key run-length matching produces correct row count + parity (the spike-validated regime); (7') **Cert G PASS (per F-W43-4) — empty-input dispatch**: at least one cell with `num_left == 0` and one cell with `num_right == 0`; assert `sort_merge_dispatch_count` reflects the chosen short-circuit (either dispatched-with-empty-output via the provider's empty fast path OR not-dispatched-via-eligibility); row-set parity vs hash (both should produce empty); no kernel-launch crash; (8) **Post-implementation bench (per F-W43-3 + F-W43-2)** shows the EXECUTOR DISPATCH PATH including BOTH detection kernel calls AND the chosen-operator launch wins by **≥ 2×** vs hash on the eligible envelope. Timed region MUST be `Executor::execute_plan` or equivalent end-to-end dispatch path (NOT `provider.sort_merge_join_v2_inner_u32_1key` direct call) so the detection cost is included in the reported numbers. The bench MUST also include side-by-side overlap cells where the same fixture is run twice — once routed through sort-merge dispatch and once forced through nested-loop — to empirically validate the D2 precedence decision; if the comparison shows nested-loop wins on the overlap, iteration-N+ amends D2; (9) all other slice-1/2/4 + W4.1 + W4.2 tests PASS (no regressions); (10) zero workspace warnings on touched files; (11) `cargo fmt --check --all` clean; (12) **(per F-W43-12 gate exception)** `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` exits 0 for every test path EXCEPT `crates/xlog-cuda/tests/test_wcoj_layout_fast_path.rs`, which is a pre-existing flake reproducible on merge-base 19f7bc5d (W4.2 closure HEAD) and out-of-W4.3-scope to fix (data-corruption signature consistent with missing stream-synchronize between kernel launch and D2H download in the v0.6.2 WCOJ layout fast-path code); the exception is narrow — every OTHER workspace-test path must still exit 0; (13) `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1 (the **authoritative gate** per MEMORY.md, which compensates for the (12) exception); (14) post-impl bench evidence committed to `docs/evidence/<YYYY-MM-DD>-w43-production-bench/README.md`. |
| **D8** | **Process locks**. | No board edit. No DONE marking. No FF-merge until separately authorized. No env-knob additions (`XLOG_SORT_MERGE_*` etc. forbidden). No `RuntimeConfig` field additions. The threshold is the existing `NESTED_LOOP_TOTAL_THRESHOLD` (shared constant, not config-tunable in v0.6.5). The existing dead `JoinStrategy::SortMerge` enum at `crates/xlog-runtime/src/statistics.rs:15` is NOT touched (mirrors W4.2's leave-the-dead-enum-alone discipline). The bench-spike branch (`bench-spike/w43-sort-merge`) stays unmerged — W4.3 does NOT graduate spike code (the production kernel + provider are written fresh with the empty-input fast path, byte-length validation, etc., that the spike skipped). |

## Read-Only Surface (recon results, augmented post-spike)

* **Existing dead-code design layer** (W4.3 leaves untouched per D8):
  * `crates/xlog-runtime/src/statistics.rs:15` — `JoinStrategy::SortMerge` enum variant. Zero production consumers.
* **Production hash-join dispatch site** (W4.3 wires after W4.2's branch):
  * `crates/xlog-runtime/src/executor/node_dispatch.rs::execute_join` — currently has W4.2's nested-loop branch + adaptive indexing + hash fallback. W4.3 inserts a sort-merge branch BEFORE the nested-loop branch (per D2 precedence).
* **GPU kernel infrastructure**:
  * `crates/xlog-cuda/kernels/sort.cu` — radix-sort family. W4.3 appends `check_ascending_sorted_u32` (D1 detection kernel).
  * `crates/xlog-cuda/kernels/join.cu` — hash-join + nested-loop kernel families. W4.3 appends `sort_merge_join_inner_u32_1key_pairs` (production kernel; the spike kernel `sort_merge_join_inner_u32_1key_pairs_spike` does NOT graduate).
  * `crates/xlog-cuda/src/provider/relational.rs` — provider fns. W4.3 adds `sort_merge_join_v2_inner_u32_1key` and `is_sorted_ascending_u32` alongside W4.2's `nested_loop_join_v2_inner_u32_1key`.
* **Existing dispatch-counter pattern** (W4.3 mirrors): `wcoj_*_dispatch_count` + `nested_loop_dispatch_count` plain-`u64` fields on `Executor`. W4.3 adds `sort_merge_dispatch_count`.
* **Sortedness-check kernel precedent** (D1 mirrors): `wcoj_layout_check_sorted_unique_u32` at `crates/xlog-cuda/kernels/wcoj.cu` + provider entry at `crates/xlog-cuda/src/provider/wcoj.rs:3137-3187` (the "scan-and-decide" pattern is well-established; D1's kernel is a strict subset).
* **Cert template** (W4.3 mirrors): `crates/xlog-integration/tests/test_w42_nested_loop_dispatch.rs` — A/B/C/C'/E pattern with executor + RirNode::Join construction. W4.3's certs follow the same shape.

## Step-by-Step Execution Plan (13 steps, mirrors W4.2 structure)

### Step 1 — Plan iteration commit (this commit)

The current plan-iteration commit (iter 1, then amendments per F-W43-N), on `feat/w43-sort-merge-join`. No code yet at iteration approval; iteration 5 captures findings surfaced during execution. The agent does NOT advance to Step 2 until the user explicitly approves the live iteration (currently iteration 5).

Commit subject: `docs(plan): W4.3 iteration 1 — sort-merge join (post-spike, 5 mandatory locks)`.

### Step 2 — Production sort-merge kernel + sortedness-detection kernel

File: `crates/xlog-cuda/kernels/join.cu` (append) and `crates/xlog-cuda/kernels/sort.cu` (append).

* `sort_merge_join_inner_u32_1key_pairs` in `join.cu`: per-thread binary-search emit-pairs design (matches the spike kernel's algorithm; production kernel is a fresh write with hardened parameter validation).
* `check_ascending_sorted_u32` in `sort.cu`: single-pass kernel that scans `keys[0..n-1]` in pairs, atomically writes 0 to a u32 output if any `keys[i] > keys[i+1]` is found. Caller initializes the output to 1 before launch; reads the result post-launch.

Register both kernels in `crates/xlog-cuda/src/kernel_manifest_data.rs` (under `"join"` and `"sort"` modules respectively). Add kernel-name constants in `crates/xlog-cuda/src/provider/mod.rs::join_kernels` and `::sort_kernels`.

Commit subject: `feat(w43): add sort-merge inner-join kernel + sortedness-detection kernel`.

### Step 3 — Provider fns (per F-W43-5 file-path correction + F-W43-4 empty fast path)

File: `crates/xlog-cuda/src/provider/relational.rs` (edit). Per F-W43-5, sort provider methods currently live in `relational.rs` (the `pub fn sort` at line 1459 + `dedup` family) — there is NO `provider/sort.rs` file. Both new fns go in `relational.rs` alongside W4.2's `nested_loop_join_v2_inner_u32_1key`. (A future refactor extracting sort-related fns to a `provider/sort.rs` module is out of W4.3 scope.)

* `pub fn sort_merge_join_v2_inner_u32_1key(left, right, left_key, right_key) -> Result<CudaBuffer>`. Mirrors W4.2's `nested_loop_join_v2_inner_u32_1key` literal idioms (per F-W42-13/14/15/16/17): byte-length lower-bound check (`<` failure), `checked_mul` for threshold, no `?` on `create_empty_buffer`, `as u64` for `row_cap`, `combine_schemas` for output schema, `gather_buffer_by_indices` for materialization.
* `pub fn is_sorted_ascending_u32(buf, key_col) -> Result<bool>`. **Empty / single-row fast path (per F-W43-4)**: `let n = self.device_row_count(buf)?; if n < 2 { return Ok(true); }` BEFORE any allocation or kernel launch. This handles `num_rows == 0` (which would otherwise pass through the threshold check `0 * 0 <= 4M`) AND `num_rows == 1` (trivially sorted). For `n >= 2`, allocates 1-element u32 output, initializes to 1 via `htod_sync_copy_into(&[1u32], ...)` (the kernel atomically writes 0 only on detected violation), launches `check_ascending_sorted_u32` with grid `(n + 255) / 256`, reads result via `dtoh_scalar_untracked`, returns `Ok(result == 1)`.

Commit subject: `feat(w43): add sort_merge_join_v2_inner_u32_1key + is_sorted_ascending_u32 provider fns`.

### Step 4 — Eligibility predicate

File: `crates/xlog-runtime/src/executor/node_dispatch.rs` (edit).

Add a private free fn `eligible_for_sort_merge(left, right, left_keys, right_keys, join_type) -> bool` mirroring `eligible_for_nested_loop`'s shape. Same checks (Inner + 1-key + matching U32/Symbol). NO sortedness check or threshold check inside the predicate — those happen at the dispatch site (Step 5) since they require runtime data (kernel launch + row counts).

Commit subject: `feat(w43): add eligible_for_sort_merge predicate`.

### Step 5 — Dispatch counter + dispatch wiring

Files: `crates/xlog-runtime/src/executor/mod.rs` (counter field) + `crates/xlog-runtime/src/executor/node_dispatch.rs` (wiring) + `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` (accessor, alongside W4.2's accessor).

At the top of `execute_join`, BEFORE the W4.2 nested-loop branch, add (per F-W43-1 fail-closed + F-W43-4 empty handling — empties take the provider fast path internally):

```rust
// W4.3 sort-merge dispatch (precedes nested-loop per D2).
// Per F-W43-1: fail-closed on detection — Err and Ok(false) both
// fall through; never propagate detection error to caller.
// Per F-W43-4: empty inputs (num_left == 0 || num_right == 0)
// pass the threshold check (0 <= 4M) and reach detection;
// `is_sorted_ascending_u32` short-circuits n < 2 → Ok(true)
// internally, then sort_merge_join_v2_inner_u32_1key's own
// empty fast path returns the empty combined-schema buffer.
if eligible_for_sort_merge(left, right, left_keys, right_keys, join_type) {
    let num_left = self.provider.device_row_count(left)? as u64;
    let num_right = self.provider.device_row_count(right)? as u64;
    let in_threshold = num_left
        .checked_mul(num_right)
        .map(|p| p <= NESTED_LOOP_TOTAL_THRESHOLD)
        .unwrap_or(false);
    if in_threshold {
        let lk = left_keys[0];
        let rk = right_keys[0];
        // Match-on-Result, NOT `?` — fail-closed per D5 + F-W43-1.
        let left_sorted = matches!(
            self.provider.is_sorted_ascending_u32(left, lk),
            Ok(true)
        );
        let right_sorted = matches!(
            self.provider.is_sorted_ascending_u32(right, rk),
            Ok(true)
        );
        if left_sorted && right_sorted {
            out = Some(self.provider.sort_merge_join_v2_inner_u32_1key(
                left, right, lk, rk,
            )?);
            self.sort_merge_dispatch_count += 1;
        }
        // Else (Ok(false) on either side, Err on either side): fall
        // through to W4.2 nested-loop branch via `out.is_none()`.
    }
}
// existing W4.2 nested-loop branch follows (only fires if out.is_none())
```

The two `matches!(...)` calls swallow `Err(_)` deliberately: detection failures fall through to nested-loop or hash, never propagate. This is the load-bearing F-W43-1 fix relative to iteration-1's `?`-based pseudocode.

The W4.2 nested-loop branch is wrapped in `if out.is_none()` already (per Step-5 patch `82d19fd1`). W4.3's branch sets `out = Some(...)` on hit; nested-loop sees `out.is_some()` and skips. All paths converge at the shared `record_join_result` block.

Commit subject: `feat(w43): wire sort-merge dispatch + counter at execute_join (precedes nested-loop)`.

### Step 6 — Cert A: pre-sorted small-Cartesian dispatch + parity + selectivity feedback

File: `crates/xlog-integration/tests/test_w43_sort_merge_dispatch.rs` (new).

Test `pre_sorted_small_cartesian_dispatches_sort_merge_and_matches_hash`. Same shape as W4.2 Cert A but with sorted-ascending fixtures. Asserts:
* `sort_merge_dispatch_count >= 1`.
* `nested_loop_dispatch_count == 0` (sort-merge took precedence).
* `BTreeSet<[u32; 4]>` row-set parity vs `provider.hash_join_v2 Inner`.
* `executor.stats().get_join_selectivity(left_rel, right_rel).is_some()` post-execute (D6 invariant carried forward from W4.2 / W2.4).

Commit subject: `test(w43): cert A — pre-sorted small dispatches sort-merge + parity + selectivity`.

### Step 7 — Cert B: unsorted-but-otherwise-eligible falls back to nested-loop

File: same as Cert A.

Test `unsorted_eligible_falls_back_to_nested_loop`. Inputs are NOT sorted (e.g., shuffled 1-key U32 inputs at L=R=100); same eligibility envelope (Inner + 1-key + U32 + small Cartesian). Asserts:
* `sort_merge_dispatch_count == 0` (D1 detection refused).
* `nested_loop_dispatch_count >= 1` (W4.2 fallback fired per D2 precedence).
* Row-set parity vs hash reference.

Commit subject: `test(w43): cert B — unsorted eligible falls back to nested-loop`.

### Step 8 — Cert C: above-threshold sorted falls back to hash

Test `above_threshold_sorted_falls_back_to_hash`. Asymmetric sorted fixture (e.g., L=50_000 R=100, sorted), 5M Cartesian above 4M threshold. Asserts:
* `sort_merge_dispatch_count == 0` (D3 threshold refused).
* `nested_loop_dispatch_count == 0` (also above threshold).
* Row-set parity vs hash reference (hash fallback).

Commit subject: `test(w43): cert C — above-threshold sorted falls back to hash`.

### Step 9 — Cert D + D': multi-col key + Semi fallback

Mirrors W4.2 Certs C/C'. Multi-col key fallback `sort_merge_dispatch_count == 0` (D4 disqualified). Semi join fallback `sort_merge_dispatch_count == 0` AND `nested_loop_dispatch_count == 0` (both Inner-only).

Commit subject: `test(w43): cert D + D' — multi-col key + Semi fall back to hash`.

### Step 10 — Cert E + Cert F + Cert G: Symbol-typed + duplicate-key + empty (per F-W43-4)

Cert E: Symbol-keyed sorted small inner join → sort-merge dispatched + parity (mirrors W4.2 Cert E).
Cert F: duplicate-key sorted 2-col fixture (e.g., 250 keys × 4× dup → 1000 rows each side, 4000 output rows; mirrors the spike's regime (b)) → sort-merge dispatched + parity vs hash + asserts output row count == 4000.
**Cert G (per F-W43-4)**: empty-input fixtures — at least one cell with `num_left == 0` (right populated) and one cell with `num_right == 0` (left populated). Asserts no kernel-launch crash AND row-set parity (both should produce empty output, possibly via different paths — sort-merge's `is_sorted_ascending_u32` short-circuits `n < 2` to `Ok(true)`, then `sort_merge_join_v2_inner_u32_1key`'s empty fast path returns the empty combined-schema buffer; hash takes its own empty fast path at `relational.rs:3165-3170`).

Commit subject: `test(w43): cert E + F + G — Symbol-typed + duplicate-key + empty-input dispatch`.

### Step 11 — Workspace gate

Mirrors W4.2 Step 11. fmt + warnings + workspace tests + CUDA cert suite. Pass-count delta = **+8** per the Acceptance Grid (Certs A, B, C, D, D', E, F, G — Cert G added per F-W43-4). The Acceptance Grid is canonical for the count.

**Workspace-test gate exception (per F-W43-12)**: `cargo test -p xlog-cuda --release --test test_wcoj_layout_fast_path` is a **pre-existing flake** unrelated to W4.3 — failures reproduce on the merge-base 19f7bc5d (W4.2 closure HEAD) when run alongside other CUDA tests; data-corruption signature is consistent with missing stream-synchronize between kernel launch and D2H download in the v0.6.2 WCOJ layout fast-path code (out-of-W4.3-scope to fix). Step 11's workspace-test gate is satisfied iff:
* `cargo fmt --check --all` exits 0,
* `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` exits 0,
* `cargo test -p xlog-cuda-tests --test certification_suite --release` is green (the **authoritative gate** per MEMORY.md),
* `cargo test -p xlog-runtime --release` is green,
* `cargo test -p xlog-integration --release --tests` is green (all integration test binaries pass, including W4.3 dispatch certs 8/8 and W4.2 dispatch certs 5/5),
* AND every test outside `test_wcoj_layout_fast_path` in the canonical workspace command exits 0.

The flake is documented in this plan and deferred to follow-up work on the v0.6.2 WCOJ fast-path code.

Commit subject (if any cleanup): `chore(w43): workspace gate green pre-bench`.

### Step 12 — Post-implementation bench (per F-W43-3 timed-region clarification + F-W43-2 overlap cells)

File: `crates/xlog-integration/benches/w43_production_sort_merge_bench.rs` (new).

**Two-part bench design:**

* **Part A — Executor-dispatch-path timing**: timed region is `Executor::execute_plan` (or equivalent end-to-end dispatch path), NOT a direct `provider.sort_merge_join_v2_inner_u32_1key` call. The detection kernel cost (`is_sorted_ascending_u32` × 2 sides) and the eligibility predicate are INSIDE the timed region. Compares end-to-end sort-merge dispatch vs end-to-end hash dispatch on multi-col fixtures matching production eligibility. **D7 acceptance #8 is satisfied iff this path wins ≥ 2× vs hash on eligible cells.** Per F-W43-3, this resolves the iteration-1 timed-region ambiguity: the bench measures what production traffic actually pays, including detection.

* **Part B — D2 precedence overlap validation (per F-W43-2)**: at least 3 cells where the same fixture is run twice — once with `RuntimeConfig::default()` (sort-merge dispatched per D2) and once with sort-merge dispatch disabled (forces nested-loop fallback under the same eligibility envelope). Compares end-to-end timings. If the comparison shows nested-loop wins on the overlap, iteration-N+ amends D2 to nested-loop-first. The disable mechanism is a test-only construct (e.g., temporary direct-hash + direct-nested-loop provider calls bypassing the eligibility predicate); does NOT add a `RuntimeConfig` field per D8.

Output: `docs/evidence/<YYYY-MM-DD>-w43-production-bench/README.md` with median timings + speedup table for Part A + Part B's overlap-comparison data + decision-validation conclusion (D2 precedence held vs needs amendment).

Commit subject: `feat(w43): add production sort-merge bench + evidence (executor-dispatch-path + D2-overlap validation)`.

### Step 13 — Closure proposal (text-only)

Plan-iteration commit + Steps 2–12 commits on `feat/w43-sort-merge-join`. No board edit. No FF-merge. No advance until separate user approval.

## Acceptance Grid (iteration-5 canonical)

| Cell | Count | Test file | Acceptance criterion |
|------|-------|-----------|----------------------|
| **Cert A — pre-sorted small dispatch** | 1 | `test_w43_sort_merge_dispatch.rs` (new) | sort_merge_dispatch_count >= 1 + nested_loop == 0 + parity + selectivity feedback |
| **Cert B — unsorted falls back to nested-loop** | 1 | same | sort_merge == 0 + nested_loop >= 1 + parity |
| **Cert C — above-threshold falls back to hash** | 1 | same | sort_merge == 0 + nested_loop == 0 + parity |
| **Cert D — multi-col key fallback** | 1 | same | sort_merge == 0 + parity |
| **Cert D' — Semi join fallback** | 1 | same | sort_merge == 0 + nested_loop == 0 + Semi parity |
| **Cert E — Symbol-typed dispatch** | 1 | same | sort_merge >= 1 + parity |
| **Cert F — duplicate-key run-length dispatch** | 1 | same | sort_merge >= 1 + output count == 4000 + parity |
| **Cert G — empty-input dispatch (per F-W43-4)** | 1 | same | no kernel-launch crash; empty output via either path; row-set parity |
| **Post-impl bench Part A** | 1 | `w43_production_sort_merge_bench.rs` (new) | Executor-dispatch-path including detection wins ≥ 2× vs hash on eligible cells |
| **Post-impl bench Part B (per F-W43-2)** | 1 | same file | Side-by-side overlap cells confirm D2 precedence (sort-merge > nested-loop) OR surface a counter-finding |
| **Workspace pass-count delta** | **+8** | — | 8 new test cells (A, B, C, D, D', E, F, G — Cert G added per F-W43-4). D folded as parity tail. |

## Source-of-Truth References (iteration-5 canonical)

* Spike evidence: `docs/evidence/2026-05-10-w43-bench-spike/README.md` (on `bench-spike/w43-sort-merge`); `fadc2700` HEAD.
* Recon: `docs/plans/2026-05-08-w43-sort-merge-join-recon.md`.
* W4.2 cert template: `crates/xlog-integration/tests/test_w42_nested_loop_dispatch.rs`.
* W4.2 production reference (provider + dispatch): `crates/xlog-cuda/src/provider/relational.rs::nested_loop_join_v2_inner_u32_1key` + `crates/xlog-runtime/src/executor/node_dispatch.rs::execute_join` (W4.3 mirrors this structure).
* Sortedness-check kernel precedent: `crates/xlog-cuda/kernels/wcoj.cu::wcoj_layout_check_sorted_unique_u32` + `provider/wcoj.rs:3137`.
* Existing `JoinStrategy::SortMerge` dead-code: `crates/xlog-runtime/src/statistics.rs:15` (untouched).

## Risk Register (informational, iteration-5 canonical)

| Risk | Mitigation |
|------|------------|
| Detection kernel cost erodes the speedup | D1 detection is a single-pass `O(L+R)` scan + 1 D2H scalar — bounded at ~5-50 µs vs sort-merge's ~300 µs win over hash. Step 12 post-impl bench validates net speedup ≥ 2× INCLUDING detection cost. |
| Sort-merge wins in spike but loses in production due to multi-col gather overhead | Spike's 2-col duplicate-key cell already exercises gather (2.56× win). Step 12 bench at production arity validates. |
| Threshold mismatch between W4.2 and W4.3 | D3 explicitly shares the constant. Iteration-1 lock prevents drift. |
| Sort-merge dispatch overrides nested-loop in cases where nested-loop is faster | Spike doesn't cross-compare sort-merge vs nested-loop directly. Step 12 bench could include a side-by-side comparison cell at small sorted Cartesian inputs to confirm sort-merge precedence is empirically right. |
| Detection kernel reports "unsorted" on edge cases (single-row inputs, empty inputs) | Per F-W43-4: empty AND single-row inputs (`n < 2`) are short-circuited by `is_sorted_ascending_u32`'s **own internal fast path** (return `Ok(true)` BEFORE allocation/launch) — they enter detection AFTER the threshold check admits the join (`0 * 0 = 0 ≤ 4M = true`). The detection kernel never launches with grid_dim 0. Once detection returns `Ok(true)` for an empty side, `sort_merge_join_v2_inner_u32_1key`'s own empty fast path returns the empty `combine_schemas` buffer (mirrors `hash_join_inner_v2`'s empty handling at `relational.rs:3165-3170`). Cert G covers both empty-left and empty-right fixtures. |
| `JoinStrategy::SortMerge` dead-enum confusion | NOT touched per D8. Future cleanup commit (out of W4.3) can delete the enum entirely. |

## Plan-Approval Gate (iteration 5)

This plan is **iteration 5** (iteration 4 was approved and Steps 2-11 executed; iteration 5 retroactively captures two execution-discovered findings — F-W43-11 from Step 5 review and F-W43-12 from Step 11 verification — to keep the plan record aligned with executed work). Iterations 1-4 follow the standard pre-execution review pattern; iteration 5 is the post-execution amendment pattern documented in the Iteration-5 Amendment Log. The live D-table + Step plan + Acceptance Grid above remain the canonical source of truth.

Common amendment vectors per the W4.2 / W4.1 plan-iteration discipline:
* Threshold sharing decision (D3) — could push back to introduce a separate constant, or argue for a different value.
* Dispatch precedence (D2) — could push back to put nested-loop first if spike data interpretation differs.
* Detection mechanism (D1) — could push back toward producer-side metadata (option C) for v0.6.5 if iteration-1 analysis surfaces a benchmark concern about detection cost.
* Hash-fallback policy (D5) — could push back to include sort-then-merge if a follow-up spike shows it's viable.
* Cert surface (D7 + Acceptance Grid) — additions, deletions, fixture-shape clarifications.
* Anything else.

The agent does NOT modify the live D-table / Step plan / Acceptance Grid based on chat alone — every amendment lands as a new iteration commit (e.g., `docs(plan): W4.3 iteration N amendment — F-W43-X..Y (severity counts)`).

## Iteration 1 Notes (historical / superseded)

* No paper-claim alignment is required — sort-merge join is internal optimization, not paper-grounded closure.
* Spike evidence is treated as **load-bearing input**: the threshold value (4M, shared with W4.2), the post-impl bench acceptance (≥ 2×), and the dispatch-precedence working hypothesis (sort-merge > nested-loop) are all derived from spike measurements. Subsequent iterations refine these per F-W43-N findings; the spike evidence README is the canonical reference for any spike-derived claim.
* The spike kernel does NOT graduate to production. Production kernel + provider are written fresh with all the F-W42-13..17 idioms (no `?` on `create_empty_buffer`, `as u64` for `row_cap`, byte-length lower-bound `<` check, `checked_mul` for threshold, etc.) that the spike skipped.

## Iteration-2 Amendment Log

User review of iteration 1 surfaced 1 blocking + 3 major + 2 minor findings. Live D-table (D1, D2, D5, D7), Step plan (Steps 3, 5, 10, 12), and Acceptance Grid all rewritten in place to be **iteration-2 canonical**. Header iteration tag bumped 1 → 2.

| ID | Severity | Finding | Iteration-1 (wrong) | Iteration-2 (corrected) |
|----|----------|---------|---------------------|--------------------------|
| **F-W43-1** | Blocking | D5 specified fail-closed semantics, but Step 5 pseudocode used `?` on `is_sorted_ascending_u32` — propagating `Err(_)` to the caller and contradicting the fail-closed contract | Step 5 had `let left_sorted = self.provider.is_sorted_ascending_u32(left, lk)?;` | Step 5 + D2 + D5 corrected: dispatch site uses `matches!(self.provider.is_sorted_ascending_u32(...), Ok(true))` to swallow `Ok(false)` AND `Err(_)` and fall through. D5 lock explicitly forbids `?` on the detection call. |
| **F-W43-2** | Major | D2 precedence claim under-evidenced — spike compared sort-merge-vs-hash + W4.2 compared nested-loop-vs-hash, but no benchmark directly compared sort-merge vs nested-loop on overlap with detection cost included | D2 stated precedence as if validated; Step 12 didn't require overlap measurement | D2 marked **PROVISIONAL**; Step 12 expanded to two-part bench (Part A executor-dispatch-path timing + Part B side-by-side overlap-validation cells). If overlap shows nested-loop wins, iteration-N+ amends D2. |
| **F-W43-3** | Major | Step 12 ambiguous about whether timed region included detection cost — said both "production kernel + dispatch path" and "Provider-direct envelope-parity"; the latter excludes executor detection | Step 12 ambiguous; D7 #8 implicitly allowed provider-direct interpretation | Step 12 + D7 #8 corrected: timed region MUST be `Executor::execute_plan` end-to-end, NOT direct provider call. Detection kernel cost (×2) is INSIDE the timed region. The bench measures what production traffic actually pays. |
| **F-W43-4** | Major | Empty inputs (num_left == 0 OR num_right == 0) pass the threshold check (0 ≤ 4M) and reach detection; without a fast path, `is_sorted_ascending_u32` would launch with grid_dim 0 (undefined). No empty-input cert in iteration-1 grid | D1 didn't specify empty handling; Step 3 didn't lock the fast path; no Cert G | D1 + Step 3 + Step 5 all reference the empty fast path: `is_sorted_ascending_u32` checks `n < 2 → Ok(true)` BEFORE allocation/launch. Cert G added (per F-W43-4) — empty-input dispatch parity. Acceptance Grid pass-count delta updated +7 → +8. |
| **F-W43-5** | Minor | Step 3 named `crates/xlog-cuda/src/provider/sort.rs` but no such file exists; sort provider methods live in `relational.rs` | Step 3 said "sort.rs (or wherever sort provider fns currently live)" | Step 3 corrected to specify `relational.rs` only. Note that a future refactor extracting sort fns to a new `provider/sort.rs` module is out of W4.3 scope. |
| **F-W43-6** | Minor | Multiple count-drift sites: "12 steps" vs Step 13 exists; "+6 placeholder" vs grid "+7"; "~430 lines" vs file actually ~239 lines | Header at line 52: "12 steps"; Step 11 line: "+6 placeholder"; Iteration 1 Notes line 236: "~430 lines" | Step header corrected to "13 steps". Step 11 placeholder dropped (Acceptance Grid is canonical). Iteration 1 Notes line-count claim removed; future iterations track line count via the file itself if needed. |

**Net effect:** D1 (empty fast path), D2 (provisional precedence), D5 (fail-closed `match`-not-`?`), D7 (executor-dispatch-path timing + Cert G + +8 delta), Step 3 (file path), Step 5 (`matches!` pseudocode + empty doc-comment), Step 10 (Cert G added), Step 12 (two-part bench design). Acceptance Grid expanded from 7 certs + 1 bench to 8 certs + 2 bench parts. Header iteration tag bumped 1 → 2.

**Process note**: per the W4.2 plan-iteration discipline, all amendments are surfaced as F-W43-N findings with explicit before/after states. The agent does NOT modify the live D-table / Step plan based on chat alone — every amendment lands as a new iteration commit.

## Iteration-3 Amendment Log

User review of iteration 2 surfaced 1 major + 1 minor finding. Both are residual stale-text drift left over from iteration 1 — content that wasn't rewritten when iteration 2 fixed the surrounding sections. Patches are surgical (no structural change to D-table, Step plan, or Acceptance Grid). Header iteration tag bumped 2 → 3.

| ID | Severity | Finding | Iteration-2 (wrong) | Iteration-3 (corrected) |
|----|----------|---------|---------------------|--------------------------|
| **F-W43-7** | Major | Step 11 still contained the iter-1 placeholder "Pass-count delta = +6 (... actually 7; placeholder count, will be exact after implementation)" — directly contradicting iter-2's Acceptance Grid which is "+8 (Cert G added)" | Step 11: "Pass-count delta = +6 ... actually 7; placeholder count" | Step 11 says "Pass-count delta = **+8** per the Acceptance Grid (Certs A, B, C, D, D', E, F, G — Cert G added per F-W43-4). The Acceptance Grid is canonical for the count." — single source of truth (the Grid) referenced explicitly. |
| **F-W43-8** | Minor | Risk Register row about edge-case detection said "Empty-input handled by D3's fast path **before** detection" — but the iter-2 design handles empties INSIDE `is_sorted_ascending_u32` AFTER the threshold check admits the join (`0 ≤ 4M = true`), then the join provider's own empty fast path returns the empty `combine_schemas` buffer | Row text: "by D3's fast path before detection" | Row rewritten to match the actual iter-2 design: empties enter detection AFTER threshold admits the join, then `is_sorted_ascending_u32`'s INTERNAL `n < 2 → Ok(true)` fast path short-circuits BEFORE allocation/launch, then `sort_merge_join_v2_inner_u32_1key`'s own empty fast path returns the empty buffer. Cites Cert G as coverage. |

**Net effect**: 2 surgical text patches (Step 11 placeholder; Risk Register row). No D-table changes, no Step plan structural changes, no Acceptance Grid changes. All iter-2 design decisions preserved unchanged.

**Iteration-3 process observation**: F-W43-7 and F-W43-8 are residual drift — text that should have been rewritten in iteration 2 alongside the related D-table/Grid edits but wasn't. Iteration 2's amendment scope was D1/D2/D5/D7 + Steps 3/5/10/12 + Grid; the iter-1 lines at Step 11 and Risk Register weren't included even though they referenced the same content. A future plan-discipline improvement: when amending a count or design fact, grep the entire plan file for related text before declaring the iteration complete. Otherwise residual-drift findings continue to cost iterations.

## Iteration-4 Amendment Log

User review of iteration 3 surfaced 1 major + 1 minor finding — live section-heading and approval-gate label drift left over from iterations 1–2. Iteration-3 patched Step 11 + Risk Register but left section headings and Step 1 approval-gate text at older iteration labels. Iteration 4 sweeps the remaining 5 label sites in a single surgical pass. No design changes; D-table, Step plan, and Acceptance Grid content unchanged. Header iteration tag bumped 3 → 4.

| ID | Severity | Finding | Iteration-3 (wrong) | Iteration-4 (corrected) |
|----|----------|---------|---------------------|--------------------------|
| **F-W43-9** | Major | Step 1 (line 56) said "agent does NOT advance to Step 2 until the user explicitly approves iteration 1" — wrong gate-target after multiple iterations | "approves iteration 1" | "approves the live iteration (currently iteration 4)" + first-line generalized to "the current plan-iteration commit (iter 1, then amendments per F-W43-N)" |
| **F-W43-10** | Minor | Four section-heading labels at older iterations:<br>• Direction (line 25): "iteration 1"<br>• Acceptance Grid (line 211): "iteration-2 canonical"<br>• Source-of-Truth (line 227): "iteration-1 canonical"<br>• Risk Register (line 236): "iteration 1" | mixed labels (1/2 with iter-3 content) | All four bumped to "iteration 4 canonical" / "iteration-4 canonical". Headers now uniformly reflect the current canonical iteration. |

**Net effect**: 5 surgical label-bump edits (header + Direction + Step 1 gate text + Acceptance Grid + Source-of-Truth + Risk Register + Plan-Approval Gate). No D-table, Step plan, or Acceptance Grid content changes. All iter-3 design decisions preserved unchanged.

**Iteration-4 process observation (extends iter-3's)**: F-W43-9/10 are the same class of residual drift as F-W43-7/8 — *content-matching* iter-3's amendment scope but missed because the labels are in different sections than the rewritten content. Future plan-discipline improvement compounding iter-3's: when bumping the iteration tag, grep for `iteration \d` and `iteration-\d` (both spellings) across the entire file and either bump or explicitly justify each occurrence. Cumulative lesson: iteration-bumping is a *file-wide concern*, not a section-local one.

## Iteration-5 Amendment Log

Iteration 5 captures two findings surfaced during execution rather than during plan review: F-W43-11 (retroactively logged from Step 5) and F-W43-12 (workspace-test gate exception surfaced during Step 11 verification). Both findings landed operationally before this amendment commit; iteration 5 brings the plan record into alignment with the executed work. Header iteration tag bumped 4 → 5.

| ID | Severity | Finding | Pre-iter-5 (wrong / silent) | Iter-5 (corrected) |
|----|----------|---------|-----------------------------|--------------------|
| **F-W43-11** | Major | D2 precedence (sort-merge > nested-loop) silently broke W4.2 Cert A and Cert E: their fixtures used trivial `(0..N).map(|i| (i, ...))` sorted-ascending keys, so after Step 5's W4.3 dispatch landed the sort-merge path took precedence and `nested_loop_dispatch_count == 1` failed. The W4.2 certs were no longer regression-detecting for the nested-loop path. | Step 5 wiring did not document a fixture-de-overlap requirement; W4.2 cert files used sorted ascending keys; no explicit guard in the plan. | Step 5 + W4.2 fixture de-overlap bundled into one commit (4ef14855). Cert A and Cert E now use rotate-halves `(N/2..N).chain(0..N/2)` — minimum-violation unsorted shape, deterministic, same key set / same row-set / same match counts as before. Bundled commit message documents both halves. Future plan-discipline improvement: dispatch-precedence changes that admit a new strategy MUST audit existing-strategy positive certs for shape collision before commit. |
| **F-W43-12** | Major | Step 11's workspace-test gate (`cargo test --workspace --release --exclude pyxlog`) does not exit 0 because `crates/xlog-cuda/tests/test_wcoj_layout_fast_path.rs` fails non-deterministically — even under `--test-threads=1`. Confirmed pre-existing: the same flake reproduces on merge-base 19f7bc5d (W4.2 closure HEAD) when running the test file alongside the rest of the workspace. Failure signature (last-row-only data corruption + u64 high-bits leaking into u32 reads) is consistent with missing stream-synchronize between kernel launch and D2H download in the v0.6.2 WCOJ layout fast-path code. | Step 11 implied the canonical workspace command must exit 0; the flake was documented in the Step 11 commit message but the gate criterion was not relaxed. | Step 11 amended in this iteration to record an explicit gate exception: every workspace-test path EXCEPT `test_wcoj_layout_fast_path` exits 0, and the cert suite (the authoritative gate per MEMORY.md) passes. Fixing the flake is **out-of-W4.3-scope** because (a) the failures are in v0.6.2 WCOJ code unrelated to W4.3, and (b) the merge-base reproduces the flake. Deferred to follow-up work on the v0.6.2 fast-path code. The Step 11 commit message references this F-W43-12 gate exception. |

**Iteration-5 process observation**: F-W43-11 and F-W43-12 are a different class from F-W43-7/8/9/10 — they are *execution-discovered* findings that could not have been surfaced by plan review alone (F-W43-11 required the W4.3 dispatch wiring to actually exist in source; F-W43-12 required running the workspace gate against the live CUDA environment). The pattern teaches: post-execution amendment commits are part of the plan-iteration discipline, not a violation of it; the discipline says "every amendment lands as a new iteration commit," and that includes amendments motivated by reality after the fact. The retroactive F-W43-11 entry is an explicit record of that pattern.
