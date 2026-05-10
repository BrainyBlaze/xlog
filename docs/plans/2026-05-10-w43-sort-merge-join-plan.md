# W4.3 Sort-Merge Join Operator — Plan (iteration 6 canonical)

**Plan iteration:** 6 (post-execution amendment capturing F-W43-14: the Step 12 production bench surfaced the counter-finding F-W43-2 was designed to anticipate — D2 precedence + D7 #8 BOTH FAIL on every cell of the 50×50–2000×2000 sorted-eligible matrix. Per user direction, W4.3 closure scope changes from "production dispatch closure" to "operator implemented, production dispatch rejected by evidence": executor wiring removed, dispatch counter + accessor + eligibility predicate removed, dispatch-only certs (B/C/D/D') retired as superseded, operator-meaningful certs (A/E/F/G) rewritten as provider/operator parity certs, provider/kernel/manifest surface preserved, bench evidence preserved as the rejection record. Iterations 1–4 were pre-execution review; iteration 5 was post-execution alignment of plan-record with executed work (F-W43-11/12/13); iteration 6 is the post-bench scope-redirect amendment.).
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

## Direction (locked, iteration 6 canonical)

| ID | Lock | Direction |
|----|------|-----------|
| **D1** | **Sortedness detection mechanism (per F-W43-4 empty-input handling + F-W43-14 operator-only scope)**. | **Option B (runtime detection kernel)** per recon. Kernel `check_ascending_sorted_u32` in `crates/xlog-cuda/kernels/sort.cu` — single-pass scan, returns `1` if `keys[i] <= keys[i+1]` for all `i`, else `0`. Provider fn `provider.is_sorted_ascending_u32(buf, key) -> Result<bool>` wraps the kernel. **Empty / single-row fast path (per F-W43-4)**: provider fn checks `device_row_count(buf)? < 2` BEFORE any allocation or kernel launch and returns `Ok(true)` (a 0- or 1-row sequence is trivially sorted). The fast-path semantic mirrors `hash_join_inner_v2`'s empty handling at `relational.rs:3165-3170`. For `n >= 2` rows the kernel runs, reads the u32 result via `dtoh_scalar_untracked` (single-u32 D2H, same metadata-only profile as `hash_join_v2`'s row-count reads). Mirrors the established WCOJ layout fast-path pattern at `crates/xlog-cuda/src/provider/wcoj.rs:3137-3187` (u32) and `:3265-3307` (u64), but checks "sorted ascending" only — duplicates allowed (sort-merge handles run-length). The kernel does NOT check uniqueness. **Iteration-6 caller surface (per F-W43-14)**: `is_sorted_ascending_u32` has NO executor-dispatch caller after iteration-6 unwiring — its only callers are operator-level certs (Cert G's empty-input subcases) and the Step 12 production bench (Path 1 detection-cost measurement). Iteration 1–5 documented an `execute_join` dispatch caller; that caller is removed by Step 4'. The provider fn + kernel + manifest entry remain as graduated implementation work for any future v0.6.6+ caller (e.g., a hypothetical sort-merge dispatch path with kernel-perf improvements). **Out of scope for W4.3**: producer-side metadata tracking (option C) and IR-level annotation (option D); both are larger structural changes that can be considered in v0.6.6+ alongside dispatch-perf investigation. |
| **D2** | **Dispatch precedence vs W4.2 nested-loop (REJECTED per F-W43-14, was PROVISIONAL per F-W43-2)**. | **W4.3 has NO production dispatch.** The iteration-1 working hypothesis (sort-merge > nested-loop > hash) was empirically rejected by the Step 12 bench: nested-loop wins 1.25×–2.46× on every cell of the 50×50–2000×2000 sorted-eligible matrix, and sort-merge fails to reach the D7 #8 ≥ 2× threshold vs hash on any cell either. The iteration-6 production decision tree at `execute_join` reverts to:<br>1. Eligible for nested-loop envelope (Inner + 1-key + matching U32/Symbol + ≤ 4M Cartesian)? → **nested-loop** (W4.2's existing path, unchanged).<br>2. Else → **hash** (existing fallback, unchanged).<br><br>The W4.3 sort-merge operator is implemented + benched + provider-cert-tested but is **not wired into the executor's dispatch decision tree**. Iteration-6 unwiring removes: (a) `eligible_for_sort_merge` predicate, (b) the `if eligible_for_sort_merge { ... }` branch in `execute_join`, (c) the `if out.is_none()` guard wrapping the W4.2 branch (no longer needed since W4.3 cannot consume the slot). The W4.2 branch returns to its pre-Step-5 unwrapped form. Per F-W43-2's anticipated outcome: *"If the bench shows nested-loop wins on the overlap, iteration-N+ amends D2."* Iteration 6 IS that amendment. |
| **D3** | **Memory-safe output sizing (operator-level invariant per F-W43-14)**. | The `provider.sort_merge_join_v2_inner_u32_1key` operator is **not** thresholded internally — callers must ensure inputs satisfy a memory-safe Cartesian bound before invoking. Iteration 1–5 wired the threshold check at the dispatch site in `execute_join` (matching W4.2's `NESTED_LOOP_TOTAL_THRESHOLD = 4_000_000`); after iteration-6 unwiring, the threshold is enforced ONLY by callers (currently: Cert F at 1M Cartesian, Cert A/E at 10K Cartesian, Cert G at 0 Cartesian, Step 12 bench cells from 2.5K to 4M Cartesian — all naturally below the W4.2 production threshold so the operator is safe in all current callers). Output worst-case remains `L * R` for arbitrary key distributions; any future v0.6.6+ caller that does not bound input sizes must add its own check. **Constant-sharing decision retained**: when iteration-1 dispatch was live, both W4.2 (nested-loop) and W4.3 (sort-merge) used the SAME `NESTED_LOOP_TOTAL_THRESHOLD` constant; iteration 6 removes W4.3's caller-side use of the constant in `execute_join` but does NOT delete the constant (W4.2 still uses it; nothing changes for W4.2). **Future iteration**: a sort-merge-specific threshold (e.g., empirical-distribution-based dynamic threshold) is out of W4.3 scope and would only be relevant if a future v0.6.6+ dispatch path is reintroduced. |
| **D4** | **Schema/key-type admissibility (operator-input precondition per F-W43-14)**. | `provider.sort_merge_join_v2_inner_u32_1key` requires (a) Inner join semantics implicit in the operator's name; (b) exactly **1 key column index** per side (`left_key: usize`, `right_key: usize`); (c) **left and right key column types EQUAL** AND that shared type is `ScalarType::U32` OR `ScalarType::Symbol`; (d) callers should bound input sizes per D3; (e) callers may pre-check sortedness via `provider.is_sorted_ascending_u32` (the operator does NOT pre-check internally — sortedness is a caller-supplied invariant). Iteration 1–5 framed this as an *executor dispatch eligibility predicate* (`eligible_for_sort_merge`); after iteration-6 unwiring the predicate is removed and the same checks become **operator-input preconditions** that callers (currently certs + bench) satisfy by construction. Multi-key, non-Inner, non-U32/Symbol, mismatched-type, above-threshold, or unsorted inputs are NOT valid arguments to the operator — there is no fall-back; passing invalid inputs is undefined behavior at the operator level (as opposed to the iteration-1 design where the dispatch site rejected such inputs and routed to nested-loop/hash). |
| **D5** | **Hash-fallback policy on detection failure (SUPERSEDED per F-W43-14, was per F-W43-1 fail-closed lock)**. | **Iteration 1–5 historical**: the executor dispatch site at `execute_join` was FAIL-CLOSED on `is_sorted_ascending_u32`'s return — `Ok(false)` AND `Err(_)` both fell through to W4.2 nested-loop OR hash, never propagating detection errors to the caller. The dispatch site used `matches!(provider.is_sorted_ascending_u32(...), Ok(true))` rather than `?` to enforce this. **Iteration 6 (per F-W43-14)**: with no executor dispatch site (D2 amended), the fail-closed contract is moot — there is no dispatch decision to fail closed. The provider fn `is_sorted_ascending_u32` retains its honest `Result<bool>` return shape (caller can inspect both the `Ok(false)` outcome and any `Err(_)`); current callers (Cert G + Step 12 bench) handle the result directly. The "sort-then-merge" out-of-scope note from iteration 1–5 also becomes moot — it described a dispatch-time alternative that no longer exists. |
| **D6** | **Dispatch counter (REMOVED per F-W43-14)**. | Iteration 1–5 added `sort_merge_dispatch_count: u64` to `Executor` + accessor for cert observability. Iteration 6 removes both: with no production dispatch (D2 amended), the counter has nothing to increment and would be permanently stuck at 0 in any production session. The W4.2 `nested_loop_dispatch_count` field/accessor remains (W4.2 still dispatches). Operator-level certs (Step 6/9/10 rewrites) verify the W4.3 provider/kernel surface directly via `provider.sort_merge_join_v2_inner_u32_1key` — no executor-side observability needed. |
| **D7** | **Acceptance gates (locked, per F-W43-3 timed-region clarification + F-W43-4 empty-input cert + F-W43-12 workspace-test gate exception + F-W43-14 operator-only scope redirect).** | **Operator-level acceptance gates (post-F-W43-14):** (1) **Cert A — sorted-key operator parity**: `provider.sort_merge_join_v2_inner_u32_1key` on a sorted-ascending 100-row 1-key U32 fixture produces `BTreeSet<[u32; 4]>` row-set parity vs `provider.hash_join_v2 Inner` reference. (2) **Cert E — Symbol-typed operator parity**: same shape on Symbol-typed buffers; the operator handles Symbol (byte-identical to U32 at the kernel level). (3) **Cert F — duplicate-key operator parity**: 250 keys × 4 dups → 1000 rows each side → `provider.sort_merge_join_v2_inner_u32_1key` produces 4000 output rows, all (k, lp, rp) tuples distinct, row-set parity vs hash. Exercises the kernel's `lower_bound`/`upper_bound` run-length emit path. (4) **Cert G — empty-input operator (per F-W43-4 layered-short-circuit contract)**: two subcases (`num_left == 0`, `num_right == 0`); each routes through `provider.is_sorted_ascending_u32`'s `n < 2 → Ok(true)` short-circuit followed by `provider.sort_merge_join_v2_inner_u32_1key`'s empty-input fast path; asserts no kernel-launch crash + empty output + row-set parity vs hash empty fast path at `relational.rs:3165-3170`. (5) **W4.2 nested-loop dispatch certs (5 tests in `test_w42_nested_loop_dispatch.rs`) PASS unchanged** — production-routing guard for the post-unwiring executor (W4.2 dispatch is the only join-operator dispatch in the executor after F-W43-14). (6) **Retired dispatch-only certs (B, C, D, D')**: superseded by F-W43-14. With no executor sort-merge dispatch path, asserting `sort_merge_dispatch_count == 0` on fall-through fixtures or `dispatch_count == 1` on positive fixtures becomes meaningless. The retirement is recorded explicitly in the cert file's deletion commit + this row; W4.2 dispatch fall-through behavior on those fixture shapes (multi-col key, Semi, above-threshold) is already covered by the W4.2 cert suite. (7) **Step 12 bench evidence is the rejection record, not an acceptance gate**. The iteration-1 D7 #8 (≥ 2× vs hash) is **REJECTED**: bench data shows sort-merge wins 1.10×–1.80× vs hash, never reaching 2×. The iteration-1 D2 precedence (sort-merge > nested-loop) is **REJECTED**: bench data shows nested-loop wins 1.25×–2.46× on every overlap cell. Both rejections are documented in `docs/evidence/2026-05-10-w43-production-bench/README.md` per F-W43-2's anticipated amendment path. (8) Workspace gates unchanged: `cargo fmt --check --all` clean; zero workspace warnings on touched files; cert suite 1/1 (authoritative per MEMORY.md); `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` exits 0 for every path EXCEPT `test_wcoj_layout_fast_path` (per F-W43-12 pre-existing flake exception). |
| **D8** | **Process locks**. | No board edit. No DONE marking. No FF-merge until separately authorized. No env-knob additions (`XLOG_SORT_MERGE_*` etc. forbidden). No `RuntimeConfig` field additions. The threshold is the existing `NESTED_LOOP_TOTAL_THRESHOLD` (shared constant, not config-tunable in v0.6.5). The existing dead `JoinStrategy::SortMerge` enum at `crates/xlog-runtime/src/statistics.rs:15` is NOT touched (mirrors W4.2's leave-the-dead-enum-alone discipline). The bench-spike branch (`bench-spike/w43-sort-merge`) stays unmerged — W4.3 does NOT graduate spike code (the production kernel + provider are written fresh with the empty-input fast path, byte-length validation, etc., that the spike skipped). |

## Read-Only Surface (recon results, augmented post-spike)

* **Existing dead-code design layer** (W4.3 leaves untouched per D8):
  * `crates/xlog-runtime/src/statistics.rs:15` — `JoinStrategy::SortMerge` enum variant. Zero production consumers.
* **Production hash-join dispatch site** (W4.3 wires after W4.2's branch):
  * `crates/xlog-runtime/src/executor/node_dispatch.rs::execute_join` — has W4.2's nested-loop branch + adaptive indexing + hash fallback. **Iteration-6 outcome (per F-W43-14 D2 amendment): NO W4.3 dispatch branch is added; the sort-merge operator is implemented at the provider layer but is not invoked from the executor's dispatch decision tree.**
* **GPU kernel infrastructure**:
  * `crates/xlog-cuda/kernels/sort.cu` — radix-sort family. W4.3 appends `check_ascending_sorted_u32` (D1 detection kernel).
  * `crates/xlog-cuda/kernels/join.cu` — hash-join + nested-loop kernel families. W4.3 appends `sort_merge_join_inner_u32_1key_pairs` (production kernel; the spike kernel `sort_merge_join_inner_u32_1key_pairs_spike` does NOT graduate).
  * `crates/xlog-cuda/src/provider/relational.rs` — provider fns. W4.3 adds `sort_merge_join_v2_inner_u32_1key` and `is_sorted_ascending_u32` alongside W4.2's `nested_loop_join_v2_inner_u32_1key`.
* **Existing dispatch-counter pattern** (W4.2 keeps; W4.3 superseded per F-W43-14): `wcoj_*_dispatch_count` + `nested_loop_dispatch_count` plain-`u64` fields on `Executor`. **Iteration 1–5 historical**: W4.3 mirrored the pattern by adding `sort_merge_dispatch_count`. **Iteration 6 (per F-W43-14 D6 amendment)**: the W4.3 counter field + accessor are removed; only the W4.2 + WCOJ counters remain on `Executor`.
* **Sortedness-check kernel precedent** (D1 mirrors): `wcoj_layout_check_sorted_unique_u32` at `crates/xlog-cuda/kernels/wcoj.cu` + provider entry at `crates/xlog-cuda/src/provider/wcoj.rs:3137-3187` (the "scan-and-decide" pattern is well-established; D1's kernel is a strict subset).
* **Cert template (iteration 1–5 historical, REWRITTEN/RETIRED per F-W43-14)**: `crates/xlog-integration/tests/test_w42_nested_loop_dispatch.rs` — A/B/C/C'/E pattern with executor + RirNode::Join construction. Iteration 1–5 W4.3 certs followed this shape. Iteration 6 retires dispatch-only certs (B/C/D/D') and rewrites operator-meaningful certs (A/E/F/G) at the provider layer (no executor, no RirNode::Join — direct `provider.sort_merge_join_v2_inner_u32_1key` calls + `BTreeSet<[u32; 4]>` parity vs `provider.hash_join_v2 Inner`). The W4.2 cert file remains the production-routing guard for the post-unwiring executor.

## Step-by-Step Execution Plan (13 steps, mirrors W4.2 structure)

### Step 1 — Plan iteration commit (this commit)

The current plan-iteration commit (iter 1, then amendments per F-W43-N), on `feat/w43-sort-merge-join`. Iteration 5 captured execution-discovered drift; iteration 6 captures the post-bench scope-redirect (F-W43-14). The agent does NOT advance to iteration-6 unwiring + cert rewrite (Steps 4'/5'/6'/7'/8') until the user explicitly approves the live iteration (currently iteration 6).

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

### Step 4 — Eligibility predicate (SUPERSEDED by F-W43-14)

**Iteration 1–5 historical**: File: `crates/xlog-runtime/src/executor/node_dispatch.rs` (edit). Added a private free fn `eligible_for_sort_merge(...)` mirroring `eligible_for_nested_loop`'s shape. Operationally landed at commit `b1a5ff76`.

**Iteration 6 (per F-W43-14)**: with no production dispatch, the predicate has no caller. Iteration-6 unwiring removes `eligible_for_sort_merge` from `node_dispatch.rs`. Step 4 is replaced by **Step 4'** (F-W43-14 unwiring step) — see "Iteration-6 Replacement Steps" below.

Commit subject (historical): `feat(w43): add eligible_for_sort_merge predicate`.

### Step 5 — Dispatch counter + dispatch wiring (SUPERSEDED by F-W43-14)

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

**Iteration 6 (per F-W43-14)**: the entire dispatch wiring shown above is removed. The `if out.is_none()` wrap on the W4.2 branch is also removed (no longer needed since W4.3 cannot consume the slot). The `sort_merge_dispatch_count` field on `Executor` and its accessor in `wcoj_dispatch.rs` are removed. The kernel-name constants `SORT_MERGE_JOIN_INNER_U32_1KEY_PAIRS` and `CHECK_ASCENDING_SORTED_U32` in `provider/mod.rs` are KEPT (provider fns still use them). Operationally landed at commit `4ef14855` originally; iteration-6 reverts that commit's executor wiring while preserving its other changes (W4.2 fixture de-overlap landed in the same commit; iteration-6 keeps the de-overlap as a no-op-but-harmless change OR optionally reverts to sorted-ascending fixtures since W4.3 dispatch precedence no longer exists — decision recorded in iteration-6 unwiring commit).

Commit subject (historical): `feat(w43): wire sort-merge dispatch + counter at execute_join (precedes nested-loop)`.

### Step 6 — Cert A: pre-sorted small-Cartesian dispatch + parity + selectivity feedback (REWRITTEN by F-W43-14)

**Iteration 1–5 historical**: File `crates/xlog-integration/tests/test_w43_sort_merge_dispatch.rs` test `pre_sorted_small_cartesian_dispatches_sort_merge_and_matches_hash` asserted `sort_merge_dispatch_count == 1` + `nested_loop_dispatch_count == 0` + parity + selectivity feedback. Operationally landed at commits `e917976e` + Step 6 patch `c665bd0e`.

**Iteration 6 (per F-W43-14)**: rewritten as **Cert A (operator-level)**: `provider.sort_merge_join_v2_inner_u32_1key` on a sorted-ascending 100-row 1-key U32 fixture produces row-set parity vs `provider.hash_join_v2 Inner`. No `sort_merge_dispatch_count` assertion (field/accessor removed per F-W43-14 D6). No `nested_loop_dispatch_count == 0` assertion (no dispatch path exists). No selectivity-feedback assertion (no executor wiring). Pure operator parity at the provider layer.

Commit subject: `test(w43): rewrite cert A as operator-level provider parity (per F-W43-14)`.

### Step 7 — Cert B: unsorted-but-otherwise-eligible falls back to nested-loop (RETIRED by F-W43-14)

**Iteration 1–5 historical**: Asserted `sort_merge_dispatch_count == 0` AND `nested_loop_dispatch_count == 1` to prove D2 precedence on unsorted fixtures. Operationally landed at commit `c1eff9eb`.

**Iteration 6 (per F-W43-14)**: **RETIRED**. With no executor sort-merge dispatch path, asserting `sort_merge_dispatch_count == 0` is vacuously true and not regression-detecting. The W4.2 cert suite already verifies nested-loop dispatch on its native fixtures. Iteration-6 cert-rewrite commit deletes this test from `test_w43_sort_merge_dispatch.rs`.

### Step 8 — Cert C: above-threshold sorted falls back to hash (RETIRED by F-W43-14)

**Iteration 1–5 historical**: Asserted `sort_merge_dispatch_count == 0 AND nested_loop_dispatch_count == 0` on above-4M-Cartesian fixtures. Operationally landed at commit `c0eb3d1a`.

**Iteration 6 (per F-W43-14)**: **RETIRED**. Same reason as Cert B — vacuous in the absence of dispatch. The W4.2 cert suite already covers above-threshold hash fallback on the same fixture shape (W4.2 Cert B `large_times_small_falls_back_to_hash_above_threshold`).

### Step 9 — Cert D + D': multi-col key + Semi fallback (RETIRED by F-W43-14)

**Iteration 1–5 historical**: Cert D asserted `sort_merge_dispatch_count == 0` on multi-col composite-key fixtures. Cert D' asserted `sort_merge_dispatch_count == 0 AND nested_loop_dispatch_count == 0` on Semi-join fixtures. Operationally landed at commit `e66daf9a`.

**Iteration 6 (per F-W43-14)**: **RETIRED**. Same reason. The W4.2 cert suite already covers multi-col key + Semi fallback (W4.2 Cert C `multi_col_key_falls_back_to_hash` + Cert C' `semi_join_falls_back_to_hash`).

### Step 10 — Cert E + Cert F + Cert G: Symbol-typed + duplicate-key + empty (REWRITTEN by F-W43-14)

**Iteration 1–5 historical**: Cert E asserted Symbol-typed dispatch + counter `== 1`. Cert F asserted duplicate-key dispatch + 4000 output count + counter `== 1`. Cert G asserted empty-input dispatch via `is_sorted_ascending_u32`'s `n < 2 → Ok(true)` short-circuit + counter `== 1` per fresh-executor subcase. Operationally landed at commits `0c01f6a9` + Step 10 patch `6f25377d`.

**Iteration 6 (per F-W43-14)**: rewritten as operator-level provider parity certs.
* **Cert E (operator)**: `provider.sort_merge_join_v2_inner_u32_1key` on Symbol-typed buffers (sorted-ascending 100-row 1-key) produces row-set parity vs `provider.hash_join_v2 Inner`. Proves Symbol-typed kernel surface (byte-identical to U32 at the kernel level).
* **Cert F (operator)**: `provider.sort_merge_join_v2_inner_u32_1key` on duplicate-key sorted 2-col fixture (250 keys × 4 dups → 1000 rows each side) produces 4000 output rows, all (k, lp, rp) tuples distinct, row-set parity vs hash. Exercises the kernel's `lower_bound`/`upper_bound` run-length emit path. The 4000-row + distinctness assertions remain — they are operator correctness, not dispatch shape.
* **Cert G (operator, per F-W43-4 layered-short-circuit contract)**: two subcases — `is_sorted_ascending_u32` on `n == 0` returns `Ok(true)` via the `n < 2` short-circuit; `provider.sort_merge_join_v2_inner_u32_1key` on each empty-input combination produces an empty combined-schema buffer with no kernel-launch crash; row-set parity vs hash empty fast path at `relational.rs:3165-3170`. The contract that motivated F-W43-4 (three layered short-circuits all bottom out cleanly) remains testable at the provider layer; only the dispatch-counter assertion drops.

Commit subject: `test(w43): rewrite certs E + F + G as operator-level provider parity (per F-W43-14)`.

Commit subject: `test(w43): cert E + F + G — Symbol-typed + duplicate-key + empty-input dispatch`.

### Step 11 — Workspace gate

Mirrors W4.2 Step 11. fmt + warnings + workspace tests + CUDA cert suite.

**Iteration 1–5 historical pass-count target**: +8 dispatch certs (A, B, C, D, D', E, F, G — Cert G added per F-W43-4). Operationally satisfied at iteration-5 commit `6d3de702`.

**Iteration-6 pass-count target (per F-W43-14)**: **+4 W4.3 operator certs** (A/E/F/G after rewrite by Step 5'). Dispatch-only certs B/C/D/D' retired (superseded by W4.2 cert suite). The Acceptance Grid is canonical for the count. Final iteration-6 verification (Step 7') re-runs the workspace gate after unwiring + cert rewrite to confirm the target lands.

**Workspace-test gate exception (per F-W43-12)**: `cargo test -p xlog-cuda --release --test test_wcoj_layout_fast_path` is a **pre-existing flake** unrelated to W4.3 — failures reproduce on the merge-base 19f7bc5d (W4.2 closure HEAD) when run alongside other CUDA tests; data-corruption signature is consistent with missing stream-synchronize between kernel launch and D2H download in the v0.6.2 WCOJ layout fast-path code (out-of-W4.3-scope to fix). Step 11's workspace-test gate is satisfied iff:
* `cargo fmt --check --all` exits 0,
* `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` exits 0,
* `cargo test -p xlog-cuda-tests --test certification_suite --release` is green (the **authoritative gate** per MEMORY.md),
* `cargo test -p xlog-runtime --release` is green,
* `cargo test -p xlog-integration --release --tests` is green (all integration test binaries pass, including the post-iteration-6 W4.3 operator certs **4/4** (was 8/8 in iter 5) and W4.2 dispatch certs **5/5** unchanged),
* AND every test outside `test_wcoj_layout_fast_path` in the canonical workspace command exits 0.

The flake is documented in this plan and deferred to follow-up work on the v0.6.2 WCOJ fast-path code.

Commit subject (if any cleanup): `chore(w43): workspace gate green pre-bench`.

### Step 12 — Post-implementation bench (per F-W43-3 timed-region clarification + F-W43-2 overlap cells)

File: `crates/xlog-integration/benches/w43_production_sort_merge_bench.rs` (new).

**Two-part bench design:**

* **Part A — sort-merge-with-detection vs hash**: timed region is `provider.is_sorted_ascending_u32(left, 0)` + `provider.is_sorted_ascending_u32(right, 0)` + `provider.sort_merge_join_v2_inner_u32_1key`. Hash baseline: direct `provider.hash_join_v2`. The initial-iteration interpretation of F-W43-3 used `executor.execute_node` as the timed region but the `execute_scan`-clone overhead is identical for sort-merge AND hash dispatch in production and therefore did not differentiate the two paths; provider-direct + explicit detection on the sort-merge side preserves F-W43-3's intent (detection cost included) while keeping the comparison apples-to-apples on kernel-level work. See bench file header for the design rationale.

* **Part B — D2 precedence overlap validation (per F-W43-2)**: same cell matrix as Part A. Path 1 = sort-merge-with-detection (provider-direct + explicit detection on the sort-merge side). Path 2 = direct `provider.nested_loop_join_v2_inner_u32_1key`. Side-by-side timing per cell. Per F-W43-2's anticipated outcome: if nested-loop wins on overlap, iteration-N+ amends D2.

Output: `docs/evidence/2026-05-10-w43-production-bench/README.md` with median timings + speedup table for Part A + Part B + decision-validation conclusion.

**Iteration-6 outcome (per F-W43-14 — counter-finding rather than acceptance success)**: Step 12 bench surfaced the counter-finding F-W43-2 was designed to catch:
* **Part A — D7 #8 (≥ 2× vs hash) FAILED on every cell**: speedups range 1.10×–1.80×, never reaching 2× across the 50×50–2000×2000 sorted-eligible matrix.
* **Part B — D2 precedence (sort-merge > nested-loop) FAILED on every cell**: nested-loop wins 1.25×–2.46× on every overlap cell.

Both rejections are documented in `docs/evidence/2026-05-10-w43-production-bench/README.md`. Step 12 in iteration 6 is the **bench evidence + rejection record**, not an acceptance success. Per F-W43-2's locked amendment path, iteration-6 (this iteration) executes the W4.3 unwiring + cert rewrite that the counter-finding mandates.

Commit subjects: `feat(w43): add production sort-merge bench + evidence (counter-finding: D2 precedence + D7 #8 both fail)` (operationally landed at `ab7021d4`).

### Step 13 — Closure proposal (text-only, scope amended per F-W43-14)

**Iteration 1–5 historical**: Plan-iteration commit + Steps 2–12 commits on `feat/w43-sort-merge-join`. Acceptance was full production sort-merge dispatch closure.

**Iteration 6 (per F-W43-14)**: closure proposal scope changes to **operator-only**:
* Plan-iteration commit + Steps 2–3 commits (operator + provider implementation) + iteration-6 unwiring commit + iteration-6 cert-rewrite commit + Step 12 bench/evidence commit + iteration-6 closure-proposal commit (this final commit) on `feat/w43-sort-merge-join`.
* The W4.3 sort-merge operator is **implemented**, **bench-validated** (vs hash 1.10×–1.80× win), **operator-cert-tested** (4 provider-level certs: A/E/F/G after F-W43-14 rewrite), but **not wired into production dispatch** (D2 amended; bench rejected).
* No board edit. No FF-merge. No DONE marking until the user explicitly approves operator-only scope as a valid closure for the W4.3 board item.
* The closure proposal explicitly raises the scope question: "W4.3 operator implemented but production dispatch rejected by evidence — does the closure board accept this as DONE, or does it require a different completion criterion (e.g., operator removed entirely; or dispatch-perf investigation deferred to v0.6.6+)?"

## Acceptance Grid (iteration-6 canonical)

| Cell | Count | Test file | Acceptance criterion |
|------|-------|-----------|----------------------|
| **Cert A (operator) — sorted-key parity** | 1 | `test_w43_sort_merge_dispatch.rs` | `provider.sort_merge_join_v2_inner_u32_1key` row-set parity vs hash on sorted 100-row 1-key U32 fixture |
| **Cert E (operator) — Symbol-typed parity** | 1 | same | `provider.sort_merge_join_v2_inner_u32_1key` row-set parity vs hash on Symbol-typed buffers |
| **Cert F (operator) — duplicate-key parity** | 1 | same | `provider.sort_merge_join_v2_inner_u32_1key` on 250 keys × 4 dups → 4000 output rows, all (k, lp, rp) tuples distinct, parity vs hash |
| **Cert G (operator) — empty-input parity (per F-W43-4)** | 1 | same | two subcases: `is_sorted_ascending_u32` returns Ok(true) on `n == 0` (n<2 short-circuit); `provider.sort_merge_join_v2_inner_u32_1key` produces empty buffer with no crash; row-set parity vs hash empty fast path |
| **Step 12 bench — counter-finding evidence** | 1 | `w43_production_sort_merge_bench.rs` + `docs/evidence/2026-05-10-w43-production-bench/README.md` | Bench runs to completion; documents D7 #8 + D2 rejection per F-W43-14. NOT a ≥2×-win acceptance criterion. |
| **W4.2 dispatch certs (production-routing guard)** | 5 (already-existing) | `test_w42_nested_loop_dispatch.rs` | All 5 W4.2 certs PASS unchanged (Inner + Symbol + multi-col + Semi + above-threshold) — the production-routing guard for the post-unwiring executor |
| **Workspace pass-count delta (iteration-6)** | **+4 W4.3 operator certs (A, E, F, G after rewrite)**, **0 W4.2 cert delta** (existing W4.2 suite unchanged), **−4 dispatch-only certs retired (B, C, D, D')** | — | Iteration 1–5 originally targeted +8 dispatch certs; iteration 6 retires B/C/D/D' as superseded and rewrites A/E/F/G as operator certs. Net W4.3 pass-count delta: +4. |
| **Iteration-1 Cert B/C/D/D' (RETIRED per F-W43-14)** | 0 | — | Removed in iteration-6 cert-rewrite commit. Replaced by W4.2 suite which already covers fall-through fixture shapes. |
| **Iteration-1 Bench Part A "≥ 2× vs hash" (REJECTED per F-W43-14)** | 0 | — | The ≥2× threshold was the iteration-1 working hypothesis. Bench measured 1.10×–1.80×; D7 #8 fails. The bench evidence is preserved as the rejection record. |
| **Iteration-1 Bench Part B "D2 precedence holds" (REJECTED per F-W43-14)** | 0 | — | Bench measured nested-loop wins 1.25×–2.46× on every overlap cell; D2 amended to "no W4.3 production dispatch". |

## Source-of-Truth References (iteration-6 canonical)

* Spike evidence: `docs/evidence/2026-05-10-w43-bench-spike/README.md` (on `bench-spike/w43-sort-merge`); `fadc2700` HEAD.
* Recon: `docs/plans/2026-05-08-w43-sort-merge-join-recon.md`.
* W4.2 cert template: `crates/xlog-integration/tests/test_w42_nested_loop_dispatch.rs`.
* W4.2 production reference (provider + dispatch): `crates/xlog-cuda/src/provider/relational.rs::nested_loop_join_v2_inner_u32_1key` + `crates/xlog-runtime/src/executor/node_dispatch.rs::execute_join` (W4.3 mirrors this structure).
* Sortedness-check kernel precedent: `crates/xlog-cuda/kernels/wcoj.cu::wcoj_layout_check_sorted_unique_u32` + `provider/wcoj.rs:3137`.
* Existing `JoinStrategy::SortMerge` dead-code: `crates/xlog-runtime/src/statistics.rs:15` (untouched).

## Risk Register (informational, iteration-6 canonical)

| Risk | Mitigation |
|------|------------|
| Detection kernel cost erodes the speedup | **REALIZED per F-W43-14**: detection cost contributes to sort-merge-with-detection's failure to reach the iteration-1 D7 #8 ≥ 2× threshold vs hash. Step 12 bench measured 1.10×–1.80× wins, never reaching 2×. Combined with the multi-col gather risk below, the iteration-1 `~5-50 µs detection vs ~300 µs win` arithmetic was empirically wrong: production-arity work dominates the win budget. Iteration-6 response: no production dispatch, so the detection cost is paid only by certs + bench (already accepted overhead). |
| Sort-merge wins in spike but loses in production due to multi-col gather overhead | **REALIZED per F-W43-14**: Step 12 bench at 3-col production arity measured sort-merge times consistently above the spike's 1-col-no-payload measurements; the multi-col `gather_buffer_by_indices` materialization erodes the spike's measured kernel-only advantage. Iteration-1 mitigation reasoning ("Spike's 2-col duplicate-key cell already exercises gather (2.56× win)") was insufficient — 2-col is not 3-col, and the duplicate-key spike cell tests run-length scaling rather than payload-width scaling. Iteration-6 response: no production dispatch; the operator's 1-col-and-payload kernel-level competence remains correct (Cert F at 250 keys × 4 dups validates run-length output) but the ≥2× threshold was the wrong gate for production-arity production traffic. |
| Threshold mismatch between W4.2 and W4.3 | Iteration 1–5 mitigation: D3 explicitly shared the `NESTED_LOOP_TOTAL_THRESHOLD = 4_000_000` constant; iteration-1 lock prevented drift. **Iteration-6 outcome (per F-W43-14)**: with no W4.3 dispatch site, the threshold-sharing concern is moot for W4.3 — W4.2 still uses the constant unchanged. The risk is closed by removing the dispatch site, not by the constant-sharing decision. |
| Sort-merge dispatch overrides nested-loop in cases where nested-loop is faster | **REALIZED per F-W43-14**: Step 12 bench surfaced the counter-finding (nested-loop wins 1.25×–2.46× on every overlap cell). Iteration-6 unwiring removes W4.3 from production dispatch entirely; nested-loop remains the production path for the shared eligibility envelope. |
| Detection kernel reports "unsorted" on edge cases (single-row inputs, empty inputs) | Per F-W43-4: empty AND single-row inputs (`n < 2`) are short-circuited by `is_sorted_ascending_u32`'s **own internal fast path** (return `Ok(true)` BEFORE allocation/launch) — they enter detection AFTER the threshold check admits the join (`0 * 0 = 0 ≤ 4M = true`). The detection kernel never launches with grid_dim 0. Once detection returns `Ok(true)` for an empty side, `sort_merge_join_v2_inner_u32_1key`'s own empty fast path returns the empty `combine_schemas` buffer (mirrors `hash_join_inner_v2`'s empty handling at `relational.rs:3165-3170`). Cert G covers both empty-left and empty-right fixtures. |
| `JoinStrategy::SortMerge` dead-enum confusion | NOT touched per D8. Future cleanup commit (out of W4.3) can delete the enum entirely. |

## Plan-Approval Gate (iteration 6)

This plan is **iteration 6** (iteration 5 was approved through Step 11; iteration 5's Step 12 bench surfaced the F-W43-2 anticipated counter-finding; iteration 6 amends the W4.3 closure scope to "operator implemented, production dispatch rejected by evidence" per user direction in response to the bench rejection). Iterations 1-4 follow the pre-execution review pattern; iteration 5 captured execution-discovered drift (F-W43-11/12/13); iteration 6 is the post-bench scope-redirect amendment (F-W43-14). The live D-table + Step plan + Acceptance Grid above remain the canonical source of truth.

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

Iteration 5 captures three findings surfaced during execution rather than during plan review: F-W43-11 (retroactively logged from Step 5), F-W43-12 (workspace-test gate exception surfaced during Step 11 verification), and F-W43-13 (cert-contract drift surfaced during iteration-5 plan review itself — the executed certs use exact-equality `== 1` / `== 0` discipline from Step 6 + Step 10 patches, but D7 + Acceptance Grid still carried the soft `>= 1` / "either path" loophole text from iteration 4). All three findings landed operationally before this amendment commit; iteration 5 brings the plan record into alignment with the executed work. Header iteration tag bumped 4 → 5.

| ID | Severity | Finding | Pre-iter-5 (wrong / silent) | Iter-5 (corrected) |
|----|----------|---------|-----------------------------|--------------------|
| **F-W43-11** | Major | D2 precedence (sort-merge > nested-loop) silently broke W4.2 Cert A and Cert E: their fixtures used trivial `(0..N).map(|i| (i, ...))` sorted-ascending keys, so after Step 5's W4.3 dispatch landed the sort-merge path took precedence and `nested_loop_dispatch_count == 1` failed. The W4.2 certs were no longer regression-detecting for the nested-loop path. | Step 5 wiring did not document a fixture-de-overlap requirement; W4.2 cert files used sorted ascending keys; no explicit guard in the plan. | Step 5 + W4.2 fixture de-overlap bundled into one commit (4ef14855). Cert A and Cert E now use rotate-halves `(N/2..N).chain(0..N/2)` — minimum-violation unsorted shape, deterministic, same key set / same row-set / same match counts as before. Bundled commit message documents both halves. Future plan-discipline improvement: dispatch-precedence changes that admit a new strategy MUST audit existing-strategy positive certs for shape collision before commit. |
| **F-W43-12** | Major | Step 11's workspace-test gate (`cargo test --workspace --release --exclude pyxlog`) does not exit 0 because `crates/xlog-cuda/tests/test_wcoj_layout_fast_path.rs` fails non-deterministically — even under `--test-threads=1`. Confirmed pre-existing: the same flake reproduces on merge-base 19f7bc5d (W4.2 closure HEAD) when running the test file alongside the rest of the workspace. Failure signature (last-row-only data corruption + u64 high-bits leaking into u32 reads) is consistent with missing stream-synchronize between kernel launch and D2H download in the v0.6.2 WCOJ layout fast-path code. | Step 11 implied the canonical workspace command must exit 0; the flake was documented in the Step 11 commit message but the gate criterion was not relaxed. | Step 11 amended in this iteration to record an explicit gate exception: every workspace-test path EXCEPT `test_wcoj_layout_fast_path` exits 0, and the cert suite (the authoritative gate per MEMORY.md) passes. Fixing the flake is **out-of-W4.3-scope** because (a) the failures are in v0.6.2 WCOJ code unrelated to W4.3, and (b) the merge-base reproduces the flake. Deferred to follow-up work on the v0.6.2 fast-path code. The Step 11 commit message references this F-W43-12 gate exception. |
| **F-W43-13** | Major | Cert-contract drift between the executed certs and the plan's canonical D7 + Acceptance Grid. Step 6 patch (commit c665bd0e) tightened Cert A's `sort_merge_dispatch_count >= 1` to `== 1` and that exact-equality discipline was extended to Certs B/E/F as the certs landed; Step 10 patch (commit 6f25377d) tightened Cert G to assert `sort_merge_dispatch_count == 1` AND `nested_loop_dispatch_count == 0` per fresh-executor subcase (closing the parity-only loophole). The plan still carried the soft `>= 1` criteria in D7 #1, #2, #6, #7 and the Acceptance Grid Certs A/B/E/F, plus the "either path" / "no kernel-launch crash; empty output via either path" loophole text in D7 #7' and Cert G. Certs are correct; canonical contract was stale. | D7 #1: `sort_merge_dispatch_count >= 1`; D7 #2: `nested_loop_dispatch_count >= 1 or hash`; D7 #7' Cert G: "assert `sort_merge_dispatch_count` reflects the chosen short-circuit (either dispatched OR not-dispatched)"; Acceptance Grid Certs A/B/E/F: `>= 1` + grid Cert G: "no kernel-launch crash; empty output via either path; row-set parity". | D7 #1: `sort_merge_dispatch_count == 1` + `nested_loop == 0`. D7 #2: `sort_merge == 0`, `nested_loop == 1`. D7 #5 + #6 (D' + E): both add explicit `nested_loop == 0`. D7 #7 (F): `sort_merge == 1`, `nested_loop == 0`, output count == 4000, all 4000 tuples distinct. D7 #7' (G): on each fresh-executor subcase `sort_merge == 1` AND `nested_loop == 0` (proves the F-W43-4 contract end-to-end: detection short-circuits n<2→Ok(true) + sortedness probe on populated side + dispatch admits + kernel empty fast path). Acceptance Grid synced to match. Same lesson as F-W43-7/8/9/10: contract changes are file-wide concerns; future plan-discipline improvement compounding the iteration-bumping lesson — when *any* contract change lands operationally, grep the canonical D-table grid AND the Acceptance Grid AND any prose Step section before declaring the patch complete. |

**Iteration-5 process observation**: F-W43-11, F-W43-12, and F-W43-13 are all *execution-discovered* findings that could not have been surfaced by plan review alone — F-W43-11 required the W4.3 dispatch wiring to exist in source; F-W43-12 required running the workspace gate against the live CUDA environment; F-W43-13 required the cert review tightenings (Step 6 patch + Step 10 patch) to land before the drift between executed certs and canonical plan became visible. The pattern teaches: post-execution amendment commits are part of the plan-iteration discipline, not a violation of it; the discipline says "every amendment lands as a new iteration commit," and that includes amendments motivated by reality after the fact. F-W43-13 specifically extends the F-W43-7/8/9/10 file-wide-concern lesson: when a contract is tightened (not just labeled), the same grep-everywhere discipline applies — D7 grid + Acceptance Grid + prose Steps must all be inspected before the iteration is declared closed.

## Iteration-6 Amendment Log

Iteration 6 captures one execution-discovered finding (F-W43-14) that materially changes the W4.3 closure scope from "production dispatch closure" to "operator implemented, production dispatch rejected by evidence." This amendment was anticipated by F-W43-2's iteration-1 instruction: *"If the bench shows nested-loop wins on the overlap, iteration-N+ amends D2."* Iteration 6 IS that amendment.

| ID | Severity | Finding | Pre-iter-6 (working hypothesis, now rejected) | Iter-6 (amended) |
|----|----------|---------|------------------------------------------------|-------------------|
| **F-W43-14** | Major | Step 12 production bench (commit `ab7021d4`) surfaced two simultaneous failures of iteration-1 design hypotheses. **Part A**: D7 #8 "≥ 2× vs hash" REJECTED — measured speedups 1.10×–1.80× on every cell of the 50×50–2000×2000 sorted-eligible matrix. **Part B**: D2 precedence "sort-merge > nested-loop" REJECTED — nested-loop wins 1.25×–2.46× on every overlap cell. F-W43-2 anticipated this exact outcome and locked in the amendment path. Per user direction: keep operator + provider + kernels + manifest + bench evidence; remove executor dispatch wiring + counter + accessor + eligibility predicate; rewrite operator-meaningful certs (A/E/F/G) at the provider layer; retire dispatch-only certs (B/C/D/D') as superseded; W4.2 cert suite remains the production-routing guard for the post-unwiring executor; closure scope changes to operator-only. | D2: sort-merge > nested-loop > hash precedence in `execute_join`. D6: `sort_merge_dispatch_count` field + accessor on `Executor`. D7 #1–#7' + Acceptance Grid: 8 dispatch certs at executor level, all asserting counter values. D7 #8: ≥ 2× vs hash bench acceptance. Step 4 added `eligible_for_sort_merge` predicate. Step 5 wired W4.3 branch in `execute_join` ahead of W4.2. Steps 6–10 cert file with 8 dispatch certs. Step 13 closure proposal: full production sort-merge dispatch closure. | D2 amended: NO production dispatch; W4.3 sort-merge implemented at provider layer only. D6: counter field/accessor REMOVED. D7 amended: 4 operator-level provider parity certs (A/E/F/G after rewrite); W4.2 cert suite kept as production-routing guard; bench evidence is the rejection record (NOT a ≥2× acceptance). Step 4 + Step 5 + (parts of) Step 9–10: superseded; iteration-6 unwiring step replaces them. Steps 6–10 cert file: A/E/F/G rewritten as provider parity certs; B/C/D/D' retired (superseded by W4.2 suite which already covers fall-through fixture shapes). Step 13 closure proposal: operator-only scope, raises closure-board scope question. The W4.3 sort-merge operator + provider + kernels + bench evidence remain as graduated implementation work. |

**Iteration-6 process observation**: F-W43-14 is unique among the iteration-5/6 findings in that it materially shrinks the closure scope rather than tightening or relaxing existing criteria. The iteration-1 plan correctly anticipated this possibility (D2 marked PROVISIONAL, F-W43-2 locked the amendment path); the iteration-1 review explicitly built in the path that iteration 6 is now executing. The discipline lesson: when a design lock is marked PROVISIONAL with an explicit "if X, then amend" instruction, the iteration-N+ amendment commit IS the closure of that conditional, not a deviation from the original plan. This is also a productive use of bench-spike-first discipline (`feedback_perf_bench_spike_first.md`) compounded with provisional-precedence: the spike validated the operator works; the production bench validated the precedence is wrong. Both kinds of evidence are load-bearing.

## Iteration-6 Replacement Steps (per F-W43-14)

These steps replace the superseded portions of Steps 4–10. They are executed in order on top of the iteration-5 commit history (no rebase, no force-push — additive commits only).

### Step 4' — Executor unwiring (per F-W43-14 D2 + D6 amendment)

File: `crates/xlog-runtime/src/executor/node_dispatch.rs` (edit) + `crates/xlog-runtime/src/executor/mod.rs` (edit) + `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` (edit).

Removals:
* `eligible_for_sort_merge` private free fn at `node_dispatch.rs`.
* The `if eligible_for_sort_merge { ... checked_mul ... is_sorted_ascending_u32 × 2 ... sort_merge_join_v2_inner_u32_1key ... }` block in `execute_join`.
* The `if out.is_none()` wrap on the W4.2 nested-loop branch (no longer needed; W4.3 cannot consume the slot).
* `sort_merge_dispatch_count: u64` field on `Executor` at `executor/mod.rs`.
* The constructor's `sort_merge_dispatch_count: 0` initializer.
* The `sort_merge_dispatch_count(&self) -> u64` accessor at `wcoj_dispatch.rs` (or wherever it lives).

Preserves:
* `crates/xlog-cuda/src/provider/relational.rs::is_sorted_ascending_u32`.
* `crates/xlog-cuda/src/provider/relational.rs::sort_merge_join_v2_inner_u32_1key`.
* `crates/xlog-cuda/kernels/sort.cu::check_ascending_sorted_u32`.
* `crates/xlog-cuda/kernels/join.cu::sort_merge_join_inner_u32_1key_pairs`.
* `crates/xlog-cuda/src/kernel_manifest_data.rs` entries for both kernels.
* `crates/xlog-cuda/src/provider/mod.rs` kernel-name constants `SORT_MERGE_JOIN_INNER_U32_1KEY_PAIRS` + `CHECK_ASCENDING_SORTED_U32`.
* `node_dispatch.rs:458` adaptive-indexing comment update from Step 11 patch (now states "Only runs if W4.2 nested-loop didn't dispatch" — drop the W4.3 reference).

Commit subject: `refactor(w43): remove executor sort-merge dispatch + counter + predicate (per F-W43-14)`.

### Step 5' — Cert rewrite (per F-W43-14 D7 amendment)

File: `crates/xlog-integration/tests/test_w43_sort_merge_dispatch.rs` (rewrite).

Retire (delete from file): Cert B (`unsorted_eligible_falls_back_to_nested_loop`), Cert C (`above_threshold_sorted_falls_back_to_hash`), Cert D (`multi_col_key_falls_back_to_hash`), Cert D' (`semi_join_falls_back_to_hash`). The W4.2 cert suite already covers these fall-through fixture shapes for the production-routing guard.

Rewrite at the operator/provider layer:
* **Cert A (operator)**: `provider.sort_merge_join_v2_inner_u32_1key` on sorted 100-row 1-key U32 fixture; `BTreeSet<[u32; 4]>` parity vs `provider.hash_join_v2 Inner`. No executor, no dispatch counter, no selectivity feedback. Pure operator parity.
* **Cert E (operator)**: same shape on Symbol-typed buffers (uses `upload_symbol_keyed`); parity vs hash.
* **Cert F (operator)**: 250 keys × 4 dups → `provider.sort_merge_join_v2_inner_u32_1key` produces 4000 output rows, all (k, lp, rp) tuples distinct, parity vs hash.
* **Cert G (operator, per F-W43-4 layered short-circuit)**: two subcases (`num_left == 0`, `num_right == 0`). Each asserts `provider.is_sorted_ascending_u32` returns `Ok(true)` on the `n < 2` short-circuit; `provider.sort_merge_join_v2_inner_u32_1key` produces empty buffer; row-set parity vs hash empty fast path.

Helpers (`make_runtime_backed_fixture`, `upload_binary_u32`, `upload_symbol_keyed`, `download_quads`) remain. `RuntimeBackedFixture` retains `provider` + `memory` only — the executor-related fields can stay or be trimmed (decision: keep, harmless). `download_pairs` + `download_triples` + `build_executor_with_two_relations` can be deleted if unused after rewrite.

The Executor + RuntimeConfig + Compiler imports can be removed if no remaining test uses them.

Commit subject: `test(w43): rewrite W4.3 certs as operator-level provider parity (per F-W43-14); retire dispatch-only certs B/C/D/D'`.

### Step 6' — Optional W4.2 fixture revert (per F-W43-14)

File: `crates/xlog-integration/tests/test_w42_nested_loop_dispatch.rs` (edit, OPTIONAL).

Iteration-5 / Step-5 commit `4ef14855` introduced rotate-halves fixtures (`(50..100u32).chain(0..50u32)`) on W4.2 Cert A and Cert E to keep them regression-detecting under W4.3 dispatch precedence (F-W43-11). With W4.3 dispatch removed in iteration 6, the rotate-halves is no longer required — the fixtures could revert to `(0..100u32)` sorted ascending.

**Decision**: keep the rotate-halves intact. Reverting introduces churn for no behavioral gain (the certs still pass with rotate-halves and the original sorted-ascending shape; rotate-halves doesn't break anything). The F-W43-11 amendment record stays accurate as a historical justification. If a future reader is confused about why W4.2 fixtures use rotate-halves, the inline comment + F-W43-11 amendment log entry explain the original motivation; the iteration-6 amendment log records why the motivation no longer applies. This is the lower-risk choice.

This step is **optional** and may be skipped if the iteration-6 unwiring + cert rewrite verification gates pass without it. Iteration-6 proceeds without the revert by default.

### Step 7' — Final verification gate (per F-W43-12 exception)

After Steps 4' + 5' (+ optional 6'):
* `cargo fmt --check --all` exits 0.
* `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` exits 0.
* `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1 (authoritative gate per MEMORY.md).
* `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` exits 0 for every test path EXCEPT `test_wcoj_layout_fast_path` (per F-W43-12 exception).
* `cargo test -p xlog-integration --release --test test_w43_sort_merge_dispatch` → 4 tests pass (A, E, F, G after rewrite; B, C, D, D' retired).
* `cargo test -p xlog-integration --release --test test_w42_nested_loop_dispatch` → 5 tests pass (W4.2 suite unchanged).
* `cargo bench -p xlog-integration --bench w43_production_sort_merge_bench --no-run` → 0 errors (bench remains valid even though its result is now a rejection record).

### Step 8' — Iteration-6 closure proposal (per F-W43-14 D7 closure-scope amendment)

Closure proposal commit (text-only commit message + this plan iteration commit; no board edit, no DONE marking, no FF-merge). Explicitly raises the scope question to the user / closure board:

> W4.3 sort-merge join operator: implemented, bench-validated (vs hash 1.10×–1.80× win), provider-cert-tested (4 operator-level parity certs). Production dispatch rejected by Step 12 bench evidence (D2 precedence + D7 #8 both fail). Per F-W43-2's anticipated amendment path, iteration-6 removes the dispatch wiring + counter + predicate while preserving the operator surface as graduated implementation work.
>
> **Closure-board question**: does the v0.6.5 closure board accept "operator implemented, production dispatch rejected by evidence" as a valid completion of the W4.3 board item, or does the board require a different completion criterion?
>
> Possible board responses:
> 1. **Accept as DONE**: W4.3 closure board mark switches OPEN → DONE; tally DONE 9→10, OPEN 10→9. Close W4.3.
> 2. **Reject; require operator removal entirely**: revert the operator + provider + kernel work; mark W4.3 ABANDONED.
> 3. **Defer**: keep W4.3 OPEN; reopen in v0.6.6 with kernel-perf investigation as the new scope.

Commit subject: `docs(w43): iteration-6 closure proposal — operator-only scope (production dispatch rejected per F-W43-14)`.
