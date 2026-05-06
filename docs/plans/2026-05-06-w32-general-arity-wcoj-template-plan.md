# W3.2 — General-Arity WCOJ Kernel Template (k = 5 and k = 6)

**Closes W3.2 only.** No skew-classifier work for k=5/k=6.
No new env knobs. No `force` / `kill` / `adaptive` toggles. No
W3.3–W3.6 work. No CUDA `.cu` changes for triangle (k=3) or
4-cycle (k=4) — those are slice-1 / slice-2 territory and stay
bit-identical. No `v0.6.6` references; no punt-to-later wording
— out-of-scope concerns are owned by W3.3+ board items, named
at the point of reference. No push, no tag. Plan-first; no
implementation until iteration is approved by the user.

**Plan iteration:** 4 (strengthen Step 11 source-audit per user iteration-3 review).
**Base:** `main` at `d5073bdb` (W3.1 closure commit).
**Worktree:** `.worktrees/w32-general-arity-wcoj-template`.
**Branch:** `feat/w32-general-arity-wcoj-template`.
**Closure board:** `docs/v065-closure-board.md:86` (W3.2 row, OPEN).

## Acceptance line (locked from board)

> xlog-cuda cert: 5-clique fixture matches CPU oracle AND
> 6-clique fixture matches CPU oracle; **k=6 cert MUST pass
> without adding any new `.cu` source for k=6** (template
> instantiation only). Counter increments on dispatch;
> binary-join fallback row-set parity at both k.

W3.2 is a **runtime-integrated kernel milestone**, not
provider-only. The "counter increments on dispatch" and
"binary-join fallback row-set parity" claims cannot be
satisfied at the xlog-cuda layer alone — promoter + dispatcher
wiring is required.

## Direction (locked with user, iteration 0)

| # | Decision | Locked answer |
|---|----------|---------------|
| D1 | Kernel template strategy | **C++/CUDA template `<int K>`.** k=5 and k=6 share one count + one materialize implementation. k=6 may have ABI wrapper / explicit instantiation lines only — no hand-written algorithm body. |
| D2 | CPU oracle | Test-only `cpu_clique_reference<T, const K: usize>(edges: &[Vec<(T, T)>]) -> Vec<[T; K]>` brute-force oracle, generic over the cell type `T` (`T: Copy + Ord + Eq + Hash`). The `edges` parameter is a runtime slice (NOT a const-array `&[...; K_CHOOSE_2]`, which is not stable-Rust implementable as a const-generic length expression). The oracle's first line is `assert_eq!(edges.len(), K * (K - 1) / 2, "clique-K oracle requires C(K,2) edge lists");` — fail-fast on caller misuse. Concretely usable at `T = u32` (covers U32 + Symbol on 4-byte path; Symbol IDs are u32 bits per the W3.1 convention) and `T = u64` (covers U64). For ergonomic test-side call-sites, two thin concrete wrappers can sit alongside: `pub fn cpu_clique5_reference<T>(edges: &[Vec<(T, T)>]) -> Vec<[T; 5]>` and `pub fn cpu_clique6_reference<T>(edges: &[Vec<(T, T)>]) -> Vec<[T; 6]>` — purely test-side sugar over the generic core, no behavioral difference. No production callers. |
| D3 | Width-class coverage | **u32, u64, AND Symbol** — Symbol parity gets its own cert at both k, not implicit "inherits via 4-byte equivalence". |
| D4 | Runtime dispatcher integration | **Required.** New `Executor::wcoj_clique5_dispatch_count` / `_clique6_dispatch_count` counters. Promoter recognizes k=5 / k=6 clique body shapes. Default-dispatch on shape match; silent fallback on shape mismatch / kernel error. **No** force-on / kill-switch / adaptive-classifier knobs — those are out of scope, owned by W3.3+ board items. |
| D5 | Promoter shape match | Flatten the join tree, collect positive Scans, validate the complete-graph edges (`C(k, 2)` atoms each on a unique unordered variable pair from the k head vars), THEN emit canonical `MultiWayJoin`. Robust to left-deep / right-deep / bushy lowered trees — no repeat of the W2.6 right-deep promoter gap. |
| D6 | Canonical edge order | Lex on `(i, j)` for `i < j`. k=5 = `[(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]`. k=6 mirrors, 15 entries. Slot vars and provider input order match this table exactly. |
| D7 | Test grid | Provider certs k=5/k=6 × u32/u64/Symbol; runtime dispatch certs both k (counter advance + binary-join row-set parity); promoter positive + negative shape certs; k=6 template-source cert (grep proves no hand-written k=6 body); negative "k=7 unsupported" test only if the API exposes that path (k=7 receives no closure credit; W3.2 is k=5/k=6). |
| D8 | Scope discipline | Branch from main `d5073bdb`. Plan as branch commit #1 of `feat/w32-general-arity-wcoj-template`. No skew classifier; no env knobs; no W3.3–W3.6 work; no push; no tag; no self-mark DONE. |

## Code-level surface (read-only audit, no edits in this plan)

**Existing kernels** (unchanged by W3.2):
* `crates/xlog-cuda/kernels/wcoj.cu:240` — `wcoj_triangle_count` (u32).
* `crates/xlog-cuda/kernels/wcoj.cu:309` — `wcoj_triangle_materialize` (u32).
* `crates/xlog-cuda/kernels/wcoj.cu:404` — `wcoj_4cycle_count` (u32).
* `crates/xlog-cuda/kernels/wcoj.cu:446` — `wcoj_4cycle_materialize` (u32).
* `_u64` mirrors for triangle (L504) and 4-cycle (L604).
* Shared device helpers: `lower_bound_u32` / `_u64`, `upper_bound_u32` / `_u64`, `intersect_count` / `_u64`, `contains_pair_u32` / `_u64`, `intersect_emit_xyz` / `_u64`.
* `wcoj_compute_total` (shape-agnostic).

**Existing provider entries** (unchanged):
* `wcoj_triangle_u32_recorded` / `_u64_recorded`.
* `wcoj_4cycle_u32_recorded` / `_u64_recorded`.

**Existing runtime call sites** (unchanged):
* `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:1111, 1223` (triangle).
* `crates/xlog-runtime/src/executor/wcoj_dispatch.rs:1503, 1577` (4-cycle).

**What W3.2 adds**:

1. **CUDA template** (single block in `crates/xlog-cuda/kernels/wcoj.cu`, appended after the slice-2 4-cycle u64 kernels):
   ```cuda
   template <int K>
   __device__ __forceinline__ uint32_t wcoj_clique_per_thread_count_u32(
       const uint32_t* __restrict__ const cols[2 * (K * (K - 1) / 2)],
       const uint32_t n_edges_arr[K * (K - 1) / 2],
       uint32_t leader_row_idx);

   template <int K>
   __device__ __forceinline__ void wcoj_clique_per_thread_emit_u32(
       /* same args + output ptrs[K] */);

   // ABI wrappers for k=5, k=6 — extern "C" only, NO algorithm
   // body beyond the template call. Mirrored at u64.
   extern "C" __global__ void wcoj_clique5_count_u32(...) {
       wcoj_clique_template_count_u32<5>(...);
   }
   extern "C" __global__ void wcoj_clique5_materialize_u32(...) { ... }
   extern "C" __global__ void wcoj_clique6_count_u32(...) {
       wcoj_clique_template_count_u32<6>(...);  // SAME template
   }
   extern "C" __global__ void wcoj_clique6_materialize_u32(...) { ... }
   // u64 mirrors:
   extern "C" __global__ void wcoj_clique5_count_u64(...) { ... }
   extern "C" __global__ void wcoj_clique5_materialize_u64(...) { ... }
   extern "C" __global__ void wcoj_clique6_count_u64(...) { ... }
   extern "C" __global__ void wcoj_clique6_materialize_u64(...) { ... }
   ```
   The 8 ABI wrappers are each ≤ 5 lines (call into the template). The k=6 entries' bodies must be **byte-identical in algorithm to k=5's** — only the `K` parameter differs.

2. **Provider entries** (`crates/xlog-cuda/src/provider/wcoj.rs`):
   * `wcoj_clique5_u32_recorded(edges: &[&CudaBuffer; 10], launch_stream)` → `CudaBuffer` (5-column output).
   * `wcoj_clique5_u64_recorded` (same shape).
   * `wcoj_clique6_u32_recorded(edges: &[&CudaBuffer; 15], launch_stream)` → `CudaBuffer` (6-column output).
   * `wcoj_clique6_u64_recorded`.
   * Each validates: runtime-backed, edge count = `C(k, 2)`, every edge is 2-column with the entry's width-class. Composes count + scan + materialize per the existing slice-1 pattern.
   * Symbol parity: each entry accepts U32 OR Symbol per column for the u32 width-class (matches `wcoj_triangle_u32_recorded` contract).
   * **Layout step**: provider entries themselves do NOT call layout-sort. The runtime dispatcher (step 7) is responsible for routing every edge through W3.1's `wcoj_layout_sort_*_recorded` BEFORE invoking the provider entry; provider entries assume sorted + deduped input as a pre-condition (same contract as `wcoj_triangle_u32_recorded`). Provider doc-comments state this contract explicitly. Per fix #7, no provider-side "if already sorted + unique" branch — the dispatcher always layouts; the provider always assumes laid-out input.

3. **Promoter** (`crates/xlog-logic/src/promote.rs`):
   * New `try_promote_clique_k(node, k, ...) -> Option<RirNode>` covering k ∈ {5, 6}.
   * **Tree-flattening pass first**: walk the body, collect every leaf `Scan`, accept the canonical outermost `Project`-of-`Join` shape, ignore Join shape underneath (left-deep / right-deep / bushy all OK). **Filter / comparison wrappers reject — they are NOT silently ignored.** A clique body with an interior `RirNode::Filter` would risk semantic loss if the filter were stripped during promotion (the WCOJ kernel evaluates the join body without re-applying the filter), so W3.2 rejects this shape to preserve semantics; the body falls through to the binary-join path. **Rejected in W3.2 to preserve semantics — no closure credit.**
   * Validate: exactly `C(k, 2)` Scans; the head has exactly k variables; each Scan binds exactly 2 distinct head variables (no constants, no shared vars within a single atom — i.e. no `e(X, X)` self-edge); the multiset of bound variable pairs equals `{(i, j) | 0 ≤ i < j < k}` — i.e. every edge of the complete graph K_k appears exactly once.
   * **Argument-permutation handling.** Without column-swap layout for clique edges (which is out of scope for W3.2), atoms must use the canonical column order: for the bound variable pair `(v_i, v_j)` with `i < j`, the atom must be `e(v_i, v_j)` not `e(v_j, v_i)`. A reversed atom like `e(Y, X)` where the canonical form would be `e(X, Y)` REJECTS — silently accepting it would mis-align the kernel's per-edge layout with the promoter's slot table. Promoter detects reversed atoms during the variable-pair extraction step and declines.
   * On match, reorder the inputs to the canonical lex `(i, j)` order from D6, build canonical `MultiWayJoin.inputs` + `slot_vars`, set fallback to original body.
   * Same `recursive_scan_count <= 1` gate as slice-4 promoter — and additionally, if `recursive_scan_count >= 1` (linear-recursive clique body), the W3.2 promoter **declines** rather than emitting a MultiWayJoin. This is explicitly resolved per user iteration-1 fix #8: W3.2 does not extend the recursive WCOJ helper to body-keyed clique dispatch; recursive clique bodies fall through to binary-join. A negative cert in step 10 pins this contract (`linear_recursive_clique5_does_not_promote`).

4. **Runtime dispatcher** (`crates/xlog-runtime/src/executor/wcoj_dispatch.rs`):
   * New `Executor::wcoj_clique5_dispatch_count: u64` and `_clique6_dispatch_count: u64`.
   * New `try_dispatch_wcoj_clique5` / `_clique6` matched on the new MultiWayJoin shapes; calls the new provider entries on the layout-sorted inputs (reusing W3.1's `wcoj_layout_sort_u32_recorded` / `_u64_recorded` for the k≥5 layout step — the slice-1 arity-2 fast-path entries don't apply since clique inputs are still 2-column per edge, but each cert checks both layout helpers route correctly).
   * Default dispatch on shape match (no force / kill / adaptive knobs in W3.2). Silent fallback to the binary-join body on mismatch / kernel error — preserves row-set parity guarantee.

## Step plan (14 steps)

### Step 1 — Audit (read-only)

Confirm via `grep` / `Read`:

* `wcoj.cu` triangle/4-cycle kernels are the only WCOJ kernels currently linked.
* `xlog-cuda/src/kernel_manifest_data.rs` registers each extern by name; W3.2 adds 8 names (4 k=5 entries + 4 k=6 entries across u32+u64 × count+materialize).
* The slice-1 promoter's atom-flattening logic — if any — and where to plug in `try_promote_clique_k`.
* Existing CPU oracle for triangle/4-cycle in test-side code; pattern to mirror.
* Runtime counter pattern (`wcoj_triangle_dispatch_count` etc.) for the new clique counters.

**Output**: a 6-bullet audit note in the evidence README confirming each finding (no code change in step 1).

### Step 2 — CUDA template kernel

Append to `crates/xlog-cuda/kernels/wcoj.cu`:

* Templated `__device__` helpers `wcoj_clique_per_thread_count_t<K>` (u32) and `_u64` for the per-thread count + emit. Algorithm: leader-edge-driven iteration; for each leader row (an edge's `(col0, col1)`), recursively bind the remaining k-2 variables via `lower_bound`/`upper_bound` ranges over the appropriate edge buffers, intersect to find valid extensions. Generic over K via `template <int K>` so the compiler unrolls the recursion at K=5 and K=6.
* 8 `extern "C" __global__` ABI wrappers (k=5/k=6 × count/materialize × u32/u64), each calling `wcoj_clique_template_*_t<K>(...)` with K = 5 or 6. **Each wrapper body is ≤ 5 lines.** The k=6 wrappers MUST contain only the template call — no hand-written algorithm.
* `kernels/CMakeLists.txt` updated only if new compile units are needed (likely not — `wcoj.cu` already builds; new templates compile in place).

### Step 3 — Kernel manifest registration

Update `crates/xlog-cuda/src/kernel_manifest_data.rs`:
* Append `"wcoj_clique5_count_u32"`, `"wcoj_clique5_materialize_u32"`, `"wcoj_clique5_count_u64"`, `"wcoj_clique5_materialize_u64"`, `"wcoj_clique6_count_u32"`, `"wcoj_clique6_materialize_u32"`, `"wcoj_clique6_count_u64"`, `"wcoj_clique6_materialize_u64"` to the manifest array.

### Step 4 — Provider entries (u32 width-class)

`crates/xlog-cuda/src/provider/wcoj.rs`:
* `wcoj_clique5_u32_recorded(edges: &[&CudaBuffer; 10], launch_stream)` — orchestrates count → scan → materialize per the slice-1 pattern; reuses `wcoj_compute_total`. Validation: runtime-backed, each input arity == 2, each column ∈ {U32, Symbol}.
* `wcoj_clique6_u32_recorded(edges: &[&CudaBuffer; 15], launch_stream)`.
* Both assume sorted + deduped input as a pre-condition (same contract as `wcoj_triangle_u32_recorded` / `wcoj_4cycle_u32_recorded`). The runtime dispatcher (step 7) routes every edge through W3.1's `wcoj_layout_sort_u32_recorded` unconditionally before calling these provider entries — the provider does NOT call layout-sort itself, and there is no provider-side "if already sorted + unique" branch (W3.2 does not implement a sortedness checker for these entries).

### Step 5 — Provider entries (u64 width-class)

`wcoj_clique5_u64_recorded` / `_clique6_u64_recorded`. Mirror of step 4 with u64-sized columns and `_u64` kernel variants.

### Step 6 — Promoter (`try_promote_clique_k`)

`crates/xlog-logic/src/promote.rs`:
* Tree-flatten helper `collect_positive_scans(node, &mut Vec<RelId>)` — only descends through outermost `Project` and into `Join` subtrees; **rejects on `Filter`** (per fix #5; semantic preservation gate).
* `try_promote_clique_k(node, k, stats, config) -> Option<RirNode>` for k ∈ {5, 6}:
  1. Outermost `Project`-of-`Join` shape match. If any descendant under the outer Project is a `Filter`, return `None` immediately.
  2. Flatten Joins → list of `(rel_id, lk, rk, atom_var_order)` triples and the head's column projection. `atom_var_order` records the (head-variable-index, head-variable-index) pair the atom binds, in the order they appear in the atom.
  3. Reject if scan count != `C(k, 2)`.
  4. Reject if head doesn't have exactly k distinct variables.
  5. Reject if `recursive_scan_count >= 1` (linear-recursive clique body — falls through to binary-join per fix #8).
  6. For each scan, derive its bound variable pair from the join tree via union-find on key equivalences (same shape as `infer_triangle_semantics`). **Reject** if the atom's variable order is reversed: i.e. for the bound pair `(v_i, v_j)` with head-index `i < j`, the atom's `(col0, col1)` must bind `(v_i, v_j)` in that order. Reversed `e(v_j, v_i)` is rejected (per fix on argument permutation).
  7. **Reject** if any atom binds the same variable twice (`e(X, X)` self-edge).
  8. **Reject** if any atom binds a constant rather than a variable.
  9. Build a `BTreeSet<(usize, usize)>` of bound pairs; reject if it doesn't equal `{(i, j) | 0 ≤ i < j < k}`.
  10. Reorder scans into canonical lex `(i, j)` order; emit `RirNode::MultiWayJoin` with that order, slot_vars matching D6's table, fallback = original body.
* Wire into `promote_multiway` after the triangle and 4-cycle promoters:
  ```rust
  if let Some(p) = try_promote_clique_k(&rule.body, 5, ...) { rule.body = p; continue; }
  if let Some(p) = try_promote_clique_k(&rule.body, 6, ...) { rule.body = p; continue; }
  ```
  Order is a doc anchor only; a body matching k=5 cannot also match k=6 (different scan count).
* **Recursive clique bodies are rejected in W3.2** (fix #8). The recursive WCOJ helper at `executor::execute_recursive_scc` is NOT extended for clique-keyed dispatch. Recursive clique bodies are rejected at step 5 above; they fall through to binary-join. **Rejected in W3.2 — no closure credit.**

### Step 7 — Runtime dispatcher

`crates/xlog-runtime/src/executor/wcoj_dispatch.rs`:

* **Storage**: two new `pub(super)` u64 fields
  `wcoj_clique5_dispatch_count` and `_clique6_dispatch_count`
  on `Executor`, mirroring the existing
  `wcoj_triangle_dispatch_count` / `wcoj_4cycle_dispatch_count`
  layout.
* **Public accessors** (per fix #6 — integration certs need to
  assert these counters across crate boundaries; `pub(super)`
  fields alone don't expose to `xlog-integration` tests):
  ```rust
  impl Executor {
      pub fn wcoj_clique5_dispatch_count(&self) -> u64 { self.wcoj_clique5_dispatch_count }
      pub fn wcoj_clique6_dispatch_count(&self) -> u64 { self.wcoj_clique6_dispatch_count }
  }
  ```
  Mirrors the existing `pub fn wcoj_triangle_dispatch_count` /
  `wcoj_4cycle_dispatch_count` accessors.
* `try_dispatch_wcoj_clique5(plan_body, slot_rels, ...)` and
  `_clique6` shape-matched on the new MultiWayJoin's
  `inputs.len()` (10 for k=5, 15 for k=6).
* Each dispatcher:
  1. Looks up the relation buffers from `slot_rels` (10 / 15
     buffers).
  2. **Always** routes every edge through W3.1's
     `wcoj_layout_sort_u32_recorded` / `_u64_recorded`
     unconditionally (per fix #7 — no "if not already sorted +
     unique" conditional shortcut; W3.2 does not implement a
     sortedness checker for the new entries, so the layout step
     runs every time). The W3.1 helper's downstream
     `dedup_full_row_recorded` handles `n == 0` and
     already-sorted inputs without additional W3.2-level branch.
  3. Calls the new provider entry (`wcoj_clique5_u32_recorded`
     / etc.) on the laid-out inputs.
  4. On success: counter += 1; result returned to the executor.
  5. On error / kernel-launch failure: silent fallback to the
     original binary-join body via `MultiWayJoin.fallback`.
     **Counter is NOT incremented** on the fallback path —
     step 9's fallback certs assert this observably.
* No force / kill / adaptive knobs in W3.2 per D8.

### Step 8 — Cert: provider × width-class grid

`crates/xlog-cuda/tests/test_wcoj_clique5.rs` — 3 tests (u32, u64, Symbol). Each:

1. Build a small (≤ 6-vertex) clique fixture as a host-side
   `Vec<(T, T)>` per edge, in canonical lex `(i, j)` edge order
   (D6).
2. Upload each edge to a `CudaBuffer` via the per-width-class
   helper.
3. **Layout-sort + dedup each edge buffer through W3.1's
   `wcoj_layout_sort_u32_recorded` (or `_u64_recorded`)
   BEFORE the provider call.** This is explicit in every cert
   in step 8 — the provider entries assume sorted+deduped
   input as a pre-condition (per fix #8 / step 4-5 contract);
   the cert MUST satisfy that pre-condition by routing through
   the W3.1 helper, exactly mirroring the runtime dispatcher's
   pattern (step 7).
4. Call provider entry on the laid-out edges.
5. Assert output row set (downloaded as `Vec<[T; 5]>`) equals
   `cpu_clique5_reference::<T>(host_edges)` ignoring row order
   (compare via `BTreeSet`).

`crates/xlog-cuda/tests/test_wcoj_clique6.rs` — 3 tests (u32, u64, Symbol). Same 5-step pattern at K=6 with 15 edge buffers.

`crates/xlog-cuda/tests/common/clique_oracle.rs` (shared test
fixture module, mirroring how W3.1's tests share `RuntimeFixture`)
— `cpu_clique_reference<T, const K: usize>(edges: &[Vec<(T, T)>]) -> Vec<[T; K]>`
brute-force enumerator, generic over `T: Copy + Ord + Eq + Hash`,
with first-line `assert_eq!(edges.len(), K * (K - 1) / 2, ...)`
runtime check. Plus thin concrete wrappers
`cpu_clique5_reference<T>(edges) -> Vec<[T; 5]>` and
`cpu_clique6_reference<T>(edges) -> Vec<[T; 6]>` for ergonomic
test-side call-sites. Instantiated at `T = u32` (covers U32 +
Symbol fixtures on the 4-byte path; Symbol IDs are u32 bits per
W3.1 convention) and `T = u64` for the U64 fixtures.

**Subtotal: 6 provider tests.**

### Step 9 — Cert: runtime dispatch + fallback counter

`crates/xlog-integration/tests/test_wcoj_clique_dispatch.rs`:

* `clique5_dispatch_counter_advances_and_row_set_matches_fallback_body`
  — compile a 5-clique rule, run under default dispatch, assert
  `executor.wcoj_clique5_dispatch_count() >= 1` AND row set equals
  the fallback body's row set. **Reference is built from
  `MultiWayJoin.fallback`**, NOT from a clique-off env / config
  knob (W3.2 forbids new force/kill/adaptive knobs). The
  reference is obtained via a test-only RIR rewrite helper
  `replace_multiway_with_fallback(plan: ExecutionPlan) -> ExecutionPlan`
  that walks the plan tree, detects `RirNode::MultiWayJoin`
  nodes whose `inputs.len()` is 10 (k=5) or 15 (k=6), and
  substitutes the node with its `fallback` field. The rewrite
  helper lives in test code only — no production callers, no
  new `RuntimeConfig` field, no env var.
* `clique6_dispatch_counter_advances_and_row_set_matches_fallback_body`
  — same shape at k=6.
* `clique5_dispatcher_decline_does_not_advance_counter_and_row_set_matches_fallback`
  — **MultiWayJoin is emitted by the promoter**, but
  `try_dispatch_wcoj_clique5` declines internally. The decline
  is engineered by uploading one of the 10 edge buffers with a
  malformed schema that passes promoter validation but fails
  dispatcher validation: e.g. column-2 buffer has `ScalarType::I64`
  in its schema (not in the U32/Symbol/U64 width-classes).
  Promoter sees the canonical clique shape and promotes;
  dispatcher's per-edge width-class check rejects, returns
  `Ok(None)`, and the executor falls through to the
  `MultiWayJoin.fallback` body. Assert:
    1. The compiled plan contains a `MultiWayJoin` with
       `inputs.len() == 10` (proves promotion happened).
    2. `executor.wcoj_clique5_dispatch_count() == 0` after the
       run (proves the dispatcher declined; counter is
       observable per fix #6).
    3. Row set matches the `MultiWayJoin.fallback` body's
       output (built via the test-only
       `replace_multiway_with_fallback` rewrite, same as the
       counter-advance certs above).
* `clique6_dispatcher_decline_does_not_advance_counter_and_row_set_matches_fallback`
  — same shape at k=6 (15 edge buffers; one edge with
  malformed schema).

These two cells specifically prove `try_dispatch_wcoj_clique*`'s
silent-decline path. They are NOT promoter-decline tests — those
are covered separately in step 10 (`linear_recursive_clique5_does_not_promote`,
`clique5_with_filter_wrapper_rejected`, etc.). The fallback-counter
cert is meaningful only when a MultiWayJoin DOES get emitted and
the dispatcher then declines internally — exactly what this pair
exercises.

**Subtotal: 4 runtime dispatch tests** (was 2 in iteration 1; +2
fallback-counter cells per user iteration-1 fix #4).

### Step 10 — Cert: promoter shape

`crates/xlog-logic/tests/test_w32_clique_promoter.rs`:
* **Positive:** `clique5_left_deep_promotes`, `clique5_right_deep_promotes`, `clique5_bushy_promotes`, `clique6_left_deep_promotes`, `clique6_right_deep_promotes`, `clique6_bushy_promotes` — 6 tests proving the flatten-and-validate strategy is shape-robust per D5.
* **Negative shape:**
  * `non_clique_5_atoms_with_missing_edge_does_not_promote` — 10 atoms but one variable pair is missing (only 9 distinct edges + 1 duplicate) → promoter declines.
  * `clique5_with_self_edge_rejected` — one atom is `e(X, X)` (self-loop binding the same head variable twice). Promoter must reject (per fix on argument permutation; named per user request "clique5_with_self_edge_rejected").
  * `cycle_5_does_not_promote` — pentagon (5 atoms / 5 edges, not 10) — must NOT match k=5 clique.
  * `disconnected_subcomponents_do_not_promote` — 10 atoms but they form K_4 + K_3 + an extra triangle, not K_5.
  * `clique_with_constant_in_atom_does_not_promote` — atom has a constant; not part of the canonical clique pattern.
  * **`clique5_with_reversed_atom_rejected`** (per fix on argument permutation) — one atom uses `e(v_j, v_i)` where the canonical form is `e(v_i, v_j)` with `i < j`. W3.2 does not implement column-swap layout for clique edges; silent acceptance would mis-align the kernel's per-edge layout with the promoter's slot table. Promoter must reject.
  * **`clique5_with_filter_wrapper_rejected`** (per fix #5) — body has the canonical 10-atom clique structure but is wrapped in a `RirNode::Filter`. Promoter must reject (filter-preservation gate; semantic loss otherwise).
  * **`linear_recursive_clique5_does_not_promote`** (per fix #8) — 5-clique body where one atom resolves to a recursive RelId. The W3.2 promoter declines (does not extend recursive WCOJ helper for clique-keyed dispatch); body falls through to binary-join path with row-set parity preserved.
* **k=7 unsupported sentinel:** `clique7_does_not_promote` — 7-clique body has 21 atoms; promoter rejects (W3.2 only handles k ∈ {5, 6}). The dispatcher's silent-fallback-to-binary-join preserves row-set parity in this case.

**Subtotal: 6 positive + 8 negative + 1 k=7 sentinel = 15 promoter tests** (was 12 in iteration 1; +3 cells per fixes #5 / #8 / argument-permutation lock).

### Step 11 — Cert: k=6 template-source

`crates/xlog-cuda/tests/test_w32_kernel_source_audit.rs`.
**Two-tier audit** of `crates/xlog-cuda/kernels/wcoj.cu`:

* **Tier 1 — ABI wrapper bodies are template-call-only** (4 cells).
  Each of the four k=6 wrapper definitions must satisfy a strict
  syntactic contract — formatting variations cannot hide a
  hand-written body inside the wrapper itself:

  * `k6_count_u32_wrapper_is_template_call_only` — read
    `wcoj.cu` as a string, locate the
    `extern "C" __global__ void wcoj_clique6_count_u32(...) { ... }`
    definition, parse the body. Assert:
    1. The body contains **exactly one statement** (one
       semicolon-terminated expression).
    2. That statement is a call to the shared template
       (`wcoj_clique_template_count_t<6>(...)` — the exact
       template name is locked in step 2).
    3. The body contains **NO loop keywords** (`for`, `while`,
       `do`).
    4. The body contains **NO conditionals** (`if`, `switch`,
       ternary `?:`).
    5. Whitespace and comments don't bypass the check (the
       parser strips both before counting statements).
  * `k6_count_u64_wrapper_is_template_call_only` — same contract
    for the U64 count wrapper.
  * `k6_materialize_u32_wrapper_is_template_call_only` — same
    contract for the U32 materialize wrapper.
  * `k6_materialize_u64_wrapper_is_template_call_only` — same
    contract for the U64 materialize wrapper.

* **Tier 2 — no k=6-specific algorithm body anywhere in `wcoj.cu`**
  (4 cells). The Tier-1 wrapper-body audit is necessary but NOT
  sufficient — a hand-written k=6 specialization could hide
  behind the template-call by living elsewhere in the file:

  * `no_explicit_k6_template_specialization` — the file MUST
    NOT contain any **explicit template specialization** for
    K=6. Forbidden patterns (regex-stripped of whitespace +
    comments before matching):
    - `template<>` / `template <>` followed by any function
      name containing `clique` or `wcoj_clique`, with `<6>` in
      its name. (e.g.
      `template <> __device__ void wcoj_clique_template_count_t<6>(...)`).
    - Any `template<...>` declaration whose body is followed
      by an explicit `<6>` instantiation that has its own
      `{ ... }` body (full specialization with custom code).
    Implicit instantiations from the wrapper's `<6>` call site
    are explicitly ALLOWED — the test parses each `<6>` token
    occurrence and asserts none of them sit at the start of a
    `template <>`-prefixed function definition.

  * `no_if_constexpr_k_equals_6_branch` — within the shared
    template's `__device__` body (located by name
    `wcoj_clique_template_count_t` / `_emit_t` / `_u64`
    variants), the file MUST NOT contain any branch keyed on
    K-equals-6 with a k=6-specific algorithm. Forbidden
    patterns:
    - `if constexpr (K == 6)` with a non-empty body.
    - `if (K == 6)` (runtime branch on the template parameter
      — the K=6 specialization must come from instantiation,
      not from a runtime check).
    - The same forbidden patterns with `K == 5` are also
      flagged — the template MUST be uniformly K-parameterized;
      neither K=5 nor K=6 is allowed to have a hand-written
      branch.
    K-independent constexpr branches (e.g.
    `if constexpr (K > 2)`) are allowed.

  * `no_clique6_helper_function_body` — outside the four ABI
    wrappers and the shared template, the file MUST NOT
    contain any `__device__` or `__global__` function whose
    name contains `clique6` (case-insensitive) and has a body
    of more than one statement. The four ABI wrappers
    (audited in Tier 1) are the only `clique6`-named entities
    permitted to have a function body. Any additional
    `clique6`-named helper would be a hand-written k=6
    algorithm hidden alongside the template.

  * `no_six_literal_in_template_body` — within the shared
    template's `__device__` body, the file MUST NOT contain
    any integer literal `6` (or `5`) used in algorithmic
    context. The template is K-parameterized; k=5 and k=6
    must come from `<5>` / `<6>` instantiation, not from
    hardcoded `6` / `5` literals embedded in the algorithm.
    Token whitelist: literals appearing inside comments, in
    template parameter ranges (e.g. `template <int K = 5>`
    default value if any), and in static-assert bounds checks
    (e.g. `static_assert(K >= 3 && K <= 6, ...)`) are allowed
    via a per-line context-strip pre-pass before the literal
    scan.

The two-tier structure closes the escape hatches the user
flagged in iteration-3 review: Tier 1 prevents a hand-written
body inside the ABI wrappers; Tier 2 prevents a hand-written
k=6 specialization, runtime branch, helper function, or
hardcoded-K literal anywhere else in the file. Together they
enforce the board contract literally: **no new `.cu` source for
k=6** beyond ABI wrapper names plus calls/instantiations using
`<6>`.

**Subtotal: 4 Tier-1 + 4 Tier-2 = 8 source-audit tests** (was 4
in iteration 3; +4 Tier-2 cells per user iteration-3 strengthen
gate).

### Step 12 — Workspace gate

* **Compile/link budget gate** (per fix #1; iteration-1 Q1
  answer): `cargo build -p xlog-cuda --release` must succeed with
  the k=6 template instantiations included AND the k=6 provider
  cert (`test_wcoj_clique6`) must compile + run without changing
  the template strategy. If register pressure or compile-time
  blow-up breaks this gate, **W3.2 is not closed** — execution
  pauses for re-direction (e.g. K-specific tuning, reduced
  template depth) before any code lands. The gate is the explicit
  hard-stop on the template strategy locked in D1.
* `cargo fmt --check --all` clean.
* `cargo test -p xlog-cuda --release --test test_wcoj_clique5` (3/3).
* `cargo test -p xlog-cuda --release --test test_wcoj_clique6` (3/3).
* `cargo test -p xlog-cuda --release --test test_w32_kernel_source_audit` (8/8 — 4 Tier-1 wrapper-body cells + 4 Tier-2 specialization/branch/helper/literal cells).
* `cargo test -p xlog-logic --release --test test_w32_clique_promoter` (15/15 — see updated step-10 count).
* `cargo test -p xlog-integration --release --test test_wcoj_clique_dispatch` (4/4 — see updated step-9 count).
* `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` — pass count must increase by **+33** (the new W3.2 tests; iteration-3 total 29 + 4 Tier-2 source-audit cells from iteration-4). 0 fail. Symbolic delta only — global pre-W3.2 baseline is reported in evidence at execution time, not pinned in the assertion.
* `cargo test -p xlog-cuda-tests --test certification_suite --release` — `run_full_certification` PASS (1/1).
* `cargo test -p xlog-cuda --release --test test_wcoj_layout_u32` and `test_wcoj_layout_u64` — 9/9 + 6/6 unchanged (W3.1 + slice-1 arity-2 hot path bit-identical).
* `cargo test -p xlog-cuda --release --test test_wcoj_layout_sort_u32` / `_u64` / `_roundtrip` — 5/5 + 5/5 + 72/72 unchanged.
* `cargo test -p xlog-integration --release --test test_wcoj_record_join_result_feedback` — 3/3 unchanged.
* `cargo test -p xlog-integration --release --test test_wcoj_recursive_dispatch` — 6/6 unchanged.
* `cargo test -p xlog-integration --release --test test_w26_heat_selectivity` — 7/7 unchanged.
* `cargo test -p xlog-integration --release --test test_w21_variable_ordering` — 11/11 unchanged.

### Step 13 — Evidence README

`docs/evidence/2026-05-06-w32-general-arity-wcoj-template/README.md`
following the W2.6 / W3.1 README structure:

* Header: "Closes W3.2 only", branch, base hash (d5073bdb), plan reference.
* Summary.
* Acceptance properties table mapping each board-line claim to its cert(s).
* Cert results: full `cargo test` output for the 6 + 4 + 15 + 8 = 33 new tests (provider × width-class + runtime dispatch incl. fallback + promoter incl. negatives + k=6 source-audit Tier 1 + Tier 2).
* Workspace tally with the post-W3.2 numbers.
* Code-level changes table (only `wcoj.cu` + `wcoj.rs` provider + `promote.rs` + `wcoj_dispatch.rs` + 5 new test files).
* Decision mapping (D1–D8, with rationale).
* Process rule compliance.
* Closure proposal (gated on user approval): board update proposal `OPEN → DONE`, tally `DONE: 6 → 7; OPEN: 13 → 12`.

### Step 14 — Closure proposal + FF-merge (gated on user approval)

* Do NOT modify `docs/v065-closure-board.md` until user approves.
* Do NOT FF-merge until user approves.
* Do NOT push, do NOT tag.

## Test counts summary (locked)

| Part | Description | # tests |
|------|-------------|--------:|
| Provider × width-class — k=5 | u32 / u64 / Symbol | 3 |
| Provider × width-class — k=6 | u32 / u64 / Symbol | 3 |
| Runtime dispatch — counter advance | k=5 + k=6 (row set vs `MultiWayJoin.fallback`) | 2 |
| Runtime dispatch — fallback cert | k=5 + k=6 (counter does NOT advance, row set parity) | 2 |
| Promoter positive shape | left-deep / right-deep / bushy × k=5/k=6 | 6 |
| Promoter negative shape | missing-edge, self-edge, cycle-5, disconnected, constant-in-atom, reversed-atom, filter-wrapped, linear-recursive | 8 |
| Promoter k=7 unsupported sentinel | shape-rejected, fallback preserved | 1 |
| k=6 source-audit Tier 1 | wrapper bodies template-call-only (u32/u64 × count/materialize) | 4 |
| k=6 source-audit Tier 2 | no explicit `<6>` specialization / no `K == 6` branch / no `clique6` helper body / no hardcoded-K literal in template body | 4 |
| **W3.2 acceptance total** | | **33** |

## Process Rule Compliance

* Process rule #1: this slice does **not** self-mark W3.2 DONE.
* Process rule #2: every commit references W3.2.
* Process rule #3: this plan opens with "Closes W3.2 only."
* Process rule #5: no `v0.6.6` references; no punt-to-later wording — out-of-scope items are owned by W3.3+ board items, named at the point of reference, not pre-named here as W3.2's responsibility.
* Process rule #6: no push, no tag.

## Iteration 3 → 4 Patch Log

One blocking item from iteration-3 review:

* **Strengthen Step 11 source-audit beyond ABI wrappers.**
  Iteration 3's source-audit (4 cells) only proved the four
  `wcoj_clique6_*` ABI wrapper bodies were template-call-only.
  That left three escape hatches for a hand-written k=6
  algorithm:
  1. Explicit template specialization
     `template <> ... wcoj_clique_template_count_t<6>(...) { ... }`
     elsewhere in the file.
  2. `if constexpr (K == 6) { ... }` branch inside the shared
     template body.
  3. A separate `clique6_*`-named helper function with a
     hand-written body, called by the template indirectly.
  Plus a fourth: hardcoded `6` / `5` integer literals in the
  template body (proxy for K-specific code).

  Step 11 restructured into a **two-tier** audit:
  * **Tier 1** (4 cells) — the iteration-3 wrapper-body
    contract, unchanged: `extern "C" __global__ void
    wcoj_clique6_*(...)` body must be exactly one statement
    that calls the shared template, no loops/conditionals.
  * **Tier 2** (4 NEW cells) — file-wide forbidden-pattern
    audit: `no_explicit_k6_template_specialization`,
    `no_if_constexpr_k_equals_6_branch` (and the symmetric
    `K == 5` check),
    `no_clique6_helper_function_body` (only the four ABI
    wrappers may have `clique6` in their name and a body),
    `no_six_literal_in_template_body` (k=5/k=6 must come from
    `<5>` / `<6>` instantiation, not from hardcoded literals;
    static-assert and template-default contexts are
    whitelisted via per-line context-strip).

  Subtotal: 4 → 8 source-audit cells. Acceptance total: 29 → 33.
  Step 12 + Step 13 arithmetic + test-count summary table all
  reconciled to 33.

  Per the user's iteration-3 lock: "The only allowed k=6-specific
  `.cu` text should be ABI wrapper names plus calls/instantiations
  using `<6>`. No `template <>`, no `if constexpr (K == 6)`
  branch with k=6-specific algorithm, no `clique6` helper
  algorithm body." Tier 2 enforces this verbatim.

## Iteration 2 → 3 Patch Log

Five blocking items + three required clarifications the user
flagged in iteration-2 review:

**Blocking #1 — punt-to-later wording removed from live body.**
Two paragraphs in iteration 2 (Step 6 filter-wrapper rejection
rationale and Step 6 recursive-clique rejection rationale)
named hypothetical follow-up work as W3.2-implied. Both
replaced with owned-now wording: "Rejected in W3.2 — no
closure credit." No follow-up-item invention. The
iteration-1 → 2 patch log's item #6 entry was also using the
banned construction; rewritten in place. Item #9 of that same
log was also using it for the recursive-clique resolution and
has been rewritten as well.

**Blocking #2 — fallback-counter cert engineers a dispatcher
decline (not a promoter decline).** Iteration 2's
`clique{5,6}_fallback_path_does_not_advance_counter_and_row_set_matches`
cells used a body the promoter would NOT promote — so they
proved "no dispatch" rather than "dispatcher fallback." Step 9
rewritten: the new cells
(`clique{5,6}_dispatcher_decline_does_not_advance_counter_and_row_set_matches_fallback`)
engineer a `MultiWayJoin` that DOES get emitted by the
promoter, then trigger an internal decline inside
`try_dispatch_wcoj_clique*` via a malformed edge buffer
(e.g. column with `ScalarType::I64` that passes promoter
validation but fails dispatcher per-edge width-class check).
Asserts: (a) MultiWayJoin emitted, (b) counter == 0, (c) row
set matches `MultiWayJoin.fallback`.

**Blocking #3 — `cpu_clique_reference` signature stable-Rust
implementable.** Iteration 2 had `edges: &[Vec<(T, T)>; K_CHOOSE_2]`
which requires `feature(generic_const_exprs)` on stable Rust
(unavailable). Rewritten as `edges: &[Vec<(T, T)>]` (runtime
slice) with first-line `assert_eq!(edges.len(), K * (K - 1) / 2,
...)` runtime check. Two ergonomic concrete wrappers
`cpu_clique5_reference` / `cpu_clique6_reference` added for
test-side call-sites.

**Blocking #4 — Step 13 evidence arithmetic.** Iteration 2's
step 13 still said "6 + 2 + 12 + 4 = 24 new tests" (iteration-1
totals). Updated to "6 + 4 + 15 + 4 = 29".

**Blocking #5 — Step 8 provider-cert pre-condition explicit.**
Iteration 2 said "call provider entry, assert output row set
matches oracle" without specifying that fixtures pre-satisfy
the provider's sorted+deduped pre-condition. Step 8 now spells
out a 5-step procedure: build host-side edges → upload →
**route through `wcoj_layout_sort_*_recorded` BEFORE provider
call** → call provider → compare against oracle. No implicit
"already sorted" assumption.

**Required clarification — recursive clique rejection.**
Step 6 + iteration-1 → 2 patch log item #6 reworded to use
"rejected in W3.2 — no closure credit" only. No "later item
can extend it" language.

**Required clarification — filter-wrapper rejection.** Step 6
reworded to use "rejected in W3.2 to preserve semantics — no
closure credit" only. No future-preserve-filter language.

**Required clarification — source-audit strictness.** Step 11
strengthened: each of the 4 k=6 wrapper cells must pass a
5-property contract — exactly-one-statement, template-call-only,
no loop keywords, no conditionals, comment/whitespace-stripping
pre-check. Closes the formatting escape hatch a hand-written
k=6 body could otherwise hide behind.

## Iteration 1 → 2 Patch Log

User answers to iteration-1 open questions, all incorporated:

| Q | Locked answer | Where applied |
|---|---------------|----------------|
| Q1 (K=6 compile budget) | Add explicit compile/link budget gate in step 12. If the template strategy breaks `cargo build -p xlog-cuda --release` at k=6 due to register pressure or compile-time blow-up, **W3.2 is not closed** — execution pauses for re-direction. | Step 12 (new first bullet). |
| Q2 (Symbol fixtures) | Real `xlog_core::symbol::intern("sym_<n>")` IDs, same as W3.1. No raw u32 bit patterns. No mixed U32/Symbol coverage in W3.2 — Symbol-only is sufficient. | Step 8 + D2 row. |
| Q3 (Negative cert breadth) | Add: `clique5_with_self_edge_rejected` (renamed from "extra_self_loop"), `clique5_with_reversed_atom_rejected` (per argument-permutation lock), `clique5_with_filter_wrapper_rejected` (per fix #5), `linear_recursive_clique5_does_not_promote` (per fix #8). | Step 10 (now 8 negative cells). |
| Q4 (Fallback observability) | Row-set parity alone is insufficient. Add explicit fallback-counter cert: `clique{5,6}_fallback_path_does_not_advance_counter_and_row_set_matches`. | Step 9 (now 4 runtime tests). |
| Q5 (Workspace delta) | Symbolic only; no global pin. After fixes #4 + #3 fold in, delta is **+29** (was +24 in iteration 1). | Step 12 + summary table. |
| Q6 (Plan-commit timing) | Procedure correct: keep plan untracked on `main` until iteration approval; on approval, create `.worktrees/w32-general-arity-wcoj-template`, copy plan there, commit as branch commit #1, remove untracked copy from `main`. | (No edit needed — procedural lock unchanged.) |

Eight blocking plan fixes the user flagged in iteration-1
review, all incorporated:

1. **K=6 compile/link budget gate.** Step 12's first bullet
   pins `cargo build -p xlog-cuda --release` and the k=6
   provider cert as the hard-stop on the template strategy.
2. **No clique-off knob in runtime cert.** Step 9 replaces
   "force-clique-OFF run on same fixture" with a test-only RIR
   rewrite helper that substitutes `MultiWayJoin` nodes with
   their `fallback` field. No new `RuntimeConfig` field, no env
   var, no force/kill/adaptive knob.
3. **Test-count table reconciled.** Iteration 1's table listed
   "Provider × width-class — k=5" twice and the row arithmetic
   didn't sum to 24. Iteration 2 lists each row once and
   totals **29**.
4. **Step count corrected.** Iteration 1's header said "Step
   plan (12 steps)" while the plan defined 14. Header now says
   "Step plan (14 steps)". Step 13 + 14 normalized to `###`
   heading depth like the others.
5. **`cpu_clique_reference` width-class generic.** D2 row
   updated: oracle is `<T, const K: usize>` with `T: Copy + Ord
   + Eq + Hash`. Concretely instantiated at `T = u32` (covers
   U32 + Symbol on 4-byte path) and `T = u64`. The previously
   monomorphic `Vec<[u32; K]>` return was insufficient for u64
   certs.
6. **Filter wrappers REJECT, do not silently strip.** Step 6
   updated: `try_promote_clique_k` rejects on any interior
   `RirNode::Filter`. Stripping a filter during promotion would
   risk semantic loss (kernel evaluates the join body without
   re-applying the filter). Rejected in W3.2 to preserve
   semantics — no closure credit.
7. **Public counter accessors.** Step 7 adds explicit
   `pub fn wcoj_clique5_dispatch_count(&self) -> u64` and
   `_clique6_dispatch_count` accessors on `Executor` —
   `pub(super)` fields alone don't expose to `xlog-integration`
   tests across the crate boundary.
8. **Provider always layout-sorts via the dispatcher.** Step 4
   + Step 5 + Step 7 reconciled: the runtime dispatcher
   unconditionally routes every edge through W3.1's
   `wcoj_layout_sort_*_recorded` BEFORE invoking the provider
   entry. Provider entries assume sorted+deduped input as a
   pre-condition (same contract as
   `wcoj_triangle_u32_recorded`); no provider-side
   "if not already sorted+unique" branch.
9. **Recursive clique behavior resolved.** Step 6 updated:
   linear-recursive clique bodies (`recursive_scan_count >= 1`)
   are explicitly rejected by `try_promote_clique_k`. The
   recursive WCOJ helper at `executor::execute_recursive_scc`
   is NOT extended for clique-keyed dispatch in W3.2. Recursive
   clique bodies fall through to binary-join. Rejected in W3.2
   — no closure credit. Step-10's
   `linear_recursive_clique5_does_not_promote` cert pins this
   contract.

## Open Questions for Iteration 5

Iteration 4 closes the iteration-3 blocker (Step 11 source-audit
strengthened to two-tier with file-wide forbidden-pattern
checks). All prior locks (live punt-to-later wording removed,
fallback-counter cert engineers a dispatcher decline,
`cpu_clique_reference` signature stable-Rust implementable,
step-13 arithmetic corrected, step-8 layout-sort pre-condition
explicit, recursive / filter rejection rewords, source-audit
strictness) remain in place. No structural or procedural
ambiguities remain. The plan-commit timing procedure is locked
per Q6: on iteration-4 approval, create
`.worktrees/w32-general-arity-wcoj-template`, copy plan there,
commit as branch commit #1 of `feat/w32-general-arity-wcoj-template`,
remove untracked plan from `main`. No code changes until
iteration 4 is explicitly approved.
