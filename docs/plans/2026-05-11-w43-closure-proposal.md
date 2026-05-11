# W4.3 Closure Proposal — Operator-Only Scope (per F-W43-14)

**Date:** 2026-05-11
**Branch:** `feat/w43-sort-merge-join`
**Plan:** `docs/plans/2026-05-10-w43-sort-merge-join-plan.md` (iteration 6 canonical)
**Step:** Step 8' (iteration-6 closure proposal)

---

## Status

W4.3 has 30 commits on `feat/w43-sort-merge-join` measured at the closure-proposal commit (`git rev-list --count 19f7bc5d..0d94ec75 = 30`). Subsequent self-amendment commits to this proposal document increase the live HEAD count but not the underlying delivered work; anchoring the count to commit `0d94ec75` keeps the headline stable. The sort-merge join operator is **implemented at the provider layer**, **bench-validated** (vs hash 1.10×–1.80× win on the 50×50–2000×2000 matrix), and **operator-cert-tested** (4 provider-level parity certs PASS). The executor dispatch wiring was **removed in iteration 6** per F-W43-14, because the Step 12 production bench empirically rejected the iteration-1 design hypotheses that motivated the dispatch path:

* **D7 #8 (≥ 2× vs hash) — REJECTED**: measured speedups 1.10×–1.80× on every cell. Sort-merge wins vs hash, but by a sub-2× margin that did not meet the gate.
* **D2 precedence (sort-merge > nested-loop) — REJECTED**: nested-loop wins 1.25×–2.46× on every overlap cell. The iteration-1 working hypothesis (sort-merge takes precedence on sorted inputs) was empirically wrong; nested-loop dominates the entire shared eligibility envelope.

The iteration-1 plan correctly anticipated this possibility (D2 marked **PROVISIONAL** per F-W43-2): *"If the bench shows nested-loop wins on the overlap, iteration-N+ amends D2."* Iteration 6 IS that amendment. The actual iteration-6 D2 is **"W4.3 has NO production dispatch"** — the executor's join decision tree reverts to W4.2 nested-loop (when eligible) then hash (otherwise); the W4.3 sort-merge operator is implemented at the provider layer but is not wired into the dispatch decision tree. The Step 12 bench is the documented rejection record, not an acceptance success.

---

## What was delivered

### Operator surface (PRESERVED as graduated implementation work)

* `crates/xlog-cuda/kernels/sort.cu::check_ascending_sorted_u32` — single-pass adjacent-pair sortedness detection kernel.
* `crates/xlog-cuda/kernels/join.cu::sort_merge_join_inner_u32_1key_pairs` — per-thread binary-search emit-pairs sort-merge join kernel (1-key U32/Symbol inner join, run-length emit path).
* `crates/xlog-cuda/src/provider/relational.rs::is_sorted_ascending_u32` — provider wrapper with `n < 2 → Ok(true)` empty-input fast path (per F-W43-4).
* `crates/xlog-cuda/src/provider/relational.rs::sort_merge_join_v2_inner_u32_1key` — provider entry point, drop-in compatible with `hash_join_v2 Inner` (same `combine_schemas` output schema).
* `crates/xlog-cuda/src/kernel_manifest_data.rs` — both kernels registered.
* `crates/xlog-cuda/src/provider/mod.rs` — kernel-name constants `SORT_MERGE_JOIN_INNER_U32_1KEY_PAIRS` + `CHECK_ASCENDING_SORTED_U32`.

### Operator-level parity certs (4 tests, all PASS)

`crates/xlog-integration/tests/test_w43_sort_merge_dispatch.rs`:

* **Cert A** — `sort_merge_operator_parity_sorted_unique_u32`: sorted 100-row 1-key U32 fixture → `BTreeSet<[u32;4]>` parity vs `hash_join_v2 Inner`.
* **Cert E** — `sort_merge_operator_parity_sorted_unique_symbol`: Symbol-typed buffers; proves byte-identical Symbol/U32 kernel surface.
* **Cert F** — `sort_merge_operator_parity_duplicate_key`: 250 keys × 4 dups → 4000 output rows, all (lk, lp, rk, rp) tuples distinct, parity vs hash. Exercises the `lower_bound`/`upper_bound` per-thread run-length emit path.
* **Cert G** — `sort_merge_operator_empty_input_layered_short_circuit`: two subcases (empty L, empty R), each verifies the F-W43-4 layered short-circuit contract end-to-end at the provider layer.

### Production bench + counter-finding evidence (PRESERVED as rejection record)

* `crates/xlog-integration/benches/w43_production_sort_merge_bench.rs` — provider-direct + detection methodology (per F-W43-3 intent). 6 cells from L=R=50 to L=R=2000.
* `docs/evidence/2026-05-10-w43-production-bench/README.md` — median timings + speedup tables for Part A (vs hash) + Part B (vs nested-loop) + decision-validation conclusion documenting D7 #8 + D2 rejections.

### What was REMOVED (per F-W43-14 iteration-6 unwiring)

* `eligible_for_sort_merge` predicate (from `node_dispatch.rs`).
* `if eligible_for_sort_merge { ... }` branch in `execute_join` (executor dispatch wiring).
* `if out.is_none()` wrap on the W4.2 nested-loop branch (no longer needed since W4.3 cannot consume the slot).
* `sort_merge_dispatch_count: u64` field on `Executor` + constructor initializer.
* `sort_merge_dispatch_count()` accessor on `Executor`.
* Iteration-1–5 dispatch-shape certs B / C / D / D' (superseded by W4.2 cert suite which already covers fall-through fixture shapes for the production-routing guard).

### W4.2 cert suite (UNCHANGED, production-routing guard)

`crates/xlog-integration/tests/test_w42_nested_loop_dispatch.rs` — 5 tests PASS unchanged. After F-W43-14 unwiring, W4.2 dispatch is the only join-operator dispatch in the executor; the W4.2 cert suite is the production-routing regression net.

---

## Verification (Step 7' final gate, per F-W43-12 + F-W43-15 enumerated-files exception)

* `cargo fmt --check --all` → exit 0.
* `RUSTFLAGS="-D warnings" cargo build --release --workspace --exclude pyxlog` → 0 warnings, 0 errors.
* `cargo bench -p xlog-integration --bench w43_production_sort_merge_bench --no-run` → exit 0.
* `cargo test -p xlog-cuda-tests --test certification_suite --release` → **1/1 (authoritative gate per MEMORY.md)**.
* `cargo test -p xlog-integration --release --test test_w43_sort_merge_dispatch` → **4/4 PASS**.
* `cargo test -p xlog-integration --release --test test_w42_nested_loop_dispatch` → **5/5 PASS**.
* `cargo test -p xlog-cuda --release --test test_wcoj_layout_sort_roundtrip --test test_wcoj_layout_sort_u32 --test test_wcoj_layout_sort_u64` → **82/82 PASS** (sibling files NOT exempt; verified during Step 7' review per F-W43-15).
* `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` → exits 0 on the user's Step 7' verification run; flakes only in the three F-W43-12 + F-W43-15 enumerated files (`test_wcoj_layout_fast_path.rs`, `test_wcoj_layout_u32.rs`, `test_wcoj_layout_u64.rs`) — pre-existing v0.6.2 WCOJ-fast-path GPU-state-pollution flake class, out-of-W4.3-scope per the F-W43-12 deferral.

---

## Closure-board question

**Does the v0.6.5 closure board accept "operator implemented, production dispatch rejected by evidence" as a valid completion of the W4.3 board item, or does it require a different completion criterion?**

The board entry in `docs/v065-closure-board.md` currently reads:

> W4.3 | ROADMAP item #15 | OPEN | — | General sort-merge join operator for pre-sorted binary relations. Triangle-layout helper is a special case; this is the generic path. | Cert: pre-sorted binary join skips the sort step, matches reference output.

The cert criterion ("pre-sorted binary join skips the sort step, matches reference output") **is satisfied at the operator/provider layer** — `provider.sort_merge_join_v2_inner_u32_1key` does exactly that, and Cert A/E/F/G verify it via row-set parity vs `hash_join_v2 Inner`. The board entry does NOT specify "must be wired into production dispatch"; that interpretation came from iteration-1's reading of the roadmap item, which iteration 6 has revised based on bench evidence.

### Possible board responses

| # | Response | Implication |
|---|----------|-------------|
| **1** | **Accept as DONE (Recommended)** | The cert criterion is met at the operator layer. Tally: DONE 9→10, OPEN 10→9. Close W4.3. The operator is preserved for any future v0.6.6+ caller (e.g., a hypothetical kernel-perf-improved dispatch path that recovers the spike's 1-col 2.5×–3.3× advantage on multi-col arity). |
| **2** | **Reject; require operator removal entirely** | Revert all W4.3 commits except plan/evidence; mark W4.3 ABANDONED. Loses the operator + kernels + bench evidence + cert suite as graduated work. Aggressive; treats "no production dispatch" as equivalent to "no work delivered." Not recommended given the operator passes correctness certs + the spike showed it CAN win on 1-col, just not on production multi-col arity. |
| **3** | **Defer** | Keep W4.3 OPEN; reopen in v0.6.6 with kernel-perf investigation (multi-col gather optimization) as the new scope. Tally unchanged. Preserves all work; defers the closure question pending kernel-perf evidence. Useful if the board wants a clear "implementation complete + perf-investigated" gate before accepting. |

### Recommendation

**Response 1 (Accept as DONE)** is the cleanest mapping of the delivered work to the board's stated cert criterion. The iteration-1 design hypothesis (sort-merge > nested-loop as production default) was empirically wrong, but that's exactly what the F-W43-2 PROVISIONAL guard was designed to catch — iteration 6 closed the conditional cleanly with the operator preserved and the production-routing guard (W4.2 cert suite) intact. The operator-only scope is honest about what production traffic gets: nested-loop is the production join for the W4.2/W4.3 shared eligibility envelope, hash is the fallback, and the W4.3 sort-merge operator is available at the provider layer for any future caller that has different perf characteristics than the current 3-col production arity.

If the board accepts Response 1, the follow-up actions are:
* Update `docs/v065-closure-board.md` W4.3 row from OPEN to DONE with an entry describing the iteration-6 operator-only scope.
* Create `memory/project_w43_closed.md` per the W4.1/W4.2 closure precedent (e.g., `project_w42_closed.md`).
* Update `MEMORY.md` v0.6.5 Closure Board section with the W4.3 closure pointer.
* FF-merge `feat/w43-sort-merge-join` to `main` (per W4.1/W4.2 precedent; bench-spike `bench-spike/w43-sort-merge` HEAD `fadc2700` remains unmerged per `feedback_perf_bench_spike_first.md`).

None of those follow-up actions are executed by this commit — Step 8' is text-only per D8 process locks. The actions land in separate commits AFTER the board decision.

---

## Commit history (30 commits on `feat/w43-sort-merge-join` measured at Step 8' commit `0d94ec75`)

Pre-execution iterations (1–4): 5 plan-iteration commits.
Iteration-5 execution + amendments: 14 commits (Steps 2–11 + iteration-5 amendments F-W43-11/12/13).
Iteration-6 execution + amendments: 10 commits (Step 12 bench + F-W43-14 amendment + Step 4'+5' unwiring + cert rewrite + doc patches + F-W43-15 amendment).
Step 8' closure proposal (commit `0d94ec75`): this document. **Subtotal at `0d94ec75`: 5 + 14 + 10 + 1 = 30 commits.**

Post-`0d94ec75` self-amendment commits (`bea79129` and later) patch this proposal document in response to closure-board review findings (F-W43-16/17/18 …). Those commits raise the live `git rev-list --count 19f7bc5d..HEAD` figure but do not change the delivered W4.3 work surface; the 30-commit subtotal anchored at `0d94ec75` is the load-bearing count for the closure board.

Spike: `bench-spike/w43-sort-merge` HEAD `fadc2700` preserved unmerged per `feedback_perf_bench_spike_first.md`.
