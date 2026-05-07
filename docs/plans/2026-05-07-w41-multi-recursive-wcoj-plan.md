# W4.1 — Multi-Recursive WCOJ (≥ 2 in-SCC body Scans)

**Closes W4.1 only.** No W2.5 default flip. No W3.3 resurrection.
No performance work. No new env knobs. No push, no tag, no
board update, no DONE marking without explicit user approval.
Plan-first; no implementation until iteration 1 is approved by
the user.

**Plan iteration:** 1 (read-only reconnaissance complete; this
plan is the locked submission).
**Base:** `main` at `610406ae` (post-W3.2 closure commit).
**Worktree:** `.worktrees/w41-multi-recursive-wcoj`.
**Branch:** `feat/w41-multi-recursive-wcoj`.
**Closure board:** `docs/v065-closure-board.md:96` (W4.1 row,
OPEN).

## Acceptance Line (locked from board)

> Cert: multi-recursive triangle + multi-recursive 4-cycle
> dispatch on each iteration's variant; row-set parity vs
> binary-join reference. Promoter gate `count <= 1` is removed;
> replaced with `count <= rule.recursive_arity` or equivalent.

W4.1 is a **promoter gate widening** + a **test surface
addition**. The runtime engine is already multi-recursive-ready
(verified by recon — see "Read-Only Surface" below). The work
is therefore narrowly scoped: one promoter gate change, one
existing test flipped/renamed, one new test added.

## Process Rule Compliance

* Process rule #1: no DONE marking under any iteration-1 outcome.
* Process rule #2: every W4.1 commit references W4.1.
* Process rule #5: no release-train references; out-of-scope
  concerns are owned by W4.2 / W5.x board items, named at the
  point of reference.
* Process rule #6: no push, no tag.

## Direction (locked, iteration 1)

| # | Decision | Locked answer |
|---|----------|---------------|
| **D1** | **Gate replacement.** | Remove the `recursive_scan_count > 1` cutoff at `crates/xlog-logic/src/promote.rs:114`. The user's "or equivalent" wording is satisfied because the triangle / 4-cycle promoter shape gates (`try_promote_triangle`, `try_promote_4cycle`) already require exactly 3 / 4 body atoms — implicitly capping the recursive-Scan count at the rule's "recursive_arity" (= number of body atoms in this shape). No new field on `Rule`. The clique gate at `promote.rs:147` (`recursive_scan_count == 0`) is **unchanged** — W3.2 explicitly excluded recursive cliques and W4.1 does not extend that scope. |
| **D2** | **Test surface delta.** | (a) Flip + rename `multirec_triangle_skips_wcoj_and_matches_binary_join` (test_wcoj_recursive_dispatch.rs:519) → `multirec_triangle_dispatches_wcoj_and_matches_binary_join`. Same fixture (`MULTIREC_TRIANGLE` at line 446 + `multirec_inputs` at line 511); same row-set-parity-vs-binary-join assertion; counter assertion flips from `== 0` to `> 0` (don't lock exact count — iteration-count-flexible per D6). (b) Add **new** test `multirec_4cycle_dispatches_wcoj_and_matches_binary_join` mirroring the triangle case for 4-cycle: new `MULTIREC_4CYCLE` program constant + `multirec_4cycle_inputs` fixture + assertion mirror. Test seam is the same file (`test_wcoj_recursive_dispatch.rs`). |
| **D3** | **No new runtime code paths.** | The variant-construction loop at `crates/xlog-runtime/src/executor/recursive.rs:460-540` already handles N variants correctly: per-(rel_id, occ, pred_name) iteration, per-variant `rewrite_scan_nth` substitution, per-variant WCOJ dispatch via `execute_wcoj_or_fallback_node`, union into `rule_delta_raw`. The rewrite helper at `crates/xlog-runtime/src/executor/rewrite.rs:283` walks `MultiWayJoin.inputs` AND `MultiWayJoin.fallback` consistently. **No code changes outside the promoter gate + test surface.** |
| **D4** | **Slice-1/2/4 backward compatibility.** | All existing tests must continue to pass: stable triangle/4-cycle in non-recursive SCC (slice 1/2 originals); stable triangle/4-cycle in recursive SCC (slice 4); linear-recursive triangle/4-cycle (slice 4 single-rec). Counter semantics on those tests are unchanged. The clique tests (W3.2) are untouched. |
| **D5** | **Correctness verification.** | Row-set parity vs binary-join reference. For each new / flipped test, the comparison fixture is `RuntimeConfig::default().with_wcoj_*_dispatch(Some(false))` (gate-off binary-join run). The WCOJ run (`Some(true)`) must produce the same row set + counter > 0. |
| **D6** | **Counter semantics.** | W4.1 enables WCOJ dispatch on multi-recursive bodies. Counter increments per `(rule, iteration, variant)`. Tests assert `counter > 0` (or `>= 2` if the fixture provably exercises both seeding + at least one iteration). Don't lock exact counts — recursive iteration termination depends on convergence and the test should not be brittle to subsequent WCOJ-internal changes. |
| **D7** | **Acceptance gates.** | All test-pass criteria (locked): (1) flipped `multirec_triangle_dispatches_wcoj_and_matches_binary_join` PASS; (2) new `multirec_4cycle_dispatches_wcoj_and_matches_binary_join` PASS; (3) all existing slice-1/2/4 tests PASS; (4) zero workspace warnings on touched files; (5) `cargo fmt --check --all` clean; (6) `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` exit 0 with workspace pass-count delta of **+1** (new 4-cycle test; rename keeps existing test count); (7) `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1 (CUDA cert suite unaffected — no `.cu` changes). |
| **D8** | **Process locks.** | No W2.5 default flip. No W3.3 resurrection — feat/w33-histogram-block-scheduling stays unmerged at `f1142b3e`. No performance work — D7 has no ratio gate. No push, no tag, no board edit, no DONE marking. No env-knob additions. No `RuntimeConfig` field additions. |

## Read-Only Surface (recon results, no edits in this plan)

**Promoter gate** (`crates/xlog-logic/src/promote.rs`):
* Line 114: `if recursive_scan_count(&rule.body, &head_rel_set) > 1 { continue; }` — the W4.1-target gate.
* Line 147: `if recursive_scan_count(&rule.body, &head_rel_set) == 0 { ... try_promote_clique_k ... }` — clique gate, unchanged (W3.2 scope).
* Line 176: `fn recursive_scan_count` — counter helper, walks all RIR variants including `MultiWayJoin.inputs`. Idempotent on already-promoted bodies.

**Variant construction** (`crates/xlog-runtime/src/executor/recursive.rs`):
* Line 35: `fn execute_wcoj_or_fallback_node` — per-variant WCOJ-or-fallback dispatch entry.
* Line 260: `pub fn execute_recursive_scc` — semi-naive loop entry.
* Lines 460-540: variant collection + rewrite + dispatch + union (already multi-recursive-ready).

**Rewrite helpers** (`crates/xlog-runtime/src/executor/rewrite.rs`):
* Line 243: `fn collect_scan_rels` — walks `MultiWayJoin.inputs` (and only inputs, per the line-273 promoter invariant).
* Line 283: `fn rewrite_scan_nth` — public entry; substitutes the `nth` occurrence of `target` with `replacement`.
* Line 295: `fn rewrite_scan_nth_impl` — recursive walker; **handles `MultiWayJoin` by rewriting both `inputs` and `fallback` consistently**.

**Existing tests** (`crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs`):
* Line 446: `MULTIREC_TRIANGLE` constant — `tri(X,Y,Z) :- r1(X,Y), r2(Y,Z), r3(X,Z)` with `r1` and `r2` recursive (count = 2).
* Line 511: `multirec_inputs` fixture.
* Line 519: `multirec_triangle_skips_wcoj_and_matches_binary_join` — asserts the **current** (pre-W4.1) behavior. **W4.1 flips this test.**
* Line 608: `linear_recursive_triangle_dispatches_on_seeding_and_per_variant` — single-rec triangle, count == 1, **unaffected by W4.1**.
* Line 696: `linear_recursive_4cycle_dispatches_on_seeding_and_per_variant` — single-rec 4-cycle, count == 1, **unaffected by W4.1**.
* **No multi-recursive 4-cycle test exists today** — W4.1 adds one.

## Step-by-Step Execution Plan

### Step 1 — Plan iteration 1 commit (this commit)

This file (`docs/plans/2026-05-07-w41-multi-recursive-wcoj-plan.md`) is committed as the first commit on `feat/w41-multi-recursive-wcoj`. Subject mentions W4.1. **No code change.** Awaits user approval.

### Step 2 — Promoter gate widening

File: `crates/xlog-logic/src/promote.rs`.

Change at line 114:

```rust
// BEFORE (slice 4 gate — W4.1 target):
if recursive_scan_count(&rule.body, &head_rel_set) > 1 {
    continue;
}

// AFTER (W4.1 — gate removed; promoter shape gates are
// sufficient because triangle / 4-cycle require exactly 3 / 4
// atoms and the recursive-Scan count cannot exceed those caps):
// (deletion — line removed)
```

Update the surrounding comment block (lines 111-116) to
reflect the new behavior:

```rust
// Slice 4 + W4.1 gate: the promoter shape gates (try_promote_*)
// require exactly 3 / 4 / k atoms, implicitly bounding the
// recursive-Scan count at the rule's atom count. Multi-
// recursive bodies (count >= 2) are accepted; the runtime's
// per-variant rewrite + dispatch loop in
// `execute_recursive_scc` handles N variants correctly.
```

Commit subject: `fix(w41): remove multi-recursive promoter gate (count > 1 cutoff)`.

### Step 3 — Flip + rename existing multirec triangle test

File: `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs`.

Change at lines 519-558:

* Rename function: `multirec_triangle_skips_wcoj_and_matches_binary_join`
  → `multirec_triangle_dispatches_wcoj_and_matches_binary_join`.
* Update the doc comment to reflect W4.1 behavior.
* Counter assertion: `assert_eq!(attempted.wcoj_triangle_dispatch_count(), 0, ...)` →
  `assert!(attempted.wcoj_triangle_dispatch_count() > 0, ...)`. Don't lock exact count
  per D6.
* Row-set assertion unchanged.
* Reference run (gate-off) unchanged.

Commit subject: `test(w41): flip multirec triangle to assert WCOJ dispatch + parity`.

### Step 4 — Add multirec 4-cycle test

File: `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs`.

Append after the linear-rec 4-cycle test (after line 741):

* New constant `MULTIREC_4CYCLE` mirroring `MULTIREC_TRIANGLE`'s
  shape: 4-cycle body with ≥ 2 recursive scans (e.g. `r1, r2`
  recursive; `r3, r4` extensional).
* New fixture `multirec_4cycle_inputs` mirroring `multirec_inputs`.
* New test `multirec_4cycle_dispatches_wcoj_and_matches_binary_join`
  mirroring the flipped triangle test:
  - Reference run: gate-off, counter == 0, capture row set.
  - Dispatched run: gate-on, counter > 0, row-set parity.

Commit subject: `test(w41): add multirec 4-cycle dispatch + parity cert`.

### Step 5 — Workspace gate

* `cargo fmt --check --all` — clean (exit 0).
* `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` —
  exit 0 with workspace pass-count delta of **+1** (new 4-cycle
  test; the renamed triangle test is the same cell).
* Zero warnings on `crates/xlog-logic/src/promote.rs` and
  `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs`.

Commit subject (if any cleanup): `chore(w41): workspace gate green`.

### Step 6 — CUDA cert suite

* `cargo test -p xlog-cuda-tests --test certification_suite --release` — 1/1.
  W4.1 makes no `.cu` changes; the suite must remain unaffected.

No new commit unless cert-side fixture wiring is needed (none anticipated).

### Step 7 — Closure proposal (no DONE marking)

After Steps 2-6 are committed:

* Plan-iteration commits (this plan + steps 2-5 commits) on
  `feat/w41-multi-recursive-wcoj`.
* No `docs/v065-closure-board.md` edit.
* No FF-merge.
* No iteration 9 / advance.

Process rule #1 / #6 hold: agent waits for explicit user
"mark W4.1 DONE" before applying the board update + FF-merge.

## Acceptance Grid

| Layer | # tests | Source | What it locks |
|-------|---------|--------|---------------|
| Multi-recursive triangle WCOJ dispatch + parity | 1 | `test_wcoj_recursive_dispatch.rs` (flipped from line 519) | counter > 0 across iterations + row-set parity vs binary-join. |
| Multi-recursive 4-cycle WCOJ dispatch + parity | 1 | `test_wcoj_recursive_dispatch.rs` (new test, append) | Same shape as triangle case at 4-cycle arity. |
| Linear-recursive triangle (slice 4 single-rec) | 1 | `test_wcoj_recursive_dispatch.rs:608` | **Unchanged** — must continue passing. |
| Linear-recursive 4-cycle (slice 4 single-rec) | 1 | `test_wcoj_recursive_dispatch.rs:696` | **Unchanged** — must continue passing. |
| Stable triangle / 4-cycle in recursive SCC | 2 | `test_wcoj_recursive_dispatch.rs:315, 391` | **Unchanged** — count = 0 path. |
| Adaptive in recursive SCC on superhub | 1 | `test_wcoj_recursive_dispatch.rs:486` | **Unchanged** — adaptive classifier path. |
| All slice-1 / slice-2 / W3.2 / W3.3 (rejected) tests | many | various | **Unchanged** — D4 backward compat. |
| **Workspace pass-count delta** | **+1** | — | **One new test; rename keeps the flipped cell at 1 → 1.** |

Total acceptance: **2 W4.1-locked tests** (1 flipped + 1 new),
zero regressions on the unaffected backward-compat tests, zero
workspace warnings, zero `cargo fmt` violations, CUDA cert suite
green.

## Source-of-Truth References

* W4.1 board entry: `docs/v065-closure-board.md:96`.
* Promoter gate: `crates/xlog-logic/src/promote.rs:114` (W4.1
  target) and `:147` (clique gate, unchanged).
* Promoter helper: `crates/xlog-logic/src/promote.rs:176`
  (recursive_scan_count).
* Per-variant dispatch: `crates/xlog-runtime/src/executor/recursive.rs:35`
  (`execute_wcoj_or_fallback_node`).
* Variant construction loop: `crates/xlog-runtime/src/executor/recursive.rs:460-540`.
* Rewrite helper: `crates/xlog-runtime/src/executor/rewrite.rs:283`
  (`rewrite_scan_nth`) + `:243` (`collect_scan_rels`).
* Existing fixture: `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs:446`
  (`MULTIREC_TRIANGLE`) + `:511` (`multirec_inputs`).
* Existing flipped test target:
  `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs:519`.
* Linear-rec triangle / 4-cycle tests (D4 backward-compat):
  `:608`, `:696`.

## Risk Register (informational)

| Risk | Mitigation |
|------|------------|
| `try_promote_triangle` / `try_promote_4cycle` reject multi-recursive bodies based on which atoms are recursive (independent of count). | Promoter shape gates verify atom count + variable matrix. They do not check which Scans are recursive. Removing the count gate exposes multi-recursive bodies to the same shape match used for stable bodies. Step 3 / Step 4 tests directly verify this — if the promoter rejects the multi-rec body for some other reason, the counter assertion fails and surfaces the issue. |
| Variant-loop performance regresses on multi-recursive bodies (N variants × N seeds ≈ N² work). | For triangle, N ≤ 3 (max 3 recursive Scans). For 4-cycle, N ≤ 4. The per-variant dispatch is in the existing slice-4 hot path and was designed for this; no perf gate in W4.1 (D7 has no ratio criterion). |
| Recursive iteration termination differs between WCOJ + binary-join paths (e.g., one converges in fewer iterations). | The row-set-parity assertion catches divergence on the final fixed point. The counter assertion is `> 0`, not `== N`, so iteration-count differences don't break the test. Both paths produce set-semantic outputs (dedup applied), so they must converge to the same row set. |
| `MULTIREC_4CYCLE` fixture design is non-obvious (requires care to avoid inadvertent triangle promotion). | The fixture must use a 4-atom body explicitly (4 distinct extensional / recursive predicates). The existing `MULTIREC_TRIANGLE` (3-atom body) is the design template. New constant + new fixture map are net additions; no rewrite of existing fixtures. |
| Test rename breaks something (e.g., `cargo test` filter expecting old name). | Grep across the workspace for the old name; if any other reference exists, update or remove. The old name is presumed test-file-local. |

## Plan-Approval Gate

This plan is **iteration 1 draft**. The agent does NOT advance
to Step 2 (promoter gate edit) until the user explicitly
approves this plan (via "Iteration 1 is approved" or equivalent).

If the user requests revisions, this plan is amended in place
(F-style finding entries appended below) and re-submitted.
Iteration 1 does not advance until the plan itself is locked.

The acceptance criteria for advancing past this gate:
* User explicit approval of the plan iteration.
* No unresolved blocking findings.
* Worktree clean at base `610406ae` + this plan commit.

## Iteration-2 Amendment Log

| Finding | Location | Resolution |
|---------|----------|-----------|
| **F-W41-1: W2.3 stats trace cert at `crates/xlog-runtime/tests/test_w23_recursive_stats.rs:691` (`multi_recursive_triangle_per_iteration_update_does_not_promote`) asserts the W4.1-target precondition (counter == 0) and is gated under the `recursive-stats-trace` Cargo feature.** Iteration 1 only updated `test_wcoj_recursive_dispatch.rs`. After the promoter gate is removed, `cargo test -p xlog-runtime --release --features recursive-stats-trace --test test_w23_recursive_stats` will fail because the Part D cert's counter assertion locks the pre-W4.1 contract. | `crates/xlog-runtime/tests/test_w23_recursive_stats.rs:686-725` (Part D — Multi-recursive bodies untouched). | Iteration 2 adds Step 5 = "W2.3 Part D cert update" before the workspace gate. The test gets renamed (`multi_recursive_triangle_per_iteration_update_does_not_promote` → `multi_recursive_triangle_per_iteration_update_dispatches_wcoj`), the docstring rewritten to reflect W4.1, the counter assertion flipped to `>= 2` (per F-W41-3), and the W2.3 trace assertion preserved verbatim (W2.3 trace is predicate-level and remains valid). The workspace gate (renumbered Step 7) adds an explicit `cargo test -p xlog-runtime --release --features recursive-stats-trace` invocation alongside the default-features run. |
| **F-W41-2: Stale contract docs at four locations claim multi-recursive WCOJ is skipped or not promoted, contradicting W4.1's outcome.** | `crates/xlog-logic/src/promote.rs:90-94` (outer `pub fn promote_multiway` doc comment; "≥ 2 recursive Scans are left as binary-join trees"); `crates/xlog-logic/src/promote.rs:111-116` (inline gate comment; "Slice 4 gate: skip multi-recursive bodies"); `crates/xlog-runtime/src/executor/recursive.rs:16-29` (`execute_wcoj_or_fallback_node` header; "Multi-recursive bodies never reach a `MultiWayJoin` here because the slice 4 promoter gate skips them"); `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs:13-16` (file header; "A **multi-recursive** triangle (≥ 2 in-SCC body Scans) is NOT promoted ..."). | Iteration 2 adds Step 6 = "Stale contract doc-scrub" between the test changes and the workspace gate. All four locations are rewritten to reflect the W4.1 outcome: the promoter admits multi-recursive bodies up to the shape's atom count, the recursive engine's `execute_wcoj_or_fallback_node` does dispatch on multi-recursive bodies, and the test header documents the new contract. The doc-scrub commit subject: `docs(w41): scrub stale "multi-recursive skip" contract notes`. |
| **F-W41-3: Counter assertion `> 0` is too weak for the board phrase "dispatch on each iteration's variant".** The MULTIREC_TRIANGLE fixture has 2 distinct recursive predicates (`r1`, `r2`), each with non-empty initial deltas (`r1_init`, `r2_init`). On the seeding pass alone the fixture should produce ≥ 2 dispatches (per-rule, per-variant). Subsequent iterations may fire more. Locking only `> 0` lets a partially-broken implementation pass with one dispatch (e.g. only seeding fires, no per-variant work). The same logic applies to the new MULTIREC_4CYCLE fixture: design it with 2 distinct recursive predicates so the lower bound `>= 2` holds. **Additionally**: the test must assert `reference_rows.len() > 0` (binary-join reference produces non-empty output). Without this, an implementation bug that produces an empty output would satisfy `attempted_rows == reference_rows` trivially. | `docs/plans/2026-05-07-w41-multi-recursive-wcoj-plan.md` (D6 + Step 3 + Step 4 sections). | Iteration 2 amends D6 to require `>= 2` instead of `> 0` and adds non-empty-row assertion to the contract. Step 3 (flipped triangle test) and Step 4 (new 4-cycle test) descriptions are updated to match: counter `>= 2` + reference rows non-empty + row-set parity. The MULTIREC_4CYCLE fixture is explicitly designed with 2 distinct recursive predicates (e.g., `r1, r2` recursive; `r3, r4` extensional) to satisfy the `>= 2` lower bound. |
| **F-W41-4: D3 overclaim that `rewrite_scan_nth` handles all multi-recursive variants correctly.** Inspection of `crates/xlog-runtime/src/executor/rewrite.rs:477-504` (`MultiWayJoin` arm of `rewrite_scan_nth_impl`) shows: the loop walks `inputs` and the subsequent `fallback` walk SHARES the same `&mut remaining` counter, AND the Scan case (line 303-311) early-returns on match without decrementing. Net effect: for **duplicate same-predicate occurrences** (e.g. self-recursive bodies like `p(X,Y) :- p(X,Z), p(Z,Y)` with two `p` Scans), calling `rewrite_scan_nth(body, p, occ=0, delta_p)` matches the 1st occurrence in inputs (replaces, returns), then continues to the next input — if that's also `Scan(p)`, remaining is still 0, so it replaces AGAIN. Same issue across the inputs→fallback boundary. The variant the executor dispatches on for occ=0 ends up with multiple slots replaced when only one was intended. | `crates/xlog-runtime/src/executor/rewrite.rs:303-311` (Scan case) + `:477-504` (MultiWayJoin arm). | Iteration 2 **scopes W4.1 explicitly to distinct recursive predicates** (no duplicate same-predicate occurrences in `MultiWayJoin.inputs`). D3 is rewritten to remove the broad "runtime multi-recursive-ready" claim and replace it with the narrower "for bodies whose recursive Scans target **distinct** predicates, the existing variant-construction loop is correct." MULTIREC_TRIANGLE and the new MULTIREC_4CYCLE fixtures both use distinct recursive predicates. The duplicate-occurrence rewrite bug is recorded as out-of-scope for W4.1 and named at the point of reference (subsequent self-recursive body work will need a `rewrite_scan_nth` fix + regression test). |

### Iteration-2 Net Plan Changes

* **D3 amended**: removes "runtime multi-recursive-ready" broad claim; scopes W4.1 to distinct recursive predicates only.
* **D6 amended**: counter assertion `>= 2` (was `> 0`); adds non-empty-row assertion to the contract.
* **D7 amended**: workspace gate adds the `recursive-stats-trace` feature run as a separate test invocation.
* **Step 3 amended**: counter assertion `>= 2`, non-empty reference rows asserted, distinct-recursive-predicate scope reaffirmed.
* **Step 4 amended**: MULTIREC_4CYCLE fixture explicitly designed with 2 distinct recursive predicates; counter `>= 2`, non-empty reference rows asserted.
* **Step 5 NEW (was workspace gate, now renumbered to Step 7)**: W2.3 Part D cert update — flip `multi_recursive_triangle_per_iteration_update_does_not_promote` (test_w23_recursive_stats.rs:691) to `_dispatches_wcoj`; counter `>= 2`; W2.3 trace assertion preserved; commit subject `test(w41): flip W2.3 Part D cert to assert multi-recursive WCOJ dispatch`.
* **Step 6 NEW (between tests and workspace gate)**: Stale contract doc-scrub at 4 locations (promote.rs:90-94, promote.rs:111-116, recursive.rs:16-29, test_wcoj_recursive_dispatch.rs:13-16). Commit subject `docs(w41): scrub stale "multi-recursive skip" contract notes`.
* **Step 7 amended** (workspace gate): adds `cargo test -p xlog-runtime --release --features recursive-stats-trace --test test_w23_recursive_stats` invocation alongside the default-features workspace run.
* **Acceptance grid amended**: total acceptance is now **3 W4.1 cells** (1 flipped triangle + 1 new 4-cycle + 1 flipped W2.3 Part D); workspace pass-count delta remains **+1** (only the new 4-cycle test adds a cell; the two flipped tests are renames keeping cell count constant).
* **Source-of-truth references amended**: adds `crates/xlog-runtime/tests/test_w23_recursive_stats.rs:691` (Part D cert) + `crates/xlog-runtime/src/executor/rewrite.rs:303-311 + :477-504` (rewrite-scope citation).
* **Risk register amended**: adds explicit risk row "duplicate-recursive-predicate bodies (self-recursion) are out of scope; if a fixture inadvertently uses a self-join, the rewrite walker mis-replaces."

### Iteration-2 Step Structure (renumbered)

1. ✅ Plan iteration 1 commit (commit `07ea1df0`).
2. (next) Plan iteration 2 amendment commit — this section.
3. Promoter gate widening (was Step 2 in iter-1).
4. Flip + rename triangle test in `test_wcoj_recursive_dispatch.rs` (was Step 3; updated assertions per F-W41-3).
5. Add multirec 4-cycle test in `test_wcoj_recursive_dispatch.rs` (was Step 4; updated assertions + distinct-predicate fixture per F-W41-3 + F-W41-4).
6. **NEW**: Flip W2.3 Part D cert in `test_w23_recursive_stats.rs:691` per F-W41-1.
7. **NEW**: Stale contract doc-scrub at 4 locations per F-W41-2.
8. Workspace gate (was Step 5; expanded with `recursive-stats-trace` feature run per F-W41-1).
9. CUDA cert suite (was Step 6; unchanged).
10. Closure proposal (was Step 7; unchanged).

Iteration 2 supersedes iteration 1's Step 2-7 numbering. The iteration-1 "Step-by-Step Execution Plan" section text (above) describes the original 7-step structure; the renumbered 10-step structure in this amendment-log section is canonical for iteration 2.

### Plan-Approval Gate (Iteration 2)

This iteration-2 amendment is **draft**. The agent does NOT advance to the renumbered Step 3 (promoter gate edit) until the user explicitly approves iteration 2 (via "Iteration 2 is approved" or equivalent). If the user requests further revisions, this amendment is amended in place (additional F-W41-N entries appended below) and re-submitted.

## Iteration-3 Amendment Log

Iteration 2 was **not approved**. The user's review surfaced the
project-relevant paper "Scaling Worst-Case Optimal Datalog to GPUs"
([arXiv:2604.20073](https://arxiv.org/abs/2604.20073)), and the
paper-alignment audit at
`docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md`
(commit `134884fc` on `feat/w3-paper-alignment-audit`) re-grounded
W4.1's direction. An earlier iteration-3 attempt (uncommitted,
reverted) used a duplicate-recursive-occurrence guard approach; the
audit verdict shows that approach contradicts paper claim **P1**
(semi-naïve semantics over body-clause OCCURRENCES, not just
predicate names — same-predicate self-recursive bodies must be
admitted, not rejected).

This iteration-3 amendment supersedes both iteration 1 and
iteration 2 in the live D-table + Step plan + Acceptance Grid +
Risk Register. Iteration 1 / 2 amendment logs are preserved above
for traceability.

### Iteration-3 Findings (paper-grounded)

| Finding | Resolution |
|---------|-----------|
| **F-W41-5: Iteration 1 + iteration 2 directions both contradict paper claim P1.** Iteration 1 removed the count > 1 gate without examining occurrence-identity correctness in `rewrite_scan_nth`. Iteration 2 added a duplicate-guard that REJECTS same-predicate self-recursive bodies — paper P1 says such bodies must be ADMITTED (every body-clause occurrence is a valid Δ-binding site). Reference: audit P1 (paper §2 lines ~169-178); audit verdict at `docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md` §"W4.1 Implications". | Iteration 3 adopts the **paper-grounded direction**: admit multi-recursive bodies including same-predicate self-recursion, AND fix `rewrite_scan_nth` so occurrence identity is preserved across `MultiWayJoin.inputs` and `MultiWayJoin.fallback`. The duplicate-guard scope-out is rejected. |
| **F-W41-6: `rewrite_scan_nth` bug requires a fix, not a scope-out.** Inspection of `crates/xlog-runtime/src/executor/rewrite.rs:303-311` (Scan case) + `:477-504` (MultiWayJoin arm) confirms: the Scan case early-returns on match without decrementing `remaining`, AND the MultiWayJoin arm shares `&mut remaining` across the inputs walk and the subsequent fallback walk. Net effect: occ=N can replace MORE than one slot when `target` appears multiple times in `MultiWayJoin.inputs`, AND the fallback walk's counter is contaminated by the inputs walk's consumption. | Iteration 3 adds Step (renumbered) for the rewrite fix. The fix uses (a) a sentinel value (e.g. `usize::MAX`) in the `remaining` counter post-replacement so subsequent matches in the same walk don't re-replace, AND (b) **separate counter copies** for the inputs walk and fallback walk in the `MultiWayJoin` arm so both copies independently target the N-th occurrence. The fix is co-located with a `#[cfg(test)]` regression test in `rewrite.rs` covering: same-predicate triple in MultiWayJoin inputs (target appears 3 times, occ ∈ {0, 1, 2}); inputs+fallback symmetry (occ=N replaces N-th in inputs AND N-th in fallback consistently). |
| **F-W41-7: Test surface must cover both regimes.** Iteration 1/2 only covered distinct-recursive-predicate fixtures (MULTIREC_TRIANGLE has `r1`, `r2` distinct). Per audit P1, same-predicate self-recursive bodies (e.g. transitive closure `tc(X,Y) :- tc(X,Z), tc(Z,Y)`-style bodies) must dispatch + match binary-join reference. | Iteration 3 amends D2 to add a **same-predicate self-recursive cert** (positive cert, not negative): new fixture `selfrec_triangle_dispatches_wcoj_and_matches_binary_join` where `tri(X,Y,Z) :- p(X,Y), p(Y,Z), q(X,Z)` with `p` recursive (appearing 2 times). After the rewrite fix, this body promotes; per-variant rewrite picks the correct occurrence; counter `>= 2` (seeding + ≥ 1 iteration); row-set parity vs binary-join. |
| **F-W41-8: Delta-outermost placement (paper P4) requires verification or explicit documentation.** Paper §"lines ~797-800" says SRDatalog "strictly places the delta relation outermost." The xlog WCOJ dispatcher currently chooses variable ordering via the W2.1 cost model. W4.1's correctness does NOT depend on delta-outermost placement (row-set parity vs binary-join is sufficient correctness), but the audit recommends verifying or documenting alignment. | Iteration 3 adds a Step for **P4 verification**: read the W2.1 cost model + variable-ordering logic at `crates/xlog-logic/src/optimizer/cost_model.rs` (or wherever the variable order is chosen for the recursive variant body); confirm whether delta is outermost in the chosen order, OR document the divergence with rationale (correctness-first; perf gap accepted in W4.1). This step is documentation-only when the implementation aligns; if a divergence is found, it's recorded as out-of-scope of W4.1 (named at point of reference) and does not block W4.1 closure. |

### Iteration-3 Direction (locked, supersedes iteration 1 + 2 D-table)

| # | Decision | Locked answer |
|---|----------|---------------|
| **D1** | **Gate replacement.** | Remove the `recursive_scan_count > 1` cutoff at `crates/xlog-logic/src/promote.rs:114` (no replacement guard). Multi-recursive bodies, including same-predicate self-recursive (per paper P1 — semi-naïve evaluation reasons over body-clause OCCURRENCES), are admitted. The triangle / 4-cycle promoter shape gates already cap atom count at 3 / 4, satisfying the user's "or equivalent" framing. The clique gate at `promote.rs:147` (`recursive_scan_count == 0`) is **unchanged** — W3.2 excluded recursive cliques. |
| **D2** | **Test surface delta.** | (a) Flip + rename `multirec_triangle_skips_wcoj_and_matches_binary_join` (test_wcoj_recursive_dispatch.rs:519) → `multirec_triangle_dispatches_wcoj_and_matches_binary_join`; counter `>= 2`; non-empty reference rows; row-set parity. (b) Add new `multirec_4cycle_dispatches_wcoj_and_matches_binary_join` (distinct recursive predicates `r1`, `r2`); counter `>= 2`; non-empty rows; parity. (c) Add new **same-predicate self-recursive POSITIVE cert** `selfrec_triangle_dispatches_wcoj_and_matches_binary_join` (per paper P1; replaces iteration-2's negative cert): `tri(X,Y,Z) :- p(X,Y), p(Y,Z), q(X,Z)` with `p` recursive. Counter `>= 2`; non-empty rows; row-set parity. (d) Flip + rename `multi_recursive_triangle_per_iteration_update_does_not_promote` (test_w23_recursive_stats.rs:691) → `_dispatches_wcoj`; counter `>= 2`; W2.3 trace assertion preserved. (e) Add `#[cfg(test)] mod` regression test in `rewrite.rs` covering the rewrite fix (F-W41-6). |
| **D3** | **Code surface (paper-grounded).** | W4.1 modifies (1) `crates/xlog-logic/src/promote.rs:114` — remove the gate (no replacement helper). (2) `crates/xlog-runtime/src/executor/rewrite.rs:303-311` (Scan case) + `:477-504` (MultiWayJoin arm) — fix `rewrite_scan_nth_impl` so occurrence identity is preserved (sentinel value post-replacement; separate counter copies for inputs vs fallback walks in MultiWayJoin). (3) Doc-scrub at 4 locations (per F-W41-2). No `RuntimeConfig` field additions. No env-knob additions. No `Rule` field additions. |
| **D4** | **Slice-1/2/4 backward compatibility.** | All existing tests must continue to pass: stable triangle/4-cycle in non-recursive SCC (slice 1/2 originals); stable triangle/4-cycle in recursive SCC (slice 4); linear-recursive triangle/4-cycle (slice 4 single-rec — `rewrite_scan_nth` fix preserves single-occurrence semantics). Counter semantics on those tests unchanged. The clique tests (W3.2) are untouched. |
| **D5** | **Correctness verification.** | Row-set parity vs binary-join reference (gate-off) for each W4.1 cert. The rewrite-fix regression test (in `rewrite.rs`) verifies occurrence-identity preservation directly at the helper level (no GPU, no executor). |
| **D6** | **Counter semantics.** | Counter `>= 2` for all 3 dispatch certs (multirec triangle, multirec 4-cycle, selfrec triangle). All 3 fixtures have non-empty initial deltas → seeding alone produces `>= 2` dispatches per-(rule, variant). Reference rows asserted non-empty. |
| **D7** | **Acceptance gates (locked).** | (1) flipped multirec triangle PASS (counter `>= 2`); (2) new multirec 4-cycle PASS; (3) **new selfrec triangle PASS** (positive cert per P1); (4) flipped W2.3 Part D PASS under `--features recursive-stats-trace`; (5) rewrite-fix regression test PASS (in `rewrite.rs` `#[cfg(test)] mod`); (6) all existing slice-1/2/4 tests PASS; (7) zero workspace warnings on touched files; (8) `cargo fmt --check --all` clean; (9) `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` exit 0 with workspace pass-count delta of **+3** (multirec_4cycle + selfrec_triangle + rewrite-fix regression test); the two flipped tests are renames keeping cell count constant; (10) `cargo test -p xlog-runtime --release --features recursive-stats-trace --test test_w23_recursive_stats` exit 0; (11) `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1. |
| **D8** | **Process locks.** | No W2.5 default flip. No W3.3 resurrection. **W3.4 / W3.5 / W3.6 stay not-started until paper-aligned W3.3 is drafted + approved (per audit recommendation).** No performance work in W4.1 — D7 has no ratio gate; iteration 3's Step for delta-outermost (P4) is documentation-only. No push, no tag, no board edit, no DONE marking. No env-knob additions. No `RuntimeConfig` field additions. |

### Iteration-3 Step Structure (canonical, supersedes iteration 1 + 2)

1. ✅ Plan iteration 1 commit (`07ea1df0`).
2. ✅ Plan iteration 2 amendment commit (`9e03747d`).
3. ✅ Plan iteration 3 amendment commit (this commit).
4. ⏳ **Promoter gate removal**: delete the `recursive_scan_count > 1` cutoff at `promote.rs:114` (no replacement guard). Update surrounding comment block. Commit subject: `fix(w41): remove multi-recursive promoter gate (paper P1 — admit occurrence-level multi-recursion)`.
5. ⏳ **`rewrite_scan_nth` fix**: modify Scan case (`rewrite.rs:303-311`) to use `usize::MAX` sentinel post-replacement; modify MultiWayJoin arm (`:477-504`) to use separate `remaining` counter copies for inputs vs fallback walks. Add `#[cfg(test)] mod` regression test covering: 3-occurrence target with occ ∈ {0, 1, 2}; inputs+fallback symmetry. Commit subject: `fix(w41): preserve occurrence identity in rewrite_scan_nth (paper P1)`.
6. ⏳ Flip + rename multirec triangle test in `test_wcoj_recursive_dispatch.rs:519`.
7. ⏳ Add new multirec 4-cycle test (distinct predicates).
8. ⏳ **NEW**: Add new selfrec_triangle POSITIVE cert (same-predicate self-recursion per P1; replaces iteration-2's negative cert).
9. ⏳ Flip W2.3 Part D cert in `test_w23_recursive_stats.rs:691`.
10. ⏳ Stale contract doc-scrub at 4 locations.
11. ⏳ **NEW**: Delta-outermost (P4) verification — read W2.1 cost-model variable-ordering logic; confirm or document divergence. Commit subject: `docs(w41): verify or document delta-outermost placement (paper P4)`.
12. ⏳ Workspace gate (default + `--features recursive-stats-trace`).
13. ⏳ CUDA cert suite 1/1.
14. ⏳ Closure proposal text-only.

### Iteration-3 Acceptance Grid (canonical)

| Layer | # tests | Source | What it locks |
|-------|---------|--------|---------------|
| Multi-recursive triangle (distinct predicates) | 1 | `test_wcoj_recursive_dispatch.rs:519` (flipped + renamed) | counter `>= 2` + non-empty rows + row-set parity. |
| Multi-recursive 4-cycle (distinct predicates) | 1 | `test_wcoj_recursive_dispatch.rs` (new test) | counter `>= 2` + non-empty rows + row-set parity. |
| **Self-recursive triangle (same-predicate; positive cert per paper P1)** | 1 | `test_wcoj_recursive_dispatch.rs` (new test) | counter `>= 2` + non-empty rows + row-set parity. **Locks paper P1 occurrence semantics in code.** |
| W2.3 Part D — multi-recursive triangle | 1 | `test_w23_recursive_stats.rs:691` (flipped + renamed) | counter `>= 2` + W2.3 trace records preserved. |
| `rewrite_scan_nth` regression test | 1 | `rewrite.rs` `#[cfg(test)] mod` | Occurrence-identity preservation: 3-occurrence target with occ ∈ {0, 1, 2}; inputs+fallback symmetry. |
| Linear-recursive triangle / 4-cycle (slice 4 single-rec) | 2 | `test_wcoj_recursive_dispatch.rs:608, :696` | **Unchanged** — count == 1 admits. |
| Stable triangle / 4-cycle in recursive SCC | 2 | `test_wcoj_recursive_dispatch.rs:315, :391` | **Unchanged.** |
| Adaptive in recursive SCC on superhub | 1 | `test_wcoj_recursive_dispatch.rs:486` | **Unchanged.** |
| All slice-1 / 2 / W3.2 / W3.3-rejected tests | many | various | **Unchanged** — D4 backward compat. |
| **Workspace pass-count delta** | **+3** | — | **Three new tests** (multirec_4cycle + selfrec_triangle + rewrite-fix regression). The two flipped tests are renames keeping cell count constant. |

Total W4.1 acceptance: **5 W4.1-locked tests** (3 dispatch certs +
1 flipped W2.3 Part D + 1 rewrite-fix regression), zero
regressions on the unaffected backward-compat tests, zero
workspace warnings, zero `cargo fmt` violations, default +
`recursive-stats-trace` workspace gate clean, CUDA cert suite
green.

### Plan-Approval Gate (Iteration 3)

This iteration-3 amendment is **draft**. The agent does NOT
advance to Step 4 (promoter gate removal) until the user
explicitly approves iteration 3 (via "Iteration 3 is approved"
or equivalent). If the user requests revisions, this amendment
is amended in place (additional F-W41-N entries appended) and
re-submitted.
