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
| **D6** | **Counter semantics.** | W4.1 enables WCOJ dispatch on multi-recursive bodies. Counter increments per `(rule, iteration, variant)`. Tests assert `counter > 0` (or `>= 2` if the fixture provably exercises both seeding + at least one iteration). Don't lock exact counts — recursive iteration termination depends on convergence and the test should not be brittle to future WCOJ-internal changes. |
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
