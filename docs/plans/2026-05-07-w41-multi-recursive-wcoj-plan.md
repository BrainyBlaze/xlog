# W4.1 — Multi-Recursive WCOJ (≥ 2 in-SCC body Scans)

**Closes W4.1 only.** No W2.5 default flip. No W3.3 resurrection.
No performance work. No new env knobs. No push, no tag, no
board update, no DONE marking without explicit user approval.
Plan-first; no implementation until iteration 4 is approved by
the user.

**Plan iteration:** 4 (paper-grounded, post-audit). The canonical
D-table, Step-by-Step Execution Plan, and Acceptance Grid below
reflect iteration 4's locked content; iteration-1 / 2 / 3 amendment
logs at the foot of this file are **historical / superseded** and
preserved only for traceability of how the design evolved.
**Base:** `main` at `610406ae` (post-W3.2 closure commit).
**Worktree:** `.worktrees/w41-multi-recursive-wcoj`.
**Branch:** `feat/w41-multi-recursive-wcoj`.
**Closure board:** `docs/v065-closure-board.md:96` (W4.1 row,
OPEN).
**Paper alignment:** the project-relevant paper "Scaling Worst-
Case Optimal Datalog to GPUs"
([arXiv:2604.20073](https://arxiv.org/abs/2604.20073)) — claims
P1 (semi-naïve occurrence semantics) and P4 (delta-outermost) are
load-bearing for iteration-4 D1 / D6 / Step 13. See audit at
`docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md`
on `feat/w3-paper-alignment-audit` HEAD `3470288f`.

## Acceptance Line (locked from board)

> Cert: multi-recursive triangle + multi-recursive 4-cycle
> dispatch on each iteration's variant; row-set parity vs
> binary-join reference. Promoter gate `count <= 1` is removed;
> replaced with `count <= rule.recursive_arity` or equivalent.

W4.1 is a **paper-grounded** (arXiv:2604.20073) promoter gate
removal + `rewrite_scan_nth` occurrence-identity fix + a test
surface delta covering distinct-recursive AND same-predicate
self-recursive bodies (per paper P1). Per the iteration-4
canonical D-table + 16-step plan below, the work surface is:
(1) promoter gate removal at `promote.rs:114`; (2) two-spot
`rewrite_scan_nth_impl` fix at `rewrite.rs:303-311 + :477-504`;
(3) two existing `rewrite.rs` test rewrites at `:577 + :691`
for the new symmetric semantic; (4) one flipped multirec triangle
test, one new multirec 4-cycle positive cert, one new self-
recursive triangle positive cert, one new rewrite-fix regression
test, and one flipped W2.3 Part D cert; (5) doc-scrub at 4
locations; (6) Step 13 delta-outermost (paper P4) verification —
documentation-only.

## Process Rule Compliance

* Process rule #1: no DONE marking under any iteration-4 outcome.
* Process rule #2: every W4.1 commit references W4.1.
* Process rule #5: no release-train references; out-of-scope
  concerns are owned by W4.2 / W5.x board items, named at the
  point of reference.
* Process rule #6: no push, no tag.

## Direction (locked, iteration 4)

Paper-grounded per arXiv:2604.20073 audit at
`docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md`.
Iteration-1 / 2 / 3 D-table content is preserved at the foot of
this file in the corresponding amendment-log sections; the table
below is the **only canonical D-table** and supersedes prior
iterations.

| # | Decision | Locked answer |
|---|----------|---------------|
| **D1** | **Gate removal (paper P1).** | Remove the `recursive_scan_count > 1` cutoff at `crates/xlog-logic/src/promote.rs:114` outright. **No replacement guard.** Multi-recursive bodies — including same-predicate self-recursive (per paper claim P1 — semi-naïve evaluation reasons over body-clause OCCURRENCES, not predicate names) — are admitted. The triangle / 4-cycle promoter shape gates already cap atom count at 3 / 4. The clique gate at `promote.rs:147` (`recursive_scan_count == 0`) is **unchanged** — W3.2 excluded recursive cliques. |
| **D2** | **Test surface delta.** | (a) Flip + rename `multirec_triangle_skips_wcoj_and_matches_binary_join` (test_wcoj_recursive_dispatch.rs:519) → `_dispatches_wcoj_and_matches_binary_join`; counter `>= 2`; non-empty reference rows; row-set parity. (b) New `multirec_4cycle_dispatches_wcoj_and_matches_binary_join` (distinct recursive predicates `r1`, `r2`); counter `>= 2`; non-empty rows; parity. (c) New **same-predicate self-recursive POSITIVE cert** `selfrec_triangle_dispatches_wcoj_and_matches_binary_join` (per paper P1; supersedes iteration-2's negative cert and iteration-3's under-specified cert). Concrete fixture pinned in iteration-4 amendment finding F-W41-13 below. (d) Flip + rename `multi_recursive_triangle_per_iteration_update_does_not_promote` (test_w23_recursive_stats.rs:691) → `_dispatches_wcoj`; counter `>= 2`; W2.3 trace assertion preserved verbatim. (e) New `#[cfg(test)] mod` regression test in `rewrite.rs` covering the F-W41-6 fix (sentinel post-replacement; separate counter copies for inputs vs fallback). **(f) Rewrite the existing tests at `rewrite.rs:577` (`rewrite_scan_nth_rewrites_inputs_and_fallback`) and `:691` (`rewrite_scan_nth_handles_4_inputs_and_fallback`)** because they encode the **stale** semantic where fallback is treated as a continuation of the inputs-walk counter (occ=N counts across both). The new semantic per F-W41-6 fix is **input/fallback symmetry**: occ=N replaces the N-th occurrence INDEPENDENTLY in inputs and in fallback. The rewritten tests must assert that occ=0 of a target appearing once in `inputs[0]` and once in `fallback` produces a result where BOTH copies are substituted (not just one). |
| **D3** | **Code surface (paper-grounded).** | W4.1 modifies (1) `crates/xlog-logic/src/promote.rs:114` — remove the gate; (2) `crates/xlog-runtime/src/executor/rewrite.rs:303-311` (Scan case) + `:477-504` (MultiWayJoin arm) — fix `rewrite_scan_nth_impl` so occurrence identity is preserved (sentinel value `usize::MAX` post-replacement; separate `remaining` counter copies for inputs vs fallback walks in MultiWayJoin); (3) `rewrite.rs:577` + `:691` test rewrites per F-W41-9b; (4) doc-scrub at 4 locations per F-W41-2. No `RuntimeConfig` field additions. No env-knob additions. No `Rule` field additions. |
| **D4** | **Slice-1/2/4 backward compatibility.** | All existing tests must continue to pass: stable triangle / 4-cycle in non-recursive SCC; stable triangle / 4-cycle in recursive SCC; linear-recursive triangle / 4-cycle (single-rec — the rewrite-fix preserves single-occurrence semantics by construction). The clique tests (W3.2) are untouched. The two `rewrite.rs` tests at `:577` + `:691` are NOT D4-protected — they're rewritten per F-W41-9b; iteration 4 explicitly accepts the existing-test rewrites because they encoded the buggy old semantic. |
| **D5** | **Correctness verification.** | Row-set parity vs binary-join reference (gate-off run) for each W4.1 cert. The rewrite-fix regression test verifies occurrence-identity preservation directly at the helper level (no GPU, no executor). |
| **D6** | **Counter semantics (corrected per F-W41-11).** | The seeding pass at `recursive.rs:331-347` evaluates each rule **once** on its **original** body (no variant rewrites); WCOJ counter increments by 1 per rule per seeding. Variants are constructed and dispatched in the **iteration loop** at `recursive.rs:455-540` (line 455 = `for _iteration in 0..max_iterations`; line 520 = the per-variant `execute_wcoj_or_fallback_node` call site). The first iteration with non-empty initial deltas is what produces multiple per-variant dispatches. Thus `>= 2` is satisfiable iff the fixture's initial deltas (after seeding) are non-empty AND iterating at least once. **All three W4.1 dispatch certs use fixtures whose initial deltas are guaranteed non-empty** (multirec_triangle's `r1_init` + `r2_init`, multirec_4cycle's analogous initial relations, selfrec_triangle's `p_init`); each produces seeding=1 dispatch + iter1 ≥ 1 variant dispatch = `>= 2` total per rule. Reference rows asserted non-empty so an empty-output run cannot trivially satisfy parity. |
| **D7** | **Acceptance gates (locked).** | (1) flipped multirec triangle PASS (counter `>= 2`); (2) new multirec 4-cycle PASS; (3) **new selfrec triangle PASS** (positive cert per P1; concrete fixture per F-W41-13); (4) flipped W2.3 Part D PASS under `--features recursive-stats-trace`; (5) rewrite-fix regression test PASS in `rewrite.rs` `#[cfg(test)] mod`; (6) **rewritten `rewrite_scan_nth_rewrites_inputs_and_fallback` and `rewrite_scan_nth_handles_4_inputs_and_fallback` PASS** with the new symmetric semantics; (7) all other slice-1/2/4 tests PASS; (8) zero workspace warnings on touched files; (9) `cargo fmt --check --all` clean; (10) `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` exit 0 with workspace pass-count delta of **+3** (new multirec 4-cycle + new selfrec triangle + new rewrite-fix regression cell; the two flipped recursive-dispatch tests + the two rewritten `rewrite.rs` tests are renames-or-rewrites keeping cell count constant; the W2.3 Part D flip is also a rename keeping cell count constant); (11) `cargo test -p xlog-runtime --release --features recursive-stats-trace --test test_w23_recursive_stats` exit 0; (12) `cargo test -p xlog-cuda-tests --test certification_suite --release` 1/1. |
| **D8** | **Process locks.** | No W2.5 default flip. No W3.3 resurrection. **W3.4 / W3.5 / W3.6 stay not-started until paper-aligned W3.3 is drafted + approved (per audit recommendation).** No performance work — D7 has no ratio gate; iteration 4's Step 13 (delta-outermost / paper P4) is documentation-only, citing actual files per F-W41-12. No push, no tag, no board edit, no DONE marking. No env-knob additions. No `RuntimeConfig` field additions. |

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

## Step-by-Step Execution Plan (16 steps, iteration-4 canonical)

Iteration-1 / 2 / 3 step structures are preserved at the foot of
this file in the corresponding amendment-log sections. The 12-step
sequence below is the **only canonical step plan**.

### Step 1–3 — Plan iteration commits (DONE)

* Step 1: iter-1 commit `07ea1df0` (initial plan).
* Step 2: iter-2 amendment commit `9e03747d` (F-W41-1 — F-W41-4).
* Step 3: iter-3 amendment commit `7889e9db` (F-W41-5 — F-W41-8;
  paper-grounded direction).

### Step 4 — Plan iteration 4 amendment commit (this commit)

Canonical D-table + Step plan + Acceptance Grid rewritten in place
to supersede iteration-1 / 2 / 3 stale wording. F-W41-9 — F-W41-13
logged. **No code change.** Awaits user approval before Step 5.

### Step 5 — Promoter gate removal

File: `crates/xlog-logic/src/promote.rs`.

Delete the gate at line 114 (3 lines: `if ... { continue; }`).
Rewrite the surrounding comment block (lines 111-116) per the
audit + paper P1: "the promoter shape gates already cap atom
count at 3 / 4 / k; multi-recursive bodies including same-
predicate self-recursive (per paper P1) are admitted; the
runtime's per-variant rewrite + dispatch loop handles N variants
after the F-W41-6 fix at `rewrite.rs:303-311 + :477-504`."

Commit subject: `fix(w41): remove multi-recursive promoter gate (paper P1)`.

### Step 6 — `rewrite_scan_nth` occurrence-identity fix

File: `crates/xlog-runtime/src/executor/rewrite.rs`.

Modify the Scan case at lines 303-311 to use `usize::MAX` sentinel
post-replacement:

```rust
RirNode::Scan { rel } => {
    if *rel == target {
        if *remaining == 0 {
            *remaining = usize::MAX;  // sentinel: don't replace again in this walk
            return (RirNode::Scan { rel: replacement }, true);
        }
        if *remaining != usize::MAX {
            *remaining -= 1;
        }
    }
    (node.clone(), false)
}
```

Modify the MultiWayJoin arm at lines 477-504 to use **separate
counter copies** for the inputs walk vs the fallback walk:

```rust
RirNode::MultiWayJoin { inputs, slot_vars, output_columns, fallback, var_order } => {
    let starting_remaining = *remaining;
    let mut inputs_remaining = starting_remaining;
    let mut new_inputs = Vec::with_capacity(inputs.len());
    let mut any_replaced = false;
    for inp in inputs {
        let (new_inp, replaced) = Self::rewrite_scan_nth_impl(
            inp, target, &mut inputs_remaining, replacement
        );
        new_inputs.push(new_inp);
        any_replaced |= replaced;
    }
    let mut fallback_remaining = starting_remaining;
    let (new_fallback, fallback_replaced) = Self::rewrite_scan_nth_impl(
        fallback, target, &mut fallback_remaining, replacement
    );
    *remaining = inputs_remaining;  // outer caller sees inputs-walk consumption
    (
        RirNode::MultiWayJoin { inputs: new_inputs, slot_vars: slot_vars.clone(),
            output_columns: output_columns.clone(),
            fallback: Box::new(new_fallback), var_order: var_order.clone() },
        any_replaced || fallback_replaced,
    )
}
```

Add a new `#[cfg(test)] mod` regression in `rewrite.rs` (next to
the existing `multiway_walker_tests` module) covering: (i) target
appearing 3 times in `MultiWayJoin.inputs`, occ ∈ {0, 1, 2} —
each call substitutes exactly one of the three; (ii)
inputs+fallback symmetry — when target appears in input[0] and in
the fallback's leftmost leaf, occ=0 must substitute BOTH copies.

Commit subject: `fix(w41): preserve occurrence identity in rewrite_scan_nth (paper P1)`.

### Step 7 — Rewrite the existing `rewrite.rs` tests at lines 577 and 691

Per F-W41-9b: the existing tests
`rewrite_scan_nth_rewrites_inputs_and_fallback` (line 577) and
`rewrite_scan_nth_handles_4_inputs_and_fallback` (line 691) encode
the **stale** semantic where fallback is treated as a continuation
of the inputs-walk counter. The new semantic per Step 6's fix is
**input/fallback symmetry**: occ=N targets the N-th occurrence
INDEPENDENTLY in inputs and in fallback.

Rewrite both tests:

* `rewrite_scan_nth_rewrites_inputs_and_fallback` → assert that
  `rewrite_scan_nth(node, RelId(10), 0, RelId(99))` substitutes
  BOTH the input[0] occurrence AND the fallback occurrence (was
  asserting nth=1 lands on fallback — now stale).
* `rewrite_scan_nth_handles_4_inputs_and_fallback` → assert
  `nth=0` substitutes input[3] AND the fallback occurrence;
  `nth=1` substitutes nothing (no second occurrence in this 4-rel
  fixture — was asserting nth=1 lands on fallback).

Commit subject: `test(w41): rewrite slice-1 rewrite_scan_nth tests for input/fallback symmetry`.

### Step 8 — Flip + rename multirec triangle test

File: `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs`.

Change at lines 519-558 per D2(a) above. Counter `>= 2`;
non-empty reference rows; row-set parity.

Commit subject: `test(w41): flip multirec triangle to assert WCOJ dispatch + parity`.

### Step 9 — Add multirec 4-cycle positive cert

File: `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs`.

Append after the linear-rec 4-cycle test (after line 741) per
D2(b): `MULTIREC_4CYCLE` constant with 4-atom body using **2
distinct recursive predicates** (`r1`, `r2` recursive; `r3`,
`r4` extensional). New fixture `multirec_4cycle_inputs`. Test
`multirec_4cycle_dispatches_wcoj_and_matches_binary_join` with
counter `>= 2`, non-empty reference rows, row-set parity.

Commit subject: `test(w41): add multirec 4-cycle dispatch + parity cert`.

### Step 10 — Add self-recursive triangle positive cert (paper P1 lock)

Per D2(c) + F-W41-13. Append after Step 9's 4-cycle test:

```rust
const SELFREC_TRIANGLE: &str = r#"
    pred p_init(u32, u32).
    pred q(u32, u32).
    pred p(u32, u32).
    pred tri(u32, u32, u32).
    p(X, Y) :- p_init(X, Y).
    p(X, Z) :- tri(X, Y, Z).
    tri(X, Y, Z) :- p(X, Y), p(Y, Z), q(X, Z).
"#;

fn selfrec_triangle_inputs() -> BTreeMap<&'static str, Vec<(u32, u32)>> {
    let mut m: BTreeMap<&'static str, Vec<(u32, u32)>> = BTreeMap::new();
    m.insert("p_init", vec![(1, 2), (2, 3)]);
    m.insert("q",      vec![(1, 3)]);
    m
}
```

Reference (gate-off) row set: `{(1, 2, 3)}` — non-empty (assert).
Dispatched (gate-on) counter: `>= 2` — seeding produces 1 dispatch
per rule (1 rule for `tri` → 1 seeding dispatch); iteration 1 has
both occ=0 and occ=1 of `p` rewritten and dispatched, so iter1
adds ≥ 2 variant dispatches; total ≥ 3 dispatches in practice,
locked at `>= 2`. Both same-predicate occurrence variants fire
because `p` is recursive (head SCC includes `p` and `tri`) and
initial delta_p = `{(1,2), (2,3)}` is non-empty after seeding.
Row-set parity vs binary-join reference.

Commit subject: `test(w41): add self-recursive triangle dispatch + parity cert (paper P1)`.

### Step 11 — Flip W2.3 Part D cert

File: `crates/xlog-runtime/tests/test_w23_recursive_stats.rs`.

Change at lines 686-725 per D2(d). Rename function from
`multi_recursive_triangle_per_iteration_update_does_not_promote`
to `_dispatches_wcoj`. Counter assertion flips to `>= 2`. W2.3
trace assertion preserved verbatim.

Commit subject: `test(w41): flip W2.3 Part D cert to assert multi-recursive WCOJ dispatch`.

### Step 12 — Stale contract doc-scrub (4 locations)

Per F-W41-2: rewrite outer doc at `promote.rs:90-94`; verify
inline comment at `promote.rs:111-116` reflects Step-5 outcome;
rewrite header at `recursive.rs:16-29`; rewrite file header at
`test_wcoj_recursive_dispatch.rs:13-16`.

Commit subject: `docs(w41): scrub stale "multi-recursive skip" contract notes`.

### Step 13 — Delta-outermost (paper P4) verification (F-W41-12)

**Documentation-only step.** Read `crates/xlog-logic/src/wcoj_var_ordering.rs:1`
(W2.1 `LeaderCardinalityModel`); examine the call site in the
promoter (`crates/xlog-logic/src/promote.rs` — wherever the
triangle / 4-cycle promoter constructs `var_order`) to determine
whether the leader is chosen at COMPILE time (over the original
relation cardinalities) or at variant-rewrite time (over the
delta-substituted body). Expected finding (per recon): the leader
is chosen at compile time on the original body; variant rewrites
substitute one Scan for delta but do NOT update `var_order`.
Paper P4 says "delta strictly outermost" — the xlog implementation
chooses min-cardinality at compile time and does not re-pick the
leader after rewrite. **Document this divergence** in the audit
follow-up section of the W4.1 plan: correctness (row-set parity)
is preserved; perf alignment with P4 is partial; W4.1 does NOT
add delta-outermost re-selection — out of scope for W4.1, named
at point of reference for any subsequent perf-focused W3.x or
W4.x work.

Commit subject: `docs(w41): document delta-outermost divergence vs paper P4`.

### Step 14 — Workspace gate (default + recursive-stats-trace feature)

* `cargo fmt --check --all` — clean (exit 0).
* `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` —
  exit 0 with workspace pass-count delta **+3** (new multirec_4cycle
  + new selfrec_triangle + new rewrite-fix regression).
* `cargo test -p xlog-runtime --release --features recursive-stats-trace --test test_w23_recursive_stats` —
  exit 0.
* Zero warnings on touched files (`promote.rs`, `rewrite.rs`,
  `recursive.rs`, `test_wcoj_recursive_dispatch.rs`,
  `test_w23_recursive_stats.rs`).

Commit subject (if any cleanup): `chore(w41): workspace gate green`.

### Step 15 — CUDA cert suite

* `cargo test -p xlog-cuda-tests --test certification_suite --release` — 1/1.

No new commit unless cert-side wiring is needed.

### Step 16 — Closure proposal (no DONE marking)

Plan-iteration commits + Steps 5–14 commits on
`feat/w41-multi-recursive-wcoj`. No `docs/v065-closure-board.md`
edit. No FF-merge. No advance.

## Acceptance Grid (iteration-4 canonical)

| Layer | # tests | Source | What it locks |
|-------|---------|--------|---------------|
| Multi-recursive triangle (distinct preds) | 1 | `test_wcoj_recursive_dispatch.rs:519` (flipped + renamed) | counter `>= 2`; non-empty reference rows; row-set parity. |
| Multi-recursive 4-cycle (distinct preds) | 1 | `test_wcoj_recursive_dispatch.rs` (new, append after :741) | counter `>= 2`; non-empty rows; parity. |
| **Self-recursive triangle (same-pred POSITIVE per paper P1)** | 1 | `test_wcoj_recursive_dispatch.rs` (new, F-W41-13 fixture) | counter `>= 2`; non-empty rows (`{(1,2,3)}`); parity. |
| W2.3 Part D — multi-recursive triangle | 1 | `test_w23_recursive_stats.rs:691` (flipped + renamed) | counter `>= 2`; W2.3 trace records preserved. Feature-gated under `recursive-stats-trace`. |
| `rewrite_scan_nth` regression test | 1 | `rewrite.rs` `#[cfg(test)] mod` (new) | Occurrence-identity preservation: 3-occurrence target with occ ∈ {0, 1, 2}; inputs+fallback symmetry. |
| **`rewrite_scan_nth_rewrites_inputs_and_fallback`** | 1 | `rewrite.rs:577` (rewritten per F-W41-9b) | New symmetric semantic: occ=0 substitutes BOTH input[0] AND fallback's RelId(10) leaf. |
| **`rewrite_scan_nth_handles_4_inputs_and_fallback`** | 1 | `rewrite.rs:691` (rewritten per F-W41-9b) | New symmetric semantic: occ=0 substitutes input[3] AND fallback's RelId(40) leaf. |
| Linear-recursive triangle / 4-cycle (slice 4 single-rec) | 2 | `test_wcoj_recursive_dispatch.rs:608, :696` | **Unchanged** — count == 1 admits; rewrite-fix preserves single-occurrence semantics by construction. |
| Stable triangle / 4-cycle in recursive SCC | 2 | `test_wcoj_recursive_dispatch.rs:315, 391` | **Unchanged.** |
| Adaptive in recursive SCC on superhub | 1 | `test_wcoj_recursive_dispatch.rs:486` | **Unchanged.** |
| All slice-1 / 2 / W3.2 / W3.3-rejected tests | many | various | **Unchanged** — D4 backward compat. |
| **Workspace pass-count delta** | **+3** | — | **Three new tests** (multirec_4cycle + selfrec_triangle + rewrite-fix regression). The two flipped recursive-dispatch tests + the two rewritten `rewrite.rs` tests + the W2.3 Part D flip are all renames-or-rewrites keeping cell count constant. |

Total W4.1 acceptance: **7 W4.1-locked cells** (3 dispatch certs +
1 W2.3 Part D + 1 rewrite-fix regression + 2 rewritten existing
`rewrite.rs` tests). Zero regressions on unaffected backward-
compat tests. Zero workspace warnings. Zero `cargo fmt`
violations. Default + `recursive-stats-trace` feature workspace
gate clean. CUDA cert suite green.

## Source-of-Truth References (iteration-4 canonical)

* W4.1 board entry: `docs/v065-closure-board.md:96`.
* Paper audit: `docs/evidence/2026-05-07-w3-paper-alignment-audit/README.md`
  on `feat/w3-paper-alignment-audit` HEAD `3470288f`.
* Promoter gate (Step 5 target): `crates/xlog-logic/src/promote.rs:114`.
* Promoter clique gate (unchanged): `crates/xlog-logic/src/promote.rs:147`.
* Promoter helper (existing): `crates/xlog-logic/src/promote.rs:176`
  (`recursive_scan_count`).
* Per-variant dispatch entry: `crates/xlog-runtime/src/executor/recursive.rs:35`
  (`execute_wcoj_or_fallback_node`).
* Seeding pass (D6 citation): `crates/xlog-runtime/src/executor/recursive.rs:331-347`
  (1 dispatch per rule, original body, no rewrites).
* Iteration loop variant construction (D6 + Step 6 citation):
  `crates/xlog-runtime/src/executor/recursive.rs:455` (loop entry);
  `:460-540` (per-rule variant collection + rewrite + dispatch +
  union); `:520` (per-variant `execute_wcoj_or_fallback_node`
  call).
* Rewrite helpers (Step 6 + Step 7 targets): `crates/xlog-runtime/src/executor/rewrite.rs:283`
  (`rewrite_scan_nth` public entry); `:295-505` (`rewrite_scan_nth_impl`
  recursive walker); `:303-311` (Scan case — Step 6 sentinel fix);
  `:477-504` (MultiWayJoin arm — Step 6 separate-counters fix);
  `:577` (`rewrite_scan_nth_rewrites_inputs_and_fallback` — Step 7
  rewrite); `:691` (`rewrite_scan_nth_handles_4_inputs_and_fallback`
  — Step 7 rewrite).
* Existing fixtures (Step 8 target): `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs:446`
  (`MULTIREC_TRIANGLE`) + `:511` (`multirec_inputs`); `:519`
  (`multirec_triangle_skips_wcoj_and_matches_binary_join`).
* W2.3 Part D test (Step 11 target):
  `crates/xlog-runtime/tests/test_w23_recursive_stats.rs:691`
  (`multi_recursive_triangle_per_iteration_update_does_not_promote`).
* Stale contract docs (Step 12 doc-scrub targets):
  `crates/xlog-logic/src/promote.rs:90-94`;
  `crates/xlog-logic/src/promote.rs:111-116` (handled by Step 5);
  `crates/xlog-runtime/src/executor/recursive.rs:16-29`;
  `crates/xlog-integration/tests/test_wcoj_recursive_dispatch.rs:13-16`.
* W2.1 leader cost model (Step 13 paper-P4 verification):
  `crates/xlog-logic/src/wcoj_var_ordering.rs:1`
  (`WcojVariableOrderingModel` trait + `LeaderCardinalityModel`).
* Linear-rec triangle / 4-cycle tests (D4 backward-compat):
  `test_wcoj_recursive_dispatch.rs:608, :696`.

## Risk Register (informational, iteration-4 canonical)

| Risk | Mitigation |
|------|------------|
| `try_promote_triangle` / `try_promote_4cycle` reject distinct-recursive multi-recursive bodies based on shape (independent of recursive count). | Promoter shape gates verify atom count + variable matrix; they do not check recursive predicate identity. Steps 8 / 9 / 10 directly verify shape match — if the promoter rejects for an unexpected shape reason, counter assertion fails. |
| `rewrite_scan_nth` Step-6 fix breaks linear-recursive single-occurrence semantics. | For single-occurrence bodies (count == 1), the new sentinel + separate-counters logic is equivalent to the old early-return logic (one match → one replacement; subsequent walks see no further matches because there are no further occurrences). Linear-rec tests at lines 608 and 696 are the regression cells; D4 backward-compat. |
| Variant-loop performance regresses on multi-recursive bodies. | For triangle / 4-cycle / k-clique, N variants ≤ shape-arity ≤ 4. No perf gate in W4.1; D7 has no ratio criterion. |
| Recursive iteration termination differs between WCOJ + binary-join paths. | Row-set-parity assertion catches divergence on the final fixed point. Counter assertion is a lower bound (`>= 2`), iteration-count flexible. |
| Self-recursive cert fixture (Step 10) doesn't exercise both occurrence variants. | Concrete tuples pinned in F-W41-13: `p_init = {(1,2),(2,3)}`, `q = {(1,3)}` with shape `tri(X,Y,Z) :- p(X,Y), p(Y,Z), q(X,Z)`. Initial delta_p (after seeding) = `{(1,2),(2,3)}` non-empty; iteration 1 fires both occ=0 and occ=1 of `p`. Reference `tri = {(1,2,3)}` non-empty. |
| Step 13's delta-outermost (paper P4) verification finds a divergence that warrants implementation. | W4.1 documents the divergence; does NOT implement re-pick of leader after rewrite. Out of scope for W4.1; named at point of reference for any subsequent perf-focused W3.x or W4.x work; correctness preserved (parity vs binary-join). |
| Test rename breaks workspace test filters or other tests. | Grep across the workspace for the old test names before each rename. The old names are presumed test-file-local; verify in Step 8 / 11. |

## Plan-Approval Gate (iteration-4 canonical)

This plan is **iteration 4 draft**. The agent does NOT advance to
Step 5 (promoter gate removal) until the user explicitly approves
iteration 4 (via "Iteration 4 is approved" or equivalent).

If the user requests revisions, this plan is amended in place
(additional F-W41-N entries appended below) and re-submitted.
Iteration 4 does not advance until the plan itself is locked.

The acceptance criteria for advancing past this gate:
* User explicit approval of iteration 4.
* No unresolved blocking findings.
* Worktree clean at base `610406ae` + iteration 1 / 2 / 3 / 4
  plan commits.

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

## Iteration-4 Amendment Log

Iteration 3 was **not approved**. The user's review surfaced 4
blocking findings + 1 medium against the iteration-3 amendment
log; iteration 3 is **superseded** by the canonical D-table +
Step plan + Acceptance Grid above (rewritten in place), with the
iteration-3 amendment log preserved as historical record.

### Iteration-4 Findings

| Finding | Resolution |
|---------|-----------|
| **F-W41-9 (Blocking): Live D-table + Step plan still contained iteration-1 stale text contradicting iteration 3.** Lines 39, 44, 45, 48 (and the entire `## Direction (locked, iteration 1)` + `## Step-by-Step Execution Plan` sections) said the runtime was "already multi-recursive-ready" with "no code changes outside the promoter gate + test surface" — directly contradicted by iteration 3's `rewrite_scan_nth` fix scope. Plan had two conflicting sources of truth. | Iteration 4 **rewrites the live D-table, Step-by-Step Execution Plan, and Acceptance Grid in place** to be iteration-4 canonical. The header iteration tag is bumped from 1 to 4. Iteration-1 / 2 / 3 amendment logs are preserved at the foot of this file but explicitly marked **historical / superseded** — their text exists only for traceability. |
| **F-W41-9b (Blocking; tied to F-W41-9): Step 5 added a new `rewrite_scan_nth` regression test but did not update existing tests at `rewrite.rs:577` (`rewrite_scan_nth_rewrites_inputs_and_fallback`) and `rewrite.rs:691` (`rewrite_scan_nth_handles_4_inputs_and_fallback`).** Both existing tests encode the **old** semantic where fallback is treated as a continuation of the inputs-walk counter (occ=1 lands on the fallback occurrence). The Step-6 fix changes this to input/fallback symmetry (occ=0 substitutes the N-th in inputs AND the N-th in fallback). After the fix, the old assertions are wrong by construction. | Iteration 4 adds **Step 7** = "Rewrite the existing `rewrite.rs` tests at lines 577 and 691 for the new symmetric semantics." D2(f) is added to the canonical D-table to capture this. The two rewritten tests assert that occ=0 substitutes the input occurrence AND the fallback occurrence consistently (was: occ=0 → input only; occ=1 → fallback). |
| **F-W41-11 (Blocking): Iteration-3 D6 counter rationale was wrong.** Iteration 3 said "all 3 fixtures have non-empty initial deltas → seeding alone produces `>= 2` dispatches per-(rule, variant)." That is incorrect. The seeding pass at `crates/xlog-runtime/src/executor/recursive.rs:331-347` evaluates each rule **once** on its **original** body (no variant rewrites); WCOJ counter increments by 1 per rule per seeding. Variants are constructed and dispatched only in the **iteration loop** at `recursive.rs:455` (loop entry) → `:520` (per-variant dispatch). | Iteration-4 D6 (canonical above) is rewritten to: "the first iteration with non-empty initial deltas is what produces multiple per-variant dispatches; all three W4.1 dispatch certs use fixtures whose initial deltas (after seeding) are guaranteed non-empty; each produces seeding=1 + iter1 ≥ 1 variant dispatch = `>= 2` total per rule." Source-of-Truth References cite both `recursive.rs:331-347` (seeding) and `:455 / :520` (iteration loop). |
| **F-W41-12 (Blocking): Iteration-3 Step 11 P4 verification cited a nonexistent path.** Step 11 referenced `crates/xlog-logic/src/optimizer/cost_model.rs`, which does not exist. The actual W2.1 leader cost model lives at `crates/xlog-logic/src/wcoj_var_ordering.rs:1` (`WcojVariableOrderingModel` trait + `LeaderCardinalityModel`). Additionally, Step 11 must answer a specific question: does the delta-substituted Scan become the leader (outermost) after the rewrite, or is the leader chosen at compile time on the original body (no re-pick after rewrite)? | Iteration-4 Step 13 (renumbered, canonical above) cites the correct file `wcoj_var_ordering.rs:1` and the call site in `promote.rs` where `var_order` is constructed. The expected finding (per recon): leader is chosen at compile time over original-body cardinalities; variant rewrite at `recursive.rs:503-511` substitutes one Scan for delta but does NOT update `var_order`. The xlog implementation **diverges** from paper P4 ("delta strictly outermost"). Step 13 documents this divergence; W4.1 does NOT add re-pick logic; correctness (row-set parity) is preserved; perf gap acknowledged but out of scope for W4.1. |
| **F-W41-13 (Medium): Iteration-3 selfrec_triangle cert was under-specified.** Iteration 3 named the rule shape `tri(X,Y,Z) :- p(X,Y), p(Y,Z), q(X,Z)` with `p` recursive but did NOT specify concrete input tuples. Without concrete tuples, the cert could be implemented with an empty-output fixture that trivially satisfies parity (false-pass). Both same-predicate occurrence variants must also be exercised. | Iteration-4 D2(c) + Step 10 pin the concrete fixture: `pred p_init`, `pred q`, `pred p`, `pred tri`; rules `p(X,Y) :- p_init(X,Y).`, `p(X,Z) :- tri(X,Y,Z).`, `tri(X,Y,Z) :- p(X,Y), p(Y,Z), q(X,Z).`; tuples `p_init = [(1,2), (2,3)]`, `q = [(1,3)]`. Reference `tri = {(1,2,3)}` non-empty (asserted). After seeding, initial delta_p = `{(1,2),(2,3)}` non-empty; iteration 1 fires both occ=0 and occ=1 of `p` (each produces a valid variant). Counter `>= 2` (seeding=1 + iter1 ≥ 1 variant); in practice ≥ 3. |

### Iteration-4 Net Plan Changes

* **Header**: iteration tag 1 → 4; paper alignment line added.
* **D-table**: rewritten in place to iteration-4 canonical (D1 paper P1; D2(f) for `rewrite.rs` test rewrites; D3 expanded to include rewrite-fix; D6 corrected per F-W41-11; D7 expanded to 7 cells).
* **Step plan**: rewritten to 16 numbered steps (Steps 1-3 = plan iteration commits; Step 4 = this commit; Steps 5-16 = code/test/doc/gate work). Step 7 = NEW (`rewrite.rs:577 + :691` rewrites). Step 13 = NEW (P4 verification with correct paths). Step 14-16 = workspace gate / CUDA cert / closure.
* **Acceptance Grid**: 7 W4.1-locked cells (was 5 in iteration 3); workspace pass-count delta +3.
* **Source-of-Truth References**: corrected to cite `wcoj_var_ordering.rs:1` (was nonexistent `optimizer/cost_model.rs`); added `recursive.rs:331-347` (seeding), `:455`, `:520`; added `rewrite.rs:577` + `:691` (Step 7 targets); added paper-audit reference.
* **Risk Register**: corrected entries to reference correct files; added rows for Step-13 P4 divergence and self-rec fixture-design.
* **Plan-Approval Gate**: iteration tag 1 → 4.
* **Banned-token compliance**: avoids release-train references and closure-avoidance wording per the project process rules.

### Plan-Approval Gate (Iteration 4)

This iteration-4 amendment is **draft**. The agent does NOT
advance to Step 5 (promoter gate removal at `promote.rs:114`)
until the user explicitly approves iteration 4 (via "Iteration 4
is approved" or equivalent). If the user requests revisions, this
amendment is amended in place (additional F-W41-N entries
appended) and re-submitted.
