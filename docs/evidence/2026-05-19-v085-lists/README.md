# v0.8.5 Language Completeness - LISTS Evidence

**Date:** 2026-05-19
**Branch:** `feat/v085-language-completeness`
**Worktree:** `/home/dev/projects/xlog/.worktrees/v085-language`
**Base evidence:** `docs/evidence/2026-05-18-v085-types/README.md`
**Scope:** `G085_LIST` only. Finite list normalization, core list built-ins, examples, and diagnostics.

---

## Changed Files

| File | Purpose |
|------|---------|
| `crates/xlog-logic/src/list_normalize.rs` | Adds finite list normalization to scalar list IDs and `__xlog_list_*` helper relations. |
| `crates/xlog-logic/src/compile.rs` | Runs list normalization after query/constraint desugaring and before stratification/lowering; query variable extraction now recurses through list patterns. |
| `crates/xlog-logic/src/lower.rs` | Lowers `list<T>` predicate columns to `u64` list-id storage. |
| `crates/xlog-logic/src/lib.rs` | Exposes `normalize_v085_lists` for tests and downstream compile wrappers. |
| `crates/xlog-logic/tests/test_v085_lists.rs` | Adds syntax, normalization, diagnostics, oracle, and example compile coverage. |
| `crates/xlog-logic/tests/test_v085_types.rs` | Updates prior type diagnostics now that `list<T>` has a lowering path. |
| `crates/xlog-gpu/src/logic.rs` | Stores normalized programs so list helper facts are loaded through the normal relation store; persistent stores include list helpers. |
| `crates/xlog-gpu/tests/v085_lists.rs` | Adds GPU execution coverage for finite list built-ins. |
| `examples/v085-language/lists/*.xlog` | Adds membership, transform, and cons-pattern examples. |
| `docs/language-reference.md`, `docs/architecture/language-v085.md`, `ROADMAP.md`, `CHANGELOG.md` | Records shipped `G085_LIST` support and current limits. |

---

## RED Evidence

| Command | Result |
|---------|--------|
| `cargo test -p xlog-logic --test test_v085_lists` | exit 101 before implementation; unresolved import `xlog_logic::normalize_v085_lists` |
| `cargo test -p xlog-gpu --test v085_lists` | initially failed on a nullary `out_is_list()` fixture; the fixture was corrected to use scalar-head projection because nullary-rule behavior is not part of `G085_LIST` |

---

## Implementation Summary

Accepted finite list values now normalize before stratification and lowering:

```text
list literal / list<T> column / safe cons pattern
  -> u64 list id
  -> __xlog_list_len, __xlog_list_item_<type>, __xlog_list_cons_<type>
  -> normal RIR joins, filters, projections, and runtime relation loading
```

Implemented in this node:

- `[]`, `[A, B]`, nested finite list literals, and safe `[Head | Tail]` patterns in declared `list<T>` columns;
- `list<T>` predicate columns as `u64` list-id storage;
- `is_list`, `member`, `memberchk`, `length`, `nth`;
- finite-literal `append`, `sort`, `msort`, and `list_to_set`;
- explicit typed rejection for heterogeneous lists, untyped list variables, unbounded append split generation, and reserved pair helpers.

Still gated to later nodes:

- arbitrary split generation for `append(A, B, KnownList)`;
- dynamic or externally generated list IDs without corresponding helper facts;
- compound, `term`, and `predref` list elements;
- `findall`, `maplist`, and other safe meta-predicates.

---

## Verification Commands

| Command | Result |
|---------|--------|
| `cargo test -p xlog-logic --test test_v085_lists` | exit 0; 7 passed |
| `cargo test -p xlog-gpu --test v085_lists` | exit 0; 1 passed on local CUDA device |
| `cargo test -p xlog-logic --test test_v085_types` | exit 0; 5 passed |
| `cargo test -p xlog-logic` | exit 0; full `xlog-logic` suite passed |
| `cargo check --workspace` | exit 0 |
| `rg -n "download_column\|dtoh\|htod\|__xlog_list_\|normalize_v085_lists" crates/xlog-logic/src/list_normalize.rs crates/xlog-gpu/src/logic.rs crates/xlog-gpu/tests/v085_lists.rs` | exit 0; list implementation uses helper relations; only existing constraint/result test download paths are present |
| `cargo fmt --check` | exit 0 |
| `git diff --check` | exit 0 |

---

## G085_LIST Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M085_LIST.1 syntax fixtures | at least 25 parse fixtures | PASS | `parses_25_list_syntax_fixtures` covers 20 accepted and 5 rejected syntax fixtures |
| M085_LIST.2 built-in coverage | core built-ins implemented or rejected with rationale | PASS | implemented `is_list`, `member`, `memberchk`, `length`, `nth`, finite-literal `append`, `sort`, `msort`, `list_to_set`; pair helpers reject with typed rationale |
| M085_LIST.3 GPU lowering | accepted built-ins lower to runtime/prob paths, not CPU evaluator | PASS | normalization emits helper relations consumed by existing RIR/runtime; `xlog-gpu` test executes the built-ins |
| M085_LIST.4 equivalence | at least 12 execution fixtures match oracle output exactly | PASS | `list_builtins_have_relational_oracle_fixtures` covers 12 relational fixtures; `v085_list_builtins_execute_through_gpu_relations` checks GPU outputs for member, memberchk, length, nth, cons, append, sort, msort, set, and is_list |
| M085_LIST.5 unsafe rejection | unbounded generation and cyclic-list forms fail with typed diagnostics | PASS | unbounded append split, untyped member/nth variables, heterogeneous lists, and pair helpers reject; cyclic list syntax is not expressible in the grammar |
| M085_LIST.6 no host hot path | no hidden DTOH/HTOD on accepted GPU path | PASS | accepted lists are helper facts loaded through relation store; scan found no list-specific DTOH/HTOD path beyond existing constraint/result reads |

---

## Example Artifacts

| Example | Coverage |
|---------|----------|
| `examples/v085-language/lists/membership.xlog` | `member`, `memberchk`, `length`, `nth` over declared `list<u32>` facts |
| `examples/v085-language/lists/transforms.xlog` | `append`, `sort`, `msort`, `list_to_set` finite-literal transforms |
| `examples/v085-language/lists/cons_patterns.xlog` | declared-list `[Head | Tail]` pattern plus tail length |

The examples are compiled by `committed_list_examples_compile`.

---

## Next Sub-Goal

Proceed to `G085_META`: safe meta-predicates, finite collection, and static predicate mapping. No push, tag, release-board update, merge, or v0.8.5 closure is authorized by this evidence.
