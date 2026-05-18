# v0.8.5 Language Completeness - NAF Evidence

**Date:** 2026-05-19
**Branch:** `feat/v085-language-completeness`
**Worktree:** `/home/dev/projects/xlog/.worktrees/v085-language`
**Base evidence:** `docs/evidence/2026-05-19-v085-meta/README.md`
**Scope:** `G085_NAF` only. Deterministic negation-as-failure safety, typed diagnostics, probabilistic WFS separation, examples, and documentation.

---

## Changed Files

| File | Purpose |
|------|---------|
| `crates/xlog-logic/src/compile.rs` | Adds compiler-path NAF safety validation after meta/list normalization and maps deterministic stratification cycles to typed `v0.8.5 naf error` diagnostics. |
| `crates/xlog-logic/tests/test_v085_naf.rs` | Adds accepted NAF, unsafe variable, unstratified cycle, probabilistic WFS separation, and committed-example coverage. |
| `crates/xlog-logic/tests/logic/stratified.xlog`, `crates/xlog-logic/tests/optimizer_integration.rs`, `crates/xlog-integration/tests/e2e_integration_tests.rs` | Updates legacy stratified-negation fixtures to use `_` for existential positions inside negated atoms. |
| `examples/v085-language/naf/closed_world.xlog` | Adds a deterministic closed-world NAF example. |
| `docs/language-reference.md`, `docs/architecture/language-v085.md`, `ROADMAP.md`, `CHANGELOG.md` | Records shipped `G085_NAF` support and the deterministic/probabilistic negation boundary. |

---

## RED Evidence

The `G085_NAF` test was introduced before implementation:

| Command | Result |
|---------|--------|
| `cargo test -p xlog-logic --test test_v085_naf` | exit 101 before implementation; unsafe named variables in negated atoms compiled as existential anti-joins instead of typed NAF errors |

---

## Implementation Summary

Deterministic `not atom(...)` remains the existing closed-world, stratified
anti-join path. `G085_NAF` adds the missing v0.8.5 semantic guard before that
path:

```text
source program
  -> query/constraint desugaring
  -> safe meta normalization
  -> finite list normalization
  -> NAF source-order safety validation
  -> stratification and existing lowering/runtime anti-join path
```

Implemented in this node:

- named variables in a negated atom must be bound by a prior source-order
  binder in the same rule body;
- anonymous `_` remains allowed for existential positions inside negated atoms;
- deterministic cycles through negation or aggregation report a compiler-path
  `v0.8.5 naf error`;
- direct low-level `stratify(&program)` callers still receive the underlying
  stratification error shape;
- probabilistic nonmonotone WFS remains separate from deterministic NAF.

This node does not add a second negation syntax, new runtime engine, or
probabilistic WFS implementation. It hardens the compiler contract around the
existing deterministic `not atom` surface.

---

## Verification Commands

| Command | Result |
|---------|--------|
| `cargo test -p xlog-logic --test test_v085_naf` | exit 0; 4 passed |
| `cargo test -p xlog-logic --test test_v085_meta` | exit 0; 6 passed |
| `cargo test -p xlog-logic` | exit 0; full `xlog-logic` package suite passed, including doc-tests |
| `cargo test -p xlog-prob --lib` | exit 0; 56 passed |
| `cargo check --workspace` | exit 0 |
| `cargo fmt --check` | exit 0 |
| `git diff --check` | exit 0 |

---

## G085_NAF Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M085_NAF.1 deterministic acceptance | stratified closed-world `not atom` compiles with list, meta, and `is` binders | PASS | `compiles_stratified_closed_world_naf_with_list_meta_and_is_binders` |
| M085_NAF.2 unsafe rejection | unbound named variables in negated atoms reject with typed diagnostics | PASS | 4 unsafe variable fixtures in `rejects_8_unsafe_or_unstratified_naf_fixtures_with_typed_diagnostics` |
| M085_NAF.3 stratification diagnostics | deterministic cycles through negation reject through the compiler path | PASS | 4 unstratified fixtures in `rejects_8_unsafe_or_unstratified_naf_fixtures_with_typed_diagnostics` |
| M085_NAF.4 probabilistic boundary | probabilistic WFS remains separate from deterministic NAF | PASS | `probabilistic_nonmonotone_wfs_remains_separate_from_deterministic_naf` |
| M085_NAF.5 example coverage | committed deterministic NAF example compiles | PASS | `committed_naf_example_compiles` |

---

## Example Artifacts

| Example | Coverage |
|---------|----------|
| `examples/v085-language/naf/closed_world.xlog` | deterministic closed-world `leaf` and `unblocked` rules using `not atom` with source-order positive binders |

The example is compiled by `committed_naf_example_compiles`.

---

## Next Sub-Goal

Proceed to `G085_MAGIC`: magic-set planning for safe bound recursive queries. No push, tag, release-board update, merge, or v0.8.5 closure is authorized by this evidence.
