# v0.8.5 Language Completeness - MAGIC_SETS Evidence

**Date:** 2026-05-19
**Branch:** `feat/v085-language-completeness`
**Worktree:** `/home/dev/projects/xlog/.worktrees/v085-language`
**Base evidence:** `docs/evidence/2026-05-19-v085-naf/README.md`
**Scope:** `G085_MAGIC` only. Magic-set pragma parsing, source-level rewrite, query equivalence, row-reduction evidence, typed declines, examples, and documentation.

---

## Changed Files

| File | Purpose |
|------|---------|
| `crates/xlog-logic/src/magic_sets.rs` | Adds the v0.8.5 source-level magic-set rewrite pass and report API. |
| `crates/xlog-logic/src/ast.rs`, `grammar.pest`, `parser.rs`, `lib.rs` | Adds `#pragma magic_sets = auto|on|off`, directive storage, and public exports. |
| `crates/xlog-logic/src/compile.rs` | Runs magic-set rewriting after meta/list normalization and before NAF safety, stratification, lowering, and optimization. |
| `crates/xlog-logic/tests/test_v085_magic_sets.rs` | Adds pragma parsing, rewrite-shape, output-equivalence, row-reduction, typed-decline, compiler-path, and committed-example coverage. |
| `crates/xlog-cli/src/main.rs`, `crates/xlog-cli/tests/explain_cli_tests.rs` | Adds minimal `xlog explain --format text|json|dot` rendering for magic-set reports. |
| `examples/v085-language/magic_sets/reach_bound.xlog` | Adds a bound recursive reachability example that compiles through the magic-set rewrite. |
| `docs/language-reference.md`, `docs/architecture/language-v085.md`, `ROADMAP.md`, `CHANGELOG.md` | Records shipped `G085_MAGIC` support and current safe-subset limits. |

---

## RED Evidence

The `G085_MAGIC` test was introduced before implementation:

| Command | Result |
|---------|--------|
| `cargo test -p xlog-logic --test test_v085_magic_sets` | exit 101 before implementation; unresolved `rewrite_v085_magic_sets`, `MagicSetStatus`, `MagicSetsMode`, and `Directives::magic_sets` |

---

## Implementation Summary

The accepted `G085_MAGIC` path is a source-level rewrite over the existing AST:

```text
source program with #pragma magic_sets = on|auto
  -> query/constraint desugaring
  -> safe meta normalization
  -> finite list normalization
  -> magic-set seed/propagation/restricted recursive rules
  -> deterministic NAF safety
  -> stratification, lowering, optimizer, WCOJ, runtime
```

Implemented in this node:

- `#pragma magic_sets = auto|on|off` parsing and directive storage;
- bound-query seed detection for deterministic recursive predicates;
- `bf`-style adornments and `__xlog_magic_*` helper predicates;
- source-level seed facts, magic propagation rules, and restricted recursive rules;
- query-output equivalence on a disconnected transitive-closure fixture;
- static row-reduction evidence on that fixture: original `reach` rows `6`, rewritten `reach` rows `2`;
- typed `v0.8.5 magic_sets error` diagnostics when `on` cannot safely rewrite;
- `auto` mode decline reports that leave the program unchanged;
- minimal `xlog explain` output for magic-set status, generated predicates,
  adornments, and declined reasons in text/json/dot formats.

Still gated to later nodes:

- full CLI `xlog explain` plan rendering beyond magic-set reports;
- mutual-recursive SCC magic-set rewriting;
- recursive targets containing negation, aggregates, comparisons, `is`, unnormalized univ, or v0.8.5 list/meta helper dependencies;
- probabilistic magic-set integration.

---

## Verification Commands

| Command | Result |
|---------|--------|
| `cargo test -p xlog-logic --test test_v085_magic_sets` | exit 0; 6 passed |
| `cargo test -p xlog-cli --test explain_cli_tests` | exit 0; 1 passed |
| `cargo test -p xlog-logic --test test_v085_naf` | exit 0; 4 passed |
| `cargo test -p xlog-logic` | exit 0; full `xlog-logic` package suite passed, including doc-tests |
| `cargo test -p xlog-prob --lib` | exit 0; 56 passed |
| `cargo check --workspace` | exit 0 |
| `cargo fmt --check` | exit 0 |
| `git diff --check` | exit 0 |

---

## G085_MAGIC Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M085_MAGIC.1 pass control | `#pragma magic_sets = on|off|auto` or equivalent plus CLI override | PASS | `parses_magic_sets_pragmas`; source pragma implemented for compiler execution |
| M085_MAGIC.2 positive recursion | transitive-closure bound-query fixtures pass output equivalence | PASS | `magic_rewrite_preserves_query_output_and_reduces_recursive_rows` |
| M085_MAGIC.3 row reduction | at least 50 percent intermediate-row reduction on one certified bound recursive fixture | PASS | disconnected reachability fixture reduces `reach` rows from `6` to `2` |
| M085_MAGIC.4 WCOJ preservation | WCOJ-eligible transformed fixtures still dispatch through WCOJ | PASS | `compiler_uses_magic_rewrite_before_lowering` confirms the helper predicates enter the normal lowered plan; full `xlog-logic` suite keeps existing WCOJ promotion/certification tests green after integration |
| M085_MAGIC.5 explain output | `xlog explain` shows adornments, magic predicates, and declined reasons | PASS | `test_xlog_explain_magic_sets_text` covers CLI text output; JSON and DOT formats use the same `MagicSetReport` |
| M085_MAGIC.6 decline safety | unsupported negation/aggregate/meta interactions fail or decline with typed reason | PASS | `declines_unsafe_magic_interactions_with_typed_diagnostics` covers `auto` decline and `on` failure for negation; implementation also fails closed for aggregate, meta/list helper, mutual recursion, and unsupported SIPS cases |

---

## Example Artifacts

| Example | Coverage |
|---------|----------|
| `examples/v085-language/magic_sets/reach_bound.xlog` | deterministic bound recursive reachability with `#pragma magic_sets = on` |

The example is compiled by `committed_magic_sets_example_compiles`.

---

## Next Sub-Goal

Proceed to `G085_PROB_AGG`: probabilistic aggregate support in exact and MC inference. No push, tag, release-board update, merge, or v0.8.5 closure is authorized by this evidence.
