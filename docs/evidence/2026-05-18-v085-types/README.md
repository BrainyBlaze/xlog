# v0.8.5 Language Completeness - TYPES Evidence

**Date:** 2026-05-18
**Branch:** `feat/v085-language-completeness`
**Worktree:** `/home/dev/projects/xlog/.worktrees/v085-language`
**Base evidence:** `docs/evidence/2026-05-18-v085-docref/README.md`
**Scope:** `G085_TYPES` only. Parser, AST, declaration metadata, scalar/domain lowering, and typed rejection foundation.

---

## Changed Files

| File | Purpose |
|------|---------|
| `crates/xlog-logic/src/ast.rs` | Adds `TypeRef`, `PredColumn`, named predicate columns, recursive variable discovery, and high-level `Term` variants for list, cons, compound, and static predicate references. |
| `crates/xlog-logic/src/grammar.pest` | Accepts list literals, cons patterns, compound terms, named predicate columns, domain aliases, `list<T>`, `term`, `compound`, and `predref` type specs. |
| `crates/xlog-logic/src/parser.rs` | Builds the new type references, predicate columns, and term forms while keeping domain/function types scalar-only where required. |
| `crates/xlog-logic/src/lower.rs` | Resolves scalar/domain declaration columns to scalar schemas, preserves declared column labels, and rejects not-yet-lowerable v0.8.5 type/term forms with typed errors. |
| `crates/xlog-logic/src/hypergraph/ir.rs` | Treats high-level term forms as non-vertices until later lowering nodes make them executable. |
| `crates/xlog-logic/src/hypergraph/reference.rs` | Rejects high-level term forms in the reference evaluator boundary. |
| `crates/xlog-logic/tests/test_v085_types.rs` | Adds RED/GREEN coverage for parser support, schema labels, negative diagnostics, and scalar compatibility. |
| `crates/xlog-prob/src/mc/buffers.rs` | Adapts MC predicate-declaration inference to `TypeRef`, preserves scalar/domain declarations, and rejects high-level forms at the probabilistic boundary. |
| `crates/xlog-prob/src/mc/mod.rs` | Propagates schema-augmentation errors. |
| `crates/xlog-prob/src/mc/results.rs` | Rejects high-level term forms in MC result matching/materialization. |
| `crates/xlog-prob/src/provenance.rs` | Rejects high-level term forms in provenance value conversion, unification, materialization, and comparison. |
| `crates/xlog-gpu/src/logic.rs` | Formats high-level terms for query/constraint display paths. |
| `crates/pyxlog/src/ilp.rs` | Stringifies `TypeRef` in ILP relation type annotation reporting. |
| `crates/xlog-integration/src/bin/xlog_run.rs` | Formats high-level terms in the integration runner. |

---

## RED Evidence

The new `G085_TYPES` test was introduced before implementation. The first run failed at compile time because the codebase did not yet expose the required surface:

| Command | Result |
|---------|--------|
| `cargo test -p xlog-logic --test test_v085_types` | exit 101 before implementation; missing `TypeRef`, `PredDecl.columns`, `Term::List`, `Term::Compound`, and `Term::Cons` |

Downstream checks also caught public-API drift after the core logic crate was green:

| Command | Initial finding |
|---------|-----------------|
| `cargo check -p xlog-prob` | `PredDecl.types` expected `Vec<ScalarType>` but became `Vec<TypeRef>`; missing `columns`; non-exhaustive high-level `Term` matches |
| `cargo check -p xlog-cli` | `xlog-gpu` formatter had non-exhaustive `Term` matches |
| `cargo check -p pyxlog` | ILP annotation reporting passed `TypeRef` to scalar-only formatter |
| `cargo check -p xlog-integration` | integration runner formatter had non-exhaustive `Term` matches |

All four boundaries were fixed as part of `G085_TYPES` because the AST/type model is a public crate surface.

---

## GREEN Verification Commands

| Command | Result |
|---------|--------|
| `cargo test -p xlog-logic --test test_v085_types` | exit 0; 5 passed |
| `cargo test -p xlog-logic` | exit 0; full `xlog-logic` unit, integration, targeted test, and doc-test suite passed |
| `cargo test -p xlog-prob --lib` | exit 0; 56 passed |
| `cargo check -p xlog-prob` | exit 0 |
| `cargo check -p xlog-cli` | exit 0 |
| `cargo check -p pyxlog` | exit 0 |
| `cargo check -p xlog-integration` | exit 0 |
| `cargo check --workspace` | exit 0 |
| `cargo fmt --check` | exit 0 |
| `git diff --check` | exit 0 |

---

## G085_TYPES Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M085_TYPES.1 type references | scalar, domain alias, `list<T>`, finite `term`, finite `compound`, and `predref` | PASS | `TypeRef::{Scalar, Domain, List, Term, Compound, PredRef}` plus parser tests |
| M085_TYPES.2 named columns | parser accepts named predicate columns and schema preserves labels | PASS | `pred edge(src: user_id, dst: u32).` lowers to schema columns `src` and `dst` |
| M085_TYPES.3 AST coverage | term enum covers list, compound, and predicate-reference forms | PASS | `Term::{List, Cons, Compound, PredRef}` added; parser test covers list, nested list, cons, and compound terms |
| M085_TYPES.4 negative diagnostics | at least 10 invalid type/term fixtures produce typed errors | PASS | `rejects_non_lowered_v085_type_and_term_forms_with_typed_errors` covers 10 invalid fixtures and requires `v0.8.5` diagnostic text |
| M085_TYPES.5 compatibility | existing scalar-only programs compile unchanged | PASS | scalar-only transitive-closure fixture compiles unchanged; full `xlog-logic` regression passes |

---

## Lowering Boundary

This checkpoint intentionally adds representation and safe diagnostics, not list/meta execution.

Lowerable now:

- scalar predicate columns;
- domain aliases over scalar types;
- named scalar/domain predicate columns preserved in `Schema`.

Parsed but compile-rejected until later `G085_*` implementation nodes:

- `list<T>` predicate columns;
- `term`, `compound`, and `predref` predicate columns;
- list literals;
- cons patterns;
- compound terms;
- static predicate-reference terms.

The rejection is explicit and typed: errors include `v0.8.5` plus the rejected form name, rather than falling through to runtime, CPU fallback, or an unsupported GPU hot path.

---

## Next Sub-Goal

Proceed to `G085_LIST`: finite list syntax execution, list built-ins, lowering, examples, and evidence. No push, tag, release-board update, merge, or v0.8.5 closure is authorized by this evidence.
