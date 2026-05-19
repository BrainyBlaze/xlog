# v0.8.5 Language Completeness And Developer Experience - PRE Evidence

**Date:** 2026-05-18
**Branch:** `feat/v085-language-completeness`
**Worktree:** `/home/dev/projects/xlog/.worktrees/v085-language`
**Goal document:** `docs/plans/2026-05-18-agent-v085-language-completeness-goal.md` from the dispatch checkout
**Scope:** `G085_PRE` only. Baseline inventory and worktree health before any v0.8.5 implementation.

---

## Baseline State

| Item | Evidence |
|------|----------|
| HEAD | `f04922035b9228133732aea1f0a1cd7fde0fd986` |
| Subject | `chore(release): prepare v0.8.0` |
| Merge-base with `main` | `f04922035b9228133732aea1f0a1cd7fde0fd986` |
| Worktree status before implementation | `git status --short --branch` printed only `## feat/v085-language-completeness` |
| Dispatch checkout note | `/home/dev/projects/xlog` had the v080/v085/v090 agent goal docs untracked when this worktree was created, so the goal documents were read from that dispatch checkout. |

The branch was created in the required isolated worktree:

```bash
git worktree add .worktrees/v085-language -b feat/v085-language-completeness main
```

The new worktree started at local `main` commit `f0492203`, satisfying the plan
requirement that the branch base be at `f0492203` or a later approved base.

---

## Baseline Commands

| Command | Result |
|---------|--------|
| `cargo fmt --check` | exit 0 |
| `cargo check -p xlog-logic` | exit 0; finished `xlog-logic v0.8.0` |
| `cargo check -p xlog-prob` | exit 0; finished `xlog-prob v0.8.0` |
| `cargo check -p xlog-cli` | exit 0; finished `xlog-cli v0.8.0` |
| `cargo test -p xlog-logic --test integration_tests test_parse_aggregate_program` | exit 0; 1 passed, 0 failed, 22 filtered out |
| `cargo test -p xlog-prob --test provenance_tc` | exit 0; 1 passed, 0 failed |
| `cargo test -p xlog-cli --test run_cli_tests` | exit 0; 1 passed, 0 failed |
| `cargo test -p xlog-cli --test prob_cli_tests` | exit 0; 0 tests compiled because the target is gated by `#![cfg(feature = "host-io")]` |
| `cargo test -p xlog-cli --features host-io --test prob_cli_tests` | exit 0; 1 passed, 0 failed |

These are `G085_PRE` baseline checks only. They do not certify list syntax,
safe meta-predicates, negation-as-failure, magic sets, probabilistic aggregates,
aggregate lifting, approximate inference, incremental parsing, REPL/watch/explain
CLI work, or v0.8.5 release closure.

---

## Authoritative References Read

| Reference | Relevance to v0.8.5 |
|-----------|---------------------|
| `ROADMAP.md` | Current release state and deferred product backlog promoted into v0.8.5. |
| `docs/language-reference.md` | Current public language reference; still advertises release context and coverage through `v0.5.2`. |
| `crates/xlog-logic/src/grammar.pest` | Current source syntax grammar. |
| `crates/xlog-logic/src/ast.rs` | Current typed AST term/body/rule representation. |
| `crates/xlog-prob/src/provenance.rs` | Current probabilistic provenance extraction and aggregate rejection points. |
| `crates/xlog-cli/src/main.rs` | Current CLI command surface. |
| `/home/dev/projects/xlog/docs/plans/2026-05-18-agent-v080-dts-ml-python-goal.md` | v0.8.0 boundary to preserve while extending language/CLI surfaces. |
| `/home/dev/projects/xlog/docs/plans/2026-05-18-agent-v090-epistemic-solver-goal.md` | v0.9.0 boundary; epistemic and solver semantics remain out of v0.8.5 scope. |

---

## Baseline Inventory

| Surface | Current state before v0.8.5 work |
|---------|----------------------------------|
| Language reference | `docs/language-reference.md` still says release context and coverage through `v0.5.2`. |
| Grammar terms | `grammar.pest` parses `term = { var_or_anon \| float_num \| integer \| string_lit \| ident }`; list and compound terms are not present. |
| Grammar aggregates | Aggregates are parsed only as `agg_op(variable)` in head terms. |
| Grammar negation | `not atom` exists as stratified/body negation syntax; there is no distinct NAF surface. |
| Grammar probabilistic pragmas | Only `prob_engine`, `prob_cache`, and `max_recursion_depth` pragmas are present. |
| AST terms | `Term` variants are variable, anonymous, integer, float, string, symbol, and aggregate. There are no list, compound, callable, or meta-predicate AST nodes. |
| Probabilistic aggregates | `provenance.rs` rejects `Term::Aggregate` with `Aggregation not supported in provenance extraction`. |
| CLI | `Command` contains only `Run` and `Prob`; no `repl`, `watch`, or `explain` subcommands exist. |

---

## Ownership Map

| Area | Primary files/modules | v0.8.5 ownership notes |
|------|-----------------------|------------------------|
| Parser and grammar | `crates/xlog-logic/src/grammar.pest`, `crates/xlog-logic/src/parser.rs` | Own new syntax acceptance, incremental parse entrypoints, and parser diagnostics. |
| Typed AST and normalization seams | `crates/xlog-logic/src/ast.rs`, `crates/xlog-logic/src/compile.rs`, `crates/xlog-logic/src/lower.rs` | Own list/compound/meta/NAF AST representation and desugaring before existing lowering. |
| Semantic validation and stratification | `crates/xlog-logic/src/stratify.rs`, `crates/xlog-logic/src/type_inference.rs` | Own finite-domain checks, safe meta-predicate validation, and NAF/magic-set safety gates. |
| RIR and deterministic execution handoff | `crates/xlog-ir`, `crates/xlog-runtime`, `crates/xlog-gpu` | Reuse existing RIR/runtime/GPU execution paths; no parallel runtime. |
| Probabilistic provenance and inference | `crates/xlog-prob/src/provenance.rs`, `crates/xlog-prob/src/pir.rs`, `crates/xlog-prob/src/exact.rs`, `crates/xlog-prob/src/mc.rs` | Own probabilistic aggregate lowering, aggregate lifting, and approximate inference reporting integration. |
| CLI product surface | `crates/xlog-cli/src/main.rs`, `crates/xlog-cli/tests` | Own `repl`, `watch`, and `explain` subcommands while preserving `run` and `prob`. |
| Examples and validators | `examples`, `scripts/validate_v085_examples.py`, `python/tests` if a validator is added | Own runnable examples and source-level validation evidence. |
| Documentation and release evidence | `docs/language-reference.md`, `docs/architecture`, `ROADMAP.md`, `CHANGELOG.md`, `docs/evidence/2026-05-18-v085-*` | Own public docs, architecture notes, evidence ledgers, and closure proposal material. |

---

## Deferred Backlog Mapping

| Roadmap backlog item | v0.8.5 node |
|----------------------|-------------|
| Add incremental parsing for interactive use. | `G085_INC_PARSE` |
| Add list syntax and list built-ins. | `G085_LIST` |
| Add meta-predicates such as `ground`, `var`, `=..`, `functor`, `findall`, and `maplist`. | `G085_META` |
| Add negation-as-failure syntax and semantics where it is distinct from existing WFS support. | `G085_NAF` |
| Add magic sets transformation. | `G085_MAGIC` |
| Add aggregate support in probabilistic programs. | `G085_PROB_AGG` |
| Add aggregate lifting for small domains. | `G085_AGG_LIFT` |
| Add approximate inference engine. | `G085_APPROX` |
| Add interactive REPL. | `G085_CLI` |
| Add watch mode. | `G085_CLI` |
| Add CLI explain/plan visualization. | `G085_CLI` |

---

## G085_PRE Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M085_PRE.1 branch base | `git merge-base HEAD main` equals `f0492203` or later approved base | PASS | merge-base and HEAD are both `f04922035b9228133732aea1f0a1cd7fde0fd986` |
| M085_PRE.2 worktree status | clean before implementation begins, except known inherited goal docs | PASS | status printed only the branch header in this isolated worktree |
| M085_PRE.3 ownership map | crate/file ownership table committed in evidence | PASS | ownership table above |
| M085_PRE.4 baseline commands | required checks plus targeted parser/prob/CLI tests recorded | PASS | command table above |
| M085_PRE.5 backlog mapping | all 11 deferred roadmap items mapped to `G085_*` nodes | PASS | backlog mapping table above |

---

## Next Sub-Goal

Proceed to `G085_DOCREF`: update the language reference and architecture docs
to define the v0.8.5 language contract before parser/runtime implementation.
No push, tag, release-board update, or merge is authorized by this evidence.
