# v0.8.5 Language Completeness - DOCREF Evidence

**Date:** 2026-05-18
**Branch:** `feat/v085-language-completeness`
**Worktree:** `/home/dev/projects/xlog/.worktrees/v085-language`
**Base evidence:** `docs/evidence/2026-05-18-v085-pre/README.md`
**Scope:** `G085_DOCREF` only. Documentation and semantic contract refresh
before parser/runtime implementation nodes.

---

## Changed Files

| File | Purpose |
|------|---------|
| `docs/language-reference.md` | Updates release context to v0.8.5, adds language contract, lists, meta-predicates, magic sets, probabilistic aggregate, aggregate lifting, approximate inference, CLI developer-experience, pragmas, grammar appendix, and unsupported diagnostic examples. |
| `docs/architecture/language-v085.md` | Adds parser/term/probability/CLI architecture contract plus v0.9.0 handoff notes and source-audit snapshot. |
| `docs/architecture/cli-reference.md` | Records that current source exposes `run`/`prob` while v0.8.5 adds implementation-gated `explain`/`repl`/`watch`. |
| `ROADMAP.md` | Promotes the deferred product backlog into an active v0.8.5 release section. |
| `CHANGELOG.md` | Adds an Unreleased entry for the v0.8.5 documentation contract. |

---

## Source-Audit Position

The docs intentionally separate v0.8.5 contract text from current source
support. At this checkpoint:

- current source grammar still has scalar `term = { var_or_anon | float_num |
  integer | string_lit | ident }`;
- `Term` in `crates/xlog-logic/src/ast.rs` still has no list, compound,
  callable, or predicate-reference variants;
- `crates/xlog-prob/src/provenance.rs` still reports `Aggregation not
  supported in provenance extraction`;
- `crates/xlog-cli/src/main.rs` still has only `Run` and `Prob` commands.

The language reference and architecture contract say that v0.8.5 additions are
unsupported until their corresponding `G085_*` implementation nodes land.

---

## Verification Commands

| Command | Result |
|---------|--------|
| `rg -n "v0\\.5\\.2" docs/language-reference.md` | exit 1; no stale language-reference release strings |
| `rg -n "Release context.*v0\\.8\\.5|## v0\\.8\\.5 Language Contract|## Lists|## Safe Meta-Predicates|## Magic Sets|### Probabilistic Aggregates|### Aggregate Lifting|## Approximate Inference|## CLI Developer Experience" docs/language-reference.md` | exit 0; required contract sections present |
| `rg -n "current source|implementation-gated|unsupported until|source-audit|still rejects|still exposes|contract extensions" docs/language-reference.md docs/architecture/language-v085.md docs/architecture/cli-reference.md` | exit 0; source-audit caveats present |
| `rg -n "term = \\{|list|compound|predref|meta_goal|magic_sets|prob_samples|prob_seed|prob_confidence|prob_method" crates/xlog-logic/src/grammar.pest docs/language-reference.md` | exit 0; current grammar and v0.8.5 grammar contract both visible |
| `rg -n "pub enum Term|List|Compound|PredRef|Meta" crates/xlog-logic/src/ast.rs` | exit 0 only for `pub enum Term`, confirming no new AST support is falsely implied by source |
| `rg -n "Aggregation not supported in provenance extraction|enum Command|Run\\(|Prob\\(|explain|repl|watch" crates/xlog-prob/src/provenance.rs crates/xlog-cli/src/main.rs docs/architecture/cli-reference.md` | exit 0; current rejection/CLI state and implementation-gated docs visible |
| `rg -n "## v0\\.8\\.5|language-v085|validate_v085_examples|Language Completeness" ROADMAP.md` | exit 0; roadmap has v0.8.5 section and validator placeholder |
| `rg -n "language-completeness documentation contract|language-v085" CHANGELOG.md` | exit 0; changelog has v0.8.5 docs-contract entry |
| `test -f docs/architecture/language-v085.md && test -f docs/evidence/2026-05-18-v085-docref/README.md` | exit 0 |
| `git diff --check` | exit 0 |
| `cargo fmt --check` | exit 0 |

---

## G085_DOCREF Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M085_DOCREF.1 release context | `docs/language-reference.md` says `v0.8.5` | PASS | header now says `Release context: XLOG v0.8.5` |
| M085_DOCREF.2 stale release strings | 0 stale `v0.5.2` references in language reference | PASS | stale-string scan above exits 1 with no matches |
| M085_DOCREF.3 coverage matrix | every G085 feature has syntax, semantics, examples, unsupported forms, and execution path | PASS | `v0.8.5 Language Contract` feature matrix covers all G085 feature nodes |
| M085_DOCREF.4 grammar appendix | grammar reference updated for lists, compounds, meta-predicates, pragmas, and CLI examples | PASS | grammar appendix includes current grammar plus v0.8.5 contract grammar and CLI examples |
| M085_DOCREF.5 source audit | docs do not claim support for forms that code rejects | PASS | source-audit caveats and implementation-gated CLI/probability notes are present |

---

## Next Sub-Goal

Proceed to `G085_TYPES`: parser, AST, type metadata, and diagnostics foundation.
No push, tag, release-board update, or merge is authorized by this evidence.
