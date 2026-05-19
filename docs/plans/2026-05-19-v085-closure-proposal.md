# v0.8.5 Language Completeness Closure Proposal

Date: 2026-05-19
Branch: `feat/v085-language-completeness`
Certification head before closure proposal: `25cf0f1e45c646abbd68b681d84632c6feb965ea`

## Recommendation

`MERGE_READY`, pending coordinator authorization for the actual merge.

This proposal does not authorize or perform a push, tag, release-board update,
or merge. v0.9.0 work should rebase or merge after v0.8.5 lands because this
branch changes parser, AST, finite-term, probability, and CLI surfaces that the
epistemic/solver branch is expected to build on.

## Sub-Goal Table

| Goal | Commit | Status | Evidence |
|------|--------|--------|----------|
| G085_PRE | `3d577556` | PASS | `docs/evidence/2026-05-18-v085-pre/README.md` |
| G085_DOCREF | `ad016b2d` | PASS | `docs/evidence/2026-05-18-v085-docref/README.md` |
| G085_TYPES | `b0415a95` | PASS | `docs/evidence/2026-05-18-v085-types/README.md` |
| G085_LIST | `3053b2e5` | PASS | `docs/evidence/2026-05-19-v085-lists/README.md` |
| G085_META | `a549921c` | PASS | `docs/evidence/2026-05-19-v085-meta/README.md` |
| G085_NAF | `5acc3a60` | PASS | `docs/evidence/2026-05-19-v085-naf/README.md` |
| G085_MAGIC | `e61961b1` | PASS | `docs/evidence/2026-05-19-v085-magic-sets/README.md` |
| G085_PROB_AGG | `23e57dcb` | PASS | `docs/evidence/2026-05-19-v085-prob-aggregates/README.md` |
| G085_AGG_LIFT | `b3087a88` | PASS | `docs/evidence/2026-05-19-v085-aggregate-lift/README.md` |
| G085_APPROX | `470564c5` | PASS | `docs/evidence/2026-05-19-v085-approx/README.md` |
| G085_INC_PARSE | `03f87db1` | PASS | `docs/evidence/2026-05-19-v085-incremental-parse/README.md` |
| G085_CLI | `72d8c9de` | PASS | `docs/evidence/2026-05-19-v085-cli/README.md` |
| G085_EXAMPLES | `19a1f6c5` | PASS | `docs/evidence/2026-05-19-v085-examples/README.md` |
| G085_INT | `25cf0f1e` | PASS | `docs/evidence/2026-05-19-v085-int/README.md` |
| G085_CLOSE | proposal commit | PASS | `docs/evidence/2026-05-19-v085-close/README.md` |

## GQM Metric Table

| Metric | Status | Raw result |
|--------|--------|------------|
| M085_PRE.1 branch/worktree | PASS | `.worktrees/v085-language` on `feat/v085-language-completeness` |
| M085_DOCREF.1 language reference | PASS | `docs/language-reference.md` refreshed to v0.8.5 contract |
| M085_DOCREF.2 architecture handoff | PASS | `docs/architecture/language-v085.md` records parser, term, probability, CLI, and v0.9.0 handoff |
| M085_TYPES.* | PASS | type/domain/list/term parser and lowering tests pass under `cargo test -p xlog-logic` |
| M085_LIST.* | PASS | finite list syntax, built-ins, cons patterns, and helper lowering tests pass; ordinary `pair/2` compatibility regression added |
| M085_META.* | PASS | `ground`, `var`, `nonvar`, `functor`, `=..`, `findall`, and `maplist` tests pass |
| M085_NAF.* | PASS | deterministic source-order-bound NAF tests pass; probabilistic WFS remains separate |
| M085_MAGIC.* | PASS | bound recursive magic-set rewrite tests pass; explain reports generated predicates |
| M085_PROB_AGG.* | PASS | exact finite probabilistic aggregate tests pass; cap diagnostics tested |
| M085_AGG_LIFT.* | PASS | count lift reports `131072` naive outcomes versus `171` DP states on the certified fixture |
| M085_APPROX.* | PASS | approximate inference pragmas and MC metadata tests pass |
| M085_INC_PARSE.* | PASS | parser-session cache, invalidation, and span diagnostics tests pass |
| M085_CLI.* | PASS | explain, REPL, and watch tests pass |
| M085_EXAMPLES.1 example count | PASS | `example_count=10` |
| M085_EXAMPLES.2 feature coverage | PASS | every required v0.8.5 feature has at least one showcase example |
| M085_EXAMPLES.3 interaction coverage | PASS | `interaction_count=10`, target `>=5` |
| M085_EXAMPLES.4 validator | PASS | `python3 scripts/validate_v085_examples.py --output /tmp/v085_examples_validation_summary.json` |
| M085_EXAMPLES.5 evidence JSON | PASS | `docs/evidence/2026-05-19-v085-examples/validation_summary.json` committed |
| M085_INT.1 format | PASS | `cargo fmt --check` |
| M085_INT.2 logic tests | PASS | `cargo test -p xlog-logic` |
| M085_INT.3 prob tests | PASS | `cargo test -p xlog-prob` |
| M085_INT.4 cli tests | PASS | `cargo test -p xlog-cli` |
| M085_INT.5 runtime/integration | PASS | `cargo test -p xlog-runtime`; `cargo test -p xlog-integration` |
| M085_INT.6 examples | PASS | v0.8.5 validator passed |
| M085_INT.7 source audit | PASS | `no_cpu_d4_in_exact`, `no_dtoh_gpu_native`, and `no_dtoh_*` tests passed inside `cargo test -p xlog-prob` |
| M085_INT.8 v0.8.0 compatibility | PASS | `pytest -q python/tests/test_v080_examples_source.py python/tests/test_v085_examples_source.py` -> `4 passed` |
| M085_INT.9 docs/hygiene | PASS | JSON validation, stale-marker scan, and `git diff --check` passed |
| M085_CLOSE.1 metric table | PASS | this table marks every metric PASS |
| M085_CLOSE.2 roadmap | PASS | `ROADMAP.md` has explicit v0.8.5 section with completed examples/certification items and v0.9.0 ordering preserved |
| M085_CLOSE.3 changelog | PASS | `CHANGELOG.md` has explicit `0.8.5` entry, migration notes, and release-status note |
| M085_CLOSE.4 closure proposal | PASS | this document |
| M085_CLOSE.5 worktree | PASS | clean before close edits; final status checked after commit |
| M085_CLOSE.6 no unauthorized actions | PASS | no push, tag, merge, or board update performed |

## Verification Matrix

| Command | Result |
|---------|--------|
| `cargo fmt --check` | exit 0 |
| `cargo check -p xlog-logic` | exit 0 |
| `cargo check -p xlog-prob` | exit 0 |
| `cargo check -p xlog-cli` | exit 0 |
| `cargo test -p xlog-logic` | exit 0 |
| `cargo test -p xlog-prob` | exit 0 |
| `cargo test -p xlog-cli` | exit 0 |
| `cargo test -p xlog-runtime` | exit 0 |
| `cargo test -p xlog-integration` | exit 0 after fixing the `pair/2` compatibility regression |
| `python3 scripts/validate_v085_examples.py --output /tmp/v085_examples_validation_summary.json` | exit 0; `example_count=10`; `interaction_count=10` |
| `pytest -q python/tests/test_v080_examples_source.py python/tests/test_v085_examples_source.py` | exit 0; `4 passed` |
| `python3 -m json.tool docs/evidence/2026-05-19-v085-examples/validation_summary.json` | exit 0 |
| `python3 -m json.tool /tmp/v085_examples_validation_summary.json` | exit 0 |
| `git diff --check` | exit 0 |
| targeted stale-marker scan | no matches |

## Known Unsupported Forms

The following remain intentionally outside v0.8.5 and are documented in
`docs/language-reference.md` and `docs/architecture/language-v085.md`:

- non-finite or open recursive terms and arbitrary CPU term heaps;
- dynamic `call/N`, runtime-variable predicate names, and unrestricted meta
  execution;
- derived-goal `findall` and non-literal `maplist` inputs in the current safe
  meta subset;
- recursive rules crossing list/meta helper predicates;
- unsafe or unbound deterministic NAF;
- magic-set rewrites for unsafe negative/probabilistic interactions;
- exact probabilistic aggregate domains above the finite caps unless routed to
  MC or reduced-domain fixtures;
- epistemic EIR, world views, GPT, FAEEL, and solver semantics, which remain
  v0.9.0 scope.

## v0.9.0 Rebase Note

v0.9.0 should rebase on or merge v0.8.5 before implementing epistemic/solver
semantics. The parser and AST now preserve v0.8.5 source forms, incremental
parse spans, probability directives, and finite helper-lowering boundaries that
v0.9.0 should treat as existing language contract rather than reimplementing in
parallel.

## Prior Goal Reuse Applied

- v0.8.0 `G080_EXAMPLES` remains a separate post-close addendum; it was not
  inserted into the original v0.8.0 closure table. v0.8.5 keeps its example
  suite and validator as a separate `G085_EXAMPLES` node with its own evidence.
- Earlier closure flows kept proposal, approval, merge, push, tag, and board
  actions separate. This proposal follows that split and stops for coordinator
  approval.

## Required Coordinator Actions

1. Review this closure proposal and the linked evidence directories.
2. Decide whether to authorize merge.
3. If merge is authorized, perform or explicitly delegate merge/push/tag steps.
