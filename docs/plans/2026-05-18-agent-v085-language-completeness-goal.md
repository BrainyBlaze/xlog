# Agent Goal 042 - v0.8.5 Language Completeness And Developer Experience

**Agent:** Agent C, v0.8.5 language implementation worker.
**Branch:** `feat/v085-language-completeness`.
**Worktree:** `.worktrees/v085-language`.
**Base:** `main` at or after `f0492203` (`chore(release): prepare v0.8.0`).
**Integration order:** v0.8.5 lands after v0.8.0 and before v0.9.0. The active v0.9.0 branch must rebase or merge after v0.8.5 because parser, AST, lowering, probabilistic, and CLI surfaces overlap.
**Status:** Dispatch-ready goal document. Implementation begins only after the worktree is created and baseline status is recorded.

## 0. Process Model

This goal uses the requested Goal-Driven Software Development Process and GQM framing.

Sources read:

- Goal-Driven Software Development Process, Wikipedia, accessed 2026-05-18:
  https://en.wikipedia.org/wiki/Goal-Driven_Software_Development_Process
- GQM, Wikipedia, accessed 2026-05-18:
  https://en.wikipedia.org/wiki/GQM

Method adaptation:

- GDSP starts from goals before requirements, and lets technical feasibility feed back into goal shape. Agent C must keep the top-down language-product goal and the bottom-up xlog GPU/RIR constraints visible at every sub-goal.
- GDSP emphasizes top-down and bottom-up convergence. New syntax is accepted only when it maps cleanly to existing production parser, RIR, optimizer, probabilistic, runtime, or CLI paths.
- GDSP favors small goal sets and vertical ownership. v0.8.5 is one vertically owned release train: parser, AST, lowerer, probabilistic semantics, CLI, docs, examples, and evidence all close together.
- GQM is used at three levels: conceptual goals, operational questions, and quantitative metrics. Every sub-goal below has explicit questions and measurable targets.
- The execution loop is: planning, definition, data collection, interpretation. Every sub-goal commit must leave raw evidence that can be interpreted against its metrics.

The GDSP Wikipedia page was marked with a proposed-deletion banner on 2026-05-18. This document uses it only as the user-requested process pointer and makes the operational goal, question, metric, and evidence rules explicit.

## 1. Business Goal

**BG085.** Ship xlog v0.8.5 as a production-grade language completeness release that promotes the v0.8.0 deferred product backlog into a coherent, typed, GPU-native XLOG language iteration.

The release is successful only if xlog gains:

- first-class list syntax and finite list built-ins;
- safe meta-predicates including `ground`, `var`, `nonvar`, `functor`, `=..`, `findall`, and `maplist`;
- explicit negation-as-failure semantics where they differ from probabilistic WFS;
- a magic-set rewrite pass for bound recursive queries;
- probabilistic aggregate support in exact and MC inference;
- small-domain aggregate lifting;
- first-class approximate inference configuration;
- incremental parsing for interactive use;
- `xlog repl`, `xlog watch`, and `xlog explain`;
- updated language reference, examples, and certification evidence.

The release must preserve v0.8.0 DTS-DLM ML/Python behavior and must not weaken v0.9.0 GPU-native epistemic/solver requirements.

## 1.1 Architectural Contract

v0.8.5 is a language-surface expansion, not a second runtime.

Accepted features must follow this shape:

```text
source syntax
  -> typed AST
  -> semantic normalization / desugaring
  -> existing RIR, probabilistic IR, optimizer, runtime, WCOJ, and CLI paths
  -> GPU execution where evaluation occurs
```

Allowed CPU responsibilities:

- parsing;
- static semantic analysis;
- desugaring and plan construction;
- CLI orchestration;
- diagnostics formatting;
- final requested result transfer.

Blocked CPU responsibilities:

- accepted list evaluation through a CPU Prolog-style term heap;
- accepted meta-predicate execution through dynamic CPU predicate calls;
- accepted magic-set execution outside the rewritten XLOG/RIR program;
- accepted probabilistic aggregate evaluation by CPU-only enumeration when exact or MC GPU mode is requested;
- hidden host transfers on GPU hot paths.

New high-level terms may exist in the parser and AST, but execution must lower to finite typed relations, scalar columns, GPU-resident buffers, and existing relation-store layouts.

## 1.2 Consumer And Release Context

v0.8.5 has three consumers:

- **General XLOG users:** need a coherent, documented language that is no longer behind the runtime capability.
- **DTS-DLM:** needs richer `.xlog` programs for future language-model and reasoning fixtures without moving structural work back into Python.
- **v0.9.0 epistemic/solver work:** needs a stronger parser, term model, explain tooling, and probabilistic aggregate substrate before epistemic programs become fully production-routable.

v0.8.5 must not implement epistemic semantics. It may prepare reusable language infrastructure that v0.9.0 consumes.

## 2. Scope Boundaries

### In Scope

- Language reference refresh from v0.5.2/v0.8.0 state to v0.8.5.
- Type model extensions for domain aliases, named columns, `list<T>`, finite `term`, finite `compound`, and static `predref`.
- List syntax and list built-ins.
- Safe meta-predicates and static meta-call expansion.
- Explicit deterministic NAF and probabilistic WFS documentation/configuration.
- Magic-set transformation for bound recursive queries.
- Probabilistic aggregates for exact and MC inference.
- Small-domain aggregate lifting.
- Approximate inference pragmas and reporting.
- Incremental parser/session cache.
- `xlog explain`, `xlog repl`, and `xlog watch`.
- Advanced `.xlog` examples and certification scripts.
- ROADMAP, CHANGELOG, language reference, and closure evidence updates.

### Out Of Scope

- v0.9.0 epistemic EIR, G91, FAEEL, world-view semantics, Generate-Propagate-Test, and solver semantics.
- v0.10.0 multi-GPU or out-of-core execution.
- Arbitrary Prolog runtime semantics: cut, assert, retract, dynamic database mutation, IO predicates, unrestricted `call/N`, unrestricted higher-order predicate variables, or open-ended unbounded list generation.
- CPU-only accepted execution paths for language features that claim GPU support.
- pyxlog public API changes unless required for examples or certification and kept additive.
- Push, tag, merge, or release-board updates without coordinator authorization.

### Coordination Locks

- **Reuse production paths.** Do not create parallel list, meta, probability, or query execution engines when existing AST, RIR, probabilistic IR, runtime, WCOJ, or solver paths can be extended.
- **Finite semantics only.** New list/meta constructs must be finite, typed, and safety-checked. Unsafe or unbounded programs fail with typed diagnostics.
- **No silent fallback.** Unsupported forms must fail explicitly. A CPU fallback may exist only as an oracle test, never as an accepted release path.
- **v0.8.0 compatibility.** Existing v0.8.0 DTS-DLM pyxlog and examples remain green.
- **v0.9.0 handoff.** Any parser/AST interfaces needed by v0.9.0 must be documented and stable enough for the v0.9.0 branch to rebase.
- **Evidence required.** Each sub-goal writes evidence under `docs/evidence/<date>-v085-*/`.

## 3. Roadmap Mapping

| ROADMAP backlog item | v0.8.5 goal node | Agent C responsibility |
|---|---|---|
| Incremental parsing for interactive use | G085_INC_PARSE | statement-indexed parser cache and invalidation |
| List syntax and list built-ins | G085_LIST | parser, AST, type checking, desugaring, execution cert |
| Meta-predicates `ground`, `var`, `=..`, `functor`, `findall`, `maplist` | G085_META | safe static meta system and lowering |
| NAF distinct from WFS | G085_NAF | explicit deterministic/probabilistic negation contract |
| Magic sets transformation | G085_MAGIC | adornment, SIPS, rewrite, explain, cert |
| Aggregate support in probabilistic programs | G085_PROB_AGG | exact provenance and MC aggregate support |
| Aggregate lifting for small domains | G085_AGG_LIFT | finite-domain lifted aggregate execution |
| Approximate inference engine | G085_APPROX | first-class MC pragmas, CLI, metrics |
| Interactive REPL | G085_CLI | `xlog repl` |
| Watch mode | G085_CLI | `xlog watch` |
| CLI explain/plan visualization | G085_CLI | `xlog explain` text/json/dot |

## 4. Current Codebase Baseline

Agent C must verify these facts at G085_PRE and update this section in evidence if the code has changed:

- `docs/language-reference.md` still declares release context `v0.5.2` and must be refreshed.
- `crates/xlog-logic/src/grammar.pest` currently supports scalar terms only: variables, floats, integers, string literals, and identifiers.
- `crates/xlog-logic/src/ast.rs` has no list, compound, callable, or meta-literal term variants.
- `crates/xlog-logic/src/compile.rs` has a usable pipeline seam before stratification/lowering and before optimizer/WCOJ promotion.
- `crates/xlog-prob/src/provenance.rs` rejects aggregate rules in exact provenance extraction.
- `crates/xlog-cli/src/main.rs` exposes `run` and `prob`, but not `explain`, `repl`, or `watch`.

## 5. Goal Hierarchy

```text
BG085 - xlog v0.8.5 language completeness
 |
 +-- G085_PRE        Baseline inventory and worktree health
 +-- G085_DOCREF     Language reference and semantic contract refresh
 +-- G085_TYPES      Type and term model foundation
 +-- G085_LIST       Lists and list built-ins
 +-- G085_META       Safe meta-predicates
 +-- G085_NAF        Explicit negation-as-failure semantics
 +-- G085_MAGIC      Magic-set transformation
 +-- G085_PROB_AGG   Probabilistic aggregate support
 +-- G085_AGG_LIFT   Small-domain aggregate lifting
 +-- G085_APPROX     Approximate inference language integration
 +-- G085_INC_PARSE  Incremental parsing/session cache
 +-- G085_CLI        REPL, watch, and explain
 +-- G085_EXAMPLES   Advanced .xlog examples and validators
 +-- G085_INT        Integration, regression, and certification
 +-- G085_CLOSE      Evidence, roadmap sync, and closure proposal
```

Recommended execution order is the hierarchy order above. G085_DOCREF may be drafted early but must be updated again after implementation.

## 6. GQM Decomposition

### G085_PRE - Baseline Inventory And Worktree Health

**Goal.** Establish a clean v0.8.5 worktree for the purpose of implementing the language completeness pack with respect to v0.8.0 release behavior and v0.9.0 branch coordination.

**Questions.**

- Q085_PRE.1: Is the branch cut from the intended v0.8.0 base?
- Q085_PRE.2: Which crates own parser, AST, lowering, runtime, probability, CLI, examples, and docs?
- Q085_PRE.3: Which current tests are the compatibility baseline?
- Q085_PRE.4: Which deferred roadmap items are promoted into v0.8.5?

**Metrics.**

| Metric | Target |
|---|---|
| M085_PRE.1 branch base | `git merge-base HEAD main` equals `f0492203` or later approved base |
| M085_PRE.2 worktree status | clean before implementation begins, except known untracked goal docs if inherited |
| M085_PRE.3 ownership map | crate/file ownership table committed in evidence |
| M085_PRE.4 baseline commands | `cargo fmt --check`, `cargo check -p xlog-logic`, `cargo check -p xlog-prob`, `cargo check -p xlog-cli`, and targeted parser/prob/CLI tests recorded |
| M085_PRE.5 backlog mapping | all 11 deferred roadmap items mapped to G085 nodes |

**Evidence.** `docs/evidence/<date>-v085-pre/README.md`.

**Definition of Done.**

- Worktree exists at `.worktrees/v085-language`.
- Baseline command results are recorded with raw command text and exit codes.
- No implementation changes happen before evidence is committed or staged for the first sub-goal.

### G085_DOCREF - Language Reference And Semantic Contract Refresh

**Goal.** Refresh the XLOG language reference for the purpose of making v0.8.5 syntax, semantics, and unsupported cases explicit from the viewpoint of users and downstream agents.

**Questions.**

- Q085_DOCREF.1: Does the language reference describe the actual current language plus v0.8.5 additions?
- Q085_DOCREF.2: Are GPU-native guarantees and typed rejection cases stated precisely?
- Q085_DOCREF.3: Are deterministic, probabilistic, approximate, and CLI semantics separated?
- Q085_DOCREF.4: Can v0.9.0 agents locate the parser/term contracts they depend on?

**Metrics.**

| Metric | Target |
|---|---|
| M085_DOCREF.1 release context | `docs/language-reference.md` says `v0.8.5` |
| M085_DOCREF.2 stale release strings | 0 stale `v0.5.2` references in language reference |
| M085_DOCREF.3 coverage matrix | every G085 feature has syntax, semantics, examples, unsupported forms, and execution path |
| M085_DOCREF.4 grammar appendix | grammar reference updated for lists, compounds, meta-predicates, pragmas, and CLI examples |
| M085_DOCREF.5 source audit | docs do not claim support for forms that code rejects |

**Expected targets.**

- `docs/language-reference.md`
- `docs/architecture/language-v085.md`
- `CHANGELOG.md`
- `ROADMAP.md`

**Definition of Done.**

- Docs are updated before final closure.
- A source scan shows no stale language-version header.
- Unsupported forms are documented with diagnostic examples.

### G085_TYPES - Type And Term Model Foundation

**Goal.** Extend the type and term model for the purpose of representing v0.8.5 language constructs safely, with respect to finite GPU-lowerable execution.

**Questions.**

- Q085_TYPES.1: Can predicate declarations use domain aliases and named columns?
- Q085_TYPES.2: Can AST represent lists, compounds, term values, and static predicate references?
- Q085_TYPES.3: Can the lowerer reject non-finite or non-GPU-lowerable terms with typed errors?
- Q085_TYPES.4: Does schema metadata preserve names and sort labels for downstream consumers?

**Metrics.**

| Metric | Target |
|---|---|
| M085_TYPES.1 type references | `TypeRef` or equivalent supports scalar, domain alias, `list<T>`, finite `term`, finite `compound`, and `predref` |
| M085_TYPES.2 named columns | parser accepts named predicate columns and schema preserves labels |
| M085_TYPES.3 AST coverage | term enum covers list, compound, and predicate-reference forms |
| M085_TYPES.4 negative diagnostics | at least 10 invalid type/term fixtures produce typed errors |
| M085_TYPES.5 compatibility | existing scalar-only programs compile unchanged |

**Expected targets.**

- `crates/xlog-logic/src/grammar.pest`
- `crates/xlog-logic/src/ast.rs`
- `crates/xlog-logic/src/parser.rs`
- `crates/xlog-logic/src/lower.rs`
- `crates/xlog-core/src/types.rs`
- parser and integration tests

**Definition of Done.**

- Parser tests prove new type forms.
- Lowering tests prove old scalar programs remain compatible.
- Evidence states which new type forms lower to scalar relation layouts and which remain compile-time only.

### G085_LIST - Lists And List Built-Ins

**Goal.** Add finite list syntax and list built-ins for the purpose of making XLOG expressive enough for collection-oriented logic programs while preserving GPU-native execution.

**Questions.**

- Q085_LIST.1: Can users write `[]`, `[A, B]`, and safe `[H|T]` patterns?
- Q085_LIST.2: Are list values normalized into typed helper relations rather than CPU term objects?
- Q085_LIST.3: Do list built-ins compose with deterministic joins, aggregates, and queries?
- Q085_LIST.4: Are unsafe open-ended list generators rejected?

**Metrics.**

| Metric | Target |
|---|---|
| M085_LIST.1 syntax fixtures | at least 25 parse fixtures for list literals, nested lists, cons patterns, and errors |
| M085_LIST.2 built-in coverage | `is_list`, `member`, `memberchk`, `length`, `nth`, `append`, `sort`, `msort`, `list_to_set`, and pair helpers implemented or explicitly rejected with rationale |
| M085_LIST.3 GPU lowering | accepted built-ins lower to RIR/runtime/prob paths, not CPU evaluator |
| M085_LIST.4 equivalence | at least 12 execution fixtures match oracle output exactly |
| M085_LIST.5 unsafe rejection | unbounded list generation and cyclic-list forms fail with typed diagnostics |
| M085_LIST.6 no host hot path | accepted GPU execution introduces no hidden hot-path DTOH/HTOD transfers beyond requested output |

**Expected targets.**

- parser/AST/lowerer files from G085_TYPES
- `crates/xlog-runtime/` only if new helper relation operators need runtime support
- `crates/xlog-logic/tests/test_v085_lists.rs`
- `examples/v085-language/lists/`

**Definition of Done.**

- Lists can be used in facts, rule bodies, queries, `findall`, and `maplist` where finite and typed.
- Accepted list forms are documented in the language reference.
- Rejected Prolog-style open list generation is documented and tested.

### G085_META - Safe Meta-Predicates

**Goal.** Add safe meta-predicates for the purpose of supporting term inspection, finite collection, and static predicate mapping, with respect to XLOG's typed Datalog semantics.

**Questions.**

- Q085_META.1: Can `ground`, `var`, and `nonvar` be evaluated at the correct static/runtime boundary?
- Q085_META.2: Can `functor` and `=..` inspect and construct finite compound terms without creating arbitrary dynamic terms?
- Q085_META.3: Can `findall` collect finite goal results into normalized list relations?
- Q085_META.4: Can `maplist` statically expand over predicate references without unrestricted dynamic `call/N`?

**Metrics.**

| Metric | Target |
|---|---|
| M085_META.1 predicate coverage | `ground`, `var`, `nonvar`, `functor`, `=..`, `findall`, and `maplist` covered |
| M085_META.2 static-call lock | runtime-variable predicate names rejected with typed error |
| M085_META.3 findall semantics | empty result succeeds with empty list; duplicate policy documented and tested |
| M085_META.4 maplist arities | unary and binary `maplist` accepted; higher arities either accepted or explicitly rejected |
| M085_META.5 fixtures | at least 20 positive and 10 negative meta-predicate fixtures |
| M085_META.6 GPU route | accepted `findall`/`maplist` programs lower to relational execution |

**Expected targets.**

- `crates/xlog-logic/src/ast.rs`
- `crates/xlog-logic/src/parser.rs`
- `crates/xlog-logic/src/compile.rs`
- `crates/xlog-logic/src/lower.rs`
- `crates/xlog-logic/tests/test_v085_meta.rs`
- `examples/v085-language/meta/`

**Definition of Done.**

- No dynamic Prolog database or arbitrary dynamic call is introduced.
- `findall` and `maplist` examples run through normal XLOG execution.
- Diagnostics explain why unsupported meta forms are unsafe.

### G085_NAF - Explicit Negation-As-Failure Semantics

**Goal.** Make negation-as-failure explicit for the purpose of avoiding ambiguity between deterministic stratified NAF and probabilistic WFS/nonmonotone semantics.

**Questions.**

- Q085_NAF.1: What does `not atom` mean in deterministic XLOG?
- Q085_NAF.2: How does deterministic NAF differ from probabilistic WFS and MC nonmonotone semantics?
- Q085_NAF.3: Can unsafe or unstratified deterministic negation fail before execution?
- Q085_NAF.4: Do new list/meta terms preserve negation safety checks?

**Metrics.**

| Metric | Target |
|---|---|
| M085_NAF.1 docs | deterministic NAF and probabilistic WFS sections exist and are cross-linked |
| M085_NAF.2 safety tests | at least 8 unsafe/unstratified negation fixtures rejected |
| M085_NAF.3 compatibility | existing `not atom` programs keep current stratified behavior |
| M085_NAF.4 explicit config | any new pragma or syntax is documented and source-tested |
| M085_NAF.5 prob separation | MC/WFS tests still distinguish nonmonotone probabilistic semantics |

**Expected targets.**

- `crates/xlog-logic/src/stratify.rs`
- `crates/xlog-logic/src/parser.rs` if syntax changes
- `crates/xlog-prob/src/mc/`
- docs and tests

**Definition of Done.**

- Deterministic NAF has a precise closed-world, stratified contract.
- Probabilistic WFS remains separate.
- v0.9.0 epistemic `not know` work is not preempted.

### G085_MAGIC - Magic-Set Transformation

**Goal.** Add a magic-set rewrite pass for the purpose of specializing bound recursive queries and reducing irrelevant intermediate tuples, with respect to output equivalence and WCOJ eligibility.

**Questions.**

- Q085_MAGIC.1: Can the compiler detect bound query arguments and compute adornments?
- Q085_MAGIC.2: Does SIPS propagate bindings through recursive dependencies correctly?
- Q085_MAGIC.3: Does the transformed program produce the same output as the original?
- Q085_MAGIC.4: Does WCOJ promotion still happen after magic-set rewriting where eligible?
- Q085_MAGIC.5: Are negation, functor/list/meta, and aggregate interactions either handled safely or declined explicitly?

**Metrics.**

| Metric | Target |
|---|---|
| M085_MAGIC.1 pass control | `#pragma magic_sets = on|off|auto` or equivalent plus CLI override |
| M085_MAGIC.2 positive recursion | transitive-closure bound-query fixtures pass output equivalence |
| M085_MAGIC.3 row reduction | at least 50 percent intermediate-row reduction on one certified bound recursive fixture |
| M085_MAGIC.4 WCOJ preservation | WCOJ-eligible transformed fixtures still dispatch through WCOJ |
| M085_MAGIC.5 explain output | `xlog explain` shows adornments, magic predicates, and declined reasons |
| M085_MAGIC.6 decline safety | unsupported negation/aggregate/meta interactions fail or decline with typed reason |

**Expected targets.**

- `crates/xlog-logic/src/optimizer/magic_sets.rs` or equivalent
- `crates/xlog-logic/src/compile.rs`
- `crates/xlog-logic/src/stratify.rs`
- `crates/xlog-logic/tests/test_v085_magic_sets.rs`
- integration tests with runtime stats

**Definition of Done.**

- Magic sets are source-level/RIR-compatible rewrites, not a runtime side engine.
- Row-reduction evidence includes raw before/after counts.
- Explain output is stable enough for docs and tests.

### G085_PROB_AGG - Probabilistic Aggregate Support

**Goal.** Add aggregate support in probabilistic programs for the purpose of allowing exact and approximate probabilistic reasoning over grouped derived values.

**Questions.**

- Q085_PROB_AGG.1: Which aggregate forms are exact-supported: `count`, `sum`, `min`, `max`, `logsumexp`?
- Q085_PROB_AGG.2: Can exact provenance represent finite aggregate outcomes without CPU-only recomputation?
- Q085_PROB_AGG.3: Can MC inference evaluate aggregate rules through the GPU deterministic core?
- Q085_PROB_AGG.4: Can evidence and queries reference aggregate outputs?
- Q085_PROB_AGG.5: What caps and diagnostics govern exact aggregate state explosion?

**Metrics.**

| Metric | Target |
|---|---|
| M085_PROB_AGG.1 exact support | exact mode supports finite `count`, `sum`, `min`, `max`, and `logsumexp` fixtures or documents a typed decline for a mathematically unsupported case |
| M085_PROB_AGG.2 provenance blocker | supported aggregate rules no longer hit the current generic "Provenance extraction does not support aggregation" error |
| M085_PROB_AGG.3 oracle parity | exact aggregate probabilities match finite oracle within `1e-9` for deterministic probabilities |
| M085_PROB_AGG.4 MC support | MC aggregate fixtures run on GPU and match exact/oracle within confidence interval |
| M085_PROB_AGG.5 evidence/query | at least one evidence fixture and one query fixture reference aggregate output |
| M085_PROB_AGG.6 cap diagnostics | exact-domain cap exceeded returns typed diagnostic with remediation |

**Expected targets.**

- `crates/xlog-prob/src/provenance.rs`
- `crates/xlog-prob/src/pir.rs`
- `crates/xlog-prob/src/exact.rs`
- `crates/xlog-prob/src/mc/`
- `crates/xlog-prob/tests/test_v085_prob_aggregates.rs`

**Definition of Done.**

- Exact-supported probabilistic aggregate programs run through the GPU exact/provenance path.
- MC-supported aggregate programs run through GPU sampling plus deterministic aggregate execution.
- Unsupported large-domain exact forms fail before misleading results are produced.

### G085_AGG_LIFT - Small-Domain Aggregate Lifting

**Goal.** Add small-domain aggregate lifting for the purpose of avoiding naive outcome enumeration when aggregate domains are finite and compact.

**Questions.**

- Q085_AGG_LIFT.1: Can finite aggregate domains be detected from schema, facts, or explicit caps?
- Q085_AGG_LIFT.2: Can lifted count/sum/min/max/logsumexp results match exact enumeration?
- Q085_AGG_LIFT.3: Does lifting reduce compile or evaluation cost on certified small-domain fixtures?
- Q085_AGG_LIFT.4: Are fallback and decline paths explicit?

**Metrics.**

| Metric | Target |
|---|---|
| M085_AGG_LIFT.1 domain detection | finite domain metadata recorded in explain JSON |
| M085_AGG_LIFT.2 operator coverage | count, sum, min, max, and logsumexp covered or explicitly declined with per-operator rationale |
| M085_AGG_LIFT.3 numeric parity | integer aggregates exact; floating/logsumexp within `1e-9` absolute or relative tolerance |
| M085_AGG_LIFT.4 speed/cost | at least 1.5x improvement over naive enumeration on one certified fixture, or documented no-regression if fixture is too small |
| M085_AGG_LIFT.5 cap safety | cap exceedance yields typed diagnostic |

**Expected targets.**

- `crates/xlog-prob/src/`
- optional CUDA/provider kernels only if existing exact path cannot express required DP/histogram operations
- tests and evidence under v085 aggregate-lift path

**Definition of Done.**

- Lifting is used only when semantics are identical to exact finite aggregation.
- Explain/cert output says when lifting fired, declined, or fell back to supported exact enumeration.

### G085_APPROX - Approximate Inference Language Integration

**Goal.** Promote approximate inference to a first-class language and CLI feature for the purpose of making MC inference configurable, reproducible, and reportable.

**Questions.**

- Q085_APPROX.1: Can source pragmas configure MC samples, seed, confidence, and sampling method?
- Q085_APPROX.2: Does CLI output report confidence intervals and sample/evidence counts?
- Q085_APPROX.3: Are approximate results deterministic under fixed seed?
- Q085_APPROX.4: Do aggregate probabilistic programs work in approximate mode?

**Metrics.**

| Metric | Target |
|---|---|
| M085_APPROX.1 pragmas | source-level configuration for samples, seed, confidence, max nonmonotone iterations, and method |
| M085_APPROX.2 CLI parity | CLI flags override or merge with pragmas according to documented precedence |
| M085_APPROX.3 deterministic replay | 100/100 fixed-seed MC replays produce identical counts |
| M085_APPROX.4 confidence reporting | pretty/json/csv output includes probability, stderr, CI low/high, sample count, evidence count |
| M085_APPROX.5 aggregate fixture | approximate probabilistic aggregate example passes |

**Expected targets.**

- `crates/xlog-logic/src/grammar.pest`
- `crates/xlog-logic/src/ast.rs`
- `crates/xlog-cli/src/main.rs`
- `crates/xlog-prob/src/mc/`
- CLI and prob tests

**Definition of Done.**

- Approximate inference is not just an internal API; it is documented as language and CLI behavior.
- Fixed-seed reproducibility is certified.

### G085_INC_PARSE - Incremental Parsing And Session Cache

**Goal.** Add incremental parsing for the purpose of supporting REPL, watch mode, and explain workflows without reparsing unchanged source unnecessarily.

**Questions.**

- Q085_INC_PARSE.1: Can source be split into statement units with stable spans/hashes?
- Q085_INC_PARSE.2: Can unchanged statements reuse parse results?
- Q085_INC_PARSE.3: Can diagnostics preserve source spans after edits?
- Q085_INC_PARSE.4: Does the cache invalidate correctly across modules?

**Metrics.**

| Metric | Target |
|---|---|
| M085_INC_PARSE.1 cache unit | statement-level parse cache implemented |
| M085_INC_PARSE.2 invalidation | edit to one statement invalidates only that statement plus dependent module state |
| M085_INC_PARSE.3 speed | at least 2x parse-time improvement on a synthetic 500-statement one-line edit fixture |
| M085_INC_PARSE.4 diagnostics | parser errors still include line/column/span |
| M085_INC_PARSE.5 tests | cache hit, miss, invalidation, module, and error fixtures pass |

**Expected targets.**

- `crates/xlog-logic/src/parser.rs`
- new parser session/cache module
- `crates/xlog-cli/src/main.rs`
- tests under `crates/xlog-logic/tests/`

**Definition of Done.**

- REPL/watch/explain can share the parser cache.
- Cache correctness is tested before performance claims.

### G085_CLI - REPL, Watch, And Explain

**Goal.** Add CLI developer-experience commands for the purpose of making the language inspectable and interactive.

**Questions.**

- Q085_CLI.1: Can users run an interactive `xlog repl` session?
- Q085_CLI.2: Can `xlog watch` rerun source files on change?
- Q085_CLI.3: Can `xlog explain` show parse, strata, RIR, optimized RIR, magic-set, WCOJ, and probabilistic plans?
- Q085_CLI.4: Are explain outputs stable enough for JSON consumers and docs?

**Metrics.**

| Metric | Target |
|---|---|
| M085_CLI.1 commands | `xlog explain`, `xlog repl`, and `xlog watch` subcommands implemented |
| M085_CLI.2 explain formats | text, json, and dot supported or unsupported formats rejected before execution |
| M085_CLI.3 explain content | AST summary, stratification, RIR, optimizer, magic-set, WCOJ eligibility, and prob engine sections present when applicable |
| M085_CLI.4 REPL smoke | multiline fact/rule/query session works and preserves relation state |
| M085_CLI.5 watch smoke | file-change rerun works with debounce and typed error display |
| M085_CLI.6 dependency audit | any new CLI dependency is justified in evidence |

**Expected targets.**

- `crates/xlog-cli/src/main.rs`
- `crates/xlog-logic/` explain APIs
- `crates/xlog-ir/` display/json helpers if needed
- `crates/xlog-cli/tests/`
- docs/examples

**Definition of Done.**

- CLI commands have source-level tests and at least one end-to-end smoke test each.
- Explain JSON is deterministic under fixed input.
- REPL/watch do not require GPU until the user runs a program that needs execution.

### G085_EXAMPLES - Advanced .xlog Examples And Validators

**Goal.** Add advanced examples for the purpose of demonstrating every v0.8.5 language feature as executable `.xlog`, not just reference prose.

**Questions.**

- Q085_EXAMPLES.1: Is every v0.8.5 feature demonstrated by at least one runnable `.xlog` program?
- Q085_EXAMPLES.2: Do examples cover interactions, not only isolated toy cases?
- Q085_EXAMPLES.3: Does the validator execute examples and record raw output/evidence?
- Q085_EXAMPLES.4: Are DTS-DLM-style and general scientific/engineering examples expressed without domain-specific legacy terminology?

**Metrics.**

| Metric | Target |
|---|---|
| M085_EXAMPLES.1 example count | at least 10 advanced examples |
| M085_EXAMPLES.2 feature coverage | 100 percent of G085 feature nodes represented |
| M085_EXAMPLES.3 interaction coverage | at least 5 examples combine two or more new features |
| M085_EXAMPLES.4 validator | `scripts/validate_v085_examples.py` or equivalent passes all examples |
| M085_EXAMPLES.5 evidence JSON | validation summary committed under `docs/evidence/<date>-v085-examples/` |

**Required example families.**

- list processing and typed relation lowering;
- `findall` plus aggregate query;
- `maplist` over static predicate reference;
- magic-set recursive bound query with explain output;
- probabilistic aggregate exact inference;
- probabilistic aggregate MC inference;
- aggregate lifting small-domain fixture;
- approximate inference with confidence reporting;
- REPL/watch/explain smoke package;
- DTS-DLM-style incremental reasoning fixture using scientific/engineering predicate names.

**Definition of Done.**

- Every example has `program.xlog`, optional input data, expected output, and README notes.
- Validator runs examples and checks expected output or expected diagnostics.
- Examples are executable from repository root.

### G085_INT - Integration, Regression, And Certification

**Goal.** Validate the composed v0.8.5 feature pack for the purpose of proving language, runtime, probability, CLI, and compatibility behavior are production-grade.

**Questions.**

- Q085_INT.1: Do all touched crates format, check, and test?
- Q085_INT.2: Do parser, lowerer, optimizer, runtime, prob, CLI, pyxlog, and examples remain compatible?
- Q085_INT.3: Are GPU-native claims backed by source audits and runtime evidence?
- Q085_INT.4: Are all metrics from G085_PRE through G085_EXAMPLES complete?

**Metrics.**

| Metric | Target |
|---|---|
| M085_INT.1 format | `cargo fmt --check` passes |
| M085_INT.2 logic tests | `cargo test -p xlog-logic` passes |
| M085_INT.3 prob tests | `cargo test -p xlog-prob` passes |
| M085_INT.4 cli tests | `cargo test -p xlog-cli` passes |
| M085_INT.5 runtime/integration | targeted `xlog-runtime`, `xlog-integration`, and WCOJ tests pass |
| M085_INT.6 examples | v085 validator passes all examples |
| M085_INT.7 source audit | no accepted feature routes through an unapproved CPU-only execution path |
| M085_INT.8 v0.8.0 compatibility | v080 DTS example/source guards still pass |
| M085_INT.9 docs | `git diff --check`, JSON validation, and stale-placeholder scans pass |

**Expected commands.**

```bash
cargo fmt --check
cargo check -p xlog-logic
cargo check -p xlog-prob
cargo check -p xlog-cli
cargo test -p xlog-logic
cargo test -p xlog-prob
cargo test -p xlog-cli
cargo test -p xlog-runtime
cargo test -p xlog-integration
python3 scripts/validate_v085_examples.py --output /tmp/v085_examples_validation_summary.json
git diff --check
```

If CUDA is unavailable, GPU-required certs must be marked BLOCKED with exact environment diagnostics and rerun on a CUDA-capable machine before closure.

**Definition of Done.**

- Every required command has raw output summarized in evidence.
- GPU certs are actually run before closure; a local no-CUDA skip is not closure.
- Worktree is clean after the final commit.

### G085_CLOSE - Evidence, Roadmap Sync, And Closure Proposal

**Goal.** Produce a closure proposal for the purpose of letting the coordinator decide whether v0.8.5 can merge, with respect to the full GQM tree and evidence trail.

**Questions.**

- Q085_CLOSE.1: Are all G085 sub-goals complete with raw metrics?
- Q085_CLOSE.2: Does ROADMAP move the deferred backlog into v0.8.5 and adjust v0.9.0/v0.10.0 ordering as approved?
- Q085_CLOSE.3: Does CHANGELOG describe user-facing language changes and migration notes?
- Q085_CLOSE.4: Are known unsupported forms documented?
- Q085_CLOSE.5: Is v0.9.0 rebase guidance included?

**Metrics.**

| Metric | Target |
|---|---|
| M085_CLOSE.1 metric table | all M085_* metrics marked PASS, WAIVED with reason, or BLOCKED with exact blocker |
| M085_CLOSE.2 roadmap | ROADMAP has explicit v0.8.5 section and no stale "deferred" copy of completed items |
| M085_CLOSE.3 changelog | CHANGELOG contains v0.8.5 entry |
| M085_CLOSE.4 closure proposal | `docs/plans/<date>-v085-closure-proposal.md` created |
| M085_CLOSE.5 worktree | clean final status |
| M085_CLOSE.6 no unauthorized actions | no push, tag, merge, or board update unless separately authorized |

**Definition of Done.**

- Closure proposal links all evidence packages.
- Closure proposal includes branch SHA, test command matrix, metric table, known limitations, and v0.9.0 rebase note.
- Agent stops for coordinator approval after closure proposal.

## 7. Release KPIs

| KPI | Definition | Gate |
|---|---|---|
| KPI085.1 backlog closure | all 11 promoted roadmap items have implementation, docs, tests, examples, and evidence | 100 percent |
| KPI085.2 GPU-native execution | accepted runtime/prob/list/meta/magic features use production GPU paths where execution occurs | no unapproved CPU hot path |
| KPI085.3 compatibility | existing v0.8.0 DTS-DLM/pyxlog examples and relevant tests remain green | 100 percent pass |
| KPI085.4 language reference | language reference matches shipped behavior | no stale release header, no false support claims |
| KPI085.5 probabilistic aggregate correctness | exact/MC aggregate fixtures match oracle or confidence interval | 100 percent pass |
| KPI085.6 magic-set value | certified bound recursive fixture reduces intermediate rows | >= 50 percent row reduction on at least one fixture |
| KPI085.7 CLI usability | explain/repl/watch all function on documented examples | 100 percent pass |
| KPI085.8 v0.9.0 readiness | parser/AST/prob changes documented for v0.9.0 rebase | handoff section present |

## 8. Implementation Protocol

Start:

```bash
git worktree add .worktrees/v085-language -b feat/v085-language-completeness main
cd .worktrees/v085-language
git status --short --branch
```

Then:

1. Read this document, `ROADMAP.md`, `docs/language-reference.md`, and the v0.8.0/v0.9.0 agent goal documents.
2. Create `docs/evidence/<date>-v085-pre/README.md` with baseline state and ownership map.
3. Execute sub-goals in order unless a dependency requires a small documented reorder.
4. Use short commits by sub-goal. Each commit message should reference the G085 node, for example `feat(v085): add finite list terms`.
5. After each sub-goal, update evidence with raw metrics and commands.
6. Do not push, tag, merge, or update release boards from this branch without explicit coordinator authorization.

## 9. Stuck Conditions

Agent C must stop and report rather than silently narrowing scope if any condition occurs:

- A requested accepted feature requires an arbitrary CPU Prolog term heap.
- Probabilistic aggregate exact semantics cannot be made finite without changing source semantics.
- Magic-set rewriting changes observable output.
- GPU-required certification cannot run because CUDA is unavailable.
- A new dependency creates license, build, or packaging risk.
- v0.8.0 compatibility breaks and cannot be fixed within v0.8.5 scope.
- v0.9.0-owned files must be changed in a way that conflicts with active v0.9.0 work.

## 10. Final Closure Package

The final closure proposal must include:

- branch and final commit SHA;
- complete GQM metric table;
- evidence directory index;
- raw command matrix with exit codes;
- example validation summary;
- GPU-native source audit summary;
- known unsupported forms;
- ROADMAP and CHANGELOG summary;
- v0.9.0 rebase guidance;
- explicit statement that no push, tag, merge, or board update was performed unless separately authorized.

**End of goal document.** Agent C begins with G085_PRE after creating `.worktrees/v085-language`. Coordinator awaits baseline report before accepting any closure claim.
