# v0.8.5 Language Architecture Contract

**Status:** implementation snapshot through `G085_PROB_AGG`.
**Branch:** `feat/v085-language-completeness`.
**Scope:** language, parser, term, probabilistic, and CLI surfaces required by
the v0.8.5 language-completeness release.

v0.8.5 is a language-surface expansion over the v0.8.0 runtime. It must reuse
the production parser, typed AST, semantic normalization, RIR, probabilistic IR,
optimizer, runtime, WCOJ, and CLI paths. It is not a second runtime and it is
not a Prolog interpreter.

## Pipeline Contract

```text
source syntax
  -> statement parser / incremental parser cache
  -> typed AST
  -> semantic normalization and desugaring
  -> existing RIR, probabilistic IR, optimizer, runtime, WCOJ, CLI paths
  -> GPU execution where evaluation occurs
```

CPU work is limited to parsing, static semantic analysis, desugaring, planning,
CLI orchestration, diagnostics formatting, and final requested result transfer.
Accepted execution must not be implemented through an arbitrary CPU term heap,
dynamic CPU predicate calls, or hidden CPU-only fallback.

## Term And Type Model

The v0.8.5 term model extends scalar terms with finite high-level terms:

| Term or type | Required representation | Lowering rule |
|--------------|-------------------------|---------------|
| Named predicate columns | Schema metadata on existing scalar columns | Preserve labels through diagnostics and explain output |
| Domain aliases | Named wrapper over existing scalar type | Resolve to scalar layout before runtime |
| `list<T>` | Finite homogeneous list value | Normalize to typed helper relations or compile-time constants |
| `term` | Finite ground term | Admit only bounded terms with a known finite representation |
| `compound` | Functor plus finite argument vector | Lower to typed finite components or reject |
| `predref` | Static predicate reference | Expand at compile time; reject runtime-variable references |

The parser may represent these forms directly, but the lowerer must either
normalize them to finite typed relation layouts or reject them before execution.

## Feature Contracts

| Node | Contract | Implementation boundary |
|------|----------|-------------------------|
| `G085_TYPES` | Extend parser/AST/type metadata for aliases, named columns, lists, finite terms, compounds, and predicate references | Parser and semantic validation only until a construct has a lowering path |
| `G085_LIST` | Accept finite list literals, safe cons patterns, and finite list built-ins | Desugar to helper relations and normal runtime operators |
| `G085_META` | Support safe `ground`, `var`, `nonvar`, `functor`, `=..`, `findall`, and unary/binary `maplist` | Static expansion and finite collection; no unrestricted `call/N` |
| `G085_NAF` | Make deterministic NAF distinct from probabilistic WFS | Existing `not atom` remains stratified closed-world negation in deterministic programs; compiler-path diagnostics reject unsafe source-order binders and deterministic cycles |
| `G085_MAGIC` | Rewrite bound recursive queries with adornments and magic predicates | Source-level rewrite before stratification/lowering/optimizer for the safe positive-recursion subset; `auto` declines and `on` fails if equivalence cannot be proven |
| `G085_PROB_AGG` | Support finite probabilistic aggregates in exact and MC modes | Exact finite outcome enumeration into provenance/PIR, sharing aggregate operator semantics with MC deterministic aggregate execution |
| `G085_AGG_LIFT` | Lift compact finite-domain aggregate computations | Use only when equivalent to finite exact enumeration |
| `G085_APPROX` | Promote MC configuration and reporting to source and CLI | Source pragmas plus CLI override/merge rules with deterministic fixed-seed replay |
| `G085_INC_PARSE` | Cache statement-level parses with spans and invalidation | Shared by explain, REPL, and watch |
| `G085_CLI` | Add `explain`, `repl`, and `watch` | CLI orchestration; GPU required only when execution is requested |

## Unsupported Forms

The following forms are deliberately outside v0.8.5:

- arbitrary Prolog `cut`, `assert`, `retract`, IO predicates, and dynamic
  database mutation;
- unrestricted `call/N` or runtime-variable predicate dispatch;
- open-ended or cyclic list generation;
- recursive compound terms without a finite representation;
- magic-set rewrites whose output equivalence cannot be proven;
- probabilistic aggregate exact paths whose finite domain exceeds the accepted
  cap;
- CPU-only accepted execution for any feature that claims GPU support.

Unsupported forms must fail with typed diagnostics that include feature area,
source span where available, reason, and remediation.

## Explain And Handoff Requirements

`xlog explain` is the shared observability surface for v0.8.5 and the v0.9.0
epistemic/solver branch. JSON explain output should include stable sections for:

- parse and AST summary;
- schema and named-column metadata;
- strata and deterministic negation safety;
- normalized RIR and optimized RIR;
- magic-set adornments, generated predicates, and declined reasons;
- WCOJ eligibility and selected execution path;
- probabilistic engine, aggregate, lifting, and approximate-inference metadata;
- parser-cache hit/miss/invalidation statistics when run from REPL/watch.

The v0.9.0 branch may depend on the parser/AST ability to preserve source spans,
finite term shapes, predicate references, and explain JSON section names. It may
not assume v0.8.5 implements epistemic EIR, world views, GPT, or solver
semantics.

## Source-Audit Snapshot

Current implementation status through `G085_INC_PARSE`:

- `docs/language-reference.md` has been updated to the v0.8.5 contract and now
  records the shipped `G085_LIST`, `G085_META`, `G085_NAF`, `G085_MAGIC`, and
  `G085_PROB_AGG` subsets.
- `crates/xlog-logic/src/grammar.pest` accepts finite list literals, cons
  patterns, compound terms, named predicate columns, and `list<T>` declarations.
- `crates/xlog-logic/src/ast.rs` represents list, cons, compound, and static
  predicate-reference terms.
- `crates/xlog-logic/src/list_normalize.rs` normalizes accepted finite list
  literals and list built-ins to scalar list identifiers plus
  `__xlog_list_*` helper relations before stratification/lowering.
- `crates/xlog-logic/src/meta_normalize.rs` normalizes accepted finite
  `ground`, `var`, `nonvar`, `functor`, `=..`, `findall`, and unary/binary
  `maplist` forms into scalar term IDs and `__xlog_meta_*` helper relations
  before list normalization, stratification, and lowering.
- `crates/xlog-logic/src/compile.rs` validates deterministic `not atom`
  source-order safety after meta/list normalization and before stratification,
  then maps compiler-path deterministic negation cycles to typed
  `v0.8.5 naf error` diagnostics.
- `crates/xlog-logic/src/magic_sets.rs` parses and applies
  `#pragma magic_sets = auto|on|off` for safe bound deterministic recursive
  queries. It emits `__xlog_magic_*` seed and propagation rules, rewrites
  recursive rules through the same AST/RIR path, and declines or fails closed
  for unsafe negation, aggregate, meta/list-helper, mutual-recursion, and
  unsupported SIPS cases.
- `xlog-gpu` stores the normalized program so helper relation facts are loaded
  through the normal relation-store path.
- `crates/xlog-prob/src/provenance.rs` accepts finite probabilistic aggregate
  rules for `count`, `sum`, `min`, `max`, and `logsumexp` by enumerating exact
  aggregate outcomes into PIR formulas. The exact cap is 16 uncertain
  contributing rows per group; exceeding it reports a typed
  `v0.8.5 prob_aggregate error` with MC/domain-reduction remediation.
- `crates/xlog-prob/src/provenance.rs` also records aggregate-lifting metadata.
  Count-only probabilistic aggregate heads over compact finite domains use exact
  cardinality dynamic programming up to 64 uncertain rows per group and report
  `fired`. `sum`, `min`, `max`, and `logsumexp` report
  `fallback_exact_enumeration` while preserving the finite exact enumeration
  path.
- `crates/xlog-prob/src/aggregates.rs` centralizes aggregate operator state so
  exact provenance enumeration and MC deterministic aggregate execution share
  `count`, `sum`, `min`, `max`, and `logsumexp` semantics.
- `crates/xlog-cli/src/main.rs` now exposes `run`, `prob`, and a minimal
  `explain` command for magic-set status/report rendering in text/json/dot
  formats. JSON/text explain output includes aggregate-lifting metadata for
  accepted probabilistic aggregate programs.
- `crates/xlog-logic/src/grammar.pest` and `ast.rs` carry MC source pragmas for
  samples, seed, confidence, sampling method, and nonmonotone iteration caps.
  `xlog-prob::mc::McEvalConfig::from_directives` maps those directives into the
  existing MC engine, and `xlog prob` applies CLI overrides field-by-field.
- `xlog prob --output json` emits MC probability, stderr, CI, sample/evidence
  counts, seed, confidence, and sampling method. Pretty/csv/arrow MC batches
  include the same reporting columns.
- `crates/xlog-logic/src/incremental_parse.rs` provides `ParserSession` for
  statement-level parse reuse with spans, cache hit/miss/invalidation stats,
  module invalidation, and original-span parse diagnostics. `xlog explain`
  parses through this session API so REPL/watch can share it in `G085_CLI`.
- Derived-goal `findall`, non-literal `maplist` inputs, REPL/watch, and full
  CLI explain plan rendering remain gated to later `G085_*` nodes.

Each later implementation node must update this snapshot or link its evidence
when it turns contract text into accepted source behavior.
