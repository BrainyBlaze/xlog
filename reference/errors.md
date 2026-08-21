# Errors and diagnostics

XLOG's typed error surface: XlogError variants, CLI exit codes, the fail-closed rejections you may hit, and how they map to Python exceptions.

XLOG fails closed: when a program asks for something the engine cannot run
soundly on the device, the engine stops with a typed error that names the rule
it violated. It does not silently fall back to a slower or semantically
different path. This page catalogs the error types, the exit codes, and the
specific rejections you are most likely to encounter.

## Error types

Every Rust-level error is a variant of a single enum, `XlogError` (defined in
`crates/xlog-core/src/error.rs`). The enum is marked non-exhaustive, so new
variants may be added without a breaking change.

| Variant | Meaning |
|---|---|
| `Parse` | Parse error from the Datalog frontend. |
| `StratificationCycle` | Stratification failed: a cycle through negation, reported with the predicates involved. |
| `UnsafeVariable` | Domain safety violation: a variable is not bound in any positive body literal. |
| `ResourceExhausted` | GPU memory budget exceeded; carries the operation context, the estimated bytes, and the budget. |
| `CompileCapacityExceeded` | The knowledge-compilation phase (the step, named D4, that turns the program into a compiled Boolean circuit) declined the input: the Boolean formula it was asked to compile — a formula in conjunctive normal form, or CNF — was too large to compile safely. Compiling it anyway would overrun the fixed-capacity output buffers and trigger a CUDA launch error that leaves the GPU unusable for the rest of the process. Means "too big to compile"; catchable; distinct from the verify-phase signal below. Available since 0.10.0. |
| `VerifyBudgetExceeded` | The GPU verifier that checks whether two formulas are equivalent declined. Its conflict-driven SAT search (CDCL) ran out of the per-verify conflict budget before reaching a definite answer. An indeterminate result is never trusted as a proof. Available since 0.10.0. |
| `Kernel` | GPU kernel launch or execution error. |
| `Type` | Type checking or inference error. |
| `Compilation` | Compilation pipeline error (also carries the resident MC rejections, the exact aggregate cap declines, and the cross-predicate type mismatch described below). |
| `UnsupportedEpistemicConstruct` | An epistemic construct known to the frontend but unsupported in the given context; names the construct and the context. Every construct name is listed under [Epistemic rejections](#epistemic-rejections). |
| `Execution` | Runtime execution error. |

## Exit codes

The `xlog` CLI returns `0` on success and `1` on any error. Every failure — parse,
compile, execution, I/O, or an exhausted memory budget — surfaces as exit code `1`
with a descriptive message on stderr. There are no other exit codes.

## Fail-closed rejections you may hit

### Module import rejection

Module resolution completes before deterministic or probabilistic compilation. It
reports these typed diagnostics rather than compiling a program with missing imported
behavior:

| Code | Trigger |
|---|---|
| `E0400` | A direct or transitive `use` path cannot be resolved to a module file. |
| `E0401` | Direct or transitive imports form a cycle. |
| `E0402` | Separate resolved import branches define the same function, or a module or entry program redefines an imported function. |
| `E0403` | A `use` import selects a predicate the source module has not declared public. The message reads `error[E0403]: cannot import private predicate ... from ...`, naming the predicate and its source module. Make the predicate public in its own module, or import something else. |
| `E0404` | A selective import names an item that the module does not export. |
| `E0405` | An imported module contains program-level content that must be declared in the entry file: probabilistic facts, annotated disjunctions, evidence, integrity constraints, neural predicate declarations, or learnable rule templates. |
| `E0406` | An exported rule or function depends on a private item or an item omitted by a selective import. |
| `E0407` | A context-free library request refers to a logical module path that identifies more than one loaded source file. Load the entry file or root module before validating or merging imports. |
| `E0408` | A declaration in the entry program or a selected public declaration in its resolved imports conflicts with another participating declaration for the same predicate in arity, column names, or resolved types. |
| `E0409` | The entry program or imported modules declare one domain alias name with different scalar types. |
| `E0410` | An imported module defines an exported function name more than once. |
| `E0411` | An imported module declares the same predicate as both public and private. |
| `E0412` | Clauses from different source programs—either the entry program and a selected import, or two selected imports—supply conflicting inferred head-column types for the same undeclared predicate name-and-arity signature. Constants, head variables typed by ordinary body atoms or built-in arithmetic bindings, and aggregate result types provide evidence; unanchored variables do not. |
| `E0413` | A clause contains invalid built-in arithmetic or aggregate type evidence while module resolution is inferring an undeclared predicate signature. The diagnostic identifies the predicate, module, and source file. User-defined function calls are expanded after imports merge and are validated during compilation. |

The entry file is loaded from the exact path supplied to the CLI. Imported module paths
continue to resolve to `.xlog` files. Queries and probabilistic queries in imported
modules are entry-file-scoped and are not merged; imported pragmas are ignored with
`warning[W0510]`. See [Modules](/language-guide/modules) for examples and visibility
rules.

### Function expansion rejection

User-defined functions expand after module resolution. Production compilation registers
function declarations in source order, then expands only definitions reachable from calls
in ordinary rules and constraints. An unused definition whose body is recursive, calls an
undefined function, or shares a name with a predicate does not block that demand-driven
path.

Rust callers can instead request strict whole-program validation with
`FunctionRegistry::from_program`. That surface inspects every definition in declaration
order and every callee in source order, including definitions that production expansion
would not reach.

| Code | Trigger |
|---|---|
| `E0501` | More than one function declaration uses the same name. |
| `E0502` | Strict `FunctionRegistry::from_program` validation finds a recursive strongly connected component (SCC) in which no function has a conditional body. The demand-driven production path does not perform this whole-program rejection; a reachable cycle is bounded by `E0504` instead. |
| `E0503` | Production expansion reaches a call to an undefined non-built-in function, or strict registry validation finds such a call in any definition, including an unused one. |
| `E0504` | Expansion tries to enter a user-defined call while already at the depth set by `#pragma max_recursion_depth`. Expansion visits both conditional branches, so a reachable recursive cycle can produce this diagnostic even when its source contains a conditional base form. |
| `E0505` | Strict registry validation finds a name used by both a function and a predicate. This whole-program check is not part of demand-driven production expansion. |
| `E0508` | A function call supplies a different number of arguments than its declaration. |
| `E0509` | The Rust library's expression-only `ExpansionContext::expand_call` surface tries to expand a predicate-bodied function without an ordinary rule or constraint body to receive its relational literals. CLI compilation expands calls in their rule or constraint context, so it does not emit this code. |
| `E0510` | An arithmetic expression other than a variable or numeric literal would have to occupy a term position in a predicate body. Bind the arithmetic result first and pass its variable. |
| `E0511` | A predicate-bodied call appears in a conditional result branch, where inserting its relational body unconditionally would change guarded semantics. |
| `E0512` | A non-variable argument would have to become the target of an `is` binding inside a predicate body. |

`E0506` and `E0507` are unassigned.

Strict validation accepts a recursive SCC when at least one member has a conditional
body. That is a structural validation result, not a runtime termination guarantee:
function normalization expands both branches eagerly, and a used cycle can still reach
`E0504` at the configured depth.

### Cross-predicate type mismatch

Compilation rejects a rule whose variables draw incompatible column types from
the declared schemas of the predicates they touch. The conflict may be between
two body atoms, or between a body atom and the head. Both forms are
`Compilation` errors that name the rule, the variable, both types, and where each
came from:

```text
Type mismatch in rule for '<head>': variable <V> is <Type> (from <predicate/position>) but <Type> is required by <predicate/position>
Type mismatch in rule for '<head>': variable <V> is <Type> (from <predicate/position>) but <head> declares <Type> at position <N>
```

The check applies only to predicates that carry an explicit `pred` declaration;
an undeclared predicate has no schema to contradict. Available since 0.12.0 —
before that the same program was accepted here and failed later with an internal
GPU schema error that did not name the rule.

### Epistemic rejections

`UnsupportedEpistemicConstruct` is one error type covering many distinct
rejections. The `construct` field in the message tells you which one you hit;
find it in the left column below. (A *modal* literal is one written with `know`
or `possible`. A *world view* is the set of models the program considers possible
at once. A *tuple key* is the argument list that identifies which tuple a modal
literal is talking about.)

| `construct` | Trigger |
|---|---|
| `epistemic tuple key` | A tuple-key epistemic fact whose key contains a variable, `_`, or an aggregate. These facts require ground terms. |
| `world view boundary` | The program admits no stable model at all, so there is no world view to reason over. |
| `epistemic GPU execution plan` | The program declares `know`/`possible`, but after reduction no rule body actually contains a modal literal, so there is nothing to build a world-view plan from. |
| `epistemic GPU world-view constraint` | Four cases, all in epistemic integrity constraints such as `:- know p(X).` — (a) the program has no epistemic rule for the constraint to be checked against; (b) the constraint body mixes ordinary or comparison literals with modal literals, e.g. `:- p(X), know q(X).`; (c) a variable is shared across two modal literals or repeated inside one, e.g. `:- know p(X), possible q(X).` or `:- know p(X, X).`; (d) a key position uses a list, compound, float, or string term. Single-occurrence variable keys like `:- know p(X).` are fine. |
| `recursive epistemic program` | Three cases — a dependency cycle the single-pass epistemic planner cannot iterate; a recursive epistemic program that also carries an epistemic integrity constraint (the recursive path never runs the constraint kernel, so it refuses rather than drop the constraint); or a modal dependency cycle reaching the stratified planner. |
| `epistemic GPU final output relation` | An epistemic program compiled through the single (non-split) GPU path produces more than one distinct output head. Single-plan execution can materialize only one final relation; use split execution for independent epistemic outputs. |
| `epistemic GPU constraint` | Split epistemic execution combined with any epistemic integrity constraint, including forms the single-plan path accepts. |
| `epistemic GPU split execution` | Split execution was selected but no split component contains an epistemic rule — that is, the program has no `know`/`possible` rules at all. |
| `cross-component epistemic coupling` | Two or more epistemic output heads land in one dependency component, and a modal literal in that component ranges over another epistemic-derived head from the same component. A single joint world-view enumeration would mis-evaluate that nesting. |
| `epistemic modal tuple key` | A modal literal's tuple key does not flatten to exactly one scalar term per target column. Use one scalar key term per column of the modal target. New in 0.12.0. |
| `epistemic derived predicate schema` | The same predicate name is defined as a rule-derived epistemic relation at more than one arity. A derived epistemic relation needs exactly one source signature per name. New in 0.12.0. |
| `epistemic augmented predicate schema` | Sibling clauses for one head bind different hidden columns. A variable that appears only inside a modal literal adds hidden columns to the head, so every clause for one predicate signature must bind the same shape. New in 0.12.0. |
| `epistemic augmented head query` | A query against a head that carries those hidden modal-bound columns, where the query's arguments are not all distinct named variables. A constant, `_`, a repeated variable, or a compound term is refused. New in 0.12.0. |
| `epistemic rule-union materialization` | A rule-derived epistemic predicate has more than one defining clause, at least one clause is epistemic, and the modal filters are not provably the same across clauses. Merging the clauses in one pass would lose which clause each row came from, so the modal filter could not be applied correctly. New in 0.12.0. |
| `Gelfond-1991 compatibility cycle through aggregation` | Under `g91` semantics, an aggregate predicate takes part in a positive `possible` compatibility cycle. The tuple-level fixpoint that class needs requires every predicate in the cycle to be monotone, and aggregation is not. New in 0.12.0. |
| `Gelfond-1991 compatibility cycle through negation` | The same cycle, but routed through a negated literal instead of an aggregate, with the same monotonicity requirement. New in 0.12.0. |
| `Gelfond-1991 tuple compatibility ordinary reduction` | A `g91` program whose positive `possible` predicates form an exact tuple-level compatibility cycle reached the one-shot reduction step. Such cycles need the explicit upper-bound fixpoint plan and cannot be rewritten as a single ordinary least-fixpoint program. New in 0.12.0. |

`g91` and `faeel` are the two epistemic semantics you pick between with
`#pragma epistemic_mode`; `faeel` is the default. See
[Epistemic support and boundaries](/reference/language#epistemic-support-and-boundaries)
for the same boundaries stated as language rules, and
[Epistemic reasoning](/epistemic/overview) for what the two modes mean.

### Resident Monte Carlo rejection

The production Monte Carlo engine — which estimates probabilities by sampling
many random possible worlds — runs entirely on the GPU ("resident") within fixed
memory bounds. At compile time it checks every rule and fact against the model of
what the device can run; anything outside that model is rejected with a typed
`ResidentRejection` (surfaced as a `Compilation` error of the form
`resident MC engine rejected program [kind=...] construct=... context=...`).
There is no silent CPU fallback.

| Rejection kind | Trigger |
|---|---|
| `negation` | A body literal uses negation. |
| `epistemic_literal` | A body literal is epistemic (`know` / `possible`). |
| `non_relational_literal` | A body literal is a comparison, arithmetic, or `univ` (non-relational). |
| `arity_too_high` | A predicate arity exceeds the cap of 3. |
| `body_too_long` | A rule body has more than 3 literals. |
| `too_many_vars` | A rule uses more than 8 distinct variables. |
| `unbounded_term` | A term is not a variable or ground constant (list, compound, functor, aggregate). |
| `domain_too_large` | The bounded constant domain exceeds 256. |
| `universe_too_large` | The bounded atom universe exceeds 65536 slots. |
| `inconsistent_arity` | A predicate appears with inconsistent arity. |
| `annotated_disjunction_unsupported` | The program uses an annotated disjunction the resident engine cannot ground. |

<Note>
On the CLI, `xlog prob --allow-cpu-oracle` lets a rejected program run on a labeled
CPU oracle instead; the result is tagged `mc_engine: "cpu-oracle"` and is never
GPU-native evidence. Without the flag, a rejected program fails. See the
[CLI reference](/reference/cli).
</Note>

### Exact aggregate caps

The exact engine (`exact_ddnnf`) computes probabilities exactly rather than by
sampling. It evaluates aggregates over finite probabilistic domains using
dynamic programming, and that evaluation is capped per aggregate group:

- **Count-only aggregates**: at most **64** uncertain rows per group.
- **All other aggregates** (`sum`, `min`, `max`, `logsumexp`): at most **16**
  uncertain rows per group.

Rows whose provenance is deterministically true or false do not count against the
cap — only rows whose membership is genuinely uncertain do. Over the cap, the
compile fails with a `Compilation` error that names the predicate, the group key,
and the cap, and tells you the way out: `use prob_engine = mc or reduce the finite
aggregate domain`. The Monte Carlo engine has no such cap because it samples
worlds instead of enumerating outcome formulas. See
[Probabilistic engines](/probabilistic/engines) for choosing between the two.

### Multiway union size caps

The GPU sort and dedup steps that merge same-head rule outputs index column
bytes with 32-bit arithmetic, so they reject any column whose logical bytes
exceed `4294967295` (`u32::MAX`) with a typed error of the form
`Sort supports at most 4294967295 bytes per column, got ... (column N)`; the
concatenation step applies analogous byte caps, `Concat: total_bytes too large: ...`
and `Concat: col_bytes too large: ...`. Since 0.12.0 the clauses of one head merge
in a single batched concat pass, and that pass additionally rejects more than
4294967295 total rows with `Concat supports at most 4294967295 rows, got ...`.
These are all fail-closed declines, never silent truncation: a relation column
past 4 GiB stops the run instead of scattering rows.

### Compile and verify budgets

These two budgets let the engine refuse an oversized problem cleanly instead of
crashing the GPU. Available since 0.10.0, the knowledge-compilation phase (D4)
and verify-phase controls decline oversized instances rather than risk a CUDA
launch failure that would leave the GPU unusable for the rest of the process:

- A CNF (Boolean formula in conjunctive normal form) whose variable or clause
  capacity exceeds `XLOG_D4_VERIFY_MAX_VARS` / `XLOG_D4_VERIFY_MAX_CLAUSES`
  declines **before any kernel launch** with `CompileCapacityExceeded`. Both
  bounds default to unbounded.
- A verify whose SAT search exhausts `XLOG_D4_VERIFY_MAX_CONFLICTS` declines
  with `VerifyBudgetExceeded` — an indeterminate search result is never
  reported as a proof. The default budget of `0` means unlimited.

Both declines are catchable, and the caller can skip the query or fall back to
the approximate `mc` engine. See
[Environment variables](/reference/environment-variables) for the knobs.

<Warning>
A fail-closed decline is a diagnostic, not a result. It blocks an unsound or
context-poisoning execution and explains why; it does not mean the query was
answered.
</Warning>

## Python exceptions

`pyxlog` maps the Rust error surface onto standard Python exception types:

| Exception | Raised for |
|---|---|
| `ValueError` | Invalid parameters: an unknown `prob_engine` (expected `exact_ddnnf` or `mc`), an unknown `sampling_method` (expected `rejection` or `evidence_clamping`), `memory_mb=0`, row counts exceeding `u32` range, neural input validation, and exporting provenance for an unstored relation. Neural input validation covers tensor shape and value checks, an undersized tensor source (in a complex query each neural atom reads its own source row — atom *k* reads row *k* — so a body with *N* neural atoms needs at least *N* rows in the active source), and `register_network` signature validation: `arg_sorts` passed without `arity`, an `arg_sorts` length that does not equal `arity`, a `bool` or non-int `arg_sorts` element, or an `arity` that disagrees with the `nn/4` declaration bound to that network name in the program. |
| `RelationMetadataError` (a `ValueError` subclass) | Invalid ordered roles, whole-fact provenance, insert evidence, or relation-provenance manifests. |
| `KeyError` | `LogicRelationSession.relation(name)` or `evidence(name)` when the named relation is not currently stored. |
| `BufferError` | A tensor-like DLPack producer reports a non-CUDA device from `__dlpack_device__()`. XLOG raises before calling `__dlpack__()` or consuming a capsule. |
| `MemoryError` | The per-call memory limit: when `memory_mb` is passed to an evaluation and the provider's allocated bytes already exceed it, the call raises before evaluating. |
| `RuntimeError` | Any `XlogError` propagated from the engine (parse, compilation, kernel, execution, resource exhaustion — including the resident MC rejection and aggregate cap messages above), host-output calls on a build without the `host-io` feature, and construction of fallback `RelationEvidence` when `pyxlog._native` is unavailable. |
| `IlpConfigError` (a `ValueError` subclass) | `induce_exact(backend="python")` without `XLOG_ALLOW_PYTHON_ILP_REFERENCE=1`. The Python reference scorer exists only to cross-check results (parity-only) and is never a production path. |
| `CarrierRefused` (a `RuntimeError` subclass) | Every typed refusal from the device-resident joint-constraint solver (`JointConstraintCarrier`) except running out of fuel: registering a schema twice, a zero-capacity dimension, using the carrier before `register_schema`, rebinding already-bound signature masks, a signature-mask shape mismatch, solving before `bind_signatures`, running the top-two stage before feasibility has solved, a malformed component plan, an out-of-range abstain label index, or an unavailable joint-solve kernel. Available since 0.11.0. |
| `SolverResourceExhausted` (a `RuntimeError` subclass) | The joint-constraint solver used up its device fuel budget — a cap on how many search nodes it may expand — during `solve_label_feasibility`. The message is `solver fuel exhausted: spent {fuel_spent} of {fuel_limit} node expansions`. The fuel meter lives in the carrier session, so retrying reproduces the identical refusal instead of making partial progress; raise `fuel_limit` on a new carrier. Available since 0.11.0. |

In a source-only import without `pyxlog._native`, package-level
`RelationMetadataError` and `RelationEvidence` remain importable. The fallback
metadata error still subclasses `ValueError`; only native evidence instances and
session operations require the extension.

## See also

- [CLI reference](/reference/cli) — subcommands, flags, and the exit-code contract
- [Probabilistic engines](/probabilistic/engines) — when to use `exact_ddnnf` vs `mc`
- [Environment variables](/reference/environment-variables) — budgets and kill switches
