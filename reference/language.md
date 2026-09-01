# Language Reference

Comprehensive reference for XLOG syntax, semantics, data types, rules, negation, aggregations, probabilistic and epistemic logic, and the complete grammar.

XLOG is a GPU-native logic programming language: you declare typed facts and
rules (in the Datalog tradition), and XLOG compiles them into GPU execution
paths that run the joins, negation, aggregation, and inference. This page is the
complete reference for the language — every statement form, data type, operator,
built-in, pragma, and grammar rule. Use it to look up an exact detail; each
section is meant to be read on its own.

### Scope and release context

This reference describes XLOG's implemented language surface. It has three
layers:

- **The core language** — facts, rules, recursion, deterministic negation,
  aggregation, and arithmetic.
- **The language-completeness contract** — finite terms and lists, safe
  meta-predicates, deterministic negation, magic sets, probabilistic aggregates,
  approximate inference, and CLI inspectability.
- **The epistemic surface** — the modal operators `know`, `possible`,
  `not know`, and `not possible`, plus `#pragma epistemic_mode = faeel|g91`.
  "Modal" here means the operator talks about the *models* (the possible
  "worlds") a program admits, not about a single fact.

Accepted epistemic programs are compiled into the **EIR** — the epistemic
intermediate representation, the compiler's internal form for `know`/`possible`
programs — and run through the high-level `xlog run` GPU epistemic runtime. An
unsupported epistemic form must fail with a typed diagnostic rather than
silently falling back to CPU evaluation.

Within the epistemic surface, the executor is complete for:

- enumerating candidate worlds from the EIR;
- checking modal membership at the value level, one tuple key at a time;
- **FAEEL** foundedness per tuple key — FAEEL is XLOG's default epistemic
  semantics, and a tuple is "founded" when it has a non-circular justification;
- ground epistemic constraints, safe splitting of independent components, and
  solving several epistemic components jointly.

Its semantic rules are also complete for:

- global and per-row modal gates, which combine with logical AND;
- finite nested modal chains (such as `know possible fact()`), which normalize
  to a single operator by parity/duality;
- predicates that share a name but differ in arity, told apart by their
  arity-qualified tuple sources;
- recursion — covering determined **stratification** (ordered evaluation after a
  modal target is materialized), positive FAEEL dependency cycles through an
  ordinary founded least fixpoint, supported positive exact-tuple Gelfond-1991
  `possible` cycles through a greatest compatibility fixpoint,
  stratified negated modals, and cyclic negated modals. Supported cycles through
  negation use the `xlog-gpu` GPU-backed Well-Founded Semantics (WFS) plan — a
  fixpoint that assigns each atom true, false, or undefined — which replaces the
  older `xlog_prob` host-side WFS solver. A positive compatibility component does
  not itself use WFS, although an unrelated nonmonotone component may use WFS
  within the same upper-bound or refinement pass.

Programs outside the executor's finite-key, relation-identity, tuple-shape, or
materialization contracts fail closed with a diagnostic instead of being
executed. The concrete boundaries are listed under
[Epistemic support and boundaries](#epistemic-support-and-boundaries).

---

## Table of Contents

1. [Overview](#overview)
2. [Basic Syntax](#basic-syntax)
3. [Data Types](#data-types)
4. [Language Completeness Contract](#language-completeness-contract)
5. [Predicates and Declarations](#predicates-and-declarations)
6. [Facts](#facts)
7. [Rules](#rules)
8. [Queries](#queries)
9. [Constraints](#constraints)
10. [Variables and Wildcards](#variables-and-wildcards)
11. [Comparisons](#comparisons)
12. [Arithmetic Expressions](#arithmetic-expressions)
13. [Negation](#negation)
14. [Aggregations](#aggregations)
15. [Lists](#lists)
16. [Safe Meta-Predicates](#safe-meta-predicates)
17. [Magic Sets](#magic-sets)
18. [User-Defined Functions](#user-defined-functions)
19. [Modules](#modules)
20. [Symbols](#symbols)
21. [Probabilistic Logic](#probabilistic-logic)
22. [Epistemic Logic](#epistemic-logic)
23. [Approximate Inference](#approximate-inference)
24. [Neural Predicates](#neural-predicates)
25. [Learnable Rules](#learnable-rules)
26. [Term Embeddings](#term-embeddings)
27. [GPU ILP Configuration](#gpu-ilp-configuration)
28. [Pragmas and Directives](#pragmas-and-directives)
29. [CLI Developer Experience](#cli-developer-experience)
30. [Float Predicates and IEEE 754 Semantics](#float-predicates-and-ieee-754-semantics)
31. [Comments](#comments)
32. [Complete Grammar Reference](#complete-grammar-reference)

---

## Overview

XLOG is a GPU-native logic programming language that compiles typed, modular
logic programs into backend-specific GPU execution paths. It supports:

- **Datalog fundamentals**: Facts, rules, recursive queries, and stratified negation
- **Arithmetic operations**: Comparisons, computed values via `is`, and built-in functions
- **Aggregations**: `count`, `sum`, `min`, `max`, and `logsumexp`
- **Probabilistic reasoning**: Probabilistic facts, annotated disjunctions, evidence, and queries
- **Epistemic reasoning**: `know`/`possible` modal body literals, FAEEL default
  semantics, and Gelfond 1991 compatibility mode through explicit EIR/GPU execution paths
- **Language-completeness contract**: Finite lists, safe meta-predicates, explicit
  negation contracts, magic-set planning, probabilistic aggregate semantics,
  approximate inference configuration, incremental parsing, and CLI
  inspectability
- **GPU execution**: All core operations (joins, sorts, aggregations) run on the GPU

### Hello World Example

```xlog
// Declare predicates with types
pred edge(u32, u32).
pred reach(u32, u32).

// Facts: a directed graph
edge(1, 2).
edge(2, 3).
edge(3, 4).

// Rules: transitive closure (reachability)
reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

// Query: what nodes can 1 reach?
?- reach(1, N).
```

Running this program:

```bash
xlog run program.xlog
```

Output:

```
reach(1, N):
| N |
|---|
| 2 |
| 3 |
| 4 |
```

---

## Basic Syntax

### Program Structure

An XLOG program consists of a sequence of statements. Each statement ends with a period (`.`).

```xlog
// Comments start with //
pred node(u32).        // Predicate declaration
node(1).               // Fact
node(2).
reach(X, Y) :- edge(X, Y).   // Rule
?- reach(1, N).        // Query
:- cycle(X).           // Constraint
```

### Statement Types

| Statement | Syntax | Description |
|-----------|--------|-------------|
| Predicate declaration | `pred name(types).` | Declares a predicate with typed columns |
| Fact | `name(values).` | Asserts a ground tuple |
| Rule | `head :- body.` | Derives new facts from existing ones |
| Query | `?- atom.` | Specifies output relations |
| Constraint | `:- body.` | Asserts that body must have no solutions |
| Function | `func name(params) = expr.` | Defines a user function |
| Use | `use module.` | Imports a module's public deterministic definitions |
| Domain | `domain name: type.` | Declares a domain type |
| Pragma | `#pragma key = value` | Sets compiler/runtime options |

### Identifiers

- **Predicates and atoms**: Start with a lowercase letter, followed by alphanumerics or underscores (`edge`, `reach_from`, `node2`)
- **Variables**: Start with an uppercase letter, followed by alphanumerics or underscores (`X`, `Node`, `Value1`)
- **Anonymous variable**: A single underscore (`_`) matches any value without binding

---

## Data Types

XLOG supports the following scalar types:

| Type | Description | Range |
|------|-------------|-------|
| `u32` | Unsigned 32-bit integer | 0 to 4,294,967,295 |
| `u64` | Unsigned 64-bit integer | 0 to 18,446,744,073,709,551,615 |
| `i32` | Signed 32-bit integer | -2,147,483,648 to 2,147,483,647 |
| `i64` | Signed 64-bit integer | -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 |
| `f32` | 32-bit floating point | IEEE 754 single precision |
| `f64` | 64-bit floating point | IEEE 754 double precision |
| `bool` | Boolean | `true` or `false` |
| `symbol` | Interned string | Reversible `u32` ID with string table |

### Literals

```xlog
// Integers (defaults to i64 in expressions)
42
-17
0

// Floating point (defaults to f64)
3.14
-0.5
1.0

// Boolean
true
false

// String literals (for symbol type)
"hello"
"user_name"
```

### Type Inference

Predicate declarations set column types explicitly. Without a declaration, compilation
infers a schema from available facts and rule information. When different source
programs—either the entry program and a selected import, or two selected imports—contribute
clauses for the same undeclared predicate name and arity, module resolution compares inferred
head-column types before merging them. Head constants, head variables typed by ordinary body
atoms or built-in arithmetic bindings, and aggregate result types supply evidence, including
through rule chains; unanchored variables do not. Conflicts produce `error[E0412]`. Invalid
built-in arithmetic or aggregate evidence produces `error[E0413]` with its source module and
file; user-defined function calls are expanded after imports merge and validated during
compilation.

```xlog
pred temperature(u32, f64).   // sensor_id: u32, temp: f64

temperature(1, 22.5).         // Correct: u32, f64
temperature(1, 22).           // Correct: the declaration converts 22 to f64
temperature(1, invalid).      // Error: a symbol is not valid for f64
```

Within a single program, compilation also rejects a rule whose variables draw
incompatible column types from the available schemas of the predicates they touch
— whether the conflict is between two body atoms or between a body atom and the
head. Schemas inferred from facts and rules participate in the same check as
explicit declarations. The error names the rule, the variable, both types, and
where each came from:

```text
Type mismatch in rule for '<head>': variable <V> is <Type> (from <predicate/position>) but <head> declares <Type> at position <N>
```

An explicit `pred` declaration makes the expected schema stable and visible at the
source boundary. Without one, compilation infers the predicate schema from the
available program evidence and still rejects conflicting type flows.

### Language-Completeness Type Forms

The language-completeness contract extends the type model with finite language-level terms that must lower
to typed relation layouts before execution:

| Type form | Meaning | Execution status |
|-----------|---------|------------------|
| `domain name: type` | Named alias for an existing scalar type | Existing source form; the contract requires alias preservation in diagnostics and schema metadata |
| `column: type` | Named predicate column | Named predicate columns are accepted and preserved in schema metadata |
| `list<T>` | Finite homogeneous list of `T` | Accepted lists lower to helper relations, not CPU term heaps |
| `term` | Finite ground term value | Only finite, typed terms are accepted |
| `compound` | Finite compound term with functor and arity | Unsupported recursive or open compounds are rejected |
| `predref` | Static predicate reference for `maplist` and related meta use | Runtime-variable predicate names are rejected |

Unsupported type forms must fail during parsing or semantic analysis before
runtime execution starts.

---

## Language Completeness Contract

The language-completeness contract is a language-surface expansion over the core
runtime. New syntax is accepted only when it has a typed normalization and
lowering path for every execution route that supports it. Deterministic programs
flow through RIR (XLOG's internal relational intermediate representation), the
optimizer, and runtime; probabilistic and epistemic constructs use their own
typed IR and execution routes. CLI commands report the route that was actually
analyzed rather than implying that every construct traverses every subsystem.

### Execution Responsibilities

| Layer | Allowed work |
|-------|--------------|
| Parser and AST | Recognize source syntax, preserve spans, and represent finite high-level terms |
| Semantic analysis | Type check, safety check, reject unbounded or dynamic forms, and desugar high-level constructs |
| Optimizer and planner | Rewrite programs, including magic sets, while preserving output semantics |
| Runtime, epistemic, and probability engines | Execute accepted relational, epistemic, aggregate, exact, and MC work through production GPU-capable paths |
| CLI | Orchestrate commands, format diagnostics, and transfer only requested final results |

Accepted language-completeness and epistemic features must not execute by constructing
an arbitrary CPU Prolog term heap, running dynamic CPU predicate calls, or
using a hidden CPU-only fallback for a GPU-claimed path.

### Feature Coverage Matrix

| Feature | Syntax | Semantics | Example | Unsupported forms | Execution path |
|---------|--------|-----------|---------|-------------------|----------------|
| Incremental parsing | Statement-level `.xlog` source units with stable spans | Reuse unchanged statement parses and invalidate changed statements plus dependent module state | Editing one rule in a REPL/watch session preserves parse cache for unchanged facts | Cache reuse across changed module dependencies without invalidation | Parser/session cache only; no runtime execution |
| List syntax and built-ins | `[]`, `[A, B]`, `[H\|T]`, `list<T>` | Finite, typed lists normalize to helper relations | `has_member(X) :- member(X, [1,2,3]).` | Open-ended generators, cyclic lists, heterogeneous lists without a declared finite term type | AST/desugar to typed helper relations and normal runtime joins/aggregates |
| Safe meta-predicates | `ground`, `var`, `nonvar`, `functor`, `=..`, `findall`, `maplist` | Static inspection and finite collection only | `xs(L) :- findall(X, edge(1, X), L).` | Unrestricted `call/N`, dynamic database mutation, runtime-variable predicate names | Compile-time/static expansion plus typed relational lowering |
| Deterministic NAF | `not atom(...)` | Closed-world stratified negation over deterministic relations; named variables in the negated atom must be bound by a prior source-order binder | `leaf(X) :- node(X), not edge(X, _).` | Unbound variables in negated atoms, unstratified deterministic cycles | Existing stratification and runtime anti-join paths |
| Magic sets | `#pragma magic_sets = on\|off\|auto` and compiler configuration override | Bound deterministic recursive queries may be rewritten with adornments and magic predicates | `?- reach(1, Y).` specializes recursive reachability | Unsafe interaction with negation, aggregates, meta/list helpers, mutual recursion, or SIPS cases if equivalence cannot be proven | Source/RIR rewrite before optimizer and WCOJ planning |
| Probabilistic aggregates | Aggregate heads and aggregate outputs in `query`/`evidence` | Finite aggregate outcomes in exact and MC inference | `query(out_degree(1, 2)).` over probabilistic `edge` facts | Exact aggregate domains over cap, unsupported numeric operator/domain pairs | Exact provenance/PIR or MC sampling plus deterministic aggregate execution |
| Aggregate lifting | Finite-domain aggregate metadata and caps | Use lifted compact-domain computation when identical to finite exact enumeration | Small count/sum domains avoid naive enumeration | Domain cap exceeded, non-finite domains, unsupported floating tolerance | Probabilistic aggregate planner and exact/MC engines |
| Approximate inference | `#pragma prob_engine = mc`, samples, seed, confidence, method | MC estimates are reproducible under fixed seed and report uncertainty | `#pragma prob_samples = 10000` with `query(rain).` | Invalid confidence ranges, unsupported methods, hidden default override ambiguity | Existing MC engine with documented source/CLI precedence |
| Epistemic literals | `know atom(...)`, `possible atom(...)`, `not know atom(...)`, `not possible atom(...)`, finite nested modal chains, `#pragma epistemic_mode = faeel\|g91` | Modal literals are preserved in EIR. The single-pass route derives candidate worlds from EIR and checks value-level tuple-key membership. Across all routes, the executor normalizes finite nested chains by parity/duality, solves same-name multi-arity by arity-qualified tuple sources, supports determined-head stratification, positive FAEEL modal dependency cycles through founded least-fixpoint reduction, supported Gelfond-1991 positive exact-tuple `possible` cycles through a greatest compatibility fixpoint, stratified negated-modal recursion, and cyclic negated-modal recursion through the `xlog-gpu` GPU-backed WFS plan without the old `xlog_prob` host-WFS solver. Gelfond-1991 refinement and WFS convergence may be host-orchestrated while relation evaluation, snapshots, and set comparison remain GPU-backed. | `accepted() :- know fact().` | Direct raw RIR lowering, genuinely unbounded/untyped modal tuple keys, unsafe unbound negated epistemic variables, CPU-only world-view scans, recursive epistemic programs that also carry modal integrity constraints, and a Gelfond-1991 compatibility component with recursive negation or aggregation | Parser/AST -> EIR -> single-pass epistemic GPU plans, reduced ordinary least-fixpoint plans, descending Gelfond-1991 compatibility plans, or GPU-backed WFS alternating-fixpoint plans through high-level `xlog run` dispatch |
| CLI explain | `xlog explain --format text\|json\|dot file.xlog` | Report parse-cache and AST counts, analysis status and summaries, provenance, proof traces, and generated-row diagnostics | `xlog explain --format json reach.xlog` | Unknown output formats; unsupported analyses report an unavailable status and reason | CLI orchestration plus compiler/explain APIs |
| CLI REPL | `xlog repl` | Read standard input to end-of-file, parse it once, and print a parse summary | Pipe a complete program to `xlog repl` | Compilation, interactive mutation, and query evaluation | Parser session |
| CLI watch | `xlog watch file.xlog` | Re-read and parse a file at a fixed interval; optionally rebuild its explain report | `xlog watch --explain --once file.xlog` | Filesystem-event subscriptions, mutation semantics, and program execution | Parser session and optional explain analysis |

### Diagnostic Contract

Unsupported language-completeness and epistemic forms must report:

- the feature area, such as `list`, `meta`, `naf`, `magic_sets`,
  `prob_aggregate`, or `epistemic`;
- the source span when available;
- the reason the form is unsafe, unbounded, or not GPU-lowerable;
- a remediation, such as adding a positive binder, a finite cap, or a static
  predicate reference.

Diagnostic examples:

```text
list error at program.xlog:12: unbounded append/3 generation is unsupported; bind at least two list arguments or add a finite split mode.
meta error at program.xlog:8: maplist predicate argument must be a static predref; replace runtime variable P with a named predicate.
negation safety error: unbound variable Y in negated atom edge/2; bind it before not with a positive atom or deterministic is expression, or use '_' for existential positions.
magic_sets declined at program.xlog:21: recursive query crosses an aggregate boundary; run with #pragma magic_sets = off or isolate the aggregate.
prob_aggregate error at program.xlog:17: exact probabilistic aggregate domain cap exceeded (16 uncertain rows per group); use prob_engine = mc.
epistemic error at program.xlog:9: unbounded modal tuple key [H|T] has no finite GPU key-column set; use a fixed-arity typed list or bind the tail to a finite relation.
```

### Epistemic support and boundaries

The source language accepts the epistemic literals and the `epistemic_mode`
pragma listed here, and routes accepted epistemic examples through `xlog run`.
The epistemic executor supports:

- candidate enumeration derived from the EIR;
- value-level modal membership;
- per-tuple-key FAEEL foundedness;
- ground and variable-keyed epistemic integrity constraints;
- safe-split equivalence;
- same-name, multi-arity disambiguation;
- finite nested-chain normalization;
- determined-head stratification;
- positive FAEEL modal dependency cycles through founded least-fixpoint reduction;
- supported positive exact-tuple Gelfond-1991 `possible` cycles through a
  greatest compatibility fixpoint;
- stratified negated-modal recursion;
- cyclic negated-modal recursion through the `xlog-gpu` GPU-backed WFS plan,
  which replaces the old `xlog_prob` host-side WFS solver.

The remaining unsupported epistemic forms are:

- direct raw RIR lowering;
- genuinely unbounded or untyped modal tuple keys;
- unsafe unbound negated epistemic variables;
- CPU-only world-view scans;
- cyclic WFS shapes outside the `xlog-gpu` GPU-backed negated-modal WFS plan;
- recursive epistemic programs that also carry modal integrity constraints;
- Gelfond-1991 compatibility components with recursive negation or aggregation;
- one derived epistemic predicate name authored at multiple source arities
  (pure extensional predicates may still share a name across arities);
- sibling clauses for an augmented epistemic head that lower to different
  hidden tuple shapes;
- queries over an augmented epistemic head whose arguments are not a tuple of
  distinct named variables; and
- single-pass clause unions whose modal conjunctions differ and are not proven
  redundant for every row of the combined relation.

---

## Predicates and Declarations

### Predicate Declarations

Predicate declarations specify the name and column types:

```xlog
pred edge(u32, u32).           // Two u32 columns
pred weight(u32, u32, f64).    // Three columns: u32, u32, f64
pred label(u32, symbol).       // Node with a string label
pred active(u32).              // Single-column predicate
```

### Visibility

By default, predicates are public and can be imported by other modules. Use `private` to hide internal predicates:

```xlog
private pred helper(u32).      // Not visible outside this module
pred public_api(u32).          // Visible to other modules

helper(X) :- internal(X).
public_api(X) :- source(X).
```

Private predicates may support other local definitions when the file is used as the
entry file. An exported rule cannot depend on a private predicate because private
definitions are not merged into importing programs; such an import is rejected with
`error[E0406]`.

### Domain Declarations

Domain declarations provide semantic naming for types:

```xlog
domain user_id: u32.
domain temperature: f64.
```

---

## Facts

Facts are ground atoms (no variables) that assert tuples exist in a relation:

```xlog
pred parent(symbol, symbol).

parent(alice, bob).
parent(bob, charlie).
parent(charlie, dave).
```

Facts can also be written on a single line:

```xlog
edge(1, 2). edge(2, 3). edge(3, 4).
```

### Numeric Facts

```xlog
pred point(u32, f64, f64).    // id, x, y

point(1, 0.0, 0.0).
point(2, 3.0, 4.0).
point(3, -1.5, 2.7).
```

---

## Rules

Rules derive new facts from existing ones. A rule has a **head** (the derived fact) and a **body** (the conditions).

```xlog
head(X, Y) :- body_atom1(X, Z), body_atom2(Z, Y).
```

### Simple Rules

```xlog
// Direct edge implies reachability
reach(X, Y) :- edge(X, Y).

// Grandparent relationship
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

### Recursive Rules

Recursive rules derive facts that depend on themselves:

```xlog
// Transitive closure
reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

// Ancestor relationship
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
```

XLOG uses **semi-naive evaluation** for efficient recursive computation, processing only new (delta) tuples in each iteration.

### Multi-Atom Bodies

Rules can have multiple atoms in the body, which are implicitly joined:

```xlog
// Find triangles in a graph
triangle(A, B, C) :- edge(A, B), edge(B, C), edge(C, A), A < B, B < C.

// Star-schema join
sales_by_category(Cat, sum(Amt)) :-
    sale(_, Customer, Product, Amt),
    customer(Customer, Segment),
    product(Product, Cat).
```

### Variable Binding

Variables are bound by appearing in positive atoms. The same variable name in different atoms creates a join condition:

```xlog
// X appears in both atoms, creating a join on that column
connected(X, Z) :- edge(X, Y), edge(Y, Z).
```

---

## Queries

Queries specify which relations to output. Use `?-` followed by an atom pattern:

```xlog
// Output all edges
?- edge(X, Y).

// Output reachable nodes from node 1
?- reach(1, N).

// Output all triangles
?- triangle(A, B, C).
```

Queries can include constants to filter results:

```xlog
// Only nodes reachable from node 1
?- reach(1, X).

// Only edges starting from node 5
?- edge(5, Y).
```

---

## Constraints

Constraints assert that a condition must have **no solutions**. If any tuples satisfy the constraint body, execution fails:

```xlog
// No self-loops allowed
:- edge(X, X).

// Graph must be acyclic (no node reaches itself)
:- reach(X, X).

// No negative values
:- temperature(_, T), T < 0.0.
```

Constraints are checked after all derivations complete. If violated, XLOG reports an error.

---

## Variables and Wildcards

### Named Variables

Variables start with an uppercase letter and are bound to values when matched:

```xlog
// X, Y, Z are variables
path(X, Z) :- edge(X, Y), edge(Y, Z).
```

### Anonymous Wildcard

The underscore (`_`) matches any value without binding. Multiple underscores in a rule are independent:

```xlog
// Check if a node has any outgoing edge
has_edge(X) :- edge(X, _).

// Check if any path exists (don't care about endpoints)
path_exists() :- path(_, _).

// Multiple wildcards are independent
connected() :- edge(_, X), edge(X, _).
```

---

## Comparisons

XLOG supports comparison operators in rule bodies:

| Operator | Meaning |
|----------|---------|
| `=` | Equal |
| `==` | Equal (alternative syntax) |
| `!=` | Not equal |
| `<` | Less than |
| `<=` | Less than or equal |
| `>` | Greater than |
| `>=` | Greater than or equal |

### Examples

```xlog
// Filter by value
adult(X) :- person(X, Age), Age >= 18.

// Inequality for self-join
different_pair(X, Y) :- node(X), node(Y), X != Y.

// Ordering for canonical representation
triangle(A, B, C) :- edge(A, B), edge(B, C), edge(C, A), A < B, B < C.

// Equality between variables
same_score(X, Y) :- score(X, S), score(Y, S), X != Y.
```

### Explicit Equality

Use `=` to explicitly compare variables (in addition to implicit join):

```xlog
// Explicit equality
equal_pair(X) :- pair(X, Y), X = Y.

// Range filter
in_range(Id) :- value(Id, V), V >= 50, V <= 100.
```

---

## Arithmetic Expressions

XLOG uses the `is` keyword for arithmetic computations. The left-hand side must be a **fresh (unbound) variable**.

### Basic Syntax

```xlog
Result is Expression
```

### Operators

| Category | Operators |
|----------|-----------|
| Binary | `+`, `-`, `*`, `/`, `%` (modulo) |
| Precedence | `*`, `/`, `%` bind tighter than `+`, `-` |
| Parentheses | Override precedence with `(...)` |

### Built-in Functions

| Function | Description | Return Type |
|----------|-------------|-------------|
| `abs(X)` | Absolute value | Same as input |
| `min(X, Y)` | Minimum of two values | Same as inputs |
| `max(X, Y)` | Maximum of two values | Same as inputs |
| `pow(X, Y)` | X raised to power Y | `f64` |
| `cast(X, type)` | Type conversion | Specified type |

### Examples

```xlog
pred input(u32, u32).
pred doubled(u32, u32).
pred distance(u32, f64).

input(1, 10).
input(2, 20).

// Double each value
doubled(X, Y) :- input(X, V), Y is V * cast(2, u32).

// Compute distance from origin
distance(Id, D) :-
    point(Id, X, Y),
    D is pow(X * X + Y * Y, 0.5).

// Complex expression with precedence
metrics(X, Result) :-
    value(X, V),
    Result is (V * 3 + 1) / 2.

// Using multiple is expressions
normalized(Id, N) :-
    raw(Id, V),
    AbsV is abs(V),
    Clamped is min(max(AbsV, 0), 100),
    N is cast(Clamped, f64) / 100.0.
```

### Type Casting

Type casting is explicit in XLOG. Use `cast(expr, type)`:

```xlog
// Cast integer to float for division
ratio(X, R) :-
    count(X, N),
    total(T),
    R is cast(N, f64) / cast(T, f64).

// Cast for type compatibility
doubled(X, Y) :- input(X, V), Y is V * cast(2, u32).
```

### Division and Error Handling

| Condition | Integer Behavior | Float Behavior |
|-----------|------------------|----------------|
| Division by zero | Returns `INT64_MAX` | Returns `NaN` or `Inf` |
| Overflow | Wraps (standard semantics) | IEEE 754 behavior |

To avoid division by zero errors, filter explicitly:

```xlog
safe_ratio(X, Y, R) :-
    numerator(X), denominator(Y),
    Y != 0,
    R is X / Y.
```

---

## Negation

XLOG supports deterministic **negation as failure** using the `not` keyword.
In deterministic programs, `not atom(...)` is closed-world, stratified
negation. This is distinct from probabilistic nonmonotone negation, where exact
inference may use Well-Founded Semantics (WFS) for accepted probabilistic
programs.

### Syntax

```xlog
not atom(args)
```

### Examples

```xlog
// Nodes with no outgoing edges
leaf(X) :- node(X), not edge(X, _).

// Nodes with no incoming edges
root(X) :- node(X), not edge(_, X).

// Isolated nodes (no edges at all)
isolated(X) :- node(X), not edge(X, _), not edge(_, X).

// Childless people
childless(X) :- person(X), not parent(X, _).

// Unreachable nodes
unreachable(X) :- node(X), not reach(start, X).
```

### Stratification Requirements

Negation must be **stratifiable**: there cannot be a cycle where a predicate depends on its own negation.

**Valid (stratifiable):**

```xlog
has_child(X) :- parent(X, _).
childless(X) :- person(X), not has_child(X).
```

**Invalid (unstratifiable):**

```xlog
// ERROR: p depends on not p
p(X) :- q(X), not p(X).
```

XLOG detects unstratifiable deterministic programs at compile time and reports
a `negation safety error`. Low-level stratification analysis still reports the
underlying dependency cycle for callers that use it directly.

### Domain Safety

Named variables in negated atoms must be **bound** by a prior source-order
binder in the same rule body. Positive atoms bind their named variables, and a
deterministic `is` expression binds its target after the expression appears.
Source order matters: a later positive atom does not make an earlier negated
atom safe. Use `_` for existential positions inside negated atoms.

```xlog
// Valid: X is bound by node(X) before used in not edge(X, _)
leaf(X) :- node(X), not edge(X, _).

// Invalid: Y is not bound by a prior source-order binder
// bad(X) :- node(X), not edge(X, Y).  // ERROR

// Invalid: X is bound too late
// bad(X) :- not edge(X, _), node(X).  // ERROR
```

---

## Aggregations

Aggregations compute summary values over groups of tuples.

### Supported Aggregates

| Aggregate | Description | Input Types | Output Type |
|-----------|-------------|-------------|-------------|
| `count(X)` | Count of values | Any | `u64` |
| `sum(X)` | Sum of values | `u32`, `u64` | `u64` |
| `min(X)` | Minimum value | `u32`, `u64` | Same as input |
| `max(X)` | Maximum value | `u32`, `u64` | Same as input |
| `logsumexp(X)` | Log-sum-exp (numerically stable) | `f64` | `f64` |

### Syntax

Aggregates appear in the rule head. Variables not in the aggregate form the grouping key:

```xlog
head(GroupKeys, aggregate(AggVar)) :- body(GroupKeys, AggVar, ...).
```

### Examples

```xlog
pred edge(u32, u32).
pred out_degree(u32, u64).

edge(1, 2). edge(1, 3). edge(2, 4).

// Count outgoing edges per node
out_degree(X, count(Y)) :- edge(X, Y).

// Result: out_degree(1, 2), out_degree(2, 1)
```

### Multi-Key Aggregation

```xlog
pred txn(u32, u32, u32).    // user, day, amount
pred daily_total(u32, u32, u64).

txn(1, 1, 10). txn(1, 1, 5). txn(1, 2, 7). txn(2, 1, 3).

// Sum amount per (user, day)
daily_total(User, Day, sum(Amount)) :- txn(User, Day, Amount).

// Result: daily_total(1, 1, 15), daily_total(1, 2, 7), daily_total(2, 1, 3)
```

### Multiple Aggregates

```xlog
pred latency(u32, u32).     // service_id, latency_ms
pred stats(u32, u32, u32).  // service_id, min_latency, max_latency

latency(1, 100). latency(1, 250). latency(1, 150).
latency(2, 7). latency(2, 3). latency(2, 5).

// Min and max per service
stats(Service, min(L), max(L)) :- latency(Service, L).

// Result: stats(1, 100, 250), stats(2, 3, 7)
```

### Computing Averages

XLOG does not have a built-in `avg` aggregate. Compute it using `sum` and `count`:

```xlog
pred txn(u32, u32).         // user_id, amount
pred total(u32, u64).
pred cnt(u32, u64).
pred avg(u32, f64).

txn(1, 10). txn(1, 5). txn(1, 7).
txn(2, 3). txn(2, 9).

total(User, sum(Amount)) :- txn(User, Amount).
cnt(User, count(Amount)) :- txn(User, Amount).

avg(User, A) :-
    total(User, Total),
    cnt(User, N),
    A is cast(Total, f64) / cast(N, f64).
```

### Log-Sum-Exp Aggregation

For probabilistic and numerical applications, `logsumexp` computes `log(sum(exp(values)))` in a numerically stable way:

```xlog
pred obs(u32, f64).         // group_id, log_value
pred score(u32, f64).

obs(1, 0.0). obs(1, 1.0). obs(1, 2.0).
obs(2, 0.0). obs(2, 0.0).

score(X, logsumexp(Y)) :- obs(X, Y).

// score(1) = log(exp(0) + exp(1) + exp(2)) = log(1 + e + e^2) ≈ 2.407
// score(2) = log(exp(0) + exp(0)) = log(2) ≈ 0.693
```

---

## Lists

The language-completeness contract defines finite list syntax and list built-ins. Lists are accepted only
when they are finite and typed; accepted programs lower to relation layouts and
normal GPU-capable execution paths.

Current finite-list implementation status: scalar, symbol, nested finite list,
and finite meta term IDs are normalized to `u64` list identifiers plus
`__xlog_list_*` helper relations. `list<T>` predicate columns lower to scalar
list-id columns while helper relations carry length, item, cons, append, sort,
msort, and set facts. The current implementation accepts finite literals,
declared `list<T>` columns, safe `[Head | Tail]` patterns in declared list
columns, and the built-ins below when their list argument has a finite literal
or a known `list<T>` type.

### Syntax

```xlog
[]                 // empty list
[A, B, C]          // finite list literal
[Head | Tail]      // finite cons pattern when Tail is bound to a finite list
list<u32>          // homogeneous finite list type
```

### Built-Ins

| Built-in | Contract |
|----------|----------|
| `is_list(X)` | Succeeds when `X` is a finite list value |
| `member(X, L)` | Enumerates or checks finite membership |
| `memberchk(X, L)` | Deterministic membership check |
| `length(L, N)` | Computes the finite list length |
| `nth(N, L, X)` | Binds/checks zero-based element `N` |
| `append(A, B, C)` | Concatenates finite lists when at least two sides are bound enough to avoid generation |
| `sort(L, S)` | Sorts finite list values with duplicate removal |
| `msort(L, S)` | Sorts finite list values preserving duplicates |
| `list_to_set(L, S)` | Removes duplicates while producing a finite normalized list |

Pair helpers may be added as typed helpers when they lower to finite relation
columns. Unsupported helpers must be rejected with a typed diagnostic rather
than simulated by a CPU term evaluator.

Current implementation limits:

- `append(A, B, C)` accepts finite first and second list literals; split
  generation such as `append(A, B, [1,2,3])` is rejected as unbounded;
- `sort`, `msort`, and `list_to_set` require a finite input literal;
- `term`, `compound`, and `predref` list elements lower to finite `u64` meta
  term IDs when they appear in declared `list<term>`, `list<compound>`, or
  `list<predref>` contexts.

### Examples

```xlog
pred seed(u32).
pred selected(u32).

selected(X) :- seed(_), member(X, [1, 2, 3]).
```

```xlog
pred path(symbol, list<symbol>).
pred path_len(symbol, u32).

path_len(Name, N) :- path(Name, Steps), length(Steps, N).
```

### Unsupported Forms

The following forms are outside the finite-list contract:

- unbounded list generation, such as asking `append(A, B, [1,2,3])` to
  enumerate every split unless a finite split mode is explicitly implemented;
- cyclic lists;
- heterogeneous lists unless represented through a declared finite `term`
  domain;
- list evaluation that requires an arbitrary CPU Prolog term heap.

---

## Safe Meta-Predicates

Safe meta-predicates provide finite term inspection and static
predicate mapping. They do not introduce an unrestricted Prolog dynamic
database or unrestricted `call/N`.

Current safe-meta implementation status: accepted safe meta forms normalize
before stratification and lowering into ordinary helper relations such as
`__xlog_meta_functor`, `__xlog_meta_univ`, `__xlog_meta_findall_*`, and
`__xlog_meta_maplist_*`. Finite `term`, `compound`, and `predref` predicate
columns lower to `u64` term IDs; compound metadata and univ parts are stored in
typed helper relations. `findall` and `maplist` currently accept finite source
facts and finite list literals only; derived-goal collection and non-literal
maplist inputs are explicitly rejected. Use head aggregates such as `count` and
`sum` to fold data that rules produce.

### Supported Meta-Predicates

| Predicate | Contract |
|-----------|----------|
| `ground(Term)` | True when the term is fully ground after static analysis or finite binding |
| `var(Term)` | True only at the supported static/runtime boundary for unbound variables |
| `nonvar(Term)` | True when `Term` is known not to be a variable |
| `functor(Term, Name, Arity)` | Inspects or checks a finite compound term's functor and arity |
| `Term =.. Parts` | Converts between finite compound terms and a finite `[Functor, Args...]` list |
| `findall(Template, Goal, List)` | Collects finite goal results into a normalized list relation; empty result succeeds with `[]` |
| `maplist(PredRef, List)` | Statically expands a unary predicate reference over a finite list |
| `maplist(PredRef, ListA, ListB)` | Statically expands a binary predicate reference over aligned finite lists |

### Examples

```xlog
pred edge(u32, u32).
pred fanout(u32, list<u32>).

fanout(Ys) :- findall(Y, edge(1, Y), Ys).
```

```xlog
pred positive(u32).
pred all_positive(list<u32>).

all_positive(1) :- maplist(positive, [1, 2, 3]).
```

### Unsupported Forms

- runtime-variable predicate names in `maplist` or any future call-like form;
- `assert`, `retract`, dynamic database mutation, or IO predicates;
- `findall` over derived goals or goals with variables that are neither bound
  before `findall` nor collected by the template;
- non-literal `maplist` inputs in the current safe-meta subset;
- constructing recursive or open compound terms with `=..`;
- higher-arity `maplist` unless explicitly implemented and tested.

---

## Magic Sets

Magic-set rewriting specializes bound deterministic recursive queries by adding
derived magic predicates and adorned rules before optimization. The accepted
accepted magic-set subset is a source-level rewrite over positive recursive rules
that can be proven query-equivalent under the supported left-to-right SIPS
(sideways-information-passing strategy — the order in which bound argument
values flow through a rule).

### Configuration

```xlog
#pragma magic_sets = auto
#pragma magic_sets = on
#pragma magic_sets = off
```

`auto` lets the compiler apply the rewrite only when it can prove that the
transformed program preserves query output. `on` requests the rewrite and fails
with a typed `magic_sets error` if the compiler cannot safely apply it.
`off` disables the rewrite.

### Example

```xlog
pred edge(u32, u32).
pred reach(u32, u32).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

?- reach(1, Y).
```

The bound query argument `1` may seed a magic predicate so recursive evaluation
does not materialize unreachable source components. `xlog explain` must show
the adornment, generated magic predicates, and any declined-rewrite reason.

The compiler currently emits helper predicates such as
`__xlog_magic_reach_bf`, where `b` marks a bound argument and `f` marks a free
argument. Accepted rules continue through the normal AST, stratification,
lowering, optimizer, WCOJ, and runtime paths.

### Supported Subset

- deterministic programs with `?-` queries over recursive predicates;
- at least one scalar constant in the queried recursive atom;
- positive recursive rules over the same head predicate;
- source-order binding propagation through prior positive body atoms;
- `auto`, `on`, and `off` source pragmas.

### Declined Forms

`auto` leaves the program unchanged and records a decline reason. `on` fails
with a typed diagnostic for:

- unbound recursive calls under the supported SIPS;
- rules with negation, aggregation, comparison, `is`, or unnormalized univ in
  the recursive target;
- recursive rules that cross language-completeness list/meta helper predicates;
- mutual-recursive SCCs in the current subset;
- probabilistic profiles, which remain governed by the probabilistic engine.

### Unsupported Forms

Magic-set rewriting must decline or fail before execution when negation,
aggregates, list/meta constructs, or probabilistic rules would make equivalence
uncertain. It must not create a runtime side engine.

---

## User-Defined Functions

XLOG supports user-defined functions for reusable arithmetic and relational lookups.

### Basic Functions

```xlog
func double(X) = X * 2.
func add(X, Y) = X + Y.
func square(X) = X * X.
```

### Conditional Functions

Use `if-then-else` for conditional logic:

```xlog
func abs(X) = if X < 0 then 0 - X else X.

func clamp(X, Lo, Hi) =
    if X < Lo then Lo
    else if X > Hi then Hi
    else X.

func sign(X) =
    if X < 0 then -1
    else if X > 0 then 1
    else 0.
```

### Predicate-bodied functions

A predicate-bodied function derives its result through a relational body:

```xlog
func get_parent(Child) = Parent :- parent(Child, Parent).
answer(Parent) :- Parent is get_parent(1).
```

Predicate-bodied calls are supported in `is` expressions in ordinary rules and
constraints. The relational literals are inserted immediately before the source `is`
binding. Multiple calls within one expression are inserted from left to right. Each
invocation receives fresh names for its non-parameter result and body-local variables,
and nested calls are expanded recursively up to `#pragma max_recursion_depth`.

The relational body may match zero, one, or many rows. Each match contributes a caller
row; predicate-bodied function syntax does not enforce a unique scalar result. Ordinary
set semantics deduplicates identical projected caller tuples even when multiple body
witnesses derive them.

When a parameter is used in a predicate-body term position, its argument must be a
variable or numeric literal. Materialize a compound arithmetic expression with an
earlier `is` binding before passing it. Predicate-bodied calls in conditional result
branches are rejected because their relational literals cannot be hoisted without
changing guarded semantics.

### Recursive definitions

Production compilation registers function declarations in source order and expands only
definitions reachable from ordinary rule or constraint calls. It rejects duplicate
function declarations, but unused definitions with recursion, undefined callees, or a
predicate-name conflict do not block this demand-driven path.

Expansion is eager across both conditional branches rather than data-dependent. If it
tries to enter another user-defined call while already at the depth configured by
`#pragma max_recursion_depth`, it reports `error[E0504]`. A reachable cyclic call chain
therefore reaches `E0504` whether it is wholly unguarded or contains a conditional branch
that looks like a base case. Function recursion is not a runtime looping construct.

The Rust library also exposes strict whole-program validation through
`FunctionRegistry::from_program`. It visits every definition in declaration order and
callees in source order. A recursive strongly connected component (SCC) with no
conditional-bodied member is rejected with `error[E0502]`; a conditional-bodied member
lets the SCC pass that structural check. Because later expansion still visits both
branches, a used guarded SCC can pass strict validation and then reach `E0504`.

### Using Functions in Rules

Functions are called on the right side of an `is` expression:

```xlog
pred doubled_value(u32, u32).
doubled_value(X, Y) :- input(X, V), Y is double(V).
```

### Type Annotations

Optional parameter and return annotations are parsed and preserved as function metadata:

```xlog
func add(X: f64, Y: f64) -> f64 = X + Y.
func celsius_to_fahrenheit(C: f64) -> f64 = C * 1.8 + 32.0.
```

Executable types are inferred from predicate schemas and expressions. Function
parameter and return annotations are not independently enforced as call signatures.

### Private Functions

Functions can be marked private and used by other definitions in the same entry file:

```xlog
private func helper(X) = X * X + 1.
func public_calc(X) = helper(X) * 2.
```

---

## Modules

XLOG supports organizing code into modules for reusability and encapsulation.

### Creating Modules

Imported modules are `.xlog` files, and the filename without its extension becomes the
module name. The entry file is loaded from the exact path supplied to the CLI and may use
another extension:

```xlog
// File: graph.xlog
pred edge(u32, u32).
edge(1, 2).
edge(2, 3).

pred reach(u32, u32).
reach(A, B) :- edge(A, B).
reach(A, C) :- edge(A, B), reach(B, C).
```

### Importing Modules

Use the `use` statement to import public deterministic definitions from modules:

```xlog
// Import all public predicates and functions
use graph.

// Import specific predicates or functions
use graph::{edge, reach}.

// Import from nested module path
use utils/math::{abs, clamp}.
use data/graph/algorithms::{shortest_path}.
```

### Module Paths

Module paths use `/` as the separator:

```
project/
  main.xlog
  graph.xlog
  utils/
    math.xlog
    string.xlog
```

```xlog
// In main.xlog
use graph.
use utils/math.
```

A `use` in the entry file first resolves relative to the entry path's directory. A nested
`use` first resolves relative to the importing module's canonical source directory, then
through the configured module search paths. Resolution follows those importer-specific
edges throughout validation and merging. Distinct files are not conflated merely because
their imports use the same spelling, and aliases of one canonical source file do not
duplicate its definitions.

Public facts and rules for the same predicate are merged across resolved import branches
into one relation. The same union semantics apply when selective imports name that
predicate in multiple modules. Participating declarations are checked under
`error[E0408]`. Without a participating declaration, inferred clause-head column types from
different source programs—either the entry program and a selected import, or two selected
imports—are compared by predicate name and arity. Constants, head variables typed by ordinary
body atoms or built-in arithmetic bindings, and aggregate result types supply evidence;
unanchored variables do not. Conflicting types for one signature produce `error[E0412]`. For
invalid built-in arithmetic or aggregate evidence, resolution reports `error[E0413]` with the
contributing module and source file. User-defined function calls are expanded after import
resolution. For import schema validation, different arities remain distinct. Functions have one body:
separate branches cannot
define the same imported function, and a module or entry file cannot redefine an imported
function. Those function conflicts produce `error[E0402]`. An imported module that defines
an exported function name more than once is rejected with `error[E0410]`.

Every declaration in the entry program and every public declaration selected by the
resolved imports participates in declaration compatibility checking. Participating
declarations for one predicate must have identical arity, column names, and resolved
types; otherwise resolution reports `error[E0408]`. Private declarations in imported
modules and public declarations omitted by a selective import are not merged and do not
participate in that comparison.
Domain-backed types are compared using the scalar type to which each local alias resolves.
Reusing one domain alias name for different scalar types in the entry program or import
closure is `error[E0409]`.
Within an imported module, every declaration of one predicate must use the same visibility;
mixing public and `private` declarations is rejected with `error[E0411]`.

### Visibility and Encapsulation

Predicates and functions are **public by default**. Use `private` to hide implementation details:

```xlog
// File: graph.xlog

// Public API
pred edge(u32, u32).
pred reach(u32, u32).

reach(A, B) :- edge(A, B).
reach(A, C) :- edge(A, B), reach(B, C).
```

Private predicates are not exported. Naming one in a selective `use` fails with
`error[E0404]`; it does not make the predicate visible to the importing program.

An exported rule or function that depends on a private item, or on a public item omitted
by a selective import, is rejected with `error[E0406]`. Include every public dependency
in a selective import.

Imports merge public deterministic predicates, functions, facts, and rules, together
with domain aliases. Probabilistic facts, annotated disjunctions, evidence, integrity
constraints, neural predicate declarations, and learnable rule templates belong in the
entry file; an imported module containing one is rejected with `error[E0405]`. Queries,
probabilistic queries, and pragmas are entry-file-scoped. Imported queries are not merged,
and imported pragmas are ignored with `warning[W0510]`.

---

## Symbols

Symbols are interned strings, represented internally as integers for efficient comparison and storage.

### Declaring Symbol Types

```xlog
pred color(symbol).
pred person(symbol, u32).   // name: symbol, age: u32

color(red).
color(green).
color(blue).

person(alice, 30).
person(bob, 25).
```

### Symbol Literals

Symbol literals are written as identifiers (lowercase) or as string literals:

```xlog
// As identifiers
color(red).
color(green).

// As string literals (for names with special characters)
city("New York").
city("Los Angeles").
```

### Querying Symbols

Symbols are displayed as their original string values in query output:

```xlog
pred status(symbol).
status(active).
status(pending).
status(completed).

?- status(S).
```

Output:

```
status(S):
| S         |
|-----------|
| active    |
| pending   |
| completed |
```

### Symbol Comparison

Symbols can be compared for equality:

```xlog
same_status(X, Y) :- item(X, S), item(Y, S), X != Y.
```

### Reversible Symbols

Symbols are **reversible**: the original string value is preserved and
displayed in query output. Symbols are stored internally as `u32` IDs with a
bidirectional mapping to strings.

```xlog
pred task(symbol, symbol).
task(project_alpha, alice).
task(project_beta, bob).

?- task(Project, Owner).
```

Output displays original string values:
```
| Project       | Owner |
|---------------|-------|
| project_alpha | alice |
| project_beta  | bob   |
```

---

## Probabilistic Logic

XLOG supports probabilistic Datalog for reasoning under uncertainty.

### Probabilistic Facts

Probabilistic facts are Bernoulli random variables with a specified probability:

```xlog
0.3::rain.           // P(rain) = 0.3
0.7::sprinkler.      // P(sprinkler) = 0.7
0.5::edge(1, 2).     // P(edge(1,2)) = 0.5
```

### Annotated Disjunctions

Annotated disjunctions represent categorical distributions (mutually exclusive outcomes):

```xlog
// Coin flip: heads (60%) or tails (40%)
0.6::coin(heads); 0.4::coin(tails).

// Die roll
0.167::die(1); 0.167::die(2); 0.167::die(3);
0.167::die(4); 0.167::die(5); 0.166::die(6).
```

If probabilities sum to less than 1, the remaining mass represents an implicit "none" outcome.

### Deterministic Rules with Probabilistic Facts

Rules can derive facts from probabilistic facts:

```xlog
0.3::rain.
0.2::sprinkler.

// Deterministic derivation from probabilistic facts
wet :- rain.
wet :- sprinkler.
```

Arithmetic values bound by `is` in probabilistic rules must be representable by
the probabilistic tuple value model. In particular, a `u64` result greater than
`i64::MAX` is rejected with a range error before it can be used by another
literal or emitted in a derived tuple. Deterministic execution continues to use
the full declared `u64` range.

### Evidence

Evidence constrains the probabilistic model to worlds consistent with observations:

```xlog
evidence(wet, true).      // Condition on wet being true
evidence(sprinkler, false). // Condition on sprinkler being false
```

### Probabilistic Queries

Use `query` to compute the probability of an atom:

```xlog
query(rain).        // Compute P(rain | evidence)
query(wet).         // Compute P(wet | evidence)
```

### Complete Probabilistic Example

```xlog
// Probabilistic facts
0.7::rain.
0.2::sprinkler.

// Deterministic rules
wet :- rain.
wet :- sprinkler.

// Evidence: we observed that wet is true
evidence(wet, true).

// Queries: compute posterior probabilities
query(rain).
query(sprinkler).
```

Running with exact inference:

```bash
xlog prob weather.xlog --prob-engine exact_ddnnf
```

Output:

```
query(rain):      P = 0.875
query(sprinkler): P = 0.25
```

### Inference Engines

XLOG supports two inference engines:

| Engine | Flag | Description |
|--------|------|-------------|
| **Exact** | `--prob-engine exact_ddnnf` | Compiles the program to a Decision-DNNF (a Boolean form that supports exact probability counting); returns exact probabilities |
| **Monte Carlo** | `--prob-engine mc` | Sampling-based; approximate with confidence intervals |

**Exact inference** supports:
- **Stratified deterministic negation**: Automatic layer detection and
  two-valued evaluation for deterministic `not atom` programs
- **Probabilistic non-monotone negation**: Well-Founded Semantics (WFS) for
  accepted cyclic probabilistic programs
- **Finite probabilistic aggregates**: `count`, `sum`, `min`, `max`, and
  `logsumexp` aggregate outputs compiled through provenance/PIR when the exact
  finite domain cap is respected
- **Gradients**: Correct gradient flow through negated literals

**Probabilistic aggregates in the language-completeness contract:** finite `count`, `sum`, `min`, `max`,
and `logsumexp` aggregate programs are supported for exact and MC inference in
the finite probabilistic-aggregate subset. Exact mode enumerates finite aggregate
outcomes into provenance/PIR formulas, so `query` and `evidence` may reference
aggregate output tuples. Exact enumeration is capped per group by the operators
in the head: a head whose aggregate expressions are all `count` accepts up to 64
uncertain contributing rows per group, while `sum`, `min`, `max`, and `logsumexp`
accept up to 16. Cap excess fails with a typed
`prob_aggregate error` that recommends MC or reducing the finite domain.

**Monte Carlo** supports probabilistic rules and non-monotone recursion.
The production MC path is the GPU-resident megakernel engine, which rejects
negation, aggregates, and other unbounded constructs with a typed
`ResidentRejection`. Programs in that fragment (including non-monotone
recursion and language-completeness probabilistic aggregates under MC) require an explicit
opt-in to the **labeled CPU oracle** — `--allow-cpu-oracle` on the CLI,
`allow_cpu_oracle=True` in Python, `McEvalConfig::allow_cpu_oracle_fallback`
in Rust — and their results carry `mc_engine = "cpu-oracle"`. Without the
opt-in, such programs fail closed with the typed rejection (corrected
2026-06-10; previously this fallback was silent and unlabeled):

```bash
xlog prob program.xlog --prob-engine mc --samples 10000 --seed 42
# resident-rejected fragment (negation / aggregates) needs the explicit oracle:
xlog prob program.xlog --prob-engine mc --samples 10000 --allow-cpu-oracle
```

### Probabilistic Aggregates

Finite probabilistic aggregate programs may query or condition on aggregate
outputs:

```xlog
0.5::edge(1, 2).
0.5::edge(1, 3).

out_degree(X, count(Y)) :- edge(X, Y).
query(out_degree(1, 2)).
```

Exact mode compiles each supported finite aggregate outcome into a Boolean PIR
formula over contributing row-presence formulas. Empty probabilistic groups do
not materialize `count(..., 0)` tuples. MC mode samples worlds on the GPU and
executes the shared deterministic aggregate path for each accepted sample batch.

### Aggregate Lifting

Small-domain aggregate lifting is permitted only when the lifted computation is
semantically identical to finite exact enumeration. For each successful lift,
explain output reports the detected finite domain, cap, operator, and `fired`
status. Unsupported operators and type, domain, or cap violations return typed
compilation errors instead of aggregate-lifting report entries.

In the aggregate-lifting subset, finite probabilistic `count` aggregate heads use
exact cardinality dynamic programming when every aggregate expression in the
head is `count(...)`. The lift constructs formulas for exactly `k` contributing
rows over the same row-presence formulas used by finite enumeration, so it does
not change exact semantics. The count lift accepts up to 64 uncertain rows per
group; larger count domains fail closed with a typed `agg_lift error`.

`sum`, `min`, `max`, and `logsumexp` use factorized aggregate-state dynamic
programming with a cap of 16 uncertain rows per group. Successful count and
numeric lifts both report `fired`. `xlog explain --format json` emits an
`aggregate_lifting` array containing the predicate, group key, operator,
finite-domain source, domain size, cap, status, reason, naive outcome count, and
dynamic-programming state count for those successful lifts.

### Monte Carlo Sampling Methods

The Monte Carlo engine supports two sampling methods:

| Method | Description |
|--------|-------------|
| `Rejection` | Sample from the prior and discard worlds where evidence is not satisfied. |
| `EvidenceClamping` | Force evidence variables directly in the CUDA sampler when the evidence is forceable at the probabilistic-fact / positive-AD level. |

Evidence clamping avoids wasted samples when rejection acceptance rates are
low. If evidence is derived, deterministic, or otherwise not directly
forceable, XLOG falls back to rejection sampling.

### Negation in Probabilistic Programs

Probabilistic exact inference supports both stratified probabilistic negation
and accepted non-monotone WFS profiles:

```xlog
// Stratified negation (layered evaluation)
0.3::rain.
dry :- not rain.
query(dry).  // P(dry) = 0.7
```

```xlog
// Non-monotone negation (WFS)
0.5::bias.
p :- bias, not q.
q :- not p.
query(p).  // WFS: atoms in cycle may be undefined
```

### Probabilistic Recursion

Probabilistic facts can be used with recursive rules:

```xlog
0.5::edge(1, 2).
0.5::edge(2, 3).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

query(reach(1, 2)).
query(reach(1, 3)).
```

### Pragma for Engine Selection

The inference engine can be specified in the source file:

```xlog
// Non-monotone negation works with exact inference (uses WFS)
#pragma prob_engine = exact_ddnnf

0.5::flip.
p :- flip.
q :- not p.
p :- not q.

query(p).  // WFS: p is undefined in worlds where bias is false
```

For approximate inference with confidence intervals:

```xlog
#pragma prob_engine = mc
#pragma prob_samples = 10000

0.5::flip.
p :- flip.
q :- not p.

query(p).  // P(p) ≈ 0.5 ± CI
```

---

## Epistemic Logic

Epistemic logic is part of the accepted source surface.

Epistemic logic adds modal body literals for bounded knowledge and possibility
reasoning. Modal literals are valid in rule bodies and use the same atom syntax
as ordinary predicates.

### Modal Literals

| Form | Meaning |
|------|---------|
| `know atom(...)` | The atom must hold in every accepted world/model. |
| `possible atom(...)` | The atom must hold in at least one accepted world/model. |
| `not know atom(...)` | The atom is not known in every accepted world/model. |
| `not possible atom(...)` | The atom is absent from every accepted world/model. |

Finite nested modal chains such as `know possible fact()`,
`not know possible fact()`, and `know not possible fact()` are accepted program
forms. The parser normalizes them to a single epistemic literal by the semantic
parity/duality rules: the operator adjacent to the atom determines the modal
operator, while leading/interior/atom-adjacent `not` tokens dualize or negate
the resulting literal according to their position.

### Epistemic Mode Pragma

```xlog
#pragma epistemic_mode = faeel
#pragma epistemic_mode = g91
```

`faeel` is the default epistemic mode when no pragma is present. `g91` is an
explicit compatibility mode for programs that need Gelfond 1991 possibility behavior.

### Runtime Boundary

Accepted epistemic programs enter EIR through `xlog_logic::build_eir` and run
through the high-level `xlog run` / `xlog_gpu::LogicProgram` dispatch path. That
path classifies dependencies before selecting a single-pass epistemic GPU plan,
an ordinary FAEEL founded least-fixpoint reduction, a Gelfond-1991 positive
exact-tuple `possible` greatest compatibility fixpoint, or a GPU-backed WFS
alternating-fixpoint plan. Direct ordinary RIR lowering of raw epistemic body
literals remains a rejection boundary and must report
`UnsupportedEpistemicConstruct`.

The Gelfond-1991 compatibility plan relaxes the selected gates once to compute a
GPU upper bound. It then reevaluates the program from its extensional inputs,
reading each selected modal target from a frozen copy of the preceding
iteration. GPU set comparison over the intensional relations detects the
greatest compatible tuple fixpoint. This is a tuple-level contract: a positive
non-invariant `possible` literal qualifies when its key exactly matches its rule
head and its head and target share a recursive dependency component. Predicate
recursion without compatible concrete tuples is not sufficient. Ordinary
integrity constraints are evaluated after tuple convergence. When a pass uses
WFS, a positive constraint atom is true only in the lower extension, while a
negated atom is true only when its tuple is absent from the upper extension; an
undefined atom satisfies neither polarity. Modal integrity constraints in a
recursive epistemic program remain unsupported. An unrelated negative cycle may
use GPU-backed WFS inside a pass; recursive negation or aggregation within the
compatibility component is unsupported because the descending operator would be
non-monotone.

Single-pass epistemic output materialization does not retain the defining
clause that supplied each row. One active clause per predicate signature is
always unambiguous. Multiple clauses are also accepted when their normalized
modal conjunctions are identical relative to the output columns, because the
shared filter distributes over the clause union. For an invariant modal target,
`know` and `possible` normalize to the same fixed-extension test.

A single epistemic clause may coexist with ordinary sibling clauses when each
of its positive modal gates is proven to be a no-op for every union row. This
includes ground atoms derived from explicit ordinary facts or rules, bijective
all-variable exact-head tuples with independent founded support, and positive
exact-head `possible` self-support in G91 compatibility mode. Repeated-variable,
constant, or wildcard heads do not qualify as bijective tuple keys.

Other mixed or differently filtered clause unions are rejected with
`epistemic rule-union materialization`; applying every clause's modal gate to
the combined relation would otherwise lose valid sibling rows. Admissible
recursive epistemic programs are not subject to this single-pass restriction.
FAEEL positive-cycle literals resolve into ordinary rule bodies before
least-fixpoint execution. Selected Gelfond-1991 gates instead become reads from
frozen relation snapshots during descending refinement.

### Examples

FAEEL default knowledge:

```xlog
pred fact().
pred accepted().

fact().

accepted() :- know fact().
```

Gelfond 1991 compatibility:

```xlog
#pragma epistemic_mode = g91

pred fact().
pred accepted().

fact().

accepted() :- possible fact().
```

Independent split components:

```xlog
pred left_fact().
pred right_fact().
pred left().
pred right().

left_fact().
right_fact().

left() :- know left_fact().
right() :- possible right_fact().
```

Run the shipped examples:

```bash
xlog run examples/epistemic/03-faeel-default.xlog
xlog run examples/epistemic/02-g91-compatibility.xlog
xlog run examples/epistemic/05-splitting.xlog
```

---

## Approximate Inference

Approximate inference is the source and CLI contract for Monte Carlo
probabilistic reasoning. It must be reproducible under a fixed seed and must
report uncertainty, sample counts, evidence handling, and sampling method.

### Source Pragmas

```xlog
#pragma prob_engine = mc
#pragma prob_samples = 10000
#pragma prob_seed = 42
#pragma prob_confidence = 0.95
#pragma prob_method = evidence_clamping
```

| Pragma | Meaning |
|--------|---------|
| `prob_samples` | Number of MC samples |
| `prob_seed` | Fixed random seed; identical input plus seed must replay identical counts |
| `prob_confidence` | Confidence level for reported intervals |
| `prob_method` | Sampling method, such as `rejection` or `evidence_clamping` |
| `prob_max_nonmonotone_iterations` | Bound for nonmonotone WFS/MC iteration where applicable |

CLI flags override source pragmas when both are provided; the effective
configuration must be visible in CLI output and explain JSON.

In the approximate-inference subset, `xlog prob` uses source `#pragma prob_engine = mc`
when `--prob-engine` is not supplied. CLI flags override source pragmas
field-by-field: `--samples`, `--seed`, `--confidence`, `--prob-method`, and
`--prob-max-nonmonotone-iterations` replace only their matching source values.
`--output json` emits a machine-readable MC report with the same probability,
standard-error, confidence-interval, sample-count, evidence-count, seed,
confidence, and sampling-method fields as the pretty/csv batch output.

### Output Contract

Approximate output formats must include:

- probability estimate;
- standard error;
- confidence interval low/high;
- sample count;
- evidence count or acceptance/clamping count;
- seed and method.

Invalid confidence ranges, unsupported methods, or ambiguous precedence must
fail before sampling begins.

---

## Neural Predicates

Neural predicate declarations embed a neural network directly into an XLOG
program as a first-class predicate.


### Declaration Forms

**Classification (`nn/4`):**

```xlog
nn(network_name, [InputVar1, InputVar2, ...], OutputVar, [label1, label2, ...]) :: atom.
```

**Embedding mode (`nn/3`):**

```xlog
nn(network_name, [InputVar], OutputVar) :: atom.
```

### Arguments

| Position | Meaning |
|----------|---------|
| `network_name` | Python-side registration name |
| `[InputVar, ...]` | Input variables passed to the forward pass |
| `OutputVar` | Predicted label or dense embedding output |
| `[label1, ...]` | Label set for `nn/4` declarations |
| `atom` | Predicate template instantiated by the declaration |

### Lowering

For `nn/4`, the network output is treated as a label distribution and lowered
to probabilistic facts that enter the inference pipeline. For `nn/3`, the
network output is a dense tensor bound into the program and used through the
embedding APIs.

### Example: MNIST Classification

```xlog
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, LeftDigit), digit(Y, RightDigit), Z is LeftDigit + RightDigit.
```

Python side:

```python
program = pyxlog.Program.compile(source)
net = MnistClassifier().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
program.register_network("mnist_net", net, optimizer)
```

### Example: Embedding Mode

```xlog
nn(entity_embed, [X], E) :: embed(X, E).
```

Python side:

```python
embedding = torch.nn.Embedding(100, 64).cuda()
program.register_embedding("entity_embed", embedding, trainable=True)
```

### Cross-Registration Validation

Classification declarations (`nn/4`) require `register_network()`. Embedding
declarations (`nn/3`) require `register_embedding()`. Reusing the same name
across both forms is rejected.

---

## Learnable Rules

Learnable rules are the source-level surface for differentiable ILP. A rule is
gated by a named mask tensor:

```xlog
learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
```

During training, candidate rules are softly weighted; at convergence, argmax
selects the winning rule and the learned artifact serializes the concrete
program.

### Training API

```python
from pyxlog.ilp import train_only, train_and_promote, TrainConfig

result = train_only(source, "W_reach", positive_examples, negative_examples, TrainConfig(...))
promotion = train_and_promote(source, "W_reach", positive_examples, negative_examples, TrainConfig(...))
```

### Multiple Learnable Rules

```xlog
learnable(W_gp)  :: grandparent(X, Y) :- bL(X, Z), bR(Z, Y).
learnable(W_col) :: colleague(X, Y)   :- bL(X, Z), bR(Y, Z).
```

### Cross-References

- [Rule learning (dILP)](/neural/rule-learning)
- [Python API (pyxlog)](/reference/python)

---

## Term Embeddings

Term embeddings associate dense vectors with discrete logic terms.

### `register_embedding()`

```python
program = pyxlog.Program.compile(source)
embedding = torch.nn.Embedding(100, 64).cuda()
program.register_embedding("entity_embed", embedding, trainable=True)

weights = torch.randn(100, 64).cuda()
program.register_embedding("pretrained_emb", weights, trainable=False)
```

### `forward_embedding()`

```python
vectors = program.forward_embedding("entity_embed", [0, 5, 12])
```

The returned tensor stays on the same device as the registered embedding. For a
trainable `nn.Embedding`, gradients remain attached; for frozen tensors they do
not.

**Current scope:** explicit training-side embedding registration and batched
lookup are supported. Rule-level embedding inference remains deferred.

---

## GPU ILP Configuration

The GPU-resident ILP path exposes runtime configuration and telemetry through
Python.

### `coo_chunk_budget`

Controls the per-chunk temporary allocation ceiling used during sparse COO
construction on GPU:

```python
trainer.set_coo_chunk_budget(256 * 1024 * 1024)
```

The older `coo_memory_cap` naming is deprecated.

### `host_transfer_stats()`

Training APIs expose transfer telemetry so strict zero device-to-host workflows can verify
that the hot path stayed device-resident:

```python
stats = trainer.host_transfer_stats()
trainer.reset_host_transfer_stats()
```

---

## Pragmas and Directives

Pragmas configure compiler and runtime behavior.

The current source grammar accepts `prob_engine`, `prob_cache`,
`epistemic_mode`, `max_recursion_depth`, `magic_sets`, and the MC configuration
pragmas below. The `epistemic_mode` pragma is accepted for the epistemic solver
surface.

### Syntax

```xlog
#pragma key = value
```

### Scoping

Pragmas are entry-file-scoped: only directives in the file passed to `xlog run`,
`xlog prob`, or `xlog explain` configure the engine. A pragma declared in an imported module
is dropped at merge time and reported with `warning[W0510]`. See
[Pragmas apply only in the entry file](/language-guide/pragmas#pragmas-apply-only-in-the-entry-file).

### Available Pragmas

| Pragma | Values | Description |
|--------|--------|-------------|
| `prob_engine` | `exact_ddnnf`, `mc` | Select probabilistic inference engine |
| `prob_cache` | `on`, `off` | Enable/disable probability caching |
| `epistemic_mode` | `faeel`, `g91` | Select FAEEL default or Gelfond 1991 compatibility epistemic semantics |
| `max_recursion_depth` | integer | Maximum configured recursion depth; also bounds Gelfond-1991 compatibility refinements and GPU WFS alternating-fixpoint iterations |
| `magic_sets` | `auto`, `on`, `off` | Control bound-recursive magic-set rewriting |
| `prob_samples` | integer | Monte Carlo sample count |
| `prob_seed` | integer | Monte Carlo random seed |
| `prob_confidence` | float in `(0, 1)` | Monte Carlo confidence level |
| `prob_method` | `rejection`, `evidence_clamping` | Monte Carlo sampling method |
| `prob_max_nonmonotone_iterations` | integer | Bound nonmonotone probabilistic iteration where applicable |

### Examples

```xlog
// Use Monte Carlo inference
#pragma prob_engine = mc

// Disable probability caching
#pragma prob_cache = off

// Select epistemic compatibility semantics
#pragma epistemic_mode = g91

// Limit recursion depth
#pragma max_recursion_depth = 100

// Let the compiler specialize safe bound recursive queries
#pragma magic_sets = auto

// Configure approximate inference
#pragma prob_samples = 10000
#pragma prob_seed = 42
#pragma prob_confidence = 0.95
#pragma prob_method = evidence_clamping
```

---

## CLI Developer Experience

Developer-experience commands make the language inspectable
and interactive without changing the runtime execution contract.

### `xlog explain`

```bash
xlog explain [--format text|json|dot] <FILE>
```

The CLI explain report includes parse-cache and AST counts, stratification status,
relational compilation and optimizer status, WCOJ reporting availability, magic-set
rewrite results, epistemic lowering status, aggregate lifting summaries, provenance,
proof traces, and generated-rule diagnostics without requiring GPU execution.
Unavailable analyses include a reason instead of a derived plan or decision. The JSON
representation includes the corresponding status, counts, summaries, and diagnostic
records and is deterministic for fixed input.

JSON output must be deterministic for fixed input. Unknown formats are rejected
before compilation.

### Incremental Parsing

`xlog_logic::ParserSession` is the shared parser cache for explain, REPL, and
watch workflows. It splits `.xlog` source into statement units, records stable
text hashes and byte/line spans, and reparses only changed statements through
the production parser. Cache statistics report hits, misses, invalidations,
module invalidations, statement count, and an estimated statement-unit speedup.

Module invalidation removes the changed module and cached sources that import
that module name. Incremental parse diagnostics include the original
line/column and byte span before forwarding the underlying parser error.

### `xlog repl`

```bash
xlog repl [--module-path DIR[:DIR...]]
```

The command reads a multiline source session from standard input until EOF,
parses it through the incremental parser/session cache, reports
statement/rule/query counts, and exits. It does not compile or execute the
session and therefore does not initialize GPU execution.

### `xlog watch`

```bash
xlog watch [--debounce-ms N] [--explain] [--once] [--module-path DIR[:DIR...]] <FILE>
```

Watch mode re-reads and reparses the entry file at a fixed interval. Each pass
prints parser statistics; `--explain` also builds a fresh explain report. Explain
passes resolve sibling imports and imports from `--module-path` again after each
re-read. `--once` runs one pass and exits, and `--debounce-ms` controls the interval
between repeated passes.

Unsupported REPL/watch mutation semantics, such as dynamic `assert` or
`retract`, remain outside the language.

---

## Float Predicates and Total-Order Semantics

XLOG uses one total order for every floating-point comparison. Host and CUDA
execution use the same bit-to-key mapping, consistent with Rust's
`f32::total_cmp` and `f64::total_cmp`.

### Comparison Semantics

| Operator | Semantics | Special-value behavior |
|----------|-----------|------------------------|
| `=`, `!=` | Total-order identity | Equal only when the float bit patterns match |
| `<`, `<=`, `>`, `>=` | Total ordering | Every NaN and infinity is comparable |

### Total Ordering

The total order has this broad shape:

```
-NaN < -Inf < ... < -0.0 < +0.0 < ... < +Inf < +NaN
```

This means:
- NaN values are at the extremes (negative NaN is smallest, positive NaN is largest)
- `-0.0` and `+0.0` are distinct, including for equality
- NaN sign, signaling state, and payload are distinct
- All values are comparable (no undefined ordering)
- `min`, `max`, sorting, difference, and deduplication use the same order

Arithmetic-generated NaNs are normalized to one positive quiet-NaN bit pattern,
so the result does not depend on a host or CUDA payload choice.

### Preventing Invalid Arithmetic

A NaN is equal to the same NaN bit pattern, so `V != V` is not a NaN test in
XLOG. Validate the producing operation instead:

```xlog
valid(Id, V) :- input(Id, Num, Denom), Denom != 0.0, V is Num / Denom.
```

When the accepted data domain has finite bounds, total-order comparisons can
also reject infinities and both signed NaN regions:

```xlog
accepted(Id, V) :- data(Id, V), V > -1000000.0, V < 1000000.0.
```

Use bounds only when they are genuine application limits. Track source
validation or import metadata when exact fault classification matters.

### Example: Data Cleaning Pipeline

```xlog
pred raw(u32, f64, f64).
pred reading(u32, f64).
pred valid(u32, f64).
pred invalid_divisor(u32).

raw(1, 22.5, 1.0).
raw(2, 0.0, 0.0).
raw(3, 28.3, 1.0).

reading(Id, V) :- raw(Id, Num, Denom), V is Num / Denom.
valid(Id, V) :- raw(Id, Num, Denom), Denom != 0.0, V is Num / Denom.
invalid_divisor(Id) :- raw(Id, _, Denom), Denom = 0.0.
```

---

## Comments

XLOG supports single-line comments:

```xlog
// This is a comment
pred edge(u32, u32).  // Inline comment

// Multi-line explanation
// spanning several lines
// before a complex rule
reach(X, Z) :- reach(X, Y), edge(Y, Z).
```

Block comments (`/* ... */`) are **not supported**.

---

## Complete Grammar Reference

The following is a summary of the current XLOG grammar plus the
language-completeness extensions and epistemic source surface in PEG-style
notation.

### Lexical Elements

```pest
WHITESPACE = _{ " " | "\t" | "\r" | "\n" }
COMMENT = _{ "//" ~ (!"\n" ~ ANY)* }

ident = @{ ASCII_ALPHA_LOWER ~ (ASCII_ALPHANUMERIC | "_")* }
variable = @{ ASCII_ALPHA_UPPER ~ (ASCII_ALPHANUMERIC | "_")* }
anonymous = @{ "_" }

integer = @{ "-"? ~ ASCII_DIGIT+ }
float_num = @{ "-"? ~ ASCII_DIGIT+ ~ "." ~ ASCII_DIGIT+ }
string_lit = @{ "\"" ~ (!"\"" ~ ANY)* ~ "\"" }
bool_lit = { "true" | "false" }
```

### Types

```pest
type_spec = { "u32" | "u64" | "i32" | "i64" | "f32" | "f64" | "bool" | "symbol" }
type_list = { type_spec ~ ("," ~ type_spec)* }
```

Language-completeness contract:

```pest
type_spec = {
    scalar_type
    | domain_ref
    | list_type
    | "term"
    | "compound"
    | "predref"
}
scalar_type = { "u32" | "u64" | "i32" | "i64" | "f32" | "f64" | "bool" | "symbol" }
domain_ref = { ident }
list_type = { "list" ~ "<" ~ type_spec ~ ">" }
named_column = { ident ~ ":" ~ type_spec }
pred_column = { named_column | type_spec }
type_list = { pred_column ~ ("," ~ pred_column)* }
```

### Terms and Atoms

```pest
term = { var_or_anon | float_num | integer | string_lit | ident }
term_list = { term ~ ("," ~ term)* }
atom = { ident ~ "(" ~ term_list? ~ ")" }
```

Language-completeness contract:

```pest
list_literal = { "[" ~ (term ~ ("," ~ term)*)? ~ "]" }
cons_pattern = { "[" ~ term ~ "|" ~ term ~ "]" }
compound_term = { ident ~ "(" ~ term_list? ~ ")" }
predref_term = { ident }
term = {
    var_or_anon
    | float_num
    | integer
    | string_lit
    | list_literal
    | cons_pattern
    | compound_term
    | predref_term
    | ident
}
term_list = { term ~ ("," ~ term)* }
atom = { ident ~ "(" ~ term_list? ~ ")" }
```

### Expressions

```pest
cmp_op = { "==" | "!=" | "<=" | ">=" | "<" | ">" | "=" }
comparison = { term ~ cmp_op ~ term }

arith_op_mul = { "*" | "/" | "%" }
arith_op_add = { "+" | "-" }

builtin_fn = { "abs" | "min" | "max" | "pow" | "cast" }
func_call = { ident ~ "(" ~ (arith_expr ~ ("," ~ arith_expr)*)? ~ ")" }

arith_primary = {
    builtin_fn ~ "(" ~ arith_expr ~ ("," ~ (arith_expr | type_spec))* ~ ")" |
    func_call |
    "(" ~ arith_expr ~ ")" |
    variable |
    float_num |
    integer
}

arith_term = { arith_primary ~ (arith_op_mul ~ arith_primary)* }
arith_expr = { arith_term ~ (arith_op_add ~ arith_term)* }

is_expr = { variable ~ "is" ~ arith_expr }
```

### Aggregates

```pest
agg_op = { "count" | "sum" | "min" | "max" | "logsumexp" }
aggregate = { agg_op ~ "(" ~ variable ~ ")" }
```

### Rules and Facts

```pest
epistemic_op = { ("know" | "possible") ~ !ident_continue }
not_kw = @{ "not" ~ !ident_continue }
nested_modal_chain = { not_kw? ~ epistemic_op ~ (not_kw? ~ epistemic_op)+ ~ not_kw? ~ atom }
epistemic_atom = { epistemic_op ~ atom }
negated_epistemic_atom = { "not" ~ epistemic_atom }
negated_atom = { "not" ~ atom }
meta_goal = {
    ("ground" | "var" | "nonvar") ~ "(" ~ term ~ ")"
    | "functor" ~ "(" ~ term ~ "," ~ term ~ "," ~ term ~ ")"
    | term ~ "=.." ~ term
    | "findall" ~ "(" ~ term ~ "," ~ atom ~ "," ~ term ~ ")"
    | "maplist" ~ "(" ~ predref_term ~ "," ~ term ~ ("," ~ term)? ~ ")"
}
body_literal = {
    nested_modal_chain
    | negated_epistemic_atom
    | epistemic_atom
    | negated_atom
    | meta_goal
    | atom
    | comparison
    | is_expr
}
body = { body_literal ~ ("," ~ body_literal)* }

head = { ident ~ "(" ~ head_term_list? ~ ")" }
rule_def = { head ~ ":-" ~ body ~ "." }
fact = { atom ~ "." }
constraint = { ":-" ~ body ~ "." }
query = { "?-" ~ atom ~ "." }
```

### Probabilistic Constructs

```pest
prob_num = @{ ASCII_DIGIT+ ~ ("." ~ ASCII_DIGIT+)? }
prob_choice = { prob_num ~ "::" ~ atom }
prob_fact = { prob_choice ~ "." }
annotated_disjunction = { prob_choice ~ (";" ~ prob_choice)+ ~ "." }

prob_query = { "query" ~ "(" ~ atom ~ ")" ~ "." }
evidence_stmt = { "evidence" ~ "(" ~ atom ~ "," ~ bool_lit ~ ")" ~ "." }
```

### Declarations

```pest
pred_decl = { private_mod? ~ "pred" ~ ident ~ "(" ~ type_list? ~ ")" ~ "." }
domain_decl = { "domain" ~ ident ~ ":" ~ type_spec ~ "." }

private_mod = { "private" }
type_annotation = { ":" ~ type_spec }
func_param = { variable ~ type_annotation? }
func_params = { func_param ~ ("," ~ func_param)* }
return_type = { "->" ~ type_spec }
cond_test = { arith_expr ~ cmp_op ~ arith_expr }
cond_expr = { "if" ~ cond_test ~ "then" ~ func_body ~ "else" ~ func_body }
func_body_arith = { cond_expr | arith_expr }
func_body_pred = { variable ~ ":-" ~ body }
func_body = { func_body_pred | func_body_arith }
func_def = {
    private_mod? ~ "func" ~ ident ~ "(" ~ func_params? ~ ")" ~ return_type?
    ~ "=" ~ func_body ~ "."
}
```

### Modules

```pest
module_path = @{ ident ~ ("/" ~ ident)* }
import_list = { "{" ~ ident ~ ("," ~ ident)* ~ "}" }
use_stmt = { "use" ~ module_path ~ ("::" ~ import_list)? ~ "." }
```

### Pragmas

```pest
prob_engine_value = { "exact_ddnnf" | "mc" }
prob_cache_value = { "on" | "off" }
epistemic_mode_value = { "g91" | "faeel" }
pragma_prob_engine = { "#pragma" ~ "prob_engine" ~ "=" ~ prob_engine_value }
pragma_prob_cache = { "#pragma" ~ "prob_cache" ~ "=" ~ prob_cache_value }
pragma_epistemic_mode = { "#pragma" ~ "epistemic_mode" ~ "=" ~ epistemic_mode_value }
pragma_max_recursion = { "#pragma" ~ "max_recursion_depth" ~ "=" ~ integer }
pragma_magic_sets = { "#pragma" ~ "magic_sets" ~ "=" ~ ("auto" | "on" | "off") }
pragma_prob_samples = { "#pragma" ~ "prob_samples" ~ "=" ~ integer }
pragma_prob_seed = { "#pragma" ~ "prob_seed" ~ "=" ~ integer }
pragma_prob_confidence = { "#pragma" ~ "prob_confidence" ~ "=" ~ float_num }
pragma_prob_method = { "#pragma" ~ "prob_method" ~ "=" ~ ("rejection" | "evidence_clamping") }
pragma_prob_max_nonmonotone_iterations = {
    "#pragma" ~ "prob_max_nonmonotone_iterations" ~ "=" ~ integer
}
pragma = {
    pragma_prob_engine
    | pragma_prob_cache
    | pragma_epistemic_mode
    | pragma_max_recursion
    | pragma_magic_sets
    | pragma_prob_samples
    | pragma_prob_seed
    | pragma_prob_confidence
    | pragma_prob_method
    | pragma_prob_max_nonmonotone_iterations
}
```

### CLI Examples

```bash
xlog run program.xlog
xlog run examples/epistemic/03-faeel-default.xlog
xlog prob program.xlog --prob-engine exact_ddnnf
xlog prob program.xlog --prob-engine mc --samples 10000 --seed 42
xlog explain --format json program.xlog
xlog repl
xlog watch --debounce-ms 250 program.xlog
```

### Program Structure

```pest
statement = {
    func_def
    | use_stmt
    | domain_decl
    | pred_decl
    | pragma
    | rule_def
    | prob_fact
    | annotated_disjunction
    | evidence_stmt
    | prob_query
    | fact
    | constraint
    | query
}

program = { SOI ~ statement* ~ EOI }
```

---

## See Also

- [Architecture Guide](/architecture/overview) - System design and implementation details
- [Language Completeness Architecture Contract](/architecture/language-completeness) - Parser, term, probability, CLI, and epistemic/solver handoff contract
- [Epistemic Semantics And EIR](/architecture/epistemic-internals) - Epistemic source surface, EIR boundary, and GPU runtime path
- [Arithmetic Expressions](/architecture/arithmetic) - Detailed `is` syntax documentation
- [Probabilistic Tier](/probabilistic/engines) - Exact and Monte Carlo inference
- [GPU Execution](/architecture/gpu-execution) - GPU-resident evaluation details
- [CLI Reference](/reference/cli) - Command-line interface documentation
- Examples - Annotated example programs
