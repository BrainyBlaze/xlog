# XLOG Language Reference

> **Release context:** XLOG `v0.8.5`
> **Language coverage:** Core v0.8.0 language plus the v0.8.5 language-completeness contract
> **Last Updated:** May 2026

This document provides a comprehensive reference for the XLOG language,
covering all syntax, semantics, and features of XLOG as a GPU-native logic
programming language.

The v0.8.5 additions are specified here as the release contract. During the
v0.8.5 implementation branch, each addition remains unsupported until its
corresponding `G085_*` implementation node lands. Unsupported forms must fail
with typed diagnostics rather than silently falling back to CPU evaluation.

---

## Table of Contents

1. [Overview](#overview)
2. [Basic Syntax](#basic-syntax)
3. [Data Types](#data-types)
4. [v0.8.5 Language Contract](#v085-language-contract)
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
22. [Approximate Inference](#approximate-inference)
23. [Neural Predicates](#neural-predicates)
24. [Learnable Rules](#learnable-rules)
25. [Term Embeddings](#term-embeddings)
26. [GPU ILP Configuration](#gpu-ilp-configuration)
27. [Pragmas and Directives](#pragmas-and-directives)
28. [CLI Developer Experience](#cli-developer-experience)
29. [Float Predicates and IEEE 754 Semantics](#float-predicates-and-ieee-754-semantics)
30. [Comments](#comments)
31. [Complete Grammar Reference](#complete-grammar-reference)

---

## Overview

XLOG is a GPU-native logic programming language that compiles typed, modular
logic programs into backend-specific GPU execution paths. It supports:

- **Datalog fundamentals**: Facts, rules, recursive queries, and stratified negation
- **Arithmetic operations**: Comparisons, computed values via `is`, and built-in functions
- **Aggregations**: `count`, `sum`, `min`, `max`, and `logsumexp`
- **Probabilistic reasoning**: Probabilistic facts, annotated disjunctions, evidence, and queries
- **v0.8.5 language contract**: Finite lists, safe meta-predicates, explicit
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
| Use | `use module.` | Imports predicates from a module |
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

Types are inferred from predicate declarations. All uses of a predicate must be consistent with its declared types.

```xlog
pred temperature(u32, f64).   // sensor_id: u32, temp: f64

temperature(1, 22.5).         // Correct: u32, f64
temperature(1, 22).           // Error: 22 inferred as i64, not f64
temperature(1, cast(22, f64)). // Correct: explicit cast
```

### v0.8.5 Type Forms

v0.8.5 extends the type model with finite language-level terms that must lower
to typed relation layouts before execution:

| Type form | Meaning | Execution status |
|-----------|---------|------------------|
| `domain name: type` | Named alias for an existing scalar type | Existing source form; v0.8.5 requires alias preservation in diagnostics and schema metadata |
| `column: type` | Named predicate column | v0.8.5 contract; rejected until `G085_TYPES` lands |
| `list<T>` | Finite homogeneous list of `T` | v0.8.5 contract; accepted lists lower to helper relations, not CPU term heaps |
| `term` | Finite ground term value | v0.8.5 contract; only finite, typed terms are accepted |
| `compound` | Finite compound term with functor and arity | v0.8.5 contract; unsupported recursive or open compounds are rejected |
| `predref` | Static predicate reference for `maplist` and related meta use | v0.8.5 contract; runtime-variable predicate names are rejected |

Unsupported type forms must fail during parsing or semantic analysis before
runtime execution starts.

---

## v0.8.5 Language Contract

v0.8.5 is a language-surface expansion over the v0.8.0 runtime. New syntax is
accepted only when it can be normalized into the existing typed AST, RIR,
probabilistic IR, optimizer, runtime, WCOJ, and CLI paths.

### Execution Responsibilities

| Layer | Allowed work |
|-------|--------------|
| Parser and AST | Recognize source syntax, preserve spans, and represent finite high-level terms |
| Semantic analysis | Type check, safety check, reject unbounded or dynamic forms, and desugar high-level constructs |
| Optimizer and planner | Rewrite programs, including magic sets, while preserving output semantics |
| Runtime and probability engines | Execute accepted relational, aggregate, exact, and MC work through production GPU-capable paths |
| CLI | Orchestrate commands, format diagnostics, and transfer only requested final results |

Accepted v0.8.5 features must not execute by constructing an arbitrary CPU
Prolog term heap, running dynamic CPU predicate calls, or using a hidden
CPU-only fallback for a GPU-claimed path.

### Feature Coverage Matrix

| Feature | Syntax | Semantics | Example | Unsupported forms | Execution path |
|---------|--------|-----------|---------|-------------------|----------------|
| Incremental parsing | Statement-level `.xlog` source units with stable spans | Reuse unchanged statement parses and invalidate changed statements plus dependent module state | Editing one rule in a REPL/watch session preserves parse cache for unchanged facts | Cache reuse across changed module dependencies without invalidation | Parser/session cache only; no runtime execution |
| List syntax and built-ins | `[]`, `[A, B]`, `[H|T]`, `list<T>` | Finite, typed lists normalize to helper relations | `has_member(X) :- member(X, [1,2,3]).` | Open-ended generators, cyclic lists, heterogeneous lists without a declared finite term type | AST/desugar to typed helper relations and normal runtime joins/aggregates |
| Safe meta-predicates | `ground`, `var`, `nonvar`, `functor`, `=..`, `findall`, `maplist` | Static inspection and finite collection only | `xs(L) :- findall(X, edge(1, X), L).` | Unrestricted `call/N`, dynamic database mutation, runtime-variable predicate names | Compile-time/static expansion plus typed relational lowering |
| Deterministic NAF | `not atom(...)` | Closed-world stratified negation over deterministic relations | `leaf(X) :- node(X), not edge(X, _).` | Unbound variables in negated atoms, unstratified deterministic cycles | Existing stratification and runtime anti-join paths |
| Magic sets | `#pragma magic_sets = on|off|auto` and CLI override | Bound recursive queries may be rewritten with adornments and magic predicates | `?- reach(1, Y).` specializes recursive reachability | Unsafe interaction with negation, aggregates, or meta constructs if equivalence cannot be proven | Source/RIR rewrite before optimizer and WCOJ planning |
| Probabilistic aggregates | Aggregate heads and aggregate outputs in `query`/`evidence` | Finite aggregate outcomes in exact and MC inference | `query(out_degree(1, 2)).` over probabilistic `edge` facts | Exact aggregate domains over cap, unsupported numeric operator/domain pairs | Exact provenance/PIR or MC sampling plus deterministic aggregate execution |
| Aggregate lifting | Finite-domain aggregate metadata and caps | Use lifted compact-domain computation when identical to finite exact enumeration | Small count/sum domains avoid naive enumeration | Domain cap exceeded, non-finite domains, unsupported floating tolerance | Probabilistic aggregate planner and exact/MC engines |
| Approximate inference | `#pragma prob_engine = mc`, samples, seed, confidence, method | MC estimates are reproducible under fixed seed and report uncertainty | `#pragma prob_samples = 10000` with `query(rain).` | Invalid confidence ranges, unsupported methods, hidden default override ambiguity | Existing MC engine with documented source/CLI precedence |
| CLI explain | `xlog explain --format text|json|dot file.xlog` | Show parse, strata, RIR, optimized RIR, magic-set, WCOJ, and probability sections when applicable | `xlog explain --format json reach.xlog` | Unknown output formats or unavailable sections without a typed reason | CLI orchestration plus compiler/explain APIs |
| CLI REPL | `xlog repl` | Interactive multiline fact/rule/query session using parser cache | Add facts, then query relation state | GPU execution before a command requests it, unsupported mutation semantics | CLI session cache plus normal compile/run on submitted programs |
| CLI watch | `xlog watch file.xlog` | Rerun or re-explain changed files with debounce and typed diagnostics | Edit a rule and see updated diagnostics/output | Silent stale cache reuse after file/module change | CLI file watcher plus parser/session cache and normal commands |

### Diagnostic Contract

Unsupported v0.8.5 forms must report:

- the feature area, such as `list`, `meta`, `magic_sets`, or `prob_aggregate`;
- the source span when available;
- the reason the form is unsafe, unbounded, or not GPU-lowerable;
- a remediation, such as adding a positive binder, a finite cap, or a static
  predicate reference.

Diagnostic examples:

```text
list error at program.xlog:12: unbounded append/3 generation is unsupported; bind at least two list arguments or add a finite split mode.
meta error at program.xlog:8: maplist predicate argument must be a static predref; replace runtime variable P with a named predicate.
magic_sets declined at program.xlog:21: recursive query crosses an aggregate boundary; run with #pragma magic_sets = off or isolate the aggregate.
prob_aggregate error at program.xlog:17: exact aggregate domain exceeds cap 1024; add a finite cap or use prob_engine = mc.
```

### Source-Audit Status

At the `G085_DOCREF` checkpoint, the current source still rejects the new
v0.8.5 parser/runtime forms except where an older feature already exists, such
as scalar terms, deterministic `not atom`, deterministic aggregates, exact/MC
probabilistic inference, and the `run`/`prob` CLI commands. Later `G085_*`
implementation nodes must update this section and the grammar appendix when
they turn a contract row into shipped support.

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
public_api(X) :- helper(X).
```

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

XLOG supports **stratified negation** using the `not` keyword. Negated atoms must be **stratifiable** (no cycles through negation).

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

XLOG detects unstratifiable programs at compile time and reports an error.

### Domain Safety

Variables in negated atoms must be **bound** by positive atoms in the same rule:

```xlog
// Valid: X is bound by node(X) before used in not edge(X, _)
leaf(X) :- node(X), not edge(X, _).

// Invalid: Y is not bound by any positive atom
// bad(X) :- node(X), not edge(X, Y).  // ERROR
```

---

## Aggregations

Aggregations compute summary values over groups of tuples.

### Supported Aggregates

| Aggregate | Description | Input Types | Output Type |
|-----------|-------------|-------------|-------------|
| `count(X)` | Count of values | Any | `u64` |
| `sum(X)` | Sum of values | `u32` | `u64` |
| `min(X)` | Minimum value | `u32` | `u32` |
| `max(X)` | Maximum value | `u32` | `u32` |
| `logsumexp(X)` | Log-sum-exp (numerically stable) | `f64` | `f64` |

### Syntax

Aggregates appear in the rule head. Variables not in the aggregate form the grouping key:

```xlog
head(GroupKeys, aggregate(AggVar)) :- body(GroupKeys, AggVar, ...).
```

### Examples

```xlog
pred edge(u32, u32).
pred out_degree(u32, u32).

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
pred cnt(u32, u32).
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

v0.8.5 defines finite list syntax and list built-ins. Lists are accepted only
when they are finite and typed; accepted programs lower to relation layouts and
normal GPU-capable execution paths.

Implementation status after `G085_META`: scalar, symbol, nested finite list,
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

- `append(A, B, C)` accepts finite first and second list literals in this node;
  split generation such as `append(A, B, [1,2,3])` is rejected as unbounded;
- `sort`, `msort`, and `list_to_set` require a finite input literal in this
  node;
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

The following forms are outside the v0.8.5 contract:

- unbounded list generation, such as asking `append(A, B, [1,2,3])` to
  enumerate every split unless a finite split mode is explicitly implemented;
- cyclic lists;
- heterogeneous lists unless represented through a declared finite `term`
  domain;
- list evaluation that requires an arbitrary CPU Prolog term heap.

---

## Safe Meta-Predicates

v0.8.5 safe meta-predicates provide finite term inspection and static
predicate mapping. They do not introduce an unrestricted Prolog dynamic
database or unrestricted `call/N`.

Implementation status after `G085_META`: accepted safe meta forms normalize
before stratification and lowering into ordinary helper relations such as
`__xlog_meta_functor`, `__xlog_meta_univ`, `__xlog_meta_findall_*`, and
`__xlog_meta_maplist_*`. Finite `term`, `compound`, and `predref` predicate
columns lower to `u64` term IDs; compound metadata and univ parts are stored in
typed helper relations. `findall` and `maplist` currently accept finite source
facts and finite list literals only; derived-goal collection and non-literal
maplist inputs are explicitly rejected until an aggregate-backed collection
path lands.

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
- non-literal `maplist` inputs in the current `G085_META` subset;
- constructing recursive or open compound terms with `=..`;
- higher-arity `maplist` unless explicitly implemented and tested.

---

## Magic Sets

Magic-set rewriting specializes bound recursive queries by adding derived
magic predicates and adorned rules before optimization.

### Configuration

```xlog
#pragma magic_sets = auto
#pragma magic_sets = on
#pragma magic_sets = off
```

`auto` lets the compiler apply the rewrite only when it can prove that the
transformed program preserves output. `on` requests the rewrite and fails with
a typed diagnostic if the compiler cannot safely apply it. `off` disables the
rewrite.

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

### Unsupported Forms

Magic-set rewriting must decline or fail before execution when negation,
aggregates, list/meta constructs, or probabilistic rules would make equivalence
uncertain. It must not create a runtime side engine.

---

## User-Defined Functions

XLOG supports user-defined functions for reusable calculations.

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

### Recursive Functions

Functions can be recursive but must have a base case:

```xlog
func factorial(N) = if N <= 1 then 1 else N * factorial(N - 1).

func fib(N) = if N <= 1 then N else fib(N - 1) + fib(N - 2).

func gcd(A, B) = if B = 0 then A else gcd(B, A % B).
```

### Using Functions in Rules

Functions can be called in rule bodies using the `is` expression or direct call syntax:

```xlog
// Using 'is' expression
pred doubled_value(u32, u32).
doubled_value(X, Y) :- input(X, V), Y is double(V).

// Direct call syntax (function result assigned to variable)
pred result(u32, u32).
result(X, Y) :- input(X, V), double(V) = Y.
```

### Type Annotations

Optional type annotations for documentation and checking:

```xlog
func add(X: f64, Y: f64) -> f64 = X + Y.
func celsius_to_fahrenheit(C: f64) -> f64 = C * 1.8 + 32.0.
```

### Private Functions

Functions can be marked private:

```xlog
private func helper(X) = X * X + 1.
func public_calc(X) = helper(X) * 2.
```

---

## Modules

XLOG supports organizing code into modules for reusability and encapsulation.

### Creating Modules

A module is a `.xlog` file. The filename (without extension) becomes the module name:

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

Use the `use` statement to import predicates from modules:

```xlog
// Import all public predicates
use graph.

// Import specific predicates
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
// In main.xl
use graph.
use utils/math.
```

### Visibility and Encapsulation

Predicates and functions are **public by default**. Use `private` to hide implementation details:

```xlog
// File: graph.xlog

// Public API
pred edge(u32, u32).
pred reach(u32, u32).

// Private helper (not visible outside this module)
private pred internal_cache(u32, u32).

reach(A, B) :- edge(A, B).
reach(A, C) :- internal_cache(A, C).  // Uses private helper
internal_cache(A, C) :- edge(A, B), reach(B, C).
```

Private predicates cannot be imported by other modules:

```xlog
// ERROR: internal_cache is private
use graph::{internal_cache}.
```

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

### Reversible Symbols (v0.3.2)

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
| **Exact** | `--prob-engine exact_ddnnf` | Knowledge compilation via D4; exact probabilities |
| **Monte Carlo** | `--prob-engine mc` | Sampling-based; approximate with confidence intervals |

**Exact inference** supports:
- **Stratified negation**: Automatic layer detection and two-valued evaluation
- **Non-monotone negation**: Well-Founded Semantics (WFS) for cyclic programs
- **Gradients**: Correct gradient flow through negated literals

**Probabilistic aggregates in v0.8.5:** finite `count`, `sum`, `min`, `max`,
and `logsumexp` aggregate programs are part of the v0.8.5 contract for exact
and MC inference. At the `G085_DOCREF` source-audit checkpoint, the current
exact provenance extractor still rejects aggregate terms; `G085_PROB_AGG` must
replace that generic rejection with supported execution or a typed per-case
decline.

**Monte Carlo** supports probabilistic rules and non-monotone recursion.
Aggregate support follows the v0.8.5 probabilistic aggregate contract and must
run through GPU sampling plus deterministic aggregate execution when accepted:

```bash
xlog prob program.xlog --prob-engine mc --samples 10000 --seed 42
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

Exact mode must either compile the finite aggregate outcomes into provenance
without host-only recomputation or fail with a typed diagnostic that explains
the unsupported operator, domain, or cap. MC mode may sample worlds and execute
the deterministic aggregate path for each accepted sample batch.

### Aggregate Lifting

Small-domain aggregate lifting is permitted only when the lifted computation is
semantically identical to finite exact enumeration. Explain output must report
the detected finite domain, cap, operator, and whether lifting fired, declined,
or fell back to supported exact enumeration.

### Monte Carlo Sampling Methods (v0.5.0)

The Monte Carlo engine supports two sampling methods:

| Method | Description |
|--------|-------------|
| `Rejection` | Sample from the prior and discard worlds where evidence is not satisfied. |
| `EvidenceClamping` | Force evidence variables directly in the CUDA sampler when the evidence is forceable at the probabilistic-fact / positive-AD level. |

Evidence clamping avoids wasted samples when rejection acceptance rates are
low. If evidence is derived, deterministic, or otherwise not directly
forceable, XLOG falls back to rejection sampling.

### Negation in Probabilistic Programs

Exact inference supports both stratified and non-monotone negation:

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

*Introduced in v0.5.0.*

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
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
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

*Introduced in v0.5.0.*

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

- [`architecture/dilp-training.md`](architecture/dilp-training.md)
- [`architecture/rfc-tensorized-ilp.md`](architecture/rfc-tensorized-ilp.md)
- [`architecture/python-bindings.md`](architecture/python-bindings.md)

---

## Term Embeddings

*Introduced in v0.5.0.*

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

*Introduced in v0.5.0.*

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

Training APIs expose transfer telemetry so strict zero-D2H workflows can verify
that the hot path stayed device-resident:

```python
stats = trainer.host_transfer_stats()
trainer.reset_host_transfer_stats()
```

---

## Pragmas and Directives

Pragmas configure compiler and runtime behavior.

At the `G085_DOCREF` checkpoint, the current source grammar accepts
`prob_engine`, `prob_cache`, and `max_recursion_depth`. The additional v0.8.5
pragmas below define the release contract and remain implementation-gated until
their corresponding `G085_*` nodes land.

### Syntax

```xlog
#pragma key = value
```

### Available Pragmas

| Pragma | Values | Description |
|--------|--------|-------------|
| `prob_engine` | `exact_ddnnf`, `mc` | Select probabilistic inference engine |
| `prob_cache` | `on`, `off` | Enable/disable probability caching |
| `max_recursion_depth` | integer | Maximum recursion iterations |
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

v0.8.5 adds developer-experience commands that make the language inspectable
and interactive without changing the runtime execution contract.

### `xlog explain`

```bash
xlog explain [--format text|json|dot] [--prob-engine exact_ddnnf|mc] <FILE>
```

Explain output must include deterministic sections when applicable:

- parse and AST summary;
- predicate declarations and schema metadata;
- stratification and negation safety;
- RIR and optimized RIR summaries;
- magic-set adornments, generated predicates, and declined reasons;
- WCOJ eligibility and selected execution path;
- probabilistic engine, aggregate, and approximate-inference sections for
  probabilistic programs.

JSON output must be deterministic for fixed input. Unknown formats are rejected
before compilation.

### `xlog repl`

```bash
xlog repl [--module-path DIR[:DIR...]]
```

The REPL accepts multiline facts, rules, declarations, pragmas, and queries.
It shares the incremental parser/session cache and does not require GPU access
until the user submits a command that needs execution.

### `xlog watch`

```bash
xlog watch [--debounce-ms N] [--explain] <FILE>
```

Watch mode reruns compilation, execution, or explanation after file changes.
It must debounce rapid writes, invalidate changed statements and dependent
module state, and display typed diagnostics without reusing stale parse
results.

Unsupported REPL/watch mutation semantics, such as dynamic `assert` or
`retract`, remain outside the language.

---

## Float Predicates and IEEE 754 Semantics

XLOG uses **hybrid semantics** for floating-point comparisons to handle special values (NaN, Infinity, signed zero) correctly.

### Comparison Semantics

| Operator | Semantics | NaN Behavior |
|----------|-----------|--------------|
| `=`, `!=` | IEEE 754 equality | `NaN = NaN` is **FALSE** |
| `<`, `<=`, `>`, `>=` | Total ordering | NaN is comparable |

### Total Ordering

For ordering comparisons, XLOG uses a **total ordering** consistent with Rust's `f64::total_cmp`:

```
-NaN < -Inf < ... < -0.0 < +0.0 < ... < +Inf < +NaN
```

This means:
- NaN values are at the extremes (negative NaN is smallest, positive NaN is largest)
- `-0.0` and `+0.0` are distinguishable in ordering (unlike IEEE equality)
- All values are comparable (no undefined ordering)

### Filtering NaN Values

Since `NaN != NaN` under IEEE 754, use equality to filter out NaN:

```xlog
// V = V is true only for non-NaN values
valid(Id, V) :- data(Id, V), V = V.
```

### Detecting Special Values

Use ordering to detect special values:

```xlog
// NaN is greater than any finite value
is_nan(Id, V) :- data(Id, V), V > 1e308.

// Infinity detection
is_infinite(Id, V) :- data(Id, V), V > 1e300, V = V.

// Finite values only
is_finite(Id, V) :- data(Id, V), V > -1e308, V < 1e308.
```

### Example: Data Cleaning Pipeline

```xlog
pred raw(u32, f64).
pred valid(u32, f64).
pred anomaly(u32, f64).

raw(1, 22.5).
raw(2, 0.0).        // Will compute NaN from 0/0
raw(3, 28.3).

// Filter valid readings (not NaN)
valid(Id, V) :- raw(Id, V), V = V.

// Detect anomalies (NaN or extreme values)
anomaly(Id, V) :- raw(Id, V), V > 1000.0.
anomaly(Id, V) :- raw(Id, V), V < -1000.0.
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

The following is a summary of the current XLOG grammar plus the v0.8.5 contract
extensions in PEG-style notation. Contract extensions are source-audited again
as their implementation nodes land.

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

v0.8.5 contract:

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

v0.8.5 contract:

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
negated_atom = { "not" ~ atom }
meta_goal = {
    ("ground" | "var" | "nonvar") ~ "(" ~ term ~ ")"
    | "functor" ~ "(" ~ term ~ "," ~ term ~ "," ~ term ~ ")"
    | term ~ "=.." ~ term
    | "findall" ~ "(" ~ term ~ "," ~ atom ~ "," ~ term ~ ")"
    | "maplist" ~ "(" ~ predref_term ~ "," ~ term ~ ("," ~ term)? ~ ")"
}
body_literal = { negated_atom | meta_goal | atom | comparison | is_expr }
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

func_param = { variable ~ type_annotation? }
func_params = { func_param ~ ("," ~ func_param)* }
return_type = { "->" ~ type_spec }
cond_expr = { "if" ~ cond_test ~ "then" ~ func_body ~ "else" ~ func_body }
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
pragma_prob_engine = { "#pragma" ~ "prob_engine" ~ "=" ~ prob_engine_value }
pragma_prob_cache = { "#pragma" ~ "prob_cache" ~ "=" ~ prob_cache_value }
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

- [Architecture Guide](ARCHITECTURE.md) - System design and implementation details
- [v0.8.5 Language Architecture Contract](architecture/language-v085.md) - Parser, term, probability, CLI, and v0.9.0 handoff contract
- [Arithmetic Expressions](architecture/arithmetic-expressions.md) - Detailed `is` syntax documentation
- [Probabilistic Tier](architecture/xlog-prob.md) - Exact and Monte Carlo inference
- [GPU Execution](architecture/gpu-execution.md) - GPU-resident evaluation details
- [CLI Reference](architecture/cli-reference.md) - Command-line interface documentation
- [Examples](../examples/) - Annotated example programs
