# XLOG Language Reference

> **Version:** v0.5.0
> **Last Updated:** March 2026

This document provides a comprehensive reference for the XLOG language, covering all syntax, semantics, and features of the GPU-accelerated Datalog query engine.

---

## Table of Contents

1. [Overview](#overview)
2. [Basic Syntax](#basic-syntax)
3. [Data Types](#data-types)
4. [Predicates and Declarations](#predicates-and-declarations)
5. [Facts](#facts)
6. [Rules](#rules)
7. [Queries](#queries)
8. [Constraints](#constraints)
9. [Variables and Wildcards](#variables-and-wildcards)
10. [Comparisons](#comparisons)
11. [Arithmetic Expressions](#arithmetic-expressions)
12. [Negation](#negation)
13. [Aggregations](#aggregations)
14. [User-Defined Functions](#user-defined-functions)
15. [Modules](#modules)
16. [Symbols](#symbols)
17. [Probabilistic Logic](#probabilistic-logic)
18. [Term Embeddings](#term-embeddings)
19. [GPU ILP Configuration](#gpu-ilp-configuration)
20. [Pragmas and Directives](#pragmas-and-directives)
21. [Float Predicates and IEEE 754 Semantics](#float-predicates-and-ieee-754-semantics)
22. [Comments](#comments)
23. [Complete Grammar Reference](#complete-grammar-reference)

---

## Overview

XLOG is a GPU-accelerated Datalog query engine that compiles declarative logic programs into optimized relational plans and executes them on NVIDIA GPUs. It supports:

- **Datalog fundamentals**: Facts, rules, recursive queries, and stratified negation
- **Arithmetic operations**: Comparisons, computed values via `is`, and built-in functions
- **Aggregations**: `count`, `sum`, `min`, `max`, and `logsumexp`
- **Probabilistic reasoning**: Probabilistic facts, annotated disjunctions, evidence, and queries
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

### Reversible Symbols

Since v0.3.2, symbols are **reversible**: the original string value is preserved and displayed in query output. Symbols are stored internally as `u32` IDs with a bidirectional mapping to strings.

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

**Exact inference** does not support aggregation in probabilistic rule bodies.

**Monte Carlo** supports all programs including aggregation and non-monotone recursion:

```bash
xlog prob program.xlog --prob-engine mc --samples 10000 --seed 42
```

### Monte Carlo Sampling Methods (v0.5.0)

The MC engine supports two sampling methods, controlled by the `sampling_method` field on `McEvalConfig`:

| Method | Description |
|--------|-------------|
| `Rejection` | Sample from the prior, discard worlds where evidence is not satisfied. Default when evidence involves derived or deterministic atoms. |
| `EvidenceClamping` | Force evidence variables directly in the CUDA sampler so every sample satisfies evidence (`evidence_samples == total_samples`). Auto-selected when all evidence maps to probabilistic facts or positive annotated-disjunction heads; falls back to `Rejection` for derived, deterministic, or negative-AD evidence. |

Evidence clamping avoids wasted samples and is especially effective when evidence acceptance rates under rejection sampling are low. The chosen method is reported in `McResult.sampling_method` and `McDeviceResult.sampling_method`.

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
:- prob_engine = exact_ddnnf.

0.5::flip.
p :- flip.
q :- not p.
p :- not q.

query(p).  // WFS: p is undefined in worlds where bias is false
```

For approximate inference with confidence intervals:

```xlog
:- prob_engine = mc.
:- samples = 10000.

0.5::flip.
p :- flip.
q :- not p.

query(p).  // P(p) ≈ 0.5 ± CI
```

---

## Term Embeddings

*Added in v0.5.0.*

Term embeddings allow associating dense vector representations with discrete terms in XLOG programs. This is used in neural-symbolic training pipelines where learned embeddings feed into differentiable logic.

### `register_embedding()`

Register an embedding table with a compiled program via the Python API:

```python
import torch
import pyxlog

program = pyxlog.compile("program.xlog")

# Trainable embedding (nn.Embedding) — included in optimizer
emb = torch.nn.Embedding(num_embeddings=100, embedding_dim=64).cuda()
program.register_embedding("entity_emb", emb)

# Frozen embedding (raw Tensor) — detached at registration, never trained
frozen = torch.randn(100, 64).cuda()
program.register_embedding("pretrained_emb", frozen)
```

Cross-registration validation prevents mixing embedding and network names: a name registered with `register_embedding()` cannot be used with `register_network()`, and vice versa.

### `forward_embedding()`

Retrieve batched embedding tensors with autograd support:

```python
ids = torch.tensor([0, 5, 12], device="cuda")
vectors = program.forward_embedding("entity_emb", ids)
# vectors.shape == (3, 64), requires_grad=True for nn.Embedding
```

The returned tensor lives on the same device as the registered embedding (CUDA-safe). For frozen (raw tensor) embeddings, `requires_grad` is always `False`.

**Note:** The inference path for term embeddings is deferred to v0.5.1+. Currently only the training path is supported.

---

## GPU ILP Configuration

*Added in v0.5.0.*

The GPU-resident ILP (Inductive Logic Programming) credit/loss pipeline introduced in v0.5.0 includes configuration parameters and diagnostic APIs accessible from Python.

### `coo_chunk_budget`

Controls the per-chunk temporary allocation ceiling (in bytes) for the COO sparse matrix construction on GPU. The final exact-NNZ COO buffer may exceed this budget.

```python
program = pyxlog.compile("program.xlog")

# Set chunk budget to 256 MB
program.set_coo_chunk_budget(256 * 1024 * 1024)
```

The previous name `coo_memory_cap` (and `set_coo_memory_cap()`) is deprecated but retained for one release cycle.

### `host_transfer_stats()`

Returns D2H (device-to-host) transfer accounting counters, useful for verifying zero-copy GPU execution:

```python
stats = program.host_transfer_stats()
print(stats["dtoh_calls"])   # Number of D2H transfer calls
print(stats["dtoh_bytes"])   # Total bytes transferred D2H
```

Use with `set_strict_zero_dtoh(True)` to enforce that no data-plane D2H transfers occur during ILP computation, which is critical for performance benchmarking and CI gates.

---

## Pragmas and Directives

Pragmas configure compiler and runtime behavior.

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

### Examples

```xlog
// Use Monte Carlo inference
#pragma prob_engine = mc

// Disable probability caching
#pragma prob_cache = off

// Limit recursion depth
#pragma max_recursion_depth = 100
```

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

The following is a summary of the XLOG grammar in PEG notation.

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

### Terms and Atoms

```pest
term = { var_or_anon | float_num | integer | string_lit | ident }
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
body_literal = { negated_atom | atom | comparison | is_expr }
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
pragma = { pragma_prob_engine | pragma_prob_cache | pragma_max_recursion }
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
- [Arithmetic Expressions](architecture/arithmetic-expressions.md) - Detailed `is` syntax documentation
- [Probabilistic Tier](architecture/xlog-prob.md) - Exact and Monte Carlo inference
- [GPU Execution](architecture/gpu-execution.md) - GPU-resident evaluation details
- [CLI Reference](architecture/cli-reference.md) - Command-line interface documentation
- [Examples](../examples/) - Annotated example programs
