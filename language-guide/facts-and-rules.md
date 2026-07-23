# Facts and rules

Declare typed predicates, state facts, and derive new relations with rules — the foundation of every XLOG program.

An XLOG program is a set of **facts** (data you assert) and **rules** (how to derive
more data from what you have). You declare the shape of your data with typed
predicates, then let the engine compute everything the rules imply.

## Predicates and facts

A predicate is a named, typed relation. Declare it with `pred`, giving a type for each
column:

```xlog
pred edge(u32, u32).
```

A fact is a predicate applied to concrete values, ending in a period:

```xlog
edge(1, 2).
edge(2, 3).
edge(3, 4).
```

Column types come from the eight scalar types — `u32`, `u64`, `i32`, `i64`, `f32`,
`f64`, `bool`, and `symbol`. Every fact must match its declaration; a value of the
wrong type is a compile-time error, caught before any kernel runs. You can also name
columns for readability:

```xlog
pred edge(src: u32, dst: u32).
```

<Note>
`symbol` values are interned strings backed by dense integer IDs. They compare and
join as fast as integers, but XLOG remembers the original text and prints it back in
query output — so you get readable results without giving up integer performance.
</Note>

## Rules

A rule derives new facts. It has a head (what it produces) and a body (the conditions),
separated by `:-`, which reads as "if":

```xlog
pred reach(u32, u32).

reach(X, Y) :- edge(X, Y).
```

This says: `reach(X, Y)` holds whenever `edge(X, Y)` holds. Names beginning with an
uppercase letter are **variables**; names beginning with lowercase are predicate or
value names. The rule above copies every edge into `reach`.

## Joins are shared variables

When a rule body has more than one atom, they are joined on the variables they share.
To follow an edge and then another edge, mention the same variable `Y` in both:

```xlog
reach(X, Z) :- edge(X, Y), edge(Y, Z).
```

The comma is conjunction ("and"). Because `Y` appears in both `edge` atoms, the engine
joins them where the destination of the first equals the source of the second — a hash
join, executed on the GPU. Variables that appear only once can be written as `_`, the
anonymous wildcard, when you do not need their value:

```xlog
has_outgoing(X) :- edge(X, _).
```

Each `_` is a fresh, independent placeholder — two underscores in the same rule are not
required to match.

## Querying results

A query asks the engine to compute and print a relation. Write it with `?-`:

```xlog
?- reach(1, N).
```

Running a program evaluates every rule to a **fixpoint** — it keeps applying rules
until no new facts can be derived — then prints the queries:

```bash
xlog run reachability.xlog
```

```text
__xlog_query_0
+-------+
| col_0 |
+-------+
| 2     |
| 3     |
| 4     |
+-------+
```

Under the hood, a query is desugared into an ordinary rule whose head the runner
prints. An **integrity constraint** is the mirror image — a headless rule that must
never hold:

```xlog
:- edge(X, X).
```

If any fact matches the body, evaluation fails with a constraint violation. Constraints
are how you state invariants your data must satisfy.

## Putting it together

```xlog
pred edge(src: u32, dst: u32).
pred reach(u32, u32).

edge(1, 2).
edge(2, 3).
edge(3, 4).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- edge(X, Y), reach(Y, Z).

?- reach(1, N).
```

The second `reach` rule refers to `reach` in its own body — that is recursion, and it
is where XLOG's **semi-naive** fixpoint evaluation — which each round works only from
the facts newly derived in the previous round, rather than recomputing everything —
does its work.

<Card title="Recursion" icon="arrows-rotate" href="/language-guide/recursion">
  How recursive rules are evaluated to a fixpoint, and how to keep them terminating.
</Card>
