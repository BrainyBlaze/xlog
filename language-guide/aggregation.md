# Aggregation

Collapse groups of rows into a single value — count, sum, min, max, and logsumexp — computed in a rule head with implicit group-by.

Joins and arithmetic work row by row. **Aggregation** goes the other way: it collapses
many rows into one summary value — how many neighbours a node has, the total weight in a
graph, the maximum score per player. In XLOG you write an aggregate directly in a rule
head.

## Where aggregates go

An aggregate may appear **only in a rule head**, never in a body. This is a deliberate
restriction: the head is where the summary lands, and keeping aggregates out of bodies
keeps evaluation well-defined. The two rules below are the canonical shapes:

```xlog
out_degree(X, count(Y)) :- edge(X, Y).
total(sum(W)) :- weight(_, W).
```

The first counts, for each `X`, how many distinct `edge(X, Y)` rows exist. The second
sums the second column of every `weight` fact into a single total.

## The five aggregates

XLOG provides exactly five aggregate operators — no others:

| Aggregate | Meaning |
|---|---|
| `count(X)` | Number of rows in the group |
| `sum(X)` | Sum of `X` over the group |
| `min(X)` | Smallest `X` in the group |
| `max(X)` | Largest `X` in the group |
| `logsumexp(X)` | Log of the sum of exponentials — the numerically stable "soft maximum" used in probabilistic and neural work |

<Note>
There is **no built-in `avg`**. Compute an average as `sum` divided by `count` — derive
each with its own rule, then combine them:

```xlog
weight_sum(G, sum(W)) :- weighted(G, W).
weight_count(G, count(W)) :- weighted(G, W).
mean(G, M) :- weight_sum(G, S), weight_count(G, N), M is S / N.
```
</Note>

## Group-by is implicit

You never write a `GROUP BY` clause. The non-aggregate variables in the head **are** the
grouping key. In:

```xlog
out_degree(X, count(Y)) :- edge(X, Y).
```

`X` is not aggregated, so the engine forms one group per distinct `X` and computes
`count(Y)` within each. When the head has no non-aggregate variable, the whole relation
is a single group — that is exactly what `total(sum(W))` does, producing one global
total.

## Value types

An aggregate's result type follows from the operator and the type of the column it
consumes:

| Aggregate | Input type | Result type |
|---|---|---|
| `count` | any type | `u64` |
| `sum` | `u32` | `u64` |
| `min` | `u32` | `u32` |
| `max` | `u32` | `u32` |
| `logsumexp` | `f64` | `f64` |

`count` always returns a `u64` regardless of what it counts. `sum` widens a `u32` column
to `u64` so a large total cannot overflow the count of narrow inputs, while `min` and
`max` return the same type they consume — they only select an existing value, so no
widening is needed.

## Aggregation is stratified

**When this matters:** only if an aggregate's inputs could loop back to its own result —
usually recursive rules; otherwise you can skip this section. *Stratified* means xlog
settles those inputs in order, computing the aggregate only after everything it reads is
final.

An aggregate introduces a **stratification boundary**: a relation defined by aggregation
cannot depend, recursively, on itself through that aggregate. Concretely, you cannot
have a rule whose head aggregates a predicate that (directly or transitively) depends on
the head. The engine computes each aggregate only once its inputs are fully determined,
so results are well-defined rather than chasing a moving target.

This mirrors how negation is stratified in [Facts and
rules](/language-guide/facts-and-rules): both need their inputs settled before they can
produce a sound answer.

<Warning>
Under exact probabilistic inference, aggregates over *uncertain* rows have finite domain
caps — beyond them you switch to the Monte Carlo engine. That is a property of the
probabilistic engines, not of deterministic aggregation, and is covered separately.
</Warning>

<Card title="Probabilistic engines" icon="dice" href="/probabilistic/engines">
  Exact versus Monte Carlo inference, and the finite caps on exact probabilistic
  aggregation.
</Card>

<Card title="Lists and meta" icon="brackets-square" href="/language-guide/lists-and-meta">
  Build and pattern-match lists, and collect results with the meta-predicates.
</Card>
