# Negation

Rule out facts with stratified negation-as-failure — closed-world, safe, and checked at compile time.

XLOG lets a rule succeed *because* something cannot be derived. Prefix a body atom with
`not` and it holds exactly when the atom does not:

```xlog
isolated(X) :- node(X), not edge(X, _).
```

A node is isolated when it is a node and there is no edge leaving it.

Reach for `not` when the *absence* of a fact is what you care about — isolated nodes,
unreached states, records with no match. You know it worked when the rule compiles and
runs: if a variable is left unbound or the program cannot be stratified, XLOG rejects it
at compile time with an explicit error (both cases are covered below).

## Negation-as-failure, closed-world

`not atom` is **negation-as-failure** under the **closed-world assumption**: the engine
first evaluates the positive relation to its fixpoint, and if a matching fact cannot be
derived, the negation succeeds. There is no separate notion of "false" — anything the
program does not prove is taken to be false. So `not edge(X, _)` succeeds precisely for
those `X` that never appear as the source of any derived `edge`.

## Negation must be stratifiable

Negation is **stratified**. XLOG splits the program into strata so that every negated
predicate is fully computed in an earlier stratum before any rule negates it. That is only
possible when **no predicate depends — directly or through a cycle — on its own
negation**. A program where `p` is defined in terms of `not p` has no stratification, and
XLOG rejects it at compile time with a stratification error rather than guessing at a
meaning.

<Note>
The practical rule: recursion may flow through positive atoms, but never through
negation. If you need to negate a recursive relation, compute it to its fixpoint in an
earlier rule, then negate the finished result.
</Note>

## Safety: bind before you negate

A negated atom filters candidates; it never invents them. So every named variable inside
`not` must already be **bound by a positive atom** — or by an earlier `is` — that appears
**before it in the body**. Source order matters: the binding literal has to come first.

```xlog
// Safe: X is bound by node(X) before the negation uses it.
isolated(X) :- node(X), not edge(X, _).
```

Here `X` is bound by `node(X)`, so `not edge(X, _)` is a well-defined test for each
particular `X`. Reverse the two literals and `X` would be unbound where the negation
needs it — an unsafe rule, and a compile-time error:

```xlog
// Unsafe: X is not yet bound when `not` runs. Rejected at compile time.
isolated(X) :- not edge(X, _), node(X).
```

For a position inside the negated atom whose value you do not care about, use `_`, the
anonymous wildcard. It matches any value existentially — `not edge(X, _)` asks whether `X`
has *any* outgoing edge at all — and, because it introduces no named variable, it never
needs to be bound beforehand.

## A different kind of negation for probabilities

The `not` described here is the deterministic, two-valued negation of ordinary Datalog
rules. Probabilistic programs use a distinct **well-founded** negation — a negation semantics
that stays well-defined even when negation and recursion interact — over uncertain
facts, which is evaluated differently and lives in its own surface.

<Card title="Arithmetic and functions" icon="calculator" href="/language-guide/arithmetic-and-functions">
  Bind variables with `is`, compare values, and factor logic into reusable functions.
</Card>
