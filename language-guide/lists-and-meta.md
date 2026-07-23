# Lists and meta-predicates

Carry a small ordered collection of values in one column, pull it apart with head/tail patterns, and use XLOG's compile-time meta-predicates to inspect and collect terms.

Use a **list** when a single fact needs to carry several values in order — the nodes
on a path, the tags on an item — instead of spreading them across many columns. Use a
**meta-predicate** when you want to look at the *shape* of a term (its name, its
arguments, its elements) or gather many answers into one list.

This page shows how to write lists, take them apart, and reach for the small set of
meta-predicates XLOG supports. XLOG keeps these features tightly bounded, and this page
is explicit about where the bounds are so you don't hit a surprise later.

## Finite lists

**When to use this.** You want one column to hold an ordered group of values.

**How.** Write a list literal with square brackets and commas:

```xlog
[a, b, c]
```

Declare a column that holds lists with the `list<T>` type, where `T` is any scalar type
(such as `u32`). List types nest, so `list<list<u32>>` is a column whose values are lists
of lists:

```xlog
pred path(nodes: list<u32>).
```

### Take a list apart with a head/tail pattern

To split a list into its first element and the rest, use the **cons pattern** `[H | T]`
(a head-and-tail pattern). `H` binds to the first element (the head) and `T` binds to
everything after it (the tail):

```xlog
first(H) :- items([H | T]).
```

Everything before the `|` is a fixed prefix of elements; everything after it is the
remaining list. You can bind several leading elements at once, as in `[A, B | Rest]`,
where `A` and `B` are the first two elements and `Rest` is the remainder.

**A minimal program.** Store one list, then read off its head:

```xlog
pred items(xs: list<u32>).
pred first(h: u32).

items([10, 20, 30]).
first(H) :- items([H | T]).
```

**How do I know it worked.** The rule derives one fact — `first(10)` — because `10` is
the head of the stored list.

<Note>
Lists in XLOG are **finite and interned**. *Interning* means each distinct list is stored
once and given a dense integer ID at compile time.

Because a `list<T>` column is really an integer ID under the hood, it joins and compares
as fast as any integer column.

The tradeoff: there are no unbounded or cyclic lists. A list you can write down is a list
the compiler can intern — and nothing else is representable.
</Note>

## Meta-predicates

A **meta-predicate** inspects or manipulates terms and goals rather than plain data.

Only one meta-predicate, `=..` (read "univ"), is built into XLOG's grammar. It relates a
compound term to a list of its functor and arguments — the same role it plays in Prolog.
(A term's *functor* is the name at its head; its arguments are the values it holds.)

Every other meta-predicate is an **ordinary atom recognized by its predicate name** during
compilation. It is not a special piece of syntax — the compiler spots the name and expands
it.

### findall/3 — collect answers into a list

**When to use this.** You want every value that satisfies a goal gathered into a single
list.

**How.** `findall/3` collects every value of a template into a list:

```xlog
collect(Xs) :- findall(Y, edge(1, Y), Xs).
```

This gathers each `Y` for which `edge(1, Y)` holds and binds `Xs` to the resulting list.

<Warning>
`findall/3` is **limited to finite source facts**. Its inner goal must range over stored
facts, not over relations produced by other rules.

A goal that depends on a rule head is rejected. Collecting over derived goals is reserved
for a later aggregate-backed collection path.

When you need to fold data that rules produce, reach for head aggregates like `count` and
`sum` instead.
</Warning>

### maplist and functor/3

`maplist` applies a predicate across the elements of a list.

`functor/3` relates a compound term to its functor name (the name at its head) and its
arity (how many arguments it takes).

Both are recognized by name and rewritten during compilation, in a step called
meta-normalization.

### ground/1, var/1, nonvar/1 — decided at compile time

`ground/1`, `var/1`, and `nonvar/1` look like runtime tests, but XLOG resolves them **at
compile time**, not while your program runs.

At the point where each call appears, the compiler already knows whether the term is bound
or ground. So the call resolves one of two ways: it succeeds and vanishes, or it becomes a
`fail` atom that prunes the rule.

They never execute as runtime checks. They are a compile-time decision about the shape of
your program, not a query against your data.

## No runtime database

XLOG runs your rules repeatedly until no new facts can be derived — a *fixpoint* — and
then stops. It has **no mutable runtime database**, so there is no way to add or remove
facts while a program runs.

The Prolog predicates that mutate the database are therefore rejected:

<Warning>
`call`, `assert`, `asserta`, `assertz`, and `retract` are **not supported** and are
rejected at compile time.

All facts are declared up front, and all derivation is done by rules. If you are porting
Prolog that mutates the database, restructure it as declared facts plus rules.
</Warning>

<Card title="Modules" icon="cubes" href="/language-guide/modules">
  Split a program across files, import selectively, and control visibility with `private`.
</Card>
