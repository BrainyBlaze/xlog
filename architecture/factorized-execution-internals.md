# Factorized Execution Internals

Implementation notes for the main-only factorized routes: WCOJ aggregate fusion, Free Join frontiers, and dense or sparse recursive deltas.

<Note>
For contributors — how xlog's factorized execution routes work internally. This
is not a usage guide.
</Note>

*Factorized execution* means computing a query's answer over a compact
representation instead of first building every intermediate row. This page
describes how those routes are shaped inside XLOG.

The routes described here live on the `main` branch. They are unreleased beyond
0.9.2.

## Design Pattern

Every factorized route follows the same six-step pattern.

1. Recognize a rule shape that can avoid a large intermediate.
2. Prove the needed key, width, and variable-layout constraints.
3. Choose a compact representation.
4. Run a CUDA route over that representation.
5. Install the result only after row-set or aggregate parity is preserved.
6. Fall back to the ordinary path (decline) when the contract is not met.

## Aggregate-Fused WCOJ

A *worst-case-optimal join* (WCOJ) computes a multiway join pattern directly,
without building the large intermediate table that a chain of binary joins would
produce. The aggregate-fused variant handles grouped-aggregate shapes: it reduces
values through CUDA kernels while grouping by one planned *root variable* (the
grouping key the aggregate reduces by).

The unfused path does three separate steps:

1. materialize all joined tuples;
2. group by the root key;
3. reduce values.

The fused path instead accumulates the aggregate during the WCOJ traversal, so
the joined tuples are never materialized. Before dispatch, the runtime still
checks the rule shape, the key width, the aggregate operator, and the group-key
position.

## Free Join Frontiers

*Free Join* is a multiway-join method that binds one variable at a time. XLOG
represents it as a *frontier* — a moving set of partial bindings — advancing over
sorted *range tries* (prefix indexes whose nodes cover contiguous ranges of a
column).

- Relation columns are laid out so prefix probes can advance level by level.
- Each frontier level binds another variable, or proves that a branch has no
  compatible tuples.
- Identity groups and probe filters avoid unnecessary expansion.
- For variables private to one relation, *count-by-root* multiplies the remaining
  trie-range lengths instead of enumerating each full binding.

The runtime order planner picks a prefix-key-compatible variable order only when
that order is expected to preserve the factorized benefit. Otherwise the route
declines to the ordinary path.

## Recursive Delta Factorization

Recursive rules are evaluated by a *semi-naive* fixpoint loop: each round joins
only the newly derived tuples (the *delta*) rather than the whole relation.
Factorized recursive deltas target rules such as transitive closure — reachability
over all chains of edges — where a delta join can produce many *witnesses* (many
distinct derivations) for the same novel tuple.

<Frame caption="Inside the semi-naive loop, the delta join routes dense, sparse, or legacy based on domain size and byte budget — every route produces the same novel set.">
  <img className="block dark:hidden" src="/assets/diagrams/factorized-recursive-delta-light.svg" alt="Factorized recursive delta routing: the delta join passes through a router checking domain and budget, dispatching to a dense GPU bitvector route, a sparse GPU hash-set route, or the legacy join-plus-diff route; all three converge on the same novel set, which feeds the next semi-naive round." />
  <img className="hidden dark:block" src="/assets/diagrams/factorized-recursive-delta-dark.svg" alt="Factorized recursive delta routing: the delta join passes through a router checking domain and budget, dispatching to a dense GPU bitvector route, a sparse GPU hash-set route, or the legacy join-plus-diff route; all three converge on the same novel set, which feeds the next semi-naive round." />
</Frame>

The dispatcher chooses among three routes.

| Route | Representation | Decline reason |
| --- | --- | --- |
| Dense | Root-indexed bitvectors over a bounded domain. | Domain too large or unsupported shape. |
| Sparse | GPU open-addressing hash set over distinct candidates. | Table estimate or byte budget exceeds limits. |
| Legacy | Hash join followed by diff. | Fallback for unsupported or declined cases. |

Both factorized routes must produce the same novel set as the legacy recursive
path.

## Loss Vetoes

Factorized execution is not always cheaper than the ordinary path. The cost model
can *veto* a route in two cases:

- available statistics show that the ordinary binary path should be cheaper; or
- no prefix-key-compatible Free Join order is viable.

When statistics are missing, the route does not assume the factorized path wins.
Instead it takes a safe default, keeps the existing order, or declines.

## Device And Host Boundaries

Control stays host-side. The executor chooses routes, owns the counters, and
installs the resulting relation buffers. CUDA kernels own the data-plane work:

- frontier expansion,
- grouped accumulation,
- dense bitvector novel-set computation,
- sparse hash-set novel-set computation.

If a route claims that its hot loop moves no tracked host data, transfer telemetry
is the signal that confirms it.

## Verification Obligations

Because a correct answer alone does not prove that a factorized CUDA path fired, a
route is not considered complete until evidence shows all of the following.

- Fallback parity: same result with the route disabled.
- The route's dispatch counter increments with the route enabled.
- Unsupported shapes decline cleanly.
- Budget and overflow conditions fail closed.
- Recursive routes converge to the same full relation as the legacy path.
- Aggregate routes match the materialize-plus-groupby result.

These obligations are route-specific. A generic docs or unit-test gate passing
does not prove that a factorized CUDA path fired.
