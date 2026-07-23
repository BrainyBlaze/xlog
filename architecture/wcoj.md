# Worst-Case Optimal Joins

How XLOG recognizes multiway join shapes, promotes them into RIR, and dispatches specialized CUDA routes with fallback parity.

<Note>
For contributors — how XLOG's worst-case-optimal join routes work internally.
If you just want to turn these speedups on, start with the user guide:
[Factorized Execution](/architecture/factorized-execution).
</Note>

Some queries join three or more relations at once — counting triangles or cycles
in a large graph, for example. Run as an ordinary chain of two-way joins, they
build a huge intermediate table before returning a tiny answer. A **worst-case
optimal join (WCOJ)** computes the whole pattern directly, so peak memory stays
flat instead of blowing up on that intermediate.

This page is about how XLOG implements those joins. XLOG has a family of WCOJ
*routes*: specialized code paths that recognize a particular join shape and run a
dedicated CUDA kernel for it. When no route applies, XLOG falls back to ordinary
execution.

Two properties are worth stating up front:

- A route that declines must return the exact same rows as the ordinary path.
  Declining changes speed, never results.
- A correct answer does not prove a WCOJ route fired. To know which path ran, read
  the executor counters described below.

## Route families

XLOG has a family of WCOJ routes. The right-hand column marks where each is
available: the `0.9.2` release line, or `main` only (built but not yet in a
release beyond `0.9.2`).

A few terms used in the table below: RIR is XLOG's *relational intermediate
representation* — the internal form a rule body is lowered into before execution.
A *witness* is one specific way a recursive tuple can be derived;
"witness-multiplied" means the same result tuple is produced once per derivation,
which is the blow-up factorized deltas avoid. A *helper-split* breaks a clique
route into smaller helper sub-joins, and *variable-order metadata* is the
per-route ordering hints the clique kernel needs.

| Route | Availability | What it does |
| --- | --- | --- |
| Triangle WCOJ | Released in the 0.9.2 line | Dedicated route for recognized triangle bodies over supported key widths. |
| 4-cycle WCOJ | Released in the 0.9.2 line | Dedicated route for recognized 4-cycle bodies. |
| K-clique WCOJ | Released in the 0.9.2 line | Planned clique routes with variable-order metadata and helper-split support. |
| Aggregate-fused WCOJ | On main, unreleased beyond 0.9.2 | Computes selected grouped aggregates without materializing the full join output. |
| Free Join | On main, unreleased beyond 0.9.2 | Generalized GPU multiway route for eligible bodies that do not match a dedicated shape. |
| Factorized recursive deltas | On main, unreleased beyond 0.9.2 | Computes novel recursive tuples without materializing witness-multiplied delta joins. |

## Planning pipeline

A WCOJ route starts in the compiler and finishes in the runtime. The stages are:

1. The lowerer emits ordinary RIR for the rule body.
2. Optimizer passes preserve semantics. They may reorder a recognized shape when
   statistics show a better inner pair.
3. `promote_multiway` converts eligible bodies to `RirNode::MultiWayJoin`.
4. The executor dispatches that multiway node through `wcoj_dispatch`.
5. The CUDA provider runs the matching kernel — dedicated WCOJ, Free Join,
   aggregate-fused, or factorized-delta — if the final runtime gate accepts it.
6. If a gate declines, the executor uses the embedded fallback route instead.

The runtime gate checks several conditions before it accepts a route: body shape,
key width, variable layout, relation availability, stream and runtime support,
memory budget, and kill switches. A *kill switch* is a runtime flag that force-
disables a specific route.

## Diagnostics: dispatch counters

To tell route *eligibility* apart from route *execution*, the executor exposes
counters. A route can be eligible yet still decline at the final gate, so these
counters are how you confirm what actually ran.

The counters are:

- triangle, 4-cycle, and clique WCOJ dispatch counts;
- WCOJ error-decline count;
- aggregate-fused groupby dispatch count;
- Free Join dispatch count (on main);
- factorized recursive-delta dispatch count (on main).

Pair the counters with a kill-switch parity check. Toggle a route's kill switch
and rerun: if only which route fired changed and the result did not, that route is
preserving semantics for that workload.

## Aggregate fusion

Aggregate fusion computes grouped aggregates without first building the joined
table. It is a `main`-only route, unreleased beyond `0.9.2`.

Ordinarily XLOG would materialize every joined tuple and then group them. Instead,
this route reduces by a root variable directly inside the kernel, skipping that
intermediate table.

The route is intentionally narrow. It accepts only the shapes, key widths, and
aggregate operators the CUDA provider implements. Any other aggregate declines: it
either falls back to the materialize-then-groupby path, or returns the same error
that path would have returned.

## Free Join

Free Join is the general multiway route for bodies that do not match a dedicated
triangle or 4-cycle kernel. It is a `main`-only route, unreleased beyond `0.9.2`.

Instead of a shape-specific kernel, it uses a frontier-based GPU algorithm that
handles a broader set of multiway bodies.

The planner can reorder inputs when the prefix-key constraints and statistics point
to a better route. It can also decline in two cases: when a candidate ordering would
lose the factorized benefit, or when the required statistics are missing.

Free Join does not replace the specialized kernels. The dedicated WCOJ shapes stay
dedicated; Free Join is the fallback general route for everything else.

## Factorized recursive deltas

This route speeds up recursive rules shaped like transitive closure. It is
`main`-only, unreleased beyond `0.9.2`.

In such rules, each iteration's *delta step* computes newly derived tuples. The
factorized route computes those novel tuples grouped by root, rather than
materializing every witness (every individual derivation) separately.

At dispatch, it picks one of three implementations:

- a dense-domain bitvector route;
- a sparse-domain hash-set route;
- the legacy hash-join and diff path.

The choice depends on domain size, table budget, arity, key width, and rule shape.
Cases none of the factorized implementations support decline to the existing
recursive evaluation path.

## Scope notes

- WCOJ is not a universal join engine. It is a set of routes for specific
  multiway shapes; everything else runs on the ordinary path.
- Free Join and factorized recursive deltas are `main`-only. They are not part of
  the `0.9.2` release.
- Result equality does not imply an optimized route ran. Use the dispatch counters
  above to confirm.
- A fallback is not a failure. When a route's gate declines cleanly, that is the
  designed behavior, and the fallback returns the same rows.

## See also

- [Factorized Execution](/architecture/factorized-execution) — the user guide.
- [Factorized Execution Internals](/architecture/factorized-execution-internals) —
  `main`-only implementation details.
