# Quickstart

Write, run, and inspect your first XLOG program, then explore the CLI's deterministic, probabilistic, and diagnostic subcommands.

XLOG lets you write a few logic rules — like "if there is an edge from X to Y,
then Y is reachable from X" — and get the answer computed for you. This page
takes you from an empty file to a running query in a couple of minutes.

By the end you will have written a program, run it from the command line, and
confirmed the result by reading the table XLOG prints back.

## Your first program

<Steps>
<Step title="Write the program">

Create a file named `reachability.xlog`. It declares two relations, lists three
edges, and defines what "reachable" means — including the recursive case, where
reaching `Z` means you can already reach some `Y` that has an edge to `Z`. The
last line, starting with `?-`, is the question you want answered.

```prolog
pred edge(u32, u32).
pred reach(u32, u32).

edge(1, 2).
edge(2, 3).
edge(3, 4).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

?- reach(1, N).
```

This query asks: starting from node `1`, which nodes `N` can I reach?

</Step>
<Step title="Run it">

```bash
./target/release/xlog run reachability.xlog
```

</Step>
<Step title="Confirm it worked">

XLOG prints the answer as a table. Starting from node `1` you can reach `2`, `3`,
and `4`, so those are the rows you should see:

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

If you get these three rows, your program ran correctly.

</Step>
</Steps>

<Note>
The [language reference](/reference/language) covers the full surface, and the
repository's `examples/` directory contains annotated programs for lists and
meta-predicates, magic sets, probabilistic aggregates, approximate inference,
epistemic reasoning (`examples/epistemic/`), and Python neural-symbolic
training (`examples/python/`).
</Note>

## CLI at a glance

Once your first program runs, the same `xlog` command gives you other ways to
execute and inspect a program. Each block below is a self-contained example; the
comment says what it is for.

Run a program and compute exact answers:

```bash
# Deterministic execution
./target/release/xlog run program.xlog
./target/release/xlog run program.xlog --output csv
./target/release/xlog run program.xlog --output arrow --output-dir ./results
```

Feed in a table from an external file, using Arrow IPC (Apache Arrow's columnar
format for exchanging tables between tools):

```bash
# External data (Arrow IPC)
./target/release/xlog run program.xlog --input edge=graph.arrow
```

Run a program whose facts have probabilities and get the probability of each
answer. `exact_ddnnf` computes exact probabilities; `mc` estimates them by
Monte Carlo sampling (repeated random trials — faster, approximate):

```bash
# Probabilistic execution
./target/release/xlog prob program.xlog --prob-engine exact_ddnnf
./target/release/xlog prob program.xlog --prob-engine mc --samples 10000 --seed 42
```

Measure where time and memory go:

```bash
# Profiling
./target/release/xlog run program.xlog --stats
./target/release/xlog run program.xlog --stats --stats-format json
```

Ask XLOG to explain how it planned and executed a program:

```bash
# Explain diagnostics
./target/release/xlog explain program.xlog
./target/release/xlog explain --format json program.xlog
```

List every flag for a subcommand:

```bash
./target/release/xlog run --help
```

See the [CLI reference](/reference/cli) for the complete flag reference.

## Next steps

- [Installation](/get-started/installation) — supported platform, source builds, PyPI, crates.io, and the CUDA kernel artifact model
- [Language reference](/reference/language) — types, predicates, rules, modules, UDFs, aggregations, and pragmas
