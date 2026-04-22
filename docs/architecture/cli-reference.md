# CLI Reference

This document describes the `xlog` command-line interface for running deterministic and probabilistic Datalog programs.

## Overview

The `xlog` CLI is implemented in the `xlog-cli` crate and provides two main subcommands:

- `xlog run` — Deterministic program execution
- `xlog prob` — Probabilistic program execution

## Installation

The CLI is built as part of the workspace:

```bash
cargo build --release -p xlog-cli
# Binary at: target/release/xlog
```

## Commands

### xlog run

Execute a deterministic Datalog program.

```bash
xlog run [OPTIONS] <FILE>
```

**Arguments:**
- `<FILE>` — Path to the `.xlog` source file

**Options:**
- `--input <REL>=<PATH>` — Load Arrow IPC file as EDB relation (repeatable)
- `--output <FORMAT>` — Output format: `pretty` (default), `csv`, `arrow`
- `--output-dir <DIR>` — Directory for Arrow output files (with `--output arrow`)
- `--device <N>` — CUDA device index (default: 0)
- `--memory-mb <MB>` — GPU memory limit in megabytes
- `--stats` — Emit execution statistics (per-stratum timing, memory usage, symbol-table size)
- `--stats-format <FORMAT>` — Stats output format: `human` (default), `json`
- `--module-path <DIRS>` — Colon-separated directories to search for imported modules (repeatable)

**Examples:**

```bash
# Basic execution
xlog run examples/xlog/00-basics/01_tc_reachability.xlog

# With external data
xlog run --input edge=graph.arrow program.xlog

# CSV output
xlog run --output csv program.xlog

# Arrow IPC output
xlog run --output arrow --output-dir ./results program.xlog

# Specify GPU device and memory
xlog run --device 1 --memory-mb 2048 program.xlog

# Profiling output
xlog run --stats program.xlog
xlog run --stats --stats-format json program.xlog

# Module imports (for programs with `use`)
xlog run --module-path ./lib:./vendor program.xlog
```

### xlog prob

Execute a probabilistic Datalog program.

```bash
xlog prob [OPTIONS] <FILE>
```

**Arguments:**
- `<FILE>` — Path to the `.xlog` source file with probabilistic facts

**Options:**
- `--prob-engine <ENGINE>` — Inference engine: `exact_ddnnf` (default), `mc`
- `--samples <N>` — Monte Carlo sample count (with `--prob-engine mc`, default: 10000)
- `--seed <N>` — Random seed for Monte Carlo (default: 0)
- `--confidence <LEVEL>` — Confidence level for MC intervals (default: 0.95)
- `--output <FORMAT>` — Output format: `pretty` (default), `csv`, `arrow`
- `--output-dir <DIR>` — Directory for Arrow output files (with `--output arrow`)
- `--device <N>` — CUDA device index (default: 0)
- `--memory-mb <MB>` — GPU memory limit in megabytes
- `--module-path <DIRS>` — Colon-separated directories to search for imported modules (repeatable)

**Examples:**

```bash
# Exact inference
xlog prob examples/prob/01-wet-conditioning.xlog --prob-engine exact_ddnnf

# Monte Carlo inference
xlog prob program.xlog --prob-engine mc --samples 10000

# MC with reproducible seed
xlog prob program.xlog --prob-engine mc --samples 10000 --seed 42

# Custom confidence interval
xlog prob program.xlog --prob-engine mc --samples 10000 --confidence 0.99
```

## Input Formats

### .xlog Source Files

Datalog source with optional probabilistic annotations:

```prolog
% Type declarations
pred edge(u32, u32).
pred reach(u32, u32).

% Facts
edge(1, 2).
edge(2, 3).

% Probabilistic facts (for xlog prob)
0.3::rain.
0.7::sprinkler.

% Rules
reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

% Queries
?- reach(1, N).
```

### Arrow IPC Files

EDB relations can be loaded from Arrow IPC files:

```bash
xlog run --input edge=edges.arrow program.xlog
```

The Arrow file schema must match the predicate declaration:
- Column count must match predicate arity
- Column types must be compatible with declared types

## Output Formats

### Pretty (Default)

Human-readable table format:

```
reach(1, N):
| N |
|---|
| 2 |
| 3 |
| 4 |
| 5 |
```

### CSV

Comma-separated values:

```
N
2
3
4
5
```

### Arrow

Arrow IPC files written to the output directory:

```bash
xlog run --output arrow --output-dir ./results program.xlog
# Creates: ./results/reach.arrow
```

## Probabilistic Output

### Exact Inference

Reports exact probabilities:

```
Query: wet
P(wet | evidence) = 0.3
```

### Monte Carlo Inference

Reports estimates with uncertainty:

```
Query: wet
P(wet) = 0.301 ± 0.009 (95% CI: [0.283, 0.319])
Samples: 10000, Seed: 42
```

## Error Handling

The CLI provides actionable error messages:

| Error | Message |
|-------|---------|
| Parse error | Syntax error with line/column |
| Type mismatch | Expected vs. actual types |
| Schema mismatch | Arrow file schema doesn't match predicate |
| OOM | Memory budget exceeded with estimates |
| CUDA error | Kernel failure with context |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_VISIBLE_DEVICES` | Control visible GPU devices |
| `XLOG_LOG_LEVEL` | Logging verbosity (debug, info, warn, error) |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Parse/compilation error |
| 2 | Execution error |
| 3 | I/O error (file not found, permission denied) |
| 4 | Resource exhausted (OOM) |

## See Also

- [GPU Execution](gpu-execution.md) — How programs are executed
- [Probabilistic Tier](xlog-prob.md) — Exact and Monte Carlo inference
- [Data Interoperability](cudf-interop.md) — Arrow IPC format details
