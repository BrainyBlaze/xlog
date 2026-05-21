# CLI Reference

This document describes the `xlog` command-line interface for running deterministic and probabilistic Datalog programs.

## Overview

The `xlog` CLI is implemented in the `xlog-cli` crate. The current workspace
exposes two execution subcommands:

- `xlog run` — Deterministic program execution
- `xlog prob` — Probabilistic program execution

The v0.8.5 language contract adds developer-experience commands:

- `xlog explain` — Inspect parse, strata, RIR, optimizer, magic-set, WCOJ,
  probabilistic plans, rule provenance, and proof traces
- `xlog repl` — Interactive multiline source/query session
- `xlog watch` — Debounced file-change rerun with typed diagnostics

Published artifacts follow tagged releases and may lag the current workspace
surface.

## Installation

Install the latest published CLI crate:

```bash
cargo install xlog-cli --features host-io
```

This path requires Rust, Cargo, CUDA Toolkit 13.x, and `nvcc` at install time.
The installed binary embeds portable PTX for runtime kernels, so it does not
require a sidecar `kernels/` directory after `cargo install` completes. If
`XLOG_CUBIN_DIR` or a binary-adjacent `kernels/` directory is present, xlog
prefers those staged artifacts before falling back to embedded PTX.

The CLI is built as part of the workspace:

```bash
cargo build --release -p xlog-cli
# Binary at: target/release/xlog

# For host-readable probabilistic output (`xlog prob`)
cargo build --release -p xlog-cli --features host-io
```

`xlog run` works with the default build. `xlog prob`'s host-readable output path
requires the `host-io` feature.

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
- `--stats` — Emit execution statistics to stderr
- `--stats-format <FORMAT>` — Statistics format: `human` (default) or `json`
- `--module-path <DIR[:DIR...]>` — Additional module search paths

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
```

### xlog prob

Execute a probabilistic Datalog program.

```bash
xlog prob [OPTIONS] <SOURCE>
```

**Arguments:**
- `<SOURCE>` — Path to the `.xlog` source file with probabilistic facts

**Options:**
- `--prob-engine <ENGINE>` — Inference engine: `exact_ddnnf` (default), `mc`
- `--samples <N>` — Monte Carlo sample count (with `--prob-engine mc`)
- `--seed <N>` — Random seed for Monte Carlo (with `--prob-engine mc`)
- `--confidence <LEVEL>` — Confidence level for MC intervals (default: 0.95)
- `--output <FORMAT>` — Output format: `pretty` (default), `csv`, `arrow`
- `--output-dir <DIR>` — Directory for Arrow output files (with `--output arrow`)
- `--device <N>` — CUDA device index (default: 0)
- `--memory-mb <MB>` — GPU memory limit in megabytes
- `--module-path <DIR[:DIR...]>` — Additional module search paths

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

### xlog explain

Inspect compilation and diagnostic state for a deterministic source file.

```bash
xlog explain [OPTIONS] <SOURCE>
```

**Arguments:**
- `<SOURCE>` — Path to the `.xlog` source file

**Options:**
- `--format <FORMAT>` — Output format: `text` (default), `json`, or `dot`

Text output prints compact sections for parse stats, magic-set rewrites,
aggregate lifting, rule provenance, proof traces, stratification, RIR, and
optimizer status. JSON output includes full `rule_provenance`, `proof_traces`,
and `generated_rule_diagnostics` arrays. The rule provenance records contain
`rule_id`, `head`, `source_kind`, `source_span`, `generation_trace_hash`,
`support_relation_ids`, and `counterexample_relation_ids`. Proof trace records
contain `query_id`, `query`, `answer_relation`, `rule_ids`, `source_facts`, and
`rejected_alternatives`. Generated-rule diagnostics contain `row_decisions`
with `row_key`, `accepted`, `failed_predicates`, `threshold_comparisons`, and
`aggregate_inputs` so accepted and rejected generated rows can be audited from
the CLI JSON report.

When a generated-rule program uses an external candidate input relation instead
of inline facts, `xlog explain --format json` also looks for a colocated
execution manifest (`xlog_hypothesis_execution.json`) and relation JSON file.
The manifest's `relation_input_columns` and `relation_input_path` let the CLI
bind external rows to rule variables and compute the same row-level threshold
decisions in `generated_rule_diagnostics`.

`--format dot` prints the magic-set dependency graph.

**Examples:**

```bash
xlog explain program.xlog
xlog explain --format json program.xlog
xlog explain --format dot program.xlog
```

For the shared v0.8.7 diagnostics model, see
[`living-world-diagnostics-v087.md`](living-world-diagnostics-v087.md).

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
# Creates: ./results/query_0.arrow
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
