# GPU-Resident Execution + Production CLI Design

**Date:** 2026-01-15
**Status:** Approved
**Scope:** GPU execution path (filters, groupby finalization, arithmetic) + production `xlog` CLI

---

## Goals

1. Eliminate CPU round-trips in the deterministic execution path for filters, groupby boundary handling, and arithmetic evaluation.
2. Ship a production-grade `xlog` CLI that runs deterministic and probabilistic programs, supports Arrow IPC inputs/outputs, and defaults to human-friendly output.
3. Preserve deterministic behavior and explicit error reporting across all modes.

## Non-Goals

- Add a REPL in this phase.
- Implement new semantics for xlog-elp or new solver functionality.
- Introduce JIT or runtime kernel compilation.

---

## Architecture Overview

### A) GPU Predicate Engine (Deterministic Runtime)

We replace the CPU-based predicate evaluation in `Executor::execute_filter` with a GPU mask pipeline that supports the full `Expr` tree:

1. **Arithmetic evaluation on GPU**
   - Use existing arithmetic ops in `CudaKernelProvider` for `+ - * / % abs min max pow cast` with the same semantics as current tests.
   - Materialize intermediate columns as GPU buffers when needed by complex predicates.
2. **Typed comparison kernels**
   - Extend filter comparison coverage to all scalar types (`u32/u64/i32/i64/f32/f64/bool/symbol`), using existing kernels where present and adding missing kernels where needed.
3. **Mask composition**
   - Use `mask_and`, `mask_or`, `mask_not` kernels to build a final selection mask for arbitrary boolean expressions.
4. **Compaction**
   - Use the multi-block scan + compact kernels to produce the filtered output buffer on-device.

**Design choice:** mask-DAG evaluation over JIT or bytecode. It is deterministic, uses existing PTX modules, and scales across complex predicates without runtime compilation complexity.

### B) GPU GroupBy Finalization (No Host Round-Trips)

The current groupby path downloads boundary masks and computes group IDs/keys on CPU. We remove all host round-trips by:

1. Computing boundary masks on GPU (existing `detect_group_boundaries`).
2. Producing group IDs with GPU prefix sum over boundary masks.
3. Computing group start indices on GPU (prefix-sum results give group offsets).
4. Gathering group key columns on GPU using packed-row gather or a dedicated gather kernel.

Result: groupby results (keys + aggregated values) remain entirely GPU-resident until final output.

### C) GPU Arithmetic Path

Arithmetic used in filters and projections must stay on GPU. All arithmetic helpers in `CudaKernelProvider` will be GPU-backed and used to build intermediate columns for predicate evaluation and projection expressions.

### D) Production `xlog` CLI

A new crate `crates/xlog-cli` will provide the `xlog` binary with two subcommands:

- `xlog run` (deterministic)
  - Compiles and executes `.xlog` using `xlog-logic` + `xlog-runtime`.
- `xlog prob` (probabilistic)
  - Supports `--prob-engine=exact_ddnnf|mc` with MC controls.

**Inputs:**
- `.xlog` facts (always supported).
- Arrow IPC files for EDB relations: `--input rel=path.arrow` (repeatable).

**Outputs:**
- Default: pretty table to stdout.
- `--output=csv` prints CSV to stdout.
- `--output=arrow` writes Arrow IPC to file (per query or to a target path).

---

## Component Changes

### xlog-runtime
- Replace `Executor::execute_filter` CPU path with a GPU-backed predicate evaluator.
- Ensure filter evaluation covers all `Expr` forms using GPU operators and mask composition.

### xlog-cuda
- Extend filter comparison kernels to missing scalar types and add host-side orchestration for full predicate evaluation.
- Add GPU groupby finalization path (group IDs and key extraction on-device).
- Ensure arithmetic helpers do not fall back to host for supported types.

### xlog-cli
- New crate and binary with Clap-based CLI.
- Implements input parsing for `.xlog` and Arrow IPC inputs.
- Uses `CudaKernelProvider` Arrow IPC helpers for I/O.
- Routes execution to deterministic or probabilistic engine based on subcommand and flags.

---

## Data Flow

1. CLI reads `.xlog` source and optional Arrow IPC inputs into GPU buffers.
2. For deterministic runs, compile via `xlog-logic`, execute via `xlog-runtime` on GPU.
3. For probabilistic runs, use `xlog-prob` exact or MC engine; MC exposes sampling controls and confidence intervals.
4. Results are rendered to stdout or written as Arrow IPC files.

---

## Error Handling

- All GPU failures surface as `XlogError::Kernel` or `XlogError::Execution` with explicit context.
- CLI validates input relations vs. inferred schemas and reports mismatches with actionable messages.
- MC engine errors on invalid confidence, zero samples, or unsatisfied evidence with deterministic messages.

---

## Testing and Verification

- GPU filter correctness tests for complex predicates (nested AND/OR/NOT + arithmetic + comparisons).
- Groupby correctness tests verifying key extraction and aggregates without host round-trips.
- CLI integration tests for:
  - `.xlog` only runs
  - Arrow IPC input round-trip
  - `--output=pretty|csv|arrow`
  - Probabilistic exact and MC paths
- Full suite: `cargo test --workspace --all-targets --exclude pyxlog --release` and CUDA certification.

---

## Rollout

1. Land GPU execution changes behind existing APIs (no user-facing breakage).
2. Add `xlog` CLI crate and wire into workspace.
3. Update `docs/ROADMAP.md` and `docs/ARCHITECTURE.md` to reflect GPU-resident filter/groupby and CLI availability.

---

## Success Criteria

- No CPU round-trips in filter/groupby paths during deterministic execution.
- CLI can run deterministic and probabilistic programs with Arrow IPC inputs and outputs.
- All tests and CUDA certification pass in release mode.
