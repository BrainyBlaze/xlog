# Solver Services

The GPU CNF and CDCL substrate used by XLOG's probabilistic and epistemic verification paths.

<Note>
For contributors — how XLOG's GPU SAT substrate works internally. This page is dense on purpose; it is not an end-user guide.
</Note>

XLOG's solver services provide a GPU-backed SAT engine that other subsystems call
when they need a yes/no proof that some Boolean formula can (SAT) or cannot
(UNSAT) be satisfied. Concretely, it offers a CUDA-resident representation of a
formula in **conjunctive normal form** (CNF — a Boolean formula written as an AND
of OR-clauses) plus a **CDCL** solver workspace. CDCL (conflict-driven clause
learning) is the standard modern SAT-solving algorithm.

The solver is a *substrate*, not a policy. It answers SAT/UNSAT questions.
Higher-level paths — probabilistic compilation, epistemic planning, and future
verification paths — decide when and how to ask.

## CNF Representation

A CNF formula lives on the GPU as an `xlog-solve::GpuCnf`. It stores the formula
in **CSR form** (compressed sparse row — a flat-array layout where one offset
array marks where each clause begins in a shared literal array). The layout
follows the standard DIMACS CNF text format.

The fields split into host-side capacities and device-resident data:

- `var_cap`, `clause_cap`, and `lit_cap` are host-known allocation capacities
  (the maximum variables, clauses, and literals the buffers can hold);
- `num_vars`, `num_clauses`, and `num_lits` are device-resident scalar buffers
  (the actual counts, living on the GPU);
- `clause_offsets` and `literals` are the device-resident CSR buffers themselves.

Every CNF buffer is owned by a specific CUDA provider (the object that manages a
GPU context). A provider-memory check rejects CNF buffers owned by a *different*
provider, so a formula cannot be silently mixed across GPU contexts.

`GpuCnf::from_host` builds a CNF from host memory; it exists for tests and
tooling. Production GPU-native paths can construct the CNF directly on device and
pass the same solver-facing structure forward, with no host round-trip.

## CDCL Workspace

A `GpuCdclSolver` owns a CUDA provider plus a `GpuCdclConfig`. The config controls
how large the solver's scratch space is and how it paces itself:

- learned-clause arena capacity;
- learned-literal capacity;
- proof-trace capacity;
- deterministic restart and reduction intervals;
- an optional conflict budget.

A `GpuCdclWorkspace` pre-allocates every device buffer a solve needs, so repeated
solves reuse the same memory. Those buffers are: assignments, levels, reasons,
variable activity, trail, watch lists, learned clauses, proof data, and scalar
status outputs. The workspace does *not* own the input CNF buffers — the CNF is
supplied per solve.

## Status And Validation

The solver exposes two levels of API: a raw level for debugging, and
expected-status helpers that check the answer for you.

- **Raw solve APIs** return device-resident assignment, status, and error buffers.
  Use these for debugging and research, where you want to inspect what the solver
  produced.
- **`solve_expect_sat`** asserts the instance is SAT (satisfiable). It validates
  the result and returns a device-resident satisfying assignment.
- **`solve_expect_unsat`** asserts the instance is UNSAT (unsatisfiable). It
  returns `Ok(())` only when that expected result is actually observed.

The expected-status helpers read scalar status and error values as *control-plane*
checks — small signals about how the solve ended. That is different from moving
the CNF, proof state, or circuit data through the host as a *data plane*; the
bulk data stays on the device.

## Fail-Closed Budgets

Hard instances can run indefinitely, so the CDCL config can cap the work with a
conflict budget. This budget and its typed `VerifyBudgetExceeded` error have been
available since 0.10.0. The same release added the compile-capacity
`CompileCapacityExceeded` error for oversized D4 compilations.

The bound is the config field `max_conflicts`. A nonzero value limits how much
work a hard verification instance may do. If the kernel hits the budget before it
establishes SAT or UNSAT, it reports a budget-exhausted status. The Rust API
surfaces that as a typed `VerifyBudgetExceeded` error.

A budget-exhausted result is *indeterminate*: the solver ran out of budget before
deciding. It is never a proof of either SAT or UNSAT.

## Joint Constraint Carrier

Available since 0.11.0. Alongside the CNF/CDCL substrate, `xlog-cuda` carries a
second device-resident solver for *joint label assignment*: picking one label per
entity subject to pairwise constraints between candidate pairs. It is exposed to
Python as `pyxlog.JointConstraintCarrier`.

A carrier is constructed with `(device, entities, domain_lanes, candidates,
labels, fuel_limit)`. `register_schema(catalog_sha, solver_identity)` pins it to a
schema and to the ABI string `pyxlog.SOLVER_ABI_IDENTITY`, once per session;
`bind_signatures(head_masks, tail_masks)` installs the constraint masks, also once.
Two solve stages then run on device in order — `solve_label_feasibility(abstain_label)`
narrows each entity's feasible label set, then `solve_label_map_top2()` computes the
per-candidate top-two scores — followed by `solve_components_exact(comp_offsets,
comp_indices)`, which solves each listed multi-candidate component by complete
enumeration.

`export_buffer(name)` hands one named result buffer back as a zero-copy DLPack
capsule (`domains`, `scores`, `constraints`, `outputs`, `feasible_sets`,
`logical_counts`, `map_results`, or `solve_status`). Because the buffer stays on
the device and xlog stays its owner, ordering is explicit: `note_producer_stream`
records an external CUDA stream whose writes the next solve must wait on, and
`note_consumer_stream` records a stream the carrier signals after a stage
completes. Without one of those — or a host synchronize — writes made through an
exported view race the launch.

Refusals are typed and fail-closed, matching the rest of the solver contract.
`pyxlog.CarrierRefused` (a `RuntimeError` subclass) covers a schema or ABI
mismatch, a re-registration, a zero capacity dimension, binding or solving before
`register_schema`, rebinding already-bound masks, a mask shape mismatch, solving
before signatures are bound, running the top-two stage before feasibility solved,
a malformed component plan, an out-of-range abstain label, and an unavailable or
unloadable kernel. `pyxlog.SolverResourceExhausted` is raised only when a solve
would exceed `fuel_limit`; the overflowing charge is never applied, so the
session's `fuel_spent` is unchanged and an identical retry refuses identically.

Two limits are worth stating plainly:

- `solve_components_exact` enumerates. A component past the stage's capacity is
  not solved — it is marked refused in `solve_status` (row value `3`), leaving
  those rows on top-two authority rather than exact. The exact memoized
  dynamic-programming stage for chain-order components
  (`xlog_cuda::JointConstraintCarrier::solve_components_memoized`, backed by the
  `joint_label_memoized` kernel) exists in the Rust carrier only; it is not
  reachable from `pyxlog.JointConstraintCarrier`.
- `map_results` is the global max-marginal label only for single-candidate
  components. For a multi-candidate component the top-two stage's answer is not
  authoritative until `solve_components_exact` has covered it.

## Consumers

Several higher-level engines call the solver services:

| Consumer | Use |
| --- | --- |
| `xlog-prob` | Verification around probabilistic circuit and knowledge-compilation paths. |
| `xlog-gpu` | Epistemic GPU planning and split-execution certification surfaces. |
| `xlog-cli` | Diagnostic plan dumps and release validation flows. |
| `pyxlog` | Direct Python access to the joint-constraint carrier: constructs it, runs the feasibility, top-two, and component solves, and imports the exported device buffers through DLPack. |

Responsibility splits cleanly. The consumer owns the *semantic contract* — what a
SAT or UNSAT answer means for its problem. The solver service owns the mechanism:
the CUDA CNF, the CDCL workspace, the expected-status checks, and the fail-closed
resource behavior.

## Scope Of The Solver Contract

The solver makes a deliberately narrow guarantee. It is *not* the source of the
stricter "no host transfers" contract — that stronger promise belongs only to the
specific integrations that track and expose it, and should not be attributed to
every solver-backed path.

What the solver service itself guarantees:

- CNF and solver workspaces are device-resident;
- status handling is explicit and typed;
- capacity or budget failures decline fail-closed (they stop safely rather than
  return a wrong answer);
- any SAT or UNSAT claim comes from an expected-status path or a caller-specific
  verifier contract — never from a bare raw solve.
