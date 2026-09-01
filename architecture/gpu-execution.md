# GPU Execution

How XLOG's host executor dispatches CUDA kernels over device-resident relation buffers for deterministic logic evaluation.

<Note>
For contributors — how XLOG's GPU execution works internally.
</Note>

XLOG's deterministic runtime keeps its control logic on the CPU and its data on
the GPU. The CPU-side executor walks the internal query plan (an *RIR plan* — the
runtime's internal representation of a query). For each plan operator it picks an
eligible CUDA kernel and runs it on the GPU. The relations produced along the way
stay in GPU memory as CUDA buffers.

The kernels cover scans, filters, joins, groupby (grouped aggregation),
arithmetic expressions, recursive delta steps (one round of a recursive rule),
and specialized multiway join routes.

<Frame caption="The host executor walks the RIR plan and dispatches one kernel family per operator; relations stay in the device-resident store, which feeds the next operator or fixpoint round (one iteration of the recursive loop, repeated until no new rows appear).">
  <img className="block dark:hidden" src="/assets/diagrams/deterministic-dataflow-light.svg" alt="Deterministic dataflow: an RIR plan tree enters the host executor, which dispatches Scan, Filter, Join, and GroupBy/Distinct kernel families inside the GPU-resident plane; results accumulate in the columnar versioned relation store, which loops back to feed the next operator or fixpoint round." />
  <img className="hidden dark:block" src="/assets/diagrams/deterministic-dataflow-dark.svg" alt="Deterministic dataflow: an RIR plan tree enters the host executor, which dispatches Scan, Filter, Join, and GroupBy/Distinct kernel families inside the GPU-resident plane; results accumulate in the columnar versioned relation store, which loops back to feed the next operator or fixpoint round." />
</Frame>

## Host Control, Device Data

The split is deliberate: the CPU decides *what* to run, the GPU holds the data and
does the work.

`xlog-runtime::Executor` runs on the host and manages:

- relation names, relation IDs, generations, and schemas;
- a `RelationStore` backed by `CudaBuffer` values;
- runtime statistics and join-selectivity observations;
- persistent build-side join indexes;
- dispatch counters for optimized routes (per-route counters that record when a
  specialized kernel actually ran — see [What To Verify](#what-to-verify));
- recursive SCC state for seed, delta, and merge phases. A *SCC* (strongly
  connected component) is a group of rules that depend on each other recursively;
  the runtime evaluates each such group in seed, delta, and merge phases.

`xlog-cuda::CudaKernelProvider` owns the device-facing operations. It loads CUDA
artifacts, allocates tracked slices, launches kernels, and records transfer
telemetry where a path needs a no-host-transfer assertion.

So the executor itself is not GPU-resident. The relation state and the kernel
workspaces are.

## RIR Evaluation

The executor turns each RIR plan operator into GPU work that writes a relation
buffer:

| RIR work | GPU behavior |
| --- | --- |
| Scan | Return the current relation buffer from the store. |
| Filter | Build boolean masks with typed comparison, arithmetic, and boolean kernels, then compact selected rows. |
| Project | Select, reorder, or compute output columns. |
| Join | Dispatch hash join, nested-loop join, WCOJ (worst-case-optimal join — computes multiway patterns directly instead of building large intermediates), Free Join (a multiway join algorithm), or retain the ordinary route depending on shape and runtime gates. |
| Groupby | Run recorded aggregate kernels for supported key/value widths. |
| Rule union | Merge every rule that writes the same head through one chunked multiway concat-and-dedup fold, into a single deduplicated accumulator. Each chunk's byte budget is `XLOG_UNION_CHUNK_BYTES` (default 1 GiB), so peak device memory tracks one chunk plus the accumulator rather than the sum of all contributions. A pass declines fail-closed above 4294967295 rows, or when an output or per-column byte count would overflow a `u32`. |
| Recursive SCC | Execute semi-naive seed and delta variants until convergence or a configured iteration limit. |

"Semi-naive" here means each recursive round processes only the rows newly
derived in the previous round, not the whole relation again.

The correctness rule is *row-set parity*: an optimized path must return exactly
the same rows as the ordinary route. An optimized path is allowed to decline and
leave the ordinary route selected when a shape, width, budget, or gate does not
match. The decline is recorded; it is not an implicit host implementation.

## Predicate And Arithmetic Masks

A filter does not delete rows in place. It lowers into a mask pipeline that marks
which rows survive, then compacts them:

1. Arithmetic expression kernels produce temporary columns.
2. Typed comparison kernels produce boolean masks (one bit per row: keep or drop).
3. Boolean mask kernels combine predicates with `and`, `or`, and `not`.
4. Stream compaction writes the filtered output buffer.

The runtime supports scalar integer, floating, boolean, and symbol comparisons,
matching the types represented in RIR and the CUDA provider kernels. Floating-point
equality follows IEEE 754 (`NaN != NaN`). Ordered comparisons use XLOG's
bit-defined total order: negative NaNs, negative infinity, negative finite values,
`-0.0`, `+0.0`, positive finite values, positive infinity, then positive NaNs.
NaN sign and payload make the order deterministic across host and CUDA paths.

## Joins And Recursion

The ordinary join is always the baseline path. For certain rule shapes the runtime
can route to a faster specialized kernel instead:

- hash joins for normal binary joins;
- nested-loop joins for small eligible products;
- WCOJ for recognized triangle, 4-cycle, and clique shapes;
- Free Join for broader multiway bodies;
- factorized recursive-delta routing for recursive rules (a compressed
  representation of the per-round delta).

The Free Join and factorized recursive-delta routes have been available since
0.10.0.

### Automatic resident recursion

For an eligible ordinary query plan, `xlog-gpu` certifies the complete
dependency-closed plan and automatically prepares one resident conditional graph.
The graph performs recursive control flow on the device with zero host iterations,
zero allocations in the measured core, and zero tracked or metadata transfers in
that core. After terminal synchronization, the runtime reads one bounded pinned
receipt containing status and counts; output relations stay in device buffers.

If certification or preparation declines, automatic mode records the typed reason
and runs the existing ordinary GPU executor. Required mode turns the same decline
into an error, while disabled mode selects the ordinary executor explicitly.

Every optimized route has its own dispatch counter — for example separate counters
for WCOJ, Free Join, fused aggregate, and factorized-delta. These let you tell "the
answer matched" apart from "the optimized route actually fired," since the two are
not the same thing.

## Ingestion And Diagnostics

Two runtime surfaces sit next to the core evaluation loop rather than inside it:
bulk graph loading and delta diagnostics.

- `xlog_gpu::biokg::StreamingGraphRelationLoader` streams JSONL, CSV, and
  N-Triples graph rows into typed edge records, with bounded-memory telemetry.
- `DeltaPlannerTelemetry` reports cache reuse, route-decline decisions, affected SCCs,
  recomputed SCCs, and estimated versus measured delta behavior.
- `pyxlog` exposes planner telemetry through diagnostic result payloads.

Use these surfaces when investigating data loading and incremental behavior. Use
the dispatch counters and transfer telemetry (below) when investigating GPU
execution.

## What To Verify

A matching final result does not prove that an optimized GPU route ran — the
ordinary route returns the same rows. To prove a workload used the intended GPU
path, check:

- the route counter for the optimized dispatch (was it greater than zero?);
- kill-switch parity: disable the optimized route and confirm the ordinary route returns
  the same rows;
- transfer telemetry, for any no-host-transfer claim;
- CUDA-required validation, when the claim depends on actual GPU execution.

Final result equality alone is not enough to prove an optimized GPU route fired.
