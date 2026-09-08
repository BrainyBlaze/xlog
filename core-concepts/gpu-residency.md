# GPU residency

Why XLOG keeps reasoning state on the device, what the residency contract guarantees, and how it fails when a program will not fit.

Many neural-symbolic systems run neural computation on the GPU and symbolic reasoning
on the CPU. Each training iteration can then pay a PCIe round-trip across that
boundary. XLOG instead runs relational, probabilistic, and solver kernels over
device-backed state, while keeping host orchestration and each route's explicitly
accounted observations visible.

<Frame caption="The host launches kernels and reads only bounded metadata; relations, deltas, circuit values, and solver state stay in the device memory budget, and results reach PyTorch, JAX, and cuDF zero-copy through DLPack and Arrow.">
  <img className="block dark:hidden" src="/assets/diagrams/gpu-residency-model-light.svg" alt="XLOG GPU residency model: the host executor and compiler launch kernels into a GPU-resident device plane holding relations, deltas, circuit values, and solver state within a memory budget, exposed zero-copy to PyTorch, JAX, and cuDF via DLPack and Arrow." />
  <img className="hidden dark:block" src="/assets/diagrams/gpu-residency-model-dark.svg" alt="XLOG GPU residency model: the host executor and compiler launch kernels into a GPU-resident device plane holding relations, deltas, circuit values, and solver state within a memory budget, exposed zero-copy to PyTorch, JAX, and cuDF via DLPack and Arrow." />
</Frame>

## The contract

Relations, deltas, probabilistic circuit values, and solver workspaces are represented
in GPU memory. The compiler and executor remain on the host: they launch kernels,
synchronize streams, and may read bounded control metadata. A zero-transfer claim is
therefore a contract of a named execution region, not a property inferred from the
words "GPU-resident."

<Note>
The certified resident conditional-graph core is the strongest deterministic
contract. It permits no tracked data transfer, provider D2H call, or untracked
metadata read while the graph runs. After one terminal synchronization, it copies one
one final pinned receipt containing only status, counts, trace counters, and schema
selections. Relation columns remain device-backed. Other GPU routes are
host-orchestrated and can have different, separately documented observation
boundaries.
</Note>

Query results and gradient tensors can be exposed as GPU-backed DLPack capsules and
Arrow arrays, so PyTorch or JAX can consume the returned buffer without copying it.
`tensor.device` confirms the final buffer location; it does **not** prove the
buffer's transfer history. Use route telemetry and the strict gate below for that
stronger claim.

### Enforce it, don't just observe it

Checking `tensor.device` tells you where one result ended up. If you want the runtime to
*enforce* the contract instead, a Python session can turn on a strict gate:

```python
session.set_strict_deterministic_d2h(True)
```

While the gate is on, any attempt to read relation data back to the host fails instead of
copying — including the small, ordinarily-permitted deterministic reads, such as the
one-byte-per-fact membership mask used to validate evidence. Bounded metadata reads stay
exempt: the scalar counts that drive control flow never trip the gate, because the
guarantee is about semantic data.

Two calls let you inspect the gate: `session.strict_deterministic_d2h_enabled()` reports
whether it is on, and `session.deterministic_d2h_violation_count()` reports how many reads
it has rejected — `session.reset_deterministic_d2h_violations()` clears that count. A
rejection is atomic: the operation leaves the relation and its evidence unchanged. See the
[Python reference](/reference/python) for the full API and
[diagnostics](/guides/diagnostics) for a worked zero-transfer audit.

## Automatic resident selection

Eligible ordinary query plans use the certified resident conditional graph
automatically. If plan certification or preparation declines, automatic mode records
the typed reason in execution statistics and runs the existing ordinary GPU route.
This is a route decision inside the CUDA engine, not a switch to CPU execution.

`XLOG_DISABLE_RESIDENT_RECURSION=1` selects the ordinary GPU route explicitly for
answer and timing comparisons. `XLOG_REQUIRE_RESIDENT_RECURSION=1` turns any resident
decline into an error. Enabling both is a configuration error.

## Bounded memory, capacity, and convergence

Every provider enforces its device memory budget. You can raise the configured limit
per program (`memory_mb` in the API, `--memory-mb` on the CLI).

If a byte-addressed allocation cannot fit, XLOG does not silently spill or switch to
the host. It fails closed with a `ResourceExhausted` error. Its fixed-order context reports the rejecting
layer, current reservation, requested bytes, exact required bytes, configured budget,
and the manager's prior peak. Required bytes are calculated without saturation; if the
exact value exceeds `u64`, `estimated_bytes` is `u64::MAX` and the context carries an
explicit overflow marker. This makes an out-of-memory condition definite and
diagnosable rather than a slow degradation. Count and representational limits use
`CapacityExceeded` with a unit such as rows, slots, or candidates. Reaching an
iteration bound without convergence uses `ConvergenceFailure`; neither condition is
reported as byte-memory exhaustion.

Concurrent requests are reserved against the local budget before allocator admission,
so in-flight requests still prevent oversubscription. The manager publishes current and
peak bytes only after admission succeeds: a request refused by CUDA or the device-runtime
budget is removed from the local guard and never appears in the admitted current value or
the reservation-lifetime high-water mark.

```text
Error: ResourceExhausted { context: "GPU memory pressure: layer=... current_bytes=... requested_bytes=... required_bytes=... required_u64_overflow=false budget_bytes=... prior_peak_bytes=...", estimated_bytes: ..., budget_bytes: ... }
```

Fail-closed behavior is a running theme in XLOG. Where a computation cannot be done on
the device within its declared bounds, the engine rejects it with a typed error instead
of quietly switching to a slower or less exact path. You always know which path
produced a result.

## Compile once, keep the structure resident

Because the compiled plan is stable across evaluations, XLOG keeps compiled artifacts
resident and reuses them. For probabilistic inference this includes the compiled
arithmetic circuit: training iterations update leaf weights and evidence in place
without recompiling the circuit structure. For deterministic queries, build-side hash
indexes for hot relations are cached and reused across evaluations in a session.

<Card title="See it in the pipeline" icon="diagram-project" href="/core-concepts/how-xlog-works">
  The compilation pipeline shows exactly where the host/device boundary sits: the
  executor orchestrates on the host, while kernels and the relation store are resident
  on the GPU.
</Card>
