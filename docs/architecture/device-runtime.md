# Device Runtime

This is the runtime allocator + stream-dependency stack that
v0.6.0 added under `crates/xlog-cuda/src/device_runtime/`. It
replaces the v0.5 per-`CudaKernelProvider` `GpuMemoryManager`
model with a per-CUDA-ordinal runtime that tracks cross-stream
allocation lifetimes, enforces a process-wide byte budget, and
records per-allocation events that the `LaunchRecorder` (see
[recorded-launch-migration.md](recorded-launch-migration.md))
uses to fence kernels safely on non-default streams.

The runtime is **opt-in**. Default `CudaKernelProvider::new` /
`GpuMemoryManager::new` continues to behave exactly as in
v0.5. Activate the runtime stack with
`CudaKernelProvider::with_runtime` /
`GpuMemoryManager::with_runtime`, and the dispatch surface
described in [Env-gated dispatch](#env-gated-dispatch) below
routes operator calls through the recorded launch paths.

## Stack shape

The recommended composition for production callers (and the
shape the integration tests + cert harness build under
`XLOG_USE_DEVICE_RUNTIME=1`) is:

```
GlobalDeviceBudget
   └── LoggingResource
          └── AsyncCudaResource
                  └── StreamPool (cudarc CudaStream handles)
```

All of these implement the
[`DeviceMemoryResource`](../../crates/xlog-cuda/src/device_runtime/resource.rs)
trait. The trait is the seam: every allocation /
deallocation / `prepare_block_use` / `finish_block_use` /
`reap_pending` call passes through the decorator chain
top-down, the inner backend does the actual work, and each
layer adds its own concern.

### `AsyncCudaResource` (innermost)

`crates/xlog-cuda/src/device_runtime/async_resource.rs`.

Stream-ordered cudarc-backed allocator. Every `allocate`
forwards to `cuMemAllocAsync` on the caller-supplied
`StreamId`. Every `deallocate` queues `cuMemFreeAsync` on the
block's `alloc_stream`.

Per-block dependency state, captured in `LiveEntry`:

  * `last_write: Option<(StreamId, CudaEvent)>` — most recent
    write event on this block, OR the **allocation-ready
    event** captured immediately after `cuMemAllocAsync`.
    Cross-stream consumers wait on this before reading or
    writing the block's bytes.
  * `outstanding_reads: Vec<(StreamId, CudaEvent)>` — read
    events recorded since the current `last_write` was
    installed. Future cross-stream writers and the eventual
    `deallocate` wait on each entry; cleared when a write
    replaces `last_write`.

ABA guard: every entry carries a monotonic `Generation`. The
`(ptr, generation)` tuple is the identity used by
`record_block_use` / `prepare_block_use` / `finish_block_use`
/ `deallocate` — a stale `DeviceBlock` whose address has been
freed and reused returns
`ResourceError::UseAfterFree { generation }` rather than
mutating an unrelated live entry.

### `LoggingResource` (decorator)

`crates/xlog-cuda/src/device_runtime/logging.rs`.

Telemetry-only decorator. Forwards every call to its inner;
on the way through, emits a `LogRecord` to a caller-supplied
`Arc<dyn LoggingSink>`. `prepare_block_use` /
`finish_block_use` are pass-through with no log emission
(those run many times per launch and would balloon the
record stream).

The cert harness uses a `DiscardSink` because it has no
consumer for the log stream during kernel testing. Production
callers can plug in any `LoggingSink` implementation.

### `GlobalDeviceBudget` (outermost)

`crates/xlog-cuda/src/device_runtime/budget.rs`.

Process-wide byte-limit decorator. Enforces a hard limit
across **every** allocation that flows through the runtime.

Allocation flow:

  1. Lock budget state. If `reserved + bytes <= limit`,
     reserve eagerly and forward to the inner. Roll back on
     inner failure.
  2. If the request does not fit AND `bytes <= limit` (i.e.,
     the request CAN fit if pending frees are reaped),
     **drain pending async frees once and retry**. Tight
     allocate-then-drop loops would otherwise exhaust the
     reservation pool while real GPU memory is free —
     `cuMemFreeAsync` only releases real bytes after stream
     completion, so without the retry the budget reservation
     accumulates faster than the inner stream can drain.
  3. If the request is genuinely oversized (`bytes > limit`),
     short-circuit with `OutOfBudget` before touching the
     inner stack so the rejection stays cheap and does not
     emit a reap log record.

Deallocation flow: sample `inner.bytes_outstanding()` before
and after the inner call (under the lock), decrement the
reservation by the observed delta. Synchronous inner →
`bytes_outstanding` drops by `block.bytes`; async inner →
`deallocate` moves bytes from "live" to "pending" so
`bytes_outstanding` is unchanged, and `reap_pending` is what
actually releases the reservation.

`reap_pending` propagates to the inner; the budget's
reservation drops by the reaped total. Cert harnesses call
this between categories (see
`crates/xlog-cuda-tests/tests/certification_suite.rs`).

### `XlogDeviceRuntime` (facade)

`crates/xlog-cuda/src/device_runtime/runtime.rs`.

Thin facade that owns the device handle, stream pool, and the
composed `Box<dyn DeviceMemoryResource + Send + Sync>`. All
runtime API entries (`allocate`, `deallocate`,
`record_block_use`, `prepare_block_use`, `finish_block_use`,
`prepare_first_use`, `finish_first_use`, `bytes_outstanding`,
`reap_pending`, `supports_block_use_tracking`) are simple
forwards through the resource lock.

#### Construction modes

  * **`XlogDeviceRuntime::try_get(ordinal)`** — singleton path
    keyed by CUDA device ordinal. Stored in a per-ordinal
    `OnceLock` table, lock-free on subsequent reads. The
    singleton's installed resource is the cudarc default
    (`DirectCudaResource`), which **does not** support
    cross-stream tracking — the runtime there returns
    `ResourceError::StreamMisuse` on `record_block_use` /
    `prepare_block_use` / `finish_block_use`. Production
    callers that need stream safety go through
    `with_resource` (below) and supply
    `AsyncCudaResource`-backed stacks.
  * **`XlogDeviceRuntime::with_resource(...)`** — composed
    runtime around a caller-supplied resource. **Not** stored
    in the singleton table; intended for tests, examples, and
    integration fixtures that need a specific backend
    (`AsyncCudaResource`) or decorator layering.

#### Helpers added in v0.6.0

  * **`prepare_first_use(slice, use_stream, access)`** — a
    convenience entry for helper-internal scratch
    allocations whose first cross-stream consumer is a raw
    `cuMemsetD8Async` / `cuMemcpyDtoDAsync_v2` /
    `kernel.launch_on_stream` call ahead of any
    `LaunchRecorder::preflight`. Looks up the `BlockId` from
    the slice's runtime block and forwards to
    `prepare_block_use`.
  * **`finish_first_use(slice, use_stream, access)`** — symmetric
    finish entry for the same callers.

If a buffer is not runtime-backed, both helpers return
`ResourceError::StreamMisuse` rather than silently doing
nothing — fail-loud is the policy on this seam.

## `Access` model

Defined in `device_runtime/resource.rs`:

```rust
pub enum Access {
    Read,
    Write,
    ReadWrite,
}
```

Every `prepare_block_use(block, use_stream, access)` /
`finish_block_use(block, use_stream, access)` call carries
the access kind. The async backend uses it to decide which
events to wait on and where the new event lands:

| Access      | prepare waits on            | finish records to                 |
|-------------|-----------------------------|-----------------------------------|
| `Read`      | `last_write` (cross-stream) | `outstanding_reads` (append)      |
| `Write`     | `last_write` + every `outstanding_read` (cross-stream) | replaces `last_write`, clears `outstanding_reads` |
| `ReadWrite` | same wait set as `Write`    | same finish behavior as `Write`   |

Same-stream waits are skipped — events recorded on the same
stream the consumer is launching on are already ordered, and
queueing `cuStreamWaitEvent` against them would be wasted
work. The matrix is what makes a `Read` after a same-stream
`Write` cheap (zero waits queued) while a cross-stream
`Write` still fences correctly behind every prior reader.

`BlockId` is the lifetime-free identity used by the recorder:
`(ptr, generation, alloc_stream, device_ordinal)`. Snapshot
once via `BlockId::from_block(&device_block)`; subsequent
calls don't need to retain the source slice borrow.

## Env-gated dispatch

Every operator surface in `crates/xlog-cuda/src/provider/`
that has a recorded variant gates the dispatch on a
per-operator env var. The umbrella
`XLOG_USE_RECORDED_OPS=1` activates all five at once.

| Env var                          | Operator surface |
|----------------------------------|------------------|
| `XLOG_USE_RECORDED_FILTERS=1`    | `filter`, `filter_columns`, `filter_fused_scan` |
| `XLOG_USE_RECORDED_SORT=1`       | `sort`, `dedup_full_row` |
| `XLOG_USE_RECORDED_DEDUP=1`      | `dedup_full_row` (subset) |
| `XLOG_USE_RECORDED_GROUPBY=1`    | `groupby_agg`, `groupby_multi_agg` |
| `XLOG_USE_RECORDED_HASH_JOIN=1`  | `hash_join_v2`, `hash_join_v2_with_index` (Inner / Semi / Anti / LeftOuter) |
| `XLOG_USE_RECORDED_OPS=1`        | All of the above. |

`env_flag(name)` semantics: empty / unset / `"0"` is false,
any other value is true.

Each per-operator dispatcher checks two things before
routing to the recorded variant:

  1. The env flag is set.
  2. `provider.memory.runtime().is_some()` — i.e., the
     manager was built via `with_runtime`. Without a runtime,
     the dispatcher falls through to the legacy path even if
     the env flag is set, because the recorded variants
     require runtime-backed `DeviceBlock`s to record events
     against.

This decoupling matters for the cert harness: it sets
`XLOG_USE_DEVICE_RUNTIME=1` to flip the test fixture to the
runtime-backed builder, then sets `XLOG_USE_RECORDED_OPS=1`
to drive the dispatcher through the recorded path. Either
flag alone is a no-op for cert testing.

## Supported certification modes

`crates/xlog-cuda-tests/tests/certification_suite.rs` is the
formal cert gate. It runs in two modes:

  * **Legacy** (default): builds the test fixture via
    `GpuMemoryManager::new` + `CudaKernelProvider::new`. Env
    flags are no-ops. Locks the v0.5 surface against
    regression. Pass: 206/206.
  * **Runtime + recorded**:
    `XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1`. The
    fixture builds the production decorator stack
    (`AsyncCudaResource → LoggingResource → GlobalDeviceBudget
    → XlogDeviceRuntime`). The categories that touch
    `provider.sort` / `filter_by_mask` / `hash_join_v2` /
    etc. exercise the recorded path end-to-end. Pass: 206/206.

The cert harness reaps pending async frees between
categories (`ctx.reap_pending()`) so a long sequence of
small allocate-then-drop tests does not pile up pending
frees against the budget. The
`GlobalDeviceBudget::allocate` retry-after-reap path covers
the same case for ad-hoc workloads that don't have a
between-category seam.

## Pre-existing concurrency residual

The v0.6.0 A3/A4 stress harness
(`crates/xlog-integration/tests/test_a3_a4_stress.rs`)
documents a known non-blocking residual: in-process
thread-of-N drift on recursive Datalog workloads (~3% on the
`friends` self-join + `reach` transitive closure shape). The
5-mode diagnostic matrix
(`XLOG_A3_FIXTURE_MODE=per_iter|per_thread|shared` ×
runtime-on/off × recorded-on/off, plus
`XLOG_A3_DIAGNOSTIC=1` to bypass the env-var gate)
demonstrates the drift fires identically on the legacy
default path. The bug class is pre-existing same-process
multi-executor concurrency against one CUDA primary context;
it is not introduced by the v0.6.0 stream runtime or
recorded launches. Tracked under v0.7.0 "Concurrency
Hardening" in `ROADMAP.md`. The v0.6.0 release gate is
**A4 fork-isolated stress + cert suite + umbrella ×50**.
