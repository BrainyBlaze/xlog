# Recorded Launch Migration Guide

A checklist for operator authors moving a kernel chain onto
the v0.6.0 `LaunchRecorder` + access-aware
prepare/finish stream-dependency manager. The runtime stack
itself is documented in
[device-runtime.md](device-runtime.md); this file is the
operator-author surface.

## When to use the recorded path

If your operator launches kernels on a caller-supplied
`launch_stream` (i.e., not the cudarc default stream) and any
buffer it reads or writes was allocated on a different stream
— **always**. That is the entire purpose of the recorder. The
contract is enforced at the kernel level by the runtime; there
is no "soft" way to use cross-stream buffers safely.

If your operator runs only on the default cudarc stream and
all buffers are default-stream allocated, you do not need the
recorder. The legacy `provider::*` non-recorded variants are
still supported (and still get exercised by the legacy cert
mode).

## API one-pager

```rust
use xlog_cuda::launch::LaunchRecorder;
use xlog_cuda::device_runtime::{Access, BlockId, StreamId};

// 1. Allocate every fresh runtime-backed output up front.
let mut indices = self.memory.alloc::<u32>(n)?;
let mut keys    = self.memory.alloc::<u32>(n)?;

// 2. Build the recorder and register every input + output.
let mut rec = LaunchRecorder::new_strict(launch_stream);
rec.read_column(input_col);          // input
rec.read(input.num_rows_device());   // input
rec.write(&indices);                 // fresh output
rec.write(&keys);                    // fresh output

// 3. Preflight: queues cuStreamWaitEvent for every recorded
//    use's cross-stream dependency BEFORE the launch.
rec.preflight(runtime).map_err(|e| {
    XlogError::Kernel(format!("my_op: preflight failed: {}", e))
})?;

// 4. Enqueue kernels on `launch_stream`.
unsafe {
    init_fn.clone().launch_on_stream(
        &cu_stream,
        launch_config,
        (&mut indices, input.num_rows_device(), n),
    )
}.map_err(|e| XlogError::Kernel(format!("init_indices: {}", e)))?;
// ... more launches on cu_stream ...

// 5. Commit: records the new event on `launch_stream`. After
//    commit, the next operator's preflight will wait on this
//    event when it reads or writes any block this op wrote.
rec.commit(runtime).map_err(|e| {
    XlogError::Kernel(format!("my_op: commit failed: {}", e))
})?;
```

## `Access` modes — which one to register

| Method                          | Access      | When |
|---------------------------------|-------------|------|
| `rec.read(slice)`               | `Read`      | Kernel reads `slice`'s bytes; does not modify them. |
| `rec.write(slice)`              | `Write`     | Kernel overwrites `slice`'s bytes. Use this for fresh outputs too — the recorder snapshots block identity at record time and drops the slice borrow, so a later `&mut slice` in a kernel param list is unaffected. |
| `rec.read_write(slice)`         | `ReadWrite` | Kernel reads AND modifies `slice` in place. |
| `rec.read_column(col)`          | `Read`      | `CudaColumn` input. External (`Dlpack` / `ArrowDevice`) columns are rejected at preflight in strict mode. |
| `rec.write_column(col)`         | `Write`     | `CudaColumn` output. Same external-column rejection. |

The recorder dedups by `(ptr, generation, device_ordinal)` —
registering the same block as `read` and `write` collapses to
a single `ReadWrite` prepare/finish call with the strongest
access. The dedup is locked by
`launch::tests::dedup_keys_on_full_block_id_not_ptr_alone`.

## What `preflight` does

For each unique recorded use, queue
`cuStreamWaitEvent(launch_stream, ...)` per the
[`Access` matrix](device-runtime.md#access-model):

  * **Read**: wait on the block's `last_write` if it lives on
    a different stream than `launch_stream`.
  * **Write** / **ReadWrite**: wait on `last_write` AND on
    every entry of `outstanding_reads` recorded on a
    different stream.

Same-stream events are skipped (already ordered).

The first operator that uses a freshly allocated buffer
implicitly waits on the **allocation-ready event** that
`AsyncCudaResource::allocate` records right after
`cuMemAllocAsync`. This is what guards the use-after-alloc
hazard across streams: you never need to call
`prepare_first_use` for a buffer the recorder is going to
register — the recorder handles the alloc fence too.

## What `commit` does

For each unique recorded use, record an event on
`launch_stream` and fold it into the block's dependency
state:

  * **Read** → append to `outstanding_reads`.
  * **Write** / **ReadWrite** → replace `last_write`, clear
    `outstanding_reads` (the prepare phase already queued
    waits on every prior reader, so any future op that
    observes the new `last_write` transitively observes those
    reads' completion).

After commit, the runtime's `deallocate` will wait on
`last_write` plus every cross-stream `outstanding_read`
before queueing `cuMemFreeAsync` — so dropping the slice at
end of scope is safe even if the launch is still in flight on
`launch_stream`.

## Helper-internal scratch: `prepare_first_use` /
   `finish_first_use`

If your operator calls a helper that allocates internal
scratch buffers and writes them via raw
`cuMemsetD8Async` / `cuMemcpyDtoDAsync_v2` /
`kernel.launch_on_stream` BEFORE any `LaunchRecorder::preflight`
runs against those buffers, you must fence the alloc-ready
event yourself. The pattern:

```rust
let bucket_counts = self.memory.alloc::<u32>(n)?;
runtime
    .prepare_first_use(&bucket_counts, launch_stream, Access::Write)
    .map_err(|e| XlogError::Kernel(format!(
        "my_helper: prepare bucket_counts failed: {}", e
    )))?;
// Now safe to memset / copy / launch on launch_stream:
unsafe {
    cudarc::driver::sys::cuMemsetD8Async(
        *bucket_counts.device_ptr(),
        0,
        n * 4,
        cu_stream.cu_stream(),
    );
}
```

After the helper's last write to that scratch:

```rust
runtime
    .finish_block_use(BlockId::from_block(b), launch_stream, Access::Write)
    .map_err(|e| XlogError::Kernel(format!(
        "my_helper: finish bucket_counts failed: {}", e
    )))?;
```

Sites where this is already in place (use them as
templates):

  * `build_hash_table_v2_on_stream` — bucket_counts /
    bucket_offsets / bucket_cursors / bucket_entries /
    bucket_entry_hashes.
  * `gather_buffer_by_indices_on_stream` — per-column
    `dst_col`s.
  * `multiblock_scan_u32_inplace_on_stream` /
    `_view_inplace_on_stream` — `block_sums`.
  * Every join variant's pre-recorder zero-fill of
    `d_count_only` / `d_output_count` / `out_col` / etc.

If you see a kernel produce torn data only under multi-stream
contention, the missing `prepare_first_use` after a
`self.memory.alloc(...)` is the first place to look.

## Host scalar reads — explicit synchronization required

`launch_stream` is a non-blocking stream. The cudarc default
stream's implicit synchronization with named streams **does
not apply**. Any host scalar read (`dtoh_sync_copy_into`,
`download_column`, etc.) inside a recorded chain must be
explicitly fenced against `launch_stream`:

```rust
rec.commit(runtime)?;     // queues finish event on launch_stream

cu_stream.synchronize().map_err(|e| {
    XlogError::Kernel(format!("my_op: launch_stream sync (host read): {}", e))
})?;

let row_count = self.read_join_output_count_metadata(&d_output_count)?;
```

Skipping the `synchronize()` will silently read pre-write
bytes when the kernel completion racing the host.

## External memory: DLPack / ArrowDevice

`CudaColumn::Dlpack` and `CudaColumn::ArrowDevice` columns
have no xlog-side runtime identity — `record_block_use`
cannot attach an event to memory the runtime did not allocate.
Strict-mode recorders reject them at `preflight` with
`StreamMisuse`.

If your operator must consume external columns, either:

  * Use `LaunchRecorder::new_permissive(launch_stream)`. The
    recorder silently skips external columns. Cross-stream
    safety is the **caller's** responsibility — the producing
    framework must synchronize externally before queueing your
    op's kernels. Use this only in low-level helpers during
    migration; it is not safe for production paths that
    routinely see external memory.
  * Refuse external memory at the operator boundary and
    document the caller-side coordination requirement.

## Anti-patterns

These are the patterns previous versions of the codebase
contained and that v0.6.0 explicitly removed. If you see them
in a PR, push back.

### `write_post_preflight_fresh` — gone

In the v0.5 → v0.6.0 transition the recorder grew an escape
hatch named `write_post_preflight_fresh` for fresh outputs
whose kernel-param `&mut` borrow rules conflicted with the
recorder's `&'b DeviceBlock` borrow. The v0.6.0 recorder is
**lifetime-free** — it snapshots `BlockId` at record time and
drops the source borrow immediately. There is no remaining
case where you need to record a write after preflight; the
escape hatch is removed and the standard `rec.write(slice)`
before `rec.preflight(...)` does the job.

### Recording uses after preflight

`rec.preflight(...)` freezes the recorded-uses set. Adding
`rec.read(...)` / `rec.write(...)` after preflight returns
`StreamMisuse("recorded after preflight ...")` at commit
time. Move every record call ABOVE the preflight call.

### Using the runtime singleton for production

`XlogDeviceRuntime::try_get(ordinal)` returns a singleton
backed by `DirectCudaResource`. That backend does NOT support
cross-stream tracking — its `record_block_use` /
`prepare_block_use` / `finish_block_use` return
`StreamMisuse`. Production callers that want recorded
launches must build the runtime via
`XlogDeviceRuntime::with_resource(...)` with an
`AsyncCudaResource`-backed stack (see
[device-runtime.md](device-runtime.md#stack-shape)).

### Tracking buffers via `DeviceBlock` references long-term

`DeviceBlock` carries a `Generation`. ABA reuse — the
underlying address gets freed and re-allocated with a new
generation — is detected by every prepare/finish/dealloc call
and surfaces as `ResourceError::UseAfterFree`. Never store a
`DeviceBlock` and dereference it after the originating slice
has been dropped.

If you need a stable identity across slice lifetimes, use
`BlockId::from_block(...)` and revalidate via
`prepare_block_use` (which checks the generation). Storing a
`BlockId` past the slice's drop is allowed only if you
expect the resulting `UseAfterFree` and handle it.

### Allocating helper scratch on `launch_stream`

Earlier in the v0.6.0 development cycle a fix was attempted
that routed all helper-internal scratch allocations onto
`launch_stream` directly. It was discarded as an
anti-pattern: forcing the launch stream to own all subsequent
frees inverts the `alloc_stream` / `use_stream` contract and
blocks other paths that legitimately want default-stream
allocation. The correct fix is always
`prepare_first_use(&buf, launch_stream, Access::Write)`
immediately after `self.memory.alloc(...)`.

## Validation when migrating an operator

After moving an operator onto the recorded path, run:

```sh
# Sanity: existing tests still green on legacy default.
cargo test -p xlog-cuda --tests --release -- --test-threads=1

# Recorded path under the cert harness.
XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1 \
    cargo test -p xlog-cuda-tests --test certification_suite --release

# End-to-end recorded dispatch over real workloads.
XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1 \
    cargo test -p xlog-integration --test real_world_tests --release \
    -- --test-threads=8

# A4 fork-isolated stream-safety stress (always run after
# operator-level changes; A3 thread-of-N drift is the
# documented pre-existing residual and is not a release gate).
XLOG_USE_DEVICE_RUNTIME=1 XLOG_USE_RECORDED_OPS=1 \
    cargo test -p xlog-integration --test test_a3_a4_stress --release \
    -- --test-threads=1 --nocapture
```

If any of the four above regresses, the migration is
incomplete. The most common cause when only the umbrella or
A4 regresses is a missing `prepare_first_use` after a
helper-internal scratch alloc.
