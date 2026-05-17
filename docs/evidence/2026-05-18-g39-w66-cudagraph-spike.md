# Goal-039 G_W66_CUDAGRAPH Spike

Date: 2026-05-18.
Branch: `bench-spike/w66-cuda-graph-g39`.
Base: `feat/w3-bundle-integration @ c1689d70ed73867233298a5116546db833ba48f7`.

## Scope

G_W66 asks for CUDA Graph capture of the Stage 4 hot-loop kernel sequence,
with acceptance gates M_W66.1-6. This spike validates whether the current
xlog-cuda runtime stack can support CUDA Graph primitives, then checks whether
the production count-scan-materialize (CSM) join path can be safely promoted to
the requested cached graph unit.

No DTS-DLM source was modified in this branch. No new env var was introduced.

## Positive Result

`crates/xlog-cuda/tests/test_w66_cuda_graph_smoke.rs` adds a runtime-backed
CUDA Graph smoke cert:

- allocate a runtime-backed device buffer;
- acquire a non-default runtime stream;
- queue the alloc-ready dependency;
- capture `cuMemsetD8Async` with `cuStreamBeginCapture_v2` /
  `cuStreamEndCapture`;
- instantiate with `cuGraphInstantiateWithFlags`;
- replay three times with `cuGraphLaunch`;
- record the runtime use event and verify the replayed bytes.

Command:

```bash
cargo test -p xlog-cuda --test test_w66_cuda_graph_smoke -- --nocapture
```

Raw result:

```text
running 1 test
test cuda_graph_replays_runtime_backed_memset_on_launch_stream ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.75s
```

This proves the CUDA Graph driver API is usable on the current runtime-backed
stream stack. The remaining blocker is not CUDA Graph availability; it is the
production CSM graph-caching shape.

## Production CSM Blockers

### B1. Host resize boundary splits the requested graph unit

`hash_join_inner_v2_count_scan_materialize_recorded` currently executes
count/scan, then synchronizes the launch stream, reads `d_logical_count` on the
host, and only then allocates `d_output_left` / `d_output_right` sized to the
runtime total.

Code anchors:

- `crates/xlog-cuda/src/provider/relational.rs:7465`: launch-stream sync for
  total read.
- `crates/xlog-cuda/src/provider/relational.rs:7470`: host metadata read of
  `d_logical_count`.
- `crates/xlog-cuda/src/provider/relational.rs:7494`: materialize output
  allocation after the host read.

G_W66's natural unit is Count -> Scan -> Resize -> Materialize. The current
path's Resize step is CPU-controlled allocation based on host-read metadata,
so a single reusable CUDA Graph cannot include the whole unit.

### B2. Scan scratch is allocated inside the captured topology

`multiblock_scan_u32_inplace_on_stream` allocates recursive `block_sums`
scratch inside the scan helper. The allocation size and recursive topology
depend on `n`.

Code anchors:

- `crates/xlog-cuda/src/provider/mod.rs:1654`: `num_blocks` derives topology
  from runtime `n`.
- `crates/xlog-cuda/src/provider/mod.rs:1655`: `block_sums` scratch allocated
  inside the helper.
- `crates/xlog-cuda/src/provider/mod.rs:1700`: recursive scan on
  `block_sums`.

A cached graph would capture concrete scratch addresses and scan depth. After
the helper returns, local scratch drops. Production replay therefore needs
graph-owned scratch with stable capacity classes, or a graph update layer that
updates every scratch node parameter before replay.

### B3. Current W66 cache key is insufficient

The plan key `(rule_id, schema_signature)` is not enough for a CUDA Graph exec.
CUDA Graph replay is sensitive to device pointers, output capacity, scan depth,
memcpy byte counts, memset sizes, and kernel launch dimensions. The CSM path
also varies by `probe_cap`, `num_right`, `max_output`, key arity, and scan
topology.

A production key must include at least:

- rule id;
- schema signature;
- join key arity and scalar type;
- probe capacity class;
- output capacity class;
- scan topology/depth;
- fixed graph node layout version.

Without node-parameter updates or stable graph-owned buffers, a replay keyed
only by rule/schema would either recapture per call or replay stale pointers.

### B4. DLPack zero-copy cannot be preserved by staging

Strict launch recording intentionally rejects external DLPack / Arrow device
columns because they have no xlog-side runtime identity:

- `crates/xlog-cuda/src/launch.rs:79`: external-memory section.
- `crates/xlog-cuda/src/launch.rs:81`: strict mode rejects DLPack /
  ArrowDevice columns.
- `crates/xlog-cuda/src/launch.rs:288`: `read_column` records owned columns
  and identifies external columns.

The tempting graph workaround is to copy DLPack inputs into xlog-owned staging
buffers so graph node pointers stay stable. That would violate the DTS-DLM
zero-copy contract. Production W66 needs external stream/event interop and
graph-node parameter updates for external input pointers, not staging copies.

## Metric Status

| Metric | Status | Raw result |
|---|---:|---|
| M_W66.1 m37c-prime Stage 4 speedup | BLOCKED | No production graph path can be enabled without B1-B4; running a wall-time benchmark would only measure baseline/CSM, not CUDA Graph capture. |
| M_W66.2 kernel launch overhead reduction | BLOCKED | Graph primitive works, but no production CSM graph replay path exists to measure launch reduction. |
| M_W66.3 determinism preserved | BLOCKED | Primitive replay cert is deterministic for a memset smoke test; no Stage 4 graph path exists for the required 100/100 subset cert. |
| M_W66.4 DLPack zero-copy preserved | BLOCKED | Staging-copy workaround rejected; production path needs external-pointer graph updates and event interop. |
| M_W66.5 peak VRAM <= 38 GB | NOT RUN | No production graph path to profile. |
| M_W66.6 recapture <= 1x per fixpoint iteration | BLOCKED | Current plan key would recapture per pointer/capacity/topology change or replay stale pointers. |

## Required Follow-up Before W66 Can Close

W66 cannot honestly close under the current implementation architecture. A
production-grade close needs a prerequisite slice that delivers:

1. A CUDA Graph execution object with explicit node inventory and safe
   `cuGraphExec*NodeSetParams` update support.
2. Graph-owned scan scratch with stable capacity classes, or fully updated
   scratch-node parameters before every replay.
3. A bounded-output CSM graph protocol that avoids the host resize boundary, or
   a two-graph split with a documented acceptance amendment replacing the
   single Count -> Scan -> Resize -> Materialize unit.
4. DLPack external-pointer update and stream/event interop without staging
   copies.
5. A revised cache key that includes topology and capacity, not only
   `(rule_id, schema_signature)`.

Until those prerequisites exist, W6.6 must remain OPEN and G_W66 should be
treated as STUCK, not DONE.
