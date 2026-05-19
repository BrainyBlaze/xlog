# v0.8.6 G086_INDEX Evidence

Node: `G086_INDEX` - Persistent Hash Index Manager.

This evidence records the persistent hash-index manager slice. The work extends
the existing executor join-index cache rather than adding a second cache: keys
now include relation ID, relation generation, key columns, schema signature,
and CUDA device ordinal; the cache exposes stable telemetry for reuse,
builds, invalidations, stale rejections, budget eviction, and background-build
mode.

## GDSP

- Consumer goal: support long-running DTS-DLM, pyxlog, and v0.9.0 solver
  sessions with reusable join indexes that do not rebuild blindly across
  repeated evaluations and do not reuse stale device buffers.
- Existing subsystem reused: the existing `xlog-runtime` executor join-index
  cache, relation generation tracking, schema metadata, CUDA device ordinal,
  provider hash-index build/reuse path, recorded CUDA launch machinery, and
  runtime configuration controls.
- Scope boundary: this slice does not introduce a second index manager or a
  private host-side mirror of relation data. Background-build mode inserts the
  built index into the existing cache and defers indexed reuse until a later
  evaluation. When recorded hash joins and a runtime-backed provider are
  available, `build_join_index_v2_background` routes through
  `build_join_index_v2_recorded`, `pack_keys_gpu_on_stream`, and
  `build_hash_table_v2_on_stream`; otherwise it falls back to the legacy
  synchronous builder while preserving deferred current-evaluation reuse.

## GQM Questions

- Are persistent index keys complete enough to reject stale relation, schema,
  key-column, or device reuse?
- Does relation mutation invalidate retained entries before reuse?
- Does the cache enforce deterministic budget eviction?
- Does background-build mode record request/completion/deferred telemetry and
  use the recorded provider path where stream-backed execution is applicable?
- Does repeated session evaluation observe reuse without tracked data-plane
  DTOH/H2D calls after fixture upload?
- Are performance claims bounded to measured reuse and explicit about the
  absence of a >=1.5x timing claim?

## Metrics

- `M086_INDEX.1 manager API`: `JoinIndexKey::new` records relation ID,
  generation, key columns, schema signature, and device ordinal.
- `M086_INDEX.2 invalidation`: relation mutation invalidates retained indexes
  before a stale version can be reused.
- `M086_INDEX.3 budget`: LRU eviction bounds retained bytes under the cache
  budget and records deterministic eviction telemetry.
- `M086_INDEX.4 background build`: background-build mode records request,
  completion, and deferred-current-use telemetry; the CUDA provider has a
  runtime-backed recorded-stream build test that consumes the built index
  through the recorded indexed join path.
- `M086_INDEX.5 performance/blocker`: repeated session evaluation reuses one
  retained build-side index. No >=1.5x timing speedup is claimed in this
  fixture; the performance metric remains explicitly bounded to observed reuse.
- `M086_INDEX.6 transfer budget`: the repeated-session fixture records zero
  tracked data-plane DTOH/H2D calls after fixture upload.

## Fresh Checks

- `cargo test -p xlog-runtime persistent_hash_index -- --nocapture`
  - 4 executor reuse/invalidation/background tests passed.
- `cargo test -p xlog-cuda test_recorded_join_index_build_runs_on_runtime_stream -- --nocapture`
  - 1 runtime-backed recorded provider test passed.
- `cargo test -p xlog-runtime persistent_cache -- --nocapture`
  - 2 cache budget/invalidation tests passed.
- `cargo test -p xlog-runtime persistent_key -- --nocapture`
  - 1 cache key-hardening test passed.

Machine-readable evidence: `measurements.json`.

## Metric Interpretation

G086_INDEX correctness, invalidation, budget, keying, recorded background-build
path, deferred current-evaluation reuse, and transfer-budget metrics are PASS.
M086_INDEX.5 does not claim a >=1.5x timing speedup; the accepted v0.8.6
performance claim is deterministic retained-index reuse with the speedup gate
left unclaimed rather than inferred from correctness tests.
