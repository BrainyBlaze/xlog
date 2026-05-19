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
  provider hash-index build/reuse path, and runtime configuration controls.
- Scope boundary: this slice does not introduce a second index manager or a
  private host-side mirror of relation data. Background-build mode records
  request/completion telemetry on the existing provider path; full asynchronous
  recorded builds remain follow-up work.

## GQM Questions

- Are persistent index keys complete enough to reject stale relation, schema,
  key-column, or device reuse?
- Does relation mutation invalidate retained entries before reuse?
- Does the cache enforce deterministic budget eviction?
- Does background-build mode expose telemetry without changing the provider
  dispatch path?
- Does repeated session evaluation observe reuse without tracked data-plane
  DTOH/H2D calls after fixture upload?
- Are performance claims bounded to measured reuse and explicitly avoid
  claiming full async build speedup?

## Metrics

- `M086_INDEX.1 manager API`: `JoinIndexKey::new` records relation ID,
  generation, key columns, schema signature, and device ordinal.
- `M086_INDEX.2 invalidation`: relation mutation invalidates retained indexes
  before a stale version can be reused.
- `M086_INDEX.3 budget`: LRU eviction bounds retained bytes under the cache
  budget and records deterministic eviction telemetry.
- `M086_INDEX.4 background build`: background-build mode records request and
  completion telemetry while staying on the existing provider build/reuse path.
- `M086_INDEX.5 performance/blocker`: repeated session evaluation reuses one
  retained build-side index; full async background build speedup remains a
  follow-up because the provider build is still synchronous.
- `M086_INDEX.6 transfer budget`: the repeated-session fixture records zero
  tracked data-plane DTOH/H2D calls after fixture upload.

## Fresh Checks

- `cargo test -p xlog-runtime persistent_hash_index -- --nocapture`
  - 3 executor reuse/invalidation/background tests passed.
- `cargo test -p xlog-runtime persistent_cache -- --nocapture`
  - 2 cache budget/invalidation tests passed.
- `cargo test -p xlog-runtime persistent_key -- --nocapture`
  - 1 cache key-hardening test passed.

Machine-readable evidence: `measurements.json`.

## Metric Interpretation

All G086_INDEX correctness, invalidation, budget, keying, telemetry, and
transfer-budget metrics are PASS. The speedup claim for full asynchronous
background index builds is BLOCKED as follow-up scope; the accepted v0.8.6
claim is deterministic reuse plus safe invalidation/budget behavior on the
existing provider build/reuse path.
