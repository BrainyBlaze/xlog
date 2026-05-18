# Goal-039 G_W66_CUDAGRAPH Integration Follow-up

Date: 2026-05-18.
Branch: `feat/w6-bundle-integration-g39`.
Base evidence superseded-in-part: `docs/evidence/2026-05-18-g39-w66-cudagraph-spike.md`.

## Delta

The spike proved CUDA Graph availability but left five production blockers.
This integration follow-up implements the first bounded production path:

- CUDA Graph node inventory via `CapturedCudaGraph::nodes()`;
- graph-exec node update wrappers for kernel and memset nodes;
- explicit CSM graph cache-key model with capacity classes, scan topology, and
  `CSM_CUDA_GRAPH_NODE_LAYOUT_VERSION`;
- provider-level bounded CSM CUDA Graph replay cache keyed by
  `CsmCudaGraphKey`, with graph-owned count/offset/output buffers and kernel
  node param updates for new runtime pointers;
- graph-owned recursive scan scratch for `multiblock_scan_u32_inplace_on_stream`;
- opt-in bounded inner-CSM graph path behind `XLOG_USE_CSM_CUDA_GRAPH=1`;
- pyxlog provider construction now uses a runtime-backed `GpuMemoryManager`
  and `CudaKernelProvider::with_runtime`, so DTS-DLM's existing pyxlog
  session path can reach recorded CSM / CUDA Graph execution;
- the common <=4-key pack/hash kernel now receives column element sizes as a
  packed scalar argument instead of uploading a hot-path metadata buffer;
- no DLPack or Arrow staging copies on the graph path.

The graph path is intentionally bounded. If `max_output` is supplied, output
index capacity is rounded to a graph capacity class. If `max_output` is absent,
the path only captures when the worst-case output fits
`XLOG_CSM_CUDA_GRAPH_AUTO_OUTPUT_CAP` (default `1_000_000` rows); otherwise it
falls back to the existing recorded CSM path.

## Raw Validation

```text
$ cargo fmt --check
exit 0

$ cargo check -p xlog-cuda --tests
Finished `dev` profile [unoptimized + debuginfo] target(s) in 2.61s

$ cargo check -p pyxlog
Finished `dev` profile [unoptimized + debuginfo] target(s) in 3.34s

$ cargo test -p xlog-cuda --test pack_keys_gpu -- --nocapture
running 2 tests
test test_pack_keys_gpu_common_path_no_host_transfers ... ok
test test_pack_keys_gpu_generic_no_dtoh ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

$ cargo test -p xlog-cuda cuda_graph::tests --lib -- --nocapture
running 2 tests
test cuda_graph::tests::csm_key_uses_capacity_classes_and_layout_version ... ok
test cuda_graph::tests::scan_topology_matches_recursive_multiblock_shape ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 145 filtered out

$ cargo test -p xlog-cuda --test test_w66_cuda_graph_smoke -- --nocapture
running 2 tests
test cuda_graph_replays_runtime_backed_memset_on_launch_stream ... ok
test csm_inner_join_uses_bounded_cuda_graph_when_enabled ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

The CSM graph smoke asserts `captures += 1`, `launches += 2`,
`cache_hits += 1`, fallback count unchanged, and correct joined rows for two
same-topology inputs with different device buffers.

$ cargo test -p xlog-cuda --test test_csm_env_dispatch -- --nocapture
running 9 tests
test dispatch_does_not_route_to_csm_when_no_recorded_env_is_set ... ok
test dispatch_routes_to_csm_for_inner_non_indexed_with_umbrella_env ... ok
test dispatch_does_not_route_to_csm_for_semi_or_anti_under_csm_env ... ok
test dispatch_routes_to_csm_for_inner_indexed_with_umbrella_env ... ok
test dispatch_does_not_route_to_csm_when_only_hash_join_env_is_set ... ok
test dispatch_routes_to_csm_for_left_outer_indexed_with_umbrella_env ... ok
test dispatch_routes_to_csm_for_left_outer_non_indexed_with_umbrella_env ... ok
test dispatch_short_circuits_before_csm_for_more_than_four_keys ... ok
test dispatch_routes_to_csm_when_recorded_csm_env_is_set_directly ... ok
test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

$ cargo build -p pyxlog --release
Finished `release` profile [optimized] target(s) in 33.97s

$ XLOG_CUBIN_DIR=target/release/build/xlog-cuda-43b482a33001fc07/out \
  PYTHONPATH=/home/dev/projects/dts-dlm/src:/tmp/pyxlog-w66 \
  python3 -m dts_dlm.pilots.m37c_xlog_graph_fixture \
  --mode graph --rows 256 --runs 100 \
  --out /tmp/m37c_xlog_graph_fixture_w66_100_rebased_v2.json
{
  "status": "passed",
  "rows": 256,
  "runs": 100,
  "surface": "step",
  "timing_source": "cuda_event",
  "wall_seconds": 69.35122062201845,
  "mean_wall_ms_per_run": 693.5122062201845,
  "cuda_event_elapsed_ms": 69350.953125,
  "mean_cuda_event_ms_per_run": 693.50953125,
  "deterministic": true,
  "unique_digest_count": 1,
  "support_rows_min": 256,
  "support_rows_max": 256,
  "usable_rows_min": 768,
  "usable_rows_max": 768,
  "expected_graph_capture_topology_cap": 4,
  "graph_delta": {
    "captures": 4,
    "launches": 1000,
    "fallbacks": 0,
    "cache_hits": 996
  },
  "host_transfer_delta": {
    "dtoh_bytes": 0,
    "dtoh_calls": 0,
    "htod_bytes": 0,
    "htod_calls": 0
  },
  "peak_vram_gib_snapshot": 1.25836181640625
}

$ XLOG_CUBIN_DIR=target/release/build/xlog-cuda-43b482a33001fc07/out \
  PYTHONPATH=/home/dev/projects/dts-dlm/src:/tmp/pyxlog-w66 \
  python3 -m dts_dlm.pilots.m37c_xlog_graph_fixture \
  --mode compare --surface evaluate --rows 96 --runs 100 --warmup-runs 5 \
  --out /tmp/m37c_xlog_graph_fixture_w66_compare_96_evaluate_rebased_v2.json
{
  "status": "failed",
  "rows": 96,
  "runs": 100,
  "warmup_runs": 5,
  "surface": "evaluate",
  "timing_source": "synchronized_wall",
  "wall_speedup": 1.0095365191184626,
  "timing_speedup": 1.0095365191184626,
  "timing_reduction_pct": 0.9446433029277612,
  "cuda_event_speedup": null,
  "cuda_event_reduction_pct": null,
  "structural_launch_reduction_pct": 75.0,
  "graphable_launch_units": 1000,
  "baseline": {
    "wall_seconds": 70.01620508154156,
    "mean_wall_ms_per_run": 700.1620508154156,
    "timing_elapsed_ms": 70016.20508154156,
    "mean_timing_ms_per_run": 700.1620508154156,
    "timing_source": "synchronized_wall",
    "cuda_event_elapsed_ms": null,
    "mean_cuda_event_ms_per_run": null,
    "graph_delta": {
      "captures": 0,
      "launches": 0,
      "fallbacks": 0,
      "cache_hits": 0
    },
    "host_transfer_delta": {
      "dtoh_bytes": 0,
      "dtoh_calls": 0,
      "htod_bytes": 0,
      "htod_calls": 0
    },
    "peak_vram_gib_snapshot": 1.23687744140625
  },
  "graph": {
    "wall_seconds": 69.35480168927461,
    "mean_wall_ms_per_run": 693.5480168927461,
    "timing_elapsed_ms": 69354.80168927461,
    "mean_timing_ms_per_run": 693.5480168927461,
    "timing_source": "synchronized_wall",
    "cuda_event_elapsed_ms": null,
    "mean_cuda_event_ms_per_run": null,
    "graph_delta": {
      "captures": 0,
      "launches": 1000,
      "fallbacks": 0,
      "cache_hits": 1000
    },
    "host_transfer_delta": {
      "dtoh_bytes": 0,
      "dtoh_calls": 0,
      "htod_bytes": 0,
      "htod_calls": 0
    },
    "peak_vram_gib_snapshot": 1.23687744140625
  },
  "failures": [
    "expected wall_speedup>=1.2, got 1.009537",
    "expected timing_reduction_pct>=50.0, got 0.944643"
  ]
}

$ git diff --check
exit 0
```

## Metric Status

| Metric | Status | Raw result |
|---|---:|---|
| M_W66.1 m37c-prime Stage 4 speedup | FAIL (bounded m37c-scale evaluate cert) | Paired DTS evaluate-surface comparison at the G_PRE median row scale (`rows=96`, `runs=100`, `warmup=5`) produced `wall_speedup=1.009537x`, below the `>=1.2x` gate. |
| M_W66.2 kernel launch overhead reduction | FAIL (bounded m37c-scale evaluate cert) | Structural graphable-unit launch reduction is 75%, but measured synchronized-wall evaluate-surface timing reduction is `0.944643%`; CUDA-event timing is not available for the direct `session.evaluate` surface (`cuda_event_reduction_pct=null`). |
| M_W66.3 determinism preserved | PASS (bounded DTS cert) | DTS Stage-4 analog fixture: 100/100 bit-exact, `unique_digest_count=1`. |
| M_W66.4 DLPack zero-copy preserved | PASS (bounded DTS cert) | DTS Stage-4 analog fixture: `dtoh_bytes=0`, `dtoh_calls=0`, `htod_bytes=0`, `htod_calls=0`. |
| M_W66.5 peak VRAM <= 38 GB | PARTIAL | DTS Stage-4 analog fixture peak snapshot 1.258 GiB; bounded evaluate comparison peak snapshot 1.237 GiB; no full m37c-prime profile yet. |
| M_W66.6 recapture <= 1x per fixpoint iteration | PASS (bounded DTS cert) | DTS Stage-4 analog fixture on the W65-corrected DTS source: 4 captures for 4 graphable topologies, 1000 launches, 996 cache hits, 0 fallbacks across 100 evaluate runs. |

## Remaining Work

W66 is no longer blocked at CUDA Graph primitives, scan scratch, bounded output
protocol, same-topology replay caching, pyxlog runtime-backed construction, or
DTS-DLM bounded graph-path certification. It is blocked on performance: the
current graph capture unit is too narrow for the m37c-scale evaluate surface
because recursive support evaluation still spends most of its time outside the
captured CSM subsequence. The next viable implementation must either capture a
broader Stage-4 sequence or replace the pyxlog session hot loop with a
prepared, reusable Stage-4 execution surface whose launch/update boundary is
large enough to affect wall time. It still needs:

1. a broader Stage-4 execution/capture implementation that clears M_W66.1 and M_W66.2;
2. m37c-prime VRAM profile.
