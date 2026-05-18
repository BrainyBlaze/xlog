# Goal-039 G_W66_CUDAGRAPH Integration Follow-up

Date: 2026-05-18.
Branch: `feat/w6-bundle-integration-g39`.
Base evidence superseded-in-part: `docs/evidence/2026-05-18-g39-w66-cudagraph-spike.md`.

## Delta

The spike proved CUDA Graph availability but left five production blockers.
This integration follow-up implements the bounded production graph path and the
Stage-4 set-maintenance widening needed for the DTS evaluate surface:

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
- graph-mode small full-row set maintenance (`union_gpu` / deterministic
  `diff_gpu` dedup inputs) routes <=1024-row multi-column buffers through a
  one-block typed row-index sort instead of the many-launch radix
  multi-column sort path;
- recursive fixed-point merge now trusts `union_gpu`'s sorted set semantics and
  no longer immediately re-dedups the union output;
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

$ cargo test -p xlog-cuda --test set_ops_tests \
  w66_graph_mode_small_i64_full_row_set_ops_match_baseline_and_use_small_sort \
  -- --nocapture
running 1 test
test w66_graph_mode_small_i64_full_row_set_ops_match_baseline_and_use_small_sort ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 34 filtered out

$ cargo test -p xlog-integration --test test_w66_recursive_setop_profile -- --nocapture
running 1 test
test w66_recursive_union_does_not_rededup_union_output ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

$ cargo test -p xlog-cuda -- --nocapture
PASS. Key aggregate lines:
- lib unit tests: 147 passed, 0 failed
- set_ops_tests: 35 passed, 0 failed
- test_full_row_set_algebra: 20 passed, 0 failed
- test_provider_launch_recorder: 51 passed, 0 failed
- type_coverage_tests: 27 passed, 0 failed
- doc-tests: 0 failed, 5 ignored

$ cargo build -p pyxlog --release
Finished `release` profile [optimized] target(s) in 13.44s

$ XLOG_CUBIN_DIR=target/release/build/xlog-cuda-43b482a33001fc07/out \
  PYTHONPATH=/home/dev/projects/dts-dlm/src:/tmp/pyxlog-w66 \
  python3 -m dts_dlm.pilots.m37c_xlog_graph_fixture \
  --mode graph --rows 256 --runs 100 \
  --out /tmp/m37c_xlog_graph_fixture_w66_100_after_setops.json
{
  "status": "passed",
  "rows": 256,
  "runs": 100,
  "surface": "step",
  "timing_source": "cuda_event",
  "wall_seconds": 2.344730542972684,
  "mean_wall_ms_per_run": 23.44730542972684,
  "cuda_event_elapsed_ms": 2344.62255859375,
  "mean_cuda_event_ms_per_run": 23.4462255859375,
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
  --out /tmp/m37c_xlog_graph_fixture_w66_compare_96_evaluate_after_setops.json
{
  "status": "passed",
  "rows": 96,
  "runs": 100,
  "warmup_runs": 5,
  "surface": "evaluate",
  "timing_source": "synchronized_wall",
  "wall_speedup": 27.739825879056802,
  "timing_speedup": 27.739825879056802,
  "timing_reduction_pct": 96.3950747046506,
  "cuda_event_speedup": null,
  "cuda_event_reduction_pct": null,
  "structural_launch_reduction_pct": 75.0,
  "graphable_launch_units": 1000,
  "baseline": {
    "wall_seconds": 59.23670756700449,
    "mean_wall_ms_per_run": 592.3670756700449,
    "timing_elapsed_ms": 59236.70756700449,
    "mean_timing_ms_per_run": 592.3670756700449,
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
    "wall_seconds": 2.1354390552151017,
    "mean_wall_ms_per_run": 21.354390552151017,
    "timing_elapsed_ms": 2135.4390552151017,
    "mean_timing_ms_per_run": 21.354390552151017,
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
  }
}

$ git diff --check
exit 0
```

## Metric Status

| Metric | Status | Raw result |
|---|---:|---|
| M_W66.1 m37c-prime Stage 4 speedup | PASS (bounded m37c-scale evaluate cert) | Paired DTS evaluate-surface comparison at the G_PRE median row scale (`rows=96`, `runs=100`, `warmup=5`) produced `wall_speedup=27.739826x`, above the `>=1.2x` gate. |
| M_W66.2 kernel launch overhead reduction | PASS (bounded m37c-scale evaluate cert) | Structural graphable-unit launch reduction is 75%; synchronized-wall evaluate-surface timing reduction is `96.395075%`. CUDA-event timing is not available for the direct `session.evaluate` surface (`cuda_event_reduction_pct=null`). |
| M_W66.3 determinism preserved | PASS (bounded DTS cert) | DTS Stage-4 analog fixture: 100/100 bit-exact, `unique_digest_count=1`. |
| M_W66.4 DLPack zero-copy preserved | PASS (bounded DTS cert) | DTS Stage-4 analog fixture: `dtoh_bytes=0`, `dtoh_calls=0`, `htod_bytes=0`, `htod_calls=0`. |
| M_W66.5 peak VRAM <= 38 GB | PASS (bounded DTS cert) | DTS Stage-4 analog fixture peak snapshot 1.258 GiB; bounded evaluate comparison peak snapshot 1.237 GiB, both below 38 GiB. Full m37c-prime replay remains covered by G_E2E KPI-5. |
| M_W66.6 recapture <= 1x per fixpoint iteration | PASS (bounded DTS cert) | DTS Stage-4 analog fixture on the W65-corrected DTS source: 4 captures for 4 graphable topologies, 1000 launches, 996 cache hits, 0 fallbacks across 100 evaluate runs. |

## Closeout

W66's bounded DTS/pyxlog certs are green after widening graph-mode Stage-4 set
maintenance. The remaining full m37c-prime replay and KPI-5 VRAM snapshot are
tracked by G_E2E, not by this bounded W66 fixture.
