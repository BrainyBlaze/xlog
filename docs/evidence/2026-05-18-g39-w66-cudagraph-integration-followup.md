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
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.06s

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

$ git diff --check
exit 0
```

## Metric Status

| Metric | Status | Raw result |
|---|---:|---|
| M_W66.1 m37c-prime Stage 4 speedup | PENDING | Bounded graph path exists; m37c-prime wall-time not run yet. |
| M_W66.2 kernel launch overhead reduction | PENDING | CSM graph smoke captures Count -> Scan -> Total -> Materialize as one graph launch; overhead benchmark not run yet. |
| M_W66.3 determinism preserved | PARTIAL | Memset graph replay and bounded inner-CSM graph fixture pass; 100/100 subset cert not run yet. |
| M_W66.4 DLPack zero-copy preserved | PARTIAL | Graph implementation does not stage/copy DLPack or Arrow columns; true external pointer event interop remains graceful-fallback territory. |
| M_W66.5 peak VRAM <= 38 GB | NOT RUN | No m37c-prime profile yet. |
| M_W66.6 recapture <= 1x per fixpoint iteration | PARTIAL | Same-topology bounded CSM fixture captures once, launches twice, and records one cache hit; full fixpoint cert not run yet. |

## Remaining Work

W66 is no longer blocked at CUDA Graph primitives, scan scratch, or bounded
output protocol, or same-topology replay caching. It still needs:

1. 100/100 deterministic subset run with `XLOG_USE_CSM_CUDA_GRAPH=1`;
2. launch-overhead and Stage 4 wall-time measurements;
3. m37c-prime VRAM profile;
4. DTS-DLM analog fixture coverage for the graph path or a documented graceful
   flag where true external DLPack event interop is unavailable.
