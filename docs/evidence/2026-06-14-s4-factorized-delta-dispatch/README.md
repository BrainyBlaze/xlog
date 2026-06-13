# D3 Phase B S4 Bench Guard — Production-Dispatch Evidence (2026-06-14)

Plan: `docs/plans/2026-06-12-d3-phase-b-plan.md` (step 5).
Source: git archive of `feat/d3-factorized-delta` HEAD `9fcf6788`.
Logs: `runpod-s4-bench.log` (authoritative), `runpod-s4-bench-CONFOUNDED-first-run.log`
(first run, discarded — see §3).

This bench differs from the S3 gate: it drives the **production executor** on the TC
program with the factorized dispatch ON vs the kill switch ON (legacy hash-join → diff),
to prove (a) the engine actually routes the factorized path and wins where it should, and
(b) it does **not** regress sparse workloads where the work floor bails.

## Verdict: **PASS** (both arms)

| Fixture | dispatch | legacy | factorized | peak ratio | wall-clock ratio | bar |
|---|---|---|---|---|---|---|
| dense block-cycle k=4 b=256 (\|TC\|=1,048,576) | 4 (fires every iteration) | 2484.3 ms / 2332.0 MiB | 228.3 ms / 56.3 MiB | **41.46×** | **0.092×** (10.9× faster) | ≥5× peak, ≤1.2× wall |
| sparse 1500-node chain (\|TC\|=1,125,750) | 0 (work floor bails) | 147209.2 ms / 56.1 MiB | 170855.4 ms / 56.1 MiB | 1.00× | **1.161×** | ≤1.2× wall |

- **Dense** confirms the S3 spike result reproduces through the production dispatch path:
  `factorized_delta_dispatch_count == 4` (one per fixpoint iteration), 41× less peak memory
  at 11× faster wall-clock. ON/OFF row counts identical (1,048,576).
- **Sparse** confirms the dense-domain work floor protects sparse/long-chain recursion:
  zero factorized dispatches, and the factorized-default config stays within the 1.2×
  no-regression bar.

## Measurement discipline

- Host: ephemeral RunPod RTX A4000 16 GB (driver 550.144.03, nvcc 12.4, rustc 1.96.0),
  community cloud, pod `claude-agent-xlog-d3-s4-bench` (`jpmwvjj80fahaf`) — deleted after
  the run, deletion confirmed (pod list shows only foreign pods).
- Build `XLOG_CUBIN_ARCHS=sm_86`; on-pod e2e suite 8/8 green before the benches.
- **Interleaved A/B**: each rep runs legacy then factorized back-to-back, with one discarded
  warm-up per arm, so monotonic drift (thermal throttling, fragmentation) lands on both arms
  equally rather than on the A/B axis. 3 reps, median reported. Shared budget per fixture
  (10 GiB dense so the legacy arm's ~2.3 GiB peak fits; 512 MiB sparse).
- Peak via `GpuMemoryManager::peak_bytes()` (`reset_peak()` after fixture upload).

## Honesty record — the first run was discarded as confounded

The first S4 run (`runpod-s4-bench-CONFOUNDED-first-run.log`, HEAD `0cb584aa`) was **not a
valid measurement** and is kept only as a record:

- Dense OOM-panicked in 0.48 s — the e2e `make_fixture()` used a 512 MiB budget but the
  legacy arm peaks ~2.3 GiB. Harness bug, fixed with a per-fixture budget.
- Sparse reported 1.72× "regression", but all 3 legacy reps ran **before** all 3 factorized
  reps; the multi-minute sparse fixpoint throttled the GPU (logged at 1560 MHz vs the
  1890–1920 MHz of the sub-second S3 runs), so thermal drift mapped onto the A/B axis.
  Interleaving collapsed the ratio to 1.161×, proving most of the 1.72× was the artifact.

A genuine ~16 % per-iteration residual remains on the sparse arm (the dispatch-attempt
recognition + work-floor check repeated ~1500× when the floor bails). It is within the
1.2× bar but the margin is thin; **follow-up**: hoist the env-gate reads and redundant
`buffer_row_count` calls out of the per-iteration bail path (FactorizedDeltaCtx already
exists per-fixpoint to cache them). Not claimed fixed here — the merge rests on the
measured 1.161×.

## Local regression (this slice)

`cargo test --workspace --all-targets --exclude pyxlog --release --no-fail-fast`: 266 test
binaries ok; 1 cross-binary GPU-contention flake (`test_free_join_e2e ::
free_join_fires_inside_recursive_scc`, the documented contention family — passes in
isolation and the full binary passes serially 6/6). The D3 change does not alter the
all-legacy recursive path.
