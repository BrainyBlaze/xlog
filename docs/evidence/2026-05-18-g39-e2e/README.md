# Goal-039 G_E2E Evidence

Date: 2026-05-18.
Branch: `feat/w6-bundle-integration-g39`.
Phase-2 HEAD: `d96b411c`.
DTS-DLM branch: `feat/g39-w66-xlog-graph-fixture`.
DTS-DLM HEAD: `d7a470c`.

This evidence closes the Goal-039 G_E2E validation gate without editing the
closure board, pushing, tagging, or merging.

## Wheel Install

`maturin develop --release` could not be used because the local Python
environment is not a virtualenv or conda environment. The accepted local install
path used a wheel build and user-site reinstall:

```text
maturin build --release --compatibility linux --auditwheel skip -o /tmp/g39-e2e-wheel
python3 -m pip install --user --force-reinstall \
  /tmp/g39-e2e-wheel/pyxlog-0.6.2-cp310-cp310-linux_x86_64.whl
```

Both commands exited 0. The installed module after the final restore was:

```text
pyxlog 0.6.2
/home/dev/.local/lib/python3.10/site-packages/pyxlog/__init__.py
/usr/bin/python3
```

## m37c-prime 50-doc Arm-C Replay

Strict baseline:

- xlog commit: `f62188b7`
- run id: `g39-e2e-baseline-f62188b7-50doc-armc-r1`
- manifest: `/tmp/g39-e2e-baseline-f62188b7-m37c/g39-e2e-baseline-f62188b7-50doc-armc-r1/eval/manifest.json`
- docs: 50
- failures: 0
- wall: `1466.4236392974854 s`

Phase-2 candidate:

- xlog commit: `d96b411c`
- run id: `g39-e2e-50doc-armc-instrumented-20260518-r2`
- metrics: `/tmp/g39-e2e-m37c-prime/g39-e2e-50doc-armc-instrumented-20260518-r2/eval/g39_e2e_metrics.json`
- docs: 50
- failures: 0
- wall: `278.3887164592743 s`
- speedup vs `f62188b7`: `5.267539783754166x`

The candidate run used process-local instrumentation to record timing, memory,
xlog provider host-transfer telemetry, and derived-fact witness coverage. DTS-DLM
source files were not edited for this instrumentation.

## Determinism and Drift

Two full 50-doc candidate runs produced identical semantic arm-C output after
excluding wall-time fields:

```text
g39-e2e-50doc-armc-instrumented-20260518-r1 80db1afa787834d02fca695979f253100d48d0a86d3b60367098daede0b06d1c
g39-e2e-50doc-armc-instrumented-20260518-r2 80db1afa787834d02fca695979f253100d48d0a86d3b60367098daede0b06d1c
```

A captured m37c-prime Stage-4 frozen input bundle from
`pubmed_41804240` was replayed 100 times:

- capture manifest: `/tmp/g39-e2e-m37c-prime/g39-e2e-m37c-stage4-capture/manifest.json`
- replay summary: `/tmp/g39-e2e-m37c-prime/g39-e2e-m37c-stage4-capture/replay_100_summary.json`
- trials: 100
- unique digest count: 1
- digest: `0e995a890102d17a6181e6558781d8437bb783efdc9ce786e9cc2c0c23590f9e`
- wall: `5.758074045181274 s`

The same captured bundle was replayed once with the `f62188b7` wheel and once
with the restored Phase-2 wheel. The digests matched exactly:

```text
f62188b7  0e995a890102d17a6181e6558781d8437bb783efdc9ce786e9cc2c0c23590f9e
d96b411c  0e995a890102d17a6181e6558781d8437bb783efdc9ce786e9cc2c0c23590f9e
```

## Instrumented Candidate Metrics

Raw values from
`/tmp/g39-e2e-m37c-prime/g39-e2e-50doc-armc-instrumented-20260518-r2/eval/g39_e2e_metrics.json`:

```text
compile_count = 281
compile_median_ms = 13.499273
compile_max_ms = 80.346288
evaluate_count = 1556
evaluate_median_ms = 31.9969655
evaluate_total_ms = 49185.592474999954
peak_vram_bytes = 12820480000
peak_vram_gib = 11.94000244140625
host_transfer_delta = { dtoh_bytes: 0, htod_bytes: 0, dtoh_calls: 0, htod_calls: 0 }
derived_fact_count = 89
derived_recoverable = 89
derived_unrecoverable = 0
derived_bad_premise_edges = 0
derived_justification_rows = 712
support_active_witness_coverage = 1.0
memory_allocated_initial_after = 1829355520
memory_allocated_final_after = 1799425024
memory_allocated_final_minus_initial = -29930496
```

## DTS-DLM API Regression

Call-site scan:

```text
rg -n "pyxlog\." src/dts_dlm
```

The scan found the expected DTS pyxlog call sites under:

- `src/dts_dlm/learn/`
- `src/dts_dlm/pilots/`
- `src/dts_dlm/integrations/pyxlog/`
- `src/dts_dlm/propagate/xlog_executor.py`

Functional regression slice after restoring the Phase-2 wheel:

```text
PYTHONPATH=src pytest -q \
  src/tests/integrations/pyxlog \
  src/tests/propagate/test_xlog_executor.py \
  src/tests/pipeline/test_pyxlog_pipeline.py \
  src/tests/govern/test_pyxlog_govern.py \
  src/tests/learn/test_m18_xlog_alpha_source.py \
  src/tests/learn/test_m18_xlog_alpha_integration.py \
  src/tests/learn/test_m18_online_learn_xlog_dispatch.py \
  src/tests/pilots/test_promotion.py \
  src/tests/pilots/test_m37c_prime_pilot_smoke.py \
  src/tests/pilots/test_m37c_prime_verdict.py \
  src/tests/pipeline/test_m37c_constraint.py
```

Result:

```text
179 passed, 14 warnings in 19.45s
```

## Explicit W4.1 Cert

Command:

```text
cargo test -p xlog-integration --test test_wcoj_recursive_dispatch -- --nocapture
```

Result:

```text
8 passed; 0 failed
```

The three W4.1-specific positive certs are included in the eight-test binary:
`multirec_triangle_dispatches_wcoj_and_matches_binary_join`,
`multirec_4cycle_dispatches_wcoj_and_matches_binary_join`, and
`selfrec_triangle_dispatches_wcoj_and_matches_binary_join`.

## Metric Status

| Metric | Status | Raw result |
|---|---|---|
| M_E2E.1 wall-time speedup vs `f62188b7` | PASS | `1466.4236392974854 / 278.3887164592743 = 5.267539783754166x` |
| M_E2E.2 determinism | PASS | m37c-prime Stage-4 frozen bundle `100/100`, one digest; full 50-doc semantic digest `2/2` |
| M_E2E.3 DTS-DLM API regression | PASS | pyxlog call-site scan plus DTS regression slice green |
| M_E2E.4 DTS-DLM serving regression equivalent | PASS | DTS regression slice `179 passed` |
| M_E2E.5 wheel install | PASS | `maturin build --release --compatibility linux --auditwheel skip` exit 0; pip reinstall exit 0 |
| M_E2E.6 peak VRAM | PASS | `12,820,480,000 bytes` <= `38 GB` |
| M_E2E.7 witness recoverability | PASS | `89/89` derived facts recoverable; `0` bad derived premise edges |
| M_E2E.8 DLPack zero-copy | PASS | xlog provider host-transfer delta `0` DtoH/HtoD bytes and calls |
| M_E2E.9 compile median | PASS | `13.499273 ms` <= `50 ms` |
| M_E2E.10 cross-doc cleanup | PASS | final-after minus initial-after allocated bytes `-29,930,496` |
| M_E2E.11 behavioral drift | PASS | `f62188b7` and `d96b411c` captured-bundle digests match |
| M_E2E.12 explicit W4.1 cert | PASS | `8 passed; 0 failed`, including the 3 W4.1 positive certs |

## Process State

- No closure-board edit was made.
- No merge was made.
- No push was made.
- No tag was created.
- DTS-DLM source files were not edited during G_E2E.
- The local DTS Python environment was restored to the Phase-2 pyxlog wheel after
  the `f62188b7` drift and baseline runs.
