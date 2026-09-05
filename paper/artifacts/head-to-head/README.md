# Head-to-head and overhead-isolation artifacts

Benchmark artifacts backing the head-to-head and overhead-isolation claims in
`sections/10_evaluation.tex`. All seven were re-measured on **ephemeral cloud
GPUs** (RunPod) between 2026-09-01 and 2026-09-03, each by the committed runner
next to it. Every artifact records its own hardware, protocol and per-cell
measurements; hardware is not mixed within a comparison.

| File | Comparison | Hardware | CPU quota | `comparison_acceptable` |
|------|-----------|----------|-----------|--------------------------|
| `mnist_addition_vs_scallop.json` | Neural: MNIST addition, xlog vs Scallop | L40S | 13.6 cores | **true** |
| `exact_inference_vs_problog2.json` | Probabilistic: exact inference, xlog vs ProbLog2 | A100-SXM4 80GB | 13.6 cores | **true** |
| `triangle_counting_vs_souffle.json` | Deterministic: fused WCOJ triangle counting vs Soufflé (heavy skew) | A100 80GB PCIe | 26.35 cores | **true** |
| `triangle_counting_moderate_skew_vs_souffle.json` | Deterministic: WCOJ vs binary vs Soufflé, moderate skew | 2x L40S | 54.4 cores | **true** |
| `residency_ablation.json` | xlog-only: forced host round-trip, single query | A100 80GB PCIe | 26.35 cores | n/a (single-system) |
| `residency_scale_ablation.json` | xlog-only: forced host round-trip vs handoff count | A100 80GB PCIe | 26.35 cores | n/a (single-system) |
| `verify_overhead_isolation.json` | xlog-only: CDCL-verify vs D4-compile split | A100 80GB PCIe | 26.35 cores | n/a (single-system) |
| `mnist_addition_vs_scallop_quota_companion.json` | companion: the 27.2-core half of the CPU-quota observation | L40S x2 | 27.2 cores | **false** (see below) |

The engine is the same build in all of them: the runners were added on top of
`a2bafef0` and touch no file under `crates/`, so the differences between rows
are hardware, recorded per artifact, and not product versions.

**The CPU quota is part of the result, not a footnote.** It is what the Soufflé
baseline gets, and it varies by which host RunPod hands out. The earlier
published run gave Soufflé 7.65 cores; these give it 26.35 and 54.4.

## Protocol notes

- **Triangle counting vs Soufflé** — fused-WCOJ count (A), enumerate-then-count
  (B) and Soufflé count (C) on hub-skewed graphs, five sizes. All three arms
  complete at every size and agree on the per-root counts, gated on the sha256
  of the relation rather than on equal cardinality. Soufflé-over-fused is
  `0.88x`, `1.96x`, `2.59x`, `3.88x`, `5.54x` from 150k to 1.2M edges — **at the
  smallest size Soufflé is faster than xlog**, and the ratio only becomes a
  speed-up from 300k edges on. What grows monotonically is the ratio itself, so
  the defensible claim is the scaling, not any single cell. The memory split is
  the wider gap: fused counting peaks at 85/204/359/618/1,033 MB of provider
  allocations against 3,287/8,403/15,247/26,497/44,979 MB for the enumerate arm.
  One caveat the artifact carries and a reader should apply: xlog's engine time
  is 25--43 ms of that 0.57--0.68 s wall, so 93--96% of the xlog side is
  process start, CUDA context and Arrow input. What is flat at these sizes is
  the fixed cost.
- **Triangle counting, moderate skew** — the companion at a hub-edge fraction of
  0.25 rather than 0.8. Fused WCOJ over xlog's own binary join is `2.00x`,
  `2.36x`, `3.67x`, `4.95x` across 40k--400k edges. Note that the binary arm
  **runs to completion at every size here**, including the two the previous
  artifact recorded as `"skipped": "binary blowup"`; the WCOJ advantage in this
  file is therefore measured against a baseline that finished, not inferred from
  one that gave up.
- **Exact inference vs ProbLog2** — 5 programs; correctness gate: query
  probabilities match the analytic answer within 1e-4 (both engines reach 0
  error). Timing is full inference (compile + evaluate), median of 3, both
  engines in-process. xlog is slower than ProbLog2 on these tiny programs, as it
  was in the previous artifact: the comparison is about exactness, not speed.
- **MNIST vs Scallop** — identical MNISTNet, data, metric and seeds, with the
  two harnesses driven on separate interpreters (`--scallop-python`), because
  the published `scallopy` wheel is cp310-only. Two protocols: the whitepaper
  512-image/5-epoch setting (both near-chance, under-trained) and a stronger
  20k-image/5-epoch setting (both ~95%). **Accuracy is reportable; the timing
  columns are not, yet** — see `paper_usage` in the artifact. Scallop's steady
  epoch is not monotone in CPU cores (31.79 s at 13.6 cores, 54.99 s at 27.2 on
  the same card and the same wheel), so the epoch ratio moves with the host
  rather than with either engine.
- **Residency ablation** — same pipeline with and without
  `XLOG_FORCE_HOST_ROUNDTRIP`; the on-minus-off per-iteration delta is the
  transfer cost residency eliminates. The single-query file measures 2 handoffs
  and reports 2.7% and 2.5% for 4 and 10 labels. The `_scale` file sweeps the
  batched path (2--512 handoffs) and is the one to cite: per-handoff round-trip
  is 61--176 us, and the round-trip's share of a step climbs from 2.7% at one
  query to 16.1% at 64 and 15.5% at 256. The earlier artifact's negative cells
  (-8.9%, -257 us per handoff) do not appear in this run.
- **Verify-overhead isolation** — `program.warmup_breakdown()` under
  `XLOG_WARMUP_PROFILE=1` splits the cold compile into D4-compile and on-GPU
  CDCL equivalence-verify. Verify is 98.8--99.4% of the cold compile across
  n=5..40, and `d4_compile_ms` is non-zero at every point, unlike the earlier
  record. `n >= 50` exceeds a CUDA grid-dimension limit on this CNF and is out
  of range. This run stops at n = 40 and does not itself contain that
  observation: the sweep's error stream stayed on the pod, so the limit is
  carried over from the earlier record rather than reproduced here.

## The MNIST quota companion

`mnist_addition_vs_scallop_quota_companion.json` is **not** an accepted
comparison and its `comparison_acceptable` is `false`: the two arms read
MNIST from different roots, which the runner flags as a protocol
divergence. It is distributed for one purpose, to back the CPU-quota
observation in Limitations: Scallop's steady epoch is 54.99 s here at a
27.2-core quota against 31.79 s at 13.6 cores in the accepted run, same
card, same wheel, same seeds. The two pods also differed in driver revision
and host, so the quota is the candidate explanation and not an isolated
cause.

## A note on file shape

`residency_ablation.json`, `residency_scale_ablation.json` and
`verify_overhead_isolation.json` used to be a bare list of numbers with no
hardware, commit or date. They are now objects with a `provenance` block and the
same rows under `results`. Nothing parses these files programmatically; the
change exists so that a reader can tell what machine produced a row.

## Reproduction

Every comparison in this directory now has a committed runner, and **the numbers
in these files were produced by those runners** — that was not true of the
previous set, which came from ephemeral on-pod scripts that were never
committed. The runners refuse a dirty checkout, carry `--self-test`, and record
the commit, input and binary hashes, normalized commands, software and hardware
versions, per-repetition failures and the correctness gates.

- `runners/triangle_counting_vs_souffle.py`
- `runners/triangle_counting_moderate_skew_vs_souffle.py`
- `runners/exact_inference_vs_problog2.py`
- `runners/mnist_addition_vs_scallop.py`
- `runners/residency_sweep.py`, `runners/residency_scale.py`,
  `runners/verify_sweep.py` for the three xlog-only isolations (run with
  `python -u` on a CUDA-enabled `pyxlog` build; the residency scripts need
  `torch`).

Two limits of that reconstruction are worth stating plainly, because they mean a
re-run is not always a repeat:

- the **moderate-skew generator parameters** were never recorded in the
  published artifact — only the edge counts — so its runner writes the hub
  fraction it used (0.25) and this file is a new measurement of the same class
  rather than a repeat of the old one. The triangle counts differ from the
  earlier artifact for that reason and not because the engine changed;
- the **MNIST seeds** were likewise unrecorded; the runner uses the repository's
  `DEFAULT_SEEDS`, which are the seeds the committed Scallop baseline results
  were produced with.

Triangle counting, end to end:

```bash
cargo build --release --locked -p xlog-cli
python -u paper/artifacts/head-to-head/runners/triangle_counting_vs_souffle.py \
  --xlog-bin target/release/xlog \
  --souffle-bin "$(command -v souffle)" \
  --nvcc-bin /usr/local/cuda/bin/nvcc \
  --souffle-jobs "$(nproc)" \
  --memory-mb 65536 \
  --output paper/artifacts/head-to-head/triangle_counting_vs_souffle.json
```

Pass the machine's real core count to `--souffle-jobs`; a constant either
starves the baseline or oversubscribes it. The runner generates the skewed graph
once per case, writes the identical relation to Arrow IPC and to Soufflé facts,
compiles one standalone Soufflé executable per case, and runs each arm three
times. One-time native builds are excluded from execution medians and each
Soufflé compile is recorded separately. No failed arm is replaced with a
different execution path.

MNIST needs the baseline's own wheel rather than a source build: `scallopy`
0.2.4 does not compile against a current Rust toolchain, and the published cp310
wheel is the supported way in. Install it into a Python 3.10 of its own and
point the runner at that interpreter with `--scallop-python`; the xlog arm stays
on the interpreter that has `pyxlog`. Do not pass `--data-dir` unless both arms
get the same root — its default is already the directory the Scallop harness
reads.
