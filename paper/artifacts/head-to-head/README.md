# Head-to-head and overhead-isolation artifacts

Benchmark artifacts backing the head-to-head and overhead-isolation claims in
`sections/10_evaluation.tex`. These were collected on **ephemeral cloud GPUs**
(RunPod), separate from the single-system ablations in the rest of the
Evaluation section, which run on the development RTX PRO 3000. Each artifact
records its own hardware, protocol, and per-cell measurements; hardware is not
mixed within a comparison.

| File | Comparison | Hardware | `comparison_acceptable` |
|------|-----------|----------|--------------------------|
| `mnist_addition_vs_scallop.json` | Neural: MNIST addition, xlog vs Scallop | RTX 3090 | **true** |
| `exact_inference_vs_problog2.json` | Probabilistic: exact inference, xlog vs ProbLog2 | RTX 4090 | **true** |
| `triangle_counting_vs_souffle.json` | Deterministic: fused WCOJ triangle counting vs Soufflé (skewed) | A40 pod; 7.65-core CPU quota | **true** |
| `triangle_counting_moderate_skew_vs_souffle.json` | Deterministic: WCOJ vs binary vs Soufflé, moderate skew | RTX 4090 | **true** |
| `residency_ablation.json` | xlog-only: forced host round-trip, single query | RTX 3090 | n/a (single-system) |
| `residency_scale_ablation.json` | xlog-only: forced host round-trip vs handoff count, batched | RTX 3090 | n/a (single-system) |
| `verify_overhead_isolation.json` | xlog-only: CDCL-verify vs D4-compile split | RTX 3090 | n/a (single-system) |

## Protocol notes

- **MNIST vs Scallop** — identical MNISTNet / data / metric / seeds. Two
  protocols: the whitepaper 512-image/5-epoch setting (both near-chance,
  under-trained) and a stronger 20k-image/5-epoch setting (both ~95%). Held-out
  addition accuracy is measured on the 10k MNIST test set. 3 seeds at 20k.
- **Exact inference vs ProbLog2** — 5 programs; correctness gate: query
  probabilities match the analytic answer within 1e-4 (both engines reach 0
  error). Timing is full inference (compile + evaluate), median of 3.
- **Triangle counting vs Soufflé** — fused-WCOJ count (A), enumerate-then-count
  (B), and Soufflé count (C) on hub-skewed graphs. All three arms complete and
  produce identical per-root counts at every size, so both
  `core_comparison_acceptable` and `comparison_acceptable` are `true`. The
  enumerate arm peaks at 3,287 MB, 8,403 MB and 15,247 MB of provider
  allocations, while fused counting uses 85 MB, 204 MB and 359 MB.
  Soufflé-to-fused-XLOG execution-time ratios are 2.37x, 7.95x and 9.00x on
  these three heavy-skew cases; the separate moderate-skew artifact remains the
  lower-bound companion and does not support a universal Datalog speed claim.
- **Residency ablation** — same pipeline with vs without
  `XLOG_FORCE_HOST_ROUNDTRIP`; the on-minus-off per-iteration delta is the
  transfer cost residency eliminates. The single-query file measures 2 handoffs
  (near-noise). The `_scale` file sweeps the batched path (2--512 handoffs per
  step) and is the one to cite: per-handoff round-trip is ~40--56 us; the
  round-trip's share of a step is within noise below ~16 handoffs
  (queries=4 even measures -8.9%), reaches ~10% at the standard batch-64
  MNIST step (7.2 ms of 72 ms), and measures 8.6% at queries=256 — the
  batch-64 point is the peak share in this sweep, not a monotone climb. `runners/residency_sweep.py`
  is the single-query runner; the scale runner is `runners/residency_scale.py`.
- **Verify-overhead isolation** — `program.warmup_breakdown()` under
  `XLOG_WARMUP_PROFILE=1` splits the cold compile into D4-compile and on-GPU
  CDCL equivalence-verify.
- **Triangle counting, moderate skew** — the companion to the skewed run: on
  moderate hub skew the GPU binary join does not blow up, so fused WCOJ is only
  `1.1`--`1.5x` over xlog's own binary join (both correct vs Soufflé). This is
  the honest lower end of the WCOJ range; the memory win appears only under
  heavy skew.

## Reproduction

`runners/residency_sweep.py` and `runners/verify_sweep.py` are the exact
scripts for the two xlog-only isolations (run with `python -u` on a
CUDA-enabled `pyxlog` build; the residency script needs `torch`). Every
head-to-head comparison now has a committed runner:
`runners/triangle_counting_vs_souffle.py`,
`runners/triangle_counting_moderate_skew_vs_souffle.py`,
`runners/exact_inference_vs_problog2.py` and
`runners/mnist_addition_vs_scallop.py`.

The published numbers in the last three files were **not** produced by those
runners — they came from ephemeral on-pod scripts that were never committed, so
the runners reconstruct the protocol from the artifact and from the committed
harnesses rather than reproduce the original invocation. Each one records what
it had to choose, and every one of them refuses a dirty checkout and carries
`--self-test`. Two consequences worth stating plainly:

- the moderate-skew generator parameters are **not** recorded in the published
  artifact — only the edge counts — so its runner writes the hub fraction it
  used, and a re-run is a new measurement of the same class rather than a
  repeat of the old one;
- the MNIST seeds are likewise unrecorded; the runner defaults to the
  repository's `DEFAULT_SEEDS`, which are the seeds the committed Scallop
  baseline results were produced with.

- **Exact vs ProbLog2** — `pyxlog.Program.compile(src)` on the five programs
  (a conditioned wet/sprinkler net and `reach_chain_{5,10,15,20}`), timed
  end-to-end (compile+evaluate), vs `problog` on the matched programs; gate:
  probabilities within `1e-4` of the analytic answer.
- **MNIST addition vs Scallop** — identical MNISTNet, batch 64, lr 1e-3;
  `pyxlog` neural predicate `nn(net,[X],Y,[0..9])::digit(X,Y)` +
  `addition(A,B,S):-digit(A,X),digit(B,Y),S is X+Y` vs Scallop
  `difftopbottomkclauses` (k=3); held-out addition accuracy on the 10k MNIST
  test. `runners/mnist_addition_vs_scallop.py` drives both committed harnesses
  — `examples/neural/01_minimal/train.py` and
  `examples/neural/baseline/scallop/mnist_addition.py` — once per (protocol,
  seed, side) and aggregates. It records `epoch_timing_source`: when the
  training history carries no per-epoch times, `train.py` divides the total
  evenly, and the first-epoch / steady-epoch split is then an approximation
  rather than a measurement.
- **Triangle counting vs Soufflé** — build the release CLI, install PyArrow
  and Soufflé, then run:

  ```bash
  cargo build --release --locked -p xlog-cli
  python -u paper/artifacts/head-to-head/runners/triangle_counting_vs_souffle.py \
    --xlog-bin target/release/xlog \
    --souffle-bin souffle \
    --nvcc-bin /usr/local/cuda/bin/nvcc \
    --souffle-jobs 8 \
    --output paper/artifacts/head-to-head/triangle_counting_vs_souffle.json
  ```

  The runner refuses a dirty checkout by default. It generates the skewed graph
  once per case, writes the identical relation to Arrow IPC and Soufflé facts,
  compiles one standalone Soufflé executable per case, and then runs fused WCOJ,
  enumerate-then-count, and that executable three times. One-time native builds
  are excluded from execution medians; each Soufflé compile is recorded
  separately. The published run used eight Soufflé jobs on a pod whose cgroup
  quota was 7.65 CPU cores. The artifact also records the exact commit, runner
  and input hashes, normalized commands, software and hardware versions, process
  RSS, XLOG's provider allocation high-water, WCOJ dispatch counters,
  per-repetition failures, and separate fused/Soufflé and all-arm correctness
  gates. No failed arm is replaced with a different execution path.
