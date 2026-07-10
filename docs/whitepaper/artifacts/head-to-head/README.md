# Head-to-head and overhead-isolation artifacts

Benchmark artifacts backing the head-to-head and overhead-isolation claims in
`sections/08_evaluation.tex`. These were collected on **ephemeral cloud GPUs**
(RunPod), separate from the single-system ablations in the rest of the
Evaluation section, which run on the development RTX PRO 3000. Each artifact
records its own hardware, protocol, and per-cell measurements; hardware is not
mixed within a comparison.

| File | Comparison | Hardware | `comparison_acceptable` |
|------|-----------|----------|--------------------------|
| `mnist_addition_vs_scallop.json` | Neural: MNIST addition, xlog vs Scallop | RTX 3090 | **true** |
| `exact_inference_vs_problog2.json` | Probabilistic: exact inference, xlog vs ProbLog2 | RTX 4090 | **true** |
| `triangle_counting_vs_souffle.json` | Deterministic: fused WCOJ triangle counting vs Soufflé (skewed) | RTX 4090 | **false** — see note |
| `triangle_counting_moderate_skew_vs_souffle.json` | Deterministic: WCOJ vs binary vs Soufflé, moderate skew | RTX 4090 | **true** |
| `residency_ablation.json` | xlog-only: forced host round-trip, single query (CRIT-2) | RTX 3090 | n/a (single-system) |
| `residency_scale_ablation.json` | xlog-only: forced host round-trip vs handoff count, batched (CRIT-2) | RTX 3090 | n/a (single-system) |
| `verify_overhead_isolation.json` | xlog-only: CDCL-verify vs D4-compile split (EIC W5) | RTX 3090 | n/a (single-system) |

## Protocol notes

- **MNIST vs Scallop** — identical MNISTNet / data / metric / seeds. Two
  protocols: the whitepaper 512-image/5-epoch setting (both near-chance,
  under-trained) and a stronger 20k-image/5-epoch setting (both ~95%). Held-out
  addition accuracy is measured on the 10k MNIST test set. 3 seeds at 20k.
- **Exact inference vs ProbLog2** — 5 programs; correctness gate: query
  probabilities match the analytic answer within 1e-4 (both engines reach 0
  error). Timing is full inference (compile + evaluate), median of 3.
- **Triangle counting vs Soufflé** — fused-WCOJ count (A), enumerate-then-count
  (B), and Soufflé count (C) on hub-skewed graphs. **`comparison_acceptable` is
  `false` on purpose**: the enumerate arm (B) exhausts the GPU memory budget at
  the 80-hub / 500k-edge case (3.2 GB alloc), so that one cell is incomplete.
  This OOM is itself the intended demonstration of the memory bound. The
  fused-vs-Soufflé (A vs C) correctness gate passes at *every* size (triangle
  counts match Soufflé exactly) and the fused wall-clock speedups (12.6–42.5x)
  are valid.
- **Residency ablation** — same pipeline with vs without
  `XLOG_FORCE_HOST_ROUNDTRIP`; the on-minus-off per-iteration delta is the
  transfer cost residency eliminates. The single-query file measures 2 handoffs
  (near-noise). The `_scale` file sweeps the batched path (2--512 handoffs per
  step) and is the one to cite: per-handoff round-trip is ~40--56 us, and the
  round-trip's share of a step rises with the handoff count to ~10% at the
  standard batch-64 MNIST step (7.2 ms of 72 ms). `runners/residency_sweep.py`
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
CUDA-enabled `pyxlog` build; the residency script needs `torch`). The
neural / probabilistic / deterministic head-to-head runs were orchestrated by
ephemeral on-pod scripts that are not committed; the programs and commands are
small enough to restate:

- **Exact vs ProbLog2** — `pyxlog.Program.compile(src)` on the five programs
  (a conditioned wet/sprinkler net and `reach_chain_{5,10,15,20}`), timed
  end-to-end (compile+evaluate), vs `problog` on the matched programs; gate:
  probabilities within `1e-4` of the analytic answer.
- **MNIST addition vs Scallop** — identical MNISTNet, batch 64, lr 1e-3;
  `pyxlog` neural predicate `nn(net,[X],Y,[0..9])::digit(X,Y)` +
  `addition(A,B,S):-digit(A,X),digit(B,Y),S is X+Y` vs Scallop
  `difftopbottomkclauses` (k=3); held-out addition accuracy on the 10k MNIST
  test. See `scripts/track_a_runner.py` for the closest committed harness.
- **Triangle counting vs Soufflé** — edges bulk-loaded as an Arrow-IPC EDB
  (`xlog run tri.xlog --input edge=edges.arrow --wcoj --stats`), rule
  `triangle(A,B,C):-edge(A,B),edge(B,C),edge(A,C)`, vs the same graph in
  Soufflé; gate: triangle count matches Soufflé exactly.
