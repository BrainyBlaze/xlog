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
| `triangle_counting_vs_souffle.json` | Deterministic: fused WCOJ triangle counting vs Soufflé | RTX 4090 | **false** — see note |
| `residency_ablation.json` | xlog-only: forced host round-trip (CRIT-2) | RTX 3090 | n/a (single-system) |
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
  transfer cost residency eliminates. 3 seeds.
- **Verify-overhead isolation** — `program.warmup_breakdown()` under
  `XLOG_WARMUP_PROFILE=1` splits the cold compile into D4-compile and on-GPU
  CDCL equivalence-verify.
