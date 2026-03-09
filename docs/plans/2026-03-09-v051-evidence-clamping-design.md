# v0.5.1 Evidence Clamping for Monte Carlo Inference

> **Date:** 2026-03-09
> **Status:** Approved
> **Scope:** `xlog-prob` MC engine, `mc_sample.cu` kernel, `pyxlog` Python API

---

## Summary

Replace rejection sampling with evidence clamping for Monte Carlo queries whose evidence
maps directly to root Bernoulli variables (probabilistic facts and positive
annotated-disjunction heads). Instead of sampling from the prior and discarding worlds
that violate evidence, force evidence variables to their observed values in the sampling
kernel. Every sample satisfies evidence by construction, eliminating wasted work for
rare-evidence queries.

This is not likelihood weighting: because all forceable evidence maps to root random
variables, clamping them and sampling the remaining variables from the prior produces
i.i.d. samples from P(· | E). All samples are equally weighted; no importance weights
or ESS are needed.

---

## Motivation

The current MC engine (rejection sampling) degrades for rare evidence. If P(E) = 0.001,
~1M samples are needed to obtain ~1000 effective evidence-satisfying worlds. Evidence
clamping eliminates this entirely: every sample counts, `evidence_samples == total_samples`.

---

## Section 1: Sampling Method Enum and Auto-Selection

### New type

```rust
// crates/xlog-prob/src/mc.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McSamplingMethod {
    Rejection,
    EvidenceClamping,
}
```

### Config

`McEvalConfig` gains a new field:

```rust
pub sampling_method: Option<McSamplingMethod>,
```

`None` = auto-select; `Some(...)` = force.

### Auto-selection rule

```
if cfg.sampling_method.is_some() {
    // user override
} else if evidence.is_empty() {
    Rejection
} else if all evidence is forceable to root Bernoulli variables {
    EvidenceClamping
} else {
    Rejection
}
```

If the user explicitly forces `EvidenceClamping` on unforceable evidence, return a
clear error.

### Result metadata

`sampling_method: McSamplingMethod` added to:

- `McResult`
- `McDeviceResult`
- PyO3 result objects (`EvalResult`, `McDeviceEvalResult`)

Python API: `program.evaluate(..., sampling_method=None)` accepts `"rejection"`,
`"evidence_clamping"`, or `None`.

---

## Section 2: Sampling Kernel + Evidence Compilation

### Kernel change

`mc_sample_bernoulli` gains two new inputs:

```cuda
__global__ void mc_sample_bernoulli(
    uint8_t* out,
    const float* probs,
    const uint8_t* force_mask,   // [num_vars]: 0 = sample, 1 = force
    const uint8_t* forced_value, // [num_vars]: value when forced
    uint32_t num_vars,
    uint32_t num_samples,
    uint64_t seed
)
```

Per-element logic:

```cuda
if (force_mask[var_idx]) {
    out[idx] = forced_value[var_idx];
} else {
    // existing: RNG -> threshold -> 0/1
}
```

`force_mask` and `forced_value` are always passed. For `Rejection` mode, both are
zero-filled.

### Evidence compilation

A new function separate from `compile_sampling_plan`:

```rust
fn compile_evidence_forcing(
    evidence: &[(GroundAtom, bool)],
    prob_fact_specs: &[ProbFactSpec],
    ad_specs: &[AdSpec],
) -> Result<EvidenceForcing>
```

`EvidenceForcing` contains:

- `force_mask: Vec<u8>` — one per Bernoulli variable
- `forced_value: Vec<u8>` — one per Bernoulli variable
- `forceable: bool` — whether all evidence was lowerable
- `reason: ForceabilityReason` — why evidence is/isn't forceable

```rust
enum ForceabilityReason {
    AllForceable,
    ContainsDerivedEvidence,
    ContainsDeterministicEvidence,
    ContainsNegativeAdHeadEvidence,
}
```

### AD chain forcing rules

For an annotated disjunction with m heads lowered to m-1 decision vars `d_0..d_{m-2}`:

- **`evidence(head_k, true)` where k < num_decision_vars:**
  Force `d_i = 0` for all `i < k`, force `d_k = 1`.

- **Last explicit head (no none branch):**
  Force all decision vars `d_0..d_{m-2}` to 0.

- **`evidence(head, false)` on any AD head:**
  Not forceable in v0.5.1. Auto-select → fallback to `Rejection`.
  Explicit `EvidenceClamping` → error.

### Provider change

`sample_bernoulli_matrix_device()` in `crates/xlog-cuda/src/provider.rs` accepts
device-side force arrays (pointers/slices). `mc.rs` owns allocation and H2D upload
of the arrays.

---

## Section 3: Estimator Under Evidence Clamping

When `sampling_method == EvidenceClamping`:

- **No weight kernel.** No `log_weights` array. No ESS.
- **`evidence_samples == total_samples`** — every sample satisfies evidence by construction.
- **Existing unweighted estimator** (`count(Q) / N`) is correct.
- **Existing Wilson interval** remains statistically valid (i.i.d. samples from P(· | E)).
- **Skip evidence checking** in the evaluation loop. No `d_evidence_count`,
  `d_evidence_ptrs`, `d_evidence_expected`, `mc_eval_query_evidence_truth`, or
  evidence branch in `mc_accumulate_counts`. Only accumulate query counts.

Result metadata:

- `sampling_method: "evidence_clamping"`
- `evidence_samples == total_samples`
- No `ess` or `ess_per_query` fields in v0.5.1

---

## Section 4: Files Changed

| File | Action |
|------|--------|
| `crates/xlog-prob/src/mc.rs` | `McSamplingMethod` enum, `EvidenceForcing` struct, `compile_evidence_forcing()`, `sampling_method` on `McEvalConfig`/`McResult`/`McDeviceResult`, auto-selection logic, clamped evaluation branch (skip evidence-count path) |
| `kernels/mc_sample.cu` | `force_mask` + `forced_value` parameters, clamping branch |
| `crates/xlog-cuda/src/provider.rs` | Launch API accepts device-side force arrays |
| `crates/pyxlog/src/lib.rs` | `sampling_method` parameter on `evaluate()`/`evaluate_device()`, result metadata packing, `evidence_count = total_samples` in clamped mode |
| `crates/xlog-prob/tests/mc.rs` | Core evidence clamping tests |
| `crates/xlog-prob/tests/gpu_mc_device_counts.rs` | Device-count semantics under clamped mode |

No other crates modified. No new kernel files. No lowerer/executor/exact-path changes.

---

## Section 5: Tests

| # | Test | Verifies |
|---|------|----------|
| 1 | `evidence_clamping_prob_fact_true_matches_exact` | Clamp prob fact `evidence(atom, true)`, P(Q\|E) matches exact inference |
| 2 | `evidence_clamping_prob_fact_false_matches_exact` | Clamp prob fact `evidence(atom, false)`, `forced_value=0` path |
| 3 | `evidence_clamping_ad_head_3way` | 3-head AD (or 2+none), force middle head, verify chain forcing + query estimate |
| 4 | `evidence_clamping_all_samples_count` | `evidence_samples == total_samples` under clamped mode |
| 5 | `evidence_clamping_derived_evidence_falls_back` | Derived-atom evidence → auto-selects `Rejection` |
| 6 | `evidence_clamping_negative_ad_falls_back` | `evidence(ad_head, false)` → auto-selects `Rejection` |
| 7 | `explicit_clamping_unforceable_evidence_errors` | User forces `EvidenceClamping` on derived evidence → error |
| 8 | `sampling_method_in_result_metadata` | Correct `sampling_method` for auto and explicit cases |
| 9 | `device_counts_clamped_correct` | GPU device-count path: correct query counts + `evidence_count == total_samples` |
| 10 | `rejection_unchanged` | Existing rejection behavior unchanged |

---

## Section 6: Non-goals

1. No general proposal distributions
2. No adaptive proposal tuning
3. No resampling
4. No automatic sample-budget adjustment
5. No per-world weight exposure
6. No exact-inference changes
7. No approximate-inference engine work
8. No API redesign beyond method selection + aggregate diagnostics
9. No neural-symbolic inference/training semantics changes

---

## Future path

The `McSamplingMethod` enum is designed for extension:

```rust
pub enum McSamplingMethod {
    Rejection,
    EvidenceClamping,
    // future: LikelihoodWeighting (for non-root evidence weighting)
}
```

True likelihood weighting would be needed if evidence conditioning were extended to
derived or deterministic atoms whose truth depends on sampled hidden structure. That is
a larger scope and not part of v0.5.1.
