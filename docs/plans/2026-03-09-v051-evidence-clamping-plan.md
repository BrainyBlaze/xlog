# v0.5.1 Evidence Clamping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace rejection sampling with evidence clamping for MC queries whose evidence maps directly to root Bernoulli variables, so every sample counts.

**Architecture:** Add `McSamplingMethod` enum and `compile_evidence_forcing()` to `mc.rs`. Modify `mc_sample_bernoulli` kernel to accept `force_mask`/`forced_value` arrays. Create a clamped evaluation branch in the GPU loop that skips all evidence-checking infrastructure. Expose method selection and result metadata through Python API.

**Tech Stack:** Rust (xlog-prob, xlog-cuda), CUDA C (kernels/mc_sample.cu), Python/PyO3 (pyxlog)

---

### Task 1: McSamplingMethod Enum + McEvalConfig Field

**Files:**
- Modify: `crates/xlog-prob/src/mc.rs:62-82` (McEvalConfig struct + Default impl)

**Step 1: Add the enum and config field**

Add after the `use` block (before `McEvalConfig`), around line 44:

```rust
/// Sampling method for Monte Carlo inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McSamplingMethod {
    /// Sample from prior, discard worlds where evidence is not satisfied.
    Rejection,
    /// Force evidence variables in the sampler; every sample counts.
    EvidenceClamping,
}
```

Add a new field to `McEvalConfig` (line 63):

```rust
#[derive(Debug, Clone)]
pub struct McEvalConfig {
    /// Number of Monte Carlo samples.
    pub samples: usize,
    /// RNG seed (deterministic).
    pub seed: u64,
    /// Two-sided confidence level in (0,1) (e.g., 0.95).
    pub confidence: f64,
    /// Maximum SCC iteration steps for non-monotone cycle detection.
    pub max_nonmonotone_iterations: usize,
    /// Sampling method override. None = auto-select based on evidence forceability.
    pub sampling_method: Option<McSamplingMethod>,
}
```

Update the `Default` impl to include `sampling_method: None`.

**Step 2: Add sampling_method to McResult and McDeviceResult**

Add `pub sampling_method: McSamplingMethod` to both `McResult` (line 96) and `McDeviceResult` (line 108).

**Step 3: Verify it compiles**

Run: `cargo check -p xlog-prob 2>&1 | head -30`

Expected: Compilation errors from existing code that constructs `McEvalConfig`, `McResult`, `McDeviceResult` without the new fields. That is expected — we fix these call sites in later tasks.

**Step 4: Fix existing call sites**

Update all existing `McEvalConfig { ... }` constructors to include `sampling_method: None`.
Update all existing `McResult { ... }` and `McDeviceResult { ... }` constructors to include `sampling_method: McSamplingMethod::Rejection` (temporary default — will be replaced by auto-selection logic in Task 3).

Call sites to fix:
- `mc.rs` — all test helpers and internal `McResult`/`McDeviceResult` construction (lines ~292, ~418, ~608)
- `crates/xlog-prob/tests/mc.rs` — all test `McEvalConfig` construction (lines ~33, ~64, ~112, ~143)
- `crates/xlog-prob/tests/gpu_mc_vs_cpu.rs` — `McEvalConfig` construction
- `crates/xlog-prob/tests/gpu_mc_device_counts.rs:38` — `McEvalConfig` construction
- `crates/pyxlog/src/lib.rs:686` and `lib.rs:732` — `McEvalConfig` construction

**Step 5: Verify it compiles**

Run: `cargo check -p xlog-prob && cargo check -p pyxlog`

Expected: Clean compilation.

**Step 6: Run existing tests**

Run: `cargo test -p xlog-prob --test mc --release 2>&1 | tail -10`

Expected: All existing MC tests pass (no behavior change yet).

**Step 7: Commit**

```bash
git add crates/xlog-prob/src/mc.rs crates/xlog-prob/tests/ crates/pyxlog/src/lib.rs
git commit -m "feat(mc): add McSamplingMethod enum and sampling_method field on config/results"
```

---

### Task 2: EvidenceForcing + compile_evidence_forcing()

**Files:**
- Modify: `crates/xlog-prob/src/mc.rs` (add types + function after `compile_sampling_plan`)
- Test: `crates/xlog-prob/tests/mc.rs` (unit tests for evidence forcing compilation)

**Step 1: Write the failing tests**

Add to `crates/xlog-prob/tests/mc.rs`:

```rust
#[test]
fn test_evidence_forcing_prob_fact_true() {
    let src = r#"
0.3::rain().
0.7::sprinkler().
evidence(rain(), true).
query(sprinkler()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let forcing = program.compile_evidence_forcing().unwrap();
    assert!(forcing.forceable);
    // rain is var 0, sprinkler is var 1
    assert_eq!(forcing.force_mask[0], 1);
    assert_eq!(forcing.forced_value[0], 1);
    assert_eq!(forcing.force_mask[1], 0);
}

#[test]
fn test_evidence_forcing_prob_fact_false() {
    let src = r#"
0.3::rain().
evidence(rain(), false).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let forcing = program.compile_evidence_forcing().unwrap();
    assert!(forcing.forceable);
    assert_eq!(forcing.force_mask[0], 1);
    assert_eq!(forcing.forced_value[0], 0);
}

#[test]
fn test_evidence_forcing_derived_atom_not_forceable() {
    let src = r#"
0.3::rain().
wet() :- rain().
evidence(wet(), true).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let forcing = program.compile_evidence_forcing().unwrap();
    assert!(!forcing.forceable);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p xlog-prob --test mc test_evidence_forcing --release 2>&1 | tail -10`

Expected: FAIL — `compile_evidence_forcing` does not exist yet.

**Step 3: Implement EvidenceForcing and compile_evidence_forcing**

Add after `compile_sampling_plan` (around line 2097 of `mc.rs`):

```rust
/// Why evidence may or may not be forceable to root Bernoulli variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForceabilityReason {
    AllForceable,
    ContainsDerivedEvidence,
    ContainsNegativeAdHeadEvidence,
    NoEvidence,
}

/// Compiled evidence forcing for the MC sampler.
#[derive(Debug, Clone)]
pub struct EvidenceForcing {
    pub force_mask: Vec<u8>,
    pub forced_value: Vec<u8>,
    pub forceable: bool,
    pub reason: ForceabilityReason,
}
```

Add a public method on `McProgram`:

```rust
impl McProgram {
    pub fn compile_evidence_forcing(&self) -> Result<EvidenceForcing> {
        let num_vars = self.bernoulli_probs.len();
        let mut force_mask = vec![0u8; num_vars];
        let mut forced_value = vec![0u8; num_vars];

        if self.evidence.is_empty() {
            return Ok(EvidenceForcing {
                force_mask,
                forced_value,
                forceable: false,
                reason: ForceabilityReason::NoEvidence,
            });
        }

        for (atom, expected) in &self.evidence {
            // Try to match against prob fact specs
            if let Some(spec) = self.prob_facts.iter().find(|s| &s.atom == atom) {
                if !*expected {
                    // evidence(prob_fact, false) — forceable (force var to 0)
                    force_mask[spec.var_idx] = 1;
                    forced_value[spec.var_idx] = 0;
                } else {
                    // evidence(prob_fact, true) — forceable (force var to 1)
                    force_mask[spec.var_idx] = 1;
                    forced_value[spec.var_idx] = 1;
                }
                continue;
            }

            // Try to match against AD choice atoms (positive evidence only)
            let mut found_ad = false;
            for ad in &self.annotated_disjunctions {
                if let Some(choice_idx) = ad.choices.iter().position(|c| c == atom) {
                    if !*expected {
                        // evidence(ad_head, false) — not forceable in v0.5.1
                        return Ok(EvidenceForcing {
                            force_mask: vec![0u8; num_vars],
                            forced_value: vec![0u8; num_vars],
                            forceable: false,
                            reason: ForceabilityReason::ContainsNegativeAdHeadEvidence,
                        });
                    }

                    let num_decision_vars = ad.decision_vars.len();
                    if choice_idx < num_decision_vars {
                        // Force d_i = 0 for all i < choice_idx, d_{choice_idx} = 1
                        for i in 0..choice_idx {
                            force_mask[ad.decision_vars[i]] = 1;
                            forced_value[ad.decision_vars[i]] = 0;
                        }
                        force_mask[ad.decision_vars[choice_idx]] = 1;
                        forced_value[ad.decision_vars[choice_idx]] = 1;
                    } else {
                        // Last head (no none branch): force all decision vars to 0
                        for &dv in &ad.decision_vars {
                            force_mask[dv] = 1;
                            forced_value[dv] = 0;
                        }
                    }
                    found_ad = true;
                    break;
                }
            }
            if found_ad {
                continue;
            }

            // Evidence atom not found in prob facts or AD choices → derived/deterministic
            return Ok(EvidenceForcing {
                force_mask: vec![0u8; num_vars],
                forced_value: vec![0u8; num_vars],
                forceable: false,
                reason: ForceabilityReason::ContainsDerivedEvidence,
            });
        }

        Ok(EvidenceForcing {
            force_mask,
            forced_value,
            forceable: true,
            reason: ForceabilityReason::AllForceable,
        })
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p xlog-prob --test mc test_evidence_forcing --release 2>&1 | tail -10`

Expected: All 3 new tests PASS.

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/mc.rs crates/xlog-prob/tests/mc.rs
git commit -m "feat(mc): add compile_evidence_forcing for root Bernoulli evidence"
```

---

### Task 3: AD Chain Forcing Tests

**Files:**
- Modify: `crates/xlog-prob/tests/mc.rs`

**Step 1: Write the AD chain forcing test**

```rust
#[test]
fn test_evidence_forcing_ad_3way_middle_head() {
    // 3 explicit heads with sum < 1.0, so there is an implicit none branch.
    // AD: 0.2::color(red); 0.3::color(blue); 0.4::color(green).
    // has_none = true (sum = 0.9 < 1.0)
    // decision_vars: [v0, v1, v2] (3 Bernoulli vars for 4-way including none)
    // evidence(color(blue), true) => force v0=0, v1=1
    let src = r#"
0.2::color(red); 0.3::color(blue); 0.4::color(green).
evidence(color(blue), true).
query(color(red)).
query(color(green)).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let forcing = program.compile_evidence_forcing().unwrap();
    assert!(forcing.forceable, "3-way AD positive evidence should be forceable");

    // v0 = P(red | remaining=1.0) = 0.2
    // v1 = P(blue | remaining=0.8) = 0.3/0.8 = 0.375
    // v2 = P(green | remaining=0.5) = 0.4/0.5 = 0.8
    // evidence(color(blue)) → choice_idx=1 → force v0=0, v1=1
    assert_eq!(forcing.force_mask[0], 1); // v0 forced
    assert_eq!(forcing.forced_value[0], 0); // v0 = 0 (not red)
    assert_eq!(forcing.force_mask[1], 1); // v1 forced
    assert_eq!(forcing.forced_value[1], 1); // v1 = 1 (blue selected)
    assert_eq!(forcing.force_mask[2], 0); // v2 not forced (irrelevant after v1=1)
}

#[test]
fn test_evidence_forcing_ad_last_head_no_none() {
    // 2 heads summing to 1.0 → no none branch
    // AD: 0.4::coin(heads); 0.6::coin(tails).
    // decision_vars: [v0] (1 Bernoulli var for 2-way, no none)
    // evidence(coin(tails), true) => last head, no none → force v0=0
    let src = r#"
0.4::coin(heads); 0.6::coin(tails).
evidence(coin(tails), true).
query(coin(heads)).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let forcing = program.compile_evidence_forcing().unwrap();
    assert!(forcing.forceable);

    assert_eq!(forcing.force_mask[0], 1);
    assert_eq!(forcing.forced_value[0], 0); // last head → all decision vars = 0
}
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p xlog-prob --test mc test_evidence_forcing_ad --release 2>&1 | tail -10`

Expected: Both PASS (implementation from Task 2 already handles these cases).

**Step 3: Commit**

```bash
git add crates/xlog-prob/tests/mc.rs
git commit -m "test(mc): verify AD chain forcing for 3-way and last-head-no-none cases"
```

---

### Task 4: CUDA Kernel — force_mask/forced_value

**Files:**
- Modify: `kernels/mc_sample.cu`
- Modify: `crates/xlog-cuda/src/provider/mod.rs:1759-1821` (sample_bernoulli_matrix_device)

**Step 1: Update the CUDA kernel**

Replace the existing `mc_sample_bernoulli` kernel in `kernels/mc_sample.cu`:

```cuda
extern "C" __global__ void mc_sample_bernoulli(
    uint8_t* __restrict__ out,
    const float* __restrict__ probs,
    const uint8_t* __restrict__ force_mask,
    const uint8_t* __restrict__ forced_value,
    uint32_t num_vars,
    uint32_t num_samples,
    uint64_t seed
) {
    const uint64_t tid = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    const uint64_t total = (uint64_t)num_vars * (uint64_t)num_samples;
    if (tid >= total) {
        return;
    }

    const uint32_t var_idx = (uint32_t)(tid % (uint64_t)num_vars);

    if (force_mask[var_idx]) {
        out[tid] = forced_value[var_idx];
        return;
    }

    const float p = probs[var_idx];

    // Clamp p defensively; callers validate probabilities but this avoids NaNs
    // propagating into unpredictable behavior.
    const float p_clamped = (p <= 0.0f) ? 0.0f : ((p >= 1.0f) ? 1.0f : p);

    // Generate a 32-bit uniform from SplitMix64, then convert to (0,1).
    const uint64_t x = splitmix64(seed ^ (tid * 0x9e3779b97f4a7c15ULL));
    const uint32_t r = (uint32_t)(x >> 32);
    const float u = ((float)r + 0.5f) * (1.0f / 4294967296.0f); // 2^-32

    out[tid] = (u < p_clamped) ? 1u : 0u;
}
```

**Step 2: Rebuild PTX**

Run: `cd kernels && cmake --build build --target mc_sample 2>&1 | tail -5`

If no CMake build dir exists:

Run: `cd kernels && mkdir -p build && cd build && cmake .. && cmake --build . --target mc_sample 2>&1 | tail -5`

Expected: PTX builds successfully.

**Step 3: Update provider launch**

In `crates/xlog-cuda/src/provider/mod.rs`, update `sample_bernoulli_matrix_device` to accept device-side force arrays:

Change the method signature from:

```rust
pub fn sample_bernoulli_matrix_device(
    &self,
    probs: &[f32],
    num_samples: usize,
    seed: u64,
) -> Result<TrackedCudaSlice<u8>> {
```

To:

```rust
pub fn sample_bernoulli_matrix_device(
    &self,
    probs: &[f32],
    num_samples: usize,
    seed: u64,
    force_mask: &CudaView<u8>,
    forced_value: &CudaView<u8>,
) -> Result<TrackedCudaSlice<u8>> {
```

Update the kernel launch call (around line 1814) from:

```rust
(&mut d_out, &d_probs, num_vars_u32, num_samples_u32, seed),
```

To:

```rust
(&mut d_out, &d_probs, force_mask, forced_value, num_vars_u32, num_samples_u32, seed),
```

Also update the older host-returning `sample_bernoulli_matrix` if it exists (around line 1741) with the same signature change.

**Step 4: Fix all call sites of sample_bernoulli_matrix_device**

In `crates/xlog-prob/src/mc.rs`, the call at line ~735 inside `evaluate_gpu_counts_with`:

```rust
provider.sample_bernoulli_matrix_device(&self.bernoulli_probs, cfg.samples, cfg.seed)?
```

This will need to allocate and upload zero-filled force arrays when using rejection mode. We handle this properly in Task 5. For now, make it compile by creating zero-filled device arrays at the call site:

```rust
let num_vars = self.bernoulli_probs.len();
let d_force_mask = provider.memory().alloc::<u8>(num_vars)?;
provider.device().inner().memset_zeros(&mut d_force_mask)?;
let d_forced_value = provider.memory().alloc::<u8>(num_vars)?;
provider.device().inner().memset_zeros(&mut d_forced_value)?;

let samples_device = provider.sample_bernoulli_matrix_device(
    &self.bernoulli_probs,
    cfg.samples,
    cfg.seed,
    &d_force_mask.slice(..),
    &d_forced_value.slice(..),
)?;
```

Fix any other call sites similarly (the CPU path in `evaluate_cpu` around line 335).

**Step 5: Verify it compiles and existing tests pass**

Run: `cargo test -p xlog-prob --test mc --release 2>&1 | tail -10`

Expected: All tests pass (zero-filled force arrays = identical behavior to old kernel).

**Step 6: Commit**

```bash
git add kernels/mc_sample.cu kernels/build/ crates/xlog-cuda/src/provider/mod.rs crates/xlog-prob/src/mc.rs
git commit -m "feat(mc): add force_mask/forced_value to mc_sample_bernoulli kernel"
```

---

### Task 5: Evidence Clamping Evaluation Branch

**Files:**
- Modify: `crates/xlog-prob/src/mc.rs` (auto-selection + clamped evaluation path)

**Step 1: Write the failing test**

Add to `crates/xlog-prob/tests/mc.rs`:

```rust
#[test]
fn test_evidence_clamping_prob_fact_true_matches_exact() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    // P(rain | sprinkler=true) with independent facts.
    // P(sprinkler) = 0.2, P(rain) = 0.7.
    // Since rain and sprinkler are independent root facts,
    // P(rain | sprinkler=true) = P(rain) = 0.7.
    let src = r#"
0.7::rain().
0.2::sprinkler().
evidence(sprinkler(), true).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 50_000,
        seed: 42,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None, // auto-select
    };
    let result = program.evaluate(cfg).unwrap();

    assert_eq!(result.sampling_method, McSamplingMethod::EvidenceClamping);
    assert_eq!(result.evidence_samples, result.total_samples);
    let p = prob_of_atom(&result, "rain");
    assert!((p - 0.7).abs() < 0.02, "p={}", p);
}

#[test]
fn test_evidence_clamping_prob_fact_false_matches_exact() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    // P(rain | sprinkler=false): since independent, P(rain) = 0.7
    let src = r#"
0.7::rain().
0.2::sprinkler().
evidence(sprinkler(), false).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 50_000,
        seed: 42,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };
    let result = program.evaluate(cfg).unwrap();

    assert_eq!(result.sampling_method, McSamplingMethod::EvidenceClamping);
    assert_eq!(result.evidence_samples, result.total_samples);
    let p = prob_of_atom(&result, "rain");
    assert!((p - 0.7).abs() < 0.02, "p={}", p);
}

#[test]
fn test_evidence_clamping_all_samples_count() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let src = r#"
0.01::rare().
0.5::other().
evidence(rare(), true).
query(other()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 1000,
        seed: 7,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };
    let result = program.evaluate(cfg).unwrap();

    assert_eq!(result.sampling_method, McSamplingMethod::EvidenceClamping);
    assert_eq!(result.evidence_samples, 1000);
    let p = prob_of_atom(&result, "other");
    assert!((p - 0.5).abs() < 0.05, "p={}", p);
}
```

**Step 2: Run to verify they fail**

Run: `cargo test -p xlog-prob --test mc test_evidence_clamping --release 2>&1 | tail -10`

Expected: FAIL — auto-selection logic does not exist yet; all evaluations use Rejection.

**Step 3: Implement auto-selection and clamped evaluation branch**

In `evaluate_gpu_device_with_provider` (around line 440 of `mc.rs`):

1. After validation, call `compile_evidence_forcing()` and resolve the method:

```rust
let forcing = self.compile_evidence_forcing()?;
let method = match cfg.sampling_method {
    Some(McSamplingMethod::EvidenceClamping) => {
        if !forcing.forceable {
            return Err(XlogError::Execution(format!(
                "Cannot use EvidenceClamping: {:?}",
                forcing.reason
            )));
        }
        McSamplingMethod::EvidenceClamping
    }
    Some(McSamplingMethod::Rejection) => McSamplingMethod::Rejection,
    None => {
        if forcing.forceable {
            McSamplingMethod::EvidenceClamping
        } else {
            McSamplingMethod::Rejection
        }
    }
};
```

2. Upload force arrays before sampling:

```rust
let num_vars = self.bernoulli_probs.len();
let (d_force_mask, d_forced_value) = if method == McSamplingMethod::EvidenceClamping {
    let mut fm = provider.memory().alloc::<u8>(num_vars)?;
    provider.device().inner().htod_sync_copy_into(&forcing.force_mask, &mut fm)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload force_mask: {}", e)))?;
    let mut fv = provider.memory().alloc::<u8>(num_vars)?;
    provider.device().inner().htod_sync_copy_into(&forcing.forced_value, &mut fv)
        .map_err(|e| XlogError::Kernel(format!("Failed to upload forced_value: {}", e)))?;
    (fm, fv)
} else {
    let mut fm = provider.memory().alloc::<u8>(num_vars)?;
    if num_vars > 0 {
        provider.device().inner().memset_zeros(&mut fm)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero force_mask: {}", e)))?;
    }
    let mut fv = provider.memory().alloc::<u8>(num_vars)?;
    if num_vars > 0 {
        provider.device().inner().memset_zeros(&mut fv)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero forced_value: {}", e)))?;
    }
    (fm, fv)
};
```

3. Pass force arrays to `sample_bernoulli_matrix_device`.

4. For the clamped branch, create a simplified `on_sample` callback that only accumulates query counts (no evidence truth check, no evidence accumulation). Construct a simpler kernel dispatch that only calls a query-count-only accumulator, or conditionally skip the evidence portions of the existing callback.

The simplest approach: in the existing `on_sample` closure inside `evaluate_gpu_device_with_provider`, conditionally skip the evidence infrastructure when `method == EvidenceClamping`:

- Skip `d_evidence_ptrs` upload
- Skip `mc_eval_query_evidence_truth` kernel (or launch with `evidence_count_u32 = 0`)
- Skip `mc_accumulate_counts` evidence branch (launch with `d_evidence_ok` forced to 1)
- Or simplest: just set `evidence_count_u32 = 0` and pre-set `d_evidence_ok` to 1 before the loop. The existing kernels will then treat every sample as evidence-satisfied.

After the evaluation loop, set `evidence_count` to `total_samples` in the result.

5. Include `method` in the `McDeviceResult` and `McResult` return values.

6. Apply the same auto-selection logic in `evaluate_cpu` for consistency.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p xlog-prob --test mc --release 2>&1 | tail -20`

Expected: All tests PASS, including old and new.

**Step 5: Commit**

```bash
git add crates/xlog-prob/src/mc.rs crates/xlog-prob/tests/mc.rs
git commit -m "feat(mc): implement evidence clamping evaluation branch with auto-selection"
```

---

### Task 6: Fallback and Error Tests

**Files:**
- Modify: `crates/xlog-prob/tests/mc.rs`

**Step 1: Write fallback and error tests**

```rust
#[test]
fn test_evidence_clamping_derived_evidence_falls_back() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let src = r#"
0.3::rain().
wet() :- rain().
evidence(wet(), true).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 50_000,
        seed: 7,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None, // auto-select → should fall back to Rejection
    };
    let result = program.evaluate(cfg).unwrap();
    assert_eq!(result.sampling_method, McSamplingMethod::Rejection);
    // P(rain | wet) = P(rain) / P(wet) = 0.3 / 0.3 = 1.0
    let p = prob_of_atom(&result, "rain");
    assert!((p - 1.0).abs() < 0.01, "p={}", p);
}

#[test]
fn test_evidence_clamping_negative_ad_falls_back() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let src = r#"
0.3::coin(1); 0.3::coin(2).
evidence(coin(1), false).
query(coin(2)).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 50_000,
        seed: 2026,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };
    let result = program.evaluate(cfg).unwrap();
    assert_eq!(result.sampling_method, McSamplingMethod::Rejection);
}

#[test]
fn test_explicit_clamping_unforceable_evidence_errors() {
    let src = r#"
0.3::rain().
wet() :- rain().
evidence(wet(), true).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 1000,
        seed: 7,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: Some(McSamplingMethod::EvidenceClamping),
    };
    let result = program.evaluate(cfg);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("EvidenceClamping") || err.contains("forceable"),
        "Error should mention clamping: {}",
        err
    );
}

#[test]
fn test_sampling_method_in_result_metadata() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    // No evidence → Rejection
    let src_no_ev = r#"
0.5::a().
query(a()).
"#;
    let prog = McProgram::compile_source(src_no_ev).unwrap();
    let cfg = McEvalConfig {
        samples: 100,
        seed: 0,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };
    let result = prog.evaluate(cfg).unwrap();
    assert_eq!(result.sampling_method, McSamplingMethod::Rejection);

    // Root evidence → EvidenceClamping
    let src_ev = r#"
0.5::a().
0.3::b().
evidence(a(), true).
query(b()).
"#;
    let prog2 = McProgram::compile_source(src_ev).unwrap();
    let cfg2 = McEvalConfig {
        samples: 100,
        seed: 0,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };
    let result2 = prog2.evaluate(cfg2).unwrap();
    assert_eq!(result2.sampling_method, McSamplingMethod::EvidenceClamping);
}

#[test]
fn test_rejection_unchanged() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    // Explicit Rejection with root evidence — should still work (old behavior)
    let src = r#"
0.7::rain().
0.2::sprinkler().
evidence(sprinkler(), true).
query(rain()).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 50_000,
        seed: 7,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: Some(McSamplingMethod::Rejection),
    };
    let result = program.evaluate(cfg).unwrap();
    assert_eq!(result.sampling_method, McSamplingMethod::Rejection);
    let p = prob_of_atom(&result, "rain");
    assert!((p - 0.7).abs() < 0.02, "p={}", p);
    // Under rejection, evidence_samples < total_samples (sprinkler satisfied in ~20% of worlds)
    assert!(result.evidence_samples < result.total_samples);
}
```

**Step 2: Run tests**

Run: `cargo test -p xlog-prob --test mc --release 2>&1 | tail -20`

Expected: All PASS.

**Step 3: Commit**

```bash
git add crates/xlog-prob/tests/mc.rs
git commit -m "test(mc): add fallback, error, metadata, and regression tests for evidence clamping"
```

---

### Task 7: AD Evidence Clamping End-to-End

**Files:**
- Modify: `crates/xlog-prob/tests/mc.rs`

**Step 1: Write end-to-end AD clamping test**

```rust
#[test]
fn test_evidence_clamping_ad_head_3way() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    // 3-head AD + none: 0.2::color(red); 0.3::color(blue); 0.4::color(green).
    // evidence(color(blue), true) → clamp, every sample has color(blue)=true
    // P(color(red) | color(blue)) = 0  (AD is exclusive)
    // P(color(green) | color(blue)) = 0
    let src = r#"
0.2::color(red); 0.3::color(blue); 0.4::color(green).
evidence(color(blue), true).
query(color(red)).
query(color(green)).
"#;
    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 10_000,
        seed: 42,
        confidence: 0.95,
        max_nonmonotone_iterations: 128,
        sampling_method: None,
    };
    let result = program.evaluate(cfg).unwrap();

    assert_eq!(result.sampling_method, McSamplingMethod::EvidenceClamping);
    assert_eq!(result.evidence_samples, result.total_samples);

    // Under clamped evidence, color(blue) is always true, others always false
    let p_red = result.query_estimates.iter()
        .find(|q| q.atom.predicate == "color" && q.atom.args.len() == 1
            && q.atom.args[0] == xlog_prob::provenance::Value::Symbol("red".to_string()))
        .unwrap().prob;
    let p_green = result.query_estimates.iter()
        .find(|q| q.atom.predicate == "color" && q.atom.args.len() == 1
            && q.atom.args[0] == xlog_prob::provenance::Value::Symbol("green".to_string()))
        .unwrap().prob;

    assert_eq!(p_red, 0.0);
    assert_eq!(p_green, 0.0);
}
```

**Step 2: Run test**

Run: `cargo test -p xlog-prob --test mc test_evidence_clamping_ad_head_3way --release 2>&1 | tail -10`

Expected: PASS.

**Step 3: Commit**

```bash
git add crates/xlog-prob/tests/mc.rs
git commit -m "test(mc): end-to-end evidence clamping with 3-way AD chain"
```

---

### Task 8: Device Counts Under Clamped Mode

**Files:**
- Modify: `crates/xlog-prob/tests/gpu_mc_device_counts.rs`

**Step 1: Write device-counts clamped test**

```rust
#[test]
fn test_device_counts_clamped_correct() -> Result<()> {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let program = McProgram::compile_source(
        r#"
        0.5::a().
        0.3::b().
        evidence(a(), true).
        query(b()).
    "#,
    )?;

    let cfg = McEvalConfig {
        samples: 100,
        seed: 42,
        confidence: 0.95,
        max_nonmonotone_iterations: 10,
        sampling_method: None,
    };

    let gpu = program.evaluate_gpu_device(cfg.clone())?;
    assert_eq!(gpu.sampling_method, xlog_prob::mc::McSamplingMethod::EvidenceClamping);

    // evidence_count should equal total_samples under clamped mode
    let mut host_evidence = [0u32];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&gpu.evidence_count, &mut host_evidence)
        .unwrap();
    assert_eq!(host_evidence[0] as usize, 100);

    // query counts should be reasonable for b() ~ 0.3
    let mut host_counts = vec![0u32; gpu.query_counts.len()];
    provider
        .device()
        .inner()
        .dtoh_sync_copy_into(&gpu.query_counts, &mut host_counts)
        .unwrap();
    let p_b = host_counts[0] as f64 / 100.0;
    assert!((p_b - 0.3).abs() < 0.15, "p_b={}", p_b); // wide tolerance for N=100

    Ok(())
}
```

**Step 2: Run test**

Run: `cargo test -p xlog-prob --test gpu_mc_device_counts test_device_counts_clamped --release 2>&1 | tail -10`

Expected: PASS.

**Step 3: Commit**

```bash
git add crates/xlog-prob/tests/gpu_mc_device_counts.rs
git commit -m "test(mc): verify device counts under evidence clamping mode"
```

---

### Task 9: Python API — sampling_method Input + Result Metadata

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:642-812` (evaluate + evaluate_device)
- Modify: `crates/pyxlog/src/lib.rs:3685-3705` (McDeviceEvalResult struct)
- Modify: `crates/pyxlog/src/lib.rs:3447-3510` (pack_result_mc)
- Modify: `crates/pyxlog/src/lib.rs:3709-3739` (EvalResult struct)

**Step 1: Add sampling_method to Python API signatures**

Update `evaluate` signature (line 642):

```rust
#[pyo3(signature = (return_grads=false, samples=None, seed=None, confidence=0.95, max_nonmonotone_iterations=1024, sampling_method=None))]
pub fn evaluate(
    &self,
    _py: Python<'_>,
    return_grads: bool,
    samples: Option<usize>,
    seed: Option<u64>,
    confidence: f64,
    max_nonmonotone_iterations: usize,
    sampling_method: Option<String>,
) -> PyResult<EvalResult> {
```

Update `evaluate_device` signature (line 712):

```rust
#[pyo3(signature = (samples=None, seed=None, confidence=0.95, max_nonmonotone_iterations=1024, sampling_method=None))]
pub fn evaluate_device(
    &self,
    py: Python<'_>,
    samples: Option<usize>,
    seed: Option<u64>,
    confidence: f64,
    max_nonmonotone_iterations: usize,
    sampling_method: Option<String>,
) -> PyResult<McDeviceEvalResult> {
```

**Step 2: Parse sampling_method string**

Add a helper at the top of the `impl CompiledProgram` block:

```rust
fn parse_sampling_method(s: Option<String>) -> PyResult<Option<McSamplingMethod>> {
    match s.as_deref() {
        None => Ok(None),
        Some("rejection") => Ok(Some(McSamplingMethod::Rejection)),
        Some("evidence_clamping") => Ok(Some(McSamplingMethod::EvidenceClamping)),
        Some(other) => Err(PyValueError::new_err(format!(
            "Unknown sampling_method '{}'. Use 'rejection' or 'evidence_clamping'.",
            other
        ))),
    }
}
```

Use it in both `evaluate` and `evaluate_device` when building `McEvalConfig`:

```rust
let cfg = McEvalConfig {
    samples: samples.unwrap_or(10000),
    seed: seed.unwrap_or(0),
    confidence,
    max_nonmonotone_iterations,
    sampling_method: Self::parse_sampling_method(sampling_method)?,
};
```

**Step 3: Add sampling_method to result structs**

Add `pub sampling_method: String` to `McDeviceEvalResult` (line 3685) and `pub sampling_method: Option<String>` to `EvalResult` (line 3709).

Populate in `evaluate_device` result construction:

```rust
sampling_method: match gpu.sampling_method {
    McSamplingMethod::Rejection => "rejection".to_string(),
    McSamplingMethod::EvidenceClamping => "evidence_clamping".to_string(),
},
```

Populate in `pack_result_mc`:

```rust
sampling_method: Some(match result.sampling_method {
    McSamplingMethod::Rejection => "rejection".to_string(),
    McSamplingMethod::EvidenceClamping => "evidence_clamping".to_string(),
}),
```

For exact inference results, set `sampling_method: None`.

**Step 4: Import McSamplingMethod**

Update the import at the top of `lib.rs` (line 24):

```rust
use xlog_prob::mc::{McEvalConfig, McProgram, McSamplingMethod};
```

**Step 5: Verify compilation**

Run: `cargo check -p pyxlog 2>&1 | tail -10`

Expected: Clean compilation.

**Step 6: Commit**

```bash
git add crates/pyxlog/src/lib.rs
git commit -m "feat(pyxlog): expose sampling_method on evaluate/evaluate_device API"
```

---

### Task 10: Final Verification + Documentation

**Files:**
- Verify: all tests pass
- Modify: `docs/ROADMAP.md` — mark importance sampling as implemented (evidence clamping)
- Modify: `CHANGELOG.md` — add evidence clamping under [Unreleased]

**Step 1: Run full test suite**

Run: `cargo test -p xlog-prob --release 2>&1 | tail -20`

Expected: All tests pass.

Run: `cargo test -p pyxlog --release 2>&1 | tail -10`

Expected: Clean (or skip if no CUDA).

**Step 2: Update CHANGELOG.md**

Add under `[Unreleased] -> Added`:

```markdown
- **Evidence clamping for MC inference** (`xlog-prob`): Monte Carlo evidence conditioning
  via `McSamplingMethod::EvidenceClamping`. Forces root Bernoulli evidence variables in the
  sampling kernel so every sample counts (`evidence_samples == total_samples`). Auto-selected
  when all evidence maps to probabilistic facts or positive AD heads; falls back to rejection
  for derived/deterministic/negative-AD evidence. New `sampling_method` field on `McEvalConfig`,
  `McResult`, `McDeviceResult`, and Python API. CUDA kernel updated with `force_mask`/`forced_value`
  inputs.
```

**Step 3: Update ROADMAP.md**

Under "Probabilistic Reasoning > Planned", change:

```markdown
- [ ] Importance sampling for rare-event queries
```

To:

```markdown
- [x] ~~Importance sampling for rare-event queries~~ (done: evidence clamping for forceable root evidence, `McSamplingMethod::EvidenceClamping`)
```

**Step 4: Commit**

```bash
git add CHANGELOG.md docs/ROADMAP.md
git commit -m "docs: add evidence clamping to changelog and roadmap"
```
