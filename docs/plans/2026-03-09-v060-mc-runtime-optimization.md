# v0.6.0 MC Runtime Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce Monte Carlo evaluation runtime by removing avoidable host synchronization, per-sample store clone churn, and redundant count-path work without changing MC semantics.

**Architecture:** Keep the current MC semantics (`Rejection` and `EvidenceClamping`) unchanged, but optimize the execution pipeline in place. The plan first instruments the hot path, then removes unnecessary row-count/device syncs, replaces whole-store clone/restore with a targeted reset plan, stabilizes per-sample pointer infrastructure, and finally cleans up the small `EvidenceClamping` no-op overhead. All changes stay inside the MC/runtime/provider path; no exact inference, lowerer, or neural semantics changes.

**Tech Stack:** Rust (`xlog-prob`, `xlog-runtime`, `xlog-cuda`), CUDA C (`kernels/mc_sample.cu` only if needed for cleanup verification), Rust integration tests with `host-io`, PyO3 smoke verification through `pyxlog`

**Non-goals:**
- No new inference algorithms
- No approximate-inference engine work
- No exact-inference changes
- No neural-symbolic inference changes
- No result/API redesign beyond optional profiling/debug output

---

### Task 1: Add MC Hot-Path Timing Instrumentation

**Files:**
- Modify: `crates/xlog-prob/src/mc.rs`
- Test: `crates/xlog-prob/tests/mc.rs`

**Step 1: Write the failing test**

Add a small unit-style integration test to `crates/xlog-prob/tests/mc.rs` that validates the timing accumulator helper, not wall-clock values:

```rust
#[test]
fn test_mc_timing_breakdown_totals_sum() {
    let mut t = xlog_prob::mc::McTimingBreakdown::default();
    t.sampler_us = 10;
    t.sample_reset_us = 20;
    t.sample_build_us = 30;
    t.eval_us = 40;
    t.count_us = 50;
    assert_eq!(t.total_us(), 150);
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cargo test -p xlog-prob --release --features host-io --test mc test_mc_timing_breakdown_totals_sum -- --nocapture
```

Expected: FAIL because `McTimingBreakdown` does not exist.

**Step 3: Write minimal implementation**

In `crates/xlog-prob/src/mc.rs`, add:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct McTimingBreakdown {
    pub sampler_us: u64,
    pub sample_reset_us: u64,
    pub sample_build_us: u64,
    pub eval_us: u64,
    pub count_us: u64,
}

impl McTimingBreakdown {
    pub fn total_us(&self) -> u64 {
        self.sampler_us
            .saturating_add(self.sample_reset_us)
            .saturating_add(self.sample_build_us)
            .saturating_add(self.eval_us)
            .saturating_add(self.count_us)
    }
}
```

Add env-gated instrumentation around the current hot path:
- sampler launch
- per-sample reset
- `build_sample_buffers`
- `evaluate_program_gpu`
- count accumulation callback

Gate it with `XLOG_MC_PROFILE=1` and print a single summary line at the end of evaluation.

**Step 4: Run test to verify it passes**

Run:
```bash
cargo test -p xlog-prob --release --features host-io --test mc test_mc_timing_breakdown_totals_sum -- --nocapture
```

Expected: PASS.

**Step 5: Manual verification**

Run:
```bash
XLOG_MC_PROFILE=1 cargo test -p xlog-prob --release --features host-io --test mc test_sampling_method_in_result_metadata -- --nocapture
```

Expected: PASS and a single timing summary line in stderr/stdout.

**Step 6: Commit**

```bash
git add crates/xlog-prob/src/mc.rs crates/xlog-prob/tests/mc.rs
git commit -m "feat(mc): add hot-path timing breakdown for MC evaluation"
```

---

### Task 2: Remove Per-Sample D2H Row-Count Reads From Sample Materialization

**Files:**
- Modify: `crates/xlog-prob/src/mc.rs`
- Test: `crates/xlog-prob/tests/gpu_mc_device_counts.rs`
- Reference: `crates/xlog-cuda/src/provider.rs:5585-5784`, `crates/xlog-cuda/src/provider.rs:1998-2055`

**Step 1: Write the failing tests**

Extend `crates/xlog-prob/tests/gpu_mc_device_counts.rs` with a provider-facing invariant test that proves host `num_rows()` is already valid after compact and dedup:

```rust
#[test]
fn test_compact_and_dedup_preserve_host_row_count() -> Result<()> {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let program = McProgram::compile_source(
        r#"
        1.0::a().
        query(a()).
        "#,
    )?;

    let cfg = McEvalConfig {
        samples: 8,
        seed: 1,
        confidence: 0.95,
        max_nonmonotone_iterations: 8,
        sampling_method: None,
    };

    let device = program.evaluate_gpu_device(cfg)?;
    let counts = torch_like_readback(&provider, &device.query_counts)?;
    assert_eq!(counts.len(), 1);
    Ok(())
}
```

Also add a focused code-structure test in `crates/xlog-prob/tests/mc_gpu_native.rs`:

```rust
#[test]
fn mc_hot_path_no_device_row_count_helper() {
    let text = std::fs::read_to_string("crates/xlog-prob/src/mc.rs").unwrap();
    assert!(!text.contains("device_row_count_u32(provider, &filtered)"));
}
```

**Step 2: Run tests to verify failure**

Run:
```bash
cargo test -p xlog-prob --release --features host-io --test mc_gpu_native mc_hot_path_no_device_row_count_helper -- --nocapture
```

Expected: FAIL because the helper call still exists.

**Step 3: Write minimal implementation**

In `crates/xlog-prob/src/mc.rs`:
- remove `device_row_count_u32()` entirely from the hot path
- replace:
  - `filtered_rows = device_row_count_u32(provider, &filtered)?`
  - `rows = device_row_count_u32(provider, buffer)?`
- with host-cached `buffer.num_rows()` checks

Refactor `dedup_relation()` to:

```rust
fn dedup_relation(provider: &Arc<CudaKernelProvider>, buffer: &CudaBuffer) -> Result<CudaBuffer> {
    let rows = buffer.num_rows();
    if rows == 0 {
        return provider.create_empty_buffer(buffer.schema().clone());
    }
    if buffer.arity() == 0 {
        return build_zero_arity_buffer(provider, 1u32, buffer.schema());
    }
    let key_cols: Vec<usize> = (0..buffer.arity()).collect();
    provider.dedup(buffer, &key_cols)
}
```

And in `build_sample_buffers()` use `filtered.num_rows()` directly.

**Step 4: Run tests to verify they pass**

Run:
```bash
cargo test -p xlog-prob --release --features host-io --test gpu_mc_device_counts -- --nocapture
cargo test -p xlog-prob --release --features host-io --test mc_gpu_native mc_hot_path_no_device_row_count_helper -- --nocapture
```

Expected: PASS.

**Step 5: Regression**

Run:
```bash
cargo test -p xlog-prob --release --features host-io --test mc test_evidence_clamping_ad_head_3way -- --nocapture
```

Expected: PASS.

**Step 6: Commit**

```bash
git add crates/xlog-prob/src/mc.rs crates/xlog-prob/tests/gpu_mc_device_counts.rs crates/xlog-prob/tests/mc_gpu_native.rs
git commit -m "perf(mc): remove hot-loop row-count DTOH reads"
```

---

### Task 3: Introduce Explicit MC Count Strategy and Skip Clamped Evidence Side Data

**Files:**
- Modify: `crates/xlog-prob/src/mc.rs`
- Test: `crates/xlog-prob/tests/mc.rs`
- Test: `crates/xlog-prob/tests/gpu_mc_device_counts.rs`

**Step 1: Write the failing test**

Add a pure-strategy test to `crates/xlog-prob/tests/mc.rs`:

```rust
#[test]
fn test_count_strategy_clamped_skips_evidence_side() {
    use xlog_prob::mc::{McCountStrategy, McSamplingMethod};
    assert_eq!(
        McCountStrategy::from_method(McSamplingMethod::EvidenceClamping),
        McCountStrategy::QueriesOnly,
    );
    assert_eq!(
        McCountStrategy::from_method(McSamplingMethod::Rejection),
        McCountStrategy::QueriesAndEvidence,
    );
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cargo test -p xlog-prob --release --features host-io --test mc test_count_strategy_clamped_skips_evidence_side -- --nocapture
```

Expected: FAIL because `McCountStrategy` does not exist.

**Step 3: Write minimal implementation**

In `crates/xlog-prob/src/mc.rs` add:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McCountStrategy {
    QueriesAndEvidence,
    QueriesOnly,
}

impl McCountStrategy {
    pub fn from_method(method: McSamplingMethod) -> Self {
        match method {
            McSamplingMethod::Rejection => Self::QueriesAndEvidence,
            McSamplingMethod::EvidenceClamping => Self::QueriesOnly,
        }
    }
}
```

Then refactor `evaluate_gpu_device_with_provider()` so that in `QueriesOnly` mode it:
- does not allocate `d_evidence_ptrs`
- does not allocate `d_evidence_expected`
- does not allocate `d_evidence_ok`
- does not launch `mc_eval_query_evidence_truth`
- directly increments `d_evidence_count` to `cfg.samples` once, outside the sample loop
- only launches query-count accumulation inside the loop

Keep `McDeviceResult.evidence_count` unchanged as public API, but set it to `total_samples` in clamped mode.

**Step 4: Run tests to verify they pass**

Run:
```bash
cargo test -p xlog-prob --release --features host-io --test mc test_count_strategy_clamped_skips_evidence_side -- --nocapture
cargo test -p xlog-prob --release --features host-io --test gpu_mc_device_counts -- --nocapture
```

Expected: PASS.

**Step 5: Manual profile check**

Run:
```bash
XLOG_MC_PROFILE=1 cargo test -p xlog-prob --release --features host-io --test mc test_evidence_clamping_all_samples_count -- --nocapture
```

Expected: PASS and lower `count_us` than Task 1 baseline.

**Step 6: Commit**

```bash
git add crates/xlog-prob/src/mc.rs crates/xlog-prob/tests/mc.rs crates/xlog-prob/tests/gpu_mc_device_counts.rs
git commit -m "perf(mc): skip evidence-side count path in clamped mode"
```

---

### Task 4: Add Executor Reset Helper for MC Base/Dynamic Relation Splitting

**Files:**
- Modify: `crates/xlog-runtime/src/executor.rs`
- Modify: `crates/xlog-runtime/src/relation.rs` (only if helper needs store-level iteration/removal support beyond current API)
- Test: `crates/xlog-runtime/src/executor.rs`

**Step 1: Write the failing test**

Add a unit test to `crates/xlog-runtime/src/executor.rs` near existing reset tests:

```rust
#[test]
fn test_reset_for_mc_relations_preserves_static_and_clears_dynamic() {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return;
    };

    let mut executor = Executor::new(provider.clone());
    executor.register_relation(RelId(1), "base_rel");
    executor.register_relation(RelId(2), "dyn_rel");

    let schema = Schema::new(vec![("x".to_string(), ScalarType::U32)]);
    let base = provider.create_buffer_from_u32_slice(&[1u32], schema.clone()).unwrap();
    let dynb = provider.create_buffer_from_u32_slice(&[9u32], schema.clone()).unwrap();
    executor.put_relation("base_rel", base);
    executor.put_relation("dyn_rel", dynb);

    executor.reset_for_mc_relations(&["base_rel"], &[("dyn_rel", schema.clone())]).unwrap();

    assert_eq!(executor.store().get("base_rel").unwrap().num_rows(), 1);
    assert_eq!(executor.store().get("dyn_rel").unwrap().num_rows(), 0);
}
```

**Step 2: Run test to verify it fails**

Run:
```bash
cargo test -p xlog-runtime --release test_reset_for_mc_relations_preserves_static_and_clears_dynamic -- --nocapture
```

Expected: FAIL because `reset_for_mc_relations` does not exist.

**Step 3: Write minimal implementation**

In `crates/xlog-runtime/src/executor.rs`, add:

```rust
pub fn reset_for_mc_relations(
    &mut self,
    preserve: &[&str],
    clear_to_empty: &[(impl AsRef<str>, Schema)],
) -> Result<()> {
    let preserve_set: HashSet<&str> = preserve.iter().copied().collect();
    let existing_names: Vec<String> = self.store.names().map(|s| s.to_string()).collect();

    for name in existing_names {
        if !preserve_set.contains(name.as_str()) {
            let _ = self.store.remove(&name);
        }
    }

    for (name, schema) in clear_to_empty {
        if !self.store.contains(name.as_ref()) {
            let empty = self.provider.create_empty_buffer(schema.clone())?;
            self.store.put(name.as_ref(), empty);
        } else {
            let empty = self.provider.create_empty_buffer(schema.clone())?;
            self.store.put(name.as_ref(), empty);
        }
    }

    self.join_index_cache.clear();
    Ok(())
}
```

Keep `reset_for_mc()` unchanged for existing callers.

**Step 4: Run test to verify it passes**

Run:
```bash
cargo test -p xlog-runtime --release test_reset_for_mc_relations_preserves_static_and_clears_dynamic -- --nocapture
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/xlog-runtime/src/executor.rs crates/xlog-runtime/src/relation.rs
git commit -m "feat(runtime): add targeted MC relation reset helper"
```

---

### Task 5: Replace Full Base-Store Clone/Restore With Sample Reset Plan

**Files:**
- Modify: `crates/xlog-prob/src/mc.rs`
- Modify: `crates/xlog-runtime/src/executor.rs` (consume the new helper from Task 4)
- Test: `crates/xlog-prob/tests/mc.rs`
- Test: `crates/xlog-prob/tests/gpu_mc_vs_cpu.rs`

**Step 1: Write the failing test**

Add a targeted no-leakage regression to `crates/xlog-prob/tests/mc.rs`:

```rust
#[test]
fn test_mc_sample_reset_plan_preserves_base_and_clears_sampled_relations() {
    if !has_cuda_device() {
        eprintln!("Skipping: no CUDA device");
        return;
    }

    let src = r#"
    base().
    0.5::flip().
    seen_base() :- base().
    seen_flip() :- flip().
    query(seen_base()).
    query(seen_flip()).
    "#;

    let program = McProgram::compile_source(src).unwrap();
    let cfg = McEvalConfig {
        samples: 2000,
        seed: 9,
        confidence: 0.95,
        max_nonmonotone_iterations: 64,
        sampling_method: Some(McSamplingMethod::Rejection),
    };

    let result = program.evaluate_gpu(cfg).unwrap();
    let p_base = prob_of_atom(&result, "seen_base");
    let p_flip = prob_of_atom(&result, "seen_flip");
    assert!((p_base - 1.0).abs() < 1e-9, "p_base={}", p_base);
    assert!((p_flip - 0.5).abs() < 0.08, "p_flip={}", p_flip);
}
```

**Step 2: Run test to verify current behavior (baseline)**

Run:
```bash
cargo test -p xlog-prob --release --features host-io --test mc test_mc_sample_reset_plan_preserves_base_and_clears_sampled_relations -- --nocapture
```

Expected: PASS on current code. This is a regression guard for the refactor.

**Step 3: Implement the reset plan**

In `crates/xlog-prob/src/mc.rs`:
- introduce an `McSampleResetPlan` containing:
  - `preserve_relation_names: Vec<String>`
  - `clear_relation_schemas: Vec<(String, Schema)>`
- build it once from the compiled plan and probabilistic metadata:
  - preserve only deterministic base relations that are not overwritten by sampled/probabilistic relations
  - clear all query temp relations and all probabilistic relation names to empty buffers each sample
- remove `snapshot_store()` / `restore_store()` from the MC loop
- replace with:

```rust
executor.reset_for_mc_relations(&preserve_refs, &clear_to_empty)?;
```

Then re-apply only the preserved deterministic buffers once during plan initialization, not every sample.

If a relation is both deterministic base and probabilistic/rule-produced, classify it as dynamic and keep the old clone path for that relation only. Do not over-optimize ambiguous predicates in this task.

**Step 4: Run tests**

Run:
```bash
cargo test -p xlog-prob --release --features host-io --test mc test_mc_sample_reset_plan_preserves_base_and_clears_sampled_relations -- --nocapture
cargo test -p xlog-prob --release --features host-io --test gpu_mc_vs_cpu -- --nocapture
```

Expected: PASS.

**Step 5: Manual profile check**

Run:
```bash
XLOG_MC_PROFILE=1 cargo test -p xlog-prob --release --features host-io --test gpu_mc_vs_cpu -- --nocapture
```

Expected: PASS and significantly lower `sample_reset_us` than Task 1 baseline.

**Step 6: Commit**

```bash
git add crates/xlog-prob/src/mc.rs crates/xlog-runtime/src/executor.rs crates/xlog-prob/tests/mc.rs crates/xlog-prob/tests/gpu_mc_vs_cpu.rs
git commit -m "perf(mc): replace full store clone with targeted sample reset plan"
```

---

### Task 6: Cache Query Count Pointer Tables After Buffer Identity Stabilizes

**Files:**
- Modify: `crates/xlog-prob/src/mc.rs`
- Test: `crates/xlog-prob/tests/gpu_mc_device_counts.rs`

**Step 1: Write the failing test**

Add a regression test that exercises multiple samples in device mode and asserts identical public behavior before/after pointer caching:

```rust
#[test]
fn test_device_counts_reuse_pointer_tables_without_semantic_change() -> Result<()> {
    let Some(provider) = setup_provider() else {
        eprintln!("Skipping: no CUDA device");
        return Ok(());
    };

    let program = McProgram::compile_source(
        r#"
        0.5::a().
        evidence(a(), true).
        query(a()).
        "#,
    )?;

    let cfg = McEvalConfig {
        samples: 64,
        seed: 7,
        confidence: 0.95,
        max_nonmonotone_iterations: 8,
        sampling_method: None,
    };

    let result = program.evaluate_gpu_device_with_provider(cfg, provider)?;
    assert_eq!(result.total_samples, 64);
    assert_eq!(result.sampling_method, McSamplingMethod::EvidenceClamping);
    Ok(())
}
```

**Step 2: Run test to verify baseline**

Run:
```bash
cargo test -p xlog-prob --release --features host-io --test gpu_mc_device_counts test_device_counts_reuse_pointer_tables_without_semantic_change -- --nocapture
```

Expected: PASS on baseline; this is regression protection.

**Step 3: Implement pointer caching**

In `crates/xlog-prob/src/mc.rs`:
- after Task 5, relation buffers for preserved/empty reset buffers should have stable device pointers across samples
- allocate and upload query pointer tables once before the sample loop
- in rejection mode, either:
  - upload evidence pointer table once if evidence relation buffers are stable, or
  - rebuild only the evidence pointer slice if some evidence relations remain dynamic
- remove per-sample `query_ptrs` host vector construction if all corresponding buffers are stable

Keep this scoped: cache only what is actually stable after Task 5.

**Step 4: Run tests**

Run:
```bash
cargo test -p xlog-prob --release --features host-io --test gpu_mc_device_counts -- --nocapture
cargo test -p xlog-prob --release --features host-io --test mc test_rejection_unchanged -- --nocapture
```

Expected: PASS.

**Step 5: Manual profile check**

Run:
```bash
XLOG_MC_PROFILE=1 cargo test -p xlog-prob --release --features host-io --test gpu_mc_device_counts -- --nocapture
```

Expected: PASS and lower `count_us` than Task 5 baseline.

**Step 6: Commit**

```bash
git add crates/xlog-prob/src/mc.rs crates/xlog-prob/tests/gpu_mc_device_counts.rs crates/xlog-prob/tests/mc.rs
git commit -m "perf(mc): cache stable query count pointer tables"
```

---

### Task 7: Reduce Slow-Test Runtime Without Losing Coverage

**Files:**
- Modify: `crates/xlog-prob/tests/mc.rs`
- Modify: `crates/xlog-prob/tests/gpu_mc_vs_cpu.rs`
- Modify: `crates/xlog-prob/tests/mc_gpu_native.rs` (if needed for structure assertions)

**Step 1: Write the failing test budget assertions**

Add a lightweight meta-test to `crates/xlog-prob/tests/mc_gpu_native.rs` that protects against accidental high sample counts in behavior-only tests:

```rust
#[test]
fn mc_behavior_tests_do_not_use_large_sample_budgets() {
    let text = std::fs::read_to_string("crates/xlog-prob/tests/mc.rs").unwrap();
    assert!(!text.contains("samples: 80_000"));
}
```

Keep exactly one parity/accuracy test at `20_000+` samples in `gpu_mc_vs_cpu.rs`.

**Step 2: Run test to verify it fails**

Run:
```bash
cargo test -p xlog-prob --release --features host-io --test mc_gpu_native mc_behavior_tests_do_not_use_large_sample_budgets -- --nocapture
```

Expected: FAIL if large sample counts are still present in behavior tests.

**Step 3: Reduce sample counts where not statistically necessary**

In `crates/xlog-prob/tests/mc.rs`:
- reduce behavior-only tests to `2_000`–`10_000` samples depending on tolerance
- keep one or two high-N tests for exactness/parity only
- add comments explaining why each retained high-N test needs that budget

Do not weaken:
- `gpu_mc_vs_cpu` parity guard
- AD chain correctness
- explicit fallback/error path coverage

**Step 4: Run tests**

Run:
```bash
cargo test -p xlog-prob --release --features host-io --test mc -- --nocapture
cargo test -p xlog-prob --release --features host-io --test gpu_mc_vs_cpu -- --nocapture
```

Expected: PASS with lower wall clock than current baseline.

**Step 5: Commit**

```bash
git add crates/xlog-prob/tests/mc.rs crates/xlog-prob/tests/gpu_mc_vs_cpu.rs crates/xlog-prob/tests/mc_gpu_native.rs
git commit -m "test(mc): trim sample budgets in behavior-only MC tests"
```

---

### Task 8: Final Regression + Optimization Report

**Files:**
- Modify: `docs/ROADMAP.md` (optional note under v0.6 runtime work)
- Create: `docs/reports/2026-03-09-mc-runtime-optimization-report.md`

**Step 1: Run the full validation matrix**

Run:
```bash
cargo test -p xlog-prob --release --features host-io
cargo test -p xlog-cuda-tests --test certification_suite --release
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH \
/home/dev/projects/xlog/.venv/bin/python - <<'PY'
import torch, pyxlog
src = '''
0.7::rain().
0.2::sprinkler().
evidence(sprinkler(), true).
query(rain()).
'''
p = pyxlog.Program.compile(src, prob_engine='mc')
r = p.evaluate(samples=2000, seed=7)
print(r.sampling_method, r.evidence_samples, torch.from_dlpack(r.prob).cpu().tolist())
PY
```

Expected: all pass.

**Step 2: Capture before/after timing evidence**

Run with `XLOG_MC_PROFILE=1` on:
- `test_sampling_method_in_result_metadata`
- `gpu_mc_vs_cpu_on_small_program`

Record:
- total runtime
- sampler_us
- sample_reset_us
- sample_build_us
- eval_us
- count_us

**Step 3: Write report**

Create `docs/reports/2026-03-09-mc-runtime-optimization-report.md` with:
- baseline findings
- changes landed per task
- before/after timing table
- any residual hotspots left for future work

**Step 4: Commit**

```bash
git add docs/reports/2026-03-09-mc-runtime-optimization-report.md docs/ROADMAP.md
git commit -m "docs: record MC runtime optimization results"
```

---

## Expected Outcome

After Tasks 1-8:
- `EvidenceClamping` keeps current semantics and public API
- clamped mode no longer pays rejection-mode evidence-count overhead
- `build_sample_buffers()` no longer performs row-count D2H reads in the hot loop
- the MC loop no longer clones the full base store every sample
- per-sample pointer uploads are reduced or eliminated where buffer identities are stable
- the MC test suite keeps semantic coverage with materially lower runtime

## Execution Notes

- Implement in a dedicated worktree branched from the evidence-clamping integration point.
- Use `superpowers:test-driven-development` for each task.
- Run `superpowers:verification-before-completion` before claiming each task is done.
- Do not broaden scope into exact inference, approximate inference, or neural semantics.
